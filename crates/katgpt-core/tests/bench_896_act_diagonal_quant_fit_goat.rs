//! Issue 886 P0 + P1 (modelless half) — activation-diagonal weight-quant
//! fitting GOAT gate (Bench 896).
//!
//! End to end: `act_channel_moments` collects the per-input-channel
//! diagonal over a planted calibration set → katgpt-types `act_aware_fit`
//! re-fits the `Q2_0_g128` group scales → the output reconstruction error
//! `E‖Wx − Ŵx‖² / E‖Wx‖²` is measured on a HELD-OUT activation sample.
//!
//! G1  quality (synthetic, RECORDED EITHER SIGN — the honest prior expects
//!     small-or-negative at ternary): weighted fits vs the activation-blind
//!     mean-abs baseline under four planted activation distributions, plus
//!     a bench-local INT4 group-128 reference (NOT a shipped tier — the
//!     crate has no 4-bit weight container) where the prior expects the
//!     real payoff. Hard asserts are instrument health only: finite
//!     metrics, the blind search never worse than the baseline on its own
//!     (unweighted) objective.
//! G2  cost: observe ns/element (`best_of_us`, bar ≤ 2.0 ns); refactored
//!     baseline vs a verbatim legacy transcription (`ab_median_ratio`, bar
//!     ≤ 1.05 — the refactor must not regress the shipped path); weighted
//!     mean-abs vs baseline (bar ≤ 1.50); search vs baseline (bar ≤ 46 =
//!     2 × the 23 carry passes it runs). `black_box` on results AND
//!     arguments, `--release`.
//! G3  a UNIFORM diagonal from the committed table (`ActChannelDiagonal::
//!     uniform`, through `to_bytes`/`from_bytes`) ⇒ payload bytes
//!     bit-identical to `quantize_from_f32`.
//! G4  0 allocs in `observe`/`observe_batch`; the act-aware fit allocates
//!     exactly what the baseline allocates (the output container).
//!
//! Run (harness = false, exit non-zero on any gate failure):
//!   cargo test -p katgpt-core --features act_channel_moments,act_aware_fit \
//!     --release --test bench_896_act_diagonal_quant_fit_goat -- --nocapture

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::{ab_median_ratio, best_of_us};
use half::f16;
use katgpt_core::act_channel_moments::{ActChannelDiagonal, ActChannelMoments};
use katgpt_types::{ActAwareScaleFit, GROUP_SIZE, TernaryGroupWeights};

// ── G4: counting allocator ────────────────────────────────────────────────

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

// ── deterministic fixtures ────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn u01(&mut self) -> f64 {
        // xorshift64* — fixed across runs/platforms by construction.
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f32 {
        let (a, b) = (self.u01(), self.u01());
        ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
    }
    fn laplace(&mut self) -> f32 {
        let u = self.u01() - 0.5;
        (-u.signum() * (1.0 - 2.0 * u.abs()).ln()) as f32 * std::f32::consts::FRAC_1_SQRT_2
    }
}

const R: usize = 256;
const C: usize = 1024;
const N_CAL: usize = 512;
const N_EVAL: usize = 256;

/// A planted activation distribution: `x_j = σ_j · z + μ_j`.
struct Dist {
    name: &'static str,
    sigma: Vec<f32>,
    mu: Vec<f32>,
}

fn dists() -> Vec<Dist> {
    let mut rng = Rng(0x886_0001);
    let heavy: Vec<bool> = (0..C).map(|j| j % 97 == 13).collect(); // ~1% of channels
    let sig_heavy: Vec<f32> = heavy.iter().map(|&h| if h { 20.0 } else { 1.0 }).collect();
    vec![
        Dist { name: "D0 uniform (control)", sigma: vec![1.0; C], mu: vec![0.0; C] },
        Dist { name: "D1 1% heavy x20, zero-mean", sigma: sig_heavy.clone(), mu: vec![0.0; C] },
        Dist {
            name: "D2 1% heavy x20, mean 0.5σ",
            mu: sig_heavy.iter().map(|s| 0.5 * s).collect(),
            sigma: sig_heavy,
        },
        Dist {
            name: "D3 log-normal spread σ=e^N(0,1)",
            sigma: (0..C).map(|_| rng.gauss().exp()).collect(),
            mu: vec![0.0; C],
        },
    ]
}

fn sample(d: &Dist, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let mut xs = vec![0.0f32; n * C];
    for row in xs.as_chunks_mut::<C>().0 {
        for (j, v) in row.iter_mut().enumerate() {
            *v = d.sigma[j] * rng.gauss() + d.mu[j];
        }
    }
    xs
}

fn weights(laplace: bool, seed: u64) -> Vec<f32> {
    weights_n(laplace, seed, R * C)
}

fn weights_n(laplace: bool, seed: u64, n: usize) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| 0.02 * if laplace { rng.laplace() } else { rng.gauss() })
        .collect()
}

fn dequant(t: &TernaryGroupWeights) -> Vec<f32> {
    let mut out = vec![0.0f32; t.rows * t.cols];
    for r in 0..t.rows {
        for c in 0..t.cols {
            out[r * t.cols + c] = t.scale_at(r, c / GROUP_SIZE) * f32::from(t.get(r, c));
        }
    }
    out
}

/// `Σ_x ‖(W − Ŵ)x‖² / Σ_x ‖Wx‖²` over the held-out sample (f64 sums).
fn rel_output_mse(w: &[f32], wq: &[f32], xs: &[f32]) -> f64 {
    let dw: Vec<f32> = w.iter().zip(wq).map(|(a, b)| a - b).collect();
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for x in xs.as_chunks::<C>().0 {
        for r in 0..R {
            let row = r * C..(r + 1) * C;
            let e: f32 = dw[row.clone()].iter().zip(x).map(|(a, b)| a * b).sum();
            let y: f32 = w[row].iter().zip(x).map(|(a, b)| a * b).sum();
            num += f64::from(e) * f64::from(e);
            den += f64::from(y) * f64::from(y);
        }
    }
    num / den
}

fn weight_mse(w: &[f32], wq: &[f32]) -> f64 {
    w.iter().zip(wq).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>() / w.len() as f64
}

fn payload_bytes(t: &TernaryGroupWeights) -> Vec<u8> {
    let mut b = Vec::new();
    for w in t.pos_bits.iter().chain(&t.neg_bits) {
        b.extend_from_slice(&w.to_le_bytes());
    }
    for s in &t.group_scale {
        b.extend_from_slice(&s.to_bits().to_le_bytes());
    }
    b
}

/// Verbatim transcription of the pre-refactor quantizer (katgpt-rs
/// `4cc5bd941`) — the G2 no-regression comparator.
fn legacy_quantize(w: &[f32], rows: usize, cols: usize) -> TernaryGroupWeights {
    let mut out = TernaryGroupWeights::new(rows, cols);
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let row_base = r * out.blocks64;
        let group_base = r * out.groups_per_row;
        for g in 0..out.groups_per_row {
            let g_start = g * GROUP_SIZE;
            let g_end = (g_start + GROUP_SIZE).min(cols);
            let group = &row[g_start..g_end];
            let abs_sum: f32 = group.iter().map(|v| v.abs()).sum();
            let scale = if abs_sum > 0.0 { abs_sum / group.len() as f32 } else { 1.0 };
            out.group_scale[group_base + g] = f16::from_f32(scale);
            let scale = out.group_scale[group_base + g].to_f32();
            let threshold = 0.5 * scale;
            let mut carry = 0.0f32;
            for (i, &val) in group.iter().enumerate() {
                let adjusted = val + carry;
                let q = match adjusted {
                    a if a > threshold => 1i8,
                    a if a < -threshold => -1i8,
                    _ => 0i8,
                };
                let col = g_start + i;
                let idx = row_base + (col >> 6);
                let mask = 1u64 << (col & 63);
                out.pos_bits[idx] |= ((q == 1) as u64) * mask;
                out.neg_bits[idx] |= ((q == -1) as u64) * mask;
                carry = adjusted - (q as f32 * scale);
            }
        }
    }
    out
}

// ── bench-local INT4 group-128 reference (NOT a shipped tier) ─────────────

/// Symmetric INT4 `q ∈ [-8, 7]`, f16 group scale. `diag = None` ⇒ RTN
/// (`s = absmax/7`); `Some(h)` ⇒ grid `s = s_rtn × m`, m ∈ 0.70..=1.30
/// (25 pts, 1.0 first) + LS refit, minimising `Σ u (w − s q)²` with
/// `u = h / max(h)` (a uniform `h` = the activation-blind search).
fn int4_dequant(w: &[f32], diag: Option<&[f32]>) -> Vec<f32> {
    let mut out = vec![0.0f32; w.len()];
    let f16x = |s: f32| f16::from_f32(s).to_f32();
    let code = |v: f32, s: f32| (v / s).round().clamp(-8.0, 7.0);
    for r in 0..R {
        for g in 0..C / GROUP_SIZE {
            let lo = r * C + g * GROUP_SIZE;
            let grp = &w[lo..lo + GROUP_SIZE];
            let amax = grp.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let s_rtn = if amax > 0.0 { amax / 7.0 } else { 1.0 };
            let s = match diag {
                None => f16x(s_rtn),
                Some(h) => {
                    let hg = &h[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
                    let hmax = hg.iter().fold(0.0f32, |m, &v| m.max(v));
                    let u: Vec<f32> = hg.iter().map(|&v| if hmax > 0.0 { v / hmax } else { 1.0 }).collect();
                    let eval = |s: f32| -> (f32, f32, f32) {
                        let (mut e, mut n, mut d) = (0.0f32, 0.0f32, 0.0f32);
                        for (&v, &uw) in grp.iter().zip(&u) {
                            let q = code(v, s);
                            e += uw * (v - s * q).powi(2);
                            n += uw * v * q;
                            d += uw * q * q;
                        }
                        (e, n, d)
                    };
                    let mut best = f16x(s_rtn);
                    let (mut be, mut bn, mut bd) = eval(best);
                    for k in 0..25 {
                        let s = f16x(s_rtn * (0.70 + 0.025 * k as f32));
                        let (e, n, d) = eval(s);
                        if e < be {
                            (best, be, bn, bd) = (s, e, n, d);
                        }
                    }
                    if bd > 0.0 {
                        let s = f16x(bn / bd);
                        if s > 0.0 && eval(s).0 < be {
                            best = s;
                        }
                    }
                    best
                }
            };
            for (o, &v) in out[lo..lo + GROUP_SIZE].iter_mut().zip(grp) {
                *o = s * code(v, s);
            }
        }
    }
    out
}

fn check(failed: &mut Vec<String>, ok: bool, msg: String) {
    println!("{} {msg}", if ok { "  ✓" } else { "  ✗" });
    if !ok {
        failed.push(msg);
    }
}

fn main() {
    let mut failed: Vec<String> = Vec::new();

    // ── G3: uniform committed diagonal ⇒ bit-identical payload ───────────
    println!("── G3 uniform diagonal bit-identity");
    let w_g = weights(false, 0x886_0003);
    let uni = ActChannelDiagonal::uniform(&[C], 0.8, 0.64, N_CAL as u64);
    let uni = ActChannelDiagonal::from_bytes(&uni.to_bytes()).expect("uniform table roundtrip");
    let base_bytes = payload_bytes(&TernaryGroupWeights::quantize_from_f32(&w_g, R, C));
    for (label, h) in [("mean_sq", uni.mean_sq(0)), ("mean_abs", uni.mean_abs(0))] {
        let t = TernaryGroupWeights::quantize_from_f32_act_aware(
            &w_g,
            R,
            C,
            h,
            ActAwareScaleFit::WeightedMeanAbs,
        );
        check(&mut failed, payload_bytes(&t) == base_bytes, format!("G3 uniform {label}: payload bytes identical ({} B)", base_bytes.len()));
    }
    check(&mut failed, 
        payload_bytes(&legacy_quantize(&w_g, R, C)) == base_bytes,
        "G3 refactored baseline == pre-refactor transcription".into(),
    );

    // ── G1: output reconstruction error under planted distributions ──────
    println!("── G1 held-out rel output MSE E‖(W−Ŵ)x‖²/E‖Wx‖² (Δ% vs baseline; − = better)");
    let mut g1_rows: Vec<String> = Vec::new();
    for (wi, laplace) in [false, true].into_iter().enumerate() {
        let w = weights(laplace, 0x886_0100 + wi as u64);
        let wname = if laplace { "Laplace W" } else { "Gaussian W" };
        let base_q = dequant(&TernaryGroupWeights::quantize_from_f32(&w, R, C));
        let base_wmse = weight_mse(&w, &base_q);
        for (di, d) in dists().iter().enumerate() {
            // P0 collector over the calibration sample.
            let mut mom = ActChannelMoments::new(&[C]);
            mom.observe_batch(0, &sample(d, N_CAL, 0x886_1000 + di as u64));
            let diag = mom.freeze();
            let eval_x = sample(d, N_EVAL, 0x886_2000 + di as u64);
            let base = rel_output_mse(&w, &base_q, &eval_x);
            let uniform = vec![1.0f32; C];
            let arms: [(&str, &[f32], ActAwareScaleFit); 5] = [
                ("WMA[E x²]", diag.mean_sq(0), ActAwareScaleFit::WeightedMeanAbs),
                ("WMA[mean|x|]", diag.mean_abs(0), ActAwareScaleFit::WeightedMeanAbs),
                ("Search[E x²]", diag.mean_sq(0), ActAwareScaleFit::WeightedSearch),
                ("Search[mean|x|]", diag.mean_abs(0), ActAwareScaleFit::WeightedSearch),
                ("Search[blind]", &uniform, ActAwareScaleFit::WeightedSearch),
            ];
            let mut line = format!("  ternary {wname} {:<32} base {base:.5}", d.name);
            for (name, h, fit) in arms {
                let q = dequant(&TernaryGroupWeights::quantize_from_f32_act_aware(&w, R, C, h, fit));
                let m = rel_output_mse(&w, &q, &eval_x);
                if !m.is_finite() || m <= 0.0 {
                    failed.push(format!("G1 non-finite metric {name} {}", d.name));
                }
                if name == "Search[blind]" {
                    let wm = weight_mse(&w, &q);
                    if wm > base_wmse * (1.0 + 1e-6) {
                        failed.push(format!("G1 blind search weight-MSE {wm} > baseline {base_wmse}"));
                    }
                }
                line += &format!(" | {name} {:+.2}%", (m / base - 1.0) * 100.0);
            }
            println!("{line}");
            g1_rows.push(line);

            // INT4 bench-local reference tier.
            let rtn = rel_output_mse(&w, &int4_dequant(&w, None), &eval_x);
            let blind = rel_output_mse(&w, &int4_dequant(&w, Some(&uniform)), &eval_x);
            let wsq = rel_output_mse(&w, &int4_dequant(&w, Some(diag.mean_sq(0))), &eval_x);
            let wab = rel_output_mse(&w, &int4_dequant(&w, Some(diag.mean_abs(0))), &eval_x);
            let line = format!(
                "  int4    {wname} {:<32} RTN  {rtn:.6} | Search[blind] {:+.2}% | Search[E x²] {:+.2}% | Search[mean|x|] {:+.2}%",
                d.name,
                (blind / rtn - 1.0) * 100.0,
                (wsq / rtn - 1.0) * 100.0,
                (wab / rtn - 1.0) * 100.0,
            );
            println!("{line}");
            if !(rtn.is_finite() && wsq.is_finite() && wab.is_finite() && blind.is_finite()) {
                failed.push(format!("G1 int4 non-finite {}", d.name));
            }
        }
    }

    // ── G2: cost ──────────────────────────────────────────────────────────
    println!("── G2 cost (release; black_box on results AND arguments)");
    const OW: usize = 4096;
    const OB: usize = 64;
    let obs: Vec<f32> = {
        let mut rng = Rng(0x886_0200);
        (0..OW * OB).map(|_| rng.gauss()).collect()
    };
    let mut mom = ActChannelMoments::new(&[OW]);
    let obs_us = best_of_us(5, 50, || {
        let t = Instant::now();
        mom.observe_batch(0, black_box(&obs));
        black_box(&mut mom);
        t.elapsed()
    });
    let ns_per_elem = obs_us * 1e3 / (OW * OB) as f64;
    check(&mut failed, ns_per_elem <= 2.0, format!("G2a observe: {ns_per_elem:.3} ns/element @ width {OW} (bar 2.0)"));

    let (fr, fc) = (256usize, 4096usize);
    let wf = weights_n(false, 0x886_0300, fr * fc);
    let diag_f: Vec<f32> = {
        let mut rng = Rng(0x886_0301);
        (0..fc).map(|_| rng.gauss().exp()).collect()
    };
    let per_w = |us: f64| us * 1e3 / (fr * fc) as f64;
    let base_us = best_of_us(2, 10, || {
        let t = Instant::now();
        let q = TernaryGroupWeights::quantize_from_f32(black_box(&wf), fr, fc);
        black_box(&q);
        t.elapsed()
    });
    let wma_us = best_of_us(2, 10, || {
        let t = Instant::now();
        let q = TernaryGroupWeights::quantize_from_f32_act_aware(
            black_box(&wf),
            fr,
            fc,
            black_box(&diag_f),
            ActAwareScaleFit::WeightedMeanAbs,
        );
        black_box(&q);
        t.elapsed()
    });
    let srch_us = best_of_us(1, 5, || {
        let t = Instant::now();
        let q = TernaryGroupWeights::quantize_from_f32_act_aware(
            black_box(&wf),
            fr,
            fc,
            black_box(&diag_f),
            ActAwareScaleFit::WeightedSearch,
        );
        black_box(&q);
        t.elapsed()
    });
    println!(
        "  fit ns/weight (best-of, {fr}x{fc}): baseline {:.3} | WMA {:.3} | Search {:.3}",
        per_w(base_us),
        per_w(wma_us),
        per_w(srch_us)
    );
    let (mut sink_a, mut sink_b) = (0u64, 0u64);
    let reg = ab_median_ratio(
        15,
        2,
        2,
        |_| {
            let q = legacy_quantize(black_box(&wf), fr, fc);
            sink_a = sink_a.wrapping_add(black_box(&q).pos_bits[0]);
        },
        |_| {
            let q = TernaryGroupWeights::quantize_from_f32(black_box(&wf), fr, fc);
            sink_b = sink_b.wrapping_add(black_box(&q).pos_bits[0]);
        },
    );
    reg.report("G2b refactored/legacy baseline");
    check(&mut failed, reg.median <= 1.05, format!("G2b refactored/legacy baseline median {:.4} (bar 1.05)", reg.median));
    let wr = ab_median_ratio(
        15,
        2,
        2,
        |_| {
            let q = TernaryGroupWeights::quantize_from_f32(black_box(&wf), fr, fc);
            sink_a = sink_a.wrapping_add(black_box(&q).pos_bits[0]);
        },
        |_| {
            let q = TernaryGroupWeights::quantize_from_f32_act_aware(
                black_box(&wf),
                fr,
                fc,
                black_box(&diag_f),
                ActAwareScaleFit::WeightedMeanAbs,
            );
            sink_b = sink_b.wrapping_add(black_box(&q).pos_bits[0]);
        },
    );
    wr.report("G2c WMA/baseline");
    check(&mut failed, wr.median <= 1.50, format!("G2c WMA/baseline median {:.4} (bar 1.50)", wr.median));
    let sr = srch_us / base_us;
    check(&mut failed, sr <= 46.0, format!("G2d Search/baseline best-of ratio {sr:.2} (bar 46 = 2 × 23 carry passes)"));
    black_box((sink_a, sink_b));

    // ── G4: allocation discipline ─────────────────────────────────────────
    println!("── G4 allocations");
    let mut mom = ActChannelMoments::new(&[OW, 128]);
    let small: Vec<f32> = obs[..128].to_vec();
    let before = allocs();
    for i in 0..OB {
        mom.observe(0, black_box(&obs[i * OW..(i + 1) * OW]));
    }
    mom.observe_batch(0, black_box(&obs));
    mom.observe(1, black_box(&small));
    let obs_allocs = allocs() - before;
    check(&mut failed, obs_allocs == 0, format!("G4a observe/observe_batch allocs: {obs_allocs}"));
    let before = allocs();
    black_box(TernaryGroupWeights::quantize_from_f32(&wf, fr, fc));
    let base_allocs = allocs() - before;
    for fit in [ActAwareScaleFit::WeightedMeanAbs, ActAwareScaleFit::WeightedSearch] {
        let before = allocs();
        black_box(TernaryGroupWeights::quantize_from_f32_act_aware(&wf, fr, fc, &diag_f, fit));
        let a = allocs() - before;
        check(&mut failed, a == base_allocs, format!("G4b {fit:?} fit allocs {a} == baseline {base_allocs}"));
    }

    if failed.is_empty() {
        println!("bench_896: ALL GATES PASSED (G1 recorded above — either sign is a valid measurement)");
    } else {
        println!("bench_896: {} GATE(S) FAILED:", failed.len());
        for f in &failed {
            println!("  - {f}");
        }
        std::process::exit(1);
    }
}
