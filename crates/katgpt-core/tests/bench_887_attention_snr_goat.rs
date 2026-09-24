//! Issue 882 P1 — streaming attention-SNR accumulators GOAT gate.
//!
//! G1a exactness: the fused tiled kernel's per-row entropy + participation
//!     ratio vs an f64 reference over the materialized softmax, including a
//!     fixture whose row max rises on every K tile (the rescale path).
//! G1b retrieval proxy: a planted gold key swept in strength — normalized
//!     entropy must track gold mass (Spearman ≤ −0.95). Trap 2 pinned as a
//!     MEASURED NEGATIVE: a planted distractor gives the same low entropy at
//!     ~zero gold mass, so entropy alone cannot certify relevance (the m_Y
//!     falsifier is P4's).
//! G1c closed loop: `fit_tau` → SSMax `Fixed` socket → kernel-measured
//!     normalized entropy lands on the target.
//! G2  latency: stats arm vs plain forward ≤ 1.02 (`ab_median_ratio`, the
//!     paired interleave protocol; black_box on inputs AND every stats read).
//! G3  no-regression: output bit-identical to `tiled_attention_forward` at
//!     N ≥ 128; ≤ 1e-5 below (plain takes its materialized fallback there).
//! G4  alloc-free: 0 allocations across repeated stats-arm calls.
//!
//! Run:
//!   cargo test -p katgpt-core --release --features attention_snr \
//!     --test bench_887_attention_snr_goat -- --nocapture
//!
//! Box state is printed at the top of the run and recorded with every figure
//! in .benchmarks/887.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::ab_median_ratio;
use katgpt_core::attention::{
    tiled_attention_forward, tiled_attention_forward_snr, tiled_snr_scratch_len,
};
use katgpt_core::attention_snr::{SnrRowStats, TauFitConfig, fit_tau};

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

// ── fixtures ──────────────────────────────────────────────────────────────

/// Deterministic xorshift fill in [-0.5, 0.5) — no RNG dep, fixed across
/// runs and platforms by construction.
fn fill(seed: u32, len: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) & 0xffff) as f32 / 65535.0 - 0.5
        })
        .collect()
}

struct Qkv {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    n: usize,
    d: usize,
}

fn random_qkv(n: usize, d: usize, seed: u32, amp: f32) -> Qkv {
    let mut q = fill(seed, n * d);
    let mut k = fill(seed.wrapping_mul(7919), n * d);
    for x in q.iter_mut().chain(k.iter_mut()) {
        *x *= amp;
    }
    Qkv {
        q,
        k,
        v: fill(seed.wrapping_mul(104_729), n * d),
        n,
        d,
    }
}

/// Keys aligned with every query at strength rising in key index: each row's
/// max moves on every K tile, so the closed-form rescale runs every tile.
fn rising_qkv(n: usize, d: usize) -> Qkv {
    let mut base = random_qkv(n, d, 5, 1.0);
    let dir = fill(99, d);
    for i in 0..n {
        base.q[i * d..(i + 1) * d].copy_from_slice(&dir);
    }
    for j in 0..n {
        let a = 6.0 * j as f32 / n as f32;
        for (kc, &dc) in base.k[j * d..(j + 1) * d].iter_mut().zip(&dir) {
            *kc = dc * a + 0.05 * *kc;
        }
    }
    base
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

/// f64 reference: (entropy, PR, gold mass at `gold`) of query row `i`.
fn reference_row(f: &Qkv, i: usize, scale: f32, gold: usize) -> (f64, f64, f64) {
    let d = f.d;
    let q = &f.q[i * d..(i + 1) * d];
    let logits: Vec<f64> = (0..f.n)
        .map(|j| scale as f64 * dot(q, &f.k[j * d..(j + 1) * d]))
        .collect();
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = logits.iter().map(|&x| (x - m).exp()).sum();
    let (mut h, mut s2) = (0.0, 0.0);
    for &x in &logits {
        let p = (x - m).exp() / z;
        if p > 0.0 {
            h -= p * p.ln();
        }
        s2 += p * p;
    }
    (h, 1.0 / s2, (logits[gold] - m).exp() / z)
}

fn run_snr(f: &Qkv, scale: f32) -> (Vec<f32>, Vec<SnrRowStats>) {
    let mut out = vec![0.0; f.n * f.d];
    let mut o_tile = vec![0.0; tiled_snr_scratch_len(f.d)];
    let mut stats = vec![SnrRowStats::default(); f.n];
    tiled_attention_forward_snr(
        &f.q,
        &f.k,
        &f.v,
        &mut out,
        f.n,
        f.d,
        scale,
        &mut o_tile,
        &mut stats,
    );
    (out, stats)
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    fn ranks(x: &[f64]) -> Vec<f64> {
        let mut idx: Vec<usize> = (0..x.len()).collect();
        idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
        let mut r = vec![0.0; x.len()];
        for (rank, &i) in idx.iter().enumerate() {
            r[i] = rank as f64;
        }
        r
    }
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = ra.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = rb.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

fn box_state() {
    let run = |cmd: &str, args: &[&str]| {
        std::process::Command::new(cmd)
            .args(args)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
            .unwrap_or_else(|| "n/a".into())
    };
    println!("box: loadavg {}", run("sysctl", &["-n", "vm.loadavg"]));
    println!(
        "box: power {}",
        run("pmset", &["-g", "batt"]).replace('\n', " | ")
    );
}

// ── gates ─────────────────────────────────────────────────────────────────

fn main() {
    box_state();
    let mut fails = 0u32;
    let mut gate = |name: &str, ok: bool, detail: String| {
        println!("{} {name}: {detail}", if ok { "PASS" } else { "FAIL" });
        fails += (!ok) as u32;
    };

    // G1a — exactness vs f64 reference.
    for (label, f) in [
        ("random N=130 D=64", random_qkv(130, 64, 1, 4.0)),
        ("random N=1024 D=64", random_qkv(1024, 64, 2, 4.0)),
        ("rising-max N=1024 D=64", rising_qkv(1024, 64)),
    ] {
        let scale = 1.0 / (f.d as f32).sqrt();
        let (_, stats) = run_snr(&f, scale);
        let (mut eh, mut epr) = (0.0f64, 0.0f64);
        for i in (0..f.n).step_by(7) {
            let (h, pr, _) = reference_row(&f, i, scale, 0);
            eh = eh.max((stats[i].entropy as f64 - h).abs());
            epr = epr.max((stats[i].participation_ratio as f64 - pr).abs() / pr);
        }
        gate(
            &format!("G1a {label}"),
            eh <= 1e-3 && epr <= 1e-3,
            format!("max |ΔH| = {eh:.2e} nats, max rel ΔPR = {epr:.2e}"),
        );
    }

    // G1b — retrieval proxy + trap 2.
    {
        let (n, d, gold, decoy) = (1024usize, 64usize, 333usize, 777usize);
        let scale = 1.0 / (d as f32).sqrt();
        let base = random_qkv(n, d, 11, 1.0);
        let qdir: Vec<f32> = base.q[..d].to_vec();
        let qn = dot(&qdir, &qdir).sqrt() as f32;
        let (mut hn, mut gm) = (Vec::new(), Vec::new());
        for step in 0..24 {
            let beta = step as f32 * 2.0;
            let mut f = random_qkv(n, d, 11, 1.0);
            for (kc, &qc) in f.k[gold * d..(gold + 1) * d].iter_mut().zip(&qdir) {
                *kc = qc / qn * beta;
            }
            let (_, stats) = run_snr(&f, scale);
            let (_, _, g) = reference_row(&f, 0, scale, gold);
            hn.push(stats[0].normalized_entropy() as f64);
            gm.push(g);
        }
        let rho = spearman(&hn, &gm);
        gate(
            "G1b retrieval proxy (entropy tracks gold mass)",
            rho <= -0.95,
            format!(
                "Spearman(Hn, gold mass) = {rho:.3} over 24 strengths; Hn {:.3}→{:.3}, gold {:.4}→{:.4}",
                hn[0], hn[23], gm[0], gm[23]
            ),
        );

        // Trap 2: the same concentration on a DISTRACTOR.
        let mut f = random_qkv(n, d, 11, 1.0);
        for (kc, &qc) in f.k[decoy * d..(decoy + 1) * d].iter_mut().zip(&qdir) {
            *kc = qc / qn * 46.0;
        }
        let (_, stats) = run_snr(&f, scale);
        let (_, _, g) = reference_row(&f, 0, scale, gold);
        let h = stats[0].normalized_entropy();
        gate(
            "G1b trap 2 pinned negative (confident-but-wrong is invisible to entropy)",
            h < 0.5 && g < 0.01,
            format!("decoy-concentrated row: Hn = {h:.3} (as sharp as a hit), gold mass = {g:.2e}"),
        );
    }

    // G1c — closed loop: fit τ → SSMax socket → kernel-measured entropy.
    {
        let f = random_qkv(1024, 64, 21, 4.0);
        let scale = 1.0 / (f.d as f32).sqrt();
        let row: Vec<f32> = (0..f.n)
            .map(|j| scale * dot(&f.q[..f.d], &f.k[j * f.d..(j + 1) * f.d]) as f32)
            .collect();
        let mut worst = 0.0f32;
        for &target in &[0.9f32, 0.7, 0.5, 0.3] {
            let fit = fit_tau(&row, target, &TauFitConfig::default());
            let mult = fit.ssmax_mode().multiplier((f.n as f32).ln());
            let (_, stats) = run_snr(&f, scale * mult);
            worst = worst.max((stats[0].normalized_entropy() - target).abs());
            assert!(!fit.saturated, "target {target} saturated");
        }
        gate(
            "G1c fit_tau → SSMax Fixed socket → kernel",
            worst <= 1e-3,
            format!("max |Hn_kernel − target| = {worst:.2e} over 4 targets"),
        );
    }

    // G3 — no-regression.
    for n in [128usize, 1024] {
        let f = random_qkv(n, 64, 31, 4.0);
        let scale = 0.125;
        let mut plain = vec![0.0; n * 64];
        tiled_attention_forward(&f.q, &f.k, &f.v, &mut plain, n, 64, scale);
        let (snr, _) = run_snr(&f, scale);
        let same = plain
            .iter()
            .zip(&snr)
            .all(|(a, b)| a.to_bits() == b.to_bits());
        gate(
            &format!("G3 bit-identical N={n}"),
            same,
            format!("{} values", n * 64),
        );
    }
    {
        let f = random_qkv(64, 64, 32, 4.0);
        let mut plain = vec![0.0; 64 * 64];
        tiled_attention_forward(&f.q, &f.k, &f.v, &mut plain, 64, 64, 0.125);
        let (snr, _) = run_snr(&f, 0.125);
        let e = plain
            .iter()
            .zip(&snr)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        gate(
            "G3 small-N (fallback vs tiled) N=64",
            e <= 1e-5,
            format!("max |Δ| = {e:.2e}"),
        );
    }

    // G4 — alloc-free stats arm.
    {
        let f = random_qkv(512, 64, 41, 4.0);
        let mut out = vec![0.0; f.n * f.d];
        let mut o_tile = vec![0.0; tiled_snr_scratch_len(f.d)];
        let mut stats = vec![SnrRowStats::default(); f.n];
        let before = allocs();
        for _ in 0..10 {
            tiled_attention_forward_snr(
                &f.q,
                &f.k,
                &f.v,
                &mut out,
                f.n,
                f.d,
                0.125,
                &mut o_tile,
                &mut stats,
            );
        }
        let n_alloc = allocs() - before;
        black_box(&stats);
        gate(
            "G4 alloc-free",
            n_alloc == 0,
            format!("{n_alloc} allocations over 10 calls"),
        );
    }

    // G2 — latency, paired interleave.
    for (n, d) in [(1024usize, 64usize), (512, 128)] {
        let f = random_qkv(n, d, 51, 4.0);
        let scale = 1.0 / (d as f32).sqrt();
        let mut out_a = vec![0.0; n * d];
        let mut out_b = vec![0.0; n * d];
        let mut o_tile = vec![0.0; tiled_snr_scratch_len(d)];
        let mut stats = vec![SnrRowStats::default(); n];
        let (mut sink_a, mut sink_b) = (0.0f32, 0.0f32);
        let ratio = ab_median_ratio(
            15,
            3,
            2,
            |i| {
                // Vary the scale per iteration so no arm is hoistable.
                let s = scale * (1.0 + (i % 3) as f32 * 1e-3);
                tiled_attention_forward(
                    black_box(&f.q),
                    black_box(&f.k),
                    black_box(&f.v),
                    &mut out_a,
                    n,
                    d,
                    black_box(s),
                );
                sink_a += black_box(out_a[i % out_a.len()]);
            },
            |i| {
                let s = scale * (1.0 + (i % 3) as f32 * 1e-3);
                tiled_attention_forward_snr(
                    black_box(&f.q),
                    black_box(&f.k),
                    black_box(&f.v),
                    &mut out_b,
                    n,
                    d,
                    black_box(s),
                    &mut o_tile,
                    &mut stats,
                );
                // Read the stats: an unread accumulator is what fat LTO deletes.
                let r = &stats[i % n];
                sink_b += black_box(out_b[i % out_b.len()])
                    + black_box(r.entropy + r.participation_ratio);
            },
        );
        ratio.report(&format!("G2 N={n} D={d} snr/plain"));
        gate(
            &format!("G2 N={n} D={d}"),
            ratio.median <= 1.02,
            format!(
                "median {:.4} (overhead {:+.2}%, range {:.4}–{:.4}, plain {:.1} µs/call)",
                ratio.median,
                ratio.overhead_pct(),
                ratio.min(),
                ratio.max(),
                ratio.a_ns_per_iter() / 1e3
            ),
        );
        black_box(sink_a + sink_b);
    }

    if fails > 0 {
        println!("\n{fails} gate(s) FAILED");
        std::process::exit(1);
    }
    println!("\nALL GATES PASS");
}
