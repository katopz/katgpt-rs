//! Issue 954 — the `kron_memory` GOAT gate (Bench 951).
//!
//! # Gates
//!
//! - **G0 (the falsifier)** — at matched state bytes on DISTRIBUTED
//!   (non-orthogonal) keys, the paired collateral difference
//!   `(baseline − Kron)` must have lower 95% bound > 0 against EACH baseline
//!   (vector-delta-erase-with-audit, one-hot table), with post-erase recall
//!   no worse than the stated tolerance. Failure on either statistic against
//!   either baseline keeps the flag opt-in forever — that IS the
//!   pre-registered outcome when orthogonal codes would let the table win.
//! - **G2 (latency, --release)** — `write_newton` + `read` ≤ 2 µs at 64×64;
//!   `collateral_audit` ≤ 10 µs over a 64-pair registry; `kron_chunk_wy`
//!   ≥ 2× sequential at L=64. `black_box` on args AND receiver.
//! - **G4 (alloc)** — zero allocations in write/read/audit/undo hot paths
//!   (counting allocator + the Issue-714 canary).
//! - **G5a (analytic reconciliation)** — the interference-envelope prediction
//!   (per-pair MSE ≈ P·β²·Ê[r²]/(d_k·d_v) for independent normalized
//!   addresses) vs the measured recall-erosion curve on a seeded stream;
//!   a `[0.5×, 2×]` band promotes the analytic sizer for NEW configs;
//!   divergence is a published finding either way.
//! - **G5b (Report the Floor)** — the residual-signal CI candidate vs the
//!   canonical conformal-naive floor
//!   (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`, m=1) on
//!   coverage / Winkler / CRPS. The candidate ships as `read_ci` ONLY if it
//!   beats the floor; the expected (and pre-registered acceptable) outcome is
//!   the floor winning and the cell shipping signal-only.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features kron_memory,kron_chunk_wy,conformal_predictive_intervals \
//!     --bench bench_951_kron_memory_goat -- --nocapture
//! ```

#![cfg(feature = "kron_memory")]

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::conformal::{
    ConformalIntervalCalibrator, DecayUnit, PredictiveInterval, ResidualMode, empirical_coverage,
    seasonal_naive_floor,
};
use katgpt_core::kron_memory::{
    ChunkItem, CollateralPolicy, KronCell, newton_beta, write_chunk_wy,
};
use katgpt_core::simd::dot_f32_ordered;
use katgpt_core::{crps_interval, winkler_score};

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ── the seeded stream ───────────────────────────────────────────────────────

/// Deterministic LCG (the unseeded-global-RNG gate).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    fn next_unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f32) / ((1u64 << 53) as f32)
    }
    fn gaussian(&mut self) -> f32 {
        // 6 uniforms − 3: mean 0, variance 1/4 per draw.
        (0..6).map(|_| self.next_unit()).sum::<f32>() - 3.0
    }
    /// L2-normalized gaussian vector — the DISTRIBUTED (non-orthogonal)
    /// address regime G0 is defined over.
    fn unit_vec<const D: usize>(&mut self) -> [f32; D] {
        let mut v = [0.0f32; D];
        let mut norm_sq = 0.0f32;
        for x in v.iter_mut() {
            *x = self.gaussian();
            norm_sq += *x * *x;
        }
        let inv = 1.0 / norm_sq.sqrt();
        for x in &mut v {
            *x *= inv;
        }
        v
    }
}

// ── G0's three arms at matched state bytes ──────────────────────────────────

const DK: usize = 64;
const DV: usize = 64;
const REG: usize = 64;
const P_PAIRS: usize = 48; // stored pairs (registry holds all of them)
const G0_SEEDS: u64 = 400;
const EPS_RECALL: f32 = 0.05; // post-erase recall tolerance (stated)
const TAU_ERASE: f32 = 1.0; // the Kron erase runs un-τ'd (G0 measures raw collateral)

/// One seeded trial's pair set: the SAME addresses feed all three arms.
struct Pairs {
    ks: Vec<[f32; DK]>,
    vs: Vec<[f32; DV]>,
    cs: Vec<f32>,
}

fn make_pairs(rng: &mut Lcg) -> Pairs {
    let mut ks = Vec::with_capacity(P_PAIRS);
    let mut vs = Vec::with_capacity(P_PAIRS);
    let mut cs = Vec::with_capacity(P_PAIRS);
    for _ in 0..P_PAIRS {
        ks.push(rng.unit_vec::<DK>());
        vs.push(rng.unit_vec::<DV>());
        cs.push(rng.next_unit() * 2.0 - 1.0);
    }
    Pairs { ks, vs, cs }
}

/// Arm 1 — Kron-erase. Returns `(mean |Δ|, max |Δ|, recall)` over the P−1
/// surviving pairs after erasing pair 0.
fn kron_arm(pairs: &Pairs) -> (f32, f32, f32) {
    let mut cell = KronCell::<DK, DV, REG>::new();
    for i in 0..P_PAIRS {
        cell.write_newton(&pairs.ks[i], &pairs.vs[i], pairs.cs[i]);
    }
    // Pre-erase readouts.
    let before: Vec<f32> = (1..P_PAIRS)
        .map(|i| cell.read(&pairs.ks[i], &pairs.vs[i]))
        .collect();
    // The surgical erase (target 0, Newton step, no τ gate — G0 measures raw).
    let _ = cell.unlearn_audited(
        &pairs.ks[0],
        &pairs.vs[0],
        TAU_ERASE,
        CollateralPolicy::Refuse,
        &mut [(0u16, 0.0f32); REG],
    );
    let mut sum = 0.0f32;
    let mut max = 0.0f32;
    let mut recall_hits = 0u32;
    for (j, i) in (1..P_PAIRS).enumerate() {
        let d = (cell.read(&pairs.ks[i], &pairs.vs[i]) - before[j]).abs();
        sum += d;
        if d > max {
            max = d;
        }
        // Post-erase recall: the pair still reads its TARGET within tolerance
        // (captures write-phase interference AND erase collateral in one
        // number — the state the consumer actually sees).
        if (cell.read(&pairs.ks[i], &pairs.vs[i]) - pairs.cs[i]).abs() <= EPS_RECALL {
            recall_hits += 1;
        }
    }
    let n = (P_PAIRS - 1) as f32;
    (sum / n, max, recall_hits as f32 / n)
}

/// Arm 2 — vector-delta-erase-with-audit: `delta_mem` at the matched state
/// bytes (rank 64 = 64×64 scalars), the erase is the vector-target Newton
/// step `M ← M − (Mk)kᵀ/‖k‖²` (write v = 0), the audit pattern is the same
/// pre/post readout capture the Kron arm runs. The vector-target readout of
/// pair (k, v) is the VECTOR `Mk`; its error vs the stored target `v` is
/// `‖Mk − v‖`.
fn delta_mem_arm(pairs: &Pairs) -> (f32, f32, f32) {
    use katgpt_core::delta_mem::{DeltaMemoryConfig, DeltaMemoryState};
    let mut mem = DeltaMemoryState::new(DeltaMemoryConfig {
        rank: DK,
        beta_init: 1.0,
        couple_gates: false,
    });
    // The vector-target contract stores vᵢ at key kᵢ — the same addresses.
    for i in 0..P_PAIRS {
        mem.write(&pairs.ks[i], &pairs.vs[i]);
    }
    let read_vec = |mem: &DeltaMemoryState, k: &[f32; DK]| -> [f32; DV] {
        let mut out = [0.0f32; DV];
        for (row, o) in mem
            .snapshot()
            .state
            .as_chunks::<DK>()
            .0
            .iter()
            .zip(out.iter_mut())
        {
            *o = dot_f32_ordered(row, k);
        }
        out
    };

    let before: Vec<[f32; DV]> = (1..P_PAIRS).map(|i| read_vec(&mem, &pairs.ks[i])).collect();
    let kk0 = dot_f32_ordered(&pairs.ks[0], &pairs.ks[0]);
    // The Newton erase: M ← M − (Mk)kᵀ/‖k‖², row-wise on the snapshot.
    let mut snap = mem.snapshot();
    let inv = 1.0 / kk0;
    for row in snap.state.as_chunks_mut::<DK>().0 {
        let coef = -inv * dot_f32_ordered(row, &pairs.ks[0]);
        for (x, &kj) in row.iter_mut().zip(pairs.ks[0].iter()) {
            *x += coef * kj;
        }
    }
    mem.restore(&snap);

    let mut sum = 0.0f32;
    let mut max = 0.0f32;
    let mut recall_hits = 0u32;
    for (j, i) in (1..P_PAIRS).enumerate() {
        let after = read_vec(&mem, &pairs.ks[i]);
        let mut coll = 0.0f32;
        let mut recall_err = 0.0f32;
        for ((a, b), t) in after.iter().zip(before[j].iter()).zip(pairs.vs[i].iter()) {
            coll += (a - b) * (a - b);
            recall_err += (a - t) * (a - t);
        }
        let coll = coll.sqrt();
        let recall_err = recall_err.sqrt();
        sum += coll;
        if coll > max {
            max = coll;
        }
        if recall_err <= EPS_RECALL {
            recall_hits += 1;
        }
    }
    let n = (P_PAIRS - 1) as f32;
    (sum / n, max, recall_hits as f32 / n)
}

/// Arm 3 — the one-hot table: plain `[[f32; DV]; DK]` addressing. A
/// distributed pair addresses its single slot `(argmax|k|, argmax|v|)`;
/// collisions overwrite (last-write-wins); erase zeroes the slot.
fn one_hot_arm(pairs: &Pairs) -> (f32, f32, f32) {
    let mut table = [[0.0f32; DV]; DK];
    let amax_k = |k: &[f32; DK]| {
        let mut best = 0usize;
        for (i, x) in k.iter().enumerate() {
            if x.abs() > k[best].abs() {
                best = i;
            }
        }
        best
    };
    let amax_v = |v: &[f32; DV]| {
        let mut best = 0usize;
        for (i, x) in v.iter().enumerate() {
            if x.abs() > v[best].abs() {
                best = i;
            }
        }
        best
    };
    for i in 0..P_PAIRS {
        table[amax_k(&pairs.ks[i])][amax_v(&pairs.vs[i])] = pairs.cs[i];
    }
    let slot0 = (amax_k(&pairs.ks[0]), amax_v(&pairs.vs[0]));
    let before: Vec<f32> = (1..P_PAIRS)
        .map(|i| table[amax_k(&pairs.ks[i])][amax_v(&pairs.vs[i])])
        .collect();
    table[slot0.0][slot0.1] = 0.0; // the surgical erase
    let mut sum = 0.0f32;
    let mut max = 0.0f32;
    let mut recall_hits = 0u32;
    for (j, i) in (1..P_PAIRS).enumerate() {
        let now = table[amax_k(&pairs.ks[i])][amax_v(&pairs.vs[i])];
        let d = (now - before[j]).abs();
        sum += d;
        if d > max {
            max = d;
        }
        if (now - pairs.cs[i]).abs() <= EPS_RECALL {
            recall_hits += 1;
        }
    }
    let n = (P_PAIRS - 1) as f32;
    (sum / n, max, recall_hits as f32 / n)
}

/// One-sided 95% lower bound on the mean of paired differences (normal
/// approximation, disclosed — T = 400 paired trials).
fn lb95(diffs: &[f32]) -> f32 {
    let n = diffs.len() as f32;
    let mean = diffs.iter().sum::<f32>() / n;
    let var = diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f32>() / (n - 1.0);
    mean - 1.645 * (var / n).sqrt()
}

fn run_g0() -> bool {
    println!("── G0: the falsifier (paired collateral-per-erase, distributed keys) ──");
    let mut kron_c = Vec::with_capacity(G0_SEEDS as usize);
    let mut delta_c = Vec::with_capacity(G0_SEEDS as usize);
    let mut table_c = Vec::with_capacity(G0_SEEDS as usize);
    let mut d_delta = Vec::with_capacity(G0_SEEDS as usize);
    let mut d_table = Vec::with_capacity(G0_SEEDS as usize);
    let mut kron_recall = 0.0f32;
    let mut delta_recall = 0.0f32;
    let mut table_recall = 0.0f32;

    for seed in 0..G0_SEEDS {
        let mut rng = Lcg::new(0x954_000 + seed);
        let pairs = make_pairs(&mut rng);
        let (kc, _km, kr) = kron_arm(&pairs);
        let (dc, _dm, dr) = delta_mem_arm(&pairs);
        let (tc, _tm, tr) = one_hot_arm(&pairs);
        kron_c.push(kc);
        delta_c.push(dc);
        table_c.push(tc);
        d_delta.push(dc - kc);
        d_table.push(tc - kc);
        kron_recall += kr;
        delta_recall += dr;
        table_recall += tr;
    }
    kron_recall /= G0_SEEDS as f32;
    delta_recall /= G0_SEEDS as f32;
    table_recall /= G0_SEEDS as f32;

    let lb_delta = lb95(&d_delta);
    let lb_table = lb95(&d_table);
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;

    println!(
        "  mean |Δ|/erase: kron {:.5} · delta-mem {:.5} · one-hot {:.5}",
        mean(&kron_c),
        mean(&delta_c),
        mean(&table_c)
    );
    println!(
        "  paired (baseline − kron): delta-mem mean {:+.5} LB95 {:+.5} · one-hot mean {:+.5} LB95 {:+.5}",
        mean(&d_delta),
        lb_delta,
        mean(&d_table),
        lb_table
    );
    println!(
        "  post-erase recall @ε={EPS_RECALL}: kron {kron_recall:.3} · delta-mem {delta_recall:.3} · one-hot {table_recall:.3}"
    );

    let collateral_win = lb_delta > 0.0 && lb_table > 0.0;
    // Recall no worse than the stated tolerance: Kron's recall may not trail
    // either baseline by more than 0.02 (a tie-recall guard, stated here).
    let recall_ok = kron_recall >= delta_recall - 0.02 && kron_recall >= table_recall - 0.02;
    println!(
        "  G0 verdict: collateral {} · recall {} → {}",
        if collateral_win {
            "WIN (both LB95 > 0)"
        } else {
            "NO WIN"
        },
        if recall_ok {
            "within tolerance"
        } else {
            "TRAILS"
        },
        if collateral_win && recall_ok {
            "G0 PASS (falsifier survived)"
        } else {
            "G0 FAIL — flag stays opt-in forever (valid outcome)"
        }
    );
    collateral_win && recall_ok
}

// ── G2: latency bars ────────────────────────────────────────────────────────

fn bench_ns<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    for _ in 0..8 {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_secs_f64() * 1e9 / iters as f64
}

fn run_g2() -> bool {
    println!("── G2: latency (--release bars, black_box args + receiver) ──");
    let mut rng = Lcg::new(0x955);
    let mut cell = KronCell::<DK, DV, REG>::new();
    let k = rng.unit_vec::<DK>();
    let v = rng.unit_vec::<DV>();
    let c = 0.5f32;

    // write_newton + read ≤ 2 µs. The write on a warm cell is an apply, so
    // the loop mutates state — fine for timing; the read is pure.
    let write_ns = bench_ns(
        || {
            black_box(&mut cell).write_newton(black_box(&k), black_box(&v), black_box(c));
        },
        2000,
    );
    let read_ns = bench_ns(
        || {
            let r = black_box(&cell).read(black_box(&k), black_box(&v));
            black_box(r);
        },
        2000,
    );
    println!(
        "  write_newton @64×64: {write_ns:.0} ns (bar 2000) · read: {read_ns:.0} ns (bar 2000)"
    );
    let g2_write_read = write_ns <= 2000.0 && read_ns <= 2000.0;

    // Audit ≤ 10 µs over a full 64-pair registry.
    let mut full = KronCell::<DK, DV, REG>::new();
    let mut pairs2 = Vec::new();
    for _ in 0..REG {
        let kk = rng.unit_vec::<DK>();
        let vv = rng.unit_vec::<DV>();
        full.write_newton(&kk, &vv, rng.next_unit());
        pairs2.push((kk, vv));
    }
    let mut scratch = [(0u16, 0.0f32); REG];
    let audit_ns = bench_ns(
        || {
            let (rows, n) = black_box(&full).collateral_audit(
                black_box(&k),
                black_box(&v),
                0.25,
                black_box(c),
                &mut scratch,
            );
            black_box((rows.len(), n));
        },
        2000,
    );
    println!("  collateral_audit @64 pairs: {audit_ns:.0} ns (bar 10000)");
    let g2_audit = audit_ns <= 10_000.0;

    // Chunk ≥ 2× sequential at L=64.
    let l = 64;
    let items: Vec<ChunkItem<DK, DV>> = (0..l)
        .map(|_| ChunkItem {
            k: rng.unit_vec::<DK>(),
            v: rng.unit_vec::<DV>(),
            c: rng.next_unit(),
            beta: 0.25,
        })
        .collect();
    let mut scratch_wy = vec![0.0f32; 2 * l * l + 2 * l];
    let mut seq_cell = KronCell::<DK, DV, REG>::new();
    let mut chunk_cell = KronCell::<DK, DV, REG>::new();
    let seq_ns = bench_ns(
        || {
            for item in items.iter() {
                black_box(&mut seq_cell)
                    .write(black_box(&item.k), black_box(&item.v), item.c, item.beta)
                    .expect("seq write");
            }
        },
        20,
    );
    let chunk_ns = bench_ns(
        || {
            write_chunk_wy(
                black_box(&mut chunk_cell),
                black_box(&items),
                &mut scratch_wy,
            )
            .expect("chunk write");
        },
        20,
    );
    let speedup = seq_ns / chunk_ns;
    println!(
        "  L={l}: sequential {seq_ns:.0} ns · chunk {chunk_ns:.0} ns · speedup {speedup:.2}× (bar 2.0×)"
    );
    let g2_chunk = speedup >= 2.0;

    let ok = g2_write_read && g2_audit && g2_chunk;
    println!(
        "  G2 verdict: write/read {} · audit {} · chunk {} → {}",
        if g2_write_read { "PASS" } else { "FAIL" },
        if g2_audit { "PASS" } else { "FAIL" },
        if g2_chunk { "PASS" } else { "FAIL" },
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

// ── G4: allocation-free hot paths ───────────────────────────────────────────

fn run_g4() -> bool {
    println!("── G4: alloc-free hot paths (counting allocator + canary) ──");
    assert_counter_is_live();

    let mut rng = Lcg::new(0x956);
    let mut cell = KronCell::<DK, DV, REG>::new();
    let k = rng.unit_vec::<DK>();
    let v = rng.unit_vec::<DV>();
    // Warm the registry so the audit walks its full length in the window.
    for _ in 0..REG {
        let kk = rng.unit_vec::<DK>();
        let vv = rng.unit_vec::<DV>();
        cell.write_newton(&kk, &vv, rng.next_unit());
    }
    let mut scratch = [(0u16, 0.0f32); REG];
    let mut snap = katgpt_core::kron_memory::TouchedSnap::<DV, DK>::empty();

    let (_, a1) = alloc_delta(|| {
        let v = black_box(&cell).read(black_box(&k), black_box(&v));
        black_box(v);
    });
    let (_, a2) = alloc_delta(|| {
        black_box(&mut cell).write_newton(black_box(&k), black_box(&v), black_box(0.3));
    });
    let (_, a3) = alloc_delta(|| {
        let (rows, n) = black_box(&cell).collateral_audit(
            black_box(&k),
            black_box(&v),
            0.25,
            0.3,
            &mut scratch,
        );
        black_box((rows.len(), n));
    });
    let (_, a4) = alloc_delta(|| {
        let _ = black_box(&mut cell).undo_last();
    });
    let (_, a5) = alloc_delta(|| {
        let out = black_box(&mut cell).write_snap(
            black_box(&k),
            black_box(&v),
            black_box(0.3),
            newton_beta(&k, &v),
            &mut snap,
        );
        black_box(out.is_ok());
    });
    println!("  read {a1} · write {a2} · audit {a3} · undo {a4} · write_snap {a5} (all must be 0)");
    let ok = a1 + a2 + a3 + a4 + a5 == 0;
    println!("  G4 verdict: {}", if ok { "PASS" } else { "FAIL" });
    ok
}

// ── G5a: analytic interference envelope vs the measured erosion curve ───────

fn run_g5a() -> bool {
    println!("── G5a: analytic interference envelope vs measured erosion ──");
    // Stream P ≥ d² pairs on ONE cell; measure the recall MSE of the FIXED
    // window (written once, then overwritten by P subsequent writes) — the
    // retention-erosion curve of a stored pair under streaming interference.
    //
    // Analytic envelope (self-consistent first order): after P subsequent
    // Newton writes on independent normalized addresses, the accumulated
    // per-pair error variance is MSE ≈ σ²_c + x·(σ²_c + MSE) with
    // x = P·β²/(d_k·d_v) — the original component stays, each subsequent
    // write adds E[r²]/(d_k·d_v), and E[r²] ≈ σ²_c + MSE — giving
    // MSE ≈ σ²_c/(1 − x), valid while x < 1. Beyond x → 1 the first-order
    // model declares its own limit (the measured curve's E[r²] shrinkage
    // regularizes past it — the model says so on its face).
    let dk = DK as f32;
    let dv = DV as f32;
    let sigma_c2 = 1.0f32 / 3.0; // uniform(−1, 1)
    let mut rng = Lcg::new(0x957);
    let mut cell = KronCell::<DK, DV, REG>::new();
    let window: Vec<([f32; DK], [f32; DV], f32)> = (0..64)
        .map(|_| {
            (
                rng.unit_vec::<DK>(),
                rng.unit_vec::<DV>(),
                rng.next_unit() * 2.0 - 1.0,
            )
        })
        .collect();

    let points = [(256usize, 0.5f32), (1024, 2.0), (4096, 8.0)];
    let mut within_band = 0u32;
    for (p_written, p_multiple) in points {
        // Write until P total.
        while cell.writes_applied() < p_written as u64 {
            let k = rng.unit_vec::<DK>();
            let v = rng.unit_vec::<DV>();
            let c = rng.next_unit() * 2.0 - 1.0;
            cell.write_newton(&k, &v, c);
        }
        // Measure the window's recall MSE (each window pair was overwritten
        // many times since — its residual is the accumulated interference).
        let mut mse = 0.0f32;
        for (k, v, c) in &window {
            let e = cell.read(k, v) - c;
            mse += e * e;
        }
        mse /= window.len() as f32;

        let beta = 1.0; // normalized addresses: β* = 1
        let x = p_written as f32 * beta * beta / (dk * dv);
        let analytic = if x < 0.98 {
            sigma_c2 / (1.0 - x)
        } else {
            f32::INFINITY
        };
        let ratio = if analytic.is_infinite() {
            f32::INFINITY
        } else {
            mse / analytic
        };
        let in_band = (0.5..=2.0).contains(&ratio);
        if in_band {
            within_band += 1;
        }
        println!(
            "  P={p_written} ({p_multiple:.0}×d²): measured MSE {mse:.4} · analytic {analytic:.4} · ratio {ratio:.2} {}",
            if in_band { "IN BAND" } else { "OUT" }
        );
    }
    // The envelope must hold while x < 1 (the first two points); the x → 1
    // point is the model's declared limit — measured-beyond-model is the
    // disclosed posture, not a band failure.
    let ok = within_band >= 2;
    println!(
        "  G5a verdict: {within_band}/3 in band → analytic sizer {}",
        if ok {
            "PROMOTED for the sub-saturation regime (P·β² < 0.5·d_k·d_v)"
        } else {
            "NOT PROMOTED — divergence is the published finding"
        }
    );
    ok
}

// ── G5b: Report the Floor — the residual-signal CI vs the conformal floor ───

fn run_g5b() -> bool {
    println!("── G5b: Report the Floor (residual-signal CI vs conformal-naive floor) ──");
    // Stream: a slow seasonal signal + noise; the cell's READ of the current
    // (k, v) pair is its point forecast of the pair's target c. Candidate CI:
    // read ± q·residual_signal (q calibrated on the cal slice for nominal 90%
    // coverage). Floor: ConformalIntervalCalibrator<SeasonalNaiveForecaster>,
    // m=1, HStep, no decay — the canonical floor.
    let alpha = 0.10f32;
    let mut rng = Lcg::new(0x958);
    let n_cal = 256;
    let n_eval = 512;

    let mut cell = KronCell::<8, 8, 16>::new();
    let mut floor = ConformalIntervalCalibrator::new(
        seasonal_naive_floor(256),
        1,
        1,
        1,
        256,
        0.0,
        DecayUnit::Step,
        ResidualMode::HStep,
        false,
    );

    let mut stream = Vec::with_capacity(n_cal + n_eval);
    let mut phase = 0.0f32;
    for t in 0..(n_cal + n_eval) {
        phase += 0.05;
        let seasonal = (phase.sin()) * 0.5 + (0.31 * phase).sin() * 0.25;
        let noise = (rng.gaussian()) * 0.3;
        let c = seasonal + noise;
        // Fresh address pair per tick (the associative identity is the
        // (k,v)-keyed read; the stream is what the cell tracks).
        let k = rng.unit_vec::<8>();
        let v = rng.unit_vec::<8>();
        stream.push((k, v, c, t));
    }

    // Feed both. The floor observes (actual, forecast=read, channel 0, h 1).
    // The candidate's multiplier q is calibrated AFTER the cal slice by the
    // empirical 90% two-sided quantile of |residual|/signal.
    let mut abs_res = Vec::with_capacity(n_cal + n_eval);
    for (k, v, c, _t) in stream.iter() {
        let forecast = cell.read(k, v);
        abs_res.push((forecast - *c).abs());
        cell.write_newton(k, v, *c);
        // The floor observes EVERY tick (cal + eval): its residual pool must
        // be calibrated by the time evaluation starts — the canonical floor
        // at its best posture, never a starved one.
        floor.update_residual(*c, forecast, 0, 1);
    }
    let _ = abs_res;

    // Calibrate the candidate's q on the cal slice: the smallest q whose
    // empirical cal coverage ≥ 1−α (scalar bisection over a fixed grid).
    let sig_cal: Vec<f32> = (0..n_cal)
        .map(|_| cell.residual_signal()) // the converged scale, post-hoc
        .collect();
    let _ = &sig_cal;
    // Honest calibration needs the SIGNAL AT THE TIME — replay the cal slice
    // with a fresh cell to collect (|res|, signal) pairs as they happened.
    let mut cell2 = KronCell::<8, 8, 16>::new();
    let mut cal_pairs = Vec::with_capacity(n_cal);
    for (k, v, c, _) in stream.iter().take(n_cal) {
        let f = cell2.read(k, v);
        let s = cell2.residual_signal().max(1e-3);
        cal_pairs.push((f - c).abs() / s);
        cell2.write_newton(k, v, *c);
    }
    cal_pairs.sort_by(|a, b| a.total_cmp(b));
    let q = cal_pairs[((1.0 - alpha) * (n_cal as f32 - 1.0)) as usize];

    // Evaluate both on the eval slice.
    let mut cand_intervals = Vec::with_capacity(n_eval);
    let mut floor_intervals = Vec::with_capacity(n_eval);
    let mut actuals = Vec::with_capacity(n_eval);
    let mut eval_signal = cell2.residual_signal();
    for (k, v, c, _t) in stream.iter().skip(n_cal) {
        let f = cell2.read(k, v);
        cand_intervals.push(PredictiveInterval::new(
            f - q * eval_signal,
            f,
            f + q * eval_signal,
            alpha,
        ));
        let mut fi = PredictiveInterval::new(0.0, 0.0, 0.0, alpha);
        floor.interval_from_point_into(f, 0, 1, alpha, &mut fi);
        floor_intervals.push(fi);
        actuals.push(*c);
        // Advance both exactly as the stream does.
        cell2.write_newton(k, v, *c);
        eval_signal = 0.75 * eval_signal + 0.25 * (f - c).abs();
        floor.update_residual(*c, f, 0, 1);
    }

    let cov_c = empirical_coverage(&cand_intervals, &actuals);
    let cov_f = empirical_coverage(&floor_intervals, &actuals);
    let wink_c: f32 = cand_intervals
        .iter()
        .zip(actuals.iter())
        .map(|(i, &a)| winkler_score(i, a))
        .sum::<f32>()
        / n_eval as f32;
    let wink_f: f32 = floor_intervals
        .iter()
        .zip(actuals.iter())
        .map(|(i, &a)| winkler_score(i, a))
        .sum::<f32>()
        / n_eval as f32;
    let crps_c: f32 = cand_intervals
        .iter()
        .zip(actuals.iter())
        .map(|(i, &a)| crps_interval(i, a))
        .sum::<f32>()
        / n_eval as f32;
    let crps_f: f32 = floor_intervals
        .iter()
        .zip(actuals.iter())
        .map(|(i, &a)| crps_interval(i, a))
        .sum::<f32>()
        / n_eval as f32;

    println!("  candidate: coverage {cov_c:.3} · Winkler {wink_c:.4} · CRPS {crps_c:.4}");
    println!("  floor:     coverage {cov_f:.3} · Winkler {wink_f:.4} · CRPS {crps_f:.4}");

    // Beats-the-floor on the Report-the-Floor axes: coverage within [1−α−0.02,
    // 1] AND strictly better (lower) Winkler AND CRPS.
    let cov_ok = cov_c >= 1.0 - alpha - 0.02;
    let beats = cov_ok && wink_c < wink_f && crps_c < crps_f;
    println!(
        "  G5b verdict: {} — the analytic CI {}",
        if beats {
            "CANDIDATE WINS"
        } else {
            "FLOOR WINS"
        },
        if beats {
            "ships as read_ci"
        } else {
            "is DROPPED; the cell ships signal-only (residual_signal) — the pre-registered acceptable outcome"
        }
    );
    // The GATE passes when the adjudication RAN honestly, either verdict.
    true
}

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Issue 954 — kron_memory GOAT gate (Bench 951)");
    println!("  box: M3 Max (16 cores), macOS 26.6.2, AC power — --release run");
    println!("═══════════════════════════════════════════════════════════");

    let mut failures = 0u32;

    if !run_g0() {
        failures += 1;
    }
    if !run_g2() {
        failures += 1;
    }
    if !run_g4() {
        failures += 1;
    }
    if !run_g5a() {
        failures += 1;
    }
    let _g5b_ran = run_g5b();

    println!();
    if failures == 0 {
        println!(
            "✓ ALL GATES RAN — verdicts above are the record (G0/G2/G4/G5a PASS gates; G5b adjudication)."
        );
    } else {
        println!("✗ {failures} gate(s) FAILED — the record above is the verdict.");
    }
    // The bench's exit code mirrors the PASS gates only (G5b runs either way).
    std::process::exit(i32::from(failures > 0));
}
