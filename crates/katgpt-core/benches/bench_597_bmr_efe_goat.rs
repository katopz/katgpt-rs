//! Plan 597 Phase 5 — BMR + EFE model-gain GOAT gate bench.
//!
//! Exercises G1/G2/G4 for the `bmr` primitive on the three-ball model space
//! (81 models × 81 columns × 3 rows). G3 (no-regression) is the external
//! feature-flag build matrix (default tests green + clippy clean + wasm32
//! check), recorded in `.benchmarks/715_bmr_efe_goat.md`.
//!
//! # Gates
//!
//! - **G1 (correctness — modelless):** the BMR log-evidence identity against
//!   the exact sequential-predictive oracle (the permanent regression
//!   anchor's bench-side rerun): `ln p(D|full) − F(m)` must match the
//!   chained marginal likelihood to 1e-9, and the symmetric edge must be
//!   exactly 0. Plus the sparse-Δ predictive posterior vs naive full
//!   recompute (max abs diff < 1e-9 over all 81 models × sampled columns).
//!
//! - **G2 (perf):** on the 81-model space:
//!   (i) `ModelSpace::new` init (O(#models × #cols));
//!   (ii) sparse-Δ predictive posterior per action;
//!   (iii) `efe_model_gain` per action (3-outcome anticipation);
//!   (iv) naive full-recompute predictive posterior baseline.
//!   **PASS** if sparse-Δ ≥ 100× faster than naive AND per-action eval is
//!   single-digit µs on this machine (M3; --release).
//!
//! - **G4 (alloc-free hot path):** `predictive_posterior_into` +
//!   `efe_model_gain` + `accumulate` allocate 0 times over 1000
//!   steady-state calls (CountingAllocator, per-thread — Issue 714).
//!
//! # Run
//!
//! ```bash
//! cargo run --release -p katgpt-core --features bmr --bench bench_597_bmr_efe_goat
//! ```

#![cfg(feature = "bmr")]

use katgpt_core::bmr::{
    self, Counts, EfeScratch, FactorLayout, ModelSpace, enumerate_isomorphic_rules,
    occam_log_bayes_factor, predictive_model_posterior,
};
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── Fixture: the three-ball model space ────────────────────────────────────

const N_MODELS: usize = 81;
const ROWS: usize = 3;
const N_COLS: usize = 81;

fn three_ball_engine(seed: u64, prior_fill: f64) -> (ModelSpace, Vec<Counts>) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut layout = FactorLayout::new(&[3, 3, 3], 3, ROWS, 1, 2);
    layout.unconstrained_fill = prior_fill;
    let models = enumerate_isomorphic_rules(&layout);
    let prior = Counts::filled(ROWS, N_COLS, prior_fill);
    let mut post = prior.clone();
    // Seed some accumulated evidence (mix of none/reward/penalty columns).
    for _ in 0..40 {
        let col = rng.usize(..N_COLS);
        let row = rng.usize(..ROWS);
        post.add(col, row, 1.0);
    }
    let engine = ModelSpace::new(prior.clone(), models.clone(), post);
    (engine, models)
}

// ─── G1: correctness ────────────────────────────────────────────────────────

/// Exact one-column marginal likelihood by sequential predictive chaining
/// (integer counts, no lgamma) — the independent oracle.
fn chained_ln_evidence_col(alpha: &[f64], units: &[usize]) -> f64 {
    let mut a = alpha.to_vec();
    let mut ln_p = 0.0;
    for (i, &k) in units.iter().enumerate() {
        for _ in 0..k {
            let s: f64 = a.iter().sum();
            ln_p += (a[i] / s).ln();
            a[i] += 1.0;
        }
    }
    ln_p
}

fn g1_correctness() -> bool {
    let mut ok = true;

    // (1) BMR evidence vs chained oracle on random small tensors.
    let mut rng = fastrand::Rng::with_seed(7151);
    let mut worst = 0.0f64;
    for _ in 0..40 {
        let rows = 2 + rng.usize(..3);
        let cols = 1 + rng.usize(..3);
        let prior = Counts::filled(rows, cols, 4.0);
        let mut post = prior.clone();
        let mut units = vec![vec![0usize; rows]; cols];
        for (c, u) in units.iter_mut().enumerate() {
            for (r, v) in u.iter_mut().enumerate() {
                *v = rng.usize(..6);
                post.add(c, r, *v as f64);
            }
        }
        let mut ln_full = 0.0;
        for (c, u) in units.iter().enumerate() {
            ln_full += chained_ln_evidence_col(prior.col(c), u);
        }
        for _ in 0..4 {
            // Random reduced priors (zeros exercise the shrinkage convention).
            let mut reduced = Counts::zero(rows, cols);
            for c in 0..cols {
                for r in 0..rows {
                    reduced.col_mut(c)[r] = [0.0f64, 1.0, 4.0, 8.0][rng.usize(..4)];
                }
            }
            let f = bmr::bmr_log_evidence(&prior, &post, &reduced);
            let mut ln_m = 0.0;
            for (c, u) in units.iter().enumerate() {
                let alpha: Vec<f64> = reduced.col(c).iter().map(|v| v.max(bmr::SHRINKAGE)).collect();
                ln_m += chained_ln_evidence_col(&alpha, u);
            }
            worst = worst.max((ln_full - f - ln_m).abs());
        }
    }
    let g1a = worst < 1e-9;
    println!(
        "G1a (BMR vs chained oracle): worst |Δ| = {worst:.3e} (target < 1e-9) → {}",
        if g1a { "PASS" } else { "FAIL" }
    );
    ok &= g1a;

    // (2) Symmetric edge: reduced == prior ⇒ exactly 0.
    let (engine, models) = three_ball_engine(7152, 4.0);
    let prior = engine.prior().clone();
    let f0 = bmr::bmr_log_evidence(&prior, engine.post(), &prior);
    let g1b = f0 == 0.0;
    println!(
        "G1b (symmetric edge): F(ã_m = ã) = {f0:.3e} (target exactly 0) → {}",
        if g1b { "PASS" } else { "FAIL" }
    );
    ok &= g1b;

    // (3) Sparse-Δ predictive vs naive full recompute over the 81-model space.
    let mut sparse = vec![0.0f64; N_MODELS];
    let mut naive = vec![0.0f64; N_MODELS];
    let mut worst = 0.0f64;
    let mut rng = fastrand::Rng::with_seed(7153);
    for _ in 0..27 {
        let col = rng.usize(..N_COLS);
        let row = rng.usize(..ROWS);
        engine.predictive_posterior_into(col, row, &mut sparse);
        predictive_model_posterior(engine.prior(), &models, engine.post(), col, row, &mut naive);
        for m in 0..N_MODELS {
            worst = worst.max((sparse[m] - naive[m]).abs());
        }
    }
    let g1c = worst < 1e-9;
    println!(
        "G1c (sparse-Δ predictive vs full recompute): worst |Δ| = {worst:.3e} (target < 1e-9) → {}",
        if g1c { "PASS" } else { "FAIL" }
    );
    ok &= g1c;

    // (4) Occam sanity on a committed posterior.
    let mut one_hot = vec![0.0f64; N_MODELS];
    one_hot[42] = 1.0;
    let occam = occam_log_bayes_factor(&one_hot);
    let g1d = occam > 30.0 && occam.is_finite();
    println!(
        "G1d (Occam saturation): one-hot Occam = {occam:.2} (finite, > 30) → {}",
        if g1d { "PASS" } else { "FAIL" }
    );
    ok &= g1d;

    ok
}

// ─── G2: perf ───────────────────────────────────────────────────────────────

fn g2_perf() -> (bool, f64, f64, f64, f64) {
    let (engine, _models) = three_ball_engine(7154, 4.0);
    let mut scratch = EfeScratch::new();
    let mut out = vec![0.0f64; N_MODELS];
    let mut post = vec![0.0f64; N_MODELS];
    engine.posterior_into(&mut post);

    // (i) init: O(#models × #cols).
    let n_init = 200;
    let t0 = Instant::now();
    for i in 0..n_init {
        let (e, _) = three_ball_engine(7154 + i, 4.0);
        std::hint::black_box(&e);
    }
    let init_us = t0.elapsed().as_secs_f64() * 1e6 / n_init as f64;

    // A representative anticipated action: pick k in cell c (3 outcomes).
    let action: Vec<(usize, usize, f64)> = {
        let q = [0.2f64, 0.5, 0.3];
        let col = 5 * 3 + 1;
        q.iter()
            .enumerate()
            .map(|(row, &p)| (col, row, p))
            .collect()
    };

    // (ii) sparse-Δ predictive posterior per action.
    let n = 100_000;
    let t0 = Instant::now();
    for _ in 0..n {
        engine.predictive_posterior_into(16, 1, &mut out);
        std::hint::black_box(&out);
    }
    let sparse_us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;

    // (iii) efe_model_gain per action (3 anticipated-outcome branches).
    let t0 = Instant::now();
    for _ in 0..n {
        std::hint::black_box(engine.efe_model_gain(&action, &mut scratch));
    }
    let efe_us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;

    // (iv) naive full-recompute predictive posterior (the baseline).
    let (_, models) = three_ball_engine(7156, 4.0);
    let prior = engine.prior().clone();
    let post_c = engine.post().clone();
    let n2 = 2_000;
    let t0 = Instant::now();
    for _ in 0..n2 {
        predictive_model_posterior(&prior, &models, &post_c, 16, 1, &mut out);
        std::hint::black_box(&out);
    }
    let naive_us = t0.elapsed().as_secs_f64() * 1e6 / n2 as f64;

    let speedup = naive_us / sparse_us;
    let pass = speedup >= 100.0 && sparse_us < 10.0 && efe_us < 10.0;
    println!(
        "G2 (81-model space, --release):\n  \
           (i)   ModelSpace::new init          = {init_us:8.1} µs\n  \
           (ii)  sparse-Δ predictive posterior = {sparse_us:8.2} µs/action\n  \
           (iii) efe_model_gain (3 branches)   = {efe_us:8.2} µs/action\n  \
           (iv)  naive full recompute          = {naive_us:8.1} µs/action\n  \
           speedup (iv)/(ii) = {speedup:.0}× (target ≥ 100×), per-action single-digit µs → {}",
        if pass { "PASS" } else { "FAIL" }
    );
    (pass, sparse_us, efe_us, naive_us, speedup)
}

// ─── G4: alloc-free hot path ────────────────────────────────────────────────

fn g4_alloc_free() -> bool {
    assert_counter_is_live();
    let (mut engine, _) = three_ball_engine(7157, 4.0);
    let mut scratch = EfeScratch::new();
    let mut out = vec![0.0f64; N_MODELS];
    let action: Vec<(usize, usize, f64)> = vec![(16, 0, 0.3), (16, 1, 0.4), (16, 2, 0.3)];

    // Warmup: ensure any lazily-initialized state is settled.
    for _ in 0..10 {
        engine.predictive_posterior_into(16, 1, &mut out);
        let _ = engine.efe_model_gain(&action, &mut scratch);
        engine.accumulate(16, 1, 1.0);
    }

    let n_calls = 1000;
    let (_, allocs) = alloc_delta(|| {
        for i in 0..n_calls {
            engine.predictive_posterior_into(16, i % 3, &mut out);
            std::hint::black_box(&out);
            std::hint::black_box(engine.efe_model_gain(&action, &mut scratch));
            engine.accumulate(16, i % 3, 1.0);
        }
    });
    let pass = allocs == 0;
    println!(
        "G4 (alloc-free hot path): {allocs} allocations over {n_calls}×3 steady-state calls (target 0) → {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 597 — BMR + EFE-over-models GOAT gate ===\n");

    let g1 = g1_correctness();
    let (g2, sparse_us, efe_us, naive_us, speedup) = g2_perf();
    let g4 = g4_alloc_free();

    println!();
    println!(
        "Verdict: G1={} G2={} G3=(external: default-tests/clippy/wasm32, see bench doc) G4={}",
        if g1 { "PASS" } else { "FAIL" },
        if g2 { "PASS" } else { "FAIL" },
        if g4 { "PASS" } else { "FAIL" },
    );
    println!(
        "        sparse-Δ {sparse_us:.2}µs vs naive {naive_us:.1}µs ({speedup:.0}×); efe {efe_us:.2}µs"
    );

    let all_pass = g1 && g2 && g4;
    println!();
    if all_pass {
        println!("ALL GATES PASS — primitive is GOAT-validated (promotion is the coordinator's call).");
    } else {
        println!("ONE OR MORE GATES FAILED — stays opt-in; record in the bench doc.");
        std::process::exit(1);
    }
}
