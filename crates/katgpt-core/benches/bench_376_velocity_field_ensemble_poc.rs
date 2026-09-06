//! Plan 376 Phase 2 — Velocity-Field Ensemble Cross-Domain Quality PoC.
//!
//! **Defend-wrong PoC (AGENTS.md §3.6):** before any cross-domain quality-parity
//! claim, run a head-to-head comparison of three competitors on a held-out
//! target domain and let the numbers speak.
//!
//! # Setup
//!
//! Synthetic linear-velocity-field domain (D=8). The "target" is a fixed random
//! linear field `b*(x) = W* x`. Three "source drafters" are constructed in two
//! regimes:
//!
//! - **Regime 1 (related sources):** each source is `W_i = W* + Δ_i` — models
//!   cross-domain composition where sources share structure with the target
//!   (the F-MNIST → MNIST case from the paper's Appendix E).
//! - **Regime 2 (unrelated sources):** each source is an independent random
//!   `W_i` — the null case; sources have no structural relation to the target.
//!
//! N_train = 200 fit pairs, N_test = 200 held-out pairs. Each pair is
//! `(x_n, İ_t_n)` with `x_n ~ N(0, I_D)` and `İ_t_n = W* x_n + ε_n` (label noise).
//!
//! # Competitors
//!
//! - **(a) single-best:** for each source `i`, evaluate `b_i` alone on train;
//!   pick the source with the best train MSE; report its test metrics. This is
//!   the paper's "frozen, no-adaptation" baseline (§3.3).
//! - **(b) cross-domain ensemble (this primitive):** ridge-solve η over the 3
//!   sources on train; report test metrics of `Σ η_i b_i(x)`.
//! - **(c) target-trained-from-scratch:** solve a single linear `W_approx`
//!   directly from the 200 train pairs via per-row least-squares (8 ridge
//!   solves of size 8, reusing `ridge_solve_direct_f32`). This is the closed-
//!   form analog of "train a fresh model on target data" — the reference upper
//!   bound.
//!
//! # Metrics (all on held-out test set)
//!
//! 1. **MSE** — `mean_n ‖b̂(x_n) − İ_t_n‖² / D` (primary regression metric).
//! 2. **top-1 agreement** — fraction of test pairs where
//!    `argmax_k b̂(x_n)[k] == argmax_k İ_t_n[k]`.
//! 3. **mean rank** — mean over test pairs of the rank of the true argmax
//!    action in the predicted ranking (lower = better; 1 = perfect).
//! 4. **NLL** — `-log(σ(s_true) / Σ_k σ(s_k))` where σ = sigmoid, `s = b̂(x)`.
//!    Sigmoid-normalized categorical (the AGENTS.md rule is honored: the
//!    primitive's η are ridge-solved, never softmax-normalized; this NLL is a
//!    measurement tool only, not part of the primitive).
//!
//! # G2 verdict (Plan 376 Phase 3 T3.2)
//!
//! **PASS** = competitor (b) beats competitor (a) on **≥ 2 of 3 primary metrics**
//! (MSE, top-1, mean-rank). NLL is reported but does not gate (it is
//! mathematically equivalent to softmax-NLL; the rule's spirit is about
//! weight *combination*, not final-action readout).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/vfe_376 cargo build --release -p katgpt-core \
//!     --features velocity_field_ensemble --bench bench_376_velocity_field_ensemble_poc
//! /tmp/vfe_376/release/bench_376_velocity_field_ensemble_poc-* --nocapture
//! ```

#![cfg(feature = "velocity_field_ensemble")]

use katgpt_core::linalg::ridge_solve::ridge_solve_direct_f32;
use katgpt_core::velocity_field_ensemble::{
    ClosureField, EnsembleFitScratch, VelocityFieldEnsemble,
};

// ── Constants ─────────────────────────────────────────────────────────────

const D: usize = 8;
const N_SOURCES: usize = 3;
const N_TRAIN: usize = 200;
const N_TEST: usize = 200;

// Source bias magnitude (Regime 1). σ_bias = 0.3 means each source is W* plus
// a perturbation of typical magnitude 0.3 per entry. W* entries are ~U(-0.5, 0.5),
// so the bias is ~60% of the signal — substantial but not overwhelming.
const SIGMA_BIAS: f32 = 0.3;
// Label noise on İ_t. σ_noise = 0.05 — small relative to the signal scale.
const SIGMA_NOISE: f32 = 0.05;

// Ridge λ for the ensemble fit and for (c)'s per-row solve.
const LAMBDA_ENSEMBLE: f32 = 1e-4;
const LAMBDA_FROM_SCRATCH: f32 = 1e-4;

const MASTER_SEED: u64 = 0x376_5EED_3765u64;

// ── Deterministic LCG (matches codebase bench convention) ─────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    /// Uniform float in [0, 1).
    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform float in [-range, +range).
    #[inline]
    fn next_signed(&mut self, range: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * range
    }
    /// Standard-normal via Box-Muller.
    #[inline]
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

// ── Linear velocity field `b(x) = W x` ────────────────────────────────────

/// A frozen linear velocity field `b(x) = W x` where `W ∈ R^{D×D}` row-major.
///
/// Stored as a flat `[f32; D*D]` so it's `Copy` and can be captured by closures.
/// Each field has a unique `field_id` for the Gram (we use the index).
#[derive(Clone, Copy)]
struct LinearFieldW {
    w: [f32; D * D],
    id: u64,
}

impl LinearFieldW {
    fn eval(&self, x: &[f32], out: &mut [f32; D]) {
        for (k, out_k) in out.iter_mut().enumerate().take(D) {
            let mut acc = 0.0f32;
            for (j, x_j) in x.iter().enumerate().take(D) {
                acc += self.w[k * D + j] * x_j;
            }
            *out_k = acc;
        }
    }
}

/// Wrap a `LinearFieldW` as a `VelocityField<D>` via a closure-field. Each field
/// needs its own closure type to be array-stored in `[F; P]`, so we leak the
/// `LinearFieldW` into the closure by copy (it's small: 65 floats).
fn make_linear_closure_field(
    field: LinearFieldW,
) -> ClosureField<D, impl Fn(&[f32], &mut [f32; D])> {
    ClosureField::new(field.id, move |x: &[f32], out: &mut [f32; D]| {
        field.eval(x, out);
    })
}

// ── Synthetic data generation ─────────────────────────────────────────────

struct Dataset {
    /// `(N, D)` input states, row-major.
    x: Vec<f32>,
    /// `(N, D)` target derivatives İ_t, row-major.
    y: Vec<f32>,
}

impl Dataset {
    fn n(&self) -> usize {
        self.x.len() / D
    }
    fn x_row(&self, n: usize) -> &[f32] {
        &self.x[n * D..(n + 1) * D]
    }
    fn y_row(&self, n: usize) -> &[f32] {
        &self.y[n * D..(n + 1) * D]
    }
}

/// Generate N pairs `(x_n, İ_t_n)` where `x_n ~ N(0, I_D)` and
/// `İ_t_n = W* x_n + ε_n` with `ε_n ~ N(0, σ²·I_D)`.
fn gen_pairs(rng: &mut Lcg, w_star: &[f32; D * D], n: usize, sigma_noise: f32) -> Dataset {
    let mut x = vec![0.0f32; n * D];
    let mut y = vec![0.0f32; n * D];
    for i in 0..n {
        for j in 0..D {
            x[i * D + j] = rng.next_normal();
        }
        for k in 0..D {
            let mut acc = 0.0f32;
            for j in 0..D {
                acc += w_star[k * D + j] * x[i * D + j];
            }
            y[i * D + k] = acc + sigma_noise * rng.next_normal();
        }
    }
    Dataset { x, y }
}

/// Generate a random `D×D` matrix with entries ~U(-mag, +mag).
fn gen_random_matrix(rng: &mut Lcg, mag: f32) -> [f32; D * D] {
    let mut w = [0.0f32; D * D];
    for v in w.iter_mut() {
        *v = rng.next_signed(mag);
    }
    w
}

// ── Metrics ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Default, Debug)]
struct Metrics {
    /// Mean squared error per coordinate: `mean_n ‖b̂(x_n) − İ_t_n‖² / D`.
    mse: f64,
    /// Fraction of test pairs where argmax matches.
    top1: f64,
    /// Mean rank of true argmax in predicted ranking (1 = perfect).
    mean_rank: f64,
    /// Mean sigmoid-normalized negative log-likelihood.
    nll: f64,
}

impl Metrics {
    /// Evaluate a predictor closure on the dataset.
    ///
    /// `predict` writes the predicted D-dim velocity into `out` for each input.
    fn evaluate<F>(ds: &Dataset, mut predict: F) -> Self
    where
        F: FnMut(usize, &mut [f32; D]),
    {
        let n = ds.n();
        let mut sum_sq_err = 0.0f64;
        let mut top1_hits = 0u64;
        let mut sum_rank = 0u64;
        let mut sum_nll = 0.0f64;
        let mut pred = [0.0f32; D];
        let mut true_v = [0.0f32; D];
        // Reusable ranking buffer: (value, original_index).
        let mut ranked: [(f32, usize); D] = [(0.0, 0); D];

        for i in 0..n {
            predict(i, &mut pred);
            true_v.copy_from_slice(ds.y_row(i));

            // MSE.
            let mut sq_err = 0.0f64;
            for k in 0..D {
                let d = (pred[k] - true_v[k]) as f64;
                sq_err += d * d;
            }
            sum_sq_err += sq_err / (D as f64);

            // Argmax of true and predicted.
            let mut true_argmax = 0usize;
            let mut true_max = f32::NEG_INFINITY;
            for (k, true_vk) in true_v.iter().enumerate().take(D) {
                if *true_vk > true_max {
                    true_max = *true_vk;
                    true_argmax = k;
                }
            }
            let mut pred_argmax = 0usize;
            let mut pred_max = f32::NEG_INFINITY;
            for (k, pred_k) in pred.iter().enumerate().take(D) {
                if *pred_k > pred_max {
                    pred_max = *pred_k;
                    pred_argmax = k;
                }
            }
            if pred_argmax == true_argmax {
                top1_hits += 1;
            }

            // Mean rank: sort predicted descending by value, find position of
            // the true argmax.
            for k in 0..D {
                ranked[k] = (pred[k], k);
            }
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
            let rank_of_true = ranked
                .iter()
                .position(|(_, idx)| *idx == true_argmax)
                .map_or(D, |p| p + 1);
            sum_rank += rank_of_true as u64;

            // NLL: sigmoid-normalized categorical.
            // p_k = sigmoid(pred_k) / Σ sigmoid(pred_j).
            // NLL = -log(p_true_argmax).
            let mut sig_sum = 0.0f32;
            let mut sig_vals = [0.0f32; D];
            for k in 0..D {
                let s = sigmoid(pred[k]);
                sig_vals[k] = s;
                sig_sum += s;
            }
            let p_true = if sig_sum > 1e-30 {
                sig_vals[true_argmax] / sig_sum
            } else {
                1e-30f32
            };
            sum_nll += -(p_true.max(1e-30).ln()) as f64;
        }

        let nf = n as f64;
        Self {
            mse: sum_sq_err / nf,
            top1: top1_hits as f64 / nf,
            mean_rank: sum_rank as f64 / nf,
            nll: sum_nll / nf,
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ── Competitor (c): target-trained-from-scratch via per-row ridge solve ───

/// Fit a `D×D` linear model `W_approx` from N pairs via per-row least-squares.
///
/// For each output dim k: solve
///   `(XᵀX + λI) w_k = Xᵀ y_k`
/// where X is `(N, D)`, y_k is the k-th output column.
///
/// Returns the row-major `W_approx` (D×D). Uses `ridge_solve_direct_f32` per
/// row — the same closed-form primitive as the ensemble.
fn fit_linear_from_scratch(ds: &Dataset, lambda: f32) -> [f32; D * D] {
    // Build XᵀX (D×D) and per-row Xᵀ y (D-dim each).
    let mut xtx = [0.0f32; D * D];
    let mut w_out = [0.0f32; D * D];
    let n = ds.n();

    // XᵀX = Σ_n x_n x_nᵀ.
    for i in 0..n {
        let xr = ds.x_row(i);
        for k in 0..D {
            for j in 0..D {
                xtx[k * D + j] += xr[k] * xr[j];
            }
        }
    }
    // Normalize by N (matches the ensemble's empirical-expectation convention).
    let inv_n = 1.0f32 / (n as f32);
    for v in xtx.iter_mut() {
        *v *= inv_n;
    }

    // Per-row: build Xᵀ y_k, solve for w_k.
    // Reusable scratch (D-sized).
    let mut gram_reg = [0.0f32; D * D];
    let mut chol = [0.0f32; D * D];
    let mut z_solve = [0.0f32; D];
    let mut rhs = [0.0f32; D];
    let mut w_k = [0.0f32; D];

    for k in 0..D {
        // rhs[j] = (1/N) Σ_n x_n[j] · y_n[k]
        for (j, rhs_j) in rhs.iter_mut().enumerate().take(D) {
            let mut acc = 0.0f32;
            for i in 0..n {
                acc += ds.x_row(i)[j] * ds.y_row(i)[k];
            }
            *rhs_j = acc * inv_n;
        }
        // gram_reg = XᵀX + λI.
        gram_reg.copy_from_slice(&xtx);
        for d in 0..D {
            gram_reg[d * D + d] += lambda;
        }
        // Solve.
        ridge_solve_direct_f32(
            &mut w_k,  // w_t (length D = d_h × n_out with n_out=1)
            &mut chol, // L scratch
            &mut z_solve,
            &gram_reg,
            &rhs,
            D,
            1,
        );
        // Stash row k.
        for j in 0..D {
            w_out[k * D + j] = w_k[j];
        }
    }
    w_out
}

// ── Gate result ───────────────────────────────────────────────────────────

struct GateResult {
    name: String,
    passed: bool,
    detail: String,
}

impl GateResult {
    fn pass(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: true,
            detail: detail.into(),
        }
    }
    fn fail(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            detail: detail.into(),
        }
    }
}

// ── PoC core: one regime (related or unrelated sources) ───────────────────

struct RegimeResult {
    regime_name: &'static str,
    /// (a) single-best.
    a: Metrics,
    /// (b) cross-domain ensemble.
    b: Metrics,
    /// (c) target-trained-from-scratch.
    c: Metrics,
    /// Solved η from the ensemble fit (for diagnostic printout).
    eta: [f32; N_SOURCES],
    /// Which source index was the single-best.
    best_source_idx: usize,
    /// Per-source test MSE (for diagnostic: how bad were the individual sources?).
    per_source_mse: [f64; N_SOURCES],
}

/// Build the sources for a regime.
///
/// - `related = true`:  `W_i = W* + Δ_i` with `Δ_i ~ U(-σ_bias, +σ_bias)`.
/// - `related = false`: `W_i` independent random `~ U(-0.5, +0.5)`.
fn build_sources(rng: &mut Lcg, w_star: &[f32; D * D], related: bool) -> [LinearFieldW; N_SOURCES] {
    let mut sources = [
        LinearFieldW {
            w: [0.0; D * D],
            id: 1,
        },
        LinearFieldW {
            w: [0.0; D * D],
            id: 2,
        },
        LinearFieldW {
            w: [0.0; D * D],
            id: 3,
        },
    ];
    for (i, src) in sources.iter_mut().enumerate() {
        if related {
            // W_i = W* + Δ_i.
            for (j, src_w) in src.w.iter_mut().enumerate().take(D * D) {
                *src_w = w_star[j] + rng.next_signed(SIGMA_BIAS);
            }
        } else {
            // W_i independent random.
            for src_w in src.w.iter_mut().take(D * D) {
                *src_w = rng.next_signed(0.5);
            }
        }
        // Distinct IDs 1..=3 for the Gram.
        src.id = (i as u64) + 1;
    }
    sources
}

fn run_regime(regime_name: &'static str, related: bool) -> RegimeResult {
    let mut rng = Lcg::new(MASTER_SEED.wrapping_add(if related { 1 } else { 2 }));

    // Target: W* ~ U(-0.5, +0.5).
    let w_star = gen_random_matrix(&mut rng, 0.5);

    // Sources.
    let sources = build_sources(&mut rng, &w_star, related);

    // Train + test datasets.
    let train = gen_pairs(&mut rng, &w_star, N_TRAIN, SIGMA_NOISE);
    let test = gen_pairs(&mut rng, &w_star, N_TEST, SIGMA_NOISE);

    // ── Competitor (a): single-best source ────────────────────────────────
    // Pick the source with lowest train MSE; report its test metrics.
    let mut per_source_train_mse = [0.0f64; N_SOURCES];
    let mut per_source_test_mse = [0.0f64; N_SOURCES];
    for (i, src) in sources.iter().enumerate() {
        let src_copy = *src;
        let train_m = Metrics::evaluate(&train, |n, out: &mut [f32; D]| {
            src_copy.eval(train.x_row(n), out);
        });
        let test_m = Metrics::evaluate(&test, |n, out: &mut [f32; D]| {
            src_copy.eval(test.x_row(n), out);
        });
        per_source_train_mse[i] = train_m.mse;
        per_source_test_mse[i] = test_m.mse;
    }
    let best_source_idx = (0..N_SOURCES)
        .min_by(|&a, &b| {
            katgpt_core::float_order::cmp_for_min_f64(per_source_train_mse[a], per_source_train_mse[b])
        })
        .unwrap();
    let best_src = sources[best_source_idx];
    let a = Metrics::evaluate(&test, |n, out: &mut [f32; D]| {
        best_src.eval(test.x_row(n), out);
    });

    // ── Competitor (b): cross-domain ensemble ─────────────────────────────
    // Build the [F; P] array of closure-fields. Each closure captures a copy
    // of the LinearFieldW (65 floats — cheap). All three closures share the
    // same anonymous type because they all come from `make_field_closure`
    // (inline closures would each get a unique type, breaking `[F; P]`).
    let fields = [
        make_field_closure(sources[0]),
        make_field_closure(sources[1]),
        make_field_closure(sources[2]),
    ];
    let mut ensemble = VelocityFieldEnsemble::<_, N_SOURCES, D>::new(fields);
    let mut fit_scratch = EnsembleFitScratch::<N_SOURCES, D>::new();

    // Collect train refs (one-time allocation outside the hot path — fit_into
    // itself is zero-alloc on the pair loop).
    let x_refs: Vec<&[f32]> = (0..N_TRAIN).map(|i| train.x_row(i)).collect();
    let y_refs: Vec<&[f32]> = (0..N_TRAIN).map(|i| train.y_row(i)).collect();
    ensemble.fit_into(&x_refs, &y_refs, LAMBDA_ENSEMBLE, &mut fit_scratch);
    let eta = *ensemble.eta();

    let mut eval_scratch = [0.0f32; D];
    let b = Metrics::evaluate(&test, |n, out: &mut [f32; D]| {
        ensemble.eval_into(test.x_row(n), out, &mut eval_scratch);
    });

    // ── Competitor (c): target-trained-from-scratch ───────────────────────
    let w_approx = fit_linear_from_scratch(&train, LAMBDA_FROM_SCRATCH);
    let c_field = LinearFieldW {
        w: w_approx,
        id: 99,
    };
    let c = Metrics::evaluate(&test, |n, out: &mut [f32; D]| {
        c_field.eval(test.x_row(n), out);
    });

    RegimeResult {
        regime_name,
        a,
        b,
        c,
        eta,
        best_source_idx,
        per_source_mse: per_source_test_mse,
    }
}

/// Helper that returns a closure-field of a single anonymous type, so that all
/// three sources can be stored in `[F; N_SOURCES]`. Inline closures each get
/// their own anonymous type, which would break the array.
fn make_field_closure(field: LinearFieldW) -> ClosureField<D, impl Fn(&[f32], &mut [f32; D])> {
    make_linear_closure_field(field)
}

// ── Verdict printing ───────────────────────────────────────────────────────

fn print_metrics_row(label: &str, m: &Metrics) {
    println!(
        "  {:<32}  MSE={:>10.5}  top1={:>5.3}  rank={:>5.2}  NLL={:>6.3}",
        label, m.mse, m.top1, m.mean_rank, m.nll
    );
}

fn print_regime(r: &RegimeResult) {
    println!("\n=== Regime: {} ===", r.regime_name);
    println!(
        "  D={D}, N_sources={N_SOURCES}, N_train={N_TRAIN}, N_test={N_TEST}, σ_bias={SIGMA_BIAS}, σ_noise={SIGMA_NOISE}"
    );
    println!(
        "  per-source test MSE: [{:.5}, {:.5}, {:.5}]  (best={} → competitor a)",
        r.per_source_mse[0], r.per_source_mse[1], r.per_source_mse[2], r.best_source_idx
    );
    println!(
        "  ensemble η = [{:+.4}, {:+.4}, {:+.4}]",
        r.eta[0], r.eta[1], r.eta[2]
    );
    println!();
    println!("  Competitor metrics (held-out test set):");
    print_metrics_row("(a) single-best source", &r.a);
    print_metrics_row("(b) cross-domain ensemble", &r.b);
    print_metrics_row("(c) from-scratch (target)", &r.c);
}

/// Evaluate G2 for a regime. PASS = (b) beats (a) on ≥ 2 of 3 primary metrics.
fn gate_g2_for_regime(r: &RegimeResult) -> GateResult {
    // Primary metrics: MSE (lower better), top-1 (higher better), mean-rank
    // (lower better). (b) beats (a) means:
    //   b.mse < a.mse
    //   b.top1 > a.top1
    //   b.mean_rank < a.mean_rank
    let mse_b_wins = r.b.mse < r.a.mse;
    let top1_b_wins = r.b.top1 > r.a.top1;
    let rank_b_wins = r.b.mean_rank < r.a.mean_rank;
    let wins = [mse_b_wins, top1_b_wins, rank_b_wins]
        .iter()
        .filter(|&&w| w)
        .count();

    let passed = wins >= 2;
    let detail = format!(
        "ensemble vs single-best: MSE {:+.5} vs {:+.5} ({}, Δ={:.2e}); \
         top1 {:.3} vs {:.3} ({}); rank {:.2} vs {:.2} ({}); \
         wins {}/3 (gate ≥ 2). η=[{:+.3}, {:+.3}, {:+.3}]",
        r.b.mse,
        r.a.mse,
        if mse_b_wins { "b wins" } else { "a wins" },
        r.a.mse - r.b.mse,
        r.b.top1,
        r.a.top1,
        if top1_b_wins { "b wins" } else { "a wins" },
        r.b.mean_rank,
        r.a.mean_rank,
        if rank_b_wins { "b wins" } else { "a wins" },
        wins,
        r.eta[0],
        r.eta[1],
        r.eta[2],
    );
    if passed {
        GateResult::pass(format!("G2 ({})", r.regime_name), detail)
    } else {
        GateResult::fail(format!("G2 ({})", r.regime_name), detail)
    }
}

// ── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("==============================================================");
    println!("  Plan 376 Phase 2 — Velocity-Field Ensemble Cross-Domain PoC");
    println!("  (defend-wrong per AGENTS.md §3.6)");
    println!("==============================================================");

    let related = run_regime("Regime 1: related sources (W_i = W* + Δ_i)", true);
    let unrelated = run_regime("Regime 2: unrelated sources (W_i independent)", false);

    print_regime(&related);
    print_regime(&unrelated);

    // G2 verdict for each regime.
    let g2_related = gate_g2_for_regime(&related);
    let g2_unrelated = gate_g2_for_regime(&unrelated);

    println!("\n=== G2 Verdicts ===");
    let gates = [g2_related, g2_unrelated];
    for g in &gates {
        let status = if g.passed { "PASS" } else { "FAIL" };
        println!("[{status}] {}: {}", g.name, g.detail);
    }

    println!();
    println!("=== Honest interpretation ===");
    println!("  Regime 1 (related sources) is the paper's claim regime: sources");
    println!("  share structure with the target (F-MNIST → MNIST analog).");
    println!("  PASS here supports the cross-domain composition claim.");
    println!();
    println!("  Regime 2 (unrelated sources) is the null regime: sources have");
    println!("  no structural relation. FAIL here is EXPECTED and is NOT a");
    println!("  refutation — it confirms the claim is conditional on relatedness.");
    println!();
    println!("  Competitor (c) (from-scratch) is the reference upper bound: it");
    println!("  uses N_train target-only pairs with the same closed-form math.");
    println!("  (b) approaching (c) is the realistic ceiling for cross-domain.");
    println!();

    // Overall G2 verdict for promotion: PASS only if Regime 1 passes.
    // (Regime 2 is informational; the paper makes no claim for unrelated sources.)
    let regime1_pass = gates[0].passed;
    if regime1_pass {
        println!("=== G2 OVERALL: PASS (Regime 1 supports the cross-domain claim) ===");
        println!("    Eligible to proceed to Phase 3 promotion decision (G1+G3+G4).");
        std::process::exit(0);
    } else {
        println!("=== G2 OVERALL: FAIL (Regime 1 did NOT beat single-best) ===");
        println!("    Cross-domain quality claim is REFUTED by this PoC.");
        println!("    Per Plan 376 T2.4: keep opt-in, record raw numbers in");
        println!("    .benchmarks/376_*.md, downgrade quality claim to .issues/ follow-up.");
        println!("    Architectural coverage (Phase 1) stands regardless.");
        std::process::exit(1);
    }
}
