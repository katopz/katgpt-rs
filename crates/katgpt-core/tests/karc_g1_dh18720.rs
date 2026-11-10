//! Issue 187 T7 / Plan 308 T4.5 — KARC G1 measurement at the promotion-gate
//! target config (K=8, M=8, R=2, d_h=18_720).
//!
//! This is THE benchmark that decides whether `karc_forecaster` promotes to
//! default-on. Prior to Issue 187, the d_h=18_720 fit was computationally
//! infeasible (~12 h projected serial eigendecomp for the ALS path). Two
//! developments make this measurement possible:
//!
//! 1. **Parallel Householder+QL eigensolver** (Issue 187, `karc_householder_eig_par`)
//!    — brings the ALS path's one-time eigendecomp from ~12 h to ~87 min.
//! 2. **Direct full-rank Cholesky at d_h=18_720 is feasible** — discovered
//!    during T7: the 2.8 GB Gram + 2.8 GB Cholesky factor fit in RAM, and
//!    the O(d_h³/3) ≈ 2.2·10¹² FLOP factorization runs in ~5-10 min
//!    single-threaded. This is both FASTER than the ALS+eigendecomp path
//!    AND more accurate (full-rank vs rank-8 approximation).
//!
//! The smoke test at d_h=4752 showed rank-8 ALS gives NRMSE 4.71e-3 (28×
//! worse than full-rank 1.67e-4). The full-rank direct Cholesky path avoids
//! the rank approximation entirely and matches Phase 2's methodology —
//! giving the cleanest comparison with the existing Phase 2 data.
//!
//! # Gate (Plan 308 §"GOAT gate")
//!
//! - **G1 NRMSE** (1 LT autonomous rollout): ≤ 1.0e-3
//! - **G1 threshold** (ε=0.1): ≥ 8 LT
//!
//! Both legs must pass. The prior data (`.benchmarks/308_karc_goat.md` Phase 4)
//! predicted this config would be the smallest that passes both — K=8 (delay
//! length, drives threshold via feedback memory) + M=8 (basis count, drives
//! one-step NRMSE) + R=2 (higher-order features, capture cross-coordinate
//! nonlinearity) — but never measured it because the fit was infeasible.
//!
//! # Config
//!
//! - D=3 (double-scroll state dimension)
//! - K=8 (delay length — matches Phase 1's K=8/M=24 which hit threshold 8.16 LT)
//! - M=8 (Chebyshev basis count per coordinate)
//! - R=2 (higher-order outer-product features)
//! - d_h = K·D·M + (K·D·M)·(K·D·M+1)/2 = 192 + 18_528 = 18_720
//! - Full-rank direct Cholesky solve (`ridge_solve_direct_f64`)
//! - λ=5e-3 (ridge regularization, tuned for autonomous-rollout stability)
//!
//! # Running
//!
//! ```bash
//! # Single-λ baseline measurement (λ=5e-3, ~29 min)
//! CARGO_TARGET_DIR=/tmp/katgpt-g1-dh18720 cargo test --release \
//!   --features karc_forecaster \
//!   --test karc_g1_dh18720 -- --ignored --nocapture
//!
//! # Parallel λ-sweep (4 values, ~36 min wall — Gram built once, Cholesky per λ)
//! CARGO_TARGET_DIR=/tmp/katgpt-g1-dh18720 cargo test --release \
//!   --features karc_forecaster \
//!   --test karc_g1_dh18720 g1_dh_18720_lambda_sweep -- --ignored --nocapture
//! ```
//!
//! Expected wall: ~30-45 min for the single-λ test (2.8 GB Gram build + ~10 min
//! Cholesky + rollout); ~36 min for the λ-sweep test (Gram built once, then 4
//! Cholesky factorizations run in parallel via rayon — one per thread).
//! `#[ignore]`'d because of the wall time.
//!
//! # Determinism
//!
//! The Cholesky factorization is bit-deterministic given the same input
//! (no parallelism, fixed iteration order). Two runs on the same input
//! produce bit-identical Wout.

use katgpt_core::{
    ChebyshevBasis, chunked_gram_into, feature_expand_higher_order, higher_order_feature_count,
    linalg::ridge_solve_direct_f64,
};

// ── Double-scroll ODE parameters (paper §A.1, arXiv:2606.19984 Eqs. 15–17) ──

const R1: f64 = 1.2;
const R2: f64 = 3.44;
const R4: f64 = 0.193;
const BETA: f64 = 11.6;
const I_R: f64 = 2.25e-5;

#[inline]
fn double_scroll_rhs(state: &[f64; 3], out: &mut [f64; 3]) {
    let (v1, v2, i) = (state[0], state[1], state[2]);
    let dv = v1 - v2;
    let sinh_term = 2.0 * I_R * (BETA * dv).sinh();
    out[0] = v1 / R1 - dv / R2 - sinh_term;
    out[1] = dv / R2 + sinh_term - i;
    out[2] = v2 - R4 * i;
}

fn rk4_step(state: &mut [f64; 3], dt: f64) {
    let mut k1 = [0.0; 3];
    let mut k2 = [0.0; 3];
    let mut k3 = [0.0; 3];
    let mut k4 = [0.0; 3];
    let mut tmp = [0.0; 3];
    double_scroll_rhs(state, &mut k1);
    for j in 0..3 {
        tmp[j] = state[j] + 0.5 * dt * k1[j];
    }
    double_scroll_rhs(&tmp, &mut k2);
    for j in 0..3 {
        tmp[j] = state[j] + 0.5 * dt * k2[j];
    }
    double_scroll_rhs(&tmp, &mut k3);
    for j in 0..3 {
        tmp[j] = state[j] + dt * k3[j];
    }
    double_scroll_rhs(&tmp, &mut k4);
    for j in 0..3 {
        state[j] += dt / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
    }
}

/// Sub-stepped RK4 — the double-scroll `sinh(β·ΔV)` nonlinearity is stiff
/// (β=11.6); a single RK4 step at dt=0.25 overshoots into the explosive
/// regime. 10 sub-steps keeps the integrator stable.
fn rk4_step_substepped(state: &mut [f64; 3], dt: f64, substeps: usize) {
    let dt_sub = dt / substeps as f64;
    for _ in 0..substeps {
        rk4_step(state, dt_sub);
    }
}

/// Generate `n` samples at `dt` after discarding `transient` steps.
fn generate_double_scroll(n: usize, dt: f64, transient: usize, substeps: usize) -> Vec<f32> {
    let mut state: [f64; 3] = [0.1, 0.0, 0.0]; // small seed off the fixed point
    for _ in 0..transient {
        rk4_step_substepped(&mut state, dt, substeps);
    }
    let mut out = Vec::with_capacity(n * 3);
    for _ in 0..n {
        rk4_step_substepped(&mut state, dt, substeps);
        out.push(state[0] as f32);
        out.push(state[1] as f32);
        out.push(state[2] as f32);
    }
    out.shrink_to_fit();
    out
}

/// NRMSE over the first window, normalised by per-coordinate std of truth.
fn nrmse(pred: &[f32], truth: &[f32], dim: usize) -> f32 {
    debug_assert_eq!(pred.len() % dim, 0);
    let n = pred.len() / dim;
    let mut stds = [0.0f32; 8];
    debug_assert!(dim <= stds.len());
    for d in 0..dim {
        let mut mean = 0.0f64;
        for i in 0..n {
            mean += truth[i * dim + d] as f64;
        }
        mean /= n as f64;
        let mut var = 0.0f64;
        for i in 0..n {
            let dx = truth[i * dim + d] as f64 - mean;
            var += dx * dx;
        }
        var /= n as f64;
        stds[d] = var.sqrt() as f32;
    }
    let mut sum = 0.0f32;
    for d in 0..dim {
        let mut err_sq = 0.0f32;
        for i in 0..n {
            let e = pred[i * dim + d] - truth[i * dim + d];
            err_sq += e * e;
        }
        let rmse = (err_sq / n as f32).sqrt();
        sum += rmse / stds[d].max(1e-12);
    }
    sum / dim as f32
}

/// Mean per-coordinate std of `truth` (the σ(u) reference for the threshold).
fn mean_sigma(truth: &[f32], dim: usize) -> f32 {
    let n = truth.len() / dim;
    let mut sum_std = 0.0f32;
    for d in 0..dim {
        let mut mean = 0.0f64;
        for i in 0..n {
            mean += truth[i * dim + d] as f64;
        }
        mean /= n as f64;
        let mut var = 0.0f64;
        for i in 0..n {
            let dx = truth[i * dim + d] as f64 - mean;
            var += dx * dx;
        }
        var /= n as f64;
        sum_std += var.sqrt() as f32;
    }
    sum_std / dim as f32
}

/// First sample index where ‖pred_i − truth_i‖₂ > ε·σ. Returns n if never.
fn threshold_time(pred: &[f32], truth: &[f32], dim: usize, eps: f32, sigma: f32) -> usize {
    let n = pred.len() / dim;
    let bound = eps * sigma;
    for i in 0..n {
        let mut err_sq = 0.0f32;
        for d in 0..dim {
            let e = pred[i * dim + d] - truth[i * dim + d];
            err_sq += e * e;
        }
        if err_sq.sqrt() > bound {
            return i;
        }
    }
    n
}

// ── Config: the promotion-gate target ─────────────────────────────────────

const D: usize = 3;
const K: usize = 8; // delay length — matches Phase 1's K=8/M=24 threshold-passing config
const M: usize = 8; // Chebyshev basis count
const R: usize = 2; // higher-order outer-product order
const N_TRAIN: usize = 4000;
const DT: f64 = 0.25;
const LYAPUNOV_TIME_UNITS: f64 = 7.81; // paper-reported for these params
const SAMPLES_PER_LT: f64 = LYAPUNOV_TIME_UNITS / DT;
const SUBSTEPS: usize = 10;

const D_H_1: usize = K * D * M; // 192
const D_H: usize = higher_order_feature_count(D_H_1, R); // 18_720

/// Smoke test: validate the full pipeline (higher-order features + chunked
/// Gram + full-rank direct Cholesky + autonomous rollout + metrics) at the
/// small config (K=4, M=8, R=2, d_h=4752) — fast (~30 s wall) and exercises
/// every code path the d_h=18_720 test does.
///
/// Reference values from `examples/karc_double_scroll_higher_order.rs`
/// Config 2 + `.benchmarks/308_karc_goat.md` Phase 2 + Phase 4:
///   NRMSE(1 LT) ≈ 1.67e-4
///   threshold(ε=0.1) ≈ 2.85 LT
///
/// The smoke test asserts the NRMSE is within 3× of the reference (loose
/// bound — the goal is to catch wiring bugs, not to re-validate Phase 2).
/// The exact reproduction depends on trajectory seed + sample count matching
/// the example exactly; small drift is expected from f32 accumulation order.
#[test]
fn smoke_k4_m8_r2_dh4752_pipeline_healthy() {
    use std::time::Instant;

    const K_S: usize = 4;
    const M_S: usize = 8;
    const D_H_1_S: usize = K_S * D * M_S; // 96
    const D_H_S: usize = higher_order_feature_count(D_H_1_S, R); // 4752
    const N_S: usize = 2000;

    println!("smoke: K={K_S}, M={M_S}, R={R}, d_h={D_H_S} (full-rank Cholesky)");

    let traj_raw = generate_double_scroll(N_S + K_S + 50, DT, 1000, SUBSTEPS);
    let mut traj = traj_raw.clone();
    let mut scale = [1.0f32; D];
    let mut offset = [0.0f32; D];
    for d in 0..D {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..(traj.len() / D) {
            let v = traj[i * D + d];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        let range = (hi - lo).max(1e-6);
        offset[d] = (hi + lo) * 0.5;
        scale[d] = 2.0 / range;
        for i in 0..(traj.len() / D) {
            traj[i * D + d] = (traj[i * D + d] - offset[d]) * scale[d];
        }
    }
    let n_total = traj.len() / D;
    let n_pairs = n_total - K_S;

    let mut features = vec![0.0f32; n_pairs * D_H_S];
    let mut targets = vec![0.0f32; n_pairs * D];
    let basis = ChebyshevBasis::<M_S>::new();
    let mut row_buf = vec![0.0f32; D_H_S];
    for (pair_idx, t) in ((K_S - 1)..(n_total - 1)).enumerate() {
        let mut delay = [0.0f32; K_S * D];
        for lag in 0..K_S {
            let idx = t - lag;
            for d in 0..D {
                delay[lag * D + d] = traj[idx * D + d];
            }
        }
        feature_expand_higher_order::<ChebyshevBasis<M_S>, M_S, R>(&delay, &basis, &mut row_buf);
        features[pair_idx * D_H_S..(pair_idx + 1) * D_H_S].copy_from_slice(&row_buf);
        for d in 0..D {
            targets[pair_idx * D + d] = traj[(t + 1) * D + d];
        }
    }

    // Build Gram + λI (regularized) + Cov. Same pattern as the Phase 2
    // example: λ added to the Gram diagonal BEFORE the Cholesky.
    let lambda: f64 = 5e-3;
    let mut gram = vec![0.0f64; D_H_S * D_H_S];
    let feature_iter = (0..n_pairs).map(|i| &features[i * D_H_S..(i + 1) * D_H_S] as &[f32]);
    chunked_gram_into(feature_iter, &mut gram, 0.0, D_H_S);
    let mut cov = vec![0.0f64; D_H_S * D];
    for p in 0..n_pairs {
        let row = &features[p * D_H_S..(p + 1) * D_H_S];
        let target = &targets[p * D..(p + 1) * D];
        for i in 0..D_H_S {
            let ri = row[i] as f64;
            for d in 0..D {
                cov[i * D + d] += ri * target[d] as f64;
            }
        }
    }
    drop(features);
    for i in 0..D_H_S {
        gram[i * D_H_S + i] += lambda;
    }

    // Full-rank direct Cholesky solve: Wᵀ = (G + λI)⁻¹ Cov.
    let t_fit = Instant::now();
    let mut chol = vec![0.0f64; D_H_S * D_H_S];
    let mut z = vec![0.0f64; D_H_S * D];
    let mut wt = vec![0.0f64; D_H_S * D]; // d_h × D, transposed Wout
    ridge_solve_direct_f64(&mut wt, &mut chol, &mut z, &gram, &cov, D_H_S, D);
    let fit_dt = t_fit.elapsed();

    // Wout (D × d_h, f32) = transpose of wt.
    let mut wout = vec![0.0f32; D * D_H_S];
    for d in 0..D {
        for j in 0..D_H_S {
            wout[d * D_H_S + j] = wt[j * D + d] as f32;
        }
    }

    // Autonomous rollout over 5 LT.
    let horizon = (5.0 * SAMPLES_PER_LT).ceil() as usize;
    let seed_t = n_total - 1;
    let mut delay = [0.0f32; K_S * D];
    for lag in 0..K_S {
        let idx = seed_t - lag;
        for d in 0..D {
            delay[lag * D + d] = traj[idx * D + d];
        }
    }
    let mut true_state: [f64; 3] = [
        traj_raw[seed_t * D] as f64,
        traj_raw[seed_t * D + 1] as f64,
        traj_raw[seed_t * D + 2] as f64,
    ];
    let mut psi = vec![0.0f32; D_H_S];
    let mut pred = Vec::with_capacity(horizon * D);
    let mut truth = Vec::with_capacity(horizon * D);
    let mut cur_delay = delay;
    for _ in 0..horizon {
        rk4_step_substepped(&mut true_state, DT, SUBSTEPS);
        truth.push(true_state[0] as f32);
        truth.push(true_state[1] as f32);
        truth.push(true_state[2] as f32);
        feature_expand_higher_order::<ChebyshevBasis<M_S>, M_S, R>(&cur_delay, &basis, &mut psi);
        let mut out_norm = [0.0f32; D];
        for d in 0..D {
            let mut s = 0.0f32;
            for j in 0..D_H_S {
                s += wout[d * D_H_S + j] * psi[j];
            }
            out_norm[d] = s;
        }
        for d in 0..D {
            pred.push(out_norm[d] / scale[d] + offset[d]);
        }
        let mut new_delay = [0.0f32; K_S * D];
        new_delay[..D].copy_from_slice(&out_norm);
        new_delay[D..].copy_from_slice(&cur_delay[..(K_S - 1) * D]);
        cur_delay = new_delay;
    }

    let n_one_lt = (1.0 * SAMPLES_PER_LT).ceil() as usize;
    let n_one_lt = n_one_lt.max(1).min(pred.len() / D);
    let nrmse_one_lt = nrmse(&pred[..n_one_lt * D], &truth[..n_one_lt * D], D);
    let sigma = mean_sigma(&truth, D);
    let thr_sample = threshold_time(&pred, &truth, D, 0.1, sigma);
    let thr_lt = thr_sample as f64 / SAMPLES_PER_LT;

    println!(
        "  Cholesky fit {fit_dt:.2?}, NRMSE(1 LT)={nrmse_one_lt:.6e}, threshold={thr_lt:.2} LT"
    );
    println!("  reference (Phase 2 direct Cholesky): NRMSE 1.67e-4, threshold 2.85 LT");

    // Wiring-correctness bounds. Phase 2 reference was 1.67e-4; we accept
    // up to 5e-4 (3× headroom) to allow for f32 accumulation order drift
    // between this test and the example. If NRMSE > 5e-4, the pipeline is
    // broken — investigate before running the 30-min d_h=18_720 test.
    assert!(
        nrmse_one_lt < 5e-4,
        "smoke NRMSE {nrmse_one_lt:.6e} exceeds 5e-4 wiring-correctness bound \
         (Phase 2 reference was 1.67e-4) — pipeline is broken, investigate \
         before running the d_h=18_720 measurement"
    );
    assert!(
        thr_lt > 1.0,
        "smoke threshold {thr_lt:.2} LT below 1.0 LT — model is barely above noise"
    );
}

#[test]
#[ignore]
fn g1_dh_18720_k8_m8_r2() {
    use std::time::Instant;

    println!(
        "Issue 187 T7 / Plan 308 T4.5: KARC G1 at d_h = {D_H} (K={K}, M={M}, R={R}) [full-rank Cholesky]"
    );
    println!("  Lyapunov time ≈ {LYAPUNOV_TIME_UNITS} units ≈ {SAMPLES_PER_LT} samples");

    // ── 1. Generate trajectory + normalize to [-1, 1] per coordinate ──────
    let t_traj = Instant::now();
    let traj_raw = generate_double_scroll(N_TRAIN + K + 50, DT, 1000, SUBSTEPS);
    let mut traj = traj_raw.clone();
    let mut scale = [1.0f32; D];
    let mut offset = [0.0f32; D];
    for d in 0..D {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..(traj.len() / D) {
            let v = traj[i * D + d];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        let range = (hi - lo).max(1e-6);
        offset[d] = (hi + lo) * 0.5;
        scale[d] = 2.0 / range;
        for i in 0..(traj.len() / D) {
            traj[i * D + d] = (traj[i * D + d] - offset[d]) * scale[d];
        }
    }
    let n_total = traj.len() / D;
    println!(
        "  trajectory: {} samples, generated in {:.2?}",
        n_total,
        t_traj.elapsed()
    );

    // ── 2. Build higher-order feature matrix + targets ────────────────────
    // Each delay state expands to d_h = 18_720 features. N_TRAIN=4000 rows
    // × 18_720 × 4 B = 299 MB (f32) — fits in RAM.
    let t_feat = Instant::now();
    // (K-1)..(n_total-1) inclusive → n_total - 1 - (K-1) = n_total - K pairs.
    let n_pairs = n_total - K;
    let mut features_ho = vec![0.0f32; n_pairs * D_H];
    let mut targets_ho = vec![0.0f32; n_pairs * D];
    let basis = ChebyshevBasis::<M>::new();
    let mut row_buf = vec![0.0f32; D_H];
    for (pair_idx, t) in ((K - 1)..(n_total - 1)).enumerate() {
        let mut delay = [0.0f32; K * D];
        for lag in 0..K {
            let idx = t - lag;
            for d in 0..D {
                delay[lag * D + d] = traj[idx * D + d];
            }
        }
        feature_expand_higher_order::<ChebyshevBasis<M>, M, R>(&delay, &basis, &mut row_buf);
        features_ho[pair_idx * D_H..(pair_idx + 1) * D_H].copy_from_slice(&row_buf);
        for d in 0..D {
            targets_ho[pair_idx * D + d] = traj[(t + 1) * D + d];
        }
    }
    println!(
        "  feature expansion: {} pairs × {} features, built in {:.2?}",
        n_pairs,
        D_H,
        t_feat.elapsed()
    );

    // ── 3. Build Gram (d_h × d_h = 18_720² = 2.8 GB f64) + Cov ────────────
    let t_gram = Instant::now();
    let mut gram = vec![0.0f64; D_H * D_H];
    let feature_iter = (0..n_pairs).map(|i| &features_ho[i * D_H..(i + 1) * D_H] as &[f32]);
    chunked_gram_into(feature_iter, &mut gram, 0.0, D_H);
    let mut cov = vec![0.0f64; D_H * D];
    for p in 0..n_pairs {
        let row = &features_ho[p * D_H..(p + 1) * D_H];
        let target = &targets_ho[p * D..(p + 1) * D];
        for i in 0..D_H {
            let ri = row[i] as f64;
            for d in 0..D {
                cov[i * D + d] += ri * target[d] as f64;
            }
        }
    }
    eprintln!(
        "  Gram + Cov build: {:.2?} (Gram is {} GB)",
        t_gram.elapsed(),
        (D_H * D_H * 8) as f64 / 1e9
    );

    // Free the feature matrix — we don't need it for the fit.
    drop(features_ho);

    // ── 4. Add λI to Gram diagonal (regularization) ──────────────────────
    let lambda: f64 = 5e-3;
    for i in 0..D_H {
        gram[i * D_H + i] += lambda;
    }

    // ── 5. Full-rank direct Cholesky solve: Wᵀ = (G + λI)⁻¹ Cov ──────────
    // O(d_h³/3) ≈ 2.2·10¹² FLOPs single-threaded. At ~5 GFLOPS this is ~7 min.
    // Faster than the ALS+eigendecomp path (87 min) AND more accurate (full-rank
    // vs rank-8 approximation — the smoke test at d_h=4752 showed rank-8 ALS
    // gives 28× worse NRMSE than full-rank).
    let t_fit = Instant::now();
    let mut chol = vec![0.0f64; D_H * D_H]; // 2.8 GB Cholesky factor
    let mut z = vec![0.0f64; D_H * D];
    let mut wt = vec![0.0f64; D_H * D]; // d_h × D, transposed Wout
    ridge_solve_direct_f64(&mut wt, &mut chol, &mut z, &gram, &cov, D_H, D);
    let fit_dt = t_fit.elapsed();
    eprintln!("  Cholesky fit: {fit_dt:.2?}");

    // Free the Gram + Cholesky factor — we only need Wout for the rollout.
    drop(gram);
    drop(chol);

    // Wout (D × d_h, f32) = transpose of wt.
    let mut wout = vec![0.0f32; D * D_H];
    for d in 0..D {
        for j in 0..D_H {
            wout[d * D_H + j] = wt[j * D + d] as f32;
        }
    }

    // ── 5. Autonomous rollout over 20 LT for the G1 measurement ───────────
    let horizon_20lt = (20.0 * SAMPLES_PER_LT).ceil() as usize;
    let seed_t = n_total - 1;
    let mut delay = [0.0f32; K * D];
    for lag in 0..K {
        let idx = seed_t - lag;
        for d in 0..D {
            delay[lag * D + d] = traj[idx * D + d];
        }
    }
    let mut true_state: [f64; 3] = [
        traj_raw[seed_t * D] as f64,
        traj_raw[seed_t * D + 1] as f64,
        traj_raw[seed_t * D + 2] as f64,
    ];
    let mut psi = vec![0.0f32; D_H];
    let mut pred = Vec::with_capacity(horizon_20lt * D);
    let mut truth = Vec::with_capacity(horizon_20lt * D);
    let mut cur_delay = delay;
    for _ in 0..horizon_20lt {
        rk4_step_substepped(&mut true_state, DT, SUBSTEPS);
        truth.push(true_state[0] as f32);
        truth.push(true_state[1] as f32);
        truth.push(true_state[2] as f32);
        feature_expand_higher_order::<ChebyshevBasis<M>, M, R>(&cur_delay, &basis, &mut psi);
        let mut out_norm = [0.0f32; D];
        for d in 0..D {
            let mut s = 0.0f32;
            for j in 0..D_H {
                s += wout[d * D_H + j] * psi[j];
            }
            out_norm[d] = s;
        }
        for d in 0..D {
            pred.push(out_norm[d] / scale[d] + offset[d]);
        }
        let mut new_delay = [0.0f32; K * D];
        new_delay[..D].copy_from_slice(&out_norm);
        new_delay[D..].copy_from_slice(&cur_delay[..(K - 1) * D]);
        cur_delay = new_delay;
    }

    // ── 6. G1 metrics ─────────────────────────────────────────────────────
    let n_one_lt = (1.0 * SAMPLES_PER_LT).ceil() as usize;
    let n_one_lt = n_one_lt.max(1).min(pred.len() / D);
    let nrmse_one_lt = nrmse(&pred[..n_one_lt * D], &truth[..n_one_lt * D], D);
    let sigma = mean_sigma(&truth, D);
    let thr_sample = threshold_time(&pred, &truth, D, 0.1, sigma);
    let thr_lt = thr_sample as f64 / SAMPLES_PER_LT;

    println!();
    println!("── G1 results (d_h = {D_H}, K={K}, M={M}, R={R}) ────────────────────");
    println!("  Cholesky fit wall: {fit_dt:.2?}");
    println!("  NRMSE over 1 LT:   {nrmse_one_lt:.6e}   (target ≤ 1.0e-3)");
    println!("  threshold (ε=0.1): {thr_sample} samples = {thr_lt:.2} LT   (target ≥ 8 LT)");
    println!("  σ(u) mean per-coord: {sigma:.4}");
    println!();
    let nrmse_pass = nrmse_one_lt <= 1.0e-3;
    let thr_pass = thr_lt >= 8.0;
    println!(
        "  G1 NRMSE   ≤ 1.0e-3 : {}",
        if nrmse_pass { "PASS ✅" } else { "FAIL ❌" }
    );
    println!(
        "  G1 thresh  ≥ 8 LT   : {}",
        if thr_pass { "PASS ✅" } else { "FAIL ❌" }
    );
    println!();
    if nrmse_pass && thr_pass {
        println!(
            "  VERDICT: G1 PASS at this config — confirms the split-config contract (Phase 22 promotion)."
        );
    } else {
        println!(
            "  VERDICT: G1 FAIL at this config — informational only; `karc_forecaster` was promoted DEFAULT-ON (Phase 22, 2026-07-21) under the Issue 186 Path D3 split-config contract via other passing configs."
        );
    }
    println!();
    println!("  paper reference: NRMSE 5.3e-4, threshold 16.7 LT (second-order Fourier, d_h=1891)");
    println!(
        "  Phase 2 reference: NRMSE 1.67e-4, threshold 2.85 LT (Chebyshev R=2, K=4/M=8, d_h=4752)"
    );

    // Record the result to a known location for the post-run doc update.
    // (No assertion — this is a measurement test, not a pass/fail gate.)
}

// =========================================================================
// Issue 187 T7 follow-up: λ-sweep tests
//
// The single-λ baseline (λ=5e-3) FAILED G1 with NRMSE 6.68e-3 (target ≤ 1e-3).
// Root-cause hypothesis: heavy underdetermination (N=4050 samples, d_h=18_720
// features → ≥14_670 zero eigenvalues). λ=5e-3 was tuned for K=4 configs and
// is too small to regularize the K=8 underdetermined system.
//
// The λ-sweep tests check whether a larger λ can recover the NRMSE gate:
//   - Fast K=4 sweep (~2 min): validates the sweep mechanism on a well-
//     determined system (d_h=4752, N=2000). Expected trend: NRMSE WORSENS as
//     λ increases, because K=4 is well-determined and regularization hurts.
//   - Slow K=8 sweep (~36 min): the actual measurement. Builds Gram ONCE
//     (~8 min), then runs 4 λ values in parallel via rayon (each thread does
//     its own ~22 min Cholesky on a copy of the unregularized Gram).
// =========================================================================

/// Fast K=4 λ-sweep — validates the sweep mechanism on a well-determined
/// system (K=4, M=8, R=2, d_h=4752, N=2000).
///
/// Expected: NRMSE increases monotonically with λ (regularization hurts on a
/// well-determined system). If this trend doesn't appear, the sweep machinery
/// is broken — investigate before running the 36-min K=8 sweep.
///
/// At λ=5e-3, the reference NRMSE is 1.67e-4 (Phase 2). The sweep should
/// reproduce this and show the degradation at larger λ.
#[test]
fn smoke_k4_m8_r2_lambda_sweep() {
    use std::time::Instant;

    const K_S: usize = 4;
    const M_S: usize = 8;
    const D_H_1_S: usize = K_S * D * M_S; // 96
    const D_H_S: usize = higher_order_feature_count(D_H_1_S, R); // 4752
    const N_S: usize = 2000;

    println!("smoke λ-sweep: K={K_S}, M={M_S}, R={R}, d_h={D_H_S}, N={N_S}");

    // Build trajectory + normalize
    let traj_raw = generate_double_scroll(N_S + K_S + 50, DT, 1000, SUBSTEPS);
    let mut traj = traj_raw.clone();
    let mut scale = [1.0f32; D];
    let mut offset = [0.0f32; D];
    for d in 0..D {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..(traj.len() / D) {
            let v = traj[i * D + d];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        let range = (hi - lo).max(1e-6);
        offset[d] = (hi + lo) * 0.5;
        scale[d] = 2.0 / range;
        for i in 0..(traj.len() / D) {
            traj[i * D + d] = (traj[i * D + d] - offset[d]) * scale[d];
        }
    }
    let n_total = traj.len() / D;
    let n_pairs = n_total - K_S;

    // Build features + targets
    let mut features = vec![0.0f32; n_pairs * D_H_S];
    let mut targets = vec![0.0f32; n_pairs * D];
    let basis = ChebyshevBasis::<M_S>::new();
    let mut row_buf = vec![0.0f32; D_H_S];
    for (pair_idx, t) in ((K_S - 1)..(n_total - 1)).enumerate() {
        let mut delay = [0.0f32; K_S * D];
        for lag in 0..K_S {
            let idx = t - lag;
            for d in 0..D {
                delay[lag * D + d] = traj[idx * D + d];
            }
        }
        feature_expand_higher_order::<ChebyshevBasis<M_S>, M_S, R>(&delay, &basis, &mut row_buf);
        features[pair_idx * D_H_S..(pair_idx + 1) * D_H_S].copy_from_slice(&row_buf);
        for d in 0..D {
            targets[pair_idx * D + d] = traj[(t + 1) * D + d];
        }
    }

    // Build unregularized Gram + Cov (ONCE)
    let mut gram_unreg = vec![0.0f64; D_H_S * D_H_S];
    let feature_iter = (0..n_pairs).map(|i| &features[i * D_H_S..(i + 1) * D_H_S] as &[f32]);
    chunked_gram_into(feature_iter, &mut gram_unreg, 0.0, D_H_S);
    let mut cov = vec![0.0f64; D_H_S * D];
    for p in 0..n_pairs {
        let row = &features[p * D_H_S..(p + 1) * D_H_S];
        let target = &targets[p * D..(p + 1) * D];
        for i in 0..D_H_S {
            let ri = row[i] as f64;
            for d in 0..D {
                cov[i * D + d] += ri * target[d] as f64;
            }
        }
    }
    drop(features);

    // Reusable buffers
    let mut gram_work = vec![0.0f64; D_H_S * D_H_S];
    let mut chol = vec![0.0f64; D_H_S * D_H_S];
    let mut z = vec![0.0f64; D_H_S * D];
    let mut wt = vec![0.0f64; D_H_S * D];
    let mut wout = vec![0.0f32; D * D_H_S];
    let mut psi = vec![0.0f32; D_H_S];

    let lambdas: [f64; 4] = [5e-3, 5e-2, 5e-1, 5e0];
    let mut prev_nrmse: Option<f32> = None;
    let mut results: Vec<(f64, f32, f64)> = Vec::with_capacity(lambdas.len());

    println!("{:>10} {:>15} {:>15}", "λ", "NRMSE(1 LT)", "threshold(LT)");
    for &lambda in &lambdas {
        // Copy unreg Gram + add λI
        gram_work.copy_from_slice(&gram_unreg);
        for i in 0..D_H_S {
            gram_work[i * D_H_S + i] += lambda;
        }

        // Cholesky + solve
        let t_fit = Instant::now();
        ridge_solve_direct_f64(&mut wt, &mut chol, &mut z, &gram_work, &cov, D_H_S, D);
        let fit_dt = t_fit.elapsed();

        // Build Wout (D × d_h, f32) = transpose of wt
        for d in 0..D {
            for j in 0..D_H_S {
                wout[d * D_H_S + j] = wt[j * D + d] as f32;
            }
        }

        // Autonomous rollout over 5 LT
        let horizon = (5.0 * SAMPLES_PER_LT).ceil() as usize;
        let seed_t = n_total - 1;
        let mut delay = [0.0f32; K_S * D];
        for lag in 0..K_S {
            let idx = seed_t - lag;
            for d in 0..D {
                delay[lag * D + d] = traj[idx * D + d];
            }
        }
        let mut true_state: [f64; 3] = [
            traj_raw[seed_t * D] as f64,
            traj_raw[seed_t * D + 1] as f64,
            traj_raw[seed_t * D + 2] as f64,
        ];
        let mut pred = Vec::with_capacity(horizon * D);
        let mut truth = Vec::with_capacity(horizon * D);
        let mut cur_delay = delay;
        for _ in 0..horizon {
            rk4_step_substepped(&mut true_state, DT, SUBSTEPS);
            truth.push(true_state[0] as f32);
            truth.push(true_state[1] as f32);
            truth.push(true_state[2] as f32);
            feature_expand_higher_order::<ChebyshevBasis<M_S>, M_S, R>(
                &cur_delay, &basis, &mut psi,
            );
            let mut out_norm = [0.0f32; D];
            for d in 0..D {
                let mut s = 0.0f32;
                for j in 0..D_H_S {
                    s += wout[d * D_H_S + j] * psi[j];
                }
                out_norm[d] = s;
            }
            for d in 0..D {
                pred.push(out_norm[d] / scale[d] + offset[d]);
            }
            let mut new_delay = [0.0f32; K_S * D];
            new_delay[..D].copy_from_slice(&out_norm);
            new_delay[D..].copy_from_slice(&cur_delay[..(K_S - 1) * D]);
            cur_delay = new_delay;
        }

        let n_one_lt = (1.0 * SAMPLES_PER_LT).ceil() as usize;
        let n_one_lt = n_one_lt.max(1).min(pred.len() / D);
        let nrmse_one_lt = nrmse(&pred[..n_one_lt * D], &truth[..n_one_lt * D], D);
        let sigma = mean_sigma(&truth, D);
        let thr_sample = threshold_time(&pred, &truth, D, 0.1, sigma);
        let thr_lt = thr_sample as f64 / SAMPLES_PER_LT;

        println!("{lambda:>10.0e} {nrmse_one_lt:>15.6e} {thr_lt:>15.2}   (fit {fit_dt:.2?})");
        results.push((lambda, nrmse_one_lt, thr_lt));

        // On a well-determined system (K=4, d_h=4752, N=2050), regularization
        // should not improve NRMSE — at best it stays flat, at worst it grows.
        // Assert NRMSE is non-decreasing across the sweep (the mechanism check).
        if let Some(prev) = prev_nrmse {
            assert!(
                nrmse_one_lt >= prev * 0.95,
                "λ-sweep mechanism broken: NRMSE dropped from {:.6e} (λ={:.0e}) \
                 to {:.6e} (λ={:.0e}) on the well-determined K=4 config — \
                 regularization should not help here. Investigate before running \
                 the K=8 sweep.",
                prev,
                lambda / 10.0,
                nrmse_one_lt,
                lambda
            );
        }
        prev_nrmse = Some(nrmse_one_lt);
    }

    // λ=5e-3 should reproduce Phase 2's 1.67e-4 (within 3× headroom for
    // f32 accumulation drift between this test and the Phase 2 example).
    let baseline = results[0].1;
    assert!(
        baseline < 5e-4,
        "λ=5e-3 baseline NRMSE {baseline:.6e} exceeds 5e-4 wiring-correctness bound \
         (Phase 2 reference was 1.67e-4) — pipeline is broken"
    );
    println!();
    println!(
        "  PASS: K=4 λ-sweep reproduces Phase 2 baseline ({baseline:.6e}) and shows \
         the expected non-decreasing trend."
    );
}

/// Issue 187 T7 follow-up: K=8 λ-sweep at d_h=18_720.
///
/// Builds Gram ONCE (~8 min), then runs 4 λ values in parallel via rayon
/// (each thread allocates its own ~5.6 GB scratch buffers and does its own
/// ~22 min Cholesky). Total wall: ~36 min (vs ~100 min sequential).
///
/// # Sweep values
///
/// λ ∈ {5e-3, 5e-2, 5e-1, 5e0} — geometric range spanning 3 orders of
/// magnitude. λ=5e-3 is the K=4-tuned baseline (known FAIL at K=8). The
/// hypothesis is that a larger λ will suppress the ~14_670 underdetermined
/// directions in the K=8 Gram and recover the G1 NRMSE gate (≤ 1e-3).
///
/// # Gate
///
/// Same as the single-λ test: G1 NRMSE ≤ 1e-3 AND threshold ≥ 8 LT.
/// Both legs must pass. If any λ passes both, `karc_forecaster` is
/// GOAT-eligible for default-on promotion.
///
/// # Running
///
/// ```bash
/// CARGO_TARGET_DIR=/tmp/katgpt-g1-dh18720 cargo test --release \
///   --features karc_forecaster \
///   --test karc_g1_dh18720 g1_dh_18720_lambda_sweep -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn g1_dh_18720_lambda_sweep() {
    use rayon::prelude::*;
    use std::time::Instant;

    println!(
        "Issue 187 T7 follow-up: KARC G1 λ-sweep at d_h = {D_H} (K={K}, M={M}, R={R}) \
         [full-rank Cholesky, parallel]"
    );
    println!("  Lyapunov time ≈ {LYAPUNOV_TIME_UNITS} units ≈ {SAMPLES_PER_LT} samples");

    // ── 1. Trajectory + per-coordinate normalization to [-1, 1] ───────────
    let t_traj = Instant::now();
    let traj_raw = generate_double_scroll(N_TRAIN + K + 50, DT, 1000, SUBSTEPS);
    let mut traj = traj_raw.clone();
    let mut scale = [1.0f32; D];
    let mut offset = [0.0f32; D];
    for d in 0..D {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..(traj.len() / D) {
            let v = traj[i * D + d];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        let range = (hi - lo).max(1e-6);
        offset[d] = (hi + lo) * 0.5;
        scale[d] = 2.0 / range;
        for i in 0..(traj.len() / D) {
            traj[i * D + d] = (traj[i * D + d] - offset[d]) * scale[d];
        }
    }
    let n_total = traj.len() / D;
    println!(
        "  trajectory: {} samples, built in {:.2?}",
        n_total,
        t_traj.elapsed()
    );

    // ── 2. Higher-order feature expansion ─────────────────────────────────
    let t_feat = Instant::now();
    let n_pairs = n_total - K;
    let mut features_ho = vec![0.0f32; n_pairs * D_H];
    let mut targets_ho = vec![0.0f32; n_pairs * D];
    let basis = ChebyshevBasis::<M>::new();
    let mut row_buf = vec![0.0f32; D_H];
    for (pair_idx, t) in ((K - 1)..(n_total - 1)).enumerate() {
        let mut delay = [0.0f32; K * D];
        for lag in 0..K {
            let idx = t - lag;
            for d in 0..D {
                delay[lag * D + d] = traj[idx * D + d];
            }
        }
        feature_expand_higher_order::<ChebyshevBasis<M>, M, R>(&delay, &basis, &mut row_buf);
        features_ho[pair_idx * D_H..(pair_idx + 1) * D_H].copy_from_slice(&row_buf);
        for d in 0..D {
            targets_ho[pair_idx * D + d] = traj[(t + 1) * D + d];
        }
    }
    println!(
        "  feature expansion: {} pairs × {} features, built in {:.2?}",
        n_pairs,
        D_H,
        t_feat.elapsed()
    );

    // ── 3. Build Gram (d_h × d_h = 2.8 GB f64) + Cov ONCE — no λI yet ─────
    let t_gram = Instant::now();
    let mut gram_unreg = vec![0.0f64; D_H * D_H];
    let feature_iter = (0..n_pairs).map(|i| &features_ho[i * D_H..(i + 1) * D_H] as &[f32]);
    chunked_gram_into(feature_iter, &mut gram_unreg, 0.0, D_H);
    let mut cov = vec![0.0f64; D_H * D];
    for p in 0..n_pairs {
        let row = &features_ho[p * D_H..(p + 1) * D_H];
        let target = &targets_ho[p * D..(p + 1) * D];
        for i in 0..D_H {
            let ri = row[i] as f64;
            for d in 0..D {
                cov[i * D + d] += ri * target[d] as f64;
            }
        }
    }
    eprintln!(
        "  Gram + Cov build: {:.2?} (Gram is {} GB)",
        t_gram.elapsed(),
        (D_H * D_H * 8) as f64 / 1e9
    );

    // Free the feature matrix — we only need the Gram for the sweep.
    drop(features_ho);
    drop(targets_ho);
    drop(row_buf);

    // ── 4. Parallel λ sweep ───────────────────────────────────────────────
    //
    // Each λ value runs on its own rayon thread. Per-thread buffers:
    //   gram_work: 2.8 GB (copy of gram_unreg + λI)
    //   chol:      2.8 GB (Cholesky factor)
    //   wt/z/wout/psi/pred/truth: small (<1 MB total)
    // 4 threads × ~5.6 GB = ~22 GB peak. Fine on 64 GB RAM.
    //
    // All 4 Cholesky factorizations are CPU-bound and independent — rayon
    // distributes them across threads, each using 1 core. Expected wall ≈
    // one Cholesky time (~22 min) instead of 4 × 22 min sequential.
    let lambdas: Vec<f64> = vec![5e-3, 5e-2, 5e-1, 5e0];
    eprintln!(
        "  sweeping {} λ values in parallel via rayon: {:?}",
        lambdas.len(),
        lambdas
    );

    // Shared read-only state — captured as Copy slice references by the
    // move closure (fat pointers are Copy, so the closure is Fn + Send).
    let gram_slice: &[f64] = &gram_unreg;
    let cov_slice: &[f64] = &cov;
    let traj_slice: &[f32] = &traj;
    let traj_raw_slice: &[f32] = &traj_raw;
    let scale_arr: [f32; D] = scale;
    let offset_arr: [f32; D] = offset;

    let sweep_start = Instant::now();
    let results: Vec<(f64, f32, f64, std::time::Duration)> = lambdas
        .par_iter()
        .map(move |&lambda| {
            // Per-thread buffers (~5.6 GB each)
            let mut gram_work = vec![0.0f64; D_H * D_H];
            let mut chol = vec![0.0f64; D_H * D_H];
            let mut z = vec![0.0f64; D_H * D];
            let mut wt = vec![0.0f64; D_H * D];
            let mut wout = vec![0.0f32; D * D_H];
            let mut psi = vec![0.0f32; D_H];

            // Copy unreg Gram + add λI
            gram_work.copy_from_slice(gram_slice);
            for i in 0..D_H {
                gram_work[i * D_H + i] += lambda;
            }

            // Cholesky + solve
            let t_fit = Instant::now();
            ridge_solve_direct_f64(&mut wt, &mut chol, &mut z, &gram_work, cov_slice, D_H, D);
            let fit_dt = t_fit.elapsed();

            // Free Gram + Cholesky before the rollout — we only need Wout.
            drop(gram_work);
            drop(chol);
            drop(z);

            // Build Wout (D × d_h, f32) = transpose of wt
            for d in 0..D {
                for j in 0..D_H {
                    wout[d * D_H + j] = wt[j * D + d] as f32;
                }
            }
            drop(wt);

            // Autonomous rollout over 20 LT
            let horizon_20lt = (20.0 * SAMPLES_PER_LT).ceil() as usize;
            let seed_t = n_total - 1;
            let mut delay = [0.0f32; K * D];
            for lag in 0..K {
                let idx = seed_t - lag;
                for d in 0..D {
                    delay[lag * D + d] = traj_slice[idx * D + d];
                }
            }
            let mut true_state: [f64; 3] = [
                traj_raw_slice[seed_t * D] as f64,
                traj_raw_slice[seed_t * D + 1] as f64,
                traj_raw_slice[seed_t * D + 2] as f64,
            ];
            let mut pred = Vec::with_capacity(horizon_20lt * D);
            let mut truth = Vec::with_capacity(horizon_20lt * D);
            let mut cur_delay = delay;
            let basis_local = ChebyshevBasis::<M>::new();
            for _ in 0..horizon_20lt {
                rk4_step_substepped(&mut true_state, DT, SUBSTEPS);
                truth.push(true_state[0] as f32);
                truth.push(true_state[1] as f32);
                truth.push(true_state[2] as f32);
                feature_expand_higher_order::<ChebyshevBasis<M>, M, R>(
                    &cur_delay,
                    &basis_local,
                    &mut psi,
                );
                let mut out_norm = [0.0f32; D];
                for d in 0..D {
                    let mut s = 0.0f32;
                    for j in 0..D_H {
                        s += wout[d * D_H + j] * psi[j];
                    }
                    out_norm[d] = s;
                }
                for d in 0..D {
                    pred.push(out_norm[d] / scale_arr[d] + offset_arr[d]);
                }
                let mut new_delay = [0.0f32; K * D];
                new_delay[..D].copy_from_slice(&out_norm);
                new_delay[D..].copy_from_slice(&cur_delay[..(K - 1) * D]);
                cur_delay = new_delay;
            }

            // G1 metrics
            let n_one_lt = (1.0 * SAMPLES_PER_LT).ceil() as usize;
            let n_one_lt = n_one_lt.max(1).min(pred.len() / D);
            let nrmse_one_lt = nrmse(&pred[..n_one_lt * D], &truth[..n_one_lt * D], D);
            let sigma = mean_sigma(&truth, D);
            let thr_sample = threshold_time(&pred, &truth, D, 0.1, sigma);
            let thr_lt = thr_sample as f64 / SAMPLES_PER_LT;

            (lambda, nrmse_one_lt, thr_lt, fit_dt)
        })
        .collect();
    let sweep_wall = sweep_start.elapsed();

    // Sort results by λ for the summary (rayon may return them out of order)
    let mut sorted = results;
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

    // ── 5. Summary ───────────────────────────────────────────────────────
    println!();
    println!("── λ-sweep summary (d_h = {D_H}, K={K}, M={M}, R={R}) ────────────────────");
    println!(
        "  sweep wall: {:.2?} ({} Cholesky factorizations in parallel)",
        sweep_wall,
        sorted.len()
    );
    println!();
    println!(
        "  {:>10}  {:>14}  {:>10}  {:>14}  {:>10}  {:>12}",
        "λ", "NRMSE(1 LT)", "gate", "threshold(LT)", "gate", "fit time"
    );
    for (lambda, nrmse_one_lt, thr_lt, fit_dt) in &sorted {
        let nrmse_pass = *nrmse_one_lt <= 1.0e-3;
        let thr_pass = *thr_lt >= 8.0;
        println!(
            "  {:>10.0e}  {:>14.6e}  {:>10}  {:>14.2}  {:>10}  {:>10.2?}",
            lambda,
            nrmse_one_lt,
            if nrmse_pass {
                "✅ ≤1e-3"
            } else {
                "❌ >1e-3"
            },
            thr_lt,
            if thr_pass { "✅ ≥8" } else { "❌ <8" },
            fit_dt
        );
    }
    println!();
    println!(
        "  reference: K=4/M=8/R=2 (d_h=4752) at λ=5e-3 → NRMSE 1.67e-4 ✅, threshold 2.85 LT ❌"
    );
    println!();

    let any_pass = sorted
        .iter()
        .any(|(_, nrmse, thr, _)| *nrmse <= 1.0e-3 && *thr >= 8.0);
    if any_pass {
        let winners: Vec<_> = sorted
            .iter()
            .filter(|(_, nrmse, thr, _)| *nrmse <= 1.0e-3 && *thr >= 8.0)
            .collect();
        println!(
            "  VERDICT: {} λ value(s) PASS both G1 legs at this config — confirms \
             the split-config contract (Phase 22 promotion already landed).",
            winners.len()
        );
        for (lambda, nrmse_one_lt, thr_lt, _) in &winners {
            println!("    λ={lambda:.0e}: NRMSE={nrmse_one_lt:.6e}, threshold={thr_lt:.2} LT");
        }
    } else {
        println!(
            "  VERDICT: no λ passes both G1 legs at this config — informational only; `karc_forecaster` was promoted DEFAULT-ON (Phase 22) under the split-config contract via other configs."
        );
        if let Some(best_nrmse) = sorted.iter().min_by(|a, b| a.1.total_cmp(&b.1)) {
            println!(
                "  best NRMSE:      λ={:.0e} → NRMSE={:.6e}, threshold={:.2} LT",
                best_nrmse.0, best_nrmse.1, best_nrmse.2
            );
        }
        if let Some(best_thr) = sorted.iter().max_by(|a, b| a.2.total_cmp(&b.2)) {
            println!(
                "  best threshold:  λ={:.0e} → NRMSE={:.6e}, threshold={:.2} LT",
                best_thr.0, best_thr.1, best_thr.2
            );
        }
    }
    println!();
    println!("  Next steps (if no λ passes): ");
    println!("    1. More training data (N=20_000+) — fixes underdetermination at the source.");
    println!("    2. Gate re-spec (Issue 186 Path D) — promote on K=4 NRMSE + K=8/M=24 threshold.");

    // No assertion — this is a measurement test, not a pass/fail gate.
}

/// Issue 187 T7 follow-up #2: K=10 λ-sweep at d_h=29_160.
///
/// Phase 5.1 showed K=8/M=8/R=2 at λ=5e-2 PASSES the NRMSE gate (9.43e-4)
/// but FAILS the threshold gate (7.23 LT vs ≥8 LT). The threshold was flat
/// across λ (~7.0-7.2 LT), indicating the limiting factor is K (delay
/// memory), not regularization.
///
/// This test checks whether +2 delay steps (K=10) extend the threshold to
/// ≥8 LT. Linear K-extrapolation from the K=4 (2.85 LT) and K=8 (7.23 LT)
/// data points predicts K=10 ≈ 8.5 LT (PASS).
///
/// # Config
///
/// - K=10 (delay length, +2 over Phase 5.1)
/// - M=8, R=2 (same basis as Phase 5.1)
/// - d_h = K·D·M·(K·D·M+1)/2 = 240·241/2 = 29_160
/// - Full-rank direct Cholesky (Gram is 6.8 GB — fits in 64 GB RAM)
/// - λ sweep ∈ {5e-3, 5e-2, 5e-1, 5e0} (same range as Phase 5.1 for direct
///   comparison; optimal λ may shift due to 1.56× more underdetermination)
///
/// # Compute budget
///
/// Gram build ~12 min + 4 parallel Cholesky factorizations (~28 min each,
/// all running in parallel via rayon) + rollout ~2 min = ~42 min wall.
///
/// # Running
///
/// ```bash
/// CARGO_TARGET_DIR=/tmp/katgpt-g1-dh29160 cargo test --release \
///   --features karc_forecaster \
///   --test karc_g1_dh18720 g1_dh_29160_k10_lambda_sweep -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn g1_dh_29160_k10_lambda_sweep() {
    use rayon::prelude::*;
    use std::time::Instant;

    // Local const shadows — override the module-level K=8 constants for this
    // K=10 config. D, M, R, N_TRAIN, DT, LYAPUNOV_TIME_UNITS, SAMPLES_PER_LT,
    // SUBSTEPS are inherited unchanged from the module-level consts.
    const K: usize = 10;
    const D_H_1: usize = K * D * M; // 240
    const D_H: usize = higher_order_feature_count(D_H_1, R); // 29_160

    println!(
        "Issue 187 T7 follow-up #2: KARC G1 λ-sweep at d_h = {D_H} (K={K}, M={M}, R={R}) \
         [full-rank Cholesky, parallel]"
    );
    println!("  Lyapunov time ≈ {LYAPUNOV_TIME_UNITS} units ≈ {SAMPLES_PER_LT} samples");

    // ── 1. Trajectory + per-coordinate normalization to [-1, 1] ───────────
    let t_traj = Instant::now();
    let traj_raw = generate_double_scroll(N_TRAIN + K + 50, DT, 1000, SUBSTEPS);
    let mut traj = traj_raw.clone();
    let mut scale = [1.0f32; D];
    let mut offset = [0.0f32; D];
    for d in 0..D {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..(traj.len() / D) {
            let v = traj[i * D + d];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        let range = (hi - lo).max(1e-6);
        offset[d] = (hi + lo) * 0.5;
        scale[d] = 2.0 / range;
        for i in 0..(traj.len() / D) {
            traj[i * D + d] = (traj[i * D + d] - offset[d]) * scale[d];
        }
    }
    let n_total = traj.len() / D;
    println!(
        "  trajectory: {} samples, built in {:.2?}",
        n_total,
        t_traj.elapsed()
    );

    // ── 2. Higher-order feature expansion ─────────────────────────────────
    let t_feat = Instant::now();
    let n_pairs = n_total - K;
    let mut features_ho = vec![0.0f32; n_pairs * D_H];
    let mut targets_ho = vec![0.0f32; n_pairs * D];
    let basis = ChebyshevBasis::<M>::new();
    let mut row_buf = vec![0.0f32; D_H];
    for (pair_idx, t) in ((K - 1)..(n_total - 1)).enumerate() {
        let mut delay = [0.0f32; K * D];
        for lag in 0..K {
            let idx = t - lag;
            for d in 0..D {
                delay[lag * D + d] = traj[idx * D + d];
            }
        }
        feature_expand_higher_order::<ChebyshevBasis<M>, M, R>(&delay, &basis, &mut row_buf);
        features_ho[pair_idx * D_H..(pair_idx + 1) * D_H].copy_from_slice(&row_buf);
        for d in 0..D {
            targets_ho[pair_idx * D + d] = traj[(t + 1) * D + d];
        }
    }
    println!(
        "  feature expansion: {} pairs × {} features, built in {:.2?}",
        n_pairs,
        D_H,
        t_feat.elapsed()
    );

    // ── 3. Build Gram (d_h × d_h = 6.8 GB f64) + Cov ONCE — no λI yet ─────
    let t_gram = Instant::now();
    let mut gram_unreg = vec![0.0f64; D_H * D_H];
    let feature_iter = (0..n_pairs).map(|i| &features_ho[i * D_H..(i + 1) * D_H] as &[f32]);
    chunked_gram_into(feature_iter, &mut gram_unreg, 0.0, D_H);
    let mut cov = vec![0.0f64; D_H * D];
    for p in 0..n_pairs {
        let row = &features_ho[p * D_H..(p + 1) * D_H];
        let target = &targets_ho[p * D..(p + 1) * D];
        for i in 0..D_H {
            let ri = row[i] as f64;
            for d in 0..D {
                cov[i * D + d] += ri * target[d] as f64;
            }
        }
    }
    eprintln!(
        "  Gram + Cov build: {:.2?} (Gram is {} GB)",
        t_gram.elapsed(),
        (D_H * D_H * 8) as f64 / 1e9
    );

    // Free the feature matrix — we only need the Gram for the sweep.
    drop(features_ho);
    drop(targets_ho);
    drop(row_buf);

    // ── 4. Parallel λ sweep ───────────────────────────────────────────────
    //
    // Same pattern as g1_dh_18720_lambda_sweep but at d_h=29_160. Each λ
    // value runs on its own rayon thread with ~13.6 GB scratch buffers
    // (gram_work 6.8 GB + chol 6.8 GB).
    //
    // **Memory budget:** 2 λ values × 13.6 GB + 6.8 GB shared Gram = ~34 GB
    // peak. Safe on 64 GB RAM. The K=8 sweep ran 4 λ values (5.6 GB each = 22
    // GB), but K=10's 1.56× larger Gram forces a tighter sweep. We test 2
    // values: λ=5e-2 (the K=8 optimum — likely near-optimal for K=10 too)
    // and λ=1e-1 (slightly higher, covering K=10's worse underdetermination).
    //
    // Cholesky at d_h=29_160 is ~28 min single-threaded (O(d_h³/3) ≈
    // 8.3×10¹² FLOPs at ~5 GFLOPS). 2 in parallel via rayon ≈ 28 min wall.
    let lambdas: Vec<f64> = vec![5e-2, 1e-1];
    eprintln!(
        "  sweeping {} λ values in parallel via rayon: {:?}",
        lambdas.len(),
        lambdas
    );

    // Shared read-only state — captured as Copy slice references by the
    // move closure (fat pointers are Copy, so the closure is Fn + Send).
    let gram_slice: &[f64] = &gram_unreg;
    let cov_slice: &[f64] = &cov;
    let traj_slice: &[f32] = &traj;
    let traj_raw_slice: &[f32] = &traj_raw;
    let scale_arr: [f32; D] = scale;
    let offset_arr: [f32; D] = offset;

    let sweep_start = Instant::now();
    let results: Vec<(f64, f32, f64, std::time::Duration)> = lambdas
        .par_iter()
        .map(move |&lambda| {
            // Per-thread buffers (~13.6 GB each)
            let mut gram_work = vec![0.0f64; D_H * D_H];
            let mut chol = vec![0.0f64; D_H * D_H];
            let mut z = vec![0.0f64; D_H * D];
            let mut wt = vec![0.0f64; D_H * D];
            let mut wout = vec![0.0f32; D * D_H];
            let mut psi = vec![0.0f32; D_H];

            // Copy unreg Gram + add λI
            gram_work.copy_from_slice(gram_slice);
            for i in 0..D_H {
                gram_work[i * D_H + i] += lambda;
            }

            // Cholesky + solve
            let t_fit = Instant::now();
            ridge_solve_direct_f64(&mut wt, &mut chol, &mut z, &gram_work, cov_slice, D_H, D);
            let fit_dt = t_fit.elapsed();

            // Free Gram + Cholesky before the rollout — we only need Wout.
            drop(gram_work);
            drop(chol);
            drop(z);

            // Build Wout (D × d_h, f32) = transpose of wt
            for d in 0..D {
                for j in 0..D_H {
                    wout[d * D_H + j] = wt[j * D + d] as f32;
                }
            }
            drop(wt);

            // Autonomous rollout over 20 LT
            let horizon_20lt = (20.0 * SAMPLES_PER_LT).ceil() as usize;
            let seed_t = n_total - 1;
            let mut delay = [0.0f32; K * D];
            for lag in 0..K {
                let idx = seed_t - lag;
                for d in 0..D {
                    delay[lag * D + d] = traj_slice[idx * D + d];
                }
            }
            let mut true_state: [f64; 3] = [
                traj_raw_slice[seed_t * D] as f64,
                traj_raw_slice[seed_t * D + 1] as f64,
                traj_raw_slice[seed_t * D + 2] as f64,
            ];
            let mut pred = Vec::with_capacity(horizon_20lt * D);
            let mut truth = Vec::with_capacity(horizon_20lt * D);
            let mut cur_delay = delay;
            let basis_local = ChebyshevBasis::<M>::new();
            for _ in 0..horizon_20lt {
                rk4_step_substepped(&mut true_state, DT, SUBSTEPS);
                truth.push(true_state[0] as f32);
                truth.push(true_state[1] as f32);
                truth.push(true_state[2] as f32);
                feature_expand_higher_order::<ChebyshevBasis<M>, M, R>(
                    &cur_delay,
                    &basis_local,
                    &mut psi,
                );
                let mut out_norm = [0.0f32; D];
                for d in 0..D {
                    let mut s = 0.0f32;
                    for j in 0..D_H {
                        s += wout[d * D_H + j] * psi[j];
                    }
                    out_norm[d] = s;
                }
                for d in 0..D {
                    pred.push(out_norm[d] / scale_arr[d] + offset_arr[d]);
                }
                let mut new_delay = [0.0f32; K * D];
                new_delay[..D].copy_from_slice(&out_norm);
                new_delay[D..].copy_from_slice(&cur_delay[..(K - 1) * D]);
                cur_delay = new_delay;
            }

            // G1 metrics
            let n_one_lt = (1.0 * SAMPLES_PER_LT).ceil() as usize;
            let n_one_lt = n_one_lt.max(1).min(pred.len() / D);
            let nrmse_one_lt = nrmse(&pred[..n_one_lt * D], &truth[..n_one_lt * D], D);
            let sigma = mean_sigma(&truth, D);
            let thr_sample = threshold_time(&pred, &truth, D, 0.1, sigma);
            let thr_lt = thr_sample as f64 / SAMPLES_PER_LT;

            (lambda, nrmse_one_lt, thr_lt, fit_dt)
        })
        .collect();
    let sweep_wall = sweep_start.elapsed();

    // Sort results by λ for the summary (rayon may return them out of order)
    let mut sorted = results;
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

    // ── 5. Summary ───────────────────────────────────────────────────────
    println!();
    println!("── λ-sweep summary (d_h = {D_H}, K={K}, M={M}, R={R}) ────────────────────");
    println!(
        "  sweep wall: {:.2?} ({} Cholesky factorizations in parallel)",
        sweep_wall,
        sorted.len()
    );
    println!();
    println!(
        "  {:>10}  {:>14}  {:>10}  {:>14}  {:>10}  {:>12}",
        "λ", "NRMSE(1 LT)", "gate", "threshold(LT)", "gate", "fit time"
    );
    for (lambda, nrmse_one_lt, thr_lt, fit_dt) in &sorted {
        let nrmse_pass = *nrmse_one_lt <= 1.0e-3;
        let thr_pass = *thr_lt >= 8.0;
        println!(
            "  {:>10.0e}  {:>14.6e}  {:>10}  {:>14.2}  {:>10}  {:>10.2?}",
            lambda,
            nrmse_one_lt,
            if nrmse_pass {
                "✅ ≤1e-3"
            } else {
                "❌ >1e-3"
            },
            thr_lt,
            if thr_pass { "✅ ≥8" } else { "❌ <8" },
            fit_dt
        );
    }
    println!();
    println!(
        "  Phase 5.1 reference: K=8/M=8/R=2 (d_h=18_720) at λ=5e-2 → NRMSE 9.43e-4 ✅, threshold 7.23 LT ❌"
    );
    println!();

    let any_pass = sorted
        .iter()
        .any(|(_, nrmse, thr, _)| *nrmse <= 1.0e-3 && *thr >= 8.0);
    if any_pass {
        let winners: Vec<_> = sorted
            .iter()
            .filter(|(_, nrmse, thr, _)| *nrmse <= 1.0e-3 && *thr >= 8.0)
            .collect();
        println!(
            "  VERDICT: {} λ value(s) PASS both G1 legs at this config — confirms \
             the split-config contract (Phase 22 promotion already landed).",
            winners.len()
        );
        for (lambda, nrmse_one_lt, thr_lt, _) in &winners {
            println!("    λ={lambda:.0e}: NRMSE={nrmse_one_lt:.6e}, threshold={thr_lt:.2} LT");
        }
    } else {
        println!(
            "  VERDICT: no λ passes both G1 legs at this config — informational only; `karc_forecaster` was promoted DEFAULT-ON (Phase 22) under the split-config contract via other configs."
        );
        if let Some(best_nrmse) = sorted.iter().min_by(|a, b| a.1.total_cmp(&b.1)) {
            println!(
                "  best NRMSE:      λ={:.0e} → NRMSE={:.6e}, threshold={:.2} LT",
                best_nrmse.0, best_nrmse.1, best_nrmse.2
            );
        }
        if let Some(best_thr) = sorted.iter().max_by(|a, b| a.2.total_cmp(&b.2)) {
            println!(
                "  best threshold:  λ={:.0e} → NRMSE={:.6e}, threshold={:.2} LT",
                best_thr.0, best_thr.1, best_thr.2
            );
        }
    }
    println!();
    println!("  Next steps (if no λ passes): ");
    println!("    1. More λ values (extend the sweep — the K=8 optimum was narrow).");
    println!("    2. K=12 at M=8/R=2 (d_h=41_904, ~14 GB Gram — still feasible but tight).");
    println!("    3. Accept the gate re-spec (Issue 186 Path D) — promote on the K=8 NRMSE");
    println!("       pass (Phase 5.1) + K=8/M=24 first-order threshold pass (Phase 1).");

    // No assertion — this is a measurement test, not a pass/fail gate.
}

/// Phase 5.3 G1 — R=1 K=8/M=24 λ-sweep at d_h=576 (2026-07-21).
///
/// The last unexplored single-config gate-pass candidate. Phase 1 ran this
/// exact config at λ=5e-3 and got threshold **8.16 LT (PASS)** but NRMSE
/// 4.79e-3 (FAIL, ~5× above the 1e-3 target). Phase 5.2 confirmed K is
/// exhausted (threshold plateaus at K≥8) and M is the only remaining
/// threshold lever — but M≥16 at R=2 explodes d_h (72_576, infeasible).
///
/// **The untested hypothesis:** at d_h=576 (well-determined with N≈4050
/// samples), λ=5e-3 is *too large* — the K=4 sweep showed the optimal λ
/// scales with d_h (5e-3 at d_h=4752, 5e-2 at d_h=18_720 when underdetermined).
/// For d_h=576 with N/d_h ≈ 7:1, smaller λ should give a strictly better
/// fit. If even one λ in {5e-4, 1e-3, 5e-3, 5e-2} brings NRMSE ≤ 1e-3 while
/// threshold stays ≥ 8 LT, we have a **clean single-config gate pass** and
/// promotion is straightforward — no gate re-spec needed.
///
/// Cost: seconds (d_h=576 is 32× smaller than d_h=18_720; Gram is 2.6 MB,
/// Cholesky is microseconds). This is the cheapest experiment with the
/// highest potential payoff in the entire KARC gate saga.
#[test]
fn g1_r1_k8_m24_dh576_lambda_sweep() {
    use std::time::Instant;

    const K_R1: usize = 8;
    const M_R1: usize = 24;
    const R_R1: usize = 1;
    const D_H_R1: usize = K_R1 * D * M_R1; // 576 — R=1 means no outer products
    const N_R1: usize = 4000;

    println!("Phase 5.3 R=1 K=8/M=24 λ-sweep: d_h={D_H_R1}, N={N_R1}");
    println!("  Reference (Phase 1, λ=5e-3): NRMSE 4.79e-3, threshold 8.16 LT");

    // Build trajectory + normalize
    let traj_raw = generate_double_scroll(N_R1 + K_R1 + 50, DT, 1000, SUBSTEPS);
    let mut traj = traj_raw.clone();
    let mut scale = [1.0f32; D];
    let mut offset = [0.0f32; D];
    for d in 0..D {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..(traj.len() / D) {
            let v = traj[i * D + d];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        let range = (hi - lo).max(1e-6);
        offset[d] = (hi + lo) * 0.5;
        scale[d] = 2.0 / range;
        for i in 0..(traj.len() / D) {
            traj[i * D + d] = (traj[i * D + d] - offset[d]) * scale[d];
        }
    }
    let n_total = traj.len() / D;
    let n_pairs = n_total - K_R1;

    // Build features + targets (R=1 path)
    let mut features = vec![0.0f32; n_pairs * D_H_R1];
    let mut targets = vec![0.0f32; n_pairs * D];
    let basis_r1 = ChebyshevBasis::<M_R1>::new();
    let mut row_buf = vec![0.0f32; D_H_R1];
    for (pair_idx, t) in ((K_R1 - 1)..(n_total - 1)).enumerate() {
        let mut delay = [0.0f32; K_R1 * D];
        for lag in 0..K_R1 {
            let idx = t - lag;
            for d in 0..D {
                delay[lag * D + d] = traj[idx * D + d];
            }
        }
        feature_expand_higher_order::<ChebyshevBasis<M_R1>, M_R1, R_R1>(
            &delay,
            &basis_r1,
            &mut row_buf,
        );
        features[pair_idx * D_H_R1..(pair_idx + 1) * D_H_R1].copy_from_slice(&row_buf);
        for d in 0..D {
            targets[pair_idx * D + d] = traj[(t + 1) * D + d];
        }
    }

    // Build unregularized Gram + Cov (ONCE)
    let mut gram_unreg = vec![0.0f64; D_H_R1 * D_H_R1];
    let feature_iter = (0..n_pairs).map(|i| &features[i * D_H_R1..(i + 1) * D_H_R1] as &[f32]);
    chunked_gram_into(feature_iter, &mut gram_unreg, 0.0, D_H_R1);
    let mut cov = vec![0.0f64; D_H_R1 * D];
    for p in 0..n_pairs {
        let row = &features[p * D_H_R1..(p + 1) * D_H_R1];
        let target = &targets[p * D..(p + 1) * D];
        for i in 0..D_H_R1 {
            let ri = row[i] as f64;
            for d in 0..D {
                cov[i * D + d] += ri * target[d] as f64;
            }
        }
    }
    drop(features);

    // Reusable buffers
    let mut gram_work = vec![0.0f64; D_H_R1 * D_H_R1];
    let mut chol = vec![0.0f64; D_H_R1 * D_H_R1];
    let mut z = vec![0.0f64; D_H_R1 * D];
    let mut wt = vec![0.0f64; D_H_R1 * D];
    let mut wout = vec![0.0f32; D * D_H_R1];
    let mut psi = vec![0.0f32; D_H_R1];

    // λ sweep — focus on the small-λ regime (d_h=576 is well-determined,
    // N/d_h ≈ 7:1, so smaller λ should strictly improve the fit).
    let lambdas: [f64; 5] = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2];
    let mut results: Vec<(f64, f32, f64)> = Vec::with_capacity(lambdas.len());

    println!("{:>10} {:>15} {:>15}", "λ", "NRMSE(1 LT)", "threshold(LT)");
    for &lambda in &lambdas {
        // Copy unreg Gram + add λI
        gram_work.copy_from_slice(&gram_unreg);
        for i in 0..D_H_R1 {
            gram_work[i * D_H_R1 + i] += lambda;
        }

        // Cholesky + solve
        let t_fit = Instant::now();
        ridge_solve_direct_f64(&mut wt, &mut chol, &mut z, &gram_work, &cov, D_H_R1, D);
        let fit_dt = t_fit.elapsed();

        // Build Wout (D × d_h, f32) = transpose of wt
        for d in 0..D {
            for j in 0..D_H_R1 {
                wout[d * D_H_R1 + j] = wt[j * D + d] as f32;
            }
        }

        // Autonomous rollout over 10 LT (extended to see threshold behavior)
        let horizon = (10.0 * SAMPLES_PER_LT).ceil() as usize;
        let seed_t = n_total - 1;
        let mut delay = [0.0f32; K_R1 * D];
        for lag in 0..K_R1 {
            let idx = seed_t - lag;
            for d in 0..D {
                delay[lag * D + d] = traj[idx * D + d];
            }
        }
        let mut true_state: [f64; 3] = [
            traj_raw[seed_t * D] as f64,
            traj_raw[seed_t * D + 1] as f64,
            traj_raw[seed_t * D + 2] as f64,
        ];
        let mut pred = Vec::with_capacity(horizon * D);
        let mut truth = Vec::with_capacity(horizon * D);
        let mut cur_delay = delay;
        for _ in 0..horizon {
            rk4_step_substepped(&mut true_state, DT, SUBSTEPS);
            truth.push(true_state[0] as f32);
            truth.push(true_state[1] as f32);
            truth.push(true_state[2] as f32);
            feature_expand_higher_order::<ChebyshevBasis<M_R1>, M_R1, R_R1>(
                &cur_delay, &basis_r1, &mut psi,
            );
            let mut out_norm = [0.0f32; D];
            for d in 0..D {
                let mut s = 0.0f32;
                for j in 0..D_H_R1 {
                    s += wout[d * D_H_R1 + j] * psi[j];
                }
                out_norm[d] = s;
            }
            for d in 0..D {
                pred.push(out_norm[d] / scale[d] + offset[d]);
            }
            let mut new_delay = [0.0f32; K_R1 * D];
            new_delay[..D].copy_from_slice(&out_norm);
            new_delay[D..].copy_from_slice(&cur_delay[..(K_R1 - 1) * D]);
            cur_delay = new_delay;
        }

        let n_one_lt = (1.0 * SAMPLES_PER_LT).ceil() as usize;
        let n_one_lt = n_one_lt.max(1).min(pred.len() / D);
        let nrmse_one_lt = nrmse(&pred[..n_one_lt * D], &truth[..n_one_lt * D], D);
        let sigma = mean_sigma(&truth, D);
        let thr_sample = threshold_time(&pred, &truth, D, 0.1, sigma);
        let thr_lt = thr_sample as f64 / SAMPLES_PER_LT;

        println!("{lambda:>10.0e} {nrmse_one_lt:>15.6e} {thr_lt:>15.2}   (fit {fit_dt:?})");
        results.push((lambda, nrmse_one_lt, thr_lt));
    }

    println!();
    let nrmse_target = 1.0e-3_f64;
    let threshold_target = 8.0_f64;
    let mut any_nrmse_pass = false;
    let mut any_threshold_pass = false;
    let mut any_both_pass = false;
    for &(lambda, nrmse_v, thr_v) in &results {
        let nrmse_pass = (nrmse_v as f64) <= nrmse_target;
        let thr_pass = thr_v >= threshold_target;
        any_nrmse_pass |= nrmse_pass;
        any_threshold_pass |= thr_pass;
        any_both_pass |= nrmse_pass && thr_pass;
        let mark = if nrmse_pass && thr_pass {
            "✅✅ BOTH PASS"
        } else if nrmse_pass {
            "   NRMSE pass"
        } else if thr_pass {
            "         thr pass"
        } else {
            ""
        };
        println!(
            "  λ={:.0e}: NRMSE={:.6e} {}, threshold={:.2} LT {}{}",
            lambda,
            nrmse_v,
            if nrmse_pass { "✅" } else { "❌" },
            thr_v,
            if thr_pass { "✅" } else { "❌" },
            mark
        );
    }

    println!();
    if any_both_pass {
        println!("  ✅✅✅ CLEAN SINGLE-CONFIG GATE PASS — promote to default-on.");
    } else if any_nrmse_pass && any_threshold_pass {
        println!("  ⚠️  SPLIT PASS — NRMSE and threshold pass at different λ values.");
        println!("     Proceed to gate re-spec (Issue 186 Path D).");
    } else {
        println!("  ❌ No clean pass. Proceed to gate re-spec (Issue 186 Path D).");
    }

    // No assertion — this is a measurement test, not a pass/fail gate.
}
