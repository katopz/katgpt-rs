//! KARC Phase 2 benchmark: higher-order features + low-rank ALS on double-scroll
//! (Plan 308 T2.5, paper Eqs. 32/44/47).
//!
//! Runs three configs on the double-scroll ODE (paper §A.1) and reports
//! autonomous-rollout NRMSE over 1 Lyapunov time for each:
//!
//! 1. **First-order full-rank** (Phase 1 baseline): `KarcForecaster<ChebyshevBasis<8>, 3, 8, 4>`,
//!    d_h = K·D·M = 96.
//! 2. **Higher-order R=2 full-rank**: d_h = 96 + 96·97/2 = 4752, fit via the
//!    chunked Gram (Eq. 44) + direct Cholesky solve. This is the path the paper
//!    uses for its headline NRMSE 5.3e-4.
//! 3. **First-order low-rank r=8**: d_h = 96, fit via ALS (Eq. 47) with r=8.
//!    Compares the low-rank A·B forecast quality to the first-order full-rank.
//!
//! **T2.5 exit criterion:** low-rank NRMSE within 1.5× of full-rank (first-order).
//!
//! Note: the higher-order R=2 low-rank fit (r=8, d_h=4752) is not run here —
//! the exact Kronecker B-step would require an (r·d_h)² = (8·4752)² ≈ 1.4B f64
//! ≈ 11.5 GB system matrix, which exceeds the benchmark memory budget. The
//! higher-order path ships the full-rank fit only in Phase 2; a future
//! large-d_h ALS path (Jacobi eigendecomposition + r separate d_h×d_h solves)
//! is tracked as future work.

use katgpt_core::{
    chunked_gram_into, feature_expand_higher_order, higher_order_feature_count,
    linalg::ridge_solve_direct_f64, ChebyshevBasis, KarcForecaster,
};

// ── Double-scroll ODE parameters (paper §A.1) ─────────────────────────────

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
    for j in 0..3 { tmp[j] = state[j] + 0.5 * dt * k1[j]; }
    double_scroll_rhs(&tmp, &mut k2);
    for j in 0..3 { tmp[j] = state[j] + 0.5 * dt * k2[j]; }
    double_scroll_rhs(&tmp, &mut k3);
    for j in 0..3 { tmp[j] = state[j] + dt * k3[j]; }
    double_scroll_rhs(&tmp, &mut k4);
    for j in 0..3 {
        state[j] += dt / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
    }
}

fn rk4_step_substepped(state: &mut [f64; 3], dt: f64, substeps: usize) {
    let dt_sub = dt / substeps as f64;
    for _ in 0..substeps { rk4_step(state, dt_sub); }
}

fn generate_double_scroll(n: usize, dt: f64, transient: usize, substeps: usize) -> Vec<f32> {
    let mut state: [f64; 3] = [0.1, 0.0, 0.0];
    for _ in 0..transient { rk4_step_substepped(&mut state, dt, substeps); }
    let mut out = Vec::with_capacity(n * 3);
    for _ in 0..n {
        rk4_step_substepped(&mut state, dt, substeps);
        out.push(state[0] as f32);
        out.push(state[1] as f32);
        out.push(state[2] as f32);
    }
    out
}

/// NRMSE over the first window, normalised by per-coordinate std of truth.
fn nrmse(pred: &[f32], truth: &[f32], dim: usize) -> f32 {
    let n = pred.len() / dim;
    let mut stds = [0.0f32; 8];
    for d in 0..dim {
        let mut mean = 0.0f64;
        for i in 0..n { mean += truth[i * dim + d] as f64; }
        mean /= n as f64;
        let mut var = 0.0f64;
        for i in 0..n { let dx = truth[i * dim + d] as f64 - mean; var += dx * dx; }
        var /= n as f64;
        stds[d] = var.sqrt() as f32;
    }
    let mut sum = 0.0f32;
    for d in 0..dim {
        let mut err_sq = 0.0f32;
        for i in 0..n { let e = pred[i * dim + d] - truth[i * dim + d]; err_sq += e * e; }
        sum += (err_sq / n as f32).sqrt() / stds[d].max(1e-12);
    }
    sum / dim as f32
}

/// Mean per-coordinate std of `truth` (used as the σ(u) reference for threshold).
fn mean_sigma(truth: &[f32], dim: usize) -> f32 {
    let n = truth.len() / dim;
    let mut sum_std = 0.0f32;
    for d in 0..dim {
        let mut mean = 0.0f64;
        for i in 0..n { mean += truth[i * dim + d] as f64; }
        mean /= n as f64;
        let mut var = 0.0f64;
        for i in 0..n { let dx = truth[i * dim + d] as f64 - mean; var += dx * dx; }
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
        if err_sq.sqrt() > bound { return i; }
    }
    n
}

// ── Config ────────────────────────────────────────────────────────────────

const D: usize = 3;
const K: usize = 4;
const M: usize = 8;
const N_TRAIN: usize = 2000;
const DT: f64 = 0.25;
const LYAPUNOV_TIME_UNITS: f64 = 7.81;
const SAMPLES_PER_LT: f64 = LYAPUNOV_TIME_UNITS / DT;
const SUBSTEPS: usize = 10;
const R: usize = 2; // higher-order outer-product order
const LR_RANK: usize = 8; // low-rank target rank

fn main() {
    println!("KARC Phase 2 benchmark: higher-order + low-rank on double-scroll");
    println!("  params: D={}, M={}, K={}, R={}, r={}", D, M, K, R, LR_RANK);
    println!("  N_train={}, dt={}, Lyapunov time ≈ {} units", N_TRAIN, DT, LYAPUNOV_TIME_UNITS);

    // Generate trajectory.
    let traj_raw = generate_double_scroll(N_TRAIN + K + 50, DT, 1000, SUBSTEPS);
    // Per-coordinate normalize to [-1,1] (Chebyshev stability).
    let mut traj = traj_raw.clone();
    let mut scale = [1.0f32; D];
    let mut offset = [0.0f32; D];
    for d in 0..D {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..(traj.len() / D) {
            let v = traj[i * D + d];
            if v < lo { lo = v; }
            if v > hi { hi = v; }
        }
        let range = (hi - lo).max(1e-6);
        offset[d] = (hi + lo) * 0.5;
        scale[d] = 2.0 / range;
        for i in 0..(traj.len() / D) {
            traj[i * D + d] = (traj[i * D + d] - offset[d]) * scale[d];
        }
    }
    let n_total = traj.len() / D;
    println!("  accumulated {} samples (normalized)", n_total);

    // ── Config 1: First-order full-rank baseline ──
    println!("\n── Config 1: first-order full-rank (Phase 1 baseline) ──");
    let basis = ChebyshevBasis::<M>::new();
    let mut f1: KarcForecaster<ChebyshevBasis<M>, D, M, K> =
        KarcForecaster::with_capacity(basis, N_TRAIN);
    for t in (K - 1)..(n_total - 1) {
        let mut delay = [0.0f32; K * D];
        for lag in 0..K {
            let idx = t - lag;
            for d in 0..D { delay[lag * D + d] = traj[idx * D + d]; }
        }
        let mut target = [0.0f32; D];
        for d in 0..D { target[d] = traj[(t + 1) * D + d]; }
        f1.accumulate_pair(&delay, &target);
    }
    let lambda = 5e-3f32;
    f1.fit_ridge(lambda).expect("fit_ridge");
    let nrmse_1 = autonomous_rollout_nrmse(&mut f1, &traj, &traj_raw, &scale, &offset, n_total);
    println!("  d_h = {}, NRMSE(1 LT) = {:.6e}", K * D * M, nrmse_1);

    // ── Config 2: Higher-order R=2 full-rank ──
    println!("\n── Config 2: higher-order R=2 full-rank (chunked Gram) ──");
    let d_h_1 = K * D * M;
    let d_h_ho = higher_order_feature_count(d_h_1, R);
    println!("  d_h_1 = {}, d_h(R=2) = {}", d_h_1, d_h_ho);
    // Build Gram and Cov via streaming (chunked_gram_into for G, manual for Cov).
    let lambda64 = lambda as f64;
    let mut gram_ho = vec![0.0f64; d_h_ho * d_h_ho];
    let mut cov_ho = vec![0.0f64; d_h_ho * D];
    // Feature buffer: we need to expand each delay state into higher-order features.
    // To stream into chunked_gram_into, we build an iterator of expanded rows.
    // For efficiency, expand all rows once into a buffer (N × d_h_ho) — at
    // N=2000, d_h_ho=4752, this is 2000×4752×4 = 38 MB (f32). Feasible.
    let n_pairs = n_total - K;
    let mut features_ho = vec![0.0f32; n_pairs * d_h_ho];
    let mut targets_ho = vec![0.0f32; n_pairs * D];
    let mut row_buf = vec![0.0f32; d_h_ho];
    {
        let basis = ChebyshevBasis::<M>::new();
        let mut pair_idx = 0;
        for t in (K - 1)..(n_total - 1) {
            let mut delay = [0.0f32; K * D];
            for lag in 0..K {
                let idx = t - lag;
                for d in 0..D { delay[lag * D + d] = traj[idx * D + d]; }
            }
            feature_expand_higher_order::<ChebyshevBasis<M>, M, R>(&delay, &basis, &mut row_buf);
            features_ho[pair_idx * d_h_ho..(pair_idx + 1) * d_h_ho].copy_from_slice(&row_buf);
            for d in 0..D { targets_ho[pair_idx * D + d] = traj[(t + 1) * D + d]; }
            pair_idx += 1;
        }
    }
    // Chunked Gram (no lambda — low_rank_fit / the direct solve adds it).
    let feature_iter = (0..n_pairs).map(|i| &features_ho[i * d_h_ho..(i + 1) * d_h_ho] as &[f32]);
    chunked_gram_into(feature_iter, &mut gram_ho, 0.0, d_h_ho);
    // Cov = X^T Y.
    for i in 0..d_h_ho { for d in 0..D { cov_ho[i * D + d] = 0.0; } }
    for p in 0..n_pairs {
        let row = &features_ho[p * d_h_ho..(p + 1) * d_h_ho];
        let target = &targets_ho[p * D..(p + 1) * D];
        for i in 0..d_h_ho {
            let ri = row[i] as f64;
            for d in 0..D { cov_ho[i * D + d] += ri * target[d] as f64; }
        }
    }
    // Direct ridge solve: W^T = (G + λI)^{-1} Cov  (f64).
    for i in 0..d_h_ho { gram_ho[i * d_h_ho + i] += lambda64; }
    let mut chol_ho = vec![0.0f64; d_h_ho * d_h_ho];
    let mut z_ho = vec![0.0f64; d_h_ho * D];
    let mut wt_ho = vec![0.0f64; d_h_ho * D];
    ridge_solve_direct_f64(&mut wt_ho, &mut chol_ho, &mut z_ho, &gram_ho, &cov_ho, d_h_ho, D);
    // Wout (D × d_h_ho, f32) = transpose of W^T.
    let mut wout_ho = vec![0.0f32; D * d_h_ho];
    for d in 0..D {
        for j in 0..d_h_ho {
            wout_ho[d * d_h_ho + j] = wt_ho[j * D + d] as f32;
        }
    }
    // Autonomous rollout with the higher-order Wout — 20 LT horizon for
    // full G1 gate (NRMSE over 1 LT + threshold at ε=0.1).
    let horizon_20lt = (20.0 * SAMPLES_PER_LT).ceil() as usize;
    let (pred_ho, truth_ho) = autonomous_rollout_higher_order(
        &wout_ho, &traj, &traj_raw, &scale, &offset, n_total, d_h_ho, horizon_20lt,
    );
    let n_one_lt = (1.0 * SAMPLES_PER_LT).ceil() as usize;
    let nrmse_2 = nrmse(&pred_ho[..n_one_lt * D], &truth_ho[..n_one_lt * D], D);
    let sigma_ho = mean_sigma(&truth_ho, D);
    let thr_sample_ho = threshold_time(&pred_ho, &truth_ho, D, 0.1, sigma_ho);
    let thr_lt_ho = thr_sample_ho as f64 / SAMPLES_PER_LT;
    println!("  NRMSE(1 LT) = {:.6e}", nrmse_2);
    println!("  threshold (ε=0.1): {} samples = {:.2} LT", thr_sample_ho, thr_lt_ho);
    println!("  σ(u) mean per-coord: {:.4}", sigma_ho);
    println!("  G1 NRMSE  ≤ 1.0e-3 : {}", if nrmse_2 <= 1.0e-3 { "PASS ✅" } else { "FAIL ❌" });
    println!("  G1 thresh ≥ 8 LT   : {}", if thr_lt_ho >= 8.0 { "PASS ✅" } else { "FAIL ❌" });
    println!("  paper reference: NRMSE 5.3e-4, threshold 16.7 LT");

    // ── Config 3: First-order low-rank r=8 ──
    println!("\n── Config 3: first-order low-rank r={} (ALS) ──", LR_RANK);
    f1.fit_low_rank(LR_RANK, lambda, 100, 1e-10).expect("fit_low_rank");
    let nrmse_3 = autonomous_rollout_low_rank(&mut f1, &traj, &traj_raw, &scale, &offset, n_total);
    println!("  d_h = {}, r = {}, NRMSE(1 LT) = {:.6e}", K * D * M, LR_RANK, nrmse_3);

    // ── Summary ──
    println!("\n── T2.5 summary ────────────────────────────────────────────");
    println!("  first-order full-rank:  NRMSE = {:.6e}", nrmse_1);
    println!("  higher-order R=2 full:  NRMSE = {:.6e}  (paper headline 5.3e-4)", nrmse_2);
    println!("  first-order low-rank:   NRMSE = {:.6e}  (r={})", nrmse_3, LR_RANK);
    let ratio = nrmse_3 / nrmse_1.max(1e-12);
    println!("  low-rank / full-rank ratio: {:.3}× (target ≤ 1.5×)", ratio);
    if ratio <= 1.5 {
        println!("  T2.5 gate: PASS ✅");
    } else {
        println!("  T2.5 gate: FAIL ❌ (low-rank NRMSE > 1.5× full-rank)");
    }
}

/// Autonomous rollout for the first-order forecaster, returns NRMSE over 1 LT.
fn autonomous_rollout_nrmse(
    f: &mut KarcForecaster<ChebyshevBasis<M>, D, M, K>,
    traj: &[f32],
    traj_raw: &[f32],
    scale: &[f32],
    offset: &[f32],
    n_total: usize,
) -> f32 {
    let horizon = (1.0 * SAMPLES_PER_LT).ceil() as usize;
    let seed_t = n_total - 1;
    let mut delay = [0.0f32; K * D];
    for lag in 0..K {
        let idx = seed_t - lag;
        for d in 0..D { delay[lag * D + d] = traj[idx * D + d]; }
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
        let mut out_norm = [0.0f32; D];
        f.forecast_into(&cur_delay, &mut out_norm);
        for d in 0..D { pred.push(out_norm[d] / scale[d] + offset[d]); }
        let mut new_delay = [0.0f32; K * D];
        new_delay[..D].copy_from_slice(&out_norm);
        new_delay[D..].copy_from_slice(&cur_delay[..(K - 1) * D]);
        cur_delay = new_delay;
    }
    nrmse(&pred, &truth, D)
}

/// Autonomous rollout with an external Wout matrix (higher-order full-rank).
/// Returns `(pred, truth)` over `horizon` steps in RAW (un-normalized) space.
fn autonomous_rollout_higher_order(
    wout: &[f32],
    traj: &[f32],
    traj_raw: &[f32],
    scale: &[f32],
    offset: &[f32],
    n_total: usize,
    d_h_ho: usize,
    horizon: usize,
) -> (Vec<f32>, Vec<f32>) {
    let seed_t = n_total - 1;
    let mut delay = [0.0f32; K * D];
    for lag in 0..K {
        let idx = seed_t - lag;
        for d in 0..D { delay[lag * D + d] = traj[idx * D + d]; }
    }
    let mut true_state: [f64; 3] = [
        traj_raw[seed_t * D] as f64,
        traj_raw[seed_t * D + 1] as f64,
        traj_raw[seed_t * D + 2] as f64,
    ];
    let basis = ChebyshevBasis::<M>::new();
    let mut psi = vec![0.0f32; d_h_ho];
    let mut pred = Vec::with_capacity(horizon * D);
    let mut truth = Vec::with_capacity(horizon * D);
    let mut cur_delay = delay;
    for _ in 0..horizon {
        rk4_step_substepped(&mut true_state, DT, SUBSTEPS);
        truth.push(true_state[0] as f32);
        truth.push(true_state[1] as f32);
        truth.push(true_state[2] as f32);
        feature_expand_higher_order::<ChebyshevBasis<M>, M, R>(&cur_delay, &basis, &mut psi);
        let mut out_norm = [0.0f32; D];
        // out = Wout · psi.
        for d in 0..D {
            let mut s = 0.0f32;
            for j in 0..d_h_ho { s += wout[d * d_h_ho + j] * psi[j]; }
            out_norm[d] = s;
        }
        for d in 0..D { pred.push(out_norm[d] / scale[d] + offset[d]); }
        let mut new_delay = [0.0f32; K * D];
        new_delay[..D].copy_from_slice(&out_norm);
        new_delay[D..].copy_from_slice(&cur_delay[..(K - 1) * D]);
        cur_delay = new_delay;
    }
    (pred, truth)
}

/// Autonomous rollout with the low-rank A·B forecast.
fn autonomous_rollout_low_rank(
    f: &mut KarcForecaster<ChebyshevBasis<M>, D, M, K>,
    traj: &[f32],
    traj_raw: &[f32],
    scale: &[f32],
    offset: &[f32],
    n_total: usize,
) -> f32 {
    let horizon = (1.0 * SAMPLES_PER_LT).ceil() as usize;
    let seed_t = n_total - 1;
    let mut delay = [0.0f32; K * D];
    for lag in 0..K {
        let idx = seed_t - lag;
        for d in 0..D { delay[lag * D + d] = traj[idx * D + d]; }
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
        let mut out_norm = [0.0f32; D];
        f.forecast_low_rank_into(&cur_delay, &mut out_norm);
        for d in 0..D { pred.push(out_norm[d] / scale[d] + offset[d]); }
        let mut new_delay = [0.0f32; K * D];
        new_delay[..D].copy_from_slice(&out_norm);
        new_delay[D..].copy_from_slice(&cur_delay[..(K - 1) * D]);
        cur_delay = new_delay;
    }
    nrmse(&pred, &truth, D)
}
