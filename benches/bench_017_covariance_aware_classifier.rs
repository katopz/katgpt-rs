//! bench_017 — Covariance-aware classifier probe
//!
//! Tests whether the per-token SNR floor (bench_016, SNR ≈ 1.0) is fundamental
//! or an artifact of the Euclidean nearest-centroid classifier.
//!
//! ## Hypothesis
//!
//! bench_016 found that value-sensitive features DO change with perturbation
//! (centroid distances 100-200× larger than the geometry encoder), but per-token
//! accuracy stays at ~50% because token-to-token variance swamps the signal.
//! The recommendation was: "a covariance-aware classifier (Mahalanobis/LDA)
//! would be needed."
//!
//! This bench tests that recommendation. If the within-class noise has
//! significant off-diagonal covariance structure, Mahalanobis whitening can
//! amplify the signal in low-variance directions, effectively boosting SNR.
//! If the noise is approximately isotropic (Σ ≈ σ²I), Mahalanobis ≈ Euclidean
//! and the floor is fundamental.
//!
//! ## Three classifiers
//!
//! 1. **Euclidean** — bench_016 baseline (unit-norm direction dot product)
//! 2. **Diagonal Mahalanobis** — per-dimension scaling (Σ = diag(σ₁²,...,σ_d²))
//!    Tests whether per-dimension variance weighting helps.
//! 3. **Full Mahalanobis** — Ledoit-Wolf shrunk covariance
//!    Tests whether off-diagonal covariance structure helps.
//!
//! ## Key diagnostic
//!
//! The Mahalanobis centroid distance d_M between class centroids. For 2-class
//! Gaussian with shared covariance, Bayes-optimal accuracy ≈ Φ(d_M / 2).
//! If d_M < 2, even the optimal classifier can't reach 80% per-token accuracy.

#![cfg(all(feature = "kimi_k3_loader", feature = "swe_trajectory_freeze"))]
#![allow(clippy::needless_range_loop)]

use katgpt_attn::gdn2::kda_forward::KdaWeights;
use katgpt_attn::mla::MlaWeights;
use katgpt_core::swe_trajectory_freeze::GeometrySummaryEncoder;
use katgpt_rs::kimi_k3::decoder_layer::{
    KimiAttentionWeights, KimiDecoderLayerWeights, KimiFfnWeights,
};
use katgpt_rs::kimi_k3::loader::{load_kimi_k3, KimiK3ModelWeights};
use katgpt_rs::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime, kimi_k3_forward_token_traced};
use katgpt_transformer::attn_res::AttnResWeights;
use katgpt_transformer::moe::{MoeWeights, SwiGluExpertWeights};

// ─── Constants ─────────────────────────────────────────────────────────────

/// Max feature dimension across all encoders.
const D_MAX: usize = 32;

/// Number of classes (Model A vs Model B(σ)).
const N_CLASSES: usize = 2;

/// Total tokens to extract trajectories for. Increased from bench_016's 32
/// to provide enough samples for covariance estimation (need N >> d for
/// reliable d×d covariance; 96 train × 2 classes = 192 samples for d ≤ 32).
const N_TOKENS: usize = 128;

/// Training split size (per model). 96 train + 32 test.
const N_TRAIN: usize = 96;

/// Truncated vocab (same as bench_016).
const BENCH_VOCAB: usize = 512;

/// Perturbation σ levels (same as bench_016).
const SIGMA_LEVELS: &[f32] = &[0.0, 0.001, 0.01, 0.05, 0.1, 0.5];

// ─── Deterministic LCG (copied from bench_015/016) ─────────────────────────

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    #[inline]
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
    }
}

// ─── Weight perturbation (copied from bench_016) ───────────────────────────

#[inline]
fn perturb_vec(v: &mut [f32], rng: &mut Lcg, sigma: f32) {
    if sigma == 0.0 {
        return;
    }
    for w in v.iter_mut() {
        let noise = rng.next_f32();
        *w *= 1.0 + sigma * noise;
    }
}

fn perturb_attn_res(w: &mut AttnResWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.norm_weight, rng, sigma);
    perturb_vec(&mut w.proj_weight, rng, sigma);
}

fn perturb_mla(w: &mut MlaWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.w_dkv, rng, sigma);
    perturb_vec(&mut w.w_dq, rng, sigma);
    perturb_vec(&mut w.w_uq, rng, sigma);
    perturb_vec(&mut w.w_qr, rng, sigma);
    perturb_vec(&mut w.w_uk, rng, sigma);
    perturb_vec(&mut w.w_uv, rng, sigma);
    perturb_vec(&mut w.w_kr, rng, sigma);
    perturb_vec(&mut w.w_o, rng, sigma);
    perturb_vec(&mut w.q_a_norm_weight, rng, sigma);
    perturb_vec(&mut w.kv_a_norm_weight, rng, sigma);
    if let Some(w_g) = w.w_g.as_mut() {
        perturb_vec(w_g, rng, sigma);
    }
}

fn perturb_kda(w: &mut KdaWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.q_proj, rng, sigma);
    perturb_vec(&mut w.k_proj, rng, sigma);
    perturb_vec(&mut w.v_proj, rng, sigma);
    perturb_vec(&mut w.q_conv_weight, rng, sigma);
    perturb_vec(&mut w.k_conv_weight, rng, sigma);
    perturb_vec(&mut w.v_conv_weight, rng, sigma);
    perturb_vec(&mut w.a_log, rng, sigma);
    perturb_vec(&mut w.f_a_proj, rng, sigma);
    perturb_vec(&mut w.f_b_proj, rng, sigma);
    perturb_vec(&mut w.dt_bias, rng, sigma);
    perturb_vec(&mut w.beta_proj, rng, sigma);
    perturb_vec(&mut w.g_proj, rng, sigma);
    perturb_vec(&mut w.o_norm_weight, rng, sigma);
    perturb_vec(&mut w.o_proj, rng, sigma);
}

fn perturb_swiglu(w: &mut SwiGluExpertWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.gate_proj, rng, sigma);
    perturb_vec(&mut w.up_proj, rng, sigma);
    perturb_vec(&mut w.down_proj, rng, sigma);
}

fn perturb_moe(w: &mut MoeWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.router_weight, rng, sigma);
    perturb_vec(&mut w.e_score_correction_bias, rng, sigma);
    for expert in w.experts.iter_mut() {
        perturb_swiglu(expert, rng, sigma);
    }
    for expert in w.shared_experts.iter_mut() {
        perturb_swiglu(expert, rng, sigma);
    }
    if let Some(p) = w.routed_expert_down_proj.as_mut() {
        perturb_vec(p, rng, sigma);
    }
    if let Some(p) = w.routed_expert_up_proj.as_mut() {
        perturb_vec(p, rng, sigma);
    }
    if let Some(p) = w.routed_expert_norm_weight.as_mut() {
        perturb_vec(p, rng, sigma);
    }
}

fn perturb_layer(w: &mut KimiDecoderLayerWeights, rng: &mut Lcg, sigma: f32) {
    perturb_vec(&mut w.input_layernorm_weight, rng, sigma);
    perturb_vec(&mut w.post_attention_layernorm_weight, rng, sigma);
    match &mut w.attention {
        KimiAttentionWeights::Mla(m) => perturb_mla(m, rng, sigma),
        KimiAttentionWeights::Kda(k) => perturb_kda(k, rng, sigma),
    }
    match &mut w.ffn {
        KimiFfnWeights::Dense(s) => perturb_swiglu(s, rng, sigma),
        KimiFfnWeights::Moe(m) => perturb_moe(m, rng, sigma),
    }
    perturb_attn_res(&mut w.self_attn_res, rng, sigma);
    perturb_attn_res(&mut w.mlp_attn_res, rng, sigma);
}

fn perturb_model(w: &mut KimiK3ModelWeights, sigma: f32) {
    let seed = (sigma * 1_000_000.0) as u64 | 0xA15E_0000;
    let mut rng = Lcg::new(seed);
    perturb_vec(&mut w.embed_weight, &mut rng, sigma);
    for layer in w.layers.iter_mut() {
        perturb_layer(layer, &mut rng, sigma);
    }
    perturb_vec(&mut w.final_norm_weight, &mut rng, sigma);
    if !w.lm_head_weight.is_empty() {
        perturb_vec(&mut w.lm_head_weight, &mut rng, sigma);
    }
    perturb_attn_res(&mut w.output_attn_res, &mut rng, sigma);
}

// ─── Trajectory extraction (copied from bench_016) ─────────────────────────

struct ExtractScratch {
    disp_curr: Vec<f32>,
    disp_prev: Vec<f32>,
    traj_buf: Vec<Vec<f32>>,
}

impl ExtractScratch {
    fn new(hidden_dim: usize) -> Self {
        Self {
            disp_curr: Vec::with_capacity(hidden_dim),
            disp_prev: Vec::with_capacity(hidden_dim),
            traj_buf: Vec::with_capacity(9),
        }
    }

    fn extract_traj(
        &mut self,
        config: &KimiK3ModelConfig,
        weights: &KimiK3ModelWeights,
        runtime: &mut KimiK3Runtime,
        token_id: u32,
    ) {
        runtime.reset();
        self.traj_buf.clear();
        let _ = kimi_k3_forward_token_traced(
            config, weights, runtime, token_id, &mut self.traj_buf,
        );
    }
}

// ─── Natural-dimension encoders ────────────────────────────────────────────
//
// Unlike bench_016 (which replicated features to D=32), these produce
// features at their NATURAL dimensionality. This avoids singular covariance
// matrices from replicated features and gives each encoder its optimal d.

#[derive(Clone, Copy, PartialEq)]
enum EncoderKind {
    DispNorms,
    DispStats,
    StateNorms,
    DispRatios,
}

impl EncoderKind {
    fn name(self) -> &'static str {
        match self {
            Self::DispNorms => "DispNorms",
            Self::DispStats => "DispStats",
            Self::StateNorms => "StateNorms",
            Self::DispRatios => "DispRatios",
        }
    }

    fn dim(self) -> usize {
        match self {
            Self::DispNorms => 8,
            Self::DispStats => 32, // 8 layers × 4 stats
            Self::StateNorms => 9,
            Self::DispRatios => 8,
        }
    }
}

/// Encode trajectory states into a natural-dimension feature vector.
/// Only the first `encoder.dim()` entries of `out` are written.
fn encode_natural(
    kind: EncoderKind,
    states: &[&[f32]],
    scratch: &mut ExtractScratch,
    out: &mut [f32; D_MAX],
) {
    // Zero entire buffer first.
    for v in out.iter_mut() {
        *v = 0.0;
    }

    match kind {
        EncoderKind::DispNorms => {
            let n_disps = states.len().saturating_sub(1).min(8);
            for l in 0..n_disps {
                let mut sum_sq = 0.0_f32;
                for i in 0..states[l].len() {
                    let diff = states[l + 1][i] - states[l][i];
                    sum_sq += diff * diff;
                }
                out[l] = sum_sq.sqrt();
            }
        }
        EncoderKind::DispStats => {
            let n_disps = states.len().saturating_sub(1).min(8);
            for l in 0..n_disps {
                let dim = states[l].len();
                let mut sum_sq = 0.0_f32;
                let mut sum = 0.0_f32;
                let mut max_abs = 0.0_f32;

                for i in 0..dim {
                    let diff = states[l + 1][i] - states[l][i];
                    sum_sq += diff * diff;
                    sum += diff;
                    let abs_diff = diff.abs();
                    if abs_diff > max_abs {
                        max_abs = abs_diff;
                    }
                }

                let l2 = sum_sq.sqrt();
                let mean = sum / dim as f32;
                let var = (sum_sq / dim as f32) - (mean * mean);

                let base = l * 4;
                out[base] = l2;
                out[base + 1] = mean;
                out[base + 2] = var;
                out[base + 3] = max_abs;
            }
        }
        EncoderKind::StateNorms => {
            let n_states = states.len().min(9);
            for l in 0..n_states {
                let mut sum_sq = 0.0_f32;
                for i in 0..states[l].len() {
                    sum_sq += states[l][i] * states[l][i];
                }
                out[l] = sum_sq.sqrt();
            }
        }
        EncoderKind::DispRatios => {
            let n_disps = states.len().saturating_sub(1).min(8);
            let mut norms = [0.0_f32; 8];
            let mut total = 0.0_f32;

            for l in 0..n_disps {
                let mut sum_sq = 0.0_f32;
                for i in 0..states[l].len() {
                    let diff = states[l + 1][i] - states[l][i];
                    sum_sq += diff * diff;
                }
                norms[l] = sum_sq.sqrt();
                total += norms[l];
            }

            if total > 0.0 {
                for l in 0..8 {
                    out[l] = norms[l] / total;
                }
            }
        }
    }

    // Suppress unused warning for scratch (needed for geometry encoder path
    // in bench_016 but not here — kept for structural consistency).
    let _ = &mut scratch.disp_curr;
    let _ = &mut scratch.disp_prev;
}

// ─── Covariance estimation (Ledoit-Wolf shrinkage) ─────────────────────────
//
// Ledoit-Wolf (2004) "A well-conditioned estimator for large-dimensional
// covariance matrices". Shrinks the sample covariance toward m*I (scaled
// identity), where the shrinkage intensity α is chosen to minimize MSE.

/// Compute pooled within-class covariance with Ledoit-Wolf shrinkage.
///
/// `train_features`: `[N_CLASSES][N_TRAIN][d]` — raw feature vectors.
/// `d`: feature dimension.
/// `cov_out`: d×d shrunk covariance (row-major), overwritten.
/// `diag_var_out`: d diagonal variances (pre-shrinkage), for diagonal classifier.
///
/// Returns the shrinkage intensity α (0 = no shrinkage, 1 = full shrink to mI).
fn pooled_covariance_ledoit_wolf(
    train_features: &[[[f32; D_MAX]; N_TRAIN]; N_CLASSES],
    d: usize,
    cov_out: &mut [f32],      // d*d
    diag_var_out: &mut [f32], // d
) -> f32 {
    let n_per_class = N_TRAIN;
    let n_total = N_CLASSES * N_TRAIN;
    let dof = n_total - N_CLASSES; // degrees of freedom

    // ── Step 1: Per-class means ──
    let mut class_means = [[0.0_f32; D_MAX]; N_CLASSES];
    for k in 0..N_CLASSES {
        for j in 0..d {
            let mut sum = 0.0;
            for i in 0..n_per_class {
                sum += train_features[k][i][j];
            }
            class_means[k][j] = sum / n_per_class as f32;
        }
    }

    // ── Step 2: Pooled within-class scatter → sample covariance ──
    // S = (1/dof) Σ_k Σ_i (x_i - μ_k)(x_i - μ_k)^T
    for ij in 0..d * d {
        cov_out[ij] = 0.0;
    }
    for k in 0..N_CLASSES {
        for i in 0..n_per_class {
            // residual r = x_i - μ_k
            let mut r = [0.0_f32; D_MAX];
            for j in 0..d {
                r[j] = train_features[k][i][j] - class_means[k][j];
            }
            // outer product r r^T
            for a in 0..d {
                for b in 0..d {
                    cov_out[a * d + b] += r[a] * r[b];
                }
            }
        }
    }
    for ij in 0..d * d {
        cov_out[ij] /= dof as f32;
    }

    // Save diagonal variances for the diagonal classifier.
    for j in 0..d {
        diag_var_out[j] = cov_out[j * d + j];
    }

    // ── Step 3: Ledoit-Wolf shrinkage toward m*I ──
    // m = trace(S) / d  (average variance)
    let m: f32 = (0..d).map(|j| cov_out[j * d + j]).sum::<f32>() / d as f32;

    // d² = ||S - mI||_F² / d
    let mut d_sq = 0.0_f32;
    for a in 0..d {
        for b in 0..d {
            let target = if a == b { m } else { 0.0 };
            let diff = cov_out[a * d + b] - target;
            d_sq += diff * diff;
        }
    }
    d_sq /= d as f32;

    // b̄² = (1/N²) Σ_k Σ_i ||r_i r_i^T - S||_F² / d
    // where r_i = x_i - μ_{class(i)} (mean-centered residual).
    let mut b_bar_sq = 0.0_f32;
    for k in 0..N_CLASSES {
        for i in 0..n_per_class {
            let mut r = [0.0_f32; D_MAX];
            for j in 0..d {
                r[j] = train_features[k][i][j] - class_means[k][j];
            }
            for a in 0..d {
                for b in 0..d {
                    let diff = r[a] * r[b] - cov_out[a * d + b];
                    b_bar_sq += diff * diff;
                }
            }
        }
    }
    b_bar_sq /= (n_total * n_total) as f32 * d as f32;

    // b² = min(b̄², d²)
    let b_sq = b_bar_sq.min(d_sq);

    // α = b² / d² (shrinkage intensity)
    let alpha = if d_sq > 1e-20 { b_sq / d_sq } else { 1.0 };

    // Σ* = α * m * I + (1 - α) * S
    for a in 0..d {
        for b in 0..d {
            let target = if a == b { m } else { 0.0 };
            cov_out[a * d + b] = alpha * target + (1.0 - alpha) * cov_out[a * d + b];
        }
    }

    // Also store class means for the caller via a side channel.
    // (We return alpha; means are recomputed by the caller if needed.)
    alpha
}

/// Compute per-class means from training features.
fn compute_class_means(
    train_features: &[[[f32; D_MAX]; N_TRAIN]; N_CLASSES],
    d: usize,
) -> [[f32; D_MAX]; N_CLASSES] {
    let mut means = [[0.0_f32; D_MAX]; N_CLASSES];
    for k in 0..N_CLASSES {
        for j in 0..d {
            let mut sum = 0.0;
            for i in 0..N_TRAIN {
                sum += train_features[k][i][j];
            }
            means[k][j] = sum / N_TRAIN as f32;
        }
    }
    means
}

// ─── Cholesky decomposition + triangular solve ─────────────────────────────

/// Cholesky decomposition of a symmetric positive-definite d×d matrix.
/// Returns lower-triangular L such that A = L L^T (row-major).
/// Returns false if A is not positive definite.
fn cholesky_decompose(a: &[f32], d: usize, l: &mut [f32]) -> bool {
    for ij in 0..d * d {
        l[ij] = 0.0;
    }
    for j in 0..d {
        // Diagonal
        let mut sum = a[j * d + j];
        for k in 0..j {
            sum -= l[j * d + k] * l[j * d + k];
        }
        if sum <= 0.0 {
            return false;
        }
        let diag = sum.sqrt();
        l[j * d + j] = diag;

        // Off-diagonal (below diagonal)
        for i in (j + 1)..d {
            let mut sum = a[i * d + j];
            for k in 0..j {
                sum -= l[i * d + k] * l[j * d + k];
            }
            l[i * d + j] = sum / diag;
        }
    }
    true
}

/// Cholesky with jitter fallback. Adds increasing diagonal jitter until PD.
fn cholesky_with_jitter(a: &mut [f32], d: usize, l: &mut [f32]) -> u32 {
    let mut jitter = 0.0_f32;
    let mut attempts = 0u32;
    loop {
        if cholesky_decompose(a, d, l) {
            return attempts;
        }
        attempts += 1;
        // Remove previous jitter before adding new.
        if jitter > 0.0 {
            for j in 0..d {
                a[j * d + j] -= jitter;
            }
        }
        jitter = if jitter == 0.0 { 1e-6 } else { jitter * 10.0 };
        if jitter > 100.0 {
            // Last resort: diagonal-only.
            for ij in 0..d * d {
                l[ij] = 0.0;
            }
            for j in 0..d {
                l[j * d + j] = a[j * d + j].max(1e-12).sqrt();
            }
            return attempts;
        }
        for j in 0..d {
            a[j * d + j] += jitter;
        }
    }
}

/// Mahalanobis distance squared: d² = (x-μ)^T Σ⁻¹ (x-μ) via Cholesky L.
/// Σ = L L^T → Σ⁻¹ = L⁻ᵀ L⁻¹ → d² = ||L⁻¹(x-μ)||².
fn mahalanobis_sq(l: &[f32], d: usize, x: &[f32], mu: &[f32]) -> f32 {
    let mut y = [0.0_f32; D_MAX];
    // Forward substitution: solve L y = (x - μ)
    for i in 0..d {
        let mut sum = x[i] - mu[i];
        for k in 0..i {
            sum -= l[i * d + k] * y[k];
        }
        let diag = l[i * d + i];
        y[i] = if diag.abs() > 1e-15 { sum / diag } else { 0.0 };
    }
    // d² = ||y||²
    let mut dist_sq = 0.0;
    for i in 0..d {
        dist_sq += y[i] * y[i];
    }
    dist_sq
}

// ─── Classifiers ───────────────────────────────────────────────────────────

/// Euclidean nearest-centroid via dot product (bench_016 method).
/// Returns predicted class (0 or 1).
fn classify_euclidean(
    x: &[f32],
    d: usize,
    directions: &[[f32; D_MAX]; N_CLASSES],
    global_centroid: &[f32; D_MAX],
) -> usize {
    let mut centered = [0.0_f32; D_MAX];
    for j in 0..d {
        centered[j] = x[j] - global_centroid[j];
    }
    let mut best = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for k in 0..N_CLASSES {
        let mut dot = 0.0;
        for j in 0..d {
            dot += centered[j] * directions[k][j];
        }
        if dot > best_dot {
            best_dot = dot;
            best = k;
        }
    }
    best
}

/// Diagonal Mahalanobis (standardized Euclidean): d² = Σ_j (x_j-μ_j)²/σ_j².
fn classify_diagonal(
    x: &[f32],
    d: usize,
    means: &[[f32; D_MAX]; N_CLASSES],
    diag_var: &[f32],
) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for k in 0..N_CLASSES {
        let mut dist_sq = 0.0;
        for j in 0..d {
            let diff = x[j] - means[k][j];
            let var = diag_var[j].max(1e-15);
            dist_sq += diff * diff / var;
        }
        if dist_sq < best_dist {
            best_dist = dist_sq;
            best = k;
        }
    }
    best
}

/// Full Mahalanobis via Cholesky: d² = (x-μ)^T Σ⁻¹ (x-μ).
fn classify_mahalanobis(
    x: &[f32],
    d: usize,
    means: &[[f32; D_MAX]; N_CLASSES],
    l: &[f32],
) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;
    for k in 0..N_CLASSES {
        let dist_sq = mahalanobis_sq(l, d, x, &means[k]);
        if dist_sq < best_dist {
            best_dist = dist_sq;
            best = k;
        }
    }
    best
}

// ─── Normal CDF (for Bayes-optimal accuracy estimate) ──────────────────────

/// Standard normal CDF Φ(x) via the Abramowitz-Stegun erf approximation.
fn normal_cdf(x: f64) -> f64 {
    // Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    let z = x / (2.0_f64).sqrt();
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let az = z.abs();
    let a1 = 0.254829592_f64;
    let a2 = -0.284496736_f64;
    let a3 = 1.421413741_f64;
    let a4 = -1.453152027_f64;
    let a5 = 1.061405429_f64;
    let p = 0.3275911_f64;
    let t = 1.0 / (1.0 + p * az);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-az * az).exp();
    0.5 * (1.0 + sign * y)
}

// ─── Result struct ─────────────────────────────────────────────────────────

struct ClassifierResult {
    encoder: EncoderKind,
    #[allow(dead_code)]
    d: usize,
    sigma: f32,
    euclidean_acc: f32,
    diagonal_acc: f32,
    mahalanobis_acc: f32,
    centroid_euclidean: f32,
    centroid_mahalanobis: f32,
    shrinkage_alpha: f32,
    bayes_optimal: f32,
}

// ─── Per-encoder, per-σ classifier comparison ──────────────────────────────

#[allow(clippy::too_many_arguments)]
fn test_classifiers(
    summaries: &[[[f32; D_MAX]; N_TOKENS]; N_CLASSES], // [model][token]
    encoder: EncoderKind,
    sigma: f32,
) -> ClassifierResult {
    let d = encoder.dim();

    // ── Split: train [0..N_TRAIN], test [N_TRAIN..N_TOKENS] ──
    let mut train_features = [[[0.0_f32; D_MAX]; N_TRAIN]; N_CLASSES];
    for k in 0..N_CLASSES {
        for i in 0..N_TRAIN {
            train_features[k][i] = summaries[k][i];
        }
    }

    // ── Class means ──
    let class_means = compute_class_means(&train_features, d);

    // ── Global centroid (for Euclidean classifier) ──
    let mut global_centroid = [0.0_f32; D_MAX];
    for k in 0..N_CLASSES {
        for i in 0..N_TRAIN {
            for j in 0..d {
                global_centroid[j] += train_features[k][i][j];
            }
        }
    }
    let total = (N_CLASSES * N_TRAIN) as f32;
    for j in 0..d {
        global_centroid[j] /= total;
    }

    // ── Euclidean directions (unit-normalized centroid - global_centroid) ──
    let mut directions = [[0.0_f32; D_MAX]; N_CLASSES];
    for k in 0..N_CLASSES {
        let mut norm_sq = 0.0_f32;
        for j in 0..d {
            directions[k][j] = class_means[k][j] - global_centroid[j];
            norm_sq += directions[k][j] * directions[k][j];
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            for j in 0..d {
                directions[k][j] /= norm;
            }
        }
    }

    // ── Pooled covariance + Ledoit-Wolf shrinkage ──
    let mut cov = vec![0.0_f32; d * d];
    let mut diag_var = [0.0_f32; D_MAX];
    let alpha = pooled_covariance_ledoit_wolf(&train_features, d, &mut cov, &mut diag_var);

    // ── Cholesky decomposition of shrunk covariance ──
    let mut l = vec![0.0_f32; d * d];
    let _jitter_attempts = cholesky_with_jitter(&mut cov, d, &mut l);

    // ── Centroid distances ──
    // Euclidean
    let mut cent_euclid_sq = 0.0_f32;
    for j in 0..d {
        let diff = class_means[0][j] - class_means[1][j];
        cent_euclid_sq += diff * diff;
    }
    let centroid_euclidean = cent_euclid_sq.sqrt();

    // Mahalanobis: d_M² = (μ_0 - μ_1)^T Σ⁻¹ (μ_0 - μ_1)
    let mut delta = [0.0_f32; D_MAX];
    for j in 0..d {
        delta[j] = class_means[0][j] - class_means[1][j];
    }
    let centroid_mahalanobis_sq = mahalanobis_sq(&l, d, &delta, &[0.0_f32; D_MAX]);
    let centroid_mahalanobis = centroid_mahalanobis_sq.sqrt();

    // Bayes-optimal accuracy: P(correct) = Φ(d_M / 2) for 2-class equal-prior
    // Gaussian with shared covariance.
    let bayes_optimal = normal_cdf(centroid_mahalanobis as f64 / 2.0) as f32;

    // ── Classify test tokens ──
    let n_test = N_TOKENS - N_TRAIN;
    let mut euclid_correct = 0usize;
    let mut diag_correct = 0usize;
    let mut maha_correct = 0usize;

    for k in 0..N_CLASSES {
        for tok_idx in N_TRAIN..N_TOKENS {
            let x = &summaries[k][tok_idx];

            let pred_e = classify_euclidean(x, d, &directions, &global_centroid);
            if pred_e == k {
                euclid_correct += 1;
            }

            let pred_d = classify_diagonal(x, d, &class_means, &diag_var);
            if pred_d == k {
                diag_correct += 1;
            }

            let pred_m = classify_mahalanobis(x, d, &class_means, &l);
            if pred_m == k {
                maha_correct += 1;
            }
        }
    }

    let total_test = n_test * N_CLASSES;

    ClassifierResult {
        encoder,
        d,
        sigma,
        euclidean_acc: euclid_correct as f32 / total_test as f32,
        diagonal_acc: diag_correct as f32 / total_test as f32,
        mahalanobis_acc: maha_correct as f32 / total_test as f32,
        centroid_euclidean,
        centroid_mahalanobis,
        shrinkage_alpha: alpha,
        bayes_optimal,
    }
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  P011 follow-up — covariance-aware classifier probe (bench_017)   ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let d_model = config.hidden_size;
    println!("Config: D_model={d_model}, layers={}", config.num_layers);
    println!("Tokens: {N_TOKENS} ({N_TRAIN} train + {} test per model)",
        N_TOKENS - N_TRAIN);
    println!("Sigma levels: {SIGMA_LEVELS:?}");
    println!();

    // ── Load real model ───────────────────────────────────────────────────
    let model_dir = std::env::var("KIMI_K3_MODEL_DIR").unwrap_or_else(|_| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{manifest_dir}/data/kimi-k3-0.40b")
    });
    let model_path = format!("{model_dir}/model.safetensors");

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("ERROR: this experiment requires real model.safetensors at {model_path}");
        std::process::exit(1);
    }

    print!("Loading real model.safetensors ... ");
    let t0 = std::time::Instant::now();
    let weights_a = load_kimi_k3(&model_path).unwrap_or_else(|e| {
        eprintln!("\n  load failed: {e}");
        std::process::exit(1);
    });
    println!("done ({:.1}s)", t0.elapsed().as_secs_f64());
    println!();

    // ── Shared setup ──────────────────────────────────────────────────────
    let max_seq_len = 64;
    let mut runtime_a = KimiK3Runtime::new(&config, max_seq_len);
    let mut runtime_b = KimiK3Runtime::new(&config, max_seq_len);
    let _geom_encoder = GeometrySummaryEncoder::default_depth_trajectory();
    let tokens: Vec<u32> = (1..=N_TOKENS as u32)
        .map(|i| (i * 7 + 3) % (BENCH_VOCAB as u32))
        .collect();
    let mut scratch = ExtractScratch::new(d_model);

    let encoders = [
        EncoderKind::DispNorms,
        EncoderKind::DispStats,
        EncoderKind::StateNorms,
        EncoderKind::DispRatios,
    ];

    // ── Cache Model A trajectories (extracted once, reused across σ) ──────
    println!("Extracting Model A trajectories (128 tokens) ...");
    let t0 = std::time::Instant::now();
    let mut traj_a: Vec<Vec<Vec<f32>>> = Vec::with_capacity(N_TOKENS);
    for &tok in &tokens {
        scratch.extract_traj(&config, &weights_a, &mut runtime_a, tok);
        traj_a.push(scratch.traj_buf.clone());
    }
    println!("  done ({:.1}s)", t0.elapsed().as_secs_f64());

    // ── Run the sweep ─────────────────────────────────────────────────────
    let mut all_results: Vec<ClassifierResult> = Vec::new();

    for &sigma in SIGMA_LEVELS {
        let mut weights_b = weights_a.clone();
        perturb_model(&mut weights_b, sigma);

        // Extract Model B trajectories for this σ.
        let t0 = std::time::Instant::now();
        let mut traj_b: Vec<Vec<Vec<f32>>> = Vec::with_capacity(N_TOKENS);
        for &tok in &tokens {
            scratch.extract_traj(&config, &weights_b, &mut runtime_b, tok);
            traj_b.push(scratch.traj_buf.clone());
        }

        println!();
        println!("── σ = {} (extract {:.1}s) ────────────────────────────────────",
            sigma, t0.elapsed().as_secs_f64());
        println!(
            "  {:>10}  {:>3}  {:>9}  {:>9}  {:>9}  {:>6}  {:>9}  {:>9}  {:>8}",
            "encoder", "d", "Euclidean", "DiagMaha", "FullMaha", "λ_LW", "d_Euclid", "d_Maha", "BayesOpt"
        );
        println!("  {}", "-".repeat(96));

        // Encode trajectories for each encoder + classify.
        for &ek in &encoders {
            let d = ek.dim();

            // Encode all trajectories: summaries[model][token]
            let mut summaries = [[[0.0_f32; D_MAX]; N_TOKENS]; N_CLASSES];

            for (tok_idx, _) in tokens.iter().enumerate() {
                // Model A
                let refs_a: Vec<&[f32]> = traj_a[tok_idx].iter().map(|v| v.as_slice()).collect();
                encode_natural(ek, &refs_a, &mut scratch, &mut summaries[0][tok_idx]);

                // Model B
                let refs_b: Vec<&[f32]> = traj_b[tok_idx].iter().map(|v| v.as_slice()).collect();
                encode_natural(ek, &refs_b, &mut scratch, &mut summaries[1][tok_idx]);
            }

            let result = test_classifiers(&summaries, ek, sigma);

            println!(
                "  {:>10}  {:>3}  {:>8.1}%  {:>8.1}%  {:>8.1}%  {:>6.3}  {:>9.3}  {:>9.3}  {:>7.1}%",
                ek.name(),
                d,
                result.euclidean_acc * 100.0,
                result.diagonal_acc * 100.0,
                result.mahalanobis_acc * 100.0,
                result.shrinkage_alpha,
                result.centroid_euclidean,
                result.centroid_mahalanobis,
                result.bayes_optimal * 100.0,
            );

            all_results.push(result);
        }
    }

    // ── Cross-σ comparison ────────────────────────────────────────────────
    println!();
    println!("══════════════════════════════════════════════════════════════════");
    println!("Best per-token accuracy per encoder (across all σ > 0):");
    println!();
    for &ek in &encoders {
        let best = all_results
            .iter()
            .filter(|r| r.encoder == ek && r.sigma > 0.0)
            .max_by(|a, b| katgpt_core::float_order::cmp_for_max(a.mahalanobis_acc, b.mahalanobis_acc));

        if let Some(r) = best {
            let improvement = (r.mahalanobis_acc - r.euclidean_acc) * 100.0;
            println!(
                "  {:>10}: Maha={:>5.1}%  Euclid={:>5.1}%  Δ={:>+5.1}pp  Bayes={:>5.1}%  d_M={:.2}  (σ={})",
                ek.name(),
                r.mahalanobis_acc * 100.0,
                r.euclidean_acc * 100.0,
                improvement,
                r.bayes_optimal * 100.0,
                r.centroid_mahalanobis,
                r.sigma,
            );
        }
    }

    // ── Verdict ───────────────────────────────────────────────────────────
    println!();
    println!("══════════════════════════════════════════════════════════════════");

    // Check if any Mahalanobis result beats Euclidean by ≥10pp.
    let any_maha_beats_euclid = all_results.iter().any(|r| {
        r.sigma > 0.0 && r.mahalanobis_acc > r.euclidean_acc + 0.10
    });

    // Check if any Mahalanobis result reaches ≥80%.
    let any_maha_80 = all_results.iter().any(|r| {
        r.sigma > 0.0 && r.mahalanobis_acc >= 0.80
    });

    // Check if any Bayes-optimal ceiling is ≥80%.
    let any_bayes_80 = all_results.iter().any(|r| {
        r.sigma > 0.0 && r.bayes_optimal >= 0.80
    });

    if any_maha_80 {
        println!("VERDICT: Full Mahalanobis achieves ≥80% per-token accuracy.");
        println!("The bench_016 SNR floor was CLASSIFIER-SPECIFIC, not fundamental.");
        println!("Covariance-aware classification overcomes the resolution floor.");
    } else if any_maha_beats_euclid {
        println!("VERDICT: Full Mahalanobis improves over Euclidean by ≥10pp,");
        println!("but does NOT reach 80%. The covariance structure helps partially");
        println!("but the per-token signal remains too weak for reliable classification.");
    } else if any_bayes_80 {
        println!("VERDICT: Bayes-optimal ceiling is ≥80% for some encoder/σ,");
        println!("but neither Euclidean nor Mahalanobis reaches it. The Gaussian model");
        println!("underestimates the difficulty — the noise is heavier-tailed than Gaussian.");
    } else {
        println!("VERDICT: Bayes-optimal ceiling itself is <80% for all encoders/σ.");
        println!("The per-token SNR floor is FUNDAMENTAL — the information content of");
        println!("a single token's depth trajectory is insufficient to discriminate");
        println!("perturbed vs original weights, regardless of classifier sophistication.");
        println!();
        println!("This is the definitive closure of the per-token classification question:");
        println!("no linear classifier (Euclidean, diagonal, or full Mahalanobis/LDA)");
        println!("can overcome the SNR floor. The depth trajectory captures the signal");
        println!("at the AGGREGATE level (centroid-of-tokens), but individual tokens");
        println!("scatter too widely for per-token decisions under any noise model.");
    }
    println!("══════════════════════════════════════════════════════════════════");
}
