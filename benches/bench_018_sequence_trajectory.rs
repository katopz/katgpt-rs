//! Bench 018 — Sequence Trajectory Discrimination Probe
//!
//! Proposal 011 Phase 5 T5.6e follow-up.
//!
//! ## Question
//!
//! Benches 015-017 all tested the **depth trajectory** (9 steps: embed → 8
//! layers, extracted per token with `reset()` between tokens). The per-token
//! classification question was definitively closed (bench_017: Bayes-optimal
//! ceiling ~54-56%).
//!
//! But there is a fundamentally different trajectory that was NEVER tested:
//! the **sequence trajectory** — the sequence of final hidden states across
//! a prompt's tokens with growing KV cache (no reset between tokens).
//!
//! This is the trajectory a SWE-bench attempt actually produces: the model
//! processes the issue + code context token by token, and the evolution of
//! the final hidden state across these tokens captures the model's "reading
//! dynamics". This trajectory is:
//! - Much longer (64 steps vs 9) → √N SNR boost
//! - Each step is a FULL forward pass (all layers), not just one layer
//! - The trajectory shape captures how the model processes real input
//!
//! The unit of classification is the PROMPT, not the token. Each prompt
//! produces one trajectory → one summary vector. This directly maps to
//! the per-attempt use case (one SWE-bench attempt = one prompt = one
//! trajectory).
//!
//! ## Method
//!
//! - 32 prompts (16 train + 16 test per model), each 64 tokens long
//! - Extract the sequence of final hidden states [h1, ..., h64]
//! - Encode with value-sensitive aggregate encoders:
//!   - SeqDispStats (d=8): aggregate displacement statistics
//!   - SeqStateStats (d=8): aggregate state norm statistics
//!   - SeqFullProfile (d=16): combined displacement + state stats
//! - Classify: Euclidean + Diagonal + Full Mahalanobis + Bayes-optimal ceiling
//! - Compare to bench_017's depth trajectory results

#![cfg(all(feature = "kimi_k3_loader", feature = "swe_trajectory_freeze"))]
#![allow(clippy::needless_range_loop)]

use katgpt_attn::gdn2::kda_forward::KdaWeights;
use katgpt_attn::mla::MlaWeights;
use katgpt_core::latent_trajectory_geometry::from_states_into;
use katgpt_core::swe_trajectory_freeze::GeometrySummaryEncoder;
use katgpt_rs::kimi_k3::decoder_layer::{
    KimiAttentionWeights, KimiDecoderLayerWeights, KimiFfnWeights,
};
use katgpt_rs::kimi_k3::loader::{load_kimi_k3, KimiK3ModelWeights};
use katgpt_rs::kimi_k3::model::{
    kimi_k3_forward_token_traced, KimiK3ModelConfig, KimiK3Runtime,
};
use katgpt_transformer::attn_res::AttnResWeights;
use katgpt_transformer::moe::{MoeWeights, SwiGluExpertWeights};

// ─── Constants ─────────────────────────────────────────────────────────────

/// Max encoder output dimension.
const D_MAX: usize = 16;

/// Number of classes (Model A vs Model B(σ)).
const N_CLASSES: usize = 2;

/// Number of prompts (each is one sample for classification).
const N_PROMPTS: usize = 32;

/// Training split (per model).
const N_TRAIN: usize = 16;

/// Sequence length per prompt (tokens processed sequentially with growing KV cache).
const SEQ_LEN: usize = 64;

/// Truncated vocab for token ID generation.
const BENCH_VOCAB: usize = 512;

/// Perturbation σ levels.
const SIGMA_LEVELS: &[f32] = &[0.0, 0.01, 0.05, 0.1, 0.5];

// ─── Deterministic LCG (copied from bench_015/016/017) ─────────────────────

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

// ─── Weight perturbation (copied from bench_016/017) ───────────────────────

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

// ─── Sequence trajectory extraction ────────────────────────────────────────
//
// KEY DIFFERENCE from bench_012-017: we do NOT reset the runtime between
// tokens within a prompt. The KV cache grows, and each token attends to
// all previous tokens. We capture the FINAL hidden state (after all layers
// + output attn-res + final RMSNorm) at each step — the return value of
// `kimi_k3_forward_token_traced`.

struct SeqExtractScratch {
    /// The sequence of final hidden states [h1, h2, ..., hN].
    /// Each is a copy of `runtime.hidden` after the full forward pass.
    seq_states: Vec<Vec<f32>>,
    /// Throwaway buffer for the depth trajectory (we don't use it).
    depth_traj: Vec<Vec<f32>>,
    /// Scratch for geometry encoder.
    disp_curr: Vec<f32>,
    disp_prev: Vec<f32>,
}

impl SeqExtractScratch {
    fn new(hidden_dim: usize) -> Self {
        Self {
            seq_states: Vec::with_capacity(SEQ_LEN),
            depth_traj: Vec::with_capacity(9),
            disp_curr: Vec::with_capacity(hidden_dim),
            disp_prev: Vec::with_capacity(hidden_dim),
        }
    }

    /// Extract the sequence trajectory for one prompt.
    ///
    /// Resets the runtime ONCE (for the new prompt), then processes all
    /// tokens sequentially WITHOUT reset. Captures the final hidden state
    /// at each step.
    fn extract_sequence(
        &mut self,
        config: &KimiK3ModelConfig,
        weights: &KimiK3ModelWeights,
        runtime: &mut KimiK3Runtime,
        tokens: &[u32],
    ) {
        runtime.reset();
        self.seq_states.clear();

        for &tok in tokens {
            self.depth_traj.clear();
            let final_hidden = kimi_k3_forward_token_traced(
                config,
                weights,
                runtime,
                tok,
                &mut self.depth_traj,
            );
            // Capture the final hidden state (after all layers + norm).
            self.seq_states.push(final_hidden.to_vec());
        }
    }
}

// ─── Sequence trajectory encoders ──────────────────────────────────────────
//
// Each encoder computes aggregate statistics from the sequence of final
// hidden states. Unlike the depth trajectory (9 steps → 8 features), the
// sequence trajectory has 64 steps, so we use AGGREGATE statistics rather
// than per-step features.

/// Aggregate displacement statistics (d=8).
///
/// Captures how the hidden state CHANGES between consecutive tokens:
/// - mean/std/max of per-step displacement L2 norms
/// - mean/std of per-step displacement mean (signed)
/// - total trajectory length
/// - drift ratio (net displacement / total path)
fn encode_seq_disp_stats(states: &[Vec<f32>], out: &mut [f32; D_MAX]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let n = states.len();
    if n < 2 {
        return;
    }
    let dim = states[0].len();
    let n_disps = n - 1;

    let mut disp_norms = vec![0.0_f32; n_disps];
    let mut disp_means = vec![0.0_f32; n_disps];
    let mut total_len = 0.0_f32;

    // Net displacement vector (last - first).
    let mut net_disp_sq = 0.0_f32;
    for j in 0..dim {
        let diff = states[n - 1][j] - states[0][j];
        net_disp_sq += diff * diff;
    }
    let net_disp = net_disp_sq.sqrt();

    for i in 0..n_disps {
        let mut sum_sq = 0.0_f32;
        let mut sum = 0.0_f32;
        for j in 0..dim {
            let diff = states[i + 1][j] - states[i][j];
            sum_sq += diff * diff;
            sum += diff;
        }
        disp_norms[i] = sum_sq.sqrt();
        disp_means[i] = sum / dim as f32;
        total_len += disp_norms[i];
    }

    // Aggregate stats.
    let mean_norm = disp_norms.iter().sum::<f32>() / n_disps as f32;
    let var_norm = disp_norms.iter().map(|x| (x - mean_norm).powi(2)).sum::<f32>()
        / n_disps as f32;
    let std_norm = var_norm.sqrt();
    let max_norm = disp_norms.iter().cloned().fold(0.0_f32, f32::max);

    let mean_disp = disp_means.iter().sum::<f32>() / n_disps as f32;
    let var_disp = disp_means.iter().map(|x| (x - mean_disp).powi(2)).sum::<f32>()
        / n_disps as f32;
    let std_disp = var_disp.sqrt();

    let drift_ratio = if total_len > 1e-12 {
        net_disp / total_len
    } else {
        0.0
    };

    out[0] = mean_norm;
    out[1] = std_norm;
    out[2] = max_norm;
    out[3] = total_len;
    out[4] = mean_disp;
    out[5] = std_disp;
    out[6] = net_disp;
    out[7] = drift_ratio;
}

/// Aggregate state norm statistics (d=8).
///
/// Captures the GROWTH profile of the hidden state magnitude:
/// - mean/std/max/min of per-step L2 norms
/// - initial/final norm
/// - norm growth ratio
/// - mean cosine similarity between consecutive states
fn encode_seq_state_stats(states: &[Vec<f32>], out: &mut [f32; D_MAX]) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let n = states.len();
    if n == 0 {
        return;
    }
    let dim = states[0].len();

    let mut norms = vec![0.0_f32; n];
    for i in 0..n {
        let mut sum_sq = 0.0_f32;
        for j in 0..dim {
            sum_sq += states[i][j] * states[i][j];
        }
        norms[i] = sum_sq.sqrt();
    }

    let mean_norm = norms.iter().sum::<f32>() / n as f32;
    let var_norm = norms.iter().map(|x| (x - mean_norm).powi(2)).sum::<f32>() / n as f32;
    let std_norm = var_norm.sqrt();
    let max_norm = norms.iter().cloned().fold(0.0_f32, f32::max);
    let min_norm = norms.iter().cloned().fold(f32::INFINITY, f32::min);
    let initial_norm = norms[0];
    let final_norm = norms[n - 1];
    let norm_ratio = if initial_norm > 1e-12 {
        final_norm / initial_norm
    } else {
        0.0
    };

    // Mean cosine similarity between consecutive states.
    let mut mean_cos = 0.0_f32;
    let mut cos_count = 0usize;
    for i in 0..n.saturating_sub(1) {
        let mut dot = 0.0_f32;
        let mut norm_a = 0.0_f32;
        let mut norm_b = 0.0_f32;
        for j in 0..dim {
            dot += states[i][j] * states[i + 1][j];
            norm_a += states[i][j] * states[i][j];
            norm_b += states[i + 1][j] * states[i + 1][j];
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom > 1e-12 {
            mean_cos += dot / denom;
            cos_count += 1;
        }
    }
    if cos_count > 0 {
        mean_cos /= cos_count as f32;
    }

    out[0] = mean_norm;
    out[1] = std_norm;
    out[2] = max_norm;
    out[3] = min_norm;
    out[4] = initial_norm;
    out[5] = final_norm;
    out[6] = norm_ratio;
    out[7] = mean_cos;
}

/// Combined displacement + state profile (d=16).
///
/// Concatenates SeqDispStats + SeqStateStats for a richer feature set.
fn encode_seq_full_profile(states: &[Vec<f32>], out: &mut [f32; D_MAX]) {
    let mut disp = [0.0_f32; D_MAX];
    let mut state = [0.0_f32; D_MAX];
    encode_seq_disp_stats(states, &mut disp);
    encode_seq_state_stats(states, &mut state);
    out[..8].copy_from_slice(&disp[..8]);
    out[8..16].copy_from_slice(&state[..8]);
}

/// Geometry encoder baseline (from_states on the sequence trajectory).
/// This tests whether the SHAPED features that failed in bench_015 work
/// better on the longer sequence trajectory.
fn encode_seq_geometry(
    states: &[Vec<f32>],
    scratch: &mut SeqExtractScratch,
    encoder: &GeometrySummaryEncoder,
    out: &mut [f32; D_MAX],
) {
    for v in out.iter_mut() {
        *v = 0.0;
    }
    let refs: Vec<&[f32]> = states.iter().map(|v| v.as_slice()).collect();
    let geom = from_states_into(&refs, &mut scratch.disp_curr, &mut scratch.disp_prev);
    let mut geom_out = [0.0_f32; 8];
    encoder.encode_into(&geom, &mut geom_out);
    // Copy the 8 geometry features into the first 8 slots.
    out[..8].copy_from_slice(&geom_out[..8]);
}

// ─── Encoder dispatch ──────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum EncoderKind {
    SeqDispStats,
    SeqStateStats,
    SeqFullProfile,
    Geometry,
}

impl EncoderKind {
    fn name(self) -> &'static str {
        match self {
            Self::SeqDispStats => "SeqDispStats",
            Self::SeqStateStats => "SeqStateStats",
            Self::SeqFullProfile => "SeqFullProfile",
            Self::Geometry => "SeqGeometry",
        }
    }

    fn dim(self) -> usize {
        match self {
            Self::SeqDispStats => 8,
            Self::SeqStateStats => 8,
            Self::SeqFullProfile => 16,
            Self::Geometry => 8,
        }
    }
}

fn encode(
    kind: EncoderKind,
    states: &[Vec<f32>],
    scratch: &mut SeqExtractScratch,
    geom_encoder: &GeometrySummaryEncoder,
    out: &mut [f32; D_MAX],
) {
    match kind {
        EncoderKind::SeqDispStats => encode_seq_disp_stats(states, out),
        EncoderKind::SeqStateStats => encode_seq_state_stats(states, out),
        EncoderKind::SeqFullProfile => encode_seq_full_profile(states, out),
        EncoderKind::Geometry => encode_seq_geometry(states, scratch, geom_encoder, out),
    }
}

// ─── Covariance + classifiers (adapted from bench_017) ─────────────────────

fn pooled_covariance_ledoit_wolf(
    train_features: &[[[f32; D_MAX]; N_TRAIN]; N_CLASSES],
    d: usize,
    cov_out: &mut [f32],
    diag_var_out: &mut [f32],
) -> f32 {
    let n_per_class = N_TRAIN;
    let n_total = N_CLASSES * N_TRAIN;
    let dof = n_total - N_CLASSES;

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

    for ij in 0..d * d {
        cov_out[ij] = 0.0;
    }
    for k in 0..N_CLASSES {
        for i in 0..n_per_class {
            let mut r = [0.0_f32; D_MAX];
            for j in 0..d {
                r[j] = train_features[k][i][j] - class_means[k][j];
            }
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

    for j in 0..d {
        diag_var_out[j] = cov_out[j * d + j];
    }

    let m: f32 = (0..d).map(|j| cov_out[j * d + j]).sum::<f32>() / d as f32;

    let mut d_sq = 0.0_f32;
    for a in 0..d {
        for b in 0..d {
            let target = if a == b { m } else { 0.0 };
            let diff = cov_out[a * d + b] - target;
            d_sq += diff * diff;
        }
    }
    d_sq /= d as f32;

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

    let b_sq = b_bar_sq.min(d_sq);
    let alpha = if d_sq > 1e-20 { b_sq / d_sq } else { 1.0 };

    for a in 0..d {
        for b in 0..d {
            let target = if a == b { m } else { 0.0 };
            cov_out[a * d + b] = alpha * target + (1.0 - alpha) * cov_out[a * d + b];
        }
    }

    alpha
}

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

fn cholesky_decompose(a: &[f32], d: usize, l: &mut [f32]) -> bool {
    for ij in 0..d * d {
        l[ij] = 0.0;
    }
    for j in 0..d {
        let mut sum = a[j * d + j];
        for k in 0..j {
            sum -= l[j * d + k] * l[j * d + k];
        }
        if sum <= 0.0 {
            return false;
        }
        let diag = sum.sqrt();
        l[j * d + j] = diag;
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

fn cholesky_with_jitter(a: &mut [f32], d: usize, l: &mut [f32]) -> u32 {
    let mut jitter = 0.0_f32;
    let mut attempts = 0u32;
    loop {
        if cholesky_decompose(a, d, l) {
            return attempts;
        }
        attempts += 1;
        if jitter > 0.0 {
            for j in 0..d {
                a[j * d + j] -= jitter;
            }
        }
        jitter = if jitter == 0.0 { 1e-6 } else { jitter * 10.0 };
        if jitter > 100.0 {
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

fn mahalanobis_sq(l: &[f32], d: usize, x: &[f32], mu: &[f32]) -> f32 {
    let mut y = [0.0_f32; D_MAX];
    for i in 0..d {
        let mut sum = x[i] - mu[i];
        for k in 0..i {
            sum -= l[i * d + k] * y[k];
        }
        let diag = l[i * d + i];
        y[i] = if diag.abs() > 1e-15 { sum / diag } else { 0.0 };
    }
    let mut dist_sq = 0.0;
    for i in 0..d {
        dist_sq += y[i] * y[i];
    }
    dist_sq
}

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

fn normal_cdf(x: f64) -> f64 {
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

// ─── Main test function ────────────────────────────────────────────────────

fn test_classifiers(
    summaries: &[[[f32; D_MAX]; N_PROMPTS]; N_CLASSES],
    encoder: EncoderKind,
    sigma: f32,
) -> ClassifierResult {
    let d = encoder.dim();

    let mut train_features = [[[0.0_f32; D_MAX]; N_TRAIN]; N_CLASSES];
    for k in 0..N_CLASSES {
        for i in 0..N_TRAIN {
            train_features[k][i] = summaries[k][i];
        }
    }

    let class_means = compute_class_means(&train_features, d);

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

    let mut cov = vec![0.0_f32; d * d];
    let mut diag_var = [0.0_f32; D_MAX];
    let alpha = pooled_covariance_ledoit_wolf(&train_features, d, &mut cov, &mut diag_var);

    let mut l = vec![0.0_f32; d * d];
    let _jitter_attempts = cholesky_with_jitter(&mut cov, d, &mut l);

    let mut cent_euclid_sq = 0.0_f32;
    for j in 0..d {
        let diff = class_means[0][j] - class_means[1][j];
        cent_euclid_sq += diff * diff;
    }
    let centroid_euclidean = cent_euclid_sq.sqrt();

    let mut delta = [0.0_f32; D_MAX];
    for j in 0..d {
        delta[j] = class_means[0][j] - class_means[1][j];
    }
    let centroid_mahalanobis_sq = mahalanobis_sq(&l, d, &delta, &[0.0_f32; D_MAX]);
    let centroid_mahalanobis = centroid_mahalanobis_sq.sqrt();

    let bayes_optimal = normal_cdf(centroid_mahalanobis as f64 / 2.0) as f32;

    let n_test = N_PROMPTS - N_TRAIN;
    let mut euclid_correct = 0usize;
    let mut diag_correct = 0usize;
    let mut maha_correct = 0usize;

    for k in 0..N_CLASSES {
        for tok_idx in N_TRAIN..N_PROMPTS {
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
    println!("║  P011 follow-up — sequence trajectory discrimination (bench_018)  ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    let config = KimiK3ModelConfig::kimi_k3_0_40b();
    let d_model = config.hidden_size;
    println!("Config: D_model={d_model}, layers={}", config.num_layers);
    println!("Sequence length: {SEQ_LEN} tokens per prompt (KV cache grows)");
    println!("Prompts: {N_PROMPTS} ({N_TRAIN} train + {} test per model)",
        N_PROMPTS - N_TRAIN);
    println!("Sigma levels: {SIGMA_LEVELS:?}");
    println!();
    println!("KEY DIFFERENCE from bench_012-017:");
    println!("  - Depth trajectory: 9 steps per token (embed -> 8 layers)");
    println!("  - Sequence trajectory: {SEQ_LEN} steps per prompt (final hidden");
    println!("    state across tokens, KV cache growing)");
    println!("  - Unit of classification: PROMPT (not token)");
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

    // ── Generate prompt token sequences ───────────────────────────────────
    // Each prompt is a deterministic sequence of SEQ_LEN token IDs.
    // Different prompts use different token sequences for within-class variance.
    let prompts: Vec<Vec<u32>> = (0..N_PROMPTS)
        .map(|p| {
            (0..SEQ_LEN)
                .map(|i| {
                    ((p as u32).wrapping_mul(31)
                        .wrapping_add((i as u32).wrapping_mul(7))
                        .wrapping_add(3))
                        % (BENCH_VOCAB as u32)
                })
                .collect()
        })
        .collect();

    // ── Shared setup ──────────────────────────────────────────────────────
    let max_seq_len = SEQ_LEN;
    let mut runtime_a = KimiK3Runtime::new(&config, max_seq_len);
    let mut runtime_b = KimiK3Runtime::new(&config, max_seq_len);
    let geom_encoder = GeometrySummaryEncoder::default_depth_trajectory();
    let mut scratch = SeqExtractScratch::new(d_model);

    let encoders = [
        EncoderKind::SeqDispStats,
        EncoderKind::SeqStateStats,
        EncoderKind::SeqFullProfile,
        EncoderKind::Geometry,
    ];

    // ── Cache Model A sequence trajectories (extracted once, reused) ──────
    println!("Extracting Model A sequence trajectories ({N_PROMPTS} prompts × {SEQ_LEN} tokens) ...");
    let t0 = std::time::Instant::now();
    let mut traj_a: Vec<Vec<Vec<f32>>> = Vec::with_capacity(N_PROMPTS);
    for prompt in &prompts {
        scratch.extract_sequence(&config, &weights_a, &mut runtime_a, prompt);
        traj_a.push(scratch.seq_states.clone());
    }
    println!("  done ({:.1}s)", t0.elapsed().as_secs_f64());

    // ── Run the sweep ─────────────────────────────────────────────────────
    let mut all_results: Vec<ClassifierResult> = Vec::new();

    for &sigma in SIGMA_LEVELS {
        let mut weights_b = weights_a.clone();
        perturb_model(&mut weights_b, sigma);

        // Extract Model B sequence trajectories for this σ.
        print!("Extracting Model B sequence trajectories (σ={sigma}) ... ");
        let t0 = std::time::Instant::now();
        let mut traj_b: Vec<Vec<Vec<f32>>> = Vec::with_capacity(N_PROMPTS);
        for prompt in &prompts {
            scratch.extract_sequence(&config, &weights_b, &mut runtime_b, prompt);
            traj_b.push(scratch.seq_states.clone());
        }
        println!("done ({:.1}s)", t0.elapsed().as_secs_f64());

        println!();
        println!("── σ = {sigma} ──────────────────────────────────────────");
        println!(
            "  {:>14}  {:>3}  {:>9}  {:>9}  {:>9}  {:>6}  {:>9}  {:>9}  {:>8}",
            "encoder", "d", "Euclidean", "DiagMaha", "FullMaha", "λ_LW", "d_Euclid", "d_Maha", "BayesOpt"
        );
        println!("  {}", "-".repeat(100));

        for &ek in &encoders {
            let d = ek.dim();

            let mut summaries = [[[0.0_f32; D_MAX]; N_PROMPTS]; N_CLASSES];

            for (prompt_idx, _) in prompts.iter().enumerate() {
                encode(ek, &traj_a[prompt_idx], &mut scratch, &geom_encoder, &mut summaries[0][prompt_idx]);
                encode(ek, &traj_b[prompt_idx], &mut scratch, &geom_encoder, &mut summaries[1][prompt_idx]);
            }

            let result = test_classifiers(&summaries, ek, sigma);

            println!(
                "  {:>14}  {:>3}  {:>8.1}%  {:>8.1}%  {:>8.1}%  {:>6.3}  {:>9.3}  {:>9.3}  {:>7.1}%",
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
        println!();
    }

    // ── Cross-σ comparison ────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════════");
    println!("Best per-prompt accuracy per encoder (across all σ > 0):");
    println!();
    for &ek in &encoders {
        let best = all_results
            .iter()
            .filter(|r| r.encoder == ek && r.sigma > 0.0)
            .max_by(|a, b| katgpt_core::float_order::cmp_for_max(a.mahalanobis_acc, b.mahalanobis_acc));

        if let Some(r) = best {
            let improvement = (r.mahalanobis_acc - r.euclidean_acc) * 100.0;
            println!(
                "  {:>14}: Maha={:>5.1}%  Euclid={:>5.1}%  Δ={:>+5.1}pp  Bayes={:>5.1}%  d_M={:.3}  (σ={})",
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
    println!();

    // ── Comparison to bench_017 depth trajectory ──────────────────────────
    println!("══════════════════════════════════════════════════════════════════");
    println!("Comparison: sequence trajectory (this bench) vs depth trajectory (bench_017):");
    println!();
    println!("  bench_017 (depth, 9 steps/token, 128 tokens):");
    println!("    Best Mahalanobis: 56.2%  Bayes-optimal: 55.7%  d_M: 0.285");
    println!();
    println!("  bench_018 (sequence, {SEQ_LEN} steps/prompt, {N_PROMPTS} prompts):");
    for &ek in &encoders {
        if let Some(r) = all_results.iter().find(|r| r.encoder == ek && r.sigma == 0.5) {
            println!(
                "    {:>14}: Maha={:>5.1}%  Bayes={:>5.1}%  d_M={:.3}  d_E={:.3}",
                ek.name(),
                r.mahalanobis_acc * 100.0,
                r.bayes_optimal * 100.0,
                r.centroid_mahalanobis,
                r.centroid_euclidean,
            );
        }
    }
    println!();

    // ── Verdict ───────────────────────────────────────────────────────────
    let any_maha_80 = all_results.iter().any(|r| {
        r.sigma > 0.0 && r.mahalanobis_acc >= 0.80
    });

    let any_bayes_80 = all_results.iter().any(|r| {
        r.sigma > 0.0 && r.bayes_optimal >= 0.80
    });

    let any_bayes_70 = all_results.iter().any(|r| {
        r.sigma > 0.0 && r.bayes_optimal >= 0.70
    });

    // Compare best d_M to bench_017's best (0.285 for DispStats).
    let best_dm = all_results
        .iter()
        .filter(|r| r.sigma == 0.5)
        .map(|r| r.centroid_mahalanobis)
        .fold(0.0_f32, f32::max);

    let bench_017_best_dm = 0.285_f32;
    let dm_ratio = if bench_017_best_dm > 1e-12 {
        best_dm / bench_017_best_dm
    } else {
        0.0
    };

    println!("══════════════════════════════════════════════════════════════════");
    if any_maha_80 {
        println!("VERDICT: Sequence trajectory Mahalanobis achieves ≥80% per-prompt");
        println!("accuracy. The longer trajectory ({SEQ_LEN} steps) provides enough √N SNR");
        println!("boost to overcome the bench_017 depth trajectory floor.");
        println!("Layer 4 per-attempt freezing is VALIDATED for value-level discrimination.");
    } else if any_bayes_80 {
        println!("VERDICT: Bayes-optimal ceiling is ≥80% for some encoder/σ, but");
        println!("Mahalanobis doesn't reach it. The Gaussian model holds but the");
        println!("classifier needs more training data or a non-linear approach.");
    } else if any_bayes_70 {
        println!("VERDICT: Bayes-optimal ceiling is ≥70% — IMPROVEMENT over bench_017's");
        println!("~55%, but still below 80%. The sequence trajectory helps but the");
        println!("per-prompt signal is insufficient for reliable discrimination.");
        println!("Best d_M = {best_dm:.3} (vs bench_017's 0.285, ratio = {dm_ratio:.1}×)");
    } else {
        println!("VERDICT: Bayes-optimal ceiling <70% for all encoders at σ=0.5.");
        println!("The sequence trajectory ({SEQ_LEN} steps) does NOT provide enough");
        println!("SNR boost over the depth trajectory (9 steps) to overcome the");
        println!("fundamental information floor.");
        println!("Best d_M = {best_dm:.3} (vs bench_017's 0.285, ratio = {dm_ratio:.1}×)");
        println!();
        println!("This closes the sequence trajectory hypothesis: the information");
        println!("deficit is not a trajectory-length problem — it's a fundamental");
        println!("property of the weight perturbation signal being too weak relative");
        println!("to the input-dependent variation across prompts.");
    }
    println!("══════════════════════════════════════════════════════════════════");
}
