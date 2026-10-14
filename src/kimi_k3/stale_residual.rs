//! Stale-residual speculative layer execution — the Kimi-K3 simulator
//! (Issue 691 / Research 508, arXiv:2608.23841 §6.3 Approach A + B).
//!
//! This module DOES what the katgpt-core `stale_residual` analysis module only
//! scores: it runs layer ℓ+1 on the **stale** residual `x_in^ℓ` while the true
//! forward's captured trace provides the rollback ground truth. Zero GPU, real
//! `model.safetensors` weights via the `kimi_k3` loader.
//!
//! # Execution model (what "speculative" means here)
//!
//! The true decode of position p runs layers 0..L−1 sequentially on the
//! running `prefix_sum` (the residual stream). The pipelined scheme starts
//! layer ℓ+1 on the stale `x_in^ℓ` (layer ℓ's INPUT) before layer ℓ's
//! contribution δℓ lands:
//!
//! - **Accept** (`‖δℓ‖/‖x_in^ℓ‖ < θ`): the speculative tail's output is used;
//!   in the real scheme the stale-written KV/KDA state persists.
//! - **Reject**: rollback + recompute on `x_out^ℓ` (consumer-side; here we
//!   simply compare against the captured true tail).
//!
//! [`capture_forward_token`] records the true run's per-layer stream states;
//! [`replay_from_layer`] re-executes layers ℓ+1..L−1 (+ output attn-res +
//! final norm + LM head) from an arbitrary starting hidden + block state on a
//! snapshot-restored runtime, so the speculative tail sees exactly the caches
//! a pipelined engine would have: past positions cached, current position
//! written from the stale input (the KV/KDA hazard the paper flags, modeled
//! rather than waved away).
//!
//! # K3 residual-stream specifics (why block_state is snapshotted per layer)
//!
//! K3's attention-residual blocks (block_size 4: boundaries at layers 0 and 4)
//! PUSH the prefix_sum into `block_state` and ZERO it at boundaries — the
//! stream restarts per block and mixes with prior blocks via softmax
//! attention-res. `x_in^{ℓ+1}` is therefore the prefix_sum at layer ℓ+1's
//! ENTRY (post-push semantics included), and a replay starting at ℓ+1 must
//! restore the block_state as of that entry. [`TokenCapture`] snapshots both
//! per layer.
//!
//! # T3 predictors (Approach B — closed-form, modelless by mandate)
//!
//! Two feature sets, both computable at speculation time (before δℓ exists):
//! - **Router-logit** (the paper's framing): layer ℓ's MoE router logits
//!   evaluated on `RMSNorm_γ(x_in^ℓ)` — an approximation of the true router
//!   input (which sees layer ℓ's attention output), i.e. the paper's
//!   pre-dense routing anchor.
//! - **x_in-linear** (the linear-predictability ceiling): δℓ ≈ W·x_in^ℓ.
//! Both fit via the OLS substrate `katgpt_attn_match::value_fitter`
//! (blocked Cholesky + jitter escalation — consumed, not reimplemented).

use crate::kimi_k3::decoder_layer::{KimiFfnWeights, kimi_decoder_layer_forward};
use crate::kimi_k3::loader::KimiK3ModelWeights;
use crate::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime};
use katgpt_attn::gdn2::kda_forward::KdaLayerCache;
use katgpt_attn_match::value_fitter::{ValueFitConfig, fit_cv_least_squares};
use katgpt_core::stale_residual::SpecOutcome;
use katgpt_core::types::math::rmsnorm_with_gamma_eps;
use katgpt_transformer::attn_res::apply_attn_res;

// ─── Runtime snapshot ──────────────────────────────────────────────────────

/// Snapshot of the stateful half of [`KimiK3Runtime`].
///
/// Scratches are excluded (overwritten per use, carry no cross-token state).
/// MLA caches snapshot the live prefix `[0..seq_len]` (the tail beyond
/// `seq_len` is dead). KDA caches are `Clone`. `block_state` residuals are
/// cloned Vecs (the `pool` slot-reuse field is reconstruction detail —
/// restoring sets `residuals` directly).
pub struct RuntimeSnapshot {
    mla: Vec<MlaSnap>,
    kda: Vec<KdaSnap>,
    block_residuals: Vec<Vec<f32>>,
}

struct MlaSnap {
    layer_idx: usize,
    latent: Vec<f32>,
    rope: Vec<f32>,
    seq_len: usize,
    d_c: usize,
    d_r: usize,
}

struct KdaSnap {
    layer_idx: usize,
    cache: KdaLayerCache,
}

impl RuntimeSnapshot {
    /// Capture the full stateful runtime (all caches + block state).
    ///
    /// `config` supplies the MLA strides (`d_c`, `d_r`) — the cache exposes
    /// flat buffers whose stride is not derivable through the public API
    /// from a cold cache.
    pub fn capture(config: &KimiK3ModelConfig, runtime: &KimiK3Runtime) -> Self {
        let mut mla = Vec::new();
        let mut kda = Vec::new();
        for (idx, layer) in runtime.layers.iter().enumerate() {
            if config.is_mla_layer(idx) {
                let crate::kimi_k3::decoder_layer::KimiAttentionState::Mla(cache) =
                    &layer.attn_state
                else {
                    panic!("config says MLA layer {idx} but runtime holds KDA");
                };
                let d_c = config.mla_config.kv_lora_rank;
                let d_r = config.mla_config.qk_rope_head_dim;
                mla.push(MlaSnap {
                    layer_idx: idx,
                    latent: cache.latent_kv[..cache.seq_len * d_c].to_vec(),
                    rope: cache.rope_key[..cache.seq_len * d_r].to_vec(),
                    seq_len: cache.seq_len,
                    d_c,
                    d_r,
                });
            } else {
                let crate::kimi_k3::decoder_layer::KimiAttentionState::Kda(cache) =
                    &layer.attn_state
                else {
                    panic!("config says KDA layer {idx} but runtime holds MLA");
                };
                kda.push(KdaSnap {
                    layer_idx: idx,
                    cache: cache.clone(),
                });
            }
        }
        Self {
            mla,
            kda,
            block_residuals: runtime.block_state.residuals().to_vec(),
        }
    }

    /// Restore caches + block state into the runtime.
    pub fn restore_into(&self, runtime: &mut KimiK3Runtime) {
        for snap in &self.mla {
            let Some(layer) = runtime.layers.get_mut(snap.layer_idx) else {
                continue;
            };
            let crate::kimi_k3::decoder_layer::KimiAttentionState::Mla(cache) =
                &mut layer.attn_state
            else {
                continue;
            };
            let live = snap.seq_len;
            cache.latent_kv[..live * snap.d_c].copy_from_slice(&snap.latent[..live * snap.d_c]);
            cache.rope_key[..live * snap.d_r].copy_from_slice(&snap.rope[..live * snap.d_r]);
            cache.seq_len = snap.seq_len;
        }
        for snap in &self.kda {
            let Some(layer) = runtime.layers.get_mut(snap.layer_idx) else {
                continue;
            };
            let crate::kimi_k3::decoder_layer::KimiAttentionState::Kda(cache) =
                &mut layer.attn_state
            else {
                continue;
            };
            *cache = snap.cache.clone();
        }
        restore_block_residuals(&self.block_residuals, runtime);
    }
}

fn restore_block_residuals(residuals: &[Vec<f32>], runtime: &mut KimiK3Runtime) {
    // restore() (not a raw clear+extend) so the per-summary Σv² cache is
    // recomputed — apply_attn_res zips scores against BOTH vectors, and an
    // empty cache silently no-ops the whole mix.
    runtime.block_state.restore(residuals);
}

/// Rewind ONLY layers `> delay_layer` to their pre-token state (MLA seq_len
/// rewind + prefix restore; KDA state restore). Layers `≤ delay_layer` keep
/// their current (post-capture) state — the pipelined-engine semantics for
/// the persistent-hazard arm: the speculative tail rewrites position p,
/// while the completed prefix layers' position-p entries are correct.
fn rewind_tail_to_pre(pre: &RuntimeSnapshot, delay_layer: usize, runtime: &mut KimiK3Runtime) {
    for snap in &pre.mla {
        if snap.layer_idx <= delay_layer {
            continue;
        }
        let Some(layer) = runtime.layers.get_mut(snap.layer_idx) else {
            continue;
        };
        let crate::kimi_k3::decoder_layer::KimiAttentionState::Mla(cache) = &mut layer.attn_state
        else {
            continue;
        };
        let live = snap.seq_len;
        cache.latent_kv[..live * snap.d_c].copy_from_slice(&snap.latent[..live * snap.d_c]);
        cache.rope_key[..live * snap.d_r].copy_from_slice(&snap.rope[..live * snap.d_r]);
        cache.seq_len = snap.seq_len;
    }
    for snap in &pre.kda {
        if snap.layer_idx <= delay_layer {
            continue;
        }
        let Some(layer) = runtime.layers.get_mut(snap.layer_idx) else {
            continue;
        };
        let crate::kimi_k3::decoder_layer::KimiAttentionState::Kda(cache) = &mut layer.attn_state
        else {
            continue;
        };
        *cache = snap.cache.clone();
    }
}

// ─── Capture forward ───────────────────────────────────────────────────────

/// The true run's per-layer residual-stream record for one token.
pub struct TokenCapture {
    /// `x_in^ℓ` — prefix_sum at layer ℓ's ENTRY (pre-forward of layer ℓ).
    /// `[n_layer][d]`. `x_in^0` = the token embedding.
    pub x_in: Vec<Vec<f32>>,
    /// `x_out^{L−1}` — the hidden after the LAST layer's residual add,
    /// BEFORE output attn-res (completes the last layer's δ row).
    pub x_out_final: Vec<f32>,
    /// Block-state residuals at each layer's entry. `[n_layer]` entries
    /// (entry-of-ℓ snapshot).
    pub block_at_entry: Vec<Vec<Vec<f32>>>,
    /// Final true logits `[vocab]`.
    pub logits: Vec<f32>,
    /// Snapshot of the runtime AFTER the full true forward (for cheap
    /// restore between replays without re-running the true path).
    pub post_state: RuntimeSnapshot,
    /// Snapshot of the runtime BEFORE the true forward (pre-token caches).
    pub pre_state: RuntimeSnapshot,
}

/// True forward of one token with per-layer capture.
///
/// Bit-identical to [`crate::kimi_k3::model::kimi_k3_forward_token`] by
/// construction (same layer calls in the same order; capture clones happen
/// between layers) — asserted by the tests + bench G1.
pub fn capture_forward_token(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    token_id: u32,
) -> TokenCapture {
    let d = config.hidden_size;
    let pre_state = RuntimeSnapshot::capture(config, runtime);
    runtime.block_state.clear();

    let embed_start = token_id as usize * d;
    runtime
        .hidden
        .copy_from_slice(&weights.embed_weight[embed_start..embed_start + d]);

    let mut x_in = Vec::with_capacity(config.num_layers);
    let mut block_at_entry = Vec::with_capacity(config.num_layers);

    for (layer_idx, layer_w) in weights.layers.iter().enumerate() {
        x_in.push(runtime.hidden.clone());
        block_at_entry.push(runtime.block_state.residuals().to_vec());
        let layer_cfg = config.layer_config(layer_idx);
        let layer_rt = &mut runtime.layers[layer_idx];
        kimi_decoder_layer_forward(
            layer_idx,
            &layer_cfg,
            layer_w,
            &mut layer_rt.attn_state,
            &mut layer_rt.attn_scratch,
            &mut layer_rt.ffn_scratch,
            &mut layer_rt.attn_res_self_scratch,
            &mut layer_rt.attn_res_mlp_scratch,
            &mut runtime.block_state,
            Some(&mut runtime.rope_freqs),
            &mut runtime.hidden,
            &mut runtime.scratch_hidden,
        );
    }

    // Output attn-res + final norm + LM head (mirrors forward_decoder_to_logits).
    let x_out_final = runtime.hidden.clone();
    if !runtime.block_state.is_empty() {
        let mixed = apply_attn_res(
            &config.attn_res_config,
            &weights.output_attn_res,
            &runtime.block_state,
            &mut runtime.output_attn_res_scratch,
            &runtime.hidden,
        );
        runtime.hidden.copy_from_slice(mixed);
    }
    rmsnorm_with_gamma_eps(
        &mut runtime.hidden,
        &weights.final_norm_weight,
        config.rms_eps as f64,
    );
    katgpt_core::simd::simd_matmul_rows(
        &mut runtime.logits,
        &weights.lm_head_weight,
        &runtime.hidden,
        config.vocab_size,
        d,
    );

    TokenCapture {
        x_in,
        x_out_final,
        block_at_entry,
        logits: runtime.logits.clone(),
        post_state: RuntimeSnapshot::capture(config, runtime),
        pre_state,
    }
}

/// Re-execute layers `start_layer..L` (+ output attn-res + final norm + LM
/// head) from `start_hidden` with `start_block` as the block state.
///
/// The runtime's caches must already be in the desired state (the simulator
/// restores [`TokenCapture::pre_state`] before calling — so layers
/// `start_layer..` see past positions cached and append/evolve the current
/// position from the stale input, exactly as a pipelined engine would).
/// Writes `runtime.logits`; returns them as an owned Vec for capture-free
/// comparison.
pub fn replay_from_layer(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    runtime: &mut KimiK3Runtime,
    start_layer: usize,
    start_hidden: &[f32],
    start_block: &[Vec<f32>],
) -> Vec<f32> {
    let d = config.hidden_size;
    debug_assert!(start_layer <= config.num_layers);
    runtime.hidden[..d].copy_from_slice(start_hidden);
    restore_block_residuals(start_block, runtime);

    for layer_idx in start_layer..config.num_layers {
        let layer_cfg = config.layer_config(layer_idx);
        let layer_w = &weights.layers[layer_idx];
        let layer_rt = &mut runtime.layers[layer_idx];
        kimi_decoder_layer_forward(
            layer_idx,
            &layer_cfg,
            layer_w,
            &mut layer_rt.attn_state,
            &mut layer_rt.attn_scratch,
            &mut layer_rt.ffn_scratch,
            &mut layer_rt.attn_res_self_scratch,
            &mut layer_rt.attn_res_mlp_scratch,
            &mut runtime.block_state,
            Some(&mut runtime.rope_freqs),
            &mut runtime.hidden,
            &mut runtime.scratch_hidden,
        );
    }

    if !runtime.block_state.is_empty() {
        let mixed = apply_attn_res(
            &config.attn_res_config,
            &weights.output_attn_res,
            &runtime.block_state,
            &mut runtime.output_attn_res_scratch,
            &runtime.hidden,
        );
        runtime.hidden.copy_from_slice(mixed);
    }
    rmsnorm_with_gamma_eps(
        &mut runtime.hidden,
        &weights.final_norm_weight,
        config.rms_eps as f64,
    );
    katgpt_core::simd::simd_matmul_rows(
        &mut runtime.logits,
        &weights.lm_head_weight,
        &runtime.hidden,
        config.vocab_size,
        d,
    );
    runtime.logits.clone()
}

// ─── Simulator ─────────────────────────────────────────────────────────────

/// One (delay-layer, position) speculative execution measurement.
#[derive(Debug, Clone)]
pub struct ReplayOutcome {
    /// Underlying katgpt-core outcome record (ratio, top1, KLs, margin).
    pub core: SpecOutcome,
    /// The delay layer ℓ (speculation ran layers ℓ+1.. on `x_in^ℓ`).
    pub delay_layer: usize,
    /// The speculative run's logits (kept for corrected-arm + trajectory
    /// comparisons in the bench).
    pub logits: Vec<f32>,
}

/// The K3 stale-residual simulator.
///
/// Wraps a runtime + weights; owns the replay mechanics.
pub struct StaleResidualSim<'a> {
    pub config: &'a KimiK3ModelConfig,
    pub weights: &'a KimiK3ModelWeights,
    pub runtime: &'a mut KimiK3Runtime,
}

impl<'a> StaleResidualSim<'a> {
    pub fn new(
        config: &'a KimiK3ModelConfig,
        weights: &'a KimiK3ModelWeights,
        runtime: &'a mut KimiK3Runtime,
    ) -> Self {
        Self {
            config,
            weights,
            runtime,
        }
    }

    /// True forward with capture (see [`capture_forward_token`]).
    pub fn capture_token(&mut self, token_id: u32) -> TokenCapture {
        capture_forward_token(self.config, self.weights, self.runtime, token_id)
    }

    /// Replay the speculative tail for delay layer ℓ on the given start
    /// hidden (stale `x_in^ℓ` or a corrected variant), measuring against the
    /// captured true logits.
    ///
    /// Restores the pre-token state before the replay and the post-true state
    /// after — measurement is side-effect-free on the trajectory (the
    /// persistent-hazard arm is driven manually via
    /// [`StaleResidualSim::replay_raw`]).
    pub fn replay_stale(
        &mut self,
        cap: &TokenCapture,
        delay_layer: usize,
        start_hidden: &[f32],
    ) -> ReplayOutcome {
        let true_logits = cap.logits.clone();
        cap.pre_state.restore_into(self.runtime);
        let spec_logits = replay_from_layer(
            self.config,
            self.weights,
            self.runtime,
            delay_layer + 1,
            start_hidden,
            &cap.block_at_entry[delay_layer + 1],
        );
        cap.post_state.restore_into(self.runtime);
        ReplayOutcome {
            core: outcome_from_logits(
                &cap.x_in[delay_layer],
                &cap.x_in[delay_layer + 1],
                &true_logits,
                &spec_logits,
            ),
            delay_layer,
            logits: spec_logits,
        }
    }

    /// Raw replay WITHOUT the post-restore — the caller controls snapshots.
    /// Used by the persistent-hazard arm (accepted stale-written KV/KDA
    /// states persist into the next token). Semantics: layers `> delay_layer`
    /// are rewound to their PRE-TOKEN state first (the pipelined engine's
    /// speculative layer ℓ+1 writes position p from the stale input — it
    /// OVERWRITES, never appends at p+1); layers `≤ delay_layer` keep their
    /// post-capture state (they completed the true position-p work).
    pub fn replay_raw(
        &mut self,
        cap: &TokenCapture,
        delay_layer: usize,
        start_hidden: &[f32],
    ) -> Vec<f32> {
        rewind_tail_to_pre(&cap.pre_state, delay_layer, self.runtime);
        replay_from_layer(
            self.config,
            self.weights,
            self.runtime,
            delay_layer + 1,
            start_hidden,
            &cap.block_at_entry[delay_layer + 1],
        )
    }
}

/// Build a [`SpecOutcome`] from the captured stream states + logits.
pub fn outcome_from_logits(
    x_in: &[f32],
    x_out: &[f32],
    true_logits: &[f32],
    spec_logits: &[f32],
) -> SpecOutcome {
    use katgpt_core::stale_residual::kl_logits;
    let ratio = {
        let mut s = 0.0f32;
        for i in 0..x_in.len() {
            let e = x_out[i] - x_in[i];
            s += e * e;
        }
        let mut q = 0.0f32;
        for &v in x_in {
            q += v * v;
        }
        if q > 0.0 {
            (s / q).sqrt()
        } else {
            f32::INFINITY
        }
    };
    // Argmax + margin on the true logits.
    let mut t_top = 0usize;
    let mut t_second = 0usize;
    for (i, &l) in true_logits.iter().enumerate() {
        if l > true_logits[t_top] {
            t_second = t_top;
            t_top = i;
        } else if i != t_top && l > true_logits[t_second] {
            t_second = i;
        }
    }
    let mut s_top = 0usize;
    for (i, &l) in spec_logits.iter().enumerate() {
        if l > spec_logits[s_top] {
            s_top = i;
        }
    }
    SpecOutcome {
        ratio,
        top1_match: t_top == s_top,
        kl_true_given_spec: kl_logits(true_logits, spec_logits),
        kl_spec_given_true: kl_logits(spec_logits, true_logits),
        true_top1_margin: true_logits[t_top] - true_logits[t_second],
    }
}

// ─── T3 predictors (Approach B) ────────────────────────────────────────────

/// Router-logit features for layer ℓ evaluated on the stale input
/// `x_in^ℓ` (available at speculation time): `router_weight ·
/// RMSNorm_γ(x_in^ℓ)` + the noaux_tc bias — 9 features for K3-0.40B
/// (8 routed experts + bias-augmented constant handled by the fitter's
/// intercept via a constant feature).
///
/// Layer 0 (dense FFN, no router) yields an empty feature vec — the
/// router-logit predictor is not defined there and the bench skips it.
pub fn router_logit_features(
    config: &KimiK3ModelConfig,
    weights: &KimiK3ModelWeights,
    layer_idx: usize,
    x_in: &[f32],
    feature_out: &mut Vec<f32>,
) -> bool {
    let d = config.hidden_size;
    let layer_w = &weights.layers[layer_idx];
    let KimiFfnWeights::Moe(moe_w) = &layer_w.ffn else {
        feature_out.clear();
        return false;
    };
    let n_r = config.moe_config.num_experts;
    // RMSNorm with the layer's post-attention gamma (the router's norm in
    // the true forward; here applied to the stale input — the pre-dense
    // routing anchor).
    let mut normed = x_in.to_vec();
    rmsnorm_with_gamma_eps(
        &mut normed,
        &layer_w.post_attention_layernorm_weight,
        config.rms_eps as f64,
    );
    feature_out.clear();
    feature_out.reserve(n_r + 1);
    for e in 0..n_r {
        let row = &moe_w.router_weight[e * d..(e + 1) * d];
        let mut dot = katgpt_core::simd::simd_dot_f32(row, &normed, d);
        dot += moe_w.e_score_correction_bias[e];
        feature_out.push(dot);
    }
    feature_out.push(1.0); // intercept feature
    true
}

/// Fit + apply a linear δ-predictor over a collected sample set.
///
/// X = feature rows `[n][t]` (row-major), Y = δ targets `[n][d]`. Fits
/// per-output-dim OLS via the katgpt-attn-match substrate (ridge on the
/// x_in-linear variant where t is large + collinear). Returns the fitted
/// matrix `W [t][d]` (row-major: δ̂ = φ·W) + the in-sample R².
#[derive(Clone)]
pub struct DeltaPredictor {
    /// `W [t][d]` row-major — `δ̂ = φ · W`.
    pub w: Vec<f32>,
    pub t: usize,
    pub d: usize,
    /// In-sample coefficient of determination (1.0 = exact).
    pub r_squared: f32,
}

impl DeltaPredictor {
    /// Fit `δ ≈ φ·W` (closed form; no gradient descent — modelless mandate).
    pub fn fit(
        features: &[f32],
        targets: &[f32],
        n: usize,
        t: usize,
        d: usize,
        ridge: f32,
    ) -> Self {
        let fit = fit_cv_least_squares(
            features,
            targets,
            n,
            t,
            d,
            &ValueFitConfig {
                ridge_lambda: ridge,
                cholesky_jitter: 1e-4,
            },
        );
        let w = fit.compact_values;
        // In-sample R²: 1 − SS_res/SS_tot over ALL output dims.
        let mut ss_res = 0.0f64;
        let mut ss_tot = 0.0f64;
        let mut mean = vec![0.0f64; d];
        for i in 0..n {
            for j in 0..d {
                mean[j] += targets[i * d + j] as f64 / n as f64;
            }
        }
        for i in 0..n {
            let phi = &features[i * t..(i + 1) * t];
            for j in 0..d {
                let mut pred = 0.0f32;
                for (k, &p) in phi.iter().enumerate() {
                    pred += p * w[k * d + j];
                }
                let res = targets[i * d + j] - pred;
                ss_res += (res * res) as f64;
                let dev = targets[i * d + j] as f64 - mean[j];
                ss_tot += dev * dev;
            }
        }
        let r_squared = if ss_tot > 0.0 {
            (1.0 - ss_res / ss_tot) as f32
        } else {
            f32::NAN
        };
        Self { w, t, d, r_squared }
    }

    /// Apply: `δ̂ = φ · W` into `out` (cleared + resized).
    pub fn predict_into(&self, phi: &[f32], out: &mut Vec<f32>) {
        debug_assert_eq!(phi.len(), self.t);
        out.clear();
        out.resize(self.d, 0.0);
        for (k, &p) in phi.iter().enumerate() {
            let row = &self.w[k * self.d..(k + 1) * self.d];
            for (o, &r) in row.iter().enumerate() {
                out[o] += p * r;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kimi_k3::loader::KimiK3ModelWeights;
    use crate::kimi_k3::model::{KimiK3ModelConfig, KimiK3Runtime};

    #[test]
    fn capture_forward_is_bit_identical_to_reference() {
        let config = KimiK3ModelConfig::kimi_k3_0_40b();
        let weights = KimiK3ModelWeights::random(&config, 7);
        let mut rt_a = KimiK3Runtime::new(&config, 32);
        let mut rt_b = KimiK3Runtime::new(&config, 32);

        // Warm both with one token (KDA state evolution + MLA cache).
        let _ = crate::kimi_k3::model::kimi_k3_forward_token(&config, &weights, &mut rt_a, 1);
        let _ = crate::kimi_k3::model::kimi_k3_forward_token(&config, &weights, &mut rt_b, 1);

        let ref_logits =
            crate::kimi_k3::model::kimi_k3_forward_token(&config, &weights, &mut rt_a, 2).to_vec();
        let cap = capture_forward_token(&config, &weights, &mut rt_b, 2);

        assert_eq!(cap.logits.len(), ref_logits.len());
        for (a, b) in cap.logits.iter().zip(ref_logits.iter()) {
            assert_eq!(a, b, "capture forward must be bit-identical");
        }
        assert_eq!(cap.x_in.len(), config.num_layers);
        assert_eq!(cap.block_at_entry.len(), config.num_layers);
    }

    #[test]
    fn true_input_replay_matches_true_logits() {
        // Replaying from the TRUE start hidden (x_in^{ℓ+1}) with pre-token
        // caches restored must reproduce the true logits bit-identically —
        // the simulator's zero-divergence anchor (G1 for the replay path).
        let config = KimiK3ModelConfig::kimi_k3_0_40b();
        let weights = KimiK3ModelWeights::random(&config, 11);
        let mut rt = KimiK3Runtime::new(&config, 32);
        let _ = crate::kimi_k3::model::kimi_k3_forward_token(&config, &weights, &mut rt, 1);
        let cap = capture_forward_token(&config, &weights, &mut rt, 2);

        let mut sim = StaleResidualSim::new(&config, &weights, &mut rt);
        for delay in 0..config.num_layers - 1 {
            let true_start = cap.x_in[delay + 1].clone();
            let out = sim.replay_stale(&cap, delay, &true_start);
            for (a, b) in out.logits.iter().zip(cap.logits.iter()) {
                assert_eq!(a, b, "true-input replay (delay {delay}) diverged");
            }
        }
    }

    #[test]
    fn snapshot_restore_roundtrip_mid_sequence() {
        let config = KimiK3ModelConfig::kimi_k3_0_40b();
        let weights = KimiK3ModelWeights::random(&config, 13);
        let mut rt = KimiK3Runtime::new(&config, 32);
        let _ = crate::kimi_k3::model::kimi_k3_forward_token(&config, &weights, &mut rt, 1);
        let snap = RuntimeSnapshot::capture(&config, &rt);
        // Run a token (mutates state), then restore + verify the next-token
        // logits equal the no-interference run.
        let _ = crate::kimi_k3::model::kimi_k3_forward_token(&config, &weights, &mut rt, 2);
        snap.restore_into(&mut rt);
        let logits_after =
            crate::kimi_k3::model::kimi_k3_forward_token(&config, &weights, &mut rt, 2).to_vec();

        let mut rt2 = KimiK3Runtime::new(&config, 32);
        let _ = crate::kimi_k3::model::kimi_k3_forward_token(&config, &weights, &mut rt2, 1);
        let logits_ref =
            crate::kimi_k3::model::kimi_k3_forward_token(&config, &weights, &mut rt2, 2).to_vec();
        assert_eq!(logits_after, logits_ref);
    }

    #[test]
    fn stale_replay_diverges_measurably() {
        // Sanity that the measurement path produces finite, in-range values
        // and that SOME divergence exists on stale input (else the POC
        // instrument is dead).
        let config = KimiK3ModelConfig::kimi_k3_0_40b();
        let weights = KimiK3ModelWeights::random(&config, 17);
        let mut rt = KimiK3Runtime::new(&config, 32);
        let _ = crate::kimi_k3::model::kimi_k3_forward_token(&config, &weights, &mut rt, 1);
        let cap = capture_forward_token(&config, &weights, &mut rt, 2);

        let mut sim = StaleResidualSim::new(&config, &weights, &mut rt);
        let mut any_divergence = false;
        for delay in 0..config.num_layers - 1 {
            let stale = cap.x_in[delay].clone();
            let out = sim.replay_stale(&cap, delay, &stale);
            assert!(out.logits.iter().all(|l| l.is_finite()));
            assert!(out.core.ratio.is_finite() && out.core.ratio >= 0.0);
            if out.core.kl_true_given_spec > 0.0 {
                any_divergence = true;
            }
        }
        assert!(any_divergence, "random weights must produce measurable KL");
    }

    #[test]
    fn predictor_exact_fit_on_linear_data() {
        // δ = φ·W exactly → R² = 1, prediction matches.
        let n = 64;
        let t = 4;
        let d = 3;
        let mut rng = katgpt_core::Rng::new(42);
        let mut features = vec![0.0f32; n * t];
        for f in features.iter_mut() {
            *f = rng.uniform();
        }
        // True W (t×d), row-major [t][d].
        let true_w = vec![
            0.5, -0.25, 0.75, //
            1.0, 0.0, -0.5, //
            0.25, 0.9, 0.1, //
            -0.3, 0.2, 0.6,
        ];
        let mut targets = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                let mut acc = 0.0f32;
                for k in 0..t {
                    acc += features[i * t + k] * true_w[k * d + j];
                }
                targets[i * d + j] = acc;
            }
        }
        let pred = DeltaPredictor::fit(&features, &targets, n, t, d, 0.0);
        assert!(pred.r_squared > 0.999, "R² {}", pred.r_squared);
        let mut out = Vec::new();
        pred.predict_into(&features[..t], &mut out);
        for (a, b) in out.iter().zip(targets[..d].iter()) {
            assert!((a - b).abs() < 1e-2, "{a} vs {b}");
        }
    }
}
