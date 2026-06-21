//! FlashAR Consensus Tri-Mode with Ternary Thermal Paths
//!
//! Plan 166 (Research 149): Replaces tri_mode's prefix-match acceptance with
//! dual-path consensus draft + ternary thermal path routing.
//!
//! Architecture:
//!   Path H: AR/MTP draft     → per-position tokens + confidence
//!   Path V: D2F block draft  → per-position tokens + confidence
//!
//!   Ternary consensus per position:
//!     +1 → H wins (conf_H > conf_V)
//!      0 → AGREE (both same token) → PLASMA PATH (skip verify)
//!     -1 → V wins (conf_V >= conf_H)
//!
//!   Thermal routing:
//!     PLASMA  (ternary=0, high conf)   → accept immediately
//!     HOT     (ternary=±1, high conf)  → accept winner
//!     WARM    (ternary=±1, mid conf)   → AR spot-check
//!     COLD    (both low conf)          → fallback prefix-match

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use crate::dllm::D2fContext;
use crate::speculative::d2f::{D2fDecodeConfig, d2f_decode_block_with_prompt_with};
use crate::speculative::sampling::sample_from_distribution;
use crate::speculative::types::{NoPruner, NoScreeningPruner};
use crate::speculative::verifier::SpeculativeVerifier;
use crate::transformer::TransformerWeights;
use crate::transformer::{ForwardContext, MultiLayerKVCache, forward};
use crate::types::{Config, Rng, softmax_scaled};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum draft width supported (bounded by typical block sizes).
pub const MAX_DRAFT_WIDTH: usize = 64;

// ---------------------------------------------------------------------------
// Types (T1)
// ---------------------------------------------------------------------------

/// Thermal path assigned per position based on ternary consensus + confidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThermalPath {
    /// Both paths agree, high confidence — accept immediately, zero verification.
    Plasma,
    /// One path wins, high confidence — accept winner without verification.
    Hot,
    /// One path wins, moderate confidence — AR spot-check this position only.
    Warm,
    /// Both paths low confidence — fallback to prefix-match verification.
    Cold,
}

/// Configuration for the FlashAR consensus thermal path router.
#[derive(Clone, Debug)]
pub struct ConsensusConfig {
    /// Confidence threshold for PLASMA path (both agree AND conf > τ_p).
    /// Default: 0.7
    pub plasma_threshold: f32,
    /// Confidence threshold for HOT path (winner conf > τ_h).
    /// Default: 0.5
    pub hot_threshold: f32,
    /// Confidence threshold for WARM path (winner conf > τ_w).
    /// Default: 0.3
    pub warm_threshold: f32,
    /// If true, use `simd_ternary_matvec` fusion gate instead of heuristic.
    /// Requires `plasma_path` feature.
    pub use_ternary_gate: bool,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            plasma_threshold: 0.7,
            hot_threshold: 0.5,
            warm_threshold: 0.3,
            use_ternary_gate: false,
        }
    }
}

/// Per-position result of the thermal routing pass.
pub struct ConsensusResult {
    /// Thermal path assigned to each position [0..len].
    pub thermal_paths: [ThermalPath; MAX_DRAFT_WIDTH],
    /// Ternary consensus code per position: +1, 0, -1.
    pub ternary_codes: [i8; MAX_DRAFT_WIDTH],
    /// Accepted token per position (winner of H vs V, or consensus token).
    pub accepted_tokens: [usize; MAX_DRAFT_WIDTH],
    /// Actual number of positions in the draft.
    pub len: usize,
}

impl Default for ConsensusResult {
    fn default() -> Self {
        Self {
            thermal_paths: [ThermalPath::Cold; MAX_DRAFT_WIDTH],
            ternary_codes: [0; MAX_DRAFT_WIDTH],
            accepted_tokens: [0; MAX_DRAFT_WIDTH],
            len: 0,
        }
    }
}

/// Result of running both draft paths.
pub struct DualPathResult {
    /// AR/MTP draft tokens (Path H).
    pub h_tokens: [usize; MAX_DRAFT_WIDTH],
    /// AR/MTP confidence per position (top1_prob from softmax).
    pub h_confidences: [f32; MAX_DRAFT_WIDTH],
    /// D2F block draft tokens (Path V).
    pub v_tokens: [usize; MAX_DRAFT_WIDTH],
    /// D2F confidence per position (top1_prob from softmax).
    pub v_confidences: [f32; MAX_DRAFT_WIDTH],
    /// Number of positions drafted.
    pub len: usize,
}

impl Default for DualPathResult {
    fn default() -> Self {
        Self {
            h_tokens: [0; MAX_DRAFT_WIDTH],
            h_confidences: [0.0; MAX_DRAFT_WIDTH],
            v_tokens: [0; MAX_DRAFT_WIDTH],
            v_confidences: [0.0; MAX_DRAFT_WIDTH],
            len: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// T2: dual_path_draft
// ---------------------------------------------------------------------------

/// Package pre-computed dual-path draft results into a `DualPathResult`.
///
/// This is a pure data-assembly function — the actual dual-path execution
/// is performed by `FlashARConsensusVerifier::speculate()` using the AR and
/// D2F forward passes. This function just packages the results into the
/// fixed-size stack arrays used by downstream functions.
pub fn dual_path_draft(
    draft_width: usize,
    h_tokens: &[usize],
    h_confidences: &[f32],
    v_tokens: &[usize],
    v_confidences: &[f32],
) -> DualPathResult {
    let k = draft_width.min(MAX_DRAFT_WIDTH);
    let mut result = DualPathResult {
        len: k,
        ..Default::default()
    };

    for i in 0..k {
        result.h_tokens[i] = *h_tokens.get(i).unwrap_or(&0);
        result.h_confidences[i] = *h_confidences.get(i).unwrap_or(&0.0);
        result.v_tokens[i] = *v_tokens.get(i).unwrap_or(&0);
        result.v_confidences[i] = *v_confidences.get(i).unwrap_or(&0.0);
    }

    result
}

// ---------------------------------------------------------------------------
// T3: compute_ternary_consensus
// ---------------------------------------------------------------------------

/// Compute per-position ternary consensus code and accepted token.
///
/// For each position:
///   - ternary = 0  if h_tokens[i] == v_tokens[i] (AGREE)
///   - ternary = +1 if h_tokens[i] != v_tokens[i] AND h_conf[i] > v_conf[i] (H wins)
///   - ternary = -1 if h_tokens[i] != v_tokens[i] AND v_conf[i] >= h_conf[i] (V wins)
///
/// Accepted token is h_tokens[i] if ternary >= 0, else v_tokens[i].
pub fn compute_ternary_consensus(
    h_tokens: &[usize],
    v_tokens: &[usize],
    h_conf: &[f32],
    v_conf: &[f32],
    len: usize,
) -> ([i8; MAX_DRAFT_WIDTH], [usize; MAX_DRAFT_WIDTH]) {
    let mut ternary = [0i8; MAX_DRAFT_WIDTH];
    let mut accepted = [0usize; MAX_DRAFT_WIDTH];

    for i in 0..len.min(MAX_DRAFT_WIDTH) {
        let h_tok = h_tokens[i];
        let v_tok = v_tokens[i];

        if h_tok == v_tok {
            // Consensus: both paths agree
            ternary[i] = 0;
            accepted[i] = h_tok;
        } else {
            // Dispute: higher confidence wins
            let h_c = h_conf[i];
            let v_c = v_conf[i];
            if h_c > v_c {
                ternary[i] = 1; // H wins
                accepted[i] = h_tok;
            } else {
                ternary[i] = -1; // V wins
                accepted[i] = v_tok;
            }
        }
    }

    (ternary, accepted)
}

// ---------------------------------------------------------------------------
// T4: route_thermal_paths
// ---------------------------------------------------------------------------

/// Route each position to a thermal path based on ternary code + confidence.
///
/// Thermal routing table:
///   PLASMA (ternary=0, min(h_conf, v_conf) >= plasma_threshold)
///   HOT    (ternary=±1, winner_conf >= hot_threshold)
///   WARM   (ternary=±1, winner_conf >= warm_threshold)
///   COLD   (everything else)
pub fn route_thermal_paths(
    ternary: &[i8; MAX_DRAFT_WIDTH],
    h_conf: &[f32],
    v_conf: &[f32],
    h_tokens: &[usize],
    v_tokens: &[usize],
    config: &ConsensusConfig,
    len: usize,
) -> ConsensusResult {
    let mut result = ConsensusResult {
        len,
        ..Default::default()
    };

    for i in 0..len.min(MAX_DRAFT_WIDTH) {
        let code = ternary[i];
        result.ternary_codes[i] = code;

        if code == 0 {
            // Consensus — both agree
            let min_conf = h_conf[i].min(v_conf[i]);
            if min_conf >= config.plasma_threshold {
                result.thermal_paths[i] = ThermalPath::Plasma;
            } else if min_conf >= config.hot_threshold {
                result.thermal_paths[i] = ThermalPath::Hot;
            } else if min_conf >= config.warm_threshold {
                result.thermal_paths[i] = ThermalPath::Warm;
            } else {
                result.thermal_paths[i] = ThermalPath::Cold;
            }
            result.accepted_tokens[i] = h_tokens[i]; // same as v_tokens[i]
        } else {
            // Disputed — pick winner
            let winner_conf = if code > 0 { h_conf[i] } else { v_conf[i] };
            let winner_tok = if code > 0 { h_tokens[i] } else { v_tokens[i] };

            if winner_conf >= config.hot_threshold {
                result.thermal_paths[i] = ThermalPath::Hot;
            } else if winner_conf >= config.warm_threshold {
                result.thermal_paths[i] = ThermalPath::Warm;
            } else {
                result.thermal_paths[i] = ThermalPath::Cold;
            }
            result.accepted_tokens[i] = winner_tok;
        }
    }

    result
}

// ---------------------------------------------------------------------------
// T5: Ternary SIMD fusion gate (optional, requires plasma_path)
// ---------------------------------------------------------------------------

#[cfg(feature = "plasma_path")]
use katgpt_core::TernaryWeights;

/// Compute fusion gate scores using ternary SIMD matvec.
///
/// Uses `simd_ternary_matvec` from the `plasma_path` feature — zero multiplication.
/// The gate_weights have rows=1, cols=6 (one row per position, 6 SamplerFeatures).
/// Output is a per-position score: higher → more confident routing.
#[cfg(feature = "plasma_path")]
pub fn ternary_fusion_gate(
    gate_weights: &TernaryWeights,
    features: &[f32], // flat [positions * 6]
) -> Vec<f32> {
    let n_positions = features.len() / 6;
    let mut scores = vec![0.0f32; n_positions];
    let feature_dim = 6;

    for pos in 0..n_positions {
        let x = &features[pos * feature_dim..(pos + 1) * feature_dim];
        let mut score = [0.0f32; 1];
        // gate_weights has rows=1, cols=6
        katgpt_core::simd_ternary_matvec(gate_weights, x, &mut score);
        scores[pos] = score[0];
    }

    scores
}

// top1_prob helper removed — not used in current implementation

// ---------------------------------------------------------------------------
// T6: FlashARConsensusVerifier
// ---------------------------------------------------------------------------

/// FlashAR Consensus Verifier — dual-path speculative decoding with ternary
/// thermal path routing.
///
/// Replaces the prefix-match acceptance of `D2fDrafterVerifier` with:
/// 1. Dual-path drafting (AR + D2F in parallel)
/// 2. Ternary consensus encoding per position
/// 3. Thermal path routing (Plasma/Hot/Warm/Cold)
/// 4. Selective verification based on thermal path
pub struct FlashARConsensusVerifier<'a> {
    pub target_weights: &'a TransformerWeights,
    pub target_config: &'a Config,
    pub d2f_config: D2fDecodeConfig,
    pub consensus_config: ConsensusConfig,
    pub draft_width: usize,

    // Internal buffers
    target_ctx: ForwardContext,
    target_cache: MultiLayerKVCache,
    d2f_ctx: D2fContext,
    probs_buf: Vec<f32>,
    /// Pre-allocated flat storage for per-position target distributions.
    /// Layout: `p_flat[(i+1) * vocab_size .. (i+2) * vocab_size]` holds the
    /// softmaxed target distribution used to score position `i`.
    /// Length = `(MAX_DRAFT_WIDTH + 1) * vocab_size`, allocated once.
    p_flat: Vec<f32>,
    /// Reusable scratch for a single forward's softmax output (alias-safe).
    forward_scratch: Vec<f32>,
}

impl<'a> FlashARConsensusVerifier<'a> {
    /// Create a new FlashAR consensus verifier.
    ///
    /// `draft_width` must match `d2f_config.block_size`.
    pub fn new(
        target_weights: &'a TransformerWeights,
        target_config: &'a Config,
        d2f_config: D2fDecodeConfig,
        consensus_config: ConsensusConfig,
        draft_width: usize,
    ) -> Self {
        let block_size = d2f_config.block_size.max(draft_width);
        let config = D2fDecodeConfig {
            block_size,
            ..d2f_config
        };
        Self {
            target_weights,
            target_config,
            d2f_config: config,
            consensus_config,
            draft_width,
            target_ctx: ForwardContext::new(target_config),
            target_cache: MultiLayerKVCache::new(target_config),
            d2f_ctx: D2fContext::new(target_config),
            probs_buf: vec![0.0f32; target_config.vocab_size],
            p_flat: vec![
                0.0f32;
                (MAX_DRAFT_WIDTH + 1) * target_config.vocab_size
            ],
            forward_scratch: vec![0.0f32; target_config.vocab_size],
        }
    }
}

impl SpeculativeVerifier for FlashARConsensusVerifier<'_> {
    #[allow(clippy::needless_range_loop)]
    fn speculate(
        &mut self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        token: usize,
        pos: usize,
        rng: &mut Rng,
    ) -> Vec<usize> {
        let target_temp = self.target_config.temperature;
        let draft_width = self.draft_width;
        let vocab_size = self.target_config.vocab_size;

        // ── Phase 0: Score initial token with target model ──────────
        self.target_cache.reset();
        {
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                token,
                pos,
                self.target_config,
            );
            self.probs_buf.copy_from_slice(logits);
            softmax_scaled(&mut self.probs_buf, 1.0 / target_temp);
            // Store p_dist[0] into the flat buffer (slot 0).
            self.p_flat[..vocab_size].copy_from_slice(&self.probs_buf);
        }

        // ── Phase 1: Path V — D2F block decode ──────────────────────
        let prompt = &[token];
        let d2f_result = d2f_decode_block_with_prompt_with(
            &mut self.d2f_ctx,
            draft_weights,
            draft_config,
            &self.d2f_config,
            prompt,
            &NoPruner,
            &NoScreeningPruner,
            rng,
        );

        let v_tokens_raw = &d2f_result.tokens;
        let k = v_tokens_raw.len().min(draft_width);

        if k == 0 {
            return vec![sample_from_distribution(&self.probs_buf, rng)];
        }

        // Copy D2F tokens to stack
        let mut v_tokens = [0usize; MAX_DRAFT_WIDTH];
        let mut v_conf = [0.0f32; MAX_DRAFT_WIDTH];
        let k_bounded = k.min(MAX_DRAFT_WIDTH);
        v_tokens[..k_bounded].copy_from_slice(&v_tokens_raw[..k_bounded]);

        // Extract D2F confidence per position from the logits.
        // Single-pass softmax+argmax: compute max_logit, then sum_exp + top1 in
        // one fold to halve iterations over the vocab.
        for i in 0..k_bounded {
            let logits_offset = i * draft_config.vocab_size;
            if logits_offset + draft_config.vocab_size <= self.d2f_ctx.logits_flat.len() {
                let logits_p = &self.d2f_ctx.logits_flat
                    [logits_offset..logits_offset + draft_config.vocab_size];
                let max_logit = logits_p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                // Fused pass: accumulate sum_exp and track top1 prob.
                let mut sum_exp = 0.0f32;
                let mut top1 = 0.0f32;
                for &l in logits_p {
                    let p = (l - max_logit).exp();
                    sum_exp += p;
                    if p > top1 {
                        top1 = p;
                    }
                }
                v_conf[i] = top1 / sum_exp.max(1e-10);
            } else {
                v_conf[i] = 0.5; // fallback confidence
            }
        }

        // ── Phase 2: Path H — AR sequential draft ───────────────────
        // We use the target model's KV cache which already has the initial token.
        // Run AR draft using the same draft_weights.
        let mut h_tokens = [0usize; MAX_DRAFT_WIDTH];
        let mut h_conf = [0.0f32; MAX_DRAFT_WIDTH];
        // Cache the target argmax per position (avoids recomputing it in
        // Phase 5 for Warm/Cold segments — each p_dist is scanned once here).
        let mut target_argmax = [0usize; MAX_DRAFT_WIDTH];

        // Sequential AR scoring: for each position, get target distribution,
        // extract argmax as H's prediction, and top1 as H's confidence.
        // Write softmaxed distribution into p_flat slot (i+1).
        for i in 0..k_bounded {
            // Use D2F token as input for sequential scoring (like D2fDrafterVerifier)
            let input_tok = if i == 0 { v_tokens[0] } else { h_tokens[i - 1] };
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                input_tok,
                pos + 1 + i,
                self.target_config,
            );
            self.forward_scratch.copy_from_slice(logits);
            softmax_scaled(&mut self.forward_scratch, 1.0 / target_temp);

            // H path: argmax of target distribution (this is what the target model prefers).
            // Single pass — extract both best_idx and best_prob.
            let mut best_idx = 0usize;
            let mut best_prob = f32::NEG_INFINITY;
            for (idx, &p) in self.forward_scratch.iter().enumerate() {
                if p > best_prob {
                    best_prob = p;
                    best_idx = idx;
                }
            }

            h_tokens[i] = best_idx;
            h_conf[i] = if best_prob == f32::NEG_INFINITY { 0.0 } else { best_prob };
            target_argmax[i] = best_idx;

            // Persist this distribution into p_flat slot (i+1).
            let start = (i + 1) * vocab_size;
            self.p_flat[start..start + vocab_size].copy_from_slice(&self.forward_scratch);
        }

        // ── Phase 3: Ternary consensus ──────────────────────────────
        let (ternary, consensus_tokens) =
            compute_ternary_consensus(&h_tokens, &v_tokens, &h_conf, &v_conf, k_bounded);

        // ── Phase 4: Thermal routing ────────────────────────────────
        let result = route_thermal_paths(
            &ternary,
            &h_conf,
            &v_conf,
            &h_tokens,
            &v_tokens,
            &self.consensus_config,
            k_bounded,
        );

        // ── Phase 5: Selective verification ─────────────────────────
        // Plasma/Hot: accept immediately
        // Warm: verify single position (we already have target_argmax from Phase 2)
        // Cold: fall back to prefix-match for that segment
        let mut accepted = Vec::with_capacity(k_bounded + 1);
        let mut all_accepted = true;

        // Track contiguous cold segments for prefix-match fallback
        let mut cold_start: Option<usize> = None;

        // Helper: flush a cold segment [start..end) using cached target_argmax.
        // Returns false on first mismatch (caller breaks out).
        let flush_cold = |start: usize, end: usize,
                          consensus_tokens: &[usize; MAX_DRAFT_WIDTH],
                          target_argmax: &[usize; MAX_DRAFT_WIDTH],
                          accepted: &mut Vec<usize>| -> bool {
            for j in start..end {
                let draft_tok = consensus_tokens[j];
                let target_tok = target_argmax[j];
                if draft_tok == target_tok {
                    accepted.push(draft_tok);
                } else {
                    accepted.push(target_tok);
                    return false;
                }
            }
            true
        };

        for i in 0..k_bounded {
            match result.thermal_paths[i] {
                ThermalPath::Plasma | ThermalPath::Hot => {
                    // Flush any pending cold segment
                    if let Some(start) = cold_start.take()
                        && !flush_cold(start, i, &consensus_tokens, &target_argmax, &mut accepted) {
                        all_accepted = false;
                        break;
                    }
                    // Accept Plasma/Hot position
                    accepted.push(result.accepted_tokens[i]);
                }
                ThermalPath::Warm => {
                    // Flush any pending cold segment
                    if let Some(start) = cold_start.take()
                        && !flush_cold(start, i, &consensus_tokens, &target_argmax, &mut accepted) {
                        all_accepted = false;
                        break;
                    }
                    // Spot-check: verify this single position against cached target argmax
                    let draft_tok = result.accepted_tokens[i];
                    let target_tok = target_argmax[i];
                    if draft_tok == target_tok {
                        accepted.push(draft_tok);
                    } else {
                        accepted.push(target_tok);
                        all_accepted = false;
                        break;
                    }
                }
                ThermalPath::Cold => {
                    // Accumulate cold positions for prefix-match fallback
                    if cold_start.is_none() {
                        cold_start = Some(i);
                    }
                }
            }

            if !all_accepted {
                break;
            }
        }

        // Flush trailing cold segment
        if all_accepted {
            if let Some(start) = cold_start.take()
                && !flush_cold(start, k_bounded, &consensus_tokens, &target_argmax, &mut accepted) {
                all_accepted = false;
            }

            // Bonus token if all accepted
            if all_accepted {
                let bonus_start = k_bounded * vocab_size;
                let bonus_end = bonus_start + vocab_size;
                let bonus_dist = &self.p_flat[bonus_start..bonus_end];
                let bonus = sample_from_distribution(bonus_dist, rng);
                accepted.push(bonus);
            }
        }

        // Safety: always return at least one token
        if accepted.is_empty() {
            let p0 = &self.p_flat[..vocab_size];
            accepted.push(sample_from_distribution(p0, rng));
        }

        accepted
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_path_draft_basic() {
        let h_tokens = [10, 20, 30, 40];
        let h_conf = [0.9, 0.8, 0.7, 0.6];
        let v_tokens = [10, 25, 30, 45];
        let v_conf = [0.8, 0.85, 0.6, 0.5];

        let result = dual_path_draft(4, &h_tokens, &h_conf, &v_tokens, &v_conf);

        assert_eq!(result.len, 4);
        assert_eq!(result.h_tokens[0], 10);
        assert_eq!(result.h_tokens[1], 20);
        assert_eq!(result.v_tokens[1], 25);
        assert!((result.h_confidences[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_ternary_consensus_agree() {
        let h = [10, 20, 30];
        let v = [10, 20, 30];
        let hc = [0.9, 0.8, 0.7];
        let vc = [0.8, 0.9, 0.6];

        let (ternary, accepted) = compute_ternary_consensus(&h, &v, &hc, &vc, 3);

        assert_eq!(ternary[0], 0); // AGREE
        assert_eq!(ternary[1], 0);
        assert_eq!(ternary[2], 0);
        assert_eq!(accepted[0], 10);
        assert_eq!(accepted[1], 20);
        assert_eq!(accepted[2], 30);
    }

    #[test]
    fn test_ternary_consensus_dispute() {
        let h = [10, 20, 30];
        let v = [10, 25, 30];
        let hc = [0.9, 0.7, 0.7];
        let vc = [0.8, 0.85, 0.6];

        let (ternary, accepted) = compute_ternary_consensus(&h, &v, &hc, &vc, 3);

        assert_eq!(ternary[0], 0); // AGREE (both 10)
        assert_eq!(accepted[0], 10);
        assert_eq!(ternary[1], -1); // V wins (0.85 > 0.7)
        assert_eq!(accepted[1], 25);
        assert_eq!(ternary[2], 0); // AGREE (both 30)
        assert_eq!(accepted[2], 30);
    }

    #[test]
    fn test_thermal_routing_plasma() {
        let ternary = [0i8; MAX_DRAFT_WIDTH];
        let h_conf = [0.9f32; MAX_DRAFT_WIDTH];
        let v_conf = [0.8f32; MAX_DRAFT_WIDTH];
        let h_tokens = [42usize; MAX_DRAFT_WIDTH];
        let v_tokens = [42usize; MAX_DRAFT_WIDTH];

        let config = ConsensusConfig::default();
        let result =
            route_thermal_paths(&ternary, &h_conf, &v_conf, &h_tokens, &v_tokens, &config, 4);

        assert_eq!(result.thermal_paths[0], ThermalPath::Plasma);
        assert_eq!(result.accepted_tokens[0], 42);
    }

    #[test]
    fn test_thermal_routing_hot() {
        let mut ternary = [0i8; MAX_DRAFT_WIDTH];
        ternary[0] = 1; // H wins
        let h_conf = [0.8f32; MAX_DRAFT_WIDTH];
        let v_conf = [0.3f32; MAX_DRAFT_WIDTH];
        let h_tokens = [10usize; MAX_DRAFT_WIDTH];
        let v_tokens = [20usize; MAX_DRAFT_WIDTH];

        let config = ConsensusConfig::default();
        let result =
            route_thermal_paths(&ternary, &h_conf, &v_conf, &h_tokens, &v_tokens, &config, 4);

        assert_eq!(result.thermal_paths[0], ThermalPath::Hot);
        assert_eq!(result.accepted_tokens[0], 10); // H wins
    }

    #[test]
    fn test_thermal_routing_cold() {
        let mut ternary = [0i8; MAX_DRAFT_WIDTH];
        ternary[0] = 1; // H wins but low confidence
        let h_conf = [0.1f32; MAX_DRAFT_WIDTH];
        let v_conf = [0.05f32; MAX_DRAFT_WIDTH];
        let h_tokens = [10usize; MAX_DRAFT_WIDTH];
        let v_tokens = [20usize; MAX_DRAFT_WIDTH];

        let config = ConsensusConfig::default();
        let result =
            route_thermal_paths(&ternary, &h_conf, &v_conf, &h_tokens, &v_tokens, &config, 4);

        assert_eq!(result.thermal_paths[0], ThermalPath::Cold);
    }

    #[test]
    fn test_consensus_config_defaults() {
        let config = ConsensusConfig::default();
        assert!((config.plasma_threshold - 0.7).abs() < 1e-6);
        assert!((config.hot_threshold - 0.5).abs() < 1e-6);
        assert!((config.warm_threshold - 0.3).abs() < 1e-6);
        assert!(!config.use_ternary_gate);
    }

    #[test]
    fn test_verifier_returns_at_least_one() {
        let mut config = Config::micro();
        config.vocab_size = 64;
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

        let d2f_config = D2fDecodeConfig::with_block_size(4);
        let consensus_config = ConsensusConfig::default();
        let mut verifier = FlashARConsensusVerifier::new(
            &target_weights,
            &config,
            d2f_config,
            consensus_config,
            4,
        );

        let accepted = verifier.speculate(
            &draft_weights,
            &config,
            config.bos_token,
            0,
            &mut Rng::new(100),
        );
        assert!(
            !accepted.is_empty(),
            "speculate must always return at least one token"
        );
    }

    #[test]
    fn test_verifier_deterministic() {
        let mut config = Config::micro();
        config.vocab_size = 64;
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

        let d2f_config = D2fDecodeConfig::with_block_size(4);
        let consensus_config = ConsensusConfig::default();

        let r1 = {
            let mut verifier = FlashARConsensusVerifier::new(
                &target_weights,
                &config,
                d2f_config,
                consensus_config.clone(),
                4,
            );
            verifier.speculate(
                &draft_weights,
                &config,
                config.bos_token,
                0,
                &mut Rng::new(100),
            )
        };

        let r2 = {
            let mut verifier = FlashARConsensusVerifier::new(
                &target_weights,
                &config,
                d2f_config,
                consensus_config,
                4,
            );
            verifier.speculate(
                &draft_weights,
                &config,
                config.bos_token,
                0,
                &mut Rng::new(100),
            )
        };

        assert_eq!(r1, r2, "same seed must produce identical output");
    }

    #[test]
    fn test_verifier_bounded_output() {
        let mut config = Config::micro();
        config.vocab_size = 64;
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

        let draft_width = 4;
        let d2f_config = D2fDecodeConfig::with_block_size(draft_width);
        let consensus_config = ConsensusConfig::default();
        let mut verifier = FlashARConsensusVerifier::new(
            &target_weights,
            &config,
            d2f_config,
            consensus_config,
            draft_width,
        );

        for seed in 0..50u64 {
            let accepted = verifier.speculate(
                &draft_weights,
                &config,
                config.bos_token,
                0,
                &mut Rng::new(seed),
            );
            assert!(
                accepted.len() <= draft_width + 1,
                "accepted {} tokens but max is {}",
                accepted.len(),
                draft_width + 1,
            );
        }
    }
}
