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
//!     WARM    (ternary=±1, mid conf)   → policy-step verify
//!     COLD    (both low conf)          → policy-step verify
//!
//! # Issue 651 (2026-08-15): exact verified paths + slot alignment
//!
//! Warm/Cold verification now runs the FLARE acceptance policy
//! (default [`DraftAcceptPolicy::SoftmaxArgmax`], Eq 21) against the
//! **slot-aligned** target distribution, instead of the legacy argmax
//! prefix-match:
//!
//! - **Aligned conditioning** — draft token `i` is tested against
//!   `p_i = P(· | anchor, accepted prefix d_0..d_{i-1})`. The pre-651 code
//!   had the same off-by-one Issue 587 fixed in `D2fDrafterVerifier`: it
//!   tested `d_i` against the position-`i+1` distribution *conditioned on
//!   the draft token being tested* (Phase 2 fed `v_0` then the H tokens),
//!   making H-win positions auto-accept and V-win positions compare against
//!   a self-conditioned argmax. The interleave now feeds the ACCEPTED token
//!   after each acceptance (the Leviathan/D2f streaming pattern).
//! - **Exact verified paths** — the consensus winner is a deterministic
//!   (point-mass) proposal, for which Eq 21 `u ≤ p(d)` + correction
//!   `y ~ p∖{d}` is the exact rejection-sampling rule. `ExactQ` is honored
//!   as this same identity (Eq 8 with q = δ_d degenerates to Eq 21);
//!   `TruncatedArgmax` (Eq 22) is honored via the shared step helper.
//! - **PLASMA/HOT REMAIN DISTRIBUTION-BIASED BY DESIGN** — they skip
//!   verification entirely (that skipping IS the latency feature; making
//!   them exact would require the rejection sampling they exist to avoid).
//!   Consumers needing sampled (temperature-faithful) output should route
//!   everything to Warm/Cold (raise `plasma_threshold`/`hot_threshold` above
//!   1.0) — the Issue 651 exactness test does exactly this.
//! - **Streaming memory win** (mirrors 587 T5): the
//!   `(MAX_DRAFT_WIDTH+1) × vocab` `p_flat` buffer and the separate
//!   `forward_scratch` are deleted — one vocab-sized `probs_buf` streams
//!   position by position, and target forwards stop at the first rejection.
//!
//! Plan 400 (2026-07-05): moved from root `src/speculative/flashar_consensus.rs`.
//! All 10 tests moved with the file (no training dependencies). Root re-exports
//! via `pub use katgpt_forward::flashar_consensus::*` so all historical
//! `katgpt_rs::speculative::flashar_consensus::*` paths resolve.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use crate::d2f::{D2fDecodeConfig, d2f_decode_block_with_prompt_with};
use crate::d2f_context::D2fContext;
#[cfg(feature = "nucleus_accept")]
use crate::d2f_verifier::nucleus_accept_step;
use crate::d2f_verifier::{
    DraftAcceptPolicy, PolicyStep, prefix_match_step, softmax_argmax_step, truncated_argmax_step,
};
use crate::{ForwardContext, forward};
use katgpt_core::simd::simd_max_f32;
use katgpt_core::speculative::sampling::sample_from_distribution;
use katgpt_core::traits::{NoPruner, NoScreeningPruner};
use katgpt_speculative::SpeculativeVerifier;
use katgpt_transformer::{MultiLayerKVCache, TransformerWeights};
use katgpt_types::{Config, Rng, softmax_scaled};

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
#[repr(u8)]
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
#[derive(Clone, Copy, Debug)]
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
    /// Warm/Cold acceptance policy (Issue 651). Default `SoftmaxArgmax`
    /// (FLARE Eq 21 — the verified paths are distribution-preserving).
    /// `PrefixMatch` is the legacy Plan 166 mode-biasing control.
    /// `ExactQ` is honored as its point-mass identity (Eq 8 with q = δ_d IS
    /// Eq 21 — the consensus winner is a deterministic proposal).
    /// Plasma/Hot paths are unaffected (they skip verification by design).
    pub accept_policy: DraftAcceptPolicy,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            plasma_threshold: 0.7,
            hot_threshold: 0.5,
            warm_threshold: 0.3,
            accept_policy: DraftAcceptPolicy::SoftmaxArgmax,
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

/// Per-position ternary consensus kernel (Issue 651): the unit the aligned
/// `speculate` loop interleaves. Returns `(code, accepted_token)` where code
/// is +1 (H wins) / 0 (AGREE) / -1 (V wins).
#[inline]
pub fn ternary_consensus_one(h_tok: usize, v_tok: usize, h_c: f32, v_c: f32) -> (i8, usize) {
    if h_tok == v_tok {
        (0, h_tok)
    } else if h_c > v_c {
        (1, h_tok)
    } else {
        (-1, v_tok)
    }
}

/// Compute per-position ternary consensus code and accepted token.
///
/// For each position:
///   - ternary = 0  if `h_tokens[i] == v_tokens[i]` (AGREE)
///   - ternary = +1 if `h_tokens[i] != v_tokens[i]` AND `h_conf[i] > v_conf[i]` (H wins)
///   - ternary = -1 if `h_tokens[i] != v_tokens[i]` AND `v_conf[i] >= h_conf[i]` (V wins)
///
/// Accepted token is `h_tokens[i]` if ternary >= 0, else `v_tokens[i]`.
/// Delegates to [`ternary_consensus_one`] per position (Issue 651 DRY).
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
        let (code, tok) = ternary_consensus_one(h_tokens[i], v_tokens[i], h_conf[i], v_conf[i]);
        ternary[i] = code;
        accepted[i] = tok;
    }

    (ternary, accepted)
}

// ---------------------------------------------------------------------------
// T4: route_thermal_paths
// ---------------------------------------------------------------------------

/// Per-position thermal routing kernel (Issue 651): the unit the aligned
/// `speculate` loop interleaves.
#[inline]
pub fn route_one(code: i8, h_conf: f32, v_conf: f32, config: &ConsensusConfig) -> ThermalPath {
    if code == 0 {
        let min_conf = h_conf.min(v_conf);
        if min_conf >= config.plasma_threshold {
            ThermalPath::Plasma
        } else if min_conf >= config.hot_threshold {
            ThermalPath::Hot
        } else if min_conf >= config.warm_threshold {
            ThermalPath::Warm
        } else {
            ThermalPath::Cold
        }
    } else {
        let winner_conf = if code > 0 { h_conf } else { v_conf };
        if winner_conf >= config.hot_threshold {
            ThermalPath::Hot
        } else if winner_conf >= config.warm_threshold {
            ThermalPath::Warm
        } else {
            ThermalPath::Cold
        }
    }
}

/// Route each position to a thermal path based on ternary code + confidence.
///
/// Thermal routing table:
///   PLASMA (ternary=0, min(h_conf, v_conf) >= plasma_threshold)
///   HOT    (ternary=±1, winner_conf >= hot_threshold)
///   WARM   (ternary=±1, winner_conf >= warm_threshold)
///   COLD   (everything else)
///
/// Delegates to [`route_one`] per position (Issue 651 DRY).
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
        result.thermal_paths[i] = route_one(code, h_conf[i], v_conf[i], config);
        result.accepted_tokens[i] = if code >= 0 { h_tokens[i] } else { v_tokens[i] };
    }

    result
}

// ---------------------------------------------------------------------------
// T5: Ternary SIMD fusion gate (optional, requires plasma_path)
// ---------------------------------------------------------------------------
// REMOVED (Issue 772 B2): `ternary_fusion_gate` + `ConsensusConfig.use_ternary_gate`
// shipped with zero callers and no weight source — a linear 6-feature gate
// cannot encode route_one's threshold-cascade semantics, and this lane is
// opt-in after its G1 quality FAIL (Issue 136). `simd_ternary_matvec` remains
// plasma-path substrate with its own consumers.

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
    /// Streaming per-position target distribution (Issue 651, mirrors 587
    /// T5): `probs_buf` holds `p_i = P(· | anchor, accepted d_0..d_{i-1})`
    /// exactly when position `i` is tested. The old `(MAX_DRAFT_WIDTH+1) ×
    /// vocab` `p_flat` buffer + `forward_scratch` are deleted.
    probs_buf: Vec<f32>,
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
        let inv_target_temp = 1.0 / target_temp;
        let draft_width = self.draft_width;
        let vocab_size = self.target_config.vocab_size;
        let policy = self.consensus_config.accept_policy;

        // ── Phase 0: Score initial token with target model ──────────
        // probs_buf ends holding p_0 = P(draft position 0 | anchor).
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
            softmax_scaled(&mut self.probs_buf, inv_target_temp);
        }

        // ── Phase 1: Path V — D2F block draft ──────────────────
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

        // Copy D2F tokens to stack.
        let mut v_tokens = [0usize; MAX_DRAFT_WIDTH];
        let mut v_conf = [0.0f32; MAX_DRAFT_WIDTH];
        let k_bounded = k.min(MAX_DRAFT_WIDTH);
        v_tokens[..k_bounded].copy_from_slice(&v_tokens_raw[..k_bounded]);

        // Extract D2F confidence per position from the logits (draft side —
        // independent of the target scoring loop below).
        for i in 0..k_bounded {
            let logits_offset = i * draft_config.vocab_size;
            if logits_offset + draft_config.vocab_size <= self.d2f_ctx.logits_flat.len() {
                use katgpt_core::simd::fast_exp;

                let logits_p = &self.d2f_ctx.logits_flat
                    [logits_offset..logits_offset + draft_config.vocab_size];
                let max_logit = simd_max_f32(logits_p);
                let mut sum_exp = 0.0f32;
                let mut top1 = 0.0f32;
                for &l in logits_p {
                    let p = fast_exp(l - max_logit);
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

        // ── Phases 2–5: aligned interleaved consensus + routing + verify ──
        // (Issue 651 — the 587-aligned streaming pattern.)
        //
        // probs_buf holds p_i = P(· | anchor, accepted d_0..d_{i-1}) exactly
        // when position i is tested. Per position:
        //   1. H scores p_i: h_i = argmax(p_i), h_conf_i = max(p_i) — the
        //      target's greedy proposal conditioned on the ACCEPTED prefix
        //      (the pre-651 code conditioned on a v_0/h hybrid and tested
        //      d_i against the position-i+1 law — the 587 off-by-one).
        //   2. Ternary consensus + thermal route (per-position kernels).
        //   3. Plasma/Hot: accept the winner unverified (biased by design).
        //      Warm/Cold: run the accept-policy step against p_i.
        //   4. Feed the ACCEPTED token → p_{i+1} (skipped on rejection —
        //      the loop stops; the bonus feed covers the last position).
        let mut accepted: Vec<usize> = Vec::with_capacity(k_bounded + 1);
        let mut all_accepted = true;
        // Issue 973: NucleusAccept's x̃ draws ride their own stream.
        #[cfg(feature = "nucleus_accept")]
        let mut verify_rng = Rng::new(rng.next() ^ crate::d2f_verifier::NUCLEUS_VERIFY_DOMAIN);

        for i in 0..k_bounded {
            // 1. H scores the aligned distribution.
            let mut h_tok = 0usize;
            let mut h_c = f32::NEG_INFINITY;
            for (idx, &p) in self.probs_buf.iter().enumerate() {
                if p > h_c {
                    h_c = p;
                    h_tok = idx;
                }
            }
            if h_c == f32::NEG_INFINITY {
                h_c = 0.0;
            }

            // 2. Consensus + route.
            let (code, consensus_tok) = ternary_consensus_one(h_tok, v_tokens[i], h_c, v_conf[i]);
            let path = route_one(code, h_c, v_conf[i], &self.consensus_config);

            // 3. Selective verification.
            match path {
                ThermalPath::Plasma | ThermalPath::Hot => {
                    accepted.push(consensus_tok);
                }
                ThermalPath::Warm | ThermalPath::Cold => {
                    let p_dist = &self.probs_buf[..vocab_size];
                    let step = match policy {
                        DraftAcceptPolicy::PrefixMatch => prefix_match_step(p_dist, consensus_tok),
                        DraftAcceptPolicy::SoftmaxArgmax | DraftAcceptPolicy::ExactQ => {
                            // Eq 21; ExactQ ≡ Eq 21 under the consensus
                            // winner's point-mass proposal law.
                            softmax_argmax_step(p_dist, consensus_tok, rng)
                        }
                        DraftAcceptPolicy::TruncatedArgmax => {
                            truncated_argmax_step(p_dist, consensus_tok, rng)
                        }
                        // Issue 973: same rule; x̃ on the domain-tagged stream.
                        #[cfg(feature = "nucleus_accept")]
                        DraftAcceptPolicy::NucleusAccept { rho, eps, greedy } => {
                            nucleus_accept_step(
                                p_dist,
                                consensus_tok,
                                rho,
                                eps,
                                greedy,
                                &mut verify_rng,
                            )
                        }
                    };
                    match step {
                        PolicyStep::Accept => accepted.push(consensus_tok),
                        PolicyStep::Correct(y) => {
                            accepted.push(y);
                            all_accepted = false;
                            break;
                        }
                    }
                }
            }

            // 4. Advance: feed the accepted token → p_{i+1}.
            if all_accepted && i + 1 < k_bounded {
                let logits = forward(
                    &mut self.target_ctx,
                    self.target_weights,
                    &mut self.target_cache,
                    consensus_tok,
                    pos + 1 + i,
                    self.target_config,
                );
                self.probs_buf.copy_from_slice(logits);
                softmax_scaled(&mut self.probs_buf, inv_target_temp);
            }
        }

        // ── Bonus token if all accepted ─────────────────────
        // p(· | anchor, d_0..d_{k-1}) — one extra feed after the last accept.
        if all_accepted {
            let last = accepted[k_bounded - 1];
            let logits = forward(
                &mut self.target_ctx,
                self.target_weights,
                &mut self.target_cache,
                last,
                pos + k_bounded,
                self.target_config,
            );
            self.probs_buf.copy_from_slice(logits);
            softmax_scaled(&mut self.probs_buf, inv_target_temp);
            accepted.push(sample_from_distribution(&self.probs_buf, rng));
        }

        // Safety: always return at least one token (position 0 always pushes
        // accept-or-correction; the k==0 early return above covers the rest).\n        debug_assert!(!accepted.is_empty());
        accepted.shrink_to_fit();
        accepted
    }
}

// ---------------------------------------------------------------------------
// Tests — all 10 moved from root (no training dependencies).
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
        // Issue 651: Eq 21 (SoftmaxArgmax) is the default verified-path policy.
        assert_eq!(config.accept_policy, DraftAcceptPolicy::SoftmaxArgmax);
    }

    // ── Issue 651: exact verified paths ─────────────────────────────────────

    /// Issue 651 G1 harness (mirrors the 587 T2 E2E level): force every
    /// position to Warm/Cold (thresholds > 1.0 — confidences are ≤ 1), then
    /// measure the empirical marginal of the FIRST accepted token against the
    /// reference target law p_0 = softmax(forward(anchor)). Position 0 is
    /// always verified under this config, and its accept-or-correct step is
    /// the only contributor to the first output token — so the marginal must
    /// match p_0 exactly under Eq 21, and collapse to a point mass under
    /// PrefixMatch.
    #[test]
    fn test_eq21_exactness_first_token_all_warm_cold() {
        let mut config = Config::micro();
        config.vocab_size = 64;
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

        // Reference: p_0 = softmax_scaled(forward(anchor), 1/temp).
        let mut ctx = crate::ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits = crate::forward(
            &mut ctx,
            &target_weights,
            &mut cache,
            config.bos_token,
            0,
            &config,
        );
        let mut p0: Vec<f32> = logits.to_vec();
        softmax_scaled(&mut p0, 1.0 / config.temperature);

        let n_rounds = 8000usize;
        let run_arm = |policy: DraftAcceptPolicy| -> Vec<f32> {
            let consensus = ConsensusConfig {
                plasma_threshold: 2.0, // unreachable: conf ≤ 1
                hot_threshold: 2.0,
                accept_policy: policy,
                ..Default::default()
            };
            let mut verifier = FlashARConsensusVerifier::new(
                &target_weights,
                &config,
                D2fDecodeConfig::with_block_size(4),
                consensus,
                4,
            );
            let mut hist = vec![0u32; config.vocab_size];
            for round in 0..n_rounds {
                let out = verifier.speculate(
                    &draft_weights,
                    &config,
                    config.bos_token,
                    0,
                    &mut Rng::new(round as u64),
                );
                hist[out[0]] += 1;
            }
            hist.iter().map(|&c| c as f32 / n_rounds as f32).collect()
        };

        let tv = |emp: &[f32]| -> f32 {
            0.5 * emp
                .iter()
                .zip(p0.iter())
                .map(|(e, p)| (e - p).abs())
                .sum::<f32>()
        };

        let tv_eq21 = tv(&run_arm(DraftAcceptPolicy::SoftmaxArgmax));
        let tv_prefix = tv(&run_arm(DraftAcceptPolicy::PrefixMatch));

        println!("eq21 TV = {tv_eq21:.4}, prefix-match TV = {tv_prefix:.4}");
        // Eq 21: exact up to sampling noise (n=8000, vocab 64 → expected
        // TV noise ≈ 0.02-0.04; 0.06 is the honest gate).
        assert!(
            tv_eq21 < 0.06,
            "SoftmaxArgmax first-token marginal must match p_0 (TV={tv_eq21:.4})"
        );
        // PrefixMatch: collapses to the argmax point mass — TV ≈ 1 − p0(am).
        assert!(
            tv_prefix > 0.3,
            "PrefixMatch must collapse toward the mode (TV={tv_prefix:.4})"
        );
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
