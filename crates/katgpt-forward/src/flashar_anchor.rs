//! FlashAR Strided Anchor-Then-Fill D2F Decoding
//!
//! Plan 166 T11 (stretch goal): Two-round decoding inspired by FlashAR's
//! diagonal-step parallel decoding pattern.
//!
//! # Architecture
//!
//! Round 1 (Anchor — diagonal analog):
//!   AR predicts every S-th position. Stride S controls anchor density.
//!   Few AR forward passes (block_size / stride) produce high-quality anchor tokens.
//!
//! Round 2 (Fill — parallel denoising):
//!   D2F decodes the remaining positions with anchor tokens pre-filled.
//!   Anchor positions start unmasked, reducing the denoising search space.
//!   Expected: fewer denoising iterations → faster convergence.
//!
//! # Feature Gate
//!
//! `flashar_anchor` (requires `dllm`)
//!
//! Plan 400 (2026-07-05): moved from root `src/speculative/flashar_anchor.rs`.
//! The two training-coupled tests (`test_anchor_then_fill_produces_valid_output`,
//! `test_anchor_then_fill_reduces_steps`) stayed in the root shim file because
//! they call `crate::dllm::{train_mini_dllm, generate_pattern_dataset}` which
//! is root-only training code. The 6 inference-only tests moved with this
//! file. Root re-exports via `pub use katgpt_forward::flashar_anchor::*` so
//! all historical `katgpt_rs::speculative::flashar_anchor::*` paths resolve.

#![allow(clippy::too_many_arguments)]

use crate::d2f::{D2fBlockResult, D2fDecodeConfig};
#[cfg(feature = "entropy_bounded_commit")]
use crate::d2f::{D2fEbScratch, d2f_commit_set_eb};
use crate::d2f_context::D2fContext;
use crate::{ForwardContext, forward};
use katgpt_core::traits::{NoPruner, NoScreeningPruner};
use katgpt_transformer::{MultiLayerKVCache, TransformerWeights};
use katgpt_types::{Config, Rng, softmax_scaled};

use katgpt_core::speculative::sampling::sample_from_distribution;

// ---------------------------------------------------------------------------
// Anchor-Then-Fill Configuration
// ---------------------------------------------------------------------------

/// Configuration for strided anchor-then-fill decoding.
#[derive(Clone, Copy, Debug)]
pub struct AnchorConfig {
    /// Stride S: predict every S-th position via AR in Round 1.
    /// S=1 → pure AR (every position anchored).
    /// S=block_size → pure D2F (no anchors).
    /// Recommended: 2–4 for balanced anchor density.
    pub stride: usize,
}

impl Default for AnchorConfig {
    fn default() -> Self {
        Self { stride: 2 }
    }
}

impl AnchorConfig {
    pub fn with_stride(stride: usize) -> Self {
        Self {
            stride: stride.max(1),
        }
    }
}

/// Configuration for the DBTM confidence-commit anchor rule (Issue 811 /
/// Plan 600 — arXiv:2609.15903 Eq 26).
///
/// Replaces the stride rule's positional anchor selection with
/// content-adaptive selection over the Round-1 AR walk's own confidence:
/// a position is anchored when its max-softmax q ≥ `kappa` (the committed
/// token is the argmax proposal). `floor = true` additionally arms the
/// [`dbtm_floor`] per-round commit floor in the fill round, which
/// guarantees the block empties within the step budget.
///
/// Measured (Issue 811 PoC): 3.1–4.7× fewer fill steps than the
/// matched-threshold stride arm at κ ∈ {0.9, 0.99} at quality parity on the
/// mini-D2F; termination property-proven at every (κ, k) cell.
///
/// Opt-in — [`AnchorConfig`] (stride) stays the default.
#[derive(Clone, Copy, Debug)]
pub struct ConfidenceAnchorConfig {
    /// Confidence threshold κ ∈ (0, 1]: anchor positions with q ≥ kappa.
    pub kappa: f32,
    /// Arm the DBTM per-round commit floor in the fill round.
    pub floor: bool,
}

impl ConfidenceAnchorConfig {
    /// # Panics
    /// Unless `0 < kappa <= 1` (constructor contract, not hot path).
    pub fn new(kappa: f32, floor: bool) -> Self {
        assert!(
            kappa > 0.0 && kappa <= 1.0,
            "kappa must be in (0, 1], got {kappa}"
        );
        Self { kappa, floor }
    }
}

impl Default for ConfidenceAnchorConfig {
    /// The promoted decode default (Plan 601, 2026-09-17): κ=0.9 + floor —
    /// the balanced GOAT cell of the real-text gate run
    /// (`.benchmarks/601_flashar_realtext_goat.md`): paired Δ(acc) +0.069
    /// over the strided incumbent, 1.5× fewer fill steps, wall 0.87×, and
    /// realized-KL ratio 0.67 on held-out Austen. Promotion satisfied Plan
    /// 600's acceptance bar (T8 all-green + T9 on real text); the strided
    /// [`AnchorConfig`] entry stays as the no-floor comparator.
    fn default() -> Self {
        Self {
            kappa: 0.9,
            floor: true,
        }
    }
}

/// Confidence-commit anchor selection: anchor every position whose
/// max-softmax confidence `q` ≥ `kappa`, committing the argmax proposal.
/// Positions whose argmax IS the mask token are never anchored.
///
/// Writes the anchor buffer into `out` (`mask_token` at unselected
/// positions) and returns the number of anchors selected. Allocation-free
/// form of [`select_confidence_anchors`] — the production
/// [`anchor_then_fill_with`] runs this on its pre-allocated Round-1 buffer
/// instead of replacing it with a fresh `Vec`.
///
/// # Panics
/// If `out.len() != argmax.len()` or the argmax/probs lengths mismatch.
pub fn select_confidence_anchors_into(
    argmax: &[usize],
    probs: &[f32],
    mask_token: usize,
    kappa: f32,
    out: &mut [usize],
) -> usize {
    assert_eq!(argmax.len(), probs.len(), "argmax/probs length mismatch");
    assert_eq!(out.len(), argmax.len(), "out buffer length mismatch");
    out.fill(mask_token);
    let mut n = 0usize;
    for (p, (&tok, &q)) in argmax.iter().zip(probs.iter()).enumerate() {
        if q >= kappa && tok != mask_token {
            out[p] = tok;
            n += 1;
        }
    }
    n
}

/// Allocating form of [`select_confidence_anchors_into`] — public API kept
/// stable for the Issue 811 PoC harness and external callers.
///
/// Returns the anchor buffer (`block_size` entries, `mask_token` at
/// unselected positions) and the number of anchors selected.
pub fn select_confidence_anchors(
    argmax: &[usize],
    probs: &[f32],
    mask_token: usize,
    kappa: f32,
) -> (Vec<usize>, usize) {
    let mut buf = vec![mask_token; argmax.len()];
    let n = select_confidence_anchors_into(argmax, probs, mask_token, kappa, &mut buf);
    (buf, n)
}

/// Result of the two-round anchor-then-fill decode.
#[derive(Clone, Debug)]
pub struct AnchorFillResult {
    /// Final decoded tokens for the block.
    pub tokens: Vec<usize>,
    /// Number of anchor positions predicted in Round 1.
    pub n_anchors: usize,
    /// Number of denoising steps used in Round 2.
    pub fill_steps_used: usize,
    /// Denoising steps used by baseline D2F (no anchors) for comparison.
    pub baseline_steps_used: usize,
    /// Reduction in denoising steps vs baseline.
    pub step_reduction: usize,
}

// ---------------------------------------------------------------------------
// Round 1: Strided AR Anchor Prediction
// ---------------------------------------------------------------------------

/// Predict anchor tokens at every S-th position using AR forward passes.
///
/// Returns the number of anchor tokens written into `token_buf`.
/// Anchor positions are `stride-1, 2*stride-1, 3*stride-1, ...` (0-indexed
/// within the block). Positions between anchors remain `mask_token`.
///
/// `argmax_out` / `probs_out` (Plan 600 T7): when `Some`, the walk also
/// records the per-position (argmax proposal, max-softmax confidence q) —
/// the signal the DBTM confidence-commit rule consumes. Pure observation:
/// the sampled context propagation and rng stream are byte-identical to the
/// `None` walk, which the parity tests pin.
#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_anchors(
    ctx: &mut ForwardContext,
    cache: &mut MultiLayerKVCache,
    weights: &TransformerWeights,
    config: &Config,
    seed_token: usize,
    start_pos: usize,
    block_size: usize,
    stride: usize,
    mask_token: usize,
    token_buf: &mut [usize],
    rng: &mut Rng,
    argmax_out: Option<&mut [usize]>,
    probs_out: Option<&mut [f32]>,
) -> usize {
    use katgpt_core::simd::simd_argmax_f32;
    let vocab = config.vocab_size;
    let temperature = config.temperature;
    let mut n_anchors = 0usize;
    let mut argmax_buf = argmax_out;
    let mut probs_buf = probs_out;

    // Initialize all block positions to mask
    for t in token_buf.iter_mut().take(block_size) {
        *t = mask_token;
    }

    // AR walk: predict tokens sequentially, but only "anchor" at stride positions
    let mut cur_token = seed_token;
    let mut ar_logits_buf = vec![0.0f32; vocab];

    for (pos_in_block, slot) in token_buf.iter_mut().enumerate().take(block_size) {
        let global_pos = start_pos + pos_in_block;

        // Forward pass at this position
        let logits = forward(ctx, weights, cache, cur_token, global_pos, config);
        ar_logits_buf.copy_from_slice(logits);

        // Sample from logits
        softmax_scaled(&mut ar_logits_buf, 1.0 / temperature);
        let next_token = sample_from_distribution_weighted(&ar_logits_buf, rng);

        // Observation only (Plan 600 T7): no extra forward, no rng draw —
        // the sampled-context walk stays byte-identical to the None walk.
        let (best, best_prob) = simd_argmax_f32(&ar_logits_buf);
        if let Some(b) = argmax_buf.as_deref_mut() {
            b[pos_in_block] = best;
        }
        if let Some(p) = probs_buf.as_deref_mut() {
            p[pos_in_block] = best_prob;
        }

        // Anchor: store at stride positions (stride-1, 2*stride-1, ...)
        if (pos_in_block + 1) % stride == 0 {
            *slot = next_token;
            n_anchors += 1;
        }

        cur_token = next_token;
    }

    n_anchors
}

// ---------------------------------------------------------------------------
// DBTM confidence-commit floor (Issue 811 / Research 563)
// ---------------------------------------------------------------------------

/// Per-round commit floor `n_r = ⌈|R_r| / (k − r + 1)⌉` (DBTM Eq 26, round
/// `r` of budget `k`, 1-indexed).
///
/// Committing at least `n_r` positions each round guarantees the block
/// empties in exactly `k` rounds: at `r = k` the denominator is 1, so the
/// floor is the whole remainder. Pure arithmetic — the caller owns the
/// selection (highest-confidence first).
///
/// Rounds beyond `budget` (a loop that outlived the budget) clamp to a
/// denominator of 1, i.e. "commit everything remaining".
#[inline]
pub fn dbtm_floor(remaining: usize, round: usize, budget: usize) -> usize {
    let denom = budget.saturating_sub(round.saturating_sub(1)).max(1);
    remaining.div_ceil(denom)
}

// ---------------------------------------------------------------------------
// Round 2: D2F Fill with Anchors Pre-filled
// ---------------------------------------------------------------------------

/// Run D2F denoising with anchor positions pre-filled in the token buffer.
///
/// This is a modified version of `d2f_decode_block_with_prompt_with` that
/// accepts pre-filled anchor tokens instead of starting from all-mask.
///
/// `commit_budget = Some(k)` arms the DBTM commit rule (Issue 811): after
/// the usual `τ_conf` threshold commits of round `r`, the
/// `n_r = dbtm_floor(remaining, r, k)` highest-confidence still-masked
/// positions are committed regardless of threshold — the
/// `Δ𝒞_r = {q ≥ κ} ∪ top_{n_r}(q)` union. `None` keeps the incumbent
/// threshold-only semantics unchanged.
#[allow(clippy::too_many_arguments)]
fn fill_with_anchors(
    dctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    prompt: &[usize],
    anchor_tokens: &[usize],
    pruner: &dyn katgpt_core::traits::ConstraintPruner,
    screener: &dyn katgpt_core::traits::ScreeningPruner,
    rng: &mut Rng,
    commit_budget: Option<usize>,
) -> D2fBlockResult {
    // Use the prompt + anchor-initialized block
    let mask = config.mask_token;
    let vocab = config.vocab_size;
    let block_size = decode_config.block_size;
    let seq_len = (prompt.len() + block_size).min(config.block_size);
    let block_start = prompt.len();
    let max_steps = decode_config.denoise_steps;
    let tau_conf = decode_config.confidence_threshold;
    let _temperature = decode_config.temperature;
    // Issue 917 T2, cousin lane (feature `entropy_bounded_commit`): the EB
    // pass-commit policy is armed by `eb_gamma >= 0`; the negative default
    // keeps the incumbent τ branch byte-identical. The proxy reads the raw
    // pass-start logits row this lane's forward wrote (the incumbent's
    // best_prob is pruner-screened + relevance-weighted and is NOT re-checked
    // armed — membership in the EB set is the decision). The DBTM commit
    // floor (Issue 811) composes unchanged: armed, only EB-selected
    // positions sample, so the floor's candidates are the EB pass's
    // proposals.
    #[cfg(feature = "entropy_bounded_commit")]
    let eb_armed = decode_config.eb_gamma >= 0.0;
    #[cfg(feature = "entropy_bounded_commit")]
    let mut eb_scratch = D2fEbScratch::new(block_size);

    // Initialize: prompt + anchor-prefilled tokens
    let mut tokens: Vec<usize> = prompt.to_vec();
    // Copy anchor tokens (non-mask positions already filled)
    tokens.extend_from_slice(anchor_tokens);
    tokens.truncate(config.block_size);

    let mut confidence_history = Vec::with_capacity(max_steps);
    let mut converged_step = max_steps;

    // Scratch buffer reused across all positions and steps — fused
    // single-pass sampling (see below). `exp_scratch[t]` caches
    // `exp(logit_t − max)` so the second sampling pass can read it without
    // re-doing the transcendentals. Per-position we also stash
    // `exp(logit-max) · relevance` into `sum_exp` directly (it's a scalar,
    // not a slice). Allocating once here and overwriting per-position
    // halves the `.exp()` calls per token in the hot path.
    let mut exp_scratch = vec![0.0f32; vocab];
    // DBTM floor candidates (Issue 811): (position, prob, token) for every
    // masked position that produced a proposal this round. Pre-allocated to
    // the block size (the provable per-round upper bound: one proposal per
    // masked position) and reused across steps — the push/retain/clear
    // cycle never allocates, in any round, ever (Plan 600 G4).
    let mut round_candidates: Vec<(usize, f32, usize)> = Vec::with_capacity(block_size);

    for step in 0..max_steps {
        let _seq_len_actual = crate::d2f_context::forward_block_causal_with(
            dctx,
            weights,
            &tokens[..seq_len],
            config,
            block_size,
        );

        // Issue 917 T2 (cousin lane): armed, the pass commit set is fixed
        // ONCE per pass from the pass-start logits; the already-committed
        // count comes from the pass-start masked count. Disarmed (default),
        // the legacy loop-top counting runs unchanged.
        #[cfg(feature = "entropy_bounded_commit")]
        let mut n_confident = if eb_armed {
            let (_, masked) = d2f_commit_set_eb(
                &dctx.logits_flat,
                &tokens,
                block_start,
                seq_len,
                mask,
                vocab,
                &mut eb_scratch,
                decode_config.eb_gamma,
                decode_config.eb_max_commit,
            );
            (seq_len - block_start) - masked
        } else {
            0
        };
        #[cfg(not(feature = "entropy_bounded_commit"))]
        let mut n_confident = 0usize;

        for p in block_start..seq_len {
            // Skip positions that are already filled (anchors or previously
            // denoised). Issue 917 T2 (cousin lane): armed, unselected
            // positions (already committed OR held for a later pass) are
            // skipped without sampling.
            #[cfg(feature = "entropy_bounded_commit")]
            if eb_armed {
                if !eb_scratch.is_selected(p - block_start) {
                    continue;
                }
            } else if tokens[p] != mask {
                n_confident += 1;
                continue;
            }
            #[cfg(not(feature = "entropy_bounded_commit"))]
            if tokens[p] != mask {
                n_confident += 1;
                continue;
            }

            let logits_start = p * vocab;
            let logits_end = logits_start + vocab;
            let logits_p = &dctx.logits_flat[logits_start..logits_end];
            let max_logit = logits_p.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let depth = p - block_start;
            let parent_tokens = &tokens[block_start..p];

            // ── Fused single pass: compute weights + sum_exp together ──
            // The original two-pass code had an asymmetry: `sum_exp` included
            // `* relevance` but the second-pass `cum` did NOT. We preserve that
            // exact semantics by caching `exp(logit-max)` in `exp_scratch` and
            // `exp(logit-max) * relevance` in `weights_scratch`. The second
            // pass reads from `exp_scratch` instead of recomputing `.exp()`.
            // Scalar `fast_exp` (not SIMD): mask + pruner branches are interleaved.
            use katgpt_core::simd::fast_exp;
            let mut sum_exp = 0.0f32;
            for t in 0..vocab {
                if t == mask {
                    exp_scratch[t] = 0.0;
                    continue;
                }
                if !pruner.is_valid(depth, t, parent_tokens) {
                    exp_scratch[t] = 0.0;
                    continue;
                }
                let relevance = screener.relevance(depth, t, parent_tokens);
                let e = fast_exp(logits_p[t] - max_logit);
                exp_scratch[t] = e;
                // The original code's sum_exp multiplied by relevance (even for
                // negative relevance, which is unusual but kept for parity).
                sum_exp += e * relevance;
            }

            if sum_exp == 0.0 {
                continue;
            }

            // ── Sample via cumulative sum over the cached `exp_scratch` ──
            // The original sampling loop used UNWEIGHTED `exp(logit-max)`
            // (no relevance). We read from `exp_scratch[t]` to avoid the
            // second `.exp()` call per token.
            let threshold = rng.uniform() * sum_exp;
            let mut best_token = mask;
            let mut cum = 0.0f32;
            for (t, &exp_val) in exp_scratch.iter().enumerate().take(vocab) {
                if t == mask || !pruner.is_valid(depth, t, parent_tokens) {
                    continue;
                }
                cum += exp_val;
                if cum >= threshold && best_token == mask {
                    best_token = t;
                    // Keep iterating to preserve the original loop semantics
                    // (no early break — matches d2f.rs reference).
                }
            }

            // Compute probability of chosen token from the cached `exp_scratch`.
            // Original semantics: UNWEIGHTED exp divided by WEIGHTED sum_exp.
            let best_prob = if best_token != mask {
                exp_scratch[best_token] / sum_exp
            } else {
                0.0
            };

            // Record the proposal for the DBTM floor before the threshold
            // gate — the floor must see every candidate, committed or not.
            // Gated so the incumbent (budget-less) path does no extra work.
            if commit_budget.is_some() && best_token != mask {
                round_candidates.push((p, best_prob, best_token));
            }

            // Issue 917 T2 (cousin lane): armed, membership in the pass's
            // EB set IS the decision; disarmed, the incumbent τ test.
            #[cfg(feature = "entropy_bounded_commit")]
            let accept = eb_armed || best_prob >= tau_conf;
            #[cfg(not(feature = "entropy_bounded_commit"))]
            let accept = best_prob >= tau_conf;
            if accept && best_token != mask {
                tokens[p] = best_token;
                n_confident += 1;
            }
        }

        // ── DBTM commit floor (Issue 811): Δ𝒞_r = {q ≥ κ} ∪ top_{n_r}(q) ──
        // The threshold commits above are the {q ≥ κ} half; the floor adds
        // the n_r highest-confidence remaining proposals so the block is
        // guaranteed to empty within the budget (at r = k the floor is the
        // whole remainder).
        if let Some(budget) = commit_budget {
            let round = step + 1;
            round_candidates.retain(|&(p, _, _)| tokens[p] == mask);
            let n_floor = dbtm_floor(round_candidates.len(), round, budget);
            if n_floor > 0 {
                round_candidates
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                for &(p, _, tok) in round_candidates.iter().take(n_floor) {
                    tokens[p] = tok;
                }
            }
        }
        round_candidates.clear();

        let confidence = n_confident as f32 / block_size as f32;
        confidence_history.push(confidence);

        // Early exit: all block positions unmasked.
        if tokens[block_start..seq_len].iter().all(|&t| t != mask) {
            converged_step = step;
            break;
        }
    }

    let all_unmasked = tokens[block_start..seq_len].iter().all(|&t| t != mask);
    let final_confidence = confidence_history.last().copied().unwrap_or(0.0);

    let state = if all_unmasked {
        crate::d2f::D2fBlockState::FullyActivated
    } else {
        crate::d2f::D2fBlockState::SemiActivated {
            step: converged_step.min(max_steps - 1),
            confidence: final_confidence,
        }
    };

    let block_tokens: Vec<usize> = tokens[block_start..seq_len].to_vec();

    crate::d2f::D2fBlockResult {
        tokens: block_tokens,
        steps_used: confidence_history.len(),
        confidence_history,
        accuracy: None,
        state,
    }
}

// ---------------------------------------------------------------------------
// Public API: Anchor-Then-Fill Decode
// ---------------------------------------------------------------------------

/// Run two-round anchor-then-fill D2F decoding.
///
/// 1. **Round 1 (Anchor):** AR predicts every S-th token.
/// 2. **Round 2 (Fill):** D2F denoises remaining positions with anchors pre-filled.
///
/// Also runs a baseline D2F decode (no anchors) to measure step reduction.
pub fn anchor_then_fill(
    ctx: &mut ForwardContext,
    cache: &mut MultiLayerKVCache,
    dctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    anchor_config: &AnchorConfig,
    seed_token: usize,
    start_pos: usize,
    rng: &mut Rng,
) -> AnchorFillResult {
    let block_size = decode_config.block_size;
    let mask = config.mask_token;

    // ── Round 1: Strided AR anchor prediction ──
    let mut anchor_buf = vec![mask; block_size];
    let n_anchors = predict_anchors(
        ctx,
        cache,
        weights,
        config,
        seed_token,
        start_pos,
        block_size,
        anchor_config.stride,
        mask,
        &mut anchor_buf,
        rng,
        None,
        None,
    );

    // ── Round 2: D2F fill with anchors ──
    let fill_result = fill_with_anchors(
        dctx,
        weights,
        config,
        decode_config,
        &[],
        &anchor_buf,
        &NoPruner,
        &NoScreeningPruner,
        rng,
        None,
    );

    // ── Baseline: D2F without anchors (for comparison) ──
    let baseline_result = crate::d2f::d2f_decode_block_with_prompt_with(
        dctx,
        weights,
        config,
        decode_config,
        &[],
        &NoPruner,
        &NoScreeningPruner,
        rng,
    );

    AnchorFillResult {
        tokens: fill_result.tokens,
        n_anchors,
        fill_steps_used: fill_result.steps_used,
        baseline_steps_used: baseline_result.steps_used,
        step_reduction: baseline_result
            .steps_used
            .saturating_sub(fill_result.steps_used),
    }
}

/// Run two-round confidence-commit decoding (DBTM κ ∪ floor, Issue 811 /
/// Plan 600).
///
/// Same two-round shape as [`anchor_then_fill`], with the anchor selection
/// driven by the walk's own confidence ([`select_confidence_anchors`])
/// instead of a position stride, and — with [`ConfidenceAnchorConfig::floor`]
/// — the [`dbtm_floor`] per-round commit floor in the fill round.
///
/// The incumbent [`anchor_then_fill`] is untouched (byte-identical path and
/// behavior); this entry exists so the two rules can race behind the same
/// `flashar_anchor` feature until the GOAT gates decide the default.
#[allow(clippy::too_many_arguments)]
pub fn anchor_then_fill_with(
    ctx: &mut ForwardContext,
    cache: &mut MultiLayerKVCache,
    dctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    confidence_config: &ConfidenceAnchorConfig,
    seed_token: usize,
    start_pos: usize,
    rng: &mut Rng,
) -> AnchorFillResult {
    let block_size = decode_config.block_size;
    let mask = config.mask_token;

    // ── Round 1: AR walk with confidence observation ──
    let mut anchor_buf = vec![mask; block_size];
    let mut argmax_buf = vec![0usize; block_size];
    let mut probs_buf = vec![0.0f32; block_size];
    predict_anchors(
        ctx,
        cache,
        weights,
        config,
        seed_token,
        start_pos,
        block_size,
        1, // stride value irrelevant: every position observed, selection below
        mask,
        &mut anchor_buf,
        rng,
        Some(&mut argmax_buf),
        Some(&mut probs_buf),
    );
    let n_anchors = select_confidence_anchors_into(
        &argmax_buf,
        &probs_buf,
        mask,
        confidence_config.kappa,
        &mut anchor_buf,
    );

    // ── Round 2: D2F fill (threshold + optional DBTM floor) ──
    let fill_result = fill_with_anchors(
        dctx,
        weights,
        config,
        decode_config,
        &[],
        &anchor_buf,
        &NoPruner,
        &NoScreeningPruner,
        rng,
        if confidence_config.floor {
            Some(decode_config.denoise_steps)
        } else {
            None
        },
    );

    // ── Baseline: D2F without anchors (for comparison) ──
    let baseline_result = crate::d2f::d2f_decode_block_with_prompt_with(
        dctx,
        weights,
        config,
        decode_config,
        &[],
        &NoPruner,
        &NoScreeningPruner,
        rng,
    );

    AnchorFillResult {
        tokens: fill_result.tokens,
        n_anchors,
        fill_steps_used: fill_result.steps_used,
        baseline_steps_used: baseline_result.steps_used,
        step_reduction: baseline_result
            .steps_used
            .saturating_sub(fill_result.steps_used),
    }
}

/// Fill round with arbitrary pre-filled anchor tokens (Issue 811 PoC seam).
///
/// The production [`anchor_then_fill`] fixes Round-1 anchor selection to a
/// position stride. This seam hands the caller the anchor buffer instead —
/// a confidence-commit rule (or any other selector) can be measured against
/// the incumbent through the SAME fill path, which is what makes the
/// comparison apples-to-apples. Pass an all-`mask_token` buffer to run the
/// plain D2F baseline.
///
/// `commit_budget = Some(k)` arms the DBTM per-round floor
/// ([`dbtm_floor`]); `None` is the incumbent threshold-only fill.
///
/// No default-path behavior change: `anchor_then_fill` keeps its stride
/// selection and threshold-only fill.
#[allow(clippy::too_many_arguments)]
pub fn anchor_fill_with_prefilled(
    dctx: &mut D2fContext,
    weights: &TransformerWeights,
    config: &Config,
    decode_config: &D2fDecodeConfig,
    anchor_tokens: &[usize],
    rng: &mut Rng,
    commit_budget: Option<usize>,
) -> D2fBlockResult {
    fill_with_anchors(
        dctx,
        weights,
        config,
        decode_config,
        &[],
        anchor_tokens,
        &NoPruner,
        &NoScreeningPruner,
        rng,
        commit_budget,
    )
}

// ---------------------------------------------------------------------------
// Helper: Weighted sampling from probability distribution
// ---------------------------------------------------------------------------

/// Sample from a probability distribution (sums to ~1.0).
/// Reuses the pattern from `sample_from_distribution` but works on raw probs.
fn sample_from_distribution_weighted(probs: &[f32], rng: &mut Rng) -> usize {
    sample_from_distribution(probs, rng)
}

// ---------------------------------------------------------------------------
// Tests — 6 inference-only (PURE) tests moved from root.
// The 2 training-coupled tests stayed in root's `src/speculative/flashar_anchor.rs`
// shim file because they need `crate::dllm::{train_mini_dllm, generate_pattern_dataset}`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::d2f::D2fDecodeConfig;
    use crate::d2f_context::D2fContext;
    use katgpt_transformer::TransformerWeights;
    use katgpt_types::{Config, Rng};

    fn make_config() -> Config {
        Config::micro_dllm()
    }

    // NOTE: `make_trained_weights` stayed in the root shim (TRAIN-coupled).

    // ── DBTM floor arithmetic (Issue 811) ──

    #[test]
    fn test_dbtm_floor_known_values() {
        assert_eq!(dbtm_floor(10, 1, 3), 4); // ⌈10/3⌉
        assert_eq!(dbtm_floor(6, 2, 3), 3); // ⌈6/2⌉
        assert_eq!(dbtm_floor(3, 3, 3), 3); // last round = whole remainder
        assert_eq!(dbtm_floor(0, 1, 4), 0); // nothing remaining
        assert_eq!(dbtm_floor(7, 1, 1), 7); // single-round budget commits all
    }

    #[test]
    fn test_dbtm_floor_terminates_within_budget() {
        // The DBTM termination guarantee, simulated at the floor's worst case
        // (each round commits exactly the floor, never more): the remainder
        // must hit zero by round == budget, for every (remaining, budget).
        for remaining in 0..=96usize {
            for budget in 1..=12usize {
                let mut r = remaining;
                for round in 1..=budget {
                    let commit = dbtm_floor(r, round, budget).min(r);
                    r -= commit;
                    if r == 0 {
                        break;
                    }
                }
                assert_eq!(r, 0, "floor must empty {remaining} within {budget} rounds");
            }
        }
    }

    // ── Confidence-commit selector + entry point (Plan 600 T6/T7) ──

    #[test]
    fn test_select_confidence_anchors_semantics() {
        let mask = 26usize;
        // q high at 0/2, q low at 1/3; argmax at 2 IS the mask (never anchored).
        let argmax = [5usize, 7, mask, 9];
        let probs = [0.95f32, 0.2, 0.99, 0.4];
        let (buf, n) = select_confidence_anchors(&argmax, &probs, mask, 0.5);
        assert_eq!(n, 1);
        assert_eq!(buf, vec![5, mask, mask, mask]);
        // κ=0 anchors everything except mask-argmax positions.
        let (buf1, n1) = select_confidence_anchors(&argmax, &probs, mask, f32::MIN_POSITIVE);
        assert_eq!(n1, 3);
        assert_eq!(buf1, vec![5, 7, mask, 9]);
        // κ=1 anchors only at certainty.
        let (_, n2) = select_confidence_anchors(&argmax, &probs, mask, 1.0);
        assert_eq!(n2, 0); // 0.99 < 1.0
    }

    #[test]
    #[should_panic(expected = "kappa must be in (0, 1]")]
    fn test_confidence_config_rejects_zero_kappa() {
        let _ = ConfidenceAnchorConfig::new(0.0, true);
    }

    #[test]
    #[should_panic(expected = "kappa must be in (0, 1]")]
    fn test_confidence_config_rejects_kappa_above_one() {
        let _ = ConfidenceAnchorConfig::new(1.5, false);
    }

    #[test]
    fn test_anchor_config_default_stride() {
        let cfg = AnchorConfig::default();
        assert_eq!(cfg.stride, 2);
    }

    #[test]
    fn test_anchor_config_with_stride_clamped() {
        let cfg = AnchorConfig::with_stride(0);
        assert_eq!(cfg.stride, 1);
    }

    #[test]
    fn test_predict_anchors_fills_stride_positions() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let block_size = 8;
        let stride = 2;
        let mask = config.mask_token;

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut token_buf = vec![mask; block_size];

        let n_anchors = predict_anchors(
            &mut ctx,
            &mut cache,
            &weights,
            &config,
            0, // seed_token
            0, // start_pos
            block_size,
            stride,
            mask,
            &mut token_buf,
            &mut rng,
            None,
            None,
        );

        // With stride=2 and block_size=8, anchors at positions 1, 3, 5, 7 → 4 anchors
        assert_eq!(n_anchors, 4);
        // Anchor positions should be non-mask
        for pos in [1, 3, 5, 7] {
            assert_ne!(token_buf[pos], mask, "position {pos} should be anchored");
        }
        // Non-anchor positions should still be mask
        for pos in [0, 2, 4, 6] {
            assert_eq!(token_buf[pos], mask, "position {pos} should still be mask");
        }
    }

    #[test]
    fn test_predict_anchors_stride_4() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let block_size = 8;
        let stride = 4;
        let mask = config.mask_token;

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut token_buf = vec![mask; block_size];

        let n_anchors = predict_anchors(
            &mut ctx,
            &mut cache,
            &weights,
            &config,
            0,
            0,
            block_size,
            stride,
            mask,
            &mut token_buf,
            &mut rng,
            None,
            None,
        );

        // stride=4: anchors at positions 3, 7 → 2 anchors
        assert_eq!(n_anchors, 2);
        assert_ne!(token_buf[3], mask);
        assert_ne!(token_buf[7], mask);
        for pos in [0, 1, 2, 4, 5, 6] {
            assert_eq!(token_buf[pos], mask);
        }
    }

    #[test]
    fn test_anchor_then_fill_stride_1_all_anchored() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let block_size = 8;

        let decode_config = D2fDecodeConfig::with_block_size(block_size);
        let anchor_config = AnchorConfig::with_stride(1); // Every position anchored

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let mut dctx = D2fContext::new(&config);

        let result = anchor_then_fill(
            &mut ctx,
            &mut cache,
            &mut dctx,
            &weights,
            &config,
            &decode_config,
            &anchor_config,
            0,
            0,
            &mut rng,
        );

        // stride=1: all positions are anchors
        assert_eq!(result.n_anchors, block_size);
        // Fill steps should be minimal (nothing to denoise)
        assert!(
            result.fill_steps_used <= 1,
            "with all anchors, fill should converge immediately, got {}",
            result.fill_steps_used
        );
    }

    #[test]
    fn test_anchor_then_fill_deterministic() {
        let config = make_config();
        let block_size = 8;
        let decode_config = D2fDecodeConfig::with_block_size(block_size);
        let anchor_config = AnchorConfig::with_stride(2);

        let mut rng1 = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng1);

        let mut rng2 = Rng::new(42);
        let weights2 = TransformerWeights::new(&config, &mut rng2);

        let mut rng1 = Rng::new(99);
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let mut dctx1 = D2fContext::new(&config);

        let result1 = anchor_then_fill(
            &mut ctx1,
            &mut cache1,
            &mut dctx1,
            &weights,
            &config,
            &decode_config,
            &anchor_config,
            0,
            0,
            &mut rng1,
        );

        let mut rng2 = Rng::new(99);
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let mut dctx2 = D2fContext::new(&config);

        let result2 = anchor_then_fill(
            &mut ctx2,
            &mut cache2,
            &mut dctx2,
            &weights2,
            &config,
            &decode_config,
            &anchor_config,
            0,
            0,
            &mut rng2,
        );

        assert_eq!(
            result1.tokens, result2.tokens,
            "results should be deterministic"
        );
        assert_eq!(result1.n_anchors, result2.n_anchors);
        assert_eq!(result1.fill_steps_used, result2.fill_steps_used);
    }

    #[test]
    fn test_anchor_density_vs_stride() {
        // Verify anchor count = block_size / stride for various strides
        let config = make_config();
        let block_size = 8;

        for stride in [1, 2, 4, 8] {
            let mut rng = Rng::new(42);
            let weights = TransformerWeights::new(&config, &mut rng);
            let mask = config.mask_token;

            let mut ctx = ForwardContext::new(&config);
            let mut cache = MultiLayerKVCache::new(&config);
            let mut token_buf = vec![mask; block_size];

            let n_anchors = predict_anchors(
                &mut ctx,
                &mut cache,
                &weights,
                &config,
                0,
                0,
                block_size,
                stride,
                mask,
                &mut token_buf,
                &mut rng,
                None,
                None,
            );

            let expected = block_size / stride;
            assert_eq!(
                n_anchors, expected,
                "stride={stride}: expected {expected} anchors, got {n_anchors}"
            );
        }
    }
}
