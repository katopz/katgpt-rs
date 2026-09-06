//! D2F Drafter Verifier — D2F diffusion drafts, AR verifies.
//!
//! Plan 089: Tri-Mode Inference — "self-speculation" mode.
//! Uses existing D2F block decode as drafter + existing AR as verifier.
//! Behind `tri_mode` feature gate.
//!
//! Issue 587 (FLARE, arXiv:2606.01774 §3.3): the acceptance policy was
//! rewritten from greedy argmax prefix-match (distribution-biasing — every
//! rejection emits the greedy token, collapsing output toward the target
//! mode) to the FLARE Eq 8/21/22 taxonomy behind [`DraftAcceptPolicy`].
//! Two further fixes landed with it:
//!
//! - **Slot-alignment fix**: Phase 3 historically compared draft token `i`
//!   against the target distribution for position `i+1` (off-by-one). The
//!   fused verify loop now scores/accepts on aligned slots.
//! - **Streaming verify (T5)**: the `[(K+1) × vocab]` p-distribution
//!   materialization is gone. One vocab-sized buffer streams position by
//!   position; scoring stops at the first rejection (early-exit saves the
//!   remaining target forward passes).
//!
//! Plan 399 (2026-07-05): moved from root `src/speculative/d2f_verifier.rs`.
//! Root's `src/speculative/d2f_verifier.rs` is a thin re-export shim.

use crate::d2f::{
    D2fDecodeConfig, d2f_decode_block_with_prompt_with, d2f_decode_block_with_prompt_with_q,
};
use crate::d2f_context::D2fContext;
use crate::{ForwardContext, forward};
use katgpt_core::speculative::sampling::{
    sample_from_distribution, sample_residual_distribution_into,
};
use katgpt_core::traits::{NoPruner, NoScreeningPruner};
use katgpt_speculative::SpeculativeVerifier;
use katgpt_transformer::MultiLayerKVCache;
use katgpt_transformer::TransformerWeights;
use katgpt_types::{Config, Rng, softmax_scaled};

/// Top-k width for [`DraftAcceptPolicy::TruncatedArgmax`] (FLARE Eq 22).
const TRUNC_TOPK: usize = 16;

/// Acceptance policy for D2F self-speculation verification
/// (Issue 587 / FLARE Eq 8/21/22 taxonomy).
///
/// Per-token rejection sampling is exact when the acceptance test uses the
/// **law that actually generated the draft token** (the Leviathan
/// proposal-consistency condition). The policies differ in how they obtain
/// that law and what they trade for it:
///
/// | Policy | Draft law q | Accept rule | Correction | Exact? |
/// |---|---|---|---|---|
/// | `PrefixMatch` | unused | `d == argmax(p)` | `argmax(p)` | ✗ (mode collapse) |
/// | `SoftmaxArgmax` | point mass (forced greedy drafting) | `u ≤ p(d)` | `p∖{d}` renormalized | ✓ (Eq 21) |
/// | `TruncatedArgmax` | point mass | `u ≤ p̃(d)` (top-k) | top-k residual | ≈ (tail dropped) |
/// | `ExactQ` | stored full q per position | `u ≤ min(1, p(d)/q(d))` | `norm(max(p−q, 0))` | ✓ (Eq 8 analog) |
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DraftAcceptPolicy {
    /// Legacy Plan 089 behavior: greedy argmax prefix-match. Kept as the
    /// GOAT control/fallback — distribution-biasing by construction.
    PrefixMatch,
    /// FLARE Eq 21 (Softmax-Argmax): accept `d ⟺ u ≤ p(d)`, correction
    /// `y* ~ p∖{d}`. Exact w.r.t. the target distribution. Requires argmax
    /// drafting — the verifier **forces** `D2fDecodeConfig::greedy_draft`
    /// under this policy (a point-mass proposal law is what makes the
    /// `u ≤ p(d)` test the correct one).
    #[default]
    SoftmaxArgmax,
    /// FLARE Eq 22 (Truncated-Argmax): verify on the top-`TRUNC_TOPK`
    /// truncated target distribution only. Approximate — the tail mass
    /// outside the truncation is discarded, so the output samples from
    /// `p̃` rather than `p`.
    TruncatedArgmax,
    /// FLARE Eq 8 analog (exact, full-width): the drafter captures the
    /// draft-time proposal law `q` per position (`d2f_decode_block_with_prompt_with_q`);
    /// verification runs true `min(1, p/q)` rejection with residual
    /// correction. Exact under *sampled* drafts — the only policy that
    /// preserves temperature-diverse drafting. We store the full `q` row
    /// rather than FLARE's top-k (a serving-memory optimization; at
    /// library scale the `[K × V]` buffer is affordable and exact).
    ExactQ,
}

/// Outcome of one acceptance-policy step: accept the draft token, or emit a
/// correction token sampled per-policy (and stop the verify loop).
///
/// `pub(crate)` (Issue 651): the step type crosses into `flashar_consensus`
/// for its Warm/Cold verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PolicyStep {
    Accept,
    Correct(usize),
}

/// Speculative verifier that uses D2F diffusion as drafter, AR as verifier.
///
/// This is the Nemotron "self-speculation" mode — same draft→verify→accept
/// pattern as LeviathanVerifier, but D2F drafts in parallel instead of
/// DFlash drafting sequentially.
///
/// Key difference from LeviathanVerifier:
/// - Draft: `d2f_decode_block()` (parallel, bidirectional within block)
/// - Verify: `forward()` with causal attention (same as Leviathan)
/// - KV caches are separate (block-causal for draft, causal for verify)
pub struct D2fDrafterVerifier<'a> {
    pub target_weights: &'a TransformerWeights,
    pub target_config: &'a Config,
    pub d2f_config: D2fDecodeConfig,
    pub draft_width: usize,
    /// Acceptance policy (Issue 587). Default [`DraftAcceptPolicy::SoftmaxArgmax`].
    pub accept_policy: DraftAcceptPolicy,
    target_ctx: ForwardContext,
    target_cache: MultiLayerKVCache,
    d2f_ctx: D2fContext,
    // Streaming per-position target distribution buffer `[vocab]` (T5).
    probs_buf: Vec<f32>,
    // Draft-time proposal laws `[block_capacity × vocab]` (ExactQ only).
    q_distributions_flat: Vec<f32>,
    // Residual-correction scratch `[vocab]` (ExactQ only).
    residual_buf: Vec<f32>,
    // Pre-allocated accepted tokens buffer: `[draft_width + 1]`.
    // Cleared + reused across speculate() calls.
    accepted_buf: Vec<usize>,
}

impl<'a> D2fDrafterVerifier<'a> {
    /// Create a new D2F drafter verifier.
    ///
    /// `draft_width` must match `d2f_config.block_size` — the number of tokens
    /// the D2F drafter produces in parallel per block decode.
    pub fn new(
        target_weights: &'a TransformerWeights,
        target_config: &'a Config,
        d2f_config: D2fDecodeConfig,
        draft_width: usize,
    ) -> Self {
        Self::with_accept_policy(
            target_weights,
            target_config,
            d2f_config,
            draft_width,
            DraftAcceptPolicy::default(),
        )
    }

    /// Create a verifier with an explicit acceptance policy (Issue 587).
    ///
    /// `SoftmaxArgmax` forces argmax drafting on the internal `d2f_config`
    /// copy (see [`DraftAcceptPolicy::SoftmaxArgmax`]).
    pub fn with_accept_policy(
        target_weights: &'a TransformerWeights,
        target_config: &'a Config,
        d2f_config: D2fDecodeConfig,
        draft_width: usize,
        accept_policy: DraftAcceptPolicy,
    ) -> Self {
        // Ensure block_size is at least draft_width
        let mut config = D2fDecodeConfig {
            block_size: d2f_config.block_size.max(draft_width),
            ..d2f_config
        };
        // SoftmaxArgmax exactness precondition: the accept test `u ≤ p(d)`
        // is only correct for a point-mass draft law, i.e. argmax drafting.
        if accept_policy == DraftAcceptPolicy::SoftmaxArgmax {
            config.greedy_draft = true;
        }
        let vocab_size = target_config.vocab_size;
        let block_capacity = config.block_size.max(draft_width);
        Self {
            target_weights,
            target_config,
            d2f_config: config,
            draft_width,
            accept_policy,
            target_ctx: ForwardContext::new(target_config),
            target_cache: MultiLayerKVCache::new(target_config),
            d2f_ctx: D2fContext::new(target_config),
            probs_buf: vec![0.0f32; vocab_size],
            q_distributions_flat: vec![0.0f32; block_capacity * vocab_size],
            residual_buf: vec![0.0f32; vocab_size],
            accepted_buf: Vec::with_capacity(draft_width + 1),
        }
    }
}

impl SpeculativeVerifier for D2fDrafterVerifier<'_> {
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
        let vocab_size = self.target_config.vocab_size;
        let inv_target_temp = 1.0 / target_temp;

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

        // ── Phase 1: D2F block decode — parallel draft ──────────────
        // Use the prompt (initial token) as context for D2F block decode.
        // ExactQ additionally captures the draft-time proposal laws.
        let prompt = &[token];
        let d2f_result = if self.accept_policy == DraftAcceptPolicy::ExactQ {
            d2f_decode_block_with_prompt_with_q(
                &mut self.d2f_ctx,
                draft_weights,
                draft_config,
                &self.d2f_config,
                prompt,
                &NoPruner,
                &NoScreeningPruner,
                rng,
                &mut self.q_distributions_flat,
            )
        } else {
            d2f_decode_block_with_prompt_with(
                &mut self.d2f_ctx,
                draft_weights,
                draft_config,
                &self.d2f_config,
                prompt,
                &NoPruner,
                &NoScreeningPruner,
                rng,
            )
        };

        let draft_tokens = &d2f_result.tokens;
        let k = draft_tokens.len().min(self.draft_width);

        if k == 0 {
            // Fallback: no draft tokens produced, sample from target distribution
            return vec![sample_from_distribution(&self.probs_buf, rng)];
        }

        // Copy draft tokens to stack to avoid borrow conflict with &mut self
        let mut token_stack = [0usize; 64];
        let k_bounded = k.min(token_stack.len());
        token_stack[..k_bounded].copy_from_slice(&draft_tokens[..k_bounded]);

        // ── Phase 2+3: fused streaming verify (Issue 587 T5) ────────
        // probs_buf holds p_i when testing d_i: the loop feeds d_i to the
        // target only AFTER it is accepted (and skips the feed on the last
        // position — the bonus feed below covers it). On rejection the
        // remaining target forwards are skipped entirely (early exit).
        //
        // Slot alignment note (Issue 587): the pre-rewrite Phase 2/3 scored
        // every draft upfront and compared d_i against the target
        // distribution for position i+1 — an off-by-one. The fused loop
        // tests d_i against p_i (aligned), matching LeviathanVerifier.
        self.accepted_buf.clear();
        let mut all_accepted = true;

        for i in 0..k_bounded {
            let draft_tok = token_stack[i];
            let p_dist = &self.probs_buf[..vocab_size];

            let step = match self.accept_policy {
                DraftAcceptPolicy::PrefixMatch => prefix_match_step(p_dist, draft_tok),
                DraftAcceptPolicy::SoftmaxArgmax => {
                    softmax_argmax_step(p_dist, draft_tok, rng)
                }
                DraftAcceptPolicy::TruncatedArgmax => {
                    truncated_argmax_step(p_dist, draft_tok, rng)
                }
                DraftAcceptPolicy::ExactQ => {
                    let q_start = i * vocab_size;
                    let q_end = q_start + vocab_size;
                    let q_dist = &self.q_distributions_flat[q_start.min(self.q_distributions_flat.len())..q_end.min(self.q_distributions_flat.len())];
                    exact_q_step(p_dist, q_dist, draft_tok, rng, &mut self.residual_buf)
                }
            };

            match step {
                PolicyStep::Accept => {
                    self.accepted_buf.push(draft_tok);
                }
                PolicyStep::Correct(y) => {
                    self.accepted_buf.push(y);
                    all_accepted = false;
                    break;
                }
            }

            // Advance: feed the accepted draft token → p_{i+1}.
            if i + 1 < k_bounded {
                let logits = forward(
                    &mut self.target_ctx,
                    self.target_weights,
                    &mut self.target_cache,
                    draft_tok,
                    pos + 1 + i,
                    self.target_config,
                );
                self.probs_buf.copy_from_slice(logits);
                softmax_scaled(&mut self.probs_buf, inv_target_temp);
            }
        }

        // ── Phase 4: Bonus token if all accepted ────────────────────
        if all_accepted {
            let last = token_stack[k_bounded - 1];
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
            self.accepted_buf
                .push(sample_from_distribution(&self.probs_buf, rng));
        }

        // Every path pushes ≥ 1 token (accept or correction at position 0),
        // so the legacy empty-buf fallback is unreachable; mem::take moves
        // the buffer out for the Vec-return ABI (one small alloc per call,
        // bounded by draft_width + 1 — pre-existing, unchanged).
        debug_assert!(!self.accepted_buf.is_empty());
        std::mem::take(&mut self.accepted_buf)
    }
}

// ── Acceptance-policy steps (Issue 587; pure, unit-testable) ─────────

/// Argmax with branch-free, NaN-deterministic comparison.
fn argmax_total_cmp(p: &[f32]) -> usize {
    p.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b)).map_or(0, |(idx, _)| idx)
}

/// Argmax excluding one index (correction fallbacks).
fn argmax_excluding(p: &[f32], exclude: usize) -> usize {
    let mut best = usize::MAX;
    let mut best_v = f32::NEG_INFINITY;
    for (t, &v) in p.iter().enumerate() {
        if t == exclude {
            continue;
        }
        if v > best_v || best == usize::MAX {
            best_v = v;
            best = t;
        }
    }
    if best == usize::MAX {
        exclude
    } else {
        best
    }
}

/// Sample `y ~ p` renormalized over `t ≠ exclude` (FLARE Eq 21 correction:
/// the residual `norm(max(p − δ_d, 0))` of a point-mass draft law).
fn sample_excluding(p: &[f32], exclude: usize, rng: &mut Rng) -> usize {
    let p_excl = p.get(exclude).copied().unwrap_or(0.0).clamp(0.0, 1.0);
    let rest = 1.0 - p_excl;
    if rest <= f32::EPSILON {
        // p(exclude) ≈ 1: complement carries no mass — argmax fallback.
        return argmax_excluding(p, exclude);
    }
    let r = rng.uniform() * rest;
    let mut cdf = 0.0f32;
    for (t, &pt) in p.iter().enumerate() {
        if t == exclude || pt <= 0.0 {
            continue;
        }
        cdf += pt;
        if r < cdf {
            return t;
        }
    }
    // f32 drift fallback.
    argmax_excluding(p, exclude)
}

/// Legacy Plan 089 prefix-match step (GOAT control): accept iff the draft
/// token IS the target argmax; correct with the argmax. Distribution-biasing
/// by construction (Issue 587 problem statement).
///
/// `pub(crate)` (Issue 651): reused by `flashar_consensus`'s Warm/Cold
/// verification under [`DraftAcceptPolicy::PrefixMatch`].
pub(crate) fn prefix_match_step(p: &[f32], d: usize) -> PolicyStep {
    let am = argmax_total_cmp(p);
    if d == am {
        PolicyStep::Accept
    } else {
        PolicyStep::Correct(am)
    }
}

/// FLARE Eq 21 (Softmax-Argmax) step: accept `d ⟺ u ≤ p(d)`; correction
/// `y* ~ p∖{d}`. Exact for point-mass draft laws (greedy drafting, or any
/// deterministic proposal — Issue 651 uses it for FlashAR's consensus
/// winner, which is a deterministic pick).
///
/// `pub(crate)` (Issue 651): reused by `flashar_consensus`'s Warm/Cold
/// verification.
pub(crate) fn softmax_argmax_step(p: &[f32], d: usize, rng: &mut Rng) -> PolicyStep {
    let pd = p.get(d).copied().unwrap_or(0.0);
    let u = rng.uniform();
    if u <= pd {
        PolicyStep::Accept
    } else {
        PolicyStep::Correct(sample_excluding(p, d, rng))
    }
}

/// Top-k scan (descending insertion) into fixed stack buffers.
/// Returns `(k_used, z_k)` where `z_k` is the truncated mass.
fn topk_into(p: &[f32], ids: &mut [usize; TRUNC_TOPK], probs: &mut [f32; TRUNC_TOPK]) -> (usize, f32) {
    let k = TRUNC_TOPK.min(p.len());
    if k == 0 {
        return (0, 0.0);
    }
    let mut used = 0usize;
    let mut z = 0.0f32;
    for (t, &pt) in p.iter().enumerate() {
        if used < k {
            let mut j = used;
            while j > 0 && probs[j - 1] < pt {
                probs[j] = probs[j - 1];
                ids[j] = ids[j - 1];
                j -= 1;
            }
            probs[j] = pt;
            ids[j] = t;
            used += 1;
            z += pt;
        } else if pt > probs[k - 1] {
            z -= probs[k - 1];
            let mut j = k - 1;
            while j > 0 && probs[j - 1] < pt {
                probs[j] = probs[j - 1];
                ids[j] = ids[j - 1];
                j -= 1;
            }
            probs[j] = pt;
            ids[j] = t;
            z += pt;
        }
    }
    (used, z)
}

/// FLARE Eq 22 (Truncated-Argmax) step: verify on the top-k truncated
/// target distribution only. Approximate — tail mass is discarded.
///
/// `pub(crate)` (Issue 651): reused by `flashar_consensus`'s Warm/Cold
/// verification.
pub(crate) fn truncated_argmax_step(p: &[f32], d: usize, rng: &mut Rng) -> PolicyStep {
    let mut ids = [0usize; TRUNC_TOPK];
    let mut probs = [0.0f32; TRUNC_TOPK];
    let (used, z_k) = topk_into(p, &mut ids, &mut probs);
    if used == 0 || z_k <= 0.0 {
        return PolicyStep::Correct(sample_excluding(p, d, rng));
    }

    let hit = (0..used).find(|&j| ids[j] == d);
    let p_tilde = match hit {
        Some(j) => probs[j] / z_k,
        None => 0.0, // draft token outside the truncated support → reject
    };
    let u = rng.uniform();
    if u <= p_tilde {
        return PolicyStep::Accept;
    }

    // Correction: renormalized truncated p over top-k ∖ {d}.
    let z_ex: f32 = (0..used).filter(|&j| ids[j] != d).map(|j| probs[j]).sum();
    if z_ex <= 0.0 {
        // Top-k is only the excluded token — full-distribution fallback.
        return PolicyStep::Correct(sample_excluding(p, d, rng));
    }
    let r = rng.uniform() * z_ex;
    let mut cdf = 0.0f32;
    let mut pick = ids[0];
    for j in 0..used {
        if ids[j] == d {
            continue;
        }
        cdf += probs[j];
        pick = ids[j];
        if r < cdf {
            break;
        }
    }
    PolicyStep::Correct(pick)
}

/// FLARE Eq 8 analog (exact, full-width q) step: accept
/// `d ⟺ u ≤ min(1, p(d)/q(d))`; correction `y* ~ norm(max(p − q, 0))`.
fn exact_q_step(p: &[f32], q: &[f32], d: usize, rng: &mut Rng, residual: &mut [f32]) -> PolicyStep {
    let pd = p.get(d).copied().unwrap_or(0.0);
    let qd = q.get(d).copied().unwrap_or(0.0);
    let acceptance = if qd > 0.0 { (pd / qd).min(1.0) } else { 1.0 };
    let u = rng.uniform();
    if u <= acceptance {
        PolicyStep::Accept
    } else {
        PolicyStep::Correct(sample_residual_distribution_into(
            p, q, residual, rng,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> Config {
        let mut c = Config::micro();
        c.vocab_size = 64;
        c
    }

    // ── Legacy shape tests (must keep passing) ──────────────────────

    #[test]
    fn test_d2f_verifier_returns_at_least_one() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

        let d2f_config = D2fDecodeConfig::with_block_size(4);
        let mut verifier = D2fDrafterVerifier::new(&target_weights, &config, d2f_config, 4);

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
    fn test_d2f_verifier_deterministic() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

        let d2f_config = D2fDecodeConfig::with_block_size(4);

        let r1 = {
            let mut verifier = D2fDrafterVerifier::new(&target_weights, &config, d2f_config, 4);
            verifier.speculate(
                &draft_weights,
                &config,
                config.bos_token,
                0,
                &mut Rng::new(100),
            )
        };

        let r2 = {
            let mut verifier = D2fDrafterVerifier::new(&target_weights, &config, d2f_config, 4);
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
    fn test_d2f_verifier_max_tokens_bounded() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

        let draft_width = 4;
        let d2f_config = D2fDecodeConfig::with_block_size(draft_width);
        let mut verifier =
            D2fDrafterVerifier::new(&target_weights, &config, d2f_config, draft_width);

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

    // ── Issue 587 G1: policy-level distribution exactness ───────────
    //
    // Fixed toy target distribution p over 32 tokens; drafter peaked
    // elsewhere (argmax q ≠ argmax p). The empirical distribution of the
    // policy step output over N rounds must match p for exact policies
    // (Leviathan identity) and must NOT match for PrefixMatch (the proof
    // of gain — its output is a constant).

    const TOY_V: usize = 32;
    const TOY_N: usize = 40_000;

    fn toy_target_p() -> Vec<f32> {
        // Non-degenerate, non-uniform, max < 0.25 (PrefixMatch TV > 0.7).
        let raw: Vec<f32> = (0..TOY_V)
            .map(|t| 1.0 + ((t * 7 + 3) % 13) as f32 / 13.0)
            .collect();
        let s: f32 = raw.iter().sum();
        raw.iter().map(|&x| x / s).collect()
    }

    fn toy_draft_token(p: &[f32]) -> usize {
        // Drafter peaks at a token where the target is NOT peaked (its own
        // argmax points elsewhere — a point-mass proposal at this token).
        p.iter()
            .enumerate()
            // float_order: a NaN prob must never win the min (total_cmp ranks
            // NaN below -inf → it would be selected as the least-peaked token).
            .min_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_min(**a, **b))
            .map_or(0, |(i, _)| i)
    }

    fn tv_distance(counts: &[usize], p: &[f32], n: usize) -> f64 {
        let mut tv = 0.0f64;
        for (t, &pt) in p.iter().enumerate() {
            let emp = counts.get(t).copied().unwrap_or(0) as f64 / n as f64;
            tv += (emp - pt as f64).abs();
        }
        0.5 * tv
    }

    #[test]
    fn test_policy_softmax_argmax_exact_distribution() {
        let p = toy_target_p();
        let d = toy_draft_token(&p);
        let mut rng = Rng::new(5871);
        let mut counts = vec![0usize; TOY_V];
        for _ in 0..TOY_N {
            let y = match softmax_argmax_step(&p, d, &mut rng) {
                PolicyStep::Accept => d,
                PolicyStep::Correct(y) => y,
            };
            counts[y] += 1;
        }
        let tv = tv_distance(&counts, &p, TOY_N);
        assert!(
            tv < 0.02,
            "SoftmaxArgmax must preserve the target distribution (TV = {tv:.4})"
        );
    }

    #[test]
    fn test_policy_prefix_match_fails_exactness() {
        // The GOAT gain proof: the legacy policy collapses to a point mass.
        let p = toy_target_p();
        let d = toy_draft_token(&p);
        let mut counts = vec![0usize; TOY_V];
        // PrefixMatch draws no RNG — its output is a constant (that IS the
        // mode-collapse failure this test pins).
        for _ in 0..TOY_N {
            let y = match prefix_match_step(&p, d) {
                PolicyStep::Accept => d,
                PolicyStep::Correct(y) => y,
            };
            counts[y] += 1;
        }
        let tv = tv_distance(&counts, &p, TOY_N);
        // Output is the constant argmax(p) (d ≠ argmax p) → TV = 1 − p_max.
        let p_max = p.iter().cloned().fold(0.0f32, f32::max) as f64;
        assert!(
            tv > 0.7,
            "PrefixMatch must FAIL exactness (TV = {tv:.4}, p_max = {p_max:.3})"
        );
    }

    #[test]
    fn test_policy_exact_q_exact_distribution() {
        // Sampled drafts from a misaligned q + stored q → still exact.
        let p = toy_target_p();
        // Drafter law: peaked away from the target mode.
        let mut q = vec![0.0f32; TOY_V];
        for (t, qi) in q.iter_mut().enumerate() {
            *qi = 0.2 + ((t * 11 + 5) % 17) as f32 / 17.0;
        }
        let qs: f32 = q.iter().sum();
        for qi in q.iter_mut() {
            *qi /= qs;
        }
        let mut rng = Rng::new(5873);
        let mut residual = vec![0.0f32; TOY_V];
        let mut counts = vec![0usize; TOY_V];
        for _ in 0..TOY_N {
            let d = sample_from_distribution(&q, &mut rng);
            let y = match exact_q_step(&p, &q, d, &mut rng, &mut residual) {
                PolicyStep::Accept => d,
                PolicyStep::Correct(y) => y,
            };
            counts[y] += 1;
        }
        let tv = tv_distance(&counts, &p, TOY_N);
        assert!(
            tv < 0.02,
            "ExactQ must preserve the target distribution (TV = {tv:.4})"
        );
    }

    #[test]
    fn test_policy_truncated_argmax_bounded_approximation() {
        // Approximate: output ≈ truncated-renormalized target. TV from p is
        // bounded by the discarded tail mass (+ sampling noise).
        let p = toy_target_p();
        let d = toy_draft_token(&p);
        let mut rng = Rng::new(5874);
        let mut counts = vec![0usize; TOY_V];
        for _ in 0..TOY_N {
            let y = match truncated_argmax_step(&p, d, &mut rng) {
                PolicyStep::Accept => d,
                PolicyStep::Correct(y) => y,
            };
            counts[y] += 1;
        }
        let tv = tv_distance(&counts, &p, TOY_N);
        // Bound: tail mass outside the top-16 support + sampling slack.
        let mut sorted = p.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let z16: f32 = sorted.iter().take(TRUNC_TOPK).sum();
        let bound = (1.0 - z16 as f64) + 0.05;
        assert!(
            tv <= bound,
            "TruncatedArgmax TV ({tv:.4}) must stay within the truncation bound ({bound:.4})"
        );
    }

    // ── Issue 587: config plumbing ───────────────────────────────────

    #[test]
    fn test_softmax_argmax_forces_greedy_draft() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let d2f_config = D2fDecodeConfig::with_block_size(4);

        let v = D2fDrafterVerifier::with_accept_policy(
            &target_weights,
            &config,
            d2f_config,
            4,
            DraftAcceptPolicy::SoftmaxArgmax,
        );
        assert!(v.d2f_config.greedy_draft, "SoftmaxArgmax must force greedy drafting");

        let v2 = D2fDrafterVerifier::with_accept_policy(
            &target_weights,
            &config,
            d2f_config,
            4,
            DraftAcceptPolicy::PrefixMatch,
        );
        assert!(
            !v2.d2f_config.greedy_draft,
            "PrefixMatch keeps the legacy sampled drafting"
        );

        let v3 = D2fDrafterVerifier::with_accept_policy(
            &target_weights,
            &config,
            d2f_config,
            4,
            DraftAcceptPolicy::ExactQ,
        );
        assert!(
            !v3.d2f_config.greedy_draft,
            "ExactQ works with sampled drafting (that's its point)"
        );
    }

    #[test]
    fn test_all_policies_shape_bounded_and_deterministic() {
        let config = make_config();
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&config, &mut rng);
        let mut draft_rng = Rng::new(99);
        let draft_weights = TransformerWeights::new(&config, &mut draft_rng);

        for policy in [
            DraftAcceptPolicy::PrefixMatch,
            DraftAcceptPolicy::SoftmaxArgmax,
            DraftAcceptPolicy::TruncatedArgmax,
            DraftAcceptPolicy::ExactQ,
        ] {
            let d2f_config = D2fDecodeConfig::with_block_size(4);
            let r1 = {
                let mut v = D2fDrafterVerifier::with_accept_policy(
                    &target_weights,
                    &config,
                    d2f_config,
                    4,
                    policy,
                );
                v.speculate(&draft_weights, &config, config.bos_token, 0, &mut Rng::new(7))
            };
            let r2 = {
                let mut v = D2fDrafterVerifier::with_accept_policy(
                    &target_weights,
                    &config,
                    d2f_config,
                    4,
                    policy,
                );
                v.speculate(&draft_weights, &config, config.bos_token, 0, &mut Rng::new(7))
            };
            assert_eq!(r1, r2, "{policy:?}: same seed must reproduce");
            assert!(!r1.is_empty(), "{policy:?}: must return ≥1 token");
            assert!(
                r1.len() <= 5,
                "{policy:?}: accepted {} > draft_width+1",
                r1.len()
            );
            assert!(
                r1.iter().all(|&t| t < config.vocab_size),
                "{policy:?}: tokens out of vocab range"
            );
        }
    }

    // ── Issue 587 T4: q-capture round-trip ───────────────────────────

    #[test]
    fn test_q_capture_rows_are_distributions() {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let d2f_config = D2fDecodeConfig {
            denoise_steps: 4,
            confidence_threshold: 0.3,
            block_size: 4,
            temperature: 1.0,
            ..D2fDecodeConfig::default()
        };
        let mut dctx = D2fContext::new(&config);
        let mut q = vec![0.0f32; 4 * config.vocab_size];

        d2f_decode_block_with_prompt_with_q(
            &mut dctx,
            &weights,
            &config,
            &d2f_config,
            &[config.bos_token],
            &NoPruner,
            &NoScreeningPruner,
            &mut rng,
            &mut q,
        );

        for row in 0..4 {
            let s: f32 = q[row * config.vocab_size..(row + 1) * config.vocab_size]
                .iter()
                .sum();
            assert!(
                (s - 1.0).abs() < 1e-3 || s == 0.0,
                "q row {row} must be a distribution (sum {s:.5})"
            );
        }
    }

    #[test]
    fn test_greedy_draft_deterministic_block() {
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let d2f_config = D2fDecodeConfig {
            denoise_steps: 4,
            confidence_threshold: 0.3,
            block_size: 4,
            greedy_draft: true,
            ..D2fDecodeConfig::default()
        };

        let t1 = {
            let mut dctx = D2fContext::new(&config);
            d2f_decode_block_with_prompt_with(
                &mut dctx,
                &weights,
                &config,
                &d2f_config,
                &[config.bos_token],
                &NoPruner,
                &NoScreeningPruner,
                &mut Rng::new(1),
            )
            .tokens
        };
        // Argmax drafting consumes no RNG for token choice → different RNG
        // seeds must produce the identical block.
        let t2 = {
            let mut dctx = D2fContext::new(&config);
            d2f_decode_block_with_prompt_with(
                &mut dctx,
                &weights,
                &config,
                &d2f_config,
                &[config.bos_token],
                &NoPruner,
                &NoScreeningPruner,
                &mut Rng::new(999),
            )
            .tokens
        };
        assert_eq!(t1, t2, "greedy drafting must be RNG-independent");
    }
}
