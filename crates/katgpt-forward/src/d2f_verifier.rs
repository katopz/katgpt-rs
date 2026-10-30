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
/// | `NucleusAccept` | unused | `d == x̃ ∨ d ∈ ρ-nucleus(p) ∨ p(d) ≥ ε` | `x̃ ~ p` (no residual) | ✓ only at ρ=0, ε=1 (Issue 973) |
///
/// `Eq` is not derived: [`DraftAcceptPolicy::NucleusAccept`] carries `f32`
/// knobs (the enum stays `Copy + PartialEq`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
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
    /// PLM relaxed acceptance (arXiv:2610.26693 Alg. 1, Issue 973): a
    /// deterministic likelihood accept-gate with NO residual correction.
    ///
    /// Per position, draw a verifier-side sample `x̃ ~ p` (one `U(0,1)` +
    /// inverse-CDF walk) and accept the draft `d` iff
    /// `d == x̃ ∨ d ∈ ρ-nucleus(p) ∨ p(d) ≥ ε`. **On rejection the emitted
    /// token is `x̃`** — the verifier draw, never a residual sample.
    ///
    /// - **Corner exactness:** at `rho = 0, eps = 1` every emitted token is
    ///   `x̃` (accepted only when `d == x̃`), so the output is exactly
    ///   p-distributed (PLM Corollary 1). Away from the corner the rule is
    ///   deliberately lossy — the ρ/ε constants are posture-fitted by the
    ///   consumer's quality gate (riir-ai Issue 1142), never here.
    /// - **Greedy-verify posture (`greedy = true`):** `x̃ = argmax p`
    ///   (the same `argmax_total_cmp` tie rule as `PrefixMatch`), so the
    ///   corner reduces bit-identically to `PrefixMatch` — the argmax bias
    ///   then belongs to the posture, not the policy.
    /// - **Seed independence:** the `x̃` draw comes from a SEPARATE stream
    ///   (domain-tagged off the caller's rng, [`NUCLEUS_VERIFY_DOMAIN`]) —
    ///   reusing the drafter's `u` makes `d == x̃` hold by construction and
    ///   accepts every sampled draft at the corner.
    /// - **ρ-nucleus:** the smallest top-ranked set with mass `≥ ρ`; ranking
    ///   is descending p with a pinned first-index tie rule, membership is
    ///   `mass ranked strictly above d < ρ` (one sequential fold, zero-alloc,
    ///   Issue-940 law). `rho = 0` ⇒ empty nucleus.
    #[cfg(feature = "nucleus_accept")]
    NucleusAccept {
        /// Nucleus mass `ρ ∈ [0, 1]` (0 disables the term).
        rho: f32,
        /// Likelihood floor `ε ∈ (0, 1]` (1 disables the term for any
        /// non-point-mass law).
        eps: f32,
        /// Greedy-verify posture: fallback `x̃ = argmax p`.
        greedy: bool,
    },
}

/// Domain tag for the [`DraftAcceptPolicy::NucleusAccept`] verifier-draw
/// stream (Issue 973 seed-independence law): the verify rng is
/// `Rng::new(caller.next() ^ NUCLEUS_VERIFY_DOMAIN)`, SplitMix-decorrelated
/// from every drafter stream.
#[cfg(feature = "nucleus_accept")]
pub const NUCLEUS_VERIFY_DOMAIN: u64 = 0x6E75_636C_5F76_6572; // "nucl_ver"

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
        // Issue 973: NucleusAccept's x̃ draws ride their own stream.
        #[cfg(feature = "nucleus_accept")]
        let mut verify_rng = Rng::new(rng.next() ^ NUCLEUS_VERIFY_DOMAIN);

        for i in 0..k_bounded {
            let draft_tok = token_stack[i];
            let p_dist = &self.probs_buf[..vocab_size];

            let step = match self.accept_policy {
                DraftAcceptPolicy::PrefixMatch => prefix_match_step(p_dist, draft_tok),
                DraftAcceptPolicy::SoftmaxArgmax => softmax_argmax_step(p_dist, draft_tok, rng),
                DraftAcceptPolicy::TruncatedArgmax => truncated_argmax_step(p_dist, draft_tok, rng),
                DraftAcceptPolicy::ExactQ => {
                    let q_start = i * vocab_size;
                    let q_end = q_start + vocab_size;
                    let q_dist = &self.q_distributions_flat[q_start
                        .min(self.q_distributions_flat.len())
                        ..q_end.min(self.q_distributions_flat.len())];
                    exact_q_step(p_dist, q_dist, draft_tok, rng, &mut self.residual_buf)
                }
                #[cfg(feature = "nucleus_accept")]
                DraftAcceptPolicy::NucleusAccept { rho, eps, greedy } => {
                    nucleus_accept_step(p_dist, draft_tok, rho, eps, greedy, &mut verify_rng)
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
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(idx, _)| idx)
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
    if best == usize::MAX { exclude } else { best }
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
fn topk_into(
    p: &[f32],
    ids: &mut [usize; TRUNC_TOPK],
    probs: &mut [f32; TRUNC_TOPK],
) -> (usize, f32) {
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
        PolicyStep::Correct(sample_residual_distribution_into(p, q, residual, rng))
    }
}

// ── Issue 973: PLM relaxed acceptance (NucleusAccept) ────────────────

// The ρ-nucleus / ε admit terms live in katgpt-core (shared with the
// riir-infer keyed spec lane, riir-ai Issue 1142).
#[cfg(feature = "nucleus_accept")]
use katgpt_core::relaxed_accept::relaxed_admit;
#[cfg(all(feature = "nucleus_accept", test))]
use katgpt_core::relaxed_accept::{in_rho_nucleus, mass_above};

/// Inverse-CDF pick for a given `u ∈ [0, 1)`: first `t` with `u < Σ_{j≤t} p_j`
/// (ascending fold). f32-drift fallback (u past the total): the last token
/// carrying mass.
#[cfg(feature = "nucleus_accept")]
#[inline]
fn inverse_cdf_pick(p: &[f32], u: f32) -> usize {
    let mut cdf = 0.0f32;
    let mut last_pos = 0usize;
    for (t, &pt) in p.iter().enumerate() {
        cdf += pt;
        if pt > 0.0 {
            last_pos = t;
        }
        if u < cdf {
            return t;
        }
    }
    last_pos
}

/// PLM relaxed-acceptance step (Issue 973, [`DraftAcceptPolicy::NucleusAccept`]):
/// accept `d ⟺ d == x̃ ∨ d ∈ ρ-nucleus(p) ∨ p(d) ≥ ε`; on rejection emit `x̃`
/// with no residual correction. `verify_rng` MUST be a stream independent of
/// the drafter's (see [`NUCLEUS_VERIFY_DOMAIN`]).
///
/// Sampled posture: exactly one uniform is drawn per call, FIRST (accept or
/// not), so a fixed seed gives the same `x̃` across (ρ, ε) settings —
/// loosening is then monotone draw-by-draw. Cost ladder: the ε term reads
/// `p(d)` only; the ρ term (full-vocab fold) runs only when `rho > 0`; the
/// `d == x̃` cell test sums the prefix `p[..d]` in the SAME ascending order
/// as [`inverse_cdf_pick`] (bit-identical cell edges); the full CDF walk for
/// `x̃` runs only on the rejection path. Greedy posture: `x̃ = argmax p`
/// (`PrefixMatch`'s tie rule), no uniform drawn.
#[cfg(feature = "nucleus_accept")]
pub(crate) fn nucleus_accept_step(
    p: &[f32],
    d: usize,
    rho: f32,
    eps: f32,
    greedy: bool,
    verify_rng: &mut Rng,
) -> PolicyStep {
    let pd = p.get(d).copied().unwrap_or(0.0);
    let u = if greedy { 0.0 } else { verify_rng.uniform() };
    if relaxed_admit(p, d, rho, eps) {
        return PolicyStep::Accept;
    }
    if greedy {
        let am = argmax_total_cmp(p);
        return if d == am {
            PolicyStep::Accept
        } else {
            PolicyStep::Correct(am)
        };
    }
    if pd > 0.0 {
        let cdf_before: f32 = p[..d].iter().fold(0.0f32, |acc, &x| acc + x);
        if u >= cdf_before && u < cdf_before + pd {
            return PolicyStep::Accept;
        }
    }
    let x_tilde = inverse_cdf_pick(p, u);
    // Drift fallback can land on d itself — x̃ == d IS the match term.
    if x_tilde == d {
        PolicyStep::Accept
    } else {
        PolicyStep::Correct(x_tilde)
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
        assert!(
            v.d2f_config.greedy_draft,
            "SoftmaxArgmax must force greedy drafting"
        );

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

        #[allow(unused_mut)]
        let mut policies = vec![
            DraftAcceptPolicy::PrefixMatch,
            DraftAcceptPolicy::SoftmaxArgmax,
            DraftAcceptPolicy::TruncatedArgmax,
            DraftAcceptPolicy::ExactQ,
        ];
        // Issue 973: the relaxed policy rides the same speculate() contract.
        #[cfg(feature = "nucleus_accept")]
        for greedy in [false, true] {
            policies.push(DraftAcceptPolicy::NucleusAccept {
                rho: 0.0,
                eps: 1.0,
                greedy,
            });
            policies.push(DraftAcceptPolicy::NucleusAccept {
                rho: 0.94,
                eps: 2e-3,
                greedy,
            });
        }
        for policy in policies {
            let d2f_config = D2fDecodeConfig::with_block_size(4);
            let r1 = {
                let mut v = D2fDrafterVerifier::with_accept_policy(
                    &target_weights,
                    &config,
                    d2f_config,
                    4,
                    policy,
                );
                v.speculate(
                    &draft_weights,
                    &config,
                    config.bos_token,
                    0,
                    &mut Rng::new(7),
                )
            };
            let r2 = {
                let mut v = D2fDrafterVerifier::with_accept_policy(
                    &target_weights,
                    &config,
                    d2f_config,
                    4,
                    policy,
                );
                v.speculate(
                    &draft_weights,
                    &config,
                    config.bos_token,
                    0,
                    &mut Rng::new(7),
                )
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

    // ── Issue 973: NucleusAccept (PLM relaxed acceptance) ───────────

    #[cfg(feature = "nucleus_accept")]
    mod nucleus {
        use super::super::*;
        use super::{TOY_N, TOY_V, toy_target_p, tv_distance};

        /// Random normalized law over `v` tokens, a few planted ties.
        fn rand_law(v: usize, rng: &mut Rng) -> Vec<f32> {
            let mut raw: Vec<f32> = (0..v).map(|_| rng.uniform().powi(3) + 1e-4).collect();
            // Planted exact ties (the first-index tie rule must hold on them).
            raw[v - 1] = raw[0];
            raw[v / 2] = raw[1];
            let s: f32 = raw.iter().sum();
            raw.iter_mut().for_each(|x| *x /= s);
            raw
        }

        fn emit(step: PolicyStep, d: usize) -> usize {
            match step {
                PolicyStep::Accept => d,
                PolicyStep::Correct(y) => y,
            }
        }

        /// G1a: at ρ=0, ε=1 under SAMPLED verification the emitted token law
        /// equals p, with the drafter sampling from a DIFFERENT law on its
        /// own stream and the verifier draw on the domain-tagged stream.
        #[test]
        fn g1a_corner_is_exactly_p_distributed() {
            let p = toy_target_p();
            // Drafter law: peaked on the target's least-likely token.
            let mut q = [0.5 / (TOY_V - 1) as f32; TOY_V];
            q[super::toy_draft_token(&p)] = 0.5;
            let mut draft_rng = Rng::new(9731);
            let mut verify_rng = Rng::new(draft_rng.next() ^ NUCLEUS_VERIFY_DOMAIN);
            let mut counts = vec![0usize; TOY_V];
            let mut accepts = 0usize;
            for _ in 0..TOY_N {
                let d = sample_from_distribution(&q, &mut draft_rng);
                let step = nucleus_accept_step(&p, d, 0.0, 1.0, false, &mut verify_rng);
                accepts += usize::from(step == PolicyStep::Accept);
                counts[emit(step, d)] += 1;
            }
            let tv = tv_distance(&counts, &p, TOY_N);
            assert!(
                tv < 0.02,
                "corner TV {tv:.4} — output must be p-distributed"
            );
            // Acceptance probability = Σ_d q(d) p(d) (same as SoftmaxArgmax's
            // u ≤ p(d) — only per-draw decisions differ).
            let want: f32 = p.iter().zip(&q).map(|(a, b)| a * b).sum();
            let got = accepts as f32 / TOY_N as f32;
            assert!(
                (got - want).abs() < 0.01,
                "accept rate {got:.4} vs Σqp {want:.4}"
            );
        }

        /// G1a negative control (the seed-independence law): reusing the
        /// drafter's uniform as the verifier's makes `d == x̃` hold by
        /// construction — every draft accepted, output = the DRAFTER's law.
        #[test]
        fn g1a_shared_stream_breaks_corner() {
            let p = toy_target_p();
            let mut counts = vec![0usize; TOY_V];
            let mut accepts = 0usize;
            for i in 0..TOY_N as u64 {
                // Same seed on both sides = the shared-u defect: the drafter
                // samples p with u, the verifier re-reads the same u.
                let d = sample_from_distribution(&p, &mut Rng::new(i));
                let step = nucleus_accept_step(&p, d, 0.0, 1.0, false, &mut Rng::new(i));
                accepts += usize::from(step == PolicyStep::Accept);
                counts[emit(step, d)] += 1;
            }
            // With the shared u every draft matches (modulo the u == 0
            // redraw in sample_from_distribution) — acceptance ≈ 1, whereas
            // an independent stream accepts at Σp² ≪ 1.
            let sum_p2: f32 = p.iter().map(|x| x * x).sum();
            let got = accepts as f32 / TOY_N as f32;
            assert!(
                got > 0.99 && sum_p2 < 0.1,
                "shared-u accept {got:.4} (Σp² {sum_p2:.4})"
            );
        }

        /// G1b: greedy-verify corner is bit-identical to PrefixMatch on EVERY
        /// draw — accepts AND rejects, both arms exercised.
        #[test]
        fn g1b_greedy_corner_bit_identical_to_prefix_match() {
            let mut rng = Rng::new(9732);
            let mut vrng = Rng::new(1);
            let (mut acc, mut rej) = (0usize, 0usize);
            for i in 0..4000 {
                let v = 8 + i % 57;
                let p = rand_law(v, &mut rng);
                // Half the draws propose the argmax (accept arm), half not.
                let d = if i % 2 == 0 {
                    argmax_total_cmp(&p)
                } else {
                    (rng.next() as usize) % v
                };
                let got = nucleus_accept_step(&p, d, 0.0, 1.0, true, &mut vrng);
                let want = prefix_match_step(&p, d);
                assert_eq!(got, want, "draw {i}: v={v} d={d}");
                match want {
                    PolicyStep::Accept => acc += 1,
                    PolicyStep::Correct(_) => rej += 1,
                }
            }
            assert!(
                acc > 1000 && rej > 1000,
                "both arms must fire: acc={acc} rej={rej}"
            );
        }

        /// Accepted fraction over a FIXED (law, draft, verify-seed) stream.
        fn accept_frac(rho: f32, eps: f32, greedy: bool) -> f64 {
            let mut rng = Rng::new(9733);
            let mut vrng = Rng::new(rng.next() ^ NUCLEUS_VERIFY_DOMAIN);
            let mut acc = 0usize;
            const N: usize = 3000;
            for i in 0..N {
                let p = rand_law(16 + i % 33, &mut rng);
                let d = (rng.next() as usize) % p.len();
                acc += usize::from(
                    nucleus_accept_step(&p, d, rho, eps, greedy, &mut vrng) == PolicyStep::Accept,
                );
            }
            acc as f64 / N as f64
        }

        /// G1c: loosening (ρ↑ or ε↓) never decreases the accepted fraction.
        #[test]
        fn g1c_monotone_in_rho_and_eps() {
            const RHOS: [f32; 6] = [0.0, 0.2, 0.5, 0.8, 0.94, 1.0];
            const EPSS: [f32; 6] = [1.0, 0.2, 0.05, 0.01, 2e-3, 1e-4];
            for greedy in [false, true] {
                for &eps in &EPSS {
                    let fr: Vec<f64> = RHOS.iter().map(|&r| accept_frac(r, eps, greedy)).collect();
                    assert!(
                        fr.windows(2).all(|w| w[0] <= w[1]),
                        "ρ↑ eps={eps} g={greedy}: {fr:?}"
                    );
                }
                for &rho in &RHOS {
                    let fr: Vec<f64> = EPSS.iter().map(|&e| accept_frac(rho, e, greedy)).collect();
                    assert!(
                        fr.windows(2).all(|w| w[0] <= w[1]),
                        "ε↓ rho={rho} g={greedy}: {fr:?}"
                    );
                }
            }
            // Non-vacuity: the loosest setting must actually relax.
            assert!(accept_frac(1.0, 1e-4, false) > accept_frac(0.0, 1.0, false) + 0.3);
        }

        /// Nucleus-calibration property: for every law and ρ, the member set
        /// is (a) mass ≥ ρ (coverage) and (b) minimal — dropping its
        /// lowest-ranked member falls below ρ; (c) members are a rank prefix.
        /// Returns the first violation.
        fn nucleus_violation(
            member: &dyn Fn(&[f32], usize, f32) -> bool,
            seed: u64,
        ) -> Option<String> {
            let mut rng = Rng::new(seed);
            for i in 0..400 {
                let p = rand_law(6 + i % 40, &mut rng);
                // Rank: descending p, first-index tie.
                let mut order: Vec<usize> = (0..p.len()).collect();
                order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
                for &rho in &[0.0f32, 0.1, 0.33, 0.5, 0.77, 0.9, 0.97] {
                    let m: Vec<bool> = order.iter().map(|&t| member(&p, t, rho)).collect();
                    let n = m.iter().take_while(|&&b| b).count();
                    if m[n..].iter().any(|&b| b) {
                        return Some(format!("law {i} ρ={rho}: members not a rank prefix"));
                    }
                    let mass: f32 = order[..n].iter().map(|&t| p[t]).sum();
                    if mass < rho {
                        return Some(format!("law {i} ρ={rho}: coverage {mass} < ρ"));
                    }
                    if n > 0 && mass - p[order[n - 1]] >= rho {
                        return Some(format!("law {i} ρ={rho}: not minimal ({n} members)"));
                    }
                }
            }
            None
        }

        fn shipped_member(p: &[f32], d: usize, rho: f32) -> bool {
            in_rho_nucleus(p[d], mass_above(p, d), rho)
        }

        #[test]
        fn nucleus_membership_is_calibrated() {
            assert_eq!(nucleus_violation(&shipped_member, 9734), None);
        }

        /// Planted-violation arm: a miscalibrated nucleus must RED the
        /// calibration property (monotonicity alone is blind to these —
        /// every planted variant is still monotone in ρ).
        #[test]
        fn planted_miscalibrated_nucleus_reds() {
            // Off-by-one mass: counts d's own mass as "above".
            let off_by_one =
                |p: &[f32], d: usize, rho: f32| p[d] > 0.0 && mass_above(p, d) + p[d] < rho;
            // Overlapping bands: ties counted on BOTH sides (no tie rule).
            let overlap = |p: &[f32], d: usize, rho: f32| {
                let above: f32 = p
                    .iter()
                    .enumerate()
                    .filter(|&(t, &x)| x > p[d] || (x == p[d] && t != d))
                    .map(|(_, &x)| x)
                    .sum();
                p[d] > 0.0 && above < rho
            };
            // Inclusive edge: `≤ ρ` admits one member too many.
            let inclusive = |p: &[f32], d: usize, rho: f32| p[d] > 0.0 && mass_above(p, d) <= rho;
            assert!(
                nucleus_violation(&off_by_one, 9734).is_some(),
                "off-by-one must red"
            );
            assert!(
                nucleus_violation(&overlap, 9734).is_some(),
                "overlapping bands must red"
            );
            assert!(
                nucleus_violation(&inclusive, 9734).is_some(),
                "inclusive edge must red"
            );
        }

        /// Out-of-range / zero-mass drafts never pass the relaxed terms.
        #[test]
        fn zero_mass_draft_never_relaxed() {
            let mut p = toy_target_p();
            p[3] = 0.0;
            let s: f32 = p.iter().sum();
            p.iter_mut().for_each(|x| *x /= s);
            let mut vrng = Rng::new(5);
            for _ in 0..2000 {
                for d in [3, TOY_V + 7] {
                    let st = nucleus_accept_step(&p, d, 1.0, 0.0, false, &mut vrng);
                    assert!(matches!(st, PolicyStep::Correct(y) if y != 3 && y < TOY_V));
                    let st = nucleus_accept_step(&p, d, 1.0, 0.0, true, &mut vrng);
                    assert!(matches!(st, PolicyStep::Correct(_)));
                }
            }
        }

        /// Throughput A/B on fixtures (Issue 973 task 6): mean accepted
        /// tokens per speculate call, strict (`SoftmaxArgmax`) vs relaxed,
        /// over a K-position chain with a drafter that proposes the target's
        /// head most of the time and a junk token 10% of the time. Count metric, plus step
        /// latency. Run: `cargo test --release -p katgpt-forward --features
        /// nucleus_accept --lib nucleus::ab_throughput -- --ignored --nocapture`.
        #[test]
        #[ignore = "A/B report — run in --release with --nocapture"]
        fn ab_throughput() {
            const K: usize = 8;
            const CALLS: usize = 20_000;
            const V: usize = 4096;
            let mut law_rng = Rng::new(9735);
            // Fixture laws: LM-shaped peaked targets — Zipf(s) with s drawn
            // per law in [1.2, 2.4] (peaked → flat-ish heads), head order
            // shuffled within the top 8 only.
            let laws: Vec<Vec<f32>> = (0..64)
                .map(|_| {
                    let s_exp = 1.2 + 1.2 * law_rng.uniform();
                    let mut p: Vec<f32> =
                        (0..V).map(|t| 1.0 / (1.0 + t as f32).powf(s_exp)).collect();
                    for _ in 0..8 {
                        let a = (law_rng.next() as usize) % 8;
                        let b = (law_rng.next() as usize) % 8;
                        p.swap(a, b);
                    }
                    let s: f32 = p.iter().sum();
                    p.iter_mut().for_each(|x| *x /= s);
                    p
                })
                .collect();
            // Drafter: proposes the target's rank-r token — r = 0 w.p. .55,
            // 1 w.p. .2, 2..7 w.p. .15, and a JUNK token (rank 64..4095) w.p.
            // .1 — a plausible-but-imperfect drafter that sometimes misses.
            let ranked: Vec<Vec<usize>> = laws
                .iter()
                .map(|p| {
                    let mut o: Vec<usize> = (0..V).collect();
                    o.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
                    o
                })
                .collect();
            type Arm = (&'static str, Option<(f32, f32)>);
            let arms: [Arm; 5] = [
                ("SoftmaxArgmax (strict)", None),
                ("Nucleus corner ρ=0 ε=1", Some((0.0, 1.0))),
                ("Nucleus ε=2e-3", Some((0.0, 2e-3))),
                ("Nucleus ρ=0.94", Some((0.94, 1.0))),
                ("Nucleus ρ=0.94 ε=2e-3", Some((0.94, 2e-3))),
            ];
            for (name, knobs) in arms {
                let mut chain_rng = Rng::new(1);
                let mut rng = Rng::new(2);
                let mut vrng = Rng::new(rng.next() ^ NUCLEUS_VERIFY_DOMAIN);
                let mut accepted = 0usize;
                let t0 = std::time::Instant::now();
                let mut steps = 0usize;
                for _ in 0..CALLS {
                    for _ in 0..K {
                        let li = (chain_rng.next() as usize) % laws.len();
                        let u = chain_rng.uniform();
                        let r = if u < 0.55 {
                            0
                        } else if u < 0.75 {
                            1
                        } else if u < 0.9 {
                            2 + (chain_rng.next() as usize) % 6
                        } else {
                            64 + (chain_rng.next() as usize) % (V - 64)
                        };
                        let d = ranked[li][r];
                        let p = std::hint::black_box(&laws[li][..]);
                        let st = match knobs {
                            None => softmax_argmax_step(p, d, &mut rng),
                            Some((rho, eps)) => {
                                nucleus_accept_step(p, d, rho, eps, false, &mut vrng)
                            }
                        };
                        steps += 1;
                        if std::hint::black_box(st) != PolicyStep::Accept {
                            break;
                        }
                        accepted += 1;
                    }
                }
                let ns = t0.elapsed().as_nanos() as f64 / steps as f64;
                println!(
                    "{name:<26} accepted/call {:.3}  (K={K}, V={V})  step {ns:.0} ns",
                    accepted as f64 / CALLS as f64
                );
            }
        }
    }
}
