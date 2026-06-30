//! DFlash — zero-alloc speculative marginal-distribution drafter (shared core).
//!
//! Issue 013 Phase B (2026-06-30): collapses the dflash fork between
//! `katgpt-rs/src/speculative/dflash.rs` and
//! `riir-engine/src/dflash.rs`. The three `_with` algorithmic cores live
//! here so improvements propagate to both consumers. The thin wrappers
//! (`dflash_predict`, `_ar`, `_conditioned`, `_parallel`) stay in each
//! crate because they construct crate-specific `SpeculativeContext` /
//! `DraftResult`.
//!
//! # Parameterization
//!
//! `forward` and the drafter state types (`ForwardContext`,
//! `MultiLayerKVCache`, `TransformerWeights`) are crate-specific — both
//! crates have their own definitions with identical signatures but
//! disjoint bodies. To share the algorithm without sharing the types, the
//! `_with` functions here are generic over `Ctx`, `Cache`, `W`:
//!
//! - `Ctx: DflashCtx<W>` — exposes the logits buffer (read-back after
//!   `forward` fills `ctx.logits`) and the MTP-conditioning operation.
//! - `Cache: DflashCache` — exposes `reset`, `invalidate_position`
//!   (Issue 053 selective invalidation), and KV-layer seeding.
//! - `forward_fn: F` where `F: Fn(&mut Ctx, &W, &mut Cache, usize, usize, &Config)`
//!   — called for its side effect (writes `ctx.logits`). The returned
//!   borrow is discarded because a generic `Fn` returning a borrow tied to
//!   `&mut Ctx` cannot be expressed with elided lifetimes; reading the
//!   logits back via `DflashCtx::logits_slice` is equivalent (verified:
//!   both crates' `forward` return `&mut ctx.logits`).
//!
//! Callers do a disjoint field borrow on their `SpeculativeContext`
//! (`&mut sctx.ctx`, `&mut sctx.cache`, `&mut sctx.probs_buf`, …) which is
//! legal in Rust and avoids any split-borrow trait dance.
//!
//! # Free win
//!
//! The Issue 053 selective `invalidate_position` optimization lived only
//! in katgpt-rs; the shared `dflash_predict_with` now uses it, so
//! riir-engine gains it for free.

use katgpt_core::speculative::sampling::sample_from_distribution;
use katgpt_types::{Config, kv_dim, softmax_scaled};

// ── Backend traits ──────────────────────────────────────────────

/// Drafter cache operations used by the shared dflash cores.
///
/// Each backend (`katgpt-transformer::MultiLayerKVCache`,
/// `riir-engine::transformer::MultiLayerKVCache`) implements this so the
/// shared cores can drive cache reset, selective invalidation, and KV
/// seeding uniformly.
pub trait DflashCache {
    /// Full cache reset (clears every layer/position).
    fn reset(&mut self);

    /// Selective single-position invalidation — O(kv_dim) instead of full
    /// reset. Used by `dflash_predict_with` between steps (Issue 053).
    /// `pos` is the cache position to clear; `kv_dim` is the per-layer KV
    /// dimension.
    fn invalidate_position(&mut self, pos: usize, kv_dim: usize);

    /// Seed every layer's key/value buffers `[0..target_dim]` from
    /// `target_hidden` and zero-fill `[target_dim..draft_kv_dim]`, where
    /// `target_dim = min(target_hidden.len(), draft_kv_dim)`. Used by the
    /// target-conditioned drafter (Option C — KV-cache seeding).
    fn seed_layers(&mut self, target_hidden: &[f32], draft_kv_dim: usize);
}

/// Drafter forward-pass context operations used by the shared dflash cores.
///
/// `Weights` is the crate-specific weights type (e.g.
/// `TransformerWeights`); it is a type parameter so the MTP-conditioning
/// method can read `weights.lm_head` via the per-crate impl without
/// leaking the weights shape into this leaf.
pub trait DflashCtx<Weights: ?Sized> {
    /// Read the logits buffer `[vocab_size]` after a `forward` pass. Both
    /// crates' `forward` write the result into `ctx.logits` and return a
    /// `&mut` to it; reading it back here is equivalent and avoids the
    /// generic-fn-returning-a-borrow lifetime problem.
    fn logits_slice(&self) -> &[f32];

    /// MTP (Multi-Token-Prediction) conditioning: add `mtp_ctx[..n_embd]`
    /// into the hidden state on the first AR step, then recompute logits
    /// via `lm_head` matmul. Each backend implements this with direct
    /// field access (`self.hidden_state`, `self.logits`, `weights.lm_head`)
    /// — direct struct access is the only way to borrow those disjoint
    /// fields simultaneously within a single method body.
    fn apply_mtp_conditioning(
        &mut self,
        weights: &Weights,
        mtp_ctx: &[f32],
        n_embd: usize,
        vocab_size: usize,
    );
}

// ── Shared zero-alloc cores ─────────────────────────────────────

/// Zero-alloc variant of `dflash_predict`.
///
/// Reuses pre-allocated buffers passed by the caller (typically the
/// caller's `SpeculativeContext` fields). Each step gets selective KV
/// invalidation (Issue 053) rather than a full per-step reset.
///
/// Returns the number of steps populated; the caller reads via its own
/// `marginals_flat[start..start+vocab_size]` slices.
///
/// # Arguments
///
/// * `ctx`, `cache` — disjoint mutable borrows of the drafter state
///   (callers split these out of their `SpeculativeContext`).
/// * `weights` — drafter weights (passed straight through to `forward_fn`).
/// * `forward_fn` — the crate-specific `forward` fn, called for its side
///   effect (fills `ctx.logits`); its return value is discarded.
/// * `probs_buf` — scratch `[vocab_size]` for in-place softmax.
/// * `marginals_flat` — output `[max_steps * vocab_size]`, step `i`
///   occupies `[i*vocab_size .. (i+1)*vocab_size]`.
#[allow(clippy::too_many_arguments)]
pub fn dflash_predict_with<Ctx, Cache, Weights, F>(
    ctx: &mut Ctx,
    cache: &mut Cache,
    weights: &Weights,
    forward_fn: F,
    probs_buf: &mut [f32],
    marginals_flat: &mut [f32],
    draft_config: &Config,
    token: usize,
    pos: usize,
) -> usize
where
    Ctx: DflashCtx<Weights>,
    Cache: DflashCache,
    F: Fn(&mut Ctx, &Weights, &mut Cache, usize, usize, &Config),
{
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));
    let vocab_size = draft_config.vocab_size;
    let temperature = draft_config.temperature;
    let kvd = kv_dim(draft_config);

    // Full reset before the loop to clear any stale data from prior calls.
    cache.reset();
    for step in 0..max_steps {
        // Issue 053: selective invalidation — only zero the position written in
        // the previous step, not the entire cache. First step needs no
        // invalidation since we just did a full reset above.
        if step > 0 {
            cache.invalidate_position(pos + step - 1, kvd);
        }
        forward_fn(ctx, weights, cache, token, pos + step, draft_config);
        let logits = ctx.logits_slice();
        probs_buf[..vocab_size].copy_from_slice(&logits[..vocab_size]);
        softmax_scaled(&mut probs_buf[..vocab_size], 1.0 / temperature);
        let start = step * vocab_size;
        marginals_flat[start..start + vocab_size].copy_from_slice(&probs_buf[..vocab_size]);
    }

    max_steps
}

/// Zero-alloc variant of `dflash_predict_ar`.
///
/// Autoregressive: a single KV cache, sampled tokens feed back as the next
/// input. Produces conditional `q(x|x_{<i})` distributions instead of
/// independent marginals.
///
/// **Caller responsibility:** reset `cache` before calling (this allows KV
/// cache preloading between reset and the AR loop — Phase 3, Plan 055).
///
/// On the first step, if `mtp_context` is `Some`, MTP conditioning is
/// applied via [`DflashCtx::apply_mtp_conditioning`] before sampling.
///
/// # Arguments
///
/// * `sampled_tokens` — output `[max_steps]`, populated with the sampled
///   token at each step.
/// * `rng` — randomness source for sampling.
#[allow(clippy::too_many_arguments)]
pub fn dflash_predict_ar_with<Ctx, Cache, Weights, F>(
    ctx: &mut Ctx,
    cache: &mut Cache,
    weights: &Weights,
    forward_fn: F,
    probs_buf: &mut [f32],
    marginals_flat: &mut [f32],
    sampled_tokens: &mut [usize],
    draft_config: &Config,
    token: usize,
    pos: usize,
    rng: &mut katgpt_types::Rng,
    mtp_context: Option<&[f32]>,
) -> usize
where
    Ctx: DflashCtx<Weights>,
    Cache: DflashCache,
    F: Fn(&mut Ctx, &Weights, &mut Cache, usize, usize, &Config),
{
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos));
    let vocab_size = draft_config.vocab_size;
    let temperature = draft_config.temperature;

    let mut cur_token = token;
    for (step, token_slot) in sampled_tokens[..max_steps].iter_mut().enumerate() {
        forward_fn(ctx, weights, cache, cur_token, pos + step, draft_config);

        // MTP conditioning: inject target activations into drafter's hidden state
        // on the first step, then re-compute logits from the conditioned state.
        if step == 0
            && let Some(mtp_ctx) = mtp_context
        {
            ctx.apply_mtp_conditioning(weights, mtp_ctx, draft_config.n_embd, vocab_size);
        }

        let logits = ctx.logits_slice();
        probs_buf[..vocab_size].copy_from_slice(&logits[..vocab_size]);
        softmax_scaled(&mut probs_buf[..vocab_size], 1.0 / temperature);

        let next_token = sample_from_distribution(&probs_buf[..vocab_size], rng);
        let start = step * vocab_size;
        marginals_flat[start..start + vocab_size].copy_from_slice(&probs_buf[..vocab_size]);
        *token_slot = next_token;
        cur_token = next_token;
    }

    max_steps
}

/// Zero-alloc variant of `dflash_predict_conditioned`.
///
/// Seeds the drafter KV cache with the target model's hidden state (Option C
/// — projects the target hidden into the drafter's KV dimension and writes
/// it as the initial cache entry), then runs autoregressive drafting.
///
/// `max_steps` is one short of `block_size - pos` to leave room for the
/// seeded position.
#[allow(clippy::too_many_arguments)]
pub fn dflash_predict_conditioned_with<Ctx, Cache, Weights, F>(
    ctx: &mut Ctx,
    cache: &mut Cache,
    weights: &Weights,
    forward_fn: F,
    probs_buf: &mut [f32],
    marginals_flat: &mut [f32],
    sampled_tokens: &mut [usize],
    draft_config: &Config,
    token: usize,
    pos: usize,
    target_hidden_state: &[f32],
    rng: &mut katgpt_types::Rng,
) -> usize
where
    Ctx: DflashCtx<Weights>,
    Cache: DflashCache,
    F: Fn(&mut Ctx, &Weights, &mut Cache, usize, usize, &Config),
{
    cache.reset();
    let max_steps = draft_config
        .draft_lookahead
        .min(draft_config.block_size.saturating_sub(pos).saturating_sub(1));

    // Seed draft KV cache with target hidden state (Option C).
    let draft_kv_dim = kv_dim(draft_config);
    cache.seed_layers(target_hidden_state, draft_kv_dim);

    let vocab_size = draft_config.vocab_size;
    let temperature = draft_config.temperature;
    let mut cur_token = token;

    for (step, token_slot) in sampled_tokens[..max_steps].iter_mut().enumerate() {
        forward_fn(ctx, weights, cache, cur_token, pos + step + 1, draft_config);
        let logits = ctx.logits_slice();
        probs_buf[..vocab_size].copy_from_slice(&logits[..vocab_size]);
        softmax_scaled(&mut probs_buf[..vocab_size], 1.0 / temperature);

        let next_token = sample_from_distribution(&probs_buf[..vocab_size], rng);
        let start = step * vocab_size;
        marginals_flat[start..start + vocab_size].copy_from_slice(&probs_buf[..vocab_size]);
        *token_slot = next_token;
        cur_token = next_token;
    }

    max_steps
}
