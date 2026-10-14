//! Attention Residual Block — Kimi-K3 block-residual softmax mixing.
//!
//! Implements the `_apply_attn_res` mechanism from
//! `modeling_kimi_k3_linear.py` (Research 330 §5). This is a per-layer +
//! per-model mechanism that wraps around BOTH attention and MLP in every
//! decoder layer when `attn_res_block_size` is set (Kimi-K3-0.40B: 4).
//!
//! # The mechanism (single-token decode)
//!
//! Matches the actual `KimiDecoderLayer.forward` + `KimiLinearModel.forward`
//! path (Research 330 §5). Every `attn_res_block_size` layers, the block
//! residual accumulates the current hidden state. Within a layer, the
//! prefix sum (running hidden state) is mixed with the block's accumulated
//! residuals via a softmax-weighted combination:
//!
//! ```text
//! Given:
//!   prefix_sum    ∈ R^{d}     — the running hidden state
//!   block_residual — list of R^{d} vectors from previous blocks
//!
//! 1. Concatenate: v = [block_residual..., prefix_sum]  → [num_entries, d]
//! 2. RMSNorm each row (WITHOUT gamma — gamma is folded into score_weight):
//!    k[i] = v[i] / sqrt(mean(v[i]^2) + eps)
//! 3. Score weight (gamma folded with proj):
//!    score_weight = norm_weight ⊙ proj_weight   ∈ R^{d}
//! 4. Scores: scores[i] = dot(k[i], score_weight)  ∈ R^{num_entries}
//! 5. Softmax: probs = softmax(scores)  ∈ R^{num_entries}
//! 6. Output (weighted average of ORIGINAL v, not normed k):
//!    out = Σ_i probs[i] · v[i]
//! ```
//!
//! # Why the output uses ORIGINAL v (not normed k)
//!
//! The RMSNorm is only used to compute the mixing weights (how much each
//! entry contributes). The actual mixed value is the un-normalized hidden
//! state — this preserves the scale of the residual stream.
//!
//! # Decoder layer integration (Phase 5 - model composition)
//!
//! Per the Kimi-K3 tech report §2.2 (Block Attention Residuals):
//! - Layers are partitioned into N blocks of S layers each.
//! - Within block n, layer outputs are summed: `bn = Σ fj(hj)`.
//! - b0 = token embedding (always included as a source).
//! - The V matrix for the i-th layer in block n is:
//!   `[b0, b1, ..., bn-1, bin]` where `bin` is the partial sum of the
//!   current block up to layer i.
//!
//! In the decoder layer forward path:
//! ```text
//! // hidden IS the partial sum bin of the current block
//! hidden = apply_attn_res(hidden, block_residual, self_attn_res_proj, self_attn_res_norm)
//! residual = hidden
//! hidden = input_layernorm(hidden)
//! hidden = attention(hidden)
//! hidden = residual + hidden           // adds attention output to partial sum
//! hidden = apply_attn_res(hidden, block_residual, mlp_res_proj, mlp_res_norm)
//! residual = hidden
//! hidden = post_attention_layernorm(hidden)
//! hidden = ffn(hidden)
//! hidden = residual + hidden           // adds FFN output to partial sum
//! if at_block_boundary:
//!     block_residual.push(hidden)       // bn = completed block sum
//! ```
//!
//! At the model level (after all layers), `output_attn_res` applies one
//! final mixing with model-level weights, aggregating all N block
//! representations.

use katgpt_core::simd::{simd_dot_f32, simd_sum_sq};
use katgpt_core::softmax;

// ─── Config ─────────────────────────────────────────────────────────────────

/// Attention residual block configuration.
///
/// Mirrors the `attn_res_block_size` config field from Kimi-K3. When set,
/// every decoder layer uses the attention residual path instead of the
/// standard residual path.
#[derive(Clone, Debug)]
pub struct AttnResConfig {
    /// Hidden dim (`d`). Kimi-K3-0.40B: 1024.
    pub hidden_size: usize,
    /// Block size — how many layers per block before accumulating a residual.
    /// Kimi-K3-0.40B: 4 (`attn_res_block_size`).
    pub block_size: usize,
    /// RMSNorm epsilon. Kimi-K3-0.40B: 1e-5 (from `rms_norm_eps`).
    pub rms_eps: f32,
}

impl AttnResConfig {
    /// Kimi-K3-0.40B attention residual block configuration.
    pub fn kimi_k3_0_40b() -> Self {
        Self {
            hidden_size: 1024,
            block_size: 4,
            rms_eps: 1e-5,
        }
    }

    /// `d` — hidden dim.
    #[inline]
    pub fn d(&self) -> usize {
        self.hidden_size
    }
}

// ─── Weights ────────────────────────────────────────────────────────────────

/// Per-layer attention residual weights (one set for self-attention, one for MLP).
///
/// Each decoder layer has TWO of these: `self_attention_res_{norm,proj}` and
/// `mlp_res_{norm,proj}`. Plus the model has one at the output level:
/// `output_attn_res_{norm,proj}`.
///
/// Layouts (all row-major `Vec<f32>`):
/// - `norm_weight`: `[hidden_size]` — RMSNorm gamma weight
/// - `proj_weight`: `[hidden_size]` — Linear(hidden, 1) weight, stored as
///   `[hidden_size]` (the single output row)
///
/// The score weight is computed as `norm_weight ⊙ proj_weight` (element-wise
/// product), folding the gamma into the projection weight.
#[derive(Clone, Debug)]
pub struct AttnResWeights {
    /// RMSNorm gamma weight `[hidden_size]`.
    pub norm_weight: Vec<f32>,
    /// Linear projection weight `[hidden_size]` (single row of `[1, hidden]`).
    pub proj_weight: Vec<f32>,
}

impl AttnResWeights {
    /// Create random weights for testing (deterministic seed).
    pub fn random(hidden_size: usize, seed: u64) -> Self {
        use katgpt_core::Rng;
        let mut rng = Rng::new(seed);
        let norm_weight = (0..hidden_size)
            .map(|_| rng.uniform() * 0.02 - 0.01)
            .collect();
        let proj_weight = (0..hidden_size)
            .map(|_| rng.uniform() * 0.02 - 0.01)
            .collect();
        Self {
            norm_weight,
            proj_weight,
        }
    }

    /// Create ones-initialized weights (norm=1, proj=small).
    pub fn ones(hidden_size: usize) -> Self {
        Self {
            norm_weight: vec![1.0; hidden_size],
            proj_weight: vec![0.01; hidden_size],
        }
    }
}

// ─── Block residual state ───────────────────────────────────────────────────

/// Accumulated block residuals for the attention residual mechanism.
///
/// Per the Kimi-K3 tech report §2.2 (Block Attention Residuals): within each
/// block, layer outputs are reduced to a single representation by summation.
/// `b0` = token embedding; `bn` = Σ fj(hj) for layers j in block n.
///
/// This state tracks the list of completed block sums. It starts with the
/// embedding (b0) and grows by one entry at each block boundary. For an
/// 8-layer model with `block_size=4`, entries are accumulated at layers 0
/// (embedding) and 4 (sum of layers 0-3), resulting in up to 2 entries by
/// the final layer.
///
/// This state is per-token-position (one per position in the decode stream)
/// and is NOT synced — it's local computation state.
///
/// The caller (decoder layer) is responsible for computing the correct
/// summed value to push at each block boundary.
///
/// # Per-summary view cache (Issue 933 T3)
///
/// Each summary's `Σ v²` (the RMSNorm reduction) depends only on the summary,
/// never on the sublayer weights, yet every `apply_attn_res` call used to
/// recompute it for every summary — `2·S` times per summary per token. It is
/// computed once in [`push`](Self::push) and read by every later call. The
/// summaries are private (read through [`residuals`](Self::residuals)), so a
/// summary can only change via `push`/`clear`, which keep the cache in step.
///
/// # Zero-alloc steady state
///
/// The inner `Vec<f32>` slots are pre-allocated once (in [`new_with_capacity`])
/// and reused across tokens via [`clear`]. [`push`] copies into the next
/// free slot + bumps the logical length — no heap allocation in the steady
/// state. Exceeding the pre-allocated capacity falls back to a normal `push`
/// (grows the outer Vec); this is a debug-asserted contract violation, not a
/// production path.
#[derive(Clone, Debug)]
pub struct AttnResBlockState {
    /// The accumulated block residuals. Each entry is `[hidden_size]`.
    ///
    /// `residuals.len()` is the logical length; the outer Vec's capacity is
    /// pre-allocated in `new_with_capacity`. Inner slots are NOT dropped on
    /// `clear()` — they move to `pool` for reuse.
    residuals: Vec<Vec<f32>>,
    /// Cached `Σ v²` per summary, parallel to `residuals` (Issue 933 T3).
    sum_sq: Vec<f32>,
    /// Pre-allocated reusable slots (zero-alloc steady state).
    ///
    /// Pop on `push`, refill on `clear`. Eliminates per-token `Vec<f32>`
    /// allocation for models with bounded block accumulation.
    pool: Vec<Vec<f32>>,
    /// Hidden dim (cached for convenience).
    hidden_size: usize,
}

impl AttnResBlockState {
    /// Create empty block state.
    ///
    /// Equivalent to `new_with_capacity(hidden_size, 0)` — the first `push`
    /// will allocate. Prefer [`new_with_capacity`](Self::new_with_capacity)
    /// for hot paths where the max block count is known.
    pub fn new(hidden_size: usize) -> Self {
        Self {
            residuals: Vec::new(),
            sum_sq: Vec::new(),
            pool: Vec::new(),
            hidden_size,
        }
    }

    /// Create empty block state with pre-allocated slots for zero-alloc pushes.
    ///
    /// `max_entries` should be the maximum number of block boundaries the model
    /// will accumulate within a single forward pass — typically
    /// `num_layers / block_size` (2 for Kimi-K3-0.40B: 8 layers, block_size 4).
    /// Pushing beyond `max_entries` will allocate (debug-asserted contract
    /// violation).
    pub fn new_with_capacity(hidden_size: usize, max_entries: usize) -> Self {
        let pool = (0..max_entries)
            .map(|_| vec![0.0f32; hidden_size])
            .collect();
        Self {
            residuals: Vec::with_capacity(max_entries),
            sum_sq: Vec::with_capacity(max_entries),
            pool,
            hidden_size,
        }
    }

    /// Number of accumulated residuals.
    #[inline]
    pub fn len(&self) -> usize {
        self.residuals.len()
    }

    /// The accumulated block summaries, oldest first (read-only: mutation
    /// goes through [`push`](Self::push) / [`clear`](Self::clear) so the
    /// per-summary cache cannot go stale).
    #[inline]
    pub fn residuals(&self) -> &[Vec<f32>] {
        &self.residuals
    }

    /// Whether the block state is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.residuals.is_empty()
    }

    /// Append a hidden state to the block residual at a block boundary.
    ///
    /// Copies `hidden` into a slot drawn from the pre-allocated pool (fast
    /// path, zero-alloc) or allocates a new slot (slow path, contract
    /// violation).
    pub fn push(&mut self, hidden: &[f32]) {
        debug_assert_eq!(hidden.len(), self.hidden_size);
        let mut slot = if let Some(s) = self.pool.pop() {
            s
        } else {
            vec![0.0f32; self.hidden_size]
        };
        debug_assert_eq!(slot.len(), self.hidden_size);
        slot.copy_from_slice(hidden);
        self.sum_sq.push(simd_sum_sq(&slot, self.hidden_size));
        self.residuals.push(slot);
    }

    /// Clear the block state (for position reset).
    ///
    /// Moves all residual slots back into the pool without deallocating.
    /// Subsequent `push` calls reuse the same memory (zero-alloc steady state).
    pub fn clear(&mut self) {
        // Drain residuals back into the pool so the slots are available for
        // reuse on the next forward pass.
        while let Some(slot) = self.residuals.pop() {
            self.pool.push(slot);
        }
        self.sum_sq.clear();
    }

    /// Replace the accumulated summaries with `rows` (checkpoint restore /
    /// speculative rewind).
    ///
    /// Goes through [`clear`](Self::clear) + [`push`](Self::push) so the slot
    /// pool and the per-summary `Σ v²` cache (Issue 933 T3) stay in step —
    /// clearing directly and extending `residuals` would leave `sum_sq`
    /// empty, and [`apply_attn_res`] zips scores against BOTH vectors (the
    /// empty cache would silently no-op the whole mix).
    pub fn restore(&mut self, rows: &[Vec<f32>]) {
        self.clear();
        for row in rows {
            self.push(row);
        }
    }
}

// ─── Forward scratch ────────────────────────────────────────────────────────

/// Scratch buffers for the attention residual forward pass (zero-alloc steady state).
///
/// Pre-allocate once per decoder layer position and reuse across tokens.
#[derive(Clone, Debug)]
pub struct AttnResScratch {
    /// Combined score weight: `norm_weight ⊙ proj_weight` `[hidden_size]`.
    pub score_weight: Vec<f32>,
    /// Scores buffer `[max_entries]` (num_block_residuals + 1).
    pub scores: Vec<f32>,
    /// Output buffer `[hidden_size]`.
    pub out: Vec<f32>,
}

impl AttnResScratch {
    /// Create scratch for a max number of block entries.
    pub fn new(config: &AttnResConfig, max_block_entries: usize) -> Self {
        let d = config.d();
        Self {
            score_weight: vec![0.0; d],
            scores: vec![0.0; max_block_entries + 1],
            out: vec![0.0; d],
        }
    }

    /// Recompute score_weight from weights (call when weights change, e.g. at load).
    pub fn recompute_score_weight(&mut self, weights: &AttnResWeights) {
        let d = self.score_weight.len();
        for i in 0..d {
            self.score_weight[i] = weights.norm_weight[i] * weights.proj_weight[i];
        }
    }
}

// ─── Forward ────────────────────────────────────────────────────────────────

/// Apply the attention residual mixing (single-token decode).
///
/// This is the core `_apply_attn_res` function. It mixes the current hidden
/// state (`prefix_sum`) with the accumulated block residuals using
/// softmax-weighted combination. The mixing weights are computed from
/// RMSNorm'd values dotted with the score weight (gamma ⊙ proj).
///
/// Re-folds `scratch.score_weight` from `weights` on every call, so it stays
/// correct if the weights change between calls. A caller whose weights are
/// frozen for the scratch's lifetime folds once
/// ([`AttnResScratch::recompute_score_weight`]) and calls
/// [`apply_attn_res_prepared`] — bit-identical output.
///
/// # Arguments
/// - `config` — the attention residual config
/// - `weights` — the per-layer weights (norm + proj)
/// - `block_state` — accumulated block residuals
/// - `scratch` — pre-allocated scratch buffers
/// - `prefix_sum` — the current hidden state `[hidden_size]`
///
/// # Returns
/// The mixed hidden state `[hidden_size]`, written into `scratch.out`.
/// The caller should copy or use `scratch.out` before the next call.
///
/// # Panics
/// If `prefix_sum.len() != config.d()` or the scratch is undersized.
pub fn apply_attn_res<'a>(
    config: &AttnResConfig,
    weights: &AttnResWeights,
    block_state: &AttnResBlockState,
    scratch: &'a mut AttnResScratch,
    prefix_sum: &[f32],
) -> &'a mut [f32] {
    scratch.recompute_score_weight(weights);
    apply_attn_res_prepared(config, block_state, scratch, prefix_sum)
}

/// [`apply_attn_res`] with `scratch.score_weight` already folded.
///
/// Precondition: [`AttnResScratch::recompute_score_weight`] was called with
/// the weights this scratch serves, after their last change. Every summary's
/// `Σ v²` comes from the block state's push-time cache (Issue 933 T3), so the
/// per-call work is one dot per summary plus the weighted sum.
pub fn apply_attn_res_prepared<'a>(
    config: &AttnResConfig,
    block_state: &AttnResBlockState,
    scratch: &'a mut AttnResScratch,
    prefix_sum: &[f32],
) -> &'a mut [f32] {
    let d = config.d();
    debug_assert_eq!(prefix_sum.len(), d);
    debug_assert_eq!(scratch.score_weight.len(), d);
    debug_assert_eq!(block_state.sum_sq.len(), block_state.residuals.len());

    // Number of entries: block_residuals + 1 (the prefix_sum)
    let num_entries = block_state.len() + 1;
    debug_assert!(
        scratch.scores.len() >= num_entries,
        "scores buffer too small: {} < {}",
        scratch.scores.len(),
        num_entries
    );

    let scores = &mut scratch.scores[..num_entries];
    let score_weight = &scratch.score_weight[..d];

    // score[i] = dot(rmsnorm(v[i]), score_weight) = inv_rms * dot(v[i], score_weight)
    // (no gamma in the norm — gamma is folded into score_weight).
    let eps = config.rms_eps;
    let inv_rms = |sum_sq: f32| 1.0 / ((sum_sq / d as f32 + eps).sqrt());

    for ((score, residual), &sum_sq) in scores
        .iter_mut()
        .zip(&block_state.residuals)
        .zip(&block_state.sum_sq)
    {
        *score = inv_rms(sum_sq) * simd_dot_f32(residual, score_weight, d);
    }
    scores[num_entries - 1] =
        inv_rms(simd_sum_sq(prefix_sum, d)) * simd_dot_f32(prefix_sum, score_weight, d);

    softmax(scores);

    // Weighted average of ORIGINAL v values (not normed k): out = Σ_i probs[i] · v[i].
    // Same summation order as before; zipped slices let LLVM vectorize.
    let out = &mut scratch.out[..d];
    out.fill(0.0);
    for (&prob, residual) in scores.iter().zip(&block_state.residuals) {
        for (o, &rv) in out.iter_mut().zip(&residual[..d]) {
            *o += prob * rv;
        }
    }
    let prob = scores[num_entries - 1];
    for (o, &pv) in out.iter_mut().zip(&prefix_sum[..d]) {
        *o += prob * pv;
    }

    out
}

// ─── Fixed-query (depth-profile) arm — Issue 933 T4 ────────────────────────

/// katgpt-rs Issue 933 T4 — the fixed-query **modelless** mixing arm
/// (Research 619 E5, MSAR's conv-removal motivation): the learned
/// `score_weight` query is replaced by a deterministic exponential-depth
/// profile, `score[i] = -λ·age[i]` (age 0 = `prefix_sum`, +1 per completed
/// block older). No RMSNorm, no content scoring — the mixing weights are a
/// pure recency prior over the depth axis. For checkpoints WITHOUT AttnRes
/// weights (Gemma-2-class) this is the only constructible form of the
/// mechanism.
///
/// `λ = f32::INFINITY` is the DISARMED identity posture: `out` is a
/// byte-copy of `prefix_sum`, no entry contributes — the α=0 bit-identity
/// law (Issue 929 T2.1 precedent), pinned by test below. Finite large λ
/// saturates toward but never exactly reaches identity (softmax tails are
/// nonzero), which is why the arm exists explicitly.
///
/// Same entry order and weighted-sum accumulation shape as
/// [`apply_attn_res_prepared`], so the two arms share numerical structure
/// wherever they overlap.
///
/// # Panics
/// Debug builds assert `prefix_sum` non-empty, `scratch.out` sized to the
/// hidden dim, and `block_state.hidden_size == prefix_sum.len()`.
pub fn apply_attn_res_depth_profile<'a>(
    block_state: &AttnResBlockState,
    scratch: &'a mut AttnResScratch,
    lambda: f32,
    prefix_sum: &[f32],
) -> &'a mut [f32] {
    let d = prefix_sum.len();
    debug_assert!(d > 0, "empty prefix_sum");
    debug_assert_eq!(scratch.out.len(), d, "out buffer sized for another d");
    debug_assert_eq!(block_state.hidden_size, d);

    let out = &mut scratch.out[..d];

    // The disarmed posture: identity by construction (a memcpy is exact).
    if !lambda.is_finite() {
        out.copy_from_slice(prefix_sum);
        return out;
    }
    debug_assert!(
        lambda >= 0.0,
        "negative λ inverts the recency prior; use a finite ≥ 0 rate"
    );

    // scores[i] = -λ·age[i]: the prefix (age 0) scores 0, each completed
    // block one λ lower. Entry order matches `apply_attn_res_prepared`.
    let num_entries = block_state.len() + 1;
    debug_assert!(
        scratch.scores.len() >= num_entries,
        "scores buffer too small: {} < {}",
        scratch.scores.len(),
        num_entries
    );
    let scores = &mut scratch.scores[..num_entries];
    for (i, score) in scores.iter_mut().enumerate() {
        *score = -lambda * (num_entries - 1 - i) as f32;
    }

    softmax(scores);

    // Weighted average of ORIGINAL v values — identical accumulation shape
    // to `apply_attn_res_prepared`.
    out.fill(0.0);
    for (&prob, residual) in scores.iter().zip(&block_state.residuals) {
        for (o, &rv) in out.iter_mut().zip(&residual[..d]) {
            *o += prob * rv;
        }
    }
    let prob = scores[num_entries - 1];
    for (o, &pv) in out.iter_mut().zip(&prefix_sum[..d]) {
        *o += prob * pv;
    }

    out
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_kimi_k3_0_40b_dims() {
        let config = AttnResConfig::kimi_k3_0_40b();
        let weights = AttnResWeights::random(config.d(), 42);
        let block_state = AttnResBlockState::new(config.d());
        let mut scratch = AttnResScratch::new(&config, max_entries(&config));

        let prefix_sum: Vec<f32> = (0..config.d()).map(|i| (i as f32).sin() * 0.1).collect();
        let out = apply_attn_res(&config, &weights, &block_state, &mut scratch, &prefix_sum);

        assert_eq!(out.len(), config.d());
        for &v in out.iter() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn kimi_k3_0_40b_config_values() {
        let config = AttnResConfig::kimi_k3_0_40b();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.block_size, 4);
        assert!(
            (config.rms_eps - 1e-5).abs() < 1e-10,
            "rms_eps must be 1e-5"
        );
    }

    #[test]
    fn empty_block_state_returns_scaled_prefix() {
        // With no block residuals, there's only 1 entry (prefix_sum).
        // softmax of a single element = [1.0], so output = prefix_sum.
        let config = AttnResConfig::kimi_k3_0_40b();
        let weights = AttnResWeights::ones(config.d());
        let block_state = AttnResBlockState::new(config.d());
        let mut scratch = AttnResScratch::new(&config, max_entries(&config));

        let prefix_sum: Vec<f32> = (0..config.d()).map(|i| (i as f32) * 0.01).collect();
        let out = apply_attn_res(&config, &weights, &block_state, &mut scratch, &prefix_sum);

        // With 1 entry, softmax is trivially [1.0], output = prefix_sum
        for i in 0..config.d() {
            assert!(
                (out[i] - prefix_sum[i]).abs() < 1e-5,
                "empty block: out[{i}] = {} expected {}",
                out[i],
                prefix_sum[i]
            );
        }
    }

    #[test]
    fn block_accumulation_grows_entries() {
        let config = AttnResConfig::kimi_k3_0_40b();
        let mut block_state = AttnResBlockState::new(config.d());

        assert_eq!(block_state.len(), 0);
        assert!(block_state.is_empty());

        let h = vec![0.1; config.d()];
        block_state.push(&h);
        assert_eq!(block_state.len(), 1);

        block_state.push(&h);
        assert_eq!(block_state.len(), 2);
    }

    #[test]
    fn output_is_convex_combination() {
        // The output must be a convex combination (weighted average) of the
        // input entries (block residuals + prefix_sum). Each output element
        // must be within [min, max] of the input entries at that position.
        let config = AttnResConfig::kimi_k3_0_40b();
        let weights = AttnResWeights::random(config.d(), 99);
        let mut block_state = AttnResBlockState::new(config.d());

        // Push two residuals with known distinct values
        let r1: Vec<f32> = vec![1.0; config.d()];
        let r2: Vec<f32> = vec![-1.0; config.d()];
        block_state.push(&r1);
        block_state.push(&r2);

        let mut scratch = AttnResScratch::new(&config, max_entries(&config));
        let prefix_sum: Vec<f32> = vec![0.5; config.d()];
        let out = apply_attn_res(&config, &weights, &block_state, &mut scratch, &prefix_sum);

        // All entries at position i are in {-1.0, 0.5, 1.0}, so output must
        // be in [-1.0, 1.0]
        for &v in out.iter() {
            assert!(
                (-1.0 - 1e-5..=1.0 + 1e-5).contains(&v),
                "output {v} outside convex hull [-1, 1]"
            );
        }
    }

    #[test]
    fn deterministic_same_inputs_same_output() {
        let config = AttnResConfig::kimi_k3_0_40b();
        let weights = AttnResWeights::random(config.d(), 42);
        let mut block_state = AttnResBlockState::new(config.d());

        let r1: Vec<f32> = (0..config.d()).map(|i| (i as f32).sin() * 0.1).collect();
        block_state.push(&r1);

        let prefix_sum: Vec<f32> = (0..config.d()).map(|i| (i as f32).cos() * 0.1).collect();

        let mut scratch1 = AttnResScratch::new(&config, max_entries(&config));
        let mut scratch2 = AttnResScratch::new(&config, max_entries(&config));

        let out1 = apply_attn_res(&config, &weights, &block_state, &mut scratch1, &prefix_sum);
        let out1_copy: Vec<f32> = out1.to_vec();
        let out2 = apply_attn_res(&config, &weights, &block_state, &mut scratch2, &prefix_sum);

        for i in 0..config.d() {
            assert!(
                (out1_copy[i] - out2[i]).abs() < 1e-7,
                "non-deterministic: out1[{i}]={}, out2[{i}]={}",
                out1_copy[i],
                out2[i]
            );
        }
    }

    #[test]
    fn score_weight_is_element_wise_product() {
        let d = 64;
        let weights = AttnResWeights {
            norm_weight: (0..d).map(|i| i as f32 * 0.1).collect(),
            proj_weight: (0..d).map(|i| i as f32 * 0.2).collect(),
        };
        let config = AttnResConfig {
            hidden_size: d,
            block_size: 4,
            rms_eps: 1e-5,
        };
        let mut scratch = AttnResScratch::new(&config, 4);
        scratch.recompute_score_weight(&weights);

        for i in 0..d {
            let expected = (i as f32 * 0.1) * (i as f32 * 0.2);
            assert!(
                (scratch.score_weight[i] - expected).abs() < 1e-6,
                "score_weight[{i}] = {} expected {}",
                scratch.score_weight[i],
                expected
            );
        }
    }

    /// Compute the max number of block entries for an 8-layer model with block_size=4.
    fn max_entries(config: &AttnResConfig) -> usize {
        // 8 layers / block_size=4 = 2 block boundaries (layers 0, 4)
        // Plus 1 for the prefix_sum entry
        // Max entries = 2 + 1 = 3. Round up generously for safety.
        let num_layers = 8;
        (num_layers / config.block_size + 1) + 1
    }

    // ── Issue 933 T4: the depth-profile (fixed-query) arm ──────────────────

    #[test]
    fn depth_profile_lambda_inf_is_bitwise_identity() {
        // The disarmed posture: out must equal prefix_sum EXACTLY (to_bits),
        // even with entries present that a finite λ would mix in.
        let config = AttnResConfig::kimi_k3_0_40b();
        let mut block_state = AttnResBlockState::new(config.d());
        block_state.push(&vec![7.0; config.d()]);
        block_state.push(&vec![-3.0; config.d()]);
        let mut scratch = AttnResScratch::new(&config, max_entries(&config));

        let prefix_sum: Vec<f32> = (0..config.d()).map(|i| (i as f32).sin() * 0.1).collect();
        let out = apply_attn_res_depth_profile(
            &block_state,
            &mut scratch,
            f32::INFINITY,
            &prefix_sum,
        );
        for i in 0..config.d() {
            assert_eq!(
                out[i].to_bits(),
                prefix_sum[i].to_bits(),
                "λ=∞ must be bitwise identity at [{i}]"
            );
        }
    }

    #[test]
    fn depth_profile_lambda_zero_is_uniform_mean() {
        // λ=0 → every score 0 → softmax uniform. With ONE residual + prefix
        // the weights are exactly [0.5, 0.5] (exp(0)/2), so constant vectors
        // 2.0 and 4.0 mix to exactly 3.0.
        let config = AttnResConfig::kimi_k3_0_40b();
        let mut block_state = AttnResBlockState::new(config.d());
        block_state.push(&vec![2.0; config.d()]);
        let mut scratch = AttnResScratch::new(&config, max_entries(&config));

        let prefix_sum = vec![4.0; config.d()];
        let out = apply_attn_res_depth_profile(&block_state, &mut scratch, 0.0, &prefix_sum);
        for i in 0..config.d() {
            assert_eq!(out[i], 3.0, "λ=0 must be the exact uniform mean at [{i}]");
        }
    }

    #[test]
    fn depth_profile_recency_is_monotone_in_lambda() {
        // λ>0 pulls the mix toward the prefix (age 0). With residual 10.0 and
        // prefix 0.0 the output must sit below the uniform mean (5.0) and
        // shrink monotonically as λ grows.
        let config = AttnResConfig::kimi_k3_0_40b();
        let mut block_state = AttnResBlockState::new(config.d());
        block_state.push(&vec![10.0; config.d()]);
        let prefix_sum = vec![0.0; config.d()];

        let mut prev = f32::MAX;
        for lambda in [0.5f32, 1.0, 2.0, 4.0, 8.0] {
            let mut scratch = AttnResScratch::new(&config, max_entries(&config));
            let out =
                apply_attn_res_depth_profile(&block_state, &mut scratch, lambda, &prefix_sum);
            assert!(
                out[0] < 5.0,
                "λ={lambda}: recency must pull below the uniform mean, got {}",
                out[0]
            );
            assert!(
                out[0] < prev,
                "λ={lambda}: mix must shrink monotonically ({} !< {prev})",
                out[0]
            );
            prev = out[0];
        }
    }

    #[test]
    fn depth_profile_empty_state_returns_prefix() {
        // No residuals → a single entry → softmax [1.0] → prefix, any λ.
        let config = AttnResConfig::kimi_k3_0_40b();
        let block_state = AttnResBlockState::new(config.d());
        let prefix_sum: Vec<f32> = (0..config.d()).map(|i| (i as f32) * 0.25).collect();
        for lambda in [0.0f32, 2.0, 100.0, f32::INFINITY] {
            let mut scratch = AttnResScratch::new(&config, max_entries(&config));
            let out =
                apply_attn_res_depth_profile(&block_state, &mut scratch, lambda, &prefix_sum);
            for i in 0..config.d() {
                assert_eq!(
                    out[i].to_bits(),
                    prefix_sum[i].to_bits(),
                    "empty state, λ={lambda}: must return prefix at [{i}]"
                );
            }
        }
    }

    #[test]
    fn depth_profile_deterministic_same_inputs_same_output() {
        let config = AttnResConfig::kimi_k3_0_40b();
        let mut block_state = AttnResBlockState::new(config.d());
        block_state.push(&(0..config.d()).map(|i| (i as f32).sin() * 0.1).collect::<Vec<_>>());
        block_state.push(&(0..config.d()).map(|i| (i as f32).cos() * 0.1).collect::<Vec<_>>());
        let prefix_sum: Vec<f32> = (0..config.d()).map(|i| (i as f32) * 0.05 - 1.0).collect();

        let mut s1 = AttnResScratch::new(&config, max_entries(&config));
        let mut s2 = AttnResScratch::new(&config, max_entries(&config));
        let o1 = apply_attn_res_depth_profile(&block_state, &mut s1, 1.5, &prefix_sum).to_vec();
        let o2 = apply_attn_res_depth_profile(&block_state, &mut s2, 1.5, &prefix_sum);
        for i in 0..config.d() {
            assert_eq!(o1[i].to_bits(), o2[i].to_bits(), "non-deterministic at [{i}]");
        }
    }
}
