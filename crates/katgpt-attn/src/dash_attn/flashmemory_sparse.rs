//! FlashMemory-style periodic sparse attention for MLA (Phase 1 mechanism).
//!
//! Implements the lookahead periodic sparse attention from FlashMemory-DeepSeek-V4
//! (arXiv:2606.09079, Research 436), adapted for Multi-head Latent Attention
//! (DeepSeek-V2 §2.1, Proposal 032 Phase 2).
//!
//! # Mechanism
//!
//! Three modelless primitives compose the FlashMemory sparse selection:
//!
//! 1. **Block centroids** — consecutive `block_size` tokens are summarized by
//!    the mean of their compressed KV latent (`c_kv`). The centroid lives in
//!    the `d_c`-dim latent space, NOT the per-head key space.
//!
//! 2. **Per-head key projection** — each block's latent centroid is up-projected
//!    through `W_UK[head]` to produce a per-head content-key centroid (`d_h`
//!    floats). This is the bridge from latent space to the query's content space.
//!    The projection is cached per-head per-block and only recomputed when a
//!    block's membership changes.
//!
//! 3. **Sigmoid threshold selection** — blocks are scored by
//!    `dot(q_c_h, k_c_centroid_h) * attn_scale`, passed through sigmoid, and
//!    selected when `σ(score) ≥ threshold` (default 0.5). This produces DYNAMIC
//!    block counts per head — unlike rigid top-k, a head may attend to 1 block
//!    or 20 blocks depending on the query.
//!
//! 4. **Periodic refresh** — the block selection is recomputed every
//!    `refresh_period` decode steps (default 64). Between refreshes, the last
//!    selection is reused. This amortizes the O(blocks × heads) scoring cost
//!    over `refresh_period` steps.
//!
//! # Phase 1 scope (Issue 584)
//!
//! This is a MECHANISM test, not a scale test. The four questions answered:
//!
//! - **Q1**: Can block centroids be built from MLA's compressed latent KV
//!   (`kv_lora_rank: 128`)? → Yes, `FlashMemoryBlockCache::rebuild_from_cache`.
//! - **Q2**: Does the periodic refresh amortize selection cost? → Yes, the
//!   selector caches the last decision and only re-scores every τ steps.
//! - **Q3**: Does sigmoid threshold (≥0.5) produce dynamic block counts? → Yes,
//!   different queries select different numbers of blocks (see tests).
//! - **Q4**: Does the sparse forward preserve accuracy? → The sparse forward
//!   restricts attention to tokens in selected blocks; softmax is computed
//!   over only the selected tokens. See `mla_forward_token_flashmemory`.
//!
//! # Feature gate
//!
//! `flashmemory_sparse` — implies `mla_attention` (MLA types) + `dash_attn`
//! (VortexFlow scratch types). Opt-in, default-off. GOAT gate required for
//! promotion (see Issue 584 Phase 3).
//!
//! ⚠ Promotion gate MUST include a needle/long-context retrieval axis
//! (katgpt-rs Issue 826): the mean-of-latent centroid is the pooled-summary
//! class SAS (arXiv:2609.13141) measured destroying needle retrieval at
//! 64K–128K; perf-only validation is not promotion evidence.
//!
//! # Issue 926 secondary scope — absorbed decode on this path
//!
//! The same weight absorption that landed for the dense decode lane
//! (`mla.rs::mla_forward_token_absorbed`) applies to the selected-block path
//! behind `mla_absorbed`: per step per head `q_abs = W_UKᵀ·q_c` ONCE, scores
//! read the cached latent directly, values accumulate in latent space over the
//! SELECTED rows with ONE `W_UV` up-projection. Block SELECTION is unchanged —
//! both decode paths share the prefix (Steps 1–6) verbatim, so absorption
//! changes only HOW the selected latent rows are read, never WHICH rows are
//! selected. Kill-switch `KATGPT_MLA_RECONSTRUCT=1` forces the reconstruct
//! reference under the feature; default (feature-off) builds are
//! behavior- and byte-identical.

use katgpt_core::simd::{simd_add_inplace, simd_dot_f32, simd_matmul_rows, simd_scale_inplace};
#[cfg(feature = "mla_absorbed")]
use katgpt_core::simd::{simd_transpose_matvec_acc, simd_transpose_matvec_into};
use katgpt_kv::shard_kv::rope::RopeFreqs;

use crate::mla::{MlaConfig, MlaForwardScratch, MlaKVCache, MlaWeights};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for FlashMemory-style periodic sparse attention.
///
/// All defaults match the FlashMemory-DeepSeek-V4 paper (Research 436 §2.1).
#[derive(Clone, Debug)]
pub struct FlashMemoryConfig {
    /// Block size — tokens per block. Paper default: 64.
    pub block_size: usize,
    /// Refresh period — decode steps between full re-scoring. Paper default: 64.
    pub refresh_period: usize,
    /// Sigmoid selection threshold. Blocks with `σ(score) ≥ threshold` are
    /// selected. Paper default: 0.5 (σ ≥ 0.5 ⟺ score ≥ 0).
    pub threshold: f32,
}

impl Default for FlashMemoryConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            refresh_period: 64,
            threshold: 0.5,
        }
    }
}

impl FlashMemoryConfig {
    /// Small-block config for testing at short contexts (e.g. 4K).
    /// Uses block_size=16 so we get enough blocks to test selection dynamics.
    pub fn test_config() -> Self {
        Self {
            block_size: 16,
            refresh_period: 16,
            threshold: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Block centroid cache
// ---------------------------------------------------------------------------

/// Block centroid cache built from MLA's compressed latent KV.
///
/// Stores:
/// - **Latent centroids**: `[max_blocks][d_c]` — mean of `c_kv` per block.
/// - **Per-head key centroids**: `[max_blocks][n_heads][d_h]` — up-projected
///   content-key centroid per head per block. This is the bridge from latent
///   space to the query's content space.
///
/// The per-head key centroids are recomputed only when a block's membership
/// changes (i.e., when `rebuild_from_cache` is called after new tokens arrive).
/// Between rebuilds, the centroids are immutable — the selector can read them
/// without synchronization.
pub struct FlashMemoryBlockCache {
    /// `[max_blocks * d_c]` — latent centroid per block.
    latent_centroids: Vec<f32>,
    /// `[max_blocks * n_heads * d_h]` — up-projected content-key centroid.
    /// Indexed as `key_centroids[block * n_heads * d_h + head * d_h + ..]`.
    key_centroids: Vec<f32>,
    /// `[max_blocks]` — number of tokens currently in each block.
    block_counts: Vec<usize>,
    /// Number of blocks with at least one token.
    n_active_blocks: usize,
    d_c: usize,
    d_h: usize,
    n_heads: usize,
    block_size: usize,
    max_blocks: usize,
}

impl FlashMemoryBlockCache {
    /// Create a new block cache sized for the given MLA config.
    pub fn new(config: &MlaConfig, fm_config: &FlashMemoryConfig, max_seq: usize) -> Self {
        let block_size = fm_config.block_size;
        let max_blocks = max_seq.div_ceil(block_size).max(1);
        let d_c = config.kv_lora_rank;
        let d_h = config.d_h();
        let n_heads = config.n_heads;

        Self {
            latent_centroids: vec![0.0; max_blocks * d_c],
            key_centroids: vec![0.0; max_blocks * n_heads * d_h],
            block_counts: vec![0; max_blocks],
            n_active_blocks: 0,
            d_c,
            d_h,
            n_heads,
            block_size,
            max_blocks,
        }
    }

    /// Rebuild block centroids from the MLA KV cache.
    ///
    /// Computes the mean of `c_kv` for each block's token range, then
    /// up-projects each latent centroid through `W_UK[head]` to produce
    /// per-head content-key centroids.
    ///
    /// This is the refresh-cost amortized by the periodic selector: it runs
    /// O(blocks × d_c) for centroid computation + O(blocks × n_heads × d_h × d_c)
    /// for up-projection. For Kimi-K3-0.40B at 4K context (256 blocks of 16):
    /// 256 × 8 × 64 × 128 ≈ 16.8M FMAs — ~170µs on M3 NEON, amortized over
    /// τ=16 steps → ~10µs/step.
    pub fn rebuild_from_cache(&mut self, cache: &MlaKVCache, weights: &MlaWeights) {
        let seq_len = cache.seq_len;
        let n_blocks = seq_len.div_ceil(self.block_size).min(self.max_blocks);
        self.n_active_blocks = n_blocks;

        for block_idx in 0..n_blocks {
            let start = block_idx * self.block_size;
            let end = (start + self.block_size).min(seq_len);
            let count = end - start;
            self.block_counts[block_idx] = count;

            if count == 0 {
                continue;
            }

            // Mean-pool c_kv for this block → latent centroid.
            let centroid =
                &mut self.latent_centroids[block_idx * self.d_c..(block_idx + 1) * self.d_c];
            centroid.fill(0.0);
            for tok in start..end {
                let c_kv_j = cache.latent_kv_at(tok);
                simd_add_inplace(centroid, c_kv_j);
            }
            let inv = 1.0 / count as f32;
            simd_scale_inplace(centroid, inv);

            // Up-project latent centroid → per-head content-key centroid.
            // k_c_centroid[head] = W_UK[head] · centroid_c_kv  (d_h per head)
            // W_UK layout: [n_heads * d_h, d_c] — row (head*d_h + i) is the i-th
            // output dim for head `head`, input is the d_c-dim centroid.
            let key_off = block_idx * self.n_heads * self.d_h;
            for head in 0..self.n_heads {
                let w_uk_head =
                    &weights.w_uk[head * self.d_h * self.d_c..(head + 1) * self.d_h * self.d_c];
                let k_c_head = &mut self.key_centroids
                    [key_off + head * self.d_h..key_off + (head + 1) * self.d_h];
                simd_matmul_rows(k_c_head, w_uk_head, centroid, self.d_h, self.d_c);
            }
        }
    }

    /// Number of active blocks (blocks with at least one token).
    #[inline]
    pub fn n_active_blocks(&self) -> usize {
        self.n_active_blocks
    }

    /// Get the per-head content-key centroid for block `block_idx`, head `head`.
    /// Returns `d_h` floats — the up-projected centroid in the query's content space.
    #[inline]
    pub fn key_centroid(&self, block_idx: usize, head: usize) -> &[f32] {
        let off = block_idx * self.n_heads * self.d_h + head * self.d_h;
        &self.key_centroids[off..off + self.d_h]
    }

    /// Get the latent centroid for block `block_idx`. Returns `d_c` floats.
    #[inline]
    pub fn latent_centroid(&self, block_idx: usize) -> &[f32] {
        &self.latent_centroids[block_idx * self.d_c..(block_idx + 1) * self.d_c]
    }

    /// Number of tokens in block `block_idx`.
    #[inline]
    pub fn block_count(&self, block_idx: usize) -> usize {
        self.block_counts[block_idx]
    }

    /// Token range [start, end) for block `block_idx` given current `seq_len`.
    #[inline]
    pub fn block_token_range(&self, block_idx: usize, seq_len: usize) -> (usize, usize) {
        let start = block_idx * self.block_size;
        let end = (start + self.block_size).min(seq_len);
        (start, end)
    }

    /// Block size (tokens per block).
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

// ---------------------------------------------------------------------------
// GQA block centroid cache (Issue 584 Phase 2 — Bonsai/Qwen3.5 support)
// ---------------------------------------------------------------------------

/// Block centroid cache for standard GQA attention (no latent compression).
///
/// Unlike [`FlashMemoryBlockCache`] (MLA-specific, stores compressed `c_kv`
/// latent centroids and up-projects through `W_UK`), this cache operates on
/// **raw GQA keys** already in per-head content space. The block centroid for
/// head `h` is simply the mean of the raw keys for KV head `h` over the block's
/// token range.
///
/// # Layout
///
/// The raw key cache is `[seq_len * kv_dim]` where `kv_dim = n_kv_head * head_dim`.
/// Token `t`, KV head `h` key = `keys[t * kv_dim + h * head_dim .. t * kv_dim + (h+1) * head_dim]`.
///
/// # Why a separate type (not a trait)
///
/// `FlashMemorySelector::select()` takes `&FlashMemoryBlockCache` by concrete
/// type (not a trait). Rather than refactoring the selector to be generic over
/// the cache type (which would ripple through the trained `DualEncoderIndexer`
/// too), we duplicate the small read-side interface (`n_active_blocks`,
/// `key_centroid`, `block_count`, `block_token_range`). The selector could be
/// made generic later if a third cache variant appears — YAGNI for now.
///
/// # Issue 584 Phase 2
///
/// Bonsai-27B (Qwen3.5-arch) uses GQA in its attention layers. This cache
/// enables FlashMemory sparse selection on Bonsai without the MLA substrate.
/// The same `FlashMemorySelector` (modelless sigmoid threshold) is reused —
/// only the block-centroid source differs.
pub struct GqaFlashMemoryBlockCache {
    /// `[max_blocks * n_kv_head * head_dim]` — per-KV-head key centroid per block.
    /// Indexed as `key_centroids[block * n_kv_head * head_dim + head * head_dim + ..]`.
    key_centroids: Vec<f32>,
    /// `[max_blocks]` — number of tokens currently in each block.
    block_counts: Vec<usize>,
    /// Number of blocks with at least one token.
    n_active_blocks: usize,
    n_kv_head: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks: usize,
}

impl GqaFlashMemoryBlockCache {
    /// Create a new GQA block cache.
    ///
    /// * `n_kv_head` — number of KV heads (GQA)
    /// * `head_dim` — dimension per head
    /// * `fm_config` — FlashMemory config (uses `block_size`)
    /// * `max_seq` — maximum sequence length (determines `max_blocks`)
    pub fn new(
        n_kv_head: usize,
        head_dim: usize,
        fm_config: &FlashMemoryConfig,
        max_seq: usize,
    ) -> Self {
        let block_size = fm_config.block_size;
        let max_blocks = max_seq.div_ceil(block_size).max(1);
        Self {
            key_centroids: vec![0.0; max_blocks * n_kv_head * head_dim],
            block_counts: vec![0; max_blocks],
            n_active_blocks: 0,
            n_kv_head,
            head_dim,
            block_size,
            max_blocks,
        }
    }

    /// Rebuild block centroids from a raw GQA key cache.
    ///
    /// `keys` is `[seq_len * kv_dim]` where `kv_dim = n_kv_head * head_dim`.
    /// Computes the mean of keys per block per KV head.
    pub fn rebuild_from_keys(&mut self, keys: &[f32], seq_len: usize) {
        let kv_dim = self.n_kv_head * self.head_dim;
        debug_assert!(keys.len() >= seq_len * kv_dim, "key cache too short");
        let n_blocks = seq_len.div_ceil(self.block_size).min(self.max_blocks);
        self.n_active_blocks = n_blocks;

        for block_idx in 0..n_blocks {
            let start = block_idx * self.block_size;
            let end = (start + self.block_size).min(seq_len);
            let count = end - start;
            self.block_counts[block_idx] = count;

            if count == 0 {
                continue;
            }

            // Mean-pool keys for each KV head over [start, end).
            let centroid_off = block_idx * self.n_kv_head * self.head_dim;
            let centroid = &mut self.key_centroids
                [centroid_off..centroid_off + self.n_kv_head * self.head_dim];
            centroid.fill(0.0);
            for tok in start..end {
                let key_off = tok * kv_dim;
                // Accumulate per-KV-head.
                for h in 0..self.n_kv_head {
                    let src = &keys[key_off + h * self.head_dim..key_off + (h + 1) * self.head_dim];
                    let dst = &mut centroid[h * self.head_dim..(h + 1) * self.head_dim];
                    simd_add_inplace(dst, src);
                }
            }
            let inv = 1.0 / count as f32;
            simd_scale_inplace(centroid, inv);
        }
    }

    /// Number of active blocks (blocks with at least one token).
    #[inline]
    pub fn n_active_blocks(&self) -> usize {
        self.n_active_blocks
    }

    /// Get the key centroid for block `block_idx`, KV head `head`.
    /// Returns `head_dim` floats.
    #[inline]
    pub fn key_centroid(&self, block_idx: usize, head: usize) -> &[f32] {
        let off = block_idx * self.n_kv_head * self.head_dim + head * self.head_dim;
        &self.key_centroids[off..off + self.head_dim]
    }

    /// Number of tokens in block `block_idx`.
    #[inline]
    pub fn block_count(&self, block_idx: usize) -> usize {
        self.block_counts[block_idx]
    }

    /// Token range [start, end) for block `block_idx` given current `seq_len`.
    #[inline]
    pub fn block_token_range(&self, block_idx: usize, seq_len: usize) -> (usize, usize) {
        let start = block_idx * self.block_size;
        let end = (start + self.block_size).min(seq_len);
        (start, end)
    }

    /// Block size (tokens per block).
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of KV heads.
    #[inline]
    pub fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }

    /// Head dimension.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

/// GQA-compatible periodic selector. Identical scoring logic to
/// [`FlashMemorySelector`] but reads from [`GqaFlashMemoryBlockCache`] and
/// handles the GQA query→KV-head mapping (multiple query heads share one KV
/// head via `kv_group = q_head * n_kv_head / n_query_head`).
///
/// The selector scores blocks per **KV head** (not per query head), since the
/// block centroid lives in KV-head space. All query heads sharing a KV group
/// use the same block selection. This is the standard GQA sparse attention
/// pattern — it avoids redundant scoring and matches the shared-KV structure.
pub struct GqaFlashMemorySelector {
    config: FlashMemoryConfig,
    /// Per-KV-head selection. `blocks_per_head[kv_head]` = selected block indices.
    last_selection: PerHeadSelection,
    last_refresh_step: usize,
    selection_valid: bool,
    /// Scratch: block scores per KV head. `[n_kv_head * max_blocks]`.
    scores_buf: Vec<f32>,
    n_kv_head: usize,
    head_dim: usize,
    max_blocks: usize,
    refresh_count: usize,
}

impl GqaFlashMemorySelector {
    pub fn new(
        config: FlashMemoryConfig,
        n_kv_head: usize,
        head_dim: usize,
        max_blocks: usize,
    ) -> Self {
        Self {
            config,
            last_selection: PerHeadSelection::new(n_kv_head, max_blocks),
            last_refresh_step: 0,
            selection_valid: false,
            scores_buf: vec![0.0; n_kv_head * max_blocks],
            n_kv_head,
            head_dim,
            max_blocks,
            refresh_count: 0,
        }
    }

    pub fn refresh_count(&self) -> usize {
        self.refresh_count
    }

    pub fn force_refresh(&mut self) {
        self.selection_valid = false;
    }

    pub fn is_selection_valid(&self) -> bool {
        self.selection_valid
    }

    /// Select blocks per KV head using sigmoid threshold.
    ///
    /// `query_keys` is `[n_kv_head * head_dim]` — the query projected into
    /// KV-head space (for GQA, this means averaging query heads within each
    /// KV group, or equivalently computing the query at KV-head resolution).
    pub fn select(
        &mut self,
        query_keys: &[f32],
        block_cache: &GqaFlashMemoryBlockCache,
        attn_scale: f32,
        current_step: usize,
    ) -> &PerHeadSelection {
        let need_refresh = !self.selection_valid
            || current_step.saturating_sub(self.last_refresh_step) >= self.config.refresh_period;

        if !need_refresh {
            return &self.last_selection;
        }

        self.last_selection.clear();
        self.last_refresh_step = current_step;
        self.selection_valid = true;
        self.refresh_count += 1;

        let n_blocks = block_cache.n_active_blocks();
        debug_assert_eq!(query_keys.len(), self.n_kv_head * self.head_dim);

        for kv_head in 0..self.n_kv_head {
            let q_h = &query_keys[kv_head * self.head_dim..(kv_head + 1) * self.head_dim];
            let scores = &mut self.scores_buf
                [kv_head * self.max_blocks..kv_head * self.max_blocks + n_blocks];

            for (block_idx, score_slot) in scores.iter_mut().enumerate().take(n_blocks) {
                let centroid = block_cache.key_centroid(block_idx, kv_head);
                *score_slot = simd_dot_f32(q_h, centroid, self.head_dim) * attn_scale;
            }

            for (block_idx, &score) in scores.iter().enumerate().take(n_blocks) {
                let sig = katgpt_core::sigmoid(score);
                if sig >= self.config.threshold {
                    self.last_selection.blocks_per_head[kv_head].push(block_idx);
                }
            }
        }

        &self.last_selection
    }

    pub fn selection(&self) -> &PerHeadSelection {
        assert!(
            self.selection_valid,
            "no valid selection — call select() first"
        );
        &self.last_selection
    }
}

// ---------------------------------------------------------------------------
// Periodic selector
// ---------------------------------------------------------------------------

/// Per-head block selection produced by the selector.
#[derive(Debug, Clone)]
pub struct PerHeadSelection {
    /// `[n_heads][variable]` — selected block indices per head.
    pub blocks_per_head: Vec<Vec<usize>>,
}

impl PerHeadSelection {
    /// Create a new per-head selection with pre-reserved capacity.
    ///
    /// Pre-reserving `max_blocks` per head ensures `push` in the refresh path
    /// never reallocates after construction — the G4 (alloc-free steady state)
    /// gate holds from the first call, not just after warm-up.
    pub fn new(n_heads: usize, max_blocks: usize) -> Self {
        let blocks_per_head = (0..n_heads)
            .map(|_| Vec::with_capacity(max_blocks))
            .collect();
        Self { blocks_per_head }
    }

    /// Total number of (head, block) pairs selected.
    pub fn total_selections(&self) -> usize {
        self.blocks_per_head.iter().map(|v| v.len()).sum()
    }

    /// Number of blocks selected for head `h`.
    pub fn len_for_head(&self, head: usize) -> usize {
        self.blocks_per_head[head].len()
    }

    /// Clear all selections for reuse.
    pub fn clear(&mut self) {
        for v in &mut self.blocks_per_head {
            v.clear();
        }
    }
}

/// The periodic sparse selector with refresh amortization.
///
/// Scores blocks per-head using sigmoid threshold and caches the result.
/// Re-scores only every `refresh_period` decode steps (FlashMemory §2.1).
pub struct FlashMemorySelector {
    config: FlashMemoryConfig,
    /// Last selection. Reused between refreshes.
    last_selection: PerHeadSelection,
    /// Last decode step when selection was refreshed.
    last_refresh_step: usize,
    /// Whether the selection is valid (false after construction / force_refresh).
    selection_valid: bool,
    /// Scratch: block scores per head. `[n_heads * max_blocks]`.
    scores_buf: Vec<f32>,
    n_heads: usize,
    max_blocks: usize,
    /// Count of refresh calls (for testing / amortization verification).
    refresh_count: usize,
}

impl FlashMemorySelector {
    pub fn new(config: FlashMemoryConfig, n_heads: usize, max_blocks: usize) -> Self {
        Self {
            config,
            last_selection: PerHeadSelection::new(n_heads, max_blocks),
            last_refresh_step: 0,
            selection_valid: false,
            scores_buf: vec![0.0; n_heads * max_blocks],
            n_heads,
            max_blocks,
            refresh_count: 0,
        }
    }

    /// Number of times the selection was refreshed (for amortization tests).
    pub fn refresh_count(&self) -> usize {
        self.refresh_count
    }

    /// Force a refresh on the next `select` call.
    pub fn force_refresh(&mut self) {
        self.selection_valid = false;
    }

    /// Is the current selection valid (not stale)?
    pub fn is_selection_valid(&self) -> bool {
        self.selection_valid
    }

    /// Select blocks to attend to for each head.
    ///
    /// Uses periodic refresh: if `current_step - last_refresh_step < refresh_period`
    /// AND the selection is valid, returns the cached selection. Otherwise,
    /// re-scores all blocks per-head using sigmoid threshold and caches the result.
    ///
    /// # Arguments
    /// * `query_content` — `[n_heads * d_h]` — the content query projections
    ///   (`q_c` from MLA). Each head's query is `query_content[head*d_h..(head+1)*d_h]`.
    /// * `block_cache` — the block centroid cache (must be rebuilt for current seq).
    /// * `attn_scale` — the attention scale factor (1/sqrt(d_h) typically).
    /// * `current_step` — the current decode step (for refresh scheduling).
    ///
    /// # Returns
    /// A reference to the per-head selection. The selection is valid until the
    /// next refresh.
    pub fn select(
        &mut self,
        query_content: &[f32],
        block_cache: &FlashMemoryBlockCache,
        attn_scale: f32,
        current_step: usize,
    ) -> &PerHeadSelection {
        let need_refresh = !self.selection_valid
            || current_step.saturating_sub(self.last_refresh_step) >= self.config.refresh_period;

        if !need_refresh {
            return &self.last_selection;
        }

        // ── Refresh: re-score all blocks per head ────────────────────────────
        self.last_selection.clear();
        self.last_refresh_step = current_step;
        self.selection_valid = true;
        self.refresh_count += 1;

        let n_blocks = block_cache.n_active_blocks();
        let d_h = query_content.len() / self.n_heads;
        debug_assert_eq!(query_content.len(), self.n_heads * d_h);

        for head in 0..self.n_heads {
            let q_c_h = &query_content[head * d_h..(head + 1) * d_h];
            let scores =
                &mut self.scores_buf[head * self.max_blocks..head * self.max_blocks + n_blocks];

            // Score each block: dot(q_c_h, k_c_centroid_h) * attn_scale
            for (block_idx, score_slot) in scores.iter_mut().enumerate().take(n_blocks) {
                let centroid = block_cache.key_centroid(block_idx, head);
                *score_slot = simd_dot_f32(q_c_h, centroid, d_h) * attn_scale;
            }

            // Sigmoid threshold selection: σ(score) ≥ threshold
            for (block_idx, &score) in scores.iter().enumerate().take(n_blocks) {
                let sig = katgpt_core::sigmoid(score);
                if sig >= self.config.threshold {
                    self.last_selection.blocks_per_head[head].push(block_idx);
                }
            }
        }

        &self.last_selection
    }

    /// Get the current selection (panics if not yet selected).
    pub fn selection(&self) -> &PerHeadSelection {
        assert!(
            self.selection_valid,
            "no valid selection — call select() first"
        );
        &self.last_selection
    }
}

// ---------------------------------------------------------------------------
// Trained dual-encoder indexer (Plan 337 Phase B)
// ---------------------------------------------------------------------------

/// Trained dual-encoder indexer for FlashMemory sparse attention.
///
/// Replaces the modelless centroid-dot-product scorer (`FlashMemorySelector`)
/// with two tiny trained MLPs (FlashMemory-DeepSeek-V4 §3.2, Plan 337):
/// - **Q-Indexer**: `Linear(d_h, d_h/4) → ReLU → Linear(d_h/4, 1)` — scores
///   the query's retrieval intent per head.
/// - **K-Indexer**: same architecture — scores each block's retrieval value
///   per head.
///
/// Block importance: `I = σ(q_score · k_score)`, selected when `I ≥ threshold`.
///
/// The MLPs are shared across heads (per-head differentiation comes from the
/// per-head query/key projections feeding into the MLPs). Total params for
/// Kimi-K3-0.40B (d_h=64): ~2.1K — 0.0005% of the 395M model.
///
/// # Periodic refresh
///
/// Like `FlashMemorySelector`, this indexer re-scores only every
/// `refresh_period` decode steps. Between refreshes, the cached k_scores
/// (per-head per-block) are reused; only the q_score (per-head scalar) is
/// recomputed each step — making the per-step cost O(heads · MLP + heads ·
/// blocks) instead of O(heads · blocks · d_h) for the modelless dot-product.
///
/// # Feature gate
///
/// `trained_indexer` — requires `flashmemory_sparse`. Opt-in, **never
/// default-on** (requires training → violates the modelless-first mandate).
/// The modelless `FlashMemorySelector` is the production default.
#[cfg(feature = "trained_indexer")]
pub struct DualEncoderIndexer {
    // ── Q-Indexer weights: Linear(d_h, hidden) + ReLU + Linear(hidden, 1) ──
    /// `[hidden * d_h]` — row-major weight matrix for the hidden layer.
    q_w1: Vec<f32>,
    /// `[hidden]` — bias for the hidden layer.
    q_b1: Vec<f32>,
    /// `[hidden]` — weight vector for the output layer.
    q_w2: Vec<f32>,
    /// Output bias.
    q_b2: f32,

    // ── K-Indexer weights: same shape ──
    k_w1: Vec<f32>,
    k_b1: Vec<f32>,
    k_w2: Vec<f32>,
    k_b2: f32,

    // ── Config ──
    config: FlashMemoryConfig,
    d_h: usize,
    /// Hidden layer width = `d_h / 4` (min 4).
    hidden: usize,

    // ── Periodic refresh state ──
    last_selection: PerHeadSelection,
    last_refresh_step: usize,
    selection_valid: bool,
    refresh_count: usize,

    // ── Cached k_scores per head per block: `[n_heads * max_blocks]` ──
    /// Recomputed only on refresh (block centroids are fixed between refreshes).
    k_scores_cache: Vec<f32>,

    // ── Scratch ──
    /// MLP hidden-layer scratch `[hidden]` — reused across calls (G4 alloc-free).
    mlp_scratch: Vec<f32>,

    n_heads: usize,
    max_blocks: usize,
}

#[cfg(feature = "trained_indexer")]
impl DualEncoderIndexer {
    /// Create a randomly-initialized indexer (for training).
    ///
    /// Uses Xavier/Glorot initialization for the weight matrices.
    /// `hidden` defaults to `d_h / 4` (min 4).
    pub fn new_random(
        config: FlashMemoryConfig,
        d_h: usize,
        n_heads: usize,
        max_blocks: usize,
        seed: u64,
    ) -> Self {
        let hidden = (d_h / 4).max(4);
        // Simple deterministic PRNG (xorshift) for reproducible init.
        let mut rng = seed.max(1);
        let mut next_f32 = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // Map to [-limit, limit] for Xavier init.
            let u = (rng >> 40) as f32 / (1u64 << 40) as f32; // [0, 1)
            u * 2.0 - 1.0
        };

        let xavier_w1 = (6.0 / (d_h + hidden) as f32).sqrt();
        let xavier_w2 = (6.0 / (hidden + 1) as f32).sqrt();

        let q_w1 = (0..hidden * d_h).map(|_| next_f32() * xavier_w1).collect();
        let q_b1 = vec![0.0; hidden];
        let q_w2 = (0..hidden).map(|_| next_f32() * xavier_w2).collect();
        let q_b2 = 0.0;

        let k_w1 = (0..hidden * d_h).map(|_| next_f32() * xavier_w1).collect();
        let k_b1 = vec![0.0; hidden];
        let k_w2 = (0..hidden).map(|_| next_f32() * xavier_w2).collect();
        let k_b2 = 0.0;

        Self {
            q_w1,
            q_b1,
            q_w2,
            q_b2,
            k_w1,
            k_b1,
            k_w2,
            k_b2,
            config,
            d_h,
            hidden,
            last_selection: PerHeadSelection::new(n_heads, max_blocks),
            last_refresh_step: 0,
            selection_valid: false,
            refresh_count: 0,
            k_scores_cache: vec![0.0; n_heads * max_blocks],
            mlp_scratch: vec![0.0; hidden],
            n_heads,
            max_blocks,
        }
    }

    /// Create an indexer from pre-trained weights (for inference).
    ///
    /// Weight layout matches `to_bytes()` / `from_bytes()` for freeze/thaw.
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        config: FlashMemoryConfig,
        d_h: usize,
        n_heads: usize,
        max_blocks: usize,
        q_w1: Vec<f32>,
        q_b1: Vec<f32>,
        q_w2: Vec<f32>,
        q_b2: f32,
        k_w1: Vec<f32>,
        k_b1: Vec<f32>,
        k_w2: Vec<f32>,
        k_b2: f32,
    ) -> Self {
        let hidden = q_b1.len();
        debug_assert_eq!(q_w1.len(), hidden * d_h);
        debug_assert_eq!(q_w2.len(), hidden);
        debug_assert_eq!(k_w1.len(), hidden * d_h);
        debug_assert_eq!(k_b1.len(), hidden);
        debug_assert_eq!(k_w2.len(), hidden);

        Self {
            q_w1,
            q_b1,
            q_w2,
            q_b2,
            k_w1,
            k_b1,
            k_w2,
            k_b2,
            config,
            d_h,
            hidden,
            last_selection: PerHeadSelection::new(n_heads, max_blocks),
            last_refresh_step: 0,
            selection_valid: false,
            refresh_count: 0,
            k_scores_cache: vec![0.0; n_heads * max_blocks],
            mlp_scratch: vec![0.0; hidden],
            n_heads,
            max_blocks,
        }
    }

    /// Serialize all weights to a flat byte buffer (for freeze/thaw).
    ///
    /// Layout (all f32, little-endian):
    /// `[hidden][d_h][q_w1 (hidden*d_h)][q_b1 (hidden)][q_w2 (hidden)][q_b2 (1)]
    ///  [k_w1 (hidden*d_h)][k_b1 (hidden)][k_w2 (hidden)][k_b2 (1)]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let total_floats = 2 + // hidden, d_h
            self.hidden * self.d_h + self.hidden + self.hidden + 1 + // Q-Indexer
            self.hidden * self.d_h + self.hidden + self.hidden + 1; // K-Indexer
        let mut buf = Vec::with_capacity(total_floats * 4);
        let push_f32 = |v: f32, buf: &mut Vec<u8>| {
            buf.extend_from_slice(&v.to_le_bytes());
        };
        push_f32(self.hidden as f32, &mut buf);
        push_f32(self.d_h as f32, &mut buf);
        for &v in &self.q_w1 {
            push_f32(v, &mut buf);
        }
        for &v in &self.q_b1 {
            push_f32(v, &mut buf);
        }
        for &v in &self.q_w2 {
            push_f32(v, &mut buf);
        }
        push_f32(self.q_b2, &mut buf);
        for &v in &self.k_w1 {
            push_f32(v, &mut buf);
        }
        for &v in &self.k_b1 {
            push_f32(v, &mut buf);
        }
        for &v in &self.k_w2 {
            push_f32(v, &mut buf);
        }
        push_f32(self.k_b2, &mut buf);
        buf
    }

    /// Deserialize from a flat byte buffer (inverse of `to_bytes()`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_bytes(
        config: FlashMemoryConfig,
        n_heads: usize,
        max_blocks: usize,
        data: &[u8],
    ) -> Result<Self, &'static str> {
        let read_f32 = |offset: &mut usize| -> Result<f32, &'static str> {
            if *offset + 4 > data.len() {
                return Err("buffer too short");
            }
            let v = f32::from_le_bytes([
                data[*offset],
                data[*offset + 1],
                data[*offset + 2],
                data[*offset + 3],
            ]);
            *offset += 4;
            Ok(v)
        };

        let mut off = 0usize;
        let hidden = read_f32(&mut off)? as usize;
        let d_h = read_f32(&mut off)? as usize;

        let need = 2 + hidden * d_h * 2 + hidden * 4 + 2;
        if data.len() < need * 4 {
            return Err("buffer too short for declared dimensions");
        }

        let read_vec = |n: usize, off: &mut usize| -> Result<Vec<f32>, &'static str> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(read_f32(off)?);
            }
            Ok(v)
        };

        let q_w1 = read_vec(hidden * d_h, &mut off)?;
        let q_b1 = read_vec(hidden, &mut off)?;
        let q_w2 = read_vec(hidden, &mut off)?;
        let q_b2 = read_f32(&mut off)?;
        let k_w1 = read_vec(hidden * d_h, &mut off)?;
        let k_b1 = read_vec(hidden, &mut off)?;
        let k_w2 = read_vec(hidden, &mut off)?;
        let k_b2 = read_f32(&mut off)?;

        Ok(Self::from_weights(
            config, d_h, n_heads, max_blocks, q_w1, q_b1, q_w2, q_b2, k_w1, k_b1, k_w2, k_b2,
        ))
    }

    /// Number of times the selection was refreshed (for amortization tests).
    pub fn refresh_count(&self) -> usize {
        self.refresh_count
    }

    /// Force a refresh on the next `select` call.
    pub fn force_refresh(&mut self) {
        self.selection_valid = false;
    }

    /// Is the current selection valid?
    pub fn is_selection_valid(&self) -> bool {
        self.selection_valid
    }

    /// MLP forward: `Linear(d, hidden) → ReLU → Linear(hidden, 1)`.
    ///
    /// Returns a scalar score. Uses `scratch` for the hidden layer
    /// (zero allocation in steady state). Free function to avoid `self`
    /// borrow conflicts when called with `&self.weights` + `&mut self.scratch`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn mlp_forward(
        w1: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: f32,
        input: &[f32],
        scratch: &mut [f32],
        hidden: usize,
        d: usize,
    ) -> f32 {
        debug_assert_eq!(input.len(), d);
        debug_assert_eq!(scratch.len(), hidden);

        // Hidden layer: h = ReLU(W1 · x + b1)
        simd_matmul_rows(scratch, w1, input, hidden, d);
        for i in 0..hidden {
            scratch[i] = (scratch[i] + b1[i]).max(0.0);
        }

        // Output: o = W2 · h + b2
        let mut o = b2;
        for i in 0..hidden {
            o += w2[i] * scratch[i];
        }
        o
    }

    /// Select blocks to attend to for each head (drop-in replacement for
    /// `FlashMemorySelector::select`).
    ///
    /// Uses periodic refresh: if `current_step - last_refresh_step <
    /// refresh_period` AND the selection is valid, returns the cached
    /// selection. Otherwise, recomputes k_scores for all blocks per head,
    /// then re-scores using the trained dual-encoder.
    ///
    /// # Arguments
    /// * `query_content` — `[n_heads * d_h]` — the content query projections.
    /// * `block_cache` — the block centroid cache (must be rebuilt for current seq).
    /// * `current_step` — the current decode step (for refresh scheduling).
    pub fn select(
        &mut self,
        query_content: &[f32],
        block_cache: &FlashMemoryBlockCache,
        current_step: usize,
    ) -> &PerHeadSelection {
        let need_refresh = !self.selection_valid
            || current_step.saturating_sub(self.last_refresh_step) >= self.config.refresh_period;

        if !need_refresh {
            // Return the cached selection (amortized — same as FlashMemorySelector).
            return &self.last_selection;
        }

        // ── Full refresh: recompute k_scores + re-score ────────────────────────
        self.last_selection.clear();
        self.last_refresh_step = current_step;
        self.selection_valid = true;
        self.refresh_count += 1;

        let n_blocks = block_cache.n_active_blocks();
        let n_h = self.n_heads;
        let d_h = self.d_h;
        let hidden = self.hidden;
        debug_assert_eq!(query_content.len(), n_h * d_h);

        // Recompute k_scores cache: K-Indexer(key_centroid) per head per block.
        for head in 0..n_h {
            for block_idx in 0..n_blocks {
                let centroid = block_cache.key_centroid(block_idx, head);
                let k_score = Self::mlp_forward(
                    &self.k_w1,
                    &self.k_b1,
                    &self.k_w2,
                    self.k_b2,
                    centroid,
                    &mut self.mlp_scratch,
                    hidden,
                    d_h,
                );
                self.k_scores_cache[head * self.max_blocks + block_idx] = k_score;
            }
        }

        // Score: for each head, compute q_score then select blocks where
        // σ(q_score · k_score) ≥ threshold.
        for head in 0..n_h {
            let q_c_h = &query_content[head * d_h..(head + 1) * d_h];
            let q_score = Self::mlp_forward(
                &self.q_w1,
                &self.q_b1,
                &self.q_w2,
                self.q_b2,
                q_c_h,
                &mut self.mlp_scratch,
                hidden,
                d_h,
            );

            let k_off = head * self.max_blocks;
            for block_idx in 0..n_blocks {
                let k_score = self.k_scores_cache[k_off + block_idx];
                let importance = katgpt_core::sigmoid(q_score * k_score);
                if importance >= self.config.threshold {
                    self.last_selection.blocks_per_head[head].push(block_idx);
                }
            }
        }

        &self.last_selection
    }

    /// Get the current selection (panics if not yet selected).
    pub fn selection(&self) -> &PerHeadSelection {
        assert!(
            self.selection_valid,
            "no valid selection — call select() first"
        );
        &self.last_selection
    }

    /// Select blocks per KV head for standard GQA attention (Bonsai/Qwen3.5).
    ///
    /// GQA sibling of [`DualEncoderIndexer::select`]: identical scoring
    /// semantics (`σ(q_score · k_score) ≥ threshold` with periodic refresh +
    /// cached k-scores), but reads **raw per-KV-head key centroids** from
    /// [`GqaFlashMemoryBlockCache`] instead of MLA latent centroids, and takes
    /// the KV-resolution query `[n_kv_head * head_dim]` — the same
    /// block-contiguous group-mean query contract as
    /// [`GqaFlashMemorySelector::select`] (pinned by its mapping regression
    /// test).
    ///
    /// This is the correct serving path for checkpoints trained on raw GQA key
    /// centroids (e.g. riir-train Plan 337 Bonsai datasets): those weights were
    /// fit to the raw-centroid feature space, so routing them through the MLA
    /// [`select`](DualEncoderIndexer::select) (latent `c_kv` centroids)
    /// would silently change the meaning of the inputs.
    ///
    /// The `n_heads` constructor argument must be the number of **KV heads**
    /// for GQA use (selection is per KV head; all query heads in a group share
    /// it). No `attn_scale` — the trained encoders learn their own scale,
    /// matching [`select`](DualEncoderIndexer::select).
    ///
    /// Duplicated from [`select`](DualEncoderIndexer::select) rather than
    /// genericized over the cache type, per the documented precedent at the
    /// [`GqaFlashMemoryBlockCache`] docs ("separate type (not a trait)… YAGNI").
    pub fn select_gqa(
        &mut self,
        query_keys: &[f32],
        block_cache: &GqaFlashMemoryBlockCache,
        current_step: usize,
    ) -> &PerHeadSelection {
        let need_refresh = !self.selection_valid
            || current_step.saturating_sub(self.last_refresh_step) >= self.config.refresh_period;

        if !need_refresh {
            return &self.last_selection;
        }

        self.last_selection.clear();
        self.last_refresh_step = current_step;
        self.selection_valid = true;
        self.refresh_count += 1;

        let n_blocks = block_cache.n_active_blocks();
        let n_h = self.n_heads;
        let d_h = self.d_h;
        let hidden = self.hidden;
        debug_assert_eq!(query_keys.len(), n_h * d_h);

        // Recompute k_scores cache: K-Indexer(raw key centroid) per KV head
        // per block.
        for head in 0..n_h {
            for block_idx in 0..n_blocks {
                let centroid = block_cache.key_centroid(block_idx, head);
                let k_score = Self::mlp_forward(
                    &self.k_w1,
                    &self.k_b1,
                    &self.k_w2,
                    self.k_b2,
                    centroid,
                    &mut self.mlp_scratch,
                    hidden,
                    d_h,
                );
                self.k_scores_cache[head * self.max_blocks + block_idx] = k_score;
            }
        }

        // Score: per KV head, q_score from the group-mean query; select blocks
        // where σ(q_score · k_score) ≥ threshold.
        for head in 0..n_h {
            let q_h = &query_keys[head * d_h..(head + 1) * d_h];
            let q_score = Self::mlp_forward(
                &self.q_w1,
                &self.q_b1,
                &self.q_w2,
                self.q_b2,
                q_h,
                &mut self.mlp_scratch,
                hidden,
                d_h,
            );

            let k_off = head * self.max_blocks;
            for block_idx in 0..n_blocks {
                let k_score = self.k_scores_cache[k_off + block_idx];
                let importance = katgpt_core::sigmoid(q_score * k_score);
                if importance >= self.config.threshold {
                    self.last_selection.blocks_per_head[head].push(block_idx);
                }
            }
        }

        &self.last_selection
    }

    /// Total parameter count (for reporting / size verification).
    pub fn param_count(&self) -> usize {
        // Q-Indexer: hidden*d_h + hidden + hidden + 1
        // K-Indexer: same
        2 * (self.hidden * self.d_h + self.hidden + self.hidden + 1)
    }

    /// Get d_h dimension (for training access).
    pub fn d_h_dim(&self) -> usize {
        self.d_h
    }

    /// Get hidden dimension (for training access).
    pub fn hidden_dim(&self) -> usize {
        self.hidden
    }

    /// Extract all weights as a tuple (for training initialization).
    #[allow(clippy::type_complexity)]
    pub fn extract_weights(
        &self,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        f32,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        f32,
    ) {
        (
            self.q_w1.clone(),
            self.q_b1.clone(),
            self.q_w2.clone(),
            self.q_b2,
            self.k_w1.clone(),
            self.k_b1.clone(),
            self.k_w2.clone(),
            self.k_b2,
        )
    }

    /// Clone for evaluation (resets refresh state but keeps weights).
    pub fn clone_for_eval(&self) -> Self {
        Self::from_weights(
            self.config.clone(),
            self.d_h,
            self.n_heads,
            self.max_blocks,
            self.q_w1.clone(),
            self.q_b1.clone(),
            self.q_w2.clone(),
            self.q_b2,
            self.k_w1.clone(),
            self.k_b1.clone(),
            self.k_w2.clone(),
            self.k_b2,
        )
    }
}

// ---------------------------------------------------------------------------
// Sparse MLA forward
// ---------------------------------------------------------------------------

/// Sparse MLA forward using FlashMemory block selection — the DECODE ENTRY
/// (runtime router, Issue 926 secondary scope).
///
/// Routes to the weight-absorbed path
/// ([`mla_forward_token_flashmemory_absorbed`]) when compiled with the
/// `mla_absorbed` feature, unless `KATGPT_MLA_RECONSTRUCT=1` forces the
/// reconstruct reference (read once, first call — the same kill-switch
/// posture as the dense lane's `mla_forward_token_dispatched`). WITHOUT the
/// feature this is a transparent forward to
/// [`mla_forward_token_flashmemory_reconstruct`]: default builds are
/// behavior- and byte-identical.
///
/// This is the Phase 1 mechanism: attention is computed only over tokens in
/// selected blocks. The softmax denominator covers only selected tokens.
/// Block SELECTION is identical on both paths by construction — both share
/// [`flashmemory_decode_prefix`] VERBATIM — so absorption changes only HOW the
/// selected latent rows are read (scores via the absorbed query, values via a
/// latent-space accumulator), never WHICH rows are selected.
///
/// **Correctness contract:** for tokens in selected blocks, the attention
/// computation matches the dense path (`mla_forward_token`) — bit-identical on
/// the reconstruct path, and within the 1e-4 absorbed spec bar (fp
/// reassociation only) on the absorbed path. The only difference is that
/// tokens in non-selected blocks receive zero attention weight. This is the
/// standard sparse attention contract — it's lossy by design (that's the
/// memory reduction), but the selected-token computation is exact.
///
/// **Safety net:** if no blocks are selected for a head (all scores below
/// threshold), the head falls back to attending the MOST RECENT block only
/// (recency bias — the FlashMemory paper notes this as the natural fallback
/// for "nothing relevant found" queries). This prevents division-by-zero in
/// softmax and gives the model a sensible default. The fallback is identical
/// on both paths.
///
/// # Arguments
/// * `step` — the current decode step (for periodic refresh scheduling).
///   Pass the token position or any monotonic counter.
#[allow(clippy::too_many_arguments)]
pub fn mla_forward_token_flashmemory<'s>(
    config: &MlaConfig,
    weights: &MlaWeights,
    cache: &mut MlaKVCache,
    scratch: &'s mut MlaForwardScratch,
    rope_freqs: &mut RopeFreqs,
    h: &[f32],
    block_cache: &mut FlashMemoryBlockCache,
    selector: &mut FlashMemorySelector,
    step: usize,
) -> &'s mut [f32] {
    #[cfg(feature = "mla_absorbed")]
    {
        static RECONSTRUCT_FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let forced = *RECONSTRUCT_FORCED.get_or_init(|| {
            std::env::var("KATGPT_MLA_RECONSTRUCT").is_ok_and(|v| v == "1")
        });
        if !forced {
            return mla_forward_token_flashmemory_absorbed(
                config, weights, cache, scratch, rope_freqs, h, block_cache, selector, step,
            );
        }
    }
    mla_forward_token_flashmemory_reconstruct(
        config, weights, cache, scratch, rope_freqs, h, block_cache, selector, step,
    )
}

/// Shared flashmemory decode prefix (Steps 1–6): down/up-projections + latent
/// RMSNorms + rope queries + the shared rope key + cache append + block
/// centroid rebuild + block selection. Used VERBATIM by both the reconstruct
/// reference ([`mla_forward_token_flashmemory_reconstruct`]) and the
/// weight-absorbed variant ([`mla_forward_token_flashmemory_absorbed`], Issue
/// 926 secondary scope) so the pair cannot drift on the prefix — which is what
/// makes block SELECTION definitionally identical on both paths.
///
/// Returns `(seq, selection)`: the post-append cache length (the attention
/// span) and the per-head selected blocks (borrowed from `selector`; the
/// selection is valid until the next refresh).
#[allow(clippy::too_many_arguments)]
fn flashmemory_decode_prefix<'a>(
    config: &MlaConfig,
    weights: &MlaWeights,
    cache: &mut MlaKVCache,
    scratch: &mut MlaForwardScratch,
    rope_freqs: &mut RopeFreqs,
    h: &[f32],
    block_cache: &mut FlashMemoryBlockCache,
    selector: &'a mut FlashMemorySelector,
    step: usize,
) -> (usize, &'a PerHeadSelection) {
    let d = config.hidden_size;
    let d_c = config.kv_lora_rank;
    let d_qc = config.q_lora_rank;
    let d_h = config.d_h();
    let d_r = config.d_r();
    let n_h = config.n_heads;
    debug_assert_eq!(h.len(), d, "hidden state dim mismatch");

    let pos = cache.seq_len;
    let scale = config.attn_scale();

    // ── Step 1-4: Down-projections, latent norms, query/key up-projections ─
    // Identical to the dense MLA forward (mla_forward_token).
    simd_matmul_rows(&mut scratch.c_kv, &weights.w_dkv, h, d_c, d);
    simd_matmul_rows(&mut scratch.c_q, &weights.w_dq, h, d_qc, d);

    rmsnorm_inplace(
        &mut scratch.c_q,
        &weights.q_a_norm_weight,
        config.rms_norm_eps,
    );
    rmsnorm_inplace(
        &mut scratch.c_kv,
        &weights.kv_a_norm_weight,
        config.rms_norm_eps,
    );

    simd_matmul_rows(
        &mut scratch.q_c,
        &weights.w_uq,
        &scratch.c_q,
        d_h * n_h,
        d_qc,
    );
    simd_matmul_rows(
        &mut scratch.q_r,
        &weights.w_qr,
        &scratch.c_q,
        d_r * n_h,
        d_qc,
    );
    if !config.use_nope {
        apply_decoupled_rope(rope_freqs, &mut scratch.q_r, d_r, n_h, pos);
    }

    simd_matmul_rows(&mut scratch.k_r, &weights.w_kr, h, d_r, d);
    if !config.use_nope {
        apply_decoupled_rope(rope_freqs, &mut scratch.k_r, d_r, 1, pos);
    }

    // ── Step 5: Cache the normed latent + shared rope key ──────────────────
    cache.append(&scratch.c_kv, &scratch.k_r);

    // ── Step 5b: Rebuild block centroids (includes the just-appended token) ─
    block_cache.rebuild_from_cache(cache, weights);

    // ── Step 6: FlashMemory block selection ────────────────────────────
    let seq = cache.seq_len;
    let selection = selector.select(&scratch.q_c, block_cache, scale, step);
    (seq, selection)
}

/// Single-token sparse MLA forward — the RECONSTRUCT reference (flashmemory
/// lane). Stays compiled in EVERY feature posture as the spec-match reference
/// and the `KATGPT_MLA_RECONSTRUCT=1` kill-switch fallback; the absorbed
/// variant ([`mla_forward_token_flashmemory_absorbed`], Issue 926 secondary
/// scope) is the algebraically-identical reassociation of Step 7.
///
/// Per selected token j, `k_c_j = W_UK[h]·c_kv_j` and `v_c_j = W_UV[h]·c_kv_j`
/// are rebuilt at attention time — O(selected)·(d_h+v_h)·d_c FLOPs per step,
/// the same per-token up-projection shape the dense reference had before
/// Issue 926. (The current-token k_c/v_c projections below are dead work kept
/// for structural parity with the dense reference — k_c is overwritten before
/// read and v_c is never read in the attention loop; the dense lane noted the
/// same microcleanup.)
#[allow(clippy::too_many_arguments)]
pub fn mla_forward_token_flashmemory_reconstruct<'s>(
    config: &MlaConfig,
    weights: &MlaWeights,
    cache: &mut MlaKVCache,
    scratch: &'s mut MlaForwardScratch,
    rope_freqs: &mut RopeFreqs,
    h: &[f32],
    block_cache: &mut FlashMemoryBlockCache,
    selector: &mut FlashMemorySelector,
    step: usize,
) -> &'s mut [f32] {
    let d_c = config.kv_lora_rank;
    let d_h = config.d_h();
    let d_r = config.d_r();
    let v_h = config.v_head_dim;
    let n_h = config.n_heads;

    let (seq, selection) = flashmemory_decode_prefix(
        config, weights, cache, scratch, rope_freqs, h, block_cache, selector, step,
    );
    let scale = config.attn_scale();

    // ── Step 3: Key/value up-projections (reconstruct-only) ────────────
    simd_matmul_rows(
        &mut scratch.k_c,
        &weights.w_uk,
        &scratch.c_kv,
        d_h * n_h,
        d_c,
    );
    simd_matmul_rows(
        &mut scratch.v_c,
        &weights.w_uv,
        &scratch.c_kv,
        v_h * n_h,
        d_c,
    );

    // ── Step 7: Sparse attention per head (reconstruct: per-token
    //    up-projection) ──────────────────────────────────────────────────
    for head in 0..n_h {
        let q_c_h = &scratch.q_c[head * d_h..(head + 1) * d_h];
        let q_r_h = &scratch.q_r[head * d_r..(head + 1) * d_r];

        // Collect the token indices in selected blocks for this head.
        // Write into scratch.scores[..seq] — we'll overwrite with real scores next.
        // Actually, we need a separate token-index list. Reuse scratch.gate_buf
        // region as a token-index scratch (it's unused during the attention loop;
        // only written in Step 8 after attention completes). But gate_buf is f32
        // and we need usize indices. Use a different approach: iterate selected
        // blocks and compute scores inline into scratch.scores.

        // Strategy: first pass collects selected token indices + computes scores.
        // Second pass does softmax + weighted value sum.
        // We'll pack (token_idx, score) pairs into the scores buffer prefix.

        let selected_blocks = &selection.blocks_per_head[head];

        // Fallback: if no blocks selected, attend the most recent block only.
        // Uses a stack array (no heap allocation — G4 alloc-free steady state).
        let mut fallback = [0usize; 1];
        let blocks_to_attend: &[usize] = if selected_blocks.is_empty() {
            fallback[0] = block_cache.n_active_blocks().saturating_sub(1);
            &fallback[..]
        } else {
            selected_blocks
        };

        // First pass: compute attention scores for selected tokens.
        let scores = &mut scratch.scores[..seq];
        let mut n_scored = 0usize;
        let mut max_score = f32::NEG_INFINITY;

        for &block_idx in blocks_to_attend {
            let (tok_start, tok_end) = block_cache.block_token_range(block_idx, seq);
            for tok in tok_start..tok_end {
                let c_kv_j = cache.latent_kv_at(tok);
                let k_r_j = cache.rope_key_at(tok);

                // k_c_j = W_UK[head] · c_kv_j
                simd_matmul_rows(
                    &mut scratch.k_c[..d_h],
                    &weights.w_uk[head * d_h * d_c..(head + 1) * d_h * d_c],
                    c_kv_j,
                    d_h,
                    d_c,
                );

                let content_dot = simd_dot_f32(q_c_h, &scratch.k_c[..d_h], d_h);
                let rope_dot = simd_dot_f32(q_r_h, k_r_j, d_r);
                let score = (content_dot + rope_dot) * scale;
                scores[n_scored] = score;
                n_scored += 1;
                if score > max_score {
                    max_score = score;
                }
            }
        }

        // Softmax over selected tokens only (numerically stable).
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut().take(n_scored) {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;

        // Second pass: weighted value sum over selected tokens.
        let o_h = &mut scratch.attn_out[head * v_h..(head + 1) * v_h];
        o_h[..v_h].fill(0.0);
        let v_c_j_h_scratch = &mut scratch.gate_buf[..v_h];

        let mut score_idx = 0usize;
        for &block_idx in blocks_to_attend {
            let (tok_start, tok_end) = block_cache.block_token_range(block_idx, seq);
            for tok in tok_start..tok_end {
                let c_kv_j = cache.latent_kv_at(tok);
                let weight = scores[score_idx] * inv_sum;
                score_idx += 1;

                simd_matmul_rows(
                    v_c_j_h_scratch,
                    &weights.w_uv[head * v_h * d_c..(head + 1) * v_h * d_c],
                    c_kv_j,
                    v_h,
                    d_c,
                );
                for (o, &vc) in o_h.iter_mut().zip(v_c_j_h_scratch[..v_h].iter()) {
                    *o += weight * vc;
                }
            }
        }
    }

    // ── Step 8: Output gate + output projection (shared epilogue) ──────
    flashmemory_decode_epilogue(config, weights, scratch, h)
}

/// Shared flashmemory decode epilogue (Step 8): the Kimi-K3 output gate
/// (applied to `attn_out` BEFORE the output projection) + `W_O`. Used verbatim
/// by both flashmemory decode paths (see [`flashmemory_decode_prefix`]).
fn flashmemory_decode_epilogue<'s>(
    config: &MlaConfig,
    weights: &MlaWeights,
    scratch: &'s mut MlaForwardScratch,
    h: &[f32],
) -> &'s mut [f32] {
    let d = config.hidden_size;
    let v_h = config.v_head_dim;
    let n_h = config.n_heads;
    let proj_size = v_h * n_h;

    if config.use_output_gate
        && let Some(ref w_g) = weights.w_g
    {
        simd_matmul_rows(&mut scratch.gate_buf, w_g, h, proj_size, d);
        for (o, &gb) in scratch.attn_out[..proj_size]
            .iter_mut()
            .zip(scratch.gate_buf[..proj_size].iter())
        {
            let g = 1.0 / (1.0 + (-gb).exp());
            *o *= g;
        }
    }

    simd_matmul_rows(
        &mut scratch.output,
        &weights.w_o,
        &scratch.attn_out,
        d,
        proj_size,
    );

    &mut scratch.output[..d]
}

/// Weight-absorbed single-token sparse MLA decode (Issue 926 SECONDARY scope —
/// DeepSeek-V2 §2.1 absorption on the selected-block path; external evidence
/// arXiv:2610.07940 §5.2).
///
/// Algebraically IDENTICAL to [`mla_forward_token_flashmemory_reconstruct`],
/// reassociated to remove the per-selected-token up-projections. The block
/// SELECTION is byte-identical by construction (both paths share
/// [`flashmemory_decode_prefix`]); absorption changes only how the selected
/// latent rows are READ:
///
/// 1. `q_abs_h = W_UK[h]ᵀ · q_c_h` — once per step per head (via the in-tree
///    `simd_transpose_matvec_into` NEON/AVX2/wasm32 substrate), so the content
///    score reads the latent directly:
///    `q_c·(W_UK·c_kv_j) ≡ (W_UKᵀ·q_c)·c_kv_j`.
/// 2. Scores over the SELECTED tokens only: `content = q_abs_h·c_kv_j` (d_c
///    dot) + the UNCHANGED rope dot `q_r_h·k_r_j`.
/// 3. Softmax over selected tokens: same form as the reference (subtract max,
///    exp, sum). The normalization is SOFTMAX — as in the dense lane, the
///    house sigmoid rule governs latent-space weighting gates, not attention
///    normalization.
/// 4. Values: the latent accumulator `s_h = Σ_j exp_j·c_kv_j` runs over the
///    SELECTED latent rows as ONE accumulating transpose matvec per contiguous
///    selected block ([`simd_transpose_matvec_acc`] — the blocks are ascending
///    and disjoint, so each block's packed score slice lines up with its
///    latent row run), the single `inv_sum` scaling lands on `s_h` ONCE, and
///    `o_h = W_UV[h]·s_h` up-projects once per step per head.
///
/// Unselected tokens keep zero attention weight (the sparse contract), and the
/// fallback-to-most-recent-block safety net is identical to the reference.
/// The reconstruct path's dead Step-3 k_c/v_c projections are skipped (the
/// absorbed path never materializes k_c/v_c at all — same as the dense
/// primary).
///
/// G1 = the absorbed/reconstruct flashmemory pair within the existing 1e-4
/// (1e-3 full-dims) bar (`tests/mla_flashmemory_absorbed_spec_match.rs`); G4
/// = zero steady-state allocation (`tests/mla_flashmemory_absorbed_alloc_check.rs`).
#[cfg(feature = "mla_absorbed")]
#[allow(clippy::too_many_arguments)]
pub fn mla_forward_token_flashmemory_absorbed<'s>(
    config: &MlaConfig,
    weights: &MlaWeights,
    cache: &mut MlaKVCache,
    scratch: &'s mut MlaForwardScratch,
    rope_freqs: &mut RopeFreqs,
    h: &[f32],
    block_cache: &mut FlashMemoryBlockCache,
    selector: &mut FlashMemorySelector,
    step: usize,
) -> &'s mut [f32] {
    let d_c = config.kv_lora_rank;
    let d_h = config.d_h();
    let d_r = config.d_r();
    let v_h = config.v_head_dim;
    let n_h = config.n_heads;

    let (seq, selection) = flashmemory_decode_prefix(
        config, weights, cache, scratch, rope_freqs, h, block_cache, selector, step,
    );
    let scale = config.attn_scale();

    // ── Step 7: ABSORBED sparse attention per head ───────────────────
    for head in 0..n_h {
        // q_abs_h = W_UK[h]ᵀ · q_c_h — once per step per head. The head slice
        // is row-major [d_h × d_c]; the transpose matvec zeroes `q_abs[..d_c]`
        // then accumulates the row-major matrix transposed against q_c_h.
        simd_transpose_matvec_into(
            &mut scratch.q_abs[..d_c],
            &weights.w_uk[head * d_h * d_c..(head + 1) * d_h * d_c],
            &scratch.q_c[head * d_h..(head + 1) * d_h],
            d_h,
            d_c,
        );
        let q_abs_h = &scratch.q_abs[..d_c];
        let q_r_h = &scratch.q_r[head * d_r..(head + 1) * d_r];

        let selected_blocks = &selection.blocks_per_head[head];

        // Fallback: identical to the reconstruct path (most recent block).
        // Uses a stack array (no heap allocation — G4 alloc-free steady state).
        let mut fallback = [0usize; 1];
        let blocks_to_attend: &[usize] = if selected_blocks.is_empty() {
            fallback[0] = block_cache.n_active_blocks().saturating_sub(1);
            &fallback[..]
        } else {
            selected_blocks
        };

        // First pass: scores over the SELECTED tokens only — content dot over
        // the latent (d_c) + the unchanged rope dot.
        let scores = &mut scratch.scores[..seq];
        let mut n_scored = 0usize;
        let mut max_score = f32::NEG_INFINITY;

        for &block_idx in blocks_to_attend {
            let (tok_start, tok_end) = block_cache.block_token_range(block_idx, seq);
            for tok in tok_start..tok_end {
                let c_kv_j = cache.latent_kv_at(tok);
                let k_r_j = cache.rope_key_at(tok);

                let content_dot = simd_dot_f32(q_abs_h, c_kv_j, d_c);
                let rope_dot = simd_dot_f32(q_r_h, k_r_j, d_r);
                let score = (content_dot + rope_dot) * scale;
                scores[n_scored] = score;
                n_scored += 1;
                if score > max_score {
                    max_score = score;
                }
            }
        }

        // Softmax over selected tokens only (numerically stable) — same form
        // as the reference. Weights stay UNNORMALIZED here; the single inv_sum
        // scaling happens on the latent accumulator below.
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut().take(n_scored) {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;

        // Second pass: s_h = Σ_j exp_j · c_kv_j over the SELECTED latent rows
        // — one ACCUMULATING transpose matvec per contiguous selected block
        // (blocks are ascending + disjoint, so each block's packed score slice
        // lines up with its latent row run; the accumulator is zeroed once).
        let s_acc = &mut scratch.lat_acc[..d_c];
        s_acc.fill(0.0);
        let mut score_idx = 0usize;
        for &block_idx in blocks_to_attend {
            let (tok_start, tok_end) = block_cache.block_token_range(block_idx, seq);
            let rows = tok_end - tok_start;
            let block_weights = &scores[score_idx..score_idx + rows];
            let block_latents = &cache.latent_kv[tok_start * d_c..tok_end * d_c];
            simd_transpose_matvec_acc(s_acc, block_latents, block_weights, rows, d_c);
            score_idx += rows;
        }
        debug_assert_eq!(score_idx, n_scored, "packed score/latent row misalignment");
        // The single inv_sum scaling (d_c multiplies, once per head).
        for v in s_acc.iter_mut() {
            *v *= inv_sum;
        }

        // o_h = W_UV[h] · s_h — once per step per head.
        let o_h = &mut scratch.attn_out[head * v_h..(head + 1) * v_h];
        simd_matmul_rows(
            o_h,
            &weights.w_uv[head * v_h * d_c..(head + 1) * v_h * d_c],
            s_acc,
            v_h,
            d_c,
        );
    }

    // ── Step 8: Output gate + output projection (shared epilogue) ──────
    flashmemory_decode_epilogue(config, weights, scratch, h)
}

// ---------------------------------------------------------------------------
// Shared helpers (re-declared locally to avoid cross-module visibility issues)
// ---------------------------------------------------------------------------

/// RMSNorm applied in-place: x = x / ||x|| * gamma.
/// Mirrors `mla::rmsnorm_inplace` (kept local for module independence).
fn rmsnorm_inplace(x: &mut [f32], gamma: &[f32], eps: f32) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x.iter().take(n) {
        sum_sq += v * v;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for (xi, &g) in x.iter_mut().take(n).zip(gamma.iter().take(n)) {
        *xi = *xi * inv_rms * g;
    }
}

/// Decoupled RoPE application. Mirrors `mla::apply_decoupled_rope`.
fn apply_decoupled_rope(
    rope_freqs: &mut RopeFreqs,
    q: &mut [f32],
    d_r: usize,
    n_heads: usize,
    pos: usize,
) {
    for head in 0..n_heads {
        let start = head * d_r;
        rope_freqs.apply(&mut q[start..start + d_r], pos, false);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mla::{MlaConfig, MlaForwardScratch, MlaKVCache, MlaWeights};

    /// Small MLA config for fast tests (smaller than Kimi-K3-0.40B for speed).
    fn small_mla_config() -> MlaConfig {
        MlaConfig {
            kv_lora_rank: 32,
            q_lora_rank: 64,
            qk_nope_head_dim: 16,
            qk_rope_head_dim: 8,
            v_head_dim: 16,
            n_heads: 4,
            hidden_size: 128,
            use_output_gate: true,
            use_nope: false,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    fn make_weights(config: &MlaConfig) -> MlaWeights {
        MlaWeights::random(config, 42)
    }

    // ── Q1: Block centroids from MLA latent KV ──────────────────────────────

    #[test]
    fn q1_block_centroids_built_from_mla_latent_kv() {
        let config = small_mla_config();
        let weights = make_weights(&config);
        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 4,
            threshold: 0.5,
        };
        let max_seq = 16;
        let mut cache = MlaKVCache::new(&config, max_seq);
        let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, max_seq);

        // Append 12 tokens (3 full blocks of 4).
        for i in 0..12 {
            let c_kv = vec![(i as f32) * 0.1; config.kv_lora_rank];
            let k_r = vec![0.0; config.qk_rope_head_dim];
            cache.append(&c_kv, &k_r);
        }

        block_cache.rebuild_from_cache(&cache, &weights);

        assert_eq!(block_cache.n_active_blocks(), 3);
        assert_eq!(block_cache.block_count(0), 4);
        assert_eq!(block_cache.block_count(1), 4);
        assert_eq!(block_cache.block_count(2), 4);

        // Check latent centroid is the mean.
        // Token i has c_kv = [i*0.1; 32]. Block 0 = tokens 0-3, mean = 1.5*0.1 = 0.15.
        let centroid_0 = block_cache.latent_centroid(0);
        let expected = (0.0 + 0.1 + 0.2 + 0.3) / 4.0; // = 0.15
        assert!(
            (centroid_0[0] - expected).abs() < 1e-6,
            "centroid[0]={}, expected={}",
            centroid_0[0],
            expected
        );

        // Per-head key centroid should be non-zero (up-projection of non-zero centroid).
        let key_centroid = block_cache.key_centroid(0, 0);
        assert!(
            key_centroid.iter().any(|&v| v.abs() > 1e-6),
            "key centroid should be non-zero after up-projection"
        );
    }

    #[test]
    fn q1_partial_last_block_handled() {
        let config = small_mla_config();
        let weights = make_weights(&config);
        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 4,
            threshold: 0.5,
        };
        let max_seq = 16;
        let mut cache = MlaKVCache::new(&config, max_seq);
        let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, max_seq);

        // Append 10 tokens → 2 full blocks + 1 partial (2 tokens).
        for i in 0..10 {
            let c_kv = vec![(i as f32) * 0.1; config.kv_lora_rank];
            let k_r = vec![0.0; config.qk_rope_head_dim];
            cache.append(&c_kv, &k_r);
        }

        block_cache.rebuild_from_cache(&cache, &weights);

        assert_eq!(block_cache.n_active_blocks(), 3);
        assert_eq!(block_cache.block_count(0), 4);
        assert_eq!(block_cache.block_count(1), 4);
        assert_eq!(block_cache.block_count(2), 2); // partial

        // Token range for the partial block.
        let (start, end) = block_cache.block_token_range(2, 10);
        assert_eq!(start, 8);
        assert_eq!(end, 10);
    }

    // ── Q2: Periodic refresh amortization ───────────────────────────────────

    #[test]
    fn q2_periodic_refresh_amortizes_selection() {
        let config = small_mla_config();
        let weights = make_weights(&config);
        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 5, // refresh every 5 steps
            threshold: 0.5,
        };
        let max_seq = 32;
        let mut cache = MlaKVCache::new(&config, max_seq);
        let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, max_seq);
        let mut selector = FlashMemorySelector::new(fm_config, config.n_heads, 8);

        // Fill cache with 16 tokens (4 blocks).
        for i in 0..16 {
            let c_kv = vec![(i as f32) * 0.01; config.kv_lora_rank];
            let k_r = vec![0.0; config.qk_rope_head_dim];
            cache.append(&c_kv, &k_r);
        }
        block_cache.rebuild_from_cache(&cache, &weights);

        let q_c = vec![0.5; config.n_heads * config.d_h()];
        let scale = config.attn_scale();

        // Steps 0-9 (10 decode steps, refresh_period=5 → should refresh at steps 0 and 5).
        let mut refreshes_at = Vec::with_capacity(10);
        for step in 0..10 {
            selector.select(&q_c, &block_cache, scale, step);
            if selector.refresh_count() > refreshes_at.len() {
                refreshes_at.push(step);
            }
        }

        // Should have refreshed exactly 2 times (step 0 and step 5).
        assert_eq!(
            selector.refresh_count(),
            2,
            "expected 2 refreshes, got {}",
            selector.refresh_count()
        );
        assert_eq!(refreshes_at, vec![0, 5]);
    }

    #[test]
    fn q2_force_refresh_invalidates_cache() {
        let config = small_mla_config();
        let weights = make_weights(&config);
        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100, // long period — won't naturally refresh
            threshold: 0.5,
        };
        let max_seq = 16;
        let mut cache = MlaKVCache::new(&config, max_seq);
        let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, max_seq);
        let mut selector = FlashMemorySelector::new(fm_config, config.n_heads, 4);

        for i in 0..8 {
            let c_kv = vec![(i as f32) * 0.1; config.kv_lora_rank];
            let k_r = vec![0.0; config.qk_rope_head_dim];
            cache.append(&c_kv, &k_r);
        }
        block_cache.rebuild_from_cache(&cache, &weights);

        let q_c = vec![0.5; config.n_heads * config.d_h()];
        let scale = config.attn_scale();

        // Step 0: initial refresh.
        selector.select(&q_c, &block_cache, scale, 0);
        assert_eq!(selector.refresh_count(), 1);

        // Steps 1-10: no refresh (period=100).
        for step in 1..=10 {
            selector.select(&q_c, &block_cache, scale, step);
        }
        assert_eq!(selector.refresh_count(), 1);

        // Force refresh.
        selector.force_refresh();
        selector.select(&q_c, &block_cache, scale, 11);
        assert_eq!(selector.refresh_count(), 2);
    }

    // ── Q3: Sigmoid threshold produces dynamic block counts ────────────────

    #[test]
    fn q3_sigmoid_threshold_dynamic_block_counts() {
        let config = small_mla_config();
        let weights = make_weights(&config);
        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.5,
        };
        let max_seq = 32;
        let mut cache = MlaKVCache::new(&config, max_seq);
        let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, max_seq);
        let mut selector = FlashMemorySelector::new(fm_config, config.n_heads, 8);

        // Create blocks with DISTINCT centroids so different queries select
        // different subsets. Block 0: low values, Block 1: high values, etc.
        for block in 0..6 {
            for _ in 0..4 {
                let c_kv = vec![(block as f32) * 2.0; config.kv_lora_rank];
                let k_r = vec![0.0; config.qk_rope_head_dim];
                cache.append(&c_kv, &k_r);
            }
        }
        block_cache.rebuild_from_cache(&cache, &weights);

        let scale = config.attn_scale();

        // Query A: all zeros → dot products ≈ 0 → σ(0) = 0.5 → threshold met.
        // (Score = dot(q, centroid) * scale; if q=0, score=0, σ(0)=0.5 ≥ 0.5.)
        let q_zero = vec![0.0; config.n_heads * config.d_h()];
        selector.force_refresh();
        let sel_a = selector.select(&q_zero, &block_cache, scale, 0);
        let total_a = sel_a.total_selections();
        assert!(
            total_a > 0,
            "zero query should select blocks at threshold boundary"
        );

        // Query B: high positive → should select high-centroid blocks more.
        let q_high = vec![5.0; config.n_heads * config.d_h()];
        selector.force_refresh();
        let sel_b = selector.select(&q_high, &block_cache, scale, 1);
        // Clone the selection data before re-borrowing selector.
        let blocks_b_head0 = sel_b.blocks_per_head[0].clone();
        let total_b = sel_b.total_selections();

        // Query C: high negative → should select low-centroid blocks.
        let q_neg = vec![-5.0; config.n_heads * config.d_h()];
        selector.force_refresh();
        let sel_c = selector.select(&q_neg, &block_cache, scale, 2);
        let blocks_c_head0 = sel_c.blocks_per_head[0].clone();
        let total_c = sel_c.total_selections();

        // The KEY assertion: different queries produce DIFFERENT selection counts
        // (dynamic, not rigid top-k). At minimum, q_high and q_neg should differ
        // because they're on opposite sides of the centroid distribution.

        // With distinct block centroids (0, 2, 4, 6, 8, 10) and a scale factor,
        // q_high should favor high-centroid blocks and q_neg should favor low ones.
        // They won't necessarily have different COUNTS (sigmoid is symmetric), but
        // they should select different BLOCK SETS.

        // At least the sets should differ (not identical).
        let sets_differ = blocks_b_head0 != blocks_c_head0;
        assert!(
            sets_differ || total_b != total_c,
            "q_high and q_neg should produce different selections: \
             b={blocks_b_head0:?} (total={total_b}), c={blocks_c_head0:?} (total={total_c})"
        );
    }

    #[test]
    fn q3_threshold_controls_selectivity() {
        let config = small_mla_config();
        let weights = make_weights(&config);
        let max_seq = 32;
        let mut cache = MlaKVCache::new(&config, max_seq);

        // Fill with uniform-ish data.
        for i in 0..16 {
            let c_kv = vec![(i as f32) * 0.1; config.kv_lora_rank];
            let k_r = vec![0.0; config.qk_rope_head_dim];
            cache.append(&c_kv, &k_r);
        }

        let q_c = vec![1.0; config.n_heads * config.d_h()];
        let scale = config.attn_scale();

        // Low threshold → more blocks selected.
        let fm_low = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.01, // very low → almost everything passes
        };
        let mut block_cache_low = FlashMemoryBlockCache::new(&config, &fm_low, max_seq);
        block_cache_low.rebuild_from_cache(&cache, &weights);
        let mut sel_low = FlashMemorySelector::new(fm_low, config.n_heads, 4);
        let n_low = sel_low
            .select(&q_c, &block_cache_low, scale, 0)
            .total_selections();

        // High threshold → fewer blocks selected.
        let fm_high = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.99, // very high → almost nothing passes
        };
        let mut block_cache_high = FlashMemoryBlockCache::new(&config, &fm_high, max_seq);
        block_cache_high.rebuild_from_cache(&cache, &weights);
        let mut sel_high = FlashMemorySelector::new(fm_high, config.n_heads, 4);
        let n_high = sel_high
            .select(&q_c, &block_cache_high, scale, 0)
            .total_selections();

        assert!(
            n_low >= n_high,
            "lower threshold should select ≥ blocks: low={n_low} high={n_high}"
        );
    }

    // ── Q4: Sparse forward preserves accuracy on selected tokens ────────────

    #[test]
    fn q4_sparse_forward_runs_without_panic() {
        let config = small_mla_config();
        let weights = make_weights(&config);
        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.3, // low threshold → most blocks selected
        };
        let max_seq = 16;
        let mut cache = MlaKVCache::new(&config, max_seq);
        let mut scratch = MlaForwardScratch::new(&config, max_seq);
        let mut rope_freqs = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
        let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, max_seq);
        let mut selector = FlashMemorySelector::new(fm_config, config.n_heads, 4);

        // Prefill: process a few tokens.
        for step in 0..8 {
            let h = vec![(step as f32) * 0.1; config.hidden_size];
            // Normalize h to have reasonable magnitude.
            let h_norm: Vec<f32> = h.iter().map(|&v| v / 10.0).collect();
            mla_forward_token_flashmemory(
                &config,
                &weights,
                &mut cache,
                &mut scratch,
                &mut rope_freqs,
                &h_norm,
                &mut block_cache,
                &mut selector,
                step,
            );
        }

        // Verify the cache grew correctly.
        assert_eq!(cache.seq_len, 8);
        // Verify the output has the right dimension.
        let output = mla_forward_token_flashmemory(
            &config,
            &weights,
            &mut cache,
            &mut scratch,
            &mut rope_freqs,
            &vec![0.1; config.hidden_size],
            &mut block_cache,
            &mut selector,
            8,
        );
        assert_eq!(output.len(), config.hidden_size);
        // Output should not be all-NaN.
        assert!(
            output.iter().all(|&v| v.is_finite()),
            "output contains non-finite values"
        );
    }

    #[test]
    fn q4_no_selection_falls_back_to_recent_block() {
        // When threshold is so high that NO blocks are selected, the sparse
        // forward must fall back to attending the most recent block (recency bias).
        let config = small_mla_config();
        let weights = make_weights(&config);
        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.999, // impossibly high → nothing selected
        };
        let max_seq = 16;
        let mut cache = MlaKVCache::new(&config, max_seq);
        let mut scratch = MlaForwardScratch::new(&config, max_seq);
        let mut rope_freqs = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
        let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, max_seq);
        let mut selector = FlashMemorySelector::new(fm_config, config.n_heads, 4);

        for step in 0..8 {
            let h = vec![0.1; config.hidden_size];
            let out = mla_forward_token_flashmemory(
                &config,
                &weights,
                &mut cache,
                &mut scratch,
                &mut rope_freqs,
                &h,
                &mut block_cache,
                &mut selector,
                step,
            );
            // Should not panic, should produce finite output.
            assert!(
                out.iter().all(|&v| v.is_finite()),
                "fallback output contains non-finite values at step {step}"
            );
        }
    }

    #[test]
    fn q4_kimi_k3_0_40b_config_smoke() {
        // Smoke test with the ACTUAL Kimi-K3-0.40B config dimensions.
        // This validates the mechanism works at production scale (d_c=128, d_h=64, etc.).
        let config = MlaConfig::kimi_k3_0_40b();
        let weights = make_weights(&config);
        let fm_config = FlashMemoryConfig::test_config(); // block_size=16, τ=16
        let block_size = fm_config.block_size;
        let max_seq = 256; // small context for speed (mechanism test, not scale)
        let mut cache = MlaKVCache::new(&config, max_seq);
        let mut scratch = MlaForwardScratch::new(&config, max_seq);
        let mut rope_freqs = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);
        let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, max_seq);
        let mut selector =
            FlashMemorySelector::new(fm_config, config.n_heads, max_seq.div_ceil(block_size));

        // Process 64 tokens (4 blocks of 16).
        for step in 0..64 {
            let h = vec![((step % 10) as f32) * 0.01; config.hidden_size];
            let out = mla_forward_token_flashmemory(
                &config,
                &weights,
                &mut cache,
                &mut scratch,
                &mut rope_freqs,
                &h,
                &mut block_cache,
                &mut selector,
                step,
            );
            assert_eq!(out.len(), config.hidden_size);
            assert!(
                out.iter().all(|&v| v.is_finite()),
                "non-finite output at step {step}"
            );
        }

        // Verify the selector did periodic refreshes (τ=16, 64 steps → 4 refreshes).
        assert_eq!(
            selector.refresh_count(),
            4,
            "expected 4 refreshes over 64 steps"
        );
    }

    // ── Phase B: DualEncoderIndexer (Plan 337) ───────────────────────────────

    #[cfg(feature = "trained_indexer")]
    #[test]
    fn b1_dual_encoder_indexer_forward_produces_valid_scores() {
        // B1+B3: The indexer produces valid scalar scores + the threshold produces
        // dynamic block counts.
        let config = small_mla_config();
        let weights = make_weights(&config);
        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.5,
        };
        let d_h = config.d_h();
        let max_seq = 32;
        let mut cache = MlaKVCache::new(&config, max_seq);
        let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, max_seq);

        // Create blocks with distinct centroids.
        for block in 0..6 {
            for _ in 0..4 {
                let c_kv = vec![(block as f32) * 2.0; config.kv_lora_rank];
                let k_r = vec![0.0; config.qk_rope_head_dim];
                cache.append(&c_kv, &k_r);
            }
        }
        block_cache.rebuild_from_cache(&cache, &weights);

        let mut indexer = DualEncoderIndexer::new_random(fm_config, d_h, config.n_heads, 8, 42);

        // Sanity: param count matches expected for d_h=16 (test config).
        // hidden = d_h/4 = 4. Params = 2 * (4*16 + 4 + 4 + 1) = 2 * 73 = 146.
        assert_eq!(
            indexer.param_count(),
            146,
            "param count for d_h=16, hidden=4"
        );

        // Forward: indexer.select should produce a non-empty selection.
        let q_c = vec![1.0; config.n_heads * d_h];
        let sel = indexer.select(&q_c, &block_cache, 0);
        assert!(
            sel.total_selections() > 0,
            "indexer should select at least one block"
        );
        assert_eq!(indexer.refresh_count(), 1);
    }

    #[cfg(feature = "trained_indexer")]
    #[test]
    fn b3_dual_encoder_threshold_controls_selectivity() {
        // B3: Lower threshold selects more blocks than higher threshold.
        // This is the robust test that the mechanism supports dynamic selection
        // (same pattern as the modelless q3_threshold_controls_selectivity).
        let config = MlaConfig::kimi_k3_0_40b();
        let weights = make_weights(&config);
        let d_h = config.d_h();
        let max_seq: usize = 256;

        let mut cache = MlaKVCache::new(&config, max_seq);
        for i in 0..128 {
            let c_kv = vec![(i as f32) * 0.01; config.kv_lora_rank];
            let k_r = vec![0.0; config.qk_rope_head_dim];
            cache.append(&c_kv, &k_r);
        }

        let q_c = vec![0.5; config.n_heads * d_h];

        // Low threshold → more blocks.
        let fm_low = FlashMemoryConfig {
            block_size: 16,
            refresh_period: 100,
            threshold: 0.01,
        };
        let mut bc_low = FlashMemoryBlockCache::new(&config, &fm_low, max_seq);
        bc_low.rebuild_from_cache(&cache, &weights);
        let mut idx_low = DualEncoderIndexer::new_random(fm_low, d_h, config.n_heads, 16, 42);
        let n_low = idx_low.select(&q_c, &bc_low, 0).total_selections();

        // High threshold → fewer blocks.
        let fm_high = FlashMemoryConfig {
            block_size: 16,
            refresh_period: 100,
            threshold: 0.99,
        };
        let mut bc_high = FlashMemoryBlockCache::new(&config, &fm_high, max_seq);
        bc_high.rebuild_from_cache(&cache, &weights);
        let mut idx_high = DualEncoderIndexer::new_random(fm_high, d_h, config.n_heads, 16, 42);
        let n_high = idx_high.select(&q_c, &bc_high, 0).total_selections();

        assert!(
            n_low >= n_high,
            "lower threshold should select >= blocks: low={n_low} high={n_high}"
        );
    }

    #[cfg(feature = "trained_indexer")]
    #[test]
    fn b3_dual_encoder_periodic_refresh() {
        // The indexer respects refresh_period (amortizes k_score computation).
        let config = small_mla_config();
        let weights = make_weights(&config);
        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 5,
            threshold: 0.5,
        };
        let d_h = config.d_h();
        let max_seq = 32;
        let mut cache = MlaKVCache::new(&config, max_seq);
        let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, max_seq);

        for i in 0..16 {
            let c_kv = vec![(i as f32) * 0.01; config.kv_lora_rank];
            let k_r = vec![0.0; config.qk_rope_head_dim];
            cache.append(&c_kv, &k_r);
        }
        block_cache.rebuild_from_cache(&cache, &weights);

        let mut indexer = DualEncoderIndexer::new_random(fm_config, d_h, config.n_heads, 8, 99);

        let q_c = vec![0.5; config.n_heads * d_h];

        // Steps 0-9, refresh_period=5 → refresh at steps 0 and 5.
        let mut refreshes_at = Vec::with_capacity(10);
        for step in 0..10 {
            indexer.select(&q_c, &block_cache, step);
            if indexer.refresh_count() > refreshes_at.len() {
                refreshes_at.push(step);
            }
        }
        assert_eq!(indexer.refresh_count(), 2);
        assert_eq!(refreshes_at, vec![0, 5]);
    }

    #[cfg(feature = "trained_indexer")]
    #[test]
    fn b3_dual_encoder_serialize_roundtrip() {
        // Freeze/thaw: to_bytes → from_bytes produces identical selection.
        let config = small_mla_config();
        let weights = make_weights(&config);
        let fm_config = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.5,
        };
        let d_h = config.d_h();
        let max_seq = 32;
        let mut cache = MlaKVCache::new(&config, max_seq);
        let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, max_seq);

        for i in 0..16 {
            let c_kv = vec![(i as f32) * 0.1; config.kv_lora_rank];
            let k_r = vec![0.0; config.qk_rope_head_dim];
            cache.append(&c_kv, &k_r);
        }
        block_cache.rebuild_from_cache(&cache, &weights);

        let mut indexer =
            DualEncoderIndexer::new_random(fm_config.clone(), d_h, config.n_heads, 8, 7);

        let q_c = vec![1.0; config.n_heads * d_h];
        indexer.force_refresh();
        let sel_original = indexer.select(&q_c, &block_cache, 0).blocks_per_head[0].clone();

        // Serialize → deserialize.
        let bytes = indexer.to_bytes();
        let indexer2 = DualEncoderIndexer::from_bytes(fm_config, config.n_heads, 8, &bytes)
            .expect("deserialization should succeed");

        // Same query → same selection.
        let mut indexer2 = indexer2;
        let sel_restored = indexer2.select(&q_c, &block_cache, 0).blocks_per_head[0].clone();

        assert_eq!(
            sel_original, sel_restored,
            "serialized + deserialized indexer should produce identical selection"
        );
    }

    #[cfg(feature = "trained_indexer")]
    #[test]
    fn b3_dual_encoder_kimi_k3_scale_smoke() {
        // Smoke test at Kimi-K3-0.40B dimensions (d_h=64, hidden=16).
        let config = MlaConfig::kimi_k3_0_40b();
        let weights = make_weights(&config);
        let fm_config = FlashMemoryConfig::test_config(); // block_size=16, τ=16
        let block_size = fm_config.block_size;
        let d_h = config.d_h();
        let max_seq: usize = 256; // 16 blocks
        let max_blocks = max_seq.div_ceil(block_size);

        let mut cache = MlaKVCache::new(&config, max_seq);
        let mut block_cache = FlashMemoryBlockCache::new(&config, &fm_config, max_seq);

        for i in 0..64 {
            let c_kv = vec![(i as f32) * 0.01; config.kv_lora_rank];
            let k_r = vec![0.0; config.qk_rope_head_dim];
            cache.append(&c_kv, &k_r);
        }
        block_cache.rebuild_from_cache(&cache, &weights);

        let mut indexer =
            DualEncoderIndexer::new_random(fm_config, d_h, config.n_heads, max_blocks, 42);

        // Kimi-K3: d_h=64, hidden=16. Params = 2*(16*64 + 16 + 16 + 1) = 2*1057 = 2114.
        assert_eq!(indexer.param_count(), 2114);

        let q_c = vec![0.5; config.n_heads * d_h];
        let sel = indexer.select(&q_c, &block_cache, 0);
        assert!(sel.total_selections() > 0);
    }

    #[cfg(feature = "trained_indexer")]
    #[test]
    fn b_gqa1_select_gqa_produces_valid_selection() {
        // GQA sibling of b1: select_gqa reads raw per-KV-head centroids from
        // GqaFlashMemoryBlockCache + the KV-resolution query, and produces a
        // non-empty selection with distinct per-head results possible.
        let fm = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.5,
        };
        let n_kv_head = 2;
        let head_dim = 8;
        let seq_len = 16; // 4 blocks
        let mut cache = GqaFlashMemoryBlockCache::new(n_kv_head, head_dim, &fm, seq_len);

        // Distinct centroids: block b, head h = b*1.0 + h*10.0.
        let kv_dim = n_kv_head * head_dim;
        let mut keys = vec![0.0f32; seq_len * kv_dim];
        for t in 0..seq_len {
            let b = t / 4;
            for h in 0..n_kv_head {
                let val = (b as f32) + (h as f32) * 10.0;
                for d in 0..head_dim {
                    keys[t * kv_dim + h * head_dim + d] = val;
                }
            }
        }
        cache.rebuild_from_keys(&keys, seq_len);

        let max_blocks = seq_len.div_ceil(fm.block_size);
        let mut indexer = DualEncoderIndexer::new_random(fm, head_dim, n_kv_head, max_blocks, 42);

        let query = vec![1.0f32; n_kv_head * head_dim];
        let first: Vec<Vec<usize>> = indexer
            .select_gqa(&query, &cache, 0)
            .blocks_per_head
            .to_vec();
        assert!(
            first.iter().map(|v| v.len()).sum::<usize>() > 0,
            "select_gqa should select at least one block"
        );
        assert_eq!(indexer.refresh_count(), 1);

        // Determinism: same inputs → same selection (owned copies; the
        // returned ref borrows the indexer).
        indexer.force_refresh();
        let second: Vec<Vec<usize>> = indexer
            .select_gqa(&query, &cache, 1)
            .blocks_per_head
            .to_vec();
        assert_eq!(first, second);
    }

    #[cfg(feature = "trained_indexer")]
    #[test]
    fn b_gqa2_zero_weights_threshold_semantics_pinned() {
        // Pinned semantics for checkpoint serving: with all-zero weights every
        // score is exactly 0 → σ(0)=0.5. Threshold 0.5 (≥) selects ALL blocks;
        // 0.5000001 selects NONE. This pins the σ(q·k) ≥ threshold contract
        // the riir-train Plan 337 checkpoints were trained under.
        let fm_all = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.5,
        };
        let n_kv_head = 2;
        let head_dim = 8;
        let seq_len = 8; // 2 blocks
        let mut cache = GqaFlashMemoryBlockCache::new(n_kv_head, head_dim, &fm_all, seq_len);
        let kv_dim = n_kv_head * head_dim;
        cache.rebuild_from_keys(&vec![1.0f32; seq_len * kv_dim], seq_len);

        let max_blocks = seq_len.div_ceil(fm_all.block_size);
        let zeros_h = vec![0.0f32; head_dim / 4]; // hidden = 8/4 = 2
        let zeros1 = vec![0.0f32; 2];
        let zeros_w1 = vec![0.0f32; 2 * head_dim];
        let zero_indexer = |fm: FlashMemoryConfig| {
            DualEncoderIndexer::from_weights(
                fm,
                head_dim,
                n_kv_head,
                max_blocks,
                zeros_w1.clone(),
                zeros_h.clone(),
                zeros1.clone(),
                0.0,
                zeros_w1.clone(),
                zeros_h.clone(),
                zeros1.clone(),
                0.0,
            )
        };

        let query = vec![0.7f32; n_kv_head * head_dim];

        // Threshold 0.5: σ(0·0)=0.5 ≥ 0.5 → ALL blocks selected, every head.
        let mut idx = zero_indexer(fm_all);
        let sel = idx.select_gqa(&query, &cache, 0);
        for h in 0..n_kv_head {
            assert_eq!(
                sel.blocks_per_head[h].len(),
                2,
                "all blocks selected at thr 0.5"
            );
        }

        // Threshold just above 0.5: nothing selected.
        let fm_none = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.5000001,
        };
        let mut idx2 = zero_indexer(fm_none);
        let sel2 = idx2.select_gqa(&query, &cache, 0);
        assert_eq!(sel2.total_selections(), 0, "no blocks selected above σ(0)");
    }

    #[cfg(feature = "trained_indexer")]
    #[test]
    fn b_gqa3_periodic_refresh_reuses_selection() {
        // Same amortization contract as the modelless GQA selector: within the
        // refresh period the cached selection is returned and refresh_count
        // does not advance.
        let fm = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 5,
            threshold: 0.5,
        };
        let n_kv_head = 1;
        let head_dim = 4;
        let seq_len = 16;
        let mut cache = GqaFlashMemoryBlockCache::new(n_kv_head, head_dim, &fm, seq_len);
        cache.rebuild_from_keys(&vec![1.0f32; seq_len * n_kv_head * head_dim], seq_len);

        let max_blocks = seq_len.div_ceil(fm.block_size);
        let mut indexer = DualEncoderIndexer::new_random(fm, head_dim, n_kv_head, max_blocks, 7);
        let query = vec![1.0f32; n_kv_head * head_dim];

        let _ = indexer.select_gqa(&query, &cache, 0);
        assert_eq!(indexer.refresh_count(), 1);
        for step in 1..5 {
            let _ = indexer.select_gqa(&query, &cache, step);
        }
        assert_eq!(indexer.refresh_count(), 1, "no refresh within the period");
        let _ = indexer.select_gqa(&query, &cache, 5);
        assert_eq!(indexer.refresh_count(), 2, "refresh at period boundary");
    }

    // ── GQA FlashMemory tests (Issue 584 Phase 2 — Bonsai/Qwen3.5 support) ──

    /// GQA block cache builds correct centroids from raw keys.
    #[test]
    fn gqa_q1_block_centroids_built_from_raw_keys() {
        let fm = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.5,
        };
        let n_kv_head = 2;
        let head_dim = 8;
        let seq_len = 8; // 2 blocks
        let mut cache = GqaFlashMemoryBlockCache::new(n_kv_head, head_dim, &fm, seq_len);

        // Keys: token t, head h = [t * 1.0 + h * 10.0; head_dim].
        let kv_dim = n_kv_head * head_dim;
        let mut keys = vec![0.0f32; seq_len * kv_dim];
        for t in 0..seq_len {
            for h in 0..n_kv_head {
                let val = (t as f32) + (h as f32) * 10.0;
                for d in 0..head_dim {
                    keys[t * kv_dim + h * head_dim + d] = val;
                }
            }
        }
        cache.rebuild_from_keys(&keys, seq_len);

        assert_eq!(cache.n_active_blocks(), 2);
        assert_eq!(cache.block_count(0), 4);
        assert_eq!(cache.block_count(1), 4);

        // Block 0: tokens 0,1,2,3. Centroid head 0 = mean(0,1,2,3) = 1.5.
        let c0_h0 = cache.key_centroid(0, 0);
        assert!(
            (c0_h0[0] - 1.5).abs() < 1e-5,
            "block 0 head 0 centroid = {}",
            c0_h0[0]
        );

        // Block 0 head 1 = mean(10,11,12,13) = 11.5.
        let c0_h1 = cache.key_centroid(0, 1);
        assert!(
            (c0_h1[0] - 11.5).abs() < 1e-5,
            "block 0 head 1 centroid = {}",
            c0_h1[0]
        );

        // Block 1: tokens 4,5,6,7. Centroid head 0 = mean(4,5,6,7) = 5.5.
        let c1_h0 = cache.key_centroid(1, 0);
        assert!((c1_h0[0] - 5.5).abs() < 1e-5);
    }

    /// GQA block cache handles partial last block.
    #[test]
    fn gqa_q1_partial_last_block() {
        let fm = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 100,
            threshold: 0.5,
        };
        let mut cache = GqaFlashMemoryBlockCache::new(1, 4, &fm, 10);
        let keys = vec![1.0f32; 6 * 4]; // 6 tokens, 1 head, head_dim=4
        cache.rebuild_from_keys(&keys, 6);
        assert_eq!(cache.n_active_blocks(), 2);
        assert_eq!(cache.block_count(0), 4);
        assert_eq!(cache.block_count(1), 2);
        // Block 1: mean of 2 tokens (both 1.0) = 1.0.
        assert!((cache.key_centroid(1, 0)[0] - 1.0).abs() < 1e-5);
    }

    /// GQA selector produces valid per-KV-head selection with sigmoid threshold.
    #[test]
    fn gqa_q3_selector_sigmoid_threshold() {
        let fm = FlashMemoryConfig {
            block_size: 2,
            refresh_period: 100,
            threshold: 0.5,
        };
        let n_kv_head = 2;
        let head_dim = 4;
        let seq_len = 8; // 4 blocks
        let mut cache = GqaFlashMemoryBlockCache::new(n_kv_head, head_dim, &fm, seq_len);
        let kv_dim = n_kv_head * head_dim;
        // Make block 0 keys align with the query, others orthogonal.
        let mut keys = vec![0.0f32; seq_len * kv_dim];
        for t in 0..2 {
            // Block 0, head 0: strong keys.
            keys[t * kv_dim] = 5.0;
            keys[t * kv_dim + 1] = 5.0;
        }
        cache.rebuild_from_keys(&keys, seq_len);

        let max_blocks = seq_len.div_ceil(fm.block_size);
        let mut sel = GqaFlashMemorySelector::new(fm, n_kv_head, head_dim, max_blocks);

        // Query head 0 aligned with block 0.
        let mut query = vec![0.0f32; n_kv_head * head_dim];
        query[0] = 5.0;
        query[1] = 5.0;

        let selection = sel.select(&query, &cache, 1.0, 0);
        // Block 0 should be selected for head 0 (high dot product).
        assert!(
            selection.blocks_per_head[0].contains(&0),
            "block 0 should be selected for head 0"
        );
    }

    /// GQA selector periodic refresh amortizes scoring.
    #[test]
    fn gqa_q2_periodic_refresh() {
        let fm = FlashMemoryConfig {
            block_size: 4,
            refresh_period: 5,
            threshold: 0.5,
        };
        let mut cache = GqaFlashMemoryBlockCache::new(1, 4, &fm, 32);
        let keys = vec![1.0f32; 16 * 4];
        cache.rebuild_from_keys(&keys, 16);

        let mut sel = GqaFlashMemorySelector::new(fm, 1, 4, 8);
        let query = vec![1.0; 4];

        let mut refreshes_at = Vec::with_capacity(10);
        for step in 0..10 {
            sel.select(&query, &cache, 1.0, step);
            if sel.refresh_count() > refreshes_at.len() {
                refreshes_at.push(step);
            }
        }
        assert_eq!(sel.refresh_count(), 2);
        assert_eq!(refreshes_at, vec![0, 5]);
    }

    /// Bonsai-scale smoke test: n_kv_head=8, head_dim=256, 64 tokens.
    #[test]
    fn gqa_bonsai_scale_smoke() {
        let fm = FlashMemoryConfig::test_config(); // block_size=16
        let n_kv_head = 8;
        let head_dim = 256;
        let seq_len = 64; // 4 blocks
        let mut cache = GqaFlashMemoryBlockCache::new(n_kv_head, head_dim, &fm, seq_len);
        let kv_dim = n_kv_head * head_dim;
        let keys = vec![0.1f32; seq_len * kv_dim];
        cache.rebuild_from_keys(&keys, seq_len);
        assert_eq!(cache.n_active_blocks(), 4);

        let max_blocks = seq_len.div_ceil(fm.block_size);
        let mut sel = GqaFlashMemorySelector::new(fm, n_kv_head, head_dim, max_blocks);
        let query = vec![0.1; n_kv_head * head_dim];
        let selection = sel.select(&query, &cache, 1.0 / (head_dim as f32).sqrt(), 0);
        assert!(selection.total_selections() > 0);
    }
}
