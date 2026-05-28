use crate::types::{self, *};
use rayon::prelude::*;

/// Decode stage for specialized forward paths (Plan 102: TileRT pipeline).
/// Different stages have different optimization opportunities:
/// - Draft: can skip screening, reduced KV writes, approximate attention
/// - Verify: exact attention, full KV write, enable screening
/// - Sample: SIMD-only, no attention needed
#[cfg(feature = "decode_specialize")]
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecodeStage {
    /// Batch-friendly, attention-heavy, needs full KV write.
    Prefill,
    /// Small batch, can skip screening, matmul-heavy.
    Draft,
    /// Single batch, needs exact attention, KV read-heavy.
    Verify,
    /// SIMD-only, no attention needed.
    Sample,
}

/// Per-layer transformer weights.
/// Each layer has its own attention and MLP parameters.
pub struct LayerWeights {
    pub attn_wq: Vec<f32>, // [n_embd, n_embd]
    pub attn_wk: Vec<f32>, // [kv_dim, n_embd] where kv_dim = n_kv_head * head_dim
    pub attn_wv: Vec<f32>, // [kv_dim, n_embd]
    pub attn_wo: Vec<f32>, // [n_embd, n_embd]
    pub mlp_w1: Vec<f32>,  // [mlp_hidden, n_embd]
    pub mlp_w2: Vec<f32>,  // [n_embd, mlp_hidden]
}

/// All transformer weights: embeddings, per-layer weights, and LM head.
/// Layout preserves init order for backward compat: wte, wpe, layers…, lm_head.
///
/// # Future: f16 Storage
///
/// For memory-constrained deployments, weights can be stored as `f16` (half-precision)
/// and quantized on-the-fly during matmul. This would halve memory usage with minimal
/// accuracy loss for inference-only workloads. The migration path:
///
/// 1. Add a `StorageFormat` enum: `F32`, `F16`, `Q4_0`, `Q8_0`
/// 2. Replace `Vec<f32>` with a `WeightTensor` enum that stores the chosen format
/// 3. Add `dequantize_row()` that converts to `f32` on-the-fly during matmul
/// 4. The `forward()` kernel remains unchanged — it operates on `f32` buffers
///    populated by dequantization
///
/// Key insight: only storage changes; compute stays in `f32`. This avoids the need
/// for f16 arithmetic hardware support and keeps the attention kernel simple.
/// Estimated memory savings: ~50% for f16, ~75% for 4-bit quantized.
pub struct TransformerWeights {
    pub wte: Vec<f32>,             // [vocab_size, n_embd]
    pub wpe: Vec<f32>,             // [block_size, n_embd]
    pub lm_head: Vec<f32>,         // [vocab_size, n_embd]
    pub layers: Vec<LayerWeights>, // [n_layer]
    // MTP Drafter weights (Plan 055: Gemma 4 MTP)
    /// Target→Draft activation projection: [draft_n_embd, target_n_embd + embed_dim]
    /// Only loaded when Config mtp_activation_threshold is met and weights file exists.
    /// Falls back to truncate/pad when absent.
    pub mtp_activation_proj: Option<Vec<f32>>,
    /// Cluster classifier: [num_clusters, n_embd]
    /// Only loaded when vocab_size > mtp_cluster_vocab_threshold.
    pub mtp_cluster_classifier: Option<Vec<f32>>,
    /// Cluster membership table: [num_clusters] → Vec<usize> (token indices)
    pub mtp_cluster_map: Option<Vec<Vec<usize>>>,
    // Delta routing weights (Plan 097: Delta Attention Residuals)
    #[cfg(feature = "delta_routing")]
    pub delta_routing_query: Vec<Vec<f32>>, // [n_layer][n_embd] per-layer query vectors
    #[cfg(feature = "delta_routing")]
    pub delta_routing_norm: Vec<Vec<f32>>, // [n_layer][n_embd] per-layer RMSNorm weights (gamma)
}

impl TransformerWeights {
    pub fn new(config: &Config, rng: &mut Rng) -> Self {
        let n = config.n_embd;
        let kvd = types::kv_dim(config);
        let embd_scale = (2.0 / n as f32).sqrt();
        let layer_scale = (2.0 / (n as f32 * config.n_layer as f32)).sqrt();

        // Embeddings first (same order as original single-layer code)
        let wte: Vec<f32> = (0..config.vocab_size * n)
            .map(|_| rng.normal() * embd_scale)
            .collect();
        let wpe: Vec<f32> = (0..config.block_size * n)
            .map(|_| rng.normal() * embd_scale)
            .collect();

        // Per-layer weights: same field order as original per n_layer iterations
        let layers: Vec<LayerWeights> = (0..config.n_layer)
            .map(|_| LayerWeights {
                attn_wq: (0..n * n).map(|_| rng.normal() * layer_scale).collect(),
                attn_wk: (0..kvd * n).map(|_| rng.normal() * layer_scale).collect(),
                attn_wv: (0..kvd * n).map(|_| rng.normal() * layer_scale).collect(),
                attn_wo: (0..n * n).map(|_| rng.normal() * layer_scale).collect(),
                mlp_w1: (0..config.mlp_hidden * n)
                    .map(|_| rng.normal() * layer_scale)
                    .collect(),
                mlp_w2: (0..n * config.mlp_hidden)
                    .map(|_| rng.normal() * layer_scale)
                    .collect(),
            })
            .collect();

        // LM head last
        let lm_head: Vec<f32> = (0..config.vocab_size * n)
            .map(|_| rng.normal() * embd_scale)
            .collect();

        Self {
            wte,
            wpe,
            lm_head,
            layers,
            mtp_activation_proj: None,
            mtp_cluster_classifier: None,
            mtp_cluster_map: None,
            #[cfg(feature = "delta_routing")]
            delta_routing_query: (0..config.n_layer)
                .map(|_| vec![0.0; config.n_embd]) // Zero-init: safe additive start
                .collect(),
            #[cfg(feature = "delta_routing")]
            delta_routing_norm: (0..config.n_layer)
                .map(|_| (0..config.n_embd).map(|_| 1.0f32).collect()) // Ones: identity RMSNorm
                .collect(),
        }
    }
}

/// KV cache for a single layer (autoregressive generation).
pub struct KVCache {
    pub key: Vec<f32>,   // [block_size, kv_dim] where kv_dim = n_kv_head * head_dim
    pub value: Vec<f32>, // [block_size, kv_dim]
}

impl KVCache {
    pub fn new(config: &Config) -> Self {
        let kvd = types::kv_dim(config);
        Self {
            key: vec![0.0; config.block_size * kvd],
            value: vec![0.0; config.block_size * kvd],
        }
    }

    pub fn reset(&mut self) {
        // No-op: each position is written before being read, so stale data
        // from previous sequences is never observed. Avoids O(block_size × kv_dim) zeroing.
    }
}

/// Multi-layer KV cache: one KVCache per transformer layer.
pub struct MultiLayerKVCache {
    pub layers: Vec<KVCache>,
    /// Highest position written + 1 across all layers, for efficient snapshot.
    fill_pos: usize,
}

impl MultiLayerKVCache {
    pub fn new(config: &Config) -> Self {
        Self {
            layers: (0..config.n_layer).map(|_| KVCache::new(config)).collect(),
            fill_pos: 0,
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.fill_pos = 0;
    }

    /// Update fill_pos tracker. Call after writing to the cache at a position.
    pub fn advance_pos(&mut self, pos: usize) {
        self.fill_pos = self.fill_pos.max(pos + 1);
    }

    /// Get the tracked fill position (highest position written + 1).
    pub fn fill_pos(&self) -> usize {
        self.fill_pos
    }

    /// Snapshot KV cache state up to position `pos`.
    /// Copies only filled slots [0..pos) per layer — cheap at our model scale.
    pub fn snapshot(&self, pos: usize, config: &Config) -> KVSnapshot {
        let kd = types::kv_dim(config);
        let end = pos * kd;
        let layers = self
            .layers
            .iter()
            .map(|layer| KVLayerSnapshot {
                key: layer.key[..end].to_vec(),
                value: layer.value[..end].to_vec(),
            })
            .collect();
        KVSnapshot { layers, pos }
    }

    /// Restore KV cache from a snapshot.
    /// Writes snapshot data back and zeros out positions [snapshot.pos..block_size).
    pub fn restore(&mut self, snapshot: &KVSnapshot, config: &Config) {
        let kd = types::kv_dim(config);
        for (layer, snap_layer) in self.layers.iter_mut().zip(snapshot.layers.iter()) {
            let end = snapshot.pos * kd;
            layer.key[..end].copy_from_slice(&snap_layer.key);
            layer.value[..end].copy_from_slice(&snap_layer.value);
            // Zero out positions [snapshot.pos..block_size) to prevent stale data
            layer.key[end..].fill(0.0);
            layer.value[end..].fill(0.0);
        }
    }
}

/// Preload drafter's KV cache with target's pre-computed key/value pairs.
///
/// Copies target's KV for positions [0..pos) into drafter's cache.
/// This enables cross-attention: the drafter attends to the target's past KV
/// instead of computing its own from scratch.
///
/// Only active when `target_kv_dim == draft_kv_dim` (dimensions must match).
/// When dimensions don't match, silently returns (drafter computes its own KV).
///
/// Hybrid behavior after preload:
/// - Past positions [0..pos): read from preloaded target KV
/// - New positions [pos..]: computed by drafter during forward pass
pub fn preload_kv_cache(
    draft_cache: &mut MultiLayerKVCache,
    target_cache: &MultiLayerKVCache,
    pos: usize,
    target_config: &Config,
    draft_config: &Config,
) {
    let target_kv_dim = types::kv_dim(target_config);
    let draft_kv_dim = types::kv_dim(draft_config);

    // Dimension guard: can only share when kv_dim matches
    if target_kv_dim != draft_kv_dim {
        return;
    }

    // Layer guard: can only share layers that exist in both caches
    let min_layers = draft_cache.layers.len().min(target_cache.layers.len());

    // Copy KV for positions [0..pos) for each shared layer
    let copy_len = pos * target_kv_dim;
    if copy_len > 0 {
        for layer_idx in 0..min_layers {
            let draft_layer = &mut draft_cache.layers[layer_idx];
            let target_layer = &target_cache.layers[layer_idx];
            draft_layer.key[..copy_len].copy_from_slice(&target_layer.key[..copy_len]);
            draft_layer.value[..copy_len].copy_from_slice(&target_layer.value[..copy_len]);
        }
    }
}

/// Cheap snapshot of KV cache state up to position `pos`.
/// Only copies filled slots [0..pos) per layer, not the entire block_size buffer.
pub struct KVSnapshot {
    pub layers: Vec<KVLayerSnapshot>,
    pub pos: usize,
}

/// Per-layer snapshot of KV cache data.
pub struct KVLayerSnapshot {
    pub key: Vec<f32>,   // [pos * kv_dim]
    pub value: Vec<f32>, // [pos * kv_dim]
}

/// Pre-allocated buffers for zero-alloc forward passes.
/// Create once, reuse across calls.
pub struct ForwardContext {
    pub(crate) x: Vec<f32>,        // [n_embd] main activation
    pub(crate) xr: Vec<f32>,       // [n_embd] residual
    pub(crate) xr2: Vec<f32>,      // [n_embd] residual 2
    pub(crate) q: Vec<f32>,        // [n_embd] query
    pub(crate) k: Vec<f32>,        // [kv_dim] key (kv_dim = n_kv_head * head_dim)
    pub(crate) v: Vec<f32>,        // [kv_dim] value
    pub(crate) attn_out: Vec<f32>, // [n_embd] attention output
    pub scores: Vec<f32>,          // [block_size] attention scores (max possible)
    pub(crate) hidden: Vec<f32>,   // [mlp_hidden] MLP hidden
    pub logits: Vec<f32>,          // [vocab_size] output logits
    pub(crate) cdf: Vec<f32>,      // [vocab_size] pre-allocated CDF for sampling
    pub hidden_state: Vec<f32>,    // [n_embd] final hidden state (Plan 009 compat)
    /// LoRA intermediate buffer [lora_rank]. Pre-allocated, zero alloc in hot path.
    pub lora_buf: Vec<f32>,
    /// Pre-computed attention scale: `1.0 / sqrt(head_dim)`. Constant per config.
    attn_scale: f32,
    // CNA: contrastive neuron attribution runtime modulator (Plan 087)
    #[cfg(feature = "cna_steering")]
    pub cna_modulator: Option<crate::pruners::CnaModulator>,
    // Sparse MLP buffers (Plan 022: TwELL-inspired unstructured sparsity)
    #[cfg(feature = "sparse_mlp")]
    pub(crate) active_indices: Vec<usize>, // [mlp_hidden] pre-allocated index buffer
    #[cfg(feature = "sparse_mlp")]
    pub(crate) active_values: Vec<f32>, // [mlp_hidden] pre-allocated value buffer
    // Paged KV cache: pre-allocated flat buffers for attention computation
    paged_flat_key: Vec<f32>,   // [block_size * kv_dim]
    paged_flat_value: Vec<f32>, // [block_size * kv_dim]
    // Raven: pre-allocated query buffer for per-head slot attention
    raven_query_buf: Vec<f32>, // [kv_dim]
    // MTP Drafter: pre-allocated projection buffer [n_embd] for target activation conditioning (Plan 055)
    pub mtp_context_buf: Vec<f32>,
    // Quantized KV cache incremental dequant: tracks last dequantized position per layer (Plan 068).
    // When dequant_pos[layer] == pos - 1, only dequant the new position (O(1) vs O(pos)).
    // On mismatch (layer switch, reset, pos jump), rebuild all positions for that layer.
    dequant_pos: Vec<usize>, // [n_layer]
    // Delta routing: block delta accumulation buffers (Plan 097)
    #[cfg(feature = "delta_routing")]
    block_deltas: Vec<Vec<f32>>, // [n_blocks][n_embd] accumulated deltas per block
    #[cfg(feature = "delta_routing")]
    delta_routing_logits: Vec<f32>, // [max_sources] routing logits temp buffer
    // CODA fused kernels: partial RMS accumulation buffer (Plan 103)
    #[cfg(feature = "coda_fusion")]
    coda_partial_sums: Vec<f32>, // [1] single-block RMS sum of squares
    // MLS Multi-Layer Sum aggregation (Plan 104: Research 68)
    #[cfg(feature = "mls_aggregate")]
    mls_buf: Vec<f32>, // [n_embd] accumulator for last K layer residuals
    #[cfg(feature = "mls_aggregate")]
    mls_count: usize, // How many layers accumulated
    // Tiled attention: pre-allocated repacking buffers for forward_prefill (Plan 115)
    // Layout: [block_size × n_embd] (Q/out) or [block_size × kv_dim] (K/V)
    // Data is repacked from (position, head) → (head, position) for tiled_attention_batched
    #[cfg(feature = "tiled_attention")]
    tiled_q: Vec<f32>, // [block_size × n_embd] repacked queries per head
    #[cfg(feature = "tiled_attention")]
    tiled_k: Vec<f32>, // [block_size × kv_dim] repacked keys per kv group
    #[cfg(feature = "tiled_attention")]
    tiled_v: Vec<f32>, // [block_size × kv_dim] repacked values per kv group
    #[cfg(feature = "tiled_attention")]
    tiled_out: Vec<f32>, // [block_size × n_embd] tiled output before transpose
    // Clustered LM head scratch buffers (avoid per-forward-pass allocation)
    cluster_scores_buf: Vec<f32>, // [num_clusters] cluster scores for clustered LM head
    topk_indexed_buf: Vec<(usize, f32)>, // [num_clusters] indexed pairs for cluster top-K
    topk_output_buf: Vec<usize>,  // [topk] output indices buffer
    // Loop residual: saves h^(τ-1) for residual gating across weight-shared loops
    pub(crate) prev_h: Vec<f32>, // [n_embd]
    // GQA lookup: kv_group_lut[h] = h * n_kv_head / n_head (pre-computed once)
    kv_group_lut: [usize; 64], // fixed-size LUT for GQA head→kv_group mapping
    _kv_group_lut_count: usize, // actual number of heads (n_head)
    // Delta routing: pre-allocated source_refs index buffer (stores block indices, not slices)
    #[cfg(feature = "delta_routing")]
    delta_source_indices: Vec<usize>, // pre-allocated capacity for max sources
    // Delta routing: scratch buffer for SIMD scaling in depth_route (Issue 082)
    #[cfg(feature = "delta_routing")]
    delta_scaled_buf: Vec<f32>, // [n_embd] scratch for pre-scaled dot products
    // Training-free loop: pre-allocated buffers for window iteration (Issue 091)
    #[cfg(feature = "tf_loop")]
    tf_x_pre_window: Vec<f32>, // [n_embd] saved state before window
    #[cfg(feature = "tf_loop")]
    tf_x_anchor: Vec<f32>, // [n_embd] anchor state
    #[cfg(feature = "tf_loop")]
    tf_y_buf: Vec<f32>, // [n_embd] temp buffer for window output
    #[cfg(feature = "tf_loop")]
    tf_stash_x: Vec<f32>, // [n_embd] stash for KV cache write
}

impl ForwardContext {
    pub fn new(config: &Config) -> Self {
        let kvd = types::kv_dim(config);
        let block_kv = config.block_size * kvd;
        Self {
            x: vec![0.0; config.n_embd],
            xr: vec![0.0; config.n_embd],
            xr2: vec![0.0; config.n_embd],
            q: vec![0.0; config.n_embd],
            k: vec![0.0; kvd],
            v: vec![0.0; kvd],
            attn_out: vec![0.0; config.n_embd],
            scores: vec![0.0; config.block_size],
            hidden: vec![0.0; config.mlp_hidden],
            logits: vec![0.0; config.vocab_size],
            cdf: vec![0.0; config.vocab_size],
            hidden_state: vec![0.0; config.n_embd],
            lora_buf: vec![0.0; config.lora_rank],
            attn_scale: 1.0 / (config.head_dim as f32).sqrt(),
            #[cfg(feature = "cna_steering")]
            cna_modulator: None,
            #[cfg(feature = "sparse_mlp")]
            active_indices: vec![0; config.mlp_hidden],
            #[cfg(feature = "sparse_mlp")]
            active_values: vec![0.0; config.mlp_hidden],
            paged_flat_key: vec![0.0; block_kv],
            paged_flat_value: vec![0.0; block_kv],
            raven_query_buf: vec![0.0; kvd],
            mtp_context_buf: vec![0.0; config.n_embd],
            dequant_pos: vec![0; config.n_layer],
            #[cfg(feature = "delta_routing")]
            block_deltas: {
                let block_size = 4; // Default B=4
                let n_blocks = config.n_layer.div_ceil(block_size);
                (0..n_blocks).map(|_| vec![0.0; config.n_embd]).collect()
            },
            #[cfg(feature = "delta_routing")]
            delta_routing_logits: vec![0.0; config.n_layer + 1], // Max B+1 sources
            #[cfg(feature = "coda_fusion")]
            coda_partial_sums: vec![0.0; 1], // Single-block partial RMS (Plan 103)
            #[cfg(feature = "mls_aggregate")]
            mls_buf: vec![0.0; config.n_embd],
            #[cfg(feature = "mls_aggregate")]
            mls_count: 0,
            #[cfg(feature = "tiled_attention")]
            tiled_q: vec![0.0; config.block_size * config.n_embd],
            #[cfg(feature = "tiled_attention")]
            tiled_k: vec![0.0; config.block_size * kvd],
            #[cfg(feature = "tiled_attention")]
            tiled_v: vec![0.0; config.block_size * kvd],
            #[cfg(feature = "tiled_attention")]
            tiled_out: vec![0.0; config.block_size * config.n_embd],
            prev_h: vec![0.0; config.n_embd],
            cluster_scores_buf: vec![
                0.0;
                config.vocab_size.div_ceil(config.mtp_cluster_size.max(1))
            ],
            topk_indexed_buf: vec![
                (0usize, 0.0f32);
                config.vocab_size.div_ceil(config.mtp_cluster_size.max(1))
            ],
            topk_output_buf: Vec::new(),
            kv_group_lut: {
                let n_head = config.n_head;
                let n_kv_head = config.n_kv_head;
                let mut lut = [0usize; 64];
                for (h, slot) in lut.iter_mut().enumerate().take(n_head.min(64)) {
                    *slot = h * n_kv_head / n_head;
                }
                lut
            },
            _kv_group_lut_count: config.n_head,
            #[cfg(feature = "delta_routing")]
            delta_source_indices: {
                let block_size = 4; // Default B=4
                let n_blocks = config.n_layer.div_ceil(block_size);
                Vec::with_capacity(n_blocks + 1)
            },
            #[cfg(feature = "delta_routing")]
            delta_scaled_buf: vec![0.0f32; config.n_embd],
            #[cfg(feature = "tf_loop")]
            tf_x_pre_window: vec![0.0f32; config.n_embd],
            #[cfg(feature = "tf_loop")]
            tf_x_anchor: vec![0.0f32; config.n_embd],
            #[cfg(feature = "tf_loop")]
            tf_y_buf: vec![0.0f32; config.n_embd],
            #[cfg(feature = "tf_loop")]
            tf_stash_x: vec![0.0f32; config.n_embd],
        }
    }

    /// Reset quantized KV cache incremental dequant state.
    /// Call when starting a new sequence or after cache reset.
    pub fn reset_dequant(&mut self) {
        self.dequant_pos.fill(0);
    }

    /// Backward-compat alias for [`reset_dequant`].
    #[cfg(feature = "turboquant")]
    pub fn reset_tq_dequant(&mut self) {
        self.reset_dequant();
    }

    /// Perform delta routing using pre-allocated index buffer (avoids Vec::new() per call).
    /// Collects block delta indices, then calls depth_route_with_deltas.
    #[cfg(feature = "delta_routing")]
    pub(crate) fn depth_route_blocks(
        &mut self,
        block_idx: usize,
        layer_idx: usize,
        query_weight: &[f32],
        norm_weight: &[f32],
        n_embd: usize,
        _weights: &TransformerWeights,
    ) {
        // Collect source indices (reuse pre-allocated buffer)
        self.delta_source_indices.clear();
        for prev_block in 0..=block_idx {
            if prev_block < self.block_deltas.len() {
                self.delta_source_indices.push(prev_block);
            }
        }

        // Call depth_route directly using pre-gathered indices
        depth_route_with_indices(DepthRouteIndicesArgs {
            residual: &mut self.x[..n_embd],
            block_deltas: &self.block_deltas,
            source_indices: &self.delta_source_indices,
            query_weight,
            norm_weight,
            logits_buf: &mut self.delta_routing_logits,
            scaled_buf: &mut self.delta_scaled_buf,
            n_embd,
        });

        // Reset current block delta
        self.block_deltas[block_idx].fill(0.0);
        let _ = layer_idx; // suppress unused warning
    }
}

// ---------------------------------------------------------------------------
// PrefillContext — Pre-allocated buffers for bidirectional prefill (Plan 025)
// ---------------------------------------------------------------------------

/// Pre-allocated context for bidirectional prefill phase.
/// Created once at startup, reused across all requests. Zero alloc in request path.
pub struct PrefillContext {
    /// Hidden states for all prompt positions, carried between layers.
    /// Size: [max_prompt_len × n_embd]. Only used when n_layer > 1.
    /// For n_layer == 1, embeddings are computed on-the-fly and this buffer is unused.
    hidden: Vec<f32>,
    /// Pre-computed Q projections from fused Phase A, reused in Phase B.
    /// Size: [max_prompt_len × n_embd]. Eliminates redundant hidden load + rmsnorm + Q matmul.
    queries: Vec<f32>,
    /// Pre-computed attention residuals (xr) from fused Phase A, reused in Phase B.
    /// Size: [max_prompt_len × n_embd]. Eliminates redundant hidden load + first rmsnorm.
    residuals: Vec<f32>,
    /// LoRA intermediate buffer. Size: [lora_rank].
    /// Reused for every LoRA application across all projections.
    lora_buf: Vec<f32>,
    /// Max prompt length this context supports (= config.block_size).
    max_prompt_len: usize,
}

impl PrefillContext {
    pub fn new(config: &Config) -> Self {
        let block_embd = config.block_size * config.n_embd;
        Self {
            hidden: vec![0.0; block_embd],
            queries: vec![0.0; block_embd],
            residuals: vec![0.0; block_embd],
            lora_buf: vec![0.0; config.lora_rank],
            max_prompt_len: config.block_size,
        }
    }
}

/// Fused attention head with GQA support: score → softmax → weighted value sum.
/// Avoids separate `softmax()` call and write-back of normalized scores.
///
/// GQA: each Q head (`q_head_offset / hd`) maps to a KV group (`kv_group_offset / hd`).
/// When `n_kv_head == n_head`, `kv_group_offset == q_head_offset` and `kv_dim == n_embd`
/// → identical to standard MHA (backward compatible).
///
/// SAFETY: caller must ensure all indices are in bounds.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn attention_head(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    scores_buf: &mut [f32],
    q_head_offset: usize,
    kv_group_offset: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
    scale: f32,
) {
    // Pass 1: compute Q·K scores and find max for numerical stability
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let k_off = t * kv_dim + kv_group_offset;
        // SAFETY: q_head_offset + hd <= n_embd (head_dim * n_head), k_off + hd <= block_size * kv_dim
        let dot = unsafe {
            let q_slice = std::slice::from_raw_parts(q.as_ptr().add(q_head_offset), hd);
            let k_slice = std::slice::from_raw_parts(key_cache.as_ptr().add(k_off), hd);
            crate::simd::simd_dot_f32(q_slice, k_slice, hd)
        };
        let score = dot * scale;
        unsafe {
            *scores_buf.get_unchecked_mut(t) = score;
        }
        if score > max_score {
            max_score = score;
        }
    }

    // Pass 2: exp(scores - max) and accumulate sum
    // Shift scores by max using SIMD broadcast add, then SIMD exp
    let scores_slice = unsafe { std::slice::from_raw_parts_mut(scores_buf.as_mut_ptr(), t_n) };
    crate::simd::simd_add_scalar_inplace(scores_slice, -max_score);
    crate::simd::simd_exp_inplace(scores_slice);
    let sum: f32 = crate::simd::simd_sum_f32(scores_slice);

    // Pass 3: normalize + weighted value accumulation (no write-back of scores)
    // Pre-scale scores once using SIMD
    let inv_sum = 1.0 / sum;
    crate::simd::simd_scale_inplace(scores_slice, inv_sum);
    // Zero the output slice before accumulation
    attn_out[q_head_offset..q_head_offset + hd].fill(0.0);
    // Accumulate: t outer → contiguous value_cache row access
    for t in 0..t_n {
        let s = unsafe { *scores_buf.get_unchecked(t) };
        let v_row = unsafe {
            std::slice::from_raw_parts(value_cache.as_ptr().add(t * kv_dim + kv_group_offset), hd)
        };
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(attn_out.as_mut_ptr().add(q_head_offset), hd) };
        crate::simd::simd_fused_scale_acc(out_slice, v_row, s, hd);
    }
}

/// Causal decode: single token forward with optional LoRA adapter.
/// Backward-compatible wrapper that passes `None` for LoRA.
#[inline(always)]
pub fn forward<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    cache.advance_pos(pos);
    #[cfg(feature = "coda_fusion")]
    {
        #[cfg(not(feature = "domain_latent"))]
        {
            forward_coda(ctx, weights, cache, token, pos, config, None)
        }
        #[cfg(feature = "domain_latent")]
        {
            forward_coda(ctx, weights, cache, token, pos, config, None, None)
        }
    }
    #[cfg(not(feature = "coda_fusion"))]
    {
        #[cfg(not(feature = "domain_latent"))]
        {
            forward_base(ctx, weights, cache, token, pos, config, None)
        }
        #[cfg(feature = "domain_latent")]
        {
            forward_base(ctx, weights, cache, token, pos, config, None, None)
        }
    }
}

/// Forward with optional LoRA and domain latent (Plan 038).
/// Convenience wrapper for callers that need both conditioning signals.
#[cfg(feature = "domain_latent")]
#[allow(clippy::too_many_arguments)]
pub fn forward_with_domain_latent<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    cache.advance_pos(pos);
    #[cfg(feature = "coda_fusion")]
    {
        forward_coda(ctx, weights, cache, token, pos, config, lora, domain_latent)
    }
    #[cfg(not(feature = "coda_fusion"))]
    {
        forward_base(ctx, weights, cache, token, pos, config, lora, domain_latent)
    }
}

/// Stage-specialized forward pass (Plan 102: TileRT pipeline).
///
/// `Draft` stage: skips screening pruner, reduces KV cache writes for positions beyond draft length.
/// `Verify` stage: exact attention with full KV write.
/// `Prefill` and `Sample` fall through to standard `forward()`.
#[cfg(feature = "decode_specialize")]
#[allow(clippy::too_many_arguments)]
pub fn forward_decode_stage<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    stage: DecodeStage,
) -> &'a mut [f32] {
    cache.advance_pos(pos);
    match stage {
        DecodeStage::Draft => forward_draft(ctx, weights, cache, token, pos, config),
        DecodeStage::Verify => forward_verify(ctx, weights, cache, token, pos, config),
        DecodeStage::Prefill | DecodeStage::Sample => {
            // Fall through to standard forward — prefill/sample don't benefit from specialization
            #[cfg(not(feature = "domain_latent"))]
            {
                forward_base(ctx, weights, cache, token, pos, config, None)
            }
            #[cfg(feature = "domain_latent")]
            {
                forward_base(ctx, weights, cache, token, pos, config, None, None)
            }
        }
    }
}

/// Draft-optimized forward: same as forward_base but marks the stage for profiling.
/// Currently identical to forward() — the optimization surface is skipping screening
/// and reducing KV writes, which requires deeper integration with the speculative step.
#[cfg(feature = "decode_specialize")]
#[allow(clippy::too_many_arguments)]
fn forward_draft<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    // Draft path: identical to forward_base for correctness.
    // Future optimization: skip screening, approximate attention, reduced KV writes.
    #[cfg(not(feature = "domain_latent"))]
    {
        forward_base(ctx, weights, cache, token, pos, config, None)
    }
    #[cfg(feature = "domain_latent")]
    {
        forward_base(ctx, weights, cache, token, pos, config, None, None)
    }
}

/// Verify-optimized forward: exact attention with full KV write.
#[cfg(feature = "decode_specialize")]
#[allow(clippy::too_many_arguments)]
fn forward_verify<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    // Verify path: identical to forward_base for correctness.
    // This is the "exact" path — full KV write, no approximations.
    #[cfg(not(feature = "domain_latent"))]
    {
        forward_base(ctx, weights, cache, token, pos, config, None)
    }
    #[cfg(feature = "domain_latent")]
    {
        forward_base(ctx, weights, cache, token, pos, config, None, None)
    }
}

// ---------------------------------------------------------------------------
// LT2 Looped Inference (Plan 108, Research 73)
// ---------------------------------------------------------------------------

/// Looped transformer forward pass — weight-shared T-pass loop.
///
/// Applies the same layer weights T times in succession, yielding effective
/// depth T×n_layer with no extra parameters. Key insight from LT2: looping
/// uniquely synergizes with subquadratic attention — T loops turn rank-1
/// DPLR state updates into rank-T updates.
///
/// Per-loop residual gate: h^(τ) = h̃^(τ) + ρ_τ ⊙ h^(τ-1)
/// Zero-init ρ_τ means first iteration is h̃^(1) (no residual from "previous").
///
/// Feature gate: `lt2_looped` (requires `hla_attention`).
#[cfg(feature = "lt2_looped")]
#[allow(dead_code, clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn forward_looped<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    ahla_cache: &mut crate::hla::MultiLayerAhlaCache,
    token: usize,
    pos: usize,
    config: &Config,
    residual_gate: &crate::types::ResidualGate,
    sdpa_gate: &crate::types::SdpaOutputGate,
) -> &'a mut [f32] {
    cache.advance_pos(pos);
    use crate::types::{HybridPattern, LoopMode};

    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = crate::types::kv_dim(config);

    let loop_count = match config.loop_mode {
        LoopMode::WeightShared { loop_count } => loop_count,
        LoopMode::None => 1,
        LoopMode::TrainingFree => 1,
    };

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    crate::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // 2. Outer loop: T passes over all layers
    for tau in 0..loop_count {
        // Save h^(τ-1) for residual gate
        ctx.prev_h[..n].copy_from_slice(&ctx.x[..n]);

        // 3. Inner loop: weight-shared layer pass
        for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
            let layer_cache = &mut cache.layers[layer_idx];

            // Determine if this layer uses full SDPA or linear attention
            let is_full = match config.hybrid_pattern {
                HybridPattern::Uniform => true,
                HybridPattern::Interleave { full_ratio } => {
                    (layer_idx % full_ratio) == full_ratio - 1
                }
                HybridPattern::Bookend => layer_idx == 0 || layer_idx == weights.layers.len() - 1,
            };

            // Pre-attention: RMSNorm → save residual
            crate::types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

            // QKV projections
            crate::types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            crate::types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            crate::types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

            if is_full {
                // Full SDPA: store K,V in cache and compute standard attention
                let pos_off = pos * kvd;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ctx.k.as_ptr(),
                        layer_cache.key.as_mut_ptr().add(pos_off),
                        kvd,
                    );
                    std::ptr::copy_nonoverlapping(
                        ctx.v.as_ptr(),
                        layer_cache.value.as_mut_ptr().add(pos_off),
                        kvd,
                    );
                }

                // Multi-head attention with GQA
                let scale = ctx.attn_scale;
                let t_n = pos + 1;
                for h in 0..config.n_head {
                    let kv_group = ctx.kv_group_lut[h];
                    unsafe {
                        attention_head(
                            &ctx.q,
                            &layer_cache.key,
                            &layer_cache.value,
                            &mut ctx.attn_out,
                            &mut ctx.scores,
                            h * hd,
                            kv_group * hd,
                            kvd,
                            hd,
                            t_n,
                            scale,
                        );
                    }
                }
            } else {
                // Linear attention via AHLA recurrent step
                let ahla_layer = &mut ahla_cache.layers[layer_idx];
                ctx.attn_out[..n].fill(0.0);

                for h in 0..config.n_head {
                    let kv_group = ctx.kv_group_lut[h];
                    let head_state = &mut ahla_layer.heads[h];

                    crate::hla::ahla_step(
                        &mut ahla_layer.pkv[kv_group],
                        &mut ahla_layer.mk[kv_group],
                        head_state,
                        &ctx.q[h * hd..(h + 1) * hd],
                        &ctx.k[kv_group * hd..(kv_group + 1) * hd],
                        &ctx.v[kv_group * hd..(kv_group + 1) * hd],
                        hd,
                        ahla_cache.gamma,
                        &mut ctx.attn_out[h * hd..(h + 1) * hd],
                        &mut ctx.scores[..hd],
                    );
                }
            }

            // SDPA output gate (if configured): sigmoid(W_gate @ attn_out) ⊙ attn_out
            // Zero-init weights → sigmoid(0) = 0.5 (neutral half-pass).
            // Paper: +0.3–0.5 avg points on zero-shot benchmarks.
            if config.gated_attn && is_full {
                sdpa_gate.forward(&mut ctx.attn_out[..n], n, &mut ctx.scores[..n]);
            }

            // Output projection + residual
            crate::types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
            crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

            // MLP: save residual → RMSNorm → MLP → residual
            ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);
            crate::types::matmul_relu(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            crate::types::matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
            );
            crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);
        }

        // Per-loop residual gate: h^(τ) = h̃^(τ) + ρ_τ ⊙ h^(τ-1)
        // ρ_τ is zero-init → first iteration: h^(0) = h̃^(0) (no residual)
        if tau > 0 {
            let gate_offset = tau * n;
            if gate_offset + n <= residual_gate.gates.len() {
                for d in 0..n {
                    ctx.x[d] += residual_gate.gates[gate_offset + d] * ctx.prev_h[d];
                }
            }
        }
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    standard_lm_head(
        &mut ctx.logits,
        &ctx.x,
        &weights.lm_head,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

// ---------------------------------------------------------------------------
// Training-Free Loop Wrapper (Plan 136, Research 94)
// ---------------------------------------------------------------------------

/// Training-free loop forward pass — ODE-refined sub-stepping over a window.
///
/// Pure inference-time retrofit: re-applies a contiguous mid-stack block of
/// layers K times with damped sub-stepping and anchor blending. No training needed.
///
/// # Algorithm (block-mode)
///
/// ```text
/// 1. Embedding: x = wte[token] + wpe[pos]
/// 2. Pre-loop:  for layer 0..window_start:  standard forward, write KV
/// 3. Anchor:    forward window once → x_anchor
/// 4. Loop K times:
///      a. Forward window layers
///      b. Sub-step: x += (1/K)·(y − x)  [damped Euler]
/// 5. Blend with anchor: x = β·x_anchor + (1−β)·x
/// 6. Stash:     single forward through window writes canonical KV
/// 7. Post-loop: for layer window_end+1..n_layer: standard forward, write KV
/// 8. LM head
/// ```
#[cfg(feature = "tf_loop")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub fn forward_training_free_loop<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    tf_config: &TrainingFreeLoopConfig,
) -> &'a mut [f32] {
    cache.advance_pos(pos);
    use crate::tf_loop::{anchor_blend, sub_step_damped_euler};
    use katgpt_core::types::{CacheStrategy, IterationMode, SubStepStrategy};

    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;
    let n_layer = weights.layers.len();
    let window_start = tf_config.window_start.min(n_layer);
    let window_end = tf_config.window_end.min(n_layer - 1);
    let k = tf_config.loop_count;
    let beta = match tf_config.strategy {
        SubStepStrategy::DampedEuler => 0.0, // no anchor blend for pure Euler
        SubStepStrategy::KStageRK { beta } => beta,
    };

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    crate::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // 2. Pre-loop layers: standard forward with KV writes
    for (layer_idx, layer_weights) in weights.layers[..window_start].iter().enumerate() {
        forward_single_layer(
            ctx,
            layer_weights,
            &mut cache.layers[layer_idx],
            pos,
            config,
            n,
            hd,
            kvd,
            n_kv,
        );
    }

    // Save state before window for anchor computation
    ctx.tf_x_pre_window[..n].copy_from_slice(&ctx.x[..n]);

    // 3. Anchor: forward window once to get x_anchor
    if beta > 0.0 {
        for layer_idx in window_start..=window_end {
            forward_single_layer(
                ctx,
                &weights.layers[layer_idx],
                &mut cache.layers[layer_idx],
                pos,
                config,
                n,
                hd,
                kvd,
                n_kv,
            );
        }
        ctx.tf_x_anchor[..n].copy_from_slice(&ctx.x[..n]);
        // Restore x to pre-window state for loop iterations
        ctx.x[..n].copy_from_slice(&ctx.tf_x_pre_window[..n]);
    }

    // Temp buffer for window output (pre-allocated on ForwardContext)
    ctx.tf_y_buf[..n].fill(0.0);

    // 4. Loop K times over the window with sub-stepping
    match tf_config.iteration_mode {
        IterationMode::Block => {
            for _ in 0..k {
                // Forward through window layers
                for layer_idx in window_start..=window_end {
                    forward_single_layer(
                        ctx,
                        &weights.layers[layer_idx],
                        &mut cache.layers[layer_idx],
                        pos,
                        config,
                        n,
                        hd,
                        kvd,
                        n_kv,
                    );
                }
                // Save window output
                ctx.tf_y_buf[..n].copy_from_slice(&ctx.x[..n]);
                // Restore x to pre-window for sub-step computation
                ctx.x[..n].copy_from_slice(&ctx.tf_x_pre_window[..n]);
                // Apply sub-step: x += (1/K)·(y − x)
                sub_step_damped_euler(&mut ctx.x[..n], &ctx.tf_y_buf[..n], k);
            }
        }
        IterationMode::Layer => {
            for _ in 0..k {
                for layer_idx in window_start..=window_end {
                    // Forward single layer
                    forward_single_layer(
                        ctx,
                        &weights.layers[layer_idx],
                        &mut cache.layers[layer_idx],
                        pos,
                        config,
                        n,
                        hd,
                        kvd,
                        n_kv,
                    );
                    // Sub-step per layer
                    ctx.tf_y_buf[..n].copy_from_slice(&ctx.x[..n]);
                    ctx.x[..n].copy_from_slice(&ctx.tf_x_pre_window[..n]);
                    sub_step_damped_euler(&mut ctx.x[..n], &ctx.tf_y_buf[..n], k);
                }
            }
        }
    }

    // 5. Blend with anchor
    if beta > 0.0 {
        anchor_blend(&mut ctx.x[..n], &ctx.tf_x_anchor[..n], beta);
    }

    // 6. Stash: single forward through window writes canonical KV entries
    {
        ctx.tf_stash_x[..n].copy_from_slice(&ctx.x[..n]);
        match tf_config.cache_strategy {
            CacheStrategy::Last => {
                // Forward with final state → writes KV
                for layer_idx in window_start..=window_end {
                    forward_single_layer(
                        ctx,
                        &weights.layers[layer_idx],
                        &mut cache.layers[layer_idx],
                        pos,
                        config,
                        n,
                        hd,
                        kvd,
                        n_kv,
                    );
                }
            }
            CacheStrategy::First => {
                // Forward with pre-window state → writes KV
                ctx.x[..n].copy_from_slice(&ctx.tf_x_pre_window[..n]);
                for layer_idx in window_start..=window_end {
                    forward_single_layer(
                        ctx,
                        &weights.layers[layer_idx],
                        &mut cache.layers[layer_idx],
                        pos,
                        config,
                        n,
                        hd,
                        kvd,
                        n_kv,
                    );
                }
                // Restore the blended state
                ctx.x[..n].copy_from_slice(&ctx.tf_stash_x[..n]);
            }
        }
    }

    // 7. Post-loop layers: standard forward with KV writes
    for layer_idx in (window_end + 1)..n_layer {
        forward_single_layer(
            ctx,
            &weights.layers[layer_idx],
            &mut cache.layers[layer_idx],
            pos,
            config,
            n,
            hd,
            kvd,
            n_kv,
        );
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // 8. LM Head
    standard_lm_head(
        &mut ctx.logits,
        &ctx.x,
        &weights.lm_head,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Single transformer layer forward: attention + MLP with KV cache write.
///
/// Extracted from `forward_base` to be reusable by both standard and looped paths.
#[cfg(feature = "tf_loop")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn forward_single_layer(
    ctx: &mut ForwardContext,
    layer_weights: &LayerWeights,
    layer_cache: &mut KVCache,
    pos: usize,
    config: &Config,
    n: usize,
    hd: usize,
    kvd: usize,
    _n_kv: usize,
) {
    // Pre-attention: RMSNorm → save residual
    types::rmsnorm(&mut ctx.x);
    ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

    // QKV projections
    types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
    types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
    types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

    // Store K,V in cache
    let pos_off = pos * kvd;
    unsafe {
        std::ptr::copy_nonoverlapping(
            ctx.k.as_ptr(),
            layer_cache.key.as_mut_ptr().add(pos_off),
            kvd,
        );
        std::ptr::copy_nonoverlapping(
            ctx.v.as_ptr(),
            layer_cache.value.as_mut_ptr().add(pos_off),
            kvd,
        );
    }

    // Multi-head attention with GQA
    let scale = ctx.attn_scale;
    let t_n = pos + 1;
    for h in 0..config.n_head {
        let kv_group = ctx.kv_group_lut[h];
        unsafe {
            attention_head(
                &ctx.q,
                &layer_cache.key,
                &layer_cache.value,
                &mut ctx.attn_out,
                &mut ctx.scores,
                h * hd,
                kv_group * hd,
                kvd,
                hd,
                t_n,
                scale,
            );
        }
    }

    // Output projection + residual
    types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
    crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

    // MLP: save residual → RMSNorm → MLP → residual
    ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
    types::rmsnorm(&mut ctx.x);
    types::matmul_relu(
        &mut ctx.hidden,
        &layer_weights.mlp_w1,
        &ctx.x,
        config.mlp_hidden,
        n,
    );
    types::matmul(
        &mut ctx.x,
        &layer_weights.mlp_w2,
        &ctx.hidden,
        n,
        config.mlp_hidden,
    );
    crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);
}

/// Standard full-vocab LM head (current behavior).
#[inline(always)]
fn standard_lm_head(
    logits: &mut [f32],
    hidden: &[f32],
    lm_head: &[f32],
    vocab_size: usize,
    n_embd: usize,
) {
    // matmul_parallel has an internal threshold (512 rows) — for small vocab
    // it falls back to serial automatically. For vocab_size >= 512 (e.g.
    // small_target vocab=4096), this parallelizes across rayon threads.
    matmul_parallel(logits, lm_head, hidden, vocab_size, n_embd);
}

/// Select top-K indices from scores (Plan 117 T25).
///
/// Uses partial selection sort: O(N + K log K).
/// Returns indices sorted by score descending (highest first).
///
/// **Note:** This function allocates internally and is intended for tests/benchmarks only.
/// For hot-path code, use [`select_topk_indices_into_buf`] which reuses pre-allocated buffers.
pub fn select_topk_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(scores.len());
    if k == 0 {
        return Vec::new();
    }

    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();

    // Partial sort to partition top K (unstable, O(N))
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Sort the top K by score descending (O(K log K))
    indexed[..k].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    indexed[..k].iter().map(|(i, _)| *i).collect()
}

/// In-place variant of [`select_topk_indices`] that reuses pre-allocated buffers.
/// Writes top-K indices into `output_buf` (cleared and filled).
///
/// This is the preferred variant for hot-path code — no heap allocations.
pub fn select_topk_indices_into_buf(
    scores: &[f32],
    k: usize,
    indexed_buf: &mut Vec<(usize, f32)>,
    output_buf: &mut Vec<usize>,
) {
    let k = k.min(scores.len());
    if k == 0 {
        output_buf.clear();
        return;
    }

    indexed_buf.clear();
    indexed_buf.extend(scores.iter().copied().enumerate());

    indexed_buf.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    indexed_buf[..k].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    output_buf.clear();
    output_buf.extend(indexed_buf[..k].iter().map(|(i, _)| *i));
}

/// Two-stage clustered LM head for large vocabularies.
///
/// Stage 1: predict cluster ID(s) via classifier matmul + top-K selection.
/// Stage 2: compute exact logits only for tokens in the selected clusters.
///
/// When `topk=1`, behavior is identical to single-cluster argmax (backward compat).
/// When `topk >= num_clusters`, all clusters are selected (no pruning).
///
/// Only called when `vocab_size >= mtp_cluster_vocab_threshold` AND
/// cluster weights are available.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn clustered_lm_head(
    logits: &mut [f32],
    hidden: &[f32],
    lm_head: &[f32],
    cluster_classifier: &[f32],
    cluster_map: &[Vec<usize>],
    vocab_size: usize,
    n_embd: usize,
    topk: usize,
    cluster_scores_buf: &mut [f32],
    topk_indexed_buf: &mut Vec<(usize, f32)>,
    topk_output_buf: &mut Vec<usize>,
) {
    let num_clusters = cluster_map.len();

    // Stage 1: compute cluster scores (reuse pre-allocated buffer)
    let cluster_scores = &mut cluster_scores_buf[..num_clusters];
    cluster_scores.fill(0.0f32);
    for (c, score) in cluster_scores.iter_mut().enumerate() {
        let row_off = c * n_embd;
        *score = crate::simd::simd_dot_f32(
            &cluster_classifier[row_off..row_off + n_embd],
            &hidden[..n_embd],
            n_embd,
        );
    }

    // Select top-K clusters (Plan 117 T27: skip selection if topk >= num_clusters)
    let selected_clusters: &[usize] = if topk >= num_clusters {
        // Fill output_buf with all cluster indices
        topk_output_buf.clear();
        topk_output_buf.extend(0..num_clusters);
        topk_output_buf
    } else {
        select_topk_indices_into_buf(cluster_scores, topk, topk_indexed_buf, topk_output_buf);
        topk_output_buf
    };

    // Stage 2: fill all logits with -inf, then compute exact for selected clusters
    logits.fill(f32::NEG_INFINITY);

    // NOTE(078): Cluster tokens are non-contiguous (round-robin assignment), so
    // batched simd_matmul_rows cannot be used directly. Individual simd_dot_f32 calls
    // are optimal here — the function is inlined and dispatch overhead is negligible.
    for &cluster_idx in selected_clusters {
        let cluster_tokens = &cluster_map[cluster_idx];
        for &token_idx in cluster_tokens {
            if token_idx < vocab_size {
                let row_off = token_idx * n_embd;
                let dot = crate::simd::simd_dot_f32(
                    &lm_head[row_off..row_off + n_embd],
                    &hidden[..n_embd],
                    n_embd,
                );
                unsafe {
                    *logits.get_unchecked_mut(token_idx) = dot;
                }
            }
        }
    }
}

/// Create a round-robin cluster assignment for tokens.
///
/// Token `i` is assigned to cluster `i / cluster_size`.
/// Deterministic, no training needed — simple baseline.
pub fn cluster_map_round_robin(vocab_size: usize, cluster_size: usize) -> Vec<Vec<usize>> {
    let num_clusters = vocab_size.div_ceil(cluster_size);
    let mut map: Vec<Vec<usize>> = (0..num_clusters)
        .map(|_| Vec::with_capacity(cluster_size))
        .collect();
    for token_id in 0..vocab_size {
        let cluster_id = token_id / cluster_size;
        map[cluster_id].push(token_id);
    }
    map
}

/// Create cluster assignment from embedding similarity (K-means style).
///
/// Groups tokens with similar embeddings together for efficient LM head computation.
/// Current implementation: round-robin baseline.
/// TODO: implement actual K-means using embedding cosine similarity (Plan 056: riir-burner).
pub fn cluster_map_from_embeddings(
    _wte: &[f32],
    vocab_size: usize,
    _n_embd: usize,
    cluster_size: usize,
) -> Vec<Vec<usize>> {
    cluster_map_round_robin(vocab_size, cluster_size)
}

/// Delta routing: softmax over delta sources, additive to residual (Plan 097).
///
/// depth_route(sources, residual, proj, norm):
///   V = stack(sources)          // [N, D]
///   K = norm(V)                  // RMSNorm
///   logits = dot(proj_weight, K) // per-source score
///   weights = softmax(logits)    // routing weights
///   return residual + weighted_sum(weights, V)  // additive
///
/// ## Stability analysis (Plan 134, MGR paper §3.2 — arXiv:2605.23259)
///
/// The MGR paper proves that convex-combination residual updates (lerp gates)
/// guarantee bounded activation norms: `x_{l+1} = (1-α)·x_l + α·f(x_l)`.
///
/// **Our routing is NOT a convex combination.** It is additive:
/// `residual += Σ_i w_i · V_i`, where `w_i = softmax(...)` and `Σ w_i = 1`.
/// Since softmax weights sum to 1 but are applied to arbitrary source vectors (not
/// the residual itself), the MGR convex-combination stability guarantee does not
/// formally apply.
///
/// Practical stability comes from two normalization mechanisms:
/// - **RMSNorm** bounds the input scale to the routing logits, preventing
///   exploding score magnitudes.
/// - **Softmax normalization** ensures routing weights are non-negative and sum
///   to 1, so the weighted sum cannot exceed the convex hull of source vectors.
///
/// Unlike MGR's convex lerp, norms *can* still grow layer-to-layer (each additive
/// step contributes additional magnitude). However, empirical testing across 36+
/// layers shows bounded growth: `‖x_L‖ ≤ 10 × ‖x_0‖` (see
/// `proof_depth_route_norm_stability` test).
///
/// ## MGR Eq. 14 — lerp gate bias initialization
///
/// If a convex-combination lerp gate were ever added (e.g. for training), the
/// MGR paper recommends initializing the gate bias as:
///
///   b_l = log(1 - 1/L)
///
/// where L is the total number of layers. For L=36, b_l ≈ -0.0285.
/// This encourages near-identity routing at initialization.
#[cfg(feature = "delta_routing")]
#[allow(dead_code, clippy::needless_range_loop)]
#[inline(always)]
fn depth_route(
    residual: &mut [f32],
    sources: &[&[f32]],     // N delta vectors, each [n_embd]
    query_weight: &[f32],   // [n_embd] per-layer query
    norm_weight: &[f32],    // [n_embd] RMSNorm gamma
    logits_buf: &mut [f32], // [N] temp buffer
    scaled_buf: &mut [f32], // [n_embd] scratch for SIMD dot
    n_embd: usize,
) {
    let n_sources = sources.len();
    if n_sources == 0 {
        return;
    }

    // 1. RMSNorm each source and compute dot product with query
    let eps = 1e-5f32;
    let mut max_logit = f32::NEG_INFINITY;

    for (i, &src) in sources.iter().enumerate() {
        // SIMD sum-of-squares for RMSNorm
        let sum_sq = crate::simd::simd_sum_sq(&src[..n_embd], n_embd);
        let rms = (sum_sq / n_embd as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Scale src * inv_rms * norm_weight into scratch via fused SIMD, then dot with query
        scaled_buf[..n_embd].copy_from_slice(&src[..n_embd]);
        crate::simd::simd_scale_mul_inplace(
            &mut scaled_buf[..n_embd],
            &norm_weight[..n_embd],
            inv_rms,
        );
        let logit = crate::simd::simd_dot_f32(&scaled_buf[..n_embd], query_weight, n_embd);

        logits_buf[i] = logit;
        if logit > max_logit {
            max_logit = logit;
        }
    }

    // 2. Softmax (numerically stable)
    let mut sum_exp = 0.0f32;
    for i in 0..n_sources {
        let exp_val = (logits_buf[i] - max_logit).exp();
        logits_buf[i] = exp_val;
        sum_exp += exp_val;
    }
    let inv_sum = 1.0 / sum_exp;

    // 3. Weighted sum of sources, added to residual (additive routing)
    //    For each source: scale into scratch buf then SIMD-accumulate into residual
    for (i, &src) in sources.iter().enumerate() {
        let weight = logits_buf[i] * inv_sum;
        scaled_buf[..n_embd].copy_from_slice(&src[..n_embd]);
        crate::simd::simd_scale_inplace(&mut scaled_buf[..n_embd], weight);
        crate::simd::simd_add_inplace(&mut residual[..n_embd], &scaled_buf[..n_embd]);
    }
}

/// Delta routing variant that takes block deltas and indices directly.
/// Avoids allocating a `Vec<&[f32]>` for source_refs by indexing into `block_deltas`.
#[cfg(feature = "delta_routing")]
struct DepthRouteIndicesArgs<'a> {
    residual: &'a mut [f32],
    block_deltas: &'a [Vec<f32>],
    source_indices: &'a [usize],
    query_weight: &'a [f32],   // [n_embd] per-layer query
    norm_weight: &'a [f32],    // [n_embd] RMSNorm gamma
    logits_buf: &'a mut [f32], // [N] temp buffer
    scaled_buf: &'a mut [f32], // [n_embd] scratch for SIMD dot
    n_embd: usize,
}

#[cfg(feature = "delta_routing")]
fn depth_route_with_indices(args: DepthRouteIndicesArgs<'_>) {
    let DepthRouteIndicesArgs {
        residual,
        block_deltas,
        source_indices,
        query_weight,
        norm_weight,
        logits_buf,
        scaled_buf,
        n_embd,
    } = args;

    let n_sources = source_indices.len();
    if n_sources == 0 {
        return;
    }

    // 1. RMSNorm each source and compute dot product with query
    let eps = 1e-5f32;
    let mut max_logit = f32::NEG_INFINITY;

    for (i, &src_idx) in source_indices.iter().enumerate() {
        let src = &block_deltas[src_idx];
        // SIMD sum-of-squares for RMSNorm
        let sum_sq = crate::simd::simd_sum_sq(&src[..n_embd], n_embd);
        let rms = (sum_sq / n_embd as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Scale src * inv_rms * norm_weight into scratch via fused SIMD, then dot with query
        scaled_buf[..n_embd].copy_from_slice(&src[..n_embd]);
        crate::simd::simd_scale_mul_inplace(
            &mut scaled_buf[..n_embd],
            &norm_weight[..n_embd],
            inv_rms,
        );
        let logit = crate::simd::simd_dot_f32(&scaled_buf[..n_embd], query_weight, n_embd);

        logits_buf[i] = logit;
        if logit > max_logit {
            max_logit = logit;
        }
    }

    // 2. Softmax (numerically stable)
    let mut sum_exp = 0.0f32;
    for val in logits_buf.iter_mut().take(n_sources) {
        let exp_val = (*val - max_logit).exp();
        *val = exp_val;
        sum_exp += exp_val;
    }
    let inv_sum = 1.0 / sum_exp;

    // 3. Weighted sum of sources, added to residual (additive routing)
    //    For each source: scale into scratch buf then SIMD-accumulate into residual
    for (i, &src_idx) in source_indices.iter().enumerate() {
        let src = &block_deltas[src_idx];
        let weight = logits_buf[i] * inv_sum;
        scaled_buf[..n_embd].copy_from_slice(&src[..n_embd]);
        crate::simd::simd_scale_inplace(&mut scaled_buf[..n_embd], weight);
        crate::simd::simd_add_inplace(&mut residual[..n_embd], &scaled_buf[..n_embd]);
    }
}

/// Compute delta routing softmax weights without modifying residual (Plan 097 T8).
///
/// Returns the routing weight distribution over sources for inspection.
/// Used by GOAT sharpness tests to verify max_weight ≥ 0.4 in deep layers.
#[cfg(feature = "delta_routing")]
#[allow(clippy::needless_range_loop)]
pub fn depth_route_weights(
    sources: &[&[f32]],   // N delta vectors, each [n_embd]
    query_weight: &[f32], // [n_embd] per-layer query
    norm_weight: &[f32],  // [n_embd] RMSNorm gamma
    n_embd: usize,
) -> Vec<f32> {
    let n_sources = sources.len();
    if n_sources == 0 {
        return Vec::new();
    }

    let eps = 1e-5f32;
    let mut logits = vec![0.0f32; n_sources];
    let mut scaled = vec![0.0f32; n_embd];
    let mut max_logit = f32::NEG_INFINITY;

    // 1. RMSNorm each source and compute dot product with query
    for (i, &src) in sources.iter().enumerate() {
        // SIMD sum-of-squares for RMSNorm
        let sum_sq = crate::simd::simd_sum_sq(&src[..n_embd], n_embd);
        let rms = (sum_sq / n_embd as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Scale src * inv_rms * norm_weight into scratch via fused SIMD, then dot with query
        scaled[..n_embd].copy_from_slice(&src[..n_embd]);
        crate::simd::simd_scale_mul_inplace(&mut scaled[..n_embd], &norm_weight[..n_embd], inv_rms);
        let logit = crate::simd::simd_dot_f32(&scaled[..n_embd], query_weight, n_embd);

        logits[i] = logit;
        if logit > max_logit {
            max_logit = logit;
        }
    }

    // 2. Softmax
    let mut sum_exp = 0.0f32;
    for logit in &mut logits {
        *logit = (*logit - max_logit).exp();
        sum_exp += *logit;
    }
    let inv_sum = 1.0 / sum_exp;
    for logit in &mut logits {
        *logit *= inv_sum;
    }

    logits
}

/// Internal forward with optional LoRA and domain latent (writer LoRA during decode).
/// Zero-alloc forward pass. Writes logits into `ctx.logits` and returns &mut to it.
/// Multi-layer: RMSNorm → Attn → Res → RMSNorm → MLP → Res per layer, then LM Head.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn forward_base<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let _n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    crate::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // MLS: reset accumulator at start of forward call (Plan 104)
    #[cfg(feature = "mls_aggregate")]
    {
        ctx.mls_buf[..n].fill(0.0);
        ctx.mls_count = 0;
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // MLS: save pre-layer state for delta computation (Plan 104)
        #[cfg(feature = "mls_aggregate")]
        if config.mls_layers > 0 && layer_idx >= weights.layers.len() - config.mls_layers {
            ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
        }

        // Pre-attention: RMSNorm → save residual
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // QKV projections from per-layer weights (GQA: K/V produce kv_dim outputs)
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut ctx.lora_buf);
        }
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut ctx.lora_buf);
        }
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut ctx.lora_buf);
        }

        // Domain latent injection at mid-layer (Plan 038: Free Transformer adaptation)
        #[cfg(feature = "domain_latent")]
        if layer_idx == config.n_layer / 2
            && let Some(dl) = domain_latent
        {
            crate::simd::simd_add_inplace(&mut ctx.k[..kvd], &dl.embedding[..kvd]);
            crate::simd::simd_add_inplace(&mut ctx.v[..kvd], &dl.embedding[..kvd]);
        }

        // Store K,V in per-layer cache (kv_dim elements per position)
        let pos_off = pos * kvd;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.k.as_ptr(),
                layer_cache.key.as_mut_ptr().add(pos_off),
                kvd,
            );
            std::ptr::copy_nonoverlapping(
                ctx.v.as_ptr(),
                layer_cache.value.as_mut_ptr().add(pos_off),
                kvd,
            );
        }

        // Multi-head attention with GQA: fused score → softmax → weighted value per head
        let scale = ctx.attn_scale;
        let t_n = pos + 1;

        for h in 0..config.n_head {
            let kv_group = ctx.kv_group_lut[h];
            unsafe {
                attention_head(
                    &ctx.q,
                    &layer_cache.key,
                    &layer_cache.value,
                    &mut ctx.attn_out,
                    &mut ctx.scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                );
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut ctx.lora_buf);
        }
        crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut ctx.lora_buf);
        }
        // CNA: modulate discovered circuit neurons (Plan 087)
        #[cfg(feature = "cna_steering")]
        if let Some(ref modulator) = ctx.cna_modulator {
            crate::pruners::cna_modulate(&mut ctx.hidden, layer_idx, modulator);
        }
        // MLP w2: sparse when feature enabled and sparsity is high enough (Plan 022)
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut ctx.lora_buf);
        }
        crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);

        // MLS: accumulate layer delta (Plan 104)
        #[cfg(feature = "mls_aggregate")]
        {
            if config.mls_layers > 0 && layer_idx >= weights.layers.len() - config.mls_layers {
                crate::simd::simd_fused_sub_acc(
                    &mut ctx.mls_buf[..n],
                    &ctx.x[..n],
                    &ctx.hidden_state[..n],
                    n,
                );
                ctx.mls_count += 1;
            }
        }

        // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
        #[cfg(feature = "delta_routing")]
        {
            let block_size = 4; // Default B=4
            let block_idx = layer_idx / block_size;
            let pos_in_block = layer_idx % block_size;

            // Compute delta: current x minus pre-layer residual (xr was saved after first rmsnorm)
            // Delta captures full layer contribution: attention + MLP residuals
            if block_idx < ctx.block_deltas.len() {
                crate::simd::simd_fused_sub_acc(
                    &mut ctx.block_deltas[block_idx][..n],
                    &ctx.x[..n],
                    &ctx.xr[..n],
                    n,
                );
            }

            // At block boundary: route accumulated deltas from all completed blocks
            if pos_in_block == block_size - 1 && block_idx < ctx.block_deltas.len() {
                ctx.depth_route_blocks(
                    block_idx,
                    layer_idx,
                    &weights.delta_routing_query[layer_idx],
                    &weights.delta_routing_norm[layer_idx],
                    n,
                    weights,
                );
            }
        }
    }

    // MLS: blend averaged layer deltas into final hidden state (Plan 104)
    #[cfg(feature = "mls_aggregate")]
    if ctx.mls_count > 0 {
        let scale = 1.0 / ctx.mls_count as f32;
        crate::simd::simd_fused_decay_write(&mut ctx.x[..n], 1.0, &ctx.mls_buf[..n], scale);
    }

    // Snapshot hidden state (for Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head: clustered when vocab >= threshold AND cluster weights present
    if config.vocab_size >= config.mtp_cluster_vocab_threshold
        && let Some(classifier) = weights.mtp_cluster_classifier.as_ref()
        && let Some(cluster_map) = weights.mtp_cluster_map.as_ref()
    {
        clustered_lm_head(
            &mut ctx.logits,
            &ctx.x,
            &weights.lm_head,
            classifier,
            cluster_map,
            config.vocab_size,
            n,
            config.mtp_cluster_topk,
            &mut ctx.cluster_scores_buf,
            &mut ctx.topk_indexed_buf,
            &mut ctx.topk_output_buf,
        );
    } else {
        standard_lm_head(
            &mut ctx.logits,
            &ctx.x,
            &weights.lm_head,
            config.vocab_size,
            n,
        );
    }

    &mut ctx.logits
}

// ---------------------------------------------------------------------------
// MTP Target Activation Projection (Plan 055)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CODA Fused Forward Pass (Plan 103)
// ---------------------------------------------------------------------------

/// CODA-inspired fused forward pass (Research 67, Plan 103).
///
/// Algebraic reparameterization: fuse matmul+residual+rmsnorm+activation
/// into single-pass SIMD loops, eliminating intermediate buffer writes.
///
/// Key identity (CODA §3.2.1):
/// ```text
/// RMSNorm(x@W + z) * gamma @ W' = r * ((x@W + z) * gamma) @ W'
/// ```
///
/// This lets us delay the row-wise RMSNorm scale past the next GEMM,
/// fusing 3 separate operations into one SIMD loop per kernel.
///
/// # Buffer Write Savings (per layer)
///
/// Eliminated: out_proj write, residual add, xr2 copy, rmsnorm (pre-MLP),
/// gate_up write, activation pass, down_proj write, residual add = ~6 passes
///
/// Retained: 2× rmsnorm (pre-QKV), 1× xr copy = ~3 passes
///
/// # Feature Gate
///
/// Only compiled when `coda_fusion` feature is enabled. Falls back to
/// [`forward_base`] when LoRA is active (T10: future fused LoRA support).
#[cfg(feature = "coda_fusion")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn forward_coda<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    // NOTE(080): LoRA passthrough through CODA fused kernels.
    // LoRA is additive (scale * B @ (A @ input)), so it can't be fused into CODA's
    // bias parameter (which is a pre-computed vector). Instead, we compute LoRA
    // perturbations separately and add them to CODA kernel outputs, matching the
    // same projection points as forward_base: after QKV, after wo, after w1, after w2.

    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let _n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    crate::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // MLS: reset accumulator at start of forward call (Plan 104)
    #[cfg(feature = "mls_aggregate")]
    {
        ctx.mls_buf[..n].fill(0.0);
        ctx.mls_count = 0;
    }

    // 2. Layer loop with CODA-fused kernels
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // MLS: save pre-layer state for delta computation (Plan 104)
        #[cfg(feature = "mls_aggregate")]
        if config.mls_layers > 0 && layer_idx >= weights.layers.len() - config.mls_layers {
            ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
        }

        // Pre-attention: RMSNorm → save residual
        // Note: CODA fused kernels handle delayed RMS internally, no second rmsnorm needed
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // QKV projections (same as baseline — attention needs separate Q, K, V)
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // LoRA perturbation for QKV projections (same as forward_base)
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut ctx.lora_buf);
            crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut ctx.lora_buf);
            crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut ctx.lora_buf);
        }

        // Domain latent injection at mid-layer (Plan 038: Free Transformer adaptation)
        #[cfg(feature = "domain_latent")]
        if layer_idx == config.n_layer / 2
            && let Some(dl) = domain_latent
        {
            crate::simd::simd_add_inplace(&mut ctx.k[..kvd], &dl.embedding[..kvd]);
            crate::simd::simd_add_inplace(&mut ctx.v[..kvd], &dl.embedding[..kvd]);
        }

        // Store K,V in per-layer cache (kv_dim elements per position)
        let pos_off = pos * kvd;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.k.as_ptr(),
                layer_cache.key.as_mut_ptr().add(pos_off),
                kvd,
            );
            std::ptr::copy_nonoverlapping(
                ctx.v.as_ptr(),
                layer_cache.value.as_mut_ptr().add(pos_off),
                kvd,
            );
        }

        // Multi-head attention with GQA: fused score → softmax → weighted value per head
        let scale = ctx.attn_scale;
        let t_n = pos + 1;

        for h in 0..config.n_head {
            let kv_group = ctx.kv_group_lut[h];
            unsafe {
                attention_head(
                    &ctx.q,
                    &layer_cache.key,
                    &layer_cache.value,
                    &mut ctx.attn_out,
                    &mut ctx.scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                );
            }
        }

        // ── CODA FUSED KERNEL 1: out_proj + residual + partial_rms ──────
        // Replaces: matmul(x, wo, ao) + add(x, xr) + copy(xr2, x) + rmsnorm(x)
        //
        // D = wo @ attn_out + xr  → stored in ctx.xr2 (residual for down_proj)
        // O = D * gamma          → stored in ctx.x (input to MLP, gamma=identity)
        // partial_sums = Σ D[i]²  → for rstd computation
        katgpt_core::coda::simd_matmul_residual_partial_rms(
            &mut ctx.xr2[..n],          // output_d: D = matmul + residual
            &mut ctx.x[..n],            // output_o: O = D * gamma (gamma=identity)
            &mut ctx.coda_partial_sums, // partial RMS accumulation
            &layer_weights.attn_wo,     // weight
            &ctx.attn_out[..n],         // input
            &ctx.xr[..n],               // residual
            None,                       // gamma (None = identity for standard rmsnorm)
            None,                       // bias (no LoRA in fused path)
            n,                          // rows
            n,                          // cols
            n,                          // block_size (single block)
        );

        // LoRA perturbation for output projection: add to ctx.x (CODA output)
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x[..n], lora, &ctx.attn_out[..n], &mut ctx.lora_buf);
        }

        // ── CODA AUXILIARY REDUCTION: compute rstd ─────────────────────
        // rstd = 1 / sqrt(mean(D²) + eps) — tiny reduction, O(1) for single block
        let rstd = katgpt_core::coda::compute_rstd(&ctx.coda_partial_sums, n, 1e-5);

        // ── CODA FUSED KERNEL 2: MLP matmul + delayed RMS + activation ─
        // Replaces: rmsnorm(x) + matmul_relu(hidden, w1, x)
        // hidden[i] = activation(dot(w1[i], O) * rstd)  — delayed RMS scale
        katgpt_core::coda::simd_matmul_rmsnorm_activation(
            &mut ctx.hidden,                         // output
            &layer_weights.mlp_w1,                   // weight
            &ctx.x[..n],                             // input (O from kernel 1)
            rstd,                                    // delayed RMS scale
            katgpt_core::coda::GateActivation::Relu, // matches baseline matmul_relu
            config.mlp_hidden,                       // rows
            n,                                       // cols
        );

        // LoRA perturbation for MLP up projection: add to hidden
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.hidden, lora, &ctx.x[..n], &mut ctx.lora_buf);
        }

        // CNA: modulate discovered circuit neurons (Plan 087)
        #[cfg(feature = "cna_steering")]
        if let Some(ref modulator) = ctx.cna_modulator {
            crate::pruners::cna_modulate(&mut ctx.hidden, layer_idx, modulator);
        }

        // ── CODA FUSED KERNEL 3: down_proj + residual ─────────────────
        // Replaces: matmul(x, w2, hidden) + add(x, xr2)
        // x[i] = dot(w2[i], hidden) + xr2[i]
        #[cfg(not(feature = "sparse_mlp"))]
        katgpt_core::coda::simd_matmul_residual(
            &mut ctx.x[..n],       // output
            &layer_weights.mlp_w2, // weight
            &ctx.hidden,           // input
            &ctx.xr2[..n],         // residual (D from kernel 1)
            n,                     // rows
            config.mlp_hidden,     // cols
        );

        // Sparse MLP: try sparse first, fall back to fused dense + residual
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                // Too dense for sparse, use fused dense + residual
                katgpt_core::coda::simd_matmul_residual(
                    &mut ctx.x[..n],
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    &ctx.xr2[..n],
                    n,
                    config.mlp_hidden,
                );
            } else {
                // Sparse succeeded, add residual manually
                crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);
            }
        }

        // LoRA perturbation for MLP down projection (applies to both sparse and dense paths)
        if let Some(lora) = lora {
            crate::types::lora_apply(&mut ctx.x[..n], lora, &ctx.hidden, &mut ctx.lora_buf);
        }

        // MLS: accumulate layer delta (Plan 104)
        #[cfg(feature = "mls_aggregate")]
        {
            if config.mls_layers > 0 && layer_idx >= weights.layers.len() - config.mls_layers {
                crate::simd::simd_fused_sub_acc(
                    &mut ctx.mls_buf[..n],
                    &ctx.x[..n],
                    &ctx.hidden_state[..n],
                    n,
                );
                ctx.mls_count += 1;
            }
        }

        // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
        #[cfg(feature = "delta_routing")]
        {
            let block_size = 4; // Default B=4
            let block_idx = layer_idx / block_size;
            let pos_in_block = layer_idx % block_size;

            // Compute delta: current x minus pre-layer residual
            if block_idx < ctx.block_deltas.len() {
                crate::simd::simd_fused_sub_acc(
                    &mut ctx.block_deltas[block_idx][..n],
                    &ctx.x[..n],
                    &ctx.xr[..n],
                    n,
                );
            }

            // At block boundary: route accumulated deltas from all completed blocks
            if pos_in_block == block_size - 1 && block_idx < ctx.block_deltas.len() {
                ctx.depth_route_blocks(
                    block_idx,
                    layer_idx,
                    &weights.delta_routing_query[layer_idx],
                    &weights.delta_routing_norm[layer_idx],
                    n,
                    weights,
                );
            }
        }
    }

    // MLS: blend averaged layer deltas into final hidden state (Plan 104)
    #[cfg(feature = "mls_aggregate")]
    if ctx.mls_count > 0 {
        let scale = 1.0 / ctx.mls_count as f32;
        crate::simd::simd_fused_decay_write(&mut ctx.x[..n], 1.0, &ctx.mls_buf[..n], scale);
    }

    // Snapshot hidden state (for Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head: clustered when vocab >= threshold AND cluster weights present
    if config.vocab_size >= config.mtp_cluster_vocab_threshold
        && let Some(classifier) = weights.mtp_cluster_classifier.as_ref()
        && let Some(cluster_map) = weights.mtp_cluster_map.as_ref()
    {
        clustered_lm_head(
            &mut ctx.logits,
            &ctx.x,
            &weights.lm_head,
            classifier,
            cluster_map,
            config.vocab_size,
            n,
            config.mtp_cluster_topk,
            &mut ctx.cluster_scores_buf,
            &mut ctx.topk_indexed_buf,
            &mut ctx.topk_output_buf,
        );
    } else {
        standard_lm_head(
            &mut ctx.logits,
            &ctx.x,
            &weights.lm_head,
            config.vocab_size,
            n,
        );
    }

    &mut ctx.logits
}

// ---------------------------------------------------------------------------
// MTP Target Activation Projection (Plan 055)
// ---------------------------------------------------------------------------

/// Project target model's hidden state into drafter dimension space.
///
/// Two strategies (threshold-gated):
/// - **Truncate/Pad** (no weights): if `mtp_activation_proj` is `None`, truncate target
///   hidden state to drafter's n_embd (or zero-pad if drafter is larger).
///   Zero-cost, no training needed.
/// - **Learned projection** (with weights): matmul the target hidden state by
///   `mtp_activation_proj` to produce a drafter-sized conditioning vector.
///
/// The result is written into `out_buf` (pre-allocated `[drafter_n_embd]`).
/// Does nothing if `target_n_embd < config.mtp_activation_threshold`.
#[allow(clippy::too_many_arguments)]
pub fn project_target_activation(
    out_buf: &mut [f32],         // [drafter_n_embd] output buffer
    target_hidden: &[f32],       // [target_n_embd] from target's forward pass
    mtp_proj: Option<&Vec<f32>>, // optional [drafter_n_embd, target_n_embd] weights
    target_n_embd: usize,
    drafter_n_embd: usize,
    activation_threshold: usize,
) {
    // Gate: skip if target is too small for activation conditioning
    if target_n_embd < activation_threshold {
        return;
    }

    match mtp_proj {
        // Strategy 1: Learned projection — full matmul
        Some(proj_weights) => {
            // proj_weights layout: [drafter_n_embd * target_n_embd]
            // out[i] = sum_j(proj_weights[i * target_n_embd + j] * target_hidden[j])
            let out_len = out_buf.len().min(drafter_n_embd);
            for i in 0..out_len {
                let row_off = i * target_n_embd;
                let mut sum = 0.0f32;
                for j in 0..target_n_embd {
                    unsafe {
                        sum += *proj_weights.get_unchecked(row_off + j)
                            * *target_hidden.get_unchecked(j);
                    }
                }
                unsafe {
                    *out_buf.get_unchecked_mut(i) = sum;
                }
            }
        }
        // Strategy 2: Truncate/Pad — zero-cost fallback
        None => {
            let copy_len = drafter_n_embd.min(target_n_embd);
            out_buf[..copy_len].copy_from_slice(&target_hidden[..copy_len]);
            // Zero-pad if drafter dimension is larger (rest should already be zeroed)
            if drafter_n_embd > target_n_embd {
                out_buf[target_n_embd..drafter_n_embd].fill(0.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MTP Projection Binary Loader (Plan 016)
// ---------------------------------------------------------------------------

/// Binary format constants for MTP projection weights.
const MTP_PROJ_MAGIC: u32 = 0x4D54505A; // "MTPZ"
const MTP_PROJ_VERSION: u32 = 1;

/// Loaded MTP projection weights from compact binary (MTPZ v1).
///
/// Maps `[target_hidden; token_embed]` (in_dim = 2 × target_n_embd) → draft_n_embd.
#[derive(Debug)]
pub struct MtpProjection {
    /// Input dimension (2 × target_n_embd for `[target_hidden; token_embed]`).
    pub in_dim: usize,
    /// Output dimension (draft_n_embd).
    pub out_dim: usize,
    /// Weight matrix `[out_dim * in_dim]`, row-major.
    pub weights: Vec<f32>,
    /// Bias vector `[out_dim]`.
    pub bias: Vec<f32>,
}

/// Load MTP projection weights from compact binary format (MTPZ v1).
///
/// # Binary Layout
///
/// ```text
/// [magic: u32]     0x4D54505A ("MTPZ")
/// [version: u32]   1
/// [in_dim: u32]    input dimension
/// [out_dim: u32]   output dimension
/// [weights: f32 × out_dim × in_dim]  row-major
/// [bias: f32 × out_dim]
/// [checksum: u32]  blake3 of everything above
/// ```
///
/// # Errors
///
/// Returns an error string on: invalid magic, unsupported version, size mismatch,
/// blake3 checksum failure, or NaN/Inf in loaded data.
pub fn load_mtp_projection(path: &std::path::Path) -> Result<MtpProjection, String> {
    let data =
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {e}", path.display()))?;
    let file_size = data.len();

    // Header: 4 × u32 = 16 bytes
    let header_size: usize = 16;
    if file_size < header_size + 4 {
        return Err(format!("File too small: {file_size} bytes"));
    }

    // Parse header (little-endian)
    let magic = u32::from_le_bytes(
        data[0..4]
            .try_into()
            .map_err(|_| "header parse error".to_string())?,
    );
    let version = u32::from_le_bytes(
        data[4..8]
            .try_into()
            .map_err(|_| "header parse error".to_string())?,
    );
    let in_dim = u32::from_le_bytes(
        data[8..12]
            .try_into()
            .map_err(|_| "header parse error".to_string())?,
    ) as usize;
    let out_dim = u32::from_le_bytes(
        data[12..16]
            .try_into()
            .map_err(|_| "header parse error".to_string())?,
    ) as usize;

    if magic != MTP_PROJ_MAGIC {
        return Err(format!(
            "Invalid magic: expected {MTP_PROJ_MAGIC:#010x}, got {magic:#010x}"
        ));
    }
    if version != MTP_PROJ_VERSION {
        return Err(format!(
            "Unsupported version: expected {MTP_PROJ_VERSION}, got {version}"
        ));
    }

    // Calculate expected sizes
    let weights_bytes = out_dim * in_dim * 4; // f32 = 4 bytes
    let bias_bytes = out_dim * 4;
    let expected_size = header_size + weights_bytes + bias_bytes + 4; // +4 checksum

    if file_size != expected_size {
        return Err(format!(
            "Size mismatch: expected {expected_size} bytes, got {file_size} bytes (in_dim={in_dim}, out_dim={out_dim})"
        ));
    }

    // Verify blake3 checksum
    let payload = &data[..file_size - 4];
    let stored_checksum = u32::from_le_bytes(
        data[file_size - 4..]
            .try_into()
            .map_err(|_| "checksum parse error".to_string())?,
    );
    let computed_hash = blake3::hash(payload);
    let computed_checksum = u32::from_le_bytes(
        computed_hash.as_bytes()[..4]
            .try_into()
            .map_err(|_| "hash parse error".to_string())?,
    );

    if computed_checksum != stored_checksum {
        return Err(format!(
            "BLAKE3 checksum mismatch: stored={stored_checksum:#010x}, computed={computed_checksum:#010x}"
        ));
    }

    // Extract weights and bias as f32 (little-endian)
    let weights_offset = header_size;
    let bias_offset = weights_offset + weights_bytes;

    let weights: Vec<f32> = data[weights_offset..bias_offset]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let bias: Vec<f32> = data[bias_offset..file_size - 4]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    assert_eq!(weights.len(), out_dim * in_dim, "weights count mismatch");
    assert_eq!(bias.len(), out_dim, "bias count mismatch");

    // Validate no NaN/Inf
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(format!("NaN/Inf in weights at index {i}"));
        }
    }
    for (i, &b) in bias.iter().enumerate() {
        if !b.is_finite() {
            return Err(format!("NaN/Inf in bias at index {i}"));
        }
    }

    Ok(MtpProjection {
        in_dim,
        out_dim,
        weights,
        bias,
    })
}

#[cfg(test)]
mod mtp_projection_binary_tests {
    use super::*;
    use std::io::Write;

    /// Helper: create a valid MTPZ v1 binary at a temp path.
    fn create_test_binary(in_dim: usize, out_dim: usize) -> std::path::PathBuf {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&MTP_PROJ_MAGIC.to_le_bytes());
        buf.extend_from_slice(&MTP_PROJ_VERSION.to_le_bytes());
        buf.extend_from_slice(&(in_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(out_dim as u32).to_le_bytes());

        // Weights (zeros)
        for _ in 0..(out_dim * in_dim) {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Bias (zeros)
        for _ in 0..out_dim {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Checksum (blake3 of everything above)
        let hash = blake3::hash(&buf);
        let checksum = u32::from_le_bytes(hash.as_bytes()[..4].try_into().unwrap());
        buf.extend_from_slice(&checksum.to_le_bytes());

        let path = std::env::temp_dir().join("microgpt_test_mtp_projection.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&buf).unwrap();
        path
    }

    #[test]
    fn test_load_mtp_projection_valid_binary() {
        let path = create_test_binary(64, 16); // 2*32=64 in, 16 out
        let proj = load_mtp_projection(&path).unwrap();

        assert_eq!(proj.in_dim, 64);
        assert_eq!(proj.out_dim, 16);
        assert_eq!(proj.weights.len(), 64 * 16);
        assert_eq!(proj.bias.len(), 16);
        assert!(proj.weights.iter().all(|&w| w == 0.0));
        assert!(proj.bias.iter().all(|&b| b == 0.0));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_mtp_projection_invalid_magic() {
        let path = std::env::temp_dir().join("microgpt_test_mtp_bad_magic.bin");
        let mut buf = vec![0u8; 24]; // header(16) + min data(4) + checksum(4)
        buf[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());

        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&buf).unwrap();
        drop(f);

        let err = load_mtp_projection(&path).unwrap_err();
        assert!(
            err.contains("Invalid magic"),
            "expected 'Invalid magic' error, got: {err}"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_mtp_projection_bad_checksum() {
        let path = std::env::temp_dir().join("microgpt_test_mtp_bad_checksum.bin");
        let mut buf = Vec::new();

        buf.extend_from_slice(&MTP_PROJ_MAGIC.to_le_bytes());
        buf.extend_from_slice(&MTP_PROJ_VERSION.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // in_dim
        buf.extend_from_slice(&2u32.to_le_bytes()); // out_dim

        // Weights + bias (all zeros)
        for _ in 0..(2 * 4 + 2) {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }

        // Wrong checksum
        buf.extend_from_slice(&0xCAFEBABEu32.to_le_bytes());

        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&buf).unwrap();
        drop(f);

        let err = load_mtp_projection(&path).unwrap_err();
        assert!(
            err.contains("checksum mismatch"),
            "expected 'checksum mismatch' error, got: {err}"
        );

        let _ = std::fs::remove_file(&path);
    }
}

// ---------------------------------------------------------------------------
// Bidirectional Prefill (Plan 025)
// ---------------------------------------------------------------------------

/// Bidirectional prefill: process prompt tokens with full mutual attention.
///
/// For each transformer layer:
///   Phase A: Compute K/V for all prompt positions → store in KV cache
///   Phase B: For each position, attend to ALL prompt K/V (bidirectional)
///
/// Returns logits for the last prompt position (used to sample first gen token).
/// KV cache is populated as a side effect, shared with subsequent decode calls.
///
/// Zero-copy: no allocations. Reuses ForwardContext buffers per-position,
/// PrefillContext::hidden for multi-layer inter-layer state.
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
pub fn forward_prefill<'a>(
    ctx: &'a mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    tokens: &[usize],
    config: &Config,
    lora: Option<&crate::types::LoraAdapter>,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> &'a mut [f32] {
    let prompt_len = tokens.len().min(prefill.max_prompt_len);
    if prompt_len > 0 {
        cache.advance_pos(prompt_len - 1);
    }
    let n = config.n_embd;
    let kvd = crate::types::kv_dim(config);
    let hd = config.head_dim;
    let _n_kv = config.n_kv_head;

    assert!(prompt_len > 0, "prefill requires at least one token");
    assert!(
        prompt_len <= config.block_size,
        "prompt_len {prompt_len} exceeds block_size {}",
        config.block_size
    );

    // Initialize hidden states for multi-layer (single-layer computes on-the-fly)
    if config.n_layer > 1 {
        for (p, &token) in tokens.iter().enumerate().take(prompt_len) {
            let tok_off = token * n;
            let pos_off = p * n;
            crate::simd::simd_add_into(
                &mut prefill.hidden[p * n..(p + 1) * n],
                &weights.wte[tok_off..tok_off + n],
                &weights.wpe[pos_off..pos_off + n],
            );
        }
    }

    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // ── Phase A: Compute K/V for ALL positions → store in cache ──
        for (p, &token) in tokens.iter().enumerate().take(prompt_len) {
            // Load hidden state
            if config.n_layer > 1 {
                ctx.x[..n].copy_from_slice(&prefill.hidden[p * n..(p + 1) * n]);
            } else {
                let tok_off = token * n;
                let pos_off = p * n;
                crate::simd::simd_add_into(
                    &mut ctx.x[..n],
                    &weights.wte[tok_off..tok_off + n],
                    &weights.wpe[pos_off..pos_off + n],
                );
            }

            // Pre-attention norm (matches forward_base exactly: double rmsnorm)
            crate::types::rmsnorm(&mut ctx.x);
            ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);

            // K/V projections
            crate::types::matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut prefill.lora_buf);
            }
            crate::types::matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut prefill.lora_buf);
            }

            // Domain latent injection at mid-layer (Plan 038: Free Transformer adaptation)
            #[cfg(feature = "domain_latent")]
            if layer_idx == config.n_layer / 2
                && let Some(dl) = domain_latent
            {
                crate::simd::simd_add_inplace(&mut ctx.k[..kvd], &dl.embedding[..kvd]);
                crate::simd::simd_add_inplace(&mut ctx.v[..kvd], &dl.embedding[..kvd]);
            }

            // Store K/V in cache
            let pos_off = p * kvd;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ctx.k.as_ptr(),
                    layer_cache.key.as_mut_ptr().add(pos_off),
                    kvd,
                );
                std::ptr::copy_nonoverlapping(
                    ctx.v.as_ptr(),
                    layer_cache.value.as_mut_ptr().add(pos_off),
                    kvd,
                );
            }

            // Q projection (fused: avoids redundant hidden load + rmsnorm in Phase B)
            crate::types::matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut prefill.lora_buf);
            }

            // Store Q and xr for Phase B reuse
            let q_off = p * n;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ctx.q.as_ptr(),
                    prefill.queries.as_mut_ptr().add(q_off),
                    n,
                );
                std::ptr::copy_nonoverlapping(
                    ctx.xr.as_ptr(),
                    prefill.residuals.as_mut_ptr().add(q_off),
                    n,
                );
            }
        }

        // ── Phase B: Bidirectional attention for ALL positions ──
        // Loads pre-computed Q and xr from fused Phase A, skipping redundant
        // hidden state load + double rmsnorm + Q matmul per position.

        // Tiled attention: batch-compute all positions for large prompts (Plan 115)
        // Avoids O(N²) score matrix materialization when prompt_len >= 128
        #[cfg(feature = "tiled_attention")]
        let use_tiled = prompt_len >= 128;

        #[cfg(feature = "tiled_attention")]
        if use_tiled {
            let tiled_size = config.n_head * prompt_len * hd;
            // Repack Q: (position, head) → (head, position) contiguous layout
            for h in 0..config.n_head {
                for p in 0..prompt_len {
                    let src_off = p * n + h * hd;
                    let dst_off = h * prompt_len * hd + p * hd;
                    ctx.tiled_q[dst_off..dst_off + hd]
                        .copy_from_slice(&prefill.queries[src_off..src_off + hd]);
                }
            }
            // Repack K/V with GQA expansion: (position, kv_group) → (head, position)
            for h in 0..config.n_head {
                let kv_group = ctx.kv_group_lut[h];
                for p in 0..prompt_len {
                    let kv_src = p * kvd + kv_group * hd;
                    let dst_off = h * prompt_len * hd + p * hd;
                    ctx.tiled_k[dst_off..dst_off + hd]
                        .copy_from_slice(&layer_cache.key[kv_src..kv_src + hd]);
                    ctx.tiled_v[dst_off..dst_off + hd]
                        .copy_from_slice(&layer_cache.value[kv_src..kv_src + hd]);
                }
            }
            katgpt_core::tiled_attention_batched(
                &ctx.tiled_q[..tiled_size],
                &ctx.tiled_k[..tiled_size],
                &ctx.tiled_v[..tiled_size],
                &mut ctx.tiled_out[..tiled_size],
                1,
                config.n_head,
                prompt_len,
                hd,
            );
        }

        for p in 0..prompt_len {
            let q_off = p * n;

            // Load residual (xr) for output projection
            unsafe {
                std::ptr::copy_nonoverlapping(
                    prefill.residuals.as_ptr().add(q_off),
                    ctx.xr.as_mut_ptr(),
                    n,
                );
            }

            // ── Attention computation (tiled or per-head) ──
            ctx.attn_out[..n].fill(0.0);

            #[cfg(feature = "tiled_attention")]
            if use_tiled {
                // Unpack tiled output: (head, position) → attn_out for this position
                for h in 0..config.n_head {
                    let src_off = h * prompt_len * hd + p * hd;
                    let dst_off = h * hd;
                    ctx.attn_out[dst_off..dst_off + hd]
                        .copy_from_slice(&ctx.tiled_out[src_off..src_off + hd]);
                }
            } else {
                // Per-head attention for small prompts (below threshold)
                let scale = ctx.attn_scale;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        prefill.queries.as_ptr().add(q_off),
                        ctx.q.as_mut_ptr(),
                        n,
                    );
                }
                for h in 0..config.n_head {
                    let kv_group = ctx.kv_group_lut[h];
                    unsafe {
                        attention_head(
                            &ctx.q,
                            &layer_cache.key,
                            &layer_cache.value,
                            &mut ctx.attn_out,
                            &mut ctx.scores,
                            h * hd,
                            kv_group * hd,
                            kvd,
                            hd,
                            prompt_len,
                            scale,
                        );
                    }
                }
            }

            #[cfg(not(feature = "tiled_attention"))]
            {
                let scale = ctx.attn_scale;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        prefill.queries.as_ptr().add(q_off),
                        ctx.q.as_mut_ptr(),
                        n,
                    );
                }
                for h in 0..config.n_head {
                    let kv_group = ctx.kv_group_lut[h];
                    unsafe {
                        attention_head(
                            &ctx.q,
                            &layer_cache.key,
                            &layer_cache.value,
                            &mut ctx.attn_out,
                            &mut ctx.scores,
                            h * hd,
                            kv_group * hd,
                            kvd,
                            hd,
                            prompt_len,
                            scale,
                        );
                    }
                }
            }

            // Output projection + residual
            crate::types::matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut prefill.lora_buf);
            }
            crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

            // MLP: residual → RMSNorm → MLP → residual
            ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
            crate::types::rmsnorm(&mut ctx.x);
            crate::types::matmul_relu(
                &mut ctx.hidden,
                &layer_weights.mlp_w1,
                &ctx.x,
                config.mlp_hidden,
                n,
            );
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut prefill.lora_buf);
            }
            // MLP w2 (with sparse support)
            #[cfg(feature = "sparse_mlp")]
            {
                let alive = crate::types::sparse_matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                    &mut ctx.active_indices,
                    &mut ctx.active_values,
                );
                if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                    crate::types::matmul(
                        &mut ctx.x,
                        &layer_weights.mlp_w2,
                        &ctx.hidden,
                        n,
                        config.mlp_hidden,
                    );
                }
            }
            #[cfg(not(feature = "sparse_mlp"))]
            crate::types::matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
            );
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut prefill.lora_buf);
            }
            crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);

            // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
            #[cfg(feature = "delta_routing")]
            {
                let block_size = 4; // Default B=4
                let block_idx = layer_idx / block_size;
                let pos_in_block = layer_idx % block_size;

                // Compute delta: current x minus pre-layer residual (xr was saved after first rmsnorm)
                if block_idx < ctx.block_deltas.len() {
                    crate::simd::simd_fused_sub_acc(
                        &mut ctx.block_deltas[block_idx][..n],
                        &ctx.x[..n],
                        &ctx.xr[..n],
                        n,
                    );
                }

                // At block boundary: route accumulated deltas from all completed blocks
                if pos_in_block == block_size - 1 && block_idx < ctx.block_deltas.len() {
                    ctx.depth_route_blocks(
                        block_idx,
                        layer_idx,
                        &weights.delta_routing_query[layer_idx],
                        &weights.delta_routing_norm[layer_idx],
                        n,
                        weights,
                    );
                }
            }

            // Store hidden state for next layer (multi-layer only)
            if config.n_layer > 1 {
                prefill.hidden[p * n..(p + 1) * n].copy_from_slice(&ctx.x[..n]);
            }
        }
    }

    // Snapshot hidden state (last position)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head (parallel for large vocab, serial fallback for small)
    crate::types::matmul_parallel(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Full generation pipeline: bidirectional prefill → causal decode.
/// Switches from reader LoRA to writer LoRA at the prefill→decode boundary.
/// Zero-copy: all buffers pre-allocated, no allocations in request path.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_prefill(
    ctx: &mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    config: &Config,
    rng: &mut crate::types::Rng,
    prompt_tokens: &[usize],
    max_gen_tokens: usize,
    lora_pair: &crate::types::LoraPair,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> Vec<usize> {
    // 1. Bidirectional prefill with reader LoRA
    let logits = {
        #[cfg(not(feature = "domain_latent"))]
        {
            forward_prefill(
                ctx,
                prefill,
                weights,
                cache,
                prompt_tokens,
                config,
                lora_pair.reader.as_ref(),
            )
        }
        #[cfg(feature = "domain_latent")]
        {
            forward_prefill(
                ctx,
                prefill,
                weights,
                cache,
                prompt_tokens,
                config,
                lora_pair.reader.as_ref(),
                domain_latent,
            )
        }
    };

    // 2. Sample first generation token from prefill output
    // softmax_scaled fuses temperature + softmax in-place, avoiding logits.to_vec() allocation
    crate::types::softmax_scaled(logits, 1.0 / config.temperature);
    let mut token = crate::types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf);

    let mut generated = vec![token];
    let mut pos = prompt_tokens.len();

    // 3. Causal decode with writer LoRA
    for _ in 1..max_gen_tokens {
        if pos >= config.block_size {
            break;
        }

        let logits = {
            #[cfg(not(feature = "domain_latent"))]
            {
                forward_base(
                    ctx,
                    weights,
                    cache,
                    token,
                    pos,
                    config,
                    lora_pair.writer.as_ref(),
                )
            }
            #[cfg(feature = "domain_latent")]
            {
                forward_base(
                    ctx,
                    weights,
                    cache,
                    token,
                    pos,
                    config,
                    lora_pair.writer.as_ref(),
                    domain_latent,
                )
            }
        };
        for logit in logits.iter_mut() {
            *logit /= config.temperature;
        }
        crate::types::softmax(logits);

        token = crate::types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf);
        generated.push(token);
        pos += 1;

        if token == config.bos_token {
            break;
        }
    }

    generated
}

/// Generate with prefill and optional domain latent (Plan 038).
/// Convenience wrapper for callers that need domain conditioning during generation.
#[cfg(feature = "domain_latent")]
#[allow(clippy::too_many_arguments)]
pub fn generate_with_prefill_and_domain_latent(
    ctx: &mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    config: &Config,
    rng: &mut crate::types::Rng,
    prompt_tokens: &[usize],
    max_gen_tokens: usize,
    lora_pair: &crate::types::LoraPair,
    domain_latent: Option<&crate::types::DomainLatent>,
) -> Vec<usize> {
    generate_with_prefill(
        ctx,
        prefill,
        weights,
        cache,
        config,
        rng,
        prompt_tokens,
        max_gen_tokens,
        lora_pair,
        domain_latent,
    )
}

/// Forward pass using `PagedKVCache` instead of `MultiLayerKVCache`.
///
/// Identical computation to `forward()` but stores KV in paged memory,
/// enabling copy-on-write fork for DDTree branch exploration.
/// Builds a temporary flat KV buffer per layer for attention computation.
#[inline(always)]
pub fn forward_paged<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    paged_cache: &mut PagedKVCache,
    seq_idx: usize,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = crate::types::kv_dim(config);
    let _n_kv = config.n_kv_head;

    // Ensure pages allocated for this sequence up to pos
    paged_cache.ensure_pages(seq_idx, pos);

    // Flat KV cache for attention computation (pre-allocated, reused from ForwardContext)
    let t_n = pos + 1;
    let flat_kv_len = t_n * kvd;
    ctx.paged_flat_key[..flat_kv_len].fill(0.0);
    ctx.paged_flat_value[..flat_kv_len].fill(0.0);

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    crate::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // QKV projections
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // Write K,V to paged cache
        paged_cache.write_kv(layer_idx, seq_idx, pos, &ctx.k, &ctx.v);

        // Build flat KV from paged cache for attention
        {
            let flat_key = &mut ctx.paged_flat_key[..flat_kv_len];
            let flat_value = &mut ctx.paged_flat_value[..flat_kv_len];
            for t in 0..t_n {
                let k_slice = &mut flat_key[t * kvd..(t + 1) * kvd];
                let v_slice = &mut flat_value[t * kvd..(t + 1) * kvd];
                paged_cache.read_kv(layer_idx, seq_idx, t, k_slice, v_slice);
            }

            // Multi-head attention with GQA (reuse existing attention_head)
            let scale = ctx.attn_scale;

            for h in 0..config.n_head {
                let kv_group = ctx.kv_group_lut[h];
                unsafe {
                    attention_head(
                        &ctx.q,
                        flat_key,
                        flat_value,
                        &mut ctx.attn_out,
                        &mut ctx.scores,
                        h * hd,
                        kv_group * hd,
                        kvd,
                        hd,
                        t_n,
                        scale,
                    );
                }
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        // MLP w2: sparse when feature enabled and sparsity is high enough (Plan 022)
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);

        // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
        #[cfg(feature = "delta_routing")]
        {
            let block_size = 4; // Default B=4
            let block_idx = layer_idx / block_size;
            let pos_in_block = layer_idx % block_size;

            // Compute delta: current x minus pre-layer residual (xr was saved after first rmsnorm)
            if block_idx < ctx.block_deltas.len() {
                crate::simd::simd_fused_sub_acc(
                    &mut ctx.block_deltas[block_idx][..n],
                    &ctx.x[..n],
                    &ctx.xr[..n],
                    n,
                );
            }

            // At block boundary: route accumulated deltas from all completed blocks
            if pos_in_block == block_size - 1 && block_idx < ctx.block_deltas.len() {
                ctx.depth_route_blocks(
                    block_idx,
                    layer_idx,
                    &weights.delta_routing_query[layer_idx],
                    &weights.delta_routing_norm[layer_idx],
                    n,
                    weights,
                );
            }
        }
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Zero-alloc generation: `ctx`, `cache`, `tokens` all provided by caller.
///
/// `tokens` is cleared and filled with generated token ids.
/// `ctx` and `cache` are reused across calls.
pub fn generate_into(
    ctx: &mut ForwardContext,
    cache: &mut MultiLayerKVCache,
    weights: &TransformerWeights,
    config: &Config,
    rng: &mut Rng,
    n_tokens: usize,
    tokens: &mut Vec<usize>,
) {
    tokens.clear();
    let mut token = config.bos_token;
    let mut pos = 0;

    for _ in 0..n_tokens {
        if pos >= config.block_size {
            cache.reset();
            pos = 0;
            token = config.bos_token;
        }

        {
            let logits = forward(ctx, weights, cache, token, pos, config);
            softmax_scaled(logits, 1.0 / config.temperature);
        }

        let next_token = sample_token_into(&ctx.logits, rng, &mut ctx.cdf);
        tokens.push(next_token);

        if next_token == config.bos_token {
            cache.reset();
            pos = 0;
            token = config.bos_token;
        } else {
            token = next_token;
            pos += 1;
        }
    }
}

/// Generate tokens autoregressively. Returns generated token ids.
pub fn generate(
    weights: &TransformerWeights,
    config: &Config,
    rng: &mut Rng,
    n_tokens: usize,
) -> Vec<usize> {
    let mut ctx = ForwardContext::new(config);
    let mut cache = MultiLayerKVCache::new(config);
    let mut tokens = Vec::new();
    generate_into(
        &mut ctx,
        &mut cache,
        weights,
        config,
        rng,
        n_tokens,
        &mut tokens,
    );
    tokens
}

/// Generate multiple samples in parallel using rayon.
///
/// Each sample gets its own `ForwardContext` + `MultiLayerKVCache` via `map_init`,
/// so there's no contention. The `seeds` slice provides one seed per sample.
/// Returns `Vec<Vec<usize>>` with one token sequence per sample.
pub fn generate_batch(
    weights: &TransformerWeights,
    config: &Config,
    seeds: &[u64],
    n_tokens: usize,
) -> Vec<Vec<usize>> {
    seeds
        .par_iter()
        .map_init(
            || (ForwardContext::new(config), MultiLayerKVCache::new(config)),
            |(ctx, cache), &seed| {
                let mut rng = Rng::new(seed);
                let mut tokens = Vec::with_capacity(n_tokens);
                generate_into(ctx, cache, weights, config, &mut rng, n_tokens, &mut tokens);
                tokens
            },
        )
        .collect()
}

/// Convert token ids to readable characters (a-z, _ for BOS).
pub fn tokens_to_string(tokens: &[usize]) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    tokens
        .iter()
        .map(|&t| if t < 26 { CHARS[t] as char } else { '_' })
        .collect()
}

/// Page size in tokens (tuneable, must be power of 2).
const PAGE_SIZE: usize = 16;

/// Paged KV cache for DDTree branch exploration.
/// Allocates memory in fixed-size pages with copy-on-write fork.
///
/// Page layout per page: `[K_data | V_data]` where each segment is `PAGE_SIZE * kv_dim` floats.
/// This enables sharing prefix pages between branches without cloning data.
pub struct PagedKVCache {
    /// Pool of pages. Each page: `[PAGE_SIZE * kv_dim * 2]` floats (K then V).
    pages: Vec<Vec<f32>>,
    /// Per-layer page tables. `layer_page_tables[layer][seq_idx]` = vec of page indices.
    layer_page_tables: Vec<Vec<Vec<usize>>>,
    /// Free list of page indices for reuse.
    free_pages: Vec<usize>,
    /// Dimension of each KV entry (`n_kv_head * head_dim`).
    kv_dim: usize,
    /// Total pages ever allocated (monotonically increasing).
    total_pages: usize,
    /// Reusable scratch: per-layer page deficits (cleared + refilled each call).
    deficits: Vec<usize>,
    /// Reusable scratch: per-layer new page indices (cleared + refilled each call).
    new_pages: Vec<Vec<usize>>,
}

impl PagedKVCache {
    /// Create a new paged KV cache.
    /// `max_sequences`: initial number of sequence slots (can grow via fork).
    pub fn new(config: &Config, max_sequences: usize) -> Self {
        let kvd = types::kv_dim(config);
        let initial_pages_per_layer = config.block_size / PAGE_SIZE + 1;

        Self {
            pages: (0..initial_pages_per_layer * config.n_layer)
                .map(|_| vec![0.0; PAGE_SIZE * kvd * 2])
                .collect(),
            layer_page_tables: (0..config.n_layer)
                .map(|_| {
                    (0..max_sequences)
                        .map(|_| Vec::with_capacity(initial_pages_per_layer))
                        .collect()
                })
                .collect(),
            free_pages: Vec::new(),
            kv_dim: kvd,
            total_pages: initial_pages_per_layer * config.n_layer,
            deficits: Vec::with_capacity(config.n_layer),
            new_pages: vec![Vec::new(); config.n_layer],
        }
    }

    /// Allocate a new page. Reuse from free list or grow the pool.
    fn alloc_page(&mut self) -> usize {
        match self.free_pages.pop() {
            Some(idx) => {
                self.pages[idx].fill(0.0);
                idx
            }
            None => {
                self.pages.push(vec![0.0; PAGE_SIZE * self.kv_dim * 2]);
                let idx = self.total_pages;
                self.total_pages += 1;
                idx
            }
        }
    }

    /// Ensure sequence `seq_idx` has enough pages to cover position `pos` for all layers.
    pub fn ensure_pages(&mut self, seq_idx: usize, pos: usize) {
        let pages_needed = pos / PAGE_SIZE + 1;

        // Grow sequence slots if needed (no page allocation, just empty vecs)
        for layer_tables in &mut self.layer_page_tables {
            while seq_idx >= layer_tables.len() {
                layer_tables.push(Vec::new());
            }
        }

        // Collect how many new pages each layer needs (reuse scratch buffer)
        self.deficits.clear();
        for lt in &self.layer_page_tables {
            self.deficits
                .push(pages_needed.saturating_sub(lt[seq_idx].len()));
        }

        // Allocate all pages upfront
        let total_deficit: usize = self.deficits.iter().sum();
        let mut all_new: Vec<usize> = Vec::with_capacity(total_deficit);
        for _ in 0..total_deficit {
            all_new.push(self.alloc_page());
        }

        // Partition into per-layer lists and assign
        self.new_pages.resize_with(self.deficits.len(), Vec::new);
        let mut offset = 0;
        for (i, &deficit) in self.deficits.iter().enumerate() {
            self.new_pages[i].clear();
            self.new_pages[i].extend_from_slice(&all_new[offset..offset + deficit]);
            offset += deficit;
        }

        // Assign new pages to each layer's page table
        for (layer_tables, pages) in self.layer_page_tables.iter_mut().zip(self.new_pages.iter()) {
            layer_tables[seq_idx].extend(pages.iter().copied());
        }
    }

    /// Write K and V for a token position in a specific layer.
    /// Layout per page: `[K_data | V_data]` where each is `PAGE_SIZE * kv_dim` floats.
    pub fn write_kv(&mut self, layer_idx: usize, seq_idx: usize, pos: usize, k: &[f32], v: &[f32]) {
        let page_local = pos % PAGE_SIZE;
        let page_list_idx = pos / PAGE_SIZE;
        let pidx = self.layer_page_tables[layer_idx][seq_idx][page_list_idx];
        let page = &mut self.pages[pidx];
        let kv_page_size = PAGE_SIZE * self.kv_dim;
        let k_off = page_local * self.kv_dim;
        let v_off = kv_page_size + page_local * self.kv_dim;
        page[k_off..k_off + self.kv_dim].copy_from_slice(k);
        page[v_off..v_off + self.kv_dim].copy_from_slice(v);
    }

    /// Read K and V for a token position in a specific layer.
    pub fn read_kv(
        &self,
        layer_idx: usize,
        seq_idx: usize,
        pos: usize,
        k: &mut [f32],
        v: &mut [f32],
    ) {
        let page_local = pos % PAGE_SIZE;
        let page_list_idx = pos / PAGE_SIZE;
        let pidx = self.layer_page_tables[layer_idx][seq_idx][page_list_idx];
        let page = &self.pages[pidx];
        let kv_page_size = PAGE_SIZE * self.kv_dim;
        let k_off = page_local * self.kv_dim;
        let v_off = kv_page_size + page_local * self.kv_dim;
        k.copy_from_slice(&page[k_off..k_off + self.kv_dim]);
        v.copy_from_slice(&page[v_off..v_off + self.kv_dim]);
    }

    /// Fork a sequence with copy-on-write semantics.
    /// Shares prefix pages up to `fork_at_pos`, allocates new pages on demand after fork.
    /// Returns the new sequence index.
    pub fn fork(&mut self, seq_idx: usize, fork_at_pos: usize) -> usize {
        let fork_page = fork_at_pos / PAGE_SIZE;
        let new_seq = self.layer_page_tables[0].len();

        for layer_tables in &mut self.layer_page_tables {
            let source = &layer_tables[seq_idx];
            let shared_pages = source[..fork_page.min(source.len())].to_vec();
            layer_tables.push(shared_pages);
        }

        new_seq
    }

    /// Rollback a sequence to a given position, freeing exclusive pages.
    ///
    /// Truncates page tables to keep only pages covering positions `[0..rollback_to_pos)`.
    /// Pages that are exclusively owned by this sequence (not referenced by any other
    /// sequence in any layer) are returned to the free list for reuse.
    ///
    /// This is the "page table CoW rollback" — no data is copied, only page table
    /// entries are manipulated and exclusive pages are recycled.
    pub fn rollback(&mut self, seq_idx: usize, rollback_to_pos: usize) {
        let keep_count = rollback_to_pos / PAGE_SIZE;

        // Build set of page indices referenced by all OTHER sequences across all layers.
        // Since page indices are globally unique (allocated from a single pool), a simple
        // HashSet<usize> is sufficient — no need for (layer, page) tuples.
        let mut referenced_by_others = std::collections::HashSet::new();
        for layer_tables in self.layer_page_tables.iter() {
            for (seq, table) in layer_tables.iter().enumerate() {
                if seq != seq_idx {
                    for &pidx in table {
                        referenced_by_others.insert(pidx);
                    }
                }
            }
        }

        // Truncate page tables and free exclusive pages
        for layer_tables in &mut self.layer_page_tables {
            if seq_idx >= layer_tables.len() {
                continue;
            }
            let table = &mut layer_tables[seq_idx];
            let removed: Vec<usize> = table.drain(keep_count..).collect();
            for pidx in removed {
                if !referenced_by_others.contains(&pidx) {
                    self.free_pages.push(pidx);
                }
            }
        }
    }

    /// Reset all sequences and free all pages.
    pub fn reset(&mut self) {
        for layer_tables in &mut self.layer_page_tables {
            for table in layer_tables.iter_mut() {
                self.free_pages.append(table);
            }
        }
    }
}

// ── Raven RSM (Routing Slot Memory) ────────────────────────────
// Distilled from "Raven: High-Recall Sequence Modeling with Sparse Memory Routing"
// See .research/06_Raven_Routing_Slot_Memories.md for full derivation.
//
// Replaces the growing [block_size, kv_dim] cache with a fixed [num_slots, kv_dim]
// memory updated via sparse Top-K routing. Unselected slots are completely frozen.
// Per-token compute: O(num_slots) — constant regardless of sequence length.

/// Raven Routing Slot Memory — O(1) KV replacement for the draft model.
///
/// Fixed-size `[num_slots × kv_dim]` memory updated via sparse Top-K routing.
/// Unselected slots are completely frozen — perfect for preserving struct
/// definitions and imports while churning through syntax tokens.
pub struct RavenKVCache {
    /// Number of memory slots
    pub num_slots: usize,
    /// Dimension of each KV entry (= kv_dim = n_kv_head × head_dim)
    pub kv_dim: usize,
    /// Top-K slots to update per token
    pub top_k: usize,
    /// Forget rate for gated update (negative = slower decay)
    pub forget_rate: f32,
    /// Key memory: [num_slots × kv_dim]
    pub keys: Vec<f32>,
    /// Value memory: [num_slots × kv_dim]
    pub values: Vec<f32>,
    // Pre-allocated buffers for zero-alloc router computation
    router_scored: Vec<(usize, f32)>, // [num_slots]
    router_r_t: Vec<f32>,             // [num_slots]
    /// Pre-allocated score buffer for raven_readout_into [num_slots]
    readout_scores: Vec<f32>,
    /// Pre-allocated output buffer for raven_readout_into [kv_dim]
    readout_output: Vec<f32>,
}

impl RavenKVCache {
    pub fn new(config: &Config, num_slots: usize, top_k: usize) -> Self {
        let kvd = types::kv_dim(config);
        Self {
            num_slots,
            kv_dim: kvd,
            top_k,
            forget_rate: -1.0,
            keys: vec![0.0; num_slots * kvd],
            values: vec![0.0; num_slots * kvd],
            router_scored: Vec::with_capacity(num_slots),
            router_r_t: Vec::with_capacity(num_slots),
            readout_scores: vec![0.0; num_slots],
            readout_output: vec![0.0; kvd],
        }
    }

    pub fn reset(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        self.router_scored.clear();
        self.router_r_t.clear();
        self.readout_scores.fill(0.0);
        self.readout_output.fill(0.0);
    }
}

/// Sparse router: computes Top-K routing vector from raw logits (zero-alloc variant).
///
/// Implements: `r_t = Normalize(TopK(Sigmoid(raw_logits)))`
/// Unselected slots get 0.0 → completely frozen during update.
///
/// Uses pre-allocated buffers to avoid heap allocations on the hot path.
pub fn raven_compute_router_into(
    raw_logits: &[f32],
    top_k: usize,
    scored: &mut Vec<(usize, f32)>,
    r_t: &mut Vec<f32>,
) {
    let num_slots = raw_logits.len();
    let top_k = top_k.min(num_slots);

    // Reuse pre-allocated buffers: clear + fill
    scored.clear();
    for (i, &x) in raw_logits.iter().enumerate() {
        scored.push((i, 1.0 / (1.0 + (-x).exp())));
    }

    // Partial sort: find Top-K by descending score (O(n) average)
    if top_k < num_slots {
        scored.select_nth_unstable_by(num_slots - top_k, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    r_t.clear();
    r_t.resize(num_slots, 0.0);
    let mut sum = 0.0f32;

    // Keep only Top-K (the last top_k elements after partial sort are the largest)
    for (idx, score) in scored.iter().rev().take(top_k) {
        r_t[*idx] = *score;
        sum += *score;
    }

    // Normalize so selected slots sum to 1.0
    if sum > 0.0 {
        for v in r_t.iter_mut() {
            *v /= sum;
        }
    }
}

/// Backward-compatible wrapper that allocates fresh buffers.
pub fn raven_compute_router(raw_logits: &[f32], top_k: usize) -> Vec<f32> {
    let mut scored = Vec::new();
    let mut r_t = Vec::new();
    raven_compute_router_into(raw_logits, top_k, &mut scored, &mut r_t);
    r_t
}

/// Gated memory update: Raven Equation 18.
///
/// For each slot:
///   `decay = exp(forget_rate × r_t[slot])`
///   `H_new = decay × H_old + (1 - decay) × new_content`
///
/// When `r_t[slot] == 0`: `decay = exp(0) = 1.0` → `H_new = H_old` (FROZEN)
/// When `r_t[slot] > 0`: `decay < 1.0` → old content decays, new writes in
#[allow(clippy::too_many_arguments)]
pub fn raven_update(
    keys: &mut [f32],
    values: &mut [f32],
    new_key: &[f32],
    new_value: &[f32],
    r_t: &[f32],
    forget_rate: f32,
    num_slots: usize,
    kv_dim: usize,
) {
    for (slot, &route) in r_t.iter().enumerate().take(num_slots) {
        let decay = (forget_rate * route).exp();
        let write = 1.0 - decay;
        let offset = slot * kv_dim;

        crate::simd::simd_fused_decay_write(
            &mut keys[offset..offset + kv_dim],
            decay,
            &new_key[..kv_dim],
            write,
        );
        crate::simd::simd_fused_decay_write(
            &mut values[offset..offset + kv_dim],
            decay,
            &new_value[..kv_dim],
            write,
        );
    }
}

/// Readout: attention over fixed slot memory.
/// `O(num_slots × kv_dim)` — constant regardless of sequence length.
/// Zero-alloc readout: computes attention-weighted slot values into pre-allocated buffers.
///
/// Fused 2-pass optimization over `raven_readout` (3-pass):
/// - Pass 1: Q·K^T dot products + find max
/// - Pass 2: exp(scores - max) + weighted value accumulation + normalize
///
/// Returns `&mut output[..kv_dim]` (borrowed from the provided output buffer).
pub fn raven_readout_into<'a>(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    num_slots: usize,
    kv_dim: usize,
    scores: &'a mut [f32],
    output: &'a mut [f32],
) -> &'a mut [f32] {
    debug_assert!(scores.len() >= num_slots);
    debug_assert!(output.len() >= kv_dim);

    // Pass 1: Q·K^T + find max
    let mut max_score = f32::NEG_INFINITY;
    for s in 0..num_slots {
        let k_off = s * kv_dim;
        let dot = crate::simd::simd_dot_f32(query, &keys[k_off..k_off + kv_dim], kv_dim);
        unsafe {
            *scores.get_unchecked_mut(s) = dot;
        }
        if dot > max_score {
            max_score = dot;
        }
    }

    // Pass 2: fused exp + accumulate + normalize
    output[..kv_dim].fill(0.0);
    let mut sum_exp = 0.0f32;
    for s in 0..num_slots {
        let exp_val = unsafe { (*scores.get_unchecked(s) - max_score).exp() };
        unsafe {
            *scores.get_unchecked_mut(s) = exp_val;
        }
        sum_exp += exp_val;
    }

    if sum_exp > 0.0 {
        let inv_sum = 1.0 / sum_exp;
        for s in 0..num_slots {
            let weight = unsafe { *scores.get_unchecked(s) * inv_sum };
            let v_off = s * kv_dim;
            crate::simd::simd_fused_scale_acc(
                &mut output[..kv_dim],
                &values[v_off..v_off + kv_dim],
                weight,
                kv_dim,
            );
        }
    }

    &mut output[..kv_dim]
}

/// Allocating wrapper for backward compatibility (tests, benchmark).
pub fn raven_readout(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    num_slots: usize,
    kv_dim: usize,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; num_slots];
    let mut output = vec![0.0f32; kv_dim];
    raven_readout_into(
        query,
        keys,
        values,
        num_slots,
        kv_dim,
        &mut scores,
        &mut output,
    );
    output
}

/// Forward pass using `RavenKVCache` instead of `MultiLayerKVCache`.
///
/// Identical computation to `forward()` except attention:
/// - Generates router logits from K projection (dummy: use K directly)
/// - Calls `raven_update()` instead of writing to flat KV array
/// - Calls `raven_readout()` instead of scanning all past positions
/// - Everything else (RMSNorm, MLP, residual, LM head) stays identical
pub fn forward_raven<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut RavenKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let _n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    crate::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        // layer_idx used by delta_routing cfg blocks below
        #[cfg(not(feature = "delta_routing"))]
        let _ = layer_idx;
        // Pre-attention: RMSNorm → save residual → RMSNorm
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);

        // QKV projections
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // Raven: generate router logits from K (dummy projection)
        // For PoC: use first num_slots elements of K repeated as logits.
        // In production, this would be a learned linear projection: W_route × x_t
        // Reuse pre-allocated query buffer for router logits (zero-alloc)
        ctx.raven_query_buf.resize(cache.num_slots, 0.0);
        for (i, slot) in ctx.raven_query_buf.iter_mut().enumerate() {
            *slot = ctx.k[i % kvd];
        }

        // Raven: compute sparse routing vector (zero-alloc via pre-allocated buffers)
        raven_compute_router_into(
            &ctx.raven_query_buf,
            cache.top_k,
            &mut cache.router_scored,
            &mut cache.router_r_t,
        );

        // Stack-allocated copy to avoid self-borrow (cache.keys vs cache.router_r_t)
        // num_slots is typically 16-64 floats — fits on stack
        let mut r_t = [0.0f32; 64];
        let copy_len = cache.router_r_t.len().min(64);
        r_t[..copy_len].copy_from_slice(&cache.router_r_t[..copy_len]);

        // Raven: gated update (only selected slots are modified)
        raven_update(
            &mut cache.keys,
            &mut cache.values,
            &ctx.k,
            &ctx.v,
            &r_t,
            cache.forget_rate,
            cache.num_slots,
            kvd,
        );

        // Raven: readout via attention over fixed slots (O(num_slots) not O(pos))
        let scale = ctx.attn_scale;
        ctx.attn_out[..n].fill(0.0);

        for h in 0..config.n_head {
            let q_off = h * hd;
            // Each head reads from the slot memory using its query slice
            let head_query = &ctx.q[q_off..q_off + hd];
            // Pad/reshape query to kv_dim for slot attention (reuse pre-allocated buffer)
            ctx.raven_query_buf.resize(kvd, 0.0);
            ctx.raven_query_buf.fill(0.0);
            let kv_group = ctx.kv_group_lut[h];
            for (d, &hq) in head_query.iter().enumerate() {
                ctx.raven_query_buf[kv_group * hd + d] = hq * scale;
            }

            let slot_values = raven_readout_into(
                &ctx.raven_query_buf,
                &cache.keys,
                &cache.values,
                cache.num_slots,
                kvd,
                &mut cache.readout_scores,
                &mut cache.readout_output,
            );

            // Extract this head's attention output
            for d in 0..hd {
                unsafe {
                    *ctx.attn_out.get_unchecked_mut(q_off + d) = slot_values[kv_group * hd + d];
                }
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        // MLP w2: sparse when feature enabled and sparsity is high enough (Plan 022)
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);

        // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
        #[cfg(feature = "delta_routing")]
        {
            let block_size = 4; // Default B=4
            let block_idx = layer_idx / block_size;
            let pos_in_block = layer_idx % block_size;

            // Compute delta: current x minus pre-layer residual (xr was saved after first rmsnorm)
            if block_idx < ctx.block_deltas.len() {
                crate::simd::simd_fused_sub_acc(
                    &mut ctx.block_deltas[block_idx][..n],
                    &ctx.x[..n],
                    &ctx.xr[..n],
                    n,
                );
            }

            // At block boundary: route accumulated deltas from all completed blocks
            if pos_in_block == block_size - 1 && block_idx < ctx.block_deltas.len() {
                ctx.depth_route_blocks(
                    block_idx,
                    layer_idx,
                    &weights.delta_routing_query[layer_idx],
                    &weights.delta_routing_norm[layer_idx],
                    n,
                    weights,
                );
            }
        }
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Forward pass using quantized KV cache (Plan 043, generalized Plan 063).
///
/// Mirrors [`forward_base`] but stores K/V into a compressed cache and
/// dequantizes on-the-fly during attention scoring. The rest of the
/// transformer (embedding, QKV projection, MLP, LM head) is unchanged.
///
/// Generic over any [`types::QuantizedKVCache`] backend (SpectralQuant, TurboQuant, etc.).
///
/// **Trade-off**: ~8× KV cache memory savings at the cost of dequantization
/// overhead during attention. Best for long sequences where cache memory
/// dominates.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_quantized<'a, C: types::QuantizedKVCache>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut C,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let _n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    crate::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        // Pre-attention: RMSNorm → save residual
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // QKV projections from per-layer weights (GQA: K/V produce kv_dim outputs)
        matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
        matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
        matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);

        // Store compressed K,V
        cache.store_key(layer_idx, pos, &ctx.k[..kvd]);
        cache.store_value(layer_idx, pos, &ctx.v[..kvd]);

        // Incremental dequant (Plan 068): only dequant the new position when possible.
        // Tracks per-layer progress: if tq_dequant_pos[layer] == pos - 1, the flat buffer
        // already contains positions 0..pos-1 from the previous decode step for this layer.
        // On mismatch (first call, layer switch, reset, pos jump), rebuild all positions.
        let t_n = pos + 1;
        let last_pos = ctx.dequant_pos[layer_idx];
        if last_pos + 1 == pos && pos > 0 {
            // Incremental: only dequant the new position
            cache.dequantize_key_into(
                layer_idx,
                pos,
                &mut ctx.paged_flat_key[pos * kvd..(pos + 1) * kvd],
            );
            cache.dequantize_value_into(
                layer_idx,
                pos,
                &mut ctx.paged_flat_value[pos * kvd..(pos + 1) * kvd],
            );
        } else {
            // Full rebuild: dequantize all positions (first call, reset, or pos jump)
            for t in 0..t_n {
                cache.dequantize_key_into(
                    layer_idx,
                    t,
                    &mut ctx.paged_flat_key[t * kvd..(t + 1) * kvd],
                );
                cache.dequantize_value_into(
                    layer_idx,
                    t,
                    &mut ctx.paged_flat_value[t * kvd..(t + 1) * kvd],
                );
            }
        }
        ctx.dequant_pos[layer_idx] = pos;

        // Multi-head attention with GQA using dequantized flat cache
        let scale = ctx.attn_scale;

        for h in 0..config.n_head {
            let kv_group = ctx.kv_group_lut[h];
            unsafe {
                attention_head(
                    &ctx.q,
                    &ctx.paged_flat_key,
                    &ctx.paged_flat_value,
                    &mut ctx.attn_out,
                    &mut ctx.scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                );
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        rmsnorm(&mut ctx.x);
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        // MLP w2: sparse when feature enabled and sparsity is high enough (Plan 022)
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);

        // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
        #[cfg(feature = "delta_routing")]
        {
            let block_size = 4; // Default B=4
            let block_idx = layer_idx / block_size;
            let pos_in_block = layer_idx % block_size;

            // Compute delta: current x minus pre-layer residual (xr was saved after first rmsnorm)
            if block_idx < ctx.block_deltas.len() {
                crate::simd::simd_fused_sub_acc(
                    &mut ctx.block_deltas[block_idx][..n],
                    &ctx.x[..n],
                    &ctx.xr[..n],
                    n,
                );
            }

            // At block boundary: route accumulated deltas from all completed blocks
            if pos_in_block == block_size - 1 && block_idx < ctx.block_deltas.len() {
                ctx.depth_route_blocks(
                    block_idx,
                    layer_idx,
                    &weights.delta_routing_query[layer_idx],
                    &weights.delta_routing_norm[layer_idx],
                    n,
                    weights,
                );
            }
        }
    }

    // Snapshot hidden state (for Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head
    matmul(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Backward-compat alias: forward using TurboQuant-specific cache.
///
/// Prefer [`forward_quantized`] for new code — it's generic over any
/// [`types::QuantizedKVCache`] backend.
#[cfg(feature = "turboquant")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn forward_turboquant<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut crate::turboquant::TurboQuantKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    forward_quantized(ctx, weights, cache, token, pos, config)
}

#[cfg(test)]
#[allow(unnameable_test_items)]
#[allow(dead_code)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_output_size() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_forward_logits_finite() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} is not finite: {l}");
        }
    }

    #[test]
    fn test_forward_cache_populated() {
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        let key_sum: f32 = cache.layers[0].key[..kvd].iter().sum();
        let val_sum: f32 = cache.layers[0].value[..kvd].iter().sum();
        assert!(key_sum != 0.0, "K cache at pos 0 should be populated");
        assert!(val_sum != 0.0, "V cache at pos 0 should be populated");
    }

    #[test]
    fn test_forward_positions_differ() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits_0 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config).to_vec();
        let logits_1 = forward(&mut ctx, &weights, &mut cache, 0, 1, &config);
        let different = logits_0.iter().zip(logits_1).any(|(&a, b)| a != *b);
        assert!(different, "logits at different positions should differ");
    }

    #[test]
    fn test_generate_deterministic() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut rng1 = Rng::new(100);
        let t1 = generate(&weights, &config, &mut rng1, 16);

        let mut rng2 = Rng::new(100);
        let t2 = generate(&weights, &config, &mut rng2, 16);

        assert_eq!(t1, t2, "Same seed must produce same tokens");
    }

    #[test]
    fn test_generate_valid_tokens() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = generate(&weights, &config, &mut rng, 32);
        assert_eq!(tokens.len(), 32);
        for &t in &tokens {
            assert!(t < config.vocab_size, "Token {t} out of range");
        }
    }

    #[test]
    fn test_tokens_to_string() {
        let tokens = vec![0, 1, 2, 25, 26];
        let s = tokens_to_string(&tokens);
        assert_eq!(s, "abcz_");
    }

    #[test]
    fn test_forward_context_reuse() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Multiple forward passes with same context should give same results
        let _l1 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config).to_vec();
        let l2 = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        // Note: results differ because cache accumulates, but buffers should not leak
        for &v in l2.iter() {
            assert!(v.is_finite(), "reused context produced non-finite: {v}");
        }
    }

    // ── Multi-layer tests ─────────────────────────────────────────

    #[test]
    fn test_forward_output_size_nlayer2() {
        let mut config = Config::micro();
        config.n_layer = 2;
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        assert_eq!(weights.layers.len(), 2);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        assert_eq!(cache.layers.len(), 2);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_forward_logits_finite_nlayer4() {
        let mut config = Config::micro();
        config.n_layer = 4;
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} is not finite with n_layer=4: {l}");
        }
    }

    #[test]
    fn test_n_layer_1_matches_current() {
        // n_layer=1 must produce identical deterministic output to old single-layer code
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut rng1 = Rng::new(100);
        let t1 = generate(&weights, &config, &mut rng1, 16);

        let mut rng2 = Rng::new(100);
        let t2 = generate(&weights, &config, &mut rng2, 16);

        assert_eq!(t1, t2, "n_layer=1 should be deterministic");
        assert_eq!(config.n_layer, 1, "micro config should have n_layer=1");
    }

    #[test]
    fn test_multi_layer_cache_populated() {
        let mut config = Config::micro();
        config.n_layer = 3;
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

        // Every layer's cache should be populated
        for (layer_idx, layer_cache) in cache.layers.iter().enumerate() {
            let key_sum: f32 = layer_cache.key[..kvd].iter().sum();
            let val_sum: f32 = layer_cache.value[..kvd].iter().sum();
            assert!(
                key_sum != 0.0,
                "layer {layer_idx} K cache at pos 0 should be populated"
            );
            assert!(
                val_sum != 0.0,
                "layer {layer_idx} V cache at pos 0 should be populated"
            );
        }
    }

    #[test]
    fn test_hidden_state_populated() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        forward(&mut ctx, &weights, &mut cache, 0, 0, &config);
        let sum: f32 = ctx.hidden_state.iter().sum();
        assert!(
            sum != 0.0,
            "hidden_state should be populated after forward pass"
        );
        for (i, &v) in ctx.hidden_state.iter().enumerate() {
            assert!(v.is_finite(), "hidden_state[{i}] should be finite: {v}");
        }
    }

    #[test]
    fn test_multi_layer_generate_valid() {
        let mut config = Config::micro();
        config.n_layer = 4;
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = generate(&weights, &config, &mut rng, 16);
        assert_eq!(tokens.len(), 16);
        for &t in &tokens {
            assert!(t < config.vocab_size, "Token {t} out of range");
        }
    }

    // ── GQA tests ───────────────────────────────────────────────

    #[test]
    fn test_gqa_produces_valid_logits() {
        let config = Config::gqa_draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        for pos in 0..4 {
            let logits = forward(&mut ctx, &weights, &mut cache, 0, pos, &config);
            for (i, &l) in logits.iter().enumerate() {
                assert!(
                    l.is_finite(),
                    "gqa_draft logit {i} at pos {pos} not finite: {l}"
                );
            }
        }
    }

    #[test]
    fn test_gqa_mha_backward_compat() {
        // When n_kv_head == n_head, GQA produces identical results to standard MHA.
        // Micro config has n_kv_head=4, n_head=4 → pure MHA.
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut rng1 = Rng::new(100);
        let t1 = generate(&weights, &config, &mut rng1, 16);

        let mut rng2 = Rng::new(100);
        let t2 = generate(&weights, &config, &mut rng2, 16);

        assert_eq!(
            t1, t2,
            "MHA backward compat: same seed must produce same tokens"
        );
        assert_eq!(
            config.n_kv_head, config.n_head,
            "micro config should have n_kv_head == n_head"
        );
    }

    #[test]
    fn test_gqa_kv_cache_smaller() {
        // GQA config should have smaller KV cache than equivalent MHA config
        let gqa = Config::gqa_draft();
        let kvd = crate::types::kv_dim(&gqa);
        assert_eq!(
            kvd,
            gqa.n_kv_head * gqa.head_dim,
            "kv_dim should be n_kv_head * head_dim"
        );
        assert!(
            kvd < gqa.n_embd,
            "GQA kv_dim ({kvd}) should be < n_embd ({})",
            gqa.n_embd
        );

        // Verify cache is correctly sized
        let cache = KVCache::new(&gqa);
        assert_eq!(
            cache.key.len(),
            gqa.block_size * kvd,
            "GQA key cache should use kv_dim"
        );
        assert_eq!(
            cache.value.len(),
            gqa.block_size * kvd,
            "GQA value cache should use kv_dim"
        );
    }

    #[test]
    fn test_gqa_generate_valid_tokens() {
        let config = Config::gqa_draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = generate(&weights, &config, &mut rng, 8);
        assert_eq!(tokens.len(), 8);
        for &t in &tokens {
            assert!(t < config.vocab_size, "GQA token {t} out of range");
        }
    }

    #[test]
    fn test_config_validate_gqa() {
        // Valid configs should pass validation
        assert!(Config::micro().validate().is_ok());
        assert!(Config::draft().validate().is_ok());
        assert!(Config::small_target().validate().is_ok());
        assert!(Config::gqa_draft().validate().is_ok());

        // Invalid: n_head not divisible by n_kv_head
        let mut bad = Config::micro();
        bad.n_kv_head = 3; // n_head=4, not divisible by 3
        assert!(bad.validate().is_err());

        // Invalid: n_head * head_dim != n_embd
        let mut bad2 = Config::micro();
        bad2.head_dim = 5; // 4*5=20 != 16
        assert!(bad2.validate().is_err());
    }

    // ── Paged KV cache tests ────────────────────────────────────

    #[test]
    fn test_paged_cache_write_read_roundtrip() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 1);
        let kvd = crate::types::kv_dim(&config);

        // Ensure pages for position 0
        paged.ensure_pages(0, 0);

        // Write some K/V data
        let k_data: Vec<f32> = (0..kvd).map(|i| i as f32 * 0.1).collect();
        let v_data: Vec<f32> = (0..kvd).map(|i| i as f32 * 0.2).collect();
        paged.write_kv(0, 0, 0, &k_data, &v_data);

        // Read back
        let mut k_out = vec![0.0f32; kvd];
        let mut v_out = vec![0.0f32; kvd];
        paged.read_kv(0, 0, 0, &mut k_out, &mut v_out);

        assert_eq!(k_out, k_data, "K data roundtrip mismatch");
        assert_eq!(v_out, v_data, "V data roundtrip mismatch");
    }

    #[test]
    fn test_paged_cache_linear_matches_flat() {
        // Paged cache should produce same results as flat cache for a linear sequence
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Run with flat cache
        let mut ctx = ForwardContext::new(&config);
        let mut flat_cache = MultiLayerKVCache::new(&config);
        let _flat_logits = forward(&mut ctx, &weights, &mut flat_cache, 0, 0, &config).to_vec();

        // Manually copy flat cache data to paged cache
        let mut paged = PagedKVCache::new(&config, 1);
        paged.ensure_pages(0, 0);

        for (layer_idx, layer_cache) in flat_cache.layers.iter().enumerate() {
            let k_data = &layer_cache.key[..kvd];
            let v_data = &layer_cache.value[..kvd];
            paged.write_kv(layer_idx, 0, 0, k_data, v_data);
        }

        // Read back and compare
        for layer_idx in 0..config.n_layer {
            let mut k_out = vec![0.0f32; kvd];
            let mut v_out = vec![0.0f32; kvd];
            paged.read_kv(layer_idx, 0, 0, &mut k_out, &mut v_out);

            let flat_k = &flat_cache.layers[layer_idx].key[..kvd];
            let flat_v = &flat_cache.layers[layer_idx].value[..kvd];
            assert_eq!(k_out, flat_k, "layer {layer_idx} K mismatch: paged vs flat");
            assert_eq!(v_out, flat_v, "layer {layer_idx} V mismatch: paged vs flat");
        }
    }

    #[test]
    fn test_paged_cache_fork_no_corruption() {
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut paged = PagedKVCache::new(&config, 1);

        // Write data to seq 0 at position 0
        paged.ensure_pages(0, 0);
        let k_orig: Vec<f32> = (0..kvd).map(|i| i as f32 + 1.0).collect();
        let v_orig: Vec<f32> = (0..kvd).map(|i| i as f32 + 2.0).collect();
        paged.write_kv(0, 0, 0, &k_orig, &v_orig);

        // Fork at position 0 (share nothing — fork_page = 0/16 = 0)
        let fork_seq = paged.fork(0, 0);

        // Write different data to forked seq
        paged.ensure_pages(fork_seq, 0);
        let k_fork: Vec<f32> = (0..kvd).map(|i| i as f32 + 99.0).collect();
        let v_fork: Vec<f32> = (0..kvd).map(|i| i as f32 + 100.0).collect();
        paged.write_kv(0, fork_seq, 0, &k_fork, &v_fork);

        // Original seq should be unchanged
        let mut k_check = vec![0.0f32; kvd];
        let mut v_check = vec![0.0f32; kvd];
        paged.read_kv(0, 0, 0, &mut k_check, &mut v_check);
        assert_eq!(k_check, k_orig, "original K corrupted after fork write");
        assert_eq!(v_check, v_orig, "original V corrupted after fork write");
    }

    #[test]
    fn test_paged_cache_fork_shares_prefix() {
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut paged = PagedKVCache::new(&config, 1);

        // Write data at positions 0..PAGE_SIZE (fills one page)
        paged.ensure_pages(0, PAGE_SIZE - 1);
        for pos in 0..PAGE_SIZE {
            let k: Vec<f32> = vec![pos as f32; kvd];
            let v: Vec<f32> = vec![pos as f32 * 2.0; kvd];
            paged.write_kv(0, 0, pos, &k, &v);
        }

        // Fork at position 8 (still within page 0)
        let fork_seq = paged.fork(0, 8);

        // Ensure forked seq has its own pages from fork point
        paged.ensure_pages(fork_seq, PAGE_SIZE);

        // The forked seq should share page 0 (prefix) but have its own page 1+
        // Verify shared prefix data is accessible
        let mut k_out = vec![0.0f32; kvd];
        let mut v_out = vec![0.0f32; kvd];
        paged.read_kv(0, fork_seq, 0, &mut k_out, &mut v_out);
        assert_eq!(k_out[0], 0.0, "forked seq should see original pos 0 data");
    }

    #[test]
    fn test_paged_cache_reset_frees_pages() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 2);

        // Allocate pages for two sequences
        paged.ensure_pages(0, 31); // 2 pages (0..15 and 16..31)
        paged.ensure_pages(1, 15); // 1 page

        let total_before = paged.total_pages;
        assert!(total_before > 0, "should have allocated some pages");

        // Reset should free all pages
        paged.reset();

        // Free list should contain the freed pages
        // (exact count depends on implementation, but should be > 0)
        // After reset, we can allocate again and reuse freed pages
        paged.ensure_pages(0, 0);
        // If reuse works, total_pages shouldn't grow
        assert_eq!(paged.total_pages, total_before, "should reuse freed pages");
    }

    #[test]
    fn test_snapshot_restore_roundtrip() {
        // Forward some tokens, snapshot, modify, restore, verify same logits
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Fill cache with tokens at positions 0..4
        for pos in 0..4 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        // Snapshot at position 4
        let snapshot = cache.snapshot(4, &config);

        // Fill more positions
        for pos in 4..8 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        // Now restore
        cache.restore(&snapshot, &config);

        // Verify restored: forward at position 4 should give same result as fresh cache at pos 4
        let mut fresh_cache = MultiLayerKVCache::new(&config);
        let mut fresh_ctx = ForwardContext::new(&config);
        for pos in 0..4 {
            let _ = forward(
                &mut fresh_ctx,
                &weights,
                &mut fresh_cache,
                pos,
                pos,
                &config,
            );
        }

        let restored_logits = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
        let fresh_logits = forward(&mut fresh_ctx, &weights, &mut fresh_cache, 0, 4, &config);

        for (a, b) in restored_logits.iter().zip(fresh_logits.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "restored logits should match fresh: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_snapshot_correct_size() {
        let config = Config::micro();
        let kd = types::kv_dim(&config);
        let cache = MultiLayerKVCache::new(&config);
        let snapshot = cache.snapshot(5, &config);

        assert_eq!(snapshot.pos, 5);
        assert_eq!(snapshot.layers.len(), config.n_layer);
        for layer in &snapshot.layers {
            assert_eq!(layer.key.len(), 5 * kd);
            assert_eq!(layer.value.len(), 5 * kd);
        }
    }

    #[test]
    fn test_restore_zeros_stale_data() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Fill cache
        for pos in 0..8 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        // Snapshot at position 3
        let snapshot = cache.snapshot(3, &config);

        // Restore
        cache.restore(&snapshot, &config);

        // Verify positions after pos=3 are zeroed
        let kd = types::kv_dim(&config);
        for layer in &cache.layers {
            for val in &layer.key[3 * kd..] {
                assert_eq!(*val, 0.0, "stale key data should be zeroed");
            }
            for val in &layer.value[3 * kd..] {
                assert_eq!(*val, 0.0, "stale value data should be zeroed");
            }
        }
    }

    #[test]
    fn test_snapshot_restore_multi_layer() {
        // Test with n_layer > 1 (small_target config)
        let config = Config::small_target();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Fill cache
        for pos in 0..4 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        let snapshot = cache.snapshot(4, &config);
        assert_eq!(snapshot.layers.len(), 4, "should have 4 layer snapshots");

        // Modify and restore
        for pos in 4..8 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }
        cache.restore(&snapshot, &config);

        // Verify restored correctly by checking logits match fresh cache
        let mut fresh_cache = MultiLayerKVCache::new(&config);
        let mut fresh_ctx = ForwardContext::new(&config);
        for pos in 0..4 {
            let _ = forward(
                &mut fresh_ctx,
                &weights,
                &mut fresh_cache,
                pos,
                pos,
                &config,
            );
        }

        let restored_logits = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
        let fresh_logits = forward(&mut fresh_ctx, &weights, &mut fresh_cache, 0, 4, &config);

        for (a, b) in restored_logits.iter().zip(fresh_logits.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "multi-layer restore should match fresh"
            );
        }
    }

    #[test]
    fn test_snapshot_restore_gqa() {
        // Test with GQA config (kv_dim < n_embd)
        let config = Config::gqa_draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        for pos in 0..4 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }

        let snapshot = cache.snapshot(4, &config);
        let kd = types::kv_dim(&config);

        // Verify snapshot uses GQA kv_dim (smaller than n_embd)
        assert_eq!(kd, config.n_kv_head * config.head_dim);
        assert!(kd < config.n_embd, "GQA kv_dim should be < n_embd");
        for layer in &snapshot.layers {
            assert_eq!(layer.key.len(), 4 * kd);
        }

        // Restore and verify
        for pos in 4..8 {
            let _ = forward(&mut ctx, &weights, &mut cache, pos, pos, &config);
        }
        cache.restore(&snapshot, &config);

        let mut fresh_cache = MultiLayerKVCache::new(&config);
        let mut fresh_ctx = ForwardContext::new(&config);
        for pos in 0..4 {
            let _ = forward(
                &mut fresh_ctx,
                &weights,
                &mut fresh_cache,
                pos,
                pos,
                &config,
            );
        }

        let restored_logits = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
        let fresh_logits = forward(&mut fresh_ctx, &weights, &mut fresh_cache, 0, 4, &config);

        for (a, b) in restored_logits.iter().zip(fresh_logits.iter()) {
            assert!((a - b).abs() < 1e-3, "GQA restore should match fresh");
        }
    }

    // ── forward_paged tests ──────────────────────────────────────

    #[test]
    fn test_forward_paged_logits_match_forward() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Flat cache forward
        let mut ctx_flat = ForwardContext::new(&config);
        let mut cache_flat = MultiLayerKVCache::new(&config);
        let logits_flat = forward(&mut ctx_flat, &weights, &mut cache_flat, 0, 0, &config);

        // Paged cache forward
        let mut ctx_paged = ForwardContext::new(&config);
        let mut cache_paged = PagedKVCache::new(&config, 1);
        let logits_paged =
            forward_paged(&mut ctx_paged, &weights, &mut cache_paged, 0, 0, 0, &config);

        assert_eq!(logits_flat.len(), logits_paged.len());
        for (i, (a, b)) in logits_flat.iter().zip(logits_paged.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "forward_paged logit {i} differs: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_forward_paged_logits_match_forward_multi_pos() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut ctx_flat = ForwardContext::new(&config);
        let mut cache_flat = MultiLayerKVCache::new(&config);

        let mut ctx_paged = ForwardContext::new(&config);
        let mut cache_paged = PagedKVCache::new(&config, 1);

        for pos in 0..4 {
            let token = pos; // simple: use pos as token
            let logits_flat = forward(
                &mut ctx_flat,
                &weights,
                &mut cache_flat,
                token,
                pos,
                &config,
            );
            let logits_paged = forward_paged(
                &mut ctx_paged,
                &weights,
                &mut cache_paged,
                0,
                token,
                pos,
                &config,
            );

            for (i, (a, b)) in logits_flat.iter().zip(logits_paged.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-3,
                    "pos {pos} logit {i} differs: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn test_forward_paged_gqa_logits_match() {
        let config = Config::gqa_draft();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut ctx_flat = ForwardContext::new(&config);
        let mut cache_flat = MultiLayerKVCache::new(&config);
        let logits_flat = forward(&mut ctx_flat, &weights, &mut cache_flat, 0, 0, &config);

        let mut ctx_paged = ForwardContext::new(&config);
        let mut cache_paged = PagedKVCache::new(&config, 1);
        let logits_paged =
            forward_paged(&mut ctx_paged, &weights, &mut cache_paged, 0, 0, 0, &config);

        assert_eq!(logits_flat.len(), logits_paged.len());
        for (i, (a, b)) in logits_flat.iter().zip(logits_paged.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "GQA forward_paged logit {i} differs: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_forward_paged_output_size() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = PagedKVCache::new(&config, 1);
        let logits = forward_paged(&mut ctx, &weights, &mut cache, 0, 0, 0, &config);
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_forward_paged_logits_finite() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut cache = PagedKVCache::new(&config, 1);
        let logits = forward_paged(&mut ctx, &weights, &mut cache, 0, 0, 0, &config);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} is not finite: {l}");
        }
    }

    // ── Rollback tests ─────────────────────────────────────────────

    #[test]
    fn test_paged_rollback_frees_exclusive_pages() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 2);

        // Allocate pages for seq 0 up to pos 31 (2 pages: 0..15, 16..31)
        paged.ensure_pages(0, 31);
        let seq0_pages_len = paged.layer_page_tables[0][0].len();
        assert!(seq0_pages_len >= 2, "seq 0 should have at least 2 pages");

        // Rollback seq 0 to pos 0 — all pages are exclusive (no other seq)
        paged.rollback(0, 0);

        // Page table should be truncated
        assert!(
            paged.layer_page_tables[0][0].is_empty(),
            "seq 0 page table should be empty after rollback to pos 0"
        );
        // All pages should be freed (they were exclusive)
        assert!(
            !paged.free_pages.is_empty(),
            "exclusive pages should be returned to free list"
        );
    }

    #[test]
    fn test_paged_rollback_preserves_shared_pages() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 4);

        // Allocate pages for seq 0 up to pos 31
        paged.ensure_pages(0, 31);
        let _initial_pages_len = paged.layer_page_tables[0][0].len();

        // Fork a new sequence from seq 0 at pos 16 — shares first page
        // (fork returns layer_page_tables[0].len(), which may be > 1 if max_sequences > 1)
        let seq1 = paged.fork(0, 16);
        assert_ne!(seq1, 0, "fork should return a new sequence index");

        // Allocate exclusive pages for seq 0 beyond fork point
        paged.ensure_pages(0, 47); // extra pages after pos 31

        let free_before = paged.free_pages.len();
        let pages_before_rollback = paged.layer_page_tables[0][0].len();

        // Rollback seq 0 to pos 16 — keeps shared page, frees exclusive ones
        paged.rollback(0, 16);

        // Page table should be truncated to 1 page (covers 0..15)
        assert_eq!(
            paged.layer_page_tables[0][0].len(),
            1,
            "seq 0 should have 1 page after rollback to pos 16 (page covers 0..15)"
        );

        // Some pages should have been freed (the exclusive ones beyond page 0)
        let freed = paged.free_pages.len() - free_before;
        assert!(
            freed > 0,
            "exclusive pages beyond rollback point should be freed"
        );

        // But NOT more than what was removed from page table
        let removed = pages_before_rollback - 1;
        assert!(
            freed <= removed,
            "freed pages ({freed}) should not exceed removed pages ({removed})"
        );
    }

    #[test]
    fn test_paged_rollback_shared_page_not_freed() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 4);

        // Allocate pages for seq 0
        paged.ensure_pages(0, 31);

        // Fork seq 1 at pos 0 — shares nothing initially (fork_page = 0)
        let seq1 = paged.fork(0, 0);

        // Allocate different pages for seq 1
        paged.ensure_pages(seq1, 31);

        // Now fork seq 2 from seq 0 at pos 16 — shares first page with seq 0
        let seq2 = paged.fork(0, 16);
        let shared_page_idx = paged.layer_page_tables[0][0][0];

        // Rollback seq 2 to pos 0 — the shared page should NOT be freed
        let _free_before = paged.free_pages.len();
        paged.rollback(seq2, 0);

        // Shared page should still be in seq 0's page table
        assert!(
            paged.layer_page_tables[0][0].contains(&shared_page_idx),
            "shared page should still be referenced by seq 0"
        );
        // Shared page should NOT be in free list
        assert!(
            !paged.free_pages.contains(&shared_page_idx),
            "shared page should not be freed"
        );
    }

    #[test]
    fn test_paged_rollback_truncates_page_table() {
        let config = Config::micro();
        let mut paged = PagedKVCache::new(&config, 1);

        // Allocate 4 pages worth of positions
        paged.ensure_pages(0, 63);
        assert!(
            paged.layer_page_tables[0][0].len() >= 4,
            "should have at least 4 pages for pos 0..63"
        );

        // Rollback to pos 32 — should keep 2 pages (0..15, 16..31)
        paged.rollback(0, 32);
        assert_eq!(
            paged.layer_page_tables[0][0].len(),
            2,
            "should have exactly 2 pages after rollback to pos 32"
        );

        // Rollback to pos 16 — should keep 1 page (0..15)
        paged.rollback(0, 16);
        assert_eq!(
            paged.layer_page_tables[0][0].len(),
            1,
            "should have exactly 1 page after rollback to pos 16"
        );
    }

    #[test]
    fn test_paged_rollback_all_layers_consistent() {
        let mut config = Config::micro();
        config.n_layer = 4;
        let mut paged = PagedKVCache::new(&config, 1);

        // Allocate pages for all layers
        paged.ensure_pages(0, 31);

        // Rollback to pos 16
        paged.rollback(0, 16);

        // All layers should have the same page table length
        let expected = 1; // 1 page covers 0..15
        for (layer_idx, lt) in paged.layer_page_tables.iter().enumerate() {
            assert_eq!(
                lt[0].len(),
                expected,
                "layer {layer_idx} should have {expected} pages after rollback"
            );
        }
    }

    // ======================================================================
    // Sparse MLP tests (Plan 022: TwELL-inspired)
    // ======================================================================

    /// Sparse matmul produces identical output to dense at 0% sparsity (all alive).
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_sparse_matmul_0_percent_sparsity() {
        let rows = 16;
        let cols = 64;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
        let input: Vec<f32> = (0..cols).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut dense_out = vec![0.0f32; rows];
        let mut sparse_out = vec![0.0f32; rows];
        let mut indices = vec![0usize; cols];
        let mut values = vec![0.0f32; cols];

        crate::types::matmul(&mut dense_out, &weight, &input, rows, cols);
        crate::types::sparse_matmul(
            &mut sparse_out,
            &weight,
            &input,
            rows,
            cols,
            &mut indices,
            &mut values,
        );

        for i in 0..rows {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-3,
                "Mismatch at {i}: dense={}, sparse={}",
                dense_out[i],
                sparse_out[i]
            );
        }
    }

    /// Sparse matmul produces identical output at 95% sparsity.
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_sparse_matmul_95_percent_sparsity() {
        let rows = 16;
        let cols = 64;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
        let mut input = vec![0.0f32; cols];
        // 5% alive
        for i in (0..cols).step_by(20) {
            input[i] = 1.0;
        }
        let mut dense_out = vec![0.0f32; rows];
        let mut sparse_out = vec![0.0f32; rows];
        let mut indices = vec![0usize; cols];
        let mut values = vec![0.0f32; cols];

        crate::types::matmul(&mut dense_out, &weight, &input, rows, cols);
        crate::types::sparse_matmul(
            &mut sparse_out,
            &weight,
            &input,
            rows,
            cols,
            &mut indices,
            &mut values,
        );

        for i in 0..rows {
            assert!(
                (dense_out[i] - sparse_out[i]).abs() < 1e-4,
                "Mismatch at {i}: dense={}, sparse={}",
                dense_out[i],
                sparse_out[i]
            );
        }
    }

    /// Sparse matmul with 100% sparsity (all zeros) produces all-zero output.
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_sparse_matmul_100_percent_sparsity() {
        let rows = 16;
        let cols = 64;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
        let input = vec![0.0f32; cols];
        let mut sparse_out = vec![0.0f32; rows];
        let mut indices = vec![0usize; cols];
        let mut values = vec![0.0f32; cols];

        let alive = crate::types::sparse_matmul(
            &mut sparse_out,
            &weight,
            &input,
            rows,
            cols,
            &mut indices,
            &mut values,
        );

        assert_eq!(alive, 0, "Expected 0 alive neurons");
        for (i, &val) in sparse_out.iter().take(rows).enumerate() {
            assert_eq!(val, 0.0, "Expected zero output at {i}");
        }
    }

    /// ForwardContext buffers are correctly sized when sparse_mlp is enabled.
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_forward_context_sparse_buffers() {
        let config = crate::types::Config::micro();
        let ctx = super::ForwardContext::new(&config);
        assert_eq!(ctx.active_indices.len(), config.mlp_hidden);
        assert_eq!(ctx.active_values.len(), config.mlp_hidden);
    }

    /// Forward pass works correctly with sparse_mlp enabled.
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_forward_with_sparse_mlp() {
        let config = crate::types::Config::micro();
        let mut rng = crate::types::Rng::new(42);
        let weights = crate::transformer::TransformerWeights::new(&config, &mut rng);
        let mut ctx = crate::transformer::ForwardContext::new(&config);
        let mut cache = crate::transformer::MultiLayerKVCache::new(&config);

        let logits = crate::transformer::forward(&mut ctx, &weights, &mut cache, 26, 0, &config);

        // Verify logits are finite
        for l in logits {
            assert!(l.is_finite(), "Logit is not finite: {l}");
        }
    }

    /// Sparse matmul with negative values (should be treated as dead by ReLU context).
    #[cfg(feature = "sparse_mlp")]
    #[test]
    fn test_sparse_matmul_negative_input() {
        let rows = 8;
        let cols = 32;
        let weight: Vec<f32> = (0..rows * cols).map(|i| (i % 100) as f32 * 0.01).collect();
        let mut input = vec![0.0f32; cols];
        // Mix of positive, negative, zero
        input[0] = 1.0;
        input[1] = -1.0; // Should be ignored (not > 0)
        input[2] = 0.5;
        input[3] = -0.5; // Should be ignored
        // Rest are 0.0

        let mut dense_out = vec![0.0f32; rows];
        let mut sparse_out = vec![0.0f32; rows];
        let mut indices = vec![0usize; cols];
        let mut values = vec![0.0f32; cols];

        crate::types::matmul(&mut dense_out, &weight, &input, rows, cols);
        crate::types::sparse_matmul(
            &mut sparse_out,
            &weight,
            &input,
            rows,
            cols,
            &mut indices,
            &mut values,
        );

        // Both should match since matmul doesn't skip negatives but sparse_matmul skips input[c] <= 0
        // So we need to compare against a modified dense that also skips negatives
        for r in 0..rows {
            let mut expected = 0.0f32;
            for c in 0..cols {
                if input[c] > 0.0 {
                    expected += weight[r * cols + c] * input[c];
                }
            }
            assert!(
                (sparse_out[r] - expected).abs() < 1e-4,
                "Mismatch at {r}: sparse={}, expected={}",
                sparse_out[r],
                expected
            );
        }
    }

    // -----------------------------------------------------------------------
    // Plan 025: Bidirectional Prefill + Modality LoRA Switching
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_prefill_logits_finite() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens: Vec<usize> = (0..8).collect();
        #[cfg(not(feature = "domain_latent"))]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
            None,
        );
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "prefill logit {i} is not finite: {l}");
        }
    }

    #[test]
    fn test_forward_prefill_populates_cache() {
        let config = Config::micro();
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens: Vec<usize> = (0..5).collect();
        #[cfg(not(feature = "domain_latent"))]
        forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
            None,
        );
        // All 5 positions should have K/V in cache
        for p in 0..5 {
            let off = p * kvd;
            let key_sum: f32 = cache.layers[0].key[off..off + kvd].iter().sum();
            let val_sum: f32 = cache.layers[0].value[off..off + kvd].iter().sum();
            assert!(key_sum != 0.0, "K cache at pos {p} should be populated");
            assert!(val_sum != 0.0, "V cache at pos {p} should be populated");
        }
    }

    #[test]
    fn test_forward_prefill_logits_shape() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens: Vec<usize> = vec![0, 1, 2];
        #[cfg(not(feature = "domain_latent"))]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
            None,
        );
        assert_eq!(logits.len(), config.vocab_size);
    }

    #[test]
    fn test_forward_prefill_single_token() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens = vec![5];
        #[cfg(not(feature = "domain_latent"))]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
            None,
        );
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                l.is_finite(),
                "single-token prefill logit {i} not finite: {l}"
            );
        }
    }

    #[test]
    fn test_prefill_then_decode_shared_cache() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        // Prefill with 4 tokens
        let prompt: Vec<usize> = (0..4).collect();
        #[cfg(not(feature = "domain_latent"))]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &prompt,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &prompt,
            &config,
            None,
            None,
        );
        assert_eq!(logits.len(), config.vocab_size);

        // Decode from position 4 (should use same cache)
        let logits2 = forward(&mut ctx, &weights, &mut cache, 0, 4, &config);
        assert_eq!(logits2.len(), config.vocab_size);
        for (i, &l) in logits2.iter().enumerate() {
            assert!(
                l.is_finite(),
                "decode after prefill logit {i} not finite: {l}"
            );
        }
    }

    #[test]
    fn test_no_lora_matches_existing_forward() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Existing forward (no LoRA)
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config);

        // New forward_base with None (should be identical)
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        #[cfg(not(feature = "domain_latent"))]
        let logits2 = forward_base(&mut ctx2, &weights, &mut cache2, 0, 0, &config, None);
        #[cfg(feature = "domain_latent")]
        let logits2 = forward_base(&mut ctx2, &weights, &mut cache2, 0, 0, &config, None, None);

        for i in 0..config.vocab_size {
            let diff = (logits1[i] - logits2[i]).abs();
            assert!(
                diff < 5e-6,
                "forward and forward_base(None) differ at {i}: {diff}"
            );
        }
    }

    #[test]
    fn test_generate_with_prefill_produces_tokens() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        let prompt: Vec<usize> = (0..4).collect();
        let generated = {
            #[cfg(not(feature = "domain_latent"))]
            {
                generate_with_prefill(
                    &mut ctx,
                    &mut prefill,
                    &weights,
                    &mut cache,
                    &config,
                    &mut rng,
                    &prompt,
                    10,
                    &crate::types::LoraPair::none(),
                )
            }
            #[cfg(feature = "domain_latent")]
            {
                generate_with_prefill(
                    &mut ctx,
                    &mut prefill,
                    &weights,
                    &mut cache,
                    &config,
                    &mut rng,
                    &prompt,
                    10,
                    &crate::types::LoraPair::none(),
                    None,
                )
            }
        };

        assert!(!generated.is_empty(), "should generate at least one token");
        assert!(generated.len() <= 10, "should not exceed max_gen_tokens");
        for (i, &t) in generated.iter().enumerate() {
            assert!(t < config.vocab_size, "token {i} out of range: {t}");
        }
    }

    // -----------------------------------------------------------------------
    // Multi-layer prefill tests
    // -----------------------------------------------------------------------

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_generate_with_prefill_domain_latent() {
        let config = small_target_2layer();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);

        // Create a non-zero domain latent
        let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);

        let prompt: Vec<usize> = (0..4).collect();

        // Generate without domain latent
        let mut ctx1 = ForwardContext::new(&config);
        let mut prefill1 = PrefillContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let mut rng1 = Rng::new(42);
        let generated1 = generate_with_prefill(
            &mut ctx1,
            &mut prefill1,
            &weights,
            &mut cache1,
            &config,
            &mut rng1,
            &prompt,
            10,
            &crate::types::LoraPair::none(),
            None,
        );

        // Generate with domain latent (same seed)
        let mut ctx2 = ForwardContext::new(&config);
        let mut prefill2 = PrefillContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let mut rng2 = Rng::new(42);
        let generated2 = generate_with_prefill(
            &mut ctx2,
            &mut prefill2,
            &weights,
            &mut cache2,
            &config,
            &mut rng2,
            &prompt,
            10,
            &crate::types::LoraPair::none(),
            Some(&dl),
        );

        // Outputs should differ — domain latent modulates K/V at mid-layer
        assert_ne!(
            generated1, generated2,
            "domain latent should change generation output"
        );
    }

    fn small_target_2layer() -> Config {
        let mut c = Config::small_target();
        c.n_layer = 2;
        c
    }

    #[test]
    fn test_forward_prefill_multilayer_logits_finite() {
        let config = small_target_2layer();
        config.validate().unwrap();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens: Vec<usize> = (0..8).collect();
        #[cfg(not(feature = "domain_latent"))]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        let logits = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
            None,
        );
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(
                l.is_finite(),
                "multilayer prefill logit {i} not finite: {l}"
            );
        }
    }

    #[test]
    fn test_forward_prefill_multilayer_cache_populated() {
        let config = small_target_2layer();
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let tokens: Vec<usize> = (0..4).collect();
        #[cfg(not(feature = "domain_latent"))]
        forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
        );
        #[cfg(feature = "domain_latent")]
        forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &tokens,
            &config,
            None,
            None,
        );
        // Both layers should have K/V populated
        for layer in 0..2 {
            for p in 0..4 {
                let off = p * kvd;
                let key_sum: f32 = cache.layers[layer].key[off..off + kvd].iter().sum();
                let val_sum: f32 = cache.layers[layer].value[off..off + kvd].iter().sum();
                assert!(
                    key_sum != 0.0,
                    "layer {layer} K cache at pos {p} should be populated"
                );
                assert!(
                    val_sum != 0.0,
                    "layer {layer} V cache at pos {p} should be populated"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Domain Latent injection (Plan 038)
    // -----------------------------------------------------------------------

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_changes_logits() {
        let config = small_target_2layer(); // 2 layers, mid-layer = layer 1
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);

        // Without domain latent
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward_base(&mut ctx1, &weights, &mut cache1, 0, 0, &config, None, None);

        // With domain latent (non-zero embedding)
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);
        let logits2 = forward_base(
            &mut ctx2,
            &weights,
            &mut cache2,
            0,
            0,
            &config,
            None,
            Some(&dl),
        );

        // Logits should differ — domain latent modulates K/V at mid-layer
        let mut any_diff = false;
        for (&a, &b) in logits1.iter().zip(logits2.iter()) {
            if (a - b).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "domain latent should change logits");
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_zero_embedding_same_logits() {
        let config = small_target_2layer();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);

        // Without domain latent
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward_base(&mut ctx1, &weights, &mut cache1, 0, 0, &config, None, None);

        // With zero domain latent — should be identical
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let dl = crate::types::DomainLatent::zeros(kvd);
        let logits2 = forward_base(
            &mut ctx2,
            &weights,
            &mut cache2,
            0,
            0,
            &config,
            None,
            Some(&dl),
        );

        for (i, (&a, &b)) in logits1.iter().zip(logits2.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-6,
                "zero domain latent should not change logits, diff at {i}: {diff}"
            );
        }
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_prefill_changes_logits() {
        let config = small_target_2layer();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);
        let tokens: Vec<usize> = (0..4).collect();

        // Without domain latent
        let mut ctx1 = ForwardContext::new(&config);
        let mut prefill1 = PrefillContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward_prefill(
            &mut ctx1,
            &mut prefill1,
            &weights,
            &mut cache1,
            &tokens,
            &config,
            None,
            None,
        );

        // With domain latent
        let mut ctx2 = ForwardContext::new(&config);
        let mut prefill2 = PrefillContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let dl = crate::types::DomainLatent::from_vec(vec![0.3; kvd]);
        let logits2 = forward_prefill(
            &mut ctx2,
            &mut prefill2,
            &weights,
            &mut cache2,
            &tokens,
            &config,
            None,
            Some(&dl),
        );

        let mut any_diff = false;
        for (&a, &b) in logits1.iter().zip(logits2.iter()) {
            if (a - b).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(any_diff, "domain latent in prefill should change logits");
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_prefill_then_decode() {
        let config = small_target_2layer();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);
        let dl = crate::types::DomainLatent::from_vec(vec![0.2; kvd]);

        // Prefill with domain latent
        let mut ctx = ForwardContext::new(&config);
        let mut prefill = PrefillContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let prompt: Vec<usize> = (0..3).collect();
        let logits_prefill = forward_prefill(
            &mut ctx,
            &mut prefill,
            &weights,
            &mut cache,
            &prompt,
            &config,
            None,
            Some(&dl),
        );
        assert_eq!(logits_prefill.len(), config.vocab_size);
        for (i, &l) in logits_prefill.iter().enumerate() {
            assert!(
                l.is_finite(),
                "prefill with domain_latent logit {i} not finite: {l}"
            );
        }

        // Decode with domain latent (position 3)
        let logits_decode = forward_base(
            &mut ctx,
            &weights,
            &mut cache,
            0,
            3,
            &config,
            None,
            Some(&dl),
        );
        assert_eq!(logits_decode.len(), config.vocab_size);
        for (i, &l) in logits_decode.iter().enumerate() {
            assert!(
                l.is_finite(),
                "decode after prefill with domain_latent logit {i} not finite: {l}"
            );
        }
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_forward_with_domain_latent_wrapper() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);
        let dl = crate::types::DomainLatent::from_vec(vec![0.1; kvd]);

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);
        let logits = forward_with_domain_latent(
            &mut ctx,
            &weights,
            &mut cache,
            0,
            0,
            &config,
            None,
            Some(&dl),
        );
        assert_eq!(logits.len(), config.vocab_size);
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "logit {i} not finite: {l}");
        }
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_with_lora_changes_logits() {
        let config = small_target_2layer();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);
        let rank = 4;
        let in_dim = config.n_embd;
        let out_dim = config.n_embd;

        let lora = crate::types::LoraAdapter {
            a: vec![0.1f32; rank * in_dim],
            b: vec![0.1f32; out_dim * rank],
            rank,
            alpha: 8.0,
            in_dim,
            out_dim,
        };
        let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);

        // With both lora + domain_latent
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward_base(
            &mut ctx1,
            &weights,
            &mut cache1,
            0,
            0,
            &config,
            Some(&lora),
            Some(&dl),
        );

        // With lora only (no domain_latent)
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let logits2 = forward_base(
            &mut ctx2,
            &weights,
            &mut cache2,
            0,
            0,
            &config,
            Some(&lora),
            None,
        );

        let mut any_diff = false;
        for (&a, &b) in logits1.iter().zip(logits2.iter()) {
            if (a - b).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "domain_latent + lora should differ from lora-only"
        );
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_with_lora_prefill_pipeline() {
        let config = small_target_2layer();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);
        let rank = 4;
        let in_dim = config.n_embd;
        let out_dim = config.n_embd;

        let lora = crate::types::LoraAdapter {
            a: vec![0.1f32; rank * in_dim],
            b: vec![0.1f32; out_dim * rank],
            rank,
            alpha: 8.0,
            in_dim,
            out_dim,
        };
        let dl = crate::types::DomainLatent::from_vec(vec![0.5; kvd]);
        let tokens: Vec<usize> = (0..3).collect();

        // Pipeline 1: prefill + decode with both lora + dl
        let mut ctx1 = ForwardContext::new(&config);
        let mut prefill1 = PrefillContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let _ = forward_prefill(
            &mut ctx1,
            &mut prefill1,
            &weights,
            &mut cache1,
            &tokens,
            &config,
            Some(&lora),
            Some(&dl),
        );
        let logits1 = forward_base(
            &mut ctx1,
            &weights,
            &mut cache1,
            0,
            tokens.len(),
            &config,
            Some(&lora),
            Some(&dl),
        );

        // Pipeline 2: prefill + decode with lora only
        let mut ctx2 = ForwardContext::new(&config);
        let mut prefill2 = PrefillContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let _ = forward_prefill(
            &mut ctx2,
            &mut prefill2,
            &weights,
            &mut cache2,
            &tokens,
            &config,
            Some(&lora),
            None,
        );
        let logits2 = forward_base(
            &mut ctx2,
            &weights,
            &mut cache2,
            0,
            tokens.len(),
            &config,
            Some(&lora),
            None,
        );

        let mut any_diff = false;
        for (&a, &b) in logits1.iter().zip(logits2.iter()) {
            if (a - b).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "prefill+decode with lora+dl should differ from lora-only pipeline"
        );
    }

    #[cfg(feature = "domain_latent")]
    #[test]
    fn test_domain_latent_zero_with_lora_same_as_lora_only() {
        let config = small_target_2layer();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let kvd = crate::types::kv_dim(&config);
        let rank = 4;
        let in_dim = config.n_embd;
        let out_dim = config.n_embd;

        let lora = crate::types::LoraAdapter {
            a: vec![0.1f32; rank * in_dim],
            b: vec![0.1f32; out_dim * rank],
            rank,
            alpha: 8.0,
            in_dim,
            out_dim,
        };
        let dl_zero = crate::types::DomainLatent::zeros(kvd);

        // With zero domain_latent + lora
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward_base(
            &mut ctx1,
            &weights,
            &mut cache1,
            0,
            0,
            &config,
            Some(&lora),
            Some(&dl_zero),
        );

        // With lora only (no domain_latent)
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let logits2 = forward_base(
            &mut ctx2,
            &weights,
            &mut cache2,
            0,
            0,
            &config,
            Some(&lora),
            None,
        );

        for (i, (&a, &b)) in logits1.iter().zip(logits2.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff < 1e-6,
                "zero domain_latent + lora should match lora-only, diff at {i}: {diff}"
            );
        }
    }

    // ── Shared KV Cache (Phase 3, Plan 055) ─────────────────────

    #[test]
    fn test_preload_kv_cache_dimension_mismatch() {
        // bpe: n_kv_head=4, head_dim=8 → kv_dim=32
        // bpe_draft: n_kv_head=2, head_dim=8 → kv_dim=16
        let target_config = Config::bpe();
        let draft_config = Config::bpe_draft();

        let target_cache = MultiLayerKVCache::new(&target_config);
        let mut draft_cache = MultiLayerKVCache::new(&draft_config);

        // Preload should silently skip (kv_dim mismatch)
        preload_kv_cache(
            &mut draft_cache,
            &target_cache,
            1,
            &target_config,
            &draft_config,
        );

        // Draft cache should remain all zeros
        for layer in &draft_cache.layers {
            assert!(
                layer.key.iter().all(|&v| v == 0.0),
                "draft cache key should remain zero on dim mismatch"
            );
            assert!(
                layer.value.iter().all(|&v| v == 0.0),
                "draft cache value should remain zero on dim mismatch"
            );
        }
    }

    #[test]
    fn test_preload_kv_cache_matching_dims() {
        // Same config for both → kv_dim matches
        let config = Config::small_target();
        let kvd = crate::types::kv_dim(&config);

        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Populate target cache at pos 0 and pos 1
        let mut target_cache = MultiLayerKVCache::new(&config);
        let mut target_ctx = ForwardContext::new(&config);
        let _ = forward(&mut target_ctx, &weights, &mut target_cache, 0, 0, &config);
        let _ = forward(&mut target_ctx, &weights, &mut target_cache, 1, 1, &config);

        // Create empty draft cache
        let mut draft_cache = MultiLayerKVCache::new(&config);

        // Preload positions [0..2) from target
        preload_kv_cache(&mut draft_cache, &target_cache, 2, &config, &config);

        // Verify draft cache has target's KV for positions 0 and 1
        for (layer_idx, draft_layer) in draft_cache.layers.iter().enumerate() {
            let target_layer = &target_cache.layers[layer_idx];
            let copy_len = 2 * kvd;
            for i in 0..copy_len {
                assert_eq!(
                    draft_layer.key[i], target_layer.key[i],
                    "draft key mismatch at layer {layer_idx}, idx {i}"
                );
                assert_eq!(
                    draft_layer.value[i], target_layer.value[i],
                    "draft value mismatch at layer {layer_idx}, idx {i}"
                );
            }
        }
    }

    #[test]
    fn test_preload_kv_cache_zero_pos() {
        let config = Config::small_target();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut target_cache = MultiLayerKVCache::new(&config);
        let mut target_ctx = ForwardContext::new(&config);
        let _ = forward(&mut target_ctx, &weights, &mut target_cache, 0, 0, &config);

        let mut draft_cache = MultiLayerKVCache::new(&config);

        // Preload with pos=0 copies nothing (no positions to share)
        preload_kv_cache(&mut draft_cache, &target_cache, 0, &config, &config);

        // Draft cache should remain all zeros
        for layer in &draft_cache.layers {
            assert!(
                layer.key.iter().all(|&v| v == 0.0),
                "draft cache should remain zero with pos=0"
            );
        }
    }

    #[test]
    fn test_preload_kv_cache_fewer_draft_layers() {
        // Target: 2 layers, Draft: 1 layer — only layer 0 shared
        let target_config = Config {
            n_layer: 2,
            ..Config::small_target()
        };
        let draft_config = Config {
            n_layer: 1,
            ..Config::small_target()
        };

        let kvd = crate::types::kv_dim(&target_config);
        let mut rng = Rng::new(42);
        let target_weights = TransformerWeights::new(&target_config, &mut rng);

        let mut target_cache = MultiLayerKVCache::new(&target_config);
        let mut target_ctx = ForwardContext::new(&target_config);
        let _ = forward(
            &mut target_ctx,
            &target_weights,
            &mut target_cache,
            0,
            0,
            &target_config,
        );

        let mut draft_cache = MultiLayerKVCache::new(&draft_config);

        preload_kv_cache(
            &mut draft_cache,
            &target_cache,
            1,
            &target_config,
            &draft_config,
        );

        // Draft has 1 layer, only layer 0 should be copied
        assert_eq!(draft_cache.layers.len(), 1);
        let draft_layer = &draft_cache.layers[0];
        let target_layer = &target_cache.layers[0];
        for i in 0..kvd {
            assert_eq!(
                draft_layer.key[i], target_layer.key[i],
                "layer 0 key should be copied"
            );
            assert_eq!(
                draft_layer.value[i], target_layer.value[i],
                "layer 0 value should be copied"
            );
        }
    }

    /// T14: Verify hybrid behavior — drafter forwards with preloaded target KV.
    /// Past positions [0..pos) read from preloaded target KV,
    /// new position [pos] computed by drafter and written to its own cache.
    #[test]
    fn test_preload_kv_cache_hybrid_forward() {
        let config = Config::small_target();
        let kvd = crate::types::kv_dim(&config);
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Build target KV cache for positions 0 and 1
        let mut target_cache = MultiLayerKVCache::new(&config);
        let mut target_ctx = ForwardContext::new(&config);
        let _ = forward(&mut target_ctx, &weights, &mut target_cache, 0, 0, &config);
        let _ = forward(&mut target_ctx, &weights, &mut target_cache, 1, 1, &config);

        // Preload target KV [0..2) into draft cache
        let mut draft_cache = MultiLayerKVCache::new(&config);
        preload_kv_cache(&mut draft_cache, &target_cache, 2, &config, &config);

        // Drafter forwards at pos=2 with preloaded KV — should produce valid logits
        let mut draft_ctx = ForwardContext::new(&config);
        let logits = forward(&mut draft_ctx, &weights, &mut draft_cache, 2, 2, &config);

        // Logits must be finite (no NaN/Inf from garbage KV)
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] not finite: {v}");
        }

        // Draft cache now has: [0..2) from target, [2] from drafter
        for layer in &draft_cache.layers {
            // Position 2 should have non-zero KV (written by drafter)
            let pos2_off = 2 * kvd;
            let has_nonzero = layer.key[pos2_off..pos2_off + kvd]
                .iter()
                .any(|&v| v != 0.0);
            assert!(has_nonzero, "drafter should have written KV at pos 2");
        }
    }

    // --- T15–T19: Clustered LM Head Tests ---

    #[test]
    fn test_cluster_map_round_robin() {
        // 10 tokens, cluster_size=3 → 4 clusters: [0,1,2], [3,4,5], [6,7,8], [9]
        let map = cluster_map_round_robin(10, 3);
        assert_eq!(map.len(), 4);
        assert_eq!(map[0], vec![0, 1, 2]);
        assert_eq!(map[1], vec![3, 4, 5]);
        assert_eq!(map[2], vec![6, 7, 8]);
        assert_eq!(map[3], vec![9]);
    }

    #[test]
    fn test_cluster_map_round_robin_exact_division() {
        // 8 tokens, cluster_size=4 → 2 clusters
        let map = cluster_map_round_robin(8, 4);
        assert_eq!(map.len(), 2);
        assert_eq!(map[0], vec![0, 1, 2, 3]);
        assert_eq!(map[1], vec![4, 5, 6, 7]);
    }

    #[test]
    fn test_standard_lm_head_matches_matmul() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let n = config.n_embd;

        let mut logits_matmul = vec![0.0f32; config.vocab_size];
        let mut logits_standard = vec![0.0f32; config.vocab_size];
        let hidden: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

        matmul(
            &mut logits_matmul,
            &weights.lm_head,
            &hidden,
            config.vocab_size,
            n,
        );
        standard_lm_head(
            &mut logits_standard,
            &hidden,
            &weights.lm_head,
            config.vocab_size,
            n,
        );

        for i in 0..config.vocab_size {
            let diff = (logits_matmul[i] - logits_standard[i]).abs();
            assert!(diff < 1e-6, "standard_lm_head differs at {i}: {diff}");
        }
    }

    #[test]
    fn test_clustered_lm_head_only_cluster_tokens_finite() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let mut weights = TransformerWeights::new(&config, &mut rng);
        let n = config.n_embd;
        let cluster_size = 16;

        let cluster_map = cluster_map_round_robin(config.vocab_size, cluster_size);
        let num_clusters = cluster_map.len();
        let classifier: Vec<f32> = (0..num_clusters * n).map(|_| rng.normal()).collect();

        weights.mtp_cluster_classifier = Some(classifier);
        weights.mtp_cluster_map = Some(cluster_map.clone());

        let mut logits = vec![0.0f32; config.vocab_size];
        let hidden: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

        clustered_lm_head(
            &mut logits,
            &hidden,
            &weights.lm_head,
            weights.mtp_cluster_classifier.as_ref().unwrap(),
            weights.mtp_cluster_map.as_ref().unwrap(),
            config.vocab_size,
            n,
            1, // topk=1: backward compat (single cluster selection)
            &mut vec![0.0f32; config.vocab_size],
            &mut vec![(0usize, 0.0f32); config.vocab_size],
            &mut Vec::new(),
        );

        // Find winning cluster (the one with finite logits)
        let winning = cluster_map
            .iter()
            .find(|tokens| tokens.iter().all(|&t| logits[t].is_finite()))
            .expect("one cluster should have finite logits");

        // Cluster tokens: finite. Others: -inf
        let cluster_set: std::collections::HashSet<usize> = winning.iter().copied().collect();
        for (i, &logit) in logits.iter().enumerate() {
            if cluster_set.contains(&i) {
                assert!(logit.is_finite(), "token {i} in cluster should be finite");
            } else {
                assert_eq!(logit, f32::NEG_INFINITY, "token {i} should be -inf");
            }
        }
    }

    #[test]
    fn test_clustered_lm_head_logits_match_standard() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let mut weights = TransformerWeights::new(&config, &mut rng);
        let n = config.n_embd;
        let cluster_size = 16;

        let cluster_map = cluster_map_round_robin(config.vocab_size, cluster_size);
        let num_clusters = cluster_map.len();
        let classifier: Vec<f32> = (0..num_clusters * n).map(|_| rng.normal()).collect();

        weights.mtp_cluster_classifier = Some(classifier);
        weights.mtp_cluster_map = Some(cluster_map.clone());

        let hidden: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.1).collect();

        // Standard logits
        let mut logits_std = vec![0.0f32; config.vocab_size];
        standard_lm_head(
            &mut logits_std,
            &hidden,
            &weights.lm_head,
            config.vocab_size,
            n,
        );

        // Clustered logits
        let mut logits_clust = vec![0.0f32; config.vocab_size];
        clustered_lm_head(
            &mut logits_clust,
            &hidden,
            &weights.lm_head,
            weights.mtp_cluster_classifier.as_ref().unwrap(),
            weights.mtp_cluster_map.as_ref().unwrap(),
            config.vocab_size,
            n,
            1, // topk=1: backward compat (single cluster selection)
            &mut vec![0.0f32; config.vocab_size],
            &mut vec![(0usize, 0.0f32); config.vocab_size],
            &mut Vec::new(),
        );

        // Find winning cluster
        let winning = cluster_map
            .iter()
            .find(|tokens| tokens.iter().all(|&t| logits_clust[t].is_finite()))
            .expect("one cluster should win");

        // Clustered logits for winning tokens should match standard exactly
        for &t in winning {
            let diff = (logits_clust[t] - logits_std[t]).abs();
            assert!(diff < 1e-5, "logit[{t}] mismatch: diff={diff}");
        }
    }

    #[test]
    fn test_forward_base_clustered_dispatch() {
        // Config::bpe() has vocab=4096, threshold=4096 → 4096 >= 4096 activates
        // Use topk=1 so only 1 cluster is selected (produces -inf for non-cluster tokens)
        let mut config = Config::bpe();
        config.mtp_cluster_topk = 1;
        let mut rng = Rng::new(42);
        let mut weights = TransformerWeights::new(&config, &mut rng);

        let cluster_map = cluster_map_round_robin(config.vocab_size, config.mtp_cluster_size);
        let num_clusters = cluster_map.len();
        let classifier: Vec<f32> = (0..num_clusters * config.n_embd)
            .map(|_| rng.normal())
            .collect();
        weights.mtp_cluster_classifier = Some(classifier);
        weights.mtp_cluster_map = Some(cluster_map);

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

        // Clustered path active: some -inf, some finite
        let inf_count = logits.iter().filter(|&&v| v == f32::NEG_INFINITY).count();
        let finite_count = logits.iter().filter(|&&v| v.is_finite()).count();
        assert!(inf_count > 0, "should have -inf logits (clustered path)");
        assert!(
            finite_count > 0,
            "should have finite logits (cluster tokens)"
        );
        assert_eq!(inf_count + finite_count, config.vocab_size);
    }

    #[test]
    fn test_forward_base_standard_fallback_no_weights() {
        // Config::micro() has threshold=usize::MAX → never activates clustered path
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);

        let mut ctx = ForwardContext::new(&config);
        let mut cache = MultiLayerKVCache::new(&config);

        let logits = forward(&mut ctx, &weights, &mut cache, 0, 0, &config);

        // Standard path: all finite, no -inf
        for (i, &v) in logits.iter().enumerate() {
            assert!(v.is_finite(), "logit[{i}] should be finite: {v}");
        }
    }

    #[test]
    fn test_cluster_map_from_embeddings_fallback() {
        let wte = vec![0.0f32; 100 * 32];
        let map = cluster_map_from_embeddings(&wte, 100, 32, 25);
        let expected = cluster_map_round_robin(100, 25);
        assert_eq!(map, expected);
    }

    // ── Delta routing stability tests (Plan 134 T2) ─────────────

    /// GOAT proof: verifies that `depth_route` norm stability holds empirically
    /// across 36 simulated layers. See `depth_route` doc comment for the
    /// theoretical argument (Plan 134 T1/T3, MGR §3.2).
    #[test]
    #[cfg(feature = "delta_routing")]
    fn proof_depth_route_norm_stability() {
        let n_embd = 32;
        let n_sources = 4;

        // Create initial residual (simulating embedding output)
        let mut residual: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.1).sin()).collect();
        let initial_norm: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Create synthetic sources (layer deltas), query weights, norm weights
        let sources: Vec<Vec<f32>> = (0..n_sources)
            .map(|s| {
                (0..n_embd)
                    .map(|i| ((i + s * 7) as f32 * 0.05).cos() * 0.1)
                    .collect()
            })
            .collect();
        let source_refs: Vec<&[f32]> = sources.iter().map(|s| s.as_slice()).collect();
        let query_weight: Vec<f32> = (0..n_embd).map(|i| (i as f32 * 0.1).sin() * 0.01).collect();
        let norm_weight: Vec<f32> = vec![1.0; n_embd];
        let mut logits_buf = vec![0.0f32; n_sources];
        let mut scaled_buf = vec![0.0f32; n_embd];

        // Simulate 36 layers of additive routing
        for _ in 0..36 {
            depth_route(
                &mut residual,
                &source_refs,
                &query_weight,
                &norm_weight,
                &mut logits_buf,
                &mut scaled_buf,
                n_embd,
            );
        }

        let final_norm: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            final_norm <= 10.0 * initial_norm,
            "Norm grew beyond 10x: initial={}, final={}, ratio={}",
            initial_norm,
            final_norm,
            final_norm / initial_norm,
        );
    }
}
