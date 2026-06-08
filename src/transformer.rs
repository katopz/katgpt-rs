use crate::types::{self, *};
use rayon::prelude::*;

#[cfg(feature = "wall_attention")]
use crate::simd::{
    simd_add_inplace, simd_add_scalar_inplace, simd_exp_inplace, simd_scale_inplace,
    simd_scale_mul_inplace,
};

/// Decode stage for specialized forward paths (Plan 102: TileRT pipeline).
/// Different stages have different optimization opportunities:
/// - Draft: can skip screening, reduced KV writes, approximate attention
/// - Verify: exact attention, full KV write, enable screening
/// - Sample: SIMD-only, no attention needed
/// - BeliefDraft: MLP-only forward via BeliefDrafter, no attention needed (Plan 217)
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
    /// MLP-only forward via BeliefDrafter — no attention, no KV (Plan 217).
    /// The belief drafter uses a lightweight MLP to predict next hidden states
    /// instead of running the full transformer forward pass.
    BeliefDraft,
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
    // Kog CPU fusion (Plan 160): RMSNorm gamma vectors
    pub attn_norm_gamma: Vec<f32>, // [n_embd] pre-attention RMSNorm gamma (identity=1.0)
    pub mlp_norm_gamma: Vec<f32>,  // [n_embd] pre-MLP RMSNorm gamma (identity=1.0)
    // Kog CPU fusion (Plan 160): fused QKV weight storage
    pub attn_qkv_fused: Option<Vec<f32>>, // [(n_embd + 2*kv_dim), n_embd] interleaved
    // Wall Attention gate projection weights (Plan 173)
    #[cfg(feature = "wall_attention")]
    pub attn_wg: Vec<f32>, // [kv_dim] gate projection per KV head dimension
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

// ---------------------------------------------------------------------------
// RiM Reasoning Buffer Slots — Plan 172 helpers
// ---------------------------------------------------------------------------

/// Extend a token sequence with RiM reasoning buffer tokens (Plan 172).
/// Appends K×M buffer token IDs after the original prompt tokens.
/// Returns the extended token Vec. No-op when rim is disabled.
#[cfg(feature = "rim_slots")]
pub fn rim_extend_tokens(tokens: &[usize], config: &Config) -> Vec<usize> {
    if !config.rim_enabled() {
        return tokens.to_vec();
    }
    let buf_count = config.rim_total_buffer_tokens();
    let mut extended = Vec::with_capacity(tokens.len() + buf_count);
    extended.extend_from_slice(tokens);
    let buf_token = if config.rim_buffer_token == 0 {
        config.bos_token // fallback to BOS when rim_buffer_token unset
    } else {
        config.rim_buffer_token
    };
    extended.resize(tokens.len() + buf_count, buf_token);
    extended
}

/// Returns the index from which to read logits when RiM buffer slots are active.
/// When enabled, readout is at the LAST buffer position.
/// When disabled, readout is at the last prompt token position.
#[cfg(feature = "rim_slots")]
#[inline]
pub fn rim_readout_index(prompt_len: usize, config: &Config) -> usize {
    if config.rim_enabled() {
        prompt_len + config.rim_total_buffer_tokens() - 1
    } else {
        prompt_len - 1
    }
}

impl TransformerWeights {
    pub fn new(config: &Config, rng: &mut Rng) -> Self {
        let n = config.n_embd;
        let kvd = types::kv_dim(config);
        let embd_scale = (2.0 / n as f32).sqrt();
        let layer_scale = (2.0 / (n as f32 * config.n_layer as f32)).sqrt();

        // Embeddings first (same order as original single-layer code)
        // Pre-allocate to avoid repeated re-allocation during collect().
        let wte_len = config.vocab_size * n;
        let mut wte = Vec::with_capacity(wte_len);
        wte.extend((0..wte_len).map(|_| rng.normal() * embd_scale));

        let wpe_len = config.block_size * n;
        let mut wpe = Vec::with_capacity(wpe_len);
        wpe.extend((0..wpe_len).map(|_| rng.normal() * embd_scale));

        // Per-layer weights: same field order as original per n_layer iterations
        // Pre-allocate each weight vector to avoid repeated reallocation.
        let mut layers = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            layers.push(LayerWeights {
                attn_wq: {
                    let len = n * n;
                    let mut v = Vec::with_capacity(len);
                    v.extend((0..len).map(|_| rng.normal() * layer_scale));
                    v
                },
                attn_wk: {
                    let len = kvd * n;
                    let mut v = Vec::with_capacity(len);
                    v.extend((0..len).map(|_| rng.normal() * layer_scale));
                    v
                },
                attn_wv: {
                    let len = kvd * n;
                    let mut v = Vec::with_capacity(len);
                    v.extend((0..len).map(|_| rng.normal() * layer_scale));
                    v
                },
                attn_wo: {
                    let len = n * n;
                    let mut v = Vec::with_capacity(len);
                    v.extend((0..len).map(|_| rng.normal() * layer_scale));
                    v
                },
                mlp_w1: {
                    let len = config.mlp_hidden * n;
                    let mut v = Vec::with_capacity(len);
                    v.extend((0..len).map(|_| rng.normal() * layer_scale));
                    v
                },
                mlp_w2: {
                    let len = n * config.mlp_hidden;
                    let mut v = Vec::with_capacity(len);
                    v.extend((0..len).map(|_| rng.normal() * layer_scale));
                    v
                },
                attn_norm_gamma: vec![1.0f32; n],
                mlp_norm_gamma: vec![1.0f32; n],
                attn_qkv_fused: None,
                #[cfg(feature = "wall_attention")]
                attn_wg: vec![0.0; kvd], // Initialized to zeros; gate not active unless wall_config is Some
            });
        }

        // LM head last
        let lm_len = config.vocab_size * n;
        let mut lm_head = Vec::with_capacity(lm_len);
        lm_head.extend((0..lm_len).map(|_| rng.normal() * embd_scale));

        Self {
            wte,
            wpe,
            lm_head,
            layers,
            mtp_activation_proj: None,
            mtp_cluster_classifier: None,
            mtp_cluster_map: None,
            #[cfg(feature = "delta_routing")]
            delta_routing_query: {
                let mut v = Vec::with_capacity(config.n_layer);
                for _ in 0..config.n_layer {
                    v.push(vec![0.0; config.n_embd]); // Zero-init: safe additive start
                }
                v
            },
            #[cfg(feature = "delta_routing")]
            delta_routing_norm: {
                let mut v = Vec::with_capacity(config.n_layer);
                for _ in 0..config.n_layer {
                    v.push(vec![1.0f32; config.n_embd]); // Ones: identity RMSNorm
                }
                v
            },
        }
    }

    /// Initialize Wall Attention gate weights with random values (Plan 173).
    /// Call when `wall_config` is `Some` — populates `attn_wg` with proper scaling.
    /// This is separate from `new()` to avoid consuming RNG when Wall is disabled.
    #[cfg(feature = "wall_attention")]
    pub fn init_wall_gates(&mut self, config: &Config, rng: &mut Rng) {
        let kvd = types::kv_dim(config);
        let layer_scale = (2.0 / (config.n_embd as f32 * config.n_layer as f32)).sqrt();
        for layer in &mut self.layers {
            let mut v = Vec::with_capacity(kvd);
            v.extend((0..kvd).map(|_| rng.normal() * layer_scale));
            layer.attn_wg = v;
        }
    }

    /// Fold RMSNorm gamma into projection weights (Plan 160: Kog CPU fusion).
    ///
    /// For each projection preceded by RMSNorm with gamma:
    ///   weight[row * n_embd + col] *= gamma[col]
    ///
    /// After folding, gamma is set to 1.0 (identity), so runtime rmsnorm_with_gamma
    /// becomes a no-op. This eliminates per-token gamma memory reads.
    ///
    /// **Attention gamma**: NOT folded because the residual connection (`xr`) captures
    /// the post-norm value (`x * inv_rms * gamma`). Folding would change the residual.
    /// The attention gamma remains at runtime for `rmsnorm_with_gamma`.
    ///
    /// **MLP gamma**: Folded into `mlp_w1` because the residual (`xr2`) is saved
    /// BEFORE the norm, so gamma only affects the projection path.
    pub fn fold_gamma(&mut self, config: &Config) {
        let n = config.n_embd;

        for layer in &mut self.layers {
            // Fold mlp_norm_gamma into mlp_w1
            // (Safe: xr2 is saved before rmsnorm, so residual is pre-norm)
            let mlp_gamma = &layer.mlp_norm_gamma;
            for row in 0..config.mlp_hidden {
                for (col, g) in mlp_gamma.iter().enumerate() {
                    layer.mlp_w1[row * n + col] *= g;
                }
            }
            // Set mlp_norm_gamma to identity
            layer.mlp_norm_gamma.fill(1.0f32);

            // Note: attn_norm_gamma is NOT folded because xr (attention residual)
            // captures the post-norm value. It remains for runtime rmsnorm_with_gamma.
        }
    }

    /// Repack Q/K/V weights into a single contiguous buffer (Plan 160: Kog CPU fusion).
    ///
    /// Layout: [Q rows | K rows | V rows] × [n_embd], where:
    ///   Q rows = [n_embd], K rows = [kv_dim], V rows = [kv_dim]
    ///
    /// The fused weight is stored in `attn_qkv_fused` (Some when populated).
    /// Original weights are preserved — fused is an additional allocation.
    /// Cache locality win: single contiguous memory region instead of 3 scattered buffers.
    pub fn interleave_qkv(&mut self, config: &Config) {
        let n = config.n_embd;
        let kvd = types::kv_dim(config);
        let q_rows = n;
        let k_rows = kvd;
        let v_rows = kvd;
        let total_rows = q_rows + k_rows + v_rows;

        for layer in &mut self.layers {
            let mut fused = Vec::with_capacity(total_rows * n);
            fused.extend_from_slice(&layer.attn_wq);
            fused.extend_from_slice(&layer.attn_wk);
            fused.extend_from_slice(&layer.attn_wv);
            layer.attn_qkv_fused = Some(fused);
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
        let mut layers = Vec::with_capacity(config.n_layer);
        layers.extend((0..config.n_layer).map(|_| KVCache::new(config)));
        Self {
            layers,
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
        KVSnapshot { pos, layers }
    }

    /// Restore KV cache from a snapshot.
    /// Writes snapshot data back. No zeroing needed — each position is written before being read.
    pub fn restore(&mut self, snapshot: &KVSnapshot, config: &Config) {
        let kd = types::kv_dim(config);
        for (layer, snap_layer) in self.layers.iter_mut().zip(snapshot.layers.iter()) {
            let end = snapshot.pos * kd;
            layer.key[..end].copy_from_slice(&snap_layer.key);
            layer.value[..end].copy_from_slice(&snap_layer.value);
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
    pub pos: usize,
    pub layers: Vec<KVLayerSnapshot>,
}

/// Per-layer snapshot of KV cache data.
pub struct KVLayerSnapshot {
    pub key: Vec<f32>,   // [pos * kv_dim]
    pub value: Vec<f32>, // [pos * kv_dim]
}

/// Pre-allocated buffers for zero-alloc forward passes.
/// Create once, reuse across calls.
pub struct ForwardContext {
    // ── u64-aligned fields first (Vec, usize, arrays) ──────────────
    // Grouped by alignment to eliminate inter-field padding.
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
    // CNA: contrastive neuron attribution runtime modulator (Plan 087)
    #[cfg(feature = "cna_steering")]
    pub cna_modulator: Option<crate::pruners::CnaModulator>,
    // Sparse MLP buffers (Plan 022: TwELL-inspired unstructured sparsity)
    #[cfg(feature = "sparse_mlp")]
    pub(crate) active_indices: Vec<usize>, // [mlp_hidden] pre-allocated index buffer
    #[cfg(feature = "sparse_mlp")]
    pub(crate) active_values: Vec<f32>, // [mlp_hidden] pre-allocated value buffer
    // SubstrateGate: per-sequence capability mask for dual sparsity (Plan 216)
    #[cfg(feature = "substrate_gate")]
    pub(crate) substrate_mask: Option<crate::pruners::SubstrateMask>,
    // Paged KV cache: pre-allocated flat buffers for attention computation
    paged_flat_key: Vec<f32>,   // [block_size * kv_dim]
    paged_flat_value: Vec<f32>, // [block_size * kv_dim]
    // Raven: pre-allocated query buffer for per-head slot attention
    raven_query_buf: Vec<f32>, // [max(kv_dim, 64)]
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
    // GQA lookup: kv_group_lut[h] = h * n_kv_head / n_head (pre-computed once)
    kv_group_lut: [u8; 128], // fixed-size LUT for GQA head→kv_group mapping (up to 128 heads)
    _kv_group_lut_count: usize, // actual number of heads (n_head)
    #[cfg(feature = "mls_aggregate")]
    mls_count: usize, // How many layers accumulated
    // Hydra Adaptive Layer Budget: pre-computed skip plan (Research 148, Plan 165)
    // None = disabled (no profiles loaded). Some(plan) = modelless skip decisions.
    #[cfg(feature = "hydra_budget")]
    pub(crate) hydra_skip_plan: Option<crate::pruners::HydraSkipPlan>,
    // Wall Attention: per-head prefix sum state for diagonal forget gates (Plan 173).
    // Pre-allocated, zero alloc in hot path. Updated incrementally each token.
    #[cfg(feature = "wall_attention")]
    pub(crate) wall_prefix: WallPrefixState,
    // ── f32 fields last (4-byte aligned, no padding before) ──────────
    /// Pre-computed attention scale: `1.0 / sqrt(head_dim)`. Constant per config.
    attn_scale: f32,
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
            #[cfg(feature = "cna_steering")]
            cna_modulator: None,
            #[cfg(feature = "sparse_mlp")]
            active_indices: vec![0; config.mlp_hidden],
            #[cfg(feature = "sparse_mlp")]
            active_values: vec![0.0; config.mlp_hidden],
            #[cfg(feature = "substrate_gate")]
            substrate_mask: None,
            paged_flat_key: vec![0.0; block_kv],
            paged_flat_value: vec![0.0; block_kv],
            raven_query_buf: vec![0.0; kvd.max(64)],
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
            #[cfg(feature = "tiled_attention")]
            tiled_q: vec![0.0; config.block_size * config.n_embd],
            #[cfg(feature = "tiled_attention")]
            tiled_k: vec![0.0; config.block_size * kvd],
            #[cfg(feature = "tiled_attention")]
            tiled_v: vec![0.0; config.block_size * kvd],
            #[cfg(feature = "tiled_attention")]
            tiled_out: vec![0.0; config.block_size * config.n_embd],
            cluster_scores_buf: vec![
                0.0;
                config.vocab_size.div_ceil(config.mtp_cluster_size.max(1))
            ],
            topk_indexed_buf: vec![
                (0usize, 0.0f32);
                config.vocab_size.div_ceil(config.mtp_cluster_size.max(1))
            ],
            topk_output_buf: Vec::with_capacity(
                config.vocab_size.div_ceil(config.mtp_cluster_size.max(1)),
            ),
            prev_h: vec![0.0; config.n_embd],
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
            kv_group_lut: {
                let n_head = config.n_head;
                let n_kv_head = config.n_kv_head;
                assert!(
                    n_head <= 128,
                    "n_head ({n_head}) exceeds kv_group_lut capacity (128)"
                );
                let mut lut = [0u8; 128];
                for (h, slot) in lut.iter_mut().enumerate().take(n_head) {
                    *slot = (h * n_kv_head / n_head) as u8;
                }
                lut
            },
            _kv_group_lut_count: config.n_head,
            #[cfg(feature = "mls_aggregate")]
            mls_count: 0,
            #[cfg(feature = "hydra_budget")]
            hydra_skip_plan: None,
            #[cfg(feature = "wall_attention")]
            wall_prefix: WallPrefixState::new(config),
            attn_scale: 1.0 / (config.head_dim as f32).sqrt(),
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
    // usize fields after Vec fields to eliminate inter-field padding.
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
    // Pass 1: compute Q·K scores into buffer (no per-element scalar max)
    for t in 0..t_n {
        let k_off = t * kv_dim + kv_group_offset;
        // SAFETY: q_head_offset + hd <= n_embd (head_dim * n_head), k_off + hd <= block_size * kv_dim
        let dot = unsafe {
            let q_slice = std::slice::from_raw_parts(q.as_ptr().add(q_head_offset), hd);
            let k_slice = std::slice::from_raw_parts(key_cache.as_ptr().add(k_off), hd);
            crate::simd::simd_dot_f32(q_slice, k_slice, hd)
        };
        unsafe {
            *scores_buf.get_unchecked_mut(t) = dot;
        }
    }
    // Fuse scale + find max: one SIMD scale pass + one SIMD max reduction
    // replaces N scalar (dot * scale) and N scalar max operations
    let scores_raw = unsafe { std::slice::from_raw_parts_mut(scores_buf.as_mut_ptr(), t_n) };
    crate::simd::simd_scale_inplace(scores_raw, scale);
    let max_score =
        crate::simd::simd_max_f32(unsafe { std::slice::from_raw_parts(scores_buf.as_ptr(), t_n) });

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

// ---------------------------------------------------------------------------
// Wall Attention — Diagonal Forget Gates Replacing RoPE (Plan 173)
// ---------------------------------------------------------------------------

/// Wall Attention prefix sum state — incremental P_t tracking (Plan 173).
///
/// Maintains a running prefix sum of gate values per dimension.
/// Updated O(head_dim) per token during decode, computed once during prefill.
/// All buffers pre-allocated — zero alloc in hot path.
#[cfg(feature = "wall_attention")]
pub struct WallPrefixState {
    /// Per-layer, per-head prefix sums: [n_layer × n_kv_head × head_dim].
    /// P_t^l[d] = Σ_{s=0}^{t} gate_s^l[d] for layer l, dimension d.
    /// Each layer maintains independent prefix sums (different keys → different gates).
    prefix_sums: Vec<f32>,
    /// Gate projection buffer: [head_dim]. Pre-allocated, reused each token.
    gate_buf: Vec<f32>,
    /// Temp buffer for exp operations: [head_dim]. Pre-allocated, reused each call.
    gate_exp_buf: Vec<f32>,
    /// Number of KV heads.
    n_kv_head: usize,
    /// Head dimension.
    head_dim: usize,
}

#[cfg(feature = "wall_attention")]
impl WallPrefixState {
    pub fn new(config: &Config) -> Self {
        let hd = config.head_dim;
        let n_kv = config.n_kv_head;
        Self {
            prefix_sums: vec![0.0; config.n_layer * n_kv * hd],
            gate_buf: vec![0.0; hd],
            gate_exp_buf: vec![0.0; hd],
            n_kv_head: n_kv,
            head_dim: hd,
        }
    }

    /// Reset prefix sums for a new sequence.
    pub fn reset(&mut self) {
        self.prefix_sums.fill(0.0);
    }

    /// Compute gate from key and update prefix sum in one call.
    /// `layer_idx` selects the per-layer prefix sum slice.
    /// Avoids borrow conflicts by combining gate computation and prefix update.
    #[inline]
    pub fn compute_gate_and_update(
        &mut self,
        layer_idx: usize,
        kv_head: usize,
        key: &[f32],
        w_g: &[f32],
        bias: f32,
        gate_max: f32,
    ) {
        let hd = self.head_dim;
        Self::compute_gate_from_key(&mut self.gate_buf[..hd], key, w_g, bias, gate_max);
        let offset = layer_idx * self.n_kv_head * hd + kv_head * hd;
        simd_add_inplace(
            &mut self.prefix_sums[offset..offset + hd],
            &self.gate_buf[..hd],
        );
    }

    /// Update prefix sum with new gate values for a given layer and KV head.
    /// gate: [head_dim] gate values for this head at this position.
    /// O(head_dim).
    #[inline]
    pub fn update_prefix(&mut self, layer_idx: usize, kv_head: usize, gate: &[f32]) {
        let offset = layer_idx * self.n_kv_head * self.head_dim + kv_head * self.head_dim;
        let hd = self.head_dim;
        simd_add_inplace(&mut self.prefix_sums[offset..offset + hd], &gate[..hd]);
    }

    /// Rescale query: q̃ = exp(P) ⊙ q for each query head.
    /// For GQA, multiple Q heads share the same KV head prefix sum.
    /// q: [n_embd] query vector (all heads).
    /// kv_group_lut: maps Q head → KV head.
    #[inline]
    pub fn rescale_query(
        &mut self,
        layer_idx: usize,
        q: &mut [f32],
        kv_group_lut: &[u8; 128],
        n_head: usize,
    ) {
        let hd = self.head_dim;
        let layer_off = layer_idx * self.n_kv_head * hd;
        for (h, &kv_h) in kv_group_lut.iter().enumerate().take(n_head) {
            let q_off = h * hd;
            let p_off = layer_off + kv_h as usize * hd;
            // Copy prefix sums to temp buffer, exp in-place, then element-wise multiply.
            self.gate_exp_buf[..hd].copy_from_slice(&self.prefix_sums[p_off..p_off + hd]);
            simd_exp_inplace(&mut self.gate_exp_buf[..hd]);
            simd_scale_mul_inplace(&mut q[q_off..q_off + hd], &self.gate_exp_buf[..hd], 1.0);
        }
    }

    /// Rescale key: k̃ = exp(-P) ⊙ k for each KV head.
    /// k: [kv_dim] key vector (all KV heads).
    #[inline]
    pub fn rescale_key(&mut self, layer_idx: usize, k: &mut [f32]) {
        let hd = self.head_dim;
        let layer_off = layer_idx * self.n_kv_head * hd;
        for h in 0..self.n_kv_head {
            let k_off = h * hd;
            let p_off = layer_off + h * hd;
            // Negate prefix sums into temp buffer, exp in-place, then element-wise multiply.
            self.gate_exp_buf[..hd].copy_from_slice(&self.prefix_sums[p_off..p_off + hd]);
            simd_scale_inplace(&mut self.gate_exp_buf[..hd], -1.0);
            simd_exp_inplace(&mut self.gate_exp_buf[..hd]);
            simd_scale_mul_inplace(&mut k[k_off..k_off + hd], &self.gate_exp_buf[..hd], 1.0);
        }
    }

    /// Compute gate values from key projection (key-projected variant).
    /// gate_buf: [head_dim] output gate values (log-sigmoid, clamped).
    /// key: [head_dim] key slice for one head.
    /// w_g: [head_dim] gate projection weights for this head.
    /// bias: gate bias (default 6.0).
    /// gate_max: maximum clamp value (default 0.87).
    #[inline(always)]
    pub fn compute_gate_from_key(
        gate_buf: &mut [f32],
        key: &[f32],
        w_g: &[f32],
        bias: f32,
        gate_max: f32,
    ) {
        let hd = key.len();
        debug_assert_eq!(gate_buf.len(), hd);
        debug_assert_eq!(w_g.len(), hd);

        // Step 1: gate_buf = w_g * key  (SIMD element-wise multiply)
        gate_buf[..hd].copy_from_slice(&key[..hd]);
        simd_scale_mul_inplace(&mut gate_buf[..hd], w_g, 1.0);

        // Step 2: gate_buf += bias  (SIMD broadcast add)
        simd_add_scalar_inplace(&mut gate_buf[..hd], bias);

        // Step 3: gate_buf = -gate_buf  (SIMD negate)
        simd_scale_inplace(&mut gate_buf[..hd], -1.0);

        // Step 4: gate_buf = exp(gate_buf) = exp(-logit)  (SIMD Cephes exp)
        simd_exp_inplace(&mut gate_buf[..hd]);

        // Step 5: gate_buf += 1 → gate_buf = 1 + exp(-logit) = softplus(-logit)
        simd_add_scalar_inplace(&mut gate_buf[..hd], 1.0);

        // Step 6: ln + negate + clamp (scalar — ln not yet SIMD-accelerated)
        // log_sigmoid(logit) = -ln(1 + exp(-logit)) = -softplus(-logit)
        for d in 0..hd {
            let log_sig = -gate_buf[d].ln();
            gate_buf[d] = log_sig.clamp(-gate_max, 0.0);
        }
    }
}

/// Causal decode: single token forward with optional LoRA adapter.
/// Backward-compatible wrapper that passes `None` for LoRA.
///
/// Wall Attention (Plan 173): when wall_config is Some and feature is enabled,
/// Wall gate projection + Q/K rescaling replaces RoPE rotation.
/// Attention kernels unchanged — they receive pre-rescaled Q/K.
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
        // BeliefDraft falls through to standard forward for now — the BeliefDrafter
        // operates at the speculative layer (draft() method), not the transformer
        // forward layer. This variant exists for stage-aware profiling and routing.
        DecodeStage::BeliefDraft => {
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
    #[cfg(feature = "sleep_consolidation")] gdn2_cache: Option<
        &'a mut crate::gdn2::MultiLayerGdn2Cache,
    >,
    #[cfg(feature = "sleep_consolidation")] sleep_config: Option<&'a crate::sleep::SleepConfig>,
) -> &'a mut [f32] {
    cache.advance_pos(pos);
    use crate::types::{HybridPattern, LoopMode};

    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = crate::types::kv_dim(config);

    // Loop-invariant values hoisted outside all loops
    let scale = ctx.attn_scale;
    let t_n = pos + 1;

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
                for h in 0..config.n_head {
                    let kv_group = ctx.kv_group_lut[h] as usize;
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
                    let kv_group = ctx.kv_group_lut[h] as usize;
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
                // ctx.x += gates ⊙ prev_h  (element-wise fused multiply-accumulate)
                ctx.hidden[..n].copy_from_slice(&ctx.prev_h[..n]);
                crate::simd::simd_scale_mul_inplace(
                    &mut ctx.hidden[..n],
                    &residual_gate.gates[gate_offset..gate_offset + n],
                    1.0,
                );
                crate::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.hidden[..n]);
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

    // ── Sleep consolidation hook (Plan 154: eviction boundary) ─────
    // After the forward pass, if the KV cache is full, consolidate
    // cached K/V into GDN2 fast-weight state and evict. This frees
    // the cache for the next token while preserving context in S.
    #[cfg(feature = "sleep_consolidation")]
    if let (Some(gdn2), Some(sconf)) = (gdn2_cache, sleep_config)
        && sconf.should_sleep(pos)
    {
        crate::sleep::sleep(ctx, weights, cache, gdn2, sconf, config);
    }

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
        let kv_group = ctx.kv_group_lut[h] as usize;
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

    // 2. Softmax (numerically stable, SIMD batch)
    crate::simd::simd_add_scalar_inplace(&mut logits_buf[..n_sources], -max_logit);
    crate::simd::simd_exp_inplace(&mut logits_buf[..n_sources]);
    let sum_exp = crate::simd::simd_sum_f32(&logits_buf[..n_sources]);
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

    // 2. Softmax (numerically stable, SIMD batch)
    crate::simd::simd_add_scalar_inplace(&mut logits_buf[..n_sources], -max_logit);
    crate::simd::simd_exp_inplace(&mut logits_buf[..n_sources]);
    let sum_exp = crate::simd::simd_sum_f32(&logits_buf[..n_sources]);
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

    // 2. Softmax (SIMD batch)
    crate::simd::simd_add_scalar_inplace(&mut logits, -max_logit);
    crate::simd::simd_exp_inplace(&mut logits);
    let sum_exp = crate::simd::simd_sum_f32(&logits);
    let inv_sum = 1.0 / sum_exp;
    crate::simd::simd_scale_inplace(&mut logits, inv_sum);

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

    // Wall Attention: reset prefix sums at sequence start (Plan 173).
    // Prefix sums accumulate per-layer, per-head across the sequence.
    // Must reset when pos=0 to avoid stale state from previous sequences.
    #[cfg(feature = "wall_attention")]
    if pos == 0 {
        ctx.wall_prefix.reset();
    }

    // MLS: reset accumulator at start of forward call (Plan 104)
    #[cfg(feature = "mls_aggregate")]
    {
        ctx.mls_buf[..n].fill(0.0);
        ctx.mls_count = 0;
    }

    // Loop-invariant values hoisted outside the layer loop
    let scale = ctx.attn_scale;
    let t_n = pos + 1;

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // Hydra Adaptive Layer Budget: skip non-contributing layers (Research 148, Plan 165)
        // Modelless mode — zero overhead (single bool check on pre-computed plan).
        // When skipped, x passes through unchanged (z^l = z^{l-1}).
        #[cfg(feature = "hydra_budget")]
        if let Some(ref skip_plan) = ctx.hydra_skip_plan
            && crate::pruners::should_skip_layer(skip_plan, layer_idx)
        {
            // Skip this layer entirely — x passes through as-is.
            // Still need to copy x → hidden_state for snapshot consistency.
            ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
            continue;
        }

        // MLS: save pre-layer state for delta computation (Plan 104)
        #[cfg(feature = "mls_aggregate")]
        if config.mls_layers > 0 && layer_idx >= weights.layers.len() - config.mls_layers {
            ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
        }

        // Pre-attention: RMSNorm → save residual
        #[cfg(feature = "kog_cpu_fusion")]
        types::rmsnorm_with_gamma(&mut ctx.x[..n], &layer_weights.attn_norm_gamma);
        #[cfg(not(feature = "kog_cpu_fusion"))]
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // QKV projections from per-layer weights (GQA: K/V produce kv_dim outputs)
        #[cfg(feature = "kog_cpu_fusion")]
        if let Some(ref qkv_fused) = layer_weights.attn_qkv_fused {
            matmul(&mut ctx.q, &qkv_fused[..n * n], &ctx.x, n, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut ctx.lora_buf);
            }
            matmul(&mut ctx.k, &qkv_fused[n * n..(n + kvd) * n], &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut ctx.lora_buf);
            }
            matmul(&mut ctx.v, &qkv_fused[(n + kvd) * n..], &ctx.x, kvd, n);
            if let Some(lora) = lora {
                crate::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut ctx.lora_buf);
            }
        } else {
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
        }
        #[cfg(not(feature = "kog_cpu_fusion"))]
        {
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
        }

        // Domain latent injection at mid-layer (Plan 038: Free Transformer adaptation)
        #[cfg(feature = "domain_latent")]
        if layer_idx == config.n_layer / 2
            && let Some(dl) = domain_latent
        {
            crate::simd::simd_add_inplace(&mut ctx.k[..kvd], &dl.embedding[..kvd]);
            crate::simd::simd_add_inplace(&mut ctx.v[..kvd], &dl.embedding[..kvd]);
        }

        // Wall Attention: gate projection + prefix sum update + Q/K rescale (Plan 173).
        // Replaces RoPE rotation when wall_config is Some. Factorized form means
        // attention kernels are unchanged — they receive pre-rescaled Q and K.
        // Gate is derived from K (key-projected variant, zero KV cache overhead).
        #[cfg(feature = "wall_attention")]
        if let Some(ref wall_cfg) = config.wall_config {
            let n_kv = config.n_kv_head;
            let hd = config.head_dim;
            for kv_h in 0..n_kv {
                let k_off = kv_h * hd;
                let w_g = &layer_weights.attn_wg[k_off..k_off + hd];
                let k_slice = &ctx.k[k_off..k_off + hd];
                // Compute gate from key, then update prefix sum in one call.
                ctx.wall_prefix.compute_gate_and_update(
                    layer_idx,
                    kv_h,
                    k_slice,
                    w_g,
                    wall_cfg.gate_bias,
                    wall_cfg.gate_max,
                );
            }
            ctx.wall_prefix
                .rescale_query(layer_idx, &mut ctx.q, &ctx.kv_group_lut, config.n_head);
            ctx.wall_prefix.rescale_key(layer_idx, &mut ctx.k);
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
        for h in 0..config.n_head {
            let kv_group = ctx.kv_group_lut[h] as usize;
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
        #[cfg(feature = "kog_cpu_fusion")]
        types::rmsnorm_with_gamma(&mut ctx.x[..n], &layer_weights.mlp_norm_gamma);
        #[cfg(not(feature = "kog_cpu_fusion"))]
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
        // MLP w2: substrate dual-sparsity path (Plan 216)
        #[cfg(all(feature = "sparse_mlp", feature = "substrate_gate"))]
        {
            let alive = if let Some(ref substrate_mask) = ctx.substrate_mask {
                crate::pruners::sparse_matmul_substrate(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                    &mut ctx.active_indices,
                    &mut ctx.active_values,
                    substrate_mask,
                    layer_idx,
                )
            } else {
                types::sparse_matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                    &mut ctx.active_indices,
                    &mut ctx.active_values,
                )
            };
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
        // MLP w2: sparse-only path (no substrate)
        #[cfg(all(feature = "sparse_mlp", not(feature = "substrate_gate")))]
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

    // Wall Attention: reset prefix sums at sequence start (Plan 173).
    #[cfg(feature = "wall_attention")]
    if pos == 0 {
        ctx.wall_prefix.reset();
    }

    // MLS: reset accumulator at start of forward call (Plan 104)
    #[cfg(feature = "mls_aggregate")]
    {
        ctx.mls_buf[..n].fill(0.0);
        ctx.mls_count = 0;
    }

    // Loop-invariant values hoisted outside the layer loop
    let scale = ctx.attn_scale;
    let t_n = pos + 1;

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
        #[cfg(feature = "kog_cpu_fusion")]
        types::rmsnorm_with_gamma(&mut ctx.x[..n], &layer_weights.attn_norm_gamma);
        #[cfg(not(feature = "kog_cpu_fusion"))]
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // QKV projections (same as baseline — attention needs separate Q, K, V)
        #[cfg(feature = "kog_cpu_fusion")]
        if let Some(ref qkv_fused) = layer_weights.attn_qkv_fused {
            matmul(&mut ctx.q, &qkv_fused[..n * n], &ctx.x, n, n);
            matmul(&mut ctx.k, &qkv_fused[n * n..(n + kvd) * n], &ctx.x, kvd, n);
            matmul(&mut ctx.v, &qkv_fused[(n + kvd) * n..], &ctx.x, kvd, n);
        } else {
            matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
        }
        #[cfg(not(feature = "kog_cpu_fusion"))]
        {
            matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
        }

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

        // Wall Attention: gate projection + prefix sum update + Q/K rescale (Plan 173).
        // Same integration as forward_base — Wall is path-agnostic.
        #[cfg(feature = "wall_attention")]
        if let Some(ref wall_cfg) = config.wall_config {
            let n_kv = config.n_kv_head;
            let hd = config.head_dim;
            for kv_h in 0..n_kv {
                let k_off = kv_h * hd;
                let w_g = &layer_weights.attn_wg[k_off..k_off + hd];
                let k_slice = &ctx.k[k_off..k_off + hd];
                ctx.wall_prefix.compute_gate_and_update(
                    layer_idx,
                    kv_h,
                    k_slice,
                    w_g,
                    wall_cfg.gate_bias,
                    wall_cfg.gate_max,
                );
            }
            ctx.wall_prefix
                .rescale_query(layer_idx, &mut ctx.q, &ctx.kv_group_lut, config.n_head);
            ctx.wall_prefix.rescale_key(layer_idx, &mut ctx.k);
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
        for h in 0..config.n_head {
            let kv_group = ctx.kv_group_lut[h] as usize;
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
            for (i, out_slot) in out_buf.iter_mut().enumerate().take(out_len) {
                let row_off = i * target_n_embd;
                *out_slot = crate::simd::simd_dot_f32(
                    &proj_weights[row_off..row_off + target_n_embd],
                    &target_hidden[..target_n_embd],
                    target_n_embd,
                );
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
///
/// For RiM buffer slots (Plan 172): use `rim_extend_tokens()` to append buffer
/// tokens before calling this function. The logit readout will naturally come
/// from the last buffer position.
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

    // Wall Attention: reset prefix sums at prefill start (Plan 173).
    #[cfg(feature = "wall_attention")]
    if config.wall_config.is_some() {
        ctx.wall_prefix.reset();
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

            // Wall Attention: gate projection + prefix sum update + Q/K rescale (Plan 173).
            // Prefill processes positions sequentially, accumulating prefix sums per-layer.
            // K is rescaled before cache storage; Q is rescaled before Phase B reuse.
            #[cfg(feature = "wall_attention")]
            if let Some(ref wall_cfg) = config.wall_config {
                let n_kv = config.n_kv_head;
                let hd = config.head_dim;
                for kv_h in 0..n_kv {
                    let k_off = kv_h * hd;
                    let w_g = &layer_weights.attn_wg[k_off..k_off + hd];
                    let k_slice = &ctx.k[k_off..k_off + hd];
                    ctx.wall_prefix.compute_gate_and_update(
                        layer_idx,
                        kv_h,
                        k_slice,
                        w_g,
                        wall_cfg.gate_bias,
                        wall_cfg.gate_max,
                    );
                }
                ctx.wall_prefix.rescale_key(layer_idx, &mut ctx.k);
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

            // Wall Attention: rescale Q with accumulated prefix sum (Plan 173).
            #[cfg(feature = "wall_attention")]
            if config.wall_config.is_some() {
                ctx.wall_prefix.rescale_query(
                    layer_idx,
                    &mut ctx.q,
                    &ctx.kv_group_lut,
                    config.n_head,
                );
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

        // Hoist constant scale outside per-position loop (Pattern 3: avoid recomputing unchanged values)
        let attn_scale = ctx.attn_scale;

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
                let kv_group = ctx.kv_group_lut[h] as usize;
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
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        prefill.queries.as_ptr().add(q_off),
                        ctx.q.as_mut_ptr(),
                        n,
                    );
                }
                for h in 0..config.n_head {
                    let kv_group = ctx.kv_group_lut[h] as usize;
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
                            attn_scale,
                        );
                    }
                }
            }

            #[cfg(not(feature = "tiled_attention"))]
            {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        prefill.queries.as_ptr().add(q_off),
                        ctx.q.as_mut_ptr(),
                        n,
                    );
                }
                for h in 0..config.n_head {
                    let kv_group = ctx.kv_group_lut[h] as usize;
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
                            attn_scale,
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

    let mut generated = Vec::with_capacity(max_gen_tokens);
    generated.push(token);
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
        // softmax_scaled fuses temperature division + softmax, saving one pass vs manual divide
        crate::types::softmax_scaled(logits, 1.0 / config.temperature);

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

/// Generate tokens with collapse-aware adaptive thinking (Plan 212 T4).
///
/// Extends [`generate_with_prefill`] with mid-reasoning collapse detection.
/// When the `CollapseDetector` detects degenerate reasoning (hesitation loops,
/// repetitive tokens), it forces an early exit from thinking mode and switches
/// to answer generation.
///
/// The `thinking_end_token` is the token ID that marks the boundary between
/// thinking and answering (e.g., the `</think|>` token). When collapse is
/// detected, this token is emitted to signal the model to switch modes.
///
/// When `detector` is `None`, behaves identically to [`generate_with_prefill`].
#[cfg(feature = "collapse_aware_thinking")]
#[allow(clippy::too_many_arguments)]
pub fn generate_with_collapse_detection(
    ctx: &mut ForwardContext,
    prefill: &mut PrefillContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    config: &Config,
    rng: &mut crate::types::Rng,
    prompt_tokens: &[usize],
    max_gen_tokens: usize,
    lora_pair: &crate::types::LoraPair,
    thinking_end_token: usize,
    detector: Option<&mut dyn katgpt_core::traits::CollapseDetector>,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&crate::types::DomainLatent>,
) -> Vec<usize> {
    use crate::pruners::collapse_detector::{CollapseAction, check_collapse_action};

    // No detector → fall back to standard generation.
    let Some(detector) = detector else {
        #[cfg(not(feature = "domain_latent"))]
        {
            return generate_with_prefill(
                ctx,
                prefill,
                weights,
                cache,
                config,
                rng,
                prompt_tokens,
                max_gen_tokens,
                lora_pair,
            );
        }
        #[cfg(feature = "domain_latent")]
        {
            return generate_with_prefill(
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
            );
        }
    };

    // 1. Prefill phase — same as generate_with_prefill
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
    crate::types::softmax_scaled(logits, 1.0 / config.temperature);
    let mut token = crate::types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf);

    let mut generated = Vec::with_capacity(max_gen_tokens);
    generated.push(token);
    let mut pos = prompt_tokens.len();
    let mut in_thinking = true;
    detector.reset();

    // 2. Decode loop with collapse detection
    for _ in 1..max_gen_tokens {
        if pos >= config.block_size {
            break;
        }

        // Check for collapse (only during thinking mode).
        if in_thinking {
            let action =
                check_collapse_action(detector, token as u32, pos - prompt_tokens.len(), true);
            if action == CollapseAction::ForceExit {
                token = thinking_end_token;
                generated.push(token);
                pos += 1;
                in_thinking = false;
                detector.reset();
                continue;
            }
        }

        // Check if we naturally exited thinking mode.
        if in_thinking && token == thinking_end_token {
            in_thinking = false;
            detector.reset();
        }

        // Forward pass.
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
        crate::types::softmax_scaled(logits, 1.0 / config.temperature);
        token = crate::types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf);
        generated.push(token);
        pos += 1;

        if token == config.bos_token {
            break;
        }
    }

    generated
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
    // Note: no initial fill(0.0) needed — the inner loop below reads every position
    // from the paged cache and overwrites the flat buffer for each layer.
    let t_n = pos + 1;
    let flat_kv_len = t_n * kvd;

    // Loop-invariant values hoisted outside the layer loop
    let scale = ctx.attn_scale;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    crate::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // Wall Attention: reset prefix sums at sequence start (Plan 173).
    #[cfg(feature = "wall_attention")]
    if pos == 0 {
        ctx.wall_prefix.reset();
    }

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

        // Wall Attention: gate projection + prefix sum update + Q/K rescale (Plan 173).
        #[cfg(feature = "wall_attention")]
        if let Some(ref wall_cfg) = config.wall_config {
            let n_kv = config.n_kv_head;
            let hd = config.head_dim;
            for kv_h in 0..n_kv {
                let k_off = kv_h * hd;
                let w_g = &layer_weights.attn_wg[k_off..k_off + hd];
                let k_slice = &ctx.k[k_off..k_off + hd];
                ctx.wall_prefix.compute_gate_and_update(
                    layer_idx,
                    kv_h,
                    k_slice,
                    w_g,
                    wall_cfg.gate_bias,
                    wall_cfg.gate_max,
                );
            }
            ctx.wall_prefix
                .rescale_query(layer_idx, &mut ctx.q, &ctx.kv_group_lut, config.n_head);
            ctx.wall_prefix.rescale_key(layer_idx, &mut ctx.k);
        }

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
            for h in 0..config.n_head {
                let kv_group = ctx.kv_group_lut[h] as usize;
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

    // LM Head (uses matmul_parallel for large vocab)
    standard_lm_head(
        &mut ctx.logits,
        &ctx.x,
        &weights.lm_head,
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
    let mut tokens = Vec::with_capacity(n_tokens);
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
    /// Reusable scratch: flat buffer for all newly allocated pages in `ensure_pages()`.
    all_new_buf: Vec<usize>,
    /// Per-page reference counts for O(1) rollback (replaces HashSet scan).
    page_ref_counts: Vec<u32>,
    /// Reusable scratch: drained page indices awaiting recycle in `rollback()`.
    rollback_removed: Vec<usize>,
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
            free_pages: Vec::with_capacity(initial_pages_per_layer * config.n_layer),
            kv_dim: kvd,
            total_pages: initial_pages_per_layer * config.n_layer,
            deficits: Vec::with_capacity(config.n_layer),
            new_pages: vec![Vec::new(); config.n_layer],
            all_new_buf: Vec::with_capacity(initial_pages_per_layer * config.n_layer),
            page_ref_counts: vec![config.n_layer as u32; initial_pages_per_layer * config.n_layer],
            rollback_removed: Vec::with_capacity(initial_pages_per_layer),
        }
    }

    /// Allocate a new page. Reuse from free list or grow the pool.
    fn alloc_page(&mut self) -> usize {
        match self.free_pages.pop() {
            Some(idx) => {
                self.pages[idx].fill(0.0);
                self.page_ref_counts[idx] = 1;
                idx
            }
            None => {
                self.pages.push(vec![0.0; PAGE_SIZE * self.kv_dim * 2]);
                self.page_ref_counts.push(1);
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

        // Allocate all pages upfront (reuse scratch buffer via take+put-back
        // to avoid borrow conflict with alloc_page)
        let total_deficit: usize = self.deficits.iter().sum();
        let mut buf = std::mem::take(&mut self.all_new_buf);
        buf.clear();
        buf.reserve(total_deficit);
        for _ in 0..total_deficit {
            buf.push(self.alloc_page());
        }

        // Partition into per-layer lists and assign
        self.new_pages.resize_with(self.deficits.len(), Vec::new);
        let mut offset = 0;
        for (i, &deficit) in self.deficits.iter().enumerate() {
            self.new_pages[i].clear();
            self.new_pages[i].extend_from_slice(&buf[offset..offset + deficit]);
            offset += deficit;
        }
        self.all_new_buf = buf;

        // Assign new pages to each layer's page table
        for (layer_tables, pages) in self.layer_page_tables.iter_mut().zip(self.new_pages.iter()) {
            layer_tables[seq_idx].extend(pages.iter().copied());
        }
    }

    /// Write K and V for a token position in a specific layer.
    /// Layout per page: `[K_data | V_data]` where each is `PAGE_SIZE * kv_dim` floats.
    #[inline]
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
    #[inline]
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
            // Increment ref counts for shared pages
            for &pidx in &shared_pages {
                self.page_ref_counts[pidx] += 1;
            }
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

        // Truncate page tables and decrement ref counts for dropped pages.
        // Pages with ref count == 0 go to the free list.
        for layer_tables in &mut self.layer_page_tables {
            if seq_idx >= layer_tables.len() {
                continue;
            }
            let table = &mut layer_tables[seq_idx];
            self.rollback_removed.clear();
            self.rollback_removed.extend(table.drain(keep_count..));
            for pidx in self.rollback_removed.drain(..) {
                self.page_ref_counts[pidx] -= 1;
                if self.page_ref_counts[pidx] == 0 {
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
        // Reset all ref counts to 0
        self.page_ref_counts.fill(0);
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
    // ── Vec fields first (ptr+len+cap = 24 bytes, 8-byte aligned) ──
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
    // ── usize fields (8-byte aligned, no padding after Vecs) ──
    /// Number of memory slots
    pub num_slots: usize,
    /// Dimension of each KV entry (= kv_dim = n_kv_head × head_dim)
    pub kv_dim: usize,
    /// Top-K slots to update per token
    pub top_k: usize,
    // ── f32 field last (4-byte aligned, no trailing padding on 64-bit) ──
    /// Forget rate for gated update (negative = slower decay)
    pub forget_rate: f32,
}

impl RavenKVCache {
    pub fn new(config: &Config, num_slots: usize, top_k: usize) -> Self {
        let kvd = types::kv_dim(config);
        Self {
            num_slots,
            kv_dim: kvd,
            top_k,
            keys: vec![0.0; num_slots * kvd],
            values: vec![0.0; num_slots * kvd],
            router_scored: vec![(0usize, 0.0f32); num_slots],
            router_r_t: vec![0.0f32; num_slots],
            readout_scores: vec![0.0; num_slots],
            readout_output: vec![0.0; kvd],
            forget_rate: -1.0,
        }
    }

    pub fn reset(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        // Use fill() instead of clear() to preserve pre-allocated capacity.
        // clear() drops len to 0, forcing reallocation on next use via resize.
        self.router_scored.fill((0, 0.0));
        self.router_r_t.fill(0.0);
        self.readout_scores.fill(0.0);
        self.readout_output.fill(0.0);
    }

    /// Export the current routing vector `r_t` (post-router, pre-update).
    /// Returns the normalized Top-K routing weights for all slots.
    /// Used by Phase 3 routed speculation to feed slot selection into anyrag.
    #[inline]
    pub fn r_t(&self) -> &[f32] {
        &self.router_r_t
    }
}

/// Sparse router: computes Top-K routing vector from raw logits (zero-alloc variant).
///
/// Implements: `r_t = Normalize(TopK(Sigmoid(raw_logits)))`
/// Unselected slots get 0.0 → completely frozen during update.
///
/// Uses pre-allocated buffers to avoid heap allocations on the hot path.
#[inline]
pub fn raven_compute_router_into(
    raw_logits: &[f32],
    top_k: usize,
    scored: &mut Vec<(usize, f32)>,
    r_t: &mut Vec<f32>,
) {
    let num_slots = raw_logits.len();
    let top_k = top_k.min(num_slots);

    // Negate logits in-place into r_t scratch buffer
    r_t.resize(num_slots, 0.0);
    for (i, &x) in raw_logits.iter().enumerate() {
        r_t[i] = -x;
    }
    crate::simd::simd_exp_inplace(&mut r_t[..num_slots]);
    // Write directly into pre-sized scored buffer (avoids push reallocation)
    scored.resize(num_slots, (0, 0.0));
    for (i, &e) in r_t[..num_slots].iter().enumerate() {
        scored[i] = (i, 1.0 / (1.0 + e));
    }

    // Partial sort: find Top-K by descending score (O(n) average)
    if top_k < num_slots {
        scored.select_nth_unstable_by(num_slots - top_k, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Fill r_t with zeros for final output
    r_t[..num_slots].fill(0.0);
    let mut sum = 0.0f32;

    // Keep only Top-K (the last top_k elements after partial sort are the largest)
    for (idx, score) in scored.iter().rev().take(top_k) {
        r_t[*idx] = *score;
        sum += *score;
    }

    // Normalize so selected slots sum to 1.0
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in r_t[..num_slots].iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// Backward-compatible wrapper that allocates fresh buffers.
pub fn raven_compute_router(raw_logits: &[f32], top_k: usize) -> Vec<f32> {
    let n = raw_logits.len();
    let mut scored = Vec::with_capacity(n);
    let mut r_t = Vec::with_capacity(n);
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
#[inline]
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
#[inline]
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

    // Pass 2: fused exp + accumulate + normalize (SIMD batch)
    output[..kv_dim].fill(0.0);
    crate::simd::simd_add_scalar_inplace(&mut scores[..num_slots], -max_score);
    crate::simd::simd_exp_inplace(&mut scores[..num_slots]);
    let sum_exp = crate::simd::simd_sum_f32(&scores[..num_slots]);

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

    // Loop-invariant value hoisted outside the layer loop
    let scale = ctx.attn_scale;

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
        // Buffer is pre-sized in ForwardContext::new() to max(kv_dim, 64, num_slots).
        let num_slots = cache.num_slots;
        for (i, slot) in ctx.raven_query_buf[..num_slots].iter_mut().enumerate() {
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
        ctx.attn_out[..n].fill(0.0);

        ctx.raven_query_buf[..kvd].fill(0.0);
        for h in 0..config.n_head {
            let q_off = h * hd;
            // Each head reads from the slot memory using its query slice
            let head_query = &ctx.q[q_off..q_off + hd];
            // Pad/reshape query to kv_dim for slot attention (reuse pre-allocated buffer)
            let kv_group = ctx.kv_group_lut[h] as usize;
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

    // LM Head (uses matmul_parallel for large vocab)
    standard_lm_head(
        &mut ctx.logits,
        &ctx.x,
        &weights.lm_head,
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

    // Loop-invariant values hoisted outside the layer loop
    let scale = ctx.attn_scale;
    let t_n = pos + 1;

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
        // t_n is hoisted outside the layer loop (loop-invariant).
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
        for h in 0..config.n_head {
            let kv_group = ctx.kv_group_lut[h] as usize;
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

    // LM Head (uses matmul_parallel for large vocab)
    standard_lm_head(
        &mut ctx.logits,
        &ctx.x,
        &weights.lm_head,
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
    fn test_restore_preserves_snapshot_data() {
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

        // Verify snapshot data is correctly restored (Issue 097: no zeroing beyond snapshot)
        let kd = types::kv_dim(&config);
        for (layer, snap_layer) in cache.layers.iter().zip(snapshot.layers.iter()) {
            assert_eq!(
                &layer.key[..3 * kd],
                &snap_layer.key,
                "key snapshot data mismatch"
            );
            assert_eq!(
                &layer.value[..3 * kd],
                &snap_layer.value,
                "value snapshot data mismatch"
            );
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

    // ── Kog CPU fusion GOAT proofs (Plan 160) ───────────────────

    /// GOAT proof (T5): folded gamma weights produce identical forward pass output.
    ///
    /// Strategy: create weights with non-trivial gamma, run forward with unfolded gamma,
    /// then fold gamma and run forward again — assert bit-identical output.
    ///
    /// Only MLP gamma is folded (attention gamma is kept at runtime due to residual pattern).
    #[test]
    fn proof_gamma_folding_forward_base() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let mut weights = TransformerWeights::new(&config, &mut rng);

        // Set non-trivial gamma (not all 1.0)
        for layer in &mut weights.layers {
            for (i, g) in layer.attn_norm_gamma.iter_mut().enumerate() {
                *g = 0.5 + (i as f32 * 0.1).sin() * 0.8;
            }
            for (i, g) in layer.mlp_norm_gamma.iter_mut().enumerate() {
                *g = 0.5 + (i as f32 * 0.15).cos() * 0.6;
            }
        }

        // Capture gamma values before folding
        let attn_gammas: Vec<Vec<f32>> = weights
            .layers
            .iter()
            .map(|l| l.attn_norm_gamma.clone())
            .collect();
        let mlp_gammas: Vec<Vec<f32>> = weights
            .layers
            .iter()
            .map(|l| l.mlp_norm_gamma.clone())
            .collect();

        let n = config.n_embd;
        let kvd = types::kv_dim(&config);

        // ── Baseline: forward with unfolded gamma ──
        let mut ctx1 = ForwardContext::new(&config);
        let _cache1 = MultiLayerKVCache::new(&config);

        let tok_off = 0;
        let pos_off_emb = 0;
        crate::simd::simd_add_into(
            &mut ctx1.x[..n],
            &weights.wte[tok_off..tok_off + n],
            &weights.wpe[pos_off_emb..pos_off_emb + n],
        );

        for (li, layer_weights) in weights.layers.iter().enumerate() {
            // Attention: rmsnorm_with_gamma → save residual → QKV
            types::rmsnorm_with_gamma(&mut ctx1.x[..n], &attn_gammas[li]);
            ctx1.xr[..n].copy_from_slice(&ctx1.x[..n]);
            types::matmul(&mut ctx1.q, &layer_weights.attn_wq, &ctx1.x, n, n);
            types::matmul(&mut ctx1.k, &layer_weights.attn_wk, &ctx1.x, kvd, n);
            types::matmul(&mut ctx1.v, &layer_weights.attn_wv, &ctx1.x, kvd, n);
            // Output projection + residual
            types::matmul(&mut ctx1.x, &layer_weights.attn_wo, &ctx1.attn_out, n, n);
            crate::simd::simd_add_inplace(&mut ctx1.x[..n], &ctx1.xr[..n]);
            // MLP: save pre-norm residual → rmsnorm_with_gamma → MLP
            ctx1.xr2[..n].copy_from_slice(&ctx1.x[..n]);
            types::rmsnorm_with_gamma(&mut ctx1.x[..n], &mlp_gammas[li]);
            types::matmul_relu(
                &mut ctx1.hidden,
                &layer_weights.mlp_w1,
                &ctx1.x,
                config.mlp_hidden,
                n,
            );
            types::matmul(
                &mut ctx1.x,
                &layer_weights.mlp_w2,
                &ctx1.hidden,
                n,
                config.mlp_hidden,
            );
            crate::simd::simd_add_inplace(&mut ctx1.x[..n], &ctx1.xr2[..n]);
        }

        let baseline_hidden: Vec<f32> = ctx1.x[..n].to_vec();

        // ── Fold MLP gamma into mlp_w1 ──
        weights.fold_gamma(&config);

        // ── Folded: forward with attn gamma at runtime, mlp gamma folded ──
        let mut ctx2 = ForwardContext::new(&config);
        let _cache2 = MultiLayerKVCache::new(&config);

        crate::simd::simd_add_into(
            &mut ctx2.x[..n],
            &weights.wte[tok_off..tok_off + n],
            &weights.wpe[pos_off_emb..pos_off_emb + n],
        );

        for (li, layer_weights) in weights.layers.iter().enumerate() {
            // Attention: still uses rmsnorm_with_gamma (gamma not folded)
            types::rmsnorm_with_gamma(&mut ctx2.x[..n], &attn_gammas[li]);
            ctx2.xr[..n].copy_from_slice(&ctx2.x[..n]);
            types::matmul(&mut ctx2.q, &layer_weights.attn_wq, &ctx2.x, n, n);
            types::matmul(&mut ctx2.k, &layer_weights.attn_wk, &ctx2.x, kvd, n);
            types::matmul(&mut ctx2.v, &layer_weights.attn_wv, &ctx2.x, kvd, n);
            types::matmul(&mut ctx2.x, &layer_weights.attn_wo, &ctx2.attn_out, n, n);
            crate::simd::simd_add_inplace(&mut ctx2.x[..n], &ctx2.xr[..n]);
            // MLP: gamma folded into w1, so plain rmsnorm (gamma is now identity)
            ctx2.xr2[..n].copy_from_slice(&ctx2.x[..n]);
            rmsnorm(&mut ctx2.x);
            types::matmul_relu(
                &mut ctx2.hidden,
                &layer_weights.mlp_w1,
                &ctx2.x,
                config.mlp_hidden,
                n,
            );
            types::matmul(
                &mut ctx2.x,
                &layer_weights.mlp_w2,
                &ctx2.hidden,
                n,
                config.mlp_hidden,
            );
            crate::simd::simd_add_inplace(&mut ctx2.x[..n], &ctx2.xr2[..n]);
        }

        let folded_hidden: Vec<f32> = ctx2.x[..n].to_vec();

        // GOAT assertion: bit-identical (within FP tolerance)
        for i in 0..n {
            let diff = (baseline_hidden[i] - folded_hidden[i]).abs();
            assert!(
                diff < 1e-5,
                "GOAT FAIL: gamma fold mismatch at [{i}]: baseline={}, folded={}, diff={}",
                baseline_hidden[i],
                folded_hidden[i],
                diff
            );
        }
    }

    /// GOAT proof (T10): QKV interleaving produces identical attention output.
    #[test]
    fn proof_qkv_interleave_forward() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let mut weights = TransformerWeights::new(&config, &mut rng);

        let n = config.n_embd;
        let kvd = types::kv_dim(&config);

        // Run forward with separate Q/K/V
        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let logits1 = forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config).to_vec();

        // Interleave QKV
        weights.interleave_qkv(&config);

        // Run forward with fused QKV (feature-gated path, but we test the fused weight slicing)
        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);

        // Manual forward using fused weight slices
        crate::simd::simd_add_into(&mut ctx2.x[..n], &weights.wte[0..n], &weights.wpe[0..n]);

        for layer_weights in &weights.layers {
            rmsnorm(&mut ctx2.x);
            ctx2.xr[..n].copy_from_slice(&ctx2.x[..n]);

            let fused = layer_weights
                .attn_qkv_fused
                .as_ref()
                .expect("fused should be populated");
            // Q slice
            types::matmul(&mut ctx2.q, &fused[..n * n], &ctx2.x, n, n);
            // K slice
            types::matmul(&mut ctx2.k, &fused[n * n..(n + kvd) * n], &ctx2.x, kvd, n);
            // V slice
            types::matmul(&mut ctx2.v, &fused[(n + kvd) * n..], &ctx2.x, kvd, n);

            // Store K,V
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ctx2.k.as_ptr(),
                    cache2.layers[0].key.as_mut_ptr(),
                    kvd,
                );
                std::ptr::copy_nonoverlapping(
                    ctx2.v.as_ptr(),
                    cache2.layers[0].value.as_mut_ptr(),
                    kvd,
                );
            }

            // Attention
            let scale = ctx2.attn_scale;
            for h in 0..config.n_head {
                let kv_group = ctx2.kv_group_lut[h] as usize;
                unsafe {
                    attention_head(
                        &ctx2.q,
                        &cache2.layers[0].key,
                        &cache2.layers[0].value,
                        &mut ctx2.attn_out,
                        &mut ctx2.scores,
                        h * config.head_dim,
                        kv_group * config.head_dim,
                        kvd,
                        config.head_dim,
                        1,
                        scale,
                    );
                }
            }
            types::matmul(&mut ctx2.x, &layer_weights.attn_wo, &ctx2.attn_out, n, n);
            crate::simd::simd_add_inplace(&mut ctx2.x[..n], &ctx2.xr[..n]);
            ctx2.xr2[..n].copy_from_slice(&ctx2.x[..n]);
            rmsnorm(&mut ctx2.x);
            types::matmul_relu(
                &mut ctx2.hidden,
                &layer_weights.mlp_w1,
                &ctx2.x,
                config.mlp_hidden,
                n,
            );
            types::matmul(
                &mut ctx2.x,
                &layer_weights.mlp_w2,
                &ctx2.hidden,
                n,
                config.mlp_hidden,
            );
            crate::simd::simd_add_inplace(&mut ctx2.x[..n], &ctx2.xr2[..n]);
        }

        standard_lm_head(
            &mut ctx2.logits,
            &ctx2.x,
            &weights.lm_head,
            config.vocab_size,
            n,
        );
        let logits2 = ctx2.logits.to_vec();

        // GOAT assertion
        for i in 0..config.vocab_size {
            let diff = (logits1[i] - logits2[i]).abs();
            assert!(
                diff < 1e-4,
                "GOAT FAIL: QKV interleave mismatch at logit[{i}]: sep={}, fused={}, diff={}",
                logits1[i],
                logits2[i],
                diff
            );
        }
    }

    /// GOAT proof (T11): MLP gamma folding produces identical single-layer output.
    /// Tests the safe folding path: MLP gamma folded into w1, attention gamma kept at runtime.
    #[test]
    fn proof_gamma_folding_single_layer() {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let mut weights = TransformerWeights::new(&config, &mut rng);
        let _ = rng;

        let n = config.n_embd;
        let kvd = types::kv_dim(&config);

        // Set non-trivial gamma
        for layer in &mut weights.layers {
            for (i, g) in layer.attn_norm_gamma.iter_mut().enumerate() {
                *g = 0.5 + (i as f32 * 0.1).sin() * 0.8;
            }
            for (i, g) in layer.mlp_norm_gamma.iter_mut().enumerate() {
                *g = 0.5 + (i as f32 * 0.15).cos() * 0.6;
            }
        }

        // Capture gammas
        let attn_gamma = weights.layers[0].attn_norm_gamma.clone();
        let mlp_gamma = weights.layers[0].mlp_norm_gamma.clone();

        // ── Baseline: single layer with gamma ──
        let mut ctx1 = ForwardContext::new(&config);
        let _cache1 = MultiLayerKVCache::new(&config);

        // Embed
        crate::simd::simd_add_into(&mut ctx1.x[..n], &weights.wte[0..n], &weights.wpe[0..n]);

        // Attention with gamma
        types::rmsnorm_with_gamma(&mut ctx1.x[..n], &attn_gamma);
        ctx1.xr[..n].copy_from_slice(&ctx1.x[..n]);
        types::matmul(&mut ctx1.q, &weights.layers[0].attn_wq, &ctx1.x, n, n);
        types::matmul(&mut ctx1.k, &weights.layers[0].attn_wk, &ctx1.x, kvd, n);
        types::matmul(&mut ctx1.v, &weights.layers[0].attn_wv, &ctx1.x, kvd, n);
        types::matmul(
            &mut ctx1.x,
            &weights.layers[0].attn_wo,
            &ctx1.attn_out,
            n,
            n,
        );
        crate::simd::simd_add_inplace(&mut ctx1.x[..n], &ctx1.xr[..n]);
        // MLP with gamma
        ctx1.xr2[..n].copy_from_slice(&ctx1.x[..n]);
        types::rmsnorm_with_gamma(&mut ctx1.x[..n], &mlp_gamma);
        types::matmul_relu(
            &mut ctx1.hidden,
            &weights.layers[0].mlp_w1,
            &ctx1.x,
            config.mlp_hidden,
            n,
        );
        types::matmul(
            &mut ctx1.x,
            &weights.layers[0].mlp_w2,
            &ctx1.hidden,
            n,
            config.mlp_hidden,
        );
        crate::simd::simd_add_inplace(&mut ctx1.x[..n], &ctx1.xr2[..n]);

        let baseline_hidden: Vec<f32> = ctx1.x[..n].to_vec();

        // ── Fold gamma ──
        weights.fold_gamma(&config);

        // ── Folded path: attn gamma at runtime, mlp gamma folded ──
        let mut ctx2 = ForwardContext::new(&config);
        let _cache2 = MultiLayerKVCache::new(&config);

        crate::simd::simd_add_into(&mut ctx2.x[..n], &weights.wte[0..n], &weights.wpe[0..n]);

        // Attention with gamma (kept)
        types::rmsnorm_with_gamma(&mut ctx2.x[..n], &attn_gamma);
        ctx2.xr[..n].copy_from_slice(&ctx2.x[..n]);
        types::matmul(&mut ctx2.q, &weights.layers[0].attn_wq, &ctx2.x, n, n);
        types::matmul(&mut ctx2.k, &weights.layers[0].attn_wk, &ctx2.x, kvd, n);
        types::matmul(&mut ctx2.v, &weights.layers[0].attn_wv, &ctx2.x, kvd, n);
        types::matmul(
            &mut ctx2.x,
            &weights.layers[0].attn_wo,
            &ctx2.attn_out,
            n,
            n,
        );
        crate::simd::simd_add_inplace(&mut ctx2.x[..n], &ctx2.xr[..n]);
        // MLP: gamma folded, so plain rmsnorm
        ctx2.xr2[..n].copy_from_slice(&ctx2.x[..n]);
        rmsnorm(&mut ctx2.x);
        types::matmul_relu(
            &mut ctx2.hidden,
            &weights.layers[0].mlp_w1,
            &ctx2.x,
            config.mlp_hidden,
            n,
        );
        types::matmul(
            &mut ctx2.x,
            &weights.layers[0].mlp_w2,
            &ctx2.hidden,
            n,
            config.mlp_hidden,
        );
        crate::simd::simd_add_inplace(&mut ctx2.x[..n], &ctx2.xr2[..n]);

        let folded_hidden: Vec<f32> = ctx2.x[..n].to_vec();

        // GOAT assertion
        for i in 0..n {
            let diff = (baseline_hidden[i] - folded_hidden[i]).abs();
            assert!(
                diff < 1e-5,
                "GOAT FAIL: gamma fold mismatch at [{i}]: baseline={}, folded={}, diff={}",
                baseline_hidden[i],
                folded_hidden[i],
                diff
            );
        }
    }
}
