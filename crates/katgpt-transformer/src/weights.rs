//! Per-layer and global transformer weight structures.
//!
//! Pure data types — no forward logic. The forward kernels live in the
//! `katgpt-rs` root crate because they compose cognitive primitives
//! (`crate::hla`, `crate::sleep`, `crate::tf_loop`, etc.) that do not exist
//! in this substrate crate.

use katgpt_core::types::{self, Config, Rng};

/// Per-layer transformer weights.
/// Each layer has its own attention and MLP parameters.
pub struct LayerWeights {
    pub attn_wq: Vec<f32>, // [n_embd, n_embd]
    pub attn_wk: Vec<f32>, // [kv_dim, n_embd] where kv_dim = n_kv_head * head_dim
    pub attn_wv: Vec<f32>, // [kv_dim, n_embd]
    pub attn_wo: Vec<f32>, // [n_embd, n_embd]
    pub mlp_w1: Vec<f32>,  // [mlp_hidden, n_embd]
    pub mlp_w2: Vec<f32>,  // [n_embd, mlp_hidden]
    // Kog CPU fusion (Plan 160): RMSNorm gamma vectors — only present when the
    // consumer enables `kog_cpu_fusion`. Consumers that don't use Kog fusion
    // (e.g. riir-engine) get the compact 6-field struct and avoid ~2×n
    // floats/layer of dead weight.
    #[cfg(feature = "kog_cpu_fusion")]
    pub attn_norm_gamma: Vec<f32>, // [n_embd] pre-attention RMSNorm gamma (identity=1.0)
    #[cfg(feature = "kog_cpu_fusion")]
    pub mlp_norm_gamma: Vec<f32>,  // [n_embd] pre-MLP RMSNorm gamma (identity=1.0)
    // Kog CPU fusion (Plan 160): fused QKV weight storage
    #[cfg(feature = "kog_cpu_fusion")]
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
                #[cfg(feature = "kog_cpu_fusion")]
                attn_norm_gamma: vec![1.0f32; n],
                #[cfg(feature = "kog_cpu_fusion")]
                mlp_norm_gamma: vec![1.0f32; n],
                #[cfg(feature = "kog_cpu_fusion")]
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
    #[cfg(feature = "kog_cpu_fusion")]
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
    #[cfg(feature = "kog_cpu_fusion")]
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
