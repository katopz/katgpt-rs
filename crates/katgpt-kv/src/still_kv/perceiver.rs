//! StillPerceiver: Cross-attention based KV cache compaction.
//!
//! A lightweight Perceiver-style architecture that uses learned latent queries
//! to cross-attend into the full KV cache, producing a compact representation.
//!
//! T11: Multi-head scaled dot-product cross-attention (Q←latents, K=V←kv_cache).
//! T12: Self-attention refinement with RMSNorm + residual.
//! T13: Identity-init output projection heads (project_keys / project_values).
//! T14: Full forward pipeline wired as num_blocks × (cross + self).

/// Configuration for the StillPerceiver compactor.
#[derive(Debug, Clone)]
pub struct StillPerceiverConfig {
    /// Latent dimension for the perceiver bottleneck.
    pub latent_dim: usize,
    /// Target compact sequence length (number of latent tokens).
    pub compact_length: usize,
    /// Number of perceiver blocks (cross-attn + self-attn pairs).
    pub num_blocks: usize,
    /// Number of heads for cross-attention.
    pub cross_attn_heads: usize,
    /// Number of heads for self-attention within latents.
    pub self_attn_heads: usize,
    /// KV dimension (must equal latent_dim for direct cross-attention).
    pub kv_dim: usize,
    /// Epsilon for RMSNorm stability.
    pub rms_norm_eps: f32,
}

impl Default for StillPerceiverConfig {
    fn default() -> Self {
        Self {
            latent_dim: 64,
            compact_length: 128,
            num_blocks: 2,
            cross_attn_heads: 4,
            self_attn_heads: 4,
            kv_dim: 64,
            rms_norm_eps: 1e-5,
        }
    }
}

impl StillPerceiverConfig {
    /// Create a new config with the given latent dimension and compact length.
    pub fn new(latent_dim: usize, compact_length: usize) -> Self {
        Self {
            latent_dim,
            compact_length,
            kv_dim: latent_dim,
            ..Default::default()
        }
    }

    /// Create a new config with separate kv_dim.
    pub fn with_kv_dim(latent_dim: usize, compact_length: usize, kv_dim: usize) -> Self {
        Self {
            latent_dim,
            compact_length,
            kv_dim,
            ..Default::default()
        }
    }

    /// Dimension per cross-attention head.
    #[inline]
    fn cross_head_dim(&self) -> usize {
        self.latent_dim / self.cross_attn_heads
    }

    /// Dimension per self-attention head.
    #[inline]
    fn self_head_dim(&self) -> usize {
        self.latent_dim / self.self_attn_heads
    }
}

/// Perceiver-based KV cache compactor.
///
/// Uses cross-attention from learned latent queries into the KV cache
/// to produce a compact representation. Multiple blocks of
/// cross-attention + self-attention refine the compact output.
#[derive(Debug, Clone)]
pub struct StillPerceiver {
    /// Configuration.
    pub config: StillPerceiverConfig,
}

impl StillPerceiver {
    /// Create a new perceiver with the given configuration.
    pub fn new(config: StillPerceiverConfig) -> Self {
        assert!(
            config.latent_dim.is_multiple_of(config.cross_attn_heads),
            "latent_dim ({}) must be divisible by cross_attn_heads ({})",
            config.latent_dim,
            config.cross_attn_heads,
        );
        assert!(
            config.latent_dim.is_multiple_of(config.self_attn_heads),
            "latent_dim ({}) must be divisible by self_attn_heads ({})",
            config.latent_dim,
            config.self_attn_heads,
        );
        Self { config }
    }

    // -----------------------------------------------------------------------
    // T11: Cross-attention
    // -----------------------------------------------------------------------

    /// Cross-attend latent queries into the KV cache (multi-head).
    ///
    /// # Arguments
    /// * `latent_queries` - Shape `[compact_length * latent_dim]` (flat, row-major)
    /// * `kv_cache` - Shape `[seq_len * kv_dim]` (flat, row-major)
    ///
    /// # Returns
    /// Updated latent queries after cross-attention + residual, shape `[compact_length * latent_dim]`.
    ///
    /// # Panics
    /// If `kv_dim != latent_dim` (dimension mismatch requiring projection).
    pub fn cross_attention(&self, latent_queries: &[f32], kv_cache: &[f32]) -> Vec<f32> {
        let cfg = &self.config;
        let d = cfg.latent_dim;
        let t = latent_queries.len() / d; // compact_length
        let big_t = kv_cache.len() / d; // seq_len
        let num_heads = cfg.cross_attn_heads;
        let head_dim = cfg.cross_head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        assert_eq!(
            cfg.kv_dim, d,
            "kv_dim must equal latent_dim for direct cross-attention"
        );
        assert_eq!(latent_queries.len(), t * d, "latent_queries shape mismatch");
        assert_eq!(kv_cache.len(), big_t * d, "kv_cache shape mismatch");

        let mut output = vec![0.0f32; t * d];

        // Process each head independently.
        for h in 0..num_heads {
            let h_off = h * head_dim;

            // Attention scores for this head: [t, big_t]
            let mut scores = vec![0.0f32; t * big_t];

            // Compute scores = Q @ K^T * scale
            for qi in 0..t {
                let q_row = qi * d + h_off;
                let score_row = qi * big_t;
                for ki in 0..big_t {
                    let k_row = ki * d + h_off;
                    let dot = dot_chunk4(
                        &latent_queries[q_row..q_row + head_dim],
                        &kv_cache[k_row..k_row + head_dim],
                    );
                    scores[score_row + ki] = dot * scale;
                }
            }

            // Softmax per row (stable: subtract max, then exp, then normalize)
            softmax_rows(&mut scores, t, big_t);

            // Weighted sum: output[i, h_off..h_off+head_dim] = sum_j attn[i,j] * V[j, h_off..h_off+head_dim]
            for qi in 0..t {
                let score_row = qi * big_t;
                let out_base = qi * d + h_off;
                for ki in 0..big_t {
                    let w = scores[score_row + ki];
                    if w < 1e-8 {
                        continue; // skip near-zero weights
                    }
                    let v_base = ki * d + h_off;
                    accumulate_chunk4(
                        &mut output[out_base..out_base + head_dim],
                        &kv_cache[v_base..v_base + head_dim],
                        w,
                    );
                }
            }
        }

        // Residual connection
        for i in 0..t * d {
            output[i] += latent_queries[i];
        }

        output
    }

    /// Cross-attend latent queries into the KV cache, returning both output and attention weights.
    ///
    /// Same as `cross_attention` but also returns the attention weight matrix.
    /// The weights are needed for β-D (VortexFlow) bias computation.
    ///
    /// # Returns
    /// `(output, attention_weights)` where:
    /// - `output` shape `[compact_length * latent_dim]` (same as `cross_attention`)
    /// - `attention_weights` shape `[compact_length * seq_len]` (row-major)
    pub fn cross_attention_with_weights(
        &self,
        latent_queries: &[f32],
        kv_cache: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let cfg = &self.config;
        let d = cfg.latent_dim;
        let t = latent_queries.len() / d;
        let big_t = kv_cache.len() / d;
        let num_heads = cfg.cross_attn_heads;
        let head_dim = cfg.cross_head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        assert_eq!(
            cfg.kv_dim, d,
            "kv_dim must equal latent_dim for direct cross-attention"
        );
        assert_eq!(latent_queries.len(), t * d, "latent_queries shape mismatch");
        assert_eq!(kv_cache.len(), big_t * d, "kv_cache shape mismatch");

        let mut output = vec![0.0f32; t * d];
        // Aggregate attention weights across heads (average)
        let mut agg_weights = vec![0.0f32; t * big_t];

        for h in 0..num_heads {
            let h_off = h * head_dim;

            let mut scores = vec![0.0f32; t * big_t];

            for qi in 0..t {
                let q_row = qi * d + h_off;
                let score_row = qi * big_t;
                for ki in 0..big_t {
                    let k_row = ki * d + h_off;
                    let dot = dot_chunk4(
                        &latent_queries[q_row..q_row + head_dim],
                        &kv_cache[k_row..k_row + head_dim],
                    );
                    scores[score_row + ki] = dot * scale;
                }
            }

            softmax_rows(&mut scores, t, big_t);

            // Accumulate head-averaged attention weights
            let head_weight = 1.0 / num_heads as f32;
            for i in 0..t * big_t {
                agg_weights[i] += scores[i] * head_weight;
            }

            for qi in 0..t {
                let score_row = qi * big_t;
                let out_base = qi * d + h_off;
                for ki in 0..big_t {
                    let w = scores[score_row + ki];
                    if w < 1e-8 {
                        continue;
                    }
                    let v_base = ki * d + h_off;
                    accumulate_chunk4(
                        &mut output[out_base..out_base + head_dim],
                        &kv_cache[v_base..v_base + head_dim],
                        w,
                    );
                }
            }
        }

        // Residual connection
        for i in 0..t * d {
            output[i] += latent_queries[i];
        }

        (output, agg_weights)
    }

    /// Self-attention among latent tokens to refine representations (multi-head).
    ///
    /// Applies RMSNorm to Q=K=V before attention, then adds residual.
    ///
    /// # Arguments
    /// * `latents` - Shape `[compact_length * latent_dim]` (flat, row-major)
    ///
    /// # Returns
    /// Refined latents after self-attention.
    pub fn self_attention(&self, latents: &[f32]) -> Vec<f32> {
        let cfg = &self.config;
        let d = cfg.latent_dim;
        let t = latents.len() / d;
        let num_heads = cfg.self_attn_heads;
        let head_dim = cfg.self_head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        assert_eq!(latents.len(), t * d, "latents shape mismatch");

        // RMSNorm Q, K, V
        let q_norm = rms_norm(latents, d, cfg.rms_norm_eps);
        let k_norm = rms_norm(latents, d, cfg.rms_norm_eps);
        let v_norm = rms_norm(latents, d, cfg.rms_norm_eps);

        let mut output = vec![0.0f32; t * d];

        // Process each head independently.
        for h in 0..num_heads {
            let h_off = h * head_dim;

            // Attention scores for this head: [t, t]
            let mut scores = vec![0.0f32; t * t];

            // Compute scores = Q @ K^T * scale
            for qi in 0..t {
                let q_row = qi * d + h_off;
                let score_row = qi * t;
                for ki in 0..t {
                    let k_row = ki * d + h_off;
                    let dot = dot_chunk4(
                        &q_norm[q_row..q_row + head_dim],
                        &k_norm[k_row..k_row + head_dim],
                    );
                    scores[score_row + ki] = dot * scale;
                }
            }

            // Softmax per row
            softmax_rows(&mut scores, t, t);

            // Weighted sum
            for qi in 0..t {
                let score_row = qi * t;
                let out_base = qi * d + h_off;
                for ki in 0..t {
                    let w = scores[score_row + ki];
                    if w < 1e-8 {
                        continue;
                    }
                    let v_base = ki * d + h_off;
                    accumulate_chunk4(
                        &mut output[out_base..out_base + head_dim],
                        &v_norm[v_base..v_base + head_dim],
                        w,
                    );
                }
            }
        }

        // Residual connection
        for i in 0..t * d {
            output[i] += latents[i];
        }

        output
    }

    // -----------------------------------------------------------------------
    // T13: Output projection heads (identity init)
    // -----------------------------------------------------------------------

    /// Project latents to compact keys (identity init).
    ///
    /// If `latent_dim == kv_dim`: returns latents as-is.
    /// If `latent_dim < kv_dim`: pads with zeros.
    /// If `latent_dim > kv_dim`: truncates.
    pub fn project_keys(&self, latents: &[f32]) -> Vec<f32> {
        identity_project(latents, self.config.latent_dim, self.config.kv_dim)
    }

    /// Project latents to compact values (identity init).
    ///
    /// Same logic as `project_keys` — identity mapping with pad/truncate.
    pub fn project_values(&self, latents: &[f32]) -> Vec<f32> {
        identity_project(latents, self.config.latent_dim, self.config.kv_dim)
    }

    // -----------------------------------------------------------------------
    // T14: Forward pipeline
    // -----------------------------------------------------------------------

    /// Run the full perceiver forward pass: num_blocks × (cross-attn + self-attn).
    ///
    /// # Arguments
    /// * `kv_cache` - Flat f32 KV cache buffer, shape `[seq_len * kv_dim]`
    /// * `query_bank` - Initial latent queries from query bank, shape `[compact_length * latent_dim]`
    ///
    /// # Returns
    /// Compacted latent representation, shape `[compact_length * latent_dim]`.
    pub fn forward(&self, kv_cache: &[f32], query_bank: &[f32]) -> Vec<f32> {
        let mut latents = query_bank.to_vec();
        for _ in 0..self.config.num_blocks {
            latents = self.cross_attention(&latents, kv_cache);
            latents = self.self_attention(&latents);
        }
        latents
    }

    /// Full forward pass with key/value projection.
    ///
    /// Returns `(compact_keys, compact_values)` both shape `[compact_length * kv_dim]`.
    pub fn forward_projected(&self, kv_cache: &[f32], query_bank: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let latents = self.forward(kv_cache, query_bank);
        let keys = self.project_keys(&latents);
        let values = self.project_values(&latents);
        (keys, values)
    }

    /// Run forward pass and return latents + cross-attention weights.
    ///
    /// The cross-attention weights are needed for β-D (VortexFlow) computation.
    /// Returns `(latents, cross_attn_weights)` where:
    /// - `latents` shape `[compact_length * latent_dim]`
    /// - `cross_attn_weights` shape `[compact_length * seq_len]` (from last block)
    pub fn forward_with_weights(
        &self,
        kv_cache: &[f32],
        query_bank: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut latents = query_bank.to_vec();
        let mut last_cross_weights = Vec::new();

        for _ in 0..self.config.num_blocks {
            let (new_latents, weights) = self.cross_attention_with_weights(&latents, kv_cache);
            latents = new_latents;
            last_cross_weights = weights;
            latents = self.self_attention(&latents);
        }

        (latents, last_cross_weights)
    }
}

// =========================================================================
// Private helpers
// =========================================================================

/// RMSNorm: normalize each row of length `dim`.
///
/// `rms = sqrt(mean(x^2) + eps)`, `output = x / rms`.
fn rms_norm(x: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let n = x.len() / dim;
    let mut out = vec![0.0f32; x.len()];

    for i in 0..n {
        let row_start = i * dim;
        let row = &x[row_start..row_start + dim];

        // Compute sum of squares with chunk-4
        let mut sum_sq = 0.0f32;
        let chunks = dim / 4;
        let remainder = dim % 4;

        for c in 0..chunks {
            let base = c * 4;
            sum_sq += row[base] * row[base]
                + row[base + 1] * row[base + 1]
                + row[base + 2] * row[base + 2]
                + row[base + 3] * row[base + 3];
        }
        for &v in &row[chunks * 4..chunks * 4 + remainder] {
            sum_sq += v * v;
        }

        let mean_sq = sum_sq / dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        let out_row = &mut out[row_start..row_start + dim];
        for c in 0..chunks {
            let base = c * 4;
            out_row[base] = row[base] * inv_rms;
            out_row[base + 1] = row[base + 1] * inv_rms;
            out_row[base + 2] = row[base + 2] * inv_rms;
            out_row[base + 3] = row[base + 3] * inv_rms;
        }
        for j in (chunks * 4)..(chunks * 4 + remainder) {
            out_row[j] = row[j] * inv_rms;
        }
    }

    out
}

/// Dot product with chunk-4 loop for SIMD-friendly accumulation.
#[inline]
fn dot_chunk4(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = 0.0f32;
    for c in 0..chunks {
        let base = c * 4;
        sum += a[base] * b[base]
            + a[base + 1] * b[base + 1]
            + a[base + 2] * b[base + 2]
            + a[base + 3] * b[base + 3];
    }
    for i in (chunks * 4)..(chunks * 4 + remainder) {
        sum += a[i] * b[i];
    }
    sum
}

/// Accumulate weighted values into output with chunk-4 loop.
///
/// `output[i] += weight * values[i]`
#[inline]
fn accumulate_chunk4(output: &mut [f32], values: &[f32], weight: f32) {
    debug_assert_eq!(output.len(), values.len());
    let n = output.len();
    let chunks = n / 4;
    let remainder = n % 4;

    for c in 0..chunks {
        let base = c * 4;
        output[base] += weight * values[base];
        output[base + 1] += weight * values[base + 1];
        output[base + 2] += weight * values[base + 2];
        output[base + 3] += weight * values[base + 3];
    }
    for i in (chunks * 4)..(chunks * 4 + remainder) {
        output[i] += weight * values[i];
    }
}

/// Numerically stable softmax applied row-wise to `[rows, cols]`.
///
/// For each row: subtract max, exp, divide by sum.
/// Uses SIMD-accelerated primitives from `katgpt_core::simd` (NEON on aarch64,
/// AVX2+FMA on x86_64) — replaces the 3 scalar loops with 3 vectorized passes.
fn softmax_rows(data: &mut [f32], rows: usize, cols: usize) {
    use katgpt_core::simd::{
        simd_add_scalar_inplace, simd_exp_sum_inplace, simd_max_f32, simd_scale_inplace,
    };
    for r in 0..rows {
        let row_start = r * cols;
        let row = &mut data[row_start..row_start + cols];

        // Pass 1: find max for numerical stability (SIMD-accelerated).
        let max_val = simd_max_f32(row);

        // Pass 2: subtract max (SIMD-accelerated).
        simd_add_scalar_inplace(row, -max_val);

        // Pass 3: SIMD exp + sum (fused) → SIMD normalize.
        let sum: f32 = simd_exp_sum_inplace(row);
        if sum > 0.0 {
            simd_scale_inplace(row, 1.0 / sum);
        }
    }
}

/// Identity projection with pad/truncate between dimensions.
///
/// `latents`: `[n, latent_dim]` → output: `[n, target_dim]`
fn identity_project(latents: &[f32], latent_dim: usize, target_dim: usize) -> Vec<f32> {
    if latent_dim == target_dim {
        return latents.to_vec();
    }

    let n = latents.len() / latent_dim;
    let mut out = vec![0.0f32; n * target_dim];

    for i in 0..n {
        let src_start = i * latent_dim;
        let dst_start = i * target_dim;
        let copy_len = latent_dim.min(target_dim);
        out[dst_start..dst_start + copy_len]
            .copy_from_slice(&latents[src_start..src_start + copy_len]);
    }

    out
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- T11: Cross-attention tests ---

    #[test]
    fn test_cross_attention_output_shape() {
        let config = StillPerceiverConfig::new(8, 4);
        let perceiver = StillPerceiver::new(config);
        // 4 queries of dim 8, 16 kv tokens of dim 8
        let queries = vec![0.5f32; 4 * 8];
        let kv = vec![1.0f32; 16 * 8];
        let result = perceiver.cross_attention(&queries, &kv);
        assert_eq!(result.len(), 4 * 8, "output shape must match query shape");
    }

    #[test]
    fn test_cross_attention_attends_to_kv() {
        let config = StillPerceiverConfig::new(8, 2);
        let perceiver = StillPerceiver::new(config);
        // Queries: all zeros → scores are all zero → softmax → uniform → output = mean(V) + residual
        let queries = vec![0.0f32; 2 * 8];
        // KV: known non-zero values
        let mut kv = vec![0.0f32; 4 * 8];
        for i in 0..4 {
            for j in 0..8 {
                kv[i * 8 + j] = (i * 8 + j + 1) as f32;
            }
        }

        let result = perceiver.cross_attention(&queries, &kv);

        // With zero queries, all scores are 0 → uniform softmax (1/4 each)
        // output = mean(V) + queries(=0) = mean(V)
        // Check output is not all zeros (i.e., it attended to kv)
        let any_nonzero = result.iter().any(|&v| v.abs() > 1e-6);
        assert!(
            any_nonzero,
            "cross-attention output must be influenced by kv_cache"
        );
    }

    #[test]
    fn test_cross_attention_residual_connection() {
        let config = StillPerceiverConfig::new(4, 1);
        let perceiver = StillPerceiver::new(config);
        // Single query, single kv token
        let queries = vec![2.0f32, 0.0, 0.0, 0.0];
        let kv = vec![0.0f32, 0.0, 0.0, 0.0]; // all-zero KV → attention output is 0
        let result = perceiver.cross_attention(&queries, &kv);
        // With zero KV, weighted sum is 0. Residual adds queries back.
        // Result should equal queries.
        for i in 0..4 {
            assert!(
                (result[i] - queries[i]).abs() < 1e-5,
                "result[{}] = {} but expected {} (residual check)",
                i,
                result[i],
                queries[i],
            );
        }
    }

    // --- T12: Self-attention tests ---

    #[test]
    fn test_self_attention_output_shape() {
        let config = StillPerceiverConfig::new(8, 4);
        let perceiver = StillPerceiver::new(config);
        let latents = vec![0.5f32; 4 * 8];
        let result = perceiver.self_attention(&latents);
        assert_eq!(
            result.len(),
            4 * 8,
            "self-attention output shape must match input"
        );
    }

    #[test]
    fn test_self_attention_not_identity() {
        let config = StillPerceiverConfig::new(8, 4);
        let perceiver = StillPerceiver::new(config);
        let latents = vec![1.0f32; 4 * 8];
        let result = perceiver.self_attention(&latents);
        // Self-attention + residual on uniform input should NOT be identity
        // (RMSNorm normalizes to same magnitude, attention produces weighted sum, residual adds back)
        // Actually with uniform input after RMSNorm all rows are identical → attention is uniform →
        // output = V (which is uniform) + residual. So result != latents due to RMSNorm scaling.
        let differs = result
            .iter()
            .zip(latents.iter())
            .any(|(&r, &l)| (r - l).abs() > 1e-5);
        assert!(
            differs,
            "self-attention should modify input through RMSNorm + attention + residual"
        );
    }

    // --- RMSNorm tests ---

    #[test]
    fn test_rms_norm_unit_vector() {
        // A unit vector should have RMS ≈ 1, so norm output ≈ input
        let x = vec![1.0f32, 0.0, 0.0, 0.0];
        let result = rms_norm(&x, 4, 1e-5);
        // rms = sqrt((1+0+0+0)/4 + eps) = sqrt(0.25 + eps) ≈ 0.5
        // output = x / rms → [1/0.5, 0, 0, 0] = [2.0, 0, 0, 0]
        assert!(
            (result[0] - 2.0).abs() < 0.01,
            "RMSNorm of [1,0,0,0] should be ≈ [2,0,0,0]"
        );
    }

    #[test]
    fn test_rms_norm_zero_vector() {
        let x = vec![0.0f32; 4];
        let result = rms_norm(&x, 4, 1e-5);
        // rms = sqrt(eps) → output = 0 / rms = 0
        for &v in &result {
            assert!(v.abs() < 1e-3, "RMSNorm of zero vector should be ≈ 0");
        }
    }

    #[test]
    fn test_rms_norm_uniform_vector() {
        let x = vec![3.0f32; 8];
        let result = rms_norm(&x, 8, 1e-5);
        // All elements should be equal (uniform input → uniform output)
        let first = result[0];
        for &v in &result {
            assert!(
                (v - first).abs() < 1e-5,
                "RMSNorm of uniform vector should be uniform"
            );
        }
        // And non-zero
        assert!(
            first.abs() > 0.1,
            "RMSNorm of non-zero uniform should be non-trivial"
        );
    }

    // --- T13: Projection tests ---

    #[test]
    fn test_project_keys_identity() {
        let config = StillPerceiverConfig::new(8, 4);
        let perceiver = StillPerceiver::new(config);
        let latents = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let keys = perceiver.project_keys(&latents);
        // latent_dim == kv_dim → identity
        assert_eq!(
            keys, latents,
            "identity projection when latent_dim == kv_dim"
        );
    }

    #[test]
    fn test_project_pad() {
        let mut config = StillPerceiverConfig::new(4, 2);
        config.kv_dim = 6;
        let perceiver = StillPerceiver::new(config);
        let latents = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = perceiver.project_keys(&latents);
        assert_eq!(result.len(), 6, "padded output should be kv_dim");
        assert_eq!(result[0], 1.0);
        assert_eq!(result[3], 4.0);
        assert_eq!(result[4], 0.0, "padding should be zero");
        assert_eq!(result[5], 0.0, "padding should be zero");
    }

    #[test]
    fn test_project_truncate() {
        let mut config = StillPerceiverConfig::new(8, 2);
        config.kv_dim = 4;
        let perceiver = StillPerceiver::new(config);
        let latents = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = perceiver.project_keys(&latents);
        assert_eq!(result.len(), 4, "truncated output should be kv_dim");
        assert_eq!(result[0], 1.0);
        assert_eq!(result[3], 4.0);
    }

    // --- T14: Forward pipeline tests ---

    #[test]
    fn test_forward_multi_block() {
        let config_1block = StillPerceiverConfig {
            latent_dim: 8,
            compact_length: 2,
            num_blocks: 1,
            cross_attn_heads: 2,
            self_attn_heads: 2,
            kv_dim: 8,
            rms_norm_eps: 1e-5,
        };
        let config_2blocks = StillPerceiverConfig {
            num_blocks: 2,
            ..config_1block.clone()
        };
        let perceiver_1 = StillPerceiver::new(config_1block);
        let perceiver_2 = StillPerceiver::new(config_2blocks);

        let kv: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let queries: Vec<f32> = (0..16).map(|i| i as f32 * 0.5 - 2.0).collect();

        let result_1 = perceiver_1.forward(&kv, &queries);
        let result_2 = perceiver_2.forward(&kv, &queries);

        // 2 blocks should produce different output than 1 block
        let differs = result_1
            .iter()
            .zip(result_2.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(
            differs,
            "2 blocks should produce different output than 1 block"
        );
    }

    #[test]
    fn test_forward_projected() {
        let config = StillPerceiverConfig {
            latent_dim: 8,
            compact_length: 2,
            num_blocks: 2,
            cross_attn_heads: 2,
            self_attn_heads: 2,
            kv_dim: 8,
            rms_norm_eps: 1e-5,
        };
        let perceiver = StillPerceiver::new(config);

        let kv: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let queries: Vec<f32> = (0..16).map(|i| i as f32 * 0.3).collect();

        let (keys, values) = perceiver.forward_projected(&kv, &queries);
        assert_eq!(
            keys.len(),
            2 * 8,
            "keys should be [compact_length * kv_dim]"
        );
        assert_eq!(
            values.len(),
            2 * 8,
            "values should be [compact_length * kv_dim]"
        );
    }

    // --- Original tests (preserved) ---

    #[test]
    fn test_config_default() {
        let config = StillPerceiverConfig::default();
        assert_eq!(config.latent_dim, 64);
        assert_eq!(config.compact_length, 128);
        assert_eq!(config.num_blocks, 2);
    }

    #[test]
    fn test_perceiver_new() {
        let config = StillPerceiverConfig::new(32, 64);
        let perceiver = StillPerceiver::new(config);
        assert_eq!(perceiver.config.latent_dim, 32);
        assert_eq!(perceiver.config.compact_length, 64);
    }

    #[test]
    fn test_forward_not_identity() {
        // Replaces test_forward_identity_passthrough — now tests real attention.
        let config = StillPerceiverConfig::new(4, 2);
        let perceiver = StillPerceiver::new(config);
        let kv: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let queries: Vec<f32> = (0..8).map(|i| (i as f32 - 2.0) * 0.5).collect();
        let result = perceiver.forward(&kv, &queries);
        // With real attention, output should differ from queries
        assert_ne!(
            result, queries,
            "forward with real attention should not be identity"
        );
    }

    // --- Dot product helper test ---

    #[test]
    fn test_dot_chunk4() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0f32, 3.0, 4.0, 5.0, 6.0];
        let result = dot_chunk4(&a, &b);
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0 + 5.0 * 6.0;
        assert!((result - expected).abs() < 1e-5, "dot_chunk4 mismatch");
    }

    // --- Softmax test ---

    #[test]
    fn test_softmax_rows() {
        let mut data = vec![1.0f32, 2.0, 3.0, 1.0, 1.0, 1.0];
        // 2 rows, 3 cols
        softmax_rows(&mut data, 2, 3);

        // Row 0: softmax([1,2,3])
        let sum0 = data[0] + data[1] + data[2];
        assert!((sum0 - 1.0).abs() < 1e-5, "softmax row 0 should sum to 1");
        assert!(data[2] > data[1], "softmax should be monotonic");
        assert!(data[1] > data[0], "softmax should be monotonic");

        // Row 1: softmax([1,1,1]) = [1/3, 1/3, 1/3]
        let sum1 = data[3] + data[4] + data[5];
        assert!((sum1 - 1.0).abs() < 1e-5, "softmax row 1 should sum to 1");
        assert!(
            (data[3] - 1.0 / 3.0).abs() < 1e-5,
            "uniform input → uniform softmax"
        );
    }
}
