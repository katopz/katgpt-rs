//! Pre-allocated context buffers for forward passes.
//!
//! `ForwardContext` is intentionally NOT here — it lives in the `katgpt-rs` root
//! crate because its fields reference root-only pruner types (`CnaModulator`,
//! `SubstrateMask`, `HydraSkipPlan`).

use katgpt_core::types::Config;

// ──────────────────────────────────────────────────────────────────────────
// PrefillContext — Pre-allocated buffers for bidirectional prefill (Plan 025)
// ──────────────────────────────────────────────────────────────────────────

/// Pre-allocated context for bidirectional prefill phase.
/// Created once at startup, reused across all requests. Zero alloc in request path.
pub struct PrefillContext {
    /// Hidden states for all prompt positions, carried between layers.
    /// Size: [max_prompt_len × n_embd]. Only used when n_layer > 1.
    /// For n_layer == 1, embeddings are computed on-the-fly and this buffer is unused.
    pub hidden: Vec<f32>,
    /// Pre-computed Q projections from fused Phase A, reused in Phase B.
    /// Size: [max_prompt_len × n_embd]. Eliminates redundant hidden load + rmsnorm + Q matmul.
    pub queries: Vec<f32>,
    /// Pre-computed attention residuals (xr) from fused Phase A, reused in Phase B.
    /// Size: [max_prompt_len × n_embd]. Eliminates redundant hidden load + first rmsnorm.
    pub residuals: Vec<f32>,
    /// LoRA intermediate buffer. Size: [lora_rank].
    /// Reused for every LoRA application across all projections.
    pub lora_buf: Vec<f32>,
    // usize fields after Vec fields to eliminate inter-field padding.
    /// Max prompt length this context supports (= config.block_size).
    pub max_prompt_len: usize,
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

// ---------------------------------------------------------------------------
// Wall Attention — Diagonal Forget Gates Replacing RoPE (Plan 173)
// ---------------------------------------------------------------------------

#[cfg(feature = "wall_attention")]
use katgpt_core::simd::{
    simd_add_inplace, simd_add_scalar_inplace, simd_exp_inplace, simd_scale_inplace,
    simd_scale_mul_inplace,
};

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
    pub(crate) prefix_sums: Vec<f32>,
    /// Gate projection buffer: [head_dim]. Pre-allocated, reused each token.
    pub(crate) gate_buf: Vec<f32>,
    /// Temp buffer for exp operations: [head_dim]. Pre-allocated, reused each call.
    pub(crate) gate_exp_buf: Vec<f32>,
    /// Number of KV heads.
    pub(crate) n_kv_head: usize,
    /// Head dimension.
    pub(crate) head_dim: usize,
}

/// Per-channel gate statistics for retrieval head analysis (Plan 173 Task 7).
///
/// Identifies "always-on" (retrieval-critical) vs "dynamic" (recency)
/// dimensions based on prefix sum statistics.
/// - High variance = content-dependent (dynamic = retrieval-critical)
/// - Low variance = stable (always-on = recency)
#[cfg(feature = "wall_attention")]
#[derive(Clone, Debug)]
pub struct GateStatistics {
    /// Per-channel mean of prefix sums.
    pub mean: Vec<f32>,
    /// Per-channel variance of prefix sums.
    pub variance: Vec<f32>,
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
    ///
    /// # GQA exp-cache
    ///
    /// In grouped-query attention (n_head > n_kv_head), consecutive Q heads
    /// typically map to the same KV head (e.g. group size 4 → heads 0..3 all
    /// hit KV head 0). The prefix-sum exp is identical across that group, so
    /// we cache `gate_exp_buf` keyed by the last-computed KV head and skip the
    /// copy + `simd_exp_inplace` on a hit. This turns an O(n_head × hd) exp
    /// pass into O(n_kv_head × hd) — a real win whenever group_size > 1. The
    /// cache is invalidated by a sentinel (`u8::MAX`) at function entry, so a
    /// stale buffer from a previous layer/sequence can never leak in.
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
        // Sentinel that never matches a real KV head index (max n_kv_head is
        // bounded by config; u8::MAX = 255 is unreachable for any sane model).
        let mut last_kv_h: u8 = u8::MAX;
        for (h, &kv_h) in kv_group_lut.iter().enumerate().take(n_head) {
            let q_off = h * hd;
            let p_off = layer_off + kv_h as usize * hd;
            // Cache miss: refill gate_exp_buf = exp(prefix_sums[KV head]).
            // On a hit (same KV head as the previous Q head), skip the copy +
            // exp entirely — the buffer already holds the right values.
            if kv_h != last_kv_h {
                self.gate_exp_buf[..hd].copy_from_slice(&self.prefix_sums[p_off..p_off + hd]);
                simd_exp_inplace(&mut self.gate_exp_buf[..hd]);
                last_kv_h = kv_h;
            }
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

        // Step 6: ln+negate+clamp (scalar — ln not yet SIMD-accelerated)
        // log_sigmoid(logit) = -ln(1 + exp(-logit)) = -softplus(-logit)
        for slot in gate_buf.iter_mut().take(hd) {
            let log_sig = -(*slot).ln();
            *slot = log_sig.clamp(-gate_max, 0.0);
        }
    }

    // ── DashAttention integration (Plan 173 Task 6) ────────────

    /// Compute minimum retention across all channels for a block.
    ///
    /// Given a block spanning positions `[block_start, block_end)`, returns
    /// the minimum `exp(P[block_end] - P[block_start])` across all channels.
    ///
    /// If `min_retention < threshold`, all channels have decayed → block can
    /// be skipped in sparse attention (DashAttention block routing).
    ///
    /// The prefix sums represent cumulative gate values. For a block spanning
    /// positions [s, e), the retention at each channel is `exp(P[e,d] - P[s,d])`.
    /// A retention close to 1.0 means the channel is well-retained; close to 0
    /// means it has decayed.
    ///
    /// Returns `1.0` if prefix sums are not yet computed (no decay).
    //
    // The chunk-4 `for dd in 0..4` loops below are intentional: a fixed
    // 4-element iteration count helps LLVM emit a single SIMD exp+min
    // reduction (see `simd_exp_inplace`). Converting to iterator form
    // (clippy::needless_range_loop) would defeat the auto-vectorizer.
    #[inline]
    #[allow(clippy::needless_range_loop)]
    pub fn min_retention_at_block(
        &self,
        layer_idx: usize,
        kv_head: usize,
        block_start: usize,
        block_end: usize,
    ) -> f32 {
        let hd = self.head_dim;
        let offset = layer_idx * self.n_kv_head * hd + kv_head * hd;
        let ps = &self.prefix_sums;

        // If prefix sums are too short, return 1.0 (no skip)
        if ps.len() < offset + hd {
            return 1.0;
        }

        let current = block_end;
        if current == 0 {
            return 1.0;
        }

        // Chunk-4 min of exp(prefix * block_span / current).
        let block_span = (block_end - block_start) as f32;
        let inv_current = 1.0 / current as f32;
        let mut min_ret: f32 = f32::MAX;

        // SIMD exp + branch-free min reduction. The chunk-4 form fills a
        // 4-element stack buffer with the scaled gate values, runs
        // `simd_exp_inplace` (one NEON/AVX2 exp approximation pass instead of
        // 4 scalar libm `expf` calls), then min-reduces the 4 results. The
        // scalar tail loop handles the `hd % 4` remainder.
        let mut d = 0;
        let mut buf = [0.0f32; 4];
        while d + 4 <= hd {
            for dd in 0..4 {
                buf[dd] = ps[offset + d + dd] * block_span * inv_current;
            }
            simd_exp_inplace(&mut buf);
            for dd in 0..4 {
                min_ret = min_ret.min(buf[dd]);
            }
            d += 4;
        }
        while d < hd {
            let gate_accumulated = ps[offset + d];
            let gate_in_block = gate_accumulated * block_span * inv_current;
            let retention = gate_in_block.exp();
            min_ret = min_ret.min(retention);
            d += 1;
        }

        min_ret
    }

    // ── RTPurbo integration (Plan 173 Task 7) ──────────────────

    /// Compute gate statistics across all KV heads for a given layer.
    ///
    /// Returns per-channel mean and variance of the prefix sums.
    /// High-variance channels are "dynamic" (content-dependent) and should
    /// be weighted more heavily in RTPurbo's low-dim projection.
    ///
    /// # Implementation
    ///
    /// `prefix_sums` is borrowed once into a local `&[_]` rather than re-borrowed
    /// per element via the old `ps(off, d, &self.prefix_sums)` helper. Hoisting
    /// the borrow lets LLVM see the contiguous read pattern and fuse the four
    /// `prefix_sums[off + d + 0..4]` reads into a single SIMD load inside the
    /// chunk-4 loop. The helper is gone — it was pure indirection.
    #[inline]
    pub fn gate_statistics(&self, layer_idx: usize) -> GateStatistics {
        let hd = self.head_dim;
        let n_heads = self.n_kv_head;
        let layer_off = layer_idx * n_heads * hd;
        // Hoist the borrow once — avoids re-borrowing `self.prefix_sums` per
        // element and lets the optimizer see the contiguous access pattern.
        let prefix_sums = self.prefix_sums.as_slice();

        let mut mean = vec![0.0f32; hd];
        let mut variance = vec![0.0f32; hd];

        // Chunk-4 mean accumulation
        for h in 0..n_heads {
            let off = layer_off + h * hd;
            let mut d = 0;
            while d + 4 <= hd {
                mean[d] += prefix_sums[off + d];
                mean[d + 1] += prefix_sums[off + d + 1];
                mean[d + 2] += prefix_sums[off + d + 2];
                mean[d + 3] += prefix_sums[off + d + 3];
                d += 4;
            }
            while d < hd {
                mean[d] += prefix_sums[off + d];
                d += 1;
            }
        }

        // Normalize mean
        let inv_n = 1.0 / n_heads as f32;
        for m in &mut mean {
            *m *= inv_n;
        }

        // Chunk-4 variance accumulation
        for h in 0..n_heads {
            let off = layer_off + h * hd;
            let mut d = 0;
            while d + 4 <= hd {
                for dd in 0..4 {
                    let diff = prefix_sums[off + d + dd] - mean[d + dd];
                    variance[d + dd] += diff * diff;
                }
                d += 4;
            }
            while d < hd {
                let diff = prefix_sums[off + d] - mean[d];
                variance[d] += diff * diff;
                d += 1;
            }
        }

        // Normalize variance (population)
        for v in &mut variance {
            *v *= inv_n;
        }

        GateStatistics { mean, variance }
    }
}
