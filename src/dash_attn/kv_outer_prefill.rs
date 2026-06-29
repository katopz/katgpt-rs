//! KV-outer sparse prefill path — reverse index from KV blocks to queries.
//!
//! Reverses the attention computation: instead of iterating over queries and
//! finding top-k KV blocks (Q-outer), iterate over KV blocks and gather
//! queries that selected them. This enables:
//! - Better cache locality: each KV block is loaded once
//! - Pre-scheduled tile chunking for hot-block load balancing
//! - Two-phase forward: partial outputs + log-sum-exp (LSE) combine
//!
//! Feature gate: `msa_kv_outer` (Plan 256 Phase 2 GOAT gate).

use katgpt_core::simd::{simd_dot_f32, simd_exp_sum_inplace, simd_fused_scale_acc, simd_max_f32};

use super::vortex_flow::{RoutingDecision, VortexFlow, VortexRouter, VortexScratch};

// ---------------------------------------------------------------------------
// KvOuterIndex — reverse index: KV block → list of query indices
// ---------------------------------------------------------------------------

/// Reverse index: for each KV block, which queries selected it.
///
/// Built from the forward (Q-outer) routing decisions. Each query selects
/// top-k blocks; the reverse index inverts this to "for each block, which
/// queries need it". This enables KV-outer iteration: load each KV block
/// once and compute attention for all gathered queries.
#[derive(Debug, Clone)]
pub struct KvOuterIndex {
    /// `queries_for_block[b]` = list of query indices that selected block `b`.
    pub queries_for_block: Vec<Vec<usize>>,
}

impl KvOuterIndex {
    /// Create an empty index with `n_blocks` slots.
    pub fn new(n_blocks: usize) -> Self {
        Self {
            queries_for_block: vec![Vec::new(); n_blocks],
        }
    }

    /// Build reverse index from per-query routing decisions.
    ///
    /// Each `RoutingDecision` contains the blocks selected for that query.
    /// We invert this to produce `queries_for_block[block] = [q0, q1, ...]`.
    pub fn build_from_decisions(decisions: &[RoutingDecision], n_blocks: usize) -> Self {
        let mut idx = Self::new(n_blocks);
        for (q, dec) in decisions.iter().enumerate() {
            for &b in &dec.blocks {
                if b < n_blocks {
                    idx.queries_for_block[b].push(q);
                }
            }
        }
        idx
    }

    /// Number of queries that selected the given block.
    pub fn query_count(&self, block_idx: usize) -> usize {
        match self.queries_for_block.get(block_idx) {
            Some(v) => v.len(),
            None => 0,
        }
    }

    /// Total work units (sum of query counts across all blocks).
    pub fn total_work(&self) -> usize {
        self.queries_for_block.iter().map(|v| v.len()).sum()
    }

    /// Hot blocks sorted by descending query count.
    ///
    /// Returns `(block_idx, query_count)` pairs for blocks with ≥1 query.
    /// Sorting by hotness enables cache-friendly processing: hot blocks are
    /// processed first when they share KV cache lines with subsequent blocks.
    pub fn hot_blocks(&self) -> Vec<(usize, usize)> {
        let mut blocks: Vec<_> = self
            .queries_for_block
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(i, v)| (i, v.len()))
            .collect();
        blocks.sort_by(|a, b| b.1.cmp(&a.1));
        blocks
    }
}

// ---------------------------------------------------------------------------
// SparsePrefillResult — output + log-sum-exp
// ---------------------------------------------------------------------------

/// Sparse prefill output with numerically stable LSE.
///
/// The `lse` (log-sum-exp) array enables safe online combining of partial
/// attention outputs across blocks. Without LSE, repeated softmax-normalized
/// additions would lose precision for large score magnitudes.
#[derive(Debug, Clone)]
pub struct SparsePrefillResult {
    /// Output tensor: `[n_queries * head_dim]` flat.
    pub output: Vec<f32>,
    /// Log-sum-exp per query: `[n_queries]`.
    pub lse: Vec<f32>,
    /// Number of KV blocks that had ≥1 query (hot blocks).
    pub n_blocks_used: usize,
}

// ---------------------------------------------------------------------------
// KvOuterPrefill — main sparse prefill driver
// ---------------------------------------------------------------------------

/// KV-outer sparse prefill: routes queries, builds reverse index, computes
/// attention by iterating over KV blocks and gathering their queries.
///
/// This is the "transpose" of the standard Q-outer approach:
/// - Q-outer: for each query → find top-k blocks → attend
/// - KV-outer: for each block → find queries that want it → attend
///
/// KV-outer wins on cache locality when blocks are reused across queries
/// (common in long-context prefill where many queries attend to the same
/// "needle" blocks).
pub struct KvOuterPrefill {
    /// Underlying router for block selection.
    pub router: VortexRouter,
    /// Tokens per KV block.
    pub block_size: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
}

impl KvOuterPrefill {
    /// Create a new KV-outer prefill driver.
    pub fn new(router: VortexRouter, block_size: usize, head_dim: usize) -> Self {
        Self {
            router,
            block_size,
            head_dim,
        }
    }

    /// Run sparse prefill: route queries → build reverse index → compute attention.
    ///
    /// # Algorithm
    ///
    /// 1. **Phase 1 — Route**: Populate router cache, then run Q-outer routing
    ///    for each query to collect `RoutingDecision`s.
    /// 2. **Phase 2 — Reverse index**: Build `KvOuterIndex` from decisions.
    /// 3. **Phase 3 — Attend**: For each hot KV block, compute attention for
    ///    all gathered queries (dot-product scores + local softmax + weighted
    ///    value accumulation).
    /// 4. **Phase 4 — Combine**: Online LSE-based merge of partial outputs.
    ///
    /// # Arguments
    /// * `queries` — flat `[n_queries * head_dim]`
    /// * `keys` — flat `[n_blocks * block_size * head_dim]`
    /// * `values` — flat `[n_blocks * block_size * head_dim]`
    /// * `n_queries` — number of query tokens
    /// * `n_blocks` — number of KV blocks
    /// * `top_k` — blocks per query to select
    #[allow(clippy::needless_range_loop)] // stride math: t indexes scores[t] AND t*hd offset into block_vals
    pub fn prefill_sparse(
        &self,
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        n_queries: usize,
        n_blocks: usize,
        top_k: usize,
    ) -> SparsePrefillResult {
        let hd = self.head_dim;
        let bs = self.block_size;
        let scale = 1.0 / (hd as f32).sqrt();

        // Phase 1: Build router cache and route all queries.
        let mut cache = self.router.cache_new(n_blocks, hd);
        for b in 0..n_blocks {
            let k_start = b * bs * hd;
            let k_end = k_start + bs * hd;
            let v_start = b * bs * hd;
            let v_end = v_start + bs * hd;
            let block_keys = &keys[k_start..k_end];
            let block_vals = &values[v_start..v_end];
            self.router
                .forward_cache(&mut cache, block_keys, block_vals, b, hd);
        }

        let mut scratch = VortexScratch::new(n_blocks);
        let mut decisions: Vec<RoutingDecision> = Vec::with_capacity(n_queries);
        for q in 0..n_queries {
            let q_start = q * hd;
            let query = &queries[q_start..q_start + hd];
            let dec = self
                .router
                .forward_indexer(query, &cache, n_blocks, top_k, &mut scratch);
            decisions.push(dec);
        }

        // Phase 2: Build reverse index.
        let rev_idx = KvOuterIndex::build_from_decisions(&decisions, n_blocks);

        // Phase 3 + 4: Initialize output and LSE, then attend per-block.
        let mut output = vec![0.0f32; n_queries * hd];
        let mut lse = vec![f32::NEG_INFINITY; n_queries];

        let hot = rev_idx.hot_blocks();
        let n_blocks_used = hot.len();

        for (block_idx, _query_count) in &hot {
            let b = *block_idx;
            let block_keys = &keys[b * bs * hd..(b + 1) * bs * hd];
            let block_vals = &values[b * bs * hd..(b + 1) * bs * hd];

            for &q in &rev_idx.queries_for_block[b] {
                let q_start = q * hd;
                let query = &queries[q_start..q_start + hd];

                // Compute attention scores: query · keys^T / sqrt(d)
                let mut scores = [0.0f32; 256];
                let actual_bs = bs.min(256);
                compute_scores(
                    query,
                    block_keys,
                    actual_bs,
                    hd,
                    scale,
                    &mut scores[..actual_bs],
                );

                // Local softmax: max, exp, sum — SIMD fused subtract+exp+sum
                let max_score = simd_max_f32(&scores[..actual_bs]);
                // Shift by max in-place, then fused exp + sum via SIMD.
                for s in scores[..actual_bs].iter_mut() {
                    *s -= max_score;
                }
                let sum_exp = simd_exp_sum_inplace(&mut scores[..actual_bs]);
                let inv_sum = 1.0 / sum_exp;

                // Local LSE: log(sum exp(scores))
                let lse_local = max_score + sum_exp.ln();

                // Weighted value accumulation: local_out = sum(w_j * v_j)
                let mut local_out = [0.0f32; 256];
                let actual_hd = hd.min(256);
                for t in 0..actual_bs {
                    let w = scores[t] * inv_sum;
                    let v_start = t * hd;
                    simd_fused_scale_acc(
                        &mut local_out[..actual_hd],
                        &block_vals[v_start..v_start + actual_hd],
                        w,
                        actual_hd,
                    );
                }

                // Online LSE combine (logaddexp trick):
                // lse_new = logaddexp(lse[q], lse_local)
                // output[q] = output[q] * exp(lse[q] - lse_new) + local_out * exp(lse_local - lse_new)
                match lse[q] {
                    l if l == f32::NEG_INFINITY => {
                        // First block for this query — set directly.
                        lse[q] = lse_local;
                        output[q_start..q_start + actual_hd]
                            .copy_from_slice(&local_out[..actual_hd]);
                    }
                    lse_prev => {
                        let lse_new = logaddexp(lse_prev, lse_local);
                        let old_scale = (lse_prev - lse_new).exp();
                        let new_scale = (lse_local - lse_new).exp();

                        let out_slice = &mut output[q_start..q_start + actual_hd];
                        for (o, &l) in out_slice.iter_mut().zip(local_out[..actual_hd].iter()) {
                            *o = *o * old_scale + l * new_scale;
                        }
                        lse[q] = lse_new;
                    }
                }
            }
        }

        SparsePrefillResult {
            output,
            lse,
            n_blocks_used,
        }
    }
}

// ---------------------------------------------------------------------------
// Inline helpers — hot-path, zero-allocation
// ---------------------------------------------------------------------------

/// Compute attention scores: `scores[t] = query · keys[t] * scale`.
///
/// Uses `simd_dot_f32` (NEON/AVX2) for the per-token dot product.
#[inline]
#[allow(clippy::needless_range_loop)] // stride math: t indexes scores[t] AND t*head_dim offset into block_keys
fn compute_scores(
    query: &[f32],
    block_keys: &[f32],
    block_size: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) {
    for t in 0..block_size {
        let k_start = t * head_dim;
        let dot = simd_dot_f32(query, &block_keys[k_start..k_start + head_dim], head_dim);
        scores[t] = dot * scale;
    }
}

/// Find maximum value in a slice.
#[inline]
#[allow(dead_code)]
fn find_max(slice: &[f32]) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for &v in slice {
        max = max.max(v);
    }
    max
}

/// Accumulate `w * values` into `out`.
#[inline]
#[allow(dead_code)]
fn accumulate_scaled(out: &mut [f32], values: &[f32], w: f32) {
    let n = out.len().min(values.len());
    // 4-way unrolled for auto-vectorization
    let chunks = n / 4;
    for c in 0..chunks {
        let i = c * 4;
        out[i] += w * values[i];
        out[i + 1] += w * values[i + 1];
        out[i + 2] += w * values[i + 2];
        out[i + 3] += w * values[i + 3];
    }
    for i in (chunks * 4)..n {
        out[i] += w * values[i];
    }
}

/// Numerically stable log-add-exp: `log(exp(a) + exp(b))`.
#[inline]
fn logaddexp(a: f32, b: f32) -> f32 {
    match (a, b) {
        (a, b) if a == f32::NEG_INFINITY => b,
        (a, b) if b == f32::NEG_INFINITY => a,
        (a, b) if a >= b => a + (1.0 + (b - a).exp()).ln(),
        (_, b) => b + (1.0 + (a - b).exp()).ln(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dash_attn::msa_distill::MaxPoolBlockScorer;

    const HD: usize = 8;
    const BS: usize = 4;

    fn make_prefill() -> KvOuterPrefill {
        let router = VortexRouter::MsaMaxPool(MaxPoolBlockScorer::new(BS));
        KvOuterPrefill::new(router, BS, HD)
    }

    // --- KvOuterIndex tests ---

    #[test]
    fn test_kv_outer_index_build() {
        let mut d0 = RoutingDecision::new();
        d0.blocks.push(0);
        d0.blocks.push(2);

        let mut d1 = RoutingDecision::new();
        d1.blocks.push(1);
        d1.blocks.push(2);

        let mut d2 = RoutingDecision::new();
        d2.blocks.push(0);

        let idx = KvOuterIndex::build_from_decisions(&[d0, d1, d2], 3);

        assert_eq!(idx.queries_for_block[0], vec![0, 2]);
        assert_eq!(idx.queries_for_block[1], vec![1]);
        assert_eq!(idx.queries_for_block[2], vec![0, 1]);
        assert_eq!(idx.query_count(0), 2);
        assert_eq!(idx.query_count(1), 1);
        assert_eq!(idx.query_count(2), 2);
        assert_eq!(idx.total_work(), 5);
    }

    #[test]
    fn test_kv_outer_hot_blocks() {
        let mut d0 = RoutingDecision::new();
        d0.blocks.push(0);

        let mut d1 = RoutingDecision::new();
        d1.blocks.push(0);
        d1.blocks.push(1);

        let mut d2 = RoutingDecision::new();
        d2.blocks.push(0);
        d2.blocks.push(1);
        d2.blocks.push(2);

        let idx = KvOuterIndex::build_from_decisions(&[d0, d1, d2], 4);
        let hot = idx.hot_blocks();

        // Block 0 has 3 queries, block 1 has 2, block 2 has 1, block 3 has 0 (excluded).
        assert_eq!(hot.len(), 3);
        assert_eq!(hot[0], (0, 3));
        assert_eq!(hot[1], (1, 2));
        assert_eq!(hot[2], (2, 1));
    }

    // --- Sparse prefill tests ---

    #[test]
    fn test_sparse_prefill_single_block() {
        // Single block = dense attention. Verify output matches manual computation.
        let n_queries = 2;
        let n_blocks = 1;

        // Queries: [1, 0, 0, 0, 0, 0, 0, 0] and [0, 1, 0, 0, 0, 0, 0, 0]
        let queries = vec![
            1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        // Keys: 4 tokens, each with dim 8
        let keys = vec![
            // token 0: aligned with query 0
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // token 1: aligned with query 1
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // token 2: orthogonal
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // token 3: orthogonal
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ];

        // Values: identity-like
        let values = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let prefill = make_prefill();
        let result = prefill.prefill_sparse(&queries, &keys, &values, n_queries, n_blocks, 1);

        assert_eq!(result.output.len(), n_queries * HD);
        assert_eq!(result.lse.len(), n_queries);
        assert_eq!(result.n_blocks_used, 1);

        // With single block and top_k=1, both queries must select block 0.
        // The output should be the softmax-weighted values.
        // Query 0 scores: [1, 0, 0, 0] * scale → softmax concentrates on token 0.
        // Output 0 ≈ value[0] = [1, 0, 0, 0, 0, 0, 0, 0]
        let scale = 1.0 / (HD as f32).sqrt();
        let score_q0_t0 = 1.0 * scale; // only non-zero score
        let exp_s = score_q0_t0.exp();
        let sum_exp = exp_s + 3.0; // 3 zero-score terms: exp(0)=1
        let w0 = exp_s / sum_exp;

        // Output for query 0 should be w0 * value[0] + (1-w0)/3 * (value[1]+value[2]+value[3])
        let out0 = &result.output[0..HD];
        let expected_v0 = w0 * 1.0;
        assert!(
            (out0[0] - expected_v0).abs() < 1e-5,
            "output[0][0] = {}, expected {}",
            out0[0],
            expected_v0
        );
        // LSE should be finite
        assert!(result.lse[0].is_finite());
    }

    #[test]
    fn test_sparse_prefill_two_blocks_needle() {
        // Two blocks: block 0 is "haystack" (low scores), block 1 is "needle" (high score).
        // Query aligns with needle. With top_k=1, only needle block selected.
        let n_queries = 1;
        let n_blocks = 2;

        // Query: [1, 0, 0, 0, 0, 0, 0, 0]
        let queries = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Block 0 (haystack): tokens with low dot product to query
        let mut keys = vec![0.0f32; n_blocks * BS * HD];
        // Block 0: all orthogonal
        for t in 0..BS {
            let offset = t * HD;
            keys[offset + 2] = 1.0; // dim 2 — orthogonal to query
        }
        // Block 1 (needle): one token with high dot product
        let b1_offset = BS * HD;
        keys[b1_offset] = 1.0; // token 0 of block 1: aligned with query

        // Values: block 0 values are zero, block 1 values are non-zero
        let mut vals = vec![0.0f32; n_blocks * BS * HD];
        let b1v_offset = BS * HD;
        vals[b1v_offset] = 5.0; // value for needle token

        let prefill = make_prefill();
        let result = prefill.prefill_sparse(&queries, &keys, &vals, n_queries, n_blocks, 1);

        // With top_k=1, only block 1 should be selected (needle has higher max score).
        // The output should be based entirely on block 1's tokens.
        assert_eq!(result.n_blocks_used, 1);
        assert!(
            result.output[0] > 0.0,
            "output should have non-zero first dim from needle value, got {}",
            result.output[0]
        );
    }

    #[test]
    fn test_lse_combine_numerical_stability() {
        // Two blocks, same query selects both. Verify combined output matches
        // sequential (dense) computation.
        let n_queries = 1;
        let n_blocks = 2;

        // Query: [1, 1, 0, 0, 0, 0, 0, 0]
        let queries = vec![1.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Block 0: tokens with moderate scores
        let mut keys = vec![0.0f32; n_blocks * BS * HD];
        // Block 0, token 0: partial alignment
        keys[0] = 0.5;
        keys[1] = 0.5;
        // Block 0, token 1: different alignment
        keys[HD] = 0.3;
        keys[HD + 1] = 0.3;

        // Block 1: tokens with strong alignment
        let b1 = BS * HD;
        keys[b1] = 1.0;
        keys[b1 + 1] = 1.0;
        keys[b1 + HD] = 0.8;
        keys[b1 + HD + 1] = 0.8;

        // Values: distinct per token
        let mut vals = vec![0.0f32; n_blocks * BS * HD];
        vals[0] = 1.0;
        vals[1] = 2.0;
        vals[HD] = 3.0;
        vals[HD + 1] = 4.0;
        vals[b1] = 10.0;
        vals[b1 + 1] = 20.0;
        vals[b1 + HD] = 30.0;
        vals[b1 + HD + 1] = 40.0;

        // Run sparse with top_k=2 (select both blocks)
        let prefill = make_prefill();
        let result = prefill.prefill_sparse(&queries, &keys, &vals, n_queries, n_blocks, 2);

        // Compute reference dense attention (all tokens concatenated)
        let all_keys = &keys[..n_blocks * BS * HD];
        let all_vals = &vals[..n_blocks * BS * HD];
        let n_tokens = n_blocks * BS;
        let scale = 1.0 / (HD as f32).sqrt();

        let mut dense_scores = vec![0.0f32; n_tokens];
        compute_scores(&queries, all_keys, n_tokens, HD, scale, &mut dense_scores);

        let max_s = find_max(&dense_scores);
        let mut sum_exp = 0.0f32;
        for s in dense_scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;

        let mut dense_out = vec![0.0f32; HD];
        // Stride math: t indexes dense_scores AND computes v_start = t * HD.
        #[allow(clippy::needless_range_loop)]
        for t in 0..n_tokens {
            let w = dense_scores[t] * inv_sum;
            let v_start = t * HD;
            accumulate_scaled(&mut dense_out, &all_vals[v_start..v_start + HD], w);
        }

        // Compare sparse output to dense reference
        // Multi-array + format: d indexes result.output AND dense_out.
        #[allow(clippy::needless_range_loop)]
        for d in 0..HD {
            let diff = (result.output[d] - dense_out[d]).abs();
            assert!(
                diff < 1e-4,
                "dim {}: sparse={}, dense={}, diff={}",
                d,
                result.output[d],
                dense_out[d],
                diff
            );
        }
    }
}

// ---------------------------------------------------------------------------
// TL;DR
//
// KV-outer sparse prefill (Plan 256 Phase 2):
// - KvOuterIndex: reverse index from KV blocks to queries
// - KvOuterPrefill: route → reverse index → per-block attention → LSE combine
// - Feature gate: msa_kv_outer (depends on msa_sparse)
// - 4-way unrolled dot products for auto-vectorization
// - Online logaddexp for numerically stable partial output merging
// - Tests: index build, hot blocks, single-block dense parity, needle-in-haystack, LSE stability
// ---------------------------------------------------------------------------
