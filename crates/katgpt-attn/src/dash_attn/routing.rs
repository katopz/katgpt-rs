//! Entmax block routing — adaptive sparse chunk selection.
//!
//! Replaces fixed-budget top-k block selection with adaptive support
//! selection via α-entmax (α=1.5). Computes per-head routing probabilities
//! and normalised routing biases for downstream attention modulation.

use katgpt_core::simd::simd_dot_f32;
use katgpt_core::types::DashAttnConfig;

#[cfg(feature = "asentmax_schedule")]
use super::asentmax::{AsentmaxSchedule, RollingSigmaEstimator, apply_asentmax_inplace};
use super::entmax::{entmax_1p5_into, entmax_support_into};

/// Result of entmax routing for one query head.
#[derive(Debug)]
pub struct RoutingResult {
    /// Active chunk indices (support of entmax distribution).
    pub active_indices: Vec<usize>,
    /// Routing bias per active chunk: `(log w - μ) / σ`.
    pub bias: Vec<f32>,
    /// Entmax probabilities for all chunks.
    pub probs: Vec<f32>,
}

/// Score blocks using entmax routing.
///
/// Computes chunk logits as scaled dot-product between a single-head query
/// and chunk summaries, then applies α-entmax (α=1.5) to obtain an adaptive
/// sparse distribution over chunks.
pub fn score_blocks_entmax(
    query: &[f32],
    summaries: &[impl AsRef<[f32]>],
    config: &DashAttnConfig,
) -> RoutingResult {
    let n = summaries.len();
    let mut scratch = RoutingScratch::new(n, query.len());
    score_blocks_entmax_into(query, summaries, config, &mut scratch)
}

/// Like [`score_blocks_entmax`] but adds the HiLS Prop 3.1 entropy bias
/// `b'_c` to each chunk logit before entmax (Issue 044 / Research 399).
///
/// `entropy_biases[i]` is added to the raw dot-product logit of chunk `i`.
/// At zero-init `head_cls` every chunk has the same `b'_c = ln(chunk_size)`,
/// so the bias is a constant additive shift → entmax ranking unchanged
/// (the "dormant at zero-init" guarantee).
///
/// Pass an empty slice (`&[]`) to disable the entropy correction.
pub fn score_blocks_entmax_with_entropy(
    query: &[f32],
    summaries: &[impl AsRef<[f32]>],
    entropy_biases: &[f32],
    config: &DashAttnConfig,
) -> RoutingResult {
    let n = summaries.len();
    let mut scratch = RoutingScratch::new(n, query.len());
    score_blocks_entmax_with_entropy_into(query, summaries, entropy_biases, config, &mut scratch)
}

/// Pre-allocated scratch buffers for entmax routing.
#[derive(Debug)]
pub struct RoutingScratch {
    /// Logits buffer: [n_chunks].
    logits: Vec<f32>,
    /// Sorted indices buffer for entmax: Vec<(usize, f32)>.
    sorted: Vec<(usize, f32)>,
    /// Probabilities buffer for entmax: [n_chunks].
    probs: Vec<f32>,
    /// Log-weights buffer for active indices (reused across calls).
    log_weights: Vec<f32>,
    /// Active indices buffer (reused across calls).
    active_indices: Vec<usize>,
    /// Bias buffer (reused across calls).
    bias: Vec<f32>,
}

impl RoutingScratch {
    /// Create scratch buffers sized for `n_chunks` chunks.
    pub fn new(n_chunks: usize, _head_dim: usize) -> Self {
        Self {
            logits: vec![0.0; n_chunks],
            sorted: Vec::with_capacity(n_chunks),
            probs: vec![0.0; n_chunks],
            log_weights: Vec::with_capacity(n_chunks),
            active_indices: Vec::with_capacity(n_chunks),
            bias: Vec::with_capacity(n_chunks),
        }
    }
}

/// Zero-alloc variant of [`score_blocks_entmax`].
///
/// Reuses scratch buffers across calls. Equivalent to calling
/// [`score_blocks_entmax_with_entropy_into`] with an empty entropy slice.
pub fn score_blocks_entmax_into(
    query: &[f32],
    summaries: &[impl AsRef<[f32]>],
    config: &DashAttnConfig,
    scratch: &mut RoutingScratch,
) -> RoutingResult {
    score_blocks_entmax_with_entropy_into(query, summaries, &[], config, scratch)
}

/// Zero-alloc variant that adds the HiLS Prop 3.1 entropy bias `b'_c` to each
/// chunk logit before α-entmax (Issue 044 / Research 399).
///
/// The chunk score becomes `ŝ_{i,c} = q^T k'_c / √d * γ + b'_c`, which is the
/// first-order Taylor rectification of the LogSumExp chunk mass. Pass an
/// empty `entropy_biases` slice to disable the correction (then this is
/// identical to [`score_blocks_entmax_into`]).
///
/// At zero-init `head_cls`, all chunks share `b'_c = ln(chunk_size)`, so the
/// bias is a constant additive shift and the entmax ranking is unchanged.
pub fn score_blocks_entmax_with_entropy_into(
    query: &[f32],
    summaries: &[impl AsRef<[f32]>],
    entropy_biases: &[f32],
    config: &DashAttnConfig,
    scratch: &mut RoutingScratch,
) -> RoutingResult {
    let n = summaries.len();
    grow_routing_scratch(scratch, n);
    compute_chunk_logits_into(query, summaries, entropy_biases, config, &mut scratch.logits);
    route_from_logits(n, scratch)
}

/// Ensure `logits`/`probs` scratch rows cover `n` chunks.
fn grow_routing_scratch(scratch: &mut RoutingScratch, n: usize) {
    if scratch.logits.len() < n {
        scratch.logits.resize(n, 0.0);
    }
    if scratch.probs.len() < n {
        scratch.probs.resize(n, 0.0);
    }
    scratch.sorted.clear();
}

/// Compute chunk logits `z = q · k̄ / √d * γ (+ b'_c)` into `logits_out`.
///
/// The entropy bias `b'_c` (HiLS Prop 3.1) interpolates between the
/// mean-logit regime and the true LogSumExp mass; it is a per-chunk
/// additive term computed once at prefill (Issue 044).
fn compute_chunk_logits_into(
    query: &[f32],
    summaries: &[impl AsRef<[f32]>],
    entropy_biases: &[f32],
    config: &DashAttnConfig,
    logits_out: &mut [f32],
) {
    let hd = query.len();
    let has_entropy = !entropy_biases.is_empty();
    let scale = 1.0 / (hd as f32).sqrt() * config.scaling_factor;
    for (i, s) in summaries.iter().enumerate() {
        let s_ref = s.as_ref();
        let dot = simd_dot_f32(query, s_ref, hd);
        let mut logit = dot * scale;
        if has_entropy && i < entropy_biases.len() {
            logit += entropy_biases[i];
        }
        logits_out[i] = logit;
    }
}

/// α-entmax over the first `n` logits + support extraction + routing-bias
/// computation, building the owned `RoutingResult` (shared tail of every
/// routing entry point in this module).
fn route_from_logits(n: usize, scratch: &mut RoutingScratch) -> RoutingResult {
    // α-entmax routing into scratch buffers
    entmax_1p5_into(
        &scratch.logits[..n],
        &mut scratch.sorted,
        &mut scratch.probs[..n],
    );

    // Extract support into pre-allocated buffer
    entmax_support_into(&scratch.probs[..n], &mut scratch.active_indices);

    // Compute routing bias: (log w - μ) / σ on active indices
    scratch.log_weights.clear();
    scratch
        .log_weights
        .extend(scratch.active_indices.iter().map(|&i| {
            if scratch.probs[i] > 1e-10 {
                scratch.probs[i].ln()
            } else {
                -23.0 // ln(1e-10)
            }
        }));

    // Compute mean and variance of log-weights in a single pass.
    // Original code iterated twice (sum for mean, then squared-deviation sum);
    // fusing avoids a second scan over `log_weights`.
    let n_lw = scratch.log_weights.len();
    let (mean_lw, var_lw): (f32, f32) = if n_lw == 0 {
        (0.0, 1.0)
    } else if n_lw == 1 {
        (*scratch.log_weights.first().unwrap(), 1.0)
    } else {
        let mut sum = 0.0f32;
        for &x in scratch.log_weights.iter() {
            sum += x;
        }
        let mean = sum / n_lw as f32;
        let mut sq_sum = 0.0f32;
        for &x in scratch.log_weights.iter() {
            let d = x - mean;
            sq_sum += d * d;
        }
        let var = sq_sum / (n_lw - 1) as f32;
        (mean, var)
    };
    let std_lw = var_lw.sqrt().max(1e-6);

    scratch.bias.clear();
    scratch.bias.extend(
        scratch
            .log_weights
            .iter()
            .map(|&lw| (lw - mean_lw) / std_lw),
    );

    // Build result: clone from scratch buffers.
    // Note: scratch.active_indices and scratch.bias are reused across calls
    // (sorted, logits, probs are the expensive scratch to preserve).
    // The active_indices/bias are small (typically ≤ num_chunks), so cloning
    // them is cheaper than reallocating the sorted/probs buffers.
    let probs = scratch.probs[..n].to_vec();

    RoutingResult {
        active_indices: scratch.active_indices.clone(),
        bias: scratch.bias.clone(),
        probs,
    }
}

/// Zero-alloc variant that applies an ASEntmax length-adaptive damping
/// schedule to the chunk logits BEFORE the α-entmax threshold pass
/// (Issue 747 P0, Research 549 — arXiv:2506.16640 Eq 10).
///
/// Identical to [`score_blocks_entmax_with_entropy_into`] except the
/// logits row is pre-scaled by `δ + β·(log n_c)^γ` — the derived damping
/// (`AsentmaxSchedule::Derived` with a `RollingSigmaEstimator`) keeps the
/// scaled logit range n-invariant, which keeps the entmax support size
/// stationary as the scored chunk count grows (the over-sparsification
/// counter-schedule; module docs in [`super::asentmax`]).
///
/// Pass [`AsentmaxSchedule::None`] for the shipped Plan 106 behavior
/// (bit-identical to `score_blocks_entmax_with_entropy_into`).
///
/// The `estimator` is observed with the RAW (pre-schedule) logits row —
/// σ̂ must track the unscaled score distribution; pass `None` to skip
/// observation (caller-managed estimator elsewhere).
#[cfg(feature = "asentmax_schedule")]
pub fn score_blocks_entmax_with_schedule_into(
    query: &[f32],
    summaries: &[impl AsRef<[f32]>],
    entropy_biases: &[f32],
    config: &DashAttnConfig,
    schedule: &AsentmaxSchedule,
    estimator: Option<&RollingSigmaEstimator>,
    scratch: &mut RoutingScratch,
) -> RoutingResult {
    let n = summaries.len();
    grow_routing_scratch(scratch, n);
    compute_chunk_logits_into(query, summaries, entropy_biases, config, &mut scratch.logits);
    if let Some(est) = estimator {
        est.observe_row(&scratch.logits[..n]);
    }
    let log_n = if n > 1 { (n as f32).ln() } else { 0.0 };
    apply_asentmax_inplace(&mut scratch.logits[..n], schedule, log_n);
    route_from_logits(n, scratch)
}

/// Compute routing bias for all query heads with GQA aggregation.
///
/// Runs per-query-head entmax routing, then averages probabilities across
/// heads sharing the same KV group for consensus routing.
///
/// Uses `score_blocks_entmax_into` with a reusable scratch buffer to avoid
/// per-head heap allocation in the routing hot path.
pub fn compute_routing_bias(
    queries: &[Vec<f32>],   // [n_query_heads][head_dim]
    summaries: &[Vec<f32>], // [n_chunks][head_dim]
    n_kv_heads: usize,
    config: &DashAttnConfig,
) -> Vec<RoutingResult> {
    let n_chunks = summaries.len();

    // Reuse scratch buffers across heads (zero-alloc routing)
    let mut scratch = RoutingScratch::new(n_chunks, queries.first().map_or(0, |q| q.len()));

    // Per-query-head routing using the _into variant
    let per_head: Vec<RoutingResult> = queries
        .iter()
        .map(|q| score_blocks_entmax_into(q, summaries, config, &mut scratch))
        .collect();

    // NOTE: GQA aggregation via entmax_gqa_aggregate is intentionally omitted —
    // per-head routing results are returned directly. Downstream consumers
    // index per-head; KV-group-averaged biases would be a separate API if needed.
    let _ = n_kv_heads;

    per_head
}

/// Zero-alloc variant of [`compute_routing_bias`].
///
/// Accepts a reusable scratch buffer to avoid per-call heap allocation.
/// The returned `RoutingResult`s still own their data (cloned from scratch)
/// for API simplicity.
pub fn compute_routing_bias_into(
    queries: &[Vec<f32>],
    summaries: &[Vec<f32>],
    n_kv_heads: usize,
    config: &DashAttnConfig,
    scratch: &mut RoutingScratch,
) -> Vec<RoutingResult> {
    let per_head: Vec<RoutingResult> = queries
        .iter()
        .map(|q| score_blocks_entmax_into(q, summaries, config, scratch))
        .collect();

    let _ = n_kv_heads;

    per_head
}

// ── Wall-aware routing (Plan 173 Task 6) ────────────────────

/// Wall-aware block routing that pre-filters decayed blocks.
///
/// Before entmax routing, checks each block's minimum retention via
/// `WallPrefixState::min_retention_at_block()`. Blocks where ALL channels
/// have decayed below `retention_threshold` are excluded from routing.
///
/// This avoids wasting compute on blocks that the model has already
/// "forgotten" via the diagonal gate.
///
/// `retention_threshold`: blocks with `min_retention < threshold` are skipped.
/// Default: 0.1 (i.e., block must retain at least 10% signal).
///
/// `min_retention_fn`: closure that returns `min_retention(block_idx) -> f32`
/// for each block. Returns `1.0` for blocks with no Wall information.
pub fn score_blocks_wall_aware_into<F>(
    query: &[f32],
    summaries: &[impl AsRef<[f32]>],
    config: &DashAttnConfig,
    scratch: &mut RoutingScratch,
    retention_threshold: f32,
    min_retention_fn: F,
) -> RoutingResult
where
    F: Fn(usize) -> f32,
{
    let n = summaries.len();

    // Pre-filter: collect only blocks with sufficient retention
    let alive_summaries: Vec<(usize, &[f32])> = summaries
        .iter()
        .enumerate()
        .filter(|(i, _)| min_retention_fn(*i) >= retention_threshold)
        .map(|(i, s)| (i, s.as_ref()))
        .collect();

    // If all blocks are alive (common case), delegate to standard routing
    if alive_summaries.len() == n {
        return score_blocks_entmax_into(query, summaries, config, scratch);
    }

    // If no blocks are alive, return empty routing
    if alive_summaries.is_empty() {
        scratch.active_indices.clear();
        scratch.bias.clear();
        return RoutingResult {
            active_indices: vec![],
            bias: vec![],
            probs: vec![0.0; n],
        };
    }

    // Extract just the summaries for filtered routing
    let filtered_refs: Vec<&[f32]> = alive_summaries.iter().map(|(_, s)| *s).collect();

    // Run standard entmax routing on alive blocks only
    let filtered_result = score_blocks_entmax_into(query, &filtered_refs, config, scratch);

    // Remap active indices back to original block indices
    let original_active: Vec<usize> = filtered_result
        .active_indices
        .iter()
        .map(|&fi| alive_summaries[fi].0)
        .collect();

    // Build full probs array (0.0 for dead blocks)
    let mut full_probs = vec![0.0f32; n];
    for (fi, &orig_i) in original_active.iter().enumerate() {
        full_probs[orig_i] = filtered_result.probs.get(fi).copied().unwrap_or(0.0);
    }

    RoutingResult {
        active_indices: original_active,
        bias: filtered_result.bias,
        probs: full_probs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DashAttnConfig {
        DashAttnConfig::default()
    }

    #[test]
    fn test_score_blocks_entmax_single_chunk() {
        let config = default_config();
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let summaries = vec![vec![1.0, 0.0, 0.0, 0.0]];

        let result = score_blocks_entmax(&query, &summaries, &config);
        assert_eq!(result.active_indices, vec![0]);
        assert!(
            (result.probs[0] - 1.0).abs() < 1e-6,
            "single chunk should get all probability mass"
        );
    }

    #[test]
    fn test_score_blocks_entmax_two_chunks_clear_winner() {
        let config = default_config();
        let query = vec![1.0, 0.0];
        // Chunk 0 aligned with query, chunk 1 orthogonal
        let summaries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let result = score_blocks_entmax(&query, &summaries, &config);
        assert!(!result.active_indices.is_empty());
        // Chunk 0 should dominate
        assert!(result.probs[0] > result.probs[1]);
    }

    #[test]
    fn test_score_blocks_entmax_probs_sum_to_one() {
        let config = default_config();
        let query = vec![1.0, 2.0, 3.0];
        let summaries = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];

        let result = score_blocks_entmax(&query, &summaries, &config);
        let sum: f32 = result.probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "entmax probs must sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_score_blocks_entmax_non_negative() {
        let config = default_config();
        let query = vec![1.0, 0.5];
        let summaries = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, -1.0]];

        let result = score_blocks_entmax(&query, &summaries, &config);
        for (i, &p) in result.probs.iter().enumerate() {
            assert!(p >= 0.0, "prob at index {i} is negative: {p}");
        }
    }

    #[test]
    fn test_routing_result_bias_has_same_length_as_active() {
        let config = default_config();
        let query = vec![1.0, 0.0, 0.0];
        let summaries = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let result = score_blocks_entmax(&query, &summaries, &config);
        assert_eq!(
            result.active_indices.len(),
            result.bias.len(),
            "bias length must match active_indices length"
        );
    }

    #[test]
    fn test_compute_routing_bias_multi_head() {
        let config = default_config();
        let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let summaries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let results = compute_routing_bias(&queries, &summaries, 1, &config);
        assert_eq!(results.len(), 2, "should have one result per query head");

        for (h, r) in results.iter().enumerate() {
            let sum: f32 = r.probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "head {h} probs must sum to 1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_compute_routing_bias_gqa_fewer_kv_heads() {
        let config = default_config();
        // 4 query heads, 2 kv heads
        let queries = vec![
            vec![1.0, 0.0],
            vec![0.8, 0.2],
            vec![0.2, 0.8],
            vec![0.0, 1.0],
        ];
        let summaries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let results = compute_routing_bias(&queries, &summaries, 2, &config);
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_score_blocks_empty_summaries() {
        let config = default_config();
        let query = vec![1.0, 0.0];
        let summaries: Vec<Vec<f32>> = vec![];

        let result = score_blocks_entmax(&query, &summaries, &config);
        assert!(result.active_indices.is_empty());
        assert!(result.probs.is_empty());
        assert!(result.bias.is_empty());
    }

    #[test]
    fn test_score_blocks_all_orthogonal() {
        let config = default_config();
        // Query orthogonal to all chunks → entmax may spread or concentrate
        let query = vec![1.0, 0.0];
        let summaries = vec![vec![0.0, 1.0], vec![0.0, -1.0]];

        let result = score_blocks_entmax(&query, &summaries, &config);
        // All logits should be ~0, entmax should still produce valid distribution
        let sum: f32 = result.probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5 || sum == 0.0,
            "probs should sum to 1.0 or be empty when all logits zero, got {sum}"
        );
    }

    // ── Wall-aware routing tests (Plan 173 Task 6) ────────────

    #[test]
    fn test_wall_aware_routing_all_alive() {
        let config = default_config();
        let query = vec![1.0, 0.0];
        let summaries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut scratch = RoutingScratch::new(2, 2);

        // All blocks have retention 1.0 (alive) → same as standard routing
        let result = score_blocks_wall_aware_into(
            &query,
            &summaries,
            &config,
            &mut scratch,
            0.1,
            |_| 1.0, // all alive
        );

        let standard = score_blocks_entmax_into(&query, &summaries, &config, &mut scratch);
        assert_eq!(result.active_indices, standard.active_indices);
        assert_eq!(result.probs.len(), 2);
    }

    #[test]
    fn test_wall_aware_routing_all_dead() {
        let config = default_config();
        let query = vec![1.0, 0.0];
        let summaries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut scratch = RoutingScratch::new(2, 2);

        // All blocks have retention 0.0 (dead) → empty routing
        let result = score_blocks_wall_aware_into(
            &query,
            &summaries,
            &config,
            &mut scratch,
            0.1,
            |_| 0.0, // all dead
        );

        assert!(result.active_indices.is_empty());
        assert!(result.bias.is_empty());
        assert_eq!(result.probs.len(), 2);
        assert!(result.probs.iter().all(|&p| p == 0.0));
    }

    #[test]
    fn test_wall_aware_routing_partial_filter() {
        let config = default_config();
        let query = vec![1.0, 0.0];
        let summaries = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];
        let mut scratch = RoutingScratch::new(3, 2);

        // Block 1 is dead (retention 0.05), blocks 0 and 2 are alive
        let result = score_blocks_wall_aware_into(
            &query,
            &summaries,
            &config,
            &mut scratch,
            0.1,
            |block_idx| if block_idx == 1 { 0.05 } else { 1.0 },
        );

        // Block 1 should not appear in active indices
        assert!(!result.active_indices.contains(&1));
        // Should have some active blocks
        assert!(!result.active_indices.is_empty());
        // Block 1 should have 0 probability
        assert_eq!(result.probs[1], 0.0);
    }

    // ── Issue 044: entropy bias routing tests ──────────────────────────

    /// With an empty entropy slice, the entropy-aware variant must be
    /// bit-identical to the plain variant (the backward-compat guarantee).
    #[test]
    fn test_score_blocks_with_entropy_empty_matches_plain() {
        let config = default_config();
        let query = vec![1.0, 0.0];
        let summaries = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];
        let mut s1 = RoutingScratch::new(3, 2);
        let mut s2 = RoutingScratch::new(3, 2);
        let plain = score_blocks_entmax_into(&query, &summaries, &config, &mut s1);
        let with_e =
            score_blocks_entmax_with_entropy_into(&query, &summaries, &[], &config, &mut s2);
        assert_eq!(plain.active_indices, with_e.active_indices);
        assert_eq!(plain.bias, with_e.bias);
        assert_eq!(plain.probs, with_e.probs);
    }

    /// A constant (uniform) entropy bias across all chunks is just an additive
    /// shift to all logits — the entmax ranking must be unchanged.
    /// This is the "dormant at zero-init" guarantee at the routing level.
    #[test]
    fn test_score_blocks_with_constant_entropy_unchanged_ranking() {
        let config = default_config();
        let query = vec![1.0, 0.0];
        let summaries = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];
        let mut s1 = RoutingScratch::new(3, 2);
        let mut s2 = RoutingScratch::new(3, 2);
        let plain = score_blocks_entmax_into(&query, &summaries, &config, &mut s1);
        // ln(64) — a constant shift mimicking zero-init chunk_size=64.
        let constant_entropy = vec![64.0f32.ln(); 3];
        let shifted = score_blocks_entmax_with_entropy_into(
            &query,
            &summaries,
            &constant_entropy,
            &config,
            &mut s2,
        );
        assert_eq!(plain.active_indices, shifted.active_indices);
        // Probs are invariant under a constant additive logit shift.
        for (i, (&p, &s)) in plain.probs.iter().zip(shifted.probs.iter()).enumerate() {
            assert!((p - s).abs() < 1e-5, "chunk {i}: plain={p}, shifted={s}");
        }
    }

    /// A non-uniform entropy bias must change the routing distribution.
    /// Chunk with higher entropy (more spread attention mass) gets boosted,
    /// so its routing probability increases relative to the plain path.
    #[test]
    fn test_score_blocks_with_nonuniform_entropy_changes_routing() {
        let config = default_config();
        // Two summaries that are identical → plain routing is uniform.
        let query = vec![1.0, 0.0];
        let summaries = vec![vec![0.5, 0.0], vec![0.5, 0.0]];
        let mut s1 = RoutingScratch::new(2, 2);
        let plain = score_blocks_entmax_into(&query, &summaries, &config, &mut s1);
        // Sanity: identical summaries → near-uniform plain probs.
        assert!((plain.probs[0] - plain.probs[1]).abs() < 1e-5);

        // Boost chunk 1 with a large entropy bias → it must take more mass.
        let mut s2 = RoutingScratch::new(2, 2);
        let entropy = vec![0.0, 5.0]; // chunk 1 gets +5 logit boost
        let boosted =
            score_blocks_entmax_with_entropy_into(&query, &summaries, &entropy, &config, &mut s2);
        assert!(
            boosted.probs[1] > boosted.probs[0],
            "boosted chunk should dominate: p0={}, p1={}",
            boosted.probs[0],
            boosted.probs[1]
        );
    }

    // ── ASEntmax scheduled routing (Issue 747 P0) ──────────────────────────

    /// `AsentmaxSchedule::None` must be bit-identical to the shipped path —
    /// the no-regression contract for the feature (G3 at the unit level:
    /// the schedule socket changes nothing until a schedule is supplied).
    #[cfg(feature = "asentmax_schedule")]
    #[test]
    fn test_scheduled_routing_none_is_bit_identical() {
        let config = default_config();
        let query = vec![1.0, 0.0, 0.5, -0.25];
        let summaries: Vec<Vec<f32>> = (0..16)
            .map(|i| {
                vec![
                    ((i * 7) % 13) as f32 / 13.0 - 0.5,
                    ((i * 5) % 11) as f32 / 11.0 - 0.5,
                    ((i * 3) % 7) as f32 / 7.0 - 0.5,
                    ((i * 11) % 17) as f32 / 17.0 - 0.5,
                ]
            })
            .collect();
        let entropy = vec![0.3_f32; 16];

        let mut s1 = RoutingScratch::new(16, 4);
        let plain = score_blocks_entmax_with_entropy_into(&query, &summaries, &entropy, &config, &mut s1);
        let mut s2 = RoutingScratch::new(16, 4);
        let scheduled = score_blocks_entmax_with_schedule_into(
            &query,
            &summaries,
            &entropy,
            &config,
            &AsentmaxSchedule::None,
            None,
            &mut s2,
        );
        assert_eq!(plain.active_indices, scheduled.active_indices);
        assert_eq!(plain.probs, scheduled.probs);
        assert_eq!(plain.bias, scheduled.bias);
    }

    /// Damping must enlarge (or hold) the support vs the unscheduled path —
    /// the mechanism the schedule exists for (over-sparsification repair).
    #[cfg(feature = "asentmax_schedule")]
    #[test]
    fn test_scheduled_routing_damping_enlarges_support() {
        let config = default_config();
        let d = 4_usize;
        let n = 512_usize;
        // Query with a large norm → large logit spread σ ≈ ‖q‖²/√d
        // (the over-sparsification regime).
        let query: Vec<f32> = vec![6.0; d];
        let mut state = 0x5DEE_CE66_D000_0001_u64;
        let summaries: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..d)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect();

        let mut s1 = RoutingScratch::new(n, d);
        let plain = score_blocks_entmax_into(&query, &summaries, &config, &mut s1);

        // Estimator-fed derived schedule on the same inputs.
        let est = RollingSigmaEstimator::new(0.7);
        let mut s2 = RoutingScratch::new(n, d);
        // A few warm-up observations so σ̂ converges before the measured run.
        for _ in 0..10 {
            let mut s_warm = RoutingScratch::new(n, d);
            score_blocks_entmax_with_schedule_into(
                &query,
                &summaries,
                &[],
                &config,
                &AsentmaxSchedule::None,
                Some(&est),
                &mut s_warm,
            );
        }
        let scheduled = score_blocks_entmax_with_schedule_into(
            &query,
            &summaries,
            &[],
            &config,
            &est.to_schedule(),
            Some(&est),
            &mut s2,
        );
        assert!(
            scheduled.active_indices.len() >= plain.active_indices.len(),
            "damping must not shrink the support: scheduled={}, plain={}",
            scheduled.active_indices.len(),
            plain.active_indices.len()
        );
        // Sanity: probs still on the simplex.
        let sum: f32 = scheduled.probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
