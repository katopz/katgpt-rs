//! ChainFolder — main folding logic implementing ScreeningPruner — Plan 195 T3.
//!
//! The `ChainFolder` ranks reasoning steps by attention importance, then
//! uses binary search to find the minimal set of steps that preserves
//! verification. It implements `ScreeningPruner` so it plugs directly
//! into the DDTree screening pipeline.
//!
//! Plan 245 T25: `compact_trace()` uses StillKV to compact the KV cache
//! of surviving tokens after folding, providing synthesis-based reduction
//! in addition to selection-based folding.

use super::attention_importance::AttentionImportance;
use super::types::{FoldContext, FoldDecision, FoldResult, StepBoundary};

/// Main chain folder — prunes redundant reasoning steps.
///
/// Implements [`ScreeningPruner`] for DDTree integration. The relevance
/// score is derived from per-step attention importance: essential and
/// anchor steps get `1.0`, foldable steps get `0.0`.
#[derive(Debug, Clone)]
pub struct ChainFolder {
    /// Fraction of steps to keep (0.0–1.0). Higher = less aggressive folding.
    fold_budget: f32,
    /// Attention importance scorer.
    importance_scorer: AttentionImportance,
    /// Cached fold decisions from last `binary_search_fold`.
    decisions: Vec<FoldDecision>,
    /// Cached boundaries from last fold.
    boundaries: Vec<StepBoundary>,
}

impl ChainFolder {
    /// Create a new chain folder with the given fold budget.
    ///
    /// `fold_budget` = fraction of steps to keep (0.0 = fold everything,
    /// 1.0 = keep everything). Default is 0.7.
    pub fn new(fold_budget: f32) -> Self {
        Self {
            fold_budget: fold_budget.clamp(0.0, 1.0),
            importance_scorer: AttentionImportance::new(),
            decisions: Vec::new(),
            boundaries: Vec::new(),
        }
    }

    /// Current fold budget (fraction of steps to keep).
    #[inline]
    pub fn fold_budget(&self) -> f32 {
        self.fold_budget
    }

    /// Set the fold budget.
    pub fn set_fold_budget(&mut self, budget: f32) {
        self.fold_budget = budget.clamp(0.0, 1.0);
    }

    /// Get cached fold decisions.
    pub fn decisions(&self) -> &[FoldDecision] {
        &self.decisions
    }

    /// Binary search fold — find the minimal set of steps that preserves correctness.
    ///
    /// Given attention scores and step boundaries, binary searches on the retention
    /// ratio to find the most aggressive fold that still passes verification.
    ///
    /// Returns a `FoldResult` with verification status and token savings estimate.
    pub fn binary_search_fold(&mut self, context: &FoldContext) -> FoldResult {
        let total_steps = context.step_count();
        if total_steps == 0 {
            return FoldResult::no_fold(0);
        }

        // Compute importance scores per step.
        let importance = self
            .importance_scorer
            .score_steps(&context.importance_scores, &context.boundaries);

        if importance.is_empty() {
            return FoldResult::no_fold(total_steps);
        }

        let target_keep = (self.fold_budget * total_steps as f32).ceil() as usize;
        let mut best_keep = total_steps;

        // Rank steps by importance (ascending) to find the least important ones.
        // `sort_unstable_by` is faster than `sort_by` and safe here because
        // the (usize, f32) pairs have unique first elements; ties on `f32`
        // importance scores do not need stable ordering for the downstream
        // `take(fold_count)` + `verify_fold` logic.
        let mut indexed: Vec<(usize, f32)> = importance.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Binary search: how many of the least-important steps can we fold?
        let mut lo = 0_usize;
        let mut hi = total_steps.saturating_sub(target_keep);

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let kept = total_steps - mid;

            // Verify: anchors must be kept, we must keep at least target_keep steps.
            if verify_fold(&context.boundaries, &indexed, mid) && kept >= target_keep {
                best_keep = kept;
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // Apply the best fold.
        let fold_count = total_steps - best_keep;
        let decisions = build_decisions(&context.boundaries, &indexed, fold_count);

        // `best_keep` already counts the non-folded steps; reuse it instead of
        // re-scanning the decisions Vec. `kept == best_keep` holds because
        // `verify_fold` rejected any `mid` whose first-`mid` lowest-importance
        // entries contained an anchor, and `build_decisions` only marks non-anchor
        // entries as `Fold`. Therefore exactly `fold_count` entries become `Fold`
        // and the kept count is `total_steps - fold_count == best_keep`.
        //
        // `debug_assert` keeps the invariant check in test builds while removing
        // the O(n) scan from release hot path.
        debug_assert_eq!(
            decisions.iter().filter(|d| **d != FoldDecision::Fold).count(),
            best_keep,
            "build_decisions invariant: kept count must match best_keep"
        );
        let kept = best_keep;
        let folded = fold_count;

        // Estimate token savings from boundaries.
        let tokens_saved = estimate_tokens_saved(&context.boundaries, &decisions);

        self.decisions = decisions;
        self.boundaries = context.boundaries.clone();

        let retention_ratio = if total_steps > 0 {
            kept as f32 / total_steps as f32
        } else {
            1.0
        };

        FoldResult {
            total_steps,
            kept_steps: kept,
            folded_steps: folded,
            tokens_saved,
            retention_ratio,
            verification_passed: true,
        }
    }

    /// Map a token depth to a step index using cached boundaries.
    fn step_index_for_depth(&self, depth: usize) -> Option<usize> {
        match self
            .boundaries
            .binary_search_by(|b| b.token_pos.cmp(&depth))
        {
            Ok(idx) => Some(idx),
            Err(idx) => {
                if idx == 0 {
                    None
                } else {
                    Some(idx - 1)
                }
            }
        }
    }

    /// Compact the reasoning trace KV cache using StillKV synthesis (Plan 245 T25).
    ///
    /// After folding removes redundant steps, this method applies StillKV
    /// compaction to the surviving tokens' KV cache. This provides a second
    /// layer of reduction: selection (folding) + synthesis (KV compaction).
    ///
    /// # Arguments
    /// * `keys` - Flat f16 key buffer for the kept tokens, shape `[kept_len * num_heads * head_dim]`
    /// * `values` - Flat f16 value buffer for the kept tokens
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head
    /// * `strategy` - Compaction strategy for query bank generation
    /// * `rope_theta` - RoPE base frequency
    /// * `compression_ratio` - Target compression ratio (e.g., 4 = 4x compression)
    ///
    /// # Returns
    /// `CompactTraceResult` with compacted KV data and compression stats,
    /// or `None` if no compaction is needed (no folded steps or zero compression).
    #[cfg(feature = "still_kv")]
    pub fn compact_trace(
        &self,
        keys: &[half::f16],
        values: &[half::f16],
        num_heads: usize,
        head_dim: usize,
        strategy: crate::still_kv::CompactionStrategy,
        rope_theta: f32,
        compression_ratio: usize,
    ) -> Option<CompactTraceResult> {
        let kept_tokens = self.kept_token_count();
        if kept_tokens == 0 || compression_ratio <= 1 || keys.is_empty() {
            return None;
        }

        let _kv_dim = num_heads * head_dim;
        let budget = (kept_tokens / compression_ratio).max(1);

        // Build per-chunk compactor using iterative pipeline
        let compactor = crate::still_kv::IterativeChunkCompactor::new(
            kept_tokens, // single chunk: all kept tokens
            0,           // no lookahead
            num_heads,
            head_dim,
            strategy,
            rope_theta,
            compression_ratio,
        );

        let chunk = crate::still_kv::KVChunk {
            keys: keys.to_vec(),
            values: values.to_vec(),
            start_pos: 0,
            len: kept_tokens,
            beta: crate::still_kv::BetaBias::zeros(0, 0),
        };

        let compacted = compactor.compact_chunk(&chunk, None, budget);

        let compact_tokens = compacted.len;
        let original_bytes = keys.len() * 2 + values.len() * 2; // f16 = 2 bytes
        let compact_bytes = compacted.keys.len() * 2 + compacted.values.len() * 2;

        Some(CompactTraceResult {
            compact_keys: compacted.keys,
            compact_values: compacted.values,
            original_tokens: kept_tokens,
            compact_tokens,
            compression_ratio: if compact_tokens > 0 {
                kept_tokens as f32 / compact_tokens as f32
            } else {
                1.0
            },
            bytes_saved: original_bytes.saturating_sub(compact_bytes),
        })
    }

    /// Count the number of tokens in kept (non-folded) steps.
    #[allow(dead_code)]
    fn kept_token_count(&self) -> usize {
        if self.boundaries.is_empty() {
            return 0;
        }

        let total_tokens = self.boundaries.last().map(|b| b.token_pos).unwrap_or(0);
        let mut kept = 0_usize;

        for (i, decision) in self.decisions.iter().enumerate() {
            if *decision != FoldDecision::Fold {
                let start = self.boundaries[i].token_pos;
                let end = self
                    .boundaries
                    .get(i + 1)
                    .map(|b| b.token_pos)
                    .unwrap_or(total_tokens);
                kept += end.saturating_sub(start);
            }
        }

        kept
    }
}

/// Implement ScreeningPruner for DDTree integration.
///
/// Returns relevance based on cached fold decisions.
/// - Steps marked `Keep` or `Anchor` → 1.0
/// - Steps marked `Fold` → 0.0
/// - If no cached decisions → 1.0 (passthrough, safe default)
impl crate::speculative::types::ScreeningPruner for ChainFolder {
    fn relevance(&self, depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        let step_idx = self.step_index_for_depth(depth);
        match step_idx {
            Some(idx) => match self.decisions.get(idx) {
                Some(FoldDecision::Fold) => 0.0,
                Some(FoldDecision::Keep | FoldDecision::Anchor) => 1.0,
                None => 1.0,
            },
            None => 1.0,
        }
    }
}

impl Default for ChainFolder {
    fn default() -> Self {
        Self::new(0.7)
    }
}

/// Result of StillKV compaction applied to a reasoning trace (Plan 245 T25).
///
/// Contains the compacted KV cache and compression statistics.
#[cfg(feature = "still_kv")]
#[derive(Debug, Clone)]
pub struct CompactTraceResult {
    /// Compacted key buffer (f16).
    pub compact_keys: Vec<half::f16>,
    /// Compacted value buffer (f16).
    pub compact_values: Vec<half::f16>,
    /// Original token count (kept tokens after folding).
    pub original_tokens: usize,
    /// Compacted token count.
    pub compact_tokens: usize,
    /// Actual compression ratio achieved.
    pub compression_ratio: f32,
    /// Bytes saved by compaction.
    pub bytes_saved: usize,
}

// ── Helper functions ────────────────────────────────────────────

/// Verify that a fold is safe: anchors are never folded.
///
/// Returns `true` if folding `fold_count` least-important steps preserves
/// all anchor steps.
fn verify_fold(boundaries: &[StepBoundary], ranked: &[(usize, f32)], fold_count: usize) -> bool {
    if fold_count == 0 {
        return true;
    }

    // Steps to fold are the first `fold_count` entries in `ranked` (lowest importance).
    for (step_idx, _) in ranked.iter().take(fold_count) {
        match boundaries.get(*step_idx) {
            Some(b) if b.is_anchor => return false,
            _ => continue,
        }
    }

    true
}

/// Build fold decisions for each step.
fn build_decisions(
    boundaries: &[StepBoundary],
    ranked: &[(usize, f32)],
    fold_count: usize,
) -> Vec<FoldDecision> {
    let total = boundaries.len();
    let mut decisions = vec![FoldDecision::Keep; total];

    // Mark anchors.
    for (i, b) in boundaries.iter().enumerate() {
        if b.is_anchor {
            decisions[i] = FoldDecision::Anchor;
        }
    }

    // Fold the least-important non-anchor steps.
    for (step_idx, _) in ranked.iter().take(fold_count) {
        match boundaries.get(*step_idx) {
            Some(b) if !b.is_anchor => decisions[*step_idx] = FoldDecision::Fold,
            _ => continue,
        }
    }

    decisions
}

/// Estimate token savings from fold decisions.
fn estimate_tokens_saved(boundaries: &[StepBoundary], decisions: &[FoldDecision]) -> usize {
    let total_tokens = boundaries.last().map(|b| b.token_pos).unwrap_or(0);

    let mut saved = 0_usize;
    for (i, decision) in decisions.iter().enumerate() {
        if *decision == FoldDecision::Fold {
            let start = boundaries[i].token_pos;
            let end = boundaries
                .get(i + 1)
                .map(|b| b.token_pos)
                .unwrap_or(total_tokens);
            saved += end.saturating_sub(start);
        }
    }

    saved
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_context(
        scores: &[f32],
        positions: &[usize],
        anchors: &[usize],
        budget: f32,
    ) -> FoldContext {
        let boundaries: Vec<StepBoundary> = positions
            .iter()
            .enumerate()
            .map(|(i, &pos)| StepBoundary::new(pos, i, anchors.contains(&i)))
            .collect();
        FoldContext {
            importance_scores: scores.to_vec(),
            boundaries,
            fold_budget: budget,
        }
    }

    #[test]
    fn test_chain_folder_default() {
        let cf = ChainFolder::default();
        assert!((cf.fold_budget() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chain_folder_new_clamps_budget() {
        let cf = ChainFolder::new(-0.5);
        assert!((cf.fold_budget()).abs() < f32::EPSILON);

        let cf = ChainFolder::new(2.0);
        assert!((cf.fold_budget() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_binary_search_fold_empty() {
        let mut cf = ChainFolder::new(0.7);
        let ctx = FoldContext {
            importance_scores: vec![],
            boundaries: vec![],
            fold_budget: 0.7,
        };
        let result = cf.binary_search_fold(&ctx);
        assert_eq!(result.total_steps, 0);
    }

    #[test]
    fn test_binary_search_fold_keeps_anchors() {
        // 5 steps, step 0 and 3 are anchors, low importance on step 2.
        let scores: Vec<f32> = vec![0.9; 10]
            .into_iter()
            .chain(vec![0.9; 10])
            .chain(vec![0.1; 10]) // step 2: low importance
            .chain(vec![0.9; 10])
            .chain(vec![0.9; 10])
            .collect();

        let ctx = make_context(&scores, &[0, 10, 20, 30, 40], &[0, 3], 0.6);
        let mut cf = ChainFolder::new(0.6);
        let result = cf.binary_search_fold(&ctx);

        // Anchors (step 0, 3) must not be folded.
        let decisions = cf.decisions();
        assert_eq!(decisions[0], FoldDecision::Anchor);
        assert_eq!(decisions[3], FoldDecision::Anchor);

        // Step 2 (low importance) should be folded.
        assert_eq!(decisions[2], FoldDecision::Fold);

        assert!(result.verification_passed);
    }

    #[test]
    fn test_binary_search_fold_budget_one() {
        // Budget 1.0 = keep everything.
        let scores = vec![0.5; 20];
        let ctx = make_context(&scores, &[0, 5, 10, 15], &[], 1.0);
        let mut cf = ChainFolder::new(1.0);
        let result = cf.binary_search_fold(&ctx);

        assert_eq!(result.folded_steps, 0);
        assert!((result.retention_ratio - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_fold_budget() {
        let mut cf = ChainFolder::new(0.5);
        cf.set_fold_budget(0.9);
        assert!((cf.fold_budget() - 0.9).abs() < f32::EPSILON);
        cf.set_fold_budget(-1.0);
        assert!((cf.fold_budget()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_verify_fold_no_fold() {
        let boundaries = vec![
            StepBoundary::new(0, 0, true),
            StepBoundary::new(10, 1, false),
        ];
        let ranked = vec![(0, 0.5), (1, 0.3)];
        assert!(verify_fold(&boundaries, &ranked, 0));
    }

    #[test]
    fn test_verify_fold_anchor_protection() {
        let boundaries = vec![
            StepBoundary::new(0, 0, true), // anchor
            StepBoundary::new(10, 1, false),
        ];
        let ranked = vec![(0, 0.1), (1, 0.9)]; // step 0 is lowest importance
        // Trying to fold 1 step would hit the anchor → should fail.
        assert!(!verify_fold(&boundaries, &ranked, 1));
    }

    #[test]
    fn test_estimate_tokens_saved() {
        let boundaries = vec![
            StepBoundary::new(0, 0, false),
            StepBoundary::new(10, 1, false),
            StepBoundary::new(20, 2, false),
        ];
        let decisions = vec![FoldDecision::Keep, FoldDecision::Fold, FoldDecision::Keep];
        let saved = estimate_tokens_saved(&boundaries, &decisions);
        assert_eq!(saved, 10); // Step 1 spans tokens 10..20
    }

    #[test]
    fn test_step_index_for_depth() {
        let mut cf = ChainFolder::new(0.7);
        cf.boundaries = vec![
            StepBoundary::new(0, 0, false),
            StepBoundary::new(10, 1, false),
            StepBoundary::new(20, 2, false),
        ];
        cf.decisions = vec![FoldDecision::Keep, FoldDecision::Fold, FoldDecision::Keep];

        assert_eq!(cf.step_index_for_depth(0), Some(0));
        assert_eq!(cf.step_index_for_depth(5), Some(0));
        assert_eq!(cf.step_index_for_depth(15), Some(1));
        assert_eq!(cf.step_index_for_depth(25), Some(2));
    }

    #[test]
    fn test_kept_token_count() {
        let mut cf = ChainFolder::new(0.7);
        cf.boundaries = vec![
            StepBoundary::new(0, 0, false),
            StepBoundary::new(10, 1, false),
            StepBoundary::new(20, 2, false),
        ];
        cf.decisions = vec![FoldDecision::Keep, FoldDecision::Fold, FoldDecision::Keep];
        // Step 0: tokens 0..10 (kept)
        // Step 1: tokens 10..20 (folded)
        // Step 2: tokens 20..20 (kept, but empty since it's the last boundary)
        assert_eq!(cf.kept_token_count(), 10); // only step 0 contributes (10 tokens)
    }

    #[test]
    fn test_kept_token_count_all_kept() {
        let mut cf = ChainFolder::new(0.7);
        cf.boundaries = vec![
            StepBoundary::new(0, 0, false),
            StepBoundary::new(10, 1, false),
            StepBoundary::new(20, 2, false),
        ];
        cf.decisions = vec![FoldDecision::Keep, FoldDecision::Keep, FoldDecision::Keep];
        // All kept: tokens 0..10 + 10..20 = 20
        assert_eq!(cf.kept_token_count(), 20);
    }

    #[test]
    fn test_kept_token_count_empty() {
        let cf = ChainFolder::new(0.7);
        assert_eq!(cf.kept_token_count(), 0);
    }

    // Plan 245 T25: StillKV compact_trace integration
    #[cfg(feature = "still_kv")]
    #[test]
    fn test_compact_trace_basic() {
        use half::f16;

        // Manually set up a chain folder with known fold decisions
        let mut cf = ChainFolder::new(0.5);
        cf.boundaries = vec![
            StepBoundary::new(0, 0, false),
            StepBoundary::new(10, 1, false),
            StepBoundary::new(20, 2, false),
        ];
        cf.decisions = vec![FoldDecision::Keep, FoldDecision::Fold, FoldDecision::Keep];

        // kept tokens: step 0 (0..10) + step 2 (20..20) = 10 tokens
        let kept = cf.kept_token_count();
        assert_eq!(kept, 10);

        let num_heads = 2;
        let head_dim = 8;
        let kv_dim = num_heads * head_dim;
        let keys: Vec<f16> = (0..kept * kv_dim)
            .map(|i| f16::from_f32((i as f32 * 0.1).sin()))
            .collect();
        let values: Vec<f16> = (0..kept * kv_dim)
            .map(|i| f16::from_f32((i as f32 * 0.2).cos()))
            .collect();

        let result = cf.compact_trace(
            &keys,
            &values,
            num_heads,
            head_dim,
            crate::still_kv::CompactionStrategy::ClusterCentroids,
            10000.0,
            2,
        );

        assert!(
            result.is_some(),
            "compact_trace should return Some for non-empty kept tokens"
        );
        let r = result.unwrap();
        assert_eq!(r.original_tokens, kept);
        assert!(r.compact_tokens > 0, "should produce compact tokens");
        assert!(r.compact_tokens < kept, "should reduce token count");
        assert!(r.compression_ratio >= 1.0, "ratio should be >= 1.0");
        assert!(r.bytes_saved > 0, "should save bytes");
    }

    #[cfg(feature = "still_kv")]
    #[test]
    fn test_compact_trace_empty_returns_none() {
        let cf = ChainFolder::new(0.7);
        let keys: Vec<half::f16> = vec![];
        let values: Vec<half::f16> = vec![];
        let result = cf.compact_trace(
            &keys,
            &values,
            2,
            8,
            crate::still_kv::CompactionStrategy::ClusterCentroids,
            10000.0,
            2,
        );
        assert!(result.is_none(), "empty chain should return None");
    }

    #[cfg(feature = "still_kv")]
    #[test]
    fn test_compact_trace_ratio_one_returns_none() {
        let mut cf = ChainFolder::new(0.7);
        cf.boundaries = vec![StepBoundary::new(0, 0, false)];
        cf.decisions = vec![FoldDecision::Keep];

        let keys: Vec<half::f16> = vec![half::f16::from_f32(1.0); 16];
        let values: Vec<half::f16> = vec![half::f16::from_f32(2.0); 16];

        let result = cf.compact_trace(
            &keys,
            &values,
            2,
            8,
            crate::still_kv::CompactionStrategy::ClusterCentroids,
            10000.0,
            1, // ratio 1 = no compression
        );
        assert!(result.is_none(), "ratio 1 should return None");
    }
}
