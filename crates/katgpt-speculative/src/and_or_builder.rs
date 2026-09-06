//! AND-OR DDTree builder: converts flat marginals into AND-OR tree using relevance signal.
//!
//! Part of Plan 190 T2. Uses `ScreeningPruner::relevance()` to detect low-confidence
//! regions in the marginal sequence. Low relevance → decompose into subgoals (AND).
//! High relevance → solve directly (leaf). The root is always OR (try alternatives).
//!
//! Subgoals are memoized via `ProofGoalCache` (blake3-keyed) so repeated structures
//! across branches are solved once and reused.

use katgpt_core::and_or::AndOrNode;
use katgpt_core::proof_cache::{GoalResult, ProofGoalCache};
use katgpt_core::traits::ScreeningPruner;

// ── Subgoal ─────────────────────────────────────────────────────

/// Subgoal for DDTree decomposition: a contiguous slice of the marginal sequence.
///
/// Each subgoal covers `[depth_start, depth_end)` in the marginal array.
/// The blake3 hash enables memoization — identical subgoals across branches
/// are solved once and cached.
#[derive(Debug, Clone, Copy)]
pub struct Subgoal {
    /// Start depth in the marginal sequence (inclusive).
    pub depth_start: usize,
    /// End depth (exclusive).
    pub depth_end: usize,
    /// Blake3 hash of canonical encoding (depth_start || depth_end || top-tokens).
    pub hash: [u8; 32],
}

impl Subgoal {
    /// Create a subgoal and compute its blake3 hash from the marginal slice.
    ///
    /// Canonical encoding: `(depth_start: u64 LE || depth_end: u64 LE || top_tokens: [u64 LE])`
    /// This is deterministic — same logical subgoal always produces the same hash.
    pub fn new(depth_start: usize, depth_end: usize, marginals: &[&[f32]]) -> Self {
        let hash = Self::compute_hash(depth_start, depth_end, marginals);
        Self {
            depth_start,
            depth_end,
            hash,
        }
    }

    /// Compute blake3 hash of canonical encoding.
    ///
    /// Encodes depth bounds + argmax token indices for each marginal in range.
    /// Using argmax (not full distribution) keeps the encoding compact and stable.
    #[inline]
    fn compute_hash(depth_start: usize, depth_end: usize, marginals: &[&[f32]]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(depth_start as u64).to_le_bytes());
        hasher.update(&(depth_end as u64).to_le_bytes());
        for d in depth_start..depth_end {
            let top = argmax_or_zero(marginals, d);
            hasher.update(&(top as u64).to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }

    /// Encode subgoal as canonical bytes (re-derivable for cache lookups).
    #[inline]
    fn canonical_bytes(&self, marginals: &[&[f32]]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + (self.depth_end - self.depth_start) * 8);
        buf.extend_from_slice(&(self.depth_start as u64).to_le_bytes());
        buf.extend_from_slice(&(self.depth_end as u64).to_le_bytes());
        for d in self.depth_start..self.depth_end {
            let top = argmax_or_zero(marginals, d);
            buf.extend_from_slice(&(top as u64).to_le_bytes());
        }
        buf
    }
}

// ── AndOrBuilder ─────────────────────────────────────────────────

/// Builds AND-OR tree from marginals using relevance signal as decomposition trigger.
///
/// Low relevance → decompose into subgoals (AND node).
/// High relevance → solve directly (leaf).
/// Root is OR — represents alternative strategies to solve the full sequence.
///
/// # Type Parameters
///
/// - `P`: Any type implementing [`ScreeningPruner`], used to compute per-position relevance.
///
/// # Lifetime
///
/// Borrows the pruner and cache — the builder is scoped to a single decode step.
pub struct AndOrBuilder<'a, P: ScreeningPruner> {
    pruner: &'a P,
    cache: &'a mut ProofGoalCache,
    /// Below this relevance → decompose into subgoals.
    relevance_threshold: f32,
    /// Maximum tree depth (bounds recursion).
    max_depth: usize,
}

/// Default relevance threshold — below 0.3 is considered "uncertain" for decomposition.
pub const DEFAULT_RELEVANCE_THRESHOLD: f32 = 0.3;

/// Default maximum tree depth — prevents pathological deep decomposition.
pub const DEFAULT_MAX_DEPTH: usize = 16;

impl<'a, P: ScreeningPruner> AndOrBuilder<'a, P> {
    /// Create a new builder with default settings.
    pub fn new(pruner: &'a P, cache: &'a mut ProofGoalCache) -> Self {
        Self {
            pruner,
            cache,
            relevance_threshold: DEFAULT_RELEVANCE_THRESHOLD,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    /// Set the relevance threshold for decomposition.
    ///
    /// Lower values → less decomposition (more conservative).
    /// Higher values → more decomposition (more aggressive).
    #[inline]
    pub fn with_relevance_threshold(mut self, threshold: f32) -> Self {
        self.relevance_threshold = threshold;
        self
    }

    /// Set the maximum tree depth.
    #[inline]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Build an AND-OR tree from flat marginals.
    ///
    /// # Algorithm
    ///
    /// 1. Compute relevance at each depth for the top-k tokens.
    /// 2. Find contiguous low-relevance regions → decomposition points.
    /// 3. Create OR root (try alternatives).
    /// 4. For each region: either decompose (AND) or solve directly (leaf).
    /// 5. Memoize solved subgoals in `ProofGoalCache`.
    ///
    /// # Returns
    ///
    /// An `AndOrNode<Subgoal, Vec<usize>>` tree. The root is always OR.
    pub fn build(&mut self, marginals: &[&[f32]]) -> AndOrNode<Subgoal, Vec<usize>> {
        let n = marginals.len();
        if n == 0 {
            let goal = Subgoal::new(0, 0, marginals);
            return AndOrNode::unsolved_leaf(goal);
        }

        // Compute min relevance across top-k tokens at each depth.
        let relevance_profile = self.compute_relevance_profile(marginals);

        // Find decomposition points: contiguous low-relevance regions.
        let regions = self.find_decomposition_regions(&relevance_profile);

        // Build root OR node covering the full marginal sequence.
        let root_goal = Subgoal::new(0, n, marginals);
        let mut root = AndOrNode::or(root_goal);

        match regions.len() {
            0 => {
                // No decomposition needed — entire sequence is high relevance.
                // Solve as a single leaf.
                let subgoal = Subgoal::new(0, n, marginals);
                match self.solve_subgoal(&subgoal, marginals) {
                    Some(path) => root.push_child(AndOrNode::solved_leaf(subgoal, path)),
                    None => root.push_child(AndOrNode::unsolved_leaf(subgoal)),
                }
            }
            1 if regions[0].0 == 0 && regions[0].1 == n => {
                // Entire sequence is low relevance — decompose as AND.
                let and_node = self.build_and_node(&regions[0], marginals);
                root.push_child(and_node);
            }
            _ => {
                // Mixed: decompose low-relevance regions, solve high-relevance directly.
                let full_node = self.build_mixed_tree(0, n, &regions, marginals);
                root.push_child(full_node);
            }
        }

        root
    }

    /// Compute per-depth relevance profile.
    ///
    /// For each depth, computes the minimum relevance across the top-k tokens.
    /// Low min relevance = uncertain = good candidate for decomposition.
    ///
    /// Short-circuits the inner fold once the running min reaches 0.0 (the
    /// smallest possible relevance) — saves up to k-1 pruner calls per depth.
    fn compute_relevance_profile(&self, marginals: &[&[f32]]) -> Vec<f32> {
        const K: usize = 4; // top-k tokens to check
        let mut out = Vec::with_capacity(marginals.len());
        for (depth, marginal) in marginals.iter().enumerate() {
            let top_k_indices = top_k_indices(marginal, K);
            let mut min_rel = f32::INFINITY;
            for &idx in &top_k_indices {
                let r = self.pruner.relevance(depth, idx, &[]);
                if r < min_rel {
                    min_rel = r;
                    if r <= 0.0 {
                        break; // cannot go lower — stop calling the pruner.
                    }
                }
            }
            out.push(min_rel);
        }
        out.shrink_to_fit();
        out
    }

    /// Find contiguous low-relevance regions that should be decomposed.
    ///
    /// Returns a list of `(start, end)` pairs where relevance is below threshold.
    fn find_decomposition_regions(&self, profile: &[f32]) -> Vec<(usize, usize)> {
        let mut regions = Vec::new();
        let mut region_start: Option<usize> = None;

        for (i, &rel) in profile.iter().enumerate() {
            if self.decompose_at(i, rel) {
                // Low relevance — extend or start region.
                if region_start.is_none() {
                    region_start = Some(i);
                }
            } else {
                // High relevance — close any open region.
                if let Some(start) = region_start.take() {
                    regions.push((start, i));
                }
            }
        }

        // Close trailing region.
        if let Some(start) = region_start.take() {
            regions.push((start, profile.len()));
        }

        regions.shrink_to_fit();
        regions
    }

    /// Whether to decompose at this depth given the relevance score.
    ///
    /// Returns `true` when relevance is below threshold (uncertain → decompose).
    #[inline]
    pub fn decompose_at(&self, _depth: usize, relevance: f32) -> bool {
        relevance < self.relevance_threshold
    }

    /// Build an AND node for a contiguous low-relevance region.
    ///
    /// Splits the region into individual depth-subgoals, each solved independently.
    /// All children must succeed (AND semantics).
    fn build_and_node(
        &mut self,
        region: &(usize, usize),
        marginals: &[&[f32]],
    ) -> AndOrNode<Subgoal, Vec<usize>> {
        let (start, end) = *region;
        let goal = Subgoal::new(start, end, marginals);
        let mut node = AndOrNode::and(goal);

        // Create a leaf subgoal per depth.
        for d in start..end {
            let subgoal = Subgoal::new(d, d + 1, marginals);
            match self.solve_subgoal(&subgoal, marginals) {
                Some(path) => node.push_child(AndOrNode::solved_leaf(subgoal, path)),
                None => node.push_child(AndOrNode::unsolved_leaf(subgoal)),
            }
        }

        // Store the combined solution as the sketch.
        let combined = self.solve_subgoal(&Subgoal::new(start, end, marginals), marginals);
        if let Some(path) = combined {
            node.set_sketch(path);
        }

        node
    }

    /// Build a mixed AND-OR tree for regions with both high and low relevance.
    ///
    /// Low-relevance regions become AND children, high-relevance regions become leaves.
    fn build_mixed_tree(
        &mut self,
        seq_start: usize,
        seq_end: usize,
        regions: &[(usize, usize)],
        marginals: &[&[f32]],
    ) -> AndOrNode<Subgoal, Vec<usize>> {
        let goal = Subgoal::new(seq_start, seq_end, marginals);
        let mut node = AndOrNode::and(goal);
        let mut cursor = seq_start;

        for &(r_start, r_end) in regions {
            // High-relevance region before this decomposition.
            if cursor < r_start {
                let subgoal = Subgoal::new(cursor, r_start, marginals);
                match self.solve_subgoal(&subgoal, marginals) {
                    Some(path) => node.push_child(AndOrNode::solved_leaf(subgoal, path)),
                    None => node.push_child(AndOrNode::unsolved_leaf(subgoal)),
                }
            }

            // Low-relevance region — decompose as AND.
            let and_child = self.build_and_node(&(r_start, r_end), marginals);
            node.push_child(and_child);

            cursor = r_end;
        }

        // Trailing high-relevance region.
        if cursor < seq_end {
            let subgoal = Subgoal::new(cursor, seq_end, marginals);
            match self.solve_subgoal(&subgoal, marginals) {
                Some(path) => node.push_child(AndOrNode::solved_leaf(subgoal, path)),
                None => node.push_child(AndOrNode::unsolved_leaf(subgoal)),
            }
        }

        node
    }

    /// Solve a subgoal via greedy argmax, with cache memoization.
    ///
    /// # Algorithm
    ///
    /// 1. Encode subgoal as canonical bytes.
    /// 2. Hash with blake3.
    /// 3. Check `ProofGoalCache` for memoized result.
    /// 4. Cache miss → greedy argmax path, insert into cache.
    ///
    /// # Returns
    ///
    /// `Some(path)` if the subgoal was solved, `None` if marginals are empty/invalid.
    pub fn solve_subgoal(&mut self, subgoal: &Subgoal, marginals: &[&[f32]]) -> Option<Vec<usize>> {
        if subgoal.depth_start >= subgoal.depth_end {
            return None;
        }

        let canonical = subgoal.canonical_bytes(marginals);

        // Check cache for memoized solution.
        match self.cache.peek(&canonical) {
            Some(GoalResult::Proved) => {
                // Cache hit — reconstruct argmax path (deterministic from marginals).
                return Some(greedy_argmax_path(
                    subgoal.depth_start,
                    subgoal.depth_end,
                    marginals,
                ));
            }
            Some(GoalResult::Disproved(_)) | Some(GoalResult::Unknown) => return None,
            None => {}
        }

        // Cache miss — solve via greedy argmax.
        let path = greedy_argmax_path(subgoal.depth_start, subgoal.depth_end, marginals);

        // Store result in cache.
        let result = if path.is_empty() {
            GoalResult::Unknown
        } else {
            GoalResult::Proved
        };
        self.cache.insert(&canonical, result);

        if path.is_empty() { None } else { Some(path) }
    }
}

// ── Free Functions ──────────────────────────────────────────────

/// Greedy argmax path through marginals in `[start, end)`.
///
/// Returns the token index with highest marginal probability at each depth.
#[inline]
fn greedy_argmax_path(start: usize, end: usize, marginals: &[&[f32]]) -> Vec<usize> {
    (start..end).map(|d| argmax_or_zero(marginals, d)).collect()
}

/// Argmax of marginal at depth `d`, or 0 if out of bounds / empty.
#[inline]
fn argmax_or_zero(marginals: &[&[f32]], d: usize) -> usize {
    match marginals.get(d) {
        Some(m) if !m.is_empty() => m
            .iter()
            .enumerate()
            // float_order: a NaN marginal must never win the argmax.
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map_or(0, |(idx, _)| idx),
        _ => 0,
    }
}

/// Top-k indices from a marginal distribution (descending by value).
///
/// Uses a fixed-size insertion-sort buffer (`[usize; K_MAX]`) to avoid the
/// heap allocation that `select_nth_unstable_by` requires. Caller typically
/// wants `k ≤ 4`, so we cap at `K_MAX = 8` and fall back to allocation only
/// when `k > K_MAX`.
#[inline]
fn top_k_indices(marginal: &[f32], k: usize) -> Vec<usize> {
    const K_MAX: usize = 8;
    let k = k.min(marginal.len());
    if k == 0 {
        return Vec::new();
    }

    // Fast path: small k fits in a stack buffer — no heap allocation.
    if k <= K_MAX {
        // Parallel arrays: indices + values, kept sorted descending by value.
        let mut idx_buf = [0usize; K_MAX];
        let mut val_buf = [f32::NEG_INFINITY; K_MAX];
        let mut filled = 0usize;

        for (i, &v) in marginal.iter().enumerate() {
            // Skip if v is smaller than the current k-th largest and buffer is full.
            if filled == k && v <= val_buf[k - 1] {
                continue;
            }
            // Insertion-sort slot: find position.
            let mut pos = filled.min(k);
            while pos > 0 && val_buf[pos - 1] < v {
                if pos < k {
                    idx_buf[pos] = idx_buf[pos - 1];
                    val_buf[pos] = val_buf[pos - 1];
                }
                pos -= 1;
            }
            if pos < k {
                idx_buf[pos] = i;
                val_buf[pos] = v;
                if filled < k {
                    filled += 1;
                }
            }
        }

        let mut out = Vec::with_capacity(k);
        out.extend(idx_buf[..filled].iter().copied());
        return out;
    }

    // Fallback for unusually large k.
    let mut indexed: Vec<(usize, f32)> = marginal.iter().copied().enumerate().collect();
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    indexed.truncate(k);
    indexed.into_iter().map(|(idx, _)| idx).collect()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Constant high-relevance pruner for testing.
    struct HighRelevancePruner;
    impl ScreeningPruner for HighRelevancePruner {
        fn relevance(&self, _: usize, _: usize, _: &[usize]) -> f32 {
            0.9
        }
    }

    /// Constant low-relevance pruner for testing.
    struct LowRelevancePruner;
    impl ScreeningPruner for LowRelevancePruner {
        fn relevance(&self, _: usize, _: usize, _: &[usize]) -> f32 {
            0.1
        }
    }

    /// Pruner that returns low relevance at even depths, high at odd.
    struct AlternatingPruner;
    impl ScreeningPruner for AlternatingPruner {
        fn relevance(&self, depth: usize, _: usize, _: &[usize]) -> f32 {
            match depth % 2 {
                0 => 0.1,
                _ => 0.9,
            }
        }
    }

    fn make_marginals(depths: usize, vocab: usize) -> Vec<Vec<f32>> {
        // Create simple marginals where token `d` has highest probability at depth `d`.
        (0..depths)
            .map(|d| {
                let mut m = vec![0.1f32; vocab];
                if d < vocab {
                    m[d] = 0.9;
                }
                m
            })
            .collect()
    }

    fn as_refs(marginals: &[Vec<f32>]) -> Vec<&[f32]> {
        marginals.iter().map(|m| m.as_slice()).collect()
    }

    // ── Decomposition Trigger Tests ────────────────────────────

    #[test]
    fn test_decompose_at_below_threshold() {
        let mut cache = ProofGoalCache::new();
        let pruner = LowRelevancePruner;
        let builder = AndOrBuilder::new(&pruner, &mut cache);
        assert!(builder.decompose_at(0, 0.1));
        assert!(builder.decompose_at(5, 0.0));
    }

    #[test]
    fn test_decompose_at_above_threshold() {
        let mut cache = ProofGoalCache::new();
        let pruner = HighRelevancePruner;
        let builder = AndOrBuilder::new(&pruner, &mut cache);
        assert!(!builder.decompose_at(0, 0.9));
        assert!(!builder.decompose_at(5, 0.5));
    }

    #[test]
    fn test_decompose_at_custom_threshold() {
        let mut cache = ProofGoalCache::new();
        let pruner = HighRelevancePruner;
        let builder = AndOrBuilder::new(&pruner, &mut cache).with_relevance_threshold(0.95);
        // 0.9 < 0.95 → decompose
        assert!(builder.decompose_at(0, 0.9));
    }

    // ── Build Tests ────────────────────────────────────────────

    #[test]
    fn test_build_empty_marginals() {
        let mut cache = ProofGoalCache::new();
        let pruner = HighRelevancePruner;
        let mut builder = AndOrBuilder::new(&pruner, &mut cache);
        let tree = builder.build(&[]);
        // Should be an unsolved leaf for empty input.
        assert!(!tree.is_solved());
    }

    #[test]
    fn test_build_high_relevance_solves_directly() {
        let mut cache = ProofGoalCache::new();
        let pruner = HighRelevancePruner;
        let mut builder = AndOrBuilder::new(&pruner, &mut cache);
        let marginals = make_marginals(4, 8);
        let refs = as_refs(&marginals);
        let tree = builder.build(&refs);

        // Root is OR.
        assert!(matches!(tree, AndOrNode::Or { .. }));
        assert!(tree.child_count() > 0);

        // Child should be a solved leaf (high relevance → solve directly).
        let child = tree.child(0).unwrap();
        assert!(child.is_solved());
    }

    #[test]
    fn test_build_low_relevance_decomposes() {
        let mut cache = ProofGoalCache::new();
        let pruner = LowRelevancePruner;
        let mut builder = AndOrBuilder::new(&pruner, &mut cache);
        let marginals = make_marginals(4, 8);
        let refs = as_refs(&marginals);
        let tree = builder.build(&refs);

        // Root is OR.
        assert!(matches!(tree, AndOrNode::Or { .. }));
        assert!(tree.child_count() > 0);

        // Child should contain AND decomposition.
        let child = tree.child(0).unwrap();
        let has_and = node_contains_and(child);
        assert!(
            has_and,
            "Expected AND node in decomposition for low relevance"
        );
    }

    #[test]
    fn test_build_mixed_relevance() {
        let mut cache = ProofGoalCache::new();
        let pruner = AlternatingPruner;
        let mut builder = AndOrBuilder::new(&pruner, &mut cache);
        let marginals = make_marginals(4, 8);
        let refs = as_refs(&marginals);
        let tree = builder.build(&refs);

        // Root is OR.
        assert!(matches!(tree, AndOrNode::Or { .. }));

        // Mixed: should have decomposition regions at depths 0, 2.
        let child = tree.child(0).unwrap();
        assert!(child.child_count() > 0);
    }

    // ── Subgoal Solving Tests ──────────────────────────────────

    #[test]
    fn test_solve_subgoal_basic() {
        let mut cache = ProofGoalCache::new();
        let pruner = HighRelevancePruner;
        let mut builder = AndOrBuilder::new(&pruner, &mut cache);
        let marginals = make_marginals(4, 8);
        let refs = as_refs(&marginals);
        let subgoal = Subgoal::new(0, 4, &refs);

        let path = builder.solve_subgoal(&subgoal, &refs);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 4);
        // Token i has highest prob at depth i.
        assert_eq!(path[0], 0);
        assert_eq!(path[1], 1);
    }

    #[test]
    fn test_solve_subgoal_empty_range() {
        let mut cache = ProofGoalCache::new();
        let pruner = HighRelevancePruner;
        let mut builder = AndOrBuilder::new(&pruner, &mut cache);
        let marginals = make_marginals(4, 8);
        let refs = as_refs(&marginals);
        let subgoal = Subgoal::new(2, 2, &refs); // empty range

        let path = builder.solve_subgoal(&subgoal, &refs);
        assert!(path.is_none());
    }

    #[test]
    fn test_solve_subgoal_cache_hit() {
        let mut cache = ProofGoalCache::new();
        let pruner = HighRelevancePruner;
        let marginals = make_marginals(4, 8);
        let refs = as_refs(&marginals);
        let subgoal = Subgoal::new(0, 4, &refs);

        // First solve: inserts into cache (peek + insert path).
        {
            let mut builder = AndOrBuilder::new(&pruner, &mut cache);
            let path1 = builder.solve_subgoal(&subgoal, &refs);
            assert!(path1.is_some());
            assert_eq!(path1.unwrap(), vec![0, 1, 2, 3]);
        }
        assert_eq!(cache.len(), 1); // Entry stored

        // Second solve: cache hit via peek (no re-insertion).
        {
            let mut builder = AndOrBuilder::new(&pruner, &mut cache);
            let path2 = builder.solve_subgoal(&subgoal, &refs);
            assert!(path2.is_some());
            assert_eq!(path2.unwrap(), vec![0, 1, 2, 3]);
        }
        assert_eq!(cache.len(), 1); // No new entry
    }

    // ── Cache Integration Tests ────────────────────────────────

    #[test]
    fn test_cache_populated_after_build() {
        let mut cache = ProofGoalCache::new();
        let pruner = HighRelevancePruner;
        let mut builder = AndOrBuilder::new(&pruner, &mut cache);
        let marginals = make_marginals(4, 8);
        let refs = as_refs(&marginals);
        let _tree = builder.build(&refs);

        // Cache should have entries from subgoal solving.
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cache_reuse_across_builds() {
        let mut cache = ProofGoalCache::new();
        let pruner = HighRelevancePruner;
        let marginals = make_marginals(4, 8);
        let refs = as_refs(&marginals);

        // First build populates cache.
        {
            let mut builder = AndOrBuilder::new(&pruner, &mut cache);
            let _tree1 = builder.build(&refs);
        }
        let entries_after_first = cache.len();
        assert!(
            entries_after_first > 0,
            "Cache should have entries after build"
        );

        // Second build should benefit from cache (same entries, no growth).
        {
            let mut builder = AndOrBuilder::new(&pruner, &mut cache);
            let _tree2 = builder.build(&refs);
        }
        assert_eq!(
            cache.len(),
            entries_after_first,
            "Cache should not grow on repeated build — entries are reused"
        );
    }

    // ── Subgoal Hash Tests ─────────────────────────────────────

    #[test]
    fn test_subgoal_hash_deterministic() {
        let marginals = make_marginals(4, 8);
        let refs = as_refs(&marginals);
        let s1 = Subgoal::new(1, 3, &refs);
        let s2 = Subgoal::new(1, 3, &refs);
        assert_eq!(s1.hash, s2.hash, "Same subgoal should produce same hash");
    }

    #[test]
    fn test_subgoal_hash_different_ranges() {
        let marginals = make_marginals(4, 8);
        let refs = as_refs(&marginals);
        let s1 = Subgoal::new(0, 2, &refs);
        let s2 = Subgoal::new(2, 4, &refs);
        assert_ne!(
            s1.hash, s2.hash,
            "Different ranges should produce different hashes"
        );
    }

    // ── Helper Functions ───────────────────────────────────────

    fn node_contains_and<G, S>(node: &AndOrNode<G, S>) -> bool {
        match node {
            AndOrNode::And { .. } => true,
            AndOrNode::Or { children, .. } => children.iter().any(node_contains_and),
            AndOrNode::Leaf { .. } => false,
        }
    }
}
