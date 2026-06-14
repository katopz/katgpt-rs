//! LDT α-operator: progressive multi-solution supervision target (T3, Plan 088).
//!
//! Distilled from "Lattice Deduction Transformers" (arXiv:2605.08605).
//!
//! ŷ = x ⊓ α({y ∈ Y | y consistent with x})
//!
//! For domains with multiple valid solutions (maze shortest paths,
//! Go joseki variations), this provides a tightening target as
//! search commits to particular branches.
//!
//! All code behind `#[cfg(feature = "lattice_deduction")]`.

use std::collections::HashSet;
use std::sync::Mutex;

use crate::speculative::types::ScreeningPruner;

/// Check if solution `y` is consistent with current state `x`.
///
/// Consistent = every committed position in `x` matches the corresponding
/// position in `y`. Uncommitted positions (`None`) are wildcards.
pub fn is_consistent(current: &[Option<usize>], solution: &[usize]) -> bool {
    current
        .iter()
        .zip(solution.iter())
        .all(|(opt, &val)| opt.is_none_or(|v| v == val))
}

/// LDT α-operator: intersect current state with union of consistent solutions.
///
/// `current` — current candidate state (`Some` = committed, `None` = open)
/// `solutions` — K pre-computed valid solutions
///
/// Returns per-position candidate sets representing the tightest sound target
/// given current commitments. As the search commits tokens, the target narrows.
///
/// # Example
///
/// ```ignore
/// use katgpt_rs::speculative::alpha::alpha_intersect;
///
/// let current = vec![Some(0), None, Some(2)];
/// let solutions = vec![
///     vec![0, 1, 2],
///     vec![0, 3, 2],
///     vec![0, 1, 2],  // duplicate ok — deduped by HashSet
/// ];
/// let alpha = alpha_intersect(&current, &solutions);
/// // alpha[0] = {0}     (committed)
/// // alpha[1] = {1, 3}  (union of consistent solutions)
/// // alpha[2] = {2}     (committed)
/// ```
pub fn alpha_intersect(current: &[Option<usize>], solutions: &[Vec<usize>]) -> Vec<HashSet<usize>> {
    // Filter to only solutions consistent with current commitments
    let consistent: Vec<&Vec<usize>> = solutions
        .iter()
        .filter(|sol| is_consistent(current, sol))
        .collect();

    // α: union of values at each position across consistent solutions
    let mut alpha: Vec<HashSet<usize>> = vec![HashSet::new(); current.len()];
    for sol in &consistent {
        for (i, &val) in sol.iter().enumerate() {
            if i < alpha.len() {
                alpha[i].insert(val);
            }
        }
    }

    // ⊓: intersect with current commitments (committed positions lock to one value)
    for (i, opt) in current.iter().enumerate() {
        if let Some(v) = opt {
            alpha[i].clear();
            alpha[i].insert(*v);
        }
    }

    alpha
}

/// Progressive α-target tracker for iterative search.
///
/// Maintains the current commitment state and recomputes the α-target
/// as tokens are committed. Designed for integration with DDTree expansion:
/// at each step, call [`AlphaTarget::commit`] then [`AlphaTarget::target`]
/// to get the screening signal.
#[derive(Debug, Clone)]
pub struct AlphaTarget {
    /// Current commitment state (`Some` = committed, `None` = open).
    current: Vec<Option<usize>>,
    /// K pre-computed valid solutions.
    solutions: Vec<Vec<usize>>,
    /// Cached α-target (invalidated on commit).
    cached_target: Option<Vec<HashSet<usize>>>,
    /// Last non-empty α-target, cached for stability when no solution remains
    /// consistent (Plan 170 F3, LDT ŷ_prev stabilization).
    cached_prev: Option<Vec<HashSet<usize>>>,
}

impl AlphaTarget {
    /// Create a new α-target tracker.
    ///
    /// - `len` — number of positions (e.g., puzzle size, path length)
    /// - `solutions` — K pre-computed valid solutions
    pub fn new(len: usize, solutions: Vec<Vec<usize>>) -> Self {
        Self {
            current: vec![None; len],
            solutions,
            cached_target: None,
            cached_prev: None,
        }
    }

    /// Commit a value at a position.
    pub fn commit(&mut self, pos: usize, val: usize) {
        if pos < self.current.len() {
            self.current[pos] = Some(val);
            self.cached_target = None; // invalidate cache
        }
    }

    /// Reset commitment at a position (uncommit).
    pub fn uncommit(&mut self, pos: usize) {
        if pos < self.current.len() {
            self.current[pos] = None;
            self.cached_target = None;
        }
    }

    /// Reset all commitments.
    pub fn reset(&mut self) {
        self.current.fill(None);
        self.cached_target = None;
        self.cached_prev = None;
    }

    /// Get the current α-target, recomputing if needed.
    ///
    /// When no solution remains consistent with the current state,
    /// returns the last valid target (LDT ŷ_prev stabilization, F3).
    /// If no previous target exists, returns the intersection result
    /// (which may have empty per-position sets).
    pub fn target(&mut self) -> &[HashSet<usize>] {
        if self.cached_target.is_none() {
            let new_target = alpha_intersect(&self.current, &self.solutions);

            // Single pass: determine whether the target has any non-empty set.
            // The original called `.iter().any(|s| !s.is_empty())` and then
            // `.iter().all(|s| s.is_empty())` — two full iterations. Since
            // `any(!empty)` is the exact logical inverse of `all(empty)`, we
            // only need a single short-circuiting scan.
            let has_content = new_target.iter().any(|s| !s.is_empty());

            // If the target has non-trivial content, cache it as ŷ_prev.
            if has_content {
                self.cached_prev = Some(new_target.clone());
            }

            // If the new target is entirely empty but we have a previous target,
            // use the previous target for stability.
            self.cached_target = Some(if has_content {
                new_target
            } else {
                match &self.cached_prev {
                    Some(prev) => prev.clone(),
                    None => new_target,
                }
            });
        }
        self.cached_target.as_ref().unwrap()
    }

    /// Check if a token at a position is consistent with the α-target.
    ///
    /// Returns `true` if the token appears in the target set for that position,
    /// meaning it's consistent with at least one remaining valid solution.
    pub fn is_allowed(&mut self, pos: usize, token: usize) -> bool {
        let target = self.target();
        target.get(pos).is_some_and(|set| set.contains(&token))
    }

    /// Count remaining consistent solutions.
    pub fn remaining_solutions(&self) -> usize {
        self.solutions
            .iter()
            .filter(|sol| is_consistent(&self.current, sol))
            .count()
    }

    /// Get the current commitment state.
    pub fn current(&self) -> &[Option<usize>] {
        &self.current
    }

    /// Get the number of positions.
    pub fn len(&self) -> usize {
        self.current.len()
    }

    /// Check if the tracker has no positions.
    pub fn is_empty(&self) -> bool {
        self.current.is_empty()
    }
}

// ── F1: AlphaScreeningPruner (Plan 170) ──────────────────────────

/// α-operator as ScreeningPruner — sound multi-solution pruning (Plan 170 F1).
///
/// Bridges [`AlphaTarget`] to the [`ScreeningPruner`] trait, providing
/// **sound by construction** multi-solution pruning:
/// - Token in α-target → relevance 1.0 (allowed)
/// - Token not in α-target → relevance 0.0 (pruned)
///
/// The pruner never eliminates a token that appears in any solution still
/// consistent with the current search state. As the search commits tokens
/// (via [`AlphaTarget::commit`]), the α-target narrows and pruning tightens.
///
/// Uses `Mutex` for thread-safe interior mutability because
/// `ScreeningPruner: Send + Sync` but `AlphaTarget::is_allowed` may lazily
/// recompute the cache. The Mutex contention is negligible because DDTree
/// expansion is typically single-threaded per pruner instance.
pub struct AlphaScreeningPruner {
    target: Mutex<AlphaTarget>,
}

impl AlphaScreeningPruner {
    /// Create a new α-screening pruner from an existing AlphaTarget.
    pub fn new(target: AlphaTarget) -> Self {
        Self {
            target: Mutex::new(target),
        }
    }

    /// Create a new α-screening pruner with fresh (uncommitted) state.
    pub fn with_solutions(len: usize, solutions: Vec<Vec<usize>>) -> Self {
        Self {
            target: Mutex::new(AlphaTarget::new(len, solutions)),
        }
    }

    /// Access the underlying AlphaTarget for commit/uncommit operations.
    pub fn target(&self) -> &Mutex<AlphaTarget> {
        &self.target
    }
}

impl ScreeningPruner for AlphaScreeningPruner {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        if self.target.lock().unwrap().is_allowed(depth, token_idx) {
            1.0
        } else {
            0.0
        }
    }
}

// ── F2: ConflictClauseDB (Plan 170) ──────────────────────────────

/// A learned conflict clause — a set of (position, value) commitments
/// known to cause conflict (Plan 170 F2).
///
/// A state violates this clause if ALL its commitments are present.
/// This is the unit propagation rule from CDCL SAT solvers, adapted
/// for DDTree search.
pub struct ConflictClause {
    /// The commitment pattern that caused the conflict.
    commitments: HashSet<(usize, usize)>,
}

impl ConflictClause {
    /// Check if the given commitments violate this clause.
    ///
    /// Returns `true` if ALL commitments in this clause are present
    /// in the given set — meaning the search is about to repeat a
    /// known-bad path.
    ///
    /// Hot path: iterate the (typically small) slice and probe the
    /// HashSet, rather than iterating the HashSet and probing the slice.
    /// The slice is usually DDTree's current commitment chain (≤ a few
    /// entries); the clause has at least 1 entry. Either ordering works
    /// but iterating the smaller side keeps the inner `contains` O(1).
    pub fn is_violated_by(&self, commitments: &[(usize, usize)]) -> bool {
        // Quick reject: if the slice has fewer entries than this clause,
        // it cannot possibly contain all of them.
        if commitments.len() < self.commitments.len() {
            return false;
        }
        self.commitments
            .iter()
            .all(|c| commitments.contains(c))
    }

    /// Number of commitments in this clause.
    pub fn len(&self) -> usize {
        self.commitments.len()
    }

    /// Check if this clause has no commitments.
    pub fn is_empty(&self) -> bool {
        self.commitments.is_empty()
    }
}

/// Database of learned conflict clauses for DDTree search acceleration
/// (Plan 170 F2, CDCL-inspired).
///
/// When a DDTree branch is flagged as conflicted by [`ConflictDetector`],
/// extract the commitment pattern and learn a clause. Future expansions
/// check against all clauses before exploring — skipping branches known
/// to lead to conflicts.
///
/// Bounded by `max_clauses` to prevent unbounded growth. When full,
/// the oldest clause is evicted (FIFO).
///
/// # Performance
///
/// Clause check: O(k × c) where k = number of clauses, c = avg clause size.
/// With default max_clauses = 64 and avg clause size = 4: ~256 HashSet lookups ≈ 200ns.
/// Well under the 1µs hot-path budget.
pub struct ConflictClauseDB {
    clauses: Vec<ConflictClause>,
    max_clauses: usize,
}

impl Default for ConflictClauseDB {
    fn default() -> Self {
        Self {
            clauses: Vec::new(),
            max_clauses: 64,
        }
    }
}

impl ConflictClauseDB {
    /// Create a new clause database with bounded capacity.
    pub fn new(max_clauses: usize) -> Self {
        Self {
            clauses: Vec::with_capacity(max_clauses),
            max_clauses,
        }
    }

    /// Learn a new clause from a conflict.
    ///
    /// `commitments` — the (position, value) pairs that led to the conflict.
    /// Only stores clauses with ≥ 1 commitment (trivially empty clauses
    /// provide no pruning value).
    pub fn learn(&mut self, commitments: HashSet<(usize, usize)>) {
        if commitments.is_empty() {
            return;
        }
        if self.clauses.len() >= self.max_clauses {
            self.clauses.remove(0); // FIFO eviction
        }
        self.clauses.push(ConflictClause { commitments });
    }

    /// Check if any learned clause is violated by the given commitments.
    ///
    /// Returns `true` if the search should skip this branch because
    /// it's a superset of a known-conflicting commitment pattern.
    pub fn is_violated(&self, commitments: &[(usize, usize)]) -> bool {
        self.clauses
            .iter()
            .any(|clause| clause.is_violated_by(commitments))
    }

    /// Number of learned clauses.
    pub fn len(&self) -> usize {
        self.clauses.len()
    }

    /// Check if the database has no learned clauses.
    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    /// Clear all learned clauses.
    pub fn clear(&mut self) {
        self.clauses.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_consistent_empty_current() {
        let current: Vec<Option<usize>> = vec![None, None, None];
        let solution = vec![1, 2, 3];
        assert!(is_consistent(&current, &solution));
    }

    #[test]
    fn test_is_consistent_all_committed_matching() {
        let current = vec![Some(1), Some(2), Some(3)];
        let solution = vec![1, 2, 3];
        assert!(is_consistent(&current, &solution));
    }

    #[test]
    fn test_is_consistent_partial_commit_matching() {
        let current = vec![Some(1), None, Some(3)];
        let solution = vec![1, 2, 3];
        assert!(is_consistent(&current, &solution));
    }

    #[test]
    fn test_is_consistent_mismatch() {
        let current = vec![Some(1), Some(9), Some(3)];
        let solution = vec![1, 2, 3];
        assert!(!is_consistent(&current, &solution));
    }

    #[test]
    fn test_alpha_intersect_empty_current() {
        let current: Vec<Option<usize>> = vec![None, None];
        let solutions = vec![vec![0, 1], vec![0, 2], vec![3, 1]];
        let alpha = alpha_intersect(&current, &solutions);
        assert_eq!(alpha.len(), 2);
        assert_eq!(alpha[0], HashSet::from([0, 3]));
        assert_eq!(alpha[1], HashSet::from([1, 2]));
    }

    #[test]
    fn test_alpha_intersect_with_commitment() {
        let current = vec![Some(0), None];
        let solutions = vec![vec![0, 1], vec![0, 2], vec![3, 1]];
        // Only [0,1] and [0,2] are consistent with commitment at pos 0
        let alpha = alpha_intersect(&current, &solutions);
        assert_eq!(alpha[0], HashSet::from([0])); // committed
        assert_eq!(alpha[1], HashSet::from([1, 2])); // union of consistent
    }

    #[test]
    fn test_alpha_intersect_no_consistent_solutions() {
        let current = vec![Some(9)];
        let solutions = vec![vec![0], vec![1]];
        let alpha = alpha_intersect(&current, &solutions);
        assert_eq!(alpha[0], HashSet::from([9])); // only the commitment
    }

    #[test]
    fn test_alpha_intersect_full_commitment() {
        let current = vec![Some(0), Some(1)];
        let solutions = vec![vec![0, 1], vec![0, 2]];
        let alpha = alpha_intersect(&current, &solutions);
        assert_eq!(alpha[0], HashSet::from([0]));
        assert_eq!(alpha[1], HashSet::from([1])); // committed, not union
    }

    #[test]
    fn test_alpha_target_commit_and_query() {
        let solutions = vec![vec![0, 1, 2], vec![0, 3, 2], vec![4, 1, 5]];
        let mut target = AlphaTarget::new(3, solutions);

        // Initially all positions are open
        assert!(target.is_allowed(0, 0));
        assert!(target.is_allowed(0, 4));

        // Commit position 0 to value 0
        target.commit(0, 0);
        assert!(target.is_allowed(0, 0));
        assert!(!target.is_allowed(0, 4)); // [4,1,5] eliminated
        assert!(target.is_allowed(1, 1));
        assert!(target.is_allowed(1, 3));
    }

    #[test]
    fn test_alpha_target_remaining_solutions() {
        let solutions = vec![vec![0, 1], vec![0, 2], vec![3, 1]];
        let mut target = AlphaTarget::new(2, solutions);
        assert_eq!(target.remaining_solutions(), 3);

        target.commit(0, 0);
        assert_eq!(target.remaining_solutions(), 2); // [0,1] and [0,2]

        target.commit(1, 1);
        assert_eq!(target.remaining_solutions(), 1); // only [0,1]
    }

    #[test]
    fn test_alpha_target_reset() {
        let solutions = vec![vec![0, 1], vec![0, 2]];
        let mut target = AlphaTarget::new(2, solutions);

        target.commit(0, 0);
        assert_eq!(target.remaining_solutions(), 2);

        target.reset();
        assert_eq!(target.remaining_solutions(), 2);
        assert_eq!(target.current(), &[None, None]);
    }

    #[test]
    fn test_alpha_target_uncommit() {
        let solutions = vec![vec![0, 1], vec![0, 2], vec![3, 1]];
        let mut target = AlphaTarget::new(2, solutions);

        target.commit(0, 0);
        assert_eq!(target.remaining_solutions(), 2);

        target.uncommit(0);
        assert_eq!(target.remaining_solutions(), 3);
    }

    #[test]
    fn test_alpha_target_len_and_empty() {
        let solutions: Vec<Vec<usize>> = vec![];
        let target = AlphaTarget::new(0, solutions);
        assert!(target.is_empty());
        assert_eq!(target.len(), 0);
    }

    // ── F1: AlphaScreeningPruner tests (Plan 170) ──────────────

    #[test]
    fn test_alpha_screening_pruner_basic() {
        let solutions = vec![vec![0, 1], vec![0, 2], vec![3, 1]];
        let target = AlphaTarget::new(2, solutions);
        let pruner = AlphaScreeningPruner::new(target);

        // At depth 0, tokens 0 and 3 are in the α-target
        assert_eq!(pruner.relevance(0, 0, &[]), 1.0); // in [0,1] or [0,2]
        assert_eq!(pruner.relevance(0, 3, &[]), 1.0); // in [3,1]
        assert_eq!(pruner.relevance(0, 9, &[]), 0.0); // not in any solution

        // At depth 1, tokens 1 and 2 are in the α-target
        assert_eq!(pruner.relevance(1, 1, &[]), 1.0); // in [0,1] and [3,1]
        assert_eq!(pruner.relevance(1, 2, &[]), 1.0); // in [0,2]
        assert_eq!(pruner.relevance(1, 9, &[]), 0.0); // not in any solution
    }

    #[test]
    fn test_alpha_screening_pruner_after_commit() {
        let solutions = vec![vec![0, 1], vec![0, 2], vec![3, 1]];
        let target = AlphaTarget::new(2, solutions);
        let pruner = AlphaScreeningPruner::new(target);

        // Commit position 0 to value 0
        pruner.target().lock().unwrap().commit(0, 0);

        // Token 3 should now be pruned at depth 0 (only [0,1] and [0,2] survive)
        assert_eq!(pruner.relevance(0, 0, &[]), 1.0); // still allowed
        assert_eq!(pruner.relevance(0, 3, &[]), 0.0); // eliminated by commit

        // Depth 1 still has tokens 1 and 2
        assert_eq!(pruner.relevance(1, 1, &[]), 1.0);
        assert_eq!(pruner.relevance(1, 2, &[]), 1.0);
    }

    // ── F2: ConflictClauseDB tests (Plan 170) ─────────────────

    #[test]
    fn test_conflict_clause_basic() {
        let mut db = ConflictClauseDB::new(4);
        assert!(db.is_empty());

        // Learn a clause: committing (0, 5) AND (1, 3) causes conflict
        let clause: HashSet<(usize, usize)> = HashSet::from([(0, 5), (1, 3)]);
        db.learn(clause);
        assert_eq!(db.len(), 1);

        // Violates: both commitments present
        assert!(db.is_violated(&[(0, 5), (1, 3)]));
        assert!(db.is_violated(&[(0, 5), (1, 3), (2, 7)])); // superset

        // Does not violate: only one commitment present
        assert!(!db.is_violated(&[(0, 5)]));
        assert!(!db.is_violated(&[(1, 3)]));
        assert!(!db.is_violated(&[(0, 1), (1, 2)])); // different values
    }

    #[test]
    fn test_conflict_clause_max_capacity() {
        let mut db = ConflictClauseDB::new(2);

        db.learn(HashSet::from([(0, 0)]));
        db.learn(HashSet::from([(0, 1)]));
        assert_eq!(db.len(), 2);

        // Adding a third clause evicts the first (FIFO)
        db.learn(HashSet::from([(0, 2)]));
        assert_eq!(db.len(), 2);

        // Clause (0,0) should be evicted
        assert!(!db.is_violated(&[(0, 0)])); // no longer known
        assert!(db.is_violated(&[(0, 1)])); // still known
        assert!(db.is_violated(&[(0, 2)])); // just added
    }

    #[test]
    fn test_conflict_clause_empty_commitment() {
        let mut db = ConflictClauseDB::default();
        db.learn(HashSet::new()); // empty clause — no pruning value
        assert!(db.is_empty()); // not stored
    }

    #[test]
    fn test_conflict_clause_clear() {
        let mut db = ConflictClauseDB::default();
        db.learn(HashSet::from([(0, 0)]));
        assert_eq!(db.len(), 1);
        db.clear();
        assert!(db.is_empty());
    }

    // ── F3: Cached ŷ_prev tests (Plan 170) ────────────────────

    #[test]
    fn test_cached_prev_after_conflict() {
        let solutions = vec![vec![0, 1], vec![0, 2]];
        let mut target = AlphaTarget::new(2, solutions);

        // Get initial target (both solutions alive)
        let t0 = target.target().to_vec();
        assert_eq!(t0[0], HashSet::from([0]));
        assert_eq!(t0[1], HashSet::from([1, 2]));

        // Commit to a consistent value
        target.commit(0, 0);
        let t1 = target.target().to_vec();
        assert_eq!(t1[1], HashSet::from([1, 2])); // narrowed

        // Commit to an INCONSISTENT value (no solution survives)
        target.commit(1, 99); // not in any solution
        let t2 = target.target();

        // F3: Should return the cached previous target, not empty
        assert!(!t2[0].is_empty()); // has the cached prev
        assert!(!t2[1].is_empty()); // has the cached prev
    }
}
