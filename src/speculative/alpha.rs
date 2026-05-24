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
    }

    /// Get the current α-target, recomputing if needed.
    pub fn target(&mut self) -> &[HashSet<usize>] {
        if self.cached_target.is_none() {
            self.cached_target = Some(alpha_intersect(&self.current, &self.solutions));
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
}
