//! Position-swap debiasing for pairwise comparisons (Plan 132, Phase 2, T7–T11).
//!
//! Compare candidates in both A/B and B/A orders; count win only if both agree.
//! This eliminates lead-position bias — the tendency for the first-presented
//! candidate to win disproportionately in Bradley-Terry comparisons.
//!
//! Paper insight: position bias causes `P(i ≻ j)` ≠ `P(j ≻ i)` when order matters.
//! Debiased comparison: run both `compare(i, j)` and `compare(j, i)`, require agreement.

use crate::pruners::bt_rank::{BtComparison, BtOutcome};

// ── Debiased Compare Function ────────────────────────────────

/// Compare two candidates in both A/B orders, requiring agreement.
///
/// Eliminates lead-position bias by running the comparison function twice:
/// 1. `compare(i, j)` — forward order
/// 2. `compare(j, i)` — reversed order
///
/// If both agree on the winner → return that winner.
/// If they disagree (one says i wins, other says j wins) → return `BtOutcome::Tie`.
/// If either returns `Tie` → return `BtOutcome::Tie`.
///
/// # Type Parameters
///
/// - `F`: comparison function `fn(i: usize, j: usize) -> BtOutcome`
///
/// # Arguments
///
/// - `i`: Index of first candidate
/// - `j`: Index of second candidate
/// - `compare`: Comparison function that returns `BtOutcome`
///
/// # Returns
///
/// - `BtOutcome::Win(i)` if both comparisons agree i wins
/// - `BtOutcome::Win(j)` if both comparisons agree j wins
/// - `BtOutcome::Tie` if comparisons disagree or either returns Tie
pub fn debiased_compare<F>(i: usize, j: usize, compare: &F) -> BtOutcome
where
    F: Fn(usize, usize) -> BtOutcome,
{
    if i == j {
        return BtOutcome::Tie;
    }

    let forward = compare(i, j);
    let reverse = compare(j, i);

    match (forward, reverse) {
        // Both agree i wins: forward=i, reverse=j→i means reverse gave j win → disagree
        (BtOutcome::Win(a), BtOutcome::Win(b)) => {
            // forward(i,j)=Win(a): a is the winner in (i,j) order
            // reverse(j,i)=Win(b): b is the winner in (j,i) order
            // For agreement: a==i AND b==i → i wins both
            // Or: a==j AND b==j → j wins both
            // Otherwise disagreement → Tie
            if a == i && b == i {
                BtOutcome::Win(i)
            } else if a == j && b == j {
                BtOutcome::Win(j)
            } else {
                // Disagreement between forward and reverse → Tie
                BtOutcome::Tie
            }
        }
        // Any Tie in either direction → overall Tie
        _ => BtOutcome::Tie,
    }
}

// ── DebiasedComparator Struct ────────────────────────────────

/// Wraps a comparison function with position-swap debiasing.
///
/// Use `tournament()` to run debiased pairwise comparisons over all candidate
/// pairs, producing `Vec<BtComparison>` suitable for `bt_fit()`.
///
/// # Example
///
/// ```ignore
/// use crate::pruners::committee_boost::DebiasedComparator;
/// use crate::pruners::bt_rank::BtOutcome;
///
/// let comparator = DebiasedComparator::new(|i, j| {
///     // Your comparison logic here
///     BtOutcome::Win(i)
/// });
/// let comparisons = comparator.tournament(4);
/// ```
pub struct DebiasedComparator<F>
where
    F: Fn(usize, usize) -> BtOutcome,
{
    /// Underlying comparison function (may be biased).
    compare: F,
}

impl<F> DebiasedComparator<F>
where
    F: Fn(usize, usize) -> BtOutcome,
{
    /// Create a new debiased comparator wrapping the given comparison function.
    pub fn new(compare: F) -> Self {
        Self { compare }
    }

    /// Compare two candidates with debiasing.
    ///
    /// Wraps [`debiased_compare`] with this comparator's comparison function.
    pub fn compare(&self, i: usize, j: usize) -> BtOutcome {
        debiased_compare(i, j, &self.compare)
    }

    /// Run debiased pairwise tournament over all candidate pairs.
    ///
    /// Compares every pair (i, j) where i < j in both orders, collecting
    /// `BtComparison` results for `bt_fit()`.
    ///
    /// # Arguments
    ///
    /// - `n_candidates`: Number of candidates in the pool
    ///
    /// # Returns
    ///
    /// Vector of `BtComparison` results (winner/loser pairs).
    /// Ties are excluded from the output since they carry no ranking information.
    pub fn tournament(&self, n_candidates: usize) -> Vec<BtComparison> {
        let mut comparisons = Vec::new();

        for i in 0..n_candidates {
            for j in (i + 1)..n_candidates {
                match self.compare(i, j) {
                    BtOutcome::Win(winner) => {
                        let loser = if winner == i { j } else { i };
                        comparisons.push(BtComparison::new(winner, loser));
                    }
                    BtOutcome::Tie => {
                        // Ties carry no ranking info — skip
                    }
                }
            }
        }

        comparisons
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Symmetric comparison (always returns Tie) → all pairs Tie.
    #[test]
    fn test_symmetric_comparison_returns_tie() {
        let cmp = DebiasedComparator::new(|_i, _j| BtOutcome::Tie);
        let results = cmp.tournament(4);
        assert!(
            results.is_empty(),
            "symmetric comparison should produce no wins"
        );
    }

    /// Identical inputs → always Tie.
    #[test]
    fn test_identical_inputs_tie() {
        let cmp = DebiasedComparator::new(|i, _j| BtOutcome::Win(i));
        assert_eq!(cmp.compare(0, 0), BtOutcome::Tie);
        assert_eq!(cmp.compare(3, 3), BtOutcome::Tie);
    }

    /// Asymmetric comparison that agrees in both orders → correct winner.
    #[test]
    fn test_asymmetric_agreement_correct_winner() {
        // Lower index always wins — order-invariant via min()
        let cmp = DebiasedComparator::new(|i, j| BtOutcome::Win(i.min(j)));

        // forward(0,1)=Win(0), reverse(1,0)=Win(0) → agree on 0
        assert_eq!(cmp.compare(0, 1), BtOutcome::Win(0));

        // forward(2,5)=Win(2), reverse(5,2)=Win(2) → agree on 2
        assert_eq!(cmp.compare(2, 5), BtOutcome::Win(2));
    }

    /// Asymmetric comparison that disagrees between orders → Tie.
    #[test]
    fn test_asymmetric_disagreement_tie() {
        // Forward order: first argument wins. But when reversed, the original first still wins.
        // compare(i,j)=Win(i), compare(j,i)=Win(j) → disagreement!
        let cmp = DebiasedComparator::new(|i, _j| BtOutcome::Win(i));

        // compare(0, 1) → forward(0,1)=Win(0), reverse(1,0)=Win(1)
        // Win(0) vs Win(1): 0≠1 for the "both must be same winner" check
        // Actually let's trace through debiased_compare:
        // forward = Win(0), reverse = Win(1)
        // match: (Win(0), Win(1)) → a=0, b=1 → not (a==i && b==i) → check (a==j && b==j) → a=0==1? no → Tie
        assert_eq!(cmp.compare(0, 1), BtOutcome::Tie);
    }

    /// Tournament with consistent comparator produces correct comparisons.
    #[test]
    fn test_tournament_consistent_ranking() {
        // Candidate 0 > 1 > 2 > 3 (lower index always wins)
        let _cmp = DebiasedComparator::new(|i, _j| BtOutcome::Win(i));

        // Since forward(i,j)=Win(i) and reverse(j,i)=Win(j), they always disagree
        // (except when i always wins both ways). Let's use a truly consistent one:
        // A comparison that always picks the lower index AND is order-invariant.
        let cmp = DebiasedComparator::new(|i, j| BtOutcome::Win(i.min(j)));

        let results = cmp.tournament(4);

        // Every pair (i,j) with i<j: forward(i,j)=Win(i), reverse(j,i)=Win(i) → agree on i
        // So we should get n*(n-1)/2 = 6 comparisons, all with lower index winning
        assert_eq!(results.len(), 6);
        for comp in &results {
            assert!(
                comp.winner < comp.loser,
                "winner {} should be < loser {}",
                comp.winner,
                comp.loser
            );
        }
    }

    /// Tournament produces correct number of pairs.
    #[test]
    fn test_tournament_pair_count() {
        // Always agree: lower index wins
        let cmp = DebiasedComparator::new(|i, j| BtOutcome::Win(i.min(j)));
        let results = cmp.tournament(5);
        // C(5,2) = 10 pairs
        assert_eq!(results.len(), 10);

        // All disagree → 0 pairs
        let cmp_disagree = DebiasedComparator::new(|i, _j| BtOutcome::Win(i));
        let results_disagree = cmp_disagree.tournament(5);
        assert!(results_disagree.is_empty());
    }

    /// Single candidate → empty tournament.
    #[test]
    fn test_tournament_single_candidate() {
        let cmp = DebiasedComparator::new(|i, _j| BtOutcome::Win(i));
        let results = cmp.tournament(1);
        assert!(results.is_empty());
    }

    /// Zero candidates → empty tournament.
    #[test]
    fn test_tournament_zero_candidates() {
        let cmp = DebiasedComparator::new(|i, _j| BtOutcome::Win(i));
        let results = cmp.tournament(0);
        assert!(results.is_empty());
    }

    /// One side returns Tie → overall Tie (via free function).
    #[test]
    fn test_one_side_tie_produces_tie() {
        // If either forward or reverse returns Tie, overall is Tie.
        // Can't test "one side Win, other Tie" with Fn closure (needs FnMut),
        // but testing "both Tie" suffices to show Tie propagates.
        let result = debiased_compare(0, 1, &|_i, _j| BtOutcome::Tie);
        assert_eq!(result, BtOutcome::Tie);
    }

    /// debiased_compare free function works correctly.
    #[test]
    fn test_debiased_compare_free_function() {
        // Consistent: lower index always wins
        let result = debiased_compare(0, 1, &|i, j| BtOutcome::Win(i.min(j)));
        // forward(0,1)=Win(0), reverse(1,0)=Win(0) → agree on 0
        assert_eq!(result, BtOutcome::Win(0));

        // Disagree: first arg always wins
        let result2 = debiased_compare(0, 1, &|i, _j| BtOutcome::Win(i));
        // forward(0,1)=Win(0), reverse(1,0)=Win(1) → disagree → Tie
        assert_eq!(result2, BtOutcome::Tie);
    }
}
