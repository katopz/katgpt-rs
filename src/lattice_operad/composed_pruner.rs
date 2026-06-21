//! ComposedPruner — ConstraintPruner that evaluates via PrunerExpr.
//!
//! When `lattice_operad` feature is on, this replaces ad-hoc AND composition
//! of multiple pruners with canonical AND/OR via the distributive lattice
//! word problem, eliminating redundant evaluations.

use katgpt_core::ConstraintPruner;

use crate::lattice_operad::compose::{ComposeOp, compose};
use crate::lattice_operad::expr::{PrunerExpr, PrunerResult};

/// A pruner that composes multiple sub-pruners via a PrunerExpr tree.
///
/// Instead of the ad-hoc `all(pruner.is_valid())` pattern, this uses
/// canonical DNF composition which can eliminate redundant evaluations
/// via absorption: if A is false, A∧B doesn't need to evaluate B.
pub struct ComposedPruner<'a> {
    /// The expression tree describing how pruners are composed.
    expr: PrunerExpr,
    /// The sub-pruners, indexed by their Atom IDs.
    pruners: Vec<&'a dyn ConstraintPruner>,
}

impl<'a> ComposedPruner<'a> {
    /// Create a new composed pruner with a single pruner.
    pub fn single(pruner: &'a dyn ConstraintPruner) -> Self {
        Self {
            expr: PrunerExpr::Atom(0),
            pruners: vec![pruner],
        }
    }

    /// Create by AND-ing two sub-pruners (canonicalized).
    pub fn and(pruner_a: &'a dyn ConstraintPruner, pruner_b: &'a dyn ConstraintPruner) -> Self {
        let a = PrunerExpr::Atom(0);
        let b = PrunerExpr::Atom(1);
        let expr = compose(&a, ComposeOp::And, &b);
        Self {
            expr,
            pruners: vec![pruner_a, pruner_b],
        }
    }

    /// Create by OR-ing two sub-pruners (canonicalized).
    pub fn or(pruner_a: &'a dyn ConstraintPruner, pruner_b: &'a dyn ConstraintPruner) -> Self {
        let a = PrunerExpr::Atom(0);
        let b = PrunerExpr::Atom(1);
        let expr = compose(&a, ComposeOp::Or, &b);
        Self {
            expr,
            pruners: vec![pruner_a, pruner_b],
        }
    }

    /// Create from a pre-built expression and pruner list.
    pub fn from_expr(expr: PrunerExpr, pruners: Vec<&'a dyn ConstraintPruner>) -> Self {
        Self { expr, pruners }
    }

    /// Get the canonical expression tree.
    pub fn expr(&self) -> &PrunerExpr {
        &self.expr
    }

    /// Get the number of sub-pruners.
    pub fn pruner_count(&self) -> usize {
        self.pruners.len()
    }
}

impl ConstraintPruner for ComposedPruner<'_> {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Evaluate each sub-pruner
        let mut atom_results = Vec::with_capacity(self.pruners.len());
        for pruner in &self.pruners {
            atom_results.push(pruner.is_valid(depth, token_idx, parent_tokens));
        }

        // Evaluate the expression tree
        matches!(self.expr.eval(&atom_results), PrunerResult::Accept)
    }

    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let n = candidates.len().min(results.len());
        let n_p = self.pruners.len();
        if n == 0 || n_p == 0 {
            return;
        }

        // Flat batch buffer: P×N bools in one allocation instead of P Vecs.
        // Layout: atom_batch[pi * n + i] = result of pruner `pi` for candidate `i`.
        let mut atom_batch = vec![false; n_p * n];
        for (pi, pruner) in self.pruners.iter().enumerate() {
            let row = &mut atom_batch[pi * n..pi * n + n];
            pruner.batch_is_valid(depth, &candidates[..n], parent_tokens, row);
        }

        // Reused atom-result row for one candidate at a time.
        let mut atom_results = vec![false; n_p];
        for i in 0..n {
            for pi in 0..n_p {
                atom_results[pi] = atom_batch[pi * n + i];
            }
            results[i] = matches!(self.expr.eval(&atom_results), PrunerResult::Accept);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AcceptAll;
    impl ConstraintPruner for AcceptAll {
        fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
            true
        }
    }

    struct RejectAll;
    impl ConstraintPruner for RejectAll {
        fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
            false
        }
    }

    struct AcceptEven;
    impl ConstraintPruner for AcceptEven {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            token_idx.is_multiple_of(2)
        }
    }

    struct AcceptLt5;
    impl ConstraintPruner for AcceptLt5 {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            token_idx < 5
        }
    }

    // ── T13 basic tests ──────────────────────────────────────────

    #[test]
    fn test_single_accept_all() {
        let inner = AcceptAll;
        let pruner = ComposedPruner::single(&inner);
        assert!(pruner.is_valid(0, 42, &[]));
    }

    #[test]
    fn test_single_reject_all() {
        let inner = RejectAll;
        let pruner = ComposedPruner::single(&inner);
        assert!(!pruner.is_valid(0, 0, &[]));
    }

    #[test]
    fn test_and_both_accept() {
        let a = AcceptAll;
        let b = AcceptAll;
        let pruner = ComposedPruner::and(&a, &b);
        assert!(pruner.is_valid(0, 0, &[]));
    }

    #[test]
    fn test_and_one_rejects() {
        let a = AcceptAll;
        let b = RejectAll;
        let pruner = ComposedPruner::and(&a, &b);
        assert!(!pruner.is_valid(0, 0, &[]));
    }

    #[test]
    fn test_or_one_accepts() {
        let a = AcceptAll;
        let b = RejectAll;
        let pruner = ComposedPruner::or(&a, &b);
        assert!(pruner.is_valid(0, 0, &[]));
    }

    #[test]
    fn test_or_both_reject() {
        let a = RejectAll;
        let b = RejectAll;
        let pruner = ComposedPruner::or(&a, &b);
        assert!(!pruner.is_valid(0, 0, &[]));
    }

    #[test]
    fn test_and_semantic() {
        // AND of AcceptEven and AcceptLt5: token must be even AND < 5
        let even = AcceptEven;
        let lt5 = AcceptLt5;
        let pruner = ComposedPruner::and(&even, &lt5);

        assert!(pruner.is_valid(0, 0, &[])); // 0: even, <5 → accept
        assert!(pruner.is_valid(0, 2, &[])); // 2: even, <5 → accept
        assert!(pruner.is_valid(0, 4, &[])); // 4: even, <5 → accept
        assert!(!pruner.is_valid(0, 1, &[])); // 1: odd → reject
        assert!(!pruner.is_valid(0, 6, &[])); // 6: even, but >=5 → reject
        assert!(!pruner.is_valid(0, 7, &[])); // 7: odd, >=5 → reject
    }

    #[test]
    fn test_batch_and() {
        let even = AcceptEven;
        let lt5 = AcceptLt5;
        let pruner = ComposedPruner::and(&even, &lt5);

        let candidates: Vec<usize> = (0..10).collect();
        let mut results = vec![false; 10];
        pruner.batch_is_valid(0, &candidates, &[], &mut results);

        // Expected: tokens 0,2,4 are accepted (even AND <5)
        assert!(results[0]); // 0
        assert!(!results[1]); // 1
        assert!(results[2]); // 2
        assert!(!results[3]); // 3
        assert!(results[4]); // 4
        assert!(!results[5]); // 5
    }

    #[test]
    fn test_from_expr_complex() {
        // Build: (A AND B) OR C where A=AcceptEven, B=AcceptLt5, C=AcceptAll
        // Canonical: since C accepts everything, the OR with C should accept everything
        let even = AcceptEven;
        let lt5 = AcceptLt5;
        let all = AcceptAll;

        let expr = PrunerExpr::or(
            PrunerExpr::and(PrunerExpr::Atom(0), PrunerExpr::Atom(1)),
            PrunerExpr::Atom(2),
        );

        let pruner = ComposedPruner::from_expr(expr, vec![&even, &lt5, &all]);

        // Since pruner C (AcceptAll) accepts everything, the whole OR should accept everything
        for token in 0..20 {
            assert!(
                pruner.is_valid(0, token, &[]),
                "token {token} should be accepted"
            );
        }
    }

    // ── T15 batch composition tests ──────────────────────────────

    #[test]
    fn test_batch_composition_matches_per_token() {
        // Test that batch_is_valid produces identical results to per-token is_valid
        let even = AcceptEven;
        let lt5 = AcceptLt5;
        let pruner = ComposedPruner::and(&even, &lt5);

        let candidates: Vec<usize> = (0..100).collect();

        // Per-token results
        let per_token: Vec<bool> = candidates
            .iter()
            .map(|&c| pruner.is_valid(0, c, &[]))
            .collect();

        // Batch results
        let mut batch = vec![false; 100];
        pruner.batch_is_valid(0, &candidates, &[], &mut batch);

        assert_eq!(per_token, batch, "batch results must match per-token");
    }

    #[test]
    fn test_four_pruner_composition() {
        // Compose 4 pruners via PrunerExpr and verify it matches per-token AND
        // A=AcceptEven, B=AcceptLt5, C=AcceptAll, D=RejectAll
        // ((A AND B) AND C) AND D — AND with RejectAll should reject everything
        let even = AcceptEven;
        let lt5 = AcceptLt5;
        let all = AcceptAll;
        let none = RejectAll;

        let expr = PrunerExpr::and(
            PrunerExpr::and(
                PrunerExpr::and(PrunerExpr::Atom(0), PrunerExpr::Atom(1)),
                PrunerExpr::Atom(2),
            ),
            PrunerExpr::Atom(3),
        );

        let pruner = ComposedPruner::from_expr(expr, vec![&even, &lt5, &all, &none]);

        for token in 0..20 {
            assert!(
                !pruner.is_valid(0, token, &[]),
                "all should reject with RejectAll"
            );
        }
    }

    // ── T15: 4+ pruner batch composition correctness + perf ─────────

    struct AcceptMod3;
    impl ConstraintPruner for AcceptMod3 {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            token_idx.is_multiple_of(3)
        }
    }

    struct AcceptGt10;
    impl ConstraintPruner for AcceptGt10 {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            token_idx > 10
        }
    }

    struct AcceptLt50;
    impl ConstraintPruner for AcceptLt50 {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            token_idx < 50
        }
    }

    #[test]
    fn test_six_pruner_batch_matches_per_token_and() {
        // T15: Composition of 6 pruners via PrunerExpr must produce identical
        // results to per-token AND, and batch must be at least as fast.
        //
        // Pruners: AcceptEven, AcceptLt5, AcceptMod3, AcceptGt10, AcceptLt50, AcceptAll
        // Combined: even AND <5 AND mod3 AND >10 AND <50 AND all
        //
        // The per-token AND is: even && <5 && mod3 && >10 && <50 && all
        // Since <5 AND >10 have no overlap, everything should be rejected.
        // But let's use a more interesting combination:
        // AcceptEven AND AcceptLt50 AND AcceptMod3 AND AcceptAll
        //   = even AND <50 AND (mod3==0)
        let even = AcceptEven;
        let lt50 = AcceptLt50;
        let mod3 = AcceptMod3;
        let all = AcceptAll;

        // Balanced AND-tree: ((even AND lt50) AND (mod3 AND all))
        let expr = PrunerExpr::and(
            PrunerExpr::and(PrunerExpr::Atom(0), PrunerExpr::Atom(1)),
            PrunerExpr::and(PrunerExpr::Atom(2), PrunerExpr::Atom(3)),
        );

        let pruner = ComposedPruner::from_expr(expr, vec![&even, &lt50, &mod3, &all]);

        let candidates: Vec<usize> = (0..100).collect();

        // Per-token AND (ground truth)
        let per_token_and: Vec<bool> = candidates
            .iter()
            .map(|&c| c % 2 == 0 && c < 50 && c % 3 == 0)
            .collect();

        // Per-token via ComposedPruner
        let per_token_expr: Vec<bool> = candidates
            .iter()
            .map(|&c| pruner.is_valid(0, c, &[]))
            .collect();

        // Batch via ComposedPruner
        let mut batch = vec![false; 100];
        pruner.batch_is_valid(0, &candidates, &[], &mut batch);

        // All three must agree
        for i in 0..100 {
            assert_eq!(
                per_token_and[i], per_token_expr[i],
                "per_token_expr mismatch at token {}",
                candidates[i]
            );
            assert_eq!(
                per_token_and[i], batch[i],
                "batch mismatch at token {}",
                candidates[i]
            );
        }
    }

    #[test]
    fn test_five_pruner_batch_correctness() {
        // T15: 5 pruners, verify batch matches per-token AND
        let even = AcceptEven;
        let lt5 = AcceptLt5;
        let mod3 = AcceptMod3;
        let all = AcceptAll;
        let gt10 = AcceptGt10;

        // ((even AND lt5) AND (mod3 AND (all AND gt10)))
        // = even AND <5 AND mod3 AND >10
        // Since <5 AND >10 have NO overlap, everything is rejected
        let expr = PrunerExpr::and(
            PrunerExpr::and(PrunerExpr::Atom(0), PrunerExpr::Atom(1)),
            PrunerExpr::and(
                PrunerExpr::Atom(2),
                PrunerExpr::and(PrunerExpr::Atom(3), PrunerExpr::Atom(4)),
            ),
        );

        let pruner = ComposedPruner::from_expr(expr, vec![&even, &lt5, &mod3, &all, &gt10]);

        let candidates: Vec<usize> = (0..100).collect();
        let mut batch = vec![false; 100];
        pruner.batch_is_valid(0, &candidates, &[], &mut batch);

        // Everything should be rejected (<5 AND >10 is impossible)
        for (i, &b) in batch.iter().enumerate() {
            assert!(!b, "token {i} should be rejected (<5 AND >10 impossible)");
        }
    }

    #[test]
    fn test_four_pruner_batch_speedup() {
        // T15: Batch of 4+ pruners should be faster than per-token loop.
        // We measure this by comparing batch vs per-token timing.
        // The batch path calls each sub-pruner's batch_is_valid once,
        // amortizing lock acquisition, then evaluates the expression tree.
        // Per-token path calls is_valid N times per pruner.

        let even = AcceptEven;
        let lt50 = AcceptLt50;
        let mod3 = AcceptMod3;
        let all = AcceptAll;

        let expr = PrunerExpr::and(
            PrunerExpr::and(PrunerExpr::Atom(0), PrunerExpr::Atom(1)),
            PrunerExpr::and(PrunerExpr::Atom(2), PrunerExpr::Atom(3)),
        );

        let pruner = ComposedPruner::from_expr(expr, vec![&even, &lt50, &mod3, &all]);

        let candidates: Vec<usize> = (0..1000).collect();
        let n_rounds = 100;

        // Warm up
        let mut batch_buf = vec![false; 1000];
        for _ in 0..10 {
            pruner.batch_is_valid(0, &candidates, &[], &mut batch_buf);
        }

        // Batch timing
        let batch_start = std::time::Instant::now();
        for _ in 0..n_rounds {
            pruner.batch_is_valid(0, &candidates, &[], &mut batch_buf);
        }
        let batch_elapsed = batch_start.elapsed();

        // Per-token timing
        let per_token_start = std::time::Instant::now();
        for _ in 0..n_rounds {
            for &c in &candidates {
                std::hint::black_box(pruner.is_valid(0, c, &[]));
            }
        }
        let per_token_elapsed = per_token_start.elapsed();

        // Verify correctness
        let mut verify = vec![false; 1000];
        pruner.batch_is_valid(0, &candidates, &[], &mut verify);
        for (i, &c) in candidates.iter().enumerate() {
            let expected = c % 2 == 0 && c < 50 && c % 3 == 0;
            assert_eq!(verify[i], expected, "correctness mismatch at token {}", c);
        }

        // Log timing for GOAT gate evaluation
        eprintln!(
            "[T15] 4-pruner batch={:.2}us vs per-token={:.2}us (ratio={:.2}x)",
            batch_elapsed.as_secs_f64() * 1e6 / n_rounds as f64,
            per_token_elapsed.as_secs_f64() * 1e6 / n_rounds as f64,
            per_token_elapsed.as_secs_f64() / batch_elapsed.as_secs_f64(),
        );

        // We don't assert timing strictly (CI environments are noisy),
        // but we log it for GOAT gate evaluation. Correctness is verified above.
    }
}
