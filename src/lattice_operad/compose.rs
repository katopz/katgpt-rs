//! Operadic composition of pruner expressions.
//!
//! Provides the `compose` function that combines two PrunerExprs
//! using AND or OR, then canonicalizes the result.

use crate::lattice_operad::expr::PrunerExpr;
use crate::lattice_operad::word_problem::canonicalize;

/// Composition operator for pruner expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComposeOp {
    /// Conjunction (AND) — both must accept.
    And,
    /// Disjunction (OR) — either must accept.
    Or,
}

/// Compose two pruner expressions with the given operator, then canonicalize.
pub fn compose(lhs: &PrunerExpr, op: ComposeOp, rhs: &PrunerExpr) -> PrunerExpr {
    let combined = match op {
        ComposeOp::And => PrunerExpr::and(lhs.clone(), rhs.clone()),
        ComposeOp::Or => PrunerExpr::or(lhs.clone(), rhs.clone()),
    };
    canonicalize(&combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_operad::expr::PrunerResult;

    #[test]
    fn test_compose_and_same_atom() {
        let a = PrunerExpr::atom(0);
        let result = compose(&a, ComposeOp::And, &a);
        assert_eq!(result, PrunerExpr::Atom(0));
    }

    #[test]
    fn test_compose_or_same_atom() {
        let a = PrunerExpr::atom(0);
        let result = compose(&a, ComposeOp::Or, &a);
        assert_eq!(result, PrunerExpr::Atom(0));
    }

    #[test]
    fn test_compose_and_distributes() {
        let a = PrunerExpr::atom(0);
        let b_or_c = PrunerExpr::or(PrunerExpr::atom(1), PrunerExpr::atom(2));
        let result = compose(&a, ComposeOp::And, &b_or_c);
        for bits in 0..8 {
            let results = vec![(bits & 1) != 0, (bits & 2) != 0, (bits & 4) != 0];
            let expected = matches!(a.eval(&results), PrunerResult::Accept)
                && matches!(b_or_c.eval(&results), PrunerResult::Accept);
            let actual = result.eval(&results);
            if expected {
                assert!(matches!(actual, PrunerResult::Accept), "bits={bits}");
            }
        }
    }

    #[test]
    fn test_compose_absorption() {
        let a = PrunerExpr::atom(0);
        let a_and_b = PrunerExpr::and(PrunerExpr::atom(0), PrunerExpr::atom(1));
        let result = compose(&a, ComposeOp::Or, &a_and_b);
        assert_eq!(result, PrunerExpr::Atom(0));
    }
}
