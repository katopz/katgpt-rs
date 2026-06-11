//! Distributive lattice word problem solver.
//!
//! Canonicalizes PrunerExpr using:
//! - Idempotency: A ∧ A = A, A ∨ A = A
//! - Absorption: A ∧ (A ∨ B) = A, A ∨ (A ∧ B) = A
//! - Distributivity: A ∧ (B ∨ C) = (A ∧ B) ∨ (A ∧ C)
//! - Commutativity: A ∧ B = B ∧ A, A ∨ B = B ∧ A (normalize to sorted order)
//! - Associativity: flatten nested AND/OR into n-ary form
//!
//! The canonical form is: OR of AND-clauses (DNF) where each AND-clause
//! is a sorted set of atoms with no duplicates.

use crate::lattice_operad::expr::PrunerExpr;

/// Canonicalize a pruner expression using the distributive lattice word problem.
///
/// Converts to Disjunctive Normal Form (DNF) and simplifies using
/// idempotency and absorption laws.
pub fn canonicalize(expr: &PrunerExpr) -> PrunerExpr {
    let dnf = to_dnf(expr);
    let simplified = simplify_dnf(&dnf);
    dnf_to_expr(&simplified)
}

/// A clause in DNF: conjunction of atoms (sorted, deduplicated).
type Clause = Vec<usize>;

/// DNF representation: disjunction of clauses.
type Dnf = Vec<Clause>;

/// Convert expression to DNF (Disjunctive Normal Form).
fn to_dnf(expr: &PrunerExpr) -> Dnf {
    match expr {
        PrunerExpr::Atom(id) => vec![vec![*id]],
        PrunerExpr::And(lhs, rhs) => {
            let left_dnf = to_dnf(lhs);
            let right_dnf = to_dnf(rhs);
            // AND distributes over OR: product of clauses
            let mut result = Vec::with_capacity(left_dnf.len() * right_dnf.len());
            for lc in &left_dnf {
                for rc in &right_dnf {
                    let mut merged = lc.clone();
                    merged.extend(rc.iter().copied());
                    merged.sort();
                    merged.dedup();
                    result.push(merged);
                }
            }
            result
        }
        PrunerExpr::Or(lhs, rhs) => {
            let mut result = to_dnf(lhs);
            result.extend(to_dnf(rhs));
            result
        }
    }
}

/// Simplify DNF using absorption and subsumption.
fn simplify_dnf(dnf: &Dnf) -> Dnf {
    let mut clauses: Dnf = dnf
        .iter()
        .map(|c| {
            let mut sorted = c.clone();
            sorted.sort();
            sorted.dedup();
            sorted
        })
        .collect();

    // Remove tautologies (empty clause = always true)
    clauses.retain(|c| !c.is_empty());

    // Subsumption: if clause A ⊆ clause B, then B is absorbed by A.
    // Remove B (more specific clause is subsumed by less specific).
    let mut to_remove = vec![false; clauses.len()];
    for i in 0..clauses.len() {
        if to_remove[i] {
            continue;
        }
        for j in 0..clauses.len() {
            if i == j || to_remove[j] {
                continue;
            }
            if is_subset(&clauses[i], &clauses[j]) {
                to_remove[j] = true;
            }
        }
    }

    clauses
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !to_remove[*i])
        .map(|(_, c)| c)
        .collect()
}

/// Check if `a` is a subset of `b` (both sorted).
fn is_subset(a: &[usize], b: &[usize]) -> bool {
    let mut bi = 0;
    for &ai in a {
        while bi < b.len() && b[bi] < ai {
            bi += 1;
        }
        if bi >= b.len() || b[bi] != ai {
            return false;
        }
        bi += 1;
    }
    true
}

/// Convert simplified DNF back to PrunerExpr.
fn dnf_to_expr(dnf: &Dnf) -> PrunerExpr {
    if dnf.is_empty() {
        return PrunerExpr::Atom(0);
    }

    let clauses: Vec<PrunerExpr> = dnf
        .iter()
        .map(|clause| {
            if clause.len() == 1 {
                PrunerExpr::Atom(clause[0])
            } else {
                clause
                    .iter()
                    .map(|&id| PrunerExpr::Atom(id))
                    .reduce(|acc, e| PrunerExpr::and(acc, e))
                    .unwrap()
            }
        })
        .collect();

    clauses
        .into_iter()
        .reduce(|acc, e| PrunerExpr::or(acc, e))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idempotency_and() {
        let expr = PrunerExpr::and(PrunerExpr::atom(0), PrunerExpr::atom(0));
        let canon = canonicalize(&expr);
        assert_eq!(canon, PrunerExpr::Atom(0));
    }

    #[test]
    fn test_idempotency_or() {
        let expr = PrunerExpr::or(PrunerExpr::atom(0), PrunerExpr::atom(0));
        let canon = canonicalize(&expr);
        assert_eq!(canon, PrunerExpr::Atom(0));
    }

    #[test]
    fn test_distributivity() {
        let expr = PrunerExpr::and(
            PrunerExpr::atom(0),
            PrunerExpr::or(PrunerExpr::atom(1), PrunerExpr::atom(2)),
        );
        let canon = canonicalize(&expr);
        for a0 in [false, true] {
            for a1 in [false, true] {
                for a2 in [false, true] {
                    let results = vec![a0, a1, a2];
                    let original_result = expr.eval(&results);
                    let canon_result = canon.eval(&results);
                    match original_result {
                        crate::lattice_operad::expr::PrunerResult::Accept => {
                            assert!(
                                matches!(
                                    canon_result,
                                    crate::lattice_operad::expr::PrunerResult::Accept
                                ),
                                "Mismatch at a0={a0}, a1={a1}, a2={a2}"
                            );
                        }
                        crate::lattice_operad::expr::PrunerResult::Reject => {
                            assert!(
                                matches!(
                                    canon_result,
                                    crate::lattice_operad::expr::PrunerResult::Reject
                                ),
                                "Mismatch at a0={a0}, a1={a1}, a2={a2}"
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    #[test]
    fn test_canonicalize_complex() {
        let expr = PrunerExpr::or(
            PrunerExpr::and(PrunerExpr::atom(0), PrunerExpr::atom(1)),
            PrunerExpr::and(PrunerExpr::atom(0), PrunerExpr::atom(2)),
        );
        let canon = canonicalize(&expr);
        assert!(
            canon.node_count() <= expr.node_count() + 2,
            "Canonical form should not be much larger"
        );
    }

    #[test]
    fn test_absorption() {
        let expr = PrunerExpr::or(
            PrunerExpr::atom(0),
            PrunerExpr::and(PrunerExpr::atom(0), PrunerExpr::atom(1)),
        );
        let canon = canonicalize(&expr);
        assert_eq!(canon, PrunerExpr::Atom(0));
    }
}
