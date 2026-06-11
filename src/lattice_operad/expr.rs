//! PrunerExpr enum — canonical AND/OR expression tree for constraint composition.

/// Result of evaluating a pruner expression for a single token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrunerResult {
    /// Token is definitely valid (all pruners accept).
    Accept,
    /// Token is definitely invalid (at least one pruner rejects).
    Reject,
    /// Result is unknown (need more information).
    Unknown,
}

/// Canonical pruner expression tree.
///
/// Represents AND/OR composition of atom IDs. After canonicalization,
/// the expression is in DNF normal form where:
/// - Idempotency: A AND A = A, A OR A = A
/// - Absorption: A AND (A OR B) = A, A OR (A AND B) = A
/// - Distributivity: A AND (B OR C) = (A AND B) OR (A AND C)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrunerExpr {
    /// Leaf: reference to a pruner by index.
    Atom(usize),
    /// Conjunction (intersection) — token must pass ALL sub-expressions.
    And(Box<PrunerExpr>, Box<PrunerExpr>),
    /// Disjunction (union) — token must pass ANY sub-expression.
    Or(Box<PrunerExpr>, Box<PrunerExpr>),
}

impl PrunerExpr {
    /// Create an atom expression.
    pub fn atom(id: usize) -> Self {
        Self::Atom(id)
    }

    /// Create an AND expression.
    pub fn and(lhs: PrunerExpr, rhs: PrunerExpr) -> Self {
        Self::And(Box::new(lhs), Box::new(rhs))
    }

    /// Create an OR expression.
    pub fn or(lhs: PrunerExpr, rhs: PrunerExpr) -> Self {
        Self::Or(Box::new(lhs), Box::new(rhs))
    }

    /// Evaluate this expression against per-atom boolean results.
    ///
    /// `atom_results` maps atom IDs to their evaluation result.
    /// Returns `Unknown` if any referenced atom is not in the map.
    pub fn eval(&self, atom_results: &[bool]) -> PrunerResult {
        match self {
            PrunerExpr::Atom(id) => match atom_results.get(*id) {
                Some(true) => PrunerResult::Accept,
                Some(false) => PrunerResult::Reject,
                None => PrunerResult::Unknown,
            },
            PrunerExpr::And(lhs, rhs) => match lhs.eval(atom_results) {
                PrunerResult::Reject => PrunerResult::Reject,
                PrunerResult::Unknown => match rhs.eval(atom_results) {
                    PrunerResult::Reject => PrunerResult::Reject,
                    _ => PrunerResult::Unknown,
                },
                PrunerResult::Accept => rhs.eval(atom_results),
            },
            PrunerExpr::Or(lhs, rhs) => match lhs.eval(atom_results) {
                PrunerResult::Accept => PrunerResult::Accept,
                PrunerResult::Unknown => match rhs.eval(atom_results) {
                    PrunerResult::Accept => PrunerResult::Accept,
                    _ => PrunerResult::Unknown,
                },
                PrunerResult::Reject => rhs.eval(atom_results),
            },
        }
    }

    /// Collect all atom IDs referenced in this expression.
    pub fn atom_ids(&self) -> Vec<usize> {
        match self {
            PrunerExpr::Atom(id) => vec![*id],
            PrunerExpr::And(l, r) | PrunerExpr::Or(l, r) => {
                let mut ids = l.atom_ids();
                ids.extend(r.atom_ids());
                ids
            }
        }
    }

    /// Count the number of nodes in the expression tree.
    pub fn node_count(&self) -> usize {
        match self {
            PrunerExpr::Atom(_) => 1,
            PrunerExpr::And(l, r) | PrunerExpr::Or(l, r) => 1 + l.node_count() + r.node_count(),
        }
    }
}
