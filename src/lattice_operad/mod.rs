//! LatticeOperad — canonical AND/OR composition for ConstraintPruner expressions.
//!
//! Uses the distributive lattice word problem to canonicalize pruner
//! expressions, eliminating redundant evaluations. Dualizes the operadic
//! structure on cubes to AND/OR composition of pruner constraints.
//!
//! Plan 252 Phase 2, Research 220 (arXiv:2503.13663)

mod compose;
mod composed_pruner;
mod expr;
mod word_problem;

pub use compose::{ComposeOp, compose};
pub use composed_pruner::ComposedPruner;
pub use expr::{PrunerExpr, PrunerResult};
pub use word_problem::canonicalize;
