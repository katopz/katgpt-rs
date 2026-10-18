//! Legal-token-set enumeration + the restricted-projection decision
//! (Issue 841, grammar-forced vocabulary-projection skip).
//!
//! # The shape of the problem
//!
//! A structural pruner — a grammar, a JSON schema, a DFA over a tool-call
//! syntax — knows its legal continuation set *from its state*. Every consumer
//! in this repo nonetheless asks it the opposite question, one token at a
//! time:
//!
//! ```text
//! for (i, &p) in marginal.iter().enumerate() {          // O(vocab)
//!     if p > 0.0 && horizon.is_valid(depth, i, parents) // O(prefix) each
//! ```
//!
//! With a 32 K vocabulary that is 32 768 predicate calls to discover the ten
//! tokens a grammar would have named directly — and in `build_dd_tree_lodestar`
//! it happens once per heap pop, with `is_valid` re-walking the prefix inside
//! every one of them.
//!
//! [`ConstraintPruner::legal_degree`] + [`ConstraintPruner::for_each_legal`]
//! invert the question. This module supplies the two pieces a pruner needs to
//! answer it and the one decision a projection consumer needs to act on it.
//!
//! [`ConstraintPruner::legal_degree`]: crate::traits::ConstraintPruner::legal_degree
//! [`ConstraintPruner::for_each_legal`]: crate::traits::ConstraintPruner::for_each_legal
//!
//! # What is here
//!
//! - [`CsrLegalSet`] — compressed sparse row over (state → sorted legal
//!   tokens). O(deg) enumeration, O(1) degree, zero allocation on read.
//!   Generalised from the one shipped state→successor enumerator in this
//!   repo, `bisimulation::graph::TransitionGraph` (`adjacency_window` +
//!   `for_each_adjacent`, CSR + callback, allocation-free) — the same shape
//!   over token ids instead of operator labels.
//! - [`ProjectionPlan`] + [`plan_projection`] — how much of the vocabulary
//!   projection a legal set of a given size actually justifies skipping.
//!
//! # ⛔ A small legal set does not always justify a gather
//!
//! The obvious rule — "|L| < V, therefore score only L" — is wrong, and this
//! repo has already measured why. A gathered-row pass over the LM head runs
//! at **20.6 GB/s against 108 GB/s** for the dense contiguous one
//! (`katgpt-forward::cluster_head` module docs, Issue 661), so the crossover
//! is a *fraction of the vocabulary*, not "any saving at all". Issue 661 put
//! the clustered head's own crossover at ~21–34 % active.
//! [`RestrictionPolicy::max_active_fraction`] carries that number as DATA
//! rather than a constant in a branch, and its default is deliberately below
//! the measured band — a gather that ties is a gather that lost, because it
//! also spends the enumeration.
//!
//! [`ProjectionPlan::Forced`] is the exception and is not a bandwidth
//! argument at all: with one legal token the projection cannot change the
//! answer, so the right amount of it to compute is none.
//!
//! # Sync boundary
//!
//! None. Pure structure over caller-owned state; no allocation on any read
//! path; no `Instant`, no RNG, no globals.
//!
//! Legal-token-set enumeration + the restricted-projection decision (Issue
//! 841) — `CsrLegalSet` inverts "is this token valid?" into "which tokens
//!      are", and `ProjectionPlan` says how much of the vocabulary projection that
//!      justifies skipping. Opt-in (`legal_token_set`).

use crate::traits::ConstraintPruner;

pub mod csr;
pub mod plan;

#[cfg(test)]
mod tests;

pub use csr::CsrLegalSet;
pub use plan::{ProjectionPlan, RestrictionPolicy, plan_projection};

/// The single legal token at this position, or `None`.
///
/// `None` covers three genuinely different states and the caller usually
/// wants [`plan_projection`] instead, which separates them: the pruner cannot
/// enumerate, the legal set is empty, or it holds more than one token.
///
/// Zero allocation.
pub fn first_legal<P: ConstraintPruner + ?Sized>(
    pruner: &P,
    depth: usize,
    parent_tokens: &[usize],
) -> Option<usize> {
    match pruner.legal_degree(depth, parent_tokens) {
        Some(1) => {
            let mut found = None;
            pruner.for_each_legal(depth, parent_tokens, &mut |t| {
                if found.is_none() {
                    found = Some(t);
                }
            });
            found
        }
        _ => None,
    }
}

/// Collect the legal set into a caller-owned buffer, reusing its capacity.
///
/// Returns `None` when the pruner cannot enumerate — `out` is then left
/// untouched, so a caller that reuses one buffer across positions cannot
/// mistake a stale set for a fresh one by reading a cleared buffer as "no
/// token is legal". Returns `Some(n)` with `out.len() == n` otherwise.
///
/// Allocates only when `out`'s capacity is too small.
pub fn collect_legal_into<P: ConstraintPruner + ?Sized>(
    pruner: &P,
    depth: usize,
    parent_tokens: &[usize],
    out: &mut Vec<usize>,
) -> Option<usize> {
    let degree = pruner.legal_degree(depth, parent_tokens)?;
    out.clear();
    out.reserve(degree);
    pruner.for_each_legal(depth, parent_tokens, &mut |t| out.push(t));
    debug_assert_eq!(
        out.len(),
        degree,
        "ConstraintPruner contract: legal_degree said {degree}, for_each_legal yielded {}",
        out.len()
    );
    Some(out.len())
}
