//! Progressive Monte-Carlo Graph Search (MCGS) — generic, modelless, MIT-licensed.
//!
//! Implements the distilled primitives from MLEvolve
//! (Du et al., Shanghai AI Lab + ECNU, arxiv 2606.06473, 2026-06-04).
//!
//! # The Three Primitives
//!
//! This module ships three transferable primitives, stripped of the paper's
//! LLM-coding-agent wrapper:
//!
//! 1. **Reference-edge graph search** — a directed graph `G = (V, E)` where
//!    `E = E_T ∪ E_ref`:
//!      - **Primary edges** `E_T` carry parent→child generative relationships
//!        and participate in selection + backprop (credit assignment).
//!      - **Reference edges** `E_ref` carry cross-branch / non-adjacent
//!        information flow and are **excluded from backprop**. They participate
//!        only in proposal construction (read at expansion time).
//!
//!    When `E_ref = ∅`, the search reduces to standard MCTS.
//!
//! 2. **Entropy-gated scheduler** — a probabilistic soft switch between UCT
//!    exploration and Elite-Guided exploitation via a decaying weight `w(t)`:
//!      - `P(UCT)   = w(t)`
//!      - `P(Elite) = 1 - w(t)`
//!
//!    The schedule is designed so the empirical branch-selection entropy
//!    `H(π_t)` decreases monotonically over search progress, concentrating
//!    compute on promising branches. Paper empirically shows 4.8 → 2.8 active
//!    branches; vanilla MCTS stays flat at ≈4.3.
//!
//! 3. **Stagnation gates** — branch-level (τ consecutive non-improving
//!    expansions) and global-level (τ_global steps without global-best
//!    refresh) triggers that fire composition/fusion expansion operators
//!    (intra-branch evolution, cross-branch reference, multi-branch aggregation).
//!
//! # Critical Invariant
//!
//! **Backprop walks `E_T` only. Never add code that propagates reward through
//! `E_ref`.** This is the single most important correctness property — it
//! guarantees that reference edges compose information without polluting
//! credit assignment. See [`ProgressiveMcgs::backprop`](graph::ProgressiveMcgs::backprop).
//!
//! # Layering Note
//!
//! [`EntropyGatedScheduler`](scheduler::EntropyGatedScheduler) operates *within a search tree*
//! (UCT vs Elite selection). It composes with — does not conflict with — `BreakevenComplexityRouter`
//! (Research 218), which routes *across inference strategies* (plasma/hot/warm).
//!
//! # Reference
//!
//! See `.research/239_MLEvolve_Progressive_MCGS_Entropy_Schedule.md` for the
//! full distillation verdict, fusion ideas, and GOAT gate matrix.
//! See `.plans/272_progressive_mcgs.md` for the implementation plan.
//!
//! Downstream consumers (game runtime, chain) instantiate the generic operators
//! with their own payloads — this module ships no game IP, no chain IP.
//!
//! # Phase 2: Orchestrator
//!
//! For most consumers, [`search::ProgressiveMcgsSearch`] is the entry point —
//! it wires the graph + scheduler + stagnation gate into a single `step()`
//! call and delegates domain-specific decisions (payload, reward) to a
//! consumer-provided [`search::SearchDomain`] impl. See the `search` module
//! docs for the full protocol.
//!
//! # DRY Audit Verdict (Phase 2)
//!
//! This module intentionally does **not** share code with `BanditPruner`
//! or `ConstraintPruner`. Audit (Plan 272 Phase 2) found they operate in
//! different domains:
//! - `BanditPruner` UCB1 is per-arm with fixed `√2` coefficient.
//! - `progressive_mcgs::uct` is MCTS UCT with parent visits, time-decayed
//!   `c(t)`, and `ε` smoothing.
//! - `ConstraintPruner::is_valid(depth, token_idx, parent_tokens) -> bool`
//!   validates token streams, not graph nodes.
//!
//! Consumers compose them at a higher layer via the `SearchDomain` trait.
//!
//! # Quick-start Example
//!
//! Minimal end-to-end search: a synthetic domain where branch 0 is "good"
//! (emits `Reward::Progress`) and all other branches emit `Reward::Failure`.
//! The scheduler should concentrate compute on branch 0 over the budget.
//!
//! ```rust
//! # #![cfg(feature = "progressive_mcgs")]
//! use katgpt_rs::progressive_mcgs::{
//!     BranchId, NodeId, ProgressiveMcgs, ProgressiveMcgsConfig,
//!     ProgressiveMcgsSearch, Reward, SearchDomain,
//! };
//!
//! struct GoodBranchDomain;
//!
//! impl SearchDomain<u32> for GoodBranchDomain {
//!     fn propose(
//!         &mut self, _g: &ProgressiveMcgs<u32>, _parent: NodeId,
//!         _branch: BranchId, _refs: &[NodeId], step_index: u32,
//!     ) -> u32 {
//!         step_index // payload is just the counter
//!     }
//!
//!     fn evaluate(&mut self, g: &ProgressiveMcgs<u32>, node: NodeId) -> Reward {
//!         if g.branch_of(node) == BranchId(0) { Reward::Progress }
//!         else { Reward::Failure }
//!     }
//! }
//!
//! let mut search = ProgressiveMcgsSearch::new(
//!     ProgressiveMcgsConfig::default(), /* n_branches */ 5,
//! ).with_max_expansions(50);
//! search.add_root(0);
//! for b in 0..5 {
//!     search.seed_branch(BranchId(b), 100 + b);
//! }
//!
//! let mut domain = GoodBranchDomain;
//! let mut rng = fastrand::Rng::with_seed(0xC0FFEE);
//! let mut breakthroughs = 0;
//! while let Some(step) = search.step(&mut domain, &mut rng) {
//!     if step.reward == Reward::Breakthrough { breakthroughs += 1; }
//! }
//! // Branch 0 should produce at least one breakthrough under the schedule.
//! assert!(breakthroughs >= 1, "expected ≥1 breakthrough on good branch");
//! ```
//!
//! See [`.docs/progressive_mcgs.md`](../../../.docs/progressive_mcgs.md) for the
//! full API reference, config knob table, and composition examples with
//! `BanditPruner` / `ConstraintPruner`.

pub mod graph;
pub mod operators;
pub mod scheduler;
pub mod search;
pub mod stagnation;
pub mod types;
pub mod uct;

pub use graph::{ExpansionOperator, ProgressiveMcgs};
pub use operators::{
    cross_branch_top_n, intra_branch_history, multi_branch_aggregate, DEFAULT_AGG_PER_BRANCH,
    DEFAULT_CROSS_BRANCH_N, DEFAULT_INTRA_BRANCH_K,
};
pub use scheduler::{EntropyGatedScheduler, RngLite, SelectMode};
pub use search::{classify_reward, ProgressiveMcgsSearch, SearchDomain, StepResult};
pub use stagnation::{
    BranchStagnationState, GlobalStagnationState, StagnationGate, StagnationTrigger, StagnationTriggers,
};
pub use types::{BranchId, EdgeKind, NodeId, ProgressiveMcgsConfig, Reward, MAX_REFS_PER_NODE};
pub use uct::{exploration_constant, uct_descend_to_leaf, uct_select_child};

#[cfg(test)]
mod tests;
