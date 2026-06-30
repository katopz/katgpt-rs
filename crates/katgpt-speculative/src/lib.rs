//! katgpt-speculative — shared speculative decoding + DDTree substrate.
//!
//! Issue 013 (2026-06-29): collapses the fork between `katgpt-rs/src/speculative/`
//! and `riir-engine/src/{dd_tree,dflash}.rs`. The core DDTree algorithm lives
//! here so improvements propagate to both consumers.
//!
//! # What lives here
//!
//! - **DDTree core** (`dd_tree`): `build_dd_tree`, `build_dd_tree_pruned`,
//!   `build_dd_tree_screened`, `build_dd_tree_balanced`, `TreeBuilder`,
//!   `extract_parent_tokens`, `extract_best_path`, `merge_retrieved_branches`,
//!   `find_valid_sequence`, etc. Pure algorithm over pre-computed marginals.
//!   No `forward` dependency.
//!
//! # What does NOT live here
//!
//! - **Types** (`TreeNode`, `DraftResult`, `ConstraintPruner`, …) → already in
//!   `katgpt_core::speculative::types` + `katgpt_core::traits` (Plan 008 Phase 2.5).
//! - **Sampling** (`sample_from_distribution`) → already in
//!   `katgpt_core::speculative::sampling` (Plan 008 Phase 2.6).
//! - **DFlash** (`dflash_predict_*_with`) → the three zero-alloc `_with`
//!   cores live here (Issue 013 Phase B). They are generic over a `DflashCtx` +
//!   `DflashCache` backend trait pair and a `forward_fn` closure, because the
//!   underlying `ForwardContext` / `MultiLayerKVCache` / `TransformerWeights`
//!   types are crate-specific. The thin wrappers (`dflash_predict`, `_ar`,
//!   `_conditioned`, `_parallel`) and feature-gated variants (`_domino`,
//!   `_routing`, `_fusion`) stay in each consumer.
//! - **Feature-gated DDTree variants** (`build_dd_tree_belief`, `_speculative`,
//!   `_kurtosis`, `_domino`, `_manifold`, `_lodestar`, `_gdsd`, …) → stay in
//!   `katgpt-rs/src/speculative/dd_tree.rs` because they reference root-only
//!   sibling modules (`super::belief_drafter`, `super::spec_generator`, etc.).

pub mod dd_tree;
pub mod dflash;

// Re-export katgpt_core's speculative types + traits so consumers can import
// everything from one place (`katgpt_speculative::{TreeNode, ConstraintPruner, …}`).
pub use katgpt_core::speculative::types::*;
pub use katgpt_core::traits::{
    BinaryScreeningPruner, CompletionHorizon, ConstraintPruner, DominoPruner, NoPruner,
    NoScreeningPruner, ScreeningPruner,
};
