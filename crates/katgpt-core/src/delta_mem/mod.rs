//! δ-mem substrate: modelless associative bandit memory.
//!
//! Substrate extraction (Plan 008 Step 6, 2026-06-28): the pure data + algorithm
//! half of δ-mem (state machine, feature hasher, multi-domain aggregator) moved
//! here verbatim from `katgpt-rs/src/pruners/delta_mem/`. Composition that needs
//! the speculative-decoding `ScreeningPruner` (`pruner.rs`, `multi_pruner.rs`)
//! stays in the root crate as a thin re-export shim — those wrappers compose
//! memory corrections on top of substrate pruners.
//!
//! # Substrate vs Composition Split
//!
//! | Tier | Location | Content | Depends on |
//! |---|---|---|---|
//! | Substrate | `katgpt-core/src/delta_mem/{state,hash,multi}.rs` | `DeltaMemoryState`, `FeatureHasher`, `MultiDomainMemory` + configs + snapshots | `serde`, `fastrand`, `temporal_deriv` (optional core feature) |
//! | Composition | `katgpt-rs/src/pruners/delta_mem/{pruner,multi_pruner}.rs` | `MemorySteeredPruner<P>`, `MultiDomainMemoryPruner<P>` | `ScreeningPruner` (in `katgpt_core::traits`) |
//!
//! Distilled from δ-mem (arXiv 2605.12357), verified against source:
//!   `delta_impl.py` L1895-1938 (_memory_affine_scan_torch)
//!
//! # Modelless Adaptation
//!
//! Paper uses learned projections (W_mq, W_mk, W_mv, W∆q, W∆o).
//! We replace them with feature hashing (FeatureHasher).
//! The delta-rule update is identical — prediction error drives learning.

pub mod hash;
pub mod multi;
pub mod state;

pub use hash::{ContextFeatures, FeatureHasher, OutcomeFeatures};
pub use multi::{AggregationStrategy, MultiDomainMemory};
pub use state::{DeltaMemoryConfig, DeltaMemorySnapshot, DeltaMemoryState};

// Re-export the surprise-gate threshold when temporal_deriv is enabled, so
// consumers can read the default without depending on the module path.
#[cfg(feature = "temporal_deriv")]
pub use state::DEFAULT_THETA_SURPRISE;
