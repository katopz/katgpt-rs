//! Speculative-decoding substrate (Plan 008 Step 5).
//!
//! Pure substrate types for speculative decoding: data types, configs,
//! algorithms, and trait implementations that depend only on
//! [`crate::types::Config`], [`crate::traits`], and std.
//!
//! ## What lives here
//! - [`types`]: `TreeNode`, `DraftResult`, `DraftEvent`, `RejectionReason`,
//!   `DecodeStrategy`, `SdeConfig`, `EarlyStopGate<P>`, `FlashPrefillConfig`,
//!   `BlockScores`, LDT conflict detector (`ConflictDetector`,
//!   `EntropyConflictDetector`), `TesNode`, `TrajectoryCredit`, and various
//!   feature-gated config types (DFlare fusion/kv-routing/progressive-budget,
//!   `LdtPruneConfig`, `SpecCostSnapshot`, `RoutingOverlapSnapshot`,
//!   `StabilitySnapshot`).
//!
//! ## What does NOT live here (stays in consumer crates)
//! - The companion traits (`ConstraintPruner`, `ScreeningPruner`, `DominoPruner`,
//!   `NoPruner`, `NoScreeningPruner`, `BinaryScreeningPruner`) — already in
//!   [`crate::traits`] since Plan 107 Phase 0.
//! - Composition types that need `katgpt-transformer`:
//!   [`SpeculativeContext`], [`DDTreeBranchCache`] — these need
//!   `ForwardContext`, `MultiLayerKVCache`, `PagedKVCache`, `forward_paged`.
//! - Consumer-crate-specific composition: `TesConfig` (needs `BanditStrategy`),
//!   `SelfSpecConfig` (needs `D2fDecodeConfig`).
//! - The DDTree builders (`build_dd_tree*`, `TreeBuilder`) — composition that
//!   drives the substrate; stays in the consumer.
//!
//! ## Feature gating
//! Always-on (no feature gate on the module itself — same pattern as
//! [`crate::simd`], [`crate::types`], [`crate::traits`], [`crate::hla`]).
//! Individual types are gated by their respective feature flags, forwarded
//! from the consumer via `katgpt-core/<feature>` (e.g. `katgpt-core/elf_sde`
//! gates `EarlyStopGate`).
//!
//! [`SpeculativeContext`]: katgpt_rs::speculative::SpeculativeContext
//! [`DDTreeBranchCache`]: katgpt_rs::speculative::DDTreeBranchCache

pub mod types;

// Re-export the substrate API at `katgpt_core::speculative::*` for ergonomic
// imports (`use katgpt_core::speculative::{TreeNode, DraftResult};`).
pub use types::*;
