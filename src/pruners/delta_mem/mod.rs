//! δ-mem modelless distillation: associative bandit memory.
//!
//! Substrate extraction (Plan 008 Step 6, 2026-06-28): the pure data +
//! algorithm half — `DeltaMemoryState`, `FeatureHasher`, `MultiDomainMemory`
//! + their configs/snapshots — moved to [`katgpt_core::delta_mem`]. The
//! composition half (pruners that wrap a `ScreeningPruner` and add
//! memory-steered corrections) stays here because the inner pruner type
//! is a root-only trait object in many call sites.
//!
//! # Substrate vs Composition Split
//!
//! | Tier | Location | Content | Depends on |
//! |---|---|---|---|
//! | Substrate | `katgpt-core/src/delta_mem/{state,hash,multi}.rs` | `DeltaMemoryState`, `FeatureHasher`, `MultiDomainMemory` + configs + snapshots | `serde`, `fastrand`, `temporal_deriv` (optional core feature) |
//! | Composition | `src/pruners/delta_mem/{pruner,multi_pruner}.rs` | `MemorySteeredPruner<P>`, `MultiDomainMemoryPruner<P>` | `ScreeningPruner` (in `katgpt_core::traits`) |
//!
//! All existing import paths (`crate::pruners::delta_mem::DeltaMemoryConfig`,
//! `crate::pruners::delta_mem::state::DEFAULT_THETA_SURPRISE`, etc.) resolve
//! unchanged through the re-exports below.
//!
//! # Source Code Mapping (preserved)
//!
//! | Paper Component    | Source Location                    | Our Equivalent              |
//! |--------------------|------------------------------------|-----------------------------|
//! | OSAM state S       | DeltaMemAttention.delta_state      | DeltaMemoryState.state      |
//! | Read S·q           | L1921: einsum("bij,bj->bi")       | DeltaMemoryState::read()    |
//! | Write S'=(1-β)S-β·pred⊗k+β·v⊗k | L1923-1929      | DeltaMemoryState::write()   |
//! | Gate β=sigmoid(W·x+b) | L917-925 with couple_lambda    | Heuristic from δ statistics  |
//! | normalize_qk       | L805-814: L2_norm(tanh(...))     | FeatureHasher::hash_key()   |
//! | delta_o correction | L2283: attn_output + delta_o      | MemorySteeredPruner          |
//! | MSW (4 heads)      | L795-803: reshape + scan          | MultiDomainMemory            |
//! | SSW (message_mean) | L2150-2215: avg then single write | write_segment()              |

// Substrate re-export — single source of truth is katgpt-core.
pub use katgpt_core::delta_mem::{
    AggregationStrategy, ContextFeatures, DeltaMemoryConfig, DeltaMemorySnapshot,
    DeltaMemoryState, FeatureHasher, MultiDomainMemory, OutcomeFeatures,
};

// Re-expose the substrate module layout so absolute paths
// (`crate::pruners::delta_mem::state::DEFAULT_THETA_SURPRISE`,
// `crate::pruners::delta_mem::hash::FeatureHasher`) keep resolving.
pub mod hash {
    pub use katgpt_core::delta_mem::hash::{ContextFeatures, FeatureHasher, OutcomeFeatures};
}
pub mod multi {
    pub use katgpt_core::delta_mem::multi::{AggregationStrategy, MultiDomainMemory};
}
pub mod state {
    pub use katgpt_core::delta_mem::state::{
        DeltaMemoryConfig, DeltaMemorySnapshot, DeltaMemoryState,
    };
    #[cfg(feature = "temporal_deriv")]
    pub use katgpt_core::delta_mem::state::DEFAULT_THETA_SURPRISE;
}

// Composition layer: pruners that wrap an inner ScreeningPruner. Local
// because the P: ScreeningPruner generic is instantiated at root call sites
// (root owns the speculative-decoding composition types).
mod pruner;
mod multi_pruner;

pub use multi_pruner::MultiDomainMemoryPruner;
pub use pruner::{CorrectionMode, MemorySteeredPruner, WriteGranularity};
