//! Auto-Dreamer Offline Memory Consolidation (Plan 107, Research 69).
//!
//! Decouples fast per-session memory acquisition from slow cross-session
//! consolidation via scheduled "dreaming" events.
//!
//! Key insight: region rewriting (treat a working region as read-only evidence,
//! synthesize a compact replacement) forces compactness structurally.
//!
//! # Usage
//!
//! ```ignore
//! use microgpt::pruners::dreamer::{DreamerScheduler, DreamerConfig};
//!
//! let config = DreamerConfig::default();
//! let mut scheduler = DreamerScheduler::new(config);
//!
//! if scheduler.should_consolidate(episode) {
//!     let region = scheduler.select_region(&bandit, episode);
//!     let replacement = consolidator.consolidate(&region);
//!     // Apply: bandit = (bandit \ region) ∪ replacement
//! }
//! ```

pub mod consolidator;
pub mod counterfactual;
pub mod decay;
pub mod frozen;
pub mod pipeline;
pub mod scheduler;
pub mod types;

pub use consolidator::DreamerConsolidator;
pub use counterfactual::CounterfactualEstimator;
pub use decay::MemoryDecay;
pub use frozen::{DreamerFrozenBank, load_frozen_dreamer, save_frozen_dreamer};
pub use pipeline::{ConsolidationResult, DreamerPipeline};
pub use scheduler::DreamerScheduler;
pub use types::{DecayPolicy, DreamerConfig, ReplacementSet, WorkingRegion};
