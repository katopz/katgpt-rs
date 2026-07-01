//! Distillation primitives — **split-boundary tagged** (Proposal 003 Phase 0.2).
//!
//! The `distill/` umbrella conflates two unrelated paper lineages and does NOT
//! survive as a unit. It is tagged here for the in-tree split; the actual file
//! moves happen in later phases:
//!
//! - **`peira`** → `katgpt-spectral` (Phase 4). PEIRA = spectral alignment
//!   metric (cross-view covariance eigenvector alignment). It's a spectral
//!   diagnostic, not a speculative-drafting primitive.
//! - **`ilc` + `trd`** → `katgpt-speculative` (Phase 6). ILC = Iterative Latent
//!   Clustering synonym-aware DDTree pruning; TRD = Trajectory-Refined Draft
//!   for speculative decoding. Both are speculative-draft screening primitives.
//!
//! Until those phases land, the modules stay here and the feature flags
//! (`peira_distill`, `ilc_distill`, `trd_refined_draft`) are unchanged.

// → katgpt-spectral (Phase 4): spectral alignment metric.
#[cfg(feature = "peira_distill")]
pub mod peira;

// → katgpt-speculative (Phase 6): synonym-aware DDTree pruning.
#[cfg(feature = "ilc_distill")]
pub mod ilc;

// → katgpt-speculative (Phase 6): trajectory-refined draft screening.
#[cfg(feature = "trd_refined_draft")]
pub mod trd;
