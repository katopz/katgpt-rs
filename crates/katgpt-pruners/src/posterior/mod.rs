//! Posterior-Guided Pruner Evolution (PGPE)
//!
//! Treats each pruner arm as a Bayesian hypothesis with per-feature precision.
//! Uses BAKE-style precision vectors to drive five lifecycle actions:
//! Explore, Patch, Split, Compress, Retire.
//!
//! Research: R211 (Bayesian-Agent distillation), R209 (BAKE precision)
//! Plan: 239

pub mod policy;
pub mod precision;
pub mod surprise;
pub mod types;
pub mod wrapper;

#[cfg(feature = "gated_summary")]
pub mod gated_summary;

#[cfg(feature = "gated_summary")]
pub use gated_summary::{
    DEFAULT_BETA, GatedSummary, SCORE_CLIP, gamma_ratio, gamma_step_view, score_weight,
};
pub use policy::{LifecycleAction, PrecisionPolicy, PrecisionPolicyConfig};
pub use precision::PrecisionVector;
pub use surprise::SurpriseComputer;
pub use types::{EvidenceContext, EvidenceOutcome, FailureMode, PosteriorEvidence};
pub use wrapper::PosteriorGuidedPruner;
