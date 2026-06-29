//! State-Source Modelless Distillation (Plan 142) + Nexus Elo P-UCB (Plan 143).
//!
//! Provides modelless OPD analogue components for bandit-based distillation:
//! - State-visitation entropy tracking for exploration coverage
//! - Continuation scoring (validator-driven, no model required)
//! - Retention metrics for GOAT proofs
//! - Generic P-UCB selector with adaptive exploration (Plan 143 Phase 2)

pub mod continuation;
pub mod pucb_selector;
pub mod retention;
pub mod visitation;

pub use continuation::{ContinuationScore, ContinuationScorer};
pub use pucb_selector::{DEFAULT_PUCB_C, DEFAULT_TOP_K, PUCBSelector, adaptive_c};
pub use retention::RetentionMetric;
pub use visitation::StateVisitationTracker;
