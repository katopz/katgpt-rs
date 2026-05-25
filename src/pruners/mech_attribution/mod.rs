//! Mechanistic Data Attribution — Catalyst Pattern Detection + Influence Proxy.
//!
//! Plan 111, Research 009 (arXiv:2601.21996).
//! Feature gate: `mech_attribution` (opt-in, requires `cna_steering`, `ropd_rubric`, `bandit`).

#[cfg(feature = "mech_attribution")]
mod augmentation;
#[cfg(feature = "mech_attribution")]
mod catalyst;
#[cfg(feature = "mech_attribution")]
mod scoring;
#[cfg(feature = "mech_attribution")]
mod types;

#[cfg(feature = "mech_attribution")]
pub use augmentation::{CatalystTemplate, extract_template, generate_synthetic};
#[cfg(feature = "mech_attribution")]
pub use catalyst::{catalyst_score, detect_catalyst_pattern};
#[cfg(feature = "mech_attribution")]
pub use scoring::{ActivationInfluenceProxy, batch_influence_rank};
#[cfg(feature = "mech_attribution")]
pub use types::*;
