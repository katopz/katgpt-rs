//! Mechanistic Data Attribution — Catalyst Pattern Detection + Influence Proxy.
//!
//! Plan 111, Research 009 (arXiv:2601.21996).
//! Feature gate: `mech_attribution` (opt-in, requires `cna_steering`, `ropd_rubric`, `bandit`).
//!
//! # T9 Integration with ROPD Rubric
//!
//! [`score_with_influence`] combines rubric scores with catalyst influence scores.
//! High-influence catalyst samples get a 50% boost on the catalyst component.
//! Feature-gated behind both `mech_attribution` and `ropd_rubric`.

#[cfg(feature = "mech_attribution")]
mod augmentation;
#[cfg(feature = "mech_attribution")]
mod catalyst;
// Issue 1237 T1/T2 — census screen (AUROC vs scrambled-label searches) +
// typed control verdicts; consumer riir-infer Issue 1241.
#[cfg(feature = "mech_attribution")]
pub mod census_screen;
#[cfg(feature = "mech_attribution")]
mod scoring;
#[cfg(feature = "mech_attribution")]
mod types;

#[cfg(all(feature = "mech_attribution", feature = "ropd_rubric"))]
mod integration;

#[cfg(feature = "mech_attribution")]
pub use augmentation::{CatalystTemplate, extract_template, generate_synthetic};
#[cfg(feature = "mech_attribution")]
pub use catalyst::{catalyst_score, detect_catalyst_pattern};
#[cfg(feature = "mech_attribution")]
pub use census_screen::{
    BandStats, ControlParams, ControlVerdict, ScreenParams, ScreenScratch, ScreenUnderFloor,
    ScreenVerdict, UnderFloor, UnitScreen, VerdictTally, control_verdict, screen_auroc,
};
#[cfg(feature = "mech_attribution")]
pub use scoring::{ActivationInfluenceProxy, batch_influence_rank};
#[cfg(feature = "mech_attribution")]
pub use types::*;

#[cfg(all(feature = "mech_attribution", feature = "ropd_rubric"))]
pub use integration::{batch_score_with_influence, score_with_influence};
