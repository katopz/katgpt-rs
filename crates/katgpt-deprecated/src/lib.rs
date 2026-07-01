//! # katgpt-deprecated — the loser crate
//!
//! Exiled primitives: dead stubs, off-topic research toys, GOAT-failed
//! mechanisms, and explicitly demoted primitives. **Never default-on.**
//!
//! ## Membership
//!
//! See `.docs/001_loser_sweep_audit.md` for the full membership table with
//! citations. Each item here carries a `TODO(deprecated): delete after ...`
//! comment — this crate exists to make deletion safe and auditable, not to
//! live forever.
//!
//! ## 3-category rule (recap)
//!
//! Only **category 3** (dead/failed) items live here. Categories 1 (pending)
//! and 2 (benchmark-loser kept for A/B) stay in their domain crates.

#[cfg(feature = "alien_sampler")]
pub mod alien_sampler;

#[cfg(feature = "feedback")]
pub mod feedback;

#[cfg(feature = "unit_distance")]
pub mod unit_distance;
