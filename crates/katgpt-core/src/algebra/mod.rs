//! Algebraic-structure primitives for katgpt-rs.
//!
//! Currently home to the tropical (max, +) semiring primitive (Plan 337,
//! Research 321). Future algebraic variants (other semirings, idempotent
//! analysis) land here as siblings.

#[cfg(feature = "tropical_algebra")]
pub mod tropical;
