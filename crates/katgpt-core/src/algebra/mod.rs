//! Algebraic-structure primitives for katgpt-rs.
//!
//! Currently home to the tropical (max, +) semiring primitive (Plan 337,
//! Research 321). Future algebraic variants (other semirings, idempotent
//! analysis) land here as siblings.
//!
//! Algebraic-structure primitives. Currently home to the tropical (max, +)
//! semiring (Plan 337, Research 321). Opt-in via `tropical_algebra`.

#[cfg(feature = "tropical_algebra")]
pub mod tropical;
