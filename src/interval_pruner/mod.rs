//! Interval-preserving token pruning (arXiv:2503.13663).
//!
//! Enforces convexity of valid token sets — eliminates "Swiss cheese" patterns
//! where scattered tokens are rejected while in-between tokens are valid.
//!
//! The paper defines ⊞ as the cube category of interval-preserving monotone maps
//! between finite Boolean lattices. The key insight: **interval preservation**
//! means "if tokens i..j are all valid, their valid image must also be contiguous" —
//! this is a structural constraint on how token validity propagates.
//!
//! # Architecture
//!
//! - [`IntervalMask`] — boolean validity mask with interval-closure operations.
//! - [`IntervalPruner`] — wraps any [`ConstraintPruner`] and enforces interval
//!   closure on its batch output.
//! - [`simd`] — SIMD-accelerated interval operations with adaptive routing.
//!
//! # Adaptive Backend (Plan 252 Phase 5)
//!
//! For vocab sizes ≥ [`INTERVAL_SIMD_THRESHOLD`] (256), SIMD-accelerated
//! operations are used automatically. Below the threshold, scalar code is
//! faster due to SIMD setup overhead. Configure via [`AdaptiveConfig`].

#[cfg(feature = "interval_pruner")]
mod interval;
#[cfg(feature = "interval_pruner")]
mod simd;

#[cfg(feature = "interval_pruner")]
pub use interval::{IntervalMask, IntervalPruner};
#[cfg(feature = "interval_pruner")]
pub use simd::{AdaptiveConfig, RouteDecision, INTERVAL_SIMD_THRESHOLD, NERVE_SIMD_THRESHOLD};
