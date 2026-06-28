//! Speculative-decoding sampling primitives — re-export shim.
//!
//! Substrate extraction (Plan 008 Step 6, 2026-06-28): the CDF + residual
//! samplers moved to [`katgpt_core::speculative::sampling`]. This file is a
//! thin re-export shim so existing call sites (`crate::speculative::sampling::*`,
//! `super::sampling::sample_from_distribution`, etc.) continue to resolve
//! unchanged.
//!
//! The samplers depend only on [`crate::types::Rng`] and
//! [`crate::simd::simd_scale_inplace`] — both already in core — so the
//! substrate code is byte-identical to the pre-move version.

#[allow(deprecated)]
pub use katgpt_core::speculative::sampling::{
    sample_from_distribution, sample_residual_distribution,
    sample_residual_distribution_into,
};
