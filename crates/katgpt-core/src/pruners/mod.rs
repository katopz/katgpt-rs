//! Pruner modules relocated from the katgpt-rs root crate so downstream
//! crates (riir-engine via katgpt-core) can consume them without depending
//! on the root crate.

#[cfg(feature = "review_metrics")]
pub mod review_metrics;

#[cfg(feature = "indicator_probe_bank")]
pub mod indicator_probe_bank;

#[cfg(feature = "indicator_similarity")]
pub mod indicator_similarity;

#[cfg(feature = "indicator_cascade")]
pub mod indicator_cascade;
