//! Pruner modules relocated from the katgpt-rs root crate so downstream
//! crates (riir-engine via katgpt-core) can consume them without depending
//! on the root crate.

/// Active-state trace contract (Plan 310 T2.6/T2.7, Issue 002).
///
/// Always available — zero dependencies. The shared bridge between producers
/// (`riir-games::ActiveStateEvent`) and consumers
/// (`katgpt-pruners::TraceInformedFeedbackBandit`).
pub mod active_state;

#[cfg(feature = "review_metrics")]
pub mod review_metrics;

#[cfg(feature = "indicator_probe_bank")]
pub mod indicator_probe_bank;

#[cfg(feature = "indicator_similarity")]
pub mod indicator_similarity;

#[cfg(feature = "indicator_cascade")]
pub mod indicator_cascade;
