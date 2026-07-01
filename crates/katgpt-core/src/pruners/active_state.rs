//! Active-state trace contract — the shared bridge between producers and
//! consumers of MUX-Latent compression events.
//!
//! Plan 310 T2.6/T2.7 (Issue 002). This module lives in `katgpt-core` (the
//! common ancestor of both the producer and the consumer) so that:
//!
//! - **Producer** (`riir-games::TrialLog` / `ActiveStateEvent`) can implement
//!   the trait via a thin zero-allocation window adapter.
//! - **Consumer** (`katgpt-pruners::TraceInformedFeedbackBandit`) reads the
//!   three scalars to bias harness-vs-weight selection.
//!
//! The trait is intentionally minimal — three `f32` accessors, no gameplay
//! types, no integrity hashes. The IP-heavy `ActiveStateEvent` struct (HLA
//! `[f32; 5]` scalars + BLAKE3 commitment) stays private to `riir-games` and
//! exposes only these scalars through a window adapter impl.
//!
//! # Why this lives here (not in katgpt-pruners)
//!
//! Rust's orphan rule forbids implementing a foreign trait for a foreign type
//! unless the trait or the type is local. The producer (`ActiveStateEvent`)
//! lives in `riir-games`; the consumer (`TraceInformedFeedbackBandit`) lives in
//! `katgpt-pruners`. Neither can see the other's types. Placing the trait in
//! `katgpt-core` — the common ancestor both depend on — resolves the orphan
//! rule cleanly: the impl lives in `riir-games` (local type, foreign trait from
//! `katgpt-core`), the consumer lives in `katgpt-pruners` (local wrapper,
//! foreign trait from `katgpt-core`).

/// Read-only view over a windowed active-state trace.
///
/// Implementors summarize a slice of active-state events into three scalars:
/// - `compression_ratio_mean` — mean compression_ratio over the window.
/// - `constraint_trend` — signed slope of active_constraint_count (rising > 0).
/// - `hla_arousal` — HLA arousal scalar (diagnostic only, not in the decision).
///
/// The windowing strategy (last N events, last T ticks, exponentially
/// weighted) is the implementor's choice. A reasonable default is
/// "the most recent `2 × patience` events" so the trace covers the same
/// horizon as stall detection.
///
/// This trait is intentionally minimal — it carries no gameplay types and
/// no integrity hashes. The IP-heavy `ActiveStateEvent` struct (HLA scalars,
/// BLAKE3 commitment) stays in riir-games and exposes only these three
/// scalars through a thin adapter impl.
pub trait ActiveStateTrace {
    /// Mean compression_ratio over the recent trace window.
    ///
    /// Returns 0.0 when the trace is empty (no events recorded yet).
    fn compression_ratio_mean(&self) -> f32;

    /// Signed slope of `active_constraint_count` over the window.
    ///
    /// Positive = constraints are accumulating (harness struggling to fit).
    /// Negative = constraints are being resolved (harness keeping up).
    /// Returns 0.0 when the trace has fewer than 2 events (slope undefined).
    fn constraint_trend(&self) -> f32;

    /// HLA arousal scalar over the window (diagnostic, not in the decision).
    ///
    /// Included in the trait so callers can log it alongside the decision
    /// without needing a second adapter. Not read by the wrapper's policy.
    fn hla_arousal(&self) -> f32;
}

/// An empty trace — the default before any MUX-Latent compression events land.
///
/// All accessors return 0.0. The consumer's policy degrades to the inner
/// `FeedbackBandit`'s stall-only path when given this (the "no regression
/// when trace is empty" acceptance criterion).
#[derive(Debug, Clone, Copy, Default)]
pub struct EmptyTrace;

impl ActiveStateTrace for EmptyTrace {
    #[inline]
    fn compression_ratio_mean(&self) -> f32 {
        0.0
    }
    #[inline]
    fn constraint_trend(&self) -> f32 {
        0.0
    }
    #[inline]
    fn hla_arousal(&self) -> f32 {
        0.0
    }
}

/// Computed trace signal — the product
/// `compression_ratio_mean × (1 + max(constraint_trend, 0))`.
///
/// Pure function of an [`ActiveStateTrace`]; exposed for callers that want
/// to log the signal alongside the decision (e.g. "trace_signal=4.7 → WeightUpdate").
///
/// Calibrated threshold: `>= 3.5` forces `WeightUpdate` in the
/// `TraceInformedFeedbackBandit` (Plan 310 T3.3 controlled-corpus GOAT).
#[inline]
pub fn trace_signal<T: ActiveStateTrace + ?Sized>(trace: &T) -> f32 {
    trace.compression_ratio_mean() * (1.0 + trace.constraint_trend().max(0.0))
}
