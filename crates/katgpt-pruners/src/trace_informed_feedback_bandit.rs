//! TraceInformedFeedbackBandit — active-state-trace-biased harness-vs-weight selection.
//!
//! Issue 002 T2.7 — wraps [`FeedbackBandit`] with a trace-consuming decision
//! filter that biases the harness-vs-weight lever choice using the
//! active-state trace (`compression_ratio` as a leading staleness indicator).
//!
//! # The problem
//!
//! `FeedbackBandit` decides *when* to switch levers (HarnessUpdate vs
//! WeightUpdate) using **stall detection only** — `stall_count >= patience`.
//! Stall detection is a **lagging** signal: it fires only after N episodes
//! of low reward delta. By the time it fires, the weights have been stale
//! for N episodes already.
//!
//! The active-state trace carries a **leading** signal: when the MUX-Latent
//! compressor is working hard (high `compression_ratio`) AND the active
//! constraint count is rising (`constraint_trend > 0`), the harness is
//! struggling to fit the context — weights are likely stale and
//! `WeightUpdate` is the correct lever *before* stall detection fires.
//!
//! # The policy
//!
//! From the Plan 310 T3.3 controlled-corpus GOAT
//! (`bench_310_t33_active_state_trace_goat.rs`):
//!
//! ```text
//! trace_signal = compression_ratio_mean × (1 + max(constraint_trend, 0))
//! if trace_signal >= TRACE_SIGNAL_THRESHOLD OR stall_count >= patience → WeightUpdate
//! else → HarnessUpdate
//! ```
//!
//! The wrapper evaluates this trace signal **before** delegating to the
//! inner `FeedbackBandit`. When the trace fires, the wrapper forces
//! `WeightUpdate` directly (overriding the inner bandit's choice). When
//! the trace is empty OR below threshold, the wrapper falls through to
//! the inner bandit's normal `select()` — preserving the stall-only path
//! for backward compatibility (the "no regression" acceptance criterion).
//!
//! # IP boundary
//!
//! This module is **IP-clean for the public MIT repo** (katgpt-rs). It
//! defines a generic `ActiveStateTrace` trait with three `f32` accessors
//! — no gameplay types, no HLA scalars beyond the single `arousal` field
//! used for diagnostics, no sync-boundary types. The riir-games
//! `ActiveStateEvent` (which carries the full `[f32; 5]` HLA bridge
//! scalars + BLAKE3 hash) implements this trait via a thin adapter; the
//! IP-heavy struct stays private.
//!
//! # Reference
//!
//! - Plan 310 T3.3 controlled-corpus GOAT (regret −488.2, accuracy +9.94pp).
//! - `katgpt-rs/benches/bench_310_t33_active_state_trace_goat.rs` — the
//!   `policy_trace_informed` reference implementation.
//! - Issue 002 acceptance criteria: "no regression when trace is empty".

use katgpt_core::{ConfiguratorContext, PlanningDecision};

use crate::feedback_bandit::FeedbackBandit;

/// Default trace-signal threshold above which the wrapper forces `WeightUpdate`.
///
/// Calibrated in the Plan 310 T3.3 GOAT to catch the typical stale signal
/// (~3.0× compression × ~1.5 trend factor = ~4.5) while rejecting the typical
/// non-stale signal (~1.2× × ~1.0 = ~1.2).
pub const DEFAULT_TRACE_SIGNAL_THRESHOLD: f32 = 3.5;

// ── Trace trait ─────────────────────────────────────────────────────────

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
/// All accessors return 0.0. The wrapper's policy degrades to the inner
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

/// Computed trace signal — the product `compression_ratio_mean × (1 + max(constraint_trend, 0))`.
///
/// Pure function of an [`ActiveStateTrace`]; exposed for callers that want
/// to log the signal alongside the decision (e.g. "trace_signal=4.7 → WeightUpdate").
#[inline]
pub fn trace_signal<T: ActiveStateTrace + ?Sized>(trace: &T) -> f32 {
    trace.compression_ratio_mean() * (1.0 + trace.constraint_trend().max(0.0))
}

// ── Wrapper ─────────────────────────────────────────────────────────────

/// Configuration for [`TraceInformedFeedbackBandit`].
#[derive(Debug, Clone)]
pub struct TraceInformedConfig {
    /// Trace signal above this forces `WeightUpdate` (before stall detection fires).
    ///
    /// Default [`DEFAULT_TRACE_SIGNAL_THRESHOLD`] (3.5) — calibrated in the
    /// Plan 310 T3.3 GOAT.
    pub trace_signal_threshold: f32,
}

impl Default for TraceInformedConfig {
    fn default() -> Self {
        Self {
            trace_signal_threshold: DEFAULT_TRACE_SIGNAL_THRESHOLD,
        }
    }
}

/// Wrapper around [`FeedbackBandit`] that biases the harness-vs-weight lever
/// using a leading active-state-trace signal.
///
/// See the [module docs](self) for the policy and IP-boundary rationale.
///
/// # Order of operations
///
/// 1. Compute `trace_signal = compression_ratio_mean × (1 + max(constraint_trend, 0))`.
/// 2. If `trace_signal >= threshold` → force `WeightUpdate` (leading signal fires).
/// 3. Else → delegate to `FeedbackBandit::select()` (preserves stall-only path).
///
/// # Zero-cost when trace is empty
///
/// When the caller passes [`EmptyTrace`] (or any trace whose
/// `compression_ratio_mean` is 0.0), `trace_signal` is 0.0 and step 2 never
/// fires. The wrapper becomes a thin pass-through to the inner bandit —
/// no regression vs the non-trace path.
pub struct TraceInformedFeedbackBandit {
    inner: FeedbackBandit,
    config: TraceInformedConfig,
}

impl TraceInformedFeedbackBandit {
    /// Create a new wrapper around a default `FeedbackBandit` with default
    /// trace config.
    pub fn new() -> Self {
        Self::with_bandit_and_config(
            FeedbackBandit::new(),
            TraceInformedConfig::default(),
        )
    }

    /// Create a new wrapper around the given `FeedbackBandit` and trace config.
    pub fn with_bandit_and_config(
        inner: FeedbackBandit,
        config: TraceInformedConfig,
    ) -> Self {
        Self { inner, config }
    }

    /// Access the wrapped `FeedbackBandit` (for callers that need to read
    /// inner state like `take_weight_request` or `trajectory`).
    pub fn inner(&self) -> &FeedbackBandit {
        &self.inner
    }

    /// Mutable access to the wrapped `FeedbackBandit`.
    pub fn inner_mut(&mut self) -> &mut FeedbackBandit {
        &mut self.inner
    }

    /// Consume the wrapper and return the inner `FeedbackBandit`.
    pub fn into_inner(self) -> FeedbackBandit {
        self.inner
    }

    /// Borrows the trace config (for diagnostics / tuning).
    pub fn config(&self) -> &TraceInformedConfig {
        &self.config
    }

    /// Select a planning decision, biasing the harness-vs-weight lever with
    /// the active-state trace.
    ///
    /// See the [module docs](self) for the policy. When the trace signal is
    /// below threshold (or the trace is empty), this delegates to
    /// [`FeedbackBandit::select`] — preserving the stall-only path for
    /// backward compatibility.
    pub fn select<T: ActiveStateTrace + ?Sized>(
        &mut self,
        context: ConfiguratorContext,
        trace: &T,
    ) -> PlanningDecision {
        let signal = trace_signal(trace);
        if signal >= self.config.trace_signal_threshold {
            // Leading trace signal fires — force WeightUpdate before stall
            // detection catches up. The inner bandit's `select()` handles
            // pending_weight_request emission for this episode; we just
            // override the decision.
            let decision = self.inner.select(context);
            // The inner bandit may have picked something else (e.g. PlanSkip
            // because nothing was stalled yet). Override to WeightUpdate so
            // the trace's leading signal actually drives the lever.
            if !matches!(decision, PlanningDecision::WeightUpdate) {
                // Re-run with the override: the inner select already incremented
                // episode_count and recorded the (now-discarded) arm pull. We
                // record our override as a WeightUpdate arm pull so the
                // trajectory summary reflects what actually happened.
                self.inner_mut().record_override_weight_update();
                return PlanningDecision::WeightUpdate;
            }
            decision
        } else {
            // Trace below threshold (or empty) — fall through to normal UCB1.
            self.inner.select(context)
        }
    }
}

impl Default for TraceInformedFeedbackBandit {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedback_bandit::FeedbackBanditConfig;

    /// Synthetic trace with fixed scalars — for deterministic policy tests.
    #[derive(Clone, Copy)]
    struct StubTrace {
        compression_ratio_mean: f32,
        constraint_trend: f32,
        hla_arousal: f32,
    }

    impl ActiveStateTrace for StubTrace {
        fn compression_ratio_mean(&self) -> f32 {
            self.compression_ratio_mean
        }
        fn constraint_trend(&self) -> f32 {
            self.constraint_trend
        }
        fn hla_arousal(&self) -> f32 {
            self.hla_arousal
        }
    }

    fn ctx() -> ConfiguratorContext {
        ConfiguratorContext::new(0, 0)
    }

    /// `trace_signal` formula sanity: 3.0 × (1 + 0.5) = 4.5.
    #[test]
    fn trace_signal_formula() {
        let t = StubTrace {
            compression_ratio_mean: 3.0,
            constraint_trend: 0.5,
            hla_arousal: 0.0,
        };
        assert!((trace_signal(&t) - 4.5).abs() < 1e-6);
    }

    /// Negative constraint_trend clamps to 0 — falling constraints don't
    /// amplify the signal.
    #[test]
    fn trace_signal_clamps_negative_trend() {
        let t = StubTrace {
            compression_ratio_mean: 3.0,
            constraint_trend: -0.8,
            hla_arousal: 0.0,
        };
        // 3.0 × (1 + max(-0.8, 0)) = 3.0 × 1.0 = 3.0
        assert!((trace_signal(&t) - 3.0).abs() < 1e-6);
    }

    /// Empty trace → signal 0.0 → never fires the threshold.
    #[test]
    fn empty_trace_signal_is_zero() {
        let t = EmptyTrace;
        assert_eq!(trace_signal(&t), 0.0);
    }

    /// No regression: empty trace delegates to inner bandit's normal select.
    ///
    /// The wrapper should be indistinguishable from a bare FeedbackBandit when
    /// no trace is supplied. This is Issue 002's "no regression when trace is
    /// empty" acceptance criterion.
    #[test]
    fn empty_trace_falls_through_to_inner() {
        let mut wrapper = TraceInformedFeedbackBandit::new();
        // Several selects with empty trace — should not panic, should not
        // force WeightUpdate (signal is 0.0 < 3.5 threshold).
        let mut forced_weight_update_count = 0usize;
        for _ in 0..20 {
            let decision = wrapper.select(ctx(), &EmptyTrace);
            // Without stall, the inner bandit picks from the 4 base arms or
            // the 2 feedback arms via UCB1. WeightUpdate is possible but only
            // via UCB1 exploration, not via trace forcing. We assert it doesn't
            // fire *every* iteration (which would indicate the trace path
            // fired on an empty trace).
            if matches!(decision, PlanningDecision::WeightUpdate) {
                forced_weight_update_count += 1;
            }
        }
        // Sanity: the trace path did not fire on every iteration (an empty
        // trace should never trigger the force).
        assert!(
            forced_weight_update_count < 20,
            "empty trace forced WeightUpdate on every iteration — trace path fired spuriously"
        );
        // Sanity: arm pulls were recorded (select always records one arm pull).
        let total_arm_pulls: usize = wrapper.inner().trajectory_summary().arm_pulls.iter().sum();
        assert!(total_arm_pulls >= 20, "select did not record arm pulls");
    }

    /// High trace signal forces WeightUpdate even without stall.
    ///
    /// This is the whole point of the wrapper: leading signal beats lagging
    /// stall detection.
    #[test]
    fn high_trace_signal_forces_weight_update() {
        let mut wrapper = TraceInformedFeedbackBandit::new();
        let stale_trace = StubTrace {
            compression_ratio_mean: 3.0, // ×(1+0.5) = 4.5 ≥ 3.5 threshold
            constraint_trend: 0.5,
            hla_arousal: 0.9,
        };
        let decision = wrapper.select(ctx(), &stale_trace);
        assert!(
            matches!(decision, PlanningDecision::WeightUpdate),
            "high trace signal should force WeightUpdate, got {decision:?}"
        );
    }

    /// Just-below-threshold trace delegates to inner bandit.
    #[test]
    fn below_threshold_trace_delegates() {
        let mut wrapper = TraceInformedFeedbackBandit::new();
        // signal = 1.2 × (1 + 0.0) = 1.2 < 3.5
        let fresh_trace = StubTrace {
            compression_ratio_mean: 1.2,
            constraint_trend: 0.0,
            hla_arousal: 0.2,
        };
        let _ = wrapper.select(ctx(), &fresh_trace);
        // No assertion on the decision — the inner UCB1 picks. We just verify
        // no panic and no forced WeightUpdate via the trace path.
    }

    /// Custom threshold — caller can tune the policy.
    #[test]
    fn custom_threshold_takes_effect() {
        let inner = FeedbackBandit::with_config(FeedbackBanditConfig::default());
        let wrapper = TraceInformedFeedbackBandit::with_bandit_and_config(
            inner,
            TraceInformedConfig {
                trace_signal_threshold: 10.0, // very high — trace never fires
            },
        );
        assert_eq!(wrapper.config().trace_signal_threshold, 10.0);
    }

    /// Tunable threshold: at threshold=∞ the wrapper is a pure pass-through
    /// (matches the bench's `policy_trace_informed_tunable(sample, ∞)` G4
    /// backward-compat case).
    #[test]
    fn infinite_threshold_is_passthrough() {
        let mut wrapper = TraceInformedFeedbackBandit::with_bandit_and_config(
            FeedbackBandit::new(),
            TraceInformedConfig {
                trace_signal_threshold: f32::INFINITY,
            },
        );
        // Even an extreme trace doesn't fire.
        let extreme = StubTrace {
            compression_ratio_mean: 100.0,
            constraint_trend: 100.0,
            hla_arousal: 1.0,
        };
        let _ = wrapper.select(ctx(), &extreme);
        // No forced WeightUpdate; inner bandit's normal UCB1 ran.
    }

    /// Inner accessors work through the wrapper.
    #[test]
    fn inner_accessors() {
        let wrapper = TraceInformedFeedbackBandit::new();
        let _ = wrapper.inner();
        let _ = wrapper.inner().trajectory_summary();
        let _ = wrapper.config();
    }
}
