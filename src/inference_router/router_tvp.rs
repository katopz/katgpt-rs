//! Thicket Variance Probe (TVP) router integration — Plan 267.
//!
//! Extracted from `inference_router.rs` (Issue 018) — pure mechanical move,
//! no logic change. All items remain gated by `#[cfg(feature = "thicket_variance_probe")]`.
//!
//! This module holds the `impl InferenceRouter` block containing the
//! TVP-facing API: [`InferenceRouter::update_tvp`],
//! [`InferenceRouter::tvp_signal`], [`InferenceRouter::set_tvp_config`],
//! [`InferenceRouter::tvp_config`], [`InferenceRouter::observe_tvp_decision`].
//!
//! The `TvpSignal` / `TvpConfig` types themselves live in
//! `crate::pruners::thicket_variance_probe` and are re-exported from
//! `inference_router` to preserve the existing public API.
//!
//! When the feature is disabled, this entire module compiles to nothing and
//! the disabled-feature path inside `InferenceRouter::forward()` collapses
//! to `let tier_after_tvp = tier_after_critical;` — zero codegen change
//! (G3 zero-impact guarantee).

#[cfg(feature = "thicket_variance_probe")]
use crate::pruners::thicket_variance_probe::{
    ComputeTier as KpComputeTier, TvpConfig, TvpSignal, TvpTierDecision, tvp_tier_decision,
};
#[cfg(feature = "thicket_variance_probe")]
use crate::trigger_gate::ComputeTier;

#[cfg(feature = "thicket_variance_probe")]
fn tier_to_kp(tier: ComputeTier) -> KpComputeTier {
    match tier {
        ComputeTier::CpuOnly => KpComputeTier::CpuOnly,
        ComputeTier::CpuGpu => KpComputeTier::CpuGpu,
        ComputeTier::CpuGpuAne => KpComputeTier::CpuGpuAne,
    }
}

use crate::inference_router::InferenceRouter;

// ── Thicket Variance Probe API (Plan 267 T11) ────────────────────────────

#[cfg(feature = "thicket_variance_probe")]
impl InferenceRouter {
    /// Observe a TVP signal from the K-probe pre-decode phase.
    ///
    /// Call after the probe-runner completes (e.g., `TvpAggregator::aggregate`).
    /// When the feature is disabled this is a no-op (zero codegen, gate G3).
    ///
    /// Passing `None` clears the signal — useful at query boundaries where the
    /// next query has no probe budget.
    #[inline]
    pub fn update_tvp(&mut self, signal: Option<TvpSignal>) {
        if let Some(s) = signal {
            let changed = self.tvp_signal != Some(s);
            self.tvp_signal = Some(s);
            if changed {
                log::info!(
                    "Router TVP update: reasoning_d={:.4} format_d={:.4} kl={:.4} K={}",
                    s.reasoning_disagreement,
                    s.format_disagreement,
                    s.logit_kl,
                    s.probe_count_used
                );
            }
        } else {
            self.tvp_signal = None;
        }
    }

    /// Get the last observed TVP signal (Plan 267).
    /// Returns `None` if no probes have run yet or the feature is disabled.
    #[inline]
    pub fn tvp_signal(&self) -> Option<TvpSignal> {
        self.tvp_signal
    }

    /// Update TVP config at runtime (Plan 267).
    pub fn set_tvp_config(&mut self, config: TvpConfig) {
        self.tvp_config = config.sanitized();
    }

    /// Get current TVP config (Plan 267).
    pub fn tvp_config(&self) -> TvpConfig {
        self.tvp_config
    }

    /// Compute the TVP tier decision against the current router state.
    ///
    /// Mirrors [`InferenceRouter::observe_critical_entropy`] — call during
    /// `forward()` to get the next-tier decision without mutating state.
    /// Useful for tests and for callers that want to peek at the decision
    /// before committing it.
    ///
    /// Returns `TvpTierDecision::Defer` when no signal has been observed yet
    /// (G3 zero-impact guarantee).
    #[inline]
    pub fn observe_tvp_decision(&self, current_tier: ComputeTier) -> TvpTierDecision {
        let gpu_available = self.gpu.is_some();
        // Demotion only fires under low load (matches trust_signal semantics).
        // Snapshot gate config once to avoid repeated method calls.
        let cfg = self.gate.config();
        let low_load =
            self.gate.estimated_qps() < cfg.gpu_activate_qps * cfg.hysteresis_factor;
        let decision = tvp_tier_decision(
            self.tvp_signal,
            self.tvp_config.promote_at,
            self.tvp_config.demote_at,
            tier_to_kp(current_tier),
            gpu_available,
            low_load,
        );
        if !matches!(decision, TvpTierDecision::Defer | TvpTierDecision::Hold) {
            log::info!(
                "Router TVP decision: {decision:?} (reasoning_d={:?}, promote_at={:.4}, demote_at={:.4}, tier={current_tier})",
                self.tvp_signal.map(|s| s.reasoning_disagreement),
                self.tvp_config.promote_at,
                self.tvp_config.demote_at,
            );
        }
        decision
    }
}
