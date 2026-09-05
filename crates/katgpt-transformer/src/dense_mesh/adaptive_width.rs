//! Adaptive Width Controller — CollapseAware + BreakevenRouter integration.
//!
//! Implements Plan 266 Phase 5: picks between a narrow and a wide
//! [`Topology`] per query, driven by two external signals:
//!
//! 1. **CollapseAwareThinking (Plan 212)** — when the `CollapseDetector`
//!    reports that hesitation is approaching the collapse threshold, the
//!    controller expands the topology width to give the mesh more capacity.
//!    This mirrors the `TvpExpansion` pattern in `S2FCollapseDetector`:
//!    genuine uncertainty expands budget, degenerate repetition contracts it.
//!    The signal here is the **ratio** `hesitation_count / threshold`, gated
//!    by a sigmoid — when it crosses an activation band the controller
//!    switches to the wide topology.
//!
//! 2. **BreakevenRouter (Plan 250)** — when the `BreakevenBandit` reports
//!    that the CPU→GPU upgrade has amortised (`cpu_to_gpu_amortized`), wide
//!    topologies (which cross the `gpu_width_threshold`) become "free" in
//!    the amortised-cost sense, so the controller prefers wide. Conversely,
//!    when no upgrade has amortised, narrow keeps every layer on CPU.
//!
//! # Decision rule
//!
//! Collapse expansion is the **stronger** signal — it reflects observed
//! model state, not just cost amortisation. So:
//!
//! - Collapse signal says **Expand** → wide, regardless of breakeven.
//! - Collapse signal says **Contract** → narrow, regardless of breakeven.
//! - Collapse signal is **Neutral** (no detector, or hesitation ratio in
//!   the dead-zone) → defer to breakeven: wide iff CPU→GPU amortised.
//! - No signals at all → narrow (cheapest baseline, matches gate 1).
//!
//! This is intentionally simple and branch-free on the hot path: a single
//! `match` on `WidthDecision` after both signals have been reduced to a
//! 3-valued enum.
//!
//! # Latent / Raw Compliance
//!
//! Width selection is a pure control-flow decision — it never touches
//! `DenseHidden` contents and never crosses `SyncBlock`. The selected
//! [`Topology`] is then handed to [`LayerwiseTopology`](super::topology)
//! which performs the actual latent forward pass.
//!
//! # References
//!
//! - Plan 266 Phase 5 (this module)
//! - Plan 212 CollapseAwareThinking — `S2FCollapseDetector`
//! - Plan 267 T12 — `TvpExpansion` (the pattern this mirrors)
//! - Plan 250 BreakevenRouter — `BreakevenBandit`

use super::types::Topology;

// ── WidthDecision ───────────────────────────────────────────────────────────

/// Per-query decision on which topology width to use.
///
/// Three-valued so the controller can express "no opinion" and let the
/// other signal (or the default) take over. This keeps the integration
/// decoupled: collapse can be enabled without breakeven and vice versa.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WidthDecision {
    /// Use the narrow topology (cheap, CPU-only).
    Contract,
    /// No signal — defer to the other decider or the default.
    Neutral,
    /// Use the wide topology (more capacity, may cross GPU threshold).
    Expand,
}

impl WidthDecision {
    /// Reduce two signals to a single decision, with collapse taking priority.
    ///
    /// `collapse` is the primary signal — when it is non-`Neutral`, it wins.
    /// Otherwise `breakeven` is consulted. If both are `Neutral`, returns
    /// `Neutral` (caller falls back to the configured default).
    #[inline]
    pub fn combine(collapse: Self, breakeven: Self) -> Self {
        match collapse {
            Self::Contract | Self::Expand => collapse,
            Self::Neutral => breakeven,
        }
    }
}

// ── AdaptiveWidthConfig ─────────────────────────────────────────────────────

/// Configuration for adaptive width selection.
///
/// Built once (cold tier), reused across queries. The two topologies must
/// have the same depth so they can be swapped without rebuilding the
/// [`LayerwiseTopology`](super::topology::LayerwiseTopology) edge matrix —
/// callers that need depth changes should construct a new mesh instead.
#[derive(Clone, Debug)]
pub struct AdaptiveWidthConfig {
    /// Narrow topology — typically `Topology::chain()` (`[1, 1]`).
    pub narrow: Topology,
    /// Wide topology — typically `Topology::wide()` (`[1, 4, 4, 4, 1]`).
    pub wide: Topology,
    /// Hesitation-ratio below which collapse says "Contract".
    ///
    /// Ratio = `hesitation_count / threshold`. Below this band the trace is
    /// healthy → no need to expand width. Default `0.25`.
    pub collapse_contract_below: f32,
    /// Hesitation-ratio above which collapse says "Expand".
    ///
    /// Above this band the trace is approaching collapse → widen the mesh
    /// to give the model more branches to route through. Default `0.75`.
    /// The band `[contract_below, expand_above]` is the dead-zone that
    /// returns `Neutral` — hysteresis against oscillation.
    pub collapse_expand_above: f32,
}

impl Default for AdaptiveWidthConfig {
    fn default() -> Self {
        Self {
            narrow: Topology::chain(),
            wide: Topology::wide(),
            collapse_contract_below: 0.25,
            collapse_expand_above: 0.75,
        }
    }
}

impl AdaptiveWidthConfig {
    /// Pick the topology for a given combined [`WidthDecision`].
    ///
    /// `Neutral` falls back to [`AdaptiveWidthConfig::default_decision`]
    /// (which is `Contract` → narrow, matching gate 1 baseline cost).
    #[inline]
    pub fn select(&self, decision: WidthDecision) -> &Topology {
        match decision {
            WidthDecision::Expand => &self.wide,
            WidthDecision::Contract | WidthDecision::Neutral => &self.narrow,
        }
    }

    /// The default decision when no signal has an opinion.
    ///
    /// Always [`WidthDecision::Contract`] — narrow is the cheapest baseline
    /// and matches the gate 1 correctness proof (`[1,1]` + IdentityEdge).
    #[inline]
    pub const fn default_decision() -> WidthDecision {
        WidthDecision::Contract
    }

    /// Convenience: full pipeline when only the collapse signal is available.
    ///
    /// Collapse-only deployment (no `breakeven_routing` feature). Reduces
    /// the collapse signal to a decision and selects the topology.
    pub fn select_from_collapse(&self, collapse: WidthDecision) -> &Topology {
        let decision = WidthDecision::combine(collapse, WidthDecision::Neutral);
        self.select(decision)
    }

    /// Convenience: full pipeline when only the breakeven signal is available.
    ///
    /// Breakeven-only deployment (no `collapse_aware_thinking` feature).
    pub fn select_from_breakeven(&self, breakeven: WidthDecision) -> &Topology {
        let decision = WidthDecision::combine(WidthDecision::Neutral, breakeven);
        self.select(decision)
    }

    /// Convenience: full pipeline with both signals.
    ///
    /// This is the canonical call site for callers that have both
    /// `collapse_aware_thinking` and `breakeven_routing` enabled.
    pub fn select_from_signals(
        &self,
        collapse: WidthDecision,
        breakeven: WidthDecision,
    ) -> &Topology {
        self.select(WidthDecision::combine(collapse, breakeven))
    }
}

// ── Collapse-aware signal derivation ─────────────────────────────────────────
//
// Feature-gated on `collapse_aware_thinking` — callers without the feature
// compile out the collapse path entirely (zero cost).

/// Derive the collapse-driven width decision from a `CollapseDetector`.
///
/// Reads `hesitation_count()` and `threshold()` and returns:
///
/// - `Contract` when `hesitation_count / threshold < collapse_contract_below`
/// - `Expand`   when `hesitation_count / threshold > collapse_expand_above`
/// - `Neutral`  in the dead-zone between the two.
///
/// `threshold == 0` is treated as "no collapse history yet" → `Neutral`
/// (avoids divide-by-zero; lets the breakeven signal or default take over).
///
/// Zero-allocation: two scalar reads + one divide + two compares.
#[cfg(feature = "collapse_aware_thinking")]
pub fn collapse_signal(
    detector: &dyn katgpt_core::traits::CollapseDetector,
    config: &AdaptiveWidthConfig,
) -> WidthDecision {
    let threshold = detector.threshold();
    if threshold == 0 {
        return WidthDecision::Neutral;
    }
    let ratio = detector.hesitation_count() as f32 / threshold as f32;
    if ratio > config.collapse_expand_above {
        WidthDecision::Expand
    } else if ratio < config.collapse_contract_below {
        WidthDecision::Contract
    } else {
        WidthDecision::Neutral
    }
}

// ── Breakeven signal derivation ──────────────────────────────────────────────
//
// Feature-gated on `breakeven_routing` — callers without the feature compile
// out the breakeven path entirely.

/// Snapshot of the breakeven state needed to make a width decision.
///
/// We accept a small struct rather than `&BreakevenBandit` so this module
/// stays decoupled from the full bandit API and can be unit-tested in
/// isolation. Callers construct this from `BreakevenBandit::stats()`.
#[cfg(feature = "breakeven_routing")]
#[derive(Clone, Copy, Debug, Default)]
pub struct BreakevenSnapshot {
    /// Has the CPU→GPU upgrade amortised? When true, wide topologies
    /// (which cross `gpu_width_threshold`) run at amortised GPU cost.
    pub cpu_to_gpu_amortized: bool,
}

/// Derive the breakeven-driven width decision.
///
/// Returns:
/// - `Expand` when the CPU→GPU upgrade has amortised → wide topology
///   (which crosses the GPU width threshold) is now cost-effective.
/// - `Contract` when it has not amortised → narrow keeps everything on
///   CPU, avoiding the GPU launch overhead (~50μs).
/// - `Neutral` is never returned by this signal (it's always binary).
///
/// Breakeven is a **cost** signal — it does not know whether the model
/// needs more capacity. Collapse is the **quality** signal. When collapse
/// has no opinion (`Neutral`), breakeven wins.
#[cfg(feature = "breakeven_routing")]
pub fn breakeven_signal(snapshot: &BreakevenSnapshot) -> WidthDecision {
    if snapshot.cpu_to_gpu_amortized {
        WidthDecision::Expand
    } else {
        WidthDecision::Contract
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── WidthDecision::combine ──────────────────────────────────────────────

    #[test]
    fn combine_collapse_wins_when_non_neutral() {
        // Collapse is the primary signal — non-Neutral collapse always wins.
        assert_eq!(
            WidthDecision::combine(WidthDecision::Contract, WidthDecision::Expand),
            WidthDecision::Contract
        );
        assert_eq!(
            WidthDecision::combine(WidthDecision::Expand, WidthDecision::Contract),
            WidthDecision::Expand
        );
    }

    #[test]
    fn combine_falls_through_to_breakeven_when_collapse_neutral() {
        assert_eq!(
            WidthDecision::combine(WidthDecision::Neutral, WidthDecision::Expand),
            WidthDecision::Expand
        );
        assert_eq!(
            WidthDecision::combine(WidthDecision::Neutral, WidthDecision::Contract),
            WidthDecision::Contract
        );
    }

    #[test]
    fn combine_returns_neutral_when_both_neutral() {
        assert_eq!(
            WidthDecision::combine(WidthDecision::Neutral, WidthDecision::Neutral),
            WidthDecision::Neutral
        );
    }

    // ── AdaptiveWidthConfig::select ─────────────────────────────────────────

    #[test]
    fn select_expand_returns_wide() {
        let cfg = AdaptiveWidthConfig::default();
        assert!(std::ptr::addr_eq(
            cfg.select(WidthDecision::Expand),
            &cfg.wide
        ));
    }

    #[test]
    fn select_contract_returns_narrow() {
        let cfg = AdaptiveWidthConfig::default();
        assert!(std::ptr::addr_eq(
            cfg.select(WidthDecision::Contract),
            &cfg.narrow
        ));
    }

    #[test]
    fn select_neutral_falls_back_to_narrow() {
        // Neutral must not expand — that's the cheapest-baseline contract.
        let cfg = AdaptiveWidthConfig::default();
        assert!(std::ptr::addr_eq(
            cfg.select(WidthDecision::Neutral),
            &cfg.narrow
        ));
    }

    #[test]
    fn default_decision_is_contract() {
        assert_eq!(
            AdaptiveWidthConfig::default_decision(),
            WidthDecision::Contract
        );
    }

    #[test]
    fn select_from_signals_combines_correctly() {
        let cfg = AdaptiveWidthConfig::default();

        // Collapse Expand overrides breakeven Contract.
        let topo = cfg.select_from_signals(WidthDecision::Expand, WidthDecision::Contract);
        assert!(std::ptr::addr_eq(topo, &cfg.wide));

        // Collapse Neutral + breakeven Expand → wide.
        let topo = cfg.select_from_signals(WidthDecision::Neutral, WidthDecision::Expand);
        assert!(std::ptr::addr_eq(topo, &cfg.wide));

        // Collapse Neutral + breakeven Contract → narrow.
        let topo = cfg.select_from_signals(WidthDecision::Neutral, WidthDecision::Contract);
        assert!(std::ptr::addr_eq(topo, &cfg.narrow));
    }

    #[test]
    fn select_from_collapse_only_path() {
        let cfg = AdaptiveWidthConfig::default();
        assert!(std::ptr::addr_eq(
            cfg.select_from_collapse(WidthDecision::Expand),
            &cfg.wide
        ));
        assert!(std::ptr::addr_eq(
            cfg.select_from_collapse(WidthDecision::Contract),
            &cfg.narrow
        ));
        // Neutral collapse with no breakeven → default (Contract → narrow).
        assert!(std::ptr::addr_eq(
            cfg.select_from_collapse(WidthDecision::Neutral),
            &cfg.narrow
        ));
    }

    #[test]
    fn select_from_breakeven_only_path() {
        let cfg = AdaptiveWidthConfig::default();
        assert!(std::ptr::addr_eq(
            cfg.select_from_breakeven(WidthDecision::Expand),
            &cfg.wide
        ));
        assert!(std::ptr::addr_eq(
            cfg.select_from_breakeven(WidthDecision::Contract),
            &cfg.narrow
        ));
    }

    // ── Hysteresis band ─────────────────────────────────────────────────────

    #[test]
    fn default_hysteresis_band_is_sensible() {
        let cfg = AdaptiveWidthConfig::default();
        // Contract below 0.25, Expand above 0.75, dead-zone in between.
        assert!(cfg.collapse_contract_below < cfg.collapse_expand_above);
        assert!(cfg.collapse_contract_below > 0.0);
        assert!(cfg.collapse_expand_above < 1.0);
        // Dead-zone must be non-empty (otherwise no Neutral state possible).
        assert!(
            cfg.collapse_expand_above - cfg.collapse_contract_below > 0.1,
            "dead-zone too narrow, risks oscillation"
        );
    }

    // ── Collapse signal (feature-gated) ─────────────────────────────────────

    #[cfg(feature = "collapse_aware_thinking")]
    mod collapse_integration {
        use super::*;
        use katgpt_core::traits::CollapseDetector;

        /// Stub detector returning canned values, for unit testing.
        struct StubDetector {
            hesitation: u32,
            thresh: u32,
        }

        impl CollapseDetector for StubDetector {
            fn check_collapse(&mut self, _token_id: u32, _position: usize) -> bool {
                false
            }
            fn reset(&mut self) {}
            #[inline]
            fn hesitation_count(&self) -> u32 {
                self.hesitation
            }
            #[inline]
            fn threshold(&self) -> u32 {
                self.thresh
            }
        }

        #[test]
        fn collapse_threshold_zero_returns_neutral() {
            // Avoids divide-by-zero; lets breakeven or default take over.
            let cfg = AdaptiveWidthConfig::default();
            let det = StubDetector {
                hesitation: 0,
                thresh: 0,
            };
            assert_eq!(collapse_signal(&det, &cfg), WidthDecision::Neutral);
        }

        #[test]
        fn collapse_low_ratio_contracts() {
            // ratio = 1/10 = 0.1 < 0.25 → Contract.
            let cfg = AdaptiveWidthConfig::default();
            let det = StubDetector {
                hesitation: 1,
                thresh: 10,
            };
            assert_eq!(collapse_signal(&det, &cfg), WidthDecision::Contract);
        }

        #[test]
        fn collapse_high_ratio_expands() {
            // ratio = 9/10 = 0.9 > 0.75 → Expand.
            let cfg = AdaptiveWidthConfig::default();
            let det = StubDetector {
                hesitation: 9,
                thresh: 10,
            };
            assert_eq!(collapse_signal(&det, &cfg), WidthDecision::Expand);
        }

        #[test]
        fn collapse_dead_zone_returns_neutral() {
            // ratio = 5/10 = 0.5 — inside [0.25, 0.75] dead-zone.
            let cfg = AdaptiveWidthConfig::default();
            let det = StubDetector {
                hesitation: 5,
                thresh: 10,
            };
            assert_eq!(collapse_signal(&det, &cfg), WidthDecision::Neutral);
        }

        #[test]
        fn collapse_custom_band_respected() {
            // Tighter band: contract below 0.1, expand above 0.5.
            let cfg = AdaptiveWidthConfig {
                collapse_contract_below: 0.1,
                collapse_expand_above: 0.5,
                ..AdaptiveWidthConfig::default()
            };
            // ratio = 0.3 → inside [0.1, 0.5] → Neutral.
            let det = StubDetector {
                hesitation: 3,
                thresh: 10,
            };
            assert_eq!(collapse_signal(&det, &cfg), WidthDecision::Neutral);
            // ratio = 0.6 → above 0.5 → Expand.
            let det = StubDetector {
                hesitation: 6,
                thresh: 10,
            };
            assert_eq!(collapse_signal(&det, &cfg), WidthDecision::Expand);
        }

        #[test]
        fn end_to_end_collapse_only_pipeline() {
            // High hesitation + default config → wide topology selected.
            let cfg = AdaptiveWidthConfig::default();
            let det = StubDetector {
                hesitation: 9,
                thresh: 10,
            };
            let topo = cfg.select_from_collapse(collapse_signal(&det, &cfg));
            assert!(std::ptr::addr_eq(topo, &cfg.wide));
        }
    }

    // ── Breakeven signal (feature-gated) ────────────────────────────────────

    #[cfg(feature = "breakeven_routing")]
    mod breakeven_integration {
        use super::*;

        #[test]
        fn breakeven_amortized_expands() {
            let snap = BreakevenSnapshot {
                cpu_to_gpu_amortized: true,
            };
            assert_eq!(breakeven_signal(&snap), WidthDecision::Expand);
        }

        #[test]
        fn breakeven_not_amortized_contracts() {
            let snap = BreakevenSnapshot {
                cpu_to_gpu_amortized: false,
            };
            assert_eq!(breakeven_signal(&snap), WidthDecision::Contract);
        }

        #[test]
        fn breakeven_default_snapshot_contracts() {
            // Default snapshot has cpu_to_gpu_amortized = false.
            let snap = BreakevenSnapshot::default();
            assert_eq!(breakeven_signal(&snap), WidthDecision::Contract);
        }

        #[test]
        fn end_to_end_breakeven_only_pipeline() {
            let cfg = AdaptiveWidthConfig::default();
            let snap = BreakevenSnapshot {
                cpu_to_gpu_amortized: true,
            };
            let topo = cfg.select_from_breakeven(breakeven_signal(&snap));
            assert!(std::ptr::addr_eq(topo, &cfg.wide));
        }

        #[test]
        fn end_to_end_both_signals_collapse_wins() {
            // Collapse says Contract (low hesitation) but breakeven says Expand.
            // Collapse is the primary signal → Contract → narrow.
            let cfg = AdaptiveWidthConfig::default();
            let collapse = WidthDecision::Contract;
            let breakeven = WidthDecision::Expand;
            let topo = cfg.select_from_signals(collapse, breakeven);
            assert!(std::ptr::addr_eq(topo, &cfg.narrow));
        }
    }
}
