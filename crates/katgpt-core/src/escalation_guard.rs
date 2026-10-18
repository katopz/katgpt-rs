//! escalation_guard — the shared serving-escalation guard primitives
//! (Issue 923 / riir-refine Plan 202 R1).
//!
//! Every escalation lane in the stack (rethink's ESC lane, refine's future
//! escalation manifest, instinct-hosted seats) needs the same three runtime
//! pieces, and they existed as private per-module copies spelled differently
//! in each repo. This module is the ONE definition, extracted from
//! riir-rethink's `escalating_backend.rs` (the first consumer) so consumers
//! migrate onto it without byte drift:
//!
//! | Concept (sibling spellings — vocabulary translation) | Here |
//! |---|---|
//! | riir-rethink `EscRateGuard` (ring, `observe`/`rate`/`latched`) | [`RollingRateLatch`] |
//! | riir-rethink `RATE_GUARD_WINDOW = 200` | [`DEFAULT_WINDOW`] |
//! | riir-rethink `kill_switch_decode` (`v != Some("0")`, absence-armed) | [`demote_only_decode`] |
//! | katgpt-core `tpr::parse_kill` (`matches!(v, Some("0"))` — same truth table, inverted polarity, private) | same table, cross-pinned by tests |
//! | riir-rethink `TIER_THINK` / `tier_cheap(arm)` / `tier_demoted(arm)` | [`receipt::tier_think`] / [`receipt::tier_cheap`] / [`receipt::tier_demoted`] |
//! | riir-instinct `EscalateSpec` `0 < min_rate < max_rate < 1` grammar check | [`rate_bounds_valid`] |
//!
//! The laws every consumer inherits (Plan 202 §2):
//!
//! - **Manifest-only arming, demote-only kill switches.** No code flip ever
//!   arms escalation; the env decode can only DEMOTE, and only on the exact
//!   literal `"0"` — unset, `"1"`, `"true"`, and junk all leave the lane
//!   armed (a junk value must never silently disarm OR arm anything).
//! - **A partial window is no measurement.** `rate()` is `None` until the
//!   window fills; a guard that never filled never ran (the low-volume
//!   disclosure — cost bounds that work from the FIRST request live at the
//!   consumer, e.g. refine's `llm_spend` daily cap).
//! - **The latch is whole-call, terminal in-process.** Once latched, more
//!   observes never unlatch; re-arming is a manifest edit + re-construction.
//! - **`rate_below_min` exists only on both-bounds lanes.** A cost-ceiling
//!   lane (refine's shape: the escalation rate IS the L0–L3 miss rate,
//!   coverage-decided — low is good) has no lower bound, and a low rate is
//!   never a demotion trigger there.
//!
//! Opt-in (`escalation_guard`). No deps, no unsafe, wasm32-clean by
//! construction (plain arithmetic + `std::env` on the non-hot helper).
//!
//! escalation_guard — the shared serving-escalation guard primitives
//! (Issue 923 / riir-refine Plan 202 R1): extracted ONCE from riir-rethink's
//! ESC lane so every escalation lane in the stack consumes ONE definition —
//! RollingRateLatch (rolling N-decision window over escalation flags, rate()
//! None until full, sticky latched demotion, O(1) alloc-free observe;
//! both-bounds and cost-ceiling-only shapes), the exact-literal demote-only
//! kill-switch decode (absence-armed; ONLY "0" demotes — the manifest-only
//! arming law; tpr::parse_kill is the same truth table at inverted polarity,
//! cross-pinned by tests), and the namespace-parameterized receipt tiers
//! (ESC:think|cheap|demoted — rethink's bytes reproduced exactly). Sibling
//! vocabulary for greps: EscRateGuard, kill_switch_decode,
//! RATE_GUARD_WINDOW, ESC_POSTURES. OPT-IN (`escalation_guard`).

/// The grammar-locked rolling window (riir-rethink Issue 017 T5's bound,
/// the GOAT's [15%, 60%] acceptance axis): the last N decisions'
/// escalation flags.
pub const DEFAULT_WINDOW: usize = 200;

/// The latched-demotion reason when the window rate exceeds `max_rate`.
pub const LATCH_ABOVE_MAX: &str = "rate_above_max";

/// The latched-demotion reason when a both-bounds lane's window rate falls
/// below `min_rate` (never fires on a cost-ceiling lane — there is no
/// lower bound to fall below).
pub const LATCH_BELOW_MIN: &str = "rate_below_min";

/// The shared rate-bounds grammar predicate (riir-instinct
/// `EscalateSpec`'s field grammar, one definition): finite rates with
/// `0 < min_rate < max_rate < 1`.
#[must_use]
pub fn rate_bounds_valid(min_rate: f64, max_rate: f64) -> bool {
    min_rate.is_finite()
        && max_rate.is_finite()
        && 0.0 < min_rate
        && min_rate < max_rate
        && max_rate < 1.0
}

/// The rolling-window rate latch (riir-rethink `EscRateGuard`, one
/// definition): a ring over the last `window` decisions' escalation flags;
/// once the window is full, a rate outside the bounds LATCHES a demotion
/// (sticky, terminal in-process — re-arming is a manifest edit + lane
/// re-construction, the only arming surface).
///
/// Two constructor shapes:
/// - [`RollingRateLatch::new`] — both bounds `[min_rate, max_rate]` (the
///   ESC grammar, `0 < min < max < 1`);
/// - [`RollingRateLatch::new_cost_ceiling`] — `max_rate` alone (refine's
///   shape: the escalation rate is the modelless miss rate, low is good —
///   `rate_below_min` cannot exist).
///
/// `observe` is O(1) and allocation-free (G4-pinned): the ring is
/// allocated once at construction.
#[derive(Debug, Clone)]
pub struct RollingRateLatch {
    ring: Vec<bool>,
    head: usize,
    filled: usize,
    escalations_in_window: usize,
    min_rate: Option<f64>,
    max_rate: f64,
    latched: Option<&'static str>,
}

impl RollingRateLatch {
    /// A both-bounds guard over `[min_rate, max_rate]` at
    /// [`DEFAULT_WINDOW`] — the manifest row's escalate table shape.
    ///
    /// # Panics
    /// Unless [`rate_bounds_valid`] holds (the shared grammar).
    #[must_use]
    pub fn new(min_rate: f64, max_rate: f64) -> Self {
        Self::with_window(DEFAULT_WINDOW, Some(min_rate), max_rate)
    }

    /// A cost-ceiling-only guard at [`DEFAULT_WINDOW`] — refine's shape:
    /// no lower bound, demotion only above `max_rate`.
    ///
    /// # Panics
    /// Unless `max_rate` is finite with `0 < max_rate < 1`.
    #[must_use]
    pub fn new_cost_ceiling(max_rate: f64) -> Self {
        Self::with_window(DEFAULT_WINDOW, None, max_rate)
    }

    /// The full-control constructor (a non-default window; still `> 0`).
    /// `min_rate = None` is the cost-ceiling shape. Validates exactly what
    /// the two named constructors validate — construction-time only, never
    /// hot.
    ///
    /// # Panics
    /// On `window == 0`, a non-finite or out-of-grammar `max_rate`, or a
    /// `min_rate` that fails [`rate_bounds_valid`] against `max_rate`.
    #[must_use]
    pub fn with_window(window: usize, min_rate: Option<f64>, max_rate: f64) -> Self {
        assert!(
            window > 0,
            "escalation_guard: window must be > 0, got {window}"
        );
        assert!(
            max_rate.is_finite() && 0.0 < max_rate && max_rate < 1.0,
            "escalation_guard: max_rate {max_rate} must be finite with 0 < max_rate < 1"
        );
        if let Some(min) = min_rate {
            assert!(
                rate_bounds_valid(min, max_rate),
                "escalation_guard: min_rate {min} / max_rate {max_rate} must satisfy \
                 0 < min < max < 1 (the shared bounds grammar)"
            );
        }
        Self {
            ring: vec![false; window],
            head: 0,
            filled: 0,
            escalations_in_window: 0,
            min_rate,
            max_rate,
            latched: None,
        }
    }

    /// Record one decision's escalation flag. O(1), allocation-free.
    /// The latch evaluation runs only when the window is full.
    pub fn observe(&mut self, escalated: bool) {
        let window = self.ring.len();
        if self.filled < window {
            // Filling: append at the tail; `head` stays 0 (the oldest).
            self.ring[self.filled] = escalated;
            self.filled += 1;
            self.escalations_in_window += usize::from(escalated);
            if self.filled == window {
                self.evaluate();
            }
            return;
        }
        // Rolling: replace the oldest, advance the ring head.
        let old = self.ring[self.head];
        self.ring[self.head] = escalated;
        self.head = (self.head + 1) % window;
        self.escalations_in_window += usize::from(escalated);
        self.escalations_in_window -= usize::from(old);
        self.evaluate();
    }

    /// The window's current escalation rate — `None` until the window is
    /// full (a partial window is no measurement). Exposed so serve paths
    /// can disclose "guard not yet running" instead of implying 0%.
    #[must_use]
    pub fn rate(&self) -> Option<f64> {
        if self.filled < self.ring.len() {
            None
        } else {
            Some(self.escalations_in_window as f64 / self.ring.len() as f64)
        }
    }

    /// The latched demotion's reason, if the guard has fired
    /// ([`LATCH_ABOVE_MAX`] / [`LATCH_BELOW_MIN`]). Sticky: once `Some`,
    /// no further `observe` changes it.
    #[must_use]
    pub fn latched(&self) -> Option<&'static str> {
        self.latched
    }

    /// Convenience: has the demotion latched?
    #[must_use]
    pub fn is_latched(&self) -> bool {
        self.latched.is_some()
    }

    /// The guard's bounds `(min_rate, max_rate)` — `min_rate` is `None` on
    /// a cost-ceiling lane. The loud latch line and the gates read these.
    #[must_use]
    pub fn bounds(&self) -> (Option<f64>, f64) {
        (self.min_rate, self.max_rate)
    }

    /// The rolling window length.
    #[must_use]
    pub fn window(&self) -> usize {
        self.ring.len()
    }

    /// Whether the window has filled (the precondition for `rate()`).
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.filled == self.ring.len()
    }

    fn evaluate(&mut self) {
        if self.latched.is_some() {
            return;
        }
        let Some(rate) = self.rate() else {
            return;
        };
        if rate > self.max_rate {
            self.latched = Some(LATCH_ABOVE_MAX);
        } else if self.min_rate.is_some_and(|min| rate < min) {
            self.latched = Some(LATCH_BELOW_MIN);
        }
    }
}

/// The demote-only kill switch's pure decode (riir-rethink
/// `kill_switch_decode`, one definition; katgpt-core `tpr::parse_kill` is
/// the same table at inverted polarity — cross-pinned by tests both sides).
///
/// **Absence-armed, exact-literal demote**: the manifest is the only ARMING
/// surface, so the env may only DEMOTE — `Some("0")` (the exact literal)
/// demotes, while `None` / `Some("1")` / `Some("true")` / junk all leave
/// the lane armed. Returns `true` when the lane stays ARMED.
#[must_use]
pub fn demote_only_decode(v: Option<&str>) -> bool {
    v != Some("0")
}

/// The wired kill switch: the named env var read through
/// [`demote_only_decode`]. Uncached by design — callers own the read-once
/// policy (construction-time only, never on the decide path; cache with a
/// `OnceLock` if the caller reads it per-call).
#[must_use]
pub fn demote_only_env_armed(env_name: &str) -> bool {
    demote_only_decode(std::env::var(env_name).ok().as_deref())
}

/// The receipt-tier vocabulary — one spelling across the stack's
/// escalation lanes, namespace-parameterized. `ns = "ESC"` reproduces
/// riir-rethink's bytes exactly (`ESC:think`, `ESC:cheap(A1)`,
/// `ESC:demoted(A0)`), so its migration onto these formatters is a pure
/// move with zero byte drift (dual-pinned on both sides).
///
/// Not hot-path: one `String` per served answer.
pub mod receipt {
    /// The receipt tier of an escalated-leg (think-head) answer.
    #[must_use]
    pub fn tier_think(ns: &str) -> String {
        format!("{ns}:think")
    }

    /// The receipt tier of an incumbent-leg (cheap-arm) answer —
    /// `"{ns}:cheap({arm})"`.
    #[must_use]
    pub fn tier_cheap(ns: &str, arm: &str) -> String {
        format!("{ns}:cheap({arm})")
    }

    /// The receipt tier of a post-latch (demoted) answer — the cheap leg's
    /// own arm named as demoted: `"{ns}:demoted({arm})"`. Every receipt
    /// after the rate guard latches carries it.
    #[must_use]
    pub fn tier_demoted(ns: &str, arm: &str) -> String {
        format!("{ns}:demoted({arm})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // G1 — rate semantics
    // ------------------------------------------------------------------

    #[test]
    fn rate_is_none_until_the_window_fills() {
        let mut g = RollingRateLatch::new(0.10, 0.60);
        assert_eq!(g.window(), DEFAULT_WINDOW);
        assert!(!g.is_full());
        for i in 0..DEFAULT_WINDOW - 1 {
            g.observe(i.is_multiple_of(2));
            assert!(g.rate().is_none(), "partial window is no measurement");
        }
        g.observe(true);
        assert!(g.is_full());
        assert!(g.rate().is_some());
    }

    #[test]
    fn rate_counts_escalations_over_the_window() {
        let mut g = RollingRateLatch::new(0.02, 0.98); // never latches
        for i in 0..DEFAULT_WINDOW {
            g.observe(i < 50); // exactly 50 escalations
        }
        let rate = g.rate().expect("full window");
        assert!((rate - 0.25).abs() < 1e-12, "50/200 = 0.25, got {rate}");
        assert!(!g.is_latched());
    }

    #[test]
    fn window_rolls_old_decisions_out() {
        let mut g = RollingRateLatch::with_window(4, Some(0.02), 0.98);
        for e in [true, true, true, false] {
            g.observe(e);
        }
        // 3/4 = 0.75
        assert!((g.rate().unwrap() - 0.75).abs() < 1e-12);
        // Roll in four non-escalations: the 3 trues age out -> 0/4.
        for _ in 0..4 {
            g.observe(false);
        }
        assert!(g.rate().unwrap().abs() < 1e-12);
    }

    #[test]
    fn determinism_same_sequence_same_state() {
        let seq = |i: usize| i.is_multiple_of(3);
        let mut a = RollingRateLatch::new(0.10, 0.60);
        let mut b = RollingRateLatch::new(0.10, 0.60);
        for i in 0..500 {
            let e = seq(i);
            a.observe(e);
            b.observe(e);
        }
        assert_eq!(a.rate(), b.rate());
        assert_eq!(a.latched(), b.latched());
    }

    // ------------------------------------------------------------------
    // G1 — the latch
    // ------------------------------------------------------------------

    #[test]
    fn latch_fires_above_max_and_is_sticky() {
        let mut g = RollingRateLatch::new(0.10, 0.60);
        // 150/200 = 0.75 > 0.60 -> latches the moment the window fills.
        for i in 0..DEFAULT_WINDOW {
            g.observe(i < 150);
        }
        assert_eq!(g.latched(), Some(LATCH_ABOVE_MAX));
        // Sticky: even a long quiet tail never unlatches.
        for _ in 0..1_000 {
            g.observe(false);
        }
        assert_eq!(g.latched(), Some(LATCH_ABOVE_MAX));
    }

    #[test]
    fn latch_fires_below_min_on_both_bounds_lanes() {
        let mut g = RollingRateLatch::new(0.20, 0.60);
        // 10/200 = 0.05 < 0.20.
        for i in 0..DEFAULT_WINDOW {
            g.observe(i < 10);
        }
        assert_eq!(g.latched(), Some(LATCH_BELOW_MIN));
    }

    #[test]
    fn cost_ceiling_lane_has_no_below_min_latch() {
        let mut g = RollingRateLatch::new_cost_ceiling(0.60);
        assert_eq!(g.bounds(), (None, 0.60));
        // ZERO escalations: a both-bounds lane would latch below-min here;
        // a cost-ceiling lane never does (low rate is good, not demotion).
        for _ in 0..DEFAULT_WINDOW {
            g.observe(false);
        }
        assert_eq!(g.rate(), Some(0.0));
        assert!(!g.is_latched(), "rate_below_min cannot exist without a min");
        // And the ceiling still works.
        for i in 0..DEFAULT_WINDOW {
            g.observe(i < 150); // 0.75 > 0.60
        }
        assert_eq!(g.latched(), Some(LATCH_ABOVE_MAX));
    }

    #[test]
    fn in_bounds_rate_never_latches() {
        let mut g = RollingRateLatch::new(0.15, 0.60);
        for i in 0..DEFAULT_WINDOW {
            g.observe(i < 60); // 0.30, inside [0.15, 0.60]
        }
        assert!(!g.is_latched());
    }

    // ------------------------------------------------------------------
    // G1 — constructor grammar
    // ------------------------------------------------------------------

    #[test]
    fn rate_bounds_grammar_truth_table() {
        assert!(rate_bounds_valid(0.15, 0.60));
        assert!(!rate_bounds_valid(0.0, 0.60), "0 < min (the ESC grammar)");
        assert!(!rate_bounds_valid(0.60, 0.60), "min < max");
        assert!(!rate_bounds_valid(0.15, 1.0), "max < 1");
        assert!(!rate_bounds_valid(0.60, 0.15), "ordered");
        assert!(!rate_bounds_valid(f64::NAN, 0.60));
        assert!(!rate_bounds_valid(0.15, f64::INFINITY));
    }

    #[test]
    #[should_panic(expected = "0 < min < max < 1")]
    fn both_bounds_constructor_enforces_the_grammar() {
        let _ = RollingRateLatch::new(0.60, 0.60);
    }

    #[test]
    #[should_panic(expected = "0 < max_rate < 1")]
    fn cost_ceiling_constructor_enforces_the_grammar() {
        let _ = RollingRateLatch::new_cost_ceiling(1.5);
    }

    #[test]
    #[should_panic(expected = "window must be > 0")]
    fn zero_window_refused() {
        let _ = RollingRateLatch::with_window(0, None, 0.5);
    }

    // ------------------------------------------------------------------
    // G1 — the kill-switch truth table (cross-pinned: riir-rethink
    // `kill_switch_decode` + katgpt-core `tpr::parse_kill` semantics)
    // ------------------------------------------------------------------

    #[test]
    fn demote_only_decode_exact_literal_table() {
        // ONLY the exact literal "0" demotes.
        assert!(demote_only_decode(None), "absence is armed");
        assert!(demote_only_decode(Some("1")));
        assert!(demote_only_decode(Some("true")));
        assert!(demote_only_decode(Some("")), "empty string is junk, armed");
        assert!(demote_only_decode(Some("junk")));
        assert!(
            demote_only_decode(Some(" 0")),
            "leading space is not the literal"
        );
        assert!(demote_only_decode(Some("0.0")), "not the literal");
        assert!(!demote_only_decode(Some("0")), "the exact literal demotes");
    }

    #[test]
    fn tpr_parse_kill_is_the_same_table_inverted() {
        // katgpt-core's existing private copy (`tpr::parse_kill`,
        // `matches!(v, Some("0"))`) fires on exactly the value this decode
        // demotes on — one truth table, two polarities, pinned together so
        // neither drifts.
        for v in [
            None,
            Some("1"),
            Some("true"),
            Some(""),
            Some("junk"),
            Some("0"),
        ] {
            assert_eq!(
                demote_only_decode(v),
                !matches!(v, Some("0")),
                "polarity cross-pin at {v:?}"
            );
        }
    }

    // ------------------------------------------------------------------
    // G1 — receipt bytes (the dual-pin anchor: riir-rethink's ESC tiers)
    // ------------------------------------------------------------------

    #[test]
    fn receipt_tiers_reproduce_the_esc_bytes() {
        use super::receipt::*;
        assert_eq!(tier_think("ESC"), "ESC:think");
        assert_eq!(tier_cheap("ESC", "A1"), "ESC:cheap(A1)");
        assert_eq!(tier_demoted("ESC", "A0"), "ESC:demoted(A0)");
    }

    #[test]
    fn receipt_tiers_namespace_parameterized() {
        use super::receipt::*;
        assert_eq!(tier_think("REFINE"), "REFINE:think");
        assert_eq!(
            tier_cheap("REFINE", "clippy_lints"),
            "REFINE:cheap(clippy_lints)"
        );
        assert_eq!(
            tier_demoted("REFINE", "rust_perf"),
            "REFINE:demoted(rust_perf)"
        );
    }

    // ------------------------------------------------------------------
    // G4 — alloc-free observe (post-construction)
    // ------------------------------------------------------------------

    #[cfg(debug_assertions)]
    #[test]
    fn observe_is_allocation_free() {
        // TrackingAllocator (per-thread counters): construction happens
        // BEFORE the reset so only the 1000 observes are counted.
        let mut g = RollingRateLatch::new(0.10, 0.60);
        crate::alloc::reset_alloc_stats();
        for i in 0usize..1_000 {
            g.observe(i.is_multiple_of(2));
        }
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(count, 0, "observe must not allocate (got {count})");
        assert!(g.is_full());
    }
}
