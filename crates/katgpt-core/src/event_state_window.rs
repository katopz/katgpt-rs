//! Event-anchored two-sided sigmoid state window + identity-free population
//! state summary (Plan 623 Phase 1 — the TWS distillation, Research 611 /
//! riir-ai Research 396; source paper arXiv:2610.03001).
//!
//! # The window (TWS Eq. 1, general interior-state form)
//!
//! A state lives between two semantic events: its left edge opens at the
//! PREVIOUS boundary's settle offset `τ = b_prev + delta`, its right edge
//! closes at the NEXT boundary `b_next` (the next transition's settle window
//! `[b_next, b_next + Δ_next)` is excluded from BOTH states — it belongs to
//! no state):
//!
//! ```text
//! w(t) = σ((t − (b_prev + delta)) / γ) · σ((b_next − t) / γ)
//! ```
//!
//! For K adjacent states sharing boundaries, `Σ_k w_k(t) ≈ 1` inside state
//! cores and `< 1` exactly on the excluded transitions — the population
//! "settles" after each event and belongs to no state while it does. Δ is a
//! PARAMETER of this primitive by design (Plan 623: no `derive_settle_offset`
//! wrapper — callers derive it at wiring time, e.g. `settling_ticks(β, ε)`
//! from `habituation_filter`, whose β is the update rate of the statistic
//! being windowed).
//!
//! Sigmoid choice: [`crate::simd::fast_sigmoid`] (the always-on simd
//! substrate, ±40 saturation → the γ→0 limit saturates to EXACT 0/1, which
//! the G1 limit tests pin). Latency-class primitive: the G2 bar is ≤ 20 ns
//! per call.
//!
//! # The summary (identity-free population token half)
//!
//! [`population_state_summary`] computes order statistics (mean, spread,
//! 10/25/50/75/90% nearest-rank quantiles with tail supports) over an
//! UNORDERED member set. The transfer law (Research 611 §design-law): any
//! aggregate compared across populations must consume no member identity —
//! order statistics are identity-free by construction. All statistics are
//! computed over the SORTED scratch buffer: the sort is already paid for the
//! quantiles, and summation over a fixed (sorted) order is
//! permutation-invariant BY CONSTRUCTION (f32 addition is non-associative —
//! input-order summation is not bit-invariant under permutation). Zero-alloc:
//! the caller owns the scratch buffer.
//!
//! # Precedent checked (substrate-first, Plan 623 T0)
//!
//! No existing two-sided event-anchored window or population-summary module
//! ships (12-variant vocabulary sweep, 2026-10-10). Consumed substrate:
//! [`crate::simd::fast_sigmoid`], [`crate::stats::nearest_rank`] (the
//! workspace percentile-of-record — its `(value, tail_support)` return is
//! carried verbatim in [`StateSummary`]). Nearest neighbors ruled different:
//! `katgpt-kv::drift_segment::sigmoid_gated_readout` (slot-gated, not
//! time-anchored), `kinematics::perception::Regime` (motion classification),
//! `habituation_filter` (anchorless EMA).

#![cfg(feature = "event_state_windows")]

use crate::simd::fast_sigmoid;

/// The two-sided sigmoid product window (TWS Eq. 1 general interior-state
/// form): rises at the previous boundary's settle offset, falls at the next
/// boundary.
///
/// * `t` — the query tick.
/// * `b_prev` — the previous event boundary.
/// * `b_next` — the next event boundary (shared with the following state).
/// * `delta` — the settle offset Δ applied AFTER `b_prev` (the transition
///   `[b_prev, b_prev + delta)` is excluded from this state).
/// * `gamma` — the sigmoid width γ > 0 (softness of both edges).
///
/// Totality contract (the T1.4 fuzz law): malformed input returns the ZERO
/// window, never a panic and never a negative/garbage value — γ ≤ 0 or NaN,
/// an empty or inverted support (`b_prev + delta ≥ b_next`, NaN included via
/// failed comparisons), or a non-finite `t` all yield exactly `0.0`.
/// Well-formed finite input yields a value in `[0, 1]`.
#[inline]
#[must_use]
pub fn event_state_window(t: f32, b_prev: f32, b_next: f32, delta: f32, gamma: f32) -> f32 {
    let lo = b_prev + delta;
    if (gamma > 0.0) && (b_next > lo) && t.is_finite() {
        fast_sigmoid((t - lo) / gamma) * fast_sigmoid((b_next - t) / gamma)
    } else {
        // Malformed: γ ≤ 0/NaN, empty/inverted support, or non-finite t.
        // NaN comparisons fail closed into this branch — the zero window.
        0.0
    }
}

/// Identity-free order statistics over an unordered member set — one
/// population token per event-delimited state.
///
/// Quantiles are the workspace's nearest-rank convention
/// ([`crate::stats::nearest_rank`], ceiling form, upper median at even n);
/// each carries its TAIL SUPPORT — `1` means "this is the max of a sample
/// too small for the percentile's name", `≥ 10` means real footing. `n` is
/// the population size (0 = the all-zero empty summary).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateSummary {
    /// Arithmetic mean over the sorted buffer (f64-accumulated, narrowed).
    pub mean: f32,
    /// Population standard deviation (÷n, two-pass over the sorted buffer).
    pub spread: f32,
    /// 10th-percentile nearest rank.
    pub p10: f32,
    /// 25th-percentile nearest rank (lower quartile, upper-median convention).
    pub p25: f32,
    /// 50th-percentile nearest rank (the upper median).
    pub p50: f32,
    /// 75th-percentile nearest rank.
    pub p75: f32,
    /// 90th-percentile nearest rank.
    pub p90: f32,
    /// Tail supports of `[p10, p25, p50, p75, p90]` in order (`n - rank_idx`).
    pub supports: [usize; 5],
    /// Population size.
    pub n: usize,
}

impl StateSummary {
    /// The empty-population summary (all zeros, `n = 0`).
    pub const EMPTY: Self = Self {
        mean: 0.0,
        spread: 0.0,
        p10: 0.0,
        p25: 0.0,
        p50: 0.0,
        p75: 0.0,
        p90: 0.0,
        supports: [0; 5],
        n: 0,
    };
}

/// Compute the identity-free [`StateSummary`] over `values`, using
/// `scratch` as the sort workspace.
///
/// Zero-alloc: nothing here allocates — the caller owns `scratch`, which
/// must be at least `values.len()` long (contract; a shorter scratch is a
/// caller bug and asserts loud in debug). The input order is IRRELEVANT to
/// the result bit-for-bit: values are copied into the scratch, sorted under
/// `f32::total_cmp` (total order, NaN last, deterministic), and every
/// statistic is computed over the sorted buffer in one fixed order.
///
/// NaN members sort last and propagate into the statistics honestly (NaN in
/// → NaN mean) — the summary reports what it was given; it never panics.
#[must_use]
pub fn population_state_summary(values: &[f32], scratch: &mut [f32]) -> StateSummary {
    let n = values.len();
    if n == 0 {
        return StateSummary::EMPTY;
    }
    debug_assert!(
        scratch.len() >= n,
        "population_state_summary: scratch.len() {} < values.len() {n}",
        scratch.len(),
    );
    let buf = &mut scratch[..n];
    buf.copy_from_slice(values);
    buf.sort_unstable_by(f32::total_cmp);

    // Mean over the sorted order — the fixed order makes the f64 fold
    // deterministic, which is what permutation-invariance is bought with.
    let mut sum = 0.0f64;
    for &v in buf.iter() {
        sum += f64::from(v);
    }
    let mean = sum / n as f64;

    // Two-pass population variance over the same sorted order.
    let mut var_acc = 0.0f64;
    for &v in buf.iter() {
        let d = f64::from(v) - mean;
        var_acc += d * d;
    }
    let spread = (var_acc / n as f64).sqrt();

    let (p10, s10) = crate::stats::nearest_rank(buf, 0.10);
    let (p25, s25) = crate::stats::nearest_rank(buf, 0.25);
    let (p50, s50) = crate::stats::nearest_rank(buf, 0.50);
    let (p75, s75) = crate::stats::nearest_rank(buf, 0.75);
    let (p90, s90) = crate::stats::nearest_rank(buf, 0.90);

    StateSummary {
        mean: mean as f32,
        spread: spread as f32,
        p10,
        p25,
        p50,
        p75,
        p90,
        supports: [s10, s25, s50, s75, s90],
        n,
    }
}

#[cfg(test)]
mod tests {
    use super::{event_state_window, population_state_summary, StateSummary};

    /// Deterministic LCG (the global-RNG gate's seeded-local law).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn f32_unit(&mut self) -> f32 {
            ((self.next() >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    // ── G1: partition of unity over K adjacent states ───────────────────

    /// K states sharing boundaries: state k has (b_{k-1}, delta_{k-1}, b_k).
    /// Terminals are composed through the SAME two-sided fn with far-away
    /// anchors (the rise/fall side saturates to exactly 1.0 past ±40γ).
    #[test]
    fn partition_of_unity_inside_states_below_one_on_transitions() {
        let gamma = 5.0f32;
        // Boundaries at 0 / 100 / 200 / 300 with settle offsets 20 after
        // each interior boundary.
        let b = [0.0f32, 100.0, 200.0, 300.0];
        let delta = [1000.0f32, 20.0, 20.0]; // delta[k] applies after b[k]
        let w = |k: usize, t: f32| match k {
            0 => event_state_window(t, b[0] - 1.0e4, b[1], 0.0, gamma),
            1 => event_state_window(t, b[1], b[2], delta[1], gamma),
            2 => event_state_window(t, b[2], b[3], delta[2], gamma),
            _ => event_state_window(t, b[3], b[3] + 1.0e4, delta[2] + 1.0e4, gamma),
            // right terminal: rise saturates (anchor 1e4 below), fall edge
            // sits 1e4 above → flat 1.0 over the whole range — the far side
            // of the LAST boundary is the terminal state's interior.
        };
        // Inside each state's core (midpoint between settle edge and next
        // boundary): sum ≈ 1.
        for (k, core_mid) in [
            (0, (b[0] + b[1]) * 0.5),
            (1, (b[1] + delta[1] + b[2]) * 0.5),
            (2, (b[2] + delta[2] + b[3]) * 0.5),
        ] {
            let sum: f32 = (0..3).map(|k| w(k, core_mid)).sum();
            assert!(
                (sum - 1.0).abs() < 1.0e-3,
                "partition inside state {k} core at t={core_mid}: sum={sum}"
            );
            // …and the state's own window carries essentially all of it.
            assert!(w(k, core_mid) > 0.99, "own-state mass at core {k}");
        }
        // On each excluded transition [b_k, b_k + delta_k): the sum is
        // STRICTLY below 1 — the settle window belongs to no state.
        for k in 1..3usize {
            let mid = b[k] + delta[k] * 0.5;
            let sum: f32 = (0..3).map(|s| w(s, mid)).sum();
            assert!(
                sum < 0.5,
                "transition [{}, {}) at t={mid}: sum={sum} — exclusion must be visible",
                b[k],
                b[k] + delta[k],
            );
        }
    }

    /// The literal exclusion pin (T1.2): at `t` strictly inside
    /// `[b, b + delta)` EVERY state's window is below one half — no state
    /// claims the settle transition.
    #[test]
    fn settle_transition_excluded_from_both_neighbors() {
        let gamma = 5.0f32;
        let (bp, bn, d) = (0.0f32, 200.0, 20.0f32);
        // Strictly-interior transition points — the exact edges are the
        // σ(0)=0.5 ambiguous points and carry no state either (sum ≈ 0.52),
        // but the per-state pin reads cleanest strictly inside (b, b+Δ).
        for t in [0.001f32, 5.0, 10.0, 15.0, 19.999] {
            let left = event_state_window(t, bp - 1.0e4, 0.0, 0.0, gamma);
            // left state's fall edge AT b=0: sigma((0 - t)/gamma) < 1/2.
            assert!(
                left < 0.5,
                "left state claims settle t={t}: w={left}"
            );
            let right = event_state_window(t, 0.0, 1.0e4, d, gamma);
            // right state's rise edge AT b+delta=20: sigma((t-20)/gamma) < 1/2.
            assert!(right < 0.5, "right state claims settle t={t}: w={right}");
            let _ = bn; // bn only bounds the composite; edges above are the pin
        }
    }

    // ── G1: γ limits ────────────────────────────────────────────────────

    #[test]
    fn gamma_to_zero_is_the_hard_indicator() {
        let (bp, bn, d) = (0.0f32, 100.0, 10.0f32);
        let g = 1.0e-3f32;
        // Strictly inside (b+delta, b_next) by ≫40γ: both sigmoids saturate
        // to EXACT 1.0 (fast_sigmoid's ±40 saturation) → exactly 1.0.
        for t in [10.5f32, 25.0, 50.0, 99.5] {
            assert_eq!(
                event_state_window(t, bp, bn, d, g),
                1.0,
                "hard indicator inside at t={t}"
            );
        }
        // Strictly outside the support by ≫40γ: exactly 0.0. (The exact edge
        // points b+Δ and b_next are the σ(0)=0.5 ambiguous jump points — the
        // hard-indicator limit is undefined THERE, by construction.)
        for t in [-1.0f32, 5.0, 9.5, 100.5, 150.0] {
            assert_eq!(
                event_state_window(t, bp, bn, d, g),
                0.0,
                "hard indicator outside at t={t}"
            );
        }
    }

    #[test]
    fn gamma_to_infinity_is_flat_and_degenerate() {
        let (bp, bn, d) = (0.0f32, 100.0, 10.0f32);
        let g = 1.0e9f32;
        let w1 = event_state_window(20.0, bp, bn, d, g);
        let w2 = event_state_window(80.0, bp, bn, d, g);
        assert!(
            (w1 - w2).abs() < 1.0e-6,
            "γ→∞ must be t-independent (flat): w(20)={w1} w(80)={w2}"
        );
        // σ(≈0)·σ(≈0) ≈ 0.25 — the degenerate quarter-window.
        assert!((w1 - 0.25).abs() < 1.0e-6, "flat limit value {w1}");
    }

    // ── T1.4: malformed boundaries — zero window, never panic ───────────

    #[test]
    fn malformed_boundaries_return_zero_never_panic() {
        // Inverted / empty support: b_prev + delta >= b_next.
        assert_eq!(event_state_window(50.0, 100.0, 100.0, 0.0, 5.0), 0.0);
        assert_eq!(event_state_window(50.0, 100.0, 50.0, 10.0, 5.0), 0.0);
        assert_eq!(event_state_window(50.0, 0.0, 100.0, 100.0, 5.0), 0.0);
        // Degenerate width.
        assert_eq!(event_state_window(50.0, 0.0, 100.0, 10.0, 0.0), 0.0);
        assert_eq!(event_state_window(50.0, 0.0, 100.0, 10.0, -1.0), 0.0);
        // NaN / Inf — malformed slots one at a time. (b_next = +inf is NOT
        // here: an infinite right boundary is the VALID terminal shape — the
        // window is the open-ended rise; see the partition test's anchors.)
        let nan = f32::NAN;
        let inf = f32::INFINITY;
        for (t, bp, bn, d, g) in [
            (nan, 0.0, 100.0, 10.0, 5.0),
            (inf, 0.0, 100.0, 10.0, 5.0),
            (-inf, 0.0, 100.0, 10.0, 5.0),
            (50.0, nan, 100.0, 10.0, 5.0),
            (50.0, 0.0, nan, 10.0, 5.0),
            (50.0, 0.0, 100.0, nan, 5.0),
            (50.0, 0.0, 100.0, 10.0, nan),
            (50.0, inf, 100.0, 10.0, 5.0),
            (50.0, -inf, -inf, 0.0, 5.0),
        ] {
            let w = event_state_window(t, bp, bn, d, g);
            assert_eq!(w, 0.0, "malformed ({t},{bp},{bn},{d},{g}) → 0.0, got {w}");
        }
    }

    #[test]
    fn fuzz_totality_output_bounded_and_refusal_exact() {
        let mut rng = Lcg(0x6230_0001);
        let mut checked = 0usize;
        for i in 0..20_000u32 {
            let t = rng.f32_unit() * 1.0e4;
            let bp = rng.f32_unit() * 1.0e3;
            let bn = bp + 10.0 + rng.f32_unit().abs() * 500.0;
            let d = rng.f32_unit() * 200.0;
            let g = {
                let u = rng.f32_unit();
                if i % 97 == 0 {
                    // inject degenerate widths at a known rate
                    if i % 3 == 0 {
                        0.0
                    } else {
                        -u.abs()
                    }
                } else {
                    0.01 + u.abs() * 50.0
                }
            };
            let w = event_state_window(t, bp, bn, d, g);
            let well_formed = (g > 0.0) && (bn > bp + d);
            if !well_formed {
                assert_eq!(w, 0.0, "malformed combo must return EXACT zero");
            } else {
                assert!(
                    (0.0..=1.0).contains(&w),
                    "well-formed window out of [0,1]: {w}"
                );
                checked += 1;
            }
        }
        assert!(checked > 15_000, "fuzz must exercise the live path too");
    }

    // ── G1: summary closed-form + permutation bit-identity ──────────────

    #[test]
    fn summary_closed_form_known_values() {
        // 1..=100 — the cleanest closed form there is.
        let values: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let mut scratch = vec![0.0f32; 100];
        let s = population_state_summary(&values, &mut scratch);
        assert_eq!(s.n, 100);
        assert_eq!(s.mean, 50.5);
        // Population std of 1..=100 = sqrt((100²−1)/12) = sqrt(837)/1 → 28.866…
        assert!((s.spread - 28.866_07).abs() < 1.0e-4, "spread {}", s.spread);
        // nearest_rank ceiling form over the ascending buffer.
        assert_eq!(s.p10, 10.0); // ceil(0.10·100)=10 → the 10th value
        assert_eq!(s.p25, 25.0);
        assert_eq!(s.p50, 50.0); // upper median at even n — the house convention
        assert_eq!(s.p75, 75.0);
        assert_eq!(s.p90, 90.0);
        assert_eq!(s.supports, [91, 76, 51, 26, 11]);
    }

    #[test]
    fn summary_permutation_invariance_bit_identical() {
        let mut rng = Lcg(0x6230_0002);
        let n = 257usize; // odd, so the median rank is exact
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            values.push(rng.f32_unit() * 1000.0);
        }
        // Duplicates + extremes — equal keys and saturating magnitudes must
        // not disturb the fixed-order fold.
        values[0] = f32::MAX;
        values[1] = -f32::MAX;
        values[2] = 0.0;
        values[3] = values[4];
        let mut scratch = vec![0.0f32; n];
        let a = population_state_summary(&values, &mut scratch);

        // Deterministic derangement: reverse + swap halved pairs.
        values.reverse();
        for i in (0..n - 1).step_by(2) {
            values.swap(i, i + 1);
        }
        let b = population_state_summary(&values, &mut scratch);
        assert_eq!(a, b, "bit-identical under permutation (PartialEq on f32 is bit equality for these fields)");
    }

    #[test]
    fn summary_empty_and_single() {
        let mut scratch: Vec<f32> = Vec::new();
        assert_eq!(population_state_summary(&[], &mut scratch), StateSummary::EMPTY);
        let mut one = [0.0f32; 1];
        let s = population_state_summary(&[42.0], &mut one);
        assert_eq!(s.n, 1);
        assert_eq!(s.mean, 42.0);
        assert_eq!(s.spread, 0.0);
        assert_eq!(s.p50, 42.0);
        assert_eq!(s.supports, [1, 1, 1, 1, 1], "n=1: every percentile is the max — support says so");
    }

    #[test]
    fn summary_nan_members_sort_last_and_propagate() {
        let values = [1.0f32, f32::NAN, 3.0];
        let mut scratch = [0.0f32; 3];
        let s = population_state_summary(&values, &mut scratch);
        assert!(s.mean.is_nan(), "NaN propagates honestly");
        // NaN sorts LAST (total_cmp): the median rank of 3 is 3.0; the NaN
        // IS the p90 (the top rank).
        assert_eq!(s.p50, 3.0, "NaN above the median rank, not at it");
        assert!(s.p90.is_nan(), "p90 is the top rank — the NaN");
        // No panic anywhere is the contract; the values above pin the shape.
    }
}
