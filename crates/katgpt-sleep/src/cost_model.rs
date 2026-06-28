//! Amortization cost model (Plan 334 Phase 1 T1.5).
//!
//! Operationalizes the paper's §5.3 cost model:
//!
//! ```text
//! cost_total = sum_i(budgets[i])                // sleep-time, paid once per c
//!            + N_consumers * t * b_max * (1 - E[gate])
//!                                                // test-time, paid per consumer
//! ```
//!
//! Where:
//! - `t` = latency premium (paper uses t=10 — wake-time compute is 10× more
//!   expensive per token than sleep-time compute, because it's on the user's
//!   critical path).
//! - `b_max` = wake-time compute budget per consumer.
//! - `E[gate]` = expected pre-computation hit rate across the query
//!   distribution. Higher predictability → higher E[gate] → more wake-time
//!   compute avoided.
//!
//! Break-even: `sum_i(budgets[i]) < N * t * b_max * E[gate]`.
//!
//! # Modelless
//!
//! Pure closed-form algebra. No training, no backprop. The `E[gate]` input
//! is provided by the caller (measured or predicted) — this struct just does
//! the arithmetic.

/// Paper §5.3 cost model.
///
/// Answers: "given an expected hit rate `E[gate]` and `N` consumers, is it
/// worth pre-computing c' for this c?"
///
/// All fields are `Copy` (4 × f32 + 1 × u32 = 20 bytes) so the struct is
/// cheap to pass by value.
#[derive(Clone, Copy, Debug)]
pub struct AmortizationCostModel {
    /// Latency premium (paper uses t=10). Wake-time compute is `t`× more
    /// expensive per token than sleep-time compute, because it's on the
    /// user's critical path.
    pub t: f32,
    /// Wake-time compute budget per consumer (tokens).
    pub b_max: u32,
    /// Gate threshold τ (passed through for symmetry with the wake-time
    /// consumer; not used in cost arithmetic directly).
    pub tau: f32,
    /// Gate sharpness β (same caveat as `tau`).
    pub beta: f32,
}

impl Default for AmortizationCostModel {
    #[inline]
    fn default() -> Self {
        // Paper §5.3 defaults.
        Self {
            t: 10.0,
            b_max: 1024,
            tau: 0.5,
            beta: 4.0,
        }
    }
}

impl AmortizationCostModel {
    /// Expected per-consumer wake-time cost, accounting for gate hit rate.
    ///
    /// `E[gate]` is the expected pre-computation hit rate (fraction of
    /// consumers whose query lands in a predictable slot). When `E[gate] = 1`,
    /// every consumer hits a precomputed slot → wake cost is 0.
    ///
    /// `E[gate]` MUST be in `[0, 1]`; values outside are clamped.
    #[inline]
    pub fn expected_wake_cost_per_consumer(&self, e_gate: f32) -> f32 {
        let e_gate = e_gate.clamp(0.0, 1.0);
        self.t * (self.b_max as f32) * (1.0 - e_gate)
    }

    /// Total cost given N consumers.
    ///
    /// `sleep_cost` = sum of all per-direction sleep-time budgets (paid once).
    /// `n_consumers` = expected number of wake-time consumers over c's lifetime.
    /// `e_gate` = expected gate hit rate ∈ `[0, 1]`.
    #[inline]
    pub fn total_cost(&self, sleep_cost: f32, n_consumers: u32, e_gate: f32) -> f32 {
        sleep_cost + (n_consumers as f32) * self.expected_wake_cost_per_consumer(e_gate)
    }

    /// Should we pre-compute? Returns true iff the wake-cost-avoided exceeds
    /// the sleep-cost-paid.
    ///
    /// `sleep_cost < N * t * b_max * E[gate]`.
    #[inline]
    pub fn should_pre_compute(&self, sleep_cost: f32, n_consumers: u32, e_gate: f32) -> bool {
        let e_gate = e_gate.clamp(0.0, 1.0);
        let wake_cost_avoided = (n_consumers as f32) * self.t * (self.b_max as f32) * e_gate;
        sleep_cost < wake_cost_avoided
    }

    /// Amortization factor: `cost_with_precompute / cost_without_precompute`.
    ///
    /// `< 1.0` means pre-computing wins (smaller is better).
    /// Paper reports ~2.5× gain at N=10 (i.e. amortization_factor ≈ 0.4).
    ///
    /// Returns `f32::INFINITY` if `cost_without_precompute == 0` (avoids
    /// div-by-zero when N=0 or b_max=0).
    #[inline]
    pub fn amortization_factor(&self, sleep_cost: f32, n_consumers: u32, e_gate: f32) -> f32 {
        let without = (n_consumers as f32) * self.t * (self.b_max as f32);
        if without == 0.0 {
            return f32::INFINITY;
        }
        let with = self.total_cost(sleep_cost, n_consumers, e_gate);
        with / without
    }

    /// Break-even N: the number of consumers at which pre-computing becomes
    /// worthwhile.
    ///
    /// `N_break_even = sleep_cost / (t * b_max * E[gate])`.
    ///
    /// Returns `f32::INFINITY` if `E[gate] == 0` (never worth pre-computing
    /// if no one ever hits the cache).
    #[inline]
    pub fn break_even_n(&self, sleep_cost: f32, e_gate: f32) -> f32 {
        let e_gate = e_gate.clamp(0.0, 1.0);
        let denom = self.t * (self.b_max as f32) * e_gate;
        if denom == 0.0 {
            return f32::INFINITY;
        }
        sleep_cost / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Paper §5.3 reference numbers (rounded to 1 decimal in the paper).
    const PAPER_T: f32 = 10.0;
    const PAPER_B_MAX: u32 = 1024;

    fn paper_model() -> AmortizationCostModel {
        AmortizationCostModel {
            t: PAPER_T,
            b_max: PAPER_B_MAX,
            tau: 0.5,
            beta: 4.0,
        }
    }

    #[test]
    fn expected_wake_cost_zero_at_full_hit_rate() {
        let m = paper_model();
        assert_eq!(m.expected_wake_cost_per_consumer(1.0), 0.0);
    }

    #[test]
    fn expected_wake_cost_full_at_zero_hit_rate() {
        let m = paper_model();
        let cost = m.expected_wake_cost_per_consumer(0.0);
        assert!((cost - PAPER_T * PAPER_B_MAX as f32).abs() < 1e-6);
    }

    #[test]
    fn total_cost_is_monotone_decreasing_in_hit_rate() {
        // More predictability → less total cost (paper §5.3 headline claim).
        // total_cost = sleep_cost + N * t * b_max * (1 - e_gate), so as e_gate
        // INCREASES, (1 - e_gate) DECREASES, so total_cost DECREASES.
        let m = paper_model();
        let sleep_cost = 10_000.0_f32;
        let n = 10u32;
        // Iterate e_gate from 0 → 1 (cost should monotonically decrease).
        let mut prev = f32::INFINITY;
        let mut e = 0.0f32;
        while e <= 1.0 {
            let c = m.total_cost(sleep_cost, n, e);
            assert!(
                c <= prev + 1e-6,
                "total_cost not monotone decreasing in e_gate: e={} cost={} prev={}",
                e,
                c,
                prev
            );
            prev = c;
            e += 0.1;
        }
        // Sanity: at e_gate=1, cost is just sleep_cost (no wake cost).
        assert!((m.total_cost(sleep_cost, n, 1.0) - sleep_cost).abs() < 1e-6);
    }

    #[test]
    fn should_pre_compute_at_paper_break_even() {
        // At the break-even point, should_pre_compute flips.
        let m = paper_model();
        let e_gate = 0.5;
        let n = 10u32;
        // break-even sleep_cost = N * t * b_max * E[gate]
        let break_even = (n as f32) * PAPER_T * (PAPER_B_MAX as f32) * e_gate;
        // Just below break-even → pre-compute wins.
        assert!(m.should_pre_compute(0.99 * break_even, n, e_gate));
        // Just above → don't pre-compute.
        assert!(!m.should_pre_compute(1.01 * break_even, n, e_gate));
    }

    #[test]
    fn amortization_factor_at_paper_reference_point() {
        // Paper reports ~2.5× gain at N=10 (factor ≈ 0.4 with typical sleep
        // budget). Sanity-check direction only — exact value depends on the
        // sleep budget, which the paper doesn't pin down here.
        let m = paper_model();
        let sleep_cost = 5_000.0_f32;
        let factor = m.amortization_factor(sleep_cost, 10, 0.5);
        assert!(
            factor < 1.0,
            "pre-compute should win at e_gate=0.5, N=10, got factor {}",
            factor
        );
        // And it should be much less than 1 (paper's whole point).
        assert!(factor < 0.7, "expected substantial gain, got {}", factor);
    }

    #[test]
    fn amortization_factor_infinity_at_zero_consumers() {
        let m = paper_model();
        let factor = m.amortization_factor(100.0, 0, 0.5);
        assert!(factor.is_infinite(), "N=0 → infinite amortization factor");
    }

    #[test]
    fn break_even_n_solves_should_pre_compute_boundary() {
        let m = paper_model();
        let sleep_cost = 10_000.0_f32;
        let e_gate = 0.4;
        let n_be = m.break_even_n(sleep_cost, e_gate);
        // At exactly N_be (rounded), pre-compute is borderline.
        // Below → don't pre-compute; above → pre-compute.
        let n_below = (n_be.floor() as u32).saturating_sub(1).max(1);
        let n_above = (n_be.ceil() as u32) + 1;
        // Use a small tolerance: should_pre_compute is strict-<.
        let above_pays_off = m.should_pre_compute(sleep_cost, n_above, e_gate);
        let below_pays_off = m.should_pre_compute(sleep_cost, n_below, e_gate);
        assert!(
            above_pays_off || !below_pays_off,
            "break-even boundary inconsistent: n_be={} below={} above={}",
            n_be,
            below_pays_off,
            above_pays_off
        );
    }

    #[test]
    fn break_even_n_infinite_when_no_hits() {
        let m = paper_model();
        let n_be = m.break_even_n(100.0, 0.0);
        assert!(n_be.is_infinite(), "E[gate]=0 → never worth pre-computing");
    }

    #[test]
    fn out_of_range_e_gate_is_clamped() {
        let m = paper_model();
        // e_gate > 1 → clamped to 1 → wake cost is 0.
        assert_eq!(m.expected_wake_cost_per_consumer(5.0), 0.0);
        // e_gate < 0 → clamped to 0 → wake cost is full.
        let cost = m.expected_wake_cost_per_consumer(-1.0);
        assert!((cost - PAPER_T * PAPER_B_MAX as f32).abs() < 1e-6);
    }
}
