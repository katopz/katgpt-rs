//! Differential habituation filter — riir-ai Issue 1006 T1 (Research 586,
//! Diff Transformer arXiv:2410.05258 §3's subtract arm transplanted from the
//! score axis to the TIME axis): respond to *changes*, not constants.
//!
//! # The primitive
//!
//! Per perceptor channel (semantic domain — latent per the domain rules; only
//! scalar outputs cross the sync boundary):
//!
//! ```text
//! b_t = (1−β)·b_{t−1} + β·s_t          // EMA baseline (update FIRST)
//! n_t = s_t − λ·b_t                     // novelty (first-order high-pass)
//! fire iff σ(κ·n_t) > θ                 // per-channel scalar gate
//! ```
//!
//! Transfer function: first-order high-pass with **DC gain exactly `(1−λ)`**
//! — the paper's own headwise `(1−λinit)` multiplier isomorphism (they
//! rescale the differential head by its common-mode passthrough; we do the
//! same on the time axis). Constant input settles to `(1−λ)·s` — attenuated,
//! deliberately NOT zeroed (contrast [`crate::temporal_deriv`]'s derivative,
//! whose constant-input output is exactly 0). λ=1 is the full-cancel limit;
//! λ=0 is the bit-identical kill switch.
//!
//! # Substrate-first verdict (why this is not a parallel system)
//!
//! - [`crate::temporal_deriv`] (`TemporalDerivativeKernel`, Plan 277): dual
//!   fast/slow EMA **derivative** — DC-nulling, whole-vector norm surprise.
//!   Different transfer function (derivative vs DC-gain-(1−λ) high-pass) and
//!   different gate granularity (vector norm vs per-channel scalar). The
//!   complement, not a duplicate: derivative answers "how fast", habituation
//!   answers "how much above my baseline".
//! - `katgpt_sense` `modality_additive` (Issue 777): first-order COMBINATION
//!   across modalities — a different axis entirely (space, not time).
//! - `engram_privilege`'s `Δ = EMA(A·δ)`: per-slot utility drift, not a
//!   signal high-pass.
//! - Reused substrate: the house fast sigmoid
//!   (`katgpt_types::simd::fast_sigmoid` — sigmoid, never softmax).
//!
//! # Pieces
//!
//! - [`HabituationFilter`]: the per-channel EMA baseline + novelty emit
//!   (`observe`) and the per-channel-λ form (`observe_with`, threat channels
//!   vs ambient/scenery classes). Baseline state is Pod-resident latent —
//!   never synced.
//! - [`novelty_gate`]: `σ(κ·n) > θ`, strict (σ(0) = 0.5 does not fire a
//!   θ ≤ 0.5 gate).
//! - [`settling_ticks`]: closed-form `t_ε = ln(1/ε)/ln(1/(1−β))` — the tick
//!   count for the baseline error `(1−β)^t` to fall below ε. ⚠ Convention
//!   note: Issue 1006 writes the same law as `ln(1/ε)/ln(1/β)`, which is the
//!   RETENTION-convention spelling (`b_t = β·b_{t−1} + (1−β)·s_t`); under the
//!   update equations above (β = new-sample weight) the consistent form is
//!   the one implemented here. They agree at β = 0.5 only.
//! - [`THREAT_LAMBDA`] / [`AMBIENT_LAMBDA`]: the channel-class λ priors —
//!   the λinit endpoints of Issue 882's frozen schedule applied as classes
//!   on the time axis (threat 0.2 keeps signal, ambient 0.8 cancels hard).
//!   Override-able priors, never laws (the 882 trap-5 posture).
//!
//! # Gates (T1 scope)
//!
//! Closed-form unit tests below: DC gain `(1−λ)` at settled constant input,
//! λ=1 nulls constants, λ=0 bit-identical (G3 kill switch), EMA recursion
//! pinned at exact ticks, step fires on the first tick, settling law vs
//! measured decay, gate threshold semantics. G2 perf + the GOAT promotion
//! ride the consumer wiring (riir-ai Issue 1006 T2).
//!
//! Opt-in feature `habituation_filter` per the no-default-consumer rule;
//! promotion rides T2's GOAT (2 FMA + 1 fast-sigmoid per channel, the
//! 11–12 ns/NPC/tick envelope). Fixed `[f32; P]` state — zero alloc by
//! construction (G4).

use katgpt_types::simd::fast_sigmoid;

/// Threat-class channels keep the signal (the λinit schedule's cold end).
pub const THREAT_LAMBDA: f32 = 0.2;

/// Ambient/scenery-class channels cancel hard (the schedule's hot end).
pub const AMBIENT_LAMBDA: f32 = 0.8;

/// Per-channel EMA baseline + novelty high-pass. `P` channels, fixed-size
/// state (Pod-resident, never synced), zero alloc.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HabituationFilter<const P: usize> {
    /// EMA baseline state `b` — the NPC's slow memory of each channel.
    pub baseline: [f32; P],
    /// EMA new-sample weight (0, 1]: `b' = (1−β)·b + β·s`. β=1 is the
    /// memoryless limit (baseline tracks the signal exactly).
    pub beta: f32,
}

impl<const P: usize> HabituationFilter<P> {
    /// Zero-initialized filter at the given β.
    ///
    /// Debug-asserts `0 < β ≤ 1` (a wiring bug must be loud; release clamps
    /// nothing — the caller owns the constant).
    #[inline]
    #[must_use]
    pub fn new(beta: f32) -> Self {
        debug_assert!(
            beta > 0.0 && beta <= 1.0,
            "habituation_filter: require 0 < beta <= 1, got {beta}"
        );
        Self {
            baseline: [0.0; P],
            beta,
        }
    }

    /// One tick: update the baseline with the current sample, then emit the
    /// novelty `n = s − λ·b` (update-then-difference, per the primitive's
    /// own equation order — `b_t` already contains `s_t`).
    ///
    /// **λ == 0.0 returns the signal bit-identical** (G3 kill switch — no
    /// arithmetic at all, so no −0.0/rounding surprises).
    #[inline]
    #[must_use]
    pub fn observe(&mut self, signal: &[f32; P], lambda: f32) -> [f32; P] {
        let b = &mut self.baseline;
        let beta = self.beta;
        for (bi, &si) in b.iter_mut().zip(signal.iter()) {
            *bi = (1.0 - beta) * *bi + beta * si;
        }
        if lambda == 0.0 {
            return *signal;
        }
        let mut n = [0.0f32; P];
        for (ni, (&bi, &si)) in b.iter().zip(signal.iter()).enumerate() {
            n[ni] = si - lambda * bi;
        }
        n
    }

    /// [`Self::observe`] with a per-channel λ — the threat/ambient class
    /// split (threat channels keep signal, ambient channels cancel hard).
    #[inline]
    #[must_use]
    pub fn observe_with(&mut self, signal: &[f32; P], lambdas: &[f32; P]) -> [f32; P] {
        let b = &mut self.baseline;
        let beta = self.beta;
        for (bi, &si) in b.iter_mut().zip(signal.iter()) {
            *bi = (1.0 - beta) * *bi + beta * si;
        }
        let mut n = [0.0f32; P];
        for i in 0..P {
            let lambda = lambdas[i];
            n[i] = if lambda == 0.0 {
                signal[i]
            } else {
                signal[i] - lambda * b[i]
            };
        }
        n
    }
}

/// The per-channel fire decision: `σ(κ·n) > θ`, strict.
///
/// `σ(0) = 0.5`, so a novelty of exactly zero never fires a gate at θ ≥ 0.5,
/// and negative novelty (signal below baseline) never fires θ > 0.5.
#[inline]
#[must_use]
pub fn novelty_gate(novelty: f32, kappa: f32, theta: f32) -> bool {
    fast_sigmoid(kappa * novelty) > theta
}

/// Closed-form settling: the tick count for the baseline error `(1−β)^t` to
/// fall below `ε` — `t_ε = ln(1/ε)/ln(1/(1−β))`.
///
/// β = 1 (memoryless) settles in 0 ticks. Fractional — take the ceiling for
/// an integer tick budget.
#[inline]
#[must_use]
pub fn settling_ticks(beta: f32, epsilon: f32) -> f32 {
    debug_assert!(
        beta > 0.0 && beta <= 1.0,
        "habituation_filter: require 0 < beta <= 1, got {beta}"
    );
    debug_assert!(
        epsilon > 0.0 && epsilon < 1.0,
        "habituation_filter: require 0 < epsilon < 1, got {epsilon}"
    );
    if beta >= 1.0 {
        return 0.0;
    }
    (1.0 / epsilon).ln() / (1.0 / (1.0 - beta)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Settled constant input reads the DC gain (1−λ) exactly.
    #[test]
    fn settled_constant_input_reads_dc_gain_one_minus_lambda() {
        let mut f: HabituationFilter<4> = HabituationFilter::new(0.5);
        let s = [2.0f32; 4];
        for _ in 0..200 {
            let _ = f.observe(&s, 0.6);
        }
        let n = f.observe(&s, 0.6);
        for ni in n {
            assert!(
                (ni - 0.4 * 2.0).abs() < 1e-5,
                "DC gain must be (1−λ): got {ni}, want {}",
                0.4 * 2.0
            );
        }
    }

    /// λ=1 nulls a constant input (the full-cancel limit).
    #[test]
    fn lambda_one_nulls_constant_input() {
        let mut f: HabituationFilter<2> = HabituationFilter::new(0.5);
        let s = [1.5f32; 2];
        let settle = settling_ticks(0.5, 1e-4).ceil() as usize + 1;
        let mut n = [0.0f32; 2];
        for _ in 0..settle {
            n = f.observe(&s, 1.0);
        }
        for ni in n {
            assert!(ni.abs() < 1e-4, "λ=1 must null constants, got {ni}");
        }
    }

    /// λ=0 is bit-identical passthrough every tick (G3 kill switch).
    #[test]
    fn lambda_zero_is_bit_identical() {
        let mut f: HabituationFilter<3> = HabituationFilter::new(0.3);
        let seq: [[f32; 3]; 5] = [
            [0.0, 0.0, 0.0],
            [1.5, -2.0, 0.25],
            [-0.0, 7.0, -3.5],
            [1e-8, -1e8, 6.02e23],
            [0.1, 0.2, 0.3],
        ];
        for s in &seq {
            let n = f.observe(s, 0.0);
            assert_eq!(n, *s, "λ=0 must return the signal bit-identical");
        }
    }

    /// EMA recursion pinned at exact ticks (β=0.5 gives f32-exact values).
    #[test]
    fn ema_recursion_pinned_at_exact_ticks() {
        let mut f: HabituationFilter<1> = HabituationFilter::new(0.5);
        let s1 = [1.0f32];
        let _ = f.observe(&s1, 0.5);
        assert_eq!(f.baseline[0], 0.5, "b1 = β·s1 = 0.5 exactly");
        let s2 = [2.0f32];
        let n2 = f.observe(&s2, 0.5);
        assert_eq!(f.baseline[0], 1.25, "b2 = 0.5·0.5 + 0.5·2 = 1.25 exactly");
        assert_eq!(n2[0], 2.0 - 0.5 * 1.25, "n2 = s2 − λ·b2");
    }

    /// A step fires on the first tick; constant silence does not.
    #[test]
    fn step_response_fires_immediately() {
        let mut f: HabituationFilter<1> = HabituationFilter::new(0.5);
        let quiet = [0.0f32];
        for _ in 0..10 {
            let n = f.observe(&quiet, THREAT_LAMBDA);
            assert!(!novelty_gate(n[0], 4.0, 0.7), "silence must not fire");
        }
        let step = [2.0f32];
        let n = f.observe(&step, THREAT_LAMBDA);
        // b = 0.5·2 = 1.0 ⇒ n = 2 − 0.2·1 = 1.8 ⇒ σ(4·1.8) ≈ 0.99925 > 0.7.
        assert!(novelty_gate(n[0], 4.0, 0.7), "a step must fire on tick 1");
    }

    /// The settling law matches the measured baseline decay (β=0.2, ε=0.01).
    #[test]
    fn settling_law_matches_measured_decay() {
        let (beta, eps) = (0.2f32, 0.01f32);
        let s = 3.0f32;
        let mut b = 0.0f32;
        let mut measured: usize = 0;
        for t in 1..=200 {
            b = (1.0 - beta) * b + beta * s;
            if (b - s).abs() <= eps * s {
                measured = t;
                break;
            }
        }
        let law = settling_ticks(beta, eps);
        assert_eq!(
            measured,
            law.ceil() as usize,
            "measured settling {measured} vs closed form {law}"
        );
    }

    /// β=1 is the memoryless limit: 0 settling ticks, DC gain on tick 1.
    #[test]
    fn beta_one_is_memoryless() {
        assert_eq!(settling_ticks(1.0, 0.01), 0.0);
        let mut f: HabituationFilter<1> = HabituationFilter::new(1.0);
        let n = f.observe(&[4.0f32], 0.5);
        assert_eq!(n[0], 2.0, "memoryless: n = (1−λ)·s on the first tick");
    }

    /// Gate threshold semantics: strict >, σ(0)=0.5, negatives stay quiet.
    #[test]
    fn novelty_gate_threshold_semantics() {
        assert!(!novelty_gate(0.0, 4.0, 0.7));
        assert!(novelty_gate(1.0, 4.0, 0.7));
        assert!(!novelty_gate(-1.0, 4.0, 0.7));
        assert!(!novelty_gate(0.0, 4.0, 0.5), "σ(0)=0.5 must not pass a 0.5 gate (strict >)");
        assert!(novelty_gate(0.001, 4.0, 0.5), "any positive novelty passes a 0.5 gate");
    }

    /// Per-channel λ routes by class: threat keeps, ambient cancels.
    #[test]
    fn per_channel_lambdas_route_by_class() {
        let mut f: HabituationFilter<2> = HabituationFilter::new(0.5);
        let lambdas = [THREAT_LAMBDA, AMBIENT_LAMBDA];
        let s = [1.0f32; 2];
        let mut n = [0.0f32; 2];
        for _ in 0..200 {
            n = f.observe_with(&s, &lambdas);
        }
        assert!((n[0] - 0.8).abs() < 1e-5, "threat channel keeps (1−0.2)·s");
        assert!((n[1] - 0.2).abs() < 1e-5, "ambient channel cancels to (1−0.8)·s");
    }

    /// DC gain sweep across the λ operating envelope.
    #[test]
    fn dc_gain_sweep() {
        for lambda in [0.0f32, 0.2, 0.5, 0.8, 1.0] {
            let mut f: HabituationFilter<1> = HabituationFilter::new(0.5);
            let s = [1.0f32];
            let mut n = [0.0f32];
            for _ in 0..300 {
                n = f.observe(&s, lambda);
            }
            let want = 1.0 - lambda;
            assert!(
                (n[0] - want).abs() < 2e-3,
                "λ={lambda}: settled n {} vs (1−λ) {want}",
                n[0]
            );
        }
    }
}
