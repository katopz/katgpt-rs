//! Offline excess/deficit certification for support gates (Plan 618
//! T6/T7 — the Research 604 App-E restatement).
//!
//! The paper's continuous bound (`vol(excess) ≤ ‖P̂−P‖₁/τ`) uses Lebesgue
//! volume, which is UNMEASURABLE under unbounded Gaussians in 256-D —
//! the stack's implementable form (verdict round 1):
//!
//! - **excess** = Φ_neg-weighted excess MASS: sample the reference GMM,
//!   measure the gate's open rate (the volume the reference actually
//!   visits — the only volume a serving consumer ever sees).
//! - **deficit** = open rate on HELD-OUT in-support data (closed-on-your-
//!   own-support is the failure the training objective fixes).
//! - **bound check (T7)** = on fixtures with KNOWN P, the L1 distance
//!   `‖P̂−P‖₁ = E_P[|p̂/p − 1|]` is Monte-Carlo estimable in closed form
//!   (both densities evaluable) — excess is compared against it with a
//!   consumer-pinned slack, at the fixed-threshold ratio form where the
//!   threshold τ = 1. Fixture-scoped sanity, NOT a tight bound.
//!
//! # The leak-by-construction canary (T6, the §3.6 discipline)
//!
//! A validator that cannot red is not a validator. The control arm is a
//! LINEAR DISCRIMINATOR gate fit on the same features — the class the
//! paper's D1 ablation refutes (discriminative gates open arbitrarily
//! outside training data) — and the extrapolation probes are planted FAR
//! OUT along its positive normal, where it is GUARANTEED to fire. If the
//! canary discriminator does NOT fire on those probes, the INSTRUMENT
//! itself is broken and [`canary_fire_rate`] returns
//! [`CertifyError::CanaryDidNotFire`] (the exit-2 class) — never a
//! confident green.

use super::fixture::sample_mixture_into;
use super::mixture::DiagGmm;
use super::projector::SplitMix64;

/// Errors returned by the certification instruments. `CanaryDidNotFire`
/// is the instrument's OWN tripwire — the exit-2 class (§3.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CertifyError {
    /// A probe/sample slice was empty.
    EmptyProbes,
    /// The canary discriminator did not fire on ALL extrapolation probes
    /// — the certification instrument is broken (fix the fixture or the
    /// discriminator), do NOT read the report as a pass.
    CanaryDidNotFire,
}

/// Instrument configuration. Defaults are POC-scale; consumers re-pin.
#[derive(Debug, Clone, Copy)]
pub struct CertifyConfig {
    /// Monte-Carlo sample count for the Φ_neg-weighted excess mass.
    pub n_reference_samples: usize,
    /// Monte-Carlo sample count for the T7 L1 estimate.
    pub n_l1_samples: usize,
    /// The excess-per-L1 multiplier of the T7 fixture-scoped relation
    /// (see [`bound_acceptance`]).
    pub bound_multiplier: f32,
    /// Additive slack of the T7 bound check.
    pub slack: f32,
    /// Seed for the Monte-Carlo streams.
    pub seed: u64,
}

impl Default for CertifyConfig {
    fn default() -> Self {
        Self {
            n_reference_samples: 4096,
            n_l1_samples: 8192,
            bound_multiplier: 2.0,
            slack: 0.02,
            seed: 7,
        }
    }
}

/// The gate-under-test abstraction: any predicate over projected points.
/// Object-safe, offline-only (the hot path takes `&[f32; E]`, this takes
/// the erase-the-const-generics slice form).
pub trait SupportDecision {
    /// The gate's open predicate on a projected point.
    fn opens(&self, x: &[f32]) -> bool;
}

impl<const E: usize, const K: usize> SupportDecision for super::gate::SupportGate<E, K> {
    fn opens(&self, x: &[f32]) -> bool {
        match <[f32; E]>::try_from(x) {
            Ok(arr) => self.open(&arr),
            Err(_) => false, // wrong-dim input is closed, never a panic
        }
    }
}

/// The deliberately-leaky control class: a linear score fit on the same
/// features, opening iff `w·x > bias` (the paper's D1 refuted class).
#[derive(Debug, Clone)]
pub struct LinearDiscriminatorGate {
    w: Vec<f32>,
    bias: f32,
}

impl LinearDiscriminatorGate {
    /// Fit the separating direction `w = normalize(μ_A − μ_B)` and the
    /// midpoint bias from class means.
    ///
    /// # Errors
    ///
    /// [`CertifyError::EmptyProbes`] on empty classes or a degenerate
    /// (zero) direction.
    pub fn fit(a: &[&[f32]], b: &[&[f32]]) -> Result<Self, CertifyError> {
        if a.is_empty() || b.is_empty() {
            return Err(CertifyError::EmptyProbes);
        }
        let d = a[0].len();
        let mut mean_a = vec![0f64; d];
        let mut mean_b = vec![0f64; d];
        for s in a {
            for (acc, &v) in mean_a.iter_mut().zip(s.iter()) {
                *acc += f64::from(v);
            }
        }
        for s in b {
            for (acc, &v) in mean_b.iter_mut().zip(s.iter()) {
                *acc += f64::from(v);
            }
        }
        for acc in mean_a.iter_mut() {
            *acc /= a.len() as f64;
        }
        for acc in mean_b.iter_mut() {
            *acc /= b.len() as f64;
        }
        let mut w = vec![0f32; d];
        let mut norm = 0.0f64;
        for i in 0..d {
            let diff = mean_a[i] - mean_b[i];
            w[i] = diff as f32;
            norm += diff * diff;
        }
        if norm <= 0.0 {
            return Err(CertifyError::EmptyProbes);
        }
        let inv = 1.0 / norm.sqrt();
        for wi in w.iter_mut() {
            *wi = (*wi as f64 * inv) as f32;
        }
        // Bias: midpoint of the class-mean projections.
        let bias_a: f32 = w
            .iter()
            .zip(mean_a.iter())
            .map(|(&wi, &m)| wi * m as f32)
            .sum();
        let bias_b: f32 = w
            .iter()
            .zip(mean_b.iter())
            .map(|(&wi, &m)| wi * m as f32)
            .sum();
        Ok(Self {
            w,
            bias: 0.5 * (bias_a + bias_b),
        })
    }

    /// The raw discriminator score `w·x`.
    #[must_use]
    pub fn score(&self, x: &[f32]) -> f32 {
        self.w.iter().zip(x.iter()).map(|(&wi, &xi)| wi * xi).sum()
    }
}

impl SupportDecision for LinearDiscriminatorGate {
    fn opens(&self, x: &[f32]) -> bool {
        self.score(x) > self.bias
    }
}

/// The certification report: the two failure directions of a support gate.
#[derive(Debug, Clone, Copy)]
pub struct CertificationReport {
    /// Deficit rate: the fraction of HELD-OUT in-support data the gate
    /// wrongly CLOSES on (`1 − open rate` — closing on your own support
    /// is the failure the training objective fixes; LOWER is better).
    pub deficit_rate: f32,
    /// Φ_neg-weighted excess mass: open rate under reference-corpus
    /// sampling (the gate opening off-support — LOWER is better).
    pub excess_mass: f32,
    /// Probe counts (disclosure: the rates above are Monte-Carlo).
    pub n_held_out: usize,
    pub n_reference: usize,
}

/// Certify a gate: deficit on held-out in-support samples, excess by
/// reference-GMM Monte-Carlo sampling.
///
/// This is the instrument that must RED on the leaky control class — run
/// it on a [`LinearDiscriminatorGate`] and watch `excess_mass` climb
/// (pinned by the module tests + the GOAT bench's G5 arm).
///
/// # Errors
///
/// [`CertifyError::EmptyProbes`] on an empty held-out slice.
pub fn certify<const E: usize, const K: usize>(
    gate: &dyn SupportDecision,
    held_out_a: &[&[f32]],
    neg: &DiagGmm<E, K>,
    cfg: &CertifyConfig,
) -> Result<CertificationReport, CertifyError> {
    if held_out_a.is_empty() {
        return Err(CertifyError::EmptyProbes);
    }
    // Deficit: wrongly-closed fraction on held-out A.
    let open_a = held_out_a.iter().filter(|s| gate.opens(s)).count();
    let deficit_rate = 1.0 - open_a as f32 / held_out_a.len() as f32;

    // Excess: open rate under reference sampling.
    let mut rng = SplitMix64::new(cfg.seed);
    let mut x = vec![0f32; E];
    let mut open_ref = 0usize;
    for _ in 0..cfg.n_reference_samples {
        sample_mixture_into(neg, &mut rng, &mut x);
        if gate.opens(&x) {
            open_ref += 1;
        }
    }
    Ok(CertificationReport {
        deficit_rate,
        excess_mass: open_ref as f32 / cfg.n_reference_samples as f32,
        n_held_out: held_out_a.len(),
        n_reference: cfg.n_reference_samples,
    })
}

/// The canary check: the discriminator control must fire on ALL
/// extrapolation probes, else the INSTRUMENT is broken (exit-2 class).
///
/// Returns the fire rate on success (1.0 by construction of the probes).
///
/// # Errors
///
/// [`CertifyError::EmptyProbes`] on empty probes;
/// [`CertifyError::CanaryDidNotFire`] when any probe fails to fire.
pub fn canary_fire_rate(
    discriminator: &LinearDiscriminatorGate,
    extrapolation_probes: &[&[f32]],
) -> Result<f32, CertifyError> {
    if extrapolation_probes.is_empty() {
        return Err(CertifyError::EmptyProbes);
    }
    let fired = extrapolation_probes
        .iter()
        .filter(|p| discriminator.opens(p))
        .count();
    if fired != extrapolation_probes.len() {
        return Err(CertifyError::CanaryDidNotFire);
    }
    Ok(fired as f32 / extrapolation_probes.len() as f32)
}

/// The T7 fixture-scoped bound report.
#[derive(Debug, Clone, Copy)]
pub struct BoundReport {
    /// Monte-Carlo `‖P̂−P‖₁ = E_P[|p̂(x)/p(x) − 1|]` (both densities
    /// evaluable in closed form on the fixture).
    pub l1_estimate: f32,
    /// Φ_neg-weighted excess mass of the gate.
    pub excess_mass: f32,
    /// `excess ≤ multiplier·l1 + slack` — see [`bound_acceptance`].
    pub holds: bool,
}

/// T7 bound-acceptance check: on a fixture whose TRUE positive density is
/// known, estimate the L1 fit error and compare the gate's excess mass
/// against it.
///
/// **Honest scope:** the paper's App-E relation bounds LEBESGUE volume,
/// which is unmeasurable under unbounded Gaussians — the implementable
/// form weights by Φ_neg instead, under which the excess-per-L1 constant
/// is NOT 1 (measured 0.67–1.4 across the fixture family). The check is
/// therefore `excess ≤ bound_multiplier·l1 + slack` with the multiplier
/// defaulted to the measured ceiling (2.0, headroom) and consumer-pinned;
/// the MONOTONICITY arm (bigger ε ⇒ bigger excess) and the leak canary
/// are the instrument's teeth, not this absolute bar.
#[must_use]
pub fn bound_acceptance<const E: usize, const K: usize>(
    true_pos: &DiagGmm<E, K>,
    fitted: &DiagGmm<E, K>,
    neg: &DiagGmm<E, K>,
    gate: &dyn SupportDecision,
    cfg: &CertifyConfig,
) -> BoundReport {
    // L1: E_P[|p̂/p − 1|] over true-P samples.
    let mut rng = SplitMix64::new(cfg.seed ^ 0x5EED_1C7E);
    let mut x = vec![0f32; E];
    let mut l1_acc = 0.0f64;
    for _ in 0..cfg.n_l1_samples {
        sample_mixture_into(true_pos, &mut rng, &mut x);
        let xa: &[f32; E] = x.as_slice().try_into().expect("len E by construction");
        let p = true_pos.loglik_mixture(xa).exp();
        let phat = fitted.loglik_mixture(xa).exp();
        if p > 0.0 && p.is_finite() && phat.is_finite() {
            l1_acc += ((phat / p) - 1.0).abs() as f64;
        }
    }
    let l1_estimate = (l1_acc / cfg.n_l1_samples as f64) as f32;

    // Excess mass (same estimator as `certify`).
    let mut open_ref = 0usize;
    for _ in 0..cfg.n_reference_samples {
        sample_mixture_into(neg, &mut rng, &mut x);
        if gate.opens(&x) {
            open_ref += 1;
        }
    }
    let excess_mass = open_ref as f32 / cfg.n_reference_samples as f32;
    BoundReport {
        l1_estimate,
        excess_mass,
        holds: excess_mass <= cfg.bound_multiplier * l1_estimate + cfg.slack,
    }
}
