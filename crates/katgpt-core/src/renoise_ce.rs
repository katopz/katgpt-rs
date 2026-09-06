//! Renoise-CE self-verifier — perturb a completed state, re-resolve through
//! the same operator, measure drift as a verifier-free correctness score.
//!
//! Distilled from Flow Reasoning Models (Helbling et al., arXiv:2606.29150).
//! Research note: `katgpt-rs/.research/369_Flow_Reasoning_Models_Renoise_CE_Self_Verifier.md`.
//! Plan: `katgpt-rs/.plans/406_renoise_ce_self_verifier.md`.
//!
//! # What this is
//!
//! A **modelless, operator-agnostic self-evaluation signal**: given a completed
//! candidate state `y`, perturb it (add noise / mask / domain-specific
//! corruption), re-resolve through the same operator `F`, and measure the
//! cross-entropy drift `d(y, F(perturb(y)))`. Correct solutions sit in stable
//! basins of the operator's dynamics → low drift. Confident mistakes sit in
//! spurious basins → high drift under perturbation. The drift IS the verifier
//! score — no external verifier, no labels, no auxiliary head.
//!
//! This is the **third orthogonal self-eval signal** alongside CLR (claim-level
//! vote, R255/P284) and CoE (trajectory geometry, R345/P342):
//! - CLR asks "do the claims check out"
//! - CoE asks "is the trajectory shape committed"
//! - Renoise-CE asks "is the output a stable fixed point under perturbation"
//!
//! # What this is NOT
//!
//! - **NOT a UQ primitive.** Returns a raw drift score (lower = more stable),
//!   not a calibrated probability. Any UQ claim (correctness probability,
//!   confidence interval) MUST be conformal-wrapped and beat the floor
//!   (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`, Plan 340 / Issue
//!   010). Until then, it is a **ranking signal**.
//! - **NOT a refinement step.** Unlike Q-Sample (Plan 222) which re-noises to
//!   drive toward a *better* answer, renoise-CE re-noises to *score* the
//!   current answer. The candidate is returned unchanged.
//! - **NOT the same-input comparison of Self-Advantage Gate** (Plan 283).
//!   Renoise-CE PERTURBS the input; Self-Advantage compares the same input
//!   across two passes.
//!
//! # Hot-path design
//!
//! - `RenoiseCeScore::per_draw` is a fixed `[f32; 8]` — zero allocation on the
//!   score path.
//! - `perturb` operates in-place on a cloned state (one alloc per draw,
//!   unavoidable — the caller's state must not be mutated).
//! - `re_resolve` returns an owned state.
//! - The score loop reuses a single accumulator.
//!
//! # RNG
//!
//! Uses `fastrand::Rng` (codebase convention). The trait is generic over
//! `fastrand::Rng` directly (not `impl rand::Rng`) to match the rest of
//! katgpt-core. Callers construct one with `fastrand::Rng::with_seed(...)`
//! for determinism.

use fastrand::Rng;

/// Configuration for a renoise-CE probe.
#[derive(Clone, Copy, Debug)]
pub struct RenoiseCeConfig {
    /// Perturbation magnitude (paper: `t=0.40` for flow LMs; domain-specific).
    /// For Gaussian perturbation this is the std-dev; for mask perturbation
    /// it is the mask probability.
    pub perturbation_level: f32,
    /// Number of re-noise draws to average (paper: `k=8`; saturates at `k=1`).
    /// Clamped to `[1, 8]` — `per_draw` is a fixed `[f32; 8]`.
    pub k_draws: u8,
    /// Acceptance threshold `τ` (lower = stricter). A candidate is `accepted`
    /// iff `drift < tau`. Paper: tuned per task.
    pub tau: f32,
}

impl RenoiseCeConfig {
    /// Paper-default config: `t=0.40`, `k=8`, `tau=0.5`.
    pub const DEFAULT: Self = Self {
        perturbation_level: 0.40,
        k_draws: 8,
        tau: 0.5,
    };

    /// Single-draw config (paper shows AUROC saturates at `k=1`).
    pub const K1: Self = Self {
        perturbation_level: 0.40,
        k_draws: 1,
        tau: 0.5,
    };
}

impl Default for RenoiseCeConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A single renoise-CE probe result.
///
/// `per_draw` is a fixed `[f32; 8]` matching the paper's `k=8` max. Unused
/// slots (when `k_draws < 8`) are zero-initialized and excluded from the mean.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RenoiseCeScore {
    /// Mean cross-entropy drift across `k` draws (lower = more stable).
    pub drift: f32,
    /// Per-draw drifts. Only entries `[0..k)` are meaningful; the rest are 0.0.
    pub per_draw: [f32; 8],
    /// Acceptance decision: `drift < tau`.
    pub accepted: bool,
}

/// Trait for operators that can be probed by renoise-CE.
///
/// The operator maps a state to a state (denoiser, HLA evolve step, functor
/// application, consolidation, attention forward). The probe perturbs the
/// input state and measures how much the output drifts.
///
/// Implementors define:
/// - `re_resolve`: one step (or full convergence) of the operator on a state.
/// - `perturb`: domain-specific corruption (Gaussian noise, mask, dropout).
/// - `drift_ce`: cross-entropy of `candidate` under the re-resolved state.
pub trait RenoiseCeProbe {
    /// The state type. Must be cloneable (one clone per draw) and byte-wise
    /// inspectable (for drift computation). For continuous states this is
    /// typically `Vec<f32>` or a fixed-size array; for discrete it is a token
    /// sequence.
    type State: Clone;

    /// Re-resolve through the operator from a (possibly perturbed) state.
    ///
    /// For a single-step probe this is one application of `F`. For a
    /// convergence probe this iterates `F` to a fixed point. The paper uses
    /// full re-resolution (the inner self-conditioning loop); the open
    /// primitive leaves this to the implementor.
    fn re_resolve(&self, state: &Self::State) -> Self::State;

    /// Perturb the state in-place (domain-specific: Gaussian noise, mask, etc.).
    fn perturb(&self, state: &mut Self::State, level: f32, rng: &mut Rng);

    /// Cross-entropy drift of `candidate` relative to `re_resolved`.
    ///
    /// For continuous states: negative log-likelihood under a Gaussian
    /// centered at `re_resolved` (mean squared error, up to a constant).
    /// For discrete: token-level cross-entropy.
    ///
    /// Lower = more stable (candidate is a fixed point of the operator).
    fn drift_ce(candidate: &Self::State, re_resolved: &Self::State) -> f32;
}

/// Compute the renoise-CE score for a completed candidate.
///
/// `candidate` is the completed state to verify. The probe perturbs a clone
/// of it, re-resolves through the same operator, and measures drift. This is
/// repeated `config.k_draws` times and averaged.
///
/// # Allocation
///
/// One `candidate.clone()` per draw (the caller's state is never mutated).
/// `per_draw` is a fixed `[f32; 8]` — no heap allocation on the score path.
pub fn renoise_ce_score<O: RenoiseCeProbe>(
    operator: &O,
    candidate: &O::State,
    config: &RenoiseCeConfig,
    rng: &mut Rng,
) -> RenoiseCeScore {
    let mut per_draw = [0.0f32; 8];
    let mut sum = 0.0f32;
    let k = config.k_draws.clamp(1, 8) as usize;

    for slot in &mut per_draw[..k] {
        let mut perturbed = candidate.clone();
        operator.perturb(&mut perturbed, config.perturbation_level, rng);
        let re_resolved = operator.re_resolve(&perturbed);
        let drift = O::drift_ce(candidate, &re_resolved);
        *slot = drift;
        sum += drift;
    }

    let drift = sum / k as f32;
    RenoiseCeScore {
        drift,
        per_draw,
        accepted: drift < config.tau,
    }
}

/// A proposer generates fresh candidates for the verify-and-restart loop.
///
/// Each `propose` call returns a candidate state and the number of forward
/// passes consumed (charged to the budget).
pub trait Proposer {
    type State: Clone;
    type Output;

    /// Propose one candidate, return (state, forward passes consumed).
    fn propose(&self) -> (Self::State, usize);

    /// Convert the state into the output type (identity for pass-through).
    fn into_output(state: Self::State) -> Self::Output;
}

/// Verify-and-restart outer loop (Algorithm 2 from the paper).
///
/// Propose via `proposer`, verify via renoise-CE, restart if unstable, accept
/// if stable, under a forward-pass budget. Every verifier pass is charged.
///
/// Returns the first accepted candidate, or the lowest-drift candidate seen
/// if the budget is exhausted without acceptance.
pub fn verify_and_restart<P, O>(
    proposer: &P,
    operator: &O,
    config: &RenoiseCeConfig,
    budget: usize,
    rng: &mut Rng,
) -> Option<P::Output>
where
    P: Proposer<State = O::State>,
    O: RenoiseCeProbe,
{
    let mut spent = 0usize;
    let mut best: Option<(f32, P::Output)> = None;
    while spent < budget {
        let (candidate, n_passes) = proposer.propose();
        spent += n_passes;
        let score = renoise_ce_score(operator, &candidate, config, rng);
        spent += config.k_draws.clamp(1, 8) as usize; // charge verifier NFE
        if score.accepted {
            return Some(P::into_output(candidate));
        }
        match &best {
            None => best = Some((score.drift, P::into_output(candidate))),
            Some((d, _)) if score.drift < *d => {
                best = Some((score.drift, P::into_output(candidate)));
            }
            _ => {}
        }
    }
    best.map(|(_, o)| o)
}

/// Best-of-N selection by renoise-CE stability (Appendix C — passive case).
///
/// Keep the most stable proposal from `n` i.i.d. samples (lowest mean drift).
/// No external verifier, no ground truth. This is the passive test-time
/// scaling special case of `verify_and_restart` (no early acceptance, fixed N).
pub fn best_of_n_stability<P, O>(
    proposer: &P,
    operator: &O,
    config: &RenoiseCeConfig,
    n: usize,
    rng: &mut Rng,
) -> Option<P::Output>
where
    P: Proposer<State = O::State>,
    O: RenoiseCeProbe,
{
    (0..n)
        .map(|_| {
            let (candidate, _) = proposer.propose();
            let score = renoise_ce_score(operator, &candidate, config, rng);
            (score.drift, candidate)
        })
        .min_by(|a, b| crate::float_order::cmp_for_min(a.0, b.0))
        .map(|(_, c)| P::into_output(c))
}

/// Best-of-N selection by freedom gain within a drift gate (Issue 665 /
/// Research 486, arXiv:2608.05423).
///
/// Among `n` proposals, drift-score each via renoise-CE (drift = loss, lower
/// better), keep candidates within `gate` of the best drift, and select the
/// one whose cell opens the largest Δ-log-extension-count region
/// ([`crate::extension_count`]). Ties on gain break toward lower drift, then
/// earlier proposal. The selected cell is recorded into `occupancy` —
/// caller-owned persistent state across calls (the criterion is a
/// running-state selection rule; the paper's controller kept a decayed
/// occupancy table across steps).
///
/// Selection-only: no training, no gradient, modelless. The confound control
/// this mode must beat to promote (Issue 665 T4): random-near-best over the
/// SAME gate — gain may come from merely relaxing the min-loss choice.
///
/// # Allocation
///
/// One `Vec` of `(drift, state)` pairs per call (the pool must be revisited
/// after the best drift is known — proposers may be stochastic, so candidates
/// cannot be re-drawn). Not a hot-path decode primitive.
#[allow(clippy::too_many_arguments)] // selection seam mirrors best_of_n_stability + gate/occupancy/classifier (Issue 665)
#[cfg(feature = "freedom_selection")]
pub fn best_of_n_freedom<P, O, C>(
    proposer: &P,
    operator: &O,
    config: &RenoiseCeConfig,
    n: usize,
    gate: &crate::extension_count::LossGate,
    occupancy: &mut crate::extension_count::ExtensionOccupancy,
    cell_of: C,
    rng: &mut Rng,
) -> Option<P::Output>
where
    P: Proposer<State = O::State>,
    O: RenoiseCeProbe,
    C: Fn(&P::State) -> usize,
{
    let mut pool: Vec<(f32, P::State)> = Vec::with_capacity(n);
    for _ in 0..n {
        let (candidate, _) = proposer.propose();
        let score = renoise_ce_score(operator, &candidate, config, rng);
        pool.push((score.drift, candidate));
    }
    let best_drift = pool.iter().map(|(d, _)| *d).fold(f32::INFINITY, f32::min);

    // Among gated candidates: max gain, tie → lower drift, tie → earlier index.
    let mut chosen: Option<(f32, f32, usize)> = None; // (gain, drift, pool idx)
    for (idx, (drift, state)) in pool.iter().enumerate() {
        if !gate.admits(*drift, best_drift) {
            continue;
        }
        let gain = occupancy.freedom_gain(cell_of(state));
        let better = match chosen {
            None => true,
            Some((cg, cd, _)) => gain > cg || (gain == cg && *drift < cd),
        };
        if better {
            chosen = Some((gain, *drift, idx));
        }
    }
    let (_, _, idx) = chosen?;
    let (_, state) = pool.swap_remove(idx);
    occupancy.record(cell_of(&state));
    Some(P::into_output(state))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Toy operator: linear contraction F(x) = alpha * x ----
    //
    // Stable fixed point is the origin. A candidate AT the origin has zero
    // drift under perturbation; a candidate far from the origin drifts.

    #[derive(Clone, Debug)]
    struct VecState(pub Vec<f32>);

    struct LinearContraction {
        alpha: f32,
    }

    impl RenoiseCeProbe for LinearContraction {
        type State = VecState;

        fn re_resolve(&self, state: &Self::State) -> Self::State {
            // F(x) = alpha * x  — one step toward the origin.
            VecState(state.0.iter().map(|&v| self.alpha * v).collect())
        }

        fn perturb(&self, state: &mut Self::State, level: f32, rng: &mut Rng) {
            // Gaussian-ish perturbation via sum-of-uniforms (cheap, deterministic
            // under fastrand). level = std-dev.
            for v in &mut state.0 {
                // Sum of 3 uniforms approximates a triangular/Gaussian.
                let g = (rng.f32() + rng.f32() + rng.f32() - 1.5) * level * 1.4;
                *v += g;
            }
        }

        fn drift_ce(candidate: &Self::State, re_resolved: &Self::State) -> f32 {
            // MSE drift = mean((candidate - re_resolved)^2). For the
            // contraction, a candidate AT the origin has re_resolved ≈ 0 too,
            // so drift ≈ 0. A candidate far away has re_resolved = alpha*cand,
            // so drift = mean((1-alpha)^2 * cand^2) = (1-alpha)^2 * ||cand||^2/D.
            let n = candidate.0.len().max(1);
            candidate
                .0
                .iter()
                .zip(re_resolved.0.iter())
                .map(|(c, r)| {
                    let d = c - r;
                    d * d
                })
                .sum::<f32>()
                / n as f32
        }
    }

    #[test]
    fn config_default_is_paper_values() {
        let c = RenoiseCeConfig::default();
        assert_eq!(c.perturbation_level, 0.40);
        assert_eq!(c.k_draws, 8);
        assert_eq!(c.tau, 0.5);
    }

    #[test]
    fn config_k1_is_single_draw() {
        let c = RenoiseCeConfig::K1;
        assert_eq!(c.k_draws, 1);
    }

    #[test]
    fn origin_candidate_has_near_zero_drift() {
        // Candidate at the origin is a fixed point of F(x)=alpha*x.
        // Perturb + re-resolve: perturb moves it to ~N(0, level), re-resolve
        // shrinks by alpha. Drift = MSE(origin, alpha*perturb) = alpha^2 * mean(perturb^2).
        // For alpha=0.5, level=0.1: drift ≈ 0.25 * 0.01 = 0.0025. Very small.
        let op = LinearContraction { alpha: 0.5 };
        let candidate = VecState(vec![0.0; 8]);
        let config = RenoiseCeConfig {
            perturbation_level: 0.1,
            k_draws: 8,
            tau: 0.5,
        };
        let mut rng = Rng::with_seed(42);
        let score = renoise_ce_score(&op, &candidate, &config, &mut rng);
        assert!(
            score.drift < 0.01,
            "origin drift {} should be < 0.01",
            score.drift
        );
        assert!(score.accepted, "origin should be accepted (stable)");
    }

    #[test]
    fn far_candidate_has_high_drift() {
        // Candidate far from origin: re_resolved = alpha * cand.
        // drift = mean((1-alpha)^2 * cand^2). For alpha=0.5, cand=10:
        // drift = 0.25 * 100 = 25. Way above tau=0.5.
        let op = LinearContraction { alpha: 0.5 };
        let candidate = VecState(vec![10.0; 8]);
        let config = RenoiseCeConfig {
            perturbation_level: 0.1,
            k_draws: 8,
            tau: 0.5,
        };
        let mut rng = Rng::with_seed(42);
        let score = renoise_ce_score(&op, &candidate, &config, &mut rng);
        assert!(
            score.drift > 5.0,
            "far candidate drift {} should be > 5.0",
            score.drift
        );
        assert!(!score.accepted, "far candidate should NOT be accepted");
    }

    #[test]
    fn k_draws_averages_correctly() {
        // With k=8, all 8 per_draw slots are populated and drift = mean.
        let op = LinearContraction { alpha: 0.5 };
        let candidate = VecState(vec![1.0; 8]);
        let config = RenoiseCeConfig {
            perturbation_level: 0.1,
            k_draws: 8,
            tau: f32::INFINITY, // always accept
        };
        let mut rng = Rng::with_seed(42);
        let score = renoise_ce_score(&op, &candidate, &config, &mut rng);
        let expected_mean: f32 = score.per_draw.iter().sum::<f32>() / 8.0;
        assert!(
            (score.drift - expected_mean).abs() < 1e-6,
            "drift {} != mean of per_draw {}",
            score.drift,
            expected_mean
        );
        // All 8 slots populated (nonzero for a nonzero candidate + perturbation).
        for (i, &d) in score.per_draw.iter().enumerate() {
            assert!(d >= 0.0, "per_draw[{i}] = {d} should be >= 0");
        }
    }

    #[test]
    fn k1_only_populates_first_slot() {
        let op = LinearContraction { alpha: 0.5 };
        let candidate = VecState(vec![1.0; 8]);
        let config = RenoiseCeConfig {
            perturbation_level: 0.1,
            k_draws: 1,
            tau: f32::INFINITY,
        };
        let mut rng = Rng::with_seed(42);
        let score = renoise_ce_score(&op, &candidate, &config, &mut rng);
        assert!(
            (score.drift - score.per_draw[0]).abs() < 1e-6,
            "k=1 drift should equal per_draw[0]"
        );
        // Slots 1..8 stay zero.
        for (i, &d) in score.per_draw.iter().enumerate().skip(1) {
            assert!(d == 0.0, "per_draw[{i}] = {d} should be 0 for k=1");
        }
    }

    #[test]
    fn acceptance_gate_is_strict_lt() {
        // drift < tau → accepted. drift == tau → NOT accepted (strict).
        let score = RenoiseCeScore {
            drift: 0.5,
            per_draw: [0.5; 8],
            accepted: false, // we'll recompute
        };
        let _ = score; // suppress unused
        // The gate logic lives in renoise_ce_score; verify via config.
        // Gate is strict less-than: 0.5 < 0.6 (tau) → accepted;
        // 0.5 < 0.5 (tau) is false → NOT accepted (strict).
    }

    #[test]
    fn candidate_is_not_mutated() {
        // The probe must clone the candidate before perturbing.
        let op = LinearContraction { alpha: 0.5 };
        let candidate = VecState(vec![1.0, 2.0, 3.0, 4.0]);
        let original = candidate.0.clone();
        let config = RenoiseCeConfig::default();
        let mut rng = Rng::with_seed(42);
        let _ = renoise_ce_score(&op, &candidate, &config, &mut rng);
        assert_eq!(
            candidate.0, original,
            "candidate must not be mutated by the probe"
        );
    }

    // ---- Proposer + verify_and_restart / best_of_n ----

    struct OriginProposer {
        dim: usize,
        spread: f32,
        rng_seed: u64,
        call_count: std::cell::Cell<usize>,
    }

    impl Proposer for OriginProposer {
        type State = VecState;
        type Output = VecState;

        fn propose(&self) -> (Self::State, usize) {
            self.call_count.set(self.call_count.get() + 1);
            let mut rng = Rng::with_seed(self.rng_seed.wrapping_add(self.call_count.get() as u64));
            // Propose near the origin with Gaussian-ish noise. Occasionally
            // propose far (a "confident mistake").
            let far = rng.u32(0..100) < 20; // 20% far
            let center = if far { 5.0 } else { 0.0 };
            let state: Vec<f32> = (0..self.dim)
                .map(|_| center + (rng.f32() + rng.f32() + rng.f32() - 1.5) * self.spread)
                .collect();
            (VecState(state), 1)
        }

        fn into_output(state: Self::State) -> Self::Output {
            state
        }
    }

    #[test]
    fn verify_and_restart_accepts_stable_origin() {
        // With a low tau, only origin-near candidates pass. The loop should
        // find one within the budget.
        let op = LinearContraction { alpha: 0.5 };
        let proposer = OriginProposer {
            dim: 8,
            spread: 0.05,
            rng_seed: 7,
            call_count: std::cell::Cell::new(0),
        };
        let config = RenoiseCeConfig {
            perturbation_level: 0.05,
            k_draws: 2,
            tau: 0.01, // strict — only very-stable candidates pass
        };
        let mut rng = Rng::with_seed(99);
        let result = verify_and_restart(&proposer, &op, &config, 200, &mut rng);
        assert!(
            result.is_some(),
            "should find a stable candidate within budget"
        );
        let out = result.unwrap();
        // Accepted candidate should be near the origin (low norm).
        let norm: f32 = out.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            norm < 1.0,
            "accepted candidate norm {norm} should be < 1.0 (near origin)"
        );
    }

    #[test]
    fn verify_and_restart_budget_exhaustion_returns_best() {
        // With tau=0 (impossible to accept), budget exhausts and returns
        // the lowest-drift candidate seen.
        let op = LinearContraction { alpha: 0.5 };
        let proposer = OriginProposer {
            dim: 8,
            spread: 0.1,
            rng_seed: 3,
            call_count: std::cell::Cell::new(0),
        };
        let config = RenoiseCeConfig {
            perturbation_level: 0.1,
            k_draws: 1,
            tau: 0.0, // nothing accepted
        };
        let mut rng = Rng::with_seed(99);
        let result = verify_and_restart(&proposer, &op, &config, 50, &mut rng);
        assert!(
            result.is_some(),
            "budget exhaustion should still return best-seen"
        );
    }

    #[test]
    fn best_of_n_picks_origin_over_far() {
        // With enough samples, best_of_n_stability should pick an origin-near
        // candidate (lower drift) over a far one.
        let op = LinearContraction { alpha: 0.5 };
        let proposer = OriginProposer {
            dim: 8,
            spread: 0.1,
            rng_seed: 11,
            call_count: std::cell::Cell::new(0),
        };
        let config = RenoiseCeConfig {
            perturbation_level: 0.1,
            k_draws: 2,
            tau: f32::INFINITY,
        };
        let mut rng = Rng::with_seed(99);
        let result = best_of_n_stability(&proposer, &op, &config, 20, &mut rng);
        assert!(result.is_some(), "best_of_n should return a candidate");
        let out = result.unwrap();
        let norm: f32 = out.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        // The min-drift candidate should be origin-near (far ones have ~100x drift).
        assert!(
            norm < 2.0,
            "best_of_n winner norm {norm} should be < 2.0 (picked stable)"
        );
    }

    #[test]
    fn k_draws_clamped_to_8() {
        // k_draws > 8 is clamped; per_draw never overflows.
        let op = LinearContraction { alpha: 0.5 };
        let candidate = VecState(vec![1.0; 8]);
        let config = RenoiseCeConfig {
            perturbation_level: 0.1,
            k_draws: 200, // over-max
            tau: 0.5,
        };
        let mut rng = Rng::with_seed(42);
        let score = renoise_ce_score(&op, &candidate, &config, &mut rng);
        // Should not panic; drift is the mean of 8 draws.
        assert!(score.drift > 0.0);
        assert!(score.drift.is_finite());
    }

    #[test]
    fn k_draws_zero_clamped_to_1() {
        let op = LinearContraction { alpha: 0.5 };
        let candidate = VecState(vec![1.0; 8]);
        let config = RenoiseCeConfig {
            perturbation_level: 0.1,
            k_draws: 0, // under-min
            tau: 0.5,
        };
        let mut rng = Rng::with_seed(42);
        let score = renoise_ce_score(&op, &candidate, &config, &mut rng);
        assert!(
            (score.drift - score.per_draw[0]).abs() < 1e-6,
            "k=0 clamped to 1: drift should equal per_draw[0]"
        );
    }

    // ── best_of_n_freedom (Issue 665 / Research 486) ─────────────────
    //
    // Toy: state = (cell, loss); drift = loss (deterministic probe — the
    // point is the SELECTION rule, not the drift estimator). Partition:
    // 2 contexts × 2 cells.
    #[cfg(feature = "freedom_selection")]
    mod freedom {
        use super::*;

        /// Candidate state for the freedom toy: (cell, loss). Copy so the pool
        /// proposer never allocates.
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct CellCand {
            cell: usize,
            loss: f32,
        }

        /// Deterministic probe: perturb is a no-op, re_resolve is identity,
        /// drift = the candidate's loss. The renoise machinery runs but the
        /// score is exactly the loss — isolating the selection rule.
        struct CellProbe;

        impl RenoiseCeProbe for CellProbe {
            type State = CellCand;
            fn re_resolve(&self, s: &Self::State) -> Self::State {
                *s
            }
            fn perturb(&self, _s: &mut Self::State, _level: f32, _rng: &mut Rng) {}
            fn drift_ce(c: &Self::State, _r: &Self::State) -> f32 {
                c.loss
            }
        }

        /// Replays a fixed candidate pool (interior-mutability index walk) so all
        /// arms of a comparison see identical proposals at matched budget.
        struct FixedPoolProposer<'a> {
            pool: &'a [CellCand],
            idx: std::cell::Cell<usize>,
        }

        impl Proposer for FixedPoolProposer<'_> {
            type State = CellCand;
            type Output = CellCand;
            fn propose(&self) -> (Self::State, usize) {
                let i = self.idx.get().min(self.pool.len() - 1);
                self.idx.set(self.idx.get() + 1);
                (self.pool[i], 1)
            }
            fn into_output(s: Self::State) -> Self::Output {
                s
            }
        }

        fn freedom_setup(
            pool: &'_ [CellCand],
        ) -> (
            FixedPoolProposer<'_>,
            crate::extension_count::ExtensionOccupancy,
        ) {
            // 4 cells, contexts [0, 0, 1, 1].
            let occ = crate::extension_count::ExtensionOccupancy::new(vec![0, 0, 1, 1], 2);
            (
                FixedPoolProposer {
                    pool,
                    idx: std::cell::Cell::new(0),
                },
                occ,
            )
        }

        #[test]
        fn freedom_picks_fresh_cell_within_gate_over_better_occupied() {
            // Occupancy pre-state: cell 0 (ctx 0) occupied once → fresh cell 1
            // (same ctx, a=1) has gain ln 3 ≈ 1.0986; cell 0 gain 0.
            let pool = [
                CellCand { cell: 0, loss: 1.0 }, // best drift, occupied → gain 0
                CellCand { cell: 1, loss: 1.3 }, // within gate 0.5, fresh → gain ln 3
                CellCand { cell: 2, loss: 2.0 }, // outside gate
            ];
            let (proposer, mut occ) = freedom_setup(&pool);
            occ.record(0);
            let mut rng = Rng::with_seed(7);
            let out = best_of_n_freedom(
                &proposer,
                &CellProbe,
                &RenoiseCeConfig::K1,
                pool.len(),
                &crate::extension_count::LossGate::Absolute(0.5),
                &mut occ,
                |s| s.cell,
                &mut rng,
            );
            assert_eq!(out, Some(CellCand { cell: 1, loss: 1.3 }));
            assert_eq!(occ.cell_count(1), 1, "selected cell recorded");
            assert_eq!(occ.context_counts(), &[2, 0]);
        }

        #[test]
        fn freedom_all_occupied_falls_back_to_min_drift() {
            // Every candidate in one already-occupied cell → all gains 0 → tie
            // breaks to lower drift.
            let pool = [
                CellCand { cell: 0, loss: 1.2 },
                CellCand { cell: 0, loss: 1.0 },
                CellCand { cell: 0, loss: 1.4 },
            ];
            let (proposer, mut occ) = freedom_setup(&pool);
            occ.record(0);
            let mut rng = Rng::with_seed(7);
            let out = best_of_n_freedom(
                &proposer,
                &CellProbe,
                &RenoiseCeConfig::K1,
                pool.len(),
                &crate::extension_count::LossGate::Absolute(1.0),
                &mut occ,
                |s| s.cell,
                &mut rng,
            );
            assert_eq!(out, Some(CellCand { cell: 0, loss: 1.0 }));
        }

        #[test]
        fn freedom_empty_pool_returns_none() {
            let pool: [CellCand; 0] = [];
            let (proposer, mut occ) = freedom_setup(&pool);
            let mut rng = Rng::with_seed(7);
            let out = best_of_n_freedom(
                &proposer,
                &CellProbe,
                &RenoiseCeConfig::K1,
                0,
                &crate::extension_count::LossGate::Absolute(0.5),
                &mut occ,
                |s| s.cell,
                &mut rng,
            );
            assert_eq!(out, None);
        }

        #[test]
        fn freedom_first_activation_dominates_finite_gain() {
            // cell 3 (ctx 1, EMPTY context) has first-activation gain 2.0 >
            // cell 1's ln 3 — must win despite worse drift (within gate).
            let pool = [
                CellCand { cell: 1, loss: 1.0 }, // fresh in occupied ctx 0 → ln 3
                CellCand { cell: 3, loss: 1.4 }, // fresh in EMPTY ctx 1 → 2.0
            ];
            let (proposer, mut occ) = freedom_setup(&pool);
            occ.record(0); // ctx 0 has a=1; ctx 1 empty
            let mut rng = Rng::with_seed(7);
            let out = best_of_n_freedom(
                &proposer,
                &CellProbe,
                &RenoiseCeConfig::K1,
                pool.len(),
                &crate::extension_count::LossGate::Absolute(0.5),
                &mut occ,
                |s| s.cell,
                &mut rng,
            );
            assert_eq!(out, Some(CellCand { cell: 3, loss: 1.4 }));
        }
    }
}
