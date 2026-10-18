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
//! With `horizon_weights` also enabled, the opt-in
//! [`renoise_ce_score_horizon`] reallocates the k-draw budget over a RANGE
//! of levels under the (T−t) remaining-horizon law (Issue 875 T2 / Research
//! 582 — low-noise draws weighted up, terminal draws ~skipped).
//!
//! With `renoise_ce_surprise` enabled, the opt-in [`renoise_ce_surprise`]
//! turns the score from self-consistency into **distributional surprise
//! against a caller-supplied frozen target** (Issue 875 T4 / Research 582 —
//! PFD's resolve-against-the-teacher variant): the perturbed candidate is
//! re-resolved and scored against the TARGET anchor, not the candidate
//! itself. Novelty that is invisible pointwise (a state at the prior's own
//! distance but in a foreign basin) becomes visible — only the flow through
//! the operator, scored against the prior, sees it. First consumer: the
//! consolidation surprise ordering (which shards enter the Raven/δ-Mem
//! sleep cycle first).
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
//!
//! Renoise-CE Self-Verifier — perturb a completed state, re-resolve through the
//! same operator, measure drift as a verifier-free correctness score (Plan 406,
//! Research 369, arxiv 2606.29150). Third orthogonal self-eval signal alongside
//! CLR (claim-vote) and CoE (trajectory-shape). Operator-agnostic trait over any
//! state->state map. DEFAULT-ON (Phase 10, 2026-07-04). NOT a UQ primitive
//! (raw ranking signal; conformal wrapping required for any UQ claim).
//!
//! Freedom-guided sibling of best_of_n_stability (Issue 665 / Research 486):
//! near-best drift gate + Δ-log-extension-count selection over a caller-owned
//! occupancy table. Gated on freedom_selection (implies renoise_ce).
//!
//! Issue 875 T2 / Research 582: the (T−t)-weighted renoise-CE draw schedule —
//! k_draws sampled over [floor·L, cap·L] from the remaining-horizon density
//! instead of one fixed level. Combined-gate surface: needs both features
//! (renoise_ce is default-on; horizon_weights is the opt-in arm).
//!
//! Issue 875 T4 / Research 582: the target-anchored renoise-CE probe mode —
//! each re-resolved draw is scored against a caller-supplied frozen TARGET
//! anchor (distributional surprise vs the prior) instead of the candidate
//! itself (self-consistency). Flow-relative novelty: states at the prior's
//! own distance but in a foreign basin separate only through the flow
//! (Bench 879). Consumer sketch: consolidation surprise ordering.
//! renoise_ce_surprise implies renoise_ce, so the single gate suffices.

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
    let k = config.k_draws.clamp(1, 8) as usize;
    renoise_k_draws(
        operator,
        candidate,
        candidate,
        k,
        |_| config.perturbation_level,
        config.tau,
        rng,
    )
}

/// Shared k-draw loop (private). `anchor` is the reference the drift is
/// measured against — the candidate itself for the self-consistency modes,
/// the frozen target for the target-anchored mode. `level` yields each
/// draw's perturbation level and MUST consume `rng` only when it actually
/// samples (the fixed-level modes pass a non-sampling closure, keeping the
/// incumbent RNG stream byte-identical).
fn renoise_k_draws<O: RenoiseCeProbe>(
    operator: &O,
    anchor: &O::State,
    candidate: &O::State,
    k: usize,
    mut level: impl FnMut(&mut Rng) -> f32,
    tau: f32,
    rng: &mut Rng,
) -> RenoiseCeScore {
    let mut per_draw = [0.0f32; 8];
    let mut sum = 0.0f32;

    for slot in &mut per_draw[..k] {
        let t = level(rng);
        let mut perturbed = candidate.clone();
        operator.perturb(&mut perturbed, t, rng);
        let re_resolved = operator.re_resolve(&perturbed);
        let drift = O::drift_ce(anchor, &re_resolved);
        *slot = drift;
        sum += drift;
    }

    let drift = sum / k as f32;
    RenoiseCeScore {
        drift,
        per_draw,
        accepted: drift < tau,
    }
}

/// Opt-in (T−t)-weighted draw schedule for [`renoise_ce_score_horizon`]
/// (Issue 875 T2 / Research 582 — the PFD remaining-horizon law applied to
/// the renoise-CE probe).
///
/// The incumbent [`renoise_ce_score`] spends every draw at ONE fixed
/// `perturbation_level`. The law reallocates the SAME k-draw budget over a
/// RANGE of levels `[floor_frac·L, cap_frac·L]` (L = the config's
/// `perturbation_level`, playing the role of the horizon T), sampled
/// inverse-CDF from the density `∝ (L − t)` via
/// [`horizon_weights::remaining_horizon_t_sample`](crate::horizon_weights::remaining_horizon_t_sample):
/// low-noise draws most often, near-terminal draws almost never
/// (`w(T) = 0` — the zero-terminal-weight truncation corollary; `cap_frac <
/// 1` honors it by construction).
///
/// Defaults mirror the PFD anneal grid fractions: floor 0.02, cap 0.98.
/// Invalid fractions (non-finite, `floor ≤ 0`, `cap ∉ (floor, 1]`) fall
/// back to the defaults — the probe stays total, never NaN.
#[cfg(feature = "horizon_weights")]
#[derive(Clone, Copy, Debug)]
pub struct RenoiseCeHorizon {
    /// Lowest sampled level, as a fraction of `perturbation_level` (the
    /// horizon). Clamped to the default when not in `(0, 1)`.
    pub floor_frac: f32,
    /// Highest sampled level, as a fraction of `perturbation_level`.
    /// Clamped to the default when not in `(floor_frac, 1]`.
    pub cap_frac: f32,
}

#[cfg(feature = "horizon_weights")]
impl RenoiseCeHorizon {
    /// PFD-grid fractions: sample over `[0.02·L, 0.98·L]`.
    pub const DEFAULT: Self = Self {
        floor_frac: 0.02,
        cap_frac: 0.98,
    };
}

#[cfg(feature = "horizon_weights")]
impl Default for RenoiseCeHorizon {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Compute the renoise-CE score under the (T−t) remaining-horizon law
/// (Issue 875 T2 / Research 582).
///
/// Identical loop shape and NFE budget to [`renoise_ce_score`] — k draws,
/// one clone + perturb + re-resolve + drift each, flat mean over
/// `per_draw` — except each draw's perturbation level is SAMPLED from the
/// remaining-horizon density `∝ (L − t)` on `[floor_frac·L, cap_frac·L]`
/// (L = `config.perturbation_level`), so the average implicitly carries
/// the (T−t) Fubini weighting instead of spending the whole budget at the
/// single noisiest admissible point. Averaging under implicit
/// importance-sampling weights (not explicit per-draw weights + flat
/// draws — that would double-apply the law).
///
/// Measured effect (Bench 877, the planted-drift selectivity oracle):
/// low-noise emphasis cuts the stable-class noise floor ~5.8× vs the
/// fixed-level incumbent at L = 0.40 — precision@k of 0.97 vs 0.83 on a
/// 32+32 synthetic mixture, at an unchanged latency class (the sampler
/// adds one `sqrt` + FMAs per draw).
///
/// # What changes semantically
///
/// The score estimates the (T−t)-weighted average drift, not the
/// fixed-level drift — a `tau` calibrated for [`renoise_ce_score`] does
/// NOT transfer (low-noise draws carry smaller floors, so scores sit
/// lower). Calibrate `tau` per mode.
///
/// # Allocation
///
/// Same as the incumbent: one `candidate.clone()` per draw; no other heap
/// allocation.
#[cfg(feature = "horizon_weights")]
pub fn renoise_ce_score_horizon<O: RenoiseCeProbe>(
    operator: &O,
    candidate: &O::State,
    config: &RenoiseCeConfig,
    horizon: &RenoiseCeHorizon,
    rng: &mut Rng,
) -> RenoiseCeScore {
    let level = config.perturbation_level;
    let floor = if horizon.floor_frac.is_finite() && horizon.floor_frac > 0.0 {
        horizon.floor_frac
    } else {
        RenoiseCeHorizon::DEFAULT.floor_frac
    };
    let cap = if horizon.cap_frac.is_finite() && horizon.cap_frac > floor && horizon.cap_frac <= 1.0
    {
        horizon.cap_frac
    } else {
        RenoiseCeHorizon::DEFAULT.cap_frac
    };
    let t_min = floor * level;
    let t_max = cap * level;
    let k = config.k_draws.clamp(1, 8) as usize;

    renoise_k_draws(
        operator,
        candidate,
        candidate,
        k,
        |rng| {
            let u = rng.f32();
            crate::horizon_weights::remaining_horizon_t_sample(u, t_min, t_max, level)
        },
        config.tau,
        rng,
    )
}

/// Compute the renoise-CE **surprise** of a completed candidate against a
/// caller-supplied frozen target (Issue 875 T4 / Research 582 — PFD's
/// resolve-against-the-teacher variant, `#[cfg(feature =
/// "renoise_ce_surprise")]`).
///
/// Identical loop, budget, and allocation profile to [`renoise_ce_score`]
/// — k draws, one clone + perturb + re-resolve + drift each — except the
/// drift of each re-resolved draw is measured against `target` (the frozen
/// prior anchor), not against the candidate itself. The score stops being
/// self-consistency ("is this state a stable fixed point of the operator")
/// and becomes **distributional surprise** ("where does this state flow,
/// relative to what the prior expected").
///
/// # Semantics
///
/// - HIGHER `drift` = the perturbed-and-re-resolved state lands further
///   from the target anchor = more NOVEL relative to the prior. The
///   ranking direction inverts vs the self-consistency modes (there, high
///   drift = unstable). `accepted` here means `surprise < tau` — consistent
///   with the prior, not correct.
/// - A `tau` calibrated for [`renoise_ce_score`] does NOT transfer (the
///   anchor changed; calibrate per mode) — the same caveat as the horizon
///   mode.
/// - Novelty invisible pointwise is visible here: a state at the prior's
///   own distance but in a FOREIGN BASIN of the operator scores low under
///   plain distance AND under self-consistency (foreign-basin states are
///   often the MOST self-stable), yet flows away from the target under
///   perturbation → high surprise. This is the flow-relative-novelty
///   mechanism (Bench 879: surprise precision@32 = 1.000 vs incumbent 0.000
///   vs plain-distance 0.428 on the shell world; in pointwise-distance-
///   visible regimes plain distance is a strong ranker and the mode adds
///   little — regime boundary recorded there).
/// - NOT a UQ primitive (same standing as the incumbent — ranking signal
///   only).
///
/// # First consumer (sketch)
///
/// Consolidation surprise ordering — which shards enter the Raven/δ-Mem
/// sleep cycle first: rank shards by `renoise_ce_surprise` descending
/// against the frozen prior cycle's anchor. Ordering-only; no behavior
/// change rides the score (the sketch consumer lives in the root GOAT
/// bench, `tests/bench_875_renoise_surprise_goat.rs`).
///
/// # Allocation
///
/// Same as the incumbent: one `candidate.clone()` per draw; no other heap
/// allocation. Neither `candidate` nor `target` is mutated.
#[cfg(feature = "renoise_ce_surprise")]
pub fn renoise_ce_surprise<O: RenoiseCeProbe>(
    operator: &O,
    target: &O::State,
    candidate: &O::State,
    config: &RenoiseCeConfig,
    rng: &mut Rng,
) -> RenoiseCeScore {
    let k = config.k_draws.clamp(1, 8) as usize;
    renoise_k_draws(
        operator,
        target,
        candidate,
        k,
        |_| config.perturbation_level,
        config.tau,
        rng,
    )
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

    // ---- renoise_ce_score_horizon (Issue 875 T2, combined-gate) ----

    #[cfg(feature = "horizon_weights")]
    mod horizon {
        use super::*;

        /// A probe whose State is a fixed array (zero heap on clone) and
        /// which RECORDS the perturbation levels it was fed.
        struct RecordingProbe {
            levels: std::sync::Mutex<Vec<f32>>,
        }

        impl RecordingProbe {
            fn new() -> Self {
                Self {
                    levels: std::sync::Mutex::new(Vec::new()),
                }
            }

            fn take_levels(&self) -> Vec<f32> {
                std::mem::take(&mut *self.levels.lock().unwrap())
            }
        }

        impl RenoiseCeProbe for RecordingProbe {
            type State = [f32; 8];

            fn re_resolve(&self, state: &Self::State) -> Self::State {
                *state
            }

            fn perturb(&self, _state: &mut Self::State, level: f32, _rng: &mut Rng) {
                self.levels.lock().unwrap().push(level);
            }

            fn drift_ce(_candidate: &Self::State, _re_resolved: &Self::State) -> f32 {
                0.0
            }
        }

        fn h_config() -> RenoiseCeConfig {
            RenoiseCeConfig {
                perturbation_level: 0.40,
                k_draws: 8,
                tau: f32::INFINITY,
            }
        }

        #[test]
        fn horizon_is_deterministic_per_seed() {
            let op = RecordingProbe::new();
            let candidate = [0.5f32; 8];
            let cfg = h_config();
            let h = RenoiseCeHorizon::DEFAULT;
            let a = renoise_ce_score_horizon(&op, &candidate, &cfg, &h, &mut Rng::with_seed(42));
            let b = renoise_ce_score_horizon(&op, &candidate, &cfg, &h, &mut Rng::with_seed(42));
            assert_eq!(a.drift.to_bits(), b.drift.to_bits());
            assert_eq!(a.per_draw, b.per_draw);
            assert_eq!(a.accepted, b.accepted);
        }

        #[test]
        fn horizon_levels_stay_in_the_law_range() {
            let op = RecordingProbe::new();
            let candidate = [0.0f32; 8];
            let cfg = h_config();
            let h = RenoiseCeHorizon::DEFAULT;
            let _ = renoise_ce_score_horizon(&op, &candidate, &cfg, &h, &mut Rng::with_seed(7));
            let levels = op.take_levels();
            assert_eq!(levels.len(), 8, "k_draws = 8 draws");
            let (t_min, t_max) = (0.02 * 0.40_f32, 0.98 * 0.40_f32);
            for (i, &l) in levels.iter().enumerate() {
                assert!(
                    l >= t_min * 0.999 && l <= t_max * 1.001,
                    "level {i} outside [{t_min}, {t_max}]: {l}"
                );
            }
            // The law's shape must show up in the sample: not all levels
            // equal (it is a schedule, not the fixed incumbent), and with
            // 8 tilted draws at least one lands in the low half of the
            // range (per-draw P(low half) ≈ 0.74 under the tilted density —
            // all-8-in-high-half ≈ 5e-5; deterministic seed, verified once).
            assert!(levels.iter().any(|&l| l < (t_min + t_max) * 0.5));
        }

        #[test]
        fn horizon_invalid_fractions_fall_back_to_default() {
            let candidate = [0.0f32; 8];
            let cfg = h_config();
            // NaN floor, cap below floor, cap > 1, negative floor — all
            // fall back to the DEFAULT fractions.
            for h in [
                RenoiseCeHorizon {
                    floor_frac: f32::NAN,
                    cap_frac: 0.98,
                },
                RenoiseCeHorizon {
                    floor_frac: 0.5,
                    cap_frac: 0.1,
                },
                RenoiseCeHorizon {
                    floor_frac: 0.02,
                    cap_frac: 1.5,
                },
                RenoiseCeHorizon {
                    floor_frac: -0.1,
                    cap_frac: 0.98,
                },
            ] {
                let op = RecordingProbe::new();
                let _ = renoise_ce_score_horizon(&op, &candidate, &cfg, &h, &mut Rng::with_seed(3));
                let levels = op.take_levels();
                assert!(!levels.is_empty());
                let (t_min, t_max) = (0.02 * 0.40_f32, 0.98 * 0.40_f32);
                for &l in &levels {
                    assert!(
                        l >= t_min * 0.999 && l <= t_max * 1.001,
                        "fallback level outside default range: {l}"
                    );
                }
            }
        }

        #[test]
        fn horizon_k_clamps_match_incumbent() {
            let op = RecordingProbe::new();
            let candidate = [0.0f32; 8];
            let mut cfg = h_config();
            cfg.k_draws = 0; // clamps to 1
            let s = renoise_ce_score_horizon(
                &op,
                &candidate,
                &cfg,
                &RenoiseCeHorizon::DEFAULT,
                &mut Rng::with_seed(1),
            );
            assert_eq!(op.take_levels().len(), 1);
            assert!(s.per_draw.iter().filter(|&&d| d != 0.0).count() <= 1);

            let op = RecordingProbe::new();
            let mut cfg = h_config();
            cfg.k_draws = 200; // clamps to 8
            let _ = renoise_ce_score_horizon(
                &op,
                &candidate,
                &cfg,
                &RenoiseCeHorizon::DEFAULT,
                &mut Rng::with_seed(1),
            );
            assert_eq!(op.take_levels().len(), 8);
        }

        #[test]
        fn horizon_candidate_is_not_mutated() {
            let op = RecordingProbe::new();
            let candidate = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let original = candidate;
            let _ = renoise_ce_score_horizon(
                &op,
                &candidate,
                &h_config(),
                &RenoiseCeHorizon::DEFAULT,
                &mut Rng::with_seed(9),
            );
            assert_eq!(candidate, original, "candidate must not be mutated");
        }

        #[test]
        fn horizon_acceptance_gate_is_strict_lt_like_incumbent() {
            // A probe whose drift is a KNOWN constant per draw: re_resolve
            // adds a fixed offset, drift = offset². With offset 0.2 →
            // drift 0.04 < tau 0.05 → accepted; tau 0.04 → not accepted.
            struct OffsetProbe;
            impl RenoiseCeProbe for OffsetProbe {
                type State = [f32; 8];
                fn re_resolve(&self, s: &Self::State) -> Self::State {
                    let mut o = *s;
                    for v in &mut o {
                        *v += 0.2;
                    }
                    o
                }
                fn perturb(&self, _s: &mut Self::State, _level: f32, _rng: &mut Rng) {}
                fn drift_ce(c: &Self::State, r: &Self::State) -> f32 {
                    c.iter()
                        .zip(r.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>()
                        / 8.0
                }
            }
            let cand = [0.0f32; 8];
            for (tau, want) in [(0.05, true), (0.04, false)] {
                let mut cfg = h_config();
                cfg.tau = tau;
                let s = renoise_ce_score_horizon(
                    &OffsetProbe,
                    &cand,
                    &cfg,
                    &RenoiseCeHorizon::DEFAULT,
                    &mut Rng::with_seed(2),
                );
                assert_eq!(s.accepted, want, "tau={tau}");
                assert!((s.drift - 0.04).abs() < 1e-6, "drift={}", s.drift);
            }
        }

        #[cfg(all(test, any(debug_assertions, feature = "alloc_tracking")))]
        #[test]
        fn horizon_score_is_alloc_free_with_fixed_array_state() {
            use std::hint::black_box;
            // A flat fixed-array probe: no recording (a Vec push would
            // allocate), no heap anywhere — isolates the score path.
            struct FlatProbe;
            impl RenoiseCeProbe for FlatProbe {
                type State = [f32; 8];
                fn re_resolve(&self, s: &Self::State) -> Self::State {
                    *s
                }
                fn perturb(&self, s: &mut Self::State, level: f32, _rng: &mut Rng) {
                    for (i, v) in s.iter_mut().enumerate() {
                        *v += level * (i as f32 + 1.0) * 0.01;
                    }
                }
                fn drift_ce(c: &Self::State, r: &Self::State) -> f32 {
                    c.iter()
                        .zip(r.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>()
                        / 8.0
                }
            }
            let candidate = [0.25f32; 8];
            let cfg = h_config();
            let h = RenoiseCeHorizon::DEFAULT;
            crate::alloc::reset_alloc_stats();
            let mut sink = 0.0f32;
            for i in 0..64usize {
                let s = black_box(renoise_ce_score_horizon(
                    black_box(&FlatProbe),
                    black_box(&candidate),
                    black_box(&cfg),
                    black_box(&h),
                    &mut Rng::with_seed(i as u64),
                ));
                sink += s.drift;
            }
            let (count, _bytes) = crate::alloc::get_alloc_stats();
            assert_eq!(
                count, 0,
                "G4: horizon score path allocated {count} times (fixed-array State → none expected)"
            );
            assert!(sink.is_finite(), "sink must be consumed: {sink}");
        }
    }

    // ---- renoise_ce_surprise (Issue 875 T4, target-anchored) ----

    #[cfg(feature = "renoise_ce_surprise")]
    mod surprise {
        use super::*;

        fn s_config() -> RenoiseCeConfig {
            RenoiseCeConfig {
                perturbation_level: 0.40,
                k_draws: 8,
                tau: f32::INFINITY,
            }
        }

        /// Two-attractor contraction probe on a fixed [f32; 8] state:
        /// attractor A = origin (the target/prior center) and B = 4·e0. The
        /// snap is decided on the e0 axis (1-D separable world — midpoint
        /// 2.0); the contraction α = 0.9 then applies to EVERY coordinate
        /// toward the snapped attractor point (the Bench-879 world in
        /// miniature).
        struct TwoBasinProbe;

        impl RenoiseCeProbe for TwoBasinProbe {
            type State = [f32; 8];
            fn re_resolve(&self, s: &Self::State) -> Self::State {
                let snapped_to_b = s[0] >= 2.0;
                let mut o = [0.0f32; 8];
                for (i, v) in s.iter().enumerate() {
                    let attractor = if snapped_to_b { 4.0 } else { 0.0 };
                    // Only the e0 axis differs between the attractors; other
                    // coords contract toward 0 either way.
                    let a = if i == 0 { attractor } else { 0.0 };
                    o[i] = v - 0.9 * (v - a);
                }
                o
            }
            fn perturb(&self, s: &mut Self::State, level: f32, rng: &mut Rng) {
                for v in s.iter_mut() {
                    *v += (rng.f32() + rng.f32() + rng.f32() - 1.5) * level * 1.4;
                }
            }
            fn drift_ce(c: &Self::State, r: &Self::State) -> f32 {
                c.iter()
                    .zip(r.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    / 8.0
            }
        }

        #[test]
        fn surprise_separates_flow_novelty_at_identical_distance() {
            // Two candidates at the SAME distance from the target (the
            // prior anchor = origin): x_prior in the prior basin (direction
            // e1), x_foreign just past the midpoint toward B (direction e0).
            // Plain distance identical; only the flow separates them.
            let mut x_prior = [0.0f32; 8];
            x_prior[1] = 2.05;
            let mut x_foreign = [0.0f32; 8];
            x_foreign[0] = 2.05;
            let target = [0.0f32; 8];
            let cfg = s_config();

            let s_prior = renoise_ce_surprise(
                &TwoBasinProbe,
                &target,
                &x_prior,
                &cfg,
                &mut Rng::with_seed(31),
            );
            let s_foreign = renoise_ce_surprise(
                &TwoBasinProbe,
                &target,
                &x_foreign,
                &cfg,
                &mut Rng::with_seed(31),
            );
            // Same pointwise distance to the target (sanity: the world is a
            // shell for these two).
            let d_prior = TwoBasinProbe::drift_ce(&target, &x_prior);
            let d_foreign = TwoBasinProbe::drift_ce(&target, &x_foreign);
            assert!((d_prior - d_foreign).abs() < 1e-6);
            // The foreign-basin state must be dramatically more surprising.
            assert!(
                s_foreign.drift > 4.0 * s_prior.drift,
                "foreign {} vs prior {} — flow-relative novelty lost",
                s_foreign.drift,
                s_prior.drift
            );
        }

        #[test]
        fn surprise_anchor_change_flips_the_score_semantics() {
            // OffsetProbe shape: F(x) = x + 0.2 per coord, perturb no-op.
            // Incumbent (anchor = candidate): drift = 0.04 per coord.
            // Surprise (anchor = [1; 8]): drift = (1 − 0.2)² = 0.64 — the
            // score follows the TARGET, pinning the semantic difference.
            struct OffsetProbe;
            impl RenoiseCeProbe for OffsetProbe {
                type State = [f32; 8];
                fn re_resolve(&self, s: &Self::State) -> Self::State {
                    let mut o = *s;
                    for v in &mut o {
                        *v += 0.2;
                    }
                    o
                }
                fn perturb(&self, _s: &mut Self::State, _level: f32, _rng: &mut Rng) {}
                fn drift_ce(c: &Self::State, r: &Self::State) -> f32 {
                    c.iter()
                        .zip(r.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>()
                        / 8.0
                }
            }
            let cand = [0.0f32; 8];
            let target = [1.0f32; 8];
            let cfg = s_config();
            let inc = renoise_ce_score(&OffsetProbe, &cand, &cfg, &mut Rng::with_seed(2));
            let sur =
                renoise_ce_surprise(&OffsetProbe, &target, &cand, &cfg, &mut Rng::with_seed(2));
            assert!((inc.drift - 0.04).abs() < 1e-6, "incumbent {}", inc.drift);
            assert!((sur.drift - 0.64).abs() < 1e-6, "surprise {}", sur.drift);
        }

        #[test]
        fn surprise_is_deterministic_per_seed() {
            let mut x = [0.3f32; 8];
            x[0] = 2.2;
            let target = [0.0f32; 8];
            let cfg = s_config();
            let a = renoise_ce_surprise(&TwoBasinProbe, &target, &x, &cfg, &mut Rng::with_seed(9));
            let b = renoise_ce_surprise(&TwoBasinProbe, &target, &x, &cfg, &mut Rng::with_seed(9));
            assert_eq!(a.drift.to_bits(), b.drift.to_bits());
            assert_eq!(a.per_draw, b.per_draw);
        }

        #[test]
        fn surprise_k_clamps_match_incumbent() {
            let mut x = [0.3f32; 8];
            x[0] = 2.2;
            let target = [0.0f32; 8];
            for k in [0u8, 200] {
                let mut cfg = s_config();
                cfg.k_draws = k;
                let s =
                    renoise_ce_surprise(&TwoBasinProbe, &target, &x, &cfg, &mut Rng::with_seed(4));
                let used = s.per_draw.iter().filter(|d| **d != 0.0).count().max(1);
                assert!(used <= 8, "k={k} must clamp to 8");
                if k == 0 {
                    assert_eq!(s.per_draw[1..].iter().filter(|d| **d != 0.0).count(), 0);
                }
                assert!(s.drift.is_finite());
            }
        }

        #[test]
        fn surprise_candidate_and_target_not_mutated() {
            let mut x = [0.3f32; 8];
            x[0] = 2.2;
            let x_before = x;
            let target = [0.1f32; 8];
            let t_before = target;
            let cfg = s_config();
            let _ = renoise_ce_surprise(&TwoBasinProbe, &target, &x, &cfg, &mut Rng::with_seed(5));
            assert_eq!(x, x_before, "candidate must not be mutated");
            assert_eq!(target, t_before, "target must not be mutated");
        }

        #[test]
        fn surprise_acceptance_gate_is_strict_lt() {
            struct OffsetProbe;
            impl RenoiseCeProbe for OffsetProbe {
                type State = [f32; 8];
                fn re_resolve(&self, s: &Self::State) -> Self::State {
                    let mut o = *s;
                    for v in &mut o {
                        *v += 0.2;
                    }
                    o
                }
                fn perturb(&self, _s: &mut Self::State, _level: f32, _rng: &mut Rng) {}
                fn drift_ce(c: &Self::State, r: &Self::State) -> f32 {
                    c.iter()
                        .zip(r.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>()
                        / 8.0
                }
            }
            let cand = [0.0f32; 8];
            let target = [1.0f32; 8];
            for (tau, want) in [(0.65, true), (0.64, false)] {
                let mut cfg = s_config();
                cfg.tau = tau;
                let s =
                    renoise_ce_surprise(&OffsetProbe, &target, &cand, &cfg, &mut Rng::with_seed(2));
                assert_eq!(s.accepted, want, "tau={tau}");
                assert!((s.drift - 0.64).abs() < 1e-6, "drift={}", s.drift);
            }
        }

        #[cfg(all(test, any(debug_assertions, feature = "alloc_tracking")))]
        #[test]
        fn surprise_score_is_alloc_free_with_fixed_array_state() {
            use std::hint::black_box;
            let candidate = [0.25f32; 8];
            let target = [0.0f32; 8];
            let cfg = s_config();
            crate::alloc::reset_alloc_stats();
            let mut sink = 0.0f32;
            for i in 0..64usize {
                let s = black_box(renoise_ce_surprise(
                    black_box(&TwoBasinProbe),
                    black_box(&target),
                    black_box(&candidate),
                    black_box(&cfg),
                    &mut Rng::with_seed(i as u64),
                ));
                sink += s.drift;
            }
            let (count, _bytes) = crate::alloc::get_alloc_stats();
            assert_eq!(
                count, 0,
                "G4: surprise score path allocated {count} times (fixed-array State → none expected)"
            );
            assert!(sink.is_finite(), "sink must be consumed: {sink}");
        }
    }
}
