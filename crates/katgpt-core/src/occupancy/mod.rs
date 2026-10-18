//! `occupancy` — generic modelless fitted occupancy-ratio estimator (FORE).
//!
//! # Plan 438 / Research 423 / arXiv:2607.05375
//!
//! Source paper: van der Laan & Kallus, *"Fitted Occupancy-Ratio Evaluation
//! without Bellman Completeness"* (2026, [arXiv:2607.05375](https://arxiv.org/abs/2607.05375)).
//!
//! This module ships the **open engine primitive**: a generic fitted-iteration
//! estimator for the discounted occupancy ratio `ω_π,γ = d^π,γ / d_ν` in
//! offline policy evaluation. The substrate-independent contribution is the
//! **adjoint Bellman KL contraction** (paper Lemma 3.1) — the operator
//! `B^γ_π ω = (1−γ)ω_0 + γ · d((ων)P_π)/dν` contracts relative entropy by
//! factor `γ` per fitted iteration, so FORE converges under **realizability
//! alone** (no Bellman completeness of a value/critic class required).
//!
//! # Algorithm
//!
//! Each FORE iteration solves a single-level convex KL projection (paper
//! Algorithm 1, verified 2026-07-14):
//!
//! ```text
//! ĥ_{k+1} = arg min_{h∈H} {
//!     log( (1/n) Σ_i e^{h(X_i)} )                                    // Λ̂_ν(h)
//!   − (1−γ) · (1/m) Σ_j h(X0_j)                                      // P̂_0(h)
//!   − γ · ( Σ_i ω̂^(k)(X_i) · h(X^+_i) ) / ( Σ_i ω̂^(k)(X_i) )        // P̂^+_{n,ω̂^(k)}(h)
//! }
//! ω̂^(k+1)(x) = e^{ĥ_{k+1}(x)} / ( (1/n) Σ_j e^{ĥ_{k+1}(X_j) } )
//! ```
//!
//! For `LinearLogRatioClass`, `h_θ(x) = θ^T x` and the objective is convex in
//! θ; we solve it via Newton's method with the PSD Hessian
//! `Cov̂_{ω_θ}(φ(X))` and Cholesky back-solve (see `linear.rs`).
//!
//! # Softmax-vs-sigmoid carve-out (READ ME)
//!
//! The global `AGENTS.md` rule says "Use sigmoid not softmax". FORE's math
//! requires a **normalized exponential class** `ω_h(x) = exp(h(x) − Λ_ν(h))`,
//! which is structurally a softmax over the offline sample. This is in
//! principled tension with the sigmoid rule, and the resolution is:
//!
//! The sigmoid rule's intent is "don't use softmax where sigmoid-on-direction-
//! vectors suffices for projections onto learned directions" (the semantic-
//! domain rule). FORE's use of softmax is **not a projection onto learned
//! directions** — it is a **density-ratio normalization** over a discrete
//! offline sample, which is the correct mathematical operation (the log-
//! partition `Λ_ν(h)` is the cumulant-generating function of the empirical
//! distribution). The sigmoid rule does not apply to density-ratio
//! normalization; it applies to direction-vector projections.
//!
//! This is the same carve-out already documented in
//! [`crate::product_key_memory`]: "Deviation from the global sigmoid rule —
//! these are convex-combination coefficients over the k²-restricted candidate
//! set, not a probability/UQ claim."
//!
//! # Honest limitations
//!
//! - **Offline transition data is the binding input.** FORE requires one-step
//!   target-policy successor pairs `(X_i, X^+_i)` where `X^+_i ∼ P_π(·|X_i)`.
//!   For NPC consumers, this means the engram/delta_mem subsystem must record
//!   not just `(state, action, reward, next-state)` but also
//!   `(next-state, target-policy-action)`. This is additional instrumentation
//!   on the consumer side, not a free lunch.
//! - **Continuous high-dimensional state spaces** are the paper's acknowledged
//!   limitation (§7). The log-ratio class must approximate `ω_π,γ` in
//!   `L²(ν)`. For 8-dim belief scalars or 64-dim `style_weights`, this is
//!   feasible. For raw pixel state or 1000+-d transformer activations, it is
//!   not. Do not over-promote.
//! - **The DEC codifferential isomorphism** (Research 423 §2.2 — "adjoint
//!   Bellman ≡ codifferential in the Markov-kernel cochain complex") is an
//!   architectural insight, not a proven theorem. It may yield a Lean 4
//!   theorem eventually; treat it as a research hypothesis until proven.
//!
//! See [`kl_contraction`] for the theorem statement.
//!
//! ── FORE — Fitted Occupancy-Ratio Estimator (Plan 438, Research 423, arxiv 2607.05375) ─
//!
//! Open primitive: generic modelless fitted-iteration estimator for the
//! discounted occupancy ratio ω_π,γ = d^π,γ / d_ν in offline policy evaluation.
//! The substrate-independent contribution is the adjoint Bellman KL contraction
//! (paper Lemma 3.1): each fitted KL projection contracts relative entropy by
//! factor γ, so convergence requires only realizability of the target ratio —
//! no Bellman completeness of a value/critic class. Three downstream fusion
//! targets (CLR re-estimation stabilization, freeze/thaw convergence guarantee,
//! FORE-ratio state equivalence) are tracked in Research 423 as out-of-scope
//! follow-ups requiring PoC validation. Phase 1 ships the type/trait surface
//! only; Phase 2 adds the fitted-iteration loop once Algorithm 1 is verified.
//! Opt-in — promotion to default-on requires a downstream consumer to
//! demonstrate the gain empirically (riir-poc Fusion A).

mod linear;
mod solve;
mod types;

pub use linear::LinearLogRatioClass;
pub use types::{InitialMoments, KlProjectionScratch, TransitionBatch};

/// Theorem-statement module for the adjoint Bellman KL contraction
/// (paper Lemma 3.1). Doc-only — no implementation.
pub mod kl_contraction;

/// Relative-θ convergence tolerance for the outer FORE loop. If the θ vector
/// changes by less than this between iterations, the loop exits early.
const FORE_THETA_TOL: f32 = 1e-6;

/// Occupancy-ratio estimator: fitted-iteration loop over a [`LogRatioClass`].
///
/// Holds the discount `gamma`, the iteration count `k_iterations`, and the
/// log-ratio class `H` (the supervised learner that realizes `log ω_π,γ`).
/// The `fit` method runs K rounds of KL projection; each round contracts
/// relative entropy by factor `gamma` (Lemma 3.1).
pub struct OccupancyRatioEstimator<H: LogRatioClass> {
    /// The supervised learner realizing `h(x) = log ω_π,γ(x)` up to a constant.
    pub log_ratio_class: H,
    /// Discount factor `gamma ∈ [0, 1)`. KL contraction factor per iteration.
    pub gamma: f32,
    /// Number of fitted KL-projection iterations K.
    pub k_iterations: usize,
}

impl<H: LogRatioClass> OccupancyRatioEstimator<H> {
    /// Construct a new estimator. Panics if `gamma >= 1.0` or `gamma < 0.0`
    /// (the contraction guarantee requires `gamma ∈ [0, 1)`).
    #[must_use]
    pub fn new(log_ratio_class: H, gamma: f32, k_iterations: usize) -> Self {
        assert!(
            (0.0..1.0).contains(&gamma),
            "gamma must be in [0, 1) for KL contraction; got {gamma}"
        );
        Self {
            log_ratio_class,
            gamma,
            k_iterations,
        }
    }

    /// Run K rounds of KL-projected adjoint Bellman iteration (Algorithm 1).
    ///
    /// Returns `ω̂^(K)(X_i)` for each transition `i = 0, ..., n−1`. The inner
    /// KL-projection loop is allocation-free (G4): all scratch is pre-allocated
    /// once in [`KlProjectionScratch`] and reused. The outer loop allocates the
    /// output `Vec<f32>` and two work buffers (`ratio`, `next_ratio`) once.
    ///
    /// # Early exit
    ///
    /// The loop exits early if the θ vector changes by less than
    /// `FORE_THETA_TOL` (relative infinity norm) between iterations — this
    /// indicates convergence to the FORE fixed point.
    #[must_use]
    pub fn fit(&self, transitions: &TransitionBatch<'_>, initial: &InitialMoments<'_>) -> Vec<f32>
    where
        H::Params: AsRef<[f32]> + AsMut<[f32]>,
    {
        let n = transitions.n;
        let d = self.log_ratio_class.feature_dim();
        let mut scratch = KlProjectionScratch::new(n, d);
        scratch.compute_initial_mean(initial);

        // ω̂^(0)(x) ≡ 1.
        let mut ratio = vec![1.0_f32; n];
        let mut next_ratio = vec![0.0_f32; n];
        let mut params = self.log_ratio_class.new_params(); // θ = 0
        let mut prev_params = self.log_ratio_class.new_params();

        for _k in 0..self.k_iterations {
            // Snapshot θ before the update (for convergence check).
            prev_params.as_mut().copy_from_slice(params.as_ref());

            self.log_ratio_class.fit_and_evaluate(
                transitions,
                initial,
                &ratio,
                self.gamma,
                &mut params,
                &mut next_ratio,
                &mut scratch,
            );
            std::mem::swap(&mut ratio, &mut next_ratio);

            // Convergence check: ||θ − θ_prev||_∞ / (||θ_prev||_∞ + ε) < tol.
            let mut delta_inf = 0.0_f32;
            let mut prev_inf = 0.0_f32;
            for (&cur, &prev) in params.as_ref().iter().zip(prev_params.as_ref().iter()) {
                let diff = (cur - prev).abs();
                if diff > delta_inf {
                    delta_inf = diff;
                }
                if prev.abs() > prev_inf {
                    prev_inf = prev.abs();
                }
            }
            if delta_inf / (prev_inf + FORE_THETA_TOL) < FORE_THETA_TOL {
                break;
            }
        }

        ratio
    }
}

/// Direct reward-reweighting value estimate (paper §5.1, `g = r`).
///
/// `V̂^π = (1/n) Σ_i ω(X_i) · r_i`
///
/// This is the simplest downstream application of a fitted ratio. For the
/// doubly-robust variant (paper §5.1, `Ψ^DR`), see the consumer-side code.
/// Kept as a free function — it needs no class state.
#[inline]
#[must_use]
pub fn value_estimate(ratio: &[f32], rewards: &[f32]) -> f32 {
    debug_assert_eq!(ratio.len(), rewards.len());
    if ratio.is_empty() {
        return 0.0;
    }
    let n = ratio.len() as f32;
    let mut acc = 0.0_f32;
    for (w, &r) in ratio.iter().zip(rewards.iter()) {
        acc += w * r;
    }
    acc / n
}

/// Trait for the supervised learner realizing `h(x) = log ω_π,γ(x)` (up to a
/// normalization constant absorbed by the log-partition `Λ_ν(h)`).
///
/// The trait is generic over the parameterization (`Self::Params`) so that
/// any sufficiently rich function class can be plugged in: linear features,
/// Fourier features, or a frozen-direction dot-product class. The FORE
/// convergence guarantee (Theorems 4.1, 4.2) requires only that the class
/// **realizes** the target ratio `ω_π,γ` — no Bellman completeness needed.
///
/// # Modelless constraint (G5)
///
/// `fit_and_evaluate` may use gradient descent on its **own** parameters
/// (`Self::Params`), but must NOT touch any base weight: no `NeuronShard`,
/// no `LoRAWeightVersion`, no `SenseModule` handle may appear in the impl.
/// The primitive is modelless by construction.
pub trait LogRatioClass {
    /// The parameterization of the log-ratio function `h(x)`.
    type Params;

    /// Feature dimension `d` (= `state_dim` for identity features).
    fn feature_dim(&self) -> usize;

    /// Allocate a fresh parameter vector initialized to the neutral point
    /// (`θ = 0`, corresponding to `ω ≡ 1`). Called once by `fit`.
    fn new_params(&self) -> Self::Params;

    /// Evaluate `h(x)` at a single point. Returns the (un-normalized) log-ratio
    /// score; the caller normalizes via the log-partition `Λ_ν(h)`.
    fn evaluate(&self, params: &Self::Params, x: &[f32]) -> f32;

    /// Fit one KL projection step (Algorithm 1 step 4) and evaluate the
    /// updated ratio (Algorithm 1 step 5).
    ///
    /// Given the current ratio `current_ratio[i] ≈ ω̂^(k)(X_i)` at each
    /// transition, solve the convex KL projection onto the log-ratio class,
    /// then compute the empirically-normalized exponential
    /// `next_ratio[i] = ω̂^(k+1)(X_i)`.
    ///
    /// The `params` buffer is both read (as the Newton warm-start point) and
    /// written (the converged θ). The `scratch` buffer is reused across
    /// iterations (G4 alloc-free inner loop); implementations must
    /// `clear_iteration()` rather than re-allocate.
    #[allow(clippy::too_many_arguments)]
    fn fit_and_evaluate(
        &self,
        transitions: &TransitionBatch<'_>,
        initial: &InitialMoments<'_>,
        current_ratio: &[f32],
        gamma: f32,
        params: &mut Self::Params,
        next_ratio: &mut [f32],
        scratch: &mut KlProjectionScratch,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: construct a tiny 2-state MRP, run FORE, assert the output
    /// is finite and non-negative. Exercises the full `fit` →
    /// `fit_and_evaluate` → Newton path without asserting precise values
    /// (those are the G1 gate in Phase 3).
    #[test]
    fn smoke_fit_produces_finite_nonneg_ratios() {
        // 2-state MRP, state_dim = 1, identity feature.
        // States encoded as f32: state 0 → 0.0, state 1 → 1.0.
        // ν = uniform, d_0 = δ_0, P(0→0)=P(0→1)=P(1→0)=P(1→1)=0.5.
        // γ = 0.9.
        //
        // True occupancy: d^π,γ(0) = 0.55, d^π,γ(1) = 0.45
        // True ratio:     ω(0) = 1.1,          ω(1) = 0.9
        let n = 200;
        let gamma = 0.9_f32;
        // Generate a reproducible mix: 100 transitions from state 0, 100 from
        // state 1, successors alternate.
        let mut states = Vec::with_capacity(n);
        let mut successors = Vec::with_capacity(n);
        for i in 0..n {
            let s = if i < n / 2 { 0.0_f32 } else { 1.0_f32 };
            let succ = if i % 2 == 0 { 0.0_f32 } else { 1.0_f32 };
            states.push(s);
            successors.push(succ);
        }
        let transitions = TransitionBatch {
            states: &states,
            successors: &successors,
            rewards: None,
            n,
            state_dim: 1,
        };
        // Initial sample: all from state 0 (d_0 = δ_0).
        let initial_buf = vec![0.0_f32; 50];
        let initial = InitialMoments {
            initial_states: &initial_buf,
            n_init: 50,
            state_dim: 1,
        };

        let class = LinearLogRatioClass::new(1);
        let est = OccupancyRatioEstimator::new(class, gamma, 20);
        let ratio = est.fit(&transitions, &initial);

        // Sanity: all finite, all non-negative, mean ≈ 1 (normalized).
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        let mut mean = 0.0_f32;
        for &r in &ratio {
            assert!(r.is_finite(), "ratio entry not finite: {r}");
            assert!(r >= 0.0, "ratio entry negative: {r}");
            min_v = min_v.min(r);
            max_v = max_v.max(r);
            mean += r;
        }
        mean /= n as f32;
        // Normalized density: empirical mean should be ≈ 1.0.
        assert!(
            (mean - 1.0).abs() < 0.01,
            "mean ratio should be ≈ 1.0 (normalized), got {mean}"
        );
        // Non-degenerate: the two states should get different ratios.
        assert!(max_v > min_v, "ratios should vary across states");
    }

    /// `value_estimate` computes `mean(ω · r)`.
    #[test]
    fn value_estimate_is_weighted_mean() {
        let ratio = [1.0_f32, 2.0, 3.0];
        let rewards = [10.0_f32, 20.0, 30.0];
        // (1·10 + 2·20 + 3·30) / 3 = (10+40+90)/3 = 140/3 ≈ 46.667
        let v = value_estimate(&ratio, &rewards);
        assert!((v - 140.0 / 3.0).abs() < 1e-4, "got {v}");
    }

    /// Empty input → 0.0 (no division by zero).
    #[test]
    fn value_estimate_empty_returns_zero() {
        let v = value_estimate(&[], &[]);
        assert_eq!(v, 0.0);
    }
}
