//! Linear log-ratio class `h_θ(x) = θ^T φ(x)` with identity feature map.
//!
//! This is the concrete supervised learner for FORE's G1/G2 gates. The KL
//! projection (Algorithm 1 step 4) reduces to a convex optimization:
//!
//! ```text
//! min_θ  L(θ) = log( (1/n) Σ_i e^{θ·φ(Xi)} )  −  θ^T m
//! ```
//!
//! where `m = (1−γ) P̂_0(φ) + γ P̂^+_{n,ω}(φ)` is a fixed moment vector for the
//! current FORE iteration (computed once per call to `fit_and_evaluate`). The
//! loss is convex (log-sum-exp minus linear); we solve it via Newton's method
//! with the PSD Hessian `Cov̂_{ω_θ}(φ(X))` and Cholesky back-solve.

use super::LogRatioClass;
use super::solve::{cholesky_inplace, cholesky_solve_into};
use super::types::{InitialMoments, KlProjectionScratch, TransitionBatch};

/// 3-iterator zip helper (avoids pulling the `itertools` crate for one fn).
/// Takes slices by reference; returns an iterator of `(&mut A, &B, &C)`.
#[inline]
fn izip3_mut<'a, A, B, C>(
    a: &'a mut [A],
    b: &'a [B],
    c: &'a [C],
) -> impl Iterator<Item = (&'a mut A, &'a B, &'a C)> {
    a.iter_mut()
        .zip(b.iter())
        .zip(c.iter())
        .map(|((a, b), c)| (a, b, c))
}

/// Evaluate the KL-projection loss `L(θ) = log((1/n)Σ e^{θ·φ_i}) − θ^T m` at the
/// given `θ`. Used by the LM line search to check whether a trial step
/// decreases the loss. Allocation-free — accumulates in registers.
///
/// Computed in **f64** for the line-search acceptance check: near the FORE
/// fixed point, the loss surface is extremely flat (gradient ~1e-3, curvature
/// ~1e-2), and f32 rounding makes `L(θ) == L(θ ± δ)` for reasonable step sizes.
/// f64 gives enough precision to distinguish trial steps that genuinely
/// decrease L. The gradient/Hessian computations in the main Newton loop stay
/// f32 (consistent with the rest of katgpt-core); only this acceptance gate
/// promotes to f64.
#[inline]
fn compute_loss(
    params: &[f32],
    transitions: &TransitionBatch<'_>,
    moment: &[f32],
    d: usize,
    n: usize,
) -> f64 {
    // Pass 1: find max score for log-sum-exp stability.
    let mut max_score = f64::NEG_INFINITY;
    for i in 0..n {
        let phi = &transitions.states[i * d..(i + 1) * d];
        let mut s = 0.0_f64;
        for (th, &v) in params.iter().zip(phi.iter()) {
            s += *th as f64 * v as f64;
        }
        if s > max_score {
            max_score = s;
        }
    }
    if !max_score.is_finite() {
        max_score = 0.0;
    }
    // Pass 2: accumulate z_sum = Σ e^{θ·φ_i − max_score}.
    let mut z_sum = 0.0_f64;
    for i in 0..n {
        let phi = &transitions.states[i * d..(i + 1) * d];
        let mut s = 0.0_f64;
        for (th, &v) in params.iter().zip(phi.iter()) {
            s += *th as f64 * v as f64;
        }
        z_sum += (s - max_score).exp();
    }
    // log_partition = log((1/n) Σ e^{θ·φ_i}) = log(z_sum) + max_score − log(n).
    let log_partition = z_sum.ln() + max_score - (n as f64).ln();
    // theta_dot_m = θ^T m.
    let mut theta_dot_m = 0.0_f64;
    for (th, &m) in params.iter().zip(moment.iter()) {
        theta_dot_m += *th as f64 * m as f64;
    }
    log_partition - theta_dot_m
}

/// Maximum Newton iterations per KL projection. Quadratic convergence makes
/// 50 ample even for ill-conditioned cases; the typical count is < 10.
const MAX_NEWTON_ITERS: usize = 50;
/// Newton convergence tolerance on `||∇L||_∞`. Right at the f32 precision
/// floor; the 1% G1 gate has enormous headroom above this.
const NEWTON_TOL: f32 = 1e-6;

// ── Levenberg-Marquardt damping parameters ────────────────────────────────
//
// Pure Newton overshoots badly when the Hessian is ill-conditioned (e.g. the
// scalar Baird-MRP G1 fixture: H = Cov(φ) ≈ 0.038 at θ=0, gradient ≈ −0.76,
// Newton step |H⁻¹g| ≈ 19.8 — overshooting θ⋆ ≈ 5.65 by 3.5×). Once θ
// overshoots, all weight collapses to one state and H → 0, making recovery
// impossible.
//
// The fix: add λ·I to the Hessian before Cholesky (LM damping). λ starts
// conservative (gradient-descent-like), decreases on accepted steps (toward
// pure Newton quadratic convergence), increases on rejected steps. This is
// the standard fix for Newton overshoot on poorly-scaled problems.

/// Initial LM damping. λ = 1.0 gives a well-damped first step for both the
/// scalar Baird fixture (H ≈ 0.04 → H+1 ≈ 1.04, step ≈ 0.73) and the 8-dim
/// G2 fixture (H diagonal ≈ 1 → H+1, step ≈ g/2).
const LM_INIT: f32 = 1.0;
/// Minimum LM damping — effectively pure Newton below this.
const LM_MIN: f32 = 1e-6;
/// λ multiplier on accepted step (decrease damping toward Newton).
const LM_DECREASE: f32 = 0.25;
/// λ multiplier on rejected step (increase damping toward gradient descent).
const LM_INCREASE: f32 = 4.0;
/// Maximum LM retries per Newton iteration before giving up.
const MAX_LM_RETRIES: usize = 12;

/// Linear log-ratio class `h_θ(x) = θ^T · x` (identity feature map).
///
/// The raw state slice IS the feature vector: `state_dim == feature_dim`.
/// Consumers with nonlinear features (Fourier, Random Kitchen Sinks) pre-
/// compute them and pass the result as `states` in [`TransitionBatch`].
///
/// # Modelless-ness (G5)
///
/// The only mutable state is `θ` (`Vec<f32>`). No `NeuronShard`,
/// `LoRAWeightVersion`, or `SenseModule` handle is touched anywhere in this
/// module. The primitive is modelless by construction.
#[derive(Debug, Clone)]
pub struct LinearLogRatioClass {
    /// Feature dimension `d` (= `state_dim` for identity features).
    pub feature_dim: usize,
}

impl LinearLogRatioClass {
    /// Construct a new linear class with `feature_dim` parameters.
    #[must_use]
    pub fn new(feature_dim: usize) -> Self {
        Self { feature_dim }
    }
}

impl LogRatioClass for LinearLogRatioClass {
    type Params = Vec<f32>;

    #[inline]
    fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    fn new_params(&self) -> Self::Params {
        vec![0.0; self.feature_dim]
    }

    #[inline]
    fn evaluate(&self, params: &Self::Params, x: &[f32]) -> f32 {
        debug_assert_eq!(params.len(), self.feature_dim);
        debug_assert!(x.len() >= self.feature_dim);
        let mut s = 0.0_f32;
        for (p, &v) in params.iter().zip(x.iter()) {
            s += p * v;
        }
        s
    }

    fn fit_and_evaluate(
        &self,
        transitions: &TransitionBatch<'_>,
        initial: &InitialMoments<'_>,
        current_ratio: &[f32],
        gamma: f32,
        params: &mut Self::Params,
        next_ratio: &mut [f32],
        scratch: &mut KlProjectionScratch,
    ) {
        let n = transitions.n;
        let d = self.feature_dim;
        debug_assert_eq!(
            transitions.state_dim, d,
            "identity feature map: state_dim == feature_dim"
        );
        debug_assert_eq!(initial.state_dim, d);
        debug_assert_eq!(current_ratio.len(), n);
        debug_assert_eq!(next_ratio.len(), n);
        debug_assert_eq!(params.len(), d);
        debug_assert_eq!(scratch.n, n);
        debug_assert_eq!(scratch.feature_dim, d);
        if n == 0 {
            return;
        }

        scratch.clear_iteration();

        // ── Step 1: compute the fixed moment vector m ──────────────────
        //
        // m = (1−γ) P̂_0(φ) + γ P̂^+_{n,ω}(φ)
        //
        // P̂_0(φ) is precomputed in scratch.initial_mean (set once before the
        // FORE loop by KlProjectionScratch::compute_initial_mean).
        //
        // P̂^+_{n,ω}(φ) = (Σ_i ω(Xi) φ(X^+_i)) / (Σ_i ω(Xi))   [self-normalized]

        let omega_sum: f32 = current_ratio.iter().sum();
        if omega_sum <= 0.0 {
            // Degenerate: all-zero ratio (shouldn't happen since ω̂^(0) ≡ 1).
            // Fall through with zero successor moment.
        } else {
            let inv_omega_sum = 1.0 / omega_sum;
            for (i, &omega) in current_ratio.iter().enumerate() {
                let w = omega * inv_omega_sum;
                let succ = &transitions.successors[i * d..(i + 1) * d];
                for (slot, &s) in scratch.successor_weighted_sum.iter_mut().zip(succ.iter()) {
                    *slot += w * s;
                }
            }
        }

        for (m_slot, &p0, &succ) in izip3_mut(
            &mut scratch.moment,
            &scratch.initial_mean,
            &scratch.successor_weighted_sum,
        ) {
            *m_slot = (1.0 - gamma) * p0 + gamma * succ;
        }

        // ── Step 2: Newton iteration on L(θ) = log(1/n Σ e^{θ·φ(Xi)}) − θ^T m ─
        //
        // Warm-start from the current params (the previous FORE iteration's θ).
        // Near the fixed point this makes Newton converge in 1–3 steps.
        //
        // Each Newton step uses Levenberg-Marquardt damping: add λ·I to the
        // Hessian before Cholesky, with λ starting conservative and decreasing
        // on accepted steps (toward pure Newton quadratic convergence) and
        // increasing on rejected steps (toward gradient descent robustness).
        // This prevents the classic Newton overshoot on ill-conditioned
        // Hessians (e.g. scalar Baird-MRP fixture where H ≈ 0.04).

        let mut lambda: f32 = LM_INIT;

        for _newton_iter in 0..MAX_NEWTON_ITERS {
            // 2a. Compute exp_buf[i] = e^{θ·φ(Xi) − max_score} (log-sum-exp trick).
            let mut max_score = f32::NEG_INFINITY;
            for i in 0..n {
                let phi = &transitions.states[i * d..(i + 1) * d];
                let mut s = 0.0_f32;
                for (th, &v) in params.iter().zip(phi.iter()) {
                    s += th * v;
                }
                if s > max_score {
                    max_score = s;
                }
            }
            if !max_score.is_finite() {
                max_score = 0.0;
            }

            let mut z_sum = 0.0_f32;
            for i in 0..n {
                let phi = &transitions.states[i * d..(i + 1) * d];
                let mut s = 0.0_f32;
                for (th, &v) in params.iter().zip(phi.iter()) {
                    s += th * v;
                }
                let e = (s - max_score).exp();
                scratch.exp_buf[i] = e;
                z_sum += e;
            }
            let inv_nz = if z_sum > 0.0 { 1.0 / z_sum } else { 0.0 };
            // mean_phi[k] = Σ_i exp_buf[i] · φ(Xi)[k] / z_sum = Ê_ν[ω_θ(X) φ(X)]
            // where ω_θ(Xi) = e^{θ·φ_i} / Σ_j e^{θ·φ_j} = exp_buf[i] / z_sum.

            // 2b. Compute weighted_feature_sum = Σ_i exp_buf[i] · φ(Xi) (length d).
            scratch.weighted_feature_sum.fill(0.0);
            for i in 0..n {
                let e = scratch.exp_buf[i];
                let phi = &transitions.states[i * d..(i + 1) * d];
                for (slot, &v) in scratch.weighted_feature_sum.iter_mut().zip(phi.iter()) {
                    *slot += e * v;
                }
            }

            // 2c. Gradient: ∇L = mean_phi − m = inv_nz · weighted_feature_sum − moment
            let mut grad_inf_norm = 0.0_f32;
            for (g, &wfs, &m) in izip3_mut(
                &mut scratch.gradient,
                &scratch.weighted_feature_sum,
                &scratch.moment,
            ) {
                let mean_phi = inv_nz * wfs;
                *g = mean_phi - m;
                let abs_g = g.abs();
                if abs_g > grad_inf_norm {
                    grad_inf_norm = abs_g;
                }
            }

            // Convergence check: ||∇L||_∞ < tol.
            if grad_inf_norm < NEWTON_TOL {
                break;
            }

            // 2d. Hessian: H = Cov̂_{ω_θ}(φ(X)) = Ê_ν[ω_θ φ φ^T] − mean_phi mean_phi^T
            scratch.hessian.fill(0.0);
            for i in 0..n {
                let e = scratch.exp_buf[i];
                let phi = &transitions.states[i * d..(i + 1) * d];
                for a in 0..d {
                    if phi[a] == 0.0 {
                        continue;
                    }
                    let ea = e * phi[a];
                    let row = &mut scratch.hessian[a * d..(a + 1) * d];
                    for b in 0..d {
                        row[b] += ea * phi[b];
                    }
                }
            }
            let inv_nz_scaled = inv_nz;
            for a in 0..d {
                let mean_a = inv_nz_scaled * scratch.weighted_feature_sum[a];
                for b in 0..d {
                    let mean_b = inv_nz_scaled * scratch.weighted_feature_sum[b];
                    scratch.hessian[a * d + b] =
                        inv_nz_scaled * scratch.hessian[a * d + b] - mean_a * mean_b;
                }
            }

            // 2e. Current loss L(θ) for LM acceptance check.
            let loss_current = compute_loss(&params[..], transitions, &scratch.moment, d, n);

            // 2f. LM inner loop: try λ, λ·LM_INCREASE, λ·LM_INCREASE², ... until
            //     the damped Newton step decreases L.
            let mut step_accepted = false;
            for _retry in 0..MAX_LM_RETRIES {
                // Copy H → H_damped, add λ·I, Cholesky, solve.
                scratch.hessian_damped.copy_from_slice(&scratch.hessian);
                for a in 0..d {
                    scratch.hessian_damped[a * d + a] += lambda;
                }

                if !cholesky_inplace(&mut scratch.hessian_damped, d) {
                    // Singular even with damping — increase λ and retry.
                    lambda *= LM_INCREASE;
                    continue;
                }
                cholesky_solve_into(
                    &scratch.hessian_damped,
                    &scratch.gradient,
                    d,
                    &mut scratch.y_buf,
                    &mut scratch.newton_step,
                );

                // Trial: θ_trial = θ − Δθ.
                for (trial, &th, &delta) in
                    izip3_mut(&mut scratch.params_trial, &params[..], &scratch.newton_step)
                {
                    *trial = th - delta;
                }

                let loss_trial =
                    compute_loss(&scratch.params_trial, transitions, &scratch.moment, d, n);

                if loss_trial < loss_current {
                    // Accept the step.
                    for (th, &trial_val) in params.iter_mut().zip(scratch.params_trial.iter()) {
                        *th = trial_val;
                    }
                    lambda = (lambda * LM_DECREASE).max(LM_MIN);
                    step_accepted = true;
                    break;
                }

                // Loss didn't decrease — increase damping.
                lambda *= LM_INCREASE;
            }

            if !step_accepted {
                // All LM retries failed to decrease L — accept current θ and
                // stop (convergence stalls; the FORE outer loop will continue).
                break;
            }
        }

        // ── Step 3: evaluate the updated ratio ω̂^(k+1)(Xi) ─────────────
        //
        // ω̂^(k+1)(Xi) = e^{θ·φ(Xi)} / (1/n Σ_j e^{θ·φ(Xj)})
        // (Algorithm 1 step 5). Uses log-sum-exp for stability.
        let mut max_score = f32::NEG_INFINITY;
        for i in 0..n {
            let phi = &transitions.states[i * d..(i + 1) * d];
            let mut s = 0.0_f32;
            for (th, &v) in params.iter().zip(phi.iter()) {
                s += th * v;
            }
            if s > max_score {
                max_score = s;
            }
        }
        if !max_score.is_finite() {
            max_score = 0.0;
        }
        let mut z_sum = 0.0_f32;
        for (i, ratio_slot) in next_ratio.iter_mut().enumerate() {
            let phi = &transitions.states[i * d..(i + 1) * d];
            let mut s = 0.0_f32;
            for (th, &v) in params.iter().zip(phi.iter()) {
                s += th * v;
            }
            let e = (s - max_score).exp();
            *ratio_slot = e;
            z_sum += e;
        }
        let inv_z = if z_sum > 0.0 { (n as f32) / z_sum } else { 1.0 };
        for slot in next_ratio {
            *slot *= inv_z;
        }
    }
}
