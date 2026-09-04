//! Data containers for the occupancy-ratio estimator.
//!
//! All containers are borrow-only (zero-copy over caller-owned buffers).
//! Scratch buffers are owned by the caller and reused across iterations to
//! keep the inner KL-projection loop allocation-free (G4).

/// One-step offline transition batch `(X_i, X^+_i)`.
///
/// States are stored as flattened `&[f32]` slices of length `n * state_dim`.
/// The successor `X^+_i` is sampled from the **target** policy's transition
/// kernel `P_π(·|X_i)` — not the behavior policy. For NPC consumers this
/// requires engram/delta_mem instrumentation to record
/// `(next-state, target-policy-action)` pairs (see module-level limitation
/// note).
///
/// For `LinearLogRatioClass`, the raw state slice IS the feature vector
/// (`state_dim = feature_dim`, identity feature map). Consumers with nonlinear
/// features (Fourier, RKS) pre-compute them and pass the result as `states`.
///
/// `rewards` is optional; present when the caller intends to compute the
/// downstream value estimate `V̂^π = mean(ω · r)`.
#[derive(Debug, Clone, Copy)]
pub struct TransitionBatch<'a> {
    /// Flattened `[n * state_dim]` source states `X_i`.
    pub states: &'a [f32],
    /// Flattened `[n * state_dim]` successor states `X^+_i ∼ P_π(·|X_i)`.
    pub successors: &'a [f32],
    /// Optional `[n]` rewards `r_i` for downstream value estimation.
    pub rewards: Option<&'a [f32]>,
    /// Number of transitions.
    pub n: usize,
    /// Dimension of each state vector.
    pub state_dim: usize,
}

impl<'a> TransitionBatch<'a> {
    /// Get the `i`-th source state as a `&[f32]` slice of length `state_dim`.
    ///
    /// Returns `None` if `i >= n`.
    #[inline]
    #[must_use]
    pub fn state(&self, i: usize) -> Option<&'a [f32]> {
        if i < self.n {
            Some(&self.states[i * self.state_dim..(i + 1) * self.state_dim])
        } else {
            None
        }
    }

    /// Get the `i`-th successor state as a `&[f32]` slice of length `state_dim`.
    ///
    /// Returns `None` if `i >= n`.
    #[inline]
    #[must_use]
    pub fn successor(&self, i: usize) -> Option<&'a [f32]> {
        if i < self.n {
            Some(&self.successors[i * self.state_dim..(i + 1) * self.state_dim])
        } else {
            None
        }
    }
}

/// Independent initial-state sample for the `P̂_0(h)` term (paper §3.3).
///
/// The paper's Algorithm 1 requires `P̂_0(h) = (1/m) Σ_j h(X0_j)` where
/// `X0_j ∼ d_0` is an **independent** sample from the target initial state-
/// action distribution (not a subsample of the offline data). For linear
/// `h_θ(x) = θ^T φ(x)`, the FORE inner loop only needs the mean feature
/// vector `P̂_0(φ) = (1/m) Σ_j φ(X0_j)`, which is precomputed once by
/// [`KlProjectionScratch::compute_initial_mean`] and reused across iterations.
///
/// `initial_states` is the flattened `[n_init * state_dim]` buffer of `X0_j`.
#[derive(Debug, Clone, Copy)]
pub struct InitialMoments<'a> {
    /// Flattened `[n_init * state_dim]` initial samples `X0_j ∼ d_0`.
    pub initial_states: &'a [f32],
    /// Number of initial samples.
    pub n_init: usize,
    /// Dimension of each state vector (must match [`TransitionBatch::state_dim`]).
    pub state_dim: usize,
}

/// Pre-allocated scratch buffers for the KL-projection inner loop.
///
/// Constructed once before the fitted-iteration loop and reused on every
/// iteration to keep the inner loop allocation-free (G4). The fields are sized
/// to `n` (the transition count) or `feature_dim` (the log-ratio class
/// parameter dimension) at construction time.
///
/// # Buffer layout (Algorithm 1 Newton step)
///
/// The KL projection for `LinearLogRatioClass` solves the convex objective
/// `L(θ) = log(1/n Σ e^{θ·φ(Xi)}) − θ^T m` via Newton's method. Each Newton
/// step needs:
/// - `exp_buf[n]` — `e^{θ·φ(Xi)}` per transition (reused for normalization)
/// - `theta[feature_dim]` — current Newton iterate (written into `Params`)
/// - `gradient[feature_dim]` — `∇L = Ê_ν[ω_θ(X)φ(X)] − m`
/// - `hessian[feature_dim²]` — `Cov̂_{ω_θ}(φ(X))` (PSD, Cholesky-factored)
/// - `cholesky[feature_dim²]` — Cholesky factor `L` (overwrites Hessian in place)
/// - `newton_step[feature_dim]` — `Δθ = H^{-1} ∇L` (the solve target)
/// - `y_buf[feature_dim]` — triangular solve scratch
/// - `moment[feature_dim]` — fixed moment `m = (1−γ)P̂_0(φ) + γP̂^+_{n,ω}(φ)`
/// - `initial_mean[feature_dim]` — precomputed `P̂_0(φ)`, set once before the loop
/// - `successor_weighted_sum[feature_dim]` — `(Σ_i ω(Xi)φ(X^+_i))` accumulator
/// - `weighted_feature_sum[feature_dim]` — `Σ_i ω_θ(Xi)φ(Xi)` for gradient/Hessian
#[derive(Debug, Clone)]
pub struct KlProjectionScratch {
    /// `[n]` exponentiated scores `e^{θ·φ(Xi)}` per transition.
    pub exp_buf: Vec<f32>,
    /// `[feature_dim]` fixed moment vector `m` for the current iteration.
    pub moment: Vec<f32>,
    /// `[feature_dim]` precomputed `P̂_0(φ)` — mean feature over initial sample.
    pub initial_mean: Vec<f32>,
    /// `[feature_dim]` weighted successor-feature sum (numerator of P̂^+_{n,ω}).
    pub successor_weighted_sum: Vec<f32>,
    /// `[feature_dim]` gradient `∇L` (Newton residual).
    pub gradient: Vec<f32>,
    /// `[feature_dim²]` Hessian `Cov̂_{ω_θ}(φ(X))`, preserved across LM retries.
    /// The damped copy (H + λI) lives in [`Self::hessian_damped`]; Cholesky
    /// overwrites that, leaving this buffer intact for retries with different λ.
    pub hessian: Vec<f32>,
    /// `[feature_dim²]` damped Hessian copy `H + λI` — Cholesky overwrites this,
    /// leaving `hessian` intact for LM retries with different λ.
    pub hessian_damped: Vec<f32>,
    /// `[feature_dim]` Newton step `Δθ` (the solve target).
    pub newton_step: Vec<f32>,
    /// `[feature_dim]` trial θ for LM line-search loss evaluation.
    pub params_trial: Vec<f32>,
    /// `[feature_dim]` triangular-solve scratch for `cholesky_solve_into`.
    pub y_buf: Vec<f32>,
    /// `[feature_dim]` accumulator for `Σ_i ω_θ(Xi) φ(Xi)` (gradient/Hessian shared).
    pub weighted_feature_sum: Vec<f32>,
    /// Cached `n` (transition count) and `feature_dim` for asserts.
    pub n: usize,
    pub feature_dim: usize,
}

impl KlProjectionScratch {
    /// Allocate scratch buffers sized for `n` transitions and `feature_dim`
    /// parameters. The vectors are allocated once here and reused inside the
    /// fitted-iteration loop (G4 alloc-free inner loop).
    #[must_use]
    pub fn new(n: usize, feature_dim: usize) -> Self {
        Self {
            exp_buf: vec![0.0; n],
            moment: vec![0.0; feature_dim],
            initial_mean: vec![0.0; feature_dim],
            successor_weighted_sum: vec![0.0; feature_dim],
            gradient: vec![0.0; feature_dim],
            hessian: vec![0.0; feature_dim * feature_dim],
            hessian_damped: vec![0.0; feature_dim * feature_dim],
            newton_step: vec![0.0; feature_dim],
            params_trial: vec![0.0; feature_dim],
            y_buf: vec![0.0; feature_dim],
            weighted_feature_sum: vec![0.0; feature_dim],
            n,
            feature_dim,
        }
    }

    /// Precompute `P̂_0(φ) = (1/m) Σ_j φ(X0_j)` from the independent initial
    /// sample. Called **once** before the fitted-iteration loop; the result
    /// is reused across all K iterations (the initial distribution does not
    /// change).
    ///
    /// For `LinearLogRatioClass`, `state_dim == feature_dim` and the raw
    /// initial state IS the feature vector, so this is a plain column mean.
    pub fn compute_initial_mean(&mut self, initial: &InitialMoments<'_>) {
        debug_assert_eq!(initial.state_dim, self.feature_dim);
        debug_assert_eq!(self.initial_mean.len(), self.feature_dim);
        self.initial_mean.fill(0.0);
        if initial.n_init == 0 {
            return;
        }
        for j in 0..initial.n_init {
            let row = &initial.initial_states[j * self.feature_dim..(j + 1) * self.feature_dim];
            for (slot, &val) in self.initial_mean.iter_mut().zip(row.iter()) {
                *slot += val;
            }
        }
        let inv = 1.0 / (initial.n_init as f32);
        for slot in &mut self.initial_mean {
            *slot *= inv;
        }
    }

    /// Reset all per-iteration buffers to zero (capacity preserved). Called at
    /// the top of each KL-projection iteration. `initial_mean` is NOT reset —
    /// it persists across iterations (computed once by `compute_initial_mean`).
    #[inline]
    pub fn clear_iteration(&mut self) {
        self.exp_buf.fill(0.0);
        self.moment.fill(0.0);
        self.successor_weighted_sum.fill(0.0);
        self.gradient.fill(0.0);
        self.hessian.fill(0.0);
        self.hessian_damped.fill(0.0);
        self.newton_step.fill(0.0);
        self.params_trial.fill(0.0);
        self.y_buf.fill(0.0);
        self.weighted_feature_sum.fill(0.0);
    }
}
