//! Mean-field distributional steering: Feynman-Kac weights + first-variation
//! reward table + damped-Picard Ψ̇ solver.
//!
//! **Plan 577** · Research 505 (Howard & Nüsken, "A Mean-Field Framework for
//! Inference-Time Distributional Control of Diffusion Models",
//! [arXiv:2608.08770](https://arxiv.org/abs/2608.08770), SPIGM @ ICML 2026).
//!
//! The distilled modelless question this module answers: *given a weighted
//! particle population `(states, weights)` and a measure-defined reward
//! `R(μ)` with a closed-form first variation, how do we steer the population
//! toward the exact tilted target* `μ*(dx) ∝ e^{λΨ(x,μ*)} p₁(x) dx`, where
//! `Ψ = δR/δμ`? The paper's answer has three pieces, all shipped here:
//!
//! 1. **First-variation table** ([`MeasureReward`]) — `Linear`, `Moment`,
//!    `Mmd` rows with closed-form `Ψ(x, μ)`, `∇Ψ`, and second variation
//!    (their Table 2 — pure calculus, modelless).
//! 2. **FK log-weight accumulation** ([`FkStepper`]) — `A_i += (b_i·∇Ψ_i +
//!    Ψ̇_i)·δt` with log-sum-exp normalization; the **weighted empirical
//!    measure** `μ̂ = Σ w_i δ_{X_i}` is the object that converges (Thm 3.4).
//! 3. **Damped-Picard Ψ̇ solver** (Alg 4 shape) — candidate next-weights →
//!    candidate next-measure → `Ψ̇ ≈ [Ψ(x, μ̃ₜ₊δₜ) − Ψ(x, μₜ)]/δt` → weight
//!    update; `K_FP = 3` iterations, kernel matrix computed **once per step**
//!    and reused across all Picard iterations (the paper measures Picard at
//!    0.036–0.24% of runtime because network evals dominate theirs; in a
//!    modelless stack the kernel build IS the cost — see the Bench 682 G2
//!    record for the honest breakdown).
//!
//! # Contract notes
//!
//! - **`b_i` is the consumer's own per-tick drift** (Research 505 caveat 3:
//!   discrete-time port — the theory is continuous-time Itô, Algorithm 1 is
//!   the Euler-discretized form shipped here). The consumer integrates
//!   positions themselves between [`FkStepper::begin_step`] and
//!   [`FkStepper::finish_step`].
//! - **Weights-only is the correct consumer shape for persistent agents**
//!   (Research 505 caveat 2): resampling duplicates particles, which is
//!   meaningless for persistent NPCs. The theorem tracks the *weighted*
//!   measure. [`residual_resample_into`] exists for sampling consumers only
//!   and is documented accordingly; [`systematic_resample_into`] is the
//!   deterministic alternative.
//! - **exp-tilt is not a softmax violation** (Research 505 caveat 4):
//!   `w_i ∝ e^{A_i}` is the defining math of exponential tilting (`μ* ∝
//!   e^{Ψ}p`) — an importance weight, not a semantic projection. The
//!   sigmoid-not-softmax house rule continues to govern gates/kernels; all
//!   weight normalization here is log-sum-exp, never naive exp-sum on raw
//!   logits.
//! - **Zero-allocation steady state**: [`SteeringScratch`] is fixed-capacity,
//!   allocated once at construction; every step routine writes into scratch
//!   or caller buffers (`*_into` suffix). The cold-path trait methods and
//!   resampling helpers document their allocation shape honestly.
//! - **Determinism**: fixed iteration order (index loops), no HashMap
//!   anywhere in the path; two-run bit-identity is pinned by tests + the
//!   Bench 682 gate.
//!
//! # Example (weights-only steering, 1-D MMD reward)
//!
//! ```ignore
//! use katgpt_core::distributional_steering::{
//!     FkStepper, MmdReward, SteeringScratch,
//! };
//! let n = 1000usize;
//! let dim = 1usize;
//! let mut states = vec![0.0f32; n * dim];      // particle positions
//! let mut log_w = vec![0.0f32; n];             // uniform start
//! let reward = MmdReward::new(0.1, target_particles, dim);
//! let mut stepper = FkStepper::default();       // λ=1, K_FP=3, damping 1.0
//! let mut scratch = SteeringScratch::new(n, dim);
//! for _ in 0..steps {
//!     stepper.begin_step(&reward, &states, &mut log_w, &mut scratch);
//!     let steer = scratch.steering();           // λ·∇Ψ per particle
//!     // consumer integrates: X += (b + steer)·δt + noise; keep b
//!     stepper.finish_step(&reward, &states, &b, dt, &mut log_w, &mut scratch);
//! }
//! // log_w now carries the FK tilt; μ̂ = Σ e^{log_w_i} δ_{X_i} ≈ μ*(λ)
//! ```
//!
//! distributional_steering — population steering toward a measure-defined
//! target (Plan 577, Research 505, arXiv:2608.08770): MeasureReward
//! first-variation table + Feynman-Kac log-weights with the mean-field Ψ̇
//! Picard correction + weighted empirical measure exposure. Opt-in pending
//! the Bench 682 targeting gate.

// ──────────────────────────────────────────────────────────────────────────
// Reward table (Plan 577 T1.1 / T1.2)
// ──────────────────────────────────────────────────────────────────────────

/// Closed reward-table dispatch (Plan 577 T1.1: "enum dispatch closed over a
/// small table"). `#[repr(u8)]` per the house sync-efficiency convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RewardKind {
    /// `Ψ = f(x)` — pointwise (degenerate; recovers plain steering).
    Linear = 0,
    /// `Ψ = F'(∫φ dμ)·φ(x)`.
    Moment = 1,
    /// `Ψ(x,μ) = 2∫k(x,y)(μ−ν)(dy)` — kernel contrast against a target.
    Mmd = 2,
    /// `Ψ(x) = r(x)` via a caller-supplied black-box closure (Plan 581 T1.1).
    /// Second variation 0; no closed-form ∇Ψ (the gradient arm uses central
    /// finite differences — see `ClosureReward`).
    Closure = 3,
}

/// A measure-defined reward `R(μ)` with closed-form first variation.
///
/// `Ψ(x, μ) = δR/δμ (x, μ)` is the tilt potential; the λ-scaled tilt
/// `e^{λΨ}` defines the implicit target `μ* ∝ e^{λΨ(·,μ*)} p₁`
/// (paper Prop 3.1). Rows ship `Linear` / `Moment` / `Mmd` (paper Table 2);
/// the entropy row is a documented non-goal (needs density estimation —
/// Research 505 risk 3; approximate via MMD-to-uniform instead).
pub trait MeasureReward {
    /// Table row for dispatch.
    fn kind(&self) -> RewardKind;

    /// State dimensionality `d` this reward evaluates over.
    fn dim(&self) -> usize;

    /// First variation `Ψ(x, μ̂)` at a single position `x` (d entries)
    /// against the weighted population; written to `out[0]`
    /// (`out.len() >= 1`; the slice form is the plan's T1.1 signature,
    /// kept for batch-extension headroom).
    fn first_variation_into(&self, x: &[f32], pop: &WeightedPopulation, out: &mut [f32]);

    /// Second variation `Φ(x, y, μ̂)` at two positions. Only the MMD row has
    /// a non-zero one (`2k(x,y)` — the form the implicit linear solver
    /// consumes); Linear/Moment return `0.0`.
    fn second_variation(&self, x: &[f32], y: &[f32], pop: &WeightedPopulation) -> f32;

    /// Scalar reward `R(μ̂)` (MMD row returns `−MMD²(μ̂,ν)` — higher better).
    fn reward(&self, pop: &WeightedPopulation) -> f32;

    /// Self-downcast for the closed-table hot loops (the three row types are
    /// known statically; this avoids trait-object upcasting).
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Linear reward row: `R(μ) = ∫ (a·x) dμ`, so `Ψ = a·x` — the degenerate
/// pointwise case that recovers plain (non-distributional) steering.
#[derive(Debug, Clone)]
pub struct LinearReward {
    /// Direction `a` (d entries): `f(x) = a·x`.
    pub dir: Vec<f32>,
}

/// The boxed black-box scorer behind the closure row (type alias — the
/// clippy type_complexity class).
pub type BoxedScorer = Box<dyn Fn(&[f32]) -> f32>;

/// Opaque pointwise reward row (Plan 581 T1.1): `R(μ) = ∫ r dμ` with `r` a
/// caller-supplied black-box closure — the degenerate case of R505 Prop 3.1
/// that recovers plain per-state reward steering, and the row that admits
/// scorers the Table-2 rows (Linear/Moment/Mmd — all closed-form) cannot
/// express (classifiers, validator composites, any external oracle).
///
/// # Consistency footing (Research 517 / No-GD advocate row 1)
///
/// Self-normalized twisted SMC is consistent for ANY positive ψ — the
/// closure is a *tilt potential* `Ψ(x) = r(x)`, never a modeled density.
/// Amortization layers over it (twist_smc's x̂₀ proxy / value memo / ridge
/// table) are variance reduction, never correctness.
///
/// # Gradient cost note
///
/// There is no closed-form `∇Ψ` for a black-box `r`; the [`FkStepper`]
/// gradient arm evaluates the closure `2d` times per particle (central
/// differences at `fd_eps`). That cost is exactly why twist_smc consumers
/// (Research 517 / Plan 581) steer WEIGHTS-ONLY — the twist reweights via
/// `ψ ∝ exp(β·V̂)` and never needs `∇Ψ`.
///
/// `r` must return finite values (`debug_assert` at the boundary — the
/// house `is_finite` discipline, as in `entropic_tilt::solve_beta`).
pub struct ClosureReward {
    dim: usize,
    fd_eps: f32,
    f: BoxedScorer,
}

impl ClosureReward {
    /// New row over a black-box scorer. `fd_eps` is the central-difference
    /// step for the (optional) [`FkStepper`] gradient arm; `1e-3` is a sane
    /// default for O(1)-scale rewards.
    pub fn new(dim: usize, fd_eps: f32, f: impl Fn(&[f32]) -> f32 + 'static) -> Self {
        assert!(dim > 0, "ClosureReward requires dim > 0 (got {dim})");
        assert!(
            fd_eps > 0.0 && fd_eps.is_finite(),
            "fd_eps must be positive finite"
        );
        Self {
            dim,
            fd_eps,
            f: Box::new(f),
        }
    }

    /// Evaluate `r(x)` with the finite boundary check.
    #[inline]
    fn eval(&self, x: &[f32]) -> f32 {
        let v = (self.f)(x);
        debug_assert!(v.is_finite(), "ClosureReward r(x) must be finite (got {v})");
        v
    }
}

/// Scalar gain `F` for the moment row (closed over a small table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MomentGain {
    /// `F(m) = −m²` (moment penalty), `F' = −2m`.
    NegativeSquare = 0,
    /// `F(m) = m²`, `F' = 2m`.
    Square = 1,
    /// `F(m) = m`, `F' = 1` (pure moment targeting).
    Identity = 2,
}

impl MomentGain {
    #[inline]
    fn derivative(self, m: f32) -> f32 {
        match self {
            Self::NegativeSquare => -2.0 * m,
            Self::Square => 2.0 * m,
            Self::Identity => 1.0,
        }
    }
    #[inline]
    fn value(self, m: f32) -> f32 {
        match self {
            Self::NegativeSquare => -m * m,
            Self::Square => m * m,
            Self::Identity => m,
        }
    }
}

/// Moment reward row: `R(μ) = F(∫ φ dμ)` with linear features
/// `φ(x) = p·x`, so `Ψ = F'(m)·(p·x)` where `m = Σᵢ wᵢ p·Xᵢ`.
#[derive(Debug, Clone)]
pub struct MomentReward {
    /// Gain `F` applied to the population moment.
    pub gain: MomentGain,
    /// Linear feature direction `p` (d entries).
    pub phi: Vec<f32>,
}

/// MMD reward row: `R(μ) = −MMD²(μ, ν)` against a uniformly-weighted target
/// particle set, so `Ψ(x,μ) = 2∫k(x,y)(μ−ν)(dy)` and the second variation is
/// `2k(x,y)` (paper Table 2).
///
/// The kernel follows the `mag/transfer.rs` convention
/// (`rbf_kernel(a,b,gamma) = fast_exp(−γ·‖a−b‖²)` via
/// `crate::simd::fast_exp`); it is implemented **locally** because mag's
/// helper is private to that module and no cross-feature dep is warranted
/// for four lines.
#[derive(Debug, Clone)]
pub struct MmdReward {
    /// RBF kernel scale γ (bandwidth `s` corresponds to `γ = 1/(2s)`).
    pub gamma: f32,
    /// Target particles, flat `M×d`, uniformly weighted (the target measure
    /// ν is a frozen particle set — no density objects, per plan T1.2).
    pub target: Vec<f32>,
    /// State dimensionality.
    pub dim: usize,
}

impl MmdReward {
    /// Construct, validating that `target.len()` is a multiple of `dim`.
    pub fn new(gamma: f32, target: Vec<f32>, dim: usize) -> Self {
        assert!(
            dim > 0 && target.len().is_multiple_of(dim),
            "MmdReward target.len() ({}) must be a multiple of dim ({})",
            target.len(),
            dim
        );
        Self { gamma, target, dim }
    }

    /// Number of target particles.
    #[inline]
    pub fn target_len(&self) -> usize {
        self.target.len() / self.dim
    }

    /// `(1/M) Σ_m k(x, Y_m)` — the target embedding at `x`. When
    /// `grad_out` is non-empty (len d) it also receives
    /// `(1/M) Σ_m (x − Y_m)·k(x, Y_m)`. No allocation.
    fn target_embedding_into(&self, x: &[f32], grad_out: &mut [f32]) -> f32 {
        let d = self.dim;
        let m = self.target_len();
        let mut sum = 0.0f64;
        for i in 0..m {
            let k = rbf_kernel(x, &self.target[i * d..(i + 1) * d], self.gamma);
            sum += k as f64;
            if !grad_out.is_empty() {
                let y = &self.target[i * d..(i + 1) * d];
                for (g, (&xj, &yj)) in grad_out.iter_mut().zip(x.iter().zip(y.iter())) {
                    *g += (xj - yj) * k;
                }
            }
        }
        if !grad_out.is_empty() {
            let inv_m = 1.0 / m as f32;
            for g in grad_out.iter_mut() {
                *g *= inv_m;
            }
        }
        (sum / m as f64) as f32
    }

    /// Self term `E_{Y,Y'~ν}[k]` (M² kernel evals — cold path).
    fn target_self_similarity(&self) -> f32 {
        let d = self.dim;
        let m = self.target_len();
        let mut sum = 0.0f64;
        for i in 0..m {
            let yi = &self.target[i * d..(i + 1) * d];
            for j in 0..m {
                sum += rbf_kernel(yi, &self.target[j * d..(j + 1) * d], self.gamma) as f64;
            }
        }
        (sum / (m * m) as f64) as f32
    }
}

/// RBF kernel `k(a,b) = fast_exp(−γ·‖a−b‖²)`.
///
/// Local twin of `mag/transfer.rs::rbf_kernel` (same formula, same
/// `crate::simd::fast_exp` backend) — that helper is private to mag and a
/// cross-feature dep would cost more than these four lines.
#[inline]
pub fn rbf_kernel(a: &[f32], b: &[f32], gamma: f32) -> f32 {
    let mut dist_sq = 0.0f32;
    for j in 0..a.len() {
        let diff = a[j] - b[j];
        dist_sq += diff * diff;
    }
    crate::simd::fast_exp(-gamma * dist_sq)
}

// ──────────────────────────────────────────────────────────────────────────
// Weighted population (Plan 577 T1.3)
// ──────────────────────────────────────────────────────────────────────────

/// A weighted particle population: flat `N×d` states + unnormalized log
/// weights. Borrowed view — the caller owns the buffers (zero alloc here).
///
/// `log_weights` are **unnormalized**; use [`WeightedPopulation::weights_into`]
/// for the log-sum-exp-normalized weights (never a naive exp-sum on raw
/// logits). The weighted empirical measure `μ̂ = Σ w_i δ_{X_i}` is the
/// object the convergence theorem (paper Thm 3.4) tracks.
pub struct WeightedPopulation<'a> {
    states: &'a [f32],
    log_weights: &'a mut [f32],
    dim: usize,
}

impl<'a> WeightedPopulation<'a> {
    /// New view; `states.len()` must equal `log_weights.len() * dim`.
    pub fn new(states: &'a [f32], log_weights: &'a mut [f32], dim: usize) -> Self {
        assert!(dim > 0, "WeightedPopulation requires dim > 0 (got {dim})");
        assert!(
            states.len() == log_weights.len() * dim,
            "states.len() ({}) must equal log_weights.len() ({}) * dim ({})",
            states.len(),
            log_weights.len(),
            dim
        );
        Self {
            states,
            log_weights,
            dim,
        }
    }

    /// Particle count.
    #[inline]
    pub fn n(&self) -> usize {
        self.log_weights.len()
    }

    /// State dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Flat `N×d` states.
    #[inline]
    pub fn states(&self) -> &[f32] {
        self.states
    }

    /// Read-only log weights (len N).
    #[inline]
    pub fn log_weights_ref(&self) -> &[f32] {
        self.log_weights
    }

    /// Log-sum-exp of the current log weights (f64 accumulator; the
    /// normalizing constant of the tilted measure).
    pub fn log_sum_weights(&self) -> f32 {
        lse(self.log_weights)
    }

    /// Normalized weights into `out` (len N) via log-sum-exp.
    ///
    /// Degenerate guard: if the LSE is non-finite (all `-inf` / NaN), the
    /// output is the uniform distribution — never NaN.
    pub fn weights_into(&self, out: &mut [f32]) {
        let n = self.n();
        debug_assert_eq!(out.len(), n, "weights_into out must be len N");
        let m = lse(self.log_weights) as f64;
        if !m.is_finite() {
            out.fill(1.0 / n as f32);
            return;
        }
        let mut sum = 0.0f64;
        for &l in self.log_weights.iter() {
            sum += ((l as f64) - m).exp();
        }
        if !(sum > 0.0 && sum.is_finite()) {
            out.fill(1.0 / n as f32);
            return;
        }
        let inv = 1.0 / sum;
        for (o, &l) in out.iter_mut().zip(self.log_weights.iter()) {
            *o = (((l as f64) - m).exp() * inv) as f32;
        }
    }
}

/// Log-sum-exp with an f64 accumulator (the stable normalization primitive
/// for every weight path here — never `exp` of raw logits).
#[inline]
fn lse(xs: &[f32]) -> f32 {
    let mut m = f64::NEG_INFINITY;
    for &x in xs {
        let xd = x as f64;
        if xd > m {
            m = xd;
        }
    }
    if m == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let mut sum = 0.0f64;
    for &x in xs {
        sum += ((x as f64) - m).exp();
    }
    (m + sum.ln()) as f32
}

// ──────────────────────────────────────────────────────────────────────────
// Row implementations (Plan 577 T1.1 / T1.2 / T1.5)
// ──────────────────────────────────────────────────────────────────────────

impl MeasureReward for LinearReward {
    #[inline]
    fn kind(&self) -> RewardKind {
        RewardKind::Linear
    }
    #[inline]
    fn dim(&self) -> usize {
        self.dir.len()
    }

    fn first_variation_into(&self, x: &[f32], _pop: &WeightedPopulation, out: &mut [f32]) {
        out[0] = dot(x, &self.dir);
    }

    fn second_variation(&self, _x: &[f32], _y: &[f32], _pop: &WeightedPopulation) -> f32 {
        0.0
    }

    fn reward(&self, pop: &WeightedPopulation) -> f32 {
        // Cold path (allocates the normalized-weight vector).
        let mut w = vec![0.0f32; pop.n()];
        pop.weights_into(&mut w);
        let d = self.dim();
        let mut acc = 0.0f64;
        for (wi, xi) in w.iter().zip(pop.states().chunks_exact(d)) {
            acc += *wi as f64 * dot(xi, &self.dir) as f64;
        }
        acc as f32
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl MeasureReward for MomentReward {
    #[inline]
    fn kind(&self) -> RewardKind {
        RewardKind::Moment
    }
    #[inline]
    fn dim(&self) -> usize {
        self.phi.len()
    }

    fn first_variation_into(&self, x: &[f32], pop: &WeightedPopulation, out: &mut [f32]) {
        let m = population_moment(pop, &self.phi);
        out[0] = self.gain.derivative(m) * dot(x, &self.phi);
    }

    fn second_variation(&self, _x: &[f32], _y: &[f32], _pop: &WeightedPopulation) -> f32 {
        0.0
    }

    fn reward(&self, pop: &WeightedPopulation) -> f32 {
        let m = population_moment(pop, &self.phi);
        self.gain.value(m)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl MeasureReward for ClosureReward {
    #[inline]
    fn kind(&self) -> RewardKind {
        RewardKind::Closure
    }
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    fn first_variation_into(&self, x: &[f32], _pop: &WeightedPopulation, out: &mut [f32]) {
        out[0] = self.eval(x);
    }

    fn second_variation(&self, _x: &[f32], _y: &[f32], _pop: &WeightedPopulation) -> f32 {
        0.0
    }

    fn reward(&self, pop: &WeightedPopulation) -> f32 {
        // Cold path (allocates the normalized-weight vector).
        let mut w = vec![0.0f32; pop.n()];
        pop.weights_into(&mut w);
        let d = self.dim();
        let mut acc = 0.0f64;
        for (wi, xi) in w.iter().zip(pop.states().chunks_exact(d)) {
            acc += *wi as f64 * self.eval(xi) as f64;
        }
        acc as f32
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl MeasureReward for MmdReward {
    #[inline]
    fn kind(&self) -> RewardKind {
        RewardKind::Mmd
    }
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    fn first_variation_into(&self, x: &[f32], pop: &WeightedPopulation, out: &mut [f32]) {
        let d = pop.dim();
        let n = pop.n();
        // Weighted sum Σ w_j k(x, X_j) via the LSE trick (no scratch alloc):
        // w_j = e^{log_w_j − lse}.
        let l = lse(pop.log_weights_ref()) as f64;
        if !l.is_finite() {
            out[0] = 0.0;
            return;
        }
        let mut emb_pop = 0.0f64;
        for j in 0..n {
            let k = rbf_kernel(x, &pop.states()[j * d..(j + 1) * d], self.gamma);
            emb_pop += ((pop.log_weights_ref()[j] as f64) - l).exp() * k as f64;
        }
        let emb_nu = self.target_embedding_into(x, &mut []);
        // δ(−MMD²)/δμ(x) = 2[emb_ν(x) − emb_μ(x)] + x-independent const —
        // the tilt is HIGHER near the target mass (attracts). NOTE: Research
        // 505's Table-2 transcription ("Ψ = 2∫k(x,y)(μ−ν)(dy)") carries a
        // sign slip against its own R = −MMD²; the calculus here is pinned
        // by the finite-difference unit test.
        out[0] = (2.0 * (emb_nu as f64 - emb_pop)) as f32;
    }

    fn second_variation(&self, x: &[f32], y: &[f32], _pop: &WeightedPopulation) -> f32 {
        // δ²(−MMD²)/δμδμ = −2k(x,y).
        -2.0 * rbf_kernel(x, y, self.gamma)
    }

    fn reward(&self, pop: &WeightedPopulation) -> f32 {
        // Cold path (allocates the normalized-weight vector).
        let d = pop.dim();
        let n = pop.n();
        let mut w = vec![0.0f32; n];
        pop.weights_into(&mut w);
        let mut self_term = 0.0f64;
        for (i, xi) in pop.states().chunks_exact(d).enumerate() {
            let mut row = 0.0f64;
            for (wj, xj) in w.iter().zip(pop.states().chunks_exact(d)) {
                row += *wj as f64 * rbf_kernel(xi, xj, self.gamma) as f64;
            }
            self_term += w[i] as f64 * row;
        }
        let mut cross = 0.0f64;
        for (wi, xi) in w.iter().zip(pop.states().chunks_exact(d)) {
            cross += *wi as f64 * self.target_embedding_into(xi, &mut []) as f64;
        }
        let mmd_sq = self_term + self.target_self_similarity() as f64 - 2.0 * cross;
        -(mmd_sq.max(0.0) as f32)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for j in 0..a.len().min(b.len()) {
        s += a[j] * b[j];
    }
    s
}

/// `m = Σᵢ wᵢ (p·Xᵢ)` with LSE-normalized weights (f64 accumulation).
fn population_moment(pop: &WeightedPopulation, phi: &[f32]) -> f32 {
    let d = pop.dim();
    let l = lse(pop.log_weights_ref()) as f64;
    if !l.is_finite() {
        return 0.0;
    }
    let mut m = 0.0f64;
    for (&lw, xi) in pop
        .log_weights_ref()
        .iter()
        .zip(pop.states().chunks_exact(d))
    {
        m += ((lw as f64) - l).exp() * dot(xi, phi) as f64;
    }
    m as f32
}

// ──────────────────────────────────────────────────────────────────────────
// Row downcast accessors (closed table — no dynamic dispatch in hot loops)
// ──────────────────────────────────────────────────────────────────────────

fn linear_dir(r: &dyn MeasureReward) -> &[f32] {
    r.as_any()
        .downcast_ref::<LinearReward>()
        .map(|lr| lr.dir.as_slice())
        .expect("RewardKind::Linear must be backed by LinearReward")
}

fn moment_parts(r: &dyn MeasureReward) -> (&[f32], MomentGain) {
    let mr = r
        .as_any()
        .downcast_ref::<MomentReward>()
        .expect("RewardKind::Moment must be backed by MomentReward");
    (mr.phi.as_slice(), mr.gain)
}

fn mmd_parts(r: &dyn MeasureReward) -> (f32, &[f32]) {
    let mr = r
        .as_any()
        .downcast_ref::<MmdReward>()
        .expect("RewardKind::Mmd must be backed by MmdReward");
    (mr.gamma, mr.target.as_slice())
}

fn closure_parts(r: &dyn MeasureReward) -> &ClosureReward {
    r.as_any()
        .downcast_ref::<ClosureReward>()
        .expect("RewardKind::Closure must be backed by ClosureReward")
}

/// Central-difference `λ·∇r` for the opaque closure row (Plan 581 T1.1) —
/// shared by the [`FkStepper`] hot path and the `gradient_steering_into`
/// cold path. `2d` scorer evals per particle; dims > 64 panic (weights-only
/// steering is the consumer shape for high-dim opaque rewards). Zero-alloc
/// (stack FD buffers).
fn closure_fd_gradient_into(
    cr: &ClosureReward,
    states: &[f32],
    n: usize,
    lam: f32,
    out: &mut [f32],
) {
    let d = cr.dim();
    let eps = cr.fd_eps;
    assert!(
        d <= 64,
        "ClosureReward FD gradient supports dim <= 64 (got {d}); \
         use weights-only steering for high-dim opaque rewards"
    );
    let mut xa = [0.0f32; 64];
    let mut xb = [0.0f32; 64];
    for i in 0..n {
        let xi = &states[i * d..(i + 1) * d];
        for q in 0..d {
            xa[..d].copy_from_slice(xi);
            xb[..d].copy_from_slice(xi);
            xa[q] += eps;
            xb[q] -= eps;
            out[i * d + q] = lam * (cr.eval(&xa[..d]) - cr.eval(&xb[..d])) / (2.0 * eps);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Gradient steering, cold-path form (Plan 577 T1.4)
// ──────────────────────────────────────────────────────────────────────────

/// Per-particle first-variation gradient `∇_x Ψ(X_i, μ̂)` into `out`
/// (flat `N×d`). Analytic per row:
///
/// - `Linear`: `a` (constant).
/// - `Moment`: `F'(m)·p` (constant across particles).
/// - `Mmd`: `2∫∇_x k(x,y)(μ̂−ν)(dy)` with `∇_x k(x,y) = −2γ(x−y)k(x,y)`
///   — the same kernel matrix a reward evaluation pays, evaluated once here
///   (cold-path form with per-call temporaries; [`FkStepper`] reuses its
///   cached matrix on the hot path instead).
///
/// This is the steering **increment source**: consumers apply it through
/// their own integrator (`X += (b + λ·∇Ψ)·δt + noise`).
pub fn gradient_steering_into(
    reward: &dyn MeasureReward,
    states: &[f32],
    log_weights: &[f32],
    dim: usize,
    out: &mut [f32],
) {
    let n = log_weights.len();
    debug_assert_eq!(states.len(), n * dim);
    debug_assert_eq!(out.len(), n * dim);
    match reward.kind() {
        RewardKind::Linear => {
            let dir = linear_dir(reward);
            for chunk in out.chunks_exact_mut(dim) {
                chunk.copy_from_slice(dir);
            }
        }
        RewardKind::Moment => {
            let (phi, gain) = moment_parts(reward);
            let mut log_tmp = log_weights.to_vec();
            let pop = WeightedPopulation::new(states, &mut log_tmp, dim);
            let m = population_moment(&pop, phi);
            let scale = gain.derivative(m);
            for chunk in out.chunks_exact_mut(dim) {
                for (o, &p) in chunk.iter_mut().zip(phi.iter()) {
                    *o = scale * p;
                }
            }
        }
        RewardKind::Mmd => {
            let (gamma, target) = mmd_parts(reward);
            let m_count = target.len() / dim;
            let l = lse(log_weights) as f64;
            let weights: Vec<f64> = if l.is_finite() {
                log_weights
                    .iter()
                    .map(|&w| ((w as f64) - l).exp())
                    .collect()
            } else {
                vec![1.0 / n as f64; n]
            };
            let coef = 4.0 * gamma as f64; // ∇Ψ = 4γ·[S_pop − S_ν] (unscaled)
            let mut acc = vec![0.0f64; dim];
            for i in 0..n {
                let xi = &states[i * dim..(i + 1) * dim];
                for a in acc.iter_mut() {
                    *a = 0.0;
                }
                // S_pop = Σⱼ wⱼ(x − Xⱼ)k(x, Xⱼ).
                for j in 0..n {
                    let xj = &states[j * dim..(j + 1) * dim];
                    let k = rbf_kernel(xi, xj, gamma) as f64;
                    let c = weights[j] * k;
                    for q in 0..dim {
                        acc[q] += c * (xi[q] - xj[q]) as f64;
                    }
                }
                // S_ν = (1/M)Σₘ(x − Yₘ)k(x, Yₘ), subtracted (attraction).
                for mm in 0..m_count {
                    let ym = &target[mm * dim..(mm + 1) * dim];
                    let k = rbf_kernel(xi, ym, gamma) as f64;
                    let c = k / m_count as f64;
                    for q in 0..dim {
                        acc[q] -= c * (xi[q] - ym[q]) as f64;
                    }
                }
                for q in 0..dim {
                    out[i * dim + q] = (coef * acc[q]) as f32;
                }
            }
        }
        RewardKind::Closure => {
            // Black-box ∇Ψ by central finite differences (Plan 581 T1.1,
            // λ=1 cold path — see closure_fd_gradient_into).
            let cr = closure_parts(reward);
            closure_fd_gradient_into(cr, states, n, 1.0, out);
        }
    }
}

/// Optional steering-norm clamp (plan T2.1: "≤10% of |b|" as config):
/// clamp `‖steer_i‖ ≤ frac·‖b_i‖` per particle into `out`. `frac <= 0`
/// disables the clamp (copy-through). Consumers with near-zero drift should
/// leave it off. Zero-alloc.
pub fn clamp_steering_norm(steer: &[f32], b: &[f32], dim: usize, frac: f32, out: &mut [f32]) {
    if frac <= 0.0 {
        out.copy_from_slice(steer);
        return;
    }
    for i in 0..steer.len() / dim {
        let si = &steer[i * dim..(i + 1) * dim];
        let bi = &b[i * dim..(i + 1) * dim];
        let mut sn = 0.0f32;
        let mut bn = 0.0f32;
        for q in 0..dim {
            sn += si[q] * si[q];
            bn += bi[q] * bi[q];
        }
        let cap = frac * bn.sqrt();
        let scale = if sn > cap * cap && sn > 0.0 {
            cap / sn.sqrt()
        } else {
            1.0
        };
        for q in 0..dim {
            out[i * dim + q] = si[q] * scale;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// FK stepper + Picard Ψ̇ solver (Plan 577 T2.1 / T2.2)
// ──────────────────────────────────────────────────────────────────────────

/// Configuration for the FK stepper (Plan 577 T2.1/T2.2).
#[derive(Debug, Clone, Copy)]
pub struct FkStepper {
    /// Steering coefficient λ — scales Ψ everywhere (tilt `e^{λΨ}`).
    pub steer_scale: f32,
    /// Picard iterations per step (paper Alg 4; default 3 — K=1 slightly
    /// biased, K∈{3,5} ≈ identical).
    pub k_fp: u8,
    /// Damping α for the Picard update (`Ψ̇ ← α·target + (1−α)·Ψ̇`).
    /// Default 1.0; 0.5 for strong tilts (the paper's own setting at large λ).
    pub damping: f32,
    /// Per-step log-weight delta clamp (the paper clips at 1.0).
    pub clip_log_delta: f32,
}

impl Default for FkStepper {
    fn default() -> Self {
        Self {
            steer_scale: 1.0,
            k_fp: 3,
            damping: 1.0,
            clip_log_delta: 1.0,
        }
    }
}

/// Fixed-capacity scratch for the FK path — allocated once at construction,
/// zero allocation in the steady state. All per-step buffers live here.
pub struct SteeringScratch {
    n: usize,
    dim: usize,
    /// Kernel matrix at the current positions (MMD row only), `N×N`.
    k_mat: Vec<f32>,
    /// Target embedding `(1/M)Σ_m k(X_i, Y_m)` at the current positions (N).
    emb: Vec<f32>,
    /// Target embedding gradient `(1/M)Σ_m (X_i−Y_m)k(X_i,Y_m)` (N×d).
    emb_grad: Vec<f32>,
    /// Ψ(X_i, current positions, current weights) — λ-scaled (N). Doubles as
    /// the `psi_old` carry across steps.
    psi_cur: Vec<f32>,
    /// ∇Ψ per particle at the begin-step positions — λ-scaled (N×d).
    grad: Vec<f32>,
    /// Normalized current weights (N).
    w: Vec<f32>,
    /// b_i·∇Ψ_i per particle at begin-step positions (N).
    b_dot_grad: Vec<f32>,
    /// Picard scratch (N each): candidate log weights, candidate weights,
    /// candidate Ψ, Ψ̇ warm-start carry, Ψ̇ target.
    log_w_next: Vec<f32>,
    w_next: Vec<f32>,
    psi_cand: Vec<f32>,
    psi_dot: Vec<f32>,
    psi_dot_target: Vec<f32>,
    /// Per-particle column accumulator (dim) for the gradient pass.
    col_acc: Vec<f64>,
    /// True until the first kernel build.
    fresh: bool,
}

impl SteeringScratch {
    /// Allocate scratch for a fixed population size `n` at dimension `dim`.
    pub fn new(n: usize, dim: usize) -> Self {
        assert!(n > 0 && dim > 0, "SteeringScratch requires n, dim > 0");
        Self {
            n,
            dim,
            k_mat: vec![0.0; n * n],
            emb: vec![0.0; n],
            emb_grad: vec![0.0; n * dim],
            psi_cur: vec![0.0; n],
            grad: vec![0.0; n * dim],
            w: vec![0.0; n],
            b_dot_grad: vec![0.0; n],
            log_w_next: vec![0.0; n],
            w_next: vec![0.0; n],
            psi_cand: vec![0.0; n],
            psi_dot: vec![0.0; n],
            psi_dot_target: vec![0.0; n],
            col_acc: vec![0.0; dim],
            fresh: true,
        }
    }

    /// The steering increment per particle (flat `N×d`): `λ·∇Ψ(X_i, μ̂_w)` at
    /// the positions passed to the last [`FkStepper::begin_step`].
    pub fn steering(&self) -> &[f32] {
        &self.grad
    }

    /// Test/diagnostic accessor: max |Ψ̇| from the last Picard solve
    /// (magnitude monitoring for weight-degeneracy diagnostics).
    #[doc(hidden)]
    pub fn psi_dot_max_debug(&self) -> f32 {
        self.psi_dot.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    }

    /// Kernel + embedding cache at `states` (the ONE kernel build per step —
    /// every Picard iteration below reuses it). Symmetric fill + one
    /// vectorized exp pass (`crate::simd::simd_exp_inplace`). Zero-alloc.
    fn build_mmd_cache(&mut self, gamma: f32, target: &[f32], states: &[f32]) {
        let n = self.n;
        let d = self.dim;
        for i in 0..n {
            for j in i..n {
                let mut dist_sq = 0.0f32;
                for q in 0..d {
                    let diff = states[i * d + q] - states[j * d + q];
                    dist_sq += diff * diff;
                }
                let v = -gamma * dist_sq;
                self.k_mat[i * n + j] = v;
                self.k_mat[j * n + i] = v;
            }
        }
        crate::simd::simd_exp_inplace(&mut self.k_mat);
        let m = target.len() / d;
        if m == 0 {
            self.emb.fill(0.0);
            self.emb_grad.fill(0.0);
            return;
        }
        for i in 0..n {
            let xi = &states[i * d..(i + 1) * d];
            let mut esum = 0.0f64;
            for v in self.col_acc.iter_mut() {
                *v = 0.0;
            }
            for mm in 0..m {
                let ym = &target[mm * d..(mm + 1) * d];
                let k = rbf_kernel(xi, ym, gamma);
                esum += k as f64;
                for q in 0..d {
                    self.col_acc[q] += (xi[q] - ym[q]) as f64 * k as f64;
                }
            }
            self.emb[i] = (esum / m as f64) as f32;
            for q in 0..d {
                self.emb_grad[i * d + q] = (self.col_acc[q] / m as f64) as f32;
            }
        }
    }
}

/// `Ψ_λ(X_i, weights)` for all particles (the Picard workhorse). MMD row is
/// one matvec over the cached kernel; Linear/Moment are closed forms.
/// Free function with explicit fields so `weights`/`out` can be disjoint
/// scratch borrows — no clones, no allocs. (The 9-arg signature is the
/// deliberate split-borrow shape; bundling would re-couple the borrows.)
#[allow(clippy::too_many_arguments)]
fn eval_psi_all_into(
    reward: &dyn MeasureReward,
    states: &[f32],
    weights: &[f32],
    lam: f32,
    k_mat: &[f32],
    emb: &[f32],
    n: usize,
    d: usize,
    out: &mut [f32],
) {
    match reward.kind() {
        RewardKind::Linear => {
            let dir = linear_dir(reward);
            for (o, xi) in out.iter_mut().zip(states.chunks_exact(d)) {
                *o = lam * dot(xi, dir);
            }
        }
        RewardKind::Moment => {
            let (phi, gain) = moment_parts(reward);
            let mut m = 0.0f64;
            for (wj, xi) in weights.iter().zip(states.chunks_exact(d)) {
                m += *wj as f64 * dot(xi, phi) as f64;
            }
            let scale = lam * gain.derivative(m as f32);
            for (o, xi) in out.iter_mut().zip(states.chunks_exact(d)) {
                *o = scale * dot(xi, phi);
            }
        }
        RewardKind::Mmd => {
            for i in 0..n {
                let row = &k_mat[i * n..(i + 1) * n];
                let mut s = 0.0f64;
                for j in 0..n {
                    s += weights[j] as f64 * row[j] as f64;
                }
                // Ψ = 2[emb_ν − emb_μ] (see the MmdReward sign note).
                out[i] = (lam as f64 * 2.0 * (emb[i] as f64 - s)) as f32;
            }
        }
        RewardKind::Closure => {
            // Pointwise: Ψ(x, μ) = r(x) — no population dependence (the
            // degenerate ∫r dμ case; the Picard Ψ̇ correction is then a
            // pure position drift, exactly like the Linear row's shape).
            let cr = closure_parts(reward);
            for (o, xi) in out.iter_mut().zip(states.chunks_exact(d)) {
                *o = lam * cr.eval(xi);
            }
        }
    }
}

impl FkStepper {
    /// Phase A of one step: evaluate `∇Ψ` at the CURRENT positions under
    /// the CURRENT weights, and expose the steering increment via
    /// [`SteeringScratch::steering`]. The consumer then integrates positions
    /// themselves (`X_new = X + (b + steer)·δt + noise`, keeping the FULL
    /// drift `b = b_base + steer` at the OLD positions) and calls
    /// [`Self::finish_step`].
    ///
    /// **Contract**: `states` must be the same slice the preceding
    /// `finish_step` received as `states_new` (the kernel cache is reused —
    /// one kernel build per step total). The first ever call builds it.
    pub fn begin_step(
        &self,
        reward: &dyn MeasureReward,
        states: &[f32],
        log_weights: &mut [f32],
        scratch: &mut SteeringScratch,
    ) {
        let n = scratch.n;
        let d = scratch.dim;
        debug_assert_eq!(states.len(), n * d);
        debug_assert_eq!(log_weights.len(), n);

        if reward.kind() == RewardKind::Mmd && scratch.fresh {
            let (gamma, target) = mmd_parts(reward);
            scratch.build_mmd_cache(gamma, target, states);
        }
        scratch.fresh = false;

        {
            let pop = WeightedPopulation::new(states, log_weights, d);
            pop.weights_into(&mut scratch.w);
        }
        // ∇Ψ at current positions into scratch.grad (the steering output).
        self.gradient_into(reward, states, scratch);
    }

    /// ∇Ψ into `scratch.grad` using the cached kernel (MMD) or closed
    /// forms. Reads the normalized weights from `scratch.w` (set by the
    /// caller immediately before); split-borrows the scratch fields so no
    /// clone is needed.
    fn gradient_into(
        &self,
        reward: &dyn MeasureReward,
        states: &[f32],
        scratch: &mut SteeringScratch,
    ) {
        let d = scratch.dim;
        let lam = self.steer_scale;
        match reward.kind() {
            RewardKind::Linear => {
                let n = scratch.n;
                let dir = linear_dir(reward);
                for chunk in scratch.grad.chunks_exact_mut(d) {
                    for (g, &dv) in chunk.iter_mut().zip(dir.iter()) {
                        *g = lam * dv;
                    }
                }
                let _ = n;
            }
            RewardKind::Moment => {
                let n = scratch.n;
                let (phi, gain) = moment_parts(reward);
                let mut m = 0.0f64;
                for (wj, xi) in scratch.w.iter().zip(states.chunks_exact(d)) {
                    m += *wj as f64 * dot(xi, phi) as f64;
                }
                let scale = lam * gain.derivative(m as f32);
                for chunk in scratch.grad.chunks_exact_mut(d) {
                    for (g, &p) in chunk.iter_mut().zip(phi.iter()) {
                        *g = scale * p;
                    }
                }
                let _ = n;
            }
            RewardKind::Mmd => {
                // Split-borrow: k_mat/emb_grad/w read, grad/col_acc written.
                // ∇Ψ_λ = λ·4γ·[S_pop − S_ν] with S_pop = Σⱼ wⱼ(x−Xⱼ)k(x,Xⱼ),
                // S_ν = (1/M)Σₘ(x−Yₘ)k(x,Yₘ) — attraction toward the target.
                let n = scratch.n;
                let (gamma, _) = mmd_parts(reward);
                let coef = lam as f64 * 4.0 * gamma as f64;
                let SteeringScratch {
                    ref k_mat,
                    ref emb_grad,
                    ref w,
                    ref mut grad,
                    ref mut col_acc,
                    n: sn,
                    dim: sd,
                    ..
                } = *scratch;
                debug_assert_eq!(sn, n);
                debug_assert_eq!(sd, d);
                for i in 0..n {
                    let xi = &states[i * d..(i + 1) * d];
                    for v in col_acc.iter_mut() {
                        *v = 0.0;
                    }
                    let row = &k_mat[i * n..(i + 1) * n];
                    for j in 0..n {
                        let c = w[j] as f64 * row[j] as f64;
                        for q in 0..d {
                            col_acc[q] += c * (xi[q] - states[j * d + q]) as f64;
                        }
                    }
                    for q in 0..d {
                        grad[i * d + q] = (coef * (col_acc[q] - emb_grad[i * d + q] as f64)) as f32;
                    }
                }
            }
            RewardKind::Closure => {
                // Black-box ∇Ψ by central finite differences (Plan 581 T1.1):
                // 2d closure evals per particle — the honest cost that makes
                // weights-only steering the default consumer shape for opaque
                // rewards (see ClosureReward's gradient cost note).
                let cr = closure_parts(reward);
                closure_fd_gradient_into(cr, states, scratch.n, lam, &mut scratch.grad);
            }
        }
    }
    /// positions (paper Alg 4 — both Ψ terms at the same advanced positions,
    /// so `Ψ̇` is pure MEASURE drift), then the FK log-weight update
    /// `A_i += (b_i·∇Ψ_i + Ψ̇_i)·δt` with the per-step delta clamp.
    ///
    /// `b` is the FULL simulated drift at the BEGIN-step positions (the
    /// consumer's base dynamics + the steering increment — the `b·∇Ψ` term
    /// carries the position transport / Girsanov overshoot correction). On
    /// return, `log_weights` carries the updated (still unnormalized) FK
    /// log-weights, and the kernel cache sits at `states_new` for the next
    /// `begin_step`.
    pub fn finish_step(
        &self,
        reward: &dyn MeasureReward,
        states_new: &[f32],
        b: &[f32],
        dt: f32,
        log_weights: &mut [f32],
        scratch: &mut SteeringScratch,
    ) {
        let n = scratch.n;
        let d = scratch.dim;
        debug_assert_eq!(states_new.len(), n * d);
        debug_assert_eq!(b.len(), n * d);
        debug_assert_eq!(log_weights.len(), n);
        let lam = self.steer_scale;

        // Rebuild the kernel cache at the advanced positions (the single
        // kernel build per step — reused by every Picard iteration below).
        if reward.kind() == RewardKind::Mmd {
            let (gamma, target) = mmd_parts(reward);
            scratch.build_mmd_cache(gamma, target, states_new);
        }

        // psi_old = Ψ_λ(X_new, μ_old) — the potential at the ADVANCED
        // positions under the OLD measure (paper Alg 4: both Ψ terms share
        // the same x, so Ψ̇ is pure measure drift; the position transport is
        // carried by the b·∇Ψ term with b = the FULL simulated drift —
        // the Girsanov overshoot correction that keeps the weighted law at
        // μ* instead of the transported law).
        {
            let pop = WeightedPopulation::new(states_new, log_weights, d);
            pop.weights_into(&mut scratch.w);
        }
        eval_psi_all_into(
            reward,
            states_new,
            &scratch.w,
            lam,
            &scratch.k_mat,
            &scratch.emb,
            n,
            d,
            &mut scratch.psi_cur,
        );

        // b·∇Ψ at the begin-step positions (∇Ψ cached by begin_step).
        for i in 0..n {
            let mut s = 0.0f64;
            for q in 0..d {
                s += b[i * d + q] as f64 * scratch.grad[i * d + q] as f64;
            }
            scratch.b_dot_grad[i] = s as f32;
        }

        // Damped Picard (Alg 4): candidate next-weights → candidate
        // next-measure → Ψ̇ finite difference → damped update. Warm-started
        // from the previous step's Ψ̇ (scratch.psi_dot).
        let clip = self.clip_log_delta;
        let k_fp = self.k_fp.max(1) as usize;
        let inv_dt = 1.0f64 / dt as f64;
        for _ in 0..k_fp {
            for ((lwn, lw), (bdg, &pd)) in scratch
                .log_w_next
                .iter_mut()
                .zip(log_weights.iter())
                .zip(scratch.b_dot_grad.iter().zip(scratch.psi_dot.iter()))
            {
                let g = bdg + pd;
                let delta = (dt * g).clamp(-clip, clip);
                *lwn = lw + delta;
            }
            // LSE-normalize the candidate weights (multi-array index loop —
            // deliberately range-based for fixed iteration order).
            let mut mx = f64::NEG_INFINITY;
            for i in 0..n {
                let v = scratch.log_w_next[i] as f64;
                if v > mx {
                    mx = v;
                }
            }
            let mut sum = 0.0f64;
            for i in 0..n {
                sum += (scratch.log_w_next[i] as f64 - mx).exp();
            }
            let inv = 1.0 / sum;
            for i in 0..n {
                scratch.w_next[i] = (((scratch.log_w_next[i] as f64) - mx).exp() * inv) as f32;
            }
            // Ψ at the candidate next-measure (advanced positions, cached
            // kernel — the reuse that makes Picard cheap).
            eval_psi_all_into(
                reward,
                states_new,
                &scratch.w_next,
                lam,
                &scratch.k_mat,
                &scratch.emb,
                n,
                d,
                &mut scratch.psi_cand,
            );
            // Finite-difference target + damping.
            for i in 0..n {
                let t = (scratch.psi_cand[i] as f64 - scratch.psi_cur[i] as f64) * inv_dt;
                scratch.psi_dot_target[i] = t as f32;
            }
            let a = self.damping as f64;
            for i in 0..n {
                let next =
                    a * scratch.psi_dot_target[i] as f64 + (1.0 - a) * scratch.psi_dot[i] as f64;
                scratch.psi_dot[i] = next as f32;
            }
        }

        // Final FK update with the converged Ψ̇ (multi-array loop).
        for (lw, (bdg, &pd)) in log_weights
            .iter_mut()
            .zip(scratch.b_dot_grad.iter().zip(scratch.psi_dot.iter()))
        {
            let g = bdg + pd;
            let delta = (dt * g).clamp(-clip, clip);
            *lw += delta;
        }

        // Carry Ψ(X_new, μ_new) for the next step's psi_old.
        {
            let pop = WeightedPopulation::new(states_new, log_weights, d);
            pop.weights_into(&mut scratch.w);
        }
        eval_psi_all_into(
            reward,
            states_new,
            &scratch.w,
            lam,
            &scratch.k_mat,
            &scratch.emb,
            n,
            d,
            &mut scratch.psi_cur,
        );
    }

    /// Self-consistency residual (plan T2.4): the Picard fixed-point gap at
    /// convergence — the L1 weight change one more Picard update would make
    /// from `(states, log_weights)` under drift `b`. Near `0` ⇒ the tilt
    /// fixed point holds (`μ̂ ≈ (1/Z)e^{Ψ(·,μ̂)}p`-form); a cheap convergence
    /// certificate for consumers.
    pub fn tilt_residual(
        &self,
        reward: &dyn MeasureReward,
        states: &[f32],
        b: &[f32],
        dt: f32,
        log_weights: &mut [f32],
        scratch: &mut SteeringScratch,
    ) -> f32 {
        let n = scratch.n;
        let d = scratch.dim;
        if reward.kind() == RewardKind::Mmd {
            let (gamma, target) = mmd_parts(reward);
            scratch.build_mmd_cache(gamma, target, states);
        }
        {
            let pop = WeightedPopulation::new(states, log_weights, d);
            pop.weights_into(&mut scratch.w);
        }
        eval_psi_all_into(
            reward,
            states,
            &scratch.w,
            self.steer_scale,
            &scratch.k_mat,
            &scratch.emb,
            n,
            d,
            &mut scratch.psi_cur,
        );
        self.gradient_into(reward, states, scratch);
        for i in 0..n {
            let mut s = 0.0f64;
            for q in 0..d {
                s += b[i * d + q] as f64 * scratch.grad[i * d + q] as f64;
            }
            scratch.b_dot_grad[i] = s as f32;
        }
        // One Picard update from the carried Ψ̇ warm start.
        for ((lwn, lw), (bdg, &pd)) in scratch
            .log_w_next
            .iter_mut()
            .zip(log_weights.iter())
            .zip(scratch.b_dot_grad.iter().zip(scratch.psi_dot.iter()))
        {
            let g = bdg + pd;
            let delta = (dt * g).clamp(-self.clip_log_delta, self.clip_log_delta);
            *lwn = lw + delta;
        }
        let mut mx = f64::NEG_INFINITY;
        for i in 0..n {
            let v = scratch.log_w_next[i] as f64;
            if v > mx {
                mx = v;
            }
        }
        let mut sum = 0.0f64;
        for i in 0..n {
            sum += (scratch.log_w_next[i] as f64 - mx).exp();
        }
        let inv = 1.0 / sum;
        let mut l1 = 0.0f64;
        for i in 0..n {
            let w_next = (((scratch.log_w_next[i] as f64) - mx).exp() * inv) as f32;
            l1 += (w_next - scratch.w[i]).abs() as f64;
        }
        l1 as f32
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Resampling (Plan 577 T2.3) — sampling consumers ONLY
// ──────────────────────────────────────────────────────────────────────────

/// Residual resampling into caller-provided ancestor indices
/// (`weights.len() == out.len() == n`).
///
/// **NOT for persistent-agent use** (Research 505 caveat 2 / the paper's own
/// §5 diversity limitation): resampling DUPLICATES particles, which is
/// meaningless when particles are persistent NPCs. The convergence theorem
/// tracks the weighted measure `μ̂ = Σ wᵢ δ_{Xᵢ}` — weights-only mode is the
/// correct consumer shape for agents; this routine exists for sampling
/// consumers (generate-then-resample pipelines) only. `u ∈ [0,1)` is the
/// caller's uniform draw (systematic within residuals). Allocates (cold,
/// sampling-cadence path).
pub fn residual_resample_into(weights: &[f32], n: usize, u: f32, out: &mut [u32]) {
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(weights.len(), n);
    debug_assert!((0.0..1.0).contains(&u));
    let mut idx: Vec<usize> = Vec::with_capacity(n);
    let mut residuals: Vec<(usize, f32)> = Vec::with_capacity(n);
    let mut count = 0usize;
    for (i, &w) in weights.iter().enumerate() {
        let ni = n as f64 * w as f64;
        let c = ni.floor() as usize;
        for _ in 0..c {
            idx.push(i);
        }
        count += c;
        residuals.push((i, ni.fract() as f32));
    }
    let remaining = n.saturating_sub(count);
    if remaining > 0 {
        residuals.sort_by(|a, b| b.1.total_cmp(&a.1));
        let total: f32 = residuals.iter().map(|r| r.1).sum();
        if total > 0.0 {
            let step = total / remaining as f32;
            let mut cursor = u * step;
            let mut ri = 0usize;
            let mut acc = 0.0f32;
            for r in &residuals {
                acc += r.1;
                while ri < remaining && cursor <= acc {
                    idx.push(r.0);
                    ri += 1;
                    cursor += step;
                }
            }
            let mut fill = 0usize;
            while ri < remaining && fill < residuals.len() {
                idx.push(residuals[fill].0);
                ri += 1;
                fill += 1;
            }
        } else {
            for k in 0..remaining {
                idx.push(k % n);
            }
        }
    }
    idx.truncate(n);
    for (slot, a) in out.iter_mut().take(idx.len()).enumerate() {
        *a = idx[slot] as u32;
    }
    for slot in out.iter_mut().skip(idx.len()) {
        *slot = 0;
    }
}

/// Systematic resampling (the deterministic alternative): one uniform draw
/// `u ∈ [0,1)`, ancestors at `(i + u)/n` through the CDF. Zero-alloc.
pub fn systematic_resample_into(weights: &[f32], n: usize, u: f32, out: &mut [u32]) {
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(weights.len(), n);
    let mut cdf = 0.0f64;
    let mut j = 0usize;
    for (i, o) in out.iter_mut().enumerate() {
        let target = (i as f64 + u as f64) / n as f64;
        while j < n && cdf <= target {
            cdf += weights[j] as f64;
            j += 1;
        }
        *o = j.saturating_sub(1) as u32;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// BoM adapter (Plan 577 T4.1) — opt-in composition
// ──────────────────────────────────────────────────────────────────────────

/// FK-weighted hypothesis selection for
/// [`crate::micro_belief::BoMSampler`](crate::micro_belief) consumers:
/// instead of the trait's argmax `select_best`, weight the K hypotheses
/// against a measure reward by solving the STATIC tilt fixed point
/// `w ∝ e^{λΨ(X,·,w)}` (the same Picard loop as the stepper, without the
/// time dimension — hypotheses are a one-shot population).
///
/// This is the theoretical grounding the BoM diversity heuristic lacked
/// (Research 505 fusion F3): the characterized target is the tilted measure
/// over hypotheses, not a scorer argmax.
///
/// **No UQ claim** (Research 505 caveat 5): this is a control mechanism; if
/// any future gate claims calibrated uncertainty from it, the "Report the
/// Floor" rule attaches there.
#[cfg(all(feature = "bom_sampling", feature = "distributional_steering"))]
pub mod bom {
    use super::{MeasureReward, WeightedPopulation, lse};

    /// Solve the static tilt fixed point over K hypotheses:
    /// `log w ← α·λΨ(X_i, w) + (1−α)·log w`, `k_fp` iterations from uniform.
    /// `hypotheses` is flat `K×d`; `out` receives normalized weights (K).
    ///
    /// Cold path (small K, per-iteration temporaries) — the population is a
    /// one-shot hypothesis batch, not a hot tick.
    pub fn hypothesis_weights_into(
        reward: &dyn MeasureReward,
        hypotheses: &[f32],
        dim: usize,
        lam: f32,
        k_fp: u8,
        damping: f32,
        out: &mut [f32],
    ) {
        let k = hypotheses.len() / dim;
        debug_assert_eq!(out.len(), k);
        let mut log_w = vec![0.0f32; k];
        let mut psi = vec![0.0f32; k];
        for _ in 0..k_fp.max(1) {
            for i in 0..k {
                let x = &hypotheses[i * dim..(i + 1) * dim];
                let pop = WeightedPopulation::new(hypotheses, &mut log_w, dim);
                let mut tmp = [0.0f32; 1];
                reward.first_variation_into(x, &pop, &mut tmp);
                psi[i] = lam * tmp[0];
            }
            let a = damping;
            for i in 0..k {
                log_w[i] = a * psi[i] + (1.0 - a) * log_w[i];
            }
        }
        let l = lse(&log_w) as f64;
        if !l.is_finite() {
            out.fill(1.0 / k as f32);
            return;
        }
        let mut sum = 0.0f64;
        for &lw in log_w.iter() {
            sum += ((lw as f64) - l).exp();
        }
        let inv = 1.0 / sum;
        for (o, &lw) in out.iter_mut().zip(log_w.iter()) {
            *o = (((lw as f64) - l).exp() * inv) as f32;
        }
    }

    /// Weighted alternative to `BoMSampler::select_best`: the argmax-weight
    /// hypothesis after the tilt fixed point (ties → lowest index, matching
    /// the trait's contract).
    pub fn select_best_fk(
        reward: &dyn MeasureReward,
        hypotheses: &[f32],
        dim: usize,
        lam: f32,
        k_fp: u8,
    ) -> usize {
        let k = hypotheses.len() / dim;
        let mut w = vec![0.0f32; k];
        hypothesis_weights_into(reward, hypotheses, dim, lam, k_fp, 1.0, &mut w);
        let mut best = 0usize;
        let mut best_w = f32::NEG_INFINITY;
        for (i, &wi) in w.iter().enumerate() {
            if wi > best_w {
                best_w = wi;
                best = i;
            }
        }
        best
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests (Plan 577 T1.5 / T2.5)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64 — the house deterministic test RNG.
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_normal(&mut self) -> f32 {
            let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
            let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        }
        #[allow(dead_code)] // kept for harness parity with the bench-682 RNG
        fn next_uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    // ── T1.3: weighted population ─────────────────────────────────────────

    #[test]
    fn weights_lse_normalizes_to_one() {
        let log_w = [0.5f32, 1.5, 2.5, -3.0];
        let states = [0.0f32; 4];
        let mut lw = log_w;
        let pop = WeightedPopulation::new(&states, &mut lw, 1);
        let mut w = [0.0f32; 4];
        pop.weights_into(&mut w);
        let s: f32 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "sum {s}");
        assert!(w[2] > w[0], "ordering preserved");
    }

    #[test]
    fn weights_lse_stable_under_large_offsets() {
        // +800 overflows naive exp; LSE must not.
        let states = [0.0f32; 2];
        let mut lw = [800.0f32, 800.0];
        let pop = WeightedPopulation::new(&states, &mut lw, 1);
        let mut w = [0.0f32; 2];
        pop.weights_into(&mut w);
        assert!((w[0] - 0.5).abs() < 1e-6 && (w[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn weights_uniform_on_degenerate_log_weights() {
        let states = [0.0f32; 3];
        let mut lw = [f32::NEG_INFINITY; 3];
        let pop = WeightedPopulation::new(&states, &mut lw, 1);
        let mut w = [0.0f32; 3];
        pop.weights_into(&mut w);
        assert!((w[0] - 1.0 / 3.0).abs() < 1e-6);
    }

    // ── T1.5: variation rows vs numerical functional differentiation ──────

    /// Numerical first variation: `(R((1−ε)μ + εδ_x) − R(μ)) / ε`, evaluated
    /// on an augmented population {X_i with weights (1−ε)w_i} ∪ {x, weight ε}.
    fn numerical_first_variation(
        reward: &dyn MeasureReward,
        states: &[f32],
        log_w: &[f32],
        dim: usize,
        x: &[f32],
        eps: f32,
    ) -> f32 {
        let n = log_w.len();
        let mut all_states = vec![0.0f32; (n + 1) * dim];
        all_states[..n * dim].copy_from_slice(states);
        all_states[n * dim..].copy_from_slice(x);
        let mut lw = vec![0.0f32; n + 1];
        let l = lse(log_w) as f64;
        for i in 0..n {
            lw[i] = (((1.0 - eps as f64) * ((log_w[i] as f64) - l).exp()).ln()) as f32;
        }
        lw[n] = eps.ln();
        let pop = WeightedPopulation::new(&all_states, &mut lw, dim);
        reward.reward(&pop)
    }

    #[test]
    fn mmd_first_variation_matches_finite_difference() {
        let mut rng = SplitMix64::new(42);
        let (n, dim) = (24usize, 2usize);
        let mut states = vec![0.0f32; n * dim];
        for s in states.iter_mut() {
            *s = rng.next_normal();
        }
        let target: Vec<f32> = (0..16 * dim).map(|_| rng.next_normal()).collect();
        let reward = MmdReward::new(0.3, target, dim);
        let mut log_w = vec![0.0f32; n];
        for (i, l) in log_w.iter_mut().enumerate() {
            *l = 0.1 * i as f32;
        }
        // `base` is unused in the probe-difference form (kept out).

        for probe in 0..4 {
            // The mass-preserving finite difference is δR/δμ(x) +
            // x-independent const; the module's Ψ is the x-dependent kernel.
            // Validating the x-DEPENDENCE via probe differences cancels the
            // constant (and is what the tilt actually uses).
            let x_a: Vec<f32> = (0..dim).map(|_| rng.next_normal()).collect();
            let x_b: Vec<f32> = (0..dim).map(|_| rng.next_normal()).collect();
            let psi_diff = {
                let pop = WeightedPopulation::new(&states, &mut log_w, dim);
                let mut oa = [0.0f32; 1];
                let mut ob = [0.0f32; 1];
                reward.first_variation_into(&x_a, &pop, &mut oa);
                reward.first_variation_into(&x_b, &pop, &mut ob);
                oa[0] - ob[0]
            };
            let eps = 1e-3f32;
            let ra = numerical_first_variation(&reward, &states, &log_w, dim, &x_a, eps);
            let rb = numerical_first_variation(&reward, &states, &log_w, dim, &x_b, eps);
            let num_diff = (ra - rb) / eps;
            assert!(
                (psi_diff - num_diff).abs() < 5e-2,
                "probe {probe}: ΔΨ analytic {psi_diff} vs finite-diff {num_diff}"
            );
        }
    }

    #[test]
    fn moment_first_variation_matches_finite_difference() {
        let mut rng = SplitMix64::new(7);
        let (n, dim) = (16usize, 2usize);
        let mut states = vec![0.0f32; n * dim];
        for s in states.iter_mut() {
            *s = rng.next_normal();
        }
        let phi = vec![1.0f32, -0.5];
        let reward = MomentReward {
            gain: MomentGain::NegativeSquare,
            phi,
        };
        let mut log_w = vec![0.0f32; n];
        for (i, l) in log_w.iter_mut().enumerate() {
            *l = 0.05 * i as f32;
        }
        let base = {
            let pop = WeightedPopulation::new(&states, &mut log_w, dim);
            reward.reward(&pop)
        };
        let _ = base; // (probe-difference form cancels it; kept for symmetry)
        let x_a = [0.7f32, -1.2];
        let x_b = [-0.4f32, 0.9];
        let psi_diff = {
            let pop = WeightedPopulation::new(&states, &mut log_w, dim);
            let mut oa = [0.0f32; 1];
            let mut ob = [0.0f32; 1];
            reward.first_variation_into(&x_a, &pop, &mut oa);
            reward.first_variation_into(&x_b, &pop, &mut ob);
            oa[0] - ob[0]
        };
        let eps = 1e-3f32;
        let ra = numerical_first_variation(&reward, &states, &log_w, dim, &x_a, eps);
        let rb = numerical_first_variation(&reward, &states, &log_w, dim, &x_b, eps);
        let num_diff = (ra - rb) / eps;
        assert!(
            (psi_diff - num_diff).abs() < 5e-2,
            "moment ΔΨ analytic {psi_diff} vs finite-diff {num_diff} (base {base})"
        );
    }

    #[test]
    fn linear_row_is_pointwise_and_gradient_is_dir() {
        let dim = 3;
        let dir = vec![0.5f32, -1.0, 2.0];
        let reward = LinearReward { dir: dir.clone() };
        let n = 5;
        let states = vec![0.25f32; n * dim];
        let mut lw = vec![0.0f32; n];
        {
            let pop = WeightedPopulation::new(&states, &mut lw, dim);
            let x = [1.0f32, 2.0, 3.0];
            let mut out = [0.0f32; 1];
            reward.first_variation_into(&x, &pop, &mut out);
            assert_eq!(out[0], dir[0] * 1.0 + dir[1] * 2.0 + dir[2] * 3.0);
        }
        let mut grad = vec![0.0f32; n * dim];
        gradient_steering_into(&reward, &states, &lw, dim, &mut grad);
        for i in 0..n {
            assert_eq!(&grad[i * dim..(i + 1) * dim], &dir[..]);
        }
    }

    // ── Plan 581 T1.3 — ClosureReward row ────────────────────────────

    #[test]
    fn closure_row_matches_linear_row_on_affine_r() {
        // T1.3: on affine r = a·x + c the closure row agrees with LinearReward
        // UP TO THE CONSTANT — the offset shifts every Ψ equally, so the
        // LSE-normalized tilt (and resampling) is bit-identical (the exact
        // "degenerate ∫r dμ case" of R505 Prop 3.1).
        let dim = 3usize;
        let dir = [0.5f32, -1.0, 2.0];
        let offset = 0.75f32;
        let closure = ClosureReward::new(dim, 1e-3, move |x: &[f32]| {
            dir.iter().zip(x).map(|(&a, &v)| a * v).sum::<f32>() + offset
        });
        let linear = LinearReward { dir: dir.to_vec() };
        let n = 4usize;
        let states = vec![
            0.25f32, -0.5, 1.5, 2.0, 0.0, -1.0, 0.5, 0.75, 1.25, -2.0, 0.1, 0.3,
        ];
        let mut lw = vec![0.0f32; n];
        lw[2] = 0.7; // non-uniform weights exercise the LSE path
        let mut lw2 = lw.clone();
        let pop_c = WeightedPopulation::new(&states, &mut lw, dim);
        let pop_l = WeightedPopulation::new(&states, &mut lw2, dim);
        let x = [1.0f32, 2.0, 3.0];
        let mut out_c = [0.0f32; 1];
        let mut out_l = [0.0f32; 1];
        closure.first_variation_into(&x, &pop_c, &mut out_c);
        linear.first_variation_into(&x, &pop_l, &mut out_l);
        let expect_lin = dir[0] * 1.0 + dir[1] * 2.0 + dir[2] * 3.0;
        assert!((out_l[0] - expect_lin).abs() < 1e-5);
        assert!((out_c[0] - (expect_lin + offset)).abs() < 1e-5);
        assert_eq!(closure.second_variation(&x, &x, &pop_c), 0.0);
        // R(μ) likewise splits by exactly the constant.
        let r_c = closure.reward(&pop_c);
        let r_l = linear.reward(&pop_l);
        assert!((r_c - (r_l + offset)).abs() < 1e-4);
    }

    #[test]
    fn closure_gradient_fd_matches_closed_form_on_quadratic() {
        // r(x) = −(x·x) has closed-form gradient ∇r = −2x; the FD arm in
        // FkStepper::gradient_into must land within FD truncation error.
        let dim = 2usize;
        let reward =
            ClosureReward::new(dim, 1e-2, |x: &[f32]| -x.iter().map(|v| v * v).sum::<f32>());
        let n = 3usize;
        let states = vec![0.5f32, -1.25, 2.0, 0.75, -0.25, 1.5];
        let lw = vec![0.0f32; n];
        let mut grad = vec![0.0f32; n * dim];
        gradient_steering_into(&reward, &states, &lw, dim, &mut grad);
        // Explicit λ=1 numeric check against the closed form (∇r = −2x;
        // gradient_steering_into is the λ=1 cold path).
        let lam = 1.0f32;
        for (i, xi) in states.chunks_exact(dim).enumerate() {
            for (q, &v) in xi.iter().enumerate() {
                assert!(grad[i * dim + q].is_finite());
                assert!((grad[i * dim + q] - lam * -2.0 * v).abs() < 5e-2);
            }
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "must be finite")]
    fn closure_nan_reward_rejected_at_boundary() {
        // T1.3: non-finite r is a caller bug — the debug_assert at the
        // boundary rejects it (house is_finite discipline).
        let reward = ClosureReward::new(1, 1e-3, |_x: &[f32]| f32::NAN);
        let states = [1.0f32];
        let mut lw = [0.0f32];
        let pop = WeightedPopulation::new(&states, &mut lw, 1);
        let mut out = [0.0f32; 1];
        reward.first_variation_into(&states, &pop, &mut out);
    }

    #[test]
    fn mmd_second_variation_is_two_kernel() {
        let target = vec![0.0f32, 0.0];
        let reward = MmdReward::new(0.5, target, 2);
        let states = [0.0f32; 2];
        let mut lw = [0.0f32; 2];
        let sv = {
            let pop = WeightedPopulation::new(&states, &mut lw, 1);
            reward.second_variation(&[1.0f32, 0.0], &[0.0f32, 1.0], &pop)
        };
        let k = rbf_kernel(&[1.0f32, 0.0], &[0.0f32, 1.0], 0.5);
        assert_eq!(sv, -2.0 * k);
    }

    #[test]
    fn stepper_mmd_gradient_matches_cold_path() {
        // Hot-path (cached-kernel) gradient == cold-path free function.
        let mut rng = SplitMix64::new(3);
        let (n, dim) = (20usize, 2usize);
        let mut states = vec![0.0f32; n * dim];
        for s in states.iter_mut() {
            *s = rng.next_normal();
        }
        let target: Vec<f32> = (0..8 * dim).map(|_| rng.next_normal()).collect();
        let reward = MmdReward::new(0.4, target, dim);
        let lam = 2.5f32;
        let stepper = FkStepper {
            steer_scale: lam,
            ..Default::default()
        };
        let mut scratch = SteeringScratch::new(n, dim);
        let mut lw = vec![0.0f32; n];
        stepper.begin_step(&reward, &states, &mut lw, &mut scratch);
        let hot = scratch.steering().to_vec();
        let mut cold = vec![0.0f32; n * dim];
        gradient_steering_into(&reward, &states, &lw, dim, &mut cold);
        for i in 0..n * dim {
            assert!(
                (hot[i] - lam * cold[i]).abs() < 1e-4,
                "grad mismatch at {i}: hot {} vs λ·cold {}",
                hot[i],
                lam * cold[i]
            );
        }
    }

    // ── T2.5: Picard Ψ̇ vs the implicit linear system (Alg 3) ─────────────

    /// Dense solve of the Alg-3-form implicit linear system
    /// `(I − MW + (Mw)wᵀ)Ψ̇ = (MW − (Mw)wᵀ)c` with `M_ij = 2λk_ij`,
    /// `c_i = b_i·∇Ψ_λ,i` (test-side reference implementation).
    fn solve_linear_psi_dot(
        gamma: f32,
        lam: f32,
        states: &[f32],
        log_w: &[f32],
        b: &[f32],
        grad: &[f32],
    ) -> Vec<f32> {
        let n = log_w.len();
        let dim = 1usize;
        let l = lse(log_w) as f64;
        let w: Vec<f64> = log_w.iter().map(|&x| ((x as f64) - l).exp()).collect();
        let mut m_mat = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let k = rbf_kernel(
                    &states[i * dim..(i + 1) * dim],
                    &states[j * dim..(j + 1) * dim],
                    gamma,
                ) as f64;
                // M = ∂Ψ_λ/∂w (uncentered): Ψ = −2λ[(Kw)−emb] → −2λk.
                m_mat[i * n + j] = -2.0 * lam as f64 * k;
            }
        }
        let c: Vec<f64> = (0..n).map(|i| b[i] as f64 * grad[i] as f64).collect();
        let mw: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| m_mat[i * n + j] * w[j]).sum())
            .collect();
        let wc: f64 = (0..n).map(|j| w[j] * c[j]).sum();
        let mut a = vec![0.0f64; n * n];
        let mut rhs = vec![0.0f64; n];
        for i in 0..n {
            let mut mwc = 0.0f64;
            for j in 0..n {
                let mw_ij = m_mat[i * n + j] * w[j];
                a[i * n + j] = if i == j { 1.0 } else { 0.0 } - mw_ij + mw[i] * w[j];
                mwc += mw_ij * c[j];
            }
            rhs[i] = mwc - mw[i] * wc;
        }
        // Gaussian elimination with partial pivoting (flat row swap).
        let mut aa = a;
        let mut bb = rhs;
        for col in 0..n {
            let mut piv = col;
            for r in (col + 1)..n {
                if aa[r * n + col].abs() > aa[piv * n + col].abs() {
                    piv = r;
                }
            }
            if piv != col {
                for c_i in 0..n {
                    aa.swap(col * n + c_i, piv * n + c_i);
                }
                bb.swap(col, piv);
            }
            let d = aa[col * n + col];
            if d.abs() < 1e-12 {
                continue;
            }
            for r in (col + 1)..n {
                let f = aa[r * n + col] / d;
                for c_i in col..n {
                    aa[r * n + c_i] -= f * aa[col * n + c_i];
                }
                bb[r] -= f * bb[col];
            }
        }
        let mut x = vec![0.0f64; n];
        for r in (0..n).rev() {
            let mut s = bb[r];
            for c_i in (r + 1)..n {
                s -= aa[r * n + c_i] * x[c_i];
            }
            let d = aa[r * n + r];
            x[r] = if d.abs() < 1e-12 { 0.0 } else { s / d };
        }
        x.iter().map(|&v| v as f32).collect()
    }

    #[test]
    fn psi_dot_picard_matches_implicit_linear_system() {
        // Small N, MMD reward, fixed drift, positions NOT moved (pure weight
        // dynamics at δt→0): Picard (large K_FP) vs the Alg-3 dense solve.
        let mut rng = SplitMix64::new(1234);
        let n = 12usize;
        let dim = 1usize;
        let mut states = vec![0.0f32; n * dim];
        for s in states.iter_mut() {
            *s = rng.next_normal();
        }
        let target: Vec<f32> = (0..8).map(|_| rng.next_normal()).collect();
        let reward = MmdReward::new(0.2, target, dim);
        let mut log_w = vec![0.0f32; n];
        for (i, l) in log_w.iter_mut().enumerate() {
            *l = 0.08 * i as f32;
        }
        let lam = 1.5f32;
        let dt = 1e-3f32;
        let b: Vec<f32> = (0..n).map(|_| 0.3 * rng.next_normal()).collect();

        // Reference: λ-scaled grad feeds c = b·∇Ψ_λ.
        let mut grad = vec![0.0f32; n * dim];
        gradient_steering_into(&reward, &states, &log_w, dim, &mut grad);
        for g in grad.iter_mut() {
            *g *= lam;
        }
        let lin = solve_linear_psi_dot(0.2, lam, &states, &log_w, &b, &grad);

        let stepper = FkStepper {
            steer_scale: lam,
            k_fp: 200,
            damping: 1.0,
            clip_log_delta: 10.0,
        };
        let mut scratch = SteeringScratch::new(n, dim);
        let mut lw = log_w.clone();
        stepper.begin_step(&reward, &states, &mut lw, &mut scratch);
        stepper.finish_step(&reward, &states, &b, dt, &mut lw, &mut scratch);
        let picard = scratch.psi_dot.clone();

        let scale = lin.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
        let max_diff = lin
            .iter()
            .zip(picard.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff / scale < 5e-2,
            "Picard Ψ̇ vs Alg-3 linear: max_diff {max_diff}, scale {scale}"
        );
    }

    #[test]
    fn weights_sum_to_one_across_k_fp_settings() {
        let mut rng = SplitMix64::new(99);
        let (n, dim) = (32usize, 1usize);
        let mut states = vec![0.0f32; n * dim];
        for s in states.iter_mut() {
            *s = rng.next_normal();
        }
        let target: Vec<f32> = (0..8).map(|_| rng.next_normal()).collect();
        let reward = MmdReward::new(0.25, target, dim);
        let b = vec![0.1f32; n * dim];
        for &k_fp in &[1u8, 3, 5, 10] {
            let stepper = FkStepper {
                steer_scale: 3.0,
                k_fp,
                ..Default::default()
            };
            let mut scratch = SteeringScratch::new(n, dim);
            let mut lw = vec![0.0f32; n];
            for _ in 0..10 {
                stepper.begin_step(&reward, &states, &mut lw, &mut scratch);
                stepper.finish_step(&reward, &states, &b, 0.05, &mut lw, &mut scratch);
            }
            let mut tmp = lw;
            let pop = WeightedPopulation::new(&states, &mut tmp, dim);
            let mut w = vec![0.0f32; n];
            pop.weights_into(&mut w);
            let s: f32 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "k_fp={k_fp}: sum {s}");
            assert!(w.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn k_fp1_vs_kfp3_bias_documented() {
        // The paper's K_FP=1-bias observation, verified in BOTH start
        // regimes on a strong tilt (λ=30, damped 0.3) — PROVIDED the
        // consumer drift is nonzero. With b = 0 the candidate weights equal
        // the old weights, Ψ̇ ≡ 0 identically (no weight change → no measure
        // drift — correct math), and every K_FP is vacuous; the earlier
        // "warm-started K_FP=1 ≡ K_FP=3" observation was exactly that
        // artifact.
        let mut rng = SplitMix64::new(5);
        let (n, dim) = (64usize, 1usize);
        let mut states = vec![0.0f32; n * dim];
        for (i, s) in states.iter_mut().enumerate() {
            *s = if i % 4 == 0 { -1.0 } else { 1.0 } + 0.1 * rng.next_normal();
        }
        let target: Vec<f32> = (0..32).map(|_| -1.0 + 0.2 * rng.next_normal()).collect();
        let reward = MmdReward::new(0.5, target, dim);
        // Nonzero consumer drift: with b = 0 the candidate weights equal the
        // old weights, Ψ̇ ≡ 0 identically (correct math — no weight change, no
        // measure drift), and K_FP is vacuous. The K_FP bias needs a driver.
        let b = vec![0.05f32; n * dim];
        let dt = 0.02f32;

        let run = |k_fp: u8, warm: bool| -> f32 {
            let stepper = FkStepper {
                steer_scale: 30.0,
                k_fp,
                damping: 0.3,
                clip_log_delta: 0.5,
            };
            let mut scratch = SteeringScratch::new(n, dim);
            let mut lw = vec![0.0f32; n];
            let mut st = states.clone();
            for _ in 0..20 {
                if !warm {
                    scratch = SteeringScratch::new(n, dim); // cold: Ψ̇ = 0
                }
                stepper.begin_step(&reward, &st, &mut lw, &mut scratch);
                let steer = scratch.steering().to_vec();
                for (si, s) in st.iter_mut().enumerate() {
                    *s += 0.1 * steer[si] * dt;
                }
                stepper.finish_step(&reward, &st, &b, dt, &mut lw, &mut scratch);
            }
            let mut tmp = lw;
            let pop = WeightedPopulation::new(&st, &mut tmp, dim);
            let mut w = vec![0.0f32; n];
            pop.weights_into(&mut w);
            w.iter().zip(st.iter()).map(|(wi, xi)| wi * xi).sum()
        };

        // Warm start (the production default): with a nonzero drift driver
        // the K_FP count matters even warm-started — the earlier
        // b = 0 experiment made Ψ̇ ≡ 0 identically (no weight change → no
        // measure drift), which trivialized the iteration.
        let m1w = run(1, true);
        let m3w = run(3, true);
        assert!(
            (m1w - m3w).abs() > 1e-4,
            "K_FP=1 vs 3 should differ on a strong tilt (m1={m1w}, m3={m3w})"
        );
        // Cold start: the paper's Alg-4 shape — same conclusion, larger gap.
        let m1c = run(1, false);
        let m3c = run(3, false);
        assert!(
            (m1c - m3c).abs() > 1e-4,
            "cold-started K_FP=1 vs 3 should differ on a strong tilt (m1={m1c}, m3={m3c})"
        );
        // Document the bias direction (K=1 under-iterates the tilt).
        println!(
            "K_FP bias on strong tilt: warm m1={m1w:.4} m3={m3w:.4}; cold m1={m1c:.4} m3={m3c:.4}"
        );
    }

    #[test]
    fn damping_rescues_strong_lambda() {
        // Strong tilt: damping 1.0 has a strictly larger fixed-point residual
        // than damping 0.5 (the paper's α=0.5 setting at large λ).
        let mut rng = SplitMix64::new(21);
        let (n, dim) = (48usize, 1usize);
        let mut states = vec![0.0f32; n * dim];
        for (i, s) in states.iter_mut().enumerate() {
            *s = if i % 3 == 0 { 1.0 } else { -1.0 } + 0.15 * rng.next_normal();
        }
        let target: Vec<f32> = (0..24).map(|_| 1.0 + 0.2 * rng.next_normal()).collect();
        let reward = MmdReward::new(0.5, target, dim);
        let b = vec![0.05f32; n * dim];
        let dt = 0.05f32;

        let run_residual = |damping: f32| -> f32 {
            let stepper = FkStepper {
                steer_scale: 40.0,
                k_fp: 3,
                damping,
                clip_log_delta: 1.0,
            };
            let mut scratch = SteeringScratch::new(n, dim);
            let mut lw = vec![0.0f32; n];
            let mut st = states.clone();
            for _ in 0..15 {
                stepper.begin_step(&reward, &st, &mut lw, &mut scratch);
                let steer = scratch.steering().to_vec();
                for (si, s) in st.iter_mut().enumerate() {
                    *s += 0.2 * steer[si] * dt;
                }
                stepper.finish_step(&reward, &st, &b, dt, &mut lw, &mut scratch);
            }
            let mut lw2 = lw;
            stepper.tilt_residual(&reward, &st, &b, dt, &mut lw2, &mut scratch)
        };

        let r_full = run_residual(1.0);
        let r_damped = run_residual(0.5);
        assert!(
            r_damped <= r_full,
            "damping 0.5 residual ({r_damped}) should be <= damping 1.0 ({r_full})"
        );
    }

    // ── T2.3: resampling ──────────────────────────────────────────────────

    #[test]
    fn residual_resample_preserves_counts() {
        // Half the mass on particle 0 of 4 → exactly 2 copies + 2 residual draws.
        let w = [0.5f32, 0.5, 0.0, 0.0];
        let mut out = [0u32; 4];
        residual_resample_into(&w, 4, 0.25, &mut out);
        assert_eq!(out.iter().filter(|&&a| a == 0).count(), 2);
        assert_eq!(out.iter().filter(|&&a| a == 1).count(), 2);
    }

    #[test]
    fn systematic_resample_is_deterministic_and_valid() {
        let w = [0.1f32, 0.2, 0.3, 0.4];
        let mut a = [0u32; 4];
        let mut b = [0u32; 4];
        systematic_resample_into(&w, 4, 0.3, &mut a);
        systematic_resample_into(&w, 4, 0.3, &mut b);
        assert_eq!(a, b, "deterministic at fixed u");
        assert!(a.iter().all(|&x| (x as usize) < 4));
        let c3 = a.iter().filter(|&&x| x == 3).count();
        let c0 = a.iter().filter(|&&x| x == 0).count();
        assert!(c3 >= c0);
    }

    #[test]
    fn clamp_steering_norm_caps_relative_to_drift() {
        let dim = 2;
        let steer = [3.0f32, 4.0, 0.1, 0.0]; // ‖s₀‖ = 5
        let b = [1.0f32, 0.0, 0.0, 1.0]; // ‖b₀‖ = 1
        let mut out = [0.0f32; 4];
        clamp_steering_norm(&steer, &b, dim, 0.1, &mut out);
        let n0 = (out[0] * out[0] + out[1] * out[1]).sqrt();
        assert!(n0 <= 0.1 + 1e-6, "clamped to 0.1·|b|, got {n0}");
        // Particle 1: ‖s‖ = 0.1 < 0.1·‖b‖ = 0.1·1 → unchanged.
        assert!((out[2] - 0.1).abs() < 1e-6 && out[3] == 0.0);
    }

    // ── Stepper determinism ───────────────────────────────────────────────

    #[test]
    fn stepper_two_runs_bit_identical() {
        let run = |seed: u64| -> Vec<f32> {
            let mut rng = SplitMix64::new(seed);
            let (n, dim) = (40usize, 1usize);
            let mut states = vec![0.0f32; n * dim];
            for s in states.iter_mut() {
                *s = rng.next_normal();
            }
            let target: Vec<f32> = (0..16).map(|_| rng.next_normal()).collect();
            let reward = MmdReward::new(0.3, target, dim);
            let stepper = FkStepper {
                steer_scale: 5.0,
                ..Default::default()
            };
            let mut scratch = SteeringScratch::new(n, dim);
            let mut lw = vec![0.0f32; n];
            let mut st = states;
            let b = vec![0.2f32; n];
            for _ in 0..20 {
                stepper.begin_step(&reward, &st, &mut lw, &mut scratch);
                let steer = scratch.steering().to_vec();
                for i in 0..n {
                    let noise = rng.next_normal();
                    st[i] += (0.2 + steer[i]) * 0.05 + 0.1 * noise;
                }
                stepper.finish_step(&reward, &st, &b, 0.05, &mut lw, &mut scratch);
            }
            lw
        };
        let a = run(777);
        let b = run(777);
        assert_eq!(a, b, "two identical-seed runs must be bit-identical");
    }

    #[test]
    fn tilt_residual_near_zero_at_weak_tilt() {
        let mut rng = SplitMix64::new(11);
        let (n, dim) = (32usize, 1usize);
        let mut states = vec![0.0f32; n * dim];
        for s in states.iter_mut() {
            *s = rng.next_normal();
        }
        let target: Vec<f32> = (0..16).map(|_| rng.next_normal()).collect();
        let reward = MmdReward::new(0.2, target, dim);
        let stepper = FkStepper {
            steer_scale: 0.5,
            k_fp: 10,
            ..Default::default()
        };
        let mut scratch = SteeringScratch::new(n, dim);
        let mut lw = vec![0.0f32; n];
        let b = vec![0.1f32; n];
        for _ in 0..5 {
            stepper.begin_step(&reward, &states, &mut lw, &mut scratch);
            stepper.finish_step(&reward, &states, &b, 0.01, &mut lw, &mut scratch);
        }
        let mut lw2 = lw;
        let r = stepper.tilt_residual(&reward, &states, &b, 0.01, &mut lw2, &mut scratch);
        assert!(r < 0.05, "weak-tilt residual should be small, got {r}");
    }

    // ── T4.1: BoM composition ─────────────────────────────────────────────

    #[cfg(all(feature = "bom_sampling", feature = "distributional_steering"))]
    #[test]
    fn bom_fk_weights_concentrate_on_target_mass() {
        use crate::micro_belief::{BoMSampler, LeakyIntegrator, NoiseQueryConfig};

        // Real composition: sample K hypotheses from a LeakyIntegrator
        // (a BoMSampler impl), then FK-weight them against an MMD reward
        // whose target sits at the first hypothesis.
        let dim = 4usize;
        let k = 8usize;
        let kernel = LeakyIntegrator::belief_default(dim);
        let s_prev = vec![0.1f32; dim];
        let x = vec![0.2f32; dim];
        let mut queries = vec![0.0f32; k * dim];
        let mut rng = SplitMix64::new(31);
        for q in queries.iter_mut() {
            *q = 0.3 * rng.next_normal();
        }
        let mut hyps = vec![0.0f32; k * dim];
        let cfg = NoiseQueryConfig::default();
        kernel.sample_k_states(&s_prev, &x, &queries, &mut hyps, &cfg);
        assert!(hyps.iter().all(|v| v.is_finite()));

        let mut target = vec![0.0f32; dim];
        target.copy_from_slice(&hyps[0..dim]);
        let reward = MmdReward::new(4.0, target, dim);
        let mut w = vec![0.0f32; k];
        bom::hypothesis_weights_into(&reward, &hyps, dim, 3.0, 5, 1.0, &mut w);
        let s: f32 = w.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "weights normalize ({s})");
        assert!(
            w[0] > 1.0 / k as f32,
            "target-adjacent hypothesis should be up-weighted (w0={}, uniform={})",
            w[0],
            1.0 / k as f32
        );

        // Both selection paths return valid indices (they disagree by
        // design — that is the point of the weighted alternative).
        let best_fk = bom::select_best_fk(&reward, &hyps, dim, 3.0, 5);
        let scorer = |h: &[f32]| -> f32 { -h.iter().sum::<f32>() / h.len() as f32 };
        let best_argmax = kernel.select_best(&hyps, scorer, k);
        assert!(best_fk < k && best_argmax < k);
    }
}
