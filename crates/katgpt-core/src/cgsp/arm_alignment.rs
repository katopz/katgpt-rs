//! `TrajectoryAlignedCuriosity` — per-arm drift-alignment curiosity gate
//! (Plan 610, Research 591; arXiv:2609.30063 modelless extraction).
//!
//! ## Pinned claim
//!
//! Each CGSP candidate is scored by how strongly its latent direction is
//! involved in the direction the bandit's behavior is currently moving:
//!
//! ```text
//! m_t   = Σ_j p_j(t) · g_j                       priority-weighted mean pull (latent)
//! d_t   = fast_ema(m) − slow_ema(m)              TemporalDerivativeKernel (10:1)
//! s_j   ← (1−α_s)·s_j + α_s·|d_j|                per-coordinate drift scale
//! û     = normalize( d_j / (s_j + κ·max_i s_i) ) preconditioned drift direction
//! r̃_k   = sigmoid( β · |⟨ĝ_k, û⟩| )              per arm, ĝ_k = g_k / ‖g_k‖
//! ```
//!
//! `DerivativeCuriosity` (the incumbent, Plan 277 F4) pools `d` away into
//! `sigmoid(β·‖d‖₂)` — one score per cycle, direction-blind. Its own module
//! docs call the lost per-arm differentiation "the key semantic loss". This
//! gate keeps the direction and scores each arm against it.
//!
//! ## Two drift summaries (Issue 899)
//!
//! The sampler is generic over a [`DriftSummary`]:
//! - [`SecondMomentDrift`] is the default and the preferred summary. It reads
//!   the per-arm share derivative through the pool's squared-cosine kernel,
//!   z-scored against a simplex-correct null. Bench 901: loop-sound in both
//!   directions; G1 held-out 0.766 and G4 2.07× miss their bars.
//! - [`FirstMomentDrift`] is the Plan 610 form documented below. Bench 900 /
//!   901: it amplifies drift toward coherent clusters and loses badly when
//!   the better family is spread. It is kept as the comparison arm.
//!
//! Neither passes every GOAT bar, so the feature stays opt-in.
//!
//! ## Why the drift lives in LATENT space (a Plan 610 correction)
//!
//! Research 591 wrote the drift as `TemporalDerivativeKernel::observe(&pref_buf)`.
//! That is the derivative of the priority vector, which is indexed by ARM.
//! The candidate direction `ĝ_k` is indexed by LATENT coordinate. A dot product
//! between the two only makes sense when the pool is the identity basis. The
//! faithful transplant of the paper's `⟨∇_θ L_i, δθ⟩` needs both vectors in the
//! same space. The movement of the priority-weighted mean pull `m` is the
//! behavior's movement in the space the candidates live in. EMAs are linear,
//! so `d = Σ_j (fast_j − slow_j)·g_j` for a fixed pool: the same per-arm
//! derivative the incumbent computes, projected through the pool.
//!
//! What this changes: an arm whose OWN priority is static, but whose direction
//! lies on the drift axis, is credited. A per-arm "is my own preference moving"
//! score cannot do that (Plan 610 G2's second baseline measures this).
//!
//! ## The preconditioner and its honest scope
//!
//! `s_j` is the EMA of `|d_j|`. Plan 610 calls it "RMS"; it is a first-moment
//! scale, homogeneous of degree 1 in `d`. Dividing by it turns covariance into
//! correlation: the paper's AdamW preconditioner, stripped of gradients. On its
//! own it is sign-like. Any coordinate that drifts consistently, however
//! slightly, is inflated to unit weight. That would let a 1%-magnitude
//! wobble in 15 noise coordinates outvote the real drift axis. The floor
//! `κ·max_i s_i` bounds the inflation to one dynamic range of `1/κ`
//! (default `κ = 0.1`, one decade):
//!
//! - **Global** rescaling `d → c·d` leaves `û` invariant for every `κ`.
//! - **Per-coordinate** rescaling `d_j → c_j·d_j` leaves `û` invariant exactly
//!   at `κ = 0`, and approximately for coordinates above the floor otherwise.
//!   Coordinates below the floor are deliberately suppressed.
//!
//! The EMA starts from zero without bias correction. The correction factor
//! `1 − (1−α)^t` is common to every coordinate, so it cancels in the
//! normalization.
//!
//! ## `|·|` credits anti-drift arms
//!
//! An arm pointing directly against the drift scores as high as one pointing
//! along it: "involved in the current learning direction", not "agrees with
//! it". On a probability simplex, raising one family lowers every other arm.
//! So a family whose pull centroid is non-zero is genuinely anti-aligned with
//! any drift toward another family, and it is credited by design.
//!
//! ## Signal-diff vs verified neighbors
//!
//! | Mechanism | Consumes | Differs by |
//! |---|---|---|
//! | `DerivativeCuriosity` (in-stack) | `‖d‖₂`, global per cycle | direction-blind, cannot rank arms |
//! | CGSP `(1 − solve_rate)·guide_score` (in-stack) | target relevance × failure | an arm can be relevant, unsolved and orthogonal to the drift |
//! | AdaS, arXiv:2006.06587 (verified 2026-09-26) | cosine of successive gradients | optimizer-side step sizing, never a selection gate |
//! | LESS, arXiv:2402.04333 | AdamW-preconditioned gradient dot vs a VALIDATION gradient | anchored to a held-out target, not the learner's own movement |
//! | RHO-LOSS, arXiv:2202.03258 | loss-magnitude learning progress | no direction term |
//!
//! ## Honest caveat
//!
//! Behavioral drift standing in for parameter movement is an assumption.
//! The riir-train-side bridge correlation (probe drift vs parameter movement
//! across two frozen checkpoints, riir-train Plan 420 T3) has not been
//! measured. Until it is, every claim here is mechanism-level.
//!
//! ## Latent vs raw boundary
//!
//! `m`, `d`, `s`, `û` and the candidate directions are latent, local, never
//! synced. Only the bounded scalars `r̃_k ∈ [0.5, 1)` may cross the sync
//! boundary, under the same contract as `DerivativeCuriosity`.
//!
//! ## Measured (Bench 900, corrected by the Bench 900 addendum + Bench 901)
//!
//! The first moment below is a GOAT FAIL:
//! - G1 planted-drift AUC is 0.789, held-out 0.605.
//! - With the kernel warm-started, the preconditioned form also FAILS the
//!   negative control (0.869), while the preconditioner-off form passes it
//!   (0.525). The preconditioner is refuted.
//! - In the loop it wins toward a coherent family (−130 cycles vs a
//!   matched-uniform bonus) and LOSES by +108 when the better family has a
//!   zero pull centroid. That is structural: the pull cannot see mass moving
//!   onto `±e_i` pairs.
//!
//! Use [`SecondMomentDrift`], the default, which is loop-sound in both
//! directions.
//!
//! ## Cost
//!
//! `O(n_arms · dim)` to form `m` (one SIMD axpy per arm with non-zero
//! priority) plus `O(dim)` kernel/preconditioner plus `O(k · dim)` scoring.
//! Zero steady-state allocations: fixed `[f32; D]` state plus a score buffer
//! reused in place.

use crate::cgsp::conjecturer::PoolConjecturer;
use crate::cgsp::loop_::{CgspConfig, renormalize_priorities};
use crate::cgsp::traits::{CollapseSignal, CuriosityConjecturer, HintDeltaBandit};
use crate::cgsp::types::{
    Candidate, CycleResult, CycleStats, DEFAULT_POOL_SIZE, Direction, Priority, ScratchBuffers,
    Target, entropy_nats, sigmoid,
};
use crate::simd::{simd_dot_f32, simd_fused_decay_write};
use crate::temporal_deriv::TemporalDerivativeKernel;

mod second_moment;
pub use second_moment::{DEFAULT_Z_BETA, SecondMomentDrift};

/// Default β for the per-arm alignment sigmoid. `|cos| = 0.5` maps to
/// `sigmoid(2) ≈ 0.88`; `|cos| = 0` maps to the neutral `0.5`.
pub const DEFAULT_ALIGN_BETA: f32 = 4.0;

/// Default EMA coefficient for the per-coordinate drift scale `s_j`.
/// Slower than the kernel's slow EMA's reciprocal horizon would suggest on
/// purpose: the scale should describe the drift's typical size, not track it.
pub const DEFAULT_SCALE_ALPHA: f32 = 0.05;

/// Default relative floor `κ`: coordinates drifting less than `κ` times the
/// dominant coordinate's scale are suppressed rather than inflated.
pub const DEFAULT_FLOOR_KAPPA: f32 = 0.1;

/// Absolute guard against `0/0` on a coordinate that has never moved.
const SCALE_EPS: f32 = 1e-12;

/// Per-coordinate drift preconditioner (the modelless AdamW `P`).
///
/// Fixed-size state, zero allocation. See the [module docs](self) for the
/// invariance contract and why the relative floor exists.
#[derive(Clone, Debug)]
pub struct DriftPreconditioner<const N: usize> {
    scale: [f32; N],
    alpha: f32,
    kappa: f32,
}

impl<const N: usize> DriftPreconditioner<N> {
    /// Build with an explicit scale-EMA coefficient `alpha ∈ (0, 1]` and
    /// relative floor `kappa ≥ 0` (`0` = pure per-coordinate normalization).
    pub fn new(alpha: f32, kappa: f32) -> Self {
        debug_assert!(
            alpha > 0.0 && alpha <= 1.0,
            "scale alpha must be in (0, 1], got {alpha}"
        );
        debug_assert!(
            kappa.is_finite() && kappa >= 0.0,
            "kappa must be finite and >= 0, got {kappa}"
        );
        Self {
            scale: [0.0; N],
            alpha: alpha.clamp(f32::MIN_POSITIVE, 1.0),
            kappa: kappa.max(0.0),
        }
    }

    /// Current per-coordinate scale (read-only; snapshot/telemetry).
    #[inline]
    pub fn scale(&self) -> &[f32; N] {
        &self.scale
    }

    /// Zero the scale state (entity respawn / session restart).
    #[inline]
    pub fn reset(&mut self) {
        self.scale = [0.0; N];
    }

    /// Absorb one drift sample and write the unit preconditioned direction
    /// into `out`. Returns the preconditioned norm before normalization;
    /// `0.0` means no usable direction, and `out` is then all zeros.
    pub fn precondition(&mut self, d: &[f32; N], out: &mut [f32; N]) -> f32 {
        let decay = 1.0 - self.alpha;
        let mut max_scale = 0.0f32;
        for (s, &x) in self.scale.iter_mut().zip(d.iter()) {
            *s = decay * *s + self.alpha * x.abs();
            max_scale = max_scale.max(*s);
        }
        let floor = self.kappa * max_scale + SCALE_EPS;
        for ((o, &x), &s) in out.iter_mut().zip(d.iter()).zip(self.scale.iter()) {
            *o = x / (s + floor);
        }
        let norm = simd_dot_f32(out, out, N).max(0.0).sqrt();
        if norm > SCALE_EPS {
            let inv = 1.0 / norm;
            for o in out.iter_mut() {
                *o *= inv;
            }
            norm
        } else {
            out.fill(0.0);
            0.0
        }
    }
}

impl<const N: usize> Default for DriftPreconditioner<N> {
    fn default() -> Self {
        Self::new(DEFAULT_SCALE_ALPHA, DEFAULT_FLOOR_KAPPA)
    }
}

/// `sigmoid(β · |⟨ĝ, û⟩|)` with `ĝ = direction / ‖direction‖` and `û` unit
/// (or zero). A zero direction or a zero drift scores the neutral `0.5`.
#[inline]
pub fn alignment_score(direction: &[f32], u_hat: &[f32], beta: f32) -> f32 {
    let n = direction.len().min(u_hat.len());
    let g2 = simd_dot_f32(direction, direction, n);
    if g2 <= SCALE_EPS {
        return 0.5;
    }
    let cos = simd_dot_f32(direction, u_hat, n) / g2.sqrt();
    sigmoid(beta * cos.abs())
}

/// How a [`TrajectoryAlignedCuriosity`] summarizes the bandit's drift and
/// scores one candidate against it (strategy; Issue 899).
///
/// Implementations own their own fixed-size state and their own β, because
/// the score's natural scale differs: a cosine for [`FirstMomentDrift`], a
/// null z-score for [`SecondMomentDrift`].
pub trait DriftSummary {
    /// Absorb the current priority table. `pool` is the frozen direction pool,
    /// in the same order as `priorities`. Returns a drift magnitude for
    /// telemetry. Must not allocate once warm.
    fn observe(&mut self, priorities: &[Priority], pool: &[Direction]) -> f32;

    /// Score one sampled candidate, in `[0.5, 1)`. `0.5` is neutral.
    fn score(&self, candidate: &Candidate, pool: &[Direction]) -> f32;

    /// Zero all temporal state (entity respawn / session restart).
    fn reset(&mut self);
}

/// Plan 610's shipped summary: the preconditioned drift of the
/// priority-weighted mean pull `m = Σ_j p_j g_j` (see the [module docs](self)).
/// Bench 900 GOAT FAIL; kept as Issue 899's comparison arm.
///
/// `D` is the LATENT dimension bound: it MUST be `>=` the pool directions'
/// dimension. Padded coordinates stay zero and contribute nothing.
#[derive(Clone, Debug)]
pub struct FirstMomentDrift<const D: usize = DEFAULT_POOL_SIZE> {
    kernel: TemporalDerivativeKernel<D>,
    precond: DriftPreconditioner<D>,
    beta: f32,
    /// Priority-weighted mean pull `m`, rebuilt each observation.
    pull_buf: [f32; D],
    /// Unit preconditioned drift `û` (zero when there is no drift).
    u_hat: [f32; D],
    /// First observation seen (the kernel is warm-started on it).
    primed: bool,
}

impl<const D: usize> FirstMomentDrift<D> {
    /// 10:1 kernel (`0.3 / 0.03`), default preconditioner, β = 4.
    pub fn new() -> Self {
        Self {
            kernel: TemporalDerivativeKernel::default(),
            precond: DriftPreconditioner::default(),
            beta: DEFAULT_ALIGN_BETA,
            pull_buf: [0.0; D],
            u_hat: [0.0; D],
            primed: false,
        }
    }

    /// Override β (alignment sigmoid inverse temperature).
    #[inline]
    pub fn with_beta(mut self, beta: f32) -> Self {
        debug_assert!(
            beta.is_finite() && beta > 0.0,
            "beta must be finite and positive, got {beta}"
        );
        self.beta = beta;
        self
    }

    /// Override the kernel's fast/slow EMA coefficients.
    #[inline]
    pub fn with_alphas(mut self, alpha_fast: f32, alpha_slow: f32) -> Self {
        self.kernel = TemporalDerivativeKernel::new(alpha_fast, alpha_slow);
        self
    }

    /// Override the preconditioner (scale EMA `alpha`, relative floor `kappa`).
    #[inline]
    pub fn with_preconditioner(mut self, alpha: f32, kappa: f32) -> Self {
        self.precond = DriftPreconditioner::new(alpha, kappa);
        self
    }

    /// Current unit preconditioned drift direction `û` (zero if none).
    #[inline]
    pub fn drift_direction(&self) -> &[f32; D] {
        &self.u_hat
    }

    /// Score one arbitrary latent direction against the current drift.
    #[inline]
    pub fn score_direction(&self, direction: &Direction) -> f32 {
        alignment_score(&direction.coords, &self.u_hat, self.beta)
    }
}

impl<const D: usize> Default for FirstMomentDrift<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> DriftSummary for FirstMomentDrift<D> {
    /// Rebuild the mean pull, advance the kernel and the preconditioner.
    /// Returns raw `‖d‖₂`.
    fn observe(&mut self, priorities: &[Priority], pool: &[Direction]) -> f32 {
        debug_assert_eq!(
            priorities.len(),
            pool.len(),
            "priority vector length must equal the pool size"
        );
        self.pull_buf.fill(0.0);
        for (&p, g) in priorities.iter().zip(pool.iter()) {
            if p != 0.0 {
                let n = g.coords.len().min(D);
                // decay = 1 turns the fused decay-write into a SIMD axpy.
                simd_fused_decay_write(&mut self.pull_buf[..n], 1.0, &g.coords[..n], p);
            }
        }
        // Warm-start on the first observation (Issue 899): from the kernel's
        // zero init, `m` reads as drifting from 0 to its mean for ~100 steps
        // (slow α = 0.03), a drift along the pool's dense directions.
        if !self.primed {
            self.kernel.fast = self.pull_buf;
            self.kernel.slow = self.pull_buf;
            self.primed = true;
        }
        let d = self.kernel.observe(&self.pull_buf);
        let norm = simd_dot_f32(&d, &d, D).max(0.0).sqrt();
        self.precond.precondition(&d, &mut self.u_hat);
        norm
    }

    #[inline]
    fn score(&self, candidate: &Candidate, _pool: &[Direction]) -> f32 {
        alignment_score(&candidate.direction.coords, &self.u_hat, self.beta)
    }

    fn reset(&mut self) {
        self.kernel.reset();
        self.precond.reset();
        self.pull_buf = [0.0; D];
        self.u_hat = [0.0; D];
        self.primed = false;
    }
}

/// Per-arm trajectory-aligned curiosity conjecturer (Plan 610, Issue 899).
///
/// Wraps a [`PoolConjecturer`] for sampling, the way [`DerivativeCuriosity`]
/// does, so the sampling distribution stays the CGSP reference. It is
/// bit-identical to a bare `PoolConjecturer` with the same seed, and the
/// [`DriftSummary`] `S` only adds scores as a side channel.
///
/// [`DerivativeCuriosity`]: crate::cgsp::DerivativeCuriosity
#[derive(Debug)]
pub struct TrajectoryAlignedCuriosity<S: DriftSummary = SecondMomentDrift> {
    pool_conjecturer: PoolConjecturer,
    summary: S,
    /// Drift magnitude from the most recent observation (telemetry).
    last_drift_norm: f32,
    /// Per-candidate alignment scores from the most recent scoring call.
    scores: Vec<f32>,
    /// Mean of `scores` (compat with `DerivativeCuriosity` telemetry).
    last_interestingness: f32,
}

impl<const A: usize> TrajectoryAlignedCuriosity<SecondMomentDrift<A>> {
    /// Build with the Issue 899 second-moment summary: the preferred summary,
    /// and the only one loop-sound in both directions (Bench 901).
    pub fn second_moment(pool: Vec<Direction>, seed: u64) -> Self {
        let summary = SecondMomentDrift::for_pool(&pool);
        Self::with_summary(pool, seed, summary)
    }
}

impl<S: DriftSummary + Default> TrajectoryAlignedCuriosity<S> {
    /// Build over a frozen direction pool with a default-constructed summary.
    /// `seed` seeds the inner sampler.
    pub fn new(pool: Vec<Direction>, seed: u64) -> Self {
        Self::with_summary(pool, seed, S::default())
    }
}

impl<S: DriftSummary> TrajectoryAlignedCuriosity<S> {
    /// Build over a frozen direction pool with an explicit summary (needed for
    /// summaries that precompute over the pool, e.g. [`SecondMomentDrift`]).
    pub fn with_summary(pool: Vec<Direction>, seed: u64, summary: S) -> Self {
        let k_hint = pool.len().min(DEFAULT_POOL_SIZE);
        Self {
            pool_conjecturer: PoolConjecturer::new(pool, seed),
            summary,
            last_drift_norm: 0.0,
            scores: Vec::with_capacity(k_hint),
            last_interestingness: 0.5,
        }
    }

    /// Enable perturbation on the inner sampler (see `PoolConjecturer`).
    #[inline]
    pub fn with_perturbation(mut self, magnitude: f32) -> Self {
        self.pool_conjecturer = self.pool_conjecturer.with_perturbation(magnitude);
        self
    }

    /// The drift summary (read-only; telemetry and summary-specific reads).
    #[inline]
    pub fn summary(&self) -> &S {
        &self.summary
    }

    /// Per-candidate scores from the most recent
    /// [`sample_candidates`](CuriosityConjecturer::sample_candidates) or
    /// [`score_candidates`](Self::score_candidates), in candidate order.
    #[inline]
    pub fn last_alignment_scores(&self) -> &[f32] {
        &self.scores
    }

    /// Mean of [`last_alignment_scores`](Self::last_alignment_scores).
    #[inline]
    pub fn last_interestingness(&self) -> f32 {
        self.last_interestingness
    }

    /// Drift magnitude from the most recent observation.
    #[inline]
    pub fn last_drift_norm(&self) -> f32 {
        self.last_drift_norm
    }

    /// Observe the bandit's priorities through the summary. Returns its drift
    /// magnitude.
    pub fn observe_drift(&mut self, priorities: &[Priority]) -> f32 {
        let pool = self.pool_conjecturer.pool_directions();
        self.last_drift_norm = self.summary.observe(priorities, pool);
        self.last_drift_norm
    }

    /// Score `candidates` against the current drift. The buffer is reused in
    /// place: zero allocations once its capacity reaches `k`.
    pub fn score_candidates(&mut self, candidates: &[Candidate]) -> &[f32] {
        let pool = self.pool_conjecturer.pool_directions();
        self.scores.clear();
        let mut sum = 0.0f32;
        for c in candidates {
            let s = self.summary.score(c, pool);
            sum += s;
            self.scores.push(s);
        }
        self.last_interestingness = match candidates.len() {
            0 => 0.5,
            n => sum / n as f32,
        };
        &self.scores
    }

    /// Score one candidate against the current drift (read-only).
    #[inline]
    pub fn score_candidate(&self, candidate: &Candidate) -> f32 {
        self.summary
            .score(candidate, self.pool_conjecturer.pool_directions())
    }

    /// Reset the summary and telemetry (respawn / restart).
    pub fn reset(&mut self) {
        self.summary.reset();
        self.last_drift_norm = 0.0;
        self.scores.clear();
        self.last_interestingness = 0.5;
    }

    /// One Solver-free cycle rewarding each sampled arm with its OWN
    /// alignment score. It mirrors `DerivativeCuriosity::cycle_curiosity`,
    /// whose reward is one global value per cycle.
    pub fn cycle_aligned<B, Col>(
        &mut self,
        target: &Target,
        bandit: &mut B,
        scratch: &mut ScratchBuffers,
        collapse: &mut Col,
        config: &CgspConfig,
    ) -> CycleResult
    where
        B: HintDeltaBandit,
        Col: CollapseSignal,
    {
        scratch.cdf_scratch.clear();
        let k = config.k;
        // `ensure_len`, not `resize(k, Candidate::new(Direction::zeros(..)))`:
        // the resize default is BUILT (one heap Vec) even when it is unused.
        scratch.ensure_len(k, target.dim());
        self.sample_candidates(
            target,
            bandit.priorities(),
            &mut scratch.candidates,
            &mut scratch.cdf_scratch,
        );
        for (c, &s) in scratch.candidates.iter().zip(self.scores.iter()) {
            if c.pool_index != usize::MAX {
                bandit.absorb(c.pool_index, s);
            }
        }
        renormalize_priorities(bandit.priorities_mut());
        let mut result = CycleResult {
            collapse_triggered: false,
            batch_degenerate: false,
            stats: CycleStats {
                candidates_sampled: k as u32,
                candidates_admitted: k as u32,
                candidates_solved: 0,
                mean_guide_score: 0.0,
                mean_r_synth: self.last_interestingness,
                priority_entropy: entropy_nats(bandit.priorities()),
            },
        };
        if collapse.check_collapse(bandit.priorities(), &result) {
            collapse.inject_exploration(bandit.priorities_mut(), config.exploration_magnitude);
            renormalize_priorities(bandit.priorities_mut());
            result.collapse_triggered = true;
            result.stats.priority_entropy = entropy_nats(bandit.priorities());
        }
        result
    }
}

impl<S: DriftSummary> CuriosityConjecturer for TrajectoryAlignedCuriosity<S> {
    /// Observe the drift, delegate sampling to the inner `PoolConjecturer`,
    /// then score the sampled candidates (read via `last_alignment_scores`).
    fn sample_candidates(
        &mut self,
        target: &Target,
        priorities: &[Priority],
        out: &mut [Candidate],
        cdf_scratch: &mut Vec<f32>,
    ) {
        self.observe_drift(priorities);
        self.pool_conjecturer
            .sample_candidates(target, priorities, out, cdf_scratch);
        self.score_candidates(out);
    }

    #[inline]
    fn pool_size(&self) -> usize {
        self.pool_conjecturer.pool_size()
    }

    #[inline]
    fn pool_directions(&self) -> &[Direction] {
        self.pool_conjecturer.pool_directions()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cgsp::loop_::EntropyCollapse;

    struct VecBandit {
        prios: Vec<f32>,
    }
    impl HintDeltaBandit for VecBandit {
        fn absorb(&mut self, arm: usize, reward: f32) {
            if let Some(p) = self.prios.get_mut(arm) {
                *p += reward.max(0.0);
            }
        }
        fn priority(&self, arm: usize) -> Priority {
            self.prios.get(arm).copied().unwrap_or(0.0)
        }
        fn priorities(&self) -> &[Priority] {
            &self.prios
        }
        fn priorities_mut(&mut self) -> &mut [Priority] {
            &mut self.prios
        }
    }

    fn axis(dim: usize, i: usize, sign: f32) -> Direction {
        let mut coords = vec![0.0f32; dim];
        coords[i] = sign;
        Direction { coords }
    }

    /// Four arms on ±e0, ±e1 plus arm 4 on e0.
    fn pool5() -> Vec<Direction> {
        vec![
            axis(4, 0, 1.0),
            axis(4, 0, -1.0),
            axis(4, 1, 1.0),
            axis(4, 1, -1.0),
            axis(4, 0, 1.0),
        ]
    }

    #[test]
    fn alignment_score_monotone_in_abs_cos() {
        let u = [1.0f32, 0.0, 0.0, 0.0];
        let mut last = 0.0;
        for step in 0..=10 {
            let t = step as f32 / 10.0; // |cos| = t
            let g = [t, (1.0 - t * t).max(0.0).sqrt(), 0.0, 0.0];
            let s = alignment_score(&g, &u, 4.0);
            assert!(s >= last - 1e-6, "not monotone at |cos|={t}: {s} < {last}");
            last = s;
        }
        // Sign-blind: anti-aligned scores equal to aligned.
        let a = alignment_score(&[1.0, 0.0, 0.0, 0.0], &u, 4.0);
        let b = alignment_score(&[-1.0, 0.0, 0.0, 0.0], &u, 4.0);
        assert!((a - b).abs() < 1e-7);
        // Length-blind: ĝ is normalized.
        let c = alignment_score(&[7.0, 0.0, 0.0, 0.0], &u, 4.0);
        assert!((a - c).abs() < 1e-6);
    }

    #[test]
    fn alignment_score_neutral_on_zero_inputs() {
        assert_eq!(alignment_score(&[0.0; 4], &[1.0, 0.0, 0.0, 0.0], 4.0), 0.5);
        assert!((alignment_score(&[1.0, 0.0, 0.0, 0.0], &[0.0; 4], 4.0) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn preconditioner_global_scale_invariant() {
        let mut a: DriftPreconditioner<4> = DriftPreconditioner::default();
        let mut b: DriftPreconditioner<4> = DriftPreconditioner::default();
        let (mut ua, mut ub) = ([0.0f32; 4], [0.0f32; 4]);
        for t in 0..50 {
            let x = t as f32 * 0.1;
            let d = [x.sin(), 0.3 * x.cos(), 0.01, -0.2];
            let d2 = d.map(|v| v * 1000.0);
            a.precondition(&d, &mut ua);
            b.precondition(&d2, &mut ub);
            for j in 0..4 {
                assert!(
                    (ua[j] - ub[j]).abs() < 1e-5,
                    "t={t} j={j}: {ua:?} vs {ub:?}"
                );
            }
        }
    }

    #[test]
    fn preconditioner_per_coordinate_invariant_at_kappa_zero() {
        let mut a: DriftPreconditioner<4> = DriftPreconditioner::new(0.05, 0.0);
        let mut b: DriftPreconditioner<4> = DriftPreconditioner::new(0.05, 0.0);
        let s = [0.5f32, 2.0, 1.7, 0.6];
        let (mut ua, mut ub) = ([0.0f32; 4], [0.0f32; 4]);
        for t in 0..50 {
            let x = t as f32 * 0.37;
            let d = [x.sin(), x.cos(), 0.5 * (2.0 * x).sin(), 0.2];
            let d2 = [d[0] * s[0], d[1] * s[1], d[2] * s[2], d[3] * s[3]];
            a.precondition(&d, &mut ua);
            b.precondition(&d2, &mut ub);
            for j in 0..4 {
                assert!(
                    (ua[j] - ub[j]).abs() < 1e-5,
                    "t={t} j={j}: {ua:?} vs {ub:?}"
                );
            }
        }
    }

    #[test]
    fn preconditioner_floor_suppresses_tiny_coordinates() {
        // κ = 0: a coordinate drifting 1000× less is inflated to equal weight.
        // κ = 0.1: it is suppressed.
        let d = [1.0f32, 0.001, 0.0, 0.0];
        let mut raw: DriftPreconditioner<4> = DriftPreconditioner::new(0.05, 0.0);
        let mut floored: DriftPreconditioner<4> = DriftPreconditioner::new(0.05, 0.1);
        let (mut ur, mut uf) = ([0.0f32; 4], [0.0f32; 4]);
        for _ in 0..100 {
            raw.precondition(&d, &mut ur);
            floored.precondition(&d, &mut uf);
        }
        assert!((ur[0] - ur[1]).abs() < 1e-3, "κ=0 should equalize: {ur:?}");
        assert!(uf[1] < 0.02 * uf[0], "κ=0.1 should suppress: {uf:?}");
    }

    #[test]
    fn preconditioner_zero_drift_yields_zero_direction() {
        let mut p: DriftPreconditioner<4> = DriftPreconditioner::default();
        let mut u = [9.0f32; 4];
        assert_eq!(p.precondition(&[0.0; 4], &mut u), 0.0);
        assert_eq!(u, [0.0; 4]);
    }

    #[test]
    fn drift_on_axis_ranks_axis_arms_above_orthogonal() {
        // Priority mass moves onto arm 0 (+e0). Arms on ±e0 (0, 1, 4) are
        // involved in the drift; arms on ±e1 (2, 3) are not. Arm 4's own
        // priority never changes except through renormalization.
        let mut tac: TrajectoryAlignedCuriosity<FirstMomentDrift<4>> =
            TrajectoryAlignedCuriosity::new(pool5(), 1);
        let mut p = [0.2f32; 5];
        for _ in 0..30 {
            p[0] += 0.02;
            let z: f32 = p.iter().sum();
            let q = p.map(|v| v / z);
            tac.observe_drift(&q);
        }
        let pool = pool5();
        let s: Vec<f32> = pool
            .iter()
            .map(|g| tac.summary().score_direction(g))
            .collect();
        for &on in &[0usize, 1, 4] {
            for &off in &[2usize, 3] {
                assert!(
                    s[on] > s[off],
                    "arm {on} ({}) !> arm {off} ({})",
                    s[on],
                    s[off]
                );
            }
        }
    }

    #[test]
    fn scores_track_candidate_order_and_mean() {
        let mut tac: TrajectoryAlignedCuriosity<FirstMomentDrift<4>> =
            TrajectoryAlignedCuriosity::new(pool5(), 3);
        let target = Target::new(axis(4, 0, 1.0));
        let mut out = vec![Candidate::new(Direction::zeros(4), usize::MAX); 3];
        let mut cdf = Vec::new();
        tac.sample_candidates(&target, &[0.6, 0.1, 0.1, 0.1, 0.1], &mut out, &mut cdf);
        let scores = tac.last_alignment_scores().to_vec();
        assert_eq!(scores.len(), 3);
        for (c, s) in out.iter().zip(&scores) {
            assert!((tac.summary().score_direction(&c.direction) - s).abs() < 1e-7);
            assert!((0.5..1.0).contains(s), "score out of range: {s}");
        }
        let mean = scores.iter().sum::<f32>() / 3.0;
        assert!((tac.last_interestingness() - mean).abs() < 1e-6);
    }

    #[test]
    fn reset_clears_state() {
        let mut tac: TrajectoryAlignedCuriosity<FirstMomentDrift<4>> =
            TrajectoryAlignedCuriosity::new(pool5(), 3);
        // The first observation only warm-starts the kernel (zero drift).
        tac.observe_drift(&[0.6, 0.1, 0.1, 0.1, 0.1]);
        assert_eq!(tac.last_drift_norm(), 0.0);
        tac.observe_drift(&[0.1, 0.6, 0.1, 0.1, 0.1]);
        assert!(tac.last_drift_norm() > 0.0);
        tac.reset();
        assert_eq!(tac.last_drift_norm(), 0.0);
        assert_eq!(tac.summary().drift_direction(), &[0.0; 4]);
        assert!(tac.last_alignment_scores().is_empty());
        assert!(tac.summary.precond.scale().iter().all(|&s| s == 0.0));
    }

    #[test]
    fn cycle_aligned_finite_and_recovers_from_collapse() {
        let pool: Vec<Direction> = (0..8).map(|i| axis(8, i, 1.0)).collect();
        let mut tac: TrajectoryAlignedCuriosity<FirstMomentDrift<8>> =
            TrajectoryAlignedCuriosity::new(pool.clone(), 5);
        let mut bandit = VecBandit {
            prios: (0..8).map(|i| if i == 3 { 1.0 } else { 0.0 }).collect(),
        };
        let h0 = entropy_nats(bandit.priorities());
        let mut collapse = EntropyCollapse::default();
        let config = CgspConfig::default();
        let target = Target::new(pool[0].clone());
        let mut scratch = ScratchBuffers::new(4, 8);
        let mut triggered = false;
        let mut max_h = h0;
        for cycle in 0..20 {
            let r = tac.cycle_aligned(&target, &mut bandit, &mut scratch, &mut collapse, &config);
            triggered |= r.collapse_triggered;
            assert!(r.stats.mean_r_synth.is_finite(), "cycle {cycle}");
            assert!(
                bandit
                    .priorities()
                    .iter()
                    .all(|p| p.is_finite() && *p >= 0.0)
            );
            max_h = max_h.max(entropy_nats(bandit.priorities()));
        }
        assert!(triggered, "collapse never fired");
        assert!(max_h > h0, "entropy never rose: {h0} -> {max_h}");
    }
}
