//! Certified frontier — a monotone, provably-sound safe-set expansion operator.
//!
//! Source: [Research 510](../../../.research/510_ActFlow_Certified_Frontier_Expansion.md)
//! distilling De Santi et al., *Active Flow Expansion for Out-of-Distribution
//! Discovery*, [arXiv:2606.08802](https://arxiv.org/abs/2606.08802). The
//! operator itself is the SAFEOPT lineage (Sui et al. 2015/2018) — known art,
//! shipped here the way this repo ships bandits: the operator is standard, the
//! fusion (grow-then-navigate with [`crate::viable_manifold_graph`]) and the
//! modelless framing are the contribution.
//!
//! Plan 580 Phase 1. Feature `certified_frontier`, std-only, zero deps.
//!
//! # What this is
//!
//! Given (a) a buffer of **binary verifier outcomes** per latent cell, (b) a
//! closed-form uncertainty model, and (c) an a-priori **Lipschitz budget**,
//! grow a set of cells that provably satisfy `p(z) >= h`, and answer two
//! questions the caller cannot answer alone:
//!
//! - *Where do I look next?* — [`CertifiedFrontier::acquire_frontier_target`]
//! - *When do I stop looking?* — [`should_advance`]
//!
//! Everything is GD-free: no training, no backprop. Weight-free by
//! construction — the "model" is a Beta-Bernoulli count per cell plus an
//! optional linear-kernel posterior variance.
//!
//! # Phase 0 measured this before it was built
//!
//! [Bench 687](../../../.benchmarks/687_certified_frontier_phase0_poc.md):
//! zero soundness violations, monotone growth, and **51.4x** separation of
//! frontier acquisition over passive sampling at an identical query budget.
//!
//! The same PoC measured something the plan did not ask, and it shapes this
//! API (T0.3): **the Lipschitz dilation is conditional, not free.** A hop is
//! admissible iff `best_cb - h >= L * spacing`, so on a coarse lattice
//! [`CertifiedFrontier::reachability_dilation`] relaxes and certifies
//! *nothing*, silently. That is why [`CertifiedFrontier::dilation_feasibility`]
//! is a first-class, cheap predicate rather than a debug aid: a caller must be
//! able to see a dead dilation without instrumenting a run. Measured crossover
//! (dense world, 6 000 queries, 0 violations throughout): 16x16 and 32x32
//! certify 0 cells by dilation; 64x64 certifies 6; 96x96 certifies 30 of 113
//! (27%). The predicted and observed crossovers agree on all four points.
//!
//! The cause is that a *global* Lipschitz constant charges a plateau hop the
//! steepest-cliff price — and the paper's `L = L_s * L_g` is global in exactly
//! the same way, so this is not an artifact of the Beta substitute. Hence
//! [`FrontierCell::lipschitz`]: a caller with a tighter **a-priori** bound for
//! a region supplies it per cell, and hops pay `max(L_from, L_to)`.
//!
//! # Soundness contract (read before setting `lipschitz`)
//!
//! `cfg.lipschitz` and `FrontierCell::lipschitz` MUST be **a-priori upper
//! bounds** on the local Lipschitz constant of `p` in probability space. They
//! are the one input this module cannot check. Estimating `L` from the same
//! observations that drive expansion is **unsound** — a too-small `L` makes
//! [`CertifiedFrontier::reachability_dilation`] certify cells that are not
//! valid, and no test in this module can see it. The uncertainty model is
//! conservative; the Lipschitz budget is the caller's proof obligation.

use core::f32;

/// Lipschitz constant of the logistic sigmoid — `sup |s'(z)| = s'(0) = 1/4`.
///
/// The `L_s` of the paper's `beta_t` schedule and of `L = L_s * L_g`.
pub const SIGMOID_LIPSCHITZ: f32 = 0.25;

#[inline]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Acquisition-lane sentinel for a cell that is not a candidate. Below every
/// real sigma (which is a non-negative sd), so the argmax skips it without a
/// branch.
const NOT_A_CANDIDATE: f32 = -1.0;

/// sd of the `Beta(1, 1)` prior — the sigma of a never-observed cell.
const BETA_PRIOR_SD: f32 = 0.288_675_13; // sqrt(1/12)

#[inline]
fn dot<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn sq_dist<const D: usize>(a: &[f32; D], b: &[f32; D]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ── configuration ──────────────────────────────────────────────────────────

/// Static configuration for one certified-frontier run.
///
/// `lipschitz` is a proof obligation, not a tuning knob — see the module-level
/// soundness contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrontierConfig {
    /// Ridge regulariser of the kernel posterior (`lambda` in `(K + lambda I)`).
    pub lambda: f32,
    /// Failure probability of the confidence schedule.
    pub delta: f32,
    /// RKHS-norm bound `B` on the latent score field `g`.
    pub b_rkhs: f32,
    /// Validity threshold: a cell is valid iff `p(z) >= h`.
    pub h: f32,
    /// Global a-priori Lipschitz bound on `p` in probability space.
    pub lipschitz: f32,
    /// Acquisition inflation factor (paper's Eq 14 `alpha`); `1.0` = Eq 33.
    pub alpha: f32,
    /// Nearest-neighbour spacing of the cell lattice. Used ONLY by
    /// [`CertifiedFrontier::dilation_feasibility`] to price a representative
    /// hop; the dilation itself uses exact pairwise distances.
    pub cell_spacing: f32,
    /// Cells within this distance of a certified cell are acquisition
    /// candidates. Set to `0.0` to sample strictly inside the certified set.
    pub acquire_radius: f32,
    /// Target certified-bound precision, the `epsilon` of the halting law.
    pub epsilon: f32,
}

impl Default for FrontierConfig {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            delta: 0.05,
            b_rkhs: 1.0,
            h: 0.6,
            lipschitz: 1.0,
            alpha: 1.0,
            cell_spacing: 1.0,
            acquire_radius: 1.5,
            epsilon: 0.05,
        }
    }
}

/// Why a dilation pass will (or will not) admit anything — the T0.3 predicate.
///
/// A dead dilation is otherwise invisible: `reachability_dilation` returns `0`
/// both when the frontier is complete and when every hop is unaffordable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DilationFeasibility {
    /// `max(cb - h)` over certified cells: the best bound headroom available
    /// to pay for a hop. Negative/`-inf` when nothing is certified.
    pub best_headroom: f32,
    /// `L * cell_spacing`: what one representative lattice hop costs.
    pub hop_cost: f32,
    /// `best_headroom >= hop_cost`.
    pub feasible: bool,
    /// `hop_cost - best_headroom`. Positive means the dilation is a no-op and
    /// the certified set can only grow by querying.
    pub deficit: f32,
}

// ── cells ──────────────────────────────────────────────────────────────────

/// One latent cell: its feature, its verifier tally, and its certified bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrontierCell<const D: usize> {
    /// Latent coordinate. Distances between these drive the dilation.
    pub feat: [f32; D],
    /// Count of `true` verifier outcomes.
    pub valid: u32,
    /// Count of `false` verifier outcomes.
    pub invalid: u32,
    /// Monotone non-decreasing lower bound on `p(z)`. Never assigned a smaller
    /// value — this is what makes the certified set monotone (Lemma E.2).
    pub cb: f32,
    /// `cb >= cfg.h` has held at least once.
    pub certified: bool,
    /// Local a-priori Lipschitz bound. `NaN` (the default) falls back to
    /// `cfg.lipschitz`. Must be an upper bound; see the soundness contract.
    pub lipschitz: f32,
    /// Externally supplied posterior sd, e.g. from [`PosteriorBuffer`]. `NaN`
    /// (the default) uses the Beta-Bernoulli sd.
    pub sigma_override: f32,
    /// `true` when a certified cell lies within `cfg.acquire_radius`. Cached
    /// so acquisition is `O(cells)` instead of `O(cells * certified)`: the
    /// neighbourhood is stamped once, when a cell becomes certified, rather
    /// than re-scanned on every query. Maintained by the module; a caller that
    /// changes `acquire_radius` mid-run must call
    /// [`CertifiedFrontier::rebuild_neighborhoods`].
    pub near_certified: bool,
    /// `true` once the cell was admitted by a Lipschitz hop rather than by its
    /// own observations. Pure bookkeeping — this is the T0.3 attribution, and
    /// it must be read at the moment `cb` crosses `h`, never at end-state.
    pub by_dilation: bool,
    /// Beta-Bernoulli sd, recomputed on each `observe` so that acquisition —
    /// which touches every cell on every query — is a flag test and a compare
    /// rather than a divide and a square root. Private: it is derived state,
    /// and a stale write here would silently corrupt every bound.
    beta_sigma: f32,
}

impl<const D: usize> FrontierCell<D> {
    /// A fresh, unobserved, uncertified cell at `feat`.
    #[must_use]
    pub fn new(feat: [f32; D]) -> Self {
        Self {
            feat,
            valid: 0,
            invalid: 0,
            cb: 0.0,
            certified: false,
            lipschitz: f32::NAN,
            sigma_override: f32::NAN,
            near_certified: false,
            by_dilation: false,
            beta_sigma: BETA_PRIOR_SD,
        }
    }
}

impl<const D: usize> Default for FrontierCell<D> {
    fn default() -> Self {
        Self::new([0.0; D])
    }
}

// ── closed-form pieces (plan functions 2, 3, 7) ────────────────────────────

/// Beta-Bernoulli posterior mean and **variance** from a verifier tally.
///
/// Laplace prior `Beta(1, 1)`, so `a = valid + 1`, `b = invalid + 1`:
/// `mu = a / (a + b)`, `var = a b / ((a+b)^2 (a+b+1))`.
///
/// This is the plan's honest substitute for the paper's kernel-logistic
/// `mu_t`, which needs a convex solve. Exact, allocation-free, and monotone in
/// the tally — the properties the soundness proof actually uses.
#[inline]
#[must_use]
pub fn beta_mean_variance(valid: u32, invalid: u32) -> (f32, f32) {
    let a = valid as f32 + 1.0;
    let b = invalid as f32 + 1.0;
    let n = a + b;
    let mean = a / n;
    let var = (a * b) / (n * n * (n + 1.0));
    (mean, var)
}

/// Maximum information gain of a linear kernel in `d` dimensions after `t`
/// observations: `gamma_t = d * ln(1 + t / (d * lambda))`.
///
/// Sub-linear in `t` — the plateau that bounds how long the halting law can
/// keep a caller querying one cell.
#[inline]
#[must_use]
pub fn linear_information_gain(t: u32, d: usize, lambda: f32) -> f32 {
    let d = d.max(1) as f32;
    d * (1.0 + t as f32 / (d * lambda.max(f32::EPSILON))).ln()
}

/// The paper's Eq 31/37 confidence width
/// `beta_t = 4 L_s B + 2 L_s sqrt(2 kappa / lambda * (gamma_t + ln(1/delta)))`
/// with `L_s = 1/4` and `kappa = 1 / (s(B) (1 - s(B)))` closed-form for the
/// sigmoid link.
///
/// Monotone non-decreasing in `t` (pinned by test) — that monotonicity is what
/// lets the union bound cover every round, which in turn is what lets `cb` be
/// a running max without breaking soundness.
#[inline]
#[must_use]
pub fn confidence_schedule(t: u32, delta: f32, lambda: f32, b_rkhs: f32, d_eff: usize) -> f32 {
    let s_b = sigmoid(b_rkhs);
    let kappa = 1.0 / (s_b * (1.0 - s_b)).max(f32::EPSILON);
    let gamma = linear_information_gain(t, d_eff, lambda);
    let delta = delta.clamp(f32::EPSILON, 1.0);
    let inner = 2.0 * kappa / lambda.max(f32::EPSILON) * (gamma + (1.0 / delta).ln());
    4.0 * SIGMOID_LIPSCHITZ * b_rkhs + 2.0 * SIGMOID_LIPSCHITZ * inner.max(0.0).sqrt()
}

/// Union-bound confidence width for the **Beta-Bernoulli substrate**:
/// `sqrt(2 ln(cells * rounds / delta))`.
///
/// The alternative to [`confidence_schedule`], and measurably tighter — see
/// the derivation and the caveat before choosing between them.
///
/// [`confidence_schedule`] is the paper's Eq 31/37, derived for a **kernel
/// logistic** model in which information pools across the input space through
/// the RKHS norm. This module's default posterior is a per-cell Beta-Bernoulli
/// with no pooling at all, so that schedule is answering a harder question than
/// the one being asked, and it shows: Bench 688 T3.4b measured the shipped
/// schedule spending **0.000 of a 0.05 budget** while certifying 35% of the
/// valid region, where a 4x narrower width still held `delta` at 0.023.
///
/// This width instead counts the comparisons directly — `cells * rounds` of
/// them — and asks for a per-comparison failure of `delta / (cells * rounds)`.
///
/// # The assumption, stated plainly
///
/// The `sqrt(2 ln(1/delta'))` z-score is the **sub-Gaussian** tail, applied
/// here to a Beta posterior that is only approximately Gaussian. That makes
/// this width *derived-but-approximate* where [`confidence_schedule`] is
/// worst-case-rigorous. It is offered, not defaulted, and a caller who adopts
/// it owes the empirical calibration check that Bench 688 runs — measured
/// violation rate against `delta`, on their own field. For a rigorous
/// small-`n` bound use an exact Clopper-Pearson / Beta quantile instead; this
/// is the closed-form, allocation-free middle.
#[inline]
#[must_use]
pub fn beta_union_bound(cells: usize, rounds: u32, delta: f32) -> f32 {
    let m = (cells.max(1) as f32) * (rounds.max(1) as f32);
    let delta = delta.clamp(f32::EPSILON, 1.0);
    (2.0 * (m / delta).ln()).max(0.0).sqrt()
}

/// The halting law: a certified hop is guaranteed once `sigma <= eps / (2 beta)`.
///
/// Answers *when do I stop looking at this cell* — the counterpart to
/// [`CertifiedFrontier::acquire_frontier_target`]'s *where do I look next*.
#[inline]
#[must_use]
pub fn should_advance(sigma: f32, beta: f32, epsilon: f32) -> bool {
    sigma <= epsilon / (2.0 * beta.max(f32::EPSILON))
}

/// Round budget implied by the halting law: `T ~ 8 alpha^2 beta^2 gamma / eps^2`.
///
/// A planning figure — how many queries a caller should expect to spend before
/// [`should_advance`] fires.
#[inline]
#[must_use]
pub fn advance_horizon(alpha: f32, beta: f32, gamma: f32, epsilon: f32) -> f32 {
    let e = epsilon.max(f32::EPSILON);
    8.0 * alpha * alpha * beta * beta * gamma / (e * e)
}

// ── Prop 1 design bounds (ship beside, per plan) ───────────────────────────

/// Fraction of the unit sphere in `m` dimensions inside a cap of half-angle
/// `phi`, in the exponential form the plan pre-registers:
/// `exp(-(m - 1) cos^2(phi) / 2)`.
///
/// This is the design law behind Phase 0's measured 51.4x: a narrow valid
/// corridor is exponentially hard to hit by passive sampling, so targeted
/// acquisition separates exponentially in the ambient dimension.
#[inline]
#[must_use]
pub fn spherical_cap_bound(m: usize, phi_rad: f32) -> f32 {
    let m = m.max(1) as f32;
    let c = phi_rad.cos();
    (-(m - 1.0) * c * c / 2.0).exp()
}

/// Laurent-Massart chi-square deviation radius:
/// `sqrt(d + 2 sqrt(d ln(1/delta)) + 2 ln(1/delta))`.
///
/// The concentration radius of a `d`-dimensional isotropic Gaussian — the
/// honest "how far out does a sample land" companion to
/// [`spherical_cap_bound`].
#[inline]
#[must_use]
pub fn laurent_massart_radius(d: usize, delta: f32) -> f32 {
    let d = d as f32;
    let l = (1.0 / delta.clamp(f32::EPSILON, 1.0)).ln();
    (d + 2.0 * (d * l).sqrt() + 2.0 * l).max(0.0).sqrt()
}

// ── diversity / coverage scoreboards (plan function 8) ─────────────────────

/// Capacity of [`sphere_exclusion_coverage`]'s alloc-free center list.
pub const SPHERE_EXCLUSION_MAX_CENTERS: usize = 256;

/// Outcome of a sphere-exclusion scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SphereExclusion {
    /// Number of accepted centers.
    pub centers: usize,
    /// `true` when the scan hit [`SPHERE_EXCLUSION_MAX_CENTERS`] and stopped
    /// accepting. A saturated count is a floor, not a measurement — raise the
    /// threshold or subsample rather than comparing two saturated runs.
    pub saturated: bool,
}

/// Greedy sphere-exclusion cluster count at `threshold`.
///
/// Order-pinned: the greedy scan runs in slice order, so a fixed input order
/// gives a bit-identical count. That determinism is the point — this is a
/// scoreboard for A/B runs, not a clustering algorithm.
///
/// Alloc-free, so the center list is capped at
/// [`SPHERE_EXCLUSION_MAX_CENTERS`]; saturation is reported rather than
/// silently truncating the count.
#[must_use]
pub fn sphere_exclusion_coverage<const D: usize>(
    samples: &[[f32; D]],
    threshold: f32,
) -> SphereExclusion {
    let t2 = threshold * threshold;
    let mut centers = 0usize;
    // Indices of accepted centers, tracked in-place over `samples`.
    let mut accepted = [0usize; SPHERE_EXCLUSION_MAX_CENTERS];
    for (i, s) in samples.iter().enumerate() {
        let covered = accepted[..centers]
            .iter()
            .any(|&c| sq_dist(s, &samples[c]) <= t2);
        if covered {
            continue;
        }
        if centers == SPHERE_EXCLUSION_MAX_CENTERS {
            return SphereExclusion {
                centers,
                saturated: true,
            };
        }
        accepted[centers] = i;
        centers += 1;
    }
    SphereExclusion {
        centers,
        saturated: false,
    }
}

/// Vendi score `exp(-sum lambda_i ln lambda_i)` on a kernel's eigenvalues.
///
/// Eigenvalues are normalised to sum 1 first; zero/negative entries are
/// skipped (`0 ln 0 = 0`). Returns `0.0` for an empty or degenerate spectrum.
#[must_use]
pub fn vendi_diversity(eigs: &[f32]) -> f32 {
    let total: f32 = eigs.iter().filter(|e| **e > 0.0).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut entropy = 0.0;
    for &e in eigs {
        if e > 0.0 {
            let p = e / total;
            entropy -= p * p.ln();
        }
    }
    entropy.exp()
}

// ── exact Eq-10 posterior variance (plan function 1) ───────────────────────

/// Fixed-capacity observation buffer carrying an **incremental Cholesky** of
/// `(K + lambda I)` for the linear kernel `k(a, b) = <a, b>`.
///
/// `append_observation` extends the factor by one row in `O(n^2)`; there is
/// never a re-solve and never an explicit inverse.
///
/// # Size
///
/// The factor is `MAX_OBS * MAX_OBS` floats — 256 KiB at `MAX_OBS = 256`. That
/// is fine on a main thread but will overflow a small spawned-thread stack;
/// box it (`Box::new(PosteriorBuffer::new(..))`) when `MAX_OBS > 256`.
#[derive(Debug, Clone)]
pub struct PosteriorBuffer<const MAX_OBS: usize, const D: usize> {
    feats: [[f32; D]; MAX_OBS],
    y: [f32; MAX_OBS],
    /// Lower-triangular Cholesky factor of `(K + lambda I)`.
    chol: [[f32; MAX_OBS]; MAX_OBS],
    /// `(K + lambda I)^-1 y`, refreshed on append.
    alpha: [f32; MAX_OBS],
    scratch: [f32; MAX_OBS],
    n: usize,
    lambda: f32,
}

impl<const MAX_OBS: usize, const D: usize> PosteriorBuffer<MAX_OBS, D> {
    /// Empty buffer with ridge `lambda`.
    #[must_use]
    pub fn new(lambda: f32) -> Self {
        Self {
            feats: [[0.0; D]; MAX_OBS],
            y: [0.0; MAX_OBS],
            chol: [[0.0; MAX_OBS]; MAX_OBS],
            alpha: [0.0; MAX_OBS],
            scratch: [0.0; MAX_OBS],
            n: 0,
            lambda,
        }
    }

    /// Number of stored observations.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// `true` when no observation has been appended.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Solve `L v = rhs` in place over `scratch[..n]` (forward substitution).
    #[inline]
    fn forward_substitute(chol: &[[f32; MAX_OBS]; MAX_OBS], out: &mut [f32], n: usize) {
        for i in 0..n {
            let mut acc = out[i];
            for j in 0..i {
                acc -= chol[i][j] * out[j];
            }
            out[i] = acc / chol[i][i];
        }
    }

    /// Solve `L^T v = rhs` in place over `out[..n]` (back substitution).
    #[inline]
    fn back_substitute(chol: &[[f32; MAX_OBS]; MAX_OBS], out: &mut [f32], n: usize) {
        for i in (0..n).rev() {
            let mut acc = out[i];
            for j in (i + 1)..n {
                acc -= chol[j][i] * out[j];
            }
            out[i] = acc / chol[i][i];
        }
    }

    /// Append one observation, extending the Cholesky factor by a rank-1 row.
    ///
    /// Returns `false` (and changes nothing) when the buffer is full.
    pub fn append_observation(&mut self, feat: &[f32; D], y: f32) -> bool {
        if self.n >= MAX_OBS {
            return false;
        }
        let n = self.n;
        // w = L^-1 k(X, x_new)
        let feats = &self.feats;
        for (s, f) in self.scratch[..n].iter_mut().zip(feats.iter()) {
            *s = dot(f, feat);
        }
        Self::forward_substitute(&self.chol, &mut self.scratch, n);
        let mut rem = self.lambda + dot(feat, feat);
        let scratch = &self.scratch;
        for (c, w) in self.chol[n][..n].iter_mut().zip(scratch.iter()) {
            *c = *w;
            rem -= *w * *w;
        }
        // `lambda > 0` keeps this strictly positive; the max is a numerical
        // floor for the near-duplicate-feature case, not a correctness patch.
        self.chol[n][n] = rem.max(self.lambda * 1e-6).sqrt();
        self.feats[n] = *feat;
        self.y[n] = y;
        self.n = n + 1;
        self.refresh_alpha();
        true
    }

    fn refresh_alpha(&mut self) {
        let n = self.n;
        self.alpha[..n].copy_from_slice(&self.y[..n]);
        Self::forward_substitute(&self.chol, &mut self.alpha, n);
        Self::back_substitute(&self.chol, &mut self.alpha, n);
    }

    /// Eq 10 exactly:
    /// `sigma^2(x) = k(x,x) - k(x,X) (K + lambda I)^-1 k(X,x)`.
    ///
    /// One forward substitution, `O(n^2)`, no allocation.
    #[must_use]
    pub fn posterior_variance_linear(&self, x: &[f32; D], scratch: &mut [f32]) -> f32 {
        let n = self.n;
        debug_assert!(scratch.len() >= n, "scratch must hold at least len() floats");
        let k_self = dot(x, x);
        if n == 0 {
            return k_self;
        }
        for (s, f) in scratch[..n].iter_mut().zip(self.feats.iter()) {
            *s = dot(f, x);
        }
        Self::forward_substitute(&self.chol, scratch, n);
        let quad: f32 = scratch[..n].iter().map(|v| v * v).sum();
        (k_self - quad).max(0.0)
    }

    /// Ridge posterior mean `k(x, X) (K + lambda I)^-1 y`, `O(n D)` off the
    /// cached `alpha`.
    #[must_use]
    pub fn ridge_mean(&self, x: &[f32; D]) -> f32 {
        self.feats[..self.n]
            .iter()
            .zip(self.alpha[..self.n].iter())
            .map(|(f, a)| dot(f, x) * a)
            .sum()
    }
}

// ── regime-conditional dual form (Plan 580 T5.3) ───────────────────────────

/// The surface both posterior parameterisations share, so a caller can be
/// written once and the *factorisation* chosen by regime.
///
/// The two impls compute the **same quantity by two identities** that are equal
/// in exact arithmetic (Woodbury, below) but not bit-identical in `f32`. Treat
/// a swap as a tolerance-equivalent change, never a bit-identical one — which
/// is why neither of these is on a sync surface.
pub trait LinearPosterior<const D: usize> {
    /// Absorb one observation. `false` means **nothing changed** — the primal
    /// says that when it is full, the dual when a coordinate is non-finite.
    fn append_observation(&mut self, feat: &[f32; D], y: f32) -> bool;

    /// `sigma^2(x)` for the linear kernel `k(a, b) = <a, b>`.
    ///
    /// `scratch` must hold at least [`Self::scratch_len`] floats. An impl whose
    /// query is `O(D^2)` in fixed state ignores it entirely.
    fn posterior_variance_linear(&self, x: &[f32; D], scratch: &mut [f32]) -> f32;

    /// Ridge posterior mean.
    fn ridge_mean(&self, x: &[f32; D]) -> f32;

    /// Observations absorbed.
    fn len(&self) -> usize;

    /// `true` when nothing has been absorbed.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Floats [`Self::posterior_variance_linear`] needs in `scratch`. Zero for
    /// a dual-form impl; a generic caller sizes off this rather than assuming.
    fn scratch_len(&self) -> usize;
}

impl<const MAX_OBS: usize, const D: usize> LinearPosterior<D> for PosteriorBuffer<MAX_OBS, D> {
    #[inline]
    fn append_observation(&mut self, feat: &[f32; D], y: f32) -> bool {
        PosteriorBuffer::append_observation(self, feat, y)
    }
    #[inline]
    fn posterior_variance_linear(&self, x: &[f32; D], scratch: &mut [f32]) -> f32 {
        PosteriorBuffer::posterior_variance_linear(self, x, scratch)
    }
    #[inline]
    fn ridge_mean(&self, x: &[f32; D]) -> f32 {
        PosteriorBuffer::ridge_mean(self, x)
    }
    #[inline]
    fn len(&self) -> usize {
        PosteriorBuffer::len(self)
    }
    #[inline]
    fn scratch_len(&self) -> usize {
        self.n
    }
}

/// Linear-kernel posterior in the **dual** (feature-space) parameterisation: an
/// incremental Cholesky of `A = X^T X + lambda I`, which is `D x D` and so
/// **independent of the observation count**.
///
/// Same three answers as [`PosteriorBuffer`], by the Woodbury identity — for a
/// linear kernel these are the same number exactly:
///
/// ```text
/// k(x,x) - k(x,X) (X X^T + lambda I)^-1 k(X,x)  ==  lambda * x^T (X^T X + lambda I)^-1 x
/// ```
///
/// # When to use which — the regime, not a preference
///
/// [`PosteriorBuffer`] factorises the `n x n` Gram matrix, which is the right
/// end when observations are scarce and the latent is wide (`n < D`) — this
/// plan's own per-cell setting, and it is the *oracle* the dual is gated
/// against. Invert the regime (a long warm-up against a narrow projected
/// feature, `n > D`) and the primal is the wrong factorisation. Measured by the
/// first external consumer (riir-train Plan 357 T1.2,
/// [Bench 563](../../../../riir-train/.benchmarks/563_plan357_t1_2_gp_uncertainty.md))
/// at `D = 32`:
///
/// | | primal | dual |
/// |---|---|---|
/// | variance @ `n = 256` | 158.7 us/query | 1.99 us/query (**79.6x**) |
/// | scaling in `n` | `O(n^2)` | **`O(1)`** (1.007x, `n` = 16 -> 4096) |
/// | state | 291 KiB @ `MAX_OBS = 256`; **64 MiB @ 4096** | **4368 B at any `n`** (`4 D (D+2)` payload + `lambda`/`n`) |
///
/// Use [`prefer_dual`] rather than re-deriving the rule. The choice is made
/// once, at construction: carrying both would give back the memory the dual
/// exists to save.
///
/// # Numerics
///
/// Better-conditioned than the primal, not merely smaller. `A`'s diagonal
/// starts at `lambda` and only grows, so every pivot is bounded below by
/// `sqrt(lambda)` for *every* observation sequence — the primal's
/// near-duplicate-feature floor (`rem.max(lambda * 1e-6)`) has no analogue
/// here and is not needed.
#[derive(Debug, Clone)]
pub struct DualPosteriorBuffer<const D: usize> {
    /// Lower-triangular Cholesky factor `L` of `A = X^T X + lambda I`.
    chol: [[f32; D]; D],
    /// `X^T y`, kept because this form does not store the observations.
    xty: [f32; D],
    /// `A^-1 X^T y` — refreshed on append so the mean is an `O(D)` dot product.
    weights: [f32; D],
    lambda: f32,
    n: usize,
}

impl<const D: usize> DualPosteriorBuffer<D> {
    /// Empty posterior with ridge `lambda`.
    ///
    /// # Panics
    ///
    /// If `lambda` is not finite and `> 0`: a non-positive ridge makes `A`
    /// singular at `n = 0`, and rejecting it here beats returning infinities
    /// from the first query.
    #[must_use]
    pub fn new(lambda: f32) -> Self {
        assert!(lambda.is_finite() && lambda > 0.0, "lambda must be finite and > 0");
        let root = lambda.sqrt();
        let mut chol = [[0.0f32; D]; D];
        for (i, row) in chol.iter_mut().enumerate() {
            row[i] = root;
        }
        Self { chol, xty: [0.0; D], weights: [0.0; D], lambda, n: 0 }
    }

    /// Observations absorbed. There is no cap.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// `true` when nothing has been absorbed.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The ridge in force.
    #[inline]
    #[must_use]
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Drop every observation, keeping the ridge. Allocation-free.
    pub fn clear(&mut self) {
        *self = Self::new(self.lambda);
    }

    /// Absorb one observation, `O(D^2)` with no re-factorisation.
    ///
    /// Returns `false` — having changed nothing — when any coordinate or the
    /// label is non-finite. That is not defensive noise: one NaN in the factor
    /// poisons it permanently, so every *later* query about an unrelated
    /// direction would return NaN too.
    pub fn append_observation(&mut self, feat: &[f32; D], y: f32) -> bool {
        if !y.is_finite() || !feat.iter().all(|v| v.is_finite()) {
            return false;
        }
        Self::rank1_update(&mut self.chol, feat);
        for (b, v) in self.xty.iter_mut().zip(feat.iter()) {
            *b += y * v;
        }
        self.n += 1;
        self.refresh_weights();
        true
    }

    /// Eq 10 in dual form: `sigma^2(x) = lambda * x^T (X^T X + lambda I)^-1 x`.
    ///
    /// `O(D^2)`, no allocation, **no dependence on `n`**. At `n = 0` this is
    /// `||x||^2`, matching the primal's `k(x, x)`.
    #[must_use]
    pub fn posterior_variance_linear(&self, x: &[f32; D]) -> f32 {
        let mut v = *x;
        Self::solve_lower(&self.chol, &mut v);
        let quad: f32 = v.iter().map(|t| t * t).sum();
        (self.lambda * quad).max(0.0)
    }

    /// `sigma(x)` — the acquisition score [`should_advance`] consumes.
    #[inline]
    #[must_use]
    pub fn sigma(&self, x: &[f32; D]) -> f32 {
        self.posterior_variance_linear(x).sqrt()
    }

    /// Ridge posterior mean `x^T (X^T X + lambda I)^-1 X^T y`, `O(D)` off the
    /// cached weights. Identical to the primal's `k(x,X)(K + lambda I)^-1 y`.
    #[must_use]
    pub fn ridge_mean(&self, x: &[f32; D]) -> f32 {
        self.weights.iter().zip(x.iter()).map(|(w, v)| w * v).sum()
    }

    fn refresh_weights(&mut self) {
        self.weights = self.xty;
        Self::solve_lower(&self.chol, &mut self.weights);
        Self::solve_lower_transposed(&self.chol, &mut self.weights);
    }

    /// `A += x x^T` in place on the Cholesky factor (Golub & Van Loan
    /// `cholupdate`), `O(D^2)` with no re-factorisation.
    fn rank1_update(chol: &mut [[f32; D]; D], x: &[f32; D]) {
        let mut w = *x;
        for k in 0..D {
            let lkk = chol[k][k];
            let r = lkk.hypot(w[k]);
            // `lkk >= sqrt(lambda) > 0` in every reachable state (the diagonal
            // of `A` only grows), so neither ratio can divide by zero.
            let c = r / lkk;
            let s = w[k] / lkk;
            chol[k][k] = r;
            for i in (k + 1)..D {
                let updated = (chol[i][k] + s * w[i]) / c;
                chol[i][k] = updated;
                w[i] = c * w[i] - s * updated;
            }
        }
    }

    /// Solve `L v = rhs` in place (forward substitution).
    fn solve_lower(chol: &[[f32; D]; D], out: &mut [f32; D]) {
        for i in 0..D {
            let mut acc = out[i];
            for j in 0..i {
                acc -= chol[i][j] * out[j];
            }
            out[i] = acc / chol[i][i];
        }
    }

    /// Solve `L^T v = rhs` in place (back substitution).
    fn solve_lower_transposed(chol: &[[f32; D]; D], out: &mut [f32; D]) {
        for i in (0..D).rev() {
            let mut acc = out[i];
            for j in (i + 1)..D {
                acc -= chol[j][i] * out[j];
            }
            out[i] = acc / chol[i][i];
        }
    }
}

impl<const D: usize> LinearPosterior<D> for DualPosteriorBuffer<D> {
    #[inline]
    fn append_observation(&mut self, feat: &[f32; D], y: f32) -> bool {
        DualPosteriorBuffer::append_observation(self, feat, y)
    }
    /// `scratch` is ignored — the dual query is `O(D^2)` in fixed state.
    #[inline]
    fn posterior_variance_linear(&self, x: &[f32; D], _scratch: &mut [f32]) -> f32 {
        DualPosteriorBuffer::posterior_variance_linear(self, x)
    }
    #[inline]
    fn ridge_mean(&self, x: &[f32; D]) -> f32 {
        DualPosteriorBuffer::ridge_mean(self, x)
    }
    #[inline]
    fn len(&self) -> usize {
        self.n
    }
    #[inline]
    fn scratch_len(&self) -> usize {
        0
    }
}

/// Which factorisation to construct, given the observation count you expect to
/// reach and the feature dimension.
///
/// `true` -> [`DualPosteriorBuffer`], `false` -> [`PosteriorBuffer`]. The rule
/// is the crossover of the two costs (`O(n^2)` vs `O(D^2)` per query, `n^2` vs
/// `D^2` state), so it is exactly `expected_obs > d`.
///
/// Take the decision **once, at construction**, off the count you expect to
/// *reach* — not the count you currently hold. Switching mid-run would mean
/// carrying both factors, which gives back the memory the dual exists to save.
#[inline]
#[must_use]
pub const fn prefer_dual(expected_obs: usize, d: usize) -> bool {
    expected_obs > d
}

// ── the frontier (plan functions 4, 5, 6 + the type) ───────────────────────

/// Fixed-capacity certified cell set. Zero-allocation by construction.
#[derive(Debug, Clone)]
pub struct CertifiedFrontier<const MAX_CELLS: usize, const D: usize> {
    cells: [FrontierCell<D>; MAX_CELLS],
    /// Pre-hop `cb` snapshot, so one `reachability_dilation` pass is exactly
    /// one hop and cannot chain through cells certified within the same pass.
    hop_cb: [f32; MAX_CELLS],
    /// Acquisition lane, held struct-of-arrays: `sigma` for a candidate cell,
    /// `-1.0` for a non-candidate.
    ///
    /// `acquire_frontier_target` runs on every query and needs four fields out
    /// of a ~56-byte cell, so scanning `cells` streams ~57 KiB through L1 to
    /// read ~5 KiB of live data. This lane is one contiguous `f32` array and
    /// turns acquisition into a branch-free argmax. Every mutation that can
    /// change a cell's candidacy or sigma refreshes it through
    /// [`Self::touch_acquisition`]; the correctness suite's
    /// `acquisition_lane_matches_a_full_rescan` pins every step of a run
    /// against a reference argmax over [`Self::cells`].
    acq_sigma: [f32; MAX_CELLS],
    /// Cells whose tally changed since the last `expand_certified`, and the
    /// flag that keeps the list deduplicated.
    dirty: [u32; MAX_CELLS],
    dirty_flag: [bool; MAX_CELLS],
    dirty_len: usize,
    /// Largest `beta` seen by `expand_certified`. Guards the incremental path:
    /// a SHRINKING width can raise an untouched cell's LCB, which is the one
    /// case the dirty set would miss.
    last_beta: f32,
    len: usize,
    certified: u32,
    /// Cells certified by a Lipschitz hop rather than by their own tally.
    dilated: u32,
}

impl<const MAX_CELLS: usize, const D: usize> Default for CertifiedFrontier<MAX_CELLS, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_CELLS: usize, const D: usize> CertifiedFrontier<MAX_CELLS, D> {
    /// An empty frontier.
    #[must_use]
    pub fn new() -> Self {
        Self {
            cells: [FrontierCell::default(); MAX_CELLS],
            hop_cb: [0.0; MAX_CELLS],
            acq_sigma: [NOT_A_CANDIDATE; MAX_CELLS],
            dirty: [0; MAX_CELLS],
            dirty_flag: [false; MAX_CELLS],
            dirty_len: 0,
            last_beta: f32::NEG_INFINITY,
            len: 0,
            certified: 0,
            dilated: 0,
        }
    }

    /// Register a cell. Returns its index, or `None` when at capacity.
    pub fn push_cell(&mut self, feat: [f32; D]) -> Option<usize> {
        if self.len >= MAX_CELLS {
            return None;
        }
        let i = self.len;
        self.cells[i] = FrontierCell::new(feat);
        self.len = i + 1;
        Some(i)
    }

    /// Number of registered cells.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when no cell has been registered.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Registered cells, in insertion order.
    #[inline]
    #[must_use]
    pub fn cells(&self) -> &[FrontierCell<D>] {
        &self.cells[..self.len]
    }

    /// Mutable access to one cell — for `lipschitz` / `sigma_override`.
    #[inline]
    pub fn cell_mut(&mut self, i: usize) -> Option<&mut FrontierCell<D>> {
        (i < self.len).then(|| &mut self.cells[i])
    }

    /// How many cells are certified.
    #[inline]
    #[must_use]
    pub fn certified_count(&self) -> u32 {
        self.certified
    }

    /// How many certified cells were admitted by a Lipschitz hop.
    ///
    /// Counted at the moment `cb` crosses `h`. Do **not** re-derive this at
    /// end-state from "certified but never queried": a frontier policy hands
    /// maximum posterior sigma to a freshly dilated cell and queries it moments
    /// later, so the end-state count reads `0` even where dilation did 27% of
    /// the work (Bench 687 T0.3).
    #[inline]
    #[must_use]
    pub fn dilated_count(&self) -> u32 {
        self.dilated
    }

    /// Queue cell `i` for the next `expand_certified`. Idempotent.
    #[inline]
    fn mark_dirty(&mut self, i: usize) {
        if !self.dirty_flag[i] {
            self.dirty_flag[i] = true;
            self.dirty[self.dirty_len] = i as u32;
            self.dirty_len += 1;
        }
    }

    /// Raise one cell's bound and certify it if it crossed `h`. Returns 1 when
    /// this call is what certified it.
    #[inline]
    fn expand_one(&mut self, i: usize, cfg: &FrontierConfig, beta: f32) -> u32 {
        let lcb = self.lcb(i, beta);
        let c = &mut self.cells[i];
        if lcb > c.cb {
            c.cb = lcb;
        }
        if c.certified || c.cb < cfg.h {
            return 0;
        }
        c.certified = true;
        self.touch_acquisition(i);
        self.mark_neighborhood(i, cfg);
        1
    }

    /// Refresh cell `i`'s acquisition lane from its current candidacy + sigma.
    #[inline]
    fn touch_acquisition(&mut self, i: usize) {
        let c = &self.cells[i];
        self.acq_sigma[i] = if c.certified || c.near_certified { if c.sigma_override.is_finite() { c.sigma_override } else { c.beta_sigma } } else { NOT_A_CANDIDATE };
    }

    /// Stamp `near_certified` on everything within `acquire_radius` of `i`.
    ///
    /// Called once per certification, which is what moves the neighbourhood
    /// scan off the per-query path.
    fn mark_neighborhood(&mut self, i: usize, cfg: &FrontierConfig) {
        let r2 = cfg.acquire_radius * cfg.acquire_radius;
        if r2 <= 0.0 {
            return;
        }
        let center = self.cells[i].feat;
        for j in 0..self.len {
            if !self.cells[j].near_certified && sq_dist(&center, &self.cells[j].feat) <= r2 {
                self.cells[j].near_certified = true;
                self.touch_acquisition(j);
            }
        }
    }

    /// Recompute every `near_certified` flag from scratch.
    ///
    /// Only needed when `cfg.acquire_radius` changes mid-run — the flags are
    /// maintained incrementally otherwise. `O(cells * certified)`.
    pub fn rebuild_neighborhoods(&mut self, cfg: &FrontierConfig) {
        for j in 0..self.len {
            self.cells[j].near_certified = false;
        }
        for i in 0..self.len {
            if self.cells[i].certified {
                self.mark_neighborhood(i, cfg);
            }
        }
        for j in 0..self.len {
            self.touch_acquisition(j);
        }
    }

    /// Seed a cell as certified from caller-side a-priori knowledge.
    ///
    /// Sets `cb` to `h` — the weakest bound consistent with "known valid", so
    /// a seed buys no free dilation headroom it did not earn.
    pub fn seed_certified(&mut self, i: usize, cfg: &FrontierConfig) -> bool {
        if i >= self.len || self.cells[i].certified {
            return false;
        }
        self.cells[i].cb = self.cells[i].cb.max(cfg.h);
        self.cells[i].certified = true;
        self.certified += 1;
        self.touch_acquisition(i);
        self.mark_neighborhood(i, cfg);
        true
    }

    /// Record one binary verifier outcome against a cell.
    pub fn observe(&mut self, i: usize, valid: bool) -> bool {
        if i >= self.len {
            return false;
        }
        let c = &mut self.cells[i];
        if valid { c.valid += 1 } else { c.invalid += 1 }
        c.beta_sigma = beta_mean_variance(c.valid, c.invalid).1.sqrt();
        self.mark_dirty(i);
        self.touch_acquisition(i);
        true
    }

    /// Posterior sd used for this cell: `sigma_override` when finite, else the
    /// Beta-Bernoulli sd.
    #[inline]
    #[must_use]
    pub fn sigma(&self, i: usize) -> f32 {
        let c = &self.cells[i];
        if c.sigma_override.is_finite() { c.sigma_override } else { c.beta_sigma }
    }

    /// Lower confidence bound `mu - beta * sigma`, clamped to `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn lcb(&self, i: usize, beta: f32) -> f32 {
        let c = &self.cells[i];
        let (mean, _) = beta_mean_variance(c.valid, c.invalid);
        (mean - beta * self.sigma(i)).clamp(0.0, 1.0)
    }

    /// Upper confidence bound `mu + beta * sigma`, clamped to `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn ucb(&self, i: usize, beta: f32) -> f32 {
        let c = &self.cells[i];
        let (mean, _) = beta_mean_variance(c.valid, c.invalid);
        (mean + beta * self.sigma(i)).clamp(0.0, 1.0)
    }

    /// Refresh every cell's `sigma_override` from a kernel posterior.
    ///
    /// Opt-in: the default path is the Beta sd, which is what Phase 0 gated on.
    pub fn refresh_kernel_sigma<const MAX_OBS: usize>(
        &mut self,
        buf: &PosteriorBuffer<MAX_OBS, D>,
        scratch: &mut [f32],
    ) {
        for i in 0..self.len {
            let feat = self.cells[i].feat;
            self.cells[i].sigma_override = buf.posterior_variance_linear(&feat, scratch).sqrt();
            // A new sigma changes every cell's LCB, so the dirty set is no
            // longer a sufficient work list.
            self.mark_dirty(i);
            self.touch_acquisition(i);
        }
    }

    /// **Eq 32 — the certified-set update.** Raise every `cb` to its LCB and
    /// certify whatever crosses `h`. Returns the number newly certified.
    ///
    /// `cb` moves by `max`, so the certified set is monotone across *any*
    /// query sequence (T2.3). Soundness rests on `beta` covering every round,
    /// which is what [`confidence_schedule`]'s monotonicity in `t` buys.
    pub fn expand_certified(&mut self, cfg: &FrontierConfig, beta: f32) -> u32 {
        // `cb` moves by max, and an untouched cell's LCB can only have FALLEN
        // if `beta` grew. So when the width is non-decreasing, only cells whose
        // tally changed can raise a bound — everything else is a no-op that
        // still costs a divide and a square root. Scanning the dirty set
        // instead turns the pass from O(cells) into O(observed since last
        // call), which for the one-observation-per-round shape is O(1).
        //
        // The one case that breaks: a SHRINKING beta raises every untouched
        // cell's LCB. Both shipped widths are non-decreasing
        // (`confidence_schedule` is monotone in `t`, `beta_union_bound` is
        // constant for fixed inputs), but a caller may pass anything, so detect
        // it and fall back rather than silently under-certifying.
        if beta < self.last_beta { self.expand_certified_full(cfg, beta) } else {
                self.last_beta = beta;
                let mut newly = 0;
                for k in 0..self.dirty_len {
                    let i = self.dirty[k] as usize;
                    self.dirty_flag[i] = false;
                    newly += self.expand_one(i, cfg, beta);
                }
                self.dirty_len = 0;
                self.certified += newly;
                newly
            }
    }

    /// [`Self::expand_certified`] over every cell, unconditionally — the
    /// reference path.
    ///
    /// `expand_certified` dispatches to this automatically when `beta` shrinks.
    /// Call it directly only after mutating cell state behind the type's back
    /// (there is no such path today) or to cross-check the incremental result;
    /// the correctness suite pins the two against each other over a full run.
    pub fn expand_certified_full(&mut self, cfg: &FrontierConfig, beta: f32) -> u32 {
        self.last_beta = self.last_beta.max(beta);
        let mut newly = 0;
        for i in 0..self.len {
            self.dirty_flag[i] = false;
            newly += self.expand_one(i, cfg, beta);
        }
        self.dirty_len = 0;
        self.certified += newly;
        newly
    }

    /// Effective Lipschitz cost of the hop `i -> j`: `max(L_i, L_j)`, falling
    /// back to `cfg.lipschitz` for any cell without a local bound.
    #[inline]
    fn hop_lipschitz(&self, i: usize, j: usize, cfg: &FrontierConfig) -> f32 {
        let li = if self.cells[i].lipschitz.is_finite() { self.cells[i].lipschitz } else { cfg.lipschitz };
        let lj = if self.cells[j].lipschitz.is_finite() { self.cells[j].lipschitz } else { cfg.lipschitz };
        li.max(lj)
    }

    /// **The T0.3 predicate.** Can a hop be afforded at all right now?
    ///
    /// Cheap (`O(n)`) and meant to be called before every dilation: a coarse
    /// lattice makes [`Self::reachability_dilation`] a silent no-op, and the
    /// return value of that call cannot distinguish "nothing left to admit"
    /// from "nothing was ever affordable".
    ///
    /// `feasible` is **necessary, not sufficient**: it prices the single best
    /// headroom against one representative lattice hop, so `!feasible`
    /// guarantees a dilation admits nothing, while `feasible` only means some
    /// hop is affordable *if* an uncertified cell sits that close to the cell
    /// holding the headroom.
    #[must_use]
    pub fn dilation_feasibility(&self, cfg: &FrontierConfig) -> DilationFeasibility {
        let mut best = f32::NEG_INFINITY;
        let mut min_l = f32::INFINITY;
        for i in 0..self.len {
            if self.cells[i].certified {
                best = best.max(self.cells[i].cb - cfg.h);
                let li = if self.cells[i].lipschitz.is_finite() { self.cells[i].lipschitz } else { cfg.lipschitz };
                min_l = min_l.min(li);
            }
        }
        let l = if min_l.is_finite() { min_l } else { cfg.lipschitz };
        let hop_cost = l * cfg.cell_spacing;
        DilationFeasibility {
            best_headroom: best,
            hop_cost,
            feasible: best >= hop_cost,
            deficit: hop_cost - best,
        }
    }

    /// **Eq 15 — one Lipschitz reachability hop per iteration.**
    ///
    /// Admits `z` when some certified `z'` has `cb(z') - L d(z, z') >= h`. The
    /// relaxed bound `cb(z') - L d` is written into `cb(z)` by `max`, so
    /// dilation is monotone exactly like [`Self::expand_certified`].
    ///
    /// `hop_budget` passes are run; each pass reads a pre-hop snapshot, so a
    /// cell certified in pass `k` can only extend the set in pass `k + 1`.
    /// Returns the number newly certified across all passes.
    ///
    /// Cost is `O(hop_budget * certified * uncertified * D)`. Call
    /// [`Self::dilation_feasibility`] first — on a coarse lattice this whole
    /// loop is guaranteed to admit nothing.
    pub fn reachability_dilation(&mut self, cfg: &FrontierConfig, hop_budget: u32) -> u32 {
        let mut newly = 0;
        for _ in 0..hop_budget {
            for j in 0..self.len {
                self.hop_cb[j] = self.cells[j].cb;
            }
            let mut admitted = 0;
            for j in 0..self.len {
                if self.cells[j].certified {
                    continue;
                }
                let mut best = self.hop_cb[j];
                for i in 0..self.len {
                    if !self.cells[i].certified {
                        continue;
                    }
                    let d = sq_dist(&self.cells[i].feat, &self.cells[j].feat).sqrt();
                    let cand = self.hop_cb[i] - self.hop_lipschitz(i, j, cfg) * d;
                    if cand > best {
                        best = cand;
                    }
                }
                if best > self.cells[j].cb {
                    self.cells[j].cb = best;
                }
                if best >= cfg.h {
                    self.cells[j].certified = true;
                    self.cells[j].by_dilation = true;
                    admitted += 1;
                    self.touch_acquisition(j);
                    self.mark_neighborhood(j, cfg);
                }
            }
            self.certified += admitted;
            self.dilated += admitted;
            newly += admitted;
            if admitted == 0 {
                break;
            }
        }
        newly
    }

    /// **Eq 33 — safe uncertainty sampling.** `argmax sigma` over certified
    /// cells and cells within `cfg.acquire_radius` of one.
    ///
    /// Ties break to the lowest index, so a fixed cell order gives a
    /// deterministic query sequence. `cfg.acquire_radius = 0.0` restricts the
    /// search to the certified set itself (the strict Eq-33 reading); the
    /// wider default is the policy that measured 51.4x in Phase 0, because
    /// restricting to certified cells makes growth depend entirely on a
    /// dilation that a coarse lattice cannot afford.
    ///
    /// `O(cells)` and branch-free — a single argmax over the contiguous
    /// acquisition lane. Candidacy is stamped once per certification rather
    /// than rescanned per query. `cfg` is accepted for signature stability and
    /// is unused; the radius is baked into the cached flags, so changing it
    /// mid-run requires [`Self::rebuild_neighborhoods`].
    ///
    /// `cfg.alpha` scales the returned cell's sigma threshold only through
    /// [`should_advance`]; acquisition itself is scale-free.
    #[must_use]
    pub fn acquire_frontier_target(&self, _cfg: &FrontierConfig) -> Option<usize> {
        let lane = &self.acq_sigma[..self.len];
        // Two branch-free passes beat one scalar argmax: tracking the index
        // inline creates a loop-carried dependency on an unpredictable branch,
        // which is what pinned this at ~1 ns/cell. Pass 1 is an 8-wide max
        // reduction with no dependency between lanes; pass 2 short-circuits.
        let mut acc = [NOT_A_CANDIDATE; 8];
        let (chunks, remainder) = lane.as_chunks::<8>();
        for ch in chunks {
            for (a, &v) in acc.iter_mut().zip(ch.iter()) {
                *a = a.max(v);
            }
        }
        let mut best = remainder.iter().fold(NOT_A_CANDIDATE, |a, &b| a.max(b));
        for &a in &acc {
            best = best.max(a);
        }
        if best <= NOT_A_CANDIDATE {
            return None;
        }
        // Exact equality is sound here: `best` is copied straight out of the
        // lane, never computed from it, and the lane holds no NaN. Ties go to
        // the lowest index, as documented.
        lane.iter().position(|&s| s == best)
    }

    /// Straddling gate: is querying this cell decision-relevant at all?
    ///
    /// `true` only when the threshold lies inside the cell's confidence band
    /// after paying for one hop — deep-inside and far-outside cells prune to
    /// zero queries. The EVPI-shaped companion to acquisition (Plan 580 T4.2).
    #[must_use]
    pub fn query_is_decision_relevant(&self, i: usize, cfg: &FrontierConfig, beta: f32) -> bool {
        if i >= self.len {
            return false;
        }
        let lcb = self.lcb(i, beta);
        let ucb = self.ucb(i, beta);
        let l = if self.cells[i].lipschitz.is_finite() { self.cells[i].lipschitz } else { cfg.lipschitz };
        lcb - l * cfg.cell_spacing < cfg.h && cfg.h <= ucb
    }
}

// ── Plan 580 T4.1 — fusion with the Viable Manifold Graph ──────────────────

/// The grow-then-navigate join: a [`SafeManifoldGraph`] built from a certified
/// frontier, plus the mapping back.
///
/// Only compiled when both features are on.
#[cfg(feature = "viable_manifold_graph")]
#[derive(Debug)]
pub struct CertifiedManifoldGraph {
    /// The navigable graph. Node ids are **graph-local** and dense.
    pub graph: crate::viable_manifold_graph::SafeManifoldGraph,
    /// `node_to_cell[node_id]` is the frontier cell index that node came from.
    ///
    /// Without this the composition is unusable: the builder drops cells, so
    /// graph ids and cell indices diverge, and a caller that navigates to a
    /// node cannot ask what its certified bound was.
    pub node_to_cell: Vec<u32>,
    /// Certified cells rejected by the pullback-volume threshold — certified
    /// but not navigable. A large count means the two criteria disagree, which
    /// is information, not an error.
    pub rejected_by_volume: usize,
}

/// Build a navigable graph over the **certified** cells only (Plan 580 T4.1).
///
/// This is the fusion the primitive exists for: [`CertifiedFrontier`] answers
/// *which latent cells are provably valid* and
/// [`crate::viable_manifold_graph`] answers *how to move between them without
/// leaving the viable set*. Growth supplies the nodes that navigation was
/// previously missing.
///
/// # Two filters, deliberately both
///
/// A cell becomes a node iff it is **certified** (`p(z) >= h`, from the
/// verifier) **and** its pullback volume is under `build_cfg.volume_threshold`
/// (the decoder is well-conditioned there). These are different questions —
/// validity versus navigability — and a cell can pass one and fail the other.
/// Both filters are applied here rather than inside the builder, which is what
/// makes an exact `node_to_cell` mapping possible: the builder is handed a
/// pre-filtered sample set with the threshold already satisfied.
///
/// # Cost
///
/// Allocates (`Vec` samples + the graph itself) and evaluates one Jacobian SVD
/// per certified cell. This is a **build-time** operation — the zero-alloc
/// guarantee covers the query path (`acquire`/`observe`/`expand`), not this.
///
/// `f` must be `Copy` because both the volume field and the builder consume it;
/// pass `&closure` if the closure itself is not `Copy`.
#[cfg(feature = "viable_manifold_graph")]
pub fn certified_manifold_graph<const MAX_CELLS: usize, const D: usize, F>(
    frontier: &CertifiedFrontier<MAX_CELLS, D>,
    f: F,
    volume_cfg: &crate::viable_manifold_graph::VolumeFieldConfig,
    build_cfg: &crate::viable_manifold_graph::GraphBuildConfig,
    scratch: &mut crate::subspace_phase_gate::JacobianSvdScratch,
) -> CertifiedManifoldGraph
where
    F: Fn(&[f32], &mut [f32]) + Copy,
{
    use crate::viable_manifold_graph::{
        ClosurePredicate, GraphBuildConfig, build_safe_manifold_graph, pullback_volume,
    };

    let mut node_to_cell = Vec::new();
    let mut samples = Vec::new();
    let mut rejected_by_volume = 0;
    for (i, cell) in frontier.cells().iter().enumerate() {
        if !cell.certified {
            continue;
        }
        if pullback_volume(f, &cell.feat, scratch, volume_cfg) > build_cfg.volume_threshold {
            rejected_by_volume += 1;
            continue;
        }
        node_to_cell.push(i as u32);
        samples.extend_from_slice(&cell.feat);
    }

    // Both filters already ran, so the builder must keep everything it is
    // handed — otherwise `node_to_cell` would silently misalign.
    let keep_all = GraphBuildConfig {
        volume_threshold: f32::INFINITY,
        ..*build_cfg
    };
    let always_viable = ClosurePredicate(|_: &[f32]| true);
    let graph = build_safe_manifold_graph(
        f,
        &samples,
        D,
        &always_viable,
        volume_cfg,
        &keep_all,
        scratch,
    );
    debug_assert_eq!(
        graph.n_nodes(),
        node_to_cell.len(),
        "builder dropped a pre-filtered node — node_to_cell would misalign"
    );
    CertifiedManifoldGraph {
        graph,
        node_to_cell,
        rejected_by_volume,
    }
}
