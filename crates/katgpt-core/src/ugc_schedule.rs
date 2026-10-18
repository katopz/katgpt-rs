//! UGC — Unmasking Growth Complexity certified schedules for masked diffusion.
//!
//! Substrate distilled from arXiv:2608.13520 (Wainwright, "The data geometry of
//! masking diffusion: Certified-optimal schedules via unmasking growth
//! complexity"), via Research 485 / Issue 664.
//!
//! Pure modelless math: Monte Carlo over a frozen denoiser's own posteriors,
//! closed-form schedule construction, and truncated empirical-Bernstein
//! certification. No training and no gradient steps anywhere.
//!
//! # The paper in one paragraph
//!
//! For a reveal process where coordinate `i` of `Z ∈ A^d` is revealed by time
//! `t` iff `U_i ≤ t` (shared iid uniforms), the Bernoulli unmasking gain
//! `h(t) = Σᵢ Info(Zᵢ; X_t | i masked)` has curvature `h′`, and the UGC mass
//! `H(p,q) = ∫_p^q t(1−t) h′(t) dt` is **additive** along the reveal path
//! (Eq 2c/3). A discretized unmasking sampler on grid `t_0 < … < t_N` has
//! `KL(P_Z ‖ P_Ẑ) ≤ Σⱼ (ψ(t_{j+1})/ψ(t_j) − 1)·H(t_j, t_{j+1}) + init +
//! completion` where `ψ(t) = t/(1−t)` are the reveal odds (Theorem 1). The
//! natural path coordinate is log-reveal-odds `λ = log ψ(t)`; the UGC density
//! `q(λ) = r²(1−r)² h′(r)` (with `r = σ(λ)`) localizes sampling difficulty.
//! Geometry-aware schedules place **equal √q-mass per step** (Theorem 3), and
//! the increments `H` are **estimable from samples** via KL increments along
//! coupled forced-mask reveal trajectories (Lemma 1 + Proposition 2), yielding
//! certified-optimal samplers: `KL ≤ 4Ĉ/N + init + completion` w.p. ≥ 1−η, and
//! `N ≥ 8Ĉ/ε` certifies a prescribed KL error ε (Theorem 2 + Eq 38).
//!
//! # Module layout
//!
//! - [`UgcDenoiser`] — exact single-site posterior abstraction (the frozen
//!   denoiser). Toy ensembles + the d2f transformer adapter implement it.
//! - [`estimate_interval`] — T1: the paper's dyadic-odds truncated
//!   empirical-Bernstein estimator of one `H(p,q)` increment (Eq 32–34).
//! - [`estimate_profile`] — coupled h-curve estimator on a uniform-in-λ
//!   grid; the input for schedule construction + paper-number reproduction.
//! - [`equal_sqrt_mass_grid`] / [`dp_partition`] / [`certified_block_plan`] /
//!   [`certified_iteration_count`] — T3 schedule construction.
//! - [`bernoulli_unmask_with_grid`] — the paper's Bernoulli unmasking sampler
//!   (Eq 11) with exact zero-cost init + completion; used by the G1-cert
//!   coverage gate to measure real KL. Optionally records the per-position
//!   reveal step (Plan 602 T2.2's `steps_out`) for the anchor view below.
//! - [`CertifiedSpineView`] — Plan 602 T2.2's certified-spine anchor view
//!   (feature `decode_order_metrics`): the certified early reveals as the
//!   paper's sparse anchor set A, the remainder as the conditional chain.
//!
//! # Honest caveats (binding, from Research 485 §3)
//!
//! 1. The KL certificate covers **random-order** Bernoulli/fixed-cardinality
//!    reveal, NOT confidence-threshold (greedy per-token) reveal. The
//!    estimator itself is policy-agnostic (it measures data geometry).
//! 2. The moment constant `B_α` in the Bernstein radius is assumed known by
//!    the paper; this implementation estimates it from the same trajectories
//!    (aggregate Minkowski bound — see `estimate_interval`) — a documented
//!    pragmatic deviation whose soundness is validated empirically by the
//!    coverage gate (coverage measured, never asserted).
//! 3. CCL25 (arXiv:2511.04647) proves you cannot compete with the optimal
//!    schedule without a-priori distribution knowledge; UGC competes within
//!    constant factors with high probability via estimation. Cite both
//!    whenever "certified-optimal" is claimed.
//!
//! Allocation discipline: `estimate_interval` and `bernoulli_unmask_with_grid`
//! are fully [`UgcScratch`]-backed (zero heap allocation after construction —
//! the audited hot paths). `estimate_profile` is the amortized once-per-model
//! constructor; its grid vectors are returned outputs. API is `f32` in/out
//! with `f64` internal accumulators.
//!
//! UGC — Unmasking Growth Complexity certified schedules for masked diffusion
//! (arXiv:2608.13520, Research 485 / Issue 664). Always-on: pure math + a
//! denoiser trait, zero deps beyond crate::types::Rng — the same no-gate
//! rationale as set_diffusion_schedule. Feature-flag plan (`ugc_schedule`)
//! opens ONLY if the Issue 664 G1b promotion gate passes.
//!
//! Plan 602 T2.2: the certified-spine anchor view over UGC reveal
//! trajectories (same feature; defined in ugc_schedule.rs per the plan
//! target — ugc_schedule itself stays always-on, only this view is gated).

use crate::types::Rng;

// Plan 602 T2.2: the anchor view is Plan-602 instrumentation and rides that
// feature (this module itself stays always-on — only the view is gated).
#[cfg(feature = "decode_order_metrics")]
use crate::dllm::UNMASKED_NEVER;

/// Sentinel marking a masked (unrevealed) coordinate in observation vectors.
pub const UGC_MASK: usize = usize::MAX;

/// Numerical floor for probability entries inside KL evaluations.
const KL_EPS: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Reveal-odds helpers
// ---------------------------------------------------------------------------

/// Reveal odds `ψ(t) = t/(1−t)`.
#[inline]
pub fn reveal_odds(t: f32) -> f32 {
    t / (1.0 - t)
}

/// Log-reveal-odds `λ(t) = log(t/(1−t))` — the paper's natural path clock.
#[inline]
pub fn log_reveal_odds(t: f32) -> f32 {
    let c = t.clamp(1e-9, 1.0 - 1e-9);
    (c / (1.0 - c)).ln()
}

/// Inverse log-reveal-odds `r = σ(λ) = e^λ/(1+e^λ)` (numerically stable).
/// Delegates to the Bench-844 substrate (Issue 861) — expression-identical
/// body, bit-identical by construction.
#[inline]
pub fn inv_log_reveal_odds(lambda: f32) -> f32 {
    crate::exact_sigmoid(lambda)
}

// ---------------------------------------------------------------------------
// Denoiser abstraction
// ---------------------------------------------------------------------------

/// Exact single-site posterior denoiser for a discrete distribution.
///
/// `posterior_into(i, x, out)` writes `P(Z_i = a | Z_j = x_j for all revealed
/// j ≠ i)` — paper Eq 8 — for masked coordinate `i`, into `out` (length =
/// `alphabet()`). Unrevealed entries of `x` carry [`UGC_MASK`].
///
/// Toy ensembles (repeated-bit / parity / mixtures) implement this with exact
/// analytic posteriors; the d2f consumer implements it with one
/// `forward_block_causal_with` pass + softmax row.
pub trait UgcDenoiser {
    /// Ambient dimension `d`.
    fn dim(&self) -> usize;
    /// Alphabet size `|A|` (symbols `0..alphabet`).
    fn alphabet(&self) -> usize;
    /// Write the posterior for masked coordinate `i` given partial
    /// observation `x` (masked entries = [`UGC_MASK`]) into `out`.
    fn posterior_into(&self, i: usize, x: &[usize], out: &mut [f32]);
}

/// KL(a ‖ b) for two discrete distributions, floored for stability.
#[inline]
fn kl_discrete(a: &[f32], b: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        if av > 0.0 {
            s += av as f64 * ((av as f64) / (bv as f64 + KL_EPS)).ln();
        }
    }
    s
}

/// Sample one symbol from a categorical vector (f64 accumulation).
#[inline]
fn sample_categorical(probs: &[f32], rng: &mut Rng) -> usize {
    let cu = rng.uniform() as f64;
    let mut acc = 0.0f64;
    for (a, &pr) in probs.iter().enumerate() {
        acc += pr as f64;
        if cu <= acc {
            return a;
        }
    }
    probs.len() - 1
}

// ---------------------------------------------------------------------------
// Scratch
// ---------------------------------------------------------------------------

/// Pre-allocated scratch for the zero-alloc estimation/sampling paths.
///
/// Allocate once per (d, alphabet, m) triple; reused across calls. The
/// reveal `order` buffer permanently holds the identity permutation `0..d`
/// and is only ever re-sorted — membership never changes.
pub struct UgcScratch {
    /// Grid times (dyadic), ascending. Capacity must cover `max_grid`.
    pub grid: Vec<f32>,
    /// Rolling posterior buffer (length = alphabet).
    post_cur: Vec<f32>,
    /// Observation buffer (length = d).
    obs: Vec<usize>,
    /// Reveal-order indices sorted per trajectory by uniform (identity
    /// membership, permuted in place).
    order: Vec<u32>,
    /// Clean-sample buffer (length = d).
    z_buf: Vec<usize>,
    /// Shared-uniform buffer (length = d).
    u_buf: Vec<f32>,
    /// Per-sample trajectory statistics Q(ℓ) (length = m).
    q_stats: Vec<f64>,
    /// Truncated statistics 2·min(Q, τ) (length = m).
    trunc: Vec<f64>,
    /// Per-coordinate posterior rows, rolling (d × alphabet).
    coord_post: Vec<f32>,
    /// Per-coordinate prior marginals (d × alphabet).
    prior_rows: Vec<f32>,
    /// Per-grid-point h accumulators for the profile walk (g+1).
    h_sum: Vec<f64>,
    /// Per-interval Δh accumulators for the profile walk (g).
    /// Masked-coordinate index buffer for the sampler steps.
    masked_buf: Vec<usize>,
}

impl UgcScratch {
    /// Size scratch for dimension `d`, alphabet `a`, `m` trajectory samples,
    /// and up to `max_grid` grid points (profile grids use `g ≤ max_grid`).
    pub fn new(d: usize, alphabet: usize, m: usize, max_grid: usize) -> Self {
        Self {
            grid: Vec::with_capacity(max_grid + 2),
            post_cur: vec![0.0; alphabet],
            obs: vec![UGC_MASK; d],
            order: (0..d as u32).collect(),
            z_buf: vec![0; d],
            u_buf: vec![0.0; d],
            q_stats: Vec::with_capacity(m),
            trunc: Vec::with_capacity(m),
            coord_post: vec![0.0; d * alphabet],
            prior_rows: vec![0.0; d * alphabet],
            h_sum: vec![0.0; max_grid + 1],
            masked_buf: Vec::with_capacity(d),
        }
    }

    #[inline]
    fn reset_obs(&mut self) {
        self.obs.fill(UGC_MASK);
    }
}

// ---------------------------------------------------------------------------
// T1 — dyadic-odds interval estimator (paper §4.3, Eq 32–34)
// ---------------------------------------------------------------------------

/// Build the reveal-odds-dyadic grid `ψ(v_j) = min(2^j·ψ(p), ψ(q))`
/// (Eq 32a) into `scratch.grid` (ascending times `v_0 = p … v_J = q`).
///
/// Each interval has reveal-odds ratio exactly 2 (except the last, ≤ 2), so
/// Lemma 1's sandwich is a factor exactly 2 per interval.
pub fn dyadic_odds_grid(p: f32, q: f32, scratch: &mut UgcScratch) {
    scratch.grid.clear();
    scratch.grid.push(p);
    let psi_p = reveal_odds(p) as f64;
    let psi_q = reveal_odds(q) as f64;
    let j_max = (psi_q / psi_p).log2().ceil().max(1.0) as i32;
    for j in 1..=j_max {
        let psi = (2.0f64.powi(j) * psi_p).min(psi_q);
        let v = (psi / (1.0 + psi)) as f32;
        if v > p && v < q {
            scratch.grid.push(v);
        }
        if psi >= psi_q {
            break;
        }
    }
    scratch.grid.push(q);
}

/// Result of one interval estimate: the truncated statistic + radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UgcIntervalEstimate {
    /// Truncated mean `Ĥ_m(p,q) = (2/m) Σ min(Q(ℓ), τ)` (Eq 33b).
    pub hat_h: f32,
    /// Empirical-Bernstein radius `r̂_m(η)` (Eq 34b).
    pub r_hat: f32,
    /// The paper's sandwich upper bound: `H(p,q) ≤ Ĥ_m + r̂_m` w.p. ≥ 1−η.
    pub upper: f32,
    /// Empirical variance `V̂` (Eq 33c) — reported, not consumed.
    pub var: f32,
}

/// One coupled forced-mask trajectory walk over `scratch.grid`
/// (Eq 32b): `Q = Σᵢ Σⱼ (v_{j+1}−v_j)·KL(π_{i,v_{j+1}} ‖ π_{i,v_j})`.
///
/// One shared uniform vector `u` couples all grid points (paper §4.3.1);
/// coordinate `i` is force-masked at every grid time (Eq 30a). Zero-alloc.
fn trajectory_q(dz: &dyn UgcDenoiser, z: &[usize], u: &[f32], scratch: &mut UgcScratch) -> f64 {
    let d = dz.dim();
    let a = dz.alphabet();
    let UgcScratch {
        grid,
        post_cur,
        obs,
        order,
        coord_post,
        ..
    } = scratch;

    let mut q_total = 0.0f64;
    order.sort_by(|&x, &y| u[x as usize].total_cmp(&u[y as usize]));
    obs.fill(UGC_MASK);
    let mut oi = 0usize;

    for g in 0..grid.len() {
        let t = grid[g];
        while oi < order.len() && u[order[oi] as usize] <= t {
            let idx = order[oi] as usize;
            obs[idx] = z[idx];
            oi += 1;
        }
        for i in 0..d {
            let saved = obs[i];
            obs[i] = UGC_MASK;
            dz.posterior_into(i, obs, post_cur);
            if g > 0 {
                // KL(π_new ‖ π_old) — later time first (Eq 30b orientation).
                let prev = &coord_post[i * a..(i + 1) * a];
                let kl = kl_discrete(post_cur, prev);
                q_total += (grid[g] - grid[g - 1]) as f64 * kl;
            }
            coord_post[i * a..(i + 1) * a].copy_from_slice(post_cur);
            obs[i] = saved;
        }
    }
    q_total
}

/// Debug-observability gate for `estimate_interval` (Issue 861).
///
/// Reads `UGC_DEBUG` ONCE via `OnceLock` — the `tpr::kill_switch` pattern.
/// The per-call `std::env::var` this replaces was debug-profile-only but
/// heap-allocated on every `estimate_interval` call on Windows/MSVC (the
/// env-var NAME is converted to a UTF-16 `Vec<u16>` for the W-series Win32
/// API): exactly 1 allocation per call, invisible on Unix where `getenv`
/// does not allocate — so the Issue-664 G4 PASS held on macOS/aarch64 while
/// `ugc_alloc_check` failed 50/50 on this Windows box. Steady-state after
/// the one-time read is a plain atomic load, zero-allocation everywhere.
#[cfg(debug_assertions)]
fn ugc_debug_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("UGC_DEBUG").is_ok())
}

/// T1 — the paper's tail-robust dyadic interval estimate (Prop 2, Eq 33/34).
///
/// `eta` ∈ (0,1) is the failure probability. `m` ≥ 2 samples. Moment order
/// α fixed at 4 (r = α/2 = 2, the weakest order the theorem supports).
/// `B_α` is estimated from the same trajectories via the aggregate
/// Minkowski bound `B̂ = (mean Q^{r})^{1/r}` (each per-interval moment is
/// dominated by the aggregate — documented deviation from the known-`B_α`
/// assumption; validated empirically by the coverage gate).
pub fn estimate_interval(
    dz: &dyn UgcDenoiser,
    p: f32,
    q: f32,
    m: usize,
    eta: f32,
    rng: &mut Rng,
    scratch: &mut UgcScratch,
) -> UgcIntervalEstimate {
    assert!(m >= 2, "m >= 2 required for empirical variance");
    assert!(0.0 < p && p < q && q < 1.0, "need 0 < p < q < 1");
    // Moment order α fixed at 4 (r = α/2 = 2, the weakest order the
    // theorem supports) — the 2/α and 1−2/α exponents below are 1/2.

    dyadic_odds_grid(p, q, scratch);

    scratch.q_stats.clear();
    for _ in 0..m {
        fill_z_buf(dz, rng, scratch);
        for ui in scratch.u_buf.iter_mut() {
            *ui = rng.uniform();
        }
        let z = std::mem::take(&mut scratch.z_buf);
        let u = std::mem::take(&mut scratch.u_buf);
        let q_stat = trajectory_q(dz, &z, &u, scratch);
        scratch.z_buf = z;
        scratch.u_buf = u;
        scratch.q_stats.push(q_stat);
    }

    // τ (Eq 33b) + Ĥ_m + V̂ (Eq 33c) + r̂_m (Eq 34b).
    let log_term = (4.0 / eta as f64).ln();
    let moment_denom = 3.0 * (m - 1) as f64;
    let mean_q_r = scratch.q_stats.iter().map(|q| q * q).sum::<f64>() / m as f64;
    let b_alpha = mean_q_r.sqrt().max(1e-9); // r = α/2 = 2
    let tau = (q - p) as f64 * b_alpha * (7.0 * log_term / moment_denom).sqrt();
    #[cfg(debug_assertions)]
    if ugc_debug_enabled() {
        let raw_mean = scratch.q_stats.iter().sum::<f64>() / m as f64;
        let raw_max = scratch.q_stats.iter().cloned().fold(0.0f64, f64::max);
        eprintln!(
            "UGC interval [{p:.3},{q:.3}] m={m}: raw Q̄={raw_mean:.5} max={raw_max:.5} B̂={b_alpha:.5} τ={tau:.5}"
        );
    }

    let hat_h = 2.0 * scratch.q_stats.iter().map(|q| q.min(tau)).sum::<f64>() / m as f64;
    scratch.trunc.clear();
    for &q in scratch.q_stats.iter() {
        scratch.trunc.push(2.0 * q.min(tau));
    }
    let var = scratch
        .trunc
        .iter()
        .map(|y| {
            let dd = y - hat_h;
            dd * dd
        })
        .sum::<f64>()
        / (m - 1) as f64;
    let r_hat = (2.0 * var * log_term / m as f64).sqrt()
        + 4.0 * (q - p) as f64 * b_alpha * (7.0 * log_term / moment_denom).sqrt();

    UgcIntervalEstimate {
        hat_h: hat_h as f32,
        r_hat: r_hat as f32,
        upper: (hat_h + r_hat) as f32,
        var: var as f32,
    }
}

/// Fill `scratch.z_buf` with a clean sample `z ~ P_Z` by sequential exact
/// conditional sampling (chain-rule factorization through the denoiser —
/// the zero-defect completion kernel of paper §3.1.3). Zero-alloc.
fn fill_z_buf(dz: &dyn UgcDenoiser, rng: &mut Rng, scratch: &mut UgcScratch) {
    let d = dz.dim();
    scratch.reset_obs();
    for i in 0..d {
        let UgcScratch { obs, post_cur, .. } = scratch;
        dz.posterior_into(i, obs, post_cur);
        let chosen = sample_categorical(post_cur, rng);
        scratch.obs[i] = chosen;
    }
    scratch.z_buf.copy_from_slice(&scratch.obs);
}

/// Draw a clean sample `z ~ P_Z` into `out` (convenience wrapper over the
/// scratch-backed path; `out.len() == dz.dim()`).
pub fn sample_z(dz: &dyn UgcDenoiser, rng: &mut Rng, out: &mut [usize], scratch: &mut UgcScratch) {
    fill_z_buf(dz, rng, scratch);
    out.copy_from_slice(&scratch.z_buf);
}

// ---------------------------------------------------------------------------
// Profile estimator — coupled h-curve on a uniform-in-λ grid
// ---------------------------------------------------------------------------

/// Estimated UGC profile over `[p, q]` on a uniform-in-λ grid.
///
/// `h` values are coupled across grid points via shared uniforms (per
/// trajectory). Per-interval masses use the **integration-by-parts form**
/// `H(p,q) = [t(1−t)h]ₚ^q + ∫ₚ^q (2t−1)h(t)dt` evaluated by the trapezoid
/// rule on the coupled ĥ curve — this form is *identically zero for any
/// constant h*, so per-trajectory constant offsets (the dominant noise in
/// the flat regions) cancel exactly. Naive midpoint-weighting
/// `t_m(1−t_m)·Δh` does not have this property and its correlated noise
/// inflates `C_UGC` by ~70% at m=48 (measured, Issue 664 debug).
#[derive(Debug, Clone)]
pub struct UgcProfile {
    /// Grid times `t_g` (ascending, `G+1` entries: endpoints included).
    pub t_grid: Vec<f32>,
    /// Log-reveal-odds of the grid times (`G+1` entries).
    pub lambda_grid: Vec<f32>,
    /// Coupled h-curve estimates at each grid time (`G+1` entries).
    pub h: Vec<f32>,
    /// Per-interval mean density `ΔH_g/Δλ_g` (`G` entries).
    pub q_density: Vec<f32>,
    /// Per-interval UGC mass estimates `ΔH_g` (`G` entries).
    pub increments: Vec<f32>,
}

impl UgcProfile {
    /// Estimated UGC mass over grid intervals `[a, b)`.
    pub fn mass(&self, a: usize, b: usize) -> f32 {
        let lo = a.min(self.increments.len());
        let hi = b.min(self.increments.len());
        self.increments[lo..hi].iter().sum()
    }
    /// Log-reveal-odds length over grid indices `a..=b`.
    pub fn lambda_len(&self, a: usize, b: usize) -> f32 {
        self.lambda_grid[b] - self.lambda_grid[a]
    }
    /// Coarse single-block complexity `C_UGC = (λ_range)·H(p,q)` (Eq 5a:
    /// on the canonical interval the prefactor `2ℓ_d` IS the full log-odds
    /// range `ln(ψ(T)/ψ(t0)) = 2·ln(d−1)`, matching Corollary 1's Geo(ρ)
    /// complexity `H·log(odds-ratio)`).
    pub fn coarse_complexity(&self) -> f32 {
        let total: f32 = self.increments.iter().sum();
        let ell =
            self.lambda_grid.last().unwrap_or(&0.0) - self.lambda_grid.first().unwrap_or(&0.0);
        ell * total
    }
    /// Fine-partition complexity `P_UGC = (Σ √(ΔH·Δλ))²` (Eq 5b + Theorem 3:
    /// converges to `(∫√q dλ)²` as the grid refines).
    pub fn fine_partition_complexity(&self) -> f32 {
        let s: f32 = self
            .increments
            .iter()
            .zip(self.lambda_grid.windows(2).map(|w| w[1] - w[0]))
            .map(|(dh, dl)| (dh.max(0.0) * dl).sqrt())
            .sum();
        s * s
    }
    /// Potential gain `C_UGC / P_UGC` (Eq 6) — ≥ 1 by Cauchy–Schwarz.
    pub fn ratio(&self) -> f32 {
        let fine = self.fine_partition_complexity();
        if fine <= 0.0 {
            return 1.0;
        }
        self.coarse_complexity() / fine
    }
}

/// One profile-walk trajectory: accumulate `KL(π_{i,t} ‖ prior_i)` per grid
/// point into `h_sum`. The per-interval increments are taken as **coupled
/// h-curve differences** `Δĥ_g = ĥ(t_{g+1}) − ĥ(t_g)` — the shared-uniform
/// coupling (common random numbers) makes the difference estimator
/// dramatically lower-variance than the raw consecutive-KL statistic
/// (heavy-tailed: the transition fires in one interval per trajectory).
/// Both are unbiased for `h(t_{g+1}) − h(t_g)` by Lemma 1 / Eq 2a.
/// Zero-alloc.
fn profile_walk(
    dz: &dyn UgcDenoiser,
    z: &[usize],
    u: &[f32],
    t_grid: &[f32],
    scratch: &mut UgcScratch,
) {
    let d = dz.dim();
    let a = dz.alphabet();
    let UgcScratch {
        post_cur,
        obs,
        order,
        prior_rows,
        h_sum,
        ..
    } = scratch;

    order.sort_by(|&x, &y| u[x as usize].total_cmp(&u[y as usize]));
    obs.fill(UGC_MASK);
    let mut oi = 0usize;
    for gi in 0..t_grid.len() {
        let t = t_grid[gi];
        while oi < order.len() && u[order[oi] as usize] <= t {
            let idx = order[oi] as usize;
            obs[idx] = z[idx];
            oi += 1;
        }
        for i in 0..d {
            let saved = obs[i];
            obs[i] = UGC_MASK;
            dz.posterior_into(i, obs, post_cur);
            h_sum[gi] += kl_discrete(post_cur, &prior_rows[i * a..(i + 1) * a]);
            obs[i] = saved;
        }
    }
}

/// Estimate the UGC profile on a uniform-in-λ grid over `[p, q]` with `g`
/// intervals (`g ≥ 1`), using `m` coupled trajectories.
///
/// Unbiased h estimates: `ĥ(t) = (1/m) Σ_ℓ Σᵢ KL(π_{i,t} ‖ prior_i)` with
/// **per-coordinate** priors (Info(Z_i; X_t | i masked) as an expected
/// posterior KL — Eq 2a; per-coordinate marginals keep the transformer
/// adapter honest when position marginals differ). Amortized once-per-model
/// constructor; grid vectors are returned outputs.
pub fn estimate_profile(
    dz: &dyn UgcDenoiser,
    p: f32,
    q: f32,
    g: usize,
    m: usize,
    rng: &mut Rng,
    scratch: &mut UgcScratch,
) -> UgcProfile {
    assert!(g >= 1, "need at least one grid interval");
    assert!(0.0 < p && p < q && q < 1.0);
    assert!(scratch.h_sum.len() > g, "scratch max_grid too small");
    let d = dz.dim();
    let a = dz.alphabet();
    let lam_p = log_reveal_odds(p) as f64;
    let lam_q = log_reveal_odds(q) as f64;

    let mut t_grid = Vec::with_capacity(g + 1);
    let mut lambda_grid = Vec::with_capacity(g + 1);
    for j in 0..=g {
        let lam = lam_p + (lam_q - lam_p) * j as f64 / g as f64;
        lambda_grid.push(lam as f32);
        t_grid.push(inv_log_reveal_odds(lam as f32));
    }
    t_grid[0] = p;
    t_grid[g] = q;

    // Per-coordinate prior marginals (posterior with nothing revealed).
    scratch.reset_obs();
    for i in 0..d {
        let UgcScratch {
            obs,
            post_cur,
            prior_rows,
            ..
        } = scratch;
        dz.posterior_into(i, obs, post_cur);
        prior_rows[i * a..(i + 1) * a].copy_from_slice(post_cur);
    }

    scratch.h_sum.clear();
    scratch.h_sum.resize(g + 1, 0.0);

    for _ in 0..m {
        fill_z_buf(dz, rng, scratch);
        for ui in scratch.u_buf.iter_mut() {
            *ui = rng.uniform();
        }
        let z = std::mem::take(&mut scratch.z_buf);
        let u = std::mem::take(&mut scratch.u_buf);
        profile_walk(dz, &z, &u, &t_grid, scratch);
        scratch.z_buf = z;
        scratch.u_buf = u;
    }

    let h: Vec<f32> = scratch
        .h_sum
        .iter()
        .map(|&v| (v / m as f64) as f32)
        .collect();
    // Per-interval masses via integration by parts on the coupled ĥ curve:
    // ΔH_g = [t(1−t)h]_{t_g}^{t_{g+1}} + ∫ (2t−1)h dt (trapezoid).
    let increments: Vec<f32> = (0..g)
        .map(|gi| {
            let (ta, tb) = (t_grid[gi], t_grid[gi + 1]);
            let dt = tb - ta;
            let tm = 0.5 * (ta + tb);
            let hm = 0.5 * (h[gi] + h[gi + 1]);
            tb * (1.0 - tb) * h[gi + 1] - ta * (1.0 - ta) * h[gi] + (2.0 * tm - 1.0) * dt * hm
        })
        .collect();
    let q_density: Vec<f32> = (0..g)
        .map(|gi| {
            let dl = (lambda_grid[gi + 1] - lambda_grid[gi]).max(1e-12);
            increments[gi] / dl
        })
        .collect();

    UgcProfile {
        t_grid,
        lambda_grid,
        h,
        q_density,
        increments,
    }
}

// ---------------------------------------------------------------------------
// T3 — schedule construction
// ---------------------------------------------------------------------------

/// Equal-√q-mass N-step reveal grid (Theorem 3's optimal grid restricted to
/// the profile's interval). Returns `n+1` ascending times in
/// `[t_grid[0], t_grid[G]]` such that each step carries ≈ 1/N of the total
/// `∫√q dλ` mass.
///
/// Generalizes `ScheduleKind::EquiProbability` (equal-CDF-mass under a fixed
/// prior) to the estimated data-geometry density.
pub fn equal_sqrt_mass_grid(profile: &UgcProfile, n: usize) -> Vec<f32> {
    assert!(n >= 1);
    let g = profile.increments.len();
    let sqrt_mass: Vec<f64> = (0..g)
        .map(|i| {
            let dl = (profile.lambda_grid[i + 1] - profile.lambda_grid[i]) as f64;
            ((profile.increments[i].max(0.0) as f64) * dl).sqrt()
        })
        .collect();
    let total: f64 = sqrt_mass.iter().sum();
    let mut out = Vec::with_capacity(n + 1);
    out.push(profile.t_grid[0]);
    if total <= 0.0 {
        // Degenerate (flat) geometry: uniform-in-λ fallback.
        for k in 1..n {
            let lam = profile.lambda_grid[0]
                + (profile.lambda_grid[g] - profile.lambda_grid[0]) * k as f32 / n as f32;
            out.push(inv_log_reveal_odds(lam));
        }
    } else {
        let mut acc = 0.0f64;
        let mut k_target = 1usize;
        for (i, &sm_i) in sqrt_mass.iter().enumerate() {
            let dl = (profile.lambda_grid[i + 1] - profile.lambda_grid[i]) as f64;
            let dens = sm_i / dl; // √q within the interval
            let mut lam_lo = profile.lambda_grid[i] as f64;
            while k_target < n && acc + sm_i >= total * k_target as f64 / n as f64 {
                let need = total * k_target as f64 / n as f64 - acc;
                let lam_cut = lam_lo + need / dens;
                out.push(inv_log_reveal_odds(lam_cut as f32));
                lam_lo = lam_cut;
                k_target += 1;
            }
            acc += sm_i;
        }
    }
    out.push(profile.t_grid[g]);
    out.shrink_to_fit();
    out
}

/// DP K-block boundary selection over the profile grid (paper §4.4.2,
/// Eq 39): minimize `Σ_k √(S_k·H_k)` with edge cost
/// `e(i,j) = √((λ_j−λ_i)·mass(i..j))`. Returns `k+1` grid indices
/// (`0` and `G` included; ascending).
///
/// DRY cross-link: `crate::partition::minmax_partition` (feature
/// `minmax_partition`) is this DP's NEIGHBOR, not its generalization —
/// min-MAX worst-case under an ε constraint, k free, over a generic
/// distance matrix (Plan 616; the T0.1 decision and the
/// never-drift-silently rule live in the module doc).
pub fn dp_partition(profile: &UgcProfile, k: usize) -> Vec<usize> {
    const INF: f64 = f64::INFINITY;

    let g = profile.increments.len();
    assert!(k >= 1 && k <= g, "need 1 <= K <= number of intervals");
    let mut prefix = vec![0.0f64; g + 1];
    for i in 0..g {
        prefix[i + 1] = prefix[i] + profile.increments[i].max(0.0) as f64;
    }
    let edge = |i: usize, j: usize| -> f64 {
        let s = (profile.lambda_grid[j] - profile.lambda_grid[i]) as f64;
        (s * (prefix[j] - prefix[i]).max(0.0)).sqrt()
    };
    let mut v = vec![vec![INF; g + 1]; k + 1];
    let mut arg = vec![vec![0usize; g + 1]; k + 1];
    v[1][0] = 0.0;
    for (j, v1_j) in v[1].iter_mut().enumerate().skip(1) {
        *v1_j = edge(0, j);
    }
    for kb in 2..=k {
        for j in kb..=g {
            let mut best = INF;
            let mut bi = 0;
            for (i, &v_prev) in v[kb - 1].iter().enumerate().take(j).skip(kb - 1) {
                let c = v_prev + edge(i, j);
                if c < best {
                    best = c;
                    bi = i;
                }
            }
            v[kb][j] = best;
            arg[kb][j] = bi;
        }
    }
    let mut idx = vec![0usize; k + 1];
    idx[k] = g;
    let mut j = g;
    let mut kb = k;
    while kb > 1 {
        let i = arg[kb][j];
        idx[kb - 1] = i;
        j = i;
        kb -= 1;
    }
    idx[0] = 0;
    idx
}

/// A certified K-block Geo(ρ) plan (Prop 1 / Theorem 2).
#[derive(Debug, Clone)]
pub struct UgcBlockPlan {
    /// Block boundary times `b_0 < … < b_K` (canonical `[t_0, T]`).
    pub boundaries: Vec<f32>,
    /// Block log-odds lengths `S_k` (K entries).
    pub s_k: Vec<f32>,
    /// Block upper estimates `Ĥ_k + r̂_k` (K entries).
    pub upper_k: Vec<f32>,
    /// Surrogate partition complexity `Ĉ_UGC(P) = (Σ √(S_k·Û_k))²` (Eq 36b).
    pub chat_partition_complexity: f32,
    /// Geometric multipliers `ρ̂_k = min(1, 4√(Ĉ/N)·√(S_k/Û_k))` (Eq 37a).
    pub multipliers: Vec<f32>,
    /// Steps per block `N_k = ⌈S_k / log(1+ρ̂_k)⌉` (K entries).
    pub steps_per_block: Vec<usize>,
    /// Total steps Σ N_k.
    pub total_steps: usize,
}

/// Build the certified K-block plan from block boundaries and their upper
/// estimates `(Ĥ_k + r̂_k)`, given an iteration budget `N`
/// (Theorem 2, Eq 36–37). `boundaries.len() == uppers.len() + 1 ≥ 2`.
pub fn certified_block_plan(boundaries: &[f32], uppers: &[f32], n_budget: usize) -> UgcBlockPlan {
    assert!(boundaries.len() >= 2);
    assert_eq!(boundaries.len(), uppers.len() + 1);
    let k = uppers.len();
    let s_k: Vec<f32> = (0..k)
        .map(|i| log_reveal_odds(boundaries[i + 1]) - log_reveal_odds(boundaries[i]))
        .collect();
    let sqrt_terms: Vec<f64> = (0..k)
        .map(|i| ((s_k[i] as f64) * (uppers[i].max(0.0) as f64)).sqrt())
        .collect();
    let chat_sqrt: f64 = sqrt_terms.iter().sum();
    let chat = chat_sqrt * chat_sqrt;
    let multipliers: Vec<f32> = (0..k)
        .map(|i| {
            let rho = 4.0
                * (chat / n_budget as f64).sqrt()
                * (s_k[i] as f64 / (uppers[i].max(1e-12) as f64)).sqrt();
            rho.min(1.0) as f32
        })
        .collect();
    let steps_per_block: Vec<usize> = (0..k)
        .map(|i| {
            let ln_1p = (1.0 + multipliers[i] as f64).ln();
            if ln_1p <= 0.0 {
                1
            } else {
                ((s_k[i] as f64) / ln_1p).ceil().max(1.0) as usize
            }
        })
        .collect();
    let steps_total: usize = steps_per_block.iter().sum();
    UgcBlockPlan {
        boundaries: boundaries.to_vec(),
        s_k,
        upper_k: uppers.to_vec(),
        chat_partition_complexity: chat as f32,
        multipliers,
        steps_per_block,
        total_steps: steps_total,
    }
}

/// Expand a block plan into the full reveal-time grid: within block `k`,
/// `ψ(t_{j+1}) = min((1+ρ̂_k)·ψ(t_j), ψ(b_{k+1}))` (Eq 20).
pub fn reveal_grid_from_plan(plan: &UgcBlockPlan) -> Vec<f32> {
    let mut grid = Vec::with_capacity(plan.upper_k.len());
    grid.push(plan.boundaries[0]);
    for k in 0..plan.upper_k.len() {
        let rho = plan.multipliers[k];
        let psi_end = reveal_odds(plan.boundaries[k + 1]) as f64;
        let mut psi = reveal_odds(plan.boundaries[k]) as f64;
        for _ in 0..plan.steps_per_block[k] {
            let psi_next = ((1.0 + rho as f64) * psi).min(psi_end);
            let t = (psi_next / (1.0 + psi_next)) as f32;
            if t > *grid.last().unwrap() {
                grid.push(t);
            }
            psi = psi_next;
            if psi >= psi_end {
                break;
            }
        }
        let t_end = plan.boundaries[k + 1];
        if *grid.last().unwrap() < t_end {
            grid.push(t_end);
        }
    }
    grid
}

/// Certified iteration count for a target KL error `ε`:
/// `N ≥ 8·Ĉ/ε` (Eq 38; with init+completion ≤ ε/2 gives total ≤ ε).
pub fn certified_iteration_count(chat: f32, eps: f32) -> usize {
    assert!(eps > 0.0);
    ((8.0 * chat as f64 / eps as f64).ceil() as usize).max(1)
}

// ---------------------------------------------------------------------------
// Bernoulli unmasking sampler (paper Eq 11) — for certificate measurement
// ---------------------------------------------------------------------------

/// Run the paper's Bernoulli unmasking sampler on `grid` (ascending times in
/// (0,1)): each masked coordinate is revealed per step with probability
/// `β_j = (t_{j+1}−t_j)/(1−t_j)` and filled from the exact posterior
/// (Eq 11a/11b). Initialization at `grid[0]` and completion at `grid[last]`
/// use sequential exact conditional sampling, so both boundary KL costs are
/// exactly zero (paper §3.1.3's zero-defect kernels).
///
/// Writes the fully-revealed sample into `out[..d]`.
///
/// `steps_out` (Plan 602 T2.2, the T1.3 π posture) optionally records the
/// commit step of each position: 0 for the init kernel, `j+1` for transition
/// `j` (`grid[j] → grid[j+1]`), `grid.len()` for the completion fill — the
/// same convention as `dllm::UNMASKED_NEVER` trajectories (never-committed
/// stays `u32::MAX`; completion always fills here, so the sentinel only
/// appears if the caller truncates the grid). Pass `None` when not
/// measuring; the recording costs one branch per reveal.
pub fn bernoulli_unmask_with_grid(
    dz: &dyn UgcDenoiser,
    grid: &[f32],
    rng: &mut Rng,
    scratch: &mut UgcScratch,
    out: &mut [usize],
    mut steps_out: Option<&mut [u32]>,
) {
    let d = dz.dim();
    debug_assert!(grid.len() >= 2 && grid.windows(2).all(|w| w[0] < w[1]));
    debug_assert!(steps_out.as_deref().is_none_or(|s| s.len() >= d));

    if let Some(steps) = steps_out.as_deref_mut() {
        steps[..d].fill(u32::MAX);
    }

    // Init at t_0: reveal each coord w.p. t_0, fill revealed coords by
    // sequential exact conditionals (== the reveal-process law at t_0).
    scratch.reset_obs();
    for i in 0..d {
        if rng.uniform() < grid[0] {
            let UgcScratch { obs, post_cur, .. } = scratch;
            dz.posterior_into(i, obs, post_cur);
            let chosen = sample_categorical(post_cur, rng);
            scratch.obs[i] = chosen;
            if let Some(steps) = steps_out.as_deref_mut() {
                steps[i] = 0;
            }
        }
    }

    for (j, w) in grid.windows(2).enumerate() {
        let (tj, tj1) = (w[0], w[1]);
        let beta = ((tj1 - tj) / (1.0 - tj)) as f64;
        if beta <= 0.0 {
            continue;
        }
        scratch.masked_buf.clear();
        scratch
            .masked_buf
            .extend((0..d).filter(|&i| scratch.obs[i] == UGC_MASK));
        let masked = std::mem::take(&mut scratch.masked_buf);
        for &i in &masked {
            if (rng.uniform() as f64) < beta {
                let UgcScratch { obs, post_cur, .. } = scratch;
                dz.posterior_into(i, obs, post_cur);
                let chosen = sample_categorical(post_cur, rng);
                scratch.obs[i] = chosen;
                if let Some(steps) = steps_out.as_deref_mut() {
                    steps[i] = j as u32 + 1;
                }
            }
        }
        scratch.masked_buf = masked;
    }

    // Completion: sequential exact conditional fill of remaining masked.
    for i in 0..d {
        if scratch.obs[i] == UGC_MASK {
            let UgcScratch { obs, post_cur, .. } = scratch;
            dz.posterior_into(i, obs, post_cur);
            let chosen = sample_categorical(post_cur, rng);
            scratch.obs[i] = chosen;
            if let Some(steps) = steps_out.as_deref_mut() {
                steps[i] = grid.len() as u32;
            }
        }
    }
    out[..d].copy_from_slice(&scratch.obs);
}

// ---------------------------------------------------------------------------
// Plan 602 T2.2 — certified-spine anchor view (feature `decode_order_metrics`)
// ---------------------------------------------------------------------------
//
// Research 575 §distilled-3 maps the dQwen3.5 paper's sparse anchor set A
// onto this module's certified reveal process: positions the certified
// kernels reveal EARLY are the high-confidence spine — they carry future
// context for everything revealed after them, and the remainder decodes as
// the conditional chain `q(x) = q(x_A) · Π_{i∉A} q(x_i | x_<i, x_A)`
// (arXiv:2609.20751 §5). A VIEW, not new math: that identity is the chain
// rule applied over this view's ordering (anchors as one joint block, then
// the chain ascending), and its exactness is inherited from the sampler's
// exact-conditional kernels — the GOAT arm pins it by enumeration on a
// synthetic joint (see `certified_spine_factorization_identity`).

/// Certified-spine anchor view over one reveal trajectory (Plan 602 T2.2).
///
/// `reveal_steps` is the per-position commit step recorded by
/// [`bernoulli_unmask_with_grid`]'s `steps_out` (or any decode path using
/// the same convention — the `dllm` π logs share it). Positions with step
/// `< anchor_cutoff` form the anchor set A; the committed remainder is the
/// conditional chain. [`CertifiedSpineView::init_spine`] is the canonical
/// cutoff: the init kernel's reveals, the certified high-confidence masks.
#[cfg(feature = "decode_order_metrics")]
#[derive(Debug, Clone, Copy)]
pub struct CertifiedSpineView<'a> {
    reveal_steps: &'a [u32],
    cutoff: u32,
}

#[cfg(feature = "decode_order_metrics")]
impl<'a> CertifiedSpineView<'a> {
    /// Split at `anchor_cutoff`: `π[i] < cutoff` ⇒ anchor.
    pub fn new(reveal_steps: &'a [u32], anchor_cutoff: u32) -> Self {
        Self {
            reveal_steps,
            cutoff: anchor_cutoff,
        }
    }

    /// The canonical spine: anchors = positions the init kernel revealed
    /// (step 0) — the certified high-confidence masks of Research 575 §3.
    pub fn init_spine(reveal_steps: &'a [u32]) -> Self {
        Self::new(reveal_steps, 1)
    }

    /// Anchor membership. Total: sentinel (never-committed) and
    /// out-of-range positions are never anchors.
    pub fn is_anchor(&self, i: usize) -> bool {
        matches!(
            self.reveal_steps.get(i),
            Some(&s) if s != UNMASKED_NEVER && s < self.cutoff
        )
    }

    /// The sparse anchor set A, ascending positions.
    pub fn anchor_positions(&self) -> Vec<usize> {
        (0..self.reveal_steps.len())
            .filter(|&i| self.is_anchor(i))
            .collect()
    }

    /// The conditional-chain remainder: committed positions that are not
    /// anchors, ascending — the chain decodes left-to-right off A.
    pub fn chain_positions(&self) -> Vec<usize> {
        (0..self.reveal_steps.len())
            .filter(|&i| {
                matches!(self.reveal_steps.get(i), Some(&s) if s != UNMASKED_NEVER)
                    && !self.is_anchor(i)
            })
            .collect()
    }

    /// Sparsity `|A| / committed`. NaN when nothing committed (the arness
    /// degenerate-input contract: never a silent ratio).
    pub fn anchor_density(&self) -> f32 {
        let committed = self
            .reveal_steps
            .iter()
            .filter(|&&s| s != UNMASKED_NEVER)
            .count();
        if committed == 0 {
            f32::NAN
        } else {
            self.anchor_positions().len() as f32 / committed as f32
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests (core math; heavy paper-number gates live in tests/ugc_664_poc.rs)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent-coordinates denoiser: posterior = prior everywhere →
    /// every KL increment is exactly 0 (paper §4.3.2's basic example).
    struct Independent;

    impl UgcDenoiser for Independent {
        fn dim(&self) -> usize {
            4
        }
        fn alphabet(&self) -> usize {
            3
        }
        fn posterior_into(&self, _i: usize, _x: &[usize], out: &mut [f32]) {
            for v in out.iter_mut() {
                *v = 1.0 / 3.0;
            }
        }
    }

    #[test]
    fn odds_helpers_roundtrip() {
        for &t in &[0.05f32, 0.2, 0.5, 0.8, 0.95] {
            let lam = log_reveal_odds(t);
            let back = inv_log_reveal_odds(lam);
            assert!((back - t).abs() < 1e-5, "roundtrip {t} -> {lam} -> {back}");
        }
        assert!(log_reveal_odds(0.5).abs() < 1e-6);
        assert!((reveal_odds(0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dyadic_grid_covers_interval_with_factor_two_odds() {
        let mut s = UgcScratch::new(4, 2, 4, 64);
        dyadic_odds_grid(0.125, 0.875, &mut s);
        assert_eq!(s.grid.first(), Some(&0.125));
        assert_eq!(s.grid.last(), Some(&0.875));
        assert!(s.grid.windows(2).all(|w| w[0] < w[1]));
        for w in s.grid.windows(2) {
            let ratio = reveal_odds(w[1]) / reveal_odds(w[0]);
            assert!(ratio <= 2.0 + 1e-4, "odds ratio {ratio} > 2");
            assert!(ratio > 1.0);
        }
        // ψ(7/8)/ψ(1/8) = 49 → J = ceil(log2 49) = 6 → 7 grid points.
        assert_eq!(s.grid.len(), 7);
    }

    #[test]
    fn independent_coordinates_have_zero_ugc() {
        let mut rng = Rng::new(7);
        let mut s = UgcScratch::new(4, 3, 16, 64);
        let est = estimate_interval(&Independent, 0.1, 0.9, 8, 0.1, &mut rng, &mut s);
        assert!(est.hat_h.abs() < 1e-6);
        assert!(est.r_hat.abs() < 1e-6);
        let prof = estimate_profile(&Independent, 0.1, 0.9, 8, 8, &mut rng, &mut s);
        assert!(prof.h.iter().all(|&v| v.abs() < 1e-6));
        assert!(prof.increments.iter().all(|v| v.abs() < 1e-6));
        assert!(prof.coarse_complexity().abs() < 1e-6);
    }

    #[test]
    fn certified_iteration_count_formula() {
        assert_eq!(certified_iteration_count(10.0, 1.0), 80);
        assert_eq!(certified_iteration_count(0.0, 1.0), 1);
        assert_eq!(certified_iteration_count(1.0, 3.0), 3);
    }

    #[test]
    fn block_plan_multipliers_match_proposition_1() {
        let b = [0.125f32, 0.5, 0.875];
        let up = [1.0f32, 1.0];
        let plan = certified_block_plan(&b, &up, 16);
        let s = 7.0f32.ln();
        let chat = 4.0 * s;
        assert!((plan.chat_partition_complexity - chat).abs() < 1e-3);
        let rho = (4.0 * (chat / 16.0).sqrt() * s.sqrt()).min(1.0);
        assert!((plan.multipliers[0] - rho).abs() < 1e-3);
        assert_eq!(plan.multipliers.len(), 2);
        assert!(plan.total_steps >= 2);
        let grid = reveal_grid_from_plan(&plan);
        assert!(grid.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(*grid.first().unwrap(), 0.125f32);
        assert_eq!(*grid.last().unwrap(), 0.875f32);
    }

    #[test]
    fn bernoulli_sampler_produces_full_samples() {
        // Fully-correlated ensemble: Z = (U, U, U).
        struct Correlated;
        impl UgcDenoiser for Correlated {
            fn dim(&self) -> usize {
                3
            }
            fn alphabet(&self) -> usize {
                2
            }
            fn posterior_into(&self, i: usize, x: &[usize], out: &mut [f32]) {
                let known = (0..3).find(|&j| j != i && x[j] != UGC_MASK);
                match known {
                    Some(j) => {
                        out.fill(0.0);
                        out[x[j]] = 1.0;
                    }
                    None => out.fill(0.5),
                }
            }
        }
        let mut rng = Rng::new(3);
        let mut s = UgcScratch::new(3, 2, 8, 64);
        let mut out = [0usize; 3];
        for _ in 0..32 {
            bernoulli_unmask_with_grid(
                &Correlated,
                &[0.2, 0.5, 0.8],
                &mut rng,
                &mut s,
                &mut out,
                None,
            );
            assert!(out[0] < 2 && out[1] < 2 && out[2] < 2);
            assert!(out[0] == out[1] && out[1] == out[2], "sampled {out:?}");
        }
    }
}

/// Plan 602 T2.2 anchor-view tests (feature-gated with the view).
#[cfg(all(test, feature = "decode_order_metrics"))]
mod anchor_view_tests {
    use super::*;
    use crate::dllm::UNMASKED_NEVER;

    /// Uniform-posterior denoiser (local twin of tests::Independent).
    struct Flat;
    impl UgcDenoiser for Flat {
        fn dim(&self) -> usize {
            4
        }
        fn alphabet(&self) -> usize {
            3
        }
        fn posterior_into(&self, _i: usize, _x: &[usize], out: &mut [f32]) {
            out.fill(1.0 / 3.0);
        }
    }

    #[test]
    fn view_splits_on_cutoff() {
        // steps: pos0 committed late (3), pos1 later (3), pos2 early (1),
        // pos3 never, pos4 init (0), pos5 mid (2).
        let steps = [3u32, 3, 1, UNMASKED_NEVER, 0, 2];
        let v = CertifiedSpineView::new(&steps, 2);
        assert_eq!(v.anchor_positions(), vec![2, 4]);
        assert_eq!(v.chain_positions(), vec![0, 1, 5]);
        assert!(!v.is_anchor(0));
        assert!(v.is_anchor(2));
        assert!(v.is_anchor(4));
        assert!(!v.is_anchor(3)); // sentinel: never an anchor
        // Density over committed (5 of 6 committed): 2/5.
        assert!((v.anchor_density() - 2.0 / 5.0).abs() < 1e-6);
    }

    #[test]
    fn init_spine_selects_step_zero() {
        let steps = [0u32, 0, 5, 0, UNMASKED_NEVER];
        let v = CertifiedSpineView::init_spine(&steps);
        assert_eq!(v.anchor_positions(), vec![0, 1, 3]);
        assert_eq!(v.chain_positions(), vec![2]);
        assert!((v.anchor_density() - 3.0 / 4.0).abs() < 1e-6);
    }

    #[test]
    fn all_sentinel_view_is_empty_and_nan() {
        let steps = [UNMASKED_NEVER; 4];
        let v = CertifiedSpineView::init_spine(&steps);
        assert!(v.anchor_positions().is_empty());
        assert!(v.chain_positions().is_empty());
        assert!(v.anchor_density().is_nan());
    }

    #[test]
    fn out_of_range_is_not_an_anchor() {
        let steps = [0u32, 1];
        let v = CertifiedSpineView::init_spine(&steps);
        assert!(!v.is_anchor(99));
        assert!(!v.is_anchor(2));
    }

    #[test]
    fn steps_out_records_every_commit() {
        let mut rng = Rng::new(5);
        let mut s = UgcScratch::new(4, 3, 16, 64);
        let mut out = [0usize; 4];
        let mut steps = [u32::MAX; 4];
        let grid = [0.5f32, 0.75];
        bernoulli_unmask_with_grid(&Flat, &grid, &mut rng, &mut s, &mut out, Some(&mut steps));
        // Everything commits (completion fills all): init=0, transition=1,
        // completion=grid.len()=2.
        assert!(steps.iter().all(|&st| st <= grid.len() as u32), "{steps:?}");
        let n_init = steps.iter().filter(|&&st| st == 0).count();
        // t_0 = 0.5 over d=4 with a fixed seed pins the init count; require
        // a genuinely sparse split so the assertions below are non-vacuous.
        assert!(
            n_init > 0 && n_init < 4,
            "seed produced |A|={n_init}: {steps:?}"
        );
        let view = CertifiedSpineView::init_spine(&steps);
        assert_eq!(view.anchor_positions().len(), n_init);
        assert_eq!(view.chain_positions().len(), 4 - n_init);
        assert!((view.anchor_density() - n_init as f32 / 4.0).abs() < 1e-6);
    }

    // ── GOAT arm: the factorization identity on a synthetic joint ─────────

    /// An explicit-table joint over d positions — a generic, fully-
    /// correlated law (seeded strictly-positive table, no accidental
    /// structure that could make the identity trivially hold).
    struct TableJoint {
        d: usize,
        alpha: usize,
        table: Vec<f32>, // alpha^d, row-major, digit i of idx left-to-right
    }

    impl TableJoint {
        fn digits(&self, idx: usize) -> Vec<usize> {
            (0..self.d)
                .map(|i| (idx / self.alpha.pow((self.d - 1 - i) as u32)) % self.alpha)
                .collect()
        }

        /// Σ table entries consistent with `fixed` (marginal / partial
        /// conditioning — the test-side oracle, independent of the
        /// denoiser impl).
        fn mass_given(&self, fixed: &[(usize, usize)]) -> f32 {
            let mut m = 0.0;
            for (idx, &p) in self.table.iter().enumerate() {
                let ds = self.digits(idx);
                if fixed.iter().all(|&(pos, val)| ds[pos] == val) {
                    m += p;
                }
            }
            m
        }
    }

    impl UgcDenoiser for TableJoint {
        fn dim(&self) -> usize {
            self.d
        }
        fn alphabet(&self) -> usize {
            self.alpha
        }
        fn posterior_into(&self, i: usize, x: &[usize], out: &mut [f32]) {
            // Exact single-site conditional of the table given revealed coords.
            out.fill(0.0);
            let mut z = 0.0;
            for (idx, &p) in self.table.iter().enumerate() {
                let ds = self.digits(idx);
                let consistent = (0..self.d).all(|j| x[j] == UGC_MASK || x[j] == ds[j]);
                if consistent {
                    out[ds[i]] += p;
                    z += p;
                }
            }
            if z > 0.0 {
                for v in out.iter_mut() {
                    *v /= z;
                }
            }
        }
    }

    #[test]
    fn certified_spine_factorization_identity() {
        // GOAT arm (Plan 602 T2.2): with A = the run's certified spine,
        //   q(x) == q(x_A) · Π_{i∈chain} q(x_i | x_<i, x_A)
        // holds EXACTLY for every x in the support — pinned by enumeration,
        // no sampling noise. The sampler supplies A (via steps_out); the
        // identity itself is the chain rule over the view's ordering.
        let (d, alpha) = (4usize, 2usize);

        // Seeded strictly-positive table (xorshift32 → 0.1 + u², normalized).
        let mut state = 0x602_575u32;
        let mut raw = vec![0.0f32; alpha.pow(d as u32)];
        for r in raw.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let u = state as f32 / u32::MAX as f32;
            *r = 0.1 + u * u;
        }
        let z: f32 = raw.iter().sum();
        let dz = TableJoint {
            d,
            alpha,
            table: raw.iter().map(|&v| v / z).collect(),
        };

        // One certified run with step recording.
        let grid = [0.35f32, 0.6, 0.85];
        let mut rng = Rng::new(11);
        let mut s = UgcScratch::new(d, alpha, 16, 64);
        let mut out = vec![0usize; d];
        let mut steps = vec![u32::MAX; d];
        bernoulli_unmask_with_grid(&dz, &grid, &mut rng, &mut s, &mut out, Some(&mut steps));

        let view = CertifiedSpineView::init_spine(&steps);
        let anchors = view.anchor_positions();
        let chain = view.chain_positions();
        // Non-vacuous: a genuinely sparse spine, everything committed.
        assert!(
            !anchors.is_empty() && anchors.len() < d,
            "seed must produce a sparse spine: steps {steps:?}"
        );
        assert_eq!(anchors.len() + chain.len(), d);

        // Enumerate the whole support: the view's factorization must equal
        // the table entry for EVERY x.
        for idx in 0..alpha.pow(d as u32) {
            let xs = dz.digits(idx);
            // q(x_A) — the joint over the anchor block.
            let fixed_a: Vec<(usize, usize)> = anchors.iter().map(|&a| (a, xs[a])).collect();
            let mut prod = dz.mass_given(&fixed_a);
            // Π q(x_i | anchors + earlier chain), ascending chain order.
            for &i in &chain {
                let mut fixed = fixed_a.clone();
                for j in chain.iter().copied().take_while(|&j| j < i) {
                    fixed.push((j, xs[j]));
                }
                let mut fixed_i = fixed.clone();
                fixed_i.push((i, xs[i]));
                let denom = dz.mass_given(&fixed);
                assert!(denom > 0.0, "zero-probability conditioning at x={xs:?}");
                prod *= dz.mass_given(&fixed_i) / denom;
            }
            assert!(
                (prod - dz.table[idx]).abs() < 1e-4,
                "factorization broke at x={xs:?}: {prod} vs {}",
                dz.table[idx]
            );
        }
    }
}
