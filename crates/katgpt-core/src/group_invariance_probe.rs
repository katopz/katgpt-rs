//! Group Invariance Probe — modelless symmetry discovery on a hypothesis Lie group.
//!
//! Distilled from LieFlow (Chen et al., arXiv:2512.20043, ICML 2026; see
//! `katgpt-rs/.research/355_LieFlow_Symmetry_Discovery_Group_Orbit_Support.md`).
//! The paper's trained flow-matching velocity field `v_θ` redirects to
//! riir-train; what ships here is the **deterministic invariance test** +
//! **support-concentration classifier** — the modelless residue that
//! generalizes [`crate::subspace_phase_gate`] from "subspace of `ℝᵈ`" to
//! "subgroup of a hypothesis group `G`".
//!
//! # The reframe (one sentence)
//!
//! LieFlow discovers a symmetry group `H ⊆ G` by learning a distribution
//! over `G` whose **support concentrates on `H`**: continuous `H` spreads
//! smoothly over a submanifold, discrete `H` peaks sharply at the finite
//! group elements. We replace "learn a distribution via trained flow
//! matching" with "**score each sampled `g ∈ G` by direct invariance
//! testing**, then read `H` off the score histogram's concentration".
//!
//! # What this computes
//!
//! Given:
//! - a hypothesis matrix Lie group `G` (via the [`GroupAction`] trait),
//! - an observation summary `q ∈ ℝᵈ` (mean trajectory, `style_weights`, ...),
//! - a distribution distance `d(q, g·q)` (caller-supplied via a closure),
//! - a sharpness `β` and a support threshold `τ`,
//!
//! the probe computes `invariance_score(g) = σ(β·(1 − d(q, g·q)))` for
//! each sampled `g`, then classifies the discovered subgroup `H`:
//!
//! ```text
//! Discrete   ⟺  score_concentration < τ_disc   (peaked: few g's near 1, rest near 0)
//! Continuous ⟺  score_concentration ≥ τ_cont   (spread: many g's with similar score)
//! Partial    ⟺  mixed (bimodal but no clean peak)
//! None       ⟺  all scores near 0 (no non-trivial invariance)
//! ```
//!
//! # Why this is NOT redundant with [`crate::subspace_phase_gate`]
//!
//! `subspace_phase_gate` answers "is the data concentrated in a *linear
//! subspace* of `ℝᵈ`?" via participation ratio / numerical rank. It cannot
//! tell "uniform on a group orbit" (a 1-D submanifold of `SO(2)`, e.g.
//! uniform on the circle) from "no structure" (i.i.d. Gaussian) — both
//! have participation ratio ≈ ambient dim. The group invariance probe
//! answers the *different* question "is the data invariant under a
//! *subgroup* of `G`?" by testing `d(q, g·q)` directly. The two are
//! orthogonal: a dataset can be subgroup-invariant without being
//! low-rank (uniform on `SO(2)` is full-rank but `SO(2)`-invariant).
//!
//! # Performance contract
//!
//! - [`invariance_score`]: O(1), one sigmoid.
//! - [`score_concentration`]: O(n) on a length-n score slice, zero-alloc,
//!   chunk-4 accumulation (mirrors [`crate::participation_ratio`]'s loop shape).
//! - [`classify_subgroup`]: O(n), zero-alloc, one [`score_concentration`]
//!   call plus a support-fraction scan.
//! - [`discover_subgroup_into`]: O(n_samples · cost(distance_fn)),
//!   zero-alloc after scratch init. The caller supplies `&mut [f32]`
//!   scratch for the per-sample scores and the rotated query buffer.
//!
//! # Determinism
//!
//! All operations are deterministic and platform-independent (pure float
//! arithmetic, no SIMD dispatch inside the math, no floating-point
//! reordering). Required for anti-cheat: a discovered-subgroup
//! classification that crosses the sync boundary (as a `u8` tag, see
//! [`SubgroupClass::as_u8`]) must be bit-identical across quorum nodes.
//!
//! # Feature gate
//!
//! `group_invariance_probe` — opt-in. Pure numeric substrate, no extra
//! deps. Sibling of `subspace_phase_gate`.

// ── Invariance score ───────────────────────────────────────────────────────

/// The per-element invariance score `σ(β·(1 − d))`.
///
/// Returns ≈ `1.0` when the normalized distance `d` is 0 (perfect
/// invariance under `g`), `0.5` when `d = 1` (the midpoint / indifference
/// point), and asymptotes to `0.0` as `d → ∞`. `β` controls sharpness:
/// higher `β` → sharper transition between "invariant" and "not invariant".
///
/// **Normalization convention:** the caller MUST normalize `d` so that the
/// "indifference" distance (where the score is exactly 0.5) maps to `d = 1`.
/// For L2 distances in `[0, d_max]`, divide by `d_max / 2`. For cosine
/// distance in `[0, 2]`, divide by 2. The recommended default is `β = 10.0`
/// which gives σ(10) ≈ 0.99995 at d=0 and σ(−10) ≈ 4.5e-5 at d=2.
///
/// This is a **shifted sigmoid** (not softmax, not RBF): `σ(β·(1−d))`. It
/// is the standard distance-to-affinity mapping that hits 1.0 at d=0 while
/// staying within the sigmoid family per the codebase convention. The
/// shift by 1 (rather than 0) is what distinguishes "perfectly invariant"
/// (d=0, score≈1) from "indifferent" (d=1, score=0.5) — without the shift,
/// `σ(−β·d)` would give 0.5 at d=0, conflating perfect invariance with
/// indifference.
#[inline]
pub fn invariance_score(distance: f32, beta: f32) -> f32 {
    // σ(β·(1 − d)). Inline sigmoid to avoid the `simd` module dependency
    // from this pure-numeric substrate; the call site is not on a SIMD
    // hot path.
    let x = beta * (1.0 - distance);
    // Numerically stable sigmoid (matches `simd::fast_sigmoid` semantics).
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

// ── Score variance (the discrete-vs-continuous classifier) ───────────────

/// Population variance of the invariance-score histogram.
///
/// Returns `f32 ∈ [0, 0.25]` for scores in `[0, 1]`:
/// - `≈ 0.25` — **maximally bimodal** (half scores near 1, half near 0)
///   → `Discrete`. A perfect `C_n` subgroup with `n/|G| = 0.5` lands here.
/// - `≈ 0.083` — **uniform spread** (`1/12`, all scores similar but varied)
///   → `Continuous`.
/// - `≈ 0` — **degenerate** (all scores identical: either all-1 for `H=G`
///   or all-0 for no symmetry).
///
/// This is the primary discrete-vs-continuous discriminator. It
/// distinguishes "sharp peaks separated by zero-score regions" (discrete:
/// high variance) from "smooth gradient" (continuous: low-to-moderate
/// variance). The participation-ratio-style [`score_concentration`] cannot
/// make this distinction when the discrete subgroup is a large fraction of
/// the hypothesis group (e.g. `C₄ ⊂ C₈` has concentration 0.5, identical to
/// a uniform spread at 50% support).
///
/// Zero-alloc, chunk-4 accumulation (two passes: mean, then variance).
#[inline]
pub fn score_variance(scores: &[f32]) -> f32 {
    let n = scores.len();
    if n == 0 {
        return 0.0;
    }
    // Pass 1: mean (chunk-4).
    let mut sum: f32 = 0.0;
    let mut i = 0;
    while i + 4 <= n {
        sum += scores[i] + scores[i + 1] + scores[i + 2] + scores[i + 3];
        i += 4;
    }
    while i < n {
        sum += scores[i];
        i += 1;
    }
    let mean = sum / (n as f32);
    // Pass 2: population variance (chunk-4).
    let mut sum_sq_dev: f32 = 0.0;
    let mut i = 0;
    while i + 4 <= n {
        let d0 = scores[i] - mean;
        let d1 = scores[i + 1] - mean;
        let d2 = scores[i + 2] - mean;
        let d3 = scores[i + 3] - mean;
        sum_sq_dev += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    while i < n {
        let d = scores[i] - mean;
        sum_sq_dev += d * d;
        i += 1;
    }
    sum_sq_dev / (n as f32)
}

// ── Score concentration (the effective-component-count measure) ────────────

/// Modelless concentration measure on the invariance-score histogram.
///
/// Returns `f32 ∈ [0, 1]`:
/// - `≈ 0` — **peaked** (a few scores near 1, rest near 0) → `Discrete`.
/// - `≈ 1` — **spread** (all scores similar) → `Continuous` or `None`.
///
/// Defined as the **participation-ratio analog of the score distribution**:
/// treating `s_i ∈ [0,1]` as a (non-negative) "energy", compute
/// `concentration = (Σ s_i)² / (n · Σ s_i²)`. This is exactly
/// [`crate::participation_ratio`] divided by `n`, normalized to `[0,1]`.
/// It equals `1.0` when all scores are equal (perfect spread) and
/// `≈ 1/k` when exactly `k` scores are `1` and the rest are `0` (perfect
/// `k`-peak discrete support).
///
/// This deliberately mirrors the private shard crate's spectral-flatness
/// semantics (Wiener entropy: 0 = single-mode, 1 = uniform) so downstream consumers
/// can reason about the same scale across the two substrates. The
/// participation-ratio form is chosen over the geometric/arithmetic-mean
/// form because it handles zero scores gracefully (geometric mean of a
/// vector containing 0 is 0, collapsing the discriminator).
///
/// Zero-alloc, chunk-4 accumulation.
#[inline]
pub fn score_concentration(scores: &[f32]) -> f32 {
    let n = scores.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum: f32 = 0.0;
    let mut sum_sq: f32 = 0.0;
    let mut i = 0;
    // Chunk-4 loop mirrors `participation_ratio` for SIMD auto-vectorisation.
    while i + 4 <= n {
        let a = scores[i].max(0.0);
        let b = scores[i + 1].max(0.0);
        let c = scores[i + 2].max(0.0);
        let d = scores[i + 3].max(0.0);
        sum += a + b + c + d;
        sum_sq += a * a + b * b + c * c + d * d;
        i += 4;
    }
    while i < n {
        let v = scores[i].max(0.0);
        sum += v;
        sum_sq += v * v;
        i += 1;
    }
    if sum_sq < f32::EPSILON {
        return 0.0;
    }
    // (Σs)² / (n · Σs²)  ∈ [0, 1] by Cauchy-Schwarz.
    let conc = (sum * sum) / (n as f32 * sum_sq);
    // Clamp for f32 round-off at the boundaries.
    conc.clamp(0.0, 1.0)
}

// ── Subgroup classification ────────────────────────────────────────────────

/// The discovered subgroup's structural class.
///
/// Determined by [`classify_subgroup`] from the score histogram. Encoded
/// as a `u8` tag ([`Self::as_u8`]) so it can cross the sync boundary as
/// a raw deterministic value (mirrors the private shard crate's freeze-gate
/// report raw fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SubgroupClass {
    /// No non-trivial invariance detected (all scores ≈ 0).
    None = 0,
    /// Discrete subgroup — score histogram is peaked (few `g`'s near 1).
    /// Examples: `C₄`, `Ico`, signed permutation groups.
    Discrete = 1,
    /// Continuous subgroup — score histogram is spread (smooth submanifold).
    /// Examples: `SO(2)`, `SO(3)`.
    Continuous = 2,
    /// Partial symmetry — bimodal but no clean peak (some `g`'s invariant
    /// for some inputs, not others). LieFlow §4.3 "partial symmetry" case.
    Partial = 3,
}

impl SubgroupClass {
    /// Stable `u8` encoding for sync-boundary transport. Inverse of
    /// [`Self::from_u8`].
    #[inline]
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Decode a [`SubgroupClass`] from its `u8` tag. Returns `None` for
    /// out-of-range values (defensive — sync consumers should treat
    /// unknown tags as `None` rather than guessing).
    #[inline]
    pub fn from_u8(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::None),
            1 => Some(Self::Discrete),
            2 => Some(Self::Continuous),
            3 => Some(Self::Partial),
            _ => None,
        }
    }
}

/// Default variance threshold above which a score histogram is
/// classified as `Discrete`. Calibrated against the theoretical maxima:
/// a perfect 50/50 bimodal split has variance 0.25; a uniform `[0,1]`
/// spread has variance `1/12 ≈ 0.083`. The threshold `0.15` sits between
/// these, classifying clearly-bimodal histograms as `Discrete` while
/// leaving smooth-spread histograms as `Continuous` or `Partial`.
pub const DEFAULT_DISCRETE_VARIANCE_THRESHOLD: f32 = 0.15;

/// Default variance threshold below which a score histogram with
/// significant support is classified as `Continuous`. Set below the
/// discrete threshold to leave a `Partial` band in between for genuinely
/// mixed distributions.
pub const DEFAULT_CONTINUOUS_VARIANCE_THRESHOLD: f32 = 0.08;

/// Default concentration threshold below which a score histogram is
/// classified as `Discrete`. Calibrated so that a clean `C₄` (4 peaks out
/// of, say, 64 samples) scores ≈ 0.06 — well under 0.3.
///
/// **Note:** this is a *secondary* signal. The primary discrete-vs-continuous
/// discriminator is [`score_variance`] (see [`DEFAULT_DISCRETE_VARIANCE_THRESHOLD`]).
/// `score_concentration` is kept for the "effective number of components"
/// interpretation and for consumers that want the `participation_ratio`-style
/// measure. When the two disagree, variance wins (it correctly handles
/// large-fraction discrete subgroups like `C₄ ⊂ C₈`).
pub const DEFAULT_DISCRETE_CONCENTRATION_THRESHOLD: f32 = 0.3;

/// Default concentration threshold above which a score histogram is
/// classified as `Continuous` (or `None` if all scores are low). Set
/// above the discrete threshold to leave a `Partial` band in between.
pub const DEFAULT_CONTINUOUS_CONCENTRATION_THRESHOLD: f32 = 0.7;

/// Default support threshold: a score `s` counts as "in the support" iff
/// `s > τ`. Calibrated for `β = 10.0` and normalized distances — adjust
/// per caller.
pub const DEFAULT_SUPPORT_TAU: f32 = 0.5;

/// Default minimum support fraction for a non-`None` classification.
/// If fewer than this fraction of samples clear `τ`, the discovered
/// subgroup is trivial (identity only) → `None`.
pub const DEFAULT_MIN_SUPPORT_FRACTION: f32 = 0.02;

/// Classify a discovered subgroup from its score histogram.
///
/// Combines [`score_variance`] (catches large-fraction discrete subgroups)
/// with [`score_concentration`] (catches small-fraction discrete subgroups)
/// and a support-fraction scan. Either variance or concentration firing
/// marks the subgroup `Discrete` — the two are complementary.
///
/// **Decision tree:**
/// 1. If support fraction < [`DEFAULT_MIN_SUPPORT_FRACTION`] → [`SubgroupClass::None`].
/// 2. Else if [`score_variance`] ≥ [`DEFAULT_DISCRETE_VARIANCE_THRESHOLD`] → [`SubgroupClass::Discrete`]
///    (large-fraction bimodal, e.g. `C₄ ⊂ C₈`).
/// 3. Else if [`score_concentration`] < [`DEFAULT_DISCRETE_CONCENTRATION_THRESHOLD`] → [`SubgroupClass::Discrete`]
///    (small-fraction peaked, e.g. `C₄ ⊂ C₆₄`).
/// 4. Else if [`score_variance`] < [`DEFAULT_CONTINUOUS_VARIANCE_THRESHOLD`] → [`SubgroupClass::Continuous`].
/// 5. Else → [`SubgroupClass::Partial`] (the band between discrete and continuous).
///
/// Zero-alloc.
#[inline]
pub fn classify_subgroup(scores: &[f32], tau: f32) -> SubgroupClass {
    classify_subgroup_with(
        scores,
        tau,
        DEFAULT_DISCRETE_VARIANCE_THRESHOLD,
        DEFAULT_CONTINUOUS_VARIANCE_THRESHOLD,
        DEFAULT_MIN_SUPPORT_FRACTION,
    )
}

/// Full-control variant of [`classify_subgroup`] — callers pass every
/// threshold. Uses BOTH [`score_variance`] (catches large-fraction discrete
/// subgroups like `C₄ ⊂ C₈` where the bimodal split gives variance ≈ 0.25)
/// AND [`score_concentration`] (catches small-fraction discrete subgroups
/// like `C₄ ⊂ C₆₄` where 4 peaks out of 64 gives concentration ≈ 0.06).
/// Either signal firing marks the subgroup `Discrete` — the two measures
/// are complementary (variance catches large-fraction, concentration
/// catches small-fraction).
#[inline]
pub fn classify_subgroup_with(
    scores: &[f32],
    tau: f32,
    discrete_variance_threshold: f32,
    continuous_variance_threshold: f32,
    min_support_fraction: f32,
) -> SubgroupClass {
    let n = scores.len();
    if n == 0 {
        return SubgroupClass::None;
    }
    // Support-fraction scan (chunk-4 for auto-vectorisation).
    let mut support: usize = 0;
    let mut i = 0;
    while i + 4 <= n {
        support += (scores[i] > tau) as usize
            + (scores[i + 1] > tau) as usize
            + (scores[i + 2] > tau) as usize
            + (scores[i + 3] > tau) as usize;
        i += 4;
    }
    while i < n {
        support += (scores[i] > tau) as usize;
        i += 1;
    }
    let support_fraction = support as f32 / n as f32;
    if support_fraction < min_support_fraction {
        return SubgroupClass::None;
    }
    let var = score_variance(scores);
    let conc = score_concentration(scores);
    // Discrete if EITHER signal fires: high variance (large-fraction
    // bimodal, e.g. C₄ ⊂ C₈) OR low concentration (small-fraction peaked,
    // e.g. C₄ ⊂ C₆₄). The two are complementary — together they cover
    // both regimes. The discrete_concentration_threshold is the
    // DEFAULT_DISCRETE_CONCENTRATION_THRESHOLD (0.3) baked in here; callers
    // wanting a different concentration cutoff should compute it manually
    // from score_concentration and classify_subgroup_with.
    let is_discrete_by_variance = var >= discrete_variance_threshold;
    let is_discrete_by_concentration = conc < DEFAULT_DISCRETE_CONCENTRATION_THRESHOLD;
    if is_discrete_by_variance || is_discrete_by_concentration {
        SubgroupClass::Discrete
    } else if var < continuous_variance_threshold {
        // Low variance + non-peaked → uniform spread → Continuous.
        SubgroupClass::Continuous
    } else {
        SubgroupClass::Partial
    }
}

// ── Report ─────────────────────────────────────────────────────────────────

/// Audit record produced by [`discover_subgroup`] / [`discover_subgroup_into`].
///
/// Mirrors the shape of the private shard crate's freeze-gate report so
/// downstream consumers can compose the two. Fields split by sync tier per the
/// global AGENTS.md raw-vs-latent rule:
///
/// - **Latent** (never synced raw): the per-sample scores, the
///   concentration scalar, the variance scalar.
/// - **Raw/deterministic** (may cross sync): `n_samples`, `n_support`,
///   `class` (as `u8`), `max_score` (a single `f32` summary).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubgroupReport {
    /// Number of `g ∈ G` samples scored.
    pub n_samples: usize,
    /// Number of samples with `score > tau`.
    pub n_support: usize,
    /// The discovered subgroup's structural class.
    pub class: SubgroupClass,
    /// The [`score_variance`] of the score histogram. Latent —
    /// informational, do not sync. This is the primary discrete-vs-
    /// continuous signal.
    pub variance: f32,
    /// The [`score_concentration`] of the score histogram. Latent —
    /// informational, do not sync. Secondary signal (effective component
    /// count); misclassifies large-fraction discrete subgroups.
    pub concentration: f32,
    /// The maximum invariance score observed. A summary statistic that
    /// MAY cross sync (single `f32`).
    pub max_score: f32,
}

// ── GroupAction trait ──────────────────────────────────────────────────────

/// A matrix Lie group acting on `ℝᵈ`, supplied by the caller.
///
/// The probe is generic over the group so it carries no group-specific
/// math. Concrete implementations (e.g. `So2Action`, `OrthogonalAction`)
/// live in downstream consumers — katgpt-rs ships only the trait and the
/// scoring/classification machinery.
///
/// # Safety contract
///
/// - [`act`](GroupAction::act) writes exactly `q.len()` values to `out`.
/// - [`sample`](GroupAction::sample) produces a deterministically-seeded
///   element when given a deterministic RNG (required for sync-bit-identity).
pub trait GroupAction {
    /// The group-element representation (e.g. `f32` angle for `SO(2)`,
    /// `[f32; 4]` quaternion for `SO(3)`, `[[f32; D]; D]` for `O(D)`).
    type Elem: Copy;

    /// Apply the group element `g` to the query vector `q`, writing the
    /// result to `out`. `out.len()` must equal `q.len()`.
    fn act(&self, g: &Self::Elem, q: &[f32], out: &mut [f32]);

    /// Sample a group element from the hypothesis prior `p(G)`. The RNG
    /// is caller-supplied so the probe stays deterministic.
    fn sample(&self, rng: &mut impl Rng) -> Self::Elem;
}

/// Convenience re-export of the `Rng` trait we need, so consumers don't
/// have to import `rand` directly just to implement [`GroupAction`].
/// We use a minimal in-house trait to keep this substrate zero-dep.
pub trait Rng {
    /// Return the next `u64` in the stream.
    fn next_u64(&mut self) -> u64;
    /// Return the next `f32` uniform in `[0, 1)`. Default impl derives
    /// from [`next_u64`](Self::next_u64).
    #[inline]
    fn next_f32(&mut self) -> f32 {
        // 24-bit mantissa → uniform in [0, 1).
        let bits = (self.next_u64() >> (64 - 24)) as u32;
        (bits as f32) * (1.0_f32 / (1u32 << 24) as f32)
    }
}

// ── Discovery (allocating and zero-alloc variants) ─────────────────────────

/// Discover the subgroup `H ⊆ G` from a stream of observations, allocating
/// the score buffer internally.
///
/// This is the convenience entry point. For hot paths, use
/// [`discover_subgroup_into`] with a pre-allocated scratch buffer.
///
/// # Arguments
///
/// * `group` — the hypothesis group `G`.
/// * `summary` — the observation summary `q` (mean trajectory, style
///   weights, etc.). The probe tests invariance of `q` under each sampled
///   `g ∈ G`.
/// * `n_samples` — number of `g`'s to sample.
/// * `distance_fn` — closure computing `d(q, g·q)` from the rotated query.
///   Receives `(g·q, q)` so the caller can pick the metric (L2, cosine,
///   Wasserstein-1 over a trajectory, ...).
/// * `beta` — invariance-score sharpness (see [`invariance_score`]).
/// * `tau` — support threshold (see [`classify_subgroup`]).
/// * `rng` — deterministic RNG (caller-supplied for sync-bit-identity).
pub fn discover_subgroup<G, F>(
    group: &G,
    summary: &[f32],
    n_samples: usize,
    distance_fn: F,
    beta: f32,
    tau: f32,
    rng: &mut impl Rng,
) -> SubgroupReport
where
    G: GroupAction,
    F: Fn(&[f32], &[f32]) -> f32,
{
    let d = summary.len();
    let mut scores = vec![0.0_f32; n_samples];
    let mut rotated = vec![0.0_f32; d];
    discover_subgroup_into(
        group,
        summary,
        n_samples,
        distance_fn,
        beta,
        tau,
        rng,
        &mut scores,
        &mut rotated,
    )
}

/// Zero-alloc variant of [`discover_subgroup`].
///
/// The caller supplies:
/// - `scores_scratch` — length `>= n_samples`, overwritten with per-sample
///   invariance scores.
/// - `rotated_scratch` — length `>= summary.len()`, scratch for `g·q`.
///
/// After the call, `scores_scratch[..n_samples]` holds the per-sample
/// scores (useful for downstream inspection or histogramming). The
/// returned [`SubgroupReport`] is the only allocation.
#[allow(clippy::too_many_arguments)]
pub fn discover_subgroup_into<G, F>(
    group: &G,
    summary: &[f32],
    n_samples: usize,
    distance_fn: F,
    beta: f32,
    tau: f32,
    rng: &mut impl Rng,
    scores_scratch: &mut [f32],
    rotated_scratch: &mut [f32],
) -> SubgroupReport
where
    G: GroupAction,
    F: Fn(&[f32], &[f32]) -> f32,
{
    debug_assert!(
        scores_scratch.len() >= n_samples,
        "scores_scratch must hold n_samples={} entries, got {}",
        n_samples,
        scores_scratch.len()
    );
    debug_assert!(
        rotated_scratch.len() >= summary.len(),
        "rotated_scratch must hold summary.len()={} entries, got {}",
        summary.len(),
        rotated_scratch.len()
    );
    let d = summary.len();
    let mut max_score: f32 = 0.0;
    for slot in scores_scratch[..n_samples].iter_mut() {
        let g = group.sample(rng);
        group.act(&g, summary, &mut rotated_scratch[..d]);
        let dist = distance_fn(&rotated_scratch[..d], summary);
        let score = invariance_score(dist, beta);
        *slot = score;
        if score > max_score {
            max_score = score;
        }
    }
    let class = classify_subgroup(&scores_scratch[..n_samples], tau);
    let n_support = scores_scratch[..n_samples]
        .iter()
        .filter(|&&s| s > tau)
        .count();
    let variance = score_variance(&scores_scratch[..n_samples]);
    let concentration = score_concentration(&scores_scratch[..n_samples]);
    SubgroupReport {
        n_samples,
        n_support,
        class,
        variance,
        concentration,
        max_score,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "group_invariance_probe")]
mod tests {
    use super::*;

    // ── invariance_score ──────────────────────────────────────────────────

    #[test]
    fn invariance_score_at_zero_distance_is_near_one() {
        // σ(β·(1−0)) = σ(β). For β=10: σ(10) ≈ 0.99995.
        assert!((invariance_score(0.0, 10.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn invariance_score_at_midpoint_is_half() {
        // σ(β·(1−1)) = σ(0) = 0.5 exactly.
        assert!((invariance_score(1.0, 10.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn invariance_score_decreases_monotonically_with_distance() {
        let beta = 5.0;
        let mut prev = 2.0; // above 1.0 to force the first comparison to hold
        for k in 0..20 {
            let d = k as f32 * 0.05;
            let s = invariance_score(d, beta);
            assert!(s <= prev + 1e-6, "score not monotone at d={d}: {s} > {prev}");
            prev = s;
        }
    }

    // ── score_variance ────────────────────────────────────────────────────

    #[test]
    fn variance_uniform_scores_is_near_zero() {
        // All scores equal → variance = 0.
        let scores = [0.5_f32; 64];
        let v = score_variance(&scores);
        assert!(v < 1e-6, "uniform scores should give v≈0, got {v}");
    }

    #[test]
    fn variance_bimodal_50_50_is_quarter() {
        // Half scores = 1, half = 0 → variance = 0.5·0.5 = 0.25 (max).
        let mut scores = [0.0_f32; 64];
        for s in scores[..32].iter_mut() {
            *s = 1.0;
        }
        let v = score_variance(&scores);
        assert!((v - 0.25).abs() < 1e-5, "50/50 bimodal should give v≈0.25, got {v}");
    }

    // ── score_concentration ───────────────────────────────────────────────

    #[test]
    fn concentration_uniform_scores_is_one() {
        // All scores equal → concentration = 1.0 (perfect spread).
        let scores = [0.5_f32; 64];
        let c = score_concentration(&scores);
        assert!((c - 1.0).abs() < 1e-5, "uniform scores should give c≈1, got {c}");
    }

    #[test]
    fn concentration_single_peak_is_low() {
        // One score = 1, rest = 0 → concentration = 1/64 ≈ 0.0156.
        let mut scores = [0.0_f32; 64];
        scores[0] = 1.0;
        let c = score_concentration(&scores);
        assert!(
            c < 0.05,
            "single-peak scores should give c<0.05, got {c}"
        );
    }

    #[test]
    fn concentration_four_peaks_matches_c4() {
        // 4 scores = 1, rest = 0 → concentration = 4/64 = 0.0625.
        // This is the C₄ signature.
        let mut scores = [0.0_f32; 64];
        for i in 0..4 {
            scores[i * 16] = 1.0;
        }
        let c = score_concentration(&scores);
        // Expected exactly 4/64 = 0.0625.
        assert!(
            (c - 0.0625).abs() < 1e-5,
            "4-peak C₄ signature should give c≈0.0625, got {c}"
        );
    }

    // ── classify_subgroup ─────────────────────────────────────────────────

    #[test]
    fn classify_uniform_low_scores_is_none() {
        // All scores ≈ 0.1 (below tau=0.5) → no support → None.
        let scores = [0.1_f32; 64];
        assert_eq!(classify_subgroup(&scores, 0.5), SubgroupClass::None);
    }

    #[test]
    fn classify_uniform_high_scores_is_continuous() {
        // All scores ≈ 0.9 → spread → Continuous.
        let scores = [0.9_f32; 64];
        assert_eq!(classify_subgroup(&scores, 0.5), SubgroupClass::Continuous);
    }

    #[test]
    fn classify_four_peaks_is_discrete() {
        // 4 scores = 1, rest = 0 → Discrete (C₄ signature).
        let mut scores = [0.0_f32; 64];
        for i in 0..4 {
            scores[i * 16] = 1.0;
        }
        assert_eq!(classify_subgroup(&scores, 0.5), SubgroupClass::Discrete);
    }

    // ── SubgroupClass round-trip ──────────────────────────────────────────

    #[test]
    fn subgroup_class_u8_round_trip() {
        for tag in 0..=3u8 {
            let cls = SubgroupClass::from_u8(tag).expect("0..=3 are valid");
            assert_eq!(cls.as_u8(), tag);
        }
        assert!(SubgroupClass::from_u8(4).is_none());
        assert!(SubgroupClass::from_u8(255).is_none());
    }

    // ── discover_subgroup_into on a synthetic SO(2) → C₄ setting ──────────

    /// Minimal in-house RNG (splitmix64 — deterministic, zero-dep).
    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
    }

    impl Rng for SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    /// `C₈` action on indicator vectors: cyclic shift by `k` positions, where
    /// `k = round(8·θ/(2π)) mod 8`. The hypothesis group is the *discrete* C₈
    /// (8 elements), standing in for the paper's continuous SO(2) — the probe
    /// works identically on both; we use C₈ here so the action is exact (no
    /// interpolation) and the test is bit-deterministic.
    struct C8Action;

    impl GroupAction for C8Action {
        type Elem = u8; // the shift k ∈ {0..7}

        fn act(&self, g: &Self::Elem, q: &[f32], out: &mut [f32]) {
            let k = (*g as usize) % 8;
            let n = q.len();
            debug_assert_eq!(n, 8, "C8Action expects 8-dim indicator vectors");
            for i in 0..n {
                out[i] = q[(i + k) % n];
            }
        }

        fn sample(&self, rng: &mut impl Rng) -> Self::Elem {
            // Uniform on {0..7}.
            (rng.next_u64() % 8) as u8
        }
    }

    /// Normalized L1 distance on 8-dim indicator vectors, scaled so the
    /// midpoint (d=1) sits between "identical" (d=0) and "complementary"
    /// (d=2). For the C₄-invariant indicator [1,0,1,0,1,0,1,0]: a C₄ shift
    /// (k ∈ {0,2,4,6}) leaves it unchanged → L1=0 → d=0. A non-C₄ shift
    /// (k ∈ {1,3,5,7}) produces [0,1,0,1,0,1,0,1] → L1=8 → d=8/4=2.
    fn normalized_l1_indicator(a: &[f32], b: &[f32]) -> f32 {
        let l1: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
        l1 * 0.25 // normalize so max L1=8 maps to d=2
    }

    #[test]
    fn discover_c8_to_c4_recovers_discrete_class() {
        // The C₄-invariant distribution on C₈: indicator [1,0,1,0,1,0,1,0].
        // Under C₄ shifts (k ∈ {0,2,4,6}): unchanged. Under non-C₄ shifts
        // (k ∈ {1,3,5,7}): flips to [0,1,0,1,0,1,0,1]. So 4 of 8 group
        // elements are symmetries, 4 are not — the discovered H is C₄ ⊂ C₈.
        let q = [1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut scores = [0.0_f32; 256];
        let mut rotated = [0.0_f32; 8];
        let mut rng = SplitMix64::new(0x356C_C4FE_ED53_56C4);
        let report = discover_subgroup_into(
            &C8Action,
            &q,
            256,
            normalized_l1_indicator,
            10.0, // β — sharp
            0.5,  // τ
            &mut rng,
            &mut scores,
            &mut rotated,
        );
        // C₄ has 4 elements out of C₈'s 8 → expect ~50% support (4/8 of
        // samples land on C₄ shifts), classification Discrete (4 sharp
        // peaks at score ≈ 1, rest at ≈ 0).
        assert_eq!(
            report.class,
            SubgroupClass::Discrete,
            "C₄ ⊂ C₈ should classify as Discrete, got {:?} (conc={}, n_support={})",
            report.class,
            report.concentration,
            report.n_support
        );
        // C₄ has 4 elements; with 256 samples uniform on C₈, expect ~128
        // support samples (those that landed on C₄ shifts). Allow variance.
        assert!(
            report.n_support >= 100,
            "expected ≥100 support samples (C₄ is 4/8 of C₈, 256 samples), got {}",
            report.n_support
        );
        // Max score should be ≈ 1.0 (C₄ shifts give d=0 → σ(β) ≈ 1).
        assert!(
            report.max_score > 0.95,
            "max score should be >0.95 (C₄ shifts give d=0), got {}",
            report.max_score
        );
    }
}
