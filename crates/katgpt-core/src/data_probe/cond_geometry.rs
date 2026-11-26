//! Conditional geometry — subset-conditional intrinsic dimension (`cond_id`)
//! and cross-space information imbalance (`cond_imbalance`).
//!
//! Plan 638 (Research 675 consumer); primitives for riir-refine Issue 239
//! axis (i). Two closed-form, modelless, offline-audit-cadence estimators
//! over point-latent populations:
//!
//! 1. **[`cond_id`]** — subset-conditional intrinsic dimension via the BIDE
//!    two-ball binomial restriction (arXiv:2608.25166, "The Changing
//!    Geometry of Grammar"): neighbor counts are computed ONCE over the
//!    FULL population around each point (inner radius r_A = τ·r_B, r_B =
//!    the m-th nearest-neighbor distance, m = ceil(0.5·√n)), and the
//!    estimate restricts the AVERAGE of those once-computed per-point
//!    statistics to the subset G: `d̂_G = ln(k̄_A^G / k̄_B^G) / ln(τ)`.
//!    Adaptive per-point neighborhood selection (ABIDE, arXiv:2405.15132)
//!    is the published method's second half and is deliberately NOT here
//!    (Plan 638 T1.5 — the static-τ closed form is the filing).
//! 2. **[`cond_imbalance`]** — cross-space neighborhood-rank prediction
//!    (Information Imbalance, Glielmo et al., arXiv:2104.15079): each
//!    point's nearest neighbor in the `from` space is located, its rank in
//!    the `to` space's distance ordering around the same point is counted,
//!    the rank is normalized by n/2, and the map is averaged over the
//!    subset. Asymmetric by construction (Δ(A→B) ≠ Δ(B→A)). Reading: 0 =
//!    neighborhoods preserved, ≈ 1 = unrelated spaces, → 2 = anti-correlated.
//!
//! # Estimand (binding, Plan 638 D2)
//!
//! The neighbor sets in [`ball_counts`] are the FULL population's — labels
//! never filter them. [`cond_id`] over G is therefore NOT the two-ball
//! estimate of G in isolation: it is the restricted average of full-
//! population statistics. The difference is a different estimand (the
//! neighbor sets themselves change in isolation), not an error — the
//! `cond_geometry_g1b_conditioning` gate asserts the identity against a
//! restricted sum of the once-computed per-point statistics AND discloses
//! the isolation gap as present-and-finite, never small.
//!
//! # Contract
//!
//! - **No UQ-bearing output (Plan 638 D4c):** no interval, quantile, or
//!   coverage is claimed or emitted — the conformal-naive floor rule does
//!   not apply. The returns are bare point estimates.
//! - **Local-only (Plan 638 D5):** both estimators are offline audit
//!   instruments that never cross a sync/quorum boundary; their determinism
//!   serves reproducibility, not quorum. Distances fold through a FIXED
//!   4-lane chunked f32 accumulation (lane order pinned in source), so
//!   results are deterministic per build; cross-platform bit-identity is
//!   NOT claimed (the sync-crossing determinism law does not bind here).
//! - **Loud refusals (Plan 638 D4a):** a subset below
//!   [`MIN_GROUP_TOKENS`] (the paper's own discard, Appendix A.1 —
//!   "tokens" is the paper's population vocabulary; here rows/points)
//!   refuses [`CondGeometryError::SubsetTooSmall`]; non-finite input
//!   refuses [`CondGeometryError::NonFinite`]; a two-ball form with an
//!   empty inner-radius mean refuses [`CondGeometryError::DegenerateCounts`]
//!   — never a silent ±inf. A scratch [`ball_counts`] never stuffed (or
//!   not stuffed for the queried rows) refuses
//!   [`CondGeometryError::ScratchNotComputed`].
//! - **Ties (Plan 638 D4b):** first-index wins on equal distances — the
//!   m-th-order heap keeps the first-seen distance on ties (strict `<`
//!   replacement) and the imbalance argmin scans with strict `<`.
//!   Zero-distance duplicates are counted as-is in the ball counts (they
//!   are the collapse signal: duplicates inflate both counts and deflate
//!   d̂, the same signal class as `knn_entropy`'s duplicates ⇒ −∞).
//!   Radius counts are inclusive (`d ≤ r`); imbalance ranks count
//!   strictly-nearer points (`d < d*`; equal-distance ties are not nearer).
//! - **Zero steady-state allocation:** every buffer lives in the caller's
//!   scratch, sized once at `with_capacity`; the hot loops write in place.
//!   The `cond_geometry_g4_alloc` gate pins the counting-allocator census.
//! - **Brute force O(n²·d) by design** (the `knn_entropy` posture: offline
//!   audit cadence, n ≤ a few thousand). [`ball_counts`] computes each
//!   selected point's distance row ONCE into the scratch workspace — the
//!   radius-count sweep is a comparison scan over the stored row, never a
//!   recomputation. kiddo/linfa-nn is the recorded upgrade path if
//!   populations outgrow the envelope (Plan 638 D6).
//!
//! # Not the Issue-708 imbalance monitor
//!
//! [`super::imbalance`] is the two-channel COLLAPSE alarm (KNN entropy +
//! cosine concentration) from Issue 708 / Research 437. This module is a
//! different instrument: conditional-geometry ESTIMATORS over labeled
//! populations. The shared word "imbalance" is a vocabulary collision, not
//! a substrate overlap — [`cond_imbalance`] consumes two representations
//! of one population; the 708 monitor consumes one population over time.
//!
//! # Citation block
//!
//! - arXiv:2608.25166 — "The Changing Geometry of Grammar" (Vallisa,
//!   Ravenda et al., 2026): the two-ball restriction estimator.
//! - arXiv:2405.15132 — ABIDE: the adaptive-selection half (deferred,
//!   Plan 638 T1.5).
//! - arXiv:2104.15079 — Glielmo et al.: Information Imbalance.

// ──────────────────────────────────────────────────────────────────────────
// Subset-support floor (Plan 638 D4a — the paper's Appendix A.1 discard).
// ──────────────────────────────────────────────────────────────────────────

/// Minimum subset support for a [`cond_id`] / [`cond_imbalance`] group
/// query. Below it the estimator refuses loud — never a silent estimate.
/// The name carries the paper's population vocabulary ("tokens"); here a
/// token is a row (point) of the population.
pub const MIN_GROUP_TOKENS: usize = 10;

// ──────────────────────────────────────────────────────────────────────────
// Errors
// ──────────────────────────────────────────────────────────────────────────

/// Refuse-loud conditional-geometry verdicts (Plan 638 D4). Every variant
/// names what it measured — the estimator refuses WITH the evidence,
/// never a bare no.
#[derive(Debug, Clone, PartialEq)]
pub enum CondGeometryError {
    /// The queried group holds fewer than [`MIN_GROUP_TOKENS`] rows.
    SubsetTooSmall {
        /// Rows the group actually carried.
        got: usize,
        /// [`MIN_GROUP_TOKENS`].
        min: usize,
    },
    /// A non-finite coordinate was scanned before any distance work ran.
    /// `at` is the flat index of the CHAINED scan — `points` in
    /// [`ball_counts`], `from ‖ to` in [`cond_imbalance`] — so an index
    /// `≥ from.len()` names a `to` coordinate.
    NonFinite {
        /// Flat float index of the first non-finite coordinate.
        at: usize,
    },
    /// The two-ball closed form is degenerate for this scratch: the
    /// inner-radius mean or the outer-radius mean over the group is ≤ 0
    /// (an all-empty inner ball is the τ-too-small / lattice-shell signal).
    /// The measured means ride the variant.
    DegenerateCounts {
        /// Mean inner-radius neighbor count over the group.
        ka_bar: f64,
        /// Mean outer-radius neighbor count over the group.
        kb_bar: f64,
    },
    /// The scratch carries no computed counts for this query — either
    /// [`ball_counts`] never ran on it, or the queried rows were not in
    /// its `indices`.
    ScratchNotComputed,
}

impl core::fmt::Display for CondGeometryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SubsetTooSmall { got, min } => {
                write!(
                    f,
                    "cond_geometry: group support {got} < MIN_GROUP_TOKENS {min} — refusing the estimate (paper Appendix A.1 discard)"
                )
            }
            Self::NonFinite { at } => {
                write!(f, "cond_geometry: non-finite coordinate at flat index {at}")
            }
            Self::DegenerateCounts { ka_bar, kb_bar } => write!(
                f,
                "cond_geometry: degenerate two-ball counts — k̄_A {ka_bar:.3} / k̄_B {kb_bar:.3}; τ too small or a collapsed/shell population"
            ),
            Self::ScratchNotComputed => write!(
                f,
                "cond_geometry: scratch carries no counts for this query — run ball_counts over these rows first"
            ),
        }
    }
}

impl std::error::Error for CondGeometryError {}

// ──────────────────────────────────────────────────────────────────────────
// Distances — fixed 4-lane chunked f32 fold (module-law determinism)
// ──────────────────────────────────────────────────────────────────────────

/// Squared Euclidean distance between two `dim`-length rows.
///
/// The fold is CHUNKED into four fixed accumulator lanes with the tail
/// folded into lane 0 and the lane fold pinned `(s0 + s1) + (s2 + s3)` —
/// deterministic per build, vectorizable, and never order-ambiguous. This
/// is a local-only instrument (module head, Plan 638 D5): cross-platform
/// bit-identity is NOT claimed.
#[inline]
fn sq_dist_row(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut c = 0;
    while c + 4 <= a.len() {
        let d0 = a[c] - b[c];
        let d1 = a[c + 1] - b[c + 1];
        let d2 = a[c + 2] - b[c + 2];
        let d3 = a[c + 3] - b[c + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
        c += 4;
    }
    while c < a.len() {
        let d = a[c] - b[c];
        s0 += d * d;
        c += 1;
    }
    (s0 + s1) + (s2 + s3)
}

// ──────────────────────────────────────────────────────────────────────────
// Bounded max-heap (m smallest squared distances; first-index tie retained)
// ──────────────────────────────────────────────────────────────────────────

fn heap_push(heap: &mut [f32], len: &mut usize, v: f32) {
    let mut i = *len;
    heap[i] = v;
    *len += 1;
    while i > 0 {
        let parent = (i - 1) / 2;
        if heap[parent] >= heap[i] {
            break;
        }
        heap.swap(parent, i);
        i = parent;
    }
}

fn heap_replace_root(heap: &mut [f32], v: f32) {
    let len = heap.len();
    heap[0] = v;
    let mut i = 0;
    loop {
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        let mut largest = i;
        if l < len && heap[l] > heap[largest] {
            largest = l;
        }
        if r < len && heap[r] > heap[largest] {
            largest = r;
        }
        if largest == i {
            break;
        }
        heap.swap(i, largest);
        i = largest;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Ball-count scratch + m-th-order radius
// ──────────────────────────────────────────────────────────────────────────

/// The m-th-order index for the outer radius: `ceil(0.5·√n)`, minimum 1 —
/// the deterministic per-point quantile rule (Plan 638 T1.3).
#[must_use]
pub fn mth_order_for(n: usize) -> usize {
    usize::max(1, ((n as f64).sqrt() * 0.5).ceil() as usize)
}

/// Caller-owned scratch for the two-ball pass (Plan 638 T1.2): per-point
/// neighbor counts, per-point outer radii, the bounded m-order heap, and
/// the per-point distance workspace. Sized ONCE at `with_capacity`; the
/// hot loops write in place (zero steady-state allocation).
///
/// The count fields are public so consumers and the G1b identity gate can
/// read the once-computed per-point statistics directly.
pub struct BallCountScratch {
    /// Per-point inner-radius (r_A = τ·r_B) neighbor count over the FULL
    /// population, self excluded. Rows [`ball_counts`] never selected stay 0.
    pub k_a: Vec<u32>,
    /// Per-point outer-radius (r_B, the m-th NN distance) neighbor count
    /// over the FULL population, self excluded.
    pub k_b: Vec<u32>,
    /// Per-point outer radius, squared (the m-th NN distance²; first-index
    /// tie rule on the selection).
    pub r2_b: Vec<f32>,
    /// Rows [`ball_counts`] actually computed. [`cond_id`] refuses a group
    /// row this flag does not cover.
    pub selected: Vec<bool>,
    /// The τ this scratch was stuffed with (0.0 = never stuffed).
    pub tau: f32,
    pub(crate) heap: Vec<f32>,
    pub(crate) dist_row: Vec<f32>,
}

impl BallCountScratch {
    /// Build the scratch for a population of `n` points. Allocates once;
    /// reuse with [`clear`](BallCountScratch::clear).
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        let m = mth_order_for(n);
        Self {
            k_a: vec![0; n],
            k_b: vec![0; n],
            r2_b: vec![0.0; n],
            selected: vec![false; n],
            tau: 0.0,
            heap: vec![0.0; m],
            dist_row: vec![0.0; n],
        }
    }

    /// Zero the counts, radii, selection and τ for reuse. The heap and
    /// distance workspace keep their capacity (contents are garbage).
    pub fn clear(&mut self) {
        self.k_a.iter_mut().for_each(|v| *v = 0);
        self.k_b.iter_mut().for_each(|v| *v = 0);
        self.r2_b.iter_mut().for_each(|v| *v = 0.0);
        self.selected.iter_mut().for_each(|v| *v = false);
        self.tau = 0.0;
    }
}

/// One pass over the population: for every index in `indices`, compute the
/// point's distance row ONCE (O(n·d) per point into the scratch
/// workspace), select the m-th nearest-neighbor distance as the outer
/// radius r_B (m = [`mth_order_for`], first-index tie rule), set the inner
/// radius r_A = τ·r_B, and count the FULL population's neighbors inside
/// both radii (self excluded, duplicates counted as-is, counts inclusive).
///
/// The counts feed any number of [`cond_id`] group queries — the neighbor
/// sets never depend on the group (Plan 638 D2, the estimand law).
///
/// # Panics
///
/// Asserts (loud shape contract, the `knn_entropy` convention): `n ≥ 4`,
/// `dim > 0`, `points.len() == n·dim`, `indices` non-empty with every
/// entry `< n`, `tau` finite in `(0, 1)`, and the scratch built for this
/// `n`.
///
/// # Errors
///
/// [`CondGeometryError::NonFinite`] naming the flat index of the first
/// non-finite coordinate — refused before any O(n²) work.
pub fn ball_counts(
    points: &[f32],
    n: usize,
    dim: usize,
    indices: &[u32],
    tau: f32,
    scratch: &mut BallCountScratch,
) -> Result<(), CondGeometryError> {
    assert!(
        n >= 4,
        "cond_geometry: n ({n}) must be ≥ 4 for the two-ball form"
    );
    assert!(dim > 0, "cond_geometry: dim must be positive");
    assert_eq!(
        points.len(),
        n * dim,
        "cond_geometry: points.len() must be n × dim = {n} × {dim}"
    );
    assert!(
        !indices.is_empty(),
        "cond_geometry: indices must be non-empty"
    );
    assert!(
        tau.is_finite() && tau > 0.0 && tau < 1.0,
        "cond_geometry: tau ({tau}) must be finite in (0, 1)"
    );
    assert_eq!(
        scratch.k_a.len(),
        n,
        "cond_geometry: scratch built for a different n"
    );
    let m = scratch.heap.len();
    assert!(
        m == mth_order_for(n),
        "cond_geometry: scratch heap sized {m}, expected {} for n = {n}",
        mth_order_for(n)
    );

    for (at, &v) in points.iter().enumerate() {
        if !v.is_finite() {
            return Err(CondGeometryError::NonFinite { at });
        }
    }

    let t2 = tau * tau;
    let heap = &mut scratch.heap[..];
    for &idx in indices {
        let i = idx as usize;
        assert!(i < n, "cond_geometry: index {i} out of range (n = {n})");
        let row = &points[i * dim..(i + 1) * dim];

        // Sweep 1 — the distance row ONCE + the m-order selection (first-
        // index tie rule: strict `<` keeps the first-seen distance on ties).
        let mut hlen = 0usize;
        for j in 0..n {
            let d2 = if j == i {
                0.0
            } else {
                sq_dist_row(row, &points[j * dim..(j + 1) * dim])
            };
            scratch.dist_row[j] = d2;
            if j == i {
                continue;
            }
            if hlen < m {
                heap_push(heap, &mut hlen, d2);
            } else if d2 < heap[0] {
                heap_replace_root(heap, d2);
            }
        }
        debug_assert_eq!(hlen, m, "population smaller than the m-order");
        let r2_b = heap[0];
        scratch.r2_b[i] = r2_b;

        // Sweep 2 — the radius counts are a comparison scan over the
        // stored row, never a recomputation.
        let r2_a = t2 * r2_b;
        let mut ka = 0u32;
        let mut kb = 0u32;
        for (j, &d2) in scratch.dist_row[..n].iter().enumerate() {
            if j == i {
                continue;
            }
            if d2 <= r2_a {
                ka += 1;
            }
            if d2 <= r2_b {
                kb += 1;
            }
        }
        scratch.k_a[i] = ka;
        scratch.k_b[i] = kb;
        scratch.selected[i] = true;
    }
    scratch.tau = tau;
    Ok(())
}

/// Subset-conditional intrinsic dimension: the closed form over the
/// restricted average of the ONCE-computed per-point ball counts —
/// `d̂_G = ln(k̄_A^G / k̄_B^G) / ln(τ)` (Plan 638 T1.4 / D2).
///
/// # Panics
///
/// Asserts every group index is in range for the scratch.
///
/// # Errors
///
/// - [`CondGeometryError::SubsetTooSmall`] — the group carries fewer than
///   [`MIN_GROUP_TOKENS`] rows.
/// - [`CondGeometryError::ScratchNotComputed`] — the scratch was never
///   stuffed, or a queried row was never selected by [`ball_counts`].
/// - [`CondGeometryError::DegenerateCounts`] — the restricted inner- or
///   outer-radius mean is ≤ 0 (the measured means ride the error).
#[must_use = "the estimate is the answer — a dropped Result is a dropped refusal"]
pub fn cond_id(counts: &BallCountScratch, group: &[u32]) -> Result<f32, CondGeometryError> {
    if counts.tau <= 0.0 {
        return Err(CondGeometryError::ScratchNotComputed);
    }
    if group.len() < MIN_GROUP_TOKENS {
        return Err(CondGeometryError::SubsetTooSmall {
            got: group.len(),
            min: MIN_GROUP_TOKENS,
        });
    }
    let mut sum_a = 0u64;
    let mut sum_b = 0u64;
    for &g in group {
        let i = g as usize;
        assert!(
            i < counts.k_a.len(),
            "cond_geometry: group index {i} out of range"
        );
        if !counts.selected[i] {
            return Err(CondGeometryError::ScratchNotComputed);
        }
        sum_a += u64::from(counts.k_a[i]);
        sum_b += u64::from(counts.k_b[i]);
    }
    let got = group.len() as f64;
    let ka_bar = sum_a as f64 / got;
    let kb_bar = sum_b as f64 / got;
    if ka_bar <= 0.0 || kb_bar <= 0.0 {
        return Err(CondGeometryError::DegenerateCounts { ka_bar, kb_bar });
    }
    // Pinned form: the difference of logs (never the log of the ratio —
    // one rounding path, the same one the G1b identity gate recomputes).
    let d = (ka_bar.ln() - kb_bar.ln()) / (counts.tau as f64).ln();
    Ok(d as f32)
}

// ──────────────────────────────────────────────────────────────────────────
// Information imbalance
// ──────────────────────────────────────────────────────────────────────────

/// Caller-owned scratch for [`cond_imbalance`] (Plan 638 T1.6): the
/// per-point nearest neighbor in the `from` space and the per-point
/// `to`-space rank of that neighbor. Both maps are LEFT POPULATED after
/// the call — the full per-point map is a diagnostic the group average
/// does not carry (multiple group reads cost nothing extra).
pub struct ImbalanceScratch {
    /// Per point: the first-index nearest neighbor in the `from` space
    /// (self excluded).
    pub nn_from: Vec<u32>,
    /// Per point: how many OTHER points are strictly nearer in the `to`
    /// space than the point's `from`-space nearest neighbor is.
    pub ranks_to: Vec<u32>,
    pub(crate) dist_a: Vec<f32>,
    pub(crate) dist_b: Vec<f32>,
}

impl ImbalanceScratch {
    /// Build the scratch for a population of `n` points in both spaces.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            nn_from: vec![0; n],
            ranks_to: vec![0; n],
            dist_a: vec![0.0; n],
            dist_b: vec![0.0; n],
        }
    }

    /// Zero the maps for reuse. The distance workspaces keep capacity.
    pub fn clear(&mut self) {
        self.nn_from.iter_mut().for_each(|v| *v = 0);
        self.ranks_to.iter_mut().for_each(|v| *v = 0);
    }
}

/// Cross-space information imbalance over a subset (Plan 638 T1.6): for
/// every point, locate its nearest neighbor in `from` (first-index tie
/// rule), count how many OTHER points are strictly nearer in `to` (the
/// rank), normalize by n/2, and average the map over `group`. The FULL
/// per-point map is computed into `scratch` regardless of the group —
/// Δ(G) is a read of it, not a separate pass.
///
/// `from` and `to` are two representations of the SAME n points. The
/// instrument is asymmetric by construction: Δ(A→B) ≠ Δ(B→A). Reading:
/// 0 = neighborhoods preserved, ≈ 1 = unrelated spaces, → 2 = anti-
/// correlated.
///
/// # Panics
///
/// Asserts `n ≥ 2`, `dim > 0`, both slices `n·dim` long, and the scratch
/// built for this `n`.
///
/// # Errors
///
/// - [`CondGeometryError::NonFinite`] — scanned in `from` first, then
///   `to`, before any O(n²) work; the flat index is continuous across the
///   chained scan (an index ≥ `from.len()` names a `to` coordinate).
/// - [`CondGeometryError::SubsetTooSmall`] — the group carries fewer than
///   [`MIN_GROUP_TOKENS`] rows.
#[must_use = "the estimate is the answer — a dropped Result is a dropped refusal"]
pub fn cond_imbalance(
    from: &[f32],
    to: &[f32],
    n: usize,
    dim: usize,
    group: &[u32],
    scratch: &mut ImbalanceScratch,
) -> Result<f32, CondGeometryError> {
    assert!(n >= 2, "cond_geometry: imbalance needs n ≥ 2 (got {n})");
    assert!(dim > 0, "cond_geometry: dim must be positive");
    assert_eq!(from.len(), n * dim, "cond_geometry: from must be n × dim");
    assert_eq!(to.len(), n * dim, "cond_geometry: to must be n × dim");
    assert_eq!(
        scratch.nn_from.len(),
        n,
        "cond_geometry: scratch built for a different n"
    );

    for (at, &v) in from.iter().chain(to.iter()).enumerate() {
        if !v.is_finite() {
            return Err(CondGeometryError::NonFinite { at });
        }
    }

    for i in 0..n {
        let row = &from[i * dim..(i + 1) * dim];
        let mut best_d2 = f32::INFINITY;
        let mut best_j = 0usize;
        for j in 0..n {
            if j == i {
                continue;
            }
            let d2 = sq_dist_row(row, &from[j * dim..(j + 1) * dim]);
            scratch.dist_a[j] = d2;
            if d2 < best_d2 {
                best_d2 = d2;
                best_j = j;
            }
        }
        scratch.nn_from[i] = best_j as u32;

        let trow = &to[i * dim..(i + 1) * dim];
        let r2_star = sq_dist_row(trow, &to[best_j * dim..(best_j + 1) * dim]);
        let mut rank = 0u32;
        for k in 0..n {
            if k == i {
                continue;
            }
            let d2 = sq_dist_row(trow, &to[k * dim..(k + 1) * dim]);
            scratch.dist_b[k] = d2;
            if d2 < r2_star {
                rank += 1;
            }
        }
        scratch.ranks_to[i] = rank;
    }

    if group.len() < MIN_GROUP_TOKENS {
        return Err(CondGeometryError::SubsetTooSmall {
            got: group.len(),
            min: MIN_GROUP_TOKENS,
        });
    }
    let half_n = n as f64 / 2.0;
    let mut sum = 0.0f64;
    for &g in group {
        let i = g as usize;
        assert!(
            i < n,
            "cond_geometry: group index {i} out of range (n = {n})"
        );
        sum += scratch.ranks_to[i] as f64 / half_n;
    }
    Ok((sum / group.len() as f64) as f32)
}

// ──────────────────────────────────────────────────────────────────────────
// Unit tests (Plan 638 T1.7)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mth_order_pins() {
        assert_eq!(mth_order_for(4), 1);
        assert_eq!(mth_order_for(9), 2);
        assert_eq!(mth_order_for(12), 2);
        assert_eq!(mth_order_for(2000), 23);
        assert_eq!(mth_order_for(1), 1, "minimum 1");
    }

    /// The closed form over synthetic counts: k̄_A/k̄_B = 0.25 at τ = 0.5
    /// is exactly d = 2; a mixed group hand-computes to ln(0.3)/ln(0.5).
    #[test]
    fn cond_id_closed_form_known_answer() {
        let mut s = BallCountScratch::with_capacity(12);
        s.tau = 0.5;
        for i in 0..12 {
            s.k_a[i] = 2;
            s.k_b[i] = 8;
            s.selected[i] = true;
        }
        let group: Vec<u32> = (0..10).collect();
        let d = cond_id(&s, &group).expect("estimate");
        assert!((d - 2.0).abs() < 1e-6, "d̂ = {d}, expected 2.0");

        // Mixed group: 8 rows at (2, 8), 2 rows at (4, 8) → k̄_A = 2.4.
        s.k_a[8] = 4;
        s.k_a[9] = 4;
        let d = cond_id(&s, &group).expect("estimate");
        let expect = (2.4f64.ln() - 8.0f64.ln()) / 0.5f64.ln();
        assert!(
            (d - expect as f32).abs() < 1e-4,
            "d̂ = {d}, expected {expect}"
        );
    }

    /// 12 collinear points (dim 1), m = 2: edge point 0 has r_B = 2
    /// (k_b = 2, k_a = 1 at τ = 0.9); interior point 5 has r_B = 1
    /// (k_b = 2, k_a = 0). τ = 0.1 empties every inner ball → the
    /// DegenerateCounts refusal, never a silent ±inf.
    #[test]
    fn ball_counts_tiny_geometry_and_degenerate_arm() {
        let pts: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let all: Vec<u32> = (0..12).collect();

        let mut s = BallCountScratch::with_capacity(12);
        ball_counts(&pts, 12, 1, &all, 0.9, &mut s).expect("counts");
        assert_eq!(s.r2_b[0], 4.0);
        assert_eq!(s.k_b[0], 2);
        assert_eq!(s.k_a[0], 1);
        assert_eq!(s.r2_b[5], 1.0);
        assert_eq!(s.k_b[5], 2);
        assert_eq!(s.k_a[5], 0);

        let mut s = BallCountScratch::with_capacity(12);
        ball_counts(&pts, 12, 1, &all, 0.1, &mut s).expect("counts");
        assert_eq!(s.k_a[0], 0);
        assert_eq!(s.k_a[5], 0);
        let err = cond_id(&s, &all).expect_err("degenerate");
        match err {
            CondGeometryError::DegenerateCounts { ka_bar, kb_bar } => {
                assert_eq!(ka_bar, 0.0);
                assert!(kb_bar > 0.0);
            }
            other => panic!("expected DegenerateCounts, got {other:?}"),
        }
    }

    #[test]
    fn nonfinite_refused_before_any_geometry() {
        let mut pts: Vec<f32> = (0..12).map(|i| i as f32).collect();
        pts[7] = f32::NAN;
        let all: Vec<u32> = (0..12).collect();
        let mut s = BallCountScratch::with_capacity(12);
        let err = ball_counts(&pts, 12, 1, &all, 0.9, &mut s).expect_err("nan");
        assert_eq!(err, CondGeometryError::NonFinite { at: 7 });

        // cond_imbalance: the scan is chained from ‖ to, so a NaN in `to`
        // at flat 5 reports at 12 + 5 — the index names the buffer.
        let from: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let mut to = from.clone();
        let mut is = ImbalanceScratch::with_capacity(12);
        to[5] = f32::INFINITY;
        let err = cond_imbalance(&from, &to, 12, 1, &all, &mut is).expect_err("inf");
        assert_eq!(err, CondGeometryError::NonFinite { at: 17 });
    }

    #[test]
    fn subset_too_small_refused() {
        let pts: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let all: Vec<u32> = (0..12).collect();
        let mut s = BallCountScratch::with_capacity(12);
        ball_counts(&pts, 12, 1, &all, 0.9, &mut s).expect("counts");

        let small: Vec<u32> = vec![0, 1, 2];
        let err = cond_id(&s, &small).expect_err("too small");
        assert_eq!(
            err,
            CondGeometryError::SubsetTooSmall {
                got: 3,
                min: MIN_GROUP_TOKENS
            }
        );
        let err = cond_id(&s, &[]).expect_err("empty");
        assert_eq!(
            err,
            CondGeometryError::SubsetTooSmall {
                got: 0,
                min: MIN_GROUP_TOKENS
            }
        );
        let err = cond_imbalance(&pts, &pts, 12, 1, &small, &mut is_new()).expect_err("too small");
        assert_eq!(
            err,
            CondGeometryError::SubsetTooSmall {
                got: 3,
                min: MIN_GROUP_TOKENS
            }
        );
    }

    fn is_new() -> ImbalanceScratch {
        ImbalanceScratch::with_capacity(12)
    }

    #[test]
    fn scratch_not_computed_refused() {
        let pts: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let all: Vec<u32> = (0..12).collect();

        // Never stuffed.
        let fresh = BallCountScratch::with_capacity(12);
        let err = cond_id(&fresh, &all).expect_err("fresh");
        assert_eq!(err, CondGeometryError::ScratchNotComputed);

        // Stuffed, but the queried row was never selected.
        let mut s = BallCountScratch::with_capacity(12);
        ball_counts(&pts, 12, 1, &all[..10], 0.9, &mut s).expect("counts");
        let err = cond_id(&s, &all).expect_err("unselected row");
        assert_eq!(err, CondGeometryError::ScratchNotComputed);
    }

    /// Duplicates are counted as-is (D4b): three identical rows put the
    /// zero distance in BOTH ball counts and collapse the m-order radius
    /// to 0 for the duplicate carrier.
    #[test]
    fn duplicates_counted_as_is() {
        let mut pts: Vec<f32> = (0..12).map(|i| i as f32).collect();
        pts[1] = 0.5;
        pts[2] = 0.5;
        pts[3] = 0.5;
        let all: Vec<u32> = (0..12).collect();
        let mut s = BallCountScratch::with_capacity(12);
        ball_counts(&pts, 12, 1, &all, 0.9, &mut s).expect("counts");
        assert_eq!(
            s.r2_b[1], 0.0,
            "collapse signal: 2 zero distances = m-order"
        );
        assert_eq!(s.k_b[1], 2);
        assert_eq!(s.k_a[1], 2, "duplicates counted as-is in the inner ball");
    }

    /// Determinism ×3: identical inputs → bit-identical counts, radii and
    /// estimates (integers assert equal; floats assert bit-equal).
    #[test]
    fn determinism_x3() {
        let pts: Vec<f32> = (0..24).map(|i| (i as f32) * 0.5).collect();
        let all: Vec<u32> = (0..24).collect();
        let group: Vec<u32> = (4..20).collect();

        let mut ref_counts = None;
        let mut ref_d = None;
        let mut ref_delta = None;
        for _ in 0..3 {
            let mut s = BallCountScratch::with_capacity(24);
            ball_counts(&pts, 24, 1, &all, 0.9, &mut s).expect("counts");
            let d = cond_id(&s, &group).expect("estimate");
            let mut is = ImbalanceScratch::with_capacity(24);
            let delta = cond_imbalance(&pts, &pts, 24, 1, &group, &mut is).expect("imbalance");
            match (&ref_counts, &ref_d, &ref_delta) {
                (None, _, _) => {
                    ref_counts = Some(s);
                    ref_d = Some(d);
                    ref_delta = Some(delta);
                }
                (Some(rc), Some(rd), Some(rdelta)) => {
                    assert_eq!(rc.k_a, s.k_a);
                    assert_eq!(rc.k_b, s.k_b);
                    for (a, b) in rc.r2_b.iter().zip(s.r2_b.iter()) {
                        assert_eq!(a.to_bits(), b.to_bits());
                    }
                    assert_eq!(rd.to_bits(), d.to_bits());
                    assert_eq!(rdelta.to_bits(), delta.to_bits());
                }
                _ => unreachable!(),
            }
        }
    }

    /// The known-answer imbalance (hand-computed, n = 12): from = grid
    /// 0..11, to = the interleave permutation [0,6,1,7,2,8,3,9,4,10,5,11].
    /// First-index tie rule pins every from-NN at i−1 (i ≥ 1); the rank
    /// interval |t[k]−t[i]| < r* is TWO-SIDED — per-point ranks
    /// [5,10,5,9,6,8,7,7,8,6,8,5] sum to 84 over denominator 6 →
    /// Δ = 84/72 = 14/12. Preservation control: to = from → every rank 0
    /// → Δ = 0.0 exact. Asymmetry: the reversed call differs.
    #[test]
    fn cond_imbalance_known_answer() {
        let from: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = [
            0.0f32, 6.0, 1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0,
        ];
        let to: Vec<f32> = t.to_vec();
        let all: Vec<u32> = (0..12).collect();
        let mut is = ImbalanceScratch::with_capacity(12);

        let delta = cond_imbalance(&from, &to, 12, 1, &all, &mut is).expect("imbalance");
        assert!((delta - 14.0 / 12.0).abs() < 1e-5, "Δ = {delta}");

        // Full-map pins: first-index tie (neighbors 1 and 3 both at d = 1
        // from point 2), and the two hand-computed ranks.
        let expect_ranks = [5u32, 10, 5, 9, 6, 8, 7, 7, 8, 6, 8, 5];
        assert_eq!(is.ranks_to[..], expect_ranks[..], "the rank table");
        assert_eq!(is.nn_from[2], 1);
        assert_eq!(is.nn_from[0], 1);

        // Preservation control: an isometric copy preserves every
        // neighborhood → Δ = 0.0 exact.
        let mut is = ImbalanceScratch::with_capacity(12);
        let zero = cond_imbalance(&from, &from, 12, 1, &all, &mut is).expect("imb");
        assert_eq!(zero, 0.0);

        // Asymmetry: Δ(A→B) ≠ Δ(B→A), both finite.
        let mut is = ImbalanceScratch::with_capacity(12);
        let back = cond_imbalance(&to, &from, 12, 1, &all, &mut is).expect("imb");
        assert!(back.is_finite());
        assert_ne!(back.to_bits(), delta.to_bits());
    }
}
