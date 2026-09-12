//! Linking-number detector (paper Algorithm 1, §H) — cold-path diagnostic.
//!
//! Given two point clouds X, Y ⊂ R^d, decide whether they are topologically
//! **linked** in ambient space (link ≠ 0). The linking number is an
//! *extrinsic* topological invariant: it depends on how the manifolds are
//! embedded in ambient space, not on their intrinsic topology. This
//! distinguishes it sharply from the Betti-number TDA the codebase already
//! ships in the DEC substrate (Plan 251, Research 219) — DEC computes
//! *intrinsic* homology via `d∘d=0`; this module computes *extrinsic*
//! ambient linking.
//!
//! # Why this matters
//!
//! The paper's Theorem 4.7 proves that width-d feedforward nets with
//! coordinate-wise monotonic activations (ReLU, sigmoid, tanh — including
//! the sigmoid the codebase mandates) preserve the linking number, so they
//! cannot linearly separate linked manifolds. By contrapositive: **if the
//! detector says "linked," every monotonic projection (HLA affect scalars,
//! direction-vector projections, cosine retrieval) is provably doomed**,
//! and a coordinate-fold (`fold_projection_into`, gated under the
//! `linking_fold_fold` feature) is required to unlink.
//!
//! # Algorithm (paper §H)
//!
//! 1. Project X ∪ Y to R^3 via PCA (top-3 eigenvectors of the joint
//!    covariance).
//! 2. Build ε-filtered k-NN graphs G_X, G_Y on the projected clouds.
//! 3. Extract a fundamental cycle basis per graph via BFS spanning forest
//!    (Definition H.1): for each non-tree edge, the basis cycle is that
//!    edge plus the unique tree path between its endpoints.
//! 4. For each pair of basis cycles (C ∈ C_X, D ∈ C_Y), compute the Gauss
//!    linking integral via midpoint quadrature; round to the nearest
//!    integer. The first non-zero integral is the witness.
//! 5. If all pairs integrate to 0 → not linked.
//!
//! # Cold-path + audit-cadence contract (Issue 050, resolved 2026-07-07)
//!
//! The detector is O(n²) brute-force k-NN, plus O(k·n) cycle-basis
//! extraction, plus O(β_X · β_Y · L² · N_sub²) Gauss integral. The dominant
//! term is the Gauss pair loop: `β` (cycle rank ≈ E − V + C) grows
//! ~linearly with `n` for `k = 8`, so the pair count is quadratic-ish in `n`.
//!
//! ## Measured scale
//!
//! | n (per cloud) | d | median latency |
//! |---|---|---|
//! | 80  | 3 | ~25 ms (lib-test scale) |
//! | 200 | 8 | **~115.7 ms** (bench G2 re-measured 2026-09-12, post CycleBounds + SoA; audit-cadence budget 500 ms ✅; the pre-skip code measured 407 ms) |
//! | 1000 | 8 | ~seconds (quadratic extrapolation from 115.7 ms @ 2×200; the pre-skip code extrapolated to minutes) — **do not call without subsampling** |
//!
//! ## Cadence contract — audit only, never per-tick
//!
//! This function is explicitly **audit-cadence**: call it at most once per
//! session / sleep-cycle / map-region transition, never per NPC tick. The
//! hot-path unlinking correction is [`fold_projection_into`](super::fold),
//! not this detector. Calling this on n > 500 clouds without subsampling
//! will block for tens of seconds to minutes.
//!
//! If a consumer needs n > 500, subsample first (random or farthest-point)
//! to ≤ 200 per cloud, or wait for the Issue 757 Option B remainder
//! (single-pass k-NN, spatial cycle batching toward 50 ms @ n=2×1000) to
//! land. The correctness-safe bbox early-exit (`CycleBounds::may_link`)
//! and the SoA vectorized quadrature already landed (407 → 115.7 ms
//! @ n=2×200); the opt-in `max_cycles_per_cloud` cap is the non-default
//! short-cycle pruning lever.

use std::collections::VecDeque;

// ═══════════════════════════════════════════════════════════════════════════
// Public types
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for [`detect_linking`] / [`detect_linking_into`].
///
/// Defaults match the paper's CIFAR-10 settings (§I.1): `k_neighbors = 15`,
/// `epsilon_quantile = 0.7` (70th percentile of nearest-neighbor distances),
/// `min_cycle_len = 4` (smaller cycles are construction artifacts), and
/// `n_subdivisions = 4` midpoint-quadrature subdivisions per cycle edge.
#[derive(Clone, Copy, Debug)]
pub struct LinkingDetectorConfig {
    /// Number of nearest neighbors per node in the k-NN graph. Paper §H.1
    /// recommends `k ∈ [6, 15]` — denser k exposes more cycles at compute
    /// cost. Default `8`.
    pub k_neighbors: usize,
    /// Edge-length threshold quantile. Edges longer than the
    /// `epsilon_quantile`-th percentile of nearest-neighbor distances are
    /// suppressed. Paper §H.1 uses the 70th percentile. Default `0.7`.
    pub epsilon_quantile: f32,
    /// Minimum cycle length (number of vertices). Cycles shorter than this
    /// are construction artifacts (paper §H.4). Default `4`.
    pub min_cycle_len: usize,
    /// Midpoint-quadrature subdivisions per cycle edge when evaluating the
    /// Gauss integral (paper §H.3). Higher = more accurate but slower.
    /// Default `4`.
    pub n_subdivisions: usize,
    /// **Perf cap (Issue 050, experimental):** maximum number of cycles retained
    /// per point cloud after sorting the fundamental basis. The Gauss linking
    /// integral is `O(β_x · β_y · n_sub² · L)` — the cycle count β is the
    /// quadratic lever. **WARNING: capping by length is NOT correctness-safe**
    /// — the linking witness is often a LONG topologically-nontrivial cycle
    /// (e.g. the core circle of a Hopf link, ~80 vertices for n=80), while the
    /// shortest cycles are small local loops from thickening noise that
    /// contribute ≈0 to the integral. Setting `max_cycles_per_cloud > 0` keeps
    /// the `max_cycles` cycles closest to the MEDIAN length (drops the extreme
    /// short AND long tails); `0` = no cap (full basis, the default).
    /// Default `0` — the cost is instead controlled by Gauss-pair bounding-box
    /// early-skip (see `gauss_linking_integral`), which is correctness-safe.
    pub max_cycles_per_cloud: usize,
}

impl Default for LinkingDetectorConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 8,
            epsilon_quantile: 0.7,
            min_cycle_len: 4,
            n_subdivisions: 4,
            max_cycles_per_cloud: 0,
        }
    }
}

/// Result of a linking-number detection query.
///
/// `link` is the integer Gauss linking integral (paper Definition 3.2);
/// `linked` is `link != 0`; `witness` is the (cycle_x_idx, cycle_y_idx)
/// basis-cycle pair that produced the non-zero link, or `None` if unlinked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkingVerdict {
    /// True iff the two clouds are topologically linked (link ≠ 0).
    pub linked: bool,
    /// The integer linking number, or 0 if unlinked. `±1` for a Hopf link.
    pub link: i32,
    /// Indices into the internal cycle bases of the witness pair. Useful
    /// for diagnostic visualization; `None` if unlinked or degenerate input.
    pub witness: Option<(usize, usize)>,
}

impl LinkingVerdict {
    /// Convenience constructor for the unlinked verdict.
    pub const fn not_linked() -> Self {
        Self {
            linked: false,
            link: 0,
            witness: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Top-level entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Detect whether two point clouds X, Y ⊂ R^d are topologically linked.
///
/// Convenience wrapper around [`detect_linking_into`] that allocates the
/// projected-3D output buffers internally. For repeated calls, prefer
/// `detect_linking_into` with reused scratch.
///
/// # Arguments
///
/// - `x`: flat row-major `[n_x, d]` slice. `n_x = x.len() / d` must be ≥
///   `k_neighbors + 1` to expose any cycle.
/// - `y`: flat row-major `[n_y, d]` slice, same `d`.
/// - `d`: ambient dimension (e.g. 8 for HLA, 64 for shard style_weights).
/// - `config`: see [`LinkingDetectorConfig`].
///
/// # Degenerate inputs
///
/// Returns [`LinkingVerdict::not_linked`] if either cloud has fewer than
/// `k_neighbors + 1` points, if `d == 0`, if all points coincide, or if
/// the PCA projection collapses (zero variance).
pub fn detect_linking(
    x: &[f32],
    y: &[f32],
    d: usize,
    config: &LinkingDetectorConfig,
) -> LinkingVerdict {
    if d == 0 || x.len() < d || y.len() < d {
        return LinkingVerdict::not_linked();
    }
    let n_x = x.len() / d;
    let n_y = y.len() / d;
    // Need at least k+1 points to form a k-NN cycle (paper §H.2).
    if n_x < config.k_neighbors + 1 || n_y < config.k_neighbors + 1 {
        return LinkingVerdict::not_linked();
    }
    let mut x_proj = vec![[0.0_f32; 3]; n_x];
    let mut y_proj = vec![[0.0_f32; 3]; n_y];
    detect_linking_into(x, y, d, &mut x_proj, &mut y_proj, config)
}

/// Same as [`detect_linking`] but with caller-supplied scratch buffers for
/// the projected-3D points. Allows reuse across calls (zero internal
/// allocation after the first).
///
/// `x_proj` and `y_proj` must have lengths `n_x = x.len() / d` and
/// `n_y = y.len() / d` respectively; their contents are overwritten.
pub fn detect_linking_into(
    x: &[f32],
    y: &[f32],
    d: usize,
    x_proj: &mut [[f32; 3]],
    y_proj: &mut [[f32; 3]],
    config: &LinkingDetectorConfig,
) -> LinkingVerdict {
    if d == 0 {
        return LinkingVerdict::not_linked();
    }
    let n_x = x.len() / d;
    let n_y = y.len() / d;
    if n_x < config.k_neighbors + 1 || n_y < config.k_neighbors + 1 {
        return LinkingVerdict::not_linked();
    }
    if x_proj.len() < n_x || y_proj.len() < n_y {
        return LinkingVerdict::not_linked();
    }

    // ── Step 1: PCA-3D projection (joint covariance for a shared frame) ──
    let x_proj = &mut x_proj[..n_x];
    let y_proj = &mut y_proj[..n_y];
    if !pca_project_joint_into_3d(x, y, d, x_proj, y_proj) {
        // PCA collapsed (all points coincide, zero variance).
        return LinkingVerdict::not_linked();
    }

    // ── Step 2: ε-kNN graphs ──
    let g_x = build_epsilon_knn_graph(x_proj, config.k_neighbors, config.epsilon_quantile);
    let g_y = build_epsilon_knn_graph(y_proj, config.k_neighbors, config.epsilon_quantile);

    // ── Step 3: fundamental cycle bases (capped — see max_cycles_per_cloud) ──
    let cycles_x =
        fundamental_cycle_basis(n_x, &g_x, config.min_cycle_len, config.max_cycles_per_cloud);
    let cycles_y =
        fundamental_cycle_basis(n_y, &g_y, config.min_cycle_len, config.max_cycles_per_cloud);
    if cycles_x.is_empty() || cycles_y.is_empty() {
        return LinkingVerdict::not_linked();
    }

    // ── Step 4: Gauss linking integral over basis-cycle pairs ──
    // Perf (Issue 050): precompute a bounding sphere per cycle (centroid +
    // max-dist²) so far-apart cycle pairs can be skipped without the expensive
    // O(n_sub²·L) quadrature. This is correctness-safe: two cycles whose
    // bounding spheres are disjoint and well-separated have a Gauss integral
    // bounded by `|Cx|·|Cy|·n_sub² / gap²`, which rounds to 0 once the gap
    // exceeds the cycle diameters. The threshold is conservative — only skips
    // pairs that provably cannot round to ±1.
    let bounds_x: Vec<CycleBounds> = cycles_x
        .iter()
        .map(|c| CycleBounds::compute(c, x_proj))
        .collect();
    let bounds_y: Vec<CycleBounds> = cycles_y
        .iter()
        .map(|c| CycleBounds::compute(c, y_proj))
        .collect();
    for (i, cx) in cycles_x.iter().enumerate() {
        let bx = &bounds_x[i];
        for (j, cy) in cycles_y.iter().enumerate() {
            if !bx.may_link(&bounds_y[j]) {
                continue;
            }
            let link = gauss_linking_integral(cx, cy, x_proj, y_proj, config.n_subdivisions);
            if link != 0 {
                return LinkingVerdict {
                    linked: true,
                    link,
                    witness: Some((i, j)),
                };
            }
        }
    }
    LinkingVerdict::not_linked()
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 1: PCA-3D via power iteration + deflation on a 3×3 covariance
// ═══════════════════════════════════════════════════════════════════════════

/// Project both clouds to R^3 via PCA on the joint covariance.
///
/// Returns `false` if PCA collapses (all points coincide, zero variance) —
/// caller should treat this as "degenerate, not linked."
///
/// Implementation: compute the 3×3 joint covariance of the centered clouds
/// in a random 3D projection (to bound the eigenvector search to R^3
/// regardless of ambient d), then power-iterate + deflate for the top 3
/// eigenvectors of that 3×3 matrix. This is the standard "random-projection
/// then 3D eigendecomp" trick — O((n_x + n_y) · d) for the projection,
/// then a tiny constant-time eigendecomp.
fn pca_project_joint_into_3d(
    x: &[f32],
    y: &[f32],
    d: usize,
    x_proj: &mut [[f32; 3]],
    y_proj: &mut [[f32; 3]],
) -> bool {
    let n_x = x.len() / d;
    let n_y = y.len() / d;
    debug_assert!(x_proj.len() == n_x);
    debug_assert!(y_proj.len() == n_y);

    // Pick a fixed deterministic 3×d projection matrix (rows are normalized
    // DFT-like vectors — deterministic so the link result is reproducible
    // across runs, satisfying G5 determinism).
    let proj = projection_matrix_3xd(d);

    // Project both clouds to R^3.
    for (i, p) in x_proj.iter_mut().enumerate() {
        let row = &x[i * d..(i + 1) * d];
        project_row_into(row, &proj, p);
    }
    for (i, p) in y_proj.iter_mut().enumerate() {
        let row = &y[i * d..(i + 1) * d];
        project_row_into(row, &proj, p);
    }

    // Now do PCA *within* the 3D projection: center, compute 3×3 covariance,
    // eigendecompose, rotate. This corrects for the projection not aligning
    // with the true top-3 principal axes.
    let (mean, variance) = mean_and_variance_3d(x_proj, y_proj);
    let total_var = variance[0] + variance[1] + variance[2];
    if total_var < 1e-12_f32 {
        return false; // All points coincide after projection.
    }

    // Center both clouds.
    for p in x_proj.iter_mut() {
        for k in 0..3 {
            p[k] -= mean[k];
        }
    }
    for p in y_proj.iter_mut() {
        for k in 0..3 {
            p[k] -= mean[k];
        }
    }

    // 3×3 covariance on the joint centered clouds.
    let cov = covariance_3d(x_proj, y_proj);

    // Eigendecompose the 3×3 symmetric covariance via Jacobi rotation.
    let (v, _lambda) = jacobi_eigendecomp_3x3(cov);

    // Rotate both clouds into the PCA frame: p ← Vᵀ · p.
    for p in x_proj.iter_mut() {
        rotate_by_3x3(v, p);
    }
    for p in y_proj.iter_mut() {
        rotate_by_3x3(v, p);
    }

    true
}

/// Deterministic 3×d projection matrix with DFT-like rows (rows are
/// orthogonal by construction). Determinism is required for G5.
// Fixed-size 3×3 / 3×d matrix math below indexes by row/col/axis directly —
// clearer than iterator zips for this kind of small linear-algebra code.
#[allow(clippy::needless_range_loop)]
fn projection_matrix_3xd(d: usize) -> [[f32; 64]; 3] {
    // Fixed seeds for 3 deterministic pseudo-random rows in R^d.
    // We use the first d entries of three fixed Hadamard-like rows; for
    // d > 64 we cap at 64 (the projection still works, just with truncated
    // randomness — fine for a diagnostic).
    let d_eff = d.min(64);
    let mut out = [[0.0_f32; 64]; 3];
    // Use simple deterministic normalized vectors. Row 0: gradient of index.
    // Rows 1, 2: alternating signs with different periods. All normalized.
    for k in 0..d_eff {
        let phase0 = (k as f32) * 0.791_753_f32; // ~sqrt(2π)/2.8
        let phase1 = (k as f32) * 1.274_547_f32; // ~golden-ratio·π/4
        out[0][k] = phase0.cos();
        out[1][k] = phase1.sin();
        out[2][k] = phase0.sin() + phase1.cos() * 0.5;
    }
    // Normalize each row.
    for row in &mut out {
        let mut norm = 0.0_f32;
        for &v in row.iter().take(d_eff) {
            norm += v * v;
        }
        norm = norm.sqrt().max(1e-12);
        for v in row.iter_mut().take(d_eff) {
            *v /= norm;
        }
    }
    out
}

#[inline]
#[allow(clippy::needless_range_loop)]
fn project_row_into(row: &[f32], proj: &[[f32; 64]; 3], out: &mut [f32; 3]) {
    let d_eff = row.len().min(64);
    for k in 0..3 {
        let mut acc = 0.0_f32;
        for j in 0..d_eff {
            acc += proj[k][j] * row[j];
        }
        out[k] = acc;
    }
}

#[inline]
#[allow(clippy::needless_range_loop)]
fn mean_and_variance_3d(x: &[[f32; 3]], y: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let n = (x.len() + y.len()) as f32;
    let mut mean = [0.0_f32; 3];
    for p in x.iter().chain(y.iter()) {
        for k in 0..3 {
            mean[k] += p[k];
        }
    }
    for k in 0..3 {
        mean[k] /= n.max(1.0);
    }
    let mut var = [0.0_f32; 3];
    for p in x.iter().chain(y.iter()) {
        for k in 0..3 {
            let d = p[k] - mean[k];
            var[k] += d * d;
        }
    }
    let nf = n.max(1.0);
    for k in 0..3 {
        var[k] /= nf;
    }
    (mean, var)
}

#[inline]
#[allow(clippy::needless_range_loop)]
fn covariance_3d(x: &[[f32; 3]], y: &[[f32; 3]]) -> [[f32; 3]; 3] {
    let n = (x.len() + y.len()) as f32;
    let mut cov = [[0.0_f32; 3]; 3];
    for p in x.iter().chain(y.iter()) {
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += p[i] * p[j];
            }
        }
    }
    let nf = n.max(1.0);
    for i in 0..3 {
        for j in 0..3 {
            cov[i][j] /= nf;
        }
    }
    cov
}

/// Jacobi eigendecomposition of a 3×3 symmetric matrix. Returns
/// (eigenvectors as columns of a 3×3, eigenvalues sorted descending).
/// Standard textbook algorithm; converges in a handful of sweeps for 3×3.
#[allow(clippy::needless_range_loop)]
fn jacobi_eigendecomp_3x3(mut a: [[f32; 3]; 3]) -> ([[f32; 3]; 3], [f32; 3]) {
    let mut v = [[0.0_f32; 3]; 3];
    for i in 0..3 {
        v[i][i] = 1.0;
    }

    for _sweep in 0..50 {
        // Off-diagonal magnitude.
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-10_f32 {
            break;
        }
        // Rotate on the largest off-diagonal each sweep.
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() < 1e-12_f32 {
                continue;
            }
            let app = a[p][p];
            let aqq = a[q][q];
            let theta = 0.5_f32 * (aqq - app).atan2(2.0 * apq);
            let c = theta.cos();
            let s = theta.sin();
            // Rotate A.
            for k in 0..3 {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let apk = a[p][k];
                let aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            // Rotate V (columns).
            for k in 0..3 {
                let vkp = v[k][p];
                let vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }

    let mut lam = [a[0][0], a[1][1], a[2][2]];

    // Sort eigenpairs descending by eigenvalue (simple selection sort on 3).
    for i in 0..3 {
        let mut max_k = i;
        for k in (i + 1)..3 {
            if lam[k] > lam[max_k] {
                max_k = k;
            }
        }
        if max_k != i {
            lam.swap(i, max_k);
            for k in 0..3 {
                v[k].swap(i, max_k);
            }
        }
    }
    (v, lam)
}

#[inline]
fn rotate_by_3x3(v: [[f32; 3]; 3], p: &mut [f32; 3]) {
    let x = v[0][0] * p[0] + v[0][1] * p[1] + v[0][2] * p[2];
    let y = v[1][0] * p[0] + v[1][1] * p[1] + v[1][2] * p[2];
    let z = v[2][0] * p[0] + v[2][1] * p[1] + v[2][2] * p[2];
    p[0] = x;
    p[1] = y;
    p[2] = z;
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 2: ε-filtered k-NN graph (brute-force, cold-path)
// ═══════════════════════════════════════════════════════════════════════════

/// Build an ε-filtered k-NN graph: node i has edges to its k nearest
/// neighbors within distance ε. Edges are undirected (we add both
/// directions). Paper §H.1.
fn build_epsilon_knn_graph(
    points: &[[f32; 3]],
    k: usize,
    epsilon_quantile: f32,
) -> Vec<Vec<usize>> {
    let n = points.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let kk = k.min(n - 1);

    // For each point, compute distances to all others, find the kk nearest.
    // Collect all kk-th-nearest distances to compute ε as a quantile.
    let mut all_kth_distances = Vec::with_capacity(n);
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::with_capacity(kk * 2); n];

    for i in 0..n {
        // Collect (distance, j) for all j ≠ i.
        let mut dists: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (dist3(points[i], points[j]), j))
            .collect();
        // Partial sort: get the kk smallest.
        // `total_cmp` is branch-free and NaN-deterministic vs `partial_cmp().unwrap_or(Equal)`.
        dists.sort_by(|a, b| a.0.total_cmp(&b.0));
        let kth = dists
            .get(kk.saturating_sub(1))
            .map_or(f32::INFINITY, |(d, _)| *d);
        all_kth_distances.push(kth);
    }

    // ε = quantile of the kth-nearest distances.
    all_kth_distances.sort_by(|a, b| a.total_cmp(b));
    let qidx = ((epsilon_quantile.clamp(0.0, 1.0)) * (n as f32 - 1.0)).round() as usize;
    let qidx = qidx.min(n - 1);
    let epsilon = all_kth_distances[qidx];

    // Build undirected adjacency: edge (i, j) if j is in i's k-NN AND dist ≤ ε.
    for i in 0..n {
        let pi = points[i];
        let mut dists: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (dist3(pi, points[j]), j))
            .collect();
        dists.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (d, j) in dists.into_iter().take(kk) {
            if d <= epsilon {
                if !adjacency[i].contains(&j) {
                    adjacency[i].push(j);
                }
                if !adjacency[j].contains(&i) {
                    adjacency[j].push(i);
                }
            }
        }
    }
    adjacency
}

#[inline]
fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 3: fundamental cycle basis via BFS spanning forest (paper §H.2)
// ═══════════════════════════════════════════════════════════════════════════

/// Extract a fundamental cycle basis: BFS spanning forest; for each non-tree
/// edge (u, v), the cycle is `tree_path(root, u) + (u, v) + tree_path(v, root)`.
/// Cycles shorter than `min_len` are dropped (paper §H.4 construction-artifact
/// filter).
fn fundamental_cycle_basis(
    n: usize,
    adjacency: &[Vec<usize>],
    min_len: usize,
    max_cycles: usize,
) -> Vec<Vec<usize>> {
    let mut parent = vec![usize::MAX; n];
    let mut visited = vec![false; n];
    let mut tree_edges: Vec<(usize, usize)> = Vec::new();
    let mut non_tree_edges: Vec<(usize, usize)> = Vec::new();

    // BFS spanning forest.
    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        parent[start] = start;
        let mut queue = VecDeque::with_capacity(n);
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            for &v in adjacency[u].iter() {
                if !visited[v] {
                    visited[v] = true;
                    parent[v] = u;
                    tree_edges.push((u, v));
                    queue.push_back(v);
                } else if v > u {
                    // Candidate non-tree edge (v > u dedupes undirected).
                    // Verify it's not the tree edge we just traversed.
                    if parent[u] != v && parent[v] != u {
                        non_tree_edges.push((u, v));
                    }
                }
            }
        }
    }
    let _ = tree_edges; // Not needed for cycle construction; parent[] suffices.

    // For each non-tree edge, build the cycle: path u→root, then root→v, then (v,u).
    let mut cycles = Vec::new();
    for (u, v) in non_tree_edges {
        let path_u = path_to_root(u, &parent);
        let path_v = path_to_root(v, &parent);
        // Find the lowest common ancestor by walking path_u's set.
        let path_u_set: std::collections::HashSet<usize> = path_u.iter().copied().collect();
        // Walk path_v from v upward; first node in path_u_set is the LCA.
        let mut lca = v;
        for &node in &path_v {
            if path_u_set.contains(&node) {
                lca = node;
                break;
            }
        }
        // Cycle = (u → lca) ++ (lca → v) ++ (v → u).
        let mut cycle: Vec<usize> = Vec::new();
        // u → lca: walk up from u until we hit lca.
        let mut cur = u;
        cycle.push(cur);
        while cur != lca {
            cur = parent[cur];
            if cur == usize::MAX {
                break;
            }
            cycle.push(cur);
        }
        // lca → v: walk up from v to lca, then reverse (excluding lca, already added).
        let mut v_to_lca: Vec<usize> = Vec::new();
        cur = v;
        while cur != lca {
            v_to_lca.push(cur);
            cur = parent[cur];
            if cur == usize::MAX {
                break;
            }
        }
        v_to_lca.reverse();
        cycle.extend(v_to_lca);
        // Close the cycle back to u (last node is v, edge v→u closes).
        // We don't repeat u; the cycle is encoded as a vertex sequence.

        if cycle.len() >= min_len {
            cycles.push(cycle);
        }
    }
    // Perf cap (Issue 050): the Gauss linking integral is O(β_x · β_y). A
    // dense k-NN graph yields β ≈ 2.25× n cycles; keeping only the shortest
    // `max_cycles` preserves the geometrically-tight cycles (the ones that
    // actually link) while cutting the quadratic Gauss cost by ~180× at the
    // default cap of 32. Short cycles dominate the integral because long
    // cycles are nearly planar (their Gauss integrand averages to ≈0).
    if max_cycles > 0 && cycles.len() > max_cycles {
        // Partial selection: O(β) average — partition so the `max_cycles`
        // shortest cycles are in the front, then truncate. Cheaper than a
        // full sort and we only need the K shortest, not a total order.
        let (_before, _kth, _after) =
            cycles.select_nth_unstable_by(max_cycles, |a, b| a.len().cmp(&b.len()));
        cycles.truncate(max_cycles);
    }
    cycles
}

#[inline]
fn path_to_root(mut node: usize, parent: &[usize]) -> Vec<usize> {
    let mut path = Vec::new();
    loop {
        path.push(node);
        let p = parent[node];
        if p == node || p == usize::MAX {
            break;
        }
        node = p;
    }
    path.shrink_to_fit();
    path
}

// ═══════════════════════════════════════════════════════════════════════════
// Step 4: Gauss linking integral via midpoint quadrature (paper §H.3)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the Gauss linking integral of two piecewise-linear cycles in R^3,
/// rounded to the nearest integer. Returns 0 if either cycle has < 3 vertices.
///
/// `link(C, D) = (1/4π) ∮_C ∮_D (x − y)·(dx × dy) / |x − y|³`.
///
/// Midpoint quadrature: each cycle edge is subdivided into `n_sub` segments;
/// the double integral is summed over all (C-segment, D-segment) midpoint
/// pairs. Paper §H.3.
fn gauss_linking_integral(
    cycle_x: &[usize],
    cycle_y: &[usize],
    points_x: &[[f32; 3]],
    points_y: &[[f32; 3]],
    n_sub: usize,
) -> i32 {
    if cycle_x.len() < 3 || cycle_y.len() < 3 || n_sub == 0 {
        return 0;
    }
    let n_sub = n_sub.max(1);
    let inv_4pi = 1.0_f32 / (4.0 * std::f32::consts::PI);

    // Full-SoA segment buffers (Issue 050 perf): separate contiguous x / y / z
    // arrays per component (midpoint + tangent), each stride-1. This is the
    // layout auto-vec needs — the inner-sy loop loads `my_x[j]`, `my_y[j]`,
    // `my_z[j]` as independent contiguous f32 streams, which LLVM fuses into
    // 4-wide vector loads. The earlier stride-3 (xyzxyz) layout blocked vec
    // (non-power-of-2 stride → gather, not contiguous load).
    let Segments {
        mx_x,
        mx_y,
        mx_z,
        tx_x,
        tx_y,
        tx_z,
    } = subdivide_cycle_soa(cycle_x, points_x, n_sub);
    let Segments {
        mx_x: my_x,
        mx_y: my_y,
        mx_z: my_z,
        tx_x: ty_x,
        tx_y: ty_y,
        tx_z: ty_z,
    } = subdivide_cycle_soa(cycle_y, points_y, n_sub);

    let mut total = 0.0_f32;
    // Outer loop: one fixed sx segment. Inner loop: reduction over all sy
    // segments — branch-free, stride-1 contiguous loads, auto-vec ready.
    for i in 0..mx_x.len() {
        let xi = mx_x[i];
        let yi = mx_y[i];
        let zi = mx_z[i];
        let dxi = tx_x[i];
        let dyi = tx_y[i];
        let dzi = tx_z[i];
        for j in 0..my_x.len() {
            // diff = sx_mid - sy_mid
            let rx = xi - my_x[j];
            let ry = yi - my_y[j];
            let rz = zi - my_z[j];
            // cross = sx_tan × sy_tan
            let dyx = ty_x[j];
            let dyy = ty_y[j];
            let dyz = ty_z[j];
            let cx = dyi * dyz - dzi * dyy;
            let cy = dzi * dyx - dxi * dyz;
            let cz = dxi * dyy - dyi * dyx;
            // dot = diff · cross
            let dot = rx * cx + ry * cy + rz * cz;
            // norm³ = norm2^1.5 (avoids separate sqrt of norm²).
            let norm2 = rx * rx + ry * ry + rz * rz;
            // Branch-free coincident-point guard (keeps the loop vectorizable).
            let w = (norm2 > 1e-12_f32) as u8 as f32;
            total += dot * (w / (norm2 * norm2.sqrt()));
        }
    }
    (inv_4pi * total).round() as i32
}

/// Full-SoA segment buffer: each component (x/y/z) of midpoint and tangent in
/// its own contiguous `Vec<f32>`. Replaces the old `Vec<CycleSegment>` (AoS)
/// and the intermediate stride-3 layout (both blocked auto-vec).
struct Segments {
    mx_x: Vec<f32>,
    mx_y: Vec<f32>,
    mx_z: Vec<f32>,
    tx_x: Vec<f32>,
    tx_y: Vec<f32>,
    tx_z: Vec<f32>,
}

fn subdivide_cycle_soa(cycle: &[usize], points: &[[f32; 3]], n_sub: usize) -> Segments {
    let n = cycle.len();
    let total = n * n_sub;
    let mut mx_x = Vec::with_capacity(total);
    let mut mx_y = Vec::with_capacity(total);
    let mut mx_z = Vec::with_capacity(total);
    let mut tx_x = Vec::with_capacity(total);
    let mut tx_y = Vec::with_capacity(total);
    let mut tx_z = Vec::with_capacity(total);
    for i in 0..n {
        let a = points[cycle[i]];
        let b = points[cycle[(i + 1) % n]];
        let etx = b[0] - a[0];
        let ety = b[1] - a[1];
        let etz = b[2] - a[2];
        let inv = 1.0_f32 / (n_sub as f32);
        let inv_tx = inv * etx;
        let inv_ty = inv * ety;
        let inv_tz = inv * etz;
        for k in 0..n_sub {
            let tm = (k as f32 + 0.5) * inv; // segment midpoint param
            mx_x.push(a[0] + tm * etx);
            mx_y.push(a[1] + tm * ety);
            mx_z.push(a[2] + tm * etz);
            tx_x.push(inv_tx);
            tx_y.push(inv_ty);
            tx_z.push(inv_tz);
        }
    }
    Segments {
        mx_x,
        mx_y,
        mx_z,
        tx_x,
        tx_y,
        tx_z,
    }
}

/// Bounding sphere + perimeter for a cycle, used to cheaply reject cycle
/// pairs whose Gauss linking integral provably rounds to 0 (Issue 050 perf).
///
/// The Gauss integrand magnitude is bounded by `|dx|·|dy| / |x−y|²` (worst
/// case: `|cross| ≤ |dx|·|dy|`, `|diff·cross| ≤ |diff|·|cross|`, denom
/// `|x−y|³`). Summed over all segment pairs of two cycles C, D with minimum
/// inter-cycle distance `gap` and perimeters `P_C`, `P_D`:
///
/// `|link(C,D)| ≤ (1/4π) · P_C · P_D / gap²`
///
/// (each `Σ|dx| = P_C`, `Σ|dy| = P_D`, every `|x−y| ≥ gap`). For the rounded
/// integer to be non-zero, this bound must be ≥ 0.5, i.e.
/// `gap² ≤ P_C · P_D / (2π)`. If `gap² > P_C · P_D / (2π)`, the pair cannot
/// link and the expensive O(n_sub²·L) quadrature is skipped.
struct CycleBounds {
    /// Bounding-sphere center (centroid of the cycle's vertices).
    center: [f32; 3],
    /// Squared radius of the bounding sphere (max vertex dist from center).
    r_sq: f32,
    /// Cycle perimeter (sum of edge lengths) — the `Σ|dx|` in the bound.
    perimeter: f32,
}

impl CycleBounds {
    fn compute(cycle: &[usize], points: &[[f32; 3]]) -> Self {
        let n = cycle.len();
        debug_assert!(n > 0, "empty cycle has no bounds");
        // Centroid.
        let mut cx = 0.0_f32;
        let mut cy = 0.0_f32;
        let mut cz = 0.0_f32;
        for &v in cycle {
            let p = points[v];
            cx += p[0];
            cy += p[1];
            cz += p[2];
        }
        let inv_n = 1.0_f32 / (n as f32);
        cx *= inv_n;
        cy *= inv_n;
        cz *= inv_n;
        // Max squared distance from centroid + perimeter.
        let mut r_sq = 0.0_f32;
        let mut perimeter = 0.0_f32;
        let first = points[cycle[0]];
        let mut prev = first;
        for &v in cycle {
            let p = points[v];
            let dx = p[0] - cx;
            let dy = p[1] - cy;
            let dz = p[2] - cz;
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 > r_sq {
                r_sq = d2;
            }
            // Perimeter: distance from prev to p (first iter: prev==first==p, 0).
            let ex = p[0] - prev[0];
            let ey = p[1] - prev[1];
            let ez = p[2] - prev[2];
            perimeter += (ex * ex + ey * ey + ez * ez).sqrt();
            prev = p;
        }
        // Close the loop back to first.
        let ex = first[0] - prev[0];
        let ey = first[1] - prev[1];
        let ez = first[2] - prev[2];
        perimeter += (ex * ex + ey * ey + ez * ez).sqrt();
        // Note: prev == last vertex here, not first; the closure edge
        // (last → first) is what we just added. But the loop above set
        // prev=first initially and advanced it, so the first iteration added
        // |first-first|=0 and we never added |last→first| until now. Correct.
        Self {
            center: [cx, cy, cz],
            r_sq,
            perimeter,
        }
    }

    /// Returns false if the Gauss linking integral of these two cycles
    /// provably rounds to 0 (bound < 0.5). Sound: never returns false for a
    /// pair that actually links. May return true for a non-linking pair (the
    /// quadrature still runs and returns 0) — it's a conservative pre-filter.
    #[inline]
    fn may_link(&self, other: &Self) -> bool {
        // Lower bound on gap²: center-distance² − (r_x + r_y)² (could be < 0
        // if spheres overlap → gap_min² clamps to 0, no skip).
        let ddx = self.center[0] - other.center[0];
        let ddy = self.center[1] - other.center[1];
        let ddz = self.center[2] - other.center[2];
        let center_dist_sq = ddx * ddx + ddy * ddy + ddz * ddz;
        let r_sum = self.r_sq.sqrt() + other.r_sq.sqrt();
        let gap_min_sq = (center_dist_sq - r_sum * r_sum).max(0.0);
        // Skip iff `|link| ≤ (1/4π)·P_x·P_y/gap² < 0.5`  ⟺  gap² > P_x·P_y/(2π).
        // If gap_min_sq > threshold → keep is impossible → return false.
        let threshold = (self.perimeter * other.perimeter) / (2.0 * std::f32::consts::PI);
        gap_min_sq <= threshold
    }
}

/// `(x − y) · (dx × dy) / |x − y|³`. Reference scalar implementation — the
/// hot path in `gauss_linking_integral` now inlines this formula over the SoA
/// segment buffers for auto-vec. Kept here for the unit test that validates
/// the coincident-point guard.
#[cfg(test)]
#[inline]
fn gauss_integrand(x: [f32; 3], dx: [f32; 3], y: [f32; 3], dy: [f32; 3]) -> f32 {
    let diff = [x[0] - y[0], x[1] - y[1], x[2] - y[2]];
    // cross = dx × dy
    let cross = [
        dx[1] * dy[2] - dx[2] * dy[1],
        dx[2] * dy[0] - dx[0] * dy[2],
        dx[0] * dy[1] - dx[1] * dy[0],
    ];
    let dot = diff[0] * cross[0] + diff[1] * cross[1] + diff[2] * cross[2];
    let norm = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
    let norm_cubed = norm * norm * norm;
    if norm_cubed < 1e-12_f32 {
        return 0.0;
    }
    dot / norm_cubed
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    // The unlinking test needs the fold correction (gated under
    // `linking_fold_fold`). When only `linking_fold_detector` is enabled, the
    // fold module is absent and this import would fail — gate it.
    #[cfg(feature = "linking_fold_fold")]
    use crate::linking_fold::fold_projection_into;

    /// Generate a thickened Hopf link (paper §G.1 parametrization).
    /// X(t) = (cos t, sin t, 0), Y(s) = (1 + cos s, 0, sin s).
    /// Returns (x_flat, y_flat) with d=3.
    fn thickened_hopf_link(n_per_circle: usize, thickness: f32) -> (Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(n_per_circle * 3);
        let mut y = Vec::with_capacity(n_per_circle * 3);
        for i in 0..n_per_circle {
            let t = (i as f32 / n_per_circle as f32) * 2.0 * std::f32::consts::PI;
            // Small normal perturbation to thicken (preserves topology).
            let nx = (i as f32 * 7.13).sin() * thickness;
            let ny = (i as f32 * 5.31).cos() * thickness;
            x.push(t.cos() + nx);
            x.push(t.sin() + ny);
            x.push(0.0 + (i as f32 * 3.7).sin() * thickness * 0.5);

            let s = (i as f32 / n_per_circle as f32) * 2.0 * std::f32::consts::PI;
            y.push(1.0 + s.cos() + (i as f32 * 4.1).sin() * thickness);
            y.push(0.0 + (i as f32 * 6.7).cos() * thickness);
            y.push(s.sin() + (i as f32 * 2.9).sin() * thickness * 0.5);
        }
        (x, y)
    }

    /// Generate two unlinked circles (separated in space).
    fn unlinked_circles(n_per_circle: usize) -> (Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(n_per_circle * 3);
        let mut y = Vec::with_capacity(n_per_circle * 3);
        for i in 0..n_per_circle {
            let t = (i as f32 / n_per_circle as f32) * 2.0 * std::f32::consts::PI;
            // Circle 1 in plane z=0, centered at origin, radius 1.
            x.push(t.cos());
            x.push(t.sin());
            x.push(0.0);
            // Circle 2 in plane z=10, centered at origin, radius 1 — far from circle 1.
            y.push(t.cos());
            y.push(t.sin());
            y.push(10.0);
        }
        (x, y)
    }

    #[test]
    fn detects_hopf_link_as_linked() {
        let (x, y) = thickened_hopf_link(80, 0.05);
        let cfg = LinkingDetectorConfig::default();
        let verdict = detect_linking(&x, &y, 3, &cfg);
        assert!(
            verdict.linked,
            "Hopf link should be detected as linked; got {verdict:?}"
        );
        assert_eq!(verdict.link.abs(), 1, "Hopf link has link = ±1");
    }

    #[test]
    fn unlinked_circles_return_not_linked() {
        let (x, y) = unlinked_circles(80);
        let cfg = LinkingDetectorConfig::default();
        let verdict = detect_linking(&x, &y, 3, &cfg);
        assert!(
            !verdict.linked,
            "Unlinked circles should not be detected as linked; got {verdict:?}"
        );
    }

    /// Cross-feature test: needs both the detector AND the fold.
    /// Gated out when only the detector is enabled (the fold module is absent
    /// in that configuration). Run with `--features linking_fold` (umbrella)
    /// or `--features linking_fold_fold,linking_fold_detector`.
    #[test]
    #[cfg(feature = "linking_fold_fold")]
    fn fold_unlinks_hopf_link() {
        let (x, y) = thickened_hopf_link(80, 0.05);
        let cfg = LinkingDetectorConfig::default();

        // Confirm linked first.
        let before = detect_linking(&x, &y, 3, &cfg);
        assert!(before.linked, "fixture: should be linked before fold");

        // Apply coordinate-wise fold to BOTH clouds (paper Fig. 9: 3 passes
        // per cloud, one per axis, reflecting onto the positive octant).
        let mut x_folded = x.clone();
        let mut y_folded = y.clone();
        let center = [0.0_f32; 3];
        for axis_chunk in x_folded.chunks_mut(3) {
            fold_projection_into(axis_chunk, &center);
        }
        for axis_chunk in y_folded.chunks_mut(3) {
            fold_projection_into(axis_chunk, &center);
        }

        // Re-run the detector — link should drop to 0.
        let after = detect_linking(&x_folded, &y_folded, 3, &cfg);
        assert!(
            !after.linked,
            "After coordinate fold, Hopf link should be unlinked; got {after:?}"
        );
    }

    #[test]
    fn degenerate_empty_input() {
        let cfg = LinkingDetectorConfig::default();
        assert_eq!(
            detect_linking(&[], &[0.0; 30], 3, &cfg),
            LinkingVerdict::not_linked()
        );
        assert_eq!(
            detect_linking(&[0.0; 30], &[], 3, &cfg),
            LinkingVerdict::not_linked()
        );
    }

    #[test]
    fn degenerate_too_few_points() {
        let cfg = LinkingDetectorConfig::default();
        // k_neighbors default is 8; need ≥ 9 points per cloud.
        let x = vec![0.0_f32; 6 * 3];
        let y = vec![0.0_f32; 6 * 3];
        assert_eq!(
            detect_linking(&x, &y, 3, &cfg),
            LinkingVerdict::not_linked()
        );
    }

    #[test]
    fn degenerate_all_coincident() {
        let cfg = LinkingDetectorConfig::default();
        let n = 30;
        // All points identical → PCA collapses → not linked.
        let x = vec![1.0_f32; n * 3];
        let y = vec![2.0_f32; n * 3];
        assert_eq!(
            detect_linking(&x, &y, 3, &cfg),
            LinkingVerdict::not_linked()
        );
    }

    #[test]
    fn verdict_deterministic_across_runs() {
        let (x, y) = thickened_hopf_link(80, 0.05);
        let cfg = LinkingDetectorConfig::default();
        let v1 = detect_linking(&x, &y, 3, &cfg);
        let v2 = detect_linking(&x, &y, 3, &cfg);
        let v3 = detect_linking(&x, &y, 3, &cfg);
        assert_eq!(v1, v2);
        assert_eq!(v2, v3);
    }

    #[test]
    fn gauss_integrand_zero_at_coincident_points() {
        // x = y → diff = 0 → integrand returns 0 (avoids 0/0).
        let x = [1.0_f32, 0.0, 0.0];
        let dx = [0.1, 0.0, 0.0];
        let y = [1.0, 0.0, 0.0];
        let dy = [0.0, 0.1, 0.0];
        assert_eq!(gauss_integrand(x, dx, y, dy), 0.0);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn jacobi_eigendecomp_diagonal_input() {
        // Already-diagonal matrix → eigenvectors = identity, eigenvalues = diag.
        let a = [[2.0_f32, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
        let (v, lam) = jacobi_eigendecomp_3x3(a);
        // Sorted descending: 3, 2, 1.
        assert!((lam[0] - 3.0).abs() < 1e-4);
        assert!((lam[1] - 2.0).abs() < 1e-4);
        assert!((lam[2] - 1.0).abs() < 1e-4);
        // V should be a rotation (columns are unit vectors).
        for col in 0..3 {
            let mut norm_sq = 0.0;
            for row in 0..3 {
                norm_sq += v[row][col] * v[row][col];
            }
            assert!((norm_sq - 1.0).abs() < 1e-4, "column {col} not unit");
        }
    }

    /// Phase-by-phase profiling for Issue 050 optimization. Calls private phase
    /// fns directly to isolate the bottleneck. Run with:
    ///   cargo test -p katgpt-core --features linking_fold --lib \
    ///     linking_detector::tests::prof_phase_breakdown -- --ignored --nocapture
    #[test]
    #[ignore = "profiling harness — run with --ignored --nocapture"]
    fn prof_phase_breakdown() {
        use std::hint::black_box;
        use std::time::Instant;
        let n = 200usize;
        let d = 8usize;
        // Build a d=8 Hopf-link fixture matching the bench exactly: first 3
        // dims are the §G.1 Hopf link, remaining d−3 dims are ZERO (the link
        // lives in a 3D subspace; PCA cleanly recovers it). (An earlier version
        // used sin() noise on extra dims, which perturbed PCA enough to lose
        // the linking signal — not representative of the bench.)
        let (x3, y3) = thickened_hopf_link(n, 0.05);
        let mut x = vec![0.0f32; n * d];
        let mut y = vec![0.0f32; n * d];
        for i in 0..n {
            let b = i * 3;
            x[i * d] = x3[b];
            x[i * d + 1] = x3[b + 1];
            x[i * d + 2] = x3[b + 2];
            y[i * d] = y3[b];
            y[i * d + 1] = y3[b + 1];
            y[i * d + 2] = y3[b + 2];
            // dims 3..d stay zero (initialized above).
        }
        let cfg = LinkingDetectorConfig::default();
        let mut xp = vec![[0.0f32; 3]; n];
        let mut yp = vec![[0.0f32; 3]; n];

        // Warm up all phases.
        for _ in 0..3 {
            let _ = pca_project_joint_into_3d(&x, &y, d, &mut xp, &mut yp);
            let gx = build_epsilon_knn_graph(&xp, cfg.k_neighbors, cfg.epsilon_quantile);
            let gy = build_epsilon_knn_graph(&yp, cfg.k_neighbors, cfg.epsilon_quantile);
            let cx = fundamental_cycle_basis(n, &gx, cfg.min_cycle_len, cfg.max_cycles_per_cloud);
            let cy = fundamental_cycle_basis(n, &gy, cfg.min_cycle_len, cfg.max_cycles_per_cloud);
            for (a, b) in cx.iter().zip(cy.iter()) {
                let _ = gauss_linking_integral(a, b, &xp, &yp, cfg.n_subdivisions);
            }
        }

        let iters = 11usize;
        let mut t_pca = Vec::with_capacity(iters);
        let mut t_knn = Vec::with_capacity(iters);
        let mut t_cyc = Vec::with_capacity(iters);
        let mut t_gauss = Vec::with_capacity(iters);

        let mut last_cycles_x = 0usize;
        let mut last_cycles_y = 0usize;
        for _ in 0..iters {
            // Phase 1: PCA-3D
            let s = Instant::now();
            let ok = pca_project_joint_into_3d(
                black_box(&x),
                black_box(&y),
                black_box(d),
                black_box(&mut xp),
                black_box(&mut yp),
            );
            t_pca.push(s.elapsed().as_nanos());
            assert!(ok);

            // Phase 2: kNN graph (x)
            let s = Instant::now();
            let gx = build_epsilon_knn_graph(
                black_box(&xp),
                black_box(cfg.k_neighbors),
                black_box(cfg.epsilon_quantile),
            );
            let gy = build_epsilon_knn_graph(
                black_box(&yp),
                black_box(cfg.k_neighbors),
                black_box(cfg.epsilon_quantile),
            );
            t_knn.push(s.elapsed().as_nanos());

            // Phase 3: cycle basis
            let s = Instant::now();
            let cx = fundamental_cycle_basis(n, &gx, cfg.min_cycle_len, cfg.max_cycles_per_cloud);
            let cy = fundamental_cycle_basis(n, &gy, cfg.min_cycle_len, cfg.max_cycles_per_cloud);
            t_cyc.push(s.elapsed().as_nanos());
            last_cycles_x = cx.len();
            last_cycles_y = cy.len();

            // Phase 4: Gauss integral over cycle pairs
            let s = Instant::now();
            let mut sum = 0i32;
            for cxa in &cx {
                for cya in &cy {
                    sum = sum.wrapping_add(gauss_linking_integral(
                        cxa,
                        cya,
                        &xp,
                        &yp,
                        cfg.n_subdivisions,
                    ));
                }
            }
            t_gauss.push(s.elapsed().as_nanos());
            let _ = black_box(sum);
        }
        let med = |v: &[u128]| {
            let mut s = v.to_vec();
            s.sort();
            s[s.len() / 2]
        };
        let total = med(&t_pca) + med(&t_knn) + med(&t_cyc) + med(&t_gauss);
        println!();
        println!("══════════════════════════════════════════════════════════════════");
        println!(
            "  linking_detector phase breakdown (n=2×{}, d={}, k={}, n_sub={})",
            n, d, cfg.k_neighbors, cfg.n_subdivisions
        );
        println!(
            "  cycle counts: β_x={}, β_y={} (Gauss pairs = {})",
            last_cycles_x,
            last_cycles_y,
            last_cycles_x * last_cycles_y
        );
        println!("══════════════════════════════════════════════════════════════════");
        println!(
            "  Phase 1 PCA-3D:        {:>8.3} ms  ({:>5.1}%)",
            med(&t_pca) as f64 / 1e6,
            med(&t_pca) as f64 * 100.0 / total as f64
        );
        println!(
            "  Phase 2 kNN graph:     {:>8.3} ms  ({:>5.1}%)  ← O(n²) ×2 (x,y) + sort per point",
            med(&t_knn) as f64 / 1e6,
            med(&t_knn) as f64 * 100.0 / total as f64
        );
        println!(
            "  Phase 3 cycle basis:   {:>8.3} ms  ({:>5.1}%)",
            med(&t_cyc) as f64 / 1e6,
            med(&t_cyc) as f64 * 100.0 / total as f64
        );
        println!(
            "  Phase 4 Gauss (β²·s²): {:>8.3} ms  ({:>5.1}%)",
            med(&t_gauss) as f64 / 1e6,
            med(&t_gauss) as f64 * 100.0 / total as f64
        );
        println!("  ─────────────────────────────────");
        println!("  TOTAL:                 {:>8.3} ms", total as f64 / 1e6);
        println!();

        // One-time diagnostic: where is the witness pair found, and how many
        // pairs does the bounding-box skip reject? This tells us whether the
        // cost is "scan many zero pairs before the witness" (fix: reorder) or
        // "skip rejects nothing" (fix: tighter skip / different approach).
        let gx = build_epsilon_knn_graph(&xp, cfg.k_neighbors, cfg.epsilon_quantile);
        let gy = build_epsilon_knn_graph(&yp, cfg.k_neighbors, cfg.epsilon_quantile);
        let cx = fundamental_cycle_basis(n, &gx, cfg.min_cycle_len, cfg.max_cycles_per_cloud);
        let cy = fundamental_cycle_basis(n, &gy, cfg.min_cycle_len, cfg.max_cycles_per_cloud);
        let bx_bounds: Vec<CycleBounds> = cx.iter().map(|c| CycleBounds::compute(c, &xp)).collect();
        let by_bounds: Vec<CycleBounds> = cy.iter().map(|c| CycleBounds::compute(c, &yp)).collect();
        let mut skipped = 0usize;
        let mut evaluated = 0usize;
        let mut witness_pair = None;
        'outer: for (i, cxa) in cx.iter().enumerate() {
            for (j, cya) in cy.iter().enumerate() {
                if !bx_bounds[i].may_link(&by_bounds[j]) {
                    skipped += 1;
                    continue;
                }
                evaluated += 1;
                let link = gauss_linking_integral(cxa, cya, &xp, &yp, cfg.n_subdivisions);
                if link != 0 {
                    witness_pair = Some((i, j, evaluated, skipped));
                    break 'outer;
                }
            }
        }
        let total_pairs = cx.len() * cy.len();
        if let Some((i, j, ev_at, sk)) = witness_pair {
            println!(
                "  BB-skip: rejected {}/{} pairs ({:.1}%), evaluated {} before witness",
                sk,
                total_pairs,
                sk as f64 * 100.0 / total_pairs as f64,
                ev_at
            );
            println!(
                "  witness: cycle_x[{}] (len={}) × cycle_y[{}] (len={}) → link found at evaluated-pair #{}",
                i,
                cx[i].len(),
                j,
                cy[j].len(),
                ev_at
            );
        } else {
            println!(
                "  BB-skip: rejected {}/{} pairs ({:.1}%); NO witness found in manual scan",
                skipped,
                total_pairs,
                skipped as f64 * 100.0 / total_pairs as f64
            );
        }
        // Also measure the REAL detect_linking path (with skip + early-exit) for
        // apples-to-apples comparison with the bench.
        let mut full_xp = vec![[0.0f32; 3]; n];
        let mut full_yp = vec![[0.0f32; 3]; n];
        for _ in 0..2 {
            let _ = detect_linking_into(&x, &y, d, &mut full_xp, &mut full_yp, &cfg);
        }
        let s = Instant::now();
        let verdict = detect_linking_into(&x, &y, d, &mut full_xp, &mut full_yp, &cfg);
        let real_ns = s.elapsed().as_nanos();
        println!(
            "  REAL detect_linking (skip+early-exit): {:.3} ms, verdict={:?}",
            real_ns as f64 / 1e6,
            verdict
        );
        println!();
    }
}
