//! MAG transfer prediction: modelless ranking of candidate datasets/experiences
//! by predicted transfer to a target capability (arXiv:2607.04222 §4).
//!
//! The headline capability: given a target capability `T` and a pool of
//! candidate experiences/datasets, predict which candidate will most improve `T`
//! — **without** running any training. The paper achieves 94.7% Top-1 accuracy
//! on their 18-dataset corpus; raw centroid cosine achieves only ρ ≈ 0.03 (near
//! random). The class-conditional MAG metrics (cosine on the positive/negative
//! class centroids) carry the signal; raw cosine does not.
//!
//! This is the **directed curiosity** signal: not "what's novel?" (entropy) but
//! "what transfers to my goal?" (MAG geometry). Consumed by CGSP curiosity
//! routing and the AnyRAG escalation gateway in riir-ai / riir-neuron-db.

use super::types::{MagError, TransferMetric, check_dim, cosine};

// recos_sim_slice(_into) live in the `similarity` module (gated on `recos`).
// When the feature is off, `TransferMetric::Recos` falls back to centroid
// cosine — the closest available metric (recos degenerates to cosine for
// linear relationships).
#[cfg(feature = "recos")]
use crate::similarity::{recos_sim_slice, recos_sim_slice_into};

// ── Dataset view ───────────────────────────────────────────────────

/// A borrowed dataset view: activation samples + per-sample verdict labels.
///
/// Generic over sample storage (`Vec<f32>`, `[f32; D]`, `&[f32]`, etc.) via
/// `AsRef<[f32]>`. Labels correspond to the model/runtime's own verdict `y_M`
/// for each sample — the same unsupervised label source used by
/// [`mine_contrast_direction`](super::mining::mine_contrast_direction).
///
/// `labels.len()` must equal `activations.len()`; this is checked by
/// [`transfer_score`] and [`rank_candidates`].
#[derive(Debug, Clone, Copy)]
pub struct DataSet<'a, S: AsRef<[f32]>> {
    /// `N × d` activation readouts. These should already be operator-applied
    /// (via [`apply_operator`](super::mining::apply_operator)) if the caller
    /// wants a MAG-operator-based score rather than a raw-activation score.
    pub activations: &'a [S],
    /// Per-sample verdict labels (`true` = positive/benign class, `false` =
    /// negative/malicious class).
    pub labels: &'a [bool],
}

impl<'a, S: AsRef<[f32]>> DataSet<'a, S> {
    /// Construct a dataset view from activations + labels.
    #[inline]
    pub fn new(activations: &'a [S], labels: &'a [bool]) -> Self {
        Self {
            activations,
            labels,
        }
    }

    /// Number of samples.
    #[inline]
    pub fn len(&self) -> usize {
        self.activations.len()
    }

    /// Whether the dataset is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.activations.is_empty()
    }
}

// ── Single-metric transfer score ───────────────────────────────────

/// Compute the transfer score of `candidate` relative to `target` under `metric`.
///
/// All metrics return **higher = better predicted transfer** — distance-based
/// metrics are negated so that `0` = identical, negative = dissimilar. See
/// [`TransferMetric`] for the semantics of each.
///
/// The activation sets may have different sample counts but must share the same
/// per-sample dimensionality `d`. Labels must match the activation count in each
/// set.
///
/// # Example
///
/// ```
/// # use katgpt_core::mag::{DataSet, TransferMetric, transfer_score};
/// // Identical candidate and target → centroid cosine = 1.0 (perfect transfer).
/// let acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
/// let labels = [true, false, true];
/// let ds = DataSet::new(&acts, &labels);
/// let score = transfer_score(&ds, &ds, TransferMetric::CentroidCosine).unwrap();
/// assert!((score - 1.0).abs() < 1e-5);
/// ```
pub fn transfer_score<S: AsRef<[f32]>>(
    candidate: &DataSet<'_, S>,
    target: &DataSet<'_, S>,
    metric: TransferMetric,
) -> Result<f32, MagError> {
    let d = check_dim(candidate.activations)?;
    let d2 = check_dim(target.activations)?;
    if d != d2 {
        return Err(MagError::DimMismatch);
    }
    if candidate.labels.len() != candidate.activations.len()
        || target.labels.len() != target.activations.len()
    {
        return Err(MagError::DimMismatch);
    }

    match metric {
        TransferMetric::CentroidCosine => {
            let c = centroid(candidate.activations, d);
            let t = centroid(target.activations, d);
            Ok(cosine(&c, &t))
        }
        TransferMetric::Euclidean => {
            let c = centroid(candidate.activations, d);
            let t = centroid(target.activations, d);
            let mut dist_sq = 0.0;
            for j in 0..d {
                let diff = c[j] - t[j];
                dist_sq += diff * diff;
            }
            Ok(-dist_sq.sqrt())
        }
        TransferMetric::Correlation => {
            let c = centroid(candidate.activations, d);
            let t = centroid(target.activations, d);
            Ok(pearson(&c, &t))
        }
        TransferMetric::RbfMmd => {
            let gamma = 1.0 / d as f32;
            Ok(-rbf_mmd_sq(
                candidate.activations,
                target.activations,
                gamma,
            ))
        }
        TransferMetric::Wasserstein1d => {
            Ok(-wasserstein1d(candidate.activations, target.activations, d))
        }
        TransferMetric::CkaLinear => Ok(cka_linear(candidate.activations, target.activations, d)),
        TransferMetric::ClassConditionalCosineMalicious => {
            let c = class_centroid(candidate.activations, candidate.labels, false, d)?;
            let t = class_centroid(target.activations, target.labels, false, d)?;
            Ok(cosine(&c, &t))
        }
        TransferMetric::ClassConditionalCosineBenign => {
            let c = class_centroid(candidate.activations, candidate.labels, true, d)?;
            let t = class_centroid(target.activations, target.labels, true, d)?;
            Ok(cosine(&c, &t))
        }
        TransferMetric::Recos => {
            let c = centroid(candidate.activations, d);
            let t = centroid(target.activations, d);
            // Cold path: recos_sim_slice allocates (sort needs owned buffers).
            // Acceptable for MAG diagnostics; the zero-alloc variant is
            // `transfer_score_into`.
            #[cfg(feature = "recos")]
            {
                Ok(recos_sim_slice(&c, &t))
            }
            #[cfg(not(feature = "recos"))]
            {
                Ok(cosine(&c, &t))
            }
        }
    }
}

// ── Zero-alloc hot-path variant ────────────────────────────────────

/// Zero-alloc variant of [`transfer_score`] for centroid-based metrics.
///
/// Uses `scratch` (length `≥ 2·d`) for centroid computation, avoiding heap
/// allocation on the hot path. Supports the five centroid-based metrics
/// ([`CentroidCosine`](TransferMetric::CentroidCosine),
/// [`Euclidean`](TransferMetric::Euclidean),
/// [`Correlation`](TransferMetric::Correlation), the two
/// `ClassConditionalCosine*` variants, and [`Recos`](TransferMetric::Recos)).
/// The distribution-based metrics
/// ([`RbfMmd`](TransferMetric::RbfMmd),
/// [`Wasserstein1d`](TransferMetric::Wasserstein1d),
/// [`CkaLinear`](TransferMetric::CkaLinear)) fall back to the allocating
/// [`transfer_score`] — they are cold-path diagnostics.
///
/// [`Recos`](TransferMetric::Recos) sorts the scratch halves in place (the
/// scratch is scratch — contents are unspecified after this call).
///
/// Returns [`MagError::DimMismatch`] if `scratch.len() < 2 * d`.
pub fn transfer_score_into<S: AsRef<[f32]>>(
    candidate: &DataSet<'_, S>,
    target: &DataSet<'_, S>,
    metric: TransferMetric,
    scratch: &mut [f32],
) -> Result<f32, MagError> {
    let d = check_dim(candidate.activations)?;
    let d2 = check_dim(target.activations)?;
    if d != d2 {
        return Err(MagError::DimMismatch);
    }
    if candidate.labels.len() != candidate.activations.len()
        || target.labels.len() != target.activations.len()
    {
        return Err(MagError::DimMismatch);
    }
    if scratch.len() < 2 * d {
        return Err(MagError::DimMismatch);
    }

    match metric {
        TransferMetric::CentroidCosine => {
            let (sc, st) = scratch.split_at_mut(d);
            centroid_into(candidate.activations, d, sc);
            centroid_into(target.activations, d, st);
            Ok(cosine(sc, st))
        }
        TransferMetric::Euclidean => {
            let (sc, st) = scratch.split_at_mut(d);
            centroid_into(candidate.activations, d, sc);
            centroid_into(target.activations, d, st);
            let mut dist_sq = 0.0;
            for j in 0..d {
                let diff = sc[j] - st[j];
                dist_sq += diff * diff;
            }
            Ok(-dist_sq.sqrt())
        }
        TransferMetric::Correlation => {
            let (sc, st) = scratch.split_at_mut(d);
            centroid_into(candidate.activations, d, sc);
            centroid_into(target.activations, d, st);
            Ok(pearson(sc, st))
        }
        TransferMetric::ClassConditionalCosineMalicious => {
            let (sc, st) = scratch.split_at_mut(d);
            class_centroid_into(candidate.activations, candidate.labels, false, d, sc)?;
            class_centroid_into(target.activations, target.labels, false, d, st)?;
            Ok(cosine(sc, st))
        }
        TransferMetric::ClassConditionalCosineBenign => {
            let (sc, st) = scratch.split_at_mut(d);
            class_centroid_into(candidate.activations, candidate.labels, true, d, sc)?;
            class_centroid_into(target.activations, target.labels, true, d, st)?;
            Ok(cosine(sc, st))
        }
        TransferMetric::Recos => {
            // Compute centroids into the two scratch halves, then apply
            // recos (sorts both halves in place) or fall back to cosine.
            let (sc, st) = scratch.split_at_mut(d);
            centroid_into(candidate.activations, d, sc);
            centroid_into(target.activations, d, st);
            #[cfg(feature = "recos")]
            {
                Ok(recos_sim_slice_into(sc, st))
            }
            #[cfg(not(feature = "recos"))]
            {
                Ok(cosine(sc, st))
            }
        }
        // Distribution-based metrics fall back to the allocating path.
        TransferMetric::RbfMmd | TransferMetric::Wasserstein1d | TransferMetric::CkaLinear => {
            transfer_score(candidate, target, metric)
        }
    }
}

// ── Multi-candidate ranking ────────────────────────────────────────

/// A ranked candidate entry from [`rank_candidates`].
#[derive(Debug, Clone)]
pub struct RankEntry {
    /// Index into the input `candidates` slice.
    pub candidate_idx: usize,
    /// Mean percentile rank across all metrics (`0.0` = worst, `1.0` = best).
    pub mean_percentile: f32,
    /// Raw per-metric scores (same order as the input `metrics` slice).
    pub per_metric_scores: Vec<f32>,
}

/// Rank candidates by aggregate transfer score across multiple metrics.
///
/// For each metric, candidates are assigned a percentile rank (the fraction of
/// other candidates they outscore). The mean percentile rank across all metrics
/// is the aggregate score (arXiv:2607.04222 §4 protocol). The returned vec is
/// sorted by `mean_percentile` descending (best predicted transfer first).
///
/// This is the modelless "which experience teaches the most" scorer. At least
/// one candidate and one metric are required; otherwise returns
/// [`MagError::Empty`]. With a single candidate the percentile is trivially
/// `1.0`; the score is still useful as a per-metric diagnostic.
pub fn rank_candidates<S: AsRef<[f32]>>(
    candidates: &[DataSet<'_, S>],
    target: &DataSet<'_, S>,
    metrics: &[TransferMetric],
) -> Result<Vec<RankEntry>, MagError> {
    if candidates.is_empty() || metrics.is_empty() {
        return Err(MagError::Empty);
    }

    let n_cand = candidates.len();
    let n_metrics = metrics.len();

    // Compute raw scores: scores[i * n_metrics + m].
    let mut scores = Vec::with_capacity(n_cand * n_metrics);
    for cand in candidates {
        for &metric in metrics {
            scores.push(transfer_score(cand, target, metric)?);
        }
    }

    // Convert to percentile ranks per metric (fraction of candidates beaten).
    let mut percentiles = vec![0.0_f32; n_cand * n_metrics];
    // Hoisted outside the metric loop — clear() + reuse per metric instead
    // of `n_metrics` × allocations of `Vec<(f32, usize)>`.
    let mut ranked: Vec<(f32, usize)> = Vec::with_capacity(n_cand);
    for m in 0..n_metrics {
        ranked.clear();
        ranked.extend((0..n_cand).map(|i| (scores[i * n_metrics + m], i)));
        // total_cmp: transfer scores are non-NaN by construction (cosine,
        // euclidean, pearson-with-zero-denom-guard all return finite f32).
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
        let denom = (n_cand - 1).max(1) as f32;
        for (rank, &(_, idx)) in ranked.iter().enumerate() {
            percentiles[idx * n_metrics + m] = rank as f32 / denom;
        }
    }

    // Mean percentile per candidate.
    let mut entries: Vec<RankEntry> = (0..n_cand)
        .map(|i| {
            let mut sum = 0.0;
            let mut per_metric = Vec::with_capacity(n_metrics);
            for m in 0..n_metrics {
                sum += percentiles[i * n_metrics + m];
                per_metric.push(scores[i * n_metrics + m]);
            }
            RankEntry {
                candidate_idx: i,
                mean_percentile: sum / n_metrics as f32,
                per_metric_scores: per_metric,
            }
        })
        .collect();

    // mean_percentile is sum of [0,1] values / n_metrics → non-NaN by
    // construction. total_cmp is branch-free and NaN-deterministic.
    entries.sort_by(|a, b| b.mean_percentile.total_cmp(&a.mean_percentile));

    Ok(entries)
}

// ── Internal metric helpers ────────────────────────────────────────

/// Compute the centroid (per-dimension mean) of a sample set. Allocates `d` f32s.
fn centroid<S: AsRef<[f32]>>(samples: &[S], d: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; d];
    let inv_n = 1.0 / samples.len() as f32;
    for s in samples {
        let s = s.as_ref();
        for (acc, &v) in out.iter_mut().zip(s) {
            *acc += v * inv_n;
        }
    }
    out
}

/// Compute the centroid of samples whose label matches `target_class`.
fn class_centroid<S: AsRef<[f32]>>(
    samples: &[S],
    labels: &[bool],
    target_class: bool,
    d: usize,
) -> Result<Vec<f32>, MagError> {
    let mut out = vec![0.0_f32; d];
    let mut count = 0;
    for (s, &label) in samples.iter().zip(labels) {
        if label == target_class {
            let s = s.as_ref();
            for (acc, &v) in out.iter_mut().zip(s) {
                *acc += v;
            }
            count += 1;
        }
    }
    if count == 0 {
        return Err(MagError::EmptyClass);
    }
    let inv_n = 1.0 / count as f32;
    for x in out.iter_mut() {
        *x *= inv_n;
    }
    Ok(out)
}

/// Compute the centroid (per-dimension mean) into a pre-allocated buffer
/// `out[0..d]`. Zero-alloc hot-path helper.
fn centroid_into<S: AsRef<[f32]>>(samples: &[S], d: usize, out: &mut [f32]) {
    out[..d].fill(0.0);
    let inv_n = 1.0 / samples.len() as f32;
    for s in samples {
        let s = s.as_ref();
        for (acc, &v) in out[..d].iter_mut().zip(s) {
            *acc += v * inv_n;
        }
    }
}

/// Compute the class-conditional centroid into a pre-allocated buffer
/// `out[0..d]`. Zero-alloc hot-path helper.
fn class_centroid_into<S: AsRef<[f32]>>(
    samples: &[S],
    labels: &[bool],
    target_class: bool,
    d: usize,
    out: &mut [f32],
) -> Result<(), MagError> {
    out[..d].fill(0.0);
    let mut count = 0;
    for (s, &label) in samples.iter().zip(labels) {
        if label == target_class {
            let s = s.as_ref();
            for (acc, &v) in out[..d].iter_mut().zip(s) {
                *acc += v;
            }
            count += 1;
        }
    }
    if count == 0 {
        return Err(MagError::EmptyClass);
    }
    let inv_n = 1.0 / count as f32;
    for x in out[..d].iter_mut() {
        *x *= inv_n;
    }
    Ok(())
}

/// Pearson correlation between two equal-length vectors.
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for j in 0..a.len() {
        let da = a[j] - mean_a;
        let db = b[j] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom == 0.0 { 0.0 } else { cov / denom }
}

/// RBF-kernel MMD² between two sample sets (biased estimator).
///
/// `mmd² = (1/m²)ΣΣ k(x_i,x_j) + (1/n²)ΣΣ k(y_i,y_j) − (2/(mn))ΣΣ k(x_i,y_j)`
fn rbf_mmd_sq<S: AsRef<[f32]>>(x: &[S], y: &[S], gamma: f32) -> f32 {
    let m = x.len();
    let n = y.len();

    let mut sum_xx = 0.0;
    for i in 0..m {
        let xi = x[i].as_ref();
        for xj in x {
            sum_xx += rbf_kernel(xi, xj.as_ref(), gamma);
        }
    }
    let mut sum_yy = 0.0;
    for i in 0..n {
        let yi = y[i].as_ref();
        for yj in y {
            sum_yy += rbf_kernel(yi, yj.as_ref(), gamma);
        }
    }
    let mut sum_xy = 0.0;
    for xi in x {
        let xi = xi.as_ref();
        for yj in y {
            sum_xy += rbf_kernel(xi, yj.as_ref(), gamma);
        }
    }

    let mmd = sum_xx / (m * m) as f32 + sum_yy / (n * n) as f32 - 2.0 * sum_xy / (m * n) as f32;
    mmd.max(0.0) // MMD² is non-negative by construction (guard against f32 drift)
}

#[inline]
fn rbf_kernel(a: &[f32], b: &[f32], gamma: f32) -> f32 {
    let mut dist_sq = 0.0;
    for j in 0..a.len() {
        let diff = a[j] - b[j];
        dist_sq += diff * diff;
    }
    crate::simd::fast_exp(-gamma * dist_sq)
}

/// Ascending sort shared by the Wasserstein paths (same comparator the
/// metric has always used; finite inputs give a total order).
///
/// Deliberately the IN-PLACE unstable sort: std's comparator `sort_by`
/// allocates its merge buffer, which would break the probe's zero-alloc hot
/// path (Issue 697 G4). Determinism holds — the unstable sort is a
/// deterministic algorithm, and the metric reads only the sorted VALUE
/// sequence, in which equal `f32`s are bit-identical and interchangeable —
/// so the result is bit-identical to a stable sort's (pinned by
/// `wasserstein_matches_independent_reference`, whose in-test twin uses a
/// stable `total_cmp` sort).
fn sort_f32_ascending(v: &mut [f32]) {
    v.sort_unstable_by(|a, b| a.total_cmp(b));
}

/// Quantile-grid core shared by `wasserstein1d` and
/// `wasserstein1d_scalar_into`: both slices must already be sorted ascending.
/// Empirical quantile functions are compared on a common grid of
/// `max(m, n)` points (handles unequal sample counts via linear
/// interpolation).
fn wasserstein1d_sorted_core(sx: &[f32], sy: &[f32]) -> f32 {
    let t = sx.len().max(sy.len());
    let mut dim_dist = 0.0;
    for k in 0..t {
        let qx = quantile_interp(sx, (k as f32 + 0.5) / t as f32);
        let qy = quantile_interp(sy, (k as f32 + 0.5) / t as f32);
        dim_dist += (qx - qy).abs();
    }
    dim_dist / t as f32
}

/// 1D Wasserstein-1 distance averaged over dimensions.
///
/// For each dimension, both sets are sorted and their empirical quantile
/// functions are compared on a common grid of `max(m, n)` points (handles
/// unequal sample counts via linear interpolation). Behavior-identical to
/// the pre-Issue-697 form (the per-dimension math moved verbatim into
/// `wasserstein1d_sorted_core`; the columns are gathered into reused
/// buffers instead of two fresh `Vec`s per dimension).
fn wasserstein1d<S: AsRef<[f32]>>(x: &[S], y: &[S], d: usize) -> f32 {
    let mut total = 0.0;
    let mut col_x: Vec<f32> = Vec::with_capacity(x.len());
    let mut col_y: Vec<f32> = Vec::with_capacity(y.len());
    for j in 0..d {
        col_x.clear();
        col_x.extend(x.iter().map(|s| s.as_ref()[j]));
        col_y.clear();
        col_y.extend(y.iter().map(|s| s.as_ref()[j]));
        sort_f32_ascending(&mut col_x);
        sort_f32_ascending(&mut col_y);
        total += wasserstein1d_sorted_core(&col_x, &col_y);
    }
    total / d as f32
}

/// Zero-alloc 1-D Wasserstein on scalar (d = 1) slices for the
/// `numeric_stability` probe (Issue 697 T1.2) — reuses the same sort +
/// quantile-grid core as `wasserstein1d` above; no metric math is duplicated
/// in the consumer module. Grow-only scratch: buffers are `clear()`ed and
/// refilled per call (contents unspecified after the call), so the steady
/// state is allocation-free once capacity covers the larger slice.
/// Bit-identical to `wasserstein1d` on `&[[f32; 1]]` views of the same data
/// (pinned by `scalar_into_matches_multidim_substrate_bit_exactly` below).
pub(crate) fn wasserstein1d_scalar_into(
    x: &[f32],
    y: &[f32],
    scratch_x: &mut Vec<f32>,
    scratch_y: &mut Vec<f32>,
) -> f32 {
    scratch_x.clear();
    scratch_x.extend_from_slice(x);
    scratch_y.clear();
    scratch_y.extend_from_slice(y);
    sort_f32_ascending(scratch_x);
    sort_f32_ascending(scratch_y);
    wasserstein1d_sorted_core(scratch_x, scratch_y)
}

/// Linear-interpolated quantile from a sorted slice at fraction `f ∈ [0, 1]`.
#[inline]
fn quantile_interp(sorted: &[f32], f: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = f.clamp(0.0, 0.9999) * sorted.len() as f32;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Linear CKA in feature space: `trace(Cx·Cy) / (‖Cx‖_F · ‖Cy‖_F)`.
///
/// Uses d×d feature-space Gram matrices (`Cx = XᵀX/m`, `Cy = YᵀY/n`) so
/// candidate and target may have different sample counts (sample-space CKA
/// requires equal `n`).
fn cka_linear<S: AsRef<[f32]>>(x: &[S], y: &[S], d: usize) -> f32 {
    let cx = feature_gram(x, d);
    let cy = feature_gram(y, d);

    // trace(Cx·Cy) = Σ_i Σ_j Cx[i,j]·Cy[j,i]; both symmetric so Cy[j,i]=Cy[i,j].
    let mut trace_val = 0.0;
    let mut norm_cx = 0.0;
    let mut norm_cy = 0.0;
    for i in 0..d {
        for j in 0..d {
            trace_val += cx[i * d + j] * cy[j * d + i];
            norm_cx += cx[i * d + j] * cx[i * d + j];
            norm_cy += cy[i * d + j] * cy[i * d + j];
        }
    }
    let denom = (norm_cx * norm_cy).sqrt();
    if denom == 0.0 { 0.0 } else { trace_val / denom }
}

/// Compute the d×d feature Gram matrix `XᵀX/n` (flattened row-major).
fn feature_gram<S: AsRef<[f32]>>(samples: &[S], d: usize) -> Vec<f32> {
    let n = samples.len();
    let mut gram = vec![0.0_f32; d * d];
    for s in samples {
        let s = s.as_ref();
        for i in 0..d {
            for j in 0..d {
                gram[i * d + j] += s[i] * s[j];
            }
        }
    }
    let inv_n = 1.0 / n as f32;
    for g in gram.iter_mut() {
        *g *= inv_n;
    }
    gram
}

#[cfg(test)]
mod tests {
    use super::*;

    // Issue 697 T1.2 substrate cross-pin: the scalar (d = 1) zero-alloc
    // wrapper used by `numeric_stability` must stay bit-identical to the
    // multi-dim metric on the same data. Compiled only when the consumer
    // feature is on (the wrapper is feature-gated to stay dead-code-free).
    #[cfg(test)]
    #[test]
    fn scalar_into_matches_multidim_substrate_bit_exactly() {
        let mut s = 0x1234_5678_9ABC_DEF0u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let n = 257; // odd length exercises the quantile grid non-trivially
        let x: Vec<f32> = (0..n)
            .map(|_| ((next() >> 40) as f32) / ((1u32 << 24) as f32))
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|_| ((next() >> 40) as f32) / ((1u32 << 24) as f32))
            .collect();
        let cx: Vec<[f32; 1]> = x.iter().map(|&v| [v]).collect();
        let cy: Vec<[f32; 1]> = y.iter().map(|&v| [v]).collect();
        let multi = wasserstein1d(&cx, &cy, 1);
        let mut scratch_x = Vec::new();
        let mut scratch_y = Vec::new();
        let scalar = wasserstein1d_scalar_into(&x, &y, &mut scratch_x, &mut scratch_y);
        assert_eq!(
            multi.to_bits(),
            scalar.to_bits(),
            "wasserstein1d_scalar_into drifted from the d=1 substrate metric"
        );
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn centroid_cosine_identical_is_one() {
        let acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let labels = [true, false, true];
        let ds = DataSet::new(&acts, &labels);
        let score = transfer_score(&ds, &ds, TransferMetric::CentroidCosine).unwrap();
        assert!(approx_eq(score, 1.0, 1e-5), "got {score}");
    }

    #[test]
    fn euclidean_identical_is_zero() {
        let acts: Vec<[f32; 3]> = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let labels = [true, false];
        let ds = DataSet::new(&acts, &labels);
        let score = transfer_score(&ds, &ds, TransferMetric::Euclidean).unwrap();
        assert!(approx_eq(score, 0.0, 1e-4), "got {score}");
    }

    #[test]
    fn class_conditional_separates_by_label() {
        // candidate: class-false centroid at [1,0], class-true at [10,0]
        let cand_acts: Vec<[f32; 2]> = vec![[1.0, 0.1], [10.0, 0.0], [1.0, -0.1], [10.0, 0.1]];
        let cand_labels = [false, true, false, true];
        let cand = DataSet::new(&cand_acts, &cand_labels);

        // target: same structure → class-conditional cosine should be ~1.0
        let target = DataSet::new(&cand_acts, &cand_labels);
        let score_mal = transfer_score(
            &cand,
            &target,
            TransferMetric::ClassConditionalCosineMalicious,
        )
        .unwrap();
        let score_ben =
            transfer_score(&cand, &target, TransferMetric::ClassConditionalCosineBenign).unwrap();
        assert!(
            approx_eq(score_mal, 1.0, 1e-5),
            "malicious cos got {score_mal}"
        );
        assert!(
            approx_eq(score_ben, 1.0, 1e-5),
            "benign cos got {score_ben}"
        );
    }

    #[test]
    fn class_conditional_empty_class_errors() {
        let acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [2.0, 0.0]];
        let labels_all_true = [true, true];
        let ds = DataSet::new(&acts, &labels_all_true);
        // Requesting malicious class (false) but no false labels → EmptyClass
        assert_eq!(
            transfer_score(&ds, &ds, TransferMetric::ClassConditionalCosineMalicious),
            Err(MagError::EmptyClass)
        );
    }

    #[test]
    fn cka_identical_is_one() {
        let acts: Vec<[f32; 2]> = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = [true, false, true];
        let ds = DataSet::new(&acts, &labels);
        let score = transfer_score(&ds, &ds, TransferMetric::CkaLinear).unwrap();
        assert!(approx_eq(score, 1.0, 1e-4), "got {score}");
    }

    #[test]
    fn mmd_identical_is_zero() {
        let acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let labels = [true, false, true];
        let ds = DataSet::new(&acts, &labels);
        let score = transfer_score(&ds, &ds, TransferMetric::RbfMmd).unwrap();
        assert!(approx_eq(score, 0.0, 1e-4), "got {score}");
    }

    #[test]
    fn wasserstein_identical_is_zero() {
        let acts: Vec<[f32; 2]> = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let labels = [true, false, true];
        let ds = DataSet::new(&acts, &labels);
        let score = transfer_score(&ds, &ds, TransferMetric::Wasserstein1d).unwrap();
        assert!(approx_eq(score, 0.0, 1e-4), "got {score}");
    }

    #[test]
    fn rank_candidates_returns_sorted_desc() {
        // 3 candidates, 1 metric. Candidate 1 is identical to target (best),
        // candidate 2 is orthogonal, candidate 3 is anti-correlated.
        let target_acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [1.0, 0.1]];
        let target_labels = [true, false];
        let target = DataSet::new(&target_acts, &target_labels);

        let c0_acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [1.0, 0.1]]; // identical
        let c1_acts: Vec<[f32; 2]> = vec![[0.0, 1.0], [0.1, 1.0]]; // orthogonal
        let c2_acts: Vec<[f32; 2]> = vec![[-1.0, 0.0], [-1.0, -0.1]]; // anti
        let labels = [true, false];
        let candidates = vec![
            DataSet::new(&c0_acts, &labels),
            DataSet::new(&c1_acts, &labels),
            DataSet::new(&c2_acts, &labels),
        ];

        let ranked =
            rank_candidates(&candidates, &target, &[TransferMetric::CentroidCosine]).unwrap();

        assert_eq!(ranked.len(), 3);
        // Best (identical) should be first.
        assert_eq!(ranked[0].candidate_idx, 0);
        assert!(approx_eq(ranked[0].mean_percentile, 1.0, 1e-5));
        // Descending order.
        assert!(ranked[0].mean_percentile >= ranked[1].mean_percentile);
        assert!(ranked[1].mean_percentile >= ranked[2].mean_percentile);
    }

    #[test]
    fn rank_candidates_multi_metric_aggregates() {
        let target_acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [0.0, 1.0]];
        let target_labels = [true, false];
        let target = DataSet::new(&target_acts, &target_labels);

        let c0_acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [0.0, 1.0]]; // identical to target
        let c1_acts: Vec<[f32; 2]> = vec![[5.0, 5.0], [5.0, 5.0]]; // different
        let labels = [true, false];
        let candidates = vec![
            DataSet::new(&c0_acts, &labels),
            DataSet::new(&c1_acts, &labels),
        ];

        let ranked = rank_candidates(
            &candidates,
            &target,
            &[TransferMetric::CentroidCosine, TransferMetric::Euclidean],
        )
        .unwrap();

        // Identical candidate should win on both metrics.
        assert_eq!(ranked[0].candidate_idx, 0);
        assert_eq!(ranked[0].per_metric_scores.len(), 2);
    }

    #[test]
    fn transfer_score_label_length_mismatch_errors() {
        let acts: Vec<[f32; 2]> = vec![[1.0, 0.0], [0.0, 1.0]];
        let labels_ok = [true, false];
        let labels_bad = [true]; // wrong length
        let a = DataSet::new(&acts, &labels_ok);
        let b = DataSet::new(&acts, &labels_bad);
        assert_eq!(
            transfer_score(&a, &b, TransferMetric::CentroidCosine),
            Err(MagError::DimMismatch)
        );
    }

    // ── Recos metric (Plan 437 Phase 3, gated on `recos` feature) ────
    //
    // These tests verify the recos-vs-cosine separation property on MAG
    // centroids: recos saturates at 1.0 under ordinal concordance (any
    // monotonic relationship), while cosine requires linear dependence.
    // When `recos` is off, TransferMetric::Recos falls back to centroid
    // cosine behavior, so these tests are feature-gated.

    #[cfg(feature = "recos")]
    #[test]
    fn recos_metric_identical_is_one() {
        // Identical datasets → identical centroids → recos = 1.0.
        let acts: Vec<Vec<f32>> = vec![vec![1.0, 3.0, 2.0, 5.0], vec![2.0, 1.0, 4.0, 3.0]];
        let labels = [true, false];
        let ds = DataSet::new(&acts, &labels);
        let score = transfer_score(&ds, &ds, TransferMetric::Recos).unwrap();
        assert!(
            (score - 1.0).abs() < 1e-5,
            "identical centroids: recos={score}"
        );
    }

    #[cfg(feature = "recos")]
    #[test]
    fn recos_metric_discordant_below_one() {
        // Reversed centroids: discordant → recos < 1.0.
        let cand_acts: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let cand_labels = [true];
        let target_acts: Vec<Vec<f32>> = vec![vec![4.0, 3.0, 2.0, 1.0]];
        let target_labels = [true];
        let cand = DataSet::new(&cand_acts, &cand_labels);
        let target = DataSet::new(&target_acts, &target_labels);
        let score = transfer_score(&cand, &target, TransferMetric::Recos).unwrap();
        assert!(score < 1.0, "discordant centroids: recos={score} < 1.0");
        assert!(score >= 0.0, "recos is non-negative: {score}");
    }

    #[cfg(feature = "recos")]
    #[test]
    fn recos_metric_distinct_from_centroid_cosine() {
        // Nonlinear-monotonic centroids: b = a². Cosine < 1.0 (not linear),
        // recos = 1.0 (ordinally concordant). This is the separation property.
        // Single-sample datasets so the sample IS the centroid.
        let cand_acts: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let cand_labels = [true];
        let target_acts: Vec<Vec<f32>> = vec![vec![1.0, 4.0, 9.0, 16.0]]; // = a²
        let target_labels = [true];
        let cand = DataSet::new(&cand_acts, &cand_labels);
        let target = DataSet::new(&target_acts, &target_labels);

        let cos = transfer_score(&cand, &target, TransferMetric::CentroidCosine).unwrap();
        let recos = transfer_score(&cand, &target, TransferMetric::Recos).unwrap();

        assert!(cos < 1.0, "cosine of nonlinear pair: cos={cos} < 1.0");
        assert!(
            (recos - 1.0).abs() < 1e-5,
            "recos of monotonic pair: recos={recos} = 1.0"
        );
        assert!(
            recos > cos,
            "recos ({recos}) > cosine ({cos}) on monotonic-nonlinear pair"
        );
    }

    #[cfg(feature = "recos")]
    #[test]
    fn recos_metric_into_matches_allocating() {
        // transfer_score_into (zero-alloc) must match transfer_score (allocating)
        // for the Recos metric.
        let cand_acts: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0, 4.0], vec![0.5, 1.5, 2.5, 3.5]];
        let cand_labels = [true, false];
        let target_acts: Vec<Vec<f32>> = vec![vec![2.0, 4.0, 6.0, 8.0]];
        let target_labels = [true];
        let cand = DataSet::new(&cand_acts, &cand_labels);
        let target = DataSet::new(&target_acts, &target_labels);

        let score_alloc = transfer_score(&cand, &target, TransferMetric::Recos).unwrap();
        let mut scratch = [0.0_f32; 8]; // 2 * d = 2 * 4
        let score_into =
            transfer_score_into(&cand, &target, TransferMetric::Recos, &mut scratch).unwrap();
        assert!(
            (score_alloc - score_into).abs() < 1e-5,
            "alloc={score_alloc} into={score_into}"
        );
    }
}
