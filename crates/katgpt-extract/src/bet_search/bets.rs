//! T1 — the authored bet vocabulary (Issue 952).
//!
//! A **bet** is one small, deterministic estimator over the comparables
//! corpus: given the comps, an optional row to leave out, and a query
//! feature row, it returns a point prediction. Bets are AUTHORED, never
//! learned (modelless-first): every "fit" below is a closed form recomputed
//! from the comps on every call — there is no stored state to train.
//!
//! The search ([`super::search`]) composes bets; [`Bet::predict`] with
//! `skip = Some(i)` is the leave-one-out backtest the search scores on the
//! SELECTION split. The calibration split never reaches a bet during
//! selection — the conformal overlay reads it afterwards.
//!
//! # Determinism (Issue 940)
//!
//! Every reduction folds sequentially in comparable-index order; order
//! statistics use `select_nth_unstable_by(f32::total_cmp)`, which is a pure
//! function of the input slice. No clock, no rng.

use crate::estimate::{knn_weighted_mean, normalize_feature};

/// Upper bound on the neighbour count a [`AuthoredBet::Knn`] may ask for —
/// sizes the k-selection slots in [`BetScratch`] once.
pub const MAX_KNN_K: usize = 32;

/// Relative singularity guard for the two-feature normal equations:
/// `|det| <= TWO_FEATURE_DET_EPS · Saa · Sbb` (collinear or degenerate
/// features) falls back to the single-feature ratio bet.
const TWO_FEATURE_DET_EPS: f64 = 1e-9;

/// A read-only view over one comps corpus, in SOURCE order (oldest first —
/// the recency bet reads position as age).
#[derive(Clone, Copy, Debug)]
pub struct CompsView<'a> {
    /// Raw features, flat `n × dim` (the ratio bets read raw magnitudes).
    pub raw: &'a [f32],
    /// Scale-normalized features, flat `n × dim` (the kNN distance space).
    pub norm: &'a [f32],
    /// Realized values, one per comparable.
    pub values: &'a [f32],
    /// Per-feature scales (`estimate::fit_scales` convention).
    pub scales: &'a [f32],
    pub dim: usize,
    pub n: usize,
}

impl CompsView<'_> {
    #[inline]
    fn raw_at(&self, i: usize, j: usize) -> f32 {
        self.raw[i * self.dim + j]
    }
}

/// Caller-owned scratch every bet borrows — allocated once, so the predict
/// path is allocation-free.
#[derive(Clone, Debug)]
pub struct BetScratch {
    query_norm: Vec<f32>,
    best_dist: Vec<f32>,
    best_value: Vec<f32>,
    order: Vec<f32>,
}

impl BetScratch {
    /// Scratch for a corpus of at most `max_n` rows of width `dim`.
    pub fn new(dim: usize, max_n: usize) -> Self {
        Self {
            query_norm: vec![0.0; dim.max(1)],
            best_dist: vec![f32::INFINITY; MAX_KNN_K],
            best_value: vec![0.0; MAX_KNN_K],
            order: Vec::with_capacity(max_n),
        }
    }
}

/// One estimator over a comps corpus.
pub trait Bet {
    /// Predict the value at `query` (raw features) from `comps`, excluding
    /// comparable `skip` when set (the leave-one-out backtest). Returns
    /// `NaN` when the bet has nothing to stand on (no usable comparable);
    /// callers treat a `NaN` prediction as a failed bet.
    fn predict(
        &self,
        comps: &CompsView<'_>,
        skip: Option<usize>,
        query: &[f32],
        scratch: &mut BetScratch,
    ) -> f32;
}

/// The authored bet family. `Copy` + fixed-size so a roster is a plain
/// slice and a composition is a bitmask over it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AuthoredBet {
    /// Median of the comparables' values (query-blind).
    MedianOfComps,
    /// Inverse-distance-weighted kNN in the scale-normalized feature space
    /// (the `estimate` head's kernel; `k` clamped to [`MAX_KNN_K`]).
    Knn { k: u8 },
    /// `median(value / x_f) · query_f` over comparables with `x_f > 0` — the
    /// price-per-unit bet. Falls back to [`Self::MedianOfComps`] when the
    /// query's `x_f` is not positive or no comparable carries one.
    RatioScaling { feature: u8 },
    /// Exponentially recency-weighted mean of the values: the newest
    /// comparable weighs 1, each step older halves after `half_life` steps.
    RecencyWeighted { half_life: f32 },
    /// Least-squares `value ≈ α·x_a + β·x_b` through the origin (closed-form
    /// 2×2 normal equations, f64 accumulation). Falls back to
    /// [`Self::RatioScaling`] on `a` when the system is singular.
    TwoFeatureRatio { a: u8, b: u8 },
}

/// The default roster — 7 bets. Order is load-bearing: it is the bit order
/// of every composition mask and the first-index tie rule of every pick.
pub const DEFAULT_BETS: [AuthoredBet; 7] = [
    AuthoredBet::MedianOfComps,
    AuthoredBet::Knn { k: 3 },
    AuthoredBet::Knn { k: 5 },
    AuthoredBet::Knn { k: 10 },
    AuthoredBet::RatioScaling { feature: 0 },
    AuthoredBet::RecencyWeighted { half_life: 4.0 },
    AuthoredBet::TwoFeatureRatio { a: 0, b: 1 },
];

impl Bet for AuthoredBet {
    fn predict(
        &self,
        comps: &CompsView<'_>,
        skip: Option<usize>,
        query: &[f32],
        scratch: &mut BetScratch,
    ) -> f32 {
        match *self {
            Self::MedianOfComps => median_of_values(comps, skip, scratch),
            Self::Knn { k } => knn(comps, skip, query, usize::from(k), scratch),
            Self::RatioScaling { feature } => {
                ratio_scaling(comps, skip, query, usize::from(feature), scratch)
            }
            Self::RecencyWeighted { half_life } => recency_weighted(comps, skip, half_life),
            Self::TwoFeatureRatio { a, b } => {
                two_feature_ratio(comps, skip, query, usize::from(a), usize::from(b), scratch)
            }
        }
    }
}

/// Median of `scratch.order` (consumed). `NaN` when empty.
fn median_in_place(order: &mut [f32]) -> f32 {
    let m = order.len();
    if m == 0 {
        return f32::NAN;
    }
    let mid = m / 2;
    let (lower, hi, _) = order.select_nth_unstable_by(mid, f32::total_cmp);
    let hi = *hi;
    if m % 2 == 1 {
        return hi;
    }
    let mut lo = f32::NEG_INFINITY;
    for &v in lower.iter() {
        if v > lo {
            lo = v;
        }
    }
    0.5 * (lo + hi)
}

fn median_of_values(comps: &CompsView<'_>, skip: Option<usize>, scratch: &mut BetScratch) -> f32 {
    scratch.order.clear();
    for (i, &v) in comps.values.iter().enumerate().take(comps.n) {
        if skip != Some(i) {
            scratch.order.push(v);
        }
    }
    median_in_place(&mut scratch.order)
}

fn knn(
    comps: &CompsView<'_>,
    skip: Option<usize>,
    query: &[f32],
    k: usize,
    scratch: &mut BetScratch,
) -> f32 {
    let available = comps.n - usize::from(skip.is_some_and(|s| s < comps.n));
    let k = k.clamp(1, MAX_KNN_K).min(available);
    if k == 0 || comps.dim == 0 {
        return f32::NAN;
    }
    knn_weighted_mean(
        query,
        comps.scales,
        &mut scratch.query_norm,
        comps.norm,
        comps.values,
        comps.dim,
        comps.n,
        k,
        skip,
        &mut scratch.best_dist[..k],
        &mut scratch.best_value[..k],
    )
}

fn ratio_scaling(
    comps: &CompsView<'_>,
    skip: Option<usize>,
    query: &[f32],
    f: usize,
    scratch: &mut BetScratch,
) -> f32 {
    let q = query.get(f).copied().unwrap_or(0.0);
    if f >= comps.dim || !(q.is_finite() && q > 0.0) {
        return median_of_values(comps, skip, scratch);
    }
    scratch.order.clear();
    for i in 0..comps.n {
        if skip == Some(i) {
            continue;
        }
        let x = comps.raw_at(i, f);
        if x.is_finite() && x > 0.0 {
            scratch.order.push(comps.values[i] / x);
        }
    }
    if scratch.order.is_empty() {
        return median_of_values(comps, skip, scratch);
    }
    median_in_place(&mut scratch.order) * q
}

fn recency_weighted(comps: &CompsView<'_>, skip: Option<usize>, half_life: f32) -> f32 {
    let h = if half_life.is_finite() && half_life > 0.0 {
        half_life
    } else {
        1.0
    };
    let newest = comps.n.saturating_sub(1);
    let mut w_sum = 0.0_f64;
    let mut v_sum = 0.0_f64;
    for (i, &v) in comps.values.iter().enumerate().take(comps.n) {
        if skip == Some(i) {
            continue;
        }
        let age = (newest - i) as f32;
        let w = f64::from((-age / h).exp2());
        w_sum += w;
        v_sum += w * f64::from(v);
    }
    if w_sum > 0.0 {
        (v_sum / w_sum) as f32
    } else {
        f32::NAN
    }
}

fn two_feature_ratio(
    comps: &CompsView<'_>,
    skip: Option<usize>,
    query: &[f32],
    a: usize,
    b: usize,
    scratch: &mut BetScratch,
) -> f32 {
    if a >= comps.dim || b >= comps.dim || a == b {
        return ratio_scaling(comps, skip, query, a, scratch);
    }
    let (mut saa, mut sbb, mut sab, mut say, mut sby) = (0.0_f64, 0.0, 0.0, 0.0, 0.0);
    for i in 0..comps.n {
        if skip == Some(i) {
            continue;
        }
        let xa = f64::from(comps.raw_at(i, a));
        let xb = f64::from(comps.raw_at(i, b));
        let y = f64::from(comps.values[i]);
        saa += xa * xa;
        sbb += xb * xb;
        sab += xa * xb;
        say += xa * y;
        sby += xb * y;
    }
    let det = saa * sbb - sab * sab;
    if !(det.is_finite() && det.abs() > TWO_FEATURE_DET_EPS * saa * sbb && det != 0.0) {
        return ratio_scaling(comps, skip, query, a, scratch);
    }
    let alpha = (say * sbb - sby * sab) / det;
    let beta = (sby * saa - say * sab) / det;
    let qa = f64::from(query.get(a).copied().unwrap_or(0.0));
    let qb = f64::from(query.get(b).copied().unwrap_or(0.0));
    (alpha * qa + beta * qb) as f32
}

/// Fill `norm` (flat `n × dim`) from `raw` under `scales` — the kNN
/// distance space every [`CompsView`] carries beside the raw features.
pub(crate) fn normalize_rows(raw: &[f32], scales: &[f32], dim: usize, norm: &mut Vec<f32>) {
    norm.clear();
    norm.reserve(raw.len());
    for row in raw.chunks_exact(dim.max(1)) {
        for (j, &v) in row.iter().enumerate() {
            norm.push(normalize_feature(
                v,
                scales.get(j).copied().unwrap_or(f32::INFINITY),
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Owned {
        raw: Vec<f32>,
        norm: Vec<f32>,
        values: Vec<f32>,
        scales: Vec<f32>,
        dim: usize,
    }

    impl Owned {
        fn new(rows: &[(&[f32], f32)]) -> Self {
            let dim = rows[0].0.len();
            let raw: Vec<f32> = rows.iter().flat_map(|r| r.0.iter().copied()).collect();
            let values = rows.iter().map(|r| r.1).collect();
            let scales = vec![1.0; dim];
            let mut norm = Vec::new();
            normalize_rows(&raw, &scales, dim, &mut norm);
            Self {
                raw,
                norm,
                values,
                scales,
                dim,
            }
        }

        fn view(&self) -> CompsView<'_> {
            CompsView {
                raw: &self.raw,
                norm: &self.norm,
                values: &self.values,
                scales: &self.scales,
                dim: self.dim,
                n: self.values.len(),
            }
        }
    }

    fn predict(bet: AuthoredBet, o: &Owned, skip: Option<usize>, q: &[f32]) -> f32 {
        let mut s = BetScratch::new(o.dim, o.values.len());
        bet.predict(&o.view(), skip, q, &mut s)
    }

    #[test]
    fn median_odd_even_and_skip() {
        let o = Owned::new(&[(&[1.0], 5.0), (&[1.0], 1.0), (&[1.0], 3.0), (&[1.0], 9.0)]);
        assert_eq!(predict(AuthoredBet::MedianOfComps, &o, None, &[0.0]), 4.0);
        assert_eq!(
            predict(AuthoredBet::MedianOfComps, &o, Some(3), &[0.0]),
            3.0
        );
    }

    #[test]
    fn ratio_scaling_recovers_price_per_unit() {
        let o = Owned::new(&[(&[10.0], 20.0), (&[20.0], 40.0), (&[5.0], 10.0)]);
        let p = predict(AuthoredBet::RatioScaling { feature: 0 }, &o, None, &[7.0]);
        assert!((p - 14.0).abs() < 1e-5, "{p}");
        // Non-positive query magnitude → median fallback.
        let p = predict(AuthoredBet::RatioScaling { feature: 0 }, &o, None, &[0.0]);
        assert_eq!(p, 20.0);
    }

    #[test]
    fn two_feature_ratio_solves_exact_plane() {
        let o = Owned::new(&[
            (&[1.0, 0.0], 2.0),
            (&[0.0, 1.0], 3.0),
            (&[1.0, 1.0], 5.0),
            (&[2.0, 1.0], 7.0),
        ]);
        let p = predict(
            AuthoredBet::TwoFeatureRatio { a: 0, b: 1 },
            &o,
            None,
            &[3.0, 2.0],
        );
        assert!((p - 12.0).abs() < 1e-4, "{p}");
        // Leave-one-out still solves the exact plane from the other three.
        let p = predict(
            AuthoredBet::TwoFeatureRatio { a: 0, b: 1 },
            &o,
            Some(3),
            &[2.0, 1.0],
        );
        assert!((p - 7.0).abs() < 1e-4, "{p}");
    }

    #[test]
    fn two_feature_ratio_collinear_falls_back_to_ratio() {
        let o = Owned::new(&[(&[1.0, 2.0], 3.0), (&[2.0, 4.0], 6.0)]);
        let two = predict(
            AuthoredBet::TwoFeatureRatio { a: 0, b: 1 },
            &o,
            None,
            &[4.0, 8.0],
        );
        let one = predict(
            AuthoredBet::RatioScaling { feature: 0 },
            &o,
            None,
            &[4.0, 8.0],
        );
        assert_eq!(two, one);
    }

    #[test]
    fn recency_weights_newest_most() {
        let o = Owned::new(&[(&[0.0], 0.0), (&[0.0], 0.0), (&[0.0], 10.0)]);
        let p = predict(
            AuthoredBet::RecencyWeighted { half_life: 1.0 },
            &o,
            None,
            &[0.0],
        );
        // weights 0.25, 0.5, 1 → 10 / 1.75
        assert!((p - 10.0 / 1.75).abs() < 1e-5, "{p}");
    }

    #[test]
    fn knn_skip_excludes_the_row_and_k_clamps() {
        let o = Owned::new(&[(&[0.0], 1.0), (&[10.0], 100.0)]);
        // With row 0 skipped the only neighbour of the query at 0 is row 1.
        assert_eq!(
            predict(AuthoredBet::Knn { k: 5 }, &o, Some(0), &[0.0]),
            100.0
        );
        let empty = Owned::new(&[(&[0.0], 1.0)]);
        assert!(predict(AuthoredBet::Knn { k: 3 }, &empty, Some(0), &[0.0]).is_nan());
    }

    #[test]
    fn empty_corpus_bets_report_nan() {
        let o = Owned::new(&[(&[1.0], 1.0)]);
        for bet in [
            AuthoredBet::MedianOfComps,
            AuthoredBet::RecencyWeighted { half_life: 2.0 },
        ] {
            assert!(predict(bet, &o, Some(0), &[1.0]).is_nan(), "{bet:?}");
        }
    }
}
