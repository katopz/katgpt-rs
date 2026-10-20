//! Cross-sectional estimate head (Plan 628 T0.4).
//!
//! The Estimate class answered modellessly: a kNN conditioner over a
//! comparables corpus (tabular content parsed FROM the state, and — Phase 2 —
//! the per-deployment outcomes ledger as it accumulates) produces the point
//! prediction; the Plan-340 conformal overlay turns the calibration residuals
//! into a p10/median/p90 interval; the caller's grid (linear/log) quantizes
//! the render and the caller's domain bounds are disclosed as `p_below`/
//! `p_above`. Conditioning does not share across the two computed classes:
//! Estimate is cross-sectional, Forecast is time-series ([`crate::forecast`])
//! — Proposal 016 §3.2.
//!
//! # Floor law (Report the Floor, Research 322 / Plan 340)
//!
//! If the conditioned head cannot beat `ConformalIntervalCalibrator<
//! SeasonalNaiveForecaster>` (m=1) on CRPS/coverage/Winkler, the floor IS the
//! served answer. The G1 plumbing ([`SyntheticCrossSectional`] +
//! [`EstimateFloorHarness`]) makes the head runnable through
//! `katgpt_core::conformal::run_floor_comparison` directly; the G1 gate
//! EVALUATES at T0.6 — this file ships measurement plumbing, never a claim.
//!
//! # Hot path
//!
//! [`EstimateHead::estimate_into`] is zero-alloc after calibration: the
//! scratch buffers (query normalization, k selection slots, jackknife+
//! quantile scratch) are owned by the head and allocated at
//! construction/calibration. The ONE documented cold path is
//! [`EstimateHead::calibrate`] — the selection/calibration split plus a
//! leave-some-out residual pass, O(n²) distance evaluations over the
//! comparables. Call it once when the corpus is ready; the head does NOT
//! re-poll the source on the hot path (a growing ledger re-calibrates on the
//! engine's epoch cadence, Phase 2).
//!
//! # The selection/calibration split (Issue 952 amendment)
//!
//! The comps corpus is partitioned into a SELECTION split and a CALIBRATION
//! split before anything consumes it — deterministically, by a seeded
//! Fisher–Yates permutation over input indices ([`SPLIT_SEED`], the same
//! fixed-seed law as the synthetic corpus; [`split_partition`] documents the
//! rule). The verbatim rationale (Issue 952, "Honest caveats" G1): selecting
//! the estimator on the calibration rows breaks exchangeability → optimistic
//! coverage → "a coverage number produced on merged comps is NOT a
//! measurement". In this single-k posture no estimator selection runs yet
//! (the bet-search selection is Issue-952 T1/T2), so the selection split is
//! RESERVED; the residual pool is fed ONLY from calibration-split rows, each
//! calibration row predicted leave-one-out WITHIN the split — the
//! cross-conformal posture — while the served point is the chosen-k kNN over
//! the full corpus (a documented approximation: the pool reflects the
//! calibration-only predictor, the query is answered by the full-set one).
//!
//! # Small comps — jackknife+ (Issue 952: `n < small_comps_max`)
//!
//! When the corpus is too small to split, the split-conformal path does not
//! serve: the head serves jackknife+ instead (Barber et al. 2021,
//! "Predictive Inference with the Jackknife+", arXiv:2105.05114) — every
//! comparable's leave-one-out prediction at the query, minus/plus its own
//! LOO abs-residual, interval = the ⌈(n+1)(1−α)⌉ order statistics of the
//! two sets, point = the LOO prediction aggregate. Which posture served is
//! disclosed: [`EstimateHead::estimate_with_posture`] returns the
//! [`EstimatePosture`] beside the distribution. The threshold is
//! `small_comps_max` (default 8) — a PROVISIONAL, documented config field
//! deriving later from the Issue-952 T4 comps-size axis, never a silent
//! magic edge.
//!
//! # Determinism
//!
//! Exact O(n) top-k selection (no full sort, no floating sort instability):
//! ties in distance resolve to the LOWEST comparable index (the earliest-
//! scanned comparable is never displaced by an equal-distance later one).
//! No clock, no rng anywhere on the library path.

use core::cell::Cell;
use core::f32::consts::TAU;

use katgpt_core::conformal::{
    ConformalIntervalCalibrator, DecayUnit, PointForecaster, PredictiveInterval, PredictiveOutput,
    ResidualMode, UqPrimitiveUnderTest,
};
use katgpt_core::extraction_wire::{EstimateDistribution, Grid, GridKind};

/// Guard on the inverse-distance weight `w = 1/(d + eps)`: two comparables
/// at distance 0 (identical normalized features) split the weight equally
/// instead of dividing by zero. Public because the hand-computed kNN
/// contract is documented against it.
pub const KNN_DISTANCE_EPS: f32 = 1e-6;

/// Fixed seed for the synthetic cross-sectional corpus. DOCUMENTED AS
/// MEASUREMENT-ONLY: the corpus exists to run the G1 floor-comparison
/// plumbing end to end; real deployments source comparables from
/// state-parsed tables and the outcomes ledger.
pub const SYNTH_SEED: u64 = 0x5EED_0628;

/// Fixed seed for the selection/calibration split permutation (the Issue-952
/// amendment: the split must be deterministic under a NAMED constant — the
/// same fixed-seed law as [`SYNTH_SEED`]). `0x5EED_0952` — the issue number
/// rides in the constant.
pub const SPLIT_SEED: u64 = 0x5EED_0952;

/// Conformal-overlay shape for the estimate head: single channel, single
/// horizon (`max_h = 1`, `m = 1`) — cross-sectional residuals have no
/// horizon structure, so the single-pool [`ResidualMode::Paper`] bucket is
/// the whole pool.
const ESTIMATE_CHANNELS: usize = 1;
const ESTIMATE_MAX_H: usize = 1;
const ESTIMATE_SEASONAL_M: usize = 1;
/// Residual-ring capacity per bucket (the CSP default; the estimate corpus
/// is a snapshot — 256 calibration residuals is the realistic ceiling).
const ESTIMATE_RESIDUAL_CAPACITY: usize = 256;
/// The field id the floor harness files its synthetic comparables under.
const ESTIMATE_FLOOR_FIELD: &str = "estimate_floor";

/// One comparable: the conditioning evidence (a feature vector over a FIXED
/// schema for the field being estimated) plus the realized value. Cold data.
#[derive(Clone, Debug)]
pub struct Comparable {
    /// Per-feature evidence. All features of one corpus share a schema; rows
    /// are padded with `0.0` / truncated to the first row's width at
    /// calibration.
    pub features: Vec<f32>,
    /// The realized value the head learns to predict.
    pub value: f32,
}

/// The comparables corpus for one estimate field. Implemented by the
/// state-parsed table source and by [`OutcomesLedger`]; the engine composes
/// both (Proposal 016 §3.2(a)+(b)).
pub trait ComparableSource {
    /// The rows for `field_id`. Cold path: called by
    /// [`EstimateHead::calibrate`] only. An unknown field yields an empty
    /// corpus (the head then serves degenerate zeros until calibrated).
    fn comparables(&self, field_id: &str) -> Vec<Comparable>;
}

/// A trivial in-memory [`ComparableSource`] — tables keyed by field id.
/// The engine's real sources (state-parsed tables, the outcomes ledger)
/// implement the trait directly or feed this one.
#[derive(Clone, Debug, Default)]
pub struct InMemorySource {
    tables: Vec<(String, Vec<Comparable>)>,
}

impl InMemorySource {
    /// An empty source.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one row to a field's table (creating it on first push).
    pub fn push(&mut self, field_id: &str, row: Comparable) {
        if let Some((_, rows)) = self.tables.iter_mut().find(|(key, _)| key == field_id) {
            rows.push(row);
        } else {
            self.tables.push((field_id.to_string(), vec![row]));
        }
    }

    /// Replace (or create) a field's table wholesale.
    pub fn insert_table(&mut self, field_id: impl Into<String>, rows: Vec<Comparable>) {
        let id = field_id.into();
        match self.tables.iter_mut().find(|(key, _)| *key == id) {
            Some((_, existing)) => *existing = rows,
            None => self.tables.push((id, rows)),
        }
    }
}

impl ComparableSource for InMemorySource {
    fn comparables(&self, field_id: &str) -> Vec<Comparable> {
        self.tables
            .iter()
            .find(|(key, _)| key == field_id)
            .map(|(_, rows)| rows.clone())
            .unwrap_or_default()
    }
}

/// The per-deployment outcomes ledger (Proposal 016 §3.2(b)) — **INTERFACE
/// STUB ONLY, Phase-2-filled**. It already behaves as a bounded FIFO
/// [`ComparableSource`] (append, capacity cap with oldest-eviction, append/
/// eviction/query counters) so the estimate head consumes it unchanged; the
/// Phase-2 work (per-field Elo/Beta routing inputs, freeze/thaw snapshots of
/// the routing tables, bitemporal provenance landing in riir-neuron-db) is
/// deliberately NOT here. NO I/O — rows live in an internal `Vec` and die
/// with the process until Phase 2 gives the ledger its durable store.
pub struct OutcomesLedger {
    rows: Vec<(String, Comparable)>,
    capacity: usize,
    appends: u64,
    evictions: u64,
    /// Saturating query counter behind a `Cell` so the shared
    /// [`ComparableSource`] read path (`&self`) can still count.
    queries: Cell<u64>,
}

impl OutcomesLedger {
    /// A ledger capped at `capacity` rows (clamped to ≥ 1). At capacity the
    /// OLDEST row (across all fields) is evicted FIFO.
    pub fn new(capacity: usize) -> Self {
        Self {
            rows: Vec::new(),
            capacity: capacity.max(1),
            appends: 0,
            evictions: 0,
            queries: Cell::new(0),
        }
    }

    /// Append one realized outcome. Evicts the oldest row(s) while at
    /// capacity (FIFO; the eviction counter advances per removed row).
    pub fn append(&mut self, field_id: &str, row: Comparable) {
        while self.rows.len() >= self.capacity {
            self.rows.remove(0);
            self.evictions = self.evictions.saturating_add(1);
        }
        self.rows.push((field_id.to_string(), row));
        self.appends = self.appends.saturating_add(1);
    }

    /// Rows currently held (across all fields).
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether no rows are held.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The configured capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total appends since construction (saturating).
    pub fn appends(&self) -> u64 {
        self.appends
    }

    /// Total FIFO evictions since construction (saturating).
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Total trait queries since construction (saturating).
    pub fn queries(&self) -> u64 {
        self.queries.get()
    }
}

impl ComparableSource for OutcomesLedger {
    fn comparables(&self, field_id: &str) -> Vec<Comparable> {
        self.queries.set(self.queries.get().saturating_add(1));
        self.rows
            .iter()
            .filter(|(key, _)| key == field_id)
            .map(|(_, row)| row.clone())
            .collect()
    }
}

/// Compute per-feature scales from a corpus: the population standard
/// deviation of each feature (f64 accumulation, f32 output). A feature with
/// zero variance (or fewer than 2 finite samples) gets an INFINITE scale,
/// which the distance kernel treats as "feature ignored" — a constant
/// feature contributes nothing to the distance instead of amplifying the
/// query's deviation from it. Caller-supplied scales follow the SAME
/// convention: non-positive or non-finite entries mean "ignored feature".
pub fn fit_scales(rows: &[Comparable]) -> Vec<f32> {
    let Some(dim) = rows.iter().map(|r| r.features.len()).max() else {
        return Vec::new();
    };
    if dim == 0 {
        return Vec::new();
    }
    let mut scales = vec![f32::INFINITY; dim];
    for (j, scale) in scales.iter_mut().enumerate() {
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut count = 0_u64;
        for row in rows {
            if let Some(&v) = row.features.get(j)
                && v.is_finite()
            {
                sum += f64::from(v);
                sum_sq += f64::from(v) * f64::from(v);
                count += 1;
            }
        }
        if count >= 2 {
            let mean = sum / (count as f64);
            let var = sum_sq / (count as f64) - mean * mean;
            if var > 0.0 && var.is_finite() {
                *scale = (var.sqrt()) as f32;
            }
        }
    }
    scales
}

/// Quantize `x` onto the caller's grid.
///
/// - [`GridKind::Linear`]: nearest multiple of `step` (linear units).
/// - [`GridKind::Log`]: nearest power of the ratio `step` — quantization
///   happens in log space; applies only to `x > 0` (a non-positive value has
///   no logarithm and is left unquantized, honestly rendered).
///
/// A non-finite `x`, or a degenerate `step` (non-finite, `<= 0` linear,
/// `<= 1.0` log), leaves `x` unchanged. The map is monotone non-decreasing,
/// so quantizing p10/point/p90 independently preserves their ordering.
pub fn quantize_onto_grid(x: f64, grid: &Grid) -> f64 {
    if !x.is_finite() {
        return x;
    }
    match grid.kind {
        GridKind::Linear => {
            if !grid.step.is_finite() || grid.step <= 0.0 {
                return x;
            }
            (x / grid.step).round() * grid.step
        }
        GridKind::Log => {
            let ln_step = grid.step.ln();
            if !ln_step.is_finite() || ln_step <= 0.0 || x <= 0.0 {
                return x;
            }
            ((x.ln() / ln_step).round() * ln_step).exp()
        }
    }
}

/// Sanitize a miscoverage level: non-finite falls back to 0.05, anything
/// outside [0, 0.5] clamps (the two-tailed conformal substrate's range).
pub(crate) fn sanitize_alpha(alpha: f32) -> f32 {
    if alpha.is_finite() {
        alpha.clamp(0.0, 0.5)
    } else {
        0.05
    }
}

/// Apply the caller's domain bounds: symmetric-tail mass outside [min, max].
fn domain_tails(p10: f64, p90: f64, point: f64, domain: Option<(f64, f64)>) -> (f64, f64) {
    let Some((min, max)) = domain else {
        return (0.0, 0.0);
    };
    // Symmetric-tail approximation (documented): the distribution is rendered
    // as uniform on the support [2·p10 − p90, 2·p90 − p10] — the level
    // interval extended symmetrically by its own width so the extended
    // support carries mass 1. Mass outside the domain is then the fraction
    // of that support below `min` / above `max`.
    let lo = 2.0 * p10 - p90;
    let hi = 2.0 * p90 - p10;
    let width = hi - lo;
    let (below, above) = if width <= 0.0 {
        // Degenerate (zero-width) interval: a point mass at `point`.
        ((min > point) as i32 as f64, (max < point) as i32 as f64)
    } else {
        (
            ((min - lo) / width).clamp(0.0, 1.0),
            ((hi - max) / width).clamp(0.0, 1.0),
        )
    };
    // The wire caps p_below + p_above at 1.0; renormalize if an inverted
    // domain (min > max) pushed the sum past it.
    let total = below + above;
    if total > 1.0 {
        (below / total, above / total)
    } else {
        (below, above)
    }
}

/// Wire-shape accommodation (disclosed): the wire requires `p10 ≤ median ≤
/// p90` with `median == point`, but a one-signed residual pool can put the
/// true conformal interval entirely off the point forecast. The interval is
/// trimmed to include the point. A head that needs the trim habitually is
/// micalibrated — the G1 floor comparison is the instrument that catches it.
pub(crate) fn order_guarded(lo: f64, point: f64, hi: f64) -> (f64, f64) {
    (lo.min(point), hi.max(point))
}

/// Apply the module's scale convention to one feature: finite positive
/// scales divide; infinite / non-positive / non-finite scales mean the
/// feature is IGNORED (normalized to 0 on both the stored rows and the
/// query, so the distance contribution cancels exactly).
#[inline]
fn normalize_feature(v: f32, scale: f32) -> f32 {
    if scale.is_finite() && scale > 0.0 {
        v / scale
    } else {
        0.0
    }
}

/// Pad / sanitize caller-supplied scales to the corpus dimension (missing
/// dims are ignored features; invalid entries are ignored features).
fn scales_padded_to_dim(scales: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![f32::INFINITY; dim];
    for (i, &s) in scales.iter().take(dim).enumerate() {
        out[i] = if s.is_finite() && s > 0.0 {
            s
        } else {
            f32::INFINITY
        };
    }
    out
}

/// Partition input indices `0..n` into `(selection, calibration)` — the
/// Issue-952 split, deterministic by construction:
///
/// - Fisher–Yates shuffle of `0..n` driven by [`Xorshift64`] seeded with the
///   named [`SPLIT_SEED`] (modulo indexing — a deterministic, documented
///   choice; uniformity is not load-bearing here, reproducibility is).
/// - The FIRST `n_cal` entries of the permutation form the CALIBRATION
///   split; the remainder form the SELECTION split.
/// - `n_cal = round(n · calibration_fraction)`, clamped so BOTH splits are
///   non-empty for `n ≥ 2` (fraction 0 keeps 1 calibration row, fraction 1
///   keeps 1 selection row). For `n ≤ 1` everything is calibration (the
///   jackknife+ posture serves there; the split is unused).
fn split_partition(n: usize, calibration_fraction: f32) -> (Vec<usize>, Vec<usize>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut rng = Xorshift64::new(SPLIT_SEED);
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    let fraction = if calibration_fraction.is_finite() {
        calibration_fraction.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let n_cal = if n >= 2 {
        ((n as f32 * fraction).round() as usize).clamp(1, n - 1)
    } else {
        n
    };
    let mut calibration = perm;
    let selection = calibration.split_off(n_cal);
    (selection, calibration)
}

/// The ⌈(n+1)(1−α)⌉ order-statistic index (1-based), clamped to `[1, n]` —
/// the conformal quantile convention the jackknife+ interval reads at.
fn conformal_index(n: usize, alpha: f32) -> usize {
    let raw = ((n as f32 + 1.0) * (1.0 - alpha)).ceil();
    if raw < 1.0 {
        return 1;
    }
    (raw as usize).clamp(1, n.max(1))
}

/// Exact O(n) top-k kNN with inverse-distance weighting over a
/// pre-normalized comparable matrix.
///
/// - `raw_query` is normalized into `scratch` with `scales` (feature
///   convention above; missing query dims read 0.0, extra dims ignored).
/// - `skip = Some(i)` excludes comparable `i` — the leave-some-out pass.
/// - `best_dist`/`best_value` are the caller-owned k selection slots
///   (ascending by distance, `+inf` = unfilled). Ties in distance keep the
///   EARLIEST-scanned comparable (lowest index): insertion position requires
///   a STRICTLY smaller distance to displace.
/// - Returns the inverse-distance weighted mean over the filled slots
///   (0.0 when nothing is comparable — no neighbors at all).
///
/// Issue-940 determinism law (synced f32 reductions): the weighted-mean
/// fold accumulates over the k selection slots in ascending-distance order,
/// and distance ties keep the LOWER comparable index first (the same
/// first-index tie rule as the selection). The order is pinned so any
/// wire-visible or quorum-compared answer is bit-identical across replays.
#[allow(clippy::too_many_arguments)] // one argument per owned scratch/knob — the substrate's own Calibrator::new convention
fn knn_weighted_mean(
    raw_query: &[f32],
    scales: &[f32],
    scratch: &mut [f32],
    flat: &[f32],
    values: &[f32],
    dim: usize,
    n: usize,
    k: usize,
    skip: Option<usize>,
    best_dist: &mut [f32],
    best_value: &mut [f32],
) -> f32 {
    if dim == 0 || n == 0 || k == 0 {
        return 0.0;
    }
    // 1. Normalize the query into the scratch buffer (zero-alloc).
    for (j, cell) in scratch.iter_mut().enumerate().take(dim) {
        let raw = raw_query.get(j).copied().unwrap_or(0.0);
        *cell = normalize_feature(raw, scales[j]);
    }
    // 2. Reset the k selection slots.
    for slot in best_dist.iter_mut() {
        *slot = f32::INFINITY;
    }
    // 3. Exact O(n·k)-worst top-k (an insertion that usually early-outs on
    //    the "not better than the worst slot" comparison).
    for i in 0..n {
        if skip == Some(i) {
            continue;
        }
        let row = &flat[i * dim..i * dim + dim];
        let mut d_sq = 0.0_f32;
        for j in 0..dim {
            let diff = scratch[j] - row[j];
            d_sq += diff * diff;
        }
        if !d_sq.is_finite() {
            continue;
        }
        let d = d_sq.sqrt();
        let mut pos = k;
        for (s, slot) in best_dist.iter().enumerate() {
            if *slot > d {
                pos = s;
                break;
            }
        }
        if pos == k {
            continue;
        }
        for s in (pos + 1..k).rev() {
            best_dist[s] = best_dist[s - 1];
            best_value[s] = best_value[s - 1];
        }
        best_dist[pos] = d;
        best_value[pos] = values[i];
    }
    // 4. Inverse-distance weighted mean over the filled slots.
    // Issue-940: the f32 fold order is PINNED — slots ascend by distance,
    // ties hold the lower comparable index first (see the doc above).
    let mut w_sum = 0.0_f32;
    let mut v_sum = 0.0_f32;
    for s in 0..k {
        let d = best_dist[s];
        if !d.is_finite() {
            break; // slots ascend; everything past the first unfilled slot is unfilled
        }
        let w = 1.0 / (d + KNN_DISTANCE_EPS);
        w_sum += w;
        v_sum += w * best_value[s];
    }
    if w_sum > 0.0 { v_sum / w_sum } else { 0.0 }
}

/// The kNN point predictor wrapped for the conformal overlay. Owns the
/// normalized comparable matrix, the scales, and every scratch buffer the
/// forecast needs — the whole forecast path is allocation-free. The query
/// features ride the [`PointForecaster`] `delay_state` parameter (the
/// seasonal family's "ignored" slot is this family's evidence slot); the
/// horizon is ignored (cross-sectional: always h=1).
struct KnnForecaster {
    /// Normalized features, flat `n × dim`.
    flat: Vec<f32>,
    /// Realized values, one per comparable.
    values: Vec<f32>,
    dim: usize,
    n: usize,
    k: usize,
    scales: Vec<f32>,
    scratch: Vec<f32>,
    best_dist: Vec<f32>,
    best_value: Vec<f32>,
}

impl KnnForecaster {
    fn new(
        flat: Vec<f32>,
        values: Vec<f32>,
        dim: usize,
        n: usize,
        k: usize,
        scales: Vec<f32>,
    ) -> Self {
        Self {
            flat,
            values,
            dim,
            n,
            k,
            scales,
            scratch: vec![0.0; dim.max(1)],
            best_dist: vec![f32::INFINITY; k],
            best_value: vec![0.0; k],
        }
    }

    fn empty(k: usize) -> Self {
        Self::new(Vec::new(), Vec::new(), 0, 0, k, Vec::new())
    }
}

impl PointForecaster for KnnForecaster {
    fn forecast_into(&mut self, delay_state: &[f32], _h: usize, out: &mut f32) {
        *out = knn_weighted_mean(
            delay_state,
            &self.scales,
            &mut self.scratch,
            &self.flat,
            &self.values,
            self.dim,
            self.n,
            self.k,
            None,
            &mut self.best_dist,
            &mut self.best_value,
        );
    }
}

/// Which calibration posture served an estimate (the Issue-952 amendment).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EstimatePosture {
    /// Split-conformal: the corpus was large enough to split
    /// (`n ≥ small_comps_max`); the interval rides the calibration-split
    /// residual pool.
    SplitConformal,
    /// Jackknife+ (Barber et al. 2021, arXiv:2105.05114): the corpus was too
    /// small to split (`n < small_comps_max`); the interval rides the
    /// leave-one-out residuals of every comparable.
    JackknifePlus,
}

/// The cross-sectional estimate head: kNN conditioner + conformal overlay
/// over a selection/calibration-split comps corpus (the Issue-952 amendment
/// — see the module docs).
///
/// Construct with [`EstimateHead::new`], optionally attach caller scales
/// ([`EstimateHead::with_scales`]), the split fraction
/// ([`EstimateHead::with_calibration_fraction`]), the small-comps threshold
/// ([`EstimateHead::with_small_comps_max`]) and domain bounds (the public
/// `domain` field), [`EstimateHead::calibrate`] once against a
/// [`ComparableSource`], then call [`EstimateHead::estimate_into`] (or the
/// posture-disclosing [`EstimateHead::estimate_with_posture`]) per query on
/// the zero-alloc hot path.
pub struct EstimateHead {
    k: usize,
    alpha: f32,
    grid: Option<Grid>,
    /// The caller's domain bounds `(min, max)` (the wire's
    /// `FieldConstraints.min/max`) used for the `p_below`/`p_above`
    /// out-of-domain mass disclosure. `None` renders both as 0.0. Set
    /// directly.
    pub domain: Option<(f64, f64)>,
    /// The calibration-split fraction (default 0.5; see [`split_partition`]).
    calibration_fraction: f32,
    /// PROVISIONAL small-comps threshold (Issue 952 amendment): corpora with
    /// `n < small_comps_max` serve the jackknife+ posture. Default 8 —
    /// derived later from the Issue-952 T4 comps-size axis; a named,
    /// documented config field, never a silent magic edge.
    small_comps_max: usize,
    scales_override: Option<Vec<f32>>,
    calibrator: ConformalIntervalCalibrator<KnnForecaster>,
    /// Jackknife+ abs LOO residuals per comparable (input order) — filled
    /// only in the [`EstimatePosture::JackknifePlus`] posture.
    jackknife_residuals: Vec<f32>,
    /// Query-time jackknife+ scratch (resized at calibrate, reused per
    /// query — the zero-alloc law).
    jk_mu: Vec<f32>,
    jk_lo: Vec<f32>,
    jk_hi: Vec<f32>,
    /// The posture the last calibration selected.
    posture: Option<EstimatePosture>,
    calibrated: bool,
}

impl EstimateHead {
    /// A head answering at miscoverage `alpha` (the interval level is
    /// `1 − alpha`; non-finite falls back to 0.05, values outside [0, 0.5]
    /// clamp), quantizing onto `grid` when present. `k` is the neighbor
    /// count (clamped to ≥ 1).
    pub fn new(k: usize, alpha: f32, grid: Option<Grid>) -> Self {
        let k = k.max(1);
        let alpha = sanitize_alpha(alpha);
        Self {
            k,
            alpha,
            grid,
            domain: None,
            calibrator: ConformalIntervalCalibrator::new(
                KnnForecaster::empty(k),
                ESTIMATE_CHANNELS,
                ESTIMATE_MAX_H,
                ESTIMATE_SEASONAL_M,
                ESTIMATE_RESIDUAL_CAPACITY,
                0.0, // no recency decay: the corpus is a snapshot
                DecayUnit::Step,
                ResidualMode::Paper,
                false,
            ),
            calibration_fraction: 0.5,
            small_comps_max: 8,
            scales_override: None,
            jackknife_residuals: Vec::new(),
            jk_mu: Vec::new(),
            jk_lo: Vec::new(),
            jk_hi: Vec::new(),
            posture: None,
            calibrated: false,
        }
    }

    /// Attach caller-supplied per-feature scales (overriding
    /// [`fit_scales`]). Convention: non-positive / non-finite entries mean
    /// "feature ignored"; entries past the corpus dimension are unused,
    /// missing dimensions are ignored features.
    pub fn with_scales(mut self, scales: Vec<f32>) -> Self {
        self.scales_override = Some(scales);
        self
    }

    /// Set the calibration-split fraction (default 0.5). Non-finite falls
    /// back to 0.5; values outside [0, 1] clamp.
    pub fn with_calibration_fraction(mut self, fraction: f32) -> Self {
        self.calibration_fraction = if fraction.is_finite() {
            fraction.clamp(0.0, 1.0)
        } else {
            0.5
        };
        self
    }

    /// Set the PROVISIONAL small-comps threshold (default 8; see the field
    /// doc). `0` disables the jackknife+ lane entirely.
    pub fn with_small_comps_max(mut self, small_comps_max: usize) -> Self {
        self.small_comps_max = small_comps_max;
        self
    }

    /// The posture the last calibration selected (`None` before
    /// calibration).
    pub fn posture(&self) -> Option<EstimatePosture> {
        self.posture
    }

    /// Whether [`EstimateHead::calibrate`] has run (successfully or against
    /// an empty corpus).
    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// The sanitized miscoverage level (the answer level is `1 − alpha`).
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Partition the corpus and build the calibration evidence from
    /// `source`'s comparables for `field_id`.
    ///
    /// Cold path, O(n²) distance work. The corpus first splits into
    /// SELECTION and CALIBRATION halves ([`split_partition`], the named
    /// [`SPLIT_SEED`] permutation — the Issue-952 exchangeability law). Then,
    /// by posture:
    ///
    /// - **Split-conformal** (`n ≥ small_comps_max`): the residual pool is
    ///   fed ONLY from calibration-split rows, each predicted leave-one-out
    ///   WITHIN the calibration split (the cross-conformal posture). The
    ///   selection split is RESERVED for estimator selection (Issue-952
    ///   T1/T2 bet search) — the single-k posture runs none.
    /// - **Jackknife+** (`n < small_comps_max`): the split is unused (LOO is
    ///   inherently leakage-safe); every comparable's abs LOO residual is
    ///   stored for the query-time jackknife+ interval.
    ///
    /// Rows with non-finite values/features or empty feature vectors are
    /// dropped; remaining rows are padded/truncated to the first row's
    /// feature width. A fresh calibrator is built every call (stale
    /// residuals from a previous corpus never leak into a new one). With
    /// fewer than two usable rows the pool stays empty and intervals
    /// collapse to the point (degenerate, disclosed).
    pub fn calibrate(&mut self, source: &dyn ComparableSource, field_id: &str) {
        let rows: Vec<Comparable> = source
            .comparables(field_id)
            .into_iter()
            .filter(|r| {
                r.value.is_finite()
                    && !r.features.is_empty()
                    && r.features.iter().all(|f| f.is_finite())
            })
            .collect();
        let dim = rows.first().map(|r| r.features.len()).unwrap_or(0);
        let scales = match &self.scales_override {
            Some(s) => scales_padded_to_dim(s, dim),
            None => fit_scales(&rows),
        };
        let n = rows.len();
        let mut flat = Vec::with_capacity(n * dim);
        let mut values = Vec::with_capacity(n);
        for row in &rows {
            for (j, &scale) in scales.iter().enumerate() {
                let raw = row.features.get(j).copied().unwrap_or(0.0);
                flat.push(normalize_feature(raw, scale));
            }
            values.push(row.value);
        }

        self.calibrator = ConformalIntervalCalibrator::new(
            KnnForecaster::empty(self.k),
            ESTIMATE_CHANNELS,
            ESTIMATE_MAX_H,
            ESTIMATE_SEASONAL_M,
            ESTIMATE_RESIDUAL_CAPACITY,
            0.0,
            DecayUnit::Step,
            ResidualMode::Paper,
            false,
        );

        let posture = if n < self.small_comps_max {
            EstimatePosture::JackknifePlus
        } else {
            EstimatePosture::SplitConformal
        };
        self.jackknife_residuals.clear();

        if n > 1 && dim > 0 {
            let mut raw = vec![0.0_f32; dim];
            let mut scratch = vec![0.0_f32; dim];
            let mut best_dist = vec![f32::INFINITY; self.k];
            let mut best_value = vec![0.0_f32; self.k];
            match posture {
                EstimatePosture::JackknifePlus => {
                    // Every comparable's abs LOO residual — the jackknife+ r_i
                    // consumed at query time. The split is unused here: LOO is
                    // inherently leakage-safe.
                    self.jackknife_residuals.reserve(n);
                    for i in 0..n {
                        for (j, cell) in raw.iter_mut().enumerate() {
                            *cell = rows[i].features.get(j).copied().unwrap_or(0.0);
                        }
                        let pred = knn_weighted_mean(
                            &raw,
                            &scales,
                            &mut scratch,
                            &flat,
                            &values,
                            dim,
                            n,
                            self.k,
                            Some(i),
                            &mut best_dist,
                            &mut best_value,
                        );
                        self.jackknife_residuals.push((values[i] - pred).abs());
                    }
                }
                EstimatePosture::SplitConformal => {
                    // The residual pool is fed ONLY from calibration-split
                    // rows, each predicted leave-one-out WITHIN the split —
                    // the cross-conformal posture (module doc). Feeding it
                    // from merged comps is the Issue-952 exchangeability sin.
                    let (selection, calibration) = split_partition(n, self.calibration_fraction);
                    // Reserved for estimator selection (Issue-952 T1/T2); the
                    // single-k posture runs no selection.
                    let _ = selection;
                    let n_cal = calibration.len();
                    let mut flat_cal = Vec::with_capacity(n_cal * dim);
                    let mut values_cal = Vec::with_capacity(n_cal);
                    for &i in &calibration {
                        for (j, &scale) in scales.iter().enumerate() {
                            let raw_j = rows[i].features.get(j).copied().unwrap_or(0.0);
                            flat_cal.push(normalize_feature(raw_j, scale));
                        }
                        values_cal.push(values[i]);
                    }
                    for (c, &i) in calibration.iter().enumerate() {
                        for (j, cell) in raw.iter_mut().enumerate() {
                            *cell = rows[i].features.get(j).copied().unwrap_or(0.0);
                        }
                        let pred = knn_weighted_mean(
                            &raw,
                            &scales,
                            &mut scratch,
                            &flat_cal,
                            &values_cal,
                            dim,
                            n_cal,
                            self.k,
                            Some(c),
                            &mut best_dist,
                            &mut best_value,
                        );
                        // update_residual(actual = residual, forecast = 0)
                        // stores the raw residual; `step()` ages the pool (a
                        // no-op at exp_lambda = 0, kept for lifecycle
                        // symmetry with forecast).
                        self.calibrator
                            .update_residual(values_cal[c] - pred, 0.0, 0, 1);
                        self.calibrator.step();
                    }
                }
            }
        }

        self.calibrator.forecaster = KnnForecaster::new(flat, values, dim, n, self.k, scales);
        self.jk_mu.resize(n, 0.0);
        self.jk_lo.resize(n, 0.0);
        self.jk_hi.resize(n, 0.0);
        self.posture = Some(posture);
        self.calibrated = true;
    }

    /// Estimate one query on the hot path, disclosing which calibration
    /// posture served ([`EstimatePosture`]).
    ///
    /// - **Split-conformal** (`n ≥ small_comps_max`): `point == median` is
    ///   the chosen-k kNN over the full corpus; `p10`/`p90` the conformal
    ///   quantiles of the calibration-split residual pool at `1 − alpha`.
    /// - **Jackknife+** (`n < small_comps_max`): `point == median` is the
    ///   LOO-prediction aggregate (mean over every comparable's
    ///   leave-one-out prediction of THIS query — an f32 fold in ascending
    ///   comparable-index order, pinned per the Issue-940 law); `p10`/`p90`
    ///   the ⌈(n+1)(1−α)⌉ order statistics of {μ₋ᵢ(q) − rᵢ} and
    ///   {μ₋ᵢ(q) + rᵢ}.
    ///
    /// Both postures quantize onto the caller grid, order-guard, and render
    /// `p_below`/`p_above` outside [`EstimateHead::domain`] (0.0 without
    /// bounds), `level = 1 − alpha`. Lazy-calibrates once against
    /// `(source, field_id)` exactly like [`EstimateHead::estimate_into`];
    /// zero-alloc after that first call.
    pub fn estimate_with_posture(
        &mut self,
        query_features: &[f32],
        source: &dyn ComparableSource,
        field_id: &str,
    ) -> (EstimateDistribution, EstimatePosture) {
        if !self.calibrated {
            self.calibrate(source, field_id);
        }
        let posture = self.posture.unwrap_or(EstimatePosture::SplitConformal);
        let mut dist = EstimateDistribution {
            point: 0.0,
            p10: 0.0,
            median: 0.0,
            p90: 0.0,
            p_below: 0.0,
            p_above: 0.0,
            level: 1.0 - self.alpha,
        };
        match posture {
            EstimatePosture::SplitConformal => {
                let mut point = 0.0_f32;
                self.calibrator
                    .forecaster
                    .forecast_into(query_features, 1, &mut point);
                let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, self.alpha);
                self.calibrator
                    .interval_from_point_into(point, 0, 1, self.alpha, &mut iv);
                self.finish_distribution(point, iv.lower, iv.upper, &mut dist);
            }
            EstimatePosture::JackknifePlus => {
                let f = &mut self.calibrator.forecaster;
                if f.n == 0 {
                    // No comparables: the degenerate zeros (documented).
                } else if f.n == 1 {
                    // A single comparable: the point IS its value; the
                    // interval collapses (no LOO evidence exists).
                    let v = f.values[0];
                    self.finish_distribution(v, v, v, &mut dist);
                } else {
                    // μ₋ᵢ(q) for every comparable — n ≤ small_comps_max−1
                    // kNN runs on the forecaster's own scratch (zero-alloc).
                    let mut center = 0.0_f32;
                    for i in 0..f.n {
                        let mu = knn_weighted_mean(
                            query_features,
                            &f.scales,
                            &mut f.scratch,
                            &f.flat,
                            &f.values,
                            f.dim,
                            f.n,
                            f.k,
                            Some(i),
                            &mut f.best_dist,
                            &mut f.best_value,
                        );
                        // Issue-940: the aggregate fold is PINNED to ascending
                        // comparable-index order.
                        center += mu;
                        self.jk_mu[i] = mu;
                    }
                    center /= f.n as f32;
                    let n = f.n;
                    for i in 0..n {
                        let r = self.jackknife_residuals[i];
                        self.jk_lo[i] = self.jk_mu[i] - r;
                        self.jk_hi[i] = self.jk_mu[i] + r;
                    }
                    self.jk_lo[..n].sort_unstable_by(|a, b| a.total_cmp(b));
                    self.jk_hi[..n].sort_unstable_by(|a, b| a.total_cmp(b));
                    let idx = conformal_index(n, self.alpha);
                    let (lo, hi) = (self.jk_lo[idx - 1], self.jk_hi[idx - 1]);
                    self.finish_distribution(center, lo, hi, &mut dist);
                }
            }
        }
        (dist, posture)
    }

    /// Shared render: grid quantization (monotone — order survives), the
    /// wire order guard, the domain tails, and the level.
    fn finish_distribution(&self, point: f32, lo: f32, hi: f32, out: &mut EstimateDistribution) {
        let mut p = f64::from(point);
        let mut lo = f64::from(lo);
        let mut hi = f64::from(hi);
        if let Some(grid) = &self.grid {
            p = quantize_onto_grid(p, grid);
            lo = quantize_onto_grid(lo, grid);
            hi = quantize_onto_grid(hi, grid);
        }
        let (lo, hi) = order_guarded(lo, p, hi);
        let (p_below, p_above) = domain_tails(lo, hi, p, self.domain);
        *out = EstimateDistribution {
            point: p,
            p10: lo,
            median: p,
            p90: hi,
            p_below,
            p_above,
            level: 1.0 - self.alpha,
        };
    }
}

impl EstimateHead {
    /// Estimate one query on the hot path. Writes the wire-shaped
    /// distribution into `out` — the [`EstimateHead::estimate_with_posture`]
    /// computation with the posture discarded. See that method for the
    /// per-posture semantics; lazy-calibrates once against
    /// `(source, field_id)` (the documented cold path); zero-alloc after.
    pub fn estimate_into(
        &mut self,
        query_features: &[f32],
        source: &dyn ComparableSource,
        field_id: &str,
        out: &mut EstimateDistribution,
    ) {
        let (dist, _posture) = self.estimate_with_posture(query_features, source, field_id);
        *out = dist;
    }
}

// ===== G1 floor-comparison plumbing (measurement-only) =====

/// A deterministic synthetic cross-sectional corpus: features are a fixed
/// function of the step index (the informative phase pair of a known period
/// plus one pure-noise distractor dimension), values are a fixed linear
/// signal on that phase plus noise-ish LCG jitter. Generated by an integer
/// xorshift with a FIXED seed ([`SYNTH_SEED`] in tests) — NO clock, NO rng
/// dependency, bit-reproducible. MEASUREMENT-ONLY: the deployment corpus
/// comes from state-parsed tables and the outcomes ledger, never from this
/// generator.
#[derive(Clone, Debug)]
pub struct SyntheticCrossSectional {
    /// Corpus name (for the floor report).
    pub name: String,
    /// Per-step feature rows (3 dims: sin, cos, distractor noise).
    pub features: Vec<Vec<f32>>,
    /// Per-step realized values (the trajectory fed to the floor harness).
    pub values: Vec<f32>,
    /// Recommended warmup (4 periods, bounded by n/4).
    pub recommended_warmup: usize,
}

impl SyntheticCrossSectional {
    /// Generate `n` steps at seasonal `period` from xorshift `seed`.
    pub fn generate(n: usize, period: usize, seed: u64) -> Self {
        let period = period.max(1);
        let mut rng = Xorshift64::new(seed);
        let mut features = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        for t in 0..n {
            let phase = TAU * (t as f32) / (period as f32);
            let f0 = phase.sin();
            let f1 = phase.cos();
            let f2 = rng.next_unit() - 0.5;
            features.push(vec![f0, f1, f2]);
            let v = 3.0 + 2.0 * f0 + 0.5 * (rng.next_unit() - 0.5);
            values.push(v);
        }
        let recommended_warmup = (4 * period).min(n / 4).max(1);
        Self {
            name: format!("synthetic_cross_sectional_n{n}_period{period}_seed{seed:#x}"),
            features,
            values,
            recommended_warmup,
        }
    }
}

/// xorshift64* — deterministic integer generator for the synthetic corpus
/// (measurement-only; a zero state never advances, hence the clamp).
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1) from the top 24 bits (f32-exact resolution).
    fn next_unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// The estimate head adapted to the Report-the-Floor harness
/// ([`UqPrimitiveUnderTest`]). The online protocol: each step conditions on
/// the synthetic feature row, then learns the realized value into an
/// in-memory source and re-calibrates (the per-step O(n²) refit is a
/// measurement-only cost, documented; the deployment engine calibrates per
/// corpus epoch instead). The values the harness observes ARE the corpus
/// values `run_floor_comparison` feeds — no lookahead: `predict_next` reads
/// only the pool of ALREADY-observed rows. The Issue-952 split law holds BY
/// CONSTRUCTION on this rolling harness: every observe re-runs the head's
/// calibrate, which re-partitions (a rolling harness may re-split per step —
/// the documented choice here), and the residual pool is fed only inside
/// calibrate from calibration-side rows. Nothing scores or asserts on merged
/// comps.
pub struct EstimateFloorHarness {
    head: EstimateHead,
    synth: SyntheticCrossSectional,
    source: InMemorySource,
    step: usize,
    alpha: f32,
}

impl EstimateFloorHarness {
    /// A harness over `synth` with neighbor count `k` at miscoverage
    /// `alpha`.
    pub fn new(k: usize, alpha: f32, synth: SyntheticCrossSectional) -> Self {
        let alpha = sanitize_alpha(alpha);
        Self {
            head: EstimateHead::new(k, alpha, None),
            synth,
            source: InMemorySource::new(),
            step: 0,
            alpha,
        }
    }
}

impl UqPrimitiveUnderTest for EstimateFloorHarness {
    fn name(&self) -> &str {
        "katgpt-extract::estimate kNN + conformal overlay (synthetic cross-sectional)"
    }

    fn predict_next(&mut self) -> PredictiveOutput {
        if self.step >= self.synth.features.len() {
            return PredictiveOutput::empty();
        }
        // Measurement-only clone: the estimate path reads a feature row by
        // reference; the synthetic rows are owned Vecs.
        let feats = self.synth.features[self.step].clone();
        let mut dist = EstimateDistribution {
            point: 0.0,
            p10: 0.0,
            median: 0.0,
            p90: 0.0,
            p_below: 0.0,
            p_above: 0.0,
            level: 1.0 - self.alpha,
        };
        self.head
            .estimate_into(&feats, &self.source, ESTIMATE_FLOOR_FIELD, &mut dist);
        PredictiveOutput::from_interval(PredictiveInterval::new(
            dist.p10 as f32,
            dist.point as f32,
            dist.p90 as f32,
            self.alpha,
        ))
    }

    fn observe(&mut self, y: f32) {
        if self.step >= self.synth.features.len() {
            return;
        }
        let feats = self.synth.features[self.step].clone();
        self.source.push(
            ESTIMATE_FLOOR_FIELD,
            Comparable {
                features: feats,
                value: y,
            },
        );
        self.head.calibrate(&self.source, ESTIMATE_FLOOR_FIELD);
        self.step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::conformal::{OverallVerdict, run_floor_comparison};

    const EPS: f32 = 1e-4;
    const EPS64: f64 = 1e-4;

    fn comp(features: &[f32], value: f32) -> Comparable {
        Comparable {
            features: features.to_vec(),
            value,
        }
    }

    #[test]
    fn knn_weighted_mean_is_hand_computed() {
        // 1-D corpus, unit scales, k=2, query 0.9. Eight rows (n ≥
        // small_comps_max → the split-conformal posture; the point is the
        // full-set kNN). Five far filler rows never enter the k=2
        // neighborhood, so the hand math stays the toy's:
        // dists: c0=0.9, c1=0.1 (fillers ≥ 99) → neighbors c1 (20.0), c0 (10.0).
        // w1 = 1/(0.1+1e-6) = 9.99990, w0 = 1/(0.9+1e-6) = 1.11111
        // pred = (20·w1 + 10·w0)/(w1+w0) = 211.109/11.111 = 19.0000…
        let mut source = InMemorySource::new();
        source.insert_table(
            "price",
            vec![
                comp(&[0.0], 10.0),
                comp(&[1.0], 20.0),
                comp(&[3.0], 40.0),
                comp(&[100.0], 1000.0),
                comp(&[101.0], 1010.0),
                comp(&[102.0], 1020.0),
                comp(&[103.0], 1030.0),
                comp(&[104.0], 1040.0),
            ],
        );
        let mut head = EstimateHead::new(2, 0.2, None).with_scales(vec![1.0]);
        head.calibrate(&source, "price");
        assert!(head.is_calibrated());
        assert_eq!(head.posture(), Some(EstimatePosture::SplitConformal));
        let (out, posture) = head.estimate_with_posture(&[0.9], &source, "price");
        assert_eq!(posture, EstimatePosture::SplitConformal);
        assert!(
            (out.point - 19.0).abs() < EPS64,
            "hand-computed kNN mean 19.0, got {}",
            out.point
        );
        assert_eq!(out.median, out.point, "median == point by construction");
        assert!((out.level - 0.8).abs() < EPS);
    }

    #[test]
    fn jackknife_plus_hand_computed() {
        // n=3 (< small_comps_max) → the jackknife+ posture. Hand-computed
        // (scales [1], k=2, query 0.9, α=0.2):
        // LOO query predictions: μ₋₀=20.9094, μ₋₁=19.0000, μ₋₂=19.0001
        //   → center = 19.6365 (ascending-index f32 fold).
        // LOO abs residuals: r₀=15, r₁=0, r₂=24.
        // conformal index ⌈4·0.8⌉=4 → clamp 3 (the max of each set):
        //   p10 = max(5.9094, 19.0000, −4.9999) = 19.0000
        //   p90 = max(35.9094, 19.0000, 43.0001) = 43.0001
        let mut source = InMemorySource::new();
        source.insert_table(
            "price",
            vec![comp(&[0.0], 10.0), comp(&[1.0], 20.0), comp(&[3.0], 40.0)],
        );
        let mut head = EstimateHead::new(2, 0.2, None).with_scales(vec![1.0]);
        head.calibrate(&source, "price");
        assert_eq!(head.posture(), Some(EstimatePosture::JackknifePlus));
        let (out, posture) = head.estimate_with_posture(&[0.9], &source, "price");
        assert_eq!(posture, EstimatePosture::JackknifePlus);
        assert!(
            (out.point - 19.6365).abs() < 1e-2,
            "LOO aggregate 19.6365, got {}",
            out.point
        );
        assert!((out.p10 - 19.0).abs() < 1e-2, "p10 19.0, got {}", out.p10);
        assert!(
            (out.p90 - 43.0001).abs() < 1e-2,
            "p90 43.0001, got {}",
            out.p90
        );
        assert!(out.p10 <= out.median && out.median <= out.p90);
    }

    #[test]
    fn split_partition_is_deterministic_and_covers() {
        let (sel_a, cal_a) = split_partition(10, 0.5);
        let (sel_b, cal_b) = split_partition(10, 0.5);
        assert_eq!(sel_a, sel_b, "fixed seed → identical partitions");
        assert_eq!(cal_a, cal_b);
        // Disjoint + exhaustive over the input indices.
        let mut all = sel_a.clone();
        all.extend_from_slice(&cal_a);
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<usize>>());
        // Both splits non-empty; sizes follow the fraction rule.
        assert_eq!(cal_a.len(), 5);
        assert_eq!(sel_a.len(), 5);
        // Degenerate sizes.
        let (sel1, cal1) = split_partition(1, 0.5);
        assert!(sel1.is_empty() && cal1.len() == 1);
        let (sel0, cal0) = split_partition(0, 0.5);
        assert!(sel0.is_empty() && cal0.is_empty());
        // Fraction clamps keep both sides non-empty at n ≥ 2.
        let (sel_all, cal_nearly) = split_partition(10, 1.0);
        assert_eq!(cal_nearly.len(), 9, "fraction 1.0 clamps to n−1 cal rows");
        assert_eq!(sel_all.len(), 1);
    }

    #[test]
    fn posture_boundary_follows_small_comps_max() {
        // n = 8 == small_comps_max → split-conformal (the law is n < max).
        let rows: Vec<Comparable> = (0..8).map(|i| comp(&[i as f32], i as f32 * 10.0)).collect();
        let mut source = InMemorySource::new();
        source.insert_table("p", rows);
        let mut head = EstimateHead::new(2, 0.2, None).with_scales(vec![1.0]);
        head.calibrate(&source, "p");
        assert_eq!(head.posture(), Some(EstimatePosture::SplitConformal));
        // n = 7 < 8 → jackknife+.
        let rows7: Vec<Comparable> = (0..7).map(|i| comp(&[i as f32], i as f32 * 10.0)).collect();
        let mut source7 = InMemorySource::new();
        source7.insert_table("p", rows7);
        let mut head7 = EstimateHead::new(2, 0.2, None).with_scales(vec![1.0]);
        head7.calibrate(&source7, "p");
        assert_eq!(head7.posture(), Some(EstimatePosture::JackknifePlus));
        // The threshold is a named, settable config field — not a magic edge.
        let mut head0 = EstimateHead::new(2, 0.2, None)
            .with_scales(vec![1.0])
            .with_small_comps_max(1);
        head0.calibrate(&source7, "p");
        assert_eq!(
            head0.posture(),
            Some(EstimatePosture::SplitConformal),
            "small_comps_max=1 disables the jackknife lane"
        );
    }

    #[test]
    fn knn_tie_breaks_to_lower_index() {
        // k=1, two equidistant comparables (±1 from the query): the
        // earliest-scanned (lower-index) comparable must win outright. Eight
        // rows (n ≥ small_comps_max → the split-conformal posture, whose
        // point is the plain full-set kNN); six far fillers never enter the
        // k=1 neighborhood.
        let mut source = InMemorySource::new();
        source.insert_table(
            "f",
            vec![
                comp(&[-1.0], 100.0),
                comp(&[1.0], 200.0),
                comp(&[100.0], 1000.0),
                comp(&[101.0], 1010.0),
                comp(&[102.0], 1020.0),
                comp(&[103.0], 1030.0),
                comp(&[104.0], 1040.0),
                comp(&[105.0], 1050.0),
            ],
        );
        let mut head = EstimateHead::new(1, 0.2, None).with_scales(vec![1.0]);
        head.calibrate(&source, "f");
        let mut out = EstimateDistribution {
            point: 0.0,
            p10: 0.0,
            median: 0.0,
            p90: 0.0,
            p_below: 0.0,
            p_above: 0.0,
            level: 0.0,
        };
        head.estimate_into(&[0.0], &source, "f", &mut out);
        assert!(
            (out.point - 100.0).abs() < EPS64,
            "lower-index tie-break expected 100.0, got {}",
            out.point
        );
    }

    #[test]
    fn fit_scales_excludes_constant_features() {
        let rows = vec![
            comp(&[0.0, 5.0], 1.0),
            comp(&[1.0, 5.0], 2.0),
            comp(&[2.0, 5.0], 3.0),
        ];
        let scales = fit_scales(&rows);
        assert_eq!(scales.len(), 2);
        assert!(
            scales[0].is_finite() && scales[0] > 0.0,
            "varying dim scales"
        );
        assert!(
            scales[1].is_infinite(),
            "constant dim must be an ignored feature"
        );
        // Distance invariance: with the constant dim scaling to zero
        // contribution, two comparables differing only in it are equidistant
        // to any query → the lower index wins the tie. Eight rows (n ≥
        // small_comps_max → the split-conformal posture; the point is the
        // plain full-set kNN), the constant dim constant across ALL rows so
        // fit_scales still marks it ignored.
        let mut source = InMemorySource::new();
        source.insert_table(
            "f",
            vec![
                comp(&[5.0, 5.0], 10.0),
                comp(&[5.0, 9.0], 20.0),
                comp(&[5.0, 100.0], 1000.0),
                comp(&[5.0, 101.0], 1010.0),
                comp(&[5.0, 102.0], 1020.0),
                comp(&[5.0, 103.0], 1030.0),
                comp(&[5.0, 104.0], 1040.0),
                comp(&[5.0, 105.0], 1050.0),
            ],
        );
        let mut head = EstimateHead::new(1, 0.2, None);
        head.calibrate(&source, "f");
        let mut out = EstimateDistribution {
            point: 0.0,
            p10: 0.0,
            median: 0.0,
            p90: 0.0,
            p_below: 0.0,
            p_above: 0.0,
            level: 0.0,
        };
        head.estimate_into(&[5.0, 7.0], &source, "f", &mut out);
        assert!(
            (out.point - 10.0).abs() < EPS64,
            "constant feature ignored → equidistant tie → lower index, got {}",
            out.point
        );
    }

    #[test]
    fn quantize_onto_grid_linear_and_log() {
        let linear = Grid {
            kind: GridKind::Linear,
            step: 0.5,
        };
        assert!((quantize_onto_grid(10.34, &linear) - 10.5).abs() < 1e-9);
        // −2.3/0.5 = −4.6 → round half-away-from-zero → −5 → −2.5.
        assert!((quantize_onto_grid(-2.3, &linear) - (-2.5)).abs() < 1e-9);
        let log10 = Grid {
            kind: GridKind::Log,
            step: 10.0,
        };
        assert!((quantize_onto_grid(95.0, &log10) - 100.0).abs() < 1e-9);
        assert!((quantize_onto_grid(0.095, &log10) - 0.1).abs() < 1e-9);
        // Log grids leave non-positive values unquantized.
        assert_eq!(quantize_onto_grid(-5.0, &log10), -5.0);
        assert_eq!(quantize_onto_grid(0.0, &log10), 0.0);
        // Degenerate steps pass through.
        let bad = Grid {
            kind: GridKind::Linear,
            step: 0.0,
        };
        assert_eq!(quantize_onto_grid(1.234, &bad), 1.234);
        let bad_log = Grid {
            kind: GridKind::Log,
            step: 1.0,
        };
        assert_eq!(quantize_onto_grid(95.0, &bad_log), 95.0);
    }

    #[test]
    fn estimate_interval_is_ordered_after_calibration() {
        // A deterministic linear corpus: value = 2x + 1 over x = 0..10.
        let mut source = InMemorySource::new();
        let rows: Vec<Comparable> = (0..10)
            .map(|x| comp(&[x as f32], 2.0 * x as f32 + 1.0))
            .collect();
        source.insert_table("y", rows);
        let mut head = EstimateHead::new(3, 0.2, None);
        head.calibrate(&source, "y");
        assert_eq!(
            head.posture(),
            Some(EstimatePosture::SplitConformal),
            "n=10 ≥ small_comps_max → split-conformal"
        );
        let mut out = EstimateDistribution {
            point: 0.0,
            p10: 0.0,
            median: 0.0,
            p90: 0.0,
            p_below: 0.0,
            p_above: 0.0,
            level: 0.0,
        };
        head.estimate_into(&[4.5], &source, "y", &mut out);
        assert!(
            out.p10 <= out.median && out.median <= out.p90,
            "wire ordering violated: {} ≤ {} ≤ {}",
            out.p10,
            out.median,
            out.p90
        );
        assert!(
            (out.point - 10.0).abs() < 1.0,
            "linear corpus midpoint ≈ 10.0, got {}",
            out.point
        );
    }

    #[test]
    fn grid_quantizes_the_estimate_answer() {
        // Four rows → the jackknife+ posture; the grid post-processing is
        // shared by both postures, so this also pins quantization on the
        // small-comps lane.
        let mut source = InMemorySource::new();
        source.insert_table(
            "p",
            vec![
                comp(&[0.0], 10.0),
                comp(&[1.0], 20.0),
                comp(&[2.0], 30.0),
                comp(&[3.0], 40.0),
            ],
        );
        let grid = Grid {
            kind: GridKind::Linear,
            step: 5.0,
        };
        let mut head = EstimateHead::new(2, 0.2, Some(grid)).with_scales(vec![1.0]);
        head.calibrate(&source, "p");
        let mut out = EstimateDistribution {
            point: 0.0,
            p10: 0.0,
            median: 0.0,
            p90: 0.0,
            p_below: 0.0,
            p_above: 0.0,
            level: 0.0,
        };
        head.estimate_into(&[1.1], &source, "p", &mut out);
        for v in [out.point, out.p10, out.median, out.p90] {
            assert!(
                (v / 5.0 - (v / 5.0).round()).abs() < 1e-9,
                "{v} is not a multiple of the grid step 5.0"
            );
        }
        assert!(out.p10 <= out.median && out.median <= out.p90);
    }

    #[test]
    fn domain_tails_sum_leq_one_and_zero_without_bounds() {
        // Direct kernel: interval [4, 6], point 5.
        assert_eq!(domain_tails(4.0, 6.0, 5.0, None), (0.0, 0.0));
        let (below, above) = domain_tails(4.0, 6.0, 5.0, Some((0.0, 10.0)));
        assert_eq!((below, above), (0.0, 0.0), "domain swallows the support");
        let (below, above) = domain_tails(4.0, 6.0, 5.0, Some((4.5, 5.5)));
        assert!((below - 5.0 / 12.0).abs() < 1e-9, "got {below}");
        assert!((above - 5.0 / 12.0).abs() < 1e-9, "got {above}");
        // Degenerate zero-width interval: a point mass.
        let (below, above) = domain_tails(5.0, 5.0, 5.0, Some((6.0, 10.0)));
        assert_eq!((below, above), (1.0, 0.0));
        // Inverted domain never exceeds the wire's p_below+p_above ≤ 1 law.
        let (below, above) = domain_tails(5.0, 5.0, 5.0, Some((6.0, 4.0)));
        assert!(below + above <= 1.0 + 1e-12);
    }

    #[test]
    fn estimate_carries_domain_tails_through_estimate_into() {
        let mut source = InMemorySource::new();
        source.insert_table(
            "p",
            vec![
                comp(&[0.0], 10.0),
                comp(&[1.0], 20.0),
                comp(&[2.0], 30.0),
                comp(&[3.0], 40.0),
            ],
        );
        let mut head = EstimateHead::new(2, 0.2, None).with_scales(vec![1.0]);
        head.domain = Some((0.0, 25.0));
        head.calibrate(&source, "p");
        let mut out = EstimateDistribution {
            point: 0.0,
            p10: 0.0,
            median: 0.0,
            p90: 0.0,
            p_below: 0.0,
            p_above: 0.0,
            level: 0.0,
        };
        head.estimate_into(&[0.9], &source, "p", &mut out);
        assert!(out.p_below >= 0.0 && out.p_below <= 1.0);
        assert!(out.p_above >= 0.0 && out.p_above <= 1.0);
        assert!(out.p_below + out.p_above <= 1.0 + 1e-9);
        // No bounds → both tails exactly zero.
        let mut head2 = EstimateHead::new(2, 0.2, None).with_scales(vec![1.0]);
        head2.calibrate(&source, "p");
        let mut out2 = EstimateDistribution {
            point: 0.0,
            p10: 0.0,
            median: 0.0,
            p90: 0.0,
            p_below: 0.0,
            p_above: 0.0,
            level: 0.0,
        };
        head2.estimate_into(&[0.9], &source, "p", &mut out2);
        assert_eq!((out2.p_below, out2.p_above), (0.0, 0.0));
    }

    #[test]
    fn outcomes_ledger_capacity_and_counters() {
        let mut ledger = OutcomesLedger::new(3);
        for i in 0..5_u32 {
            ledger.append("price", comp(&[i as f32], i as f32 * 10.0));
        }
        assert_eq!(ledger.len(), 3, "capacity cap holds");
        assert_eq!(ledger.appends(), 5);
        assert_eq!(ledger.evictions(), 2, "oldest two rows evicted");
        let rows = ledger.comparables("price");
        assert_eq!(ledger.queries(), 1);
        let values: Vec<f32> = rows.iter().map(|r| r.value).collect();
        assert_eq!(values, vec![20.0, 30.0, 40.0], "FIFO kept the newest");
        // Unknown field: empty corpus, query still counted.
        assert!(ledger.comparables("other").is_empty());
        assert_eq!(ledger.queries(), 2);
    }

    #[test]
    fn floor_comparison_plumbing_returns_report() {
        // The G1 plumbing test: the harness must run through the real floor
        // harness and produce a populated report. ANY verdict — the G1 gate
        // EVALUATES at T0.6; this asserts the plumbing, never a win.
        let synth = SyntheticCrossSectional::generate(240, 12, SYNTH_SEED);
        let mut harness = EstimateFloorHarness::new(3, 0.05, synth.clone());
        let report = run_floor_comparison(
            &mut harness,
            &synth.values,
            0.05,
            synth.recommended_warmup,
            "synthetic_cross_sectional_v1",
        );
        report.pretty_print();
        assert!(report.n_scored > 0, "scored steps must exist after warmup");
        assert!(
            !report.is_not_applicable(),
            "the head produces scorable intervals after warmup"
        );
        assert!(matches!(
            report.overall,
            OverallVerdict::BeatsFloor
                | OverallVerdict::TiesFloor
                | OverallVerdict::LosesToFloor
                | OverallVerdict::Mixed
        ));
    }

    #[test]
    fn uncalibrated_head_serves_degenerate_but_valid_output() {
        // Empty corpus → calibrate once → zeros, still wire-ordered.
        let source = InMemorySource::new();
        let mut head = EstimateHead::new(3, 0.2, None);
        head.calibrate(&source, "none");
        let mut out = EstimateDistribution {
            point: 0.0,
            p10: 0.0,
            median: 0.0,
            p90: 0.0,
            p_below: 0.0,
            p_above: 0.0,
            level: 0.0,
        };
        head.estimate_into(&[1.0], &source, "none", &mut out);
        assert_eq!(out.point, 0.0);
        assert!(out.p10 <= out.median && out.median <= out.p90);
        // A non-finite query degrades to the zero point (finite distance
        // guard), never NaN out.
        head.estimate_into(&[f32::NAN], &source, "none", &mut out);
        assert!(out.point.is_finite());
    }

    #[test]
    fn synthetic_corpus_is_deterministic() {
        let a = SyntheticCrossSectional::generate(64, 8, SYNTH_SEED);
        let b = SyntheticCrossSectional::generate(64, 8, SYNTH_SEED);
        assert_eq!(a.values, b.values);
        assert_eq!(a.features, b.features);
        assert!((a.values[0] - 3.0).abs() < 3.0, "values near the level 3.0");
    }

    // ---- G4: allocation gate (crate::alloc_gate; see its module doc) -------

    #[test]
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    fn estimate_into_second_call_is_zero_alloc() {
        use crate::alloc_gate::{ALLOC_COUNT, MeasuringWindow};
        use std::sync::atomic::Ordering;

        // 12 comparables (>= small_comps_max 8 -> the split-conformal
        // posture), 2-feature schema.
        let mut source = InMemorySource::new();
        for i in 0..12 {
            let x = i as f32 * 0.5;
            source.push(
                "price",
                Comparable {
                    features: vec![x, x * 2.0],
                    value: 10.0 + 3.0 * x,
                },
            );
        }
        let mut head = EstimateHead::new(3, 0.05, None);
        let mut out = EstimateDistribution {
            point: 0.0,
            p10: 0.0,
            median: 0.0,
            p90: 0.0,
            p_below: 0.0,
            p_above: 0.0,
            level: 0.0,
        };
        let query = [1.0, 2.0];
        // Call #1: the documented lazy-calibrate cold path (MAY allocate —
        // it clones the corpus and fits scales/calibration).
        head.estimate_into(&query, &source, "price", &mut out);
        assert!(head.is_calibrated());
        assert!(out.point.is_finite());

        // Call #2 on the same (source, field_id): the pinned zero-alloc
        // claim (module doc — scratch is owned by the head after
        // calibration).
        let _window = MeasuringWindow::enter();
        ALLOC_COUNT.store(0, Ordering::Relaxed);
        head.estimate_into(&query, &source, "price", &mut out);
        assert_eq!(
            ALLOC_COUNT.load(Ordering::Relaxed),
            0,
            "estimate_into allocated after calibration"
        );
        drop(_window);
        assert!(out.point.is_finite());
    }
}
