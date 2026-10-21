//! The calculation brain — budgeted search over authored bet compositions
//! for the Estimate class (Issue 952 T1–T3; feature `bet_search`).
//!
//! The estimate head ([`crate::estimate`]) answers with one fixed kNN. This
//! module answers with the best COMPOSITION of authored bets
//! ([`bets::DEFAULT_BETS`]: median, kNN{3,5,10}, ratio scaling, recency,
//! two-feature ratio), chosen by a bounded search ([`search`]) over the
//! comps — modelless: every bet is a closed form, the search selects and
//! never trains.
//!
//! # The split law (Issue 952, "Honest caveats" G1)
//!
//! [`BetSearchHead::calibrate`] partitions the comps with
//! `estimate::split_partition` (the same seeded rule as the estimate head)
//! and restores SOURCE order inside each split:
//!
//! 1. **Selection split** — the search scores every candidate composition
//!    by its leave-one-out CRPS here, and the picked composition's
//!    predictor is fitted on these rows only (feature scales included).
//! 2. **Calibration split** — never seen by the search. Each calibration
//!    row is predicted by the selection-fitted composition; the absolute
//!    residuals' ⌈(n+1)(1−α)⌉ order statistic is the symmetric interval
//!    half-width (split conformal). Selecting on these rows would break
//!    exchangeability and overstate coverage.
//!
//! # Demote to the floor
//!
//! The head serves the conformal-naive floor
//! (`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`, m=1, fed the
//! comps in source order — `katgpt_core::conformal::FloorAdapter`) when the
//! split leaves nothing to search with (fewer than 2 selection rows or no
//! calibration row), or when the floor-relative skill gate
//! ([`SkillGate`]) reads at or below one half. The posture that served is
//! disclosed ([`BetPosture`]).
//!
//! # T3 — the skill gate and the ledger intake
//!
//! `skill = clamp((CRPS_floor − CRPS_bet) / max(CRPS_floor, ε), −1, 1)`,
//! `gate = σ(β · skill)` — normalized, so scores are comparable across comps
//! sets and states (raw CRPS is scale-dependent). Both CRPS terms are
//! measured on the SAME selection rows (rows 1.. in source order — the
//! floor's last-value forecast has no prediction for row 0).
//! [`BetScoreIntake`] is the consolidation stub: a fixed-capacity ring of
//! [`BetScoreRecord`]s, consolidated later, never at answer time.
//! **Determinism (Issue 940):** every score here folds sequentially (rows
//! ascending, members ascending) and every pick breaks ties to the lowest
//! mask, so a record may be committed, hashed or replay-compared. A caller
//! that feeds the intake from any other fold must mark the intake
//! local-only at its call site.
//!
//! # Hot path
//!
//! [`BetSearchHead::estimate_into`] is allocation-free after calibration
//! (scratch owned by the head). [`BetSearchHead::calibrate`] is the cold
//! path: the split, the search's LOO columns (`O(n²)` per materialized
//! bet), and the calibration residual pass.
//!
//! # Boundary (Issue 952 T6, owner-flagged)
//!
//! This is the GENERIC kernel — PUCT over authored bets plus a sigmoid
//! skill gate, pure modelless math, the same tier as moka/PUCT. The
//! extraction PRODUCT composition (a self-evolving economy over bet
//! outcomes, the calculation brain as a sample farm feeding trained lanes,
//! hosted-only escalation postures) is riir-rethink's charter
//! (Proposal 016 §9) and never lands in this public crate.

pub mod bets;
pub mod search;

#[cfg(feature = "bet_search_puct")]
use katgpt_core::chance_puct::ChancePuctConfig;
use katgpt_core::conformal::{FloorAdapter, PredictiveInterval, UqPrimitiveUnderTest};
use katgpt_core::exact_sigmoid;

use crate::estimate::{Comparable, conformal_index, fit_scales, sanitize_alpha, split_partition};

use bets::{AuthoredBet, Bet, BetScratch, CompsView, DEFAULT_BETS, normalize_rows};
#[cfg(feature = "bet_search_puct")]
use search::puct_search;
use search::{BetEvaluator, SearchOutcome, exhaustive_search};

/// Denominator floor for the skill ratio: a near-constant comps set can
/// drive the floor's CRPS to ~0.
pub const SKILL_DENOM_EPS: f32 = 1e-6;

/// How the head picks its composition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SearchMode {
    /// `katgpt_core::chance_puct` over the composition tree (opt-in,
    /// feature `bet_search_puct` — it did not beat [`Self::Exhaustive`]
    /// at equal budget, Issue 952 T4).
    #[cfg(feature = "bet_search_puct")]
    Puct { cfg: ChancePuctConfig, seed: u64 },
    /// Score the first `evaluations` canonical compositions
    /// (`usize::MAX` = the whole space).
    Exhaustive { evaluations: usize },
    /// No search: serve this composition mask (a control arm).
    Fixed { mask: u32 },
}

/// The floor-relative skill gate (T3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkillGate {
    /// Sigmoid slope over the clamped skill.
    pub beta: f32,
}

impl Default for SkillGate {
    fn default() -> Self {
        Self { beta: 8.0 }
    }
}

impl SkillGate {
    /// `clamp((floor − bet) / max(floor, ε), −1, 1)`; `−1` when either
    /// score is not finite (an unscoreable bet never outranks the floor).
    #[inline]
    pub fn skill(crps_floor: f32, crps_bet: f32) -> f32 {
        if !(crps_floor.is_finite() && crps_bet.is_finite()) {
            return -1.0;
        }
        ((crps_floor - crps_bet) / crps_floor.max(SKILL_DENOM_EPS)).clamp(-1.0, 1.0)
    }

    /// `σ(β · skill)` — strictly above 0.5 ⇔ the bet beat the floor.
    #[inline]
    pub fn gate(&self, skill: f32) -> f32 {
        exact_sigmoid(self.beta * skill)
    }
}

/// Head configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct BetSearchConfig {
    /// The authored roster (bit order of every mask; ≤ `search::MAX_ROSTER`).
    pub roster: Vec<AuthoredBet>,
    /// Largest composition size.
    pub max_depth: u8,
    /// Two-tailed miscoverage of the served interval.
    pub alpha: f32,
    /// Fraction of comps reserved for calibration.
    pub calibration_fraction: f32,
    pub mode: SearchMode,
    pub gate: SkillGate,
    /// Demote to the floor when the gate reads ≤ 0.5. `false` serves the
    /// searched composition whenever one is scoreable (the bench's raw
    /// control arms); the split-starved floor posture still applies.
    pub demote: bool,
}

impl Default for BetSearchConfig {
    fn default() -> Self {
        Self {
            roster: DEFAULT_BETS.to_vec(),
            max_depth: 3,
            alpha: 0.2,
            calibration_fraction: 0.5,
            mode: SearchMode::Exhaustive {
                evaluations: usize::MAX,
            },
            gate: SkillGate::default(),
            demote: true,
        }
    }
}

/// Which arm served an estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BetPosture {
    /// The searched composition, split-conformal interval.
    Bet,
    /// The conformal-naive floor (too few comps to split, or the gate
    /// demoted the bet).
    Floor,
}

/// What the last calibration decided, for reports and the intake.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetScoreRecord {
    pub search: SearchOutcome,
    /// The floor's CRPS on the skill rows.
    pub crps_floor: f32,
    /// The picked composition's CRPS on the same rows.
    pub crps_bet: f32,
    pub skill: f32,
    pub gate: f32,
    pub posture: BetPosture,
    /// `true` when the calibration split is too small for the conformal
    /// index at this α (the interval reads the max residual and is
    /// under-covered by construction).
    pub undersized_calibration: bool,
}

/// T3 consolidation stub — a fixed-capacity ring of score records.
#[derive(Clone, Debug)]
pub struct BetScoreIntake {
    ring: Vec<BetScoreRecord>,
    capacity: usize,
    head: usize,
    appended: u64,
}

impl BetScoreIntake {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            ring: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            appended: 0,
        }
    }

    /// Append, overwriting the oldest record once full (no allocation after
    /// the ring fills).
    pub fn push(&mut self, record: BetScoreRecord) {
        if self.ring.len() < self.capacity {
            self.ring.push(record);
        } else {
            self.ring[self.head] = record;
            self.head = (self.head + 1) % self.capacity;
        }
        self.appended += 1;
    }

    /// Records oldest → newest.
    pub fn iter(&self) -> impl Iterator<Item = &BetScoreRecord> {
        let (newer, older) = self.ring.split_at(self.head);
        older.iter().chain(newer.iter())
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Total appends (including overwritten records).
    pub fn appended(&self) -> u64 {
        self.appended
    }
}

/// The served-path state after calibration.
struct Fitted {
    raw: Vec<f32>,
    norm: Vec<f32>,
    values: Vec<f32>,
    scales: Vec<f32>,
    dim: usize,
    mask: u32,
    half_width: f32,
    scratch: BetScratch,
}

/// The calculation-brain estimate head.
pub struct BetSearchHead {
    cfg: BetSearchConfig,
    fitted: Option<Fitted>,
    floor: FloorAdapter,
    record: Option<BetScoreRecord>,
}

impl BetSearchHead {
    pub fn new(mut cfg: BetSearchConfig) -> Self {
        cfg.alpha = sanitize_alpha(cfg.alpha);
        cfg.roster.truncate(search::MAX_ROSTER);
        let floor = FloorAdapter::new(cfg.alpha);
        Self {
            cfg,
            fitted: None,
            floor,
            record: None,
        }
    }

    pub fn config(&self) -> &BetSearchConfig {
        &self.cfg
    }

    /// The last calibration's decision (`None` before the first).
    pub fn record(&self) -> Option<&BetScoreRecord> {
        self.record.as_ref()
    }

    /// Calibrate on `rows` (SOURCE order, oldest first). Rows with a
    /// non-finite value or feature are dropped; features are padded with
    /// `0.0` / truncated to the first row's width. Cold path.
    pub fn calibrate(&mut self, rows: &[Comparable]) {
        let rows: Vec<&Comparable> = rows
            .iter()
            .filter(|r| {
                r.value.is_finite()
                    && !r.features.is_empty()
                    && r.features.iter().all(|f| f.is_finite())
            })
            .collect();
        let dim = rows.first().map_or(0, |r| r.features.len());

        // The floor sees every comparable, in source order.
        self.floor = FloorAdapter::new(self.cfg.alpha);
        for r in &rows {
            let _ = self.floor.predict_next();
            self.floor.observe(r.value);
        }

        let (mut sel, mut cal) = split_partition(rows.len(), self.cfg.calibration_fraction);
        sel.sort_unstable();
        cal.sort_unstable();
        if sel.len() < 2 || cal.is_empty() || dim == 0 {
            self.fitted = None;
            self.record = Some(BetScoreRecord {
                search: SearchOutcome {
                    mask: 0,
                    crps: f32::INFINITY,
                    evaluations: 0,
                    bets_materialized: 0,
                },
                crps_floor: f32::NAN,
                crps_bet: f32::NAN,
                skill: -1.0,
                gate: self.cfg.gate.gate(-1.0),
                posture: BetPosture::Floor,
                undersized_calibration: true,
            });
            return;
        }

        // Selection split: raw + scale-normalized features (scales fitted on
        // the selection rows only).
        let sel_rows: Vec<Comparable> = sel.iter().map(|&i| rows[i].clone()).collect();
        let scales = fit_scales(&sel_rows);
        let mut raw = Vec::with_capacity(sel.len() * dim);
        let mut values = Vec::with_capacity(sel.len());
        for r in &sel_rows {
            for j in 0..dim {
                raw.push(r.features.get(j).copied().unwrap_or(0.0));
            }
            values.push(r.value);
        }
        let mut norm = Vec::new();
        normalize_rows(&raw, &scales, dim, &mut norm);
        let view = CompsView {
            raw: &raw,
            norm: &norm,
            values: &values,
            scales: &scales,
            dim,
            n: values.len(),
        };

        let eval = BetEvaluator::new(&self.cfg.roster, view, self.cfg.max_depth);
        let outcome = match self.cfg.mode {
            #[cfg(feature = "bet_search_puct")]
            SearchMode::Puct { cfg, seed } => puct_search(&eval, cfg, seed),
            SearchMode::Exhaustive { evaluations } => exhaustive_search(&eval, evaluations),
            SearchMode::Fixed { mask } => {
                let mask = mask & ((1_u32 << self.cfg.roster.len()) - 1);
                let crps = eval.crps(mask);
                SearchOutcome {
                    mask: if crps.is_finite() { mask } else { 0 },
                    crps,
                    evaluations: eval.evaluations(),
                    bets_materialized: mask.count_ones(),
                }
            }
        };

        // T3: like-for-like skill on selection rows 1.. (source order).
        let mut scratch = BetScratch::new(dim, values.len());
        let (crps_floor, crps_bet) = if outcome.mask == 0 {
            (f32::NAN, f32::NAN)
        } else {
            let mut floor_err = 0.0_f64;
            let mut bet_err = 0.0_f64;
            for i in 1..values.len() {
                floor_err += f64::from((values[i - 1] - values[i]).abs());
                let q = &raw[i * dim..i * dim + dim];
                let p = predict_mask(
                    &self.cfg.roster,
                    outcome.mask,
                    &view,
                    Some(i),
                    q,
                    &mut scratch,
                );
                bet_err += f64::from((p - values[i]).abs());
            }
            let m = (values.len() - 1) as f64;
            ((floor_err / m) as f32, (bet_err / m) as f32)
        };
        let skill = SkillGate::skill(crps_floor, crps_bet);
        let gate = self.cfg.gate.gate(skill);

        // Calibration split: residuals of the selection-fitted composition.
        let mut residuals: Vec<f32> = Vec::with_capacity(cal.len());
        if outcome.mask != 0 {
            for &ci in &cal {
                let r = rows[ci];
                let p = predict_mask(
                    &self.cfg.roster,
                    outcome.mask,
                    &view,
                    None,
                    &r.features,
                    &mut scratch,
                );
                let e = (p - r.value).abs();
                residuals.push(if e.is_finite() { e } else { f32::INFINITY });
            }
        }
        residuals.sort_unstable_by(f32::total_cmp);
        let k = conformal_index(residuals.len(), self.cfg.alpha);
        let undersized = ((residuals.len() as f32 + 1.0) * (1.0 - self.cfg.alpha)).ceil()
            > residuals.len() as f32;
        let half_width = residuals
            .get(k.saturating_sub(1))
            .copied()
            .unwrap_or(f32::INFINITY);

        let posture =
            if outcome.mask != 0 && half_width.is_finite() && (gate > 0.5 || !self.cfg.demote) {
                BetPosture::Bet
            } else {
                BetPosture::Floor
            };
        self.record = Some(BetScoreRecord {
            search: outcome,
            crps_floor,
            crps_bet,
            skill,
            gate,
            posture,
            undersized_calibration: undersized,
        });
        self.fitted = match posture {
            BetPosture::Bet => Some(Fitted {
                raw,
                norm,
                values,
                scales,
                dim,
                mask: outcome.mask,
                half_width,
                scratch,
            }),
            BetPosture::Floor => None,
        };
    }

    /// Which arm the next [`Self::estimate_into`] serves.
    pub fn posture(&self) -> BetPosture {
        if self.fitted.is_some() {
            BetPosture::Bet
        } else {
            BetPosture::Floor
        }
    }

    /// Serve the interval at `query` (raw features). Allocation-free.
    pub fn estimate_into(&mut self, query: &[f32], out: &mut PredictiveInterval) {
        let alpha = self.cfg.alpha;
        match self.fitted.as_mut() {
            Some(f) => {
                let view = CompsView {
                    raw: &f.raw,
                    norm: &f.norm,
                    values: &f.values,
                    scales: &f.scales,
                    dim: f.dim,
                    n: f.values.len(),
                };
                let p = predict_mask(&self.cfg.roster, f.mask, &view, None, query, &mut f.scratch);
                *out = PredictiveInterval::new(p - f.half_width, p, p + f.half_width, alpha);
            }
            None => {
                *out = self
                    .floor
                    .predict_next()
                    .interval
                    .unwrap_or(PredictiveInterval::new(f32::NAN, f32::NAN, f32::NAN, alpha));
            }
        }
    }
}

/// Equal-weight mean of the masked members' predictions (members ascending
/// — the Issue-940 fold order). `NaN` if any member cannot predict.
fn predict_mask(
    roster: &[AuthoredBet],
    mask: u32,
    comps: &CompsView<'_>,
    skip: Option<usize>,
    query: &[f32],
    scratch: &mut BetScratch,
) -> f32 {
    let mut s = 0.0_f32;
    for (b, bet) in roster.iter().enumerate() {
        if mask & (1 << b) != 0 {
            s += bet.predict(comps, skip, query, scratch);
        }
    }
    s / mask.count_ones().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize, f: impl Fn(usize) -> (Vec<f32>, f32)) -> Vec<Comparable> {
        (0..n)
            .map(|i| {
                let (features, value) = f(i);
                Comparable { features, value }
            })
            .collect()
    }

    /// value = 4 · x0 with a small deterministic wobble.
    fn ratio_rows(n: usize) -> Vec<Comparable> {
        rows(n, |i| {
            let x0 = 1.0 + ((i * 37) % 23) as f32;
            let wobble = ((i * 13) % 7) as f32 * 0.01 - 0.03;
            (
                vec![x0, ((i * 5) % 3) as f32 + 1.0],
                4.0 * x0 * (1.0 + wobble),
            )
        })
    }

    #[test]
    fn skill_is_clamped_guarded_and_signed() {
        assert_eq!(SkillGate::skill(2.0, 1.0), 0.5);
        assert_eq!(SkillGate::skill(1.0, 5.0), -1.0);
        assert_eq!(SkillGate::skill(0.0, 0.0), 0.0);
        assert_eq!(SkillGate::skill(0.0, 1.0), -1.0);
        assert_eq!(SkillGate::skill(f32::NAN, 1.0), -1.0);
        let g = SkillGate::default();
        assert!(g.gate(0.1) > 0.5 && g.gate(-0.1) < 0.5);
        assert_eq!(g.gate(0.0), 0.5);
    }

    #[test]
    fn ratio_world_serves_the_bet_and_covers() {
        let comps = ratio_rows(40);
        let mut head = BetSearchHead::new(BetSearchConfig::default());
        head.calibrate(&comps);
        let rec = *head.record().unwrap();
        assert_eq!(rec.posture, BetPosture::Bet, "{rec:?}");
        assert!(rec.skill > 0.5, "{rec:?}");
        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, 0.0);
        head.estimate_into(&[10.0, 2.0], &mut iv);
        assert!(iv.contains(40.0), "{iv:?}");
        assert!(iv.half_width() < 4.0, "{iv:?}");
    }

    #[test]
    fn too_few_comps_demote_to_the_floor() {
        let comps = ratio_rows(3);
        let mut head = BetSearchHead::new(BetSearchConfig::default());
        head.calibrate(&comps);
        assert_eq!(head.posture(), BetPosture::Floor);
        let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, 0.0);
        head.estimate_into(&[10.0, 2.0], &mut iv);
        assert!(iv.lower <= iv.point && iv.point <= iv.upper, "{iv:?}");
    }

    #[test]
    fn calibration_rows_never_reach_the_search() {
        // Poison every calibration-split row: if the search or the scales
        // read them, the pick or the fit moves. The selection rows alone
        // decide; the poison only widens the conformal half-width.
        let clean = ratio_rows(30);
        let (_, cal) = split_partition(clean.len(), 0.5);
        let mut poisoned = clean.clone();
        for &i in &cal {
            poisoned[i].value *= 50.0;
            poisoned[i].features[1] *= 1000.0;
        }
        let mut a = BetSearchHead::new(BetSearchConfig::default());
        let mut b = BetSearchHead::new(BetSearchConfig::default());
        a.calibrate(&clean);
        b.calibrate(&poisoned);
        let (ra, rb) = (a.record().unwrap(), b.record().unwrap());
        assert_eq!(ra.search, rb.search);
        assert_eq!(ra.crps_bet.to_bits(), rb.crps_bet.to_bits());
    }

    #[test]
    #[cfg(feature = "bet_search_puct")]
    fn puct_and_exhaustive_agree_on_a_full_budget() {
        let comps = ratio_rows(24);
        let mut ex = BetSearchHead::new(BetSearchConfig::default());
        let mut pu = BetSearchHead::new(BetSearchConfig {
            mode: SearchMode::Puct {
                cfg: ChancePuctConfig {
                    budget: 256,
                    ..ChancePuctConfig::default()
                },
                seed: 3,
            },
            ..BetSearchConfig::default()
        });
        ex.calibrate(&comps);
        pu.calibrate(&comps);
        let (re, rp) = (ex.record().unwrap(), pu.record().unwrap());
        assert!(rp.search.crps >= re.search.crps, "{re:?} {rp:?}");
    }

    #[test]
    fn intake_ring_keeps_the_newest() {
        let mut head = BetSearchHead::new(BetSearchConfig::default());
        head.calibrate(&ratio_rows(20));
        let rec = *head.record().unwrap();
        let mut intake = BetScoreIntake::new(2);
        for i in 0..3 {
            let mut r = rec;
            r.search.evaluations = i;
            intake.push(r);
        }
        let seen: Vec<u32> = intake.iter().map(|r| r.search.evaluations).collect();
        assert_eq!(seen, vec![1, 2]);
        assert_eq!(intake.appended(), 3);
    }

    // ---- G4: allocation gate (crate::alloc_gate) ----------------------------

    #[test]
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    fn estimate_into_is_zero_alloc_on_both_postures() {
        use crate::alloc_gate::{ALLOC_COUNT, MeasuringWindow};
        use std::sync::atomic::Ordering;

        for n in [40, 3] {
            let mut head = BetSearchHead::new(BetSearchConfig::default());
            head.calibrate(&ratio_rows(n));
            let mut iv = PredictiveInterval::new(0.0, 0.0, 0.0, 0.0);
            head.estimate_into(&[10.0, 2.0], &mut iv);
            let window = MeasuringWindow::enter();
            ALLOC_COUNT.store(0, Ordering::Relaxed);
            head.estimate_into(&[10.0, 2.0], &mut iv);
            let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
            drop(window);
            assert_eq!(
                allocs,
                0,
                "estimate_into allocated (n={n}, {:?})",
                head.posture()
            );
        }
    }
}
