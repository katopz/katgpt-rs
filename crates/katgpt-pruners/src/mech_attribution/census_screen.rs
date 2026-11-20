//! Neuron-census screen + typed control verdicts (Issue 1237 T1/T2,
//! Research 684 — the Fermion census discipline).
//!
//! Two modelless scoring halves for riir-infer Issue 1241's census lane
//! (the runner + checkpoints live there; these are the scorers it consumes):
//!
//! 1. [`screen_auroc`] — the cheap READ screen. Per unit, the closed-form
//!    rank AUROC of its activations against binary labels, then the SAME
//!    population search re-run under [`SCREEN_N_SCRAMBLES`] scrambled
//!    labelings. The search-corrected null is the best effect any unit
//!    reached under any scramble; a unit is admitted iff its true effect
//!    `|AUROC − ½|` clears that null by [`SCREEN_ADMISSION_MARGIN`]. Ranks
//!    are computed ONCE per unit and every labeling is an O(n) rank-sum —
//!    so the 20 scrambles cost 20 passes over a rank row, not 20 sorts.
//!    Zero allocation: all buffers live in [`ScreenScratch`].
//! 2. [`control_verdict`] — the interventional CONFIRMATION verdict. A
//!    candidate's pushed effect against a matched-random control band,
//!    typed `SwitchFound` / `NoSwitch{band_measured}` /
//!    `Undecided{under_floor}`. A confident negative REQUIRES the band to
//!    have been measured at floor — "no liar neuron" is a result only when
//!    the random band was actually read; otherwise it is an absence and the
//!    verdict says so. [`VerdictTally`] counts the three classes separately
//!    and exposes no pooled pass rate.
//!
//! Complementary to [`super::ActivationInfluenceProxy`] /
//! `cna_steering` (activation-difference attribution): those infer, this
//! screens against a search-corrected null and confirms against a measured
//! band. Kill bar shared with riir-infer 1241 — if its census cell closes
//! NEGATIVE, this module demotes with it.
//!
//! Determinism (Issue 940 law): ranks sort by `total_cmp` with first-index
//! tie-break, ties average; every fold is sequential; scrambles are
//! SplitMix64 Fisher–Yates from a pinned seed.

/// Scrambled-label searches per screen (Fermion's count — pinned).
pub const SCREEN_N_SCRAMBLES: usize = 20;
/// Admission margin over the search-corrected null, in `|AUROC − ½|` units
/// (Fermion's 0.20 — pinned).
pub const SCREEN_ADMISSION_MARGIN: f64 = 0.20;
/// Pre-registered matched-random control floor per candidate (riir-infer
/// 1241 population floor `min_control_neurons`; Fermion used 30).
pub const MIN_CONTROL_UNITS: usize = 20;

// ─────────────────────────────────────────────────────────────────────
// T1 — AUROC screen vs scrambled-label searches
// ─────────────────────────────────────────────────────────────────────

/// Screen parameters. [`Default`] pins the Fermion values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenParams {
    pub n_scrambles: usize,
    pub margin: f64,
    /// Minimum samples per class; below it the screen is UNDECIDED.
    pub min_per_class: usize,
    /// Scramble seed.
    pub seed: u64,
}

impl Default for ScreenParams {
    fn default() -> Self {
        Self {
            n_scrambles: SCREEN_N_SCRAMBLES,
            margin: SCREEN_ADMISSION_MARGIN,
            min_per_class: 8,
            seed: 0x1237_A0C0_5C2E_E700,
        }
    }
}

/// Preallocated screen buffers — sized once, reused across screens.
#[derive(Clone, Debug)]
pub struct ScreenScratch {
    order: Vec<usize>,
    ranks: Vec<f64>,
    /// `n_scrambles × n_samples` scrambled label rows.
    scrambled: Vec<bool>,
    /// Per-scramble best effect across the unit population.
    null_max: Vec<f64>,
    n_samples: usize,
    n_scrambles: usize,
}

impl ScreenScratch {
    pub fn new(n_samples: usize, n_scrambles: usize) -> Self {
        Self {
            order: vec![0; n_samples],
            ranks: vec![0.0; n_samples],
            scrambled: vec![false; n_samples * n_scrambles],
            null_max: vec![0.0; n_scrambles],
            n_samples,
            n_scrambles,
        }
    }
}

/// Per-unit screen row.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct UnitScreen {
    pub unit: u32,
    /// Rank AUROC under the true labels.
    pub auroc: f64,
    /// `|auroc − ½|`.
    pub effect: f64,
    /// Direction sign: `true` when positives rank HIGH (`auroc > ½`).
    pub positive: bool,
    /// Cleared `null_search_max + margin`.
    pub admitted: bool,
}

/// Why a screen cannot speak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenUnderFloor {
    /// A class has fewer than `min_per_class` samples.
    ClassCount { pos: usize, neg: usize, need: usize },
    /// Shapes disagree (activations, labels, scratch, output).
    Shape,
    /// A non-finite activation.
    NonFinite,
    /// `n_scrambles == 0` — no null was measured.
    NoNull,
    /// The search-corrected null is so wide that `threshold > ½`: no unit
    /// COULD be admitted, so an empty admit set is not a negative.
    NullTooWide,
}

/// Screen verdict.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScreenVerdict {
    /// The null was measured and the threshold is reachable; `admitted`
    /// may be zero (a measured negative).
    Screened {
        admitted: usize,
        null_search_max: f64,
        threshold: f64,
    },
    /// No verdict — `out` rows are not meaningful.
    Undecided(ScreenUnderFloor),
}

#[inline]
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Average (tie-aware, 1-based) ranks of `row` into `ranks`, using `order`
/// as the index scratch. Zero allocation.
fn rank_into(row: &[f32], order: &mut [usize], ranks: &mut [f64]) {
    for (i, o) in order.iter_mut().enumerate() {
        *o = i;
    }
    order.sort_unstable_by(|&a, &b| row[a].total_cmp(&row[b]).then(a.cmp(&b)));
    let n = row.len();
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && row[order[j]] == row[order[i]] {
            j += 1;
        }
        let avg = (i + 1 + j) as f64 / 2.0;
        for &k in &order[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
}

/// Closed-form AUROC from precomputed ranks:
/// `(Σ_pos rank − n_pos(n_pos+1)/2) / (n_pos·n_neg)`.
#[inline]
fn auroc_from_ranks(ranks: &[f64], labels: &[bool], n_pos: usize, n_neg: usize) -> f64 {
    let mut sum = 0.0f64;
    for (r, &l) in ranks.iter().zip(labels.iter()) {
        if l {
            sum += r;
        }
    }
    let np = n_pos as f64;
    (sum - np * (np + 1.0) / 2.0) / (np * n_neg as f64)
}

/// Screen `n_units` units. `acts` is row-major `[n_units][n_samples]`;
/// `labels` has `n_samples` entries; `out` has `n_units` rows. Zero
/// allocation (buffers in `scratch`).
pub fn screen_auroc(
    acts: &[f32],
    n_units: usize,
    labels: &[bool],
    params: &ScreenParams,
    scratch: &mut ScreenScratch,
    out: &mut [UnitScreen],
) -> ScreenVerdict {
    let n = labels.len();
    if n_units == 0
        || acts.len() != n_units * n
        || out.len() != n_units
        || scratch.n_samples != n
        || scratch.n_scrambles != params.n_scrambles
        || n_units > u32::MAX as usize
    {
        return ScreenVerdict::Undecided(ScreenUnderFloor::Shape);
    }
    if params.n_scrambles == 0 {
        return ScreenVerdict::Undecided(ScreenUnderFloor::NoNull);
    }
    let n_pos = labels.iter().filter(|&&l| l).count();
    let n_neg = n - n_pos;
    if n_pos < params.min_per_class.max(1) || n_neg < params.min_per_class.max(1) {
        return ScreenVerdict::Undecided(ScreenUnderFloor::ClassCount {
            pos: n_pos,
            neg: n_neg,
            need: params.min_per_class,
        });
    }
    if acts.iter().any(|v| !v.is_finite()) {
        return ScreenVerdict::Undecided(ScreenUnderFloor::NonFinite);
    }

    // Scrambled labelings: Fisher–Yates permutations (class counts kept).
    let mut state = params.seed;
    for s in 0..params.n_scrambles {
        let row = &mut scratch.scrambled[s * n..(s + 1) * n];
        row.copy_from_slice(labels);
        for i in (1..n).rev() {
            let j = (splitmix(&mut state) % (i as u64 + 1)) as usize;
            row.swap(i, j);
        }
    }
    scratch.null_max.fill(0.0);

    for (u, slot) in out.iter_mut().enumerate() {
        let row = &acts[u * n..(u + 1) * n];
        rank_into(row, &mut scratch.order, &mut scratch.ranks);
        let auc = auroc_from_ranks(&scratch.ranks, labels, n_pos, n_neg);
        *slot = UnitScreen {
            unit: u as u32,
            auroc: auc,
            effect: (auc - 0.5).abs(),
            positive: auc > 0.5,
            admitted: false,
        };
        for s in 0..params.n_scrambles {
            let lab = &scratch.scrambled[s * n..(s + 1) * n];
            let e = (auroc_from_ranks(&scratch.ranks, lab, n_pos, n_neg) - 0.5).abs();
            if e > scratch.null_max[s] {
                scratch.null_max[s] = e;
            }
        }
    }

    let null_search_max = scratch.null_max.iter().copied().fold(0.0f64, f64::max);
    let threshold = null_search_max + params.margin;
    if threshold > 0.5 {
        return ScreenVerdict::Undecided(ScreenUnderFloor::NullTooWide);
    }
    let mut admitted = 0usize;
    for slot in out.iter_mut() {
        if slot.effect >= threshold {
            slot.admitted = true;
            admitted += 1;
        }
    }
    ScreenVerdict::Screened {
        admitted,
        null_search_max,
        threshold,
    }
}

// ─────────────────────────────────────────────────────────────────────
// T2 — typed control-verdict battery
// ─────────────────────────────────────────────────────────────────────

/// Control-battery parameters. No `Default` for the effect bars on
/// purpose: Fermion's 40-vs-1.4 ratio is THEIR scale — pre-register per
/// fixture (riir-infer 1241 T1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlParams {
    /// Matched-random control units required (floor; default
    /// [`MIN_CONTROL_UNITS`]).
    pub min_controls: usize,
    /// Held-out reads required behind every effect number.
    pub min_reads: usize,
    /// Absolute effect floor.
    pub abs_floor: f64,
    /// Effect must reach `band_ratio × mean |control|`.
    pub band_ratio: f64,
}

impl ControlParams {
    pub fn new(min_reads: usize, abs_floor: f64, band_ratio: f64) -> Self {
        Self {
            min_controls: MIN_CONTROL_UNITS,
            min_reads,
            abs_floor,
            band_ratio,
        }
    }
}

/// The measured random band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandStats {
    pub n_controls: usize,
    pub n_reads: usize,
    pub mean_abs: f64,
    pub max_abs: f64,
}

/// Which floor kept the battery from speaking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnderFloor {
    Reads {
        got: usize,
        need: usize,
    },
    Controls {
        got: usize,
        need: usize,
    },
    NonFinite,
    /// Parameters themselves invalid (non-finite / negative bars).
    Params,
}

/// Typed control verdict — never pooled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ControlVerdict {
    /// Effect cleared the absolute floor, `band_ratio × mean band`, AND the
    /// band maximum.
    SwitchFound { effect: f64, band: BandStats },
    /// A CONFIDENT negative — permitted only because the band was measured
    /// at floor.
    NoSwitch {
        effect: f64,
        band_measured: BandStats,
    },
    /// No verdict; an absence, not a negative.
    Undecided { under_floor: UnderFloor },
}

/// Adjudicate one candidate. `effect` is the candidate's pushed effect,
/// `control_effects` the matched-random units pushed equally hard, both
/// measured over `n_reads` held-out reads. Signs are ignored (magnitudes).
/// Zero allocation.
pub fn control_verdict(
    effect: f64,
    n_reads: usize,
    control_effects: &[f64],
    p: &ControlParams,
) -> ControlVerdict {
    if !(p.abs_floor.is_finite() && p.band_ratio.is_finite())
        || p.abs_floor < 0.0
        || p.band_ratio < 0.0
    {
        return ControlVerdict::Undecided {
            under_floor: UnderFloor::Params,
        };
    }
    if n_reads < p.min_reads.max(1) {
        return ControlVerdict::Undecided {
            under_floor: UnderFloor::Reads {
                got: n_reads,
                need: p.min_reads,
            },
        };
    }
    if control_effects.len() < p.min_controls.max(1) {
        return ControlVerdict::Undecided {
            under_floor: UnderFloor::Controls {
                got: control_effects.len(),
                need: p.min_controls,
            },
        };
    }
    if !effect.is_finite() || control_effects.iter().any(|c| !c.is_finite()) {
        return ControlVerdict::Undecided {
            under_floor: UnderFloor::NonFinite,
        };
    }
    let mut sum = 0.0f64;
    let mut max_abs = 0.0f64;
    for c in control_effects {
        let a = c.abs();
        sum += a;
        if a > max_abs {
            max_abs = a;
        }
    }
    let band = BandStats {
        n_controls: control_effects.len(),
        n_reads,
        mean_abs: sum / control_effects.len() as f64,
        max_abs,
    };
    let e = effect.abs();
    if e >= p.abs_floor && e >= p.band_ratio * band.mean_abs && e > band.max_abs {
        ControlVerdict::SwitchFound { effect, band }
    } else {
        ControlVerdict::NoSwitch {
            effect,
            band_measured: band,
        }
    }
}

/// Per-class verdict counts. Deliberately exposes no pooled pass rate:
/// UNDECIDED is never folded into either side.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VerdictTally {
    pub switch_found: usize,
    pub no_switch: usize,
    pub undecided: usize,
}

impl VerdictTally {
    pub fn record(&mut self, v: &ControlVerdict) {
        match v {
            ControlVerdict::SwitchFound { .. } => self.switch_found += 1,
            ControlVerdict::NoSwitch { .. } => self.no_switch += 1,
            ControlVerdict::Undecided { .. } => self.undecided += 1,
        }
    }

    /// A population-level confident negative ("no switch for X"): at least
    /// one candidate decided, zero switches, zero undecided. Any undecided
    /// row keeps the claim open.
    pub fn confident_negative(&self) -> bool {
        self.switch_found == 0 && self.undecided == 0 && self.no_switch > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss(state: &mut u64) -> f32 {
        // Irwin–Hall(12) − 6 ≈ N(0, 1), deterministic.
        let mut s = 0.0f32;
        for _ in 0..12 {
            s += (splitmix(state) >> 40) as f32 / (1u64 << 24) as f32;
        }
        s - 6.0
    }

    /// `n_units` noise units over `n` samples; unit `planted` (if any)
    /// reads `label·gap + noise`.
    fn fixture(
        n_units: usize,
        n: usize,
        planted: Option<(usize, f32)>,
        seed: u64,
    ) -> (Vec<f32>, Vec<bool>) {
        let mut st = seed;
        let labels: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let mut acts = vec![0.0f32; n_units * n];
        for u in 0..n_units {
            for i in 0..n {
                let mut v = gauss(&mut st);
                if let Some((pu, gap)) = planted
                    && pu == u
                    && labels[i]
                {
                    v += gap;
                }
                acts[u * n + i] = v;
            }
        }
        (acts, labels)
    }

    #[test]
    fn param_pin() {
        let p = ScreenParams::default();
        assert_eq!(p.n_scrambles, 20);
        assert_eq!(p.margin, 0.20);
        assert_eq!(SCREEN_N_SCRAMBLES, 20);
        assert_eq!(SCREEN_ADMISSION_MARGIN, 0.20);
        assert_eq!(MIN_CONTROL_UNITS, 20);
        assert_eq!(ControlParams::new(10, 1.0, 5.0).min_controls, 20);
    }

    #[test]
    fn auroc_matches_pairwise_oracle() {
        let (acts, labels) = fixture(6, 41, Some((2, 0.9)), 7);
        // Inject ties.
        let mut acts = acts;
        acts[..10].fill(0.25);
        let n = labels.len();
        let mut scr = ScreenScratch::new(n, 20);
        let mut out = vec![UnitScreen::default(); 6];
        let p = ScreenParams {
            min_per_class: 1,
            ..Default::default()
        };
        let _ = screen_auroc(&acts, 6, &labels, &p, &mut scr, &mut out);
        for (u, row) in out.iter().enumerate() {
            let oracle =
                katgpt_core::loop_alignment_probe::auroc(&acts[u * n..(u + 1) * n], &labels);
            assert!(
                (row.auroc - oracle).abs() < 1e-12,
                "unit {u}: {} vs {oracle}",
                row.auroc
            );
        }
    }

    /// G1 both directions: a planted unit fires; genuinely uninformative
    /// labels (no planted unit — every activation independent of the
    /// label) refuse to admit anything.
    #[test]
    fn g1_planted_fires_shuffled_refuses() {
        let (n_units, n) = (64, 200);
        let mut scr = ScreenScratch::new(n, SCREEN_N_SCRAMBLES);
        let mut out = vec![UnitScreen::default(); n_units];
        let p = ScreenParams::default();

        let (acts, labels) = fixture(n_units, n, Some((17, 2.5)), 11);
        let v = screen_auroc(&acts, n_units, &labels, &p, &mut scr, &mut out);
        let ScreenVerdict::Screened {
            admitted,
            threshold,
            ..
        } = v
        else {
            panic!("expected Screened, got {v:?}");
        };
        assert!(
            out[17].admitted && out[17].positive,
            "{:?} thr {threshold}",
            out[17]
        );
        assert_eq!(admitted, 1, "only the planted unit");

        // Same planted unit, but labels genuinely shuffled relative to the
        // activations: the plant no longer aligns with the labels.
        let mut shuffled = labels.clone();
        let mut st = 99u64;
        for i in (1..n).rev() {
            let j = (splitmix(&mut st) % (i as u64 + 1)) as usize;
            shuffled.swap(i, j);
        }
        let v = screen_auroc(&acts, n_units, &shuffled, &p, &mut scr, &mut out);
        assert!(
            matches!(v, ScreenVerdict::Screened { admitted: 0, .. }),
            "shuffled labels must admit nothing: {v:?}"
        );
    }

    #[test]
    fn screen_refusals_typed_and_deterministic() {
        let (acts, labels) = fixture(4, 12, None, 3);
        let mut scr = ScreenScratch::new(12, 20);
        let mut out = vec![UnitScreen::default(); 4];
        // 6 per class < default floor 8.
        assert!(matches!(
            screen_auroc(
                &acts,
                4,
                &labels,
                &ScreenParams::default(),
                &mut scr,
                &mut out
            ),
            ScreenVerdict::Undecided(ScreenUnderFloor::ClassCount { .. })
        ));
        // Tiny sample: the null is so wide no unit could be admitted.
        let p = ScreenParams {
            min_per_class: 1,
            ..Default::default()
        };
        assert_eq!(
            screen_auroc(&acts, 4, &labels, &p, &mut scr, &mut out),
            ScreenVerdict::Undecided(ScreenUnderFloor::NullTooWide)
        );
        let mut bad = acts.clone();
        bad[3] = f32::NAN;
        assert_eq!(
            screen_auroc(&bad, 4, &labels, &p, &mut scr, &mut out),
            ScreenVerdict::Undecided(ScreenUnderFloor::NonFinite)
        );
        assert_eq!(
            screen_auroc(&acts, 3, &labels, &p, &mut scr, &mut out),
            ScreenVerdict::Undecided(ScreenUnderFloor::Shape)
        );
        // Determinism.
        let (acts, labels) = fixture(16, 120, Some((3, 1.5)), 5);
        let mut scr = ScreenScratch::new(120, 20);
        let mut a = vec![UnitScreen::default(); 16];
        let mut b = vec![UnitScreen::default(); 16];
        let va = screen_auroc(
            &acts,
            16,
            &labels,
            &ScreenParams::default(),
            &mut scr,
            &mut a,
        );
        let vb = screen_auroc(
            &acts,
            16,
            &labels,
            &ScreenParams::default(),
            &mut scr,
            &mut b,
        );
        assert_eq!(va, vb);
        assert_eq!(a, b);
    }

    /// G4: the scorer allocates nothing once scratch is sized — proven by
    /// capacity/pointer stability of every scratch buffer across a screen
    /// (the function takes no owned buffers and returns a Copy verdict).
    #[test]
    fn g4_scorer_alloc_free_scratch_stable() {
        let (acts, labels) = fixture(32, 160, Some((5, 2.0)), 21);
        let mut scr = ScreenScratch::new(160, 20);
        let mut out = vec![UnitScreen::default(); 32];
        let ptrs = (
            scr.order.as_ptr() as usize,
            scr.ranks.as_ptr() as usize,
            scr.scrambled.as_ptr() as usize,
            scr.null_max.as_ptr() as usize,
        );
        let caps = (
            scr.order.capacity(),
            scr.ranks.capacity(),
            scr.scrambled.capacity(),
            scr.null_max.capacity(),
        );
        for _ in 0..3 {
            let _ = screen_auroc(
                &acts,
                32,
                &labels,
                &ScreenParams::default(),
                &mut scr,
                &mut out,
            );
        }
        assert_eq!(
            ptrs,
            (
                scr.order.as_ptr() as usize,
                scr.ranks.as_ptr() as usize,
                scr.scrambled.as_ptr() as usize,
                scr.null_max.as_ptr() as usize,
            )
        );
        assert_eq!(
            caps,
            (
                scr.order.capacity(),
                scr.ranks.capacity(),
                scr.scrambled.capacity(),
                scr.null_max.capacity(),
            )
        );
    }

    fn band(n: usize, scale: f64) -> Vec<f64> {
        (0..n)
            .map(|i| scale * (((i * 37) % 11) as f64 / 10.0 - 0.5))
            .collect()
    }

    /// G1: three canary arms.
    #[test]
    fn g1_control_verdict_three_arms() {
        let p = ControlParams::new(50, 10.0, 20.0);
        let controls = band(20, 2.0); // |c| ≤ 1.0, mean ≈ 0.5
        // (1) planted strong effect → SwitchFound.
        assert!(matches!(
            control_verdict(40.0, 150, &controls, &p),
            ControlVerdict::SwitchFound { .. }
        ));
        // (2) under-floor read count → Undecided, even for a strong effect.
        assert_eq!(
            control_verdict(40.0, 49, &controls, &p),
            ControlVerdict::Undecided {
                under_floor: UnderFloor::Reads { got: 49, need: 50 }
            }
        );
        // Under-floor control count → Undecided.
        assert_eq!(
            control_verdict(40.0, 150, &controls[..19], &p),
            ControlVerdict::Undecided {
                under_floor: UnderFloor::Controls { got: 19, need: 20 }
            }
        );
        // (3) full read, band measured, weak effect → NoSwitch permitted,
        // and it carries the measured band.
        match control_verdict(1.2, 150, &controls, &p) {
            ControlVerdict::NoSwitch { band_measured, .. } => {
                assert_eq!(band_measured.n_controls, 20);
                assert_eq!(band_measured.n_reads, 150);
                assert!(band_measured.mean_abs > 0.0);
            }
            other => panic!("expected NoSwitch, got {other:?}"),
        }
        // Effect above floor + ratio but not above a control's max → NoSwitch.
        let mut spiky = band(20, 0.1);
        spiky[7] = 50.0;
        assert!(matches!(
            control_verdict(40.0, 150, &spiky, &p),
            ControlVerdict::NoSwitch { .. }
        ));
    }

    #[test]
    fn tally_never_pools() {
        let p = ControlParams::new(10, 1.0, 5.0);
        let c = band(20, 0.2);
        let mut t = VerdictTally::default();
        t.record(&control_verdict(0.01, 20, &c, &p));
        t.record(&control_verdict(0.02, 20, &c, &p));
        assert!(t.confident_negative());
        t.record(&control_verdict(0.02, 3, &c, &p));
        assert_eq!(
            t,
            VerdictTally {
                switch_found: 0,
                no_switch: 2,
                undecided: 1
            }
        );
        assert!(
            !t.confident_negative(),
            "an undecided row keeps the claim open"
        );
        assert!(!VerdictTally::default().confident_negative());
    }
}
