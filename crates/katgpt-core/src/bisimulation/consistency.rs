//! `consistency` — rule-application consistency metric (Issue 586).
//!
//! Source paper: BDH-CQ (arXiv:2608.09888 §6.4) reports an 18.5-point gap
//! between test-pair accuracy (77.9%) and strict-task accuracy (59.4%) on
//! ConceptARC: 52/160 tasks produce one or two correct outputs but are never
//! solved as a whole. Interpretation: a correctly induced rule should
//! transfer to every test input; partial success means the transformation is
//! **not applied consistently** — and isolated correct outputs are
//! indistinguishable from a narrower rule applied completely.
//!
//! This module ships the missing half of "demonstration-conditioned operator
//! binding": [`rule_consistency`] measures whether a bound/inferred operator
//! ([`super::OperatorSchema`], via [`super::infer_operators`]) is applied
//! consistently across repeated/parallel applications, WITHOUT any training.
//! Binding without this gate can promote a flaky operator to committed state
//! (engram / chain commitment) with no signal that it only works on half the
//! inputs.
//!
//! # What it measures
//!
//! Given N applications of one operator, grouped into tasks (an application
//! is one invocation; a task is one "solve this input" episode with ≥1
//! applications — the paper's ARC task ↔ test-pairs split):
//!
//! - **3-bin histogram** over tasks: strict (all correct) / partial (some) /
//!   none — the paper's strict-task accuracy decomposition.
//! - **Gap** = application accuracy − strict-task accuracy, with a
//!   small-sample sigmoid guard (`gap_confidence`) so 2-task samples don't
//!   produce overconfident "inconsistent" verdicts. Sigmoid, NOT softmax
//!   (global rule: a binomial-confidence analog is a per-quantity bound).
//! - **Structure-preservation breakdown** over failures: localized error
//!   (output shape correct — the paper's extrapolation-failure signature)
//!   vs construction failure (wrong shape — execution-failure signature).
//! - **Complexity-clustered regime**: failures forming a contiguous suffix
//!   of complexity levels beyond a clean prefix. This is the coverage-failure
//!   signature that triggers exemplar-seeking in the downstream curiosity
//!   policy (riir-ai Issue 672).
//!
//! # Regime semantics (the T3 PoC's three separable regimes)
//!
//! | Regime | Failure signature | Right response |
//! |---|---|---|
//! | [`ConsistencyRegime::Consistent`] | none (or ≤ `NOISE_TOL` scattered) | gate on accuracy alone |
//! | [`ConsistencyRegime::NoisyFlaky`] | uniform across levels (i.i.d.) | do NOT commit; re-estimate reliability (CLR `should_write_memory` territory) |
//! | [`ConsistencyRegime::ComplexityClustered`] | clean prefix, broken suffix | seek ONE exemplar at the boundary level (Issue 672) |
//! | [`ConsistencyRegime::Ambiguous`] | graded / mixed | gather more evidence |
//!
//! # Modelless + zero-alloc
//!
//! Pure functions over a borrowed slice; the report is a fixed-size `Copy`
//! struct (two `[u16; 33]` per-level histograms). No allocation, no weights,
//! no training. Cold-path by design: called per skill-evaluation session,
//! not per tick.
//!
//! # Input contract
//!
//! Applications MUST be sorted by `task` (non-decreasing) so each task's
//! outcomes are contiguous — enforced by a `debug_assert`. The caller
//! collects outcomes per task anyway, so the contract is free to satisfy.
//!
//! # Lookup-binding caveat (Research 479 design caution)
//!
//! The paper's parameter binding is demonstrated-value lookup, NOT
//! interpolation — unseen values score 0% even when in-range. A
//! `ComplexityClustered` verdict therefore means "cover the boundary level",
//! and one exemplar at level c repairs exactly level c in the modelless
//! analog (the trained model partially transfers to neighbors; the
//! modelless policy re-targets after re-measuring instead — see Issue 672).

use crate::sigmoid;

// ─── Constants ─────────────────────────────────────────────────────────────

/// Maximum complexity level tracked per-level. Levels above this clamp into
/// slot `MAX_LEVEL` (nesting depth / ordering length / waypoint count are all
/// ≪ 32 in practice).
pub const MAX_LEVEL: u8 = 32;

/// Number of per-level histogram slots (`0..=MAX_LEVEL`).
pub const LEVEL_SLOTS: usize = MAX_LEVEL as usize + 1;

/// Prefix cleanliness bar: a level counts as "working below the boundary" if
/// its failure rate is ≤ this.
pub const LOW_BAR: f32 = 0.25;

/// Suffix brokenness bar: a level counts as "broken above the boundary" if
/// its failure rate is ≥ this.
pub const HIGH_BAR: f32 = 0.75;

/// A cluster verdict requires ≥ this many total failures in the broken
/// suffix — one stray failure is not a coverage signature.
pub const MIN_SUFFIX_FAILURES: u32 = 2;

/// Overall failure rate at or below this is treated as consistent
/// (tolerated noise): e.g. 1 failure in 100 applications.
pub const NOISE_TOL: f32 = 0.02;

/// Failure-rate spread (max − min across sampled levels) at or below this is
/// "uniform" — the i.i.d. flakiness signature.
pub const UNIFORM_BAND: f32 = 0.5;

/// Pseudo-task count for the shrunk gap (Bayesian shrinkage toward 0 with the
/// strength of `GAP_SHRINK_PSEUDO` observations at gap = 0).
pub const GAP_SHRINK_PSEUDO: f32 = 4.0;

/// `gap_confidence = sigmoid(GAP_CONFIDENCE_SLOPE · (n_tasks − MIDPOINT))`:
/// 4 tasks → 0.5, 10 tasks → 0.95, 24 tasks → ~1.0.
pub const GAP_CONFIDENCE_SLOPE: f32 = 0.5;
/// Sample size at which `gap_confidence` crosses 0.5.
pub const GAP_CONFIDENCE_MIDPOINT: f32 = 4.0;

// ─── Input ─────────────────────────────────────────────────────────────────

/// One application of a bound/inferred operator, with its outcome.
///
/// `Copy`, 12 bytes-ish, no heap. Produced by the consumer when it applies
/// the operator and compares against the expected result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApplicationOutcome {
    /// Did this application produce the expected result?
    pub correct: bool,
    /// Did the output preserve the expected structure (shape / length /
    /// grid dims)? For incorrect applications: `true` = localized error
    /// (extrapolation-failure signature), `false` = construction failure
    /// (execution-failure signature). For correct applications this is
    /// expected to be `true` (a correct output matched the expected shape);
    /// the field is only tallied for failures.
    pub structure_preserved: bool,
    /// Complexity level of this application's input (producer-defined dense
    /// small int: nesting depth, ordering length, waypoint count, …).
    /// Clamped to [`MAX_LEVEL`] internally.
    pub level: u8,
    /// Task (episode) this application belongs to. Applications MUST be
    /// sorted by `task` so each task's outcomes are contiguous.
    pub task: u32,
}

impl ApplicationOutcome {
    /// Convenience constructor.
    #[inline]
    pub const fn new(correct: bool, structure_preserved: bool, level: u8, task: u32) -> Self {
        Self {
            correct,
            structure_preserved,
            level,
            task,
        }
    }
}

// ─── Report ────────────────────────────────────────────────────────────────

/// How the operator's failures are distributed — the decision signal.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsistencyRegime {
    /// Applications agree; failures (if any) are ≤ [`NOISE_TOL`] scattered
    /// noise. Covers both all-correct and consistently-wrong — check
    /// `application_accuracy` for which.
    Consistent = 0,
    /// Failures spread uniformly across complexity levels — i.i.d.
    /// flakiness. The binding is unreliable everywhere; exemplar-seeking
    /// will NOT fix this (a demonstration doesn't repair noise).
    NoisyFlaky = 1,
    /// Failures form a contiguous suffix of levels beyond a clean prefix —
    /// the coverage-failure signature. `level` is the lowest failing level:
    /// the exemplar-seeking target (riir-ai Issue 672 trigger).
    ComplexityClustered { level: u8 } = 2,
    /// Failures exist but fit neither signature (graded degradation, mixed
    /// partial cluster). Gather more evidence before acting.
    Ambiguous = 3,
}

/// Zero-alloc consistency report over one operator's applications.
///
/// Fixed-size `Copy` struct (~140 bytes + two 66-byte histograms). Built by
/// [`rule_consistency`]; consumed by [`promotion_verdict`] or directly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConsistencyReport {
    /// Total applications tallied.
    pub n_applications: u32,
    /// Distinct tasks (contiguous `task` runs in the input).
    pub n_tasks: u32,
    /// Fraction of applications correct (paper: test-pair accuracy).
    pub application_accuracy: f32,
    /// Fraction of tasks where ALL applications were correct (paper:
    /// strict-task accuracy).
    pub strict_task_accuracy: f32,
    /// `application_accuracy − strict_task_accuracy`. Paper headline:
    /// 0.779 − 0.594 = 0.185.
    pub gap: f32,
    /// Small-sample-shrunk gap: `gap · n/(n + GAP_SHRINK_PSEUDO)`.
    pub gap_shrunk: f32,
    /// `sigmoid(slope·(n_tasks − midpoint))` ∈ (0,1) — how much to trust the
    /// gap estimate at this sample size.
    pub gap_confidence: f32,
    /// 3-bin histogram: tasks with all applications correct.
    pub strict_tasks: u32,
    /// 3-bin histogram: tasks with some-but-not-all correct.
    pub partial_tasks: u32,
    /// 3-bin histogram: tasks with zero correct.
    pub none_tasks: u32,
    /// Of the incorrect applications: output structure preserved (localized
    /// error — extrapolation signature).
    pub localized_errors: u32,
    /// Of the incorrect applications: structure destroyed (construction
    /// failure — execution signature).
    pub construction_failures: u32,
    /// `localized_errors / (localized + construction)`: 1.0 = every failure
    /// is structure-preserving (pure extrapolation shortfall). 0.0 when
    /// there are no failures.
    pub extrapolation_share: f32,
    /// Per-level application counts (index = `min(level, MAX_LEVEL)`).
    pub level_counts: [u16; LEVEL_SLOTS],
    /// Per-level failure counts (index = `min(level, MAX_LEVEL)`).
    pub level_failures: [u16; LEVEL_SLOTS],
    /// Highest level with ≥1 application (`0` when input is empty).
    pub max_sampled_level: u8,
    /// Regime classification — the downstream decision signal.
    pub regime: ConsistencyRegime,
}

impl ConsistencyReport {
    /// The exemplar-seeking target, if the regime is complexity-clustered.
    #[inline]
    pub fn cluster_level(&self) -> Option<u8> {
        match self.regime {
            ConsistencyRegime::ComplexityClustered { level } => Some(level),
            _ => None,
        }
    }
}

// ─── Metric ────────────────────────────────────────────────────────────────

/// Measure rule-application consistency over `applications`.
///
/// **Contract:** `applications` must be sorted by `task` (non-decreasing) so
/// each task's outcomes are contiguous (debug-asserted). Empty input yields
/// an all-zero report with regime [`ConsistencyRegime::Ambiguous`] — no
/// evidence is not consistency.
///
/// One pass over the slice + one split search over ≤ 33 level slots. Zero
/// allocation; the report is returned by value.
pub fn rule_consistency(applications: &[ApplicationOutcome]) -> ConsistencyReport {
    let mut n_tasks: u32 = 0;
    let mut strict_tasks: u32 = 0;
    let mut partial_tasks: u32 = 0;
    let mut none_tasks: u32 = 0;
    let mut cur_task: Option<u32> = None;
    let mut task_correct: u32 = 0;
    let mut task_total: u32 = 0;

    let mut n_applications: u32 = 0;
    let mut n_correct: u32 = 0;
    let mut localized_errors: u32 = 0;
    let mut construction_failures: u32 = 0;
    let mut level_counts = [0u16; LEVEL_SLOTS];
    let mut level_failures = [0u16; LEVEL_SLOTS];
    let mut max_sampled_level: u8 = 0;

    let mut close_task = |correct: u32, total: u32| {
        n_tasks += 1;
        if correct == total {
            strict_tasks += 1;
        } else if correct == 0 {
            none_tasks += 1;
        } else {
            partial_tasks += 1;
        }
    };

    for app in applications {
        debug_assert!(
            cur_task.is_none_or(|t| app.task >= t),
            "applications must be sorted by task"
        );
        if cur_task != Some(app.task) {
            if cur_task.is_some() {
                close_task(task_correct, task_total);
            }
            cur_task = Some(app.task);
            task_correct = 0;
            task_total = 0;
        }
        task_total += 1;
        n_applications += 1;
        let slot = app.level.min(MAX_LEVEL) as usize;
        level_counts[slot] = level_counts[slot].saturating_add(1);
        if app.level > max_sampled_level {
            max_sampled_level = app.level.min(MAX_LEVEL);
        }
        if app.correct {
            task_correct += 1;
            n_correct += 1;
        } else {
            level_failures[slot] = level_failures[slot].saturating_add(1);
            if app.structure_preserved {
                localized_errors += 1;
            } else {
                construction_failures += 1;
            }
        }
    }
    if cur_task.is_some() {
        close_task(task_correct, task_total);
    }

    let application_accuracy = if n_applications > 0 {
        n_correct as f32 / n_applications as f32
    } else {
        0.0
    };
    let strict_task_accuracy = if n_tasks > 0 {
        strict_tasks as f32 / n_tasks as f32
    } else {
        0.0
    };
    let gap = application_accuracy - strict_task_accuracy;
    let n = n_tasks as f32;
    let gap_shrunk = gap * n / (n + GAP_SHRINK_PSEUDO);
    let gap_confidence = sigmoid(GAP_CONFIDENCE_SLOPE * (n - GAP_CONFIDENCE_MIDPOINT));
    let n_failures = n_applications - n_correct;
    let extrapolation_share = if n_failures > 0 {
        localized_errors as f32 / n_failures as f32
    } else {
        0.0
    };

    let regime = classify_regime(n_failures, n_applications, &level_counts, &level_failures);

    ConsistencyReport {
        n_applications,
        n_tasks,
        application_accuracy,
        strict_task_accuracy,
        gap,
        gap_shrunk,
        gap_confidence,
        strict_tasks,
        partial_tasks,
        none_tasks,
        localized_errors,
        construction_failures,
        extrapolation_share,
        level_counts,
        level_failures,
        max_sampled_level,
        regime,
    }
}

/// Regime classification: consistent → clustered → uniform-flaky → ambiguous.
///
/// Order matters — the cluster signature is strictly more specific than the
/// uniform one, so it is checked first. A valid cluster split requires:
/// - a non-empty prefix of sampled levels all with failure rate ≤ [`LOW_BAR`],
/// - a suffix of sampled levels all with failure rate ≥ [`HIGH_BAR`],
/// - ≥ [`MIN_SUFFIX_FAILURES`] total failures in the suffix,
/// - the split level `s` is the lowest sampled level of the suffix (the
///   earliest boundary — the paper's "failing complexity").
fn classify_regime(
    n_failures: u32,
    n_applications: u32,
    level_counts: &[u16; LEVEL_SLOTS],
    level_failures: &[u16; LEVEL_SLOTS],
) -> ConsistencyRegime {
    if n_applications == 0 {
        return ConsistencyRegime::Ambiguous;
    }
    if n_failures == 0 {
        return ConsistencyRegime::Consistent;
    }
    if n_applications > 0 && (n_failures as f32 / n_applications as f32) <= NOISE_TOL {
        return ConsistencyRegime::Consistent;
    }

    // Sampled slots in ascending order (stack-only, no allocation).
    let sampled = sampled_slots(level_counts);
    let sl = sampled.as_slice();

    // Split search (see fn docs). Iterate candidate boundaries over sampled
    // slots, skipping the first (prefix must be non-empty).
    let mut split: Option<u8> = None;
    'outer: for s in 1..sl.len() {
        let boundary = sl[s];
        let mut suffix_failures: u32 = 0;
        for &lvl in &sl[..s] {
            let rate = level_failures[lvl as usize] as f32 / level_counts[lvl as usize] as f32;
            if rate > LOW_BAR {
                continue 'outer;
            }
        }
        for &lvl in &sl[s..] {
            let rate = level_failures[lvl as usize] as f32 / level_counts[lvl as usize] as f32;
            if rate < HIGH_BAR {
                continue 'outer;
            }
            suffix_failures += level_failures[lvl as usize] as u32;
        }
        if suffix_failures >= MIN_SUFFIX_FAILURES {
            split = Some(boundary);
            break;
        }
    }
    if let Some(level) = split {
        return ConsistencyRegime::ComplexityClustered { level };
    }

    // Uniform check: max − min failure rate across sampled levels.
    let mut min_rate = f32::INFINITY;
    let mut max_rate = f32::NEG_INFINITY;
    for &lvl in sl {
        let rate = level_failures[lvl as usize] as f32 / level_counts[lvl as usize] as f32;
        min_rate = min_rate.min(rate);
        max_rate = max_rate.max(rate);
    }
    if max_rate - min_rate <= UNIFORM_BAND {
        return ConsistencyRegime::NoisyFlaky;
    }
    ConsistencyRegime::Ambiguous
}

/// Stack-only collector for sampled level slots (avoids allocating a Vec for
/// the split search). 33 slots max — a fixed array + len.
#[inline]
fn sampled_slots(level_counts: &[u16; LEVEL_SLOTS]) -> SampledSlots {
    let mut out = SampledSlots {
        slots: [0u8; LEVEL_SLOTS],
        len: 0,
    };
    for (slot, &count) in level_counts.iter().enumerate() {
        if count > 0 {
            out.slots[out.len] = slot as u8;
            out.len += 1;
        }
    }
    out
}

/// Fixed-capacity list of sampled level slots.
struct SampledSlots {
    slots: [u8; LEVEL_SLOTS],
    len: usize,
}

impl SampledSlots {
    #[inline]
    fn as_slice(&self) -> &[u8] {
        &self.slots[..self.len]
    }
}

// ─── Promotion gate (T4 wiring on `infer_operators` output) ────────────────

/// Thresholds for promoting an inferred/bound operator to committed state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConsistencyGateConfig {
    /// Minimum application accuracy (fraction correct) outside the clustered
    /// regime. Below this the binding is wrong — discard.
    pub min_application_accuracy: f32,
    /// Minimum strict-task accuracy (all-applications-correct fraction).
    pub min_strict_task_accuracy: f32,
    /// Minimum `gap_confidence` — how much evidence before any verdict.
    pub min_gap_confidence: f32,
}

impl Default for ConsistencyGateConfig {
    #[inline]
    fn default() -> Self {
        Self {
            min_application_accuracy: 0.8,
            min_strict_task_accuracy: 0.6,
            min_gap_confidence: 0.75,
        }
    }
}

/// What to do with a bound operator, given its consistency report.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromotionVerdict {
    /// Consistent + accurate + enough evidence → safe to commit (engram /
    /// chain commitment path).
    Promote = 0,
    /// Complexity-clustered failures at `level` → seek ONE exemplar at
    /// `level`, then re-measure (riir-ai Issue 672 policy hook). Checked
    /// BEFORE the accuracy floor: the paper's coverage failures present as
    /// low overall accuracy at the failing family, and the single-exemplar
    /// repair is cheap.
    SeekExemplar { level: u8 } = 1,
    /// Uniform flakiness / ambiguity / low confidence → do NOT commit;
    /// re-estimate reliability or gather more evidence.
    Hold = 2,
    /// Accuracy floor failed outside the clustered regime → the binding is
    /// wrong; discard rather than commit.
    Reject = 3,
}

/// Combine a [`ConsistencyReport`] with [`ConsistencyGateConfig`] into a
/// promotion decision for an operator inferred by [`super::infer_operators`].
pub fn promotion_verdict(
    report: &ConsistencyReport,
    config: &ConsistencyGateConfig,
) -> PromotionVerdict {
    // The clustered regime is checked first: it is the one regime where low
    // accuracy has a cheap, targeted repair (one exemplar at the boundary).
    if let Some(level) = report.cluster_level() {
        return PromotionVerdict::SeekExemplar { level };
    }
    if report.gap_confidence < config.min_gap_confidence {
        return PromotionVerdict::Hold;
    }
    if report.application_accuracy < config.min_application_accuracy {
        return PromotionVerdict::Reject;
    }
    match report.regime {
        ConsistencyRegime::Consistent => {
            if report.strict_task_accuracy >= config.min_strict_task_accuracy {
                PromotionVerdict::Promote
            } else {
                PromotionVerdict::Hold
            }
        }
        _ => PromotionVerdict::Hold,
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Paper anchor (§6.4): 19/30 test pairs correct but only 2/10 tasks
    /// fully solved → partial-heavy, high gap.
    fn paper_anchor_partial_apps() -> Vec<ApplicationOutcome> {
        let mut apps = Vec::with_capacity(30);
        let mut task = 0u32;
        // 2 fully-correct tasks (3 apps each, levels spread 0..=2).
        for _ in 0..2 {
            for lvl in 0..3u8 {
                apps.push(ApplicationOutcome::new(true, true, lvl, task));
            }
            task += 1;
        }
        // 5 partial tasks with 2/3 correct.
        for _ in 0..5 {
            apps.push(ApplicationOutcome::new(true, true, 0, task));
            apps.push(ApplicationOutcome::new(true, true, 1, task));
            apps.push(ApplicationOutcome::new(false, true, 2, task));
            task += 1;
        }
        // 3 partial tasks with 1/3 correct.
        for _ in 0..3 {
            apps.push(ApplicationOutcome::new(true, true, 3, task));
            apps.push(ApplicationOutcome::new(false, true, 2, task));
            apps.push(ApplicationOutcome::new(false, false, 4, task));
            task += 1;
        }
        apps.shrink_to_fit();
        apps
    }

    #[test]
    fn paper_anchor_partial_gap_high() {
        let apps = paper_anchor_partial_apps();
        let r = rule_consistency(&apps);
        assert_eq!(r.n_applications, 30);
        assert_eq!(r.n_tasks, 10);
        // 6 + 10 + 3 = 19 correct of 30.
        assert!((r.application_accuracy - 19.0 / 30.0).abs() < 1e-6);
        assert_eq!(r.strict_tasks, 2);
        assert_eq!(r.partial_tasks, 8);
        assert_eq!(r.none_tasks, 0);
        assert!((r.strict_task_accuracy - 0.2).abs() < 1e-6);
        // Gap ≈ 0.433 — "high" by any reading of the paper's 0.185 headline.
        assert!(r.gap > 0.4, "gap {}", r.gap);
        // Failures: 5 localized (true) + 3 construction... wait: the 5-task
        // loop contributes 5 localized; the 3-task loop contributes 3
        // localized (lvl 2) + 3 construction (lvl 4). localized = 8,
        // construction = 3.
        assert_eq!(r.localized_errors, 8);
        assert_eq!(r.construction_failures, 3);
        assert!((r.extrapolation_share - 8.0 / 11.0).abs() < 1e-6);
        // 10 tasks → confident gap.
        assert!(r.gap_confidence > 0.95);
    }

    #[test]
    fn empty_input_is_ambiguous_not_consistent() {
        let r = rule_consistency(&[]);
        assert_eq!(r.n_applications, 0);
        assert_eq!(r.n_tasks, 0);
        assert_eq!(r.regime, ConsistencyRegime::Ambiguous);
        // The gate holds on empty evidence — no data is "gather more", not
        // "discard" (the confidence guard fires before the accuracy floor).
        let v = promotion_verdict(&r, &ConsistencyGateConfig::default());
        assert_eq!(v, PromotionVerdict::Hold);
    }

    #[test]
    fn all_correct_is_consistent_and_promotable() {
        // 8 tasks × 3 apps, levels 0..=3, all correct.
        let apps: Vec<ApplicationOutcome> = (0..8u32)
            .flat_map(|t| {
                (0..3u8).map(move |i| ApplicationOutcome::new(true, true, (t + i as u32) as u8 % 4, t))
            })
            .collect();
        let r = rule_consistency(&apps);
        assert_eq!(r.regime, ConsistencyRegime::Consistent);
        assert!((r.application_accuracy - 1.0).abs() < 1e-6);
        assert!((r.gap).abs() < 1e-6);
        let v = promotion_verdict(&r, &ConsistencyGateConfig::default());
        assert_eq!(v, PromotionVerdict::Promote);
    }

    #[test]
    fn consistently_wrong_is_consistent_but_rejected() {
        // Every application fails at every level — agreement without
        // correctness. Accuracy gate must reject.
        let apps: Vec<ApplicationOutcome> = (0..8u32)
            .flat_map(|t| (0..3u8).map(move |i| ApplicationOutcome::new(false, false, i, t)))
            .collect();
        let r = rule_consistency(&apps);
        // Uniform total failure: no clean prefix → not clustered; uniform →
        // flaky-classified distribution-wise, but accuracy drives the gate.
        assert_eq!(r.n_tasks, 8);
        assert_eq!(r.none_tasks, 8);
        let v = promotion_verdict(&r, &ConsistencyGateConfig::default());
        assert_eq!(v, PromotionVerdict::Reject);
    }

    // ── T3: the three regimes must separate ─────────────────────────────

    /// (a) Consistent application: 95%+ correct everywhere — exactly ONE
    /// tolerated failure in 72 applications (1.4% ≤ NOISE_TOL).
    #[test]
    fn regime_a_consistent_application() {
        let mut apps = Vec::with_capacity(6);
        let mut task = 0u32;
        // 6 levels × 4 tasks × 3 apps; a single failure on the first
        // level-3 task's last application.
        for lvl in 0..6u8 {
            for t in 0..4 {
                for k in 0..3 {
                    let correct = !(lvl == 3 && t == 0 && k == 2);
                    apps.push(ApplicationOutcome::new(correct, correct, lvl, task));
                }
                task += 1;
            }
        }
        let r = rule_consistency(&apps);
        // 1 failure in 72 → 1.4% ≤ NOISE_TOL.
        assert_eq!(r.regime, ConsistencyRegime::Consistent);
        assert!(r.application_accuracy > 0.98);
    }

    /// (b) Random flakiness: ~14% i.i.d.-ish failure at EVERY level
    /// (decorrelated: fail bit cycles mod 7, level cycles mod 6 — coprime,
    /// so each level sees the same fail rate).
    #[test]
    fn regime_b_random_flakiness() {
        let mut apps = Vec::with_capacity(6);
        let mut task = 0u32;
        let mut idx = 0usize;
        for _lvl in 0..6u8 {
            for _ in 0..4 {
                for _ in 0..4 {
                    let correct = idx % 7 != 3;
                    let level = ((idx * 5) % 6) as u8;
                    apps.push(ApplicationOutcome::new(correct, correct, level, task));
                    idx += 1;
                }
                task += 1;
            }
        }
        let r = rule_consistency(&apps);
        // Uniform ~14% failure at all levels → no cluster, spread ≤ band.
        assert_eq!(r.regime, ConsistencyRegime::NoisyFlaky);
        // Accuracy stays above the gate floor (0.86 ≥ 0.8) but application
        // is inconsistent → Hold, NOT Promote and NOT SeekExemplar.
        assert!(r.application_accuracy >= 0.8, "acc {}", r.application_accuracy);
        assert_eq!(r.cluster_level(), None);
        let v = promotion_verdict(&r, &ConsistencyGateConfig::default());
        assert_eq!(v, PromotionVerdict::Hold);
    }

    /// (c) Complexity-clustered: levels 0..=3 clean, 4..=5 broken — the
    /// paper's nesting-depth signature (19/24 at short context).
    #[test]
    fn regime_c_complexity_clustered_nesting_anchor() {
        let mut apps = Vec::with_capacity(5);
        let mut task = 0u32;
        // 19 correct applications across levels 1..=4, 5 failures at level 5
        // — exactly the paper's 19/24 nesting-depth-5 row.
        let per_level_correct: [(u8, usize); 4] = [(1, 5), (2, 5), (3, 5), (4, 4)];
        for (lvl, n) in per_level_correct {
            for _ in 0..n {
                apps.push(ApplicationOutcome::new(true, true, lvl, task));
                task += 1;
            }
        }
        for _ in 0..5 {
            apps.push(ApplicationOutcome::new(false, true, 5, task));
            task += 1;
        }
        let r = rule_consistency(&apps);
        assert_eq!(r.n_applications, 24);
        assert!((r.application_accuracy - 19.0 / 24.0).abs() < 1e-6);
        // Cluster at the boundary level 5.
        assert_eq!(r.regime, ConsistencyRegime::ComplexityClustered { level: 5 });
        // Extrapolation signature: all failures structure-preserving.
        assert!((r.extrapolation_share - 1.0).abs() < 1e-6);
        // Gate: seek one exemplar at level 5 (NOT Reject despite 79% acc).
        let v = promotion_verdict(&r, &ConsistencyGateConfig::default());
        assert_eq!(v, PromotionVerdict::SeekExemplar { level: 5 });
    }

    /// (c′) The paper's ordering-length signature: failures at a MULTI-level
    /// suffix (6,7,8) — cluster reports the LOWEST failing level.
    #[test]
    fn regime_c_multi_level_suffix_reports_earliest_boundary() {
        let mut apps = Vec::with_capacity(3);
        let mut task = 0u32;
        for lvl in 1..=5u8 {
            for _ in 0..3 {
                apps.push(ApplicationOutcome::new(true, true, lvl, task));
                task += 1;
            }
        }
        for lvl in 6..=8u8 {
            for _ in 0..3 {
                apps.push(ApplicationOutcome::new(false, true, lvl, task));
                task += 1;
            }
        }
        let r = rule_consistency(&apps);
        assert_eq!(
            r.regime,
            ConsistencyRegime::ComplexityClustered { level: 6 },
            "boundary must be the lowest failing level, got {:?}",
            r.regime
        );
    }

    /// Single stray failure at one level is NOT a cluster (MIN_SUFFIX_FAILURES).
    #[test]
    fn single_stray_failure_is_not_a_cluster() {
        let mut apps = Vec::with_capacity(5);
        let mut task = 0u32;
        for lvl in 0..5u8 {
            for _ in 0..4 {
                apps.push(ApplicationOutcome::new(true, true, lvl, task));
                task += 1;
            }
        }
        // One failure at level 2 — 1/21 ≈ 4.8% overall (> NOISE_TOL), single.
        apps.push(ApplicationOutcome::new(false, true, 2, task));
        let r = rule_consistency(&apps);
        assert_ne!(r.regime, ConsistencyRegime::ComplexityClustered { level: 2 });
    }

    /// Graded degradation (0.0, 0.6, 1.0 failure rates) is honestly Ambiguous.
    #[test]
    fn graded_degradation_is_ambiguous() {
        let mut apps = Vec::with_capacity(5);
        let mut task = 0u32;
        // Level 0: 5 apps all correct. Level 1: 5 apps, 3 fail (60%). Level 2: 5 apps all fail.
        for _ in 0..5 {
            apps.push(ApplicationOutcome::new(true, true, 0, task));
            task += 1;
        }
        for i in 0..5 {
            apps.push(ApplicationOutcome::new(i < 2, true, 1, task));
            task += 1;
        }
        for _ in 0..5 {
            apps.push(ApplicationOutcome::new(false, true, 2, task));
            task += 1;
        }
        let r = rule_consistency(&apps);
        assert_eq!(r.regime, ConsistencyRegime::Ambiguous);
    }

    // ── Confidence guards ────────────────────────────────────────────────

    #[test]
    fn small_sample_gap_is_low_confidence() {
        // 2 tasks × 3 apps, one partial → gap exists but n=2.
        let apps = vec![
            ApplicationOutcome::new(true, true, 0, 0),
            ApplicationOutcome::new(true, true, 0, 0),
            ApplicationOutcome::new(true, true, 0, 0),
            ApplicationOutcome::new(true, true, 1, 1),
            ApplicationOutcome::new(false, true, 1, 1),
            ApplicationOutcome::new(false, true, 1, 1),
        ];
        let r = rule_consistency(&apps);
        assert!(r.gap > 0.1);
        // n_tasks=2 < midpoint 4 → confidence < 0.5.
        assert!(r.gap_confidence < 0.5, "conf {}", r.gap_confidence);
        // Gate holds on low confidence.
        let v = promotion_verdict(&r, &ConsistencyGateConfig::default());
        assert_eq!(v, PromotionVerdict::Hold);
    }

    #[test]
    fn gap_shrunk_shrinks_with_sample_size() {
        // SAME raw gap (1/3) at n=2 tasks vs n=20 tasks: the shrunk gap must
        // be materially smaller at n=2 (2/6 of raw) than at n=20 (20/24).
        let small: Vec<ApplicationOutcome> = [
            (0u32, [true, true, true]),
            (1, [true, true, false]),
        ]
        .into_iter()
        .flat_map(|(t, outs)| {
            outs.into_iter()
                .map(move |c| ApplicationOutcome::new(c, true, 1, t))
                .collect::<Vec<_>>()
        })
        .collect();
        let mut big = small.clone();
        for t in 2..12u32 {
            for _ in 0..3 {
                big.push(ApplicationOutcome::new(true, true, 1, t));
            }
        }
        for t in 12..22u32 {
            big.push(ApplicationOutcome::new(true, true, 1, t));
            big.push(ApplicationOutcome::new(true, true, 1, t));
            big.push(ApplicationOutcome::new(false, true, 1, t));
        }
        let r_small = rule_consistency(&small);
        let r_big = rule_consistency(&big);
        // Identical raw gaps: 5/6 − 1/2 = 1/3 and 50/60 − 10/20 = 1/3.
        assert!((r_small.gap - 1.0 / 3.0).abs() < 1e-6);
        assert!((r_big.gap - 1.0 / 3.0).abs() < 1e-6);
        // ...but the shrunk gap at n=2 is pulled toward 0 far harder.
        assert!(r_small.gap_shrunk < r_small.gap * 0.34);
        assert!(r_big.gap_shrunk > r_big.gap * 0.8);
        // And confidence reflects the sample size.
        assert!(r_small.gap_confidence < 0.5);
        assert!(r_big.gap_confidence > 0.99);
    }

    // ── Wire-format hygiene ──────────────────────────────────────────────

    #[test]
    fn report_is_copy_and_level_arrays_are_fixed() {
        let apps = paper_anchor_partial_apps();
        let r = rule_consistency(&apps);
        let r2 = r; // Copy move — must compile.
        assert_eq!(r, r2);
        assert_eq!(r.level_counts.len(), LEVEL_SLOTS);
        assert_eq!(r.level_failures.len(), LEVEL_SLOTS);
    }

    #[test]
    fn levels_clamp_above_max() {
        let apps = vec![
            ApplicationOutcome::new(false, true, 200, 0),
            ApplicationOutcome::new(false, true, 200, 0),
            ApplicationOutcome::new(false, true, 200, 0),
            ApplicationOutcome::new(true, true, 1, 1),
            ApplicationOutcome::new(true, true, 1, 1),
            ApplicationOutcome::new(true, true, 1, 1),
            ApplicationOutcome::new(true, true, 1, 2),
            ApplicationOutcome::new(true, true, 1, 2),
            ApplicationOutcome::new(true, true, 1, 2),
        ];
        let r = rule_consistency(&apps);
        // Level 200 clamps into slot 32.
        assert_eq!(r.level_counts[32], 3);
        assert_eq!(r.level_failures[32], 3);
        assert_eq!(r.max_sampled_level, MAX_LEVEL);
    }
}
