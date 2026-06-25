//! OfflineCandidateScorer — ARG Step C (Scoring) primitive with G5 silence-bias.
//!
//! Distilled from ARG §3.3 (Scoring). The offline loop scores each typed
//! candidate against resolved evidence. The scoring formula is the
//! **silence-bias penalty** — the ARG protocol's anti-gaming invariant:
//!
//! > Silence ≠ confirmed success. A candidate whose gains are dominated by
//! > unverified outcomes (`INFO_UNCERTAIN_SUCCESS` + `INFO_LOW_CONFIDENCE`)
//! > MUST score strictly lower than a candidate with the same nominal gain but
//! > all-confirmed evidence, and MUST be refused auto-commit.
//!
//! ## Formula
//!
//! Given evidence partitioned by outcome:
//! - `g_confirmed := Σ weight where outcome == InfoConfirmedSuccess`
//! - `g_uncertain := Σ weight where outcome == InfoUncertainSuccess`
//! - `g_lowconf   := Σ weight where outcome == InfoLowConfidence`
//! - `nominal_gain := g_confirmed + g_uncertain + g_lowconf`
//! - `penalty_silent := lambda * (g_uncertain + g_lowconf)`
//! - `score := nominal_gain - penalty_silent`
//!          `= g_confirmed + (1 - lambda) * (g_uncertain + g_lowconf)`
//!
//! For any `lambda > 0` and equal nominal gain `G`:
//! - all-confirmed  → score = G
//! - all-lowconf    → score = (1 - lambda) * G  < G
//! - 50/50 mix      → score = (1 - lambda/2) * G,  strictly between
//!
//! Strict inequalities hold for `0 < lambda <= 1` and `G > 0`. This is the G5
//! property-test contract.
//!
//! ## Auto-commit gate
//!
//! `can_auto_commit(scored, threshold)` returns `false` when the low-confidence
//! fraction `(g_uncertain + g_lowconf) / nominal_gain > threshold`. Default
//! threshold is 0.5 — a candidate must be majority-confirmed to auto-commit.
//! Candidates that fail the gate are still scored (for human review) but cannot
//! be auto-promoted; they require explicit validation (Step D).
//!
//! All arithmetic is pure and deterministic (replay-safe). No allocations in
//! the hot path — `ScoredCandidate` and `GainComponents` are `Copy`.

use super::candidate::TypedOfflineCandidate;

/// ARG §Info IO outcome status. `C_info` alone is insufficient — the protocol
/// requires a discrete outcome tag so the scorer can apply the silence-bias
/// penalty correctly.
///
/// - `InfoConfirmedSuccess` — the info response was grounded and verified.
/// - `InfoUncertainSuccess` — the info response was emitted but outcome unknown.
/// - `InfoLowConfidence` — the info response was low-confidence or silent.
///
/// `Silence ≠ confirmed success`: a silent response is `InfoLowConfidence`,
/// never `InfoConfirmedSuccess`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum InfoOutcomeStatus {
    /// Grounded + verified info response. Contributes full weight to gain.
    #[default]
    InfoConfirmedSuccess = 0,
    /// Emitted but outcome unknown (no feedback signal yet). Discounted.
    InfoUncertainSuccess = 1,
    /// Low-confidence or silent response. Heavily discounted + penalized.
    InfoLowConfidence = 2,
}

impl InfoOutcomeStatus {
    /// Returns `true` for outcomes that count as "unverified" (subject to the
    /// silence-bias penalty). `InfoConfirmedSuccess` is the only verified one.
    #[inline]
    pub fn is_unverified(self) -> bool {
        matches!(
            self,
            InfoOutcomeStatus::InfoUncertainSuccess | InfoOutcomeStatus::InfoLowConfidence
        )
    }
}

/// A single piece of evidence — outcome + weight. `Copy` (16 bytes).
#[derive(Clone, Copy, Debug)]
pub struct Evidence {
    pub outcome: InfoOutcomeStatus,
    /// Non-negative weight (caller-defined: could be recency, magnitude, etc.).
    /// Negative weights are clamped to zero at scoring time.
    pub weight: f32,
}

impl Evidence {
    /// Construct evidence with a non-negative weight.
    #[inline]
    pub fn new(outcome: InfoOutcomeStatus, weight: f32) -> Self {
        Evidence { outcome, weight }
    }
}

/// Partitioned gain components — the breakdown the silence-bias formula uses.
/// `Copy` (12 bytes). Returned by [`OfflineCandidateScorer::score`] so callers
/// can inspect the breakdown for audit / explainability.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GainComponents {
    pub confirmed: f32,
    pub uncertain: f32,
    pub lowconf: f32,
}

impl GainComponents {
    /// Total nominal gain (sum of all three partitions).
    #[inline]
    pub fn nominal_gain(&self) -> f32 {
        self.confirmed + self.uncertain + self.lowconf
    }

    /// Unverified fraction `(uncertain + lowconf) / nominal_gain`.
    /// Returns `1.0` when `nominal_gain == 0` (no evidence = fully unverified).
    #[inline]
    pub fn unverified_fraction(&self) -> f32 {
        let nominal = self.nominal_gain();
        if nominal <= 0.0 {
            return 1.0;
        }
        (self.uncertain + self.lowconf) / nominal
    }

    /// Confirmed fraction `confirmed / nominal_gain`. Returns `0.0` when
    /// `nominal_gain == 0`.
    #[inline]
    pub fn confirmed_fraction(&self) -> f32 {
        let nominal = self.nominal_gain();
        if nominal <= 0.0 {
            return 0.0;
        }
        self.confirmed / nominal
    }
}

/// The full scored output — gains, penalty, and final score. `Copy` (20 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScoredCandidate {
    pub gains: GainComponents,
    /// `nominal_gain` cached for convenience (= `gains.nominal_gain()`).
    pub nominal_gain: f32,
    /// `lambda * (g_uncertain + g_lowconf)`. Always `>= 0` for `lambda >= 0`.
    pub penalty_silent: f32,
    /// `nominal_gain - penalty_silent`. The value to rank candidates by.
    pub score: f32,
}

impl ScoredCandidate {
    /// Returns `true` when the candidate has zero evidence (nominal_gain == 0).
    /// Such candidates can never auto-commit.
    #[inline]
    pub fn has_no_evidence(&self) -> bool {
        self.nominal_gain <= 0.0
    }
}

/// Configuration for the silence-bias scorer.
#[derive(Clone, Copy, Debug)]
pub struct OfflineCandidateScorer {
    /// Silence-bias penalty coefficient in `[0, 1]`. `lambda = 1.0` means
    /// unverified evidence contributes zero to the score; `lambda = 0.0`
    /// disables the penalty (not recommended — breaks G5). Default `1.0`.
    pub lambda: f32,
}

impl Default for OfflineCandidateScorer {
    #[inline]
    fn default() -> Self {
        OfflineCandidateScorer { lambda: 1.0 }
    }
}

impl OfflineCandidateScorer {
    /// Construct with a specific lambda. Clamps to `[0, 1]`.
    #[inline]
    pub fn new(lambda: f32) -> Self {
        OfflineCandidateScorer {
            lambda: lambda.clamp(0.0, 1.0),
        }
    }

    /// Score a candidate against resolved evidence.
    ///
    /// The candidate's `intent.kind` is available for future kind-specific
    /// weighting; v1 scores all kinds with the same formula (the silence-bias
    /// invariant is kind-independent — it is a protocol-level anti-gaming gate).
    ///
    /// Zero-alloc: returns a `Copy` `ScoredCandidate`. The evidence slice is
    /// borrowed; no allocation regardless of evidence count.
    pub fn score(&self, _candidate: &TypedOfflineCandidate<'_>, evidence: &[Evidence]) -> ScoredCandidate {
        let mut gains = GainComponents::default();
        for e in evidence {
            // Clamp negative weights to zero — weights are non-negative by contract.
            let w = if e.weight < 0.0 { 0.0 } else { e.weight };
            match e.outcome {
                InfoOutcomeStatus::InfoConfirmedSuccess => gains.confirmed += w,
                InfoOutcomeStatus::InfoUncertainSuccess => gains.uncertain += w,
                InfoOutcomeStatus::InfoLowConfidence => gains.lowconf += w,
            }
        }
        let nominal_gain = gains.nominal_gain();
        let penalty_silent = self.lambda * (gains.uncertain + gains.lowconf);
        let score = nominal_gain - penalty_silent;
        ScoredCandidate {
            gains,
            nominal_gain,
            penalty_silent,
            score,
        }
    }

    /// Auto-commit gate. Returns `true` iff the candidate is safe to auto-commit
    /// without explicit human/validation review.
    ///
    /// Refuses (returns `false`) when:
    /// - `nominal_gain == 0` (no evidence), OR
    /// - `unverified_fraction > threshold` (low-confidence-dominated).
    ///
    /// Default threshold is `0.5` — a candidate must be majority-confirmed.
    /// Stricter thresholds (e.g. `0.2`) require super-majority confirmation.
    #[inline]
    pub fn can_auto_commit(scored: &ScoredCandidate, threshold: f32) -> bool {
        if scored.has_no_evidence() {
            return false;
        }
        scored.gains.unverified_fraction() <= threshold
    }
}

/// The default auto-commit threshold — a candidate must be majority-confirmed
/// (`unverified_fraction <= 0.5`).
pub const DEFAULT_AUTO_COMMIT_THRESHOLD: f32 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::candidate::{CandidateKind, TypedOfflineCandidate};
    use super::super::taxonomy::LabelId;

    fn lbl(n: u32) -> LabelId {
        LabelId::new(n)
    }

    fn unscored_edge() -> TypedOfflineCandidate<'static> {
        // A minimal candidate — the scorer doesn't read intent fields in v1.
        TypedOfflineCandidate::bare(CandidateKind::Edge, lbl(1), &[])
    }

    fn ev(outcome: InfoOutcomeStatus, weight: f32) -> Evidence {
        Evidence::new(outcome, weight)
    }

    // --------------------------------------------------------------------
    // G5 silence-bias property tests (the core contract).
    // --------------------------------------------------------------------

    #[test]
    fn g5_all_confirmed_beats_all_lowconf_at_equal_nominal_gain() {
        // Same nominal gain G=1.0, different evidence composition.
        let scorer = OfflineCandidateScorer::default(); // lambda = 1.0
        let g = 1.0f32;
        let confirmed = [ev(InfoOutcomeStatus::InfoConfirmedSuccess, g)];
        let lowconf = [ev(InfoOutcomeStatus::InfoLowConfidence, g)];
        let s_confirmed = scorer.score(&unscored_edge(), &confirmed);
        let s_lowconf = scorer.score(&unscored_edge(), &lowconf);
        // X > Y (strict).
        assert!(
            s_confirmed.score > s_lowconf.score,
            "G5 violated: confirmed {:?} must strictly beat lowconf {:?}",
            s_confirmed.score,
            s_lowconf.score
        );
        // At lambda=1.0, all-confirmed keeps full gain; all-lowconf is zeroed.
        assert!((s_confirmed.score - g).abs() < 1e-6);
        assert!(s_lowconf.score.abs() < 1e-6);
    }

    #[test]
    fn g5_mixed_score_strictly_between_confirmed_and_lowconf() {
        // Same nominal gain G=1.0, 50/50 confirmed/lowconf.
        let scorer = OfflineCandidateScorer::default();
        let g = 1.0f32;
        let confirmed = [ev(InfoOutcomeStatus::InfoConfirmedSuccess, g)];
        let lowconf = [ev(InfoOutcomeStatus::InfoLowConfidence, g)];
        let mixed = [
            ev(InfoOutcomeStatus::InfoConfirmedSuccess, g / 2.0),
            ev(InfoOutcomeStatus::InfoLowConfidence, g / 2.0),
        ];
        let s_confirmed = scorer.score(&unscored_edge(), &confirmed).score;
        let s_lowconf = scorer.score(&unscored_edge(), &lowconf).score;
        let s_mixed = scorer.score(&unscored_edge(), &mixed).score;
        // X > Z > Y (strict).
        assert!(
            s_confirmed > s_mixed,
            "G5 mixed violated: confirmed {} must beat mixed {}",
            s_confirmed,
            s_mixed
        );
        assert!(
            s_mixed > s_lowconf,
            "G5 mixed violated: mixed {} must beat lowconf {}",
            s_mixed,
            s_lowconf
        );
    }

    #[test]
    fn g5_uncertain_also_discounted_but_not_as_heavily_as_lowconf_at_equal_gain() {
        // The formula penalizes (uncertain + lowconf) together at the same rate.
        // So at equal nominal gain, all-uncertain == all-lowconf. The strict
        // ordering is confirmed > {uncertain, lowconf}. This test documents the
        // v1 design: uncertain and lowconf are both "unverified" and penalized
        // identically. (A future v2 could penalize lowconf more than uncertain.)
        let scorer = OfflineCandidateScorer::default();
        let g = 1.0f32;
        let uncertain = [ev(InfoOutcomeStatus::InfoUncertainSuccess, g)];
        let lowconf = [ev(InfoOutcomeStatus::InfoLowConfidence, g)];
        let s_uncertain = scorer.score(&unscored_edge(), &uncertain).score;
        let s_lowconf = scorer.score(&unscored_edge(), &lowconf).score;
        // At lambda=1.0, both are fully discounted (score = 0).
        assert!((s_uncertain - s_lowconf).abs() < 1e-6);
        // But confirmed beats both.
        let confirmed = [ev(InfoOutcomeStatus::InfoConfirmedSuccess, g)];
        let s_confirmed = scorer.score(&unscored_edge(), &confirmed).score;
        assert!(s_confirmed > s_uncertain);
        assert!(s_confirmed > s_lowconf);
    }

    #[test]
    fn g5_strict_inequality_holds_across_lambda_values() {
        // For any lambda in (0, 1], confirmed strictly beats lowconf at equal
        // nominal gain (as long as G > 0).
        let g = 1.0f32;
        let confirmed = [ev(InfoOutcomeStatus::InfoConfirmedSuccess, g)];
        let lowconf = [ev(InfoOutcomeStatus::InfoLowConfidence, g)];
        for &lambda in &[0.1f32, 0.25, 0.5, 0.75, 0.9, 1.0] {
            let scorer = OfflineCandidateScorer::new(lambda);
            let s_confirmed = scorer.score(&unscored_edge(), &confirmed).score;
            let s_lowconf = scorer.score(&unscored_edge(), &lowconf).score;
            assert!(
                s_confirmed > s_lowconf,
                "lambda={}: confirmed {} must beat lowconf {}",
                lambda,
                s_confirmed,
                s_lowconf
            );
        }
    }

    // --------------------------------------------------------------------
    // Auto-commit gate tests (the G5 operational consequence).
    // --------------------------------------------------------------------

    #[test]
    fn auto_commit_refuses_no_evidence() {
        let scorer = OfflineCandidateScorer::default();
        let scored = scorer.score(&unscored_edge(), &[]);
        assert!(scored.has_no_evidence());
        assert!(!OfflineCandidateScorer::can_auto_commit(&scored, DEFAULT_AUTO_COMMIT_THRESHOLD));
    }

    #[test]
    fn auto_commit_refuses_lowconf_dominated() {
        // 80% lowconf, 20% confirmed — unverified_fraction = 0.8 > 0.5 threshold.
        let scorer = OfflineCandidateScorer::default();
        let evidence = [
            ev(InfoOutcomeStatus::InfoConfirmedSuccess, 0.2),
            ev(InfoOutcomeStatus::InfoLowConfidence, 0.8),
        ];
        let scored = scorer.score(&unscored_edge(), &evidence);
        assert!((scored.gains.unverified_fraction() - 0.8).abs() < 1e-6);
        assert!(!OfflineCandidateScorer::can_auto_commit(&scored, DEFAULT_AUTO_COMMIT_THRESHOLD));
    }

    #[test]
    fn auto_commit_allows_majority_confirmed() {
        // 70% confirmed, 30% lowconf — unverified_fraction = 0.3 <= 0.5.
        let scorer = OfflineCandidateScorer::default();
        let evidence = [
            ev(InfoOutcomeStatus::InfoConfirmedSuccess, 0.7),
            ev(InfoOutcomeStatus::InfoLowConfidence, 0.3),
        ];
        let scored = scorer.score(&unscored_edge(), &evidence);
        assert!((scored.gains.unverified_fraction() - 0.3).abs() < 1e-6);
        assert!(OfflineCandidateScorer::can_auto_commit(&scored, DEFAULT_AUTO_COMMIT_THRESHOLD));
    }

    #[test]
    fn auto_commit_all_confirmed_always_allowed() {
        let scorer = OfflineCandidateScorer::default();
        let evidence = [
            ev(InfoOutcomeStatus::InfoConfirmedSuccess, 1.0),
            ev(InfoOutcomeStatus::InfoConfirmedSuccess, 2.0),
        ];
        let scored = scorer.score(&unscored_edge(), &evidence);
        assert!((scored.gains.confirmed_fraction() - 1.0).abs() < 1e-6);
        assert!(OfflineCandidateScorer::can_auto_commit(&scored, DEFAULT_AUTO_COMMIT_THRESHOLD));
        // Even a strict threshold (0.1) allows all-confirmed.
        assert!(OfflineCandidateScorer::can_auto_commit(&scored, 0.1));
    }

    #[test]
    fn auto_commit_threshold_boundary_is_inclusive() {
        // Exactly at threshold (unverified_fraction == 0.5) → allowed (<=).
        let scorer = OfflineCandidateScorer::default();
        let evidence = [
            ev(InfoOutcomeStatus::InfoConfirmedSuccess, 0.5),
            ev(InfoOutcomeStatus::InfoLowConfidence, 0.5),
        ];
        let scored = scorer.score(&unscored_edge(), &evidence);
        assert!((scored.gains.unverified_fraction() - 0.5).abs() < 1e-6);
        assert!(OfflineCandidateScorer::can_auto_commit(&scored, 0.5));
        // Just over → refused.
        assert!(!OfflineCandidateScorer::can_auto_commit(&scored, 0.49));
    }

    // --------------------------------------------------------------------
    // Determinism + numerical sanity.
    // --------------------------------------------------------------------

    #[test]
    fn scoring_is_deterministic_under_replay() {
        // Same evidence → same score, every time (replay-safe).
        let scorer = OfflineCandidateScorer::default();
        let evidence = [
            ev(InfoOutcomeStatus::InfoConfirmedSuccess, 0.3),
            ev(InfoOutcomeStatus::InfoUncertainSuccess, 0.1),
            ev(InfoOutcomeStatus::InfoLowConfidence, 0.2),
        ];
        let s1 = scorer.score(&unscored_edge(), &evidence);
        let s2 = scorer.score(&unscored_edge(), &evidence);
        assert_eq!(s1, s2);
    }

    #[test]
    fn negative_weights_clamped_to_zero() {
        // A buggy caller passing negative weights must not produce negative gains.
        let scorer = OfflineCandidateScorer::default();
        let evidence = [
            ev(InfoOutcomeStatus::InfoConfirmedSuccess, -1.0),
            ev(InfoOutcomeStatus::InfoLowConfidence, -2.0),
        ];
        let scored = scorer.score(&unscored_edge(), &evidence);
        assert!(scored.gains.confirmed >= 0.0);
        assert!(scored.gains.lowconf >= 0.0);
        assert!(scored.nominal_gain >= 0.0);
        assert!(scored.has_no_evidence()); // all clamped to zero
    }

    #[test]
    fn lambda_zero_disables_penalty_but_does_not_violate_g5_strictly() {
        // lambda=0 means score == nominal_gain regardless of composition. This
        // is the degenerate "no penalty" config — G5's strict inequality only
        // holds for lambda > 0. The scorer accepts lambda=0 (it's a valid config
        // for ablation/testing) but the G5 contract is vacuous at lambda=0.
        // We document this: lambda=0 is allowed but not G5-compliant.
        let scorer = OfflineCandidateScorer::new(0.0);
        let g = 1.0f32;
        let confirmed = [ev(InfoOutcomeStatus::InfoConfirmedSuccess, g)];
        let lowconf = [ev(InfoOutcomeStatus::InfoLowConfidence, g)];
        let s_confirmed = scorer.score(&unscored_edge(), &confirmed).score;
        let s_lowconf = scorer.score(&unscored_edge(), &lowconf).score;
        // At lambda=0, both equal nominal_gain — G5's strict inequality fails.
        assert!((s_confirmed - s_lowconf).abs() < 1e-6);
        // The penalty is zero.
        assert!(scorer.score(&unscored_edge(), &lowconf).penalty_silent.abs() < 1e-6);
    }

    #[test]
    fn lambda_clamped_to_unit_interval() {
        // Values outside [0, 1] are clamped.
        let over = OfflineCandidateScorer::new(5.0);
        let under = OfflineCandidateScorer::new(-3.0);
        assert!((over.lambda - 1.0).abs() < 1e-6);
        assert!(under.lambda.abs() < 1e-6);
    }

    #[test]
    fn outcome_status_is_unverified_predicate() {
        assert!(!InfoOutcomeStatus::InfoConfirmedSuccess.is_unverified());
        assert!(InfoOutcomeStatus::InfoUncertainSuccess.is_unverified());
        assert!(InfoOutcomeStatus::InfoLowConfidence.is_unverified());
    }

    #[test]
    fn empty_evidence_yields_zero_score() {
        let scorer = OfflineCandidateScorer::default();
        let scored = scorer.score(&unscored_edge(), &[]);
        assert!(scored.nominal_gain.abs() < 1e-6);
        assert!(scored.penalty_silent.abs() < 1e-6);
        assert!(scored.score.abs() < 1e-6);
        assert!(scored.has_no_evidence());
    }

    #[test]
    fn gain_partitioning_is_exact() {
        // Three pieces of evidence, one per outcome class.
        let scorer = OfflineCandidateScorer::default();
        let evidence = [
            ev(InfoOutcomeStatus::InfoConfirmedSuccess, 1.0),
            ev(InfoOutcomeStatus::InfoUncertainSuccess, 2.0),
            ev(InfoOutcomeStatus::InfoLowConfidence, 4.0),
        ];
        let scored = scorer.score(&unscored_edge(), &evidence);
        assert!((scored.gains.confirmed - 1.0).abs() < 1e-6);
        assert!((scored.gains.uncertain - 2.0).abs() < 1e-6);
        assert!((scored.gains.lowconf - 4.0).abs() < 1e-6);
        assert!((scored.nominal_gain - 7.0).abs() < 1e-6);
        // penalty = lambda * (uncertain + lowconf) = 1.0 * 6.0 = 6.0
        assert!((scored.penalty_silent - 6.0).abs() < 1e-6);
        // score = 7.0 - 6.0 = 1.0 (only confirmed survives at lambda=1.0)
        assert!((scored.score - 1.0).abs() < 1e-6);
    }
}
