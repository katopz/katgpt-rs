//! Claim validator — grades a [`Claim`] against the R287 §2.2 rubric (Plan 307).
//!
//! The validator is stateless and rule-based. It computes a [`Grade`] from
//! `claim.text + claim.satisfied + claim.declared_level + claim.feature_class`.
//!
//! ## Algorithm
//!
//! 1. **Evidence level** — walk levels from L3 down to L0; the highest level
//!    whose [`checklist::requirements`] are a subset of `claim.satisfied` is
//!    `evidence_level`.
//! 2. **Honest level** — `honest_level = evidence_level`. Vocabulary does
//!    **not** change `honest_level` (silently forcing it down would hide the
//!    violation). Vocabulary is enforced separately at step 5.
//! 3. **Missing for declared** — items required at `declared_level` but
//!    absent from `satisfied`. Drives [`Self::promote_advice`].
//! 4. **Vocabulary violations** —
//!    [`crate::claim_rubric::vocabulary::scan`] is called with
//!    `honest_level` (the evidence-backed level). Verbs whose minimum level
//!    exceeds `honest_level` are recorded as [`VocabularyViolation`]s —
//!    "vocabulary must match evidence level" (R287 §2.3).
//! 5. **Downgraded** — `true` iff `honest_level < declared_level`.
//!
//! ## Why vocabulary does NOT silently force `honest_level` down
//!
//! R287 §2.3 says vocabulary must match evidence level. The cleanest
//! reading: verbs in the text imply a level the author is claiming. If that
//! implied level exceeds the evidence, the claim is *overclaiming* — the
//! violation is recorded in [`Grade::vocabulary_violations`] so the author
//! can fix it (either by rewording to L1/L2 language or by adding the
//! missing evidence). Silently lowering `honest_level` to hide the violation
//! would defeat the rubric's purpose: an L3 verb on L1 evidence should
//! produce an honest_level=L1 grade *and* a visible violation, not a silent
//! L1 grade with no explanation.
//!
//! Consequence: a claim with only L1-safe verbs (e.g. "the probe reads X")
//! is graded purely on its evidence — no vocabulary penalty. This is the
//! correct behavior: L1-safe verbs make no level claim, so they impose no
//! constraint.

use crate::claim_rubric::checklist;
use crate::claim_rubric::types::{
    Claim, EvidenceItemId, EvidenceLevel, Grade, VocabularyViolation,
};
use crate::claim_rubric::vocabulary;

/// Stateless claim validator. Construct with `Default::default()` or
/// `ClaimValidator` unit literal.
///
/// The statelessness is intentional — the rubric is data-only (R287 §2.2 +
/// §2.3 tables live in `checklist` / `vocabulary`), so there is no per-instance
/// configuration to carry. A v2 may add per-instance feature flags
/// (e.g. "allow LLM-judge-only for L2" overrides).
#[derive(Default, Clone, Copy, Debug)]
pub struct ClaimValidator;

impl ClaimValidator {
    /// Construct a new validator. Equivalent to `ClaimValidator::default()`.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Grade `claim` per the algorithm documented at the module top.
    #[must_use]
    pub fn grade(&self, claim: &Claim) -> Grade {
        let evidence_level = self.evidence_level(&claim.satisfied);
        // Per R287 §2.3 + Plan 307's chosen semantic (see module doc):
        // honest_level = evidence_level. Vocabulary does NOT silently force
        // honest_level down — that would hide the violation. Instead, verbs
        // above honest_level are recorded as violations so the author can fix
        // them (reword or add evidence).
        let honest_level = evidence_level;

        let missing_for_declared: Vec<EvidenceItemId> =
            checklist::requirements(claim.declared_level)
                .iter()
                .filter(|r| !claim.satisfied.contains(*r))
                .copied()
                .collect();

        let vocabulary_violations: Vec<VocabularyViolation> =
            vocabulary::scan(&claim.text, honest_level);

        let downgraded = honest_level < claim.declared_level;

        Grade {
            honest_level,
            declared_level: claim.declared_level,
            missing_for_declared,
            vocabulary_violations,
            downgraded,
        }
    }

    /// GOAT-gate primitive: returns `true` iff the claim's `honest_level` is
    /// at least `required_level`.
    ///
    /// Use this at promote/demote decision points: a primitive may only be
    /// advertised at L2/L3 if `passes(claim, L2)` / `passes(claim, L3)`.
    #[must_use]
    pub fn passes(&self, claim: &Claim, required_level: EvidenceLevel) -> bool {
        self.grade(claim).honest_level >= required_level
    }

    /// Returns one human-readable string per missing item, explaining how to
    /// upgrade the claim to its declared level.
    ///
    /// Empty when the claim is already honest at its declared level (i.e. no
    /// missing items). Lookups come from [`checklist::full_checklist`], so
    /// the descriptions stay in sync with the §5 tables.
    #[must_use]
    pub fn promote_advice(&self, grade: &Grade) -> Vec<String> {
        if grade.missing_for_declared.is_empty() {
            return Vec::new();
        }
        let target = grade.declared_level.label();
        // Build a lookup from EvidenceItemId → description by iterating the
        // §5 tables. Linear scan over 17 items — cheap, and avoids any
        // external HashMap allocation.
        let tables = checklist::full_checklist();
        grade
            .missing_for_declared
            .iter()
            .map(|missing| {
                let desc = find_description(&tables, *missing);
                format!("to upgrade to {target}, add evidence for: {desc}")
            })
            .collect()
    }

    /// Walk levels from L3 down to L0; return the highest level whose
    /// requirements are a subset of `satisfied`. Returns `L0` if even L1
    /// requirements are not all met.
    #[inline]
    fn evidence_level(&self, satisfied: &[EvidenceItemId]) -> EvidenceLevel {
        // High-to-low walk so the first level whose requirements are a subset
        // is the max such level. Early-return on the first hit.
        if supports_level(satisfied, EvidenceLevel::L3) {
            return EvidenceLevel::L3;
        }
        if supports_level(satisfied, EvidenceLevel::L2) {
            return EvidenceLevel::L2;
        }
        if supports_level(satisfied, EvidenceLevel::L1) {
            return EvidenceLevel::L1;
        }
        EvidenceLevel::L0
    }
}

/// Helper: do the `requirements(level)` items all appear in `satisfied`?
///
/// `Vec::contains` is O(n) per lookup but the satisfied set is small (≤17)
/// and `requirements(level)` is small (≤17), so this is a flat 17×17 worst
/// case — well under a microsecond. If a future caller passes a large
/// satisfied set, the caller should pre-sort + dedup and this helper can
/// switch to a binary-search variant.
#[inline]
fn supports_level(satisfied: &[EvidenceItemId], level: EvidenceLevel) -> bool {
    checklist::requirements(level)
        .iter()
        .all(|req| satisfied.contains(req))
}

/// Linear scan over `tables` for the description of `id`. Returns a generic
/// fallback string if the id is somehow not in the §5 tables (shouldn't
/// happen, but `EvidenceItemId` is `#[non_exhaustive]`).
#[inline]
fn find_description(
    tables: &[(
        crate::claim_rubric::types::ChecklistSection,
        &'static [crate::claim_rubric::types::EvidenceItem],
    )],
    id: EvidenceItemId,
) -> &'static str {
    for (_, items) in tables {
        for item in *items {
            if item.id == id {
                return item.description;
            }
        }
    }
    "(no description available)"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claim_rubric::types::{Claim, EvidenceItemId::*, EvidenceLevel::*};
    use katgpt_core::traits::FeatureClass;

    const L1_ITEMS: &[EvidenceItemId] = &[
        OperationalDefinition,
        SampleSize,
        Ablation,
        Exclusions,
        LinearProbeCalibration,
    ];

    #[test]
    fn l1_claim_with_l1_evidence_passes_cleanly() {
        let claim = Claim::new(
            "the probe detects desperation in latent state",
            FeatureClass::Detection,
            L1,
        )
        .with_evidence(L1_ITEMS);
        let g = ClaimValidator.grade(&claim);
        assert_eq!(g.honest_level, L1);
        assert!(!g.downgraded);
        assert!(g.vocabulary_violations.is_empty());
        assert!(g.missing_for_declared.is_empty());
    }

    #[test]
    fn overclaim_l3_with_l1_evidence_flags_l3_verb() {
        // L1 evidence, but text uses an L3 verb. honest_level should be L1
        // (the evidence); the L3 verb is flagged as a violation, and
        // downgraded=true because declared=L3 > honest=L1.
        let claim = Claim::new(
            "the probe causally controls desperation",
            FeatureClass::Detection,
            L3,
        )
        .with_evidence(L1_ITEMS);
        let g = ClaimValidator.grade(&claim);
        assert_eq!(g.honest_level, L1, "evidence is the binding constraint");
        assert!(g.downgraded);
        assert_eq!(g.vocabulary_violations.len(), 1);
        assert_eq!(g.vocabulary_violations[0].verb, "causally controls");
    }

    #[test]
    fn l2_verb_at_l1_claim_is_violation() {
        let claim = Claim::new(
            "the probe induces behavior change",
            FeatureClass::Detection,
            L1,
        )
        .with_evidence(L1_ITEMS);
        let g = ClaimValidator.grade(&claim);
        // L2 verb raises vocab_floor to L2; evidence is L1 → honest = L1.
        assert_eq!(g.honest_level, L1);
        assert!(g.vocabulary_violations.iter().any(|v| v.verb == "induces"));
    }

    #[test]
    fn l2_verb_at_l2_claim_with_l2_evidence_is_honest() {
        // Need all L2 items to claim L2 honestly. L2 = 5 L1 + 6 L2.
        let satisfied: &[EvidenceItemId] = &[
            OperationalDefinition,
            SampleSize,
            Ablation,
            Exclusions,
            LinearProbeCalibration,
            DownstreamEffect,
            Generalization3Variations,
            HumanGroundedValidation,
            BaseRateReported,
            LatentFreshnessCheck,
            BenignShiftControl,
        ];
        let claim = Claim::new(
            "the probe functionally steers behavior",
            FeatureClass::Detection,
            L2,
        )
        .with_evidence(satisfied);
        let g = ClaimValidator.grade(&claim);
        assert_eq!(g.honest_level, L2);
        assert!(!g.downgraded);
        assert!(g.vocabulary_violations.is_empty());
    }

    #[test]
    fn prediction_class_without_predict_control_parity_caps_below_l3() {
        // All L3 items EXCEPT PredictControlParity satisfied. R287 §3 row 2:
        // a Prediction claim that fails predict-control parity cannot be L3.
        let satisfied: &[EvidenceItemId] = &[
            // L1
            OperationalDefinition,
            SampleSize,
            Ablation,
            Exclusions,
            LinearProbeCalibration,
            // L2
            DownstreamEffect,
            Generalization3Variations,
            HumanGroundedValidation,
            BaseRateReported,
            LatentFreshnessCheck,
            BenignShiftControl,
            // L3 minus PredictControlParity
            Intervention,
            Specificity,
            GeneralCapabilityControl,
            FalsifiableCompetingExplanation,
            FailureCases,
        ];
        let claim = Claim::new(
            "the prediction probe forecasts future defection",
            FeatureClass::Prediction,
            L3,
        )
        .with_evidence(satisfied);
        let g = ClaimValidator.grade(&claim);
        // Without PredictControlParity, evidence_level caps at L2 (since L3
        // requires PredictControlParity, and L2 doesn't).
        assert_eq!(g.honest_level, L2);
        assert!(g.downgraded, "claim is downgraded from L3 to L2");
        assert!(g.missing_for_declared.contains(&PredictControlParity));
    }

    #[test]
    fn promote_advice_lists_missing_items() {
        // Claim declared L2 but missing all L2 items.
        let claim = Claim::new("the probe reads behavior", FeatureClass::Detection, L2)
            .with_evidence(L1_ITEMS);
        let g = ClaimValidator.grade(&claim);
        let advice = ClaimValidator.promote_advice(&g);
        // Should mention all 6 L2 items.
        assert_eq!(advice.len(), 6);
        assert!(advice[0].contains("to upgrade to L2"));
    }

    #[test]
    fn empty_satisfied_grades_l0() {
        let claim = Claim::new("the probe reads behavior", FeatureClass::Detection, L1);
        let g = ClaimValidator.grade(&claim);
        assert_eq!(g.honest_level, L0);
        assert!(g.downgraded);
    }
}
