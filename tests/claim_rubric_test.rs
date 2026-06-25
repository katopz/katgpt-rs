//! Plan 307 Phase 2 — Round-trip tests on R287 §4 primitive scores.
//!
//! These tests ARE the R287 §4 table. For each of the seven shipped
//! probe/steering primitives, we construct a `Claim` fixture whose
//! `satisfied` set matches R287 §4's "Current evidence level" column, and
//! assert the validator's `honest_level` reproduces R287's score.
//!
//! If R287 revises a primitive's score, the corresponding fixture here is
//! the single source-of-truth update site.
//!
//! See `katgpt-rs/.research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md`
//! §4 for the canonical scoring table.

#![cfg(feature = "claim_rubric")]

use katgpt_rs::claim_rubric::{
    Claim, ClaimValidator, EvidenceItemId, EvidenceItemId::*, EvidenceLevel::*,
};
use katgpt_rs::claim_rubric::FeatureClass;

// ──────────────────────────────────────────────────────────────────────────
// Shared fixture slices (R287 §2.2 requirements table)
// ──────────────────────────────────────────────────────────────────────────

/// All five L1 items (R287 §2.2 row 1).
const L1_ITEMS: &[EvidenceItemId] = &[
    OperationalDefinition,
    SampleSize,
    Ablation,
    Exclusions,
    LinearProbeCalibration,
];

/// L1 items plus the six L2 items (R287 §2.2 row 2 = row 1 + L2 additions).
const L2_ITEMS: &[EvidenceItemId] = &[
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
];

// ──────────────────────────────────────────────────────────────────────────
// Test 1 — EvidenceLevel ordering
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn evidence_level_ordering() {
    assert!(L0 < L1);
    assert!(L1 < L2);
    assert!(L2 < L3);
    // Sanity: min/max behave as the validator's helper relies on.
    assert_eq!(L1.min(L3), L1);
    assert_eq!(L1.max(L3), L3);
}

// ──────────────────────────────────────────────────────────────────────────
// Test 2 — L1 claim with L1 evidence passes cleanly
// ──────────────────────────────────────────────────────────────────────────

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
    assert!(!g.downgraded, "L1 evidence backs an L1 claim");
    assert!(
        g.vocabulary_violations.is_empty(),
        "L1-safe verb 'detects' is permitted at L1"
    );
    assert!(g.missing_for_declared.is_empty());
}

// ──────────────────────────────────────────────────────────────────────────
// Test 3 — Overclaim: L3 verb on L1 evidence is flagged, honest_level = L1
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn overclaim_l1_with_l3_verb_downgrades() {
    // L1 evidence but the text uses "causally controls" (L3-only verb).
    // Per the chosen semantic (Plan 307 validator module doc):
    //   - honest_level = evidence_level = L1 (vocabulary does NOT force it down)
    //   - the L3 verb is recorded as a VocabularyViolation
    //   - downgraded = true because declared (L3) > honest (L1)
    let claim = Claim::new(
        "the probe causally controls desperation",
        FeatureClass::Detection,
        L3,
    )
    .with_evidence(L1_ITEMS);
    let g = ClaimValidator.grade(&claim);
    assert_eq!(
        g.honest_level,
        L1,
        "evidence is the binding constraint; vocab does not silently lower honest_level"
    );
    assert!(
        g.downgraded,
        "declared L3 > honest L1 → downgraded"
    );
    assert_eq!(
        g.vocabulary_violations.len(),
        1,
        "L3 verb on L1 evidence is exactly one violation"
    );
    assert_eq!(g.vocabulary_violations[0].verb, "causally controls");
    assert_eq!(g.vocabulary_violations[0].found_at_level, L1);
    assert_eq!(g.vocabulary_violations[0].max_allowed_level, L3);
}

// ──────────────────────────────────────────────────────────────────────────
// Test 4 — Prediction class without PredictControlParity cannot reach L3
//          (R287 §3 row 2)
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn prediction_class_requires_predict_control_parity_for_l3() {
    // All L3 items EXCEPT PredictControlParity satisfied. R287 §3 row 2:
    // a Prediction claim that fails predict-control parity cannot be L3.
    // The L3 requirements table includes PredictControlParity, so missing it
    // caps evidence_level at L2.
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
    assert!(
        g.honest_level < L3,
        "missing PredictControlParity caps the claim below L3"
    );
    assert_eq!(g.honest_level, L2, "evidence supports L2 (all L2 items present)");
    assert!(g.downgraded);
    assert!(
        g.missing_for_declared.contains(&PredictControlParity),
        "PredictControlParity must be flagged as missing for the L3 declaration"
    );
}

// ──────────────────────────────────────────────────────────────────────────
// Test 5 — All seven R287 §4 primitives round-trip through the validator
// ──────────────────────────────────────────────────────────────────────────

/// R287 §4 row 1: `EmotionDirections::project` (Plan 162).
/// Current evidence level: L1 (read of current emotion projection).
/// Satisfied: the five L1 items. No L2 generalization-across-variations or
/// benign-shift control yet (those are the documented gap to L2).
#[test]
fn r287_s4_emotion_directions_project_is_l1() {
    let claim = Claim::new(
        "the EmotionDirections projection reads the current valence/arousal/desperation/calm/fear state",
        FeatureClass::Detection,
        L1,
    )
    .with_evidence(L1_ITEMS);
    let g = ClaimValidator.grade(&claim);
    assert_eq!(g.honest_level, L1, "R287 §4 row 1: EmotionDirections is L1");
    assert!(!g.downgraded);
}

/// R287 §4 row 2: CNA contrastive neuron attribution (Plan 087).
/// Current evidence level: L1+ (detection + informal modulation evidence).
/// R287 §4 prose: "L1+ (detection + informal modulation evidence)". We model
/// the "+" as "L1 evidence present, L2 modulation evidence informal (not
/// formalized as the Generalization3Variations item)". So honest_level = L1.
#[test]
fn r287_s4_cna_contrastive_is_l1() {
    let claim = Claim::new(
        "the CNA contrastive direction detects deception-related neuron attribution",
        FeatureClass::Detection,
        L1,
    )
    .with_evidence(L1_ITEMS);
    let g = ClaimValidator.grade(&claim);
    assert_eq!(g.honest_level, L1, "R287 §4 row 2: CNA is L1+ → honest L1");
    assert!(!g.downgraded);
}

/// R287 §4 row 3: `FaithfulnessProbe::behavior_delta` (Plan 278).
/// Current evidence level: L2 candidate (designed as intervention; specificity
/// control TBD). R287 §4 prose: "L2 if the specificity control ships, else
/// L1+". We model the L2-candidate state as: L1 items + DownstreamEffect +
/// BenignShiftControl satisfied (the intervention-shaped evidence), but
/// Generalization3Variations / HumanGroundedValidation / BaseRateReported /
/// LatentFreshnessCheck NOT yet formalized. That gives evidence_level = L1
/// (since L2 requires ALL six L2 items).
///
/// Reconciliation note: R287 §4 says "L2 candidate" but the §2.2 L2 row
/// requires generalization across ≥3 variations as a hard gate. Since that
/// evidence is not yet shipped (Plan 278 Phase 4), the honest grade is L1,
/// NOT L2. This is the rubric doing its job — flagging that "L2 candidate"
/// is aspirational until the generalization evidence lands.
#[test]
fn r287_s4_faithfulness_probe_behavior_delta_is_l1() {
    // L1 items + DownstreamEffect + BenignShiftControl (the intervention-
    // shaped evidence FaithfulnessProbe has). Missing the other four L2
    // items → evidence caps at L1.
    let satisfied: &[EvidenceItemId] = &[
        OperationalDefinition,
        SampleSize,
        Ablation,
        Exclusions,
        LinearProbeCalibration,
        DownstreamEffect,
        BenignShiftControl,
    ];
    let claim = Claim::new(
        "the FaithfulnessProbe behavior_delta detects injected-memory unfaithfulness",
        FeatureClass::Detection,
        L2, // declared L2 candidate
    )
    .with_evidence(satisfied);
    let g = ClaimValidator.grade(&claim);
    assert_eq!(
        g.honest_level,
        L1,
        "R287 §4 row 3: FaithfulnessProbe is L2-candidate → honest L1 until generalization evidence ships"
    );
    assert!(
        g.downgraded,
        "declared L2 > honest L1 → downgraded (the rubric flags the gap)"
    );
}

/// R287 §4 row 4: `FutureBehaviorProbe` (FPCG, Plan 292).
/// Current evidence level: L1 (planned; blocked on offline training, Issue 032).
/// Declared L1; satisfied = L1 items. honest_level = L1.
#[test]
fn r287_s4_future_behavior_probe_is_l1() {
    let claim = Claim::new(
        "the FutureBehaviorProbe reads a linear forecast of future defection probability",
        FeatureClass::Prediction,
        L1,
    )
    .with_evidence(L1_ITEMS);
    let g = ClaimValidator.grade(&claim);
    assert_eq!(g.honest_level, L1, "R287 §4 row 4: FPCG is L1 (planned)");
    assert!(!g.downgraded);
}

/// R287 §4 row 5: `PosteriorGuidedPruner` (Plan 239).
/// Current evidence level: L1–L2 (records evidence + gates; gain measured).
/// R287 §4 prose: "L2: generalization across regime shifts". We model the
/// upper bound (L2) by satisfying all L2 items, reflecting that the gain has
/// been measured and generalization evidence is in Plan 239. honest_level = L2.
///
/// Reconciliation note: R287 §4 says "L1–L2" which is a range. We pick the
/// upper bound L2 for this fixture because Plan 239's GOAT gate measured the
/// gain across regime shifts (which is the Generalization3Variations item).
/// If a future audit downgrades this to L1-only, drop the L2 items from
/// `satisfied` and the test will assert L1.
#[test]
fn r287_s4_posterior_guided_pruner_is_l2() {
    let claim = Claim::new(
        "the PosteriorGuidedPruner records evidence and gates by Bayesian precision",
        FeatureClass::Detection,
        L2,
    )
    .with_evidence(L2_ITEMS);
    let g = ClaimValidator.grade(&claim);
    assert_eq!(
        g.honest_level,
        L2,
        "R287 §4 row 5: PosteriorGuidedPruner is L1–L2 → honest L2 (upper bound, gain measured)"
    );
    assert!(!g.downgraded);
}

/// R287 §4 row 6: HLA `evolve_hla` (`katgpt-core/src/sense/reconstruction.rs`).
/// Current evidence level: L1 (state update; no downstream-causal claim).
/// Declared L1; satisfied = L1 items. honest_level = L1.
#[test]
fn r287_s4_hla_evolve_is_l1() {
    let claim = Claim::new(
        "the HLA evolve_hla kernel reads the 8-dim latent state update",
        FeatureClass::Detection,
        L1,
    )
    .with_evidence(L1_ITEMS);
    let g = ClaimValidator.grade(&claim);
    assert_eq!(g.honest_level, L1, "R287 §4 row 6: HLA evolve_hla is L1");
    assert!(!g.downgraded);
}

/// R287 §4 row 7: Spectral probes (EGA / SpectralQuant / irrep).
/// Current evidence level: L1 (eigenbasis read).
/// Declared L1; satisfied = L1 items. honest_level = L1.
#[test]
fn r287_s4_spectral_probes_is_l1() {
    let claim = Claim::new(
        "the spectral probe reads the eigenbasis salience of the residual stream",
        FeatureClass::Detection,
        L1,
    )
    .with_evidence(L1_ITEMS);
    let g = ClaimValidator.grade(&claim);
    assert_eq!(g.honest_level, L1, "R287 §4 row 7: spectral probes are L1");
    assert!(!g.downgraded);
}

// ──────────────────────────────────────────────────────────────────────────
// Test 6 — Vocabulary overclaim downgrade (Plan 307 T2.4)
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn vocabulary_overclaim_l3_verb_on_l1_evidence_downgrades() {
    // "causally controls" with only L1 evidence: honest_level = L1 (evidence
    // is binding), the verb is a violation, and downgraded = true because
    // we'll declare L3.
    let claim = Claim::new(
        "the probe causally controls desperation",
        FeatureClass::Detection,
        L3,
    )
    .with_evidence(L1_ITEMS);
    let g = ClaimValidator.grade(&claim);
    assert_eq!(g.honest_level, L1);
    assert!(g.downgraded);
    assert!(!g.vocabulary_violations.is_empty());
    assert_eq!(g.vocabulary_violations[0].verb, "causally controls");
}

// ──────────────────────────────────────────────────────────────────────────
// Test 7 — Vocabulary-allowed-at-level (Plan 307 T2.5)
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn l1_safe_verb_reads_at_l1_passes_cleanly() {
    let claim = Claim::new(
        "the probe reads behavior X",
        FeatureClass::Detection,
        L1,
    )
    .with_evidence(L1_ITEMS);
    let g = ClaimValidator.grade(&claim);
    assert_eq!(g.honest_level, L1);
    assert!(g.vocabulary_violations.is_empty());
}

#[test]
fn l2_verb_induces_at_l1_is_violation() {
    // "induces" is L2-only. At L1 evidence, it's a violation. honest_level
    // stays L1 (evidence-bound); the verb is flagged.
    let claim = Claim::new(
        "the probe induces behavior change",
        FeatureClass::Detection,
        L1,
    )
    .with_evidence(L1_ITEMS);
    let g = ClaimValidator.grade(&claim);
    assert_eq!(g.honest_level, L1);
    assert!(g
        .vocabulary_violations
        .iter()
        .any(|v| v.verb == "induces"));
}

#[test]
fn l2_verb_induces_at_l2_with_l2_evidence_passes() {
    let claim = Claim::new(
        "the probe induces behavior change across three paraphrases",
        FeatureClass::Detection,
        L2,
    )
    .with_evidence(L2_ITEMS);
    let g = ClaimValidator.grade(&claim);
    assert_eq!(g.honest_level, L2);
    assert!(
        g.vocabulary_violations.is_empty(),
        "L2 verb at L2 evidence is permitted"
    );
}

// ──────────────────────────────────────────────────────────────────────────
// Test 8 — Feature-class interaction (Plan 307 T2.6)
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn prediction_class_without_parity_does_not_auto_promote_to_l3() {
    // Same as Test 4 but framed as the T2.6 contract: a Prediction claim
    // missing PredictControlParity must NOT reach L3 even if every other L3
    // item is satisfied. R287 §3 row 2.
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
    assert!(g.honest_level < L3);
    assert!(g.missing_for_declared.contains(&PredictControlParity));
}

// ──────────────────────────────────────────────────────────────────────────
// Test 9 — Promote advice lists the missing items (Plan 307 T1.10)
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn promote_advice_lists_missing_items_for_upgrade() {
    let claim = Claim::new(
        "the probe reads behavior",
        FeatureClass::Detection,
        L2,
    )
    .with_evidence(L1_ITEMS);
    let g = ClaimValidator.grade(&claim);
    let advice = ClaimValidator.promote_advice(&g);
    // L2 requires 6 additional items beyond L1; all 6 should be listed.
    assert_eq!(advice.len(), 6);
    assert!(advice.iter().all(|s| s.contains("to upgrade to L2")));
}
