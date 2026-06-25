//! Plan 307 Phase 4 — Claim Rubric GOAT gate (correctness, not perf).
//!
//! Re-runs the seven R287 §4 primitive fixtures + the overclaim fixtures +
//! the feature-class parity fixture. All must PASS. Gates promotion of the
//! `claim_rubric` feature to default-on.
//!
//! This file is deliberately self-contained (duplicates the fixtures from
//! `tests/claim_rubric_test.rs`) so the GOAT gate can be run in isolation
//! without depending on test-module sharing machinery.

#![cfg(feature = "claim_rubric")]

use katgpt_rs::claim_rubric::{
    Claim, ClaimValidator, EvidenceItemId, EvidenceItemId::*, EvidenceLevel::*,
};
use katgpt_rs::claim_rubric::FeatureClass;

// ──────────────────────────────────────────────────────────────────────────
// Fixture slices (R287 §2.2 requirements table)
// ──────────────────────────────────────────────────────────────────────────

const L1_ITEMS: &[EvidenceItemId] = &[
    OperationalDefinition,
    SampleSize,
    Ablation,
    Exclusions,
    LinearProbeCalibration,
];

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

/// L3 items minus PredictControlParity — used for the Prediction-class parity
/// gate (R287 §3 row 2).
const L3_MINUS_PARITY: &[EvidenceItemId] = &[
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

// ──────────────────────────────────────────────────────────────────────────
// GOAT gate — all seven R287 §4 fixtures + edge cases must pass
// ──────────────────────────────────────────────────────────────────────────

#[test]
fn goat_gate_all_pass() {
    let validator = ClaimValidator;

    // ── R287 §4 row 1: EmotionDirections::project → L1 ──────────────────
    let g = validator.grade(
        &Claim::new(
            "the EmotionDirections projection reads the current emotion state",
            FeatureClass::Detection,
            L1,
        )
        .with_evidence(L1_ITEMS),
    );
    assert_eq!(g.honest_level, L1, "§4 row 1: EmotionDirections");

    // ── R287 §4 row 2: CNA contrastive → L1 ─────────────────────────────
    let g = validator.grade(
        &Claim::new(
            "the CNA contrastive direction detects neuron attribution",
            FeatureClass::Detection,
            L1,
        )
        .with_evidence(L1_ITEMS),
    );
    assert_eq!(g.honest_level, L1, "§4 row 2: CNA");

    // ── R287 §4 row 3: FaithfulnessProbe::behavior_delta → L1 ───────────
    // (L2 candidate; honest L1 until generalization evidence ships)
    let g = validator.grade(
        &Claim::new(
            "the FaithfulnessProbe behavior_delta detects unfaithfulness",
            FeatureClass::Detection,
            L2,
        )
        .with_evidence(&[
            // L1
            OperationalDefinition,
            SampleSize,
            Ablation,
            Exclusions,
            LinearProbeCalibration,
            // Partial L2 (intervention-shaped evidence only)
            DownstreamEffect,
            BenignShiftControl,
        ]),
    );
    assert_eq!(g.honest_level, L1, "§4 row 3: FaithfulnessProbe (L2-candidate → honest L1)");

    // ── R287 §4 row 4: FutureBehaviorProbe → L1 ─────────────────────────
    let g = validator.grade(
        &Claim::new(
            "the FutureBehaviorProbe reads a forecast of future behavior",
            FeatureClass::Prediction,
            L1,
        )
        .with_evidence(L1_ITEMS),
    );
    assert_eq!(g.honest_level, L1, "§4 row 4: FutureBehaviorProbe");

    // ── R287 §4 row 5: PosteriorGuidedPruner → L2 ───────────────────────
    let g = validator.grade(
        &Claim::new(
            "the PosteriorGuidedPruner records evidence and gates by precision",
            FeatureClass::Detection,
            L2,
        )
        .with_evidence(L2_ITEMS),
    );
    assert_eq!(g.honest_level, L2, "§4 row 5: PosteriorGuidedPruner (L1–L2 → honest L2)");

    // ── R287 §4 row 6: HLA evolve_hla → L1 ──────────────────────────────
    let g = validator.grade(
        &Claim::new(
            "the HLA evolve_hla kernel reads the latent state update",
            FeatureClass::Detection,
            L1,
        )
        .with_evidence(L1_ITEMS),
    );
    assert_eq!(g.honest_level, L1, "§4 row 6: HLA evolve_hla");

    // ── R287 §4 row 7: Spectral probes → L1 ─────────────────────────────
    let g = validator.grade(
        &Claim::new(
            "the spectral probe reads the eigenbasis salience",
            FeatureClass::Detection,
            L1,
        )
        .with_evidence(L1_ITEMS),
    );
    assert_eq!(g.honest_level, L1, "§4 row 7: spectral probes");

    // ── Overclaim: L3 verb on L1 evidence → honest L1 + violation ───────
    let g = validator.grade(
        &Claim::new(
            "the probe causally controls desperation",
            FeatureClass::Detection,
            L3,
        )
        .with_evidence(L1_ITEMS),
    );
    assert_eq!(g.honest_level, L1, "overclaim: L3 verb on L1 evidence");
    assert!(g.downgraded);
    assert!(!g.vocabulary_violations.is_empty());

    // ── Prediction class without PredictControlParity → caps below L3 ───
    let g = validator.grade(
        &Claim::new(
            "the prediction probe forecasts future defection",
            FeatureClass::Prediction,
            L3,
        )
        .with_evidence(L3_MINUS_PARITY),
    );
    assert!(g.honest_level < L3, "Prediction without parity cannot reach L3");
    assert_eq!(g.honest_level, L2);
    assert!(g.missing_for_declared.contains(&PredictControlParity));

    // ── Clean L1 claim with L1-safe verb → no violations ───────────────
    let g = validator.grade(
        &Claim::new(
            "the probe reads behavior X",
            FeatureClass::Detection,
            L1,
        )
        .with_evidence(L1_ITEMS),
    );
    assert_eq!(g.honest_level, L1);
    assert!(g.vocabulary_violations.is_empty());
    assert!(!g.downgraded);
}
