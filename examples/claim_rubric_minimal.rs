//! Plan 307 Phase 3 — Claim Rubric minimal example.
//!
//! Constructs a claim for a fictional probe, grades it, and prints the
//! result. Demonstrates the `Claim` → `Grade` round-trip, including a
//! vocabulary overclaim detection.
//!
//! Run with:
//! ```bash
//! cargo run --no-default-features --features claim_rubric --example claim_rubric_minimal
//! ```

#![cfg(feature = "claim_rubric")]

use katgpt_rs::claim_rubric::FeatureClass;
use katgpt_rs::claim_rubric::{Claim, ClaimValidator, EvidenceItemId, EvidenceLevel};

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Plan 307 — Claim Rubric Runtime minimal demo");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    // ── Claim 1: honest L1 read ──────────────────────────────────────────
    // A Detection-side probe that has all five L1 items satisfied.
    let honest_l1 = Claim::new(
        "the desperation probe detects desperation at 87% accuracy",
        FeatureClass::Detection,
        EvidenceLevel::L1,
    )
    .with_evidence(&[
        EvidenceItemId::OperationalDefinition,
        EvidenceItemId::SampleSize,
        EvidenceItemId::Ablation,
        EvidenceItemId::Exclusions,
        EvidenceItemId::LinearProbeCalibration,
    ]);

    let validator = ClaimValidator;
    let grade = validator.grade(&honest_l1);
    println!("Claim 1 (honest L1):");
    println!("  text           : {}", honest_l1.text);
    println!("  declared       : {}", grade.declared_level.label());
    println!("  honest_level   : {}", grade.honest_level.label());
    println!("  downgraded     : {}", grade.downgraded);
    println!("  violations     : {}", grade.vocabulary_violations.len());
    println!();

    assert_eq!(grade.honest_level, EvidenceLevel::L1);
    assert!(!grade.downgraded);

    // ── Claim 2: overclaim (L3 verb on L1 evidence) ──────────────────────
    // Same L1 evidence, but the text uses "causally controls" (L3-only verb)
    // and declares L3. The validator flags the verb and downgrades.
    let overclaim = Claim::new(
        "the desperation probe causally controls NPC behavior",
        FeatureClass::Detection,
        EvidenceLevel::L3,
    )
    .with_evidence(&[
        EvidenceItemId::OperationalDefinition,
        EvidenceItemId::SampleSize,
        EvidenceItemId::Ablation,
        EvidenceItemId::Exclusions,
        EvidenceItemId::LinearProbeCalibration,
    ]);

    let grade = validator.grade(&overclaim);
    println!("Claim 2 (overclaim — L3 verb on L1 evidence):");
    println!("  text           : {}", overclaim.text);
    println!("  declared       : {}", grade.declared_level.label());
    println!("  honest_level   : {}", grade.honest_level.label());
    println!("  downgraded     : {}", grade.downgraded);
    println!("  violations     : {}", grade.vocabulary_violations.len());
    for v in &grade.vocabulary_violations {
        println!("    - {}", v.description());
    }
    println!();

    assert_eq!(grade.honest_level, EvidenceLevel::L1);
    assert!(grade.downgraded);
    assert!(!grade.vocabulary_violations.is_empty());

    // ── Claim 3: promote advice (declared L2, missing L2 items) ──────────
    let gap_to_l2 = Claim::new(
        "the probe reads current behavior",
        FeatureClass::Detection,
        EvidenceLevel::L2,
    )
    .with_evidence(&[
        EvidenceItemId::OperationalDefinition,
        EvidenceItemId::SampleSize,
        EvidenceItemId::Ablation,
        EvidenceItemId::Exclusions,
        EvidenceItemId::LinearProbeCalibration,
    ]);

    let grade = validator.grade(&gap_to_l2);
    let advice = validator.promote_advice(&grade);
    println!("Claim 3 (declared L2, missing L2 items — promote advice):");
    println!("  text           : {}", gap_to_l2.text);
    println!("  declared       : {}", grade.declared_level.label());
    println!("  honest_level   : {}", grade.honest_level.label());
    println!("  missing items  : {}", grade.missing_for_declared.len());
    for line in &advice {
        println!("    - {line}");
    }
    println!();

    assert_eq!(grade.honest_level, EvidenceLevel::L1);
    assert!(grade.downgraded);
    assert_eq!(
        advice.len(),
        6,
        "all six L2 items should be listed as missing"
    );

    // ── GOAT-gate primitive: ClaimValidator::passes ──────────────────────
    println!("GOAT-gate primitive (ClaimValidator::passes):");
    println!(
        "  honest L1 claim passes(L1)? : {}",
        validator.passes(&honest_l1, EvidenceLevel::L1)
    );
    println!(
        "  honest L1 claim passes(L2)? : {}",
        validator.passes(&honest_l1, EvidenceLevel::L2)
    );
    println!();

    assert!(validator.passes(&honest_l1, EvidenceLevel::L1));
    assert!(!validator.passes(&honest_l1, EvidenceLevel::L2));

    println!("✅ Claim Rubric Runtime grades three claims correctly:");
    println!("   1. honest L1 read passes cleanly");
    println!("   2. L3 verb on L1 evidence is downgraded with a violation");
    println!("   3. promote_advice lists the gap to upgrade to L2");
    println!();
    println!("See also:");
    println!("  .plans/307_claim_rubric_runtime.md   — full design");
    println!(
        "  .research/287_Probe_Steering_Claim_Evidence_Ladder_Fusion_With_267.md  — rubric source"
    );
}
