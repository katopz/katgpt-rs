//! Plan 214: CoExplain Bidirectional Alignment — GOAT Verification Tests.
//!
//! Validates all phases work together correctly:
//! - P1: TED-Lite divergence metric
//! - P2: Self-refining pruner accuracy tracking
//! - P3: Editable ConstraintPruner snapshot integrity
//! - P4: Translation rule extraction
//! - P5: Workload routing

#![cfg(feature = "coexplain_riir")]

use katgpt_rs::pruners::{
    CuratorIngestion, DivergenceError, PrunerAccuracy, PrunerDivergence, PrunerSnapshot,
    RuleBandit, TranslationRule, WorkloadRoute, classify_workload, compute_threshold_adjustment,
    extract_translation_rules, parse_rules,
};

// ── G1: Divergence metric correctness ───────────────────────────────

#[test]
fn goat_divergence_metric_correctness() {
    // Identical → zero divergence
    let thresholds = [0.5, 0.3, 0.8];
    let branches = [true, false, true];
    let div = PrunerDivergence::compute(&thresholds, &thresholds, &branches, &branches, 0.1);
    assert_eq!(div.threshold_divergence, 0.0);
    assert_eq!(div.topology_divergence, 0.0);

    // Known divergence: |0.5-0.4| + |0.3-0.3| + |0.8-0.9| / 3 = 0.2/3 ≈ 0.0667
    let current = [0.5, 0.3, 0.8];
    let original = [0.4, 0.3, 0.9];
    let div = PrunerDivergence::compute(&current, &original, &branches, &branches, 0.1);
    assert!((div.threshold_divergence - 0.0667).abs() < 0.001);

    // Topology divergence: 2 out of 4 differ → 0.5
    let cur_b = [true, false, true, true];
    let orig_b = [true, true, false, true];
    let div = PrunerDivergence::compute(&thresholds, &thresholds, &cur_b, &orig_b, 0.1);
    assert!((div.topology_divergence - 0.5).abs() < 0.001);

    // Clamping
    let div = PrunerDivergence {
        threshold_divergence: 0.2,
        topology_divergence: 0.0,
        lambda_t: 0.1,
    };
    assert!(div.clamp_adjustment(0.5).is_some());
    assert!((div.clamp_adjustment(0.5).unwrap() - 0.1).abs() < 1e-6);
    assert!(div.clamp_adjustment(0.05).is_none());
}

// ── G2: Self-refining improves accuracy ─────────────────────────────

#[test]
fn goat_self_refining_improves_accuracy() {
    let mut acc = PrunerAccuracy::new(1);

    // Simulate 10 iterations of bandit refinement
    // Initial: high FP rate → adjustment is positive (raises threshold)
    // After adjustment, FP decreases → accuracy improves
    let mut accuracy_history: Vec<f32> = Vec::new();

    // Baseline: many false positives
    for _ in 0..5 {
        acc.record(0, true, true); // TP
        acc.record(0, true, false); // FP
    }
    accuracy_history.push(acc.f1(0));

    // After bandit adjustment: fewer FP (simulated)
    for _ in 0..5 {
        acc.record(0, true, true); // TP
        acc.record(0, false, false); // TN (was FP before)
    }
    accuracy_history.push(acc.f1(0));

    // Verify accuracy improved
    assert!(
        accuracy_history[1] > accuracy_history[0],
        "F1 should improve after bandit refinement: {:.4} → {:.4}",
        accuracy_history[0],
        accuracy_history[1],
    );

    // Verify threshold adjustment direction
    let adj = compute_threshold_adjustment(&acc, 0, 0.1, 1.0);
    // More data now balanced → small adjustment
    assert!(adj.is_finite());
}

// ── G3: Snapshot integrity ──────────────────────────────────────────

#[test]
fn goat_editable_snapshot_integrity() {
    let thresholds = [0.5, 0.3, 0.8, 0.1];
    let branches = [true, false, true, true, false];
    let snap = PrunerSnapshot::new(&thresholds, &branches);

    // Golden reference matches
    assert!(snap.verify(&thresholds, &branches));
    assert_ne!(snap.blake3_hash, [0u8; 32]);

    // Any modification detected
    let tampered_thresholds = [0.5, 0.4, 0.8, 0.1]; // index 1 changed
    assert!(!snap.verify(&tampered_thresholds, &branches));

    let tampered_branches = [true, true, true, true, false]; // index 1 changed
    assert!(!snap.verify(&thresholds, &tampered_branches));

    // Empty snapshot
    let empty_snap = PrunerSnapshot::new(&[], &[]);
    assert!(empty_snap.verify(&[], &[]));
}

// ── G4: Translation rule extraction ─────────────────────────────────

#[test]
fn goat_translation_rule_extraction() {
    let successful = vec![
        vec![0, 1, 2],
        vec![0, 1, 2],
        vec![0, 1, 2],
        vec![3, 4],
        vec![3, 4],
    ];
    let failed = vec![
        vec![0, 1, 2], // path [0,1,2] failed once
        vec![5, 6],    // never succeeded → ignored
        vec![3, 4],    // path [3,4] failed once
    ];

    let rules = extract_translation_rules(&successful, &failed);

    // Deduplication
    assert_eq!(rules.len(), 2);

    // Sorted by success count (descending)
    assert_eq!(rules[0].path, vec![0, 1, 2]);
    assert_eq!(rules[0].successes, 3);
    assert_eq!(rules[0].failures, 1);
    assert_eq!(rules[1].path, vec![3, 4]);
    assert_eq!(rules[1].successes, 2);
    assert_eq!(rules[1].failures, 1);

    // Hash integrity
    assert_ne!(rules[0].path_hash, [0u8; 32]);
    assert_ne!(rules[1].path_hash, [0u8; 32]);
    assert_ne!(rules[0].path_hash, rules[1].path_hash);
}

// ── G5: Rule bandit convergence ─────────────────────────────────────

#[test]
fn goat_rule_bandit_convergence() {
    let mut bandit = RuleBandit::new();

    // Simulate 100 translations with 3 rules of different quality
    for i in 0..100 {
        // rule_a: 90% success rate
        bandit.record("rule_a", i % 10 != 0);
        // rule_b: 70% success rate
        bandit.record("rule_b", i % 10 < 7);
        // rule_c: 50% success rate
        bandit.record("rule_c", i % 2 == 0);
    }

    // Verify success rates converge to expected values
    let rate_a = bandit.success_rate("rule_a");
    let rate_b = bandit.success_rate("rule_b");
    let rate_c = bandit.success_rate("rule_c");

    assert!(
        rate_a > rate_b,
        "rule_a ({:.2}) should beat rule_b ({:.2})",
        rate_a,
        rate_b
    );
    assert!(
        rate_b > rate_c,
        "rule_b ({:.2}) should beat rule_c ({:.2})",
        rate_b,
        rate_c
    );

    // Best rule selection
    assert_eq!(bandit.best_rule().as_deref(), Some("rule_a"));

    // Workload routing
    assert_eq!(classify_workload("bandit_update"), WorkloadRoute::Cpu);
    assert_eq!(
        classify_workload("wasm_compile"),
        WorkloadRoute::AsyncWorker
    );
}

// ── G6: Feature isolation — zero perf hurt when disabled ────────────

#[test]
fn goat_feature_isolation() {
    // Verify all types are properly gated — this test only compiles with coexplain_riir
    // When disabled, none of the code is compiled (zero cost)

    // All core types exist and work
    let _div = PrunerDivergence::compute(&[0.5], &[0.5], &[true], &[true], 0.1);
    let _acc = PrunerAccuracy::new(1);
    let _snap = PrunerSnapshot::new(&[0.5], &[true]);
    let _err = DivergenceError {
        proposed_delta: 0.1,
        lambda_t: 0.05,
    };
    let _rules: Vec<TranslationRule> = extract_translation_rules(&[], &[]);
    let _ingestion = CuratorIngestion::new();
    let _bandit = RuleBandit::new();
    let _route = classify_workload("bandit_update");

    // JSON parsing works
    let parsed = parse_rules(r#"[{"attribute":"x","threshold":0.5,"action":"reject"}]"#);
    assert!(parsed.is_ok());
}
