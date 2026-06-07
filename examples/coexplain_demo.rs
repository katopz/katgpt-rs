//! CoExplain Bidirectional Alignment Demo (Plan 214).
//!
//! Demonstrates the full CoExplain pipeline:
//! 1. TED-Lite divergence computation
//! 2. Self-refining pruner accuracy tracking
//! 3. Editable ConstraintPruner snapshot/verify
//! 4. Translation rule extraction
//! 5. Curator rule ingestion + bandit refinement

#![cfg(feature = "coexplain_riir")]

use katgpt_rs::pruners::{
    CuratorIngestion, CuratorRule, PrunerAccuracy, PrunerDivergence, PrunerSnapshot, RuleBandit,
    TopologyAction, WorkloadRoute, adjust_topology, classify_workload,
    compute_threshold_adjustment, extract_translation_rules,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  Plan 214: CoExplain Bidirectional Alignment Demo");
    println!("═══════════════════════════════════════════════════════════\n");

    // ── Section 1: TED-Lite Divergence ───────────────────────────────
    println!("── Section 1: TED-Lite Divergence Computation ──────────");

    let original = [0.5, 0.3, 0.8];
    let current = [0.45, 0.32, 0.75];
    let branches_orig = [true, false, true];
    let branches_curr = [true, false, true];

    let div = PrunerDivergence::compute(&current, &original, &branches_curr, &branches_orig, 0.1);

    println!(
        "  Threshold divergence: {:.4} (L1 / N)",
        div.threshold_divergence
    );
    println!(
        "  Topology divergence: {:.4} (Hamming / N)",
        div.topology_divergence
    );
    println!("  Lambda_t clamp:      {:.4}", div.lambda_t);

    // Test clamping
    let clamped = div.clamp_adjustment(0.15);
    println!(
        "  Proposed Δ=0.15 → {}",
        match clamped {
            Some(v) => format!("CLAMPED to {v:.4}"),
            None => "ACCEPTED".to_string(),
        }
    );

    div.emit_diagnostic("demo_pruner", 100, 100);
    println!();

    // ── Section 2: Self-Refining Pruner ──────────────────────────────
    println!("── Section 2: Self-Refining Pruner Accuracy ────────────");

    let mut acc = PrunerAccuracy::new(3);

    // Simulate predictions
    for _ in 0..8 {
        acc.record(0, true, true); // TP — good slot
        acc.record(0, false, false); // TN
    }
    for _ in 0..4 {
        acc.record(1, true, true); // TP — ok slot
        acc.record(1, true, false); // FP
    }
    for _ in 0..2 {
        acc.record(2, false, true); // FN — bad slot
        acc.record(2, true, false); // FP
    }

    for slot in 0..3 {
        println!(
            "  Slot {}: precision={:.3} recall={:.3} f1={:.3} acceptance={:.3}",
            slot,
            acc.precision(slot),
            acc.recall(slot),
            acc.f1(slot),
            acc.acceptance_rate(slot),
        );
    }

    // Threshold adjustment
    let adj = compute_threshold_adjustment(&acc, 1, 0.1, 0.1);
    println!("  Slot 1 threshold adjustment: {adj:.4}");

    // Topology actions
    let actions = adjust_topology(&acc, 0.3, 0.7);
    println!(
        "  Topology actions: {:?}",
        actions
            .iter()
            .map(|a| match a {
                TopologyAction::Prune => "Prune",
                TopologyAction::Expand => "Expand",
                TopologyAction::Keep => "Keep",
            })
            .collect::<Vec<_>>()
    );
    println!();

    // ── Section 3: Editable ConstraintPruner Snapshot ────────────────
    println!("── Section 3: Snapshot Integrity ────────────────────────");

    let thresholds = [0.5, 0.3, 0.8, 0.1];
    let branches = [true, false, true, true, false];
    let snap = PrunerSnapshot::new(&thresholds, &branches);

    println!(
        "  Blake3 hash: {:02x}...{:02x}",
        snap.blake3_hash[0], snap.blake3_hash[31]
    );
    println!("  Verify original: {}", snap.verify(&thresholds, &branches));

    let tampered = [0.5, 0.4, 0.8, 0.1];
    println!("  Detect tampering: {}", !snap.verify(&tampered, &branches));
    println!();

    // ── Section 4: Translation Rule Extraction ──────────────────────
    println!("── Section 4: Translation Rule Extraction ──────────────");

    let successful = vec![
        vec![0, 1, 2],
        vec![0, 1, 2],
        vec![0, 1, 2],
        vec![3, 4],
        vec![3, 4],
    ];
    let failed = vec![vec![0, 1, 2], vec![5, 6]];

    let rules = extract_translation_rules(&successful, &failed);
    println!("  Extracted {} unique rules:", rules.len());
    for rule in &rules {
        println!(
            "    path={:?} successes={} failures={}",
            rule.path, rule.successes, rule.failures
        );
    }
    println!();

    // ── Section 5: Curator Ingestion + Bandit Refinement ─────────────
    println!("── Section 5: Curator Rules + Bandit Refinement ────────");

    // Ingest Curator rules
    let mut ingestion = CuratorIngestion::new();
    ingestion.ingest(CuratorRule {
        name: "bracket_depth_limit".to_string(),
        rule: r#"{"attribute":"bracket_depth","threshold":3.0,"action":"reject"}"#.to_string(),
        source: "curator".to_string(),
    });
    ingestion.ingest(CuratorRule {
        name: "token_entropy_boost".to_string(),
        rule: r#"{"attribute":"token_entropy","threshold":0.5,"action":"accept"}"#.to_string(),
        source: "user".to_string(),
    });

    let curator_rules = ingestion.drain();
    println!("  Ingested {} Curator rules:", curator_rules.len());
    for cr in &curator_rules {
        println!("    [{}] {} — {}", cr.source, cr.name, cr.rule);
    }

    // Bandit refinement
    let mut bandit = RuleBandit::new();
    bandit.record("bracket_depth_limit", true);
    bandit.record("bracket_depth_limit", true);
    bandit.record("bracket_depth_limit", true);
    bandit.record("bracket_depth_limit", false);
    bandit.record("token_entropy_boost", true);
    bandit.record("token_entropy_boost", false);
    bandit.record("token_entropy_boost", false);

    println!("\n  Bandit stats:");
    for name in &["bracket_depth_limit", "token_entropy_boost"] {
        println!(
            "    {name}: success_rate={:.1}%",
            bandit.success_rate(name) * 100.0
        );
    }
    println!("  Best rule: {}", bandit.best_rule().unwrap_or_default());

    // Workload routing
    println!("\n  Workload routing:");
    for task in &[
        "bandit_update",
        "ted_lite",
        "rule_compile",
        "wasm_compile",
        "unknown",
    ] {
        let route = classify_workload(task);
        println!(
            "    {task} → {}",
            match route {
                WorkloadRoute::Cpu => "CPU",
                WorkloadRoute::AsyncWorker => "AsyncWorker",
            }
        );
    }

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Demo complete. All phases operational.");
    println!("═══════════════════════════════════════════════════════════");
}
