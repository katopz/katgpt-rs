//! Speculative Reconciliation Interactive Demo
//!
//! Demonstrates 4 scenarios of the reconciliation engine plus an adaptive bandit demo.
//!
//! Run with:
//!   cargo run --example spec_reconciliation_demo --features spec_reconciliation

use katgpt_rs::spec_reconciliation::{
    AdaptiveReconciler, ReconciliationConfig, ReconciliationResult, ReconciliationVerdict,
    SpecReconciler, TrajectoryPoint,
};
use katgpt_rs::types::Rng;

use std::f32::consts::PI;

// ── Helpers ────────────────────────────────────────────────────

const SEED: u64 = 42;

fn make_config() -> ReconciliationConfig {
    ReconciliationConfig {
        k: 16,
        max_speed: 600.0,
        map_bounds: [0.0, 0.0, 4096.0, 4096.0],
        accept_threshold: 0.5,
        quarantine_threshold: 0.2,
        kill_rate_sigma: 5.0,
        noise_sigma: 0.1,
        dt: 1.0 / 60.0,
    }
}

fn make_h_last() -> TrajectoryPoint {
    TrajectoryPoint::from_fields(2048.0, 2048.0, 10.0, 5.0, 2.0, 0.0, 1.0, 0.0)
}

fn print_header(title: &str) {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  {title}");
    println!("══════════════════════════════════════════════════════════════");
}

fn print_result(name: &str, result: &ReconciliationResult, expected: &str) {
    let verdict_str = format!("{:?}", result.verdict);
    let match_icon = if result_matches_expected(&result.verdict, expected) {
        "✓"
    } else {
        "✗"
    };
    println!(
        "  {match_icon} {name:<30} verdict={verdict_str:<12} max_sim={:.4}  avg_sim={:.4}  hard_bounds={}  manifold={}",
        result.max_similarity,
        result.avg_similarity,
        result.hard_bounds_pass,
        result.manifold_count,
    );
    println!("    Expected: {expected}");
}

fn result_matches_expected(verdict: &ReconciliationVerdict, expected: &str) -> bool {
    match expected {
        "Accept" => *verdict == ReconciliationVerdict::Accept,
        "Quarantine" => *verdict == ReconciliationVerdict::Quarantine,
        "Quarantine/Uncertain" => {
            *verdict == ReconciliationVerdict::Quarantine
                || *verdict == ReconciliationVerdict::Uncertain
        }
        _ => false,
    }
}

// ── Scenario 1: Legitimate Play ────────────────────────────────

fn scenario_1_legitimate() {
    print_header("Scenario 1: Legitimate Play");
    println!("  Client drifts slowly from h_last — should be accepted.");
    println!();

    let config = make_config();
    let mut reconciler = SpecReconciler::new(config);
    let h_last = make_h_last();
    let mut rng = Rng::new(SEED);

    // 10 points drifting slowly: +0.1 x/frame, +0.05 y/frame
    let client: Vec<TrajectoryPoint> = (0..10)
        .map(|i| {
            let t = i as f32;
            TrajectoryPoint::from_fields(
                h_last.pos_x() + t * 0.1,
                h_last.pos_y() + t * 0.05,
                10.0,
                5.0,
                2.0, // kills unchanged
                0.0,
                1.0,
                0.0,
            )
        })
        .collect();

    let result = reconciler.reconcile(&h_last, &client, &[], 10, &mut rng);
    print_result("Legitimate drift", &result, "Accept");
}

// ── Scenario 2: Teleport Hack ──────────────────────────────────

fn scenario_2_teleport() {
    print_header("Scenario 2: Teleport Hack");
    println!("  Client jumps 5000 units in one frame — hard bound violation.");
    println!();

    let config = make_config();
    let mut reconciler = SpecReconciler::new(config);
    let h_last = make_h_last();
    let mut rng = Rng::new(SEED);

    let client = vec![
        h_last,
        TrajectoryPoint::from_fields(
            h_last.pos_x() + 5000.0, // teleport!
            h_last.pos_y(),
            0.0,
            0.0,
            2.0,
            0.0,
            1.0,
            0.0,
        ),
    ];

    let result = reconciler.reconcile(&h_last, &client, &[], 10, &mut rng);
    print_result("Teleport +5000 units", &result, "Quarantine");
}

// ── Scenario 3: Kill-Rate Hack ─────────────────────────────────

fn scenario_3_kill_rate() {
    print_header("Scenario 3: Kill-Rate Hack");
    println!("  Client claims 50 kills in one frame — kill-rate bound violation.");
    println!();

    let config = make_config();
    let mut reconciler = SpecReconciler::new(config);
    let h_last = make_h_last();
    let mut rng = Rng::new(SEED);

    let client = vec![
        h_last,
        TrajectoryPoint::from_fields(
            h_last.pos_x() + 0.1,
            h_last.pos_y(),
            10.0,
            5.0,
            50.0, // 48 kills in one frame!
            0.0,
            1.0,
            0.0,
        ),
    ];

    let result = reconciler.reconcile(&h_last, &client, &[], 10, &mut rng);
    print_result("50 kills in one frame", &result, "Quarantine");
}

// ── Scenario 4: Direction Mismatch ─────────────────────────────

fn scenario_4_direction() {
    print_header("Scenario 4: Direction Mismatch");
    println!("  h_last faces right (dir=0), client moves consistently leftward (dir=π).");
    println!();

    let config = make_config();
    let mut reconciler = SpecReconciler::new(config);
    // h_last facing right
    let h_last = TrajectoryPoint::from_fields(2048.0, 2048.0, 10.0, 0.0, 2.0, 0.0, 1.0, 0.0);
    let mut rng = Rng::new(SEED);

    // Client trajectory: moves leftward consistently (negative x velocity, direction = π)
    let client: Vec<TrajectoryPoint> = (0..10)
        .map(|i| {
            let t = i as f32;
            TrajectoryPoint::from_fields(
                h_last.pos_x() - t * 5.0, // moving left
                h_last.pos_y(),
                -50.0, // negative x velocity
                0.0,
                2.0,
                0.0,
                1.0,
                PI, // facing left
            )
        })
        .collect();

    let result = reconciler.reconcile(&h_last, &client, &[], 10, &mut rng);
    print_result(
        "Direction mismatch (right→left)",
        &result,
        "Quarantine/Uncertain",
    );
}

// ── Bonus: Adaptive Reconciler ─────────────────────────────────

fn scenario_5_adaptive() {
    print_header("Bonus: Adaptive Reconciler — Bandit Convergence");
    println!("  50 episodes of mixed legitimate / hack trajectories.");
    println!("  Bandit learns the best accept threshold via ε-greedy.");
    println!();

    let config = make_config();
    let mut adaptive = AdaptiveReconciler::new(config);
    let h_last = make_h_last();
    let mut rng = Rng::new(SEED);

    let total_episodes = 50;
    let mut correct = 0usize;
    let mut false_positives = 0usize;
    let mut false_negatives = 0usize;

    println!(
        "  {:>4}  {:<12}  {:<14}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}",
        "Ep", "Was Legit?", "Verdict", "Q[str]", "Q[med]", "Q[len]", "ε", "Best Thr"
    );
    println!("  {}", "-".repeat(96));

    for ep in 0..total_episodes {
        // ~70% legitimate, ~30% hacks
        let is_hack = rng.uniform() < 0.3;
        let was_legitimate = !is_hack;

        let client = if is_hack {
            // Mix of hack types
            let hack_type = (rng.uniform() * 3.0) as usize % 3;
            match hack_type {
                0 => {
                    // Teleport
                    vec![
                        h_last,
                        TrajectoryPoint::from_fields(
                            h_last.pos_x() + 5000.0,
                            h_last.pos_y(),
                            0.0,
                            0.0,
                            2.0,
                            0.0,
                            1.0,
                            0.0,
                        ),
                    ]
                }
                1 => {
                    // Kill-rate
                    vec![
                        h_last,
                        TrajectoryPoint::from_fields(
                            h_last.pos_x() + 0.1,
                            h_last.pos_y(),
                            10.0,
                            5.0,
                            50.0,
                            0.0,
                            1.0,
                            0.0,
                        ),
                    ]
                }
                _ => {
                    // Direction flip
                    (0..5)
                        .map(|i| {
                            TrajectoryPoint::from_fields(
                                h_last.pos_x() - i as f32 * 5.0,
                                h_last.pos_y(),
                                -50.0,
                                0.0,
                                2.0,
                                0.0,
                                1.0,
                                PI,
                            )
                        })
                        .collect()
                }
            }
        } else {
            // Legitimate: small drift
            (0..5)
                .map(|i| {
                    TrajectoryPoint::from_fields(
                        h_last.pos_x() + i as f32 * 0.1,
                        h_last.pos_y() + i as f32 * 0.05,
                        10.0,
                        5.0,
                        2.0,
                        0.0,
                        1.0,
                        0.0,
                    )
                })
                .collect()
        };

        let result = adaptive.reconcile(&h_last, &client, &[], 5, &mut rng);
        adaptive.observe_outcome(result.verdict, was_legitimate);

        // Track accuracy
        let is_correct = match (result.verdict, was_legitimate) {
            (ReconciliationVerdict::Accept, true) => true,
            (ReconciliationVerdict::Quarantine, false) => true,
            (ReconciliationVerdict::Uncertain, _) => true,
            (ReconciliationVerdict::Accept, false) => {
                false_negatives += 1;
                false
            }
            (ReconciliationVerdict::Quarantine, true) => {
                false_positives += 1;
                false
            }
        };
        if is_correct {
            correct += 1;
        }

        // Print every 10th episode and the last one
        if ep % 10 == 0 || ep == total_episodes - 1 {
            let q = adaptive.q_values();
            println!(
                "  {:>4}  {:<12}  {:<14?}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.4}  {:>10.2}",
                ep + 1,
                if was_legitimate { "legit" } else { "HACK" },
                result.verdict,
                q[0],
                q[1],
                q[2],
                adaptive.epsilon(),
                adaptive.best_threshold(),
            );
        }
    }

    println!();
    println!("  ┌──────────────────────────────────────────────┐");
    println!(
        "  │  Adaptive Bandit Summary ({} episodes)       │",
        total_episodes
    );
    println!("  ├──────────────────────────────────────────────┤");
    println!(
        "  │  Accuracy:          {:>5} / {} ({:.1}%)       │",
        correct,
        total_episodes,
        correct as f32 / total_episodes as f32 * 100.0
    );
    println!("  │  False Positives:   {false_positives:<5}                      │");
    println!("  │  False Negatives:   {false_negatives:<5}                      │");
    println!(
        "  │  Best Threshold:    {:>6.2}                     │",
        adaptive.best_threshold()
    );
    println!(
        "  │  Final Epsilon:     {:>6.4}                   │",
        adaptive.epsilon()
    );
    let q = adaptive.q_values();
    println!(
        "  │  Q-values: [{:.3}, {:.3}, {:.3}]            │",
        q[0], q[1], q[2]
    );
    println!("  └──────────────────────────────────────────────┘");

    // Show frozen state
    let frozen = adaptive.freeze();
    println!();
    println!(
        "  Frozen state: magic={:?} version={} total_pulls={}",
        std::str::from_utf8(&frozen.magic).unwrap_or("????"),
        frozen.version,
        frozen.total_pulls,
    );
}

// ── Summary Table ──────────────────────────────────────────────

fn print_summary() {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("══════════════════════════════════════════════════════════════");
    println!();
    println!("  ┌──────────────────────────────────┬───────────────────────┐");
    println!("  │ Scenario                         │ Expected Verdict      │");
    println!("  ├──────────────────────────────────┼───────────────────────┤");
    println!("  │ 1. Legitimate Play               │ Accept                │");
    println!("  │ 2. Teleport Hack                 │ Quarantine            │");
    println!("  │ 3. Kill-Rate Hack                │ Quarantine            │");
    println!("  │ 4. Direction Mismatch            │ Quarantine / Uncertain│");
    println!("  │ 5. Adaptive Bandit (50 episodes) │ Learns threshold      │");
    println!("  └──────────────────────────────────┴───────────────────────┘");
    println!();
    println!("  Pipeline: Hard Bounds → Manifold Generation → Soft Scoring → Verdict");
    println!();
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Speculative Reconciliation Engine — Interactive Demo      ║");
    println!("║  Verify offline game trajectories against plausibility     ║");
    println!("║  manifolds without any neural forward pass.                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    scenario_1_legitimate();
    scenario_2_teleport();
    scenario_3_kill_rate();
    scenario_4_direction();
    scenario_5_adaptive();
    print_summary();

    println!("Done.");
}
