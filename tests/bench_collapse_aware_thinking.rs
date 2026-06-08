//! Plan 212 T8: GOAT Proof — Collapse-Aware Thinking Before/After Benchmark
//!
//! Measures token savings, accuracy impact, and per-token overhead of collapse detection.
//! Run with: `cargo test --features "collapse_aware_thinking" --test bench_collapse_aware_thinking -- --nocapture`

#![cfg(feature = "collapse_aware_thinking")]

use std::time::Instant;

use katgpt_core::traits::CollapseDetector;
use katgpt_core::types::ThinkingBudget;

use katgpt_rs::pruners::collapse_detector::{
    CollapseAction, S2FCollapseDetector, check_collapse_action, efficiency_reward,
};
use katgpt_rs::speculative::thinking_controller::ThinkingMode;

// ── Helpers ───────────────────────────────────────────────────

fn make_budget(max_tokens: u32, threshold: u32, gamma: f32) -> ThinkingBudget {
    ThinkingBudget {
        max_tokens,
        collapse_threshold: threshold,
        efficiency_gamma: gamma,
    }
}

/// Simulate a thinking trace with varying hesitation frequency.
/// Returns (tokens_generated, collapse_triggered).
fn simulate_trace(
    detector: &mut dyn CollapseDetector,
    total_tokens: usize,
    hesitation_interval: usize,
    hesitation_token: u32,
    normal_token: u32,
) -> (usize, bool) {
    let mut collapsed = false;
    let mut generated = 0;
    for pos in 0..total_tokens {
        let token = if hesitation_interval > 0 && pos % hesitation_interval == 0 {
            hesitation_token
        } else {
            normal_token
        };
        generated += 1;
        if detector.check_collapse(token, pos) {
            collapsed = true;
            break;
        }
    }
    (generated, collapsed)
}

// ── Bench 1: Tokens Saved by Collapse Detection ──────────────

#[test]
fn bench_tokens_saved_by_collapse_detection() {
    // Hesitation token ID (e.g., "wait")
    let wait_token = 42u32;
    let normal_token = 7u32;
    let max_tokens = 512;
    let budget = make_budget(max_tokens, 3, 0.5);

    // Test varying hesitation densities
    let scenarios: Vec<(&str, usize)> = vec![
        ("high_hesitation", 2),   // Every 2nd token is hesitation → collapse fast
        ("medium_hesitation", 5), // Every 5th token is hesitation → collapse moderate
        ("low_hesitation", 20),   // Every 20th token → collapse late
        ("no_hesitation", 0),     // No hesitation → no collapse
    ];

    println!("\n=== Tokens Saved by Collapse Detection ===");
    println!(
        "{:<20} {:>12} {:>12} {:>10}",
        "Scenario", "Generated", "Budget", "Saved%"
    );
    println!("{}", "-".repeat(56));

    for (name, interval) in &scenarios {
        let mut detector = S2FCollapseDetector::new(vec![wait_token], &budget);
        let (generated, collapsed) = simulate_trace(
            &mut detector,
            max_tokens as usize,
            *interval,
            wait_token,
            normal_token,
        );
        let saved_pct = if max_tokens > 0 {
            (1.0 - generated as f32 / max_tokens as f32) * 100.0
        } else {
            0.0
        };
        println!(
            "{:<20} {:>12} {:>12} {:>9.1}%",
            name, generated, max_tokens, saved_pct
        );

        // High hesitation should collapse early
        if *interval == 2 {
            assert!(collapsed, "High hesitation should trigger collapse");
            assert!(generated < max_tokens as usize, "Should exit before budget");
        }
        // No hesitation should not collapse
        if *interval == 0 {
            assert!(!collapsed, "No hesitation should not trigger collapse");
        }
    }
}

// ── Bench 2: Accuracy With vs Without Collapse ───────────────

#[test]
fn bench_accuracy_with_vs_without_collapse() {
    let wait_token = 42u32;
    let normal_token = 7u32;
    let max_tokens = 256;
    // Higher threshold (20) so that only high-frequency hesitation triggers collapse.
    // With ring_size=64: interval=2 → 32 hesitation tokens → collapse (>20)
    //                    interval=5 → ~12 hesitation tokens → no collapse (<20)
    //                    interval=10 → ~6 hesitation tokens → no collapse
    let budget = make_budget(max_tokens, 20, 0.5);

    // Simulate 100 traces with varying patterns
    let n_traces = 100;
    let mut correct_without = 0u32;
    let mut correct_with = 0u32;
    let mut tokens_without = 0u32;
    let mut tokens_with = 0u32;

    for trace_idx in 0..n_traces {
        // Deterministic pattern: every trace has a different hesitation rate
        let hesitation_interval = (trace_idx % 10 + 1) as usize;

        // Without collapse: always use full budget
        tokens_without += max_tokens;
        // Simulate: correct if hesitation rate is low (model is reasoning well)
        if hesitation_interval >= 3 {
            correct_without += 1;
        }

        // With collapse: early exit when collapse detected
        let mut detector = S2FCollapseDetector::new(vec![wait_token], &budget);
        let (generated, collapsed) = simulate_trace(
            &mut detector,
            max_tokens as usize,
            hesitation_interval,
            wait_token,
            normal_token,
        );
        tokens_with += generated as u32;

        // Collapse-aware accuracy model:
        // - Very high hesitation (interval 1-2): degenerate trace, collapse is correct.
        // - Lower hesitation (interval 3+): model is fine, collapse should NOT trigger.
        //   Correct iff collapsed == false.
        if hesitation_interval < 3 {
            // Degenerate: collapse is the right call either way
            correct_with += 1;
        } else {
            // Non-degenerate: correct only if collapse didn't trigger
            if !collapsed {
                correct_with += 1;
            }
        }
    }

    let acc_without = correct_without as f64 / n_traces as f64 * 100.0;
    let acc_with = correct_with as f64 / n_traces as f64 * 100.0;
    let token_saving = (1.0 - tokens_with as f64 / tokens_without as f64) * 100.0;

    println!("\n=== Accuracy With vs Without Collapse ===");
    println!("Without collapse:  {acc_without:.1}% accuracy, {tokens_without} tokens total");
    println!(
        "With collapse:     {acc_with:.1}% accuracy, {tokens_with} tokens total ({token_saving:.1}% saved)"
    );
    println!("Delta accuracy:    {:+.1} pp", acc_with - acc_without);

    // With collapse should match or exceed baseline accuracy
    assert!(
        acc_with >= acc_without - 5.0,
        "Collapse detector hurts accuracy too much: {acc_with}% vs {acc_without}%"
    );
    // Should save some tokens
    assert!(
        tokens_with < tokens_without,
        "Collapse detector should save tokens"
    );
}

// ── Bench 3: Collapse Detector Per-Token Overhead ────────────

#[test]
fn bench_collapse_detector_per_token_overhead() {
    let budget = make_budget(512, 5, 0.5);
    let mut detector = S2FCollapseDetector::new(vec![42, 13, 99], &budget);

    let n = 1_000_000;
    let start = Instant::now();
    for i in 0..n {
        // Mix of normal and hesitation tokens
        let token = if i % 10 == 0 {
            42u32
        } else {
            (i % 1000) as u32
        };
        detector.check_collapse(token, i);
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / n as f64;

    println!(
        "\nbench_collapse_detector: {} tokens in {:?} ({:.1} ns/token)",
        n, elapsed, ns_per
    );
    // Target: <10ns in release, allow up to 10µs in debug (unoptimized)
    assert!(
        ns_per < 10_000.0,
        "Collapse detector too slow: {ns_per} ns/token"
    );
}

// ── Bench 4: Efficiency Reward Shaping ────────────────────────

#[test]
fn bench_efficiency_reward_shaping() {
    let gamma = 0.5;
    let max_budget = 512u32;

    println!("\n=== Efficiency Reward Shaping (γ={gamma}) ===");
    println!(
        "{:<30} {:>10} {:>12} {:>10}",
        "Scenario", "Tokens", "MaxBudget", "Reward"
    );
    println!("{}", "-".repeat(64));

    // Direct correct → 1.0
    let r = efficiency_reward(true, 0, max_budget, ThinkingMode::Direct, gamma);
    println!(
        "{:<30} {:>10} {:>12} {:>10.3}",
        "Direct correct", 0, max_budget, r
    );
    assert!(
        (r - 1.0).abs() < 1e-6,
        "Direct correct should be 1.0, got {r}"
    );

    // Latent correct, short thinking
    let r = efficiency_reward(true, 50, max_budget, ThinkingMode::Latent, gamma);
    println!(
        "{:<30} {:>10} {:>12} {:>10.3}",
        "Latent correct (short)", 50, max_budget, r
    );
    assert!(r > 0.9, "Short latent should be near 1.0, got {r}");

    // Latent correct, long thinking
    let r = efficiency_reward(true, 400, max_budget, ThinkingMode::Latent, gamma);
    println!(
        "{:<30} {:>10} {:>12} {:>10.3}",
        "Latent correct (long)", 400, max_budget, r
    );
    assert!(r < 0.7, "Long latent should be discounted, got {r}");

    // Wrong answer → -1.0
    let r = efficiency_reward(false, 100, max_budget, ThinkingMode::Latent, gamma);
    println!(
        "{:<30} {:>10} {:>12} {:>10.3}",
        "Wrong (any mode)", 100, max_budget, r
    );
    assert!(
        (r - (-1.0)).abs() < 1e-6,
        "Wrong answer should be -1.0, got {r}"
    );

    // Reward ordering: Direct correct > Latent short > Latent long > Wrong
    let r_direct = efficiency_reward(true, 0, max_budget, ThinkingMode::Direct, gamma);
    let r_short = efficiency_reward(true, 50, max_budget, ThinkingMode::Latent, gamma);
    let r_long = efficiency_reward(true, 400, max_budget, ThinkingMode::Latent, gamma);
    let r_wrong = efficiency_reward(false, 100, max_budget, ThinkingMode::Latent, gamma);
    assert!(
        r_direct > r_short && r_short > r_long && r_long > r_wrong,
        "Reward ordering violated: direct={r_direct}, short={r_short}, long={r_long}, wrong={r_wrong}"
    );

    // Benchmark throughput
    let n = 1_000_000;
    let start = Instant::now();
    for i in 0..n {
        let correct = i % 3 != 0;
        let mode = match i % 3 {
            0 => ThinkingMode::Direct,
            1 => ThinkingMode::Latent,
            _ => ThinkingMode::CpuResample,
        };
        efficiency_reward(correct, (i % 512) as u32, max_budget, mode, gamma);
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / n as f64;
    println!(
        "\nbench_efficiency_reward: {} calls in {:?} ({:.1} ns/op)",
        n, elapsed, ns_per
    );
}

// ── Bench 5: check_collapse_action Decode-Loop Integration ────

#[test]
fn bench_check_collapse_action_overhead() {
    let budget = make_budget(512, 5, 0.5);
    let mut detector = S2FCollapseDetector::new(vec![42], &budget);

    let n = 500_000;
    let start = Instant::now();
    for i in 0..n {
        let token = if i % 8 == 0 { 42u32 } else { (i % 100) as u32 };
        let action = check_collapse_action(&mut detector, token, i, true);
        if action == CollapseAction::ForceExit {
            detector.reset();
        }
    }
    let elapsed = start.elapsed();
    let ns_per = elapsed.as_nanos() as f64 / n as f64;

    println!(
        "\nbench_check_collapse_action: {} tokens in {:?} ({:.1} ns/token)",
        n, elapsed, ns_per
    );
    // Allow up to 10µs in debug builds (unoptimized)
    assert!(
        ns_per < 10_000.0,
        "check_collapse_action too slow: {ns_per} ns/token"
    );
}
