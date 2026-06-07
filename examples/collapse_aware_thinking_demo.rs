//! Collapse-Aware Adaptive Thinking Demo (Plan 212)
//!
//! Demonstrates the three-layer adaptive thinking stack:
//! 1. Pre-Decide: SelectivityRouter decides whether to think
//! 2. Mid-Think: CollapseDetector monitors reasoning for hesitation
//! 3. Post-Verify: OptionStripper prevents option-matching shortcuts
//!
//! Run: cargo run --features collapse_aware_thinking --example collapse_aware_thinking_demo

#![cfg(feature = "collapse_aware_thinking")]

use katgpt_core::traits::{CollapseDetector, ScreeningPruner};
use katgpt_rs::pruners::{OptionStripper, S2FCollapseDetector, efficiency_reward};
use katgpt_rs::speculative::{NoScreeningPruner, ThinkingMode};
use katgpt_rs::types::ThinkingBudget;

// ── Helpers ────────────────────────────────────────────────────────

fn separator(title: &str) {
    println!();
    println!("{}", "═".repeat(64));
    println!("  {title}");
    println!("{}", "═".repeat(64));
    println!();
}

fn make_detector(hesitation_tokens: Vec<u32>, threshold: u32) -> S2FCollapseDetector {
    let budget = ThinkingBudget {
        max_tokens: 4096,
        collapse_threshold: threshold,
        efficiency_gamma: 0.5,
    };
    S2FCollapseDetector::new(hesitation_tokens, &budget)
}

// ── Main ───────────────────────────────────────────────────────────

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   Collapse-Aware Adaptive Thinking Demo — Plan 212           ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    // ── Section 1: Collapse Detection — Normal Stream ────────────
    separator("Section 1: Collapse Detection — Normal Token Stream");

    let mut detector = make_detector(vec![5, 10, 15], 3);
    println!("  Hesitation tokens: [5, 10, 15]");
    println!("  Threshold τ: 3");
    println!();

    // Feed non-hesitation tokens (1, 2, 3, 4, 6, 7, 8, 9, ...)
    let normal_tokens: Vec<u32> = (1..=20).filter(|t| ![5, 10, 15].contains(t)).collect();
    println!(
        "  Feeding {} normal tokens: {:?}",
        normal_tokens.len(),
        normal_tokens
    );

    let mut collapsed = false;
    for (i, &token) in normal_tokens.iter().enumerate() {
        if detector.check_collapse(token, i) {
            collapsed = true;
            println!("    ⚠ Collapse detected at position {i} (token {token})");
            break;
        }
    }
    if !collapsed {
        println!("    ✓ No collapse detected — normal reasoning proceeds");
    }
    println!(
        "    Hesitation count: {}/{}",
        detector.hesitation_count(),
        detector.threshold()
    );

    // ── Section 2: Collapse Detection — Hesitation Stream ────────
    separator("Section 2: Collapse Detection — Hesitation Token Stream");

    let mut detector = make_detector(vec![5, 10, 15], 3);
    println!("  Feeding hesitation-heavy stream: [5, 5, 5, 10, 15, ...]");
    println!();

    let hesitation_stream: Vec<u32> = vec![5, 5, 5, 10, 15, 20, 5, 10];
    let mut collapse_position = None;
    for (i, &token) in hesitation_stream.iter().enumerate() {
        if detector.check_collapse(token, i) {
            collapse_position = Some(i);
            println!("    ⚠ Collapse detected at position {i} (token {token})");
            println!(
                "    Hesitation count: {}/{}",
                detector.hesitation_count(),
                detector.threshold()
            );
            break;
        } else {
            println!(
                "    Position {i}: token {token} → no collapse (count: {})",
                detector.hesitation_count()
            );
        }
    }
    assert!(
        collapse_position.is_some(),
        "Hesitation stream must trigger collapse"
    );
    println!(
        "    ✓ Early exit would save {} tokens",
        hesitation_stream.len() - collapse_position.unwrap() - 1
    );

    // ── Section 3: Efficiency Reward Shaping ─────────────────────
    separator("Section 3: Efficiency Reward — Direct vs Latent");

    let gamma = 0.5;
    let max_budget: u32 = 4096;
    println!("  γ (efficiency trade-off): {gamma}");
    println!("  Max budget: {max_budget} tokens");
    println!();
    println!(
        "  {:<20} {:<12} {:<12} {:<12}",
        "Scenario", "Tokens Used", "Mode", "Reward"
    );
    println!("  {}", "-".repeat(56));

    let scenarios = [
        (true, 0, ThinkingMode::Direct, "Direct correct"),
        (true, 410, ThinkingMode::Latent, "Latent 10% budget"),
        (true, 2048, ThinkingMode::Latent, "Latent 50% budget"),
        (true, 3277, ThinkingMode::Latent, "Latent 80% budget"),
        (false, 100, ThinkingMode::Direct, "Direct wrong"),
        (false, 2000, ThinkingMode::Latent, "Latent wrong"),
    ];

    for (correct, tokens, mode, label) in &scenarios {
        let reward = efficiency_reward(*correct, *tokens, max_budget, *mode, gamma);
        let mode_str = match mode {
            ThinkingMode::Direct => "Direct",
            ThinkingMode::Latent => "Latent",
            ThinkingMode::CpuResample => "CpuResample",
        };
        println!(
            "  {:<20} {:<12} {:<12} {reward:.4}",
            label, tokens, mode_str
        );
    }

    // ── Section 4: ThinkingBudget Defaults ───────────────────────
    separator("Section 4: ThinkingBudget Defaults");

    let default_budget = ThinkingBudget::default();
    println!("  max_tokens:         {}", default_budget.max_tokens);
    println!(
        "  collapse_threshold: {}",
        default_budget.collapse_threshold
    );
    println!("  efficiency_gamma:   {}", default_budget.efficiency_gamma);

    // ── Section 5: OptionStripper — MCQ Anti-Shortcut ────────────
    separator("Section 5: OptionStripper — MCQ Anti-Shortcut Verification");

    let mut stripper = OptionStripper::new(NoScreeningPruner);
    let mcq_prompt = "What is the capital of France?\nA) Paris\nB) London\nC) Berlin\nD) Madrid\nPlease think step by step.";

    println!("  Original prompt:");
    for line in mcq_prompt.lines() {
        println!("    | {line}");
    }
    println!();

    let stripped = stripper.strip_options(mcq_prompt);
    println!("  Stripped prompt (options removed):");
    for line in stripped.lines() {
        println!("    | {line}");
    }
    println!();
    println!("  Options stripped: {}", stripper.is_stripped());

    // Two-pass scoring
    println!();
    println!("  Two-pass scoring:");
    let score_matched = stripper.two_pass_score(0, 0, &[], true);
    let score_unmatched = stripper.two_pass_score(0, 0, &[], false);
    println!("    answer_matches_option=true:  score = {score_matched:.1} (both passes succeed)");
    println!("    answer_matches_option=false: score = {score_unmatched:.1} (shortcut blocked)");

    // Restore options
    let relevance = {
        let inner = stripper.restore_options();
        ScreeningPruner::relevance(inner, 0, 0, &[])
    };
    println!("  Restored options: is_stripped={}", stripper.is_stripped());
    println!("  Inner pruner relevance: {relevance:.1}");

    // ── Section 6: Freeze/Thaw Roundtrip ─────────────────────────
    separator("Section 6: Freeze/Thaw Roundtrip");

    let dir = std::env::temp_dir().join("collapse_aware_demo");
    let freeze_path = dir.join("detector.bin");

    let budget = ThinkingBudget {
        max_tokens: 4096,
        collapse_threshold: 7,
        efficiency_gamma: 0.7,
    };
    let mut detector_freeze = S2FCollapseDetector::new(vec![5, 10, 15], &budget);
    println!("  Original threshold: {}", detector_freeze.threshold());

    // Feed some tokens to build state
    for i in 0..10 {
        detector_freeze.check_collapse(5, i);
    }
    println!(
        "  After feeding 10 hesitation tokens: hesitation={}",
        detector_freeze.hesitation_count()
    );

    // Freeze
    detector_freeze.freeze(&freeze_path).expect("freeze");
    println!("  ✓ Frozen to {:?}", freeze_path);

    // Thaw into a fresh detector
    let mut detector_thaw = make_detector(vec![5, 10, 15], 1);
    println!(
        "  Fresh detector threshold (before thaw): {}",
        detector_thaw.threshold()
    );
    detector_thaw.thaw(&freeze_path).expect("thaw");
    println!(
        "  ✓ Thawed — restored threshold: {}",
        detector_thaw.threshold()
    );
    assert_eq!(
        detector_thaw.threshold(),
        7,
        "Threshold must be restored from frozen state"
    );

    // Clean up
    let _ = std::fs::remove_file(&freeze_path);

    // ── Summary Table ────────────────────────────────────────────
    separator("Summary");

    println!("  ┌──────────────────────────────────────────────────────────┐");
    println!("  │ Component          │ Key Metric                           │");
    println!("  ├──────────────────────────────────────────────────────────┤");
    println!("  │ S2FCollapseDetector│ O(1) ring buffer, zero-alloc check   │");
    println!("  │ efficiency_reward  │ Sigmoid-bounded [-1, 1] reward       │");
    println!("  │ OptionStripper     │ Two-pass min(pure, matched) gate     │");
    println!("  │ Freeze/Thaw        │ repr(C) binary, magic+version check  │");
    println!("  │ ThinkingBudget     │ Default: 4096 tok, τ=3, γ=0.5       │");
    println!("  └──────────────────────────────────────────────────────────┘");
    println!();
    println!("  ✓ Three-layer adaptive thinking pipeline demonstrated");
    println!("  ✓ Collapse detection catches hesitation patterns");
    println!("  ✓ Efficiency reward favors Direct over Latent modes");
    println!("  ✓ OptionStripper blocks option-matching shortcuts");
    println!("  ✓ Freeze/thaw preserves detector state across sessions");
    println!();
}

// TL;DR: Demonstrates the three-layer collapse-aware adaptive thinking pipeline — S2FCollapseDetector for mid-reasoning early exit, efficiency_reward for ThinkingBandit signal shaping, OptionStripper for post-verify anti-shortcut gating, and freeze/thaw for persistent state.
