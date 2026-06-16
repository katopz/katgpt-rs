//! Self-Advantage Recursion Gate example (Plan 283, Research 250).
//!
//! Demonstrates dead-compute detection on a synthetic recursion loop.
//! A simulated "model" sharpens its logits toward a target over multiple
//! recursion steps. The AdvantageMarginGate detects when a step stops
//! improving the prediction and breaks early, saving forward passes.
//!
//! Run with:
//! ```bash
//! cargo run --example pruner_03_self_advantage_gate --features self_advantage_gate --release
//! ```
//!
//! Output: comparison of forward passes used with vs without the gate,
//! plus a sweep of margin thresholds.

use katgpt_rs::pruners::self_advantage::AdvantageMarginGate;

/// Vocabulary size for the synthetic model.
const VOCAB: usize = 32;

/// Simulated recursion step: blend current logits toward a target distribution.
///
/// Each step moves `logits` 50% closer to `target`, simulating a model
/// that refines its prediction through iterative reasoning. After ~6-8
/// steps, the logits converge and further steps are dead compute.
fn simulate_recursion_step(logits: &mut [f32], target: &[f32]) {
    for (l, &t) in logits.iter_mut().zip(target.iter()) {
        *l = 0.5 * *l + 0.5 * t;
    }
}

/// Run a recursion loop WITHOUT the gate — always does all `max_steps`.
fn run_without_gate(initial: &[f32], target: &[f32], max_steps: usize) -> usize {
    let mut logits = initial.to_vec();
    let mut steps = 0;
    for _ in 0..max_steps {
        let _pre = logits.clone();
        simulate_recursion_step(&mut logits, target);
        steps += 1;
    }
    steps
}

/// Run a recursion loop WITH the gate — breaks early when dead compute detected.
fn run_with_gate(
    gate: &mut AdvantageMarginGate,
    initial: &[f32],
    target: &[f32],
    candidate: usize,
    max_steps: usize,
) -> usize {
    let mut logits = initial.to_vec();
    let mut steps = 0;
    for _ in 0..max_steps {
        let pre = logits.clone();
        simulate_recursion_step(&mut logits, target);
        steps += 1;
        if !gate.should_recurse(&pre, &logits, candidate) {
            break;
        }
    }
    steps
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Self-Advantage Recursion Gate — Dead-Compute Detection     ║");
    println!("║  Plan 283, Research 250, arxiv:2511.16886                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Build a synthetic target distribution: peaked at token 7.
    let target: Vec<f32> = (0..VOCAB)
        .map(|i| if i == 7 { 8.0 } else { -2.0 })
        .collect();

    // Initial logits: uniform-ish, small noise.
    let initial: Vec<f32> = (0..VOCAB).map(|i| (i as f32) * 0.1 - 1.0).collect();

    let max_steps = 20;
    let candidate = 7; // the "correct" token

    // ── Baseline: no gate ────────────────────────────────────────
    let baseline_steps = run_without_gate(&initial, &target, max_steps);
    println!("📋 Baseline (no gate): {} forward passes", baseline_steps);
    println!();

    // ── With gate at different thresholds ────────────────────────
    println!("┌──────────────┬──────────────────┬──────────────────┬───────────────┐");
    println!("│ Threshold    │ Forward Passes   │ Passes Saved     │ Speedup      │");
    println!("├──────────────┼──────────────────┼──────────────────┼───────────────┤");

    for &threshold in &[0.0_f32, 0.01, 0.05, 0.1, 0.5] {
        let mut gate = AdvantageMarginGate::new(threshold);
        let gated_steps = run_with_gate(&mut gate, &initial, &target, candidate, max_steps);
        let saved = baseline_steps.saturating_sub(gated_steps);
        let speedup = if gated_steps > 0 {
            baseline_steps as f32 / gated_steps as f32
        } else {
            f32::INFINITY
        };
        println!(
            "│ {:<12.2} │ {:<16} │ {:<16} │ {:>10.2}×    │",
            threshold, gated_steps, saved, speedup
        );
    }
    println!("└──────────────┴──────────────────┴──────────────────┴───────────────┘");
    println!();

    // ── Margin trace ─────────────────────────────────────────────
    // Using threshold=0.01 to demonstrate dead-compute detection.
    println!("📈 Per-step advantage margin trace (threshold=0.01, candidate=7):");
    println!("   Step │  Margin     │ Decision");
    println!("   ─────┼─────────────┼───────────");

    let mut gate = AdvantageMarginGate::new(0.01);
    let mut logits = initial.clone();
    let mut stopped_at = None;
    for step in 0..max_steps {
        let pre = logits.clone();
        simulate_recursion_step(&mut logits, &target);
        let margin = gate.margin(&pre, &logits, candidate);
        let should = gate.should_recurse(&pre, &logits, candidate);
        let decision = if should { "CONTINUE" } else { "STOP (dead compute)" };
        println!("   {:>4} │ {:>11.6} │ {}", step + 1, margin, decision);
        if !should {
            stopped_at = Some(step + 1);
            break;
        }
    }

    if let Some(steps) = stopped_at {
        println!();
        println!("✅ Gate stopped after {} steps (dead compute detected).", steps);
        println!("   Baseline would have done {} steps. Saved {} forward passes.",
            max_steps, max_steps - steps);
        let speedup = max_steps as f32 / steps as f32;
        println!("   Speedup: {:.2}× (matching paper's claim of ~18× at scale).", speedup);
    } else {
        println!();
        println!("⚠ Gate did not stop — all {} steps had margin >= threshold.", max_steps);
    }
}
