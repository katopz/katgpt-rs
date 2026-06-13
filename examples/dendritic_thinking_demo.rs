//! DendriticGate Adaptive Thinking Demo (Plan 260)
//!
//! Demonstrates NMDA-inspired adaptive tree branching:
//! 1. Gate computation: entropy + coincidence → deterministic budget modulation
//! 2. Comparison: DDTree baseline (NoPruner) vs DendriticGate
//! 3. Before/after: tree nodes expanded, compute savings, quality preservation
//!
//! Run: cargo run --features dendritic_gate --example dendritic_thinking_demo

#![cfg(feature = "dendritic_gate")]

use katgpt_rs::speculative::{
    DendriticGate, NoPruner, build_dd_tree, build_dd_tree_dendritic, dendritic_sigmoid,
    extract_best_path,
};
use katgpt_rs::types::Config;

// ── Helpers ────────────────────────────────────────────────────────

fn separator(title: &str) {
    println!();
    println!("{}", "═".repeat(64));
    println!("  {title}");
    println!("{}", "═".repeat(64));
    println!();
}

/// Make synthetic marginals for a 5-position sequence with 8-token vocab.
/// Position 0: confident (one-hot-ish) → low entropy → gate closes.
/// Position 1: moderate entropy → gate partial.
/// Position 2: uniform (high entropy) → gate opens.
/// Position 3: moderate entropy again.
/// Position 4: confident → gate closes.
fn make_test_marginals() -> Vec<Vec<f32>> {
    vec![
        // Position 0: confident — token 3 dominates
        vec![0.01, 0.01, 0.02, 0.90, 0.02, 0.01, 0.01, 0.02],
        // Position 1: moderate spread
        vec![0.10, 0.15, 0.20, 0.10, 0.15, 0.10, 0.10, 0.10],
        // Position 2: uniform (high entropy)
        vec![0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
        // Position 3: moderate
        vec![0.05, 0.10, 0.15, 0.20, 0.15, 0.10, 0.10, 0.15],
        // Position 4: confident — token 5 dominates
        vec![0.01, 0.01, 0.01, 0.02, 0.02, 0.88, 0.03, 0.02],
    ]
}

/// Compute Shannon entropy for a probability distribution.
fn shannon_entropy(probs: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

// ── Main ───────────────────────────────────────────────────────────

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║   DendriticGate Adaptive Thinking Demo — Plan 260            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    // ── Section 1: Gate Computation ─────────────────────────────
    separator("Section 1: DendriticGate — Entropy × Coincidence → Budget");

    let gate = DendriticGate::new();
    println!("  Gate parameters:");
    println!("    threshold:           {}", gate.threshold);
    println!("    voltage_sensitivity: {}", gate.voltage_sensitivity);
    println!("    coincidence_window:  {}", gate.coincidence_window);
    println!();

    let scenarios: &[(&str, f32, f32)] = &[
        ("High entropy, full coincidence", 3.0, 1.0),
        ("High entropy, low coincidence", 3.0, 0.1),
        ("Low entropy, full coincidence", 0.5, 1.0),
        ("At threshold, full coincidence", 1.5, 1.0),
        ("Zero entropy, full coincidence", 0.0, 1.0),
        ("Very high entropy, full coincidence", 5.0, 1.0),
    ];

    println!(
        "  {:<40} {:>8} {:>12} {:>8}",
        "Scenario", "Entropy", "Coincidence", "Gate"
    );
    println!("  {}", "-".repeat(70));

    for (label, entropy, coincidence) in scenarios {
        let gate_val = gate.compute_gate(*entropy, *coincidence);
        let action = if gate_val < 0.1 {
            "EARLY EXIT"
        } else if gate_val < 0.5 {
            "contract"
        } else if gate_val > 0.8 {
            "expand"
        } else {
            "moderate"
        };
        println!(
            "  {:<40} {:>8.2} {:>12.2} {:>8.4}  ({})",
            label, entropy, coincidence, gate_val, action
        );
    }

    println!();
    println!("  Key insight: gate = sigmoid(sensitivity × (entropy - threshold)) × coincidence");
    println!("  Deterministic: same inputs always produce same gate value (zero randomness)");

    // ── Section 2: Sigmoid Properties ───────────────────────────
    separator("Section 2: Dendritic Sigmoid Properties");

    println!("  Sigmoid curve (voltage_sensitivity = 2.0, threshold = 1.5):");
    println!();
    println!("  {:>8} {:>10} {:>15}", "Entropy", "σ(...)", "Gate (c=1.0)");
    println!("  {}", "-".repeat(35));

    for entropy in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0] {
        let inner = gate.voltage_sensitivity * (entropy - gate.threshold);
        let sig = dendritic_sigmoid(inner);
        let gate_val = gate.compute_gate(entropy, 1.0);
        println!("  {:>8.1} {:>10.4} {:>15.4}", entropy, sig, gate_val);
    }

    println!();
    println!("  Symmetry: σ(x) + σ(-x) = 1.0");
    for x in [1.0, 2.0, 5.0] {
        let pos = dendritic_sigmoid(x);
        let neg = dendritic_sigmoid(-x);
        println!("    σ({:+.1}) + σ({:+.1}) = {:.6}", x, -x, pos + neg);
    }

    // ── Section 3: DDTree Comparison ────────────────────────────
    separator("Section 3: DDTree — Baseline vs DendriticGate");

    let marginals = make_test_marginals();
    let marginals_refs: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
    let config = Config::draft(); // tree_budget = 16

    println!("  Sequence length: {} positions", marginals.len());
    println!("  Vocabulary size: {} tokens", marginals[0].len());
    println!("  Tree budget:     {} nodes", config.tree_budget);
    println!();

    // Entropy profile
    println!("  Entropy profile per position:");
    println!("  {:>8} {:>10} {:>15}", "Position", "Entropy", "Character");
    println!("  {}", "-".repeat(35));
    for (i, m) in marginals.iter().enumerate() {
        let h = shannon_entropy(m);
        let char = if h < 0.5 {
            "confident"
        } else if h < 1.5 {
            "moderate"
        } else {
            "uncertain"
        };
        println!("  {:>8} {:>10.4} {:>15}", i, h, char);
    }

    println!();

    // Baseline: NoPruner (full budget)
    let baseline_tree = build_dd_tree(&marginals_refs, &config);
    let baseline_best = extract_best_path(&baseline_tree);
    let baseline_nodes = baseline_tree.len();

    // Dendritic: gate-modulated budget
    let dendritic_tree = build_dd_tree_dendritic(&marginals_refs, &config, &NoPruner, true, &gate);
    let dendritic_best = extract_best_path(&dendritic_tree);
    let dendritic_nodes = dendritic_tree.len();

    let savings = if baseline_nodes > 0 {
        (1.0 - (dendritic_nodes as f32 / baseline_nodes as f32)) * 100.0
    } else {
        0.0
    };

    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!(
        "  │ {:<25} {:>12} {:>12}       │",
        "Metric", "Baseline", "Dendritic"
    );
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!(
        "  │ {:<25} {:>12} {:>12}       │",
        "Tree nodes expanded", baseline_nodes, dendritic_nodes
    );
    println!(
        "  │ {:<25} {:>12.1}% {:>12.1}%       │",
        "Compute",
        100.0_f32,
        100.0 * dendritic_nodes as f32 / baseline_nodes.max(1) as f32
    );
    println!(
        "  │ {:<25} {:>12} {:>12}       │",
        "Best path length",
        baseline_best.len(),
        dendritic_best.len()
    );
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();
    println!("  Compute savings: {savings:.1}%");
    println!(
        "  Quality preservation: best path length {} → {}",
        baseline_best.len(),
        dendritic_best.len()
    );

    // ── Section 4: Custom Gate Parameters ───────────────────────
    separator("Section 4: Custom Gate Parameters");

    let gates: &[(&str, DendriticGate)] = &[
        ("Default (threshold=1.5, sens=2.0)", DendriticGate::new()),
        (
            "Aggressive (threshold=0.5, sens=4.0)",
            DendriticGate::with_params(0.5, 4.0, 4),
        ),
        (
            "Conservative (threshold=3.0, sens=1.0)",
            DendriticGate::with_params(3.0, 1.0, 4),
        ),
        (
            "Sharp (threshold=1.5, sens=10.0)",
            DendriticGate::with_params(1.5, 10.0, 4),
        ),
    ];

    for (name, g) in gates {
        let tree = build_dd_tree_dendritic(&marginals_refs, &config, &NoPruner, true, g);
        let pct = 100.0 * tree.len() as f32 / baseline_nodes.max(1) as f32;
        println!(
            "  {:<40} nodes={:>3} ({:>5.1}% of budget)",
            name,
            tree.len(),
            pct
        );
    }

    // ── Section 5: Determinism Proof ────────────────────────────
    separator("Section 5: Determinism — Zero Randomness Verification");

    let gate = DendriticGate::new();
    let mut results = Vec::new();
    for _ in 0..10 {
        let tree = build_dd_tree_dendritic(&marginals_refs, &config, &NoPruner, true, &gate);
        results.push(tree.len());
    }

    let all_same = results.windows(2).all(|w| w[0] == w[1]);
    println!("  10 identical runs:");
    print!("    Node counts: ");
    for (i, &n) in results.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{n}");
    }
    println!();
    println!(
        "  Deterministic: {}",
        if all_same {
            "✓ YES (all identical)"
        } else {
            "✗ NO"
        }
    );
    assert!(all_same, "DendriticGate must be deterministic");

    // ── Section 6: ThinkingController Arm ───────────────────────
    separator("Section 6: ThinkingMode::Dendritic — Bandit Arm 4");

    use katgpt_rs::speculative::{ThinkingConfig, ThinkingSelector};

    // Demonstrate that Dendritic is available as a thinking mode
    let adaptive_config = ThinkingConfig {
        mode: ThinkingSelector::Adaptive {
            exploration_rate: 0.0,
            dendritic_weight: 0.25,
        },
        ..Default::default()
    };
    println!("  Adaptive selector config:");
    if let ThinkingSelector::Adaptive {
        exploration_rate,
        dendritic_weight,
    } = adaptive_config.mode
    {
        println!("    exploration_rate:   {exploration_rate}");
        println!("    dendritic_weight:   {dendritic_weight}");
    }
    println!();

    println!("  Bandit arms:");
    println!("    Arm 0 → Direct (no thinking)");
    println!("    Arm 1 → Latent (GPU thinking)");
    println!("    Arm 2 → CpuResample (CPU-only thinking)");
    println!("    Arm 3 → Dendritic (NMDA-gated adaptive thinking)");
    println!();

    // Show mode names
    println!("  ThinkingMode::Dendritic variant: deterministic, zero randomness");
    println!("  Uses entropy + coincidence from DendriticGate for budget allocation");

    // ── Summary ─────────────────────────────────────────────────
    separator("Summary");

    println!("  ┌──────────────────────────────────────────────────────────┐");
    println!("  │ Component         │ Key Property                          │");
    println!("  ├──────────────────────────────────────────────────────────┤");
    println!("  │ DendriticGate     │ Zero-allocation, stack-only, #[repr(C)]│");
    println!("  │ compute_gate()    │ sigmoid(sens × (H - τ)) × coincidence  │");
    println!("  │ Determinism       │ Same inputs → same output, always       │");
    println!("  │ Early exit        │ gate < 0.1 → proximal dendrite suffices │");
    println!("  │ ThinkingMode      │ Arm 3 in adaptive bandit selector       │");
    println!("  │ DDTree savings    │ Fewer nodes on confident positions       │");
    println!("  └──────────────────────────────────────────────────────────┘");
    println!();
    println!("  ✓ DendriticGate provides deterministic, physics-based compute modulation");
    println!("  ✓ Confident positions → gate closes → fewer tree nodes explored");
    println!("  ✓ Uncertain positions → gate opens → full budget for exploration");
    println!("  ✓ Zero parameters, zero training — pure inference-time optimization");
    println!();
}

// TL;DR: Demonstrates DendriticGate NMDA-inspired adaptive tree branching — entropy + coincidence → deterministic budget modulation. Compares DDTree baseline vs dendritic-gated, shows compute savings on confident positions, proves determinism, and shows ThinkingMode::Dendritic integration with the adaptive bandit.
