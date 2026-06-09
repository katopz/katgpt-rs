//! CNA Steering Example — Runtime modulation with discovered circuits.
//!
//! Demonstrates:
//! - Building a circuit from known neurons
//! - Creating a `CnaModulator`
//! - Sweeping multiplier values
//! - Measuring activation change and non-circuit neuron preservation
//!
//! Run: `cargo run --example cna_02_steering --features cna_steering`
//!
//! # What This Proves
//!
//! - **Modulation correctness**: `cna_modulate()` only touches circuit neurons
//!   for the current layer, leaving all other activations unchanged.
//! - **Multiplier semantics**: `m=0.0` ablates, `m=1.0` is no-op, `m>1.0` amplifies.
//! - **Quality preservation**: Non-circuit neurons have RMSE ≈ 0 after modulation.
//!
//! # What This Does NOT Prove
//!
//! - **Behavioral change** — requires real model output measurement.
//! - **Quality at scale** — synthetic activations, not real transformer output.

use std::collections::HashSet;

use katgpt_rs::pruners::{CnaCircuit, CnaModulator, CnaNeuron, cna_modulate};

fn main() {
    println!("=== CNA Steering Example ===\n");

    // ── Build a known circuit: 5 neurons in layers 4-5 ────────────
    let neurons = vec![
        CnaNeuron {
            layer: 4,
            index: 10,
            delta: 1.8,
        },
        CnaNeuron {
            layer: 4,
            index: 15,
            delta: 1.5,
        },
        CnaNeuron {
            layer: 4,
            index: 22,
            delta: 1.2,
        },
        CnaNeuron {
            layer: 5,
            index: 8,
            delta: 0.9,
        },
        CnaNeuron {
            layer: 5,
            index: 33,
            delta: 0.7,
        },
    ];

    let neuron_set = neurons.iter().map(|n| (n.layer, n.index)).collect();
    let circuit = CnaCircuit {
        neurons,
        neuron_set,
        universal_excluded: vec![],
        universal_excluded_set: HashSet::new(),
        n_positive: 10,
        n_negative: 10,
        total_mlp_activations: 6 * 128,
    };

    println!(
        "Circuit: {} neurons across layers 4-5",
        circuit.neurons.len()
    );
    println!("Total MLP activations: {}", circuit.total_mlp_activations);
    println!(
        "Circuit density: {:.4}%\n",
        circuit.neurons.len() as f32 / circuit.total_mlp_activations as f32 * 100.0
    );

    // ── Simulate hidden activations ───────────────────────────────
    let mlp_hidden = 128;
    let original: Vec<f32> = (0..mlp_hidden).map(|i| i as f32 * 0.1 + 0.5).collect();

    // Circuit neurons for layer 4 (indices 10, 15, 22)
    let circuit_indices_l4: Vec<usize> = circuit
        .neurons
        .iter()
        .filter(|n| n.layer == 4)
        .map(|n| n.index)
        .collect();

    // ── Sweep multiplier values ───────────────────────────────────
    let multipliers = [0.0, 0.5, 1.0, 1.5, 2.0];

    println!("Multiplier sweep (layer 4):");
    println!(
        "{:>12} {:>12} {:>12} {:>12} {:>12}",
        "Multiplier", "Neuron[10]", "Neuron[15]", "Neuron[22]", "Neuron[50]"
    );
    println!("{}", "-".repeat(64));

    for &m in &multipliers {
        let mut hidden = original.clone();
        let modulator = CnaModulator {
            circuit: circuit.clone(),
            multiplier: m,
        };

        cna_modulate(&mut hidden, 4, &modulator);

        println!(
            "{:>12.1} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            m,
            hidden[10],
            hidden[15],
            hidden[22],
            hidden[50], // not in circuit for layer 4 — should be unchanged
        );
    }

    // ── Quality preservation test ─────────────────────────────────
    println!("\nQuality preservation test (layer 4):");
    println!(
        "{:>12} {:>16} {:>12}",
        "Multiplier", "Non-circuit RMSE", "Status"
    );
    println!("{}", "-".repeat(44));

    for &m in &multipliers {
        let mut hidden = original.clone();
        let modulator = CnaModulator {
            circuit: circuit.clone(),
            multiplier: m,
        };
        cna_modulate(&mut hidden, 4, &modulator);

        // Measure RMSE of non-circuit neurons
        let mut sum_sq_diff = 0.0f32;
        let mut count = 0;
        for i in 0..mlp_hidden {
            if !circuit_indices_l4.contains(&i) {
                let diff = hidden[i] - original[i];
                sum_sq_diff += diff * diff;
                count += 1;
            }
        }
        let rmse = (sum_sq_diff / count.max(1) as f32).sqrt();
        let status = if rmse < 0.001 {
            "PRESERVED"
        } else {
            "DEGRADED"
        };
        println!("{:>12.1} {:>16.6} {:>12}", m, rmse, status);
    }

    // ── Cross-layer isolation test ────────────────────────────────
    println!("\nCross-layer isolation test:");
    println!("Modulating layer 5 should NOT affect layer 4 circuit neurons.");

    let mut hidden_l4 = original.clone();
    let modulator_l5 = CnaModulator {
        circuit: circuit.clone(),
        multiplier: 0.0, // ablate
    };

    // Modulate layer 5 — layer 4 neurons should be untouched.
    cna_modulate(&mut hidden_l4, 5, &modulator_l5);

    let l4_neuron_10_unchanged = (hidden_l4[10] - original[10]).abs() < f32::EPSILON;
    let l4_neuron_15_unchanged = (hidden_l4[15] - original[15]).abs() < f32::EPSILON;
    let l4_neuron_50_unchanged = (hidden_l4[50] - original[50]).abs() < f32::EPSILON;

    println!(
        "  Neuron[10] (L4 circuit): {}",
        if l4_neuron_10_unchanged {
            "unchanged"
        } else {
            "CHANGED (BUG)"
        }
    );
    println!(
        "  Neuron[15] (L4 circuit): {}",
        if l4_neuron_15_unchanged {
            "unchanged"
        } else {
            "CHANGED (BUG)"
        }
    );
    println!(
        "  Neuron[50] (non-circuit): {}",
        if l4_neuron_50_unchanged {
            "unchanged"
        } else {
            "CHANGED (BUG)"
        }
    );

    // Now modulate layer 4 — those neurons should be zeroed.
    cna_modulate(&mut hidden_l4, 4, &modulator_l5);

    let l4_neuron_10_ablated = hidden_l4[10].abs() < f32::EPSILON;
    let l4_neuron_15_ablated = hidden_l4[15].abs() < f32::EPSILON;
    let l4_neuron_50_preserved = (hidden_l4[50] - original[50]).abs() < f32::EPSILON;

    println!(
        "  Neuron[10] after L4 ablate: {}",
        if l4_neuron_10_ablated {
            "ablated"
        } else {
            "NOT ABLATED (BUG)"
        }
    );
    println!(
        "  Neuron[15] after L4 ablate: {}",
        if l4_neuron_15_ablated {
            "ablated"
        } else {
            "NOT ABLATED (BUG)"
        }
    );
    println!(
        "  Neuron[50] after L4 ablate: {}",
        if l4_neuron_50_preserved {
            "preserved"
        } else {
            "CHANGED (BUG)"
        }
    );

    println!("\nSteering sweep complete.");
    println!("\nKey insight: CNA modulates only circuit neurons,");
    println!("  preserving output quality (non-circuit RMSE = 0).");
}
