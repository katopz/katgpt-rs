//! CNA Discovery Example — Discover contrastive neuron circuits from synthetic activations.
//!
//! Demonstrates:
//! - Constructing positive/negative activation sets
//! - Running `cna_discover()` to find discriminating neurons
//! - Universal neuron detection
//! - Late-layer concentration analysis
//!
//! Run: `cargo run --example cna_01_discovery --features cna_steering`
//!
//! # What This Proves
//!
//! - **Discovery correctness**: `cna_discover()` identifies neurons with the largest
//!   mean activation difference between positive and negative sets.
//! - **Late-layer concentration**: Synthetic signal injected in layers 4-5 should
//!   appear in the discovered circuit's layer distribution.
//! - **Universal filtering**: `detect_universal_neurons()` flags neurons that fire
//!   regardless of prompt content.
//!
//! # What This Does NOT Prove
//!
//! - **Real model steering** — requires actual transformer forward passes.
//! - **Cross-architecture transfer** — uses synthetic activations, not real MLP output.

use katgpt_rs::pruners::{CnaDiscoveryConfig, cna_discover, detect_universal_neurons};

fn main() {
    // Simulate a small transformer: 6 layers, 128 MLP hidden units.
    let n_layers = 6;
    let mlp_hidden = 128;

    println!("=== CNA Discovery Example ===\n");
    println!("Model: {n_layers} layers x {mlp_hidden} MLP hidden");
    println!("Total MLP activations: {}\n", n_layers * mlp_hidden);

    // ── Build positive activations ────────────────────────────────
    // 10 prompts, each with activations for layers 4-5.
    // Inject signal in neurons 10-19 (high activation in positive class).
    let mut pos_data: Vec<Vec<f32>> = Vec::new();
    let mut pos_layers: Vec<usize> = Vec::new();
    let mut rng = fastrand::Rng::with_seed(42);
    for _prompt in 0..10 {
        for layer in 4..6 {
            let mut acts = vec![0.1f32; mlp_hidden];
            for val in acts[10..20].iter_mut() {
                *val = 2.0 + rng.f32();
            }
            pos_data.push(acts);
            pos_layers.push(layer);
        }
    }
    let positive_refs: Vec<(usize, &[f32])> = pos_layers
        .iter()
        .zip(pos_data.iter())
        .map(|(&l, d)| (l, d.as_slice()))
        .collect();

    // ── Build negative activations ────────────────────────────────
    // 10 prompts, same layers but no signal (uniform low activation).
    let mut neg_data: Vec<Vec<f32>> = Vec::new();
    let mut neg_layers: Vec<usize> = Vec::new();
    for _prompt in 0..10 {
        for layer in 4..6 {
            let acts = vec![0.1f32; mlp_hidden];
            neg_data.push(acts);
            neg_layers.push(layer);
        }
    }
    let negative_refs: Vec<(usize, &[f32])> = neg_layers
        .iter()
        .zip(neg_data.iter())
        .map(|(&l, d)| (l, d.as_slice()))
        .collect();

    // ── Discover circuit ──────────────────────────────────────────
    let config = CnaDiscoveryConfig::default();
    let circuit = cna_discover(
        &positive_refs,
        &negative_refs,
        n_layers,
        mlp_hidden,
        &config,
    );

    println!("Discovered circuit:");
    println!("  Neurons:           {}", circuit.neurons.len());
    println!("  Universal excluded: {}", circuit.universal_excluded.len());
    println!("  Positive samples:  {}", circuit.n_positive);
    println!("  Negative samples:  {}", circuit.n_negative);
    println!(
        "  Circuit density:   {:.4}%",
        circuit.neurons.len() as f32 / circuit.total_mlp_activations as f32 * 100.0
    );

    // Print top neurons by |δ|
    println!("\nTop neurons (by |delta|):");
    for (i, neuron) in circuit.neurons.iter().take(15).enumerate() {
        println!(
            "  #{i:2}: layer={} index={:3} delta={:.4}",
            neuron.layer, neuron.index, neuron.delta
        );
    }

    // ── Layer distribution ────────────────────────────────────────
    let mut layer_counts = vec![0usize; n_layers];
    for n in &circuit.neurons {
        if n.layer < n_layers {
            layer_counts[n.layer] += 1;
        }
    }
    println!("\nLayer distribution:");
    for (layer, count) in layer_counts.iter().enumerate() {
        let pct = *count as f32 / circuit.neurons.len().max(1) as f32 * 100.0;
        println!("  Layer {layer}: {count:2} neurons ({pct:5.1}%)");
    }

    // ── Late-layer concentration ──────────────────────────────────
    let late_start = (n_layers as f32 * (1.0 - config.late_layer_fraction)).ceil() as usize;
    let late_count: usize = layer_counts[late_start..].iter().sum();
    let late_pct = late_count as f32 / circuit.neurons.len().max(1) as f32 * 100.0;
    println!("\nLate-layer concentration (layers {late_start}-{n_layers}): {late_pct:.1}%");
    println!("Paper benchmark: ~85% in final 10% layers");

    // ── Universal neuron detection ────────────────────────────────
    // Neuron 50 fires in ALL prompts regardless of content → universal.
    let diverse: Vec<Vec<(usize, Vec<f32>)>> = (0..5)
        .map(|_| {
            (0..n_layers)
                .map(|layer| {
                    let mut acts = vec![0.0f32; mlp_hidden];
                    acts[50] = 5.0; // Always-on neuron
                    for val in acts.iter_mut() {
                        *val += rng.f32() * 0.1;
                    }
                    (layer, acts)
                })
                .collect()
        })
        .collect();
    let universal = detect_universal_neurons(&diverse, n_layers, mlp_hidden, 0.8);

    println!("\nUniversal neurons detected: {}", universal.len());
    for (layer, idx) in &universal {
        println!("  layer={layer} index={idx}");
    }

    println!("\nDiscovery complete.");
}
