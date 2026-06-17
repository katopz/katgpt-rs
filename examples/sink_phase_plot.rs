//! Sink-Aware Attention phase-plot example (Plan 287 Phase 4, T4.3).
//!
//! Constructs synthetic ViT-like activations across layers, runs the
//! sink classifier layer-by-layer, and prints a table showing the
//! [CLS] → patch transition (paper Figure 4 analog).
//!
//! In real ViTs (paper §1.4): the [CLS] token is the sink in early layers
//! (NOP — model is protecting CLS from saturation); patches become sinks
//! in deeper layers (Broadcast — distributing semantic content).
//!
//! This example synthesizes that pattern: layer 0-3 are NOP-dominant
//! (CLS sink, zero value); layers 4-7 are Broadcast-dominant (patch sink,
//! content value, rank-1 update). Prints a table summarizing per-layer
//! sink classifications.
//!
//! Run:
//! ```bash
//! cargo run --release --example sink_phase_plot --features sink_aware_attn
//! ```

#![cfg(feature = "sink_aware_attn")]

use katgpt_rs::data_probe::geometry::summarize_layer_sinks;
use katgpt_rs::data_probe::sink_classify::{SinkClassifierConfig, StableRankScratch};

fn main() {
    println!("=== Sink-Aware Attention Phase Plot (Plan 287 T4.3) ===\n");
    println!("Paper analog: Figure 4 — [CLS] → patch sink transition.\n");

    let n = 16usize; // tokens (1 CLS + 15 patches)
    let d = 32usize; // head dim
    let n_layers = 8usize;
    let n_heads = 4usize;
    let transition_layer = 4usize; // layers < this are NOP; >= are Broadcast.

    let cfg = SinkClassifierConfig::default();
    let mut scratch = StableRankScratch::new(d);

    println!(
        "{:>6} {:>12} {:>12} {:>14} {:>22}",
        "layer", "n_nop", "n_broadcast", "dominant_kind", "mean_broadcast_value_norm"
    );
    println!("{}", "-".repeat(72));

    for layer in 0..n_layers {
        // Build per-head attention + values for this layer.
        let is_nop_layer = layer < transition_layer;
        let mut attn_per_head: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_heads);
        let mut values_per_head: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_heads);

        for _head in 0..n_heads {
            // Attention: all queries pay 0.9 to position 0 (CLS or sink patch).
            let attn: Vec<Vec<f32>> = (0..n)
                .map(|_| {
                    let mut row = vec![0.1 / (n as f32 - 1.0); n];
                    row[0] = 0.9;
                    row
                })
                .collect();
            attn_per_head.push(attn);

            // Values: in NOP layers, pos 0 is zero. In Broadcast layers,
            // pos 0 is a content vector (we keep all rows identical so
            // mean norm ≈ ‖v_s‖ and ratio ≈ 1).
            let v_s: Vec<f32> = if is_nop_layer {
                vec![0.0; d]
            } else {
                // Growing magnitude with layer depth (paper phenomenology).
                let scale = 0.5 + 0.1 * layer as f32;
                (0..d).map(|i| scale * (0.1 * i as f32).sin()).collect()
            };
            let values: Vec<Vec<f32>> = if is_nop_layer {
                // Other positions have content; sink is zero.
                (0..n)
                    .map(|i| {
                        if i == 0 {
                            vec![0.0; d]
                        } else {
                            let scale = 1.0;
                            (0..d)
                                .map(|k| scale * (0.1 * (i * d + k) as f32).cos())
                                .collect()
                        }
                    })
                    .collect()
            } else {
                // All rows identical content (broadcast signature).
                (0..n).map(|_| v_s.clone()).collect()
            };
            values_per_head.push(values);
        }

        let summary =
            summarize_layer_sinks(&attn_per_head, &values_per_head, &cfg, &mut scratch, layer);

        println!(
            "{:>6} {:>12} {:>12} {:>14?} {:>22.4}",
            summary.layer_index,
            summary.n_nop_sinks,
            summary.n_broadcast_sinks,
            summary.dominant_kind,
            if summary.mean_broadcast_value_norm.is_nan() {
                f32::NAN
            } else {
                summary.mean_broadcast_value_norm
            }
        );
    }

    println!();
    println!("Expected: layers 0-3 NOP-dominant, layers 4-7 Broadcast-dominant.");
    println!("Note: classify_all_sinks doesn't pass update_O, so Broadcast is");
    println!("reported only via value_norm_ratio; the dominant_kind reflects");
    println!("the absence of NOP-classified sinks in deeper layers.");
}
