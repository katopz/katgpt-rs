#![cfg(feature = "ega_attn")]
//! GOAT Proof Examples — Energy-Gated Attention (EGA) Proofs T5–T11 (Plan 139)
//!
//! Validates signal-to-noise improvement, energy monotonicity, eviction behavior,
//! combined pipeline invariants, and produces documentation tables.
//!
//! Run: `cargo test --features ega_attn --test test_139_ega_examples -- --nocapture`

use katgpt_rs::ega_attn::{EgaGate, compute_energy_gate, sigmoid, z_normalize};

// ═══════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════

/// L2 distance between two slices.
fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Compute Y = A · V where A is [seq_len × seq_len] and V is [seq_len × dim].
fn matmul_attn_values(attn: &[f32], values: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; seq_len * dim];
    for i in 0..seq_len {
        for j in 0..dim {
            let mut sum = 0.0f32;
            for k in 0..seq_len {
                sum += attn[i * seq_len + k] * values[k * dim + j];
            }
            y[i * dim + j] = sum;
        }
    }
    y
}

/// Mean vector of selected rows from a [n × dim] matrix.
fn mean_rows(data: &[f32], row_indices: &[usize], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    if row_indices.is_empty() {
        return out;
    }
    for &r in row_indices {
        for j in 0..dim {
            out[j] += data[r * dim + j];
        }
    }
    let inv = 1.0 / row_indices.len() as f32;
    for v in out.iter_mut() {
        *v *= inv;
    }
    out
}

// ═══════════════════════════════════════════════════════════════════
//  T5 — Validation loss ablation (with vs without EGA gating)
// ═══════════════════════════════════════════════════════════════════

/// EGA gating improves signal-to-noise: gated output is closer to the true
/// signal mean than the ungated output.
#[test]
fn proof_t5_gating_improves_signal_to_noise() {
    let seq_len = 16;
    let head_dim = 32;
    let signal_positions: Vec<usize> = vec![0, 4, 8, 12];

    // Build values: signal positions get magnitude 5.0, rest are noise 0.1
    let mut values = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        let mag = if signal_positions.contains(&i) {
            5.0
        } else {
            0.1
        };
        for j in 0..head_dim {
            values[i * head_dim + j] = mag;
        }
    }

    // Build embeddings X: signal positions get magnitude 2.0, noise get 0.1
    let mut x = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        let mag = if signal_positions.contains(&i) {
            2.0
        } else {
            0.1
        };
        for j in 0..head_dim {
            x[i * head_dim + j] = mag;
        }
    }

    let gate = EgaGate::new(head_dim);
    let energy = gate.energy_scores(&x, seq_len, head_dim);

    // Uniform attention weights
    let attn_ungated = vec![1.0 / seq_len as f32; seq_len * seq_len];

    // Y_no_gate = A · V
    let y_ungated = matmul_attn_values(&attn_ungated, &values, seq_len, head_dim);

    // Apply EGA gate to a copy of attention
    let mut attn_gated = attn_ungated.clone();
    let mut gate_buf = vec![0.0f32; seq_len];
    gate.gate_attention(&mut attn_gated, &energy, seq_len, &mut gate_buf);

    // Y_gated = A_gated · V (use row 0 as representative)
    let y_gated = matmul_attn_values(&attn_gated, &values, seq_len, head_dim);

    // Signal mean vector
    let signal_mean = mean_rows(&values, &signal_positions, head_dim);

    // Compare L2 distance from the signal mean (average across all query rows)
    let dist_ungated = l2_dist(&y_ungated, &signal_mean);
    let dist_gated = l2_dist(&y_gated, &signal_mean);

    assert!(
        dist_gated < dist_ungated,
        "Gated output (dist={:.4}) should be closer to signal mean than ungated (dist={:.4})",
        dist_gated,
        dist_ungated,
    );
    println!(
        "T5 PASS: gating improves signal-to-noise (gated dist={:.4} < ungated dist={:.4})",
        dist_gated, dist_ungated
    );
}

/// Reversed energy (noise gets high energy, signal gets low) produces WORSE
/// output than no gating, proving directionality matters.
#[test]
fn proof_t5_reversed_energy_worsens_output() {
    let seq_len = 16;
    let head_dim = 32;
    let signal_positions: Vec<usize> = vec![0, 4, 8, 12];

    let mut values = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        let mag = if signal_positions.contains(&i) {
            5.0
        } else {
            0.1
        };
        for j in 0..head_dim {
            values[i * head_dim + j] = mag;
        }
    }

    let mut x = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        let mag = if signal_positions.contains(&i) {
            2.0
        } else {
            0.1
        };
        for j in 0..head_dim {
            x[i * head_dim + j] = mag;
        }
    }

    let gate = EgaGate::new(head_dim);
    let energy = gate.energy_scores(&x, seq_len, head_dim);

    // Reverse: noise positions get signal energy and vice versa
    let mut reversed_energy = vec![0.0f32; seq_len];
    for i in 0..seq_len {
        let partner = seq_len - 1 - i;
        reversed_energy[i] = energy[partner];
    }

    let attn_ungated = vec![1.0 / seq_len as f32; seq_len * seq_len];
    let y_ungated = matmul_attn_values(&attn_ungated, &values, seq_len, head_dim);

    let mut attn_reversed = attn_ungated.clone();
    let mut gate_buf = vec![0.0f32; seq_len];
    gate.gate_attention(&mut attn_reversed, &reversed_energy, seq_len, &mut gate_buf);

    let y_reversed = matmul_attn_values(&attn_reversed, &values, seq_len, head_dim);
    let signal_mean = mean_rows(&values, &signal_positions, head_dim);

    let dist_ungated = l2_dist(&y_ungated, &signal_mean);
    let dist_reversed = l2_dist(&y_reversed, &signal_mean);

    assert!(
        dist_reversed > dist_ungated,
        "Reversed-energy output (dist={:.4}) should be WORSE than ungated (dist={:.4})",
        dist_reversed,
        dist_ungated,
    );
    println!(
        "T5 PASS: reversed energy worsens output (reversed dist={:.4} > ungated dist={:.4})",
        dist_reversed, dist_ungated
    );
}

// ═══════════════════════════════════════════════════════════════════
//  T6 — Energy profile over sequence
// ═══════════════════════════════════════════════════════════════════

/// Gate values are monotonically non-decreasing with energy.
#[test]
fn proof_t6_gate_monotonic_with_energy() {
    let energy = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0f32];
    let gate = compute_energy_gate(&energy, 2.2, 0.35);

    for i in 1..gate.len() {
        assert!(
            gate[i] >= gate[i - 1] - 1e-6,
            "gate[{}]={:.6} should be >= gate[{}]={:.6}",
            i,
            gate[i],
            i - 1,
            gate[i - 1]
        );
    }
    println!("T6 PASS: gate is monotonic with energy");
    for (i, (&e, &g)) in energy.iter().zip(&gate).enumerate() {
        println!("  energy[{}] = {:>6.1}  →  gate = {:.6}", i, e, g);
    }
}

/// High alpha (10.0) produces near-binary gate values.
#[test]
fn proof_t6_high_alpha_sharper_gate() {
    let energy = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0f32];
    let gate = compute_energy_gate(&energy, 10.0, 0.35);

    let sharp_count = gate.iter().filter(|&&g| g < 0.1 || g > 0.9).count();

    // With high alpha, most values should be near-binary (at least 5 of 8)
    assert!(
        sharp_count >= 5,
        "Expected >= 5 sharp values with alpha=10.0, got {}",
        sharp_count
    );
    println!(
        "T6 PASS: high alpha produces {} sharp (near-binary) values out of {}",
        sharp_count,
        gate.len()
    );
    for (i, (&e, &g)) in energy.iter().zip(&gate).enumerate() {
        let bar_len = (g * 40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  energy[{}] = {:>6.1}  →  gate = {:.4} {}", i, e, g, bar);
    }
}

/// Increasing tau causes more positions to fall below the 0.5 gate threshold.
#[test]
fn proof_t6_tau_shifts_threshold() {
    let energy = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0f32];
    let tau_values = [0.0, 0.5, 1.0, 2.0f32];

    let mut counts_below = Vec::new();
    for &tau in &tau_values {
        let gate = compute_energy_gate(&energy, 2.2, tau);
        let count = gate.iter().filter(|&&g| g < 0.5).count();
        counts_below.push(count);
    }

    // Counts should be non-decreasing with tau
    for w in counts_below.windows(2) {
        assert!(
            w[1] >= w[0],
            "count below 0.5 should not decrease with higher tau: {:?}",
            counts_below
        );
    }
    println!(
        "T6 PASS: tau shifts threshold (counts below 0.5: {:?})",
        counts_below
    );
    for (&tau, &count) in tau_values.iter().zip(&counts_below) {
        println!("  tau={:.1} → {} positions below 0.5", tau, count);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  T7 — Eviction behavior
// ═══════════════════════════════════════════════════════════════════

/// Content positions are never in the eviction set (they have higher energy).
#[test]
fn proof_t7_eviction_removes_low_energy_first() {
    let seq_len = 32;
    let head_dim = 16;
    let content_positions: Vec<usize> = (0..8).collect(); // first 8 are content

    let mut x = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        let mag = if content_positions.contains(&i) {
            3.0
        } else {
            0.1
        };
        for j in 0..head_dim {
            x[i * head_dim + j] = mag;
        }
    }

    let gate = EgaGate::new(head_dim);
    let energy = gate.energy_scores(&x, seq_len, head_dim);

    // Sort positions by energy ascending
    let mut indexed: Vec<(usize, f32)> = energy.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("T7 — Eviction report:");
    for (rank, (pos, e)) in indexed.iter().enumerate() {
        let kind = if content_positions.contains(pos) {
            "CONTENT"
        } else {
            "padding"
        };
        println!(
            "  rank {:>2}: pos {:>2} energy={:.4} [{}]",
            rank, pos, e, kind
        );
    }

    for k in [8, 16, 24] {
        let evicted: Vec<usize> = indexed[..k].iter().map(|(p, _)| *p).collect();
        let content_evicted: Vec<usize> = evicted
            .iter()
            .filter(|p| content_positions.contains(p))
            .copied()
            .collect();
        assert!(
            content_evicted.is_empty(),
            "K={}: no content positions should be evicted, but got {:?}",
            k,
            content_evicted
        );
        println!("  K={:>2}: evicted {} padding positions, 0 content ✓", k, k);
    }
    println!("T7 PASS: eviction always removes padding first");
}

/// Evicting bottom-25% by energy and renormalizing improves signal fidelity.
#[test]
fn proof_t7_eviction_preserves_attention_quality() {
    let seq_len = 16;
    let head_dim = 16;
    let signal_positions: Vec<usize> = vec![0, 4, 8, 12];

    // Values: signal positions magnitude 5.0, noise 0.1
    let mut values = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        let mag = if signal_positions.contains(&i) {
            5.0
        } else {
            0.1
        };
        for j in 0..head_dim {
            values[i * head_dim + j] = mag;
        }
    }

    // Embeddings: signal magnitude 3.0, noise 0.1
    let mut x = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        let mag = if signal_positions.contains(&i) {
            3.0
        } else {
            0.1
        };
        for j in 0..head_dim {
            x[i * head_dim + j] = mag;
        }
    }

    let gate = EgaGate::new(head_dim);
    let energy = gate.energy_scores(&x, seq_len, head_dim);

    // Full uniform attention
    let mut attn_full = vec![1.0 / seq_len as f32; seq_len * seq_len];
    let y_full = matmul_attn_values(&attn_full, &values, seq_len, head_dim);

    // Evict bottom 25% (4 positions) by energy
    let k = seq_len / 4;
    let mut indexed: Vec<(usize, f32)> = energy.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let evicted: Vec<usize> = indexed[..k].iter().map(|(p, _)| *p).collect();

    // Zero out evicted positions and renormalize
    let mut attn_evicted = attn_full.clone();
    for &pos in &evicted {
        for i in 0..seq_len {
            attn_evicted[i * seq_len + pos] = 0.0;
        }
    }
    // Renormalize rows
    for i in 0..seq_len {
        let row = &mut attn_evicted[i * seq_len..(i + 1) * seq_len];
        let sum: f32 = row.iter().sum();
        if sum > 1e-10 {
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }

    let y_evicted = matmul_attn_values(&attn_evicted, &values, seq_len, head_dim);
    let signal_mean = mean_rows(&values, &signal_positions, head_dim);

    let dist_full = l2_dist(&y_full, &signal_mean);
    let dist_evicted = l2_dist(&y_evicted, &signal_mean);

    assert!(
        dist_evicted < dist_full,
        "Evicted output (dist={:.4}) should be closer to signal than full (dist={:.4})",
        dist_evicted,
        dist_full
    );
    println!(
        "T7 PASS: eviction improves quality (evicted dist={:.4} < full dist={:.4})",
        dist_evicted, dist_full
    );
    println!("  evicted positions (bottom 25%): {:?}", evicted);
}

// ═══════════════════════════════════════════════════════════════════
//  T8 — Combined scenario
// ═══════════════════════════════════════════════════════════════════

/// Full pipeline: energy computation → gating → eviction → renormalize.
/// All invariants hold simultaneously.
#[test]
fn proof_t8_combined_pipeline() {
    let seq_len = 32;
    let head_dim = 16;
    let signal_positions: Vec<usize> = (0..8).collect();

    // Step 1: Create X with signal (8 positions, mag 3.0) + noise (24 positions, mag 0.1)
    let mut x = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        let mag = if signal_positions.contains(&i) {
            3.0
        } else {
            0.1
        };
        for j in 0..head_dim {
            x[i * head_dim + j] = mag;
        }
    }

    let ega = EgaGate::new(head_dim);

    // Step 2: Compute energy → verify signal > noise
    let energy = ega.energy_scores(&x, seq_len, head_dim);
    let min_signal_energy = signal_positions
        .iter()
        .map(|&p| energy[p])
        .fold(f32::INFINITY, f32::min);
    let max_noise_energy = (0..seq_len)
        .filter(|p| !signal_positions.contains(p))
        .map(|p| energy[p])
        .fold(f32::NEG_INFINITY, f32::max);

    assert!(
        min_signal_energy > max_noise_energy,
        "Min signal energy ({:.4}) should exceed max noise energy ({:.4})",
        min_signal_energy,
        max_noise_energy
    );
    println!(
        "T8 Step 2: energy separation OK (min_signal={:.4} > max_noise={:.4})",
        min_signal_energy, max_noise_energy
    );

    // Step 3: Apply EGA gate to attention → verify rows sum to 1
    let mut attn = vec![1.0 / seq_len as f32; seq_len * seq_len];
    let mut gate_buf = vec![0.0f32; seq_len];
    ega.gate_attention(&mut attn, &energy, seq_len, &mut gate_buf);

    for i in 0..seq_len {
        let row_sum: f32 = attn[i * seq_len..(i + 1) * seq_len].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-4,
            "After gating, row {} sums to {:.6}, expected 1.0",
            i,
            row_sum
        );
    }
    println!("T8 Step 3: gated attention rows sum to 1.0 ✓");

    // Step 4: Evict bottom 25% by energy → verify no signal evicted
    let k = seq_len / 4; // 8
    let mut indexed: Vec<(usize, f32)> = energy.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let evicted: Vec<usize> = indexed[..k].iter().map(|(p, _)| *p).collect();

    let evicted_signal: Vec<usize> = evicted
        .iter()
        .filter(|p| signal_positions.contains(p))
        .copied()
        .collect();
    assert!(
        evicted_signal.is_empty(),
        "No signal positions should be evicted, but got {:?}",
        evicted_signal
    );
    println!(
        "T8 Step 4: bottom-25% eviction removes only noise ({} positions) ✓",
        k
    );

    // Step 5: Re-normalize → verify rows still sum to 1
    for &pos in &evicted {
        for i in 0..seq_len {
            attn[i * seq_len + pos] = 0.0;
        }
    }
    for i in 0..seq_len {
        let row = &mut attn[i * seq_len..(i + 1) * seq_len];
        let sum: f32 = row.iter().sum();
        if sum > 1e-10 {
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }

    for i in 0..seq_len {
        let row_sum: f32 = attn[i * seq_len..(i + 1) * seq_len].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-4,
            "After eviction+renorm, row {} sums to {:.6}, expected 1.0",
            i,
            row_sum
        );
    }
    println!("T8 Step 5: post-eviction renormalization rows sum to 1.0 ✓");
    println!("T8 PASS: combined pipeline — all invariants hold");
}

// ═══════════════════════════════════════════════════════════════════
//  T9 — Example outputs (documentation tables)
// ═══════════════════════════════════════════════════════════════════

/// Print markdown table of energy → z-norm → gate for various alpha values.
/// Print ASCII bar chart of gate values.
#[test]
fn proof_t9_energy_profile_table() {
    let energy = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0f32];
    let alpha_values = [1.0, 2.2, 5.0, 10.0f32];
    let tau = 0.35;

    // Z-normalize for display
    let mut z = energy.to_vec();
    z_normalize(&mut z);

    println!("\n### T9 — Energy Profile Table\n");
    println!("| Energy | z-norm | α=1.0 | α=2.2 | α=5.0 | α=10.0 |");
    println!("|-------:|-------:|------:|------:|------:|-------:|");

    for (i, (&e, &zv)) in energy.iter().zip(&z).enumerate() {
        print!("| {:>6.1} | {:>6.3} |", e, zv);
        for &alpha in &alpha_values {
            let g = sigmoid(alpha * (zv - tau));
            print!(" {:>.4} |", g);
        }
        println!();
    }

    // ASCII bar chart for default alpha=2.2
    let gate_default = compute_energy_gate(&energy, 2.2, tau);
    println!("\n### Gate Bar Chart (α=2.2, τ=0.35)\n");
    println!("```");
    for (i, (&e, &g)) in energy.iter().zip(&gate_default).enumerate() {
        let bar_len = (g * 40.0).round() as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  E={:>5.1} │{} {:.3}", e, bar, g);
    }
    println!("           └────────────────────────────────────────");
    println!("           0.0                                1.0");
    println!("```");
    println!("\nT9 PASS: energy profile table printed");
}

/// Print step-by-step eviction simulation.
#[test]
fn proof_t9_eviction_simulation() {
    let seq_len = 16;
    let head_dim = 8;
    let signal_positions: Vec<usize> = vec![0, 4, 8, 12];

    let mut x = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        let mag = if signal_positions.contains(&i) {
            3.0
        } else {
            0.1
        };
        for j in 0..head_dim {
            x[i * head_dim + j] = mag;
        }
    }

    let gate = EgaGate::new(head_dim);
    let energy = gate.energy_scores(&x, seq_len, head_dim);

    // Sort by energy ascending
    let mut indexed: Vec<(usize, f32)> = energy.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!(
        "\n### T9 — Eviction Simulation (seq_len={}, head_dim={})\n",
        seq_len, head_dim
    );
    println!("**Ranked positions (low → high energy):**\n");
    println!("| Rank | Pos | Energy   | Type     |");
    println!("|-----:|----:|---------:|:---------|");
    for (rank, (pos, e)) in indexed.iter().enumerate() {
        let kind = if signal_positions.contains(pos) {
            "SIGNAL"
        } else {
            "noise"
        };
        println!("| {:>4} | {:>3} | {:>8.4} | {:8} |", rank, pos, e, kind);
    }

    // Simulate eviction at K=4, K=8, K=12
    for &k in &[4, 8, 12] {
        let evicted: Vec<usize> = indexed[..k].iter().map(|(p, _)| *p).collect();
        let evicted_signal: Vec<usize> = evicted
            .iter()
            .filter(|p| signal_positions.contains(p))
            .copied()
            .collect();
        let kept: Vec<usize> = indexed[k..].iter().map(|(p, _)| *p).collect();

        println!(
            "\n**Eviction K={}** (evict {} lowest-energy positions):",
            k, k
        );
        println!("- Evicted: {:?}", evicted);
        println!("- Kept: {:?}", kept);
        if evicted_signal.is_empty() {
            println!("- ✓ No signal positions evicted");
        } else {
            println!("- ✗ Signal positions evicted: {:?}", evicted_signal);
        }
    }
    println!("\nT9 PASS: eviction simulation printed");
}

// ═══════════════════════════════════════════════════════════════════
//  T11 — Summary
// ═══════════════════════════════════════════════════════════════════

/// Summary of all T5–T9 GOAT proofs for Plan 139.
#[test]
fn summary_goat_139_ega_examples() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  GOAT Plan 139 — EGA Example Proofs (T5–T11) Summary   ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║ T5  Gating improves signal-to-noise                    ║");
    println!("║ T5  Reversed energy worsens output                     ║");
    println!("║ T6  Gate monotonic with energy                         ║");
    println!("║ T6  High alpha → sharper (near-binary) gate            ║");
    println!("║ T6  Tau shifts threshold                               ║");
    println!("║ T7  Eviction removes low-energy positions first        ║");
    println!("║ T7  Eviction preserves/improves attention quality      ║");
    println!("║ T8  Combined pipeline invariants                       ║");
    println!("║ T9  Energy profile table + eviction simulation         ║");
    println!("║ T10 (documentation in benchmark file — separate task)  ║");
    println!("║ T11 Integration: this test suite = integration proof   ║");
    println!("╚══════════════════════════════════════════════════════════╝");
}
