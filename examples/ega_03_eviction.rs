//! EGA KV Eviction Example — Energy-Threshold Cache Compression (Plan 139, T7)
//!
//! Demonstrates using EGA energy scores as a KV cache eviction signal:
//! 1. Synthetic sequence with known content/noise positions
//! 2. Energy-ranked eviction: lowest-energy positions evicted first
//! 3. Fixed-K eviction comparison at multiple levels
//! 4. Energy threshold (τ) based eviction analysis
//! 5. Quality vs eviction fraction trade-off curve
//!
//! Run: `cargo run --example ega_03_eviction --features ega_attn`

#![cfg(feature = "ega_attn")]

use katgpt_rs::ega_attn::{EgaGate, compute_energy_gate};

// ── Configuration ──────────────────────────────────────────────
const SEQ_LEN: usize = 32;
const HEAD_DIM: usize = 16;
const CONTENT_MAG: f32 = 3.0;
const NOISE_MAG: f32 = 0.1;

/// Indices of content (high-signal) positions.
const CONTENT_POSITIONS: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];

// ── Helper functions ───────────────────────────────────────────

fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn matmul_attn_values(attn: &[f32], values: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
    // attn: [seq_len × seq_len] (single query row used — last row for autoregressive)
    // values: [seq_len × dim]
    // Returns: [dim] — attention-weighted sum of values using the last query row.
    let query_row = &attn[(seq_len - 1) * seq_len..seq_len * seq_len];
    let mut out = vec![0.0f32; dim];
    for j in 0..seq_len {
        let w = query_row[j];
        let v_off = j * dim;
        for d in 0..dim {
            out[d] += w * values[v_off + d];
        }
    }
    out
}

fn mean_rows(data: &[f32], row_indices: &[usize], dim: usize) -> Vec<f32> {
    if row_indices.is_empty() {
        return vec![0.0; dim];
    }
    let mut out = vec![0.0f32; dim];
    for &i in row_indices {
        let off = i * dim;
        for d in 0..dim {
            out[d] += data[off + d];
        }
    }
    let scale = 1.0 / row_indices.len() as f32;
    for x in out.iter_mut() {
        *x *= scale;
    }
    out
}

/// Returns true if position is a content position.
fn is_content(pos: usize) -> bool {
    CONTENT_POSITIONS.contains(&pos)
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EGA KV Eviction — Energy-Threshold Cache Compression       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("seq_len={SEQ_LEN}, head_dim={HEAD_DIM}");
    println!("content_mag={CONTENT_MAG}, noise_mag={NOISE_MAG}");
    println!("content_positions={:?}", CONTENT_POSITIONS.as_slice());

    // ── Phase 1: Sequence Setup ────────────────────────────────────
    println!("\n═══ Phase 1: Sequence Setup ═══");

    // Simple deterministic PRNG for reproducibility.
    let mut rng_state: u64 = 42;
    let mut next_f32 = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0 // [-1, 1]
    };

    // Build embeddings: content positions get magnitude CONTENT_MAG, noise get NOISE_MAG.
    let mut embeddings = vec![0.0f32; SEQ_LEN * HEAD_DIM];
    let mut content_indices = Vec::new();
    let mut noise_indices = Vec::new();

    for pos in 0..SEQ_LEN {
        let mag = if is_content(pos) {
            CONTENT_MAG
        } else {
            NOISE_MAG
        };
        let off = pos * HEAD_DIM;
        for d in 0..HEAD_DIM {
            embeddings[off + d] = next_f32() * mag;
        }
        if is_content(pos) {
            content_indices.push(pos);
        } else {
            noise_indices.push(pos);
        }
    }

    // Print position table.
    println!("┌──────────┬──────────┬────────────┐");
    println!("│ Position │ Type     │ Magnitude  │");
    println!("├──────────┼──────────┼────────────┤");
    for pos in 0..SEQ_LEN {
        let kind = if is_content(pos) {
            "CONTENT"
        } else {
            "noise  "
        };
        let mag = if is_content(pos) {
            CONTENT_MAG
        } else {
            NOISE_MAG
        };
        println!("│ {pos:>8} │ {kind}  │ {mag:>10.1} │");
    }
    println!("└──────────┴──────────┴────────────┘");
    println!(
        "Total: {} content, {} noise",
        content_indices.len(),
        noise_indices.len()
    );

    // ── Phase 2: Energy Ranking ────────────────────────────────────
    println!("\n═══ Phase 2: Energy Ranking ═══");

    let gate = EgaGate::new(HEAD_DIM);
    let energy = gate.energy_scores(&embeddings, SEQ_LEN, HEAD_DIM);

    // Sort positions by energy ascending.
    let mut ranked: Vec<(usize, f32)> = energy.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("┌──────┬──────────┬────────────┬──────────┐");
    println!("│ Rank │ Position │ Energy     │ Type     │");
    println!("├──────┼──────────┼────────────┼──────────┤");
    for (rank, (pos, e)) in ranked.iter().enumerate() {
        let kind = if is_content(*pos) {
            "CONTENT"
        } else {
            "noise  "
        };
        println!("│ {rank:>4} │ {pos:>8} │ {e:>10.4} │ {kind}  │");
    }
    println!("└──────┴──────────┴────────────┴──────────┘");

    // Verify: content positions should rank higher (have higher energy).
    let max_noise_rank = ranked
        .iter()
        .enumerate()
        .filter(|(_, (pos, _))| !is_content(*pos))
        .map(|(rank, _)| rank)
        .max()
        .unwrap();
    let min_content_rank = ranked
        .iter()
        .enumerate()
        .filter(|(_, (pos, _))| is_content(*pos))
        .map(|(rank, _)| rank)
        .min()
        .unwrap();

    if min_content_rank > max_noise_rank {
        println!(
            "✓ All content positions rank above noise (min_content_rank={}, max_noise_rank={})",
            min_content_rank, max_noise_rank
        );
    } else {
        println!(
            "⚠ Content/noise overlap: min_content_rank={}, max_noise_rank={}",
            min_content_rank, max_noise_rank
        );
    }

    // Compute signal mean (ground truth for quality comparison).
    let signal_mean = mean_rows(&embeddings, &content_indices, HEAD_DIM);

    // ── Phase 3: Eviction at K=8,16,24 ────────────────────────────
    println!("\n═══ Phase 3: Fixed-K Eviction ═══");

    // Build a simple uniform attention matrix for the last query row.
    let mut attn_weights = vec![1.0 / SEQ_LEN as f32; SEQ_LEN * SEQ_LEN];
    // Make it slightly peaky towards content positions.
    for i in 0..SEQ_LEN {
        let row_off = i * SEQ_LEN;
        for &cp in &CONTENT_POSITIONS {
            attn_weights[row_off + cp] *= 2.0;
        }
        // Re-normalize row.
        let row_sum: f32 = attn_weights[row_off..row_off + SEQ_LEN].iter().sum();
        for j in 0..SEQ_LEN {
            attn_weights[row_off + j] /= row_sum;
        }
    }

    // Baseline: no eviction.
    let baseline_output = matmul_attn_values(&attn_weights, &embeddings, SEQ_LEN, HEAD_DIM);
    let baseline_dist = l2_dist(&baseline_output, &signal_mean);
    println!("Baseline (no eviction): L2 to signal mean = {baseline_dist:.4}");

    for &k in &[8usize, 16, 24] {
        println!("\n── Evicting K={k} lowest-energy positions ──");

        // Positions to evict: the K lowest-energy.
        let evict_set: Vec<usize> = ranked.iter().take(k).map(|(pos, _)| *pos).collect();
        let keep_set: Vec<usize> = ranked.iter().skip(k).map(|(pos, _)| *pos).collect();

        let evicted_content: Vec<usize> = evict_set
            .iter()
            .copied()
            .filter(|p| is_content(*p))
            .collect();
        let evicted_noise: Vec<usize> = evict_set
            .iter()
            .copied()
            .filter(|p| !is_content(*p))
            .collect();

        println!("  Evicted positions: {:?}", evict_set);
        println!("  Kept positions:    {:?}", keep_set);
        println!(
            "  Content evicted: {} / {}",
            evicted_content.len(),
            content_indices.len()
        );
        println!(
            "  Noise evicted:   {} / {}",
            evicted_noise.len(),
            noise_indices.len()
        );

        // Compute attention output with evicted positions zeroed out + renormalized.
        let mut evicted_attn = attn_weights.clone();
        for i in 0..SEQ_LEN {
            let row_off = i * SEQ_LEN;
            for &ep in &evict_set {
                evicted_attn[row_off + ep] = 0.0;
            }
            // Renormalize row.
            let row_sum: f32 = evicted_attn[row_off..row_off + SEQ_LEN].iter().sum();
            if row_sum > 1e-8 {
                for j in 0..SEQ_LEN {
                    evicted_attn[row_off + j] /= row_sum;
                }
            }
        }

        let evicted_output = matmul_attn_values(&evicted_attn, &embeddings, SEQ_LEN, HEAD_DIM);
        let evicted_dist = l2_dist(&evicted_output, &signal_mean);

        let delta = evicted_dist - baseline_dist;
        let arrow = if delta < 0.0 {
            "↓"
        } else if delta > 0.0 {
            "↑"
        } else {
            "="
        };
        println!(
            "  L2 to signal mean: {evicted_dist:.4} (Δ={delta:+.4} {arrow} vs baseline {baseline_dist:.4})"
        );
    }

    // ── Phase 4: Energy Threshold Eviction ─────────────────────────
    println!("\n═══ Phase 4: Energy Threshold (τ) Eviction ═══");

    println!("Using EGA gate with α=2.2 to compute per-position gate values.");
    println!("A position is 'evictable' if its gate value < 0.5.\n");

    let alpha = 2.2;
    let tau_values = [0.0f32, 0.5, 1.0, 2.0];

    println!("┌──────┬───────────┬────────────────┬──────────────┬──────────────┐");
    println!("│  τ   │ #evicted  │ #signal_evicted│ #noise_evict │ evict_frac   │");
    println!("├──────┼───────────┼────────────────┼──────────────┼──────────────┤");
    for &tau in &tau_values {
        let gate_vals = compute_energy_gate(&energy, alpha, tau);

        let mut evicted = Vec::new();
        for (pos, &gv) in gate_vals.iter().enumerate() {
            if gv < 0.5 {
                evicted.push(pos);
            }
        }

        let signal_evicted = evicted.iter().filter(|&&p| is_content(p)).count();
        let noise_evicted = evicted.iter().filter(|&&p| !is_content(p)).count();
        let n_evicted = evicted.len();
        let frac = evicted.len() as f32 / SEQ_LEN as f32;

        let frac_pct = format!("{:.0}%", frac * 100.0);
        println!(
            "│ {tau:>4.1} │ {n_evicted:>9} │ {signal_evicted:>14} │ {noise_evicted:>12} │ {frac_pct:>12} │",
        );
    }
    println!("└──────┴───────────┴────────────────┴──────────────┴──────────────┘");

    // ── Phase 5: Quality vs Eviction Trade-off ─────────────────────
    println!("\n═══ Phase 5: Quality vs Eviction Trade-off ═══");

    let fractions: [f32; 4] = [0.0, 0.25, 0.50, 0.75];

    println!("┌───────────────┬──────────────────┬─────────────────┐");
    println!("│ evict_frac    │ L2_dist_to_signal│ signal_preserved │");
    println!("├───────────────┼──────────────────┼─────────────────┤");

    for &frac in &fractions {
        let k = (frac * SEQ_LEN as f32).round() as usize;

        let frac_pct = format!("{:.0}%", frac * 100.0);

        if k == 0 {
            let dist = l2_dist(&baseline_output, &signal_mean);
            let yes_str = "✓ yes";
            println!("│ {frac_pct:>13} │ {dist:>16.4} │ {yes_str:>15} │");
            continue;
        }

        let evict_set: Vec<usize> = ranked.iter().take(k).map(|(pos, _)| *pos).collect();

        // Check if any content was evicted.
        let signal_preserved = evict_set.iter().all(|&p| !is_content(p));

        // Compute quality.
        let mut evicted_attn = attn_weights.clone();
        for i in 0..SEQ_LEN {
            let row_off = i * SEQ_LEN;
            for &ep in &evict_set {
                evicted_attn[row_off + ep] = 0.0;
            }
            let row_sum: f32 = evicted_attn[row_off..row_off + SEQ_LEN].iter().sum();
            if row_sum > 1e-8 {
                for j in 0..SEQ_LEN {
                    evicted_attn[row_off + j] /= row_sum;
                }
            }
        }

        let evicted_output = matmul_attn_values(&evicted_attn, &embeddings, SEQ_LEN, HEAD_DIM);
        let dist = l2_dist(&evicted_output, &signal_mean);

        let preserved_str = if signal_preserved {
            "✓ yes"
        } else {
            "✗ no"
        };
        println!("│ {frac_pct:>13} │ {dist:>16.4} │ {preserved_str:>15} │");
    }
    println!("└───────────────┴──────────────────┴─────────────────┘");

    // ── Summary ────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  • Content positions (mag={CONTENT_MAG}) have higher energy than     ║");
    println!("║    noise positions (mag={NOISE_MAG}), enabling clean separation.    ║");
    println!("║  • Fixed-K eviction removes noise first, preserving signal. ║");
    println!("║  • Energy threshold (τ) controls eviction fraction; higher ║");
    println!("║    τ evicts more aggressively.                              ║");
    println!("║  • Quality degrades gracefully: evicting low-energy noise   ║");
    println!("║    improves or maintains output quality.                     ║");
    println!("║  • EGA energy scores provide a principled eviction signal   ║");
    println!("║    for KV cache compression without auxiliary models.       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
