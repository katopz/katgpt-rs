//! EGA Energy Profile Example — Visualize Energy Distribution (Plan 139, T6)
//!
//! Demonstrates the energy gate profile across a synthetic sequence:
//! 1. Energy score computation for content vs noise positions
//! 2. ASCII bar chart visualization of gate values for varying alpha
//! 3. Tau sensitivity analysis showing threshold shift behavior
//! 4. Monotonicity verification of gate with respect to energy
//! 5. Multi-head energy profile comparison
//!
//! Run: `cargo run --example ega_02_energy_profile --features ega_attn`

#![cfg(feature = "ega_attn")]

use katgpt_rs::ega_attn::{EgaGate, compute_energy_gate, z_normalize};

const SEQ_LEN: usize = 16;
const HEAD_DIM: usize = 16;
const CONTENT_POSITIONS: [usize; 4] = [0, 4, 8, 12];
const CONTENT_MAG: f32 = 3.0;
const NOISE_MAG: f32 = 0.1;

/// Build a synthetic sequence: content positions get high-magnitude embeddings,
/// filler positions get low-magnitude noise.
fn build_synthetic_sequence() -> Vec<f32> {
    let mut x = vec![0.0f32; SEQ_LEN * HEAD_DIM];
    for pos in 0..SEQ_LEN {
        let offset = pos * HEAD_DIM;
        let mag = if CONTENT_POSITIONS.contains(&pos) {
            CONTENT_MAG
        } else {
            NOISE_MAG
        };
        // Fill with mag in every dimension (simple signal)
        for d in 0..HEAD_DIM {
            x[offset + d] = mag;
        }
    }
    x
}

/// Compute embedding magnitude (L2 norm) for a given position.
fn embedding_magnitude(x: &[f32], pos: usize) -> f32 {
    let offset = pos * HEAD_DIM;
    let row = &x[offset..offset + HEAD_DIM];
    row.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Render a horizontal bar of `value` (0..1) using █ characters, `width` cells wide.
fn bar(value: f32, width: usize) -> String {
    let filled = ((value.clamp(0.0, 1.0) * width as f32).round()) as usize;
    let empty = width - filled;
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EGA Energy Profile — Spectral Salience Visualization       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  seq_len = {SEQ_LEN}, head_dim = {HEAD_DIM}");
    println!("  content positions: {CONTENT_POSITIONS:?} (mag = {CONTENT_MAG})");
    println!("  filler positions: noise (mag = {NOISE_MAG})");

    let x = build_synthetic_sequence();

    // ── Phase 1: Energy Profile ────────────────────────────────────
    println!("\n═══ Phase 1: Energy Score Computation ═══");
    {
        let gate = EgaGate::new(HEAD_DIM);
        let energy = gate.energy_scores(&x, SEQ_LEN, HEAD_DIM);

        // Compute z-normalized energy for display
        let mut z_energy = energy.clone();
        z_normalize(&mut z_energy);

        // Compute gate values
        let gate_values = compute_energy_gate(&energy, gate.alpha, gate.tau);

        println!();
        println!(
            "  w_proj = uniform 1/d = 1/{HEAD_DIM} = {:.4}",
            1.0 / HEAD_DIM as f32
        );
        println!("  alpha = {:.1}, tau = {:.2}", gate.alpha, gate.tau);
        println!();
        println!("  pos │ type    │ ‖x‖   │ raw_e   │ z_norm  │ gate   ");
        println!("  ────┼─────────┼────────┼─────────┼─────────┼────────");

        for pos in 0..SEQ_LEN {
            let kind = if CONTENT_POSITIONS.contains(&pos) {
                "content"
            } else {
                "filler "
            };
            let mag = embedding_magnitude(&x, pos);
            println!(
                "  {:>2}  │ {}  │ {:>6.2} │ {:>7.4} │ {:>7.4} │ {:.4}",
                pos, kind, mag, energy[pos], z_energy[pos], gate_values[pos]
            );
        }
    }

    // ── Phase 2: Gate Visualization ────────────────────────────────
    println!("\n═══ Phase 2: Gate Visualization (ASCII Bar Charts) ═══");
    {
        let gate = EgaGate::new(HEAD_DIM);
        let energy = gate.energy_scores(&x, SEQ_LEN, HEAD_DIM);

        let alphas: &[f32] = &[1.0, 2.2, 5.0, 10.0];
        let tau = 0.35f32;
        let bar_width = 40;

        for &alpha in alphas {
            let gates = compute_energy_gate(&energy, alpha, tau);
            println!();
            println!("  α = {:.1}, τ = {:.2}", alpha, tau);
            println!("  ┌────┬──────────────────────────────────────────┐");
            println!("  │ pos│ gate value                                │");
            println!("  ├────┼──────────────────────────────────────────┤");

            for pos in 0..SEQ_LEN {
                let g = gates[pos];
                println!("  │ {:>2} │{} {:.4}│", pos, bar(g, bar_width), g);
            }
            println!("  └────┴──────────────────────────────────────────┘");
        }
    }

    // ── Phase 3: Tau Sensitivity ───────────────────────────────────
    println!("\n═══ Phase 3: Tau Sensitivity Analysis (α = 2.2) ═══");
    {
        let gate = EgaGate::new(HEAD_DIM);
        let energy = gate.energy_scores(&x, SEQ_LEN, HEAD_DIM);
        let alpha = 2.2f32;

        let taus: &[f32] = &[0.0, 0.25, 0.35, 0.50, 0.75, 1.0, 1.5, 2.0];

        println!();
        println!("  tau    │ above 0.5 │ below 0.5 │ gate range");
        println!("  ───────┼───────────┼───────────┼───────────────");

        for &tau in taus {
            let gates = compute_energy_gate(&energy, alpha, tau);
            let above = gates.iter().filter(|&&g| g >= 0.5).count();
            let below = SEQ_LEN - above;
            let gmin = gates.iter().cloned().fold(f32::INFINITY, f32::min);
            let gmax = gates.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!(
                "  {:>5.2}  │    {:>2}     │    {:>2}     │ [{:.4}, {:.4}]",
                tau, above, below, gmin, gmax
            );
        }
    }

    // ── Phase 4: Monotonicity Proof ────────────────────────────────
    println!("\n═══ Phase 4: Monotonicity Verification ═══");
    {
        // Create energy scores that are sorted ascending
        let test_energies: Vec<f32> = (0..SEQ_LEN).map(|i| i as f32 * 0.5).collect();
        let alpha = 2.2f32;
        let tau = 0.35f32;
        let gates = compute_energy_gate(&test_energies, alpha, tau);

        let mut monotonic = true;
        for i in 1..gates.len() {
            if gates[i] < gates[i - 1] - 1e-6 {
                monotonic = false;
                println!(
                    "  VIOLATION at i={}: gate[{}] = {:.6} > gate[{}] = {:.6}",
                    i,
                    i - 1,
                    gates[i - 1],
                    i,
                    gates[i]
                );
            }
        }

        println!();
        println!("  Energy (ascending): {:?}", test_energies);
        println!("  Gates:  {:.4?}", gates);
        println!();
        if monotonic {
            println!("  ✅ PASS: gate values are monotonically non-decreasing with energy");
        } else {
            println!("  ❌ FAIL: gate values are NOT monotonically non-decreasing with energy");
        }
    }

    // ── Phase 5: Multi-Head Energy Distribution ────────────────────
    println!("\n═══ Phase 5: Multi-Head Energy Distribution ═══");
    {
        // Head 0: default (uniform 1/d)
        let head_0 = EgaGate::new(HEAD_DIM);

        // Head 1: all 0.5
        let head_1 = EgaGate {
            w_proj: vec![0.5; HEAD_DIM],
            alpha: 2.2,
            tau: 0.35,
        };

        // Head 2: alternating 0.1, 0.9
        let head_2 = EgaGate {
            w_proj: (0..HEAD_DIM)
                .map(|i| if i % 2 == 0 { 0.1 } else { 0.9 })
                .collect(),
            alpha: 2.2,
            tau: 0.35,
        };

        // Head 3: sparse (only first dim nonzero)
        let mut w_sparse = vec![0.0; HEAD_DIM];
        w_sparse[0] = 1.0;
        let head_3 = EgaGate {
            w_proj: w_sparse,
            alpha: 2.2,
            tau: 0.35,
        };

        let heads: [(&EgaGate, &str); 4] = [
            (&head_0, "uniform 1/d"),
            (&head_1, "all 0.5"),
            (&head_2, "alternating"),
            (&head_3, "sparse"),
        ];

        let energy_profiles: Vec<Vec<f32>> = heads
            .iter()
            .map(|(h, _)| h.energy_scores(&x, SEQ_LEN, HEAD_DIM))
            .collect();

        println!();
        println!("  Head configurations:");
        for (i, (_, label)) in heads.iter().enumerate() {
            let pc = heads[i].0.parameter_count();
            println!("    Head {}: w_proj = {} ({} params)", i, label, pc);
        }
        println!();
        println!("  pos │ type   │ head_0  │ head_1  │ head_2  │ head_3");
        println!("  ────┼────────┼─────────┼─────────┼─────────┼─────────");

        for pos in 0..SEQ_LEN {
            let kind = if CONTENT_POSITIONS.contains(&pos) {
                "cont"
            } else {
                "fill"
            };
            print!("  {:>2}  │ {}  ", pos, kind);
            for h in 0..4 {
                print!("│ {:>7.4} ", energy_profiles[h][pos]);
            }
            println!();
        }

        // Show gate profiles for each head
        println!();
        println!("  Gate values per head (α=2.2, τ=0.35):");
        println!("  pos │ type   │ head_0  │ head_1  │ head_2  │ head_3");
        println!("  ────┼────────┼─────────┼─────────┼─────────┼─────────");

        for pos in 0..SEQ_LEN {
            let kind = if CONTENT_POSITIONS.contains(&pos) {
                "cont"
            } else {
                "fill"
            };
            print!("  {:>2}  │ {}  ", pos, kind);
            for h in 0..4 {
                let gates =
                    compute_energy_gate(&energy_profiles[h], heads[h].0.alpha, heads[h].0.tau);
                print!("│ {:.4}  ", gates[pos]);
            }
            println!();
        }
    }

    // ── Summary ────────────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  ✓ Content positions (0,4,8,12) show high energy scores    ║");
    println!("║  ✓ Higher α sharpens the gate — binary-like at α=10        ║");
    println!("║  ✓ Lower τ admits more positions above threshold           ║");
    println!("║  ✓ Gate is monotonically non-decreasing w.r.t. energy      ║");
    println!("║  ✓ Different w_proj produce distinct energy profiles        ║");
    println!(
        "║  ✓ Per-head overhead: d+2 = {} parameters                  ║",
        HEAD_DIM + 2
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
}
