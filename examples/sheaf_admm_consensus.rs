//! Sheaf-ADMM Consensus Demo — Plan 407 Phase 3 T3.4.
//!
//! 16 agents on a 4×4 grid reach consensus via Sheaf-ADMM. Each agent holds
//! three state variables (primal x_i, consensus z_i, dual u_i) and runs ADMM
//! over the grid's cellular sheaf with identity restriction maps. The demo
//! shows:
//! - The primal proposals converging to the zone consensus.
//! - The dual u_i vectors starting at zero and growing with disagreement.
//! - The edge disagreement ‖F_i x_i − F_j x_j‖ decaying to ~0.
//!
//! Run with:
//! ```bash
//! cargo run --example sheaf_admm_consensus --features sheaf_admm --release
//! ```

#![cfg(feature = "sheaf_admm")]

use katgpt_core::dec::{
    AdmmScratch, CellComplex, CochainField, LocalObjective, SheafMaps, sheaf_admm_step,
};

fn main() {
    // 16 agents on a 4×4 grid.
    let cx = CellComplex::grid_2d(4, 4);
    let n = cx.n_vertices(); // 16
    let d_v = 4; // 4-dim state per agent (e.g. 4 HLA affect scalars)
    let d_e = 4; // identity maps: agree on all 4 dims

    // Identity restriction maps — homogeneous consensus (agree on everything).
    let maps = SheafMaps::identity(&cx, d_v, d_e);

    // Each agent has a DIFFERENT local objective (different q_i), so their
    // primal proposals start divergent. The sheaf diffusion pulls them toward
    // consensus. Use diag_q = 1 (well-conditioned), q_i = random per agent.
    let diag_q = vec![1.0f32; n * d_v];
    let mut q = vec![0.0f32; n * d_v];
    // Deterministic splitmix64 PRNG for reproducibility (no `rand` dep).
    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_BABE;
    for rand_f_slot in q.iter_mut().take(n * d_v) {
        // splitmix64
        rng_state = rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let rand_f = (z as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0; // [-1, 1]
        *rand_f_slot = rand_f;
    }
    let objective = LocalObjective::DiagonalQuadratic { diag_q, q };

    // Initialize: primal x = 0, consensus z = 0, dual u = 0.
    let mut primal_x = CochainField::zeros(0, n, d_v);
    let mut consensus_z = CochainField::zeros(0, n, d_v);
    let mut dual_u = CochainField::zeros(0, n, d_v);
    let mut scratch = AdmmScratch::new(&cx, d_v, d_e);

    // ADMM hyperparameters. T (diffusion_steps) controls how close each
    // z-update gets to the true harmonic projection; with finite T the
    // z-update is an inexact projection and the fixed point retains a
    // small non-harmonic residual. T=50 drives the 4×4 grid's slowest
    // non-harmonic mode down enough to clear the 1e-3 consensus bar in
    // K=50 iterations.
    let rho = 1.0;
    let eta = 0.25;
    let diffusion_steps = 50;
    let k_iters = 50;

    println!("=== Sheaf-ADMM Consensus Demo (Plan 407 T3.4) ===");
    println!("{n} agents on a 4×4 grid, d_v={d_v}, d_e={d_e} (identity maps)");
    println!(
        "ADMM: rho={rho}, eta={eta}, diffusion_steps={diffusion_steps}, iterations={k_iters}\n"
    );

    // Print initial state.
    print_iteration(0, &primal_x, &consensus_z, &dual_u, &cx, d_v);

    // Run K iterations, printing every 10.
    for k in 1..=k_iters {
        sheaf_admm_step(
            &cx,
            &maps,
            &mut primal_x,
            &mut consensus_z,
            &mut dual_u,
            &objective,
            rho,
            eta,
            diffusion_steps,
            &mut scratch,
        );
        if k % 10 == 0 || k == k_iters {
            print_iteration(k, &primal_x, &consensus_z, &dual_u, &cx, d_v);
        }
    }

    // Final summary: max edge disagreement (the consensus metric).
    let max_disagree = max_edge_disagreement(&cx, &primal_x, d_v);
    println!("\n=== Final: max edge disagreement ‖F_i x_i − F_j x_j‖_∞ = {max_disagree:.2e} ===");
    if max_disagree < 1e-3 {
        println!("✅ Consensus reached (agents aligned on all {d_e} dims).");
    } else {
        println!("⚠ Consensus not yet reached (run more iterations).");
    }
}

fn print_iteration(
    k: usize,
    primal_x: &CochainField,
    consensus_z: &CochainField,
    dual_u: &CochainField,
    cx: &CellComplex,
    d_v: usize,
) {
    let max_disagree = max_edge_disagreement(cx, primal_x, d_v);
    let dual_norm: f32 = dual_u.data.iter().map(|v| v * v).sum::<f32>().sqrt();
    let primal_spread: f32 = {
        // L2 norm of (primal_x - consensus_z) — how far locals are from consensus.
        primal_x
            .data
            .iter()
            .zip(consensus_z.data.iter())
            .map(|(x, z)| {
                let d = x - z;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    };
    println!(
        "iter {k:>3}: max_edge_disagree={max_disagree:.4e}, ‖u‖₂={dual_norm:.4e}, ‖x−z‖₂={primal_spread:.4e}"
    );
}

fn max_edge_disagreement(cx: &CellComplex, field: &CochainField, d_v: usize) -> f32 {
    let entries = cx.boundary_entries(0);
    let mut max_d = 0.0f32;
    for pair in entries.as_chunks::<2>().0 {
        let v_tail = pair[0].0;
        let v_head = pair[1].0;
        let x_tail = &field.data[v_tail * d_v..(v_tail + 1) * d_v];
        let x_head = &field.data[v_head * d_v..(v_head + 1) * d_v];
        for d in 0..d_v {
            let diff = (x_tail[d] - x_head[d]).abs();
            if diff > max_d {
                max_d = diff;
            }
        }
    }
    max_d
}
