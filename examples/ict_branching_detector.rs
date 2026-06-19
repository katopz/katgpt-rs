//! Plan 294 T8.3 — `BranchingDetector` on synthetic NPC-decision trajectories.
//!
//! Simulates K=8 candidate action distributions per tick (as if sampled from
//! a policy by MCTS/CLR/action-proposal), runs the detector, and shows the
//! per-trajectory branching mask + population β. Run with:
//!
//! ```text
//! cargo run --example ict_branching_detector --features ict_branching
//! ```

#![cfg(feature = "ict_branching")]

use katgpt_core::ict::detector::BranchingDetector;

fn main() {
    println!("=== ICT BranchingDetector — Synthetic K=8 trajectories ===\n");

    let k = 8_usize;
    let action_dim = 6_usize;
    let mut det = BranchingDetector::new(k, action_dim, 0.10, 0.05);

    // Synthetic tick 1: most trajectories agree, one diverges.
    // The divergent trajectory (k=0) should be flagged by top-10%.
    let mut tick1: Vec<Vec<f32>> = Vec::with_capacity(k);
    tick1.push(vec![0.7, 0.1, 0.1, 0.05, 0.03, 0.02]); // divergent
    for _ in 1..k {
        tick1.push(vec![0.25, 0.25, 0.2, 0.15, 0.1, 0.05]); // consensus
    }
    let trajs1: Vec<&[f32]> = tick1.iter().map(|v| v.as_slice()).collect();
    let r1 = det.observe_and_detect(&trajs1);
    println!("Tick 1 — one divergent trajectory (idx 0):");
    println!("  mask              = {:?}", r1.mask);
    println!("  uniqueness scores = {:?}", r1.uniqueness_scores);
    println!("  population β      = {:.4}   ema_β = {:.4}", r1.beta_per_step[0], det.ema_beta);
    println!("  branching_count   = {}", r1.branching_count());

    // Synthetic tick 2: all trajectories identical → no branching.
    let identical = vec![0.4_f32, 0.3, 0.15, 0.1, 0.04, 0.01];
    let tick2: Vec<&[f32]> = (0..k).map(|_| identical.as_slice()).collect();
    let r2 = det.observe_and_detect(&tick2);
    println!("\nTick 2 — all identical (no divergence):");
    println!("  mask              = {:?}", r2.mask);
    println!("  uniqueness scores = {:?}", r2.uniqueness_scores);
    println!("  branching_count   = {} (top-k% always flags ≥ 1 by design)", r2.branching_count());

    // Synthetic tick 3: every trajectory different — top-10% picks the single
    // most divergent one.
    let mut tick3: Vec<Vec<f32>> = Vec::with_capacity(k);
    for i in 0..k {
        let mut p = vec![0.1_f32; action_dim];
        p[i % action_dim] = 0.55;
        tick3.push(p);
    }
    let trajs3: Vec<&[f32]> = tick3.iter().map(|v| v.as_slice()).collect();
    let r3 = det.observe_and_detect(&trajs3);
    println!("\nTick 3 — each trajectory peaks at a different action:");
    println!("  mask              = {:?}", r3.mask);
    println!("  uniqueness scores = {:?}", r3.uniqueness_scores);
    println!("  population β      = {:.4}", r3.beta_per_step[0]);
    println!("  branching_count   = {}", r3.branching_count());

    println!("\nDone. See `ict_paper_figure_1a` for the H_1 vs β bifurcation proof.");
}
