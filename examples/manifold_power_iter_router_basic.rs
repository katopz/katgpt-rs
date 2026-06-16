//! Manifold Power Iteration MoE Router — before/after demo (Plan 279).
//!
//! Run:
//! ```bash
//! cargo run --example manifold_power_iter_router_basic \
//!            --features manifold_power_iter_router --release
//! ```
//!
//! Demonstrates the paper's λ alignment + MaxVio gains (arXiv:2606.12397)
//! on a synthetic MoE pool: N=8 experts, D=256. Shows:
//! - λ_alignment before → after (paper target: 0.27 → 0.66 shape)
//! - maxvio before → after (paper target: 1.13 → 0.96 shape)
//! - timing (target: sub-ms for game-scale pool)
//! - sigmoid top-k gating on a sample token

#![cfg(feature = "manifold_power_iter_router")]

use katgpt_rs::manifold_power_iter_router::{
    compute_diagnostics, compute_expert_gram_into, gate_sigmoid_topk, manifold_power_iter_router,
    MpiRouterConfig,
};
use katgpt_rs::spectral_retract::PowerRetractScratch;
use std::time::Instant;

/// Deterministic xorshift64 PRNG.
fn seeded_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        v.push(((s & 0xFFFF) as f32 / 0x8000 as f32) - 1.0);
    }
    v
}

fn main() {
    let n = 8usize; // game-scale NPC expert pool
    let d = 256usize; // typical LoRA D
    let cfg = MpiRouterConfig::default(); // iters=1, c_prime=1.0, beta=1.0

    println!("=== Manifold Power Iteration MoE Router (Plan 279) ===");
    println!(
        "config: N={}, D={}, iters={}, c_prime={:.3}, beta={:.3}",
        n, d, cfg.iters, cfg.c_prime, cfg.beta_sigmoid
    );
    println!();

    // Build router R and per-expert gate weights W_g[i].
    let mut r = seeded_vec(42, n * d);
    let w_g: Vec<Vec<f32>> = (0..n).map(|i| seeded_vec(100 + i as u64, d * d)).collect();

    // Phase A: build expert grams (warm tier — once per snapshot).
    let t_gram = Instant::now();
    let grams: Vec<Vec<f32>> = w_g
        .iter()
        .map(|w| {
            let mut g = vec![0.0f32; d * d];
            compute_expert_gram_into(w, d, &mut g);
            g
        })
        .collect();
    let dt_gram = t_gram.elapsed();
    println!("Gram build: {:?} ({} experts × {}×{})", dt_gram, n, d, d);
    println!();

    // "Before" diagnostics (vanilla unconditioned router).
    let target_norm = cfg.c_prime / (n as f32).sqrt();
    let grams_ref: Vec<&[f32]> = grams.iter().map(|g| g.as_slice()).collect();
    let (lambda_before, maxvio_before) = compute_diagnostics(&r, &grams_ref, n, d, target_norm);

    // Phase B: MPI recondition (paper Eq. 4–5).
    let t_mpi = Instant::now();
    let mut scratch = PowerRetractScratch::new(d);
    let res = manifold_power_iter_router(
        &mut r,
        &grams_ref,
        n,
        d,
        cfg.c_prime,
        cfg.iters,
        &mut scratch,
    );
    let dt_mpi = t_mpi.elapsed();
    println!("MPI recondition: {:?}", dt_mpi);
    println!();

    println!("┌──────────────────────────────────────────────────┐");
    println!("│ λ alignment  before={:.4}   after={:.4}   │", lambda_before, res.lambda_alignment);
    println!("│ MaxVio       before={:.4}   after={:.4}   │", maxvio_before, res.maxvio);
    println!("└──────────────────────────────────────────────────┘");
    println!();
    println!(
        "paper targets (§1.4): λ 0.27→0.66 shape, MaxVio 1.13→0.96 shape"
    );
    println!(
        "this run: λ gain = {:.2}×, MaxVio reduction = {:.1}%",
        res.lambda_alignment / lambda_before.abs().max(1e-6),
        100.0 * (1.0 - res.maxvio / maxvio_before.abs().max(1e-6))
    );
    println!();

    // Phase C: sigmoid top-k gating on a sample token (NEVER softmax).
    let x = seeded_vec(7, d);
    let mut scores = vec![0.0f32; n];
    let t_gate = Instant::now();
    let topk = gate_sigmoid_topk(&x, &r, n, d, cfg.beta_sigmoid, 3, &mut scores);
    let dt_gate = t_gate.elapsed();
    println!("Sigmoid top-3 gate: {:?} (NEVER softmax — G7)", dt_gate);
    println!("  token dot-prod scores: {:?}", &scores);
    println!("  top-3 expert indices:  {:?}", topk);
    println!();
    println!("total snapshot-swap cost: {:?}", dt_gram + dt_mpi);
    println!("per-token overhead: 0 (router is precomputed — paper §4.2)");
}
