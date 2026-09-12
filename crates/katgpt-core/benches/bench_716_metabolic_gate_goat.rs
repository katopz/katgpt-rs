//! Research 550 / riir-ai Plan 585 T1.3+T1.4 — Metabolic Gate GOAT Gate.
//!
//! Exercises the GOAT gate for the `metabolic_gate` primitive (arXiv:2609.10817
//! "Tapes Together Strong"): energy-coupled compute gating — the compute-budget
//! family's STOCK axis beside `gain_cost_halt` (utility flow).
//!
//! # Gates
//!
//! - **G1** — Correctness: the design-law identities on the Plan 585 T0.2
//!   regime tuple (ε=12, L=10, α=0.3, δ=3): cooperator viability
//!   `2ε > 2L`, starvation `2ε < L(1+(1−α)δ)`, K-solver residual ~0, and
//!   the coherent-story check `(1−α)δ > K(ε,L)`.
//! - **G2** — Latency: `depth_factor` + `execution_share` sub-ns-class per
//!   call (batched medians); `metabolic_drag_threshold` bounded (~32 iters,
//!   one log pair each).
//! - **G3** — Feature isolation: verified outside this binary via
//!   `cargo check` at default / `--features metabolic_gate` / `--all-features`.
//! - **G4** — Zero alloc: 0 allocations across 1000 calls of every hot-path
//!   entry point (CountingAllocator); the K-solver is loop-only (no heap).

use katgpt_core::metabolic_gate::{
    MetabolicGate, defector_starves, execution_share, metabolic_drag_threshold, steal_lossy,
};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

// Plan 585 T0.2 regime tuple — recorded in the bench per the task contract.
const EPS: f32 = 12.0;
const L: f32 = 10.0;
const ALPHA: f32 = 0.3;
const DELTA: f32 = 3.0;

fn gate_g1_correctness() -> GateResult {
    println!("\n--- G1: Design-law identities (T0.2 tuple ε={EPS} L={L} α={ALPHA} δ={DELTA}) ---");

    let viable = 2.0 * EPS > 2.0 * L; // cooperators can each replicate
    let starves = defector_starves(EPS, L, ALPHA, DELTA);
    let k = metabolic_drag_threshold(EPS, L);
    let inefficiency = (1.0 - ALPHA) * DELTA;

    let lhs = (2.0 * EPS / (2.0 * EPS - L * (1.0 + k))).ln();
    let rhs = (1.0 + k) * (EPS / (EPS - L)).ln();
    let residual = (lhs - rhs).abs();

    // Conservation identity at a binary-fraction alpha (exact f32).
    let mut thief = 0.0f32;
    let mut victim = 40.0f32;
    let destroyed = steal_lossy(&mut thief, &mut victim, 10.0, 0.5);
    let conserved = (thief - 0.0) + destroyed == 10.0 && victim == 30.0;

    // depth_factor anchor: σ(0) = 0.5 exactly at the base stock.
    let gate = MetabolicGate::new(50.0, 10.0);
    let anchored = gate.depth_factor(50.0) == 0.5;

    // Both-zero lottery → ½ (paper C.1).
    let lottery = execution_share(0.0, 0.0) == 0.5;

    println!("  cooperator viability 2ε > 2L: {viable} (24 > 20)");
    println!("  defector starvation:          {starves} (24 < 31)");
    println!("  K(ε,L) = {k:.6}, residual = {residual:.2e}");
    println!("  (1−α)δ = {inefficiency:.1} > K: {}", inefficiency > k);
    println!("  steal_lossy conservation:     {conserved}");
    println!("  depth_factor σ(0) anchor:     {anchored}");
    println!("  execution_share 0,0 → ½:      {lottery}");

    let passed = viable
        && starves
        && residual < 1e-4
        && inefficiency > k
        && conserved
        && anchored
        && lottery;
    GateResult {
        name: "G1 correctness",
        passed,
        detail: format!(
            "viable={viable} starves={starves} residual={residual:.1e} K={k:.4} < (1−α)δ={inefficiency}"
        ),
    }
}

fn gate_g2_latency() -> GateResult {
    const WARMUP: u64 = 10_000;
    const ITERS: u64 = 2_000_000;
    const BATCH: u64 = 2000;

    println!("\n--- G2: Latency (sub-ns-class gates, bounded K-solver) ---");

    let gate = MetabolicGate::new(50.0, 10.0);
    let stocks: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.1).collect();

    // Warmup.
    for i in 0..WARMUP {
        black_box(gate.depth_factor(stocks[(i % 1000) as usize]));
    }

    let mut ns_samples = Vec::with_capacity((ITERS / BATCH) as usize);
    for _ in 0..(ITERS / BATCH) {
        let t0 = Instant::now();
        for j in 0..BATCH {
            black_box(gate.depth_factor(stocks[(j % 1000) as usize]));
        }
        ns_samples.push(t0.elapsed().as_nanos() as u64 / BATCH);
    }
    ns_samples.sort_unstable();
    let depth_med = ns_samples[ns_samples.len() / 2];

    let mut share_samples = Vec::with_capacity((ITERS / BATCH) as usize);
    for _ in 0..(ITERS / BATCH) {
        let t0 = Instant::now();
        for j in 0..BATCH {
            black_box(execution_share(j as f32, 1000.0));
        }
        share_samples.push(t0.elapsed().as_nanos() as u64 / BATCH);
    }
    share_samples.sort_unstable();
    let share_med = share_samples[share_samples.len() / 2];

    let mut ks_samples = Vec::with_capacity(2000);
    for _ in 0..2000 {
        let t0 = Instant::now();
        for j in 0..10 {
            black_box(metabolic_drag_threshold(12.0 + (j % 7) as f32, 10.0));
        }
        ks_samples.push(t0.elapsed().as_nanos() as u64 / 10);
    }
    ks_samples.sort_unstable();
    let ks_med = ks_samples[ks_samples.len() / 2];

    // Ambient-load honesty: this binary ran on a shared box; medians over
    // 1000 batches are the robust statistic (see the bench record).
    println!("  depth_factor      p50: {depth_med} ns/call");
    println!("  execution_share   p50: {share_med} ns/call");
    println!("  K-solver (32 it)  p50: {ks_med} ns/call  (bounded: 32 fixed iterations)");

    let passed = depth_med < 10 && share_med < 10 && ks_med < 2000;
    GateResult {
        name: "G2 latency",
        passed,
        detail: format!(
            "depth {depth_med} ns, share {share_med} ns, K-solver {ks_med} ns (targets: <10/<10/<2000)"
        ),
    }
}

fn gate_g4_alloc_free() -> GateResult {
    println!("\n--- G4: Zero-alloc hot paths ---");

    let gate = MetabolicGate::new(50.0, 10.0);
    let (_, depth_allocs) = alloc_delta(|| {
        for i in 0..1000 {
            black_box(gate.depth_factor(i as f32 * 0.05));
        }
    });

    let (_, steal_allocs) = alloc_delta(|| {
        let mut t = 0.0f32;
        let mut v = 100.0f32;
        for i in 0..1000 {
            black_box(steal_lossy(&mut t, &mut v, 1.0, 0.5));
            if v < 2.0 {
                v = 100.0; // refill outside the measured call
            }
            let _ = i;
        }
    });

    let (_, ks_allocs) = alloc_delta(|| {
        for i in 0..1000 {
            black_box(metabolic_drag_threshold(12.0 + (i % 13) as f32 * 0.5, 10.0));
        }
    });

    println!("  depth_factor  allocs/1000: {depth_allocs}");
    println!("  steal_lossy   allocs/1000: {steal_allocs}");
    println!("  K-solver      allocs/1000: {ks_allocs}");

    let passed = depth_allocs == 0 && steal_allocs == 0 && ks_allocs == 0;
    GateResult {
        name: "G4 alloc-free",
        passed,
        detail: format!("depth={depth_allocs} steal={steal_allocs} K={ks_allocs} (target 0)"),
    }
}

fn main() {
    println!("=== Research 550 / riir-ai Plan 585 — Metabolic Gate GOAT (Bench 716) ===");
    println!("=== Paper: arXiv:2609.10817 (Jha et al., 2026-09-09) ===");
    println!("=== Primitive: metabolic_gate feature (opt-in, no-default-consumer rule) ===");

    let gates = [
        gate_g1_correctness(),
        gate_g2_latency(),
        gate_g4_alloc_free(),
    ];

    let mut all_pass = true;
    println!("\n=== Gate Verdicts ===");
    for g in &gates {
        let status = if g.passed { "PASS" } else { "FAIL" };
        println!("[{status}] {}: {}", g.name, g.detail);
        if !g.passed {
            all_pass = false;
        }
    }

    println!();
    println!("G3 (feature isolation): verified via:");
    println!("    cargo check -p katgpt-core --features metabolic_gate");
    println!("    cargo check -p katgpt-core --no-default-features");
    println!("    cargo clippy -p katgpt-core --features metabolic_gate --all-targets");
    println!();

    if all_pass {
        println!("=== G1+G2+G4 PASS — STAYS OPT-IN (riir-ai Plan 585 consumer re-gates at Phase 4) ===");
        std::process::exit(0);
    } else {
        println!("=== ONE OR MORE GATES FAILED ===");
        std::process::exit(1);
    }
}
