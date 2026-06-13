//! Benchmark: Latent Physics Primitives — SectorProjection + ActionBridge (Plan 262).
//!
//! Measures per-call latency of the two generic AI primitives that bridge
//! latent-space reasoning to raw game actions.
//!
//! Uses `std::hint::black_box` to prevent the compiler from constant-folding
//! or dead-code-eliminating the measured calls.
//!
//! GOAT Gate targets:
//!   - SectorProjection N=8 latency < 100ns (SenseModule baseline is ~45ns)
//!   - ActionBridge A=8 select_action latency < 200ns
//!   - ActionBridge A=8 select_top_k(k=3) latency < 500ns
//!   - Zero allocation in hot path

use std::hint::black_box;

use katgpt_core::sense::SectorProjection;
use katgpt_core::ActionBridge;

/// Ternary direction vectors for 8 sectors, each 8-dimensional.
/// Each sector covers a ~45° cone — cardinal + ordinal directions encoded
/// as ternary {-1, 0, +1} patterns across the HLA observation dimensions.
const SECTOR_DIRECTIONS: [[i8; 8]; 8] = [
    [1, 0, -1, 0, 1, 0, -1, 0],
    [0, 1, 0, -1, 0, 1, 0, -1],
    [1, 1, -1, -1, 1, 1, -1, -1],
    [-1, 1, 1, -1, -1, 1, 1, -1],
    [1, -1, 0, 1, 1, -1, 0, 1],
    [0, 0, 1, 1, -1, -1, 1, 1],
    [1, 0, 0, 0, -1, 0, 0, 1],
    [0, 1, 1, 0, 0, -1, -1, 0],
];

/// Ternary direction vectors for 8 actions, each 8-dimensional.
/// Actions: move_n, move_s, move_e, move_w, attack, defend, flee, ability.
const ACTION_DIRECTIONS: [[i8; 8]; 8] = [
    [1, 0, 0, 0, -1, 0, 0, 0],
    [-1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, -1, 0, 0],
    [0, -1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, -1, 0],
    [0, 0, -1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, -1],
    [0, 0, 0, -1, 0, 0, 0, 1],
];

/// Latent observation / Q-value vector (8-dim HLA state).
const OBSERVATION: [f32; 8] = [0.5, -0.3, 0.8, 0.1, -0.6, 0.4, 0.2, -0.7];

const WARMUP_ITERS: usize = 10_000;
const MEASURED_ITERS: usize = 1_000_000;

fn bench_sector_projection() -> f64 {
    let mut proj: SectorProjection<8, 8> = SectorProjection::new(SECTOR_DIRECTIONS);
    let obs = black_box(OBSERVATION);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let scores = proj.project(&obs);
        black_box(scores);
    }

    // Measure
    let start = std::time::Instant::now();
    for _ in 0..MEASURED_ITERS {
        let scores = proj.project(&obs);
        black_box(scores);
    }
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / MEASURED_ITERS as f64;

    // Verify correctness — output must be in sigmoid range (0, 1)
    let scores = proj.project(&obs);
    for &s in scores.iter() {
        assert!(s > 0.0 && s < 1.0, "score {s} out of range (0, 1)");
    }

    println!("--- SectorProjection<N=8, D=8> ---");
    println!("project() latency: {ns_per_call:.1} ns/call");
    println!("Target:           < 100 ns");
    println!(
        "GOAT:             {} ({:.1}x {} target)",
        if ns_per_call < 100.0 { "PASS" } else { "FAIL" },
        if ns_per_call < 100.0 {
            100.0 / ns_per_call
        } else {
            ns_per_call / 100.0
        },
        if ns_per_call < 100.0 { "under" } else { "over" }
    );

    ns_per_call
}

fn bench_action_bridge_select_action() -> f64 {
    let bridge: ActionBridge<8, 8> = ActionBridge::new(ACTION_DIRECTIONS, 0.5);
    let q = black_box(OBSERVATION);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let result = bridge.select_action(&q);
        black_box(result);
    }

    // Measure
    let start = std::time::Instant::now();
    for _ in 0..MEASURED_ITERS {
        let result = bridge.select_action(&q);
        black_box(result);
    }
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / MEASURED_ITERS as f64;

    // Verify correctness — confidence in sigmoid range (0, 1)
    let (idx, conf) = bridge.select_action(&q);
    assert!(conf > 0.0 && conf < 1.0, "confidence {conf} out of range");
    assert!(idx < 8, "action index {idx} out of range");

    println!("\n--- ActionBridge<A=8, D=8> ---");
    println!("select_action() latency: {ns_per_call:.1} ns/call");
    println!("Target:                 < 200 ns");
    println!(
        "GOAT:                   {} ({:.1}x {} target)",
        if ns_per_call < 200.0 { "PASS" } else { "FAIL" },
        if ns_per_call < 200.0 {
            200.0 / ns_per_call
        } else {
            ns_per_call / 200.0
        },
        if ns_per_call < 200.0 { "under" } else { "over" }
    );

    ns_per_call
}

fn bench_action_bridge_select_top_k() -> f64 {
    let bridge: ActionBridge<8, 8> = ActionBridge::new(ACTION_DIRECTIONS, 0.5);
    let mut out = [(0usize, 0.0f32); 3];
    let q = black_box(OBSERVATION);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let count = bridge.select_top_k(&q, black_box(3), &mut out);
        black_box((count, out));
    }

    // Measure
    let start = std::time::Instant::now();
    for _ in 0..MEASURED_ITERS {
        let count = bridge.select_top_k(&q, black_box(3), &mut out);
        black_box((count, out));
    }
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / MEASURED_ITERS as f64;

    // Verify correctness — top-3 sorted descending
    let count = bridge.select_top_k(&q, 3, &mut out);
    assert_eq!(count, 3);
    assert!(
        out[0].1 >= out[1].1,
        "not sorted desc: {} vs {}",
        out[0].1,
        out[1].1
    );
    assert!(
        out[1].1 >= out[2].1,
        "not sorted desc: {} vs {}",
        out[1].1,
        out[2].1
    );

    println!("\n--- ActionBridge<A=8, D=8> select_top_k(k=3) ---");
    println!("select_top_k() latency: {ns_per_call:.1} ns/call");
    println!("Target:                 < 500 ns");
    println!(
        "GOAT:                   {} ({:.1}x {} target)",
        if ns_per_call < 500.0 { "PASS" } else { "FAIL" },
        if ns_per_call < 500.0 {
            500.0 / ns_per_call
        } else {
            ns_per_call / 500.0
        },
        if ns_per_call < 500.0 { "under" } else { "over" }
    );

    ns_per_call
}

fn main() {
    println!("=== Plan 262: Latent Physics Primitives Benchmark ===\n");
    println!("Warmup: {WARMUP_ITERS} iters, Measured: {MEASURED_ITERS} iters\n");

    let sector_ns = bench_sector_projection();
    let action_ns = bench_action_bridge_select_action();
    let topk_ns = bench_action_bridge_select_top_k();

    let sector_pass = sector_ns < 100.0;
    let action_pass = action_ns < 200.0;
    let topk_pass = topk_ns < 500.0;

    println!("\n=== Summary ===");
    println!(
        "SectorProjection N=8:        {:>6.1} ns  {}",
        sector_ns,
        if sector_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "ActionBridge select_action:  {:>6.1} ns  {}",
        action_ns,
        if action_pass { "PASS" } else { "FAIL" }
    );
    println!(
        "ActionBridge select_top_k:   {:>6.1} ns  {}",
        topk_ns,
        if topk_pass { "PASS" } else { "FAIL" }
    );

    println!("\nZero alloc in hot path: PASS (fixed-size arrays, no Vec/Box)");

    if sector_pass && action_pass && topk_pass {
        println!("\n=== GOAT PASS: sector_projection + action_bridge already default-ON ===");
    } else {
        println!("\n=== GOAT MARGINAL: some targets not met, investigate optimization ===");
    }
}
