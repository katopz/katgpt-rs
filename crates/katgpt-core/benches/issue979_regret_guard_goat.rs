//! Issue 979 T4 — the regret_guard G2/G4 GOAT bench.
//!
//! G2: ns/round for both primitives at --release (the paper's own regret
//! condition costs O(T·|A|·|Θ|) per sequence, i.e. O(|A|·|Θ|) per round —
//! the bar is µs-class per round; measured here in ns).
//!
//! G4: zero-alloc steady state (observe() rounds only; construction's
//! fixed-buffer allocation is outside the measured window, the same
//! convention as the gate test).
//!
//! harness = false + std::time::Instant (the DEC GOAT convention). Box
//! state is printed beside the numbers and MUST be quoted with them
//! (Issue-021 law). Run: cargo bench -p katgpt-core --features
//! regret_guard --bench issue979_regret_guard_goat

#![cfg(feature = "regret_guard")]

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::regret_guard::{RegretGapConfig, RegretGapGate, VDriftConfig, VDriftDetector};

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// Issue-855 treatment: the timed regions run through `best_of_us` (the
// shared harness — loud-zero defence: it panics on an all-zero reading,
// i.e. a loop the optimiser deleted), best-of over per-round Durations.
#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

const T: usize = 512;
const ROUNDS: usize = 50_000;

fn main() {
    println!("== Issue 979 regret_guard GOAT bench (G2 ns/round + G4 alloc) ==");
    println!(
        "box_state: cpus={} | note: quote `uptime` + power state beside these numbers (Issue-021 law)",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );

    // Binary indicator gate (the battery's posture).
    let mut gate = RegretGapGate::new(
        &[1.0f32, 0.0, 0.0, 1.0],
        2,
        2,
        &RegretGapConfig::new(0.5, T).expect("cfg"),
    )
    .expect("gate");
    let mu = [0.4f32, 0.6];
    for t in 0..1000 {
        let _ = gate.observe(&mu, t % 2).expect("warmup");
    }
    let per_round = ab_timing::best_of_us(8, 200, || {
        let t0 = Instant::now();
        for t in 0..ROUNDS {
            black_box(gate.observe(black_box(&mu), t % 2).expect("round"));
        }
        t0.elapsed()
    }) * 1000.0
        / ROUNDS as f64;
    println!("| RegretGapGate 2×2 indicator | {per_round:.1} ns/round | O(|A|·|Θ|)=4 |");
    assert!(
        per_round < 1_000.0,
        "G2 bar: µs-class per round, measured {per_round:.1} ns"
    );

    // A wider gate (4 actions × 3 states) to show the |A|·|Θ| scaling.
    let mut u = Vec::with_capacity(12);
    for i in 0..12 {
        u.push(((i as f32) * 0.17).sin().clamp(-1.0, 1.0));
    }
    let mut gate_w = wide_gate(&u);
    for t in 0..1000 {
        let _ = gate_w.observe(&[0.2, 0.5, 0.3], t % 3).expect("warmup");
    }
    let per_round_w = ab_timing::best_of_us(8, 200, || {
        let t1 = Instant::now();
        for t in 0..ROUNDS {
            black_box(
                gate_w
                    .observe(black_box(&[0.2, 0.5, 0.3]), t % 3)
                    .expect("round"),
            );
        }
        t1.elapsed()
    }) * 1000.0
        / ROUNDS as f64;
    println!("| RegretGapGate 4×3 general U | {per_round_w:.1} ns/round | O(|A|·|Θ|)=12 |");
    assert!(
        per_round_w < 1_000.0,
        "G2 bar: measured {per_round_w:.1} ns"
    );

    // V-drift detector (ε = 0.05 → 21 tips).
    let mut det = VDriftDetector::new(&VDriftConfig {
        eps: 0.05,
        delta: 0.01,
    })
    .expect("detector");
    for t in 0..1000 {
        let _ = det.observe(0.55, t % 2).expect("warmup");
    }
    let per_round_v = ab_timing::best_of_us(8, 200, || {
        let t2 = Instant::now();
        for t in 0..ROUNDS {
            black_box(det.observe(black_box(0.55), t % 2).expect("round"));
        }
        t2.elapsed()
    }) * 1000.0
        / ROUNDS as f64;
    println!("| VDriftDetector ε=0.05 (21 tips) | {per_round_v:.1} ns/round | O(N_ε) |");
    assert!(
        per_round_v < 1_000.0,
        "G2 bar: measured {per_round_v:.1} ns"
    );

    // G4: zero-alloc steady state (construction outside the window).
    assert_counter_is_live();
    let mut g4_gate = RegretGapGate::new(
        &[1.0f32, 0.0, 0.0, 1.0],
        2,
        2,
        &RegretGapConfig::new(0.5, T).expect("cfg"),
    )
    .expect("gate");
    let mut g4_det = VDriftDetector::new(&VDriftConfig {
        eps: 0.05,
        delta: 0.01,
    })
    .expect("detector");
    let (_, allocs) = alloc_delta(|| {
        for t in 0..ROUNDS {
            let _ = g4_gate.observe(&mu, t % 2).expect("round");
            let _ = g4_det.observe(0.55, t % 2).expect("round");
        }
    });
    println!("| G4 steady-state allocs (gate + detector, {ROUNDS} rounds) | {allocs} | — |");
    assert_eq!(allocs, 0, "G4: observe() must be alloc-free");

    println!();
    println!(
        "G1 = the T3 battery (tests/issue979_regret_guard_gates.rs); this bench carries G2/G4."
    );
}

fn wide_gate(u: &[f32]) -> RegretGapGate {
    RegretGapGate::new(u, 4, 3, &RegretGapConfig::new(0.5, T).expect("cfg")).expect("wide gate")
}
