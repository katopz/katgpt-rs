//! Plan 593 T4.1 — GOAT gate bench for the Saddle-Trap Escape Gate (G2
//! latency, G5 bit-reproducibility; G1 lives in `tests/saddle_escape_poc.rs`,
//! G4 in `tests/saddle_escape_alloc_check.rs`).
//!
//! # Gates measured here
//!
//! - **G2 (perf)**: `decide()` on the steady-state Continue path — target
//!   ≤ 500 ns/loop (mirrors the halter's own G-gate budget class); the Kick
//!   emission path (BLAKE3 seed) reported, target ≤ 5 µs (rare op);
//!   `apply_kick` at d=1024 reported (rare op, not gated).
//! - **G5 (determinism)**: two identical full episodes produce
//!   bit-identical decision streams (eps via `f32::to_bits`, seeds by
//!   equality).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/plan593 cargo bench -p katgpt-core \
//!   --features saddle_escape --bench saddle_escape_bench -- --nocapture
//! ```

#![cfg(feature = "saddle_escape")]

use katgpt_core::gain_cost_halt::GainCostLoopHalter;
use katgpt_core::saddle_escape::{
    GateDecision, SaddleEscapeGate, TrapConfig, TrapObservables, apply_kick,
};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

const CONTINUE_BUDGET_NS: f64 = 500.0;
const KICK_BUDGET_NS: f64 = 5_000.0;

fn gate() -> SaddleEscapeGate {
    SaddleEscapeGate::wrap(
        GainCostLoopHalter::new(1.0, 2, 1),
        TrapConfig {
            flip_tau: 0.5,
            kick_budget: 2,
            eps0: 0.1,
            eps_decay: 0.5,
            window: 4,
            probe_tau: 0.1,
        },
    )
}

/// Steady Continue stream: aligned steps, stable decode key.
fn g2_continue_latency() -> (f64, bool) {
    let mut g = gate();
    let sb = [3u8; 64];
    let obs = |i: usize| TrapObservables {
        loop_idx: i,
        gain: 1.0,
        cost: 0.5,
        cos_theta: 0.9,
        step_norm: 0.5,
        decoded_key: Some(42),
        probe_drift: None,
        state_bytes: &sb,
    };
    // Warmup.
    for i in 1..=64 {
        black_box(g.decide(obs(i)));
    }
    let n = 10_000;
    let t0 = Instant::now();
    let mut halt = 0u32;
    for i in 65..=(64 + n) {
        if let GateDecision::Halt(_) = black_box(g.decide(obs(i))) {
            halt += 1;
        }
    }
    let ns = t0.elapsed().as_nanos() as f64 / n as f64;
    (ns, ns <= CONTINUE_BUDGET_NS && halt == 0)
}

/// Trap-shaped stream: every loop emits Kick/Trapped decisions (seed path).
fn g2_kick_latency() -> (f64, bool) {
    let n = 2_048;
    let sb = [5u8; 64];
    let t0 = Instant::now();
    let mut kicks = 0u32;
    for e in 0..n {
        // Fresh episode per iteration: kick paths fire every episode.
        let mut g = gate();
        for i in 1..=12u32 {
            let key = Some(3 + u64::from(i % 2 == 0));
            let d = black_box(g.decide(TrapObservables {
                loop_idx: i as usize,
                gain: 1.0,
                cost: 0.01,
                cos_theta: -1.0,
                step_norm: 0.01,
                decoded_key: key,
                probe_drift: None,
                state_bytes: &sb,
            }));
            if matches!(d, GateDecision::Kick { .. }) {
                kicks += 1;
            }
            let _ = e;
        }
    }
    let ns = t0.elapsed().as_nanos() as f64 / (n as f64 * 12.0);
    (ns, ns <= KICK_BUDGET_NS && kicks > 0)
}

fn g2_apply_kick_report() -> f64 {
    let mut state = [0.5f32; 1024];
    let seed = [9u8; 32];
    for _ in 0..8 {
        apply_kick(&mut state, seed, 0.1);
    }
    let n = 1_000;
    let t0 = Instant::now();
    for i in 0..n {
        apply_kick(&mut state, [(i % 251) as u8; 32], 0.1);
    }
    black_box(state[0]);
    t0.elapsed().as_nanos() as f64 / n as f64
}

fn g5_bit_reproducible() -> bool {
    let run = || {
        let mut g = gate();
        let sb = [11u8; 32];
        let mut out: Vec<(u8, [u8; 32], u32)> = Vec::new();
        for i in 1..=64u32 {
            let key = Some(3 + u64::from(i % 2 == 0));
            let d = g.decide(TrapObservables {
                loop_idx: i as usize,
                gain: 1.0,
                cost: 0.01,
                cos_theta: -1.0,
                step_norm: 0.01,
                decoded_key: key,
                probe_drift: None,
                state_bytes: &sb,
            });
            match d {
                GateDecision::Continue => out.push((0, [0; 32], 0)),
                GateDecision::Kick { dir_seed, eps } => {
                    out.push((1, dir_seed, eps.to_bits()))
                }
                GateDecision::Halt(_) => out.push((2, [0; 32], 0)),
            }
        }
        out
    };
    run() == run()
}

fn main() {
    let mut pass = true;

    let (ns, ok) = g2_continue_latency();
    println!("G2 decide() Continue  : {ns:8.1} ns/loop  (≤ {CONTINUE_BUDGET_NS:.0})  {}", if ok { "PASS" } else { "FAIL" });
    pass &= ok;

    let (ns, ok) = g2_kick_latency();
    println!("G2 decide() trap-path : {ns:8.1} ns/loop  (≤ {KICK_BUDGET_NS:.0})  {}", if ok { "PASS" } else { "FAIL" });
    pass &= ok;

    let ns = g2_apply_kick_report();
    println!("G2 apply_kick d=1024  : {ns:8.1} ns/kick  (reported, rare op)");

    let ok = g5_bit_reproducible();
    println!("G5 bit-reproducible   : {}", if ok { "PASS" } else { "FAIL" });
    pass &= ok;

    if pass {
        println!("\nPlan 593 T4.1 GOAT gates: ALL PASS");
    } else {
        println!("\nPlan 593 T4.1 GOAT gates: FAIL");
        std::process::exit(1);
    }
}
