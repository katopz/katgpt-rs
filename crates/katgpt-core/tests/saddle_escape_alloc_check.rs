//! Plan 593 T4.1 G4 — zero-allocation gate for the SaddleEscapeGate hot path.
//!
//! `SaddleEscapeGate::decide` on the steady-state Continue path (and the
//! Kick emission path, seed hashing included) must allocate 0 bytes after
//! warmup: the gate is a per-loop hot primitive beside
//! `GainCostLoopHalter::halt_decision` (whose own G4 gate this mirrors).
//! `apply_kick` is NOT gated here — it is a rare (≤ kick_budget/episode)
//! op and its BLAKE3 XOF buffers are stack-fixed; its latency is reported
//! by the G2 bench.
//!
//! Separate test binary (the `subspace_phase_gate_alloc_check` convention):
//! `#[global_allocator]` is crate-binary-unique.

#![cfg(feature = "saddle_escape")]

use katgpt_core::gain_cost_halt::GainCostLoopHalter;
use katgpt_core::saddle_escape::{GateDecision, SaddleEscapeGate, TrapConfig, TrapObservables};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

#[test]
fn decide_zero_alloc_after_warmup() {
    let mut gate = SaddleEscapeGate::wrap(
        GainCostLoopHalter::new(1.0, 2, 1),
        TrapConfig {
            flip_tau: 0.5,
            kick_budget: 2,
            eps0: 0.1,
            eps_decay: 0.5,
            window: 2,
            probe_tau: 0.1,
        },
    );
    let sb = [7u8; 64];

    let run = |gate: &mut SaddleEscapeGate, kicks: &mut u32| {
        let mut sink = 0u64;
        for i in 1..=2048u32 {
            // Trap-shaped input so the Kick emission path (BLAKE3 seed +
            // eps schedule) is exercised inside the measured window too.
            let key = Some(3 + u64::from(i % 2 == 0));
            let d = gate.decide(TrapObservables {
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
                GateDecision::Continue => sink += 1,
                GateDecision::Kick { .. } => *kicks += 1,
                GateDecision::Halt(_) => sink += 2,
            }
        }
        sink
    };

    // Warmup: episode 1 (constructs nothing heap-side, but primes any lazy
    // paths). The gate is per-episode — build fresh state, keep the config.
    let mut warm_kicks = 0;
    let warm = run(&mut gate, &mut warm_kicks);
    std::hint::black_box(warm);
    assert!(warm_kicks >= 1, "warmup must exercise the Kick path");

    // Measured episode: fresh gate (the real per-episode pattern), zero
    // allocations across 2048 decides including Kick emissions.
    let mut gate2 = SaddleEscapeGate::wrap(GainCostLoopHalter::new(1.0, 2, 1), *gate.config());
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let mut kicks = 0;
    let sink = run(&mut gate2, &mut kicks);
    std::hint::black_box(sink);
    let allocated = ALLOC_COUNT.load(Ordering::Relaxed) - before;
    assert_eq!(
        allocated, 0,
        "steady-state decide() allocated {allocated} bytes over 2048 loops ({kicks} kicks)"
    );
    assert!(kicks >= 1, "measured episode must include Kick emissions");
}
