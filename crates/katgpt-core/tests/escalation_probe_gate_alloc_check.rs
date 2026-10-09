//! Plan 621 Phase 2/3 G4 — zero-alloc steady state for the adaptive
//! classifier + the escalation gate (the flip log and the option ladder
//! are caller-owned slices; the selection is scalar folds). Separate
//! single-fn binary (the bench-655 convention: parallel tests share the
//! global counting allocator).

#![cfg(feature = "escalation_probe_gate")]

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::escalation_probe_gate::{EscalationOption, GateInput, gate};
use katgpt_core::state_probe::{AdaptiveClassifier, ClassFlip, ProbeClass, ProbeInput, probe};
use std::sync::atomic::Ordering;

const CALLS: usize = 1_000;

/// Seeded LCG (no global RNG).
fn lcg_unit(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 40) as f64 / (1u64 << 24) as f64
    }
}

/// G4: 0 allocations across 1000 classify→gate cycles with a reused flip
/// log, a reused probe histogram, and a stack ladder.
#[test]
fn g4_zero_alloc_classify_and_gate() {
    let hist = [3u16, 1, 0, 2];
    let ladder = [
        EscalationOption { depth: 1, success_prior: 0.7, corpus_distance: 0.2 },
        EscalationOption { depth: 2, success_prior: 0.9, corpus_distance: 0.6 },
    ];
    let mut unit = lcg_unit(0x0621_A11C);
    let mut run = || {
        let mut log = [ClassFlip {
            at_n: 0,
            from: ProbeClass::Undetermined,
            to: ProbeClass::Undetermined,
            interval_width: 0.0,
        }; 4];
        let mut cl = AdaptiveClassifier::new(0.1, 64, &mut log);
        for n in [8u32, 16, 32] {
            let k = (unit() * f64::from(n)) as u32;
            let e = probe(&ProbeInput { pass_count: k, n, histogram: &hist }, 1.959_963_984_540_054);
            let _ = cl.observe(&e, n);
        }
        let e = probe(&ProbeInput { pass_count: 0, n: 64, histogram: &hist }, 1.959_963_984_540_054);
        gate(
            &GateInput {
                probe: e,
                options: &ladder,
                lambda: 0.1,
            },
            0.1,
            0.5,
        )
    };

    // Warmup before the measured window.
    for _ in 0..5 {
        let _ = run();
    }
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);
    let mut last = katgpt_core::escalation_probe_gate::GateDecision::StayLocal;
    for _ in 0..CALLS {
        last = run();
    }
    let allocs = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let deallocs = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;
    std::hint::black_box((last, &ladder));
    assert_eq!(
        (allocs, deallocs),
        (0, 0),
        "G4 FAIL: {allocs} allocs / {deallocs} deallocs across {CALLS} classify+gate cycles"
    );
}
