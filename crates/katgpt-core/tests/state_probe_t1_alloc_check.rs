//! Plan 621 Phase 1 G4 — zero-alloc steady state for the `state_probe`
//! kernel. Separate test binary (the house single-fn convention: parallel
//! tests share the counting allocator — `bench_576_hint_regret_alloc_check.rs`
//! pattern).

#![cfg(feature = "state_probe")]

use katgpt_core::state_probe::{ProbeInput, probe};

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

/// G4: 10⁴ probe calls over a varying pass-count (defeats LLVM hoisting of
/// the pure kernel) allocate ZERO times. The histogram is fixed stack data
/// borrowed by every call; the pass-count varies through a `black_box`ed
/// base so the loop body cannot constant-fold into a single call.
#[test]
fn g4_zero_alloc_probe_steady_state() {
    assert_counter_is_live();

    let hist: [u16; 64] = {
        let mut h = [1u16; 64];
        h[0] = u16::MAX - 63; // one dominant bin, sum stays u16-safe
        h
    };
    let base = std::hint::black_box(21usize);

    let (sink, allocs) = alloc_delta(|| {
        let mut acc = 0.0f64;
        for i in 0..10_000usize {
            let pass_count = ((base + i) % 64) as u32;
            let input = ProbeInput {
                pass_count,
                n: 64,
                histogram: &hist,
            };
            let est = probe(&input, 1.959_963_984_540_054);
            acc += est.v_hat + est.h_mm + est.wilson_lo + est.wilson_hi;
        }
        acc
    });
    assert_eq!(
        allocs, 0,
        "G4 FAIL: probe() allocated {allocs} times over 10_000 calls"
    );
    std::hint::black_box(sink);
}
