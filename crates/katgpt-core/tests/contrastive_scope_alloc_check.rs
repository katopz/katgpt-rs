//! Issue 674 G4 — zero-allocation steady state for the contrastive-scope hot
//! paths (T2 `scope_score` / `scope_score_from_pairs` document scorer, T3
//! haircut gate). The builder + `finish()` are cold-path corpus statistics
//! (allocation by design); the per-input read side is gated here.
//!
//! Separate test binary (mirrors `switch_cost_alloc_check`) so the
//! `CountingAllocator` global picks up only this module's allocations.
//! **One `#[test]` function** — the counter is process-global.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core \
//!     --test contrastive_scope_alloc_check --release -- --nocapture
//! ```

use katgpt_core::contrastive_scope::{
    ContrastiveScoreBuilder, ScopeGate, oos_probe_battery, scope_score, scope_score_from_pairs,
};
use std::hint::black_box;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

#[test]
fn g4_zero_alloc_steady_state() {
    // Build the table OUTSIDE the measured section (cold path).
    let mut b = ContrastiveScoreBuilder::new(4096, 0.5);
    for i in 0..200u32 {
        b.observe_in(&[i % 2048, i % 977, i % 613]);
        b.observe_out(&[2048 + i % 2048, 2048 + i % 1301]);
    }
    let table = b.finish();
    let gate = ScopeGate {
        kappa: 0.05,
        theta: 8.0,
    };
    let doc: Vec<u32> = (0..10_000u32).map(|i| (i * 7 + 3) % 4096).collect();
    let pairs: Vec<(u32, f32)> = (0..2048u32).map(|w| (w, (w % 5) as f32 + 1.0)).collect();
    let probe_a: Vec<u32> = (0..64u32).map(|i| i % 2048).collect();
    let probe_b: Vec<u32> = (0..64u32).map(|i| 2048 + i % 2048).collect();
    let probes_in: Vec<&[u32]> = vec![&probe_a, &probe_a];
    let probes_out: Vec<&[u32]> = vec![&probe_b, &probe_b];

    let ((), allocs) = alloc_delta(|| {
        let mut acc = 0.0f32;
        for i in 0..10_000u32 {
            acc += scope_score(&table, &doc);
            acc += scope_score_from_pairs(&table, &pairs);
            let v = gate.apply(0.9, acc + i as f32);
            acc += v.haircut;
        }
        black_box(acc);
    });
    assert_eq!(
        allocs, 0,
        "steady-state contrastive_scope paths allocated {allocs}×"
    );

    // The battery is a cold audit statistic — its single documented Vec
    // (the report) is the only allocation allowed per call.
    let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let _report = oos_probe_battery(&table, &gate, &probes_in, &probes_out, 0.9);
    let battery_allocs = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed) - before;
    assert!(
        battery_allocs <= 2,
        "battery allocated {battery_allocs}× (report Vec expected)"
    );
}
