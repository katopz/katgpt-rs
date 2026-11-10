//! Issue 656 G4 — zero-allocation hot path for the privileged engram fuse.
//!
//! Separate test binary (mirrors `bench_655_propagation_alloc_check` /
//! `analytic_lattice_alloc_check`) so the `CountingAllocator` global picks up
//! only this operator's allocations rather than whatever the sibling test
//! binaries are doing in parallel.
//!
//! **One `#[test]` function, not four.** The allocator counter is a process
//! global, so tests in the same binary running on different threads corrupt
//! each other's deltas — the first draft of this file split the checks across
//! four `#[test]`s and read 28 spurious allocs from sibling threads. Every
//! check here runs serially inside one function.
//!
//! Covers:
//! - `fuse_into_hidden_state_privileged` — the fusion hot path.
//! - `PrivilegeLedger::observe` / `observe_trace` / `tick_dual` — the amortized
//!   update path runs less often, not never, so it must not allocate either.
//! - `PrivilegeLedger::privilege` — a cached array read (the sigmoid moved to
//!   the update path in the G2 fix).
//! - `PrivilegeTrace` — stack-only, and `Copy` so snapshotting a trace for a
//!   deferred outcome report is free.
//!
//! # Run
//!
//! ```bash
//! cargo test -p katgpt-core --features engram \
//!     --test bench_656_privilege_alloc_check --release -- --nocapture
//! ```

#![cfg(feature = "engram")]

use katgpt_core::engram::{
    CreditAssignment, EngramConfig, EngramHash, EngramTable, EngramTableBuilder, K_MAX,
    PrivilegeConfig, PrivilegeLedger, PrivilegeTrace, fuse_into_hidden_state_privileged,
};
use std::hint::black_box;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

const D: usize = 32;
const N_SLOTS: usize = 128;
const ITERS: usize = 1_000;

fn build() -> impl EngramTable {
    let mut b = EngramTableBuilder::new(N_SLOTS, D);
    for i in 0..32u64 {
        let pat: Vec<f32> = (0..D)
            .map(|j| ((i as f32) * 0.13 + j as f32 * 0.07).sin())
            .collect();
        b.add_pattern(EngramHash(i), &pat);
    }
    b.build()
}

fn keys() -> [EngramHash; K_MAX] {
    let mut k = [EngramHash(0); K_MAX];
    for (i, slot) in k.iter_mut().enumerate() {
        *slot = EngramHash(i as u64);
    }
    k
}

/// G4: every privilege-gating path is zero-alloc in steady state. Single
/// function so the checks run serially (they share the global
/// `CountingAllocator`).
#[test]
fn g4_zero_alloc_steady_state() {
    let table = build();
    let cfg = EngramConfig::for_dim(D);
    let mut ledger = PrivilegeLedger::for_table(&table, PrivilegeConfig::for_delta_scale(0.3));
    let ks = keys();

    let mut hidden = vec![0.0f32; D];
    let query: Vec<f32> = (0..D).map(|i| ((i as f32) * 0.31).cos()).collect();
    let mut trace = PrivilegeTrace::new();
    let mut lookup = vec![0.0f32; K_MAX * D];
    let mut out = vec![0.0f32; D];

    // ── Warm-up ──────────────────────────────────────────────────────────────
    // Anything that could lazily allocate (the table's commitment cache, the
    // first-call code paths) must have done so before a counted window opens.
    for _ in 0..16 {
        fuse_into_hidden_state_privileged(
            &mut hidden,
            &query,
            &table,
            &ks,
            &cfg,
            &ledger,
            &mut trace,
            &mut lookup,
            &mut out,
        );
        hidden.iter_mut().for_each(|h| *h = 0.0);
    }
    ledger.observe(0, 1.0, 0.2);
    ledger.observe_trace(&trace, 1.0, 0.2, CreditAssignment::GateWeighted);
    ledger.observe_trace(&trace, 1.0, 0.2, CreditAssignment::Uniform);
    ledger.tick_dual();
    black_box(ledger.privilege(0));

    // ── 1. Fusion hot path ───────────────────────────────────────────────────
    let (_, fuse_allocs) = alloc_delta(|| {
        for _ in 0..ITERS {
            fuse_into_hidden_state_privileged(
                black_box(&mut hidden),
                black_box(&query),
                &table,
                black_box(&ks),
                &cfg,
                &ledger,
                &mut trace,
                &mut lookup,
                &mut out,
            );
            hidden.iter_mut().for_each(|h| *h = 0.0);
        }
    });
    black_box(&hidden);
    black_box(trace.len());
    assert_eq!(
        fuse_allocs, 0,
        "privileged fuse must be zero-alloc: {fuse_allocs} allocs over {ITERS} calls"
    );

    // ── 2. Amortized update path ─────────────────────────────────────────────
    let mut full_trace = PrivilegeTrace::new();
    for i in 0..K_MAX {
        full_trace.push(i as u32, 0.1 * (i as f32 + 1.0));
    }
    let (_, update_allocs) = alloc_delta(|| {
        for i in 0..ITERS {
            ledger.observe(black_box(i % N_SLOTS), 1.0, black_box(0.2));
            ledger.observe_trace(
                black_box(&full_trace),
                1.0,
                black_box(0.15),
                CreditAssignment::GateWeighted,
            );
            ledger.observe_trace(
                black_box(&full_trace),
                1.0,
                black_box(0.15),
                CreditAssignment::Uniform,
            );
            ledger.tick_dual();
        }
    });
    black_box(ledger.beta());
    assert_eq!(
        update_allocs, 0,
        "ledger update path must be zero-alloc: {update_allocs} allocs over {ITERS} iterations"
    );

    // ── 3. Privilege read (cached — no sigmoid on the hot path) ──────────────
    let (sum, read_allocs) = alloc_delta(|| {
        let mut s = 0.0f32;
        for i in 0..ITERS {
            s += ledger.privilege(black_box(i % N_SLOTS));
        }
        s
    });
    black_box(sum);
    assert_eq!(
        read_allocs, 0,
        "privilege() must be a pure cached read: {read_allocs} allocs"
    );

    // ── 4. Trace is stack-only, and Copy ─────────────────────────────────────
    let mut t = PrivilegeTrace::new();
    black_box(t.len());
    let (_, trace_allocs) = alloc_delta(|| {
        for _ in 0..ITERS {
            t.clear();
            for i in 0..K_MAX {
                t.push(black_box(i as u32), black_box(1.0));
            }
        }
    });
    black_box(t.len());
    assert_eq!(
        trace_allocs, 0,
        "PrivilegeTrace must be stack-only: {trace_allocs} allocs"
    );

    // Snapshotting a trace for a deferred outcome report must not allocate.
    let (snapshot, copy_allocs) = alloc_delta(|| t);
    assert_eq!(copy_allocs, 0, "PrivilegeTrace copy must not allocate");
    assert_eq!(snapshot.len(), t.len());

    println!(
        "G4 PASS — fuse {fuse_allocs} / update {update_allocs} / read {read_allocs} / \
         trace {trace_allocs} allocs (all must be 0)"
    );
}
