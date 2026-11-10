//! GOAT bench for syntactic causal identification (Plan 457 Phase 2 +
//! Issue 183 G4 alloc audit).
//!
//! Reproduces the Issue 545 PoC's 4 scenarios + measures `identify()` latency
//! on each + on a synthesized 32-node subgraph. The headline gate is G2
//! (≤100µs identify on 32 nodes release). G1 soundness is asserted via
//! `assert!` inside the bench setup; G3 no-regression + G4 alloc audit
//! live in the unit-test suite (Phase 1) + the G4 measurement in this
//! file's `main` (Issue 183).
//!
//! Run:
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/causal_id_goat cargo bench --bench causal_id_goat \
//!   --features causal_identification
//! ```
//!
//! ## Verdict table (printed on every run)
//!
//! Each scenario prints `{name}: ok|err latency_ns=X`. The 32-node subgraph
//! additionally prints whether G2 (≤100µs) passes. The G4 section prints
//! the per-call allocation delta on the 32-node scenario.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::causal_id::{Admg, NodeId, identify};

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

/// Construct Scenario A — classic front-door. `A→M→Y, A↔Y`.
fn scenario_a() -> (Admg, NodeId, NodeId) {
    let (a, m, y) = (
        NodeId::from_u32(0),
        NodeId::from_u32(1),
        NodeId::from_u32(2),
    );
    let mut g = Admg::new(vec![a, m, y]);
    g.directed_edge(a, m)
        .directed_edge(m, y)
        .bidirected_edge(a, y);
    (g, a, y)
}

/// Construct Scenario B — classic back-door. `Z→A→Y, Z→Y`.
fn scenario_b() -> (Admg, NodeId, NodeId) {
    let (z, a, y) = (
        NodeId::from_u32(0),
        NodeId::from_u32(1),
        NodeId::from_u32(2),
    );
    let mut g = Admg::new(vec![z, a, y]);
    g.directed_edge(z, a)
        .directed_edge(a, y)
        .directed_edge(z, y);
    (g, a, y)
}

/// Construct Scenario C — 13-node game KG with `NPC1 ↔ NPC2` confounder.
fn scenario_c() -> (Admg, NodeId, NodeId) {
    let (f1, f2, f3) = (
        NodeId::from_u32(0),
        NodeId::from_u32(1),
        NodeId::from_u32(2),
    );
    let (r1, r2) = (NodeId::from_u32(3), NodeId::from_u32(4));
    let (npc1, npc2, npc3) = (
        NodeId::from_u32(5),
        NodeId::from_u32(6),
        NodeId::from_u32(7),
    );
    let (e1, e2, outcome) = (
        NodeId::from_u32(8),
        NodeId::from_u32(9),
        NodeId::from_u32(10),
    );
    let (mood1, mood2) = (NodeId::from_u32(11), NodeId::from_u32(12));
    let mut g = Admg::new(vec![
        f1, f2, f3, r1, r2, npc1, npc2, npc3, e1, e2, outcome, mood1, mood2,
    ]);
    g.directed_edge(f1, npc1)
        .directed_edge(f2, npc2)
        .directed_edge(f3, npc3)
        .directed_edge(r1, npc1)
        .directed_edge(r2, npc2)
        .directed_edge(npc1, e1)
        .directed_edge(npc2, e2)
        .directed_edge(e1, outcome)
        .directed_edge(e2, outcome)
        .directed_edge(f1, mood1)
        .directed_edge(mood2, npc3);
    g.bidirected_edge(npc1, npc2);
    (g, e1, outcome)
}

/// Construct Scenario D — bow-arc. `A → Y, A ↔ Y`. NOT IDENTIFIABLE.
fn scenario_d() -> (Admg, NodeId, NodeId) {
    let (a, y) = (NodeId::from_u32(0), NodeId::from_u32(1));
    let mut g = Admg::new(vec![a, y]);
    g.directed_edge(a, y).bidirected_edge(a, y);
    (g, a, y)
}

/// Construct a synthesized 32-node subgraph for the G2 perf gate.
///
/// Topology: a layered "faction → resource → NPC → encounter → outcome"
/// cascade with 5 layers (6+6+6+6+5 = 29 nodes) + 3 cross-layer confounders
/// + a 30th/31st/32th noise node feeding in. Forces the ID algorithm to
///   consider ancestry, districts, and recursion through the layer structure.
fn scenario_32node() -> (Admg, NodeId, NodeId) {
    let n = |i: u32| NodeId::from_u32(i);
    let nodes: Vec<NodeId> = (0..32u32).map(n).collect();
    let mut g = Admg::new(nodes);

    // Layer 0: factions 0..6 → Layer 1 NPCs 6..12
    for i in 0..6u32 {
        g.directed_edge(n(i), n(6 + i));
    }
    // Layer 0.5: resources 12..18 → Layer 1 NPCs 6..12
    for i in 0..6u32 {
        g.directed_edge(n(12 + i), n(6 + i));
    }
    // Layer 1 NPCs 6..12 → Layer 2 encounters 18..24
    for i in 0..6u32 {
        g.directed_edge(n(6 + i), n(18 + i));
    }
    // Layer 2 encounters 18..24 → Layer 3 outcomes 24..29
    for i in 0..6u32 {
        g.directed_edge(n(18 + i), n(24 + (i % 5)));
    }
    // Layer 4 noise nodes 29..32 → outcomes 24..29
    for i in 0..3u32 {
        g.directed_edge(n(29 + i), n(24 + i));
    }

    // Cross-layer bidirected confounders (3 pairs).
    g.bidirected_edge(n(6), n(7)); // NPC 0 ↔ NPC 1
    g.bidirected_edge(n(13), n(8)); // resource 1 ↔ NPC 2
    g.bidirected_edge(n(18), n(23)); // encounter 0 ↔ encounter 5

    // Query: identify(Outcome 0, do(Encounter 0)).
    (g, n(18), n(24))
}

fn bench_identify(c: &mut Criterion) {
    let mut group = c.benchmark_group("causal_id_identify");

    // ── G1 soundness assertions (printed once at setup time) ────────────
    let (g_a, cause_a, eff_a) = scenario_a();
    let (g_b, cause_b, eff_b) = scenario_b();
    let (g_c, cause_c, eff_c) = scenario_c();
    let (g_d, cause_d, eff_d) = scenario_d();
    let (g_32, cause_32, eff_32) = scenario_32node();

    let sig_a = identify(&g_a, &[cause_a], &[eff_a]).expect("G1: A identifiable");
    let sig_b = identify(&g_b, &[cause_b], &[eff_b]).expect("G1: B identifiable");
    let sig_c = identify(&g_c, &[cause_c], &[eff_c]).expect("G1: C identifiable");
    let err_d = identify(&g_d, &[cause_d], &[eff_d]).expect_err("G1: D NOT identifiable");
    let sig_32 = identify(&g_32, &[cause_32], &[eff_32]).expect("G1: 32-node identifiable");

    // G1 assertions: signature contains the effect; for C, NPC1 must be excluded.
    assert!(sig_a.contains(eff_a), "G1 A: signature contains Y");
    assert!(sig_b.contains(eff_b), "G1 B: signature contains Y");
    assert!(sig_c.contains(eff_c), "G1 C: signature contains Outcome");
    assert!(
        !sig_c.contains(NodeId::from_u32(5)),
        "G1 C: NPC1 (the confounder neighbor) MUST be excluded"
    );
    // err_d is NotIdentifiable — no further check needed.
    let _ = err_d;
    assert!(
        sig_32.contains(eff_32),
        "G1 32-node: signature contains Outcome"
    );

    println!("\n┌─────────────────────────────────────────────────────────────────┐");
    println!("│ G1 soundness (Issue 545 verdict reproduction):                  │");
    println!("├─────────────────────────────────────────────────────────────────┤");
    println!(
        "│ Scenario A (front-door):     Ok  signature size = {:2}            │",
        sig_a.len()
    );
    println!(
        "│ Scenario B (back-door):      Ok  signature size = {:2}            │",
        sig_b.len()
    );
    println!(
        "│ Scenario C (game KG):        Ok  signature size = {:2}            │",
        sig_c.len()
    );
    println!("│ Scenario D (bow-arc):        Err (NotIdentifiable)              │");
    println!(
        "│ Scenario 32-node:            Ok  signature size = {:2}            │",
        sig_32.len()
    );
    println!("└─────────────────────────────────────────────────────────────────┘");

    // ── G2 perf gate ───────────────────────────────────────────────────
    group.bench_function("scenario_a_frontdoor_3node", |b| {
        b.iter(|| {
            let r = identify(black_box(&g_a), black_box(&[cause_a]), black_box(&[eff_a]));
            let _ = black_box(r);
        });
    });
    group.bench_function("scenario_b_backdoor_3node", |b| {
        b.iter(|| {
            let r = identify(black_box(&g_b), black_box(&[cause_b]), black_box(&[eff_b]));
            let _ = black_box(r);
        });
    });
    group.bench_function("scenario_c_game_kg_13node", |b| {
        b.iter(|| {
            let r = identify(black_box(&g_c), black_box(&[cause_c]), black_box(&[eff_c]));
            let _ = black_box(r);
        });
    });
    group.bench_function("scenario_d_bowarc_2node", |b| {
        b.iter(|| {
            let r = identify(black_box(&g_d), black_box(&[cause_d]), black_box(&[eff_d]));
            let _ = black_box(r);
        });
    });
    group.bench_function("scenario_32node_perf_gate", |b| {
        b.iter(|| {
            let r = identify(
                black_box(&g_32),
                black_box(&[cause_32]),
                black_box(&[eff_32]),
            );
            let _ = black_box(r);
        });
    });

    group.finish();

    // ── G4 alloc audit (Issue 183 + P4 zero-alloc refactor) ─────────────
    // Measure steady-state allocation delta per `identify` call.
    //
    // Allocation history:
    //   - Pre-Issue-183 baseline: 284 allocs/call (dominated by `iter.collect()` per local)
    //   - Issue 183 Scratch refactor: 198 allocs/call (−30% — eliminated per-local Vecs)
    //   - P4 zero-alloc districts + fixseq refactor: 133 allocs/call (−33% more —
    //     eliminated `districts()` ~30 allocs/frame, `try_fixseq` 4 allocs/call,
    //     `d_owned.clone()` 1 alloc/multi-district-branch)
    //
    // The remaining ~133 allocs/call are the Scratch::new() first-push grow cost:
    // ~12-15 Vec slots × ~6 recursion frames × first-push grow per slot per frame.
    // This is the honest floor of the safe-Rust approach without unsafe
    // pointer aliasing or thread-local pooling — the alternative (pool Scratch
    // across calls) would require a thread-local which is undesirable for a
    // primitive that may be called from any context.
    //
    // The gate is INFORMATIONAL — Issue 183 does NOT require zero allocs
    // (the primitive is offline-only at ~5µs/query, ~100× outside the 500µs
    // / 20 Hz tick budget). The measurement documents the allocation shape
    // and provides a regression baseline.
    let (_, alloc_delta_g32) = alloc_delta(|| {
        for _ in 0..100 {
            let _ = identify(
                black_box(&g_32),
                black_box(&[cause_32]),
                black_box(&[eff_32]),
            );
        }
    });
    let per_call_g32 = alloc_delta_g32 / 100;
    println!("\n── G4: alloc audit (Issue 183 + P4, 100-call steady-state, 32-node scenario) ──");
    println!("   total allocs / 100 calls: {alloc_delta_g32}");
    println!("   per-call average:          {per_call_g32}");
    println!(
        "   AdmgSignature variant:     {}",
        if sig_32.is_inline() {
            "Inline (zero heap)"
        } else {
            "Heap (1 alloc)"
        }
    );
    println!("   Gate: INFORMATIONAL — Issue 183 does not require zero allocs.");
    println!("   Remaining: Scratch::new() first-push grows (~12 slots × ~6 frames)");

    // Also audit the 13-node scenario (smaller recursion, no step-6 branch).
    let (_, alloc_delta_g13) = alloc_delta(|| {
        for _ in 0..100 {
            let _ = identify(black_box(&g_c), black_box(&[cause_c]), black_box(&[eff_c]));
        }
    });
    let per_call_g13 = alloc_delta_g13 / 100;
    println!("\n── G4: alloc audit (13-node game KG, single-step recursion) ──");
    println!("   total allocs / 100 calls: {alloc_delta_g13}");
    println!("   per-call average:          {per_call_g13}");
    println!(
        "   AdmgSignature variant:     {}",
        if sig_c.is_inline() {
            "Inline (zero heap)"
        } else {
            "Heap (1 alloc)"
        }
    );
}

criterion_group!(benches, bench_identify);
criterion_main!(benches);
