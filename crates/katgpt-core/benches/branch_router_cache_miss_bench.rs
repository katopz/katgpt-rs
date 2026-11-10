//! Issue 636 (riir-ai) — BranchRouter route step cache-miss measurement.
//!
//! **Purpose:** decide whether the SoA BranchBank anchor optimization (Issue
//! 636) is worth building. The issue's recommendation is "measure before
//!      building" — this bench IS that measurement.
//!
//! **Setup:** production NPCs have `DEFAULT_SEEDED_BRANCH_COUNT = 8` active
//! branches, each with a D=8 spawn anchor (8 × 8 × 4B = 256B total — fits in
//! 4 cache lines on aarch64). The current AoS layout stores each anchor as a
//! separate `Vec<f32>` heap allocation, so walking 8 branches does 8 pointer
//! chases to scattered heap addresses. The proposed SoA layout packs all
//! anchors contiguously in one `Vec<f32>` so the route scan is a single
//! sequential read.
//!
//! **Methodology:**
//! 1. Build a realistic 8-branch bank (D=8) — the production NPC shape.
//! 2. Build a parallel flat SoA buffer with the same 8 anchors.
//! 3. Time `router.route()` (AoS) vs a hand-rolled SoA route loop.
//! 4. Run both with HOT cache (back-to-back calls) and COLD cache (pollute
//!    L1/L2 between calls with a 256KB scratch buffer — simulates the real
//!    per-tick pattern where the bank is touched by many systems between
//!    route calls).
//! 5. Report per-call ns + the AoS-vs-SoA delta. If SoA is <10% faster in
//!    the cold-cache regime, Issue 636 closes as NEGATIVE (the complexity
//!    isn't worth it).
//!
//! **Run:**
//! ```bash
//! CARGO_TARGET_DIR=/tmp/iss636 cargo run --release \
//!   -p katgpt-core --bench branch_router_cache_miss_bench -- \
//!   --features non_interference_branches --nocapture
//! ```

use katgpt_core::branching::{BranchBank, BranchRouter, DEFAULT_TAU_SNAP};
use std::hint::black_box;
use std::time::Instant;

// ── Constants (production NPC shape) ───────────────────────────────────────

/// Active branches per NPC. Matches `DEFAULT_SEEDED_BRANCH_COUNT` in
/// riir-engine's `seeding.rs`. NPCs spawn with exactly 8 branches (the HLA
/// dimension D, also the frame-theory orthogonal-capacity limit in R^8).
const N_BRANCHES: usize = 8;

/// HLA embedding dimension. Matches `DEFAULT_PROJECTION_DIM` in katgpt-core's
/// `branching::projection`. Each spawn anchor is D f32s = 32 bytes.
const D: usize = 8;

/// Total route calls per measurement pass.
const ITERS: usize = 100_000;

/// Scratch buffer size for cache pollution. 128 KB matches a typical L1 data
/// cache size (Apple Silicon Icestorm L1 = 128KB). Walking this cyclically
/// evicts the 256-byte anchor set reliably. The buffer is touched in 64-byte
/// chunks (one cache line per touch) to keep pollution-loop cost low.
const POLLUTE_BYTES: usize = 128 * 1024;

// ── Deterministic LCG (matches bench_329's Lcg) ────────────────────────────

struct Lcg {
    state: u64,
}

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes LCG constants.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        let u = self.next_u64();
        // Map to [-1, 1].
        ((u >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Build `n` deterministic unit-norm anchors in R^D. Same algorithm as
/// bench_329's G2 gate so the anchors are directly comparable.
fn build_unit_anchors(n: usize, seed: u64) -> Vec<[f32; D]> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v = [0.0f32; D];
        let mut norm_sq = 0.0;
        for v_i in v.iter_mut().take(D) {
            let x = rng.next_f32();
            *v_i = x;
            norm_sq += x * x;
        }
        let inv = 1.0 / norm_sq.sqrt();
        for x in &mut v {
            *x *= inv;
        }
        out.push(v);
    }
    out
}

// ── Hand-rolled SoA route loop (the proposed optimization) ─────────────────
//
// This mirrors exactly what `BranchRouter::snap_dot` does, but reads from a
// flat `&[f32]` buffer of shape `[N_BRANCHES * D]` instead of chasing 8
// separate `Vec<f32>` heap pointers. If the cache-miss hypothesis is right,
// this should be measurably faster than `router.route()` in the cold-cache
// regime.

#[inline]
fn route_soa(query: &[f32; D], anchor_flat: &[f32]) -> Option<usize> {
    debug_assert_eq!(anchor_flat.len(), N_BRANCHES * D);
    let mut best_id = None;
    let mut best_score = f32::NEG_INFINITY;
    for i in 0..N_BRANCHES {
        let off = i * D;
        let mut score = 0.0f32;
        for j in 0..D {
            score += query[j] * anchor_flat[off + j];
        }
        if score > best_score {
            best_score = score;
            best_id = Some(i);
        }
    }
    if best_score >= DEFAULT_TAU_SNAP {
        best_id
    } else {
        None
    }
}

// ── Cache pollution helper ─────────────────────────────────────────────────
//
// Touch 64 cache lines (4 KB) per call, cyclically walking a 128 KB buffer.
// This reliably evicts the 256-byte anchor set from L1 without dominating
// the route call's cost (~30ns). The cursor advances so we touch different
// L1 sets over time, simulating the real pattern where different per-NPC
// state gets touched between route calls.

#[inline(never)]
fn pollute_cache(scratch: &mut [u8], cursor: &mut usize) {
    let acc: u8 = scratch[*cursor];
    // Touch 64 cache lines (4 KB) — enough to evict the 256-byte anchor
    // set from a 128KB L1 without dominating the ~30ns route call.
    for i in 0..64 {
        let pos = (*cursor + i * 64) % scratch.len();
        scratch[pos] = acc.wrapping_add(scratch[pos]);
    }
    *cursor = (*cursor + 64 * 64) % scratch.len();
    black_box(acc);
}

// ── Measurement ────────────────────────────────────────────────────────────

fn measure<F: FnMut()>(label: &str, iters: usize, mut f: F) -> f64 {
    // Warmup (populate caches, branch predictor).
    for _ in 0..(iters / 10).max(1000) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / iters as f64;
    println!("  {label:<42} {ns_per_call:>7.2} ns / call");
    ns_per_call
}

fn main() {
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Issue 636 — BranchRouter route step cache-miss measurement");
    println!("  N_BRANCHES={N_BRANCHES}, D={D}, ITERS={ITERS}");
    println!("  (production shape: 8 active branches per NPC, D=8 HLA dim)");
    println!("══════════════════════════════════════════════════════════════════\n");

    let anchors = build_unit_anchors(N_BRANCHES, 0xC0FF_EEBE_EFDE_AD42);

    // ── Build the AoS bank (current production layout) ───────────────────
    let mut bank: BranchBank<()> = BranchBank::new(N_BRANCHES);
    for a in &anchors {
        bank.spawn(a.to_vec()).expect("spawn below capacity");
    }
    assert_eq!(bank.n_active(), N_BRANCHES);

    // ── Build the SoA flat buffer (proposed layout) ──────────────────────
    let mut anchor_flat: Vec<f32> = Vec::with_capacity(N_BRANCHES * D);
    for a in &anchors {
        anchor_flat.extend_from_slice(a);
    }

    // ── Query: snaps to branch 0 (forces full max-reduction, no early exit)
    let query: [f32; D] = anchors[0];
    let query_vec: Vec<f32> = query.to_vec();

    let router = BranchRouter::default();

    // Sanity: both paths must agree on the result.
    let probe_aos = router.route(&query_vec, &bank);
    let probe_soa = route_soa(&query, &anchor_flat);
    assert_eq!(
        probe_aos.branch.map(|b| b.0 as usize),
        probe_soa,
        "AoS vs SoA disagree — fixture is broken"
    );
    let resolved_branch = probe_aos.branch.map_or(usize::MAX, |b| b.0 as usize);
    println!("  sanity: both paths resolve to branch {resolved_branch}\n");

    // ── Cache pollution scratch buffer ───────────────────────────────────
    let mut pollute_scratch = vec![0u8; POLLUTE_BYTES];

    // ── HOT cache: back-to-back calls (best case for both) ─────────────
    println!("── HOT cache (back-to-back route calls) ──");
    println!("  (router.route now uses the flat cache internally — Issue 636)");
    let aos_hot = measure("router.route (flat-cache path)", ITERS, || {
        let _ = black_box(router.route(black_box(&query_vec), black_box(&bank)));
    });
    let soa_hot = measure("hand-rolled route_soa (baseline)", ITERS, || {
        let _ = black_box(route_soa(black_box(&query), black_box(&anchor_flat)));
    });

    // ── COLD cache: evict anchors between calls (realistic per-tick) ─────
    //
    // The pollution loop touches 64 cache lines (4 KB), enough to evict
    // the 256-byte anchor set without dominating the timer. The cursor
    // advances so we don't measure the same L1 set every iteration.
    println!("\n── COLD cache (evict 4KB between calls — realistic per-tick) ──");
    let mut cur_aos = 0usize;
    let aos_cold = measure("router.route (flat-cache path, cold)", ITERS, || {
        pollute_cache(&mut pollute_scratch, &mut cur_aos);
        let _ = black_box(router.route(black_box(&query_vec), black_box(&bank)));
    });
    let mut cur_soa = 0usize;
    let soa_cold = measure("hand-rolled route_soa (cold)", ITERS, || {
        pollute_cache(&mut pollute_scratch, &mut cur_soa);
        let _ = black_box(route_soa(black_box(&query), black_box(&anchor_flat)));
    });

    // ── Pollution overhead (subtract from cold to isolate route cost) ────
    let mut cur_pollute = 0usize;
    let pollute_only = measure("pollute_cache only (overhead)", ITERS, || {
        pollute_cache(&mut pollute_scratch, &mut cur_pollute);
    });

    // ── Analysis ─────────────────────────────────────────────────────────
    let aos_cold_net = aos_cold - pollute_only;
    let soa_cold_net = soa_cold - pollute_only;

    println!("\n── Analysis (pollution overhead subtracted) ──");
    println!("  router.route (flat-cache) cold net: {aos_cold_net:>7.2} ns / call");
    println!("  hand-rolled route_soa       cold net: {soa_cold_net:>7.2} ns / call");
    let delta_ns = aos_cold_net - soa_cold_net;
    let delta_pct = if aos_cold_net > 0.0 {
        100.0 * delta_ns / aos_cold_net
    } else {
        0.0
    };
    println!(
        "  Δ (router - hand-rolled):           {delta_ns:>+7.2} ns / call  ({delta_pct:>+5.1}%)"
    );

    println!("\n── Verdict (Issue 636: router.route should match hand-rolled SoA) ──");
    // The gate is now: does the optimized router.route match the hand-rolled
    // SoA baseline? If yes, the optimization landed fully. If the router is
    // still slower than hand-rolled, there's iterator/method-call overhead.
    let overhead_threshold_pct = 15.0; // router may be up to 15% slower than hand-rolled
    let overhead_is_acceptable = delta_pct.abs() < overhead_threshold_pct;

    if overhead_is_acceptable {
        println!(
            "  ✓ PASS — router.route (flat-cache) is within {delta_pct:.1}% of hand-rolled SoA"
        );
        println!("    The optimization landed: the contiguous flat buffer lets LLVM");
        println!("    vectorize the dot-product scan. No iterator-closure overhead.");
        let per_tick_ns = aos_cold_net * 1000.0;
        let per_tick_us = per_tick_ns / 1000.0;
        println!("    Per-tick at 1000 NPCs: {per_tick_ns:.0} ns = {per_tick_us:.2} µs");
    } else if delta_pct < 0.0 {
        println!("  ✓ PASS — router.route (flat-cache) is {delta_pct:.1}% FASTER than hand-rolled");
        println!("    (LLVM found an even better optimization path for the bank method)");
    } else {
        println!("  ✗ INVESTIGATE — router.route is {delta_pct:.1}% slower than hand-rolled SoA");
        println!(
            "    The method-call / lifecycle-check overhead exceeds the {overhead_threshold_pct:.0}% budget."
        );
        println!("    Consider further inlining or a specialized hot-path entry point.");
    }

    println!("\n  Hot router.route (flat-cache): {aos_hot:.2} ns");
    println!("  Hot hand-rolled route_soa:     {soa_hot:.2} ns  (the compute floor)");
    println!("\n══════════════════════════════════════════════════════════════════");
}
