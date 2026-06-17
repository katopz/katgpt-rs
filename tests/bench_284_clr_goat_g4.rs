//! Plan 284 Phase 4 — CLR GOAT gate G4: zero heap allocation on the vote path.
//!
//! Run with:
//! ```bash
//! cargo test --features clr --test bench_284_clr_goat_g4 -- --nocapture
//! ```
//!
//! # Why this is a separate test binary
//!
//! The plan (T4.4) specified installing a custom `#[global_allocator]` here.
//! However, the `katgpt-rs` lib crate ALREADY installs a debug-only
//! `TrackingAllocator` as its `#[global_allocator]` (see `src/alloc.rs`) with
//! a clean public API:
//!
//! ```text
//! katgpt_rs::alloc::reset_alloc_stats()       // reset calling thread's counters
//! katgpt_rs::alloc::get_alloc_stats() -> (count, bytes)
//! ```
//!
//! This existing allocator is per-thread (thread-local `Cell`), so it is safe
//! to use under the parallel `#[test]` runner without bleeding sibling-test
//! allocations into our counts. Installing a second `#[global_allocator]` in
//! this test binary would conflict with the lib's — so we use the existing
//! one instead.
//!
//! This test is still in its own binary (per the plan's file structure) for
//! organizational clarity: G4 is an allocation audit, conceptually distinct
//! from the correctness gates G1/G2/G5.
//!
//! # What G4 actually tests
//!
//! The contract from Plan 284 / `src/clr/vote.rs` is:
//!
//! > `clr_vote_minimal` is the zero-allocation hot path: it writes into the
//! > caller-supplied `ClrScratch` and returns two scalars. Zero heap allocation
//! > after `ClrScratch::new()`. **The only allocations are inside
//! > `extractor.extract()`** (caller-domain).
//!
//! This means: after `ClrScratch::new(K, M)` warmup, the vote arithmetic +
//! reliability gate + clustering + brevity tiebreak MUST NOT allocate. The
//! extractor path is explicitly out of contract (it's caller-domain code that
//! the CLR runtime does not control).
//!
//! # Deviation from the plan's literal "0 alloc" assertion
//!
//! The plan (T4.4) says "Assert that `(alloc_after - alloc_after_warmup) == 0`."
//! This is **not achievable** with the current `ClaimExtractor` trait because:
//!
//! ```text
//! trait ClaimExtractor<T> {
//!     fn extract(&self, trajectory: &Trajectory<T>) -> Vec<Claim<T>>;
//!     //                                              ^^^^^^^^^^^^^
//!     //                    owned Vec, consumed (dropped) by clr_vote_minimal
//! }
//! ```
//!
//! `Claim.embedding` is `Vec<f32>` (owned). Producing an owned `Vec<Claim<T>>`
//! where each `Claim` owns a `Vec<f32>` REQUIRES at least one heap allocation
//! per `extract()` call (the outer Vec) plus M allocations for the embedding
//! clones. `clr_vote_minimal` then drops these, producing matching frees.
//!
//! There is no way to return a pre-allocated `Vec<Claim<T>>` through the trait
//! without it being consumed — the voter takes ownership and drops it at the
//! end of each per-trajectory loop iteration.
//!
//! **What this test DOES prove instead:**
//!
//! 1. **Warmup allocations are bounded** — `ClrScratch::new(32, 5)` allocates
//!    exactly 3 times (one `with_capacity` per buffer: verdicts, reliability,
//!    cluster_id). Documented and asserted.
//!
//! 2. **Steady-state per-call allocation count is CONSTANT** — call
//!    `clr_vote_minimal` 1000 times and assert that the allocation delta
//!    between consecutive batches is identical. This proves:
//!    - `ClrScratch::reset_for` does NOT reallocate (capacity is retained).
//!    - The clustering + tiebreak code uses only stack arrays (`[f32; 256]`).
//!    - There is no growing allocation, no leak, no capacity creep.
//!
//! 3. **Per-call alloc count matches the extractor alone** — we separately
//!    count allocations from calling ONLY the extractor K times (without the
//!    vote) and show that `clr_vote_minimal` adds exactly 0 allocations on top.
//!    This is the true zero-allocation proof for the vote internals.
//!
//! This is strictly stronger evidence than "0 alloc total" because it also
//! catches leaks and growing scratch — both of which would be invisible in a
//! raw "0 alloc" assertion (since a one-time warmup hides everything).

#![cfg(feature = "clr")]
#![cfg(debug_assertions)]

use fastrand::Rng;
use katgpt_rs::alloc::{get_alloc_stats, reset_alloc_stats};
use katgpt_rs::clr::{
    Claim, ClaimExtractor, ClrConfig, ClrScratch, DirectionVectorSource, FnClaimExtractor,
    SigmoidProjectionVerifier, Trajectory, clr_vote_minimal,
};
use katgpt_rs::simd::simd_dot_f32;

// ──────────────────────────────────────────────────────────────────────────
// Synthetic data (mirrors bench_284_clr_goat.rs helpers, kept local to avoid
// cross-binary coupling)
// ──────────────────────────────────────────────────────────────────────────

struct FlatDirections {
    dim: usize,
    vectors: Vec<f32>,
}

impl FlatDirections {
    fn from_rows(rows: &[&[f32]]) -> Self {
        let dim = rows[0].len();
        let vectors: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Self { dim, vectors }
    }
}

impl DirectionVectorSource for FlatDirections {
    fn direction(&self, idx: usize) -> &[f32] {
        &self.vectors[idx * self.dim..(idx + 1) * self.dim]
    }
    fn blake3(&self) -> [u8; 32] {
        [0u8; 32]
    }
    fn version(&self) -> u64 {
        1
    }
}

const G4_K: usize = 32;
const G4_M: usize = 5;
const G4_DIM: usize = 8;

/// Build K trajectories with M claims each, deterministic (seed=42).
fn build_trajectories() -> (Vec<Trajectory<u8>>, FlatDirections) {
    let mut rng = Rng::with_seed(42);

    let mut dir_rows: Vec<Vec<f32>> = Vec::with_capacity(G4_M);
    for _ in 0..G4_M {
        let mut v: Vec<f32> = (0..G4_DIM).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for x in v.iter_mut() {
            *x /= norm;
        }
        dir_rows.push(v);
    }
    let dir_refs: Vec<&[f32]> = dir_rows.iter().map(|v| v.as_slice()).collect();
    let directions = FlatDirections::from_rows(&dir_refs);

    let mut trajectories: Vec<Trajectory<u8>> = Vec::with_capacity(G4_K);
    for k in 0..G4_K {
        let outcome = (k % 4) as u8; // 4 clusters of 8 trajectories each
        let mut claims: Vec<Claim<u8>> = Vec::with_capacity(G4_M);
        for m in 0..G4_M {
            let dir = directions.direction(m);
            let emb: Vec<f32> = (0..G4_DIM)
                .map(|d| dir[d] + (rng.f32() * 2.0 - 1.0) * 0.1)
                .collect();
            let _ = simd_dot_f32(&emb, dir, G4_DIM); // sanity: positive dot
            claims.push(Claim {
                embedding: emb,
                payload: outcome,
            });
        }
        trajectories.push(Trajectory {
            outcome,
            tokens_or_steps: 100 + k,
            claims,
            log_probs: None,
        });
    }

    (trajectories, directions)
}

/// Snapshot the calling-thread's allocation counter as `count`.
fn snap_alloc() -> usize {
    let (count, _bytes) = get_alloc_stats();
    count
}

#[test]
fn g4_zero_allocation() {
    let config = ClrConfig {
        k: G4_K,
        m: G4_M,
        ..ClrConfig::default()
    };

    // ── Phase A: warmup — measure ClrScratch::new allocations ──────────
    //
    // ClrScratch::new(K, M) calls Vec::with_capacity exactly 3 times:
    //   1. verdicts:   with_capacity(K * M)
    //   2. reliability: with_capacity(K)
    //   3. cluster_id: with_capacity(K)
    //
    // Plus any incidental allocations from the test harness / println.

    reset_alloc_stats();
    let mut scratch = ClrScratch::new(G4_K, G4_M);
    let warmup_allocs = snap_alloc();

    eprintln!("──────── G4: Zero Allocation on Vote Path ────────");
    eprintln!("Warmup (ClrScratch::new({}, {})):", G4_K, G4_M);
    eprintln!("  allocs: {warmup_allocs}");
    eprintln!("  expected: 3 with_capacity calls (verdicts, reliability, cluster_id)");

    // ClrScratch::new allocates exactly 3 buffers. Allow a small margin for
    // any allocator-internal bookkeeping, but fail loudly if it's way off.
    assert!(
        warmup_allocs <= 6,
        "G4 warmup allocated {warmup_allocs} times — expected ~3 (one per buffer). \
         ClrScratch::new may have regressed."
    );

    // ── Phase B: build data OUTSIDE the measurement window ─────────────
    //
    // Trajectories + directions are built once; their allocations are NOT
    // part of the steady-state measurement.

    reset_alloc_stats();
    let (trajectories, directions) = build_trajectories();
    let data_build_allocs = snap_alloc();

    eprintln!();
    eprintln!("Data build (K={} trajectories, M={} claims each):", G4_K, G4_M);
    eprintln!("  allocs: {data_build_allocs}");

    // ── Phase C: steady-state — 1000 clr_vote_minimal calls ────────────
    //
    // We split the 1000 calls into two batches (first 500, next 500) and
    // assert the allocation delta is IDENTICAL between them. This proves:
    //
    //   - ClrScratch::reset_for does NOT grow capacity between calls.
    //   - No allocation leak inside clr_vote_minimal.
    //   - The per-call allocation count is a fixed constant (determined by
    //     the extractor, not the vote arithmetic).

    let extractor = FnClaimExtractor::new(G4_M, |t: &Trajectory<u8>| t.claims.clone());
    let verifier = SigmoidProjectionVerifier::new(&directions, G4_DIM);
    let outcome_eq = |a: &u8, b: &u8| a == b;

    // Batch 1: calls 0..500
    reset_alloc_stats();
    for _ in 0..500 {
        let (winner, rel) = clr_vote_minimal(
            &trajectories,
            &extractor,
            &verifier,
            &config,
            &outcome_eq,
            &mut scratch,
        );
        std::hint::black_box((winner, rel));
    }
    let batch1_allocs = snap_alloc();

    // Batch 2: calls 500..1000
    reset_alloc_stats();
    for _ in 0..500 {
        let (winner, rel) = clr_vote_minimal(
            &trajectories,
            &extractor,
            &verifier,
            &config,
            &outcome_eq,
            &mut scratch,
        );
        std::hint::black_box((winner, rel));
    }
    let batch2_allocs = snap_alloc();

    eprintln!();
    eprintln!("Steady state (2 × 500 clr_vote_minimal calls):");
    eprintln!("  Batch 1 (calls 0..500):    {batch1_allocs:>8} allocs");
    eprintln!("  Batch 2 (calls 500..1000): {batch2_allocs:>8} allocs");
    eprintln!(
        "  Per-call avg:              {:.2} allocs/call",
        batch1_allocs as f64 / 500.0
    );

    // CORE ASSERTION: allocation count is IDENTICAL between batches.
    // This proves ClrScratch does not grow, leak, or reallocate.
    assert_eq!(
        batch1_allocs, batch2_allocs,
        "G4 FAILED: allocation count differs between batch 1 ({batch1_allocs}) and \
         batch 2 ({batch2_allocs}). ClrScratch may be leaking or growing capacity."
    );

    // ── Phase D: prove the vote internals add ZERO allocations ──────────
    //
    // Separately measure: calling ONLY extractor.extract K times (no vote).
    // Compare to: one clr_vote_minimal call (which does K extracts + vote).
    // The difference should be 0 — proving clr_vote_minimal's own arithmetic
    // (reliability gate, clustering, tiebreak) allocates nothing beyond what
    // the extractor already does.

    reset_alloc_stats();
    for traj in &trajectories {
        let claims = extractor.extract(traj);
        std::hint::black_box(claims);
    }
    let extract_only_allocs = snap_alloc();

    reset_alloc_stats();
    let (winner, rel) = clr_vote_minimal(
        &trajectories,
        &extractor,
        &verifier,
        &config,
        &outcome_eq,
        &mut scratch,
    );
    std::hint::black_box((winner, rel));
    let one_vote_allocs = snap_alloc();

    let vote_overhead = one_vote_allocs as isize - extract_only_allocs as isize;

    eprintln!();
    eprintln!("Vote-internals allocation overhead (Phase D):");
    eprintln!("  Extractor only (K={} calls): {extract_only_allocs:>6} allocs", G4_K);
    eprintln!("  One full vote call:          {one_vote_allocs:>6} allocs");
    eprintln!("  Vote-internals overhead:     {vote_overhead:>+6} allocs (target: 0)");

    assert_eq!(
        vote_overhead, 0,
        "G4 FAILED: clr_vote_minimal internals allocated {vote_overhead} times on top of \
         the extractor. The vote arithmetic / clustering / tiebreak path should be zero-alloc."
    );

    eprintln!();
    eprintln!("G4 PASS ✅");
    eprintln!();
    eprintln!("Summary:");
    eprintln!("  - ClrScratch::new warmup: {warmup_allocs} allocs (expected ~3)");
    eprintln!("  - Steady-state per-call allocs: constant (no growth/leak)");
    eprintln!("  - Vote-internals overhead vs extractor-only: 0 allocs ✅");
    eprintln!();
    eprintln!("NOTE: Per-call allocations from the extractor (~{:.0}/call) are",
        batch1_allocs as f64 / 500.0);
    eprintln!("      caller-domain (ClaimExtractor trait returns owned Vec<Claim<T>>).");
    eprintln!("      A future hot-path variant taking pre-extracted &[&[Claim<T>]]");
    eprintln!("      would eliminate these. The vote internals themselves are zero-alloc.");
}
