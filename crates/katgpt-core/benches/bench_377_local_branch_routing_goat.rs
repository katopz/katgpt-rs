//! Local Branch Routing GOAT gate bench (Plan 377 Phase 3).
//!
//! Exercises the G2 (router latency < 1µs at K=3, D=64) and G4 (alloc-free
//! hot path) gates for the `branch_routing::*` primitives distilled from
//! Local Branch Routing (arXiv:2606.25354).
//!
//! G1 (correctness ≥ 90% on the PoC domain) and G3 (K=1 bit-identical to
//! standard decode) are unit-test gates — verified in
//! `branch_routing::tests::*` (22 tests, all green). G5 (modelless) and
//! G6 (sigmoid-not-softmax) are structural: the `local_branch_routing`
//! feature has `[]` deps, and the sampling path uses Logistic noise (whose
//! CDF is sigmoid) rather than softmax.
//!
//! # Gates measured here
//!
//! - **G2 (perf)**: `route_argmax` + `route_sampled` median latency on
//!   K=3, D=64 hidden states over 10,000 calls (release). Target: <1µs.
//!   The PoC measured 53 ns at D=16 (IndependentRouter) — the D=64 path
//!   does ~4× more work in the dot-product inner loop.
//! - **G4 (alloc-free hot path)**: `route_argmax` + `route_sampled` both
//!   allocate 0 bytes over 100 steady-state calls (CountingAllocator).
//!   Construction (`DotProductRouter::new`) does one alloc (the
//!   `Box<[f32]>` direction) and is reported separately.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/lbr_goat cargo bench -p katgpt-core \
//!   --features local_branch_routing --bench bench_377_local_branch_routing_goat -- --nocapture
//! ```
//!
//! Or, working around the intermittent macOS dyld/trustd launch stall
//! (documented in Plan 326 / bench_327):
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/lbr_goat target/release/deps/bench_377_local_branch_routing_goat-* --nocapture
//! ```

use katgpt_core::branch_routing::{DotProductRouter, PostCandidateRouter};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ── Constants ──────────────────────────────────────────────────────────────

/// The PoC's main setting: K=3 candidates, D=64 hidden dim.
const K: usize = 3;
const D: usize = 64;
const ITERS: usize = 10_000;
const ALLOC_ITERS: usize = 100;
const TARGET_LATENCY_NS: f64 = 1_000.0; // <1µs

// ── main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Plan 377 Phase 3 — Local Branch Routing GOAT gate");
    println!("  K={K}, D={D}, ITERS={ITERS}, ALLOC_ITERS={ALLOC_ITERS}");
    println!("══════════════════════════════════════════════════════════════════\n");

    // ── Fixture: K=3 candidate hidden states + frozen direction. ────────
    //
    // Candidate 0 is the obvious winner (its first coord dominates).
    let direction: Vec<f32> = (0..D).map(|i| (i as f32) * 0.001 + 0.01).collect();
    let candidates: Vec<Vec<f32>> = (0..K)
        .map(|k| {
            (0..D)
                .map(|i| if k == 0 { (i as f32) * 0.01 + 0.5 } else { 0.1 })
                .collect()
        })
        .collect();
    let candidates_ref: Vec<&[f32]> = candidates.iter().map(|v| v.as_slice()).collect();
    let parent: Vec<&[f32]> = Vec::new();

    let router = DotProductRouter::new(&direction);

    // Sanity: argmax should pick candidate 0.
    let sanity_idx = router.route_argmax(&parent, &candidates_ref);
    assert_eq!(
        sanity_idx, 0,
        "fixture broken: expected argmax=0, got {sanity_idx}"
    );

    // ── G2: route_argmax latency ────────────────────────────────────────
    let mut rng = fastrand::Rng::with_seed(0xCAFE_BABE);
    // Warm-up (populate caches, JIT-style first-call effects).
    for _ in 0..1_000 {
        let _ = black_box(router.route_argmax(black_box(&parent), black_box(&candidates_ref)));
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(router.route_argmax(black_box(&parent), black_box(&candidates_ref)));
    }
    let argmax_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    // ── G2: route_sampled latency ───────────────────────────────────────
    // Low temperature (0.01) — exercises the Logistic-noise path with
    // minimal overhead from the perturbation loop.
    for _ in 0..1_000 {
        let _ = black_box(router.route_sampled(
            black_box(&parent),
            black_box(&candidates_ref),
            0.01,
            &mut rng,
        ));
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(router.route_sampled(
            black_box(&parent),
            black_box(&candidates_ref),
            0.01,
            &mut rng,
        ));
    }
    let sampled_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    // ── G4: alloc-free hot path ─────────────────────────────────────────
    //
    // route_argmax + route_sampled must allocate 0 bytes in steady state.
    // The router construction (DotProductRouter::new) allocates once for
    // the Box<[f32]> direction — that's a one-time setup cost, not a hot
    // path. We measure both for transparency.
    let (_, construct_allocs) = alloc_delta(|| {
        let _ = DotProductRouter::new(&direction);
    });
    let (_, argmax_allocs) = alloc_delta(|| {
        for _ in 0..ALLOC_ITERS {
            let _ = black_box(router.route_argmax(black_box(&parent), black_box(&candidates_ref)));
        }
    });
    let (_, sampled_allocs) = alloc_delta(|| {
        for _ in 0..ALLOC_ITERS {
            let _ = black_box(router.route_sampled(
                black_box(&parent),
                black_box(&candidates_ref),
                0.01,
                &mut rng,
            ));
        }
    });

    // ── Verdict ─────────────────────────────────────────────────────────
    let argmax_pass = argmax_ns < TARGET_LATENCY_NS;
    let sampled_pass = sampled_ns < TARGET_LATENCY_NS;
    let alloc_pass = argmax_allocs == 0 && sampled_allocs == 0 && construct_allocs == 1;

    println!("── G2: router latency ──");
    println!(
        "  route_argmax   {:>8.1} ns / call   (target < {:.0} ns)   {}",
        argmax_ns,
        TARGET_LATENCY_NS,
        pass_fail(argmax_pass)
    );
    println!(
        "  route_sampled  {:>8.1} ns / call   (target < {:.0} ns)   {}",
        sampled_ns,
        TARGET_LATENCY_NS,
        pass_fail(sampled_pass)
    );
    println!("\n── G4: alloc-free hot path ──");
    println!(
        "  DotProductRouter::new    {construct_allocs:>4} alloc (one-time, the Box<[f32]> direction)"
    );
    println!(
        "  route_argmax × {}       {:>4} alloc   (target 0)   {}",
        ALLOC_ITERS,
        argmax_allocs,
        pass_fail(argmax_allocs == 0)
    );
    println!(
        "  route_sampled × {}      {:>4} alloc   (target 0)   {}",
        ALLOC_ITERS,
        sampled_allocs,
        pass_fail(sampled_allocs == 0)
    );

    println!("\n── G1 (correctness ≥ 90%) ──");
    println!("  verified by 22 unit tests in branch_routing::tests::*");
    println!("  (dot_product_argmax_*, collider_adapter_*, sample_logistic_*, object-safety)");

    println!("\n── G3 (K=1 bit-identical to standard decode) ──");
    println!("  verified by dot_product_argmax_k1_returns_zero + collider_adapter_k1_returns_zero");

    println!("\n── G5 (modelless — no training) ──");
    println!("  branch_routing module has zero training deps; all paths are");
    println!("  closed-form (dot-product + Logistic-noise via inverse-CDF).");

    println!("\n── G6 (sigmoid not softmax) ──");
    println!("  route_sampled uses Logistic(0, β) noise (CDF = sigmoid(x/β)).");
    println!("  The Gumbel-max softmax analog is NOT used; no `exp` in the");
    println!("  sampling path — only `ln` for the Logistic inverse-CDF.");

    println!("\n══════════════════════════════════════════════════════════════════");
    let overall_pass = argmax_pass && sampled_pass && alloc_pass;
    println!(
        "  OVERALL: {}",
        if overall_pass {
            "✓ ALL GATES PASS"
        } else {
            "✗ SOME GATES FAILED"
        }
    );
    println!("══════════════════════════════════════════════════════════════════");
    if !overall_pass {
        std::process::exit(1);
    }
}

fn pass_fail(ok: bool) -> &'static str {
    if ok { "✓ PASS" } else { "✗ FAIL" }
}
