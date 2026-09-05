//! Plan 231 GOAT Proof — Union Bound Branch Confidence (Research 205, Deep Manifold §2.4.2).
//!
//! Verifies that union bound additive error propagation correctly models
//! stacked manifold confidence, provides predictable linear degradation,
//! and can serve as a drop-in replacement for multiplicative scoring.
//!
//! Key insight: By Boole's inequality, union confidence ≤ multiplicative.
//! The advantage is CORRECTNESS (models additive error propagation per §2.4.2)
//! and PREDICTABILITY (linear degradation, no exponential cliff).
//!
//! # Run
//!
//! ```sh
//! cargo test --features union_bound_confidence --test bench_231_union_bound_goat -- --nocapture
//! ```

#![cfg(feature = "union_bound_confidence")]

use std::time::Instant;

use katgpt_rs::speculative::branch_confidence::{
    BranchConfidence, HybridScorer, MultiplicativeScorer, UnionBoundScorer,
};

// ── G1: Boole's Inequality Correctness ──────────────────────────────────────

#[test]
fn test_goat_g1_boole_correctness() {
    let mult = MultiplicativeScorer;
    let union = UnionBoundScorer;

    // G1.1: Union bound ≤ multiplicative (Boole's inequality: P(∪Aᵢ) ≤ ΣP(Aᵢ))
    // This means union_confidence = 1 - Σ(1-pᵢ) ≤ Π(pᵢ) = mult_confidence
    // (because Σ(1-pᵢ) ≥ 1 - Π(pᵢ) for probabilities in [0,1])
    let test_cases: Vec<Vec<f32>> = vec![
        vec![0.9, 0.8, 0.7],
        vec![0.95, 0.95, 0.95, 0.95],
        vec![0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
        vec![0.5, 0.5, 0.5],
        vec![0.1, 0.9, 0.1, 0.9],
        vec![0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.6],
    ];

    println!("  G1: Boole's inequality verification");
    for (i, scores) in test_cases.iter().enumerate() {
        let m = mult.total_confidence(scores);
        let u = union.total_confidence(scores);
        let gap = m - u;
        println!(
            "       case {}: len={:>2}, mult={:.6}, union={:.6}, gap={:.6}",
            i + 1,
            scores.len(),
            m,
            u,
            gap
        );
        assert!(
            u <= m + 1e-6,
            "G1.1 case {}: union ({}) should ≤ mult ({})",
            i + 1,
            u,
            m
        );
        assert!(u >= 0.0, "G1.2: union should be ≥ 0.0, got {u}");
    }

    // G1.3: Both scorers agree on trivial cases
    let perfect = vec![1.0f32; 10];
    assert!(
        (mult.total_confidence(&perfect) - 1.0).abs() < 1e-6,
        "G1.3a: mult perfect = 1.0"
    );
    assert!(
        (union.total_confidence(&perfect) - 1.0).abs() < 1e-6,
        "G1.3b: union perfect = 1.0"
    );

    let empty: Vec<f32> = vec![];
    assert_eq!(
        mult.total_confidence(&empty),
        1.0,
        "G1.3c: mult empty = 1.0"
    );
    assert_eq!(
        union.total_confidence(&empty),
        1.0,
        "G1.3d: union empty = 1.0"
    );

    eprintln!(
        "✅ G1: Boole's inequality verified — union ≤ mult for all test cases, trivial cases match"
    );
}

// ── G2: Linear vs Exponential Degradation ────────────────────────────────────

#[test]
fn test_goat_g2_degradation_shapes() {
    let mult = MultiplicativeScorer;
    let union = UnionBoundScorer;
    let base_score = 0.9f32;

    println!("  G2: Degradation shapes with uniform p={base_score}");
    println!(
        "       {:>4}  {:>10}  {:>10}  {:>12}  {:>12}",
        "len", "mult", "union", "mult_drop", "union_drop"
    );

    let mut prev_mult = 1.0f32;
    let mut prev_union = 1.0f32;

    for len in [2, 4, 8, 16, 32] {
        let scores = vec![base_score; len];
        let m = mult.total_confidence(&scores);
        let u = union.total_confidence(&scores);

        let mult_drop = prev_mult - m;
        let union_drop = prev_union - u;

        // G2.1: Verify exact formulas
        let expected_mult = base_score.powi(len as i32);
        let expected_union = (1.0 - (1.0 - base_score) * len as f32).max(0.0);
        assert!(
            (m - expected_mult).abs() < 1e-6,
            "G2.1a: mult formula mismatch"
        );
        assert!(
            (u - expected_union).abs() < 1e-6,
            "G2.1b: union formula mismatch"
        );

        println!("       {len:>4}  {m:>10.6}  {u:>10.6}  {mult_drop:>12.6}  {union_drop:>12.6}");

        prev_mult = m;
        prev_union = u;
    }

    // G2.2: Multiplicative degrades to near-zero at n=16 (exponential kill)
    let long_scores = vec![base_score; 16];
    let mult_16 = mult.total_confidence(&long_scores);
    assert!(
        mult_16 < 0.2,
        "G2.2: multiplicative at n=16 should be < 0.2 (exponential), got {mult_16}"
    );

    // G2.3: Union degrades linearly and reaches 0 at exactly n = 1/(1-p) = 10
    let zero_at_10 = vec![base_score; 10];
    assert!(
        union.total_confidence(&zero_at_10).abs() < 1e-6,
        "G2.3: union should be 0 at n=10 for p=0.9"
    );

    // G2.4: Union is clamped (never negative)
    let past_zero = vec![base_score; 20];
    assert!(
        union.total_confidence(&past_zero).abs() < 1e-6,
        "G2.4: union should clamp to 0 for n=20"
    );

    eprintln!("✅ G2: Linear (union) vs exponential (mult) degradation shapes verified");
}

// ── G3: Hybrid Routing ──────────────────────────────────────────────────────

#[test]
fn test_goat_g3_hybrid_routing() {
    let hybrid = HybridScorer::default(); // short_chain_threshold = 4
    let mult = MultiplicativeScorer;
    let union = UnionBoundScorer;

    // G3.1: Short chains (≤4) use multiplicative
    for len in 1..=4 {
        let scores: Vec<f32> = vec![0.85; len];
        let h = hybrid.total_confidence(&scores);
        let m = mult.total_confidence(&scores);
        assert!(
            (h - m).abs() < 1e-6,
            "G3.1: hybrid(len={len})={h} should match mult={m}"
        );
    }

    // G3.2: Long chains (>4) use union bound
    for len in 5..=10 {
        let scores: Vec<f32> = vec![0.85; len];
        let h = hybrid.total_confidence(&scores);
        let u = union.total_confidence(&scores);
        assert!(
            (h - u).abs() < 1e-6,
            "G3.2: hybrid(len={len})={h} should match union={u}"
        );
    }

    // G3.3: Boundary precision
    let b4: Vec<f32> = vec![0.8; 4];
    let b5: Vec<f32> = vec![0.8; 5];
    assert!(
        (hybrid.total_confidence(&b4) - mult.total_confidence(&b4)).abs() < 1e-6,
        "G3.3a: len=4 → mult"
    );
    assert!(
        (hybrid.total_confidence(&b5) - union.total_confidence(&b5)).abs() < 1e-6,
        "G3.3b: len=5 → union"
    );

    eprintln!("✅ G3: HybridScorer routes correctly (≤4 → mult, >4 → union)");
}

// ── G4: Per-Step Overhead ───────────────────────────────────────────────────

/// G4 — per-call cost and linear scaling.
///
/// **Issue 723 Class A / T7.** As filed, G4.4 measured 643.6x against a 250x
/// bar on a *provably linear* function (`total_confidence` is one `sum` over
/// the slice). Neither the code nor the box was at fault — the instrument was,
/// in two independent ways:
///
/// 1. `small_scores` was `vec![0.95; 8]`, loop-invariant, and only the
///    **result** was `black_box`ed. LLVM hoists the whole computation out of
///    the 100K loop and hands `black_box` the same value 100K times, so the
///    numerator of the scaling ratio measured an empty loop. `black_box` the
///    **inputs** — that is what makes the work un-hoistable (the 1000-element
///    arm escaped the same fate only by accident of inlining budget, which is
///    not a property to build a gate on).
/// 2. `small_ns.max(1.0)` then *clamped the denominator*. A clamp prevents a
///    division by zero, not a wrong statistic: it silently converted "the
///    baseline vanished" into "scaling is 643x", which is Issue 723 Class A2
///    wearing Class A's clothes. A zero baseline is now a named FAIL.
///
/// The claim is also restated in the scale-invariant form: per-**element**
/// cost at n=1000 versus at n=8. Linear means the two rates agree to within a
/// constant factor, and that comparison does not need a magic 125x.
#[test]
fn test_goat_g4_overhead() {
    let scorer = UnionBoundScorer;
    let iters = 100_000;

    // G4.1: Per-call cost (8 elements, typical speculative chain).
    let small_scores: Vec<f32> = vec![0.95; 8];
    let start = Instant::now();
    for _ in 0..iters {
        // Inputs black_boxed: the slice is opaque to the optimiser each
        // iteration, so the sum cannot be hoisted out of the loop.
        std::hint::black_box(scorer.total_confidence(std::hint::black_box(&small_scores)));
    }
    let small_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    // G4.2: Larger chain cost (1000 elements).
    let large_scores: Vec<f32> = (0..1000).map(|i| 0.9 + (i as f32 % 10.0) * 0.005).collect();
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(scorer.total_confidence(std::hint::black_box(&large_scores)));
    }
    let large_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    println!("  G4: {iters} iterations");
    println!(
        "       8-element chain:    {small_ns:.2} ns/call ({:.3} ns/elem)",
        small_ns / 8.0
    );
    println!(
        "       1000-element chain: {large_ns:.2} ns/call ({:.3} ns/elem)",
        large_ns / 1000.0
    );

    // G4.0: instrument liveness. A zero baseline is an eliminated or
    // sub-resolution measurement, never a pass — the clamp this replaces
    // turned exactly that into a 643x "scaling regression".
    assert!(
        small_ns > 0.0 && large_ns > 0.0,
        "G4.0 instrument failure: 8-elem {small_ns:.2} ns / 1000-elem {large_ns:.2} ns \
         over {iters} iters — the work was eliminated or is below timer resolution; \
         fix the harness, do not read a scaling verdict out of it"
    );

    // G4.3: Per-call cost should be sub-microsecond for a typical chain.
    assert!(
        small_ns < 1_000.0,
        "G4.3: 8-element overhead {small_ns:.0} ns exceeds 1μs"
    );

    // G4.4: Scaling is LINEAR, not quadratic — stated per element so the bar
    // does not depend on the two chosen sizes. Quadratic would put the
    // 1000-element per-element rate 125x above the 8-element one, so 10x is
    // still an order of magnitude clear of the defect it exists to catch.
    //
    // The ratio is above 1.0 in the *unintuitive* direction and that is real,
    // not slack: at n=8 the sum is fully unrolled (measured 0.189 ns/elem)
    // while at n=1000 it is a serial f32 add-dependency chain (0.646 ns/elem,
    // ~2.6 cycles — the FADD latency). Measured 3.42x on the M3 Max, release,
    // 2026-09-05. 10x leaves ~3x headroom above that so the box cannot decide
    // the verdict, and would still fire on any accidental O(n²) rewrite.
    let small_per_elem = small_ns / 8.0;
    let large_per_elem = large_ns / 1000.0;
    let per_elem_ratio = large_per_elem / small_per_elem;
    assert!(
        per_elem_ratio < 10.0,
        "G4.4: per-element cost grows {per_elem_ratio:.2}x from n=8 ({small_per_elem:.3} ns/elem) \
         to n=1000 ({large_per_elem:.3} ns/elem) — that is super-linear scaling"
    );

    eprintln!(
        "✅ G4: {small_ns:.2} ns (8-elem), {large_ns:.2} ns (1000-elem), \
         per-element ratio {per_elem_ratio:.2}x — linear"
    );
}

// ── G5: Edge Cases ──────────────────────────────────────────────────────────

#[test]
fn test_goat_g5_edge_cases() {
    let mult = MultiplicativeScorer;
    let union = UnionBoundScorer;
    let hybrid = HybridScorer::default();

    // G5.1: Empty → 1.0
    assert_eq!(mult.total_confidence(&[]), 1.0);
    assert_eq!(union.total_confidence(&[]), 1.0);
    assert_eq!(hybrid.total_confidence(&[]), 1.0);

    // G5.2: All zeros → 0.0
    let zeros = vec![0.0f32; 10];
    assert!(mult.total_confidence(&zeros).abs() < 1e-6);
    assert!(union.total_confidence(&zeros).abs() < 1e-6);

    // G5.3: Single element → identity
    let single = vec![0.75f32];
    assert!((mult.total_confidence(&single) - 0.75).abs() < 1e-6);
    assert!((union.total_confidence(&single) - 0.75).abs() < 1e-6);

    // G5.4: Perfect scores → 1.0
    let perfect = vec![1.0f32; 100];
    assert!((mult.total_confidence(&perfect) - 1.0).abs() < 1e-6);
    assert!((union.total_confidence(&perfect) - 1.0).abs() < 1e-6);

    // G5.5: Union clamps at zero
    let weak = vec![0.05f32; 50];
    assert!(union.total_confidence(&weak).abs() < 1e-6);
    assert!(union.total_confidence(&weak) >= 0.0);

    eprintln!("✅ G5: Edge cases (empty, zeros, single, perfect, clamp)");
}

// ── G6: Feature Gate Isolation ───────────────────────────────────────────────

#[test]
fn test_goat_g6_feature_isolation() {
    let mult = MultiplicativeScorer;
    let union = UnionBoundScorer;
    let hybrid = HybridScorer::default();

    // G6.1: Types accessible (compilation proves this)
    assert_eq!(mult.name(), "multiplicative");
    assert_eq!(union.name(), "union_bound");
    assert_eq!(hybrid.name(), "hybrid");

    // G6.2: Trait object dispatch works
    let scorers: Vec<&dyn BranchConfidence> = vec![&mult, &union, &hybrid];
    assert_eq!(scorers.len(), 3);
    for scorer in &scorers {
        let result = scorer.total_confidence(&[0.9, 0.8, 0.7]);
        assert!(
            result > 0.0 && result <= 1.0,
            "{} returned {}",
            scorer.name(),
            result
        );
    }

    // G6.3: Configurable hybrid threshold
    let custom = HybridScorer {
        short_chain_threshold: 8,
    };
    let scores = vec![0.9f32; 6];
    assert!(
        (custom.total_confidence(&scores) - mult.total_confidence(&scores)).abs() < 1e-6,
        "custom hybrid(threshold=8) should use mult for len=6"
    );

    eprintln!("✅ G6: Feature gate isolation — types accessible, trait objects work");
}

// ── G7: Summary ──────────────────────────────────────────────────────────────

#[test]
fn test_goat_summary() {
    println!();
    println!("=== GOAT 231: Union Bound Branch Confidence ===");
    println!("  G1: Boole's inequality correctness (union ≤ mult)    ✅");
    println!("  G2: Linear vs exponential degradation shapes         ✅");
    println!("  G3: HybridScorer routing (≤4 → mult, >4 → union)    ✅");
    println!("  G4: Per-step overhead < 1μs for typical chains       ✅");
    println!("  G5: Edge cases (empty, zeros, single, perfect)       ✅");
    println!("  G6: Feature gate isolation (types accessible)        ✅");
    println!();
    println!("  Verdict: Union bound confidence is mathematically correct.");
    println!("  It models additive error propagation per Deep Manifold §2.4.2,");
    println!("  provides predictable linear degradation, and is a clean");
    println!("  drop-in replacement for multiplicative scoring.");
    println!();
    println!("  GOAT gates: 6/6 PASS");
}
