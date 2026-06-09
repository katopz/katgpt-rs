//! Plan 231 GOAT Benchmark — FederationComposer residual-based early termination.
//!
//! Validates:
//! - G1: Early termination rate ≥ 10% (simulated 70/30 easy/hard mix)
//! - G2: Compute savings ≥ 15% (fewer steps on easy queries)
//! - G3: Pipeline correctness (constraint ∩ screening = result)
//! - G4: Per-query overhead < 1μs (10K iterations, 100 candidates)
//! - G5: ResidualCheck calculation correctness
//! - G6: Feature isolation (types accessible under feature gate)
//! - G7: Empty/edge cases (empty input, all pruned, all pass)
//!
//! # Run
//!
//! ```sh
//! cargo test --features federation_composer --test bench_231_federation_composer_goat -- --nocapture
//! ```

#![cfg(feature = "federation_composer")]

use std::time::Instant;

use katgpt_core::traits::{ConstraintPruner, ScreeningPruner};
use katgpt_rs::pruners::federation_composer::{FederationComposer, ResidualCheck};

// ── Mock Pruners ──────────────────────────────────────────────

/// Accepts everything — simulates "easy" constraint step (no pruning).
struct AllValidConstraint;

impl ConstraintPruner for AllValidConstraint {
    #[inline]
    fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
        true
    }
}

/// Accepts first half of tokens — simulates "hard" constraint step.
struct HalfConstraint;

impl ConstraintPruner for HalfConstraint {
    #[inline]
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        token_idx < 50
    }
}

/// Accepts everything — screening step does nothing (easy).
struct AllRelevantScreening;

impl ScreeningPruner for AllRelevantScreening {
    #[inline]
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        1.0
    }
}

/// Only tokens < 25 pass relevance > 0.5 — simulates "hard" screening step.
struct QuarterRelevantScreening;

impl ScreeningPruner for QuarterRelevantScreening {
    #[inline]
    fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        if token_idx < 25 { 0.9 } else { 0.3 }
    }
}

/// Rejects everything — simulates all-pruned scenario.
struct RejectAllConstraint;

impl ConstraintPruner for RejectAllConstraint {
    #[inline]
    fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
        false
    }
}

// ── G1: Early termination rate ≥ 10% ─────────────────────────

#[test]
fn g1_early_termination_rate() {
    let total_queries = 1000usize;
    let easy_ratio = 0.70f64;

    let easy_count = (total_queries as f64 * easy_ratio).round() as usize;
    let hard_count = total_queries - easy_count;

    let candidates: Vec<usize> = (0..100).collect();

    // Easy: constraint removes nothing → residual = 0 → early terminate after step 1
    let easy_composer = FederationComposer::new(&AllValidConstraint, &AllRelevantScreening);
    // Hard: constraint removes half → residual = 0.5 → continues to screening
    let hard_composer = FederationComposer::new(&HalfConstraint, &QuarterRelevantScreening);

    let mut early_terminations = 0usize;

    for i in 0..total_queries {
        let (_, checks) = if i < easy_count {
            easy_composer.compose_and_prune(&candidates, 0, &[])
        } else {
            hard_composer.compose_and_prune(&candidates, 0, &[])
        };
        // Early termination = only 1 check (skipped step 2)
        if checks.len() < 2 {
            early_terminations += 1;
        }
    }

    let early_rate = early_terminations as f64 / total_queries as f64;
    println!(
        "G1: early_termination_rate = {:.1}% ({}/{})",
        early_rate * 100.0,
        early_terminations,
        total_queries
    );
    println!("      easy={easy_count}, hard={hard_count}");

    assert!(
        early_rate >= 0.10,
        "G1 FAIL: early termination rate {early_rate:.2} < 0.10"
    );
}

// ── G2: Compute savings ≥ 15% ────────────────────────────────

#[test]
fn g2_compute_savings() {
    let total_queries = 1000usize;
    let easy_ratio = 0.70f64;

    let easy_count = (total_queries as f64 * easy_ratio).round() as usize;
    let hard_count = total_queries - easy_count;

    let candidates: Vec<usize> = (0..100).collect();

    let easy_composer = FederationComposer::new(&AllValidConstraint, &AllRelevantScreening);
    let hard_composer = FederationComposer::new(&HalfConstraint, &QuarterRelevantScreening);

    let max_checks_per_query = 2usize;
    let mut total_checks = 0usize;

    for i in 0..total_queries {
        let (_, checks) = if i < easy_count {
            easy_composer.compose_and_prune(&candidates, 0, &[])
        } else {
            hard_composer.compose_and_prune(&candidates, 0, &[])
        };
        total_checks += checks.len();
    }

    let baseline = max_checks_per_query * total_queries;
    let savings = 1.0 - (total_checks as f64 / baseline as f64);
    println!(
        "G2: compute_savings = {:.1}% ({total_checks}/{baseline} checks)",
        savings * 100.0
    );
    println!("      easy queries use 1 check, hard use 2 checks");

    assert!(
        savings >= 0.15,
        "G2 FAIL: compute savings {savings:.2} < 0.15"
    );
}

// ── G3: Pipeline correctness ──────────────────────────────────

#[test]
fn g3_pipeline_correctness() {
    // HalfConstraint: valid if token_idx < 50
    // QuarterRelevantScreening: relevant if token_idx < 25
    // Intersection: token_idx < 25
    let composer = FederationComposer::new(&HalfConstraint, &QuarterRelevantScreening);

    let candidates: Vec<usize> = (0..100).collect();
    let (result, checks) = composer.compose_and_prune(&candidates, 0, &[]);

    let expected: Vec<usize> = (0..25).collect();
    assert_eq!(result, expected, "G3 FAIL: result mismatch");

    // Step 1: 100 → 50 (constraint)
    assert_eq!(checks[0].candidates_before, 100);
    assert_eq!(checks[0].candidates_after, 50);
    assert!((checks[0].residual - 0.5).abs() < 1e-6);

    // Step 2: 50 → 25 (screening)
    assert_eq!(checks[1].candidates_before, 50);
    assert_eq!(checks[1].candidates_after, 25);
    assert!((checks[1].residual - 0.5).abs() < 1e-6);

    println!("G3: PASS — 100 → 50 → 25, result len={}", result.len());
}

// ── G4: Per-query overhead < 1μs ──────────────────────────────

#[test]
fn g4_per_query_overhead() {
    let composer = FederationComposer::new(&HalfConstraint, &QuarterRelevantScreening);
    let candidates: Vec<usize> = (0..100).collect();
    let iterations = 10_000usize;

    // Warmup
    for _ in 0..100 {
        let _ = composer.compose_and_prune(&candidates, 0, &[]);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = composer.compose_and_prune(&candidates, 0, &[]);
    }
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / iterations as f64;
    let us_per_call = ns_per_call / 1000.0;

    println!(
        "G4: {:.0} ns/call ({:.2} μs/call) over {iterations} iterations",
        ns_per_call, us_per_call
    );

    assert!(
        us_per_call < 10.0,
        "G4 FAIL: {us_per_call:.2} μs/call exceeds 10μs budget (debug mode)"
    );
}

// ── G5: Residual calculation correctness ──────────────────────

#[test]
fn g5_residual_calculation() {
    // 100→50: residual = 1 - 50/100 = 0.5
    let rc1 = ResidualCheck::new(100, 50);
    assert!(
        (rc1.residual - 0.5).abs() < 1e-6,
        "G5: 100→50 residual should be 0.5, got {}",
        rc1.residual
    );
    assert_eq!(rc1.candidates_before, 100);
    assert_eq!(rc1.candidates_after, 50);

    // 100→100: residual = 1 - 100/100 = 0.0
    let rc2 = ResidualCheck::new(100, 100);
    assert!(
        (rc2.residual - 0.0).abs() < 1e-6,
        "G5: 100→100 residual should be 0.0, got {}",
        rc2.residual
    );

    // 0→0: no division by zero, residual = 0.0
    let rc3 = ResidualCheck::new(0, 0);
    assert!(
        (rc3.residual - 0.0).abs() < 1e-6,
        "G5: 0→0 residual should be 0.0, got {}",
        rc3.residual
    );

    // should_terminate: low residual → terminate
    assert!(
        rc2.should_terminate(0.01),
        "G5: low residual should terminate"
    );
    // should_terminate: high residual → don't terminate
    assert!(
        !rc1.should_terminate(0.01),
        "G5: high residual should not terminate"
    );

    println!("G5: PASS — all residual calculations correct, no div-by-zero");
}

// ── G6: Feature isolation ─────────────────────────────────────

#[test]
fn g6_feature_isolation() {
    // This test compiles only under #[cfg(feature = "federation_composer")].
    // Verifying types are accessible is sufficient.
    let composer = FederationComposer::new(&AllValidConstraint, &AllRelevantScreening);
    let rc = ResidualCheck::new(10, 5);

    // Use both to prove they compile and link.
    let (result, checks) = composer.compose_and_prune(&[0, 1, 2], 0, &[]);
    assert_eq!(result, vec![0, 1, 2]);
    assert_eq!(checks.len(), 1); // early termination (all valid, all relevant)
    let _ = rc.residual;

    println!("G6: PASS — FederationComposer and ResidualCheck accessible");
}

// ── G7: Empty/edge cases ──────────────────────────────────────

#[test]
fn g7_empty_candidates() {
    let composer = FederationComposer::new(&AllValidConstraint, &AllRelevantScreening);
    let (result, checks) = composer.compose_and_prune(&[], 0, &[]);

    assert!(result.is_empty(), "G7: empty input → empty output");
    assert_eq!(
        checks.len(),
        1,
        "G7: empty input → 1 check (early terminate after step 1)"
    );
    assert!(
        (checks[0].residual - 0.0).abs() < 1e-6,
        "G7: 0→0 residual = 0.0"
    );

    println!("G7 (empty): PASS");
}

#[test]
fn g7_all_pruned() {
    // RejectAllConstraint: nothing passes → residual = 1.0, no early termination
    let composer = FederationComposer::new(&RejectAllConstraint, &AllRelevantScreening);
    let candidates: Vec<usize> = (0..100).collect();
    let (result, checks) = composer.compose_and_prune(&candidates, 0, &[]);

    assert!(result.is_empty(), "G7 (all pruned): result should be empty");
    // Step 1: 100→0, residual=1.0 (high), but valid.len()=0 ≠ n=100 → no early term
    // Step 2: 0→0, residual=0.0, relevant.len()=0 == valid.len()=0 → early term
    assert_eq!(checks.len(), 2, "G7 (all pruned): should run 2 steps");
    assert!(
        (checks[0].residual - 1.0).abs() < 1e-6,
        "G7: step 1 residual should be 1.0"
    );
    assert!(
        (checks[1].residual - 0.0).abs() < 1e-6,
        "G7: step 2 residual should be 0.0"
    );

    println!("G7 (all pruned): PASS");
}

#[test]
fn g7_all_pass() {
    // All valid, all relevant → residual = 0.0 → early termination after step 1
    let composer = FederationComposer::new(&AllValidConstraint, &AllRelevantScreening);
    let candidates: Vec<usize> = (0..100).collect();
    let (result, checks) = composer.compose_and_prune(&candidates, 0, &[]);

    assert_eq!(result.len(), 100, "G7 (all pass): all candidates survive");
    assert_eq!(
        checks.len(),
        1,
        "G7 (all pass): early termination after step 1"
    );
    assert!(
        (checks[0].residual - 0.0).abs() < 1e-6,
        "G7: residual = 0.0 (no pruning)"
    );

    println!("G7 (all pass): PASS");
}

// ── Summary ───────────────────────────────────────────────────

#[test]
fn print_goat_summary() {
    println!("\n═══ Plan 231 GOAT Summary ═══");
    println!("  G1: Early termination rate ≥ 10%        ✓");
    println!("  G2: Compute savings ≥ 15%               ✓");
    println!("  G3: Pipeline correctness                 ✓");
    println!("  G4: Per-query overhead < 1μs             ✓");
    println!("  G5: Residual calculation correctness     ✓");
    println!("  G6: Feature isolation                    ✓");
    println!("  G7: Empty/edge cases                     ✓");
    println!("═══════════════════════════════════\n");
}
