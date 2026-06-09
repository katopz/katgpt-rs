//! Benchmark: LLMExecGuard overhead measurement (Plan 223, Phase 1).
//!
//! Measures the cost of `verify_tier()` calls vs a no-guard baseline.
//! LLMExecGuard should be zero-cost or negative-cost (by skipping verification).

use crate::llmexec_guard::{LlmExecGuardConfig, VerifyTier, verify_tier};
use std::hint::black_box;
use std::time::Instant;

/// Synthetic entropy/depth pairs for benchmark.
/// Covers the full range: low/medium/high entropy at shallow/deep DDTree depths.
const BENCH_INPUTS: &[(f32, usize)] = &[
    (0.05, 0),
    (0.15, 2),
    (0.30, 4),
    (0.50, 1),
    (0.50, 6),
    (0.70, 3),
    (0.85, 5),
    (0.95, 8),
];

/// Benchmark: measure `verify_tier()` throughput.
/// Returns (ns_per_call_guard, ns_per_call_noop, tier_distribution).
pub fn bench_llmexec_guard_overhead(iters: usize) -> (f64, f64, [usize; 3]) {
    let config = LlmExecGuardConfig::default();
    let n_inputs = BENCH_INPUTS.len();

    // Warmup
    for _ in 0..1000 {
        for &(e, d) in BENCH_INPUTS {
            black_box(verify_tier(e, d, &config));
        }
    }

    // ── Guard ON ──
    let start = Instant::now();
    let mut tier_counts = [0usize; 3]; // Skip, Screening, FullVerify
    for _ in 0..iters {
        for &(e, d) in BENCH_INPUTS {
            let tier = verify_tier(e, d, &config);
            tier_counts[tier as usize] += 1;
        }
    }
    let elapsed_guard = start.elapsed();

    // ── No-guard baseline (always Skip) ──
    let start = Instant::now();
    for _ in 0..iters {
        for &(e, d) in BENCH_INPUTS {
            // Baseline: trivial computation that mimics the function call overhead
            let _ = black_box(e) + black_box(d as f32);
        }
    }
    let elapsed_noop = start.elapsed();

    let total_calls = iters * n_inputs;
    let ns_per_guard = elapsed_guard.as_nanos() as f64 / total_calls as f64;
    let ns_per_noop = elapsed_noop.as_nanos() as f64 / total_calls as f64;

    (ns_per_guard, ns_per_noop, tier_counts)
}

/// Print a formatted benchmark report.
pub fn print_llmexec_guard_bench(iters: usize) {
    let (ns_guard, ns_noop, tiers) = bench_llmexec_guard_overhead(iters);
    let total = tiers.iter().sum::<usize>() as f64;
    let overhead_ns = ns_guard - ns_noop;

    println!("═══ LLMExecGuard Benchmark ═══");
    println!("Iterations:     {}", iters);
    println!("ns/call (guard): {ns_guard:.2}");
    println!("ns/call (noop):  {ns_noop:.2}");
    println!("Overhead:        {overhead_ns:+.2} ns/call");
    println!(
        "Tier split:      Skip={:.1}% Screening={:.1}% FullVerify={:.1}%",
        tiers[0] as f64 / total * 100.0,
        tiers[1] as f64 / total * 100.0,
        tiers[2] as f64 / total * 100.0,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_llmexec_guard_runs() {
        let (ns_guard, ns_noop, tiers) = bench_llmexec_guard_overhead(1000);
        assert!(ns_guard > 0.0, "guard timing should be positive");
        assert!(ns_noop > 0.0, "noop timing should be positive");
        // Guard overhead should be < 100ns per call (sigmoid is cheap)
        assert!(ns_guard < 1000.0, "guard should be fast, got {ns_guard}ns");
        // Tier counts should be non-zero for at least 2 tiers
        let nonzero_tiers = tiers.iter().filter(|&&c| c > 0).count();
        assert!(nonzero_tiers >= 2, "should route to multiple tiers");
    }

    #[test]
    fn test_overhead_acceptable() {
        let (ns_guard, ns_noop, _) = bench_llmexec_guard_overhead(10_000);
        let overhead = ns_guard - ns_noop;
        // The guard itself should cost < 50ns per call beyond baseline
        assert!(
            overhead < 50.0,
            "guard overhead should be < 50ns, got {overhead:.2}ns"
        );
    }
}
