//! Plan 411 S2 — SSMax built-in rolling-Δ estimator GOAT gate.
//!
//! Measures the gates that decide whether the built-in estimator is worth
//! promoting from opt-in (`ssmax_adaptive`) to default-on:
//!
//! - **G1 (convergence)**: The estimator's `max-mean` proxy converges to the
//!   true gold-distractor gap Δ under stationary distribution. Target:
//!   `|Δ_estimated − Δ_true| / Δ_true < 10%` after 50 observations.
//! - **G2 (retrieval parity)**: The adaptive mode produced by the estimator
//!   achieves cosine similarity parity with the analytical `s_L = 1/Δ` mode
//!   on the synthetic retrieval task. Target: `cos_adaptive ≥ 0.95 · cos_analytical`.
//! - **G3 (latency)**: `observe_row` + `to_mode` overhead per forward pass.
//!   Target: < 100µs at N=10k (negligible vs attention forward).
//! - **G4 (alloc-free)**: 0 allocations over 1000 `observe_row` calls
//!   (CountingAllocator).
//! - **G5 (no-regression)**: When the estimator is warm-started (Δ=1.0,
//!   before any observation), `to_mode()` produces `s_L = 1.0` — identical to
//!   `Fixed { s_l: 1.0 }`. Zero behavior change at warm-start.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/ssmax_adaptive_gate cargo bench -p katgpt-core \
//!   --features ssmax_adaptive --bench bench_411_ssmax_adaptive_goat -- --nocapture
//! ```

use katgpt_core::ssmax::{RollingDeltaEstimator, SsmaxMode, apply_ssmax_inplace};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ── Synthetic retrieval task (same as bench_411_ssmax_goat) ────────────────

const DELTA: f32 = 0.5;

fn build_retrieval_task(n: usize, delta: f32, seed: u64) -> (Vec<f32>, usize) {
    let gold_index = (seed % n as u64) as usize;
    let mut logits = vec![0.0_f32; n];
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for slot in logits.iter_mut() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let noise = ((z >> 40) as f32) / ((1u64 << 24) as f32) * 0.01;
        *slot = 1.0 + noise;
    }
    logits[gold_index] = 1.0 + delta;
    (logits, gold_index)
}

#[allow(dead_code)]
fn softmax_gold_mass(logits: &[f32], gold_index: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0_f32;
    let mut gold_exp = 0.0_f32;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max).exp();
        sum_exp += e;
        if i == gold_index {
            gold_exp = e;
        }
    }
    if sum_exp > 0.0 {
        gold_exp / sum_exp
    } else {
        0.0
    }
}

/// Cosine similarity between attention output and gold value vector.
/// Each key j has a one-hot value vector v_j = e_{j mod d_model}. The
/// attention output o = Σ_j α_j v_j should point toward v_gold when retrieval
/// succeeds.
fn attention_output_cosine_sim(logits: &[f32], gold_index: usize, d_model: usize) -> f32 {
    let n = logits.len();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0_f32;
    let mut exps = vec![0.0_f32; n];
    for (i, &l) in logits.iter().enumerate() {
        exps[i] = (l - max).exp();
        sum_exp += exps[i];
    }
    if sum_exp <= 0.0 {
        return 0.0;
    }
    // Output vector o[d] = Σ_j α_j * v_j[d] where v_j = e_{j mod d_model}.
    let mut output = vec![0.0_f32; d_model];
    for (j, &e) in exps.iter().enumerate() {
        let alpha = e / sum_exp;
        output[j % d_model] += alpha;
    }
    // Gold value vector = e_{gold_index mod d_model}.
    let mut gold_vec = vec![0.0_f32; d_model];
    gold_vec[gold_index % d_model] = 1.0;
    // Cosine similarity.
    let dot: f32 = output.iter().zip(gold_vec.iter()).map(|(a, b)| a * b).sum();
    let norm_o: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_o > 1e-10 { dot / norm_o } else { 0.0 }
}

// ── main ──────────────────────────────────────────────────────────────────

fn main() {
    use std::sync::atomic::Ordering;

    println!("══════════════════════════════════════════════════════════════════");
    println!("  Plan 411 S2 — SSMax Rolling-Δ Estimator GOAT gate");
    println!("  Δ (true gold-distractor gap) = {DELTA}");
    println!("══════════════════════════════════════════════════════════════════\n");

    let ns: &[usize] = &[64, 1_000, 10_000];

    // ── G1: Convergence — does the max-mean proxy converge to Δ? ──────────
    println!("── G1 (convergence): max-mean proxy vs true Δ ──────────────────");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>12}  {:>12}",
        "N", "true_delta", "est_delta", "rel_err%", "converged?"
    );

    let mut g1_pass = true;
    for &n in ns {
        // Use α = 0.3 (moderate adaptation) and observe 50 forward passes.
        let est = RollingDeltaEstimator::new(0.3);
        for step in 0..50 {
            let (logits, _) = build_retrieval_task(n, DELTA, 42 + step);
            est.observe_row(&logits);
        }
        let est_delta = est.resolve_delta();
        let rel_err = ((est_delta - DELTA).abs() / DELTA) * 100.0;
        let converged = rel_err < 10.0;
        if !converged {
            g1_pass = false;
        }
        println!(
            "{:>10}  {:>12.6}  {:>12.6}  {:>12.2}  {:>12}",
            n,
            DELTA,
            est_delta,
            rel_err,
            if converged { "YES" } else { "NO" }
        );
    }
    println!(
        "\nG1 (convergence): {}\n",
        if g1_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    // ── G2: Retrieval parity — estimator vs analytical s_L=1/Δ ────────────
    println!("── G2 (retrieval parity): estimator vs analytical s_L=1/Δ ──────");
    println!(
        "{:>10}  {:>14}  {:>14}  {:>14}  {:>10}",
        "N", "base_cos_sim", "analytical_cos", "estimator_cos", "parity?"
    );

    let d_model = 16;
    let mut g2_pass = true;
    for &n in ns {
        // Warm up the estimator with 50 observations of the same task distribution.
        let est = RollingDeltaEstimator::new(0.3);
        for step in 0..50 {
            let (logits, _) = build_retrieval_task(n, DELTA, 42 + step);
            est.observe_row(&logits);
        }

        // Base (no SSMax).
        let (base_logits, gold_index) = build_retrieval_task(n, DELTA, 999);
        let base_cos = attention_output_cosine_sim(&base_logits, gold_index, d_model);

        // Analytical SSMax: s_L = 1/Δ (the oracle).
        let log_n = (n as f32).ln();
        let (mut analytical_logits, _) = build_retrieval_task(n, DELTA, 999);
        let mode_analytical = SsmaxMode::Adaptive {
            rolling_delta: DELTA,
        };
        apply_ssmax_inplace(&mut analytical_logits, &mode_analytical, log_n);
        let analytical_cos = attention_output_cosine_sim(&analytical_logits, gold_index, d_model);

        // Estimator SSMax: s_L from max-mean proxy.
        let (mut est_logits, _) = build_retrieval_task(n, DELTA, 999);
        let mode_est = est.to_mode();
        apply_ssmax_inplace(&mut est_logits, &mode_est, log_n);
        let est_cos = attention_output_cosine_sim(&est_logits, gold_index, d_model);

        // Parity: estimator_cos ≥ 0.95 × analytical_cos.
        let parity = if analytical_cos > 0.0 {
            est_cos >= 0.95 * analytical_cos
        } else {
            true // edge case: both ~0
        };
        if !parity {
            g2_pass = false;
        }

        println!(
            "{:>10}  {:>14.6}  {:>14.6}  {:>14.6}  {:>10}",
            n,
            base_cos,
            analytical_cos,
            est_cos,
            if parity { "✓" } else { "✗" }
        );
    }
    println!(
        "\nG2 (retrieval parity): {}\n",
        if g2_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    // ── G3: Latency — observe_row + to_mode overhead ──────────────────────
    println!("── G3 (latency): observe_row + to_mode overhead ────────────────");
    let n = 10_000;
    let (logits, _) = build_retrieval_task(n, DELTA, 42);
    let est = RollingDeltaEstimator::default();

    // Warm up.
    for _ in 0..100 {
        est.observe_row(&logits);
    }
    let _ = est.to_mode();

    let iters = 10_000;
    let start = Instant::now();
    for _ in 0..iters {
        est.observe_row(black_box(&logits));
        let _mode = black_box(est.to_mode());
    }
    let elapsed = start.elapsed();
    let per_call_ns = elapsed.as_nanos() as f64 / iters as f64;
    let g3_pass = per_call_ns < 100_000.0; // < 100µs target
    println!(
        "  observe_row + to_mode at N={}: {:.1} ns/call ({:.2} µs/call)",
        n,
        per_call_ns,
        per_call_ns / 1000.0
    );
    println!("  Target: < 100,000 ns (100 µs)");
    println!(
        "\nG3 (latency): {}\n",
        if g3_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    // ── G4: Alloc-free — 0 allocations over 1000 observe_row calls ────────
    println!("── G4 (alloc-free): 0 allocations over 1000 calls ──────────────");
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let est2 = RollingDeltaEstimator::default();
    for _ in 0..1000 {
        est2.observe_row(&logits);
        let _ = est2.to_mode();
    }
    let after = ALLOC_COUNT.load(Ordering::Relaxed);
    let allocs = after - before;
    let g4_pass = allocs == 0;
    println!("  Allocations: {allocs}");
    println!(
        "\nG4 (alloc-free): {}\n",
        if g4_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    // ── G5: No-regression — warm-start matches Fixed { s_l: 1.0 } ─────────
    println!("── G5 (no-regression): warm-start matches Fixed {{ s_l: 1.0 }} ──");
    let est3 = RollingDeltaEstimator::default();
    let warm_mode = est3.to_mode();
    let warm_s_l = warm_mode.resolve_s_l();
    let fixed_s_l = SsmaxMode::Fixed { s_l: 1.0 }.resolve_s_l();
    let g5_pass = (warm_s_l - fixed_s_l).abs() < 1e-6;
    println!("  Warm-start s_L = {warm_s_l:.6}, Fixed s_L = {fixed_s_l:.6}");
    println!(
        "\nG5 (no-regression): {}\n",
        if g5_pass { "✅ PASS" } else { "❌ FAIL" }
    );

    // ── Summary ───────────────────────────────────────────────────────────
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("══════════════════════════════════════════════════════════════════");
    println!(
        "  G1 (convergence):     {}",
        if g1_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G2 (retrieval parity):{}",
        if g2_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G3 (latency):         {}",
        if g3_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G4 (alloc-free):      {}",
        if g4_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "  G5 (no-regression):   {}",
        if g5_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    let all_pass = g1_pass && g2_pass && g3_pass && g4_pass && g5_pass;
    println!(
        "\n  Overall: {}",
        if all_pass {
            "✅ ALL GATES PASS"
        } else {
            "❌ SOME GATES FAILED"
        }
    );
    println!("\n══════════════════════════════════════════════════════════════════");
}
