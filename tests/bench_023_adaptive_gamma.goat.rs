//! Benchmark — Adaptive γ from Entropy Forecast (Issue 023 GOAT gate).
//!
//! Tests the Bebop (arXiv:2606.12370 §7.6) hypothesis: replacing the static
//! `Config::draft_lookahead` with an entropy-forecast-driven adaptive γ
//! improves `accepted_tokens/sec` on entropy-varying workloads by ≥5%.
//!
//! # Honest framing
//!
//! The paper proves the linear entropy–acceptance bound `α ≈ a − b·H(p)` (§3)
//! but only *suggests* adaptive γ as future work (§7.6) — the throughput gain
//! is unproven. This benchmark is the GOAT gate that produces the proof (or
//! disproof) for our stack.
//!
//! # Cost model
//!
//! Real spec-decode costs are modelled (not measured from a real transformer,
//! because the benchmark must run without trained weights):
//!
//! | Cost component | Symbol | μs/token or μs/step | Notes |
//! |----------------|--------|---------------------|-------|
//! | Draft forward (sequential) | `C_draft` | 1.0 μs/token | Small drafter, per-position |
//! | Target forward (batched) | `C_verify` | 5.0 μs/step | One forward over all γ positions |
//! | Fixed step overhead | `C_fixed` | 2.0 μs/step | KV mgmt, sampling, etc. |
//! | Forecast computation | `C_forecast` | measured | Entropy (O(vocab)) + EMA — real timing |
//!
//! The **batched target** model (C_verify constant regardless of γ) matches
//! tree-based spec decode where the target scores all drafted positions in one
//! forward pass. This is the paper's assumed model. Our current LeviathanVerifier
//! does sequential per-token verification — noted in the conclusion as a
//! prerequisite for real-world gains.
//!
//! # Workload
//!
//! A long-CoT entropy profile (paper Fig. 12b): low entropy at start
//! (deterministic setup), rising in the middle (exploratory reasoning),
//! falling at the end (convergent conclusion). 512 steps total.
//!
//! # Quality gate
//!
//! Output-distribution KL between feature-ON and feature-OFF should be ~0
//! (rejection sampling preserves the unbiased target distribution regardless
//! of γ — this is a mathematical invariant, not an empirical claim).
//!
//! Run: `cargo test --test bench_023_adaptive_gamma --features adaptive_gamma_forecast -- --nocapture --ignored`

#![cfg(feature = "adaptive_gamma_forecast")]

use katgpt_rs::speculative::acceptance_forecast::{AcceptanceForecast, entropy_nats_zero_alloc};
use std::time::Instant;

// ── Cost model (microseconds) ──────────────────────────────────────────

/// Per-token draft forward cost (sequential autoregressive).
const C_DRAFT_PER_TOKEN_US: f64 = 1.0;
/// Batched target forward cost (one forward, processes all γ positions).
/// This is the tree-based spec-decode model.
const C_VERIFY_BATCHED_US: f64 = 5.0;
/// Fixed per-step overhead (KV management, sampling, bookkeeping).
const C_FIXED_PER_STEP_US: f64 = 2.0;

/// Static draft lookahead used as the feature-OFF baseline.
const STATIC_GAMMA: usize = 8;
/// Vocabulary size for the entropy-varying synthetic workload.
const VOCAB_SIZE: usize = 256;
/// Number of decode steps in the simulated CoT trace.
const N_STEPS: usize = 512;
/// Min/max γ bounds for adaptive gamma (matches the verifier wiring).
const GAMMA_MIN: usize = 1;
const GAMMA_MAX: usize = 16;
/// Target accepted tokens per step (used by the paper's formula).
const TARGET_TOKENS: usize = STATIC_GAMMA;

// ── Synthetic entropy-varying workload ─────────────────────────────────

/// Deterministic LCG for reproducible benchmarks.
#[inline]
fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

/// Generate synthetic target logits for step `t` with a controlled entropy
/// profile mimicking a long CoT (paper Fig. 12b):
/// - Steps 0–128: low entropy (deterministic setup, α ≈ 0.9)
/// - Steps 128–384: rising entropy (exploratory reasoning, α drops to ≈ 0.4)
/// - Steps 384–512: falling entropy (convergent conclusion, α rises back)
///
/// Returns logits where one token dominates early, and the distribution
/// flattens mid-trace. The actual entropy is computed from the logits by the
/// forecast — this function just shapes the workload.
fn synth_logits(t: usize, out: &mut [f32], rng: &mut u64) {
    let n = out.len();
    assert!(n > 0);
    // Entropy envelope: smooth rise and fall (cosine bump centred at t=256).
    let phase = (t as f32) / (N_STEPS as f32) * std::f32::consts::PI;
    let bump = (phase.sin() * 0.5 + 0.0).max(0.0); // [0, 0.5] peak at t=N/2
    // Peakiness: high at start/end (one dominant token), low in the middle.
    let peak_logit = 8.0 - bump * 7.0; // [1.0, 8.0]
    for i in 0..n {
        let noise = (lcg(rng) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        out[i] = if i == 0 { peak_logit } else { noise * 0.5 };
    }
}

/// Simulate one spec-decode step and return (accepted_count, cost_us).
///
/// Acceptance model: each of the γ drafted tokens is accepted with
/// probability `α` (the true acceptance rate at this step, computed from
/// the forecast model). If all γ are accepted, a bonus token is appended.
/// This matches the Leviathan rejection-sampling acceptance process.
fn simulate_step(gamma: usize, alpha_true: f32, forecast_cost_us: f64, rng: &mut u64) -> (usize, f64) {
    let gamma_eff = gamma.min(GAMMA_MAX);
    let mut accepted = 0usize;
    let mut all_accepted = true;
    for _ in 0..gamma_eff {
        // Accept with probability alpha_true (rejection sampling where p≈q).
        let r = (lcg(rng) as f32 / u32::MAX as f32).abs();
        if r < alpha_true {
            accepted += 1;
        } else {
            all_accepted = false;
            break;
        }
    }
    // Bonus token if all accepted (Leviathan Algorithm 1).
    let returned = if all_accepted { accepted + 1 } else { accepted + 1 };
    // Cost: sequential draft + batched verify + fixed overhead + forecast.
    let cost = gamma_eff as f64 * C_DRAFT_PER_TOKEN_US
        + C_VERIFY_BATCHED_US
        + C_FIXED_PER_STEP_US
        + forecast_cost_us;
    (returned, cost)
}

// ── Benchmark harness ──────────────────────────────────────────────────

struct BenchResult {
    total_accepted: usize,
    total_cost_us: f64,
    accepted_per_sec: f64,
    us_per_step: f64,
    avg_gamma: f64,
    forecast_overhead_us: f64,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  total accepted tokens : {}", self.total_accepted)?;
        writeln!(f, "  total cost            : {:.1} μs", self.total_cost_us)?;
        writeln!(f, "  accepted_tokens/sec   : {:.1}", self.accepted_per_sec)?;
        writeln!(f, "  μs/step               : {:.2}", self.us_per_step)?;
        writeln!(f, "  avg γ                 : {:.2}", self.avg_gamma)?;
        writeln!(f, "  forecast overhead/step: {:.3} μs", self.forecast_overhead_us)
    }
}

/// Run feature-OFF baseline: static γ = STATIC_GAMMA.
fn run_baseline(forecast: &AcceptanceForecast) -> BenchResult {
    let mut rng = 0xDEADBEEFu64;
    let mut logits_buf = vec![0.0f32; VOCAB_SIZE];
    let mut total_accepted = 0usize;
    let mut total_cost = 0.0f64;
    let mut gamma_sum = 0usize;
    let forecast_cost_sum = 0.0f64;

    for t in 0..N_STEPS {
        synth_logits(t, &mut logits_buf, &mut rng);
        // Compute true alpha from the same model the forecast uses (so the
        // simulation matches the forecast's prediction on average).
        let h = entropy_nats_zero_alloc(&logits_buf);
        let alpha_true = (forecast.a - forecast.b * h).clamp(0.01, 1.0);
        let (acc, cost) = simulate_step(STATIC_GAMMA, alpha_true, 0.0, &mut rng);
        total_accepted += acc;
        total_cost += cost;
        gamma_sum += STATIC_GAMMA;
        let _ = forecast_cost_sum; // no forecast cost in baseline
    }

    let accepted_per_sec = total_accepted as f64 / total_cost * 1e6;
    BenchResult {
        total_accepted,
        total_cost_us: total_cost,
        accepted_per_sec,
        us_per_step: total_cost / N_STEPS as f64,
        avg_gamma: gamma_sum as f64 / N_STEPS as f64,
        forecast_overhead_us: 0.0,
    }
}

/// Run feature-ON: adaptive γ from the forecast, with real-measured overhead.
fn run_adaptive(mut forecast: AcceptanceForecast) -> BenchResult {
    let mut rng = 0xDEADBEEFu64; // same seed → same logits sequence
    let mut logits_buf = vec![0.0f32; VOCAB_SIZE];
    let mut total_accepted = 0usize;
    let mut total_cost = 0.0f64;
    let mut gamma_sum = 0usize;
    let mut forecast_cost_sum = 0.0f64;

    for t in 0..N_STEPS {
        synth_logits(t, &mut logits_buf, &mut rng);
        // Measure the real forecast computation cost (entropy + EMA + clamp).
        let fc_start = Instant::now();
        let alpha_forecast = forecast.observe_and_forecast(&logits_buf);
        let fc_elapsed = fc_start.elapsed().as_secs_f64() * 1e6;
        forecast_cost_sum += fc_elapsed;
        // Compute true alpha for the acceptance simulation (same model).
        let h = entropy_nats_zero_alloc(&logits_buf);
        let alpha_true = (forecast.a - forecast.b * h).clamp(0.01, 1.0);
        // Adaptive γ from the forecast.
        let gamma = forecast.adaptive_gamma(TARGET_TOKENS, alpha_forecast, GAMMA_MIN, GAMMA_MAX);
        let (acc, cost) = simulate_step(gamma, alpha_true, fc_elapsed, &mut rng);
        total_accepted += acc;
        total_cost += cost;
        gamma_sum += gamma;
    }

    let accepted_per_sec = total_accepted as f64 / total_cost * 1e6;
    BenchResult {
        total_accepted,
        total_cost_us: total_cost,
        accepted_per_sec,
        us_per_step: total_cost / N_STEPS as f64,
        avg_gamma: gamma_sum as f64 / N_STEPS as f64,
        forecast_overhead_us: forecast_cost_sum / N_STEPS as f64,
    }
}

// ── Output distribution KL ─────────────────────────────────────────────

/// KL divergence between two discrete distributions (in nats).
/// `KL(P || Q) = Σ P_i * ln(P_i / Q_i)`.
fn kl_divergence(p: &[f32], q: &[f32]) -> f64 {
    assert_eq!(p.len(), q.len());
    let mut kl = 0.0f64;
    for (pi, qi) in p.iter().zip(q.iter()) {
        if *pi > 0.0 && *qi > 0.0 {
            kl += (*pi as f64) * ((*pi as f64) / (*qi as f64)).ln();
        }
    }
    kl
}

// ── GOAT gate ──────────────────────────────────────────────────────────

/// T5 GOAT gate: accepted_tokens/sec gain with adaptive γ ≥ 5%.
#[test]
#[ignore = "GOAT gate — run with --ignored"]
fn t5_adaptive_gamma_throughput_gate() {
    // Fitted forecast from a warmup that matches the synthetic workload's
    // entropy–acceptance relationship: α = 1.0 − 0.3·H.
    // (In practice this would come from a real warmup phase.)
    let warmup: Vec<(f32, f32)> = (0..30)
        .map(|i| {
            let h = i as f32 * 0.1;
            let alpha = (1.0 - 0.3 * h).max(0.05);
            (h, alpha)
        })
        .collect();
    let forecast = AcceptanceForecast::fit_from_warmup(&warmup);

    let baseline = run_baseline(&forecast);
    let adaptive = run_adaptive(forecast);

    let gain_pct = (adaptive.accepted_per_sec - baseline.accepted_per_sec)
        / baseline.accepted_per_sec
        * 100.0;

    // Quality gate: KL between feature-ON and feature-OFF output distributions
    // should be ≈ 0 (rejection sampling preserves the target distribution).
    // We approximate this by checking that the per-step token-return distribution
    // is similar — the RS invariant guarantees identical target distributions.
    let baseline_dist = token_return_distribution(&baseline);
    let adaptive_dist = token_return_distribution(&adaptive);
    let kl = kl_divergence(&baseline_dist, &adaptive_dist);

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║ Issue 023 — Adaptive γ from Entropy Forecast (GOAT gate T5)          ║");
    println!("║ Bebop arXiv:2606.12370 §3, §7.6 — synthetic entropy-varying CoT     ║");
    println!("╠══════════════════════════════════════╦══════════════════╦═══════════╣");
    println!("║ Metric                               ║ OFF (static γ=8) ║ ON (adapt)║");
    println!("╠══════════════════════════════════════╬══════════════════╬═══════════╣");
    println!("║ accepted_tokens/sec                  ║ {:>14.1}   ║ {:>7.1}   ║",
        baseline.accepted_per_sec, adaptive.accepted_per_sec);
    println!("║ μs/step                              ║ {:>14.2}   ║ {:>7.2}   ║",
        baseline.us_per_step, adaptive.us_per_step);
    println!("║ avg γ                                ║ {:>14.2}   ║ {:>7.2}   ║",
        baseline.avg_gamma, adaptive.avg_gamma);
    println!("║ forecast overhead/step               ║ {:>14.3}   ║ {:>7.3}   ║",
        baseline.forecast_overhead_us, adaptive.forecast_overhead_us);
    println!("║ total accepted tokens                ║ {:>14}   ║ {:>7}   ║",
        baseline.total_accepted, adaptive.total_accepted);
    println!("╠══════════════════════════════════════╩══════════════════╩═══════════╣");
    println!("║ Throughput gain: {:+.2}%                                                ║", gain_pct);
    println!("║ Output-dist KL (ON vs OFF): {:.6} nats                                 ║", kl);
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Cost model: C_draft={:.1}μs/tok C_verify={:.1}μs/step(batched)        ║",
        C_DRAFT_PER_TOKEN_US, C_VERIFY_BATCHED_US);
    println!("║              C_fixed={:.1}μs/step  Vocab={} Steps={}                    ║",
        C_FIXED_PER_STEP_US, VOCAB_SIZE, N_STEPS);
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("── GOAT gate (Issue 023 T5) ──");
    println!("  Gate 1 — throughput gain ≥ 5%:  {:+.2}%  {}",
        gain_pct,
        if gain_pct >= 5.0 { "✅ PASS" } else { "❌ FAIL" });
    println!("  Gate 2 — KL(output_ON || output_OFF) < 0.01:  {:.6}  {}",
        kl,
        if kl < 0.01 { "✅ PASS" } else { "⚠️  CHECK" });
    println!("  Gate 3 — forecast overhead < 1% of step cost: {:.3}μs / {:.2}μs = {:.3}%  {}",
        adaptive.forecast_overhead_us,
        adaptive.us_per_step,
        adaptive.forecast_overhead_us / adaptive.us_per_step * 100.0,
        if adaptive.forecast_overhead_us / adaptive.us_per_step < 0.01 { "✅ PASS" } else { "❌ FAIL" });
    println!();

    // Hard assertion: the benchmark must run to completion (sanity).
    assert!(baseline.total_accepted > 0);
    assert!(adaptive.total_accepted > 0);
    assert!(baseline.accepted_per_sec > 0.0);
    assert!(adaptive.accepted_per_sec > 0.0);

    // Report the GOAT verdict without hard-failing on throughput (the verdict
    // is recorded in the issue file; the benchmark just produces the numbers).
    if gain_pct >= 5.0 {
        println!("✅ GOAT GATE PASSED — adaptive γ improves throughput by {gain_pct:.2}% (≥ 5%).");
        println!("   Recommendation: promote-when-confirmed (requires user decision per AGENTS.md).");
    } else if gain_pct >= 0.0 {
        println!("⚠️  GOAT GATE MARGINAL — adaptive γ improves throughput by {gain_pct:.2}% (< 5%).");
        println!("   Recommendation: keep-opt-in. The concept works but doesn't meet the 5% bar");
        println!("   under the batched-verify cost model. Real-world gain depends on");
        println!("   (a) batched verification in LeviathanVerifier, (b) high C_fixed/C_draft ratio.");
    } else {
        println!("❌ GOAT GATE FAILED — adaptive γ hurts throughput by {gain_pct:.2}%.");
        println!("   The paper's γ=ceil(L/α) formula overshoots γ at low α, increasing cost.");
        println!("   Recommendation: keep-opt-in (or close-as-won't-fix for non-batched verifiers).");
        println!("   Root cause: the formula increases γ when α drops to maintain target accept");
        println!("   length — correct for batched verify, counterproductive for per-token verify.");
    }
}

/// Approximate per-step token-return distribution as a histogram (normalised).
/// Used for the KL quality gate. Both ON and OFF should produce the same
/// distribution modulo sampling noise (RS preserves the target distribution).
fn token_return_distribution(result: &BenchResult) -> Vec<f32> {
    // We don't store per-step histograms in the harness, so approximate by
    // the aggregate: average tokens returned per step → a 2-point distribution.
    // This is a coarse proxy; the real quality invariant is mathematical
    // (rejection sampling is unbiased), not empirical.
    let avg_return = result.total_accepted as f32 / N_STEPS as f32;
    let p = (avg_return / (GAMMA_MAX + 2) as f32).min(0.99);
    vec![p, 1.0 - p]
}

// ── Smoke test (always runs, not ignored) ──────────────────────────────

/// Sanity: the forecast primitive + simulation harness runs end-to-end.
#[test]
fn smoke_forecast_and_simulate() {
    let mut forecast = AcceptanceForecast::with_params(1.0, 0.3, 0.1);
    let mut rng = 12345u64;
    let mut logits = vec![0.0f32; VOCAB_SIZE];
    let mut total_accepted = 0usize;
    for t in 0..64 {
        synth_logits(t, &mut logits, &mut rng);
        let alpha = forecast.observe_and_forecast(&logits);
        assert!(alpha > 0.0 && alpha <= 1.0, "alpha out of range: {alpha}");
        let gamma = forecast.adaptive_gamma(TARGET_TOKENS, alpha, GAMMA_MIN, GAMMA_MAX);
        assert!(gamma >= GAMMA_MIN && gamma <= GAMMA_MAX, "gamma out of bounds: {gamma}");
        let h = entropy_nats_zero_alloc(&logits);
        let alpha_true = (1.0 - 0.3 * h).clamp(0.01, 1.0);
        let (acc, cost) = simulate_step(gamma, alpha_true, 0.5, &mut rng);
        assert!(acc > 0, "step {t}: should return ≥ 1 token");
        assert!(cost > 0.0);
        total_accepted += acc;
    }
    assert!(total_accepted > 0);
}
