use super::{BenchCategory, BenchResult};
use std::time::Instant;

/// Run G-Zero component benchmarks: HintDelta, TemplateProposer, Δ-Absorb, Δ-Bandit, full pipeline.
///
/// Each benchmark runs with real timing, same warmup/iters pattern as other benches.
/// Returns `BenchResult` structs for CSV + PNG artifact generation.
#[cfg(feature = "g_zero")]
pub fn bench_g_zero() -> Vec<BenchResult> {
    use crate::pruners::{
        AbsorbCompressLayer, BanditPruner, BanditStrategy, CompressConfig, DeltaBanditPruner,
        DeltaGatedAbsorbCompress, DeltaGatedConfig, HintDelta, TemplateProposer,
    };
    use crate::speculative::types::{NoScreeningPruner, ScreeningPruner};
    use std::hint::black_box;

    let warmup = 1_000;
    let iters = 50_000;
    let num_arms = 6; // 6 template categories

    println!("   G-Zero heuristic learning ({iters} iters, {warmup} warmup)...");

    let mut results = Vec::new();

    // Helper: simulated log-probs
    let make_logprobs =
        |len: usize, base: f32| -> Vec<f32> { (0..len).map(|i| base - i as f32 * 0.01).collect() };

    // ── T1: HintDelta::compute (64 tok) ─────────────────────────

    let logp_q = make_logprobs(64, -2.0);
    let logp_qh = make_logprobs(64, -2.5);

    // Warmup
    for _ in 0..warmup {
        let _ = black_box(HintDelta::compute(
            &logp_q,
            &logp_qh,
            "q",
            "h",
            "a_hard",
            "a_assisted",
        ));
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = black_box(HintDelta::compute(
            &logp_q,
            &logp_qh,
            "q",
            "h",
            "a_hard",
            "a_assisted",
        ));
    }
    let elapsed = start.elapsed();
    let t1_throughput = iters as f64 / elapsed.as_secs_f64();
    let t1_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    results.push(BenchResult {
        label: "Hint-\u{03b4} compute (64 tok)".into(),
        throughput: t1_throughput,
        time_per_step_us: t1_us,
        avg_acceptance_len: 0.0,
        color: (70, 130, 180), // steel blue
        category: BenchCategory::HeuristicLearning,
        feature_dim: "Game".into(),
    });

    // ── T2: TemplateProposer::propose() ──────────────────────────

    let mut proposer = TemplateProposer::new(fastrand::Rng::new());

    for _ in 0..warmup {
        let _ = black_box(proposer.propose());
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _ = black_box(proposer.propose());
    }
    let elapsed = start.elapsed();
    let t2_throughput = iters as f64 / elapsed.as_secs_f64();
    let t2_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    results.push(BenchResult {
        label: "TemplateProposer".into(),
        throughput: t2_throughput,
        time_per_step_us: t2_us,
        avg_acceptance_len: 0.0,
        color: (60, 179, 113), // medium sea green
        category: BenchCategory::HeuristicLearning,
        feature_dim: "Game".into(),
    });

    // ── T3: Full G-Zero Pipeline ────────────────────────────────
    // propose → compute δ → feed absorb + bandit + proposer

    let inner_absorb =
        AbsorbCompressLayer::new(NoScreeningPruner, num_arms, CompressConfig::default());
    let mut absorb =
        DeltaGatedAbsorbCompress::new(inner_absorb, num_arms, DeltaGatedConfig::default());

    let inner_bandit = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
    let mut bandit = DeltaBanditPruner::new(inner_bandit, num_arms);

    let mut proposer_pipe = TemplateProposer::new(fastrand::Rng::new());

    let logp_q_pipe = make_logprobs(32, -2.0);
    let logp_qh_pipe = make_logprobs(32, -2.3);

    // Warmup
    for _ in 0..warmup {
        let pair = proposer_pipe.propose();
        let delta = HintDelta::compute(
            &logp_q_pipe,
            &logp_qh_pipe,
            &pair.query,
            &pair.hint,
            "a_hard",
            &pair.hint,
        );
        absorb.observe_hint_delta(pair.template_id, &delta);
        bandit.observe_hint_delta(pair.template_id, &delta);
        proposer_pipe.observe_delta(pair.template_id, delta.value);
    }

    let start = Instant::now();
    for _ in 0..iters {
        let pair = proposer_pipe.propose();
        let delta = HintDelta::compute(
            &logp_q_pipe,
            &logp_qh_pipe,
            &pair.query,
            &pair.hint,
            "a_hard",
            &pair.hint,
        );
        absorb.observe_hint_delta(pair.template_id, &delta);
        bandit.observe_hint_delta(pair.template_id, &delta);
        proposer_pipe.observe_delta(pair.template_id, delta.value);
    }
    let elapsed = start.elapsed();
    let t3_throughput = iters as f64 / elapsed.as_secs_f64();
    let t3_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    results.push(BenchResult {
        label: "G-Zero Pipeline".into(),
        throughput: t3_throughput,
        time_per_step_us: t3_us,
        avg_acceptance_len: 0.0,
        color: (255, 165, 0), // orange
        category: BenchCategory::HeuristicLearning,
        feature_dim: "Game".into(),
    });

    // ── T4: Blind Spot Arms (absorb) ────────────────────────────

    let inner_absorb2 =
        AbsorbCompressLayer::new(NoScreeningPruner, num_arms, CompressConfig::default());
    let mut blind_absorb =
        DeltaGatedAbsorbCompress::new(inner_absorb2, num_arms, DeltaGatedConfig::default());

    // Seed δ observations
    for arm in 0..num_arms {
        for _ in 0..10 {
            blind_absorb.observe_delta(arm, arm as f32 * 0.05, 0.5);
        }
    }

    let blind_iters = 10_000;
    for _ in 0..warmup {
        let _ = black_box(blind_absorb.blind_spot_arms(3));
    }

    let start = Instant::now();
    for _ in 0..blind_iters {
        let _ = black_box(blind_absorb.blind_spot_arms(3));
    }
    let elapsed = start.elapsed();
    let t4_throughput = blind_iters as f64 / elapsed.as_secs_f64();
    let t4_us = elapsed.as_secs_f64() * 1_000_000.0 / blind_iters as f64;

    results.push(BenchResult {
        label: "Blind Spot Arms (absorb)".into(),
        throughput: t4_throughput,
        time_per_step_us: t4_us,
        avg_acceptance_len: 0.0,
        color: (147, 112, 219), // medium purple
        category: BenchCategory::HeuristicLearning,
        feature_dim: "Game".into(),
    });

    // ── T5: Blind Spot Arms (bandit) ────────────────────────────

    let inner_bandit2 = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
    let mut blind_bandit = DeltaBanditPruner::new(inner_bandit2, num_arms);

    // Seed δ observations
    for arm in 0..num_arms {
        for _ in 0..10 {
            blind_bandit.observe_delta(arm, arm as f32 * 0.05);
        }
    }

    for _ in 0..warmup {
        let _ = black_box(blind_bandit.blind_spot_arms(3));
    }

    let start = Instant::now();
    for _ in 0..blind_iters {
        let _ = black_box(blind_bandit.blind_spot_arms(3));
    }
    let elapsed = start.elapsed();
    let t5_throughput = blind_iters as f64 / elapsed.as_secs_f64();
    let t5_us = elapsed.as_secs_f64() * 1_000_000.0 / blind_iters as f64;

    results.push(BenchResult {
        label: "Blind Spot Arms (bandit)".into(),
        throughput: t5_throughput,
        time_per_step_us: t5_us,
        avg_acceptance_len: 0.0,
        color: (255, 105, 180), // hot pink
        category: BenchCategory::HeuristicLearning,
        feature_dim: "Game".into(),
    });

    // ── T6: Δ-Absorb observe_delta relevance ────────────────────

    let inner_absorb3 =
        AbsorbCompressLayer::new(NoScreeningPruner, num_arms, CompressConfig::default());
    let mut rel_absorb =
        DeltaGatedAbsorbCompress::new(inner_absorb3, num_arms, DeltaGatedConfig::default());

    for _ in 0..warmup {
        rel_absorb.observe_delta(0, 0.15, 0.5);
        let _ = black_box(rel_absorb.relevance(0, 0, &[]));
    }

    let start = Instant::now();
    for i in 0..iters {
        let arm = i as usize % num_arms;
        rel_absorb.observe_delta(arm, 0.15, 0.5);
        let _ = black_box(rel_absorb.relevance(0, arm, &[]));
    }
    let elapsed = start.elapsed();
    let t6_throughput = iters as f64 / elapsed.as_secs_f64();
    let t6_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    results.push(BenchResult {
        label: "\u{0394}-Absorb relevance".into(),
        throughput: t6_throughput,
        time_per_step_us: t6_us,
        avg_acceptance_len: 0.0,
        color: (169, 169, 169), // dark gray
        category: BenchCategory::HeuristicLearning,
        feature_dim: "Game".into(),
    });

    // ── T7: Δ-Bandit observe_delta relevance ────────────────────

    let inner_bandit3 = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, num_arms);
    let mut rel_bandit = DeltaBanditPruner::new(inner_bandit3, num_arms);

    for _ in 0..warmup {
        rel_bandit.observe_delta(0, 0.15);
        let _ = black_box(rel_bandit.relevance(0, 0, &[]));
    }

    let start = Instant::now();
    for i in 0..iters {
        let arm = i as usize % num_arms;
        rel_bandit.observe_delta(arm, 0.15);
        let _ = black_box(rel_bandit.relevance(0, arm, &[]));
    }
    let elapsed = start.elapsed();
    let t7_throughput = iters as f64 / elapsed.as_secs_f64();
    let t7_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;

    results.push(BenchResult {
        label: "\u{0394}-Bandit relevance".into(),
        throughput: t7_throughput,
        time_per_step_us: t7_us,
        avg_acceptance_len: 0.0,
        color: (169, 169, 169), // dark gray
        category: BenchCategory::HeuristicLearning,
        feature_dim: "Game".into(),
    });

    results
}

/// Placeholder when `g_zero` feature is disabled.
#[cfg(not(feature = "g_zero"))]
pub fn bench_g_zero() -> Vec<BenchResult> {
    Vec::new()
}

// ── FFT G-Zero Pruner Benchmark ────────────────────────────────
