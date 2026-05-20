//! RePlaid Variance-Minimized Schedules modelless benchmark — run with:
//! cargo test --features "replaid_schedules,bandit,sdar_gate,dllm" --test bench_replaid_variance_schedules --release -- --nocapture
//!
//! Plan 078: Benchmarks three RePlaid-inspired variance-minimized subsystems:
//! 1. VarianceMinimizer: overhead + convergence
//! 2. AdaptiveNoiseSchedule: variance reduction vs fixed schedule
//! 3. Bandit VarianceEpsilon: convergence vs EpsilonGreedy/UCB1
//! 4. SDAR SdarLearnedBeta: β adaptation vs fixed β

// ── 1. VarianceMinimizer Overhead + Convergence ──────────────────

#[cfg(all(feature = "replaid_schedules", feature = "bandit"))]
#[test]
fn bench_variance_minimizer_overhead() {
    use microgpt_rs::pruners::{VarianceMinimizer, VarianceMinimizerConfig};
    use std::time::Instant;

    let iters = 1_000_000;

    println!("\n🧪 VarianceMinimizer Overhead Benchmark ({iters} iters)");
    println!("{}", "═".repeat(70));

    let config = VarianceMinimizerConfig::default();
    let mut vm = VarianceMinimizer::new(config);

    // Warmup
    for i in 0..1000 {
        vm.observe_and_adapt(i as f32 * 0.01);
    }

    let start = Instant::now();
    for i in 0..iters {
        let cost = (i as f32 * 0.001).sin();
        vm.observe_and_adapt(cost);
    }
    let elapsed = start.elapsed();
    let ns_per_obs = elapsed.as_nanos() as f64 / iters as f64;

    println!("  observe_and_adapt: {ns_per_obs:.1} ns/obs");
    println!("  Final param: {:.4}", vm.param());
    println!("  Final variance: {:.6}", vm.variance());
    println!("  Observations: {}", vm.n_observations());

    // Verify: should be fast
    assert!(
        ns_per_obs < 100.0,
        "observe_and_adapt should be < 100ns, got {ns_per_obs:.1}ns"
    );
}

#[cfg(all(feature = "replaid_schedules", feature = "bandit"))]
#[test]
fn bench_variance_minimizer_convergence() {
    use microgpt_rs::pruners::{VarianceMinimizer, VarianceMinimizerConfig};

    println!("\n🧪 VarianceMinimizer Convergence Benchmark");
    println!("{}", "═".repeat(70));

    let config = VarianceMinimizerConfig {
        mean_decay: 0.9,
        var_decay: 0.9,
        lr: 0.1,
        min_param: 0.1,
        max_param: 0.9,
    };
    let mut vm = VarianceMinimizer::new(config);

    // Phase 1: high variance costs
    let mut variances: Vec<f32> = Vec::new();
    for _batch in 0..20 {
        for i in 0..50 {
            let cost = if i % 2 == 0 { 0.1 } else { 0.9 }; // bimodal
            vm.observe_and_adapt(cost);
        }
        variances.push(vm.variance());
    }

    println!("  Variance over batches:");
    for (i, v) in variances.iter().enumerate().take(10) {
        println!("    Batch {i:2}: {v:.4}");
    }

    // Variance should be tracked and non-zero for bimodal data
    assert!(
        vm.variance() > 0.0,
        "variance should be positive for bimodal costs"
    );
}

// ── 2. AdaptiveNoiseSchedule Variance Reduction ──────────────────

#[cfg(all(feature = "replaid_schedules", feature = "dllm"))]
#[test]
fn bench_adaptive_noise_schedule() {
    use microgpt_rs::dllm::AdaptiveNoiseSchedule;

    println!("\n🧪 AdaptiveNoiseSchedule Benchmark");
    println!("{}", "═".repeat(70));

    let mut schedule = AdaptiveNoiseSchedule::new(0.1, 0.9, 8);

    // Simulate training: record losses per block
    let n_epochs = 50;
    let mut _fixed_variance_history: Vec<f32> = Vec::new();
    let mut adaptive_variance_history: Vec<f32> = Vec::new();

    // Fixed schedule baseline
    let fixed_ratios = schedule.ratios().to_vec();

    for epoch in 0..n_epochs {
        // Simulate losses: earlier steps easier, later harder
        for (block_idx, _) in fixed_ratios.iter().enumerate() {
            let loss = 0.1 + 0.1 * block_idx as f32; // linear increasing loss
            schedule.record_step_loss(block_idx, loss);
        }

        let ratios = schedule.adapt_ratios();
        let mean_loss: f32 = ratios.iter().sum::<f32>() / ratios.len() as f32;
        let variance =
            ratios.iter().map(|r| (r - mean_loss).powi(2)).sum::<f32>() / ratios.len() as f32;

        adaptive_variance_history.push(variance);

        if epoch < 5 || epoch % 10 == 0 {
            println!(
                "  Epoch {epoch:3}: ratios = {:?}",
                ratios.iter().map(|r| format!("{r:.3}")).collect::<Vec<_>>()
            );
        }
    }

    println!("  Adaptations: {}", schedule.adaptations());
    println!(
        "  Final ratios: {:?}",
        schedule
            .ratios()
            .iter()
            .map(|r| format!("{r:.3}"))
            .collect::<Vec<_>>()
    );

    // Verify: should have adapted
    assert!(schedule.adaptations() > 0, "should have adapted");
}

// ── 3. Bandit VarianceEpsilon Convergence ────────────────────────

#[cfg(all(feature = "replaid_schedules", feature = "bandit"))]
#[test]
fn bench_variance_epsilon_convergence() {
    use microgpt_rs::pruners::{BanditSession, BanditStrategy, BernoulliEnv};

    println!("\n🧪 Bandit VarianceEpsilon Convergence Benchmark");
    println!("{}", "═".repeat(70));

    let probs = vec![0.1f32, 0.3, 0.5, 0.7, 0.9]; // arm 4 is optimal
    let episodes = 5000;

    // Compare strategies
    let strategies: Vec<(String, BanditStrategy)> = vec![
        ("UCB1".to_string(), BanditStrategy::Ucb1),
        (
            "EpsilonGreedy(0.3)".to_string(),
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.3,
                decay: 1.0,
            },
        ),
        (
            "VarianceEpsilon(0.3)".to_string(),
            BanditStrategy::VarianceEpsilon {
                epsilon: 0.3,
                var_decay: 0.99,
                lr: 0.1,
            },
        ),
    ];

    for (name, strategy) in strategies {
        let env = BernoulliEnv::new(&probs);
        let mut rng = microgpt_rs::types::Rng::new(42);
        let session = BanditSession::new(env, strategy);
        let (_events, result) = session.run(episodes, &mut rng);

        println!(
            "  {name:25}: reward={:.3}, regret={:.3}, found_optimal={}",
            result.avg_reward(),
            result.avg_regret(),
            result.found_optimal()
        );
    }
}

// ── 4. SDAR SdarLearnedBeta ──────────────────────────────────────

#[cfg(all(feature = "replaid_schedules", feature = "sdar_gate"))]
#[test]
fn bench_sdar_learned_beta() {
    use microgpt_rs::pruners::sdar_gate::{SDAR_BETA, SdarLearnedBeta};
    use std::time::Instant;

    println!("\n🧪 SDAR SdarLearnedBeta Benchmark");
    println!("{}", "═".repeat(70));

    let mut lb = SdarLearnedBeta::new(SDAR_BETA);

    let iters = 100_000;
    let start = Instant::now();
    for i in 0..iters {
        let signal = (i as f32 * 0.01).sin() * 0.5 + 0.5;
        lb.observe_and_adapt(signal);
    }
    let elapsed = start.elapsed();
    let ns_per_obs = elapsed.as_nanos() as f64 / iters as f64;

    println!("  observe_and_adapt: {ns_per_obs:.1} ns/obs");
    println!("  Initial β: {SDAR_BETA}");
    println!("  Final β: {:.4}", lb.beta());

    assert!(
        ns_per_obs < 200.0,
        "observe_and_adapt should be < 200ns, got {ns_per_obs:.1}ns"
    );
}

#[cfg(all(feature = "replaid_schedules", feature = "sdar_gate"))]
#[test]
fn bench_sdar_learned_beta_vs_fixed() {
    use microgpt_rs::pruners::sdar_gate::{SDAR_BETA, SdarLearnedBeta, sdar_gated_reward};

    println!("\n🧪 SDAR Learned β vs Fixed β Comparison");
    println!("{}", "═".repeat(70));

    // Simulate: observe gated rewards, compare variance of output
    let mut learned = SdarLearnedBeta::new(SDAR_BETA);

    let fixed_beta = SDAR_BETA;
    let n_episodes = 200;

    let mut fixed_variances = Vec::new();
    let mut learned_variances = Vec::new();
    let mut fixed_rewards = Vec::new();
    let mut learned_rewards = Vec::new();

    for i in 0usize..n_episodes {
        let gap = ((i as f32 * 0.1).sin()) * 0.5;
        let reward = 0.5 + 0.3 * (i as f32 * 0.05).sin();

        let fixed_gated = sdar_gated_reward(reward, gap, fixed_beta);
        let learned_gated = sdar_gated_reward(reward, gap, learned.beta());

        fixed_rewards.push(fixed_gated);
        learned_rewards.push(learned_gated);

        learned.observe_and_adapt(learned_gated);

        if i >= 10 {
            let start = i.saturating_sub(10);
            let fixed_var = variance(&fixed_rewards[start..=i]);
            let learned_var = variance(&learned_rewards[start..=i]);
            fixed_variances.push(fixed_var);
            learned_variances.push(learned_var);
        }
    }

    let mean_fixed_var = fixed_variances.iter().sum::<f32>() / fixed_variances.len() as f32;
    let mean_learned_var = learned_variances.iter().sum::<f32>() / learned_variances.len() as f32;

    println!("  Mean variance (fixed β={fixed_beta}): {mean_fixed_var:.4}");
    println!("  Mean variance (learned β): {mean_learned_var:.4}");
    println!("  Final learned β: {:.4}", learned.beta());
}

#[allow(dead_code)]
fn variance(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32
}
