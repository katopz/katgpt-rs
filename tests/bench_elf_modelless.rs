//! ELF Embedded Language Flows modelless benchmark — run with:
//! cargo test --features "dllm" --test bench_elf_modelless --release -- --nocapture
//!
//! Plan 079: Benchmarks two ELF-inspired modelless techniques:
//! 1. SDE noise injection for DDTree path diversity
//! 2. Logit-normal schedule for D2F step allocation
//!
//! Both are additive, feature-gated, and require GOAT proof before adoption.

#[cfg(feature = "dllm")]
use katgpt_rs::speculative::d2f::{D2fDecodeConfig, ScheduleKind};
use katgpt_rs::speculative::dd_tree::{build_dd_tree_sde, inject_sde_noise};
use katgpt_rs::speculative::types::{NoScreeningPruner, SdeConfig};
use katgpt_rs::transformer::TransformerWeights;
use katgpt_rs::types::{Config, Rng};

// ── 1. SDE Noise Injection: Marginals Perturbation ───────────────

#[test]
fn bench_sde_noise_injection_overhead() {
    let config = Config::draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    // Generate marginals via dflash
    let marginals = katgpt_rs::speculative::dflash::dflash_predict(&weights, &config, 0, 0);
    let marginals_refs: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    println!("\n🧪 SDE Noise Injection Overhead Benchmark");
    println!("{}", "═".repeat(70));

    let iters = 10_000;
    let gamma_values = [0.0, 0.5, 1.0, 2.0];

    for &gamma in &gamma_values {
        let sde_config = SdeConfig {
            gamma,
            ..Default::default()
        };

        let start = std::time::Instant::now();
        for _ in 0..iters {
            let _noisy = inject_sde_noise(&marginals_refs, &sde_config, &mut rng);
        }
        let elapsed = start.elapsed();
        let us_per_call = elapsed.as_micros() as f64 / iters as f64;

        println!("  γ={gamma:.1}: {us_per_call:.1} µs/call ({iters} iters)");
    }

    // γ=0 should be faster (early return / no noise computation)
    let sde_disabled = SdeConfig::default();
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _noisy = inject_sde_noise(&marginals_refs, &sde_disabled, &mut rng);
    }
    let disabled_us = start.elapsed().as_micros() as f64 / iters as f64;

    let sde_enabled = SdeConfig::elf_default();
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _noisy = inject_sde_noise(&marginals_refs, &sde_enabled, &mut rng);
    }
    let enabled_us = start.elapsed().as_micros() as f64 / iters as f64;

    println!("\n  Disabled (γ=0): {disabled_us:.1} µs/call");
    println!("  Enabled (γ=1):  {enabled_us:.1} µs/call");
    println!("  Overhead:       {:.1} µs", enabled_us - disabled_us);
}

// ── 2. SDE Noise: DDTree Path Diversity ──────────────────────────

#[test]
fn bench_sde_noise_path_diversity() {
    let config = Config::draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let marginals = katgpt_rs::speculative::dflash::dflash_predict(&weights, &config, 0, 0);
    let marginals_refs: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let screener = NoScreeningPruner;

    println!("\n🧪 SDE Noise: DDTree Path Diversity Benchmark");
    println!("{}", "═".repeat(70));

    let n_trials = 100;
    let gamma_values = [0.0, 0.5, 1.0, 2.0];

    for &gamma in &gamma_values {
        let sde_config = SdeConfig {
            gamma,
            ..Default::default()
        };

        let mut unique_prefixes = std::collections::HashSet::new();
        let mut total_nodes = 0usize;

        for trial in 0..n_trials {
            let mut trial_rng = Rng::new(42 + trial as u64);
            let tree = build_dd_tree_sde(
                &marginals_refs,
                &config,
                &screener,
                false,
                &sde_config,
                &mut trial_rng,
            );

            total_nodes += tree.len();

            // Count unique prefixes (first 3 tokens of each path)
            for node in &tree {
                if node.depth >= 2 {
                    let prefix = extract_prefix(node.parent_path, 3);
                    unique_prefixes.insert(prefix);
                }
            }
        }

        let diversity = unique_prefixes.len() as f64 / n_trials as f64;
        println!(
            "  γ={gamma:.1}: {} unique prefixes across {n_trials} trials ({diversity:.1} avg/trial), avg tree size: {:.0}",
            unique_prefixes.len(),
            total_nodes as f64 / n_trials as f64
        );
    }
}

fn extract_prefix(parent_path: u128, depth: usize) -> Vec<usize> {
    (0..depth)
        .map(|k| ((parent_path >> ((depth - 1 - k) * 16)) & 0xFFFF) as usize)
        .collect()
}

// ── 3. SDE Noise: Win Rate Comparison (DDTree Quality) ───────────

#[test]
fn bench_sde_noise_quality() {
    let config = Config::draft();
    let mut rng = Rng::new(42);
    let weights = TransformerWeights::new(&config, &mut rng);

    let marginals = katgpt_rs::speculative::dflash::dflash_predict(&weights, &config, 0, 0);
    let marginals_refs: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let screener = NoScreeningPruner;

    println!("\n🧪 SDE Noise: DDTree Quality Benchmark");
    println!("{}", "═".repeat(70));

    let n_trials = 200;
    let gamma_values = [0.0, 0.5, 1.0, 2.0];

    for &gamma in &gamma_values {
        let sde_config = SdeConfig {
            gamma,
            ..Default::default()
        };

        let mut total_top_prob = 0.0f32;
        let mut total_tree_size = 0usize;

        for trial in 0..n_trials {
            let mut trial_rng = Rng::new(42 + trial as u64);
            let tree = build_dd_tree_sde(
                &marginals_refs,
                &config,
                &screener,
                false,
                &sde_config,
                &mut trial_rng,
            );

            total_tree_size += tree.len();

            // Best path probability (sum of marginals along best path)
            if let Some(best) = tree.first() {
                total_top_prob += best.score.exp();
            }
        }

        let avg_top_prob = total_top_prob / n_trials as f32;
        let avg_tree_size = total_tree_size as f32 / n_trials as f32;

        println!(
            "  γ={gamma:.1}: avg top prob = {avg_top_prob:.4}, avg tree size = {avg_tree_size:.0}"
        );
    }
}

// ── 4. Logit-Normal Schedule: Step Distribution ──────────────────

#[cfg(feature = "dllm")]
#[test]
fn bench_logit_normal_schedule() {
    let mut rng = Rng::new(42);

    println!("\n🧪 Logit-Normal Schedule Benchmark");
    println!("{}", "═".repeat(70));

    let n_steps_values = [4, 8, 16, 32];

    for &n_steps in &n_steps_values {
        let uniform = ScheduleKind::Uniform;
        let logit_normal = ScheduleKind::elf_default(); // LogitNormal { mean: -1.5, std: 0.8 }

        let uniform_steps = uniform.generate_steps(n_steps, &mut rng);
        let ln_steps = logit_normal.generate_steps(n_steps, &mut rng);

        let uniform_mean = uniform_steps.iter().sum::<f32>() / n_steps as f32;
        let ln_mean = ln_steps.iter().sum::<f32>() / n_steps as f32;

        // Compute concentration: how many steps are in [0.0, 0.3] range
        let uniform_concentrated = uniform_steps.iter().filter(|&&s| s < 0.3).count();
        let ln_concentrated = ln_steps.iter().filter(|&&s| s < 0.3).count();

        println!("  n_steps={n_steps:2}:");
        println!(
            "    Uniform:    mean={uniform_mean:.3}, steps={:?}",
            uniform_steps
                .iter()
                .map(|s| format!("{s:.2}"))
                .collect::<Vec<_>>()
        );
        println!(
            "    LogitNorm:  mean={ln_mean:.3}, steps={:?}",
            ln_steps
                .iter()
                .map(|s| format!("{s:.2}"))
                .collect::<Vec<_>>()
        );
        println!(
            "    Concentrated (<0.3): uniform={uniform_concentrated}/{n_steps}, logit-normal={ln_concentrated}/{n_steps}"
        );
    }
}

// ── 5. Logit-Normal Schedule: Overhead ───────────────────────────

#[cfg(feature = "dllm")]
#[test]
fn bench_logit_normal_schedule_overhead() {
    let mut rng = Rng::new(42);

    println!("\n🧪 Logit-Normal Schedule Overhead Benchmark");
    println!("{}", "═".repeat(70));

    let iters = 100_000;
    let n_steps = 8;

    let uniform = ScheduleKind::Uniform;
    let logit_normal = ScheduleKind::elf_default();

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _steps = uniform.generate_steps(n_steps, &mut rng);
    }
    let uniform_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _steps = logit_normal.generate_steps(n_steps, &mut rng);
    }
    let ln_ns = start.elapsed().as_nanos() as f64 / iters as f64;

    println!("  Uniform:    {uniform_ns:.0} ns/call");
    println!("  LogitNorm:  {ln_ns:.0} ns/call");
    println!("  Overhead:   {:.0} ns", ln_ns - uniform_ns);

    // Logit-normal should add some overhead but be reasonable
    assert!(
        ln_ns < 10000.0,
        "logit-normal should be < 10µs/call, got {ln_ns:.0}ns"
    );
}

// ── 6. D2F Decode: Uniform vs Logit-Normal ───────────────────────

#[cfg(feature = "dllm")]
#[test]
fn bench_d2f_schedule_comparison() {
    use katgpt_rs::speculative::d2f::d2f_decode_block;
    use katgpt_rs::speculative::types::NoPruner;
    use katgpt_rs::types::Rng;

    println!("\n🧪 D2F Schedule Comparison: Uniform vs Logit-Normal");
    println!("{}", "═".repeat(70));

    let config = Config::micro_dllm();
    let n_trials = 50;

    for (name, schedule) in [
        ("Uniform", ScheduleKind::Uniform),
        ("LogitNorm(-1.5, 0.8)", ScheduleKind::elf_default()),
    ] {
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let decode_config = D2fDecodeConfig {
            schedule,
            ..D2fDecodeConfig::with_block_size(4)
        };

        let mut total_steps = 0usize;
        let mut total_confidence = 0.0f32;
        let mut fully_activated = 0usize;

        for _ in 0..n_trials {
            let result = d2f_decode_block(&weights, &config, &decode_config, &NoPruner, &mut rng);
            total_steps += result.steps_used;
            total_confidence += result.confidence_history.last().copied().unwrap_or(0.0);
            if result.state.is_fully_activated() {
                fully_activated += 1;
            }
        }

        println!(
            "  {name:25}: avg steps = {:.1}, avg final conf = {:.3}, fully activated = {fully_activated}/{n_trials}",
            total_steps as f32 / n_trials as f32,
            total_confidence / n_trials as f32
        );
    }
}
