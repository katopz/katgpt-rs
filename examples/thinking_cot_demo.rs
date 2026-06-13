//! Adaptive CoT Thinking Demo — Thinking vs Non-Thinking (Plan 194 T6).
//!
//! Demonstrates the before/after quality difference between direct answering
//! and adaptive thinking with the bandit controller.
//!
//! Run with:
//!   cargo run --features thinking_cot --example thinking_cot_demo

use katgpt_rs::speculative::thinking_controller::Rng;
use katgpt_rs::speculative::{ThinkingConfig, ThinkingController, ThinkingMode, ThinkingSelector};

/// Simple xorshift32 RNG.
struct DemoRng {
    state: u32,
}

impl DemoRng {
    fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }
}

impl Rng for DemoRng {
    fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 17;
        self.state ^= self.state << 5;
        self.state
    }
}

/// Simulate answer quality for a given mode and difficulty.
fn simulate_quality(mode: ThinkingMode, difficulty: f32) -> f32 {
    let base = 1.0 - difficulty;
    match mode {
        ThinkingMode::Direct => base * 0.6 + 0.1,
        ThinkingMode::Latent => base * 0.3 + 0.55,
        ThinkingMode::CpuResample => base * 0.4 + 0.4,
        ThinkingMode::Dendritic => base * 0.35 + 0.5,
    }
}

/// Simulate normalized cost.
fn simulate_cost(mode: ThinkingMode) -> f32 {
    match mode {
        ThinkingMode::Direct => 0.1,
        ThinkingMode::Latent => 0.7,
        ThinkingMode::CpuResample => 0.2,
        ThinkingMode::Dendritic => 0.35,
    }
}

fn main() {
    println!("=== Adaptive CoT Thinking Demo ===\n");

    let difficulty = 0.8; // Hard scenario
    println!("Scenario: Hard reasoning task (difficulty = {difficulty})");
    println!("Domain: constraint-satisfaction\n");

    let hard_queries = 25;

    // --- Direct Mode ---
    println!("--- Direct Mode (no thinking) ---");
    let direct_config = ThinkingConfig {
        mode: ThinkingSelector::AlwaysDirect,
        ..Default::default()
    };
    let mut direct_ctrl = ThinkingController::new(direct_config);
    let mut rng = DemoRng::new(42);
    let mut direct_quality = 0.0f32;
    for _ in 0..hard_queries {
        let mode = direct_ctrl.select_mode(0.2, &mut rng);
        let q = simulate_quality(mode, difficulty);
        direct_ctrl.record_reward(mode, q, simulate_cost(mode));
        direct_quality += q;
    }
    let direct_avg = direct_quality / hard_queries as f32;
    println!("  Queries: {hard_queries}");
    println!("  Avg quality: {direct_avg:.3}");
    println!("  Avg cost:    0.100\n");

    // --- Latent Mode ---
    println!("--- Latent Mode (RiM buffer slots, K=8) ---");
    let latent_config = ThinkingConfig {
        mode: ThinkingSelector::AlwaysLatent,
        max_blocks: 8,
        ..Default::default()
    };
    let mut latent_ctrl = ThinkingController::new(latent_config);
    let mut rng = DemoRng::new(42);
    let mut latent_quality = 0.0f32;
    for _ in 0..hard_queries {
        let mode = latent_ctrl.select_mode(0.2, &mut rng);
        let q = simulate_quality(mode, difficulty);
        latent_ctrl.record_reward(mode, q, simulate_cost(mode));
        latent_quality += q;
    }
    let latent_avg = latent_quality / hard_queries as f32;
    let quality_gain = (latent_avg - direct_avg) / direct_avg * 100.0;
    println!("  Queries: {hard_queries}");
    println!("  Avg quality: {latent_avg:.3}");
    println!("  Avg cost:    0.700");
    println!("  Quality gain: +{quality_gain:.1}%\n");

    // --- CPU Resample Mode ---
    println!("--- CPU Resample Mode (PPoT, m=5) ---");
    let cpu_config = ThinkingConfig {
        mode: ThinkingSelector::Adaptive {
            exploration_rate: 0.0,
            dendritic_weight: 0.25,
        },
        ..Default::default()
    };
    let mut cpu_ctrl = ThinkingController::with_gpu_load(cpu_config, 0.9);
    let mut rng = DemoRng::new(42);
    let mut cpu_quality = 0.0f32;
    let mut cpu_cost = 0.0f32;
    for _ in 0..hard_queries {
        let mode = cpu_ctrl.select_mode(0.2, &mut rng);
        let q = simulate_quality(mode, difficulty);
        let c = simulate_cost(mode);
        cpu_ctrl.record_reward(mode, q, c);
        cpu_quality += q;
        cpu_cost += c;
    }
    let cpu_avg = cpu_quality / hard_queries as f32;
    let cpu_avg_cost = cpu_cost / hard_queries as f32;
    let cpu_gain = (cpu_avg - direct_avg) / direct_avg * 100.0;
    println!("  Queries: {hard_queries}");
    println!("  Avg quality: {cpu_avg:.3}");
    println!("  Avg cost:    {cpu_avg_cost:.3} (CPU only, no GPU overhead)");
    println!("  Quality gain: +{cpu_gain:.1}%\n");

    // --- Adaptive Mode ---
    println!("--- Adaptive Mode (bandit, 50 episodes) ---");
    let adaptive_config = ThinkingConfig {
        mode: ThinkingSelector::Adaptive {
            exploration_rate: 0.1,
            dendritic_weight: 0.25,
        },
        ..Default::default()
    };
    let mut adaptive_ctrl = ThinkingController::new(adaptive_config);
    let mut rng = DemoRng::new(42);
    let episodes = 50;

    let mut mode_counts = [0usize; 3];
    let mut mode_rewards = [0.0f32; 3];
    let mut total_quality = 0.0f32;
    let mut total_cost = 0.0f32;

    for i in 0..episodes {
        let difficulty = if i < 20 {
            0.2
        } else if i < 35 {
            0.5
        } else {
            0.8
        };
        let confidence = 1.0 - difficulty;

        let mode = adaptive_ctrl.select_mode(confidence, &mut rng);
        let q = simulate_quality(mode, difficulty);
        let c = simulate_cost(mode);

        adaptive_ctrl.record_reward(mode, q, c);

        let arm = mode as usize;
        mode_counts[arm] += 1;
        mode_rewards[arm] += q;
        total_quality += q;
        total_cost += c;
    }

    println!("  Episodes: {episodes}");
    println!("  Bandit learned:");
    let mode_names = ["Direct", "Latent", "CpuResample"];
    for arm in 0..3 {
        if mode_counts[arm] > 0 {
            let avg_r = mode_rewards[arm] / mode_counts[arm] as f32;
            let pct = mode_counts[arm] as f32 / episodes as f32 * 100.0;
            println!(
                "    {:12} {}/{} ({pct:.0}%) — avg reward {avg_r:.3}",
                mode_names[arm], mode_counts[arm], episodes
            );
        }
    }

    let overall_quality = total_quality / episodes as f32;
    let overall_cost = total_cost / episodes as f32;
    let efficiency = overall_quality / overall_cost;
    let direct_efficiency = direct_avg / 0.1;
    println!(
        "  Overall quality: {overall_quality:.3} (vs {direct_avg:.3} direct, +{:.1}%)",
        (overall_quality - direct_avg) / direct_avg * 100.0
    );
    println!(
        "  Overall cost:    {overall_cost:.3} (vs 0.700 always-latent, -{:.0}%)",
        (1.0 - overall_cost / 0.7) * 100.0
    );
    println!(
        "  Efficiency:      {efficiency:.2} (vs {direct_efficiency:.2} direct, {:.2}x better)",
        efficiency / direct_efficiency
    );
}
