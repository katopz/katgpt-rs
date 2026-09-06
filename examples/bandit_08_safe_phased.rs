//! Safe-Phased Bandit Demo — PrudentBanker Delay-Calibrated Exploration Safety
//!
//! Demonstrates the SafePhased strategy (Plan 137) on a 5-armed Bernoulli bandit:
//! - Arm 0: safe baseline (mean=0.5)
//! - Arm 3: best arm (mean=0.8)
//! - Arms 1, 2, 4: suboptimal (means 0.3, 0.4, 0.6)
//!
//! Compares UCB1 vs SafePhased vs ε-greedy, showing:
//! - α escalation over time (phased aggression)
//! - Baseline regret stays bounded (safe mixture)
//! - Competitive performance with unconstrained strategies
//!
//! Run: `cargo run --example bandit_08_safe_phased --features safe_bandit`

// Stub main when feature is not enabled.
#[cfg(not(feature = "safe_bandit"))]
fn main() {
    eprintln!("This example requires --features safe_bandit");
}

#[cfg(feature = "safe_bandit")]
fn main() {
    use katgpt_rs::pruners::{BanditSession, BanditStrategy, BernoulliEnv};
    use katgpt_rs::types::Rng;

    const EPISODES: usize = 2000;
    const SEED: u64 = 42;
    const BASELINE_ARM: usize = 0;
    const DELTA: f32 = 0.2;
    const ESTIMATED_DELAY: u32 = 0;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    Safe-Phased Bandit — PrudentBanker Exploration Safety     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Strategy: phased aggression with safe baseline mixture.");
    println!("  α escalates geometrically only when evidence certifies");
    println!("  the baseline arm is suboptimal. Delay-calibrated slack ξ(D)");
    println!("  prevents premature aggression under delayed feedback.");
    println!();

    let probs = [0.5f32, 0.3, 0.4, 0.8, 0.6];
    println!("🎯 Bernoulli Bandit Environment:");
    println!("   Arms:     {}", probs.len());
    println!(
        "   Means:    [{}]",
        probs
            .iter()
            .map(|p| format!("{p:.1}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let optimal = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
        .unwrap();
    println!("   Optimal:  Arm {} (mean={:.1})", optimal.0, optimal.1);
    println!(
        "   Baseline: Arm {BASELINE_ARM} (mean={:.1}) ← safe fallback",
        probs[BASELINE_ARM]
    );
    println!("   Episodes: {EPISODES}");
    println!("   δ={DELTA}, D̂={ESTIMATED_DELAY}");
    println!();

    let strategies = vec![
        ("UCB1".to_string(), BanditStrategy::Ucb1),
        (
            "ε-greedy".to_string(),
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.1,
                decay: 0.999,
            },
        ),
        (
            format!("SafePhased(base={BASELINE_ARM}, δ={DELTA}, D̂={ESTIMATED_DELAY})"),
            BanditStrategy::SafePhased {
                baseline_arm: BASELINE_ARM,
                delta: DELTA,
                estimated_delay: ESTIMATED_DELAY,
            },
        ),
    ];

    let mut all_results = Vec::with_capacity(strategies.len());

    for (name, strategy) in &strategies {
        let env = BernoulliEnv::new(&probs);
        let session = BanditSession::new(env, strategy.clone());
        let (events, result) = session.run(EPISODES, &mut Rng::new(SEED));

        // Extract regret curve
        let mut regret_curve = Vec::new();
        let mut reward_curve = Vec::new();
        for event in &events {
            if let katgpt_rs::pruners::BanditEvent::EpisodeComplete {
                cumulative_reward,
                cumulative_regret,
                ..
            } = event
            {
                regret_curve.push(*cumulative_regret);
                reward_curve.push(*cumulative_reward);
            }
        }

        let total_visits: u32 = result.visits.iter().sum();
        let baseline_visits = result.visits[BASELINE_ARM];
        let baseline_pct = baseline_visits as f32 / total_visits as f32 * 100.0;

        println!("┌─ {name} ─────────────────────────────────────────");
        println!(
            "│ Best arm:     {} (optimal: {})",
            result.best_arm, result.optimal_arm
        );
        println!(
            "│ Found optimal: {}",
            if result.found_optimal() { "✅" } else { "❌" }
        );
        println!("│ Total reward:  {:.1}", result.total_reward);
        println!("│ Total regret:  {:.2}", result.total_regret);
        println!("│ Avg reward:    {:.4}", result.avg_reward());
        println!("│");
        println!("│ Baseline arm ({BASELINE_ARM}):");
        println!("│   Visits: {baseline_visits} ({baseline_pct:.1}% of total)");
        println!(
            "│   Q-value: {:.4} (true mean: {:.1})",
            result.q_values[BASELINE_ARM], probs[BASELINE_ARM]
        );
        println!("│");
        println!("│ Q-values:");
        for (i, &q) in result.q_values.iter().enumerate() {
            let marker = if i == result.optimal_arm {
                " ← optimal"
            } else if i == BASELINE_ARM {
                " ← baseline"
            } else {
                ""
            };
            let bar_len = (q * 40.0) as usize;
            let bar: String = "█".repeat(bar_len);
            println!("│   Arm {i}: {q:.4} {bar}{marker}");
        }
        println!("│");
        println!("│ Visit counts:");
        for (i, &v) in result.visits.iter().enumerate() {
            let pct = v as f32 / total_visits as f32 * 100.0;
            let bar_len = (pct / 2.5) as usize;
            let bar: String = "█".repeat(bar_len);
            let marker = if i == result.optimal_arm {
                " ← optimal"
            } else if i == BASELINE_ARM {
                " ← baseline"
            } else {
                ""
            };
            println!("│   Arm {i}: {v:>4} ({pct:>5.1}%) {bar}{marker}");
        }
        println!("└────────────────────────────────────────────────────");
        println!();

        all_results.push((
            name.clone(),
            regret_curve,
            reward_curve,
            result.total_regret,
            result.total_reward,
            result.optimal_arm,
            result.best_arm,
        ));
    }

    // Regret comparison table
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              Cumulative Regret Comparison                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let checkpoints: [usize; 7] = [100, 250, 500, 1000, 1500, 1750, 2000];
    let checkpoint_labels: Vec<String> = checkpoints.map(|c| format!("{c}")).to_vec();

    print!("{:<40}", "Episode");
    for label in &checkpoint_labels {
        print!("{label:>8}");
    }
    println!();
    println!("{}", "─".repeat(40 + checkpoint_labels.len() * 8));

    for (name, regret_curve, _, _, _, _, _) in &all_results {
        print!("{name:<40}");
        for &cp in &checkpoints {
            let idx = cp.saturating_sub(1);
            if idx < regret_curve.len() {
                print!("{:>8.1}", regret_curve[idx]);
            } else {
                print!("{:>8}", "—");
            }
        }
        println!();
    }
    println!();

    // Baseline regret analysis
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Baseline Regret Analysis (vs Arm 0 = 0.5)           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  \"Baseline regret\" = how much worse each strategy does vs");
    println!(
        "  always playing arm 0 (the safe baseline, mean={:.1}).",
        probs[BASELINE_ARM]
    );
    println!("  If baseline regret stays bounded, SafePhased is doing its job.",);
    println!();

    let baseline_reward_per_ep = probs[BASELINE_ARM];
    let bl_checkpoints: [usize; 6] = [100, 250, 500, 1000, 1500, 2000];

    print!("{:<40}", "Episode");
    for &cp in &bl_checkpoints {
        print!("{cp:>8}");
    }
    println!();
    println!("{}", "─".repeat(40 + bl_checkpoints.len() * 8));

    for (name, _, reward_curve, _, _, _, _) in &all_results {
        print!("{name:<40}");
        for &cp in &bl_checkpoints {
            let idx = cp.saturating_sub(1);
            if idx < reward_curve.len() {
                let baseline_total = baseline_reward_per_ep * cp as f32;
                let baseline_regret = baseline_total - reward_curve[idx];
                print!("{baseline_regret:>8.1}");
            } else {
                print!("{:>8}", "—");
            }
        }
        println!();
    }
    println!();

    // ASCII regret plot
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Regret Growth (ASCII Plot)                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let width = 60;
    let height = 12;

    let max_regret = all_results
        .iter()
        .flat_map(|(_, r, _, _, _, _, _)| r.iter().copied())
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1.0);

    let symbols = ['●', '■', '▲'];
    let mut grid = vec![vec![' '; width]; height];

    #[allow(clippy::needless_range_loop)]
    for row in 0..height {
        let val = max_regret * (1.0 - row as f32 / (height - 1) as f32);
        print!("{val:>7.1} │");
        #[allow(clippy::needless_range_loop)]
        for col in 0..width {
            let episode = (col as f32 / (width - 1) as f32 * (EPISODES - 1) as f32) as usize;
            for (si, (_, regret_curve, _, _, _, _, _)) in all_results.iter().enumerate() {
                if episode < regret_curve.len() {
                    let regret = regret_curve[episode];
                    let y = ((1.0 - regret / max_regret) * (height - 1) as f32) as usize;
                    if y == row {
                        grid[row][col] = symbols[si % symbols.len()];
                    }
                }
            }
            print!("{}", grid[row][col]);
        }
        println!();
    }

    // X-axis
    print!("        └");
    for _ in 0..width {
        print!("─");
    }
    println!();
    print!("         ");
    for i in 0..=4 {
        let ep = i * EPISODES / 4;
        print!("{ep:<15}");
    }
    println!("  episodes");
    println!();

    // Legend
    for (i, (name, _, _, _, _, _, _)) in all_results.iter().enumerate() {
        println!("  {} = {}", symbols[i % symbols.len()], name);
    }
    println!();

    // Summary
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                      Summary                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut best_regret = f32::MAX;
    let mut best_name = String::new();

    for (name, _, _, total_regret, _, _, _) in &all_results {
        if *total_regret < best_regret {
            best_regret = *total_regret;
            best_name = name.clone();
        }
    }

    println!("  Lowest final regret: {best_name} ({best_regret:.2})");
    println!();
    println!("  Key takeaways:");
    println!("  • SafePhased: bounded regret vs baseline, competitive with UCB1");
    println!("  • α escalation: starts conservative (≈1/R̂), doubles each phase");
    println!("  • When baseline is suboptimal, phases escalate → α → 1 → full UCB1");
    println!("  • Baseline regret stays O(1) — the safety guarantee");
    println!("  • Delay slack ξ(D) prevents premature aggression under delayed feedback");
    println!();

    println!("  Optimal arm found:");
    for (name, _, _, _, _, optimal_arm, best_arm) in &all_results {
        let status = if best_arm == optimal_arm {
            "✅"
        } else {
            "❌"
        };
        println!("    {status} {name} → arm {best_arm} (optimal: {optimal_arm})");
    }
    println!();
}
