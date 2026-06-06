//! Problem Evolution Demo — Plan 191 T4.2
//!
//! Demonstrates EvolutionArena with BomberConfigMutator and GoConfigMutator:
//! config mutation → arena run → difficulty progression → diversity metrics.
//!
//! Run: `cargo run --features "problem_mutator" --example problem_evolution_demo`

#[cfg(feature = "problem_mutator")]
use std::collections::HashSet;

#[cfg(feature = "problem_mutator")]
use katgpt_core::GameConfig;
#[cfg(feature = "problem_mutator")]
use katgpt_rs::pruners::problem_mutator::{BomberConfigMutator, EvolutionArena, GoConfigMutator};

/// A lightweight fingerprint for deduplicating configs by their observable fields.
#[cfg(feature = "problem_mutator")]
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct ConfigFingerprint {
    grid_size: u32,
    opponent_count: u32,
    max_steps: u32,
    survival_weight_bits: u32,
    kill_weight_bits: u32,
}

#[cfg(feature = "problem_mutator")]
impl From<&GameConfig> for ConfigFingerprint {
    fn from(c: &GameConfig) -> Self {
        Self {
            grid_size: c.grid_size,
            opponent_count: c.opponent_count,
            max_steps: c.max_steps,
            survival_weight_bits: c.survival_weight.to_bits(),
            kill_weight_bits: c.kill_weight.to_bits(),
        }
    }
}

#[cfg(feature = "problem_mutator")]
fn main() {
    println!("=== Plan 191: Problem Evolution Demo ===\n");

    let base = GameConfig::default();

    // ── BomberConfigMutator: 3 cycles × 5 rounds ─────────────────
    let mut bomber_arena = EvolutionArena::new(
        base.clone(),
        Box::new(BomberConfigMutator),
        5, // max_rounds_per_cycle
    );

    let mut all_bomber_configs: Vec<GameConfig> = Vec::new();

    for cycle in 1..=3 {
        println!("Cycle {cycle} — Bomber configs:");
        for i in 1..=5 {
            let cfg = bomber_arena.next_config();
            println!(
                "  Round {}: grid={}, opponents={}, steps={}, surv_w={:.2}, kill_w={:.2}",
                (cycle - 1) * 5 + i,
                cfg.grid_size,
                cfg.opponent_count,
                cfg.max_steps,
                cfg.survival_weight,
                cfg.kill_weight,
            );
            all_bomber_configs.push(cfg);
        }
        println!();
    }

    // ── GoConfigMutator: 5 rounds ─────────────────────────────────
    println!("Go Config Evolution:");
    let go_base = GameConfig {
        grid_size: 9,
        opponent_count: 1,
        max_steps: 200,
        survival_weight: 0.5,
        kill_weight: 0.5,
    };

    let go_mutator = GoConfigMutator {
        territory_weight: 0.5,
        board_sizes: vec![9, 13, 19],
    };

    let mut go_arena = EvolutionArena::new(go_base.clone(), Box::new(go_mutator), 10);

    let mut all_go_configs: Vec<GameConfig> = Vec::new();
    for round in 1..=5 {
        let cfg = go_arena.next_config();
        println!(
            "  Round {}: board={}, opponents={}, surv_w={:.2}, kill_w={:.2}",
            round, cfg.grid_size, cfg.opponent_count, cfg.survival_weight, cfg.kill_weight,
        );
        all_go_configs.push(cfg);
    }
    println!();

    // ── Diversity metrics ─────────────────────────────────────────
    let all_configs: Vec<&GameConfig> = all_bomber_configs
        .iter()
        .chain(all_go_configs.iter())
        .collect();

    let total = all_configs.len();
    let distinct: HashSet<ConfigFingerprint> = all_configs
        .iter()
        .map(|c| ConfigFingerprint::from(*c))
        .collect();

    let distinct_survival: HashSet<u32> = all_configs
        .iter()
        .map(|c| c.survival_weight.to_bits())
        .collect();
    let distinct_kill: HashSet<u32> = all_configs
        .iter()
        .map(|c| c.kill_weight.to_bits())
        .collect();
    let distinct_grid: HashSet<u32> = all_configs.iter().map(|c| c.grid_size).collect();

    println!(
        "Diversity: {} distinct configs out of {} generated",
        distinct.len(),
        total
    );
    println!(
        "  survival_weight variants: {}, kill_weight variants: {}, grid_size variants: {}",
        distinct_survival.len(),
        distinct_kill.len(),
        distinct_grid.len(),
    );

    println!("\n=== Demo Complete ===");
}

// TL;DR: Demonstrates EvolutionArena cycling through BomberConfigMutator (3×5 rounds)
// and GoConfigMutator (5 rounds), printing config parameters and diversity metrics.

#[cfg(not(feature = "problem_mutator"))]
fn main() {
    eprintln!(
        "Enable problem_mutator feature: cargo run --features problem_mutator --example problem_evolution_demo"
    );
}
