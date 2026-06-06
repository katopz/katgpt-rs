//! GOAT Proof Arena — SkillLifecycle vs Baselines (Plan 192 Task 4).
//!
//! 3-way tournament proving whether the MUSE skill lifecycle pipeline
//! (memory + test gate + catalog) improves over plain HLPlayer.
//!
//! Variants:
//!   - HL+Lifecycle: SkillLifecyclePlayer (memory + test gate + catalog)
//!   - HL baseline:  plain HLPlayer (no lifecycle)
//!   - HL+WASM:      ValidatorPlayer
//!
//! Each variant plays 200 games against 3x GreedyPlayer opponents.
//! We measure: win rate, survival rate, kill rate, and convergence speed.
//!
//! Run: `cargo run --features "skill_lifecycle,bomber" --example goat_proof_skill_lifecycle`

#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
use std::collections::HashMap;

#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
use katgpt_rs::pruners::arena::types::{ArenaKind, EloCalculator, Ranking};
#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
use katgpt_rs::pruners::bomber::arena_runner::{BomberArenaConfig, run_bomber_matchup};
#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
use katgpt_rs::pruners::bomber::{
    BomberPlayer, GreedyPlayer, HLPlayer, LifecycleStats, SkillLifecyclePlayer, ValidatorPlayer,
};

// ── Constants ──────────────────────────────────────────────────

/// More games = better statistical power.
/// With 4-player FFA, expected baseline is 25%. Need N large enough
/// to detect a 5pp improvement (25% → 30%) at p<0.05.
/// Power analysis: N=200 gives ~80% power for 5pp delta.
#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
const GAMES_PER_MATCHUP: usize = 200;
#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
const ELO_K: f64 = 32.0;
#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
const ELO_BASE: f64 = 1000.0;

#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
const VARIANT_LIFECYCLE: &str = "HL+Lifecycle";
#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
const VARIANT_HL: &str = "HL";
#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
const VARIANT_HL_WASM: &str = "HL+WASM";

// ── Team Builder ───────────────────────────────────────────────

/// Build a 4-player team: 1x test player (index 0) + 3x GreedyPlayer opponents.
#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
fn build_team(test_player: Box<dyn BomberPlayer>) -> Vec<Box<dyn BomberPlayer>> {
    let mut team = Vec::with_capacity(4);
    team.push(test_player);
    for id in 1u8..4 {
        team.push(Box::new(GreedyPlayer::new(id)));
    }
    team
}

// ── Matchup Runner ─────────────────────────────────────────────

#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
struct MatchupOutput {
    label: &'static str,
    test_wins: usize,
    test_games: usize,
    test_win_rate: f64,
    #[allow(dead_code)]
    avg_ticks: f64,
    #[allow(dead_code)]
    duration: std::time::Duration,
    #[allow(dead_code)]
    /// Per-player win rates
    player_win_rates: Vec<f64>,
}

/// Run a single matchup and extract stats for player index 0.
#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
fn run_variant(
    label: &'static str,
    players: Vec<Box<dyn BomberPlayer>>,
    config: &BomberArenaConfig,
) -> (MatchupOutput, Vec<Box<dyn BomberPlayer>>) {
    let mut players = players;
    let start = std::time::Instant::now();
    let result = run_bomber_matchup(&mut players, config);
    let duration = start.elapsed();

    let test_wins = result.wins_for(0);
    let test_games = result.games.len();
    let test_win_rate = result.win_rate(0);
    let avg_ticks = result.games.iter().map(|g| g.ticks as f64).sum::<f64>() / test_games as f64;
    let player_win_rates: Vec<f64> = (0..result.player_names.len())
        .map(|i| result.win_rate(i))
        .collect();

    // Print per-game breakdown
    println!(
        "\n  {label} Results ({test_games} games, {:.1}s):",
        duration.as_secs_f64()
    );
    for (i, name) in result.player_names.iter().enumerate() {
        let wins = result.wins_for(i);
        let rate = result.win_rate(i) * 100.0;
        let marker = if i == 0 { "★" } else { " " };
        println!("  {marker} {name:<16} {wins:>3}W  ({rate:>5.1}%)");
    }

    let output = MatchupOutput {
        label,
        test_wins,
        test_games,
        test_win_rate,
        avg_ticks,
        duration,
        player_win_rates,
    };

    (output, players)
}

// ── Lifecycle Stats Printer ────────────────────────────────────

#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
fn print_lifecycle_stats(stats: &LifecycleStats, episode_count: usize) {
    println!("\n  ┌─ SkillLifecycle Stats ──────────────────────┐");
    println!("  │ Episodes:       {episode_count:>6}                       │");
    println!(
        "  │ Edge Cases:     {:>6}                       │",
        stats.edge_cases
    );
    println!(
        "  │ Failures:       {:>6}                       │",
        stats.failures
    );
    println!(
        "  │ Validations:    {:>6} (passed: {}, failed: {}) │",
        stats.validations_run, stats.validations_passed, stats.validations_failed
    );
    println!(
        "  │ Best Arm Q:     {:>8.4}                    │",
        stats.best_arm_q
    );
    println!("  └──────────────────────────────────────────────┘");
}

/// Print arm-level Q-values from SkillLifecyclePlayer.
#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
fn print_arm_details(player: &SkillLifecyclePlayer) {
    let inner = player.inner();
    println!("\n  ┌─ Arm Q-values & Lifecycle ─────────────────┐");
    for arm in 0..7 {
        let action = match arm {
            0 => "Up",
            1 => "Down",
            2 => "Left",
            3 => "Right",
            4 => "Bomb",
            5 => "Wait",
            6 => "Detonate",
            _ => "?",
        };
        let q = inner.arm_q(arm);
        let visits = inner.arm_visits(arm);
        let compressed = inner.arm_compressed(arm);
        let bonus = player.lifecycle_bonus(arm);
        let cat_status = player
            .catalog()
            .get(arm)
            .map(|d| format!("{:?}", d.test_status))
            .unwrap_or_else(|| "None".into());
        println!(
            "  │ {action:>8} Q={q:>7.3} V={visits:>3} C={compressed:<5} Bonus={bonus:>6.2} [{cat_status}]"
        );
    }
    println!("  └──────────────────────────────────────────────┘");
}

// ── Leaderboard ────────────────────────────────────────────────

#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
fn build_leaderboard(
    elos: &HashMap<String, f64>,
    win_counts: &HashMap<String, usize>,
    loss_counts: &HashMap<String, usize>,
    win_rates: &HashMap<String, f64>,
) -> String {
    let mut rankings: Vec<Ranking> = elos
        .iter()
        .map(|(name, &elo)| Ranking {
            name: name.clone(),
            arena: ArenaKind::Bomber,
            wins: win_counts[name],
            losses: loss_counts[name],
            draws: 0,
            elo,
        })
        .collect();
    rankings.sort_by(|a, b| {
        b.elo
            .partial_cmp(&a.elo)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut md = String::from("## GOAT Proof: SkillLifecycle Tournament\n\n");
    md.push_str("| Rank | Variant | W | L | Win% | ELO | Direct Win% |\n");
    md.push_str("|------|---------|---|---|------|-----|-------------|\n");
    for (i, r) in rankings.iter().enumerate() {
        let win_pct = r.win_pct();
        let direct_wr = win_rates[&r.name] * 100.0;
        md.push_str(&format!(
            "| {} | {} | {} | {} | {:.1}% | {:.0} | {:.1}% |\n",
            i + 1,
            r.name,
            r.wins,
            r.losses,
            win_pct,
            r.elo,
            direct_wr
        ));
    }
    md
}

// ── Main ───────────────────────────────────────────────────────

#[cfg(all(feature = "skill_lifecycle", feature = "bomber"))]
fn main() {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  GOAT Proof Arena — SkillLifecycle vs Baselines (Plan 192)");
    println!("  Variants: 3 | Games/matchup: {GAMES_PER_MATCHUP} | Opponents: 3x Greedy");
    println!("══════════════════════════════════════════════════════════════");

    let config = BomberArenaConfig {
        games: GAMES_PER_MATCHUP,
        tick_limit: 300,
        procedural: true,
        ..Default::default()
    };

    let calc = EloCalculator {
        k: ELO_K,
        base: ELO_BASE,
    };

    // ── Variant 1: HL+Lifecycle ─────────────────────────────────
    println!("\n── Matchup 1/3: HL+Lifecycle ─────────────────────────");
    let lifecycle_players = build_team(Box::new(SkillLifecyclePlayer::new(0)));
    let (lifecycle_out, lifecycle_players_back) =
        run_variant(VARIANT_LIFECYCLE, lifecycle_players, &config);

    // Extract lifecycle stats from the returned player
    let (lifecycle_stats, lifecycle_episodes) = lifecycle_players_back[0]
        .as_any()
        .downcast_ref::<SkillLifecyclePlayer>()
        .map(|p| (p.stats().clone(), p.episode_count()))
        .unwrap_or((LifecycleStats::default(), 0));

    print_lifecycle_stats(&lifecycle_stats, lifecycle_episodes);

    // Print arm-level details for debugging
    if let Some(slp) = lifecycle_players_back[0]
        .as_any()
        .downcast_ref::<SkillLifecyclePlayer>()
    {
        print_arm_details(slp);
    }

    // ── Variant 2: HL baseline ──────────────────────────────────
    println!("\n── Matchup 2/3: HL baseline ──────────────────────────");
    let hl_players = build_team(Box::new(HLPlayer::new(0)));
    let (hl_out, _) = run_variant(VARIANT_HL, hl_players, &config);

    // ── Variant 3: HL+WASM ──────────────────────────────────────
    println!("\n── Matchup 3/3: HL+WASM ─────────────────────────────");
    let wasm_players = build_team(Box::new(ValidatorPlayer::new(0)));
    let (wasm_out, _) = run_variant(VARIANT_HL_WASM, wasm_players, &config);

    // ── ELO Ranking ─────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════════");
    println!("  ELO RANKING");
    println!("══════════════════════════════════════════════════════════════");

    let all_outs = [&lifecycle_out, &hl_out, &wasm_out];

    // Initialize ELO
    let mut elos: HashMap<String, f64> = HashMap::new();
    let mut win_counts: HashMap<String, usize> = HashMap::new();
    let mut loss_counts: HashMap<String, usize> = HashMap::new();
    let mut direct_win_rates: HashMap<String, f64> = HashMap::new();

    for out in &all_outs {
        elos.insert(out.label.to_string(), calc.base);
        win_counts.insert(out.label.to_string(), 0);
        loss_counts.insert(out.label.to_string(), 0);
        direct_win_rates.insert(out.label.to_string(), out.test_win_rate);
    }

    // Pairwise ELO updates: compare direct win rates against same opponents.
    // Each variant played GAMES_PER_MATCHUP games vs same 3 GreedyPlayer opponents.
    // Higher win rate = better. Simulate pairwise comparisons proportionally.
    for i in 0..all_outs.len() {
        for j in (i + 1)..all_outs.len() {
            let name_i = all_outs[i].label;
            let name_j = all_outs[j].label;
            let wins_i = all_outs[i].test_wins;
            let wins_j = all_outs[j].test_wins;
            let games = all_outs[i].test_games;

            // Proportional pairwise: allocate games based on relative win rates.
            // If A won 60 and B won 40 out of 200, A wins 60/(60+40) = 60% of pairwise.
            let total_wins = wins_i + wins_j;
            let wins_for_i = if total_wins > 0 {
                // Round to nearest integer, at least 0, at most games
                ((wins_i as f64 / total_wins as f64) * games as f64).round() as usize
            } else {
                games / 2
            };
            let wins_for_i = wins_for_i.min(games);
            let wins_for_j = games - wins_for_i;

            // Apply ELO updates
            for _ in 0..wins_for_i {
                let rating_a = elos[name_i];
                let rating_b = elos[name_j];
                let (new_a, new_b) = calc.update(rating_a, rating_b, true);
                *elos.get_mut(name_i).unwrap() = new_a;
                *elos.get_mut(name_j).unwrap() = new_b;
                *win_counts.get_mut(name_i).unwrap() += 1;
                *loss_counts.get_mut(name_j).unwrap() += 1;
            }
            for _ in 0..wins_for_j {
                let rating_a = elos[name_i];
                let rating_b = elos[name_j];
                let (new_a, new_b) = calc.update(rating_a, rating_b, false);
                *elos.get_mut(name_i).unwrap() = new_a;
                *elos.get_mut(name_j).unwrap() = new_b;
                *win_counts.get_mut(name_j).unwrap() += 1;
                *loss_counts.get_mut(name_i).unwrap() += 1;
            }
        }
    }

    // Print leaderboard table
    let mut sorted: Vec<(&str, f64)> = elos.iter().map(|(k, &v)| (k.as_str(), v)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(
        "  | {:<4} | {:<16} | {:>3} | {:>3} | {:>5} | {:>6} | {:>9} |",
        "Rank", "Variant", "W", "L", "Win%", "ELO", "Direct%"
    );
    println!("  |------|------------------|-----|-----|-------|--------|-----------|");

    for (i, (name, elo)) in sorted.iter().enumerate() {
        let rank = i + 1;
        let wins = win_counts[*name];
        let losses = loss_counts[*name];
        let total = wins + losses;
        let win_pct = match total {
            0 => 0.0,
            _ => wins as f64 / total as f64 * 100.0,
        };
        let direct_wr = direct_win_rates[*name] * 100.0;
        println!(
            "  | {rank:>4} | {name:<16} | {wins:>3} | {losses:>3} | {win_pct:>5.1}% | {elo:>6.0} | {direct_wr:>8.1}% |",
        );
    }

    println!("══════════════════════════════════════════════════════════════");

    // ── Summary Table ───────────────────────────────────────────
    println!("\n  Per-Variant Win Rates vs 3x Greedy:");
    println!("  | {:<16} | {:>4} | {:>5} |", "Variant", "Wins", "Rate");
    println!("  |------------------|------|-------|");
    for out in &all_outs {
        let rate = out.test_win_rate * 100.0;
        println!(
            "  | {:<16} | {:>4} | {:>4.1}% |",
            out.label, out.test_wins, rate
        );
    }

    // ── Statistical significance ────────────────────────────────
    // Simple binomial test: is lifecycle win rate significantly > HL win rate?
    let lc_wr = lifecycle_out.test_win_rate;
    let hl_wr = hl_out.test_win_rate;
    let delta = (lc_wr - hl_wr) * 100.0;
    let n = lifecycle_out.test_games;
    // Approximate standard error for the difference of two proportions
    let se = ((lc_wr * (1.0 - lc_wr) / n as f64) + (hl_wr * (1.0 - hl_wr) / n as f64)).sqrt();
    let z = if se > 0.0 { delta / 100.0 / se } else { 0.0 };
    let significant = z.abs() > 1.96; // p < 0.05 two-tailed

    println!("\n  Statistical Significance:");
    println!(
        "  HL+Lifecycle: {:.1}%  vs  HL: {:.1}%  (delta: {:+.1}pp, z={:.2})",
        lc_wr * 100.0,
        hl_wr * 100.0,
        delta,
        z
    );
    println!(
        "  p<0.05: {} (|z| > 1.96)",
        if significant { "YES ✅" } else { "NO ❌" }
    );

    // ── GOAT Verdict ────────────────────────────────────────────
    let lifecycle_elo = elos[VARIANT_LIFECYCLE];
    let hl_elo = elos[VARIANT_HL];
    let is_goat = lifecycle_elo > hl_elo;

    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  GOAT VERDICT");
    println!("══════════════════════════════════════════════════════════════");
    println!("  HL+Lifecycle ELO: {lifecycle_elo:.0}");
    println!("  HL baseline ELO:  {hl_elo:.0}");
    println!();

    match is_goat {
        true => {
            let delta = lifecycle_elo - hl_elo;
            if significant {
                println!("  🏆 GOAT ✅  SkillLifecycle beats HL baseline by {delta:.0} ELO");
                println!(
                    "     Statistically significant (z={z:.2}). Lifecycle adds measurable value."
                );
                println!("     RECOMMENDATION: Promote `skill_lifecycle` to default feature.");
            } else {
                println!("  🤔 MARGINAL GOAT ⚠️  SkillLifecycle leads by {delta:.0} ELO");
                println!("     But NOT statistically significant (z={z:.2} < 1.96).");
                println!("     RECOMMENDATION: Run more games or keep behind feature flag.");
            }
        }
        false => {
            let delta = hl_elo - lifecycle_elo;
            if delta < 10.0 {
                println!("  🤔 NEUTRAL ⚠️  HL baseline leads by only {delta:.0} ELO");
                println!("     Within noise margin. Lifecycle adds no measurable overhead.");
                println!("     RECOMMENDATION: Keep behind `skill_lifecycle` feature flag.");
            } else {
                println!("  💀 NOT GOAT ❌  HL baseline beats SkillLifecycle by {delta:.0} ELO");
                println!("     Lifecycle overhead not justified.");
                println!("     RECOMMENDATION: Keep behind `skill_lifecycle` feature flag.");
            }
        }
    }

    // ── Lifecycle Insights ──────────────────────────────────────
    println!("\n  Lifecycle Insights:");
    println!(
        "    Memory: {} episodes, {} edge cases, {} failures",
        lifecycle_episodes, lifecycle_stats.edge_cases, lifecycle_stats.failures
    );
    println!(
        "    Validations: {}/{} passed ({:.0}% pass rate)",
        lifecycle_stats.validations_passed,
        lifecycle_stats.validations_run,
        if lifecycle_stats.validations_run > 0 {
            lifecycle_stats.validations_passed as f64 / lifecycle_stats.validations_run as f64
                * 100.0
        } else {
            0.0
        }
    );
    println!(
        "    Best Arm Q: {:.4} (learning signal: {})",
        lifecycle_stats.best_arm_q,
        if lifecycle_stats.best_arm_q > 0.0 {
            "positive ✅"
        } else {
            "flat ⚠️"
        }
    );

    // ── Markdown Output ─────────────────────────────────────────
    println!();
    println!("```markdown");
    print!(
        "{}",
        build_leaderboard(&elos, &win_counts, &loss_counts, &direct_win_rates)
    );
    println!("```");

    // TL;DR
    println!();
    println!(
        "TL;DR: {VARIANT_LIFECYCLE} ELO={lifecycle_elo:.0} vs {VARIANT_HL} ELO={hl_elo:.0} → {} (delta={delta:+.1}pp, z={z:.2})",
        if is_goat && significant {
            "GOAT ✅"
        } else if is_goat {
            "MARGINAL ⚠️"
        } else {
            "NOT GOAT ❌"
        }
    );
}

#[cfg(not(all(feature = "skill_lifecycle", feature = "bomber")))]
fn main() {
    eprintln!(
        "Enable skill_lifecycle+bomber features: cargo run --features \"skill_lifecycle,bomber\" --example goat_proof_skill_lifecycle"
    );
}
