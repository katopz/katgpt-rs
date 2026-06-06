//! SDPG Oracle Teacher Q Generation + GOAT Gate (Plan 180, Phase 9).
//!
//! Phase 1 (Burn-in): Run SDPG + HL + Greedy + Validator for 500 rounds.
//!   SDPG starts with uniform teacher Q, learns from game outcomes via its bandit.
//!   After burn-in, extract SDPG's learned per-template Q-values as oracle teacher Q.
//!
//! Phase 2 (GOAT Gate): Run fresh SDPG initialized with oracle teacher Q
//!   vs HL + Greedy + Validator for 50 games. Verify SDPG(oracle) > HL.
//!
//! Run: `cargo run --example bomber_19_sdpg_replay_gen --features sdpg_bandit,bomber`

// Inner module gates all code behind the required features.
#[cfg(all(feature = "sdpg_bandit", feature = "bomber"))]
mod inner {

    use katgpt_rs::pruners::bomber::arena_runner::{BomberArenaConfig, run_bomber_matchup};
    use katgpt_rs::pruners::bomber::sdpg_player::SdpgPlayer;
    use katgpt_rs::pruners::bomber::{BomberPlayer, GreedyPlayer, HLPlayer, ValidatorPlayer};

    // ── Constants ──────────────────────────────────────────────────

    /// Burn-in rounds: SDPG learns from self-play.
    const BURNIN_ROUNDS: usize = 500;

    /// GOAT gate verification games.
    const GOAT_GAMES: usize = 50;

    // ── Helpers ────────────────────────────────────────────────────

    /// Run a matchup and return (wins_per_slot, total_games, duration).
    fn run_matchup(
        players: &mut Vec<Box<dyn BomberPlayer>>,
        config: &BomberArenaConfig,
    ) -> (Vec<usize>, usize, std::time::Duration) {
        let start = std::time::Instant::now();
        let result = run_bomber_matchup(players, config);
        let duration = start.elapsed();

        let total = result.games.len();
        let mut wins = vec![0usize; 4];
        for game in &result.games {
            if let Some(w) = game.winner {
                wins[w] += 1;
            }
        }
        (wins, total, duration)
    }

    /// Print win stats for a 4-player matchup.
    fn print_stats(
        names: &[&str],
        emojis: &[&str],
        wins: &[usize],
        total: usize,
        duration: std::time::Duration,
    ) {
        let secs = duration.as_secs_f64();
        println!("\n  Results ({total} games, {secs:.1}s):");
        let mut indexed: Vec<(usize, usize)> = wins.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        for (idx, win_count) in indexed {
            let name = names[idx];
            let emoji = emojis[idx];
            let losses = total.saturating_sub(win_count);
            let pct = win_count as f64 / total as f64 * 100.0;
            println!("    {emoji} {name:<12} {win_count:>3}W {losses:>3}L  ({pct:>5.1}%)");
        }
    }

    // ── Main ───────────────────────────────────────────────────────

    pub fn main() {
        println!("══════════════════════════════════════════════════════════════");
        println!("  SDPG Oracle Teacher Q Generation (Plan 180, Phase 9)");
        println!("══════════════════════════════════════════════════════════════");

        let config = BomberArenaConfig {
            games: BURNIN_ROUNDS,
            tick_limit: 300,
            procedural: true,
            ..Default::default()
        };

        // ── Phase 1: Burn-in ─────────────────────────────────────────
        println!("\nPhase 1: Burn-in ({BURNIN_ROUNDS} rounds, SDPG learning from self-play)");
        println!("─────────────────────────────────────────────────────────");

        let burnin_names = ["SDPG", "HL", "Greedy", "Validator"];
        let burnin_emojis = ["🎓", "🐵", "🐱", "🐶"];
        println!(
            "  {} {} · {} {} · {} {} · {} {}",
            burnin_emojis[0],
            burnin_names[0],
            burnin_emojis[1],
            burnin_names[1],
            burnin_emojis[2],
            burnin_names[2],
            burnin_emojis[3],
            burnin_names[3],
        );

        let mut burnin_players: Vec<Box<dyn BomberPlayer>> = vec![
            Box::new(SdpgPlayer::new(0)),
            Box::new(HLPlayer::new(1)),
            Box::new(GreedyPlayer::new(2)),
            Box::new(ValidatorPlayer::new(3)),
        ];

        let (burnin_wins, burnin_total, burnin_dur) = run_matchup(&mut burnin_players, &config);
        print_stats(
            &burnin_names,
            &burnin_emojis,
            &burnin_wins,
            burnin_total,
            burnin_dur,
        );

        // Extract SDPG's learned per-template Q-values
        let sdpg_player = burnin_players[0]
            .as_any()
            .downcast_ref::<SdpgPlayer>()
            .expect("P0 should be SdpgPlayer");
        let teacher_q = sdpg_player.sdpg_bandit().q_values().to_vec();
        let visits = sdpg_player.sdpg_bandit().visits().to_vec();
        let beta = sdpg_player.sdpg_bandit().beta();

        println!("\n  SDPG learned template Q-values (β={beta:.3}):");
        for (i, (&q, &v)) in teacher_q.iter().zip(visits.iter()).enumerate() {
            let bar_len = ((q.abs() * 10.0).clamp(0.0, 30.0)) as usize;
            let bar = "█".repeat(bar_len);
            let sign = if q >= 0.0 { "+" } else { "" };
            println!("    Template {i}: Q={sign}{q:.4}  visits={v:>4}  {bar}");
        }

        // ── Phase 2: GOAT Gate ───────────────────────────────────────
        println!("\nPhase 2: GOAT Gate ({GOAT_GAMES} games, SDPG(oracle) vs baselines)");
        println!("─────────────────────────────────────────────────────────");

        let goat_config = BomberArenaConfig {
            games: GOAT_GAMES,
            ..config.clone()
        };

        // --- Gate A: SDPG(oracle) vs HL + Greedy + Validator ---
        let gate_a_names = ["SDPG(oracle)", "HL", "Greedy", "Validator"];
        let gate_a_emojis = ["🎓", "🐵", "🐱", "🐶"];
        println!(
            "\n  Gate A: {} {} · {} {} · {} {} · {} {}",
            gate_a_emojis[0],
            gate_a_names[0],
            gate_a_emojis[1],
            gate_a_names[1],
            gate_a_emojis[2],
            gate_a_names[2],
            gate_a_emojis[3],
            gate_a_names[3],
        );

        let mut gate_a_players: Vec<Box<dyn BomberPlayer>> = vec![
            Box::new(SdpgPlayer::with_teacher_q(0, teacher_q.clone())),
            Box::new(HLPlayer::new(1)),
            Box::new(GreedyPlayer::new(2)),
            Box::new(ValidatorPlayer::new(3)),
        ];

        let (gate_a_wins, gate_a_total, gate_a_dur) =
            run_matchup(&mut gate_a_players, &goat_config);
        print_stats(
            &gate_a_names,
            &gate_a_emojis,
            &gate_a_wins,
            gate_a_total,
            gate_a_dur,
        );

        // --- Gate B: SDPG(uniform) vs HL + Greedy + Validator (control) ---
        let gate_b_names = ["SDPG(uniform)", "HL", "Greedy", "Validator"];
        let gate_b_emojis = ["🎓", "🐵", "🐱", "🐶"];
        println!(
            "\n  Gate B (control): {} {} · {} {} · {} {} · {} {}",
            gate_b_emojis[0],
            gate_b_names[0],
            gate_b_emojis[1],
            gate_b_names[1],
            gate_b_emojis[2],
            gate_b_names[2],
            gate_b_emojis[3],
            gate_b_names[3],
        );

        let mut gate_b_players: Vec<Box<dyn BomberPlayer>> = vec![
            Box::new(SdpgPlayer::new(0)),
            Box::new(HLPlayer::new(1)),
            Box::new(GreedyPlayer::new(2)),
            Box::new(ValidatorPlayer::new(3)),
        ];

        let (gate_b_wins, gate_b_total, gate_b_dur) =
            run_matchup(&mut gate_b_players, &goat_config);
        print_stats(
            &gate_b_names,
            &gate_b_emojis,
            &gate_b_wins,
            gate_b_total,
            gate_b_dur,
        );

        // ── Win Rate Comparison ──────────────────────────────────────
        println!("\n  Win Rate Comparison:");
        println!("─────────────────────────────────────────────────────────");

        let oracle_wr = gate_a_wins[0] as f64 / gate_a_total as f64;
        let uniform_wr = gate_b_wins[0] as f64 / gate_b_total as f64;
        let hl_vs_oracle_wr = gate_a_wins[1] as f64 / gate_a_total as f64;
        let hl_vs_uniform_wr = gate_b_wins[1] as f64 / gate_b_total as f64;

        println!(
            "    SDPG(oracle)  win rate: {:>5.1}%  (Gate A, {GOAT_GAMES} games)",
            oracle_wr * 100.0
        );
        println!(
            "    SDPG(uniform) win rate: {:>5.1}%  (Gate B, control)",
            uniform_wr * 100.0
        );
        println!(
            "    HL (vs oracle)  wr:     {:>5.1}%",
            hl_vs_oracle_wr * 100.0
        );
        println!(
            "    HL (vs uniform) wr:     {:>5.1}%",
            hl_vs_uniform_wr * 100.0
        );

        // ── GOAT Verdict ─────────────────────────────────────────────
        println!("\n══════════════════════════════════════════════════════════════");
        println!("  GOAT GATE VERDICT");
        println!("══════════════════════════════════════════════════════════════");

        let oracle_beats_hl = gate_a_wins[0] > gate_a_wins[1];
        let oracle_improves = gate_a_wins[0] >= gate_b_wins[0];
        let is_goat = oracle_beats_hl && oracle_improves;

        println!();
        println!(
            "    SDPG(oracle) beats HL?  {}",
            if oracle_beats_hl { "✅ YES" } else { "❌ NO" }
        );
        println!(
            "    Oracle ≥ Uniform?       {}",
            if oracle_improves { "✅ YES" } else { "❌ NO" }
        );
        println!();
        println!(
            "    🏆 GOAT Gate: {}",
            if is_goat {
                "PASS — SDPG(oracle teacher Q) is GOAT!"
            } else {
                "FAIL — oracle teacher Q needs more burn-in or tuning"
            }
        );
        println!("══════════════════════════════════════════════════════════════");
    }
} // mod inner

#[cfg(all(feature = "sdpg_bandit", feature = "bomber"))]
fn main() {
    inner::main();
}

#[cfg(not(all(feature = "sdpg_bandit", feature = "bomber")))]
fn main() {
    eprintln!("This example requires --features sdpg_bandit,bomber");
}
