//! Plan 065 Phase 4 T37: G-Zero Self-Play with per-round metrics + δ tracking.
//!
//! Runs G-Zero self-play: two template-proposer players play against each other
//! while tracking Hint-δ evolution, template exploration, and win rates.
//!
//! ```sh
//! # Quick: 50 episodes, 9×9
//! GO_EPISODES=50 cargo run --features go --example go_04_gzero
//!
//! # Full: 500 episodes (paper default)
//! cargo run --features go --example go_04_gzero
//!
//! # Quick demo: 10 episodes only
//! GO_SET=quick cargo run --features go --example go_04_gzero
//!
//! # Disable delta-gated absorb-compress
//! GO_NO_DELTA=1 cargo run --features go --example go_04_gzero
//! ```

use std::env;
use std::io::Write;

use katgpt_rs::pruners::go::{
    GoCell, GoDeltaGatedConfig, GoGZeroSelfPlayConfig, GoTemplate, run_gzero_selfplay,
};

// ── Constants ──────────────────────────────────────────────────

/// Default number of episodes.
const DEFAULT_EPISODES: usize = 500;

/// Default board size.
const DEFAULT_BOARD_SIZE: usize = 9;

/// Default progress print interval.
const DEFAULT_PROGRESS: usize = 50;

/// Quick demo episode count.
const QUICK_EPISODES: usize = 10;

// ── Output Formatting ─────────────────────────────────────────

/// Print the experiment header.
fn print_header(
    num_episodes: usize,
    board_size: usize,
    delta_gating: bool,
    adaptive_komi: bool,
    initial_komi: f32,
) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║       G-Zero Self-Play — {board_size}×{board_size}                          ║",
        board_size = board_size
    );
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Episodes          : {num_episodes:<6}                              ║",
        num_episodes = num_episodes
    );
    println!(
        "║  Delta-gating      : {delta_gating:<6}                              ║",
        delta_gating = if delta_gating { "YES" } else { "NO" }
    );
    println!(
        "║  Adaptive komi     : {adaptive_komi:<6}  (initial={initial_komi:.1})           ║",
        adaptive_komi = if adaptive_komi { "YES" } else { "NO" }
    );
    println!("║  Templates         : 4 (CornerStar, Capture, Defend, Tenuki) ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

/// Print a section header.
fn print_section(title: &str) {
    println!();
    println!("━━━ {title} ━━━");
}

/// Template name from index.
fn template_name(idx: usize) -> &'static str {
    match idx {
        0 => "CornerStar",
        1 => "Capture",
        2 => "Defend",
        3 => "Tenuki",
        _ => "Unknown",
    }
}

/// Format duration as human-readable.
fn fmt_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 1.0 {
        format!("{:.0}ms", d.as_millis())
    } else if secs < 60.0 {
        format!("{:.1}s", secs)
    } else {
        format!("{:.1}m", secs / 60.0)
    }
}

// ── Section 1: Quick Demo ─────────────────────────────────────

/// Run a quick self-play demo with 10 episodes.
fn section_quick_demo(board_size: usize) {
    print_section("Section 1: Quick Demo (10 episodes)");

    let config = GoGZeroSelfPlayConfig {
        board_size,
        num_episodes: QUICK_EPISODES,
        use_delta_gating: false,
        delta_config: GoDeltaGatedConfig::default(),
        progress_interval: 5,
        initial_komi: 7.5,
        adaptive_komi: false,
        komi_adjustment_step: 2.0,
        komi_min: 0.0,
        komi_max: 20.0,
        komi_window: 100,
        score_based_rewards: false,
        swap_colors: false,
    };

    let mut rng = fastrand::Rng::with_seed(42);
    let results = run_gzero_selfplay(&config, &mut rng);

    // Print per-episode summary
    println!();
    println!(
        "  {:>4}  {:>6}  {:>6}  {:>8}  {:>8}",
        "Ep", "Winner", "Moves", "Avg δ", "Time"
    );
    println!("  ────  ──────  ──────  ────────  ────────");

    for ep in &results.episodes {
        let winner_str = match ep.winner {
            Some(GoCell::Black) => "Black",
            Some(GoCell::White) => "White",
            Some(GoCell::Empty) | None => "Draw",
        };

        let avg_delta = if ep.move_deltas.is_empty() {
            0.0
        } else {
            ep.move_deltas.iter().map(|d| d.delta).sum::<f32>() / ep.move_deltas.len() as f32
        };

        println!(
            "  {:>4}  {:>6}  {:>6}  {:>+8.4}  {:>8}",
            ep.episode,
            winner_str,
            ep.total_moves,
            avg_delta,
            fmt_duration(ep.duration),
        );
    }

    println!();
    println!(
        "  {} episodes completed in {}",
        results.episodes.len(),
        fmt_duration(results.duration),
    );
}

// ── Section 2: Full Self-Play ─────────────────────────────────

/// Run full self-play with delta-gated absorb-compress.
fn section_full_selfplay(num_episodes: usize, board_size: usize, delta_gating: bool) {
    print_section(&format!(
        "Section 2: Full Self-Play ({num_episodes} episodes)"
    ));

    let config = GoGZeroSelfPlayConfig {
        board_size,
        num_episodes,
        use_delta_gating: delta_gating,
        delta_config: GoDeltaGatedConfig {
            delta_threshold: 0.1,
            min_observations: 50,
            max_promotions: 1,
        },
        progress_interval: DEFAULT_PROGRESS,
        initial_komi: 7.5,
        adaptive_komi: true,
        komi_adjustment_step: 10.0,
        komi_min: 0.0,
        komi_max: 50.0,
        komi_window: 50,
        score_based_rewards: true,
        swap_colors: true,
    };

    println!("  Running self-play...");
    let _ = std::io::stdout().flush();

    let mut rng = fastrand::Rng::with_seed(123);
    let results = run_gzero_selfplay(&config, &mut rng);

    // Aggregate stats
    let total_games = results.episodes.len();
    let black_wr = if total_games > 0 {
        results.black_wins as f32 / total_games as f32 * 100.0
    } else {
        0.0
    };
    let white_wr = if total_games > 0 {
        results.white_wins as f32 / total_games as f32 * 100.0
    } else {
        0.0
    };
    let draw_rate = if total_games > 0 {
        results.draws as f32 / total_games as f32 * 100.0
    } else {
        0.0
    };
    let avg_moves = if total_games > 0 {
        results
            .episodes
            .iter()
            .map(|e| e.total_moves)
            .sum::<usize>() as f32
            / total_games as f32
    } else {
        0.0
    };
    let episodes_per_sec = if results.duration.as_secs_f32() > 0.0 {
        total_games as f32 / results.duration.as_secs_f32()
    } else {
        0.0
    };

    println!();
    println!("  ┌────────────────────────────────────────────┐");
    println!("  │  SELF-PLAY RESULTS                         │");
    println!("  ├────────────────────────────────────────────┤");
    println!(
        "  │  Total episodes : {total_games:<24} │",
        total_games = total_games
    );
    println!(
        "  │  Duration       : {:<24} │",
        fmt_duration(results.duration)
    );
    println!(
        "  │  Episodes/sec   : {episodes_per_sec:<24.1} │",
        episodes_per_sec = episodes_per_sec
    );
    println!(
        "  │  Black wins     : {} ({black_wr:.1}%)              │",
        results.black_wins
    );
    println!(
        "  │  White wins     : {} ({white_wr:.1}%)              │",
        results.white_wins
    );
    println!(
        "  │  Draws          : {} ({draw_rate:.1}%)              │",
        results.draws
    );
    println!("  │  Final komi     : {:<24.1} │", results.final_komi);
    println!(
        "  │  Avg moves/game : {avg_moves:<24.1} │",
        avg_moves = avg_moves
    );
    println!(
        "  │  Total δ        : {total_delta:<+24.4} │",
        total_delta = results.total_delta
    );
    println!(
        "  │  Avg δ/move     : {avg_delta:<+24.4} │",
        avg_delta = results.avg_delta_per_move
    );
    println!("  └────────────────────────────────────────────┘");

    // Show komi adjustments
    if !results.komi_history.is_empty() {
        println!();
        println!("  Komi adjustments:");
        for (ep, k) in &results.komi_history {
            println!("    Episode {ep}: komi = {k:.1}");
        }
    }

    // Show promoted templates
    if !results.promoted_templates.is_empty() {
        println!();
        println!("  Promoted templates (absorb-compress):");
        for tmpl in &results.promoted_templates {
            let name = match tmpl {
                GoTemplate::CornerStar => "CornerStar",
                GoTemplate::Capture => "Capture",
                GoTemplate::Defend => "Defend",
                GoTemplate::Tenuki => "Tenuki",
            };
            println!("    ✓ {name}");
        }
    }
}

// ── Section 3: Template δ Evolution ───────────────────────────

/// Print template delta evolution analysis.
fn section_delta_evolution(num_episodes: usize, board_size: usize) {
    print_section("Section 3: Template δ Evolution");

    // Run a focused self-play to collect delta history
    let config = GoGZeroSelfPlayConfig {
        board_size,
        num_episodes,
        use_delta_gating: false, // No gating — observe raw evolution
        delta_config: GoDeltaGatedConfig::default(),
        progress_interval: num_episodes, // No intermediate prints
        initial_komi: 7.5,
        adaptive_komi: false,
        komi_adjustment_step: 2.0,
        komi_min: 0.0,
        komi_max: 20.0,
        komi_window: 100,
        score_based_rewards: false,
        swap_colors: false,
    };

    println!("  Collecting δ evolution data...");
    let _ = std::io::stdout().flush();

    let mut rng = fastrand::Rng::with_seed(999);
    let results = run_gzero_selfplay(&config, &mut rng);

    // Print delta evolution table
    println!();
    println!(
        "  {:>12}  {:>10}  {:>10}  {:>10}  {:>10}",
        "Episode", "CornerStar", "Capture", "Defend", "Tenuki"
    );
    println!("  ────────────  ──────────  ──────────  ──────────  ──────────");

    // Sample at intervals
    let sample_count = 10usize;
    let interval = (num_episodes / sample_count).max(1);

    for sample_idx in 0..=sample_count {
        let episode = (sample_idx * interval).min(num_episodes);
        if episode == 0 {
            continue;
        }

        let mut deltas = [0.0f32; 4];
        for (tmpl_idx, delta_slot) in deltas.iter_mut().enumerate().take(4) {
            if let Some(history) = results.template_delta_history.get(tmpl_idx) {
                // Find the last entry at or before this episode
                let relevant: Vec<_> = history.iter().filter(|(ep, _)| *ep <= episode).collect();
                if let Some((_, delta)) = relevant.last() {
                    *delta_slot = *delta;
                }
            }
        }

        println!(
            "  {:>12}  {:>+10.4}  {:>+10.4}  {:>+10.4}  {:>+10.4}",
            episode, deltas[0], deltas[1], deltas[2], deltas[3],
        );
    }

    // Summary: which template has highest mean delta
    println!();
    println!("  Template ranking by final δ:");
    let mut ranked: Vec<(usize, f32)> = results
        .template_delta_history
        .iter()
        .enumerate()
        .map(|(idx, history)| {
            let mean = history.last().map(|(_, d)| *d).unwrap_or(0.0);
            (idx, mean)
        })
        .collect();

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (rank, (idx, delta)) in ranked.iter().enumerate() {
        let medal = match rank {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        println!(
            "    {medal} {:>10} δ = {delta:+.4}",
            template_name(*idx),
            delta = delta,
        );
    }
}

// ── Section 4: Absorb-Compress Demo ──────────────────────────

/// Demonstrate delta-gated absorb-compress promotion.
fn section_absorb_compress(board_size: usize) {
    print_section("Section 4: Absorb-Compress Demo (200 episodes)");

    // Run with aggressive gating to demonstrate promotion
    let config = GoGZeroSelfPlayConfig {
        board_size,
        num_episodes: 200,
        use_delta_gating: true,
        delta_config: GoDeltaGatedConfig {
            delta_threshold: 0.05, // Lower threshold → easier promotion
            min_observations: 30,  // Fewer observations needed
            max_promotions: 2,     // Allow up to 2 promotions
        },
        progress_interval: 200, // Only print at end
        initial_komi: 7.5,
        adaptive_komi: false,
        komi_adjustment_step: 2.0,
        komi_min: 0.0,
        komi_max: 20.0,
        komi_window: 100,
        score_based_rewards: false,
        swap_colors: false,
    };

    println!("  Running with aggressive absorb-compress settings...");
    let _ = std::io::stdout().flush();

    let mut rng = fastrand::Rng::with_seed(777);
    let results = run_gzero_selfplay(&config, &mut rng);

    println!();
    println!("  Absorb-Compress Results:");
    println!(
        "    Delta threshold : {:.2}",
        config.delta_config.delta_threshold
    );
    println!(
        "    Min observations: {}",
        config.delta_config.min_observations
    );
    println!(
        "    Max promotions  : {}",
        config.delta_config.max_promotions
    );

    if results.promoted_templates.is_empty() {
        println!();
        println!("    No templates promoted (δ below threshold for all templates)");
        println!("    This is normal — promotion requires consistently positive δ");
    } else {
        println!("    Promoted templates:");
        for tmpl in &results.promoted_templates {
            let name = match tmpl {
                GoTemplate::CornerStar => "CornerStar",
                GoTemplate::Capture => "Capture",
                GoTemplate::Defend => "Defend",
                GoTemplate::Tenuki => "Tenuki",
            };
            println!("      ✓ {name} — promoted to hard constraint");
        }
    }

    println!();
    println!(
        "  Final stats: {}B {}W {}D | avg δ/move: {:+.4}",
        results.black_wins, results.white_wins, results.draws, results.avg_delta_per_move,
    );
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    // Parse env configuration
    let num_episodes: usize = env::var("GO_EPISODES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EPISODES);

    let board_size: usize = env::var("GO_BOARD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BOARD_SIZE);

    let delta_gating = env::var("GO_NO_DELTA").ok().is_none_or(|v| v != "1");

    // Select which sections to run
    let section_set = env::var("GO_SET").ok().unwrap_or_else(|| "all".to_string());

    print_header(num_episodes, board_size, delta_gating, true, 7.5);

    match section_set.as_str() {
        "quick" => {
            section_quick_demo(board_size);
        }
        "full" => {
            section_full_selfplay(num_episodes, board_size, delta_gating);
        }
        "evolution" => {
            section_delta_evolution(num_episodes.min(200), board_size);
        }
        "absorb" => {
            section_absorb_compress(board_size);
        }
        _ => {
            // "all" — run everything
            section_quick_demo(board_size);
            section_full_selfplay(num_episodes, board_size, delta_gating);
            section_delta_evolution(num_episodes.min(200), board_size);
            section_absorb_compress(board_size);
        }
    }

    println!();
    println!("━━━ Done ━━━");
    println!();
}
