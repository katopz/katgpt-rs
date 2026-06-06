//! Bomber arena tournament runner — reusable N-game match runner.
//!
//! Adapted from `bomber_01_arena.rs` into a library function for tournament examples.
//! Runs a configured number of Bomber games between 4 players and produces
//! [`MatchupResult`] suitable for leaderboard aggregation.

use std::time::Instant;

use bevy_ecs::prelude::*;
use fastrand::Rng;

use crate::pruners::arena::types::{ArenaKind, GameResult, MatchupResult};

use super::arena::STANDARD_ARENA;
#[cfg(feature = "g_zero")]
use super::g_zero_player::GZeroPlayer;
use super::players::BomberPlayer;
use super::players::HLPlayer;
#[cfg(feature = "ropd_rubric")]
use super::rubric_player::RubricPlayer;
#[cfg(feature = "sdar_gate")]
use super::sdar_player::SdarPlayer;
#[cfg(feature = "sdpg_bandit")]
use super::sdpg_player::SdpgPlayer;
#[cfg(feature = "skill_lifecycle")]
use super::skill_lifecycle_player::SkillLifecyclePlayer;
use super::{ArenaGrid, GameEvent, init_world, init_world_with_arena, run_tick, spawn_players};

// ── Config ─────────────────────────────────────────────────────

/// Configuration for a Bomber arena tournament matchup.
#[derive(Clone, Debug)]
pub struct BomberArenaConfig {
    /// Number of games per matchup.
    pub games: usize,
    /// Tick limit per game.
    pub tick_limit: u32,
    /// Use procedural map generation (destructible walls, powerups) for decisive results.
    /// When `true`, each game generates a fresh map from the rng seed.
    /// When `false`, uses the fixed `arena_template`.
    pub procedural: bool,
    /// Fixed arena template (used only when `procedural` is `false`).
    pub arena_template: &'static str,
}

impl Default for BomberArenaConfig {
    fn default() -> Self {
        Self {
            games: 20,
            tick_limit: 300,
            procedural: true,
            arena_template: STANDARD_ARENA,
        }
    }
}

// ── Per-Round Result ───────────────────────────────────────────

/// Result of a single Bomber round (extended with per-player data).
#[derive(Clone, Debug)]
pub struct BomberRoundResult {
    /// Per-player scores for this round.
    pub scores: Vec<i32>,
    /// Index of the round winner (`None` for draw/timeout).
    pub winner: Option<usize>,
    /// Indices of players who died.
    pub deaths: Vec<usize>,
    /// (killer, victim) pairs.
    pub kills: Vec<(usize, usize)>,
    /// (player, count) powerup pairs.
    pub powerups: Vec<(usize, u32)>,
    /// Ticks elapsed.
    pub ticks: u32,
    /// Wall-clock duration of the round.
    pub duration: std::time::Duration,
}

// ── Single Game Runner ─────────────────────────────────────────

/// Run a single 4-player Bomber game.
///
/// Creates the arena from the configured template, runs the game loop,
/// and returns per-player scores, kills, deaths, and powerups.
pub fn run_bomber_game(
    players: &mut [Box<dyn BomberPlayer>],
    config: &BomberArenaConfig,
    rng: &mut Rng,
) -> BomberRoundResult {
    let start = Instant::now();

    // Build world: procedural maps have destructible walls + powerups for decisive results
    let mut world = match config.procedural {
        true => init_world(rng.u64(..)),
        false => {
            let arena = ArenaGrid::fixed(config.arena_template)
                .unwrap_or_else(|e| panic!("Invalid arena template: {e}"));
            init_world_with_arena(arena)
        }
    };
    let entities = spawn_players(&mut world);

    // Reset all players for new round
    for p in players.iter_mut() {
        p.reset();
    }

    let mut round_events: Vec<GameEvent> = Vec::new();

    // ── Tick loop ──────────────────────────────────────────────
    for _tick in 0..config.tick_limit {
        // Drain events from previous tick
        let tick_events: Vec<GameEvent> = {
            let mut event_reader = world.resource_mut::<Events<GameEvent>>();
            event_reader.drain().collect()
        };
        round_events.extend(tick_events.iter().cloned());

        // Each alive player selects an action
        let mut actions = [None; 4];
        for (i, player) in players.iter_mut().enumerate() {
            let pos = world
                .get::<super::GridPos>(entities[i])
                .copied()
                .unwrap_or_default();
            let alive = world.get::<super::Alive>(entities[i]).is_some();
            if alive {
                let grid = world.resource::<ArenaGrid>().clone();
                actions[i] = Some(player.select_action(&grid, pos, &tick_events, rng));
            }
        }

        let ongoing = run_tick(&mut world, actions);
        if !ongoing {
            break;
        }
    }

    // Drain remaining events after loop ends
    {
        let mut event_reader = world.resource_mut::<Events<GameEvent>>();
        round_events.extend(event_reader.drain().collect::<Vec<GameEvent>>());
    }

    // ── Score computation from events ──────────────────────────
    let player_count = players.len();
    let mut scores = vec![0i32; player_count];
    let mut deaths = Vec::new();
    let mut kills = Vec::new();
    let mut powerups = Vec::new();
    let mut survivors = Vec::new();

    for event in &round_events {
        match event {
            GameEvent::PlayerKilled { victim, killer } => {
                let v = *victim as usize;
                deaths.push(v);
                scores[v] -= 3;
                match killer {
                    Some(k) if *k as usize != v => {
                        kills.push((*k as usize, v));
                        scores[*k as usize] += 3;
                    }
                    _ => {
                        // Suicide or unknown killer
                        scores[v] -= 2;
                    }
                }
            }
            GameEvent::PowerUpCollected { player, .. } => {
                let p = *player as usize;
                scores[p] += 1;
                powerups.push((p, 1));
            }
            GameEvent::RoundEnd { survivors: s } => {
                survivors = s.iter().map(|&id| id as usize).collect();
            }
            _ => {}
        }
    }

    // Winner bonus
    let winner = match survivors.len() {
        1 => {
            scores[survivors[0]] += 5;
            Some(survivors[0])
        }
        2.. => {
            // Timeout: survivors each get +3
            for &s in &survivors {
                scores[s] += 3;
            }
            None
        }
        _ => None,
    };

    let ticks = world.resource::<super::TickCounter>().tick;

    BomberRoundResult {
        scores,
        winner,
        deaths,
        kills,
        powerups,
        ticks,
        duration: start.elapsed(),
    }
}

// ── Matchup Runner ─────────────────────────────────────────────

/// Run a full Bomber tournament matchup (N games between the same 4 players).
///
/// Between games, calls [`HLPlayer::update_outcome`] (and GZero/Rubric when
/// feature-gated) on learning players so bandit state persists across rounds.
/// Does **not** call `reset` between games — learning state is preserved.
pub fn run_bomber_matchup(
    players: &mut [Box<dyn BomberPlayer>],
    config: &BomberArenaConfig,
) -> MatchupResult {
    let mut rng = Rng::with_seed(42);
    let player_names: Vec<String> = players.iter().map(|p| p.name().to_string()).collect();
    let mut games: Vec<GameResult> = Vec::with_capacity(config.games);

    for _ in 0..config.games {
        let round = run_bomber_game(players, config, &mut rng);

        // Update learning players with outcome
        update_learning_players(players, &round);

        // Convert BomberRoundResult → GameResult
        games.push(GameResult {
            winner: round.winner,
            scores: round.scores,
            ticks: round.ticks,
            duration: round.duration,
        });
    }

    MatchupResult {
        arena: ArenaKind::Bomber,
        player_names,
        games,
    }
}

// ── Learning Player Update ─────────────────────────────────────

/// Call `update_outcome` on all learning players that support it.
fn update_learning_players(players: &mut [Box<dyn BomberPlayer>], result: &BomberRoundResult) {
    for (i, player) in players.iter_mut().enumerate() {
        let survived = !result.deaths.contains(&i);
        let killed = result.kills.iter().any(|(killer, _)| *killer == i);
        let powerup_count = result
            .powerups
            .iter()
            .filter(|(p, _)| *p == i)
            .map(|(_, c)| *c)
            .sum::<u32>();

        // Try each learning player type via downcast
        update_if_hl(player, survived, killed, powerup_count);
        #[cfg(feature = "skill_lifecycle")]
        update_if_skill_lifecycle(player, survived, killed, powerup_count);
        #[cfg(feature = "g_zero")]
        update_if_g_zero(player, survived, killed, powerup_count);
        #[cfg(feature = "ropd_rubric")]
        update_if_rubric(player, survived, killed, powerup_count);
        #[cfg(feature = "sdar_gate")]
        update_if_sdar(player, survived, killed, powerup_count);
        #[cfg(feature = "sdpg_bandit")]
        update_if_sdpg(player, survived, killed, powerup_count);
    }
}

/// Update [`HLPlayer`] if the player is one.
fn update_if_hl(player: &mut Box<dyn BomberPlayer>, survived: bool, killed: bool, powerups: u32) {
    if let Some(hl) = player.as_any_mut().downcast_mut::<HLPlayer>() {
        hl.update_outcome(survived, killed, powerups);
    }
}

/// Update [`SkillLifecyclePlayer`] if the player is one.
#[cfg(feature = "skill_lifecycle")]
fn update_if_skill_lifecycle(
    player: &mut Box<dyn BomberPlayer>,
    survived: bool,
    killed: bool,
    powerups: u32,
) {
    if let Some(slp) = player.as_any_mut().downcast_mut::<SkillLifecyclePlayer>() {
        slp.update_outcome(survived, killed, powerups);
    }
}

/// Update [`GZeroPlayer`] if the player is one.
#[cfg(feature = "g_zero")]
fn update_if_g_zero(
    player: &mut Box<dyn BomberPlayer>,
    survived: bool,
    killed: bool,
    powerups: u32,
) {
    if let Some(gz) = player.as_any_mut().downcast_mut::<GZeroPlayer>() {
        gz.update_outcome(survived, killed, powerups);
    }
}

/// Update [`RubricPlayer`] if the player is one.
#[cfg(feature = "ropd_rubric")]
fn update_if_rubric(
    player: &mut Box<dyn BomberPlayer>,
    survived: bool,
    killed: bool,
    powerups: u32,
) {
    if let Some(rp) = player.as_any_mut().downcast_mut::<RubricPlayer>() {
        rp.update_outcome(survived, killed, powerups);
    }
}

/// Update [`SdarPlayer`] if the player is one.
#[cfg(feature = "sdar_gate")]
fn update_if_sdar(player: &mut Box<dyn BomberPlayer>, survived: bool, killed: bool, powerups: u32) {
    if let Some(sp) = player.as_any_mut().downcast_mut::<SdarPlayer>() {
        sp.update_outcome(survived, killed, powerups);
    }
}

/// Update [`SdpgPlayer`] if the player is one.
#[cfg(feature = "sdpg_bandit")]
fn update_if_sdpg(player: &mut Box<dyn BomberPlayer>, survived: bool, killed: bool, powerups: u32) {
    if let Some(sp) = player.as_any_mut().downcast_mut::<SdpgPlayer>() {
        sp.update_outcome(survived, killed, powerups);
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::bomber::{GreedyPlayer, RandomPlayer};

    /// Create a default 4-player roster for testing.
    fn make_test_players() -> Vec<Box<dyn BomberPlayer>> {
        vec![
            Box::new(RandomPlayer::new(0)),
            Box::new(RandomPlayer::new(1)),
            Box::new(GreedyPlayer::new(2)),
            Box::new(GreedyPlayer::new(3)),
        ]
    }

    #[test]
    fn run_bomber_game_returns_valid_result() {
        let mut players = make_test_players();
        let config = BomberArenaConfig::default();
        let mut rng = Rng::with_seed(123);

        let result = run_bomber_game(&mut players, &config, &mut rng);

        assert_eq!(result.scores.len(), 4, "should have 4 player scores");
        assert!(result.ticks > 0, "should have run at least 1 tick");
        assert!(
            result.ticks <= config.tick_limit,
            "should not exceed tick limit"
        );
    }

    #[test]
    fn run_bomber_game_scores_sum_near_zero() {
        // In a zero-sum-ish game, scores should roughly balance
        let mut players = make_test_players();
        let config = BomberArenaConfig::default();
        let mut rng = Rng::with_seed(456);

        let result = run_bomber_game(&mut players, &config, &mut rng);
        let sum: i32 = result.scores.iter().sum();

        // Scores can drift from powerups, but should be bounded
        assert!(
            sum.abs() < 20,
            "score sum should be bounded, got {sum}: {:?}",
            result.scores
        );
    }

    #[test]
    fn run_bomber_matchup_returns_correct_game_count() {
        let mut players = make_test_players();
        let config = BomberArenaConfig {
            games: 5,
            tick_limit: 50,
            procedural: true,
            arena_template: STANDARD_ARENA,
        };

        let result = run_bomber_matchup(&mut players, &config);

        assert_eq!(result.games.len(), 5, "should have 5 game results");
        assert_eq!(result.player_names.len(), 4, "should have 4 player names");
        assert_eq!(result.arena, ArenaKind::Bomber);
    }

    #[test]
    fn run_bomber_matchup_accumulates_wins() {
        let mut players = make_test_players();
        let config = BomberArenaConfig {
            games: 10,
            tick_limit: 100,
            procedural: true,
            arena_template: STANDARD_ARENA,
        };

        let result = run_bomber_matchup(&mut players, &config);

        let total_wins: usize = (0..4).map(|i| result.wins_for(i)).sum();
        // Each game has at most 1 winner, some may be draws
        assert!(
            total_wins <= config.games,
            "total wins ({total_wins}) should not exceed game count ({})",
            config.games
        );
    }

    #[test]
    fn run_bomber_game_with_hl_player_updates_outcome() {
        let mut players: Vec<Box<dyn BomberPlayer>> = vec![
            Box::new(RandomPlayer::new(0)),
            Box::new(RandomPlayer::new(1)),
            Box::new(RandomPlayer::new(2)),
            Box::new(HLPlayer::new(3)),
        ];
        let config = BomberArenaConfig {
            games: 1,
            tick_limit: 50,
            procedural: true,
            arena_template: STANDARD_ARENA,
        };

        let result = run_bomber_matchup(&mut players, &config);

        assert_eq!(result.games.len(), 1);
        // HLPlayer should have been updated without panic
    }

    #[test]
    fn run_bomber_game_deterministic_with_same_seed() {
        let mut players_a = make_test_players();
        let mut players_b = make_test_players();
        let config = BomberArenaConfig::default();

        let result_a = run_bomber_game(&mut players_a, &config, &mut Rng::with_seed(999));
        let result_b = run_bomber_game(&mut players_b, &config, &mut Rng::with_seed(999));

        assert_eq!(result_a.ticks, result_b.ticks, "ticks should match");
        assert_eq!(result_a.scores, result_b.scores, "scores should match");
        assert_eq!(result_a.winner, result_b.winner, "winner should match");
    }

    #[test]
    fn run_bomber_matchup_avg_duration_is_positive() {
        let mut players = make_test_players();
        let config = BomberArenaConfig {
            games: 3,
            tick_limit: 50,
            procedural: true,
            arena_template: STANDARD_ARENA,
        };

        let result = run_bomber_matchup(&mut players, &config);

        assert!(
            result.avg_duration() > std::time::Duration::ZERO,
            "avg duration should be positive"
        );
    }

    #[test]
    fn bomber_round_result_tracks_kills_and_deaths() {
        let mut players = make_test_players();
        let config = BomberArenaConfig {
            games: 1,
            tick_limit: 200,
            procedural: true,
            arena_template: STANDARD_ARENA,
        };
        let mut rng = Rng::with_seed(777);

        let result = run_bomber_game(&mut players, &config, &mut rng);

        // In a game that runs to completion, there should be events tracked
        // (not asserting specific values since outcome depends on player behavior)
        // kills excludes suicides (killer == victim), so kills.len() <= deaths.len()
        assert!(
            result.kills.len() <= result.deaths.len(),
            "kills ({}) should not exceed deaths ({})",
            result.kills.len(),
            result.deaths.len()
        );
        // deaths and kills vectors exist and are consistent
        for (killer, victim) in &result.kills {
            assert_ne!(killer, victim, "kills should not contain suicides");
            assert!(
                result.deaths.contains(victim),
                "every kill victim should be in deaths"
            );
        }
    }
}
