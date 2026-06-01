//! FFT Tactics arena tournament runner — reusable N-battle match runner.
//!
//! Provides `run_fft_battle` for a single ATB fight and `run_fft_matchup`
//! for a full N-game series between two teams of `FftPlayer` strategies.

use std::cmp::Ordering;
use std::time::Instant;

use fastrand::Rng;

use crate::pruners::arena::types::{ArenaKind, GameResult, MatchupResult};

use super::battle::{BattleState, resolve_action};
use super::players::FftPlayer;
use super::types::{Action, GameEvent, TURN_LIMIT, Team};

/// Configuration for an FFT arena tournament matchup.
#[derive(Clone, Debug)]
pub struct FftArenaConfig {
    /// Number of battles per matchup.
    pub games: usize,
    /// Turn limit per battle.
    pub turn_limit: u32,
}

impl Default for FftArenaConfig {
    fn default() -> Self {
        Self {
            games: 20,
            turn_limit: TURN_LIMIT,
        }
    }
}

/// Result of a single FFT battle.
#[derive(Clone, Debug)]
pub struct FftBattleResult {
    /// Winning team (`None` if draw / turn limit reached).
    pub winner: Option<Team>,
    /// Per-unit scores (8 units: 4 party + 4 enemy).
    pub scores: Vec<i32>,
    /// Ticks (ATB rounds) played.
    pub ticks: u32,
    /// Party units surviving.
    pub party_survivors: usize,
    /// Enemy units surviving.
    pub enemy_survivors: usize,
    /// Game events (damage, heals, kills, etc.) for learning players.
    pub events: Vec<GameEvent>,
    /// Wall-clock duration of the battle.
    pub duration: std::time::Duration,
}

/// Run a single FFT battle with given player strategies.
///
/// * `party_players` — 4 `FftPlayer` instances, one per party unit (ids 0–3)
/// * `enemy_players` — 4 `FftPlayer` instances, one per enemy unit (ids 4–7)
/// * `turn_limit` — max ATB ticks before declaring a draw
/// * `rng` — deterministic RNG for reproducibility
pub fn run_fft_battle(
    party_players: &mut [Box<dyn FftPlayer>],
    enemy_players: &mut [Box<dyn FftPlayer>],
    turn_limit: u32,
    rng: &mut Rng,
) -> FftBattleResult {
    let start = Instant::now();
    let mut battle = BattleState::new();
    let mut ticks = 0u32;

    for _ in 0..turn_limit {
        battle.advance_ct();

        // Tick status effects first (same order as riir-examples runner)
        battle.tick_effects();

        let ready = battle.ready_units();

        if ready.is_empty() {
            ticks += 1;
            continue;
        }

        // Collect all actions first, then resolve (deterministic)
        let mut actions: Vec<(u8, Action)> = Vec::with_capacity(8);
        for &unit_id in &ready {
            // Clear defending flag each turn
            battle.units[unit_id as usize].defending = false;

            let unit = &battle.units[unit_id as usize];
            if !unit.alive {
                continue;
            }

            let player = match unit.team {
                Team::Party => &mut party_players[unit_id as usize],
                Team::Enemy => &mut enemy_players[(unit_id - 4) as usize],
            };

            let action = player.select_action(unit_id, &battle, rng);
            actions.push((unit_id, action));
        }

        // Resolve actions in order
        for (unit_id, action) in &actions {
            resolve_action(&mut battle, *unit_id, action, rng);
            if battle.check_winner().is_some() {
                break;
            }
        }

        // Reset CT for acted units
        battle.reset_ct(&ready);
        ticks += 1;

        if battle.check_winner().is_some() {
            break;
        }
    }

    let winner = battle.check_winner().or_else(|| {
        // Timeout: compare HP to determine winner (same as riir-examples runner)
        let party_hp = battle.team_hp(Team::Party);
        let enemy_hp = battle.team_hp(Team::Enemy);
        match party_hp.cmp(&enemy_hp) {
            Ordering::Greater => Some(Team::Party),
            Ordering::Less => Some(Team::Enemy),
            Ordering::Equal => None,
        }
    });
    let party_survivors = battle
        .units
        .iter()
        .filter(|u| u.alive && u.team == Team::Party)
        .count();
    let enemy_survivors = battle
        .units
        .iter()
        .filter(|u| u.alive && u.team == Team::Enemy)
        .count();

    // Score: +3 surviving on winning team, +1 surviving on losing team, -2 dead
    let winning_team = winner.unwrap_or(Team::Party);
    let scores: Vec<i32> = battle
        .units
        .iter()
        .map(|unit| match (unit.alive, unit.team == winning_team) {
            (true, true) => 3,
            (true, false) => 1,
            (false, _) => -2,
        })
        .collect();

    FftBattleResult {
        winner,
        scores,
        ticks,
        party_survivors,
        enemy_survivors,
        events: battle.events,
        duration: start.elapsed(),
    }
}

/// Run a full FFT tournament matchup (N battles).
///
/// Players are reset between games (learning state persists via bandit Q-values).
/// Uses a fixed seed (`42`) for deterministic tournament results.
pub fn run_fft_matchup(
    party_players: &mut [Box<dyn FftPlayer>],
    enemy_players: &mut [Box<dyn FftPlayer>],
    config: &FftArenaConfig,
) -> MatchupResult {
    let mut rng = Rng::with_seed(42);
    let mut games = Vec::with_capacity(config.games);

    for _ in 0..config.games {
        let result = run_fft_battle(party_players, enemy_players, config.turn_limit, &mut rng);

        // Convert Team winner to player index (0 = Party, 1 = Enemy)
        let winner = result.winner.map(|team| match team {
            Team::Party => 0,
            Team::Enemy => 1,
        });

        // Extract per-unit outcome stats for learning players
        let unit_outcomes: Vec<(bool, u32, i32, i32)> = (0u8..8)
            .map(|uid| {
                let survived = result.scores[uid as usize] > 0;
                let kills = result
                    .events
                    .iter()
                    .filter(|e| matches!(e, GameEvent::UnitDied { killer, .. } if *killer == uid))
                    .count() as u32;
                let damage: i32 = result
                    .events
                    .iter()
                    .filter_map(|e| match e {
                        GameEvent::DamageDealt {
                            attacker, damage, ..
                        } if *attacker == uid => Some(*damage),
                        _ => None,
                    })
                    .sum();
                let healing: i32 = result
                    .events
                    .iter()
                    .filter_map(|e| match e {
                        GameEvent::Healed { healer, amount, .. } if *healer == uid => Some(*amount),
                        _ => None,
                    })
                    .sum();
                (survived, kills, damage, healing)
            })
            .collect();

        games.push(GameResult {
            winner,
            scores: result.scores,
            ticks: result.ticks,
            duration: result.duration,
        });

        // Feed game outcome to learning players (survival, kills, damage, healing)
        for (idx, p) in party_players.iter_mut().enumerate() {
            let (survived, kills, damage, healing) = unit_outcomes[idx];
            p.on_game_end(idx as u8, survived, kills, damage, healing);
        }
        for (idx, p) in enemy_players.iter_mut().enumerate() {
            let (survived, kills, damage, healing) = unit_outcomes[idx + 4];
            p.on_game_end((idx + 4) as u8, survived, kills, damage, healing);
        }

        // Reset players between games (bandit Q-values persist across resets)
        for p in party_players.iter_mut() {
            p.reset();
        }
        for p in enemy_players.iter_mut() {
            p.reset();
        }
    }

    let player_names = vec!["Party".to_string(), "Enemy".to_string()];

    MatchupResult {
        arena: ArenaKind::Fft,
        player_names,
        games,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_greedy_party() -> Vec<Box<dyn FftPlayer>> {
        vec![
            Box::new(crate::pruners::fft::GreedyFFTPlayer),
            Box::new(crate::pruners::fft::GreedyFFTPlayer),
            Box::new(crate::pruners::fft::GreedyFFTPlayer),
            Box::new(crate::pruners::fft::GreedyFFTPlayer),
        ]
    }

    fn make_validator_enemy() -> Vec<Box<dyn FftPlayer>> {
        vec![
            Box::new(crate::pruners::fft::ValidatorFFTPlayer),
            Box::new(crate::pruners::fft::ValidatorFFTPlayer),
            Box::new(crate::pruners::fft::ValidatorFFTPlayer),
            Box::new(crate::pruners::fft::ValidatorFFTPlayer),
        ]
    }

    #[test]
    fn fft_arena_config_default() {
        let config = FftArenaConfig::default();
        assert_eq!(config.games, 20);
        assert_eq!(config.turn_limit, TURN_LIMIT);
    }

    #[test]
    fn run_fft_battle_completes() {
        let mut party = make_greedy_party();
        let mut enemy = make_greedy_party();
        let mut rng = Rng::with_seed(123);

        let result = run_fft_battle(&mut party, &mut enemy, TURN_LIMIT, &mut rng);

        assert!(result.ticks > 0);
        assert_eq!(result.scores.len(), 8);
        assert!(result.party_survivors + result.enemy_survivors > 0);
        assert!(result.duration.as_nanos() > 0);
    }

    #[test]
    fn run_fft_battle_respects_turn_limit() {
        let mut party = make_greedy_party();
        let mut enemy = make_validator_enemy();
        let mut rng = Rng::with_seed(99);

        let result = run_fft_battle(&mut party, &mut enemy, 10, &mut rng);

        assert!(result.ticks <= 10);
    }

    #[test]
    fn run_fft_matchup_correct_game_count() {
        let mut party = make_greedy_party();
        let mut enemy = make_validator_enemy();
        let config = FftArenaConfig {
            games: 5,
            turn_limit: 100,
        };

        let matchup = run_fft_matchup(&mut party, &mut enemy, &config);

        assert_eq!(matchup.games.len(), 5);
        assert_eq!(matchup.arena, ArenaKind::Fft);
        assert_eq!(matchup.player_names.len(), 2);
        assert_eq!(matchup.player_names[0], "Party");
        assert_eq!(matchup.player_names[1], "Enemy");
    }

    #[test]
    fn run_fft_matchup_tracks_wins() {
        let mut party = make_greedy_party();
        let mut enemy = make_validator_enemy();
        let config = FftArenaConfig {
            games: 10,
            turn_limit: 100,
        };

        let matchup = run_fft_matchup(&mut party, &mut enemy, &config);

        let party_wins = matchup.wins_for(0);
        let enemy_wins = matchup.wins_for(1);
        let total_wins = party_wins + enemy_wins;
        assert!(total_wins <= 10);
    }

    #[test]
    fn battle_scores_are_valid() {
        let mut party = make_greedy_party();
        let mut enemy = make_greedy_party();
        let mut rng = Rng::with_seed(77);

        let result = run_fft_battle(&mut party, &mut enemy, TURN_LIMIT, &mut rng);

        for score in &result.scores {
            assert!(*score == -2 || *score == 1 || *score == 3);
        }
    }
}
