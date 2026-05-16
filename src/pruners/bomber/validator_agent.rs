//! validator_agent.rs — Coding Agent Validator Loop (Issue 052, Tasks C1-C4)
//!
//! Foundational structs and arena evaluation for generating and testing
//! rule-based validator candidates in the bomber arena.
//!
//! The validator candidate is a rule-based AST (not freeform code) — bounded
//! search space, deterministic output. Rules are compiled to a scoring function
//! that the `RulePlayer` uses to select actions in the arena.
//!
//! ## Architecture
//!
//! ```text
//! ValidatorCandidate (rules AST)
//!       │
//!       ▼
//! RulePlayer (implements BomberPlayer)
//!       │
//!       ▼
//! evaluate_validator() → ArenaEvaluation
//!       │
//!       ├── survival_rate, kill_rate, avg_score
//!       └── failure_traces (C4: rounds where fatal moves were approved)
//! ```

#[cfg(feature = "bomber")]
use std::any::Any;

#[cfg(feature = "bomber")]
use fastrand::Rng;

#[cfg(feature = "bomber")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "bomber")]
use super::{
    Alive, ArenaGrid, BOMB_FUSE_TICKS, BomberAction, Cell, DEFAULT_BLAST_RANGE, GameEvent, GridPos,
    TickCounter, init_world_with_arena, run_tick, spawn_players,
};

#[cfg(feature = "bomber")]
use super::arena::STANDARD_ARENA;

#[cfg(feature = "bomber")]
use super::players::{
    BomberPlayer, RandomPlayer, count_escape_routes, is_safe_action, move_target,
};

// ── C1: Validator Candidate Structs ────────────────────────────

/// A candidate validator described as a serializable rule AST.
///
/// Rules form a bounded search space — no freeform code, deterministic output.
/// Each candidate represents a strategy that can be evaluated in the arena.
#[cfg(feature = "bomber")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ValidatorCandidate {
    /// Unique ID for this candidate.
    pub id: String,
    /// Generation number (0 = initial).
    pub generation: u32,
    /// Rule templates with configurable thresholds.
    pub rules: Vec<ValidatorRule>,
}

/// A single rule in the validator AST.
///
/// Each rule contributes a score modifier to action evaluation.
/// The `RulePlayer` sums all rule scores to pick the best action.
#[cfg(feature = "bomber")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ValidatorRule {
    /// Avoid blast zone within N ticks.
    AvoidBlast { lookahead: u32 },
    /// Stay away from bombs within N cells.
    DistanceFromBomb { min_distance: u32 },
    /// Prefer moving toward power-ups.
    SeekPowerUp { priority: f32 },
    /// Avoid corners (dead ends).
    AvoidDeadEnd { lookahead: u32 },
    /// Block opponents from reaching power-ups.
    BlockOpponent { aggression: f32 },
}

// ── C2: Arena Evaluation Structs ───────────────────────────────

/// Result of evaluating a validator candidate in the arena.
#[cfg(feature = "bomber")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArenaEvaluation {
    /// Candidate that was evaluated.
    pub candidate_id: String,
    /// Number of rounds played.
    pub rounds: u32,
    /// Survival rate (0.0 - 1.0).
    pub survival_rate: f32,
    /// Kill rate (opponents killed per round, 0.0+).
    pub kill_rate: f32,
    /// Average score per round.
    pub avg_score: f32,
    /// Rounds where the validator approved a fatal move.
    pub failure_traces: Vec<FailureTrace>,
}

/// Record of a round where the validator failed (approved a fatal move).
#[cfg(feature = "bomber")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FailureTrace {
    /// Round number.
    pub round: u32,
    /// Tick when death occurred.
    pub death_tick: u32,
    /// Action that was approved (and led to death).
    pub approved_action: u8,
    /// Safe actions that were available but not chosen.
    pub safe_actions: Vec<u8>,
}

// ── Internal Types ─────────────────────────────────────────────

/// Tracked bomb: (position, blast_range, fuse_ticks_remaining).
#[cfg(feature = "bomber")]
type TrackedBomb = ((i32, i32), u32, u32);

/// Tracked opponent: (player_id, current_pos, prev_pos).
#[cfg(feature = "bomber")]
type TrackedOpponent = (u8, (i32, i32), Option<(i32, i32)>);

/// Tick limit for evaluation rounds (shorter than default for speed).
#[cfg(feature = "bomber")]
const EVAL_TICK_LIMIT: u32 = 200;

// ── Helper Functions ───────────────────────────────────────────

/// Check if position is in blast range of a single bomb (wall-blocking aware).
///
/// Duplicated from `players::is_in_single_blast` (private) for rule scoring
/// with per-bomb fuse filtering.
#[cfg(feature = "bomber")]
fn is_in_blast_range(pos: GridPos, grid: &ArenaGrid, bomb_pos: (i32, i32), range: u32) -> bool {
    let (bx, by) = bomb_pos;

    // Standing on the bomb itself
    if pos.x == bx && pos.y == by {
        return true;
    }

    // Same row (horizontal blast)
    if pos.y == by {
        let dx = pos.x - bx;
        if dx.unsigned_abs() <= range {
            let step = dx.signum();
            let mut x = bx + step;
            while x != pos.x {
                match grid.get(x, by) {
                    Cell::FixedWall | Cell::DestructibleWall | Cell::PowerUpHidden(_) => {
                        return false;
                    }
                    _ => {}
                }
                x += step;
            }
            return true;
        }
    }

    // Same column (vertical blast)
    if pos.x == bx {
        let dy = pos.y - by;
        if dy.unsigned_abs() <= range {
            let step = dy.signum();
            let mut y = by + step;
            while y != pos.y {
                match grid.get(bx, y) {
                    Cell::FixedWall | Cell::DestructibleWall | Cell::PowerUpHidden(_) => {
                        return false;
                    }
                    _ => {}
                }
                y += step;
            }
            return true;
        }
    }

    false
}

/// Update tracked bombs from game events.
///
/// Tracks `(position, blast_range, fuse_remaining)` and decrements fuses each call.
#[cfg(feature = "bomber")]
fn update_tracked_bombs(bombs: &mut Vec<TrackedBomb>, events: &[GameEvent]) {
    // Decrement fuses each tick
    for bomb in bombs.iter_mut() {
        bomb.2 = bomb.2.saturating_sub(1);
    }
    for event in events {
        match event {
            GameEvent::BombPlaced { pos, .. } => {
                if !bombs.iter().any(|(p, _, _)| *p == *pos) {
                    bombs.push((*pos, DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
                }
            }
            GameEvent::BombExploded { pos, .. } => {
                bombs.retain(|(p, _, _)| *p != *pos);
            }
            _ => {}
        }
    }
}

/// Update tracked power-up positions from game events.
#[cfg(feature = "bomber")]
fn update_tracked_powerups(powerups: &mut Vec<(i32, i32)>, events: &[GameEvent]) {
    for event in events {
        match event {
            GameEvent::PowerUpRevealed { pos, .. } => {
                if !powerups.contains(pos) {
                    powerups.push(*pos);
                }
            }
            GameEvent::PowerUpCollected { pos, .. } => {
                powerups.retain(|p| p != pos);
            }
            _ => {}
        }
    }
}

/// Update tracked opponent positions from game events.
#[cfg(feature = "bomber")]
fn update_tracked_opponents(opponents: &mut Vec<TrackedOpponent>, events: &[GameEvent], my_id: u8) {
    for event in events {
        match event {
            GameEvent::PlayerMoved { player, to, .. } => {
                if *player == my_id {
                    continue;
                }
                if let Some(entry) = opponents.iter_mut().find(|(p, _, _)| *p == *player) {
                    entry.2 = Some(entry.1);
                    entry.1 = *to;
                } else {
                    opponents.push((*player, *to, None));
                }
            }
            GameEvent::PlayerKilled { victim, .. } => {
                opponents.retain(|(p, _, _)| *p != *victim);
            }
            _ => {}
        }
    }
}

/// Score an action based on the candidate's rules.
///
/// Each rule contributes a score modifier. The total is summed across all rules.
#[cfg(feature = "bomber")]
fn score_by_rules(
    action: &BomberAction,
    rules: &[ValidatorRule],
    grid: &ArenaGrid,
    pos: GridPos,
    bombs: &[TrackedBomb],
    powerups: &[(i32, i32)],
    opponents: &[TrackedOpponent],
) -> f32 {
    let target = move_target(action, pos);
    let mut score = 0.0;

    for rule in rules {
        match rule {
            ValidatorRule::AvoidBlast { lookahead } => {
                // Penalize if target is in blast zone of bombs exploding within lookahead ticks
                let in_danger = bombs.iter().any(|&(bomb_pos, range, fuse)| {
                    fuse <= *lookahead && is_in_blast_range(target, grid, bomb_pos, range)
                });
                if in_danger {
                    score -= 10.0;
                }
            }
            ValidatorRule::DistanceFromBomb { min_distance } => {
                // Penalize if any bomb is within min_distance manhattan distance of target
                let too_close = bombs.iter().any(|&(bomb_pos, _, _)| {
                    let dist = (target.x - bomb_pos.0).abs() + (target.y - bomb_pos.1).abs();
                    dist <= *min_distance as i32
                });
                if too_close {
                    score -= 5.0;
                }
            }
            ValidatorRule::SeekPowerUp { priority } => {
                // Reward moving toward nearest revealed power-up
                if !powerups.is_empty() {
                    let current_min = powerups
                        .iter()
                        .map(|&(px, py)| (pos.x - px).abs() + (pos.y - py).abs())
                        .min()
                        .unwrap_or(i32::MAX);
                    let target_min = powerups
                        .iter()
                        .map(|&(px, py)| (target.x - px).abs() + (target.y - py).abs())
                        .min()
                        .unwrap_or(i32::MAX);
                    if target_min < current_min {
                        score += priority;
                    }
                }
            }
            ValidatorRule::AvoidDeadEnd { lookahead: _ } => {
                // Penalize positions with few escape routes (dead-end prone)
                let routes = count_escape_routes((target.x, target.y), grid);
                match routes {
                    0 => score -= 8.0,
                    1 => score -= 4.0,
                    _ => {}
                }
            }
            ValidatorRule::BlockOpponent { aggression } => {
                // Reward moving closer to nearest opponent (aggressive positioning)
                if !opponents.is_empty() {
                    let current_min = opponents
                        .iter()
                        .map(|(_, (ox, oy), _)| (pos.x - ox).abs() + (pos.y - oy).abs())
                        .min()
                        .unwrap_or(i32::MAX);
                    let target_min = opponents
                        .iter()
                        .map(|(_, (ox, oy), _)| (target.x - ox).abs() + (target.y - oy).abs())
                        .min()
                        .unwrap_or(i32::MAX);
                    if target_min < current_min {
                        score += aggression;
                    }
                }
            }
        }
    }

    score
}

// ── C3: Rule Player ────────────────────────────────────────────

/// A player that scores actions based on a [`ValidatorCandidate`]'s rules.
///
/// Implements `BomberPlayer` so it can participate in the arena.
/// For each action, sums rule scores and picks the highest-scored safe action.
/// Falls back to best-scored action (regardless of safety) when no safe option exists.
#[cfg(feature = "bomber")]
pub struct RulePlayer {
    _id: u8,
    rules: Vec<ValidatorRule>,
    known_bombs: Vec<TrackedBomb>,
    known_powerups: Vec<(i32, i32)>,
    known_opponents: Vec<TrackedOpponent>,
    last_action: Option<BomberAction>,
}

#[cfg(feature = "bomber")]
impl RulePlayer {
    /// Create a new RulePlayer from a validator candidate's rules.
    pub fn new(id: u8, candidate: &ValidatorCandidate) -> Self {
        Self {
            _id: id,
            rules: candidate.rules.clone(),
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            last_action: None,
        }
    }

    /// Get the player's tracked bombs (for failure trace extraction).
    pub fn known_bombs(&self) -> &[TrackedBomb] {
        &self.known_bombs
    }

    /// Get the player's last chosen action (for failure trace extraction).
    pub fn last_action(&self) -> Option<BomberAction> {
        self.last_action
    }
}

#[cfg(feature = "bomber")]
impl BomberPlayer for RulePlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        _rng: &mut Rng,
    ) -> BomberAction {
        update_tracked_bombs(&mut self.known_bombs, events);
        update_tracked_powerups(&mut self.known_powerups, events);
        update_tracked_opponents(&mut self.known_opponents, events, self._id);

        let mut best_safe = BomberAction::Wait;
        let mut best_safe_score = f32::NEG_INFINITY;
        let mut has_safe = false;

        let mut best_any = BomberAction::Wait;
        let mut best_any_score = f32::NEG_INFINITY;

        for action in BomberAction::all() {
            let is_move = matches!(
                action,
                BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
            );

            // Hard constraint: unwalkable target gets -inf
            let effective_score = if is_move {
                let target = move_target(&action, pos);
                if !grid.is_walkable(target.x, target.y) {
                    f32::NEG_INFINITY
                } else {
                    score_by_rules(
                        &action,
                        &self.rules,
                        grid,
                        pos,
                        &self.known_bombs,
                        &self.known_powerups,
                        &self.known_opponents,
                    )
                }
            } else {
                score_by_rules(
                    &action,
                    &self.rules,
                    grid,
                    pos,
                    &self.known_bombs,
                    &self.known_powerups,
                    &self.known_opponents,
                )
            };

            if effective_score > best_any_score {
                best_any_score = effective_score;
                best_any = action;
            }

            if is_safe_action(&action, grid, pos, &self.known_bombs) {
                has_safe = true;
                if effective_score > best_safe_score {
                    best_safe_score = effective_score;
                    best_safe = action;
                }
            }
        }

        // Prefer safe actions; fall back to best-scored when nothing is safe
        let chosen = if has_safe { best_safe } else { best_any };

        // Track bomb placement for internal state consistency
        if chosen == BomberAction::Bomb {
            self.known_bombs
                .push(((pos.x, pos.y), DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
        }

        self.last_action = Some(chosen);
        chosen
    }

    fn name(&self) -> &str {
        "RuleAgent"
    }

    fn emoji(&self) -> &str {
        "🤖"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.known_opponents.clear();
        self.last_action = None;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── C3+C4: Arena Evaluation ────────────────────────────────────

/// Evaluate a validator candidate by running it as a RulePlayer in the arena.
///
/// Creates a fixed `STANDARD_ARENA` for reproducibility, runs N rounds with
/// 3 `RandomPlayer`s + 1 `RulePlayer`, and collects metrics.
///
/// ## Failure Trace Extraction (C4)
///
/// When the `RulePlayer` dies, records:
/// - Which action was approved that led to death
/// - What safe alternatives existed at that moment
/// - The tick and round number
///
/// These traces feed back into the agent loop (C5+) for rule refinement.
#[cfg(feature = "bomber")]
pub fn evaluate_validator(candidate: &ValidatorCandidate, rounds: u32) -> ArenaEvaluation {
    let arena = ArenaGrid::fixed(STANDARD_ARENA).expect("STANDARD_ARENA must be valid");
    let mut rng = Rng::with_seed(42);

    let mut survival_count = 0u32;
    let mut total_kills = 0u32;
    let mut total_score = 0i32;
    let mut failure_traces: Vec<FailureTrace> = Vec::new();

    for round in 0..rounds {
        let mut world = init_world_with_arena(arena.clone());
        let entities = spawn_players(&mut world);

        // Create players: RulePlayer is player 0, others are Random
        let mut rule_player = RulePlayer::new(0, candidate);
        let mut random_players = [
            RandomPlayer::new(1),
            RandomPlayer::new(2),
            RandomPlayer::new(3),
        ];

        rule_player.reset();
        for p in &mut random_players {
            p.reset();
        }

        let mut round_events: Vec<GameEvent> = Vec::new();
        let mut last_approved: Option<BomberAction> = None;
        let mut last_safe_actions: Vec<BomberAction> = Vec::new();
        let mut rule_player_died = false;
        let mut death_tick = 0u32;

        // Run tick loop
        for _tick in 0..EVAL_TICK_LIMIT {
            // Drain events from previous tick
            let tick_events: Vec<GameEvent> = {
                let mut event_reader = world.resource_mut::<bevy_ecs::event::Events<GameEvent>>();
                event_reader.drain().collect()
            };
            round_events.extend(tick_events.iter().cloned());

            // Check if rule player died in previous tick
            for event in &tick_events {
                if let GameEvent::PlayerKilled { victim: 0, .. } = event {
                    rule_player_died = true;
                    death_tick = world.resource::<TickCounter>().tick;
                }
            }
            if rule_player_died {
                break;
            }

            // Each player selects an action
            let mut actions = [None; 4];

            // Rule player (index 0) — separate variable for failure trace access
            let pos0 = world
                .get::<GridPos>(entities[0])
                .copied()
                .unwrap_or_default();
            let alive0 = world.get::<Alive>(entities[0]).is_some();
            if alive0 {
                let grid = world.resource::<ArenaGrid>().clone();
                let action = rule_player.select_action(&grid, pos0, &tick_events, &mut rng);

                // C4: Capture state for failure trace extraction
                last_approved = Some(action);
                last_safe_actions = BomberAction::all()
                    .iter()
                    .filter(|a| is_safe_action(a, &grid, pos0, rule_player.known_bombs()))
                    .copied()
                    .collect();

                actions[0] = Some(action);
            }

            // Random players (indices 1-3)
            for (i, player) in random_players.iter_mut().enumerate() {
                let pos = world
                    .get::<GridPos>(entities[i + 1])
                    .copied()
                    .unwrap_or_default();
                let alive = world.get::<Alive>(entities[i + 1]).is_some();
                if alive {
                    actions[i + 1] = Some(player.select_action(
                        &world.resource::<ArenaGrid>().clone(),
                        pos,
                        &tick_events,
                        &mut rng,
                    ));
                }
            }

            let ongoing = run_tick(&mut world, actions);
            if !ongoing {
                break;
            }
        }

        // Drain remaining events
        {
            let mut event_reader = world.resource_mut::<bevy_ecs::event::Events<GameEvent>>();
            round_events.extend(event_reader.drain().collect::<Vec<GameEvent>>());
        }

        // Compute round metrics from events
        let mut round_score = 0i32;
        let mut round_kills = 0u32;
        let mut round_survivors: Vec<u8> = Vec::new();

        for event in &round_events {
            match event {
                GameEvent::PlayerKilled { victim, killer } => {
                    if *victim == 0 {
                        // Rule player died
                        round_score -= 3;
                        match killer {
                            Some(k) if *k != 0 => {} // Killed by opponent
                            _ => round_score -= 2,   // Suicide or unknown
                        }
                        // C4: Create failure trace
                        if let Some(approved) = last_approved {
                            failure_traces.push(FailureTrace {
                                round,
                                death_tick,
                                approved_action: approved.as_usize() as u8,
                                safe_actions: last_safe_actions
                                    .iter()
                                    .map(|a| a.as_usize() as u8)
                                    .collect(),
                            });
                        }
                    }
                    // Track kills by rule player
                    match killer {
                        Some(0) if *victim != 0 => {
                            round_kills += 1;
                            round_score += 3;
                        }
                        _ => {}
                    }
                }
                GameEvent::PowerUpCollected { player: 0, .. } => {
                    round_score += 1;
                }
                GameEvent::RoundEnd { survivors } => {
                    round_survivors = survivors.clone();
                }
                _ => {}
            }
        }

        // Determine survival and apply winner/timeout bonus
        let survived = round_survivors.contains(&0);
        if survived {
            survival_count += 1;
            match round_survivors.len() {
                1 => round_score += 5, // Winner bonus
                _ => round_score += 3, // Timeout survival bonus
            }
        }

        total_kills += round_kills;
        total_score += round_score;
    }

    let survival_rate = match rounds {
        0 => 0.0,
        _ => survival_count as f32 / rounds as f32,
    };
    let kill_rate = match rounds {
        0 => 0.0,
        _ => total_kills as f32 / rounds as f32,
    };
    let avg_score = match rounds {
        0 => 0.0,
        _ => total_score as f32 / rounds as f32,
    };

    ArenaEvaluation {
        candidate_id: candidate.id.clone(),
        rounds,
        survival_rate,
        kill_rate,
        avg_score,
        failure_traces,
    }
}
