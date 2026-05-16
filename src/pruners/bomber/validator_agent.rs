//! validator_agent.rs — Coding Agent Validator Loop (Issue 052, Tasks C1-C8)
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
//! TemplateProposer (C5: rule templates)
//!       │
//!       ▼
//! ValidatorCandidate (C1: rules AST)
//!       │
//!       ▼
//! RulePlayer (C3: implements BomberPlayer)
//!       │
//!       ▼
//! evaluate_validator() → ArenaEvaluation (C2+C4)
//!       │
//!       ├── survival_rate, kill_rate, avg_score
//!       ├── failure_traces (C4: rounds where fatal moves were approved)
//!       │
//!       ▼
//! propose_from_trace() (C6: mutate from failures)
//!       │
//!       ▼
//! AgentLoop (C7: propose → evaluate → filter → iterate)
//! ```
//!
//! ## Tasks
//!
//! - C1: `ValidatorCandidate`, `ValidatorRule` AST
//! - C2: `ArenaEvaluation` metrics
//! - C3: `RulePlayer` (implements `BomberPlayer`)
//! - C4: `evaluate_validator()` with failure trace extraction
//! - C5: `TemplateProposer` — rule templates with random subsets
//! - C6: `propose_from_trace()` — mutate candidates from failure patterns
//! - C7: `AgentLoop` — propose → evaluate → filter → iterate
//! - C8: `bomber_08_agent_loop` example

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

// ── C5: Template Proposer ──────────────────────────────────────

/// Proposes new validator candidates from rule templates.
///
/// Generates initial candidates by combining rule templates with random
/// threshold variations. Each template provides a sensible default, and
/// the proposer randomizes parameters within small ranges for diversity.
#[cfg(feature = "bomber")]
pub struct TemplateProposer {
    /// Rule templates with default thresholds.
    templates: Vec<ValidatorRule>,
    /// Counter for unique candidate IDs.
    next_id: u64,
}

#[cfg(feature = "bomber")]
impl Default for TemplateProposer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "bomber")]
impl TemplateProposer {
    /// Create a new proposer with default rule templates.
    pub fn new() -> Self {
        Self {
            templates: vec![
                ValidatorRule::AvoidBlast { lookahead: 2 },
                ValidatorRule::DistanceFromBomb { min_distance: 2 },
                ValidatorRule::SeekPowerUp { priority: 1.0 },
                ValidatorRule::AvoidDeadEnd { lookahead: 2 },
                ValidatorRule::BlockOpponent { aggression: 1.0 },
            ],
            next_id: 0,
        }
    }

    /// Generate N candidates with random subsets of templates.
    ///
    /// Each candidate gets 2-5 rules (random subset) with randomized thresholds.
    pub fn propose_initial(&mut self, n: usize, rng: &mut Rng) -> Vec<ValidatorCandidate> {
        (0..n)
            .map(|_| {
                let id = self.next_id;
                self.next_id += 1;
                ValidatorCandidate {
                    id: format!("{id}"),
                    generation: 0,
                    rules: self.random_subset(rng),
                }
            })
            .collect()
    }

    /// Pick a random subset (2-5 rules) with randomized thresholds.
    fn random_subset(&self, rng: &mut Rng) -> Vec<ValidatorRule> {
        let count = rng.usize(2..=self.templates.len());
        let mut indices: Vec<usize> = (0..self.templates.len()).collect();

        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = rng.usize(0..=i);
            indices.swap(i, j);
        }

        indices[..count]
            .iter()
            .map(|&i| Self::randomize_rule(&self.templates[i], rng))
            .collect()
    }

    /// Randomize thresholds around defaults.
    fn randomize_rule(rule: &ValidatorRule, rng: &mut Rng) -> ValidatorRule {
        match rule {
            ValidatorRule::AvoidBlast { .. } => ValidatorRule::AvoidBlast {
                lookahead: rng.u32(1..=4),
            },
            ValidatorRule::DistanceFromBomb { .. } => ValidatorRule::DistanceFromBomb {
                min_distance: rng.u32(1..=3),
            },
            ValidatorRule::SeekPowerUp { .. } => ValidatorRule::SeekPowerUp {
                priority: 0.5 + rng.f32() * 1.5,
            },
            ValidatorRule::AvoidDeadEnd { .. } => ValidatorRule::AvoidDeadEnd {
                lookahead: rng.u32(1..=4),
            },
            ValidatorRule::BlockOpponent { .. } => ValidatorRule::BlockOpponent {
                aggression: 0.3 + rng.f32() * 1.7,
            },
        }
    }
}

// ── C6: Failure Trace Analysis ─────────────────────────────────

/// Failure pattern classification for rule adjustment.
#[cfg(feature = "bomber")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FailurePattern {
    /// Died while waiting (stayed in blast zone).
    BlastZone,
    /// Died from own bomb placement.
    SelfBomb,
    /// Died in a dead-end with few escape routes.
    CornerTrap,
    /// General movement into danger.
    MovedIntoDanger,
}

/// Classify a failure trace into a failure pattern.
#[cfg(feature = "bomber")]
fn classify_failure(trace: &FailureTrace) -> FailurePattern {
    let action = BomberAction::from(trace.approved_action as usize);
    match action {
        BomberAction::Wait => FailurePattern::BlastZone,
        BomberAction::Bomb => FailurePattern::SelfBomb,
        _ => match trace.safe_actions.len() {
            0..=1 => FailurePattern::CornerTrap,
            _ => FailurePattern::MovedIntoDanger,
        },
    }
}

/// Mutate a random rule in the list using template randomization.
#[cfg(feature = "bomber")]
fn mutate_random_rule(rules: &mut [ValidatorRule], rng: &mut Rng) {
    if rules.is_empty() {
        return;
    }
    let idx = rng.usize(0..rules.len());
    rules[idx] = TemplateProposer::randomize_rule(&rules[idx], rng);
}

/// Generate fix candidates from failure patterns.
///
/// Analyzes failure traces and proposes candidates with adjusted rules
/// to address the specific failure modes.
///
/// Pattern matching on failure modes:
/// - Died in blast → increase `AvoidBlast` lookahead
/// - Died near bomb → increase `DistanceFromBomb` min_distance
/// - Died in corner → increase `AvoidDeadEnd` lookahead
///
/// Returns 3-5 variants with adjusted parameters.
#[cfg(feature = "bomber")]
pub fn propose_from_trace(
    base: &ValidatorCandidate,
    failures: &[FailureTrace],
    rng: &mut Rng,
) -> Vec<ValidatorCandidate> {
    // Handle no-failure case: candidate survived, explore mutations
    if failures.is_empty() {
        return (0..3)
            .map(|i| {
                let mut rules = base.rules.clone();
                mutate_random_rule(&mut rules, rng);
                ValidatorCandidate {
                    id: format!("{}_explore{i}", base.id),
                    generation: base.generation + 1,
                    rules,
                }
            })
            .collect();
    }

    // Classify all failures
    let patterns: Vec<FailurePattern> = failures.iter().map(classify_failure).collect();

    let has_blast = patterns.contains(&FailurePattern::BlastZone)
        || patterns.contains(&FailurePattern::MovedIntoDanger);
    let has_self_bomb = patterns.contains(&FailurePattern::SelfBomb);
    let has_corner = patterns.contains(&FailurePattern::CornerTrap);

    let mut variants = Vec::new();

    // Variant 1: Increase blast avoidance
    if has_blast {
        let mut rules = base.rules.clone();
        for rule in &mut rules {
            if let ValidatorRule::AvoidBlast { lookahead } = rule {
                *lookahead = (*lookahead + 1).min(4);
            }
        }
        variants.push(ValidatorCandidate {
            id: format!("{}_blast", base.id),
            generation: base.generation + 1,
            rules,
        });
    }

    // Variant 2: Increase bomb distance
    if has_self_bomb {
        let mut rules = base.rules.clone();
        for rule in &mut rules {
            if let ValidatorRule::DistanceFromBomb { min_distance } = rule {
                *min_distance = (*min_distance + 1).min(3);
            }
        }
        variants.push(ValidatorCandidate {
            id: format!("{}_dist", base.id),
            generation: base.generation + 1,
            rules,
        });
    }

    // Variant 3: Increase dead-end avoidance
    if has_corner {
        let mut rules = base.rules.clone();
        for rule in &mut rules {
            if let ValidatorRule::AvoidDeadEnd { lookahead } = rule {
                *lookahead = (*lookahead + 1).min(4);
            }
        }
        variants.push(ValidatorCandidate {
            id: format!("{}_corner", base.id),
            generation: base.generation + 1,
            rules,
        });
    }

    // Variant 4: Random mutation (always)
    {
        let mut rules = base.rules.clone();
        mutate_random_rule(&mut rules, rng);
        variants.push(ValidatorCandidate {
            id: format!("{}_mutate", base.id),
            generation: base.generation + 1,
            rules,
        });
    }

    variants
}

// ── C7: Agent Loop ─────────────────────────────────────────────

/// Result of the agent loop.
#[cfg(feature = "bomber")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentLoopResult {
    /// Best candidate discovered.
    pub best_candidate: ValidatorCandidate,
    /// Evaluation of the best candidate.
    pub best_evaluation: ArenaEvaluation,
    /// Number of generations run.
    pub generations_run: u32,
    /// Total candidates evaluated across all generations.
    pub total_candidates_evaluated: usize,
}

/// The main agent loop: propose → evaluate → filter → iterate.
///
/// Uses a seeded RNG for reproducibility. The loop:
/// 1. Generate initial population from templates
/// 2. Evaluate each candidate in the arena
/// 3. Select top performers (top 50%)
/// 4. Propose mutations from failure traces
/// 5. Repeat until convergence or max generations
///
/// CPU-only — no GPU, no training weights, just rule search + arena evaluation.
#[cfg(feature = "bomber")]
pub struct AgentLoop {
    proposer: TemplateProposer,
    max_generations: u32,
    rounds_per_eval: u32,
    population_size: usize,
    convergence_threshold: f32,
    seed: u64,
}

#[cfg(feature = "bomber")]
impl Default for AgentLoop {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "bomber")]
impl AgentLoop {
    /// Create a new agent loop with default settings.
    pub fn new() -> Self {
        Self {
            proposer: TemplateProposer::new(),
            max_generations: 10,
            rounds_per_eval: 50,
            population_size: 10,
            convergence_threshold: 0.1,
            seed: 42,
        }
    }

    /// Set max generations (default: 10).
    pub fn max_generations(mut self, n: u32) -> Self {
        self.max_generations = n;
        self
    }

    /// Set rounds per evaluation (default: 50).
    pub fn rounds_per_eval(mut self, n: u32) -> Self {
        self.rounds_per_eval = n;
        self
    }

    /// Set population size (default: 10).
    pub fn population_size(mut self, n: usize) -> Self {
        self.population_size = n;
        self
    }

    /// Set convergence threshold — minimum score improvement to reset stagnation (default: 0.1).
    pub fn convergence_threshold(mut self, t: f32) -> Self {
        self.convergence_threshold = t;
        self
    }

    /// Set RNG seed for reproducibility (default: 42).
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Run the full agent loop.
    ///
    /// Returns the best candidate found.
    pub fn run(mut self) -> AgentLoopResult {
        let mut rng = Rng::with_seed(self.seed);

        // 1. Generate initial population
        let mut population = self
            .proposer
            .propose_initial(self.population_size, &mut rng);
        let mut best_score = f32::NEG_INFINITY;
        let mut best_candidate = population[0].clone();
        let mut best_evaluation = ArenaEvaluation {
            candidate_id: String::new(),
            rounds: 0,
            survival_rate: 0.0,
            kill_rate: 0.0,
            avg_score: f32::NEG_INFINITY,
            failure_traces: Vec::new(),
        };
        let mut generations_without_improvement = 0u32;
        let mut total_evaluated = 0usize;
        let mut generations_completed = 0u32;

        println!("Agent Loop — Starting optimization");
        println!(
            "  Config: pop={} gens={} rounds/gen={}",
            self.population_size, self.max_generations, self.rounds_per_eval
        );

        for generation in 0..self.max_generations {
            generations_completed = generation + 1;

            // 2. Evaluate each candidate
            let mut scored: Vec<(ValidatorCandidate, ArenaEvaluation)> = population
                .into_iter()
                .map(|c| {
                    let eval = evaluate_validator(&c, self.rounds_per_eval);
                    total_evaluated += 1;
                    (c, eval)
                })
                .collect();

            // 3. Sort by avg_score descending
            scored.sort_by(|a, b| {
                b.1.avg_score
                    .partial_cmp(&a.1.avg_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // 4. Update best
            let gen_best_score = scored[0].1.avg_score;
            if gen_best_score > best_score + self.convergence_threshold {
                best_score = gen_best_score;
                best_candidate = scored[0].0.clone();
                best_evaluation = scored[0].1.clone();
                generations_without_improvement = 0;
            } else {
                generations_without_improvement += 1;
            }

            println!(
                "  Gen {:>2}/{}: best={:>6.1} gen_best={:>6.1} pop={:>2} stagnation={}",
                generations_completed,
                self.max_generations,
                best_score,
                gen_best_score,
                scored.len(),
                generations_without_improvement,
            );

            // 5. Check convergence
            if generations_without_improvement >= 3 {
                println!("  ✓ Converged — no improvement for 3 generations");
                break;
            }

            // 6. Select top 50%
            let survivor_count = (scored.len() / 2).max(1);
            let survivors: Vec<(ValidatorCandidate, ArenaEvaluation)> =
                scored.into_iter().take(survivor_count).collect();

            // 7. Propose mutations from survivors
            let mut next_pop = Vec::new();
            for (candidate, eval) in &survivors {
                next_pop.push(candidate.clone());
                let variants = propose_from_trace(candidate, &eval.failure_traces, &mut rng);
                next_pop.extend(variants);
            }

            // Fill remaining slots with fresh candidates
            while next_pop.len() < self.population_size {
                let fresh = self.proposer.propose_initial(1, &mut rng);
                next_pop.extend(fresh);
            }

            // Cap population to configured size (truncate excess variants)
            next_pop.truncate(self.population_size);
            population = next_pop;
        }

        AgentLoopResult {
            best_candidate,
            best_evaluation,
            generations_run: generations_completed,
            total_candidates_evaluated: total_evaluated,
        }
    }
}
