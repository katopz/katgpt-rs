//! Fog-of-War Tactical Puzzle TUI — Exploration Strategies
//!
//! Three exploration strategies under fog of war:
//! - BF (🐻): BFS to nearest frontier
//! - AI (🐰): Heuristic frontier scoring
//! - Hybrid (🦊): AI region selection + BF pathfinding
//!
//! The player only sees tiles within BFS vision radius (blocked by walls).
//! Explored tiles are remembered (dimmed), hidden tiles show ❓.
//!
//! Game flow under fog:
//!   1. Explore to discover keys, boxes, levers, traps
//!   2. Collect keys (avoid traps and boss)
//!   3. Pull levers in correct order (opens bridge)
//!   4. Cross bridge, open boxes, reach goal
//!
//! Run: `cargo run --example tactical_09_fog`

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{self, Stdout};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::{Frame, Terminal};

// ── Emoji ──────────────────────────────────────────────────────

const BEAR: &str = "🐻";
const BOSS_LIVE: &str = "👹";
const BOSS_DEAD: &str = "💀";
const TRAP: &str = "🪤";
const KEY_EMOJI: &str = "🔑";
const BOX_CLOSED: &str = "📦";
const BOX_OPEN: &str = "📭";
const LEVER: &str = "🔧";
const LEVER_ON: &str = "🛠️";
const BRIDGE_CLOSED: &str = "🌊";
const BRIDGE_OPEN: &str = "🌉";
const GOAL: &str = "🚪";
const GOAL_WIN: &str = "🏆";
const WALL: &str = "🧱";
const FLOOR: &str = "◼️";
const CHECK: &str = "✓";
const ARROW: &str = "▸";
const SKULL: &str = "☠";
const RABBIT: &str = "🐰";
const FOX: &str = "🦊";
const FOG: &str = "❓";

// ── Timing & Limits ────────────────────────────────────────────

const TICK_MS: u64 = 50;
const MOVE_MS: u64 = 80;
const BOSS_SPEED: u32 = 2;
const VISION_RADIUS: usize = 4;
const MAX_STEPS: usize = 500;

// ── Directions ─────────────────────────────────────────────────

const DIR_DELTA: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

fn action_name(action: usize) -> &'static str {
    match action {
        0 => "↑ Up",
        1 => "↓ Down",
        2 => "← Left",
        3 => "→ Right",
        _ => "???",
    }
}

// ── 16×16 Strategic Map ────────────────────────────────────────
//
//  Same layout as tactical_07:
//  Top area (rows 1-10): keys, boss, traps, levers
//  Row 11: solid wall with bridge chokepoint at cols 7-9
//  Bottom area (rows 12-14): boxes, goal

const MAP: &str = "\
# # # # # # # # # # # # # # # #
# B . . . . . . . . . . k . . #
# . # # . # . . # . # . . # . #
# . . . . . . . . . . . . . . #
# . # . ! # . # # . # ! . # . #
# . . . . . . . . . . . . . . #
# . # # . . . . . . . # # # . #
# . . j . . ! O . . . . . . . #
# . # . # . # . . # . # . # . #
# . . . . . 1 . . 2 . . . . . #
# . # # . . . . 3 . . . # # . #
# # # # # # # = = = # # # # # #
# . . a . . . . . . b . . . . #
# . . . . . . . . . . . . . . #
# . . . . . . . . . . . . G . #
# # # # # # # # # # # # # # # #";

// ── Types ──────────────────────────────────────────────────────

/// Full game state including boss position and puzzle progress.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StrategicState {
    r: usize,
    c: usize,
    keys_held: u8,
    keys_used: u8,
    boxes_opened: u8,
    lever_state: u8,
    bridge_open: bool,
    boss_r: usize,
    boss_c: usize,
    boss_alive: bool,
    total_cost: u32,
    dead: bool,
    boss_last_seen_player: Option<(usize, usize)>,
}

/// Fog-of-war state tracking discovered information.
#[derive(Clone, Debug)]
struct FogState {
    seen: HashSet<(usize, usize)>,
    visible: HashSet<(usize, usize)>,
    discovered_keys: Vec<bool>,
    discovered_boxes: Vec<bool>,
    discovered_levers: Vec<bool>,
    discovered_traps: HashSet<(usize, usize)>,
    goal_pos: Option<(usize, usize)>,
    bridge_seen: bool,
    boss_last_known: Option<(usize, usize)>,
    boss_alive: Option<bool>,
}

impl FogState {
    fn new(num_keys: usize, num_boxes: usize, num_levers: usize) -> Self {
        Self {
            seen: HashSet::new(),
            visible: HashSet::new(),
            discovered_keys: vec![false; num_keys],
            discovered_boxes: vec![false; num_boxes],
            discovered_levers: vec![false; num_levers],
            discovered_traps: HashSet::new(),
            goal_pos: None,
            bridge_seen: false,
            boss_last_known: None,
            boss_alive: None,
        }
    }

    /// Update fog state from current visible set.
    fn update(
        &mut self,
        visible: &HashSet<(usize, usize)>,
        game: &StrategicGame,
        state: &StrategicState,
    ) {
        self.visible = visible.clone();
        self.seen.extend(visible.iter());

        for &pos in visible {
            for (i, &kpos) in game.keys.iter().enumerate() {
                if kpos == pos {
                    self.discovered_keys[i] = true;
                }
            }
            for (j, &bpos) in game.boxes.iter().enumerate() {
                if bpos == pos {
                    self.discovered_boxes[j] = true;
                }
            }
            for (l, &lpos) in game.levers.iter().enumerate() {
                if lpos == pos {
                    self.discovered_levers[l] = true;
                }
            }
            if game.traps.contains(&pos) {
                self.discovered_traps.insert(pos);
            }
            if game.goal == pos {
                self.goal_pos = Some(pos);
            }
            if game.bridge.contains(&pos) {
                self.bridge_seen = true;
            }
        }

        // Track boss position when visible
        if state.boss_alive {
            let boss_pos = (state.boss_r, state.boss_c);
            if visible.contains(&boss_pos) {
                self.boss_last_known = Some(boss_pos);
                self.boss_alive = Some(true);
            }
        } else {
            // Boss is dead in game state
            if self.boss_alive.is_none() || self.boss_alive == Some(true) {
                self.boss_alive = Some(false);
                self.boss_last_known = None;
            }
        }
    }

    /// Find frontier tiles: seen floor tiles with at least one unseen neighbor.
    fn frontier_tiles(&self, grid: &[Vec<char>]) -> Vec<(usize, usize)> {
        let rows = grid.len();
        let cols = grid.first().map_or(0, |r| r.len());
        let mut frontiers = Vec::new();

        for &pos in &self.seen {
            let (r, c) = pos;
            if r >= rows || c >= cols || grid[r][c] == '#' {
                continue;
            }
            for &(dr, dc) in &DIR_DELTA {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if nr < 0 || nc < 0 {
                    continue;
                }
                let nr = nr as usize;
                let nc = nc as usize;
                if nr >= rows || nc >= cols {
                    continue;
                }
                if !self.seen.contains(&(nr, nc)) {
                    frontiers.push(pos);
                    break;
                }
            }
        }
        frontiers
    }
}

/// Computed solution with exploration metadata.
#[derive(Clone)]
struct SolveResult {
    steps: usize,
    states: Vec<StrategicState>,
    fog_states: Vec<FogState>,
    solve_time_ms: u64,
    #[allow(dead_code)]
    discovered_at_step: usize,
    success: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SolveMode {
    BruteForce,
    Ai,
    Hybrid,
}

// ── Game Engine ────────────────────────────────────────────────

/// The strategic puzzle game engine (reused from tactical_07).
struct StrategicGame {
    grid: Vec<Vec<char>>,
    start: (usize, usize),
    boss_start: (usize, usize),
    keys: Vec<(usize, usize)>,
    boxes: Vec<(usize, usize)>,
    levers: Vec<(usize, usize)>,
    traps: HashSet<(usize, usize)>,
    bridge: HashSet<(usize, usize)>,
    goal: (usize, usize),
    key_mapping: [usize; 2],
    target_lever_mask: u8,
}

impl StrategicGame {
    fn new(map_str: &str, seed: u64) -> Self {
        let mut grid = Vec::new();
        let mut start = (0, 0);
        let mut goal = (0, 0);
        let mut bridge = HashSet::new();
        let mut bridge_row = usize::MAX;

        for (r, line) in map_str.lines().enumerate() {
            let mut row = Vec::new();
            for (c, token) in line.split_whitespace().enumerate() {
                let ch = token.chars().next().unwrap();
                match ch {
                    'B' => {
                        start = (r, c);
                        row.push('.');
                    }
                    'G' => {
                        goal = (r, c);
                        row.push('.');
                    }
                    '=' => {
                        bridge.insert((r, c));
                        bridge_row = r;
                        row.push('=');
                    }
                    '#' => row.push('#'),
                    _ => row.push('.'),
                }
            }
            grid.push(row);
        }

        let mut upper_floors = Vec::new();
        let mut lower_floors = Vec::new();
        for (r, row) in grid.iter().enumerate() {
            for (c, &cell) in row.iter().enumerate() {
                if cell == '.' && (r, c) != start && (r, c) != goal {
                    if r < bridge_row {
                        upper_floors.push((r, c));
                    } else if r > bridge_row {
                        lower_floors.push((r, c));
                    }
                }
            }
        }

        let mut rng_s = seed;
        for i in (1..upper_floors.len()).rev() {
            rng_s = rng_s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_s as usize) % (i + 1);
            upper_floors.swap(i, j);
        }
        for i in (1..lower_floors.len()).rev() {
            rng_s = rng_s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_s as usize) % (i + 1);
            lower_floors.swap(i, j);
        }

        let boss_start = upper_floors[0];
        let mut levers = vec![upper_floors[1], upper_floors[2], upper_floors[3]];
        let traps: HashSet<_> = [upper_floors[4], upper_floors[5]].into_iter().collect();
        let mut keys = vec![upper_floors[6], upper_floors[7]];
        let mut boxes = vec![lower_floors[0], lower_floors[1]];
        levers.sort();
        keys.sort();
        boxes.sort();

        let (key_mapping, target_lever_mask) = Self::generate_config(seed);

        Self {
            grid,
            start,
            boss_start,
            keys,
            boxes,
            levers,
            traps,
            bridge,
            goal,
            key_mapping,
            target_lever_mask,
        }
    }

    fn generate_config(seed: u64) -> ([usize; 2], u8) {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            s
        };

        let mut key_mapping = [0, 1];
        for i in (1..2).rev() {
            let j = (next() as usize) % (i + 1);
            key_mapping.swap(i, j);
        }
        if key_mapping[0] == 0 {
            key_mapping.swap(0, 1);
        }

        let target_lever_mask = ((next() as usize) % 6 + 1) as u8;
        (key_mapping, target_lever_mask)
    }

    fn rows(&self) -> usize {
        self.grid.len()
    }
    fn cols(&self) -> usize {
        self.grid.first().map_or(0, |r| r.len())
    }

    fn initial_state(&self) -> StrategicState {
        StrategicState {
            r: self.start.0,
            c: self.start.1,
            keys_held: 0,
            keys_used: 0,
            boxes_opened: 0,
            lever_state: 0,
            bridge_open: false,
            boss_r: self.boss_start.0,
            boss_c: self.boss_start.1,
            boss_alive: true,
            total_cost: 0,
            dead: false,
            boss_last_seen_player: None,
        }
    }

    /// Boss AI: line-of-sight chase, investigate last known, or idle.
    fn boss_next_move(&self, state: &mut StrategicState) -> (usize, usize) {
        if !state.boss_alive {
            return (state.boss_r, state.boss_c);
        }
        let boss_pos = (state.boss_r, state.boss_c);
        let player_pos = (state.r, state.c);
        if boss_pos == player_pos {
            return boss_pos;
        }

        // Compute boss's vision from its position
        let boss_vision = compute_visible(&self.grid, boss_pos, state.bridge_open, &self.bridge);

        if boss_vision.contains(&player_pos) {
            // CHASE: player is visible → remember position, BFS toward player
            state.boss_last_seen_player = Some(player_pos);
            self.bfs_boss_step(boss_pos, player_pos)
        } else if let Some(last_seen) = state.boss_last_seen_player {
            if last_seen == boss_pos {
                // Reached last known position, can't see player → go idle
                state.boss_last_seen_player = None;
                boss_pos
            } else {
                // INVESTIGATE: move toward last known player position
                self.bfs_boss_step(boss_pos, last_seen)
            }
        } else {
            // IDLE: no player info → stay put
            boss_pos
        }
    }

    /// BFS from boss toward target. Boss does NOT avoid traps — it can be lured!
    /// Only walls and closed bridges block movement.
    fn bfs_boss_step(&self, start: (usize, usize), target: (usize, usize)) -> (usize, usize) {
        if start == target {
            return start;
        }
        let rows = self.rows();
        let cols = self.cols();
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut came_from = HashMap::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(pos) = queue.pop_front() {
            if pos == target {
                let mut current = target;
                loop {
                    let prev = came_from[&current];
                    if prev == start {
                        return current;
                    }
                    current = prev;
                }
            }
            for &(dr, dc) in &DIR_DELTA {
                let nr = pos.0 as isize + dr;
                let nc = pos.1 as isize + dc;
                if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                    continue;
                }
                let next = (nr as usize, nc as usize);
                if visited.contains(&next) {
                    continue;
                }
                if self.grid[next.0][next.1] == '#' {
                    continue;
                }
                if self.bridge.contains(&next) {
                    continue;
                }
                // Boss does NOT avoid traps — player can lure boss through them
                visited.insert(next);
                came_from.insert(next, pos);
                queue.push_back(next);
            }
        }
        start
    }

    /// Handle automatic interactions at current position.
    fn interact(&self, state: &mut StrategicState) {
        let pos = (state.r, state.c);

        for (i, &kpos) in self.keys.iter().enumerate() {
            if kpos == pos && (state.keys_held & (1 << i)) == 0 {
                state.keys_held |= 1 << i;
            }
        }

        for (j, &bpos) in self.boxes.iter().enumerate() {
            if bpos == pos && (state.boxes_opened & (1 << j)) == 0 && state.keys_held != 0 {
                for k in 0..self.keys.len() {
                    if (state.keys_held & (1 << k)) != 0 && self.key_mapping[k] == j {
                        state.boxes_opened |= 1 << j;
                        state.keys_held &= !(1 << k);
                        state.keys_used |= 1 << k;
                        break;
                    }
                }
            }
        }

        for (l, &lpos) in self.levers.iter().enumerate() {
            if lpos == pos {
                state.lever_state ^= 1 << l;
                state.bridge_open = state.lever_state == self.target_lever_mask;
            }
        }
    }

    /// Apply action WITH boss simulation.
    fn apply_action(&self, state: &StrategicState, action: usize) -> Option<StrategicState> {
        if action > 3 {
            return None;
        }
        let mut next = state.clone();

        let (dr, dc) = DIR_DELTA[action];
        let nr = next.r as isize + dr;
        let nc = next.c as isize + dc;
        if nr < 0 || nc < 0 || nr >= self.rows() as isize || nc >= self.cols() as isize {
            return None;
        }
        let nr = nr as usize;
        let nc = nc as usize;

        if self.grid[nr][nc] == '#' {
            return None;
        }
        if self.bridge.contains(&(nr, nc)) && !next.bridge_open {
            return None;
        }
        if self.traps.contains(&(nr, nc)) {
            next.dead = true;
            return Some(next);
        }

        next.r = nr;
        next.c = nc;
        next.total_cost += 1;

        self.interact(&mut next);

        if next.boss_alive && next.total_cost.is_multiple_of(BOSS_SPEED) {
            let (new_br, new_bc) = self.boss_next_move(&mut next);
            next.boss_r = new_br;
            next.boss_c = new_bc;
            if self.traps.contains(&(new_br, new_bc)) {
                next.boss_alive = false;
            }
        }

        if next.boss_alive && next.boss_r == next.r && next.boss_c == next.c {
            next.dead = true;
        }

        Some(next)
    }
}

// ── Vision System ──────────────────────────────────────────────

/// BFS flood-fill from pos to compute visible tiles.
/// Can see wall tiles but not THROUGH them.
/// Closed bridge tiles block vision like walls.
/// Open bridge tiles allow vision through.
fn compute_visible(
    grid: &[Vec<char>],
    pos: (usize, usize),
    bridge_open: bool,
    bridge: &HashSet<(usize, usize)>,
) -> HashSet<(usize, usize)> {
    let mut visible = HashSet::new();
    let mut queue = VecDeque::new();
    let rows = grid.len();
    let cols = grid.first().map_or(0, |r| r.len());

    queue.push_back((pos.0, pos.1, 0usize));
    visible.insert(pos);

    while let Some((r, c, dist)) = queue.pop_front() {
        if dist >= VISION_RADIUS {
            continue;
        }
        for &(dr, dc) in &DIR_DELTA {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                continue;
            }
            let next = (nr as usize, nc as usize);

            if visible.contains(&next) {
                continue;
            }

            // Wall: visible but blocks further vision
            if grid[next.0][next.1] == '#' {
                visible.insert(next);
                continue;
            }

            // Closed bridge: visible but blocks further vision
            if bridge.contains(&next) && !bridge_open {
                visible.insert(next);
                continue;
            }

            // Floor or open bridge: visible and continue BFS
            visible.insert(next);
            queue.push_back((next.0, next.1, dist + 1));
        }
    }
    visible
}

// ── Pathfinding ────────────────────────────────────────────────

fn manhattan_dist(a: (usize, usize), b: (usize, usize)) -> usize {
    (a.0 as isize - b.0 as isize).unsigned_abs() + (a.1 as isize - b.1 as isize).unsigned_abs()
}

/// BFS from `from` through seen tiles to find nearest target.
/// Returns the first action (direction from `from`) on the shortest path.
fn bfs_first_action(
    game: &StrategicGame,
    from: (usize, usize),
    targets: &HashSet<(usize, usize)>,
    seen: &HashSet<(usize, usize)>,
    state: &StrategicState,
    discovered_traps: &HashSet<(usize, usize)>,
) -> Option<usize> {
    if targets.is_empty() {
        return None;
    }

    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();
    let mut came_from: HashMap<(usize, usize), ((usize, usize), usize)> = HashMap::new();

    queue.push_back(from);
    visited.insert(from);

    while let Some(pos) = queue.pop_front() {
        for (action, &(dr, dc)) in DIR_DELTA.iter().enumerate() {
            let nr = pos.0 as isize + dr;
            let nc = pos.1 as isize + dc;
            if nr < 0 || nc < 0 || nr >= game.rows() as isize || nc >= game.cols() as isize {
                continue;
            }
            let next = (nr as usize, nc as usize);

            if visited.contains(&next) {
                continue;
            }
            if !seen.contains(&next) {
                continue;
            }
            if game.grid[next.0][next.1] == '#' {
                continue;
            }
            if discovered_traps.contains(&next) {
                continue;
            }
            if game.bridge.contains(&next) && !state.bridge_open {
                continue;
            }

            visited.insert(next);
            came_from.insert(next, (pos, action));

            if targets.contains(&next) && next != from {
                let mut current = next;
                loop {
                    let (parent, act) = came_from[&current];
                    if parent == from {
                        return Some(act);
                    }
                    current = parent;
                }
            }

            queue.push_back(next);
        }
    }
    None
}

/// Return any passable action (last-resort fallback).
fn any_passable_action(
    game: &StrategicGame,
    state: &StrategicState,
    discovered_traps: &HashSet<(usize, usize)>,
) -> Option<usize> {
    for (action, &(dr, dc)) in DIR_DELTA.iter().enumerate() {
        let nr = state.r as isize + dr;
        let nc = state.c as isize + dc;
        if nr < 0 || nc < 0 || nr >= game.rows() as isize || nc >= game.cols() as isize {
            continue;
        }
        let next = (nr as usize, nc as usize);
        if game.grid[next.0][next.1] == '#' {
            continue;
        }
        if game.bridge.contains(&next) && !state.bridge_open {
            continue;
        }
        if discovered_traps.contains(&next) {
            continue;
        }
        return Some(action);
    }
    None
}

// ── Explorer Trait ─────────────────────────────────────────────

trait Explorer {
    fn choose_action(
        &mut self,
        game: &StrategicGame,
        state: &StrategicState,
        fog: &FogState,
    ) -> Option<usize>;
}

/// Try to lure the boss to a trap. The player positions adjacent to a trap
/// on the FAR side from the boss, so the boss's BFS path goes through the trap.
/// Returns None if no lure opportunity exists.
fn lure_boss_to_trap(
    game: &StrategicGame,
    state: &StrategicState,
    fog: &FogState,
) -> Option<usize> {
    let boss_pos = fog.boss_last_known?;
    let alive = fog.boss_alive?;
    if !alive || !fog.visible.contains(&boss_pos) {
        return None;
    }

    let boss_dist = manhattan_dist((state.r, state.c), boss_pos);
    if boss_dist > 8 {
        return None;
    }

    let mut best_action = None;
    let mut best_score = i32::MIN;

    for &trap_pos in &fog.discovered_traps {
        for &(dr, dc) in &DIR_DELTA {
            let adj_r = trap_pos.0 as isize + dr;
            let adj_c = trap_pos.1 as isize + dc;
            if adj_r < 0 || adj_c < 0 {
                continue;
            }
            let adj = (adj_r as usize, adj_c as usize);
            if adj.0 >= game.rows() || adj.1 >= game.cols() {
                continue;
            }
            if game.grid[adj.0][adj.1] == '#' {
                continue;
            }
            if game.bridge.contains(&adj) && !state.bridge_open {
                continue;
            }
            if fog.discovered_traps.contains(&adj) {
                continue;
            }
            if !fog.seen.contains(&adj) {
                continue;
            }

            let trap_to_boss = manhattan_dist(trap_pos, boss_pos);
            let adj_to_boss = manhattan_dist(adj, boss_pos);

            if adj_to_boss <= trap_to_boss {
                continue;
            }

            let dist_to_lure = manhattan_dist((state.r, state.c), adj) as i32;
            let score = -dist_to_lure;

            if score > best_score {
                best_score = score;
                let target_set = HashSet::from([adj]);
                best_action = bfs_first_action(
                    game,
                    (state.r, state.c),
                    &target_set,
                    &fog.seen,
                    state,
                    &fog.discovered_traps,
                );
            }
        }
    }

    best_action
}

/// Navigate to the highest-priority known uncompleted target.
fn navigate_to_known_target(
    game: &StrategicGame,
    state: &StrategicState,
    fog: &FogState,
) -> Option<usize> {
    // Priority 0: Lure boss to trap (when boss is chasing and trap is between boss and player)
    if state.boss_alive
        && !fog.discovered_traps.is_empty()
        && let Some(action) = lure_boss_to_trap(game, state, fog)
    {
        return Some(action);
    }

    // 1. Uncollected keys
    let keys: HashSet<_> = game
        .keys
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            fog.discovered_keys[*i]
                && (state.keys_held & (1 << i)) == 0
                && (state.keys_used & (1 << i)) == 0
        })
        .map(|(_, &pos)| pos)
        .collect();
    if !keys.is_empty()
        && let Some(a) = bfs_first_action(
            game,
            (state.r, state.c),
            &keys,
            &fog.seen,
            state,
            &fog.discovered_traps,
        )
    {
        return Some(a);
    }

    // 2. Levers (bridge not open)
    if !state.bridge_open {
        let levers: HashSet<_> = game
            .levers
            .iter()
            .enumerate()
            .filter(|(l, _)| fog.discovered_levers[*l])
            .map(|(_, &pos)| pos)
            .collect();
        if !levers.is_empty()
            && let Some(a) = bfs_first_action(
                game,
                (state.r, state.c),
                &levers,
                &fog.seen,
                state,
                &fog.discovered_traps,
            )
        {
            return Some(a);
        }
    }

    // 3. Unopened boxes (if holding keys)
    if state.keys_held != 0 {
        let boxes: HashSet<_> = game
            .boxes
            .iter()
            .enumerate()
            .filter(|(j, _)| fog.discovered_boxes[*j] && (state.boxes_opened & (1 << j)) == 0)
            .map(|(_, &pos)| pos)
            .collect();
        if !boxes.is_empty()
            && let Some(a) = bfs_first_action(
                game,
                (state.r, state.c),
                &boxes,
                &fog.seen,
                state,
                &fog.discovered_traps,
            )
        {
            return Some(a);
        }
    }

    // 4. Goal (if conditions met)
    if let Some(goal) = fog.goal_pos
        && state.bridge_open
    {
        let all_boxes = (1 << game.boxes.len()) - 1;
        if state.boxes_opened == all_boxes {
            let goals = HashSet::from([goal]);
            if let Some(a) = bfs_first_action(
                game,
                (state.r, state.c),
                &goals,
                &fog.seen,
                state,
                &fog.discovered_traps,
            ) {
                return Some(a);
            }
        }
    }

    // 5. Any unopened box (keep exploring toward it)
    let boxes: HashSet<_> = game
        .boxes
        .iter()
        .enumerate()
        .filter(|(j, _)| fog.discovered_boxes[*j] && (state.boxes_opened & (1 << j)) == 0)
        .map(|(_, &pos)| pos)
        .collect();
    if !boxes.is_empty()
        && let Some(a) = bfs_first_action(
            game,
            (state.r, state.c),
            &boxes,
            &fog.seen,
            state,
            &fog.discovered_traps,
        )
    {
        return Some(a);
    }

    // 6. Goal (even if conditions not met)
    if let Some(goal) = fog.goal_pos {
        let goals = HashSet::from([goal]);
        if let Some(a) = bfs_first_action(
            game,
            (state.r, state.c),
            &goals,
            &fog.seen,
            state,
            &fog.discovered_traps,
        ) {
            return Some(a);
        }
    }

    // 7. Any passable direction
    any_passable_action(game, state, &fog.discovered_traps)
}

// ── BfExplorer ─────────────────────────────────────────────────

/// Brute-force explorer: BFS to nearest frontier.
struct BfExplorer;

impl Explorer for BfExplorer {
    fn choose_action(
        &mut self,
        game: &StrategicGame,
        state: &StrategicState,
        fog: &FogState,
    ) -> Option<usize> {
        let frontiers = fog.frontier_tiles(&game.grid);

        if !frontiers.is_empty() {
            let target_set: HashSet<_> = frontiers.into_iter().collect();
            if let Some(action) = bfs_first_action(
                game,
                (state.r, state.c),
                &target_set,
                &fog.seen,
                state,
                &fog.discovered_traps,
            ) {
                return Some(action);
            }
        }

        navigate_to_known_target(game, state, fog)
    }
}

/// Dodge only when the boss is visible at distance ≤1 and preferred action steps
/// onto the boss tile (certain death). Picks the best alternative direction that
/// moves away from the boss. Minimal intervention — only prevents immediate collision.
fn dodge_boss_if_adjacent(
    game: &StrategicGame,
    state: &StrategicState,
    fog: &FogState,
    preferred: usize,
) -> usize {
    let Some(boss_pos) = fog.boss_last_known else {
        return preferred;
    };
    let Some(alive) = fog.boss_alive else {
        return preferred;
    };
    if !alive || !fog.visible.contains(&boss_pos) {
        return preferred;
    }

    // Check if preferred action steps directly onto the boss tile
    let (dr, dc) = DIR_DELTA[preferred];
    let new_r = state.r as isize + dr;
    let new_c = state.c as isize + dc;
    if new_r < 0 || new_c < 0 {
        return preferred;
    }
    let new_pos = (new_r as usize, new_c as usize);
    if new_pos != boss_pos {
        return preferred; // not stepping on boss — no dodge needed
    }

    // Stepping onto boss — find best alternative that doesn't step on boss
    let mut best_action = preferred;
    let mut best_dist = 0i32;

    for (action, &(adr, adc)) in DIR_DELTA.iter().enumerate() {
        let ar = state.r as isize + adr;
        let ac = state.c as isize + adc;
        if ar < 0 || ac < 0 {
            continue;
        }
        let ar = ar as usize;
        let ac = ac as usize;
        if ar >= game.rows() || ac >= game.cols() {
            continue;
        }
        if game.grid[ar][ac] == '#' {
            continue;
        }
        if game.bridge.contains(&(ar, ac)) && !state.bridge_open {
            continue;
        }
        if fog.discovered_traps.contains(&(ar, ac)) {
            continue;
        }
        if (ar, ac) == boss_pos {
            continue; // skip boss tile
        }

        let alt_dist = manhattan_dist((ar, ac), boss_pos);
        if alt_dist as i32 > best_dist {
            best_dist = alt_dist as i32;
            best_action = action;
        }
    }

    best_action
}

// ── AiExplorer ─────────────────────────────────────────────────

/// AI explorer: heuristic frontier scoring.
/// Scores frontiers by: unseen neighbors, chokepoints, bridge proximity.
struct AiExplorer;

impl AiExplorer {
    /// Score a frontier tile by heuristic.
    fn score_frontier(
        &self,
        pos: (usize, usize),
        game: &StrategicGame,
        state: &StrategicState,
        fog: &FogState,
    ) -> i32 {
        let mut score = 0i32;

        // +1 per unseen neighbor
        for &(dr, dc) in &DIR_DELTA {
            let nr = pos.0 as isize + dr;
            let nc = pos.1 as isize + dc;
            if nr < 0 || nc < 0 {
                continue;
            }
            let next = (nr as usize, nc as usize);
            if next.0 >= game.rows() || next.1 >= game.cols() {
                continue;
            }
            if !fog.seen.contains(&next) {
                score += 1;
            }
        }

        // +2 if adjacent to a chokepoint (exactly 2 passable seen neighbors)
        // +1 if adjacent to a dead-end (exactly 1 passable seen neighbor)
        for &(dr, dc) in &DIR_DELTA {
            let nr = pos.0 as isize + dr;
            let nc = pos.1 as isize + dc;
            if nr < 0 || nc < 0 {
                continue;
            }
            let neighbor = (nr as usize, nc as usize);
            if neighbor.0 >= game.rows() || neighbor.1 >= game.cols() {
                continue;
            }
            if !fog.seen.contains(&neighbor) || game.grid[neighbor.0][neighbor.1] == '#' {
                continue;
            }
            if game.bridge.contains(&neighbor) && !state.bridge_open {
                continue;
            }
            let passable_n = count_passable_neighbors(neighbor, game, fog, state);
            if passable_n == 2 {
                score += 2; // Chokepoint
            } else if passable_n == 1 {
                score += 1; // Dead-end
            }
        }

        // +3 if keys held and bridge seen and frontier near bridge
        if state.keys_held != 0 && fog.bridge_seen {
            for &bpos in &game.bridge {
                if manhattan_dist(pos, bpos) <= 3 {
                    score += 3;
                    break;
                }
            }
        }

        score
    }
}

fn count_passable_neighbors(
    pos: (usize, usize),
    game: &StrategicGame,
    fog: &FogState,
    state: &StrategicState,
) -> usize {
    DIR_DELTA
        .iter()
        .filter(|&&(dr, dc)| {
            let nr = pos.0 as isize + dr;
            let nc = pos.1 as isize + dc;
            if nr < 0 || nc < 0 {
                return false;
            }
            let next = (nr as usize, nc as usize);
            if next.0 >= game.rows() || next.1 >= game.cols() {
                return false;
            }
            if !fog.seen.contains(&next) {
                return false;
            }
            if game.grid[next.0][next.1] == '#' {
                return false;
            }
            if game.bridge.contains(&next) && !state.bridge_open {
                return false;
            }
            true
        })
        .count()
}

impl Explorer for AiExplorer {
    fn choose_action(
        &mut self,
        game: &StrategicGame,
        state: &StrategicState,
        fog: &FogState,
    ) -> Option<usize> {
        let frontiers = fog.frontier_tiles(&game.grid);

        if !frontiers.is_empty() {
            // Score each frontier and pick the best
            let best = frontiers
                .iter()
                .max_by_key(|&&pos| {
                    let score = self.score_frontier(pos, game, state, fog);
                    // Tiebreak: prefer closer frontiers
                    let dist = manhattan_dist(pos, (state.r, state.c));
                    (score, -(dist as i32))
                })
                .copied();

            if let Some(best_pos) = best {
                let target_set = HashSet::from([best_pos]);
                if let Some(action) = bfs_first_action(
                    game,
                    (state.r, state.c),
                    &target_set,
                    &fog.seen,
                    state,
                    &fog.discovered_traps,
                ) {
                    return Some(dodge_boss_if_adjacent(game, state, fog, action));
                }
            }

            // Fallback: try all frontiers as a set
            let target_set: HashSet<_> = frontiers.into_iter().collect();
            if let Some(action) = bfs_first_action(
                game,
                (state.r, state.c),
                &target_set,
                &fog.seen,
                state,
                &fog.discovered_traps,
            ) {
                return Some(dodge_boss_if_adjacent(game, state, fog, action));
            }
        }

        navigate_to_known_target(game, state, fog)
    }
}

// ── HybridExplorer ─────────────────────────────────────────────

/// Hybrid explorer: AI picks best frontier REGION, BF pathfinds to nearest in that region.
struct HybridExplorer;

impl HybridExplorer {
    /// Cluster frontiers into regions (within Manhattan distance 3).
    fn cluster_frontiers(&self, frontiers: &[(usize, usize)]) -> Vec<Vec<(usize, usize)>> {
        let mut clusters: Vec<Vec<(usize, usize)>> = Vec::new();
        let mut assigned: HashSet<usize> = HashSet::new();

        for (i, &f) in frontiers.iter().enumerate() {
            if assigned.contains(&i) {
                continue;
            }
            let mut cluster = vec![f];
            assigned.insert(i);

            let mut queue = VecDeque::new();
            queue.push_back(f);

            while let Some(pos) = queue.pop_front() {
                for (j, &other) in frontiers.iter().enumerate() {
                    if assigned.contains(&j) {
                        continue;
                    }
                    if manhattan_dist(pos, other) <= 3 {
                        assigned.insert(j);
                        cluster.push(other);
                        queue.push_back(other);
                    }
                }
            }
            clusters.push(cluster);
        }
        clusters
    }
}

impl Explorer for HybridExplorer {
    fn choose_action(
        &mut self,
        game: &StrategicGame,
        state: &StrategicState,
        fog: &FogState,
    ) -> Option<usize> {
        let frontiers = fog.frontier_tiles(&game.grid);

        if !frontiers.is_empty() {
            let clusters = self.cluster_frontiers(&frontiers);

            // Score each cluster by AI heuristic (sum of frontier scores)
            let ai = AiExplorer;
            let best_cluster = clusters.iter().max_by_key(|cluster| {
                let total_score: i32 = cluster
                    .iter()
                    .map(|&pos| ai.score_frontier(pos, game, state, fog))
                    .sum();
                total_score
            });

            if let Some(cluster) = best_cluster {
                // BF: find nearest frontier in the best cluster
                let target_set: HashSet<_> = cluster.iter().copied().collect();
                if let Some(action) = bfs_first_action(
                    game,
                    (state.r, state.c),
                    &target_set,
                    &fog.seen,
                    state,
                    &fog.discovered_traps,
                ) {
                    return Some(dodge_boss_if_adjacent(game, state, fog, action));
                }
            }

            // Fallback: try all frontiers
            let target_set: HashSet<_> = frontiers.into_iter().collect();
            if let Some(action) = bfs_first_action(
                game,
                (state.r, state.c),
                &target_set,
                &fog.seen,
                state,
                &fog.discovered_traps,
            ) {
                return Some(dodge_boss_if_adjacent(game, state, fog, action));
            }
        }

        navigate_to_known_target(game, state, fog)
    }
}

// ── Solve Function ─────────────────────────────────────────────

/// Run an explorer to solve the puzzle under fog of war.
fn solve_exploring<E: Explorer>(game: &StrategicGame, explorer: &mut E) -> SolveResult {
    let mut state = game.initial_state();
    let mut fog = FogState::new(game.keys.len(), game.boxes.len(), game.levers.len());
    let mut steps = 0;
    let mut all_states = vec![state.clone()];
    let mut discovered_at_step = 0;
    let mut all_discovered = false;
    let mut success = false;

    // Initial vision update
    let visible = compute_visible(
        &game.grid,
        (state.r, state.c),
        state.bridge_open,
        &game.bridge,
    );
    fog.update(&visible, game, &state);
    let mut all_fog = vec![fog.clone()];

    let start = Instant::now();

    while steps < MAX_STEPS {
        // Check win
        if (state.r, state.c) == game.goal
            && state.boxes_opened == (1 << game.boxes.len()) - 1
            && state.bridge_open
        {
            success = true;
            break;
        }

        // Check discovered
        if !all_discovered {
            let keys_ok = fog.discovered_keys.iter().all(|&d| d);
            let boxes_ok = fog.discovered_boxes.iter().all(|&d| d);
            let levers_ok = fog.discovered_levers.iter().all(|&d| d);
            let goal_ok = fog.goal_pos.is_some();
            let bridge_ok = fog.bridge_seen;
            if keys_ok && boxes_ok && levers_ok && goal_ok && bridge_ok {
                all_discovered = true;
                discovered_at_step = steps;
            }
        }

        // Check dead
        if state.dead {
            break;
        }

        // Choose action
        let Some(action) = explorer.choose_action(game, &state, &fog) else {
            break;
        };

        // Execute
        let Some(next_state) = game.apply_action(&state, action) else {
            break;
        };
        state = next_state;
        steps += 1;

        // Update vision
        let visible = compute_visible(
            &game.grid,
            (state.r, state.c),
            state.bridge_open,
            &game.bridge,
        );
        fog.update(&visible, game, &state);

        all_states.push(state.clone());
        all_fog.push(fog.clone());

        if state.dead {
            break;
        }
    }

    let solve_time_ms = start.elapsed().as_millis() as u64;

    SolveResult {
        steps,
        states: all_states,
        fog_states: all_fog,
        solve_time_ms,
        discovered_at_step,
        success,
    }
}

// ── TUI App ────────────────────────────────────────────────────

enum Phase {
    Exploring,
    Done,
}

struct AnimState {
    #[allow(dead_code)]
    from: (usize, usize),
    to: (usize, usize),
    action: usize,
    start: Instant,
    duration_ms: u64,
}

enum KeyAction {
    Continue,
    Restart,
    Quit,
}

struct App {
    game: StrategicGame,
    bf: SolveResult,
    ai: SolveResult,
    hybrid: SolveResult,
    mode: SolveMode,
    current: usize,
    anim: Option<AnimState>,
    auto_play: bool,
    seed: u64,
}

impl App {
    fn new(
        game: StrategicGame,
        bf: SolveResult,
        ai: SolveResult,
        hybrid: SolveResult,
        seed: u64,
    ) -> Self {
        Self {
            game,
            bf,
            ai,
            hybrid,
            mode: SolveMode::BruteForce,
            current: 0,
            anim: None,
            auto_play: true,
            seed,
        }
    }

    fn result(&self) -> &SolveResult {
        match self.mode {
            SolveMode::BruteForce => &self.bf,
            SolveMode::Ai => &self.ai,
            SolveMode::Hybrid => &self.hybrid,
        }
    }

    fn next_round(&mut self) {
        if !self.is_at_end() {
            return;
        }
        match self.mode {
            SolveMode::BruteForce => {
                self.mode = SolveMode::Ai;
                self.current = 0;
                self.anim = None;
                self.auto_play = true;
            }
            SolveMode::Ai => {
                self.mode = SolveMode::Hybrid;
                self.current = 0;
                self.anim = None;
                self.auto_play = true;
            }
            SolveMode::Hybrid => {}
        }
    }

    fn restart(&mut self) {
        let mut seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let (game, bf, ai, hybrid) = loop {
            let game = StrategicGame::new(MAP, seed);
            let mut bf_explorer = BfExplorer;
            let bf = solve_exploring(&game, &mut bf_explorer);
            let mut ai_explorer = AiExplorer;
            let ai = solve_exploring(&game, &mut ai_explorer);
            let mut hybrid_explorer = HybridExplorer;
            let hybrid = solve_exploring(&game, &mut hybrid_explorer);

            if bf.success || ai.success || hybrid.success {
                break (game, bf, ai, hybrid);
            }
            seed = seed.wrapping_add(1);
        };

        self.seed = seed;
        self.game = game;
        self.bf = bf;
        self.ai = ai;
        self.hybrid = hybrid;
        self.mode = SolveMode::BruteForce;
        self.current = 0;
        self.anim = None;
        self.auto_play = false;
    }

    fn current_state(&self) -> &StrategicState {
        &self.result().states[self.current]
    }

    fn current_fog(&self) -> &FogState {
        &self.result().fog_states[self.current]
    }

    fn total_steps(&self) -> usize {
        self.result().steps
    }

    fn is_at_start(&self) -> bool {
        self.current == 0
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.total_steps()
    }

    fn bear_pos(&self) -> (usize, usize) {
        if let Some(ref a) = self.anim {
            a.to
        } else {
            let s = self.current_state();
            (s.r, s.c)
        }
    }

    fn phase(&self) -> Phase {
        if self.is_at_end() {
            Phase::Done
        } else {
            Phase::Exploring
        }
    }

    fn start_animation(&mut self) {
        if self.is_at_end() {
            return;
        }
        let result = self.result();
        let prev = &result.states[self.current];
        let action = match result.states.get(self.current + 1) {
            Some(next) => {
                if next.r < prev.r {
                    0
                } else if next.r > prev.r {
                    1
                } else if next.c < prev.c {
                    2
                } else {
                    3
                }
            }
            None => return,
        };
        let (dr, dc) = DIR_DELTA[action];
        let to = (
            (prev.r as isize + dr) as usize,
            (prev.c as isize + dc) as usize,
        );

        self.anim = Some(AnimState {
            from: (prev.r, prev.c),
            to,
            action,
            start: Instant::now(),
            duration_ms: MOVE_MS,
        });
    }

    fn tick_animation(&mut self) -> bool {
        let Some(ref anim) = self.anim else {
            return false;
        };
        if anim.start.elapsed().as_millis() as u64 >= anim.duration_ms {
            self.anim = None;
            self.current += 1;
            return true;
        }
        false
    }
}

// ── Main ───────────────────────────────────────────────────────

fn main() -> io::Result<()> {
    let mut seed = 42u64;
    let (game, bf, ai, hybrid) = loop {
        let game = StrategicGame::new(MAP, seed);
        eprintln!(
            "🎯 Config: seed={seed}, key_mapping={:?}, target_lever_mask=0b{:03b}",
            game.key_mapping, game.target_lever_mask,
        );

        let mut bf_explorer = BfExplorer;
        let bf = solve_exploring(&game, &mut bf_explorer);
        eprintln!(
            "🐻 BF: {} steps · {}ms · {}",
            bf.steps,
            bf.solve_time_ms,
            if bf.success { "✓" } else { "✗" }
        );

        let mut ai_explorer = AiExplorer;
        let ai = solve_exploring(&game, &mut ai_explorer);
        eprintln!(
            "🐰 AI: {} steps · {}ms · {}",
            ai.steps,
            ai.solve_time_ms,
            if ai.success { "✓" } else { "✗" }
        );

        let mut hybrid_explorer = HybridExplorer;
        let hybrid = solve_exploring(&game, &mut hybrid_explorer);
        eprintln!(
            "🦊 Hybrid: {} steps · {}ms · {}",
            hybrid.steps,
            hybrid.solve_time_ms,
            if hybrid.success { "✓" } else { "✗" }
        );

        if bf.success || ai.success || hybrid.success {
            break (game, bf, ai, hybrid);
        }
        eprintln!("   ⚠ All failed, retrying with seed {}", seed + 1);
        seed = seed.wrapping_add(1);
    };
    eprintln!();

    let mut terminal = setup()?;
    let mut app = App::new(game, bf, ai, hybrid, seed);
    let res = run_with(&mut terminal, &mut app);
    teardown(&mut terminal)?;
    res
}

fn setup() -> io::Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(stdout))
}

fn teardown(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> io::Result<()> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn run_with(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &mut App) -> io::Result<()> {
    loop {
        terminal.draw(|f| draw(f, app))?;

        let completed = app.tick_animation();
        if completed && app.auto_play && !app.is_at_end() {
            app.start_animation();
        }
        if app.is_at_end() {
            app.auto_play = false;
        }

        let timeout = if app.anim.is_some() || app.auto_play {
            Duration::from_millis(TICK_MS)
        } else {
            Duration::from_millis(100)
        };

        if event::poll(timeout)?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            match handle_key(app, key.code) {
                KeyAction::Quit => return Ok(()),
                KeyAction::Restart => {
                    terminal.clear()?;
                    continue;
                }
                KeyAction::Continue => {}
            }
        }
    }
}

fn handle_key(app: &mut App, code: KeyCode) -> KeyAction {
    match code {
        KeyCode::Char('q') | KeyCode::Esc => return KeyAction::Quit,
        KeyCode::Char('r') => {
            app.restart();
            return KeyAction::Restart;
        }
        KeyCode::Right | KeyCode::Enter | KeyCode::Char('n') => {
            if app.anim.is_none() && app.is_at_end() {
                app.next_round();
            } else if app.anim.is_none() && !app.is_at_end() {
                app.start_animation();
            }
        }
        KeyCode::Char('.') => {
            if app.anim.is_none() && !app.is_at_end() {
                app.current += 1;
            }
        }
        KeyCode::Left | KeyCode::Backspace | KeyCode::Char('p') => {
            if app.anim.is_none() && !app.is_at_start() {
                app.current -= 1;
            }
        }
        KeyCode::Char(' ') => {
            if app.is_at_end() && app.mode != SolveMode::Hybrid {
                app.next_round();
            } else {
                app.auto_play = !app.auto_play;
                if app.auto_play && app.anim.is_none() && !app.is_at_end() {
                    app.start_animation();
                }
            }
        }
        KeyCode::Home => {
            app.anim = None;
            app.auto_play = false;
            app.current = 0;
        }
        KeyCode::End => {
            app.anim = None;
            app.auto_play = false;
            app.current = app.total_steps();
        }
        _ => {}
    }
    KeyAction::Continue
}

// ── Drawing ────────────────────────────────────────────────────

fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title
            Constraint::Min(17),   // content
            Constraint::Length(3), // nav
        ])
        .split(area);

    draw_title(f, chunks[0], app);
    draw_content(f, chunks[1], app);
    draw_nav(f, chunks[2], app);
}

fn draw_title(f: &mut Frame, area: Rect, app: &App) {
    let auto = if app.auto_play { " ⏵AUTO" } else { "" };
    let (icon, round) = match app.mode {
        SolveMode::BruteForce => (BEAR, 1),
        SolveMode::Ai => (RABBIT, 2),
        SolveMode::Hybrid => (FOX, 3),
    };
    let result = app.result();
    let status = if result.success {
        "✓"
    } else if result.states.last().is_some_and(|s| s.dead) {
        "☠"
    } else {
        "…"
    };
    let fog = app.current_fog();
    let discovered = fog.discovered_keys.iter().filter(|&&d| d).count()
        + fog.discovered_boxes.iter().filter(|&&d| d).count()
        + fog.discovered_levers.iter().filter(|&&d| d).count();
    let total_discoverable = app.game.keys.len() + app.game.boxes.len() + app.game.levers.len() + 2; // +2 for goal and bridge
    let goal_disc = if fog.goal_pos.is_some() { 1 } else { 0 };
    let bridge_disc = if fog.bridge_seen { 1 } else { 0 };
    let total_found = discovered + goal_disc + bridge_disc;

    let line = Line::from(vec![
        Span::styled(
            format!(" {icon} Round {round} "),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(
                " {}/{} steps · {}ms · {status}{auto} · {total_found}/{total_discoverable} discovered ",
                app.current,
                app.total_steps(),
                result.solve_time_ms,
            ),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled(
            " ← → Space · R New · Q Quit ",
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    let para = Paragraph::new(line).block(Block::default().borders(Borders::ALL));
    f.render_widget(para, area);
}

fn draw_content(f: &mut Frame, area: Rect, app: &App) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(49), Constraint::Min(30)])
        .split(area);

    draw_map(f, cols[0], app);

    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(11), Constraint::Min(6)])
        .split(cols[1]);

    draw_discovery(f, right[0], app);
    draw_status(f, right[1], app);
}

// ── Map ────────────────────────────────────────────────────────

fn draw_map(f: &mut Frame, area: Rect, app: &App) {
    let state = app.current_state();
    let fog = app.current_fog();
    let (bear_r, bear_c) = app.bear_pos();
    let (boss_r, boss_c) = (state.boss_r, state.boss_c);

    let player_emoji = match app.mode {
        SolveMode::BruteForce => BEAR,
        SolveMode::Ai => RABBIT,
        SolveMode::Hybrid => FOX,
    };

    let mut lines = Vec::new();
    for r in 0..app.game.rows() {
        let mut spans = Vec::new();
        for c in 0..app.game.cols() {
            let is_player = bear_r == r && bear_c == c;
            let is_boss_visible =
                boss_r == r && boss_c == c && state.boss_alive && fog.visible.contains(&(r, c));

            let (emoji, style) = if is_player {
                (player_emoji.to_string(), Style::default())
            } else if is_boss_visible {
                (BOSS_LIVE.to_string(), Style::default().fg(Color::Red))
            } else {
                cell_render_fog(&app.game, state, fog, r, c)
            };

            spans.push(Span::styled(format!("{emoji} "), style));
        }
        lines.push(Line::from(spans));
    }

    let para = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" 🗺 Map "));
    f.render_widget(para, area);
}

/// Render a cell with fog-of-war styling.
fn cell_render_fog(
    game: &StrategicGame,
    state: &StrategicState,
    fog: &FogState,
    r: usize,
    c: usize,
) -> (String, Style) {
    let pos = (r, c);

    // Unseen tile
    if !fog.seen.contains(&pos) {
        return (FOG.to_string(), Style::default().fg(Color::DarkGray));
    }

    let is_visible = fog.visible.contains(&pos);

    // Boss dead (visible only)
    if state.boss_r == r && state.boss_c == c && !state.boss_alive && is_visible {
        return (BOSS_DEAD.to_string(), Style::default().fg(Color::DarkGray));
    }

    // Keys (discovered and still present)
    for (i, &(kr, kc)) in game.keys.iter().enumerate() {
        if (kr, kc) == pos && fog.discovered_keys[i] {
            let held = (state.keys_held & (1 << i)) != 0;
            let used = (state.keys_used & (1 << i)) != 0;
            if !held && !used {
                let style = dim_style(Style::default().fg(Color::Yellow), is_visible);
                return (KEY_EMOJI.to_string(), style);
            }
        }
    }

    // Boxes (discovered)
    for (j, &(br, bc)) in game.boxes.iter().enumerate() {
        if (br, bc) == pos && fog.discovered_boxes[j] {
            let opened = (state.boxes_opened & (1 << j)) != 0;
            let (emoji, color) = if opened {
                (BOX_OPEN, Color::Green)
            } else {
                (BOX_CLOSED, Color::Magenta)
            };
            return (
                emoji.to_string(),
                dim_style(Style::default().fg(color), is_visible),
            );
        }
    }

    // Levers (discovered)
    for (l, &(lr, lc)) in game.levers.iter().enumerate() {
        if (lr, lc) == pos && fog.discovered_levers[l] {
            let is_on = (state.lever_state & (1 << l)) != 0;
            let emoji = if is_on { LEVER_ON } else { LEVER };
            let color = if is_on { Color::Yellow } else { Color::Cyan };
            return (
                emoji.to_string(),
                dim_style(Style::default().fg(color), is_visible),
            );
        }
    }

    // Goal (discovered)
    if fog.goal_pos == Some(pos) {
        let all_boxes = (1 << game.boxes.len()) - 1;
        let emoji = if state.boxes_opened == all_boxes && state.bridge_open {
            GOAL_WIN
        } else {
            GOAL
        };
        return (
            emoji.to_string(),
            dim_style(Style::default().fg(Color::Yellow), is_visible),
        );
    }

    // Traps (discovered)
    if fog.discovered_traps.contains(&pos) {
        return (
            TRAP.to_string(),
            dim_style(Style::default().fg(Color::Red), is_visible),
        );
    }

    // Bridge (seen)
    if game.bridge.contains(&pos) {
        let emoji = if state.bridge_open {
            BRIDGE_OPEN
        } else {
            BRIDGE_CLOSED
        };
        return (
            emoji.to_string(),
            dim_style(Style::default().fg(Color::Blue), is_visible),
        );
    }

    // Wall
    if game.grid[r][c] == '#' {
        return (
            WALL.to_string(),
            dim_style(Style::default().fg(Color::White), is_visible),
        );
    }

    // Floor
    (
        FLOOR.to_string(),
        dim_style(Style::default().fg(Color::DarkGray), is_visible),
    )
}

fn dim_style(base: Style, is_visible: bool) -> Style {
    if is_visible {
        base
    } else {
        base.add_modifier(Modifier::DIM)
    }
}

// ── Discovery Panel ────────────────────────────────────────────

fn draw_discovery(f: &mut Frame, area: Rect, app: &App) {
    let state = app.current_state();
    let fog = app.current_fog();
    let game = &app.game;

    let keys_found = fog.discovered_keys.iter().filter(|&&d| d).count();
    let boxes_found = fog.discovered_boxes.iter().filter(|&&d| d).count();
    let levers_found = fog.discovered_levers.iter().filter(|&&d| d).count();
    let goal_found = if fog.goal_pos.is_some() { CHECK } else { "✗" };
    let bridge_found = if fog.bridge_seen { CHECK } else { "✗" };
    let traps_found = fog.discovered_traps.len();

    // Map exploration percentage
    let total_floors = game
        .grid
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&c| c != '#')
        .count();
    let seen_floors = fog
        .seen
        .iter()
        .filter(|&&(r, c)| game.grid[r][c] != '#')
        .count();
    let explored_pct = if total_floors > 0 {
        seen_floors * 100 / total_floors
    } else {
        100
    };

    // Frontier count
    let frontiers = fog.frontier_tiles(&game.grid);

    // Vision radius indicator
    let vision_indicator = "●".repeat(VISION_RADIUS);

    let lines = vec![
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(KEY_EMOJI, Style::default().fg(Color::Yellow)),
            Span::styled(
                format!(" Keys:   {keys_found}/{}", game.keys.len()),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                if state.keys_held != 0 {
                    format!(" ({} held)", state.keys_held.count_ones())
                } else {
                    String::new()
                },
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(BOX_CLOSED, Style::default().fg(Color::Magenta)),
            Span::styled(
                format!(" Boxes:  {boxes_found}/{}", game.boxes.len()),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                if state.boxes_opened != 0 {
                    format!(" ({} opened)", state.boxes_opened.count_ones())
                } else {
                    String::new()
                },
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(LEVER, Style::default().fg(Color::Cyan)),
            Span::styled(
                format!(" Levers: {levers_found}/{}", game.levers.len()),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                format!(" state=0b{:03b}", state.lever_state),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(GOAL, Style::default().fg(Color::Yellow)),
            Span::styled(
                format!(" Goal:   {goal_found}"),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(
                if state.bridge_open {
                    BRIDGE_OPEN
                } else {
                    BRIDGE_CLOSED
                },
                Style::default().fg(Color::Blue),
            ),
            Span::styled(
                format!(" Bridge: {bridge_found}"),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                if state.bridge_open { " (OPEN)" } else { "" },
                Style::default().fg(Color::Green),
            ),
        ]),
        Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(TRAP, Style::default().fg(Color::Red)),
            Span::styled(
                format!(" Traps:  {traps_found} found"),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Map:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{explored_pct}% explored"),
                Style::default().fg(if explored_pct >= 100 {
                    Color::Green
                } else {
                    Color::White
                }),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Vision: ", Style::default().fg(Color::DarkGray)),
            Span::styled(vision_indicator, Style::default().fg(Color::Cyan)),
            Span::styled(
                format!(" (radius {VISION_RADIUS})"),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Frontiers: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", frontiers.len()),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![Span::styled(
            format!(
                "  Config: seed={} keys={:?} target=0b{:03b}",
                app.seed, game.key_mapping, game.target_lever_mask
            ),
            Style::default().fg(Color::DarkGray),
        )]),
    ];

    let para =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Discovery "));
    f.render_widget(para, area);
}

// ── Status Panel ───────────────────────────────────────────────

fn draw_status(f: &mut Frame, area: Rect, app: &App) {
    let state = app.current_state();
    let phase = app.phase();

    let (phase_label, phase_color) = match phase {
        Phase::Exploring => ("Exploring", Color::Cyan),
        Phase::Done => {
            if state.dead {
                ("☠ Dead", Color::Red)
            } else if app.result().success {
                ("Done 🎉", Color::Green)
            } else {
                ("Stuck ⚠", Color::Yellow)
            }
        }
    };

    let boss_status = if state.dead {
        format!("{SKULL} DEAD")
    } else if state.boss_alive {
        if app
            .current_fog()
            .visible
            .contains(&(state.boss_r, state.boss_c))
        {
            format!("{BOSS_LIVE} ({},{})", state.boss_r, state.boss_c)
        } else {
            format!("{BOSS_LIVE} ???")
        }
    } else {
        format!("{BOSS_DEAD} Killed!")
    };

    let bridge_status = if state.bridge_open {
        format!("{BRIDGE_OPEN} Open")
    } else {
        format!("{BRIDGE_CLOSED} Closed")
    };

    let key_strs: Vec<String> = (0..app.game.keys.len())
        .map(|i| {
            if (state.keys_held & (1 << i)) != 0 {
                format!("{KEY_EMOJI}{i}")
            } else if (state.keys_used & (1 << i)) != 0 {
                format!("{CHECK}{i}")
            } else {
                format!("·{i}")
            }
        })
        .collect();

    let lever_strs: Vec<String> = (0..app.game.levers.len())
        .map(|l| {
            if (state.lever_state & (1 << l)) != 0 {
                format!("{LEVER_ON}{l}")
            } else {
                format!("·{l}")
            }
        })
        .collect();

    let target_str: String = (0..app.game.levers.len())
        .map(|l| {
            if (app.game.target_lever_mask & (1 << l)) != 0 {
                "ON "
            } else {
                "OFF"
            }
        })
        .collect::<Vec<_>>()
        .join(" ");

    let lines = vec![
        Line::from(vec![
            Span::styled("  Phase:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(phase_label, Style::default().fg(phase_color)),
        ]),
        Line::from(vec![
            Span::styled("  Step:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}/{}", app.current, app.total_steps()),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Pos:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("({},{})", state.r, state.c),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Boss:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                boss_status,
                Style::default().fg(if state.boss_alive {
                    Color::Red
                } else {
                    Color::Green
                }),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Bridge: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                bridge_status,
                Style::default().fg(if state.bridge_open {
                    Color::Green
                } else {
                    Color::Blue
                }),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Keys:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(key_strs.join(" "), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("  Levers: ", Style::default().fg(Color::DarkGray)),
            Span::styled(lever_strs.join(" "), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("  Target: ", Style::default().fg(Color::DarkGray)),
            Span::styled(target_str, Style::default().fg(Color::Cyan)),
        ]),
    ];

    let para =
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" Status "));
    f.render_widget(para, area);
}

// ── Navigation Bar ─────────────────────────────────────────────

fn draw_nav(f: &mut Frame, area: Rect, app: &App) {
    let cur = app.current;
    let total = app.total_steps();

    let back_style = if app.is_at_start() {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD)
    };
    let next_style = if app.is_at_end() && app.mode == SolveMode::Hybrid {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD)
    };

    let center = if let Some(ref anim) = app.anim {
        let name = action_name(anim.action);
        format!("⟳ {name}...")
    } else if total == 0 {
        "No steps".into()
    } else if app.is_at_start() {
        format!("{ARROW} Start {ARROW}")
    } else if app.is_at_end() {
        match app.mode {
            SolveMode::BruteForce => {
                let state = app.current_state();
                let tag = if state.dead {
                    format!("{SKULL} Failed")
                } else if app.bf.success {
                    "🎉 Done".into()
                } else {
                    "⚠ Stuck".into()
                };
                format!("{tag} · → AI round {RABBIT}")
            }
            SolveMode::Ai => {
                let bf_steps = app.bf.steps;
                let ai_steps = app.ai.steps;
                let bf_time = app.bf.solve_time_ms;
                let ai_time = app.ai.solve_time_ms;

                let step_cmp = compare_steps(ai_steps, bf_steps);

                let ai_tag = if app.ai.success { "✓" } else { "✗" };
                let bf_tag = if app.bf.success { "✓" } else { "✗" };

                format!(
                    "{RABBIT} {ai_steps}{ai_tag} vs {BEAR} {bf_steps}{bf_tag} {step_cmp} · \
                     {ai_time}ms vs {bf_time}ms · → Hybrid {FOX}"
                )
            }
            SolveMode::Hybrid => {
                let bf_steps = app.bf.steps;
                let ai_steps = app.ai.steps;
                let hy_steps = app.hybrid.steps;
                let hy_time = app.hybrid.solve_time_ms;
                let bf_time = app.bf.solve_time_ms;

                let step_vs_bf = compare_steps(hy_steps, bf_steps);

                let vs_ai = if hy_steps < ai_steps {
                    format!(" · 🏆 <{RABBIT}")
                } else if hy_steps > ai_steps {
                    format!(
                        " · +{}% vs {RABBIT}",
                        (hy_steps - ai_steps) * 100 / ai_steps.max(1)
                    )
                } else {
                    String::new()
                };

                let hy_tag = if app.hybrid.success { "✓" } else { "✗" };
                let bf_tag = if app.bf.success { "✓" } else { "✗" };

                format!(
                    "{FOX} {hy_steps}{hy_tag} vs {BEAR} {bf_steps}{bf_tag} {step_vs_bf}{vs_ai} · \
                     {hy_time}ms vs {bf_time}ms"
                )
            }
        }
    } else {
        format!("Step {cur}/{total}")
    };

    let auto_str = if app.auto_play { " ⏵" } else { "" };

    let line = Line::from(vec![
        Span::styled(" ◀ Back ", back_style),
        Span::styled(
            format!("   {center}{auto_str}   "),
            Style::default().fg(Color::Yellow),
        ),
        Span::styled(" Next ▶ ", next_style),
    ]);

    let para = Paragraph::new(line).block(Block::default().borders(Borders::ALL));
    f.render_widget(para, area);
}

fn compare_steps(a: usize, b: usize) -> String {
    if b == 0 {
        return format!("({a} vs {b})");
    }
    let pct = if a < b {
        format!("⚡{}%↓", (b - a) * 100 / b)
    } else if a > b {
        format!("+{}%", (a - b) * 100 / b)
    } else {
        "same".into()
    };
    format!("({pct})")
}
