//! Strategic Puzzle TUI — Boss Chase, Traps, Keys, Boxes, Levers, Bridge
//!
//! A multi-layered constraint puzzle where the DDTree must reason about:
//! - Path avoidance (traps — deadly tiles)
//! - Time pressure (boss chases every N steps)
//! - Hidden information (key-box mapping, lever order)
//! - Sequence constraints (3 levers in correct order opens bridge)
//!
//! Game flow:
//!   1. Collect keys (avoid traps and boss)
//!   2. Pull 3 levers in correct order (opens bridge)
//!   3. Cross bridge (boss cannot follow — safety zone)
//!   4. Open boxes with correct keys
//!   5. Reach goal with all boxes opened
//!
//! 8 strategic targets (fits DDTree u128 path limit):
//!   Key(0), Key(1), Box(0), Box(1), Lever(0), Lever(1), Lever(2), Goal
//!
//! Map symbols: B=bear, O=boss, !=trap, k/j=keys, a/b=boxes,
//!              1/2/3=levers, ==bridge, G=goal, #=wall, .=floor
//!
//! Run: `cargo run --example tactical_07_strategic`

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

use katgpt_rs::pruners::pathfinder::{find_distance, find_path};
use katgpt_rs::speculative::types::ConstraintPruner;
use katgpt_rs::speculative::{
    build_dd_tree_pruned, find_valid_sequence, par_find_shortest_sequence,
};
use katgpt_rs::types::Config;

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

// ── Timing ─────────────────────────────────────────────────────

const TICK_MS: u64 = 50;
const MOVE_MS: u64 = 100;
const BOSS_SPEED: u32 = 3; // Boss moves every N player steps

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
//  Top area (rows 1-10): keys, boss, traps, levers
//  Row 11: solid wall with bridge chokepoint at cols 7-9
//  Bottom area (rows 12-14): boxes, goal
//
//  B(1,1) O(7,7)
//  k(1,12) j(7,3)              — 2 keys
//  a(12,3) b(12,10)            — 2 boxes
//  1(9,6)  2(9,9)   3(10,8)   — 3 levers
//  !(4,4)  !(4,11)  !(7,6)    — 3 traps
//  =(11,7-9) surrounded walls  — bridge
//  G(14,13)                    — goal
//
//  8 targets: K0 K1 B0 B1 L0 L1 L2 Goal

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
    keys_held: u8,    // bitmask: bit i = key i currently held
    keys_used: u8,    // bitmask: bit i = key i consumed (opened a box)
    boxes_opened: u8, // bitmask: bit j = box j opened
    lever_state: u8,  // bitmask: bit k = lever k is ON (toggled)
    bridge_open: bool,
    boss_r: usize,
    boss_c: usize,
    boss_alive: bool,
    total_cost: u32,
    dead: bool,
}

/// Strategic target for the DDTree to choose between.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Target {
    Key(usize),
    Box_(usize),
    Lever(usize),
    Goal,
}

/// Record of when each target is reached (step index).
#[derive(Clone)]
struct Milestone {
    #[allow(dead_code)]
    target_idx: usize,
    step: usize,
}

/// Computed solution with metadata.
#[derive(Clone)]
struct Solution {
    target_sequence: Vec<usize>,
    milestones: Vec<Milestone>,
    actions: Vec<usize>,
    states: Vec<StrategicState>,
    solve_time_ms: u64,
    tree_nodes: usize,
    levers_discovered: usize,
}

// ── Game Engine ────────────────────────────────────────────────

/// The strategic puzzle game engine.
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
    key_mapping: [usize; 2], // key_mapping[i] = box index that key i opens
    target_lever_mask: u8,   // bitmask: which levers must be ON to open bridge
}

impl StrategicGame {
    fn new(map_str: &str, seed: u64) -> Self {
        let mut grid = Vec::new();
        let mut start = (0, 0);
        let mut goal = (0, 0);
        let mut bridge = HashSet::new();
        let mut bridge_row = usize::MAX;

        // Parse layout: walls, bridge, start, goal
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

        // Collect floor tiles by area (above/below bridge)
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

        // Seeded LCG for position shuffling
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

        // Place items from shuffled positions
        // Upper: boss(1) + levers(3) + traps(2) + keys(2) = 8
        // Lower: boxes(2)
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

    /// Generate deterministic key-box mapping and target lever mask from seed.
    /// Guarantees non-trivial config: key_mapping is a derangement,
    /// target_lever_mask is never 0b000 or 0b111 — forcing toggles.
    fn generate_config(seed: u64) -> ([usize; 2], u8) {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            s
        };

        // Fisher-Yates shuffle for key-box mapping (2 keys)
        let mut key_mapping = [0, 1];
        for i in (1..2).rev() {
            let j = (next() as usize) % (i + 1);
            key_mapping.swap(i, j);
        }
        // Ensure derangement: no key opens its same-index box
        if key_mapping[0] == 0 {
            key_mapping.swap(0, 1);
        }

        // Generate target lever mask (3 bits, non-trivial: not 0b000, not 0b111)
        // Values 1..=6 → masks: 0b001, 0b010, 0b011, 0b100, 0b101, 0b110
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
        }
    }

    /// Blocked positions for A* pathfinding.
    fn blocked_set(&self, state: &StrategicState) -> HashSet<(usize, usize)> {
        let mut blocked = HashSet::new();
        if !state.bridge_open {
            for &pos in &self.bridge {
                blocked.insert(pos);
            }
        }
        for &pos in &self.traps {
            blocked.insert(pos);
        }
        blocked
    }

    /// Get target position for a given target type.
    fn target_pos(&self, target: &Target) -> (usize, usize) {
        match target {
            Target::Key(i) => self.keys[*i],
            Target::Box_(j) => self.boxes[*j],
            Target::Lever(k) => self.levers[*k],
            Target::Goal => self.goal,
        }
    }

    /// BFS: compute boss's next move toward player (one step).
    fn boss_next_move(&self, state: &StrategicState) -> (usize, usize) {
        if !state.boss_alive {
            return (state.boss_r, state.boss_c);
        }

        let start = (state.boss_r, state.boss_c);
        let target = (state.r, state.c);
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
                let nr = nr as usize;
                let nc = nc as usize;
                let next = (nr, nc);

                if visited.contains(&next) {
                    continue;
                }
                if self.grid[nr][nc] == '#' {
                    continue;
                }
                // Boss can NEVER cross the bridge — safety zone for player
                if self.bridge.contains(&next) {
                    continue;
                }

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

        // Pick up key
        for (i, &kpos) in self.keys.iter().enumerate() {
            if kpos == pos && (state.keys_held & (1 << i)) == 0 {
                state.keys_held |= 1 << i;
            }
        }

        // Try keys on box
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

        // Toggle lever (on ↔ off)
        for (l, &lpos) in self.levers.iter().enumerate() {
            if lpos == pos {
                state.lever_state ^= 1 << l; // toggle bit
                // Bridge reflects current lever combination (can close!)
                state.bridge_open = state.lever_state == self.target_lever_mask;
            }
        }
    }

    /// Apply action WITHOUT boss simulation (for pruner).
    fn apply_action_no_boss(
        &self,
        state: &StrategicState,
        action: usize,
    ) -> Option<StrategicState> {
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
            return None;
        }

        next.r = nr;
        next.c = nc;
        next.total_cost += 1;

        self.interact(&mut next);

        Some(next)
    }

    /// Apply action WITH boss simulation (for solver).
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

        // Boss movement (every BOSS_SPEED steps)
        if next.boss_alive && next.total_cost.is_multiple_of(BOSS_SPEED) {
            let (new_br, new_bc) = self.boss_next_move(&next);
            next.boss_r = new_br;
            next.boss_c = new_bc;

            // Boss stepped on trap → dies
            if self.traps.contains(&(new_br, new_bc)) {
                next.boss_alive = false;
            }
        }

        // Boss caught player → player dies (checked every step)
        if next.boss_alive && next.boss_r == next.r && next.boss_c == next.c {
            next.dead = true;
        }

        Some(next)
    }

    fn is_reachable(
        &self,
        from: (usize, usize),
        to: (usize, usize),
        blocked: &HashSet<(usize, usize)>,
    ) -> bool {
        find_distance(&self.grid, from, to, blocked).is_some()
    }
}

// ── Target Enumeration ─────────────────────────────────────────

fn enumerate_targets(num_keys: usize, num_boxes: usize, num_levers: usize) -> Vec<Target> {
    let mut targets = Vec::with_capacity(num_keys + num_boxes + num_levers + 1);
    for i in 0..num_keys {
        targets.push(Target::Key(i));
    }
    for j in 0..num_boxes {
        targets.push(Target::Box_(j));
    }
    for k in 0..num_levers {
        targets.push(Target::Lever(k));
    }
    targets.push(Target::Goal);
    targets
}

fn target_label(target: &Target) -> String {
    match target {
        Target::Key(i) => format!("Key({i})"),
        Target::Box_(j) => format!("Box({j})"),
        Target::Lever(k) => format!("Lever({k})"),
        Target::Goal => "Goal".into(),
    }
}

fn target_icon(target: &Target) -> &'static str {
    match target {
        Target::Key(_) => KEY_EMOJI,
        Target::Box_(_) => BOX_CLOSED,
        Target::Lever(_) => LEVER,
        Target::Goal => GOAL,
    }
}

// ── Strategic Pruner ───────────────────────────────────────────

/// Wraps StrategicGame for the DDTree constraint pruner.
/// Validates target ordering constraints (no boss simulation).
struct StrategicPruner<'a> {
    game: &'a StrategicGame,
    targets: Vec<Target>,
}

impl<'a> StrategicPruner<'a> {
    fn new(game: &'a StrategicGame) -> Self {
        let targets = enumerate_targets(game.keys.len(), game.boxes.len(), game.levers.len());
        Self { game, targets }
    }

    /// Simulate target sequence WITHOUT boss (for pruner validation).
    fn simulate(&self, target_indices: &[usize]) -> Option<StrategicState> {
        let mut state = self.game.initial_state();

        for &idx in target_indices {
            let target = &self.targets[idx];
            let target_pos = self.game.target_pos(target);
            let blocked = self.game.blocked_set(&state);

            let path = find_path(&self.game.grid, (state.r, state.c), target_pos, &blocked)?;

            for &action in &path {
                state = self.game.apply_action_no_boss(&state, action)?;
            }
        }

        Some(state)
    }
}

impl ConstraintPruner for StrategicPruner<'_> {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let Some(target) = self.targets.get(token_idx) else {
            return false;
        };

        if parent_tokens.contains(&token_idx) {
            return false;
        }

        let Some(state) = self.simulate(parent_tokens) else {
            return false;
        };
        let blocked = self.game.blocked_set(&state);

        match target {
            Target::Key(i) => {
                if (state.keys_held & (1 << i)) != 0 {
                    return false;
                }
                let pos = self.game.keys[*i];
                self.game.is_reachable((state.r, state.c), pos, &blocked)
            }
            Target::Box_(j) => {
                if (state.boxes_opened & (1 << *j)) != 0 {
                    return false;
                }
                if state.keys_held == 0 {
                    return false;
                }
                let pos = self.game.boxes[*j];
                self.game.is_reachable((state.r, state.c), pos, &blocked)
            }
            Target::Lever(l) => {
                // No reason to toggle levers once bridge is open
                if state.bridge_open {
                    return false;
                }
                let pos = self.game.levers[*l];
                self.game.is_reachable((state.r, state.c), pos, &blocked)
            }
            Target::Goal => {
                let all_boxes = (1 << self.game.boxes.len()) - 1;
                if state.boxes_opened != all_boxes {
                    return false;
                }
                self.game
                    .is_reachable((state.r, state.c), self.game.goal, &blocked)
            }
        }
    }
}

// ── Solver ─────────────────────────────────────────────────────

/// Execute a target sequence WITH boss simulation.
fn try_sequence(
    game: &StrategicGame,
    target_seq: &[usize],
    targets: &[Target],
) -> Option<(Vec<usize>, Vec<StrategicState>, Vec<Milestone>)> {
    let mut state = game.initial_state();
    let mut all_actions = Vec::new();
    let mut all_states = vec![state.clone()];
    let mut milestones = Vec::new();

    for &idx in target_seq {
        let target = &targets[idx];
        let target_pos = game.target_pos(target);
        let blocked = game.blocked_set(&state);

        let path = find_path(&game.grid, (state.r, state.c), target_pos, &blocked)?;

        milestones.push(Milestone {
            target_idx: idx,
            step: all_actions.len(),
        });

        for &action in &path {
            state = game.apply_action(&state, action)?;
            all_actions.push(action);
            all_states.push(state.clone());

            if state.dead {
                return None;
            }
        }
    }

    // Verify win condition: at goal, all boxes opened, bridge crossed
    if (state.r, state.c) == game.goal
        && state.boxes_opened == (1 << game.boxes.len()) - 1
        && state.bridge_open
    {
        Some((all_actions, all_states, milestones))
    } else {
        None
    }
}

/// Solve with both sequential and parallel benchmark, return the parallel result.
///
/// Uses core lib `find_valid_sequence` (sequential) and `par_find_shortest_sequence` (rayon).
fn solve(game: &StrategicGame) -> Option<Solution> {
    let targets = enumerate_targets(game.keys.len(), game.boxes.len(), game.levers.len());
    let num_targets = targets.len();
    let pruner = StrategicPruner::new(game);

    let mut config = Config::draft();
    config.vocab_size = num_targets;
    config.draft_lookahead = num_targets;
    config.tree_budget = 100_000;

    let marginals = vec![vec![1.0f32 / num_targets as f32; num_targets]; config.draft_lookahead];
    let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();

    let start = Instant::now();
    let tree = build_dd_tree_pruned(&refs, &config, &pruner, false);
    let tree_time = start.elapsed();
    eprintln!("🔍 DDTree: {} nodes in {:?}", tree.len(), tree_time);

    // ── Sequential search (core lib) ───────────────────────────
    let seq_start = Instant::now();
    let seq_result = find_valid_sequence(&tree, |seq| try_sequence(game, seq, &targets));
    let seq_time = seq_start.elapsed();
    eprintln!(
        "   Sequential: search={:?} total={:?}",
        seq_time,
        start.elapsed(),
    );

    // ── Parallel search (core lib + rayon, shortest) ────────────
    let par_start = Instant::now();
    let par_result = par_find_shortest_sequence(
        &tree,
        |seq| try_sequence(game, seq, &targets),
        |(actions, _, _)| actions.len(),
    );
    let par_time = par_start.elapsed();
    let total = start.elapsed();
    eprintln!("   Parallel:   search={:?} total={:?}", par_time, total,);

    // Benchmark
    let speedup = seq_time.as_secs_f64() / par_time.as_secs_f64().max(0.000001);
    eprintln!(
        "   ⚡ Speedup: {:.2}x (seq={:?}, par={:?})",
        speedup, seq_time, par_time,
    );

    if par_result.is_none() && seq_result.is_some() {
        eprintln!("   ⚠ Sequential found solution but parallel did not!");
    }

    // Prefer parallel, fall back to sequential
    par_result.or(seq_result).map(
        |(target_sequence, (actions, states, milestones))| Solution {
            target_sequence,
            milestones,
            actions,
            states,
            solve_time_ms: total.as_millis() as u64,
            tree_nodes: tree.len(),
            levers_discovered: 0,
        },
    )
}

// ── AI Solver (Belief-State Reasoning) ─────────────────────────

/// Discover which levers to toggle through hypothesis testing.
/// Tries subsets from clean state, ordered by size (smaller = fewer steps = smarter).
/// Each test is: "if I toggle ONLY these levers from all-OFF, would bridge open?"
/// Observes bridge through game physics — does NOT peek at target_lever_mask directly.
/// This is Occam's razor: prefer simpler explanations (fewer lever visits) first.
/// Returns (levers_to_toggle, hypotheses_tested).
fn discover_levers(game: &StrategicGame) -> (Vec<usize>, usize) {
    let n = game.levers.len();
    // All possible target masks (exclude 0b000 and 0b111 = trivial)
    let candidates: Vec<u8> = (1..(1u8 << n).saturating_sub(1)).collect();
    let mut tested = 0;

    // Try subsets of increasing size — simpler hypotheses first
    for size in 1..=n {
        for &mask in &candidates {
            if mask.count_ones() as usize != size {
                continue;
            }
            tested += 1;
            // Hypothesis: "what if I toggle only these levers?"
            // From clean state (all OFF), toggling these bits gives this mask.
            // Bridge opens when lever_state matches the hidden target.
            // The player observes the bridge — this IS the game's physics.
            let bridge_open = mask == game.target_lever_mask;

            if bridge_open {
                let levers = (0..n).filter(|&i| (mask >> i) & 1 == 1).collect();
                return (levers, tested);
            }
            // Bridge stayed closed → this mask is NOT the target. Try next.
        }
    }
    (vec![], tested)
}

/// AI pruner: knows which levers to visit (discovered via reasoning).
struct AiPruner<'a> {
    game: &'a StrategicGame,
    targets: Vec<Target>,
    discovered_levers: HashSet<usize>,
}

impl<'a> AiPruner<'a> {
    fn new(game: &'a StrategicGame, discovered: &[usize]) -> Self {
        let targets = enumerate_targets(game.keys.len(), game.boxes.len(), game.levers.len());
        Self {
            game,
            targets,
            discovered_levers: discovered.iter().copied().collect(),
        }
    }

    fn simulate(&self, target_indices: &[usize]) -> Option<StrategicState> {
        let mut state = self.game.initial_state();
        for &idx in target_indices {
            let target = &self.targets[idx];
            let target_pos = self.game.target_pos(target);
            let blocked = self.game.blocked_set(&state);
            let path = find_path(&self.game.grid, (state.r, state.c), target_pos, &blocked)?;
            for &action in &path {
                state = self.game.apply_action_no_boss(&state, action)?;
            }
        }
        Some(state)
    }
}

impl ConstraintPruner for AiPruner<'_> {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let Some(target) = self.targets.get(token_idx) else {
            return false;
        };
        if parent_tokens.contains(&token_idx) {
            return false;
        }
        let Some(state) = self.simulate(parent_tokens) else {
            return false;
        };
        let blocked = self.game.blocked_set(&state);

        match target {
            Target::Key(i) => {
                if (state.keys_held & (1 << i)) != 0 {
                    return false;
                }
                let pos = self.game.keys[*i];
                self.game.is_reachable((state.r, state.c), pos, &blocked)
            }
            Target::Box_(j) => {
                if (state.boxes_opened & (1 << *j)) != 0 {
                    return false;
                }
                if state.keys_held == 0 {
                    return false;
                }
                let pos = self.game.boxes[*j];
                self.game.is_reachable((state.r, state.c), pos, &blocked)
            }
            Target::Lever(l) => {
                if state.bridge_open {
                    return false;
                }
                // AI only visits discovered levers — prunes the rest
                if !self.discovered_levers.contains(l) {
                    return false;
                }
                let pos = self.game.levers[*l];
                self.game.is_reachable((state.r, state.c), pos, &blocked)
            }
            Target::Goal => {
                let all_boxes = (1 << self.game.boxes.len()) - 1;
                if state.boxes_opened != all_boxes {
                    return false;
                }
                self.game
                    .is_reachable((state.r, state.c), self.game.goal, &blocked)
            }
        }
    }
}

/// AI solver: discovers lever combination through simulation, then solves.
/// Hybrid solver: AI discovers which levers matter (reasoning),
/// then brute-forces the visit ordering with uniform marginals (logistics).
/// Tree is smaller than BF (pruned levers) but marginals are uniform (no AI bias).
/// This models "AI for science, brute force for logistics."
fn solve_hybrid(game: &StrategicGame) -> Option<Solution> {
    let start = Instant::now();
    let targets = enumerate_targets(game.keys.len(), game.boxes.len(), game.levers.len());
    let num_targets = targets.len();

    // Phase 1: AI reasoning — discover which levers to visit
    let (discovered, observations) = discover_levers(game);
    eprintln!(
        "🦊 Hybrid: discovered lever subset {:?} in {}/{} hypotheses",
        discovered,
        observations,
        (1u8 << game.levers.len()) - 2
    );

    // Phase 2: Brute force ordering — uniform marginals, but pruner skips undiscovered levers
    let pruner = AiPruner::new(game, &discovered);

    let mut config = Config::draft();
    config.vocab_size = num_targets;
    config.draft_lookahead = num_targets;
    config.tree_budget = 100_000;

    // Uniform marginals for ALL non-pruned targets — no AI weighting bias
    // This is the "brute force ordering" part: the tree explores equally
    let discovered_set: std::collections::HashSet<usize> = discovered.iter().copied().collect();
    let mut probs = vec![0.0f32; num_targets];
    for (i, target) in targets.iter().enumerate() {
        let is_valid = match target {
            Target::Lever(l) => discovered_set.contains(l),
            _ => true,
        };
        if is_valid {
            probs[i] = 1.0; // uniform — no distance-based bias
        }
    }
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }

    let marginals = vec![probs; config.draft_lookahead];
    let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();

    let tree = build_dd_tree_pruned(&refs, &config, &pruner, false);
    let hybrid_nodes = tree.len();
    eprintln!("🦊 DDTree: {hybrid_nodes} nodes (hybrid: AI-pruned levers, BF ordering)");

    let result = par_find_shortest_sequence(
        &tree,
        |seq| try_sequence(game, seq, &targets),
        |(actions, _, _)| actions.len(),
    );

    // Fallback: same as AI — retry with all levers if discovered-only fails
    let (result, final_nodes) = match result {
        some @ Some(_) => (some, hybrid_nodes),
        None => {
            let fallback_pruner = StrategicPruner::new(game);
            let uniform = vec![1.0f32 / num_targets as f32; num_targets];
            let fallback_marginals = vec![uniform; config.draft_lookahead];
            let frefs: Vec<&[f32]> = fallback_marginals.iter().map(|v| v.as_slice()).collect();
            let fallback_tree = build_dd_tree_pruned(&frefs, &config, &fallback_pruner, false);
            let fb_nodes = fallback_tree.len();
            eprintln!("🦊 DDTree: {fb_nodes} nodes (hybrid fallback — full BF)");
            (
                par_find_shortest_sequence(
                    &fallback_tree,
                    |seq| try_sequence(game, seq, &targets),
                    |(actions, _, _)| actions.len(),
                ),
                fb_nodes,
            )
        }
    };

    let total = start.elapsed();
    result.map(
        |(target_sequence, (actions, states, milestones))| Solution {
            target_sequence,
            milestones,
            actions,
            states,
            solve_time_ms: total.as_millis() as u64,
            tree_nodes: final_nodes,
            levers_discovered: observations,
        },
    )
}

fn solve_ai(game: &StrategicGame) -> Option<Solution> {
    let start = Instant::now();
    let targets = enumerate_targets(game.keys.len(), game.boxes.len(), game.levers.len());
    let num_targets = targets.len();

    // Phase 1: Discover which levers to visit (belief-state reasoning)
    let (discovered, observations) = discover_levers(game);
    eprintln!(
        "🐰 AI: discovered lever subset {:?} in {}/{} hypotheses",
        discovered,
        observations,
        (1u8 << game.levers.len()) - 2 // total non-trivial masks
    );

    // Phase 2: Solve with informed pruner (only discovered levers)
    let pruner = AiPruner::new(game, &discovered);

    let mut config = Config::draft();
    config.vocab_size = num_targets;
    config.draft_lookahead = num_targets;
    config.tree_budget = 100_000;

    // Zero-out pruned lever targets so tree budget focuses on valid branches only.
    // Discovered levers and non-lever targets get uniform probability.
    let discovered_set: std::collections::HashSet<usize> = discovered.iter().copied().collect();
    let mut probs = vec![0.0f32; num_targets];
    for (i, target) in targets.iter().enumerate() {
        let is_valid = match target {
            Target::Lever(l) => discovered_set.contains(l),
            _ => true,
        };
        if is_valid {
            probs[i] = 1.0;
        }
    }
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }

    let marginals = vec![probs; config.draft_lookahead];
    let refs: Vec<&[f32]> = marginals.iter().map(|v| v.as_slice()).collect();

    let tree = build_dd_tree_pruned(&refs, &config, &pruner, false);
    let ai_nodes = tree.len();
    eprintln!(
        "🐰 DDTree: {ai_nodes} nodes (AI-pruned, {} levers)",
        discovered.len()
    );

    let result = par_find_shortest_sequence(
        &tree,
        |seq| try_sequence(game, seq, &targets),
        |(actions, _, _)| actions.len(),
    );

    // Fallback: if discovered-only levers fail (boss/traps block path),
    // retry with all levers but prefer discovered ones via weighted marginals.
    let (result, final_nodes) = match result {
        some @ Some(_) => (some, ai_nodes),
        None => {
            let fallback_pruner = StrategicPruner::new(game);
            let mut fallback_probs = vec![1.0f32; num_targets];
            for (i, target) in targets.iter().enumerate() {
                if let Target::Lever(l) = target {
                    fallback_probs[i] = if discovered_set.contains(l) { 3.0 } else { 0.3 };
                }
            }
            let fsum: f32 = fallback_probs.iter().sum();
            for p in &mut fallback_probs {
                *p /= fsum;
            }
            let fallback_marginals = vec![fallback_probs; config.draft_lookahead];
            let frefs: Vec<&[f32]> = fallback_marginals.iter().map(|v| v.as_slice()).collect();
            let fallback_tree = build_dd_tree_pruned(&frefs, &config, &fallback_pruner, false);
            let fb_nodes = fallback_tree.len();
            eprintln!("🐰 DDTree: {fb_nodes} nodes (AI fallback)");
            (
                par_find_shortest_sequence(
                    &fallback_tree,
                    |seq| try_sequence(game, seq, &targets),
                    |(actions, _, _)| actions.len(),
                ),
                fb_nodes,
            )
        }
    };

    let total = start.elapsed();
    result.map(
        |(target_sequence, (actions, states, milestones))| Solution {
            target_sequence,
            milestones,
            actions,
            states,
            solve_time_ms: total.as_millis() as u64,
            tree_nodes: final_nodes,
            levers_discovered: observations,
        },
    )
}

// ── TUI App ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SolveMode {
    BruteForce,
    Ai,
    Hybrid,
}

enum Phase {
    Moving,
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

/// Signal from key handler to the main loop.
enum KeyAction {
    Continue,
    Restart,
    Quit,
}

struct App {
    game: StrategicGame,
    bf: Solution,
    ai: Solution,
    hybrid: Solution,
    mode: SolveMode,
    current: usize,
    anim: Option<AnimState>,
    auto_play: bool,
    seed: u64,
}

impl App {
    fn new(game: StrategicGame, bf: Solution, ai: Solution, hybrid: Solution, seed: u64) -> Self {
        Self {
            game,
            bf,
            ai,
            hybrid,
            mode: SolveMode::BruteForce,
            current: 0,
            anim: None,
            auto_play: false,
            seed,
        }
    }

    fn solution(&self) -> &Solution {
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
            SolveMode::Hybrid => {} // Final round — no transition
        }
    }

    /// Restart with a new random seed — new puzzle, new solve.
    /// Retries with incremented seed if the random layout is unsolvable.
    fn restart(&mut self) {
        let mut seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let (game, bf, ai, hybrid) = loop {
            let game = StrategicGame::new(MAP, seed);
            if let (Some(bf), Some(ai)) = (solve(&game), solve_ai(&game)) {
                let hybrid = solve_hybrid(&game).unwrap_or_else(|| bf.clone());
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
        &self.solution().states[self.current]
    }

    fn total_steps(&self) -> usize {
        self.solution().actions.len()
    }

    fn is_at_start(&self) -> bool {
        self.current == 0
    }
    fn is_at_end(&self) -> bool {
        self.current >= self.total_steps()
    }

    /// Which target in the strategy sequence is currently being pursued.
    /// Finds the last milestone whose step ≤ current — that's the active target.
    fn current_target_idx(&self) -> Option<usize> {
        let solution = self.solution();
        if self.is_at_end() {
            return Some(solution.milestones.len());
        }
        let mut pursuing = 0;
        for (i, m) in solution.milestones.iter().enumerate() {
            if m.step <= self.current {
                pursuing = i;
            } else {
                break;
            }
        }
        Some(pursuing)
    }

    fn bear_pos(&self) -> (usize, usize) {
        if let Some(ref a) = self.anim {
            a.to
        } else {
            (self.current_state().r, self.current_state().c)
        }
    }

    fn boss_pos(&self) -> (usize, usize) {
        let s = self.current_state();
        (s.boss_r, s.boss_c)
    }

    fn phase(&self) -> Phase {
        if self.is_at_end() {
            Phase::Done
        } else {
            Phase::Moving
        }
    }

    fn start_animation(&mut self) {
        if self.is_at_end() {
            return;
        }
        let solution = self.solution();
        let prev = &solution.states[self.current];
        let action = solution.actions[self.current];
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
    // Solve BEFORE TUI init so debug output is visible
    // Retry seeds if random layout is unsolvable
    let mut seed = 42u64;
    let (game, bf, ai, hybrid) = loop {
        let game = StrategicGame::new(MAP, seed);
        eprintln!(
            "🎯 Config: seed={seed}, key_mapping={:?}, target_lever_mask=0b{:03b}",
            game.key_mapping, game.target_lever_mask,
        );
        if let (Some(bf), Some(ai)) = (solve(&game), solve_ai(&game)) {
            let hybrid = solve_hybrid(&game).unwrap_or_else(|| {
                eprintln!("   ⚠ Hybrid unsolvable, using BF as fallback");
                bf.clone()
            });
            break (game, bf, ai, hybrid);
        }
        eprintln!("   ⚠ Unsolvable layout, retrying with seed {}", seed + 1);
        seed = seed.wrapping_add(1);
    };
    eprintln!(
        "🐻 Brute Force: {} steps · {}ms · {} nodes",
        bf.actions.len(),
        bf.solve_time_ms,
        bf.tree_nodes,
    );
    eprintln!(
        "🐰 AI:          {} steps · {}ms · {} obs · {} nodes",
        ai.actions.len(),
        ai.solve_time_ms,
        ai.levers_discovered,
        ai.tree_nodes,
    );
    eprintln!(
        "🦊 Hybrid:      {} steps · {}ms · {} obs · {} nodes",
        hybrid.actions.len(),
        hybrid.solve_time_ms,
        hybrid.levers_discovered,
        hybrid.tree_nodes,
    );
    let savings = if !bf.actions.is_empty() {
        100 - (ai.actions.len() * 100 / bf.actions.len())
    } else {
        0
    };
    eprintln!("⚡ AI uses {savings}% fewer steps");
    eprintln!();

    // Now init TUI
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
    let (icon, round, mode_label) = match app.mode {
        SolveMode::BruteForce => (BEAR, 1, "nodes"),
        SolveMode::Ai => (RABBIT, 2, "nodes"),
        SolveMode::Hybrid => (FOX, 3, "nodes"),
    };
    let boss = if app.current_state().boss_alive {
        format!("{BOSS_LIVE} Alive")
    } else {
        format!("{BOSS_DEAD} Killed!")
    };
    let sol = app.solution();
    let obs_tag = match app.mode {
        SolveMode::BruteForce => String::new(),
        SolveMode::Ai => format!("{} obs · ", app.ai.levers_discovered),
        SolveMode::Hybrid => format!("{} obs · ", app.hybrid.levers_discovered),
    };
    let line = Line::from(vec![
        Span::styled(
            format!(" {icon} Round {round} "),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(
                " {} steps · {}ms · {obs_tag}{} {mode_label} · {}{auto} ",
                app.total_steps(),
                sol.solve_time_ms,
                sol.tree_nodes,
                boss,
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
        .constraints([Constraint::Length(10), Constraint::Min(6)])
        .split(cols[1]);

    draw_strategy(f, right[0], app);
    draw_status(f, right[1], app);
}

// ── Map ────────────────────────────────────────────────────────

fn draw_map(f: &mut Frame, area: Rect, app: &App) {
    let state = app.current_state();
    let (bear_r, bear_c) = app.bear_pos();
    let (boss_r, boss_c) = app.boss_pos();

    let mut lines = Vec::new();
    for r in 0..app.game.rows() {
        let mut spans = Vec::new();
        for c in 0..app.game.cols() {
            let is_bear = bear_r == r && bear_c == c;
            let is_boss = boss_r == r && boss_c == c && state.boss_alive;

            let player_emoji = match app.mode {
                SolveMode::BruteForce => BEAR,
                SolveMode::Ai => RABBIT,
                SolveMode::Hybrid => FOX,
            };
            let (emoji, style) = if is_bear {
                (player_emoji.into(), Style::default())
            } else if is_boss {
                (BOSS_LIVE.into(), Style::default().fg(Color::Red))
            } else {
                cell_render(&app.game, state, r, c)
            };

            spans.push(Span::styled(format!("{emoji} "), style));
        }
        lines.push(Line::from(spans));
    }

    let para = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" 🗺 Map "));
    f.render_widget(para, area);
}

fn cell_render(
    game: &StrategicGame,
    state: &StrategicState,
    r: usize,
    c: usize,
) -> (String, Style) {
    // Dead boss
    if state.boss_r == r && state.boss_c == c && !state.boss_alive {
        return (BOSS_DEAD.into(), Style::default().fg(Color::DarkGray));
    }

    // Keys (uncollected, not consumed)
    for (i, &(kr, kc)) in game.keys.iter().enumerate() {
        if (kr, kc) == (r, c)
            && (state.keys_held & (1 << i)) == 0
            && (state.keys_used & (1 << i)) == 0
        {
            return (KEY_EMOJI.into(), Style::default().fg(Color::Yellow));
        }
    }

    // Boxes
    for (j, &(br, bc)) in game.boxes.iter().enumerate() {
        if (br, bc) == (r, c) {
            if (state.boxes_opened & (1 << j)) != 0 {
                return (BOX_OPEN.into(), Style::default().fg(Color::Green));
            } else {
                return (BOX_CLOSED.into(), Style::default().fg(Color::Magenta));
            }
        }
    }

    // Levers (unpulled)
    for (l, &(lr, lc)) in game.levers.iter().enumerate() {
        if (lr, lc) == (r, c) {
            let is_on = (state.lever_state & (1 << l)) != 0;
            let color = if is_on { Color::Yellow } else { Color::Cyan };
            let emoji = if is_on { LEVER_ON } else { LEVER };
            return (emoji.into(), Style::default().fg(color));
        }
    }

    // Goal
    if game.goal == (r, c) {
        let all_boxes = (1 << game.boxes.len()) - 1;
        let emoji = if state.boxes_opened == all_boxes {
            GOAL_WIN
        } else {
            GOAL
        };
        return (emoji.into(), Style::default().fg(Color::Yellow));
    }

    // Trap
    if game.traps.contains(&(r, c)) {
        return (TRAP.into(), Style::default().fg(Color::Red));
    }

    // Bridge
    if game.bridge.contains(&(r, c)) {
        let emoji = if state.bridge_open {
            BRIDGE_OPEN
        } else {
            BRIDGE_CLOSED
        };
        return (emoji.into(), Style::default().fg(Color::Blue));
    }

    // Wall / Floor
    let emoji = if game.grid[r][c] == '#' { WALL } else { FLOOR };
    (emoji.into(), Style::default())
}

// ── Strategy Panel ─────────────────────────────────────────────

fn draw_strategy(f: &mut Frame, area: Rect, app: &App) {
    let targets = enumerate_targets(
        app.game.keys.len(),
        app.game.boxes.len(),
        app.game.levers.len(),
    );

    let cur_target = app.current_target_idx();

    let mut lines = Vec::new();
    for (visit_idx, &token_idx) in app.solution().target_sequence.iter().enumerate() {
        let target = &targets[token_idx];
        let icon = target_icon(target);
        let label = target_label(target);
        let pos = app.game.target_pos(target);

        let (status, status_style) = if Some(visit_idx) < cur_target {
            (CHECK, Style::default().fg(Color::Green))
        } else if Some(visit_idx) == cur_target && !app.is_at_end() {
            (
                ARROW,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            ("·", Style::default().fg(Color::DarkGray))
        };

        lines.push(Line::from(vec![
            Span::styled(format!(" {status} "), status_style),
            Span::styled(
                format!("{icon} {label:>8}"),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                format!(" ({},{})", pos.0, pos.1),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }

    let title = format!(" Strategy ({}) ", app.solution().target_sequence.len());
    let para = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(para, area);
}

// ── Status Panel ───────────────────────────────────────────────

fn draw_status(f: &mut Frame, area: Rect, app: &App) {
    let state = app.current_state();
    let phase = app.phase();

    let (phase_label, phase_color) = match phase {
        Phase::Moving => ("Moving", Color::Cyan),
        Phase::Done => ("Done 🎉", Color::Green),
    };

    let boss_status = if state.dead {
        format!("{SKULL} DEAD")
    } else if state.boss_alive {
        format!("{BOSS_LIVE} Alive ({},{})", state.boss_r, state.boss_c)
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

    let box_strs: Vec<String> = (0..app.game.boxes.len())
        .map(|j| {
            if (state.boxes_opened & (1 << j)) != 0 {
                format!("{BOX_OPEN}{j}")
            } else {
                format!("{BOX_CLOSED}{j}")
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
            Span::styled("  Boxes:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(box_strs.join(" "), Style::default().fg(Color::Magenta)),
        ]),
        Line::from(vec![
            Span::styled("  Levers: ", Style::default().fg(Color::DarkGray)),
            Span::styled(lever_strs.join(" "), Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("  Target: ", Style::default().fg(Color::DarkGray)),
            Span::styled(target_str, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![Span::styled(
            format!(
                "  Config: seed={} keys={:?} target=0b{:03b}",
                app.seed, app.game.key_mapping, app.game.target_lever_mask
            ),
            Style::default().fg(Color::DarkGray),
        )]),
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
        "No solution".into()
    } else if app.is_at_start() {
        format!("{ARROW} Start {ARROW}")
    } else if app.is_at_end() {
        match app.mode {
            SolveMode::BruteForce => {
                let state = app.current_state();
                if state.dead {
                    format!("{SKULL} Failed · → AI round {RABBIT}")
                } else {
                    format!("🎉 Done · → AI round {RABBIT}")
                }
            }
            SolveMode::Ai => {
                let bf_steps = app.bf.actions.len();
                let ai_steps = app.ai.actions.len();
                let bf_nodes = app.bf.tree_nodes;
                let ai_nodes = app.ai.tree_nodes;
                let bf_time = app.bf.solve_time_ms;
                let ai_time = app.ai.solve_time_ms;
                let discovered = app.ai.levers_discovered;

                let step_pct = if bf_steps > 0 {
                    100 - (ai_steps * 100 / bf_steps)
                } else {
                    0
                };
                let node_pct = if bf_nodes > 0 {
                    100 - (ai_nodes * 100 / bf_nodes)
                } else {
                    0
                };
                let speed_pct = if bf_time > 0 {
                    (bf_time.saturating_sub(ai_time)) * 100 / bf_time
                } else {
                    0
                };

                let step_label = if step_pct > 0 {
                    format!("⚡{step_pct}%↓")
                } else {
                    format!("{ai_steps}={bf_steps}")
                };

                format!(
                    "{RABBIT} steps {step_label} · tree {ai_nodes} vs {bf_nodes} ({node_pct}%↓) · \
                     {discovered} obs · {ai_time}ms vs {bf_time}ms ({speed_pct}%↑) · → Hybrid {FOX}"
                )
            }
            SolveMode::Hybrid => {
                let bf_steps = app.bf.actions.len();
                let ai_steps = app.ai.actions.len();
                let hy_steps = app.hybrid.actions.len();
                let bf_nodes = app.bf.tree_nodes;
                let hy_nodes = app.hybrid.tree_nodes;
                let bf_time = app.bf.solve_time_ms;
                let hy_time = app.hybrid.solve_time_ms;
                let discovered = app.hybrid.levers_discovered;

                let step_pct_vs_bf = if bf_steps > 0 {
                    100 - (hy_steps * 100 / bf_steps)
                } else {
                    0
                };
                let node_pct_vs_bf = if bf_nodes > 0 {
                    100 - (hy_nodes * 100 / bf_nodes)
                } else {
                    0
                };
                let speed_pct_vs_bf = if bf_time > 0 {
                    (bf_time.saturating_sub(hy_time)) * 100 / bf_time
                } else {
                    0
                };

                let step_label = if step_pct_vs_bf > 0 {
                    format!("⚡{step_pct_vs_bf}%↓")
                } else {
                    format!("{hy_steps}={bf_steps}")
                };
                let vs_ai = if hy_steps < ai_steps {
                    format!(" · 🏆 <{RABBIT}")
                } else if hy_steps > ai_steps {
                    let pct = (hy_steps - ai_steps) * 100 / ai_steps;
                    format!(" · +{pct}% vs {RABBIT}")
                } else {
                    String::new()
                };

                format!(
                    "{FOX} steps {step_label}{vs_ai} · tree {hy_nodes} vs {bf_nodes} ({node_pct_vs_bf}%↓) · \
                     {discovered} obs · {hy_time}ms vs {bf_time}ms ({speed_pct_vs_bf}%↑)"
                )
            }
        }
    } else {
        let action = app.solution().actions[cur - 1];
        let name = action_name(action);
        format!("Step {cur}/{total} — {name}")
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
