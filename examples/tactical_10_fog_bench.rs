//! Fog-of-War Exploration Strategy Benchmark — Headless
//!
//! Compares three fog-of-war exploration strategies across multiple seeds:
//! - BF (🐻): BFS to nearest frontier
//! - AI (🐰): Heuristic frontier scoring
//! - Hybrid (🦊): AI region selection + BF pathfinding
//!
//! Run: `cargo run --example tactical_10_fog_bench`

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

// ── Constants ──────────────────────────────────────────────────

const BOSS_SPEED: u32 = 2;
const VISION_RADIUS: usize = 4;
const MAX_STEPS: usize = 500;

const DIR_DELTA: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

// ── Map ────────────────────────────────────────────────────────

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

#[derive(Clone)]
struct SolveResult {
    steps: usize,
    solve_time_ms: u64,
    discovered_at_step: usize,
    success: bool,
}

// ── Game Engine ────────────────────────────────────────────────

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

            if grid[next.0][next.1] == '#' {
                visible.insert(next);
                continue;
            }

            if bridge.contains(&next) && !bridge_open {
                visible.insert(next);
                continue;
            }

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
    // Only lure when boss is visible and close enough to chase
    let boss_pos = fog.boss_last_known?;
    let alive = fog.boss_alive?;
    if !alive || !fog.visible.contains(&boss_pos) {
        return None;
    }

    let boss_dist = manhattan_dist((state.r, state.c), boss_pos);
    // Only consider luring when boss is reasonably close (within 8 tiles)
    if boss_dist > 8 {
        return None;
    }

    // For each discovered trap, find if there's a position on the far side
    // that would make the boss path through the trap
    let mut best_action = None;
    let mut best_score = i32::MIN;

    for &trap_pos in &fog.discovered_traps {
        // Find positions adjacent to the trap that are:
        // 1. Passable (floor, not trap)
        // 2. On the far side from the boss (manhattan dist from boss > trap dist from boss)
        // 3. Reachable from current position without stepping on traps
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
                continue; // don't step on other traps
            }
            if !fog.seen.contains(&adj) {
                continue; // must be explored territory
            }

            // Check if this position is on the far side of the trap from the boss
            let trap_to_boss = manhattan_dist(trap_pos, boss_pos);
            let adj_to_boss = manhattan_dist(adj, boss_pos);

            // "Far side" means: boss→trap→adj is the natural chase direction
            // adj should be farther from boss than the trap
            if adj_to_boss <= trap_to_boss {
                continue;
            }

            // Score: closer to current position is better (faster to reach)
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

    // 5. Any unopened box
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

struct AiExplorer;

impl AiExplorer {
    fn score_frontier(
        &self,
        pos: (usize, usize),
        game: &StrategicGame,
        state: &StrategicState,
        fog: &FogState,
    ) -> i32 {
        let mut score = 0i32;

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
                score += 2;
            } else if passable_n == 1 {
                score += 1;
            }
        }

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
            let best = frontiers
                .iter()
                .max_by_key(|&&pos| {
                    let score = self.score_frontier(pos, game, state, fog);
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

/// Steps without new tiles seen before switching to BF fallback.
const STUCK_THRESHOLD: usize = 15;

struct HybridExplorer {
    /// Steps since last new tile was seen.
    steps_without_progress: usize,
    /// Number of seen tiles at last check.
    last_seen_count: usize,
    /// Whether currently in BF fallback mode.
    bf_fallback: bool,
}

impl HybridExplorer {
    fn new() -> Self {
        Self {
            steps_without_progress: 0,
            last_seen_count: 0,
            bf_fallback: false,
        }
    }

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
        // Stuck detector: if no new tiles seen for STUCK_THRESHOLD steps,
        // switch to BF nearest-frontier mode.
        let current_seen = fog.seen.len();
        if current_seen > self.last_seen_count {
            self.steps_without_progress = 0;
            self.last_seen_count = current_seen;
            self.bf_fallback = false;
        } else {
            self.steps_without_progress += 1;
            if self.steps_without_progress >= STUCK_THRESHOLD {
                self.bf_fallback = true;
            }
        }

        let frontiers = fog.frontier_tiles(&game.grid);

        if !frontiers.is_empty() {
            // BF fallback: go to nearest frontier (same as BfExplorer)
            if self.bf_fallback {
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
                return navigate_to_known_target(game, state, fog);
            }

            let clusters = self.cluster_frontiers(&frontiers);

            let ai = AiExplorer;
            let best_cluster = clusters.iter().max_by_key(|cluster| {
                cluster
                    .iter()
                    .map(|&pos| ai.score_frontier(pos, game, state, fog))
                    .sum::<i32>()
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

fn solve_exploring<E: Explorer>(game: &StrategicGame, explorer: &mut E) -> SolveResult {
    let mut state = game.initial_state();
    let mut fog = FogState::new(game.keys.len(), game.boxes.len(), game.levers.len());
    let mut steps = 0;
    let mut discovered_at_step = 0;
    let mut all_discovered = false;
    let mut success = false;

    let visible = compute_visible(
        &game.grid,
        (state.r, state.c),
        state.bridge_open,
        &game.bridge,
    );
    fog.update(&visible, game, &state);

    let start = Instant::now();

    while steps < MAX_STEPS {
        if (state.r, state.c) == game.goal
            && state.boxes_opened == (1 << game.boxes.len()) - 1
            && state.bridge_open
        {
            success = true;
            break;
        }

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

        if state.dead {
            break;
        }

        let Some(action) = explorer.choose_action(game, &state, &fog) else {
            break;
        };

        let Some(next_state) = game.apply_action(&state, action) else {
            break;
        };
        state = next_state;
        steps += 1;

        let visible = compute_visible(
            &game.grid,
            (state.r, state.c),
            state.bridge_open,
            &game.bridge,
        );
        fog.update(&visible, game, &state);

        if state.dead {
            break;
        }
    }

    let solve_time_ms = start.elapsed().as_millis() as u64;

    SolveResult {
        steps,
        solve_time_ms,
        discovered_at_step,
        success,
    }
}

// ── Benchmark Result ───────────────────────────────────────────

#[derive(Debug)]
struct BenchResult {
    seed: u64,
    lever_mask: u8,
    bf_steps: usize,
    bf_ms: u64,
    bf_disc: usize,
    bf_success: bool,
    ai_steps: usize,
    ai_ms: u64,
    ai_disc: usize,
    ai_success: bool,
    hy_steps: usize,
    hy_ms: u64,
    hy_disc: usize,
    hy_success: bool,
}

fn bench_seed(seed: u64) -> BenchResult {
    let game = StrategicGame::new(MAP, seed);

    let mut bf_explorer = BfExplorer;
    let bf = solve_exploring(&game, &mut bf_explorer);

    let mut ai_explorer = AiExplorer;
    let ai = solve_exploring(&game, &mut ai_explorer);

    let mut hy_explorer = HybridExplorer::new();
    let hy = solve_exploring(&game, &mut hy_explorer);

    BenchResult {
        seed,
        lever_mask: game.target_lever_mask,
        bf_steps: bf.steps,
        bf_ms: bf.solve_time_ms,
        bf_disc: bf.discovered_at_step,
        bf_success: bf.success,
        ai_steps: ai.steps,
        ai_ms: ai.solve_time_ms,
        ai_disc: ai.discovered_at_step,
        ai_success: ai.success,
        hy_steps: hy.steps,
        hy_ms: hy.solve_time_ms,
        hy_disc: hy.discovered_at_step,
        hy_success: hy.success,
    }
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    let seeds: Vec<u64> = (42..=71).collect();
    let mut results = Vec::new();

    println!(
        "╔═════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║         🐻 BF vs 🐰 AI vs 🦊 Hybrid — Fog-of-War Exploration Strategy Benchmark                    ║"
    );
    println!(
        "╚═════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!();

    for &seed in &seeds {
        eprint!("seed={seed:>3}...");
        let r = bench_seed(seed);
        eprintln!(
            " BF:{}{} {:>3}ms d:{}  AI:{}{} {:>3}ms d:{}  HY:{}{} {:>3}ms d:{}",
            r.bf_steps,
            if r.bf_success { "✓" } else { "✗" },
            r.bf_ms,
            r.bf_disc,
            r.ai_steps,
            if r.ai_success { "✓" } else { "✗" },
            r.ai_ms,
            r.ai_disc,
            r.hy_steps,
            if r.hy_success { "✓" } else { "✗" },
            r.hy_ms,
            r.hy_disc,
        );
        results.push(r);
    }

    println!();

    // ── Per-seed comparison table ──
    println!(
        "┌──────┬────────┬─────────────────────────────────┬─────────────────────────────────┬─────────────────────────────────┐"
    );
    println!(
        "│ Seed │ LvrMsk │ 🐻 BF (Steps/Disc/ms/OK)       │ 🐰 AI (Steps/Disc/ms/OK)       │ 🦊 Hybrid (Steps/Disc/ms/OK)   │"
    );
    println!(
        "├──────┼────────┼─────────────────────────────────┼─────────────────────────────────┼─────────────────────────────────┤"
    );

    for r in &results {
        let bf_tag = if r.bf_success { "✓" } else { "✗" };
        let ai_tag = if r.ai_success { "✓" } else { "✗" };
        let hy_tag = if r.hy_success { "✓" } else { "✗" };

        let bf_step_diff = format_step_diff(r.bf_steps, r.ai_steps, r.hy_steps);
        let ai_step_diff = format_step_diff(r.ai_steps, r.bf_steps, r.hy_steps);
        let hy_step_diff = format_step_diff(r.hy_steps, r.bf_steps, r.ai_steps);

        println!(
            "│ {:>4} │ 0b{:03b}  │ {:>4}/{:>3}/{:>3}ms {} {:>6} │ {:>4}/{:>3}/{:>3}ms {} {:>6} │ {:>4}/{:>3}/{:>3}ms {} {:>6} │",
            r.seed,
            r.lever_mask,
            r.bf_steps,
            r.bf_disc,
            r.bf_ms,
            bf_tag,
            bf_step_diff,
            r.ai_steps,
            r.ai_disc,
            r.ai_ms,
            ai_tag,
            ai_step_diff,
            r.hy_steps,
            r.hy_disc,
            r.hy_ms,
            hy_tag,
            hy_step_diff,
        );
    }

    println!(
        "└──────┴────────┴─────────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┘"
    );

    // ── Aggregate stats ──
    let count = results.len();
    if count == 0 {
        println!("\n⚠ No seeds in range");
        return;
    }

    let avg_bf_steps: f64 = results.iter().map(|r| r.bf_steps as f64).sum::<f64>() / count as f64;
    let avg_ai_steps: f64 = results.iter().map(|r| r.ai_steps as f64).sum::<f64>() / count as f64;
    let avg_hy_steps: f64 = results.iter().map(|r| r.hy_steps as f64).sum::<f64>() / count as f64;

    let avg_bf_ms: f64 = results.iter().map(|r| r.bf_ms as f64).sum::<f64>() / count as f64;
    let avg_ai_ms: f64 = results.iter().map(|r| r.ai_ms as f64).sum::<f64>() / count as f64;
    let avg_hy_ms: f64 = results.iter().map(|r| r.hy_ms as f64).sum::<f64>() / count as f64;

    let avg_bf_disc: f64 = results.iter().map(|r| r.bf_disc as f64).sum::<f64>() / count as f64;
    let avg_ai_disc: f64 = results.iter().map(|r| r.ai_disc as f64).sum::<f64>() / count as f64;
    let avg_hy_disc: f64 = results.iter().map(|r| r.hy_disc as f64).sum::<f64>() / count as f64;

    let bf_wins = results.iter().filter(|r| r.bf_success).count();
    let ai_wins = results.iter().filter(|r| r.ai_success).count();
    let hy_wins = results.iter().filter(|r| r.hy_success).count();

    let ai_beats_bf = results
        .iter()
        .filter(|r| r.ai_success && r.bf_success && r.ai_steps < r.bf_steps)
        .count();
    let hy_beats_bf = results
        .iter()
        .filter(|r| r.hy_success && r.bf_success && r.hy_steps < r.bf_steps)
        .count();
    let bf_beats_ai = results
        .iter()
        .filter(|r| r.ai_success && r.bf_success && r.bf_steps < r.ai_steps)
        .count();
    let bf_beats_hy = results
        .iter()
        .filter(|r| r.hy_success && r.bf_success && r.bf_steps < r.hy_steps)
        .count();
    let ai_beats_hy = results
        .iter()
        .filter(|r| r.ai_success && r.hy_success && r.ai_steps < r.hy_steps)
        .count();
    let hy_beats_ai = results
        .iter()
        .filter(|r| r.ai_success && r.hy_success && r.hy_steps < r.ai_steps)
        .count();

    let solvable = results
        .iter()
        .filter(|r| r.bf_success || r.ai_success || r.hy_success)
        .count();

    println!();
    println!("┌─────────────────────────────────────────────────────────────────────────────┐");
    println!(
        "│ Aggregate ({count} seeds, {solvable} with ≥1 solver success)                       │"
    );
    println!("├─────────────────────────────────────────────────────────────────────────────┤");
    println!("│                  │ 🐻 BF       │ 🐰 AI       │ 🦊 Hybrid    │");
    println!(
        "│ Avg Steps        │ {avg_bf_steps:>11.1} │ {avg_ai_steps:>11.1} │ {avg_hy_steps:>12.1} │"
    );
    println!("│ Avg Time(ms)     │ {avg_bf_ms:>11.1} │ {avg_ai_ms:>11.1} │ {avg_hy_ms:>12.1} │");
    println!(
        "│ Avg Disc Step    │ {avg_bf_disc:>11.1} │ {avg_ai_disc:>11.1} │ {avg_hy_disc:>12.1} │"
    );
    println!("│ Wins (success)   │ {bf_wins:>11} │ {ai_wins:>11} │ {hy_wins:>12} │");
    println!("├──────────────────┴─────────────┴──────────────┴───────────────┤");
    println!(
        "│ 🐰 AI beats 🐻 BF in steps:   {ai_beats_bf:>3}/{count} seeds ({:.0}%)",
        ai_beats_bf as f64 / count as f64 * 100.0
    );
    println!(
        "│ 🦊 HY beats 🐻 BF in steps:   {hy_beats_bf:>3}/{count} seeds ({:.0}%)",
        hy_beats_bf as f64 / count as f64 * 100.0
    );
    println!(
        "│ 🐻 BF beats 🐰 AI in steps:   {bf_beats_ai:>3}/{count} seeds ({:.0}%)",
        bf_beats_ai as f64 / count as f64 * 100.0
    );
    println!(
        "│ 🐻 BF beats 🦊 HY in steps:   {bf_beats_hy:>3}/{count} seeds ({:.0}%)",
        bf_beats_hy as f64 / count as f64 * 100.0
    );
    println!(
        "│ 🐰 AI beats 🦊 HY in steps:   {ai_beats_hy:>3}/{count} seeds ({:.0}%)",
        ai_beats_hy as f64 / count as f64 * 100.0
    );
    println!(
        "│ 🦊 HY beats 🐰 AI in steps:   {hy_beats_ai:>3}/{count} seeds ({:.0}%)",
        hy_beats_ai as f64 / count as f64 * 100.0
    );
    println!("└─────────────────────────────────────────────────────────────────────────────┘");

    // ── Verdict ──
    println!();
    println!("🔍 Verdict:");

    println!(
        "   🐻 BF:     avg {avg_bf_steps:.1} steps, avg disc at {avg_bf_disc:.1}, {bf_wins}/{count} wins"
    );
    println!(
        "   🐰 AI:     avg {avg_ai_steps:.1} steps, avg disc at {avg_ai_disc:.1}, {ai_wins}/{count} wins"
    );
    println!(
        "   🦊 Hybrid: avg {avg_hy_steps:.1} steps, avg disc at {avg_hy_disc:.1}, {hy_wins}/{count} wins"
    );

    println!();
    println!("   📊 Discovery Efficiency (lower disc step = faster map knowledge):");
    let disc_winner = if avg_ai_disc <= avg_bf_disc && avg_ai_disc <= avg_hy_disc {
        "🐰 AI"
    } else if avg_hy_disc <= avg_bf_disc && avg_hy_disc <= avg_ai_disc {
        "🦊 Hybrid"
    } else {
        "🐻 BF"
    };
    println!(
        "      🏆 {disc_winner} discovers all targets earliest (BF:{avg_bf_disc:.1} AI:{avg_ai_disc:.1} HY:{avg_hy_disc:.1})"
    );

    let step_winner = if avg_ai_steps <= avg_bf_steps && avg_ai_steps <= avg_hy_steps {
        "🐰 AI"
    } else if avg_hy_steps <= avg_bf_steps && avg_hy_steps <= avg_ai_steps {
        "🦊 Hybrid"
    } else {
        "🐻 BF"
    };
    println!(
        "      🏆 {step_winner} solves in fewest steps (BF:{avg_bf_steps:.1} AI:{avg_ai_steps:.1} HY:{avg_hy_steps:.1})"
    );

    let all_three_win = results
        .iter()
        .filter(|r| r.bf_success && r.ai_success && r.hy_success)
        .count();
    println!(
        "   📊 All 3 solvers succeed: {all_three_win}/{count} seeds ({:.0}%)",
        all_three_win as f64 / count as f64 * 100.0
    );
    println!(
        "   📊 Fog-of-war differentiation: AI vs HY disagree on {} seeds",
        results
            .iter()
            .filter(|r| r.ai_success && r.hy_success && r.ai_steps != r.hy_steps)
            .count()
    );
}

fn format_step_diff(own: usize, other_a: usize, other_b: usize) -> String {
    let min_other = other_a.min(other_b);
    if own < min_other {
        format!("⚡{}", min_other - own)
    } else if own > other_a || own > other_b {
        let best = other_a.min(other_b);
        if own > best {
            format!("+{}", own - best)
        } else {
            "=".into()
        }
    } else {
        "=".into()
    }
}
