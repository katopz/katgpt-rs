//! BomberState snapshot — lightweight Clone struct for Bomber forward model.
//!
//! Full implementation of `GameState` trait for the Bomberman arena.
//! Simulates deterministic game mechanics on a snapshot struct (~2KB),
//! avoiding `bevy_ecs::World` (which isn't `Clone`).
//!
//! Game tick order (matches ECS `run_tick()`):
//! 1. Tick bomb fuses → collect expired
//! 2. Process explosions → chain reactions, destroy walls, kill players, reveal power-ups
//! 3. Apply player action → movement, bomb placement, or wait
//! 4. Collect revealed power-ups
//! 5. Increment tick

use std::collections::{HashSet, VecDeque};

use katgpt_core::traits::{GameState, StateHeuristic};
use crate::pruners::bomber::{
    ARENA_H, ARENA_W, ArenaGrid, BOMB_FUSE_TICKS, BomberAction, Cell, DEFAULT_BLAST_RANGE,
    DEFAULT_MAX_BOMBS, PowerUpKind, SPAWN_POSITIONS, TICK_LIMIT,
};

/// Four cardinal directions for blast propagation and movement.
const DIRECTIONS: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

// ── Snapshot Types ─────────────────────────────────────────────

/// Lightweight player snapshot for forward model simulation.
#[derive(Clone, Debug)]
pub struct PlayerSnapshot {
    pub pos: (i32, i32),
    pub alive: bool,
    pub max_bombs: u8,
    pub active_bombs: u8,
    pub blast_range: u32,
}

impl Default for PlayerSnapshot {
    fn default() -> Self {
        Self {
            pos: (0, 0),
            alive: true,
            max_bombs: DEFAULT_MAX_BOMBS,
            active_bombs: 0,
            blast_range: DEFAULT_BLAST_RANGE,
        }
    }
}

/// Lightweight bomb snapshot for forward model simulation.
#[derive(Clone, Debug)]
pub struct BombSnapshot {
    pub pos: (i32, i32),
    pub fuse: u32,
    pub range: u32,
    pub owner: u8,
}

/// Lightweight Bomberman state snapshot for forward model simulation.
///
/// Extract only what MCTS needs — no ECS dependency in the trait.
/// The arena converts `World → BomberState` once per tick.
#[derive(Clone, Debug)]
pub struct BomberState {
    /// 13×13 flat grid: `cells[y * ARENA_W + x]`
    pub cells: [Cell; ARENA_W * ARENA_H],
    pub players: [PlayerSnapshot; 4],
    pub bombs: Vec<BombSnapshot>,
    /// Power-ups revealed by blast (waiting to be collected).
    pub revealed_powerups: Vec<((i32, i32), PowerUpKind)>,
    pub tick: u32,
    pub max_ticks: u32,
}

// ── BomberState Helpers ────────────────────────────────────────

impl BomberState {
    /// Create initial state from an arena grid at tick 0.
    ///
    /// Players placed at standard spawn positions with default stats.
    /// No bombs on the field, no revealed power-ups.
    pub fn from_grid(grid: &ArenaGrid) -> Self {
        let players = SPAWN_POSITIONS.map(|(x, y)| PlayerSnapshot {
            pos: (x, y),
            ..PlayerSnapshot::default()
        });

        let mut cells = [Cell::FixedWall; ARENA_W * ARENA_H];
        for y in 0..ARENA_H {
            for x in 0..ARENA_W {
                cells[y * ARENA_W + x] = grid.cells[y][x];
            }
        }

        Self {
            cells,
            players,
            bombs: Vec::new(),
            revealed_powerups: Vec::new(),
            tick: 0,
            max_ticks: TICK_LIMIT,
        }
    }

    /// Count alive players.
    pub fn alive_count(&self) -> usize {
        self.players.iter().filter(|p| p.alive).count()
    }

    /// Safe cell access. Returns `FixedWall` for out-of-bounds.
    fn get_cell(&self, x: i32, y: i32) -> Cell {
        if x < 0 || (x as usize) >= ARENA_W || y < 0 || (y as usize) >= ARENA_H {
            Cell::FixedWall
        } else {
            self.cells[y as usize * ARENA_W + x as usize]
        }
    }

    /// Set cell at (x, y). No-op for out-of-bounds.
    fn set_cell(&mut self, x: i32, y: i32, cell: Cell) {
        if x >= 0 && (x as usize) < ARENA_W && y >= 0 && (y as usize) < ARENA_H {
            self.cells[y as usize * ARENA_W + x as usize] = cell;
        }
    }

    /// True if the cell is walkable (Floor only, no bomb entity).
    pub fn is_walkable(&self, x: i32, y: i32) -> bool {
        if x < 0 || (x as usize) >= ARENA_W || y < 0 || (y as usize) >= ARENA_H {
            return false;
        }
        matches!(self.cells[y as usize * ARENA_W + x as usize], Cell::Floor)
            && !self.bombs.iter().any(|b| b.pos == (x, y))
    }

    /// Compute target position after applying a directional action.
    fn move_target(action: BomberAction, pos: (i32, i32)) -> (i32, i32) {
        match action {
            BomberAction::Up => (pos.0, pos.1 - 1),
            BomberAction::Down => (pos.0, pos.1 + 1),
            BomberAction::Left => (pos.0 - 1, pos.1),
            BomberAction::Right => (pos.0 + 1, pos.1),
            BomberAction::Bomb | BomberAction::Wait | BomberAction::Detonate => pos,
        }
    }

    /// Check if position is in the blast zone of a single bomb (with wall blocking).
    fn is_pos_in_blast(&self, pos: (i32, i32), bomb_pos: (i32, i32), range: u32) -> bool {
        let (px, py) = pos;
        let (bx, by) = bomb_pos;

        // Standing on the bomb itself
        if px == bx && py == by {
            return true;
        }

        // Horizontal blast
        if py == by {
            let dx = px - bx;
            if dx.unsigned_abs() <= range {
                let step = dx.signum();
                let mut x = bx + step;
                while x != px {
                    if !matches!(self.get_cell(x, by), Cell::Floor) {
                        return false;
                    }
                    x += step;
                }
                return true;
            }
        }

        // Vertical blast
        if px == bx {
            let dy = py - by;
            if dy.unsigned_abs() <= range {
                let step = dy.signum();
                let mut y = by + step;
                while y != py {
                    if !matches!(self.get_cell(bx, y), Cell::Floor) {
                        return false;
                    }
                    y += step;
                }
                return true;
            }
        }

        false
    }

    /// Check if position is in ANY bomb's blast zone.
    pub fn is_in_blast_zone(&self, pos: (i32, i32)) -> bool {
        self.bombs
            .iter()
            .any(|b| self.is_pos_in_blast(pos, b.pos, b.range))
    }

    /// Pre-compute the full blast zone grid (169 cells) for all bombs.
    ///
    /// Each cell is `true` if it falls within any bomb's blast range,
    /// accounting for wall blocking (same rules as `is_pos_in_blast`).
    /// O(bombs × range) once, then O(1) lookups.
    fn compute_blast_zone(&self) -> [bool; ARENA_W * ARENA_H] {
        let mut zone = [false; ARENA_W * ARENA_H];
        for bomb in &self.bombs {
            let bx = bomb.pos.0;
            let by = bomb.pos.1;

            // Mark bomb cell itself
            if bx >= 0 && (bx as usize) < ARENA_W && by >= 0 && (by as usize) < ARENA_H {
                zone[by as usize * ARENA_W + bx as usize] = true;
            }

            // Propagate in 4 cardinal directions
            for &(dx, dy) in &DIRECTIONS {
                for dist in 1..=bomb.range as i32 {
                    let cx = bx + dx * dist;
                    let cy = by + dy * dist;
                    if cx < 0 || (cx as usize) >= ARENA_W || cy < 0 || (cy as usize) >= ARENA_H {
                        break;
                    }
                    let ci = cy as usize * ARENA_W + cx as usize;
                    match self.cells[ci] {
                        Cell::FixedWall => break,
                        Cell::DestructibleWall | Cell::PowerUpHidden(_) => {
                            zone[ci] = true;
                            break;
                        }
                        Cell::Floor => zone[ci] = true,
                    }
                }
            }
        }
        zone
    }

    /// BFS escape distance from blast zone. Returns `None` if trapped.
    ///
    /// Uses pre-computed blast zone (O(bombs×range) once) and flat bitset
    /// visited array instead of HashSet.
    pub fn escape_distance(&self, pos: (i32, i32)) -> Option<i32> {
        let blast = self.compute_blast_zone();
        self.escape_distance_with_blast(pos, &blast)
    }

    /// BFS escape using a pre-computed blast zone grid.
    fn escape_distance_with_blast(
        &self,
        pos: (i32, i32),
        blast: &[bool; ARENA_W * ARENA_H],
    ) -> Option<i32> {
        let pi = pos.1 as usize * ARENA_W + pos.0 as usize;
        if !blast[pi] {
            return Some(0);
        }

        let mut visited = [false; ARENA_W * ARENA_H];
        let mut queue = VecDeque::new();

        visited[pi] = true;
        queue.push_back((pos, 0));

        while let Some(((cx, cy), dist)) = queue.pop_front() {
            for (nx, ny) in [(cx, cy - 1), (cx, cy + 1), (cx - 1, cy), (cx + 1, cy)] {
                if nx < 0 || (nx as usize) >= ARENA_W || ny < 0 || (ny as usize) >= ARENA_H {
                    continue;
                }
                let ni = ny as usize * ARENA_W + nx as usize;
                if visited[ni] {
                    continue;
                }
                visited[ni] = true;
                if !self.is_walkable(nx, ny) {
                    continue;
                }
                let next_dist = dist + 1;
                if !blast[ni] {
                    return Some(next_dist);
                }
                queue.push_back(((nx, ny), next_dist));
            }
        }

        None
    }

    /// Count walkable adjacent cells from a position.
    fn count_escape_routes(&self, pos: (i32, i32)) -> usize {
        DIRECTIONS
            .iter()
            .filter(|&&(dx, dy)| self.is_walkable(pos.0 + dx, pos.1 + dy))
            .count()
    }

    /// Kill all players at the given position.
    fn kill_players_at(&mut self, pos: (i32, i32)) {
        for player in &mut self.players {
            if player.alive && player.pos == pos {
                player.alive = false;
            }
        }
    }

    /// Decrement bomb fuses and return expired bombs.
    fn tick_bomb_fuses(&mut self) -> Vec<BombSnapshot> {
        let mut expired = Vec::new();
        for bomb in &mut self.bombs {
            bomb.fuse = bomb.fuse.saturating_sub(1);
            if bomb.fuse == 0 {
                expired.push(bomb.clone());
            }
        }
        expired
    }

    /// Process chain explosions: destroy walls, kill players, reveal power-ups.
    fn process_explosions(&mut self, expired: Vec<BombSnapshot>) {
        if expired.is_empty() {
            return;
        }

        // Phase 1: Find all bombs that will explode (chain reactions).
        // Uses current cell state — walls block blast propagation.
        let mut to_explode: HashSet<(i32, i32)> = HashSet::new();
        let mut queue: Vec<BombSnapshot> = expired;

        while let Some(bomb) = queue.pop() {
            if to_explode.contains(&bomb.pos) {
                continue;
            }
            to_explode.insert(bomb.pos);

            for &(dx, dy) in &DIRECTIONS {
                for dist in 1..=bomb.range as i32 {
                    let cx = bomb.pos.0 + dx * dist;
                    let cy = bomb.pos.1 + dy * dist;

                    match self.get_cell(cx, cy) {
                        Cell::FixedWall => break,
                        Cell::DestructibleWall | Cell::PowerUpHidden(_) => break,
                        Cell::Floor => {
                            // Chain reaction: check for unexploded bomb
                            if let Some(chain) = self
                                .bombs
                                .iter()
                                .find(|b| b.pos == (cx, cy) && !to_explode.contains(&b.pos))
                            {
                                queue.push(chain.clone());
                            }
                        }
                    }
                }
            }
        }

        // Phase 2: Apply effects using to_explode set.
        for &pos in &to_explode {
            let bomb = self.bombs.iter().find(|b| b.pos == pos);
            let range = bomb.map(|b| b.range).unwrap_or(DEFAULT_BLAST_RANGE);
            let owner = bomb.map(|b| b.owner);

            // Decrement owner's active bomb count
            if let Some(oid) = owner {
                self.players[oid as usize].active_bombs =
                    self.players[oid as usize].active_bombs.saturating_sub(1);
            }

            // Kill players at bomb position
            self.kill_players_at(pos);

            // Propagate blast in 4 directions
            for &(dx, dy) in &DIRECTIONS {
                for dist in 1..=range as i32 {
                    let cx = pos.0 + dx * dist;
                    let cy = pos.1 + dy * dist;

                    match self.get_cell(cx, cy) {
                        Cell::FixedWall => break,
                        Cell::DestructibleWall => {
                            self.set_cell(cx, cy, Cell::Floor);
                            self.kill_players_at((cx, cy));
                            break;
                        }
                        Cell::PowerUpHidden(kind) => {
                            self.set_cell(cx, cy, Cell::Floor);
                            self.revealed_powerups.push(((cx, cy), kind));
                            self.kill_players_at((cx, cy));
                            break;
                        }
                        Cell::Floor => {
                            self.kill_players_at((cx, cy));
                        }
                    }
                }
            }
        }

        // Phase 3: Remove all exploded bombs
        self.bombs.retain(|b| !to_explode.contains(&b.pos));
    }

    /// Apply a single player's action (movement, bomb, or wait).
    fn apply_action(&mut self, action: &BomberAction, player_id: u8) {
        // Read phase: snapshot player data to avoid borrow conflicts
        let player = &self.players[player_id as usize];
        if !player.alive {
            return;
        }
        let player_pos = player.pos;
        let can_bomb = player.active_bombs < player.max_bombs;
        let blast_range = player.blast_range;
        let has_bomb_at_pos = self.bombs.iter().any(|b| b.pos == player_pos);

        // Write phase: compute target first, then mutate
        match action {
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right => {
                let target = Self::move_target(*action, player_pos);
                if self.is_walkable(target.0, target.1) {
                    self.players[player_id as usize].pos = target;
                }
                // If blocked, position unchanged (action wasted)
            }
            BomberAction::Bomb => {
                if can_bomb && !has_bomb_at_pos {
                    self.bombs.push(BombSnapshot {
                        pos: player_pos,
                        fuse: BOMB_FUSE_TICKS,
                        range: blast_range,
                        owner: player_id,
                    });
                    self.players[player_id as usize].active_bombs += 1;
                }
            }
            BomberAction::Wait | BomberAction::Detonate => {}
        }
    }

    /// Collect revealed power-ups at player positions.
    fn collect_powerups(&mut self) {
        // Phase 1: Find collections (player_idx, position, kind)
        let mut collections: Vec<(usize, (i32, i32), PowerUpKind)> = Vec::new();
        let mut claimed: HashSet<(i32, i32)> = HashSet::new();

        for i in 0..4 {
            let player = &self.players[i];
            if !player.alive {
                continue;
            }
            for (pos, kind) in &self.revealed_powerups {
                if *pos == player.pos && claimed.insert(*pos) {
                    collections.push((i, *pos, *kind));
                    break;
                }
            }
        }

        // Phase 2: Remove collected power-ups
        for (_, pos, _) in &collections {
            if let Some(idx) = self.revealed_powerups.iter().position(|(p, _)| p == pos) {
                self.revealed_powerups.remove(idx);
            }
        }

        // Phase 3: Apply effects
        for (player_idx, _, kind) in collections {
            let player = &mut self.players[player_idx];
            match kind {
                PowerUpKind::BombUp => player.max_bombs += 1,
                PowerUpKind::FireUp => player.blast_range += 1,
                PowerUpKind::SpeedUp => { /* speed not tracked in snapshot */ }
            }
        }
    }
}

// ── GameState Implementation ───────────────────────────────────

impl GameState for BomberState {
    type Action = BomberAction;

    fn available_actions(&self, player_id: u8) -> Vec<Self::Action> {
        let mut buf = Vec::with_capacity(6);
        self.available_actions_into(player_id, &mut buf);
        buf
    }

    fn available_actions_into(&self, player_id: u8, buf: &mut Vec<Self::Action>) {
        buf.clear();
        let player = &self.players[player_id as usize];
        if !player.alive {
            return;
        }

        // Movement: check walkability (cell type + bomb blocking)
        for &action in &[
            BomberAction::Up,
            BomberAction::Down,
            BomberAction::Left,
            BomberAction::Right,
        ] {
            let target = Self::move_target(action, player.pos);
            if self.is_walkable(target.0, target.1) {
                buf.push(action);
            }
        }

        // Bomb: check capacity and no bomb at current position
        if player.active_bombs < player.max_bombs && !self.bombs.iter().any(|b| b.pos == player.pos)
        {
            buf.push(BomberAction::Bomb);
        }

        // Wait: always legal when alive
        buf.push(BomberAction::Wait);
    }

    fn action_space_size(&self, player_id: u8) -> usize {
        let player = &self.players[player_id as usize];
        if !player.alive {
            return 0;
        }

        // Wait is always legal
        let mut count = 1;

        // Movement directions
        for &action in &[
            BomberAction::Up,
            BomberAction::Down,
            BomberAction::Left,
            BomberAction::Right,
        ] {
            let target = Self::move_target(action, player.pos);
            if self.is_walkable(target.0, target.1) {
                count += 1;
            }
        }

        // Bomb
        if player.active_bombs < player.max_bombs && !self.bombs.iter().any(|b| b.pos == player.pos)
        {
            count += 1;
        }

        count
    }

    fn advance(&self, action: &Self::Action, player_id: u8) -> Self {
        let mut state = self.clone();

        // 1. Tick bomb fuses, collect expired
        let expired = state.tick_bomb_fuses();

        // 2. Process explosions (chain reactions, destroy walls, kill players)
        state.process_explosions(expired);

        // 3. Apply player action (movement, bomb, or wait)
        state.apply_action(action, player_id);

        // 4. Collect revealed power-ups at player positions
        state.collect_powerups();

        // 5. Increment tick
        state.tick += 1;

        state
    }

    fn is_terminal(&self) -> bool {
        self.alive_count() <= 1 || self.tick >= self.max_ticks
    }

    fn reward(&self, player_id: u8) -> f32 {
        let player = &self.players[player_id as usize];
        match (player.alive, self.alive_count()) {
            // Sole survivor → win
            (true, 1) => 1.0,
            // Dead → loss
            (false, _) => 0.0,
            // Multiple alive or all dead (tick limit) → partial
            (true, _) => 0.5,
        }
    }

    fn tick(&self) -> u32 {
        self.tick
    }
}

// ── BomberHeuristic ────────────────────────────────────────────

/// Domain-specific heuristic for Bomberman state evaluation.
///
/// Adapted from `score_action()` in `players.rs`:
/// - Safety: penalize blast zone exposure, reward escape routes
/// - Resources: reward power-up collection potential
/// - Position: slight center bias
/// - Progress: reward fewer alive opponents
pub struct BomberHeuristic;

impl StateHeuristic<BomberState> for BomberHeuristic {
    fn evaluate(&self, state: &BomberState, player_id: u8) -> f32 {
        let player = &state.players[player_id as usize];
        if !player.alive {
            return -1.0;
        }

        let mut score = 0.0;

        // Pre-compute blast zone once — shared for safety check + escape distance
        let blast = state.compute_blast_zone();
        let pi = player.pos.1 as usize * ARENA_W + player.pos.0 as usize;

        // Safety: penalize blast zone exposure
        if blast[pi] {
            match state.escape_distance_with_blast(player.pos, &blast) {
                Some(d) => score -= 5.0 + d as f32 * 0.5,
                None => return -0.8, // Trapped = nearly dead
            }
        } else {
            score += 2.0; // Safe is good
        }

        // Escape routes: more routes = safer position
        let routes = state.count_escape_routes(player.pos);
        score += routes as f32 * 0.3;

        // Power-ups: reward proximity to revealed power-ups
        for &(pos, _) in &state.revealed_powerups {
            let dist = (player.pos.0 - pos.0).abs() + (player.pos.1 - pos.1).abs();
            if dist == 0 {
                score += 2.0; // Standing on power-up (will collect next tick)
            } else if dist <= 3 {
                score += 0.5; // Near power-up
            }
        }

        // Resources: collected power-ups improve future potential
        score += (player.max_bombs - DEFAULT_MAX_BOMBS) as f32 * 0.3;
        score += (player.blast_range - DEFAULT_BLAST_RANGE) as f32 * 0.3;

        // Position: slight center bias (more options near center)
        let center_dist = (player.pos.0 - 6).abs() as f32 + (player.pos.1 - 6).abs() as f32;
        score -= center_dist * 0.02;

        // Progress: fewer opponents = closer to winning
        let alive_opponents = state
            .players
            .iter()
            .enumerate()
            .filter(|(i, p)| *i != player_id as usize && p.alive)
            .count();
        score += (3 - alive_opponents) as f32 * 1.0;

        score
    }
}

// ── BanditBomberHeuristic ─────────────────────────────────────

/// Combined heuristic: domain knowledge + bandit backward signal.
///
/// Plan 067 (NFSP/MCTS Duality): fuses `BomberHeuristic` (forward-looking
/// domain features) with `BanditStats` Q-values (backward-looking episode data).
///
/// Evaluation: `domain.evaluate(s, pid) + λ * avg_q_bonus(s, pid)`
///
/// The `λ` (bandit_weight) controls how much to trust bandit Q-values vs domain:
/// - `λ = 0.0`: pure domain heuristic (equivalent to `BomberHeuristic`)
/// - `λ = 1.0`: equal weight
/// - `λ = 2.0+`: bandit signal dominates
///
/// The Q-value bonus is the average Q-value of available actions (higher → state
/// is reachable via historically good actions).
#[cfg(feature = "bandit")]
pub struct BanditBomberHeuristic {
    /// Domain-specific heuristic (safety, resources, position, progress).
    domain: BomberHeuristic,
    /// Bandit statistics accumulated across episodes.
    stats: crate::pruners::bandit::BanditStats,
    /// Weight for bandit Q-value bonus (λ). Domain heuristic is always weight 1.0.
    bandit_weight: f32,
}

#[cfg(feature = "bandit")]
impl BanditBomberHeuristic {
    /// Create a new combined heuristic.
    ///
    /// # Arguments
    /// * `stats` — bandit statistics with accumulated Q-values from prior episodes
    /// * `bandit_weight` — weight for bandit Q-value bonus (0.0 = pure domain)
    ///
    /// # Panics
    /// Panics if `bandit_weight` is negative.
    pub fn new(stats: crate::pruners::bandit::BanditStats, bandit_weight: f32) -> Self {
        assert!(
            bandit_weight >= 0.0,
            "bandit_weight must be >= 0.0, got {bandit_weight}"
        );
        Self {
            domain: BomberHeuristic,
            stats,
            bandit_weight,
        }
    }

    /// Update bandit statistics after an episode.
    pub fn update(&mut self, arm: usize, reward: f32) {
        self.stats.update(arm, reward);
    }

    /// Access the underlying bandit statistics.
    pub fn stats(&self) -> &crate::pruners::bandit::BanditStats {
        &self.stats
    }

    /// Access the underlying bandit statistics (mutable).
    pub fn stats_mut(&mut self) -> &mut crate::pruners::bandit::BanditStats {
        &mut self.stats
    }
}

#[cfg(feature = "bandit")]
impl StateHeuristic<BomberState> for BanditBomberHeuristic {
    fn evaluate(&self, state: &BomberState, player_id: u8) -> f32 {
        let domain_score = self.domain.evaluate(state, player_id);

        // Bandit Q-value bonus: average Q of available actions
        let actions = state.available_actions(player_id);
        if actions.is_empty() || self.bandit_weight == 0.0 {
            return domain_score;
        }

        let q_sum: f32 = actions
            .iter()
            .map(|a| self.stats.q_value(a.as_usize()))
            .sum();
        let avg_q = q_sum / actions.len() as f32;
        let bandit_bonus = avg_q * self.bandit_weight;

        domain_score + bandit_bonus
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bomber::arena::EMPTY_ARENA;

    /// Create a state from the empty arena template (no destructible walls).
    fn empty_state() -> BomberState {
        BomberState::from_grid(&ArenaGrid::fixed(EMPTY_ARENA).unwrap())
    }

    #[test]
    fn from_grid_creates_valid_state() {
        let grid = ArenaGrid::generate(42);
        let state = BomberState::from_grid(&grid);

        assert_eq!(state.tick, 0);
        assert_eq!(state.bombs.len(), 0);
        assert!(state.revealed_powerups.is_empty());
        assert_eq!(state.alive_count(), 4);
        assert!(!state.is_terminal());

        // Players at spawn positions
        for (i, &(sx, sy)) in SPAWN_POSITIONS.iter().enumerate() {
            assert_eq!(state.players[i].pos, (sx, sy));
            assert!(state.players[i].alive);
        }
    }

    #[test]
    fn dead_player_has_no_actions() {
        let grid = ArenaGrid::generate(42);
        let mut state = BomberState::from_grid(&grid);
        state.players[0].alive = false;

        assert!(state.available_actions(0).is_empty());
    }

    #[test]
    fn terminal_when_one_alive() {
        let grid = ArenaGrid::generate(42);
        let mut state = BomberState::from_grid(&grid);
        state.players[1].alive = false;
        state.players[2].alive = false;
        state.players[3].alive = false;

        assert!(state.is_terminal());
        assert!((state.reward(0) - 1.0).abs() < f32::EPSILON);
        assert!((state.reward(1)).abs() < f32::EPSILON);
    }

    #[test]
    fn terminal_at_tick_limit() {
        let grid = ArenaGrid::generate(42);
        let mut state = BomberState::from_grid(&grid);
        state.tick = TICK_LIMIT;

        assert!(state.is_terminal());
    }

    // ── Movement Tests ─────────────────────────────────────────

    #[test]
    fn advance_moves_player() {
        let state = empty_state();
        // Player 0 at (1,1), move right to (2,1) — pillar at (2,2) but (2,1) is floor
        assert_eq!(state.players[0].pos, (1, 1));

        let next = state.advance(&BomberAction::Right, 0);
        assert_eq!(next.players[0].pos, (2, 1));
        assert_eq!(state.players[0].pos, (1, 1)); // original unchanged (pure)
    }

    #[test]
    fn advance_blocked_by_wall() {
        let state = empty_state();
        // Player 0 at (1,1), move left to (0,1) — border wall
        assert_eq!(state.players[0].pos, (1, 1));

        let next = state.advance(&BomberAction::Left, 0);
        assert_eq!(next.players[0].pos, (1, 1)); // unchanged, blocked
    }

    #[test]
    fn advance_blocked_by_pillar() {
        let mut state = empty_state();
        // Player 0 at (1,1), move down to (1,2) is floor, then (1,2)→(2,2) is pillar
        // Let's test: from (1,1), move right to (2,1), then down to (2,2) — pillar
        state.players[0].pos = (1, 2);

        let next = state.advance(&BomberAction::Right, 0);
        // (2,2) is a pillar (even x, even y) → blocked
        assert_eq!(next.players[0].pos, (1, 2));
    }

    #[test]
    fn advance_blocked_by_bomb() {
        let mut state = empty_state();
        state.players[0].pos = (3, 1);
        state.bombs.push(BombSnapshot {
            pos: (4, 1),
            fuse: 3,
            range: 2,
            owner: 1,
        });

        let next = state.advance(&BomberAction::Right, 0);
        // (4,1) has a bomb → blocked
        assert_eq!(next.players[0].pos, (3, 1));
    }

    // ── Bomb Placement Tests ───────────────────────────────────

    #[test]
    fn advance_places_bomb() {
        let state = empty_state();

        let next = state.advance(&BomberAction::Bomb, 0);
        assert_eq!(next.bombs.len(), 1);
        assert_eq!(next.bombs[0].pos, (1, 1));
        assert_eq!(next.bombs[0].fuse, BOMB_FUSE_TICKS);
        assert_eq!(next.bombs[0].owner, 0);
        assert_eq!(next.players[0].active_bombs, 1);
    }

    #[test]
    fn bomb_placement_at_capacity_rejected() {
        let mut state = empty_state();
        state.players[0].active_bombs = state.players[0].max_bombs;

        let next = state.advance(&BomberAction::Bomb, 0);
        assert!(next.bombs.is_empty());
    }

    #[test]
    fn bomb_placement_duplicate_position_rejected() {
        let mut state = empty_state();
        state.bombs.push(BombSnapshot {
            pos: (1, 1),
            fuse: 2,
            range: 2,
            owner: 0,
        });
        state.players[0].active_bombs = 1;

        let next = state.advance(&BomberAction::Bomb, 0);
        // Should not place another bomb at (1,1)
        assert_eq!(next.bombs.len(), 1);
    }

    // ── Explosion Tests ────────────────────────────────────────

    #[test]
    fn bomb_explodes_after_fuse_ticks() {
        let mut state = empty_state();
        state.players[0].pos = (5, 1); // Move away from bomb
        state.bombs.push(BombSnapshot {
            pos: (1, 1),
            fuse: 1,
            range: 2,
            owner: 0,
        });
        state.players[0].active_bombs = 1;

        // Tick: fuse 1→0, bomb explodes
        let next = state.advance(&BomberAction::Wait, 0);
        assert!(next.bombs.is_empty()); // Bomb gone
        assert_eq!(next.players[0].active_bombs, 0); // Active decremented
    }

    #[test]
    fn explosion_kills_player() {
        let mut state = empty_state();
        // Player 1 standing at (3,1), bomb at (1,1) with range 3
        state.players[1].pos = (3, 1);
        state.bombs.push(BombSnapshot {
            pos: (1, 1),
            fuse: 1,
            range: 3,
            owner: 0,
        });

        let next = state.advance(&BomberAction::Wait, 0);
        assert!(!next.players[1].alive); // Killed by blast
    }

    #[test]
    fn explosion_stopped_by_wall() {
        let mut state = empty_state();
        // Player 1 at (3,3), bomb at (1,1) with range 5
        // Pillar at (2,2) blocks diagonal — but blast goes cardinally
        // Bomb at (1,1): blast right goes (2,1)(3,1)(4,1) — no pillar in that row
        // Player at (3,3): bomb blast down goes (1,2)(1,3) — player at (3,3) is NOT in same row/col
        state.players[1].pos = (3, 3);
        state.bombs.push(BombSnapshot {
            pos: (1, 1),
            fuse: 1,
            range: 5,
            owner: 0,
        });

        let next = state.advance(&BomberAction::Wait, 0);
        assert!(next.players[1].alive); // Different row and column, safe
    }

    #[test]
    fn explosion_destroys_wall() {
        // Create a custom state with a destructible wall
        let grid = ArenaGrid::generate(999);
        let mut state = BomberState::from_grid(&grid);

        // Place a bomb near a destructible wall
        // Force player 0 away, place bomb with fuse=1
        state.players[0].pos = (5, 5);
        state.bombs.push(BombSnapshot {
            pos: (1, 1),
            fuse: 1,
            range: 3,
            owner: 0,
        });

        // Check if there's a destructible wall in blast range
        let has_destructible =
            (1..=4).any(|x| matches!(state.get_cell(x, 1), Cell::DestructibleWall));

        let next = state.advance(&BomberAction::Wait, 0);
        if has_destructible {
            // At least one wall should be destroyed (now Floor)
            let destroyed = (1..=4).any(|x| {
                matches!(state.get_cell(x, 1), Cell::DestructibleWall)
                    && matches!(next.get_cell(x, 1), Cell::Floor)
            });
            assert!(
                destroyed,
                "at least one destructible wall should be destroyed"
            );
        }
    }

    #[test]
    fn chain_explosion() {
        let mut state = empty_state();
        state.players[0].pos = (9, 1); // Far away

        // Bomb A at (1,1) fuse=1, Bomb B at (3,1) fuse=4
        // When A explodes, blast reaches (3,1) and triggers B
        state.bombs.push(BombSnapshot {
            pos: (1, 1),
            fuse: 1,
            range: 3,
            owner: 0,
        });
        state.bombs.push(BombSnapshot {
            pos: (3, 1),
            fuse: 4,
            range: 2,
            owner: 0,
        });
        state.players[0].active_bombs = 2;

        let next = state.advance(&BomberAction::Wait, 0);
        // Both bombs should be gone (chain reaction)
        assert!(next.bombs.is_empty());
        assert_eq!(next.players[0].active_bombs, 0);
    }

    #[test]
    fn explosion_reveals_powerup() {
        let grid = ArenaGrid::fixed(
            "#############\
             \n#...........#\
             \n#.#.#.#.#.#.#\
             \n#...........#\
             \n#.#.#.#.#.#.#\
             \n#...........#\
             \n#.#.#.#.#.#.#\
             \n#...........#\
             \n#.#.#.#.#.#.#\
             \n#...........#\
             \n#.#.#.#.#.#.#\
             \n#...........#\
             \n#############",
        )
        .unwrap();
        let mut state = BomberState::from_grid(&grid);

        // Manually place a PowerUpHidden wall in blast range
        state.cells[ARENA_W + 3] = Cell::PowerUpHidden(PowerUpKind::BombUp);
        state.players[0].pos = (5, 1);
        state.bombs.push(BombSnapshot {
            pos: (1, 1),
            fuse: 1,
            range: 3,
            owner: 0,
        });

        let next = state.advance(&BomberAction::Wait, 0);
        // Wall destroyed, power-up revealed
        assert_eq!(next.cells[ARENA_W + 3], Cell::Floor);
        assert_eq!(next.revealed_powerups.len(), 1);
        assert_eq!(next.revealed_powerups[0], ((3, 1), PowerUpKind::BombUp));
    }

    // ── Power-up Collection Tests ──────────────────────────────

    #[test]
    fn powerup_collected_by_player() {
        let mut state = empty_state();
        state.players[0].pos = (3, 1);
        state.revealed_powerups.push(((3, 1), PowerUpKind::BombUp));

        let next = state.advance(&BomberAction::Wait, 0);
        assert!(next.revealed_powerups.is_empty());
        assert_eq!(next.players[0].max_bombs, DEFAULT_MAX_BOMBS + 1);
    }

    #[test]
    fn fire_up_increases_range() {
        let mut state = empty_state();
        state.players[0].pos = (3, 1);
        state.revealed_powerups.push(((3, 1), PowerUpKind::FireUp));

        let next = state.advance(&BomberAction::Wait, 0);
        assert_eq!(next.players[0].blast_range, DEFAULT_BLAST_RANGE + 1);
    }

    // ── Available Actions Tests ────────────────────────────────

    #[test]
    fn available_actions_filters_walls() {
        let state = empty_state();
        // Player 0 at (1,1): left is border wall, up is border wall
        let actions = state.available_actions(0);

        // Should not include Left (wall at 0,1) or Up (wall at 1,0)
        assert!(!actions.contains(&BomberAction::Left));
        assert!(!actions.contains(&BomberAction::Up));
        // Should include Right and Down (floor cells)
        assert!(actions.contains(&BomberAction::Right));
        assert!(actions.contains(&BomberAction::Down));
        // Should include Bomb and Wait
        assert!(actions.contains(&BomberAction::Bomb));
        assert!(actions.contains(&BomberAction::Wait));
    }

    #[test]
    fn available_actions_includes_wait_always() {
        let state = empty_state();
        for pid in 0..4u8 {
            let actions = state.available_actions(pid);
            assert!(
                actions.contains(&BomberAction::Wait),
                "player {pid} should have Wait"
            );
        }
    }

    // ── Heuristic Tests ────────────────────────────────────────

    #[test]
    fn heuristic_dead_negative() {
        let grid = ArenaGrid::generate(42);
        let mut state = BomberState::from_grid(&grid);
        state.players[0].alive = false;
        let h = BomberHeuristic;

        let value = h.evaluate(&state, 0);
        assert!((value - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn heuristic_safe_positive() {
        let state = empty_state();
        let h = BomberHeuristic;

        let value = h.evaluate(&state, 0);
        // Safe position: +2.0 base, plus escape routes, minus center dist
        assert!(
            value > 0.0,
            "safe alive player should have positive heuristic, got {value}"
        );
    }

    #[test]
    fn heuristic_blast_zone_penalty() {
        let mut state = empty_state();
        state.players[0].pos = (3, 1);
        // Place a bomb that threatens player 0
        state.bombs.push(BombSnapshot {
            pos: (3, 1),
            fuse: 2,
            range: 2,
            owner: 1,
        });

        let h = BomberHeuristic;
        let safe_value = h.evaluate(&empty_state(), 0);
        let danger_value = h.evaluate(&state, 0);

        assert!(
            danger_value < safe_value,
            "blast zone should be penalized: safe={safe_value}, danger={danger_value}"
        );
    }

    #[test]
    fn heuristic_trapped_near_death() {
        let mut state = empty_state();
        // Player surrounded by walls with bomb on top
        state.players[0].pos = (1, 1);
        state.bombs.push(BombSnapshot {
            pos: (1, 1),
            fuse: 2,
            range: 2,
            owner: 1,
        });
        // Block all exits
        state.cells[ARENA_W + 2] = Cell::DestructibleWall;
        state.cells[2 * ARENA_W + 1] = Cell::DestructibleWall;

        let h = BomberHeuristic;
        let value = h.evaluate(&state, 0);
        assert!(
            value < -0.5,
            "trapped player should have very low heuristic, got {value}"
        );
    }

    // ── Advance Purity Tests ───────────────────────────────────

    #[test]
    fn advance_increments_tick() {
        let state = empty_state();
        let next = state.advance(&BomberAction::Wait, 0);
        assert_eq!(next.tick(), 1);
        assert_eq!(state.tick(), 0); // original unchanged (pure)
    }

    #[test]
    fn advance_does_not_mutate_original() {
        let state = empty_state();
        let original_tick = state.tick;
        let original_pos = state.players[0].pos;
        let original_bombs = state.bombs.len();

        let _next = state.advance(&BomberAction::Right, 0);

        assert_eq!(state.tick, original_tick);
        assert_eq!(state.players[0].pos, original_pos);
        assert_eq!(state.bombs.len(), original_bombs);
    }

    // ── Is Walkable Tests ──────────────────────────────────────

    #[test]
    fn is_walkable_checks_bounds() {
        let state = empty_state();
        assert!(!state.is_walkable(-1, 0));
        assert!(!state.is_walkable(0, -1));
        assert!(!state.is_walkable(13, 0));
        assert!(!state.is_walkable(0, 13));
    }

    #[test]
    fn is_walkable_checks_walls() {
        let state = empty_state();
        // Border walls
        assert!(!state.is_walkable(0, 0));
        assert!(!state.is_walkable(12, 12));
        // Pillars
        assert!(!state.is_walkable(2, 2));
    }

    #[test]
    fn is_walkable_floor_cells() {
        let state = empty_state();
        // Floor cells in empty arena
        assert!(state.is_walkable(1, 1));
        assert!(state.is_walkable(3, 1));
        assert!(state.is_walkable(1, 3));
    }
}
