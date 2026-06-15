//! Grid-based Tactical Puzzle — Constraint Pruner
//!
//! A generic deterministic rules engine for grid-based puzzle games with:
//! - Wall collisions and grid bounds
//! - Monsters that can be killed and drop items
//! - Locked treasures requiring inventory items
//! - Goal/exit locked until all treasures collected
//! - Inventory system (configurable max)
//!
//! Used with DDTree as a pure A* / Best-First search engine.
//! The pruner eliminates invalid state transitions, keeping the branching
//! factor small enough for exhaustive search within the DDTree budget.

use crate::speculative::types::ConstraintPruner;

/// Branch-free direction deltas: 0=Up(-1,0), 1=Down(1,0), 2=Left(0,-1), 3=Right(0,1).
const DIR_DELTAS: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// Sentinel value indicating no entity at a tile.
const NO_ENTITY: i32 = -1;

/// Represents the deterministic state of a grid-based tactical puzzle.
///
/// Fields ordered by alignment (u64/u32 → u8) to minimize padding.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GameState {
    pub total_cost: u32,
    pub killed_monsters: u32,
    pub collected_treasures: u32,
    pub dropped_items: u32,
    pub r: usize,
    pub c: usize,
    pub inventory: u8,
}

impl GameState {
    /// Returns a compact summary string of the state for display.
    pub fn summary(&self) -> String {
        format!(
            "pos=({}, {}) inv={} cost={} killed={:02b} treasures={:02b} dropped={:02b}",
            self.r,
            self.c,
            self.inventory,
            self.total_cost,
            self.killed_monsters,
            self.collected_treasures,
            self.dropped_items
        )
    }
}

/// Generic grid-based tactical puzzle constraint pruner.
///
/// Enforces physical and combat rules for games with grid movement,
/// monsters, locked treasures, inventory items, and goal conditions.
pub struct TacticalPruner {
    pub grid: Vec<Vec<char>>,
    pub start_r: usize,
    pub start_c: usize,
    pub monsters: Vec<(usize, usize)>,
    pub treasures: Vec<(usize, usize)>,
    pub goal: (usize, usize),
    // ── Precomputed hot-path lookups (built once in `new`) ─────────
    /// Grid column count. Assumes rectangular rows (same as existing bounds checks).
    cols: usize,
    /// Bitmask of all treasures: `(1 << treasures.len()) - 1`. Precomputed for goal-lock check.
    all_treasures_mask: u32,
    /// Flat monster lookup: `monster_at[r * cols + c]` → monster index or `NO_ENTITY`.
    monster_at: Vec<i32>,
    /// Flat treasure lookup: `treasure_at[r * cols + c]` → treasure index or `NO_ENTITY`.
    treasure_at: Vec<i32>,
}

impl TacticalPruner {
    /// Parse a grid string into the internal representation.
    ///
    /// Map symbols:
    /// - `B` or `S` = Start (player)
    /// - `M` = Monster
    /// - `T` = Treasure (locked, needs item)
    /// - `G` = Goal/Exit
    /// - `X` = Monster + Treasure on same tile
    /// - `#` = Wall
    /// - `.` = Floor
    pub fn new(map_str: &str) -> Self {
        let mut grid = Vec::new();
        let mut start_r = 0;
        let mut start_c = 0;
        let mut monsters = Vec::new();
        let mut treasures = Vec::new();
        let mut goal = (0, 0);

        for (r, line) in map_str.lines().enumerate() {
            let mut row = Vec::new();
            for (c, ch) in line.split_whitespace().enumerate() {
                let char_val = ch.chars().next().unwrap();
                match char_val {
                    'B' | 'S' => {
                        start_r = r;
                        start_c = c;
                        row.push('.');
                    }
                    'M' => {
                        monsters.push((r, c));
                        row.push('.');
                    }
                    'T' => {
                        treasures.push((r, c));
                        row.push('.');
                    }
                    'G' => {
                        goal = (r, c);
                        row.push('.');
                    }
                    'X' => {
                        // Monster and Treasure on same tile
                        monsters.push((r, c));
                        treasures.push((r, c));
                        row.push('.');
                    }
                    _ => row.push(char_val),
                }
            }
            grid.push(row);
        }

        // ── Precompute flat entity lookups (O(1) per tile in `apply_action`) ──
        let cols = grid.first().map_or(0, |r| r.len());
        let area = grid.len() * cols;
        let all_treasures_mask = if treasures.is_empty() {
            0
        } else {
            ((1u64 << treasures.len()) - 1) as u32
        };
        let mut monster_at = vec![NO_ENTITY; area];
        let mut treasure_at = vec![NO_ENTITY; area];
        for (i, &(mr, mc)) in monsters.iter().enumerate() {
            let flat = mr * cols + mc;
            if flat < area {
                monster_at[flat] = i as i32;
            }
        }
        for (i, &(tr, tc)) in treasures.iter().enumerate() {
            let flat = tr * cols + tc;
            if flat < area {
                treasure_at[flat] = i as i32;
            }
        }

        Self {
            grid,
            start_r,
            start_c,
            monsters,
            treasures,
            goal,
            cols,
            all_treasures_mask,
            monster_at,
            treasure_at,
        }
    }

    /// The starting state before any actions are taken.
    pub fn initial_state(&self) -> GameState {
        GameState {
            r: self.start_r,
            c: self.start_c,
            inventory: 0,
            killed_monsters: 0,
            collected_treasures: 0,
            dropped_items: 0,
            total_cost: 0,
        }
    }

    /// Returns the terrain cost of stepping onto tile `(r, c)`.
    ///
    /// Terrain costs:
    /// - `.` = Grass (cost 1)
    /// - `~` = Sand (cost 2)
    /// - `w` = Water (cost 3)
    /// - `#` = Wall (impassable — cost is irrelevant, blocked before reaching here)
    pub fn terrain_cost(&self, r: usize, c: usize) -> u32 {
        match self.grid[r][c] {
            '~' => 2,
            'w' => 3,
            _ => 1, // grass, floor, or any other passable tile
        }
    }

    /// Applies a single action (0:Up, 1:Down, 2:Left, 3:Right, 4:Attack) to the state.
    /// Returns `None` if the move is physically or logically impossible.
    pub fn apply_action(&self, state: &GameState, action: usize) -> Option<GameState> {
        let mut next = state.clone();

        match action {
            0..=3 => {
                // MOVE ACTION — branch-free direction lookup
                let (dr, dc) = DIR_DELTAS[action];

                let nr = next.r as isize + dr;
                let nc = next.c as isize + dc;

                // 1. Grid bounds (precomputed cols avoids grid[0].len() per call)
                if nr < 0
                    || nc < 0
                    || nr >= self.grid.len() as isize
                    || nc >= self.cols as isize
                {
                    return None;
                }

                let nr = nr as usize;
                let nc = nc as usize;
                let flat = nr * self.cols + nc;

                // 2. Wall collisions
                if self.grid[nr][nc] == '#' {
                    return None;
                }

                // 3. Goal validation (exit locked until all treasures collected)
                if (nr, nc) == self.goal
                    && next.collected_treasures != self.all_treasures_mask
                {
                    return None;
                }

                // 4. Check for a LIVE monster at the target tile — O(1) flat lookup
                let m_idx = self.monster_at[flat];
                let live_monster_here = m_idx != NO_ENTITY
                    && (next.killed_monsters & (1 << m_idx)) == 0;

                // 5. Treasure collection (locked without item) — O(1) flat lookup
                if !live_monster_here {
                    let t_idx = self.treasure_at[flat];
                    if t_idx != NO_ENTITY
                        && (next.collected_treasures & (1 << t_idx)) == 0
                    {
                        if next.inventory > 0 {
                            next.inventory -= 1;
                            next.collected_treasures |= 1 << t_idx;
                        } else {
                            // Cannot walk onto locked treasure without item
                            return None;
                        }
                    }
                }

                // Update coordinates
                next.r = nr;
                next.c = nc;

                // 6. Accumulate movement cost (terrain-dependent)
                next.total_cost += self.terrain_cost(nr, nc);

                // 7. Auto-pickup dropped items at new tile — O(1) flat lookup
                if m_idx != NO_ENTITY
                    && (next.dropped_items & (1 << m_idx)) != 0
                    && next.inventory < 2
                {
                    next.inventory += 1;
                    next.dropped_items &= !(1 << m_idx);
                }
            }
            4 => {
                // ATTACK ACTION — must be on a live monster's tile — O(1) flat lookup
                let flat = next.r * self.cols + next.c;
                let m_idx = self.monster_at[flat];

                if m_idx == NO_ENTITY || (next.killed_monsters & (1 << m_idx)) != 0 {
                    return None; // No live monster here to attack
                }

                next.killed_monsters |= 1 << m_idx;
                next.dropped_items |= 1 << m_idx;

                // Auto-pickup if inventory allows
                if next.inventory < 2 {
                    next.inventory += 1;
                    next.dropped_items &= !(1 << m_idx);
                }

                // Check for treasure underneath the killed monster — O(1) flat lookup
                let t_idx = self.treasure_at[flat];
                if t_idx != NO_ENTITY
                    && (next.collected_treasures & (1 << t_idx)) == 0
                    && next.inventory > 0
                {
                    next.inventory -= 1;
                    next.collected_treasures |= 1 << t_idx;
                }
            }
            _ => return None,
        }

        Some(next)
    }

    /// Replays a sequence of actions from the starting position.
    pub fn replay_state(&self, actions: &[usize]) -> Option<GameState> {
        let mut state = self.initial_state();
        for &action in actions {
            state = self.apply_action(&state, action)?;
        }
        Some(state)
    }

    /// Returns the action name for display.
    pub fn action_name(action: usize) -> &'static str {
        match action {
            0 => "Up",
            1 => "Down",
            2 => "Left",
            3 => "Right",
            4 => "Attack",
            _ => "?",
        }
    }
}

impl ConstraintPruner for TacticalPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        match self.replay_state(parent_tokens) {
            Some(state) => self.apply_action(&state, token_idx).is_some(),
            None => false,
        }
    }
}
