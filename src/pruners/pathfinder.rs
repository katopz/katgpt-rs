//! A* Pathfinder for grid-based tactical puzzles.
//!
//! Provides stateless pathfinding functions that operate on a grid with walls
//! and dynamically blocked positions (e.g., live monsters). Used by the
//! hierarchical AI as the **tactical layer** — computing paths between
//! strategic targets chosen by the DDTree.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

// ── Direction Constants ────────────────────────────────────────
// Matches TacticalPruner action encoding: 0=Up, 1=Down, 2=Left, 3=Right

const DIRS: [(isize, isize); 5] = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)];

/// Returns the direction delta for an action index (0-3).
#[inline]
pub fn dir_delta(action: usize) -> (isize, isize) {
    DIRS[action.min(3)]
}

/// Flat index helper: `(r, c)` → `r * cols + c`.
#[inline]
fn flat_index(r: usize, c: usize, cols: usize) -> usize {
    r * cols + c
}

/// Returns the action name for display.
pub fn action_name(action: usize) -> &'static str {
    match action {
        0 => "↑ Up",
        1 => "↓ Down",
        2 => "← Left",
        3 => "→ Right",
        _ => "???",
    }
}

// ── A* Node ────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Node {
    pos: (usize, usize),
    g: u32, // cost from start
    f: u32, // g + heuristic
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower f is better
        other.f.cmp(&self.f).then_with(|| other.g.cmp(&self.g))
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── Grid Helpers ───────────────────────────────────────────────

/// Check if a position is within grid bounds and not a wall.
#[inline]
pub fn is_passable(grid: &[Vec<char>], r: usize, c: usize) -> bool {
    r < grid.len() && c < grid[r].len() && grid[r][c] != '#'
}

/// Returns the terrain cost of stepping onto tile `(r, c)`.
///
/// Terrain costs: `.` = 1 (grass), `~` = 2 (sand), `w` = 3 (water).
/// Walls should never reach here (filtered by `is_passable`).
#[inline]
pub fn terrain_cost(grid: &[Vec<char>], r: usize, c: usize) -> u32 {
    match grid[r][c] {
        '~' => 2,
        'w' => 3,
        _ => 1,
    }
}

/// Manhattan distance heuristic for A*.
#[inline]
pub fn manhattan(a: (usize, usize), b: (usize, usize)) -> u32 {
    (a.0 as isize - b.0 as isize).unsigned_abs() as u32
        + (a.1 as isize - b.1 as isize).unsigned_abs() as u32
}

// ── Core A* ────────────────────────────────────────────────────

/// A* pathfinding on a grid.
///
/// Returns the shortest path as a list of action indices (0=Up, 1=Down,
/// 2=Left, 3=Right), or `None` if no path exists.
///
/// `blocked` positions are treated as impassable (e.g., live monsters).
pub fn find_path(
    grid: &[Vec<char>],
    from: (usize, usize),
    to: (usize, usize),
    blocked: &HashSet<(usize, usize)>,
) -> Option<Vec<usize>> {
    if from == to {
        return Some(Vec::new());
    }
    if !is_passable(grid, to.0, to.1) || blocked.contains(&to) {
        return None;
    }

    let rows = grid.len();
    let cols = grid.first().map_or(0, |r| r.len());
    if cols == 0 {
        return None;
    }
    let area = rows * cols;
    let from_flat = flat_index(from.0, from.1, cols);
    let to_flat = flat_index(to.0, to.1, cols);

    // Skip HashSet hashing when blocked is empty (common case).
    let has_blocked = !blocked.is_empty();

    let mut open = BinaryHeap::with_capacity(area);
    // Flat visited array — O(1) direct index, far better cache locality than HashSet<(usize, usize)>.
    let mut visited = vec![false; area];
    // Flat came_from: (prev_flat_index, action). Only accessed for visited cells during reconstruction.
    let mut came_from: Vec<(usize, u8)> = vec![(0, 0); area];

    open.push(Node {
        pos: from,
        g: 0,
        f: manhattan(from, to),
    });
    visited[from_flat] = true;

    while let Some(current) = open.pop() {
        if current.pos == to {
            // Reconstruct path from flat came_from chain.
            let mut path = Vec::new();
            let mut flat = to_flat;
            while flat != from_flat {
                let (prev_flat, action) = came_from[flat];
                path.push(action as usize);
                flat = prev_flat;
            }
            path.reverse();
            return Some(path);
        }

        let cur_flat = flat_index(current.pos.0, current.pos.1, cols);

        for (action, &(dr, dc)) in DIRS.iter().enumerate().take(4) {
            let nr = current.pos.0 as isize + dr;
            let nc = current.pos.1 as isize + dc;

            if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                continue;
            }
            let nr = nr as usize;
            let nc = nc as usize;
            let next_flat = flat_index(nr, nc, cols);

            if visited[next_flat] {
                continue;
            }
            // Bounds-safe grid access: is_passable handles jagged rows.
            if !is_passable(grid, nr, nc) {
                continue;
            }
            if has_blocked && blocked.contains(&(nr, nc)) {
                continue;
            }

            visited[next_flat] = true;
            let g = current.g + terrain_cost(grid, nr, nc);
            let f = g + manhattan((nr, nc), to);
            came_from[next_flat] = (cur_flat, action as u8);
            open.push(Node { pos: (nr, nc), g, f });
        }
    }

    None // No path found
}

/// A* distance only — faster than `find_path` because no path reconstruction.
///
/// Returns the shortest distance in steps, or `None` if unreachable.
pub fn find_distance(
    grid: &[Vec<char>],
    from: (usize, usize),
    to: (usize, usize),
    blocked: &HashSet<(usize, usize)>,
) -> Option<u32> {
    if from == to {
        return Some(0);
    }
    if !is_passable(grid, to.0, to.1) || blocked.contains(&to) {
        return None;
    }

    let rows = grid.len();
    let cols = grid.first().map_or(0, |r| r.len());
    if cols == 0 {
        return None;
    }
    let area = rows * cols;

    // Skip HashSet hashing when blocked is empty (common case).
    let has_blocked = !blocked.is_empty();

    let mut open = BinaryHeap::with_capacity(area);
    let mut visited = vec![false; area];

    open.push(Node {
        pos: from,
        g: 0,
        f: manhattan(from, to),
    });
    visited[flat_index(from.0, from.1, cols)] = true;

    while let Some(current) = open.pop() {
        if current.pos == to {
            return Some(current.g);
        }

        for &(dr, dc) in DIRS.iter().take(4) {
            let nr = current.pos.0 as isize + dr;
            let nc = current.pos.1 as isize + dc;

            if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                continue;
            }
            let nr = nr as usize;
            let nc = nc as usize;
            let next_flat = flat_index(nr, nc, cols);

            if visited[next_flat] {
                continue;
            }
            if !is_passable(grid, nr, nc) {
                continue;
            }
            if has_blocked && blocked.contains(&(nr, nc)) {
                continue;
            }

            visited[next_flat] = true;
            let g = current.g + terrain_cost(grid, nr, nc);
            let f = g + manhattan((nr, nc), to);
            open.push(Node { pos: (nr, nc), g, f });
        }
    }

    None
}

/// BFS flood fill — returns a flat visited bitmap plus column count.
///
/// Index by `r * cols + c` where `cols` is the returned column count.
/// This is the cache-friendly core used by both `reachable_positions` and
/// internal callers (e.g., map generation) that don't need a `HashSet`.
#[inline]
pub(crate) fn reachable_flat(
    grid: &[Vec<char>],
    from: (usize, usize),
    blocked: &HashSet<(usize, usize)>,
) -> (Vec<bool>, usize) {
    let rows = grid.len();
    let cols = grid.first().map_or(0, |r| r.len());
    let area = rows * cols;

    let has_blocked = !blocked.is_empty();
    let mut visited = if cols > 0 {
        vec![false; area]
    } else {
        return (Vec::new(), 0);
    };
    let mut queue = std::collections::VecDeque::with_capacity(area);

    queue.push_back(from);
    visited[flat_index(from.0, from.1, cols)] = true;

    while let Some(pos) = queue.pop_front() {
        for &(dr, dc) in DIRS.iter().take(4) {
            let nr = pos.0 as isize + dr;
            let nc = pos.1 as isize + dc;

            if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                continue;
            }
            let nr = nr as usize;
            let nc = nc as usize;
            let next_flat = flat_index(nr, nc, cols);

            if visited[next_flat] {
                continue;
            }
            if !is_passable(grid, nr, nc) {
                continue;
            }
            if has_blocked && blocked.contains(&(nr, nc)) {
                continue;
            }

            visited[next_flat] = true;
            queue.push_back((nr, nc));
        }
    }

    (visited, cols)
}

/// BFS flood fill — returns all positions reachable from `from`.
///
/// Useful for cost evaluation: which targets are actually reachable.
pub fn reachable_positions(
    grid: &[Vec<char>],
    from: (usize, usize),
    blocked: &HashSet<(usize, usize)>,
) -> HashSet<(usize, usize)> {
    let rows = grid.len();
    let (visited, cols) = reachable_flat(grid, from, blocked);

    // Collect reachable positions — sequential scan is cache-friendly.
    let mut count = 0;
    for &v in visited.iter() {
        if v {
            count += 1;
        }
    }
    let mut result = HashSet::with_capacity(count);
    for r in 0..rows {
        let base = r * cols;
        for c in 0..cols {
            if visited[base + c] {
                result.insert((r, c));
            }
        }
    }
    result
}

// ── Target System ──────────────────────────────────────────────

/// A strategic target that the DDTree chooses between.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    /// Kill the monster at index `i`.
    Monster(usize),
    /// Collect the treasure at index `j`.
    Treasure(usize),
    /// Reach the goal/exit.
    Goal,
}

impl Target {
    /// Returns the grid position of this target from the pruner's data.
    pub fn pos(
        &self,
        monsters: &[(usize, usize)],
        treasures: &[(usize, usize)],
        goal: (usize, usize),
    ) -> (usize, usize) {
        match self {
            Target::Monster(i) => monsters[*i],
            Target::Treasure(j) => treasures[*j],
            Target::Goal => goal,
        }
    }
}

/// Enumerates all strategic targets from the map data.
///
/// Order: monsters first, then treasures, then goal.
/// Token indices map directly: `targets[token_idx]`.
pub fn enumerate_targets(num_monsters: usize, num_treasures: usize) -> Vec<Target> {
    let mut targets = Vec::with_capacity(num_monsters + num_treasures + 1);
    for i in 0..num_monsters {
        targets.push(Target::Monster(i));
    }
    for j in 0..num_treasures {
        targets.push(Target::Treasure(j));
    }
    targets.push(Target::Goal);
    targets
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_grid() -> Vec<Vec<char>> {
        let s = "\
            . . .\n\
            . # .\n\
            . . .";
        s.lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|c| c.chars().next().unwrap())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_find_path_straight() {
        let grid = test_grid();
        let blocked = HashSet::new();
        let path = find_path(&grid, (0, 0), (0, 2), &blocked).unwrap();
        assert_eq!(path, vec![3, 3]); // Right, Right
    }

    #[test]
    fn test_find_path_around_wall() {
        let grid = test_grid();
        let blocked = HashSet::new();
        let path = find_path(&grid, (0, 1), (2, 1), &blocked).unwrap();
        // Wall at (1,1), must go around: either left or right
        assert!(path.len() > 2, "Path must go around wall, got {path:?}");
        // Verify path actually reaches target
        let mut pos = (0usize, 1usize);
        for &action in &path {
            let (dr, dc) = DIRS[action];
            pos = (
                (pos.0 as isize + dr) as usize,
                (pos.1 as isize + dc) as usize,
            );
        }
        assert_eq!(pos, (2, 1));
    }

    #[test]
    fn test_find_path_blocked_unreachable() {
        let grid = test_grid();
        let mut blocked = HashSet::new();
        blocked.insert((0, 1));
        blocked.insert((1, 0));
        blocked.insert((1, 2));
        // (2,1) reachable from (2,0) or (2,2), but not from (0,0) with those blocked
        // Actually it IS reachable via (0,0)→(0,1) blocked... let me fix
        // From (0,0): can go down to (1,0) blocked, right to (0,1) blocked
        // So (0,0) is stuck
        let path = find_path(&grid, (0, 0), (2, 1), &blocked);
        assert!(path.is_none(), "Should be unreachable");
    }

    #[test]
    fn test_find_path_same_pos() {
        let grid = test_grid();
        let blocked = HashSet::new();
        let path = find_path(&grid, (1, 1), (1, 1), &blocked).unwrap();
        assert!(path.is_empty());
    }

    #[test]
    fn test_find_distance_matches_path_length() {
        let grid = test_grid();
        let blocked = HashSet::new();
        let path = find_path(&grid, (0, 0), (2, 2), &blocked).unwrap();
        let dist = find_distance(&grid, (0, 0), (2, 2), &blocked).unwrap();
        // Uniform terrain (all grass=1): path length == distance
        assert_eq!(path.len() as u32, dist);
    }

    #[test]
    fn test_terrain_cost_affects_path() {
        // Grid with a sand shortcut vs grass detour
        let s = "\
            . ~ .\n\
            . . .\n\
            . . .";
        let grid: Vec<Vec<char>> = s
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|c| c.chars().next().unwrap())
                    .collect()
            })
            .collect();
        let blocked = HashSet::new();

        // Direct path through sand: cost = 1 (start) + 2 (sand) = 3, length = 2
        // Detour via grass: cost = 1 + 1 + 1 = 3, length = 3
        // Both have same cost, but A* should find the shorter direct path
        let path = find_path(&grid, (0, 0), (0, 2), &blocked).unwrap();
        assert_eq!(path.len(), 2); // Right, Right (through sand)

        // Distance should be 3 (1 grass + 2 sand)
        let dist = find_distance(&grid, (0, 0), (0, 2), &blocked).unwrap();
        assert_eq!(dist, 3);
    }

    #[test]
    fn test_reachable_positions() {
        let grid = test_grid();
        let blocked = HashSet::new();
        let reachable = reachable_positions(&grid, (0, 0), &blocked);
        // All 8 floor tiles (center is wall)
        assert_eq!(reachable.len(), 8);
        assert!(reachable.contains(&(0, 0)));
        assert!(reachable.contains(&(2, 2)));
        assert!(!reachable.contains(&(1, 1))); // wall
    }

    #[test]
    fn test_enumerate_targets() {
        let targets = enumerate_targets(3, 2);
        assert_eq!(targets.len(), 6); // 3 monsters + 2 treasures + 1 goal
        assert_eq!(targets[0], Target::Monster(0));
        assert_eq!(targets[2], Target::Monster(2));
        assert_eq!(targets[3], Target::Treasure(0));
        assert_eq!(targets[4], Target::Treasure(1));
        assert_eq!(targets[5], Target::Goal);
    }
}
