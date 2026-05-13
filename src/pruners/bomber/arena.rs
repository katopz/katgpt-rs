use bevy_ecs::prelude::Resource;

use super::{ARENA_H, ARENA_W, Cell, DESTRUCTIBLE_FILL, PowerUpKind, SPAWN_POSITIONS};

/// The 13×13 Bomberman arena grid.
///
/// Grid coordinates: `cells[y][x]` where (0,0) is top-left.
/// Standard Bomberman layout: fixed walls at even row/col intersections,
/// destructible walls randomly placed, 3×3 corners kept clear for spawns.
#[derive(Clone, Debug, Resource)]
pub struct ArenaGrid {
    /// Grid cells: `cells[y][x]`
    pub cells: Vec<Vec<Cell>>,
    pub width: usize,
    pub height: usize,
}

impl ArenaGrid {
    /// Generate a 13×13 arena grid from the given seed.
    #[allow(clippy::needless_range_loop)]
    pub fn generate(seed: u64) -> Self {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut cells = vec![vec![Cell::Floor; ARENA_W]; ARENA_H];

        // Border walls
        for y in 0..ARENA_H {
            for x in 0..ARENA_W {
                if x == 0 || x == ARENA_W - 1 || y == 0 || y == ARENA_H - 1 {
                    cells[y][x] = Cell::FixedWall;
                }
            }
        }

        // Interior pillars at even x, even y (0-indexed)
        for y in 2..ARENA_H - 1 {
            for x in 2..ARENA_W - 1 {
                if x % 2 == 0 && y % 2 == 0 {
                    cells[y][x] = Cell::FixedWall;
                }
            }
        }

        // Destructible walls + hidden power-ups (~40% fill, exclude spawn zones)
        for y in 1..ARENA_H - 1 {
            for x in 1..ARENA_W - 1 {
                if cells[y][x] != Cell::Floor || Self::is_in_spawn_zone(x, y) {
                    continue;
                }
                if rng.f32() < DESTRUCTIBLE_FILL {
                    cells[y][x] = Self::random_destructible(&mut rng);
                }
            }
        }

        Self {
            cells,
            width: ARENA_W,
            height: ARENA_H,
        }
    }

    /// Pick `DestructibleWall` or `PowerUpHidden` (20% power-up chance).
    fn random_destructible(rng: &mut fastrand::Rng) -> Cell {
        match rng.f32() < 0.2 {
            true => {
                let kind = match rng.u8(0..3) {
                    0 => PowerUpKind::BombUp,
                    1 => PowerUpKind::FireUp,
                    _ => PowerUpKind::SpeedUp,
                };
                Cell::PowerUpHidden(kind)
            }
            false => Cell::DestructibleWall,
        }
    }

    /// Check if (x, y) is within any player's 3×3 spawn safe zone.
    fn is_in_spawn_zone(x: usize, y: usize) -> bool {
        SPAWN_POSITIONS.iter().any(|&(sx, sy)| {
            (x as i32 - sx).unsigned_abs() <= 1 && (y as i32 - sy).unsigned_abs() <= 1
        })
    }

    /// Safe cell access. Returns `FixedWall` for out-of-bounds.
    pub fn get(&self, x: i32, y: i32) -> Cell {
        match self.is_in_bounds(x, y) {
            true => self.cells[y as usize][x as usize],
            false => Cell::FixedWall,
        }
    }

    /// Set cell at (x, y). No-op for out-of-bounds.
    pub fn set(&mut self, x: i32, y: i32, cell: Cell) {
        if !self.is_in_bounds(x, y) {
            return;
        }
        self.cells[y as usize][x as usize] = cell;
    }

    /// True if the cell is walkable (only `Floor`).
    pub fn is_walkable(&self, x: i32, y: i32) -> bool {
        matches!(self.get(x, y), Cell::Floor)
    }

    /// True if (x, y) is within grid bounds.
    pub fn is_in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && (x as usize) < self.width && y >= 0 && (y as usize) < self.height
    }

    /// Returns the spawn position for the given player (0..3).
    pub fn spawn_pos(&self, player_id: u8) -> (i32, i32) {
        SPAWN_POSITIONS[player_id as usize]
    }

    /// Clear spawn zones of destructible walls and power-ups for safe respawning.
    pub fn clear_for_respawn(&mut self) {
        for &(sx, sy) in &SPAWN_POSITIONS {
            for dy in -1_i32..=1 {
                for dx in -1_i32..=1 {
                    let (x, y) = (sx + dx, sy + dy);
                    match self.get(x, y) {
                        Cell::DestructibleWall | Cell::PowerUpHidden(_) => {
                            self.set(x, y, Cell::Floor);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Create arena from a 2D cell grid. Validates dimensions, border walls, and spawn zones.
    pub fn from_cells(cells: &[Vec<Cell>]) -> Result<Self, String> {
        // Validate dimensions
        if cells.len() != ARENA_H {
            return Err(format!("Expected {ARENA_H} rows, got {}", cells.len()));
        }
        for (y, row) in cells.iter().enumerate() {
            if row.len() != ARENA_W {
                return Err(format!(
                    "Row {y}: expected {ARENA_W} cols, got {}",
                    row.len()
                ));
            }
        }

        // Validate border walls
        for (y, row) in cells.iter().enumerate() {
            for (x, cell) in row.iter().enumerate() {
                let is_border = x == 0 || x == ARENA_W - 1 || y == 0 || y == ARENA_H - 1;
                if is_border && *cell != Cell::FixedWall {
                    return Err(format!("Border ({x},{y}) must be FixedWall"));
                }
            }
        }

        // Validate pillar positions (even x, even y, not border)
        for y in (2..ARENA_H - 1).step_by(2) {
            for x in (2..ARENA_W - 1).step_by(2) {
                if cells[y][x] != Cell::FixedWall {
                    return Err(format!("Pillar ({x},{y}) must be FixedWall"));
                }
            }
        }

        // Validate spawn zones — non-pillar, non-border cells must be Floor
        for &(sx, sy) in &SPAWN_POSITIONS {
            for dy in -1_i32..=1 {
                for dx in -1_i32..=1 {
                    let x = (sx + dx) as usize;
                    let y = (sy + dy) as usize;
                    let is_border = x == 0 || x == ARENA_W - 1 || y == 0 || y == ARENA_H - 1;
                    let is_pillar = x.is_multiple_of(2)
                        && y.is_multiple_of(2)
                        && (2..ARENA_W - 1).contains(&x)
                        && (2..ARENA_H - 1).contains(&y);
                    if is_border || is_pillar {
                        continue;
                    }
                    if cells[y][x] != Cell::Floor {
                        return Err(format!("Spawn zone ({x},{y}) must be Floor"));
                    }
                }
            }
        }

        Ok(Self {
            cells: cells.to_vec(),
            width: ARENA_W,
            height: ARENA_H,
        })
    }

    /// Create arena from a compact string template.
    /// `#` = FixedWall, `.` = Floor, `D` = DestructibleWall, `P` = PowerUpHidden
    /// Rows separated by `\n`
    pub fn fixed(template: &str) -> Result<Self, String> {
        let rows: Vec<&str> = template.split('\n').collect();
        if rows.len() != ARENA_H {
            return Err(format!("Expected {ARENA_H} rows, got {}", rows.len()));
        }

        let mut cells = Vec::with_capacity(ARENA_H);
        for (y, row) in rows.iter().enumerate() {
            if row.len() != ARENA_W {
                return Err(format!(
                    "Row {y}: expected {ARENA_W} cols, got {}",
                    row.len()
                ));
            }
            let mut row_cells = Vec::with_capacity(ARENA_W);
            for (x, ch) in row.chars().enumerate() {
                let cell = match ch {
                    '#' => Cell::FixedWall,
                    '.' => Cell::Floor,
                    'D' => Cell::DestructibleWall,
                    'P' => Cell::PowerUpHidden(PowerUpKind::BombUp),
                    other => return Err(format!("Row {y}, col {x}: invalid char '{other}'")),
                };
                row_cells.push(cell);
            }
            cells.push(row_cells);
        }

        Self::from_cells(&cells)
    }
}

// ── Preset Arena Templates ─────────────────────────────────────

/// Empty arena — all floors except borders and pillars.
pub const EMPTY_ARENA: &str = "\
#############\
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
\n#############";

/// Standard arena with ~40% destructible fill (mimics generate(seed) output).
pub const STANDARD_ARENA: &str = "\
#############\
\n#..DDD.DD...#\
\n#.#.#.#.#.#.#\
\n#.D.DD.DD.DD#\
\n#.#.#.#.#.#.#\
\n#.DD..DDD.D.#\
\n#.#.#.#.#.#.#\
\n#..D.DD.D.DD#\
\n#.#.#.#.#.#.#\
\n#.D.DD.DD.D.#\
\n#.#.#.#.#.#.#\
\n#...DD.DDD..#\
\n#############";

/// Pillar-heavy arena with extra fixed walls.
pub const PILLAR_HEAVY_ARENA: &str = "\
#############\
\n#...........#\
\n#.#.#.#.#.#.#\
\n#.#.......#.#\
\n#.#.#.#.#.#.#\
\n#...#...#...#\
\n#.#.#.#.#.#.#\
\n#.#.......#.#\
\n#.#.#.#.#.#.#\
\n#...#...#...#\
\n#.#.#.#.#.#.#\
\n#...........#\
\n#############";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_dimensions() {
        let grid = ArenaGrid::generate(42);
        assert_eq!(grid.width, 13);
        assert_eq!(grid.height, 13);
        assert_eq!(grid.cells.len(), 13);
        assert!(grid.cells.iter().all(|row| row.len() == 13));
    }

    #[test]
    fn test_border_walls() {
        let grid = ArenaGrid::generate(42);
        for y in 0..13 {
            for x in 0..13 {
                if x == 0 || x == 12 || y == 0 || y == 12 {
                    assert_eq!(grid.cells[y][x], Cell::FixedWall, "border ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn test_fixed_walls_pattern() {
        let grid = ArenaGrid::generate(42);
        for y in 2..11 {
            for x in 2..11 {
                if x % 2 == 0 && y % 2 == 0 {
                    assert_eq!(grid.cells[y][x], Cell::FixedWall, "pillar ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn test_corners_clear() {
        let grid = ArenaGrid::generate(123);
        for &(sx, sy) in &SPAWN_POSITIONS {
            for dy in -1_i32..=1 {
                for dx in -1_i32..=1 {
                    let (x, y) = (sx + dx, sy + dy);
                    if x < 1 || x > 11 || y < 1 || y > 11 {
                        continue;
                    }
                    match grid.cells[y as usize][x as usize] {
                        Cell::Floor | Cell::FixedWall => {}
                        other => panic!("spawn ({x},{y}) has {other:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn test_seed_reproducibility() {
        let a = ArenaGrid::generate(999);
        let b = ArenaGrid::generate(999);
        assert_eq!(a.cells, b.cells);
    }

    #[test]
    fn test_from_cells_valid() {
        let grid = ArenaGrid::fixed(EMPTY_ARENA).expect("empty arena should parse");
        assert_eq!(grid.width, 13);
        assert_eq!(grid.height, 13);
        // Verify specific cells
        assert_eq!(grid.cells[0][0], Cell::FixedWall, "top-left border");
        assert_eq!(grid.cells[1][1], Cell::Floor, "top-left spawn");
        assert_eq!(grid.cells[2][2], Cell::FixedWall, "first pillar");
        assert_eq!(grid.cells[6][6], Cell::FixedWall, "center pillar");
    }

    #[test]
    fn test_from_cells_bad_dimensions() {
        // Wrong number of rows
        let small = vec![vec![Cell::Floor; 13]; 10];
        let err = ArenaGrid::from_cells(&small).unwrap_err();
        assert!(err.contains("rows"), "Expected row error: {err}");

        // Wrong number of cols
        let narrow = vec![vec![Cell::Floor; 10]; 13];
        let err = ArenaGrid::from_cells(&narrow).unwrap_err();
        assert!(err.contains("cols"), "Expected col error: {err}");
    }

    #[test]
    fn test_from_cells_missing_border() {
        let mut cells = vec![vec![Cell::Floor; 13]; 13];
        // Set proper borders
        for y in 0..13 {
            for x in 0..13 {
                if x == 0 || x == 12 || y == 0 || y == 12 {
                    cells[y][x] = Cell::FixedWall;
                }
            }
        }
        // Set pillars
        for y in (2..12).step_by(2) {
            for x in (2..12).step_by(2) {
                cells[y][x] = Cell::FixedWall;
            }
        }
        // Break a border cell
        cells[0][5] = Cell::Floor;
        let err = ArenaGrid::from_cells(&cells).unwrap_err();
        assert!(err.contains("Border"), "Expected border error: {err}");
    }

    #[test]
    fn test_fixed_parsing() {
        let grid = ArenaGrid::fixed(EMPTY_ARENA).expect("should parse");
        // All border cells are FixedWall
        for y in 0..13 {
            assert_eq!(grid.cells[y][0], Cell::FixedWall, "left border y={y}");
            assert_eq!(grid.cells[y][12], Cell::FixedWall, "right border y={y}");
        }
        for x in 0..13 {
            assert_eq!(grid.cells[0][x], Cell::FixedWall, "top border x={x}");
            assert_eq!(grid.cells[12][x], Cell::FixedWall, "bottom border x={x}");
        }
        // Pillars at even (x, y) interior positions
        for y in (2..12).step_by(2) {
            for x in (2..12).step_by(2) {
                assert_eq!(grid.cells[y][x], Cell::FixedWall, "pillar ({x},{y})");
            }
        }
        // Non-pillar, non-border interior cells are Floor
        for y in 1..12 {
            for x in 1..12 {
                if x % 2 == 0 && y % 2 == 0 {
                    continue; // pillar
                }
                assert_eq!(grid.cells[y][x], Cell::Floor, "floor ({x},{y})");
            }
        }
    }

    #[test]
    fn test_fixed_roundtrip() {
        // STANDARD_ARENA: verify destructible count
        let grid = ArenaGrid::fixed(STANDARD_ARENA).expect("standard arena should parse");
        let d_count = grid
            .cells
            .iter()
            .flatten()
            .filter(|c| matches!(c, Cell::DestructibleWall))
            .count();
        assert_eq!(
            d_count, 35,
            "Expected 35 destructible walls in STANDARD_ARENA"
        );

        // Verify spawn zones are clear (Floor or FixedWall only)
        for &(sx, sy) in &SPAWN_POSITIONS {
            for dy in -1_i32..=1 {
                for dx in -1_i32..=1 {
                    let (x, y) = (sx + dx, sy + dy);
                    if x < 1 || x > 11 || y < 1 || y > 11 {
                        continue;
                    }
                    match grid.get(x, y) {
                        Cell::Floor | Cell::FixedWall => {}
                        other => panic!("spawn ({x},{y}) has {other:?}"),
                    }
                }
            }
        }

        // PILLAR_HEAVY_ARENA: verify extra fixed walls at non-pillar positions
        let grid = ArenaGrid::fixed(PILLAR_HEAVY_ARENA).expect("pillar-heavy should parse");
        assert_eq!(grid.cells[3][2], Cell::FixedWall, "extra wall at (2,3)");
        assert_eq!(grid.cells[3][10], Cell::FixedWall, "extra wall at (10,3)");
        assert_eq!(grid.cells[5][4], Cell::FixedWall, "extra wall at (4,5)");
        assert_eq!(grid.cells[5][8], Cell::FixedWall, "extra wall at (8,5)");
    }
}
