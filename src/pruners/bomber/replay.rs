//! Replay training data pipeline for Bomberman — serialize game state to JSONL
//! for downstream training / analysis.

use super::{ARENA_H, ARENA_W, ArenaGrid, Bomb, BombFuse, BombRange, Cell, GridPos, PowerUp};
#[cfg(test)]
use super::{BomberAction, PowerUpKind};
use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::{BufWriter, Write};
use std::path::Path;

// ── ReplaySample ───────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplaySample {
    pub board: Vec<u8>,
    pub player_pos: [u8; 2],
    pub player_id: u8,
    pub bombs: Vec<[u8; 4]>,
    pub powerups: Vec<[u8; 2]>,
    pub action: u8,
    pub quality: f32,
    pub tick: u32,
    pub round: u32,
    pub player_type: String,
    /// 0=safe, 1=adjacent to blast zone, 2=in blast zone.
    #[serde(default)]
    pub danger_level: u8,
    /// Manhattan distance to nearest opponent. 255 if none.
    #[serde(default)]
    pub nearest_opponent_dist: u8,
    /// Count of walkable adjacent cells.
    #[serde(default)]
    pub escape_routes: u8,
    /// Bomb type per bomb: 0=Timed, 1=Piercing, 2=Remote, 3=Landmine.
    /// Defaults to empty for backward compat (old replays → all Timed).
    #[serde(default)]
    pub bomb_types: Vec<u8>,
    /// Strategy template ID (0-7). Only set by template-using players (GZero/Rubric/SDAR/SDPG).
    /// 255 = not set (raw action player like Random/Greedy/Validator/HL).
    #[serde(default = "default_template_id")]
    pub template_id: u8,
}

fn default_template_id() -> u8 {
    255
}

impl ReplaySample {
    /// Create a new ReplaySample with enriched features computed from game state.
    #[allow(clippy::too_many_arguments)]
    pub fn new_enriched(
        board: Vec<u8>,
        player_pos: [u8; 2],
        player_id: u8,
        bombs: Vec<[u8; 4]>,
        powerups: Vec<[u8; 2]>,
        action: u8,
        quality: f32,
        tick: u32,
        round: u32,
        player_type: String,
        grid: &super::ArenaGrid,
        opponent_positions: &[(i32, i32)],
    ) -> Self {
        let px = player_pos[0] as i32;
        let py = player_pos[1] as i32;

        let danger_level = Self::compute_danger_level(grid, &bombs, px, py);
        let nearest_opponent_dist = match opponent_positions.is_empty() {
            true => 255,
            false => opponent_positions
                .iter()
                .map(|&(ox, oy)| (px - ox).unsigned_abs() + (py - oy).unsigned_abs())
                .min()
                .unwrap_or(255) as u8,
        };
        let escape_routes = Self::count_escape_routes(grid, px, py);
        let bomb_count = bombs.len();

        Self {
            board,
            player_pos,
            player_id,
            bombs,
            powerups,
            action,
            quality,
            tick,
            round,
            player_type,
            danger_level,
            nearest_opponent_dist,
            escape_routes,
            bomb_types: vec![0; bomb_count],
            template_id: 255,
        }
    }

    /// Check if position (x, y) is within any bomb's blast zone.
    fn is_in_blast_zone(grid: &super::ArenaGrid, bombs: &[[u8; 4]], x: i32, y: i32) -> bool {
        let directions: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

        for &[bx, by, range, _fuse] in bombs {
            // Bomb position itself is in blast zone
            if bx as i32 == x && by as i32 == y {
                return true;
            }

            for &(dx, dy) in &directions {
                for dist in 1..=range as i32 {
                    let cx = bx as i32 + dx * dist;
                    let cy = by as i32 + dy * dist;

                    if cx == x && cy == y {
                        return true;
                    }

                    // Stop propagation at walls
                    match grid.get(cx, cy) {
                        Cell::FixedWall | Cell::DestructibleWall | Cell::PowerUpHidden(_) => break,
                        Cell::Floor => {}
                    }
                }
            }
        }
        false
    }

    /// Compute danger level: 2=in blast zone, 1=adjacent to blast zone, 0=safe.
    fn compute_danger_level(grid: &super::ArenaGrid, bombs: &[[u8; 4]], px: i32, py: i32) -> u8 {
        if Self::is_in_blast_zone(grid, bombs, px, py) {
            return 2;
        }

        let adjacent: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
        for &(dx, dy) in &adjacent {
            if Self::is_in_blast_zone(grid, bombs, px + dx, py + dy) {
                return 1;
            }
        }

        0
    }

    /// Count walkable adjacent cells (escape routes).
    fn count_escape_routes(grid: &super::ArenaGrid, px: i32, py: i32) -> u8 {
        let adjacent: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
        let mut count = 0u8;
        for &(dx, dy) in &adjacent {
            if grid.is_walkable(px + dx, py + dy) {
                count += 1;
            }
        }
        count
    }

    /// Compute quality score from game outcome.
    ///
    /// - Death → 0.0, Survived → 0.5, Winner → 1.0
    /// - +0.05 per powerup (capped at +0.2)
    /// - +0.1  per kill     (capped at +0.3)
    pub fn quality(survived: bool, winner: bool, powerups: u32, kills: u32) -> f32 {
        let base = if winner {
            1.0
        } else if survived {
            0.5
        } else {
            0.0
        };
        let pu_bonus = (powerups as f32 * 0.05).min(0.2);
        let kill_bonus = (kills as f32 * 0.1).min(0.3);
        (base + pu_bonus + kill_bonus).clamp(0.0, 1.0)
    }

    /// Serialize to a single JSON line (for JSONL output).
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Deserialize from a JSON line.
    pub fn from_json(line: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(line)
    }
}

// ── ReplayWriter ───────────────────────────────────────────────

pub struct ReplayWriter {
    file: BufWriter<std::fs::File>,
    round: u32,
    sample_count: u64,
}

impl ReplayWriter {
    /// Open (or create) a JSONL file for replay samples.
    pub fn create(path: &Path, round: u32) -> std::io::Result<Self> {
        let file = std::fs::File::create(path)?;
        Ok(Self {
            file: BufWriter::new(file),
            round,
            sample_count: 0,
        })
    }

    /// Write one sample as a JSON line.
    pub fn write_sample(&mut self, sample: &ReplaySample) -> std::io::Result<()> {
        let json = sample.to_json();
        writeln!(self.file, "{json}")?;
        self.sample_count += 1;
        Ok(())
    }

    /// Flush buffered writes to disk.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }

    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    pub fn round(&self) -> u32 {
        self.round
    }
}

// ── Game-State Serialization ───────────────────────────────────

/// Convert `ArenaGrid` cells into a flat `Vec<u8>` (169 bytes for 13×13).
///
/// Encoding: Floor=0, FixedWall=1, DestructibleWall=2, PowerUpHidden=3
pub fn serialize_board(grid: &ArenaGrid) -> Vec<u8> {
    let mut board = vec![0u8; ARENA_W * ARENA_H];
    for y in 0..ARENA_H {
        for x in 0..ARENA_W {
            let cell_byte = match grid.cells[y][x] {
                Cell::Floor => 0,
                Cell::FixedWall => 1,
                Cell::DestructibleWall => 2,
                Cell::PowerUpHidden(_) => 3,
            };
            board[y * ARENA_W + x] = cell_byte;
        }
    }
    board
}

/// Extract all bomb entities as `(x, y, blast_range, fuse_ticks)`.
pub fn serialize_bombs(world: &mut World) -> Vec<[u8; 4]> {
    let mut bombs = Vec::new();
    let mut query = world.query_filtered::<(&GridPos, &BombRange, &BombFuse), With<Bomb>>();
    for (pos, range, fuse) in query.iter(world) {
        bombs.push([
            pos.x as u8,
            pos.y as u8,
            range.cells as u8,
            fuse.ticks_remaining as u8,
        ]);
    }
    bombs
}

/// Extract all powerup entities as `(x, y)`.
pub fn serialize_powerups(world: &mut World) -> Vec<[u8; 2]> {
    let mut powerups = Vec::new();
    let mut query = world.query_filtered::<(&GridPos, &PowerUp), ()>();
    for (pos, _pu) in query.iter(world) {
        powerups.push([pos.x as u8, pos.y as u8]);
    }
    powerups
}

/// Extract bomb type bytes from all bomb entities.
///
/// Encoding: 0=Timed, 1=Piercing, 2=Remote, 3=Landmine.
/// Order matches [`serialize_bombs`] output — uses identical entity set for consistent iteration.
pub fn serialize_bomb_types(world: &mut World) -> Vec<u8> {
    let mut types = Vec::new();
    // Same components as serialize_bombs (+ Bomb) for consistent iteration order
    let mut query = world.query_filtered::<(&GridPos, &BombRange, &BombFuse, &Bomb), ()>();
    for (_, _, _, bomb) in query.iter(world) {
        types.push(bomb.bomb_type.to_u8());
    }
    types
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Quality computation ────────────────────────────────────

    #[test]
    fn quality_death_is_zero() {
        let q = ReplaySample::quality(false, false, 0, 0);
        assert!((q - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn quality_survived_is_half() {
        let q = ReplaySample::quality(true, false, 0, 0);
        assert!((q - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn quality_winner_is_one() {
        let q = ReplaySample::quality(true, true, 0, 0);
        assert!((q - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn quality_powerup_bonus_capped() {
        // 5 powerups × 0.05 = 0.25, capped at 0.2 → 0.5 + 0.2 = 0.7
        let q = ReplaySample::quality(true, false, 5, 0);
        assert!((q - 0.7).abs() < 1e-6);
    }

    #[test]
    fn quality_kill_bonus_capped() {
        // 5 kills × 0.1 = 0.5, capped at 0.3 → 0.5 + 0.3 = 0.8
        let q = ReplaySample::quality(true, false, 0, 5);
        assert!((q - 0.8).abs() < 1e-6);
    }

    #[test]
    fn quality_winner_does_not_exceed_one() {
        let q = ReplaySample::quality(true, true, 10, 10);
        assert!(q <= 1.0);
    }

    #[test]
    fn quality_single_powerup_and_kill() {
        // survived(0.5) + 1pu(0.05) + 1kill(0.1) = 0.65
        let q = ReplaySample::quality(true, false, 1, 1);
        assert!((q - 0.65).abs() < 1e-6);
    }

    // ── JSON roundtrip ─────────────────────────────────────────

    #[test]
    fn sample_json_roundtrip() {
        let sample = ReplaySample {
            board: vec![0u8; 169],
            player_pos: [5, 7],
            player_id: 2,
            bombs: vec![[3, 3, 2, 4]],
            powerups: vec![[1, 1]],
            action: BomberAction::Bomb.as_usize() as u8,
            quality: 0.85,
            tick: 42,
            round: 7,
            player_type: "Greedy".to_string(),
            danger_level: 0,
            nearest_opponent_dist: 0,
            escape_routes: 0,
            bomb_types: vec![0],
            template_id: 255,
        };

        let json = sample.to_json();
        let restored = ReplaySample::from_json(&json).expect("deserialization should succeed");

        assert_eq!(restored.player_pos, [5, 7]);
        assert_eq!(restored.player_id, 2);
        assert_eq!(restored.bombs, vec![[3, 3, 2, 4]]);
        assert_eq!(restored.powerups, vec![[1, 1]]);
        assert_eq!(restored.action, 4); // Bomb
        assert!((restored.quality - 0.85).abs() < 1e-6);
        assert_eq!(restored.tick, 42);
        assert_eq!(restored.round, 7);
        assert_eq!(restored.player_type, "Greedy");
        assert_eq!(restored.bomb_types, vec![0]);
    }

    // ── Board serialization ────────────────────────────────────

    #[test]
    fn serialize_board_known_grid() {
        let grid = ArenaGrid {
            cells: vec![vec![Cell::Floor; ARENA_W]; ARENA_H],
            width: ARENA_W,
            height: ARENA_H,
        };
        let board = serialize_board(&grid);
        assert_eq!(board, vec![0u8; 169]);
    }

    #[test]
    fn serialize_board_mixed_cells() {
        let mut grid = ArenaGrid {
            cells: vec![vec![Cell::Floor; ARENA_W]; ARENA_H],
            width: ARENA_W,
            height: ARENA_H,
        };
        grid.cells[0][0] = Cell::FixedWall;
        grid.cells[1][1] = Cell::DestructibleWall;
        grid.cells[2][3] = Cell::PowerUpHidden(PowerUpKind::BombUp);

        let board = serialize_board(&grid);

        assert_eq!(board[0], 1); // (0,0) FixedWall
        assert_eq!(board[ARENA_W + 1], 2); // (1,1) DestructibleWall
        assert_eq!(board[2 * ARENA_W + 3], 3); // (3,2) PowerUpHidden
    }

    // ── ReplayWriter ───────────────────────────────────────────

    #[test]
    fn writer_writes_and_counts_samples() {
        let dir = std::env::temp_dir().join("bomber_replay_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_replay.jsonl");

        let sample = ReplaySample {
            board: vec![0u8; 169],
            player_pos: [1, 1],
            player_id: 0,
            bombs: vec![],
            powerups: vec![],
            action: 0,
            quality: 0.5,
            tick: 1,
            round: 1,
            player_type: "Random".to_string(),
            danger_level: 0,
            nearest_opponent_dist: 0,
            escape_routes: 0,
            bomb_types: vec![],
            template_id: 255,
        };

        {
            let mut writer = ReplayWriter::create(&path, 1).unwrap();
            for _ in 0..5 {
                writer.write_sample(&sample).unwrap();
            }
            writer.flush().unwrap();
            assert_eq!(writer.sample_count(), 5);
            assert_eq!(writer.round(), 1);
        }

        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.trim().lines().collect();
        assert_eq!(lines.len(), 5);

        for line in &lines {
            let s = ReplaySample::from_json(line).unwrap();
            assert_eq!(s.player_type, "Random");
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    // ── new_enriched constructor ───────────────────────────────

    #[test]
    fn enriched_no_bombs_no_opponents_safe() {
        let grid = ArenaGrid {
            cells: vec![vec![Cell::Floor; ARENA_W]; ARENA_H],
            width: ARENA_W,
            height: ARENA_H,
        };

        let sample = ReplaySample::new_enriched(
            vec![0u8; 169],
            [5, 5],
            0,
            vec![], // no bombs
            vec![],
            0,
            0.5,
            1,
            1,
            "Test".to_string(),
            &grid,
            &[], // no opponents
        );

        assert_eq!(sample.danger_level, 0);
        assert_eq!(sample.nearest_opponent_dist, 255);
        assert_eq!(sample.escape_routes, 4); // open floor, all 4 directions
    }

    #[test]
    fn enriched_player_in_blast_zone() {
        let grid = ArenaGrid {
            cells: vec![vec![Cell::Floor; ARENA_W]; ARENA_H],
            width: ARENA_W,
            height: ARENA_H,
        };

        // Bomb at (3, 5) with range 3 — blast reaches (5, 5) going right
        let sample = ReplaySample::new_enriched(
            vec![0u8; 169],
            [5, 5],
            0,
            vec![[3, 5, 3, 2]],
            vec![],
            0,
            0.5,
            1,
            1,
            "Test".to_string(),
            &grid,
            &[],
        );

        assert_eq!(sample.danger_level, 2); // directly in blast zone
    }

    #[test]
    fn enriched_player_adjacent_to_blast_zone() {
        let grid = ArenaGrid {
            cells: vec![vec![Cell::Floor; ARENA_W]; ARENA_H],
            width: ARENA_W,
            height: ARENA_H,
        };

        // Bomb at (3, 5) with range 1 — blast covers (3,5), (4,5), (2,5), (3,4), (3,6)
        // Player at (5, 5) — not in blast, but (4, 5) is adjacent and in blast
        let sample = ReplaySample::new_enriched(
            vec![0u8; 169],
            [5, 5],
            0,
            vec![[3, 5, 1, 2]],
            vec![],
            0,
            0.5,
            1,
            1,
            "Test".to_string(),
            &grid,
            &[],
        );

        assert_eq!(sample.danger_level, 1); // adjacent to blast zone
    }

    #[test]
    fn enriched_nearest_opponent_distance() {
        let grid = ArenaGrid {
            cells: vec![vec![Cell::Floor; ARENA_W]; ARENA_H],
            width: ARENA_W,
            height: ARENA_H,
        };

        let sample = ReplaySample::new_enriched(
            vec![0u8; 169],
            [5, 5],
            0,
            vec![],
            vec![],
            0,
            0.5,
            1,
            1,
            "Test".to_string(),
            &grid,
            &[(8, 5), (5, 9), (1, 1)], // manhattan: 3, 4, 8 → min = 3
        );

        assert_eq!(sample.nearest_opponent_dist, 3);
    }

    #[test]
    fn enriched_escape_routes_with_walls() {
        let mut grid = ArenaGrid {
            cells: vec![vec![Cell::Floor; ARENA_W]; ARENA_H],
            width: ARENA_W,
            height: ARENA_H,
        };
        // Block left and up with walls
        grid.cells[5][4] = Cell::FixedWall; // left of (5,5)
        grid.cells[4][5] = Cell::DestructibleWall; // above (5,5)

        let sample = ReplaySample::new_enriched(
            vec![0u8; 169],
            [5, 5],
            0,
            vec![],
            vec![],
            0,
            0.5,
            1,
            1,
            "Test".to_string(),
            &grid,
            &[],
        );

        assert_eq!(sample.escape_routes, 2); // only right and down
    }

    #[test]
    fn enriched_backward_compat_json_without_new_fields() {
        // Old JSON without danger_level, nearest_opponent_dist, escape_routes
        let old_json = r#"{"board":[0,0,0],"player_pos":[1,1],"player_id":0,"bombs":[],"powerups":[],"action":0,"quality":0.5,"tick":1,"round":1,"player_type":"Old"}"#;

        let sample = ReplaySample::from_json(old_json).expect("should parse old format");
        assert_eq!(sample.danger_level, 0);
        assert_eq!(sample.nearest_opponent_dist, 0);
        assert_eq!(sample.escape_routes, 0);
        assert!(sample.bomb_types.is_empty());
    }

    #[test]
    fn enriched_backward_compat_json_without_template_id() {
        // Old JSON without template_id field — should default to 255
        let old_json = r#"{"board":[0,0,0],"player_pos":[1,1],"player_id":0,"bombs":[],"powerups":[],"action":0,"quality":0.5,"tick":1,"round":1,"player_type":"Old","danger_level":0,"nearest_opponent_dist":0,"escape_routes":0,"bomb_types":[]}"#;

        let sample = ReplaySample::from_json(old_json).expect("should parse old format");
        assert_eq!(
            sample.template_id, 255,
            "template_id should default to 255 for old replays"
        );
    }
}
