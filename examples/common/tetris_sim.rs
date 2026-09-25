//! Tetris board sim + Dellacherie-class features + the pinned laya-protocol
//! sentence grammar — the shared substrate of the Plan 607 game-decision
//! arenas (`tetris_01_state_enum` T4a, `tetris_02_arena` T4).
//!
//! Included via `#[path]` by each example (the `tests/common/ab_timing.rs`
//! precedent — `examples/common/` carries no `main.rs` so cargo never
//! auto-discovers it as a target).
//!
//! ## The laya Tetris protocol (pinned grammar `laya-tetris-v1`)
//!
//! Per <https://brainfunctioncollapse.com/laya>: code enumerates the landing
//! spots and does the counting; the model reads ONE sentence per distinct
//! spot — their recorded example is `The piece leaves one hole under it and
//! makes a small bump on top` — and returns P(clean); the piece goes where
//! the code saw the cleanest stack. Number WORDS are in-protocol ("one hole"
//! is the counted conclusion handed over in words — the model cannot read
//! digits, so arithmetic stays here). Wording sensitivity is a measured trap
//! on their page (`blocked by a barrier` 0.75 vs `blocked by a train` 0.45),
//! which is why the grammar is CLOSED and pinned: every clause template and
//! every quantization band lives in this file and the T0b fixture freezes
//! the rendered bytes.
//!
//! ## Determinism law
//!
//! The T0b fixture's provenance sha covers these bytes: same seed in, same
//! dump out, bit-identical on every box (integer feature arithmetic; the
//! only floats are exact halves/quarters from division by 2/4, plus the
//! Dellacherie weights which are read-only constants).

#![allow(dead_code)] // arena examples consume different subsets

// ── Board ────────────────────────────────────────────────────────────────

use std::sync::OnceLock;

pub const WIDTH: usize = 10;
pub const HEIGHT: usize = 20;

/// Cell grid, row 0 = top. `true` = occupied.
#[derive(Clone, PartialEq, Eq)]
pub struct Board {
    cells: [[bool; WIDTH]; HEIGHT],
}

impl Board {
    pub fn empty() -> Self {
        Self {
            cells: [[false; WIDTH]; HEIGHT],
        }
    }

    pub fn cell(&self, row: usize, col: usize) -> bool {
        self.cells[row][col]
    }

    /// Column height = occupied cells counted from the floor (0 for an
    /// empty column).
    pub fn col_height(&self, col: usize) -> usize {
        for row in 0..HEIGHT {
            if self.cells[row][col] {
                return HEIGHT - row;
            }
        }
        0
    }

    pub fn heights(&self) -> [usize; WIDTH] {
        let mut h = [0usize; WIDTH];
        for (c, hc) in h.iter_mut().enumerate() {
            *hc = self.col_height(c);
        }
        h
    }

    pub fn hole_count(&self) -> usize {
        let mut holes = 0;
        for c in 0..WIDTH {
            let mut seen = false;
            for r in 0..HEIGHT {
                if self.cells[r][c] {
                    seen = true;
                } else if seen {
                    holes += 1;
                }
            }
        }
        holes
    }

    /// Place `cells` (absolute (row, col) pairs). Rows are NOT cleared.
    pub fn place(&mut self, cells: &[(usize, usize)]) {
        for &(r, c) in cells {
            self.cells[r][c] = true;
        }
    }

    /// Place `cells`, then clear complete rows in one bottom-up compaction
    /// pass (identical result to `place` + `full_rows` + `clear_rows`, without
    /// the intermediate `Vec`). Returns rows cleared.
    pub fn place_and_clear(&mut self, cells: &[(usize, usize)]) -> u32 {
        for &(r, c) in cells {
            self.cells[r][c] = true;
        }
        let mut write = HEIGHT;
        let mut cleared = 0u32;
        for r in (0..HEIGHT).rev() {
            if self.cells[r].iter().all(|&b| b) {
                cleared += 1;
            } else {
                write -= 1;
                if write != r {
                    self.cells[write] = self.cells[r];
                }
            }
        }
        for row in &mut self.cells[..write] {
            *row = [false; WIDTH];
        }
        cleared
    }

    pub fn full_rows(&self) -> Vec<usize> {
        (0..HEIGHT)
            .filter(|&r| (0..WIDTH).all(|c| self.cells[r][c]))
            .collect()
    }

    /// Clear `rows` (descending-independent: each cleared row pulls everything
    /// above it down by one).
    pub fn clear_rows(&mut self, rows: &[usize]) {
        for &r in rows {
            for rr in (1..=r).rev() {
                self.cells[rr] = self.cells[rr - 1];
            }
            self.cells[0] = [false; WIDTH];
        }
    }

    /// Render rows as `#`/`.` strings (top row first) — the dump format.
    pub fn to_strings(&self) -> Vec<String> {
        (0..HEIGHT)
            .map(|r| {
                (0..WIDTH)
                    .map(|c| if self.cells[r][c] { '#' } else { '.' })
                    .collect()
            })
            .collect()
    }

    /// Build from `#`/`.` row strings (inverse of [`Self::to_strings`]).
    pub fn from_strings(rows: &[&str]) -> Self {
        let mut b = Self::empty();
        for (r, row) in rows.iter().enumerate().take(HEIGHT) {
            for (c, ch) in row.chars().enumerate().take(WIDTH) {
                b.cells[r][c] = ch == '#';
            }
        }
        b
    }

    /// Fused single-scan board features (the search hot path): heights +
    /// row/col transitions + holes + hole cover, integer-exact against the
    /// per-feature scans (`heights`/`hole_count`/`f_row_trans`-shape walks).
    /// Two passes over the 200-byte grid: one column-major, one row-major.
    pub fn scan(&self) -> BoardScan {
        let mut heights = [0usize; WIDTH];
        let mut col_trans = 0u32;
        let mut holes = 0u32;
        let mut hole_cover = 0u32;
        for (c, hc) in heights.iter_mut().enumerate() {
            let mut top: isize = -1;
            let mut filled = 0u32;
            let mut prev = false;
            let mut filled_above = 0u32;
            for r in 0..HEIGHT {
                let cur = self.cells[r][c];
                col_trans += u32::from(cur != prev);
                prev = cur;
                if cur {
                    if top < 0 {
                        top = r as isize;
                    }
                    filled += 1;
                    filled_above += 1;
                } else if top >= 0 {
                    // A hole: everything above it covers it.
                    hole_cover += filled_above;
                }
            }
            col_trans += u32::from(!prev); // floor (bottom cell empty → transition)
            *hc = if top < 0 {
                0
            } else {
                (HEIGHT as isize - top) as usize
            };
            holes += *hc as u32 - filled;
        }
        let mut row_trans = 0u32;
        for r in 0..HEIGHT {
            let mut prev = true; // wall
            for c in 0..WIDTH {
                let cur = self.cells[r][c];
                row_trans += u32::from(cur != prev);
                prev = cur;
            }
            row_trans += u32::from(!prev); // right wall
        }
        BoardScan {
            heights,
            row_trans,
            col_trans,
            holes,
            hole_cover,
        }
    }
}

/// [`Board::scan`] output — integer-valued features (exact in f64).
#[derive(Clone, Copy, Debug)]
pub struct BoardScan {
    pub heights: [usize; WIDTH],
    pub row_trans: u32,
    pub col_trans: u32,
    pub holes: u32,
    pub hole_cover: u32,
}

// ── Pieces ───────────────────────────────────────────────────────────────

/// The seven tetrominoes. Prose uses [`Self::spoken`] (word form, never a
/// bare letter — the protocol hands conclusions over in words).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Piece {
    I,
    O,
    T,
    S,
    Z,
    J,
    L,
}

impl Piece {
    pub const ALL: [Self; 7] = [
        Self::I,
        Self::O,
        Self::T,
        Self::S,
        Self::Z,
        Self::J,
        Self::L,
    ];

    /// Spoken name for the state sentence.
    pub fn spoken(&self) -> &'static str {
        match self {
            Self::I => "long straight",
            Self::O => "square",
            Self::T => "T shaped",
            Self::S => "S shaped",
            Self::Z => "Z shaped",
            Self::J => "left leaning ell",
            Self::L => "right leaning ell",
        }
    }

    /// Letter id for the structured dump (machine field, never prose).
    pub fn id(&self) -> &'static str {
        match self {
            Self::I => "I",
            Self::O => "O",
            Self::T => "T",
            Self::S => "S",
            Self::Z => "Z",
            Self::J => "J",
            Self::L => "L",
        }
    }

    /// Distinct rotations as cell-offset lists (dy, dx), dy=0 at the top of
    /// the bounding box, normalized to the origin. Deduped (I: 2, O: 1,
    /// rest: 4) — order is pinned (rotation counterclockwise from base).
    /// Backed by a process-wide static table (built by the same algorithm
    /// once): allocation-free reads for the search hot path.
    pub fn rotations(&self) -> Vec<Vec<(usize, usize)>> {
        rotations_static(*self)
            .iter()
            .map(|f| f.cells.to_vec())
            .collect()
    }

    /// Index into `Piece::ALL` (the static rotation table's key).
    pub fn index(self) -> usize {
        match self {
            Self::I => 0,
            Self::O => 1,
            Self::T => 2,
            Self::S => 3,
            Self::Z => 4,
            Self::J => 5,
            Self::L => 6,
        }
    }
}

/// One distinct rotation in the static table: normalized (sorted) cells +
/// bounding width — exactly what `Piece::rotations()` returns, minus the
/// per-call allocation.
pub struct RotForm {
    pub cells: [(usize, usize); 4],
    /// max dx + 1 (the column-span of the form).
    pub width: usize,
}

static ROT_FORMS: OnceLock<[Vec<RotForm>; 7]> = OnceLock::new();

/// The process-wide rotation table (built once from the pinned normalize/
/// dedupe pipeline; `Piece::rotations()` is a view over it).
pub fn rotations_static(piece: Piece) -> &'static [RotForm] {
    &ROT_FORMS.get_or_init(build_rot_forms)[piece.index()]
}

fn build_rot_forms() -> [Vec<RotForm>; 7] {
    let mut out: [Vec<RotForm>; 7] = Default::default();
    for (i, &piece) in Piece::ALL.iter().enumerate() {
        let base: Vec<(usize, usize)> = match piece {
            Piece::I => vec![(1, 0), (1, 1), (1, 2), (1, 3)],
            Piece::O => vec![(0, 1), (0, 2), (1, 1), (1, 2)],
            Piece::T => vec![(0, 1), (1, 0), (1, 1), (1, 2)],
            Piece::S => vec![(0, 1), (0, 2), (1, 0), (1, 1)],
            Piece::Z => vec![(0, 0), (0, 1), (1, 1), (1, 2)],
            Piece::J => vec![(0, 0), (1, 0), (1, 1), (1, 2)],
            Piece::L => vec![(0, 2), (1, 0), (1, 1), (1, 2)],
        };
        let dim = 4usize; // rotation inside a 4x4 box
        let mut cur = base;
        let mut norms: Vec<Vec<(usize, usize)>> = Vec::with_capacity(4);
        for _ in 0..4 {
            let norm = normalize(&cur);
            if !norms.contains(&norm) {
                norms.push(norm);
            }
            cur = rotate(&cur, dim);
        }
        out[i] = norms
            .iter()
            .map(|v| {
                let mut cells = [(0usize, 0usize); 4];
                for (dst, &c) in cells.iter_mut().zip(v.iter()) {
                    *dst = c;
                }
                let width = cells.iter().map(|&(_, dx)| dx).max().unwrap_or(0) + 1;
                RotForm { cells, width }
            })
            .collect();
    }
    out
}

/// Rotate (dy, dx) clockwise inside a `dim`-box: (dy, dx) -> (dx, dim-1-dy).
fn rotate(cells: &[(usize, usize)], dim: usize) -> Vec<(usize, usize)> {
    cells.iter().map(|&(dy, dx)| (dx, dim - 1 - dy)).collect()
}

/// Shift cells so min dy/dx are 0 (canonical placement-independent form).
fn normalize(cells: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let min_dy = cells.iter().map(|&(dy, _)| dy).min().unwrap_or(0);
    let min_dx = cells.iter().map(|&(_, dx)| dx).min().unwrap_or(0);
    let mut v: Vec<(usize, usize)> = cells
        .iter()
        .map(|&(dy, dx)| (dy - min_dy, dx - min_dx))
        .collect();
    v.sort_unstable();
    v
}

// ── Landing enumeration (hard-drop semantics) ────────────────────────────

/// One legal resting placement.
#[derive(Clone, Debug)]
pub struct Placement {
    /// Rotation index into `Piece::rotations()` output order.
    pub rot: usize,
    /// Left column of the piece's bounding box at rest.
    pub col: usize,
    /// Top row of the piece's bounding box at rest.
    pub row: usize,
    /// Absolute resting cells (row, col), sorted. A tetromino is always
    /// exactly 4 cells — the fixed array keeps the search hot path
    /// allocation-free.
    pub cells: [(usize, usize); 4],
}

/// Is the piece at bounding-box top-left (row, col) collision-free?
fn fits(board: &Board, cells: &[(usize, usize)], row: usize, col: usize) -> bool {
    cells.iter().all(|&(dy, dx)| {
        let r = row + dy;
        let c = col + dx;
        c < WIDTH && r < HEIGHT && !board.cell(r, c)
    })
}

/// How a piece comes to rest in a column — the one axis the v2 and v3
/// fixtures differ on (katgpt-rs Issue 884). The sentence grammar, feature
/// arithmetic and option ORDER are shared; only the rest row differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DropRule {
    /// `laya-tetris-v2` (pinned): rest at the DEEPEST collision-free row,
    /// scanning bottom-up — under an overhang the piece passes through the
    /// roof into the cave below (3 of 2660 v2 options).
    DeepestFit,
    /// `laya-tetris-v3`: a real hard drop — spawn at the top of the column,
    /// descend while the next row is free, stop at the first collision.
    /// A column whose top row is blocked has no landing (top-out there).
    FromTop,
}

impl DropRule {
    /// The grammar id a dump built under this rule is stamped with.
    pub fn grammar_id(self) -> &'static str {
        match self {
            Self::DeepestFit => GRAMMAR_ID,
            Self::FromTop => GRAMMAR_ID_V3,
        }
    }

    /// Inverse of [`grammar_id`](Self::grammar_id); `None` for an unknown id.
    pub fn from_grammar(id: &str) -> Option<Self> {
        match id {
            GRAMMAR_ID => Some(Self::DeepestFit),
            GRAMMAR_ID_V3 => Some(Self::FromTop),
            _ => None,
        }
    }
}

/// Hard drop down column `col` under the pinned v2 rule
/// ([`DropRule::DeepestFit`]). None when the piece fits at no row.
pub fn hard_drop(board: &Board, cells: &[(usize, usize)], col: usize) -> Option<Placement> {
    hard_drop_with(board, cells, col, DropRule::DeepestFit)
}

/// Hard drop down column `col` under `rule`. None when there is no landing.
pub fn hard_drop_with(
    board: &Board,
    cells: &[(usize, usize)],
    col: usize,
    rule: DropRule,
) -> Option<Placement> {
    let rest = match rule {
        DropRule::DeepestFit => (0..HEIGHT).rev().find(|&row| fits(board, cells, row, col)),
        DropRule::FromTop => from_top_rest(board, cells, col).map(|r| r as usize),
    }?;
    Some(Placement {
        rot: 0, // caller owns the rotation index
        col,
        row: rest,
        cells: {
            let mut v = [(0usize, 0usize); 4];
            for (dst, &(dy, dx)) in v.iter_mut().zip(cells.iter()) {
                *dst = (rest + dy, col + dx);
            }
            v.sort_unstable();
            v
        },
    })
}

/// FromTop rest row from per-column occupancy masks instead of the
/// row-by-row descent scan: each piece cell is blocked by the FIRST
/// occupied row at or below its spawn row (`trailing_zeros` over the
/// column mask), and the piece rests one row above its first collision.
/// Returns `None` when the spawn itself is blocked (`min room <= 0`).
/// Bit-identical to the descent scan in `hard_drop_with` (pinned by the
/// equivalence tests below) — a plain heights shortcut is WRONG under
/// overhangs (the shadow below a floating cell is open).
fn from_top_rest(board: &Board, cells: &[(usize, usize)], col: usize) -> Option<u8> {
    let mut masks = [0u32; WIDTH];
    for (c, m) in masks.iter_mut().enumerate() {
        let mut v = 0u32;
        for r in 0..HEIGHT {
            if board.cell(r, c) {
                v |= 1 << r;
            }
        }
        *m = v;
    }
    let mut min_room = isize::MAX;
    for &(dy, dx) in cells {
        let below = masks[col + dx] & !((1u32 << dy) - 1); // rows ≥ dy
        let first_occ = if below == 0 {
            HEIGHT as u32
        } else {
            below.trailing_zeros()
        };
        let room = first_occ as isize - dy as isize;
        if room < min_room {
            min_room = room;
        }
    }
    (min_room > 0).then_some((min_room - 1) as u8)
}

/// Enumerate every distinct hard-drop landing for `piece` on `board` under
/// the pinned v2 rule ([`DropRule::DeepestFit`]), in pinned order: rotation
/// ascending, then column ascending. This is the option ORDER the fixture
/// freezes — the oracle's and the arena's argmax index both refer to it.
pub fn landing_options(board: &Board, piece: Piece) -> Vec<Placement> {
    landing_options_with(board, piece, DropRule::DeepestFit)
}

/// [`landing_options`] under an explicit drop rule (same pinned order).
/// FromTop takes the heights-based fast path (no per-option descent scan);
/// DeepestFit keeps the scan (fixture-fidelity lane).
pub fn landing_options_with(board: &Board, piece: Piece, rule: DropRule) -> Vec<Placement> {
    let mut out = Vec::with_capacity(34);
    match rule {
        DropRule::FromTop => {
            // Per-column occupancy masks: bit r set ⇔ (r, c) occupied. The
            // FromTop rest row needs, per piece cell, the FIRST occupied row
            // at or below the cell's spawn row — a heights shortcut is wrong
            // under overhangs (the shadow below a floating cell is open).
            let mut masks = [0u32; WIDTH];
            for (c, m) in masks.iter_mut().enumerate() {
                let mut v = 0u32;
                for r in 0..HEIGHT {
                    if board.cell(r, c) {
                        v |= 1 << r;
                    }
                }
                *m = v;
            }
            for (ri, form) in rotations_static(piece).iter().enumerate() {
                for col in 0..=(WIDTH - form.width) {
                    let mut min_room = isize::MAX;
                    for &(dy, dx) in &form.cells {
                        let below = masks[col + dx] & !((1u32 << dy) - 1); // rows ≥ dy
                        let first_occ = if below == 0 {
                            HEIGHT as u32
                        } else {
                            below.trailing_zeros()
                        };
                        let room = first_occ as isize - dy as isize;
                        if room < min_room {
                            min_room = room;
                        }
                    }
                    if min_room <= 0 {
                        continue; // spawn blocked — no v3 landing here
                    }
                    let rest = (min_room - 1) as usize;
                    let mut cells = [(0usize, 0usize); 4];
                    for (dst, &(dy, dx)) in cells.iter_mut().zip(&form.cells) {
                        *dst = (rest + dy, col + dx);
                    }
                    cells.sort_unstable();
                    out.push(Placement {
                        rot: ri,
                        col,
                        row: rest,
                        cells,
                    });
                }
            }
        }
        _ => {
            for (ri, cells) in piece.rotations().iter().enumerate() {
                let width = cells.iter().map(|&(_, dx)| dx).max().unwrap_or(0) + 1;
                for col in 0..=(WIDTH - width) {
                    if let Some(mut p) = hard_drop_with(board, cells, col, rule) {
                        p.rot = ri;
                        out.push(p);
                    }
                }
            }
        }
    }
    out
}

// ── Dellacherie-class outcome features ───────────────────────────────────

/// Post-landing outcome features for one placement (post-placement,
/// pre-clear). All integer-derived; the plan's named four (holes /
/// bumpiness / stack height / line clears) plus the Dellacherie set the
/// arena may select among.
#[derive(Clone, Copy, Debug, Default)]
pub struct OutcomeFeatures {
    /// Rows completed by this placement (before clearing).
    pub lines_cleared: u32,
    /// Holes on the board AFTER placing (pre-clear).
    pub holes: u32,
    /// Holes created by this placement (after − before).
    pub holes_delta: i32,
    /// Σ |`h_c` − `h_{c+1`}| over adjacent columns.
    pub bumpiness: u32,
    /// Tallest column.
    pub max_height: u32,
    /// Σ column heights.
    pub aggregate_height: u32,
    /// Mean height of the piece cells from the floor (Dellacherie's
    /// landing height; exact halves/quarters only).
    pub landing_height: f32,
    /// Filled↔empty horizontal transitions (walls count filled).
    pub row_transitions: u32,
    /// Filled↔empty vertical transitions (floor counts filled).
    pub col_transitions: u32,
    /// Σ well depths 1+2+…+d over every well.
    pub cumulative_wells: u32,
    /// Piece cells inside completed rows (Dellacherie's eroded piece
    /// cells, unweighted).
    pub eroded_cells: u32,
}

/// Compute the outcome features of placing `p` on `board`.
pub fn outcome_features(board: &Board, p: &Placement) -> OutcomeFeatures {
    let holes_before = board.hole_count() as i32;
    let mut after = board.clone();
    after.place(&p.cells);
    let full = after.full_rows();

    let heights = after.heights();
    let max_height = heights.iter().copied().max().unwrap_or(0) as u32;
    let aggregate_height: u32 = heights.iter().sum::<usize>() as u32;
    let bumpiness: u32 = (1..WIDTH)
        .map(|i| heights[i].abs_diff(heights[i - 1]))
        .sum::<usize>() as u32;

    let holes_after = after.hole_count() as u32;

    // Row transitions: per row, flank changes; walls read as filled.
    let mut row_transitions = 0u32;
    for r in 0..HEIGHT {
        let mut prev = true;
        for c in 0..WIDTH {
            let cur = after.cell(r, c);
            if cur != prev {
                row_transitions += 1;
            }
            prev = cur;
        }
        if !prev {
            row_transitions += 1;
        }
    }

    // Column transitions: per column, flank changes; floor reads filled.
    let mut col_transitions = 0u32;
    for c in 0..WIDTH {
        let mut prev = false;
        for r in 0..HEIGHT {
            let cur = after.cell(r, c);
            if cur != prev {
                col_transitions += 1;
            }
            prev = cur;
        }
        if !prev {
            col_transitions += 1;
        }
    }

    // Wells: a column below both neighbours is a well of depth d =
    // min(neighbours) − h; each contributes 1+2+…+d.
    let mut cumulative_wells = 0u32;
    for c in 0..WIDTH {
        let h = heights[c];
        let left = if c == 0 { usize::MAX } else { heights[c - 1] };
        let right = if c == WIDTH - 1 {
            usize::MAX
        } else {
            heights[c + 1]
        };
        if left > h && right > h {
            let d = left.min(right) - h;
            cumulative_wells += (d * (d + 1) / 2) as u32;
        }
    }

    let eroded_cells = p.cells.iter().filter(|&&(r, _)| full.contains(&r)).count() as u32;

    let landing_height = p
        .cells
        .iter()
        .map(|&(r, _)| (HEIGHT - r) as f32)
        .sum::<f32>()
        / p.cells.len() as f32;

    OutcomeFeatures {
        lines_cleared: full.len() as u32,
        holes: holes_after,
        holes_delta: holes_after as i32 - holes_before,
        bumpiness,
        max_height,
        aggregate_height,
        landing_height,
        row_transitions,
        col_transitions,
        cumulative_wells,
        eroded_cells,
    }
}

/// Classic Dellacherie weights (the literature landing heuristic — "solved
/// engineering" per the plan). Used by the seeded play ladder and as the
/// arena's heuristic baseline.
pub const DELLACHERIE_WEIGHTS: [f32; 6] = [
    -4.500_158_3, // landing height
    3.418_126_8,  // eroded piece cells
    -3.217_888_4, // row transitions
    -9.348_696,   // col transitions
    -7.899_265_3, // holes
    -3.385_597_2, // cumulative wells
];

/// Dellacherie score of an outcome (higher = better placement).
pub fn dellacherie_score(f: &OutcomeFeatures) -> f32 {
    DELLACHERIE_WEIGHTS[0] * f.landing_height
        + DELLACHERIE_WEIGHTS[1] * f.eroded_cells as f32
        + DELLACHERIE_WEIGHTS[2] * f.row_transitions as f32
        + DELLACHERIE_WEIGHTS[3] * f.col_transitions as f32
        + DELLACHERIE_WEIGHTS[4] * f.holes as f32
        + DELLACHERIE_WEIGHTS[5] * f.cumulative_wells as f32
}

// ── The pinned sentence grammar (`laya-tetris-v1`) ───────────────────────

/// Grammar identity stamped into every dump record.
pub const GRAMMAR_ID: &str = "laya-tetris-v2";

/// The v3 grammar id: the v2 sentence grammar under a real hard drop
/// (`DropRule::FromTop`, katgpt-rs Issue 884). Sentences are unchanged;
/// the option set differs wherever v2 tunnelled through a roof.
pub const GRAMMAR_ID_V3: &str = "laya-tetris-v3";

/// The v4 grammar id: the v3 game under a next-piece preview — the state
/// sentence gains `The next piece is the {piece} piece.` (plan 609 T1.2);
/// option sentences stay byte-compatible with v3's renders (the preview
/// lives in the state line only).
pub const GRAMMAR_ID_V4: &str = "laya-tetris-v4";

/// The per-spot question, world-anchored (never "what should I do" — the
/// wording lesson from laya's own page). P(clean) is the oracle signal.
pub const SPOT_QUESTION: &str = "Does the stack look clean?";

/// How the resting piece meets the pre-placement surface — the surface
/// clause's quantization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BumpBand {
    /// Top within 1 of the surrounding surface.
    Flat,
    /// 2–3 above the surface.
    Small,
    /// ≥ 4 above the surface.
    Tall,
    /// Settles into terrain below both flanks by ≥ 2 (gap/well fill).
    Gap,
}

/// Classify the placement's meeting with the pre-placement surface.
pub fn bump_band(board: &Board, p: &Placement) -> BumpBand {
    let mut piece_cols: Vec<usize> = p.cells.iter().map(|&(_, c)| c).collect();
    piece_cols.sort_unstable();
    piece_cols.dedup();
    let top_of_piece = p.cells.iter().map(|&(r, _)| r).min().unwrap_or(HEIGHT);
    let heights = board.heights();

    // Flank heights: the columns just outside the span, plus any interior
    // column of the span that sits below both of ITS piece-covered
    // neighbours by ≥ 2 (the piece bridges over it — reads as a gap).
    let first = piece_cols[0];
    let last = *piece_cols.last().unwrap();
    let mut gap_under = false;
    let mut flanks: Vec<usize> = Vec::with_capacity(4);
    if first > 0 {
        flanks.push(heights[first - 1]);
    }
    if last + 1 < WIDTH {
        flanks.push(heights[last + 1]);
    }
    for i in 0..piece_cols.len() {
        let h = heights[piece_cols[i]];
        if i > 0 {
            let lh = heights[piece_cols[i - 1]];
            if h + 2 <= lh {
                gap_under = true;
            }
        }
        if i + 1 < piece_cols.len() {
            let rh = heights[piece_cols[i + 1]];
            if h + 2 <= rh {
                gap_under = true;
            }
        }
    }
    if gap_under {
        return BumpBand::Gap;
    }
    if flanks.is_empty() {
        return BumpBand::Flat;
    }
    let piece_top_height = HEIGHT - top_of_piece; // piece top, floor-relative
    let max_flank = flanks.iter().copied().max().unwrap_or(0);
    if flanks.iter().all(|&h| piece_top_height + 2 <= h) {
        return BumpBand::Gap;
    }
    match (piece_top_height as i64) - (max_flank as i64) {
        ..=1 => BumpBand::Flat,
        2..=3 => BumpBand::Small,
        _ => BumpBand::Tall,
    }
}

/// The horizontal landing band — where the piece rests across the board.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SideBand {
    LeftEdge,
    LeftSide,
    Middle,
    RightSide,
    RightEdge,
}

impl SideBand {
    /// Band from the placement's column-span center (WIDTH=10: two
    /// columns per band).
    pub fn of_center(center: usize) -> Self {
        match center {
            0..=1 => Self::LeftEdge,
            2..=3 => Self::LeftSide,
            4..=5 => Self::Middle,
            6..=7 => Self::RightSide,
            _ => Self::RightEdge,
        }
    }

    pub fn clause(&self) -> &'static str {
        match self {
            Self::LeftEdge => "on the left edge",
            Self::LeftSide => "on the left side",
            Self::Middle => "in the middle",
            Self::RightSide => "on the right side",
            Self::RightEdge => "on the right edge",
        }
    }
}

/// The resulting stack-height band (post-placement, pre-clear).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeightBand {
    Low,
    Medium,
    Tall,
}

impl HeightBand {
    pub fn of_max_height(h: u32) -> Self {
        match h {
            0..=6 => Self::Low,
            7..=11 => Self::Medium,
            _ => Self::Tall,
        }
    }

    pub fn clause(&self) -> &'static str {
        match self {
            Self::Low => "the stack stays low",
            Self::Medium => "the stack stands medium",
            Self::Tall => "the stack grows tall",
        }
    }
}

/// Render the per-spot sentence: what the piece DOES to the stack and
/// WHERE it lands (their recorded example shape — `The piece leaves one
/// hole under it and makes a small bump on top` — widened with side +
/// resulting-height clauses so within-state options separate; the v1
/// two-clause grammar tied 66/120 oracle states at identical p_clean and
/// the argmax degenerated to the index tie-break). Closed template; every
/// band a pure function of the board + placement:
///
/// `The piece leaves {no holes|one hole|two holes|a few holes|many holes}
///  under it {on the left edge|on the left side|in the middle|on the right
///  side|on the right edge}, {sits flat on the surface|makes a small bump
///  on top|makes a tall step on top|fills a deep gap}, and the stack {stays
///  low|stands medium|grows tall}{, and clears a line|…}`.
pub fn render_spot_sentence(board: &Board, p: &Placement, f: &OutcomeFeatures) -> String {
    let holes_clause = match f.holes_delta {
        0 => "leaves no holes",
        1 => "leaves one hole",
        2 => "leaves two holes",
        3..=4 => "leaves a few holes",
        _ => "leaves many holes",
    };
    let band = bump_band(board, p);
    let surface_clause = match band {
        BumpBand::Flat => "sits flat on the surface",
        BumpBand::Small => "makes a small bump on top",
        BumpBand::Tall => "makes a tall step on top",
        BumpBand::Gap => "fills a deep gap",
    };
    let mut piece_cols: Vec<usize> = p.cells.iter().map(|&(_, c)| c).collect();
    piece_cols.sort_unstable();
    let center = (piece_cols[0] + piece_cols[piece_cols.len() - 1]) / 2;
    let side_clause = SideBand::of_center(center).clause();
    let height_clause = HeightBand::of_max_height(f.max_height).clause();
    let clears_clause = match f.lines_cleared {
        0 => "",
        1 => ", and clears a line",
        2 => ", and clears two lines",
        3 => ", and clears three lines",
        _ => ", and clears four lines",
    };
    format!(
        "The piece {holes_clause} under it {side_clause}, {surface_clause}, and {height_clause}{clears_clause}."
    )
}

/// The state context sentence (board-level facts, all in words — the lanes
/// demo's row-description shape). Shared prefix context per state; the
/// per-spot sentences remain the decision surface.
pub fn render_state_sentence(board: &Board, piece: Piece) -> String {
    let h = board.heights();
    let max_h = *h.iter().max().unwrap_or(&0);
    let avg = |range: std::ops::Range<usize>| -> u32 {
        let len = range.len();
        range.map(|c| h[c]).sum::<usize>() as u32 / (len as u32)
    };
    let (l, m, r) = (avg(0..3), avg(3..7), avg(7..10));
    let height_word = match max_h {
        0..=4 => "low",
        5..=10 => "of medium height",
        _ => "tall",
    };
    let mut s = String::new();
    let spread = l.max(m).max(r) - l.min(m).min(r);
    if spread >= 3 {
        let mut v = [(l, "left"), (m, "middle"), (r, "right")];
        v.sort_unstable_by_key(|&(x, _)| x);
        let (tallest, lowest) = (v[2].1, v[0].1);
        s.push_str(&format!(
            "The stack stands {height_word}, tall on the {tallest} and low on the {lowest}. "
        ));
    } else {
        s.push_str(&format!(
            "The stack stands {height_word} and the surface is mostly flat. "
        ));
    }
    let holes_clause = match board.hole_count() {
        0 => "There are no holes under the blocks.".to_string(),
        1 => "There is one hole under the blocks.".to_string(),
        2 => "There are two holes under the blocks.".to_string(),
        3..=4 => "There are a few holes under the blocks.".to_string(),
        _ => "There are many holes under the blocks.".to_string(),
    };
    s.push_str(&holes_clause);
    s.push(' ');
    s.push_str(&format!("The {} piece is falling.", piece.spoken()));
    s
}

/// The v4 state sentence: the v3 render plus the next-piece preview
/// (plan 609 T1.2). The preview sentence is the ONLY delta — the base
/// render stays byte-identical to v3's (the v2/v3 pins are untouched by
/// construction: this composes [`render_state_sentence`], never re-spells
/// it).
pub fn render_state_sentence_with_preview(board: &Board, piece: Piece, next: Piece) -> String {
    let mut s = render_state_sentence(board, piece);
    s.push_str(&format!(" The next piece is the {} piece.", next.spoken()));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotations_dedupe() {
        assert_eq!(Piece::I.rotations().len(), 2);
        assert_eq!(Piece::O.rotations().len(), 1);
        assert_eq!(Piece::T.rotations().len(), 4);
        assert_eq!(Piece::L.rotations().len(), 4);
    }

    #[test]
    fn option_count_is_in_the_plan_shape_on_an_empty_board() {
        // The plan's "~34 landing options": four-rotation pieces enumerate
        // in the thirties/forties; the degenerate rotations pin exactly
        // (O = 1 rot × 9 cols, I = 2 rots × (10 + 7) cols).
        for piece in Piece::ALL {
            let n = landing_options(&Board::empty(), piece).len();
            assert!((9..=40).contains(&n), "{piece:?} -> {n}");
        }
        assert_eq!(landing_options(&Board::empty(), Piece::O).len(), 9);
        assert_eq!(landing_options(&Board::empty(), Piece::I).len(), 17);
    }

    #[test]
    fn hard_drop_rests_on_the_floor_of_an_empty_board() {
        let cells = Piece::O.rotations()[0].clone();
        let p = hard_drop(&Board::empty(), &cells, 0).unwrap();
        assert_eq!(p.row, HEIGHT - 2);
    }

    #[test]
    fn from_top_stops_on_a_roof_where_v2_tunnels() {
        // Roof at row 17 cols 0..3 over an empty cave (Issue 884 probe).
        let mut b = Board::empty();
        b.place(&[(17, 0), (17, 1), (17, 2), (17, 3)]);
        let cells = Piece::O.rotations()[0].clone();
        let v2 = hard_drop(&b, &cells, 0).unwrap();
        let v3 = hard_drop_with(&b, &cells, 0, DropRule::FromTop).unwrap();
        assert_eq!(v2.row, 18, "v2 tunnels into the cave");
        assert_eq!(v3.row, 15, "v3 rests on the roof");
        // Open column: both rules agree (the floor).
        let open = hard_drop_with(&b, &cells, 5, DropRule::FromTop).unwrap();
        assert_eq!(open.row, hard_drop(&b, &cells, 5).unwrap().row);
    }

    #[test]
    fn from_top_rules_agree_on_every_empty_board_spot_and_top_out_is_none() {
        for piece in Piece::ALL {
            let a = landing_options(&Board::empty(), piece);
            let b = landing_options_with(&Board::empty(), piece, DropRule::FromTop);
            assert_eq!(a.len(), b.len());
            assert!(
                a.iter()
                    .zip(&b)
                    .all(|(x, y)| x.row == y.row && x.cells == y.cells)
            );
        }
        // A blocked top row: no v3 landing in that column.
        let mut b = Board::empty();
        b.place(&[(0, 0), (1, 0)]);
        let cells = Piece::O.rotations()[0].clone();
        assert!(hard_drop_with(&b, &cells, 0, DropRule::FromTop).is_none());
    }

    #[test]
    fn from_top_fast_path_is_bit_identical_to_the_descent_scan() {
        // Independent reference: the ORIGINAL row-by-row descent, inlined
        // (reaches no shared helper — immune to implementation drift).
        let scan_rest = |board: &Board, cells: &[(usize, usize); 4], col: usize| -> Option<usize> {
            let fits_at = |row: usize| {
                cells.iter().all(|&(dy, dx)| {
                    let r = row + dy;
                    let c = col + dx;
                    c < WIDTH && r < HEIGHT && !board.cells[r][c]
                })
            };
            if !fits_at(0) {
                return None;
            }
            let mut row = 0usize;
            while row + 1 < HEIGHT && fits_at(row + 1) {
                row += 1;
            }
            Some(row)
        };
        let mut boards = vec![
            Board::empty(),
            garbage_like(7, 55),
            garbage_like(8, 75),
            garbage_like(9, 90),
        ];
        let mut x = 0x243F6A8885A308D3u64;
        for _ in 0..3000 {
            let p = 30 + (x % 60);
            let mut b = Board::empty();
            for r in 0..HEIGHT {
                for c in 0..WIDTH {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    if x % 100 < p {
                        b.cells[r][c] = true;
                    }
                }
            }
            boards.push(b);
        }
        for board in &boards {
            for piece in Piece::ALL {
                let fast = landing_options_with(board, piece, DropRule::FromTop);
                let forms = rotations_static(piece);
                let mut mism = String::new();
                let mut n = 0usize;
                for (ri, form) in forms.iter().enumerate() {
                    for col in 0..=(WIDTH - form.width) {
                        match scan_rest(board, &form.cells, col) {
                            Some(rest) => {
                                if n < fast.len()
                                    && fast[n].rot == ri
                                    && fast[n].col == col
                                    && fast[n].row == rest
                                {
                                    n += 1;
                                } else if mism.is_empty() {
                                    mism = format!(
                                        "{piece:?} rot {ri} col {col}: fast {:?} vs scan rest {rest}",
                                        fast.get(n)
                                    );
                                }
                            }
                            None => {
                                if mism.is_empty()
                                    && n < fast.len()
                                    && fast[n].rot == ri
                                    && fast[n].col == col
                                {
                                    mism = format!(
                                        "{piece:?} rot {ri} col {col}: fast has a placement, scan says blocked"
                                    );
                                }
                            }
                        }
                    }
                }
                if fast.len() != n + usize::from(!mism.is_empty()) {
                    eprintln!(
                        "COUNT MISMATCH {piece:?}: fast {} scan-side {} mism: {}",
                        fast.len(),
                        n,
                        mism
                    );
                    for row in board.to_strings() {
                        eprintln!("  {row}");
                    }
                    eprintln!("  heights: {:?}", board.heights());
                    for (ri, form) in forms.iter().enumerate() {
                        eprintln!("  rot {ri} cells {:?} w {}", form.cells, form.width);
                    }
                }
                assert_eq!(
                    fast.len(),
                    n + usize::from(!mism.is_empty()),
                    "count {piece:?}"
                );
                assert!(
                    mism.is_empty(),
                    "board:\n{:?}\n{}",
                    board.to_strings(),
                    mism
                );
            }
        }
    }

    /// A garbage-shaped board (deterministic, for the equivalence walk).
    fn garbage_like(seed: u64, fill_pct: u64) -> Board {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut b = Board::empty();
        for r in HEIGHT.saturating_sub(18)..HEIGHT {
            for c in 0..WIDTH {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                if x % 100 < fill_pct {
                    b.cells[r][c] = true;
                }
            }
        }
        b
    }

    #[test]
    fn scan_matches_the_per_feature_walks() {
        let mut boards = vec![
            Board::empty(),
            garbage_like(7, 55),
            garbage_like(8, 75),
            garbage_like(9, 90),
        ];
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..2000 {
            let p = 25 + (x % 70);
            let mut b = Board::empty();
            for r in 0..HEIGHT {
                for c in 0..WIDTH {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    if x % 100 < p {
                        b.cells[r][c] = true;
                    }
                }
            }
            boards.push(b);
        }
        for b in &boards {
            let s = b.scan();
            assert_eq!(s.heights, b.heights(), "heights");
            assert_eq!(s.holes as usize, b.hole_count(), "holes");
            // row/col transitions vs the reference walks (the lookahead/
            // rulebook wall conventions).
            let mut row_t = 0u32;
            for r in 0..HEIGHT {
                let mut prev = true;
                for c in 0..WIDTH {
                    let cur = b.cell(r, c);
                    if cur != prev {
                        row_t += 1;
                    }
                    prev = cur;
                }
                if !prev {
                    row_t += 1;
                }
            }
            let mut col_t = 0u32;
            for c in 0..WIDTH {
                let mut prev = false;
                for r in 0..HEIGHT {
                    let cur = b.cell(r, c);
                    if cur != prev {
                        col_t += 1;
                    }
                    prev = cur;
                }
                if !prev {
                    col_t += 1;
                }
            }
            assert_eq!(s.row_trans, row_t, "row_trans");
            assert_eq!(s.col_trans, col_t, "col_trans");
            // hole cover vs the rulebook walk (filled above each hole).
            let h = s.heights;
            let mut cover = 0usize;
            for c in 0..WIDTH {
                let top = HEIGHT - h[c];
                let mut filled_above = 0usize;
                for r in top..HEIGHT {
                    if b.cell(r, c) {
                        filled_above += 1;
                    } else {
                        cover += filled_above;
                    }
                }
            }
            assert_eq!(s.hole_cover as usize, cover, "hole_cover");
        }
    }

    #[test]
    fn drop_rule_grammar_ids_round_trip() {
        for rule in [DropRule::DeepestFit, DropRule::FromTop] {
            assert_eq!(DropRule::from_grammar(rule.grammar_id()), Some(rule));
        }
        assert_eq!(DropRule::from_grammar("laya-tetris-v1"), None);
    }

    #[test]
    fn hole_and_clear_counting() {
        // Floor of three + a floating cell above col 1 → one hole at (18, 1).
        let mut b = Board::empty();
        b.place(&[(19, 0), (19, 1), (19, 2), (17, 1)]);
        assert_eq!(b.hole_count(), 1);
        // Fill row 19 completely → one full row.
        let mut after = b.clone();
        after.place(&[
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
        ]);
        assert_eq!(after.full_rows().len(), 1);
        // Clearing shifts whole rows down one: the floater lands at (18, 1)
        // and the cleared floor leaves (19, 1) empty beneath it — a
        // bottom-edge hole is LEGAL under the whole-row-shift mechanic
        // (real Tetris drops by exactly one row per cleared line, not to
        // contact). That mechanic is what this pins.
        let rows = after.full_rows();
        after.clear_rows(&rows);
        assert!(after.cell(18, 1));
        assert_eq!(after.hole_count(), 1);
    }

    #[test]
    fn spot_sentences_are_closed_grammar() {
        // Pinned prefix, terminal period, and NO digits anywhere (words
        // only — the model cannot read numbers).
        let b = Board::empty();
        for piece in Piece::ALL {
            for p in landing_options(&b, piece) {
                let f = outcome_features(&b, &p);
                let s = render_spot_sentence(&b, &p, &f);
                assert!(s.starts_with("The piece "), "{s}");
                assert!(s.ends_with('.'), "{s}");
                assert!(!s.chars().any(|c| c.is_ascii_digit()), "{s}");
            }
        }
    }

    #[test]
    fn their_recorded_example_sentence_renders() {
        // The page's example prefix, producible under the v2 grammar
        // (their two-clause shape — `The piece leaves one hole under it and
        // makes a small bump on top` — widened with side + height clauses):
        // an S piece flat on a flat floor (one open shaft column keeps the
        // floor pre-clear) leaves exactly one hole under its top overhang,
        // rests on the left edge (span 0–2), sits 2 above the flanking
        // surface (Small band), and the resulting stack stands medium
        // (max height 8).
        let mut b = Board::empty();
        for c in 0..WIDTH - 1 {
            for r in 14..HEIGHT {
                b.place(&[(r, c)]);
            }
        }
        let s_flat = Piece::S.rotations()[0].clone();
        let p = hard_drop(&b, &s_flat, 0).expect("fits");
        let f = outcome_features(&b, &p);
        assert_eq!(f.holes_delta, 1, "one hole under the overhang");
        assert_eq!(f.lines_cleared, 0);
        assert_eq!(f.max_height, 8);
        assert_eq!(
            render_spot_sentence(&b, &p, &f),
            "The piece leaves one hole under it on the left edge, makes a small bump on top, and the stack stands medium."
        );
    }

    #[test]
    fn state_sentence_has_no_digits() {
        let mut b = Board::empty();
        for c in 0..6 {
            for r in 15..HEIGHT {
                b.place(&[(r, c)]);
            }
        }
        let s = render_state_sentence(&b, Piece::T);
        assert!(!s.chars().any(|c| c.is_ascii_digit()), "{s}");
        assert!(s.contains("T shaped piece is falling"), "{s}");
    }

    #[test]
    fn v4_preview_sentence_extends_the_v3_render_verbatim() {
        // The v4 render is the v3 render + exactly one appended sentence
        // (plan 609 T1.2); words only, and the base render is untouched.
        let mut b = Board::empty();
        for c in 0..6 {
            for r in 15..HEIGHT {
                b.place(&[(r, c)]);
            }
        }
        let v3 = render_state_sentence(&b, Piece::T);
        let v4 = render_state_sentence_with_preview(&b, Piece::T, Piece::L);
        assert!(v4.starts_with(&v3), "{v4:?} must extend {v3:?}");
        assert_eq!(
            &v4[v3.len()..],
            " The next piece is the right leaning ell piece."
        );
        assert!(!v4.chars().any(|c| c.is_ascii_digit()), "{v4}");
        // Every piece's preview is a pure suffix of its own v3 render.
        for next in Piece::ALL {
            let s = render_state_sentence_with_preview(&b, Piece::T, next);
            assert!(s.starts_with(&v3));
            assert!(s.contains(&format!("The next piece is the {} piece.", next.spoken())));
        }
    }
}
