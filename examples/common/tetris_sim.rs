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
    pub const ALL: [Self; 7] = [Self::I, Self::O, Self::T, Self::S, Self::Z, Self::J, Self::L];

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
    pub fn rotations(&self) -> Vec<Vec<(usize, usize)>> {
        let base: Vec<(usize, usize)> = match self {
            Self::I => vec![(1, 0), (1, 1), (1, 2), (1, 3)],
            Self::O => vec![(0, 1), (0, 2), (1, 1), (1, 2)],
            Self::T => vec![(0, 1), (1, 0), (1, 1), (1, 2)],
            Self::S => vec![(0, 1), (0, 2), (1, 0), (1, 1)],
            Self::Z => vec![(0, 0), (0, 1), (1, 1), (1, 2)],
            Self::J => vec![(0, 0), (1, 0), (1, 1), (1, 2)],
            Self::L => vec![(0, 2), (1, 0), (1, 1), (1, 2)],
        };
        let dim = 4usize; // rotation inside a 4x4 box
        let mut cur = base;
        let mut out: Vec<Vec<(usize, usize)>> = Vec::with_capacity(4);
        for _ in 0..4 {
            let norm = normalize(&cur);
            if !out.contains(&norm) {
                out.push(norm);
            }
            cur = rotate(&cur, dim);
        }
        out
    }
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
    /// Absolute resting cells (row, col), sorted.
    pub cells: Vec<(usize, usize)>,
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
        DropRule::FromTop => {
            if !fits(board, cells, 0, col) {
                return None;
            }
            let mut row = 0;
            while row + 1 < HEIGHT && fits(board, cells, row + 1, col) {
                row += 1;
            }
            Some(row)
        }
    }?;
    Some(Placement {
        rot: 0, // caller owns the rotation index
        col,
        row: rest,
        cells: {
            let mut v: Vec<(usize, usize)> =
                cells.iter().map(|&(dy, dx)| (rest + dy, col + dx)).collect();
            v.sort_unstable();
            v
        },
    })
}

/// Enumerate every distinct hard-drop landing for `piece` on `board` under
/// the pinned v2 rule ([`DropRule::DeepestFit`]), in pinned order: rotation
/// ascending, then column ascending. This is the option ORDER the fixture
/// freezes — the oracle's and the arena's argmax index both refer to it.
pub fn landing_options(board: &Board, piece: Piece) -> Vec<Placement> {
    landing_options_with(board, piece, DropRule::DeepestFit)
}

/// [`landing_options`] under an explicit drop rule (same pinned order).
pub fn landing_options_with(board: &Board, piece: Piece, rule: DropRule) -> Vec<Placement> {
    let mut out = Vec::with_capacity(34);
    for (ri, cells) in piece.rotations().iter().enumerate() {
        let width = cells.iter().map(|&(_, dx)| dx).max().unwrap_or(0) + 1;
        for col in 0..=(WIDTH - width) {
            if let Some(mut p) = hard_drop_with(board, cells, col, rule) {
                p.rot = ri;
                out.push(p);
            }
        }
    }
    out.shrink_to_fit();
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
        let right = if c == WIDTH - 1 { usize::MAX } else { heights[c + 1] };
        if left > h && right > h {
            let d = left.min(right) - h;
            cumulative_wells += (d * (d + 1) / 2) as u32;
        }
    }

    let eroded_cells = p
        .cells
        .iter()
        .filter(|&&(r, _)| full.contains(&r))
        .count() as u32;

    let landing_height =
        p.cells.iter().map(|&(r, _)| (HEIGHT - r) as f32).sum::<f32>() / p.cells.len() as f32;

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
    -9.348_696,  // col transitions
    -7.899_265_3,  // holes
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
    format!("The piece {holes_clause} under it {side_clause}, {surface_clause}, and {height_clause}{clears_clause}.")
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
            assert!(a.iter().zip(&b).all(|(x, y)| x.row == y.row && x.cells == y.cells));
        }
        // A blocked top row: no v3 landing in that column.
        let mut b = Board::empty();
        b.place(&[(0, 0), (1, 0)]);
        let cells = Piece::O.rotations()[0].clone();
        assert!(hard_drop_with(&b, &cells, 0, DropRule::FromTop).is_none());
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
