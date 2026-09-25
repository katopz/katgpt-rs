//! Shared lookahead-player substrate for the tetris examples (extracted from
//! `tetris_05_lookahead_poc`, Issue 892): the Lee-style terminal-board eval,
//! the owner's deep-well shaping, the depth-1/depth-2 pickers, the seeded
//! 7-bag and the garbage-start board. One copy, so the POC, the rulebook
//! arena (`tetris_06_rulebook_arena`) and the laya head-to-head
//! (`tetris_07_laya_h2h`) all measure the SAME player. One crate copy since
//! Issue 893 — the `#[path]` per-example inclusion era is over; this module
//! reaches the sim via `crate::sim`.

#![allow(dead_code)] // each example consumes a different slice

use crate::sim::{Board, DropRule, HEIGHT, Piece, WIDTH, landing_options_with};

// ── Terminal-board evaluation (shared; disclosed) ────────────────────────

pub const W_LINES: f64 = 5.0;
pub const W_ROW_TRANS: f64 = -3.2;
pub const W_COL_TRANS: f64 = -9.3;
pub const W_HOLES: f64 = -7.9;
pub const W_WELLS: f64 = -3.4;
/// The owner's strategy: a well deeper than 2 left open costs SUPERLINEARLY
/// — fill it with whatever piece fits now ("even for one line"), because the
/// long piece may never come.
pub const W_DEEP_WELL: f64 = -2.0;
pub const W_MAX_H: f64 = -0.1;

pub struct BoardFeats {
    pub lines: u32,
    pub row_trans: f64,
    pub col_trans: f64,
    pub holes: f64,
    pub wells: f64,
    pub deep_well: f64,
    pub max_h: f64,
}

pub fn board_features(b: &Board, lines_cleared: u32) -> BoardFeats {
    // Fused two-pass scan (identical values to the historical per-feature
    // walks — heights/wells read the same height array, transitions count
    // the same wall conventions).
    let s = b.scan();
    let h = &s.heights;
    // Cumulative wells: per column, depth = max(0, min(neighbors) − h);
    // walls count at full height (the edge I-slot is a well like any other).
    let mut wells = 0.0;
    let mut deep_well = 0.0;
    for c in 0..WIDTH {
        let l = if c == 0 { HEIGHT } else { h[c - 1] };
        let r = if c + 1 == WIDTH { HEIGHT } else { h[c + 1] };
        let d = l.min(r).saturating_sub(h[c]);
        wells += d as f64;
        if d > 2 {
            deep_well += ((d - 2) * (d - 2)) as f64;
        }
    }
    BoardFeats {
        lines: lines_cleared,
        row_trans: s.row_trans as f64,
        col_trans: s.col_trans as f64,
        holes: s.holes as f64,
        wells,
        deep_well,
        max_h: *h.iter().max().unwrap_or(&0) as f64,
    }
}

pub fn eval_board(f: &BoardFeats, shaped: bool) -> f64 {
    let mut s = W_LINES * f.lines as f64
        + W_ROW_TRANS * f.row_trans
        + W_COL_TRANS * f.col_trans
        + W_HOLES * f.holes
        + W_WELLS * f.wells
        + W_MAX_H * f.max_h;
    if shaped {
        s += W_DEEP_WELL * f.deep_well;
    }
    s
}

/// Apply a placement and clear complete rows. Returns the next board and
/// the number of rows cleared.
pub fn apply(board: &Board, cells: &[(usize, usize)]) -> (Board, u32) {
    let mut b = board.clone();
    let n = b.place_and_clear(cells);
    (b, n)
}

// ── The Bench-891 players ────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Player {
    Ply1Classic,
    Ply1Shaped,
    Ply2Shaped,
}

impl Player {
    pub const ALL: [Self; 3] = [Self::Ply1Classic, Self::Ply1Shaped, Self::Ply2Shaped];

    pub fn name(self) -> &'static str {
        match self {
            Self::Ply1Classic => "ply1-classic",
            Self::Ply1Shaped => "ply1-shaped",
            Self::Ply2Shaped => "ply2-shaped (the reflexer candidate)",
        }
    }
    pub fn shaped(self) -> bool {
        !matches!(self, Self::Ply1Classic)
    }
    pub fn depth2(self) -> bool {
        matches!(self, Self::Ply2Shaped)
    }
}

/// Index into `landing_options_with(board, piece, FromTop)` of the chosen
/// placement; `None` = top out.
pub fn pick(board: &Board, piece: Piece, next: Piece, player: Player) -> Option<usize> {
    let options = landing_options_with(board, piece, DropRule::FromTop);
    if options.is_empty() {
        return None;
    }
    let shaped = player.shaped();
    let mut best_i = 0usize;
    let mut best_s = f64::NEG_INFINITY;
    for (i, p) in options.iter().enumerate() {
        let (b1, l1) = apply(board, &p.cells);
        let s1 = if player.depth2() {
            // Depth-2: the value of p is the BEST continuation over every
            // next-piece placement ("not blocking the next next one" is
            // native — a blocking placement scores through its own forced
            // continuation).
            let opts2 = landing_options_with(&b1, next, DropRule::FromTop);
            if opts2.is_empty() {
                f64::NEG_INFINITY / 2.0
            } else {
                let mut best_q = f64::NEG_INFINITY;
                for q in &opts2 {
                    let (b2, l2) = apply(&b1, &q.cells);
                    let f = board_features(&b2, l1 + l2);
                    best_q = best_q.max(eval_board(&f, shaped));
                }
                best_q
            }
        } else {
            let f = board_features(&b1, l1);
            eval_board(&f, shaped)
        };
        if s1 > best_s {
            best_s = s1;
            best_i = i;
        }
    }
    Some(best_i)
}

// ── The seeded 7-bag ─────────────────────────────────────────────────────

/// Guideline 7-bag, seeded. `remaining()` exposes the bag contents — the
/// EXACT chance distribution of the next unseen piece (uniform over what is
/// left in the current bag; a fresh full bag once it empties).
pub struct Bag {
    rng: fastrand::Rng,
    queue: Vec<Piece>,
}

impl Bag {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: fastrand::Rng::with_seed(seed),
            queue: Vec::with_capacity(7),
        }
    }
    pub fn draw(&mut self) -> Piece {
        if self.queue.is_empty() {
            let mut bag = Piece::ALL;
            for k in (1..7).rev() {
                let j = self.rng.usize(0..=k);
                bag.swap(k, j);
            }
            self.queue.extend_from_slice(&bag);
        }
        self.queue.remove(0)
    }
    /// Pieces still in the current bag (the support of the next unseen
    /// draw). Empty ⇒ the next draw opens a fresh full bag.
    pub fn remaining(&self) -> &[Piece] {
        &self.queue
    }
}

/// Guideline line-clear scoring (40/100/300/1200).
pub const LINES_SCORE: [u64; 5] = [0, 40, 100, 300, 1200];

/// A rugged garbage-start board: the bottom `rows` rows filled with prob
/// `fill_pct`% per cell — natural deep wells + covered holes.
pub fn garbage_board(seed: u64, rows: usize, fill_pct: u64) -> Board {
    let mut rng = fastrand::Rng::with_seed(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut b = Board::empty();
    let start = HEIGHT.saturating_sub(rows);
    for r in start..HEIGHT {
        for c in 0..WIDTH {
            if rng.u64(0..100) < fill_pct {
                b.place(&[(r, c)]);
            }
        }
    }
    b
}
