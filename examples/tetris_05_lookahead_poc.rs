//! tetris_05_lookahead_poc — the owner-directive POC (2026-09-25): stop
//! imitating laya and OUT-PLAY it. Depth-2 lookahead over the next piece
//! ("try every next-next combination, one step ahead of laya"), plus the
//! owner's strategy shaping: deep wells are filled by WHATEVER piece fits
//! (superlinear urgency — the I may never come), and compact placements
//! beat random ones by construction of the evaluation.
//!
//! Three players, one shared terminal-board evaluation (so each delta is
//! isolated):
//! - **ply1-classic**: 1-ply greedy, the Lee-style Dellacherie-class weights.
//! - **ply1-shaped**: 1-ply greedy + the deep-well urgency shaping.
//! - **ply2-shaped**: depth-2 exhaustive lookahead (every current placement ×
//!   every next-piece placement, ~900 evaluated continuations per decision)
//!   + shaping — the "reflexer" candidate.
//!
//! Same seeded 7-bag piece sequence for every player per seed. Real hard
//! drop (`FromTop` — the v3 physics). Guideline scoring (40/100/300/1200).
//!
//! Run: `cargo run --release --example tetris_05_lookahead_poc [-- <games>]`
//!
//! Recorded context (reflex-site arena, engine `00aa6221`, T12): the fitted
//! modelless head played 140 pts / 3 lines / 46 pieces; laya played
//! 660 pts / 11 lines / 70 pieces @ p50 394 ms per piece. This POC is the
//! search-based answer.

#[path = "common/tetris_sim.rs"]
mod tetris_sim;

use tetris_sim::{Board, DropRule, Piece, HEIGHT, WIDTH};

// ── Terminal-board evaluation (shared; disclosed) ────────────────────────
// Lee-style 1-ply weights over BOARD features (no landing height — the
// 2-ply player judges final boards, and the shared eval is what isolates
// the lookahead delta from the weight delta).

const W_LINES: f64 = 5.0;
const W_ROW_TRANS: f64 = -3.2;
const W_COL_TRANS: f64 = -9.3;
const W_HOLES: f64 = -7.9;
const W_WELLS: f64 = -3.4;
/// The owner's strategy: a well deeper than 2 left open costs SUPERLINEARLY
/// — fill it with whatever piece fits now ("even for one line"), because the
/// long piece may never come.
const W_DEEP_WELL: f64 = -2.0;
const W_MAX_H: f64 = -0.1;

struct BoardFeats {
    lines: u32,
    row_trans: f64,
    col_trans: f64,
    holes: f64,
    wells: f64,
    deep_well: f64,
    max_h: f64,
}

fn board_features(b: &Board, lines_cleared: u32) -> BoardFeats {
    let h = b.heights();
    // Row transitions (walls filled), column transitions (open ceiling,
    // filled floor).
    let mut row_trans = 0.0;
    for r in 0..HEIGHT {
        let mut prev = true; // wall
        for c in 0..WIDTH {
            let cur = b.cell(r, c);
            if cur != prev {
                row_trans += 1.0;
            }
            prev = cur;
        }
        if !prev {
            row_trans += 1.0; // right wall
        }
    }
    let mut col_trans = 0.0;
    for c in 0..WIDTH {
        let mut prev = false; // open ceiling
        for r in 0..HEIGHT {
            let cur = b.cell(r, c);
            if cur != prev {
                col_trans += 1.0;
            }
            prev = cur;
        }
        if !prev {
            col_trans += 1.0; // floor
        }
    }
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
        row_trans,
        col_trans,
        holes: b.hole_count() as f64,
        wells,
        deep_well,
        max_h: *h.iter().max().unwrap_or(&0) as f64,
    }
}

fn eval_board(f: &BoardFeats, shaped: bool) -> f64 {
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
fn apply(board: &Board, cells: &[(usize, usize)]) -> (Board, u32) {
    let mut b = board.clone();
    b.place(cells);
    let full = b.full_rows();
    let n = full.len() as u32;
    b.clear_rows(&full);
    (b, n)
}

// ── Players ──────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Player {
    Ply1Classic,
    Ply1Shaped,
    Ply2Shaped,
}

impl Player {
    fn name(self) -> &'static str {
        match self {
            Self::Ply1Classic => "ply1-classic",
            Self::Ply1Shaped => "ply1-shaped",
            Self::Ply2Shaped => "ply2-shaped (the reflexer candidate)",
        }
    }
    fn shaped(self) -> bool {
        !matches!(self, Self::Ply1Classic)
    }
    fn depth2(self) -> bool {
        matches!(self, Self::Ply2Shaped)
    }
}

fn pick(board: &Board, piece: Piece, next: Piece, player: Player) -> Option<usize> {
    use tetris_sim::landing_options_with;
    let options = landing_options_with(board, piece, DropRule::FromTop);
    if options.is_empty() {
        return None; // top out
    }
    let shaped = player.shaped();
    let mut best_i = 0usize;
    let mut best_s = f64::NEG_INFINITY;
    for (i, p) in options.iter().enumerate() {
        let (b1, l1) = apply(board, &p.cells);
        let s1 = if player.depth2() {
            // Depth-2: the value of p is the BEST continuation over every
            // next-piece placement ("not blocking the next next one" is
            // native to this — blocking placements score through their
            // worst-forced continuation).
            let opts2 = landing_options_with(&b1, next, DropRule::FromTop);
            if opts2.is_empty() {
                f64::NEG_INFINITY / 2.0 // forced top-out next piece
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

struct Bag {
    rng: fastrand::Rng,
    queue: Vec<Piece>,
}

impl Bag {
    fn new(seed: u64) -> Self {
        Self {
            rng: fastrand::Rng::with_seed(seed),
            queue: Vec::with_capacity(7),
        }
    }
    fn draw(&mut self) -> Piece {
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
}

// ── The game loop ────────────────────────────────────────────────────────

const MAX_PIECES: usize = 500;
const LINES_SCORE: [u64; 5] = [0, 40, 100, 300, 1200];

/// A rugged garbage-start board: the bottom `rows` rows filled with prob
/// `p`% per cell — natural deep wells + covered holes, the regime where the
/// owner's shaping ("fill it even for one line; the I may never come") has
/// something to bite on.
fn garbage_board(seed: u64, rows: usize, p: u64) -> Board {
    let mut rng = fastrand::Rng::with_seed(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut b = Board::empty();
    let start = HEIGHT.saturating_sub(rows);
    for r in start..HEIGHT {
        for c in 0..WIDTH {
            if rng.u64(0..100) < p {
                b.place(&[(r, c)]);
            }
        }
    }
    b
}

fn play(seed: u64, player: Player, cap: usize, garbage_rows: usize) -> (u64, u32, usize) {
    let mut bag = Bag::new(seed);
    let mut board = if garbage_rows > 0 {
        garbage_board(seed, garbage_rows, 75)
    } else {
        Board::empty()
    };
    let mut next = bag.draw();
    let (mut points, mut lines) = (0u64, 0u32);
    let mut pieces = 0usize;
    while pieces < cap {
        let cur = next;
        next = bag.draw();
        let Some(i) = pick(&board, cur, next, player) else {
            break; // topped out
        };
        use tetris_sim::landing_options_with;
        let options = landing_options_with(&board, cur, DropRule::FromTop);
        let (b1, l1) = apply(&board, &options[i].cells);
        board = b1;
        lines += l1;
        points += LINES_SCORE[l1.min(4) as usize];
        pieces += 1;
    }
    (points, lines, pieces)
}

fn main() {
    let games: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(10);
    let cap: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(MAX_PIECES);
    let garbage_rows: usize = std::env::args()
        .nth(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0);

    println!("== tetris_05_lookahead_poc — depth-2 next-piece lookahead ==");
    println!(
        "players share ONE terminal eval (Lee-style weights: lines {W_LINES:+}, row_trans {W_ROW_TRANS}, \
         col_trans {W_COL_TRANS}, holes {W_HOLES}, wells {W_WELLS}, max_h {W_MAX_H})"
    );
    println!(
        "owner shaping (shaped players only): deep-well urgency — a well deeper than 2 left open \
         costs {W_DEEP_WELL} × (depth−2)²  (\"fill it even for one line; the I may never come\")"
    );
    println!(
        "physics: FromTop (real hard drop); guideline 7-bag, seeded; scoring 40/100/300/1200; cap {cap} pieces; garbage-start {garbage_rows} rows @ 75% fill"
    );
    println!();

    let players = [
        Player::Ply1Classic,
        Player::Ply1Shaped,
        Player::Ply2Shaped,
    ];
    println!(
        "{:<38} {:>6} {:>7} {:>7} {:>8} {:>8} {:>9}",
        "player", "games", "survived", "lines", "lines/g", "points/g", "pieces/g"
    );
    for p in players {
        let (mut surv, mut lines_t, mut pts_t, mut pieces_t) = (0usize, 0u32, 0u64, 0usize);
        for seed in 1..=games as u64 {
            let (pts, lines, pieces) = play(seed, p, cap, garbage_rows);
            surv += usize::from(pieces == cap);
            lines_t += lines;
            pts_t += pts;
            pieces_t += pieces;
        }
        println!(
            "{:<38} {:>6} {:>7} {:>7} {:>8.1} {:>8.0} {:>9.0}",
            p.name(),
            games,
            format!("{surv}/{games}"),
            lines_t,
            lines_t as f64 / games as f64,
            pts_t as f64 / games as f64,
            pieces_t as f64 / games as f64
        );
    }
    println!();
    println!("context (reflex-site arena T12, engine 00aa6221): fitted modelless head 140 pts / 3 lines / 46 pieces · laya 660 pts / 11 lines / 70 pieces @ 394 ms/piece");
}
