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
//! Run: `cargo run --release --example tetris_05_lookahead_poc [-- <games> <cap> <garbage_rows> <fill_pct>]`
//!
//! Recorded context (reflex-site arena, engine `00aa6221`, T12): the fitted
//! modelless head played 140 pts / 3 lines / 46 pieces; laya played
//! 660 pts / 11 lines / 70 pieces @ p50 394 ms per piece. This POC is the
//! search-based answer.

#[path = "common/tetris_sim.rs"]
mod tetris_sim;
#[path = "common/tetris_lookahead.rs"]
mod tetris_lookahead;

use tetris_lookahead::{
    apply, garbage_board, pick, Bag, Player, LINES_SCORE, W_COL_TRANS, W_DEEP_WELL, W_HOLES,
    W_LINES, W_MAX_H, W_ROW_TRANS, W_WELLS,
};
use tetris_sim::{landing_options_with, Board, DropRule};

// ── The game loop ────────────────────────────────────────────────────────

const MAX_PIECES: usize = 500;
/// Default garbage fill. NOTE (Issue 892): Bench 891's prose says "@ 85%";
/// the committed code at `bb7aa12b6` ran 75% — the fill is a CLI arg now so
/// the record states which one produced each row.
const DEFAULT_FILL_PCT: u64 = 75;

fn play(
    seed: u64,
    player: Player,
    cap: usize,
    garbage_rows: usize,
    fill_pct: u64,
) -> (u64, u32, usize) {
    let mut bag = Bag::new(seed);
    let mut board = if garbage_rows > 0 {
        garbage_board(seed, garbage_rows, fill_pct)
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
    let fill_pct: u64 = std::env::args()
        .nth(4)
        .and_then(|a| a.parse().ok())
        .unwrap_or(DEFAULT_FILL_PCT);

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
        "physics: FromTop (real hard drop); guideline 7-bag, seeded; scoring 40/100/300/1200; cap {cap} pieces; garbage-start {garbage_rows} rows @ {fill_pct}% fill"
    );
    println!();

    let players = Player::ALL;
    println!(
        "{:<38} {:>6} {:>7} {:>7} {:>8} {:>8} {:>9}",
        "player", "games", "survived", "lines", "lines/g", "points/g", "pieces/g"
    );
    for p in players {
        let (mut surv, mut lines_t, mut pts_t, mut pieces_t) = (0usize, 0u32, 0u64, 0usize);
        for seed in 1..=games as u64 {
            let (pts, lines, pieces) = play(seed, p, cap, garbage_rows, fill_pct);
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
