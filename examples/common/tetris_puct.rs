//! Tetris adapter for the generic chance-node PUCT (`katgpt_core::chance_puct`,
//! Issue 892 T3) — the moka trick on a 7-bag.
//!
//! Include AFTER the sim and the lookahead module:
//! ```ignore
//! #[path = "common/tetris_sim.rs"] mod tetris_sim;
//! #[path = "common/tetris_lookahead.rs"] mod tetris_lookahead;
//! #[path = "common/tetris_puct.rs"] mod tetris_puct;
//! ```
//!
//! # The information structure (what a node knows)
//!
//! A DECISION state is `(board, cur, preview, bag)` — exactly what the player
//! sees: the piece to place and the one-piece preview. Placing `cur` gives a
//! CHANCE (post-decision) state whose single random event is the NEW preview:
//! the old preview becomes the piece to place, and the new one is drawn
//! uniformly from the 7-bag remainder (a fresh full bag once it empties).
//! So the root's known `next` is never a chance event — it rides the state
//! — and every deeper decision sees its own preview, as the real game does.
//! (A bag with one piece left is the `p = 1` chance node.)
//!
//! Prior score of a placement = the caller's 1-ply `eval` of the board it
//! leaves (lines cleared along the path included); leaf value = `eval` of
//! the node's board. A decision state with no landing is a top-out = loss.
//! The `eval` closure is the seam the rulebook (T1) plugs into.

#![allow(dead_code)]

use crate::tetris_lookahead::apply;
use crate::tetris_sim::{Board, DropRule, Piece, landing_options_with};
use katgpt_core::chance_puct::{ChanceGame, ChancePuct};

pub use katgpt_core::chance_puct::ChancePuctConfig as PuctCfg;

/// Board evaluation: `(board, lines cleared along the path) → score`.
pub type Eval<'e> = &'e dyn Fn(&Board, u32) -> f64;

const FULL_BAG: u8 = 0x7F;

#[inline]
fn bit(p: Piece) -> u8 {
    1 << Piece::ALL
        .iter()
        .position(|&q| q == p)
        .expect("piece in ALL")
}

/// A placement, inline (a tetromino is always 4 cells).
#[derive(Clone, Copy, Debug)]
pub struct Move {
    /// Index into `landing_options_with(board, cur, FromTop)`.
    pub idx: u16,
    pub cells: [(u8, u8); 4],
}

#[derive(Clone)]
pub struct TetrisState<'e> {
    board: Board,
    /// Decision: the piece to place. Chance: unused (the placed piece).
    cur: Piece,
    /// Decision: the visible preview. Chance: the piece to place next.
    preview: Piece,
    /// Pieces left in the current bag (0 ⇒ the next draw opens a full bag).
    bag: u8,
    lines: u32,
    after: bool,
    eval: Eval<'e>,
}

impl<'e> TetrisState<'e> {
    pub fn root(
        board: &Board,
        cur: Piece,
        next: Piece,
        bag_remaining: &[Piece],
        eval: Eval<'e>,
    ) -> Self {
        Self {
            board: board.clone(),
            cur,
            preview: next,
            bag: bag_remaining.iter().fold(0u8, |m, &p| m | bit(p)),
            lines: 0,
            after: false,
            eval,
        }
    }
}

impl ChanceGame for TetrisState<'_> {
    type Action = Move;
    type Outcome = Piece;

    fn is_terminal(&self) -> bool {
        // Consulted only for decision states with no actions (and for
        // chance states, which are never a loss under FromTop).
        !self.after && landing_options_with(&self.board, self.cur, DropRule::FromTop).is_empty()
    }

    fn value(&self) -> f32 {
        (self.eval)(&self.board, self.lines) as f32
    }

    fn actions(&self, out: &mut Vec<(Move, f32)>) {
        if self.after {
            return;
        }
        for (i, p) in landing_options_with(&self.board, self.cur, DropRule::FromTop)
            .iter()
            .enumerate()
        {
            let (b1, l1) = apply(&self.board, &p.cells);
            let mut cells = [(0u8, 0u8); 4];
            for (dst, &(r, c)) in cells.iter_mut().zip(&p.cells) {
                *dst = (r as u8, c as u8);
            }
            out.push((
                Move {
                    idx: i as u16,
                    cells,
                },
                (self.eval)(&b1, self.lines + l1) as f32,
            ));
        }
    }

    fn apply(&self, m: Move) -> Self {
        let cells = m.cells.map(|(r, c)| (r as usize, c as usize));
        let (board, n) = apply(&self.board, &cells);
        Self {
            board,
            lines: self.lines + n,
            after: true,
            ..self.clone()
        }
    }

    fn outcomes(&self, out: &mut Vec<(Piece, f32)>) {
        let bag = if self.bag == 0 { FULL_BAG } else { self.bag };
        let p = 1.0 / bag.count_ones() as f32;
        out.extend(
            Piece::ALL
                .iter()
                .filter(|&&q| bag & bit(q) != 0)
                .map(|&q| (q, p)),
        );
    }

    fn resolve(&self, drawn: Piece) -> Self {
        let bag = if self.bag == 0 { FULL_BAG } else { self.bag };
        Self {
            cur: self.preview,
            preview: drawn,
            bag: bag & !bit(drawn),
            after: false,
            ..self.clone()
        }
    }
}

/// One PUCT decision. Returns an index into
/// `landing_options_with(board, cur, DropRule::FromTop)`; `None` = top out.
/// `bag_remaining` = the 7-bag contents after `next` was drawn (empty ⇒ a
/// fresh full bag). Builds a fresh searcher per call (the arena is not
/// reused across decisions here — the eval borrow lives in the state).
pub fn pick_puct(
    board: &Board,
    cur: Piece,
    next: Piece,
    bag_remaining: &[Piece],
    cfg: &PuctCfg,
    rng: &mut fastrand::Rng,
    eval: &dyn Fn(&Board, u32) -> f64,
) -> Option<usize> {
    let root = TetrisState::root(board, cur, next, bag_remaining, eval);
    let mut search = ChancePuct::new(*cfg);
    search.search(&root, rng).map(|p| p.index)
}
