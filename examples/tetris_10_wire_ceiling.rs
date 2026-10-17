//! tetris_10_wire_ceiling — riir-instinct Issue 009 T5: does the tetris
//! spot wire carry enough information for a champion-level decoder, or
//! must the wire widen before T7's distillation?
//!
//! THE QUESTION. The serving wire is text-only: one spot sentence per
//! option, decoding to FIVE coarse class ordinals (holes / side / surface
//! / height / clears — `examples/common/grammar_tables.rs`, the v2
//! grammar reflex-site renders), plus a state sentence (hw / holes2 /
//! piece, v2 shape). The reflex head is a LINEAR ridge over the five
//! ordinals. Issue 009 T5 asks whether the wire bounds a richer decoder,
//! or whether the Instinct lane must receive the raw afterstate grid.
//!
//! THE INSTRUMENT. One serving-matched full-feature reference vs six
//! wire-constrained decoders, all 1-ply, no preview, hold OFF, fresh bag:
//!
//! - `champ1ply` — the rulebook champion genome `68cae9d382014662`
//!   (`Genome::champion_hybrid`) with NextPreview OFF and `depth = 1`
//!   (`plies()` clamps to ≥2 while the preview rule is on, and the depth-2
//!   recursion injects `TOPOUT` (−1e12) values the wire arms can never
//!   predict): per-option `decide_scored` leaf values over the FULL
//!   afterstate. This is T8's baseline shape (the 1-ply champion
//!   evaluator) and the wire's information-ceiling reference.
//! - `wire-lin` — standardized ridge over the five ordinals (the reflex
//!   head's own shape), fit to predict champ1ply's per-option value;
//!   λ from `RIDGE_GRID` by leave-one-TRAINING-GAME-out MSE via
//!   sufficient-statistics subtraction (no row storage, deterministic).
//! - `wire-t1` — saturated cell-mean table over the 5-tuple (1500 cells):
//!   the MAXIMAL decoder over the option wire. Cell mean when support
//!   ≥ `TABLE_MIN`, else `wire-lin`.
//! - `wire-t2` — t1 ⊗ piece (10 500 cells): adds the state sentence's
//!   piece clause.
//! - `wire-t3` — t2 ⊗ hw ⊗ holes2 ⊗ tallest/lowest region (the full v2
//!   state sentence, ≤ 1 575 000 cells), hierarchical fallback
//!   t3 → t2 → t1 → lin. Each table arm falls back down its own chain,
//!   so every arm is a complete policy.
//! - `wire-t1d` / `wire-t3d` — the DEMEANED-table class (added 2026-09-27,
//!   the Issue-825 cross-review's owed decoder): cells bumped with
//!   `value − decision mean`, targeting within-point ranking directly.
//!   An absolute cell mean tracks position goodness as much as option
//!   rank; if ANY wire decoder carries the champion's ranking, these
//!   are the ones. Chains t1d → lin and t3d → t1d → lin.
//!
//! A THEOREM, but a NARROW one (scoped 2026-09-27, the Issue-825 cross-review):
//! within one decision point the state sentence and the piece are CONSTANT
//! across options, so no decoder that adds a state term to an ADDITIVE
//! option term can change its within-point ranking through them. It does
//! NOT bind decoders that combine the two (a keyed table t2/t3 does),
//! which is exactly what t2/t3 measure. And a cell-mean of ABSOLUTE values
//! tracks position goodness as much as option rank — the DEMEANED arms
//! (t1d/t3d, bumped with value − decision mean) are the table class that
//! targets within-point ranking directly.
//!
//! Protocol: train on champ1ply-driven games (the teacher's own state
//! distribution — the T7 distill flow), seeds 1..=40, regimes empty /
//! 16:75 / 18:75 garbage starts, cap 300. Evaluate on held seeds
//! 101..=140 (same regimes, cap 5000): board outcomes paired per seed,
//! sign tests, per-decision argmax agreement + the rank of champ1ply's
//! pick under each decoder (computed on champ1ply's own held decisions),
//! and cell occupancy. Seed 607 stays beside (the arena's pin) — never
//! used here.
//!
//! Pre-declared verdict rules (judged on `wire-t3`, the maximal wire
//! decoder; every arm reported):
//! - WIRE-SUFFICIENT — t3 argmax agreement ≥ 0.95 AND the t3-vs-champ1ply
//!   paired sign test on points p ≥ 0.05 AND mean lines ratio ≥ 0.95.
//! - WIRE-MUST-WIDEN — t3 agreement ≤ 0.85 OR (sign-test p < 0.05 AND
//!   mean points ratio ≤ 0.80).
//! - MIXED — anything between: adjudicate on the board outcomes (the
//!   product metric) before any wire redesign.
//!
//! Drift pins (asserted at start): the state-key computation mirrors
//! `sim::render_state_sentence` (clause-substring checks over real
//! boards — the same drift detector the corpus round-trip uses); the
//! champion id; the empty-board option counts (O=9, I=17); pack
//! round-trips incl. the max keys.
//!
//! Run: `cargo run --release --example tetris_10_wire_ceiling [--quick]`
//! (`--quick`: 6 train / 6 held seeds, caps 120/800 — a compile smoke,
//! never quoted). This probe measures ACCURACY/outcomes, not latency; no
//! timing number it prints is a latency claim.

// The sufficient-statistics and Gauss–Jordan loops are small fixed-width
// (5/6) matrix updates — indexed form is the honest one.
#![allow(clippy::needless_range_loop)]

use katgpt_tetris::lookahead as tetris_lookahead;
use katgpt_tetris::rulebook as tetris_rulebook;
use katgpt_tetris::sim as tetris_sim;

// grammar_tables consumes the whole sim family through the consumer's
// crate root (`crate::tetris_sim` etc.) — the shared single-instance law.
#[path = "common/flappy_sim.rs"]
mod flappy_sim;
#[path = "common/grammar_tables.rs"]
mod grammar_tables;
#[path = "common/lanes_sim.rs"]
mod lanes_sim;

use rayon::prelude::*;

use tetris_lookahead::{Bag, LINES_SCORE, apply, garbage_board};
use tetris_rulebook::{Genome, RuleId, View, decide_scored};
use tetris_sim::{
    Board, DropRule, Piece, landing_options, landing_options_with, outcome_features,
    render_state_sentence,
};

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// The champion's pinned genome id (Bench 892) — the SAME genome
/// riir-instinct Issue 009 names as the teacher evaluator.
const HYBRID_ID: &str = "68cae9d382014662";

/// Pinned λ grid (standardized scale) — the published fit recipe's grid.
const RIDGE_GRID: [f64; 4] = [1e-3, 1e-2, 1e-1, 1.0];
/// A cell needs this many training samples before its mean is trusted;
/// below it the arm falls down its chain. Pre-declared.
const TABLE_MIN: u32 = 4;

const REGIMES: [(usize, u64); 3] = [(0, 0), (16, 75), (18, 75)];

// ── wire packing ────────────────────────────────────────────────────────
// Order IS contract (the grammar's vocabulary order): holes, side,
// surface, height, clears.

fn pack_t1(t: &[u8; 5]) -> u16 {
    let (h, s, surf, ht, cl) = (t[0], t[1], t[2], t[3], t[4]);
    u16::from(cl)
        + 5 * (u16::from(ht) + 3 * (u16::from(surf) + 4 * (u16::from(s) + 5 * u16::from(h))))
}

fn unpack_t1(k: u16) -> [u8; 5] {
    let k = u32::from(k);
    [
        (k / 300) as u8,
        ((k / 60) % 5) as u8,
        ((k / 15) % 4) as u8,
        ((k / 5) % 3) as u8,
        (k % 5) as u8,
    ]
}

/// The state sentence's own key: piece + hw + holes2 + tallest/lowest
/// region (spread shape) — mirroring `render_state_sentence`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct StateKey {
    hw: u8,
    holes: u8,
    /// 1 = spread (tallest/lowest regions named), 0 = flat.
    spread: u8,
    tallest: u8,
    lowest: u8,
    piece: u8,
}

fn state_key(board: &Board, piece: Piece) -> StateKey {
    let h = board.heights();
    let max_h = h.iter().copied().max().unwrap_or(0);
    let hw = match max_h {
        0..=4 => 0u8,
        5..=10 => 1,
        _ => 2,
    };
    let avg = |r: std::ops::Range<usize>| r.clone().map(|c| h[c]).sum::<usize>() / r.len();
    let (l, m, r) = (avg(0..3), avg(3..7), avg(7..10));
    let span = l.max(m).max(r) - l.min(m).min(r);
    let (spread, tallest, lowest) = if span >= 3 {
        let mut v = [(l, 0u8), (m, 1u8), (r, 2u8)];
        v.sort_unstable_by_key(|&(x, _)| x);
        (1u8, v[2].1, v[0].1)
    } else {
        (0u8, 0, 0)
    };
    let holes = match board.hole_count() {
        0 => 0u8,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        _ => 4,
    };
    StateKey {
        hw,
        holes,
        spread,
        tallest,
        lowest,
        piece: piece.index() as u8,
    }
}

fn t2_key(t1k: u16, sk: StateKey) -> u16 {
    t1k * 7 + u16::from(sk.piece)
}

fn t3_key(t1k: u16, sk: StateKey) -> u32 {
    let sidepat = if sk.spread == 1 {
        1 + sk.tallest * 3 + sk.lowest
    } else {
        0
    };
    ((u32::from(t2_key(t1k, sk)) * 3 + u32::from(sk.hw)) * 5 + u32::from(sk.holes)) * 10
        + u32::from(sidepat)
}

// ── linear decoder (the reflex head's shape) ────────────────────────────

#[derive(Clone)]
struct Linear {
    w: [f64; 6],
    mean: [f64; 5],
    inv_std: [f64; 5],
}

impl Linear {
    fn predict(&self, t: &[u8; 5]) -> f64 {
        let mut acc = self.w[5];
        for i in 0..5 {
            acc += self.w[i] * (f64::from(t[i]) - self.mean[i]) * self.inv_std[i];
        }
        acc
    }
}

/// Per-training-game sufficient statistics (closed-form ridge without
/// row storage).
#[derive(Clone, Copy, Default)]
struct GameRows {
    n: f64,
    sx: [f64; 5],
    sxx: [[f64; 5]; 5],
    sy: f64,
    syy: f64,
    sxy: [f64; 5],
}

impl GameRows {
    fn row(&mut self, t: &[u8; 5], y: f64) {
        self.n += 1.0;
        self.sy += y;
        self.syy += y * y;
        for i in 0..5 {
            let xi = f64::from(t[i]);
            self.sx[i] += xi;
            self.sxy[i] += xi * y;
            for j in 0..5 {
                self.sxx[i][j] += xi * f64::from(t[j]);
            }
        }
    }
}

/// Gauss–Jordan with partial pivoting, 6×6.
fn solve6(m: &mut [[f64; 6]; 6], rhs: &mut [f64; 6]) -> Option<[f64; 6]> {
    for col in 0..6 {
        let mut piv = col;
        for r in col + 1..6 {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, piv);
        rhs.swap(col, piv);
        let d = m[col][col];
        for j in 0..6 {
            m[col][j] /= d;
        }
        rhs[col] /= d;
        for r in 0..6 {
            if r != col && m[r][col] != 0.0 {
                let f = m[r][col];
                for j in 0..6 {
                    m[r][j] -= f * m[col][j];
                }
                rhs[r] -= f * rhs[col];
            }
        }
    }
    let mut x = [0.0; 6];
    x.copy_from_slice(rhs);
    Some(x)
}

/// Per-game standardized sufficient statistics: (XᵀX, Xᵀy, Σy², n).
type GameMoments = ([[f64; 6]; 6], [f64; 6], f64, f64);

/// Fit the ridge; λ by leave-one-game-out MSE via sufficient-stats
/// subtraction. Returns (model, chosen λ, per-λ LOO MSE).
fn fit_linear(games: &[GameRows]) -> (Linear, f64, Vec<f64>) {
    let n_tot: f64 = games.iter().map(|s| s.n).sum();
    assert!(n_tot > 0.0, "no training rows");
    let mut mu = [0.0; 5];
    for s in games {
        for i in 0..5 {
            mu[i] += s.sx[i];
        }
    }
    for v in &mut mu {
        *v /= n_tot;
    }
    let mut var = [0.0; 5];
    for s in games {
        for i in 0..5 {
            var[i] += s.sxx[i][i] - s.n * mu[i] * mu[i];
        }
    }
    let inv_std: [f64; 5] = std::array::from_fn(|i| {
        let v = var[i] / (n_tot - 1.0).max(1.0);
        if v > 1e-12 { 1.0 / v.sqrt() } else { 1.0 }
    });

    // Per-game standardized (XᵀX, Xᵀy): the intercept column keeps each
    // game's Σx_std (NOT zero — μ is global); its diagonal is n.
    let standardized = |s: &GameRows| -> ([[f64; 6]; 6], [f64; 6]) {
        let mut a = [[0.0; 6]; 6];
        let mut b = [0.0; 6];
        for i in 0..5 {
            for j in 0..5 {
                a[i][j] = (s.sxx[i][j] - s.n * mu[i] * mu[j]) * inv_std[i] * inv_std[j];
            }
            a[i][5] = (s.sx[i] - s.n * mu[i]) * inv_std[i];
            a[5][i] = a[i][5];
            b[i] = (s.sxy[i] - mu[i] * s.sy) * inv_std[i];
        }
        a[5][5] = s.n;
        b[5] = s.sy;
        (a, b)
    };

    let mut a_tot = [[0.0; 6]; 6];
    let mut b_tot = [0.0; 6];
    let mut per: Vec<GameMoments> = Vec::with_capacity(games.len());
    for s in games {
        let (a, b) = standardized(s);
        for i in 0..6 {
            b_tot[i] += b[i];
            for j in 0..6 {
                a_tot[i][j] += a[i][j];
            }
        }
        per.push((a, b, s.syy, s.n));
    }

    let mut loo = Vec::with_capacity(RIDGE_GRID.len());
    for &lam in &RIDGE_GRID {
        let mut acc = 0.0f64;
        let mut used = 0.0f64;
        for (a_g, b_g, syy, n_g) in &per {
            let mut m = [[0.0; 6]; 6];
            let mut rhs = [0.0; 6];
            for i in 0..6 {
                for j in 0..6 {
                    m[i][j] = a_tot[i][j] - a_g[i][j] + f64::from(i == j) * lam;
                }
                rhs[i] = b_tot[i] - b_g[i];
            }
            if let Some(w) = solve6(&mut m, &mut rhs) {
                // SSE on the held-out game: wᵀA_g w − 2 wᵀb_g + Σy².
                let mut sse = *syy;
                for i in 0..6 {
                    sse -= 2.0 * w[i] * b_g[i];
                    for j in 0..6 {
                        sse += w[i] * a_g[i][j] * w[j];
                    }
                }
                acc += sse / n_g.max(1.0);
                used += 1.0;
            }
        }
        loo.push(acc / used.max(1.0));
    }

    let best = loo
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(RIDGE_GRID.len() - 1);
    let lam = RIDGE_GRID[best];
    let mut m = a_tot;
    let mut rhs = b_tot;
    for i in 0..6 {
        m[i][i] += lam * f64::from(i < 5); // intercept unpenalized
    }
    let w = solve6(&mut m, &mut rhs).expect("full-data ridge solve");
    (
        Linear {
            w,
            mean: mu,
            inv_std,
        },
        lam,
        loo,
    )
}

// ── the decoder family ──────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum WireArm {
    Lin,
    T1,
    T2,
    T3,
    /// Demeaned-target t1: cells bumped with `value − decision mean` — the
    /// table class that targets within-point ranking directly (an absolute
    /// cell mean tracks position goodness as much as option rank). Chain
    /// t1d → lin; the rare lin fallback (t1 is dense) mixes scales for that
    /// option only — disclosed via the level-usage line.
    T1D,
    /// Demeaned-target t3: chain t3d → t1d → lin.
    T3D,
}

const WIRE_ARMS: [WireArm; 6] = [
    WireArm::Lin,
    WireArm::T1,
    WireArm::T2,
    WireArm::T3,
    WireArm::T1D,
    WireArm::T3D,
];

impl WireArm {
    fn name(self) -> &'static str {
        match self {
            WireArm::Lin => "wire-lin",
            WireArm::T1 => "wire-t1",
            WireArm::T2 => "wire-t2",
            WireArm::T3 => "wire-t3",
            WireArm::T1D => "wire-t1d",
            WireArm::T3D => "wire-t3d",
        }
    }
    /// The highest table level the arm may consult (Lin = 0). The demeaned
    /// arms reuse levels 1/3; `demeaned` says which table family to read.
    fn max_level(self) -> u8 {
        match self {
            WireArm::Lin => 0,
            WireArm::T1 | WireArm::T1D => 1,
            WireArm::T2 => 2,
            WireArm::T3 | WireArm::T3D => 3,
        }
    }
    fn demeaned(self) -> bool {
        matches!(self, WireArm::T1D | WireArm::T3D)
    }
    /// Index into the rank/level arrays (the WIRE_ARMS order).
    fn idx(self) -> usize {
        WIRE_ARMS
            .iter()
            .position(|&w| w == self)
            .expect("arm in WIRE_ARMS")
    }
}

#[derive(Default, Clone)]
struct Cell {
    sum: f64,
    n: u32,
}

/// All six decoders share one trained state; each arm reads its own
/// chain down to the linear model. Immutable after training (Sync).
/// `t1d`/`t3d` carry the DEMEANED targets (value − decision mean); the
/// absolute tables are untouched, so the original arms reproduce exactly.
struct Decoder {
    t1: HashMap<u16, Cell>,
    t2: HashMap<u16, Cell>,
    t3: HashMap<u32, Cell>,
    t1d: HashMap<u16, Cell>,
    t3d: HashMap<u32, Cell>,
    lin: Linear,
}

impl Decoder {
    fn bump(m: &mut HashMap<u16, Cell>, k: u16, v: f64) {
        let c = m.entry(k).or_default();
        c.sum += v;
        c.n += 1;
    }

    fn bump3(m: &mut HashMap<u32, Cell>, k: u32, v: f64) {
        let c = m.entry(k).or_default();
        c.sum += v;
        c.n += 1;
    }

    fn table_mean(m: &HashMap<u16, Cell>, k: u16, min: u32) -> Option<f64> {
        let c = m.get(&k)?;
        (c.n >= min).then(|| c.sum / f64::from(c.n))
    }

    fn table_mean3(m: &HashMap<u32, Cell>, k: u32, min: u32) -> Option<f64> {
        let c = m.get(&k)?;
        (c.n >= min).then(|| c.sum / f64::from(c.n))
    }

    /// The arm's chain: highest table with support ≥ min, else down,
    /// else the linear model. Returns (value, level used 0..=3). Demeaned
    /// arms read the demeaned family and skip level 2 (no t2d — the piece
    /// clause is absorbed by t3d's state key).
    fn predict(&self, arm: WireArm, t: &[u8; 5], sk: StateKey) -> (f64, u8) {
        let t1k = pack_t1(t);
        let lvl = arm.max_level();
        if arm.demeaned() {
            if lvl >= 3
                && let Some(v) = Self::table_mean3(&self.t3d, t3_key(t1k, sk), TABLE_MIN)
            {
                return (v, 3);
            }
            if let Some(v) = Self::table_mean(&self.t1d, t1k, TABLE_MIN) {
                return (v, 1);
            }
            return (self.lin.predict(t), 0);
        }
        if lvl >= 3
            && let Some(v) = Self::table_mean3(&self.t3, t3_key(t1k, sk), TABLE_MIN)
        {
            return (v, 3);
        }
        if lvl >= 2
            && let Some(v) = Self::table_mean(&self.t2, t2_key(t1k, sk), TABLE_MIN)
        {
            return (v, 2);
        }
        if lvl >= 1
            && let Some(v) = Self::table_mean(&self.t1, t1k, TABLE_MIN)
        {
            return (v, 1);
        }
        (self.lin.predict(t), 0)
    }
}

// ── argmax (first strict max — decide()'s own fold) ─────────────────────

fn argmax_first(vals: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, &v) in vals.iter().enumerate() {
        if v > vals[best] {
            best = i;
        }
    }
    best
}

// ── board play ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Default)]
struct Outcome {
    points: u64,
    lines: u32,
    tetrises: u32,
    pieces: usize,
    topout: bool,
    /// Wire arms: how many decisions each chain rung answered (0=lin..3).
    levels: [u64; 4],
}

/// Per-decision rank statistics over champ1ply's own held decisions,
/// per wire arm (index = WIRE_ARMS position).
#[derive(Clone, Copy, Default)]
struct RankPartial {
    n: u64,
    agree: [u64; WIRE_ARMS.len()],
    rank_sum: [u64; WIRE_ARMS.len()],
    top3: [u64; WIRE_ARMS.len()],
}

impl RankPartial {
    fn merge(&mut self, o: &RankPartial) {
        self.n += o.n;
        for i in 0..WIRE_ARMS.len() {
            self.agree[i] += o.agree[i];
            self.rank_sum[i] += o.rank_sum[i];
            self.top3[i] += o.top3[i];
        }
    }
}

/// One seeded game, one arm. All arms consume the bag identically
/// (no-hold: one draw per piece after the initial preview draw), so
/// boards are paired per seed across arms.
fn play_game(
    arm: Option<WireArm>, // None = champ1ply
    dec: &Decoder,
    g1: &Genome,
    seed: u64,
    rows: usize,
    fill: u64,
    cap: usize,
) -> (Outcome, Option<RankPartial>) {
    let mut bag = Bag::new(seed);
    let mut board = garbage_board(seed, rows, fill);
    let mut next = bag.draw();
    let mut out = Outcome::default();
    let mut rank = RankPartial::default();
    let mut has_rank = false;
    while out.pieces < cap {
        let cur = next;
        next = bag.draw();
        let opts = landing_options_with(&board, cur, DropRule::FromTop);
        if opts.is_empty() {
            out.topout = true;
            break;
        }
        let mut tuples: Vec<[u8; 5]> = Vec::with_capacity(opts.len());
        for p in &opts {
            let feat = outcome_features(&board, p);
            tuples.push(grammar_tables::tetris_spot_forward(&board, p, &feat));
        }
        let pick = match arm {
            Some(w) => {
                let sk = state_key(&board, cur);
                let mut preds = Vec::with_capacity(tuples.len());
                for t in &tuples {
                    let (v, level) = dec.predict(w, t, sk);
                    preds.push(v);
                    out.levels[level as usize] += 1; // per-option rung usage
                }
                argmax_first(&preds)
            }
            None => {
                let view = View {
                    board: &board,
                    cur,
                    next,
                    held: None,
                    hold_ready: false,
                    bag_remaining: &[],
                };
                let scored = decide_scored(g1, &view);
                debug_assert_eq!(scored.len(), opts.len(), "no-hold candidates = options");
                let vals: Vec<f64> = scored.iter().map(|(_, v)| *v).collect();
                let star = argmax_first(&vals);
                let sk = state_key(&board, cur);
                for (di, w) in WIRE_ARMS.iter().enumerate() {
                    let mut preds = Vec::with_capacity(tuples.len());
                    for t in &tuples {
                        preds.push(dec.predict(*w, t, sk).0);
                    }
                    let am = argmax_first(&preds);
                    if am == star {
                        rank.agree[di] += 1;
                    }
                    let mut better = 0usize;
                    for (j, &pv) in preds.iter().enumerate() {
                        if j != star && pv > preds[star] {
                            better += 1;
                        }
                    }
                    rank.rank_sum[di] += better as u64;
                    if better < 3 {
                        rank.top3[di] += 1;
                    }
                }
                rank.n += 1;
                has_rank = true;
                star
            }
        };
        let (nb, l) = apply(&board, &opts[pick].cells);
        board = nb;
        out.lines += l;
        out.tetrises += u32::from(l == 4);
        out.points += LINES_SCORE[l.min(4) as usize];
        out.pieces += 1;
    }
    (out, has_rank.then_some(rank))
}

// ── training ────────────────────────────────────────────────────────────

struct TrainOut {
    dec: Decoder,
    lam: f64,
    loo: Vec<f64>,
    rows: u64,
    states: u64,
    occ1: usize,
    occ2: usize,
    occ3: usize,
}

fn train(g1: &Genome, seeds: &[u64], cap: usize) -> TrainOut {
    let mut t1: HashMap<u16, Cell> = HashMap::new();
    let mut t2: HashMap<u16, Cell> = HashMap::new();
    let mut t3: HashMap<u32, Cell> = HashMap::new();
    let mut t1d: HashMap<u16, Cell> = HashMap::new();
    let mut t3d: HashMap<u32, Cell> = HashMap::new();
    let mut games: Vec<GameRows> = Vec::new();
    let mut rows = 0u64;
    let mut states = 0u64;
    for &seed in seeds {
        for &(r, f) in &REGIMES {
            let mut st = GameRows::default();
            let mut bag = Bag::new(seed);
            let mut board = garbage_board(seed, r, f);
            let mut next = bag.draw();
            let mut pieces = 0usize;
            while pieces < cap {
                let cur = next;
                next = bag.draw();
                let opts = landing_options_with(&board, cur, DropRule::FromTop);
                if opts.is_empty() {
                    break; // champion topped out — no further decisions exist
                }
                let view = View {
                    board: &board,
                    cur,
                    next,
                    held: None,
                    hold_ready: false,
                    bag_remaining: &[],
                };
                let scored = decide_scored(g1, &view);
                debug_assert_eq!(scored.len(), opts.len());
                let sk = state_key(&board, cur);
                // The demeaned tables' target: value − decision mean. The
                // per-decision constant is identical across the decision's
                // options, so argmax over demeaned predictions equals
                // argmax over their re-centered values.
                let dmean = scored.iter().map(|(_, v)| *v).sum::<f64>() / scored.len() as f64;
                for (p, (_, v)) in opts.iter().zip(&scored) {
                    let feat = outcome_features(&board, p);
                    let t = grammar_tables::tetris_spot_forward(&board, p, &feat);
                    let t1k = pack_t1(&t);
                    Decoder::bump(&mut t1, t1k, *v);
                    Decoder::bump(&mut t2, t2_key(t1k, sk), *v);
                    Decoder::bump3(&mut t3, t3_key(t1k, sk), *v);
                    Decoder::bump(&mut t1d, t1k, v - dmean);
                    Decoder::bump3(&mut t3d, t3_key(t1k, sk), v - dmean);
                    st.row(&t, *v);
                    rows += 1;
                }
                states += 1;
                let vals: Vec<f64> = scored.iter().map(|(_, v)| *v).collect();
                let star = argmax_first(&vals);
                let (nb, _) = apply(&board, &opts[star].cells);
                board = nb;
                pieces += 1;
            }
            games.push(st);
        }
    }
    let (lin, lam, loo) = fit_linear(&games);
    let occ = |m: &HashMap<u16, Cell>| m.values().filter(|c| c.n >= TABLE_MIN).count();
    let (occ1, occ2) = (occ(&t1), occ(&t2));
    let occ3 = t3.values().filter(|c| c.n >= TABLE_MIN).count();
    TrainOut {
        dec: Decoder {
            t1,
            t2,
            t3,
            t1d,
            t3d,
            lin,
        },
        lam,
        loo,
        rows,
        states,
        occ1,
        occ2,
        occ3,
    }
}

// ── sign test (exact two-sided binomial on non-tied pairs) ──────────────

fn sign_test_p(wins: usize, losses: usize) -> f64 {
    let n = wins + losses;
    if n == 0 {
        return 1.0;
    }
    let kmax = wins.min(losses);
    let mut c = 1.0f64; // C(n, 0)
    let mut acc = 0.0f64;
    for k in 0..=kmax {
        if k > 0 {
            c *= (n - k + 1) as f64 / k as f64;
        }
        acc += c;
    }
    (acc * 2.0f64.powi(-(n as i32)) * 2.0).min(1.0)
}

// ── self tests ──────────────────────────────────────────────────────────

fn self_test() {
    // Champion id (at its native depth — the line hash includes d=).
    let g = Genome::champion_hybrid();
    assert_eq!(g.id(), HYBRID_ID, "hybrid champion genome drifted");
    // Empty-board option counts (the sim's own pins).
    assert_eq!(landing_options(&Board::empty(), Piece::O).len(), 9);
    assert_eq!(landing_options(&Board::empty(), Piece::I).len(), 17);
    // Pack bounds + round-trips.
    let max_t = [4u8, 4, 3, 2, 4];
    assert_eq!(pack_t1(&max_t), 1499);
    assert_eq!(unpack_t1(1499), max_t);
    for k in [0u16, 1, 60, 300, 749, 1499] {
        assert_eq!(pack_t1(&unpack_t1(k)), k, "t1 round-trip {k}");
    }
    let sk = StateKey {
        hw: 2,
        holes: 4,
        spread: 1,
        tallest: 2,
        lowest: 0,
        piece: 6,
    };
    assert_eq!(t2_key(1499, sk), 1499 * 7 + 6);
    assert_eq!(t3_key(1499, sk), ((10499u32 * 3 + 2) * 5 + 4) * 10 + 7);
    // State-key mirror pin: the computed clauses appear in the rendered
    // sentence, on real boards across shapes (the render is the contract).
    let hw_words = ["low", "of medium height", "tall"];
    let regions = ["left", "middle", "right"];
    let hole_clauses = [
        "There are no holes under the blocks.",
        "There is one hole under the blocks.",
        "There are two holes under the blocks.",
        "There are a few holes under the blocks.",
        "There are many holes under the blocks.",
    ];
    for &(r, f) in &REGIMES {
        let board = garbage_board(607, r, f);
        for piece in Piece::ALL {
            let k = state_key(&board, piece);
            let s = render_state_sentence(&board, piece);
            let stem = &s[..s.find(". ").unwrap_or(s.len())];
            assert!(
                stem.contains(&format!("The stack stands {}", hw_words[k.hw as usize])),
                "hw clause drift: {s:?}"
            );
            if k.spread == 1 {
                assert!(
                    stem.contains(&format!("tall on the {}", regions[k.tallest as usize])),
                    "tallest drift: {s:?}"
                );
                assert!(
                    stem.contains(&format!("low on the {}", regions[k.lowest as usize])),
                    "lowest drift: {s:?}"
                );
            } else {
                assert!(
                    stem.contains("the surface is mostly flat"),
                    "flat drift: {s:?}"
                );
            }
            assert!(
                s.contains(hole_clauses[k.holes as usize]),
                "holes drift: {s:?}"
            );
            assert!(
                s.contains(&format!("The {} piece is falling.", piece.spoken())),
                "piece drift: {s:?}"
            );
        }
    }
}

// ── main ────────────────────────────────────────────────────────────────

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    let t0 = Instant::now();
    self_test();
    println!("self-test: ok (genome {HYBRID_ID} pinned, mirror + packs verified)");

    let (train_seeds, held_seeds, train_cap, eval_cap): (Vec<u64>, Vec<u64>, usize, usize) =
        if quick {
            ((1..=6).collect(), (101..=106).collect(), 120, 800)
        } else {
            ((1..=40).collect(), (101..=140).collect(), 300, 5000)
        };

    let mut g1 = Genome::champion_hybrid();
    // TRUE 1-ply: `plies()` clamps to ≥2 while the NextPreview rule is on,
    // and the depth-2 recursion injects `TOPOUT` (−1e12) values whenever
    // the known next piece has no landing — both wrong for the serving-
    // matched reference. Kill the rule, pin depth: per-option values are
    // then pure `leaf_value` over the afterstate (no preview consumed).
    g1.set(RuleId::NextPreview, false);
    g1.depth = 1;

    let tr = train(&g1, &train_seeds, train_cap);
    println!(
        "train: {} states / {} rows over {} games; table cells ≥{TABLE_MIN}: t1 {}/1500, t2 {}/10500, t3 {}/1575000",
        tr.states,
        tr.rows,
        train_seeds.len() * REGIMES.len(),
        tr.occ1,
        tr.occ2,
        tr.occ3
    );
    println!("ridge λ={} (LOO-by-game MSE per λ: {:?})", tr.lam, tr.loo);

    // Evaluation: all arms × held seeds × regimes, parallel across games.
    let dec = tr.dec;
    let done = AtomicUsize::new(0);
    let total = held_seeds.len() * REGIMES.len() * (WIRE_ARMS.len() + 1);
    let jobs: Vec<(Option<WireArm>, usize, u64)> = {
        let mut v = Vec::new();
        for arm in WIRE_ARMS
            .iter()
            .copied()
            .map(Some)
            .chain(std::iter::once(None))
        {
            for ri in 0..REGIMES.len() {
                for &s in &held_seeds {
                    v.push((arm, ri, s));
                }
            }
        }
        v
    };

    /// One evaluated game: (arm, regime index, seed, outcome, rank stats).
    type GameResult = (Option<WireArm>, usize, u64, Outcome, Option<RankPartial>);

    let results: Vec<GameResult> = jobs
        .par_iter()
        .map(|&(arm, ri, seed)| {
            let (r, f) = REGIMES[ri];
            let (out, rp) = play_game(arm, &dec, &g1, seed, r, f, eval_cap);
            let n = done.fetch_add(1, Ordering::SeqCst) + 1;
            if n.is_multiple_of(40) || n == total {
                eprintln!("  eval {n}/{total}");
            }
            (arm, ri, seed, out, rp)
        })
        .collect();

    // champ1ply rank stats merged across its games.
    let mut rank = RankPartial::default();
    for (_, _, _, _, rp) in &results {
        if let Some(rp) = rp {
            rank.merge(rp);
        }
    }

    let arm_name = |a: Option<WireArm>| match a {
        None => "champ1ply".to_string(),
        Some(w) => w.name().to_string(),
    };

    println!("\n== board outcomes by regime (held seeds, cap {eval_cap}) ==");
    println!(
        "{:<11} {:>8} {:>9} {:>8} {:>8} {:>8} {:>7}",
        "arm", "regime", "points", "lines", "tetrises", "pieces", "n/top"
    );
    for arm in WIRE_ARMS
        .iter()
        .copied()
        .map(Some)
        .chain(std::iter::once(None))
    {
        for (ri, _) in REGIMES.iter().enumerate() {
            let rs: Vec<_> = results.iter().filter(|r| r.0 == arm && r.1 == ri).collect();
            let n = rs.len();
            let mean = |f: &dyn Fn(&Outcome) -> f64| {
                rs.iter().map(|r| f(&r.3)).sum::<f64>() / n.max(1) as f64
            };
            let tops = rs.iter().filter(|r| r.3.topout).count();
            println!(
                "{:<11} {:>8} {:>9.0} {:>8.1} {:>8.2} {:>8.0} {:>3}/{:<3}",
                arm_name(arm),
                format!("{}:{}", REGIMES[ri].0, REGIMES[ri].1),
                mean(&|o: &Outcome| o.points as f64),
                mean(&|o: &Outcome| o.lines as f64),
                mean(&|o: &Outcome| o.tetrises as f64),
                mean(&|o: &Outcome| o.pieces as f64),
                n,
                tops,
            );
        }
    }

    println!("\n== overall per arm ==");
    let mut overall: HashMap<Option<WireArm>, (f64, f64, f64, f64)> = HashMap::new();
    for arm in WIRE_ARMS
        .iter()
        .copied()
        .map(Some)
        .chain(std::iter::once(None))
    {
        let rs: Vec<_> = results.iter().filter(|r| r.0 == arm).collect();
        let n = rs.len().max(1) as f64;
        let pts = rs.iter().map(|r| r.3.points).sum::<u64>() as f64 / n;
        let lines = rs.iter().map(|r| r.3.lines).sum::<u32>() as f64 / n;
        let tetr = rs.iter().map(|r| r.3.tetrises).sum::<u32>() as f64 / n;
        let pieces = rs.iter().map(|r| r.3.pieces).sum::<usize>() as f64 / n;
        overall.insert(arm, (pts, lines, tetr, pieces));
        println!(
            "{:<11} pts {:>9.0}  lines {:>7.1}  tetr {:>6.2}  pieces {:>6.0}",
            arm_name(arm),
            pts,
            lines,
            tetr,
            pieces
        );
    }

    // Paired per-seed diffs vs champ1ply.
    let champ: HashMap<(usize, u64), Outcome> = results
        .iter()
        .filter(|r| r.0.is_none())
        .map(|r| ((r.1, r.2), r.3))
        .collect();
    println!("\n== paired per-seed vs champ1ply (positive Δ = champ ahead) ==");
    println!(
        "{:<11} {:>10} {:>6} {:>6} {:>5} {:>11} {:>9}",
        "arm", "meanΔpts", "wins", "loss", "ties", "meanΔlines", "sign_p(pts)"
    );
    let mut sig_p: HashMap<WireArm, f64> = HashMap::new();
    for w in WIRE_ARMS {
        let mut dpts = Vec::new();
        let mut dlines = Vec::new();
        for r in results.iter().filter(|r| r.0 == Some(w)) {
            let c = champ[&(r.1, r.2)];
            dpts.push(c.points as f64 - r.3.points as f64);
            dlines.push(f64::from(c.lines) - f64::from(r.3.lines));
        }
        let wins = dpts.iter().filter(|d| **d > 0.0).count();
        let loss = dpts.iter().filter(|d| **d < 0.0).count();
        let ties = dpts.len() - wins - loss;
        let sp = sign_test_p(wins, loss);
        sig_p.insert(w, sp);
        let np = dpts.len().max(1) as f64;
        println!(
            "{:<11} {:>10.0} {:>6} {:>6} {:>5} {:>11.1} {:>9.4}",
            w.name(),
            dpts.iter().sum::<f64>() / np,
            wins,
            loss,
            ties,
            dlines.iter().sum::<f64>() / np,
            sp
        );
    }

    // Rank agreement on champ1ply's own held decisions.
    println!("\n== rank agreement on champ1ply decisions ==");
    println!(
        "{:<11} {:>10} {:>12} {:>12} {:>10}",
        "decoder", "n", "agree", "mean_rank", "champ_top3"
    );
    if rank.n == 0 {
        println!("  (champ1ply made no held decision — no rank stats)");
    } else {
        for (di, w) in WIRE_ARMS.iter().enumerate() {
            println!(
                "{:<11} {:>10} {:>12.4} {:>12.3} {:>10.4}",
                w.name(),
                rank.n,
                rank.agree[di] as f64 / rank.n as f64,
                rank.rank_sum[di] as f64 / rank.n as f64,
                rank.top3[di] as f64 / rank.n as f64,
            );
        }
    }

    // Decoder chain level usage per wire arm (which rung answered).
    println!("\n== chain level usage (share of decisions, lin/t1/t2/t3) ==");
    for w in WIRE_ARMS {
        let rs: Vec<_> = results.iter().filter(|r| r.0 == Some(w)).collect();
        let decs: u64 = rs.iter().map(|r| r.3.levels.iter().sum::<u64>()).sum();
        if decs == 0 {
            continue;
        }
        let mix: Vec<f64> = (0..4)
            .map(|i| rs.iter().map(|r| r.3.levels[i]).sum::<u64>() as f64 / decs as f64)
            .collect();
        println!(
            "{:<11} {:.3} / {:.3} / {:.3} / {:.3}",
            w.name(),
            mix[0],
            mix[1],
            mix[2],
            mix[3]
        );
    }

    // The pre-declared verdict (t3 — the maximal ADDITIVE-class decoder of
    // the original run), plus the post-hoc demeaned-class line the
    // Issue-825 cross-review owed (a cell mean of ABSOLUTE values tracks
    // position goodness as much as option rank; the demeaned tables target
    // within-point ranking directly). The binding verdict is the BEST wire
    // arm across both classes — if any wire-constrained decoder clears the
    // gates, the wire carries a champion-level ranking.
    let agree_of = |w: WireArm| {
        if rank.n > 0 {
            rank.agree[w.idx()] as f64 / rank.n as f64
        } else {
            0.0
        }
    };
    let agree3 = agree_of(WireArm::T3);
    let best_arm = WIRE_ARMS
        .iter()
        .copied()
        .filter(|w| *w != WireArm::Lin)
        .max_by(|a, b| agree_of(*a).total_cmp(&agree_of(*b)))
        .expect("wire arms non-empty");
    let agree_best = agree_of(best_arm);
    let (champ_pts, champ_lines, _, _) = overall[&None];
    let (t3_pts, t3_lines, _, _) = overall[&Some(WireArm::T3)];
    let sp3 = sig_p[&WireArm::T3];
    let pts_ratio = t3_pts / champ_pts;
    let lines_ratio = t3_lines / champ_lines;
    println!("\n== verdict (pre-declared rules, judged on wire-t3) ==");
    println!(
        "agreement {agree3:.4} | sign_p(pts) {sp3:.4} | pts ratio {pts_ratio:.3} | lines ratio {lines_ratio:.3}"
    );
    let sp_best = sig_p[&best_arm];
    let (best_pts, best_lines, _, _) = overall[&Some(best_arm)];
    let best_pts_ratio = best_pts / champ_pts;
    let best_lines_ratio = best_lines / champ_lines;
    println!(
        "best wire arm: {} agreement {agree_best:.4} | sign_p {sp_best:.4} | pts ratio {best_pts_ratio:.3} | lines ratio {best_lines_ratio:.3}",
        best_arm.name()
    );
    let (v3, vbest) = (
        agree3 >= 0.95 && sp3 >= 0.05 && lines_ratio >= 0.95,
        agree_best >= 0.95 && sp_best >= 0.05 && best_lines_ratio >= 0.95,
    );
    let (must3, mustbest) = (
        agree3 <= 0.85 || (sp3 < 0.05 && pts_ratio <= 0.80),
        agree_best <= 0.85 || (sp_best < 0.05 && best_pts_ratio <= 0.80),
    );
    if v3 && vbest {
        println!(
            "VERDICT: WIRE-SUFFICIENT — the five ordinals carry the champion's 1-ply ranking under both decoder classes; no wire widening."
        );
    } else if must3 && mustbest {
        println!(
            "VERDICT: WIRE-MUST-WIDEN — no wire-constrained decoder class carries champion-level ranking (absolute-cell-mean AND demeaned tables both fail); the Instinct lane needs the raw afterstate grid (or its classes)."
        );
    } else {
        println!(
            "VERDICT: MIXED — the two decoder classes disagree; adjudicate on the board outcomes before any wire redesign."
        );
    }

    println!(
        "\nwall: {:.1}s (accuracy probe — no latency claim)",
        t0.elapsed().as_secs_f64()
    );
}
