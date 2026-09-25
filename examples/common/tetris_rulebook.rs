//! The tetris strategy RULEBOOK — Issue 892 T1, the ruliology surface.
//!
//! The owner's technique list (2026-09-25) plus the classic board terms,
//! encoded as DATA so a later self-evolve loop can enumerate, mutate,
//! select and promote them (the riir-clippy ruliology → self-evolve →
//! kat_promotion shape; riir-ai Plan 576 "FSM thought ruliology, not
//! per-scene rules"):
//!
//! - **Rule** — one row of [`RULES`]: id, KG triple `(subject, predicate,
//!   object)`, the source clause it came from, a PHYSICS precondition, a
//!   kind (board weight / search policy / native / inapplicable), default
//!   weights PER PLAY MODE, and a feature function.
//! - **Mode FSM** — `Build` / `Downstack` / `Survive`, chosen from the root
//!   board by data transition predicates ([`Genome::mode_of`]); every leaf
//!   of one decision is judged under the ROOT's mode (one decision, one
//!   intent).
//! - **Genome** — enable mask + per-mode weight table + search policy
//!   (depth, beam, hold) + FSM thresholds. One-line text round-trip and a
//!   BLAKE3 id over that line (the `katgpt-ruliology::SimpleProgram::id`
//!   convention), complexity = enabled-rule count (the Pareto axis).
//!
//! Physics is part of a rule: under `DropRule::FromTop` (the laya arena's
//! real hard drop) T-spins are UNREACHABLE and rotation direction is
//! NATIVE — both stay in the table with that precondition recorded, so the
//! rulebook is complete and honest about what the arena admits.
//!
//! Include after `mod tetris_sim;` and `mod tetris_lookahead;`.

#![allow(dead_code)] // each example consumes a different slice

use crate::tetris_lookahead::{apply, Bag, LINES_SCORE};
use crate::tetris_sim::{landing_options_with, Board, DropRule, Piece, HEIGHT, WIDTH};

// ── Vocabulary ────────────────────────────────────────────────────────────

/// Arena physics a rule may require.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Physics {
    /// Real hard drop from the top (laya-tetris-v3/v4 arena, our sim).
    FromTop,
    /// Soft drop + rotation during descent (guideline SRS games).
    SoftDropRotate,
}

/// The play-mode FSM state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Mode {
    /// Clean, low stack: build the 9-1 stack, keep the well, seek tetrises.
    Build = 0,
    /// A hole is covered: stop building up, clear the lines above it.
    Downstack = 1,
    /// Stack is high: take any line, forget the well.
    Survive = 2,
}

pub const N_MODES: usize = 3;
pub const MODES: [Mode; N_MODES] = [Mode::Build, Mode::Downstack, Mode::Survive];

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Downstack => "downstack",
            Self::Survive => "survive",
        }
    }
}

/// What a rule does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleKind {
    /// Adds `weight[mode] × feature(board)` to the leaf evaluation.
    BoardWeight,
    /// Shapes the SEARCH (depth / hold), not the evaluation.
    Search,
    /// Satisfied by construction of the move generator — no weight.
    Native,
    /// Precondition unmet under the arena physics — recorded, inert.
    Inapplicable,
}

/// Stable rule ids — the genome's bit positions. Append-only (a genome
/// line written today must decode the same tomorrow).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RuleId {
    Lines = 0,
    RowTrans = 1,
    ColTrans = 2,
    Holes = 3,
    Wells = 4,
    MaxHeight = 5,
    DeepWellUrgency = 6,
    NineOneWell = 7,
    TetrisBonus = 8,
    FlatTop = 9,
    HoleCover = 10,
    HoldI = 11,
    NextPreview = 12,
    HoldQueue = 13,
    RotateBoth = 14,
    TSpin = 15,
}

pub const N_RULES: usize = 16;

/// Leaf context a feature reads.
pub struct Leaf<'a> {
    pub board: &'a Board,
    pub heights: [usize; WIDTH],
    /// Lines cleared along the path root → leaf.
    pub lines: u32,
    /// Tetrises (4-line clears) along the path.
    pub tetrises: u32,
    /// Piece in the hold slot at the leaf (None = empty).
    pub held: Option<Piece>,
}

pub struct Rule {
    pub id: RuleId,
    /// Short, stable, genome-line name.
    pub key: &'static str,
    /// KG triple (subject, predicate, object).
    pub triple: (&'static str, &'static str, &'static str),
    /// The clause this row encodes.
    pub source: &'static str,
    /// `None` = any physics.
    pub requires: Option<Physics>,
    pub kind: RuleKind,
    /// Default weight per mode `[build, downstack, survive]`.
    pub w: [f64; N_MODES],
    pub feature: fn(&Leaf) -> f64,
}

/// The reserved 9-1 well column (far right).
pub const WELL_COL: usize = WIDTH - 1;

fn f_zero(_: &Leaf) -> f64 {
    0.0
}
fn f_lines(l: &Leaf) -> f64 {
    l.lines as f64
}
fn f_row_trans(l: &Leaf) -> f64 {
    let mut n = 0u32;
    for r in 0..HEIGHT {
        let mut prev = true;
        for c in 0..WIDTH {
            let cur = l.board.cell(r, c);
            n += u32::from(cur != prev);
            prev = cur;
        }
        n += u32::from(!prev);
    }
    n as f64
}
fn f_col_trans(l: &Leaf) -> f64 {
    let mut n = 0u32;
    for c in 0..WIDTH {
        let mut prev = false;
        for r in 0..HEIGHT {
            let cur = l.board.cell(r, c);
            n += u32::from(cur != prev);
            prev = cur;
        }
        n += u32::from(!prev);
    }
    n as f64
}
fn f_holes(l: &Leaf) -> f64 {
    l.board.hole_count() as f64
}
/// Well depth of column `c` (walls at full height).
fn well_depth(h: &[usize; WIDTH], c: usize) -> usize {
    let lft = if c == 0 { HEIGHT } else { h[c - 1] };
    let rgt = if c + 1 == WIDTH { HEIGHT } else { h[c + 1] };
    lft.min(rgt).saturating_sub(h[c])
}
fn f_wells(l: &Leaf) -> f64 {
    (0..WIDTH).map(|c| well_depth(&l.heights, c)).sum::<usize>() as f64
}
fn f_max_h(l: &Leaf) -> f64 {
    *l.heights.iter().max().unwrap_or(&0) as f64
}
fn f_deep_well(l: &Leaf) -> f64 {
    (0..WIDTH)
        .map(|c| {
            let d = well_depth(&l.heights, c);
            if d > 2 { (d - 2) * (d - 2) } else { 0 }
        })
        .sum::<usize>() as f64
}
/// 9-1: the edge well kept open, credited up to tetris depth (4). Positive
/// weight rebates the generic wells / deep-well cost on the RESERVED column
/// only — the "leave a single 1-wide well" clause.
fn f_nine_one_well(l: &Leaf) -> f64 {
    well_depth(&l.heights, WELL_COL).min(4) as f64
}
fn f_tetrises(l: &Leaf) -> f64 {
    l.tetrises as f64
}
/// Bumpiness over the 9 stack columns (the well column excluded — it is
/// supposed to be deep).
fn f_flat_top(l: &Leaf) -> f64 {
    (1..WELL_COL)
        .map(|c| l.heights[c].abs_diff(l.heights[c - 1]))
        .sum::<usize>() as f64
}
/// Filled cells stacked above holes — what Downstack digs through.
fn f_hole_cover(l: &Leaf) -> f64 {
    let mut cover = 0usize;
    for c in 0..WIDTH {
        let top = HEIGHT - l.heights[c]; // first filled row (or HEIGHT)
        let mut filled_above = 0usize;
        for r in top..HEIGHT {
            if l.board.cell(r, c) {
                filled_above += 1;
            } else {
                cover += filled_above; // a hole: everything above it covers it
            }
        }
    }
    cover as f64
}
fn f_hold_i(l: &Leaf) -> f64 {
    f64::from(u8::from(l.held == Some(Piece::I)))
}

/// THE RULEBOOK. Row order == `RuleId` discriminant (asserted).
pub const RULES: [Rule; N_RULES] = [
    Rule {
        id: RuleId::Lines,
        key: "lines",
        triple: ("placement", "clears", "lines"),
        source: "classic (Lee-style eval)",
        requires: None,
        kind: RuleKind::BoardWeight,
        w: [5.0, 8.0, 12.0],
        feature: f_lines,
    },
    Rule {
        id: RuleId::RowTrans,
        key: "row_trans",
        triple: ("row", "avoids", "transitions"),
        source: "classic (Dellacherie)",
        requires: None,
        kind: RuleKind::BoardWeight,
        w: [-3.2, -3.2, -3.2],
        feature: f_row_trans,
    },
    Rule {
        id: RuleId::ColTrans,
        key: "col_trans",
        triple: ("column", "avoids", "transitions"),
        source: "classic (Dellacherie)",
        requires: None,
        kind: RuleKind::BoardWeight,
        w: [-9.3, -9.3, -9.3],
        feature: f_col_trans,
    },
    Rule {
        id: RuleId::Holes,
        key: "holes",
        triple: ("stack", "avoids", "holes"),
        source: "classic (Dellacherie)",
        requires: None,
        kind: RuleKind::BoardWeight,
        w: [-7.9, -12.0, -7.9],
        feature: f_holes,
    },
    Rule {
        id: RuleId::Wells,
        key: "wells",
        triple: ("stack", "avoids", "wells"),
        source: "classic (Dellacherie)",
        requires: None,
        kind: RuleKind::BoardWeight,
        w: [-3.4, -3.4, -3.4],
        feature: f_wells,
    },
    Rule {
        id: RuleId::MaxHeight,
        key: "max_h",
        triple: ("stack", "stays", "low"),
        source: "classic",
        requires: None,
        kind: RuleKind::BoardWeight,
        w: [-0.1, -0.1, -2.0],
        feature: f_max_h,
    },
    Rule {
        id: RuleId::DeepWellUrgency,
        key: "deep_well",
        triple: ("deep_well", "gets_filled_by", "any_fitting_piece"),
        source: "owner 2026-09-25 (Bench 891): fill it even for one line, the I may never come",
        requires: None,
        kind: RuleKind::BoardWeight,
        w: [-2.0, -2.0, -2.0],
        feature: f_deep_well,
    },
    Rule {
        id: RuleId::NineOneWell,
        key: "nine_one",
        triple: ("stack", "reserves", "edge_well"),
        source: "owner tips: 9-1 stack — flat across 9 columns, one 1-wide well at the edge",
        requires: None,
        kind: RuleKind::BoardWeight,
        w: [4.0, 0.0, 0.0],
        feature: f_nine_one_well,
    },
    Rule {
        id: RuleId::TetrisBonus,
        key: "tetris",
        triple: ("I_piece", "clears", "tetris"),
        source: "owner tips: drop the long I-piece down the well to clear 4 lines at once",
        requires: None,
        kind: RuleKind::BoardWeight,
        w: [30.0, 0.0, 0.0],
        feature: f_tetrises,
    },
    Rule {
        id: RuleId::FlatTop,
        key: "flat",
        triple: ("top", "stays", "flat"),
        source: "owner tips: keep the top flat — no spikes, no jagged peaks",
        requires: None,
        kind: RuleKind::BoardWeight,
        w: [-1.0, -0.5, -0.5],
        feature: f_flat_top,
    },
    Rule {
        id: RuleId::HoleCover,
        key: "cover",
        triple: ("covered_hole", "triggers", "downstack"),
        source: "owner tips: fix misdrops immediately — clear the lines on top of the hole",
        requires: None,
        kind: RuleKind::BoardWeight,
        w: [-0.5, -2.0, -0.5],
        feature: f_hole_cover,
    },
    Rule {
        id: RuleId::HoldI,
        key: "hold_i",
        triple: ("hold", "reserves", "I_piece"),
        source: "owner tips: hold an I-piece in reserve for the ready well",
        requires: None,
        kind: RuleKind::BoardWeight,
        w: [6.0, 0.0, 0.0],
        feature: f_hold_i,
    },
    Rule {
        id: RuleId::NextPreview,
        key: "preview",
        triple: ("player", "plans_with", "next_preview"),
        source: "owner tips: watch the Next preview, plan 2-3 moves ahead",
        requires: None,
        kind: RuleKind::Search,
        w: [0.0; N_MODES],
        feature: f_zero,
    },
    Rule {
        id: RuleId::HoldQueue,
        key: "hold",
        triple: ("player", "uses", "hold_queue"),
        source: "owner tips: leverage the hold queue (save awkward S/Z for a flat spot)",
        requires: None,
        kind: RuleKind::Search,
        w: [0.0; N_MODES],
        feature: f_zero,
    },
    Rule {
        id: RuleId::RotateBoth,
        key: "rotate_both",
        triple: ("player", "rotates", "both_directions"),
        source: "owner tips: rotate in both directions (one press, not three)",
        requires: None,
        kind: RuleKind::Native,
        w: [0.0; N_MODES],
        feature: f_zero,
    },
    Rule {
        id: RuleId::TSpin,
        key: "tspin",
        triple: ("T_piece", "spins_into", "covered_slot"),
        source: "owner tips: learn T-spins (advanced)",
        requires: Some(Physics::SoftDropRotate),
        kind: RuleKind::Inapplicable,
        w: [0.0; N_MODES],
        feature: f_zero,
    },
];

/// The rule's effective kind under `physics`: a row whose precondition is
/// unmet is inert regardless of its declared kind.
pub fn effective_kind(rule: &Rule, physics: Physics) -> RuleKind {
    match rule.requires {
        Some(p) if p != physics => RuleKind::Inapplicable,
        _ => rule.kind,
    }
}

/// The toggleable rules (board weights + search policies) — the ruliology
/// enumeration axis. Classic terms and the owner's shaping are toggleable
/// too, so the enumeration can ask whether ANY term earns its place.
pub fn toggleable(physics: Physics) -> Vec<RuleId> {
    RULES
        .iter()
        .filter(|r| {
            matches!(
                effective_kind(r, physics),
                RuleKind::BoardWeight | RuleKind::Search
            )
        })
        .map(|r| r.id)
        .collect()
}

// ── The genome (self-evolve surface) ─────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct Genome {
    /// Bit `RuleId as u8` set ⇒ rule enabled.
    pub enabled: u32,
    /// `w[rule][mode]`.
    pub w: [[f64; N_MODES]; N_RULES],
    /// Plies of search when `NextPreview` is on: 2 = current + preview
    /// (known), 3 = + the piece after (expectation over the 7-bag remainder).
    pub depth: u8,
    /// Prior-pruned beam per ply beyond the first (the moka trick's top_k).
    pub beam: u8,
    /// FSM: max height at/above which the mode is `Survive`.
    pub survive_h: u8,
    /// FSM: holes at/above which the mode is `Downstack`.
    pub downstack_holes: u8,
}

impl Genome {
    /// Every applicable rule on, default weights, depth 2, beam 6.
    pub fn full(physics: Physics) -> Self {
        let mut enabled = 0u32;
        for r in &RULES {
            if matches!(
                effective_kind(r, physics),
                RuleKind::BoardWeight | RuleKind::Search
            ) {
                enabled |= 1 << (r.id as u8);
            }
        }
        let mut w = [[0.0; N_MODES]; N_RULES];
        for r in &RULES {
            w[r.id as usize] = r.w;
        }
        Self {
            enabled,
            w,
            depth: 2,
            beam: 6,
            survive_h: 13,
            downstack_holes: 1,
        }
    }

    /// Bench 891's ply2-shaped expressed as a genome: the classic terms +
    /// deep-well urgency, build-mode weights everywhere (no FSM effect),
    /// preview on, no hold. The reproduction anchor for T2.
    pub fn bench891_ply2_shaped() -> Self {
        let mut g = Self::full(Physics::FromTop);
        g.enabled = 0;
        for id in [
            RuleId::Lines,
            RuleId::RowTrans,
            RuleId::ColTrans,
            RuleId::Holes,
            RuleId::Wells,
            RuleId::MaxHeight,
            RuleId::DeepWellUrgency,
            RuleId::NextPreview,
        ] {
            g.enabled |= 1 << (id as u8);
        }
        for r in &RULES {
            g.w[r.id as usize] = [r.w[0]; N_MODES];
        }
        g.w[RuleId::Lines as usize] = [5.0; N_MODES];
        g.w[RuleId::Holes as usize] = [-7.9; N_MODES];
        g.w[RuleId::MaxHeight as usize] = [-0.1; N_MODES];
        g.beam = 255; // exhaustive
        g
    }

    pub fn on(&self, id: RuleId) -> bool {
        self.enabled & (1 << (id as u8)) != 0
    }
    pub fn set(&mut self, id: RuleId, on: bool) {
        if on {
            self.enabled |= 1 << (id as u8);
        } else {
            self.enabled &= !(1 << (id as u8));
        }
    }

    /// Complexity = enabled rule count (the Pareto axis).
    pub fn complexity(&self) -> u32 {
        self.enabled.count_ones()
    }

    /// Effective search plies.
    pub fn plies(&self) -> u8 {
        if self.on(RuleId::NextPreview) { self.depth.max(2) } else { 1 }
    }

    /// The FSM: mode of a ROOT board (data predicates, evaluated in order).
    pub fn mode_of(&self, board: &Board) -> Mode {
        let h = board.heights();
        let max_h = *h.iter().max().unwrap_or(&0);
        if max_h >= self.survive_h as usize {
            Mode::Survive
        } else if self.on(RuleId::HoleCover)
            && board.hole_count() >= self.downstack_holes.max(1) as usize
        {
            Mode::Downstack
        } else {
            Mode::Build
        }
    }

    /// Leaf evaluation under `mode`.
    pub fn eval(&self, leaf: &Leaf, mode: Mode) -> f64 {
        let m = mode as usize;
        let mut s = 0.0;
        for r in &RULES {
            if r.kind == RuleKind::BoardWeight && self.on(r.id) {
                let w = self.w[r.id as usize][m];
                if w != 0.0 {
                    s += w * (r.feature)(leaf);
                }
            }
        }
        s
    }

    /// Canonical one-line text form (round-trips via [`Genome::from_line`]).
    pub fn to_line(&self) -> String {
        let mut s = format!(
            "tetris-rulebook-v1 en={:#06x} d={} b={} sh={} dh={} w=",
            self.enabled, self.depth, self.beam, self.survive_h, self.downstack_holes
        );
        let mut first = true;
        for r in &RULES {
            if r.kind != RuleKind::BoardWeight {
                continue;
            }
            let w = self.w[r.id as usize];
            if !first {
                s.push(';');
            }
            first = false;
            s.push_str(&format!("{}:{}/{}/{}", r.key, w[0], w[1], w[2]));
        }
        s
    }

    pub fn from_line(line: &str) -> Option<Self> {
        let mut g = Self::full(Physics::FromTop);
        let mut parts = line.split_whitespace();
        if parts.next()? != "tetris-rulebook-v1" {
            return None;
        }
        for p in parts {
            let (k, v) = p.split_once('=')?;
            match k {
                "en" => g.enabled = u32::from_str_radix(v.trim_start_matches("0x"), 16).ok()?,
                "d" => g.depth = v.parse().ok()?,
                "b" => g.beam = v.parse().ok()?,
                "sh" => g.survive_h = v.parse().ok()?,
                "dh" => g.downstack_holes = v.parse().ok()?,
                "w" => {
                    for item in v.split(';') {
                        let (key, ws) = item.split_once(':')?;
                        let r = RULES.iter().find(|r| r.key == key)?;
                        let mut it = ws.split('/');
                        for m in 0..N_MODES {
                            g.w[r.id as usize][m] = it.next()?.parse().ok()?;
                        }
                    }
                }
                _ => return None,
            }
        }
        Some(g)
    }

    /// BLAKE3 id of the canonical line (first 16 hex chars).
    pub fn id(&self) -> String {
        let h = blake3::hash(self.to_line().as_bytes());
        h.to_hex()[..16].to_string()
    }
}

/// The SCORE champion (Issue 892 T4): delta-gated climb, no hold, empty
/// board, fitness = points; 22/150 mutations accepted. Held-out (seeds
/// 101..=120, cap 1000): 64,682 points/g · 44.7 tetrises/g · 20/20 survival
/// vs Bench 891 ply2-shaped 17,412 · 0.10 · 20/20. The climb DISABLED
/// `lines`, `row_trans` and `deep_well` and kept `nine_one` + `tetris` +
/// `flat`: per-line reward and deep-well urgency fight the 9-1 well.
pub const CHAMPION_POINTS_LINE: &str = "tetris-rulebook-v1 en=0x17bc d=2 b=6 sh=11 dh=1 w=lines:5/8/12;row_trans:-3.2/-3.2/-3.2;col_trans:-7.44/-9.3/-9.3;holes:-9.875/-12/-7.9;wells:-2.72/-4.25/-4.25;max_h:-0.1/-0.1/-2;deep_well:-2/-2.5/-2;nine_one:5/0.64/0;tetris:30/0/0;flat:-1.25/-0.625/-0.5;cover:-0.5/-2/-0.625;hold_i:6/0/0";

impl Genome {
    pub fn champion_points() -> Self {
        Self::from_line(CHAMPION_POINTS_LINE).expect("champion line parses")
    }
}

/// The HYBRID champion (Issue 892 T4) — the FSM doing its job: the score
/// champion's 9-1 / tetris / flat-top weights in `Build`, Bench 891's
/// survival weights verbatim in `Downstack` and `Survive` (per-mode weight
/// 0 = the rule is off in that mode), depth 3 beam 6. Thresholds chosen
/// from a 6-point (sh, dh) sweep; validated on FRESH seeds in Bench 892.
/// 19@75 n=60: 28/60 (= Bench-891 weights at depth 3) at 34,023 pts/g
/// (4.0×); empty board: 79,652 pts/g · 59.9 tetrises/g.
pub const CHAMPION_HYBRID_LINE: &str = "tetris-rulebook-v1 en=0x17ff d=3 b=6 sh=12 dh=3 w=lines:0/5/5;row_trans:0/-3.2/-3.2;col_trans:-7.44/-9.3/-9.3;holes:-9.875/-7.9/-7.9;wells:-2.72/-3.4/-3.4;max_h:-0.1/-0.1/-0.1;deep_well:0/-2/-2;nine_one:5/0/0;tetris:30/0/0;flat:-1.25/0/0;cover:-0.5/0/0;hold_i:6/0/0";

impl Genome {
    pub fn champion_hybrid() -> Self {
        Self::from_line(CHAMPION_HYBRID_LINE).expect("hybrid line parses")
    }
}

// ── Search ───────────────────────────────────────────────────────────────

/// A decision: optionally swap with hold, then place option `index` of
/// `landing_options_with(board, placed_piece, FromTop)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    pub use_hold: bool,
    pub index: usize,
}

/// What the searcher knows at a decision.
pub struct View<'a> {
    pub board: &'a Board,
    pub cur: Piece,
    pub next: Piece,
    pub held: Option<Piece>,
    /// Hold not yet used for the current piece (guideline: once per drop).
    pub hold_ready: bool,
    /// The 7-bag remainder AFTER `next` was drawn (support of the piece
    /// after next; empty ⇒ a fresh full bag).
    pub bag_remaining: &'a [Piece],
}

fn leaf_value(g: &Genome, mode: Mode, b: &Board, lines: u32, tetrises: u32, held: Option<Piece>) -> f64 {
    let leaf = Leaf {
        board: b,
        heights: b.heights(),
        lines,
        tetrises,
        held,
    };
    g.eval(&leaf, mode)
}

const TOPOUT: f64 = -1.0e12;

/// Best value of placing `piece` on `b` then continuing `plies_left − 1`
/// more plies. `after` = the known piece sequence after `piece` (preview),
/// `support` = the bag support for the first UNKNOWN piece.
#[allow(clippy::too_many_arguments)]
fn best_value(
    g: &Genome,
    mode: Mode,
    b: &Board,
    piece: Piece,
    after: Option<Piece>,
    support: &[Piece],
    plies_left: u8,
    lines: u32,
    tetrises: u32,
    held: Option<Piece>,
) -> f64 {
    let opts = landing_options_with(b, piece, DropRule::FromTop);
    if opts.is_empty() {
        return TOPOUT;
    }
    // Children with their 1-ply leaf value (the prior).
    let mut kids: Vec<(Board, u32, u32, f64)> = opts
        .iter()
        .map(|p| {
            let (nb, l) = apply(b, &p.cells);
            let t = tetrises + u32::from(l == 4);
            let v = leaf_value(g, mode, &nb, lines + l, t, held);
            (nb, lines + l, t, v)
        })
        .collect();
    if plies_left <= 1 {
        return kids.iter().map(|k| k.3).fold(f64::NEG_INFINITY, f64::max);
    }
    // The moka trick: prune to the prior's top-k before expanding.
    kids.sort_by(|a, b| b.3.total_cmp(&a.3));
    kids.truncate(g.beam.max(1) as usize);
    let mut best = f64::NEG_INFINITY;
    for (nb, l, t, _) in &kids {
        let v = match after {
            Some(nxt) => best_value(g, mode, nb, nxt, None, support, plies_left - 1, *l, *t, held),
            None => {
                // Chance node: exact expectation over the 7-bag support.
                let sup: &[Piece] = if support.is_empty() { &Piece::ALL } else { support };
                let mut acc = 0.0;
                for &x in sup {
                    acc += best_value(g, mode, nb, x, None, &[], plies_left - 1, *l, *t, held);
                }
                acc / sup.len() as f64
            }
        };
        best = best.max(v);
    }
    best
}

/// Choose a decision under genome `g`. `None` = top out.
pub fn decide(g: &Genome, v: &View) -> Option<Decision> {
    let mode = g.mode_of(v.board);
    let plies = g.plies();
    let hold_on = g.on(RuleId::HoldQueue) && v.hold_ready;
    // Candidate (use_hold, placed piece, known piece after it, held after).
    let mut cands: Vec<(bool, Piece, Option<Piece>, Option<Piece>)> = Vec::with_capacity(2);
    cands.push((false, v.cur, Some(v.next), v.held));
    if hold_on {
        match v.held {
            // Swap: play the held piece, cur goes to hold, next stays next.
            Some(h) if h != v.cur => cands.push((true, h, Some(v.next), Some(v.cur))),
            Some(_) => {}
            // Empty hold: stash cur, play next; the piece after is unknown.
            None => cands.push((true, v.next, None, Some(v.cur))),
        }
    }
    let mut best: Option<(Decision, f64)> = None;
    for (use_hold, piece, after, held_after) in cands {
        let opts = landing_options_with(v.board, piece, DropRule::FromTop);
        for (i, p) in opts.iter().enumerate() {
            let (nb, l) = apply(v.board, &p.cells);
            let t = u32::from(l == 4);
            let val = if plies <= 1 {
                leaf_value(g, mode, &nb, l, t, held_after)
            } else {
                match after {
                    Some(nxt) => {
                        // Piece after `nxt` is unknown → bag support.
                        best_value(g, mode, &nb, nxt, None, v.bag_remaining, plies - 1, l, t, held_after)
                    }
                    None => {
                        let sup: &[Piece] =
                            if v.bag_remaining.is_empty() { &Piece::ALL } else { v.bag_remaining };
                        let mut acc = 0.0;
                        for &x in sup {
                            acc += best_value(g, mode, &nb, x, None, &[], plies - 1, l, t, held_after);
                        }
                        acc / sup.len() as f64
                    }
                }
            };
            let d = Decision { use_hold, index: i };
            if best.is_none_or(|(_, bv)| val > bv) {
                best = Some((d, val));
            }
        }
    }
    best.map(|(d, _)| d)
}

/// The harness-signature adapter (`fn(&Board, cur, next) -> Option<usize>`,
/// `tetris_07_laya_h2h`): no hold, and no bag knowledge — exact for
/// `plies() <= 2`; at depth 3 the unknown piece is taken as uniform over a
/// fresh bag (an approximation the game-loop path does not need).
pub fn pick_no_hold(g: &Genome, board: &Board, cur: Piece, next: Piece) -> Option<usize> {
    let view = View {
        board,
        cur,
        next,
        held: None,
        hold_ready: false,
        bag_remaining: &[],
    };
    decide(g, &view).map(|d| d.index)
}

// ── The game loop (shared by the arena and the head-to-head) ─────────────

#[derive(Clone, Copy, Debug, Default)]
pub struct GameStats {
    pub pieces: usize,
    pub lines: u32,
    pub points: u64,
    pub tetrises: u32,
    pub holds: u32,
    /// Decisions per mode `[build, downstack, survive]`.
    pub mode_counts: [u32; N_MODES],
}

/// Play one seeded game under genome `g` (7-bag + preview + optional hold).
pub fn play_game(
    g: &Genome,
    seed: u64,
    cap: usize,
    start: Board,
) -> GameStats {
    let mut bag = Bag::new(seed);
    let mut board = start;
    let mut next = bag.draw();
    let mut held: Option<Piece> = None;
    let mut st = GameStats::default();
    while st.pieces < cap {
        let cur = next;
        next = bag.draw();
        let view = View {
            board: &board,
            cur,
            next,
            held,
            hold_ready: true,
            bag_remaining: bag.remaining(),
        };
        st.mode_counts[g.mode_of(&board) as usize] += 1;
        let Some(d) = decide(g, &view) else {
            break;
        };
        let piece = if d.use_hold {
            st.holds += 1;
            match held {
                Some(h) => {
                    held = Some(cur);
                    h
                }
                None => {
                    // Stash cur, play next, draw a fresh preview.
                    held = Some(cur);
                    let p = next;
                    next = bag.draw();
                    p
                }
            }
        } else {
            cur
        };
        let opts = landing_options_with(&board, piece, DropRule::FromTop);
        let (nb, l) = apply(&board, &opts[d.index].cells);
        board = nb;
        st.lines += l;
        st.tetrises += u32::from(l == 4);
        st.points += LINES_SCORE[l.min(4) as usize];
        st.pieces += 1;
    }
    st
}

// ── Self-checks (run by the arena before any measurement) ────────────────

pub fn selftest() {
    for (i, r) in RULES.iter().enumerate() {
        assert_eq!(r.id as usize, i, "RULES row order must match RuleId");
    }
    let keys: std::collections::HashSet<_> = RULES.iter().map(|r| r.key).collect();
    assert_eq!(keys.len(), N_RULES, "rule keys must be unique");
    for g in [
        Genome::full(Physics::FromTop),
        Genome::bench891_ply2_shaped(),
        Genome::champion_points(),
        Genome::champion_hybrid(),
    ] {
        let line = g.to_line();
        let back = Genome::from_line(&line).expect("genome line must parse");
        assert_eq!(back, g, "genome line must round-trip: {line}");
        assert_eq!(back.id(), g.id());
    }
    // The recorded champion's id is pinned (a drifted line is a new genome).
    assert_eq!(Genome::champion_points().id(), "ed5aa14b7d68472e");
    assert_eq!(Genome::champion_hybrid().id(), "68cae9d382014662");
    // Physics is part of the rule: T-spin inert under FromTop, live under
    // soft-drop physics.
    let ts = &RULES[RuleId::TSpin as usize];
    assert_eq!(effective_kind(ts, Physics::FromTop), RuleKind::Inapplicable);
    assert!(!Genome::full(Physics::FromTop).on(RuleId::TSpin));
    assert!(!toggleable(Physics::FromTop).contains(&RuleId::TSpin));
    assert!(!toggleable(Physics::FromTop).contains(&RuleId::RotateBoth));
    // FSM: an empty board builds; a covered hole downstacks; a tall stack
    // survives.
    let g = Genome::full(Physics::FromTop);
    assert_eq!(g.mode_of(&Board::empty()), Mode::Build);
    let mut holed = Board::empty();
    holed.place(&[(HEIGHT - 2, 0)]); // (HEIGHT-1, 0) stays empty below it
    assert_eq!(g.mode_of(&holed), Mode::Downstack);
    let mut tall = Board::empty();
    for r in (HEIGHT - 14)..HEIGHT {
        tall.place(&[(r, 3)]);
    }
    assert_eq!(g.mode_of(&tall), Mode::Survive);
    // Features on a known board: one hole under 1 cell → cover 1.
    let leaf = Leaf {
        board: &holed,
        heights: holed.heights(),
        lines: 0,
        tetrises: 0,
        held: None,
    };
    assert_eq!(f_hole_cover(&leaf), 1.0);
    assert_eq!(f_holes(&leaf), 1.0);
}
