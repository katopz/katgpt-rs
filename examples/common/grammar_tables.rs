//! The closed-grammar decode tables — katgpt-rs Plan 607 T2 (+ Issue 876's
//! v3 widening). Each
//! table mirrors its renderer's matches EXACTLY (`tetris_sim::
//! render_spot_sentence`, `flappy_sim`, `lanes_sim`): the fill ORDINAL is
//! the decoded feature value, so vocabulary order is contract. The
//! drift detector is corpus-level: every fixture sentence must decode,
//! re-render byte-identically, and land on the fills the renderer's own
//! band functions produce (asserted by the decode example's tests).
//!
//! Nothing here renders free text or scores — the tables + the fill→feature
//! mappings only. Fit/scoring glue lives in the gated example
//! (`decode_01_losslessness`); this module is importable wherever
//! `template_decode` is on.
//!
//! Domain notes (corpus-limited decode, stated not hidden):
//! - flappy motion: `falling fast`/`climbing fast` clamp the tails, so a
//!   foreign v outside [−2, 2] decodes to the CLAMPED value — correct
//!   within the pinned domain, lossy outside it. The v3 offset clause
//!   clamps |post_rel| at ±2 and the crash tails collapse (|rel| ≥ h+2
//!   reads identically) — that residual loss is the v3 render's, which is
//!   exactly what the v3 arm measures. The v3 post-motion vocabulary is
//!   bijective on the clamped −2..=2 domain.
//! - flappy gap width: `narrow`→2 / `wide`→3 is bijective only because the
//!   arena's gap half-height domain is {2, 3}.
//! - tetris holes/clears: 3–4 and 5+ share a fill ("a few"/"many"); the
//!   ordinal is the CLASS, not the exact count — that loss is the render's,
//!   which is exactly what the losslessness arm measures.

// The decode example compiles this module alone (its tests use the band
// mappers; main uses the decoders) — per-consumer dead-code warns would
// fire on the other half. The sims come from the CONSUMER's crate root
// (`crate::tetris_sim` etc.) so the whole example crate shares ONE
// tetris_sim/flappy_sim/lanes_sim instance — two `#[path]` instances of
// the same file would be two distinct types.
#![allow(dead_code)]

use katgpt_core::template_decode::{DecodeError, Grammar, Seg, Template};

use crate::flappy_sim::{self, Action, FlappyState};
use crate::lanes_sim::{self, LanesState};
use crate::tetris_sim::{self, Board, OutcomeFeatures, Placement, Piece, SideBand};

// ── Vocabularies (fill index = feature ordinal; ORDER IS CONTRACT) ──────

/// holes_delta class: exact 0/1/2, then the render's bands.
static TETRIS_HOLES: [&str; 5] = [
    "leaves no holes",
    "leaves one hole",
    "leaves two holes",
    "leaves a few holes",
    "leaves many holes",
];
/// `SideBand` clause order (LeftEdge..RightEdge).
static TETRIS_SIDE: [&str; 5] = [
    "on the left edge",
    "on the left side",
    "in the middle",
    "on the right side",
    "on the right edge",
];
/// `BumpBand` order (Flat, Small, Tall, Gap).
static TETRIS_SURFACE: [&str; 4] = [
    "sits flat on the surface",
    "makes a small bump on top",
    "makes a tall step on top",
    "fills a deep gap",
];
/// `HeightBand` order (Low, Medium, Tall) — NOTE: post-placement bands
/// (0–6 / 7–11 / 12+), different cuts than the state sentence's.
static TETRIS_HEIGHT: [&str; 3] = [
    "the stack stays low",
    "the stack stands medium",
    "the stack grows tall",
];
/// lines_cleared: "" = no clear (the render appends nothing).
static TETRIS_CLEARS: [&str; 5] = [
    "",
    ", and clears a line",
    ", and clears two lines",
    ", and clears three lines",
    ", and clears four lines",
];
/// State-sentence max-height word (0–4 / 5–10 / 11+ — DIFFERENT cuts than
/// the option's height band).
static TETRIS_HW: [&str; 3] = ["low", "of medium height", "tall"];
/// State-sentence region names, left-to-right order.
static TETRIS_SIDE3: [&str; 3] = ["left", "middle", "right"];
/// State-sentence board-hole class.
static TETRIS_HOLES2: [&str; 5] = [
    "There are no holes under the blocks.",
    "There is one hole under the blocks.",
    "There are two holes under the blocks.",
    "There are a few holes under the blocks.",
    "There are many holes under the blocks.",
];
/// `Piece::spoken()` in `Piece::ALL` order.
static TETRIS_PIECE: [&str; 7] = [
    "long straight",
    "square",
    "T shaped",
    "S shaped",
    "Z shaped",
    "left leaning ell",
    "right leaning ell",
];

/// `PosBand` order (Below..Above) — the ONLY option slot the flappy v2
/// grammar renders (the v1 motion clause was the measured confound).
static FLAPPY_POS: [&str; 7] = [
    "sinks below the gap",
    "squeezes through the bottom of the gap",
    "glides through the lower half of the gap",
    "glides through the middle of the gap",
    "glides through the upper half of the gap",
    "squeezes through the top of the gap",
    "flies above the gap",
];
/// State-sentence relative-position band (well above..well below).
static FLAPPY_REL: [&str; 5] = [
    "well above",
    "a little above",
    "level with",
    "a little below",
    "well below",
];
/// `motion_clause(v)` in v = −2..=2 order (bijective on the clamped domain).
static FLAPPY_MOT: [&str; 5] = [
    "falling fast",
    "falling",
    "flying level",
    "rising",
    "climbing fast",
];
/// `gap_clause(h)`: 2 → narrow, else wide (bijective on {2, 3}).
static FLAPPY_GAP: [&str; 2] = ["narrow", "wide"];
/// v3 option offset clause — fine post_rel relative to the gap center,
/// clamped at ±2 (`flappy_sim::offset_clause` order).
static FLAPPY_OFFSET: [&str; 5] = [
    "under the center",
    "just under the center",
    "at the center",
    "just over the center",
    "over the center",
];
/// v3 option post-motion clause — neutral kinematic wording
/// (`flappy_sim::post_motion_clause` order, bijective on the clamped
/// −2..=2 domain).
static FLAPPY_PMOT: [&str; 5] = [
    "drifting down two steps",
    "drifting down one step",
    "holding this height",
    "drifting up one step",
    "drifting up two steps",
];

/// `LANE_NAMES` order.
static LANES_LANE: [&str; 3] = ["left", "middle", "right"];
static LANES_DIST: [&str; 2] = ["close", "far"];
/// One noun per obstacle class — the wording pin (`lanes_sim::obstacle_noun`).
static LANES_NOUN: [&str; 3] = ["a barrier", "a train", "a rock"];

// ── Templates (vocab indices per grammar, below) ─────────────────────────

static T_TETRIS_SPOT: [Seg; 11] = [
    Seg::Lit("The piece "),
    Seg::Slot(0), // TETRIS_HOLES
    Seg::Lit(" under it "),
    Seg::Slot(1), // TETRIS_SIDE
    Seg::Lit(", "),
    Seg::Slot(2), // TETRIS_SURFACE
    Seg::Lit(", and "),
    Seg::Slot(3), // TETRIS_HEIGHT
    Seg::Lit(""),  // separator only — the two fills are adjacent in the sentence
    Seg::Slot(4), // TETRIS_CLEARS ("" renders as nothing)
    Seg::Lit("."),
];
static T_TETRIS_STATE_SPREAD: [Seg; 11] = [
    Seg::Lit("The stack stands "),
    Seg::Slot(0), // TETRIS_HW
    Seg::Lit(", tall on the "),
    Seg::Slot(1), // TETRIS_SIDE3 (tallest region)
    Seg::Lit(" and low on the "),
    Seg::Slot(1), // TETRIS_SIDE3 (lowest region)
    Seg::Lit(". "),
    Seg::Slot(2), // TETRIS_HOLES2
    Seg::Lit(" The "),
    Seg::Slot(3), // TETRIS_PIECE
    Seg::Lit(" piece is falling."),
];
static T_TETRIS_STATE_FLAT: [Seg; 7] = [
    Seg::Lit("The stack stands "),
    Seg::Slot(0), // TETRIS_HW
    Seg::Lit(" and the surface is mostly flat. "),
    Seg::Slot(2), // TETRIS_HOLES2
    Seg::Lit(" The "),
    Seg::Slot(3), // TETRIS_PIECE
    Seg::Lit(" piece is falling."),
];
/// v4 state templates: the v2 shapes plus the next-piece preview sentence
/// (plan 609 T1.2) — the 5th slot reuses `TETRIS_PIECE`. Option templates
/// are UNCHANGED (the preview lives in the state line only).
static T_TETRIS_STATE_V4_SPREAD: [Seg; 13] = [
    Seg::Lit("The stack stands "),
    Seg::Slot(0), // TETRIS_HW
    Seg::Lit(", tall on the "),
    Seg::Slot(1), // TETRIS_SIDE3 (tallest region)
    Seg::Lit(" and low on the "),
    Seg::Slot(1), // TETRIS_SIDE3 (lowest region)
    Seg::Lit(". "),
    Seg::Slot(2), // TETRIS_HOLES2
    Seg::Lit(" The "),
    Seg::Slot(3), // TETRIS_PIECE
    Seg::Lit(" piece is falling. The next piece is the "),
    Seg::Slot(4), // TETRIS_PIECE (reused — the preview)
    Seg::Lit(" piece."),
];
static T_TETRIS_STATE_V4_FLAT: [Seg; 9] = [
    Seg::Lit("The stack stands "),
    Seg::Slot(0), // TETRIS_HW
    Seg::Lit(" and the surface is mostly flat. "),
    Seg::Slot(2), // TETRIS_HOLES2
    Seg::Lit(" The "),
    Seg::Slot(3), // TETRIS_PIECE
    Seg::Lit(" piece is falling. The next piece is the "),
    Seg::Slot(4), // TETRIS_PIECE (reused — the preview)
    Seg::Lit(" piece."),
];
static T_FLAPPY_OPTION: [Seg; 3] = [Seg::Lit("The bird "), Seg::Slot(0), Seg::Lit(".")];
/// v3 option template: band + offset + neutral post-motion (Issue 876).
static T_FLAPPY_OPTION_V3: [Seg; 7] = [
    Seg::Lit("The bird "),
    Seg::Slot(0), // FLAPPY_POS
    Seg::Lit(", "),
    Seg::Slot(1), // FLAPPY_OFFSET
    Seg::Lit(", "),
    Seg::Slot(2), // FLAPPY_PMOT
    Seg::Lit("."),
];
static T_FLAPPY_STATE: [Seg; 7] = [
    Seg::Lit("The bird is "),
    Seg::Slot(0), // FLAPPY_REL
    Seg::Lit(" the gap center, "),
    Seg::Slot(1), // FLAPPY_MOT
    Seg::Lit(". The gap is "),
    Seg::Slot(2), // FLAPPY_GAP
    Seg::Lit(". The pipe is just ahead."),
];
static T_LANES_CLEAR: [Seg; 3] = [Seg::Lit("The "), Seg::Slot(0), Seg::Lit(" lane is clear ahead.")];
static T_LANES_BLOCKED: [Seg; 7] = [
    Seg::Lit("The "),
    Seg::Slot(0), // LANES_LANE
    Seg::Lit(" lane is blocked "),
    Seg::Slot(1), // LANES_DIST
    Seg::Lit(" ahead by "),
    Seg::Slot(2), // LANES_NOUN
    Seg::Lit("."),
];

// ── Grammar constructors (validated at every build) ─────────────────────

/// `laya-tetris-v2` per-spot option sentence: 5 slots
/// (holes class, side band, surface band, height band, clears class).
pub fn tetris_spot() -> Grammar {
    static VOCABS: [&[&str]; 5] = [
        &TETRIS_HOLES,
        &TETRIS_SIDE,
        &TETRIS_SURFACE,
        &TETRIS_HEIGHT,
        &TETRIS_CLEARS,
    ];
    static TEMPLATES: [Template; 1] = [Template(&T_TETRIS_SPOT)];
    Grammar::new(&VOCABS, &TEMPLATES)
}

/// `laya-tetris-v2` state context sentence: two shapes (spread / flat).
pub fn tetris_state() -> Grammar {
    static VOCABS: [&[&str]; 4] = [&TETRIS_HW, &TETRIS_SIDE3, &TETRIS_HOLES2, &TETRIS_PIECE];
    static TEMPLATES: [Template; 2] = [
        Template(&T_TETRIS_STATE_SPREAD),
        Template(&T_TETRIS_STATE_FLAT),
    ];
    Grammar::new(&VOCABS, &TEMPLATES)
}

/// `laya-tetris-v4` state context sentence (plan 609 T1.3): the v2 shapes
/// plus the preview tail. The 5th slot REUSES `TETRIS_PIECE` — same
/// vocabulary, its own slot ordinal.
pub fn tetris_state_v4() -> Grammar {
    static VOCABS: [&[&str]; 5] = [
        &TETRIS_HW,
        &TETRIS_SIDE3,
        &TETRIS_HOLES2,
        &TETRIS_PIECE,
        &TETRIS_PIECE, // slot 4 — the preview, same vocabulary
    ];
    static TEMPLATES: [Template; 2] = [
        Template(&T_TETRIS_STATE_V4_SPREAD),
        Template(&T_TETRIS_STATE_V4_FLAT),
    ];
    Grammar::new(&VOCABS, &TEMPLATES)
}

/// `laya-flappy-v2` per-option sentence: the position band alone.
/// FROZEN — the committed v2 fixture's decode table (Bench 880/881
/// provenance); the live grammar is `flappy_option_v3`.
pub fn flappy_option() -> Grammar {
    static VOCABS: [&[&str]; 1] = [&FLAPPY_POS];
    static TEMPLATES: [Template; 1] = [Template(&T_FLAPPY_OPTION)];
    Grammar::new(&VOCABS, &TEMPLATES)
}

/// `laya-flappy-v3` per-option sentence: position band + quantized offset
/// + neutral post-motion (Issue 876's widening).
pub fn flappy_option_v3() -> Grammar {
    static VOCABS: [&[&str]; 3] = [&FLAPPY_POS, &FLAPPY_OFFSET, &FLAPPY_PMOT];
    static TEMPLATES: [Template; 1] = [Template(&T_FLAPPY_OPTION_V3)];
    Grammar::new(&VOCABS, &TEMPLATES)
}

/// `laya-flappy-v2` state context sentence: rel band + motion + gap width.
pub fn flappy_state() -> Grammar {
    static VOCABS: [&[&str]; 3] = [&FLAPPY_REL, &FLAPPY_MOT, &FLAPPY_GAP];
    static TEMPLATES: [Template; 1] = [Template(&T_FLAPPY_STATE)];
    Grammar::new(&VOCABS, &TEMPLATES)
}

/// `laya-lanes-v1` per-lane sentence: two shapes (clear / blocked).
pub fn lanes_option() -> Grammar {
    static VOCABS: [&[&str]; 3] = [&LANES_LANE, &LANES_DIST, &LANES_NOUN];
    static TEMPLATES: [Template; 2] = [Template(&T_LANES_CLEAR), Template(&T_LANES_BLOCKED)];
    Grammar::new(&VOCABS, &TEMPLATES)
}

// ── The bounded claims, checked ─────────────────────────────────────────

/// Every table's full fill product is tiny; this cap is the checked bound.
pub const CLOSED_SPACE_CAP: usize = 100_000;

/// `verify_closed` over every table — the full closed-space proof.
pub fn verify_all_closed() -> Result<(), String> {
    tetris_spot().verify_closed(CLOSED_SPACE_CAP)?;
    tetris_state().verify_closed(CLOSED_SPACE_CAP)?;
    tetris_state_v4().verify_closed(CLOSED_SPACE_CAP)?;
    flappy_option().verify_closed(CLOSED_SPACE_CAP)?;
    flappy_option_v3().verify_closed(CLOSED_SPACE_CAP)?;
    flappy_state().verify_closed(CLOSED_SPACE_CAP)?;
    lanes_option().verify_closed(CLOSED_SPACE_CAP)?;
    Ok(())
}

// ── Decoders (sentence → typed fills) ───────────────────────────────────

/// Decode a tetris spot sentence → (holes, side, surface, height, clears)
/// fill ordinals.
pub fn decode_tetris_spot(g: &Grammar, sentence: &str) -> Result<[u8; 5], DecodeError> {
    let m = g.decode(sentence)?;
    debug_assert_eq!(m.template, 0);
    debug_assert_eq!(m.n_slots, 5);
    Ok([m.fills[0], m.fills[1], m.fills[2], m.fills[3], m.fills[4]])
}

/// Decode a tetris v4 state sentence → the v2 fills plus the preview
/// piece fill: `Ok` = the SPREAD shape (fills [hw, tallest, lowest, holes2,
/// piece, next], template 0); `Err` = the FLAT shape (fills [hw, holes2,
/// piece, next], template 1). A sentence outside the grammar is a contract
/// violation (the caller renders through the same table) — loud refusal.
pub fn decode_tetris_state_v4(g: &Grammar, sentence: &str) -> Result<[u8; 6], [u8; 4]> {
    let m = g
        .decode(sentence)
        .unwrap_or_else(|e| panic!("v4 state sentence must decode ({e:?}): {sentence:?}"));
    match m.template {
        0 => Ok([
            m.fills[0], m.fills[1], m.fills[2], m.fills[3], m.fills[4], m.fills[5],
        ]),
        1 => Err([m.fills[0], m.fills[1], m.fills[2], m.fills[3]]),
        t => unreachable!("v4 state grammar has 2 templates, decoded {t}"),
    }
}

/// Decode a flappy v2 option sentence → the `PosBand` fill ordinal.
pub fn decode_flappy_option(g: &Grammar, sentence: &str) -> Result<u8, DecodeError> {
    let m = g.decode(sentence)?;
    debug_assert_eq!(m.template, 0);
    debug_assert_eq!(m.n_slots, 1);
    Ok(m.fills[0])
}

/// Decode a flappy v3 option sentence → (band, offset, post-motion) fill
/// ordinals.
pub fn decode_flappy_option_v3(g: &Grammar, sentence: &str) -> Result<[u8; 3], DecodeError> {
    let m = g.decode(sentence)?;
    debug_assert_eq!(m.template, 0);
    debug_assert_eq!(m.n_slots, 3);
    Ok([m.fills[0], m.fills[1], m.fills[2]])
}

/// Decode a flappy state sentence → (rel-band ordinal, exact v, exact h).
/// v/h are EXACT on the pinned domain (see the module notes).
pub fn decode_flappy_state(g: &Grammar, sentence: &str) -> Result<(u8, i32, i32), DecodeError> {
    let m = g.decode(sentence)?;
    debug_assert_eq!(m.template, 0);
    debug_assert_eq!(m.n_slots, 3);
    let v = m.fills[1] as i32 - 2;
    let h = if m.fills[2] == 0 { 2 } else { 3 };
    Ok((m.fills[0], v, h))
}

/// One lane's decoded obstacle: kind ordinal (0 = clear, then noun order)
/// and the distance fill when blocked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneDecoded {
    pub kind: u8,
    pub dist: Option<u8>,
    /// The lane-name fill — which lane this sentence describes.
    pub lane: u8,
}

/// Decode one lanes option sentence.
pub fn decode_lanes_option(g: &Grammar, sentence: &str) -> Result<LaneDecoded, DecodeError> {
    let m = g.decode(sentence)?;
    let lane = m.fills[0];
    match m.template {
        0 => Ok(LaneDecoded {
            kind: 0,
            dist: None,
            lane,
        }),
        1 => Ok(LaneDecoded {
            kind: 1 + m.fills[2],
            dist: Some(m.fills[1]),
            lane,
        }),
        t => unreachable!("lanes grammar has 2 templates, decoded {t}"),
    }
}

/// All three lanes, decoded from their THREE option sentences (the option
/// set IS the state — the state context sentence is redundant for the
/// features and is not part of the lanes table).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LanesDecoded {
    pub lanes: [LaneDecoded; 3],
}

/// Decode the lanes state from its three option sentences (pinned order).
pub fn decode_lanes_state(g: &Grammar, sentences: &[&str; 3]) -> Result<LanesDecoded, DecodeError> {
    let mut lanes = [LaneDecoded {
        kind: 0,
        dist: None,
        lane: 0,
    }; 3];
    for (i, s) in sentences.iter().enumerate() {
        lanes[i] = decode_lanes_option(g, s)?;
    }
    Ok(LanesDecoded { lanes })
}

// ── Forward mappers (semantic fills, for the corpus validation) ──────────

/// The fills `render_spot_sentence` WOULD emit, computed from the same
/// band functions the renderer calls. This mirrors the renderer's matches;
/// the corpus round-trip test is the drift detector between them.
pub fn tetris_spot_forward(board: &Board, p: &Placement, f: &OutcomeFeatures) -> [u8; 5] {
    let holes = match f.holes_delta {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        _ => 4,
    };
    let mut cols: Vec<usize> = p.cells.iter().map(|&(_, c)| c).collect();
    cols.sort_unstable();
    let center = (cols[0] + cols[cols.len() - 1]) / 2;
    let side = match SideBand::of_center(center) {
        SideBand::LeftEdge => 0,
        SideBand::LeftSide => 1,
        SideBand::Middle => 2,
        SideBand::RightSide => 3,
        SideBand::RightEdge => 4,
    };
    let surface = match tetris_sim::bump_band(board, p) {
        tetris_sim::BumpBand::Flat => 0,
        tetris_sim::BumpBand::Small => 1,
        tetris_sim::BumpBand::Tall => 2,
        tetris_sim::BumpBand::Gap => 3,
    };
    let height = match tetris_sim::HeightBand::of_max_height(f.max_height) {
        tetris_sim::HeightBand::Low => 0,
        tetris_sim::HeightBand::Medium => 1,
        tetris_sim::HeightBand::Tall => 2,
    };
    let clears = match f.lines_cleared {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        _ => 4,
    };
    [holes, side, surface, height, clears]
}

/// The tetris losslessness arm's feature width: the five fill ordinals.
pub const TETRIS_DECODED_F: usize = 5;

/// A decoded tetris spot as the losslessness arm's feature row (the CLASS
/// ordinals — see the module notes on bands).
pub fn tetris_decoded_features(fills: &[u8; 5]) -> [f64; TETRIS_DECODED_F] {
    [
        fills[0] as f64,
        fills[1] as f64,
        fills[2] as f64,
        fills[3] as f64,
        fills[4] as f64,
    ]
}

/// The flappy losslessness arm's feature width: [post-band ordinal,
/// pre-rel-band ordinal, exact v, exact gap half]. The option sentence
/// supplies the post band; the state sentence supplies rel/v/h.
pub const FLAPPY_DECODED_F: usize = 4;

pub fn flappy_decoded_features(post: u8, rel: u8, v: i32, h: i32) -> [f64; FLAPPY_DECODED_F] {
    [post as f64, rel as f64, v as f64, h as f64]
}

/// The flappy v3 losslessness arm's feature width and row: the decoded
/// fills reconstructed into STRUCTURED UNITS (the lanes anchor's own
/// pattern — decoded features live in the units the render describes).
/// Column order mirrors `flappy_sim::FEATURE_NAMES`:
/// [post_rel, post_abs_rel, post_v, pre_rel, pre_v, in_gap, edge_margin,
/// gap_half].
///
/// Reconstruction law (exact on every rendered combination):
/// - (band, offset, h) → post_rel: Squeeze/Lower/Upper/Middle rows pin the
///   exact cell for |rel| ≤ h (Lower "under" = −2 needs h = 3; the h = 2
///   Lower band only renders −1 = "just under"); the crash tails collapse
///   in the render, so they reconstruct at the boundary ±(h+1).
/// - post-motion → post_v: bijective on the clamped −2..=2 domain.
/// - state rel band → pre_rel, clamped ±2 (the band's own collapse).
pub const FLAPPY_V3_DECODED_F: usize = 8;

pub fn flappy_v3_decoded_features(
    post: [u8; 3],
    rel: u8,
    v: i32,
    h: i32,
) -> [f64; FLAPPY_V3_DECODED_F] {
    let (band, offset) = (post[0], post[1]);
    let hh = h;
    let post_rel: i32 = match band {
        0 => -(hh + 1),                          // Below: tail → boundary
        1 => -hh,                                // SqueezeBottom: exact
        2 => if offset == 1 { -1 } else { -2 },  // Lower: just-under exact
        3 => 0,                                  // Middle: exact
        4 => if offset == 3 { 1 } else { 2 },    // Upper: just-over exact
        5 => hh,                                 // SqueezeTop: exact
        _ => hh + 1,                             // Above: tail → boundary
    };
    let pre_rel: i32 = match rel {
        0 => 2,
        1 => 1,
        2 => 0,
        3 => -1,
        _ => -2,
    };
    let post_v = post[2] as i32 - 2;
    let abs = post_rel.abs();
    [
        post_rel as f64,
        abs as f64,
        post_v as f64,
        pre_rel as f64,
        v as f64,
        (abs <= hh) as u8 as f64,
        (hh - abs) as f64,
        h as f64,
    ]
}

/// The flappy state's semantic fills — the renderer's own matches.
pub fn flappy_state_forward(s: &FlappyState) -> (u8, i32, i32) {
    let rel = s.y - s.g;
    let rel_idx = match rel.cmp(&0) {
        std::cmp::Ordering::Greater if rel > 1 => 0,
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Equal => 2,
        std::cmp::Ordering::Less if rel < -1 => 4,
        std::cmp::Ordering::Less => 3,
    };
    let mot_idx = match s.v {
        i32::MIN..=-2 => 0,
        -1 => 1,
        0 => 2,
        1 => 3,
        _ => 4,
    };
    let gap_idx = if s.h == 2 { 0 } else { 1 };
    (rel_idx, mot_idx, gap_idx)
}

/// The flappy option's semantic fill — `pos_band` of the resulting state.
pub fn flappy_option_forward(s: &FlappyState, a: Action) -> u8 {
    let (y2, _) = flappy_sim::result(s, a);
    match flappy_sim::pos_band(y2 - s.g, s.h) {
        flappy_sim::PosBand::Below => 0,
        flappy_sim::PosBand::SqueezeBottom => 1,
        flappy_sim::PosBand::Lower => 2,
        flappy_sim::PosBand::Middle => 3,
        flappy_sim::PosBand::Upper => 4,
        flappy_sim::PosBand::SqueezeTop => 5,
        flappy_sim::PosBand::Above => 6,
    }
}

/// The flappy v3 option's semantic fills — (band, offset, post-motion),
/// mirroring `render_option_sentence`'s matches.
pub fn flappy_option_forward_v3(s: &FlappyState, a: Action) -> [u8; 3] {
    let (y2, v2) = flappy_sim::result(s, a);
    let rel = y2 - s.g;
    let band = match flappy_sim::pos_band(rel, s.h) {
        flappy_sim::PosBand::Below => 0,
        flappy_sim::PosBand::SqueezeBottom => 1,
        flappy_sim::PosBand::Lower => 2,
        flappy_sim::PosBand::Middle => 3,
        flappy_sim::PosBand::Upper => 4,
        flappy_sim::PosBand::SqueezeTop => 5,
        flappy_sim::PosBand::Above => 6,
    };
    let offset = match rel {
        i32::MIN..=-2 => 0,
        -1 => 1,
        0 => 2,
        1 => 3,
        _ => 4,
    };
    let pmot = match v2 {
        i32::MIN..=-2 => 0,
        -1 => 1,
        0 => 2,
        1 => 3,
        _ => 4,
    };
    [band, offset, pmot]
}

/// The lanes semantic decode — from the structured state itself.
pub fn lanes_forward(s: &LanesState) -> LanesDecoded {
    let lane = |l: lanes_sim::LaneObs, idx: u8| LaneDecoded {
        kind: match l.kind {
            lanes_sim::ObsKind::Clear => 0,
            lanes_sim::ObsKind::Barrier => 1,
            lanes_sim::ObsKind::Train => 2,
            lanes_sim::ObsKind::Rock => 3,
        },
        dist: match l.kind {
            lanes_sim::ObsKind::Clear => None,
            _ => match l.dist {
                lanes_sim::Dist::Close => Some(0),
                lanes_sim::Dist::Far => Some(1),
            },
        },
        lane: idx,
    };
    LanesDecoded {
        lanes: [
            lane(s.lanes[0], 0),
            lane(s.lanes[1], 1),
            lane(s.lanes[2], 2),
        ],
    }
}

/// The lanes decoded feature width — EXACTLY the structured row (all eight
/// columns are recoverable from the three sentences; the lossless anchor
/// of the whole arm).
pub const LANES_DECODED_F: usize = 8;

pub fn lanes_decoded_features(d: &LanesDecoded, lane: usize) -> [f64; LANES_DECODED_F] {
    let l = d.lanes[lane];
    let blocked = l.kind != 0;
    let dist_close = l.dist == Some(0);
    let dist_far = l.dist == Some(1);
    [
        blocked as u8 as f64,
        (blocked && dist_close) as u8 as f64,
        (blocked && dist_far) as u8 as f64,
        (l.kind == 1) as u8 as f64,
        (l.kind == 2) as u8 as f64,
        (l.kind == 3) as u8 as f64,
        d.lanes
            .iter()
            .enumerate()
            .filter(|&(i, x)| i != lane && x.kind != 0)
            .count() as f64,
        d.lanes.iter().filter(|x| x.kind == 0).count() as f64,
    ]
}

/// Piece id ("I".."L") → the `spoken()` fill ordinal.
pub fn tetris_piece_fill(id: &str) -> u8 {
    let p = Piece::ALL
        .iter()
        .find(|p| p.id() == id)
        .unwrap_or_else(|| panic!("unknown piece id {id:?}"));
    match p {
        Piece::I => 0,
        Piece::O => 1,
        Piece::T => 2,
        Piece::S => 3,
        Piece::Z => 4,
        Piece::J => 5,
        Piece::L => 6,
    }
}

/// The state sentence's semantic fills, computed the way
/// `render_state_sentence` computes them. `Ok` = the SPREAD shape (fills
/// [hw, tallest, lowest, holes2, piece], template 0); `Err` = the FLAT
/// shape (fills [hw, holes2, piece], template 1). The spread's region
/// ordinals are 0 = left / 1 = middle / 2 = right (`TETRIS_SIDE3` order).
pub fn tetris_state_forward(
    board: &Board,
    piece_fill: u8,
) -> Result<[u8; 5], [u8; 3]> {
    let h = board.heights();
    let max_h = *h.iter().max().unwrap_or(&0);
    let hw = match max_h {
        0..=4 => 0,
        5..=10 => 1,
        _ => 2,
    };
    let holes2 = match board.hole_count() {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        _ => 4,
    };
    let avg = |r: std::ops::Range<usize>| -> u32 {
        let len = r.len() as u32;
        r.map(|c| h[c] as u32).sum::<u32>() / len
    };
    let (l, m, r) = (avg(0..3), avg(3..7), avg(7..10));
    let spread = l.max(m).max(r) - l.min(m).min(r);
    if spread >= 3 {
        let mut v = [(l, 0u8), (m, 1u8), (r, 2u8)];
        v.sort_unstable_by_key(|&(x, _)| x);
        Ok([hw, v[2].1, v[0].1, holes2, piece_fill])
    } else {
        Err([hw, holes2, piece_fill])
    }
}

/// The v4 state sentence's semantic fills (plan 609 T1.3): the v2 forward
/// plus the preview fill. `Ok` = SPREAD ([hw, tallest, lowest, holes2,
/// piece, next]); `Err` = FLAT ([hw, holes2, piece, next]). Composes
/// [`tetris_state_forward`] — one copy of the band math.
pub fn tetris_state_forward_v4(
    board: &Board,
    piece_fill: u8,
    next_fill: u8,
) -> Result<[u8; 6], [u8; 4]> {
    match tetris_state_forward(board, piece_fill) {
        Ok([hw, tallest, lowest, holes2, piece]) => {
            Ok([hw, tallest, lowest, holes2, piece, next_fill])
        }
        Err([hw, holes2, piece]) => Err([hw, holes2, piece, next_fill]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetris_sim::render_spot_sentence;

    #[test]
    fn all_tables_verify_closed_over_the_full_space() {
        verify_all_closed().expect("all five tables ambiguity-free over their fill products");
    }

    #[test]
    fn tetris_spot_round_trip() {
        let g = tetris_spot();
        let rendered = g.render(0, &[1, 2, 1, 1, 1]);
        assert_eq!(
            rendered,
            "The piece leaves one hole under it in the middle, makes a small bump \
             on top, and the stack stands medium, and clears a line."
        );
        assert_eq!(decode_tetris_spot(&g, &rendered).unwrap(), [1, 2, 1, 1, 1]);
        // The empty-clears fill renders as nothing.
        let no_clear = g.render(0, &[0, 4, 3, 0, 0]);
        assert!(no_clear.ends_with("the stack stays low."));
        assert_eq!(decode_tetris_spot(&g, &no_clear).unwrap(), [0, 4, 3, 0, 0]);
    }

    #[test]
    fn tetris_spot_forward_matches_the_real_renderer() {
        // Any placement's forward fills must decode its true sentence back.
        let board = Board::from_strings(&["..........", "..XX....XX"]);
        let piece = Piece::T;
        for p in tetris_sim::landing_options(&board, piece) {
            let f = tetris_sim::outcome_features(&board, &p);
            let sentence = render_spot_sentence(&board, &p, &f);
            let g = tetris_spot();
            let dec = decode_tetris_spot(&g, &sentence)
                .unwrap_or_else(|e| panic!("{sentence:?}: {e:?}"));
            assert_eq!(
                dec,
                tetris_spot_forward(&board, &p, &f),
                "sentence: {sentence:?}"
            );
        }
    }

    #[test]
    fn flappy_state_semantics_are_exact() {
        let g = flappy_state();
        let s = FlappyState {
            y: 5,
            v: -1,
            g: 6,
            h: 2,
        };
        let (r, m, gp) = flappy_state_forward(&s);
        let rendered = g.render(0, &[r, m as u8, gp as u8]);
        assert_eq!(
            rendered,
            "The bird is a little below the gap center, falling. The gap is narrow. \
             The pipe is just ahead."
        );
        let (rel, v, h) = decode_flappy_state(&g, &rendered).expect("decodes");
        // forward returns FILL indices; decode returns (rel fill, v, h).
        let (rf, mf, gf) = flappy_state_forward(&s);
        assert_eq!(
            (rel, v, h),
            (rf, mf as i32 - 2, if gf == 0 { 2 } else { 3 }),
            "state fills must round-trip to the exact semantic band/v/h"
        );
        assert_eq!(v, s.v, "motion vocab is bijective on the domain");
        assert_eq!(h, s.h, "gap vocab is bijective on the domain");
    }

    #[test]
    fn flappy_option_semantics() {
        // The FROZEN v2 table: the committed v2 fixture's decode contract.
        let g = flappy_option();
        let s = FlappyState {
            y: 4,
            v: 0,
            g: 6,
            h: 2,
        };
        // Coast: y2 = 4, rel = -2 = -h → SqueezeBottom (ordinal 1).
        let sent = flappy_sim::render_option_sentence_v2(&s, Action::Coast);
        assert_eq!(sent, "The bird squeezes through the bottom of the gap.");
        assert_eq!(decode_flappy_option(&g, &sent).unwrap(), 1);
        assert_eq!(flappy_option_forward(&s, Action::Coast), 1);
    }

    #[test]
    fn flappy_v3_option_semantics() {
        // The v3 table: band + offset + post-motion, decode == forward and
        // re-render byte-identical on a sample of the full fill product.
        let g = flappy_option_v3();
        let s = FlappyState {
            y: 4,
            v: 0,
            g: 6,
            h: 2,
        };
        // Coast: y2 = 4 → (SqueezeBottom, under-the-center); post-v = −1 →
        // down-one-step.
        let sent = flappy_sim::render_option_sentence(&s, Action::Coast);
        assert_eq!(
            sent,
            "The bird squeezes through the bottom of the gap, under the center, \
             drifting down one step."
        );
        let fills = [1u8, 0, 1];
        assert_eq!(decode_flappy_option_v3(&g, &sent).unwrap(), fills);
        assert_eq!(flappy_option_forward_v3(&s, Action::Coast), fills);
        assert_eq!(g.render(0, &fills), sent);
        // Flap: y2 = 6 = g → (Middle, at-the-center); post-v = +2 → up-two.
        let sent_f = flappy_sim::render_option_sentence(&s, Action::Flap);
        assert_eq!(
            sent_f,
            "The bird glides through the middle of the gap, at the center, \
             drifting up two steps."
        );
        let fills_f = [3u8, 2, 4];
        assert_eq!(decode_flappy_option_v3(&g, &sent_f).unwrap(), fills_f);
        assert_eq!(flappy_option_forward_v3(&s, Action::Flap), fills_f);
        assert_eq!(g.render(0, &fills_f), sent_f);
    }

    #[test]
    fn flappy_v3_option_forward_matches_the_renderer_over_enumerated_states() {
        // Any enumerated state's two option sentences must decode to the
        // forward fills and re-render byte-identically — the drift detector
        // between the table and the renderer.
        let g = flappy_option_v3();
        for (id, s) in flappy_sim::enumerate_states(607, 60) {
            for &a in flappy_sim::ACTIONS.iter() {
                let sent = flappy_sim::render_option_sentence(&s, a);
                let dec = decode_flappy_option_v3(&g, &sent)
                    .unwrap_or_else(|e| panic!("{id} {a:?}: {e:?}"));
                assert_eq!(dec, flappy_option_forward_v3(&s, a), "{id} {a:?}");
                assert_eq!(g.render(0, &dec), sent, "{id} {a:?}: re-render");
            }
        }
    }

    #[test]
    fn flappy_v3_reconstruction_is_exact_where_the_render_is_exact() {
        // Per-column exactness law over the enumerated corpus: post_v,
        // pre_v, in_gap and gap_half are ALWAYS exact; post_rel/abs/edge
        // are exact except the Below/Above tails (which pin to ±(h+1));
        // pre_rel is exact except |pre_rel| ≥ 2 (which clamps to ±2).
        let g = flappy_option_v3();
        let mut n_exact = 0usize;
        for (id, s) in flappy_sim::enumerate_states(607, 100) {
            let (rel_fill, _, _) = flappy_state_forward(&s);
            for &a in flappy_sim::ACTIONS.iter() {
                let sent = flappy_sim::render_option_sentence(&s, a);
                let post =
                    decode_flappy_option_v3(&g, &sent).unwrap_or_else(|e| panic!("{id}: {e:?}"));
                let dec = flappy_v3_decoded_features(post, rel_fill, s.v, s.h);
                let st = flappy_sim::feature_row(&s, a);
                let rel2 = st[0] as i32;
                let tail = rel2.abs() > s.h;
                let pre_far = (s.y - s.g).abs() >= 2;
                assert_eq!(dec[2], st[2], "{id} {a:?}: post_v");
                assert_eq!(dec[4], st[4], "{id} {a:?}: pre_v");
                assert_eq!(dec[5], st[5], "{id} {a:?}: in_gap");
                assert_eq!(dec[7], st[7], "{id} {a:?}: gap_half");
                if !tail {
                    assert_eq!(dec[0], st[0], "{id} {a:?}: post_rel");
                    assert_eq!(dec[1], st[1], "{id} {a:?}: post_abs");
                    assert_eq!(dec[6], st[6], "{id} {a:?}: edge_margin");
                } else {
                    let b = (s.h + 1) * rel2.signum();
                    assert_eq!(dec[0] as i32, b, "{id} {a:?}: tail pins to ±(h+1)");
                    assert_eq!(dec[1] as i32, s.h + 1, "{id} {a:?}: tail abs");
                }
                if !pre_far {
                    assert_eq!(dec[3], st[3], "{id} {a:?}: pre_rel");
                } else {
                    assert_eq!(
                        dec[3] as i32,
                        2 * (s.y - s.g).signum(),
                        "{id} {a:?}: pre_rel clamps to ±2"
                    );
                }
                if !tail && !pre_far {
                    assert_eq!(dec, st, "{id} {a:?}: fully exact row");
                    n_exact += 1;
                }
            }
        }
        assert!(n_exact > 0, "the corpus must contain fully-exact rows");
    }

    #[test]
    fn lanes_decoded_features_equal_the_structured_row() {
        let g = lanes_option();
        let s = LanesState {
            lanes: [
                lanes_sim::LaneObs {
                    kind: lanes_sim::ObsKind::Clear,
                    dist: lanes_sim::Dist::Close,
                },
                lanes_sim::LaneObs {
                    kind: lanes_sim::ObsKind::Train,
                    dist: lanes_sim::Dist::Close,
                },
                lanes_sim::LaneObs {
                    kind: lanes_sim::ObsKind::Rock,
                    dist: lanes_sim::Dist::Far,
                },
            ],
        };
        let sents = [
            lanes_sim::render_option_sentence(&s, 0),
            lanes_sim::render_option_sentence(&s, 1),
            lanes_sim::render_option_sentence(&s, 2),
        ];
        let dec = decode_lanes_state(&g, &[&sents[0], &sents[1], &sents[2]]).expect("decodes");
        assert_eq!(dec, lanes_forward(&s), "decode == semantic forward");
        for lane in 0..3 {
            assert_eq!(
                lanes_decoded_features(&dec, lane),
                lanes_sim::feature_row(&s, lane),
                "lane {lane}: the decoded row must EQUAL the structured row exactly"
            );
        }
    }

    #[test]
    fn lanes_lane_slot_names_the_lane() {
        let g = lanes_option();
        let d =
            decode_lanes_option(&g, "The right lane is blocked far ahead by a rock.").expect("ok");
        assert_eq!(d.lane, 2);
        assert_eq!(d.kind, 3);
        assert_eq!(d.dist, Some(1));
    }

    #[test]
    fn tetris_v4_state_round_trip_both_shapes() {
        // The v4 table renders exactly what the preview renderer spells,
        // decodes back to the forward fills, and re-renders byte-identically
        // — whichever shape the board takes (plan 609 T1.3).
        let g = tetris_state_v4();
        let board = Board::from_strings(&["..........", "..XX....XX"]);
        let (mut spread_n, mut flat_n) = (0usize, 0usize);
        for piece in Piece::ALL {
            for next in Piece::ALL {
                let rendered =
                    tetris_sim::render_state_sentence_with_preview(&board, piece, next);
                let fwd = tetris_state_forward_v4(
                    &board,
                    tetris_piece_fill(piece.id()),
                    tetris_piece_fill(next.id()),
                );
                match (&decode_tetris_state_v4(&g, &rendered), &fwd) {
                    (Ok(dec), Ok(f)) => {
                        spread_n += 1;
                        assert_eq!(dec, f, "{piece:?}>{next:?}: decode == forward");
                        assert_eq!(g.render(0, f), rendered, "{piece:?}>{next:?}: re-render");
                    }
                    (Err(dec), Err(f)) => {
                        flat_n += 1;
                        assert_eq!(dec, f, "{piece:?}>{next:?}: decode == forward");
                        assert_eq!(g.render(1, f), rendered, "{piece:?}>{next:?}: re-render");
                    }
                    _ => panic!("{piece:?}>{next:?}: shape disagreement decode vs forward"),
                }
            }
        }
        // Shape coverage (both shapes exercised) is the archetype-board
        // test's job — this board is flat-shaped throughout.
    }

    #[test]
    fn tetris_v4_state_forward_matches_the_renderer_over_all_boards() {
        // Drift detector over the archetype board shapes + every piece pair:
        // whatever the renderer says, the forward must decode it back.
        let g = tetris_state_v4();
        let mut spread_n = 0usize;
        let mut flat_n = 0usize;
        for heights in [
            [3usize; 10],
            [14, 13, 12, 10, 8, 6, 5, 4, 3, 2],
            [2, 3, 4, 5, 6, 8, 10, 12, 13, 14],
            [2, 4, 7, 9, 11, 11, 9, 7, 4, 2],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ] {
            let mut b = Board::empty();
            for (c, &h) in heights.iter().enumerate() {
                for r in (tetris_sim::HEIGHT - h)..tetris_sim::HEIGHT {
                    b.place(&[(r, c)]);
                }
            }
            for piece in Piece::ALL {
                for next in Piece::ALL {
                    let rendered =
                        tetris_sim::render_state_sentence_with_preview(&b, piece, next);
                    let fwd = tetris_state_forward_v4(
                        &b,
                        tetris_piece_fill(piece.id()),
                        tetris_piece_fill(next.id()),
                    );
                    match fwd {
                        Ok(f) => {
                            spread_n += 1;
                            assert_eq!(
                                decode_tetris_state_v4(&g, &rendered).unwrap(),
                                f,
                                "{rendered:?}"
                            );
                            assert_eq!(g.render(0, &f), rendered);
                        }
                        Err(e) => {
                            flat_n += 1;
                            assert_eq!(
                                decode_tetris_state_v4(&g, &rendered).unwrap_err(),
                                e,
                                "{rendered:?}"
                            );
                            assert_eq!(g.render(1, &e), rendered);
                        }
                    }
                }
            }
        }
        assert!(spread_n > 0 && flat_n > 0, "both shapes must be exercised");
    }
}
