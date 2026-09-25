//! Shared Tetris-arena substrate — Plan 607 (the `tests/common/ab_timing.rs`
//! `#[path]` precedent, one level out): the fixture schema + drift detector,
//! the seeded play loop and the Dellacherie policy, so the T4
//! sentence-cosine arena (`tetris_02_option_arena`) and the T3 fitted-head
//! arena (`tetris_03_head_fit`) consume ONE copy and cannot drift apart.
//!
//! Nothing here scores — policy is a `&dyn Fn(&Recomputed) -> usize` at the
//! seams; the scoring primitives live in katgpt-core (T1) and the
//! per-arena embed/fit glue lives in each example.

// Each arena compiles this module separately, and each consumes a different
// HALF of the facade (tetris_02 the sentence path, tetris_03 the structured
// path) — per-consumer dead-code warns would fire on the other half.
#![allow(dead_code)]

use serde::Deserialize;
use std::path::PathBuf;

#[path = "hash_embed.rs"]
pub mod hash_embed;
#[path = "tetris_sim.rs"]
pub mod tetris_sim;

// re-export facade: each arena consumer takes what it needs (tetris_02
// embeds sentences, tetris_03 reads structured features), so unused warns
// would fire per-consumer — allowed here by design.
#[allow(unused_imports)]
pub use hash_embed::{EMBED_DIM, embed};
pub use tetris_sim::{
    Board, DropRule, Piece, dellacherie_score, landing_options_with, outcome_features,
    render_spot_sentence, render_state_sentence,
};

pub fn default_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tetris_oracle_laya_en_v2.jsonl")
}

// ── Fixture schema ───────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct FixtureState {
    pub state_id: String,
    pub grammar: String,
    pub state_sentence: String,
    pub board: Vec<String>,
    pub piece: String,
    pub options: Vec<FixtureOption>,
    pub argmax: usize,
}

#[derive(Deserialize)]
pub struct FixtureOption {
    pub rot: usize,
    pub col: usize,
    pub row: usize,
    pub cells: Vec<(usize, usize)>,
    pub features: FixtureFeatures,
    pub sentence: String,
    #[serde(default)]
    pub p_clean: Option<f64>,
}

#[derive(Deserialize)]
pub struct FixtureFeatures {
    pub lines_cleared: u32,
    pub holes: u32,
    pub holes_delta: i32,
    pub bumpiness: u32,
    pub max_height: u32,
    pub aggregate_height: u32,
    pub landing_height: f32,
    pub row_transitions: u32,
    pub col_transitions: u32,
    pub cumulative_wells: u32,
    pub eroded_cells: u32,
}

impl FixtureFeatures {
    fn matches(&self, f: &tetris_sim::OutcomeFeatures) -> bool {
        self.lines_cleared == f.lines_cleared
            && self.holes == f.holes
            && self.holes_delta == f.holes_delta
            && self.bumpiness == f.bumpiness
            && self.max_height == f.max_height
            && self.aggregate_height == f.aggregate_height
            && self.landing_height == f.landing_height
            && self.row_transitions == f.row_transitions
            && self.col_transitions == f.col_transitions
            && self.cumulative_wells == f.cumulative_wells
            && self.eroded_cells == f.eroded_cells
    }
}

pub fn piece_from_id(id: &str) -> Piece {
    *Piece::ALL
        .iter()
        .find(|p| p.id() == id)
        .unwrap_or_else(|| panic!("unknown piece id {id:?}"))
}

/// The drift detector: recompute EVERYTHING an arena consumes from
/// (board, piece) and demand byte-identity with the fixture — options
/// (pinned order: rot/col/row/cells), outcome features (exact), and both
/// sentence layers (the closed grammar). Returns the recomputed state.
pub fn recompute_and_verify(st: &FixtureState) -> Result<Recomputed, String> {
    // The drop rule is part of the grammar (v2 DeepestFit, v3 FromTop —
    // Issue 884): recompute under the rule the fixture was dumped with.
    let rule = DropRule::from_grammar(&st.grammar)
        .ok_or_else(|| format!("{}: unknown grammar {:?}", st.state_id, st.grammar))?;
    let board = Board::from_strings(&st.board.iter().map(String::as_str).collect::<Vec<_>>());
    let piece = piece_from_id(&st.piece);
    let options = landing_options_with(&board, piece, rule);
    if options.len() != st.options.len() {
        return Err(format!(
            "{}: option count drifted (fixture {}, recomputed {})",
            st.state_id,
            st.options.len(),
            options.len()
        ));
    }
    let mut spot_sentences = Vec::with_capacity(options.len());
    for (p, fo) in options.iter().zip(&st.options) {
        if p.rot != fo.rot || p.col != fo.col || p.row != fo.row || p.cells[..] != fo.cells[..] {
            return Err(format!(
                "{}: placement drifted at rot {} col {}",
                st.state_id, fo.rot, fo.col
            ));
        }
        let f = outcome_features(&board, p);
        if !fo.features.matches(&f) {
            return Err(format!("{}: outcome features drifted", st.state_id));
        }
        let sentence = render_spot_sentence(&board, p, &f);
        if sentence != fo.sentence {
            return Err(format!(
                "{}: spot sentence drifted\n  fixture:   {:?}\n  recomputed: {:?}",
                st.state_id, fo.sentence, sentence
            ));
        }
        spot_sentences.push(sentence);
    }
    let state_sentence = render_state_sentence(&board, piece);
    if state_sentence != st.state_sentence {
        return Err(format!(
            "{}: state sentence drifted\n  fixture:    {:?}\n  recomputed: {:?}",
            st.state_id, st.state_sentence, state_sentence
        ));
    }
    Ok(Recomputed {
        board,
        options,
        spot_sentences,
        state_sentence,
    })
}

#[derive(Clone)]
pub struct Recomputed {
    pub board: Board,
    pub options: Vec<tetris_sim::Placement>,
    pub spot_sentences: Vec<String>,
    pub state_sentence: String,
}

/// Parse the fixture file (skipping the `_meta` provenance record) and
/// drift-check every state. Panics on drift — the fixture is
/// provenance-digested and must recompute byte-identically before any
/// scoring happens.
pub fn load_fixture_states(path: &PathBuf) -> Vec<(FixtureState, Recomputed)> {
    let raw = std::fs::read_to_string(path).expect("read fixture");
    let mut states = Vec::new();
    for (ln, line) in raw.lines().enumerate() {
        let value: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("fixture line {}: {e}", ln + 1));
        if value["state_id"] == "_meta" {
            continue; // the provenance record — not a state
        }
        let f: FixtureState = serde_json::from_value(value)
            .unwrap_or_else(|e| panic!("fixture line {}: {e}", ln + 1));
        let re = recompute_and_verify(&f).unwrap_or_else(|e| panic!("DRIFT: {e}"));
        states.push((f, re));
    }
    states
}

// ── Shared play loop + the classic policy ────────────────────────────────

/// The one drop rule every state of a loaded fixture shares (panics on a
/// mixed or unknown-grammar fixture — the drift detector's standing).
pub fn fixture_rule(states: &[(FixtureState, Recomputed)]) -> DropRule {
    let first = states.first().expect("fixture has no states");
    let rule = DropRule::from_grammar(&first.0.grammar).expect("unknown grammar");
    assert!(
        states.iter().all(|(f, _)| f.grammar == first.0.grammar),
        "fixture mixes grammars"
    );
    rule
}

/// One game under `policy` over a pre-generated piece stream, landing
/// pieces under `rule` (pass [`fixture_rule`] so live play matches the
/// fixture's grammar); returns (lines cleared, placements). Top-out or
/// stream exhaustion ends it.
pub fn play_game(
    policy: &dyn Fn(&Recomputed) -> usize,
    pieces: &[Piece],
    rule: DropRule,
) -> (u32, usize) {
    let mut board = Board::empty();
    let mut cleared = 0u32;
    for (n, &piece) in pieces.iter().enumerate() {
        let options = landing_options_with(&board, piece, rule);
        if options.is_empty() {
            return (cleared, n); // top-out
        }
        let mut spot_sentences = Vec::with_capacity(options.len());
        for p in &options {
            let f = outcome_features(&board, p);
            spot_sentences.push(render_spot_sentence(&board, p, &f));
        }
        let re = Recomputed {
            state_sentence: render_state_sentence(&board, piece),
            board,
            options,
            spot_sentences,
        };
        let pick = policy(&re);
        let mut after = re.board.clone();
        let cells = re.options[pick].cells;
        after.place(&cells);
        let full = after.full_rows();
        cleared += full.len() as u32;
        after.clear_rows(&full);
        board = after;
    }
    (cleared, pieces.len())
}

/// The classic heuristic's decision (the pinned lowest-index tie-break,
/// same shape as tetris_01's greedy_pick).
pub fn dellacherie_pick(re: &Recomputed) -> usize {
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, p) in re.options.iter().enumerate() {
        let s = dellacherie_score(&outcome_features(&re.board, p));
        if s > best_score {
            best_score = s;
            best = i;
        }
    }
    best
}
