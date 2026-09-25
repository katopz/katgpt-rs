//! Tetris state enumerator + laya-format dump — katgpt-rs Plan 607 T4a.
//!
//! Enumerates a stratified, fully deterministic fixture of Tetris decision
//! states (authored board archetypes × every piece, plus a seeded
//! Dellacherie-greedy play ladder), renders the pinned laya-protocol
//! sentences (`laya-tetris-v1`, see `common/tetris_sim.rs`), and dumps one
//! JSONL record per state:
//!
//! ```json
//! {
//!   "state_id": "arch:left_stack:I",
//!   "grammar": "laya-tetris-v1",
//!   "question": "Does the stack look clean?",
//!   "state_sentence": "The stack stands tall, tall on the left and low on the right. …",
//!   "board": ["..........", "…"],
//!   "piece": "I",
//!   "options": [
//!     {"rot": 0, "col": 0, "row": 18, "cells": [[18,0]], "features": {...},
//!      "sentence": "The piece leaves one hole under it and makes a small bump on top."}
//!   ]
//! }
//! ```
//!
//! The option ORDER is pinned (rotation asc, then column asc) — the T0b
//! oracle's and the T4 arena's argmax index both refer to it. The run ends
//! with a BLAKE3 digest on stderr — that hash is the provenance anchor the
//! T0b fixture records (same seed in, same bytes out, on every box).
//!
//! Run: `cargo run --release --example tetris_01_state_enum [-- --out <path> --seed <u64> --grammar laya-tetris-v3]`
//! (`--grammar` defaults to the pinned `laya-tetris-v2`; `laya-tetris-v3` is the
//! same grammar under a real hard drop, `DropRule::FromTop` — Issue 884).
//! Tests: `cargo test --example tetris_01_state_enum` (the common module's
//! closed-grammar / enumeration unit tests).

use std::path::PathBuf;

use serde::Serialize;

#[path = "common/tetris_sim.rs"]
mod tetris_sim;

use tetris_sim::{
    Board, DropRule, Piece, SPOT_QUESTION, GRAMMAR_ID_V4, dellacherie_score,
    landing_options_with, outcome_features, render_spot_sentence, render_state_sentence,
    render_state_sentence_with_preview, HEIGHT, WIDTH,
};

// ── Dump record shape ────────────────────────────────────────────────────

#[derive(Serialize)]
struct OptionRecord {
    rot: usize,
    col: usize,
    row: usize,
    cells: Vec<(usize, usize)>,
    features: FeatureRecord,
    sentence: String,
}

#[derive(Serialize)]
struct FeatureRecord {
    lines_cleared: u32,
    holes: u32,
    holes_delta: i32,
    bumpiness: u32,
    max_height: u32,
    aggregate_height: u32,
    landing_height: f32,
    row_transitions: u32,
    col_transitions: u32,
    cumulative_wells: u32,
    eroded_cells: u32,
}

impl From<tetris_sim::OutcomeFeatures> for FeatureRecord {
    fn from(f: tetris_sim::OutcomeFeatures) -> Self {
        Self {
            lines_cleared: f.lines_cleared,
            holes: f.holes,
            holes_delta: f.holes_delta,
            bumpiness: f.bumpiness,
            max_height: f.max_height,
            aggregate_height: f.aggregate_height,
            landing_height: f.landing_height,
            row_transitions: f.row_transitions,
            col_transitions: f.col_transitions,
            cumulative_wells: f.cumulative_wells,
            eroded_cells: f.eroded_cells,
        }
    }
}

#[derive(Serialize)]
struct StateRecord {
    state_id: String,
    grammar: &'static str,
    question: &'static str,
    state_sentence: String,
    board: Vec<String>,
    piece: &'static str,
    options: Vec<OptionRecord>,
}

// ── Authored board archetypes (deterministic, stratified) ───────────────

/// Build a board from a per-column height profile.
fn board_from_heights(heights: &[usize; WIDTH]) -> Board {
    let mut b = Board::empty();
    for (c, &h) in heights.iter().enumerate() {
        for r in (HEIGHT - h)..HEIGHT {
            b.place(&[(r, c)]);
        }
    }
    b
}

/// A flat floor of `floor` with a single 1-wide well of depth 4 at
/// `well_col`, plus the open shaft column that keeps every row pre-clear.
fn well_heights(floor: usize, well_col: usize) -> [usize; WIDTH] {
    let mut h = [floor; WIDTH];
    h[well_col] = floor.saturating_sub(4);
    h[WIDTH - 1] = 0; // open shaft: no row can complete (pre-clear law)
    h
}

/// The authored archetype ladder: name → board. Every board is PRE-CLEAR
/// (no complete rows — real Tetris never carries them; each heights-based
/// profile keeps one 0-height shaft column so no row can complete) and
/// carries a distinct surface shape so the fixture spans the grammar's
/// bands. `holes_archetype` (covered pockets) is separate — height profiles
/// cannot express sub-surface cavities.
fn archetypes() -> Vec<(&'static str, Board)> {
    let shaft = |mut h: [usize; WIDTH]| {
        h[WIDTH - 1] = 0;
        h
    };
    vec![
        ("empty", Board::empty()),
        ("floor_low", board_from_heights(&shaft([3; WIDTH]))),
        ("floor_high", board_from_heights(&shaft([8; WIDTH]))),
        (
            "left_stack",
            board_from_heights(&shaft([14, 13, 12, 10, 8, 6, 5, 4, 3, 2])),
        ),
        (
            "right_stack",
            board_from_heights(&shaft([2, 3, 4, 5, 6, 8, 10, 12, 13, 14])),
        ),
        (
            "center_mound",
            board_from_heights(&shaft([2, 4, 7, 9, 11, 11, 9, 7, 4, 2])),
        ),
        (
            "staircase",
            board_from_heights(&shaft([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
        ),
        (
            "jagged",
            board_from_heights(&shaft([9, 2, 8, 3, 7, 4, 6, 5, 10, 2])),
        ),
        ("well_left", board_from_heights(&well_heights(8, 0))),
        ("well_right", board_from_heights(&well_heights(8, 2))),
        ("well_center", board_from_heights(&well_heights(8, WIDTH / 2))),
        ("holes", holes_archetype()),
    ]
}

/// A plateau (height 8, columns 0–8) with the open shaft at col 9 and three
/// covered holes — hand-authored mask, pinned bytes: holes at (16, 2),
/// (17, 5), (16, 8).
fn holes_archetype() -> Board {
    let mut rows: Vec<String> = Vec::with_capacity(HEIGHT);
    for r in 0..HEIGHT {
        let mut row = String::with_capacity(WIDTH);
        for c in 0..WIDTH {
            let occupied = r >= HEIGHT - 8
                && c != WIDTH - 1
                && !((r == 16 && (c == 2 || c == 8)) || (r == 17 && c == 5));
            row.push(if occupied { '#' } else { '.' });
        }
        rows.push(row);
    }
    let refs: Vec<&str> = rows.iter().map(|s| s.as_str()).collect();
    Board::from_strings(&refs)
}

// ── Seeded Dellacherie-greedy play ladder ────────────────────────────────

/// Dellacherie argmax with the pinned tie-break (lowest index wins).
fn greedy_pick(feats: &[tetris_sim::OutcomeFeatures]) -> usize {
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, f) in feats.iter().enumerate() {
        let s = dellacherie_score(f);
        if s > best_score {
            best_score = s;
            best = i;
        }
    }
    best
}

/// Play deterministic games with `rng`; snapshot the board BEFORE every
/// placement from the 5th onward, up to `want` snapshots. Top-out ends the
/// game; the next game continues the same rng stream (still deterministic).
fn play_ladder(
    rng: &mut fastrand::Rng,
    want: usize,
    rule: DropRule,
) -> Vec<(String, Board, Piece)> {
    let mut out: Vec<(String, Board, Piece)> = Vec::with_capacity(want);
    let mut game = 0usize;
    while out.len() < want {
        let mut board = Board::empty();
        let mut placement_n = 0usize;
        loop {
            let piece = Piece::ALL[rng.usize(0..7)];
            let options = landing_options_with(&board, piece, rule);
            if options.is_empty() {
                break; // top-out — next game
            }
            let feats: Vec<_> = options.iter().map(|p| outcome_features(&board, p)).collect();
            placement_n += 1;
            if out.len() < want && placement_n >= 5 {
                out.push((format!("play{game:02}:{placement_n:03}"), board.clone(), piece));
            }
            let pick = greedy_pick(&feats);
            let mut after = board.clone();
            after.place(&options[pick].cells);
            let full = after.full_rows();
            after.clear_rows(&full);
            board = after;
            if placement_n >= 200 {
                break; // safety bound; greedy rarely reaches this
            }
        }
        game += 1;
    }
    out.shrink_to_fit();
    out
}

// ── Dump assembly ────────────────────────────────────────────────────────

/// `preview` is the v4 next-piece (`Some`) or the v2/v3 shape (`None`) —
/// the ONLY sentence delta between the grammars (plan 609 T1.2/T1.4).
fn build_record(
    state_id: String,
    board: &Board,
    piece: Piece,
    rule: DropRule,
    grammar: &'static str,
    preview: Option<Piece>,
) -> StateRecord {
    let options = landing_options_with(board, piece, rule);
    let mut recs = Vec::with_capacity(options.len());
    for p in &options {
        let f = outcome_features(board, p);
        let sentence = render_spot_sentence(board, p, &f);
        recs.push(OptionRecord {
            rot: p.rot,
            col: p.col,
            row: p.row,
            cells: p.cells.clone(),
            features: f.into(),
            sentence,
        });
    }
    let state_sentence = match preview {
        Some(next) => render_state_sentence_with_preview(board, piece, next),
        None => render_state_sentence(board, piece),
    };
    StateRecord {
        state_id,
        grammar,
        question: SPOT_QUESTION,
        state_sentence,
        board: board.to_strings(),
        piece: piece.id(),
        options: recs,
    }
}

// ── The oracle self-join (dump + laya decisions → committed fixture) ──────

/// A fixture option: the dump option plus laya's P(clean), in the committed
/// key order (the v2 fixture's bytes — `flatten` keeps the base order).
#[derive(Serialize)]
struct FixtureOptionOut<'a> {
    #[serde(flatten)]
    base: &'a OptionRecord,
    p_clean: f64,
}

#[derive(Serialize)]
struct FixtureStateOut<'a> {
    state_id: &'a str,
    grammar: &'static str,
    question: &'static str,
    state_sentence: &'a str,
    board: &'a [String],
    piece: &'static str,
    /// v4 only — the next-piece preview id. Skipped when absent so the
    /// v2/v3 join output stays byte-identical.
    #[serde(skip_serializing_if = "Option::is_none")]
    next_piece: Option<&'static str>,
    options: Vec<FixtureOptionOut<'a>>,
    argmax: usize,
}

#[derive(Serialize)]
struct FixtureMeta<'a> {
    state_id: &'static str,
    protocol: String,
    grammar: &'static str,
    question: &'static str,
    checkpoint: &'a str,
    generator: &'a str,
    dump_blake3: String,
    oracle_blake3_raw: &'a str,
    dump_command: String,
    notes: String,
}

/// The v4 fixture's `_meta` (plan 609 T1.4): the two-arm envelope, the bag
/// policy, the inherited drop rule, and the v3 baseline digest all disclosed.
#[derive(Serialize)]
struct V4FixtureMeta<'a> {
    state_id: &'static str,
    protocol: String,
    grammar: &'static str,
    question: &'static str,
    checkpoint: &'a str,
    generator: &'a str,
    /// BLAKE3 of the arm-B full dump this fixture was joined from.
    dump_blake3: String,
    oracle_blake3_raw: &'a str,
    /// BLAKE3 of the arm-A (masked-envelope) oracle run, when supplied.
    oracle_arm_a_blake3: Option<&'a str>,
    /// BLAKE3 of the v3 baseline fixture the option sets were verified
    /// against (the inherited-drop-rule evidence).
    baseline_blake3: String,
    baseline_file: String,
    bag_policy: &'static str,
    envelope: &'static str,
    dump_command: String,
    notes: String,
}

/// One oracle-manifest record: the two-line ENVELOPE is `state_sentence`
/// (joined onto every option's forward by the oracle tool) plus the option
/// sentences (plan 609 T1.4/T1.6).
#[derive(Serialize)]
struct ManifestRecord<'a> {
    state_id: &'a str,
    question: &'static str,
    state_sentence: &'a str,
    options: Vec<ManifestOption<'a>>,
}

/// The oracle tool reads each option's `sentence` field (an object, not a
/// bare string).
#[derive(Serialize)]
struct ManifestOption<'a> {
    sentence: &'a str,
}

fn manifest_jsonl(records: &[ManifestRecord<'_>]) -> String {
    let mut buf = String::new();
    for r in records {
        buf.push_str(&serde_json::to_string(r).expect("serialize manifest record"));
        buf.push('\n');
    }
    buf
}

/// Where each state's P(clean) came from — a fresh oracle run, or carried
/// verbatim from a prior fixture whose option sentences are byte-identical.
struct Decisions {
    p_clean: Vec<f64>,
    argmax: usize,
}

/// Pinned lowest-index tie-break argmax (the oracle's own rule).
fn argmax_lowest(ps: &[f64]) -> usize {
    let mut bi = 0usize;
    let mut bp = f64::NEG_INFINITY;
    for (i, &p) in ps.iter().enumerate() {
        if p > bp {
            bp = p;
            bi = i;
        }
    }
    bi
}

fn read_jsonl_by_id(path: &PathBuf) -> std::collections::HashMap<String, serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("parse JSONL line"))
        .filter(|v| v["state_id"] != "_meta")
        .map(|v| (v["state_id"].as_str().expect("state_id").to_owned(), v))
        .collect()
}

fn f64_array(v: &serde_json::Value, what: &str) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("{what}: not an array"))
        .iter()
        .map(|p| p.as_f64().expect("f64"))
        .collect()
}

/// Resolve every state's decisions: the fresh oracle wins; a state it does
/// not cover is carried from `carry` iff every option sentence is
/// byte-identical there (the oracle's answer is a function of the sentence).
/// Also reports, for fresh states, how many sentence-identical options
/// reproduce the carried P(clean) bit-exactly — the numerics-parity read.
fn resolve_decisions(
    records: &[StateRecord],
    oracle: &std::collections::HashMap<String, serde_json::Value>,
    carry: Option<&std::collections::HashMap<String, serde_json::Value>>,
) -> (Vec<Decisions>, usize, usize, usize, usize) {
    let (mut fresh, mut carried, mut parity_same, mut parity_n) = (0, 0, 0, 0);
    let mut out = Vec::with_capacity(records.len());
    for r in records {
        let prior = carry.and_then(|c| c.get(&r.state_id));
        if let Some(o) = oracle.get(&r.state_id) {
            let ps = f64_array(&o["p_clean"], &r.state_id);
            assert_eq!(ps.len(), r.options.len(), "{}: oracle option count", r.state_id);
            let oarg = o["argmax"].as_u64().expect("oracle argmax") as usize;
            assert_eq!(argmax_lowest(&ps), oarg, "{}: oracle argmax vs its own p_clean", r.state_id);
            if let Some(pr) = prior {
                for (opt, (&p, po)) in r.options.iter().zip(ps.iter().zip(
                    pr["options"].as_array().expect("prior options"),
                )) {
                    if po["sentence"].as_str() == Some(opt.sentence.as_str()) {
                        parity_n += 1;
                        parity_same += usize::from(po["p_clean"].as_f64() == Some(p));
                    }
                }
            }
            fresh += 1;
            out.push(Decisions { p_clean: ps, argmax: oarg });
            continue;
        }
        let pr = prior.unwrap_or_else(|| {
            panic!("{}: not in the oracle and no --carry-from state", r.state_id)
        });
        let popts = pr["options"].as_array().expect("prior options");
        assert_eq!(popts.len(), r.options.len(), "{}: carry option count", r.state_id);
        for (opt, po) in r.options.iter().zip(popts) {
            assert_eq!(
                po["sentence"].as_str(),
                Some(opt.sentence.as_str()),
                "{}: sentence differs — cannot carry, re-run the oracle on it",
                r.state_id
            );
        }
        let ps: Vec<f64> = popts.iter().map(|o| o["p_clean"].as_f64().expect("p_clean")).collect();
        let arg = argmax_lowest(&ps);
        assert_eq!(pr["argmax"].as_u64(), Some(arg as u64), "{}: carried argmax", r.state_id);
        carried += 1;
        out.push(Decisions { p_clean: ps, argmax: arg });
    }
    (out, fresh, carried, parity_same, parity_n)
}

fn main() {
    let mut out_path = PathBuf::from("output/tetris_states/states.jsonl");
    let mut seed = 607u64;
    let mut rule = DropRule::DeepestFit; // v2 stays the default: its pinned digest reproduces
    let mut preview = false; // v4 — the state sentence gains the next-piece line
    let mut join: Option<PathBuf> = None;
    let mut carry_from: Option<PathBuf> = None;
    let mut join_arm_a: Option<PathBuf> = None;
    let mut baseline: Option<PathBuf> = None;
    let mut fixture_out: Option<PathBuf> = None;
    let mut oracle_blake3 = String::new();
    let mut generator = String::new();
    let mut checkpoint = String::from("english");
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" if i + 1 < args.len() => {
                i += 1;
                out_path = PathBuf::from(&args[i]);
            }
            "--seed" if i + 1 < args.len() => {
                i += 1;
                seed = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("--seed must be an integer");
                    std::process::exit(1);
                });
            }
            "--grammar" if i + 1 < args.len() => {
                i += 1;
                match args[i].as_str() {
                    "laya-tetris-v2" => {
                        rule = DropRule::DeepestFit;
                        preview = false;
                    }
                    "laya-tetris-v3" => {
                        rule = DropRule::FromTop;
                        preview = false;
                    }
                    "laya-tetris-v4" => {
                        rule = DropRule::FromTop;
                        preview = true;
                    }
                    _other => {
                        eprintln!(
                            "--grammar must be laya-tetris-v2, laya-tetris-v3 or laya-tetris-v4"
                        );
                        std::process::exit(1);
                    }
                }
            }
            "--join" if i + 1 < args.len() => {
                i += 1;
                join = Some(PathBuf::from(&args[i]));
            }
            "--carry-from" if i + 1 < args.len() => {
                i += 1;
                carry_from = Some(PathBuf::from(&args[i]));
            }
            "--join-arm-a" if i + 1 < args.len() => {
                i += 1;
                join_arm_a = Some(PathBuf::from(&args[i]));
            }
            "--baseline" if i + 1 < args.len() => {
                i += 1;
                baseline = Some(PathBuf::from(&args[i]));
            }
            "--fixture-out" if i + 1 < args.len() => {
                i += 1;
                fixture_out = Some(PathBuf::from(&args[i]));
            }
            "--oracle-blake3" if i + 1 < args.len() => {
                i += 1;
                oracle_blake3 = args[i].clone();
            }
            "--generator" if i + 1 < args.len() => {
                i += 1;
                generator = args[i].clone();
            }
            "--checkpoint" if i + 1 < args.len() => {
                i += 1;
                checkpoint = args[i].clone();
            }
            other => {
                eprintln!(
                    "Unknown arg: {other}. Usage: [--out <path>] [--seed <u64>] \
                     [--grammar laya-tetris-v2|laya-tetris-v3|laya-tetris-v4] \
                     [--join <arm-B oracle.jsonl> --fixture-out <path> --oracle-blake3 <hex> --generator <str> \
                     [--carry-from <prior fixture> | v4: --baseline <v3 fixture> --join-arm-a <arm-A oracle>]] \
                     [--checkpoint <name>]"
                );
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let grammar_id: &'static str = if preview {
        GRAMMAR_ID_V4
    } else {
        rule.grammar_id()
    };

    // 1) Authored archetypes × every piece (84 states).
    let mut base: Vec<(String, Board, Piece)> = Vec::new();
    for (name, board) in archetypes() {
        for piece in Piece::ALL {
            base.push((format!("arch:{name}:{}", piece.id()), board.clone(), piece));
        }
    }

    // 2) Seeded play ladder (36 states) — Dellacherie-greedy, deterministic
    // given the seed.
    let mut rng = fastrand::Rng::with_seed(seed);
    for (id, board, piece) in play_ladder(&mut rng, 36, rule) {
        base.push((id, board, piece));
    }

    // The records this run dumps: the 120 v3-shaped states, or the 840
    // paired v4 states (every base board × all 7 next pieces).
    let records: Vec<StateRecord> = if !preview {
        base.iter()
            .map(|(id, board, piece)| {
                build_record(id.clone(), board, *piece, rule, grammar_id, None)
            })
            .collect()
    } else {
        // The seeded bag walk (plan 609 T1.2): one full 7-piece bag dealt
        // per board in dump order, variants emitted in dealt order. An
        // independent fastrand stream from the same seed (the ladder's own
        // stream is consumed by play_ladder); deterministic, disclosed in
        // the fixture's _meta bag_policy.
        let mut bag_rng = fastrand::Rng::with_seed(seed);
        let mut recs = Vec::with_capacity(base.len() * 7);
        for (id, board, piece) in &base {
            let mut bag: [Piece; 7] = Piece::ALL;
            for k in (1..7).rev() {
                let j = bag_rng.usize(0..=k);
                bag.swap(k, j);
            }
            for &next in &bag {
                recs.push(build_record(
                    format!("{id}|next:{}", next.id()),
                    board,
                    *piece,
                    rule,
                    grammar_id,
                    Some(next),
                ));
            }
        }
        recs
    };

    let n_states = records.len();
    let n_options: usize = records.iter().map(|r| r.options.len()).sum();
    let min_options = records.iter().map(|r| r.options.len()).min().unwrap_or(0);

    if let Some(dir) = out_path.parent() {
        std::fs::create_dir_all(dir).expect("create out dir");
    }
    let mut buf = String::new();
    for r in &records {
        buf.push_str(&serde_json::to_string(r).expect("serialize record"));
        buf.push('\n');
    }
    std::fs::write(&out_path, &buf).expect("write dump");

    let digest = blake3::hash(buf.as_bytes());
    eprintln!(
        "dumped {n_states} states ({n_options} options, min {min_options}/state) -> {}",
        out_path.display()
    );
    eprintln!("blake3: {digest}");
    eprintln!("grammar: {}  seed: {seed}", grammar_id);

    // ── v4: also emit the two oracle-manifest arms (plan 609 T1.4) ──────
    if preview {
        let arm_a_path = sibling_path(&out_path, "_arm_a.jsonl");
        let arm_b_path = sibling_path(&out_path, "_arm_b.jsonl");

        // The v3-shaped base records (the masked envelope's state lines and
        // option sets; also the join's parent shape).
        let base_records: Vec<StateRecord> = base
            .iter()
            .map(|(id, board, piece)| {
                build_record(id.clone(), board, *piece, rule, grammar_id, None)
            })
            .collect();

        // Arm A — masked envelope: the v3 state sentence (no preview line)
        // on the two-line payload, plus ONE duplicated state as the oracle
        // run's determinism check.
        let mut arm_a: Vec<ManifestRecord> = Vec::with_capacity(base_records.len() + 1);
        for r in &base_records {
            arm_a.push(ManifestRecord {
                state_id: &r.state_id,
                question: SPOT_QUESTION,
                state_sentence: &r.state_sentence,
                options: r
                    .options
                    .iter()
                    .map(|o| ManifestOption { sentence: o.sentence.as_str() })
                    .collect(),
            });
        }
        {
            let r0 = &base_records[0];
            arm_a.push(ManifestRecord {
                state_id: "arch:empty:I|dup",
                question: SPOT_QUESTION,
                state_sentence: &r0.state_sentence,
                options: r0
                    .options
                    .iter()
                    .map(|o| ManifestOption { sentence: o.sentence.as_str() })
                    .collect(),
            });
        }
        // Arm B — the paired states: the preview line present.
        let arm_b: Vec<ManifestRecord> = records
            .iter()
            .map(|r| ManifestRecord {
                state_id: &r.state_id,
                question: SPOT_QUESTION,
                state_sentence: &r.state_sentence,
                options: r
                    .options
                    .iter()
                    .map(|o| ManifestOption { sentence: o.sentence.as_str() })
                    .collect(),
            })
            .collect();

        let arm_a_buf = manifest_jsonl(&arm_a);
        let arm_b_buf = manifest_jsonl(&arm_b);
        std::fs::write(&arm_a_path, &arm_a_buf).expect("write arm-A manifest");
        std::fs::write(&arm_b_path, &arm_b_buf).expect("write arm-B manifest");
        eprintln!(
            "arm A (masked envelope + dup): {} records -> {}",
            arm_a.len(),
            arm_a_path.display()
        );
        eprintln!("arm A blake3: {}", blake3::hash(arm_a_buf.as_bytes()));
        eprintln!(
            "arm B (preview envelope): {} records -> {}",
            arm_b.len(),
            arm_b_path.display()
        );
        eprintln!("arm B blake3: {}", blake3::hash(arm_b_buf.as_bytes()));
        eprintln!(
            "disclosure: bag policy = one full 7-piece bag per board (fastrand seed {seed}, \
             independent stream), variants in dealt order; envelope = state line + option line; \
             drop rule inherited from v3 (FromTop)"
        );
    }

    let Some(oracle_path) = join else { return };
    let fixture_out = fixture_out.expect("--join needs --fixture-out");
    assert!(
        oracle_blake3.len() == 64 && !generator.is_empty(),
        "--join needs --oracle-blake3 <hex> and --generator <str> (the oracle run's provenance)"
    );
    let oracle = read_jsonl_by_id(&oracle_path);
    if !preview {
        let carry = carry_from.as_ref().map(read_jsonl_by_id);
        let (decisions, fresh, carried, parity_same, parity_n) =
            resolve_decisions(&records, &oracle, carry.as_ref());
        assert_eq!(fresh, oracle.len(), "oracle carries states the dump does not");
        let version = grammar_id.rsplit('-').next().unwrap_or("");
        let carry_note = match &carry_from {
            Some(p) if carried > 0 => format!(
                "; {fresh} states oracled fresh, {carried} carried verbatim from {} (every option sentence byte-identical; the oracle answer is a function of the sentence) — fresh-state parity: {parity_same}/{parity_n} sentence-identical options reproduce the carried p_clean bit-exactly",
                p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()
            ),
            _ => String::new(),
        };
        let ties = ties_count(&decisions);
        let meta = FixtureMeta {
            state_id: "_meta",
            protocol: format!(
                "katgpt-rs Plan 607 T0b \u{2014} laya Tetris oracle fixture {version}"
            ),
            grammar: grammar_id,
            question: SPOT_QUESTION,
            checkpoint: &checkpoint,
            generator: &generator,
            dump_blake3: digest.to_hex().to_string(),
            oracle_blake3_raw: &oracle_blake3,
            dump_command: format!(
                "cargo run --release --example tetris_01_state_enum -- --grammar {} (katgpt-rs, seed {seed})",
                grammar_id
            ),
            notes: format!(
                "options array carries sentence+features+p_clean per option in the pinned order (rotation asc, col asc); argmax = laya decision with lowest-index tie-break; {ties}/{n_states} states have p_clean ties for the best option{carry_note}"
            ),
        };
        write_fixture(
            &fixture_out,
            &serde_json::to_string(&meta).expect("serialize meta"),
            records
                .iter()
                .zip(&decisions)
                .map(|(r, d)| {
                    serde_json::to_string(&FixtureStateOut {
                        state_id: &r.state_id,
                        grammar: r.grammar,
                        question: r.question,
                        state_sentence: &r.state_sentence,
                        board: &r.board,
                        piece: r.piece,
                        next_piece: None,
                        options: r
                            .options
                            .iter()
                            .zip(&d.p_clean)
                            .map(|(o, &p)| FixtureOptionOut { base: o, p_clean: p })
                            .collect(),
                        argmax: d.argmax,
                    })
                    .expect("serialize fixture record")
                })
                .collect(),
        );
        return;
    }

    // ── v4 join (plan 609 T1.8): arm-B oracle + v3 baseline ─────────────
    assert!(
        carry_from.is_none(),
        "--carry-from does not apply to v4 (the envelope differs; use --baseline)"
    );
    let baseline_path = baseline.unwrap_or_else(|| {
        PathBuf::from("tests/fixtures/tetris_oracle_laya_en_v3.jsonl")
    });
    let baseline = read_jsonl_by_id(&baseline_path);
    let baseline_bytes = std::fs::read(&baseline_path).expect("read baseline fixture");

    let mut decisions: Vec<Decisions> = Vec::with_capacity(records.len());
    let mut baseline_checked = 0usize;
    for r in &records {
        let o = oracle.get(&r.state_id).unwrap_or_else(|| {
            panic!("{}: missing from the arm-B oracle", r.state_id)
        });
        let ps = f64_array(&o["p_clean"], &r.state_id);
        assert_eq!(ps.len(), r.options.len(), "{}: oracle option count", r.state_id);
        let oarg = o["argmax"].as_u64().expect("oracle argmax") as usize;
        assert_eq!(
            argmax_lowest(&ps),
            oarg,
            "{}: oracle argmax vs its own p_clean",
            r.state_id
        );
        // The inherited-drop-rule evidence: the option set must be the v3
        // baseline's, byte for byte (the preview lives in the state line
        // only).
        let parent_id = r.state_id.split("|next:").next().expect("parent prefix");
        let b = baseline.get(parent_id).unwrap_or_else(|| {
            panic!("{}: parent {parent_id} missing from the v3 baseline", r.state_id)
        });
        let bopts = b["options"].as_array().expect("baseline options");
        assert_eq!(
            bopts.len(),
            r.options.len(),
            "{parent_id}: option count drifted from the v3 baseline"
        );
        for (opt, bo) in r.options.iter().zip(bopts) {
            assert_eq!(
                bo["sentence"].as_str(),
                Some(opt.sentence.as_str()),
                "{parent_id}: option sentence drifted from the v3 baseline"
            );
        }
        baseline_checked += 1;
        decisions.push(Decisions { p_clean: ps, argmax: oarg });
    }
    assert_eq!(
        decisions.len(),
        oracle.len(),
        "the arm-B oracle carries states the dump does not"
    );

    // Arm-A analysis (optional but expected): the dup determinism check and
    // the masked-envelope vs option-only-labels delta.
    let mut arm_a_blake3_hex: Option<String> = None;
    let mut arm_a_note = String::new();
    if let Some(arm_a_path) = &join_arm_a {
        let arm_a = read_jsonl_by_id(arm_a_path);
        let bytes = std::fs::read(arm_a_path).expect("read arm-A oracle");
        arm_a_blake3_hex = Some(blake3::hash(&bytes).to_hex().to_string());
        // Determinism: the duplicated state's forward must reproduce its
        // original bit-exactly.
        let dup_id = "arch:empty:I|dup";
        let (dup, orig) = (
            arm_a.get(dup_id).unwrap_or_else(|| panic!("{dup_id} missing")),
            arm_a
                .get("arch:empty:I")
                .unwrap_or_else(|| panic!("arch:empty:I missing")),
        );
        let dp = f64_array(&dup["p_clean"], dup_id);
        let op = f64_array(&orig["p_clean"], "arch:empty:I");
        assert_eq!(dp, op, "determinism: the duplicated state's forwards diverged");
        // Masked-envelope vs v3 option-only labels: identical sentences, so
        // any delta is the envelope's (the state line's) effect.
        let mut n_bit = 0usize;
        let mut n_seen = 0usize;
        let mut max_dp = 0.0f64;
        let mut agree = 0usize;
        let mut compared = 0usize;
        for (id, _board, _piece) in &base {
            let a = arm_a
                .get(id)
                .unwrap_or_else(|| panic!("arm-A oracle missing {id}"));
            let ps = f64_array(&a["p_clean"], id);
            let b = baseline
                .get(id)
                .unwrap_or_else(|| panic!("baseline missing {id}"));
            let bopts = b["options"].as_array().expect("baseline options");
            assert_eq!(ps.len(), bopts.len(), "{id}: option count");
            for (&p, bo) in ps.iter().zip(bopts) {
                let bp = bo["p_clean"].as_f64().expect("baseline p_clean");
                n_seen += 1;
                n_bit += usize::from(p == bp);
                max_dp = max_dp.max((p - bp).abs());
            }
            let barg = b["argmax"].as_u64().expect("baseline argmax") as usize;
            compared += 1;
            agree += usize::from(argmax_lowest(&ps) == barg);
        }
        arm_a_note = format!(
            "; arm A: dup determinism PASS (bit-exact), envelope-vs-v3 labels: \
             {n_bit}/{n_seen} p_clean bit-exact, max |\u{394}p| {max_dp:.3e}, argmax {agree}/{compared}"
        );
    }

    let ties = ties_count(&decisions);
    let n_boards = records.len() / 7;
    let meta = V4FixtureMeta {
        state_id: "_meta",
        protocol: "katgpt-rs Plan 609 T1.4/T1.8 \u{2014} laya Tetris v4 preview fixture".to_string(),
        grammar: grammar_id,
        question: SPOT_QUESTION,
        checkpoint: &checkpoint,
        generator: &generator,
        dump_blake3: digest.to_hex().to_string(),
        oracle_blake3_raw: &oracle_blake3,
        oracle_arm_a_blake3: arm_a_blake3_hex.as_deref(),
        baseline_blake3: blake3::hash(&baseline_bytes).to_hex().to_string(),
        baseline_file: baseline_path.display().to_string(),
        bag_policy: "one full 7-piece bag per board in dump order (fastrand, the dump seed's independent stream); variants emitted in dealt order; all 7 next pieces per board",
        envelope: "two-line payload: state_sentence line + option sentence line, joined by the oracle tool (laya_oracle_batch state_sentence field); arm A = the v3 state sentence (preview masked) + one duplicated state (determinism); arm B = the preview sentence present",
        dump_command: format!(
            "cargo run --release --example tetris_01_state_enum -- --grammar laya-tetris-v4 (katgpt-rs, seed {seed})"
        ),
        notes: format!(
            "options array carries sentence+features+p_clean per option in the pinned order (rotation asc, col asc), byte-identical to the v3 baseline's parent options ({baseline_checked}/{n_boards} parents verified); argmax = laya decision with lowest-index tie-break; {ties}/{n_states} states have p_clean ties for the best option{arm_a_note}"
        ),
    };
    write_fixture(
        &fixture_out,
        &serde_json::to_string(&meta).expect("serialize v4 meta"),
        records
            .iter()
            .zip(&decisions)
            .map(|(r, d)| {
                let next_piece = r
                    .state_id
                    .split("|next:")
                    .nth(1)
                    .map(|s| s.to_owned())
                    .expect("v4 state id carries |next:");
                // Leak-free &'static: the piece ids are the seven static
                // letters — map back through Piece rather than leaking a
                // local String.
                let next_static = Piece::ALL
                    .iter()
                    .find(|p| p.id() == next_piece)
                    .unwrap_or_else(|| panic!("unknown next piece {next_piece:?}"))
                    .id();
                serde_json::to_string(&FixtureStateOut {
                    state_id: &r.state_id,
                    grammar: r.grammar,
                    question: r.question,
                    state_sentence: &r.state_sentence,
                    board: &r.board,
                    piece: r.piece,
                    next_piece: Some(next_static),
                    options: r
                        .options
                        .iter()
                        .zip(&d.p_clean)
                        .map(|(o, &p)| FixtureOptionOut { base: o, p_clean: p })
                        .collect(),
                    argmax: d.argmax,
                })
                .expect("serialize fixture record")
            })
            .collect(),
    );
}

/// `<stem><suffix>` beside the given file (states.jsonl → states_arm_a.jsonl).
fn sibling_path(p: &std::path::Path, suffix: &str) -> PathBuf {
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("states");
    p.with_file_name(format!("{stem}{suffix}"))
}

fn ties_count(decisions: &[Decisions]) -> usize {
    decisions
        .iter()
        .filter(|d| {
            let best = d.p_clean[d.argmax];
            d.p_clean.iter().filter(|&&p| p == best).count() > 1
        })
        .count()
}

fn write_fixture(path: &std::path::Path, meta_line: &str, record_lines: Vec<String>) {
    let mut fx = String::from(meta_line);
    fx.push('\n');
    for line in record_lines {
        fx.push_str(&line);
        fx.push('\n');
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).expect("create fixture dir");
    }
    std::fs::write(path, &fx).expect("write fixture");
    eprintln!("fixture blake3: {}", blake3::hash(fx.as_bytes()));
}
