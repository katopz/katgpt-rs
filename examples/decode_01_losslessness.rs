//! Plan 607 T2 — the losslessness arm (bounded template decode).
//!
//! Decodes the three committed oracle fixtures' sentences through the
//! closed-grammar tables (`common/grammar_tables.rs`), re-fits the SAME
//! corpus recipe (standardize → λ by state-level LOO MSE → in-corpus + LOO
//! readings) over the DECODED feature rows, and reports the AGREEMENT
//! DELTA against the structured arm — the plan's two jobs, measured:
//!
//! - **(a) the losslessness measurement** — a non-zero delta is a finding
//!   about the RENDER (information the grammar never encoded), not
//!   automatically a decode bug: the decode layer itself is asserted exact
//!   before any scoring (every sentence decodes, re-renders
//!   byte-identically through the table, and lands on the renderer's own
//!   band fills).
//! - **(b) third-party laya-format traffic intake** — the durable consumer
//!   justification (R6: the sentence is the reference model's input
//!   requirement, not the task's).
//!
//! The flappy v2 section is the FROZEN Bench 880/881 record (its structured
//! anchors are asserted byte-identically). The flappy v3 section is Issue
//! 876's render widening (band + quantized offset + neutral post-motion),
//! measured against a freshly regenerated oracle over the IDENTICAL state
//! set — its gate: the decoded arm must beat constant-pick with ≥ 2
//! distinct picks.
//!
//! The structured-arm anchors are asserted against the published benches
//! (878 tetris, 880 flappy/lanes) — this run is also the proof that the
//! T2 width-genericized fit recipe (`micro_fit`) is arithmetic-identical
//! to the T3/T5 copies it generalizes.
//!
//! Re-run:
//! ```text
//! cargo run --release --features state_option_scoring,template_decode --example decode_01_losslessness
//! cargo test  --release --features state_option_scoring,template_decode --example decode_01_losslessness
//! ```

#![allow(dead_code)]

#[path = "common/tetris_sim.rs"]
mod tetris_sim;
#[path = "common/flappy_sim.rs"]
mod flappy_sim;
#[path = "common/lanes_sim.rs"]
mod lanes_sim;
#[path = "common/micro_dump.rs"]
mod micro_dump;
#[path = "common/micro_fit.rs"]
mod micro_fit;
#[path = "common/grammar_tables.rs"]
mod grammar_tables;

use katgpt_core::state_option_scoring::head::HeadFitter;
use micro_dump::MicroStateFixture;
use micro_fit::HeadCorpus;

use flappy_sim::FlappyState;
use lanes_sim::LanesState;
use tetris_sim::{Board, OutcomeFeatures, Placement, Piece};

use grammar_tables::{
    decode_flappy_option, decode_flappy_option_v3, decode_flappy_state, decode_lanes_state,
    decode_tetris_spot, flappy_decoded_features, flappy_option_forward, flappy_option_forward_v3,
    flappy_option_v3, flappy_state_forward, flappy_v3_decoded_features, lanes_decoded_features,
    lanes_forward, tetris_decoded_features, tetris_piece_fill, tetris_spot_forward,
    tetris_state_forward, verify_all_closed, FLAPPY_DECODED_F, FLAPPY_V3_DECODED_F,
    LANES_DECODED_F, TETRIS_DECODED_F,
};

// ── Published anchors (the structured arms this run must reproduce) ─────
//
// ⛔ ONE ANCHOR RE-PINNED 2026-09-25 (Plan 609 T1.5's G3 check, the
// Issue-884 follow-through): `TETRIS_HEAD_ANCHOR` was originally fitted
// from p_clean values parsed by serde_json's DEFAULT parser — measured 1
// ULP off on ~75% of fixture lines. The Issue-884 join added the dev-dep
// `float_roundtrip` feature (exact parsing — simply correct), which changed
// every example's parsed targets, and the tetris structured fit is the one
// whose weight BYTES moved: 65409c14… → b3c91ee0… (measured at the parent
// commit 1a05a9764 — decode_01 has been red since that commit landed; its
// in-corpus/LOO agreement numbers never moved, 36/35, only the digest).
// The flappy v2/v3 + lanes anchors MEASURED identical under both parsers
// (this run asserts them and passes). The old digest below is the
// parser-bug-era record; the new one is the exact-parse anchor.
// Cross-repo implication for the fixture_pins lane: a consumer fitting
// from these fixtures must parse with float_roundtrip to land on the
// same head bytes.

const TETRIS_HEAD_ANCHOR: &str =
    "b3c91ee05bde4086c3760eb917a3470884c9f47c951764bd63eb40d830083729";
const FLAPPY_HEAD_PREFIX: &str = "4ac0a13c";
const LANES_HEAD_PREFIX: &str = "7d3f1d8e";
// Issue 876 / Bench 882: the v3 render-widening fixture's heads (full digests
// — both arms of the new measurement, two-box portable per the T3 law).
const FLAPPY_V3_HEAD_ANCHOR: &str =
    "dc6bcf735ec7071b97efa6405adb92feb3603df2b2e8ec71277bf6bc95ab9fd2";
const FLAPPY_V3_DECODED_HEAD_ANCHOR: &str =
    "c93d36dc79c0490334c20353ce5d6479eaee3448b686ac057f8a4ad4b98ae3c5";

fn pct(n: usize, d: usize) -> String {
    format!("{:.1}%", 100.0 * n as f64 / d.max(1) as f64)
}

// ── The shared recipe, width-generic ─────────────────────────────────────

/// Corpus-side standardization through the shared recipe: fit the
/// Standardizer on the raw rows, then design every row (F standardized
/// features + intercept at index F).
fn standardize<const F: usize, const D: usize>(raws: &[[f64; F]]) -> Vec<[f64; D]> {
    let std = micro_fit::Standardizer::<F>::fit(raws);
    raws.iter().map(|r| std.design(r)).collect()
}

struct ArmReading {
    chosen: f64,
    in_agree: usize,
    loo_agree: usize,
    digest: blake3::Hash,
    loo_picks: Vec<usize>,
    /// distinct LOO picks — the discrimination floor (G1: never constant).
    loo_distinct: usize,
}

/// The tetris_03/T5 recipe, width-generic: standardize, select λ by
/// state-level LOO MSE (never the agreement number), report in-corpus + LOO.
fn run_arm<const D: usize>(
    rows: Vec<[f64; D]>,
    targets: Vec<f64>,
    offsets: Vec<usize>,
    argmaxes: &[usize],
) -> ArmReading {
    let corpus = HeadCorpus::<D> {
        rows,
        targets,
        offsets,
    };
    let mut fitter = HeadFitter::<D>::new();
    let (chosen, loo_picks, _lam_rows) = micro_fit::loo_select(&mut fitter, &corpus, argmaxes);
    let head = fitter.fit_into(&corpus.rows, &corpus.targets, chosen);
    let mut in_agree = 0usize;
    for (s, &arg) in argmaxes.iter().enumerate() {
        let (a, b) = (corpus.offsets[s], corpus.offsets[s + 1]);
        if head.pick(&corpus.rows[a..b], b - a) == arg {
            in_agree += 1;
        }
    }
    let loo_agree = loo_picks
        .iter()
        .zip(argmaxes.iter())
        .filter(|(p, a)| p == a)
        .count();
    let loo_distinct = loo_picks.iter().collect::<std::collections::HashSet<_>>().len();
    ArmReading {
        chosen,
        in_agree,
        loo_agree,
        digest: micro_fit::head_digest(&head),
        loo_picks,
        loo_distinct,
    }
}

fn offsets_of(counts: impl IntoIterator<Item = usize>) -> Vec<usize> {
    let mut out = vec![0usize];
    for c in counts {
        let next = out.last().unwrap() + c;
        out.push(next);
    }
    out
}

/// constant-pick (always the oracle-majority index) + mean 1/K chance —
/// the G1 baselines, computed per arena so the reading is self-contained.
fn baselines(argmaxes: &[usize], option_counts: &[usize]) -> (usize, usize, f64) {
    let mut hist = std::collections::HashMap::new();
    let mut chance = 0.0f64;
    for (&a, &k) in argmaxes.iter().zip(option_counts) {
        *hist.entry(a).or_insert(0) += 1;
        chance += 1.0 / k as f64;
    }
    let (idx, c) = hist.into_iter().max_by_key(|(_, c)| *c).unwrap();
    (idx, c, chance / argmaxes.len() as f64)
}

// ── Fixture loading + the decode-layer validation ────────────────────────

#[derive(serde::Deserialize)]
struct TetrisFixtureState {
    state_id: String,
    state_sentence: String,
    board: Vec<String>,
    piece: String,
    options: Vec<TetrisFixtureOption>,
    argmax: usize,
}

#[derive(serde::Deserialize)]
struct TetrisFixtureOption {
    rot: usize,
    col: usize,
    row: usize,
    cells: Vec<(usize, usize)>,
    sentence: String,
    #[serde(default)]
    p_clean: Option<f64>,
}

struct TetrisLoaded {
    states: Vec<TetrisFixtureState>,
    boards: Vec<Board>,
    options: Vec<Vec<Placement>>,
    features: Vec<Vec<OutcomeFeatures>>,
}

fn load_tetris(path: &std::path::Path) -> TetrisLoaded {
    let raw = std::fs::read_to_string(path).expect("read tetris fixture");
    let mut states = Vec::new();
    for (ln, line) in raw.lines().enumerate() {
        let v: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("fixture line {}: {e}", ln + 1));
        if v["state_id"] == "_meta" {
            continue;
        }
        let s: TetrisFixtureState =
            serde_json::from_value(v).unwrap_or_else(|e| panic!("fixture line {}: {e}", ln + 1));
        states.push(s);
    }
    let mut boards = Vec::with_capacity(states.len());
    let mut options = Vec::with_capacity(states.len());
    let mut features = Vec::with_capacity(states.len());
    for st in &states {
        let board = Board::from_strings(&st.board.iter().map(String::as_str).collect::<Vec<_>>());
        let piece = *Piece::ALL
            .iter()
            .find(|p| p.id() == st.piece)
            .unwrap_or_else(|| panic!("{}: unknown piece {}", st.state_id, st.piece));
        let opts = tetris_sim::landing_options(&board, piece);
        assert_eq!(
            opts.len(),
            st.options.len(),
            "{}: option count drifted",
            st.state_id
        );
        let mut fs = Vec::with_capacity(opts.len());
        for (p, fo) in opts.iter().zip(&st.options) {
            assert!(
                p.rot == fo.rot && p.col == fo.col && p.row == fo.row && p.cells[..] == fo.cells[..],
                "{}: placement drifted at rot {} col {}",
                st.state_id,
                fo.rot,
                fo.col
            );
            fs.push(tetris_sim::outcome_features(&board, p));
        }
        boards.push(board);
        options.push(opts);
        features.push(fs);
    }
    TetrisLoaded {
        states,
        boards,
        options,
        features,
    }
}

/// Decode-layer validation (tetris): every option sentence decodes,
/// re-renders byte-identically through the table, and lands on the
/// renderer's own band fills; every state sentence decodes to its semantic
/// shape and re-renders. Panics on any mismatch; returns counts.
fn validate_tetris(t: &TetrisLoaded) -> (usize, usize) {
    let g = grammar_tables::tetris_spot();
    let gs = grammar_tables::tetris_state();
    let mut n_opts = 0usize;
    let mut n_states = 0usize;
    for (s, st) in t.states.iter().enumerate() {
        for (oi, fo) in st.options.iter().enumerate() {
            let p = &t.options[s][oi];
            let f = &t.features[s][oi];
            let dec = decode_tetris_spot(&g, &fo.sentence)
                .unwrap_or_else(|e| panic!("{}: {:?}: {e:?}", st.state_id, fo.sentence));
            assert_eq!(
                dec,
                tetris_spot_forward(&t.boards[s], p, f),
                "{}: decoded fills != semantic forward\n  sentence: {:?}",
                st.state_id,
                fo.sentence
            );
            assert_eq!(
                g.render(0, &dec),
                fo.sentence,
                "{}: re-render drifted",
                st.state_id
            );
            n_opts += 1;
        }
        let piece_fill = tetris_piece_fill(&st.piece);
        let m = gs
            .decode(&st.state_sentence)
            .unwrap_or_else(|e| panic!("{}: state sentence: {e:?}", st.state_id));
        match (
            m.template,
            tetris_state_forward(&t.boards[s], piece_fill),
        ) {
            (0, Ok(fills)) => assert_eq!(
                &m.fills[..5],
                &fills,
                "{}: state fills != semantic forward (spread)",
                st.state_id
            ),
            (1, Err(fills)) => assert_eq!(
                &m.fills[..3],
                &fills,
                "{}: state fills != semantic forward (flat)",
                st.state_id
            ),
            (t, shape) => panic!(
                "{}: state shape mismatch — template {t} vs {shape:?}",
                st.state_id
            ),
        }
        assert_eq!(
            gs.render(m.template, &m.fills[..m.n_slots]),
            st.state_sentence,
            "{}: state re-render drifted",
            st.state_id
        );
        n_states += 1;
    }
    (n_opts, n_states)
}

fn load_flappy_render(
    path: &std::path::Path,
    render: fn(&FlappyState, flappy_sim::Action) -> String,
) -> Vec<(MicroStateFixture, FlappyState)> {
    micro_dump::load_micro_states(path, |f| {
        let s: FlappyState = serde_json::from_value(f.state.clone())
            .map_err(|e| format!("{}: state: {e}", f.state_id))?;
        for (i, o) in f.options.iter().enumerate() {
            let expect = render(&s, flappy_sim::ACTIONS[i]);
            if expect != o.sentence {
                return Err(format!(
                    "{}: option {i} sentence drifted\n  fixture:    {:?}\n  recomputed: {:?}",
                    f.state_id, o.sentence, expect
                ));
            }
        }
        let expect = flappy_sim::render_state_sentence(&s);
        if expect != f.state_sentence {
            return Err(format!(
                "{}: state sentence drifted\n  fixture:    {:?}\n  recomputed: {:?}",
                f.state_id, f.state_sentence, expect
            ));
        }
        Ok(s)
    })
}

/// The FROZEN v2 record (`flappy_oracle_laya_en_v2.jsonl`, Bench 880/881)
/// drift-checks against the frozen v2 render.
fn load_flappy(path: &std::path::Path) -> Vec<(MicroStateFixture, FlappyState)> {
    load_flappy_render(path, flappy_sim::render_option_sentence_v2)
}

/// The v3 record (`flappy_oracle_laya_en_v3.jsonl`, Issue 876) drift-checks
/// against the live render.
fn load_flappy_v3(path: &std::path::Path) -> Vec<(MicroStateFixture, FlappyState)> {
    load_flappy_render(path, flappy_sim::render_option_sentence)
}

fn load_lanes(path: &std::path::Path) -> Vec<(MicroStateFixture, LanesState)> {
    micro_dump::load_micro_states(path, |f| {
        let s: LanesState = serde_json::from_value(f.state.clone())
            .map_err(|e| format!("{}: state: {e}", f.state_id))?;
        for (i, o) in f.options.iter().enumerate() {
            let expect = lanes_sim::render_option_sentence(&s, i);
            if expect != o.sentence {
                return Err(format!(
                    "{}: lane {i} sentence drifted\n  fixture:    {:?}\n  recomputed: {:?}",
                    f.state_id, o.sentence, expect
                ));
            }
        }
        Ok(s)
    })
}

/// The shared state-sentence half: decode == semantic forward, v and h
/// recover EXACTLY, re-render identical. Returns (rel fill, exact v, exact
/// h) for the caller's assertions.
fn validate_flappy_state_sentence(
    f: &MicroStateFixture,
    s: &FlappyState,
) -> (u8, i32, i32) {
    let gs = grammar_tables::flappy_state();
    let (rel, v, h) = decode_flappy_state(&gs, &f.state_sentence)
        .unwrap_or_else(|e| panic!("{}: state sentence: {e:?}", f.state_id));
    // The forward mapper returns FILL indices; decode returns (rel
    // fill, exact v, exact h) — convert before comparing.
    let (rel_f, mot_f, gap_f) = flappy_state_forward(s);
    assert_eq!(
        (rel, v, h),
        (rel_f, mot_f - 2, if gap_f == 0 { 2 } else { 3 }),
        "{}: state fills != semantic forward",
        f.state_id
    );
    assert_eq!(v, s.v, "{}: v must recover exactly", f.state_id);
    assert_eq!(h, s.h, "{}: h must recover exactly", f.state_id);
    assert_eq!(
        gs.render(0, &[rel, (v + 2) as u8, if h == 2 { 0 } else { 1 }]),
        f.state_sentence,
        "{}: state re-render drifted",
        f.state_id
    );
    (rel, v, h)
}

/// Decode-layer validation (flappy v2, the frozen record): option fills
/// land on `pos_band`, re-renders identical.
fn validate_flappy(states: &[(MicroStateFixture, FlappyState)]) -> usize {
    let go = grammar_tables::flappy_option();
    let mut n = 0usize;
    for (f, s) in states {
        validate_flappy_state_sentence(f, s);
        for (i, o) in f.options.iter().enumerate() {
            let post = decode_flappy_option(&go, &o.sentence)
                .unwrap_or_else(|e| panic!("{}: option {i}: {e:?}", f.state_id));
            assert_eq!(
                post,
                flappy_option_forward(s, flappy_sim::ACTIONS[i]),
                "{}: option {i} fill != semantic forward",
                f.state_id
            );
            assert_eq!(go.render(0, &[post]), o.sentence, "{}: re-render", f.state_id);
            n += 1;
        }
    }
    n
}

/// Decode-layer validation (flappy v3, Issue 876): (band, offset,
/// post-motion) fills land on the semantic forward, re-renders identical.
fn validate_flappy_v3(states: &[(MicroStateFixture, FlappyState)]) -> usize {
    let go = flappy_option_v3();
    let mut n = 0usize;
    for (f, s) in states {
        validate_flappy_state_sentence(f, s);
        for (i, o) in f.options.iter().enumerate() {
            let post = decode_flappy_option_v3(&go, &o.sentence)
                .unwrap_or_else(|e| panic!("{}: option {i}: {e:?}", f.state_id));
            assert_eq!(
                post,
                flappy_option_forward_v3(s, flappy_sim::ACTIONS[i]),
                "{}: option {i} fill != semantic forward",
                f.state_id
            );
            assert_eq!(go.render(0, &post), o.sentence, "{}: re-render", f.state_id);
            n += 1;
        }
    }
    n
}

/// Decode-layer validation (lanes): the three option sentences decode to
/// the semantic forward, lane slots name their own lane, re-renders match.
fn validate_lanes(states: &[(MicroStateFixture, LanesState)]) -> usize {
    let gl = grammar_tables::lanes_option();
    let mut n = 0usize;
    for (f, s) in states {
        let sents: Vec<&str> = f.options.iter().map(|o| o.sentence.as_str()).collect();
        let dec = decode_lanes_state(&gl, &[sents[0], sents[1], sents[2]])
            .unwrap_or_else(|e| panic!("{}: {e:?}", f.state_id));
        assert_eq!(dec, lanes_forward(s), "{}: decode != forward", f.state_id);
        for (i, o) in f.options.iter().enumerate() {
            assert_eq!(dec.lanes[i].lane, i as u8, "{}: lane slot order", f.state_id);
            let d = &dec.lanes[i];
            let rendered = match d.kind {
                0 => gl.render(0, &[d.lane]),
                k => gl.render(1, &[d.lane, d.dist.expect("blocked has a dist"), k - 1]),
            };
            assert_eq!(rendered, o.sentence, "{}: lane {i} re-render", f.state_id);
            n += 1;
        }
    }
    n
}

// ── main ─────────────────────────────────────────────────────────────────

struct SummaryRow {
    name: &'static str,
    n_states: usize,
    s_in: usize,
    s_loo: usize,
    d_in: usize,
    d_loo: usize,
    flips: usize,
    s_distinct: usize,
    d_distinct: usize,
}

fn main() {
    println!("== Plan 607 T2 — bounded template decode: the losslessness arm ==");
    println!(
        "R6 framing: the sentence is the reference model's input requirement, not \
         the task's — the delta below measures the RENDER, not the decoder."
    );

    verify_all_closed().expect("closed-space proof over the grammar tables");
    println!(
        "closed-space proof: 7/7 tables verify_closed over their full fill \
         products (cap {}) — PASS",
        grammar_tables::CLOSED_SPACE_CAP
    );

    // ── Tetris ───────────────────────────────────────────────────────────────
    println!("\n── tetris (laya-tetris-v2) ──────────────────────────────────");
    let t = load_tetris(&micro_dump::default_fixture("tetris", "v2"));
    let (n_opts, n_states) = validate_tetris(&t);
    println!(
        "decode layer: {n_opts}/{n_opts} option sentences decode · re-render \
         byte-identical · fills == semantic forward; {n_states}/{n_states} state sentences"
    );

    let argmaxes: Vec<usize> = t.states.iter().map(|s| s.argmax).collect();
    let offsets = offsets_of(t.states.iter().map(|s| s.options.len()));
    let counts: Vec<usize> = t.states.iter().map(|s| s.options.len()).collect();
    let (ci, cn, ch) = baselines(&argmaxes, &counts);
    println!(
        "baselines: constant-pick {cn}/{} (index {ci}) · chance {:.1}%",
        n_states,
        100.0 * ch
    );
    let targets: Vec<f64> = t
        .states
        .iter()
        .flat_map(|s| s.options.iter().map(|o| o.p_clean.expect("p_clean")))
        .collect();

    // structured arm — tetris_03's exact corpus (recomputed features,
    // tetris_03's pinned column order)
    let raws: Vec<[f64; 11]> = t
        .features
        .iter()
        .flat_map(|fs| {
            fs.iter().map(|f| {
                [
                    f.lines_cleared as f64,
                    f.holes as f64,
                    f.holes_delta as f64,
                    f.bumpiness as f64,
                    f.max_height as f64,
                    f.aggregate_height as f64,
                    f.landing_height as f64,
                    f.row_transitions as f64,
                    f.col_transitions as f64,
                    f.cumulative_wells as f64,
                    f.eroded_cells as f64,
                ]
            })
        })
        .collect();
    let struct_arm = run_arm(
        standardize::<11, 12>(&raws),
        targets.clone(),
        offsets.clone(),
        &argmaxes,
    );
    assert_eq!(
        struct_arm.in_agree, 36,
        "tetris structured in-corpus drifted from Bench 878 — investigate \
         before trusting the delta"
    );
    assert_eq!(
        struct_arm.loo_agree, 35,
        "tetris structured LOO drifted from Bench 878"
    );
    assert_eq!(
        struct_arm.digest.to_hex().as_str(),
        TETRIS_HEAD_ANCHOR,
        "tetris structured head digest drifted from the Bench 878 anchor"
    );

    // decoded arm — features from the SENTENCES ONLY (the five fill classes)
    let g = grammar_tables::tetris_spot();
    let dec_raws: Vec<[f64; TETRIS_DECODED_F]> = t
        .states
        .iter()
        .flat_map(|st| {
            st.options.iter().map(|fo| {
                let fills = decode_tetris_spot(&g, &fo.sentence).expect("validated above");
                tetris_decoded_features(&fills)
            })
        })
        .collect();
    let dec_arm = run_arm(
        standardize::<TETRIS_DECODED_F, 6>(&dec_raws),
        targets.clone(),
        offsets.clone(),
        &argmaxes,
    );
    let mut summary: Vec<SummaryRow> = Vec::with_capacity(3);
    let mut row = report_arena("tetris", n_states, &struct_arm, &dec_arm, &[
        "encodes: holes CLASS, side band, surface band, height band, clears",
        "drops:   bumpiness, wells, transitions, eroded, exact heights, aggregate",
    ]);
    row.s_distinct = struct_arm.loo_distinct;
    row.d_distinct = dec_arm.loo_distinct;
    summary.push(row);

    // ── Flappy v2 — the frozen record (Bench 880/881) ─────────────────────
    println!("\n── flappy v2 (laya-flappy-v2, frozen record) ────────────");
    let f_states = load_flappy(&micro_dump::default_fixture("flappy", "v2"));
    let n = validate_flappy(&f_states);
    println!("decode layer: {n}/{n} option sentences decode · re-render byte-identical · fills == semantic forward; v/h recover EXACTLY");

    let argmaxes_f: Vec<usize> = f_states.iter().map(|(f, _)| f.argmax).collect();
    let offsets_f = offsets_of(f_states.iter().map(|(f, _)| f.options.len()));
    let counts_f: Vec<usize> = f_states.iter().map(|(f, _)| f.options.len()).collect();
    let (ci, cn, ch) = baselines(&argmaxes_f, &counts_f);
    println!(
        "baselines: constant-pick {cn}/{} (index {ci}) · chance {:.1}%",
        f_states.len(),
        100.0 * ch
    );
    let targets_f: Vec<f64> = f_states
        .iter()
        .flat_map(|(f, _)| f.options.iter().map(|o| o.p_clean.expect("p_clean")))
        .collect();

    let raws_f: Vec<[f64; 8]> = f_states
        .iter()
        .flat_map(|(_, s)| {
            flappy_sim::ACTIONS
                .iter()
                .map(|&a| flappy_sim::feature_row(s, a))
                .collect::<Vec<_>>()
        })
        .collect();
    let struct_arm_f = run_arm(
        standardize::<8, 9>(&raws_f),
        targets_f.clone(),
        offsets_f.clone(),
        &argmaxes_f,
    );
    assert_eq!(
        struct_arm_f.in_agree, 96,
        "flappy structured in-corpus drifted from Bench 880"
    );
    assert_eq!(
        struct_arm_f.loo_agree, 96,
        "flappy structured LOO drifted from Bench 880"
    );
    assert!(
        struct_arm_f
            .digest
            .to_hex()
            .as_str()
            .starts_with(FLAPPY_HEAD_PREFIX),
        "flappy structured head digest drifted from the Bench 880 anchor prefix"
    );

    let go = grammar_tables::flappy_option();
    let gs = grammar_tables::flappy_state();
    let dec_raws_f: Vec<[f64; FLAPPY_DECODED_F]> = f_states
        .iter()
        .flat_map(|(f, _)| {
            let (rel, v, h) =
                decode_flappy_state(&gs, &f.state_sentence).expect("validated above");
            f.options.iter().map(move |o| {
                let post = decode_flappy_option(&go, &o.sentence).expect("validated above");
                flappy_decoded_features(post, rel, v, h)
            })
        })
        .collect();
    let dec_arm_f = run_arm(
        standardize::<FLAPPY_DECODED_F, 5>(&dec_raws_f),
        targets_f.clone(),
        offsets_f.clone(),
        &argmaxes_f,
    );
    let mut row = report_arena("flappy", f_states.len(), &struct_arm_f, &dec_arm_f, &[
        "encodes: post POSITION BAND only (v2 dropped the motion clause — the measured confound)",
        "drops:   exact post_rel, post_v, in_gap, edge_margin; pre-rel is banded; v/h exact",
    ]);
    row.s_distinct = struct_arm_f.loo_distinct;
    row.d_distinct = dec_arm_f.loo_distinct;
    summary.push(row);

    // ── Flappy v3 — Issue 876: the render widening ──────────────────
    println!("\n── flappy v3 (laya-flappy-v3, Issue 876 widening) ────────");
    let f3_states = load_flappy_v3(&micro_dump::default_fixture("flappy", "v3"));
    let n3 = validate_flappy_v3(&f3_states);
    println!("decode layer: {n3}/{n3} option sentences decode · re-render byte-identical · fills == semantic forward; v/h recover EXACTLY");

    // The controlled-comparison premise: the v3 fixture carries the IDENTICAL
    // state set as the v2 record (same seed, same enumerator exclusions) —
    // only the render (and hence the oracle's reads) moved.
    assert_eq!(f3_states.len(), f_states.len(), "v2/v3 corpora must match in size");
    for ((f3, s3), (f2, s2)) in f3_states.iter().zip(&f_states) {
        assert_eq!(f3.state_id, f2.state_id, "v2/v3 state order must match");
        assert_eq!(s3, s2, "{}: v2/v3 seed states must be identical", f3.state_id);
    }
    println!(
        "controlled comparison: {} states identical to the v2 record — only the render moved",
        f3_states.len()
    );

    let argmaxes_3: Vec<usize> = f3_states.iter().map(|(f, _)| f.argmax).collect();
    let offsets_3 = offsets_of(f3_states.iter().map(|(f, _)| f.options.len()));
    let counts_3: Vec<usize> = f3_states.iter().map(|(f, _)| f.options.len()).collect();
    let (ci3, cn3, ch3) = baselines(&argmaxes_3, &counts_3);
    println!(
        "baselines: constant-pick {cn3}/{} (index {ci3}) · chance {:.1}%",
        f3_states.len(),
        100.0 * ch3
    );
    let targets_3: Vec<f64> = f3_states
        .iter()
        .flat_map(|(f, _)| f.options.iter().map(|o| o.p_clean.expect("p_clean")))
        .collect();

    let raws_3: Vec<[f64; 8]> = f3_states
        .iter()
        .flat_map(|(_, s)| {
            flappy_sim::ACTIONS
                .iter()
                .map(|&a| flappy_sim::feature_row(s, a))
                .collect::<Vec<_>>()
        })
        .collect();
    let struct_arm_3 = run_arm(
        standardize::<8, 9>(&raws_3),
        targets_3.clone(),
        offsets_3.clone(),
        &argmaxes_3,
    );
    assert_eq!(
        struct_arm_3.in_agree, 96,
        "flappy v3 structured in-corpus drifted from Bench 882"
    );
    assert_eq!(
        struct_arm_3.loo_agree, 96,
        "flappy v3 structured LOO drifted from Bench 882"
    );
    assert_eq!(
        struct_arm_3.digest.to_hex().as_str(),
        FLAPPY_V3_HEAD_ANCHOR,
        "flappy v3 structured head digest drifted from the Bench 882 anchor"
    );
    assert_eq!(
        struct_arm_3.digest.to_hex().as_str(),
        FLAPPY_V3_HEAD_ANCHOR,
        "flappy v3 structured head digest drifted from the Bench 882 anchor"
    );

    let go3 = flappy_option_v3();
    let gs3 = grammar_tables::flappy_state();
    let dec_raws_3: Vec<[f64; FLAPPY_V3_DECODED_F]> = f3_states
        .iter()
        .flat_map(|(f, _)| {
            let (rel, v, h) =
                decode_flappy_state(&gs3, &f.state_sentence).expect("validated above");
            f.options.iter().map(move |o| {
                let post = decode_flappy_option_v3(&go3, &o.sentence).expect("validated above");
                flappy_v3_decoded_features(post, rel, v, h)
            })
        })
        .collect();
    let dec_arm_3 = run_arm(
        standardize::<FLAPPY_V3_DECODED_F, 9>(&dec_raws_3),
        targets_3.clone(),
        offsets_3.clone(),
        &argmaxes_3,
    );
    assert_eq!(
        dec_arm_3.digest.to_hex().as_str(),
        FLAPPY_V3_DECODED_HEAD_ANCHOR,
        "flappy v3 decoded head digest drifted from the Bench 882 anchor"
    );

    // Issue 876's gate: the decoded arm must beat constant-pick AND keep
    // ≥ 2 distinct picks — the discrimination floor — before any
    // decode-based consumer is considered on flappy.
    assert!(
        dec_arm_3.loo_agree > cn3,
        "flappy v3 decoded arm must beat constant-pick ({cn3}) — got {}",
        dec_arm_3.loo_agree
    );
    assert!(
        dec_arm_3.loo_distinct >= 2,
        "flappy v3 decoded arm discrimination floor: {} distinct picks",
        dec_arm_3.loo_distinct
    );
    let mut row = report_arena("flappy3", f3_states.len(), &struct_arm_3, &dec_arm_3, &[
        "encodes: post_rel in cells (band+offset+h; tails at ±(h+1)), post_v exact, pre_rel (band ±2), pre_v/h",
        "drops:   crash-tail post_rel collapses to ±(h+1); |pre_rel| ≥ 2 collapses to ±2",
    ]);
    row.s_distinct = struct_arm_3.loo_distinct;
    row.d_distinct = dec_arm_3.loo_distinct;
    summary.push(row);

    // ── Lanes ────────────────────────────────────────────────────────────
    println!("\n── lanes (laya-lanes-v1) ────────────────────────────────────");
    let l_states = load_lanes(&micro_dump::default_fixture("lanes", "v1"));
    let n = validate_lanes(&l_states);
    println!("decode layer: {n}/{n} lane sentences decode · re-render byte-identical · decode == semantic forward");

    let argmaxes_l: Vec<usize> = l_states.iter().map(|(f, _)| f.argmax).collect();
    let offsets_l = offsets_of(l_states.iter().map(|(f, _)| f.options.len()));
    let counts_l: Vec<usize> = l_states.iter().map(|(f, _)| f.options.len()).collect();
    let (ci, cn, ch) = baselines(&argmaxes_l, &counts_l);
    println!(
        "baselines: constant-pick {cn}/{} (index {ci}) · chance {:.1}%",
        l_states.len(),
        100.0 * ch
    );
    let targets_l: Vec<f64> = l_states
        .iter()
        .flat_map(|(f, _)| f.options.iter().map(|o| o.p_clean.expect("p_clean")))
        .collect();

    // the lossless anchor: every decoded row must EQUAL the structured row
    let gl = grammar_tables::lanes_option();
    let mut matched = 0usize;
    for (f, s) in &l_states {
        let sents: Vec<&str> = f.options.iter().map(|o| o.sentence.as_str()).collect();
        let dec = decode_lanes_state(&gl, &[sents[0], sents[1], sents[2]])
            .expect("validated above");
        for lane in 0..3 {
            assert_eq!(
                lanes_decoded_features(&dec, lane),
                lanes_sim::feature_row(s, lane),
                "{}: lane {lane} — the render is NOT lossless over the features",
                f.state_id
            );
            matched += 1;
        }
    }
    println!("lossless anchor: {matched}/{matched} decoded rows bit-identical to the structured rows");

    let raws_l: Vec<[f64; 8]> = l_states
        .iter()
        .flat_map(|(_, s)| (0..3).map(|lane| lanes_sim::feature_row(s, lane)).collect::<Vec<_>>())
        .collect();
    let struct_arm_l = run_arm(
        standardize::<8, 9>(&raws_l),
        targets_l.clone(),
        offsets_l.clone(),
        &argmaxes_l,
    );
    assert_eq!(
        struct_arm_l.in_agree, 84,
        "lanes structured in-corpus drifted from Bench 880"
    );
    assert_eq!(
        struct_arm_l.loo_agree, 84,
        "lanes structured LOO drifted from Bench 880"
    );
    assert!(
        struct_arm_l
            .digest
            .to_hex()
            .as_str()
            .starts_with(LANES_HEAD_PREFIX),
        "lanes structured head digest drifted from the Bench 880 anchor prefix"
    );

    let dec_raws_l: Vec<[f64; LANES_DECODED_F]> = l_states
        .iter()
        .flat_map(|(f, _)| {
            let sents: Vec<&str> = f.options.iter().map(|o| o.sentence.as_str()).collect();
            let dec = decode_lanes_state(&gl, &[sents[0], sents[1], sents[2]])
                .expect("validated above");
            (0..3)
                .map(|lane| lanes_decoded_features(&dec, lane))
                .collect::<Vec<_>>()
        })
        .collect();
    let dec_arm_l = run_arm(
        standardize::<LANES_DECODED_F, 9>(&dec_raws_l),
        targets_l.clone(),
        offsets_l.clone(),
        &argmaxes_l,
    );
    assert_eq!(
        dec_arm_l.digest, struct_arm_l.digest,
        "lossless anchor: identical corpora must fit the identical head"
    );
    let mut row = report_arena("lanes", l_states.len(), &struct_arm_l, &dec_arm_l, &[
        "encodes: EVERYTHING the structured features read (blocked/close/far, the three nouns, neighbors, clears) — the render is lossless here",
        "",
    ]);
    row.s_distinct = struct_arm_l.loo_distinct;
    row.d_distinct = dec_arm_l.loo_distinct;
    summary.push(row);

    // ── Summary ──────────────────────────────────────────────────────────
    println!("\n== summary — the agreement delta (decoded − structured) ==");
    println!("  arena   | structured in/LOO | decoded in/LOO | Δin  | Δloo | LOO flips | distinct s/d");
    for r in &summary {
        println!(
            "  {:<7} | {:>2}/{:<3} · {:>2}/{:<3} | {:>2}/{:<3} · {:>2}/{:<3} | {:>3} | {:>4} | {}/{} | {}/{}",
            r.name,
            r.s_in, r.n_states, r.s_loo, r.n_states,
            r.d_in, r.n_states, r.d_loo, r.n_states,
            r.d_in as i64 - r.s_in as i64,
            r.d_loo as i64 - r.s_loo as i64,
            r.flips, r.n_states,
            r.s_distinct, r.d_distinct,
        );
    }
    println!(
        "\nreading: a NEGATIVE delta is information the RENDER never carried \
         (a finding about the grammar, not the decoder); a ~ZERO delta means \
         the sentence arm recovers the structured arm's decisions — the \
         decode is lossless ENOUGH for the decision."
    );
}

/// The per-arena print + the delta row; returns the summary row.
fn report_arena(
    name: &'static str,
    n_states: usize,
    s: &ArmReading,
    d: &ArmReading,
    coverage: &[&str; 2],
) -> SummaryRow {
    println!(
        "structured arm (the published corpus): λ={:<5} | in-corpus {}/{} ({}) | LOO {}/{} ({})",
        s.chosen,
        s.in_agree,
        n_states,
        pct(s.in_agree, n_states),
        s.loo_agree,
        n_states,
        pct(s.loo_agree, n_states),
    );
    println!(
        "decoded arm   (sentence-only input):  λ={:<5} | in-corpus {}/{} ({}) | LOO {}/{} ({})",
        d.chosen,
        d.in_agree,
        n_states,
        pct(d.in_agree, n_states),
        d.loo_agree,
        n_states,
        pct(d.loo_agree, n_states),
    );
    let flips = d
        .loo_picks
        .iter()
        .zip(s.loo_picks.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "DELTA (decoded − structured): in-corpus {:+} | LOO {:+} | LOO flips {}/{}",
        d.in_agree as i64 - s.in_agree as i64,
        d.loo_agree as i64 - s.loo_agree as i64,
        flips,
        n_states,
    );
    println!(
        "discrimination (distinct LOO picks): structured {} · decoded {} [floor ≥ 2] {}",
        s.loo_distinct,
        d.loo_distinct,
        if s.loo_distinct >= 2 && d.loo_distinct >= 2 { "PASS" } else { "FAIL" }
    );
    println!("  render coverage: {}", coverage[0]);
    if !coverage[1].is_empty() {
        println!("                   {}", coverage[1]);
    }
    SummaryRow {
        name,
        n_states,
        s_in: s.in_agree,
        s_loo: s.loo_agree,
        d_in: d.in_agree,
        d_loo: d.loo_agree,
        flips,
        s_distinct: s.loo_distinct,
        d_distinct: d.loo_distinct,
    }
}

// ── Tests (the decode layer over the real corpora + fit determinism) ────

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_core::state_option_scoring::head::FittedHead;

    #[test]
    fn tetris_corpus_decodes_exactly() {
        let t = load_tetris(&micro_dump::default_fixture("tetris", "v2"));
        let (n_opts, n_states) = validate_tetris(&t);
        assert_eq!(n_states, 120, "the committed corpus is 120 states");
        assert_eq!(n_opts, 2660, "the committed corpus is 2660 options");
    }

    #[test]
    fn flappy_corpus_decodes_exactly() {
        let states = load_flappy(&micro_dump::default_fixture("flappy", "v2"));
        assert_eq!(states.len(), 100, "the committed corpus is 100 states");
        assert_eq!(validate_flappy(&states), 200);
    }

    #[test]
    fn flappy_v3_corpus_decodes_exactly() {
        let states = load_flappy_v3(&micro_dump::default_fixture("flappy", "v3"));
        assert_eq!(states.len(), 100, "the committed corpus is 100 states");
        assert_eq!(validate_flappy_v3(&states), 200);
    }

    #[test]
    fn lanes_corpus_decodes_exactly() {
        let states = load_lanes(&micro_dump::default_fixture("lanes", "v1"));
        assert_eq!(states.len(), 100, "the committed corpus is 100 states");
        assert_eq!(validate_lanes(&states), 300);
    }

    #[test]
    fn lanes_decode_is_lossless_over_the_corpus() {
        let gl = grammar_tables::lanes_option();
        let states = load_lanes(&micro_dump::default_fixture("lanes", "v1"));
        for (f, s) in &states {
            let sents: Vec<&str> = f.options.iter().map(|o| o.sentence.as_str()).collect();
            let dec = decode_lanes_state(&gl, &[sents[0], sents[1], sents[2]]).expect("decodes");
            for lane in 0..3 {
                assert_eq!(
                    lanes_decoded_features(&dec, lane),
                    lanes_sim::feature_row(s, lane)
                );
            }
        }
    }

    #[test]
    fn tetris_decoded_head_is_bit_deterministic() {
        let t = load_tetris(&micro_dump::default_fixture("tetris", "v2"));
        let g = grammar_tables::tetris_spot();
        let dec_raws: Vec<[f64; TETRIS_DECODED_F]> = t
            .states
            .iter()
            .flat_map(|st| {
                st.options.iter().map(|fo| {
                    let fills = decode_tetris_spot(&g, &fo.sentence).expect("decodes");
                    tetris_decoded_features(&fills)
                })
            })
            .collect();
        let rows = standardize::<TETRIS_DECODED_F, 6>(&dec_raws);
        let targets: Vec<f64> = t
            .states
            .iter()
            .flat_map(|s| s.options.iter().map(|o| o.p_clean.expect("p_clean")))
            .collect();
        let mut fitter = HeadFitter::<6>::new();
        let head_a = fitter.fit_into(&rows, &targets, 1e-2);
        let head_b = fitter.fit_into(&rows, &targets, 1e-2);
        let digest = |h: &FittedHead<6>| micro_fit::head_digest(h).to_string();
        assert_eq!(digest(&head_a), digest(&head_b));
    }
}
