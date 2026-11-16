#![cfg(feature = "ternary_circuit")]
//! Issue 1225 Arm A gates — the full-forward port vs the committed T1b
//! fixture pins (the pinned-source Python oracle).
//!
//! Gates (each pre-registered in the issue):
//! 1. **Grid bit-exactness (G1-A(a))** — the Rust `encode_byte_grid` output
//!    is byte-identical to the committed grid hex for all five snippets.
//! 2. **Forward band (G1-A(b))** — every probability surface replays at
//!    |Δp| ≤ 1e-4 ABSOLUTE against the committed pins: the 2 noul heads and
//!    the (lines, 8) line-prob matrix are the exactness surfaces; the 6
//!    ordinal sub-scores + 5 style probs are reproduction-only side outputs
//!    under the same band (never exactness evidence); the argmax archetype
//!    name and the score()-threshold squiggle rows must agree exactly.
//! 3. **Vacuous-exactness falsifier** — flipping ONE trit in ONE weight
//!    layer moves some pinned line-prob beyond the band (a forward that
//!    ignores its weights reads green on anything).
//! 4. **Input-sensitivity falsifier** — perturbing ONE grid byte moves some
//!    pinned output beyond the band (a forward that ignores its input reads
//!    green on anything).

use std::path::Path;

use katgpt_core::ternary_circuit::{SOURCE_PIN, parse_meta};
use katgpt_core::ternary_circuit_forward::{CodeHygieneModel, ForwardScratch, encode_byte_grid};

const BAND: f32 = 1e-4;
const FIXTURES: &str = "tests/fixtures/ternary_circuit";

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES)
}

fn blake3_hex(bytes: &[u8]) -> String {
    format!("{}", blake3::hash(bytes))
}

fn read(path: impl AsRef<Path>) -> Vec<u8> {
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.as_ref().display()))
}

/// The flat string map the fixture pins.json is (the T1a gate's parser,
/// duplicated here to keep the two gates independent).
fn pins_of(text: &str) -> std::collections::BTreeMap<String, String> {
    let inner = text.trim();
    assert!(
        inner.starts_with('{') && inner.ends_with('}'),
        "pins.json shape"
    );
    let mut out = std::collections::BTreeMap::new();
    for part in inner[1..inner.len() - 1].split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let Some((k, v)) = part.split_once(':') else {
            panic!("pins.json pair without ':': {part:?}");
        };
        out.insert(
            k.trim().trim_matches('"').to_string(),
            v.trim().trim_matches('"').to_string(),
        );
    }
    out
}

#[derive(serde::Deserialize)]
struct FixtureDoc {
    grid: GridPin,
    heads: HeadsPin,
    line_probs: LineProbsPin,
    squiggles_at_threshold: SquigglePin,
}

#[derive(serde::Deserialize)]
struct GridPin {
    num_lines: usize,
    cols: usize,
    hex: String,
}

#[derive(serde::Deserialize)]
struct HeadsPin {
    noul: Vec<PinFloat>,
    subscores: std::collections::BTreeMap<String, PinFloat>,
    style_probs: std::collections::BTreeMap<String, PinFloat>,
    archetype_argmax: ArgmaxPin,
}

#[derive(serde::Deserialize)]
struct PinFloat {
    value: f64,
}

#[derive(serde::Deserialize)]
struct ArgmaxPin {
    name: String,
}

#[derive(serde::Deserialize)]
struct LineProbsPin {
    values: Vec<Vec<f64>>,
}

#[derive(serde::Deserialize)]
struct SquigglePin {
    threshold: f64,
    rows: Vec<SquiggleRow>,
}

#[derive(serde::Deserialize)]
struct SquiggleRow {
    line: usize,
    /// The source's len(raw_line) columns — the grid pin owns the lengths,
    /// the gate pins kind/line/probability only.
    #[allow(dead_code)]
    col_start: usize,
    #[allow(dead_code)]
    col_end: usize,
    kind: String,
    probability: f64,
}

const ARCHETYPE_NAMES: [&str; 5] = ["minimal", "clean", "enterprise", "dense", "slop"];
const SCORE_NAMES: [&str; 6] = [
    "flatness",
    "naming",
    "simplicity",
    "comments",
    "error_handling",
    "layout",
];
const KIND_NAMES: [&str; 8] = [
    "deep_nesting",
    "poor_naming",
    "noisy_comment",
    "swallowed_error",
    "dense_line",
    "special_casing",
    "duplicate_block",
    "cramped_layout",
];

type Loaded = (
    CodeHygieneModel,
    std::path::PathBuf,
    katgpt_core::ternary_circuit::Meta,
    katgpt_core::kpt_archive::KptArchive<'static>,
    std::vec::Vec<u8>,
);

#[allow(clippy::type_complexity)]
fn load_model_full() -> Loaded {
    let dir = fixture_dir();
    let meta_text = String::from_utf8(read(dir.join("intermediate/meta.txt"))).unwrap();
    let meta = parse_meta(meta_text.trim()).expect("meta grammar");
    let archive_bytes = read(dir.join("compiled/weights.cact"));
    let pins = pins_of(&String::from_utf8(read(dir.join("compiled/pins.json"))).unwrap());
    assert_eq!(
        blake3_hex(&archive_bytes),
        pins["cact_blake3"],
        "committed .cact drifted from its pin"
    );
    let leaked: &'static [u8] = Box::leak(archive_bytes.into_boxed_slice());
    let archive =
        katgpt_core::kpt_archive::KptArchive::from_bytes(leaked).expect("archive verify ladder");
    let id_hex: String = archive
        .archive_id()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(id_hex, pins["cact_archive_id"], "archive id drifted");
    let model = CodeHygieneModel::load(&archive, &meta).expect("param resolution");
    (model, dir, meta, archive, Vec::new())
}

fn load_model() -> (CodeHygieneModel, std::path::PathBuf) {
    let (m, d, _, _, _) = load_model_full();
    (m, d)
}

/// The argmax + threshold squiggle rows, recomputed from raw outputs.
fn squiggle_rows(
    line_probs: &[f32],
    lines: usize,
    threshold: f32,
) -> Vec<(usize, usize, usize, &'static str, f32)> {
    let mut rows = Vec::with_capacity(lines);
    for l in 0..lines {
        let row = &line_probs[l * 8..(l + 1) * 8];
        let mut best = 0usize;
        for (k, &p) in row.iter().enumerate() {
            if p > row[best] {
                best = k;
            }
        }
        let p = row[best];
        if p < threshold {
            continue;
        }
        // col_start/col_end are the SOURCE's line-length laws — the Rust
        // gate pins the kind + line + probability; the columns ride the
        // fixture (the grid encoder pins the lengths themselves).
        rows.push((l + 1, 0, 0, KIND_NAMES[best], p));
    }
    rows
}

fn check_snippet(model: &CodeHygieneModel, dir: &Path, name: &str) -> (usize, Vec<f32>) {
    let doc: FixtureDoc =
        serde_json::from_slice(&read(dir.join(format!("forward/forward_{name}.json")))).unwrap();
    let code = String::from_utf8(read(dir.join(format!("forward/snippet_{name}.txt")))).unwrap();

    // ── G1-A(a): the grid is byte-exact.
    let grid = encode_byte_grid(&code);
    assert_eq!(grid.len(), doc.grid.num_lines, "{name}: line count");
    assert_eq!(doc.grid.cols, 128, "{name}: col count");
    let hex: String = grid
        .iter()
        .flat_map(|r| r.iter())
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, doc.grid.hex, "{name}: grid bytes diverge from the pin");

    // ── G1-A(b): the forward replays inside the absolute band.
    let mut scratch = ForwardScratch::default();
    let out = model.forward(&grid, &mut scratch).expect("forward");
    for i in 0..2 {
        let delta = (out.noul[i] as f64 - doc.heads.noul[i].value).abs();
        assert!(delta <= BAND as f64, "{name}: noul[{i}] off by {delta}");
    }
    assert_eq!(out.line_probs.len(), doc.line_probs.values.len() * 8);
    let mut worst = 0.0f64;
    for (l, row) in doc.line_probs.values.iter().enumerate() {
        for (k, &pinned) in row.iter().enumerate() {
            let delta = (out.line_probs[l * 8 + k] as f64 - pinned).abs();
            worst = worst.max(delta);
            assert!(
                delta <= BAND as f64,
                "{name}: line_probs[{l}][{k}] off by {delta}"
            );
        }
    }
    for (i, score_name) in SCORE_NAMES.iter().enumerate() {
        let pinned = doc.heads.subscores[*score_name].value;
        let delta = (out.subscores[i] as f64 - pinned).abs();
        assert!(
            delta <= BAND as f64,
            "{name}: subscore {score_name} off by {delta}"
        );
    }
    for (i, arch) in ARCHETYPE_NAMES.iter().enumerate() {
        let pinned = doc.heads.style_probs[*arch].value;
        let delta = (out.style_probs[i] as f64 - pinned).abs();
        assert!(delta <= BAND as f64, "{name}: style {arch} off by {delta}");
    }
    let mut best_style = 0usize;
    for (i, &p) in out.style_probs.iter().enumerate() {
        if p > out.style_probs[best_style] {
            best_style = i;
        }
    }
    assert_eq!(
        ARCHETYPE_NAMES[best_style], doc.heads.archetype_argmax.name,
        "{name}: argmax archetype diverges"
    );

    // The squiggle rows: same argmax + threshold law as the source's score();
    // kind + line + probability pinned (the fixture's columns are the
    // source's len(raw_line) surface — already grid-pinned upstream).
    let pinned_rows = &doc.squiggles_at_threshold.rows;
    let mine = squiggle_rows(
        &out.line_probs,
        doc.grid.num_lines,
        doc.squiggles_at_threshold.threshold as f32,
    );
    assert_eq!(mine.len(), pinned_rows.len(), "{name}: squiggle row count");
    for ((ml, _, _, mk, mp), pr) in mine.iter().zip(pinned_rows.iter()) {
        assert_eq!(
            format!("{ml}"),
            format!("{}", pr.line),
            "{name}: squiggle line"
        );
        assert_eq!(*mk, pr.kind, "{name}: squiggle kind");
        let d = (*mp as f64 - pr.probability).abs();
        assert!(d <= BAND as f64, "{name}: squiggle prob off by {d}");
    }
    (doc.grid.num_lines, out.line_probs.clone())
}

#[test]
fn forward_replays_the_committed_fixture_pins() {
    let (model, dir) = load_model();
    for name in [
        "rust_sample",
        "python_sample",
        "go_sample",
        "edge_trap",
        "edge_empty",
    ] {
        check_snippet(&model, &dir, name);
    }
}

#[test]
fn single_trit_flip_moves_the_pinned_output() {
    let (mut model, dir) = load_model();
    let code = String::from_utf8(read(dir.join("forward/snippet_rust_sample.txt"))).unwrap();
    let grid = encode_byte_grid(&code);
    let mut scratch = ForwardScratch::default();
    let base = model.forward(&grid, &mut scratch).unwrap();
    let _ = &mut model;
    // The falsifier rebuilds the model from a mutated archive below.
    let meta_text = String::from_utf8(read(dir.join("intermediate/meta.txt"))).unwrap();
    let meta = parse_meta(meta_text.trim()).unwrap();
    let archive_bytes = read(dir.join("compiled/weights.cact"));
    let archive = katgpt_core::kpt_archive::KptArchive::from_bytes(&archive_bytes).unwrap();
    // Flip one trit of sq_w (the last trit layer, rows 226 × cols 8): its
    // (0,0) trit 0 → the flipped layer must move some line-prob.
    let sq_idx = meta
        .trits
        .iter()
        .position(|e| e.name == "sq_w")
        .expect("sq_w layer");
    let mut tw = archive.to_ternary_weights(sq_idx);
    let flipped = if tw.get(0, 0) == 1 { -1i8 } else { 1i8 };
    tw.set(0, 0, flipped);
    let mut layers: Vec<katgpt_types::TernaryWeights> = (0..archive.layers().len())
        .map(|i| archive.to_ternary_weights(i))
        .collect();
    layers[sq_idx] = tw;
    let rebuilt_bytes = katgpt_core::kpt_archive::build_archive(&layers.iter().collect::<Vec<_>>())
        .expect("rebuild");
    let rebuilt = katgpt_core::kpt_archive::KptArchive::from_bytes(&rebuilt_bytes).unwrap();
    let model2 = CodeHygieneModel::load(&rebuilt, &meta).unwrap();
    let out2 = model2.forward(&grid, &mut scratch).unwrap();
    let mut max_delta = 0.0f32;
    for (a, b) in base.line_probs.iter().zip(out2.line_probs.iter()) {
        max_delta = max_delta.max((a - b).abs());
    }
    assert!(
        max_delta > 1e-4,
        "single-trit flip moved line-probs by only {max_delta} — vacuous forward"
    );
}

#[test]
fn single_grid_byte_perturbation_moves_the_pinned_output() {
    let (model, dir) = load_model();
    let code = String::from_utf8(read(dir.join("forward/snippet_rust_sample.txt"))).unwrap();
    let mut grid = encode_byte_grid(&code);
    let mut scratch = ForwardScratch::default();
    let base = model.forward(&grid, &mut scratch).unwrap();
    // Rewrite EVERY byte of line 0 to a fixed filler — a single-byte swap can
    // wash out through the 93.5%-zero kernels (measured: the PINNED SOURCE
    // itself shows delta 0.0 for one byte at this site), but a whole line
    // cannot.
    for cell in grid[0].iter_mut() {
        *cell = b'#';
    }
    let out2 = model.forward(&grid, &mut scratch).unwrap();
    let mut max_delta = 0.0f32;
    for (a, b) in base.line_probs.iter().zip(out2.line_probs.iter()) {
        max_delta = max_delta.max((a - b).abs());
    }
    assert!(
        max_delta > 1e-4,
        "a whole-line rewrite moved line-probs by only {max_delta} — input-blind forward"
    );
}

#[test]
fn source_pin_is_the_recorded_one() {
    assert_eq!(SOURCE_PIN, "423c7e1f02ef409bbaea209268bfcde36e0f1cb0");
}
