#![cfg(feature = "ternary_circuit")]
//! Issue 1225 Arm A G2 — the warm per-file latency bar (≤ 10 ms/file).
//!
//! Pre-registration (the issue row): "G2-A: warm ≤ 10 ms/file on a QUIET m3,
//! `scripts/bench_preflight.sh` PROVENANCE line quoted beside the number".
//! m3 takes no jobs from this lane — the sanctioned substitute (recorded in
//! the issue's claim history) is a QUIET bmo/amo window with the box delta
//! disclosed. A number measured on a busy box is evidence of nothing; the
//! bar is the pre-registered ceiling, not a tuning target.
//!
//! Runbook (the ONLY valid posture for the bar — quiet box, AC, load well
//! under half the cores):
//!
//! ```sh
//! ../riir-reflex/scripts/bench_preflight.sh   # quote its PROVENANCE line
//! CARGO_TARGET_DIR=/tmp/i1225g2a \
//!   cargo test --release -p katgpt-core --features ternary_circuit \
//!   --test ternary_circuit_forward_g2 -- --ignored --nocapture
//! ```
//!
//! Surfaces: [`forward_into`](katgpt_core::ternary_circuit_forward::CodeHygieneModel::forward_into)
//! is THE bar surface (the steady zero-alloc serving path, G4-A); the
//! `forward()` convenience form is timed and PRINTED as a disclosure row
//! only (one small `Vec` per call), never barred.
//!
//! Loud-zero defences (the timed-region law): every timed call's output
//! feeds a `black_box`ed accumulator (a body whose work the compiler deletes
//! cannot satisfy the bar — the vanished-numerator class reads a plausible
//! number); population floors (5 files, warmup + window minimums) so an
//! empty harness cannot read green; and the wiring selfcheck below pins the
//! harness to bit-identical two-pass determinism so the timing loop cannot
//! drift into a different computation than the one G1-A pinned.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use katgpt_core::ternary_circuit::parse_meta;
use katgpt_core::ternary_circuit_forward::{
    CodeHygieneModel, ForwardOutput, ForwardScratch, MAX_COLS, NOUL_HEADS, NUM_ARCHETYPES,
    NUM_SCORES, encode_byte_grid,
};

/// The five committed fixture snippets — a "file" here is one snippet's
/// byte grid through one warm steady-state forward.
const SNIPPETS: [&str; 5] = [
    "rust_sample",
    "python_sample",
    "go_sample",
    "edge_trap",
    "edge_empty",
];

/// The pre-registered bar (issue 1225 G2-A): warm ≤ 10 ms/file.
const BAR_NS_PER_FILE: f64 = 10_000_000.0;

const WARMUP_PASSES: usize = 8;
const WINDOW_PASSES: usize = 24;
const WINDOWS: usize = 3;

const FIXTURES: &str = "tests/fixtures/ternary_circuit";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURES)
}

fn read(path: impl AsRef<Path>) -> std::vec::Vec<u8> {
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.as_ref().display()))
}

/// The one pins.json key the harness verifies (the .cact pin) — parsed
/// locally so this bench stays independent of the G1 gate's loader.
fn pins_key(text: &str, key: &str) -> String {
    let inner = text.trim();
    assert!(
        inner.starts_with('{') && inner.ends_with('}'),
        "pins.json shape"
    );
    for part in inner[1..inner.len() - 1].split(',') {
        let part = part.trim();
        if let Some((k, v)) = part.split_once(':')
            && k.trim().trim_matches('"') == key
        {
            return v.trim().trim_matches('"').to_string();
        }
    }
    panic!("pins.json has no {key}");
}

type Loaded = (
    CodeHygieneModel,
    katgpt_core::kpt_archive::KptArchive<'static>,
);

fn load_model() -> Loaded {
    let dir = fixture_dir();
    let meta_text = String::from_utf8(read(dir.join("intermediate/meta.txt"))).unwrap();
    let meta = parse_meta(meta_text.trim()).expect("meta grammar");
    let archive_bytes = read(dir.join("compiled/weights.cact"));
    let pins = String::from_utf8(read(dir.join("compiled/pins.json"))).unwrap();
    let want = pins_key(&pins, "cact_blake3");
    let got = format!("{}", blake3::hash(&archive_bytes));
    assert_eq!(got, want, "committed .cact drifted from its pin");
    let leaked: &'static [u8] = Box::leak(archive_bytes.into_boxed_slice());
    let archive =
        katgpt_core::kpt_archive::KptArchive::from_bytes(leaked).expect("archive verify ladder");
    let model = CodeHygieneModel::load(&archive, &meta).expect("param resolution");
    (model, archive)
}

fn load_grids() -> std::vec::Vec<(&'static str, std::vec::Vec<[u8; MAX_COLS]>)> {
    let dir = fixture_dir();
    SNIPPETS
        .iter()
        .map(|&name| {
            let code =
                String::from_utf8(read(dir.join(format!("forward/snippet_{name}.txt")))).unwrap();
            let grid = encode_byte_grid(&code);
            assert!(!grid.is_empty(), "{name}: empty grid (population floor)");
            (name, grid)
        })
        .collect()
}

/// One `black_box`ed tap per call — the accumulator the timed windows fold
/// into, so a deleted forward body cannot satisfy (or poison) the bar.
#[inline]
fn tap(out: &ForwardOutput, acc: f64) -> f64 {
    let t = out.noul[0] as f64 + out.line_probs[out.line_probs.len() - 1] as f64;
    black_box(acc + t)
}

fn fresh_output() -> ForwardOutput {
    ForwardOutput {
        line_probs: std::vec::Vec::new(),
        noul: [0.0; NOUL_HEADS],
        subscores: [0.0; NUM_SCORES],
        style_probs: [0.0; NUM_ARCHETYPES],
    }
}

#[test]
#[ignore = "G2-A latency bar — QUIET-box runbook in the module docs (bench_preflight PROVENANCE required; a busy-box number is evidence of nothing)"]
fn g2_warm_per_file_latency_bar() {
    let (model, _archive) = load_model();
    let grids = load_grids();
    assert_eq!(grids.len(), SNIPPETS.len(), "population floor: all 5 files");

    // Global warmup across ALL snippets first — `forward_into`'s scratch and
    // output capacities settle at the max line count after the first calls
    // (the G4-A steady state), so no measured window pays a first-touch cost.
    let mut scratch = ForwardScratch::default();
    let mut out = fresh_output();
    let mut acc = 0.0f64;
    for _ in 0..WARMUP_PASSES {
        for (_, grid) in &grids {
            model
                .forward_into(grid, &mut scratch, &mut out)
                .expect("warmup forward");
            assert!(
                out.noul.iter().all(|v| v.is_finite()),
                "warmup produced a non-finite head"
            );
            acc = tap(&out, acc);
        }
    }
    black_box(acc);

    // Per-snippet cells: best-of-WINDOWS windows of WINDOW_PASSES calls each;
    // the MIN window is the scheduler-shed estimator (box state is part of
    // every latency claim — a min over windows is the honest read for a bar).
    let mut worst = f64::MIN;
    let mut rows: std::vec::Vec<String> = std::vec::Vec::new();
    for (name, grid) in &grids {
        let files = 1usize;
        let mut best = f64::MAX;
        let mut cell_acc = 0.0f64;
        for _ in 0..WINDOWS {
            let t0 = Instant::now();
            for _ in 0..WINDOW_PASSES {
                model
                    .forward_into(grid, &mut scratch, &mut out)
                    .expect("timed forward");
                cell_acc = tap(&out, cell_acc);
            }
            let ns = t0.elapsed().as_nanos() as f64 / (WINDOW_PASSES * files) as f64;
            if ns < best {
                best = ns;
            }
        }
        black_box(cell_acc);
        assert!(
            best <= BAR_NS_PER_FILE,
            "{name}: {best:.0} ns/file exceeds the {BAR_NS_PER_FILE:.0} ns bar"
        );
        worst = worst.max(best);
        rows.push(format!(
            "G2-A {name}: {best:.0} ns/file ({:.3} ms/file) — lines {}",
            best / 1e6,
            grid.len()
        ));
    }

    // Disclosure row only (never barred): the convenience `forward()` form
    // allocates one small `Vec` per call — the steady path above is the
    // serving surface the bar pre-registered.
    let (_, grid) = grids.first().expect("non-empty");
    let t0 = Instant::now();
    let mut conv_acc = 0.0f64;
    for _ in 0..WINDOW_PASSES {
        let o = model.forward(grid, &mut scratch).expect("convenience form");
        conv_acc = tap(&o, conv_acc);
    }
    let conv_ns = t0.elapsed().as_nanos() as f64 / WINDOW_PASSES as f64;
    black_box(conv_acc);

    for r in &rows {
        println!("{r}");
    }
    println!(
        "G2-A worst-per-file: {worst:.0} ns/file ({:.3} ms/file) — bar ≤ 10.000 ms/file PASS",
        worst / 1e6
    );
    println!(
        "G2-A disclosure forward() convenience form: {conv_ns:.0} ns/file (one Vec alloc/call; not barred)"
    );
    println!(
        "G2-A PROVENANCE: quote `../riir-reflex/scripts/bench_preflight.sh` output beside these numbers — a busy-box run is evidence of nothing (issue 1225 posture: QUIET m3, bmo/amo quiet-window substitute disclosed)"
    );
}

/// The in-sweep half (NOT ignored): the harness wiring is load-independent —
/// one forward per snippet, outputs finite, and two passes bit-identical
/// (determinism the timing loop above leans on). No timing assertions here:
/// a ns-class wall-clock bar is unadjudicable in debug or on a busy box
/// (the Issue 1227 profile split law).
#[test]
fn g2_harness_wiring_selfcheck() {
    let (model, _archive) = load_model();
    let grids = load_grids();
    assert_eq!(grids.len(), SNIPPETS.len(), "population floor: all 5 files");

    let mut scratch = ForwardScratch::default();
    let mut out = fresh_output();
    for (name, grid) in &grids {
        model
            .forward_into(grid, &mut scratch, &mut out)
            .unwrap_or_else(|e| panic!("{name}: forward failed: {e:?}"));
        assert!(
            out.noul.iter().all(|v| v.is_finite())
                && out.subscores.iter().all(|v| v.is_finite())
                && out.style_probs.iter().all(|v| v.is_finite())
                && out.line_probs.iter().all(|v| v.is_finite()),
            "{name}: non-finite output surface"
        );
        let first: std::vec::Vec<u32> = out
            .line_probs
            .iter()
            .map(|v| v.to_bits())
            .chain(out.noul.iter().map(|v| v.to_bits()))
            .collect();
        let first_style: [u32; 5] = out.style_probs.map(|v| v.to_bits());
        model
            .forward_into(grid, &mut scratch, &mut out)
            .unwrap_or_else(|e| panic!("{name}: second forward failed: {e:?}"));
        let second: std::vec::Vec<u32> = out
            .line_probs
            .iter()
            .map(|v| v.to_bits())
            .chain(out.noul.iter().map(|v| v.to_bits()))
            .collect();
        assert_eq!(first, second, "{name}: forward is not bit-deterministic");
        assert_eq!(
            first_style,
            out.style_probs.map(|v| v.to_bits()),
            "{name}: style surface not bit-deterministic"
        );
    }
}
