#![cfg(feature = "grafts")]
//! bench_988_grafts_goat — issue 988 T4, the `grafts` GOAT bench.
//!
//! Posture: the FROZEN-READ REPLAY. The T2b real-activation cell
//! (riir-infer issue 1206 / Bench 971, `e46a310`) exported its fitted
//! artifact + frozen decision rows through the `grafts_export` bin; this
//! bench mounts that artifact and replays the sealed cell's decision
//! surface end-to-end — no model, no GPU, deterministic. The sealed
//! ACCURACY verdict stays anchored at Bench 971 (this repo carries no
//! model); the bench re-asserts the machinery that produced it plus the
//! mechanical GOAT legs:
//!
//! - **G1** — fixture digest pins, artifact decode + the `GraftSlot`
//!   mount laws, per-row reconstruction: activations rebuilt from the
//!   artifact's OWN read/switch weights to hit each frozen row's
//!   recorded triple + arming bit (a per-slot 2-functional Gram solve;
//!   margins absorb f32 error), verified through the REAL read path
//!   (`value_at` / `armed_at`), then `apply` over the corpus: every
//!   fired solved row must decide its true answer (the mined table is
//!   conflict-free by construction), determinism (double-apply byte
//!   identity), the scramble arm (deterministic; wiring-wrong), and the
//!   sealed bar arithmetic (graft 0.3925 ≥ baseline 0.1250 + 20pp).
//! - **G2** — hot-path latency at the T2b shape (3 reads + 1 switch ×
//!   9 segments × 1024 dims = 18,432 sequential f32 MACs/apply): the
//!   pre-registered bar is **≤ 30 µs/apply** (min-window mean of 3 ×
//!   1000 applies) — ~2.3–5× the quiet-scalar expectation (6–13 µs)
//!   and ≤ 0.2% of a ~15 ms/token decode forward: the graft must stay
//!   negligible on the sampling seam. Box state is part of any quoted
//!   number (`bench_preflight.sh` provenance; a loaded-box run prints a
//!   provisional number).
//! - **G3(a)** — switches DISARMED: byte-identical outputs (exact — the
//!   writer never touched), the unarmed no-op law.
//! - **G3(b)** — the 998 T2 no-harm legs (pre-registration in
//!   `.issues/998`; posture-3 re-registration verdict AGREE 2026-11-12,
//!   reviewer session 40a812a8, committed BEFORE the fresh capture
//!   `ebe59f611`): the detector switch's GATED legs are (1) the task
//!   contract — disarm rate 0 on the dev-200 held-out task rows (the
//!   sealed-400 capture-only arm check runs cell-side, never in this
//!   bench); (2) the FRESH off-task TEST corpus (seed OFFTASK_SEED_TEST_P3
//!   0x998_0FF3, captured after the posture-3 pre-registration commit —
//!   the ruling's contamination ban) — gate = Clopper–Pearson 95% UB ≤
//!   0.5% ⇔ **0 false arms at n = 600**; (3) the hard-negative sub-leg
//!   (digits-non-arithmetic + arithmetic-words, n = 200, same gate,
//!   thinner power disclosed: 0/200 certifies ≤ 1.5%). The threshold is
//!   the CALIBRATION split's max raw ridge score × M = 2.0 (frozen a
//!   priori; expected exceedance 0.0357/600 from the pre-registered GPD
//!   tail fit — the posture-2 flaw, a gate demanding 0 at an expected-3
//!   placement, fixed). The task-corpus base-right damage is REPORTED
//!   (the accepted trade: +26.75pp net lane accuracy), never gated. KL
//!   is **Zipf-surface** KL(top-40) — the documented Zipf logit surface
//!   (−1.07·ln(i+8)) is the synthetic stand-in for the host LM head,
//!   never a real-model no-harm proof.
//! - **G4** — zero allocation on the armed hot path (counting
//!   allocator, delta over 1000 applies == 0; the `graft_g4_alloc`
//!   fixture is the primary evidence, this is the re-assert).
//!
//! **History (disclosed):** the ORIGINAL 988-T4 posture fitted the
//! switch on the {base-wrong} label; it armed on EVERY row (T1 measured
//! the mechanism: in-sample memorization AUC 1.0000, held-out 0.5966 —
//! `base_wrong` is not state-readable; see `.issues/998`). That
//! pre-registration's G3(b) surface (task-corpus base-right) was
//! unsatisfiable for every lane — T1's measurement is what forced the
//! 998 T2 re-scope to the off-task surface this bench now gates.
//!
//! Provenance chain: the fixture files committed beside this bench
//! (`data/grafts_t2b_{artifact,rows}.bin` + the manifest JSON) were
//! produced by riir-infer `grafts_export` from the slot-era sample
//! cache; the fit reproduces the sealed cell's (solved triple coverage
//! 316/400 = 0.79 == the sealed report's `solved_triple_coverage`;
//! checkpoint + payload digests live in the manifest).
//!
//! Run: `cargo bench -p katgpt-core --features grafts --bench
//! bench_988_grafts_goat` (release by construction).

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use katgpt_core::graft::{ActivationReader, Graft, LogitWriter};
use katgpt_core::{GraftArtifact, GraftInstall, GraftSlot};

// ── Pins (fixture digests + the sealed T2b record) ──────────────────────

const ARTIFACT_BYTES: &[u8] = include_bytes!("data/grafts_t2b_artifact.bin");
const ROWS_BYTES: &[u8] = include_bytes!("data/grafts_t2b_rows.bin");

const ARTIFACT_BLAKE3: &str = "2c026fdf86226585c97445f533530cccb02cb41e313232f580d3ff2f8a277dff";
const ROWS_BLAKE3: &str = "de4dee347ad2434f94e5d5ca0d4725d19e2b92ef485f54f031504d219556afb2";
const GRAFT_DIGEST_HEX: &str = "53638776826ee472c672a23592a4000389305ec8b8e643e3b98fb1b7db0cdf53";
const SEALED_PAYLOAD_BLAKE3: &str =
    "fac21420025b309ec6b2374080fb6ece50f8a304cd74a8edf0428a3438e0a747";

/// The sealed T2b verdict rows (Bench 971) + the pre-registered G1 bar.
const SEALED_BASELINE_ACC: f64 = 0.125;
const SEALED_GRAFT_ACC: f64 = 0.3925;
const SEALED_ACC_BAR: f64 = 0.325; // baseline + 20pp

/// G2: the pre-registered hot-path bar (min-window mean µs/apply).
const G2_BAR_US: f64 = 30.0;
const G2_WINDOWS: usize = 3;
const G2_APPLIES: usize = 1000;

/// G3(b): the pre-registered false-arm target (998 T2). The GATE is the
/// Clopper–Pearson 95% upper bound ≤ this bar ⇔ 0 false arms at n = 600
/// (1/600 ≈ 0.79% would FAIL); the bar prints on every off-task verdict
/// line. KL against the documented Zipf logit surface is REPORTED, never
/// a real-model no-harm proof.
const G3_FLIP_BAR: f64 = 0.005;
const WRITE_BIAS: f32 = 8.0;
const KL_TOPK: usize = 40;

/// The T2b cell's pinned scramble seed.
const SCRAMBLE_SEED: u64 = 0x5EED_9880;

/// Rows v2 (998 T2): +class byte — strict version match, v1 refused loud.
const ROWS_VERSION: u32 = 2;
const ROW_BYTES: usize = 15;
/// Split ids (998 T2 leg map, posture 3): 3 = the FRESH P3 test corpus
/// (the only off-task gate surface), 4 = the CALIBRATION segment
/// (the threshold's only source — placement data, reported, never
/// gated).
const SPLIT_SOLVED: u8 = 0;
const SPLIT_DEV: u8 = 1;
const SPLIT_OFFTASK_FIT: u8 = 2;
const SPLIT_OFFTASK_TEST: u8 = 3;
const SPLIT_OFFTASK_CALIBRATION: u8 = 4;
/// Pinned leg sizes — the parse asserts them so a truncated corpus
/// cannot pass a power-disclosed gate by shrinking silently.
const LEG_SOLVED_N: usize = 400;
const LEG_DEV_N: usize = 200;
const LEG_OFFTASK_FIT_N: usize = 400;
const LEG_OFFTASK_TEST_N: usize = 600;
const LEG_CALIBRATION_N: usize = 600;
const LEG_HARD_SUB_N: usize = 200;

// ── G4: counting allocator ───────────────────────────────────────────────

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc_zeroed(l) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

// ── Fixture containers ──────────────────────────────────────────────────

struct FrozenRow {
    instance: u32,
    triple: [u16; 3],
    answer: u8,
    base_wrong: bool,
    armed: bool,
    /// 0 task / 1 off-task easy / 2 off-task hard-A / 3 off-task hard-B.
    class: u8,
    /// 0 solved (task) / 1 dev (task) / 2 off-task fit / 3 off-task
    /// held-out.
    split: u8,
}

fn parse_rows(bytes: &[u8]) -> Vec<FrozenRow> {
    assert!(bytes.len() >= 12, "rows fixture truncated (header)");
    assert_eq!(&bytes[0..4], b"GRFW".as_slice(), "rows fixture magic");
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(
        version, ROWS_VERSION,
        "rows fixture version (strict: v1 refused loud)"
    );
    let n = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    assert_eq!(
        bytes.len(),
        12 + n * ROW_BYTES,
        "rows fixture length vs count"
    );
    assert!(n >= 400, "rows fixture implausibly small");
    (0..n)
        .map(|i| {
            let o = 12 + i * ROW_BYTES;
            let le16 = |k: usize| u16::from_le_bytes([bytes[o + k], bytes[o + k + 1]]);
            FrozenRow {
                instance: u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]),
                triple: [le16(4), le16(6), le16(8)],
                answer: bytes[o + 10],
                base_wrong: bytes[o + 11] != 0,
                armed: bytes[o + 12] != 0,
                class: bytes[o + 13],
                split: bytes[o + 14],
            }
        })
        .collect()
}

fn parse_artifact(bytes: &[u8]) -> GraftArtifact {
    assert!(bytes.len() >= 52, "artifact fixture truncated (header)");
    assert_eq!(&bytes[0..4], b"GRFZ".as_slice(), "artifact fixture magic");
    let format_version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&bytes[8..40]);
    let epoch = u64::from_le_bytes([
        bytes[40], bytes[41], bytes[42], bytes[43], bytes[44], bytes[45], bytes[46], bytes[47],
    ]);
    let canonical_len = u32::from_le_bytes([bytes[48], bytes[49], bytes[50], bytes[51]]) as usize;
    assert_eq!(
        bytes.len(),
        52 + canonical_len,
        "artifact fixture length vs canonical_len"
    );
    GraftArtifact {
        format_version,
        epoch,
        graft_digest: digest,
        canonical: bytes[52..].to_vec(),
    }
}

// ── The replay reader (the host seam's frozen-read stand-in) ────────────

/// Resolves the cell's read ids (`slot * 100 + layer`) against a frozen
/// slot-major, layer-major activation buffer — the same address space
/// riir-infer's `RowReader` serves. An id the buffer does not hold
/// yields the empty slice (the loud length-assert downstream), exactly
/// the `RowReader` contract.
struct ReplayReader<'a, 'b> {
    n_embd: usize,
    layers: &'a [u32],
    slots: Vec<&'b [f32]>,
}

impl ActivationReader for ReplayReader<'_, '_> {
    fn read(&self, id: u32) -> &[f32] {
        let s = (id / 100) as usize;
        let l = id % 100;
        match self.layers.iter().position(|&x| x == l) {
            Some(k) => {
                // Each slot slice is already the slot's own k×n_embd block
                // (sliced out of the flat buffer); the layer offset is
                // WITHIN the slice.
                let off = k * self.n_embd;
                &self.slots[s][off..off + self.n_embd]
            }
            None => &[],
        }
    }
}

// ── Writers ──────────────────────────────────────────────────────────────

struct VecWriter {
    biases: Vec<(u32, f32)>,
}

impl LogitWriter for VecWriter {
    fn add_bias(&mut self, token: u32, bias: f32) {
        self.biases.push((token, bias));
    }
}

struct NullWriter {
    n: usize,
}

impl LogitWriter for NullWriter {
    fn add_bias(&mut self, _token: u32, _bias: f32) {
        self.n += 1;
    }
}

// ── Per-slot reconstruction solve ────────────────────────────────────────

/// Per slot: the read functional `a` (that read's segments concatenated,
/// layer-major), the switch functional `b` (the switch's slot-s segments
/// concatenated), and the 2×2 Gram inverse (f64; jitter fallback on a
/// degenerate functional pair).
struct SlotSolve {
    a: Vec<f32>,
    b: Vec<f32>,
    iaa: f64,
    iab: f64,
    ibb: f64,
}

fn build_solves(graft: &Graft) -> Vec<SlotSolve> {
    let slots = graft.reads.len();
    assert_eq!(
        graft.switches.len(),
        1,
        "the frozen-read reconstruction expects exactly one fitted switch"
    );
    (0..slots)
        .map(|s| {
            let mut a = Vec::new();
            for seg in &graft.reads[s].segments {
                assert_eq!(
                    seg.layer / 100,
                    s as u32,
                    "read segment names a foreign slot"
                );
                a.extend_from_slice(&seg.weights);
            }
            let mut b = Vec::new();
            for seg in &graft.switches[0].segments {
                if seg.layer / 100 == s as u32 {
                    b.extend_from_slice(&seg.weights);
                }
            }
            assert!(!b.is_empty(), "switch has no segments for slot {s}");
            assert_eq!(a.len(), b.len(), "slot functionals disagree in width");
            let aa: f64 = a.iter().map(|&w| f64::from(w) * f64::from(w)).sum();
            let ab: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum();
            let bb: f64 = b.iter().map(|&w| f64::from(w) * f64::from(w)).sum();
            let (mut aa_j, mut bb_j) = (aa, bb);
            if (aa * bb - ab * ab).abs() < 1e-9 {
                // Collinear read/switch functionals: ridge-jitter the Gram
                // (the solution shifts marginally; the verification below
                // is the authority, so a bad jitter fails loud there).
                let j = 1e-6 * (aa + bb);
                aa_j = aa + j;
                bb_j = bb + j;
            }
            let det = aa_j * bb_j - ab * ab;
            SlotSolve {
                a,
                b,
                iaa: bb_j / det,
                iab: -ab / det,
                ibb: aa_j / det,
            }
        })
        .collect()
}

// ── The replay harness ──────────────────────────────────────────────────

struct Harness<'a> {
    solves: &'a [SlotSolve],
    graft: &'a Graft,
    n_embd: usize,
    layers: &'a [u32],
    slots: usize,
    d: usize,
}

impl Harness<'_> {
    /// Rebuild one row's activations: per slot, solve [α, β] so the read
    /// functional lands exactly on the frozen triple value and the switch
    /// functional lands ±1 (clearly on the recorded side of the 0.5
    /// threshold — `exact_sigmoid(±1) = 0.73/0.27`). Margins absorb the
    /// f32 reconstruction error (~1e-3 vs a 0.5 quantization bucket).
    fn fill(&self, row: &FrozenRow, x: &mut [f32]) {
        let target = if row.armed { 1.0f64 } else { -1.0f64 };
        let s_bias = f64::from(self.graft.switches[0].bias);
        for (s, solve) in self.solves.iter().enumerate() {
            let r = f64::from(row.triple[s]) - f64::from(self.graft.reads[s].bias);
            let c = (target - s_bias) / 3.0;
            let alpha = solve.iaa * r + solve.iab * c;
            let beta = solve.iab * r + solve.ibb * c;
            let base = s * self.d;
            for (i, (av, bv)) in solve.a.iter().zip(solve.b.iter()).enumerate() {
                x[base + i] = (alpha * f64::from(*av) + beta * f64::from(*bv)) as f32;
            }
        }
    }

    fn reader<'b>(&self, x: &'b [f32]) -> ReplayReader<'_, 'b> {
        ReplayReader {
            n_embd: self.n_embd,
            layers: self.layers,
            slots: (0..self.slots)
                .map(|s| &x[s * self.d..(s + 1) * self.d])
                .collect(),
        }
    }

    /// The reconstruction's authority: the REAL read path must reproduce
    /// the frozen row exactly (triple + arming bit) or the fixture is
    /// invalid — loud, never silent.
    fn verify(&self, row: &FrozenRow, reader: &ReplayReader) {
        for (j, read) in self.graft.reads.iter().enumerate() {
            let v = read.value_at(reader);
            assert_eq!(
                v, row.triple[j],
                "row {} read {j}: replay {v} vs frozen {}",
                row.instance, row.triple[j]
            );
        }
        assert_eq!(
            self.graft.switches[0].armed_at(reader),
            row.armed,
            "row {} arming replay mismatch",
            row.instance
        );
    }

    /// One arm over a frozen corpus: replay every row, apply, count the
    /// fires + folds any fired biases into the Zipf surface (Zipf-surface
    /// KL(top-40) + top-1 flip per FIRED row — reported, never a real
    /// model). Returns (mean Zipf-KL over fired rows, flips, fires).
    #[allow(clippy::too_many_arguments)]
    fn measure_fires(
        &self,
        arm: &Graft,
        corpus: &[&FrozenRow],
        base: &[f64],
        top: &[usize],
        x: &mut [f32],
        iv: &mut [u16],
        ov: &mut [u16],
    ) -> (f64, usize, usize) {
        let mut kl_sum = 0.0f64;
        let mut flips = 0usize;
        let mut fires = 0usize;
        for row in corpus {
            self.fill(row, x);
            let reader = self.reader(x);
            let mut vw = VecWriter { biases: Vec::new() };
            let applied = arm.apply(&reader, &mut vw, iv, ov);
            if applied.fired {
                fires += 1;
                let mut biased = base.to_vec();
                for (tok, b) in &vw.biases {
                    biased[*tok as usize] += f64::from(*b);
                }
                let (kl, flip) = kl_flip(base, &biased, top);
                kl_sum += kl;
                if flip {
                    flips += 1;
                }
            }
        }
        (kl_sum / fires.max(1) as f64, flips, fires)
    }
}

/// Top-k restricted, renormalized KL(p‖q) in f64 sequential folds (the
/// partition functions cancel, so the restricted exponentials are exact)
/// + the full-vocab top-1 flip verdict (strict `>` — first index wins).
fn kl_flip(base: &[f64], biased: &[f64], top: &[usize]) -> (f64, bool) {
    let argmax_first = |v: &[f64]| -> usize {
        let mut best = 0usize;
        let mut bv = f64::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                best = i;
            }
        }
        best
    };
    let mut sp = 0.0f64;
    let mut sq = 0.0f64;
    for &i in top {
        sp += base[i].exp();
        sq += biased[i].exp();
    }
    let mut kl = 0.0f64;
    for &i in top {
        let p = base[i].exp() / sp;
        let q = biased[i].exp() / sq;
        kl += p * (p / q).ln();
    }
    (kl, argmax_first(base) != argmax_first(biased))
}

fn gate(ok: bool) -> &'static str {
    if ok { "OK" } else { "FAIL" }
}

fn main() {
    println!("bench_988_grafts_goat — issue 988 T4 (frozen-read replay posture)");

    // ── Fixture pins + decode + mount laws ──
    let af_hash = blake3::hash(ARTIFACT_BYTES).to_hex();
    assert_eq!(
        af_hash.as_str(),
        ARTIFACT_BLAKE3,
        "artifact fixture digest pin"
    );
    let rf_hash = blake3::hash(ROWS_BYTES).to_hex();
    assert_eq!(rf_hash.as_str(), ROWS_BLAKE3, "rows fixture digest pin");
    let artifact = parse_artifact(ARTIFACT_BYTES);
    assert_eq!(
        artifact.graft_digest_hex(),
        GRAFT_DIGEST_HEX,
        "graft content digest pin"
    );
    let graft = artifact.decode().expect("artifact decode");
    let rows = parse_rows(ROWS_BYTES);
    // The pinned leg map (998 T2 pre-registration): a truncated corpus
    // cannot pass a power-disclosed gate by shrinking silently.
    let leg = |class_task: bool, split: u8| -> Vec<&FrozenRow> {
        rows.iter()
            .filter(|r| r.split == split && (class_task == (r.class == 0)))
            .collect()
    };
    let solved_leg = leg(true, SPLIT_SOLVED);
    let dev_leg = leg(true, SPLIT_DEV);
    let offfit_leg = leg(false, SPLIT_OFFTASK_FIT);
    let test_leg = leg(false, SPLIT_OFFTASK_TEST);
    let cal_leg = leg(false, SPLIT_OFFTASK_CALIBRATION);
    let hard_leg: Vec<&FrozenRow> = test_leg.iter().copied().filter(|r| r.class >= 2).collect();
    assert_eq!(solved_leg.len(), LEG_SOLVED_N, "solved leg size");
    assert_eq!(dev_leg.len(), LEG_DEV_N, "dev leg size");
    assert_eq!(offfit_leg.len(), LEG_OFFTASK_FIT_N, "off-task fit leg size");
    assert_eq!(test_leg.len(), LEG_OFFTASK_TEST_N, "off-task TEST leg size");
    assert_eq!(hard_leg.len(), LEG_HARD_SUB_N, "hard-negative sub-leg size");
    assert_eq!(
        cal_leg.len(),
        LEG_CALIBRATION_N,
        "calibration leg size (reported, never gated)"
    );
    let mut mount = GraftSlot::new();
    assert!(matches!(
        mount.install(&artifact),
        Ok(GraftInstall::Advanced)
    ));
    assert!(matches!(
        mount.install(&artifact),
        Ok(GraftInstall::Idempotent)
    ));
    assert!(mount.graft().is_some());

    // ── Derived shape (everything comes from the artifact) ──
    let slots = graft.reads.len();
    let n_embd = graft.reads[0].segments[0].weights.len();
    let k = graft.reads[0].segments.len();
    let d = k * n_embd;
    let layers: Vec<u32> = graft.reads[0]
        .segments
        .iter()
        .map(|s| s.layer % 100)
        .collect();
    for (ri, read) in graft.reads.iter().enumerate() {
        assert_eq!(read.segments.len(), k, "read {ri} segment count");
        assert_eq!(read.domain, 10, "read {ri} domain");
        for (ki, seg) in read.segments.iter().enumerate() {
            assert_eq!(seg.weights.len(), n_embd, "read {ri} segment {ki} width");
            assert_eq!(seg.layer, ri as u32 * 100 + layers[ki], "read {ri} id map");
        }
    }
    assert_eq!(graft.writes.len(), 1, "single write spec expected");
    assert_eq!(graft.writes[0].cell, 0, "answer cell");
    assert_eq!(graft.writes[0].entries.len(), 10, "ten digit write entries");
    let mut write_tokens: Vec<u32> = Vec::new();
    for e in &graft.writes[0].entries {
        assert_eq!(e.biases.len(), 1, "one bias per digit entry");
        let (tok, bias) = e.biases[0];
        assert!((bias - WRITE_BIAS).abs() < 1e-6, "write delta drift");
        write_tokens.push(tok);
    }
    assert_eq!(
        write_tokens
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        10,
        "digit tokens distinct"
    );
    let vocab = usize::try_from(*write_tokens.iter().max().unwrap_or(&0)).unwrap_or(0) + 1;
    println!(
        "shape: {slots} reads × {k} segments × {n_embd} dims · switch segments {} · table entries {} · write surface 10 tokens (max id {}, vocab {vocab})",
        graft.switches[0].segments.len(),
        graft.table.rules[0].entries.len(),
        vocab - 1
    );

    // ── The harness + reusable buffers ──
    let solves = build_solves(&graft);
    let harness = Harness {
        solves: &solves,
        graft: &graft,
        n_embd,
        layers: &layers,
        slots,
        d,
    };
    let mut x = vec![0f32; slots * d];
    let mut iv = vec![0u16; graft.n_inputs()];
    let mut ov = vec![0u16; graft.n_outputs()];
    assert_eq!(ov.len(), 1, "single answer cell expected");

    // The unconditional decision-layer variant (fields are pub; the
    // switches list is the only difference).
    let graft_uncond = Graft::new(
        graft.reads.clone(),
        graft.table.clone(),
        graft.writes.clone(),
        Vec::new(),
    )
    .expect("unconditional variant");
    let mut disarmed = graft.clone();
    disarmed.disarm();

    // ── G1: the frozen-read replay ──
    println!("\n── G1: frozen-read replay ──");
    let solved_n = rows.iter().filter(|r| r.split == 0).count();
    assert_eq!(solved_n, 400, "solved split size");
    let mut solved_fired = 0usize;
    let mut dev_rows_n = 0usize;
    let mut dev_fired = 0usize;
    let mut dev_correct = 0usize;
    for row in &rows {
        harness.fill(row, &mut x);
        let reader = harness.reader(&x);
        harness.verify(row, &reader);
        let mut vw = VecWriter { biases: Vec::new() };
        let applied = graft_uncond.apply(&reader, &mut vw, &mut iv, &mut ov);
        if row.split == 0 {
            if applied.fired {
                solved_fired += 1;
                assert_eq!(
                    ov[0],
                    u16::from(row.answer),
                    "solved row {} fired with a wrong answer",
                    row.instance
                );
            }
        } else {
            dev_rows_n += 1;
            if applied.fired {
                dev_fired += 1;
                if ov[0] == u16::from(row.answer) {
                    dev_correct += 1;
                }
            }
        }
    }
    assert!(
        solved_fired >= 300,
        "replay fired implausibly few solved rows ({solved_fired}/400 — coverage was 0.79)"
    );
    println!(
        "solved: fired {solved_fired}/400 — every fired row decided its TRUE answer (asserted)"
    );
    println!(
        "dev:    fired {dev_fired}/{dev_rows_n}, correct-on-fired {dev_correct} (held-out, reported)"
    );

    // Determinism: double-apply byte identity + scramble construction.
    {
        let row = &rows[0];
        harness.fill(row, &mut x);
        let reader = harness.reader(&x);
        let mut w1 = VecWriter { biases: Vec::new() };
        let mut w2 = VecWriter { biases: Vec::new() };
        graft_uncond.apply(&reader, &mut w1, &mut iv, &mut ov);
        graft_uncond.apply(&reader, &mut w2, &mut iv, &mut ov);
        assert_eq!(w1.biases, w2.biases, "double-apply must be byte-identical");
        let s1 = graft_uncond.scrambled(SCRAMBLE_SEED);
        let s2 = graft_uncond.scrambled(SCRAMBLE_SEED);
        let mut o1 = vec![0u16; graft.n_outputs()];
        let mut o2 = vec![0u16; graft.n_outputs()];
        let a1 = s1.apply(&reader, &mut NullWriter { n: 0 }, &mut iv, &mut o1);
        let a2 = s2.apply(&reader, &mut NullWriter { n: 0 }, &mut iv, &mut o2);
        assert_eq!(
            (a1.fired, o1[0]),
            (a2.fired, o2[0]),
            "scramble construction must be deterministic"
        );
    }

    // Scramble arm: the wiring is wrong by construction — decisions must
    // differ somewhere across the corpus (or the permutation is the
    // identity, disclosed; the graft.rs fixtures pin the mechanism).
    let scr = graft_uncond.scrambled(SCRAMBLE_SEED);
    let mut scr_differs = 0usize;
    for row in &rows {
        harness.fill(row, &mut x);
        let reader = harness.reader(&x);
        let mut o0 = vec![0u16; graft.n_outputs()];
        let mut o1 = vec![0u16; graft.n_outputs()];
        let a0 = graft_uncond.apply(&reader, &mut NullWriter { n: 0 }, &mut iv, &mut o0);
        let a1 = scr.apply(&reader, &mut NullWriter { n: 0 }, &mut iv, &mut o1);
        if (a0.fired, o0.as_slice()) != (a1.fired, o1.as_slice()) {
            scr_differs += 1;
        }
    }
    println!(
        "scramble: decisions differ on {scr_differs} rows{}",
        if scr_differs > 0 {
            String::new()
        } else {
            " (identity permutation — disclosed; mechanism pinned by the graft fixtures)"
                .to_string()
        }
    );

    let sealed_ok = SEALED_GRAFT_ACC >= SEALED_ACC_BAR;
    assert!(
        sealed_ok,
        "the sealed G1 bar arithmetic must hold (it passed at Bench 971)"
    );
    println!(
        "sealed pin: baseline {SEALED_BASELINE_ACC} · graft {SEALED_GRAFT_ACC} · payload {} · bar {SEALED_ACC_BAR} → OK",
        &SEALED_PAYLOAD_BLAKE3[..16]
    );
    // Reaching this line means every hard G1 assert passed.
    let g1 = true;
    println!("G1: OK (pins · mount · reconstruction · decisions · determinism · sealed bar)");

    // ── G3(a): switches disarmed — byte-identity ──
    println!("\n── G3(a): switches disarmed — byte-identity ──");
    let mut touched = 0usize;
    for row in &rows {
        harness.fill(row, &mut x);
        let reader = harness.reader(&x);
        let mut nw = NullWriter { n: 0 };
        disarmed.apply(&reader, &mut nw, &mut iv, &mut ov);
        touched += nw.n;
    }
    let g3a = touched == 0;
    println!(
        "disarmed: {} applies — writer touched {touched} times → {}",
        rows.len(),
        gate(g3a)
    );

    // ── G3(b): the 998 T2 no-harm legs at +8.0 ──
    // GATED leg 2 — the FRESH off-task TEST corpus (n = 600): gate = CP
    // 95% UB ≤ 0.5% ⇔ 0 false ARMS (1/600 ≈ 0.79% = FAIL). The gate
    // quantity is ARMS — the pre-registered unit (the 998 T2 posture-2
    // + posture-3 rulings: the text decides, and arms is the durable quantity: the
    // fire count depends on the CURRENT mined table, whose growth turns
    // false arms into fires). FIRES are reported beside, never gating:
    // the all-or-nothing table law abstains undecided triples, so the
    // fire count is today's table hiding the arm rate. KL is
    // Zipf-SURFACE KL — never a real-model no-harm proof.
    println!(
        "\n── G3(b) off-task FRESH-TEST leg (gated: 0 false ARMS at n = {} = CP 95% UB ≤ {:.1}%); fires reported beside ──",
        test_leg.len(),
        G3_FLIP_BAR * 100.0
    );
    let zipf = |i: usize| -> f64 { -1.07 * (i as f64 + 8.0).ln() };
    let base: Vec<f64> = (0..vocab).map(zipf).collect();
    let mut top: Vec<usize> = (0..vocab).collect();
    top.sort_by(|&a, &b| base[b].total_cmp(&base[a]).then(a.cmp(&b)));
    top.truncate(KL_TOPK);
    // The gate quantity: ARMS — every recorded armed bit re-verified
    // through the REAL armed_at on the reconstruction (the same
    // law the task contract uses), never trusted from the fixture.
    let mut off_arms = 0usize;
    for row in &test_leg {
        harness.fill(row, &mut x);
        let reader = harness.reader(&x);
        let replay = graft.switches_armed_at(&reader);
        assert_eq!(
            replay, row.armed,
            "test row {} arming replay mismatch",
            row.instance
        );
        if row.armed {
            off_arms += 1;
        }
    }
    let (off_kl, off_flips, off_fires) =
        harness.measure_fires(&graft, &test_leg, &base, &top, &mut x, &mut iv, &mut ov);
    // One-sided Clopper–Pearson 95% upper bound for the observed count.
    let cp_ub = |fires: usize, n: usize| -> f64 {
        // Solve sum_{k≤fires} C(n,k) p^k (1-p)^(n-k) = 0.05 by bisection
        // on p — 60 iterations is far past f64 resolution.
        if fires >= n {
            return 1.0;
        }
        let mut lo = 0.0f64;
        let mut hi = 1.0f64;
        let cdf = |p: f64| -> f64 {
            let mut total = 0.0f64;
            let mut choose = 1.0f64; // C(n, 0)
            let mut pk = 1.0f64; // p^0
            let mut failure = (1.0 - p).powi(n as i32);
            for k in 0..=fires {
                total += choose * pk * failure;
                choose *= (n - k) as f64 / (k + 1) as f64;
                pk *= p;
                failure /= 1.0 - p;
            }
            total
        };
        for _ in 0..60 {
            let mid = (lo + hi) / 2.0;
            // cdf is DECREASING in p; we want the crossing of 0.05.
            // Above the crossing cdf > 0.05, below it cdf <= 0.05.
            if cdf(mid) > 0.05 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) / 2.0
    };
    let off_ub = cp_ub(off_arms, test_leg.len());
    let g3b_offtask = off_arms == 0;
    println!(
        "off-task FRESH TEST: {off_arms} false ARMS / {} (the gated unit) — fires {off_fires} (reported; the all-or-nothing table law abstains undecided triples, so today's table hides the arm rate) — Zipf-surface KL(top-40) {off_kl:.4} nats over fired rows, {off_flips} flips — CP 95% UB on arms {off_ub:.4} → {}",
        test_leg.len(),
        gate(g3b_offtask)
    );

    // GATED leg 3 — hard-negative sub-leg (class ≥ 2 TEST rows, n = 200):
    // same gate shape; the thinner power is DISCLOSED on this line
    // (0/200 certifies a CP 95% UB ≤ 1.5%, not ≤ 0.5%). If the detector
    // only passes the easy rows, the artifact is not self-selective.
    println!(
        "\n── G3(b) hard-negative sub-leg (gated: 0 false arms at n = {}; disclosed power: 0/200 certifies ≤ 1.5%) ──",
        hard_leg.len()
    );
    // The gate quantity: ARMS (the pre-registered unit — same law as
    // the full-test leg), verified through the REAL armed_at replay.
    let mut hard_arms = 0usize;
    for row in &hard_leg {
        harness.fill(row, &mut x);
        let reader = harness.reader(&x);
        let replay = graft.switches_armed_at(&reader);
        assert_eq!(
            replay, row.armed,
            "hard row {} arming replay mismatch",
            row.instance
        );
        if row.armed {
            hard_arms += 1;
        }
    }
    let (hard_kl, hard_flips, hard_fires) =
        harness.measure_fires(&graft, &hard_leg, &base, &top, &mut x, &mut iv, &mut ov);
    let hard_ub = cp_ub(hard_arms, hard_leg.len());
    let g3b_hard = hard_arms == 0;
    println!(
        "hard-negative (digits-non-arithmetic + arithmetic-words): {hard_arms} false ARMS / {} (the gated unit) — fires {hard_fires} (reported) — Zipf-surface KL {hard_kl:.4} nats, {hard_flips} flips — CP 95% UB on arms {hard_ub:.4} → {}",
        hard_leg.len(),
        gate(g3b_hard)
    );

    // GATED leg 1 — the task contract on the dev-200 HELD-OUT rows: the
    // recorded armed bit must be 1 everywhere, verified through the REAL
    // armed_at replay. Any disarm = lane-A GOAT FAIL (998 T2: the
    // threshold never moves on held-out evidence; the sealed-400
    // capture-only arm check is the cell-side half of this contract).
    println!(
        "\n── G3(b) task contract, dev-200 held-out (gated: disarm rate = 0; sealed-400 capture-only check runs cell-side) ──"
    );
    let mut dev_disarmed = 0usize;
    for row in &dev_leg {
        harness.fill(row, &mut x);
        let reader = harness.reader(&x);
        let replay = graft.switches_armed_at(&reader);
        assert_eq!(
            replay, row.armed,
            "dev row {} arming replay mismatch",
            row.instance
        );
        if !row.armed {
            dev_disarmed += 1;
        }
    }
    let g3b_contract = dev_disarmed == 0;
    println!(
        "task contract (dev held-out): {dev_disarmed} disarmed / {} → {} (fit-split rates live in the export manifest — reported, never gated)",
        dev_leg.len(),
        gate(g3b_contract)
    );

    // REPORTED — the task-corpus base-right damage (the accepted trade,
    // 998 T2 pre-registration: never gated, printed beside every verdict).
    println!("\n── G3(b) task base-right corpus (REPORTED, accepted trade — never gated) ──");
    let base_right: Vec<&FrozenRow> = rows
        .iter()
        .filter(|r| r.class == 0 && !r.base_wrong)
        .collect();
    let (trade_kl, trade_flips, trade_fires) =
        harness.measure_fires(&graft, &base_right, &base, &top, &mut x, &mut iv, &mut ov);
    println!(
        "task base-right: {trade_fires} fires / {} rows (armed-posture replay) — Zipf-surface KL {trade_kl:.4} nats, {trade_flips} flips — the accepted trade against +{}pp net lane accuracy (Bench 971: baseline 0.125 → graft 0.3925)",
        base_right.len(),
        ((SEALED_GRAFT_ACC - SEALED_BASELINE_ACC) * 100.0) as i64
    );

    let g3b = g3b_offtask && g3b_hard && g3b_contract;

    // ── G2: hot-path latency ──
    println!(
        "\n── G2: hot-path latency (bar ≤ {G2_BAR_US} µs/apply, best_of_us {G2_WINDOWS}×{G2_APPLIES}) ──"
    );
    let mut syn = vec![0f32; slots * d];
    let mut sm: u64 = 0x988_7A5E;
    for v in &mut syn {
        sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = sm;
        let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB) ^ (z >> 31);
        *v = ((z & 0xFFFF) as f32 / 65535.0) * 2.0 - 1.0;
    }
    let syn_reader = harness.reader(&syn);
    let mut nw = NullWriter { n: 0 };
    let mut acc = 0usize;
    // Issue-855 treatment: the timed region runs through `best_of_us` (the
    // shared harness — the loud-zero defence: it panics when every window
    // reads 0 ns, i.e. an apply loop the optimiser deleted). Same quantity
    // as before: the min over windows of the per-apply mean.
    let best = ab_timing::best_of_us(1, G2_WINDOWS, || {
        let t = Instant::now();
        for _ in 0..G2_APPLIES {
            let applied = graft.apply(black_box(&syn_reader), &mut nw, &mut iv, &mut ov);
            acc += applied.decided;
        }
        t.elapsed()
    }) / G2_APPLIES as f64;
    black_box(acc);
    black_box(nw.n);
    let g2 = best <= G2_BAR_US;
    println!(
        "best of {G2_WINDOWS}×{G2_APPLIES} applies: {best:.2} µs/apply vs bar {G2_BAR_US} → {} (quote numbers only with bench_preflight provenance)",
        gate(g2)
    );

    // ── G4: zero allocation on the armed hot path ──
    println!("\n── G4: zero allocation on the armed hot path ──");
    for _ in 0..100 {
        let _ = graft.apply(&syn_reader, &mut nw, &mut iv, &mut ov);
    }
    let a0 = ALLOCS.load(Ordering::SeqCst);
    for _ in 0..1000 {
        let _ = graft.apply(&syn_reader, &mut nw, &mut iv, &mut ov);
    }
    let a1 = ALLOCS.load(Ordering::SeqCst);
    let g4 = a1 == a0;
    println!("alloc delta over 1000 applies: {} → {}", a1 - a0, gate(g4));

    // ── Verdict ──
    println!("\n── Verdict ──");
    println!(
        "GATES: G1 {} · G2 {} · G3(a) {} · G3(b) {} · G4 {}",
        gate(g1),
        gate(g2),
        gate(g3a),
        gate(g3b),
        gate(g4)
    );
    if g2 && g3a && g3b && g4 {
        println!(
            "→ GOAT: all mechanical gates PASS — promotion-eligible (the default-on flip is the promotion call per the Feature Flag Discipline)"
        );
    } else {
        let failed: Vec<&str> = [("G2", g2), ("G3(a)", g3a), ("G3(b)", g3b), ("G4", g4)]
            .iter()
            .filter(|(_, ok)| !ok)
            .map(|(name, _)| *name)
            .collect();
        println!(
            "→ GOAT: NOT PROMOTED — failing gates: {}",
            failed.join(", ")
        );
        if !g3b {
            if !g3b_contract {
                println!(
                    "  G3(b) task-contract FAIL: the detector disarmed held-out task rows — lane-A GOAT FAIL \
                     (998 T2: the threshold never moves on held-out evidence; a fresh sealed read is forced)."
                );
            }
            if !g3b_offtask {
                println!(
                    "  G3(b) off-task FAIL: {off_arms} false arm(s) (gated unit) — the artifact is not self-selective at the placed \
                     threshold; FAIL is a measured verdict, never a moved threshold (998 T2 posture-2 + posture-3 verdicts; posture 3 permanently closes this placement family on FAIL)."
                );
            }
            if !g3b_hard {
                println!(
                    "  G3(b) hard-negative FAIL: {hard_arms} false arm(s) on the digits/arithmetic-words corpus — the \
                     detector passes only easy rows; the artifact is not self-selective on real traffic."
                );
            }
        }
        if !g2 {
            println!(
                "  G2 note: a loaded box inflates the min-window mean — the record verdict needs a \
                 bench_preflight.sh PROVENANCE run."
            );
        }
        std::process::exit(1);
    }
}
