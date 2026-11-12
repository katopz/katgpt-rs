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
//! - **G3(b)** — switches ARMED on the UNRELATED corpus (the frozen
//!   base-right rows: traffic the base already answers — the graft must
//!   not disturb it) at the pinned write delta 8.0: per-position
//!   KL(top-40 vs baseline) corpus mean **≤ 0.02 nats** AND top-1 flip
//!   rate **≤ 0.5%** (48 rows ⇒ the flip bar admits zero fires: 1/48 =
//!   2.08%). The Zipf logit surface (−1.07·ln(i+8)) is the documented
//!   synthetic stand-in for the host LM head; the damage a fire causes
//!   is what the bars quantify.
//! - **G4** — zero allocation on the armed hot path (counting
//!   allocator, delta over 1000 applies == 0; the `graft_g4_alloc`
//!   fixture is the primary evidence, this is the re-assert).
//!
//! **Pre-registered expectation (disclosed before this run):** the
//! export crosstab measured the fitted switch ARMING ON EVERY row
//! (solved right+armed 31 / wrong+armed 369; dev right+armed 17 /
//! wrong+armed 183 — zero disarmed anywhere). `base_wrong` is an
//! output-space property (what the argmax will do), weakly encoded in
//! the input state at the capture position; the {0,1} LS fit had no
//! separating signal and collapsed to the arm-always predictor (92% of
//! labels are 1). G3(b) is therefore expected to FAIL at this fit — the
//! bench measures it anyway and the GOAT refuses promotion on the
//! measured verdict. `grafts` stays opt-in.
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

const ARTIFACT_BLAKE3: &str = "c8dd641527dc04081f5e08d4e441b8991a488482068d5895dba6d4d968fb8083";
const ROWS_BLAKE3: &str = "3dfa188a0a5c529eb387963133a7d885766bf4e5823224c15ac34329910203da";
const GRAFT_DIGEST_HEX: &str = "1faa5513363c77591cd34c3dff774c1cf676663bbfb6e8e8f9a20385e6e6c8d8";
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

/// G3(b): the pre-registered no-regression bars at the pinned delta.
const G3_KL_BAR_NATS: f64 = 0.02;
const G3_FLIP_BAR: f64 = 0.005;
const WRITE_BIAS: f32 = 8.0;
const KL_TOPK: usize = 40;

/// The T2b cell's pinned scramble seed.
const SCRAMBLE_SEED: u64 = 0x5EED_9880;

const ROWS_VERSION: u32 = 1;
const ROW_BYTES: usize = 14;

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
    split: u8, // 0 = solved, 1 = dev
}

fn parse_rows(bytes: &[u8]) -> Vec<FrozenRow> {
    assert!(bytes.len() >= 12, "rows fixture truncated (header)");
    assert_eq!(&bytes[0..4], b"GRFW".as_slice(), "rows fixture magic");
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(version, ROWS_VERSION, "rows fixture version");
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
                split: bytes[o + 13],
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

    /// One arm over the base-right (unrelated) corpus: apply, fold any
    /// fired biases into the Zipf surface, measure KL(top-40) + top-1
    /// flip per position, average over the whole corpus. Returns
    /// (mean KL nats, flip rate, fires, corpus size).
    #[allow(clippy::too_many_arguments)]
    fn measure_arm(
        &self,
        arm: &Graft,
        label: &str,
        rows: &[FrozenRow],
        base: &[f64],
        top: &[usize],
        x: &mut [f32],
        iv: &mut [u16],
        ov: &mut [u16],
    ) -> (f64, f64, usize, usize) {
        let corpus: Vec<&FrozenRow> = rows.iter().filter(|r| !r.base_wrong).collect();
        let mut kl_sum = 0.0f64;
        let mut flips = 0usize;
        let mut fires = 0usize;
        for row in &corpus {
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
        let n = corpus.len();
        let mean = kl_sum / n.max(1) as f64;
        let rate = flips as f64 / n.max(1) as f64;
        println!(
            "{label}: {n} base-right rows · fires {fires} · mean KL(top-{KL_TOPK}) {mean:.4} nats · flips {flips} ({:.2}%)",
            rate * 100.0
        );
        (mean, rate, fires, n)
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

    // ── G3(b): armed on the unrelated (base-right) corpus at +8.0 ──
    println!("\n── G3(b): armed on the unrelated (base-right) corpus at +{WRITE_BIAS} ──");
    let zipf = |i: usize| -> f64 { -1.07 * (i as f64 + 8.0).ln() };
    let base: Vec<f64> = (0..vocab).map(zipf).collect();
    let mut top: Vec<usize> = (0..vocab).collect();
    top.sort_by(|&a, &b| base[b].total_cmp(&base[a]).then(a.cmp(&b)));
    top.truncate(KL_TOPK);
    let (cond_kl, cond_rate, cond_fires, corpus_n) = harness.measure_arm(
        &graft,
        "conditional",
        &rows,
        &base,
        &top,
        &mut x,
        &mut iv,
        &mut ov,
    );
    let _ = harness.measure_arm(
        &graft_uncond,
        "unconditional (contrast)",
        &rows,
        &base,
        &top,
        &mut x,
        &mut iv,
        &mut ov,
    );
    let g3b = cond_kl <= G3_KL_BAR_NATS && cond_rate <= G3_FLIP_BAR;
    println!(
        "bars: mean KL ≤ {G3_KL_BAR_NATS} nats · flip rate ≤ {:.1}% → {} ({cond_fires} fires over {corpus_n} rows)",
        G3_FLIP_BAR * 100.0,
        gate(g3b)
    );

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
            println!(
                "  G3(b) root cause (measured, issue 998): the fitted switch arms the WHOLE task surface — \
                 the direction that separates base-wrongness in-sample is memorization (train AUC 1.0000, held-out AUC 0.5966), \
                 and every fitted score is positive so the 0.5-sigmoid threshold never closes the gate. \
                 A task-detector switch (needs non-task fit rows) or consumer-side mounting is the fix; grafts stays opt-in."
            );
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
