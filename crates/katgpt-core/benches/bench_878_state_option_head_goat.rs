//! Plan 607 T3 — `state_option_scoring::head` GOAT gate (G1 floors + G2
//! latency + determinism digest).
//!
//! Gates (run: `cargo bench -p katgpt-core --features distance_abstain
//! --bench bench_878_state_option_head_goat` at `--release` — the profile
//! AGENTS.md mandates for gates):
//!
//! - **G1a planted-fit correctness** — a planted linear head generates the
//!   targets; the fit must recover the RANKING (the argmax, not the
//!   magnitudes) at every rotating option index across 200 decision sets:
//!   a fitter that cannot re-rank its own generated data is broken, full
//!   stop.
//! - **G1b reflex floors** — the shipped reflex floor forms, never laya
//!   alone (Plan 607 R2): distinct picks ≥ 2 over distinct state rows, and
//!   the constant-pick fraction printed.
//! - **G2 latency** — p99 per decision SET (head `pick` over K rows at the
//!   design width D) with the OPTION COUNT printed beside it (the
//!   sibling's shipped bar form). Bar: p99 ≤ 1 ms — the head's pick is a
//!   D-dot per option, strictly cheaper than the T1 table path. Tail
//!   support printed beside every percentile (the repo's
//!   percentile-reporting rule).
//! - **G4** — pointer only here: the alloc gate is the separate
//!   `state_option_head_alloc_check` binary (the `*_alloc_check`
//!   convention).
//! - **Determinism** — the fitted weights ARE the determinism-committed
//!   artifact (Plan 607 T3's line: no RNG, no iterations, no gradient
//!   descent): double fit bit-identical, asserted AND digested (BLAKE3
//!   over the weight bytes + over the G1a decision stream). The printed
//!   digests are the two-box comparison anchors (recorded in
//!   `.benchmarks/878_state_option_head_goat.md`).
//!
//! The fixture-replay AGREEMENT half of the GOAT (G1 vs the T0b oracle
//! triples + constant-pick + chance, in-corpus AND leave-one-state-out)
//! lives in the ROOT package's arena (`examples/tetris_03_head_fit.rs`) —
//! game vocabulary stays out of katgpt-core; this bench is self-contained
//! synthetic.
//!
//! Loud-zero defense: every timed loop's outputs feed `black_box` sinks so
//! LTO cannot delete the work it timed (the timed-region rule).
#![cfg(feature = "distance_abstain")]

use katgpt_core::state_option_scoring::head::{FittedHead, HeadFitter};
use std::hint::black_box;
use std::time::Instant;

/// Design width (the first consumer's shape: 11 standardized features +
/// intercept). Widths, not vocabulary — nothing here knows what the
/// columns mean.
const D: usize = 12;
/// The first consumer's decision-set widths (9 / 17 / 34 — the fixture's
/// real option counts).
const OPTION_WIDTHS: [usize; 3] = [9, 17, 34];
/// Pinned fit ridge (standardized scale; the arena selects within the
/// grid, the synthetic bench pins one).
const RIDGE: f64 = 1e-2;
const SETS: usize = 200;
const TRAIN_ROWS: usize = 256;
const TIMED_ITERS: usize = 2000;
const WARMUP: usize = 64;

/// Deterministic LCG — no RNG dep, no global state (the global-RNG gate's
/// seeded-local law).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn f64_unit(&mut self) -> f64 {
        ((self.next() >> 40) as f64) / (1u64 << 24) as f64 * 2.0 - 1.0
    }
    fn vector<const N: usize>(&mut self) -> [f64; N] {
        let mut v = [0.0f64; N];
        for x in v.iter_mut() {
            *x = self.f64_unit();
        }
        v
    }
}

fn main() {
    println!("== Plan 607 T3 — state_option_scoring::head GOAT (D={D}, ridge={RIDGE}) ==");

    // ── G1a + G1b: planted-fit correctness across rotating indices ──────
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    let planted_w: [f64; D] = {
        let mut w = [0.5f64; D];
        for (i, x) in w.iter_mut().enumerate() {
            *x *= if i % 2 == 0 { 1.0 } else { -1.0 } * (1.0 + i as f64 * 0.125);
        }
        w
    };
    let mut fitter = HeadFitter::<D>::new();
    let mut planted_hits = 0usize;
    let mut picks_seen = std::collections::HashMap::new();
    let mut decision_stream: Vec<u8> = Vec::with_capacity(SETS);
    for _s in 0..SETS {
        // A training corpus whose targets ARE the planted head's outputs
        // (deterministic rows; the fit must re-cover the ranking).
        let mut train: Vec<[f64; D]> = Vec::with_capacity(TRAIN_ROWS);
        let mut y: Vec<f64> = Vec::with_capacity(TRAIN_ROWS);
        for _r in 0..TRAIN_ROWS {
            let row = rng.vector::<D>();
            let t = planted_w.iter().zip(row.iter()).map(|(w, v)| w * v).sum();
            train.push(row);
            y.push(t);
        }
        let head = fitter.fit_into(&train, &y, RIDGE);
        // The decision set: two candidate rows; the planted ranking says
        // which wins. Pairs with a near-tie planted margin are REDRAWN
        // (deterministic — same seed, same draws) so the assertion is a
        // ranking claim, never a coin flip on f64 noise.
        const MIN_MARGIN: f64 = 0.25;
        let (a, b, oracle_pick) = loop {
            let a = rng.vector::<D>();
            let b = rng.vector::<D>();
            let sa = planted_w
                .iter()
                .zip(a.iter())
                .map(|(w, v)| w * v)
                .sum::<f64>();
            let sb = planted_w
                .iter()
                .zip(b.iter())
                .map(|(w, v)| w * v)
                .sum::<f64>();
            if (sa - sb).abs() >= MIN_MARGIN {
                break (a, b, if sa >= sb { 0 } else { 1 });
            }
        };
        let rows = [a, b];
        let pick = head.pick(&rows, 2);
        if pick == oracle_pick {
            planted_hits += 1;
        }
        *picks_seen.entry(pick).or_insert(0) += 1;
        decision_stream.push(pick as u8);
        black_box((pick, &head, &rows));
    }
    println!("G1a planted-fit ranking: {planted_hits}/{SETS} (bar: {SETS}/{SETS})");
    assert_eq!(
        planted_hits, SETS,
        "the fit must recover the planted head's ranking on every fresh decision pair"
    );
    let distinct = picks_seen.len();
    let max_pick_freq = picks_seen.values().copied().max().unwrap_or(0);
    println!(
        "G1b reflex floors: distinct picks {distinct} (bar ≥ 2); constant-pick fraction {:.1}% [a constant picker's best case = the max single-pick frequency]",
        100.0 * max_pick_freq as f32 / SETS as f32
    );
    assert!(
        distinct >= 2,
        "discrimination floor: distinct picks ≥ 2 over distinct state vectors"
    );

    // ── G2: p99 per decision SET, option count printed beside it ────────
    // Decision path only (fit is cold): pre-fit one head, time `pick`.
    let mut train: Vec<[f64; D]> = Vec::with_capacity(TRAIN_ROWS);
    let mut y: Vec<f64> = Vec::with_capacity(TRAIN_ROWS);
    for _ in 0..TRAIN_ROWS {
        let row = rng.vector::<D>();
        train.push(row);
        y.push(row.iter().sum::<f64>());
    }
    let head = fitter.fit_into(&train, &y, RIDGE);
    println!("G2 latency per decision SET (head pick, {TRAIN_ROWS}-row trained, --release):");
    for &k in &OPTION_WIDTHS {
        let rows: Vec<[f64; D]> = (0..k).map(|_| rng.vector::<D>()).collect();
        let mut sink = 0usize;
        let mut acc = 0.0f64;
        let run = |sink: &mut usize, acc: &mut f64| {
            let pick = head.pick(&rows, k);
            *sink = (*sink + pick) % k.max(1);
            *acc += head.score(&rows[0]);
        };
        for _ in 0..WARMUP {
            run(&mut sink, &mut acc); // first-touch / cache settle
        }
        let mut samples: Vec<u128> = Vec::with_capacity(TIMED_ITERS);
        for _ in 0..TIMED_ITERS {
            let t = Instant::now();
            run(&mut sink, &mut acc);
            samples.push(t.elapsed().as_nanos());
        }
        black_box((sink, acc));
        samples.sort_unstable();
        let (p50, _) = katgpt_core::stats::nearest_rank(&samples, 0.50);
        let (p90, _) = katgpt_core::stats::nearest_rank(&samples, 0.90);
        let (p99, _) = katgpt_core::stats::nearest_rank(&samples, 0.99);
        let support_p99 = samples.len() - samples.partition_point(|&x| x < p99);
        println!(
            "  K={k:>2}  p50 {p50:>6} ns | p90 {p90:>6} ns | p99 {p99:>6} ns  (n={TIMED_ITERS}, tail@p99={support_p99})  [bar p99 ≤ 1 ms]"
        );
        assert!(
            p99 <= 1_000_000,
            "G2 FAIL: p99 {p99} ns > 1 ms per decision set at K={k}"
        );
    }

    // ── Determinism: double fit + BLAKE3 digests ─────────────────────────
    let h1 = fitter.fit_into(&train, &y, RIDGE);
    let h2 = fitter.fit_into(&train, &y, RIDGE);
    let bytes = |h: &FittedHead<D>| -> Vec<u8> {
        h.weights().iter().flat_map(|f| f.to_le_bytes()).collect()
    };
    let (b1, b2) = (bytes(&h1), bytes(&h2));
    assert_eq!(b1, b2, "same corpus → bit-identical head");
    println!("determinism: double-fit bit-identical ✓");
    println!("  head blake3      : {}", blake3::hash(&b1));
    println!(
        "  decisions blake3 : {}  (two-box anchors — record both)",
        blake3::hash(&decision_stream)
    );
    println!("G4: alloc gate = tests/state_option_head_alloc_check.rs (separate binary)");
    println!("G1 agreement vs the T0b oracle triples: examples/tetris_03_head_fit.rs (root)");
}
