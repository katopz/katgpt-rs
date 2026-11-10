//! Plan 607 T1 — `state_option_scoring` GOAT gate (G1 floors + G2 latency +
//! determinism digest).
//!
//! Gates (run: `cargo bench -p katgpt-core --features distance_abstain
//! --bench bench_876_state_option_scoring_goat` at `--release` — the profile
//! AGENTS.md mandates for gates):
//!
//! - **G1a planted correctness** — the planted option (the state's own
//!   direction, cosine 1.0) is picked at EVERY rotating index across 200
//!   decision sets: a scorer that cannot rank a perfect match against
//!   generic distractors is broken, full stop.
//! - **G1b reflex floors** — the shipped reflex floor forms, never laya
//!   alone (Plan 607 R2: reflex T7 measured 4-of-5 families
//!   constant-picking, so agreement must first be shown not to BE a
//!   constant pick): distinct picks ≥ 2 over distinct state vectors, and
//!   the constant-pick fraction is printed.
//! - **G2 latency** — p99 per decision SET with the OPTION COUNT printed
//!   beside it (the sibling's shipped bar form — a latency bar without its
//!   option count is the box-state defect class). Bar: p99 ≤ 1 ms per
//!   decision set at the first consumer's real option widths (9 / 17 /
//!   34; reflex measured 59–79 µs at ~20 options on the same box class).
//!   Tail support printed beside every percentile (the repo's
//!   percentile-reporting rule — an index without its support is the max
//!   wearing a percentile's name).
//! - **G4** — pointer only here: the alloc gate is the separate
//!   `state_option_scoring_alloc_check` binary (the `*_alloc_check`
//!   convention — a counting allocator would pick up sibling work in a
//!   shared binary).
//! - **Determinism** — the unit-normalized table is the scoring state:
//!   same corpus → bit-identical table, asserted by double-build AND
//!   digested (BLAKE3 over the row bytes + over the G1a decision stream).
//!   The printed digests are the two-box comparison anchors (recorded in
//!   `.benchmarks/876_state_option_scoring_goat.md`).
//!
//! The fixture-replay AGREEMENT half of the GOAT (G1 vs the T0b oracle
//! triples + constant-pick + chance baselines) lives in the ROOT package's
//! arena (`examples/tetris_02_option_arena.rs`) — game vocabulary stays
//! out of katgpt-core; this bench is self-contained synthetic.
//!
//! Loud-zero defense: every timed loop's outputs feed `black_box` sinks so
//! LTO cannot delete the work it timed (the timed-region rule).
#![cfg(feature = "distance_abstain")]

use katgpt_core::state_option_scoring::CentroidTable;
use std::hint::black_box;
use std::time::Instant;

const D: usize = 64;
/// The first consumer's decision-set widths (9 = the 1-rotation piece, 17 =
/// the 4-rotation 1-wide piece, 34 = the 4-rotation 2-wide pieces) —
/// widths, not vocabulary; nothing here knows what the options mean.
const OPTION_WIDTHS: [usize; 3] = [9, 17, 34];
const SCALE: f32 = 8.0; // the substrate's route scale (reflex ROUTE_SCALE)
const SETS: usize = 200;
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
    fn f32_unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn vector<const N: usize>(&mut self) -> [f32; N] {
        let mut v = [0.0f32; N];
        for x in v.iter_mut() {
            *x = self.f32_unit();
        }
        v
    }
}

/// One full decision set at width `K`: build the (stack-copied) option
/// matrix → table → pick → score. Returns (pick, picked score) so the
/// caller can sink them.
fn run_set<const K: usize>(options: &[[f32; D]; 34], state: &[f32; D]) -> (usize, f32) {
    let mut mat = [[0.0f32; D]; K];
    for (dst, src) in mat.iter_mut().zip(options.iter().take(K)) {
        *dst = *src;
    }
    let table = CentroidTable::<D, K>::new(&mat);
    let pick = table.pick(state);
    let mut scores = [0.0f32; K];
    let argmax = table.score_into(state, SCALE, &mut scores);
    debug_assert_eq!(pick, argmax, "pick == score_into argmax for scale > 0");
    (pick, scores[pick])
}

fn main() {
    println!("== Plan 607 T1 — state_option_scoring GOAT (D={D}, scale={SCALE}) ==");

    // ── G1a + G1b: planted correctness across rotating indices ──────────
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    let mut planted_hits = 0usize;
    let mut picks_seen = std::collections::HashMap::new();
    let mut decision_stream: Vec<u8> = Vec::with_capacity(SETS);
    for s in 0..SETS {
        let k = OPTION_WIDTHS[s % OPTION_WIDTHS.len()];
        let state: [f32; D] = rng.vector();
        let planted = s % k;
        let mut options = [[0.0f32; D]; 34];
        for o in options[..k].iter_mut() {
            *o = rng.vector();
        }
        options[planted] = state; // cosine 1.0 — a perfect match the scorer must rank first
        let (pick, _) = match k {
            9 => run_set::<9>(&options, &state),
            17 => run_set::<17>(&options, &state),
            34 => run_set::<34>(&options, &state),
            _ => unreachable!(),
        };
        if pick == planted {
            planted_hits += 1;
        }
        *picks_seen.entry(pick).or_insert(0) += 1;
        decision_stream.push(pick as u8);
        black_box((pick, &options, &state));
    }
    println!("G1a planted correctness: {planted_hits}/{SETS} (bar: {SETS}/{SETS})");
    assert_eq!(
        planted_hits, SETS,
        "a planted perfect match must win at every rotating index"
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
    println!("G2 latency per decision SET (build + pick + score_into, --release):");
    for &k in &OPTION_WIDTHS {
        let state: [f32; D] = rng.vector();
        let mut options = [[0.0f32; D]; 34];
        for o in options[..k].iter_mut() {
            *o = rng.vector();
        }
        let mut sink = 0usize;
        let mut acc = 0.0f32;
        let run = |sink: &mut usize, acc: &mut f32| {
            let (pick, score) = match k {
                9 => run_set::<9>(&options, &state),
                17 => run_set::<17>(&options, &state),
                _ => run_set::<34>(&options, &state),
            };
            *sink = (*sink + pick) % 34;
            *acc += score;
        };
        for _ in 0..WARMUP {
            run(&mut sink, &mut acc); // first-touch / allocator settle
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

    // ── Determinism: double build + BLAKE3 digests ──────────────────────
    let corpus_arr: [[f32; D]; 34] = core::array::from_fn(|_| rng.vector());
    let t1 = CentroidTable::<D, 34>::new(&corpus_arr);
    let t2 = CentroidTable::<D, 34>::new(&corpus_arr);
    let bytes = |t: &CentroidTable<D, 34>| -> Vec<u8> {
        t.rows()
            .iter()
            .flat_map(|r| r.iter().flat_map(|f| f.to_le_bytes()))
            .collect()
    };
    let (b1, b2) = (bytes(&t1), bytes(&t2));
    assert_eq!(b1, b2, "same corpus → bit-identical table");
    println!("determinism: double-build bit-identical ✓");
    println!("  table blake3     : {}", blake3::hash(&b1));
    println!(
        "  decisions blake3 : {}  (two-box anchors — record both)",
        blake3::hash(&decision_stream)
    );
    println!("G4: alloc gate = tests/state_option_scoring_alloc_check.rs (separate binary)");
    println!("G1 agreement vs the T0b oracle triples: examples/tetris_02_option_arena.rs (root)");
}
