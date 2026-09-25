//! Tetris decision arena — katgpt-rs Plan 607 T4, the modelless
//! game-decision lane's FIRST GOAT READING.
//!
//! Replays the T0b oracle fixture (`tests/fixtures/
//! tetris_oracle_laya_en_v2.jsonl`): the code recomputes every state's
//! landing options, outcome features and grammar sentences from the
//! board+piece (the drift detector — any mismatch with the fixture is a
//! loud exit, the katgpt-device-verify rule), embeds the state + option
//! sentences with the arena's deterministic trigram embedder, scores every
//! landing spot with the T1 primitive (`CentroidTable::pick`), and reads
//! agreement against laya's recorded decisions.
//!
//! The reading (Plan 607's G1 form — never vs laya alone, R2):
//!
//! - **raw argmax agreement** — our pick == the oracle's pinned argmax;
//! - **class-level agreement** — our pick's SENTENCE == the oracle pick's
//!   sentence (the fixture's 33 same-sentence ties are honest grammar
//!   equivalences on symmetric boards; disagreeing with a tied twin is not
//!   a decision disagreement);
//! - **constant-pick baseline** — the majority-class oracle index, the
//!   strongest static policy (reflex T7 measured 4-of-5 families
//!   constant-picking — this baseline is the honest bar);
//! - **chance baseline** — mean 1/K over states (uniform random picker vs
//!   the pinned index);
//! - **discrimination floor** — distinct picks ≥ 2 over the 120 states;
//! - **context** — agreement of our picks with the Dellacherie argmax, and
//!   seeded lines-cleared games: Dellacherie policy vs the T1 sentence
//!   policy on SHARED piece streams.
//!
//! The scorer is measured UNTUNED (Plan 607: no embedder/scale tuning
//! against the oracle in this unit — the first reading must measure the
//! honest scorer; scale is the substrate's route scale, and the argmax is
//! scale-invariant for scale > 0 anyway).
//!
//! Run:  cargo run --release --features state_option_scoring --example tetris_02_option_arena
//! Tests: cargo test  --features state_option_scoring --example tetris_02_option_arena
//!
//! Latency: this example prints per-decision-set context timings; the
//! FORMAL G2 bar lives in katgpt-core's
//! `bench_876_state_option_scoring_goat`, and the G4 alloc gate in
//! katgpt-core's `state_option_scoring_alloc_check`.

use std::path::PathBuf;

#[path = "common/tetris_fixture.rs"]
mod tetris_fixture;

use katgpt_core::state_option_scoring::CentroidTable;
use tetris_fixture::{
    EMBED_DIM, FixtureState, Piece, Recomputed, default_fixture, dellacherie_pick, embed,
    fixture_rule, load_fixture_states, play_game,
};

/// The substrate's route scale (riir-reflex `ROUTE_SCALE`). The argmax is
/// scale-invariant for scale > 0 (exact_sigmoid is strictly monotone); the
/// arena only argmaxes.
/// The substrate's route scale (riir-reflex `ROUTE_SCALE`), passed to
/// `score_into` when scores (not just the argmax) are wanted. The arena
/// only argmaxes, and the argmax is scale-INVARIANT for scale > 0 (the
/// exact sigmoid is strictly monotone) — so no score path here consumes
/// it; the formal scored bar lives in katgpt-core's bench_876.
#[allow(dead_code)]
const SCALE: f32 = 8.0;

// ── T1 scoring over embedded sentences ─────────────────────────────

/// The fixture's decision sets are exactly {9, 17, 34} wide, but LIVE
/// games reach any width (blocked columns shrink a piece's landing set),
/// so the arena always pads to the widest table: zero rows can never win
/// the argmax — embeddings are non-negative count vectors, so cosine ≥ 0,
/// a zero row sits at the floor, and any tie AT the floor breaks to the
/// lowest index, which is always a real row (padded rows live at indices
/// ≥ k). Picking is therefore identical to an exact-k table on fixture
/// states.
fn t1_pick(state_vec: &[f32; EMBED_DIM], vecs: &[[f32; EMBED_DIM]; 34], k: usize) -> usize {
    assert!(k <= 34, "option width {k} overflows the arena table");
    let mut mat = [[0.0f32; EMBED_DIM]; 34];
    for (dst, src) in mat.iter_mut().zip(vecs.iter().take(k)) {
        *dst = *src;
    }
    CentroidTable::<EMBED_DIM, 34>::new(&mat).pick(state_vec)
}

/// The arena's decision: embed the state + spot sentences, score with T1.
fn decide(re: &Recomputed) -> usize {
    let state_vec = embed(&re.state_sentence);
    let mut vecs = [[0.0f32; EMBED_DIM]; 34];
    for (v, s) in vecs.iter_mut().zip(&re.spot_sentences) {
        *v = embed(s);
    }
    t1_pick(&state_vec, &vecs, re.options.len())
}

// ── Seeded lines-cleared games (shared piece streams) ────────────────────
// play_game + dellacherie_pick live in the shared `tetris_fixture` module —
// the T3 fitted-head arena consumes the identical loop and policy.

// ── The reading ──────────────────────────────────────────────────────────

struct Reading {
    n_states: usize,
    raw_agree: usize,
    class_agree: usize,
    constant_agree: usize,
    constant_index: usize,
    chance: f64,
    distinct_picks: usize,
    dellacherie_agree: usize,
    oracle_ties: usize,
    our_ties_at_oracle: usize,
}

fn read_reading(states: &[(FixtureState, Recomputed)]) -> Reading {
    let n_states = states.len();

    // Our decisions + baselines.
    let mut raw_agree = 0usize;
    let mut class_agree = 0usize;
    let mut dellacherie_agree = 0usize;
    let mut picks_seen = std::collections::HashSet::new();
    let mut index_hist = std::collections::HashMap::<usize, usize>::new();
    let mut chance_sum = 0.0f64;
    let mut oracle_ties = 0usize;
    let mut our_ties_at_oracle = 0usize;

    for (f, re) in states {
        let pick = decide(re);
        let k = f.options.len();
        picks_seen.insert(pick);
        *index_hist.entry(f.argmax).or_insert(0) += 1;
        chance_sum += 1.0 / k as f64;
        if pick == f.argmax {
            raw_agree += 1;
        }
        if re.spot_sentences[pick] == re.spot_sentences[f.argmax] {
            class_agree += 1;
        }
        if pick == dellacherie_pick(re) {
            dellacherie_agree += 1;
        }
        // Oracle-side honesty stats (fixture README pins 33 same-sentence
        // ties at the oracle's max p_clean). p_clean lives at the OPTION
        // level in the joined fixture.
        if f.options.iter().all(|o| o.p_clean.is_some()) {
            let pcs: Vec<f64> = f.options.iter().filter_map(|o| o.p_clean).collect();
            let max = pcs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let tied = pcs.iter().filter(|&&p| p == max).count();
            if tied > 1 {
                oracle_ties += 1;
                if re.spot_sentences[pick] == re.spot_sentences[f.argmax] {
                    our_ties_at_oracle += 1;
                }
            }
        }
    }

    // Strongest constant policy: always the majority oracle index. Ties go
    // to the LOWEST index — `index_hist` is a HashMap, so an unordered
    // max_by_key printed 1 or 16 run to run on the v3 fixture (13/120 each).
    let constant_index = *index_hist
        .iter()
        .max_by_key(|&(i, c)| (*c, std::cmp::Reverse(*i)))
        .unwrap()
        .0;
    let constant_agree = index_hist[&constant_index];

    Reading {
        n_states,
        raw_agree,
        class_agree,
        constant_agree,
        constant_index,
        chance: chance_sum / n_states as f64,
        distinct_picks: picks_seen.len(),
        dellacherie_agree,
        oracle_ties,
        our_ties_at_oracle,
    }
}

fn pct(n: usize, d: usize) -> String {
    format!("{:.1}%", 100.0 * n as f64 / d as f64)
}

// ── main ─────────────────────────────────────────────────────────────────

fn main() {
    let mut fixture_path = default_fixture();
    let mut games = 8usize;
    let mut seed = 607u64;
    let placements_cap = 500usize;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--fixture" if i + 1 < args.len() => {
                i += 1;
                fixture_path = PathBuf::from(&args[i]);
            }
            "--games" if i + 1 < args.len() => {
                i += 1;
                games = args[i].parse().expect("--games <n>");
            }
            "--seed" if i + 1 < args.len() => {
                i += 1;
                seed = args[i].parse().expect("--seed <u64>");
            }
            other => {
                eprintln!(
                    "Unknown arg: {other}. Usage: [--fixture <path>] [--games <n>] [--seed <u64>]"
                );
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("== Plan 607 T4 — Tetris decision arena (first GOAT reading) ==");
    println!("fixture: {}", fixture_path.display());

    // ── Parse + drift-check ──────────────────────────────────────────
    let states = load_fixture_states(&fixture_path);
    let rule = fixture_rule(&states);
    let n_options: usize = states.iter().map(|(f, _)| f.options.len()).sum();
    println!(
        "drift check: PASS — {} states / {n_options} options recompute byte-identically",
        states.len()
    );

    // ── The G1 reading ───────────────────────────────────────────────────
    let r = read_reading(&states);
    println!("\nG1 decision agreement (T1 sentence-cosine vs laya oracle, UNTUNED):");
    println!(
        "  raw argmax agreement:    {}/{} ({})",
        r.raw_agree,
        r.n_states,
        pct(r.raw_agree, r.n_states)
    );
    println!(
        "  class-level agreement:   {}/{} ({})  [same-sentence equivalence, oracle ties {}/{} states — our pick lands on an oracle-tied twin in {} of them]",
        r.class_agree,
        r.n_states,
        pct(r.class_agree, r.n_states),
        r.oracle_ties,
        r.n_states,
        r.our_ties_at_oracle
    );
    println!(
        "  constant-pick baseline:  {}/{} ({})  [always oracle-majority index {}]",
        r.constant_agree,
        r.n_states,
        pct(r.constant_agree, r.n_states),
        r.constant_index
    );
    println!("  chance baseline:         {:.1}%", 100.0 * r.chance);
    println!(
        "  vs Dellacherie argmax:   {}/{} ({})  [context]",
        r.dellacherie_agree,
        r.n_states,
        pct(r.dellacherie_agree, r.n_states)
    );
    println!(
        "  distinct picks: {} of {} states [floor ≥ 2] {}",
        r.distinct_picks,
        r.n_states,
        if r.distinct_picks >= 2 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let g1_holds =
        r.raw_agree > r.constant_agree && (r.raw_agree as f64) > r.chance * r.n_states as f64;
    println!(
        "  G1 verdict: {} (raw > constant-pick AND raw > chance — the reading that gates {{T2, T3}})",
        if g1_holds { "HOLDS" } else { "DOES NOT HOLD" }
    );

    // ── Latency context (formal G2 = katgpt-core bench_876) ─────────────
    let mut samples: Vec<u128> = Vec::with_capacity(states.len() * 25);
    for _ in 0..25 {
        for (_, re) in &states {
            let t = std::time::Instant::now();
            let pick = decide(re);
            samples.push(t.elapsed().as_nanos());
            std::hint::black_box(pick);
        }
    }
    samples.sort_unstable();
    let (p50, _) = katgpt_core::stats::nearest_rank(&samples, 0.50);
    let (p99, _) = katgpt_core::stats::nearest_rank(&samples, 0.99);
    println!(
        "\nlatency context (embed + build + pick per state; formal bar = bench_876): p50 {p50} ns | p99 {p99} ns (n={}, K∈{{9,17,34}})",
        samples.len()
    );

    // ── Determinism: two full passes, byte-identical ─────────────────────
    let pass_digest = || -> blake3::Hash {
        let mut stream: Vec<u8> = Vec::new();
        for (f, re) in states.iter() {
            let state_vec = embed(&re.state_sentence);
            let mut vecs = [[0.0f32; EMBED_DIM]; 34];
            for (v, s) in vecs.iter_mut().zip(&re.spot_sentences) {
                *v = embed(s);
            }
            let k = f.options.len();
            // digest the TABLE bytes + the decision — the determinism row
            let mut mat = [[0.0f32; EMBED_DIM]; 34];
            mat.copy_from_slice(&vecs);
            for row in mat.iter().take(k) {
                for x in row.iter() {
                    stream.extend_from_slice(&x.to_le_bytes());
                }
            }
            stream.push(t1_pick(&state_vec, &vecs, k) as u8);
        }
        blake3::hash(&stream)
    };
    let d1 = pass_digest();
    let d2 = pass_digest();
    println!(
        "determinism: two passes {} (blake3 {d1})",
        if d1 == d2 {
            "byte-identical ✓"
        } else {
            "DIVERGED ✗"
        }
    );
    assert_eq!(d1, d2, "same corpus → bit-identical scoring state");

    // ── Lines-cleared games: Dellacherie vs the T1 sentence policy ──────
    println!(
        "\nlines cleared ({games} seeded games/policy, shared piece streams, cap {placements_cap} placements):"
    );
    // One rng generates ALL games' streams up front; both policies consume
    // the SAME streams (a fair paired comparison).
    let mut rng = fastrand::Rng::with_seed(seed);
    let streams: Vec<Vec<Piece>> = (0..games)
        .map(|_| {
            (0..placements_cap)
                .map(|_| Piece::ALL[rng.usize(0..7)])
                .collect()
        })
        .collect();
    for (name, policy) in [
        (
            "dellacherie",
            &dellacherie_pick as &dyn Fn(&Recomputed) -> usize,
        ),
        ("t1_sentence", &(decide as fn(&Recomputed) -> usize)),
    ] {
        let mut total = 0u32;
        let mut per_game: Vec<u32> = Vec::with_capacity(games);
        let mut placements = 0usize;
        for stream in &streams {
            let (c, n) = play_game(policy, stream, rule);
            total += c;
            placements += n;
            per_game.push(c);
        }
        per_game.sort_unstable();
        let mean = total as f64 / games as f64;
        let median = per_game[games / 2];
        let max = per_game[games - 1];
        println!(
            "  {name:>12}: total {total:>4} | mean {mean:6.2} | median {median} | max {max} | placements {placements}"
        );
    }

    println!(
        "\nGate pointers: G2 = katgpt-core bench_876_state_option_scoring_goat · G4 = katgpt-core state_option_scoring_alloc_check · this run's agreement = the .benchmarks/876 G1 row"
    );
}

// ── Tests (the drift detector + determinism as executing lanes) ──────────

#[cfg(test)]
mod tests {
    use super::*;

    fn load_states() -> Vec<(FixtureState, Recomputed)> {
        let states = load_fixture_states(&default_fixture());
        assert_eq!(states.len(), 120, "fixture carries 120 states");
        states
    }

    #[test]
    fn fixture_recomputes_byte_identically() {
        load_states();
    }

    #[test]
    fn decisions_are_deterministic_across_passes() {
        let states = load_states();
        let run = || -> Vec<usize> { states.iter().map(|(_, re)| decide(re)).collect() };
        assert_eq!(run(), run(), "two full passes must decide identically");
    }

    #[test]
    fn discrimination_floor_holds() {
        let states = load_states();
        let picks: std::collections::HashSet<usize> =
            states.iter().map(|(_, re)| decide(re)).collect();
        assert!(
            picks.len() >= 2,
            "reflex discrimination floor: distinct picks ≥ 2 over distinct states, got {picks:?}"
        );
    }
}
