//! Plan 607 T3 — the corpus-fitted head over frozen features, replayed
//! against the T0b oracle fixture (the `tetris_02_option_arena` reading's
//! designated lever).
//!
//! Bench 876's first reading: the untuned sentence-cosine scorer
//! discriminates but is not accurate (10.8%, tying constant-pick). This
//! arena is the evidence-directed T3 answer: fit a `FittedHead` over the
//! frozen per-option features the fixture already carries (holes,
//! bumpiness, Dellacherie-class columns) to imitate the oracle's per-option
//! `p_clean`, then read the SAME G1 gate shape on the SAME states —
//! in-corpus (the owner's "80–90% corpus-viable" hypothesis) AND
//! leave-one-state-out (the honest generalization reading; sibling options
//! of one state never leak into that state's fit).
//!
//! Determinism (Plan 607 T3's line): fixed λ grid, fixed fit recipe
//! (closed-form ridge via `linalg::ridge_solve`'s f64 path — no RNG, no
//! iterations, no gradient descent), head weights bit-identical across
//! runs/boxes (BLAKE3-digested beside the decision stream). λ is selected
//! by state-level LOO MSE over the pinned grid — corpus-only information,
//! never the agreement number.
//!
//! Gates: G2/G4/determinism formal rows live in katgpt-core's
//! `bench_878_state_option_head_goat` + `state_option_head_alloc_check`;
//! this arena is the fixture-replay AGREEMENT half (the bench_876 split).

#[path = "common/tetris_fixture.rs"]
mod tetris_fixture;

use std::path::PathBuf;

use katgpt_core::state_option_scoring::head::{FittedHead, HeadFitter};
use tetris_fixture::{
    Piece, Recomputed, default_fixture, dellacherie_pick, fixture_rule, load_fixture_states, outcome_features,
    play_game,
};

/// Design width: 11 frozen features (standardized) + intercept.
const D: usize = 12;
/// The 11 feature columns in fixture order (this ORDER is part of the
/// committed recipe — the head weights are only meaningful against it).
const FEATURE_COLS: usize = 11;
/// Pinned λ grid (standardized scale). Selection criterion: state-level
/// LOO MSE — corpus-only, deterministic, never the agreement number.
const RIDGE_GRID: [f64; 4] = [1e-3, 1e-2, 1e-1, 1.0];

fn feature_row(f: &tetris_fixture::FixtureFeatures) -> [f64; FEATURE_COLS] {
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
}

/// A live option's design row: corpus-standardized features + intercept.
/// `stats` are the CORPUS stats (fit time), frozen into the decision path.
fn design_row(
    raw: &[f64; FEATURE_COLS],
    mean: &[f64; FEATURE_COLS],
    inv_std: &[f64; FEATURE_COLS],
) -> [f64; D] {
    let mut row = [0.0f64; D];
    for i in 0..FEATURE_COLS {
        row[i] = (raw[i] - mean[i]) * inv_std[i];
    }
    row[FEATURE_COLS] = 1.0; // intercept
    row
}

fn pct(n: usize, d: usize) -> String {
    format!("{:.1}%", 100.0 * n as f64 / d as f64)
}

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

    println!("== Plan 607 T3 — the corpus-fitted head (Tetris fixture replay) ==");
    println!("fixture: {}", fixture_path.display());

    // ── Parse + drift-check (the shared detector) ────────────────────────
    let states = load_fixture_states(&fixture_path);
    let rule = fixture_rule(&states);
    let n_options: usize = states.iter().map(|(f, _)| f.options.len()).sum();
    println!(
        "drift check: PASS — {} states / {n_options} options recompute byte-identically",
        states.len()
    );

    // ── Build the corpus ─────────────────────────────────────────────────
    // Rows in corpus order: state 0's options, state 1's options, ...
    // `row_offset[s]` is where state s's options start; y = p_clean (the
    // oracle's own per-option read — the thing the T1 scorer could not
    // recover from sentence overlap alone).
    let mut corpus: Vec<[f64; D]> = Vec::with_capacity(n_options);
    let mut targets: Vec<f64> = Vec::with_capacity(n_options);
    let mut row_offsets: Vec<usize> = Vec::with_capacity(states.len() + 1);
    let mut raw_rows: Vec<[f64; FEATURE_COLS]> = Vec::with_capacity(n_options);
    for (f, _) in &states {
        row_offsets.push(corpus.len());
        for o in &f.options {
            let p = o.p_clean.unwrap_or_else(|| {
                panic!(
                    "{}: option p_clean missing — the T3 corpus needs the oracle's per-option read",
                    f.state_id
                )
            });
            let raw = feature_row(&o.features);
            raw_rows.push(raw);
            corpus.push([0.0; D]); // standardized in place, below
            targets.push(p);
        }
    }
    row_offsets.push(corpus.len());

    // Standardization stats (corpus mean/std, fixed order — part of the
    // committed recipe; std == 0 → the column carries no signal, map to 0).
    let mut mean = [0.0f64; FEATURE_COLS];
    for row in &raw_rows {
        for (m, x) in mean.iter_mut().zip(row.iter()) {
            *m += x;
        }
    }
    for m in mean.iter_mut() {
        *m /= raw_rows.len() as f64;
    }
    let mut var = [0.0f64; FEATURE_COLS];
    for row in &raw_rows {
        for (v, (x, m)) in var.iter_mut().zip(row.iter().zip(mean.iter())) {
            *v += (x - m) * (x - m);
        }
    }
    let mut inv_std = [0.0f64; FEATURE_COLS];
    for i in 0..FEATURE_COLS {
        let s = (var[i] / raw_rows.len() as f64).sqrt();
        inv_std[i] = if s > 0.0 { 1.0 / s } else { 0.0 };
    }
    for (dst, raw) in corpus.iter_mut().zip(raw_rows.iter()) {
        *dst = design_row(raw, &mean, &inv_std);
    }

    // ── λ selection + the LOO reading (one pass per grid point) ──────────
    // Leave ONE STATE out: fit on the other states' options, predict the
    // held-out state's options, argmax = that state's LOO decision. Sibling
    // options never leak. MSE selects λ; agreement is REPORTED at the
    // chosen λ (selection never sees the agreement number).
    println!(
        "\nλ selection (state-level LOO MSE over the pinned grid; agreement reported, never selected):"
    );
    let mut fitter = HeadFitter::<D>::new();
    let mut chosen = RIDGE_GRID[0];
    let mut chosen_mse = f64::INFINITY;
    let mut chosen_loo_picks = vec![0usize; states.len()];
    for &lam in &RIDGE_GRID {
        let mut sq = 0.0f64;
        let mut agree = 0usize;
        let mut picks = vec![0usize; states.len()];
        for (s, (f, _)) in states.iter().enumerate() {
            let (a, b) = (row_offsets[s], row_offsets[s + 1]);
            let mut train: Vec<[f64; D]> = Vec::with_capacity(corpus.len() - (b - a));
            train.extend_from_slice(&corpus[..a]);
            train.extend_from_slice(&corpus[b..]);
            let mut ty = Vec::with_capacity(targets.len() - (b - a));
            ty.extend_from_slice(&targets[..a]);
            ty.extend_from_slice(&targets[b..]);
            let head = fitter.fit_into(&train, &ty, lam);
            let mut best_pred = f64::NEG_INFINITY;
            let mut best_idx = 0usize;
            for (j, row) in corpus[a..b].iter().enumerate() {
                let p = head.score(row);
                sq += (p - targets[a + j]) * (p - targets[a + j]);
                if p > best_pred {
                    best_pred = p;
                    best_idx = j;
                }
            }
            picks[s] = best_idx;
            if best_idx == f.argmax {
                agree += 1;
            }
        }
        let mse = sq / targets.len() as f64;
        println!(
            "  λ={lam:<5.3}  LOO MSE {mse:.6}  LOO agreement {agree}/{} ({})",
            states.len(),
            pct(agree, states.len())
        );
        if mse < chosen_mse {
            chosen_mse = mse;
            chosen = lam;
            chosen_loo_picks = picks;
        }
    }
    println!("  chosen λ = {chosen} (lowest LOO MSE)");

    // ── In-corpus reading (fit ALL rows, read ALL states — the
    // "corpus-viable" number, explicitly NOT a generalization claim) ─────
    let head = fitter.fit_into(&corpus, &targets, chosen);
    let mut in_agree = 0usize;
    let mut in_class = 0usize;
    let mut dellacherie_agree = 0usize;
    let mut picks_seen = std::collections::HashSet::new();
    let mut index_hist = std::collections::HashMap::<usize, usize>::new();
    let mut chance_sum = 0.0f64;
    let mut loo_agree = 0usize;
    let mut loo_distinct = std::collections::HashSet::new();
    for (s, (f, re)) in states.iter().enumerate() {
        let k = f.options.len();
        let (a, b) = (row_offsets[s], row_offsets[s + 1]);
        let pick = head.pick(&corpus[a..b], k);
        picks_seen.insert(pick);
        *index_hist.entry(f.argmax).or_insert(0) += 1;
        chance_sum += 1.0 / k as f64;
        if pick == f.argmax {
            in_agree += 1;
        }
        if re.spot_sentences[pick] == re.spot_sentences[f.argmax] {
            in_class += 1;
        }
        if pick == dellacherie_pick(re) {
            dellacherie_agree += 1;
        }
        let loo_pick = chosen_loo_picks[s];
        loo_distinct.insert(loo_pick);
        if loo_pick == f.argmax {
            loo_agree += 1;
        }
    }
    let n = states.len();
    let constant_index = *index_hist.iter().max_by_key(|(_, c)| **c).unwrap().0;
    let constant_agree = index_hist[&constant_index];
    let chance = chance_sum / n as f64;

    println!("\nG1 decision agreement (T3 corpus-fitted head vs laya oracle):");
    println!(
        "  in-corpus raw agreement: {in_agree}/{n} ({})   [fit on ALL states — the corpus-viable reading]",
        pct(in_agree, n)
    );
    println!(
        "  LOO raw agreement:       {loo_agree}/{n} ({})   [fit per held-out state — the generalization reading]",
        pct(loo_agree, n)
    );
    println!(
        "  in-corpus class-level:   {in_class}/{n} ({})  [same-sentence equivalence]",
        pct(in_class, n)
    );
    println!(
        "  constant-pick baseline:  {constant_agree}/{n} ({})  [always oracle-majority index {constant_index}]",
        pct(constant_agree, n)
    );
    println!("  chance baseline:         {:.1}%", 100.0 * chance);
    println!(
        "  vs Dellacherie argmax:   {dellacherie_agree}/{n} ({})  [context]",
        pct(dellacherie_agree, n)
    );
    println!(
        "  distinct picks: in-corpus {} / LOO {} over {n} states [floor ≥ 2] {}",
        picks_seen.len(),
        loo_distinct.len(),
        if picks_seen.len() >= 2 && loo_distinct.len() >= 2 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let g1_in = in_agree > constant_agree && (in_agree as f64) > chance * n as f64;
    let g1_loo = loo_agree > constant_agree && (loo_agree as f64) > chance * n as f64;
    println!(
        "  G1 verdict: in-corpus {} · LOO {} (raw > constant-pick AND raw > chance)",
        if g1_in { "HOLDS" } else { "DOES NOT HOLD" },
        if g1_loo { "HOLDS" } else { "DOES NOT HOLD" }
    );

    // ── Determinism: double fit bit-identical + BLAKE3 anchors ───────────
    let head2 = fitter.fit_into(&corpus, &targets, chosen);
    let digest = |h: &FittedHead<D>| {
        let mut bytes = Vec::with_capacity(D * 8);
        for w in h.weights() {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        blake3::hash(&bytes)
    };
    let (d1, d2) = (digest(&head), digest(&head2));
    println!(
        "\ndeterminism: double-fit {} (head blake3 {d1})",
        if d1 == d2 {
            "bit-identical ✓"
        } else {
            "DIVERGED ✗"
        }
    );
    assert_eq!(
        d1, d2,
        "same corpus → bit-identical head (Plan 607 T3's line)"
    );
    let pass_digest = || -> blake3::Hash {
        let mut stream: Vec<u8> = Vec::new();
        for (s, (f, _)) in states.iter().enumerate() {
            let (a, b) = (row_offsets[s], row_offsets[s + 1]);
            stream.push(head.pick(&corpus[a..b], f.options.len()) as u8);
        }
        blake3::hash(&stream)
    };
    let p1 = pass_digest();
    let p2 = pass_digest();
    println!(
        "decisions: two passes {} (blake3 {p1})",
        if p1 == p2 {
            "byte-identical ✓"
        } else {
            "DIVERGED ✗"
        }
    );
    assert_eq!(p1, p2, "same corpus → bit-identical decisions");

    // ── Latency context (formal G2 = katgpt-core bench_878) ─────────────
    let mut samples: Vec<u128> = Vec::with_capacity(states.len() * 25);
    for _ in 0..25 {
        for (s, (f, _)) in states.iter().enumerate() {
            let (a, b) = (row_offsets[s], row_offsets[s + 1]);
            let t = std::time::Instant::now();
            let pick = head.pick(&corpus[a..b], f.options.len());
            samples.push(t.elapsed().as_nanos());
            std::hint::black_box(pick);
        }
    }
    samples.sort_unstable();
    let (p50, sup50) = katgpt_core::stats::nearest_rank(&samples, 0.50);
    let (p99, sup99) = katgpt_core::stats::nearest_rank(&samples, 0.99);
    println!(
        "\nlatency context (head pick per decision SET; formal bar = bench_878): p50 {p50} ns | p99 {p99} ns (n={}, K∈{{9,17,34}}, D={D}; tail support {sup50}/{sup99})",
        samples.len()
    );

    // ── Lines-cleared games: Dellacherie vs the fitted-head policy ──────
    println!(
        "\nlines cleared ({games} seeded games/policy, shared piece streams, cap {placements_cap} placements):"
    );
    let mut rng = fastrand::Rng::with_seed(seed);
    let streams: Vec<Vec<Piece>> = (0..games)
        .map(|_| {
            (0..placements_cap)
                .map(|_| Piece::ALL[rng.usize(0..7)])
                .collect()
        })
        .collect();
    let head_policy = |re: &Recomputed| -> usize {
        let mut rows: Vec<[f64; D]> = Vec::with_capacity(re.options.len());
        for p in &re.options {
            let raw = feature_row_raw(&re.board, p);
            rows.push(design_row(&raw, &mean, &inv_std));
        }
        head.pick(&rows, re.options.len())
    };
    for (name, policy) in [
        (
            "dellacherie",
            &dellacherie_pick as &dyn Fn(&Recomputed) -> usize,
        ),
        (
            "t3_fitted_head",
            &head_policy as &dyn Fn(&Recomputed) -> usize,
        ),
    ] {
        let mut total = 0u32;
        let mut per_game: Vec<u32> = Vec::with_capacity(games);
        let mut placements = 0usize;
        for stream in &streams {
            let (c, npl) = play_game(policy, stream, rule);
            total += c;
            placements += npl;
            per_game.push(c);
        }
        per_game.sort_unstable();
        let mean_lines = total as f64 / games as f64;
        let median = per_game[games / 2];
        let max = per_game[games - 1];
        println!(
            "  {name:>15}: total {total:>4} | mean {mean_lines:6.2} | median {median} | max {max} | placements {placements}"
        );
    }

    println!(
        "\nGate pointers: G2 = katgpt-core bench_878_state_option_head_goat · G4 = katgpt-core state_option_head_alloc_check · this run's agreement = the .benchmarks/878 G1 row"
    );
}

/// Live-game feature extraction — the SAME columns in the SAME order as the
/// fixture corpus (a decision path that drifted from the fit path would
/// standardize against nothing).
fn feature_row_raw(
    board: &tetris_fixture::Board,
    p: &tetris_fixture::tetris_sim::Placement,
) -> [f64; FEATURE_COLS] {
    let f = outcome_features(board, p);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_states() -> Vec<(tetris_fixture::FixtureState, Recomputed)> {
        let states = load_fixture_states(&default_fixture());
        assert_eq!(states.len(), 120, "fixture carries 120 states");
        states
    }

    #[test]
    fn fixture_recomputes_byte_identically() {
        load_states();
    }

    #[test]
    fn head_fit_is_bit_deterministic_on_the_corpus() {
        let states = load_states();
        let mut corpus = Vec::new();
        let mut targets = Vec::new();
        for (f, _) in &states {
            for o in &f.options {
                let mut row = [0.0; D];
                let raw = feature_row(&o.features);
                row[..FEATURE_COLS].copy_from_slice(&raw);
                row[FEATURE_COLS] = 1.0;
                corpus.push(row);
                targets.push(o.p_clean.expect("p_clean"));
            }
        }
        let mut fitter = HeadFitter::<D>::new();
        let a = fitter.fit_into(&corpus, &targets, 1e-2);
        let b = fitter.fit_into(&corpus, &targets, 1e-2);
        let ba: Vec<u8> = a.weights().iter().flat_map(|f| f.to_le_bytes()).collect();
        let bb: Vec<u8> = b.weights().iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(ba, bb, "same corpus → bit-identical head (T3's line)");
    }

    #[test]
    fn discrimination_floor_holds() {
        let states = load_states();
        let mut fitter = HeadFitter::<D>::new();
        // tiny self-contained corpus fit (features + intercept, no
        // standardization — the floor is about DISTINCT PICKS, not
        // accuracy)
        let mut corpus = Vec::new();
        let mut targets = Vec::new();
        for (f, _) in &states {
            for o in &f.options {
                let mut row = [0.0; D];
                let raw = feature_row(&o.features);
                row[..FEATURE_COLS].copy_from_slice(&raw);
                row[FEATURE_COLS] = 1.0;
                corpus.push(row);
                targets.push(o.p_clean.expect("p_clean"));
            }
        }
        let head = fitter.fit_into(&corpus, &targets, 1e-2);
        let mut offset = 0usize;
        let mut picks = std::collections::HashSet::new();
        for (f, _) in &states {
            let k = f.options.len();
            picks.insert(head.pick(&corpus[offset..offset + k], k));
            offset += k;
        }
        assert!(
            picks.len() >= 2,
            "reflex discrimination floor: distinct picks ≥ 2 over distinct states, got {picks:?}"
        );
    }
}
