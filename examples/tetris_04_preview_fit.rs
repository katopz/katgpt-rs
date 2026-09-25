//! Plan 609 T1.7 + T1.9 — the v4 preview go/no-go and the paired-corpus
//! head fits.
//!
//! Loads the v4 fixture (arm-B oracle labels over the 840 paired states:
//! 120 v3 boards × all 7 next pieces) and the v3 baseline (the
//! preview-blind argmaxes), then reports:
//!
//! - **T1.7 the go/no-go** — the flip fraction: boards whose arm-B oracle
//!   argmax is NOT identical across the 7 preview variants. ≈ 0 → the
//!   preview ships in the sentence only (fidelity) and the
//!   crossed-head/consumer/G1 work STOPS (the plan's abort path). The raw
//!   MI-style readings are NOT decision-reliable; the flip fraction is.
//! - **T1.9 the head fits** — the spot-only head (5 decoded spot fills +
//!   intercept: the G1 preview-blind comparator, which ranks all 7 variants
//!   of a board identically by construction) vs the crossed head (next-piece
//!   one-hot × spot interactions, 41 raw columns), under the BOARD-GROUPED
//!   holdout (`katgpt-core` `head::loo_group_select`: all 7 preview states
//!   of a board hold out together). Metrics: argmax agreement, pairwise
//!   concordance within a state, board-centered MSE.
//!
//! Re-run:
//! ```text
//! cargo run --release --features state_option_scoring,template_decode --example tetris_04_preview_fit
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

use katgpt_core::state_option_scoring::head::{HeadFitter, loo_group_select};
use micro_fit::RIDGE_GRID;

use tetris_sim::{Board, OutcomeFeatures, Placement, Piece};

const F_SPOT: usize = 5;
const D_SPOT: usize = F_SPOT + 1; // + intercept
/// spot(5) + next one-hot(6, baseline I) + one-hot×spot crossed(30).
const F_CROSS: usize = 5 + 6 + 30;
const D_CROSS: usize = F_CROSS + 1;

// ── Fixture loading ──────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct V4FixtureState {
    state_id: String,
    state_sentence: String,
    board: Vec<String>,
    piece: String,
    next_piece: String,
    options: Vec<V4FixtureOption>,
    argmax: usize,
}

#[derive(serde::Deserialize)]
struct V4FixtureOption {
    rot: usize,
    col: usize,
    row: usize,
    cells: Vec<(usize, usize)>,
    sentence: String,
    #[serde(default)]
    p_clean: Option<f64>,
}

struct V4Loaded {
    states: Vec<V4FixtureState>,
    boards: Vec<Board>,
    options: Vec<Vec<Placement>>,
    features: Vec<Vec<OutcomeFeatures>>,
    /// parent state id ("arch:empty:I") per state, in state order.
    parents: Vec<String>,
}

/// Load + validate the v4 fixture: re-enumerate every state's placements
/// under the inherited FromTop drop rule and require the fixture's
/// rot/col/row/cells to match exactly (the drift detector).
fn load_v4(path: &std::path::Path) -> V4Loaded {
    let raw = std::fs::read_to_string(path).expect("read v4 fixture");
    let mut states = Vec::new();
    for (ln, line) in raw.lines().enumerate() {
        let v: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("fixture line {}: {e}", ln + 1));
        if v["state_id"] == "_meta" {
            continue;
        }
        let s: V4FixtureState =
            serde_json::from_value(v).unwrap_or_else(|e| panic!("fixture line {}: {e}", ln + 1));
        states.push(s);
    }
    let mut boards = Vec::with_capacity(states.len());
    let mut options = Vec::with_capacity(states.len());
    let mut features = Vec::with_capacity(states.len());
    let mut parents = Vec::with_capacity(states.len());
    for st in &states {
        let board = Board::from_strings(&st.board.iter().map(String::as_str).collect::<Vec<_>>());
        let piece = *Piece::ALL
            .iter()
            .find(|p| p.id() == st.piece)
            .unwrap_or_else(|| panic!("{}: unknown piece {}", st.state_id, st.piece));
        assert!(
            Piece::ALL.iter().any(|p| p.id() == st.next_piece),
            "{}: unknown next piece {}",
            st.state_id,
            st.next_piece
        );
        let opts = tetris_sim::landing_options_with(&board, piece, tetris_sim::DropRule::FromTop);
        assert_eq!(
            opts.len(),
            st.options.len(),
            "{}: option count drifted",
            st.state_id
        );
        let mut fs = Vec::with_capacity(opts.len());
        for (p, fo) in opts.iter().zip(&st.options) {
            assert!(
                p.rot == fo.rot && p.col == fo.col && p.row == fo.row && p.cells == fo.cells,
                "{}: placement drifted at rot {} col {}",
                st.state_id,
                fo.rot,
                fo.col
            );
            fs.push(tetris_sim::outcome_features(&board, p));
        }
        parents.push(
            st.state_id
                .split("|next:")
                .next()
                .expect("parent prefix")
                .to_owned(),
        );
        boards.push(board);
        options.push(opts);
        features.push(fs);
    }
    V4Loaded {
        states,
        boards,
        options,
        features,
        parents,
    }
}

/// The v3 baseline's argmax per state id (the preview-blind picks).
fn load_baseline_argmaxes(path: &std::path::Path) -> std::collections::HashMap<String, usize> {
    let raw = std::fs::read_to_string(path).expect("read v3 baseline fixture");
    let mut out = std::collections::HashMap::new();
    for line in raw.lines() {
        let v: serde_json::Value =
            serde_json::from_str(line).expect("parse baseline JSONL line");
        if v["state_id"] == "_meta" {
            continue;
        }
        out.insert(
            v["state_id"].as_str().expect("state_id").to_owned(),
            v["argmax"].as_u64().expect("argmax") as usize,
        );
    }
    out
}

/// Contiguous board groups: runs of 7 states sharing a parent. Returns the
/// group offsets over STATE space (for `loo_group_select`) and the group id
/// per state.
fn board_groups(parents: &[String]) -> (Vec<usize>, Vec<usize>) {
    let mut group_of = Vec::with_capacity(parents.len());
    let mut offsets = vec![0usize];
    let mut cur = parents[0].clone();
    for (i, p) in parents.iter().enumerate() {
        if *p != cur {
            cur = p.clone();
            offsets.push(i);
        }
        group_of.push(offsets.len() - 1);
    }
    offsets.push(parents.len());
    for w in offsets.windows(2) {
        assert_eq!(w[1] - w[0], 7, "every board group must hold all 7 previews");
    }
    (offsets, group_of)
}

fn state_offsets(option_counts: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize];
    for c in option_counts {
        out.push(out.last().unwrap() + c);
    }
    out
}

// ── The two heads ────────────────────────────────────────────────────────

struct HeadReading {
    name: &'static str,
    lam: f64,
    loo_agree: usize,
    n_states: usize,
    distinct: usize,
    concord: usize,
    pairs: usize,
    board_mse: f64,
    /// boards whose LOO picks differ across the 7 previews (0 for the
    /// spot-only head — its defining property, asserted).
    boards_flipped: usize,
    digest: blake3::Hash,
}

/// Fit one head over the paired corpus under the board-grouped holdout and
/// read the G1 metrics at the selected λ.
fn run_head<const F: usize, const D: usize>(
    name: &'static str,
    raws: &[[f64; F]],
    targets: &[f64],
    state_off: &[usize],
    group_off: &[usize],
    argmaxes: &[usize],
) -> HeadReading {
    assert_eq!(D, F + 1, "design = standardized features + intercept");
    let std = micro_fit::Standardizer::<F>::fit(raws);
    let rows: Vec<[f64; D]> = raws.iter().map(|r| std.design(r)).collect();
    let mut fitter = HeadFitter::<D>::new();
    let out = loo_group_select(
        &mut fitter,
        &rows,
        targets,
        state_off,
        group_off,
        argmaxes,
        &RIDGE_GRID,
    );
    let loo_agree = out
        .picks
        .iter()
        .zip(argmaxes.iter())
        .filter(|(p, a)| p == a)
        .count();
    let distinct = out.picks.iter().collect::<std::collections::HashSet<_>>().len();

    // Pairwise concordance within each state (head scores vs oracle
    // p_clean ordering; exact ties in either → the pair is skipped).
    let mut concord = 0usize;
    let mut pairs = 0usize;
    for s in 0..argmaxes.len() {
        let (a, b) = (state_off[s], state_off[s + 1]);
        for i in a..b {
            for j in i + 1..b {
                let dp = out.preds[i] - out.preds[j];
                let dy = targets[i] - targets[j];
                if dp * dy > 0.0 {
                    concord += 1;
                }
                if dp * dy != 0.0 {
                    pairs += 1;
                }
            }
        }
    }

    // Board-centered MSE: per-board demeaned prediction vs target.
    let mut sq = 0.0f64;
    for g in 0..group_off.len() - 1 {
        let (gs, ge) = (group_off[g], group_off[g + 1]);
        let (ra, rb) = (state_off[gs], state_off[ge]);
        let n = (rb - ra) as f64;
        let mp = out.preds[ra..rb].iter().sum::<f64>() / n;
        let mt = targets[ra..rb].iter().sum::<f64>() / n;
        for (p, &t) in out.preds[ra..rb].iter().zip(&targets[ra..rb]) {
            let e = (p - mp) - (t - mt);
            sq += e * e;
        }
    }
    let board_mse = sq / targets.len() as f64;

    // Within-board pick flips (the crossed head's preview sensitivity; the
    // spot-only head must show zero by construction).
    let mut boards_flipped = 0usize;
    for g in 0..group_off.len() - 1 {
        let (gs, ge) = (group_off[g], group_off[g + 1]);
        let first = out.picks[gs];
        if out.picks[gs..ge].iter().any(|&p| p != first) {
            boards_flipped += 1;
        }
    }

    // The in-corpus full fit at the chosen λ — its digest is the analysis
    // anchor (the LOO path never has one head).
    let head = fitter.fit_into(&rows, targets, out.lam);
    HeadReading {
        name,
        lam: out.lam,
        loo_agree,
        n_states: argmaxes.len(),
        distinct,
        concord,
        pairs,
        board_mse,
        boards_flipped,
        digest: micro_fit::head_digest(&head),
    }
}

fn pct(n: usize, d: usize) -> String {
    format!("{:.1}%", 100.0 * n as f64 / d.max(1) as f64)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let v4_path = std::path::PathBuf::from(
        args.first().map(String::as_str).unwrap_or("tests/fixtures/tetris_oracle_laya_en_v4.jsonl"),
    );
    let v3_path = std::path::PathBuf::from(
        args.get(1).map(String::as_str).unwrap_or("tests/fixtures/tetris_oracle_laya_en_v3.jsonl"),
    );

    let v4 = load_v4(&v4_path);
    let baseline = load_baseline_argmaxes(&v3_path);
    let n_states = v4.states.len();
    assert_eq!(n_states, 840, "the paired corpus is 120 boards × 7 previews");
    let (group_off, _group_of) = board_groups(&v4.parents);
    let n_boards = group_off.len() - 1;
    assert_eq!(n_boards, 120);

    // ── T1.7: the go/no-go flip fraction ────────────────────────────────
    let mut boards_flipped = 0usize;
    let mut states_changed = 0usize;
    let mut flipped_detail: Vec<String> = Vec::new();
    for g in 0..n_boards {
        let (gs, ge) = (group_off[g], group_off[g + 1]);
        let first = v4.states[gs].argmax;
        let same = v4.states[gs..ge].iter().all(|s| s.argmax == first);
        if !same {
            boards_flipped += 1;
            let mut d = format!("  FLIP {}: argmaxes [", v4.parents[gs]);
            for s in &v4.states[gs..ge] {
                d.push_str(&format!("{}:{}", s.next_piece, s.argmax));
                if s.next_piece != v4.states[ge - 1].next_piece {
                    d.push_str(", ");
                }
            }
            d.push(']');
            flipped_detail.push(d);
        }
        let barg = baseline
            .get(&v4.parents[gs])
            .unwrap_or_else(|| panic!("baseline missing {}", v4.parents[gs]));
        for s in &v4.states[gs..ge] {
            states_changed += usize::from(s.argmax != *barg);
        }
    }
    println!("── T1.7 go/no-go ──────────────────────────────────────────");
    println!(
        "flip fraction: {boards_flipped}/{n_boards} boards ({}) — the arm-B oracle argmax \
         changes across the 7 previews",
        pct(boards_flipped, n_boards)
    );
    println!(
        "states moved off the preview-blind pick: {states_changed}/{n_states} ({})",
        pct(states_changed, n_states)
    );
    for d in flipped_detail.iter().take(12) {
        println!("{d}");
    }
    if flipped_detail.len() > 12 {
        println!("  … {} more", flipped_detail.len() - 12);
    }

    // ── T1.9: the paired-corpus head fits ───────────────────────────────
    let option_counts: Vec<usize> = v4.states.iter().map(|s| s.options.len()).collect();
    let state_off = state_offsets(&option_counts);
    let argmaxes: Vec<usize> = v4.states.iter().map(|s| s.argmax).collect();
    let targets: Vec<f64> = v4
        .states
        .iter()
        .flat_map(|s| s.options.iter().map(|o| o.p_clean.expect("arm-B p_clean")))
        .collect();

    // Raw feature rows per option: the 5 decoded spot fills; the crossed
    // rows add the next-piece one-hot (baseline = I) and its products.
    let mut spot_raws: Vec<[f64; F_SPOT]> = Vec::with_capacity(targets.len());
    let mut cross_raws: Vec<[f64; F_CROSS]> = Vec::with_capacity(targets.len());
    for (s, st) in v4.states.iter().enumerate() {
        let next_fill = grammar_tables::tetris_piece_fill(&st.next_piece) as usize;
        for (oi, p) in v4.options[s].iter().enumerate() {
            let f = &v4.features[s][oi];
            let spot = grammar_tables::tetris_spot_forward(&v4.boards[s], p, f);
            let spot_f: [f64; F_SPOT] = [
                spot[0] as f64,
                spot[1] as f64,
                spot[2] as f64,
                spot[3] as f64,
                spot[4] as f64,
            ];
            let mut cross = [0.0f64; F_CROSS];
            cross[..F_SPOT].copy_from_slice(&spot_f);
            // dummies 1..7 (baseline I = 0), then dummy × spot.
            for k in 1..7 {
                cross[F_SPOT + (k - 1)] = (next_fill == k) as u8 as f64;
            }
            let d_base = F_SPOT + 6;
            for k in 1..7 {
                let d = (next_fill == k) as u8 as f64;
                for j in 0..F_SPOT {
                    cross[d_base + (k - 1) * F_SPOT + j] = d * spot_f[j];
                }
            }
            spot_raws.push(spot_f);
            cross_raws.push(cross);
        }
    }

    println!("── T1.9 head fits (board-grouped holdout) ─────────────────");
    let spot = run_head::<F_SPOT, D_SPOT>(
        "spot-only",
        &spot_raws,
        &targets,
        &state_off,
        &group_off,
        &argmaxes,
    );
    let crossed = run_head::<F_CROSS, D_CROSS>(
        "crossed",
        &cross_raws,
        &targets,
        &state_off,
        &group_off,
        &argmaxes,
    );
    for r in [&spot, &crossed] {
        println!(
            "{:<9} λ={:<5} argmax {}/{} ({}) | distinct {} | concord {}/{} ({}) | board-MSE {:.5} | board-flips {} | head {}",
            r.name,
            r.lam,
            r.loo_agree,
            r.n_states,
            pct(r.loo_agree, r.n_states),
            r.distinct,
            r.concord,
            r.pairs,
            pct(r.concord, r.pairs),
            r.board_mse,
            r.boards_flipped,
            r.digest,
        );
    }

    // Constant-pick + mean 1/K chance, over the 840 states.
    let mut hist = std::collections::HashMap::new();
    let mut chance = 0.0f64;
    for (&a, &k) in argmaxes.iter().zip(&option_counts) {
        *hist.entry(a).or_insert(0usize) += 1;
        chance += 1.0 / k as f64;
    }
    let (_, c) = hist.into_iter().max_by_key(|(_, c)| *c).unwrap();
    let chance = chance / argmaxes.len() as f64;
    println!(
        "baselines  constant-pick {c}/{n_states} ({}) | chance {}",
        pct(c, n_states),
        pct((chance * n_states as f64) as usize, n_states)
    );

    // ── Gates ───────────────────────────────────────────────────────────
    // The comparator's defining property (harness sanity).
    assert_eq!(spot.boards_flipped, 0, "the spot-only head must rank all 7 previews of a board identically");
    // Discrimination floor (G1: never constant).
    assert!(spot.distinct >= 2 && crossed.distinct >= 2, "discrimination floor");
    // The verdict (printed, never asserted — the record decides).
    let g1 = crossed.loo_agree > spot.loo_agree
        && crossed.concord as f64 / crossed.pairs.max(1) as f64
            > spot.concord as f64 / spot.pairs.max(1) as f64
        && crossed.board_mse < spot.board_mse;
    println!("── G1 verdict ─────────────────────────────────────────────");
    println!(
        "crossed vs spot-only: argmax {} | concord {} | board-MSE {} → {}",
        crossed.loo_agree > spot.loo_agree,
        (crossed.concord as f64 / crossed.pairs.max(1) as f64
            > spot.concord as f64 / spot.pairs.max(1) as f64),
        crossed.board_mse < spot.board_mse,
        if boards_flipped == 0 {
            "MOOT (no preview signal — T1.7 abort path)"
        } else if g1 {
            "G1 PASS (crossed beats the preview-blind comparator on all three within-board metrics)"
        } else {
            "G1 FAIL (crossed does not dominate — no promotion on this evidence)"
        }
    );
}
