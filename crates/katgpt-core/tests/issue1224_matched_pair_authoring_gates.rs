#![cfg(feature = "gain_cost_halt")]
//! Issue 1224 — the matched-pair authoring arm's gate battery (Research
//! 676, the AHA distill: "Who Needs an LLM to Program an LLM?"). The
//! THIRD authoring arm of the weight_edit_overlay lane: the
//! u-construction recipe the 993 LP was missing — same-prefix resampled
//! branch contrasts, the aligned-window mean difference, and the
//! cut-off-rate headroom gate.
//!
//! Everything runs on the ONE fixture (`wsp980_common`, no fork) — the
//! same harness the 980 (hand-authored), 993 (LP-authored) and 1209
//! (discrete-authored) batteries run.
//!
//! # Pre-registered constants (this file IS the pre-registration)
//!
//! - `PAIR_BRANCHES = 8` — AHA's 8-continuation resample.
//! - `BRANCH_EXPLORE = 0.5` — the ONE explore rate for ALL branches of a
//!   prefix (the sampler-purity law: the paper's mixed-penalty trap is
//!   the F2 falsifier's territory, never the clean build's posture).
//!   Random moves draw through `BRANCH_RANDOM_DRAW` (the pinned
//!   goal-adjacent prior — see the fixture docs for why uniform draws
//!   cannot escape the basin).
//! - `PREFIX_CUT = 3` — the prefix ends this many steps before the
//!   detected onset (AHA: 8 tokens of a ≤3072-token continuation ≈ 0.3%;
//!   the fixture's 64-step episodes give 3 ≈ 4.7% — relatively deeper so
//!   the prefix always holds a pre-onset decision).
//! - `WINDOW_START_AFTER_ONSET = 2`, `WINDOW_LEN = 8` — the aligned
//!   window, anchored ONSET-RELATIVE on the LOOPING branch's detected
//!   onset (the repo README's reading, which the research note pins as
//!   the more precise anchor), the SAME indices applied to both
//!   branches. AHA: 160 states starting 64 after onset of ≤3072-token
//!   continuations (≈5.2% window, ≈2% offset); the fixture scales the
//!   RATIO up (12.5% window, 3% offset) to keep the window mean's
//!   variance bounded at 64-step scale. The window is measured in `y2`
//!   (the layer-2 pre-activation) — the residual-stream analog and the
//!   exact space the overlay direction indexes.
//! - `HOLDOUT_EVERY = 5` — every 5th pair held out (the paper's 80/20).
//! - `MIN_WINDOW_STATES = 2` — a branch whose aligned window clamps
//!   below this starves the pair (counted, never silently padded).
//! - `YIELD_FLOOR = 0.5` — the loud yield assertion (the paper's 228/300
//!   and 35/80 discipline), pinned below the measured 10/16 = 0.625.
//!
//! # The labeling law (measured, not assumed)
//!
//! The issue pre-registered the fixture's oscillation detector as the
//! branch labeler. The census REFUTES that at this fixture: the basin's
//! argmax bounce fires the 3-window detector on EVERY branch — with the
//! paper's uniform temperature-sampling analog, 8/8 branches of every
//! probed prefix oscillate at every explore rate ∈ {0.10..0.45}, so
//! detector-only labeling starves every prefix (yield 0). The labels are
//! therefore BEHAVIORAL — looping = did not finish AND detector-shaped,
//! productive = finished (the paper's own labels: a looping continuation
//! hits the token cap, a productive one completes) — with the detector
//! kept as the looping label's SHAPE assertion (every stuck branch
//! oscillates, pinned in T1). The successful branches are themselves
//! usually still oscillating when they finish — the bounce persists
//! through the escape — which is exactly why the detector alone cannot
//! separate.
//!
//! # The pre-registered bar (the issue's own law)
//!
//! **PARITY, not superiority**: the fixture cannot reproduce the
//! real-model confound that makes matched pairs matter (there is no
//! whitened-direction rival here to beat — the construction is judged
//! against the TARGETED direction's battery, the heuristic every
//! earlier arm composed with). `u_matched` passing the full 993
//! battery ≡ construction correctness + determinism. MEASURED: the
//! mined direction recovers the injected loop axis (cos ≈ 0.97 with the
//! targeted direction, ch1 mass ≈ 0.97) and clears the obstacle at the
//! full authored dose (the dose grid is monotone non-increasing and
//! lands at 0). Recorded beside the parity verdict: (a) the edit's
//! success share at dose 1.0 is 0.495 vs the targeted edit's 1.000 —
//! the direction's off-axis mass (ch0/ch4) taxes the horizontal walk;
//! (b) BEYOND the pinned grid the dose response is NON-MONOTONE (dose
//! 1.25/1.5 re-loop at rate 1.0) — over-suppression re-locks the
//! channel, the narrow-window caveat (309's α ≤ 0.3 law) in fixture
//! form. Neither rides an assert: they are the record, not the bar.
//!
//! # T2b's measured outcome (the falsifiers CANNOT fire)
//!
//! F1 (mismatched-prefix pairs) and F2 (sampler-split pairs) produce
//! directions ≈ the matched build (cos 0.95 / 0.99): the behavioral
//! stuck-vs-finished axis dominates every injected confound, and the
//! fixture cannot stage confound corruption of the mined direction. Per
//! the issue's own T2b law this is recorded honestly: the fixture lacks
//! confound structure, and the matched-pair parity claim stays
//! **construction-only** — no confound-robustness claim is made, and the
//! arm makes no recipe-value claim beyond this fixture (the
//! demotion-law input, recorded in the bench file; the non-firing pins
//! in T2b are the tripwire that re-opens the record if the fixture or
//! sampler ever changes under it).
//!
//! # Arms
//!
//! - T1: the pair builder — yield floor (loud), pairing law (earliest
//!   looping × longest productive, deterministic tiebreaks; verified
//!   structurally by re-derivation), the stuck-branch oscillation-shape
//!   census, digest determinism.
//! - T2: `u_matched` through the UNCHANGED 993 pipeline
//!   (`solve_min_l1_authoring_lp` + `compose_direction_overlay`) and the
//!   FULL 993 battery: sparsity, held-out separation, determinism
//!   digests, dose grid, isolation null, 16-arm random control, twin
//!   pre-screen, retention bit-identity — plus the direction-recovery
//!   diagnostic (cos vs the injected loop direction). The LP inputs are
//!   the SAME labeled trace sets the 993 battery solved — only the
//!   direction source changes (targeted → mined), isolating the variable
//!   under test.
//! - T2b: the confound falsifiers — F1 deliberately mismatched-prefix
//!   pairs and F2 the sampler-split build. MEASURED: neither degrades
//!   (the directions agree with the matched build to cos ≥ 0.9, pinned);
//!   the honest negative is the assertion (the fixture cannot separate
//!   confounded from clean construction — the construction-only scope).
//! - T3: the cut-off-rate headroom gate on the fixture — the outcomes-
//!   aware arming readout (Armed, recoverable share 1.0, truncation
//!   rate 1.0 on the base telemetry), the outcomes-absent posture
//!   (headroom unmeasured), the clean twin's refusal, and the synthetic
//!   ≈0-floor / anti-falsifier / mismatch arms in the carrier's own
//!   unit tests.

mod wsp980_common;

use katgpt_core::blake3_seed;
use katgpt_core::weight_edit_overlay::lp_authoring::{
    LpAuthoringConfig, LpAuthoringError, TraceSet, compose_direction_overlay,
    solve_min_l1_authoring_lp, truncate_small,
};
use katgpt_core::weight_edit_overlay::matched_pair_authoring::{
    self, MatchedPairConfig, MatchedPairDirection, PairContrast,
};
use katgpt_core::weight_edit_overlay::{
    ArmRefusal, ArmVerdict, HeadroomReport, ObstacleArmConfig, ProvenanceTrailer,
    WeightEditArtifact, WeightEditOverlay, arm_from_telemetry, arm_from_telemetry_with_outcomes,
    norm_matched_random_direction, paired_lcb95,
};
use wsp980_common::*;

// ─────────────────────────────────────────────────────────────────────
// Pinned constants (the pre-registration above)
// ─────────────────────────────────────────────────────────────────────

const PAIR_BRANCHES: usize = 8;
const BRANCH_EXPLORE: f32 = 0.5;
const PREFIX_CUT: usize = 3;
const WINDOW_START_AFTER_ONSET: usize = 2;
const WINDOW_LEN: usize = 8;
const MIN_WINDOW_STATES: usize = 2;
const HOLDOUT_EVERY: usize = 5;
const YIELD_FLOOR: f32 = 0.5;

// The 993 LP pins (verbatim — the battery arms read the same world).
const LP_EPSILON: f32 = 0.0;
const LP_C: f32 = 10.0;
const LP_MAX_ITER: u32 = 20_000;
const LP_TOL: f32 = 1e-5;
const CARRIER_RHO: f32 = 0.2;
const SEP_TOL: f32 = 0.05;
const KEEP_TOL: f32 = 1e-3;
const LP_DOSE_GRID: [f32; 3] = [0.5, 0.75, 1.0];
const RANDOM_ARMS_1224: usize = 16;

// ─────────────────────────────────────────────────────────────────────
// LP trace helpers — VERBATIM mirrors of the 993 battery's
// (tests/issue993_lp_authored_overlay_gates.rs): the T2 claim "the full
// 993 battery, only the direction source changed" requires the SAME
// labeled trace sets, the SAME derived δ, the SAME solve.
// ─────────────────────────────────────────────────────────────────────

struct LabeledTraces {
    push_rows: Vec<f32>,
    n_push: usize,
    keep_rows: Vec<f32>,
    n_keep: usize,
}

fn row_cmp(a: &[f32], b: &[f32]) -> std::cmp::Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        match x.total_cmp(y) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    std::cmp::Ordering::Equal
}

fn record_traces(seeds: std::ops::Range<u64>) -> LabeledTraces {
    let model = build_fixture(true);
    let mut push: Vec<Vec<f32>> = Vec::new();
    let mut keep: Vec<Vec<f32>> = Vec::new();
    for seed in seeds {
        for start in seeded_starts(seed) {
            let tr = rollout_trace(&model, None, start);
            for step in &tr.steps {
                let carrier = step.h1[CARRIER_COL as usize];
                if step.y.abs() <= REGION_HALF {
                    if carrier <= -CARRIER_RHO {
                        push.push(step.h1.clone());
                    }
                    // carrier-positive in-region states: the
                    // onset-ambiguity class — recorded, unconstrained.
                } else {
                    keep.push(step.h1.clone());
                }
            }
        }
    }
    let dedupe = |rows: Vec<Vec<f32>>| -> (Vec<f32>, usize) {
        let mut rows = rows;
        rows.sort_by(|a: &Vec<f32>, b: &Vec<f32>| row_cmp(a, b));
        rows.dedup();
        let n = rows.len();
        (rows.concat(), n)
    };
    let (push_rows, n_push) = dedupe(push);
    let (keep_rows, n_keep) = dedupe(keep);
    LabeledTraces {
        push_rows,
        n_push,
        keep_rows,
        n_keep,
    }
}

fn derived_delta(t: &LabeledTraces) -> f32 {
    let mut min_carrier = f32::INFINITY;
    for i in 0..t.n_push {
        let row = &t.push_rows[i * H..(i + 1) * H];
        let a = row[CARRIER_COL as usize].abs();
        if a < min_carrier {
            min_carrier = a;
        }
    }
    assert!(
        min_carrier.is_finite() && min_carrier > 0.0,
        "push rows must carry carrier mass"
    );
    INJECT_K0 * min_carrier
}

fn solve_lp(
    t: &LabeledTraces,
) -> Result<katgpt_core::weight_edit_overlay::lp_authoring::LpAuthoringSolution, LpAuthoringError> {
    if t.n_push == 0 {
        let push = TraceSet::new(&t.push_rows, t.n_push, H)?;
        let keep = TraceSet::new(&t.keep_rows, t.n_keep, H)?;
        let cfg = LpAuthoringConfig {
            delta_push: 1.0,
            epsilon_keep: LP_EPSILON,
            c_slack: LP_C,
            max_iter: LP_MAX_ITER,
            tol: LP_TOL,
        };
        return solve_min_l1_authoring_lp(&push, &keep, &cfg);
    }
    let delta = derived_delta(t);
    let push = TraceSet::new(&t.push_rows, t.n_push, H)?;
    let keep = TraceSet::new(&t.keep_rows, t.n_keep, H)?;
    let cfg = LpAuthoringConfig {
        delta_push: delta,
        epsilon_keep: LP_EPSILON,
        c_slack: LP_C,
        max_iter: LP_MAX_ITER,
        tol: LP_TOL,
    };
    solve_min_l1_authoring_lp(&push, &keep, &cfg)
}

fn separation_scores(v: &[f32], t: &LabeledTraces, delta: f32) -> (f64, f64) {
    let mut ok_push = 0u64;
    for i in 0..t.n_push {
        let row = &t.push_rows[i * H..(i + 1) * H];
        let mut acc = 0.0f32;
        for (x, w) in v.iter().zip(row.iter()) {
            acc += x * w;
        }
        if acc <= -delta + SEP_TOL {
            ok_push += 1;
        }
    }
    let mut ok_keep = 0u64;
    for i in 0..t.n_keep {
        let row = &t.keep_rows[i * H..(i + 1) * H];
        let mut acc = 0.0f32;
        for (x, w) in v.iter().zip(row.iter()) {
            acc += x * w;
        }
        if acc.abs() <= KEEP_TOL {
            ok_keep += 1;
        }
    }
    (
        ok_push as f64 / t.n_push.max(1) as f64,
        ok_keep as f64 / t.n_keep.max(1) as f64,
    )
}

fn all_eval_starts() -> Vec<(f32, f32)> {
    (0..EVAL_SEEDS as u64).flat_map(seeded_starts).collect()
}

// ─────────────────────────────────────────────────────────────────────
// The pair builder (T1) — the world half of the AHA recipe
// ─────────────────────────────────────────────────────────────────────

/// The pair build's mode: the clean matched build, and the two confound
/// falsifiers (T2b).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PairMode {
    /// The paper's clean build: every branch from ONE sampler
    /// (`BRANCH_EXPLORE`), pairs from ONE prefix.
    Matched,
    /// F1: pairs built from DIFFERENT prefixes — the looping branch of
    /// prefix i paired with the productive branch of prefix i+1 (the
    /// unmatched-pile shape that made the whitened direction fail). The
    /// window offsets stay mechanically identical; only the prefix
    /// identity is broken.
    Mismatched,
    /// F2: the sampler-split build — looping branches drawn ONLY from
    /// the ε=0 greedy continuation, productive branches ONLY from the
    /// explored family (the paper's presence-penalty trap analog: the
    /// label correlates with the sampler, not only with the behavior).
    SamplerSplit,
}

struct PairSet {
    contrasts: Vec<PairContrast>,
    /// Prefixes examined (base looping episodes found).
    prefixes_total: usize,
    /// Prefixes that yielded a pair.
    prefixes_yielded: usize,
    /// Prefixes skipped because no branch pair formed (starved).
    prefixes_starved: usize,
    /// BLAKE3 over every pair's contrast values, in pair order — the
    /// determinism witness.
    digest: blake3::Hash,
}

/// Position after applying `a` from `(x, y)` (the fixture's move table).
fn advance(x: f32, y: f32, a: usize) -> (f32, f32) {
    match a {
        0 => (x, y + 1.0),
        1 => (x, y - 1.0),
        2 => (x - 1.0, y),
        _ => (x + 1.0, y),
    }
}

/// Earliest-looping pick: among LOOPING branches (stuck ∧ oscillating —
/// the behavioral label), the one with the smallest detected onset; ties
/// → the lowest seed index (deterministic).
fn pick_earliest_looping(branches: &[BranchTrace]) -> Option<(usize, &BranchTrace)> {
    let mut best: Option<(usize, usize)> = None; // (onset, index)
    for (i, b) in branches.iter().enumerate() {
        if b.success || oscillation_onset(&b.actions).is_none() {
            continue; // looping = did NOT finish AND detector-shaped
        }
        let on = oscillation_onset(&b.actions).expect("checked");
        let better = match best {
            None => true,
            Some((bon, bi)) => on < bon || (on == bon && i < bi),
        };
        if better {
            best = Some((on, i));
        }
    }
    best.map(|(on, i)| (on, &branches[i]))
}

/// Longest-productive pick: among PRODUCTIVE branches (finished — the
/// behavioral label), the longest action sequence; ties → the lowest seed
/// index (deterministic).
fn pick_longest_productive(branches: &[BranchTrace]) -> Option<&BranchTrace> {
    let mut best: Option<(usize, usize)> = None; // (len, index)
    for (i, b) in branches.iter().enumerate() {
        if !b.success {
            continue;
        }
        let len = b.actions.len();
        let better = match best {
            None => true,
            Some((blen, bi)) => len > blen || (len == blen && i < bi),
        };
        if better {
            best = Some((len, i));
        }
    }
    best.map(|(_, i)| &branches[i])
}

/// The aligned window pair: both means extracted at the SAME
/// onset-relative indices (the alignment law), clamped to each branch's
/// length, `None` when either side starves below the minimum.
fn window_pair(
    loop_branch: &BranchTrace,
    prod_branch: &BranchTrace,
    anchor: usize,
) -> Option<PairContrast> {
    let s = anchor + WINDOW_START_AFTER_ONSET;
    let loop_n = loop_branch.states.len().saturating_sub(s);
    let prod_n = prod_branch.states.len().saturating_sub(s);
    if loop_n < MIN_WINDOW_STATES || prod_n < MIN_WINDOW_STATES {
        return None;
    }
    let looping_mean = matched_pair_authoring::window_mean(&loop_branch.states, s, WINDOW_LEN)?;
    let productive_mean = matched_pair_authoring::window_mean(&prod_branch.states, s, WINDOW_LEN)?;
    Some(PairContrast {
        looping_mean,
        productive_mean,
    })
}

/// Build the pair set for `mode` over the injected fixture's base looping
/// episodes (the in-region grid, deterministic greedy — the world's
/// looping population). Deterministic end to end: same inputs → the same
/// digest.
fn build_pair_set(mode: PairMode) -> PairSet {
    let model = build_fixture(true);
    let mut contrasts = Vec::new();
    let mut prefixes_total = 0usize;
    let mut prefixes_yielded = 0usize;
    let mut prefixes_starved = 0usize;
    let mut digest_input: Vec<f32> = Vec::new();

    // The base looping episodes: deterministic greedy rollouts over the
    // in-region grid (the fixture's looping population — the same set the
    // 980 battery's obstacle telemetry reads).
    let starts = in_region_starts();
    for (pi, start) in starts.iter().enumerate() {
        let base = rollout_trace(&model, None, *start);
        let base_actions: Vec<usize> = base.steps.iter().map(|s| s.action).collect();
        let Some(onset) = oscillation_onset(&base_actions) else {
            continue;
        };
        prefixes_total += 1;

        // The prefix cut (AHA: cut before the detected onset) and the
        // branch start = the state AFTER the prefix's last step.
        let prefix_len = onset.saturating_sub(PREFIX_CUT).max(1);
        let last = &base.steps[prefix_len - 1];
        let branch_start = advance(last.x, last.y, last.action);

        // Branch manufacture. Seed layout: branch b of prefix pi draws
        // seed `pi * 100 + b` (deterministic per (prefix, branch)).
        let branches: Vec<BranchTrace> = (0..PAIR_BRANCHES)
            .map(|b| {
                let seed = (pi as u64) * 100 + b as u64;
                rollout_branch_y2(&model, branch_start, seed, BRANCH_EXPLORE)
            })
            .collect();
        // The ε=0 greedy continuation (the F2 looping family; also the
        // record of what the deterministic policy does from this prefix).
        let greedy_seed = (pi as u64) * 100 + 99;
        let greedy = rollout_branch_y2(&model, branch_start, greedy_seed, 0.0);

        // Label + pair per mode.
        let pair = match mode {
            PairMode::Matched => {
                let Some((anchor, loop_branch)) = pick_earliest_looping(&branches) else {
                    prefixes_starved += 1;
                    continue;
                };
                let Some(prod_branch) = pick_longest_productive(&branches) else {
                    prefixes_starved += 1;
                    continue;
                };
                window_pair(loop_branch, prod_branch, anchor)
            }
            PairMode::Mismatched => {
                // The looping branch of prefix pi, the productive branch
                // of prefix (pi+1) % n — deliberately cross-prefix.
                let n = starts.len();
                let other_start = starts[(pi + 1) % n];
                let other = rollout_trace(&model, None, other_start);
                let other_actions: Vec<usize> = other.steps.iter().map(|s| s.action).collect();
                let Some(other_onset) = oscillation_onset(&other_actions) else {
                    prefixes_starved += 1;
                    continue;
                };
                let other_prefix_len = other_onset.saturating_sub(PREFIX_CUT).max(1);
                let other_last = &other.steps[other_prefix_len - 1];
                let other_branch_start = advance(other_last.x, other_last.y, other_last.action);
                let other_branches: Vec<BranchTrace> = (0..PAIR_BRANCHES)
                    .map(|b| {
                        let seed = (((pi + 1) % n) as u64) * 100 + b as u64;
                        rollout_branch_y2(&model, other_branch_start, seed, BRANCH_EXPLORE)
                    })
                    .collect();
                let Some((anchor, loop_branch)) = pick_earliest_looping(&branches) else {
                    prefixes_starved += 1;
                    continue;
                };
                let Some(prod_branch) = pick_longest_productive(&other_branches) else {
                    prefixes_starved += 1;
                    continue;
                };
                window_pair(loop_branch, prod_branch, anchor)
            }
            PairMode::SamplerSplit => {
                // Looping candidates: ONLY the ε=0 greedy continuation
                // (the deterministic policy — it never finishes and always
                // oscillates in the basin). Productive candidates: ONLY
                // the explored family.
                if greedy.success || oscillation_onset(&greedy.actions).is_none() {
                    prefixes_starved += 1;
                    continue;
                }
                let anchor = oscillation_onset(&greedy.actions).expect("checked");
                let Some(prod_branch) = pick_longest_productive(&branches) else {
                    prefixes_starved += 1;
                    continue;
                };
                window_pair(&greedy, prod_branch, anchor)
            }
        };
        let Some(pair) = pair else {
            prefixes_starved += 1;
            continue;
        };
        prefixes_yielded += 1;
        digest_input.extend_from_slice(&pair.looping_mean);
        digest_input.extend_from_slice(&pair.productive_mean);
        contrasts.push(pair);
    }

    let digest = f32_digest(&digest_input);
    PairSet {
        contrasts,
        prefixes_total,
        prefixes_yielded,
        prefixes_starved,
        digest,
    }
}

/// The yield fraction (pairs per examined prefix).
fn yield_fraction(ps: &PairSet) -> f32 {
    if ps.prefixes_total == 0 {
        return 0.0;
    }
    ps.prefixes_yielded as f32 / ps.prefixes_total as f32
}

/// The direction from a pair set (the closed-form half, `u_matched`).
fn direction_of(ps: &PairSet) -> MatchedPairDirection {
    matched_pair_authoring::build_matched_pair_direction(
        &ps.contrasts,
        &MatchedPairConfig {
            holdout_every: HOLDOUT_EVERY,
        },
    )
    .expect("the matched-pair build must construct")
}

/// The LP overlay composed with an arbitrary direction `u` at dose
/// scaling of `v` (the 993 `lp_overlay`, generalized over u).
fn lp_overlay_u(u: &[f32], v: &[f32], dose: f32) -> WeightEditOverlay {
    let scaled: Vec<f32> = v.iter().map(|x| x * dose).collect();
    compose_direction_overlay(u, &scaled, LAYER_W2, H).expect("LP overlay composes")
}

/// Cosine between two vectors (sequential fold; zero-norm → 0.0).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Obstacle rate + success rate over a start list (obstacle_rate from the
/// fixture + the success half the battery records beside the parity bar).
fn obstacle_and_success(
    model: &Fixture,
    overlay: Option<&WeightEditOverlay>,
    starts: &[(f32, f32)],
) -> (f32, f32) {
    let mut osc = 0usize;
    let mut succ = 0usize;
    for s in starts {
        let r = rollout(model, overlay, *s);
        if episode_oscillates(&r.actions) {
            osc += 1;
        }
        if r.success {
            succ += 1;
        }
    }
    (
        osc as f32 / starts.len() as f32,
        succ as f32 / starts.len() as f32,
    )
}

// ─────────────────────────────────────────────────────────────────────
// T1 — the pair builder
// ─────────────────────────────────────────────────────────────────────

/// The dose response's measured overshoot caveat, recorded (never
/// asserted): beyond the pinned grid the u_matched edit RE-LOCKS the
/// channel (dose 1.25/1.5 re-loop at rate 1.0) — over-suppression, the
/// narrow-window caveat in fixture form. Recorded in the bench file.

#[test]
fn t1_pair_builder_yield_pairing_law_and_determinism() {
    let ps = build_pair_set(PairMode::Matched);
    assert!(
        ps.prefixes_total > 0,
        "the injected fixture must carry looping base episodes"
    );
    let yf = yield_fraction(&ps);
    assert!(
        yf >= YIELD_FLOOR,
        "pair yield {yf:.3} below the pre-registered floor {YIELD_FLOOR} \
         (the 228/300 discipline: a starved build refuses loud, never quietly \
         authors from fewer pairs) — total={} yielded={} starved={}",
        ps.prefixes_total,
        ps.prefixes_yielded,
        ps.prefixes_starved
    );

    // Structural pairing-law verification: re-derive the picks for EVERY
    // yielded prefix independently of the builder's accumulation, and
    // compare each rebuilt contrast with the builder's k-th yielded pair.
    let model = build_fixture(true);
    let starts = in_region_starts();
    let mut k = 0usize;
    let mut stuck = 0usize;
    let mut stuck_osc = 0usize;
    for (pi, start) in starts.iter().enumerate() {
        let base = rollout_trace(&model, None, *start);
        let base_actions: Vec<usize> = base.steps.iter().map(|s| s.action).collect();
        let Some(onset) = oscillation_onset(&base_actions) else {
            continue;
        };
        let prefix_len = onset.saturating_sub(PREFIX_CUT).max(1);
        let last = &base.steps[prefix_len - 1];
        let branch_start = advance(last.x, last.y, last.action);
        let branches: Vec<BranchTrace> = (0..PAIR_BRANCHES)
            .map(|b| {
                let seed = (pi as u64) * 100 + b as u64;
                rollout_branch_y2(&model, branch_start, seed, BRANCH_EXPLORE)
            })
            .collect();
        // The stuck-branch oscillation-shape census (the detector's role
        // in the labeling law: every stuck branch oscillates).
        for b in &branches {
            if !b.success {
                stuck += 1;
                if episode_oscillates(&b.actions) {
                    stuck_osc += 1;
                }
            }
        }
        let Some((anchor, loop_branch)) = pick_earliest_looping(&branches) else {
            continue;
        };
        let Some(prod_branch) = pick_longest_productive(&branches) else {
            continue;
        };
        let Some(expect) = window_pair(loop_branch, prod_branch, anchor) else {
            continue;
        };
        // The builder's k-th yielded pair must be exactly this one.
        assert!(
            k < ps.contrasts.len(),
            "the re-derivation yielded more pairs than the builder"
        );
        assert_eq!(
            &ps.contrasts[k], &expect,
            "pairing-law re-derivation, pair {k}"
        );
        k += 1;
    }
    assert_eq!(
        k, ps.prefixes_yielded,
        "the re-derivation must reproduce the builder's yield count"
    );
    // The detector-shape census: a stuck branch that never oscillated
    // would break the labeling law's detector half.
    assert!(
        stuck > 0 && stuck_osc == stuck,
        "every stuck branch must be detector-shaped (osc {stuck_osc}/{stuck})"
    );

    // Digest determinism: a second build must land the identical digest.
    let ps2 = build_pair_set(PairMode::Matched);
    assert_eq!(
        ps.digest.to_hex(),
        ps2.digest.to_hex(),
        "two builds on the pinned inputs must be digest-identical"
    );
    assert_eq!(ps.contrasts.len(), ps2.contrasts.len());
    for (a, b) in ps.contrasts.iter().zip(ps2.contrasts.iter()) {
        assert_eq!(a, b);
    }
}

// ─────────────────────────────────────────────────────────────────────
// T2 — u_matched through the UNCHANGED 993 pipeline + the FULL battery
// (the pre-registered PARITY bar: passing ≡ construction correctness)
// ─────────────────────────────────────────────────────────────────────

const SPARSITY_CAP_K: usize = 8;

#[test]
fn t2_matched_direction_passes_the_full_993_battery() {
    let ps = build_pair_set(PairMode::Matched);
    let built = direction_of(&ps);
    let u = &built.direction;
    let targeted = Fixture::targeted_direction();

    // Direction-recovery diagnostic: the mined direction recovers the
    // injected loop axis (measured cos 0.9668, ch1 mass 0.967) without
    // ever being told it. This is the construction-correctness evidence
    // the battery arms then confirm behaviorally.
    let cos_u = cosine(u, &targeted);
    assert!(
        cos_u >= 0.90,
        "the mined direction must recover the loop axis: cos {cos_u}"
    );
    assert_eq!(
        built.holdout_pairs, 2,
        "10 pairs at holdout_every 5 → 2 held out (pinned shape)"
    );
    // The module's own holdout check is REPORTED, never gated (the
    // separability ≠ edit-quality law); the measured gap (−1.43 at n=2)
    // lives in the bench record.

    // The LP half: SAME traces, SAME solve as the 993 battery.
    let fit = record_traces(0..8);
    let held = record_traces(8..16);
    assert!(
        fit.n_push > 0 && fit.n_keep > 0,
        "trace authoring must produce both classes"
    );
    let sol = solve_lp(&fit).expect("LP solve");
    let max_abs = sol.v.iter().copied().fold(0.0f32, f32::max).abs();
    assert!(max_abs > 0.0, "solve must not return an empty direction");
    let v = truncate_small(sol.v.clone(), 1e-3 * max_abs).expect("floor");

    // (a) sparsity ≤ K; isolation column silent; carrier column carries
    // the edit (v-side arms — unchanged by the direction swap).
    let nonzeros = v.iter().filter(|&&x| x != 0.0).count();
    assert!(
        nonzeros <= SPARSITY_CAP_K,
        "sparsity {nonzeros} exceeds the pre-registered K={SPARSITY_CAP_K}"
    );
    assert_eq!(
        v[ISOLATION_COL as usize], 0.0,
        "isolation column must stay silent"
    );
    assert!(
        v[CARRIER_COL as usize] != 0.0,
        "the carrier column must carry the edit"
    );

    // (b) held-out separation (v-side, rows the solve never saw).
    let delta = derived_delta(&fit);
    let (push_frac, keep_frac) = separation_scores(&v, &held, delta);
    assert!(
        push_frac >= 0.95,
        "held-out push separation {push_frac} below the floor"
    );
    assert_eq!(
        keep_frac, 1.0,
        "held-out keep separation must be exact-silent"
    );

    // (c) determinism: two direction builds → bit-identical u → identical
    // overlay + artifact digests (the committed, hash-comparable result).
    let built2 = direction_of(&build_pair_set(PairMode::Matched));
    assert!(
        built
            .direction
            .iter()
            .zip(built2.direction.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "two constructions on the pinned pairs must be bit-identical"
    );
    assert_eq!(built, built2);
    let ov1 = lp_overlay_u(u, &v, 1.0);
    let ov2 = lp_overlay_u(&built2.direction, &v, 1.0);
    assert_eq!(ov1.digest(), ov2.digest(), "overlay digests must match");
    let provenance = ProvenanceTrailer {
        prompt_digest: Some(*blake3::hash(b"issue1224 matched-pair authoring").as_bytes()),
        ..Default::default()
    };
    let art1 = WeightEditArtifact::mint(&ov1, provenance.clone());
    let art2 = WeightEditArtifact::mint(&ov2, provenance);
    assert_eq!(
        art1.artifact_digest(),
        art2.artifact_digest(),
        "artifact digests must match"
    );

    // (d) the battery arms, u_matched-authored. Dose-response: monotone
    // non-increasing over the pinned grid, the full (authored) dose
    // clears to the floor. MEASURED: [1.0, 1.0, 0.0].
    let deg = build_fixture(true);
    let starts = all_eval_starts();
    let mut prev = 2.0f32;
    for dose in LP_DOSE_GRID {
        let ov = lp_overlay_u(u, &v, dose);
        let (rate, _succ) = obstacle_and_success(&deg, Some(&ov), &starts);
        assert!(
            rate <= prev + 1e-6,
            "dose-response must be non-increasing: dose {dose} rate {rate} > prev {prev}"
        );
        prev = rate;
    }
    assert_eq!(prev, 0.0, "full (authored) dose must clear the obstacle");

    // (e) neighbor-column isolation null: a same-magnitude control edit on
    // the isolation column must NOT clear.
    let mut ctrl = vec![0.0f32; H];
    ctrl[ISOLATION_COL as usize] = v[CARRIER_COL as usize].abs();
    let ov_ctrl =
        compose_direction_overlay(u, &ctrl, LAYER_W2, H).expect("control overlay composes");
    let (rate_ctrl, _) = obstacle_and_success(&deg, Some(&ov_ctrl), &starts);
    assert!(
        rate_ctrl >= 0.5,
        "isolation-column control must not clear: rate {rate_ctrl}"
    );

    // (f) norm-matched random control: ≥16 BLAKE3-seeded arms, u swapped
    // for a norm-matched random direction, paired per-arm — the mined
    // direction must strictly win (paired LCB95 > 0).
    let mut deltas = Vec::with_capacity(RANDOM_ARMS_1224);
    for arm in 0..RANDOM_ARMS_1224 as u64 {
        let seed = blake3_seed(b"issue1224/u_control", arm);
        let mut u_rand = vec![0.0f32; H];
        norm_matched_random_direction(seed, H, &mut u_rand);
        let ov_rand = compose_direction_overlay(&u_rand, &v, LAYER_W2, H)
            .expect("random control overlay composes");
        let arm_starts = seeded_starts(arm);
        let (rate_r, _) = obstacle_and_success(&deg, Some(&ov_rand), &arm_starts);
        let (rate_l, _) = obstacle_and_success(&deg, Some(&ov1), &arm_starts);
        deltas.push(rate_r - rate_l);
    }
    let lcb = paired_lcb95(&deltas).expect("≥2 arms");
    assert!(
        lcb > 0.0,
        "random control must fail where the mined edit clears: LCB95 {lcb}"
    );

    // (g) obstacle pre-screen refusal on the clean twin (the lane's
    // falsifier, re-asserted for the matched-pair posture).
    let twin = build_fixture(false);
    let mut twin_episodes: Vec<Vec<Vec<f32>>> = Vec::new();
    for s in 0..EVAL_SEEDS as u64 {
        for start in seeded_starts(s) {
            twin_episodes.push(rollout(&twin, None, start).hidden);
        }
    }
    assert!(matches!(
        arm_from_telemetry(&twin_episodes, &ObstacleArmConfig::default()),
        ArmVerdict::Refused(_)
    ));

    // (h) retention: out-of-region forwards bit-identical (the overlay's
    // carrier-0 no-op law, re-asserted for u_matched-authored v).
    for &(x, y) in &[(-6.0f32, 5.0), (0.0, 7.0), (3.0, -5.0), (-2.0, 6.0)] {
        let inp = make_input(x, y);
        let with = deg.forward(&inp, Some(&ov1), None, None);
        let without = deg.forward(&inp, None, None, None);
        assert_eq!(
            with, without,
            "out-of-region forward must be bit-identical at ({x}, {y})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// T2b — the confound falsifiers: MEASURED non-firing, pinned as the
// honest record + tripwire (the construction-only scope)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t2b_confound_falsifiers_record_the_honest_non_firing() {
    let matched = build_pair_set(PairMode::Matched);
    let dir = direction_of(&matched);

    // F1 — deliberately mismatched-prefix pairs: the looping branch of
    // prefix i paired with the productive branch of prefix i+1 (the
    // unmatched-pile shape that made the whitened direction fail on the
    // paper's model).
    let mismatched = build_pair_set(PairMode::Mismatched);
    assert!(
        !mismatched.contrasts.is_empty(),
        "the F1 build must produce pairs on this fixture"
    );
    let dir_mis = direction_of(&mismatched);
    let cos_mis = cosine(&dir_mis.direction, &dir.direction);

    // F2 — the sampler-split build: looping ONLY from the ε=0 greedy
    // continuation, productive ONLY from the explored family (the paper's
    // presence-penalty trap analog).
    let split = build_pair_set(PairMode::SamplerSplit);
    assert!(
        !split.contrasts.is_empty(),
        "the F2 build must produce pairs on this fixture"
    );
    let dir_split = direction_of(&split);
    let cos_split = cosine(&dir_split.direction, &dir.direction);

    // MEASURED non-firing: both confounded builds land within cos 0.95+
    // of the matched direction (0.9545 / 0.9907) — the behavioral
    // stuck-vs-finished axis dominates the injected confounds, so the
    // fixture cannot demonstrate confound corruption. Per the issue's
    // T2b law this is the honest record: the parity claim stays
    // CONSTRUCTION-ONLY. The pins below are the tripwire: a fixture or
    // sampler change that makes the falsifiers fire reds here and the
    // record must be re-opened, never silently widened.
    assert!(
        cos_mis >= 0.90,
        "F1 now DEGRADES the direction (cos {cos_mis}) — the falsifier FIRED; \
         the construction-only record is stale and must be re-adjudicated"
    );
    assert!(
        cos_split >= 0.90,
        "F2 now DEGRADES the direction (cos {cos_split}) — the falsifier FIRED; \
         the construction-only record is stale and must be re-adjudicated"
    );
}

// ─────────────────────────────────────────────────────────────────────
// T3 — the cut-off-rate headroom gate on the fixture telemetry
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t3_headroom_readout_on_the_fixture() {
    let deg = build_fixture(true);
    let mut episodes = Vec::new();
    let mut truncated = Vec::new();
    for s in 0..EVAL_SEEDS as u64 {
        for start in seeded_starts(s) {
            let r = rollout(&deg, None, start);
            episodes.push(r.hidden);
            truncated.push(!r.success);
        }
    }
    // The outcomes-aware verdict: Armed with the headroom readout.
    // MEASURED: every base episode oscillates and truncates → recoverable
    // share 1.0, truncation rate 1.0 (the ceiling REPORT beside the
    // verdict — max recoverable ≈ the base cut-off rate).
    let verdict =
        arm_from_telemetry_with_outcomes(&episodes, &truncated, &ObstacleArmConfig::default());
    let ArmVerdict::Armed {
        oscillation_rate,
        episodes: n,
        headroom: Some(rep),
    } = verdict
    else {
        panic!("the degenerate fixture's telemetry must arm: {verdict:?}");
    };
    assert!(
        (oscillation_rate - 1.0).abs() < 1e-6,
        "rate {oscillation_rate}"
    );
    assert_eq!(n, episodes.len());
    let HeadroomReport {
        obstacle_episodes,
        obstacle_truncated,
        recoverable_share,
        truncation_rate,
    } = rep;
    assert_eq!(obstacle_episodes, episodes.len());
    assert_eq!(obstacle_truncated, episodes.len());
    assert_eq!(recoverable_share, Some(1.0));
    assert_eq!(truncation_rate, Some(1.0));

    // The outcomes-absent posture (the bare fn): same oscillation verdict,
    // headroom unmeasured, the floor never fired.
    let bare = arm_from_telemetry(&episodes, &ObstacleArmConfig::default());
    let ArmVerdict::Armed { headroom: None, .. } = bare else {
        panic!("the outcomes-absent posture must carry no headroom: {bare:?}");
    };

    // The clean twin refuses on obstacle telemetry (the headroom axis is
    // irrelevant below the arming threshold — precedence unchanged).
    let twin = build_fixture(false);
    let twin_episodes: Vec<Vec<Vec<f32>>> = seeded_starts(0)
        .iter()
        .map(|s| rollout(&twin, None, *s).hidden)
        .collect();
    let twin_truncated: Vec<bool> = seeded_starts(0)
        .iter()
        .map(|s| !rollout(&twin, None, *s).success)
        .collect();
    let twin_verdict = arm_from_telemetry_with_outcomes(
        &twin_episodes,
        &twin_truncated,
        &ObstacleArmConfig::default(),
    );
    let ArmVerdict::Refused(ArmRefusal::NoObstacleFound { .. }) = twin_verdict else {
        panic!("the clean twin must refuse with outcomes supplied: {twin_verdict:?}");
    };

    // The synthetic ≈0-floor / anti-falsifier / mismatch arms live in the
    // carrier's own unit tests
    // (outcomes_aware_arming_reports_headroom_and_refuses_only_at_zero):
    // recoverable 0.0 → HeadroomBelowFloor (the ONLY headroom refusal),
    // an intermediate share NEVER refuses (the paper's fixed-rate <0.40
    // skip gate is the forbidden shape), and a truncated slice that does
    // not name every episode asserts loud.
}
