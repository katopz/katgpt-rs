#![cfg(feature = "gain_cost_halt")]
//! Issue 993 T2–T4 — the LP-authored overlay arm's gate battery
//! (Research 664): the min-L1 authoring LP (PDHG, `lp_authoring`) authors
//! the carrier repair on the SAME seeded-degeneracy world as the 980
//! battery (shared fixture: `wsp980_common`), and the composed rank-1
//! overlay passes the existing gate battery arms with LP-authored v.
//!
//! # Pre-registered constants (this file IS the pre-registration)
//!
//! The fixture's own geometry pins the knobs (derivations in the test
//! bodies; the analytic shape lives in `wsp980_common`'s header):
//!
//! - **Trace labeling.** Fit = eval seeds 0..8, held-out = 8..16 (the
//!   980 pairing grid). Rollouts from the DEGENERATE model record `h1`
//!   (the W2 input the per-column coefficients act through). Labels:
//!   **R** (push) = in-region ∧ carrier ≤ −ρ with ρ = 0.2 (the seized
//!   marching states — carrier magnitude above half-saturation);
//!   **P** (keep) = out-of-region (the carrier is exactly silent there —
//!   the retention law as a constraint); carrier-positive in-region
//!   states are the onset-ambiguity class (the blog's AUC-0.73-at-onset
//!   finding) — recorded, unconstrained. Rows are deduped to distinct
//!   states (LE-byte lexicographic sort — a deterministic total order;
//!   the LP optimum is multiplicity-invariant here, the binding state
//!   pins the coefficient).
//! - **δ = k₀ · min_R|h1[4]| (derived, not hand-typed).** The push target
//!   anchors the authored coefficient to the fixture's own documented
//!   injection magnitude k₀ = 8 (the same constant the lane's targeted
//!   arm uses for its full dose): the LP's kink optimum lands at
//!   `c* = δ/min_R|h1[4]| = k₀` — exact injection offset — with |residual|
//!   ≤ solver noise. WHY EXACT OFFSET: the oscillation statistic is NOT
//!   monotone in the coefficient (fixture finding, measured this session):
//!   a residual r = k₀ − c re-flips the decision NEAR THE GOAL, where the
//!   horizontal pull collapses (gap_h → ≈0.03 at x→GOAL), and a FAILING
//!   ISLAND exists around r ≈ 0.09–0.18 (c ≈ 7.82–7.91) — a c* of 7.895
//!   (δ hand-pinned 2.3) fails there while the lane's own targeted arm at
//!   the same strength fails identically (the LP overlay is
//!   behaviorally-indistinguishable from the targeted arm — the island is
//!   the world's, not the authoring's). At |r| ≤ 0.05 the residual flip
//!   is dead (|gap_v| = 1.6·tanh(0.2027·|r|) < gap_h(≥0.029)) and every
//!   trajectory is monotone-right after ≤ 1 corrective step. The 980 dose
//!   grid {1, 2, 4, 8} never samples the island — recorded here as the
//!   fixture's dose-response fine structure.
//! - **ε = 0.0** — the retention law as a constraint. The fixture's
//!   productive class is exactly carrier-silent (bump = 0 ⇒ h1[4] = 0.0
//!   bitwise), so exact silence is satisfiable by the carrier column and
//!   REFUSES every uniform-mass column (each would violate the band out
//!   of region). A real-model authoring pins ε > 0 to the measured
//!   productive noise floor instead.
//! - **C = 10, max_iter = 20k, tol = 1e-5**; dose grid {0.5, 0.75, 1.0}×
//!   the authored v (the LP's feasibility-optimized magnitude sits at
//!   the window's edge BY DESIGN — min-L1 minimizes the edit — so the
//!   sweep stops at 1×; narrow viable strength bands are the family
//!   trait of Plan 309 / Research 664).
//! - **Matched-K incumbent grid** α ∈ {6.0 … 9.0} on the LP's own
//!   support columns (selection parity); **tie law**: paired LCB95 of
//!   (LP − grid) ≥ 0 keeps the arm (exactly 0 = UNDECIDED); the negative
//!   fires ONLY if the grid STRICTLY wins (LCB95 < 0) → demotion law:
//!   record the negative, strike the arm from the lane docs, lane's
//!   reasoning-authored path untouched (Issue-990 precedent).
//! - **Steering probe**: constant push at {0.5, 1, 2}× the LP edit's
//!   signed mean push over the fit push rows; recovered ⟺ (osc == 0 ∧
//!   success ≥ 0.99). Decision law (module doc): probe steering first;
//!   bake only on selectivity need or injection-cost refusal.

mod wsp980_common;

use katgpt_core::blake3_seed;
use katgpt_core::weight_edit_overlay::lp_authoring::{
    LpAuthoringConfig, LpAuthoringError, TraceSet, compose_direction_overlay,
    solve_min_l1_authoring_lp, truncate_small,
};
use katgpt_core::weight_edit_overlay::{
    ArmVerdict, ObstacleArmConfig, ProvenanceTrailer, WeightEditArtifact, WeightEditOverlay,
    arm_from_telemetry, norm_matched_random_direction, paired_lcb95,
};
use wsp980_common::*;

// ─────────────────────────────────────────────────────────────────────
// Pinned constants (the pre-registration above)
// ─────────────────────────────────────────────────────────────────────

const LP_EPSILON: f32 = 0.0;
const LP_C: f32 = 10.0;
const LP_MAX_ITER: u32 = 20_000;
const LP_TOL: f32 = 1e-5;
const CARRIER_RHO: f32 = 0.2;
const SEP_TOL: f32 = 0.05;
const KEEP_TOL: f32 = 1e-3;
const LP_DOSE_GRID: [f32; 3] = [0.5, 0.75, 1.0];
const GRID_ALPHA: [f32; 7] = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0];
const SPARSITY_CAP_K: usize = 8;
const RANDOM_ARMS_993: usize = 16;

// ─────────────────────────────────────────────────────────────────────
// Trace authoring + helpers
// ─────────────────────────────────────────────────────────────────────

struct LabeledTraces {
    push_rows: Vec<f32>,
    n_push: usize,
    keep_rows: Vec<f32>,
    n_keep: usize,
}

/// Deterministic total order over trace rows (`total_cmp` — a total order
/// over all f32 including NaN; no HashMap order anywhere).
fn row_cmp(a: &[f32], b: &[f32]) -> std::cmp::Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        match x.total_cmp(y) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    std::cmp::Ordering::Equal
}

/// Record the labeled activation traces from the DEGENERATE model's
/// rollouts (seeds `seeds`), label by geometry, dedupe to distinct
/// states. Fit: seeds 0..8. Held-out: seeds 8..16.
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

/// The pre-registered derived push target: `δ = INJECT_K0 · min_R|h1[4]|`
/// (sequential scan, deterministic) — shared by the solve and the
/// separation checks.
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

/// Solve the pinned authoring LP on a labeled trace set. δ is DERIVED
/// per-solve (see `derived_delta`): the LP's kink optimum then lands at
/// c* ≈ k₀, the exact injection offset, island-free (the failing island
/// around residual r ≈ 0.09–0.18 is unreachable at |r| ≤ solver noise).
fn solve_lp(
    t: &LabeledTraces,
) -> Result<katgpt_core::weight_edit_overlay::lp_authoring::LpAuthoringSolution, LpAuthoringError> {
    // No push rows → skip the derivation (there is nothing to anchor) and
    // let the solve refuse loud (the unconstrained-degenerate refusal).
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

/// Compose the LP overlay (single-column edits over the authored support,
/// u = the lane's targeted direction) at a dose scaling of `v`.
fn lp_overlay(v: &[f32], dose: f32) -> WeightEditOverlay {
    let scaled: Vec<f32> = v.iter().map(|x| x * dose).collect();
    compose_direction_overlay(&Fixture::targeted_direction(), &scaled, LAYER_W2, H)
        .expect("LP overlay composes")
}

/// Held-out separation: (push_frac, keep_frac) over a labeled set, at the
/// pre-registered tolerances (SEP_TOL absorbs the solver's convergence
/// tail; KEEP_TOL pins the retention-exactness arm).
fn separation_scores(v: &[f32], t: &LabeledTraces, delta: f32) -> (f64, f64) {
    let mut ok_push = 0u64;
    for i in 0..t.n_push {
        let row = &t.push_rows[i * H..(i + 1) * H];
        let mut acc = 0.0f32;
        for (x, w) in v.iter().zip(row.iter()) {
            acc += x * w;
        }
        // The push threshold is the DERIVED binding target (the LP's kink
        // optimum lands v·h = −δ at the weakest seized state).
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

/// The steering probe's per-arm stats over every eval seed's starts.
fn steering_stats(model: &Fixture, y2_bias: f32) -> (f32, f32) {
    let mut osc = 0usize;
    let mut succ = 0usize;
    let mut n = 0usize;
    for s in 0..EVAL_SEEDS as u64 {
        for start in seeded_starts(s) {
            let r = rollout_with_bias(model, start, y2_bias);
            if episode_oscillates(&r.actions) {
                osc += 1;
            }
            if r.success {
                succ += 1;
            }
            n += 1;
        }
    }
    (osc as f32 / n as f32, succ as f32 / n as f32)
}

// ─────────────────────────────────────────────────────────────────────
// T2 — synthetic fit + battery fit (a)–(d)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t2_lp_authored_overlay_passes_the_battery() {
    let fit = record_traces(0..8);
    let held = record_traces(8..16);
    assert!(
        fit.n_push > 0 && fit.n_keep > 0,
        "trace authoring must produce both classes"
    );
    let sol = solve_lp(&fit).expect("LP solve");

    // (a) sparsity ≤ K; the support excludes the isolation column and is
    // carried by the carrier column (the formulation forces it — push
    // mass only exists on carrier-correlated columns and the exact-silence
    // keep band refuses every uniform-mass column).
    let max_abs = sol.v.iter().copied().fold(0.0f32, f32::max).abs();
    assert!(max_abs > 0.0, "solve must not return an empty direction");
    let v = truncate_small(sol.v.clone(), 1e-3 * max_abs).expect("floor");
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

    // (b) held-out separation (rows the solve never saw).
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

    // (c) determinism: two solves → bit-identical v → identical overlay +
    // artifact digests (the committed, hash-comparable result).
    let sol2 = solve_lp(&fit).expect("re-solve");
    assert!(
        sol.v
            .iter()
            .zip(sol2.v.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "two solves on the same pinned traces must be bit-identical"
    );
    let ov1 = lp_overlay(&v, 1.0);
    let ov2 = lp_overlay(
        &truncate_small(sol2.v.clone(), 1e-3 * max_abs).unwrap(),
        1.0,
    );
    assert_eq!(ov1.digest(), ov2.digest(), "overlay digests must match");
    let provenance = ProvenanceTrailer {
        prompt_digest: Some(*blake3::hash(b"issue993 lp authoring").as_bytes()),
        ..Default::default()
    };
    let art1 = WeightEditArtifact::mint(&ov1, provenance.clone());
    let art2 = WeightEditArtifact::mint(&ov2, provenance);
    assert_eq!(
        art1.artifact_digest(),
        art2.artifact_digest(),
        "artifact digests must match"
    );

    // (d) the battery arms, LP-authored.
    let deg = build_fixture(true);
    let starts = all_eval_starts();

    // dose-response: monotone non-increasing over the pinned grid, and the
    // full (authored) dose clears to the floor.
    let mut prev = 2.0f32;
    for dose in LP_DOSE_GRID {
        let ov = lp_overlay(&v, dose);
        let (rate, _osc) = obstacle_rate(&deg, Some(&ov), &starts);
        assert!(
            rate <= prev + 1e-6,
            "dose-response must be non-increasing: dose {dose} rate {rate} > prev {prev}"
        );
        prev = rate;
    }
    assert_eq!(prev, 0.0, "full (authored) dose must clear the obstacle");

    // neighbor-column isolation null: a same-magnitude control edit on the
    // isolation column must NOT clear (the edit's mass rides the carrier).
    let mut ctrl = vec![0.0f32; H];
    ctrl[ISOLATION_COL as usize] = v[CARRIER_COL as usize].abs();
    let ov_ctrl = compose_direction_overlay(&Fixture::targeted_direction(), &ctrl, LAYER_W2, H)
        .expect("control overlay composes");
    let (rate_ctrl, _) = obstacle_rate(&deg, Some(&ov_ctrl), &starts);
    assert!(
        rate_ctrl >= 0.5,
        "isolation-column control must not clear: rate {rate_ctrl}"
    );

    // norm-matched random control: ≥16 BLAKE3-seeded arms, u swapped for a
    // norm-matched random direction, paired per-arm (same starts) — the
    // LP direction must strictly win (paired LCB95 > 0).
    let mut deltas = Vec::with_capacity(RANDOM_ARMS_993);
    for arm in 0..RANDOM_ARMS_993 as u64 {
        let seed = blake3_seed(b"issue993/u_control", arm);
        let mut u_rand = vec![0.0f32; H];
        norm_matched_random_direction(seed, H, &mut u_rand);
        let ov_rand = compose_direction_overlay(&u_rand, &v, LAYER_W2, H)
            .expect("random control overlay composes");
        let arm_starts = seeded_starts(arm);
        let (rate_r, _) = obstacle_rate(&deg, Some(&ov_rand), &arm_starts);
        let (rate_l, _) = obstacle_rate(&deg, Some(&ov1), &arm_starts);
        deltas.push(rate_r - rate_l);
    }
    let lcb = paired_lcb95(&deltas).expect("≥2 arms");
    assert!(
        lcb > 0.0,
        "random control must fail where the LP edit clears: LCB95 {lcb}"
    );

    // obstacle pre-screen refusal on the clean twin (the lane's falsifier,
    // re-asserted for the LP-authored posture).
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

    // retention: out-of-region forwards bit-identical (the overlay's
    // carrier-0 no-op law, re-asserted for LP-authored v).
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
// T3a — the fair incumbent: matched-K α-grid, pre-registered tie law
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t3a_matched_k_grid_incumbent_does_not_strictly_win() {
    let fit = record_traces(0..8);
    let sol = solve_lp(&fit).expect("LP solve");
    let max_abs = sol.v.iter().copied().fold(0.0f32, f32::max).abs();
    let v = truncate_small(sol.v.clone(), 1e-3 * max_abs).expect("floor");
    let k = v.iter().filter(|&&x| x != 0.0).count();

    // Selection parity: the incumbent gets the SAME columns as the LP's
    // solution (a separation-per-sparsity metric would be rigged toward
    // the L1 objective by construction).
    let cols: Vec<usize> = (0..H).filter(|&j| v[j] != 0.0).collect();
    assert_eq!(cols.len(), k, "support census must match the nonzero count");

    // Held-out sets, PER SEED (the pairing granularity for the tie law).
    let held_per_seed: Vec<LabeledTraces> = (8..16u64).map(|s| record_traces(s..s + 1)).collect();

    // α-grid incumbent: best held-out separation on that column set
    // (strength swept on the LP's own sign convention; tie → first).
    let sign = v[cols[0]].signum();
    let delta = derived_delta(&fit);
    let mut best = (f64::MIN, GRID_ALPHA[0]);
    let mut best_v = vec![0.0f32; H];
    for &alpha in &GRID_ALPHA {
        let mut cand = vec![0.0f32; H];
        for &c in &cols {
            cand[c] = alpha * sign;
        }
        let (push_frac, keep_frac) = separation_scores(&cand, &held_per_seed[0], delta);
        let score = push_frac + keep_frac;
        if score > best.0 {
            best = (score, alpha);
            best_v = cand;
        }
    }

    // Paired per-seed deltas (LP − grid-best) on held-out separation.
    let mut deltas = Vec::with_capacity(held_per_seed.len());
    for hs in &held_per_seed {
        let (lp_p, lp_k) = separation_scores(&v, hs, delta);
        let (g_p, g_k) = separation_scores(&best_v, hs, delta);
        deltas.push(((lp_p + lp_k) - (g_p + g_k)) as f32);
    }
    let lcb = paired_lcb95(&deltas).expect("8 held seeds");
    // Pre-registered tie law: the negative fires ONLY if the grid STRICTLY
    // wins (LCB95 < 0). LCB95 ≥ 0 = the LP wins or ties ⇒ the arm stays
    // (exactly 0 = UNDECIDED: both arms saturate held-out separation at
    // K=1 here — the honest expected outcome, per the issue's own law).
    assert!(
        lcb >= 0.0,
        "the grid incumbent STRICTLY wins (LCB95 {lcb}) — the demotion law fires"
    );
}

// ─────────────────────────────────────────────────────────────────────
// T3b — the steering-equivalence probe (recorded, not asserted-to-pass)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t3b_steering_probe_records_the_selectivity_need() {
    let fit = record_traces(0..8);
    let sol = solve_lp(&fit).expect("LP solve");

    // The LP edit's SIGNED mean push over the fit push rows (v·h; the edit
    // adds −(v·h)·u, u = −e₁ ⇒ y2[1] += v·h — negative at carrier-negative
    // states, i.e. the repair pushes them toward silence).
    let mut sum = 0.0f32;
    for i in 0..fit.n_push {
        let row = &fit.push_rows[i * H..(i + 1) * H];
        let mut acc = 0.0f32;
        for (x, w) in sol.v.iter().zip(row.iter()) {
            acc += x * w;
        }
        sum += acc;
    }
    let mean_push = sum / fit.n_push as f32;
    assert!(mean_push < 0.0, "the authored edit must push, not pull");

    let deg = build_fixture(true);
    let mut recovered = false;
    let mut window = String::new();
    for m in [0.5f32, 1.0, 2.0] {
        let bias = m * mean_push;
        let (osc, succ) = steering_stats(&deg, bias);
        window += &format!("m={m}: osc={osc:.3} succ={succ:.3}; ");
        if osc == 0.0 && succ >= 0.99 {
            recovered = true;
        }
    }

    // Decision law (module doc): probe steering first; bake only on
    // selectivity need or injection-cost refusal. This fixture's carrier
    // is SIGN-VARYING across the obstacle set (unlike the blog's
    // 89%-active near-constant columns), so no constant push recovers
    // success-without-oscillation — per-state selectivity is load-bearing
    // here ⇒ the bake (the weight edit) is the justified arm. If steering
    // DID recover, this gate reds and the decision law flips to steering.
    assert!(
        !recovered,
        "the constant-push arm recovered the fixture effect — steering window \
         nonempty, the decision law selects the frozen-model quadrant: {window}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// T4 — falsifier arms (each must RED, or the gate is blind)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t4_falsifier_arms_red() {
    let fit = record_traces(0..8);
    let sol = solve_lp(&fit).expect("LP solve");
    let max_abs = sol.v.iter().copied().fold(0.0f32, f32::max).abs();
    let v = truncate_small(sol.v.clone(), 1e-3 * max_abs).expect("floor");

    // (i) unconstrained LP (drop the push class) → v = 0 → LOUD refusal,
    // never a silent empty overlay.
    let keep_only = LabeledTraces {
        push_rows: Vec::new(),
        n_push: 0,
        keep_rows: fit.keep_rows.clone(),
        n_keep: fit.n_keep,
    };
    let err = solve_lp(&keep_only).unwrap_err();
    assert_eq!(err, LpAuthoringError::DegenerateZeroSolution);

    // (ii) the random-u control must FAIL clearance per-arm (the battery's
    // falsifier: if random directions started clearing, the gate is blind).
    let deg = build_fixture(true);
    let mut cleared = 0usize;
    for arm in 0..RANDOM_ARMS_993 as u64 {
        let seed = blake3_seed(b"issue993/u_control", arm);
        let mut u_rand = vec![0.0f32; H];
        norm_matched_random_direction(seed, H, &mut u_rand);
        let ov_rand =
            compose_direction_overlay(&u_rand, &v, LAYER_W2, H).expect("control composes");
        let (rate_r, _) = obstacle_rate(&deg, Some(&ov_rand), &seeded_starts(arm));
        if rate_r == 0.0 {
            cleared += 1;
        }
    }
    assert!(
        cleared * 2 < RANDOM_ARMS_993,
        "{cleared}/{RANDOM_ARMS_993} random-u arms cleared — the falsifier is blind"
    );

    // (iii) the clean twin's pre-screen refusal is asserted in T2 (it is
    // the lane's own battery arm there); the clean twin's refusal is the
    // obstacle-vs-ability law, not an LP-specific falsifier.

    // (iv) a tampered artifact byte refuses thaw (the existing envelope
    // behavior, re-asserted for LP-authored artifacts).
    let ov = lp_overlay(&v, 1.0);
    let provenance = ProvenanceTrailer {
        prompt_digest: Some(*blake3::hash(b"issue993 lp authoring").as_bytes()),
        ..Default::default()
    };
    let art = WeightEditArtifact::mint(&ov, provenance);
    assert!(art.verify_header(), "mint must verify its own header");
    let mut bytes = art.to_bytes();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x01;
    assert!(
        WeightEditArtifact::thaw(&bytes).is_none(),
        "tampered artifact must refuse thaw"
    );
    // And the untampered bytes round-trip (the control for (iv)).
    assert!(
        WeightEditArtifact::thaw(&art.to_bytes()).is_some(),
        "untampered artifact must thaw"
    );
}
