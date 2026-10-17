//! The S1-S5 set-awareness suite (Plan 627 section-3) — the gate reflex
//! probe A2 cannot provide: a working `set_refine` deliberately breaks
//! decoy-invariance, so its OWN gates are these.
//!
//! - S1 reversal capability — a constructible fixture where the independent
//!   scorer picks A from {A,B} and the refined scorer picks B once a decoy
//!   shaped like B joins the set (the Jeva Prop-1 demonstration, modelless
//!   edition). The independent scorer provably cannot do this: its scores
//!   do not depend on the set (asserted as the contrast arm).
//! - S2 permutation invariance — same order, bit-identical; permuted order,
//!   agreement to float precision (peer-order addition — the Plan 354 G1
//!   law) with the same argmax winner.
//! - S3 sigmoid-not-softmax — gates in (0,1), monotone in dot(x,k), and
//!   the denominator-free law: high-attendance options keep independent
//!   weights whose sum exceeds 1 (softmax would force sum = 1).
//! - S4 immutability + bound — option rows byte-identical before/after;
//!   the refined-state displacement bounded by (gamma/n) * sum w_i *
//!   (||k_i|| + ||x||).
//! - S5 gamma=0 byte-identity — the OFF posture is a verbatim copy.
//!
//! Plus the d=256 smoke (the reflex EMBED_DIM — the wide kernel path this
//! plan exists beside) and the 100-call stability arm.

#![cfg(feature = "set_refine")]
#![allow(clippy::float_cmp)]

use katgpt_core::set_refine::{SetRefineConfig, SetRefineScratch, gate_weight, refine_state_into};

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for m in 0..a.len() {
        s += a[m] * b[m];
    }
    s
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// argmax with the lowest-index tie-break (the state_option_scoring law).
fn argmax_lowest(scores: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, s) in scores.iter().enumerate() {
        if *s > scores[best] {
            best = i;
        }
    }
    best
}

// ─────────────────────────────────────────────────────────────────────
// The S1 fixture (verified numerically at design time; f32 margins are
// comfortable: ~0.04 on {A,B}, ~0.08 on {A,B,D})
//
// x = e1; A = rot(+25 deg), B = rot(-35 deg), D = rot(-45 deg) in the
// (d0,d1)-plane; tau = 0.5, gamma = 1.0. D sits BEHIND B (further from x,
// lower independent score, near-B direction): B beats D structurally
// (higher independent score AND the nearer cluster center), and D's mass
// pulls the state toward B's neighborhood enough to flip A -> B.
// Working envelope: tau 0.4-0.7, gamma 0.8-1.5 (gamma >= 2 over-refines
// and flips {A,B} directly — documented, not a defect).
// ─────────────────────────────────────────────────────────────────────

const TAU: f32 = 0.5;
const GAMMA: f32 = 1.0;
const D: usize = 4;

fn s1_state() -> Vec<f32> {
    vec![1.0, 0.0, 0.0, 0.0]
}
fn s1_a() -> Vec<f32> {
    vec![0.9063078, 0.4226183, 0.0, 0.0] // rot(+25 deg)
}
fn s1_b() -> Vec<f32> {
    vec![0.8191521, -0.5735764, 0.0, 0.0] // rot(-35 deg)
}
fn s1_d() -> Vec<f32> {
    use std::f32::consts::FRAC_1_SQRT_2;
    vec![FRAC_1_SQRT_2, -FRAC_1_SQRT_2, 0.0, 0.0] // rot(-45 deg)
}
fn s1_cfg() -> SetRefineConfig {
    SetRefineConfig::new(TAU, GAMMA, D)
}

fn refine(x: &[f32], opts: &[&[f32]], cfg: &SetRefineConfig) -> Vec<f32> {
    let flat: Vec<f32> = opts.iter().flat_map(|o| o.iter().copied()).collect();
    let mut out = vec![0.0f32; cfg.d];
    let mut sc = SetRefineScratch::new(cfg.d, opts.len());
    refine_state_into(x, &flat, &mut out, cfg, &mut sc).unwrap();
    out
}

// ─────────────────────────────────────────────────────────────────────
// S1 — reversal capability
// ─────────────────────────────────────────────────────────────────────

/// The capability test: the independent scorer picks A on BOTH sets (its
/// scores are set-independent — the Prop-1 impossibility, asserted); the
/// refined scorer picks A from {A,B} and B from {A,B,D}. The decoy never
/// wins. RED on any scorer that cannot reverse set-dependence.
#[test]
fn s1_set_dependent_reversal() {
    let x = s1_state();
    let a = s1_a();
    let b = s1_b();
    let d = s1_d();
    let cfg = s1_cfg();

    // Contrast arm: the INDEPENDENT scorer cannot reverse — dot(x, .)
    // scores do not read the set at all.
    let indep_ab = [dot(&x, &a), dot(&x, &b)];
    let indep_abd = [dot(&x, &a), dot(&x, &b), dot(&x, &d)];
    assert_eq!(
        argmax_lowest(&indep_ab),
        0,
        "independent: A must win on {{A,B}}"
    );
    assert_eq!(
        argmax_lowest(&indep_abd),
        0,
        "independent: A must win on {{A,B,D}} — an independent scorer CANNOT reverse (Prop 1)"
    );

    // Refined scorer, {A,B}: A wins.
    let x2 = refine(&x, &[&a, &b], &cfg);
    let s_ab = [dot(&x2, &a), dot(&x2, &b)];
    assert_eq!(
        argmax_lowest(&s_ab),
        0,
        "refined {{A,B}}: A must win (margin {})",
        s_ab[0] - s_ab[1]
    );

    // Refined scorer, {A,B,D}: B wins, the decoy never does.
    let x3 = refine(&x, &[&a, &b, &d], &cfg);
    let s_abd = [dot(&x3, &a), dot(&x3, &b), dot(&x3, &d)];
    assert_eq!(
        argmax_lowest(&s_abd),
        1,
        "refined {{A,B,D}}: B must win — the set-dependent REVERSAL (A={:.4} B={:.4} D={:.4})",
        s_abd[0],
        s_abd[1],
        s_abd[2]
    );
}

/// Adding the decoy must not merely narrow the gap — the A-vs-B order
/// itself must FLIP (the reversal is ordinal, not scalar).
#[test]
fn s1_reversal_is_ordinal() {
    let x = s1_state();
    let a = s1_a();
    let b = s1_b();
    let d = s1_d();
    let cfg = s1_cfg();
    let x2 = refine(&x, &[&a, &b], &cfg);
    let x3 = refine(&x, &[&a, &b, &d], &cfg);
    let ab2 = dot(&x2, &a) - dot(&x2, &b);
    let ab3 = dot(&x3, &a) - dot(&x3, &b);
    assert!(
        ab2 > 0.01,
        "{{A,B}} margin must be comfortably positive: {ab2}"
    );
    assert!(
        ab3 < -0.01,
        "{{A,B,D}} margin must be comfortably negative: {ab3}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// S2 — permutation invariance
// ─────────────────────────────────────────────────────────────────────

/// Same order twice: bit-identical (the deterministic-fold law).
#[test]
fn s2_same_order_bit_identical() {
    let x = s1_state();
    let a = s1_a();
    let b = s1_b();
    let d = s1_d();
    let cfg = s1_cfg();
    let flat1: Vec<f32> = [&a[..], &b[..], &d[..]].concat();
    let flat2: Vec<f32> = [&a[..], &b[..], &d[..]].concat();
    let mut o1 = vec![0.0f32; D];
    let mut o2 = vec![0.0f32; D];
    let mut sc1 = SetRefineScratch::new(D, 3);
    let mut sc2 = SetRefineScratch::new(D, 3);
    refine_state_into(&x, &flat1, &mut o1, &cfg, &mut sc1).unwrap();
    refine_state_into(&x, &flat2, &mut o2, &cfg, &mut sc2).unwrap();
    assert_eq!(
        bits(&o1),
        bits(&o2),
        "S2: same inputs, bit-identical refined state"
    );

    // And across a REUSED scratch (the steady-state path).
    let mut o3 = vec![0.0f32; D];
    refine_state_into(&x, &flat1, &mut o3, &cfg, &mut sc1).unwrap();
    assert_eq!(
        bits(&o1),
        bits(&o3),
        "S2: scratch reuse stays bit-identical"
    );
}

/// Permuted option order: float-precision agreement (peer-order addition)
/// and the same argmax winner, across all 6 permutations of a 3-option set.
#[test]
fn s2_permuted_options_float_precision() {
    let x = s1_state();
    let opts: Vec<Vec<f32>> = vec![s1_a(), s1_b(), s1_d()];
    let cfg = s1_cfg();
    let base = refine(&x, &[&opts[0], &opts[1], &opts[2]], &cfg);
    for perm in [
        [0usize, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let rows: Vec<&[f32]> = perm.iter().map(|i| opts[*i].as_slice()).collect();
        let xp = refine(&x, &rows, &cfg);
        let mut max_drift: f32 = 0.0;
        for m in 0..D {
            max_drift = max_drift.max((base[m] - xp[m]).abs());
        }
        assert!(
            max_drift <= 1e-6,
            "S2: perm {:?} drift {max_drift} exceeds 1e-6",
            perm
        );
        // Same winner under every permutation (score the base set order).
        let sb: Vec<f32> = opts.iter().map(|o| dot(&xp, o)).collect();
        assert_eq!(
            argmax_lowest(&sb),
            1,
            "S2: perm {:?} winner must stay B",
            perm
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// S3 — sigmoid, never softmax
// ─────────────────────────────────────────────────────────────────────

/// Gates live in (0,1), are monotone in dot(x,k), and stay INDEPENDENT:
/// a set of high-attendance options has sum-of-gates far above 1 (softmax
/// would force exactly 1 — the denominator-free law).
#[test]
fn s3_gates_sigmoid_not_softmax() {
    let x = s1_state();
    let a = s1_a();
    let b = s1_b();
    let d = s1_d();
    let gates = [
        gate_weight(&x, &a, TAU),
        gate_weight(&x, &b, TAU),
        gate_weight(&x, &d, TAU),
    ];
    for g in gates {
        assert!(g > 0.0 && g < 1.0, "S3: gate {g} outside (0,1)");
    }
    // Monotone: dot(x,A) > dot(x,B) > dot(x,D) implies w_A > w_B > w_D.
    assert!(
        gates[0] > gates[1] && gates[1] > gates[2],
        "S3: gates must be monotone in dot"
    );
    // Denominator-free: three high-attendance options each above 0.7 —
    // their sum exceeds 1 by far; a softmax spelling would sum to exactly 1.
    assert!(
        gates.iter().all(|g| *g > 0.7),
        "S3: fixture must be high-attendance"
    );
    assert!(
        gates.iter().sum::<f32>() > 2.0,
        "S3: sum of independent gates must exceed 1 by far (softmax sums to 1)"
    );
}

// ─────────────────────────────────────────────────────────────────────
// S4 — immutability + displacement bound
// ─────────────────────────────────────────────────────────────────────

/// Option rows are byte-identical before/after the call (read-only by
/// construction — the suite pins it), and the refined-state displacement
/// obeys ||x' - x|| <= (gamma/n) * sum_i w_i * (||k_i|| + ||x||).
#[test]
fn s4_options_immutable_and_displacement_bounded() {
    let x = s1_state();
    let a = s1_a();
    let b = s1_b();
    let d = s1_d();
    let cfg = s1_cfg();
    let flat: Vec<f32> = [&a[..], &b[..], &d[..]].concat();
    let before: Vec<u32> = flat.iter().map(|v| v.to_bits()).collect();
    let mut out = vec![0.0f32; D];
    let mut sc = SetRefineScratch::new(D, 3);
    refine_state_into(&x, &flat, &mut out, &cfg, &mut sc).unwrap();
    let after: Vec<u32> = flat.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        before, after,
        "S4: option rows must be byte-identical after the call"
    );

    // Bound (f64 accumulation; gate_weight matches the kernel's sigmoid to ~1 ULP).
    let n = 3 + 1;
    let mut bound = 0.0f64;
    for k in [&a[..], &b[..], &d[..]] {
        let w = gate_weight(&x, k, TAU) as f64;
        bound += w * ((norm(k) as f64) + (norm(&x) as f64));
    }
    bound *= (GAMMA as f64) / (n as f64);
    let disp = norm(&(0..D).map(|m| out[m] - x[m]).collect::<Vec<_>>()) as f64;
    assert!(
        disp <= bound + 1e-6,
        "S4: displacement {disp} exceeds bound {bound}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// S5 — gamma = 0 byte-identity (the OFF posture)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn s5_gamma_zero_byte_identity() {
    let x = s1_state();
    let flat: Vec<f32> = [&s1_a()[..], &s1_b()[..], &s1_d()[..]].concat();
    let mut out = vec![7.0f32; D];
    let cfg = SetRefineConfig::new(TAU, 0.0, D);
    let mut sc = SetRefineScratch::new(D, 3);
    refine_state_into(&x, &flat, &mut out, &cfg, &mut sc).unwrap();
    assert_eq!(bits(&out), bits(&x), "S5: gamma=0 must be a verbatim copy");
}

// ─────────────────────────────────────────────────────────────────────
// d = 256 — the wide kernel path (the reflex EMBED_DIM)
// ─────────────────────────────────────────────────────────────────────

/// Deterministic LCG fixture (the Plan 354 G1 pattern).
fn lcg_vec(seed: &mut u64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 33) as f32) / (u32::MAX >> 1) as f32 - 1.0
        })
        .collect()
}

#[test]
fn d256_refine_finite_permutation_stable() {
    const D256: usize = 256;
    let mut seed = 0x05EE_D627_u64;
    let x = lcg_vec(&mut seed, D256);
    let mut opts: Vec<Vec<f32>> = Vec::with_capacity(8);
    for _ in 0..8 {
        let mut v = lcg_vec(&mut seed, D256);
        // normalize each option (unit rows — embedding-like)
        let nrm = norm(&v);
        for e in v.iter_mut() {
            *e /= nrm;
        }
        opts.push(v);
    }
    let cfg = SetRefineConfig::new(1.0, 0.5, D256);
    let flat: Vec<f32> = opts.concat();
    let mut out = vec![0.0f32; D256];
    let mut sc = SetRefineScratch::new(D256, 8);
    refine_state_into(&x, &flat, &mut out, &cfg, &mut sc).unwrap();
    assert!(
        out.iter().all(|v| v.is_finite()),
        "d256: refined state must be finite"
    );

    // Determinism: rerun bit-identical.
    let mut out2 = vec![0.0f32; D256];
    refine_state_into(&x, &flat, &mut out2, &cfg, &mut sc).unwrap();
    assert_eq!(bits(&out), bits(&out2), "d256: rerun bit-identical");

    // Permutation drift to float precision, same argmax winner.
    let scores: Vec<f32> = opts.iter().map(|o| dot(&out, o)).collect();
    let winner = argmax_lowest(&scores);
    let mut perm: Vec<&[f32]> = opts.iter().map(|o| o.as_slice()).collect();
    perm.reverse();
    let xp = refine(&x, &perm, &cfg);
    let mut max_drift = 0.0f32;
    for m in 0..D256 {
        max_drift = max_drift.max((out[m] - xp[m]).abs());
    }
    assert!(
        max_drift <= 1e-6,
        "d256: permuted drift {max_drift} exceeds 1e-6"
    );
    let scores_p: Vec<f32> = opts.iter().map(|o| dot(&xp, o)).collect();
    assert_eq!(
        argmax_lowest(&scores_p),
        winner,
        "d256: permuted winner must match"
    );

    // gamma=0 verbatim copy at the wide dim too.
    let cfg_off = SetRefineConfig::new(1.0, 0.0, D256);
    let mut out0 = vec![0.0f32; D256];
    refine_state_into(&x, &flat, &mut out0, &cfg_off, &mut sc).unwrap();
    assert_eq!(bits(&out0), bits(&x), "d256: gamma=0 verbatim copy");
}

// ─────────────────────────────────────────────────────────────────────
// Stability (G4 correctness-side: 100 calls, deterministic, finite —
// the wrapper performs no allocation with a pre-sized scratch by
// construction: copy_from_slice + the kernel's caller-owned buffers)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn hundred_calls_stable_and_finite() {
    let x = s1_state();
    let a = s1_a();
    let b = s1_b();
    let d = s1_d();
    let cfg = s1_cfg();
    let flat: Vec<f32> = [&a[..], &b[..], &d[..]].concat();
    let mut out = vec![0.0f32; D];
    let mut sc = SetRefineScratch::new(D, 3);
    refine_state_into(&x, &flat, &mut out, &cfg, &mut sc).unwrap();
    let first: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
    for _ in 0..99 {
        refine_state_into(&x, &flat, &mut out, &cfg, &mut sc).unwrap();
    }
    let last: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
    assert_eq!(first, last, "100 calls: outputs must be bit-stable");
    assert!(out.iter().all(|v| v.is_finite()));
}
