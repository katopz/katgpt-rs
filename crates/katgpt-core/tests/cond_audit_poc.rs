//! Issue 719 T1 — conditioning-consistency audit PoC gates
//! (`.benchmarks/700_cond_audit_poc.md`).
//!
//! Falsifiable shape:
//! - the calibrated-zero (compression-off) control arm must measure exactly
//!   0.0 KL / 0 flips / PASS;
//! - planted logit corruption in the student arm must EXCEED the Pinsker TV
//!   threshold and FAIL the verdict (G8 non-vacuity — an audit that cannot
//!   fail proves nothing; Bench 804 gate-9 lesson), while the control arm
//!   stays at zero in the same run;
//! - the measured KL must be monotone in corruption magnitude;
//! - outputs must be bit-identical across 3 runs (determinism);
//! - the audit's own math must stay within a MEASURED multiple of the paired
//!   forwards it rides on (G2, release-gated).
//!
//! Opt-in POC discipline (Issue 719): no live consumer, no GOAT promotion
//! claim, no citation of the paper's drift figures — our numbers are ours.

#![cfg(feature = "cond_audit")]

use katgpt_core::cond_audit::{audit_conditioning, pinsker_tv_bound, AuditReport, CondAuditConfig};

const VOCAB: usize = 512;
/// Hidden width of the fixture's vocab projection (see `teacher_forward`).
const HID: usize = 16;
/// 8 junctions scattered across decode positions.
const POSITIONS: [u32; 8] = [0, 7, 13, 64, 100, 255, 300, 511];

/// Deterministic synthetic logit: SplitMix-style finalizer keyed by
/// (position, token) → [-1, 1). Seeded, CPU-only, zero runtime randomness.
fn synth_logit(pos: u32, token: usize) -> f32 {
    let mut h = (pos as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= (token as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    let unit = (h >> 40) as f32 / (1u32 << 23) as f32; // [0, 1)
    unit * 2.0 - 1.0
}

/// Teacher arm: the full-context forward (the reference conditioning).
///
/// Deliberately NOT a table lookup: each logit is a HID-dim deterministic
/// projection of a per-position hidden vector — the minimal stand-in for the
/// vocab-projection matvec every real forward pays. (The first fixture draft
/// was a pure lookup; its forwards were so cheap that the audit's own
/// O(vocab) KL passes dominated the G2 ratio — a fixture artifact, not an
/// audit cost. No serving site has a zero-compute forward.)
fn teacher_forward(pos: u32, out: &mut [f32]) {
    let mut x = [0.0f32; HID];
    for (d, xd) in x.iter_mut().enumerate() {
        *xd = synth_logit(pos, d);
    }
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (d, xd) in x.iter().enumerate() {
            acc += synth_logit(pos.wrapping_add(i as u32), d) * xd;
        }
        *o = acc;
    }
}

/// Student arm with additive conditioning noise of the given scale — the
/// proxy for "semantically compressed context" (deterministic, seeded).
fn noisy_student(noise_scale: f32) -> impl FnMut(u32, &mut [f32]) {
    move |pos: u32, out: &mut [f32]| {
        teacher_forward(pos, out);
        for (i, o) in out.iter_mut().enumerate() {
            *o += noise_scale * synth_logit(pos.wrapping_add(0x1000), i.wrapping_add(17));
        }
    }
}

fn argmax_of(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best
}

/// The compression failure mode: the student's context dropped what the
/// teacher's argmax token depends on — pull that token's logit down hard so
/// the student loses mass the teacher carries (the deficit side).
fn deficit_student() -> impl FnMut(u32, &mut [f32]) {
    move |pos: u32, out: &mut [f32]| {
        teacher_forward(pos, out);
        let top = argmax_of(out);
        out[top] -= 12.0;
    }
}

fn run(student: impl FnMut(u32, &mut [f32]), cfg: &CondAuditConfig) -> AuditReport {
    audit_conditioning(&POSITIONS, VOCAB, student, teacher_forward, cfg)
}

/// G8 control half: compression-off (bit-identical arms) is the numeric
/// floor — exactly zero KL, zero flips, verdict PASS.
#[test]
fn calibrated_zero_arm_is_exactly_zero_and_passes() {
    let cfg = CondAuditConfig::default();
    let r = audit_conditioning(&POSITIONS, VOCAB, teacher_forward, teacher_forward, &cfg);
    assert_eq!(r.junctions, POSITIONS.len());
    assert_eq!(r.eps_kl, 0.0, "bit-identical arms must sum to exactly 0.0");
    assert_eq!(r.tv_bound, 0.0);
    assert_eq!(r.tv_bound_chain, 0.0);
    assert_eq!(r.greedy_flips, 0);
    assert_eq!(r.max_junction_kl, 0.0);
    assert!(r.verdict_pass);
    assert!(r.per_junction_kl.iter().all(|&k| k == 0.0));
    println!(
        "calibrated-zero arm: eps_kl = {} nats (exact), flips = {}, verdict PASS",
        r.eps_kl, r.greedy_flips
    );
}

/// G8 non-vacuity (the Bench 804 gate-9 lesson): planted logit corruption in
/// the student arm must EXCEED the TV threshold and FAIL the verdict while
/// the calibrated control arm stays at zero in the same run.
#[test]
fn g8_planted_corruption_exceeds_threshold_calibrated_stays_zero() {
    let cfg = CondAuditConfig::default();
    let control = audit_conditioning(&POSITIONS, VOCAB, teacher_forward, teacher_forward, &cfg);
    assert_eq!(control.eps_kl, 0.0);
    assert!(control.verdict_pass, "control arm must pass");

    let treated = run(deficit_student(), &cfg);
    println!(
        "deficit arm: eps_kl = {:.6} nats, tv_bound = {:.6}, tv_chain = {:.6}, flips = {}/{}",
        treated.eps_kl, treated.tv_bound, treated.tv_bound_chain, treated.greedy_flips, treated.junctions
    );
    assert!(
        treated.eps_kl > 0.1,
        "deficit KL {} must clear the 0.1-nat non-vacuity floor (control {})",
        treated.eps_kl,
        control.eps_kl
    );
    assert!(
        treated.tv_bound > cfg.tv_threshold,
        "TV bound {} must EXCEED the threshold {} — the audit must be able to FAIL",
        treated.tv_bound,
        cfg.tv_threshold
    );
    assert!(!treated.verdict_pass, "G8: verdict must flip to FAIL under corruption");
    assert!(
        treated.greedy_flips >= 1,
        "a 12-nat argmax deficit must move at least one greedy argmax"
    );
}

/// The measured gap must grow monotonically with corruption magnitude — a
/// graded response, not a step function.
#[test]
fn kl_is_monotone_in_corruption_magnitude() {
    let cfg = CondAuditConfig::default();
    let scales = [0.0f32, 0.25, 1.0, 4.0];
    let mut prev = -1.0f32;
    let mut observed = Vec::with_capacity(scales.len());
    for &s in &scales {
        let r = run(noisy_student(s), &cfg);
        println!("noise {s:>4}: eps_kl = {:.6} nats, tv_bound = {:.6}", r.eps_kl, r.tv_bound);
        assert!(
            r.eps_kl > prev,
            "eps_kl must increase with noise scale: scale {s} gave {} after {prev}",
            r.eps_kl
        );
        prev = r.eps_kl;
        observed.push(r.eps_kl);
    }
    assert!(observed[0] == 0.0, "zero noise is the calibrated arm");
    assert!(prev > 0.5, "the loudest arm must be clearly nonzero, got {prev}");
}

/// Same inputs → bit-identical outputs across 3 runs (Issue 719 T1).
#[test]
fn deterministic_bit_identical_x3() {
    let cfg = CondAuditConfig::default();
    let r1 = run(deficit_student(), &cfg);
    for run_idx in 0..2 {
        let r = run(deficit_student(), &cfg);
        assert_eq!(r.eps_kl.to_bits(), r1.eps_kl.to_bits(), "run {run_idx}: eps_kl");
        assert_eq!(r.tv_bound.to_bits(), r1.tv_bound.to_bits(), "run {run_idx}: tv_bound");
        assert_eq!(
            r.tv_bound_chain.to_bits(),
            r1.tv_bound_chain.to_bits(),
            "run {run_idx}: tv_bound_chain"
        );
        assert_eq!(
            r.max_junction_kl.to_bits(),
            r1.max_junction_kl.to_bits(),
            "run {run_idx}: max_junction_kl"
        );
        assert_eq!(r.greedy_flips, r1.greedy_flips, "run {run_idx}: flips");
        assert_eq!(r.junctions, r1.junctions);
        assert_eq!(r.per_junction_kl.len(), r1.per_junction_kl.len());
        for (i, (a, b)) in r.per_junction_kl.iter().zip(&r1.per_junction_kl).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "run {run_idx} junction {i}");
        }
    }
}

/// Report internal consistency: eps_kl is the junction sum, max is the max,
/// and both Pinsker forms match their closed formulas.
#[test]
fn report_internal_consistency_and_pinsker_math() {
    let cfg = CondAuditConfig::default();
    let r = run(deficit_student(), &cfg);
    assert_eq!(r.junctions, POSITIONS.len());
    assert_eq!(r.per_junction_kl.len(), POSITIONS.len());

    let sum: f32 = r.per_junction_kl.iter().sum();
    assert_eq!(sum.to_bits(), r.eps_kl.to_bits(), "eps_kl is the ordered junction sum");
    let mx = r.per_junction_kl.iter().copied().fold(0.0f32, f32::max);
    assert_eq!(mx.to_bits(), r.max_junction_kl.to_bits());

    assert_eq!(r.tv_bound.to_bits(), pinsker_tv_bound(r.eps_kl).to_bits());
    let expected_chain = (POSITIONS.len() as f32 * r.eps_kl / 2.0).sqrt();
    assert_eq!(r.tv_bound_chain.to_bits(), expected_chain.to_bits());
    assert_eq!(r.verdict_pass, r.tv_bound <= cfg.tv_threshold);

    // Cauchy–Schwarz (f32-rounded): the chain form must UPPER-bound the
    // triangle sum of the exact per-junction Pinsker bounds, and K >= 1
    // makes the chain form looser than the single-junction form.
    let triangle_sum: f32 = r.per_junction_kl.iter().map(|&k| pinsker_tv_bound(k)).sum();
    assert!(
        r.tv_bound_chain >= triangle_sum * (1.0 - 1e-3) || r.eps_kl == 0.0,
        "chain bound {} must dominate the triangle sum {} (Cauchy–Schwarz)",
        r.tv_bound_chain,
        triangle_sum
    );
    assert!(
        r.tv_bound_chain >= r.tv_bound,
        "K >= 1: chain bound {} must be looser than the single-junction form {}",
        r.tv_bound_chain,
        r.tv_bound
    );
}

/// G2 (Issue 719): the audit's own math must stay within a MEASURED multiple
/// of the paired forwards it rides on. Interleaved median-of-ratios (the
/// Bench 728 discipline) — ratios share thermal/contention state where
/// absolute wall-clock does not. Release-only: debug timing measures an
/// unoptimised binary (the house `#[cfg_attr(debug_assertions, ignore)]`
/// pattern; this box runs load 60+).
///
/// Honest caveat (bench doc): the fixture's forwards are the MINIMAL real
/// shape (a 16-dim vocab projection per token) — real transformer forwards
/// cost orders of magnitude more, so the measured ratio upper-bounds the
/// production fraction. The gate catches the audit accidentally doing
/// something pathological (per-element allocs, O(vocab²) passes) relative
/// to even the cheapest honest forward.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn g2_audit_overhead_ratio_measured() {
    use std::hint::black_box;
    use std::time::Instant;

    const REPS: usize = 15;
    const INNER: usize = 25;
    let cfg = CondAuditConfig::default();
    let mut ratios: Vec<f64> = Vec::with_capacity(REPS);

    for _ in 0..REPS {
        // Arm A — paired forwards only (what the audit must stay cheap vs).
        let mut student = noisy_student(1.0);
        let mut s_buf = vec![0.0f32; VOCAB];
        let mut t_buf = vec![0.0f32; VOCAB];
        let t0 = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..INNER {
            for &pos in &POSITIONS {
                student(pos, &mut s_buf);
                teacher_forward(pos, &mut t_buf);
                acc += s_buf[0] + t_buf[0];
            }
        }
        black_box(acc);
        let forwards = t0.elapsed().as_secs_f64();

        // Arm B — the full audit (forwards + KL math + report allocs).
        let t1 = Instant::now();
        let mut eps = 0.0f32;
        for _ in 0..INNER {
            let r = audit_conditioning(&POSITIONS, VOCAB, noisy_student(1.0), teacher_forward, &cfg);
            eps += r.eps_kl;
        }
        black_box(eps);
        let audit = t1.elapsed().as_secs_f64();

        ratios.push(audit / forwards);
    }

    ratios.sort_by(|a, b| katgpt_core::float_order::asc_f64(*a, *b));
    let median = ratios[REPS / 2];
    println!(
        "g2 cond_audit: median audit/forward ratio = {median:.3} (overhead fraction {:.3}) over {REPS} interleaved reps x {INNER} calls",
        median - 1.0
    );
    assert!(
        median <= 4.0,
        "audit overhead {median:.3}x of the paired forwards exceeds the 4.0x measured budget"
    );
}
