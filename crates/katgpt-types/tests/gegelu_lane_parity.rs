//! Issue 1220 — `gegelu_tanh` lane-parity property test (the falsification arm).
//!
//! Both lanes spell the sigmoid form `g·up·σ(2·inner)` as `1/(1+exp(−2·inner))`
//! (the body since issue 1223, the scalar tail since issue 1220), so a row's
//! answer does not depend on `len % 64`. History: the tail ORIGINALLY spelled
//! the algebraically-equal `0.5·g·(1+tanh inner)`, whose `1+tanh` sum cancels
//! to a rounded zero in the deep negative tail (scalar read −0.0 from g ≈ −5.2
//! while the body kept the true value); the body ORIGINALLY spelled
//! `exp(2x)/(exp(2x)+1)`, which NaNs at exp overflow for |inner| ≳ 44.35.
//! Research 673 §4.1 has the atlas row.
//!
//! Falsification predicate: per-element lane gap bounded by a RELATIVE band.
//! A plain absolute cap cannot serve: the pre-fix gaps (≈7e−8 peak) fit under
//! `1e−7·|up|` — the relative band is what makes the cancellation visible
//! (it failed at a relative 100% pre-fix).
//!
//! Band floor: the exp difference as it propagates through σ. Issue 1223
//! fixed the exp kernel's range reduction (the NEON path truncated with
//! `vrndq_f32`/FRINTZ where nearest was intended — 236 ulps measured vs
//! libm; the post-fix census reads 2 ulps), so the floor is no longer
//! kernel-class: it is the two lanes' ~2-ulp exp difference scaled by the
//! σ propagation factor (sub-ulp mid-band, up to |inner| ulps in the deep
//! tail — see CAP_ULPS below). Any regression — a return to FRINTZ,
//! a fast_tanh spelling (~2.5% relative), or a form divergence (relative
//! 100%) — fails the band by orders of magnitude.
//!
//! Oracle (pinned at 3 significant figures from the verdict-round
//! measurement, Research 673 §4.1 — do not re-derive by hand; the `probe`
//! arm re-prints the table for re-pins; values re-confirmed identical
//! post-1223, the kernel fix moved them only in the 4th+ digit):
//! length 65, up = 1: g −5.0 → SIMD −2.29e−7; −5.2 → −5.69e−8 (onset);
//! −5.5 → −5.93e−9; −6.0 → −8.44e−11; +5.0 → +5.0 (one ulp of 5.0).

use katgpt_types::gegelu_tanh;
use katgpt_types::simd::simd_exp_inplace;

/// Relative lane-parity band. Post-1223 the exp kernel is nearest-rounded
/// (census 2 ulps vs libm; pre-fix 236), so the honest floor is the exp
/// difference as it propagates through σ. dσ/σ = de/(1+e) ≈ ε·min(e, 1):
/// sub-ulp mid-band, but in the deep tail (|e| ≫ 1) the kernel's absolute
/// exp error rides the answer at up to |inner| ulps (measured: gaps ≈ 24
/// ulps of the answer at the grid's deepest g, ≈ 47 with the up = −2 arm).
/// The cap therefore scales with |inner| — the same ε·|inner| association
/// scale the probe reports — with ~4× headroom over the measured worst.
/// Every regression still fails by orders of magnitude: a FRINTZ return
/// (236-ulp exp error → ~24k-ulp-class gaps here), fast_tanh ≈ 2.5%
/// relative, cancellation = relative 100%.
const CAP_ULPS: f32 = 4.0;
/// The exp kernel's own pinned accuracy floor: the measured census class plus
/// headroom. Pre-1223 this was 3.2e−5 (236 ulps) and it DOMINATED this band —
/// the kernel bug's measured class was the honest floor then. Post-fix it
/// sits BELOW the cap, so the cap drives; the constant stays as the kernel's
/// honest record and the regression tripwire.
const EXP_KERNEL_REL: f32 = 3.0e-7;

/// The verdict-round g values (the test's oracle grid, Research 673 §4.1).
const ORACLE_G: [f32; 5] = [-5.0, -5.2, -5.5, -6.0, 5.0];

/// SIMD-side oracle pins (3 significant figures from the verdict round).
const ORACLE_SIMD: [f32; 5] = [-2.29e-7, -5.69e-8, -5.93e-9, -8.44e-11, 5.0];

/// One answer of one lane: `g` sits at the LAST index of a `len`-wide row.
/// `len % 64 != 0` puts the last element on the scalar path, a multiple of 64
/// keeps it in the SIMD body — that is the lane split under test.
fn lane_answer(g: f32, up: f32, len: usize) -> f32 {
    let mut gate = vec![0.0f32; len];
    let up_row = vec![up; len];
    gate[len - 1] = g;
    let mut hidden = vec![0.0f32; len];
    gegelu_tanh(&mut hidden, &gate, &up_row);
    hidden[len - 1]
}

fn ulp_band(v: f32, inner: f32) -> f32 {
    let rel = CAP_ULPS * f32::EPSILON * inner.abs().max(1.0) * v.abs().max(f32::MIN_POSITIVE);
    rel.max(EXP_KERNEL_REL * v.abs())
}

/// BODY lane: every grid point through two body postures (single chunk and
/// two chunks) must agree bit-exactly, and the five oracle points must match
/// the pinned SIMD values within the 3-significant-figure pin band.
#[test]
fn simd_body_is_length_invariant_and_matches_the_oracle() {
    let mut g = -8.0f32;
    while g <= 8.0 {
        let single = lane_answer(g, 1.0, 64);
        let double = lane_answer(g, 1.0, 128);
        assert_eq!(
            single, double,
            "body lane moved across the chunk boundary at g={g}"
        );
        g += 0.05;
    }
    for (i, &g) in ORACLE_G.iter().enumerate() {
        let body = lane_answer(g, 1.0, 64);
        let pinned = ORACLE_SIMD[i];
        let band = 0.02 * pinned.abs().max(1e-12);
        assert!(
            (body - pinned).abs() <= band,
            "SIMD oracle drift at g={g}: got {body:e}, pinned {pinned:e}"
        );
    }
}

/// TAIL lane: all three scalar-remainder postures agree bit-exactly.
#[test]
fn scalar_tail_is_length_invariant() {
    let mut g = -8.0f32;
    while g <= 8.0 {
        let a = lane_answer(g, 1.0, 63);
        let b = lane_answer(g, 1.0, 65);
        let c = lane_answer(g, 1.0, 127);
        assert_eq!(a, b, "tail lane moved between len 63 and 65 at g={g}");
        assert_eq!(b, c, "tail lane moved between len 65 and 127 at g={g}");
        g += 0.05;
    }
}

/// The falsification arm: SIMD body vs scalar tail within 4 relative ulps
/// over the full band, at every chunk-boundary length. This is the arm that
/// FAILED on the pre-fix tail (−0.0 vs −5.69e−8 at g = −5.2 is a relative
/// 100% loss) and passes post-alignment.
#[test]
fn lanes_agree_within_the_pinned_relative_band() {
    for &len in &[63usize, 64, 65, 127, 128] {
        let mut g = -8.0f32;
        while g <= 8.0 {
            let body = lane_answer(g, 1.0, 64);
            let tail = lane_answer(g, 1.0, len);
            let gap = (body - tail).abs();
            let inner = 0.797_884_6 * (g + 0.044715 * g * g * g);
            let bound = ulp_band(body.abs().max(tail.abs()), inner);
            assert!(
                gap <= bound,
                "lane gap {gap:e} > {bound:e} (4·|inner| ulp) at g={g}, len={len}: body={body:e} tail={tail:e}"
            );
            g += 0.05;
        }
    }
}

/// The up operand rides both lanes: sign propagation and scaling must agree
/// at the oracle points too (up = −2 exercises the sign flip).
#[test]
fn lanes_agree_with_a_negative_up_operand() {
    for &g in ORACLE_G.iter() {
        let body = lane_answer(g, -2.0, 64);
        let tail = lane_answer(g, -2.0, 65);
        let gap = (body - tail).abs();
        let inner = 0.797_884_6 * (g + 0.044715 * g * g * g);
        let bound = ulp_band(body.abs().max(tail.abs()), inner);
        assert!(
            gap <= bound,
            "lane gap with up=−2 at g={g}: body={body:e} tail={tail:e} gap={gap:e} bound={bound:e}"
        );
    }
}

/// Re-pin instrument: prints the five oracle rows at length 65, up = 1 for
/// both lanes, then scans the full grid for the worst lane gaps relative to
/// the association-amplification scale `ε·|inner|·|answer|`. Run with
/// `cargo test -p katgpt-types --test gegelu_lane_parity probe -- --ignored
/// --nocapture` after any exp-kernel or spelling change.
#[test]
#[ignore]
fn probe() {
    for &g in ORACLE_G.iter() {
        let body = lane_answer(g, 1.0, 64);
        let tail = lane_answer(g, 1.0, 65);
        println!(
            "g {g:+5.1}  body {body:+12.6e}  tail {tail:+12.6e}  gap {:+e}",
            (body - tail).abs()
        );
    }
    let mut worst: Vec<(f32, f32, f32, f32)> = Vec::new();
    let mut g = -8.0f32;
    while g <= 8.0 {
        let body = lane_answer(g, 1.0, 64);
        let tail = lane_answer(g, 1.0, 65);
        let inner = 0.797_884_6 * (g + 0.044715 * g * g * g);
        let rel_scale = f32::EPSILON * inner.abs().max(1.0) * body.abs().max(f32::MIN_POSITIVE);
        worst.push((rel_scale, (body - tail).abs(), body, g));
        g += 0.05;
    }
    worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (scale, gap, body, g) in worst.iter().take(5) {
        println!(
            "worst g {g:+5.2}  gap {gap:+e}  body {body:+e}  rel_scale(eps·|inner|·|v|) {scale:+e}  ratio {:+.1}",
            gap / scale
        );
    }
}

/// Exp-kernel census: simd_exp_inplace vs libm `f32::exp` over the lane's
/// working range. Backs the lane-parity band — the two exps are the floor
/// under every post-alignment lane gap. Post-1223 measured class: ~2 ulp
/// worst (pre-fix: 236 ulps — the `vrndq_f32` truncation, issue 1223).
/// `#[ignore]`d like `probe`.
#[test]
#[ignore]
fn exp_kernel_census() {
    let n = 4001;
    let mut xs: Vec<f32> = (0..n)
        .map(|i| -10.0 + 20.0 * (i as f32) / ((n - 1) as f32))
        .collect();
    let libm: Vec<f32> = xs.iter().map(|&x| x.exp()).collect();
    simd_exp_inplace(&mut xs);
    let mut worst: Vec<(f32, f32, f32)> = Vec::new();
    for i in 0..n {
        let rel = (xs[i] - libm[i]).abs() / libm[i].abs().max(f32::MIN_POSITIVE);
        worst.push((
            rel,
            xs[i] - libm[i],
            -10.0 + 20.0 * (i as f32) / ((n - 1) as f32),
        ));
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let ulps = |rel: f32| rel / f32::EPSILON;
    println!(
        "max rel diff {:+e} (={:.0} ulps) at x={:+.3}",
        worst[0].0,
        ulps(worst[0].0),
        worst[0].2
    );
    for (rel, abs, x) in worst.iter().take(5) {
        println!(
            "  x {x:+8.3}  abs {abs:+e}  rel {rel:+e} ({:.0} ulps)",
            ulps(*rel)
        );
    }
}
