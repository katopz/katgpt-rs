//! Issue 696 T2 — the **anchored signed-reach blend operator** (Research
//! 433 / riir-train, arXiv:2608.23664 "RVM", Eq. 7 — the unified anchored
//! regression family: `E[c(r)/2·‖(v−v_anc) − A(r)·(v*−v_anc)‖²]`).
//!
//! One branch-free operator with a signed reach `A ∈ ℝ` sweeping five
//! regimes, plus reward-modulated reach schedules `A(r)`:
//!
//! ```text
//! out = anchor + A·(candidate − anchor)
//!
//! A = 0    clamp      out ≡ anchor          (exact bits)
//! 0 < A<1  blend      out strictly between  (the convex incumbent's only regime)
//! A = 1    adopt      out ≡ candidate       (exact bits)
//! A > 1    overshoot  out past candidate    (DiffusionNFT lead prediction)
//! A < 0    repel      out on the far side   (contrastive aversion)
//!
//! A(r):  linear       A = r                    (raw signed reward)
//!        house-sig    A = 2σ(kr) − 1  ∈ [−1,1] (saturation-symmetric)
//!        sign-flip    A = (2r − 1)/β̄           (DiffusionNFT: r=1 ⇒ adopt,
//!                                               2r−1 > β̄ ⇒ overshoot)
//! ```
//!
//! Sigmoid, never softmax — by construction: every schedule is pointwise;
//! nothing normalizes across a group (RVM itself never normalizes across
//! the group — signed unnormalized weights suffice).
//!
//! # Bit-identity at the poles (the load-bearing contract)
//!
//! At `A == 1.0` the operator returns the **candidate verbatim**; at
//! `A == 0.0` the **anchor verbatim** — explicit fast paths, because the
//! composed form is NOT bit-identical under floating point (two roundings:
//! the subtract, then the re-add). Demonstrated counterexample, pinned by
//! test: `anchor = 1e-30, candidate = 1e-40` (subnormal) — the subtraction
//! `1e-40 − 1e-30` rounds to exactly `−1e-30`, the re-add collapses to
//! `+0.0`, and **the candidate is lost entirely** (`0.0` vs `1e-40`).
//! "Bit-identical at A=1 with plain adoption" (issue gate) therefore needs
//! the fast path; it costs one comparison and it is load-bearing, not
//! decorative.
//!
//! # Honest deviations / scope (doc-truth)
//!
//! - **No β(r) scale arm.** Eq. 7 carries a scale `c(r)` ("how hard")
//!   beside the reach `A(r)` ("where to"); the scale is a training-side
//!   knob (a loss weight) with no modelless scalar analog — Research 433
//!   routes it to riir-train Plan 360. Only the reach ships here.
//! - **`clip_reward` reproduces the paper's ±5 signed-reward clip** (r =
//!   group-standardized reward clipped ±5); it is a convenience, not a
//!   gate — the schedules do not force it.
//! - **Saturation is closed, not open.** `2σ(kr)−1` reaches exactly ±1.0
//!   at `|kr| > 40` (the crate's `fast_sigmoid` early-exit) — the
//!   documented range is the CLOSED interval [−1, 1].
//! - **Degenerate schedule inputs are dead schedules, not errors:**
//!   non-finite / non-positive `β̄`, non-finite `k`, non-finite `r` into
//!   the guarded schedules → `A = 0.0` (no reach). `schedule_linear` is
//!   the honest identity — it does not sanitize.
//! - **`out` must not alias `anchor`/`candidate`** in the `*_into` forms
//!   (the `A == 1` fast path is a `copy_from_slice`; overlap is UB).
//!
//! # Domain classification
//!
//! Latent, local, never synced: a state-update operator over caller-owned
//! buffers (think-brain beliefs, steering anchors, grudge conditioning —
//! T4's consumer list). No sync dependency, no replay coupling. Raw
//! physicals stay raw: the operator moves values, never invents them.
//!
//! Feature: `anchored_reach` (opt-in POC, independent of
//! `anti_common_mode`). Promotion only via consumer A/Bs (issue T4: lead
//! prediction on moving platforms / contrastive aversion /
//! planner-as-anchor), each falsifiable and filed separately — per the
//! issue's GOAT shape the headline promotion gate for the PAIR is T3's
//! CLR crowd-panic PoC; until any consumer lands this is an unproven
//! extraction, not a GOAT.
//!
//! RVM modelless extraction (Issue 696 / Research 433, riir-train,
//! arXiv:2608.23664 — anchored reward-weighted velocity regression, Choi et
//! al.): the DT2 ANTI-COMMON-MODE scalar gate (peak-quantile statistic +
//! context-scaled threshold + median subtraction over the hack-carrying
//! population + band window — resists capture by the population-dominant
//! degenerate mode; the CLR crowd-panic failure is the named consumer, T3
//! PoC pending) and the ANCHORED SIGNED-REACH blend operator (out = anchor
//! + A·(cand − anchor); five regimes clamp/blend/adopt/overshoot/repel;
//! bit-identical pole fast paths at A ∈ {0,1}; A(r) schedules linear /
//! 2σ(kr)−1 / (2r−1)/β̄). Both zero-alloc, modelless,
//! sigmoid-not-softmax by construction. Opt-in POC — promotion only via the
//! issue's consumer PoC gates (T3 headline; T4 operator A/Bs).

use crate::simd::fast_sigmoid;

/// The paper's signed-reward clip: `r = clip(r, ±5)` before any schedule.
pub const RVM_CLIP: f32 = 5.0;

/// Clip a (group-standardized) signed reward to the paper's ±5 window.
/// Non-finite rewards carry no admissible signal → `0.0`.
#[must_use]
#[inline]
pub fn clip_reward(r: f32) -> f32 {
    if r.is_finite() {
        r.clamp(-RVM_CLIP, RVM_CLIP)
    } else {
        0.0
    }
}

/// The anchored signed-reach blend, scalar form:
/// `out = anchor + A·(candidate − anchor)`.
///
/// **Pole fast paths:** `A == 1.0` returns the candidate EXACTLY (same
/// bits) and `A == 0.0` the anchor EXACTLY — see the module doc for the
/// floating-point counterexample that makes these load-bearing.
#[must_use]
#[inline]
pub fn reach_scalar(anchor: f32, candidate: f32, a: f32) -> f32 {
    if a == 1.0 {
        return candidate;
    }
    if a == 0.0 {
        return anchor;
    }
    anchor + a * (candidate - anchor)
}

/// Anchored reach over slices with a SCALAR reach, writing into caller
/// output (zero-alloc; the pole fast paths hoist OUT of the loop — one
/// branch for the whole slice).
///
/// # Panics
/// Unless `anchor.len() == candidate.len() && candidate.len() == out.len()`.
///
/// # Aliasing
/// `out` must not alias `anchor` or `candidate`.
pub fn blend_scalar_into(anchor: &[f32], candidate: &[f32], a: f32, out: &mut [f32]) {
    assert_eq!(
        anchor.len(),
        candidate.len(),
        "anchor/candidate length mismatch"
    );
    assert_eq!(candidate.len(), out.len(), "candidate/out length mismatch");
    if a == 1.0 {
        out.copy_from_slice(candidate);
        return;
    }
    if a == 0.0 {
        out.copy_from_slice(anchor);
        return;
    }
    for i in 0..out.len() {
        out[i] = anchor[i] + a * (candidate[i] - anchor[i]);
    }
}

/// Anchored reach over slices with PER-AXIS reach, writing into caller
/// output (zero-alloc). Anisotropic reach: e.g. lead the along-velocity
/// axis harder than the lateral axis.
///
/// # Panics
/// Unless `anchor.len() == candidate.len() && candidate.len() == a.len()
/// && a.len() == out.len()`.
///
/// # Aliasing
/// `out` must not alias `anchor`, `candidate`, or `a`.
pub fn blend_into(anchor: &[f32], candidate: &[f32], a: &[f32], out: &mut [f32]) {
    assert_eq!(
        anchor.len(),
        candidate.len(),
        "anchor/candidate length mismatch"
    );
    assert_eq!(candidate.len(), a.len(), "candidate/reach length mismatch");
    assert_eq!(a.len(), out.len(), "reach/out length mismatch");
    for i in 0..out.len() {
        out[i] = reach_scalar(anchor[i], candidate[i], a[i]);
    }
}

/// Linear reach schedule: `A(r) = r` — the raw signed reward IS the reach.
/// Documented range: all of ℝ (clip with [`clip_reward`] if the paper's
/// ±5 window is wanted; this fn deliberately does not sanitize).
#[must_use]
#[inline]
pub fn schedule_linear(r: f32) -> f32 {
    r
}

/// House-sigmoid reach schedule: `A(r) = 2σ(kr) − 1 ∈ [−1, 1]` (closed —
/// saturates to exactly ±1.0 at `|kr| > 40` via the crate's
/// `fast_sigmoid`). Monotone in `r` for `k > 0`; `k = 0` is the dead
/// schedule (`A ≡ 0`); negative `k` flips polarity (documented caller
/// semantics); non-finite `r` or `k` → `0.0`.
#[must_use]
#[inline]
pub fn schedule_sigmoid(r: f32, k: f32) -> f32 {
    if !r.is_finite() || !k.is_finite() {
        return 0.0;
    }
    2.0 * fast_sigmoid(k * r) - 1.0
}

/// Sign-flip reach schedule (the DiffusionNFT form): `A(r) = (2r − 1)/β̄`.
///
/// At `β̄ = 1`: `r = 0` → `−1` (repel), `r = 1` → `+1` (adopt), `r = 2` →
/// `+3` (overshoot ×3 — the paper's "overshoots past v whenever
/// `2r_nft − 1 > β̄`"). Non-finite / non-positive `β̄` or non-finite `r`
/// → `0.0` (a dead schedule applies no reach).
#[must_use]
#[inline]
pub fn schedule_sign_flip(r: f32, beta_bar: f32) -> f32 {
    if !r.is_finite() || !beta_bar.is_finite() || beta_bar <= 0.0 {
        return 0.0;
    }
    (2.0 * r - 1.0) / beta_bar
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    // SWEEP's 3.14159 is an arbitrary non-power-of-two decimal chosen as a
    // bit-identity hazard, not an approximation of pi. Substituting
    // `consts::PI` would silently change a pinned test vector.
    #![allow(clippy::approx_constant)]

    use super::*;

    /// A value sweep covering the bit-identity hazards: zeros, negatives,
    /// subnormals, extremes, cross-binade pairs, non-power-of-two decimals.
    const SWEEP: [f32; 16] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        3.14159,
        1e-30,
        -1e-30,
        1e-40,
        1e30,
        -1e30,
        f32::MIN_POSITIVE,
        16777217.0,
        -0.1,
        123.456,
        f32::from_bits(1), // smallest subnormal
    ];

    #[test]
    fn g1_five_regimes_produce_predicted_outputs() {
        let (anchor, cand) = (2.0f32, 6.0);
        // A = 0 → clamp to the anchor, exact bits.
        assert_eq!(reach_scalar(anchor, cand, 0.0).to_bits(), anchor.to_bits());
        // 0 < A < 1 → strictly between.
        let half = reach_scalar(anchor, cand, 0.5);
        assert!(half > anchor && half < cand, "blend must be interior");
        assert!((half - 4.0).abs() < 1e-6);
        // A = 1 → adopt the candidate, exact bits.
        assert_eq!(reach_scalar(anchor, cand, 1.0).to_bits(), cand.to_bits());
        // A > 1 → overshoot past the candidate.
        let over = reach_scalar(anchor, cand, 1.5);
        assert!(over > cand, "A=1.5 must overshoot");
        assert!((over - 8.0).abs() < 1e-5);
        // A < 0 → repel to the opposite side of the anchor.
        let rep = reach_scalar(anchor, cand, -0.5);
        assert!(rep < anchor, "A=−0.5 must repel");
        assert!((rep - 0.0).abs() < 1e-6);
        // The same five regimes through the slice path (scalar A).
        let an = [anchor, 0.0, -10.0];
        let ca = [cand, 5.0, -30.0];
        let mut out = [0.0f32; 3];
        blend_scalar_into(&an, &ca, 0.5, &mut out);
        assert!((out[0] - 4.0).abs() < 1e-6);
        assert!((out[1] - 2.5).abs() < 1e-6);
        assert!((out[2] + 20.0).abs() < 1e-6);
        blend_scalar_into(&an, &ca, 1.5, &mut out);
        assert!(
            out[0] > 6.0 && out[1] > 5.0 && out[2] < -30.0,
            "overshoot axis-wise"
        );
        blend_scalar_into(&an, &ca, -0.5, &mut out);
        assert!(
            out[0] < 2.0 && out[1] < 0.0 && out[2] > -10.0,
            "repel axis-wise"
        );
    }

    #[test]
    fn g1_bit_identity_at_poles() {
        // The naive composed form demonstrably loses the candidate on the
        // pinned counterexample pair (subnormal absorbed by the subtract):
        let (a, c) = (1e-30f32, 1e-40f32);
        let naive = a + 1.0f32 * (c - a);
        assert_eq!(naive, 0.0, "1e-40 − 1e-30 rounds to −1e-30; re-add → 0");
        assert_ne!(
            naive.to_bits(),
            c.to_bits(),
            "the fast path is load-bearing"
        );
        // The operator's poles are exact over the full sweep, both orders.
        for &anchor in &SWEEP {
            for &cand in &SWEEP {
                assert_eq!(
                    reach_scalar(anchor, cand, 1.0).to_bits(),
                    cand.to_bits(),
                    "A=1 must return the candidate verbatim ({anchor}, {cand})"
                );
                assert_eq!(
                    reach_scalar(anchor, cand, 0.0).to_bits(),
                    anchor.to_bits(),
                    "A=0 must return the anchor verbatim ({anchor}, {cand})"
                );
            }
        }
        // ... and through both slice paths.
        let an: Vec<f32> = SWEEP.to_vec();
        let ca: Vec<f32> = SWEEP.iter().rev().copied().collect();
        let mut out = vec![0.0f32; SWEEP.len()];
        blend_scalar_into(&an, &ca, 1.0, &mut out);
        for i in 0..out.len() {
            assert_eq!(out[i].to_bits(), ca[i].to_bits());
        }
        blend_scalar_into(&an, &ca, 0.0, &mut out);
        for i in 0..out.len() {
            assert_eq!(out[i].to_bits(), an[i].to_bits());
        }
        let ones = vec![1.0f32; SWEEP.len()];
        let zeros = vec![0.0f32; SWEEP.len()];
        blend_into(&an, &ca, &ones, &mut out);
        for i in 0..out.len() {
            assert_eq!(out[i].to_bits(), ca[i].to_bits());
        }
        blend_into(&an, &ca, &zeros, &mut out);
        for i in 0..out.len() {
            assert_eq!(out[i].to_bits(), an[i].to_bits());
        }
    }

    #[test]
    fn g1_per_axis_matches_scalar_elementwise() {
        let an = [1.0f32, -2.0, 0.25, 1e6, -3.5];
        let ca = [4.0f32, 8.0, -0.75, 1e6 + 512.0, 7.25];
        // Heterogeneous per-axis A (one of each regime class).
        let axes = [0.5f32, 1.5, -0.25, 1.0, 0.0];
        let mut out = [0.0f32; 5];
        blend_into(&an, &ca, &axes, &mut out);
        for i in 0..5 {
            let expect = reach_scalar(an[i], ca[i], axes[i]);
            assert_eq!(out[i].to_bits(), expect.to_bits(), "axis {i}");
        }
        // A uniform per-axis A must equal the scalar-A path bitwise.
        let uniform = [0.75f32; 5];
        let mut out_scalar = [0.0f32; 5];
        let mut out_uniform = [0.0f32; 5];
        blend_scalar_into(&an, &ca, 0.75, &mut out_scalar);
        blend_into(&an, &ca, &uniform, &mut out_uniform);
        assert_eq!(out_scalar, out_uniform);
    }

    #[test]
    fn g1_schedule_constructors_documented_ranges() {
        // Linear: the honest identity (no sanitize).
        for &r in &[-100.0f32, -5.0, 0.0, 0.5, 1.0, 5.0, 100.0] {
            assert_eq!(schedule_linear(r), r);
        }
        // House-sigmoid: closed [−1, 1], monotone in r, exact zero at r=0.
        for &r in &[-100.0f32, -5.0, -1.0, -0.5, 0.5, 1.0, 5.0, 100.0] {
            let a = schedule_sigmoid(r, 1.0);
            assert!(
                (-1.0..=1.0).contains(&a),
                "2σ(kr)−1 out of [−1,1] at r={r}: {a}"
            );
        }
        assert!(schedule_sigmoid(-1.0, 1.0) < schedule_sigmoid(0.0, 1.0));
        assert!(schedule_sigmoid(0.0, 1.0) < schedule_sigmoid(1.0, 1.0));
        assert_eq!(schedule_sigmoid(0.0, 1.0), 0.0, "σ(0) = 0.5 exactly");
        // Saturates to the closed bounds at extreme kr.
        assert_eq!(schedule_sigmoid(100.0, 1.0), 1.0);
        assert_eq!(schedule_sigmoid(-100.0, 1.0), -1.0);
        // Dead / guarded schedules.
        assert_eq!(schedule_sigmoid(10.0, 0.0), 0.0, "k=0 is the dead schedule");
        assert_eq!(schedule_sigmoid(f32::NAN, 1.0), 0.0);
        assert_eq!(schedule_sigmoid(1.0, f32::INFINITY), 0.0);
        // Sign-flip at β̄ = 1: repel / adopt / overshoot ×3.
        assert_eq!(schedule_sign_flip(0.0, 1.0), -1.0);
        assert_eq!(schedule_sign_flip(1.0, 1.0), 1.0);
        assert_eq!(schedule_sign_flip(2.0, 1.0), 3.0);
        assert!((schedule_sign_flip(0.5, 2.0) - 0.0).abs() < 1e-7);
        assert_eq!(schedule_sign_flip(1.0, 0.0), 0.0, "β̄=0 → dead schedule");
        assert_eq!(schedule_sign_flip(1.0, -2.0), 0.0, "β̄<0 → dead schedule");
        assert_eq!(schedule_sign_flip(f32::NAN, 1.0), 0.0);
        // The paper's clip.
        assert_eq!(clip_reward(-7.0), -5.0);
        assert_eq!(clip_reward(3.0), 3.0);
        assert_eq!(clip_reward(9.0), 5.0);
        assert_eq!(clip_reward(f32::NAN), 0.0);
    }

    #[test]
    fn g1_schedule_feeds_operator_integration() {
        // The paper's composition: A(r) drives the operator's reach. Pin
        // one row per schedule through reach_scalar.
        let (anchor, cand) = (0.0f32, 1.0);
        let a_sig = schedule_sigmoid(2.0, 1.0);
        let out_sig = reach_scalar(anchor, cand, a_sig);
        assert!(out_sig > 0.5 && out_sig < 1.0, "σ(2) blend: {out_sig}");
        let a_flip = schedule_sign_flip(2.0, 1.0); // 3.0 — overshoot ×3
        assert!((reach_scalar(anchor, cand, a_flip) - 3.0).abs() < 1e-6);
        let a_lin = clip_reward(schedule_linear(-3.0)); // −3 — repel
        assert!((reach_scalar(anchor, cand, a_lin) + 3.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn g1_length_mismatch_panics() {
        let mut out = [0.0f32; 3];
        blend_scalar_into(&[1.0, 2.0], &[1.0, 2.0, 3.0], 0.5, &mut out);
    }

    #[cfg_attr(debug_assertions, ignore = "timing gate — release-only")]
    #[test]
    fn g2_blend_under_budget_at_n1000() {
        // Scalar-A slice path: the pole check hoists out of the loop; the
        // interior path is a branch-free FMA loop — measured per-call at
        // N=1000 (the issue's sub-µs ask is recorded honestly in
        // .benchmarks/688 §G2).
        const RUNS: u32 = 5_000;

        let n = 1000usize;
        let anchor: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let cand: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 + 3.0).collect();
        let axes: Vec<f32> = (0..n).map(|i| 0.25 + (i % 7) as f32 * 0.25).collect();
        let mut out = vec![0.0f32; n];
        // Warm up.
        for _ in 0..200 {
            blend_scalar_into(&anchor, &cand, 0.5, &mut out);
            blend_into(&anchor, &cand, &axes, &mut out);
        }
        std::hint::black_box(&out);
        let t0 = std::time::Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..RUNS {
            blend_scalar_into(&anchor, &cand, 0.5, &mut out);
            acc += out[0];
            blend_into(&anchor, &cand, &axes, &mut out);
            acc += out[1];
        }
        let dt = t0.elapsed();
        // Two blends per run — report per-blend.
        let per = dt.as_nanos() as f64 / (RUNS as f64 * 2.0);
        std::eprintln!("g2 blend N=1000: {per:.0} ns/blend (acc {acc})");
        assert!(acc.is_finite());
        assert!(
            per <= 5_000.0,
            "{per:.0} ns/blend > 5 µs regression floor @ N=1000"
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    fn g4_alloc_free_blended_path() {
        // Construction sits outside the measured region; the blended path
        // (both slice forms + scalar + schedules) allocates nothing.
        let n = 1000usize;
        let anchor: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let cand: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let axes: Vec<f32> = vec![0.5; n];
        let mut out = vec![0.0f32; n];
        crate::alloc::reset_alloc_stats();
        blend_scalar_into(&anchor, &cand, 0.5, &mut out);
        blend_into(&anchor, &cand, &axes, &mut out);
        let s = reach_scalar(1.0, 2.0, 0.25);
        let a1 = schedule_sigmoid(0.7, 2.0);
        let a2 = schedule_sign_flip(0.3, 1.5);
        let a3 = clip_reward(schedule_linear(-2.0));
        let (count, _bytes) = crate::alloc::get_alloc_stats();
        assert_eq!(
            count, 0,
            "blended path must be alloc-free (s={s} a1={a1} a2={a2} a3={a3})"
        );
    }
}
