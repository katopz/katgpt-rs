//! Fixed-form softmax numerics primitives — the stable-form family of
//! arXiv:2610.31642 (Barron, Gnanapragasam & Tschannen, Google DeepMind,
//! "A Fix for Softmax and Cross-Entropy", 2026-10-22), distilled in
//! Research 632 and filed as Issue 959.
//!
//! Two floating-point bugs live in every standard max-subtraction
//! softmax/log-softmax/cross-entropy, including the workspace's ~10
//! hand-rolled Rust variants:
//!
//! **Softmax Collapse.** Max-subtraction evaluates the denominator as
//! `1 + e^{-m}`; once the logit margin `m` passes the format's precision
//! cliff `c_d = p·ln2` (p = significand bits of the ACCUMULATOR format),
//! the `e^{-m}` term is absorbed and the target slot saturates at exactly
//! `−0.0` — forward loss zeroed, backward gradient sum broken. The cliff
//! binds where `1 + e^{-m}` is EVALUATED: this stack stores f16/bf16 but
//! accumulates f32, so the operative stack-wide cliff is 16.64 nats
//! (fp32), never 5.5 (bf16) — scoping the table per-accumulator is part
//! of the API, not a footnote.
//!
//! **Softmax Cancellation.** Probability-space subtraction of
//! near-equal softmax values loses the difference: error ~eps·p against
//! a true difference ~δ·p. The fixed form differences in log space via
//! `expm1`, and is EXACTLY 0.0 at equality — the same resolution law
//! [`crate::noisy_or_stable`] ships for noisy-OR spans.
//!
//! **Scoping (pre-conceded, Research 632 §2.4):** the forward absolute
//! win is ≤ machine-epsilon per value. Every benefit is a DIFFERENCING
//! benefit — KL checks, drift/capture verification, quant deltas,
//! comparison-lane agreement. Instrument-grade, never a
//! serving-forward replacement: the clef lane's expf-fidelity contract
//! and attention's O(1) convex mixtures are correct as-is.

use core::f64::consts::LN_2;

/// Floating-point formats whose accumulator cliff the family scopes.
///
/// `p` is the significand precision (explicit stored bits + the implicit
/// leading 1) — the bits that decide when `1 + e^{-m}` absorbs `e^{-m}`.
/// `fp16` and `tf32` share p = 11 and therefore one cliff; both variants
/// exist so an audit names the format the config actually runs.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatFormat {
    /// e4m3: 3 stored + 1 implicit = 4 significand bits.
    Fp8E4M3,
    /// bfloat16: 7 stored + 1 implicit = 8 significand bits.
    Bf16,
    /// IEEE half: 10 stored + 1 implicit = 11 significand bits.
    Fp16,
    /// TF32 (Ampere+ tensor-core format): 10 stored + 1 implicit = 11 bits.
    Tf32,
    /// IEEE single: 23 stored + 1 implicit = 24 significand bits.
    Fp32,
    /// IEEE double: 52 stored + 1 implicit = 53 significand bits.
    Fp64,
}

/// Significand bits p per format (explicit + implicit leading 1).
#[inline]
pub const fn significand_bits(format: FloatFormat) -> u32 {
    match format {
        FloatFormat::Fp8E4M3 => 4,
        FloatFormat::Bf16 => 8,
        FloatFormat::Fp16 | FloatFormat::Tf32 => 11,
        FloatFormat::Fp32 => 24,
        FloatFormat::Fp64 => 53,
    }
}

/// The precision-cliff table: `(format, p, c_d = p·ln2)` in nats.
///
/// The paper states the numbers (5.5 / 16.6 / 36.7); the closed form
/// `c_d = p·ln2` is ours (Research 632 §2.1) — the table records both.
/// Read with [`precision_cliff`]; iterate for audits.
pub const PRECISION_CLIFF_NATS: &[(FloatFormat, u32, f64)] = &[
    (FloatFormat::Fp8E4M3, 4, 4.0 * LN_2),
    (FloatFormat::Bf16, 8, 8.0 * LN_2),
    (FloatFormat::Fp16, 11, 11.0 * LN_2),
    (FloatFormat::Tf32, 11, 11.0 * LN_2),
    (FloatFormat::Fp32, 24, 24.0 * LN_2),
    (FloatFormat::Fp64, 53, 53.0 * LN_2),
];

/// The precision cliff `c_d = p·ln2` (nats) for a format's accumulator.
///
/// Past this margin a max-subtraction softmax/log-softmax target slot
/// reads exactly `−0.0` (loss zeroed, gradient sum broken) because
/// `1 + e^{-m}` rounds to 1.0. Goldens vs the paper (±0.1 nat) are
/// pinned by the module tests: 2.8 / 5.5 / 7.6 / 16.6 / 36.7.
#[inline]
pub fn precision_cliff(format: FloatFormat) -> f64 {
    precision_cliff_bits(significand_bits(format))
}

/// The precision cliff for an arbitrary accumulator significand width
/// (the law, not just the named formats): `c_d = p·ln2`.
#[inline]
pub fn precision_cliff_bits(p: u32) -> f64 {
    f64::from(p) * LN_2
}

/// Margin-equilibrium verdict for a training/eval config: given the
/// optimizer epsilon and the accumulator format, does the predicted
/// post-memorization logit margin `m* = ln(1/ε)` sit PAST the format's
/// precision cliff?
///
/// The paper's margin-equilibrium law: under AdamW + weight decay the
/// margin saturates at `m* ≈ ln(1/ε)`. Default `ε = 1e-8` →
/// `m* ≈ 18.42`, just ABOVE the fp32 cliff 16.64 — the standard form
/// hits the cliff right before equilibrium (the "Slingshot" regime).
/// `ε ≥ 1e-7` → `m* ≤ 16.12` < cliff: clear. This is the Rust half of
/// the offline audit instrument (the CLI form lives at
/// `scripts/margin_equilibrium_audit.py`); the verdict names the
/// REGIME, it does not measure a live model.
#[inline]
pub fn margin_equilibrium(eps: f64, format: FloatFormat) -> (f64, f64, bool) {
    let m_star = -(eps.ln());
    let cliff = precision_cliff(format);
    (m_star, cliff, m_star > cliff)
}

/// Log-space probability difference `e^a − e^b`, exact 0.0 at equality.
///
/// The [`crate::noisy_or_stable`] resolution law pointed at softmax
/// differencing (Research 632 §2.5 — one law, now both homes): the
/// probability-space incumbent `exp(a) − exp(b)` carries residual
/// ~eps·p against a true difference ~δ·p, and reads an EXACT zero once
/// δ·p drops below half an ulp of p — a 100 % relative miss at
/// differences the paper's capture/verification consumers must see.
/// The fixed form factors the difference:
///
/// ```text
/// e^a − e^b = e^b · (e^{a−b} − 1) = e^b · expm1(a − b)   (a ≥ b)
/// ```
///
/// so at `a == b` the result is `expm1(0) = 0` EXACTLY — the faithful
/// zero — and at `δ = a−b = 1e-8` it still resolves `e^b·δ` to
/// correctly-rounded accuracy where the incumbent returns 0.0.
///
/// **Inputs are LOG-domain values** (log-probabilities, or logits over a
/// shared log-partition — the fixed soft-target `q_i − p_i` with
/// `a = log q_i`, `b = log p_i`). Overflow-safe: past the exp argument
/// bound the farther term is below relative resolution of the nearer
/// one, so the pure nearer exponential is the correctly-rounded answer
/// (no `inf·0` NaN).
///
/// Instrument-grade (Research 632 §2.4): KL checks, drift/capture
/// verification, quant deltas, comparison-lane agreement — never a
/// serving-forward replacement. Opt-in until a consumer GOAT
/// (Issue 959 T4).
#[cfg(feature = "logspace_diff")]
#[inline]
pub fn logspace_diff(a: f32, b: f32) -> f32 {
    let d = a - b;
    if d >= 0.0 {
        if d > EXP_ARG_BOUND_F32 {
            a.exp()
        } else {
            b.exp() * d.exp_m1()
        }
    } else if -d > EXP_ARG_BOUND_F32 {
        -b.exp()
    } else {
        -(a.exp() * (-d).exp_m1())
    }
}

/// [`logspace_diff`] in f64 — the oracle posture. Callers that compute
/// differences in f64 (golden arms, NLL oracles) delegate here rather
/// than narrowing first; narrowing the INPUTS would be a numerics
/// change, not a delegation (the [`crate::exact_sigmoid_f64`] law).
#[cfg(feature = "logspace_diff")]
#[inline]
pub fn logspace_diff_f64(a: f64, b: f64) -> f64 {
    let d = a - b;
    if d >= 0.0 {
        if d > EXP_ARG_BOUND_F64 {
            a.exp()
        } else {
            b.exp() * d.exp_m1()
        }
    } else if -d > EXP_ARG_BOUND_F64 {
        -b.exp()
    } else {
        -(a.exp() * (-d).exp_m1())
    }
}

/// exp's finite argument bound for f32: `ln(f32::MAX) ≈ 88.7228`. Past
/// it `exp_m1` overflows to inf; at that separation the b-side term is
/// below relative resolution of the a-side anyway.
#[cfg(feature = "logspace_diff")]
const EXP_ARG_BOUND_F32: f32 = 88.722_84;

/// exp's finite argument bound for f64: `ln(f64::MAX) ≈ 709.7827`.
#[cfg(feature = "logspace_diff")]
const EXP_ARG_BOUND_F64: f64 = 709.782_712_893_384;

#[cfg(test)]
mod tests {
    use super::*;

    /// T1 G1: goldens vs the paper's stated cliffs, ±0.1 nat.
    /// (fp8e4m3 2.8, bf16 5.5, fp16/tf32 7.6, fp32 16.6, fp64 36.7.)
    #[test]
    fn cliffs_match_the_paper_goldens() {
        let cases = [
            (FloatFormat::Fp8E4M3, 2.8),
            (FloatFormat::Bf16, 5.5),
            (FloatFormat::Fp16, 7.6),
            (FloatFormat::Tf32, 7.6),
            (FloatFormat::Fp32, 16.6),
            (FloatFormat::Fp64, 36.7),
        ];
        for (format, paper) in cases {
            let c = precision_cliff(format);
            assert!(
                (c - paper).abs() < 0.1,
                "{format:?}: cliff {c} vs paper {paper} beyond ±0.1 nat"
            );
        }
    }

    /// T1 G1: the scaling law — every table row is exactly p·ln2, and
    /// the shared-p formats agree bit-for-bit.
    #[test]
    fn cliff_is_p_times_ln2_for_every_table_row() {
        for &(format, p, c) in PRECISION_CLIFF_NATS {
            assert_eq!(significand_bits(format), p);
            assert_eq!(c, precision_cliff_bits(p));
            assert_eq!(c, f64::from(p) * core::f64::consts::LN_2);
        }
        assert_eq!(
            precision_cliff(FloatFormat::Fp16),
            precision_cliff(FloatFormat::Tf32)
        );
    }

    /// The enum is a 1-byte discriminant (the `#[repr(u8)]` guarantee)
    /// and the table covers every variant exactly once.
    #[test]
    fn format_enum_is_byte_sized_and_fully_tabled() {
        assert_eq!(core::mem::size_of::<FloatFormat>(), 1);
        for byte in 0u8..6 {
            let format = match byte {
                0 => FloatFormat::Fp8E4M3,
                1 => FloatFormat::Bf16,
                2 => FloatFormat::Fp16,
                3 => FloatFormat::Tf32,
                4 => FloatFormat::Fp32,
                _ => FloatFormat::Fp64,
            };
            assert_eq!(format as u8, byte);
            assert!(
                PRECISION_CLIFF_NATS.iter().any(|&(f, _, _)| f == format),
                "{format:?} missing from the cliff table"
            );
        }
    }

    /// T3's in-crate half: the paper's Slingshot golden — default
    /// AdamW ε = 1e-8 puts m* = 18.42 just past the fp32 cliff 16.64
    /// (HAZARD); ε = 1e-7 clears it. The verdict side is the CLI
    /// instrument's authority; this pins the arithmetic.
    #[test]
    fn margin_equilibrium_golden_slingshot_regime() {
        let (m_star, cliff, hazard) = margin_equilibrium(1e-8, FloatFormat::Fp32);
        assert!((m_star - 18.420_680_743_952_367).abs() < 1e-12);
        assert!((cliff - 16.635_532_333_438_687).abs() < 1e-12);
        assert!(hazard, "default eps must read HAZARD at fp32");

        let (m7, cliff7, hazard7) = margin_equilibrium(1e-7, FloatFormat::Fp32);
        assert!((m7 - 16.118_095_650_958_33).abs() < 1e-12);
        assert_eq!(cliff7, cliff);
        assert!(!hazard7, "eps = 1e-7 must read clear at fp32");

        // bf16's 5.545 cliff is hazardous at BOTH defaults — a
        // bf16-accumulated loss path would sit deep in the regime.
        assert!(margin_equilibrium(1e-7, FloatFormat::Bf16).2);
    }

    /// T2 G1: the fixed-CE-target-gradient identity — the numerically
    /// correct CE target gradient IS `−σ(−m)` (arXiv:2610.31642 Eq. 3),
    /// so pinning [`crate::sigmoid`]'s fast_sigmoid delegate against the
    /// exact f64 form `−1/(1+e^m)` across the saturation band
    /// m ∈ [5.5, 40] IS pinning it against the fixed gradient. The band
    /// runs from the fp32 cliff 16.64 to the ±40 early-exit envelope —
    /// precisely where the Cephes polynomial deserves its hardest audit
    /// (the regime the paper's fix targets). Relative-error budget
    /// 1e-6: the shipped accuracy contract ("well within the 1e-6
    /// tolerance used by every sigmoid assertion in the codebase").
    #[test]
    fn fast_sigmoid_tracks_the_fixed_ce_gradient_across_the_cliff_band() {
        let mut m = 5.5_f64;
        let mut worst = 0.0_f64;
        while m <= 40.0 {
            // Exact −σ(−m) in f64: e^m is large-positive here, no
            // cancellation on this form for m > 0.
            let exact = -1.0 / (1.0 + m.exp());
            let approx = -f64::from(crate::sigmoid(-m as f32));
            let rel = ((approx - exact) / exact).abs();
            worst = worst.max(rel);
            assert!(
                rel < 1e-6,
                "m = {m}: fast_sigmoid rel err {rel} vs exact −σ(−m) {exact}"
            );
            m += 0.25;
        }
        // The audit record is the worst-point envelope, asserted above.
        assert!(worst < 1e-6);
    }

    // ── T4 (logspace_diff) — gated with the feature ──────────────────
    #[cfg(feature = "logspace_diff")]
    mod diff {
        use super::*;

        /// T4 G1: the faithful zero — EXACT +0.0 at equality, f32+f64.
        #[test]
        fn exact_zero_at_equality() {
            assert_eq!(logspace_diff(-3.5_f32, -3.5), 0.0);
            assert!(!logspace_diff(-3.5_f32, -3.5).is_sign_negative());
            assert_eq!(logspace_diff(0.0_f32, 0.0), 0.0);
            assert_eq!(logspace_diff_f64(-708.0, -708.0), 0.0);
        }

        /// T4 G1: no false zero below the incumbent's band. The
        /// catastrophic band is probabilities near 1.0 (log values near
        /// 0): with b = 0 and δ = 1e-8/5e-8, `exp(a)` and `exp(b)` are
        /// the SAME f32 (ULP(1.0) = 1.19e-7), so the probability-space
        /// incumbent returns exactly 0.0 — the paper's collapse figure
        /// class — while the fixed form resolves e^b·δ against the f64
        /// oracle to ulp scale. δ = 1e-7 survives the incumbent but
        /// rounds to ONE ULP (~19 % off); recorded as the residual
        /// class the fixed form removes.
        #[test]
        fn resolves_small_deltas_where_the_incumbent_zeroes() {
            let b = 0.0_f32;
            for &delta in &[1e-8_f32, 5e-8] {
                let a = b + delta; // representable: ULP near 1e-8 is ~1e-15
                let incumbent = a.exp() - b.exp();
                let fixed = logspace_diff(a, b);
                let oracle = logspace_diff_f64(f64::from(a), f64::from(b));
                assert_eq!(
                    incumbent, 0.0,
                    "fixture premise lost: incumbent must zero at delta {delta}"
                );
                assert!(fixed > 0.0, "fixed must stay positive at delta {delta}");
                let rel = ((f64::from(fixed) - oracle) / oracle).abs();
                assert!(
                    rel < 1e-6,
                    "delta {delta}: fixed rel err {rel} vs oracle {oracle}"
                );
            }
            // The one-ULP residual: a difference the incumbent DOES see
            // but rounds to a single ulp of 1.0 (~19 % relative error).
            let a = b + 1e-7_f32;
            let incumbent = a.exp() - b.exp();
            let fixed = logspace_diff(a, b);
            let oracle = logspace_diff_f64(f64::from(a), f64::from(b));
            assert!(incumbent > 0.0 && incumbent != fixed);
            let rel = ((f64::from(fixed) - oracle) / oracle).abs();
            assert!(rel < 1e-6, "delta 1e-7: fixed rel err {rel}");
        }

        /// T4 G1: at the issue's named deltas {0, 1e-6, 1e-3} the fixed
        /// form tracks the f64 oracle to ulp scale in BOTH directions.
        #[test]
        fn tracks_the_f64_oracle_at_named_deltas_both_directions() {
            let b = -16.0_f32;
            for &delta in &[1e-6_f32, 1e-3] {
                for &sign in &[1.0_f32, -1.0] {
                    let (a, b_) = if sign > 0.0 {
                        (b + delta, b)
                    } else {
                        (b, b + delta)
                    };
                    let fixed = logspace_diff(a, b_);
                    let oracle = logspace_diff_f64(f64::from(a), f64::from(b_));
                    let scale = f64::max(oracle.abs(), 1e-30);
                    let rel = (f64::from(fixed) - oracle).abs() / scale;
                    assert!(rel < 1e-6, "delta {delta} sign {sign}: rel {rel}");
                    // Sign follows the log order.
                    assert_eq!(fixed > 0.0, a > b_);
                }
            }
        }

        /// Antisymmetry to ulp scale (the two anchored forms round
        /// differently; NOT bit-exact — asserted, not assumed).
        #[test]
        fn antisymmetric_to_ulp_scale_not_bits() {
            for &(a, b) in &[(-3.5_f32, -3.7), (2.0, -1.5), (-16.0, -16.0 + 1e-3)] {
                let forward = logspace_diff(a, b);
                let backward = logspace_diff(b, a);
                let sum = forward + backward;
                let scale = forward.abs().max(f32::EPSILON);
                assert!(
                    sum.abs() / scale < 1e-5,
                    "({a},{b}): f(a,b)+f(b,a) = {sum} vs |f| {forward}"
                );
            }
        }

        /// The overflow branch: past the expm1 argument bound the
        /// result is the nearer exponential — finite (the nearer value
        /// itself stays inside exp's range in these fixtures), correct
        /// sign, no NaN from an `inf · 0`.
        #[test]
        fn overflow_branch_returns_the_nearer_exponential() {
            // d = 100 > ln(f32::MAX): the far term is below relative
            // resolution of e^80, so the answer IS e^80 — and the f64
            // oracle agrees exactly.
            let v = logspace_diff(80.0_f32, -20.0);
            assert!(v.is_finite());
            assert!(!v.is_nan());
            assert_eq!(v, 80.0_f32.exp());
            let neg = logspace_diff(-20.0_f32, 80.0);
            assert_eq!(neg, -(80.0_f32.exp()));

            let v64 = logspace_diff_f64(700.0, -20.0);
            assert!(v64.is_finite() && !v64.is_nan());
            assert_eq!(v64, 700.0_f64.exp());
        }

        /// Monotone non-decreasing in δ over a fixed base — a
        /// differencing primitive that folds or sorts may rely on it.
        #[test]
        fn monotone_in_delta() {
            let b = -10.0_f32;
            let mut prev = logspace_diff(b, b);
            assert_eq!(prev, 0.0);
            let mut d = 1e-7_f32;
            while d < 5.0 {
                let v = logspace_diff(b + d, b);
                assert!(v >= prev, "delta {d}: {v} < prev {prev}");
                prev = v;
                d *= 2.0;
            }
        }

        /// The capture-fidelity consumer shape: a one-ULP log-domain
        /// separation must read a NONZERO difference exactly where the
        /// capture-verification delta lives.
        #[test]
        fn one_ulp_log_separation_stays_resolvable() {
            let base = -20.0_f32;
            let bumped = base + f32::EPSILON * base.abs();
            let d = bumped - base; // the ACTUAL post-rounding separation
            assert!(
                d > 0.0,
                "bump must survive f32 rounding to a distinct value"
            );
            let fixed = logspace_diff(bumped, base);
            assert!(fixed > 0.0, "one-ulp separation must not false-zero");
            let expected = f64::from(base.exp()) * f64::from(d);
            let rel = ((f64::from(fixed) - expected) / expected).abs();
            assert!(rel < 1e-5, "rel {rel} vs expected {expected}");
        }
    }
}
