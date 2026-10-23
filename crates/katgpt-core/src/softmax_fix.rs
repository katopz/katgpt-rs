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

/// Fixed-form log-softmax (arXiv:2610.31642 Eq. 5, Issue 959 T5): the
/// argmax-excluded LSE closed with `log1p`, so the target slot's
/// log-prob resolves from the precision cliff `c_d` out to exponent
/// underflow instead of saturating at exactly 0.0.
///
/// **The bug this removes.** The incumbent max-subtraction form
/// evaluates the target (max) slot as `−ln(1 + e^{−m})`, where `m` is
/// the best competitor's margin. The `1 +` and the `e^{−m}` are summed
/// in the accumulator format, and once `m` passes `c_d = p·ln2` (fp32:
/// 16.64 nats) the addition rounds to `1.0` — the target log-prob reads
/// EXACTLY 0.0 (the slot's loss zeroed; the G1 test pins the witness).
/// The fix never adds 1 to `e^{−m}`: the argmax is pulled OUT of the
/// sum,
///
/// ```text
/// log p_i = (z_i − z_a) − ln(1 + R),   R = Σ_{j≠a} e^{z_j − z_a}
/// ```
///
/// so every summed term is a STANDALONE `exp(−margin)` — never added to
/// 1 — that keeps its own relative precision down to subnormals, and
/// the close is `log1p(R)`, which resolves tiny R at full relative
/// precision. The target slot stays faithful while any competitor is
/// still representable as an exp argument (~103 nats f32, ~745 nats
/// f64) instead of only to the cliff. Every `exp` argument is
/// `z_j − z_a ≤ +0.0`, so `R ∈ [0, n−1]` — the accumulator cannot
/// overflow, by construction.
///
/// **Determinism-law seam (Issue 940 class).** The excluded-argmax fold
/// pins its order at this site: argmax by a FIRST-index strictly-greater
/// walk (ties → lowest index — the fixed strict tie rule); `R` as a
/// sequential ascending-`j` plain `+=` fold skipping exactly `j == a`;
/// the `log1p` close. Never a platform-lane SIMD kernel. This primitive
/// is instrument-grade and LOCAL — any future sync/quorum crossing of
/// its outputs must go through the certified precedent shape
/// ([`crate::closure::bridge_certified`]: sequential ordered fold +
/// version hash) and stay feature-gated until its own GOAT.
///
/// **Edges.** A single-element row returns exactly `[0.0]` (softmax of
/// one element is 1). At most one `+inf` per row: the lone `+inf` max
/// prices exactly (1, 0, …), but a duplicated `+inf` max makes
/// `inf − inf = NaN`, which propagates. An all-`−inf` row is a
/// precondition violation (its softmax is undefined) and reads NaN.
/// NaN logits propagate.
///
/// **Gates.** G1 golden (margins 0..~103 vs the f64 oracle, incumbent
/// saturation witness): module tests. G2 release parity bench:
/// `benches/softmax_fix_log_softmax_goat.rs`. G4:
/// `tests/softmax_fix_g4_alloc.rs`. G3 is structural — the feature is
/// opt-in and default builds compile none of this. Instrument-grade
/// only — never a serving-forward replacement (the clef expf-fidelity
/// contract and attention's O(1) mixtures stay untouched).
///
/// Panics if `z.len() != out.len()` or `z` is empty.
#[cfg(feature = "fixed_log_softmax")]
pub fn fixed_log_softmax(z: &[f32], out: &mut [f32]) {
    assert_eq!(
        z.len(),
        out.len(),
        "fixed_log_softmax: z/out length mismatch"
    );
    assert!(!z.is_empty(), "fixed_log_softmax: empty row");
    if z.len() == 1 {
        out[0] = 0.0;
        return;
    }
    let (a, r) = argmax_first_and_remaining_mass(z);
    let l = r.ln_1p();
    let za = z[a];
    // Branch-free hot loop; the argmax slot is overwritten after (for
    // finite logits `(z_a − z_a) − l == −l` bit-for-bit, and the
    // overwrite is what keeps a `+inf` max from NaN-ing on
    // `inf − inf`).
    for (o, &v) in out.iter_mut().zip(z.iter()) {
        *o = (v - za) - l;
    }
    out[a] = -l;
}

/// [`fixed_log_softmax`] in f64 — the oracle posture. Callers that
/// compute log-probabilities in f64 (golden arms, NLL oracles — the
/// Issue 959 T6 consumer shape) delegate here rather than widening
/// first; narrowing the INPUTS would be a numerics change, not a
/// delegation (the [`crate::exact_sigmoid_f64`] law).
#[cfg(feature = "fixed_log_softmax")]
pub fn fixed_log_softmax_f64(z: &[f64], out: &mut [f64]) {
    assert_eq!(
        z.len(),
        out.len(),
        "fixed_log_softmax_f64: z/out length mismatch"
    );
    assert!(!z.is_empty(), "fixed_log_softmax_f64: empty row");
    if z.len() == 1 {
        out[0] = 0.0;
        return;
    }
    let (a, r) = argmax_first_and_remaining_mass_f64(z);
    let l = r.ln_1p();
    let za = z[a];
    for (o, &v) in out.iter_mut().zip(z.iter()) {
        *o = (v - za) - l;
    }
    out[a] = -l;
}

/// Single-slot [`fixed_log_softmax`]: `log p_target` only — the NLL
/// oracle's consumption shape (one gold token per row; the full vector
/// would be n−1 wasted subtractions). Bit-identical to the slot the
/// full form writes (same pinned fold, same `−l` argmax branch).
#[cfg(feature = "fixed_log_softmax")]
#[must_use]
pub fn fixed_log_softmax_target(z: &[f32], target: usize) -> f32 {
    assert!(
        target < z.len(),
        "fixed_log_softmax_target: target out of range"
    );
    if z.len() == 1 {
        return 0.0;
    }
    let (a, r) = argmax_first_and_remaining_mass(z);
    let l = r.ln_1p();
    if target == a {
        -l
    } else {
        (z[target] - z[a]) - l
    }
}

/// [`fixed_log_softmax_target`] in f64.
#[cfg(feature = "fixed_log_softmax")]
#[must_use]
pub fn fixed_log_softmax_target_f64(z: &[f64], target: usize) -> f64 {
    assert!(
        target < z.len(),
        "fixed_log_softmax_target_f64: target out of range"
    );
    if z.len() == 1 {
        return 0.0;
    }
    let (a, r) = argmax_first_and_remaining_mass_f64(z);
    let l = r.ln_1p();
    if target == a {
        -l
    } else {
        (z[target] - z[a]) - l
    }
}

/// The argmax (FIRST index on ties — the fixed strict tie rule) and the
/// argmax-excluded remaining mass `R = Σ_{j≠a} e^{z_j − z_a}`, folded in
/// ascending index order with plain `+=` — the pinned fold order of the
/// determinism-law seam (see [`fixed_log_softmax`]). Every exp argument
/// is ≤ +0.0, so the fold cannot overflow.
#[cfg(feature = "fixed_log_softmax")]
fn argmax_first_and_remaining_mass(z: &[f32]) -> (usize, f32) {
    let mut a = 0;
    let mut best = z[0];
    for (i, &v) in z.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            a = i;
        }
    }
    let mut r = 0.0_f32;
    for (j, &v) in z.iter().enumerate() {
        if j != a {
            r += (v - best).exp();
        }
    }
    (a, r)
}

/// [`argmax_first_and_remaining_mass`] in f64.
#[cfg(feature = "fixed_log_softmax")]
fn argmax_first_and_remaining_mass_f64(z: &[f64]) -> (usize, f64) {
    let mut a = 0;
    let mut best = z[0];
    for (i, &v) in z.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            a = i;
        }
    }
    let mut r = 0.0_f64;
    for (j, &v) in z.iter().enumerate() {
        if j != a {
            r += (v - best).exp();
        }
    }
    (a, r)
}

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

    // ── T5 (fixed_log_softmax) — gated with the feature ──────────────
    #[cfg(feature = "fixed_log_softmax")]
    mod lsm {
        use super::*;

        /// ULP distance over the IEEE-754 total order (sign-magnitude
        /// bits remapped monotonically). ±0.0 are adjacent — the right
        /// metric for "within 1 ulp of truth" down the subnormal tail.
        fn f32_ulp_distance(a: f32, b: f32) -> u32 {
            let ordered = |x: f32| {
                let bits = x.to_bits();
                if bits & 0x8000_0000 != 0 {
                    !bits
                } else {
                    bits | 0x8000_0000
                }
            };
            ordered(a).abs_diff(ordered(b))
        }

        /// T5 G1 — the golden ladder: margins 0..~103 in 0.5-nat steps,
        /// two-slot rows, BOTH slots within the measured ulp budget of
        /// the f64 oracle on every rung — including the subnormal tail
        /// where exp(−m) itself is subnormal and the incumbent has been
        /// saturating since m ≈ 16.64. The budget is the f32 libm
        /// composition's (expf ≤ 1 ulp + log1p ≤ 1 ulp → worst 2), not
        /// the form's — the incumbent's error on the same rungs is
        /// 100 % (exact 0.0 saturation, witnessed below). The oracle
        /// closes with `ln_1p`, NOT `ln(1.0 + e^{−m})`: the naive form
        /// reproduces the paper's own collapse class INSIDE this test —
        /// measured on this box, at m = 28.5, `ln(1.0 + 4.1938e-13)`
        /// returns 4.19442e-13, a 1.5e-4 relative corruption of ε (the
        /// add round's half-ulp of 1.0 is 2.65e-4 of ε), while `ln_1p`
        /// is faithful. The single-slot form must be bit-identical to
        /// the slot the full form writes (same pinned fold, same
        /// branch).
        #[test]
        fn tracks_the_f64_oracle_across_the_underflow_band() {
            let mut worst = 0_u32;
            let mut worst_m = 0.0_f32;
            let mut m = 0.0_f32;
            while m <= 103.0 {
                let z = [0.0_f32, -m];
                let mut out = [0.0_f32; 2];
                fixed_log_softmax(&z, &mut out);
                let m64 = f64::from(m);
                let l = f64::exp(-m64).ln_1p();
                let oracle0 = -l;
                let oracle1 = -m64 - l;
                let d0 = f32_ulp_distance(out[0], oracle0 as f32);
                let d1 = f32_ulp_distance(out[1], oracle1 as f32);
                if d0.max(d1) > worst {
                    worst = d0.max(d1);
                    worst_m = m;
                }
                assert!(
                    d0 <= 2,
                    "m = {m}: target slot {} vs oracle {oracle0:e} ({d0} ulp)",
                    out[0]
                );
                assert!(
                    d1 <= 2,
                    "m = {m}: competitor slot {} vs oracle {oracle1:e} ({d1} ulp)",
                    out[1]
                );
                assert_eq!(
                    fixed_log_softmax_target(&z, 0).to_bits(),
                    out[0].to_bits(),
                    "m = {m}: target form must be bit-identical"
                );
                assert_eq!(
                    fixed_log_softmax_target(&z, 1).to_bits(),
                    out[1].to_bits(),
                    "m = {m}: target form must be bit-identical"
                );
                m += 0.5;
            }
            assert!(
                worst <= 2,
                "ladder worst {worst} ulp at m = {worst_m} — budget is 2"
            );
        }

        /// T5 G1 — the incumbent-saturation witness: the standard
        /// max-subtract form in f32 prices the target slot at EXACTLY
        /// 0.0 once the margin reaches the fp32 cliff (the `1 + e^{−m}`
        /// addition absorbs the term), while the fixed form keeps a
        /// strictly negative, oracle-faithful value. The sign of the
        /// zero is formulation-dependent; the zero itself is the defect.
        #[test]
        fn incumbent_saturates_at_zero_past_the_cliff() {
            let cliff = precision_cliff(FloatFormat::Fp32) as f32;
            for &m in &[cliff, 17.0_f32, 20.0, 40.0, 80.0] {
                let z = [0.0_f32, -m];
                let sum = (z[0] - z[0]).exp() + (z[1] - z[0]).exp();
                let incumbent = (z[0] - z[0]) - sum.ln();
                assert_eq!(
                    incumbent, 0.0,
                    "m = {m}: incumbent must saturate at exactly 0.0"
                );
                let mut out = [0.0_f32; 2];
                fixed_log_softmax(&z, &mut out);
                assert!(
                    out[0] < 0.0,
                    "m = {m}: fixed must resolve a negative log-prob, got {}",
                    out[0]
                );
            }
        }

        /// T5 G1 — a 5-slot row: a tie at the max (first-index rule —
        /// slot 0 is the argmax, the equal slot 4 prices like any
        /// competitor), and deep margins on slots 2–4 exercising the
        /// excluded-LSE fold. Every slot within the 2-ulp f32 libm
        /// composition budget of the f64 oracle.
        #[test]
        fn multi_slot_tie_and_deep_margins_track_the_oracle() {
            let z = [2.0_f32, -1.5, -17.0, -100.0, 2.0];
            let mut out = [0.0_f32; 5];
            fixed_log_softmax(&z, &mut out);
            let mut r = 0.0_f64;
            for (j, &v) in z.iter().enumerate() {
                if j != 0 {
                    r += f64::exp(f64::from(v) - 2.0);
                }
            }
            let l = f64::ln(1.0 + r);
            for (i, &v) in z.iter().enumerate() {
                let oracle = (f64::from(v) - 2.0) - l;
                let d = f32_ulp_distance(out[i], oracle as f32);
                assert!(
                    d <= 2,
                    "slot {i}: {} vs oracle {oracle:e} ({d} ulp)",
                    out[i]
                );
            }
            assert_eq!(fixed_log_softmax_target(&z, 0).to_bits(), out[0].to_bits());
        }

        /// All-equal rows price every slot at −ln(n); a single-element
        /// row is exactly 0.0.
        #[test]
        fn uniform_rows_and_singletons() {
            let z = [7.0_f32; 3];
            let mut out = [0.0_f32; 3];
            fixed_log_softmax(&z, &mut out);
            for &o in &out {
                assert!((f64::from(o) + 3.0f64.ln()).abs() < 1e-6);
            }
            let mut one = [9.9_f32];
            fixed_log_softmax(&[4.2_f32], &mut one);
            assert_eq!(one[0], 0.0);
            assert_eq!(fixed_log_softmax_target(&[4.2_f32], 0), 0.0);
        }

        /// f64 arm: the normalization identity e^{o0} + e^{o1} = 1 must
        /// hold to 1e-15 across the f64 resolution band (out to margins
        /// where the incumbent form saturated ~660 nats earlier), and
        /// the incumbent's own f64 saturation past its 36.7 cliff is
        /// witnessed.
        #[test]
        fn f64_arm_normalizes_across_the_band() {
            for &m in &[
                0.0_f64,
                1.0,
                precision_cliff(FloatFormat::Fp64),
                100.0,
                300.0,
                700.0,
            ] {
                let z = [0.0_f64, -m];
                let mut out = [0.0_f64; 2];
                fixed_log_softmax_f64(&z, &mut out);
                let mass = out[0].exp() + out[1].exp();
                assert!(
                    (mass - 1.0).abs() < 1e-15,
                    "m = {m}: e^{{o0}} + e^{{o1}} = {mass} != 1"
                );
                assert_eq!(
                    fixed_log_softmax_target_f64(&z, 0).to_bits(),
                    out[0].to_bits()
                );
            }
            // Incumbent witness at f64: past the 36.7 cliff the target
            // slot reads exactly 0.0 in the standard form too — the sum
            // of centered softmax terms is `1 + e^{−m}`, and the `1 +`
            // absorbs the term.
            let m = 40.0_f64;
            let sum = 1.0_f64 + (-m).exp();
            let incumbent = 0.0_f64 - sum.ln();
            assert_eq!(incumbent, 0.0);
            let mut out = [0.0_f64; 2];
            fixed_log_softmax_f64(&[0.0_f64, -m], &mut out);
            assert!(out[0] < 0.0);
        }

        /// Edges: a lone +inf max prices exactly — the argmax slot at
        /// 0.0 (never computing `inf − inf`), every competitor at −inf
        /// (the exact-math limit p = (1, 0, …)); the single-slot form
        /// agrees.
        #[test]
        fn inf_edges_are_exact() {
            let z = [f32::INFINITY, 3.0, f32::NEG_INFINITY];
            let mut out = [0.0_f32; 3];
            fixed_log_softmax(&z, &mut out);
            assert_eq!(out[0], 0.0);
            assert_eq!(out[1], f32::NEG_INFINITY);
            assert_eq!(out[2], f32::NEG_INFINITY);
            assert_eq!(fixed_log_softmax_target(&z, 0), 0.0);
            // Finite-max row with a −inf competitor: the max slot
            // prices against the finite rival alone.
            let zf = [5.0_f32, 3.0, f32::NEG_INFINITY];
            let mut outf = [0.0_f32; 3];
            fixed_log_softmax(&zf, &mut outf);
            let l = f64::ln_1p(f64::exp(-2.0)); // R = e^{−2} + 0
            let oracle_max = -l;
            assert!(
                f32_ulp_distance(outf[0], oracle_max as f32) <= 1,
                "max slot {} vs oracle {oracle_max:e}",
                outf[0]
            );
            assert_eq!(outf[2], f32::NEG_INFINITY);
        }
    }
}
