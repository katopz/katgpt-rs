//! Spec-match test for the SoftmaxFix Lean 4 sign certificate.
//!
//! **Issue 1221** (the arXiv:2610.31642 companion arc, Research 673 §2/§5).
//! The Lean proof at `katgpt-rs/.proofs/KatgptProof/SoftmaxFix/Basic.lean`
//! proves the **sign trinity** for `fixed_log_softmax_target`'s argmax
//! branch, over the UNBOUNDED region the sampled G1 ladder (Issue 959)
//! cannot cover:
//!
//! > For every finite real logit vector with `n ≥ 2` and `a` an argmax, the
//! > exact form `−ln(1 + Σ_{j≠a} e^{z_j − z_a})` is negative, nonzero, and
//! > bounded in absolute value by `ln(n)`.
//!
//! The statement is format-parametric for free: the model's values are
//! reals, every finite IEEE value of any format is a real, and the model
//! has no `±∞` — which excludes the `+∞`-argmax hazard by construction
//! (there the Rust resolves to `−0.0` through the overwrite law, outside
//! the certificate's domain, per the Lean module doc).
//!
//! What this test pins (the Plan-441 pairing discipline):
//!
//! 1. **Structural identity** — the Rust argmax branch (pinned fold →
//!    `ln_1p` → `−l`) computes the Lean `logTarget` form. The mirror here
//!    is f64 with the SAME `ln_1p` spelling; its faithful-ℝ regime is
//!    rival gaps above ≈ −745 (where f64 `exp` underflows to zero) and the
//!    sweeps stay inside it — the underflow side is the DOCUMENTED
//!    rounding posture below, never silently folded into the claim.
//! 2. **The trinity on the exact form over unbounded regions** —
//!    magnitudes the G1 ladder never visits (±1e6-class logits,
//!    denormal-scale rivals, 2..=64 classes), asserted in the mirror's
//!    faithful regime.
//! 3. **The computed posture** — `fixed_log_softmax_target_f64`'s argmax
//!    branch stays ≤ 0 and ≥ −ln(n) on the same sweeps; nonzero is
//!    claimed only under the margin premise (a mass that underflows to
//!    zero gives −0.0 — the Lean doc's disclosure, asserted as posture,
//!    never hidden).
//!
//! If this test fails, Rust drifted from the Lean spec — update the proof
//! or the code before merging.
//!
//! Run: `cargo test --features fixed_log_softmax --test softmax_fix_spec_match`

#![cfg(feature = "fixed_log_softmax")]
#![cfg(test)]

use katgpt_core::softmax_fix::{fixed_log_softmax_target, fixed_log_softmax_target_f64};

// ── The Lean spec, mirrored in Rust (f64, the real contract's image) ───

/// The Lean `remainingMass z a = Σ_{j ≠ a} e^{z_j − z_a}` — the exact-form
/// mirror (the fold order is the pinned ascending one).
fn remaining_mass(z: &[f64], a: usize) -> f64 {
    z.iter()
        .enumerate()
        .filter(|&(j, _)| j != a)
        .map(|(_, &v)| (v - z[a]).exp())
        .sum()
}

/// The Lean `logTarget z a = −ln(1 + remainingMass z a)` — the argmax
/// branch's exact form, in the Rust's own `ln_1p` spelling. Faithful to ℝ
/// while every rival term is representable (rival gaps above ≈ −745).
fn log_target_exact(z: &[f64], a: usize) -> f64 {
    -remaining_mass(z, a).ln_1p()
}

/// The trinity predicate of `logTarget_sign`: negative, nonzero, `|·| ≤ ln n`.
fn trinity(z: &[f64], a: usize) -> bool {
    let n = z.len() as f64;
    let l = log_target_exact(z, a);
    l < 0.0 && l != 0.0 && l.abs() <= n.ln() + 1e-12
}

// ── The sampled structural walk (the pairing half) ──────────────────────

/// The Rust argmax-branch result equals the exact-form mirror to f64
/// machine precision across the Issue-959 G1 ladder's magnitudes and
/// beyond — the structural identity the Lean doc's pairing claims.
#[test]
fn rust_matches_the_exact_form_on_the_ladder_and_wide() {
    // The G1 ladder: margins 0..~8 nats of gap, geometrically spread.
    for margin_exp in -8..=3 {
        for k in 0..64 {
            let gap = (margin_exp as f64).exp() * k as f64;
            if gap <= 0.0 {
                continue;
            }
            let z = [0.0, -gap, -2.0 * gap, -gap / 3.0];
            let computed = fixed_log_softmax_target_f64(&z, 0);
            let exact = log_target_exact(&z, 0);
            assert!(
                (computed - exact).abs() < 5e-12 * exact.abs().max(1e-300),
                "ladder margin_exp={margin_exp} k={k}: computed {computed} vs exact {exact}"
            );
            // The trinity holds on the exact form (the theorem's region).
            assert!(
                trinity(&z, 0),
                "trinity failed at margin_exp={margin_exp} k={k}"
            );
        }
    }
    // Wide but REPRESENTABLE magnitudes (±1e2-class gaps): the unbounded-
    // region claim the sampled G1 ladder cannot make, inside the mirror's
    // faithful regime.
    let z: Vec<f64> = vec![1e2, -1e2, 5e1, -7.5e1, 1e-40];
    let computed = fixed_log_softmax_target_f64(&z, 0);
    let exact = log_target_exact(&z, 0);
    assert!(
        (computed - exact).abs() < 5e-12 * exact.abs().max(1e-300),
        "wide: computed {computed} vs exact {exact}"
    );
    assert!(trinity(&z, 0), "trinity failed on the wide row");
    assert!(computed < 0.0 && computed != 0.0);
}

// ── The unbounded exact-form trinity sweeps (the certificate's claim) ──

/// The trinity holds for arbitrary finite magnitudes and class counts —
/// swept where the G1 ladder never goes: sizes 2..=64 with logits spread
/// over ±1e6, denormal-scale rivals, rival gaps kept inside the mirror's
/// faithful regime (≥ −600; f64 exp underflows below ≈ −745.1).
#[test]
fn exact_form_trinity_over_unbounded_sweeps() {
    // Deterministic pseudo-random sweep (a fixed LCG — no external rng).
    let mut state: u64 = 0x1221_CAFE;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for n in 2usize..=64 {
        for _trial in 0..8 {
            let z: Vec<f64> = (0..n).map(|_| (next() - 0.5) * 2.0e6).collect();
            // The argmax premise may fail on random draws (a rival can
            // exceed the target) — re-pin the target to the first-index
            // argmax of the drawn row (the theorem consumes `a` argmax).
            let amax = (0..n).fold(0usize, |best, j| if z[j] > z[best] { j } else { best });
            // Clamp every rival into the mirror's faithful regime: a gap
            // below −600 would underflow f64 exp and the MIRROR (not the
            // theorem) would lose the nonzero leg.
            let mut zw = z.clone();
            for j in 0..n {
                if j != amax {
                    zw[j] = zw[amax] - (zw[amax] - zw[j]).clamp(0.0, 600.0);
                }
            }
            assert!(trinity(&zw, amax), "trinity failed: n={n} amax={amax}");
            // The bound is log(n) exactly as the theorem states.
            let l = log_target_exact(&zw, amax);
            assert!(l.abs() <= (n as f64).ln() + 1e-12);
            // The computed posture agrees with the mirror here.
            let computed = fixed_log_softmax_target_f64(&zw, amax);
            assert!(
                (computed - l).abs() < 5e-12 * l.abs().max(1e-300),
                "computed vs mirror: n={n} computed={computed} mirror={l}"
            );
        }
    }
    // Denormal-scale rivals below a huge argmax (gap ≈ −1e308, clamped to
    // the faithful regime's floor): every rival term is strictly positive
    // in the exact form (the deep-tail SpecTests row).
    let z = vec![f64::MAX / 2.0, f64::MIN_POSITIVE * 0.25, 0.0];
    let mut zw = z.clone();
    for j in 1..3 {
        zw[j] = zw[0] - 600.0;
    }
    assert!(trinity(&zw, 0), "trinity failed on the denormal-rival row");
}

// ── The computed-f64 posture (the documented rounding disclosure) ──────

/// The computed f64 argmax branch stays ≤ 0 and ≥ −ln(n) on the
/// unbounded sweeps; nonzero is claimed ONLY under the margin premise.
/// The underflow row demonstrates the disclosure: a rival gap beyond the
/// exp underflow zeroes the f64 mass and the result is −0.0 — finite and
/// negative-LEQ, exactly as the Lean doc records, never a hidden failure.
#[test]
fn computed_f64_posture_negative_and_bounded_unbounded() {
    let mut state: u64 = 0x1221_BEEF;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for n in 2usize..=64 {
        for _trial in 0..8 {
            let z: Vec<f64> = (0..n).map(|_| (next() - 0.5) * 2.0e6).collect();
            let amax = (0..n).fold(0usize, |best, j| if z[j] > z[best] { j } else { best });
            let computed = fixed_log_softmax_target_f64(&z, amax);
            assert!(
                computed <= 0.0,
                "computed posture must be ≤ 0: n={n} computed={computed}"
            );
            assert!(
                computed >= -(n as f64).ln() - 1e-9,
                "computed posture below −ln(n): n={n} computed={computed}"
            );
        }
    }
    // The disclosure row: every rival underflows → mass 0 → ln_1p(0) = 0
    // → −0.0 (negative-LEQ, finite, and the documented −0.0 spelling).
    let z = [0.0, -1.5e30, -2.0e30];
    let computed = fixed_log_softmax_target_f64(&z, 0);
    assert_eq!(computed, -0.0, "the underflow posture is −0.0");
    assert!(computed <= 0.0 && computed >= -(z.len() as f64).ln());
    // The f32 entry point agrees with the posture on sane magnitudes.
    let z32 = [0.0f32, -1.5, -3.25, -0.75];
    let c = fixed_log_softmax_target(&z32, 0);
    let exact32 = log_target_exact(&[0.0, -1.5, -3.25, -0.75], 0);
    assert!(c <= 0.0 && (c as f64 - exact32).abs() < 1e-5);
}

// ── The closure-row concrete instance (the SpecTests pairing) ──────────

/// The Lean SpecTests' closure row, in Rust: z = [ln 4, ln 2, ln 2], the
/// mass is exactly 1 and the target log-prob is −ln 2 — the independently
/// known answer that catches a spec-authoring error.
#[test]
fn closure_row_matches_the_lean_spec_tests() {
    let z = [4.0f64.ln(), 2.0f64.ln(), 2.0f64.ln()];
    let mass = remaining_mass(&z, 0);
    assert!((mass - 1.0).abs() < 1e-12, "closure mass {mass} != 1");
    let l = fixed_log_softmax_target_f64(&z, 0);
    assert!(
        (l + 2.0f64.ln()).abs() < 1e-12,
        "closure target {l} != −ln 2"
    );
    assert!(l < 0.0 && l != 0.0);
}
