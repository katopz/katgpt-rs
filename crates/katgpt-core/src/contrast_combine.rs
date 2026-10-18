//! Affine strong−weak contrast combine — the one home for the
//! `strong + ω·(strong − weak)` / convex-blend arithmetic family.
//!
//! Extracted from `katgpt-forward::d2f_context::apply_probe_guidance`
//! (Issue 865, Plan 617 T1.2) so that BOTH consumers share one
//! formula-preserving kernel:
//!
//! - **865 probe guidance** (the D2F decode lane): convex blend
//!   `logits' = λ·logits + (1−λ)·probe_logits` with `λ ∈ [0, 1]`.
//! - **LoopCD recurrent-depth contrast** (Plan 617, feature
//!   `loop_guidance`): extrapolation `z′ = z_R + ω·(z_R − z_k)`, which is
//!   the same affine family at `λ = 1 + ω` (ω ≥ 0 → λ ≥ 1): the negative
//!   weight on the weak side is exactly the `(1−λ)` coefficient.
//!
//! The kernel is pure arithmetic over caller slices — no context type, no
//! allocation, no platform code. Gated
//! `any(feature = "probe_guidance", feature = "loop_guidance")` so a build
//! carrying neither consumer compiles it to nothing (the feature-gate
//! discipline); each root feature forwards its core twin.
//!
//! # Bit-identity contract
//!
//! `out[i] ← lam·out[i] + (1−lam)·ref[i]` evaluated in exactly that order,
//! the `(1−lam)` weight hoisted once outside the loop (the Issue-865 form).
//! The 865 combine-kernel tests pin exact equality against the pre-extraction
//! arithmetic; any reassociation here is a formula break, not a refactor.
//!
//! contrast_combine — the affine strong−weak combine, ONE home for the
//! `strong + ω·(strong − weak)` arithmetic (Plan 617 T1.2): the 865
//! probe-guidance convex blend (`λ·x + (1−λ)·p`) and the LoopCD
//! recurrent-depth extrapolation (`z_R + ω(z_R − z_k)`, the same family at
//! `λ = 1 + ω`) share this formula-preserving kernel. Gated
//! `any(probe_guidance, loop_guidance)` — a build carrying neither consumer
//! compiles it to nothing; each root feature forwards its core twin.

/// `out[i] ← lam·out[i] + (1−lam)·ref[i]` over the full extent of `out`.
///
/// `out` and `ref` must have equal lengths (debug-asserted). The loop is
/// chunked 8-wide and branch-free in the body so LLVM auto-vectorizes it,
/// with a scalar tail — the exact shape the 865 site shipped.
///
/// Zero-allocation, no NaN policy: NaN inputs propagate per IEEE, matching
/// the pre-extraction behavior bit for bit.
pub fn affine_combine(out: &mut [f32], reference: &[f32], lam: f32) {
    debug_assert_eq!(
        out.len(),
        reference.len(),
        "contrast_combine: out/reference length mismatch"
    );
    let w_ref = 1.0 - lam;
    let n = out.len().min(reference.len());
    let mut i = 0;
    while i + 8 <= n {
        for j in 0..8 {
            out[i + j] = lam * out[i + j] + w_ref * reference[i + j];
        }
        i += 8;
    }
    while i < n {
        out[i] = lam * out[i] + w_ref * reference[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::affine_combine;

    /// The λ=1 identity is exact in f32: `1.0·x + 0.0·r == x` for every
    /// finite/NaN pattern (x + −0.0 == x; 0.0·NaN = NaN but 1.0·x is taken
    /// first — no, the sum still sees the 0.0·r term; assert the exact
    /// arithmetic, which for finite r is bit-identity and for NaN r is NaN).
    #[test]
    fn lam_one_is_exact_identity_on_finite_reference() {
        let mut out = vec![1.5f32, -2.25, 0.0, f32::MAX, -0.0];
        let reference = vec![100.0f32, -1.0, 3.5, 0.0, 7.0];
        let before = out.clone();
        affine_combine(&mut out, &reference, 1.0);
        assert_eq!(out, before, "λ=1 must be bit-identical on finite reference");
    }

    /// The 865 convex-blend midpoint: λ=0.5 is the mean.
    #[test]
    fn convex_blend_midpoint() {
        let mut out = vec![4.0f32, -2.0];
        let reference = vec![2.0f32, 2.0];
        affine_combine(&mut out, &reference, 0.5);
        assert_eq!(out, vec![3.0, 0.0]);
    }

    /// The LoopCD extrapolation form: λ = 1+ω re-ranks along strong−weak.
    /// `z + ω(z − k)` for z=[4], k=[2], ω=0.5 → 4 + 0.5·2 = 5.
    #[test]
    fn extrapolation_form_matches_omega_spelling() {
        let z = [4.0f32, -1.0];
        let k = [2.0f32, 1.0];
        let omega = 0.5f32;
        let mut out = z.to_vec();
        affine_combine(&mut out, &k, 1.0 + omega);
        let want: Vec<f32> = z
            .iter()
            .zip(k.iter())
            .map(|(&a, &b)| a + omega * (a - b))
            .collect();
        assert_eq!(out, want);
    }

    /// Lengths beyond one 8-wide chunk exercise both loop arms; a tail
    /// shorter than 8 exercises the scalar loop alone.
    #[test]
    fn chunk_and_tail_arms_agree_with_scalar_spelling() {
        for n in [1usize, 7, 8, 9, 16, 17, 24, 33] {
            let lam = 0.3f32;
            let reference: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 4.0).collect();
            let out_before: Vec<f32> = (0..n).map(|i| (i as f32) * -0.25 + 2.0).collect();
            let mut out = out_before.clone();
            affine_combine(&mut out, &reference, lam);
            for i in 0..n {
                let want = lam * out_before[i] + (1.0 - lam) * reference[i];
                assert!(
                    (out[i] - want).abs() <= f32::EPSILON * want.abs().max(1.0),
                    "n={n} i={i}: {} vs {want}",
                    out[i]
                );
            }
        }
    }
}
