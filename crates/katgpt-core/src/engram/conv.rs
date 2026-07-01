//! Depthwise causal 1D convolution (paper §2.3 eq 5).
//!
//! Plan 299 Phase 3 T3.7. The Engram paper applies a small depthwise causal
//! conv over the retrieved memory patterns `Ṽ` before the residual fuse:
//!
//! ```text
//! Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ
//! ```
//!
//! This module implements the `Conv1D` term only — the caller does RMSNorm +
//! SiLU + residual add. Keeping the conv as a standalone zero-alloc primitive
//! lets the host compose it with the sigmoid gate and projection weights in
//! any order (paper §2.4 puts conv *after* the mHC shared-`V` projection;
//! other orderings are valid for ablation).
//!
//! # CRITICAL — never softmax
//!
//! Per AGENTS.md this module contains **no `softmax` symbol**. The conv is a
//! purely linear operator (weighted sum of past taps); the only nonlinearity
//! in the paper's recipe is `SiLU`, which the applied by the caller. The
//! sigmoid gate lives in [`crate::engram::kernel`].
//!
//! # Conv Zero Init
//!
//! Per the paper's "Conv Zero Init" hyperparameter, the default kernel is
//! [`IDENTITY_KERNEL`] = `[0, 0, 1, 0]`. With this kernel the conv is the
//! identity: `out == v_tilde`. The output then reduces to pure residual
//! (`Y = SiLU(RMSNorm(Ṽ)) + Ṽ`), matching the "no conv at init" training
//! stability trick used in the paper's pretrained checkpoints.
//!
//! # Indexing convention
//!
//! The kernel is laid out left-to-right from **oldest** to **newest** tap:
//! - `kernel[0]` = tap at `t - 3 * dilation` (oldest)
//! - `kernel[1]` = tap at `t - 2 * dilation`
//! - `kernel[2]` = tap at `t - 1 * dilation`
//! - `kernel[3]` = tap at `t - 0 * dilation` (current)
//!
//! This matches the standard CNN causal-conv convention. The spec's literal
//! `[0, 0, 1, 0]` activates `kernel[2]` (the 1-step-back tap), which is **not**
//! strictly identity under this convention. To honor the spec's intent —
//! "zero conv → pure residual" — we interpret "identity" as "the conv output
//! equals the input bit-identically", which requires the current-tap weight
//! to be 1. So [`IDENTITY_KERNEL`] = `[0, 0, 0, 1]` for strict identity.
//!
//! But the spec's literal `[0, 0, 1, 0]` is also exposed as
//! [`SPEC_KERNEL`] for direct paper-text reproduction. The unit test for
//! "identity kernel → out == v_tilde" uses [`IDENTITY_KERNEL`] (strict).
//!
//! # Layout
//!
//! `v_tilde` and `out` are flat slices treated as a 1D signal. For a true
//! depthwise conv across `D` channels, the caller loops:
//!
//! ```text
//! for d in 0..D {
//!     conv_causal_into(&v_tilde[d..], &mut out[d..], kernel, dilation);
//!     // strided by D — caller slices `v_tilde[d..n*D].step_by(D)`.
//! }
//! ```
//!
//! The flat-slice signature keeps the API simple and matches the spec.
//!
//! # Hot-path contract
//!
//! [`conv_causal_into`] is **zero-allocation**: caller provides `out` of size
//! `v_tilde.len()`. The kernel is a fixed-size `[f32; 4]` stack value. Inner
//! loop is `O(4n)` multiply-adds with at most 3 boundary checks per output.

/// Identity kernel — strict passthrough (`out == v_tilde`).
///
/// `kernel[3]` (current-position tap) is 1; all others are 0. With this
/// kernel the conv contributes nothing and the residual
/// `Y = SiLU(RMSNorm(Ṽ)) + Ṽ` is recovered. This is the operational form
/// of the paper's "Conv Zero Init" hyperparameter — training (when done in
/// riir-train) starts from a known-good baseline.
pub const IDENTITY_KERNEL: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// Spec-literal kernel — the paper text's `[0, 0, 1, 0]`.
///
/// Under our left-to-right oldest→newest convention, `[0, 0, 1, 0]` activates
/// `kernel[2]` (the 1-step-back tap). With dilation=1 this shifts `v_tilde`
/// forward by 1 (with a leading zero) — NOT identity. Kept here for
/// paper-text reproduction; use [`IDENTITY_KERNEL`] for strict passthrough.
pub const SPEC_KERNEL: [f32; 4] = [0.0, 0.0, 1.0, 0.0];

/// Zero kernel — true zero conv.
///
/// All taps are 0, so `out = 0` and `Y = SiLU(0) + Ṽ = Ṽ`. This is the
/// strictest reading of "Conv Zero Init".
pub const ZERO_KERNEL: [f32; 4] = [0.0; 4];

/// Apply a depthwise causal 1D convolution to `v_tilde`, writing into `out`.
///
/// Plan 299 Phase 3 T3.7. See the module docs for the layout and zero-init
/// convention.
///
/// # Formula
///
/// For each position `t ∈ [0, n)`:
/// ```text
/// out[t] = Σ_{j=0..4} kernel[j] * v_tilde[t - (3 - j) * δ]
/// ```
/// where `δ = max(dilation, 1)` and out-of-range indices contribute 0.
/// `kernel[3]` is the current-position tap; `kernel[0]` is the oldest
/// (3 × dilation positions back).
///
/// # Arguments
///
/// - `v_tilde` — input slice. Treated as a 1D signal of length `n`.
/// - `out` — output slice. MUST equal `v_tilde.len()` (debug_asserted).
/// - `kernel` — 4 tap weights. See [`IDENTITY_KERNEL`] for the passthrough.
/// - `dilation` — stride between taps. `dilation = 1` is a standard causal
///   conv; the paper uses `dilation = max N-gram order` (= 3 for trigram).
///   `dilation = 0` is treated as 1 (degenerate).
///
/// # Panics (debug only)
///
/// `debug_assert!` checks `out.len() == v_tilde.len()`. Zero-length input is
/// a no-op.
#[inline]
pub fn conv_causal_into(v_tilde: &[f32], out: &mut [f32], kernel: [f32; 4], dilation: usize) {
    let n = v_tilde.len();
    if n == 0 {
        return;
    }
    debug_assert_eq!(
        out.len(),
        n,
        "conv_causal_into: out.len() must equal v_tilde.len()"
    );

    let dil = dilation.max(1) as isize;
    for (t, out_slot) in out.iter_mut().enumerate().take(n) {
        let mut acc = 0.0f32;
        // kernel[0] = oldest tap (t - 3δ); kernel[3] = current (t - 0δ).
        // Out-of-range taps contribute 0 (zero-padding at the left edge).
        for (j, &k) in kernel.iter().enumerate() {
            let offset = (3 - j) as isize * dil;
            let tap_t = t as isize - offset;
            if tap_t >= 0 {
                acc += k * v_tilde[tap_t as usize];
            }
        }
        *out_slot = acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_kernel_is_strict_passthrough() {
        // IDENTITY_KERNEL → out == v_tilde, bit-identically, for any input.
        let v_tilde = [1.0f32, 2.0, 3.0, 4.0, 5.0, 7.0, 11.0, 13.0];
        let mut out = [0.0f32; 8];
        conv_causal_into(&v_tilde, &mut out, IDENTITY_KERNEL, 1);
        assert_eq!(out, v_tilde, "IDENTITY_KERNEL → out == v_tilde");

        // Also true for dilation > 1 (the current tap is always in range).
        let mut out2 = [0.0f32; 8];
        conv_causal_into(&v_tilde, &mut out2, IDENTITY_KERNEL, 3);
        assert_eq!(out2, v_tilde, "identity holds for any dilation");

        // And dilation = 0 (treated as 1).
        let mut out3 = [0.0f32; 8];
        conv_causal_into(&v_tilde, &mut out3, IDENTITY_KERNEL, 0);
        assert_eq!(out3, v_tilde, "identity holds for dilation=0 (→ 1)");
    }

    #[test]
    fn zero_kernel_produces_zero_output() {
        let v_tilde = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [99.0f32; 5];
        conv_causal_into(&v_tilde, &mut out, ZERO_KERNEL, 1);
        assert!(out.iter().all(|&v| v == 0.0), "ZERO_KERNEL → all zeros");
    }

    #[test]
    fn zero_input_produces_zero_output() {
        // Any kernel × zero input = zero output.
        let v_tilde = [0.0f32; 8];
        let mut out = [99.0f32; 8];
        let kernel = [0.25, 0.25, 0.25, 0.25]; // averaging kernel
        conv_causal_into(&v_tilde, &mut out, kernel, 1);
        assert!(out.iter().all(|&v| v == 0.0), "zero input → zero output");
    }

    #[test]
    fn non_trivial_kernel_convolves() {
        // kernel = [0.0, 0.0, 0.5, 0.5] → out[t] = 0.5*v[t] + 0.5*v[t-1]
        // (current + previous, both with weight 0.5).
        // At t=0: only current tap in range → out[0] = 0.5 * v[0].
        // At t=1+: out[t] = 0.5 * (v[t] + v[t-1]).
        let v_tilde = [2.0f32, 4.0, 6.0, 8.0];
        let mut out = [0.0f32; 4];
        let kernel = [0.0, 0.0, 0.5, 0.5];
        conv_causal_into(&v_tilde, &mut out, kernel, 1);

        // Expected: out[0] = 0.5*2 = 1; out[1] = 0.5*4 + 0.5*2 = 3; etc.
        let expected = [1.0f32, 3.0, 5.0, 7.0];
        for i in 0..4 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-6,
                "out[{i}] = {}, expected {}",
                out[i],
                expected[i]
            );
        }
    }

    #[test]
    fn averaging_kernel_smooths() {
        // kernel = [0.25; 4] → moving average over 4 taps.
        // At t=0: only current tap in range → 0.25 * v[0].
        // At t=1: current + 1-back → 0.25 * (v[1] + v[0]).
        // At t=3+: all 4 taps in range → 0.25 * sum.
        let v_tilde = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut out = [0.0f32; 6];
        let kernel = [0.25; 4];
        conv_causal_into(&v_tilde, &mut out, kernel, 1);

        // out[0..3] = 0.25, 0.5, 0.75 (partial window); out[3..] = 1.0 (full).
        assert!((out[0] - 0.25).abs() < 1e-6, "out[0] = {}", out[0]);
        assert!((out[1] - 0.5).abs() < 1e-6, "out[1] = {}", out[1]);
        assert!((out[2] - 0.75).abs() < 1e-6, "out[2] = {}", out[2]);
        for (i, oi) in out[3..6].iter().enumerate() {
            assert!((*oi - 1.0).abs() < 1e-6, "out[{}] = {}", i + 3, oi);
        }
    }

    #[test]
    fn dilation_stretches_tap_stride() {
        // kernel = [0.0, 0.0, 1.0, 0.0], dilation = 2 → out[t] = v[t-2].
        // At t < 2, the tap is out of range → out[t] = 0.
        let v_tilde = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let mut out = [0.0f32; 5];
        let kernel = [0.0, 0.0, 1.0, 0.0]; // tap at offset 1*δ = 2
        conv_causal_into(&v_tilde, &mut out, kernel, 2);

        // Expected: out = [0, 0, 10, 20, 30] (shift by 2).
        let expected = [0.0f32, 0.0, 10.0, 20.0, 30.0];
        for i in 0..5 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-6,
                "out[{i}] = {}, expected {}",
                out[i],
                expected[i]
            );
        }
    }

    #[test]
    fn empty_input_is_noop() {
        let v_tilde: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        conv_causal_into(&v_tilde, &mut out, IDENTITY_KERNEL, 1); // must not panic
    }

    #[test]
    fn spec_kernel_is_one_step_shift() {
        // SPEC_KERNEL = [0, 0, 1, 0] — paper text's literal value. Under our
        // convention this activates kernel[2] = tap at offset δ. The output
        // is v_tilde shifted forward by δ (with leading zeros). Document this
        // so the discrepancy between spec text and behavior is explicit.
        let v_tilde = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0f32; 5];
        conv_causal_into(&v_tilde, &mut out, SPEC_KERNEL, 1);
        // out[t] = v[t-1] (or 0 if t < 1)
        let expected = [0.0f32, 1.0, 2.0, 3.0, 4.0];
        for i in 0..5 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-6,
                "SPEC_KERNEL out[{i}] = {}, expected {}",
                out[i],
                expected[i]
            );
        }
    }
}
