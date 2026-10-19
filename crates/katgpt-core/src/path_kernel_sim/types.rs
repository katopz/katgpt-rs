//! Decoupled types for the sensitivity-fingerprint kernel (Issue 949 T1).
//!
//! The compute half lives in [`crate::path_kernel_sim`] (mod.rs); this file
//! carries the vocabulary so consumers can name the structs without the
//! compute machinery.

/// The scorer abstraction the kernel differentiates.
///
/// A [`KnobScorer`] is an *authored, untrained* deterministic scorer
/// `f(x; w)` — input `x`, `p ≤ [MAX_KNOBS]` authored knobs `w` held at a
/// fixed operating point `w0`. The fingerprint is `∇_w f(x; w0)`.
///
/// Two differentiation arms (Issue 949 T2):
/// - **closed-form** — implement [`KnobScorer::grad_knobs_closed_form`]; the
///   canonical sigmoid-gated shape `f = σ(w·φ(x))` differentiates to
///   `σ'(z)·φ(x)` with `σ'` computed from the SAME
///   [`crate::simd::fast_sigmoid`] the scorer itself evaluates with
/// - **central difference** — leave the hook returning `false`; the kernel
///   runs the pinned-`h` stencil per knob in fixed ascending order
pub trait KnobScorer {
    /// Knob count `p` (≤ [`MAX_KNOBS`]).
    fn knobs(&self) -> usize;

    /// Evaluate the scorer at input `x` with knobs `w`.
    fn eval(&self, x: &[f32], w: &[f32]) -> f32;

    /// Write `∇_w f(x; w)` into `out` (len [`KnobScorer::knobs`]).
    /// Return `true` when written; `false` falls back to the
    /// central-difference arm. Default: not available.
    fn grad_knobs_closed_form(&self, _x: &[f32], _w: &[f32], _out: &mut [f32]) -> bool {
        false
    }
}

/// Knob-count ceiling (the POC envelope; Issue 949 T6 sizes benches at 64).
pub const MAX_KNOBS: usize = 64;

/// Default central-difference step. Pinned (Issue 949 T2) — the stencil is
/// `f(w+h·e_j) − f(w−h·e_j)` over `2h`, evaluated knob-ascending. `1e-3`
/// sits near the f32 central-difference optimum for unit-scale knobs.
pub const DEFAULT_CENTRAL_DIFF_H: f32 = 1.0e-3;

/// Default near-zero-norm floor for the abstain guard. A fingerprint whose
/// L2 norm falls under the floor ABSTAINS (sentinel, never NaN) — the
/// T1 stationary-point case (`∇_w sin(wx) = x·cos(wx) = 0` at the sine's
/// stationary knobs).
///
/// ⚠ The central-difference arm carries cancellation noise of order
/// `eps·|f|/h` per component; a caller differentiating an opaque scorer
/// whose gradient can be genuinely tiny should RAISE the floor above that
/// noise (the guard is a config, not a constant, for exactly this reason).
pub const DEFAULT_ZERO_NORM_FLOOR: f32 = 1.0e-6;

/// Kernel configuration (Issue 949 T1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FingerprintKernel {
    /// Central-difference step for the opaque arm.
    pub central_diff_h: f32,
    /// Near-zero-norm abstain floor (L2 over the p-dim fingerprint).
    pub zero_norm_floor: f32,
}

impl Default for FingerprintKernel {
    fn default() -> Self {
        Self {
            central_diff_h: DEFAULT_CENTRAL_DIFF_H,
            zero_norm_floor: DEFAULT_ZERO_NORM_FLOOR,
        }
    }
}

/// Owned convenience form of one anchor's fingerprint — for storage, tests
/// and the fixture battery. The hot path writes caller-owned scratch via
/// [`crate::path_kernel_sim::fingerprint_into`] and never builds this.
///
/// `norm <= 0.0` is the ABSTAIN sentinel ([`SensitivityFingerprint::is_abstain`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SensitivityFingerprint {
    /// The p-dim gradient `∇_w f(x; w0)` (zeroed on abstain).
    pub grad: Vec<f32>,
    /// L2 norm at write time (`0.0` = abstain).
    pub norm: f32,
}

impl SensitivityFingerprint {
    /// The abstain sentinel (Issue 949 T1's zero-norm guard).
    pub fn abstain(p: usize) -> Self {
        Self {
            grad: vec![0.0; p],
            norm: 0.0,
        }
    }

    /// `true` when the anchor abstained (near-zero-norm fingerprint).
    pub fn is_abstain(&self) -> bool {
        self.norm <= 0.0
    }
}
