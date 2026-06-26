//! Analytic Lattice — k×k transport operator chain composer + ASOC trait
//! shapes + direction-vector SIMD decoder + spectral audit.
//!
//! Distillation of R311 (revised): the cross-entity analog of Functional
//! Attention's token-level operator algebra. Plan 330 (katgpt-core half).
//!
//! # Layering contract (critical)
//!
//! This module ships ONLY:
//! - **Generic trait shapes** ([`PlasmaDraft`], [`RederiveOp`], [`ComposerCtx`])
//!   with **NO** `GpuFuture` import. [`RederiveOp::Fut`] has no bound at the
//!   trait level — the `GpuFuture<Output = TransportOperator>` bound is applied
//!   at the impl site in `riir-engine` (Phase 1b, separate task).
//! - **Pure math primitives**: [`compose_chain`], [`batch_compose_chain`],
//!   [`direction_vector_decode`], [`spectral_audit`].
//!
//! It does NOT ship `ComposerTick` (the `GpuFuture` impl) or `Join3` — those
//! live in `riir-engine/src/analytic_lattice/asoc.rs` because they need
//! `riir-gpu-async`, which is private to riir-ai. Adding that dep here would
//! invert the 5-repo commercial boundary (katgpt-core is the leaf crate; it
//! must not depend on riir-ai's private runtime).
//!
//! # Constraints (per AGENTS.md + Plan 330)
//!
//! - **Modelless**: deterministic closed-form math only. No training, no
//!   backprop, no gradient descent.
//! - **Sigmoid, NOT softmax**: the decoder uses `sigmoid(dot · τ / N)`.
//! - **Zero-alloc hot paths**: [`compose_chain_into`], [`batch_compose_chain_into`],
//!   [`direction_vector_decode`] must not allocate on the hot path (G5 gate).
//! - **BLAKE3** for any commitment (per AGENTS.md).
//!
//! # References
//!
//! - Research: [katgpt-rs/.research/311_Analytic_Lattice_Encoder_Decoder_Primitive.md](../../../.research/311_Analytic_Lattice_Encoder_Decoder_Primitive.md)
//! - Plan: [katgpt-rs/.plans/330_analytic_lattice_encoder_decoder_primitive.md](../../../.plans/330_analytic_lattice_encoder_decoder_primitive.md)
//! - Token-level analog: [`crate::funcattn`] (Tikhonov k×k spectral transport)
//! - Asymmetric generalization: [`crate::cross_resolution`]
//! - DCT-II audit inspiration: arxiv 2606.02427

pub mod asoc;
pub mod audit;
pub mod batch_chain;
pub mod chain;
pub mod decoder;

pub use asoc::{ComposerCtx, PlasmaDraft, RederiveOp};
pub use audit::{AuditReport, spectral_audit};
pub use batch_chain::{batch_compose_chain, batch_compose_chain_into};
pub use chain::{ChainError, compose_chain, compose_chain_into};
pub use decoder::{direction_vector_decode, direction_vector_decode_into};

// ── Core types ─────────────────────────────────────────────────────────────

/// Typed per-slot lattice vector. Slot semantics are CALLER-defined (game IP);
/// this primitive is slot-agnostic. The 8-lane eggshell default matches
/// Plan 335's transport lanes (k=8), but any const-generic N works.
///
/// `#[repr(transparent)]` makes `LatticeVector<N>` ABI-identical to `[f32; N]`,
/// so it can be passed to SIMD kernels without a transmute.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct LatticeVector<const N: usize>(pub [f32; N]);

impl<const N: usize> LatticeVector<N> {
    /// Construct from a fixed-size array.
    #[inline]
    pub const fn new(data: [f32; N]) -> Self {
        Self(data)
    }

    /// Construct an all-zeros vector.
    #[inline]
    pub const fn zero() -> Self {
        Self([0.0; N])
    }

    /// View as a slice (for SIMD dispatch).
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// View as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.0
    }
}

impl<const N: usize> Default for LatticeVector<N> {
    #[inline]
    fn default() -> Self {
        Self::zero()
    }
}

impl<const N: usize> From<[f32; N]> for LatticeVector<N> {
    #[inline]
    fn from(data: [f32; N]) -> Self {
        Self(data)
    }
}

/// A k×k transport operator stored as a row-major `Vec<f32>` of length `k²`.
///
/// The cross-entity analog of [`crate::funcattn`]'s token-level k×k operators.
/// Produced by FuncAttn (`funcattn_forward`), `cross_resolution_transport`, or
/// any other rederive op. [`compose_chain`] reduces a sequence of these into a
/// single composite operator via row-major matmul.
///
/// **Layout:** `data[r * k + c]` is row `r`, column `c` (standard math
/// convention). All operators in a chain MUST share the same `k`.
#[derive(Clone, Debug)]
pub struct TransportOperator {
    /// Operator dimension `k` (square).
    pub k: usize,
    /// Row-major `k × k` data, length `k²`.
    pub data: Vec<f32>,
}

impl TransportOperator {
    /// Construct from a flat row-major slice. Validates `data.len() == k²`.
    pub fn from_row_major(k: usize, data: Vec<f32>) -> Result<Self, ChainError> {
        let expected = k.checked_mul(k).ok_or(ChainError::DimensionOverflow)?;
        if data.len() != expected {
            return Err(ChainError::DimensionMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self { k, data })
    }

    /// Construct from a flat row-major slice WITHOUT validation. Caller
    /// guarantees `data.len() == k²`. Used when the data comes from a
    /// trusted source (e.g. a freshly-allocated scratch buffer of known size).
    ///
    /// Debug-asserts the invariant; callers MUST ensure `data.len() == k * k`.
    /// Subsequent operations (matmul, audit) will panic or produce garbage if
    /// this is violated.
    pub fn from_row_major_unchecked(k: usize, data: Vec<f32>) -> Self {
        debug_assert_eq!(data.len(), k * k, "from_row_major_unchecked: len != k*k");
        Self { k, data }
    }

    /// k×k identity operator.
    pub fn identity(k: usize) -> Self {
        let mut data = vec![0.0f32; k * k];
        for i in 0..k {
            data[i * k + i] = 1.0;
        }
        Self { k, data }
    }

    /// Allocate a zeroed operator with the given `k`. Useful as the `out`
    /// slot for [`compose_chain_into`].
    pub fn zeros(k: usize) -> Self {
        Self {
            k,
            data: vec![0.0f32; k * k],
        }
    }

    /// Resize in-place. Resets all entries to zero. Useful for reusing an
    /// operator buffer across chains of different k (rare; prefer fixed-k).
    pub fn resize(&mut self, k: usize) {
        self.k = k;
        self.data.clear();
        self.data.resize(k * k, 0.0);
    }

    /// View as a row-major `k × k` slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// View as a mutable row-major `k × k` slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Read entry at `(row, col)` (row-major). Panics if out of bounds.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.k + col]
    }

    /// Write entry at `(row, col)` (row-major). Panics if out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.k + col] = value;
    }

    /// Frobenius norm `sqrt(Σ|a_ij|²)`. Used by the GOAT G3 associativity test.
    pub fn frobenius_norm(&self) -> f32 {
        let s: f32 = self.data.iter().map(|x| x * x).sum();
        s.sqrt()
    }
}

impl Default for TransportOperator {
    fn default() -> Self {
        Self::identity(0)
    }
}

/// `out ← op · v` matvec. `out.len()` MUST equal `op.k`.
///
/// Reuses [`crate::simd::simd_matvec`] — no allocation on the hot path.
#[inline]
pub fn apply_operator_into(op: &TransportOperator, v: &[f32], out: &mut [f32]) {
    debug_assert_eq!(v.len(), op.k, "apply_operator_into: v.len() != op.k");
    debug_assert_eq!(out.len(), op.k, "apply_operator_into: out.len() != op.k");
    crate::simd::simd_matvec(out, op.as_slice(), v, op.k, op.k);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_vector_zero_and_new() {
        let v = LatticeVector::<4>::new([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        let z = LatticeVector::<4>::zero();
        assert_eq!(z.as_slice(), &[0.0; 4]);
    }

    #[test]
    fn transport_operator_identity() {
        let i = TransportOperator::identity(3);
        assert_eq!(i.k, 3);
        assert_eq!(
            i.as_slice(),
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn transport_operator_from_row_major_validates() {
        let good = TransportOperator::from_row_major(2, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(good.is_ok());
        let bad = TransportOperator::from_row_major(2, vec![1.0, 2.0, 3.0]);
        assert!(matches!(bad, Err(ChainError::DimensionMismatch { .. })));
    }

    #[test]
    fn apply_operator_into_identity_is_passthrough() {
        let i = TransportOperator::identity(3);
        let v = [1.0, 2.0, 3.0];
        let mut out = [0.0f32; 3];
        apply_operator_into(&i, &v, &mut out);
        assert_eq!(out, v);
    }
}
