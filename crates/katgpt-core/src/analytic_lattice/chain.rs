//! `compose_chain` — k×k transport operator chain product.
//!
//! Computes `out = C[n-1] × ... × C[1] × C[0]` for an arbitrary-length chain
//! of [`TransportOperator`]s. The cross-entity analog of `funcattn_compose`
//! (which is token-level).
//!
//! # Layout
//!
//! All operators are row-major `k × k`. The product is also row-major `k × k`.
//! All operators in a chain MUST share the same `k`.
//!
//! # Hot-path API
//!
//! [`compose_chain_into`] is the zero-alloc variant for ASOC's
//! `ComposerTick::poll` — it reuses a caller-provided scratch buffer and writes
//! into a caller-provided `out` operator. [`compose_chain`] is the convenience
//! variant (allocates the result).
//!
//! # Numerical stability
//!
//! Chain length is capped at [`MAX_CHAIN_LEN`] (16) in v1 per Plan 330 risk
//! mitigation. For longer chains, normalize each operator (operator norm ≤ 1)
//! before multiplication and document the renormalization. Beyond 16, the
//! Frobenius-error associativity budget (G3 gate, 1e-5) cannot be guaranteed.

use crate::analytic_lattice::TransportOperator;
use crate::simd::simd_dot_f32;

/// Maximum chain length supported by [`compose_chain`] in v1.
///
/// Capped to keep the G3 associativity gate (Frobenius error ≤ 1e-5)
/// satisfiable. Longer chains accumulate rounding error super-linearly; the
/// cap is a conservative bound based on k=8 eggshell lanes with well-conditioned
/// operators. Extend only after re-verifying G3 at the new length.
pub const MAX_CHAIN_LEN: usize = 16;

/// Errors returned by [`compose_chain`] / [`compose_chain_into`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainError {
    /// Chain length is 0 (no operators to compose) or exceeds [`MAX_CHAIN_LEN`].
    ChainLengthInvalid { len: usize, max: usize },
    /// Operators in the chain have mismatched `k` values.
    DimensionMismatchK { expected: usize, got: usize },
    /// An operator's internal `data.len()` is not `k²` (corrupted buffer).
    DimensionMismatch { expected: usize, got: usize },
    /// `k * k` overflowed `usize` (degenerate input).
    DimensionOverflow,
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChainLengthInvalid { len, max } => {
                write!(f, "chain length {len} invalid (max {max})")
            }
            Self::DimensionMismatchK { expected, got } => {
                write!(f, "operator k mismatch: expected {expected}, got {got}")
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::DimensionOverflow => write!(f, "k*k overflow"),
        }
    }
}

impl std::error::Error for ChainError {}

/// Compose a chain of k×k transport operators: `out = C[n-1] × ... × C[0]`.
///
/// Allocates the result operator. For hot paths, use [`compose_chain_into`] to
/// reuse a scratch buffer.
///
/// # Errors
///
/// - [`ChainError::ChainLengthInvalid`] if `ops` is empty or longer than 16.
/// - [`ChainError::DimensionMismatchK`] if operators have different `k`.
/// - [`ChainError::DimensionMismatch`] if any operator's `data.len() != k²`.
pub fn compose_chain(ops: &[TransportOperator]) -> Result<TransportOperator, ChainError> {
    if ops.is_empty() || ops.len() > MAX_CHAIN_LEN {
        return Err(ChainError::ChainLengthInvalid {
            len: ops.len(),
            max: MAX_CHAIN_LEN,
        });
    }
    let k = ops[0].k;
    validate_same_k_and_size(ops, k)?;

    let mut out = TransportOperator::zeros(k);
    let mut scratch = Vec::with_capacity(k * k);
    compose_chain_into(ops, &mut scratch, &mut out)?;
    Ok(out)
}

/// In-place chain compose for hot paths (zero-alloc after warmup).
///
/// Writes the composite operator into `out`. The `scratch` buffer is used as
/// the intermediate accumulator and is grown to `k²` if smaller (no-op on the
/// hot path after the first call).
///
/// # Contract
///
/// - `out.k` MUST equal `ops[0].k` (resize with [`TransportOperator::resize`]
///   if reusing across different k).
/// - `out.data.len()` MUST equal `out.k²`.
/// - `scratch` is reused across calls; grow it once, never alloc again.
///
/// # Algorithm
///
/// Reduces left-to-right with one matmul per step. For a 3-op chain
/// `[A, B, C]`, computes `tmp = B × A`, then `out = C × tmp`. This matches
/// the paper's `C[n-1] × ... × C[0]` convention.
pub fn compose_chain_into(
    ops: &[TransportOperator],
    scratch: &mut Vec<f32>,
    out: &mut TransportOperator,
) -> Result<(), ChainError> {
    if ops.is_empty() || ops.len() > MAX_CHAIN_LEN {
        return Err(ChainError::ChainLengthInvalid {
            len: ops.len(),
            max: MAX_CHAIN_LEN,
        });
    }

    let k = ops[0].k;
    let k2 = k.checked_mul(k).ok_or(ChainError::DimensionOverflow)?;
    validate_same_k_and_size(ops, k)?;

    // Ensure scratch is sized once; clear() keeps capacity, never reallocates
    // after the first grow.
    if scratch.len() < k2 {
        scratch.resize(k2, 0.0);
    }
    if out.k != k {
        out.resize(k);
    }

    match ops.len() {
        1 => {
            // Single-op chain: just copy.
            out.data.copy_from_slice(&ops[0].data);
        }
        2 => {
            // Two-op chain: out = C[1] × C[0], no scratch needed.
            matmul_row_major(&ops[1].data, &ops[0].data, &mut out.data, k);
        }
        _ => {
            // Multi-op chain: reduce left-to-right.
            // tmp = C[1] × C[0]
            matmul_row_major(&ops[1].data, &ops[0].data, scratch, k);
            // for i in 2..n-1: tmp = C[i] × tmp
            for i in 2..ops.len() - 1 {
                // out = C[i] × tmp
                matmul_row_major(&ops[i].data, scratch, &mut out.data, k);
                // tmp = out  (swap roles; avoids a copy by ping-ponging)
                scratch.copy_from_slice(&out.data);
            }
            // Final step: out = C[n-1] × tmp
            matmul_row_major(
                &ops[ops.len() - 1].data,
                scratch,
                &mut out.data,
                k,
            );
        }
    }

    Ok(())
}

/// Row-major matmul: `out = a × b` where all three are `k × k`.
///
/// Uses [`simd_dot_f32`] for the inner product — same pattern as `funcattn`.
/// Inner loop is branch-free and FMA-friendly.
#[inline]
fn matmul_row_major(a: &[f32], b: &[f32], out: &mut [f32], k: usize) {
    debug_assert_eq!(a.len(), k * k);
    debug_assert_eq!(b.len(), k * k);
    debug_assert_eq!(out.len(), k * k);

    // For each (row i in a, col j in b): out[i*k+j] = Σ_l a[i*k+l] * b[l*k+j]
    //
    // `b` is accessed by column (strided), which is cache-unfriendly. For k ≤ 16
    // (the v1 cap) the whole matrix fits in L1, so this is fine. If we ever
    // raise MAX_CHAIN_LEN past ~32 we'd want a tiled / transposed-B layout.
    //
    // We pull b's column into a stack-local scratch to make the dot call
    // contiguous (one pass over b per column). k ≤ 16 keeps this ≤ 64 bytes.
    let mut col_buf = [0.0f32; MAX_CHAIN_LEN];
    let col = &mut col_buf[..k];

    for i in 0..k {
        let a_row = &a[i * k..(i + 1) * k];
        let out_row = &mut out[i * k..(i + 1) * k];
        for j in 0..k {
            // Gather column j of b into col (contiguous for the dot).
            for l in 0..k {
                col[l] = b[l * k + j];
            }
            out_row[j] = simd_dot_f32(a_row, col, k);
        }
    }
}

/// Validate all operators share `k` and have `data.len() == k²`.
fn validate_same_k_and_size(ops: &[TransportOperator], k: usize) -> Result<(), ChainError> {
    let expected = k.checked_mul(k).ok_or(ChainError::DimensionOverflow)?;
    for op in ops {
        if op.k != k {
            return Err(ChainError::DimensionMismatchK {
                expected: k,
                got: op.k,
            });
        }
        if op.data.len() != expected {
            return Err(ChainError::DimensionMismatch {
                expected,
                got: op.data.len(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_2x2(a: f32, b: f32, c: f32, d: f32) -> TransportOperator {
        TransportOperator::from_row_major(2, vec![a, b, c, d]).unwrap()
    }

    #[test]
    fn empty_chain_errors() {
        let err = compose_chain(&[]).unwrap_err();
        assert_eq!(
            err,
            ChainError::ChainLengthInvalid {
                len: 0,
                max: MAX_CHAIN_LEN
            }
        );
    }

    #[test]
    fn too_long_chain_errors() {
        let id = TransportOperator::identity(2);
        let ops: Vec<TransportOperator> = std::iter::repeat_n(id, MAX_CHAIN_LEN + 1).collect();
        let err = compose_chain(&ops).unwrap_err();
        assert_eq!(
            err,
            ChainError::ChainLengthInvalid {
                len: MAX_CHAIN_LEN + 1,
                max: MAX_CHAIN_LEN
            }
        );
    }

    #[test]
    fn single_op_chain_is_copy() {
        let a = make_2x2(1.0, 2.0, 3.0, 4.0);
        let out = compose_chain(&[a.clone()]).unwrap();
        assert_eq!(out.as_slice(), a.as_slice());
    }

    #[test]
    fn identity_pair_is_identity() {
        let id = TransportOperator::identity(3);
        let out = compose_chain(&[id.clone(), id.clone()]).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((out.get(i, j) - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn two_op_chain_matches_manual_matmul() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // B × A = [[5*1+6*3, 5*2+6*4], [7*1+8*3, 7*2+8*4]] = [[23,34],[31,46]]
        let a = make_2x2(1.0, 2.0, 3.0, 4.0);
        let b = make_2x2(5.0, 6.0, 7.0, 8.0);
        let out = compose_chain(&[a, b]).unwrap();
        assert!((out.get(0, 0) - 23.0).abs() < 1e-5);
        assert!((out.get(0, 1) - 34.0).abs() < 1e-5);
        assert!((out.get(1, 0) - 31.0).abs() < 1e-5);
        assert!((out.get(1, 1) - 46.0).abs() < 1e-5);
    }

    #[test]
    fn k_mismatch_errors() {
        let a = TransportOperator::identity(2);
        let b = TransportOperator::identity(3);
        let err = compose_chain(&[a, b]).unwrap_err();
        assert_eq!(err, ChainError::DimensionMismatchK {
            expected: 2,
            got: 3
        });
    }

    #[test]
    fn compose_chain_into_reuses_scratch() {
        let a = make_2x2(1.0, 2.0, 3.0, 4.0);
        let b = make_2x2(5.0, 6.0, 7.0, 8.0);
        let mut scratch = Vec::new();
        let mut out = TransportOperator::zeros(2);

        compose_chain_into(&[a.clone(), b.clone()], &mut scratch, &mut out).unwrap();
        assert!((out.get(0, 0) - 23.0).abs() < 1e-5);

        // Reuse — scratch should already be sized, no reallocation.
        let scratch_cap_before = scratch.capacity();
        compose_chain_into(&[a, b], &mut scratch, &mut out).unwrap();
        assert_eq!(scratch.capacity(), scratch_cap_before);
        assert!((out.get(0, 0) - 23.0).abs() < 1e-5);
    }

    #[test]
    fn associativity_g3_hold_within_tolerance() {
        // G3 gate core: (A×B)×C ≈ A×(B×C) within Frobenius ≤ 1e-5.
        // Use well-conditioned small-norm operators (so rounding is bounded).
        let a = make_2x2(0.6, 0.1, 0.2, 0.5);
        let b = make_2x2(0.3, 0.4, 0.7, 0.2);
        let c = make_2x2(0.5, 0.3, 0.1, 0.6);

        // (A×B)×C  →  chain [A, B] then [that, C]
        let ab = compose_chain(&[a.clone(), b.clone()]).unwrap();
        let left = compose_chain(&[ab, c.clone()]).unwrap();

        // A×(B×C)  →  chain [B, C] then [A, that]
        let bc = compose_chain(&[b, c]).unwrap();
        let right = compose_chain(&[a, bc]).unwrap();

        let frob_err: f32 = left
            .as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(l, r)| (l - r).abs())
            .sum();
        assert!(
            frob_err < 1e-5,
            "G3 FAIL: associativity Frobenius error {frob_err} >= 1e-5"
        );
    }
}
