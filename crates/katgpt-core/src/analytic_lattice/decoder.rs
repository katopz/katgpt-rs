//! `direction_vector_decode` — SIMD projection of a latent state onto a
//! direction vector, producing an action-score scalar in `(0, 1)`.
//!
//! The GENERALIZED version of riir-games `scalar_projection::project_to_scalars`,
//! lifted out of HLA-specific 5-scalar semantics into a generic single-direction
//! primitive. The 5-scalar HLA bridge in riir-games becomes a thin wrapper that
//! calls this 5 times.
//!
//! # Math
//!
//! ```text
//! z   = dot(state, direction) / N            // normalized dot product
//! out = sigmoid(z * temperature)             // AGENTS.md: sigmoid, NOT softmax
//! ```
//!
//! The `/N` normalization keeps `z` in a sane range regardless of `N`, so a
//! single `temperature` hyperparameter transfers across dimensions.
//!
//! # Determinism
//!
//! Uses [`crate::simd::simd_dot_f32`] (NEON/AVX2/scalar dispatch) and
//! [`crate::simd::fast_sigmoid`] (stable `1/(1+e^{-x})`). Both are deterministic
//! and cross-architecture identical (the SIMD dispatch picks the same code path
//! given the same CPU features; `fast_sigmoid` uses `libm` `exp` which is
//! IEEE-754 deterministic).
//!
//! # Zero-alloc
//!
//! [`direction_vector_decode`] is fully stack-local (no heap). The batched
//! [`direction_vector_decode_into`] writes into a caller-provided slice.

use crate::analytic_lattice::LatticeVector;
use crate::simd::{fast_sigmoid, simd_dot_f32};

/// Project `state` onto `direction`, return scalar action score in `(0, 1)`.
///
/// `out = sigmoid((dot(state, direction) / N) * temperature)`
///
/// # Const generic
///
/// `N` is a compile-time constant (the eggshell default is 8). This lets the
/// compiler unroll the dot product and avoid slice bounds checks on the hot
/// path.
///
/// # Determinism
///
/// Bit-identical across runs given the same `(state, direction, temperature)`
/// and the same CPU feature set (NEON vs AVX2 vs scalar). Cross-architecture
/// determinism holds because `simd_dot_f32` uses FMA-preserving scalar fallback
/// and `fast_sigmoid` uses IEEE-754 `exp`.
#[inline]
pub fn direction_vector_decode<const N: usize>(
    state: &LatticeVector<N>,
    direction: &LatticeVector<N>,
    temperature: f32,
) -> f32 {
    let z = simd_dot_f32(state.as_slice(), direction.as_slice(), N) / N as f32;
    fast_sigmoid(z * temperature)
}

/// Batched decode: project a single `state` onto `directions.len()` direction
/// vectors, writing the scores into `out`.
///
/// This is the ASOC hot-path variant — used when one composite operator decodes
/// against multiple action-type direction vectors (e.g. 5 HLA scalars, or 8
/// action-type one-hots).
///
/// # Contract
///
/// - `out.len()` MUST equal `directions.len()`.
/// - `out[i] = direction_vector_decode(state, directions[i], temperature)`.
///
/// # Zero-alloc
///
/// No heap allocation. `state` is `Copy` (it's a fixed-size array under the
/// hood), so the loop body is pure stack arithmetic.
#[inline]
pub fn direction_vector_decode_into<const N: usize>(
    state: &LatticeVector<N>,
    directions: &[LatticeVector<N>],
    temperature: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(
        out.len(),
        directions.len(),
        "direction_vector_decode_into: out.len() != directions.len()"
    );
    let inv_n = 1.0 / N as f32;
    for (i, dir) in directions.iter().enumerate() {
        let z = simd_dot_f32(state.as_slice(), dir.as_slice(), N) * inv_n;
        out[i] = fast_sigmoid(z * temperature);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_returns_value_in_open_unit_interval() {
        let state = LatticeVector::<4>::new([1.0, 0.0, -1.0, 0.5]);
        let dir = LatticeVector::<4>::new([0.5, 0.5, 0.5, 0.5]);
        let score = direction_vector_decode(&state, &dir, 1.0);
        assert!(score > 0.0 && score < 1.0, "score {score} not in (0,1)");
    }

    #[test]
    fn decode_orthogonal_direction_is_half() {
        // dot(orthogonal vectors) = 0 → z = 0 → sigmoid(0) = 0.5
        let state = LatticeVector::<2>::new([1.0, 0.0]);
        let dir = LatticeVector::<2>::new([0.0, 1.0]);
        let score = direction_vector_decode(&state, &dir, 1.0);
        assert!((score - 0.5).abs() < 1e-6, "orthogonal decode {score} != 0.5");
    }

    #[test]
    fn decode_aligned_direction_gives_high_score() {
        // state aligned with direction → high dot → z > 0 → sigmoid(z*τ) > 0.5
        let state = LatticeVector::<4>::new([1.0, 1.0, 1.0, 1.0]);
        let dir = LatticeVector::<4>::new([1.0, 1.0, 1.0, 1.0]);
        let score = direction_vector_decode(&state, &dir, 2.0);
        assert!(score > 0.5, "aligned decode {score} <= 0.5");
    }

    #[test]
    fn decode_anti_aligned_direction_gives_low_score() {
        // state anti-aligned → z < 0 → sigmoid(z*τ) < 0.5
        let state = LatticeVector::<4>::new([1.0, 1.0, 1.0, 1.0]);
        let dir = LatticeVector::<4>::new([-1.0, -1.0, -1.0, -1.0]);
        let score = direction_vector_decode(&state, &dir, 2.0);
        assert!(score < 0.5, "anti-aligned decode {score} >= 0.5");
    }

    #[test]
    fn decode_is_deterministic() {
        let state = LatticeVector::<8>::new([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        let dir = LatticeVector::<8>::new([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]);
        let a = direction_vector_decode(&state, &dir, 1.5);
        let b = direction_vector_decode(&state, &dir, 1.5);
        assert_eq!(a.to_bits(), b.to_bits(), "decode not bit-identical");
    }

    #[test]
    fn decode_into_writes_all_directions() {
        let state = LatticeVector::<2>::new([1.0, 0.0]);
        let dirs = [
            LatticeVector::<2>::new([1.0, 0.0]),  // aligned → high
            LatticeVector::<2>::new([0.0, 1.0]),  // orthogonal → 0.5
            LatticeVector::<2>::new([-1.0, 0.0]), // anti-aligned → low
        ];
        let mut out = [0.0f32; 3];
        direction_vector_decode_into(&state, &dirs, 2.0, &mut out);

        assert!(out[0] > 0.5, "aligned should be > 0.5, got {}", out[0]);
        assert!((out[1] - 0.5).abs() < 1e-6, "orthogonal should be 0.5, got {}", out[1]);
        assert!(out[2] < 0.5, "anti-aligned should be < 0.5, got {}", out[2]);
    }

    #[test]
    fn decode_into_matches_scalar_version() {
        let state = LatticeVector::<4>::new([0.5, -0.3, 0.8, 0.1]);
        let dirs = [
            LatticeVector::<4>::new([1.0, 0.0, 0.0, 0.0]),
            LatticeVector::<4>::new([0.0, 1.0, 0.0, 0.0]),
            LatticeVector::<4>::new([0.0, 0.0, 1.0, 0.0]),
            LatticeVector::<4>::new([0.0, 0.0, 0.0, 1.0]),
        ];
        let mut batched = [0.0f32; 4];
        direction_vector_decode_into(&state, &dirs, 1.0, &mut batched);

        for (i, dir) in dirs.iter().enumerate() {
            let scalar = direction_vector_decode(&state, dir, 1.0);
            assert!(
                (batched[i] - scalar).abs() < 1e-7,
                "batched[{}] = {}, scalar = {}",
                i,
                batched[i],
                scalar
            );
        }
    }

    #[test]
    fn decode_ranking_matches_reference() {
        // G2 gate: ranking (by score) matches brute-force ranking.
        // Reference: brute-force dot product (no SIMD), then rank.
        let state = LatticeVector::<4>::new([0.9, 0.1, 0.5, 0.3]);
        let directions = [
            LatticeVector::<4>::new([1.0, 0.0, 0.0, 0.0]),
            LatticeVector::<4>::new([0.0, 1.0, 0.0, 0.0]),
            LatticeVector::<4>::new([0.5, 0.5, 0.5, 0.5]),
            LatticeVector::<4>::new([-1.0, 0.0, 0.0, 0.0]),
        ];

        let mut simd_scores: Vec<f32> = directions
            .iter()
            .map(|d| direction_vector_decode(&state, d, 1.0))
            .collect();

        // Reference: brute-force dot / N, then sigmoid.
        let mut ref_scores: Vec<f32> = directions
            .iter()
            .map(|d| {
                let z = state
                    .as_slice()
                    .iter()
                    .zip(d.as_slice())
                    .map(|(s, d)| s * d)
                    .sum::<f32>()
                    / 4.0;
                fast_sigmoid(z)
            })
            .collect();

        // Sort both by score descending.
        simd_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        ref_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Ranking must match exactly (sigmoid is monotone, so ranking is
        // determined by the dot product alone).
        for (i, (s, r)) in simd_scores.iter().zip(ref_scores.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-6,
                "rank {}: simd {} vs ref {}",
                i,
                s,
                r
            );
        }
    }
}
