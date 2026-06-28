//! Latent → raw bridge for `MicroRecurrentBeliefState`.
//!
//! This module exposes the bridge function used by all kernel families to
//! project a belief vector `s_t` to bounded raw scalars that may cross the sync
//! boundary. The bridge is intentionally a single free function — it contains
//! no family-specific logic, so every `MicroRecurrentBeliefState` impl delegates
//! to it (see Plan 276 T0.5: the existing `SenseModule::project` already does
//! dot-product + sigmoid; we reuse the same pattern, not duplicate it).
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! - `state` (the belief vector) is latent, local, never synced.
//! - `out` (the projected scalars) is raw, synced, replayed bit-identically.
//! - The direction vectors are latent and private to the caller (riir-ai owns
//!   the 5 emotion-channel direction vectors). They must never be synced
//!   alongside the scalars — otherwise an attacker could partially reconstruct
//!   `s_t` from the synced scalar stream.
//!
//! # Determinism
//!
//! Reuses `katgpt_types::simd::simd_dot_f32` (deterministic SIMD reduction — same
//! instruction order every run) and `katgpt_types::simd::fast_sigmoid` (exact libm
//! path, no polynomial approximation). Bit-identical across runs (G1.1).

use crate::types::project_to_scalars_bridge as impl_bridge;

/// Project a belief vector to K bounded scalars via sigmoid(dot).
///
/// For each `k` in `0..out.len()`:
///
/// ```text
/// out[k] = fast_sigmoid( simd_dot_f32(state, &directions[k*dim .. (k+1)*dim], dim) )
/// ```
///
/// # Arguments
///
/// - `state` — the belief vector `s_t`, length `dim`. Latent.
/// - `directions` — flattened `[K * dim]` row-major slice of projection
///   direction vectors. Latent, caller-owned, never synced.
/// - `dim` — the belief-vector dimension (must equal `state.len()` and the row
///   stride of `directions`).
/// - `out` — output buffer of length `K`. Raw, syncable.
///
/// # Zero-allocation
///
/// Reads `state` / `directions`, writes `out`. No `Vec`, no iterator chains
/// that allocate.
///
/// # Bridge ranking preservation (G1.3)
///
/// Because `fast_sigmoid` is strictly monotonically increasing, the bridge
/// preserves belief ranking: `dot_a > dot_b ⟺ sigmoid(dot_a) > sigmoid(dot_b)`.
/// This is the G1.3 property — verified in `tests.rs`.
#[inline(always)]
pub fn project_to_scalars(state: &[f32], directions: &[f32], dim: usize, out: &mut [f32]) {
    impl_bridge(state, directions, dim, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_types::simd::fast_sigmoid;

    #[test]
    fn bridge_zeros_for_zero_state() {
        let dim = 8usize;
        let state = [0.0f32; 8];
        let directions = [1.0f32; 16]; // 2 rows of 8
        let mut out = [0.0f32; 2];
        project_to_scalars(&state, &directions, dim, &mut out);
        // dot(0, ·) = 0 → sigmoid(0) = 0.5
        assert_eq!(out, [0.5, 0.5]);
    }

    #[test]
    fn bridge_saturates_for_large_aligned_dot() {
        let dim = 4usize;
        let state = [10.0f32; 4];
        let directions = [10.0f32; 4]; // 1 row of 4, dot = 400
        let mut out = [0.0f32; 1];
        project_to_scalars(&state, &directions, dim, &mut out);
        // sigmoid(400) saturates to 1.0 in f32
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn bridge_is_strictly_monotone() {
        // Increasing the dot must strictly increase the output.
        let dim = 2usize;
        let directions = [1.0f32, 0.0]; // dot = state[0]
        let mut out = [0.0f32; 1];
        for v in [-2.0f32, -0.5, 0.0, 0.5, 2.0] {
            let state = [v, 0.0f32];
            project_to_scalars(&state, &directions, dim, &mut out);
            let prev = if v == -2.0 { f32::NEG_INFINITY } else {
                let s = [v - 0.1, 0.0f32];
                let mut o = [0.0f32; 1];
                project_to_scalars(&s, &directions, dim, &mut o);
                o[0]
            };
            assert!(out[0] > prev, "not monotone at v={v}: out={o0:?} prev={prev:?}", o0 = out[0]);
        }
    }

    #[test]
    fn bridge_handles_k_larger_than_one() {
        let dim = 3usize;
        let state = [1.0f32, 1.0, 1.0];
        // 3 directions: identity axes
        let directions: [f32; 9] = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let mut out = [0.0f32; 3];
        project_to_scalars(&state, &directions, dim, &mut out);
        // Each dot = 1.0 → sigmoid(1) ≈ 0.731
        let expected = fast_sigmoid(1.0);
        for v in &out {
            assert!((v - expected).abs() < 1e-6);
        }
    }
}
