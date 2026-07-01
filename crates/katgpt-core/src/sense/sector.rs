//! SectorProjection — multi-sector spatial projection for NPC perception (Plan 262).
//!
//! Divides space around an NPC into `N` sectors and projects each into a latent
//! score using pre-computed ternary direction vectors (`{-1, 0, +1}`) and a
//! sigmoid non-linearity.
//!
//! Zero allocation, fixed-size. Uses `sigmoid(dot())` — never softmax.

// Sigmoid delegates to shared crate::simd::fast_sigmoid (bounded (0,1), libm-exp).

#![allow(clippy::needless_range_loop)]

/// Multi-sector spatial projection for NPC perception.
///
/// Divides space around an NPC into `N` sectors, projects each into a latent
/// score using pre-computed ternary direction vectors (`{-1, 0, +1}`) and a
/// sigmoid non-linearity.
///
/// Zero allocation, fixed-size. Uses `sigmoid(dot())` — never softmax.
///
/// # Direction storage: `f32`, not `i8`
///
/// Direction vectors are stored as `f32` even though the public constructor
/// takes `i8` ternary values. This trades 4× direction memory for SIMD
/// auto-vectorization: an `i8 as f32` cast inside the dot-product loop
/// previously defeated LLVM's vectorizer, forcing a scalar reduction. With
/// pre-converted `f32` directions, the inner loop is a plain FMA chain that
/// the compiler maps to `vfmla`/`vfmadd` lanes. Direction memory is `N·D·4`
/// bytes (e.g. 256 B at N=8, D=8) — negligible vs the L1-resident observation
/// vector it multiplies against.
pub struct SectorProjection<const N: usize, const D: usize> {
    /// Pre-computed direction vectors per sector as `f32` (cast once at
    /// construction from ternary `i8` inputs). Storing `f32` instead of the
    /// original `i8` lets the dot-product inner loop auto-vectorize — the
    /// `i8 as f32` cast was the one instruction LLVM refused to hoist out of
    /// the reduction. Direction vectors are immutable after `new()` /
    /// `update_directions()`, so the f32 representation is the canonical form.
    sector_directions: [[f32; D]; N],
    /// Last projection scores per sector (updated on `project` call).
    scores: [f32; N],
}

impl<const N: usize, const D: usize> SectorProjection<N, D> {
    /// Creates a new `SectorProjection` from pre-computed direction vectors.
    ///
    /// The `i8` ternary directions are converted to `f32` once at construction
    /// (see struct docs for the SIMD rationale). Scores are initialized to
    /// zero; call `project` to compute them.
    #[inline]
    pub fn new(directions: [[i8; D]; N]) -> Self {
        // One-time i8→f32 cast: pays 4× direction memory for SIMD-vectorizable
        // dot products on every `project()` call thereafter.
        let mut f32_directions = [[0.0f32; D]; N];
        for i in 0..N {
            for j in 0..D {
                f32_directions[i][j] = directions[i][j] as f32;
            }
        }
        Self {
            sector_directions: f32_directions,
            scores: [0.0; N],
        }
    }

    /// Projects an observation vector into per-sector latent scores.
    ///
    /// For each sector `i`: `scores[i] = sigmoid(dot(observation, sector_directions[i]))`.
    ///
    /// Zero allocation — writes into the internal fixed-size buffer. Inner
    /// dot-product is auto-vectorizable (plain `f32 × f32` FMA chain).
    ///
    /// Returns a reference to the updated scores array.
    #[inline]
    pub fn project(&mut self, observation: &[f32; D]) -> &[f32; N] {
        for i in 0..N {
            let dir = &self.sector_directions[i];
            let mut dot = 0.0f32;
            // Plain f32 FMA chain — LLVM maps to SIMD lanes (vfmla on NEON,
            // vfmadd on AVX2). The earlier `i8 as f32` cast here blocked this.
            for j in 0..D {
                dot = observation[j].mul_add(dir[j], dot);
            }
            self.scores[i] = crate::simd::fast_sigmoid(dot);
        }
        &self.scores
    }

    /// Hot-swaps direction vectors without restarting the NPC.
    ///
    /// Useful for adaptive behavior — e.g., shifting attention sectors based on
    /// game phase or threat level.
    #[inline]
    pub fn update_directions(&mut self, new_directions: [[i8; D]; N]) {
        for i in 0..N {
            for j in 0..D {
                self.sector_directions[i][j] = new_directions[i][j] as f32;
            }
        }
    }

    /// Read-only access to the last computed scores.
    ///
    /// Returns the scores from the most recent `project` call.
    #[inline]
    pub const fn scores(&self) -> &[f32; N] {
        &self.scores
    }
}

impl<const N: usize, const D: usize> Default for SectorProjection<N, D> {
    #[inline]
    fn default() -> Self {
        Self::new([[0; D]; N])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_output_range() {
        // 2 sectors, 4-dim observation
        let directions: [[i8; 4]; 2] = [[1, -1, 0, 1], [-1, 0, 1, -1]];
        let mut proj = SectorProjection::new(directions);

        let obs: [f32; 4] = [1.5, -2.0, 0.3, 4.1];
        let scores = proj.project(&obs);

        for &s in scores.iter() {
            assert!(s > 0.0 && s < 1.0, "score {s} out of range (0, 1)");
        }
    }

    #[test]
    fn test_project_known_value() {
        // Single sector: direction = [1, 0], observation = [0.0, 0.0]
        // dot = 0 → sigmoid(0) = 0.5
        let directions: [[i8; 2]; 1] = [[1, 0]];
        let mut proj = SectorProjection::new(directions);

        let obs: [f32; 2] = [0.0, 0.0];
        let scores = proj.project(&obs);
        assert!(
            (scores[0] - 0.5).abs() < 1e-5,
            "sigmoid(0) should be 0.5, got {}",
            scores[0]
        );
    }

    #[test]
    fn test_different_observations_produce_different_scores() {
        let directions: [[i8; 3]; 1] = [[1, 1, 1]];
        let mut proj = SectorProjection::new(directions);

        let obs_a: [f32; 3] = [10.0, 10.0, 10.0];
        let obs_b: [f32; 3] = [-10.0, -10.0, -10.0];

        let score_a = proj.project(&obs_a)[0];
        let score_b = proj.project(&obs_b)[0];

        assert!(
            score_a > 0.99,
            "large positive dot should be near 1, got {score_a}"
        );
        assert!(
            score_b < 0.01,
            "large negative dot should be near 0, got {score_b}"
        );
        assert_ne!(score_a, score_b);
    }

    #[test]
    fn test_update_directions_changes_result() {
        let directions: [[i8; 2]; 1] = [[1, 0]];
        let mut proj = SectorProjection::new(directions);

        let obs: [f32; 2] = [5.0, 5.0];
        let score_before = proj.project(&obs)[0];
        assert!(score_before > 0.99, "dot=5 → near 1, got {score_before}");

        // Flip direction: now dot = -5
        proj.update_directions([[-1, 0]]);
        let score_after = proj.project(&obs)[0];
        assert!(score_after < 0.01, "dot=-5 → near 0, got {score_after}");

        assert_ne!(score_before, score_after);
    }

    #[test]
    fn test_scores_accessor_matches_project() {
        let directions: [[i8; 3]; 2] = [[1, 0, -1], [0, 1, 0]];
        let mut proj = SectorProjection::new(directions);

        let obs: [f32; 3] = [1.0, 2.0, 3.0];
        let project_result = proj.project(&obs);
        let project_copy = *project_result;

        // scores() should match the last project output
        let accessor_scores = proj.scores();
        assert_eq!(project_copy.len(), accessor_scores.len());
        for i in 0..project_copy.len() {
            assert!(
                (project_copy[i] - accessor_scores[i]).abs() < 1e-7,
                "scores mismatch at {i}: {} vs {}",
                project_copy[i],
                accessor_scores[i]
            );
        }
    }

    #[test]
    fn test_zero_size_edge_case_n1() {
        // N=1, D=1: minimal valid configuration
        let directions: [[i8; 1]; 1] = [[1]];
        let mut proj = SectorProjection::new(directions);

        let obs: [f32; 1] = [2.0];
        let scores = proj.project(&obs);

        assert_eq!(scores.len(), 1);
        let expected = crate::simd::fast_sigmoid(2.0);
        assert!(
            (scores[0] - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            scores[0]
        );
    }

    #[test]
    fn test_zero_directions_yield_half() {
        // All-zero directions → dot=0 → sigmoid(0)=0.5 for all sectors
        let directions: [[i8; 4]; 3] = [[0; 4], [0; 4], [0; 4]];
        let mut proj = SectorProjection::new(directions);

        let obs: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let scores = proj.project(&obs);

        for &s in scores.iter() {
            assert!((s - 0.5).abs() < 1e-5, "zero dot → 0.5, got {s}");
        }
    }

    #[test]
    fn test_default_is_zero_scores() {
        let proj: SectorProjection<4, 2> = SectorProjection::default();
        for &s in proj.scores().iter() {
            assert_eq!(s, 0.0);
        }
    }
}
