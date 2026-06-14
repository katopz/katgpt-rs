//! SectorProjection — multi-sector spatial projection for NPC perception (Plan 262).
//!
//! Divides space around an NPC into `N` sectors and projects each into a latent
//! score using pre-computed ternary direction vectors (`{-1, 0, +1}`) and a
//! sigmoid non-linearity.
//!
//! Zero allocation, fixed-size. Uses `sigmoid(dot())` — never softmax.

// Sigmoid delegates to shared crate::simd::fast_sigmoid (bounded (0,1), libm-exp).

/// Dot product of an `f32` observation vector with an `i8` ternary direction vector.
///
/// `sum(observation[i] * direction[i] as f32)`
#[inline(always)]
fn dot_f32_i8<const D: usize>(observation: &[f32; D], direction: &[i8; D]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..D {
        acc += observation[i] * direction[i] as f32;
    }
    acc
}

/// Multi-sector spatial projection for NPC perception.
///
/// Divides space around an NPC into `N` sectors, projects each into a latent
/// score using pre-computed ternary direction vectors.
///
/// Zero allocation, fixed-size. Uses `sigmoid(dot())` — never softmax.
///
/// ## Type Parameters
///
/// - `N` — Number of sectors (e.g., 4 for quadrant, 8 for octant).
/// - `D` — Dimension of the observation/latent vector (e.g., 8 for HLA state).
pub struct SectorProjection<const N: usize, const D: usize> {
    /// Pre-computed direction vectors per sector (ternary `{-1, 0, +1}`).
    sector_directions: [[i8; D]; N],
    /// Last projection scores per sector (updated on `project` call).
    scores: [f32; N],
}

impl<const N: usize, const D: usize> SectorProjection<N, D> {
    /// Creates a new `SectorProjection` from pre-computed direction vectors.
    ///
    /// Scores are initialized to zero. Call `project` to compute them.
    #[inline]
    pub fn new(directions: [[i8; D]; N]) -> Self {
        Self {
            sector_directions: directions,
            scores: [0.0; N],
        }
    }

    /// Projects an observation vector into per-sector latent scores.
    ///
    /// For each sector `i`: `scores[i] = sigmoid(dot(observation, sector_directions[i]))`.
    ///
    /// Zero allocation — writes into the internal fixed-size buffer.
    ///
    /// Returns a reference to the updated scores array.
    #[inline]
    pub fn project(&mut self, observation: &[f32; D]) -> &[f32; N] {
        for i in 0..N {
            let dot = dot_f32_i8(observation, &self.sector_directions[i]);
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
        self.sector_directions = new_directions;
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
