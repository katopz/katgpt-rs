//! HyperplanePruner — Geometric half-space intersection of constraint pruners (Plan 234).
//!
//! Composes multiple ConstraintPruners via their constraint_vector() outputs.
//! Valid = geometric intersection of all half-spaces.
//! Soft scoring = product of sigmoid(-distance/temperature) per constraint.

use katgpt_core::traits::ConstraintPruner;

/// HyperplanePruner: intersects multiple constraint pruners as half-spaces.
pub struct HyperplanePruner<'a> {
    pub pruners: Vec<&'a dyn ConstraintPruner>,
    pub temperature: f32,
}

impl<'a> HyperplanePruner<'a> {
    pub fn new(pruners: Vec<&'a dyn ConstraintPruner>) -> Self {
        Self {
            pruners,
            temperature: 1.0,
        }
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Batch manifold score for all candidates at once.
    /// Writes scores into `scores` slice: `scores[i] = manifold_score(depth, candidates[i], parent_tokens)`.
    /// Processes all candidates in one pass — no per-candidate trait dispatch overhead.
    #[cfg(feature = "manifold_pruner")]
    pub fn batch_manifold_score(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        scores: &mut [f32],
    ) {
        let len = candidates.len().min(scores.len());
        // Initialize all scores to 1.0 (product identity)
        for s in scores.iter_mut().take(len) {
            *s = 1.0;
        }
        // For each pruner, compute its contribution and multiply in-place
        for pruner in &self.pruners {
            let has_cv = pruner.constraint_vector(depth, parent_tokens).is_some();
            for i in 0..len {
                // Skip candidates already zeroed by a previous pruner
                if scores[i] <= 0.0 {
                    continue;
                }
                let score = if has_cv {
                    let raw = pruner.manifold_score(depth, candidates[i], parent_tokens);
                    let x = (raw - 0.5) / self.temperature;
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let raw = pruner.manifold_score(depth, candidates[i], parent_tokens);
                    if raw > 0.5 { 1.0 } else { 0.0 }
                };
                scores[i] *= score;
            }
        }
    }
}

impl ConstraintPruner for HyperplanePruner<'_> {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        for pruner in &self.pruners {
            if let Some((_normal, _threshold)) = pruner.constraint_vector(depth, parent_tokens) {
                // Half-space check: use manifold_score > 0.5
                if pruner.manifold_score(depth, token_idx, parent_tokens) <= 0.5 {
                    return false;
                }
            } else {
                // Fall back to boolean check
                if !pruner.is_valid(depth, token_idx, parent_tokens) {
                    return false;
                }
            }
        }
        true
    }

    fn manifold_score(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if self.pruners.is_empty() {
            return 1.0;
        }
        let mut product = 1.0f32;
        for pruner in &self.pruners {
            let score = match pruner.constraint_vector(depth, parent_tokens) {
                Some(_) => {
                    // Sigmoid-softened score
                    let raw = pruner.manifold_score(depth, token_idx, parent_tokens);
                    let x = (raw - 0.5) / self.temperature;
                    1.0 / (1.0 + (-x).exp())
                }
                None => {
                    // Binary: 1.0 or 0.0
                    let raw = pruner.manifold_score(depth, token_idx, parent_tokens);
                    match raw > 0.5 {
                        true => 1.0,
                        false => 0.0,
                    }
                }
            };
            product *= score;
            // Early exit: if any constraint is zero, total is zero
            if product <= 0.0 {
                return 0.0;
            }
        }
        product
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimplePruner {
        threshold: usize,
    }
    impl ConstraintPruner for SimplePruner {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            token_idx < self.threshold
        }
    }

    #[test]
    fn single_pruner_matches_constraint() {
        let inner = SimplePruner { threshold: 5 };
        let hyper = HyperplanePruner::new(vec![&inner]);
        assert!(hyper.is_valid(0, 3, &[]));
        assert!(!hyper.is_valid(0, 7, &[]));
    }

    #[test]
    fn two_pruners_intersection_stricter() {
        let p1 = SimplePruner { threshold: 5 };
        let p2 = SimplePruner { threshold: 3 };
        let hyper = HyperplanePruner::new(vec![&p1, &p2]);
        // Token 2: valid for both (2 < 3, 2 < 5)
        assert!(hyper.is_valid(0, 2, &[]));
        // Token 4: valid for p1 (4 < 5), invalid for p2 (4 >= 3)
        assert!(!hyper.is_valid(0, 4, &[]));
    }

    #[test]
    fn empty_pruners_accepts_all() {
        let hyper = HyperplanePruner::new(vec![]);
        assert!(hyper.is_valid(0, 999, &[]));
        assert_eq!(hyper.manifold_score(0, 999, &[]), 1.0);
    }

    #[test]
    fn manifold_score_product() {
        let p1 = SimplePruner { threshold: 5 };
        let hyper = HyperplanePruner::new(vec![&p1]);
        // Valid token -> score 1.0
        assert!((hyper.manifold_score(0, 3, &[]) - 1.0).abs() < 1e-5);
        // Invalid token -> score 0.0
        assert!((hyper.manifold_score(0, 7, &[]) - 0.0).abs() < 1e-5);
    }

    #[cfg(feature = "manifold_pruner")]
    #[test]
    fn batch_manifold_score_matches_individual() {
        let p1 = SimplePruner { threshold: 5 };
        let p2 = SimplePruner { threshold: 8 };
        let hyper = HyperplanePruner::new(vec![&p1, &p2]).with_temperature(0.5);

        let candidates: Vec<usize> = (0..12).collect();
        let mut batch_scores = vec![0.0f32; 12];
        hyper.batch_manifold_score(0, &candidates, &[], &mut batch_scores);

        for (i, &token) in candidates.iter().enumerate() {
            let individual = hyper.manifold_score(0, token, &[]);
            let diff = (batch_scores[i] - individual).abs();
            assert!(
                diff < 1e-5,
                "token {}: batch={}, individual={}, diff={}",
                token,
                batch_scores[i],
                individual,
                diff
            );
        }
    }
}
