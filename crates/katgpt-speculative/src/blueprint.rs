//! Blueprint pre-pass: cheap argmax generation to guide DDTree search.
//! Inspired by LEAP's informal-formal planning pipeline (arXiv 2606.03303).
//!
//! The blueprint is a greedy argmax plan: O(depth * vocab) with no tree search.
//! It provides a hint to the expensive DDTree about which tokens are promising.
//! Tokens compatible with the blueprint get a bonus in the heap.

/// Blueprint pre-pass: generates cheap argmax plan from marginals.
pub struct BlueprintPass;

impl BlueprintPass {
    /// Generate a cheap argmax plan from marginals.
    /// O(depth * vocab) — no tree search, just greedy argmax at each depth.
    /// Returns `Vec<usize>` where `result[d] = argmax of marginals[d]`.
    pub fn generate(marginals: &[&[f32]]) -> Vec<usize> {
        marginals
            .iter()
            .map(|m| {
                m.iter()
                    .enumerate()
                    // float_order: a NaN marginal must never win the argmax.
                    .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
                    .map_or(0, |(idx, _)| idx)
            })
            .collect()
    }

    /// Score a token against the blueprint plan.
    /// Returns bonus for compatibility, 0.0 for deviation.
    /// If `blueprint[d] == token_idx` → returns bonus (default 0.1).
    /// Otherwise → 0.0.
    pub fn compatibility(depth: usize, token_idx: usize, blueprint: &[usize], bonus: f32) -> f32 {
        match blueprint.get(depth) {
            Some(&bp_token) if bp_token == token_idx => bonus,
            _ => 0.0,
        }
    }

    /// Batch compatibility: apply blueprint bonus to a score array in-place.
    /// For each token at `depth`: if token matches blueprint, add bonus.
    pub fn apply_bonus(depth: usize, scores: &mut [f32], blueprint: &[usize], bonus: f32) {
        match blueprint.get(depth) {
            Some(&bp_token) if bp_token < scores.len() => scores[bp_token] += bonus,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_argmax_from_simple_marginals() {
        // Marginals: depth 0 → token 2 is max, depth 1 → token 1 is max
        let m0: &[f32] = &[0.1, 0.3, 0.6, 0.0];
        let m1: &[f32] = &[0.2, 0.5, 0.1, 0.2];
        let blueprint = BlueprintPass::generate(&[m0, m1]);
        assert_eq!(blueprint, vec![2, 1]);
    }

    #[test]
    fn compatibility_match_and_mismatch() {
        let blueprint = vec![3, 1, 0];
        let bonus = 0.15;

        // Match at depth 0
        assert_eq!(BlueprintPass::compatibility(0, 3, &blueprint, bonus), 0.15);
        // Mismatch at depth 0
        assert_eq!(BlueprintPass::compatibility(0, 1, &blueprint, bonus), 0.0);
        // Match at depth 1
        assert_eq!(BlueprintPass::compatibility(1, 1, &blueprint, bonus), 0.15);
        // Depth out of range
        assert_eq!(BlueprintPass::compatibility(5, 0, &blueprint, bonus), 0.0);
    }

    #[test]
    fn apply_bonus_in_place() {
        let blueprint = vec![2, 0];
        let mut scores = [1.0, 2.0, 3.0, 4.0];

        BlueprintPass::apply_bonus(0, &mut scores, &blueprint, 0.5);
        assert_eq!(scores, [1.0, 2.0, 3.5, 4.0]); // token 2 gets bonus

        BlueprintPass::apply_bonus(1, &mut scores, &blueprint, 0.5);
        assert_eq!(scores, [1.5, 2.0, 3.5, 4.0]); // token 0 gets bonus
    }

    #[test]
    fn apply_bonus_out_of_bounds() {
        let blueprint = vec![10]; // token 10 doesn't exist in scores
        let mut scores = [1.0, 2.0, 3.0];
        BlueprintPass::apply_bonus(0, &mut scores, &blueprint, 0.5);
        assert_eq!(scores, [1.0, 2.0, 3.0]); // unchanged, no panic

        let mut scores2 = [1.0, 2.0];
        BlueprintPass::apply_bonus(5, &mut scores2, &blueprint, 0.5); // depth out of range
        assert_eq!(scores2, [1.0, 2.0]); // unchanged, no panic
    }

    #[test]
    fn empty_marginals() {
        let blueprint = BlueprintPass::generate(&[]);
        assert!(blueprint.is_empty());

        // Single empty slice → argmax returns index 0 (default)
        let empty_slice: &[f32] = &[];
        let blueprint = BlueprintPass::generate(&[empty_slice]);
        assert_eq!(blueprint, vec![0]);
    }

    #[test]
    fn uniform_marginals_any_token_valid() {
        // All marginals equal → any index is a valid argmax
        let m: &[f32] = &[0.25, 0.25, 0.25, 0.25];
        let blueprint = BlueprintPass::generate(&[m]);
        // Should pick some index (implementation picks first max encountered)
        assert!(blueprint[0] < 4);
    }

    #[test]
    fn generate_multi_depth() {
        let m0: &[f32] = &[0.0, 1.0]; // argmax = 1
        let m1: &[f32] = &[0.9, 0.1]; // argmax = 0
        let m2: &[f32] = &[0.3, 0.7]; // argmax = 1
        let blueprint = BlueprintPass::generate(&[m0, m1, m2]);
        assert_eq!(blueprint, vec![1, 0, 1]);
    }
}
