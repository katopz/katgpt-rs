//! Rosetta Pruners — Universal Cross-Domain Meta-Pruner (Plan 201).
//!
//! Mines universal constraint concepts from multiple `ConstraintPruner`s:
//! - Universal: ALL pruners agree → fast path (O(1) concept map lookup)
//! - Contested: pruners disagree → slow path (majority vote)
//!
//! Uses papaya lock-free HashMap for the concept map.
//!
//! Feature flag: `rosetta_pruner`

use crate::speculative::types::ConstraintPruner;
use std::sync::Arc;

/// A universal constraint concept mined from cross-pruner agreement.
#[derive(Debug, Clone)]
pub struct ConstraintConcept {
    /// Unique concept ID.
    pub id: usize,
    /// Tree depth this concept applies to.
    pub depth: usize,
    /// Token indices that form this concept.
    pub tokens: Vec<usize>,
    /// Fraction of pruners that agree on validity [0.0, 1.0].
    pub agreement_ratio: f32,
}

// ── papaya-based implementation ─────────────────────────────────

#[cfg(feature = "papaya")]
pub struct RosettaPruner<P: ConstraintPruner + ?Sized> {
    /// Underlying pruners
    pruners: Vec<Arc<P>>,
    /// Pre-computed universal concept map: (depth, token_idx) → agreement ratio
    concept_map: papaya::HashMap<(usize, usize), f32>,
    /// Universal concepts: positions where agreement > threshold
    universal_concepts: Vec<ConstraintConcept>,
    /// Number of pruners
    n_pruners: usize,
    /// Agreement threshold for "universal" (default: 0.9)
    threshold: f32,
    /// Next concept ID for mining
    next_concept_id: usize,
}

#[cfg(feature = "papaya")]
impl<P: ConstraintPruner + ?Sized> RosettaPruner<P> {
    /// Create a new RosettaPruner from a list of pruners.
    pub fn new(pruners: Vec<Arc<P>>) -> Self {
        let n = pruners.len();
        Self {
            pruners,
            concept_map: papaya::HashMap::new(),
            universal_concepts: Vec::new(),
            n_pruners: n,
            threshold: 0.9,
            next_concept_id: 0,
        }
    }

    /// Set the agreement threshold for "universal" concepts.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.5, 1.0);
        self
    }

    /// Mine universal concepts by probing all pruners across depths and tokens.
    ///
    /// For each (depth, token) pair, queries all pruners and computes agreement.
    /// Pairs above `threshold` are stored in the concept map for O(1) lookup.
    ///
    /// # Returns
    /// Number of universal concepts discovered.
    pub fn mine_concepts(
        &mut self,
        max_depth: usize,
        tokens: &[usize],
        parent_tokens: &[usize],
    ) -> usize {
        let mut discovered = 0;
        let pin = self.concept_map.pin();

        for depth in 0..max_depth {
            for &token_idx in tokens {
                let valid_count = self
                    .pruners
                    .iter()
                    .filter(|p| p.is_valid(depth, token_idx, parent_tokens))
                    .count();

                let agreement = valid_count as f32 / self.n_pruners as f32;

                pin.insert((depth, token_idx), agreement);

                if agreement >= self.threshold {
                    let concept = ConstraintConcept {
                        id: self.next_concept_id,
                        depth,
                        tokens: vec![token_idx],
                        agreement_ratio: agreement,
                    };
                    self.next_concept_id += 1;
                    self.universal_concepts.push(concept);
                    discovered += 1;
                }
            }
        }

        discovered
    }

    /// Get the number of universal concepts discovered.
    pub fn universal_concept_count(&self) -> usize {
        self.universal_concepts.len()
    }

    /// Get the number of pruners.
    pub fn pruner_count(&self) -> usize {
        self.n_pruners
    }

    /// Clear all mined concepts.
    pub fn clear_concepts(&mut self) {
        self.concept_map.pin().clear();
        self.universal_concepts.clear();
        self.next_concept_id = 0;
    }
}

#[cfg(feature = "papaya")]
impl<P: ConstraintPruner + ?Sized> ConstraintPruner for RosettaPruner<P> {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        // Fast path: check pre-computed concept map
        let pin = self.concept_map.pin();
        if let Some(&agreement) = pin.get(&(depth, token_idx)) {
            if agreement >= self.threshold {
                return true; // Universal concept — all pruners agree
            }
            if agreement <= (1.0 - self.threshold) {
                return false; // Universal rejection — all pruners reject
            }
        }

        // Slow path: query all pruners, majority vote
        let valid_count = self
            .pruners
            .iter()
            .filter(|p| p.is_valid(depth, token_idx, parent_tokens))
            .count();
        valid_count > self.n_pruners / 2
    }

    fn batch_is_valid(
        &self,
        depth: usize,
        candidates: &[usize],
        parent_tokens: &[usize],
        results: &mut [bool],
    ) {
        let pin = self.concept_map.pin();
        let len = candidates.len().min(results.len());

        // Check concept map for all candidates first (batch fast path)
        let mut need_slow_path = Vec::new();
        for i in 0..len {
            if let Some(&agreement) = pin.get(&(depth, candidates[i])) {
                if agreement >= self.threshold {
                    results[i] = true;
                    continue;
                }
                if agreement <= (1.0 - self.threshold) {
                    results[i] = false;
                    continue;
                }
            }
            need_slow_path.push(i);
        }

        // Slow path for contested candidates
        for &i in &need_slow_path {
            let valid_count = self
                .pruners
                .iter()
                .filter(|p| p.is_valid(depth, candidates[i], parent_tokens))
                .count();
            results[i] = valid_count > self.n_pruners / 2;
        }
    }
}

// ── Without papaya: simple Vec-based fallback ────────────────────

#[cfg(not(feature = "papaya"))]
pub struct RosettaPruner<P: ConstraintPruner + ?Sized> {
    pruners: Vec<Arc<P>>,
    /// Fallback: Vec-based concept storage
    concept_entries: Vec<((usize, usize), f32)>,
    universal_concepts: Vec<ConstraintConcept>,
    n_pruners: usize,
    threshold: f32,
    next_concept_id: usize,
}

#[cfg(not(feature = "papaya"))]
impl<P: ConstraintPruner + ?Sized> RosettaPruner<P> {
    pub fn new(pruners: Vec<Arc<P>>) -> Self {
        let n = pruners.len();
        Self {
            pruners,
            concept_entries: Vec::new(),
            universal_concepts: Vec::new(),
            n_pruners: n,
            threshold: 0.9,
            next_concept_id: 0,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold.clamp(0.5, 1.0);
        self
    }

    fn lookup_concept(&self, depth: usize, token_idx: usize) -> Option<f32> {
        // Linear scan — fine for small concept maps, use papaya for production
        self.concept_entries
            .iter()
            .find(|((d, t), _)| *d == depth && *t == token_idx)
            .map(|(_, agreement)| *agreement)
    }

    pub fn mine_concepts(
        &mut self,
        max_depth: usize,
        tokens: &[usize],
        parent_tokens: &[usize],
    ) -> usize {
        let mut discovered = 0;

        for depth in 0..max_depth {
            for &token_idx in tokens {
                let valid_count = self
                    .pruners
                    .iter()
                    .filter(|p| p.is_valid(depth, token_idx, parent_tokens))
                    .count();

                let agreement = valid_count as f32 / self.n_pruners as f32;
                self.concept_entries.push(((depth, token_idx), agreement));

                if agreement >= self.threshold {
                    let concept = ConstraintConcept {
                        id: self.next_concept_id,
                        depth,
                        tokens: vec![token_idx],
                        agreement_ratio: agreement,
                    };
                    self.next_concept_id += 1;
                    self.universal_concepts.push(concept);
                    discovered += 1;
                }
            }
        }

        discovered
    }

    pub fn universal_concept_count(&self) -> usize {
        self.universal_concepts.len()
    }

    pub fn pruner_count(&self) -> usize {
        self.n_pruners
    }

    pub fn clear_concepts(&mut self) {
        self.concept_entries.clear();
        self.universal_concepts.clear();
        self.next_concept_id = 0;
    }
}

#[cfg(not(feature = "papaya"))]
impl<P: ConstraintPruner + ?Sized> ConstraintPruner for RosettaPruner<P> {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        if let Some(agreement) = self.lookup_concept(depth, token_idx) {
            if agreement >= self.threshold {
                return true;
            }
            if agreement <= (1.0 - self.threshold) {
                return false;
            }
        }

        let valid_count = self
            .pruners
            .iter()
            .filter(|p| p.is_valid(depth, token_idx, parent_tokens))
            .count();
        valid_count > self.n_pruners / 2
    }
}

// ── ScreeningPruner impl (Plan 201) ──────────────────────────────
// Agreement-weighted relevance: universal concepts get relevance 1.0,
// contested get weighted by majority agreement ratio, rejected get 0.0.

#[cfg(feature = "papaya")]
impl<P: ConstraintPruner + ?Sized> crate::speculative::types::ScreeningPruner for RosettaPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let pin = self.concept_map.pin();
        if let Some(&agreement) = pin.get(&(depth, token_idx)) {
            if agreement >= self.threshold {
                return 1.0; // Universal concept
            }
            if agreement <= (1.0 - self.threshold) {
                return 0.0; // Universal rejection
            }
            // Contested: return agreement as soft relevance
            return agreement;
        }

        // Not in concept map: compute on-the-fly
        let valid_count = self
            .pruners
            .iter()
            .filter(|p| p.is_valid(depth, token_idx, parent_tokens))
            .count();
        valid_count as f32 / self.n_pruners as f32
    }
}

#[cfg(not(feature = "papaya"))]
impl<P: ConstraintPruner + ?Sized> crate::speculative::types::ScreeningPruner for RosettaPruner<P> {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        if let Some(agreement) = self.lookup_concept(depth, token_idx) {
            if agreement >= self.threshold {
                return 1.0;
            }
            if agreement <= (1.0 - self.threshold) {
                return 0.0;
            }
            return agreement;
        }

        let valid_count = self
            .pruners
            .iter()
            .filter(|p| p.is_valid(depth, token_idx, parent_tokens))
            .count();
        valid_count as f32 / self.n_pruners as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pruner that accepts tokens where token_idx % modulus == 0
    struct ModPruner {
        modulus: usize,
    }

    impl ConstraintPruner for ModPruner {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            token_idx.is_multiple_of(self.modulus)
        }
    }

    /// Pruner that accepts all tokens.
    struct AcceptAllPruner;

    impl ConstraintPruner for AcceptAllPruner {
        fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
            true
        }
    }

    /// Pruner that rejects all tokens.
    struct RejectAllPruner;

    impl ConstraintPruner for RejectAllPruner {
        fn is_valid(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> bool {
            false
        }
    }

    #[test]
    fn test_rosetta_universal_accept() {
        // All pruners accept → universal concept
        let pruners: Vec<Arc<dyn ConstraintPruner>> =
            vec![Arc::new(AcceptAllPruner), Arc::new(AcceptAllPruner)];
        let mut rosetta = RosettaPruner::new(pruners);
        let discovered = rosetta.mine_concepts(3, &[0, 1, 2], &[]);
        // All (depth, token) pairs are universal
        assert_eq!(discovered, 9); // 3 depths × 3 tokens
        assert!(rosetta.is_valid(0, 0, &[]));
        assert!(rosetta.is_valid(1, 2, &[]));
    }

    #[test]
    fn test_rosetta_universal_reject() {
        // All pruners reject → universal rejection
        let pruners: Vec<Arc<dyn ConstraintPruner>> =
            vec![Arc::new(RejectAllPruner), Arc::new(RejectAllPruner)];
        let mut rosetta = RosettaPruner::new(pruners);
        rosetta.mine_concepts(2, &[0, 1], &[]);
        assert!(!rosetta.is_valid(0, 0, &[]));
        assert!(!rosetta.is_valid(1, 1, &[]));
    }

    #[test]
    fn test_rosetta_majority_vote() {
        // 2 accept, 1 reject → majority accepts
        let pruners: Vec<Arc<dyn ConstraintPruner>> = vec![
            Arc::new(AcceptAllPruner),
            Arc::new(AcceptAllPruner),
            Arc::new(RejectAllPruner),
        ];
        let mut rosetta = RosettaPruner::new(pruners);
        rosetta.mine_concepts(1, &[0], &[]);
        // Agreement = 2/3 ≈ 0.667 < 0.9 threshold, so not universal
        // But majority (2/3 > 1) should still accept
        assert!(rosetta.is_valid(0, 0, &[]));
    }

    #[test]
    fn test_rosetta_mod_pruner_intersection() {
        // Mod2 ∩ Mod3: universal for tokens that are multiples of both (0, 6, 12, ...)
        let pruners: Vec<Arc<dyn ConstraintPruner>> = vec![
            Arc::new(ModPruner { modulus: 2 }),
            Arc::new(ModPruner { modulus: 3 }),
        ];
        let mut rosetta = RosettaPruner::new(pruners);
        let discovered = rosetta.mine_concepts(1, &[0, 1, 2, 3, 4, 5, 6], &[]);
        // Only token 0 and 6 are accepted by BOTH pruners → universal concepts
        assert_eq!(discovered, 2);
        assert!(rosetta.is_valid(0, 0, &[])); // universal
        assert!(rosetta.is_valid(0, 6, &[])); // universal
        // Token 2: accepted by Mod2, rejected by Mod3 → contested
        // 1/2 valid = 0.5, not majority
        assert!(!rosetta.is_valid(0, 2, &[]));
    }

    #[test]
    fn test_rosetta_threshold_custom() {
        let pruners: Vec<Arc<dyn ConstraintPruner>> =
            vec![Arc::new(AcceptAllPruner), Arc::new(AcceptAllPruner)];
        let _rosetta = RosettaPruner::new(pruners).with_threshold(0.5);
    }

    #[test]
    fn test_rosetta_clear_concepts() {
        let pruners: Vec<Arc<dyn ConstraintPruner>> = vec![Arc::new(AcceptAllPruner)];
        let mut rosetta = RosettaPruner::new(pruners);
        rosetta.mine_concepts(5, &[0, 1, 2], &[]);
        assert!(rosetta.universal_concept_count() > 0);
        rosetta.clear_concepts();
        assert_eq!(rosetta.universal_concept_count(), 0);
    }

    #[test]
    fn test_rosetta_batch_is_valid() {
        let pruners: Vec<Arc<dyn ConstraintPruner>> =
            vec![Arc::new(AcceptAllPruner), Arc::new(AcceptAllPruner)];
        let mut rosetta = RosettaPruner::new(pruners);
        rosetta.mine_concepts(1, &[0, 1, 2, 3], &[]);
        let mut results = [false; 4];
        rosetta.batch_is_valid(0, &[0, 1, 2, 3], &[], &mut results);
        assert!(results[0]);
        assert!(results[1]);
        assert!(results[2]);
        assert!(results[3]);
    }

    #[test]
    fn test_rosetta_screening_universal_accept() {
        use crate::speculative::types::ScreeningPruner;

        let pruners: Vec<Arc<dyn ConstraintPruner>> =
            vec![Arc::new(AcceptAllPruner), Arc::new(AcceptAllPruner)];
        let mut rosetta = RosettaPruner::new(pruners);
        rosetta.mine_concepts(2, &[0, 1, 2], &[]);
        // Universal acceptance → relevance 1.0
        assert!((rosetta.relevance(0, 0, &[]) - 1.0).abs() < 0.01);
        assert!((rosetta.relevance(1, 2, &[]) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rosetta_screening_universal_reject() {
        use crate::speculative::types::ScreeningPruner;

        let pruners: Vec<Arc<dyn ConstraintPruner>> =
            vec![Arc::new(RejectAllPruner), Arc::new(RejectAllPruner)];
        let mut rosetta = RosettaPruner::new(pruners);
        rosetta.mine_concepts(1, &[0, 1], &[]);
        // Universal rejection → relevance 0.0
        assert!((rosetta.relevance(0, 0, &[]) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_rosetta_screening_contested() {
        use crate::speculative::types::ScreeningPruner;

        // 2 accept, 1 reject → agreement = 0.667
        let pruners: Vec<Arc<dyn ConstraintPruner>> = vec![
            Arc::new(AcceptAllPruner),
            Arc::new(AcceptAllPruner),
            Arc::new(RejectAllPruner),
        ];
        let mut rosetta = RosettaPruner::new(pruners);
        rosetta.mine_concepts(1, &[0], &[]);
        // Contested: relevance should be agreement ratio ≈ 0.667
        let rel = rosetta.relevance(0, 0, &[]);
        assert!(
            (rel - 0.667).abs() < 0.05,
            "contested relevance should be ~0.667, got {rel}"
        );
    }
}
