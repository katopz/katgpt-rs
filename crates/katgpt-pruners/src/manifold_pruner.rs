//! ManifoldPruner — Soft sigmoid validity scoring (Plan 234).
//!
//! Converts any ConstraintPruner into a soft scorer via temperature-controlled sigmoid.
//! For pruners with constraint_vector(): distance = |normal · token - threshold|
//! For pruners without: binary fallback

use katgpt_core::traits::ConstraintPruner;

/// ManifoldPruner: soft validity wrapper for any ConstraintPruner.
pub struct ManifoldPruner<P> {
    pub inner: P,
    pub temperature: f32,
}

impl<P: ConstraintPruner> ManifoldPruner<P> {
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            temperature: 1.0,
        }
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
}

impl<P: ConstraintPruner> ConstraintPruner for ManifoldPruner<P> {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        self.inner.is_valid(depth, token_idx, parent_tokens)
    }

    fn manifold_score(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        match self.inner.constraint_vector(depth, parent_tokens) {
            Some(_) => {
                // Has geometric constraint: use sigmoid softened score
                let raw = self.inner.manifold_score(depth, token_idx, parent_tokens);
                let x = (raw - 0.5) / self.temperature;
                1.0 / (1.0 + (-x).exp())
            }
            None => {
                // Binary fallback: sigmoid around boundary
                let valid = self.inner.is_valid(depth, token_idx, parent_tokens);
                match valid {
                    true => 1.0 / (1.0 + (-1.0 / self.temperature).exp()),
                    false => 1.0 / (1.0 + (1.0 / self.temperature).exp()),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct BinaryPruner {
        threshold: usize,
    }
    impl ConstraintPruner for BinaryPruner {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            token_idx < self.threshold
        }
    }

    #[test]
    fn high_temperature_nearly_uniform() {
        let inner = BinaryPruner { threshold: 5 };
        let soft = ManifoldPruner::new(inner).with_temperature(100.0);
        let valid_score = soft.manifold_score(0, 3, &[]);
        let invalid_score = soft.manifold_score(0, 7, &[]);
        // High temp -> both scores close to 0.5
        assert!(
            (valid_score - 0.5).abs() < 0.1,
            "valid_score {} should be ~0.5",
            valid_score
        );
        assert!(
            (invalid_score - 0.5).abs() < 0.1,
            "invalid_score {} should be ~0.5",
            invalid_score
        );
    }

    #[test]
    fn low_temperature_near_binary() {
        let inner = BinaryPruner { threshold: 5 };
        let soft = ManifoldPruner::new(inner).with_temperature(0.01);
        let valid_score = soft.manifold_score(0, 3, &[]);
        let invalid_score = soft.manifold_score(0, 7, &[]);
        // Low temp -> valid ~1.0, invalid ~0.0
        assert!(
            valid_score > 0.99,
            "valid_score {} should be ~1.0",
            valid_score
        );
        assert!(
            invalid_score < 0.01,
            "invalid_score {} should be ~0.0",
            invalid_score
        );
    }

    #[test]
    fn is_valid_passes_through() {
        let inner = BinaryPruner { threshold: 5 };
        let soft = ManifoldPruner::new(inner);
        assert!(soft.is_valid(0, 3, &[]));
        assert!(!soft.is_valid(0, 7, &[]));
    }
}
