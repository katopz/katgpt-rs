//! BAKE Precision-Gated Bayesian Embedding Update (Plan 236).
//!
//! Per-dimension precision tracking for KgEmbedding.
//! High precision → anchor (resist change). Low precision → explore (absorb eagerly).
//! O(8) arithmetic per update, zero-alloc, SIMD-friendly.

/// Uninformative prior precision for new entities.
pub const UNINFORMATIVE_PRECISION: f32 = 0.1;

/// Default observation precision.
pub const DEFAULT_OBS_PRECISION: f32 = 1.0;

/// BAKE eq 2: Bayesian precision update.
/// λ_new = λ_old + λ_obs  (precision grows monotonically)
#[inline]
pub fn bake_update_precision(lambda_old: &[f32; 8], lambda_obs: f32) -> [f32; 8] {
    let mut lambda_new = *lambda_old;
    for d in 0..8 {
        lambda_new[d] += lambda_obs;
    }
    lambda_new
}

/// BAKE eq 3: Precision-weighted mean update.
/// μ_new = (λ_old ⊙ μ_old + λ_obs ⊙ obs) / λ_new
/// SIMD-friendly: operates on [f32; 8] which auto-vectorizes.
#[inline]
pub fn bake_update_mean(
    mu_old: &[f32; 8],
    lambda_old: &[f32; 8],
    observation: &[f32; 8],
    lambda_obs: f32,
) -> [f32; 8] {
    let lambda_new = bake_update_precision(lambda_old, lambda_obs);
    let mut mu_new = [0.0f32; 8];
    for d in 0..8 {
        mu_new[d] = (lambda_old[d] * mu_old[d] + lambda_obs * observation[d]) / lambda_new[d];
    }
    mu_new
}

/// Combined BAKE update: returns (new_mean, new_precision).
#[inline]
pub fn bake_update(
    mu_old: &[f32; 8],
    lambda_old: &[f32; 8],
    observation: &[f32; 8],
    lambda_obs: f32,
) -> ([f32; 8], [f32; 8]) {
    let lambda_new = bake_update_precision(lambda_old, lambda_obs);
    let mu_new = bake_update_mean(mu_old, lambda_old, observation, lambda_obs);
    (mu_new, lambda_new)
}

/// BAKE eq 4: Precision-weighted regularization penalty.
/// β · √(λ ⊙ (μ_current - μ_old)²)
/// Returns penalty — high when current deviates from high-precision prior.
#[inline]
pub fn bake_regularize(
    mu_old: &[f32; 8],
    lambda: &[f32; 8],
    mu_current: &[f32; 8],
    beta: f32,
) -> f32 {
    let mut penalty = 0.0f32;
    for d in 0..8 {
        let diff = mu_current[d] - mu_old[d];
        penalty += (lambda[d] * diff * diff).sqrt();
    }
    penalty * beta
}

/// Compute effective confidence from precision vector.
/// confidence = sigmoid(mean(precision) - 1.0)
/// Higher average precision → higher confidence.
#[inline]
pub fn precision_to_confidence(lambda: &[f32; 8]) -> f32 {
    let mean_lambda: f32 = lambda.iter().sum::<f32>() / 8.0;
    1.0 / (1.0 + (-(mean_lambda - 1.0)).exp()) // sigmoid
}

/// Exploration priority for a dimension (0..7).
/// Lower precision → higher priority for exploration.
/// Returns value in [0, 1] where 1 = highest priority.
#[inline]
pub fn exploration_priority(lambda: &[f32; 8], dimension: usize) -> f32 {
    debug_assert!(dimension < 8, "dimension must be 0..7");
    let max_lambda = lambda.iter().cloned().fold(0.0f32, f32::max);
    if max_lambda < 1e-6 {
        return 1.0;
    }
    1.0 - lambda[dimension] / max_lambda
}

/// Informed prior precision from schema class density.
/// Dense classes (many entities) → higher precision (confident centroid).
/// λ_init = class_count / (1 + class_count) ∈ [0, 1).
#[inline]
pub fn informed_prior_precision(class_count: usize) -> [f32; 8] {
    let p = (class_count as f32) / (1.0 + class_count as f32);
    [p; 8]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_monotonicity() {
        let mut lambda = [0.1f32; 8];
        for _ in 0..10 {
            let old = lambda;
            lambda = bake_update_precision(&old, 1.0);
            for d in 0..8 {
                assert!(
                    lambda[d] >= old[d],
                    "precision should be monotonically non-decreasing"
                );
            }
        }
    }

    #[test]
    fn test_uninformative_prior_absorbs() {
        let mu_old = [0.0f32; 8];
        let lambda_old = [0.01f32; 8];
        let obs = [1.0f32; 8];
        let (mu_new, _) = bake_update(&mu_old, &lambda_old, &obs, 10.0);
        for d in 0..8 {
            assert!(
                (mu_new[d] - 1.0).abs() < 0.01,
                "should absorb observation when precision is low"
            );
        }
    }

    #[test]
    fn test_high_precision_resists() {
        let mu_old = [0.0f32; 8];
        let lambda_old = [100.0f32; 8];
        let obs = [1.0f32; 8];
        let (mu_new, _) = bake_update(&mu_old, &lambda_old, &obs, 1.0);
        for d in 0..8 {
            assert!(
                mu_new[d].abs() < 0.02,
                "should resist change when precision is high, got {}",
                mu_new[d]
            );
        }
    }

    #[test]
    fn test_regularize_zero_when_aligned() {
        let mu = [0.5f32; 8];
        let lambda = [1.0f32; 8];
        let penalty = bake_regularize(&mu, &lambda, &mu, 1.0);
        assert!(penalty.abs() < 1e-6, "penalty should be zero when aligned");
    }

    #[test]
    fn test_regularize_high_when_deviant() {
        let mu_old = [0.0f32; 8];
        let lambda = [10.0f32; 8];
        let mu_current = [1.0f32; 8];
        let penalty = bake_regularize(&mu_old, &lambda, &mu_current, 1.0);
        assert!(
            penalty > 3.0,
            "penalty should be high when deviating from high-precision prior, got {}",
            penalty
        );
    }

    #[test]
    fn test_confidence_increases_with_precision() {
        let low = precision_to_confidence(&[0.1f32; 8]);
        let high = precision_to_confidence(&[10.0f32; 8]);
        assert!(high > low, "higher precision should give higher confidence");
    }

    #[test]
    fn test_exploration_priority_inversely_related() {
        let lambda = [1.0, 5.0, 10.0, 0.5, 2.0, 8.0, 3.0, 0.1];
        let p7 = exploration_priority(&lambda, 7);
        let p2 = exploration_priority(&lambda, 2);
        assert!(
            p7 > p2,
            "low precision dim should have higher exploration priority"
        );
    }

    #[test]
    fn test_informed_prior_dense_class_higher() {
        let sparse = informed_prior_precision(1);
        let dense = informed_prior_precision(100);
        assert!(
            dense[0] > sparse[0],
            "dense classes should have higher initial precision"
        );
    }
}
