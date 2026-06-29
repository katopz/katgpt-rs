/// Retention metric for GOAT proofs: does method X preserve Y% of
/// baseline action diversity?
#[derive(Debug, Clone)]
pub struct RetentionMetric {
    /// Entropy of the baseline (reference) distribution.
    pub baseline_entropy: f32,
    /// Entropy of the post-method distribution.
    pub post_entropy: f32,
    /// exp(-KL(post || baseline)). 1.0 = perfect retention, → 0 as divergence grows.
    pub retention_ratio: f32,
}

impl RetentionMetric {
    /// Compute from two action distributions.
    ///
    /// Uses KL divergence: `retention = exp(-KL(post || baseline))`.
    /// Terms where `post[i] == 0` or `baseline[i] == 0` are skipped.
    pub fn compute(baseline: &[f32], post: &[f32]) -> Self {
        let baseline_entropy = entropy(baseline);
        let post_entropy = entropy(post);

        let kl = kl_divergence(post, baseline);
        let retention_ratio = (-kl).exp();

        Self {
            baseline_entropy,
            post_entropy,
            retention_ratio,
        }
    }
}

/// Shannon entropy H = -Σ p_i * ln(p_i), natural log.
fn entropy(dist: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in dist {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

/// KL(post || baseline) = Σ post[i] * ln(post[i] / baseline[i]).
///
/// Skips terms where either distribution is zero.
fn kl_divergence(post: &[f32], baseline: &[f32]) -> f32 {
    let mut kl = 0.0f32;
    for (&p, &b) in post.iter().zip(baseline.iter()) {
        if p > 0.0 && b > 0.0 {
            kl += p * (p / b).ln();
        }
    }
    kl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_distributions_perfect_retention() {
        let dist = &[0.25f32, 0.25, 0.25, 0.25];
        let m = RetentionMetric::compute(dist, dist);
        assert!(
            (m.retention_ratio - 1.0).abs() < 0.001,
            "expected ~1.0, got {}",
            m.retention_ratio
        );
    }

    #[test]
    fn divergent_distributions_lower_retention() {
        let baseline = &[0.5f32, 0.5];
        let post = &[0.9f32, 0.1];
        let m = RetentionMetric::compute(baseline, post);
        assert!(
            m.retention_ratio < 0.9,
            "expected < 0.9, got {}",
            m.retention_ratio
        );
    }

    #[test]
    fn zeros_handled_gracefully() {
        let baseline = &[0.0f32, 1.0];
        let post = &[0.0f32, 1.0];
        let m = RetentionMetric::compute(baseline, post);
        assert!(
            (m.retention_ratio - 1.0).abs() < 0.001,
            "expected ~1.0, got {}",
            m.retention_ratio
        );
    }
}
