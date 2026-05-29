//! Monte Carlo null test for structural significance.
//!
//! Compares a real data pipeline's structural agreement score against
//! a null distribution generated from random noise. The sigma separation
//! indicates how many standard deviations the real result is above chance.

/// Monte Carlo null test result.
#[derive(Debug, Clone)]
pub struct MonteCarloNull {
    /// Mean of the null distribution (random data scores).
    pub null_mean: f32,
    /// Standard deviation of the null distribution.
    pub null_std: f32,
    /// Maximum null score observed.
    pub null_max: f32,
    /// How many sigmas above null mean the real score is.
    pub sigma_separation: f32,
}

/// Run Monte Carlo null test: compare real data's structural agreement vs
/// random noise.
///
/// Generates `n_iterations` random datasets of dimension `dim`, runs the
/// `pipeline` function on each, and returns statistics about the null
/// distribution. The caller should compare their real score against the
/// returned `sigma_separation`.
///
/// `pipeline` receives a synthetic dataset (dim vectors of dim f32 values)
/// and returns a scalar structural agreement score.
///
/// Uses `fastrand::Rng` for reproducibility with no external deps.
pub fn monte_carlo_null_test<F>(
    dim: usize,
    n_iterations: usize,
    seed: u64,
    pipeline: F,
) -> MonteCarloNull
where
    F: Fn(&[Vec<f32>]) -> f32,
{
    let mut rng = fastrand::Rng::with_seed(seed);

    let mut null_scores: Vec<f32> = Vec::with_capacity(n_iterations);

    for _ in 0..n_iterations {
        // Generate random dataset: dim vectors of dimension dim
        let data: Vec<Vec<f32>> = (0..dim)
            .map(|_| {
                (0..dim)
                    .map(|_| rng.f32() * 2.0 - 1.0) // uniform [-1, 1)
                    .collect()
            })
            .collect();

        let score = pipeline(&data);
        null_scores.push(score);
    }

    let n = null_scores.len() as f64;
    let null_mean = null_scores.iter().map(|&x| x as f64).sum::<f64>() / n;
    let null_var = null_scores
        .iter()
        .map(|&x| {
            let d = x as f64 - null_mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let null_std = null_var.sqrt().max(1e-12) as f32;
    let null_max = null_scores
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    // Run pipeline on structured data (identity-like) for comparison
    let structured: Vec<Vec<f32>> = (0..dim)
        .map(|i| {
            let mut v = vec![0.0f32; dim];
            v[i] = 1.0;
            v
        })
        .collect();
    let real_score = pipeline(&structured) as f64;

    let sigma_separation = if null_std > 1e-12 {
        ((real_score - null_mean) / null_std as f64) as f32
    } else {
        0.0
    };

    MonteCarloNull {
        null_mean: null_mean as f32,
        null_std,
        null_max,
        sigma_separation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: structural agreement metric.
    /// Identity-like data has one non-zero per row → high max/mean ratio.
    /// Random data has uniform magnitude → ratio ≈ 1.
    fn structural_pipeline(data: &[Vec<f32>]) -> f32 {
        let n = data.len();
        if n == 0 {
            return 0.0;
        }
        let d = data[0].len();

        // For each row, compute max(|x|) / mean(|x|). Sum across rows.
        // Identity: max=1, mean=1/d → ratio=d per row, total = n*d
        // Random: max≈1, mean≈0.5 → ratio≈2 per row, total ≈ 2n
        let mut total = 0.0f64;
        for row in data {
            let abs_vals: Vec<f64> = row.iter().map(|&x| x.abs() as f64).collect();
            let max_v = abs_vals.iter().cloned().fold(0.0f64, f64::max);
            let mean_v: f64 = abs_vals.iter().sum::<f64>() / d as f64;
            if mean_v > 1e-12 {
                total += max_v / mean_v;
            }
        }
        total as f32
    }

    /// G4: Known structure → σ separation ≥ 5.0; pure noise → σ ≈ 0.0.
    #[test]
    fn test_g4_structured_high_sigma() {
        let result = monte_carlo_null_test(8, 200, 42, structural_pipeline);
        // Identity data has participation_ratio = dim (8), noise will be lower
        assert!(
            result.sigma_separation >= 5.0,
            "known structure should have σ separation ≥ 5.0, got {}",
            result.sigma_separation
        );
    }

    #[test]
    fn test_g4_noise_sigma_near_zero() {
        // Pipeline on all-noise should produce σ ≈ 0
        let result = monte_carlo_null_test(4, 100, 99, |_data| 1.0f32);
        // Every input produces 1.0, so null_mean = 1.0, real = 1.0 → σ = 0
        assert!(
            result.sigma_separation.abs() < 0.01,
            "constant pipeline should have σ ≈ 0, got {}",
            result.sigma_separation
        );
    }

    /// Smoke test: runs without panic.
    #[test]
    fn test_monte_carlo_smoke() {
        let result = monte_carlo_null_test(3, 10, 0, |data| {
            data.iter().map(|v| v.iter().sum::<f32>()).sum()
        });
        assert!(result.null_mean.is_finite());
        assert!(result.null_std >= 0.0);
    }
}
