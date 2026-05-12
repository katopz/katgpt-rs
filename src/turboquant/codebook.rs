//! Lloyd-Max scalar quantizer for Beta-distributed coordinates.
//!
//! After random rotation of d-dimensional unit vectors, each coordinate follows:
//! `f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2)`
//!
//! The Lloyd-Max algorithm iterates between:
//!
//! 1. Computing centroids (expected values in each bin)
//! 2. Computing boundaries (midpoints between adjacent centroids)
//!
//! until convergence, yielding the minimum-MSE scalar codebook.

use super::types::TurboQuantCodebook;

/// Compute Lloyd-Max optimal codebook for Beta distribution on [-1, 1].
///
/// Returns a codebook with `2^bits` centroid levels and `2^bits - 1` decision boundaries,
/// optimized for the marginal distribution of a single coordinate of a randomly rotated
/// `dim`-dimensional unit vector.
pub fn compute_codebook(dim: usize, bits: u8) -> TurboQuantCodebook {
    debug_assert!((2..=8).contains(&bits), "bits must be in [2, 8]");
    debug_assert!(dim >= 3, "dim must be >= 3 for Beta distribution");

    let n_levels = 1usize << bits;
    let n_boundaries = n_levels - 1;

    // Initialize boundaries at uniform quantiles of [-1, 1]
    let mut boundaries: Vec<f32> = (0..n_boundaries)
        .map(|i| -1.0 + 2.0 * (i + 1) as f32 / n_levels as f32)
        .collect();

    // Lloyd-Max iteration (max 50 iterations)
    for _ in 0..50 {
        let centroids = compute_centroids(&boundaries, dim);
        let new_boundaries = compute_boundaries_from_centroids(&centroids);
        if converged(&boundaries, &new_boundaries, 1e-6) {
            boundaries = new_boundaries;
            let final_centroids = compute_centroids(&boundaries, dim);
            let mse = compute_mse(&boundaries, &final_centroids, dim);
            return TurboQuantCodebook {
                centroids: final_centroids,
                boundaries,
                mse_per_coord: mse,
                dim,
                bits,
            };
        }
        boundaries = new_boundaries;
    }

    // Did not converge in 50 iterations — return best effort
    let centroids = compute_centroids(&boundaries, dim);
    let mse = compute_mse(&boundaries, &centroids, dim);
    TurboQuantCodebook {
        centroids,
        boundaries,
        mse_per_coord: mse,
        dim,
        bits,
    }
}

/// Beta-like PDF for coordinates of randomly rotated d-dim vectors.
///
/// For a unit vector in R^d rotated by a random orthogonal matrix,
/// each coordinate has marginal density proportional to (1-x²)^((d-3)/2).
fn beta_pdf(x: f32, d: usize) -> f32 {
    if x.abs() >= 1.0 {
        return 0.0;
    }
    let exponent = ((d - 3) as f32) / 2.0;
    let c = gamma_ratio(d);
    c * (1.0 - x * x).powf(exponent)
}

/// Normalizing constant: Γ(d/2) / (√π · Γ((d-1)/2)).
///
/// Uses the Lanczos approximation via `ln_gamma` for numerical stability.
fn gamma_ratio(d: usize) -> f32 {
    let d_f = d as f64;
    let log_ratio =
        ln_gamma(d_f / 2.0) - ln_gamma((d_f - 1.0) / 2.0) - (std::f64::consts::PI / 2.0).ln();
    log_ratio.exp() as f32
}

/// Log of Gamma function using the Lanczos approximation.
///
/// Accurate to ~15 significant digits for x > 0.5.
/// Uses the reflection formula for x < 0.5.
fn ln_gamma(x: f64) -> f64 {
    let g = 7.0;
    let coef = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        // Reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
        return std::f64::consts::PI / (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x);
    }

    let z = x - 1.0;
    let a = coef[0];
    let t = z + g + 0.5;
    let series = coef
        .iter()
        .skip(1)
        .enumerate()
        .fold(a, |acc, (i, c)| acc + c / (z + i as f64 + 1.0));
    (2.0 * std::f64::consts::PI).sqrt().ln() + (t + 0.5).ln() * (z + 0.5) - t + series.ln()
}

/// Compute centroids for given boundaries via numerical integration (trapezoidal rule).
///
/// For each bin [boundary[i-1], boundary[i]], computes:
/// `centroid[i] = ∫ x·f(x) dx / ∫ f(x) dx`
fn compute_centroids(boundaries: &[f32], dim: usize) -> Vec<f32> {
    let n = boundaries.len() + 1;
    let mut centroids = vec![0.0f32; n];
    let n_samples = 1000;

    // Interval edges: -1.0, boundary[0], boundary[1], ..., 1.0
    let mut edges = vec![-1.0f32];
    edges.extend_from_slice(boundaries);
    edges.push(1.0);

    for i in 0..n {
        let lo = edges[i];
        let hi = edges[i + 1];
        let dx = (hi - lo) / n_samples as f32;
        let mut sum_fx = 0.0f64;
        let mut sum_xfx = 0.0f64;
        for j in 0..=n_samples {
            let x = lo + dx * j as f32;
            let pdf = beta_pdf(x, dim);
            let w = if j == 0 || j == n_samples { 0.5 } else { 1.0 };
            sum_fx += pdf as f64 * w;
            sum_xfx += x as f64 * pdf as f64 * w;
        }
        centroids[i] = if sum_fx > 1e-12 {
            (sum_xfx / sum_fx) as f32
        } else {
            (lo + hi) / 2.0
        };
    }
    centroids
}

/// Compute decision boundaries as midpoints between adjacent centroids.
fn compute_boundaries_from_centroids(centroids: &[f32]) -> Vec<f32> {
    (0..centroids.len() - 1)
        .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
        .collect()
}

/// Check convergence of Lloyd-Max iteration.
fn converged(old: &[f32], new: &[f32], tol: f32) -> bool {
    old.iter().zip(new).all(|(a, b)| (a - b).abs() < tol)
}

/// Compute MSE per coordinate for the codebook.
///
/// MSE = Σ_i ∫_{bin_i} (x - centroid[i])² · f(x) dx
fn compute_mse(boundaries: &[f32], centroids: &[f32], dim: usize) -> f32 {
    let n = centroids.len();
    let mut edges = vec![-1.0f32];
    edges.extend_from_slice(boundaries);
    edges.push(1.0);

    let n_samples = 500;
    let mut total_mse = 0.0f64;

    for i in 0..n {
        let lo = edges[i];
        let hi = edges[i + 1];
        let dx = (hi - lo) / n_samples as f32;
        let mut interval_mse = 0.0f64;
        for j in 0..=n_samples {
            let x = lo + dx * j as f32;
            let pdf = beta_pdf(x, dim);
            let w = if j == 0 || j == n_samples { 0.5 } else { 1.0 };
            let diff = x as f64 - centroids[i] as f64;
            interval_mse += diff * diff * pdf as f64 * w;
        }
        total_mse += interval_mse * dx as f64;
    }
    total_mse as f32
}

impl TurboQuantCodebook {
    /// Quantize a value using the codebook. Returns index in `[0, 2^bits)`.
    pub fn quantize(&self, value: f32) -> u8 {
        for (i, &b) in self.boundaries.iter().enumerate() {
            if value < b {
                return i as u8;
            }
        }
        self.boundaries.len() as u8
    }

    /// Dequantize an index back to the centroid value.
    pub fn dequantize(&self, index: u8) -> f32 {
        self.centroids.get(index as usize).copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_2bit() {
        let cb = compute_codebook(128, 2);
        assert_eq!(cb.centroids.len(), 4);
        assert_eq!(cb.boundaries.len(), 3);
        // Centroids should be symmetric around 0
        assert!(cb.centroids[0] < 0.0, "first centroid should be negative");
        assert!(cb.centroids[3] > 0.0, "last centroid should be positive");
    }

    #[test]
    fn test_codebook_3bit() {
        let cb = compute_codebook(128, 3);
        assert_eq!(cb.centroids.len(), 8);
        assert_eq!(cb.boundaries.len(), 7);
    }

    #[test]
    fn test_codebook_4bit() {
        let cb = compute_codebook(64, 4);
        assert_eq!(cb.centroids.len(), 16);
        assert_eq!(cb.boundaries.len(), 15);
        // Boundaries should be monotonically increasing
        for w in cb.boundaries.windows(2) {
            assert!(w[0] < w[1], "boundaries must be monotonically increasing");
        }
    }

    #[test]
    fn test_codebook_roundtrip() {
        let cb = compute_codebook(64, 4);
        // Use values representative of the Beta distribution (concentrated near 0)
        let values = [-0.5f32, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5];
        for &v in &values {
            let idx = cb.quantize(v);
            let reconstructed = cb.dequantize(idx);
            assert!(
                (reconstructed - v).abs() < 0.5,
                "roundtrip failed for {v} -> idx {idx} -> {reconstructed}"
            );
        }
    }

    #[test]
    fn test_mse_decreases_with_bits() {
        let mse_2 = compute_codebook(128, 2).mse_per_coord;
        let mse_3 = compute_codebook(128, 3).mse_per_coord;
        let mse_4 = compute_codebook(128, 4).mse_per_coord;
        assert!(
            mse_2 > mse_3,
            "2-bit MSE {mse_2} should be > 3-bit MSE {mse_3}"
        );
        assert!(
            mse_3 > mse_4,
            "3-bit MSE {mse_3} should be > 4-bit MSE {mse_4}"
        );
    }

    #[test]
    fn test_quantize_extremes() {
        let cb = compute_codebook(64, 3);
        // Very negative → index 0
        assert_eq!(cb.quantize(-10.0), 0);
        // Very positive → last index
        assert_eq!(cb.quantize(10.0), 7);
    }

    #[test]
    fn test_mse_positive() {
        for bits in 2..=6u8 {
            let cb = compute_codebook(64, bits);
            assert!(
                cb.mse_per_coord > 0.0,
                "MSE should be positive for {bits} bits"
            );
        }
    }
}
