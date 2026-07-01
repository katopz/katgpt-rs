//! Lloyd-Max scalar quantizers for OCTOPUS triplet components.
//!
//! Two marginals:
//! 1. **Triplet norm** ρ = ||t_i||₂ ∈ [0,1] — Beta-derived distribution
//!    (concentrated near √(3/d), fewer bits via b-1 split)
//! 2. **Oct-coordinate** ξ,η ∈ [-1,1] — triangular marginal from
//!    equal-area octahedral map (more bits via b+1 split)

/// Scalar codebook: centroids + decision boundaries from Lloyd-Max.
#[derive(Debug, Clone)]
pub struct ScalarCodebook {
    /// Centroid values (2^bits entries).
    pub centroids: Vec<f32>,
    /// Decision boundaries (2^bits - 1 entries).
    pub boundaries: Vec<f32>,
    /// MSE at this setting.
    pub mse: f32,
    /// Bits per symbol.
    pub bits: u8,
}

/// Build triplet-norm codebook for dimension `dim` with `nrm_bits` bits.
///
/// After rotation of a d-dimensional unit vector, the norm of each
/// contiguous 3-block (triplet) has distribution:
///   f(ρ) = 2ρ² · (1-ρ²)^((d-5)/2) / B(3/2, (d-3)/2)   for ρ ∈ [0,1]
///
/// Derived from ρ² ~ Beta(3/2, (d-3)/2) via change of variables.
pub fn build_norm_codebook(dim: usize, nrm_bits: u8) -> ScalarCodebook {
    debug_assert!(dim >= 3, "dim must be >= 3 for triplet norm");
    debug_assert!((1..=8).contains(&nrm_bits), "nrm_bits must be in [1, 8]");

    let n_levels = 1usize << nrm_bits;
    let n_boundaries = n_levels - 1;

    // Initialize boundaries at uniform quantiles of [0, 1]
    let mut boundaries: Vec<f32> = (0..n_boundaries)
        .map(|i| (i + 1) as f32 / n_levels as f32)
        .collect();

    // Lloyd-Max iteration
    for _ in 0..50 {
        let centroids = compute_centroids_norm(&boundaries, dim);
        let new_boundaries = compute_boundaries_from_centroids(&centroids);
        if converged(&boundaries, &new_boundaries, 1e-6) {
            let final_centroids = compute_centroids_norm(&new_boundaries, dim);
            let mse = compute_mse_norm(&new_boundaries, &final_centroids, dim);
            return ScalarCodebook {
                centroids: final_centroids,
                boundaries: new_boundaries,
                mse,
                bits: nrm_bits,
            };
        }
        boundaries = new_boundaries;
    }

    let centroids = compute_centroids_norm(&boundaries, dim);
    let mse = compute_mse_norm(&boundaries, &centroids, dim);
    ScalarCodebook {
        centroids,
        boundaries,
        mse,
        bits: nrm_bits,
    }
}

/// Build oct-coordinate codebook with `dir_bits` bits.
///
/// The octahedral map is equal-area, so for uniform random points on S²,
/// (ξ,η) is uniform on the diamond |ξ|+|η|≤1. The 1D marginal is triangular:
///   f(ξ) = 1 - |ξ|   for ξ ∈ [-1,1]
pub fn build_oct_codebook(dir_bits: u8) -> ScalarCodebook {
    debug_assert!((2..=8).contains(&dir_bits), "dir_bits must be in [2, 8]");

    let n_levels = 1usize << dir_bits;
    let n_boundaries = n_levels - 1;

    // Initialize boundaries at uniform quantiles of [-1, 1]
    let mut boundaries: Vec<f32> = (0..n_boundaries)
        .map(|i| -1.0 + 2.0 * (i + 1) as f32 / n_levels as f32)
        .collect();

    // Lloyd-Max iteration with triangular marginal
    for _ in 0..50 {
        let centroids = compute_centroids_oct(&boundaries);
        let new_boundaries = compute_boundaries_from_centroids(&centroids);
        if converged(&boundaries, &new_boundaries, 1e-6) {
            let final_centroids = compute_centroids_oct(&new_boundaries);
            let mse = compute_mse_oct(&new_boundaries, &final_centroids);
            return ScalarCodebook {
                centroids: final_centroids,
                boundaries: new_boundaries,
                mse,
                bits: dir_bits,
            };
        }
        boundaries = new_boundaries;
    }

    let centroids = compute_centroids_oct(&boundaries);
    let mse = compute_mse_oct(&boundaries, &centroids);
    ScalarCodebook {
        centroids,
        boundaries,
        mse,
        bits: dir_bits,
    }
}

// ── Triplet-norm marginal ────────────────────────────────────

/// Triplet-norm PDF on [0,1]: f(ρ) = C · ρ² · (1-ρ²)^((d-5)/2)
///
/// Derived from ρ² ~ Beta(3/2, (d-3)/2) via change of variables.
///
/// Retained for direct testing; hot paths use [`norm_pdf_with`] with pre-computed
/// constants to avoid recomputing `ln_gamma` per sample.
#[cfg(test)]
#[allow(dead_code)]
fn norm_pdf(x: f32, dim: usize) -> f32 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    let (c, exponent) = norm_pdf_constants(dim);
    c * (x * x) * (1.0 - x * x).powf(exponent)
}

/// Pre-computed norm-PDF constants for a given dimension.
///
/// Returns `(C, exponent)` where the PDF is `C · x² · (1-x²)^exponent`.
/// `ln_gamma` (Lanczos, ~10-term series) is the dominant cost per `norm_pdf`
/// call — precomputing once per codebook build avoids millions of redundant
/// gamma evaluations across the Lloyd-Max inner loops.
#[inline]
fn norm_pdf_constants(dim: usize) -> (f32, f32) {
    let d = dim as f64;
    // C = 2 / B(3/2, (d-3)/2) = 2 · Γ(d/2) / (Γ(3/2) · Γ((d-3)/2))
    let log_c =
        std::f64::consts::LN_2 + ln_gamma(d / 2.0) - ln_gamma(1.5) - ln_gamma((d - 3.0) / 2.0);
    let c = log_c.exp() as f32;
    let exponent = ((dim - 5) as f32) / 2.0;
    (c, exponent)
}

/// Evaluate norm PDF using pre-computed constants (hot-loop variant).
#[inline]
fn norm_pdf_with(x: f32, c: f32, exponent: f32) -> f32 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    c * (x * x) * (1.0 - x * x).powf(exponent)
}

/// Compute centroids for norm codebook via numerical integration.
fn compute_centroids_norm(boundaries: &[f32], dim: usize) -> Vec<f32> {
    let n = boundaries.len() + 1;
    let mut centroids = vec![0.0f32; n];
    let n_samples = 1000;

    // Build edges on the stack: [-1.0, boundaries..., 1.0]
    // Using a small fixed-size prefix avoids Vec allocation.
    let mut edges = vec![0.0f32; n + 1];
    edges[0] = 0.0;
    edges[1..=n - 1].copy_from_slice(boundaries);
    edges[n] = 1.0;

    // Precompute PDF constants ONCE per call (was recomputed per sample).
    let (c, exponent) = norm_pdf_constants(dim);

    for i in 0..n {
        let lo = edges[i];
        let hi = edges[i + 1];
        let dx = (hi - lo) / n_samples as f32;
        let mut sum_fx = 0.0f64;
        let mut sum_xfx = 0.0f64;
        for j in 0..=n_samples {
            let x = lo + dx * j as f32;
            let pdf = norm_pdf_with(x, c, exponent);
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

/// MSE for norm codebook.
fn compute_mse_norm(boundaries: &[f32], centroids: &[f32], dim: usize) -> f32 {
    let n = centroids.len();
    let mut edges = vec![0.0f32; n + 1];
    edges[0] = 0.0;
    edges[1..=n - 1].copy_from_slice(boundaries);
    edges[n] = 1.0;

    // Precompute PDF constants ONCE per call (was recomputed per sample).
    let (c, exponent) = norm_pdf_constants(dim);

    let n_samples = 500;
    let mut total_mse = 0.0f64;

    for i in 0..n {
        let lo = edges[i];
        let hi = edges[i + 1];
        let dx = (hi - lo) / n_samples as f32;
        let mut interval_mse = 0.0f64;
        for j in 0..=n_samples {
            let x = lo + dx * j as f32;
            let pdf = norm_pdf_with(x, c, exponent);
            let w = if j == 0 || j == n_samples { 0.5 } else { 1.0 };
            let diff = x as f64 - centroids[i] as f64;
            interval_mse += diff * diff * pdf as f64 * w;
        }
        total_mse += interval_mse * dx as f64;
    }
    total_mse as f32
}

// ── Oct-coordinate marginal ──────────────────────────────────

/// Oct-coordinate PDF: f(ξ) = 1 - |ξ| on [-1,1] (triangular).
///
/// Marginal of uniform distribution on the diamond |ξ|+|η|≤1.
fn oct_pdf(x: f32) -> f32 {
    if x.abs() >= 1.0 {
        return 0.0;
    }
    (1.0 - x.abs()).max(0.0)
}

/// Compute centroids for oct codebook via numerical integration.
fn compute_centroids_oct(boundaries: &[f32]) -> Vec<f32> {
    let n = boundaries.len() + 1;
    let mut centroids = vec![0.0f32; n];
    let n_samples = 1000;

    // Pre-allocate edges: [-1.0, boundaries..., 1.0]
    let mut edges = vec![0.0f32; n + 1];
    edges[0] = -1.0;
    edges[1..=n - 1].copy_from_slice(boundaries);
    edges[n] = 1.0;

    for i in 0..n {
        let lo = edges[i];
        let hi = edges[i + 1];
        let dx = (hi - lo) / n_samples as f32;
        let mut sum_fx = 0.0f64;
        let mut sum_xfx = 0.0f64;
        for j in 0..=n_samples {
            let x = lo + dx * j as f32;
            let pdf = oct_pdf(x);
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

/// MSE for oct codebook.
fn compute_mse_oct(boundaries: &[f32], centroids: &[f32]) -> f32 {
    let n = centroids.len();
    let mut edges = vec![0.0f32; n + 1];
    edges[0] = -1.0;
    edges[1..=n - 1].copy_from_slice(boundaries);
    edges[n] = 1.0;

    let n_samples = 500;
    let mut total_mse = 0.0f64;

    for i in 0..n {
        let lo = edges[i];
        let hi = edges[i + 1];
        let dx = (hi - lo) / n_samples as f32;
        let mut interval_mse = 0.0f64;
        for j in 0..=n_samples {
            let x = lo + dx * j as f32;
            let pdf = oct_pdf(x);
            let w = if j == 0 || j == n_samples { 0.5 } else { 1.0 };
            let diff = x as f64 - centroids[i] as f64;
            interval_mse += diff * diff * pdf as f64 * w;
        }
        total_mse += interval_mse * dx as f64;
    }
    total_mse as f32
}

// ── Shared helpers ────────────────────────────────────────────

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

impl ScalarCodebook {
    /// Quantize a value using nearest-centroid search. Returns index in `[0, 2^bits)`.
    /// Uses binary search on monotonic boundaries — O(log n) instead of O(n).
    #[inline]
    pub fn quantize(&self, value: f32) -> u16 {
        self.boundaries.partition_point(|&b| value >= b) as u16
    }

    /// Dequantize an index back to the centroid value.
    #[inline]
    pub fn dequantize(&self, index: u16) -> f32 {
        self.centroids.get(index as usize).copied().unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_codebook_basic() {
        let cb = build_norm_codebook(128, 2);
        assert_eq!(cb.centroids.len(), 4);
        assert_eq!(cb.boundaries.len(), 3);
        // Norm centroids should be in [0, 1]
        for &c in &cb.centroids {
            assert!((0.0..=1.0).contains(&c), "norm centroid {c} out of [0,1]");
        }
    }

    #[test]
    fn test_oct_codebook_basic() {
        let cb = build_oct_codebook(3);
        assert_eq!(cb.centroids.len(), 8);
        assert_eq!(cb.boundaries.len(), 7);
        // Oct centroids should be in [-1, 1]
        for &c in &cb.centroids {
            assert!((-1.0..=1.0).contains(&c), "oct centroid {c} out of [-1,1]");
        }
    }

    #[test]
    fn test_norm_centroids_near_sqrt_3_over_d() {
        // For d=128, E[ρ] ≈ √(3/d) ≈ 0.153
        let cb = build_norm_codebook(128, 4);
        let mean: f32 = cb.centroids.iter().sum::<f32>() / cb.centroids.len() as f32;
        let expected = (3.0f32 / 128.0).sqrt();
        assert!(
            (mean - expected).abs() < 0.15,
            "mean {mean} too far from expected {expected}"
        );
    }

    #[test]
    fn test_norm_codebook_roundtrip() {
        let cb = build_norm_codebook(128, 3);
        // Values near the expected norm √(3/128) ≈ 0.153
        let values = [0.05f32, 0.1, 0.15, 0.2, 0.3, 0.5];
        for &v in &values {
            let idx = cb.quantize(v);
            let reconstructed = cb.dequantize(idx);
            assert!(
                (reconstructed - v).abs() < 0.3,
                "roundtrip failed for {v} -> idx {idx} -> {reconstructed}"
            );
        }
    }

    #[test]
    fn test_oct_codebook_roundtrip() {
        let cb = build_oct_codebook(4);
        let values = [-0.8f32, -0.4, -0.1, 0.0, 0.1, 0.4, 0.8];
        for &v in &values {
            let idx = cb.quantize(v);
            let reconstructed = cb.dequantize(idx);
            assert!(
                (reconstructed - v).abs() < 0.3,
                "roundtrip failed for {v} -> idx {idx} -> {reconstructed}"
            );
        }
    }

    #[test]
    fn test_mse_decreases_with_bits_norm() {
        let mse_1 = build_norm_codebook(128, 1).mse;
        let mse_2 = build_norm_codebook(128, 2).mse;
        let mse_3 = build_norm_codebook(128, 3).mse;
        assert!(
            mse_1 > mse_2,
            "1-bit MSE {mse_1} should be > 2-bit MSE {mse_2}"
        );
        assert!(
            mse_2 > mse_3,
            "2-bit MSE {mse_2} should be > 3-bit MSE {mse_3}"
        );
    }

    #[test]
    fn test_mse_decreases_with_bits_oct() {
        let mse_2 = build_oct_codebook(2).mse;
        let mse_3 = build_oct_codebook(3).mse;
        let mse_4 = build_oct_codebook(4).mse;
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
    fn test_oct_pdf_triangular() {
        assert!((oct_pdf(0.0) - 1.0).abs() < 1e-6, "peak at 0");
        assert!(oct_pdf(-1.0).abs() < 1e-6, "zero at -1");
        assert!(oct_pdf(1.0).abs() < 1e-6, "zero at 1");
        assert!((oct_pdf(0.5) - 0.5).abs() < 1e-6, "0.5 at 0.5");
    }

    #[test]
    fn test_norm_pdf_peaks_near_sqrt_3_over_d() {
        let d = 128;
        let mode = (3.0f32 / d as f32).sqrt();
        let pdf_mode = norm_pdf(mode, d);
        let pdf_half = norm_pdf(0.5, d);
        assert!(
            pdf_mode > pdf_half,
            "PDF should peak near √(3/d): pdf({mode})={pdf_mode} vs pdf(0.5)={pdf_half}"
        );
    }

    #[test]
    fn test_boundaries_monotonic_norm() {
        let cb = build_norm_codebook(128, 4);
        for w in cb.boundaries.windows(2) {
            assert!(
                w[0] < w[1],
                "norm boundaries must be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_boundaries_monotonic_oct() {
        let cb = build_oct_codebook(4);
        for w in cb.boundaries.windows(2) {
            assert!(
                w[0] < w[1],
                "oct boundaries must be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_quantize_extremes_norm() {
        let cb = build_norm_codebook(128, 3);
        assert_eq!(cb.quantize(-1.0), 0, "very negative → index 0");
        assert_eq!(cb.quantize(2.0), 7, "very positive → last index");
    }

    #[test]
    fn test_quantize_extremes_oct() {
        let cb = build_oct_codebook(3);
        assert_eq!(cb.quantize(-10.0), 0, "very negative → index 0");
        assert_eq!(cb.quantize(10.0), 7, "very positive → last index");
    }

    #[test]
    fn test_oct_symmetry() {
        let cb = build_oct_codebook(4);
        // Triangular distribution is symmetric → centroids should be approximately antisymmetric
        let n = cb.centroids.len();
        for i in 0..n / 2 {
            let lo = cb.centroids[i];
            let hi = cb.centroids[n - 1 - i];
            assert!(
                (lo + hi).abs() < 0.05,
                "centroids not symmetric: [{lo}] + [{hi}] = {}",
                lo + hi
            );
        }
    }
}
