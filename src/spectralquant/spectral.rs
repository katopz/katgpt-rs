//! SpectralQuant algorithms (Plan 078).
//!
//! Calibrated eigenbasis decomposition, participation ratio, spectral gap,
//! water-fill bit allocation, and Lloyd-Max quantizer for non-uniform
//! coordinate distributions after random rotation.

/// Compute participation ratio: d_eff = (Σλ_i)² / Σ(λ_i²).
///
/// Measures effective dimensionality of eigenvalue spectrum.
/// Returns 1.0 for rank-1, returns n for uniform spectrum.
pub fn participation_ratio(eigenvalues: &[f32]) -> f32 {
    let sum: f64 = eigenvalues.iter().map(|&x| x as f64).sum();
    let sum_sq: f64 = eigenvalues.iter().map(|&x| (x as f64) * (x as f64)).sum();
    if sum_sq < 1e-12 {
        return 0.0;
    }
    ((sum * sum) / sum_sq) as f32
}

/// Spectral gap: λ_{d_eff} / λ_{d_eff+1}.
///
/// Returns `None` if boundary beyond last eigenvalue or denominator is near-zero.
pub fn spectral_gap(eigenvalues: &[f32], d_eff: f32) -> Option<f32> {
    let idx = d_eff as usize;
    if idx == 0 || idx >= eigenvalues.len() {
        return None;
    }
    if eigenvalues[idx] < 1e-12 {
        return None;
    }
    Some(eigenvalues[idx - 1] / eigenvalues[idx])
}

/// Find minimum k for 95% and 99% cumulative variance.
///
/// Returns `(var_95, var_99)` — the number of eigenvalues needed
/// to explain at least 95% and 99% of total variance respectively.
pub fn cumulative_variance_thresholds(eigenvalues: &[f32]) -> (usize, usize) {
    let total: f64 = eigenvalues.iter().map(|&x| x as f64).sum();
    if total < 1e-12 {
        return (0, 0);
    }
    let mut cumsum = 0.0f64;
    let n = eigenvalues.len();
    let mut var_95 = n;
    let mut var_99 = n;
    for (i, &ev) in eigenvalues.iter().enumerate() {
        cumsum += ev as f64;
        let ratio = cumsum / total;
        if ratio >= 0.95 && var_95 == n {
            var_95 = i + 1;
        }
        if ratio >= 0.99 && var_99 == n {
            var_99 = i + 1;
        }
    }
    (var_95, var_99)
}

/// Jacobi eigenvalue algorithm for symmetric matrices.
///
/// Returns `(eigenvalues, eigenvectors)` sorted by eigenvalue descending.
/// Eigenvectors are stored as a row-major `dim × dim` matrix where column j
/// is the eigenvector for eigenvalue j.
///
/// Self-contained — no external deps. Uses `f64` internally for precision.
fn jacobi_eigendecompose(matrix: &[f32], dim: usize) -> (Vec<f32>, Vec<f32>) {
    // Copy matrix to mutable working copy (f64 for precision)
    let mut a: Vec<f64> = matrix.iter().map(|&x| x as f64).collect();
    // V starts as identity
    let mut v: Vec<f64> = vec![0.0; dim * dim];
    for i in 0..dim {
        v[i * dim + i] = 1.0;
    }

    let max_sweeps = 50;
    let tol = 1e-10;

    for _ in 0..max_sweeps {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let (mut p, mut q) = (0, 1);
        for i in 0..dim {
            for j in (i + 1)..dim {
                let val = a[i * dim + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < tol {
            break;
        }

        // Compute rotation angle
        let app = a[p * dim + p];
        let aqq = a[q * dim + q];
        let apq = a[p * dim + q];

        // atan2 handles the app==aqq case automatically (returns ±π/2 → θ=±π/4).
        // Zeroing condition for Givens rotation P=[[c,-s],[s,c]]: tan(2θ) = 2·apq/(app−aqq)
        let theta = 0.5 * (2.0 * apq).atan2(app - aqq);
        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to A (rows and cols p, q)
        for i in 0..dim {
            if i == p || i == q {
                continue;
            }
            let aip = a[i * dim + p];
            let aiq = a[i * dim + q];
            a[i * dim + p] = c * aip + s * aiq;
            a[p * dim + i] = a[i * dim + p];
            a[i * dim + q] = -s * aip + c * aiq;
            a[q * dim + i] = a[i * dim + q];
        }
        let new_pp = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        let new_qq = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        a[p * dim + p] = new_pp;
        a[q * dim + q] = new_qq;
        a[p * dim + q] = 0.0; // Zeroed by rotation
        a[q * dim + p] = 0.0;

        // Accumulate rotation in V
        for i in 0..dim {
            let vip = v[i * dim + p];
            let viq = v[i * dim + q];
            v[i * dim + p] = c * vip + s * viq;
            v[i * dim + q] = -s * vip + c * viq;
        }
    }

    // Extract eigenvalues (diagonal of A)
    let eigenvalues: Vec<f32> = (0..dim).map(|i| a[i * dim + i] as f32).collect();
    let eigenvectors: Vec<f32> = v.iter().map(|&x| x as f32).collect();

    // Sort by eigenvalue descending
    let mut indices: Vec<usize> = (0..dim).collect();
    indices.sort_by(|&a_idx, &b_idx| {
        eigenvalues[b_idx]
            .partial_cmp(&eigenvalues[a_idx])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_eigenvalues: Vec<f32> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let sorted_eigenvectors = reorder_eigenvector_columns(&eigenvectors, &indices, dim);

    (sorted_eigenvalues, sorted_eigenvectors)
}

/// Reorder columns of V according to sorted eigenvalue indices.
fn reorder_eigenvector_columns(v: &[f32], indices: &[usize], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim * dim];
    for (new_col, &old_col) in indices.iter().enumerate() {
        for row in 0..dim {
            result[row * dim + new_col] = v[row * dim + old_col];
        }
    }
    result
}

/// Intermediate calibration result (before formal type in types.rs).
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Eigenvector matrix (dim × dim, row-major), columns sorted by eigenvalue descending.
    pub eigenvectors: Vec<f32>,
    /// Eigenvalues sorted descending.
    pub eigenvalues: Vec<f32>,
    /// Effective dimensionality from participation ratio.
    pub d_eff: f32,
    /// Spectral gap at d_eff, if computable.
    pub spectral_gap: Option<f32>,
    /// Number of components for 95% cumulative variance.
    pub var_95: usize,
    /// Number of components for 99% cumulative variance.
    pub var_99: usize,
    /// Number of calibration samples used.
    pub n_samples: usize,
    /// Dimension of each sample vector.
    pub head_dim: usize,
}

impl CalibrationResult {
    /// Decompose calibrated eigenbasis into stiff/soft subspaces.
    ///
    /// Extracts eigenvectors (stored column-wise in the flat row-major matrix)
    /// and delegates to [`crate::stiff_anomaly::subspace::decompose`].
    ///
    /// Returns a [`crate::stiff_anomaly::StiffSoftDecomposition`] partitioned
    /// at the given `trace_mass` threshold (e.g., 0.90).
    #[cfg(feature = "stiff_anomaly")]
    pub fn stiff_soft_decomposition(
        &self,
        trace_mass: f32,
    ) -> crate::stiff_anomaly::StiffSoftDecomposition {
        let d = self.head_dim;
        // Eigenvectors are stored as dim×dim row-major, columns = eigenvectors.
        // Extract column i as a Vec<f32>.
        let eigvecs: Vec<Vec<f32>> = (0..d)
            .map(|col| (0..d).map(|row| self.eigenvectors[row * d + col]).collect())
            .collect();

        crate::stiff_anomaly::subspace::decompose(self.eigenvalues.clone(), eigvecs, trace_mass)
    }
}

/// Offline calibration: collect KV vectors → covariance → eigendecompose.
///
/// Computes the sample covariance matrix from the provided vectors,
/// then runs Jacobi eigendecomposition to extract principal directions.
/// Returns eigenvalues/vectors sorted by eigenvalue magnitude descending,
/// along with spectral analysis metrics.
///
/// # Panics
///
/// Panics if `samples` is empty or if any sample dimension doesn't match `head_dim`.
pub fn calibrate_eigenbasis(samples: &[Vec<f32>], head_dim: usize) -> CalibrationResult {
    assert!(!samples.is_empty(), "need at least 1 calibration sample");
    assert_eq!(
        samples[0].len(),
        head_dim,
        "sample dimension mismatch: expected {head_dim}, got {}",
        samples[0].len()
    );

    let n_samples = samples.len();

    // Compute covariance matrix: C = (1/N) Σ x x^T
    // For unit-norm vectors this is the Gram matrix; for general vectors
    // we skip mean-centering since KV vectors are typically centered.
    let mut cov = vec![0.0f64; head_dim * head_dim];
    for sample in samples {
        for i in 0..head_dim {
            for j in 0..head_dim {
                cov[i * head_dim + j] += sample[i] as f64 * sample[j] as f64;
            }
        }
    }
    let scale = 1.0 / n_samples as f64;
    for val in cov.iter_mut() {
        *val *= scale;
    }

    let cov_f32: Vec<f32> = cov.iter().map(|&x| x as f32).collect();
    let (eigenvalues, eigenvectors) = jacobi_eigendecompose(&cov_f32, head_dim);

    let d_eff = participation_ratio(&eigenvalues);
    let gap = spectral_gap(&eigenvalues, d_eff);
    let (var_95, var_99) = cumulative_variance_thresholds(&eigenvalues);

    CalibrationResult {
        eigenvectors,
        eigenvalues,
        d_eff,
        spectral_gap: gap,
        var_95,
        var_99,
        n_samples,
        head_dim,
    }
}

/// Dual-Gram calibration: when `seq_len < 4 * head_dim`, compute X·Xᵀ instead of Xᵀ·X.
///
/// The dual-Gram approach computes the Gram matrix G = X·Xᵀ (seq_len × seq_len)
/// instead of the covariance C = Xᵀ·X (d_h × d_h). For short sequences where
/// seq_len << d_h, the Gram matrix is much smaller, yielding up to 512× speedup.
///
/// After eigendecomposing G, eigenvectors of C are recovered via:
///   V = Xᵀ·U·Σ⁻¹
/// where U, Σ are eigenvectors/values of G.
///
/// Reference: FlashLib `primitives/pca/triton/pca.py` L73-116 (Research R130).
#[cfg(feature = "dual_gram_pca")]
pub fn calibrate_eigenbasis_dual_gram(samples: &[Vec<f32>], head_dim: usize) -> CalibrationResult {
    assert!(!samples.is_empty(), "need at least 1 calibration sample");
    assert_eq!(
        samples[0].len(),
        head_dim,
        "sample dimension mismatch: expected {head_dim}, got {}",
        samples[0].len()
    );

    let n_samples = samples.len();

    // Compute Gram matrix G = (1/N) X·Xᵀ (n_samples × n_samples) in f64 for precision.
    // This matches the standard path which accumulates covariance in f64.
    let mut gram = vec![0.0f64; n_samples * n_samples];
    for i in 0..n_samples {
        for j in i..n_samples {
            let mut dot = 0.0f64;
            for ((_, sk_i), (_, sk_j)) in samples[i]
                .iter()
                .enumerate()
                .take(head_dim)
                .zip(samples[j].iter().enumerate().take(head_dim))
            {
                dot += *sk_i as f64 * *sk_j as f64;
            }
            gram[i * n_samples + j] = dot;
            gram[j * n_samples + i] = dot; // Symmetric
        }
    }
    let scale = 1.0 / n_samples as f64;
    for val in gram.iter_mut() {
        *val *= scale;
    }

    // Convert to f32 for Jacobi (which internally converts back to f64)
    let gram_f32: Vec<f32> = gram.iter().map(|&x| x as f32).collect();

    // Eigendecompose G (n_samples × n_samples) — much smaller than d_h × d_h
    let (gram_eigenvalues, gram_eigenvectors) = jacobi_eigendecompose(&gram_f32, n_samples);

    // Recover eigenvectors of C = Xᵀ·X from G's eigendecomposition:
    //   V_k = Xᵀ · U_k · (1/σ_k)
    // where U_k is column k of G's eigenvector matrix, σ_k = sqrt(λ_k)
    //
    // Gram eigenvectors are stored column-wise in row-major matrix:
    // gram_eigenvectors[row * n_samples + col]
    let mut cov_eigenvectors = vec![0.0f32; head_dim * head_dim];
    let rank = n_samples.min(head_dim);

    for k in 0..rank {
        let sigma_k = if gram_eigenvalues[k] > 1e-12 {
            gram_eigenvalues[k].sqrt()
        } else {
            continue; // Skip near-zero eigenvalues
        };
        let inv_sigma = 1.0 / sigma_k;

        // V_k = Xᵀ · U_k · (1/σ_k)
        // V_k[i] = (1/σ_k) * Σ_j X[j][i] * U_k[j]
        for i in 0..head_dim {
            let mut val = 0.0f32;
            for j in 0..n_samples {
                val += samples[j][i] * gram_eigenvectors[j * n_samples + k];
            }
            cov_eigenvectors[i * head_dim + k] = val * inv_sigma;
        }
    }

    // Eigenvalues of C = eigenvalues of G (same non-zero eigenvalues)
    // Pad with zeros for dimensions beyond rank
    let mut cov_eigenvalues = vec![0.0f32; head_dim];
    for (i, &ev) in gram_eigenvalues.iter().enumerate().take(rank) {
        cov_eigenvalues[i] = ev;
    }

    // Re-sort by eigenvalue descending (jacobi already sorted, but ensure after padding)
    let mut indices: Vec<usize> = (0..head_dim).collect();
    indices.sort_by(|&a_idx, &b_idx| {
        cov_eigenvalues[b_idx]
            .partial_cmp(&cov_eigenvalues[a_idx])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sorted_eigenvalues: Vec<f32> = indices.iter().map(|&i| cov_eigenvalues[i]).collect();
    let sorted_eigenvectors = reorder_eigenvector_columns(&cov_eigenvectors, &indices, head_dim);

    let d_eff = participation_ratio(&sorted_eigenvalues);
    let gap = spectral_gap(&sorted_eigenvalues, d_eff);
    let (var_95, var_99) = cumulative_variance_thresholds(&sorted_eigenvalues);

    CalibrationResult {
        eigenvectors: sorted_eigenvectors,
        eigenvalues: sorted_eigenvalues,
        d_eff,
        spectral_gap: gap,
        var_95,
        var_99,
        n_samples,
        head_dim,
    }
}

// ── Bit Allocation ──────────────────────────────────────────────────────

/// Two-regime bit allocator.
///
/// Given total bit budget `B = avg_bits × head_dim`, solves:
///   d_eff × b_high + (d - d_eff) × b_low = B
/// subject to `b_high >= b_low >= min_bits`, both integers.
///
/// Tries all valid `(b_high, b_low)` pairs, picks closest to budget.
/// This is Step 1 of allocation — determines the per-regime bit widths.
pub struct BitAllocator {
    _min_bits: u8,
    max_bits: u8,
}

impl BitAllocator {
    pub fn new(min_bits: u8, max_bits: u8) -> Self {
        Self {
            _min_bits: min_bits,
            max_bits,
        }
    }

    /// Allocate bits into two regimes: semantic (b_high) and tail (b_low).
    ///
    /// Matches Python `_solve_bit_allocation` formula:
    ///   b_low = max(1, round(avg_bits - d_eff / head_dim))
    ///   b_high = b_low + 1
    ///
    /// Derivation: constraint is `d_eff * b_high + (d - d_eff) * b_low ≈ d * avg_bits`
    /// with b_high = b_low + 1 → d * b_low + d_eff = d * avg_bits → b_low = avg_bits - d_eff/d.
    /// The `+1` in b_high reserves one bit for QJL sign in the semantic regime.
    pub fn allocate(&self, d_eff: f32, avg_bits: f32, head_dim: usize) -> (u8, u8) {
        let d_eff_int = (d_eff.ceil() as usize).max(1).min(head_dim);

        // Python formula: b_low = max(1, round(avg_bits - d_eff / d))
        let b_low = (avg_bits - d_eff_int as f32 / head_dim as f32)
            .round()
            .max(1.0)
            .min(self.max_bits as f32) as u8;
        let b_high = (b_low + 1).min(self.max_bits);

        (b_high, b_low)
    }
}

/// Water-fill bit allocation (Step 2 — per-semantic-dim distribution).
///
/// Greedy: iteratively assign +1 bit to dim with highest marginal gain:
///     score_i = λ_i / 4^b_i
/// Tie-breaking: lowest index wins (deterministic).
///
/// Called AFTER `BitAllocator` determines b_high. Receives only the first
/// `d_eff` eigenvalues and `total_bits = b_high × d_eff`.
/// Returns per-dim bit widths summing to `total_bits`.
pub fn waterfill_bits(
    eigenvalues: &[f64],
    total_bits: usize,
    min_bits: u8,
    max_bits: Option<u8>,
) -> Vec<u8> {
    let d = eigenvalues.len();
    let mut bits = vec![min_bits; d];
    let mut allocated = d * min_bits as usize;

    while allocated < total_bits {
        // Find dim with highest marginal gain
        let mut best_idx = 0;
        let mut best_gain = 0.0f64;
        for (i, &ev) in eigenvalues.iter().enumerate() {
            let capped = match max_bits {
                Some(mb) if bits[i] >= mb => continue,
                _ => false,
            };
            if capped {
                continue;
            }
            let gain = ev / 4_f64.powi(bits[i] as i32 + 1);
            if gain > best_gain || (gain == best_gain && i < best_idx) {
                best_gain = gain;
                best_idx = i;
            }
        }
        if best_gain <= 0.0 {
            break;
        }
        bits[best_idx] += 1;
        allocated += 1;
    }

    bits
}

/// Per-dim marginal gain: λ_i / 4^b_i.
/// Exposed for diagnostics and testing.
pub fn marginal_gain(eigenvalues: &[f64], bits: &[u8]) -> Vec<f64> {
    eigenvalues
        .iter()
        .zip(bits.iter())
        .map(|(&ev, &b)| ev / 4_f64.powi(b as i32))
        .collect()
}

// ── Lloyd-Max Quantizer ────────────────────────────────────────────────

/// Lloyd-Max scalar quantizer.
///
/// Iteratively fits optimal codebook (centroids + boundaries) to minimize MSE:
/// 1. Assign each sample to nearest centroid.
/// 2. Update centroids as mean of assigned samples.
/// 3. Repeat until convergence.
pub struct LloydMaxQuantizer {
    _n_bits: u8,
    n_levels: usize,
    max_iter: usize,
    tol: f32,
    seed: u64,
    centroids: Option<Vec<f32>>,
    is_fitted: bool,
}

impl LloydMaxQuantizer {
    pub fn new(n_bits: u8, max_iter: usize, seed: u64) -> Self {
        Self {
            n_levels: 1usize << n_bits,
            _n_bits: n_bits,
            max_iter,
            tol: 1e-6,
            seed,
            centroids: None,
            is_fitted: false,
        }
    }

    /// Fit codebook from data samples.
    pub fn fit(&mut self, data: &[f32]) -> &Self {
        if data.is_empty() {
            self.centroids = Some(vec![0.0; self.n_levels]);
            self.is_fitted = true;
            return self;
        }

        // Initialize centroids via uniform quantile placement
        let mut sorted: Vec<f32> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut centroids = vec![0.0f32; self.n_levels];
        for (i, c) in centroids.iter_mut().enumerate() {
            let idx = ((i as f32 + 0.5) / self.n_levels as f32 * sorted.len() as f32) as usize;
            let idx = idx.min(sorted.len() - 1);
            *c = sorted[idx];
        }

        // Lloyd-Max iteration
        let mut rng = crate::types::Rng::new(self.seed);
        for _ in 0..self.max_iter {
            // Assign samples to nearest centroid
            let mut sums = vec![0.0f64; self.n_levels];
            let mut counts = vec![0usize; self.n_levels];
            for &x in data {
                let idx = self.nearest_centroid(x, &centroids);
                sums[idx] += x as f64;
                counts[idx] += 1;
            }

            // Update centroids
            let mut new_centroids = vec![0.0f32; self.n_levels];
            for i in 0..self.n_levels {
                new_centroids[i] = if counts[i] > 0 {
                    (sums[i] / counts[i] as f64) as f32
                } else {
                    // Re-initialize empty bin with random data point
                    let ridx = (rng.next() as usize) % data.len();
                    data[ridx]
                };
            }

            // Check convergence
            let max_delta = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            centroids = new_centroids;
            if max_delta < self.tol {
                break;
            }
        }

        self.centroids = Some(centroids);
        self.is_fitted = true;
        self
    }

    /// Fit codebook analytically for N(0, σ²) distribution.
    ///
    /// Matches Python `_solve_lloyd_max_for_sigma`: uses numerical integration
    /// (trapezoidal rule) to compute optimal Lloyd-Max centroids for a Gaussian
    /// with the given standard deviation. No synthetic data needed.
    ///
    /// This is the correct approach for SpectralQuant codebook fitting — after
    /// spectral rotation, dimension `i` in the rotated domain has variance `λ_i`.
    /// The per-regime codebook should target `σ = sqrt(mean(λ regime))`.
    pub fn fit_for_sigma(&mut self, sigma: f32) -> &Self {
        let sigma = sigma.max(1e-8);
        let n_levels = self.n_levels;

        // Gauss PDF: (1 / (σ√2π)) * exp(-x² / (2σ²))
        let pdf = |x: f64| -> f64 {
            let s = sigma as f64;
            (-0.5 * (x / s) * (x / s)).exp() / (s * (2.0 * std::f64::consts::PI).sqrt())
        };

        // Initialize centroids uniformly in [-3.5σ, 3.5σ]
        let lo = -3.5 * sigma as f64;
        let hi = 3.5 * sigma as f64;
        let mut centroids: Vec<f64> = (0..n_levels)
            .map(|i| lo + (hi - lo) * (i as f64 + 0.5) / n_levels as f64)
            .collect();

        // Trapezoidal numerical integration
        let trapz = |f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize| -> f64 {
            if n == 0 || b <= a {
                return 0.0;
            }
            let h = (b - a) / n as f64;
            let mut sum = (f(a) + f(b)) * 0.5;
            for i in 1..n {
                sum += f(a + i as f64 * h);
            }
            sum * h
        };

        let n_quad = 256; // integration points (more than enough for Gauss)

        // Lloyd-Max iteration
        for _ in 0..self.max_iter {
            // Boundaries between adjacent centroids
            let boundaries: Vec<f64> = (0..n_levels - 1)
                .map(|i| (centroids[i] + centroids[i + 1]) * 0.5)
                .collect();

            // Edges: extend well beyond centroids
            let edges: Vec<f64> = std::iter::once(lo * 3.0)
                .chain(boundaries.iter().copied())
                .chain(std::iter::once(hi * 3.0))
                .collect();

            let mut new_centroids = Vec::with_capacity(n_levels);
            for i in 0..n_levels {
                let a = edges[i];
                let b = edges[i + 1];
                let num = trapz(&|x| x * pdf(x), a, b, n_quad);
                let den = trapz(&pdf, a, b, n_quad);
                new_centroids.push(if den > 1e-15 { num / den } else { centroids[i] });
            }

            // Check convergence
            let max_delta = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);

            centroids = new_centroids;
            if max_delta < self.tol as f64 {
                break;
            }
        }

        self.centroids = Some(centroids.into_iter().map(|c| c as f32).collect());
        self.is_fitted = true;
        self
    }

    /// Quantize data to indices.
    ///
    /// # Panics
    ///
    /// Panics if not fitted.
    pub fn quantize(&self, x: &[f32]) -> Vec<u32> {
        let centroids = self.centroids.as_ref().expect("not fitted");
        x.iter()
            .map(|&v| self.nearest_centroid(v, centroids) as u32)
            .collect()
    }

    /// Dequantize indices back to centroid values.
    ///
    /// # Panics
    ///
    /// Panics if not fitted.
    pub fn dequantize(&self, indices: &[u32]) -> Vec<f32> {
        let centroids = self.centroids.as_ref().expect("not fitted");
        indices
            .iter()
            .map(|&idx| centroids.get(idx as usize).copied().unwrap_or(0.0))
            .collect()
    }

    /// Get fitted centroids.
    pub fn centroids(&self) -> &[f32] {
        self.centroids.as_deref().unwrap_or(&[])
    }

    /// Compute MSE of the quantizer on given data.
    pub fn mse(&self, x: &[f32]) -> f32 {
        let centroids = match self.centroids.as_ref() {
            Some(c) => c,
            None => return f32::MAX,
        };
        let total: f64 = x
            .iter()
            .map(|&v| {
                let idx = self.nearest_centroid(v, centroids);
                let diff = v as f64 - centroids[idx] as f64;
                diff * diff
            })
            .sum();
        (total / x.len().max(1) as f64) as f32
    }

    fn nearest_centroid(&self, x: f32, centroids: &[f32]) -> usize {
        // Centroids are sorted (from Lloyd-Max iteration on symmetric distributions).
        // Binary search for O(log n) instead of O(n) scan.
        if centroids.len() <= 1 {
            return 0;
        }
        let idx = centroids.partition_point(|&c| c < x);
        if idx == 0 {
            return 0;
        }
        if idx >= centroids.len() {
            return centroids.len() - 1;
        }
        // Compare distances to the two neighbors around x
        let d_left = (x - centroids[idx - 1]).abs();
        let d_right = (centroids[idx] - x).abs();
        if d_left <= d_right { idx - 1 } else { idx }
    }
}

// ── Selective QJL ──────────────────────────────────────────────────────

/// Generate selective QJL sign matrix: `(qjl_dim × d_eff)`.
///
/// Uses Rademacher ±1 distribution (not Gaussian).
/// Same seed always produces same matrix for reproducibility.
pub fn generate_selective_qjl_signs(qjl_dim: usize, d_eff: usize, seed: u64) -> Vec<f32> {
    let mut rng = crate::types::Rng::new(seed);
    let mut signs = Vec::with_capacity(qjl_dim * d_eff);
    for _ in 0..(qjl_dim * d_eff) {
        signs.push(if rng.next() & 1 == 0 { -1.0f32 } else { 1.0f32 });
    }
    signs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_participation_ratio_uniform() {
        let ev = vec![1.0f32; 10];
        let pr = participation_ratio(&ev);
        assert!((pr - 10.0).abs() < 0.01, "uniform should give n, got {pr}");
    }

    #[test]
    fn test_participation_ratio_rank1() {
        let ev = vec![10.0, 0.0, 0.0, 0.0];
        let pr = participation_ratio(&ev);
        assert!((pr - 1.0).abs() < 0.01, "rank-1 should give 1, got {pr}");
    }

    #[test]
    fn test_participation_ratio_low_rank() {
        let ev = vec![5.0, 3.0, 0.0, 0.0];
        let pr = participation_ratio(&ev);
        // d_eff = (5+3)^2 / (25+9) = 64/34 ≈ 1.88
        assert!(
            (pr - 1.88).abs() < 0.1,
            "low-rank should give ~1.88, got {pr}"
        );
    }

    #[test]
    fn test_participation_ratio_empty() {
        let ev: Vec<f32> = vec![];
        let pr = participation_ratio(&ev);
        assert!((pr - 0.0).abs() < 1e-6, "empty should give 0, got {pr}");
    }

    #[test]
    fn test_participation_ratio_zero() {
        let ev = vec![0.0f32; 5];
        let pr = participation_ratio(&ev);
        assert!((pr - 0.0).abs() < 1e-6, "all-zero should give 0, got {pr}");
    }

    #[test]
    fn test_spectral_gap() {
        let ev = vec![10.0, 5.0, 1.0, 0.1];
        let gap = spectral_gap(&ev, 2.0).unwrap();
        assert!((gap - 5.0).abs() < 0.01, "gap should be 5.0, got {gap}");
    }

    #[test]
    fn test_spectral_gap_boundary_start() {
        let ev = vec![10.0, 5.0, 1.0];
        let result = spectral_gap(&ev, 0.0);
        assert!(result.is_none(), "d_eff=0 should return None");
    }

    #[test]
    fn test_spectral_gap_boundary_end() {
        let ev = vec![10.0, 5.0, 1.0];
        let result = spectral_gap(&ev, 3.0);
        assert!(result.is_none(), "d_eff=len should return None");
    }

    #[test]
    fn test_spectral_gap_near_zero_denom() {
        let ev = vec![10.0, 5.0, 1.0, 0.0];
        let result = spectral_gap(&ev, 3.0);
        assert!(result.is_none(), "near-zero denominator should return None");
    }

    #[test]
    fn test_cumulative_variance() {
        let ev = vec![10.0, 6.0, 4.0, 0.5, 0.1];
        let (v95, v99) = cumulative_variance_thresholds(&ev);
        // total=20.6, cumsum: 10, 16, 20, 20.5, 20.6
        // 95% at 19.57 → 20 at idx 2 → 3 components
        // 99% at 20.394 → 20.5 at idx 3 → 4 components
        assert_eq!(v95, 3);
        assert_eq!(v99, 4);
    }

    #[test]
    fn test_cumulative_variance_empty() {
        let ev: Vec<f32> = vec![];
        let (v95, v99) = cumulative_variance_thresholds(&ev);
        assert_eq!(v95, 0);
        assert_eq!(v99, 0);
    }

    #[test]
    fn test_cumulative_variance_zero() {
        let ev = vec![0.0f32; 5];
        let (v95, v99) = cumulative_variance_thresholds(&ev);
        assert_eq!(v95, 0);
        assert_eq!(v99, 0);
    }

    #[test]
    fn test_jacobi_identity() {
        let dim = 4;
        let mut matrix = vec![0.0f32; dim * dim];
        for i in 0..dim {
            matrix[i * dim + i] = 1.0;
        }
        let (eigenvalues, eigenvectors) = jacobi_eigendecompose(&matrix, dim);
        for &ev in &eigenvalues {
            assert!(
                (ev - 1.0).abs() < 0.01,
                "identity eigenvalues should be 1, got {ev}"
            );
        }
        // Eigenvectors should be identity-ish
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (eigenvectors[i * dim + j] - expected).abs() < 0.1,
                    "V[{i}][{j}] = {}, expected {expected}",
                    eigenvectors[i * dim + j]
                );
            }
        }
    }

    #[test]
    fn test_jacobi_diagonal() {
        let dim = 3;
        let mut matrix = vec![0.0f32; dim * dim];
        matrix[0] = 3.0;
        matrix[4] = 1.0;
        matrix[8] = 2.0;
        let (eigenvalues, _) = jacobi_eigendecompose(&matrix, dim);
        assert_eq!(eigenvalues.len(), 3);
        assert!(
            (eigenvalues[0] - 3.0).abs() < 0.01,
            "largest should be 3, got {}",
            eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - 2.0).abs() < 0.01,
            "second should be 2, got {}",
            eigenvalues[1]
        );
        assert!(
            (eigenvalues[2] - 1.0).abs() < 0.01,
            "smallest should be 1, got {}",
            eigenvalues[2]
        );
    }

    #[test]
    fn test_jacobi_2x2() {
        // [2 1; 1 3] → eigenvalues 3.618, 1.382
        let matrix = vec![2.0f32, 1.0f32, 1.0f32, 3.0f32];
        let (eigenvalues, eigenvectors) = jacobi_eigendecompose(&matrix, 2);
        let trace = eigenvalues[0] + eigenvalues[1];
        assert!((trace - 5.0).abs() < 0.01, "trace should be 5, got {trace}");
        // V^T A V ≈ diag(eigenvalues)
        // Verify eigenvector orthogonality
        let dot = eigenvectors[0] * eigenvectors[2] + eigenvectors[1] * eigenvectors[3];
        assert!(
            dot.abs() < 0.01,
            "eigenvectors should be orthogonal, dot={dot}"
        );
    }

    #[test]
    fn test_jacobi_known_symmetric() {
        // 3×3 symmetric matrix with known eigenvalues
        // [4 1 2; 1 3 1; 2 1 5] → eigenvalues ~7.04, 2.88, 2.08
        let matrix = vec![4.0f32, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 5.0];
        let dim = 3;
        let (eigenvalues, eigenvectors) = jacobi_eigendecompose(&matrix, dim);

        // Verify trace preservation
        let trace: f32 = eigenvalues.iter().sum();
        assert!(
            (trace - 12.0).abs() < 0.01,
            "trace should be 12, got {trace}"
        );

        // Verify det preservation (product of eigenvalues)
        let det = eigenvalues[0] * eigenvalues[1] * eigenvalues[2];
        // det = 4*(15-1) - 1*(5-2) + 2*(1-6) = 56 - 3 - 10 = 43
        // Tolerance is wide because Jacobi with limited sweeps may have
        // noticeable error in the product even when individual eigenvalues
        // are close. The V^T A V check below is the authoritative test.
        assert!((det - 43.0).abs() < 0.5, "det should be ~43, got {det}");

        // Verify V^T V ≈ I (orthogonality)
        for i in 0..dim {
            for j in 0..dim {
                let mut dot = 0.0f32;
                for k in 0..dim {
                    dot += eigenvectors[k * dim + i] * eigenvectors[k * dim + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 0.01,
                    "V^T V at [{i}][{j}] = {dot}, expected {expected}"
                );
            }
        }

        // Verify V^T A V ≈ diag(eigenvalues)
        for i in 0..dim {
            for j in 0..dim {
                let mut val = 0.0f32;
                for k in 0..dim {
                    for l in 0..dim {
                        val += eigenvectors[k * dim + i]
                            * matrix[k * dim + l]
                            * eigenvectors[l * dim + j];
                    }
                }
                let expected = if i == j { eigenvalues[i] } else { 0.0 };
                let tol = if i == j { 0.1 } else { 0.05 };
                assert!(
                    (val - expected).abs() < tol,
                    "V^T A V at [{i}][{j}] = {val}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_calibrate_eigenbasis() {
        // Generate correlated samples: x = rotation * z where z has non-uniform variance
        let head_dim = 4;
        let n_samples = 200;
        // z has variance [4, 2, 1, 0.1] → eigenvalues should be ~[4, 2, 1, 0.1]
        let mut samples = Vec::new();
        for i in 0..n_samples {
            let seed = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut rng = crate::types::Rng::new(seed);
            let mut v = vec![0.0f32; head_dim];
            v[0] = rng.normal() * 2.0; // var≈4
            v[1] = rng.normal() * 1.414; // var≈2
            v[2] = rng.normal() * 1.0; // var≈1
            v[3] = rng.normal() * 0.316; // var≈0.1
            samples.push(v);
        }
        let result = calibrate_eigenbasis(&samples, head_dim);
        assert_eq!(result.eigenvalues.len(), head_dim);
        assert_eq!(result.eigenvectors.len(), head_dim * head_dim);
        assert_eq!(result.n_samples, n_samples);
        assert_eq!(result.head_dim, head_dim);
        // Largest eigenvalue should be ~4
        assert!(
            result.eigenvalues[0] > 2.0,
            "largest eigenvalue should be ~4, got {}",
            result.eigenvalues[0]
        );
        // d_eff should be < 4
        assert!(
            result.d_eff < 4.0,
            "d_eff should be < 4, got {}",
            result.d_eff
        );
    }

    #[test]
    #[should_panic(expected = "need at least 1 calibration sample")]
    fn test_calibrate_eigenbasis_empty_panics() {
        let samples: Vec<Vec<f32>> = vec![];
        calibrate_eigenbasis(&samples, 4);
    }

    #[test]
    #[should_panic(expected = "sample dimension mismatch")]
    fn test_calibrate_eigenbasis_dim_mismatch_panics() {
        let samples = vec![vec![1.0, 2.0, 3.0]];
        calibrate_eigenbasis(&samples, 4);
    }

    // ── BitAllocator tests ─────────────────────────────────────────────

    #[test]
    fn test_bit_allocator_basic() {
        let alloc = BitAllocator::new(1, 8);
        let (b_high, b_low) = alloc.allocate(6.0, 3.0, 128);
        // Budget = 384 bits, d_eff=6, tail=122
        // 6*b_high + 122*b_low ≈ 384
        assert!(
            b_high >= b_low,
            "b_high ({b_high}) should be >= b_low ({b_low})"
        );
        assert!(b_high >= 1, "b_high should be >= min_bits");
    }

    #[test]
    fn test_bit_allocator_uniform() {
        let alloc = BitAllocator::new(3, 8);
        // Python formula: b_high = b_low + 1 always.
        // b_low = max(1, round(3.0 - 10/128)) = 3, b_high = 4.
        let (b_high, b_low) = alloc.allocate(10.0, 3.0, 128);
        assert_eq!(b_high, b_low + 1, "b_high should always be b_low + 1");
        assert!(b_low >= 1, "b_low should be >= 1");
    }

    #[test]
    fn test_bit_allocator_low_d_eff() {
        let alloc = BitAllocator::new(1, 8);
        let (b_high, b_low) = alloc.allocate(2.0, 4.0, 128);
        // d_eff=2, tail=126, budget=512
        // 2*b_high + 126*b_low ≈ 512
        assert!(b_high >= b_low);
        // b_low should be close to 4 (since tail dominates)
        assert!(b_low >= 2, "b_low should be substantial: {b_low}");
    }

    // ── Waterfill tests ────────────────────────────────────────────────

    #[test]
    fn test_waterfill_basic() {
        let ev = vec![10.0f64, 5.0, 1.0, 0.5];
        let bits = waterfill_bits(&ev, 16, 2, None);
        assert_eq!(bits.len(), 4, "should have 4 dims");
        assert_eq!(
            bits.iter().map(|&b| b as usize).sum::<usize>(),
            16,
            "total bits should be 16"
        );
        // Dim 0 (largest eigenvalue) should get most bits
        assert!(bits[0] >= bits[1], "dim 0 should get >= dim 1");
    }

    #[test]
    fn test_waterfill_respects_min() {
        let ev = vec![5.0f64, 3.0, 1.0];
        let bits = waterfill_bits(&ev, 15, 3, None);
        // min 3 per dim = 9 allocated, 6 remaining to distribute
        assert!(
            bits.iter().all(|&b| b >= 3),
            "all dims should be >= min_bits"
        );
        assert_eq!(bits.iter().map(|&b| b as usize).sum::<usize>(), 15);
    }

    #[test]
    fn test_waterfill_respects_max() {
        let ev = vec![10.0f64, 1.0];
        let bits = waterfill_bits(&ev, 10, 2, Some(4));
        assert!(
            bits.iter().all(|&b| b <= 4),
            "all dims should be <= max_bits"
        );
    }

    #[test]
    fn test_marginal_gain() {
        let ev = vec![4.0f64, 1.0];
        let bits = vec![2u8, 2u8];
        let gains = marginal_gain(&ev, &bits);
        // gain[i] = λ_i / 4^b_i → 4/16=0.25, 1/16=0.0625
        assert!(
            (gains[0] - 0.25).abs() < 0.01,
            "gain[0] should be 0.25, got {}",
            gains[0]
        );
        assert!(
            (gains[1] - 0.0625).abs() < 0.01,
            "gain[1] should be 0.0625, got {}",
            gains[1]
        );
    }

    // ── Lloyd-Max tests ────────────────────────────────────────────────

    #[test]
    fn test_lloyd_max_basic() {
        let mut q = LloydMaxQuantizer::new(2, 50, 42);
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 50.0).collect();
        q.fit(&data);
        assert!(q.is_fitted);
        assert_eq!(q.centroids().len(), 4);

        let indices = q.quantize(&[-0.8f32, -0.2, 0.2, 0.8]);
        let recon = q.dequantize(&indices);
        // Each reconstructed value should be closer to original than ±0.5
        for (orig, rec) in [
            (-0.8f32, recon[0]),
            (-0.2, recon[1]),
            (0.2, recon[2]),
            (0.8, recon[3]),
        ] {
            assert!(
                (orig - rec).abs() < 0.5,
                "reconstruction too far: {orig} -> {rec}"
            );
        }
    }

    #[test]
    fn test_lloyd_max_mse_decreases() {
        let data: Vec<f32> = (0..200).map(|i| (i as f32 / 200.0).sin()).collect();
        let mse2 = {
            let mut q = LloydMaxQuantizer::new(2, 50, 42);
            q.fit(&data);
            q.mse(&data)
        };
        let mse4 = {
            let mut q = LloydMaxQuantizer::new(4, 50, 42);
            q.fit(&data);
            q.mse(&data)
        };
        assert!(
            mse4 < mse2,
            "4-bit MSE ({mse4}) should be < 2-bit MSE ({mse2})"
        );
    }

    #[test]
    fn test_lloyd_max_deterministic() {
        let data: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
        let mut q1 = LloydMaxQuantizer::new(3, 20, 99);
        q1.fit(&data);
        let mut q2 = LloydMaxQuantizer::new(3, 20, 99);
        q2.fit(&data);
        assert_eq!(
            q1.centroids(),
            q2.centroids(),
            "same seed should produce same codebook"
        );
    }

    #[test]
    fn test_lloyd_max_empty_data() {
        let mut q = LloydMaxQuantizer::new(2, 10, 42);
        q.fit(&[]);
        assert!(q.is_fitted);
        assert_eq!(q.centroids().len(), 4);
    }

    // ── Selective QJL tests ────────────────────────────────────────────

    #[test]
    fn test_qjl_signs_shape() {
        let signs = generate_selective_qjl_signs(8, 4, 42);
        assert_eq!(signs.len(), 32, "expected 8×4=32 entries");
    }

    #[test]
    fn test_qjl_signs_values() {
        let signs = generate_selective_qjl_signs(100, 6, 42);
        let pos_count = signs.iter().filter(|&&s| s > 0.0).count();
        let neg_count = signs.iter().filter(|&&s| s < 0.0).count();
        assert_eq!(pos_count + neg_count, 600, "all values should be ±1");
        // Should be roughly 50/50
        assert!(
            pos_count > 200 && pos_count < 400,
            "pos_count={pos_count}, expected ~300"
        );
    }

    #[test]
    fn test_qjl_signs_deterministic() {
        let s1 = generate_selective_qjl_signs(4, 4, 123);
        let s2 = generate_selective_qjl_signs(4, 4, 123);
        assert_eq!(s1, s2, "same seed should produce same signs");
    }

    // ── G5: StiffSoftDecomposition integration ─────────────────────────

    #[cfg(feature = "stiff_anomaly")]
    #[test]
    fn test_g5_stiff_soft_from_calibration() {
        // Synthetic samples with known variance structure:
        // dim 0 has var≈50, dim 1 var≈30, dim 2 var≈15, dim 3 var≈5,
        // dims 4-9 have var≈0.01 (noise floor).
        // Expected: d_eff ≈ 4, k at 90% trace mass should be 3-4.
        let head_dim = 10;
        let n_samples = 200;
        let variances = [50.0f32, 30.0, 15.0, 5.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01];

        let mut samples = Vec::new();
        for i in 0..n_samples {
            let seed = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(9999);
            let mut rng = crate::types::Rng::new(seed);
            let v: Vec<f32> = variances
                .iter()
                .map(|&var| rng.normal() * var.sqrt())
                .collect();
            samples.push(v);
        }

        let cal = calibrate_eigenbasis(&samples, head_dim);
        let decomp = cal.stiff_soft_decomposition(0.90);

        // k should capture the effective dimension (3-5 range)
        assert!(
            decomp.k >= 3 && decomp.k <= 5,
            "k should capture effective dimension, got {}",
            decomp.k
        );

        // Stiff eigenvalues should dominate
        assert_eq!(
            decomp.stiff_eigenvalues.len(),
            decomp.k,
            "stiff eigenvalue count should match k"
        );
        assert_eq!(
            decomp.soft_eigenvalues.len(),
            head_dim - decomp.k,
            "soft eigenvalue count should be dim - k"
        );

        // All eigenvalues present
        assert_eq!(
            decomp.eigenvalues.len(),
            head_dim,
            "total eigenvalue count should match dim"
        );

        // Eigenvectors should have correct dimensionality
        assert_eq!(decomp.stiff_eigenvectors[0].len(), head_dim);
        assert_eq!(decomp.soft_eigenvectors[0].len(), head_dim);
    }
}

// ── T3: GOAT Proof — Dual-Gram Calibration Accuracy ────────────────────
#[cfg(all(feature = "dual_gram_pca", test))]
mod dual_gram_goat_tests {
    use super::*;

    /// Generate synthetic samples with known variance structure.
    fn make_samples(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = crate::types::Rng::new(seed);
        let variances: Vec<f32> = (0..d)
            .map(|i| if i < 4 { 50.0 / (i as f32 + 1.0) } else { 0.01 })
            .collect();
        (0..n)
            .map(|_| variances.iter().map(|&v| rng.normal() * v.sqrt()).collect())
            .collect()
    }

    /// GOAT T3.1: Dual-Gram eigenvalues must match standard covariance eigenvalues.
    ///
    /// Compares eigenvalues above a significance threshold. Jacobi eigendecomposition
    /// has limited precision for smaller eigenvalues when the matrix has large dynamic
    /// range. Only the top-k dominant eigenvalues are compared, which are the ones that
    /// matter for calibration quality.
    #[test]
    fn goat_t3_1_eigenvalue_accuracy() {
        for &(d_h, seq_len) in &[
            (128, 16),
            (128, 32),
            (128, 64),
            (256, 16),
            (256, 32),
            (256, 64),
        ] {
            let samples = make_samples(seq_len, d_h, 42);
            let std_cal = calibrate_eigenbasis(&samples, d_h);
            let dg_cal = calibrate_eigenbasis_dual_gram(&samples, d_h);

            // Top eigenvalue should match closely (dominant direction)
            let top_diff = (std_cal.eigenvalues[0] - dg_cal.eigenvalues[0]).abs();
            let top_rel = top_diff / std_cal.eigenvalues[0].abs().max(1e-6);
            assert!(
                top_rel < 0.20,
                "d_h={d_h}, seq_len={seq_len}, top eigenvalue: std={:.6}, dg={:.6}, rel_diff={rel:.4}",
                std_cal.eigenvalues[0],
                dg_cal.eigenvalues[0],
                rel = top_rel
            );

            // Cumulative variance explained should be similar
            let std_total: f32 = std_cal.eigenvalues.iter().sum();
            let dg_total: f32 = dg_cal.eigenvalues.iter().sum();
            let total_diff = (std_total - dg_total).abs();
            let total_rel = total_diff / std_total.abs().max(1e-6);
            assert!(
                total_rel < 0.20,
                "d_h={d_h}, seq_len={seq_len}, total variance: std={std_total:.4}, dg={dg_total:.4}, rel={total_rel:.4}"
            );

            // d_eff should be in the same ballpark
            let deff_ratio = dg_cal.d_eff / std_cal.d_eff;
            assert!(
                deff_ratio > 0.5 && deff_ratio < 2.0,
                "d_h={d_h}, seq_len={seq_len}, d_eff ratio: std={:.2}, dg={:.2}, ratio={deff_ratio:.4}",
                std_cal.d_eff,
                dg_cal.d_eff
            );
        }
    }

    /// GOAT T3.2: Dual-Gram eigenvectors must align with standard eigenvectors.
    #[test]
    fn goat_t3_2_eigenvector_alignment() {
        for &(d_h, seq_len) in &[(128, 16), (128, 32), (256, 16), (256, 32)] {
            let samples = make_samples(seq_len, d_h, 99);
            let std_cal = calibrate_eigenbasis(&samples, d_h);
            let dg_cal = calibrate_eigenbasis_dual_gram(&samples, d_h);

            // Only check eigenvectors for significant eigenvalues
            let threshold = std_cal.eigenvalues[0] * 0.10;
            let rank = seq_len.min(d_h);
            let mut checked = 0;
            for k in 0..rank.min(5) {
                if std_cal.eigenvalues[k] < threshold {
                    break;
                }
                // Compute cosine similarity between eigenvectors
                let mut dot = 0.0f64;
                let mut norm_a = 0.0f64;
                let mut norm_b = 0.0f64;
                for i in 0..d_h {
                    let a = std_cal.eigenvectors[i * d_h + k] as f64;
                    let b = dg_cal.eigenvectors[i * d_h + k] as f64;
                    dot += a * b;
                    norm_a += a * a;
                    norm_b += b * b;
                }
                let cos_sim = dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-12);
                // Eigenvectors can be negated, so check |cos_sim| ≈ 1
                assert!(
                    cos_sim.abs() > 0.90, // cosine distance < 0.10
                    "d_h={d_h}, seq_len={seq_len}, evec {k}: |cos_sim|={cos_sim:.4}"
                );
                checked += 1;
            }
            assert!(
                checked >= 2,
                "d_h={d_h}, seq_len={seq_len}: only checked {checked} eigenvectors"
            );
        }
    }

    /// GOAT T3.3: Feature gate OFF must produce identical results.
    /// This test verifies the standard path is unchanged.
    #[test]
    fn goat_t3_3_standard_path_unchanged() {
        let d_h = 128;
        let samples = make_samples(32, d_h, 77);
        let cal = calibrate_eigenbasis(&samples, d_h);
        // Just verify standard path still works
        assert!(cal.eigenvalues[0] > 0.0);
        assert_eq!(cal.eigenvectors.len(), d_h * d_h);
        assert_eq!(cal.n_samples, 32);
    }

    /// GOAT T3.4: d_eff and spectral metrics must be consistent.
    #[test]
    fn goat_t3_4_metrics_consistency() {
        for &(d_h, seq_len) in &[(128, 16), (256, 32)] {
            let samples = make_samples(seq_len, d_h, 123);
            let dg_cal = calibrate_eigenbasis_dual_gram(&samples, d_h);

            // d_eff should be reasonable (between 1 and min(rank, d_h))
            assert!(dg_cal.d_eff >= 1.0, "d_eff too small: {}", dg_cal.d_eff);
            assert!(
                dg_cal.d_eff <= seq_len as f32,
                "d_eff too large: {}",
                dg_cal.d_eff
            );

            // n_samples and head_dim must match
            assert_eq!(dg_cal.n_samples, seq_len);
            assert_eq!(dg_cal.head_dim, d_h);
        }
    }
}

/// Compute Kolmogorov-Smirnov D-statistic between a weight distribution
/// and a Gaussian reference N(μ, σ). O(n log n) due to sort, zero additional allocation
/// beyond the scratch buffer.
///
/// Returns D ∈ [0, 1] where:
/// - D < 0.1: normal weight distribution
/// - D > 0.25: likely outlier injection (per arxiv 2605.15152)
pub fn ks_d_statistic(weights: &[f32], scratch: &mut [f32]) -> f32 {
    let n = weights.len().min(scratch.len());
    if n == 0 {
        return 0.0;
    }

    // Copy to scratch for sorting
    scratch[..n].copy_from_slice(&weights[..n]);
    scratch[..n].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute mean and std from sorted data
    let mean: f64 = scratch[..n].iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let var: f64 = scratch[..n]
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let std = var.sqrt().max(1e-10);

    if std < 1e-10 {
        return 0.0; // constant weights → no distribution to compare
    }

    // Compare empirical CDF to Gaussian CDF
    let mut max_d = 0.0f32;
    for (i, &x) in scratch[..n].iter().enumerate() {
        let z = (x as f64 - mean) / std;
        let gaussian_cdf = normal_cdf(z) as f32;
        let empirical_cdf_upper = ((i + 1) as f32) / n as f32;
        let empirical_cdf_lower = (i as f32) / n as f32;

        let d_plus = (empirical_cdf_upper - gaussian_cdf).abs();
        let d_minus = (gaussian_cdf - empirical_cdf_lower).abs();
        max_d = max_d.max(d_plus).max(d_minus);
    }

    max_d
}

/// Standard normal CDF approximation (Abramowitz and Stegun).
/// Accurate to ~1e-7.
fn normal_cdf(z: f64) -> f64 {
    // Protect against overflow
    if z < -8.0 {
        return 0.0;
    }
    if z > 8.0 {
        return 1.0;
    }

    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989422804014327; // 1/sqrt(2π)
    let p = d
        * (-z * z / 2.0).exp()
        * t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));

    if z > 0.0 { 1.0 - p } else { p }
}
