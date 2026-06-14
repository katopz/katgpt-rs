//! Representation geometry diagnostics (Plan 151, Research 113).
//!
//! Measures representation health of hidden state vectors via:
//! - **Effective rank** (entropy-based, Roy & Vetterli 2007)
//! - **Average pairwise cosine similarity** (anisotropy metric)
//!
//! High effective rank + low cosine similarity = healthy, isotropic representations.
//! Low effective rank + high cosine similarity = degenerate, collapsed representations.

// ── Core types ──────────────────────────────────────────────────

/// Combined representation geometry report for a set of hidden states.
#[derive(Debug, Clone)]
pub struct GeometryReport {
    pub layer_index: usize,
    pub n_tokens: usize,
    pub hidden_dim: usize,
    pub effective_rank: f32,
    pub avg_cosine_sim: f32,
}

// ── Core functions ──────────────────────────────────────────────

/// Compute the effective rank of a set of hidden state vectors.
///
/// Uses entropy-based effective rank (Roy & Vetterli, 2007) from the
/// eigenvalue spectrum of the empirical covariance matrix.
///
/// High effective rank = healthy, diverse representations.
/// Low effective rank = degenerate, collapsed representations.
///
/// # Panics
/// Panics if `hidden_states` is empty or if vectors have inconsistent dimensions.
pub fn effective_rank(hidden_states: &[Vec<f32>]) -> f32 {
    assert!(!hidden_states.is_empty(), "hidden_states must not be empty");
    let n = hidden_states.len();
    let dim = hidden_states[0].len();
    assert!(dim > 0, "hidden state vectors must be non-empty");

    // Verify consistent dimensions.
    for (i, v) in hidden_states.iter().enumerate() {
        assert_eq!(
            v.len(),
            dim,
            "inconsistent dimensions: vector 0 has len {dim}, vector {i} has len {}",
            v.len()
        );
    }

    // If only one vector, rank is 0 (no variance).
    if n == 1 {
        return 0.0;
    }

    // 1. Compute mean.
    let mut mean = vec![0.0f64; dim];
    for v in hidden_states {
        for (j, &val) in v.iter().enumerate() {
            mean[j] += val as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    // 2. Center each vector (build centered matrix X: n × dim).
    let mut centered = vec![0.0f64; n * dim];
    for (i, v) in hidden_states.iter().enumerate() {
        for (j, &val) in v.iter().enumerate() {
            centered[i * dim + j] = val as f64 - mean[j];
        }
    }

    // 3. Compute covariance matrix C = (1/N) * X^T * X  (dim × dim).
    // We use the smaller of n and dim to decide approach.
    // For typical hidden states, n << dim, so compute C directly.
    let scale = 1.0 / n as f64;
    let mut cov = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let mut sum = 0.0f64;
            for k in 0..n {
                sum += centered[k * dim + i] * centered[k * dim + j];
            }
            let val = sum * scale;
            cov[i * dim + j] = val;
            cov[j * dim + i] = val;
        }
    }

    // 4. Compute eigenvalues via Jacobi iteration.
    let eigenvalues = jacobi_eigenvalues(&mut cov, dim, 50);

    // 5. Normalize eigenvalues to sum to 1.0.
    let total: f64 = eigenvalues.iter().sum();
    if total < 1e-15 {
        return 0.0;
    }
    let inv_total = 1.0 / total;

    // 6. Effective rank = exp(-Σ λ_i * log(λ_i)) — fused with normalization
    // to avoid allocating an intermediate Vec.
    let entropy: f64 = eigenvalues
        .iter()
        .map(|&v| v * inv_total)
        .filter(|&v| v > 1e-15)
        .map(|v| -v * v.ln())
        .sum();

    entropy.exp() as f32
}

/// Compute average pairwise cosine similarity between hidden states.
///
/// High similarity = anisotropic (degenerate), Low = isotropic (healthy).
///
/// # Panics
/// Panics if `hidden_states` is empty or if vectors have inconsistent dimensions
/// or zero norm.
pub fn avg_cosine_similarity(hidden_states: &[Vec<f32>]) -> f32 {
    assert!(!hidden_states.is_empty(), "hidden_states must not be empty");
    let n = hidden_states.len();
    let dim = hidden_states[0].len();
    assert!(dim > 0, "hidden state vectors must be non-empty");

    if n < 2 {
        return 1.0; // Trivially similar to itself.
    }

    // Normalize each vector to unit length.
    let mut normalized = Vec::with_capacity(n * dim);
    for v in hidden_states {
        let norm: f64 = v
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        assert!(norm > 1e-10, "zero-norm vector encountered");
        let inv_norm = 1.0 / norm;
        for &x in v {
            normalized.push((x as f64) * inv_norm);
        }
    }

    // Compute average pairwise dot product.
    let mut total = 0.0f64;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut dot = 0.0f64;
            for d in 0..dim {
                dot += normalized[i * dim + d] * normalized[j * dim + d];
            }
            total += dot;
            count += 1;
        }
    }

    (total / count.max(1) as f64) as f32
}

/// Compute a combined representation geometry report.
///
/// # Panics
/// Panics if `hidden_states` is empty.
pub fn representation_geometry_report(
    hidden_states: &[Vec<f32>],
    layer_index: usize,
) -> GeometryReport {
    assert!(!hidden_states.is_empty(), "hidden_states must not be empty");
    let n_tokens = hidden_states.len();
    let hidden_dim = hidden_states[0].len();

    GeometryReport {
        effective_rank: effective_rank(hidden_states),
        avg_cosine_sim: avg_cosine_similarity(hidden_states),
        layer_index,
        n_tokens,
        hidden_dim,
    }
}

// ── Jacobi eigenvalue algorithm (symmetric matrix) ─────────────
//
// Simple iterative Jacobi rotation to find eigenvalues of a real symmetric
// matrix. Not optimized for large matrices — fine for diagnostic use on
// covariance matrices up to ~256×256.

#[inline]
fn jacobi_eigenvalues(mat: &mut [f64], dim: usize, max_sweeps: usize) -> Vec<f64> {
    // Extract diagonal as initial eigenvalue estimates.
    let mut eigenvalues: Vec<f64> = (0..dim).map(|i| mat[i * dim + i]).collect();

    for _ in 0..max_sweeps {
        // Find the largest off-diagonal element.
        let mut max_val = 0.0f64;
        let (mut p, mut q) = (0, 1);
        for i in 0..dim {
            for j in (i + 1)..dim {
                let val = mat[i * dim + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        // Converged if off-diagonal is negligible.
        if max_val < 1e-12 {
            break;
        }

        // Compute Jacobi rotation angle.
        let app = mat[p * dim + p];
        let aqq = mat[q * dim + q];
        let apq = mat[p * dim + q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply rotation to rows/cols p, q.
        for r in 0..dim {
            if r == p || r == q {
                continue;
            }
            let arp = mat[r * dim + p];
            let arq = mat[r * dim + q];
            mat[r * dim + p] = cos_t * arp + sin_t * arq;
            mat[p * dim + r] = mat[r * dim + p];
            mat[r * dim + q] = -sin_t * arp + cos_t * arq;
            mat[q * dim + r] = mat[r * dim + q];
        }

        let new_pp = cos_t * cos_t * app + 2.0 * sin_t * cos_t * apq + sin_t * sin_t * aqq;
        let new_qq = sin_t * sin_t * app - 2.0 * sin_t * cos_t * apq + cos_t * cos_t * aqq;
        mat[p * dim + p] = new_pp;
        mat[q * dim + q] = new_qq;
        mat[p * dim + q] = 0.0;
        mat[q * dim + p] = 0.0;

        eigenvalues[p] = new_pp;
        eigenvalues[q] = new_qq;
    }

    // Filter out near-zero eigenvalues (numerical noise).
    eigenvalues.retain(|&v| v > 1e-10);
    eigenvalues
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: simple Gaussian-like noise using central limit theorem.
    fn gaussian_noise(rng: &mut fastrand::Rng) -> f32 {
        let sum: f32 = (0..12).map(|_| rng.f32()).sum();
        sum - 6.0
    }

    // ── G1: effective_rank() on known matrix → correct value ─────

    #[test]
    fn g1_effective_rank_known_matrix() {
        // Hand-constructed case: dim=3, 6 vectors that span all 3 dims.
        // After mean-centering, the covariance matrix is full rank.
        // Effective rank is entropy-based, so for full rank with equal
        // eigenvalues it equals dim; for uneven eigenvalues it's < dim.
        // We verify: collapsed → low rank, full-rank → rank close to dim.
        let states: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![-1.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0],
            vec![0.0, 0.0, -1.0],
        ];

        let rank = effective_rank(&states);
        // Symmetric ±basis → mean is zero → covariance is (1/6)*I → all eigenvalues
        // equal → effective rank = 3.
        let dim = 3;
        assert!(
            (rank - dim as f32).abs() < 0.05,
            "effective rank of symmetric ±basis should be ~{dim}, got {rank}"
        );
    }

    #[test]
    fn g1_effective_rank_collapsed_matrix() {
        // All identical vectors → rank ≈ 0 (no variance).
        let states = vec![vec![1.0f32, 2.0, 3.0]; 10];
        let rank = effective_rank(&states);
        assert!(
            rank < 0.1,
            "effective rank of identical vectors should be ~0, got {rank}"
        );
    }

    #[test]
    fn g1_effective_rank_single_vector() {
        let states = vec![vec![1.0f32, 2.0, 3.0]];
        let rank = effective_rank(&states);
        assert!(
            rank < 0.01,
            "effective rank of single vector should be 0, got {rank}"
        );
    }

    // ── G2: avg_cosine_similarity() on orthogonal / identical ────

    #[test]
    fn g2_orthogonal_vectors_similarity_zero() {
        // Standard basis vectors are orthogonal → cosine sim ≈ 0.
        let dim = 4;
        let states: Vec<Vec<f32>> = (0..dim)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i] = 1.0;
                v
            })
            .collect();

        let sim = avg_cosine_similarity(&states);
        assert!(
            sim.abs() < 0.01,
            "orthogonal vectors should have cosine sim ≈ 0, got {sim}"
        );
    }

    #[test]
    fn g2_identical_vectors_similarity_one() {
        let states = vec![vec![1.0f32, 2.0, 3.0, 4.0]; 5];
        let sim = avg_cosine_similarity(&states);
        assert!(
            (sim - 1.0).abs() < 0.01,
            "identical vectors should have cosine sim ≈ 1.0, got {sim}"
        );
    }

    #[test]
    fn g2_opposite_vectors_similarity_minus_one() {
        let states = vec![vec![1.0f32, 0.0, 0.0], vec![-1.0f32, 0.0, 0.0]];
        let sim = avg_cosine_similarity(&states);
        assert!(
            (sim - (-1.0)).abs() < 0.01,
            "opposite vectors should have cosine sim ≈ -1.0, got {sim}"
        );
    }

    // ── G3: Random init → effective_rank > 0.5 * dim ─────────────

    #[test]
    fn g3_random_init_high_effective_rank() {
        let mut rng = fastrand::Rng::with_seed(42);
        let dim = 16;
        let n_tokens = 32;

        // Random isotropic vectors — should span most dimensions.
        let states: Vec<Vec<f32>> = (0..n_tokens)
            .map(|_| (0..dim).map(|_| gaussian_noise(&mut rng)).collect())
            .collect();

        let rank = effective_rank(&states);
        assert!(
            rank > 0.5 * dim as f32,
            "random init should have effective_rank > 0.5 * dim={dim}, got {rank}"
        );
    }

    #[test]
    fn g3_random_init_low_cosine_similarity() {
        let mut rng = fastrand::Rng::with_seed(123);
        let dim = 16;
        let n_tokens = 32;

        let states: Vec<Vec<f32>> = (0..n_tokens)
            .map(|_| (0..dim).map(|_| gaussian_noise(&mut rng)).collect())
            .collect();

        let sim = avg_cosine_similarity(&states);
        assert!(
            sim.abs() < 0.3,
            "random init should have avg_cosine_sim near 0, got {sim}"
        );
    }

    // ── G5: GeometryReport integrates correctly ──────────────────

    #[test]
    fn g5_geometry_report_fields() {
        let mut rng = fastrand::Rng::with_seed(99);
        let dim = 8;
        let n = 10;
        let layer = 3;

        let states: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| gaussian_noise(&mut rng)).collect())
            .collect();

        let report = representation_geometry_report(&states, layer);

        assert_eq!(report.layer_index, layer);
        assert_eq!(report.n_tokens, n);
        assert_eq!(report.hidden_dim, dim);
        assert!(report.effective_rank > 0.0);
        assert!(report.avg_cosine_sim > -1.0 && report.avg_cosine_sim < 1.0);
    }

    #[test]
    fn g5_geometry_report_consistent_with_individual_calls() {
        let mut rng = fastrand::Rng::with_seed(77);
        let dim = 8;
        let n = 12;

        let states: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| gaussian_noise(&mut rng)).collect())
            .collect();

        let erank = effective_rank(&states);
        let asim = avg_cosine_similarity(&states);
        let report = representation_geometry_report(&states, 7);

        assert!(
            (report.effective_rank - erank).abs() < 1e-4,
            "report effective_rank {} != direct {}",
            report.effective_rank,
            erank
        );
        assert!(
            (report.avg_cosine_sim - asim).abs() < 1e-4,
            "report avg_cosine_sim {} != direct {}",
            report.avg_cosine_sim,
            asim
        );
    }
}
