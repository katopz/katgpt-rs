//! PEIRA: Predictive Encoders through Inter-View Regressor Alignment
//!
//! Implementation of the PEIRA auxiliary loss (arXiv:2605.17671) for
//! collapse-free representation alignment. The core computation maintains
//! EMA estimates of cross-view (Σ) and within-view (N) covariance matrices,
//! then computes a closed-form predictor and auxiliary loss without
//! backpropagating through the matrix inverse.
//!
//! All matrices are k×k where k is the representation dimension (typically
//! 128–512), so inversion is O(k³) which is negligible on CPU. No GPU/WGSL
//! needed.

/// Configuration for PEIRA distillation.
///
/// Controls the regularization strength (λ), EMA momentum for covariance
/// tracking, and representation dimension.
#[derive(Debug, Clone)]
pub struct PeiraConfig {
    /// Regularization parameter λ > 0.
    ///
    /// Controls the effective rank of recovered CCA subspace:
    /// - Larger λ → fewer canonical directions recovered (more conservative)
    /// - Smaller λ → more directions (more expressive, potentially noisy)
    ///
    /// Default: 0.1
    pub lambda: f64,
    /// EMA momentum for covariance estimates (0 < α < 1).
    ///
    /// Higher = slower tracking, more stable. Lower = faster adaptation.
    ///
    /// Default: 0.9
    pub ema_rate: f64,
    /// Representation dimension k.
    /// All internal matrices are k×k.
    pub dim: usize,
}

impl Default for PeiraConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            ema_rate: 0.9,
            dim: 8,
        }
    }
}

impl PeiraConfig {
    /// Create a new config with the given dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ..Default::default()
        }
    }

    /// Set regularization λ.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        assert!(lambda > 0.0, "PEIRA λ must be positive, got {lambda}");
        self.lambda = lambda;
        self
    }

    /// Set EMA momentum.
    pub fn with_ema_rate(mut self, rate: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&rate),
            "EMA rate must be in (0, 1), got {rate}"
        );
        self.ema_rate = rate;
        self
    }
}

/// EMA covariance tracker for PEIRA.
///
/// Maintains running estimates of:
/// - **Σ** (cross-view covariance): how student and teacher representations co-vary
/// - **N** (within-view covariance): auto-covariance averaged over both views
///
/// Both are k×k matrices stored in row-major flat layout.
#[derive(Debug, Clone)]
pub struct PeiraCovariance {
    /// Cross-view covariance Σ (k×k), row-major.
    sigma: Vec<f64>,
    /// Within-view covariance N (k×k), row-major.
    n: Vec<f64>,
    /// Configuration.
    config: PeiraConfig,
    /// Number of EMA updates applied.
    step_count: usize,
    /// Pre-allocated scratch for peira_aux_loss
    sigma_sample: Vec<f64>,
    n_sample: Vec<f64>,
    pm: Vec<f64>,
}

impl PeiraCovariance {
    /// Create a new zero-initialized covariance tracker.
    pub fn new(config: PeiraConfig) -> Self {
        let k = config.dim;
        Self {
            sigma: vec![0.0; k * k],
            n: vec![0.0; k * k],
            config,
            step_count: 0,
            sigma_sample: vec![0.0; k * k],
            n_sample: vec![0.0; k * k],
            pm: vec![0.0; k * k],
        }
    }

    /// Get the dimension k.
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    /// Get the number of updates.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Update EMA covariance estimates with a student-teacher pair.
    ///
    /// Both slices must have length `dim`.
    pub fn update(&mut self, student: &[f32], teacher: &[f32]) {
        let k = self.config.dim;
        assert_eq!(student.len(), k, "student repr length mismatch");
        assert_eq!(teacher.len(), k, "teacher repr length mismatch");

        let alpha = self.config.ema_rate;

        // Compute outer products and update EMA
        for i in 0..k {
            let si = student[i] as f64;
            let ti = teacher[i] as f64;
            for j in 0..k {
                let sj = student[j] as f64;
                let tj = teacher[j] as f64;

                // Cross-view: Σ[i,j] = E[u_i * v_j]
                let sigma_ij = si * tj;
                // Within-view: N[i,j] = E[(u_i*u_j + v_i*v_j) / 2]
                let n_ij = (si * sj + ti * tj) / 2.0;

                let idx = i * k + j;
                if self.step_count == 0 {
                    self.sigma[idx] = sigma_ij;
                    self.n[idx] = n_ij;
                } else {
                    self.sigma[idx] = alpha * self.sigma[idx] + (1.0 - alpha) * sigma_ij;
                    self.n[idx] = alpha * self.n[idx] + (1.0 - alpha) * n_ij;
                }
            }
        }
        self.step_count += 1;
    }

    /// Compute the closed-form predictor matrices (P*, Q*).
    ///
    /// - P* = Σ (N + λI)⁻¹  — the optimal linear predictor
    /// - Q* = (N + λI)⁻¹     — the regularized inverse
    ///
    /// Returns (P*, Q*) as flat k×k row-major vectors.
    pub fn predictor(&self) -> (Vec<f64>, Vec<f64>) {
        let k = self.config.dim;
        let lambda = self.config.lambda;

        // Build N + λI
        let mut n_reg = self.n.clone();
        for i in 0..k {
            n_reg[i * k + i] += lambda;
        }

        // Invert N + λI
        let q_star = invert_spd(&n_reg, k);

        // P* = Σ @ Q*
        let p_star = matmul(&self.sigma, &q_star, k);

        (p_star, q_star)
    }

    /// Get a reference to the current Σ matrix (row-major).
    pub fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    /// Get a reference to the current N matrix (row-major).
    pub fn n_matrix(&self) -> &[f64] {
        &self.n
    }

    /// Reset covariance estimates (e.g., at episode boundaries).
    pub fn reset(&mut self) {
        self.sigma.fill(0.0);
        self.n.fill(0.0);
        self.sigma_sample.fill(0.0);
        self.n_sample.fill(0.0);
        self.pm.fill(0.0);
        self.step_count = 0;
    }
}

/// Compute the PEIRA auxiliary loss L_aux.
///
/// L_aux = -½ Tr(Σ A^T) + ¼ Tr(A (N + λI) A^T)
///
/// This formulation avoids differentiating through the matrix inverse.
/// At the optimum A* = (N + λI)⁻¹ Σ^T, L_aux equals the PEIRA objective.
///
/// # Arguments
/// * `student` — Student representation (length k)
/// * `teacher` — Teacher representation (length k)
/// * `p_star` — Predictor matrix P* = Σ(N + λI)⁻¹ (k×k row-major)
/// * `q_star` — Inverse Q* = (N + λI)⁻¹ (k×k row-major)
/// * `lambda` — Regularization parameter
///
/// # Returns
/// The scalar auxiliary loss value.
pub fn peira_aux_loss(
    student: &[f32],
    teacher: &[f32],
    p_star: &[f64],
    q_star: &[f64],
    lambda: f64,
    sigma_sample: &mut [f64],
    n_sample: &mut [f64],
    pm: &mut [f64],
) -> f64 {
    let k = student.len();
    assert_eq!(teacher.len(), k);
    assert_eq!(p_star.len(), k * k);
    assert_eq!(q_star.len(), k * k);
    assert_eq!(sigma_sample.len(), k * k);
    assert_eq!(n_sample.len(), k * k);
    assert_eq!(pm.len(), k * k);

    // Compute the auxiliary loss using the closed-form predictor:
    // L_aux = -½ Tr(Σ P*^T) + ¼ Tr(P* (N + λI) P*^T)
    //
    // Since P* = Σ Q*, and Q* = (N + λI)⁻¹:
    // The loss simplifies to: -½ Tr(Σ Q* Σ^T) + ¼ Tr(Σ Q* Σ^T)
    //                        = -¼ Tr(P* Σ^T)
    //
    // But for numerical accuracy, we compute the full form using the
    // current sample's outer products.

    // Compute sample cross-covariance: sigma_sample = u ⊗ v
    // and sample within-covariance: n_sample = (u ⊗ u + v ⊗ v) / 2
    sigma_sample.fill(0.0);
    n_sample.fill(0.0);

    for i in 0..k {
        let si = student[i] as f64;
        let ti = teacher[i] as f64;
        for j in 0..k {
            let sj = student[j] as f64;
            let tj = teacher[j] as f64;
            sigma_sample[i * k + j] = si * tj;
            n_sample[i * k + j] = (si * sj + ti * tj) / 2.0;
        }
    }

    // Term 1: -½ Tr(Σ_sample @ P*^T) = -½ Σ_{i,j} sigma_sample[i,j] * p_star[j,i]
    // In row-major: Tr(A @ B^T) = Σ_{i,j} A[i,j] * B[i,j] when both are row-major
    let mut term1 = 0.0f64;
    for i in 0..k * k {
        term1 += sigma_sample[i] * p_star[i];
    }
    term1 *= -0.5;

    // Term 2: ¼ Tr(P* @ (N_sample + λI) @ P*^T)
    // = ¼ Tr(P* @ M @ P*^T) where M = N_sample + λI
    // P* @ M is k×k, then (P* @ M) @ P*^T trace
    pm.fill(0.0);
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0f64;
            for l in 0..k {
                let m_lj = if l == j {
                    n_sample[l * k + j] + lambda
                } else {
                    n_sample[l * k + j]
                };
                sum += p_star[i * k + l] * m_lj;
            }
            pm[i * k + j] = sum;
        }
    }

    // Tr(PM @ P^T) = Σ_{i,j} pm[i,j] * p_star[i,j]
    let mut term2 = 0.0f64;
    for i in 0..k * k {
        term2 += pm[i] * p_star[i];
    }
    term2 *= 0.25;

    // Add the regularization penalty: + λ/2 (||u||² + ||v||²)
    let norm_sq_u: f64 = student.iter().map(|x| (*x as f64).powi(2)).sum();
    let norm_sq_v: f64 = teacher.iter().map(|x| (*x as f64).powi(2)).sum();
    let reg = lambda / 2.0 * (norm_sq_u + norm_sq_v);

    term1 + term2 + reg
}

/// Invert a symmetric positive definite (SPD) matrix using Cholesky decomposition.
/// More efficient than Gauss-Jordan for SPD matrices: exploits symmetry,
/// no partial pivoting needed, uses half the memory.
fn invert_spd(mat: &[f64], k: usize) -> Vec<f64> {
    // Step 1: Cholesky decomposition — L such that L * L^T = mat
    let mut l = vec![0.0f64; k * k];
    for j in 0..k {
        // Diagonal
        let mut sum = 0.0f64;
        for p in 0..j {
            sum += l[j * k + p] * l[j * k + p];
        }
        let diag = mat[j * k + j] - sum;
        assert!(
            diag > 0.0,
            "Matrix not positive definite in Cholesky decomposition"
        );
        l[j * k + j] = diag.sqrt();

        // Off-diagonal (lower triangle only)
        for i in (j + 1)..k {
            let mut sum = 0.0f64;
            for p in 0..j {
                sum += l[i * k + p] * l[j * k + p];
            }
            l[i * k + j] = (mat[i * k + j] - sum) / l[j * k + j];
        }
    }

    // Step 2: Invert lower triangular L → L_inv
    let mut l_inv = vec![0.0f64; k * k];
    for i in 0..k {
        l_inv[i * k + i] = 1.0 / l[i * k + i];
        for j in 0..i {
            let mut sum = 0.0f64;
            for p in j..i {
                sum += l[i * k + p] * l_inv[p * k + j];
            }
            l_inv[i * k + j] = -sum / l[i * k + i];
        }
    }

    // Step 3: M_inv = L_inv^T * L_inv (only lower triangle, then mirror)
    let mut inv = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = 0.0f64;
            for p in i..k {
                sum += l_inv[p * k + i] * l_inv[p * k + j];
            }
            inv[i * k + j] = sum;
            inv[j * k + i] = sum; // symmetric
        }
    }

    inv
}

/// Compute matrix product C = A @ B where all are k×k row-major.
fn matmul(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0f64;
            for l in 0..k {
                sum += a[i * k + l] * b[l * k + j];
            }
            c[i * k + j] = sum;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peira_config_validates() {
        let cfg = PeiraConfig::new(16).with_lambda(0.5).with_ema_rate(0.95);
        assert_eq!(cfg.dim, 16);
        assert_eq!(cfg.lambda, 0.5);
        assert_eq!(cfg.ema_rate, 0.95);
    }

    #[test]
    #[should_panic(expected = "must be positive")]
    fn peira_config_rejects_zero_lambda() {
        PeiraConfig::new(4).with_lambda(0.0);
    }

    #[test]
    fn matrix_inverse_identity() {
        let k = 3;
        let identity: Vec<f64> = (0..k)
            .flat_map(|i| (0..k).map(move |j| if i == j { 1.0 } else { 0.0 }))
            .collect();
        let inv = invert_spd(&identity, k);
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i * k + j] - expected).abs() < 1e-10,
                    "inv[{i},{j}] = {} expected {expected}",
                    inv[i * k + j]
                );
            }
        }
    }

    #[test]
    fn matrix_inverse_known() {
        // [[2, 1], [1, 3]] inverse is [[3/5, -1/5], [-1/5, 2/5]]
        let k = 2;
        let mat = vec![2.0, 1.0, 1.0, 3.0];
        let inv = invert_spd(&mat, k);
        let expected = vec![0.6, -0.2, -0.2, 0.4];
        for i in 0..4 {
            assert!(
                (inv[i] - expected[i]).abs() < 1e-10,
                "inv[{i}] = {} expected {}",
                inv[i],
                expected[i]
            );
        }
    }

    #[test]
    fn ema_covariance_tracks_identity() {
        // Feed many identical student=teacher pairs → Σ and N should converge to I
        let k = 4;
        let mut cov = PeiraCovariance::new(PeiraConfig::new(k).with_ema_rate(0.5));

        let repr: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        for _ in 0..100 {
            cov.update(&repr, &repr);
        }

        // Σ[0,0] should be ~1.0, Σ[i,j] for i≠j should be ~0
        let sigma = cov.sigma();
        assert!((sigma[0] - 1.0).abs() < 0.1, "Σ[0,0] = {}", sigma[0]);
        assert!(sigma[1].abs() < 0.1, "Σ[0,1] = {}", sigma[1]);

        // N[0,0] should be ~1.0 (auto-covariance of [1,0,0,0])
        let n = cov.n_matrix();
        assert!((n[0] - 1.0).abs() < 0.1, "N[0,0] = {}", n[0]);
    }

    #[test]
    fn predictor_yields_valid_matrices() {
        let k = 4;
        let mut cov = PeiraCovariance::new(PeiraConfig::new(k).with_lambda(0.1));

        // Feed correlated views
        for _ in 0..50 {
            let student: Vec<f32> = vec![1.0, 0.5, 0.0, 0.0];
            let teacher: Vec<f32> = vec![0.8, 0.4, 0.0, 0.0];
            cov.update(&student, &teacher);
        }

        let (p_star, q_star) = cov.predictor();
        assert_eq!(p_star.len(), k * k);
        assert_eq!(q_star.len(), k * k);

        // Q* should be symmetric positive definite (diagonal dominant)
        for i in 0..k {
            assert!(
                q_star[i * k + i] > 0.0,
                "Q*[{i},{i}] = {} not positive",
                q_star[i * k + i]
            );
        }
    }

    #[test]
    fn aux_loss_is_finite() {
        let k = 4;
        let mut cov = PeiraCovariance::new(PeiraConfig::new(k));
        cov.update(&[1.0, 0.5, -0.3, 0.0], &[0.8, 0.4, -0.2, 0.1]);
        let (p, q) = cov.predictor();
        let mut sigma_sample = vec![0.0f64; k * k];
        let mut n_sample = vec![0.0f64; k * k];
        let mut pm = vec![0.0f64; k * k];
        let loss = peira_aux_loss(
            &[1.0, 0.5, -0.3, 0.0],
            &[0.8, 0.4, -0.2, 0.1],
            &p,
            &q,
            0.1,
            &mut sigma_sample,
            &mut n_sample,
            &mut pm,
        );
        assert!(loss.is_finite(), "Loss is not finite: {loss}");
    }
}
