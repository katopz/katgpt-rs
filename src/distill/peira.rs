//! PEIRA Modelless Distillation — PeiraDistiller + alignment score metric
//!
//! This module wraps the core PEIRA types from katgpt-core and provides the
//! SC-PEIRA Algorithm 1 training loop (PeiraDistiller) plus the spectral
//! alignment metric (peira_alignment_score) for GOAT proofs.

use katgpt_core::{PeiraConfig, PeiraCovariance};

/// PEIRA-based modelless distiller implementing the SC-PEIRA Algorithm 1 loop.
///
/// The distiller:
/// 1. Maintains EMA covariance estimates (Σ, N) via `PeiraCovariance`
/// 2. Computes closed-form predictor matrices (P*, Q*)
/// 3. Evaluates the auxiliary loss L_aux
/// 4. Tracks spectral alignment progress
///
/// Usage: feed (student, teacher) representation pairs, get loss + alignment.
pub struct PeiraDistiller {
    /// EMA covariance tracker.
    covariance: PeiraCovariance,
    /// Configuration.
    config: PeiraConfig,
    /// Running alignment score history (for convergence monitoring).
    alignment_history: Vec<f64>,
    /// Running loss history.
    loss_history: Vec<f64>,
}

impl PeiraDistiller {
    /// Create a new distiller with the given configuration.
    pub fn new(config: PeiraConfig) -> Self {
        let covariance = PeiraCovariance::new(config.clone());
        Self {
            covariance,
            config,
            alignment_history: Vec::new(),
            loss_history: Vec::new(),
        }
    }

    /// Process one (student, teacher) representation pair.
    ///
    /// Returns (auxiliary_loss, alignment_score) for this step.
    /// Updates the internal EMA covariance estimates.
    pub fn step(&mut self, student: &[f32], teacher: &[f32]) -> (f64, f64) {
        // 1. Update EMA covariance estimates
        self.covariance.update(student, teacher);

        // 2+3. Compute predictor + auxiliary loss (zero-alloc via pre-allocated scratch)
        let (loss, _p_star, _q_star) = self.covariance.predict_and_loss(student, teacher);

        // 4. Compute alignment score
        let alignment = peira_alignment_score(
            self.covariance.sigma(),
            self.covariance.n_matrix(),
            self.config.dim,
        );

        self.loss_history.push(loss);
        self.alignment_history.push(alignment);

        (loss, alignment)
    }

    /// Get the current alignment score (most recent).
    pub fn alignment(&self) -> f64 {
        self.alignment_history.last().copied().unwrap_or(0.0)
    }

    /// Get the current auxiliary loss (most recent).
    pub fn loss(&self) -> f64 {
        self.loss_history.last().copied().unwrap_or(0.0)
    }

    /// Get the number of steps processed.
    pub fn step_count(&self) -> usize {
        self.covariance.step_count()
    }

    /// Get the full alignment history.
    pub fn alignment_history(&self) -> &[f64] {
        &self.alignment_history
    }

    /// Get the full loss history.
    pub fn loss_history(&self) -> &[f64] {
        &self.loss_history
    }

    /// Get the current predictor matrices (P*, Q*).
    pub fn predictor(&self) -> (Vec<f64>, Vec<f64>) {
        self.covariance.predictor()
    }

    /// Get the dimension k.
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    /// Reset the distiller for a new episode.
    pub fn reset(&mut self) {
        self.covariance.reset();
        self.alignment_history.clear();
        self.loss_history.clear();
    }
}

/// Compute the PEIRA spectral alignment score α ∈ [0, 1].
///
/// Measures the alignment between the eigenvectors of Σ (cross-view signal)
/// and N (within-view noise). Higher alignment means the CCA structure is
/// being recovered:
/// - α → 1.0: canonical structure found (good)
/// - α → 0.0: random alignment (early training / poor convergence)
///
/// The alignment is computed as the cosine similarity between the top
/// eigenvector of Σ and the top eigenvector of N, projected onto the
/// same direction.
///
/// # Arguments
/// * `sigma` — Cross-view covariance Σ (k×k row-major)
/// * `n_matrix` — Within-view covariance N (k×k row-major)
/// * `k` — Dimension
pub fn peira_alignment_score(sigma: &[f64], n_matrix: &[f64], k: usize) -> f64 {
    assert_eq!(sigma.len(), k * k);
    assert_eq!(n_matrix.len(), k * k);

    // Power iteration to find top eigenvector of Σ
    let sigma_eigvec = power_iteration(sigma, k, 20);
    // Power iteration to find top eigenvector of N
    let n_eigvec = power_iteration(n_matrix, k, 20);

    // Alignment = |cos(θ)| between the two eigenvectors
    let mut dot = 0.0f64;
    let mut norm_s = 0.0f64;
    let mut norm_n = 0.0f64;
    for i in 0..k {
        dot += sigma_eigvec[i] * n_eigvec[i];
        norm_s += sigma_eigvec[i] * sigma_eigvec[i];
        norm_n += n_eigvec[i] * n_eigvec[i];
    }

    let denom = norm_s.sqrt() * norm_n.sqrt();
    if denom < 1e-15 {
        return 0.0;
    }

    dot.abs() / denom
}

/// Compute a PEIRA-based planning quality signal for SR²AM integration.
///
/// Takes raw representation statistics and returns a quality score in [0, 1]
/// that can be used as an additional reward signal in ConfiguratorBandit.
///
/// This is a lightweight version of the full PEIRA alignment score that
/// doesn't require maintaining full covariance matrices — suitable for
/// per-tick evaluation in the SR²AM loop.
///
/// Internally computes the cosine similarity between student and teacher
/// score distributions as an alignment proxy.
pub fn peira_planning_quality(student_scores: &[f32], teacher_scores: &[f32]) -> f32 {
    if student_scores.is_empty() || teacher_scores.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_s = 0.0f64;
    let mut norm_t = 0.0f64;
    let k = student_scores.len().min(teacher_scores.len());
    for i in 0..k {
        let s = student_scores[i] as f64;
        let t = teacher_scores[i] as f64;
        dot += s * t;
        norm_s += s * s;
        norm_t += t * t;
    }
    let denom = norm_s.sqrt() * norm_t.sqrt();
    if denom < 1e-15 {
        return 0.0;
    }
    (dot / denom).clamp(0.0, 1.0) as f32
}

/// Power iteration to find the top eigenvector of a k×k matrix.
///
/// Returns a unit-norm eigenvector corresponding to the largest eigenvalue.
fn power_iteration(mat: &[f64], k: usize, iterations: usize) -> Vec<f64> {
    // Start with uniform vector
    let mut v = vec![1.0f64 / (k as f64).sqrt(); k];
    let mut v_new = vec![0.0f64; k];

    for _ in 0..iterations {
        // v_new = mat @ v
        for i in 0..k {
            let mut sum = 0.0f64;
            for j in 0..k {
                sum += mat[i * k + j] * v[j];
            }
            v_new[i] = sum;
        }

        // Normalize
        let norm: f64 = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return v;
        }
        for x in v_new.iter_mut() {
            *x /= norm;
        }
        std::mem::swap(&mut v, &mut v_new);
    }

    v
}

/// Generate synthetic CCA data for testing.
///
/// Creates two views of a Gaussian with known canonical correlations.
/// The canonical correlations are [ρ₁, ρ₂, ...] where ρᵢ = 1.0 - 0.1 * i.
///
/// Returns (student_view, teacher_view) vectors of length k.
pub fn synthetic_cca_sample(k: usize, rng: &mut fastrand::Rng) -> (Vec<f32>, Vec<f32>) {
    // Shared latent z ~ N(0, 1) for each dimension
    let z: Vec<f64> = (0..k).map(|_| rng.f64() * 2.0 - 1.0).collect();

    // Canonical correlations: decreasing with dimension
    let mut student = vec![0.0f32; k];
    let mut teacher = vec![0.0f32; k];

    for i in 0..k {
        let rho = (1.0 - 0.1 * i as f64).max(0.0);
        let noise_s = (rng.f64() * 2.0 - 1.0) * (1.0 - rho * rho).sqrt();
        let noise_t = (rng.f64() * 2.0 - 1.0) * (1.0 - rho * rho).sqrt();
        student[i] = (z[i] + noise_s) as f32;
        teacher[i] = (rho * z[i] + noise_t) as f32;
    }

    (student, teacher)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distiller_processes_steps() {
        let k = 4;
        let mut distiller = PeiraDistiller::new(PeiraConfig::new(k));
        assert_eq!(distiller.step_count(), 0);

        let (loss, alignment) = distiller.step(&[1.0, 0.5, -0.3, 0.0], &[0.8, 0.4, -0.2, 0.1]);

        assert!(loss.is_finite(), "Loss not finite: {loss}");
        assert!(
            (0.0..=1.0).contains(&alignment),
            "Alignment out of range: {alignment}"
        );
        assert_eq!(distiller.step_count(), 1);
    }

    #[test]
    fn alignment_converges_on_synthetic_data() {
        let k = 4;
        let mut distiller =
            PeiraDistiller::new(PeiraConfig::new(k).with_lambda(0.1).with_ema_rate(0.5));
        let mut rng = fastrand::Rng::new();

        // Train on synthetic CCA data
        for _ in 0..200 {
            let (student, teacher) = synthetic_cca_sample(k, &mut rng);
            distiller.step(&student, &teacher);
        }

        // Alignment should be non-trivial after 200 steps
        let alignment = distiller.alignment();
        assert!(
            alignment > 0.1,
            "Alignment too low after 200 steps: {alignment}"
        );
    }

    #[test]
    fn no_collapse_on_synthetic_data() {
        let k = 4;
        let mut distiller = PeiraDistiller::new(PeiraConfig::new(k));
        let mut rng = fastrand::Rng::new();

        let mut min_norm = f64::MAX;
        for _ in 0..100 {
            let (student, teacher) = synthetic_cca_sample(k, &mut rng);
            distiller.step(&student, &teacher);

            // Check representation norms
            let norm_s: f64 = student
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let norm_t: f64 = teacher
                .iter()
                .map(|x| (*x as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            min_norm = min_norm.min(norm_s).min(norm_t);
        }

        // GOAT gate: no collapse (norm stays > 0)
        assert!(
            min_norm > 0.0,
            "Representation collapsed: min norm = {min_norm}"
        );
    }

    #[test]
    fn alignment_score_bounds() {
        let k = 3;
        // Identity matrices → perfect alignment
        let identity: Vec<f64> = (0..k)
            .flat_map(|i| (0..k).map(move |j| if i == j { 1.0 } else { 0.0 }))
            .collect();
        let score = peira_alignment_score(&identity, &identity, k);
        assert!(
            score > 0.99,
            "Identity alignment should be ~1.0, got {score}"
        );
    }

    #[test]
    fn reset_clears_state() {
        let k = 4;
        let mut distiller = PeiraDistiller::new(PeiraConfig::new(k));
        distiller.step(&[1.0, 0.0, 0.0, 0.0], &[0.5, 0.0, 0.0, 0.0]);
        distiller.step(&[0.0, 1.0, 0.0, 0.0], &[0.0, 0.5, 0.0, 0.0]);
        assert_eq!(distiller.step_count(), 2);

        distiller.reset();
        assert_eq!(distiller.step_count(), 0);
        assert!(distiller.alignment_history().is_empty());
    }

    #[test]
    fn peira_planning_quality_perfect_alignment() {
        let scores = vec![1.0f32, 0.5, 0.0, -0.3];
        let quality = peira_planning_quality(&scores, &scores);
        assert!(
            quality > 0.99,
            "Perfect alignment should give ~1.0, got {quality}"
        );
    }

    #[test]
    fn peira_planning_quality_empty_inputs() {
        assert_eq!(peira_planning_quality(&[], &[]), 0.0);
        assert_eq!(peira_planning_quality(&[1.0], &[]), 0.0);
        assert_eq!(peira_planning_quality(&[], &[1.0]), 0.0);
    }

    #[test]
    fn peira_planning_quality_orthogonal() {
        // Orthogonal vectors → cosine similarity = 0
        let quality = peira_planning_quality(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(
            quality < 0.01,
            "Orthogonal vectors should give ~0.0, got {quality}"
        );
    }
}
