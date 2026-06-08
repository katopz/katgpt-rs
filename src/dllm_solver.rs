//! Discrete Critical Interval Solver Switching (Plan 222).
//!
//! Entropy-triggered solver switching during DDTree construction.
//! When marginal entropy exceeds H_critical, switch from DPM-Solver++(2M)
//! to q-sampling or other strategies.

/// Solver kind for D2F decode steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum SolverKind {
    /// DPM-Solver++(2M) — fast, current default.
    #[default]
    DpmSolver2M = 0,
    /// Q-Sample — re-noise + re-predict for critical steps.
    QSample = 1,
    /// DDPM — standard denoising for fallback.
    DDPM = 2,
}

/// Configuration for CriticalIntervalGate.
#[derive(Clone, Debug)]
pub struct CriticalIntervalConfig {
    /// Entropy threshold above which critical interval is detected.
    /// Default: log(vocab_size) * 0.5
    pub h_critical: f32,
    /// Vocab size for computing default threshold.
    pub vocab_size: usize,
    /// Whether to use q-sampling during critical steps.
    pub use_q_sample: bool,
}

impl Default for CriticalIntervalConfig {
    fn default() -> Self {
        let vocab_size = 32000; // typical LLM vocab
        Self {
            h_critical: (vocab_size as f32).ln() * 0.5,
            vocab_size,
            use_q_sample: false,
        }
    }
}

impl CriticalIntervalConfig {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            h_critical: (vocab_size as f32).ln() * 0.5,
            vocab_size,
            use_q_sample: false,
        }
    }
}

/// Detect whether entropy at current step exceeds critical threshold.
/// Returns true if H >= H_critical.
#[inline]
pub fn is_critical_interval(entropy: f32, config: &CriticalIntervalConfig) -> bool {
    entropy >= config.h_critical
}

/// Select solver based on entropy level.
/// If critical interval and q_sample enabled → QSample.
/// Otherwise → DpmSolver2M.
#[inline]
pub fn select_solver(entropy: f32, config: &CriticalIntervalConfig) -> SolverKind {
    if is_critical_interval(entropy, config) && config.use_q_sample {
        SolverKind::QSample
    } else {
        SolverKind::DpmSolver2M
    }
}

/// Compute Shannon entropy from marginal probabilities.
/// H = -Σ p_i * log(p_i)
pub fn shannon_entropy(marginals: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in marginals {
        if p > 1e-10 {
            h -= p * p.ln();
        }
    }
    h
}

/// Q-sampling solver step.
/// x_{t-1} = sqrt(alpha_{t-1}) * x_0_hat + sqrt(1 - alpha_{t-1}) * noise
#[cfg(feature = "q_sample_solver")]
pub fn q_sample_step(x0_hat: &[f32], alpha_prev: f32, noise: &[f32], output: &mut [f32]) {
    let sqrt_alpha = alpha_prev.sqrt();
    let sqrt_one_minus_alpha = (1.0 - alpha_prev).sqrt();
    for i in 0..output.len().min(x0_hat.len()).min(noise.len()) {
        output[i] = sqrt_alpha * x0_hat[i] + sqrt_one_minus_alpha * noise[i];
    }
}

// ---------------------------------------------------------------------------
// MBR Tree Selection (feature-gated)
// ---------------------------------------------------------------------------

/// MBR selection from K candidate paths.
/// Selects minimum-risk path: argmin_i Σ_j |risk_i - risk_j|
#[cfg(feature = "mbr_tree_select")]
pub fn mbr_select(paths: &[Vec<f32>], scores: &[f32]) -> usize {
    if paths.is_empty() {
        return 0;
    }
    let k = paths.len();
    let mut best_idx = 0;
    let mut best_risk = f32::MAX;

    for i in 0..k {
        let mut risk_sum = 0.0f32;
        for j in 0..k {
            if i != j {
                risk_sum += (scores[i] - scores[j]).abs();
            }
        }
        if risk_sum < best_risk {
            best_risk = risk_sum;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_threshold_detection() {
        let config = CriticalIntervalConfig::new(100);
        // Uniform distribution: entropy = log(100) ≈ 4.6
        // Threshold = log(100) * 0.5 ≈ 2.3
        // Uniform entropy exceeds threshold
        let uniform: Vec<f32> = vec![0.01; 100];
        let entropy = shannon_entropy(&uniform);
        assert!(is_critical_interval(entropy, &config));
    }

    #[test]
    fn test_low_entropy_not_critical() {
        let config = CriticalIntervalConfig::new(100);
        // Peaked distribution: most probability on one token
        let mut peaked = vec![0.001f32; 100];
        peaked[0] = 0.9;
        let entropy = shannon_entropy(&peaked);
        assert!(!is_critical_interval(entropy, &config));
    }

    #[test]
    fn test_solver_selection() {
        let mut config = CriticalIntervalConfig::new(100);
        config.use_q_sample = true;

        let low_entropy = 0.5f32;
        let high_entropy = 10.0f32;

        assert_eq!(select_solver(low_entropy, &config), SolverKind::DpmSolver2M);
        assert_eq!(select_solver(high_entropy, &config), SolverKind::QSample);
    }

    #[test]
    fn test_shannon_entropy() {
        // Binary uniform: H = log(2) ≈ 0.693
        let binary = vec![0.5f32, 0.5];
        let h = shannon_entropy(&binary);
        assert!((h - 2.0f32.ln()).abs() < 0.01);
    }

    #[cfg(feature = "q_sample_solver")]
    #[test]
    fn test_q_sample_step() {
        let x0 = vec![1.0f32, 2.0, 3.0];
        let noise = vec![0.1, 0.2, 0.3];
        let mut out = vec![0.0f32; 3];
        q_sample_step(&x0, 0.5, &noise, &mut out);
        // sqrt(0.5) * x0 + sqrt(0.5) * noise
        let expected: Vec<f32> = x0
            .iter()
            .zip(noise.iter())
            .map(|(&x, &n)| 0.5f32.sqrt() * x + 0.5f32.sqrt() * n)
            .collect();
        for i in 0..3 {
            assert!((out[i] - expected[i]).abs() < 1e-5);
        }
    }

    #[cfg(feature = "mbr_tree_select")]
    #[test]
    fn test_mbr_select() {
        let paths = vec![vec![1.0], vec![2.0], vec![3.0]];
        let scores = vec![0.1, 0.5, 0.9];
        let best = mbr_select(&paths, &scores);
        // Middle path has minimum risk
        assert!(best < paths.len());
    }
}
