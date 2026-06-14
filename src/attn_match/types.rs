//! Configuration and result types for Attention Matching compaction.

use crate::attn_match::{DEFAULT_W_LOWER, DEFAULT_W_UPPER};

/// Method for aggregating per-query attention scores into a per-key importance score.
///
/// Paper Appendix F.1 compares mean / RMS / max aggregation. RMS is the default
/// because it offers the best balance (slightly more robust than mean, less
/// noisy than max).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ScoreMethod {
    /// Arithmetic mean of per-query attention weights.
    Mean = 0,
    /// Root mean square — `sqrt(mean(a_ij^2))` — default, most robust.
    Rms = 1,
    /// Maximum per-query attention weight.
    Max = 2,
}

impl Default for ScoreMethod {
    #[inline]
    fn default() -> Self {
        Self::Rms
    }
}

/// Which key selector to use for choosing `Ck ⊂ K`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KeySelector {
    /// Top-t keys by aggregated attention score (fastest).
    HighestAttnKeys = 0,
    /// Orthogonal Matching Pursuit on the mass feature matrix (best quality).
    Omp = 1,
    /// OMP with k=4 keys per iteration, NNLS refit every τ=2 iterations (4-8× faster).
    OmpFast = 2,
}

impl Default for KeySelector {
    #[inline]
    fn default() -> Self {
        // OMP-fast is the paper's recommended default for production use.
        Self::OmpFast
    }
}

/// Least-squares solver for the value-fitting step.
///
/// Paper Appendix C.2 ranks quality: `lstsq > cholesky > pinv`.
/// We default to Cholesky (fastest in pure Rust, no LAPACK dependency)
/// with a jitter fallback for rank-deficient cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SolverChoice {
    /// Cholesky decomposition of `X^T X` with diagonal jitter fallback.
    /// Fastest in Rust, no external LAPACK required.
    Cholesky = 0,
    /// Pseudoinverse via SVD — most robust but slowest.
    Pinv = 1,
}

impl Default for SolverChoice {
    #[inline]
    fn default() -> Self {
        Self::Cholesky
    }
}

/// Top-level configuration for an Attention Matching compaction run.
#[derive(Debug, Clone)]
pub struct AmConfig {
    /// Number of compacted tokens `t` to retain.
    pub compact_size: usize,

    /// Which key selector to use.
    pub selector: KeySelector,

    /// Attention score aggregation method for `HighestAttnKeys`.
    pub score_method: ScoreMethod,

    /// OMP: keys selected per greedy iteration (paper `k`, default 4 for fast variant).
    pub omp_keys_per_iter: usize,

    /// OMP: NNLS refit interval (paper `τ`, default 2 for fast variant).
    pub omp_refit_interval: usize,

    /// NNLS projected-gradient descent iteration count for β fitting.
    /// 0 = closed-form clamped least squares (paper default for OMP).
    pub nnls_iters: usize,

    /// Lower bound on `w = exp(β)` (default `e^-3` for HighestAttn, `e^-6.9` for OMP).
    pub w_lower: f32,

    /// Upper bound on `w = exp(β)` (default `e^3` for HighestAttn, `e^7` for OMP).
    pub w_upper: f32,

    /// Power-iteration steps for estimating `L ≈ ||M||²` in NNLS step size.
    pub power_iter_steps: usize,

    /// Solver for the Cv least-squares problem.
    pub cv_solver: SolverChoice,

    /// Ridge regularization λ for Cv fitting (default 0 — paper found it hurts).
    pub cv_ridge_lambda: f32,

    /// Diagonal jitter added to `X^T X` if Cholesky fails (rank-deficient).
    pub cholesky_jitter: f32,

    /// Whether to compute the reconstruction report (small overhead).
    pub report_reconstruction: bool,
}

impl AmConfig {
    /// Build a config for the `HighestAttnKeys` selector with paper defaults.
    pub fn highest_attn(compact_size: usize) -> Self {
        Self {
            compact_size,
            selector: KeySelector::HighestAttnKeys,
            score_method: ScoreMethod::Rms,
            omp_keys_per_iter: 1,
            omp_refit_interval: 1,
            nnls_iters: 2, // paper: 2 projected-GD iters for HighestAttnKeys
            w_lower: 1e-3, // e^-3 ≈ 0.0498 — but paper uses e^-3 for β; we use w = e^β so e^-3 maps to β=-3
            w_upper: 20.0855, // e^3
            power_iter_steps: 3,
            cv_solver: SolverChoice::Cholesky,
            cv_ridge_lambda: 0.0,
            cholesky_jitter: 1e-6,
            report_reconstruction: true,
        }
    }

    /// Build a config for the `OMP` selector with paper defaults.
    pub fn omp(compact_size: usize) -> Self {
        Self {
            compact_size,
            selector: KeySelector::Omp,
            score_method: ScoreMethod::Rms,
            omp_keys_per_iter: 1,
            omp_refit_interval: 1,
            nnls_iters: 0, // paper: 0 iters for OMP (closed-form clamped LS)
            w_lower: DEFAULT_W_LOWER,
            w_upper: DEFAULT_W_UPPER,
            power_iter_steps: 3,
            cv_solver: SolverChoice::Cholesky,
            cv_ridge_lambda: 0.0,
            cholesky_jitter: 1e-6,
            report_reconstruction: true,
        }
    }

    /// Build a config for the `OMP-fast` variant (k=4, τ=2).
    pub fn omp_fast(compact_size: usize) -> Self {
        let mut cfg = Self::omp(compact_size);
        cfg.selector = KeySelector::OmpFast;
        cfg.omp_keys_per_iter = 4;
        cfg.omp_refit_interval = 2;
        cfg
    }

    /// Validate the configuration; returns a human-readable error on failure.
    pub fn validate(&self, original_len: usize) -> Result<(), String> {
        if self.compact_size == 0 {
            return Err("compact_size must be > 0".into());
        }
        if self.compact_size >= original_len {
            return Err(format!(
                "compact_size ({}) must be < original_len ({}) — no compaction needed",
                self.compact_size, original_len
            ));
        }
        if self.omp_keys_per_iter == 0 {
            return Err("omp_keys_per_iter must be > 0".into());
        }
        if self.omp_refit_interval == 0 {
            return Err("omp_refit_interval must be > 0".into());
        }
        if !(self.w_lower > 0.0) {
            return Err("w_lower must be > 0".into());
        }
        if self.w_upper <= self.w_lower {
            return Err("w_upper must be > w_lower".into());
        }
        Ok(())
    }
}

impl Default for AmConfig {
    #[inline]
    fn default() -> Self {
        // OMP-fast is the paper's recommended production default.
        Self::omp_fast(64)
    }
}

/// Reconstruction-quality report (computed when `report_reconstruction = true`).
#[derive(Debug, Clone, Default)]
pub struct ReconstructionReport {
    /// Relative Frobenius error of attention output reconstruction:
    /// `||X·Cv − Y||_F / ||Y||_F`.
    pub relative_attn_output_error: f32,

    /// Relative error of attention mass reconstruction:
    /// `||A·w − m||_2 / ||m||_2`.
    pub relative_mass_error: f32,

    /// Coverage: fraction of total RMS attention mass captured by selected keys.
    /// (HighestAttnKeys GOAT G4: should exceed 0.8.)
    pub selected_mass_coverage: f32,
}

/// Result of an Attention Matching compaction.
#[derive(Debug, Clone)]
pub struct AmResult {
    /// Selected key indices into the original `K` (length `t`).
    pub selected_indices: Vec<usize>,

    /// Compact keys `Ck = K[selected_indices]` — flat `t * d` f32.
    pub compact_keys: Vec<f32>,

    /// Per-token additive attention bias β (length `t`).
    /// These are `β = log(w)` where `w` solves the NNLS mass-matching problem.
    pub beta: Vec<f32>,

    /// Compact values `Cv` fitted via least squares — flat `t * d` f32.
    pub compact_values: Vec<f32>,

    /// Original sequence length `T`.
    pub original_len: usize,

    /// Compact sequence length `t`.
    pub compact_len: usize,

    /// Head dimension `d`.
    pub head_dim: usize,

    /// Reconstruction report (populated iff `config.report_reconstruction`).
    pub report: Option<ReconstructionReport>,
}

impl AmResult {
    /// Compression ratio `T / t`.
    #[inline]
    pub fn compression_ratio(&self) -> f32 {
        if self.compact_len == 0 {
            return 1.0;
        }
        self.original_len as f32 / self.compact_len as f32
    }

    /// Total bytes saved by compaction (assuming f32 storage).
    #[inline]
    pub fn bytes_saved(&self) -> usize {
        let original_bytes = self.original_len * self.head_dim * 2 * std::mem::size_of::<f32>();
        let compact_bytes = self.compact_len * (self.head_dim * 2 + 1) * std::mem::size_of::<f32>();
        original_bytes.saturating_sub(compact_bytes)
    }
}

#[cfg(test)]
mod type_tests {
    use super::*;

    #[test]
    fn test_config_validate_ok() {
        let cfg = AmConfig::highest_attn(32);
        assert!(cfg.validate(64).is_ok());
    }

    #[test]
    fn test_config_validate_zero_compact() {
        let mut cfg = AmConfig::highest_attn(0);
        cfg.compact_size = 0;
        assert!(cfg.validate(64).is_err());
    }

    #[test]
    fn test_config_validate_no_op() {
        let cfg = AmConfig::highest_attn(64);
        assert!(cfg.validate(64).is_err()); // compact == original — no compaction
    }

    #[test]
    fn test_compression_ratio() {
        let r = AmResult {
            selected_indices: vec![0],
            compact_keys: vec![0.0],
            beta: vec![0.0],
            compact_values: vec![0.0],
            original_len: 100,
            compact_len: 10,
            head_dim: 1,
            report: None,
        };
        assert!((r.compression_ratio() - 10.0).abs() < 1e-6);
    }
}
