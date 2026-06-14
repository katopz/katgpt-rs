//! Top-level Attention Matching compaction orchestrator.
//!
//! Wires together the three AM stages into a single call:
//! 1. Select compact keys `Ck` (via [`KeySelector`])
//! 2. Fit per-token bias `β` via NNLS ([`fit_beta_nnls`])
//! 3. Fit compact values `Cv` via least squares ([`fit_cv_least_squares`])
//!
//! Returns an [`AmResult`] containing the compact `(Ck, β, Cv)` along with
//! a [`ReconstructionReport`] if requested.

use crate::attn_match::{
    beta_fitter::{fit_beta_nnls, BetaFitConfig},
    key_selection::{highest_attn::select_highest_attn_keys, omp::select_omp_keys, KeySelection},
    score_matrix::{compute_score_matrix, compute_softmax_attention},
    types::{AmConfig, AmResult, KeySelector, ReconstructionReport},
    value_fitter::{compute_compact_attention, fit_cv_least_squares, ValueFitConfig},
};

/// Error returned by [`compact`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactError {
    /// Invalid configuration (e.g., `t >= T`).
    InvalidConfig(String),
    /// Dimension mismatch between keys, values, or queries.
    DimensionMismatch(String),
}

impl std::fmt::Display for CompactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(s) => write!(f, "invalid config: {}", s),
            Self::DimensionMismatch(s) => write!(f, "dimension mismatch: {}", s),
        }
    }
}

impl std::error::Error for CompactError {}

/// Output of a successful compaction (alias for [`AmResult`] for API symmetry).
pub type CompactOutput = AmResult;

/// Run Attention Matching compaction.
///
/// # Arguments
/// * `keys` - Original `(T, d)` key matrix, flat row-major.
/// * `values` - Original `(T, d)` value matrix, flat row-major.
/// * `queries` - Reference queries `(n, d)`, flat row-major.
/// * `t_len` - Original sequence length `T`.
/// * `d` - Head dimension.
/// * `n` - Number of reference queries.
/// * `config` - Compaction configuration.
pub fn compact(
    keys: &[f32],
    values: &[f32],
    queries: &[f32],
    t_len: usize,
    d: usize,
    n: usize,
    config: &AmConfig,
) -> Result<CompactOutput, CompactError> {
    // Validate.
    config.validate(t_len).map_err(CompactError::InvalidConfig)?;
    if keys.len() != t_len * d {
        return Err(CompactError::DimensionMismatch(format!(
            "keys.len()={} but T*d={}*{}={}",
            keys.len(),
            t_len,
            d,
            t_len * d
        )));
    }
    if values.len() != t_len * d {
        return Err(CompactError::DimensionMismatch(format!(
            "values.len()={} but T*d={}",
            values.len(),
            t_len * d
        )));
    }
    if queries.len() != n * d {
        return Err(CompactError::DimensionMismatch(format!(
            "queries.len()={} but n*d={}",
            queries.len(),
            n * d
        )));
    }

    let t = config.compact_size;

    // Stage 1: Select compact keys Ck and (optionally) initial weights.
    let selection: KeySelection = match config.selector {
        KeySelector::HighestAttnKeys => {
            let mut s1 = Vec::new();
            let mut s2 = Vec::new();
            select_highest_attn_keys(
                keys,
                queries,
                t,
                config.score_method,
                t_len,
                d,
                n,
                &mut s1,
                &mut s2,
            )
        }
        KeySelector::Omp | KeySelector::OmpFast => select_omp_keys(
            keys,
            queries,
            t,
            config.omp_keys_per_iter,
            config.omp_refit_interval,
            t_len,
            d,
            n,
            config.w_lower,
            config.w_upper,
        ),
    };

    // Extract Ck = K[selection.indices].
    let selected_indices = selection.indices.clone();
    let compact_keys: Vec<f32> = selected_indices
        .iter()
        .flat_map(|&idx| keys[idx * d..(idx + 1) * d].iter().copied())
        .collect();

    // Stage 2: Fit β via NNLS on the selected subset.
    // Build mass feature matrix A ∈ R^{n×t}: A_ij = exp(q_i (Ck)_j^T / √d).
    // And target m ∈ R^n: m_i = Σ_k exp(q_i K_k^T / √d).
    let mut a_mass = vec![0.0f32; n * t];
    let inv_sqrt_d = 1.0f32 / (d as f32).sqrt();
    for i in 0..n {
        let q_row = &queries[i * d..(i + 1) * d];
        let a_row = &mut a_mass[i * t..(i + 1) * t];
        for j in 0..t {
            let ck_row = &compact_keys[j * d..(j + 1) * d];
            let mut dot = 0.0f32;
            for k in 0..d {
                dot += q_row[k] * ck_row[k];
            }
            a_row[j] = (dot * inv_sqrt_d).max(-50.0).exp(); // clamp for stability
        }
    }
    // Target mass m: compute from full K.
    let mut full_scores = vec![0.0f32; n * t_len];
    compute_score_matrix(queries, keys, n, t_len, d, &mut full_scores);
    let mut full_attn = vec![0.0f32; n * t_len];
    let mut m_target = vec![0.0f32; n];
    compute_softmax_attention(
        &full_scores,
        n,
        t_len,
        &mut full_attn,
        &mut m_target,
    );

    // Fit β. For OMP we already have weights from selection; we re-fit here to
    // also produce a relative error estimate.
    let beta_cfg = BetaFitConfig {
        iters: config.nnls_iters,
        w_lower: config.w_lower,
        w_upper: config.w_upper,
        power_iter_steps: config.power_iter_steps,
    };
    let beta_result = fit_beta_nnls(&a_mass, &m_target, n, t, &beta_cfg);
    let beta = beta_result.beta.clone();
    let weights = beta_result.weights.clone();
    let relative_mass_error = beta_result.relative_error;

    // Stage 3: Fit Cv via least squares.
    // Build X ∈ R^{n×t}: X_i = softmax((q_i Ck^T + β) / √d).
    let mut x_attn = vec![0.0f32; n * t];
    compute_compact_attention(queries, &compact_keys, &beta, n, t, d, &mut x_attn);

    // Build Y ∈ R^{n×d}: Y_i = softmax(q_i K^T / √d) V = full_attn[i] · V.
    let mut y_target = vec![0.0f32; n * d];
    for i in 0..n {
        let attn_row = &full_attn[i * t_len..(i + 1) * t_len];
        let y_row = &mut y_target[i * d..(i + 1) * d];
        for k in 0..d {
            let mut s = 0.0f32;
            for j in 0..t_len {
                s += attn_row[j] * values[j * d + k];
            }
            y_row[k] = s;
        }
    }

    let cv_cfg = ValueFitConfig {
        ridge_lambda: config.cv_ridge_lambda,
        cholesky_jitter: config.cholesky_jitter,
    };
    let cv_result = fit_cv_least_squares(&x_attn, &y_target, n, t, d, &cv_cfg);
    let compact_values = cv_result.compact_values;
    let relative_attn_output_error = cv_result.relative_error;

    // Optional reconstruction report.
    let report = if config.report_reconstruction {
        // Compute selected_mass_coverage: fraction of total RMS attention mass
        // captured by selected keys.
        let mut sel_mass_sq = 0.0f32;
        let mut tot_mass_sq = 0.0f32;
        for j in 0..t_len {
            let mut sum_sq = 0.0f32;
            for i in 0..n {
                let a = full_attn[i * t_len + j];
                sum_sq += a * a;
            }
            let rms = (sum_sq / (n as f32)).sqrt();
            tot_mass_sq += rms * rms;
            if selected_indices.contains(&j) {
                sel_mass_sq += rms * rms;
            }
        }
        let selected_mass_coverage = if tot_mass_sq > 0.0 {
            (sel_mass_sq / tot_mass_sq).sqrt()
        } else {
            0.0
        };
        Some(ReconstructionReport {
            relative_attn_output_error,
            relative_mass_error,
            selected_mass_coverage,
        })
    } else {
        None
    };

    // Use weights from β fit (matches paper).
    let _ = weights; // already used via beta

    Ok(AmResult {
        selected_indices,
        compact_keys,
        beta,
        compact_values,
        original_len: t_len,
        compact_len: t,
        head_dim: d,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_kv(t_len: usize, d: usize, seed: usize) -> (Vec<f32>, Vec<f32>) {
        let mut keys = vec![0.0f32; t_len * d];
        let mut values = vec![0.0f32; t_len * d];
        for i in 0..t_len {
            for k in 0..d {
                let x = ((i + seed) as f32) * 0.1 + (k as f32) * 0.01;
                keys[i * d + k] = x.sin() * 0.5;
                values[i * d + k] = x.cos() * 0.3;
            }
        }
        (keys, values)
    }

    fn synth_queries(n: usize, d: usize, seed: usize) -> Vec<f32> {
        let mut q = vec![0.0f32; n * d];
        for i in 0..n {
            for k in 0..d {
                let x = ((i + seed + 100) as f32) * 0.2 + (k as f32) * 0.05;
                q[i * d + k] = x.sin() * 0.4;
            }
        }
        q
    }

    #[test]
    fn test_compact_highest_attn() {
        let (keys, values) = synth_kv(32, 8, 1);
        let queries = synth_queries(4, 8, 1);
        let cfg = AmConfig::highest_attn(8);
        let result = compact(&keys, &values, &queries, 32, 8, 4, &cfg).expect("compact ok");
        assert_eq!(result.compact_len, 8);
        assert_eq!(result.original_len, 32);
        assert_eq!(result.head_dim, 8);
        assert_eq!(result.compact_keys.len(), 8 * 8);
        assert_eq!(result.compact_values.len(), 8 * 8);
        assert_eq!(result.beta.len(), 8);
        assert_eq!(result.selected_indices.len(), 8);
        let report = result.report.expect("report should be present");
        // β should be finite.
        for &b in &result.beta {
            assert!(b.is_finite(), "beta contains non-finite value");
        }
        let _ = report; // silence unused warning
    }

    #[test]
    fn test_compact_omp() {
        let (keys, values) = synth_kv(24, 4, 2);
        let queries = synth_queries(3, 4, 2);
        let cfg = AmConfig::omp(6);
        let result = compact(&keys, &values, &queries, 24, 4, 3, &cfg).expect("compact ok");
        assert_eq!(result.compact_len, 6);
        // OMP weights produce finite β.
        for &b in &result.beta {
            assert!(b.is_finite());
        }
    }

    #[test]
    fn test_compact_omp_fast() {
        let (keys, values) = synth_kv(40, 6, 3);
        let queries = synth_queries(5, 6, 3);
        let cfg = AmConfig::omp_fast(10);
        let result = compact(&keys, &values, &queries, 40, 6, 5, &cfg).expect("compact ok");
        assert_eq!(result.compact_len, 10);
    }

    #[test]
    fn test_compact_invalid_config() {
        let (keys, values) = synth_kv(8, 4, 1);
        let queries = synth_queries(2, 4, 1);
        let mut cfg = AmConfig::highest_attn(8);
        cfg.compact_size = 8; // equal to T → invalid
        let err = compact(&keys, &values, &queries, 8, 4, 2, &cfg).unwrap_err();
        assert!(matches!(err, CompactError::InvalidConfig(_)));
    }

    #[test]
    fn test_compact_dim_mismatch() {
        let (keys, _values) = synth_kv(8, 4, 1);
        let values = vec![0.0f32; 7 * 4]; // wrong size
        let queries = synth_queries(2, 4, 1);
        let cfg = AmConfig::highest_attn(4);
        let err = compact(&keys, &values, &queries, 8, 4, 2, &cfg).unwrap_err();
        assert!(matches!(err, CompactError::DimensionMismatch(_)));
    }

    #[test]
    fn test_compact_compression_ratio() {
        let (keys, values) = synth_kv(64, 8, 5);
        let queries = synth_queries(8, 8, 5);
        let cfg = AmConfig::omp_fast(8);
        let result = compact(&keys, &values, &queries, 64, 8, 8, &cfg).expect("compact ok");
        assert!((result.compression_ratio() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_compact_deterministic() {
        let (keys, values) = synth_kv(32, 4, 7);
        let queries = synth_queries(4, 4, 7);
        let cfg = AmConfig::omp(4);
        let r1 = compact(&keys, &values, &queries, 32, 4, 4, &cfg).expect("ok");
        let r2 = compact(&keys, &values, &queries, 32, 4, 4, &cfg).expect("ok");
        // Same input → same output (determinism, GOAT G-prereq).
        assert_eq!(r1.selected_indices, r2.selected_indices);
        for j in 0..r1.beta.len() {
            assert!((r1.beta[j] - r2.beta[j]).abs() < 1e-6);
        }
        for j in 0..r1.compact_values.len() {
            assert!((r1.compact_values[j] - r2.compact_values[j]).abs() < 1e-6);
        }
    }
}
