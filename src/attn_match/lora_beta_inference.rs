//! Plan 297 Phase D / Recipe 4 — LoRA β predictor inference path (T-D.4).
//!
//! Provides the inference-time integration of the trained [`LoraBetaPredictor`]
//! as a drop-in replacement for AM's NNLS β fitter inside [`compact`](super::compact).
//!
//! # Two-step API
//!
//! The LoRA predictor operates on **pooled multi-head KV statistics** (80 floats),
//! while [`compact`](super::compact) operates on **single-head** data. The integration
//! is therefore split into two steps:
//!
//! 1. **Compute KV stats** from all heads:
//!    ```ignore
//!    let kv_stats = compute_kv_stats_for_heads(&keys_per_head, &queries_per_head, t_len, d, n)?;
//!    ```
//! 2. **Predict per-head β** and compact each head:
//!    ```ignore
//!    let beta_per_head = predictor.predict(&kv_stats); // [N_HEADS]
//!    for h in 0..N_HEADS {
//!        let result = compact_with_fixed_beta(
//!            &keys_per_head[h], &values_per_head[h], &queries_per_head[h],
//!            t_len, d, n, &config, beta_per_head[h],
//!        )?;
//!    }
//!    ```
//!
//! `compact_with_fixed_beta` is identical to [`compact`](super::compact) except it
//! skips Stage 2 (NNLS β fitting) and uses the given scalar β for all tokens.
//! This is the "drop-in replacement for public NNLS" — the caller swaps the β
//! source (NNLS → LoRA) without changing the downstream stages.
//!
//! # Per-head β vs per-token β
//!
//! NNLS produces a **per-token** β vector of length `t` (the compact size). The
//! LoRA predictor produces a **per-head** β scalar. At inference, the per-head
//! scalar is replicated across all tokens: `beta = [beta_head; t]`. This is a
//! deliberate approximation — the predictor captures the average attention bias
//! per head, not the per-token variation. The GOAT gate G2 (T-D.5) checks this
//! approximation maintains ≥95% downstream accuracy.
//!
//! # Latent vs Raw
//!
//! All data is latent (KV statistics, β values). No sync boundary.

#![cfg(feature = "lora_beta_predictor")]

use crate::attn_match::{
    lora_beta_predictor::{
        LORA_INPUT_DIM, N_HEADS, STATS_PER_HEAD, TOP_K,
    },
    score_matrix::{compute_score_matrix, compute_softmax_attention},
    types::{AmConfig, KeySelector, ReconstructionReport},
    value_fitter::{compute_compact_attention, fit_cv_least_squares, ValueFitConfig},
    CompactError, CompactOutput,
};
use crate::attn_match::key_selection::{highest_attn::select_highest_attn_keys, omp::select_omp_keys, KeySelection};

// ── KV Stats Pooling ───────────────────────────────────────────────

/// Compute the pooled KV statistics (LoRA input) from multi-head data.
///
/// This is the inference-time equivalent of the corpus generator's
/// `pool_kv_to_stats` (T-D.1, riir-data). The algorithm is identical:
///
/// For each head `h`:
/// - `mean_k`: mean of all key entries (flattened across (t, d)).
/// - `var_k`: variance of all key entries.
/// - `top_K_attn`: top-K softmax attention scores over all (t, q) pairs.
///
/// All heads' features are concatenated into a single `LORA_INPUT_DIM`-length
/// vector: `features[h * STATS_PER_HEAD + 0..STATS_PER_HEAD]`.
///
/// # Arguments
///
/// - `per_head_keys`: `[N_HEADS]` slices, each of length `t_len * d`.
/// - `per_head_queries`: `[N_HEADS]` slices, each of length `n * d`.
/// - `t_len`: sequence length.
/// - `d`: head dimension (must match the value used during corpus generation).
/// - `n`: number of queries.
///
/// # Returns
///
/// A `Vec<f32>` of length `LORA_INPUT_DIM` (80).
///
/// # Errors
///
/// Returns [`CompactError::DimensionMismatch`] if the number of heads or the
/// slice lengths don't match the expected dimensions.
pub fn compute_kv_stats_for_heads(
    per_head_keys: &[&[f32]],
    per_head_queries: &[&[f32]],
    t_len: usize,
    d: usize,
    n: usize,
) -> Result<Vec<f32>, CompactError> {
    if per_head_keys.len() != N_HEADS {
        return Err(CompactError::DimensionMismatch(format!(
            "per_head_keys.len()={} but N_HEADS={}",
            per_head_keys.len(),
            N_HEADS
        )));
    }
    if per_head_queries.len() != N_HEADS {
        return Err(CompactError::DimensionMismatch(format!(
            "per_head_queries.len()={} but N_HEADS={}",
            per_head_queries.len(),
            N_HEADS
        )));
    }

    let mut features = vec![0.0f32; LORA_INPUT_DIM];
    let inv_sqrt_d = 1.0 / (d as f32).sqrt();

    for h in 0..N_HEADS {
        let keys_h = per_head_keys[h];
        let queries_h = per_head_queries[h];

        if keys_h.len() != t_len * d {
            return Err(CompactError::DimensionMismatch(format!(
                "head {}: keys.len()={} but t_len*d={}*{}={}",
                h,
                keys_h.len(),
                t_len,
                d,
                t_len * d
            )));
        }
        if queries_h.len() != n * d {
            return Err(CompactError::DimensionMismatch(format!(
                "head {}: queries.len()={} but n*d={}*{}={}",
                h,
                queries_h.len(),
                n,
                d,
                n * d
            )));
        }

        // Per-head mean / variance over ALL key entries.
        let count = keys_h.len();
        let mut sum = 0.0f32;
        for &v in keys_h {
            sum += v;
        }
        let mean = sum / (count.max(1) as f32);
        let mut var_sum = 0.0f32;
        for &v in keys_h {
            let dx = v - mean;
            var_sum += dx * dx;
        }
        let var = var_sum / (count.max(1) as f32);
        features[h * STATS_PER_HEAD + 0] = mean;
        features[h * STATS_PER_HEAD + 1] = var;

        // Per-head top-K attention scores.
        // Logits: keys_h[t,:] · queries_h[q,:] / sqrt(d) for all (t, q).
        let logit_count = t_len * n;
        if logit_count == 0 {
            continue;
        }
        let mut logits: Vec<f32> = Vec::with_capacity(logit_count);
        for q in 0..n {
            let q_row = &queries_h[q * d..(q + 1) * d];
            for t in 0..t_len {
                let k_row = &keys_h[t * d..(t + 1) * d];
                let mut dot = 0.0f32;
                for k in 0..d {
                    dot += k_row[k] * q_row[k];
                }
                logits.push(dot * inv_sqrt_d);
            }
        }

        // Softmax over all (t, q) for this head.
        let max_logit = logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for l in &mut logits {
            *l = (*l - max_logit).exp();
            exp_sum += *l;
        }
        let inv_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };
        for l in &mut logits {
            *l *= inv_sum;
        }

        // Top-K (descending). Pad with 0 if fewer than TOP_K.
        logits.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        for k in 0..TOP_K {
            let v = if k < logits.len() { logits[k] } else { 0.0 };
            features[h * STATS_PER_HEAD + 2 + k] = v;
        }
    }

    Ok(features)
}

// ── Compact with Fixed β ───────────────────────────────────────────

/// Compact a single head with a pre-computed β scalar (LoRA inference path).
///
/// This is identical to [`compact`](super::compact) except it skips Stage 2
/// (NNLS β fitting) and uses `fixed_beta` for all `t` tokens. The per-head β
/// from the LoRA predictor is replicated across all tokens.
///
/// # When to use
///
/// Use this when you have a pre-computed β (e.g. from
/// [`LoraBetaPredictor::predict`](super::lora_beta_predictor::LoraBetaPredictor::predict))
/// and want to skip the NNLS solve. This is the "drop-in replacement for NNLS"
/// integration point for Recipe 4.
///
/// # Stages
///
/// 1. **Key selection** — same as `compact` (highest-attn or OMP).
/// 2. **β assignment** — `beta = [fixed_beta; t]` (skip NNLS).
/// 3. **Cv fitting** — same least-squares fit as `compact`.
///
/// The `relative_mass_error` in the report is set to `NaN` (not computed when
/// skipping NNLS), and `weights` are set to the exponential of the fixed β.
pub fn compact_with_fixed_beta(
    keys: &[f32],
    values: &[f32],
    queries: &[f32],
    t_len: usize,
    d: usize,
    n: usize,
    config: &AmConfig,
    fixed_beta: f32,
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

    // Stage 1: Select compact keys Ck.
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

    let selected_indices = selection.indices.clone();
    let compact_keys: Vec<f32> = selected_indices
        .iter()
        .flat_map(|&idx| keys[idx * d..(idx + 1) * d].iter().copied())
        .collect();

    // Stage 2: Use fixed β (skip NNLS).
    let beta = vec![fixed_beta; t];
    // Weights = exp(β) — matches how NNLS-derived β maps to weights.
    let weights: Vec<f32> = beta.iter().map(|&b| b.exp()).collect();
    let relative_mass_error = f32::NAN; // not computed without NNLS

    // Stage 3: Fit Cv via least squares.
    // Build X ∈ R^{n×t}: X_i = softmax((q_i Ck^T + β) / √d).
    let mut x_attn = vec![0.0f32; n * t];
    compute_compact_attention(queries, &compact_keys, &beta, n, t, d, &mut x_attn);

    // Build full attention for Y target and optional report.
    let mut full_scores = vec![0.0f32; n * t_len];
    compute_score_matrix(queries, keys, n, t_len, d, &mut full_scores);
    let mut full_attn = vec![0.0f32; n * t_len];
    let mut m_target = vec![0.0f32; n];
    compute_softmax_attention(&full_scores, n, t_len, &mut full_attn, &mut m_target);

    // Build Y ∈ R^{n×d}: Y_i = softmax(q_i K^T / √d) V.
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

    let _ = weights;

    Ok(CompactOutput {
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

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attn_match::compact::compact;
    use crate::attn_match::lora_beta_predictor::LoraBetaPredictor;
    use crate::attn_match::types::AmConfig;

    fn synth_kv(t_len: usize, d: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        use std::num::Wrapping;
        let mut state = Wrapping(seed as u32);
        let mut next_f = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state.0 as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let keys: Vec<f32> = (0..t_len * d).map(|_| next_f()).collect();
        let values: Vec<f32> = (0..t_len * d).map(|_| next_f()).collect();
        (keys, values)
    }

    fn synth_queries(n: usize, d: usize, seed: u64) -> Vec<f32> {
        use std::num::Wrapping;
        let mut state = Wrapping(seed as u32);
        let mut next_f = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state.0 as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        (0..n * d).map(|_| next_f()).collect()
    }

    #[test]
    fn kv_stats_correct_length() {
        let d = 8;
        let t_len = 64;
        let n = 4;
        let keys: Vec<Vec<f32>> = (0..N_HEADS).map(|h| synth_kv(t_len, d, h as u64 + 1).0).collect();
        let queries: Vec<Vec<f32>> = (0..N_HEADS).map(|h| synth_queries(n, d, h as u64 + 100).clone()).collect();
        let key_refs: Vec<&[f32]> = keys.iter().map(|v| v.as_slice()).collect();
        let query_refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();

        let stats = compute_kv_stats_for_heads(&key_refs, &query_refs, t_len, d, n).unwrap();
        assert_eq!(stats.len(), LORA_INPUT_DIM);
    }

    #[test]
    fn kv_stats_mean_and_var_are_finite() {
        let d = 8;
        let t_len = 32;
        let n = 4;
        let keys: Vec<Vec<f32>> = (0..N_HEADS).map(|h| synth_kv(t_len, d, h as u64 + 1).0).collect();
        let queries: Vec<Vec<f32>> = (0..N_HEADS).map(|h| synth_queries(n, d, h as u64 + 100).clone()).collect();
        let key_refs: Vec<&[f32]> = keys.iter().map(|v| v.as_slice()).collect();
        let query_refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();

        let stats = compute_kv_stats_for_heads(&key_refs, &query_refs, t_len, d, n).unwrap();
        for h in 0..N_HEADS {
            let mean = stats[h * STATS_PER_HEAD + 0];
            let var = stats[h * STATS_PER_HEAD + 1];
            assert!(mean.is_finite(), "head {h} mean not finite: {mean}");
            assert!(var.is_finite(), "head {h} var not finite: {var}");
            assert!(var >= 0.0, "head {h} var negative: {var}");
        }
    }

    #[test]
    fn kv_stats_topk_sorted_descending() {
        let d = 8;
        let t_len = 64;
        let n = 4;
        let keys: Vec<Vec<f32>> = (0..N_HEADS).map(|h| synth_kv(t_len, d, h as u64 + 1).0).collect();
        let queries: Vec<Vec<f32>> = (0..N_HEADS).map(|h| synth_queries(n, d, h as u64 + 100).clone()).collect();
        let key_refs: Vec<&[f32]> = keys.iter().map(|v| v.as_slice()).collect();
        let query_refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();

        let stats = compute_kv_stats_for_heads(&key_refs, &query_refs, t_len, d, n).unwrap();
        for h in 0..N_HEADS {
            let topk = &stats[h * STATS_PER_HEAD + 2..h * STATS_PER_HEAD + 2 + TOP_K];
            for k in 1..TOP_K {
                assert!(topk[k] <= topk[k - 1] + 1e-6, "head {h} topk not sorted at {k}");
            }
        }
    }

    #[test]
    fn kv_stats_rejects_wrong_head_count() {
        let d = 8;
        let t_len = 32;
        let n = 4;
        let keys = vec![vec![0.0f32; t_len * d]; N_HEADS + 1]; // too many heads
        let queries = vec![vec![0.0f32; n * d]; N_HEADS];
        let key_refs: Vec<&[f32]> = keys.iter().map(|v| v.as_slice()).collect();
        let query_refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();

        assert!(compute_kv_stats_for_heads(&key_refs, &query_refs, t_len, d, n).is_err());
    }

    #[test]
    fn kv_stats_rejects_wrong_key_length() {
        let d = 8;
        let t_len = 32;
        let n = 4;
        let mut keys = vec![vec![0.0f32; t_len * d]; N_HEADS];
        keys[0] = vec![0.0f32; 10]; // wrong length
        let queries = vec![vec![0.0f32; n * d]; N_HEADS];
        let key_refs: Vec<&[f32]> = keys.iter().map(|v| v.as_slice()).collect();
        let query_refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();

        assert!(compute_kv_stats_for_heads(&key_refs, &query_refs, t_len, d, n).is_err());
    }

    #[test]
    fn compact_with_fixed_beta_produces_valid_output() {
        let d = 8;
        let t_len = 64;
        let n = 4;
        let (keys, values) = synth_kv(t_len, d, 42);
        let queries = synth_queries(n, d, 99);
        let cfg = AmConfig::highest_attn(8);

        let result = compact_with_fixed_beta(&keys, &values, &queries, t_len, d, n, &cfg, -1.0).unwrap();
        assert_eq!(result.compact_len, 8);
        assert_eq!(result.beta.len(), 8);
        assert_eq!(result.compact_keys.len(), 8 * 8);
        assert_eq!(result.compact_values.len(), 8 * 8);
        // All β should be the fixed value.
        for &b in &result.beta {
            assert!((b - (-1.0)).abs() < 1e-6, "beta should be -1.0, got {b}");
        }
    }

    #[test]
    fn compact_with_fixed_beta_matches_beta_value() {
        let d = 8;
        let t_len = 32;
        let n = 4;
        let (keys, values) = synth_kv(t_len, d, 7);
        let queries = synth_queries(n, d, 13);
        let cfg = AmConfig::highest_attn(8);

        for beta_val in [0.0, -1.0, -3.0, 1.5] {
            let result = compact_with_fixed_beta(&keys, &values, &queries, t_len, d, n, &cfg, beta_val).unwrap();
            for &b in &result.beta {
                assert!((b - beta_val).abs() < 1e-6, "beta mismatch: expected {beta_val}, got {b}");
            }
        }
    }

    #[test]
    fn compact_with_fixed_beta_finite_output() {
        let d = 8;
        let t_len = 32;
        let n = 4;
        let (keys, values) = synth_kv(t_len, d, 77);
        let queries = synth_queries(n, d, 88);
        let cfg = AmConfig::highest_attn(8);

        let result = compact_with_fixed_beta(&keys, &values, &queries, t_len, d, n, &cfg, -2.0).unwrap();
        for &v in &result.compact_values {
            assert!(v.is_finite(), "compact_values non-finite: {v}");
        }
        for &v in &result.compact_keys {
            assert!(v.is_finite(), "compact_keys non-finite: {v}");
        }
    }

    #[test]
    fn end_to_end_lora_predict_then_compact() {
        // Full pipeline: compute kv_stats → predict β → compact each head.
        let d = 8;
        let t_len = 64;
        let n = 4;

        let keys: Vec<Vec<f32>> = (0..N_HEADS).map(|h| synth_kv(t_len, d, h as u64 + 1).0).collect();
        let values: Vec<Vec<f32>> = (0..N_HEADS).map(|h| synth_kv(t_len, d, h as u64 + 200).1).collect();
        let queries: Vec<Vec<f32>> = (0..N_HEADS).map(|h| synth_queries(n, d, h as u64 + 100)).collect();

        let key_refs: Vec<&[f32]> = keys.iter().map(|v| v.as_slice()).collect();
        let query_refs: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();

        let kv_stats = compute_kv_stats_for_heads(&key_refs, &query_refs, t_len, d, n).unwrap();
        assert_eq!(kv_stats.len(), LORA_INPUT_DIM);

        let predictor = LoraBetaPredictor::new();
        let beta_per_head = predictor.predict(&kv_stats);
        assert_eq!(beta_per_head.len(), N_HEADS);

        // Compact each head.
        let cfg = AmConfig::highest_attn(8);
        for h in 0..N_HEADS {
            let result = compact_with_fixed_beta(
                &keys[h], &values[h], &queries[h],
                t_len, d, n, &cfg, beta_per_head[h],
            ).unwrap();
            assert_eq!(result.compact_len, 8);
            assert_eq!(result.beta.len(), 8);
            for &b in &result.beta {
                assert!((b - beta_per_head[h]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn compact_with_fixed_beta_rejects_bad_dims() {
        let d = 8;
        let t_len = 32;
        let n = 4;
        let (keys, values) = synth_kv(t_len, d, 5);
        let queries = synth_queries(n, d, 10);
        let cfg = AmConfig::highest_attn(8);

        // Wrong keys length.
        assert!(compact_with_fixed_beta(&[0.0; 10], &values, &queries, t_len, d, n, &cfg, 0.0).is_err());
        // Wrong queries length.
        assert!(compact_with_fixed_beta(&keys, &values, &[0.0; 3], t_len, d, n, &cfg, 0.0).is_err());
    }

    #[test]
    fn compact_with_fixed_beta_report_has_nan_mass_error() {
        let d = 8;
        let t_len = 32;
        let n = 4;
        let (keys, values) = synth_kv(t_len, d, 55);
        let queries = synth_queries(n, d, 66);
        let mut cfg = AmConfig::highest_attn(8);
        cfg.report_reconstruction = true;

        let result = compact_with_fixed_beta(&keys, &values, &queries, t_len, d, n, &cfg, -1.0).unwrap();
        let report = result.report.expect("report should be present");
        assert!(report.relative_mass_error.is_nan(), "mass_error should be NaN without NNLS");
    }
}
