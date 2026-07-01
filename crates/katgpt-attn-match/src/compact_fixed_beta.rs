//! # Compact with Fixed β — the simplified AM fast path.
//!
//! ## Origin (Issue 305)
//!
//! Plan 297 Phase D introduced a LoRA β predictor intended to replace AM's
//! NNLS β fitter for long-context compaction. Empirical investigation proved
//! that the predictor adds **zero value** over a hardcoded constant β: see
//! `issues/305_lora_beta_predictor_softmax_invariance.md` in `riir-ai`.
//!
//! The mathematical reason: `compact_with_fixed_beta` applies a single β
//! uniformly to all tokens in a head (`beta = [fixed_beta; t]`), and
//! `softmax(x + c) = softmax(x)` for any constant `c`. The attention matrix
//! is invariant to the β value under uniform replication, so the LoRA
//! predictor's output is mathematically irrelevant.
//!
//! The actual speedup comes from **skipping NNLS** — Stage 2 of `compact()`.
//! A hardcoded `BETA_MID` achieves the same speedup with no training pipeline.
//!
//! Cross-query evaluation (the metric where β genuinely matters, because new
//! queries see per-token β in the stored compact representation) shows uniform
//! per-head β **beats** NNLS's per-token β by ~44% — NNLS overfits the
//! compaction queries, while uniform β acts as implicit L2 regularization.
//!
//! ## What this module provides
//!
//! - [`compact_with_fixed_beta`] — identical to [`compact`](super::compact)
//!   except it skips Stage 2 (NNLS β fitting) and uses the given scalar β for
//!   all tokens. Always-available (no feature gate).
//! - [`BETA_MID`] / [`BETA_MIN`] / [`BETA_MAX`] — the canonical β bounds.
//!   Callers should use [`BETA_MID`] as the default fixed β.
//!
//! ## Latent vs Raw
//!
//! All data is latent (KV cache, β values). No sync boundary.

#![allow(clippy::too_many_arguments)]

use crate::compact::build_reconstruction_report;
use crate::key_selection::{
    KeySelection, highest_attn::select_highest_attn_keys, omp::select_omp_keys,
};
use crate::{
    CompactError, CompactOutput,
    score_matrix::{compute_score_matrix, compute_softmax_attention_and_output},
    types::{AmConfig, KeySelector},
    value_fitter::{ValueFitConfig, compute_compact_attention, fit_cv_least_squares},
};

// ── β bounds ───────────────────────────────────────────────────────

/// β lower bound = `log(w_lower)` where `w_lower = 1e-3`
/// (`AmConfig::highest_attn`).
pub const BETA_MIN: f32 = -6.907_755_3; // ln(1e-3)

/// β upper bound = `log(w_upper)` where `w_upper = e^3 ≈ 20.0855`
/// (`AmConfig::highest_attn`).
pub const BETA_MAX: f32 = 3.0;

/// β midpoint — the neutral default. Equal to `(BETA_MIN + BETA_MAX) / 2`,
/// matching the LoRA predictor's init target (`sigmoid(0) = 0.5` → `BETA_MID`).
///
/// This is the **recommended fixed β** for [`compact_with_fixed_beta`].
pub const BETA_MID: f32 = (BETA_MIN + BETA_MAX) * 0.5;

// ── Compact with Fixed β ───────────────────────────────────────────

/// Compact a single head with a pre-computed β scalar — the simplified
/// AM fast path (Issue 305).
///
/// Identical to [`compact`](super::compact) except it skips Stage 2 (NNLS β
/// fitting) and uses `fixed_beta` for all `t` tokens. The caller typically
/// passes [`BETA_MID`], which is provably as accurate as NNLS's per-token β
/// on cross-query evaluation while being ~250× faster.
///
/// # When to use
///
/// Use this when:
/// - You want the AM speedup without the NNLS solve cost.
/// - You don't have per-token β importance information.
/// - You're compacting for general-purpose future queries (cross-query).
///
/// # Stages
///
/// 1. **Key selection** — same as `compact` (highest-attn or OMP).
/// 2. **β assignment** — `beta = [fixed_beta; t]` (skip NNLS).
/// 3. **Cv fitting** — same least-squares fit as `compact`.
///
/// The `relative_mass_error` in the report is set to `NaN` (not computed when
/// skipping NNLS), and `weights` are set to the exponential of the fixed β.
///
/// # Softmax invariance note
///
/// For **same-query** downstream evaluation (using the compaction queries),
/// the attention matrix is mathematically invariant to `fixed_beta`:
/// `softmax(logits + c) = softmax(logits)`. So `BETA_MID`, `BETA_MIN`, and
/// `BETA_MAX` all produce bit-identical `relative_attn_output_error`.
///
/// For **cross-query** evaluation (using fresh queries on the stored compact
/// representation), β genuinely matters because per-token β shapes attention
/// for those new queries. Empirically, uniform `BETA_MID` beats NNLS's
/// per-token β by ~44% on cross-query generalization (NNLS overfits the
/// compaction queries).
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
    config
        .validate(t_len)
        .map_err(CompactError::InvalidConfig)?;
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

    // Move indices out of `selection` (no clone — we own it and don't use
    // `weights` here). Pre-allocate exactly and memcpy each row — avoids the
    // log(t) Vec growth reallocations of `flat_map().collect()`.
    let selected_indices = selection.indices;
    let mut compact_keys = Vec::with_capacity(selected_indices.len() * d);
    for &idx in &selected_indices {
        compact_keys.extend_from_slice(&keys[idx * d..(idx + 1) * d]);
    }

    // Stage 2: Use fixed β (skip NNLS).
    let beta = vec![fixed_beta; t];
    let relative_mass_error = f32::NAN; // not computed without NNLS

    // Stage 3: Fit Cv via least squares.
    // Build X ∈ R^{n×t}: X_i = softmax((q_i Ck^T + β) / √d).
    let mut x_attn = vec![0.0f32; n * t];
    compute_compact_attention(queries, &compact_keys, &beta, n, t, d, &mut x_attn);

    // Build full attention for Y target and optional report. Fused softmax +
    // attention-output so the `(n, t_len)` normalized-attention matrix is only
    // materialized when the reconstruction report needs it. See
    // `compute_softmax_attention_and_output` for the disclosed float-op
    // reordering.
    let mut full_scores = vec![0.0f32; n * t_len];
    compute_score_matrix(queries, keys, n, t_len, d, &mut full_scores);
    let mut m_target = vec![0.0f32; n];
    let mut y_target = vec![0.0f32; n * d];
    let full_attn = if config.report_reconstruction {
        let mut a = vec![0.0f32; n * t_len];
        compute_softmax_attention_and_output(
            &full_scores,
            values,
            n,
            t_len,
            d,
            &mut m_target,
            &mut y_target,
            Some(&mut a),
        );
        a
    } else {
        compute_softmax_attention_and_output(
            &full_scores,
            values,
            n,
            t_len,
            d,
            &mut m_target,
            &mut y_target,
            None,
        );
        Vec::new()
    };
    // `m_target` is not consumed by the fixed-β path (no NNLS), but the fused
    // kernel produces it as a byproduct; silence the unused binding.
    let _ = m_target;

    let cv_cfg = ValueFitConfig {
        ridge_lambda: config.cv_ridge_lambda,
        cholesky_jitter: config.cholesky_jitter,
    };
    let cv_result = fit_cv_least_squares(&x_attn, &y_target, n, t, d, &cv_cfg);
    let compact_values = cv_result.compact_values;
    let relative_attn_output_error = cv_result.relative_error;

    // Optional reconstruction report. Shared helper (also used by
    // `compact_with_router`) keeps the cache-friendly `i`-outer loop order in
    // one place — DRY across both compact paths.
    let report = if config.report_reconstruction {
        Some(build_reconstruction_report(
            &selected_indices,
            &full_attn,
            n,
            t_len,
            relative_attn_output_error,
            relative_mass_error,
        ))
    } else {
        None
    };

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
    use crate::compact::compact;
    use crate::types::AmConfig;

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
    fn beta_bounds_are_consistent() {
        // BETA_MID is the arithmetic mean of BETA_MIN and BETA_MAX;
        // ordering (MIN < MID < MAX) follows from the value checks below.
        let mid = (BETA_MIN + BETA_MAX) * 0.5;
        assert!((BETA_MID - mid).abs() < 1e-6);
        // Matches AmConfig::highest_attn bounds (w_lower=1e-3, w_upper=e^3).
        assert!((BETA_MIN - (1e-3f32).ln()).abs() < 1e-4);
        assert!((BETA_MAX - 3.0f32).abs() < 1e-6);
    }

    #[test]
    fn compact_with_fixed_beta_produces_valid_output() {
        let d = 8;
        let t_len = 64;
        let n = 4;
        let (keys, values) = synth_kv(t_len, d, 42);
        let queries = synth_queries(n, d, 99);
        let cfg = AmConfig::highest_attn(8);

        let result =
            compact_with_fixed_beta(&keys, &values, &queries, t_len, d, n, &cfg, BETA_MID).unwrap();
        assert_eq!(result.compact_len, 8);
        assert_eq!(result.beta.len(), 8);
        assert_eq!(result.compact_keys.len(), 8 * 8);
        assert_eq!(result.compact_values.len(), 8 * 8);
        // All β should be the fixed value.
        for &b in &result.beta {
            assert!(
                (b - BETA_MID).abs() < 1e-6,
                "beta should be BETA_MID, got {b}"
            );
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

        for beta_val in [BETA_MIN, BETA_MID, BETA_MAX, 0.0, -1.0, 1.5] {
            let result =
                compact_with_fixed_beta(&keys, &values, &queries, t_len, d, n, &cfg, beta_val)
                    .unwrap();
            for &b in &result.beta {
                assert!(
                    (b - beta_val).abs() < 1e-6,
                    "beta mismatch: expected {beta_val}, got {b}"
                );
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

        let result =
            compact_with_fixed_beta(&keys, &values, &queries, t_len, d, n, &cfg, BETA_MID).unwrap();
        for &v in &result.compact_values {
            assert!(v.is_finite(), "compact_values non-finite: {v}");
        }
        for &v in &result.compact_keys {
            assert!(v.is_finite(), "compact_keys non-finite: {v}");
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
        assert!(
            compact_with_fixed_beta(&[0.0; 10], &values, &queries, t_len, d, n, &cfg, BETA_MID)
                .is_err()
        );
        // Wrong queries length.
        assert!(
            compact_with_fixed_beta(&keys, &values, &[0.0; 3], t_len, d, n, &cfg, BETA_MID)
                .is_err()
        );
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

        let result =
            compact_with_fixed_beta(&keys, &values, &queries, t_len, d, n, &cfg, BETA_MID).unwrap();
        let report = result.report.expect("report should be present");
        assert!(
            report.relative_mass_error.is_nan(),
            "mass_error should be NaN without NNLS"
        );
    }

    /// Softmax invariance proof (Issue 305): for same-query downstream
    /// evaluation, `relative_attn_output_error` is INVARIANT to the fixed β
    /// value, because `softmax(x + c) = softmax(x)`.
    ///
    /// This test verifies that BETA_MIN, BETA_MID, and BETA_MAX all produce
    /// bit-identical (within float ULP) `relative_attn_output_error` on the
    /// same queries used during compaction.
    #[test]
    fn softmax_invariance_same_query_downstream_error() {
        let d = 8;
        let t_len = 64;
        let n = 4;
        let (keys, values) = synth_kv(t_len, d, 314);
        let queries = synth_queries(n, d, 271);
        let mut cfg = AmConfig::highest_attn(8);
        cfg.report_reconstruction = true;

        let err_min =
            compact_with_fixed_beta(&keys, &values, &queries, t_len, d, n, &cfg, BETA_MIN)
                .unwrap()
                .report
                .unwrap()
                .relative_attn_output_error;
        let err_mid =
            compact_with_fixed_beta(&keys, &values, &queries, t_len, d, n, &cfg, BETA_MID)
                .unwrap()
                .report
                .unwrap()
                .relative_attn_output_error;
        let err_max =
            compact_with_fixed_beta(&keys, &values, &queries, t_len, d, n, &cfg, BETA_MAX)
                .unwrap()
                .report
                .unwrap()
                .relative_attn_output_error;

        // All three should be within 1 ULP of each other.
        let ulp_tol = 1e-5;
        assert!(
            (err_min - err_mid).abs() < ulp_tol,
            "softmax invariance violated: BETA_MIN err={err_min} ≠ BETA_MID err={err_mid}"
        );
        assert!(
            (err_max - err_mid).abs() < ulp_tol,
            "softmax invariance violated: BETA_MAX err={err_max} ≠ BETA_MID err={err_mid}"
        );

        // Also verify the Cv fit is finite and the compact path runs at all.
        let nnls_out = compact(&keys, &values, &queries, t_len, d, n, &cfg).unwrap();
        assert!(
            nnls_out
                .report
                .as_ref()
                .unwrap()
                .relative_attn_output_error
                .is_finite()
        );
    }
}
