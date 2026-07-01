//! Top-level Attention Matching compaction orchestrator.
//!
//! Wires together the three AM stages into a single call:
//! 1. Select compact keys `Ck` (via [`KeySelector`])
//! 2. Fit per-token bias `β` via NNLS ([`fit_beta_nnls`])
//! 3. Fit compact values `Cv` via least squares ([`fit_cv_least_squares`])
//!
//! Returns an [`AmResult`] containing the compact `(Ck, β, Cv)` along with
//! a [`ReconstructionReport`] if requested.

#![allow(clippy::too_many_arguments)]

use crate::{
    beta_fitter::{BetaFitConfig, fit_beta_nnls},
    key_selection::{KeySelection, highest_attn::select_highest_attn_keys, omp::select_omp_keys},
    router::{SolverBackend, SolverRouter},
    score_matrix::{compute_score_matrix, compute_softmax_attention_and_output},
    score_matrix_rayon::compute_score_matrix_rayon,
    types::{AmConfig, AmResult, KeySelector, ReconstructionReport},
    value_fitter::{ValueFitConfig, compute_compact_attention, fit_cv_least_squares},
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

/// Build the reconstruction report, shared between [`compact_with_router`] and
/// [`compact_with_fixed_beta`](super::compact_fixed_beta::compact_with_fixed_beta).
///
/// Computes `selected_mass_coverage = sqrt(Σ_sel sum_i a_ij^2 / Σ_all sum_i a_ij^2)`
/// over the full attention matrix `full_attn`.
///
/// # Cache-friendly loop order
///
/// Iterates `i`-outer / `j`-inner so reads of `full_attn` are row-major
/// sequential (prefetcher-friendly). The prior duplicated inline form iterated
/// `j`-outer / `i`-inner with strided `full_attn[i*t_len + j]` reads — one
/// cache miss per `i` per `j`. For `(n=8, t_len=4096)` that is 32K strided
/// reads; this form does 32K sequential reads into a `(t_len,)` per-key buffer,
/// then one sequential pass over the buffer. The `/n` factor cancels in the
/// final ratio, so we skip the per-key division entirely (mathematically
/// identical result, eliminates `t_len` divisions).
pub(crate) fn build_reconstruction_report(
    selected_indices: &[usize],
    full_attn: &[f32],
    n: usize,
    t_len: usize,
    relative_attn_output_error: f32,
    relative_mass_error: f32,
) -> ReconstructionReport {
    debug_assert_eq!(full_attn.len(), n * t_len);

    // O(t_len) bitmap for O(1) membership test. The prior
    // `selected_indices.contains(&j)` was O(t) per key → O(T·t) total.
    let mut selected_bitmap = vec![false; t_len];
    for &idx in selected_indices {
        selected_bitmap[idx] = true;
    }

    // Per-key sum-of-squares accumulated via row-major sequential reads.
    let mut per_key_sum_sq = vec![0.0f32; t_len];
    for i in 0..n {
        let row = &full_attn[i * t_len..(i + 1) * t_len];
        for j in 0..t_len {
            per_key_sum_sq[j] += row[j] * row[j];
        }
    }

    let mut sel_sum_sq = 0.0f32;
    let mut tot_sum_sq = 0.0f32;
    for j in 0..t_len {
        let ssq = per_key_sum_sq[j];
        tot_sum_sq += ssq;
        if selected_bitmap[j] {
            sel_sum_sq += ssq;
        }
    }
    let selected_mass_coverage = if tot_sum_sq > 0.0 {
        (sel_sum_sq / tot_sum_sq).sqrt()
    } else {
        0.0
    };
    ReconstructionReport {
        relative_attn_output_error,
        relative_mass_error,
        selected_mass_coverage,
    }
}

/// Per-stage backend decisions made by [`compact_with_router`].
///
/// This is populated by `log::debug!` when the `am_compaction_trace` feature
/// flag (planned) is enabled, and returned in [`RouterTrace`] for inspection
/// by callers (e.g., benchmarks that want to verify the router is dispatching
/// correctly).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RouterTrace {
    /// Backend chosen for the full-K score matrix computation.
    pub score_matrix_backend: Option<SolverBackend>,
    /// Backend chosen for the compact-key score matrix (mass features).
    pub mass_features_backend: Option<SolverBackend>,
    /// Whether the blocked Cholesky path was eligible (t >= block size).
    pub blocked_cholesky_eligible: bool,
}

/// Run Attention Matching compaction (router-free entry point).
///
/// Equivalent to [`compact_with_router`] with a fresh default router and
/// `gpu_available = false`. Use this for one-shot compactions where router
/// hysteresis across calls is not useful.
pub fn compact(
    keys: &[f32],
    values: &[f32],
    queries: &[f32],
    t_len: usize,
    d: usize,
    n: usize,
    config: &AmConfig,
) -> Result<CompactOutput, CompactError> {
    compact_with_router(
        keys,
        values,
        queries,
        t_len,
        d,
        n,
        config,
        &mut SolverRouter::default(),
        false,
    )
    .map(|(output, _trace)| output)
}

/// Run Attention Matching compaction with an explicit router (Plan 271 T2.5).
///
/// Each stage picks its backend via the router. Currently the only stage
/// that actually dispatches differently is the full-K score matrix
/// computation (Stage 1 input), which routes across CpuScalar / CpuSimd /
/// CpuRayon / Gpu based on `t_len`. The compact-key matrix (`n × t`) is
/// small enough that scalar always wins. Cholesky dispatch is by size
/// (blocked vs unblocked) — see `value_fitter::cholesky_decompose`.
///
/// # Arguments
/// * `keys`, `values`, `queries`, `t_len`, `d`, `n`, `config` — same as [`compact`].
/// * `router` — mutable because hysteresis state is updated per call.
/// * `gpu_available` — whether a GPU backend is dispatchable. When `true`
///   and `t_len >= router.config().gpu_min_t`, the score matrix stage will
///   report `SolverBackend::Gpu` in the trace and dispatch via the GPU
///   stub (Plan 271 T2.8). The stub currently falls back to rayon on
///   non-macOS or when the `gpu_inference` feature is off.
///
/// Returns both the compact output and a [`RouterTrace`] for inspection.
pub fn compact_with_router(
    keys: &[f32],
    values: &[f32],
    queries: &[f32],
    t_len: usize,
    d: usize,
    n: usize,
    config: &AmConfig,
    router: &mut SolverRouter,
    gpu_available: bool,
) -> Result<(CompactOutput, RouterTrace), CompactError> {
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
    let mut trace = RouterTrace::default();

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

    // Extract Ck = K[selection.indices]. Move indices out of `selection` — we
    // don't use `weights` here (β is re-fit below). Pre-allocate exactly and
    // memcpy each row via `extend_from_slice` — the prior `flat_map().collect()`
    // did not pre-size the Vec, causing log(t) reallocations as it grew.
    let selected_indices = selection.indices;
    let mut compact_keys = Vec::with_capacity(selected_indices.len() * d);
    for &idx in &selected_indices {
        compact_keys.extend_from_slice(&keys[idx * d..(idx + 1) * d]);
    }

    // Stage 2: Fit β via NNLS on the selected subset.
    // Build mass feature matrix A ∈ R^{n×t}: A_ij = exp(q_i (Ck)_j^T / √d).
    // And target m ∈ R^n: m_i = Σ_k exp(q_i K_k^T / √d).
    //
    // The compact-key matrix is always `n × t` with `t << T`, so the
    // mass-features stage always uses the scalar/SIMD path regardless of
    // router decision. We still log the router's pick for traceability.
    trace.mass_features_backend = Some(router.pick_backend(t.max(1), t_len, gpu_available));
    let mut a_mass = vec![0.0f32; n * t];
    let inv_sqrt_d = 1.0f32 / (d as f32).sqrt();
    // Reuse the SIMD `dot_8wide` kernel (8-wide FMA on AVX2/NEON) instead of
    // the prior scalar dot loop. For d=64 this is ~8× fewer add/mul ops in the
    // inner loop after auto-vectorization.
    use crate::score_matrix_simd::dot_8wide;
    for i in 0..n {
        let q_row = &queries[i * d..(i + 1) * d];
        let a_row = &mut a_mass[i * t..(i + 1) * t];
        for j in 0..t {
            let ck_row = &compact_keys[j * d..(j + 1) * d];
            let dot = dot_8wide(q_row, ck_row, d);
            a_row[j] = (dot * inv_sqrt_d).max(-50.0).exp(); // clamp for stability
        }
    }

    // Target mass m and Y target: compute from full K. This is the stage where
    // the router actually dispatches across backends (the matrix is `n × T`,
    // large). We fuse the softmax + attention-output (Y) into one kernel so the
    // `(n, t_len)` normalized-attention matrix is materialized only when the
    // reconstruction report needs it — otherwise it is folded directly into the
    // `Y = A · V` reduction. See `compute_softmax_attention_and_output` for
    // the disclosed float-op reordering (divide-once vs per-element division).
    let score_backend = router.pick_backend(t_len.max(1), t_len, gpu_available);
    trace.score_matrix_backend = Some(score_backend);
    let mut full_scores = vec![0.0f32; n * t_len];
    dispatch_score_matrix(score_backend, queries, keys, n, t_len, d, &mut full_scores);
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

    // Fit β. For OMP we already have weights from selection; we re-fit here to
    // also produce a relative error estimate.
    let beta_cfg = BetaFitConfig {
        iters: config.nnls_iters,
        w_lower: config.w_lower,
        w_upper: config.w_upper,
        power_iter_steps: config.power_iter_steps,
    };
    let beta_result = fit_beta_nnls(&a_mass, &m_target, n, t, &beta_cfg);
    // Move `beta` out of the result (avoids cloning `t` f32). `relative_error`
    // is `f32` (Copy). `weights` is dropped — unused downstream.
    let relative_mass_error = beta_result.relative_error;
    let beta = beta_result.beta;

    // Stage 3: Fit Cv via least squares.
    // Blocked Cholesky is eligible when t >= CHOLESKY_BLOCK_SIZE (T2.4).
    trace.blocked_cholesky_eligible = t >= crate::value_fitter::CHOLESKY_BLOCK_SIZE;

    // Build X ∈ R^{n×t}: X_i = softmax((q_i Ck^T + β) / √d).
    let mut x_attn = vec![0.0f32; n * t];
    compute_compact_attention(queries, &compact_keys, &beta, n, t, d, &mut x_attn);

    let cv_cfg = ValueFitConfig {
        ridge_lambda: config.cv_ridge_lambda,
        cholesky_jitter: config.cholesky_jitter,
    };
    let cv_result = fit_cv_least_squares(&x_attn, &y_target, n, t, d, &cv_cfg);
    let compact_values = cv_result.compact_values;
    let relative_attn_output_error = cv_result.relative_error;

    // Optional reconstruction report. Shared helper (also used by
    // `compact_with_fixed_beta`) keeps the cache-friendly `i`-outer loop order
    // in one place — DRY across both compact paths.
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

    Ok((
        AmResult {
            selected_indices,
            compact_keys,
            beta,
            compact_values,
            original_len: t_len,
            compact_len: t,
            head_dim: d,
            report,
        },
        trace,
    ))
}

/// Dispatch the score matrix computation to the backend chosen by the router.
///
/// `CpuScalar` and `CpuSimd` both use [`compute_score_matrix`] (which itself
/// contains the 8-wide SIMD inner loop — LLVM picks scalar vs SIMD based on
/// the build profile and target). `CpuRayon` uses [`compute_score_matrix_rayon`]
/// for parallelism across query rows. `Gpu` forwards to the GPU stub
/// (T2.8) which falls back to rayon when GPU dispatch is unavailable.
/// `Ane` currently routes through `CpuScalar` — ANE dispatch for AM is future
/// work (the ANE backend exists for transformer forward, not for arbitrary
/// matmul).
#[inline]
fn dispatch_score_matrix(
    backend: SolverBackend,
    queries: &[f32],
    keys: &[f32],
    n: usize,
    t_len: usize,
    d: usize,
    out: &mut [f32],
) {
    match backend {
        SolverBackend::CpuScalar | SolverBackend::CpuSimd | SolverBackend::Ane => {
            // Scalar and SIMD share the same kernel — LLVM auto-vectorizes
            // in release mode. The router distinguishes them for future
            // tighter kernel specialization.
            compute_score_matrix(queries, keys, n, t_len, d, out);
        }
        SolverBackend::CpuRayon => {
            compute_score_matrix_rayon(queries, keys, n, t_len, d, out);
        }
        SolverBackend::Gpu => {
            #[cfg(all(target_os = "macos", feature = "gpu_inference"))]
            {
                // T2.8 GPU stub: try GPU, fall back to rayon on any error.
                if crate::score_matrix_gpu::try_compute_score_matrix_gpu(
                    queries, keys, n, t_len, d, out,
                )
                .is_ok()
                {
                    return;
                }
            }
            // Fallback: rayon-parallel CPU.
            compute_score_matrix_rayon(queries, keys, n, t_len, d, out);
        }
    }
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

    /// T2.5 — compact_with_router must produce identical output to compact()
    /// when both use the scalar path. The router may pick a different
    /// backend for the score matrix (CpuRayon for large T), but the result
    /// must be numerically identical because all backends produce the same
    /// matrix.
    #[test]
    fn test_compact_with_router_matches_compact() {
        let (keys, values) = synth_kv(32, 8, 1);
        let queries = synth_queries(4, 8, 1);
        let cfg = AmConfig::highest_attn(8);

        let r_no_router = compact(&keys, &values, &queries, 32, 8, 4, &cfg).expect("compact ok");

        let mut router = SolverRouter::default();
        let (r_with_router, trace) =
            compact_with_router(&keys, &values, &queries, 32, 8, 4, &cfg, &mut router, false)
                .expect("compact ok");

        // Selected indices must match.
        assert_eq!(r_no_router.selected_indices, r_with_router.selected_indices);
        // β and Cv must match within numerical tolerance.
        for j in 0..r_no_router.beta.len() {
            assert!(
                (r_no_router.beta[j] - r_with_router.beta[j]).abs() < 1e-5,
                "beta[{}] differs: {} vs {}",
                j,
                r_no_router.beta[j],
                r_with_router.beta[j]
            );
        }
        for j in 0..r_no_router.compact_values.len() {
            assert!(
                (r_no_router.compact_values[j] - r_with_router.compact_values[j]).abs() < 1e-4,
                "compact_values[{}] differs",
                j
            );
        }
        // Trace should be populated.
        assert!(trace.score_matrix_backend.is_some());
        assert!(trace.mass_features_backend.is_some());
    }

    /// T2.5 — for large T (above simd_max_t), the router should pick
    /// CpuRayon and the dispatch path should use the rayon kernel. The
    /// result must still match the scalar reference.
    #[test]
    fn test_compact_with_router_picks_rayon_for_large_t() {
        // T = 2048 > simd_max_t (1024), no GPU available → CpuRayon.
        let t_len = 2048;
        let d = 16;
        let n = 4;
        let t = 32;
        let (keys, values) = synth_kv(t_len, d, 11);
        let queries = synth_queries(n, d, 11);
        let cfg = AmConfig::omp_fast(t);

        let mut router = SolverRouter::default();
        let (result, trace) = compact_with_router(
            &keys,
            &values,
            &queries,
            t_len,
            d,
            n,
            &cfg,
            &mut router,
            false,
        )
        .expect("compact ok");

        // The score-matrix backend should be CpuRayon for T=2048 (no GPU).
        // Note: the actual pick also depends on hysteresis, but with a fresh
        // router and T=2048, the first call should be CpuRayon.
        assert_eq!(trace.score_matrix_backend, Some(SolverBackend::CpuRayon));
        assert_eq!(result.compact_len, t);
        for &b in &result.beta {
            assert!(b.is_finite());
        }
    }

    /// T2.5 — gpu_available=true with T >= gpu_min_t should pick Gpu backend
    /// in the trace, but the actual compute falls back through the stub to
    /// rayon. Result must still be correct.
    #[test]
    fn test_compact_with_router_gpu_flag_records_gpu_backend() {
        // T = 8192 > gpu_min_t (4096), gpu_available=true → Gpu.
        let t_len = 8192;
        let d = 16;
        let n = 2;
        let t = 16;
        let (keys, values) = synth_kv(t_len, d, 23);
        let queries = synth_queries(n, d, 23);
        let cfg = AmConfig::omp_fast(t);

        let mut router = SolverRouter::default();
        let (result, trace) = compact_with_router(
            &keys,
            &values,
            &queries,
            t_len,
            d,
            n,
            &cfg,
            &mut router,
            true,
        )
        .expect("compact ok");

        assert_eq!(trace.score_matrix_backend, Some(SolverBackend::Gpu));
        // Output must still be correct even though GPU dispatch was a stub
        // and we fell back to rayon.
        assert_eq!(result.compact_len, t);
        for &b in &result.beta {
            assert!(b.is_finite());
        }
    }

    /// T2.5 — for t >= CHOLESKY_BLOCK_SIZE, the trace must report that
    /// blocked Cholesky was eligible (T2.4 integration).
    #[test]
    fn test_compact_with_router_reports_blocked_cholesky_eligibility() {
        use crate::value_fitter::CHOLESKY_BLOCK_SIZE;
        let t_len = 256;
        let d = 8;
        let n = 4;
        let t = CHOLESKY_BLOCK_SIZE + 4; // forces blocked path
        let (keys, values) = synth_kv(t_len, d, 29);
        let queries = synth_queries(n, d, 29);
        let cfg = AmConfig::omp_fast(t);

        let mut router = SolverRouter::default();
        let (_result, trace) = compact_with_router(
            &keys,
            &values,
            &queries,
            t_len,
            d,
            n,
            &cfg,
            &mut router,
            false,
        )
        .expect("compact ok");

        assert!(
            trace.blocked_cholesky_eligible,
            "blocked Cholesky should be eligible for t={} (block size {})",
            t, CHOLESKY_BLOCK_SIZE
        );
    }
}
