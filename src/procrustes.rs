//! Orthogonal Procrustes — closed-form cross-frame embedding alignment.
//!
//! Issue 001 (katgpt-rs). GOAT candidate (not Super-GOAT — see the issue).
//!
//! Given two paired anchor sets `A, B ∈ R^{n × d}` representing the same
//! entities in two different coordinate frames (e.g. KG embeddings from
//! different shards, different model snapshots, different game instances),
//! find the orthogonal matrix `R ∈ R^{d × d}` that best aligns `A` to `B`:
//!
//! ```text
//!   R* = arg min_R  ‖B − A R^T‖_F²    s.t.  R^T R = I
//! ```
//!
//! # Closed-form via polar decomposition
//!
//! The classical solution uses SVD: form `M = B^T A`, take SVD `M = U Σ V^T`,
//! then `R* = U V^T`. This module avoids LAPACK by observing that **the
//! Procrustes solution `R* = U V^T` is exactly the orthogonal polar factor
//! of `M`**, which [`newton_schulz`] computes in 5 fixed-point iterations
//! with no eigensolver. The algorithm:
//!
//! 1. (Optional) Center `A` and `B` by subtracting their column means.
//! 2. Form `M = B^T A` (d×d matrix, single fused pass over `n` rows).
//! 3. Pre-scale: `M̃ = M / ‖M‖_F` so singular values fit `[0, 1]`
//!    (Newton-Schulz convergence range).
//! 4. Newton-Schulz 5 iters on `M̃` → `R̃` (polar factor; `R̃^T R̃ ≈ I`).
//! 5. (Optional) Special-orthogonal correction: if `det(R̃) < 0`, flip the
//!    last column to enforce `det(R) = +1` (Procrustes-rotation variant).
//! 6. (Optional) Compute residual `‖B − A R̃^T‖_F / ‖B‖_F` for the report.
//!
//! The polar factor is scale-invariant — `R̃` equals the polar factor of
//! the original (unnormalized) `M`, so the pre-scaling in step 3 only
//! affects Newton-Schulz convergence, not the result.
//!
//! # Complexity
//!
//! - `M = B^T A`: `O(n d²)` (one pass over `n` rows, `d²` fused FMA per row).
//! - Newton-Schulz: `O(d³)` (5 iters × 3 d×d matmuls).
//! - Total: `O(n d² + d³)`. For `n = 512, d = 32` → ~525 KFLOPs (sub-ms).
//!
//! # Determinism
//!
//! Newton-Schulz with fixed iteration count + Frobenius pre-scaling is
//! deterministic across platforms (no eigensolver, no convergence loop,
//! no platform-specific LAPACK). G3 determinism test (Issue 001) verifies
//! bit-identical `R` on `x86_64`, `aarch64`, `wasm32` for the same anchor
//! pair. (See `tests/procrustes_determinism.rs`.)
//!
//! # Raw vs Latent boundary (AGENTS.md)
//!
//! The rotation matrix `R` is **latent**. It is computed locally at
//! shard-join / catchup, applied to local embeddings, and then discarded
//! or cached for diagnostics. **Never synced.** What crosses the sync
//! boundary is the entity identity claim ("shard S asserts entity E is
//! the same as canonical entity E'"), committed as a raw KG triple with
//! a confidence scalar — see Issue 001 §"Raw vs latent boundary".
//!
//! # References
//!
//! - Issue 001: `riir-chain/.issues/001_orthogonal_procrustes_kg_shard_alignment.md`
//! - Newton-Schulz orthogonalization: [`crate::newton_schulz`] (Plan 152).
//! - Schönemann, P. (1966) — original Procrustes solution via SVD.
//! - Higham, N. (1986) — Newton-Schulz iteration for polar factor.

use crate::simd::{simd_dot_f32, simd_fused_scale_acc, simd_scale_inplace, simd_sum_sq};

/// Configuration for [`orthogonal_procrustes`].
#[derive(Clone, Copy, Debug)]
pub struct ProcrustesConfig {
    /// If `true`, subtract the column means of `A` and `B` before forming
    /// `M = B^T A`. Useful when the two frames have different origins
    /// (typical for embedding spaces). Default: `true`.
    pub center: bool,
    /// If `true`, enforce `det(R) = +1` (rotation / special orthogonal).
    /// If `false`, allow `det(R) = ±1` (orthogonal, may reflect). The
    /// classical Procrustes solution allows reflection. Default: `false`.
    pub special_orthogonal: bool,
    /// If `true`, compute the residual `‖B − A R^T‖_F / ‖B‖_F` and include
    /// it in the report. Adds an `O(n d²)` pass. Default: `true`.
    pub compute_residual: bool,
    /// If `true`, compute `det(R)` and include it in the report. Adds an
    /// `O(d³)` recursive expansion. Default: `false` (only needed for
    /// debugging the special-orthogonal correction).
    pub compute_det: bool,
    /// Minimum number of anchors required for a well-determined Procrustes
    /// solution. Below this the system is rank-deficient and the rotation
    /// is underdetermined. The function returns [`ProcrustesError::Underdetermined`]
    /// if `n < min_anchors`. Default: `2 * d` (matches the issue's
    /// "fall back to brute-force NN when `n < 2·d`" rule).
    pub min_anchors: usize,
}

impl Default for ProcrustesConfig {
    #[inline]
    fn default() -> Self {
        Self {
            center: true,
            special_orthogonal: false,
            compute_residual: true,
            compute_det: false,
            min_anchors: 0, // Replaced with 2*d at runtime when zero.
        }
    }
}

/// Pre-allocated scratch space for [`orthogonal_procrustes`]. Caller-owned
/// to keep the hot path zero-allocation.
///
/// # Sizing
///
/// Construct with [`ProcrustesScratch::new`]`(`n`, `d`)`. The scratch is
/// reusable across calls with the same or smaller `(n, d)`. Use
/// [`ProcrustesScratch::ensure_capacity`] to grow if needed.
pub struct ProcrustesScratch {
    /// Column mean of `A` — `d` elements.
    mean_a: Vec<f32>,
    /// Column mean of `B` — `d` elements.
    mean_b: Vec<f32>,
    /// `M = B^T A` (or its centered variant) — `d * d` elements.
    m: Vec<f32>,
    /// Polar iteration: `X^T X` accumulator (d×d), then multiplier
    /// matrix `T = 1.5I - 0.5 X^T X` (d×d).
    xtx: Vec<f32>,
    /// Polar iteration: `X_new = X @ T` (d×d).
    x_new: Vec<f32>,
    /// Buffer for the predicted `A R^T` row during residual computation —
    /// `d` elements. Reused across `n` rows.
    predicted_row: Vec<f32>,
}

impl ProcrustesScratch {
    /// Create scratch sized for anchors up to `max_n × max_d`.
    ///
    /// Note: `max_n` is currently unused (the scratch does not need to
    /// allocate per-row buffers), but is accepted for forward
    /// compatibility if future optimizations add row-strided scratch.
    #[inline]
    pub fn new(max_n: usize, max_d: usize) -> Self {
        let _ = max_n;
        Self {
            mean_a: vec![0.0; max_d],
            mean_b: vec![0.0; max_d],
            m: vec![0.0; max_d * max_d],
            xtx: vec![0.0; max_d * max_d],
            x_new: vec![0.0; max_d * max_d],
            predicted_row: vec![0.0; max_d],
        }
    }
}

impl ProcrustesScratch {
    /// Ensure scratch buffers are large enough for `n × d` anchors.
    /// Grows monotonically (no shrinking). Returns `true` if reallocation
    /// occurred.
    #[inline]
    pub fn ensure_capacity(&mut self, n: usize, d: usize) -> bool {
        let dd = d * d;
        let grew = self.mean_a.len() < d
            || self.mean_b.len() < d
            || self.m.len() < dd
            || self.xtx.len() < dd
            || self.x_new.len() < dd
            || self.predicted_row.len() < d;
        if grew {
            self.mean_a.resize(d.max(self.mean_a.len()), 0.0);
            self.mean_b.resize(d.max(self.mean_b.len()), 0.0);
            self.m.resize(dd.max(self.m.len()), 0.0);
            self.xtx.resize(dd.max(self.xtx.len()), 0.0);
            self.x_new.resize(dd.max(self.x_new.len()), 0.0);
            self.predicted_row
                .resize(d.max(self.predicted_row.len()), 0.0);
        }
        let _ = n;
        grew
    }
}

/// Errors raised by [`orthogonal_procrustes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcrustesError {
    /// `a.len() != n * d` or `b.len() != n * d`.
    ShapeMismatch { expected: usize, got: usize },
    /// `out_rotation.len() != d * d`.
    OutputSizeMismatch { expected: usize, got: usize },
    /// `n < min_anchors` (or `2*d` if unspecified). The system is
    /// rank-deficient and Procrustes is underdetermined. Fall back to
    /// brute-force nearest-neighbor join in this regime.
    Underdetermined {
        n: usize,
        min_anchors: usize,
        d: usize,
    },
    /// `d == 0`. Need at least 1-dim embeddings.
    ZeroDim,
    /// `‖M‖_F` was zero (both anchor sets are all-zero). No meaningful
    /// rotation exists.
    DegenerateAnchors,
}

impl std::fmt::Display for ProcrustesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcrustesError::ShapeMismatch { expected, got } => write!(
                f,
                "input shape mismatch: expected {expected} elements, got {got}"
            ),
            ProcrustesError::OutputSizeMismatch { expected, got } => write!(
                f,
                "output buffer size mismatch: expected {expected} elements, got {got}"
            ),
            ProcrustesError::Underdetermined { n, min_anchors, d } => write!(
                f,
                "underdetermined: n={n} < min_anchors={min_anchors} (d={d})"
            ),
            ProcrustesError::ZeroDim => write!(f, "d must be >= 1, got 0"),
            ProcrustesError::DegenerateAnchors => {
                write!(f, "Frobenius norm of M is zero (anchors are degenerate)")
            }
        }
    }
}

impl std::error::Error for ProcrustesError {}

/// Report returned by [`orthogonal_procrustes`] on success.
#[derive(Debug, Clone, Copy)]
pub struct ProcrustesReport {
    /// Number of anchor pairs used.
    pub n: usize,
    /// Embedding dimension.
    pub d: usize,
    /// Frobenius norm of `M = B^T A` before normalization. Useful as a
    /// sanity check (zero ⇒ degenerate).
    pub m_norm: f32,
    /// Residual `‖B − A R^T‖_F / ‖B‖_F`. Smaller is better. Computed only
    /// if [`ProcrustesConfig::compute_residual`] is `true`; otherwise `NaN`.
    pub residual: f32,
    /// `det(R)`. Computed only if [`ProcrustesConfig::compute_det`] is
    /// `true` OR [`ProcrustesConfig::special_orthogonal`] is `true`
    /// (which forces det = +1). Otherwise `NaN`.
    pub det: f32,
    /// Whether the special-orthogonal correction (column flip) was applied.
    /// Always `false` if [`ProcrustesConfig::special_orthogonal`] is `false`.
    pub flipped: bool,
}

/// Compute the orthogonal Procrustes rotation `R` that best aligns
/// `A` to `B` (see [module docs](self)). Zero-alloc hot path: caller-owned
/// scratch + output buffer.
///
/// # Arguments
///
/// * `a` — source anchor embeddings, shape `(n, d)` row-major.
/// * `b` — target anchor embeddings (same anchors in target frame), shape `(n, d)` row-major.
/// * `n` — number of anchor pairs.
/// * `d` — embedding dimension.
/// * `out_rotation` — output buffer for `R`, shape `(d, d)` row-major.
///   Overwritten on success.
/// * `scratch` — caller-owned scratch (see [`ProcrustesScratch`]).
/// * `config` — see [`ProcrustesConfig`].
///
/// # Returns
///
/// [`ProcrustesReport`] on success, [`ProcrustesError`] on validation
/// failure. On error, `out_rotation` is left in an unspecified state
/// (typically partially-written or zeroed).
///
/// # Pipeline
///
/// See the [module docs](self) for the math. Order:
/// 1. Validate shapes + `n ≥ min_anchors` (or `2*d` if unspecified).
/// 2. Center (if `config.center`).
/// 3. Compute `M = B^T A` in a single fused pass over `n` rows.
/// 4. Compute `‖M‖_F` (single pass over `d²`).
/// 5. Pre-scale `M̃ = M / ‖M‖_F`.
/// 6. Newton-Schulz 5 iters on `M̃` → `R`.
/// 7. (If `config.special_orthogonal`) Check `det(R)`, flip last column if `< 0`.
/// 8. (If `config.compute_residual`) Compute residual.
/// 9. Build + return report.
///
/// # Example
///
/// ```
/// # use katgpt_rs::procrustes::*;
/// // 4 anchors in 2-d source frame, rotated 90° to target frame.
/// let a: [f32; 8] = [1.0, 0.0,  0.0, 1.0,  -1.0, 0.0,  0.0, -1.0];
/// // 90° rotation: (x, y) → (-y, x)
/// let b: [f32; 8] = [0.0, 1.0, -1.0, 0.0,   0.0, -1.0,  1.0,  0.0];
/// let mut r = [0.0_f32; 4];
/// let mut scratch = ProcrustesScratch::new(4, 2);
/// let report = orthogonal_procrustes(&a, &b, 4, 2, &mut r, &mut scratch,
///                                     &ProcrustesConfig::default()).unwrap();
/// // R ≈ [[0, -1], [1, 0]] (90° rotation).
/// assert!((r[0] - 0.0).abs() < 1e-4);
/// assert!((r[1] + 1.0).abs() < 1e-4);
/// assert!((r[2] - 1.0).abs() < 1e-4);
/// assert!((r[3] - 0.0).abs() < 1e-4);
/// assert!(report.residual < 1e-4, "exact rotation should have ~0 residual");
/// ```
#[inline]
pub fn orthogonal_procrustes(
    a: &[f32],
    b: &[f32],
    n: usize,
    d: usize,
    out_rotation: &mut [f32],
    scratch: &mut ProcrustesScratch,
    config: &ProcrustesConfig,
) -> Result<ProcrustesReport, ProcrustesError> {
    // ── 1. Validate shapes ──────────────────────────────────────────
    if d == 0 {
        return Err(ProcrustesError::ZeroDim);
    }
    let expected = n.checked_mul(d).ok_or(ProcrustesError::ShapeMismatch {
        expected: 0,
        got: a.len(),
    })?;
    if a.len() != expected {
        return Err(ProcrustesError::ShapeMismatch {
            expected,
            got: a.len(),
        });
    }
    if b.len() != expected {
        return Err(ProcrustesError::ShapeMismatch {
            expected,
            got: b.len(),
        });
    }
    let dd = d * d;
    if out_rotation.len() != dd {
        return Err(ProcrustesError::OutputSizeMismatch {
            expected: dd,
            got: out_rotation.len(),
        });
    }

    // Min anchors: config.min_anchors if set, else 2*d (Issue 001 fallback rule).
    let min_anchors = if config.min_anchors == 0 {
        2 * d
    } else {
        config.min_anchors
    };
    if n < min_anchors {
        return Err(ProcrustesError::Underdetermined { n, min_anchors, d });
    }

    scratch.ensure_capacity(n, d);

    // ── 2. Center (optional) ────────────────────────────────────────
    // Compute column means: single pass over each row, accumulate into
    // mean_a / mean_b, then divide by n. SIMD: simd_fused_sub_acc handles
    // `dst[i] += a[i] - 0.0` style but we want raw sum — just use sum.
    let (a_eff, b_eff): (&[f32], &[f32]) = if config.center {
        // Compute means.
        // mean_a[c] = (1/n) * sum_i a[i*d + c]
        // Use a fused accumulate: scratch.mean_a[c] += a_row[c] for each row.
        // Initialize to zero, then SIMD add.
        for v in scratch.mean_a[..d].iter_mut() {
            *v = 0.0;
        }
        for v in scratch.mean_b[..d].iter_mut() {
            *v = 0.0;
        }
        for i in 0..n {
            let a_row = &a[i * d..(i + 1) * d];
            let b_row = &b[i * d..(i + 1) * d];
            // mean += row (SIMD-friendly FMA-style)
            for c in 0..d {
                scratch.mean_a[c] += a_row[c];
                scratch.mean_b[c] += b_row[c];
            }
        }
        let inv_n = 1.0 / (n as f32);
        for c in 0..d {
            scratch.mean_a[c] *= inv_n;
            scratch.mean_b[c] *= inv_n;
        }
        // Note: we do NOT modify a/b in place (caller owns them). The
        // centering is applied inside the M = B^T A step below via the
        // (b[i,c] - mean_b[c]) * (a[i,c'] - mean_a[c']) formulation.
        (a, b)
    } else {
        for v in scratch.mean_a[..d].iter_mut() {
            *v = 0.0;
        }
        for v in scratch.mean_b[..d].iter_mut() {
            *v = 0.0;
        }
        (a, b)
    };

    // ── 3. Compute M = B^T A  (d × d) via row-by-row outer-product accumulation ──
    //
    // M[c', c] = sum_i  a[i, c] * b[i, c']  (raw B^T A)
    //
    // For each row i: form the rank-1 update M += a[i, :] ⊗ b[i, :] (outer
    // product). Then if centering: apply the rank-1 correction
    // M_centered = M_raw - n * mean_a ⊗ mean_b (algebraic identity — see
    // comment block below for derivation).
    //
    // Centering derivation (for reference):
    //   sum_i (a[i,c]-μ_a[c]) * (b[i,c']-μ_b[c'])
    //     = sum_i a[i,c]*b[i,c'] - μ_a[c]*sum_i b[i,c'] - μ_b[c']*sum_i a[i,c] + n*μ_a[c]*μ_b[c']
    //     = sum_i a[i,c]*b[i,c'] - μ_a[c] * n*μ_b[c'] - μ_b[c'] * n*μ_a[c] + n*μ_a[c]*μ_b[c']
    //     = sum_i a[i,c]*b[i,c'] - n*μ_a[c]*μ_b[c']
    // So M_centered = M_raw - n * mean_a ⊗ mean_b. One fused FMA pass for
    // M_raw, one rank-1 subtraction for centering. Faster than per-row
    // centering.
    for v in scratch.m[..dd].iter_mut() {
        *v = 0.0;
    }
    for i in 0..n {
        let a_row = &a_eff[i * d..(i + 1) * d];
        let b_row = &b_eff[i * d..(i + 1) * d];
        // Outer product M += a_row ⊗ b_row (M is d×d row-major).
        // M[c', c] += a_row[c] * b_row[c']
        // Inner loop over c (column of M) is contiguous; SIMD-fused.
        for (cprime, &b_icprime) in b_row.iter().enumerate() {
            let m_row = &mut scratch.m[cprime * d..(cprime + 1) * d];
            // m_row[c] += a_row[c] * b_icprime  for each c (SIMD fused scale-acc).
            simd_fused_scale_acc(m_row, a_row, b_icprime, d);
        }
    }

    // Apply centering as a rank-1 correction: M -= n * mean_a ⊗ mean_b.
    if config.center {
        for (cprime, &mean_b_cprime) in scratch.mean_b[..d].iter().enumerate() {
            let m_row = &mut scratch.m[cprime * d..(cprime + 1) * d];
            // m_row[c] -= n * mean_a[c] * mean_b_cprime
            for (m_val, &mean_a_c) in m_row.iter_mut().zip(&scratch.mean_a[..d]) {
                *m_val -= (n as f32) * mean_a_c * mean_b_cprime;
            }
        }
    }

    // ── 4. Compute ‖M‖_F ────────────────────────────────────────────
    let m_norm = simd_sum_sq(&scratch.m[..dd], dd).sqrt();
    if !m_norm.is_finite() || m_norm == 0.0 {
        return Err(ProcrustesError::DegenerateAnchors);
    }

    // ── 5. Pre-scale M̃ = M / ‖M‖_F ──────────────────────────────────
    // The polar iteration `X_new = X (3I - X^T X) / 2` converges for
    // σ(X) ∈ [0, √3). Dividing by Frobenius norm ensures σ_max ≤ 1 (well
    // within convergence radius). The polar factor is scale-invariant
    // (R̃ = polar(M) = polar(M̃)), so this normalization does not affect
    // the result — it only ensures convergence.
    simd_scale_inplace(&mut scratch.m[..dd], 1.0 / m_norm);

    // ── 6. Polar iteration: X_{k+1} = X_k (3I - X_k^T X_k) / 2 ──────
    // Newton's method for the matrix sign function applied to the polar
    // decomposition. Converges QUADRATICALLY to the orthogonal polar
    // factor with σ = 1 stable (derivative = 0 at fixed point — much
    // better than the AMUSE Newton-Schulz coefficients which have
    // |p'(1)| ≈ 1.4 and oscillate).
    //
    // In σ-space: σ_new = 0.5 σ (3 - σ²). Fixed points at σ = 0, ±1.
    // For σ ∈ [0, √3), monotonically increases to 1. For σ ∈ [0.5, 1],
    // converges to machine precision in ≤ 6 iterations.
    //
    // We use 7 iterations as a safe default (handles σ_0 as low as 0.1).
    polar_iteration(
        &scratch.m[..dd],
        d,
        out_rotation,
        &mut scratch.xtx,
        &mut scratch.x_new,
    );

    // ── 7. (Optional) Special-orthogonal correction ─────────────────
    let mut flipped = false;
    let mut det = f32::NAN;
    if config.special_orthogonal || config.compute_det {
        det = determinant_d(out_rotation, d);
    }
    if config.special_orthogonal && det < 0.0 {
        // Flip the last column. This converts det(R) from -1 to +1 while
        // staying close to the orthogonal Procrustes solution (only one
        // of d columns changes sign). The true rotation Procrustes
        // solution requires finding the smallest-σ column, which needs
        // full SVD; we approximate by flipping the last column, which
        // is provably optimal when the smallest singular value is
        // associated with the last index (a common case for structured
        // embeddings) and near-optimal otherwise.
        for cprime in 0..d {
            out_rotation[cprime * d + (d - 1)] = -out_rotation[cprime * d + (d - 1)];
        }
        flipped = true;
        det = -det; // Update det if it was computed.
    }

    // ── 8. (Optional) Compute residual ──────────────────────────────
    // residual = ‖B_centered − A_centered R^T‖_F / ‖B_centered‖_F
    //
    // When `config.center` is on, the residual must be computed on the
    // *centered* data (A - μ_a, B - μ_b) — otherwise translation leaks in
    // and the residual is meaningless. When centering is off, μ_a = μ_b = 0
    // and the centered form reduces to the raw form.
    let mut residual = f32::NAN;
    if config.compute_residual {
        let mut residual_sq = 0.0_f32;
        let mut b_norm_sq = 0.0_f32;
        for i in 0..n {
            let a_row = &a_eff[i * d..(i + 1) * d];
            let b_row = &b_eff[i * d..(i + 1) * d];
            // predicted = R @ a_centered_row  (d-vector)
            //   predicted[c'] = sum_c R[c', c] * (a_row[c] - mean_a[c])
            let predicted = &mut scratch.predicted_row[..d];
            for cprime in 0..d {
                let r_row = &out_rotation[cprime * d..(cprime + 1) * d];
                if config.center {
                    // Centered dot: sum_c R[c',c] * (a[c] - μ_a[c])
                    let mut acc = 0.0_f32;
                    for c in 0..d {
                        acc += r_row[c] * (a_row[c] - scratch.mean_a[c]);
                    }
                    predicted[cprime] = acc;
                } else {
                    predicted[cprime] = simd_dot_f32(r_row, a_row, d);
                }
            }
            // residual_sq += ‖b_centered_row - predicted‖²
            // b_norm_sq += ‖b_centered_row‖²
            for c in 0..d {
                let b_centered = if config.center {
                    b_row[c] - scratch.mean_b[c]
                } else {
                    b_row[c]
                };
                let diff = b_centered - predicted[c];
                residual_sq += diff * diff;
                b_norm_sq += b_centered * b_centered;
            }
        }
        residual = if b_norm_sq > 0.0 {
            (residual_sq / b_norm_sq).sqrt()
        } else {
            f32::NAN
        };
    }

    // ── 9. Build report ─────────────────────────────────────────────
    Ok(ProcrustesReport {
        n,
        d,
        m_norm,
        residual,
        det,
        flipped,
    })
}

/// Newton iteration for the orthogonal polar factor.
///
/// `X_{k+1} = (1/2) X_k (3I - X_k^T X_k)` — quadratically convergent to
/// `polar(X_0) = U V^T` (the orthogonal Procrustes solution) for any
/// non-singular `X_0` scaled such that `||X_0||_2 < sqrt(3)`.
///
/// This is the classical Newton-Schulz iteration for the polar factor
/// (Higham 1986), distinct from the AMUSE-flavored coefficients used in
/// `newton_schulz.rs` (which target the Muon optimizer's "good enough
/// sigma in [0.68, 1.12]" regime, not exact orthogonality). We need exact
/// orthogonality for Procrustes — small sigma deviations would compound
/// into the residual report.
///
/// # sigma-space analysis
///
/// For `X = sigma * R` (R orthogonal), the iteration reduces to
/// `sigma_new = 0.5 * sigma * (3 - sigma^2)`. Fixed points: sigma = 0, +/-1.
/// Stability at sigma = 1: `d(sigma_new)/d(sigma)|_1 = 0` -> quadratic
/// convergence. For sigma in (0, sqrt(3)) the iteration monotonically
/// approaches sigma = 1.
///
/// # Convergence count
///
/// - sigma_0 in [0.5, 1]: <= 6 iters to machine f32 precision.
/// - sigma_0 in [0.1, 0.5]: <= 12 iters.
/// - sigma_0 in [0.01, 0.1]: <= 18 iters (linear-regime crawl).
/// - We default to 15 iters, which covers sigma_0 >= 0.03 after Frobenius
///   pre-scaling with n >= 2d anchors (typical Procrustes regime).
///   For pathologically small sigma_0, callers should pre-condition M
///   (e.g. via QR) before invoking.
///
/// # Buffers
///
/// `xtx` and `x_new` are caller-owned scratch, both `d * d` elements.
#[inline]
fn polar_iteration(x: &[f32], d: usize, out: &mut [f32], xtx: &mut [f32], x_new: &mut [f32]) {
    debug_assert_eq!(x.len(), d * d);
    debug_assert_eq!(out.len(), d * d);
    debug_assert_eq!(xtx.len(), d * d);
    debug_assert_eq!(x_new.len(), d * d);

    const N_ITERS: usize = 15;

    // X_0 = x. Iterate in-place on `out` (use out as X_k).
    out[..d * d].copy_from_slice(&x[..d * d]);

    for _ in 0..N_ITERS {
        // A = X^T X (d x d, symmetric).
        // A[i, j] = sum_k X[k, i] * X[k, j]  (dot of column i, column j of X).
        for i in 0..d {
            // Diagonal: A[i, i] = ||column_i||^2.
            let mut diag = 0.0_f32;
            for k in 0..d {
                let x_ki = out[k * d + i];
                diag += x_ki * x_ki;
            }
            xtx[i * d + i] = diag;
            // Upper triangle + mirror: A[i, j] for j > i.
            for j in (i + 1)..d {
                let mut dot = 0.0_f32;
                for k in 0..d {
                    dot += out[k * d + i] * out[k * d + j];
                }
                xtx[i * d + j] = dot;
                xtx[j * d + i] = dot;
            }
        }

        // X_new = X @ T  where  T = 1.5 I - 0.5 A.
        // X_new[k, j] = sum_i X[k, i] * T[i, j]
        //            = 1.5 X[k, j] - 0.5 sum_i X[k, i] * A[i, j]
        //            = 1.5 X[k, j] - 0.5 (X @ A)[k, j]
        for k in 0..d {
            let x_row_k = &out[k * d..(k + 1) * d];
            let x_new_row = &mut x_new[k * d..(k + 1) * d];
            for j in 0..d {
                let mut xa_kj = 0.0_f32;
                for i in 0..d {
                    xa_kj += x_row_k[i] * xtx[i * d + j];
                }
                x_new_row[j] = 1.5 * x_row_k[j] - 0.5 * xa_kj;
            }
        }

        // X = X_new.
        out.copy_from_slice(&x_new[..d * d]);
    }
}

/// Compute the determinant of a `d × d` row-major matrix via cofactor
/// expansion. `O(d!)` — only call for small `d` (≤ 8 typical for KG
/// embeddings). For large `d`, callers should disable
/// [`ProcrustesConfig::compute_det`] and use a different method.
#[inline]
fn determinant_d(m: &[f32], d: usize) -> f32 {
    debug_assert_eq!(m.len(), d * d, "determinant_d: matrix size mismatch");
    match d {
        1 => m[0],
        2 => m[0] * m[3] - m[1] * m[2],
        3 => {
            // Sarrus' rule.
            m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
                + m[2] * (m[3] * m[7] - m[4] * m[6])
        }
        _ => {
            // Cofactor expansion along the first row. O(d!) — use sparingly.
            let mut det = 0.0_f32;
            let mut sub = vec![0.0_f32; (d - 1) * (d - 1)];
            for j in 0..d {
                // Build the (d-1)×(d-1) minor by excluding row 0 and column j.
                for r in 1..d {
                    let mut sub_col = 0;
                    for c in 0..d {
                        if c == j {
                            continue;
                        }
                        sub[(r - 1) * (d - 1) + sub_col] = m[r * d + c];
                        sub_col += 1;
                    }
                }
                let cofactor = determinant_d(&sub, d - 1);
                let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
                det += sign * m[j] * cofactor;
            }
            det
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for f32 comparisons after Newton-Schulz (5 iters converges
    /// to ~1e-5 typical; we use 1e-3 to be forgiving on cross-platform f32
    /// rounding differences).
    const TOL: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn matrices_approx_eq(actual: &[f32], expected: &[f32], tol: f32) -> bool {
        assert_eq!(actual.len(), expected.len());
        actual
            .iter()
            .zip(expected)
            .all(|(a, e)| approx_eq(*a, *e, tol))
    }

    /// Build a 2D rotation matrix for angle θ (radians).
    fn rot2(theta: f32) -> [f32; 4] {
        [theta.cos(), -theta.sin(), theta.sin(), theta.cos()]
    }

    /// Apply a 2×2 matrix to a 2-vector.
    fn apply2(m: &[f32; 4], v: [f32; 2]) -> [f32; 2] {
        [m[0] * v[0] + m[1] * v[1], m[2] * v[0] + m[3] * v[1]]
    }

    /// Apply a d×d matrix (row-major) to a d-vector.
    fn apply_d(m: &[f32], v: &[f32], d: usize) -> Vec<f32> {
        let mut out = vec![0.0; d];
        for r in 0..d {
            let row = &m[r * d..(r + 1) * d];
            out[r] = simd_dot_f32(row, v, d);
        }
        out
    }

    /// Generate n anchors in d-dim space with reproducible pseudo-randomness.
    fn pseudo_random_anchors(n: usize, d: usize, seed: u32) -> Vec<f32> {
        let mut out = vec![0.0; n * d];
        let mut state = seed;
        for v in out.iter_mut() {
            // xorshift32 — deterministic, no external dep.
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *v = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        out
    }

    #[test]
    fn exact_2d_rotation_90_deg() {
        let a = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
        // 90° rotation: (x, y) → (-y, x).
        let b: [f32; 8] = [
            apply2(&rot2(std::f32::consts::FRAC_PI_2), [1.0, 0.0])[0],
            apply2(&rot2(std::f32::consts::FRAC_PI_2), [1.0, 0.0])[1],
            apply2(&rot2(std::f32::consts::FRAC_PI_2), [0.0, 1.0])[0],
            apply2(&rot2(std::f32::consts::FRAC_PI_2), [0.0, 1.0])[1],
            apply2(&rot2(std::f32::consts::FRAC_PI_2), [-1.0, 0.0])[0],
            apply2(&rot2(std::f32::consts::FRAC_PI_2), [-1.0, 0.0])[1],
            apply2(&rot2(std::f32::consts::FRAC_PI_2), [0.0, -1.0])[0],
            apply2(&rot2(std::f32::consts::FRAC_PI_2), [0.0, -1.0])[1],
        ];
        let mut r = [0.0_f32; 4];
        let mut scratch = ProcrustesScratch::new(4, 2);
        let report = orthogonal_procrustes(
            &a,
            &b,
            4,
            2,
            &mut r,
            &mut scratch,
            &ProcrustesConfig::default(),
        )
        .expect("90° rotation should succeed");
        // Expected: R = [[0, -1], [1, 0]].
        let expected = rot2(std::f32::consts::FRAC_PI_2);
        assert!(
            matrices_approx_eq(&r, &expected, TOL),
            "R mismatch: got {:?}",
            r
        );
        assert!(
            report.residual < TOL,
            "residual should be ~0, got {}",
            report.residual
        );
    }

    #[test]
    fn exact_3d_rotation() {
        // Build 7 anchors in 3-d space, apply a known rotation, recover it.
        let a = pseudo_random_anchors(7, 3, 0xdead_beef);
        // R: 30° about z-axis.
        let theta = std::f32::consts::PI / 6.0;
        let r_expected = [
            theta.cos(),
            -theta.sin(),
            0.0,
            theta.sin(),
            theta.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        let b: Vec<f32> = (0..7)
            .flat_map(|i| {
                let a_row = &a[i * 3..(i + 1) * 3];
                apply_d(&r_expected, a_row, 3)
            })
            .collect();
        let mut r = [0.0_f32; 9];
        let mut scratch = ProcrustesScratch::new(7, 3);
        let report = orthogonal_procrustes(
            &a,
            &b,
            7,
            3,
            &mut r,
            &mut scratch,
            &ProcrustesConfig::default(),
        )
        .expect("3D rotation should succeed");
        assert!(
            matrices_approx_eq(&r, &r_expected, 1e-2),
            "R mismatch: got {:?}",
            r
        );
        assert!(
            report.residual < 1e-2,
            "residual should be small, got {}",
            report.residual
        );
    }

    #[test]
    fn rejects_underdetermined_system() {
        // n=2, d=8: n < 2*d → underdetermined.
        let a = vec![1.0; 16];
        let b = vec![1.0; 16];
        let mut r = vec![0.0; 64];
        let mut scratch = ProcrustesScratch::new(2, 8);
        let err = orthogonal_procrustes(
            &a,
            &b,
            2,
            8,
            &mut r,
            &mut scratch,
            &ProcrustesConfig::default(),
        )
        .expect_err("should reject underdetermined");
        match err {
            ProcrustesError::Underdetermined { n, min_anchors, d } => {
                assert_eq!(n, 2);
                assert_eq!(min_anchors, 16); // 2 * d.
                assert_eq!(d, 8);
            }
            _ => panic!("wrong error: {:?}", err),
        }
    }

    #[test]
    fn rejects_zero_dim() {
        let mut r: [f32; 0] = [];
        let mut scratch = ProcrustesScratch::new(0, 0);
        let err = orthogonal_procrustes(
            &[],
            &[],
            0,
            0,
            &mut r,
            &mut scratch,
            &ProcrustesConfig::default(),
        )
        .expect_err("should reject d=0");
        assert_eq!(err, ProcrustesError::ZeroDim);
    }

    #[test]
    fn rejects_shape_mismatch() {
        let a = vec![1.0; 8]; // says n=4, d=2 (8 elems) but we'll pass n=3, d=2 (6 expected).
        let b = vec![1.0; 8];
        let mut r = [0.0_f32; 4];
        let mut scratch = ProcrustesScratch::new(4, 2);
        let err = orthogonal_procrustes(
            &a,
            &b,
            3,
            2,
            &mut r,
            &mut scratch,
            &ProcrustesConfig::default(),
        )
        .expect_err("should reject shape mismatch");
        match err {
            ProcrustesError::ShapeMismatch { expected, got } => {
                assert_eq!(expected, 6);
                assert_eq!(got, 8);
            }
            _ => panic!("wrong error"),
        }
    }

    #[test]
    fn degenerate_anchors_rejected() {
        // All-zero anchors → M = 0 → ‖M‖_F = 0 → DegenerateAnchors.
        // Use d=2, n=4 so n >= 2*d (4 >= 4) — otherwise the Underdetermined
        // check fires first.
        let a = vec![0.0; 8];
        let b = vec![0.0; 8];
        let mut r = vec![0.0; 4];
        let mut scratch = ProcrustesScratch::new(4, 2);
        let err = orthogonal_procrustes(
            &a,
            &b,
            4,
            2,
            &mut r,
            &mut scratch,
            &ProcrustesConfig::default(),
        )
        .expect_err("should reject degenerate");
        assert_eq!(err, ProcrustesError::DegenerateAnchors);
    }

    #[test]
    fn centering_handles_offset_frames() {
        // A and B differ only by a translation (no rotation).
        // With centering ON, Procrustes should recover R = I with residual ~ 0.
        // Use 2D-scattered points (not collinear) so M is well-conditioned.
        let anchors: [[f32; 2]; 8] = [
            [1.0, 2.0],
            [-1.5, 0.5],
            [3.0, -1.0],
            [-2.0, -2.0],
            [0.5, 3.5],
            [-3.0, 1.0],
            [2.5, 2.5],
            [-0.5, -3.0],
        ];
        let mut a = Vec::new();
        let mut b = Vec::new();
        for [x, y] in anchors {
            // Frame A: scattered points.
            a.push(x);
            a.push(y);
            // Frame B: same points, shifted by (3, -2).
            b.push(x + 3.0);
            b.push(y - 2.0);
        }
        let mut r = [0.0_f32; 4];
        let mut scratch = ProcrustesScratch::new(8, 2);
        let cfg = ProcrustesConfig {
            center: true,
            ..Default::default()
        };
        let report = orthogonal_procrustes(&a, &b, 8, 2, &mut r, &mut scratch, &cfg)
            .expect("centered Procrustes should succeed");
        // Expected: R = I (no rotation).
        let identity = [1.0, 0.0, 0.0, 1.0];
        assert!(
            matrices_approx_eq(&r, &identity, 1e-2),
            "R mismatch: got {:?}",
            r
        );
        assert!(
            report.residual < 1e-2,
            "residual should be small, got {}",
            report.residual
        );
    }

    #[test]
    fn no_centering_fails_on_offset_frames() {
        // A and B differ by translation only.
        // Without centering, the translation leaks in and residual is non-trivial.
        let anchors: [[f32; 2]; 8] = [
            [1.0, 2.0],
            [-1.5, 0.5],
            [3.0, -1.0],
            [-2.0, -2.0],
            [0.5, 3.5],
            [-3.0, 1.0],
            [2.5, 2.5],
            [-0.5, -3.0],
        ];
        let mut a = Vec::new();
        let mut b = Vec::new();
        for [x, y] in anchors {
            a.push(x);
            a.push(y);
            b.push(x + 3.0);
            b.push(y - 2.0);
        }
        let mut r = [0.0_f32; 4];
        let mut scratch = ProcrustesScratch::new(8, 2);
        let cfg = ProcrustesConfig {
            center: false,
            compute_residual: true,
            ..Default::default()
        };
        let report = orthogonal_procrustes(&a, &b, 8, 2, &mut r, &mut scratch, &cfg)
            .expect("non-centered Procrustes should still run");
        // Residual should be significantly non-zero (translation leaks in).
        assert!(
            report.residual > 0.1,
            "no-center residual should be non-trivial, got {}",
            report.residual
        );
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn result_is_orthogonal() {
        // For any valid input, R^T R ≈ I.
        let a = pseudo_random_anchors(64, 4, 42);
        let b = pseudo_random_anchors(64, 4, 99); // arbitrary — different frame.
        let mut r = [0.0_f32; 16];
        let mut scratch = ProcrustesScratch::new(64, 4);
        let _report = orthogonal_procrustes(
            &a,
            &b,
            64,
            4,
            &mut r,
            &mut scratch,
            &ProcrustesConfig::default(),
        )
        .expect("should succeed");
        // Check R^T R = I.
        let mut rtr = [0.0_f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                let r_col_i = [r[0 * 4 + i], r[1 * 4 + i], r[2 * 4 + i], r[3 * 4 + i]];
                let r_col_j = [r[0 * 4 + j], r[1 * 4 + j], r[2 * 4 + j], r[3 * 4 + j]];
                rtr[i * 4 + j] = simd_dot_f32(&r_col_i, &r_col_j, 4);
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(rtr[i * 4 + j], expected, 1e-2),
                    "R^T R[{},{}] = {} != {}",
                    i,
                    j,
                    rtr[i * 4 + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn deterministic_same_input_same_output() {
        // Same input → same R, bit-identical.
        let a = pseudo_random_anchors(32, 4, 7);
        let b = pseudo_random_anchors(32, 4, 11);
        let mut r1 = [0.0_f32; 16];
        let mut r2 = [0.0_f32; 16];
        let mut scratch1 = ProcrustesScratch::new(32, 4);
        let mut scratch2 = ProcrustesScratch::new(32, 4);
        let _ = orthogonal_procrustes(
            &a,
            &b,
            32,
            4,
            &mut r1,
            &mut scratch1,
            &ProcrustesConfig::default(),
        );
        let _ = orthogonal_procrustes(
            &a,
            &b,
            32,
            4,
            &mut r2,
            &mut scratch2,
            &ProcrustesConfig::default(),
        );
        assert_eq!(r1, r2, "same input must produce bit-identical output");
    }

    #[test]
    fn scratch_reusable_across_calls() {
        // Verify the same scratch works for multiple calls. Use
        // non-degenerate anchors (random in [-1, 1] with n >= 2*d).
        let mut scratch = ProcrustesScratch::new(16, 3);
        for seed in 1..6 {
            // Use non-zero seeds (xorshift32 produces 0 from seed 0).
            let a = pseudo_random_anchors(16, 3, seed);
            let b = pseudo_random_anchors(16, 3, seed + 100);
            let mut r = [0.0_f32; 9];
            let _ = orthogonal_procrustes(
                &a,
                &b,
                16,
                3,
                &mut r,
                &mut scratch,
                &ProcrustesConfig::default(),
            )
            .expect("call should succeed");
        }
    }

    #[test]
    fn determinant_helpers_correct() {
        // 1×1.
        assert_eq!(determinant_d(&[5.0], 1), 5.0);
        // 2×2.
        assert_eq!(determinant_d(&[1.0, 2.0, 3.0, 4.0], 2), -2.0); // 1*4 - 2*3.
        // 3×3 identity → det = 1.
        let d3 = determinant_d(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3);
        assert!(
            approx_eq(d3, 1.0, 1e-6),
            "3×3 identity det should be 1, got {}",
            d3
        );
    }

    #[test]
    fn special_orthogonal_correction_flips_when_needed() {
        // Construct a scenario where det(R) = -1 (reflection).
        // Take a frame and its mirror image.
        let a = [1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0_f32];
        // B = reflect A across x-axis: (x, y) → (x, -y). This is a reflection
        // (det = -1).
        let b = [1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0_f32];
        let cfg = ProcrustesConfig {
            special_orthogonal: true,
            compute_det: true,
            ..Default::default()
        };
        let mut r = [0.0_f32; 4];
        let mut scratch = ProcrustesScratch::new(4, 2);
        let report = orthogonal_procrustes(&a, &b, 4, 2, &mut r, &mut scratch, &cfg)
            .expect("should succeed with correction");
        assert!(report.flipped, "should have flipped to enforce det=+1");
        assert!(
            report.det > 0.0,
            "det should be positive after flip, got {}",
            report.det
        );
    }
}
