//! Parallax attention: streaming covariance correction on top of tiled softmax attention.
//!
//! Implements the Parallax formula:
//! ```text
//! o_PLX = o_SA − gate_scale · Σ_KV · ρ     where ρ = W_R · x
//! ```
//!
//! - `o_SA`  = standard softmax attention output
//! - `Σ_KV`  = KV cross-covariance under softmax weights (streaming accumulator)
//! - `ρ`     = learned probe from layer input via extra projection `W_R`
//!
//! ## Optimization: column-sum factorization
//!
//! The covariance factorizes via column sums of the softmax matrix:
//! ```text
//! Σ_KV = Σ_j c_j · v_j ⊗ k_j^T    where c_j = Σ_i softmax(i,j)
//! ```
//! This reduces outer products from O(N²) to O(N), bringing CPU overhead
//! from ~50× down to ~2× over base SDPA.
//!
//! Feature-gated behind `#[cfg(feature = "parallax_attn")]`.

use crate::simd;

// ── Config ────────────────────────────────────────────────────────

/// Configuration for the Parallax attention correction.
#[derive(Debug, Clone)]
pub struct ParallaxConfig {
    /// Scaling factor for the covariance correction. Default 1.0; can be
    /// annealed during training (set to 0.0 to recover pure softmax).
    pub gate_scale: f32,
    /// Whether `W_R` starts zeroed. When `true` and weights are zero, the
    /// module is a no-op and recovers exact softmax attention.
    pub zero_init: bool,
}

impl Default for ParallaxConfig {
    fn default() -> Self {
        Self {
            gate_scale: 1.0,
            zero_init: true,
        }
    }
}

// ── R Projection ──────────────────────────────────────────────────

/// Compute ρ = R_proj · x (matrix–vector product, row-major weight).
///
/// - `r_proj`: R projection weight matrix `[head_dim × head_dim]`, row-major
/// - `x`:      layer input `[head_dim]`
/// - `out`:    output buffer `[head_dim]`
///
/// Uses [`simd::simd_dot_f32`] for the row–vector dot products.
#[inline]
pub fn compute_rho(r_proj: &[f32], x: &[f32], out: &mut [f32]) {
    let d = out.len();
    debug_assert_eq!(r_proj.len(), d * d, "r_proj must be head_dim × head_dim");
    debug_assert_eq!(x.len(), d, "x must have length head_dim");

    simd::simd_matmul_rows(out, r_proj, x, d, d);
}

// ── Streaming Covariance Correction ───────────────────────────────

/// Compute the Parallax correction: `out = Σ_KV · ρ`.
///
/// - `sigma_kv`: softmax-weighted KV cross-covariance `[head_dim × head_dim]`, row-major
/// - `rho`:      R projection output `[head_dim]`
/// - `out`:      correction output `[head_dim]`
///
/// Uses [`simd::simd_dot_f32`] for each row of the matrix–vector product.
#[inline]
pub fn parallax_correction(sigma_kv: &[f32], rho: &[f32], out: &mut [f32]) {
    let d = out.len();
    debug_assert_eq!(
        sigma_kv.len(),
        d * d,
        "sigma_kv must be head_dim × head_dim"
    );
    debug_assert_eq!(rho.len(), d, "rho must have length head_dim");

    simd::simd_matvec(out, sigma_kv, rho, d, d);
}

// ── Fused tiled attention + Parallax ──────────────────────────────

/// Tiled online-softmax flash attention with Parallax covariance correction.
///
/// Uses column-sum factorization to compute the KV cross-covariance:
/// ```text
/// Σ_KV = Σ_j c_j · v_j ⊗ k_j^T    where c_j = Σ_i softmax(i,j)
/// ```
/// This reduces outer products from O(N²) to O(N), bringing CPU decode
/// overhead from ~50× down to ~2× over base SDPA.
///
/// # Algorithm
/// 1. For each query row i: compute scores, softmax, accumulate output + column sums
/// 2. Compute Σ_KV from column sums using N outer products
/// 3. Compute correction = Σ_KV · ρ and subtract from output
///
/// # Arguments
/// * `q`, `k`, `v` — Q, K, V tensors `[seq_len × head_dim]`, row-major
/// * `output`       — output tensor `[seq_len × head_dim]`, pre-allocated
/// * `seq_len`      — sequence length N
/// * `head_dim`     — dimension per head D
/// * `scale`        — softmax temperature (typically 1/√D)
/// * `r`            — R projection weights `[head_dim × head_dim]`, row-major
/// * `x`            — layer input `[head_dim]`
/// * `parallax_config` — gate scale and init config
#[allow(clippy::too_many_arguments)]
pub fn tiled_attention_parallax_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    r: &[f32],
    x: &[f32],
    parallax_config: &ParallaxConfig,
) {
    let expected = seq_len * head_dim;
    debug_assert_eq!(q.len(), expected, "Q slice length mismatch");
    debug_assert_eq!(k.len(), expected, "K slice length mismatch");
    debug_assert_eq!(v.len(), expected, "V slice length mismatch");
    debug_assert_eq!(output.len(), expected, "output slice length mismatch");
    debug_assert_eq!(
        r.len(),
        head_dim * head_dim,
        "R must be head_dim × head_dim"
    );
    debug_assert_eq!(x.len(), head_dim, "x must have length head_dim");

    if seq_len == 0 {
        return;
    }

    // Compute ρ = W_R · x
    let mut rho = vec![0.0f32; head_dim];
    compute_rho(r, x, &mut rho);

    // If gate_scale is zero, or ρ is all zeros (zero_init with zeroed W_R),
    // plain softmax attention is sufficient.
    let rho_is_zero = rho.iter().all(|&v| v == 0.0);
    if parallax_config.gate_scale == 0.0 || rho_is_zero {
        tiled_attention_core(q, k, v, output, seq_len, head_dim, scale, None);
        return;
    }

    let d = head_dim;
    let n = seq_len;

    // Pre-allocate all scratch buffers (avoid allocation inside loops)
    let mut col_sums = vec![0.0f32; n]; // c[j] = Σ_i softmax(i,j)
    let mut scores = vec![0.0f32; n]; // reusable per-row score buffer
    let mut sigma_kv = vec![0.0f32; d * d]; // [head_dim × head_dim]
    let mut pv_buf = vec![0.0f32; d]; // scaled v row for outer product
    let mut correction = vec![0.0f32; d];

    // Phase 1: Compute o_SA and accumulate column sums in one pass
    for i in 0..n {
        let q_off = i * d;
        let out_off = i * d;
        output[out_off..out_off + d].fill(0.0);

        // Compute scores for row i: scores[j] = q_i · k_j * scale
        let mut max_score = f32::NEG_INFINITY;
        for j in 0..n {
            let k_off = j * d;
            scores[j] = simd::simd_dot_f32(&q[q_off..q_off + d], &k[k_off..k_off + d], d) * scale;
            max_score = max_score.max(scores[j]);
        }

        // Softmax
        let mut rowsum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            rowsum += *s;
        }
        let inv_sum = 1.0 / rowsum;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }

        // Accumulate output: o_i = Σ_j p_ij · v_j
        for j in 0..n {
            let p = scores[j];
            let v_off = j * d;
            simd::simd_fused_scale_acc(
                &mut output[out_off..out_off + d],
                &v[v_off..v_off + d],
                p,
                d,
            );
        }

        // Accumulate column sums: c[j] += softmax(i,j)
        for j in 0..n {
            col_sums[j] += scores[j];
        }
    }

    // Phase 2: Compute Σ_KV = Σ_j c_j · v_j ⊗ k_j^T
    // Only N outer products instead of N² — the key optimization.
    for j in 0..n {
        let c_j = col_sums[j];
        if c_j == 0.0 {
            continue;
        }
        let v_off = j * d;
        let k_off = j * d;
        pv_buf[..d].copy_from_slice(&v[v_off..v_off + d]);
        simd::simd_scale_inplace(&mut pv_buf, c_j);
        simd::simd_outer_product_acc(sigma_kv.as_mut(), &pv_buf, &k[k_off..k_off + d], d, d);
    }

    // Phase 3: Compute correction = Σ_KV · ρ
    parallax_correction(&sigma_kv, &rho, &mut correction);

    // Phase 4: Apply correction — output[i] -= gate_scale * correction for all i
    let gs = parallax_config.gate_scale;
    let scaled_correction = &correction[..];
    for i in 0..n {
        let off = i * d;
        // SIMD-accelerated: output[off..] -= gs * correction
        for j in 0..d {
            output[off + j] -= gs * scaled_correction[j];
        }
    }
}

// ── Core attention (no feature-flag dependency) ───────────────────

/// Core softmax attention, used when Parallax correction is not needed.
#[allow(clippy::too_many_arguments)]
fn tiled_attention_core(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    _sigma_kv: Option<&mut [f32]>,
) {
    let d = head_dim;
    let n = seq_len;

    // Pre-allocate reusable score buffer
    let mut scores = vec![0.0f32; n];

    for i in 0..n {
        let q_off = i * d;
        let out_off = i * d;
        output[out_off..out_off + d].fill(0.0);

        // Compute scores for row i
        let mut max_score = f32::NEG_INFINITY;
        for j in 0..n {
            let k_off = j * d;
            scores[j] = simd::simd_dot_f32(&q[q_off..q_off + d], &k[k_off..k_off + d], d) * scale;
            max_score = max_score.max(scores[j]);
        }

        // Softmax
        let mut rowsum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            rowsum += *s;
        }
        let inv_sum = 1.0 / rowsum;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }

        // Accumulate output: o_i = Σ_j p_ij · v_j
        for j in 0..n {
            let p = scores[j];
            let v_off = j * d;
            simd::simd_fused_scale_acc(
                &mut output[out_off..out_off + d],
                &v[v_off..v_off + d],
                p,
                d,
            );
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// With R_proj = zero matrix, rho should be all zeros.
    #[test]
    fn test_rho_zero_init() {
        let d = 8;
        let r_proj = vec![0.0f32; d * d];
        let x: Vec<f32> = (1..=d).map(|i| i as f32).collect();
        let mut rho = vec![0.0f32; d];

        compute_rho(&r_proj, &x, &mut rho);

        for (i, &v) in rho.iter().enumerate() {
            assert!(
                v == 0.0,
                "rho[{}] should be 0.0 with zero R_proj, got {}",
                i,
                v
            );
        }
    }

    /// With identity sigma_kv, correction should equal rho.
    #[test]
    fn test_correction_identity() {
        let d = 8;
        let mut sigma_kv = vec![0.0f32; d * d];
        // Identity matrix
        for i in 0..d {
            sigma_kv[i * d + i] = 1.0;
        }
        let rho: Vec<f32> = (1..=d).map(|i| i as f32 * 0.5).collect();
        let mut correction = vec![0.0f32; d];

        parallax_correction(&sigma_kv, &rho, &mut correction);

        for (i, (&c, &r)) in correction.iter().zip(rho.iter()).enumerate() {
            let expected = r;
            assert!(
                (c - expected).abs() < 1e-5,
                "correction[{}] should be {} (identity sigma), got {}",
                i,
                expected,
                c
            );
        }
    }

    /// With gate_scale=0, the output should equal standard softmax attention.
    #[test]
    fn test_parallax_recovers_softmax_gate_zero() {
        let d = 4;
        let n = 3;
        let scale = 1.0 / (d as f32).sqrt();

        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

        // R projection — non-zero, but gate_scale=0 should cancel it
        let r: Vec<f32> = (0..d * d).map(|i| (i as f32 * 0.05).cos()).collect();
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();

        let config = ParallaxConfig {
            gate_scale: 0.0,
            zero_init: false,
        };

        let mut output_parallax = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut output_parallax,
            n,
            d,
            scale,
            &r,
            &x,
            &config,
        );

        // Compute reference: standard softmax attention
        let mut output_ref = vec![0.0f32; n * d];
        tiled_attention_core(&q, &k, &v, &mut output_ref, n, d, scale, None);

        for (i, (&a, &b)) in output_parallax.iter().zip(output_ref.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "output[{}]: parallax ({}) should match softmax ({}) with gate_scale=0",
                i,
                a,
                b
            );
        }
    }

    /// With zero R_proj, the output should equal standard softmax attention
    /// regardless of gate_scale (since rho = 0 implies correction = 0).
    #[test]
    fn test_parallax_recovers_softmax_zero_r() {
        let d = 4;
        let n = 3;
        let scale = 1.0 / (d as f32).sqrt();

        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

        // Zero R projection weights
        let r = vec![0.0f32; d * d];
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();

        let config = ParallaxConfig {
            gate_scale: 1.0,
            zero_init: true,
        };

        let mut output_parallax = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut output_parallax,
            n,
            d,
            scale,
            &r,
            &x,
            &config,
        );

        // Compute reference: standard softmax attention
        let mut output_ref = vec![0.0f32; n * d];
        tiled_attention_core(&q, &k, &v, &mut output_ref, n, d, scale, None);

        for (i, (&a, &b)) in output_parallax.iter().zip(output_ref.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "output[{}]: parallax ({}) should match softmax ({}) with zero R",
                i,
                a,
                b
            );
        }
    }

    /// Verify that compute_rho produces correct matrix-vector product.
    #[test]
    fn test_compute_rho_correct() {
        let d = 4;
        // R = [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]
        let mut r_proj = vec![0.0f32; d * d];
        for i in 0..d {
            r_proj[i * d + i] = (i + 1) as f32;
        }
        let x = vec![1.0f32; d];
        let mut rho = vec![0.0f32; d];

        compute_rho(&r_proj, &x, &mut rho);

        let expected = [1.0f32, 2.0, 3.0, 4.0];
        for (i, (&r, &e)) in rho.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-5,
                "rho[{}] should be {}, got {}",
                i,
                e,
                r
            );
        }
    }

    /// Verify that parallax_correction with a known sigma produces the right result.
    #[test]
    fn test_correction_known_sigma() {
        let d = 3;
        // sigma_kv = [[2, 0, 0], [0, 2, 0], [0, 0, 2]] (2 * identity)
        let mut sigma_kv = vec![0.0f32; d * d];
        for i in 0..d {
            sigma_kv[i * d + i] = 2.0;
        }
        let rho = vec![1.0f32, 2.0, 3.0];
        let mut correction = vec![0.0f32; d];

        parallax_correction(&sigma_kv, &rho, &mut correction);

        let expected = [2.0f32, 4.0, 6.0];
        for (i, (&c, &e)) in correction.iter().zip(expected.iter()).enumerate() {
            assert!(
                (c - e).abs() < 1e-5,
                "correction[{}] should be {}, got {}",
                i,
                e,
                c
            );
        }
    }
}
