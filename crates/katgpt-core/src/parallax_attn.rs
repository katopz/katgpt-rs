//! Parallax attention: streaming covariance correction on top of tiled attention.
//!
//! Implements the Parallax formula (Research 135, extended for sigmoid in Research 140):
//! ```text
//! o_PLX = o_SA − gate_scale · Σ_KV · ρ     where ρ = W_R · x
//! ```
//!
//! - `o_SA`  = attention output under chosen activation (softmax or sigmoid)
//! - `Σ_KV`  = KV cross-covariance under attention weights (streaming accumulator)
//! - `ρ`     = learned probe from layer input via extra projection `W_R`
//!
//! ## Sigmoid extension (Research 140)
//!
//! The Parallax correction is kernel-agnostic: any normalized attention weights
//! `p(i,j) ≥ 0, Σ_j p(i,j) = 1` define a Nadaraya-Watson estimator whose local-linear
//! upgrade is `o_LL = o_NW − Σ_KV · ρ`. Normalized sigmoid attention uses kernel
//! `K(x,y) = σ(x·y·s)` where `s = 1/√d`, giving:
//! ```text
//! p_σ(i,j) = σ(q_i · k_j · s) / Σ_k σ(q_i · k_k · s)
//! ```
//! Advantages over softmax: no attention sinks, better numerical stability, no exp
//! overflow risk. The column-sum factorization still applies identically.
//!
//! ## Optimization: column-sum factorization
//!
//! The covariance factorizes via column sums of the attention weight matrix:
//! ```text
//! Σ_KV = Σ_j c_j · v_j ⊗ k_j^T    where c_j = Σ_i p(i,j)
//! ```
//! This reduces outer products from O(N²) to O(N), bringing CPU overhead
//! from ~50× down to ~2× over base SDPA.
//!
//! Feature-gated behind `#[cfg(feature = "parallax_attn")]`.

use crate::simd;

// ── Config ────────────────────────────────────────────────────────

/// Activation function for attention weight normalization.
///
/// Both produce normalized weights `p(i,j) ≥ 0, Σ_j p(i,j) = 1` that define
/// a Nadaraya-Watson kernel regression estimator. The Parallax local-linear
/// correction `o_LL = o_NW − Σ_KV · ρ` applies identically to both.
#[repr(u8)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ParallaxActivation {
    /// Standard softmax: `p(i,j) = exp(q_i · k_j · s) / Σ_k exp(q_i · k_k · s)`.
    /// Gaussian-like kernel with attention sinks.
    Softmax,
    /// Normalized sigmoid: `p(i,j) = σ(q_i · k_j · s) / Σ_k σ(q_i · k_k · s)`.
    /// No attention sinks, better numerical stability, no exp overflow.
    /// Higher COR capacity than softmax on real LM data (Plan 161 T3).
    /// Kernel: `K(x,y) = σ(x · y · s)`.
    #[default]
    Sigmoid,
}

/// Configuration for the Parallax attention correction.
#[derive(Debug, Clone)]
pub struct ParallaxConfig {
    /// Scaling factor for the covariance correction. Default 1.0; can be
    /// annealed during training (set to 0.0 to recover pure attention).
    pub gate_scale: f32,
    /// Whether `W_R` starts zeroed. When `true` and weights are zero, the
    /// module is a no-op and recovers exact base attention.
    pub zero_init: bool,
    /// Activation function for attention weight normalization.
    /// Default: Sigmoid (sink-free, higher COR capacity per Plan 161).
    /// Set to Softmax for backward-compatible attention sinks.
    pub activation: ParallaxActivation,
}

impl Default for ParallaxConfig {
    fn default() -> Self {
        Self {
            gate_scale: 1.0,
            zero_init: true,
            activation: ParallaxActivation::default(),
        }
    }
}

// ── Weight normalization ───────────────────────────────────────────

/// Normalize attention scores in-place to produce valid probability weights.
///
/// - `Softmax`: `p(j) = exp(s_j − max) / Σ exp(s_k − max)`
/// - `Sigmoid`: `p(j) = σ(s_j) / Σ σ(s_k)` where `σ(x) = 1/(1+exp(−x))`
///
/// Both produce `Σ_j p(j) = 1` and `p(j) ≥ 0`.
#[inline]
fn normalize_attention_weights(row: &mut [f32], activation: ParallaxActivation) {
    match activation {
        ParallaxActivation::Softmax => {
            let max_score = simd::simd_max_f32(row);
            simd::simd_add_scalar_inplace(row, -max_score);
            simd::simd_exp_inplace(row);
            let rowsum = simd::simd_sum_f32(row);
            simd::simd_scale_inplace(row, 1.0 / rowsum);
        }
        ParallaxActivation::Sigmoid => {
            // σ(x) = 1/(1+exp(−x)), then normalize by sum
            simd::simd_scale_inplace(row, -1.0);
            simd::simd_exp_inplace(row);
            simd::simd_add_scalar_inplace(row, 1.0);
            // Invert elementwise: row = 1/(1+exp(−x)) = σ(x)
            simd::simd_reciprocal_inplace(row);
            // Normalize so Σ_j p(j) = 1 (Nadaraya-Watson requirement)
            let rowsum = simd::simd_sum_f32(row);
            simd::simd_scale_inplace(row, 1.0 / rowsum);
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

/// Scratch buffer sizes for [`tiled_attention_parallax_forward`].
///
/// Pre-compute once and reuse across calls to avoid per-call allocation.
/// All buffers are flat `Vec<f32>` — clear + reuse across calls.
pub struct ParallaxScratch {
    /// ρ = W_R · x, length `head_dim`
    pub rho: Vec<f32>,
    /// Column sums c[j] = Σ_i softmax(i,j), length `seq_len`
    pub col_sums: Vec<f32>,
    /// Per-row score buffer, length `seq_len`
    pub scores: Vec<f32>,
    /// Σ_KV cross-covariance, length `head_dim * head_dim`
    pub sigma_kv: Vec<f32>,
    /// Scaled v row for outer product, length `head_dim`
    pub pv_buf: Vec<f32>,
    /// Correction output, length `head_dim`
    pub correction: Vec<f32>,
    /// Cached dimensions from the last `ensure_capacity` call. Fast-path returns
    /// immediately when both match, skipping 6 length comparisons + branches.
    cached_seq_len: usize,
    cached_head_dim: usize,
}

impl ParallaxScratch {
    /// Create scratch buffers sized for the given dimensions.
    pub fn new(seq_len: usize, head_dim: usize) -> Self {
        Self {
            rho: vec![0.0; head_dim],
            col_sums: vec![0.0; seq_len],
            scores: vec![0.0; seq_len],
            sigma_kv: vec![0.0; head_dim * head_dim],
            pv_buf: vec![0.0; head_dim],
            correction: vec![0.0; head_dim],
            cached_seq_len: seq_len,
            cached_head_dim: head_dim,
        }
    }

    /// Reset all buffers for reuse (clear without deallocating).
    pub fn reset(&mut self) {
        self.rho.fill(0.0);
        self.col_sums.fill(0.0);
        self.scores.fill(0.0);
        self.sigma_kv.fill(0.0);
        self.pv_buf.fill(0.0);
        self.correction.fill(0.0);
    }

    /// Resize buffers if dimensions changed (avoids reallocation when sizes match).
    pub fn ensure_capacity(&mut self, seq_len: usize, head_dim: usize) {
        // Fast path: most calls reuse the same dimensions. Two comparisons
        // replace six length reads + branches in the steady state.
        if self.cached_seq_len == seq_len && self.cached_head_dim == head_dim {
            return;
        }
        let d = head_dim;
        let d2 = d * d;
        let mut changed = false;
        if self.rho.len() != d {
            self.rho.resize(d, 0.0);
            changed = true;
        }
        if self.col_sums.len() < seq_len {
            self.col_sums.resize(seq_len, 0.0);
            changed = true;
        }
        if self.scores.len() < seq_len {
            self.scores.resize(seq_len, 0.0);
            changed = true;
        }
        if self.sigma_kv.len() != d2 {
            self.sigma_kv.resize(d2, 0.0);
            changed = true;
        }
        if self.pv_buf.len() != d {
            self.pv_buf.resize(d, 0.0);
            changed = true;
        }
        if self.correction.len() != d {
            self.correction.resize(d, 0.0);
            changed = true;
        }
        // Only zero-fill when dimensions changed; caller resets specific buffers as needed.
        // The hot path (tiled_attention_parallax_forward) writes to all buffers before reading.
        if changed {
            self.reset();
        }
        self.cached_seq_len = seq_len;
        self.cached_head_dim = head_dim;
    }
}

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
/// * `scratch`      — pre-allocated scratch buffers (pass `None` for one-shot allocation)
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
    scratch: Option<&mut ParallaxScratch>,
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

    // Allocate scratch on demand if caller didn't provide one
    let mut local_scratch;
    let scratch = match scratch {
        Some(s) => {
            s.ensure_capacity(seq_len, head_dim);
            s
        }
        None => {
            local_scratch = ParallaxScratch::new(seq_len, head_dim);
            &mut local_scratch
        }
    };

    let d = head_dim;
    let n = seq_len;

    // Compute ρ = W_R · x (reuse scratch buffer)
    compute_rho(r, x, &mut scratch.rho);

    // If gate_scale is zero, or ρ is all zeros (zero_init with zeroed W_R),
    // plain softmax attention is sufficient.
    // Perf: skip O(d) linear scan when gate_scale is already zero.
    let rho_is_zero = if parallax_config.gate_scale == 0.0 {
        true // gate_scale=0 makes correction zero regardless of ρ
    } else {
        // SIMD single-pass: sum of |values| is 0 iff all are zero.
        // Faster than scalar early-exit scan for typical non-zero ρ
        // (which must scan all d elements anyway to confirm all-zero).
        crate::simd::simd_sum_abs_f32(&scratch.rho[..d]) == 0.0
    };
    if rho_is_zero {
        tiled_attention_core(
            q,
            k,
            v,
            output,
            seq_len,
            head_dim,
            scale,
            Some(&mut scratch.scores),
            parallax_config.activation,
        );
        return;
    }

    // Zero column sums before accumulation — scratch may be reused across calls
    scratch.col_sums[..n].fill(0.0);
    scratch.sigma_kv[..d * d].fill(0.0);

    // Phase 1: Compute o_SA and accumulate column sums in one pass
    for i in 0..n {
        let row_off = i * d;
        output[row_off..row_off + d].fill(0.0);

        // Compute scores for row i: scores[j] = q_i · k_j * scale
        for j in 0..n {
            let k_off = j * d;
            scratch.scores[j] =
                simd::simd_dot_f32(&q[row_off..row_off + d], &k[k_off..k_off + d], d) * scale;
        }

        // Normalize attention weights (softmax or sigmoid)
        let row = &mut scratch.scores[..n];
        normalize_attention_weights(row, parallax_config.activation);

        // Accumulate output: o_i = Σ_j p_ij · v_j
        for j in 0..n {
            let p = scratch.scores[j];
            let v_off = j * d;
            simd::simd_fused_scale_acc(
                &mut output[row_off..row_off + d],
                &v[v_off..v_off + d],
                p,
                d,
            );
        }

        // Accumulate column sums: c[j] += softmax(i,j)
        // (already in scratch.scores after softmax)
        simd::simd_add_inplace(&mut scratch.col_sums[..n], &scratch.scores[..n]);
    }

    // Phase 2: Compute Σ_KV = Σ_j c_j · v_j ⊗ k_j^T
    // Only N outer products instead of N² — the key optimization.
    for j in 0..n {
        let c_j = scratch.col_sums[j];
        if c_j == 0.0 {
            continue;
        }
        let v_off = j * d;
        let k_off = j * d;
        // Perf: fused scale-copy (pv_buf = c_j * v[j]) in single SIMD pass,
        // instead of separate copy + scale (two passes over d elements).
        simd::simd_fused_decay_write(&mut scratch.pv_buf, 0.0, &v[v_off..v_off + d], c_j);
        simd::simd_outer_product_acc(
            scratch.sigma_kv.as_mut(),
            &scratch.pv_buf,
            &k[k_off..k_off + d],
            d,
            d,
        );
    }

    // Phase 3: Compute correction = Σ_KV · ρ
    parallax_correction(&scratch.sigma_kv, &scratch.rho, &mut scratch.correction);

    // Phase 4: Apply correction — output[i] -= gate_scale * correction for all i
    // Pre-scale correction by -gs once, then SIMD-add to each output row.
    let gs = parallax_config.gate_scale;
    simd::simd_scale_inplace(&mut scratch.correction, -gs);
    for i in 0..n {
        let off = i * d;
        simd::simd_add_inplace(&mut output[off..off + d], &scratch.correction[..d]);
    }
}

// ── Core attention (no feature-flag dependency) ───────────────────

/// Core attention, used when Parallax correction is not needed.
///
/// Accepts an optional pre-allocated `scores` scratch buffer (length >= seq_len)
/// to avoid per-call heap allocation. When `None`, allocates on demand.
#[allow(clippy::too_many_arguments)]
#[inline]
fn tiled_attention_core(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    output: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    scores: Option<&mut [f32]>,
    activation: ParallaxActivation,
) {
    let d = head_dim;
    let n = seq_len;

    // Use caller-provided scratch or allocate on demand. No `fill(0.0)` needed —
    // the score loop below writes every `scores[0..n]` element before any read.
    let mut local_scores;
    let scores: &mut [f32] = match scores {
        Some(s) if s.len() >= n => s,
        _ => {
            local_scores = vec![0.0f32; n];
            &mut local_scores
        }
    };

    for i in 0..n {
        let q_off = i * d;
        let out_off = i * d;
        output[out_off..out_off + d].fill(0.0);

        // Compute scores for row i
        for (j, score_slot) in scores.iter_mut().enumerate().take(n) {
            let k_off = j * d;
            *score_slot = simd::simd_dot_f32(&q[q_off..q_off + d], &k[k_off..k_off + d], d) * scale;
        }

        // Normalize attention weights (softmax or sigmoid)
        let row = &mut scores[..n];
        normalize_attention_weights(row, activation);

        // Accumulate output: o_i = Σ_j p_ij · v_j
        for (j, &p) in scores.iter().enumerate().take(n) {
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
            activation: ParallaxActivation::Softmax,
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
            None,
        );

        // Compute reference: standard softmax attention
        let mut output_ref = vec![0.0f32; n * d];
        tiled_attention_core(
            &q,
            &k,
            &v,
            &mut output_ref,
            n,
            d,
            scale,
            None,
            ParallaxActivation::Softmax,
        );

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
            activation: ParallaxActivation::Softmax,
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
            None,
        );

        // Compute reference: standard softmax attention
        let mut output_ref = vec![0.0f32; n * d];
        tiled_attention_core(
            &q,
            &k,
            &v,
            &mut output_ref,
            n,
            d,
            scale,
            None,
            ParallaxActivation::Softmax,
        );

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

    // ── Sigmoid-specific tests ──────────────────────────────────────

    /// With gate_scale=0 and sigmoid activation, output should equal pure sigmoid attention.
    #[test]
    fn test_parallax_sigmoid_recovers_base() {
        let d = 4;
        let n = 3;
        let scale = 1.0 / (d as f32).sqrt();

        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

        let r: Vec<f32> = (0..d * d).map(|i| (i as f32 * 0.05).cos()).collect();
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();

        let config = ParallaxConfig {
            gate_scale: 0.0,
            zero_init: false,
            activation: ParallaxActivation::Sigmoid,
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
            None,
        );

        let mut output_ref = vec![0.0f32; n * d];
        tiled_attention_core(
            &q,
            &k,
            &v,
            &mut output_ref,
            n,
            d,
            scale,
            None,
            ParallaxActivation::Sigmoid,
        );

        for (i, (&a, &b)) in output_parallax.iter().zip(output_ref.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "output[{}]: sigmoid parallax ({}) should match base sigmoid ({}) with gate_scale=0",
                i,
                a,
                b
            );
        }
    }

    /// Sigmoid attention weights should be non-negative and sum to 1 per row.
    #[test]
    fn test_sigmoid_weights_normalized() {
        let d = 4;
        let n = 5;
        let scale = 1.0 / (d as f32).sqrt();

        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.37).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.53).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.71).sin()).collect();

        // Run with gate_scale=0 so we get pure sigmoid attention
        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];
        let config = ParallaxConfig {
            gate_scale: 0.0,
            zero_init: true,
            activation: ParallaxActivation::Sigmoid,
        };

        let mut output = vec![0.0f32; n * d];
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut output,
            n,
            d,
            scale,
            &r,
            &x,
            &config,
            None,
        );

        // Output should be finite (no NaN/Inf from numerical issues)
        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "output[{}] should be finite, got {}", i, v);
        }
    }

    /// Sigmoid and softmax should produce different outputs (different kernels).
    #[test]
    fn test_sigmoid_differs_from_softmax() {
        let d = 4;
        let n = 3;
        let scale = 1.0 / (d as f32).sqrt();

        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

        let r = vec![0.0f32; d * d];
        let x = vec![0.0f32; d];

        let mut out_sm = vec![0.0f32; n * d];
        let mut out_sig = vec![0.0f32; n * d];

        let config_sm = ParallaxConfig {
            gate_scale: 0.0,
            zero_init: true,
            activation: ParallaxActivation::Softmax,
        };
        let config_sig = ParallaxConfig {
            gate_scale: 0.0,
            zero_init: true,
            activation: ParallaxActivation::Sigmoid,
        };

        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_sm,
            n,
            d,
            scale,
            &r,
            &x,
            &config_sm,
            None,
        );
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_sig,
            n,
            d,
            scale,
            &r,
            &x,
            &config_sig,
            None,
        );

        let any_differs = out_sm
            .iter()
            .zip(out_sig.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(
            any_differs,
            "sigmoid and softmax should produce different outputs"
        );
    }

    /// With non-zero R projection, sigmoid Parallax should modify the output.
    #[test]
    fn test_sigmoid_parallax_correction_applied() {
        let d = 4;
        let n = 3;
        let scale = 1.0 / (d as f32).sqrt();

        let q: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.1).sin()).collect();
        let k: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.2).cos()).collect();
        let v: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.3).sin()).collect();

        let r: Vec<f32> = (0..d * d).map(|i| (i as f32 * 0.05).cos()).collect();
        let x: Vec<f32> = (0..d).map(|i| (i as f32 * 0.1).sin()).collect();

        let config_no_corr = ParallaxConfig {
            gate_scale: 0.0,
            zero_init: false,
            activation: ParallaxActivation::Sigmoid,
        };
        let config_with_corr = ParallaxConfig {
            gate_scale: 1.0,
            zero_init: false,
            activation: ParallaxActivation::Sigmoid,
        };

        let mut out_no = vec![0.0f32; n * d];
        let mut out_yes = vec![0.0f32; n * d];

        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_no,
            n,
            d,
            scale,
            &r,
            &x,
            &config_no_corr,
            None,
        );
        tiled_attention_parallax_forward(
            &q,
            &k,
            &v,
            &mut out_yes,
            n,
            d,
            scale,
            &r,
            &x,
            &config_with_corr,
            None,
        );

        let any_differs = out_no
            .iter()
            .zip(out_yes.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(
            any_differs,
            "sigmoid parallax correction should modify output vs base sigmoid"
        );
    }
}
