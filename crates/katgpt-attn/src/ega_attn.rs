//! Energy-Gated Attention (EGA) — Spectral salience gating for attention.
//!
//! Feature gate: `ega_attn` (Plan 139, opt-in).
//!
//! Gates value aggregation by the spectral energy of key token embeddings.
//! Each key position's attention weight is scaled by a learned sigmoid gate
//! derived from the dot-product energy of the input embedding with a learned
//! projection vector.
//!
//! # Paper Algorithm (Algorithm 1)
//!
//! ```text
//! Q, K, V ← XW_Q, XW_K, XW_V
//! S ← QKᵀ/√d + causal_mask
//! A ← softmax(S)
//!
//! e ← X · w_proj                    // [seq_len] energy scores
//! ẽ ← (e - μ) / (σ + ε)             // z-normalize
//! g ← σ(α · (ẽ - τ))                // sigmoid gate [seq_len]
//!
//! Âᵢⱼ ← Aᵢⱼ · gⱼ                   // gate each key position
//! Âᵢⱼ ← Âᵢⱼ / Σₖ(Âᵢₖ + ε)          // renormalize (sum-to-one)
//! Y ← Â · V                         // value aggregation
//! ```
//!
//! # Parameter overhead
//!
//! Per attention head: `d + 2` parameters (`w_proj`: d, `alpha`: 1, `tau`: 1).
//! Paper converges to α ≈ 2.2, τ ≈ 0.35.

/// Numerical epsilon for division safety.
const EPS: f32 = 1e-8;

// ── Helper functions ──────────────────────────────────────────

/// Standard sigmoid: σ(x) = 1 / (1 + exp(-x)).
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Z-normalize scores in-place: (x - μ) / (σ + ε).
///
/// Handles the degenerate case where all values are equal (σ → 0)
/// by producing all-zero output.
/// Uses SIMD for sum-of-squares computation.
#[inline]
pub fn z_normalize(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let n = scores.len() as f32;
    let mean = katgpt_core::simd::simd_sum_f32(scores) / n;
    // Subtract mean in-place via SIMD
    katgpt_core::simd::simd_add_scalar_inplace(scores, -mean);
    // Compute variance via SIMD sum-of-squares
    let variance = katgpt_core::simd::simd_sum_sq(scores, scores.len()) / n;
    let std_dev = variance.sqrt() + EPS;
    katgpt_core::simd::simd_scale_inplace(scores, 1.0 / std_dev);
}

/// Compute the energy gate vector g from energy scores.
///
/// Returns g[j] = σ(α · (ẽ[j] - τ)) where ẽ is the z-normalized energy.
pub fn compute_energy_gate(energy: &[f32], alpha: f32, tau: f32) -> Vec<f32> {
    let mut out = vec![0.0; energy.len()];
    compute_energy_gate_into(energy, alpha, tau, &mut out);
    out
}

/// Zero-alloc variant of [`compute_energy_gate`].
///
/// Writes the gate vector into `out[..energy.len()]`.
#[inline]
pub fn compute_energy_gate_into(energy: &[f32], alpha: f32, tau: f32, out: &mut [f32]) {
    let len = energy.len();
    out[..len].copy_from_slice(energy);
    z_normalize(&mut out[..len]);
    // Compute alpha * (z - tau) = alpha*z - alpha*tau in-place
    katgpt_core::simd::simd_add_scalar_inplace(&mut out[..len], -tau);
    katgpt_core::simd::simd_scale_inplace(&mut out[..len], alpha);
    // sigmoid(x) = 1/(1+exp(-x)). Negate then exp for SIMD-friendly batch exp.
    katgpt_core::simd::simd_scale_inplace(&mut out[..len], -1.0);
    katgpt_core::simd::simd_exp_inplace(&mut out[..len]);
    // Branch-free reciprocal: out[i] = 1.0 / (1.0 + out[i])
    // SIMD-accelerated +1.0, then scalar reciprocal (LLVM auto-vectorizes the branch-free div).
    katgpt_core::simd::simd_add_scalar_inplace(&mut out[..len], 1.0);
    for o in out[..len].iter_mut() {
        *o = 1.0 / *o;
    }
}

// ── EgaGate ───────────────────────────────────────────────────

/// Energy-Gated Attention parameters per attention head.
///
/// Adds `d + 2` learnable parameters per head:
/// - `w_proj` (d): energy projection vector
/// - `alpha` (1): gate sharpness (paper converges to ~2.2)
/// - `tau` (1): energy threshold (paper converges to ~0.35)
///
/// Field order: Vec (24 bytes) before f32s (4 bytes each) eliminates padding.
#[derive(Clone, Debug)]
pub struct EgaGate {
    /// Learned energy projection vector [head_dim].
    pub w_proj: Vec<f32>,
    /// Gate sharpness parameter. Higher α → sharper gate transition.
    pub alpha: f32,
    /// Energy threshold. Tokens with energy above τ are preserved,
    /// tokens below are suppressed.
    pub tau: f32,
}

impl EgaGate {
    /// Create a new EGA gate with default initialization.
    ///
    /// - `w_proj`: initialized to 1/d (uniform energy prior)
    /// - `alpha`: 2.2 (paper converged value)
    /// - `tau`: 0.35 (paper converged value)
    pub fn new(head_dim: usize) -> Self {
        let uniform = 1.0 / head_dim as f32;
        Self {
            w_proj: vec![uniform; head_dim],
            alpha: 2.2,
            tau: 0.35,
        }
    }

    /// Total number of learnable parameters: head_dim + 2.
    #[inline]
    pub fn parameter_count(&self) -> usize {
        self.w_proj.len() + 2
    }

    /// Compute energy scores for all positions: e = X · w_proj.
    ///
    /// - `x`: input embeddings, `[seq_len * dim]` row-major
    /// - `seq_len`: number of token positions
    /// - `dim`: embedding dimension (must equal `w_proj.len()`)
    ///
    /// Returns energy scores `[seq_len]`.
    pub fn energy_scores(&self, x: &[f32], seq_len: usize, dim: usize) -> Vec<f32> {
        let mut out = vec![0.0; seq_len];
        self.energy_scores_into(x, seq_len, dim, &mut out);
        out
    }

    /// Zero-alloc variant of [`energy_scores`].
    ///
    /// Writes energy scores into `out[..seq_len]`.
    #[inline]
    pub fn energy_scores_into(&self, x: &[f32], seq_len: usize, dim: usize, out: &mut [f32]) {
        assert_eq!(x.len(), seq_len * dim, "x must have seq_len × dim elements");
        assert_eq!(dim, self.w_proj.len(), "dim must match w_proj length");

        for (i, out_slot) in out.iter_mut().enumerate().take(seq_len) {
            let row_off = i * dim;
            *out_slot = katgpt_core::simd::simd_dot_f32(&x[row_off..row_off + dim], &self.w_proj, dim);
        }
    }

    /// Apply EGA gate to attention weights (in-place).
    ///
    /// - `attn_weights`: `[seq_len × seq_len]` row-major attention weight matrix.
    ///   Row i contains the attention distribution from query i to all keys.
    /// - `energy`: energy scores `[seq_len]` (one per key position)
    /// - `seq_len`: sequence length
    ///
    /// After this call, `attn_weights` contains gated + renormalized weights:
    /// Âᵢⱼ = Aᵢⱼ · gⱼ / Σₖ(Aᵢₖ · gₖ + ε)
    ///
    /// **Note:** This allocates a gate buffer internally. For decode loops, prefer
    /// [`gate_attention_into`] to avoid per-call heap allocation.
    pub fn gate_attention(
        &self,
        attn_weights: &mut [f32],
        energy: &[f32],
        seq_len: usize,
        gate_buf: &mut [f32],
    ) {
        assert_eq!(attn_weights.len(), seq_len * seq_len);
        assert_eq!(energy.len(), seq_len);
        assert!(gate_buf.len() >= seq_len);

        self.gate_attention_into(attn_weights, energy, seq_len, gate_buf);
    }

    /// Zero-alloc variant of [`gate_attention`] that reuses a pre-allocated gate buffer.
    ///
    /// Pass a `gate_buf` of length `>= seq_len` to avoid per-call `vec![0.0; seq_len]` allocation.
    #[inline]
    pub fn gate_attention_into(
        &self,
        attn_weights: &mut [f32],
        energy: &[f32],
        seq_len: usize,
        gate_buf: &mut [f32],
    ) {
        assert_eq!(attn_weights.len(), seq_len * seq_len);
        assert_eq!(energy.len(), seq_len);
        assert!(gate_buf.len() >= seq_len);

        compute_energy_gate_into(energy, self.alpha, self.tau, gate_buf);

        for i in 0..seq_len {
            let row_start = i * seq_len;
            let row = &mut attn_weights[row_start..row_start + seq_len];

            // Apply gate to each key position and compute sum (SIMD)
            katgpt_core::simd::simd_scale_mul_inplace(row, gate_buf, 1.0);
            let row_sum = katgpt_core::simd::simd_sum_f32(row);

            // Renormalize (SIMD)
            let inv_sum = 1.0 / (row_sum + EPS);
            katgpt_core::simd::simd_scale_inplace(row, inv_sum);
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_bounds() {
        for &x in &[-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let s = sigmoid(x);
            assert!((0.0..=1.0).contains(&s), "sigmoid({x}) = {s} out of [0,1]");
        }
    }

    #[test]
    fn sigmoid_symmetry() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
        assert!((sigmoid(2.0) + sigmoid(-2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn z_normalize_zero_variance() {
        let mut scores = [3.0; 4];
        z_normalize(&mut scores);
        // All equal → should produce ~0 after normalization
        for &s in &scores {
            assert!(s.abs() < 1e-6, "expected ~0, got {s}");
        }
    }

    #[test]
    fn z_normalize_mean_zero() {
        let mut scores = [1.0, 2.0, 3.0, 4.0, 5.0];
        z_normalize(&mut scores);
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        assert!(
            mean.abs() < 1e-5,
            "z-normalized mean should be ~0, got {mean}"
        );
    }

    #[test]
    fn ega_gate_parameter_count() {
        let gate = EgaGate::new(64);
        assert_eq!(gate.parameter_count(), 66); // 64 + 2
    }

    #[test]
    fn ega_energy_scores_basic() {
        let head_dim = 4;
        let gate = EgaGate::new(head_dim);
        // w_proj = [0.25; 4], x = [1,2,3,4] (seq_len=1, dim=4)
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let energy = gate.energy_scores(&x, 1, head_dim);
        assert_eq!(energy.len(), 1);
        // dot product: 1*0.25 + 2*0.25 + 3*0.25 + 4*0.25 = 2.5
        assert!((energy[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn ega_gate_attention_sums_to_one() {
        let seq_len = 4;
        let head_dim = 8;
        let gate = EgaGate::new(head_dim);

        // Uniform attention weights
        let mut attn = vec![1.0 / seq_len as f32; seq_len * seq_len];
        let energy = vec![1.0, 2.0, 3.0, 4.0];

        let mut gate_buf = vec![0.0; seq_len];
        gate.gate_attention(&mut attn, &energy, seq_len, &mut gate_buf);

        // Each row should sum to 1
        for i in 0..seq_len {
            let row_sum: f32 = attn[i * seq_len..(i + 1) * seq_len].iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "row {i} sums to {row_sum}");
        }
    }
}
