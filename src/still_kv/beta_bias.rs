//! Heuristic β (beta) additive attention bias for modelless StillKV compaction.
//!
//! Still's trained compactor produces learned per-latent attention biases (β).
//! For modelless inference, we approximate β using heuristic strategies.
//!
//! β-A: log(T/t) mass-matching — uniform scalar baseline
//! β-D: VortexFlow routing — attention-concentration-weighted per-latent bias

/// Strategy for computing heuristic β bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BetaStrategy {
    /// β-A: log(T/t) mass-matching — uniform scalar for all latents.
    MassMatching = 0,
    /// β-D: VortexFlow routing — attention-concentration-weighted.
    VortexFlowRouting = 1,
}

/// Computed β bias for a compacted KV cache.
///
/// One scalar bias per latent token. During generation, this bias is added
/// to the attention logits over compact slots so the frozen model can
/// calibrate attention to synthetic latent entries.
#[derive(Debug, Clone)]
pub struct BetaBias {
    /// Per-latent bias values. Shape: `[compact_len]` (one scalar per latent).
    pub biases: Vec<f32>,
    /// Strategy used to compute this bias.
    pub strategy: BetaStrategy,
    /// Original sequence length T (before compaction).
    pub original_len: usize,
    /// Compact sequence length t (after compaction).
    pub compact_len: usize,
}

impl BetaBias {
    /// Create a new BetaBias with the given values.
    pub fn new(
        biases: Vec<f32>,
        strategy: BetaStrategy,
        original_len: usize,
        compact_len: usize,
    ) -> Self {
        Self {
            biases,
            strategy,
            original_len,
            compact_len,
        }
    }

    /// Create a zero-valued BetaBias for the given dimensions.
    pub fn zeros(original_len: usize, compact_len: usize) -> Self {
        Self {
            biases: vec![0.0; compact_len],
            strategy: BetaStrategy::MassMatching,
            original_len,
            compact_len,
        }
    }

    /// Returns the uniform mass-matching offset for log(T/t).
    ///
    /// When T tokens are compressed to t latents, each latent absorbs T/t
    /// tokens of attention mass. The log(T/t) offset compensates for this.
    #[inline]
    pub fn mass_matching_offset(original_len: usize, compact_len: usize) -> f32 {
        if compact_len == 0 || original_len == 0 {
            return 0.0;
        }
        let ratio = original_len as f32 / compact_len as f32;
        ratio.ln()
    }
}

/// Compute β-A: mass-matching baseline bias.
///
/// Returns a BetaBias where every latent gets the same `log(T/t)` offset.
/// This is the simplest heuristic — no per-latent differentiation.
///
/// # Arguments
/// * `original_len` - T, the number of tokens before compaction
/// * `compact_len` - t, the number of latent tokens after compaction
pub fn compute_beta_mass_matching(original_len: usize, compact_len: usize) -> BetaBias {
    let offset = BetaBias::mass_matching_offset(original_len, compact_len);
    BetaBias {
        biases: vec![offset; compact_len],
        strategy: BetaStrategy::MassMatching,
        original_len,
        compact_len,
    }
}

/// Compute β-D: VortexFlow routing bias.
///
/// Uses cross-attention weights to compute per-latent bias values.
/// Latents that captured more attention mass (higher concentration) get
/// proportionally scaled bias, weighted by sigmoid of concentration deviation.
///
/// # Arguments
/// * `cross_attn_weights` - Attention weights from cross-attention, shape `[compact_len * original_len]`
///   (row-major, one row per latent, each row is a probability distribution over original tokens)
/// * `original_len` - T, the number of tokens before compaction
/// * `compact_len` - t, the number of latent tokens after compaction
///
/// # Formula
/// For each latent i:
///   concentration_i = max_j(attn[i,j])   // how peaked is this latent's attention
///   expected_uniform = 1.0 / T            // uniform baseline
///   deviation_i = sigmoid((concentration_i - expected_uniform) * T * 0.5)
///   beta_i = log(T/t) * deviation_i
pub fn compute_beta_vortex_flow(
    cross_attn_weights: &[f32],
    original_len: usize,
    compact_len: usize,
) -> BetaBias {
    if compact_len == 0 || original_len == 0 {
        return BetaBias::zeros(original_len, compact_len);
    }

    let log_offset = BetaBias::mass_matching_offset(original_len, compact_len);
    let expected_uniform = 1.0 / original_len as f32;
    let scale = original_len as f32 * 0.5;

    let mut biases = Vec::with_capacity(compact_len);

    for i in 0..compact_len {
        let row_start = i * original_len;
        let row_end = row_start + original_len;

        // Safety: if cross_attn_weights is shorter than expected, fall back to mass-matching
        if row_end > cross_attn_weights.len() {
            biases.push(log_offset);
            continue;
        }

        let row = &cross_attn_weights[row_start..row_end];

        // Compute concentration = max attention weight for this latent
        let mut max_weight = 0.0f32;
        for &w in row {
            if w > max_weight {
                max_weight = w;
            }
        }

        // Sigmoid of deviation from uniform
        let deviation = max_weight - expected_uniform;
        let sigmoid_val = sigmoid(deviation * scale);

        biases.push(log_offset * sigmoid_val);
    }

    BetaBias {
        biases,
        strategy: BetaStrategy::VortexFlowRouting,
        original_len,
        compact_len,
    }
}

/// Standard sigmoid function.
/// `sigmoid(x) = 1 / (1 + exp(-x))`
#[inline]
fn sigmoid(x: f32) -> f32 {
    // Numerically stable: use the positive/negative split
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_matching_offset_basic() {
        // T=1024, t=128 → ratio=8, log(8)≈2.079
        let offset = BetaBias::mass_matching_offset(1024, 128);
        assert!(
            (offset - 8.0f32.ln()).abs() < 1e-4,
            "offset should be log(8), got {offset}"
        );
    }

    #[test]
    fn test_mass_matching_offset_identity() {
        // T=t → ratio=1, log(1)=0
        let offset = BetaBias::mass_matching_offset(128, 128);
        assert!(
            offset.abs() < 1e-6,
            "identity offset should be 0, got {offset}"
        );
    }

    #[test]
    fn test_mass_matching_offset_zero_safety() {
        assert_eq!(BetaBias::mass_matching_offset(0, 128), 0.0);
        assert_eq!(BetaBias::mass_matching_offset(128, 0), 0.0);
    }

    #[test]
    fn test_compute_beta_mass_matching_uniform() {
        let beta = compute_beta_mass_matching(1024, 128);
        assert_eq!(beta.biases.len(), 128);
        assert_eq!(beta.strategy, BetaStrategy::MassMatching);
        assert_eq!(beta.original_len, 1024);
        assert_eq!(beta.compact_len, 128);

        // All biases should be identical = log(8)
        let expected = 8.0f32.ln();
        for (i, &b) in beta.biases.iter().enumerate() {
            assert!(
                (b - expected).abs() < 1e-4,
                "bias[{i}] = {b}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_compute_beta_mass_matching_empty() {
        let beta = compute_beta_mass_matching(1024, 0);
        assert!(beta.biases.is_empty());
    }

    #[test]
    fn test_compute_beta_vortex_flow_basic() {
        // 4 latents, 8 original tokens
        // Uniform attention: each latent attends uniformly to all 8 tokens
        let weights: Vec<f32> = vec![0.125; 4 * 8]; // uniform 1/8

        let beta = compute_beta_vortex_flow(&weights, 8, 4);
        assert_eq!(beta.biases.len(), 4);
        assert_eq!(beta.strategy, BetaStrategy::VortexFlowRouting);

        // Uniform attention → concentration = 0.125 = expected_uniform → deviation = 0
        // sigmoid(0) = 0.5 → beta_i = log(8/4) * 0.5 = log(2) * 0.5
        let expected = 2.0f32.ln() * 0.5;
        for (i, &b) in beta.biases.iter().enumerate() {
            assert!(
                (b - expected).abs() < 0.01,
                "uniform beta[{i}] = {b}, expected ~{expected}"
            );
        }
    }

    #[test]
    fn test_compute_beta_vortex_flow_concentrated() {
        // One latent has very concentrated attention
        let mut weights = vec![0.0f32; 4 * 8];
        // Latent 0: all attention on token 0 (high concentration)
        weights[0] = 1.0;
        // Latent 1: uniform
        for j in 0..8 {
            weights[8 + j] = 0.125;
        }
        // Latent 2: somewhat concentrated
        weights[16] = 0.5;
        weights[17] = 0.5;
        // Latent 3: uniform
        for j in 0..8 {
            weights[24 + j] = 0.125;
        }

        let beta = compute_beta_vortex_flow(&weights, 8, 4);
        assert_eq!(beta.biases.len(), 4);

        // Latent 0 (concentrated) should have higher bias than latent 1 (uniform)
        assert!(
            beta.biases[0] > beta.biases[1],
            "concentrated latent should have higher β: {} vs {}",
            beta.biases[0],
            beta.biases[1]
        );
    }

    #[test]
    fn test_compute_beta_vortex_flow_empty() {
        let beta = compute_beta_vortex_flow(&[], 0, 0);
        assert!(beta.biases.is_empty());
    }

    #[test]
    fn test_compute_beta_vortex_flow_short_weights_fallback() {
        // If cross_attn_weights is shorter than expected, fall back to mass-matching
        let beta = compute_beta_vortex_flow(&[0.5, 0.5], 8, 4);
        // Should not panic, and should have 4 biases
        assert_eq!(beta.biases.len(), 4);
        // Latents 0-1 get computed, latents 2-3 get fallback
    }

    #[test]
    fn test_beta_zeros() {
        let beta = BetaBias::zeros(1024, 128);
        assert_eq!(beta.biases.len(), 128);
        for &b in &beta.biases {
            assert_eq!(b, 0.0);
        }
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!(sigmoid(-100.0) < 0.01);
        assert!(sigmoid(100.0) > 0.99);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_repr_u8() {
        assert_eq!(BetaStrategy::MassMatching as u8, 0);
        assert_eq!(BetaStrategy::VortexFlowRouting as u8, 1);
    }
}
