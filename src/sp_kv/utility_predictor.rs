//! Utility predictor: 2-layer MLP predicting per-KV-head future utility.
//!
//! Architecture: `h ∈ R^{d_model} → SiLU(W1·h + b1) → sigmoid(W2·hidden + b2) → u ∈ (0,1)^{n_kv_heads}`
//!
//! Based on SP-KV paper (arXiv:2605.14037):
//! - SiLU activation in hidden layer (not ReLU — paper Section 3.1)
//! - Sigmoid output ensures utilities ∈ (0, 1)
//! - Bias `b2` initialized to +5.0 → σ(5) ≈ 0.993 (gates start open)
//! - No auxiliary loss needed — gradients flow through log(u) gate bias

use crate::sp_kv::types::UtilityPredictorWeights;

/// SiLU activation: x * σ(x) = x / (1 + exp(-x)).
/// Smooth, non-monotonic, self-gated. Paper uses this over ReLU for better gradient flow.
#[inline(always)]
fn silu(x: f32) -> f32 {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    x * sigmoid
}

/// Sigmoid activation: 1 / (1 + exp(-x)).
/// Maps logits to (0, 1) range for utility scores.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Predict utility for each KV head from a hidden state (zero-alloc variant).
///
/// # Arguments
/// * `weights` - Predictor weights for one layer
/// * `h` - Hidden state vector [d_model] (post-RMSNorm, pre-attention)
/// * `d_model` - Model embedding dimension
/// * `hidden` - Predictor hidden dimension
/// * `n_kv_heads` - Number of KV heads (GQA groups)
/// * `buf` - Pre-allocated scratch buffer [hidden] for intermediate activations
/// * `out` - Output buffer [n_kv_heads] for utilities
///
/// # Layout
/// ```text
/// w1: [hidden, d_model]  row-major
/// w2: [n_kv_heads, hidden]  row-major
/// b1: [hidden]
/// b2: [n_kv_heads]  (initialized to +5.0)
/// ```
///
/// Zero-alloc variant of [`predict`] that writes utilities into a pre-allocated buffer.
///
/// Use in decode loops to avoid per-call `vec![0.0f32; n_kv_heads]` heap allocation.
#[allow(clippy::needless_range_loop)]
pub fn predict_into(
    weights: &UtilityPredictorWeights,
    h: &[f32],
    d_model: usize,
    hidden: usize,
    n_kv_heads: usize,
    buf: &mut [f32],
    out: &mut [f32],
) {
    debug_assert_eq!(h.len(), d_model);
    debug_assert_eq!(weights.w1.len(), hidden * d_model);
    debug_assert_eq!(weights.b1.len(), hidden);
    debug_assert_eq!(weights.w2.len(), n_kv_heads * hidden);
    debug_assert_eq!(weights.b2.len(), n_kv_heads);
    debug_assert!(buf.len() >= hidden);
    debug_assert!(out.len() >= n_kv_heads);

    // Layer 1: hidden = SiLU(W1 · h + b1)
    for i in 0..hidden {
        let row_off = i * d_model;
        let dot = crate::simd::simd_dot_f32(&weights.w1[row_off..row_off + d_model], h, d_model);
        buf[i] = silu(dot + weights.b1[i]);
    }

    // Layer 2: u = sigmoid(W2 · hidden + b2)
    for k in 0..n_kv_heads {
        let row_off = k * hidden;
        let dot = crate::simd::simd_dot_f32(
            &weights.w2[row_off..row_off + hidden],
            &buf[..hidden],
            hidden,
        );
        out[k] = sigmoid(dot + weights.b2[k]);
    }
}

/// Allocating wrapper around [`predict_into`].
///
/// **Note:** This allocates a utilities Vec internally. For decode loops, prefer
/// [`predict_into`] to avoid per-call heap allocation.
#[allow(clippy::needless_range_loop)]
pub fn predict(
    weights: &UtilityPredictorWeights,
    h: &[f32],
    d_model: usize,
    hidden: usize,
    n_kv_heads: usize,
    buf: &mut [f32],
) -> Vec<f32> {
    let mut utilities = vec![0.0f32; n_kv_heads];
    predict_into(weights, h, d_model, hidden, n_kv_heads, buf, &mut utilities);
    utilities
}

/// Predict utility for a single KV head (GQA optimization).
///
/// When all query heads in a GQA group share the same KV head,
/// this avoids computing utilities for all heads when only one is needed.
///
/// # Arguments
/// * `kv_head_idx` - Which KV head to predict utility for (0-indexed)
#[allow(clippy::needless_range_loop)]
pub fn predict_single_head(
    weights: &UtilityPredictorWeights,
    h: &[f32],
    kv_head_idx: usize,
    d_model: usize,
    hidden: usize,
    buf: &mut [f32],
) -> f32 {
    debug_assert!(kv_head_idx < weights.b2.len());

    // Layer 1: hidden = SiLU(W1 · h + b1)
    for i in 0..hidden {
        let row_off = i * d_model;
        let dot = crate::simd::simd_dot_f32(&weights.w1[row_off..row_off + d_model], h, d_model);
        buf[i] = silu(dot + weights.b1[i]);
    }

    // Layer 2: only compute the requested head
    let row_off = kv_head_idx * hidden;
    let dot = crate::simd::simd_dot_f32(
        &weights.w2[row_off..row_off + hidden],
        &buf[..hidden],
        hidden,
    );
    sigmoid(dot + weights.b2[kv_head_idx])
}

/// Compute gate bias from utility for soft gating (training phase 1).
///
/// bias = log(u + ε). When u → 1, bias → 0 (no effect).
/// When u → 0, bias → -∞ (position masked out).
/// Gradients flow through: ∂bias/∂u = 1/(u + ε).
#[inline(always)]
pub fn soft_gate_bias(utility: f32) -> f32 {
    (utility + 1e-8f32).ln()
}

/// Compute gate bias from utility for hard gating (inference).
///
/// bias = 0 if u ≥ τ (retain), -∞ if u < τ (prune).
#[inline(always)]
pub fn hard_gate_bias(utility: f32, threshold: f32) -> f32 {
    match utility >= threshold {
        true => 0.0,
        false => f32::NEG_INFINITY,
    }
}

/// Compute gate bias for TAHG annealing (training phase 2).
///
/// Blends soft and hard: ũ = (1-α)·u + α·1[u≥τ], then bias = log(ũ + ε).
/// α ramps from 0→1 over annealing period.
#[inline(always)]
pub fn tahg_gate_bias(utility: f32, threshold: f32, alpha: f32) -> f32 {
    let hard_indicator = match utility >= threshold {
        true => 1.0f32,
        false => 0.0f32,
    };
    let blended = (1.0 - alpha) * utility + alpha * hard_indicator;
    (blended + 1e-8f32).ln()
}

/// Aggregate per-KV-head utilities to a single scalar per position.
///
/// For cache write decisions, we need one value per position.
/// Options: max (keep if any head needs it), mean (democratic), or first (GQA broadcast).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UtilityAggregation {
    /// Maximum utility across all KV heads.
    /// Conservative: retains position if ANY head finds it useful.
    Max,
    /// Mean utility across all KV heads.
    /// Democratic: retains based on average utility.
    Mean,
    /// Use the first KV head's utility (GQA: all heads in a group share the same utility).
    First,
}

/// Aggregate per-head utilities to a single scalar.
pub fn aggregate_utilities(utilities: &[f32], mode: UtilityAggregation) -> f32 {
    match (mode, utilities.len()) {
        (UtilityAggregation::First, _) if !utilities.is_empty() => utilities[0],
        (UtilityAggregation::Max, 0) => 1.0,
        (UtilityAggregation::Mean, 0) => 1.0,
        (_, 0) => 1.0,
        (UtilityAggregation::Max, _) => {
            let mut max = utilities[0];
            for &u in &utilities[1..] {
                max = max.max(u);
            }
            max
        }
        (UtilityAggregation::Mean, _) => {
            let sum: f32 = utilities.iter().copied().sum();
            sum / utilities.len() as f32
        }
        (UtilityAggregation::First, _) => utilities[0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_activation() {
        // SiLU(0) = 0 * σ(0) = 0 * 0.5 = 0
        assert!((silu(0.0)).abs() < 1e-6);
        // SiLU(1) ≈ 1 * 0.731 = 0.731
        assert!((silu(1.0) - 0.7311).abs() < 0.01);
        // SiLU(-1) ≈ -1 * 0.269 = -0.269
        assert!((silu(-1.0) - (-0.2689)).abs() < 0.01);
        // SiLU(large) ≈ large (sigmoid → 1)
        assert!(silu(100.0) > 99.0);
    }

    #[test]
    fn test_sigmoid_activation() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(5.0) > 0.99);
        assert!(sigmoid(-5.0) < 0.01);
        assert!(sigmoid(100.0) <= 1.0);
        assert!(sigmoid(-100.0) >= 0.0);
    }

    #[test]
    fn test_predict_output_range() {
        let d_model = 32;
        let hidden = 16;
        let n_kv_heads = 4;
        let weights = UtilityPredictorWeights::new(d_model, hidden, n_kv_heads, 5.0);
        let h = vec![1.0; d_model];
        let mut buf = vec![0.0; hidden];

        let utilities = predict(&weights, &h, d_model, hidden, n_kv_heads, &mut buf);

        assert_eq!(utilities.len(), n_kv_heads);
        for &u in &utilities {
            assert!(u > 0.0 && u < 1.0, "Utility {u} not in (0, 1)");
        }
    }

    #[test]
    fn test_predict_init_bias_open_gates() {
        let d_model = 32;
        let hidden = 16;
        let n_kv_heads = 4;
        // With init_bias=5.0 and zero-ish hidden, sigmoid(5.0) ≈ 0.993
        let weights = UtilityPredictorWeights::new(d_model, hidden, n_kv_heads, 5.0);
        let h = vec![0.0; d_model]; // zero input → W·h = 0, so logits = b2 = 5.0
        let mut buf = vec![0.0; hidden];

        let utilities = predict(&weights, &h, d_model, hidden, n_kv_heads, &mut buf);

        for &u in &utilities {
            assert!(u > 0.99, "Gate should start nearly open, got {u}");
        }
    }

    #[test]
    fn test_soft_gate_bias_range() {
        // u=1 → bias ≈ 0
        assert!(soft_gate_bias(1.0).abs() < 0.01);
        // u=0.5 → bias ≈ log(0.5) ≈ -0.693
        assert!((soft_gate_bias(0.5) - (-0.693)).abs() < 0.01);
        // u≈0 → bias → -∞
        assert!(soft_gate_bias(0.001) < -5.0);
    }

    #[test]
    fn test_hard_gate_bias_threshold() {
        assert_eq!(hard_gate_bias(0.7, 0.5), 0.0);
        assert_eq!(hard_gate_bias(0.3, 0.5), f32::NEG_INFINITY);
        assert_eq!(hard_gate_bias(0.5, 0.5), 0.0); // exactly at threshold → retain
    }

    #[test]
    fn test_tahg_gate_bias_annealing() {
        // α=0: same as soft
        let soft = soft_gate_bias(0.3);
        let tahg_0 = tahg_gate_bias(0.3, 0.5, 0.0);
        assert!((soft - tahg_0).abs() < 0.01);

        // α=1: blended = 0 (u < threshold → hard_indicator=0), bias = log(ε) ≈ -18.4
        // Not -inf because we use log(blended + ε), and blended=0 → log(ε)
        let tahg_1 = tahg_gate_bias(0.3, 0.5, 1.0);
        assert!(
            tahg_1 < -15.0,
            "TAHG α=1 with u<τ should be very negative, got {tahg_1}"
        );

        // α=0.5: blended
        let tahg_half = tahg_gate_bias(0.8, 0.5, 0.5);
        assert!(tahg_half > -1.0); // should be less negative since 0.8 > threshold
    }

    #[test]
    fn test_aggregate_max() {
        let utils = vec![0.1, 0.5, 0.9, 0.3];
        assert!((aggregate_utilities(&utils, UtilityAggregation::Max) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_mean() {
        let utils = vec![0.1, 0.5, 0.9, 0.3];
        let expected = 0.45;
        assert!((aggregate_utilities(&utils, UtilityAggregation::Mean) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_first() {
        let utils = vec![0.1, 0.5, 0.9, 0.3];
        assert!((aggregate_utilities(&utils, UtilityAggregation::First) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_empty() {
        let utils: Vec<f32> = vec![];
        assert_eq!(aggregate_utilities(&utils, UtilityAggregation::Max), 1.0);
        assert_eq!(aggregate_utilities(&utils, UtilityAggregation::Mean), 1.0);
    }

    #[test]
    fn test_predict_single_head_matches_full() {
        let d_model = 32;
        let hidden = 16;
        let n_kv_heads = 4;
        let weights = UtilityPredictorWeights::new(d_model, hidden, n_kv_heads, 5.0);
        let h = vec![1.0; d_model];
        let mut buf = vec![0.0; hidden];
        let mut buf2 = vec![0.0; hidden];

        let full = predict(&weights, &h, d_model, hidden, n_kv_heads, &mut buf);

        for k in 0..n_kv_heads {
            let single = predict_single_head(&weights, &h, k, d_model, hidden, &mut buf2);
            assert!(
                (full[k] - single).abs() < 1e-5,
                "Head {k}: full={full_k}, single={single}",
                full_k = full[k],
            );
        }
    }
}
