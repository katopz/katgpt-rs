// katgpt-rs types: re-exports from katgpt-core + project-specific items.
//
// All shared types (Config, Rng, InferenceOverrides, math utilities, LoRA,
// DomainLatent) are defined in katgpt-core and re-exported here.
// This module adds only katgpt-rs-specific items.

// Re-export all shared types from core
pub use katgpt_core::types::*;

// ---------------------------------------------------------------------------
// QuantizedKVCache — katgpt-rs only
// ---------------------------------------------------------------------------

/// Shared interface for quantized KV caches.
///
/// Enables [`crate::transformer::forward_quantized`] to work with any
/// compression backend (TurboQuant, SpectralQuant, or future methods).
pub trait QuantizedKVCache {
    /// Quantize and store a key vector at given layer and position.
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]);
    /// Quantize and store a value vector at given layer and position.
    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]);
    /// Dequantize a key into a pre-allocated buffer (zero-alloc hot path).
    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]);
    /// Dequantize a value into a pre-allocated buffer (zero-alloc hot path).
    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]);
    /// Reset cache for a new sequence.
    fn reset(&mut self);
    /// Current write position.
    fn pos(&self) -> usize;
    /// Set the current write position.
    fn set_pos(&mut self, pos: usize);
}

// ---------------------------------------------------------------------------
// AsymmetricKVConfig — asymmetric K/V cache compression (Plan 123)
// ---------------------------------------------------------------------------

/// Asymmetric KV cache configuration.
///
/// Research 081 proves V-side compression is quality-free (softmax amplifies K errors
/// O(e^ε) but V errors only scale linearly O(w·ε)). Recommended: key_bits=8, val_bits=3.
///
/// Plan 123: Asymmetric K/V Cache Compression — GOAT proof.
#[derive(Clone, Debug)]
pub struct AsymmetricKVConfig {
    /// Bits for key quantization (precision-critical due to softmax amplification).
    pub key_bits: u8,
    /// Bits for value quantization (quality-free compression opportunity).
    pub val_bits: u8,
}

impl Default for AsymmetricKVConfig {
    fn default() -> Self {
        Self {
            key_bits: 8,
            val_bits: 3,
        }
    }
}

impl AsymmetricKVConfig {
    /// Create a new asymmetric config.
    pub fn new(key_bits: u8, val_bits: u8) -> Self {
        Self { key_bits, val_bits }
    }

    /// Symmetric config (same bits for K and V).
    pub fn symmetric(bits: u8) -> Self {
        Self {
            key_bits: bits,
            val_bits: bits,
        }
    }

    /// Whether this config is asymmetric (key_bits ≠ val_bits).
    #[inline]
    pub fn is_asymmetric(&self) -> bool {
        self.key_bits != self.val_bits
    }

    /// Theoretical compression ratio vs fp32 (32 bits per element).
    /// Returns ratio of fp32 size to quantized size.
    #[inline]
    pub fn compression_ratio(&self) -> f32 {
        let fp32_bits = 32.0;
        let avg_bits = (self.key_bits as f32 + self.val_bits as f32) / 2.0;
        fp32_bits / avg_bits
    }

    /// Total bits per KV pair.
    #[inline]
    pub fn total_bits(&self) -> u8 {
        self.key_bits + self.val_bits
    }
}

// ---------------------------------------------------------------------------
// Adaptive Top-p Coreset Selection (dMoE distillation, Research 161, Plan 181)
// ---------------------------------------------------------------------------

/// Adaptive top-p coreset selection.
///
/// Given a slice of scores, returns a boolean mask selecting the minimal
/// set of indices whose cumulative probability mass >= `p`.
///
/// Algorithm:
/// 1. Sort scores descending
/// 2. Normalize to probability distribution
/// 3. Cumulative sum
/// 4. Select all indices where cumsum < p (plus the first that crosses)
///
/// # Arguments
/// * `scores` - Score values for each element
/// * `p` - Cumulative probability threshold (0.0 to 1.0)
/// * `scratch_indices` - Pre-allocated scratch buffer for indices (caller-owned)
/// * `scratch_sorted` - Pre-allocated scratch buffer for sorted scores (caller-owned)
/// * `mask` - Output boolean mask (caller-owned, initialized by this function)
///
/// # Returns
/// Number of selected elements.
#[inline]
pub fn top_p_coreset(
    scores: &[f32],
    p: f32,
    scratch_indices: &mut [usize],
    scratch_sorted: &mut [f32],
    mask: &mut [bool],
) -> usize {
    let n = scores.len();
    debug_assert_eq!(scratch_indices.len(), n);
    debug_assert_eq!(scratch_sorted.len(), n);
    debug_assert_eq!(mask.len(), n);

    // Initialize indices
    for (i, idx) in scratch_indices.iter_mut().enumerate() {
        *idx = i;
    }

    // Sort by score descending
    scratch_indices.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute total and normalize
    let total: f32 = scratch_indices.iter().map(|&i| scores[i].max(0.0)).sum();

    if total <= 0.0 {
        // Degenerate: select all
        for m in mask.iter_mut() {
            *m = true;
        }
        return n;
    }

    let mut cumsum = 0.0f32;
    let mut selected = 0usize;
    for (rank, &idx) in scratch_indices.iter().enumerate() {
        let prob = scores[idx].max(0.0) / total;
        cumsum += prob;
        mask[idx] = true;
        selected += 1;
        if cumsum >= p {
            // Fill remaining with false
            for &remaining_idx in &scratch_indices[rank + 1..] {
                mask[remaining_idx] = false;
            }
            break;
        }
    }

    selected
}

/// Convenience version of `top_p_coreset` that allocates internally.
/// Use this for non-hot-path calls. For hot paths, use `top_p_coreset` with pre-allocated buffers.
pub fn top_p_coreset_allocating(scores: &[f32], p: f32) -> (Vec<bool>, usize) {
    let n = scores.len();
    let mut scratch_indices = vec![0usize; n];
    let mut scratch_sorted = vec![0.0f32; n];
    let mut mask = vec![false; n];
    let count = top_p_coreset(
        scores,
        p,
        &mut scratch_indices,
        &mut scratch_sorted,
        &mut mask,
    );
    (mask, count)
}

// ---------------------------------------------------------------------------
// Outlier-Aware Quantization Guard (Plan 224)
// ---------------------------------------------------------------------------

/// Action to take when outlier injection is detected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum OutlierAction {
    /// Log warning, continue loading. Default for MIT engine.
    #[default]
    Warn = 0,
    /// Reject the model (return error). Useful for SaaS deployment.
    Reject = 1,
    /// Silent — just record metrics, no warning. Useful for benchmarking.
    Silent = 2,
}

/// Configuration for the outlier-aware quantization guard.
/// Runs once at model load time to detect outlier injection attacks.
#[derive(Clone, Debug)]
pub struct OutlierGuardConfig {
    /// KS D-statistic threshold above which a layer is flagged.
    /// Default: 0.15 (conservative midpoint between normal <0.1 and attacked >0.25).
    pub ks_threshold: f32,
    /// What to do when an outlier is detected.
    pub on_detection: OutlierAction,
    /// Whether to also check StiffSoft eigenvalue distribution if available.
    pub use_stiffsoft_crosscheck: bool,
}

impl Default for OutlierGuardConfig {
    fn default() -> Self {
        Self {
            ks_threshold: 0.15,
            on_detection: OutlierAction::Warn,
            use_stiffsoft_crosscheck: false,
        }
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for OutlierGuardConfig {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = s.serialize_struct("OutlierGuardConfig", 3)?;
        state.serialize_field("ks_threshold", &self.ks_threshold)?;
        state.serialize_field("on_detection", &(self.on_detection as u8))?;
        state.serialize_field("use_stiffsoft_crosscheck", &self.use_stiffsoft_crosscheck)?;
        state.end()
    }
}
