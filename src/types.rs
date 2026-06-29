// katgpt-rs types: re-exports from katgpt-core + project-specific items.
//
// All shared types (Config, Rng, InferenceOverrides, math utilities, LoRA,
// DomainLatent) are defined in katgpt-core and re-exported here.
// This module adds only katgpt-rs-specific items.

// Re-export all shared types from core
pub use katgpt_core::types::*;

// ---------------------------------------------------------------------------
// QuantizedKVCache — re-exported from katgpt-types (Issue 015 Phase 1)
// ---------------------------------------------------------------------------
//
// The core trait was promoted to `katgpt_types::kv_cache` so backend crates
// (`katgpt-kv`, future sibling engine crates) can implement against it
// without depending on the root `katgpt-rs` crate. Re-export here preserves
// the historical `katgpt_rs::types::QuantizedKVCache` path.

pub use katgpt_core::types::QuantizedKVCache;

/// Extension trait: optional KV-cache compaction.
///
/// Lives in the root crate (not katgpt-types) because it references
/// `crate::still_kv::CompactionStrategy` / `CompactKVCache`, which are
/// feature-gated KV-concrete types. Backends that support compaction
/// implement this in addition to `QuantizedKVCache`. The default impl
/// returns `Err` so every cache type is soundly default-impl-able.
///
/// Historical note: this was originally a `#[cfg(feature = "still_kv")]`
/// method on `QuantizedKVCache` itself. Split out into an extension trait
// during Issue 015 Phase 1 so the core trait could move to the leaf crate.
#[cfg(feature = "still_kv")]
pub trait CompactableKVCache: QuantizedKVCache {
    /// Compact the KV cache into a smaller representation.
    fn compact_into(
        &mut self,
        _budget: usize,
        _strategy: crate::still_kv::CompactionStrategy,
    ) -> Result<crate::still_kv::CompactKVCache, String> {
        Err("compact_into not supported by this cache type".to_string())
    }
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

    // Pre-initialize mask to all false up front. This makes the selection loop
    // branch-free for the unselected tail — no second pass needed after break.
    mask.fill(false);

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
        mask.fill(true);
        return n;
    }

    // Normalize once outside the loop (avoid repeated division).
    let inv_total = 1.0 / total;
    let mut cumsum = 0.0f32;
    let mut selected = 0usize;
    for &idx in scratch_indices.iter() {
        let prob = scores[idx].max(0.0) * inv_total;
        cumsum += prob;
        mask[idx] = true;
        selected += 1;
        if cumsum >= p {
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
// Outlier-Aware Quantization Guard types — re-exported from katgpt-spectral
// (Issue 015 Phase 2)
// ---------------------------------------------------------------------------
// The type definitions were relocated to
// `crates/katgpt-spectral/src/outlier_guard.rs` because they are consumed
// exclusively by the outlier guard, which lives in katgpt-spectral. The
// re-export preserves the historical `katgpt_rs::types::OutlierGuard*` path.

#[cfg(feature = "outlier_guard")]
pub use katgpt_spectral::outlier_guard::{OutlierAction, OutlierGuardConfig};
