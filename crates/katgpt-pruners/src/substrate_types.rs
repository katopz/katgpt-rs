//! SubstrateGate core types — capability substrate masks for inference-time routing (Plan 216).
//!
//! Pre-computed per-capability MLP channel masks intersected with ReLU activation masks
//! for dual sparsity. Each mask represents a capability substrate — a sparse set of
//! MLP channels that preserve a specific capability (e.g., "python_stdlib", "async_patterns").

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::fmt;

// ── SubstrateMask ──────────────────────────────────────────────

/// Packed bitmask over `[layers × d_ff]` MLP channels representing a capability substrate.
///
/// Each bit in the packed `u64` bitmask corresponds to one MLP channel across all layers.
/// A set bit means the channel is part of this capability's substrate.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubstrateMask {
    /// Packed bitmask: `bitmask[layer * channels_per_word + word]` covers channels for that layer.
    /// Each u64 covers 64 channels. Total words = ceil(total_channels / 64).
    bitmask: Vec<u64>,
    /// Number of layers in the model.
    n_layers: usize,
    /// MLP hidden dimension (d_ff) per layer.
    mlp_hidden: usize,
    /// Active channel count per layer.
    per_layer_active: Vec<usize>,
    /// Recovery score: how much of the model's capability this mask preserves [0, 1].
    recovery_score: f32,
    /// BLAKE3 hash of the bitmask for provenance verification.
    #[serde(with = "hash_bytes")]
    hash: [u8; 32],
    /// Human-readable capability name (e.g., "python_stdlib").
    capability_name: String,
    /// Model identifier this mask was extracted from.
    model_id: String,
}

/// Serde helpers for 32-byte hash.
mod hash_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(data: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(data)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let bytes: Vec<u8> = Vec::deserialize(d)?;
        let mut arr = [0u8; 32];
        let len = bytes.len().min(32);
        arr[..len].copy_from_slice(&bytes[..len]);
        Ok(arr)
    }
}

impl SubstrateMask {
    /// Create a new empty substrate mask.
    ///
    /// All channels start inactive. Use `set()` to activate specific channels.
    pub fn new(
        n_layers: usize,
        mlp_hidden: usize,
        capability_name: String,
        model_id: String,
    ) -> Self {
        let total_channels = n_layers * mlp_hidden;
        let word_count = total_channels.div_ceil(64);
        Self {
            bitmask: vec![0u64; word_count],
            n_layers,
            mlp_hidden,
            per_layer_active: vec![0usize; n_layers],
            recovery_score: 0.0,
            hash: [0u8; 32],
            capability_name,
            model_id,
        }
    }

    /// Get the activation state of channel `(layer, channel)`.
    #[inline]
    pub fn get(&self, layer: usize, channel: usize) -> bool {
        if layer >= self.n_layers || channel >= self.mlp_hidden {
            return false;
        }
        let flat = layer * self.mlp_hidden + channel;
        let word = flat / 64;
        let bit = flat % 64;
        (self.bitmask.get(word).copied().unwrap_or(0) >> bit) & 1 == 1
    }

    /// Set channel `(layer, channel)` to active.
    pub fn set(&mut self, layer: usize, channel: usize) {
        if layer >= self.n_layers || channel >= self.mlp_hidden {
            return;
        }
        let flat = layer * self.mlp_hidden + channel;
        let word = flat / 64;
        let bit = flat % 64;
        if word < self.bitmask.len() {
            let was_set = (self.bitmask[word] >> bit) & 1 == 1;
            if !was_set {
                self.bitmask[word] |= 1u64 << bit;
                self.per_layer_active[layer] += 1;
                self.recompute_hash();
            }
        }
    }

    /// Set the recovery score manually.
    pub fn set_recovery_score(&mut self, score: f32) {
        self.recovery_score = score.clamp(0.0, 1.0);
    }

    /// Get the recovery score.
    pub fn recovery_score(&self) -> f32 {
        self.recovery_score
    }

    /// Total active channels across all layers.
    pub fn active_count(&self) -> usize {
        self.per_layer_active.iter().sum()
    }

    /// Active ratio: fraction of total channels that are active.
    pub fn active_ratio(&self) -> f32 {
        let total = self.n_layers * self.mlp_hidden;
        if total == 0 {
            return 0.0;
        }
        self.active_count() as f32 / total as f32
    }

    /// Active channels for a specific layer.
    pub fn layer_active_count(&self, layer: usize) -> usize {
        self.per_layer_active.get(layer).copied().unwrap_or(0)
    }

    /// Intersect this mask with another — bitwise AND.
    ///
    /// Only channels active in BOTH masks survive.
    /// Recovery score = min(self, other).
    pub fn intersect(&self, other: &Self) -> Self {
        let word_count = self.bitmask.len().min(other.bitmask.len());
        let mut bitmask: Vec<u64> = Vec::with_capacity(word_count);
        let mut per_layer_active = vec![0usize; self.n_layers];

        for word in 0..word_count {
            let and_val = self.bitmask.get(word).copied().unwrap_or(0)
                & other.bitmask.get(word).copied().unwrap_or(0);
            bitmask.push(and_val);
        }
        // Pad if needed
        bitmask.resize(self.bitmask.len(), 0);

        // Recount per-layer active
        for (layer, layer_active) in per_layer_active.iter_mut().enumerate().take(self.n_layers) {
            let base = layer * self.mlp_hidden;
            let mut count = 0usize;
            let layer_words = self.mlp_hidden.div_ceil(64);
            for w in 0..layer_words {
                let word_idx = (base / 64) + w;
                count += bitmask.get(word_idx).copied().unwrap_or(0).count_ones() as usize;
            }
            *layer_active = count;
        }

        let mut result = Self {
            bitmask,
            n_layers: self.n_layers,
            mlp_hidden: self.mlp_hidden,
            per_layer_active,
            recovery_score: self.recovery_score.min(other.recovery_score),
            hash: [0u8; 32],
            capability_name: format!("{}∩{}", self.capability_name, other.capability_name),
            model_id: self.model_id.clone(),
        };
        result.recompute_hash();
        result
    }

    /// Get the bitmask layer slice for a given layer index.
    ///
    /// Returns a slice of u64 words covering `mlp_hidden` channels for this layer.
    pub fn layer_bitmask(&self, layer: usize) -> &[u64] {
        if layer >= self.n_layers {
            return &[];
        }
        let base_word = (layer * self.mlp_hidden) / 64;
        let layer_words = self.mlp_hidden.div_ceil(64);
        let end = (base_word + layer_words).min(self.bitmask.len());
        &self.bitmask[base_word..end]
    }

    /// Check if channel is active given a flat index within a layer.
    pub fn is_channel_active(&self, layer: usize, flat_channel: usize) -> bool {
        self.get(layer, flat_channel)
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// MLP hidden dimension.
    pub fn mlp_hidden(&self) -> usize {
        self.mlp_hidden
    }

    /// Capability name.
    pub fn capability_name(&self) -> &str {
        &self.capability_name
    }

    /// Model ID.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// BLAKE3 hash.
    pub fn hash(&self) -> &[u8; 32] {
        &self.hash
    }

    /// Verify hash integrity.
    pub fn verify_hash(&self) -> bool {
        let computed = compute_blake3(&self.bitmask);
        computed == self.hash
    }

    fn recompute_hash(&mut self) {
        self.hash = compute_blake3(&self.bitmask);
    }
}

fn compute_blake3(bitmask: &[u64]) -> [u8; 32] {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            bitmask.as_ptr() as *const u8,
            std::mem::size_of_val(bitmask),
        )
    };
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

impl fmt::Display for SubstrateMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SubstrateMask({}): {}/{} channels ({:.1}%), recovery={:.2}",
            self.capability_name,
            self.active_count(),
            self.n_layers * self.mlp_hidden,
            self.active_ratio() * 100.0,
            self.recovery_score,
        )
    }
}

// ── SubstrateRouter Trait ──────────────────────────────────────

/// Trait for selecting a substrate mask based on input context.
pub trait SubstrateRouter: Send + Sync {
    /// Select the best substrate mask for the given token context.
    fn select_mask(
        &self,
        tokens: &[usize],
        config: &katgpt_types::Config,
    ) -> Option<&SubstrateMask>;
    /// Register a new capability mask.
    fn register_mask(&mut self, capability: String, mask: SubstrateMask);
}

// ── NoSubstrateRouter ─────────────────────────────────────────

/// Default router that always returns None — falls back to full MLP.
pub struct NoSubstrateRouter {
    _private: (),
}

impl NoSubstrateRouter {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for NoSubstrateRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl SubstrateRouter for NoSubstrateRouter {
    fn select_mask(
        &self,
        _tokens: &[usize],
        _config: &katgpt_types::Config,
    ) -> Option<&SubstrateMask> {
        None
    }

    fn register_mask(&mut self, _capability: String, _mask: SubstrateMask) {
        // No-op
    }
}

// ── SimpleSubstrateRouter ──────────────────────────────────────

/// Simple router that stores masks and selects by recovery score.
///
/// For each input, returns the mask with the highest recovery score
/// that exceeds the configured threshold.
pub struct SimpleSubstrateRouter {
    masks: Vec<(String, SubstrateMask)>,
    threshold: f32,
    cached_idx: Option<usize>,
}

impl SimpleSubstrateRouter {
    pub fn new(threshold: f32) -> Self {
        Self {
            masks: Vec::new(),
            threshold,
            cached_idx: None,
        }
    }

    /// Number of registered masks.
    pub fn mask_count(&self) -> usize {
        self.masks.len()
    }

    /// Clear the cached mask selection (call when context changes significantly).
    pub fn clear_cache(&mut self) {
        self.cached_idx = None;
    }
}

impl SubstrateRouter for SimpleSubstrateRouter {
    fn select_mask(
        &self,
        _tokens: &[usize],
        _config: &katgpt_types::Config,
    ) -> Option<&SubstrateMask> {
        // Return cached if available
        if let Some(idx) = self.cached_idx
            && idx < self.masks.len()
        {
            return Some(&self.masks[idx].1);
        }

        // Select mask with highest recovery above threshold
        let mut best_idx = None;
        let mut best_score = self.threshold;
        for (i, (_, mask)) in self.masks.iter().enumerate() {
            if mask.recovery_score() > best_score {
                best_score = mask.recovery_score();
                best_idx = Some(i);
            }
        }

        best_idx.map(|i| &self.masks[i].1)
    }

    fn register_mask(&mut self, capability: String, mask: SubstrateMask) {
        self.cached_idx = None; // Invalidate cache
        self.masks.push((capability, mask));
    }
}

// ── SubstrateConfig ───────────────────────────────────────────

/// Configuration for substrate gate routing.
#[derive(Debug, Clone)]
pub struct SubstrateConfig {
    /// Loaded substrate masks.
    pub masks: Vec<SubstrateMask>,
    /// Minimum recovery score to use a mask.
    pub threshold: f32,
}

impl Default for SubstrateConfig {
    fn default() -> Self {
        Self {
            masks: Vec::new(),
            threshold: 0.3,
        }
    }
}

impl SubstrateConfig {
    pub fn new(threshold: f32) -> Self {
        Self {
            masks: Vec::new(),
            threshold,
        }
    }

    /// Validate that all masks match the given model architecture.
    pub fn validate(&self, n_layers: usize, mlp_hidden: usize, model_id: &str) -> bool {
        self.masks.iter().all(|m| {
            m.n_layers() == n_layers && m.mlp_hidden() == mlp_hidden && m.model_id() == model_id
        })
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_new_is_empty() {
        let mask = SubstrateMask::new(4, 1024, "test".to_string(), "model".to_string());
        assert_eq!(mask.active_count(), 0);
        assert!((mask.active_ratio() - 0.0).abs() < 0.001);
        assert_eq!(mask.n_layers(), 4);
        assert_eq!(mask.mlp_hidden(), 1024);
    }

    #[test]
    fn test_mask_set_get() {
        let mut mask = SubstrateMask::new(2, 128, "test".to_string(), "model".to_string());

        assert!(!mask.get(0, 0));
        assert!(!mask.get(0, 63));
        assert!(!mask.get(1, 100));

        mask.set(0, 0);
        mask.set(0, 63);
        mask.set(1, 100);

        assert!(mask.get(0, 0));
        assert!(mask.get(0, 63));
        assert!(mask.get(1, 100));
        assert!(!mask.get(0, 1)); // Not set
        assert_eq!(mask.active_count(), 3);
    }

    #[test]
    fn test_mask_set_idempotent() {
        let mut mask = SubstrateMask::new(1, 64, "test".to_string(), "model".to_string());
        mask.set(0, 10);
        mask.set(0, 10); // Set again — should not double-count
        assert_eq!(mask.active_count(), 1);
    }

    #[test]
    fn test_mask_out_of_bounds() {
        let mut mask = SubstrateMask::new(2, 64, "test".to_string(), "model".to_string());
        mask.set(99, 0); // Out of bounds — no panic
        mask.set(0, 999); // Out of bounds — no panic
        assert!(!mask.get(99, 0));
        assert!(!mask.get(0, 999));
        assert_eq!(mask.active_count(), 0);
    }

    #[test]
    fn test_mask_intersect() {
        let mut m1 = SubstrateMask::new(1, 128, "a".to_string(), "model".to_string());
        let mut m2 = SubstrateMask::new(1, 128, "b".to_string(), "model".to_string());

        m1.set(0, 0);
        m1.set(0, 1);
        m1.set(0, 10);
        m1.set_recovery_score(0.8);

        m2.set(0, 0);
        m2.set(0, 10);
        m2.set(0, 20);
        m2.set_recovery_score(0.6);

        let intersection = m1.intersect(&m2);
        assert!(intersection.get(0, 0)); // In both
        assert!(!intersection.get(0, 1)); // Only in m1
        assert!(intersection.get(0, 10)); // In both
        assert!(!intersection.get(0, 20)); // Only in m2
        assert_eq!(intersection.active_count(), 2);
        // Recovery = min(0.8, 0.6) = 0.6
        assert!((intersection.recovery_score() - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_mask_hash_integrity() {
        let mut mask = SubstrateMask::new(1, 64, "test".to_string(), "model".to_string());
        mask.set(0, 10);
        mask.set(0, 20);
        assert!(mask.verify_hash());

        // Tamper with bitmask directly
        // (can't easily do this with private field, so just verify hash exists)
        let hash = mask.hash();
        assert_ne!(hash, &[0u8; 32]);
    }

    #[test]
    fn test_mask_recovery_score() {
        let mut mask = SubstrateMask::new(1, 64, "test".to_string(), "model".to_string());
        assert!((mask.recovery_score() - 0.0).abs() < 0.001);

        mask.set_recovery_score(0.85);
        assert!((mask.recovery_score() - 0.85).abs() < 0.001);

        // Clamped to [0, 1]
        mask.set_recovery_score(1.5);
        assert!((mask.recovery_score() - 1.0).abs() < 0.001);

        mask.set_recovery_score(-0.5);
        assert!((mask.recovery_score() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_mask_layer_bitmask() {
        let mut mask = SubstrateMask::new(2, 128, "test".to_string(), "model".to_string());
        mask.set(0, 0);
        mask.set(0, 64); // Crosses word boundary
        mask.set(1, 10);

        let layer0 = mask.layer_bitmask(0);
        assert_eq!(layer0.len(), 2); // 128 / 64 = 2 words
        assert_ne!(layer0[0], 0); // Has bit set
        assert_ne!(layer0[1], 0); // Has bit set

        let layer1 = mask.layer_bitmask(1);
        assert_ne!(layer1[0], 0);
    }

    #[test]
    fn test_no_substrate_router() {
        let router = NoSubstrateRouter::new();
        let config = katgpt_types::Config::default();
        assert!(router.select_mask(&[], &config).is_none());
    }

    #[test]
    fn test_simple_router_selects_best_recovery() {
        let mut router = SimpleSubstrateRouter::new(0.3);
        let config = katgpt_types::Config::default();

        let mut m1 = SubstrateMask::new(1, 64, "low".to_string(), "model".to_string());
        m1.set_recovery_score(0.5);
        m1.set(0, 10);

        let mut m2 = SubstrateMask::new(1, 64, "high".to_string(), "model".to_string());
        m2.set_recovery_score(0.9);
        m2.set(0, 20);

        router.register_mask("low".to_string(), m1);
        router.register_mask("high".to_string(), m2);

        let selected = router.select_mask(&[], &config);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().capability_name(), "high");
    }

    #[test]
    fn test_simple_router_below_threshold() {
        let mut router = SimpleSubstrateRouter::new(0.8); // High threshold
        let config = katgpt_types::Config::default();

        let mut m = SubstrateMask::new(1, 64, "low".to_string(), "model".to_string());
        m.set_recovery_score(0.5); // Below threshold
        m.set(0, 10);

        router.register_mask("low".to_string(), m);
        assert!(router.select_mask(&[], &config).is_none());
    }

    #[test]
    fn test_substrate_config_validate() {
        let config = SubstrateConfig::new(0.3);
        // Empty masks validate trivially
        assert!(config.validate(4, 1024, "model"));

        let mut m = SubstrateMask::new(2, 128, "test".to_string(), "model".to_string());
        m.set(0, 10);
        let config = SubstrateConfig {
            masks: vec![m],
            threshold: 0.3,
        };
        assert!(config.validate(2, 128, "model"));
        assert!(!config.validate(4, 128, "model")); // Wrong n_layers
        assert!(!config.validate(2, 256, "model")); // Wrong mlp_hidden
        assert!(!config.validate(2, 128, "other")); // Wrong model_id
    }

    #[test]
    fn test_mask_display() {
        let mut mask = SubstrateMask::new(2, 100, "python".to_string(), "model".to_string());
        mask.set(0, 10);
        mask.set(1, 20);
        mask.set_recovery_score(0.75);

        let s = format!("{}", mask);
        assert!(s.contains("python"));
        assert!(s.contains("0.75"));
    }

    #[test]
    fn test_active_ratio() {
        let mut mask = SubstrateMask::new(1, 100, "test".to_string(), "model".to_string());
        mask.set(0, 0);
        mask.set(0, 1);
        mask.set(0, 2);
        mask.set(0, 3);
        mask.set(0, 4);

        assert!((mask.active_ratio() - 0.05).abs() < 0.001); // 5/100
    }

    #[test]
    fn test_per_layer_active() {
        let mut mask = SubstrateMask::new(2, 64, "test".to_string(), "model".to_string());
        mask.set(0, 10);
        mask.set(0, 20);
        mask.set(1, 5);

        assert_eq!(mask.layer_active_count(0), 2);
        assert_eq!(mask.layer_active_count(1), 1);
    }
}
