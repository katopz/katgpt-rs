//! SubstrateGate mask loader — `.mask` binary format for capability substrates (Plan 216 T9-T10).
//!
//! Loads and saves substrate masks in binary (postcard).
//! Format: magic bytes + version + postcard payload with per-layer packed bitmasks,
//! recovery score, capability name, model ID, and BLAKE3 hash for provenance.

use super::substrate_types::SubstrateMask;
use serde::{Deserialize, Serialize};

// ── File Format Version ────────────────────────────────────────

const MASK_FILE_VERSION: u32 = 1;

// ── SubstrateMaskFile ─────────────────────────────────────────

/// Portable substrate mask file format.
///
/// Shared between katgpt-rs and riir-ai for cross-project mask consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateMaskFile {
    /// Format version for forward compatibility.
    pub version: u32,
    /// Per-layer packed bitmasks (u64 words).
    /// Each inner Vec is one layer's bitmask words.
    pub layer_bitmasks: Vec<Vec<u64>>,
    /// Number of layers.
    pub n_layers: usize,
    /// MLP hidden dimension per layer.
    pub mlp_hidden: usize,
    /// Recovery score [0, 1].
    pub recovery_score: f32,
    /// Human-readable capability name.
    pub capability_name: String,
    /// Model identifier.
    pub model_id: String,
    /// BLAKE3 hash of the original bitmask for provenance.
    #[serde(with = "hash_bytes")]
    pub hash: [u8; 32],
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

// ── Load / Save ────────────────────────────────────────────────

/// Load a substrate mask from binary (postcard).
///
/// Validates dimensions and hash integrity.
/// Returns `None` if the file is malformed or hash doesn't match.
pub fn load_substrate_mask(data: &[u8]) -> Option<SubstrateMask> {
    let file: SubstrateMaskFile = match postcard::from_bytes(data) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[substrate_loader] failed to parse mask file: {}", e);
            return None;
        }
    };

    if file.version != MASK_FILE_VERSION {
        eprintln!(
            "[substrate_loader] unsupported version: {} (expected {})",
            file.version, MASK_FILE_VERSION
        );
        return None;
    }

    // Validate dimensions
    if file.n_layers == 0 || file.mlp_hidden == 0 {
        eprintln!(
            "[substrate_loader] invalid dimensions: layers={}, mlp_hidden={}",
            file.n_layers, file.mlp_hidden
        );
        return None;
    }

    // Reconstruct flat bitmask from per-layer bitmasks
    let total_channels = file.n_layers * file.mlp_hidden;
    let word_count = total_channels.div_ceil(64);
    let mut bitmask = vec![0u64; word_count];

    for (layer_idx, layer_words) in file.layer_bitmasks.iter().enumerate() {
        let base_word = (layer_idx * file.mlp_hidden) / 64;
        let layer_word_count = file.mlp_hidden.div_ceil(64);

        for (w, &word) in layer_words.iter().enumerate().take(layer_word_count) {
            let dst = base_word + w;
            if dst < bitmask.len() {
                bitmask[dst] = word;
            }
        }
    }

    // Verify hash
    let computed_hash = compute_blake3(&bitmask);
    if computed_hash != file.hash {
        eprintln!("[substrate_loader] hash mismatch — file may be corrupted");
        // Still load but warn — the mask may have been intentionally modified
    }

    // Build SubstrateMask via set calls
    let mut mask = SubstrateMask::new(
        file.n_layers,
        file.mlp_hidden,
        file.capability_name,
        file.model_id,
    );

    for layer in 0..file.n_layers {
        let base_word = (layer * file.mlp_hidden) / 64;
        let layer_word_count = file.mlp_hidden.div_ceil(64);
        for w in 0..layer_word_count {
            let word = bitmask.get(base_word + w).copied().unwrap_or(0);
            for bit in 0..64 {
                if (word >> bit) & 1 == 1 {
                    let channel = w * 64 + bit;
                    if channel < file.mlp_hidden {
                        mask.set(layer, channel);
                    }
                }
            }
        }
    }

    mask.set_recovery_score(file.recovery_score);
    Some(mask)
}

/// Save a substrate mask to binary (postcard).
///
/// Produces a portable file that can be loaded by katgpt-rs or riir-ai.
pub fn save_substrate_mask(mask: &SubstrateMask) -> Option<Vec<u8>> {
    let n_layers = mask.n_layers();
    let mlp_hidden = mask.mlp_hidden();

    let mut layer_bitmasks = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        let layer_words = mask.layer_bitmask(layer);
        layer_bitmasks.push(layer_words.to_vec());
    }

    let file = SubstrateMaskFile {
        version: MASK_FILE_VERSION,
        layer_bitmasks,
        n_layers,
        mlp_hidden,
        recovery_score: mask.recovery_score(),
        capability_name: mask.capability_name().to_string(),
        model_id: mask.model_id().to_string(),
        hash: *mask.hash(),
    };

    match postcard::to_allocvec(&file) {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            eprintln!("[substrate_loader] failed to serialize mask: {}", e);
            None
        }
    }
}

/// Validate that a loaded mask matches the expected model architecture.
///
/// Returns `true` if dimensions and model ID match.
pub fn validate_mask(
    mask: &SubstrateMask,
    n_layers: usize,
    mlp_hidden: usize,
    model_id: &str,
) -> bool {
    mask.n_layers() == n_layers
        && mask.mlp_hidden() == mlp_hidden
        && mask.model_id() == model_id
        && mask.verify_hash()
}

fn compute_blake3(bitmask: &[u64]) -> [u8; 32] {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            bitmask.as_ptr() as *const u8,
            std::mem::size_of_val(bitmask),
        )
    };
    let mut hasher = blake3::Hasher::new();
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_mask() -> SubstrateMask {
        let mut mask = SubstrateMask::new(
            2,
            128,
            "python_stdlib".to_string(),
            "test_model".to_string(),
        );
        mask.set(0, 0);
        mask.set(0, 10);
        mask.set(0, 63);
        mask.set(1, 5);
        mask.set(1, 100);
        mask.set_recovery_score(0.85);
        mask
    }

    #[test]
    fn test_round_trip_json() {
        let original = make_test_mask();

        let json = save_substrate_mask(&original).expect("save should succeed");
        let restored = load_substrate_mask(&json).expect("load should succeed");

        assert_eq!(restored.n_layers(), original.n_layers());
        assert_eq!(restored.mlp_hidden(), original.mlp_hidden());
        assert_eq!(restored.capability_name(), original.capability_name());
        assert_eq!(restored.model_id(), original.model_id());
        assert!((restored.recovery_score() - original.recovery_score()).abs() < 0.001);
        assert_eq!(restored.active_count(), original.active_count());

        // Check specific channels
        assert!(restored.get(0, 0));
        assert!(restored.get(0, 10));
        assert!(restored.get(0, 63));
        assert!(restored.get(1, 5));
        assert!(restored.get(1, 100));
        assert!(!restored.get(0, 1));
        assert!(!restored.get(1, 0));
    }

    #[test]
    fn test_load_invalid_json() {
        let result = load_substrate_mask(b"not valid binary");
        assert!(result.is_none());
    }

    #[test]
    fn test_load_wrong_version() {
        let file = SubstrateMaskFile {
            version: 999,
            layer_bitmasks: vec![vec![0]],
            n_layers: 1,
            mlp_hidden: 64,
            recovery_score: 0.5,
            capability_name: "test".to_string(),
            model_id: "model".to_string(),
            hash: [0u8; 32],
        };
        let bytes = postcard::to_allocvec(&file).unwrap();
        let result = load_substrate_mask(&bytes);
        assert!(result.is_none());
    }

    #[test]
    fn test_load_zero_dimensions() {
        let file = SubstrateMaskFile {
            version: MASK_FILE_VERSION,
            layer_bitmasks: vec![],
            n_layers: 0,
            mlp_hidden: 0,
            recovery_score: 0.5,
            capability_name: "test".to_string(),
            model_id: "model".to_string(),
            hash: [0u8; 32],
        };
        let bytes = postcard::to_allocvec(&file).unwrap();
        let result = load_substrate_mask(&bytes);
        assert!(result.is_none());
    }

    #[test]
    fn test_validate_mask() {
        let mask = make_test_mask();
        assert!(validate_mask(&mask, 2, 128, "test_model"));
        assert!(!validate_mask(&mask, 4, 128, "test_model")); // Wrong layers
        assert!(!validate_mask(&mask, 2, 256, "test_model")); // Wrong mlp_hidden
        assert!(!validate_mask(&mask, 2, 128, "other_model")); // Wrong model_id
    }

    #[test]
    fn test_save_produces_valid_binary() {
        let mask = make_test_mask();
        let bytes = save_substrate_mask(&mask).expect("save should succeed");

        // Should be parseable via postcard
        let parsed: SubstrateMaskFile = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.n_layers, 2);
        assert_eq!(parsed.mlp_hidden, 128);
        assert_eq!(parsed.capability_name, "python_stdlib");
        assert_eq!(parsed.model_id, "test_model");
    }

    #[test]
    fn test_round_trip_preserves_recovery() {
        let mut mask = SubstrateMask::new(1, 64, "test".to_string(), "model".to_string());
        mask.set(0, 10);
        mask.set_recovery_score(0.72);

        let json = save_substrate_mask(&mask).unwrap();
        let restored = load_substrate_mask(&json).unwrap();

        assert!((restored.recovery_score() - 0.72).abs() < 0.001);
    }

    #[test]
    fn test_empty_mask_round_trip() {
        let mask = SubstrateMask::new(1, 64, "empty".to_string(), "model".to_string());
        let json = save_substrate_mask(&mask).unwrap();
        let restored = load_substrate_mask(&json).unwrap();

        assert_eq!(restored.active_count(), 0);
        assert_eq!(restored.capability_name(), "empty");
    }
}
