use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// BPE tokenizer with vocabulary and merge rules.
#[derive(Clone, Serialize, Deserialize)]
pub struct BpeTokenizer {
    /// Token string → ID mapping.
    #[serde(with = "map_serde")]
    pub vocab_to_id: HashMap<String, usize>,
    /// ID → token string mapping.
    pub id_to_vocab: Vec<String>,
    /// Ordered merge rules: apply in this order during encoding.
    pub merges: Vec<MergeRule>,
    /// Merge rule → rank (index in merges vec) for fast lookup.
    #[serde(skip)]
    pub merge_ranks: HashMap<(String, String), usize>,
    /// ID-pair → merge rank (lower = higher priority). Hot-path lookup table
    /// for `encode`, which operates on `Vec<usize>` end-to-end and avoids
    /// per-pair `String` allocations. Kept in sync with `merge_ranks` by
    /// `rebuild_ranks`.
    #[serde(skip)]
    pub merge_ranks_id: HashMap<(usize, usize), usize>,
    /// `merge_target_id[rank]` = merged token ID for the rule at that rank.
    /// Indexed by rank so `encode` can resolve the replacement with a single
    /// indexed load instead of a `vocab_to_id` lookup per merge pass.
    #[serde(skip)]
    pub merge_target_id: Vec<usize>,
    /// Beginning-of-sequence token ID.
    pub bos_id: usize,
    /// End-of-sequence token ID.
    pub eos_id: usize,
    /// Padding token ID.
    pub pad_id: usize,
}

/// A single BPE merge rule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MergeRule {
    pub left: String,
    pub right: String,
    pub merged: String,
}

impl BpeTokenizer {
    /// Unknown token ID (last entry in vocabulary).
    pub fn unk_id(&self) -> usize {
        self.id_to_vocab.len().saturating_sub(1)
    }

    /// Rebuild merge_ranks from merges vector.
    ///
    /// Also rebuilds the ID-keyed tables (`merge_ranks_id`, `merge_target_id`)
    /// that `encode` uses to skip per-pair `String` allocation in the hot path.
    pub fn rebuild_ranks(&mut self) {
        let n = self.merges.len();
        self.merge_ranks = HashMap::with_capacity(n);
        self.merge_ranks_id = HashMap::with_capacity(n);
        self.merge_target_id = Vec::with_capacity(n);
        let unk = self.unk_id();
        for (rank, rule) in self.merges.iter().enumerate() {
            self.merge_ranks
                .insert((rule.left.clone(), rule.right.clone()), rank);
            // Resolve left/right/merged to IDs. Unknown tokens map to `unk`,
            // which simply makes the rule inert in the ID-keyed path (its key
            // won't match real token pairs whose IDs differ from `unk`).
            let l = self.vocab_to_id.get(&rule.left).copied().unwrap_or(unk);
            let r = self.vocab_to_id.get(&rule.right).copied().unwrap_or(unk);
            let m = self.vocab_to_id.get(&rule.merged).copied().unwrap_or(unk);
            self.merge_ranks_id.insert((l, r), rank);
            self.merge_target_id.push(m);
        }
    }
}

mod map_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S: Serializer>(map: &HashMap<String, usize>, s: S) -> Result<S::Ok, S::Error> {
        let vec: Vec<(&String, &usize)> = map.iter().collect();
        vec.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<HashMap<String, usize>, D::Error> {
        let vec: Vec<(String, usize)> = Vec::deserialize(d)?;
        Ok(vec.into_iter().collect())
    }
}

/// Serde module for `HashMap<Vec<u8>, usize>` — keys as hex strings for readability.
/// Used by ToaST tokenizer (Plan 122). Feature-gated to avoid dead_code warnings.
#[cfg(feature = "toast_tokenizer")]
#[allow(dead_code)]
pub mod map_serde_bytes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::HashMap;

    pub fn serialize<S: Serializer>(
        map: &HashMap<Vec<u8>, usize>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let vec: Vec<(String, usize)> = map.iter().map(|(k, &v)| (bytes_to_hex(k), v)).collect();
        vec.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<HashMap<Vec<u8>, usize>, D::Error> {
        let vec: Vec<(String, usize)> = Vec::deserialize(d)?;
        Ok(vec
            .into_iter()
            .filter_map(|(hex, id)| hex_to_bytes(&hex).map(|bytes| (bytes, id)))
            .collect())
    }

    fn bytes_to_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn hex_to_bytes(hex: &str) -> Option<Vec<u8>> {
        if !hex.len().is_multiple_of(2) {
            return None;
        }
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok())
            .collect()
    }
}
