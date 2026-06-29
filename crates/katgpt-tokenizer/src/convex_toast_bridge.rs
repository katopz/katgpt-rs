//! Bridge from ConvexTok rounded vocabulary to ToaST tokenizer for inference.
//!
//! ConvexTok optimizes **which** tokens to include; ToaST optimizes **how** to segment
//! with those tokens. This bridge converts the LP-optimized vocabulary into a ToaST
//! tokenizer ready for inference.
//!
//! **Source:** Tempus et al. (2026). Tokenisation via Convex Relaxations. arXiv:2605.22821

use std::collections::HashMap;

use super::convex_types::RoundedVocabulary;
use super::toast_builder::SplitTreeBuilder;
use super::toast_types::ToastTokenizer;

/// Default minimum n-gram count for split tree construction.
const DEFAULT_MIN_COUNT: u64 = 10;

/// Special token bytes for the tokenizer.
#[derive(Clone, Debug)]
pub struct SpecialTokens {
    /// Beginning-of-sequence token bytes.
    pub bos: Vec<u8>,
    /// End-of-sequence token bytes.
    pub eos: Vec<u8>,
    /// Padding token bytes.
    pub pad: Vec<u8>,
    /// Unknown token bytes.
    pub unk: Vec<u8>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos: vec![b'<', b'B', b'O', b'S', b'>'],
            eos: vec![b'<', b'E', b'O', b'S', b'>'],
            pad: vec![b'<', b'P', b'A', b'D', b'>'],
            unk: vec![b'<', b'U', b'N', b'K', b'>'],
        }
    }
}

/// Bridge from ConvexTok rounded vocabulary to ToaST tokenizer.
///
/// ConvexTok determines the optimal vocabulary via LP relaxation.
/// ToaST determines optimal segmentation via split trees.
/// This bridge connects the two: vocabulary selection → inference-ready tokenizer.
pub struct ConvexToToastBridge;

impl ConvexToToastBridge {
    /// Convert a ConvexTok rounded vocabulary to a ToaST tokenizer.
    ///
    /// # Algorithm
    /// 1. Register special tokens first (BOS, EOS, PAD, UNK)
    /// 2. Register all 256 single bytes (0x00–0xFF)
    /// 3. Register selected multi-byte tokens from ConvexTok
    /// 4. Build split trees for each selected multi-byte token using n-gram counts
    /// 5. Return `ToastTokenizer` ready for inference
    ///
    /// # Arguments
    /// * `rounded` — The ConvexTok rounded vocabulary (LP-optimized token selection)
    /// * `ngram_counts` — Byte n-gram frequency counts for split tree construction
    /// * `special_tokens` — Special token byte sequences (BOS, EOS, PAD, UNK)
    pub fn to_toast_tokenizer(
        rounded: &RoundedVocabulary,
        ngram_counts: &HashMap<Vec<u8>, u64>,
        special_tokens: &SpecialTokens,
    ) -> ToastTokenizer {
        Self::to_toast_tokenizer_with_min_count(
            rounded,
            ngram_counts,
            special_tokens,
            DEFAULT_MIN_COUNT,
        )
    }

    /// Convert with a custom minimum n-gram count for split tree construction.
    ///
    /// Lower `min_count` produces more split candidates but may include noisy splits.
    /// Higher `min_count` produces conservative splits from well-supported n-grams.
    pub fn to_toast_tokenizer_with_min_count(
        rounded: &RoundedVocabulary,
        ngram_counts: &HashMap<Vec<u8>, u64>,
        special_tokens: &SpecialTokens,
        min_count: u64,
    ) -> ToastTokenizer {
        let mut vocab_to_id: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut id_to_vocab: Vec<Vec<u8>> = Vec::new();

        let mut add_token = |bytes: Vec<u8>| -> usize {
            match vocab_to_id.get(&bytes) {
                Some(&id) => id,
                None => {
                    let id = id_to_vocab.len();
                    vocab_to_id.insert(bytes.clone(), id);
                    id_to_vocab.push(bytes);
                    id
                }
            }
        };

        // Step 1: Special tokens first (IDs 0–3)
        let bos_id = add_token(special_tokens.bos.clone());
        let eos_id = add_token(special_tokens.eos.clone());
        let pad_id = add_token(special_tokens.pad.clone());
        let unk_id = add_token(special_tokens.unk.clone());

        // Step 2: All 256 single bytes (IDs 4–259)
        for b in 0u8..=255 {
            add_token(vec![b]);
        }

        // Step 3: Selected multi-byte tokens from ConvexTok
        let mut multi_byte_tokens: Vec<&[u8]> = Vec::new();
        for bytes in &rounded.selected_bytes {
            if bytes.len() >= 2 {
                add_token(bytes.clone());
                multi_byte_tokens.push(bytes.as_slice());
            }
        }

        // Step 4: Build split trees for each selected multi-byte token
        let builder = SplitTreeBuilder::new(ngram_counts, min_count);
        let mut trees = HashMap::new();
        for pretoken in &multi_byte_tokens {
            let tree = builder.build(pretoken);
            trees.insert(pretoken.to_vec(), tree);
        }

        let mut tokenizer = ToastTokenizer {
            vocab_to_id,
            id_to_vocab,
            trees,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
            datrie_vocab: None,
        };
        tokenizer.post_load();
        tokenizer
    }

    /// Convenience: build a tokenizer with default special tokens.
    pub fn to_toast_tokenizer_default_special(
        rounded: &RoundedVocabulary,
        ngram_counts: &HashMap<Vec<u8>, u64>,
    ) -> ToastTokenizer {
        Self::to_toast_tokenizer(rounded, ngram_counts, &SpecialTokens::default())
    }
}

#[cfg(test)]
mod tests {
    use super::super::convex_types::{ColourId, RoundingScheme};
    use super::*;

    /// Helper: create a minimal rounded vocabulary for testing.
    fn make_rounded_vocab(selected_bytes: Vec<Vec<u8>>) -> RoundedVocabulary {
        let n_selected = selected_bytes.len();
        let selected_colours = (0..n_selected).map(|i| ColourId(i as u32)).collect();
        RoundedVocabulary {
            selected_colours,
            selected_bytes,
            n_selected,
            compression_value: 0.5,
            rounding_scheme: RoundingScheme::Det,
        }
    }

    /// Helper: create simple n-gram counts for testing.
    fn make_ngram_counts() -> HashMap<Vec<u8>, u64> {
        let mut counts = HashMap::new();
        // Single byte counts
        for b in b"abcdefg" {
            counts.insert(vec![*b], 100);
        }
        // Bigrams
        counts.insert(b"ab".to_vec(), 50);
        counts.insert(b"cd".to_vec(), 50);
        counts.insert(b"ef".to_vec(), 50);
        counts.insert(b"bc".to_vec(), 30);
        counts.insert(b"de".to_vec(), 30);
        counts.insert(b"fg".to_vec(), 30);
        // Trigrams
        counts.insert(b"abc".to_vec(), 20);
        counts.insert(b"def".to_vec(), 20);
        counts.insert(b"cde".to_vec(), 15);
        // 4-grams
        counts.insert(b"abcd".to_vec(), 10);
        counts.insert(b"cdef".to_vec(), 10);
        counts.insert(b"bcde".to_vec(), 8);
        counts
    }

    #[test]
    fn test_bridge_basic_construction() {
        let rounded = make_rounded_vocab(vec![b"ab".to_vec(), b"cd".to_vec(), b"abcd".to_vec()]);
        let counts = make_ngram_counts();
        let special = SpecialTokens::default();

        let tokenizer = ConvexToToastBridge::to_toast_tokenizer(&rounded, &counts, &special);

        // 4 special + 256 single bytes + 3 multi-byte = 263
        assert_eq!(tokenizer.vocab_size(), 263);
    }

    #[test]
    fn test_special_token_ids() {
        let rounded = make_rounded_vocab(vec![]);
        let counts = make_ngram_counts();
        let special = SpecialTokens::default();

        let tokenizer = ConvexToToastBridge::to_toast_tokenizer(&rounded, &counts, &special);

        // Special tokens come first (IDs 0–3)
        assert_eq!(tokenizer.bos_id, 0);
        assert_eq!(tokenizer.eos_id, 1);
        assert_eq!(tokenizer.pad_id, 2);
        assert_eq!(tokenizer.unk_id, 3);

        // Verify byte content
        assert_eq!(tokenizer.id_to_vocab[0], b"<BOS>");
        assert_eq!(tokenizer.id_to_vocab[1], b"<EOS>");
        assert_eq!(tokenizer.id_to_vocab[2], b"<PAD>");
        assert_eq!(tokenizer.id_to_vocab[3], b"<UNK>");
    }

    #[test]
    fn test_single_bytes_registered() {
        let rounded = make_rounded_vocab(vec![]);
        let counts = make_ngram_counts();
        let special = SpecialTokens::default();

        let tokenizer = ConvexToToastBridge::to_toast_tokenizer(&rounded, &counts, &special);

        // All 256 single bytes should be in vocab (IDs 4–259)
        for b in 0u8..=255u8 {
            let id = tokenizer.vocab_to_id.get(&vec![b]);
            assert!(id.is_some(), "byte {b} not in vocab");
            assert_eq!(*id.unwrap(), 4 + b as usize);
        }
    }

    #[test]
    fn test_multi_byte_tokens_registered() {
        let rounded = make_rounded_vocab(vec![b"ab".to_vec(), b"cde".to_vec()]);
        let counts = make_ngram_counts();
        let special = SpecialTokens::default();

        let tokenizer = ConvexToToastBridge::to_toast_tokenizer(&rounded, &counts, &special);

        // Multi-byte tokens come after single bytes
        assert!(tokenizer.vocab_to_id.contains_key(b"ab".as_slice()));
        assert!(tokenizer.vocab_to_id.contains_key(b"cde".as_slice()));
    }

    #[test]
    fn test_single_byte_selected_tokens_not_duplicated() {
        // Single byte "a" should not create a duplicate — already registered as single byte
        let rounded = make_rounded_vocab(vec![
            b"a".to_vec(), // single byte, should be skipped as multi-byte
            b"ab".to_vec(),
        ]);
        let counts = make_ngram_counts();
        let special = SpecialTokens::default();

        let tokenizer = ConvexToToastBridge::to_toast_tokenizer(&rounded, &counts, &special);

        // 4 special + 256 single bytes + 1 multi-byte ("ab") = 261
        assert_eq!(tokenizer.vocab_size(), 261);
    }

    #[test]
    fn test_split_trees_built_for_multi_byte_tokens() {
        let rounded = make_rounded_vocab(vec![b"ab".to_vec(), b"abcd".to_vec(), b"cdef".to_vec()]);
        let counts = make_ngram_counts();
        let special = SpecialTokens::default();

        let tokenizer = ConvexToToastBridge::to_toast_tokenizer(&rounded, &counts, &special);

        // Split trees should exist for each multi-byte token
        assert!(tokenizer.trees.contains_key(b"ab".as_slice()));
        assert!(tokenizer.trees.contains_key(b"abcd".as_slice()));
        assert!(tokenizer.trees.contains_key(b"cdef".as_slice()));

        // Each tree's pretoken should match
        let tree_ab = tokenizer.trees.get(b"ab".as_slice()).unwrap();
        assert_eq!(tree_ab.pretoken, b"ab");

        let tree_abcd = tokenizer.trees.get(b"abcd".as_slice()).unwrap();
        assert_eq!(tree_abcd.pretoken, b"abcd");
        assert!(tree_abcd.nodes[0].left.is_some());
        assert!(tree_abcd.nodes[0].right.is_some());
    }

    #[test]
    fn test_custom_special_tokens() {
        let rounded = make_rounded_vocab(vec![]);
        let counts = make_ngram_counts();
        let special = SpecialTokens {
            bos: b"[BOS]".to_vec(),
            eos: b"[EOS]".to_vec(),
            pad: b"[PAD]".to_vec(),
            unk: b"[UNK]".to_vec(),
        };

        let tokenizer = ConvexToToastBridge::to_toast_tokenizer(&rounded, &counts, &special);

        assert_eq!(tokenizer.id_to_vocab[tokenizer.bos_id], b"[BOS]");
        assert_eq!(tokenizer.id_to_vocab[tokenizer.eos_id], b"[EOS]");
        assert_eq!(tokenizer.id_to_vocab[tokenizer.pad_id], b"[PAD]");
        assert_eq!(tokenizer.id_to_vocab[tokenizer.unk_id], b"[UNK]");
    }

    #[test]
    fn test_custom_min_count() {
        let rounded = make_rounded_vocab(vec![b"abcd".to_vec()]);
        let counts = make_ngram_counts();

        let tokenizer_high = ConvexToToastBridge::to_toast_tokenizer_with_min_count(
            &rounded,
            &counts,
            &SpecialTokens::default(),
            100, // Very high — few splits qualify
        );

        let tokenizer_low = ConvexToToastBridge::to_toast_tokenizer_with_min_count(
            &rounded,
            &counts,
            &SpecialTokens::default(),
            1, // Very low — most splits qualify
        );

        // Both should have same vocabulary
        assert_eq!(tokenizer_high.vocab_size(), tokenizer_low.vocab_size());

        // But trees may differ in structure (high min_count → fewer valid splits → simpler trees)
        let tree_high = tokenizer_high.trees.get(b"abcd".as_slice()).unwrap();
        let tree_low = tokenizer_low.trees.get(b"abcd".as_slice()).unwrap();

        // Both trees cover the same pretoken
        assert_eq!(tree_high.pretoken, tree_low.pretoken);
    }

    #[test]
    fn test_default_special_convenience() {
        let rounded = make_rounded_vocab(vec![b"ab".to_vec()]);
        let counts = make_ngram_counts();

        let tokenizer = ConvexToToastBridge::to_toast_tokenizer_default_special(&rounded, &counts);

        assert_eq!(tokenizer.id_to_vocab[tokenizer.bos_id], b"<BOS>");
        assert_eq!(tokenizer.vocab_size(), 261);
    }

    #[test]
    fn test_empty_rounded_vocabulary() {
        let rounded = make_rounded_vocab(vec![]);
        let counts = make_ngram_counts();

        let tokenizer = ConvexToToastBridge::to_toast_tokenizer_default_special(&rounded, &counts);

        // 4 special + 256 single bytes = 260
        assert_eq!(tokenizer.vocab_size(), 260);
        assert!(tokenizer.trees.is_empty());
    }

    #[test]
    fn test_duplicate_selected_bytes_no_panic() {
        // ConvexTok shouldn't produce duplicates, but handle gracefully
        let rounded = make_rounded_vocab(vec![
            b"ab".to_vec(),
            b"ab".to_vec(), // duplicate
            b"cd".to_vec(),
        ]);
        let counts = make_ngram_counts();

        let tokenizer = ConvexToToastBridge::to_toast_tokenizer_default_special(&rounded, &counts);

        // Duplicates should be deduped via add_token
        assert_eq!(
            tokenizer.vocab_to_id.get(b"ab".as_slice()).copied(),
            Some(260)
        );
        // 4 special + 256 single bytes + 2 unique multi-byte = 262
        assert_eq!(tokenizer.vocab_size(), 262);
    }

    #[test]
    fn test_vocab_to_id_and_id_to_vocab_consistent() {
        let rounded = make_rounded_vocab(vec![b"ab".to_vec(), b"cde".to_vec(), b"abcd".to_vec()]);
        let counts = make_ngram_counts();

        let tokenizer = ConvexToToastBridge::to_toast_tokenizer_default_special(&rounded, &counts);

        // Every entry in vocab_to_id should have a corresponding id_to_vocab entry
        for (bytes, &id) in &tokenizer.vocab_to_id {
            assert_eq!(
                tokenizer.id_to_vocab.get(id),
                Some(bytes),
                "Mismatch: vocab_to_id[{id}] != id_to_vocab[{id}]"
            );
        }

        // Every entry in id_to_vocab should have a corresponding vocab_to_id entry
        for (id, bytes) in tokenizer.id_to_vocab.iter().enumerate() {
            assert_eq!(
                tokenizer.vocab_to_id.get(bytes),
                Some(&id),
                "Mismatch: id_to_vocab[{id}] not in vocab_to_id"
            );
        }
    }
}
