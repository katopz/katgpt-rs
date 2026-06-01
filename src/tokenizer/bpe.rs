use super::types::{BpeTokenizer, MergeRule};
use std::collections::HashMap;

/// BPE encoder/decoder implementation.
pub struct BpeTokenizerImpl;

impl BpeTokenizerImpl {
    /// Encode a string into token IDs using BPE merge rules.
    pub fn encode(tokenizer: &BpeTokenizer, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with character-level tokens — avoid per-char String allocation
        let mut tokens: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let mut new_tokens = Vec::with_capacity(tokens.len());

        // Iteratively merge the highest-priority (lowest-rank) pair
        loop {
            let mut best_pair: Option<(usize, usize, usize)> = None; // (rank, left_idx, right_idx)

            for i in 0..tokens.len().saturating_sub(1) {
                if let Some(&rank) = tokenizer
                    .merge_ranks
                    .get(&(tokens[i].clone(), tokens[i + 1].clone()))
                {
                    match best_pair {
                        Some((best_rank, _, _)) if best_rank <= rank => {}
                        _ => best_pair = Some((rank, i, i + 1)),
                    }
                }
            }

            let Some((_rank, left_idx, _right_idx)) = best_pair else {
                break;
            };

            let left = tokens[left_idx].clone();
            let right = tokens[left_idx + 1].clone();
            let merged_len = left.len() + right.len();

            // Merge all occurrences of this pair
            new_tokens.clear();
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == left && tokens[i + 1] == right {
                    // Merge in-place by concatenating left + right
                    let mut merged = String::with_capacity(merged_len);
                    merged.push_str(&left);
                    merged.push_str(&right);
                    new_tokens.push(merged);
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            std::mem::swap(&mut tokens, &mut new_tokens);
        }

        // Map tokens to IDs
        let unk = tokenizer.unk_id();
        tokens
            .into_iter()
            .map(|t| *tokenizer.vocab_to_id.get(&t).unwrap_or(&unk))
            .collect()
    }

    /// Decode token IDs back to string.
    pub fn decode(tokenizer: &BpeTokenizer, ids: &[usize]) -> String {
        let mut result = String::with_capacity(ids.len() * 4); // estimate ~4 bytes per token
        for &id in ids {
            match tokenizer.id_to_vocab.get(id) {
                Some(s) => result.push_str(s),
                None => result.push('\u{fffd}'), // replacement character
            }
        }
        result
    }
}

/// BPE trainer: learns merge rules from a corpus.
pub struct BpeTrainer;

impl BpeTrainer {
    /// Train a BPE tokenizer from a text corpus.
    /// `vocab_size`: target vocabulary size (including special tokens).
    /// `corpus`: training text.
    pub fn train(corpus: &str, vocab_size: usize) -> BpeTokenizer {
        let mut vocab_to_id: HashMap<String, usize> = HashMap::new();
        let mut id_to_vocab: Vec<String> = Vec::new();

        // Special tokens: <pad>=0, <bos>=1, <eos>=2, <unk>=3
        let special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"];
        for (i, tok) in special_tokens.iter().enumerate() {
            vocab_to_id.insert(tok.to_string(), i);
            id_to_vocab.push(tok.to_string());
        }

        // Add all unique characters from corpus
        for ch in corpus.chars() {
            let s = ch.to_string();
            if !vocab_to_id.contains_key(&s) {
                let id = id_to_vocab.len();
                vocab_to_id.insert(s.clone(), id);
                id_to_vocab.push(s);
            }
        }

        let mut merges: Vec<MergeRule> = Vec::new();
        let num_merges = vocab_size.saturating_sub(id_to_vocab.len());

        // Split corpus into words (simple whitespace splitting)
        let words: Vec<Vec<String>> = corpus
            .split_whitespace()
            .map(|w| w.chars().map(|c| c.to_string()).collect())
            .collect();

        // Learn merge rules
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
        for _ in 0..num_merges {
            // Count all adjacent pairs
            pair_counts.clear();
            for word in &words {
                let tokens = Self::apply_merges(word, &merges);
                for i in 0..tokens.len().saturating_sub(1) {
                    let pair = (tokens[i].clone(), tokens[i + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }

            // Find most frequent pair
            let best_pair = pair_counts.drain().max_by_key(|(_, count)| *count);

            let Some((pair, count)) = best_pair else {
                break;
            };

            if count < 2 {
                break; // Stop if no pair appears more than once
            }

            let merged = format!("{}{}", pair.0, pair.1);

            // Add merged token to vocabulary
            if !vocab_to_id.contains_key(&merged) {
                let id = id_to_vocab.len();
                vocab_to_id.insert(merged.clone(), id);
                id_to_vocab.push(merged.clone());
            }

            merges.push(MergeRule {
                left: pair.0,
                right: pair.1,
                merged,
            });
        }

        let mut tokenizer = BpeTokenizer {
            vocab_to_id,
            id_to_vocab,
            merges,
            merge_ranks: HashMap::new(),
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
        };
        tokenizer.rebuild_ranks();
        tokenizer
    }

    /// Apply existing merge rules to a sequence of tokens.
    fn apply_merges(tokens: &[String], merges: &[MergeRule]) -> Vec<String> {
        let mut buf_a = tokens.to_vec();
        let mut buf_b = Vec::with_capacity(tokens.len());
        for rule in merges {
            buf_b.clear();
            let mut i = 0;
            while i < buf_a.len() {
                if i + 1 < buf_a.len() && buf_a[i] == rule.left && buf_a[i + 1] == rule.right {
                    buf_b.push(rule.merged.clone());
                    i += 2;
                } else {
                    buf_b.push(buf_a[i].clone());
                    i += 1;
                }
            }
            std::mem::swap(&mut buf_a, &mut buf_b);
        }
        buf_a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_encode_decode_roundtrip() {
        let corpus = "hello world hello rust";
        let tokenizer = BpeTrainer::train(corpus, 64);
        let text = "hello";
        let ids = BpeTokenizerImpl::encode(&tokenizer, text);
        let decoded = BpeTokenizerImpl::decode(&tokenizer, &ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_special_tokens() {
        let tokenizer = BpeTrainer::train("abc", 32);
        assert_eq!(tokenizer.pad_id, 0);
        assert_eq!(tokenizer.bos_id, 1);
        assert_eq!(tokenizer.eos_id, 2);
        // unk_id is the last vocab entry
        assert!(tokenizer.unk_id() >= 3);
        // Verify special tokens in vocab
        assert_eq!(tokenizer.vocab_to_id["<pad>"], 0);
        assert_eq!(tokenizer.vocab_to_id["<bos>"], 1);
        assert_eq!(tokenizer.vocab_to_id["<eos>"], 2);
        assert_eq!(tokenizer.vocab_to_id["<unk>"], 3);
    }

    #[test]
    fn test_bpe_train_produces_merges() {
        // Use a corpus with repeated patterns to guarantee merges
        let corpus = "ab ab ab ab ab ab ab ab ab ab";
        let tokenizer = BpeTrainer::train(corpus, 64);
        // "a" + "b" → "ab" should be learned as a merge
        assert!(
            !tokenizer.merges.is_empty(),
            "Expected at least one merge rule from repeated 'ab' patterns"
        );
        // Verify the merge exists
        let has_ab_merge = tokenizer
            .merges
            .iter()
            .any(|m| m.left == "a" && m.right == "b" && m.merged == "ab");
        assert!(has_ab_merge, "Expected merge rule 'a'+'b'→'ab'");
    }

    #[test]
    fn test_bpe_vocab_coverage() {
        let corpus = "hello world";
        let tokenizer = BpeTrainer::train(corpus, 64);
        // All characters from the corpus must be in the vocabulary
        for ch in corpus.chars() {
            let s = ch.to_string();
            assert!(
                tokenizer.vocab_to_id.contains_key(&s),
                "Character '{s}' missing from vocabulary"
            );
        }
    }

    #[test]
    fn test_bpe_encode_empty() {
        let tokenizer = BpeTrainer::train("hello", 32);
        let ids = BpeTokenizerImpl::encode(&tokenizer, "");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_bpe_decode_unknown_id() {
        let tokenizer = BpeTrainer::train("hello", 32);
        // Use an out-of-range ID
        let decoded = BpeTokenizerImpl::decode(&tokenizer, &[9999]);
        assert_eq!(decoded, "�");
    }
}
