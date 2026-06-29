use super::types::{BpeTokenizer, MergeRule};
use std::collections::HashMap;

/// BPE encoder/decoder implementation.
pub struct BpeTokenizerImpl;

impl BpeTokenizerImpl {
    /// Encode a string into token IDs using BPE merge rules.
    ///
    /// Hot-path design: operates on `Vec<usize>` (token IDs) end-to-end. The
    /// merge-rank lookup uses `merge_ranks_id: HashMap<(usize, usize), usize>`
    /// — no `String` allocation per pair. The replacement ID is resolved via
    /// `merge_target_id[rank]` — no `vocab_to_id` lookup per merge pass.
    ///
    /// Per AGENTS.md hot-loop rules: no allocation inside the merge loop.
    /// The only allocation is the initial char→ID map and the two ping-pong
    /// token buffers, both pre-sized.
    pub fn encode(tokenizer: &BpeTokenizer, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return Vec::new();
        }

        // Map each char to its token ID up front. Unknown chars map to `unk`.
        // Uses a fixed-size stack buffer (`encode_utf8` writes ≤4 bytes) —
        // zero heap allocation for the entire char→ID map step.
        let unk = tokenizer.unk_id();
        let char_count = text.chars().count();
        let mut tokens: Vec<usize> = Vec::with_capacity(char_count);
        let mut buf = [0u8; 4];
        for c in text.chars() {
            let s = c.encode_utf8(&mut buf);
            let id = tokenizer.vocab_to_id.get(s).copied().unwrap_or(unk);
            tokens.push(id);
        }

        // Fast path: no merges configured (or tables not rebuilt).
        if tokenizer.merge_ranks_id.is_empty() {
            return tokens;
        }

        let mut new_tokens: Vec<usize> = Vec::with_capacity(tokens.len());

        // Iteratively merge the highest-priority (lowest-rank) pair.
        loop {
            // Find the lowest-rank applicable merge across all adjacent pairs.
            // `windows(2)` lets LLVM drop the per-iteration bounds check on
            // `tokens[i + 1]` that the manual index loop forces.
            let mut best: Option<(usize, usize)> = None; // (rank, left_idx)
            for (i, w) in tokens.windows(2).enumerate() {
                if let Some(&rank) = tokenizer.merge_ranks_id.get(&(w[0], w[1])) {
                    match best {
                        Some((best_rank, _)) if best_rank <= rank => {}
                        _ => best = Some((rank, i)),
                    }
                }
            }

            let Some((best_rank, left_idx)) = best else { break };

            // Resolve the merged token ID via the rank-indexed table — no
            // hashmap lookup, just a slice index.
            let merged_id = tokenizer.merge_target_id[best_rank];
            let left_id = tokens[left_idx];
            let right_id = tokens[left_idx + 1];

            // Apply the merge to all adjacent occurrences of (left, right).
            // Indices are `usize` (Copy) — zero allocation in this loop.
            new_tokens.clear();
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == left_id && tokens[i + 1] == right_id {
                    new_tokens.push(merged_id);
                    i += 2;
                } else {
                    new_tokens.push(tokens[i]);
                    i += 1;
                }
            }
            std::mem::swap(&mut tokens, &mut new_tokens);
        }

        tokens
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
            merge_ranks_id: HashMap::new(),
            merge_target_id: Vec::new(),
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
