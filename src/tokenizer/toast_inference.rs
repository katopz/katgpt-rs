//! ToaST recursive descent tokenization inference.
//!
//! **Source:** Schmidt et al. (2026). Tokenization with Split Trees. arXiv:2605.22705

use super::toast_types::{SplitTree, ToastTokenizer};

/// ToaST tokenizer encoder/decoder implementation.
pub struct ToastTokenizerImpl;

impl ToastTokenizerImpl {
    /// Encode a string into token IDs using ToaST recursive descent.
    pub fn encode(tokenizer: &ToastTokenizer, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return Vec::new();
        }

        let bytes = text.as_bytes();
        let mut token_ids = Vec::with_capacity(bytes.len());

        // Simple pretokenization: split on whitespace boundaries.
        // Each "word" is a pretoken, whitespace is its own token(s).
        let mut start = 0;
        for (i, &b) in bytes.iter().enumerate() {
            if b.is_ascii_whitespace() {
                // Process word before whitespace
                if start < i {
                    Self::encode_pretoken(tokenizer, &bytes[start..i], &mut token_ids);
                }
                // Encode whitespace byte directly via routed lookup
                match tokenizer.vocab_lookup(&bytes[i..i + 1]) {
                    Some(id) => token_ids.push(id),
                    None => token_ids.push(tokenizer.unk_id),
                }
                start = i + 1;
            }
        }
        // Handle last word
        if start < bytes.len() {
            Self::encode_pretoken(tokenizer, &bytes[start..], &mut token_ids);
        }

        token_ids
    }

    fn encode_pretoken(tokenizer: &ToastTokenizer, pretoken: &[u8], token_ids: &mut Vec<usize>) {
        // Check if full pretoken is in vocab via routed lookup
        if let Some(id) = tokenizer.vocab_lookup(pretoken) {
            token_ids.push(id);
            return;
        }

        // Look up split tree
        match tokenizer.trees.get(pretoken) {
            Some(tree) => {
                Self::recursive_descent(tree, 0, tokenizer, token_ids, tokenizer.unk_id);
            }
            None => {
                // Fallback: encode byte by byte
                for &b in pretoken {
                    match tokenizer.vocab_lookup(&[b]) {
                        Some(id) => token_ids.push(id),
                        None => token_ids.push(tokenizer.unk_id),
                    }
                }
            }
        }
    }

    fn recursive_descent(
        tree: &SplitTree,
        node_idx: u32,
        tokenizer: &ToastTokenizer,
        tokens: &mut Vec<usize>,
        unk_id: usize,
    ) {
        let node = &tree.nodes[node_idx as usize];
        let start = node.start as usize;
        let end = node.end as usize;
        let span = &tree.pretoken[start..end];

        // If this span is in vocabulary, emit it and stop
        if let Some(id) = tokenizer.vocab_lookup(span) {
            tokens.push(id);
            return;
        }

        // Otherwise, recurse into children
        match (node.left, node.right) {
            (Some(l), Some(r)) => {
                Self::recursive_descent(tree, l, tokenizer, tokens, unk_id);
                Self::recursive_descent(tree, r, tokenizer, tokens, unk_id);
            }
            _ => {
                // Leaf node (single byte) — must be in vocab by construction
                match tokenizer.vocab_lookup(span) {
                    Some(id) => tokens.push(id),
                    None => tokens.push(unk_id),
                }
            }
        }
    }

    /// Decode token IDs back to string.
    pub fn decode(tokenizer: &ToastTokenizer, ids: &[usize]) -> String {
        let total_bytes: usize = ids
            .iter()
            .filter_map(|&id| tokenizer.id_to_vocab.get(id))
            .map(|v| v.len())
            .sum();
        let mut bytes = Vec::with_capacity(total_bytes);
        for &id in ids {
            if let Some(vocab) = tokenizer.id_to_vocab.get(id) {
                bytes.extend_from_slice(vocab);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
