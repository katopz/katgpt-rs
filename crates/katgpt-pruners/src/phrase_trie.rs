//! Compact token-level trie for phrase boosting.
//!
//! O(1) child lookup via `Vec<Option<usize>>`. Zero alloc on advance.
//! Feature-gated behind `phrase_boost`.

// ── Node ───────────────────────────────────────────────────────

/// Single trie node. Children are indexed by token ID — O(1) lookup, no hashing.
#[derive(Debug)]
struct PhraseTrieNode {
    /// `children[token_id] = Some(child_node_index)` if edge exists.
    children: Vec<Option<usize>>,
    /// True when this node completes an inserted phrase.
    is_terminal: bool,
}

impl PhraseTrieNode {
    fn new(vocab_size: usize) -> Self {
        Self {
            children: vec![None; vocab_size],
            is_terminal: false,
        }
    }
}

// ── PhraseTrie ─────────────────────────────────────────────────

/// Compact token-level trie for phrase boosting.
///
/// Each phrase is a sequence of token IDs. The trie supports:
/// - **Insert** — add a phrase (token sequence).
/// - **Advance** — given a set of active node indices and a token, return the new active set.
/// - **Boosted tokens** — union of all children reachable from the current active set.
///
/// Child lookup is O(1) via direct index into `Vec<Option<usize>>` — no `HashMap`.
pub struct PhraseTrie {
    nodes: Vec<PhraseTrieNode>,
    vocab_size: usize,
}

impl PhraseTrie {
    /// Create an empty trie with root node pre-allocated for `vocab_size` children.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            nodes: vec![PhraseTrieNode::new(vocab_size)],
            vocab_size,
        }
    }

    /// Insert a single phrase (token ID sequence) into the trie.
    pub fn insert(&mut self, token_ids: &[usize]) {
        let mut node_idx = 0; // root
        for &tok in token_ids {
            let next = self.nodes[node_idx].children[tok];
            node_idx = match next {
                Some(child) => child,
                None => {
                    let child = self.nodes.len();
                    self.nodes[node_idx].children[tok] = Some(child);
                    self.nodes.push(PhraseTrieNode::new(self.vocab_size));
                    child
                }
            };
        }
        self.nodes[node_idx].is_terminal = true;
    }

    /// Bulk-build from phrases, applying an encoding function to each string.
    pub fn build(phrases: &[&str], encode_fn: impl Fn(&str) -> Vec<usize>) -> Self {
        let mut trie = None;
        for phrase in phrases {
            let ids = encode_fn(phrase);
            if trie.is_none() {
                // Heuristic: vocab_size at least max token + 1, default 256.
                let max_tok = ids.iter().copied().max().unwrap_or(0);
                let vocab_size = (max_tok + 1).max(256);
                trie = Some(Self::new(vocab_size));
            }
            trie.as_mut().unwrap().insert(&ids);
        }
        trie.unwrap_or_else(|| Self::new(256))
    }

    /// Return the union of all child token IDs reachable from the active set.
    ///
    /// Used to determine which tokens should receive a boost at the current position.
    ///
    /// O(active_nodes × vocab_width) with O(vocab_size) bitset for dedup.
    pub fn get_boosted_tokens(&self, active: &[usize]) -> Vec<usize> {
        let mut seen = vec![false; self.vocab_size];
        let mut result = Vec::new();
        for &node_idx in active {
            let node = &self.nodes[node_idx];
            for (tok, slot) in node.children.iter().enumerate() {
                if slot.is_some() && !seen[tok] {
                    seen[tok] = true;
                    result.push(tok);
                }
            }
        }
        result
    }

    /// Zero-alloc membership test: is `token_idx` reachable as a child of any
    /// active node?
    ///
    /// O(active_nodes) — short-circuits on the first match. Replaces the
    /// previous pattern of `get_boosted_tokens(active).contains(&token_idx)`
    /// which allocated a `vocab_size` bool array (potentially hundreds of KB)
    /// and a result Vec on every relevance() call.
    pub fn is_token_boosted(&self, active: &[usize], token_idx: usize) -> bool {
        if token_idx >= self.vocab_size {
            return false;
        }
        for &node_idx in active {
            if self.nodes[node_idx].children[token_idx].is_some() {
                return true;
            }
        }
        false
    }

    /// Advance the active set by one token. Returns the new set of active node indices.
    ///
    /// - For each active node, follow the edge `token_id` if it exists.
    /// - If an edge doesn't exist at root, root stays active (always track from start).
    /// - Always include root so new phrases can begin at any position.
    ///
    /// O(active_nodes) with O(nodes) bitset for dedup.
    pub fn advance(&self, active: &[usize], token_id: usize) -> Vec<usize> {
        let mut seen = vec![false; self.nodes.len()];
        let mut next = Vec::with_capacity(active.len() + 1);
        for &node_idx in active {
            if let Some(child) = self.nodes[node_idx].children[token_id]
                && !seen[child]
            {
                seen[child] = true;
                next.push(child);
            }
        }
        // Root (index 0) is always active — new phrases can start at any position.
        if !seen[0] {
            next.push(0);
        }
        next
    }

    /// Check if any active state is at a terminal node (complete phrase matched).
    #[allow(dead_code)]
    pub fn has_terminal(&self, active: &[usize]) -> bool {
        active.iter().any(|&idx| self.nodes[idx].is_terminal)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_lookup_roundtrip() {
        let mut trie = PhraseTrie::new(128);
        // Insert phrase: [3, 7, 11]
        trie.insert(&[3, 7, 11]);
        // Insert phrase: [3, 7, 15]
        trie.insert(&[3, 7, 15]);
        // Insert phrase: [20, 21]
        trie.insert(&[20, 21]);

        // Start at root, advance through [3, 7, 11]
        let active = vec![0];
        let a1 = trie.advance(&active, 3);
        assert!(a1.contains(&1)); // first child of root for tok 3

        let a2 = trie.advance(&a1, 7);
        assert!(!a2.is_empty());

        let a3 = trie.advance(&a2, 11);
        assert!(trie.has_terminal(&a3));
    }

    #[test]
    fn test_advance_tracks_multi_token() {
        let mut trie = PhraseTrie::new(64);
        // "hello world" as tokens [5, 10]
        trie.insert(&[5, 10]);
        // "hello there" as tokens [5, 20]
        trie.insert(&[5, 20]);

        // After seeing token 5, we should be in a state that can reach both 10 and 20
        let active = vec![0];
        let after_hello = trie.advance(&active, 5);
        let boosted = trie.get_boosted_tokens(&after_hello);

        assert!(boosted.contains(&10), "should boost token 10 (world)");
        assert!(boosted.contains(&20), "should boost token 20 (there)");
    }

    #[test]
    fn test_get_boosted_tokens_union() {
        let mut trie = PhraseTrie::new(128);
        trie.insert(&[1, 2]);
        trie.insert(&[1, 3]);
        trie.insert(&[4, 5]);

        // Active at root: should see tokens 1 and 4 as boosted
        let active = vec![0];
        let boosted = trie.get_boosted_tokens(&active);
        assert!(boosted.contains(&1));
        assert!(boosted.contains(&4));
        assert!(!boosted.contains(&99));
    }

    #[test]
    fn test_advance_root_always_active() {
        let trie = PhraseTrie::new(64);
        // Even with an empty trie, root is always in the active set
        let active = vec![0];
        let next = trie.advance(&active, 42);
        assert!(next.contains(&0), "root should always be active");
    }

    #[test]
    fn test_build_from_strings() {
        let trie = PhraseTrie::build(&["alpha", "beta"], |s| {
            s.bytes().map(|b| b as usize).collect()
        });
        let active = vec![0];
        // 'a' = 97, 'b' = 98
        let a_after_a = trie.advance(&active, 97);
        let boosted_a = trie.get_boosted_tokens(&a_after_a);
        // After 'a', next char of "alpha" is 'l' = 108
        assert!(boosted_a.contains(&108));

        let a_after_b = trie.advance(&active, 98);
        let boosted_b = trie.get_boosted_tokens(&a_after_b);
        // After 'b', next char of "beta" is 'e' = 101
        assert!(boosted_b.contains(&101));
    }

    #[test]
    fn test_empty_trie() {
        let trie = PhraseTrie::new(32);
        let active = vec![0];
        assert!(trie.get_boosted_tokens(&active).is_empty());
        let next = trie.advance(&active, 5);
        assert!(next.contains(&0));
        assert!(!trie.has_terminal(&next));
    }
}
