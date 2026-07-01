//! Double-array trie for zero-alloc byte-key lookups (Research 137).
//!
//! Replaces `HashMap<Vec<u8>, usize>` and `HashMap<Vec<u8>, SplitTree>`
//! with a compact two-array structure: `base[]` + `check[]`, plus an
//! optional `value[]` slot per node. Lookup is one add + one compare per
//! byte — no hashing, no allocation.
//!
//! **Source:** Aoe, J. (1989). An efficient digital search algorithm by using
//! a double-array structure. IEEE TSE.
//!
//! Feature-gated behind `datrie_vocab` — default OFF until benchmarked.

use std::collections::HashMap;

// ── Core double-array trie ──────────────────────────────────────────────────

/// Double-array trie storing an optional `u32` value at each node.
///
/// Invariant: for a transition on byte `b` from state `s`:
///   child = base[s] + b  (as usize, base is always ≥ 0 after build)
///   check[child] == s     (ownership guard)
///   value[child] = Some(v) if this node is a terminal
struct Datrie {
    base: Vec<i32>,
    check: Vec<u32>,
    value: Vec<Option<u32>>,
}

impl Clone for Datrie {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            check: self.check.clone(),
            value: self.value.clone(),
        }
    }
}

impl std::fmt::Debug for Datrie {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Datrie")
            .field("slots", &self.base.len())
            .finish()
    }
}

/// Sentinel value: check[s] == UNDEF means slot `s` is unoccupied.
const UNDEF: u32 = u32::MAX;

impl Datrie {
    /// Build an empty trie with a pre-allocated capacity.
    fn with_capacity(cap: usize) -> Self {
        let mut base = vec![0i32; cap];
        let check = vec![UNDEF; cap];
        let value = vec![None; cap];
        // Slot 0 is the root; base[0] will be assigned during build.
        base[0] = 1; // root starts pointing past itself
        Self { base, check, value }
    }

    /// Ensure the arrays are at least `min_len` elements.
    fn grow_to(&mut self, min_len: usize) {
        if self.base.len() >= min_len {
            return;
        }
        let new_len = min_len.next_power_of_two();
        self.base.resize(new_len, 0);
        self.check.resize(new_len, UNDEF);
        self.value.resize(new_len, None);
    }

    /// Insert `key → val`. Panics on duplicate keys.
    fn insert(&mut self, key: &[u8], val: u32) {
        let mut state: usize = 0;

        for &byte in key {
            let b = byte as usize;
            let child = (self.base[state] as usize)
                .checked_add(b)
                .expect("base + byte overflow");

            self.grow_to(child + 1);

            if self.check[child] == UNDEF {
                // Free slot — claim it.
                self.check[child] = state as u32;
                // base[child] starts at 0; will be fixed if it gets children.
                self.base[child] = 0;
                state = child;
            } else if self.check[child] == state as u32 {
                // Already our child — walk down.
                state = child;
            } else {
                // Collision — the slot is owned by another parent.
                // Relocate the current node's existing children.
                self.resolve_collision(state, b, child);
                // After resolution, `base[state] + b` is now ours.
                let new_child = (self.base[state] as usize) + b;
                assert!(
                    new_child < self.check.len(),
                    "grow should have happened in resolve"
                );
                self.check[new_child] = state as u32;
                self.base[new_child] = 0;
                state = new_child;
            }
        }

        assert!(
            self.value[state].is_none(),
            "duplicate key in datrie (slot {state})"
        );
        self.value[state] = Some(val);
    }

    /// Resolve a collision at `child` when trying to insert byte `b` from `parent`.
    ///
    /// Strategy: relocate whichever node has fewer children (the loser). This
    /// is the classic Aoe approach.
    fn resolve_collision(&mut self, parent: usize, byte: usize, child: usize) {
        let occupant = self.check[child] as usize;

        // Collect children of parent and occupant at current base positions.
        let parent_children = self.children_at(parent);
        let occ_children = self.children_at(occupant);

        // Pick the node with fewer children to relocate.
        let (loser, loser_children) = if parent_children.len() <= occ_children.len() {
            (parent, &parent_children)
        } else {
            (occupant, &occ_children)
        };

        // Find a new base for the loser that accommodates all its children
        // AND the new byte (if the loser is `parent`).
        let new_base = self.find_new_base(
            loser,
            loser_children,
            if loser == parent { Some(byte) } else { None },
        );

        // Relocate loser's children to new_base.
        let old_base = self.base[loser] as usize;
        for &c in loser_children {
            let old_idx = old_base + c;
            let new_idx = new_base + c;

            self.grow_to(new_idx + 1);
            // Move slot.
            self.check[new_idx] = self.check[old_idx];
            self.base[new_idx] = self.base[old_idx];
            self.value[new_idx] = self.value[old_idx].take();

            // Patch children of the moved node to point back to new_idx.
            self.reparent_children(new_idx, old_idx);

            // Free old slot.
            self.check[old_idx] = UNDEF;
            self.base[old_idx] = 0;
        }

        self.base[loser] = new_base as i32;
    }

    /// Collect byte values of children of node `s` at its current base.
    fn children_at(&self, s: usize) -> Vec<usize> {
        let b = self.base[s] as usize;
        // A node can have at most 256 children (one per byte value).
        let mut out = Vec::with_capacity(256);
        // Scan all 256 possible byte offsets.
        for byte in 0..256 {
            let child = b + byte;
            if child < self.check.len() && self.check[child] == s as u32 {
                out.push(byte);
            }
        }
        out
    }

    /// Reparent: after moving a node from `old_idx` to `new_idx`, all its
    /// children have `check[child] == old_idx`; update them to `new_idx`.
    fn reparent_children(&mut self, new_idx: usize, old_idx: usize) {
        let b = self.base[new_idx] as usize;
        for byte in 0..256u32 {
            let child = b + byte as usize;
            if child < self.check.len() && self.check[child] == old_idx as u32 {
                self.check[child] = new_idx as u32;
            }
        }
    }

    /// Find a new base for `loser` that can host all `children` bytes
    /// (and optionally `extra_byte`) without collisions.
    //
    // Zero-allocation: iterates over `children` and optionally checks the
    // `extra_byte` separately instead of building a merged Vec.
    fn find_new_base(&self, _loser: usize, children: &[usize], extra_byte: Option<usize>) -> usize {
        // Collision predicate at a given candidate base: true if any byte in
        // the required set (children ∪ {extra_byte}) collides with an occupied slot.
        let collides = |candidate: usize, eb: Option<usize>| -> bool {
            // Check `extra_byte` first when present: it's the byte that triggered
            // the collision, so it's the most likely to keep colliding.
            if let Some(b) = eb {
                let idx = candidate + b;
                if idx < self.check.len() && self.check[idx] != UNDEF {
                    return true;
                }
            }
            for &b in children {
                let idx = candidate + b;
                if idx < self.check.len() && self.check[idx] != UNDEF {
                    return true;
                }
            }
            false
        };

        // Try successive offsets until we find a collision-free base.
        let mut candidate = 1usize;
        while collides(candidate, extra_byte) {
            candidate += 1;
        }
        candidate
    }

    /// Look up `key` in the trie. Returns `Some(value)` if found, `None` otherwise.
    /// Zero allocations.
    #[inline]
    fn lookup(&self, key: &[u8]) -> Option<u32> {
        let mut state: usize = 0;
        for &byte in key {
            let child = (self.base[state] as usize).wrapping_add(byte as usize);
            if child >= self.check.len() || self.check[child] != state as u32 {
                return None;
            }
            state = child;
        }
        self.value[state]
    }

    /// Longest-prefix match: walk `input` from `start`, returning
    /// `(value, end_offset)` of the longest matching prefix.
    /// Returns `None` if no prefix matches.
    #[inline]
    fn longest_prefix(&self, input: &[u8], start: usize) -> Option<(u32, usize)> {
        let mut state: usize = 0;
        let mut best: Option<(u32, usize)> = None;

        for (i, &byte) in input.iter().enumerate().skip(start) {
            let child = (self.base[state] as usize).wrapping_add(byte as usize);
            if child >= self.check.len() || self.check[child] != state as u32 {
                break;
            }
            state = child;
            if let Some(v) = self.value[state] {
                best = Some((v, i + 1));
            }
        }
        best
    }
}

// ── T1: DatrieVocab ─────────────────────────────────────────────────────────

/// Double-array trie replacing `HashMap<Vec<u8>, usize>` for token vocab lookup.
///
/// Build once from the tokenizer vocabulary, then use `lookup()` during encode.
/// Zero allocations on the hot path.
#[derive(Clone, Debug)]
pub struct DatrieVocab {
    inner: Datrie,
}

impl DatrieVocab {
    /// Build a `DatrieVocab` from a `HashMap<Vec<u8>, usize>`.
    ///
    /// Keys are token byte sequences, values are token IDs.
    /// The input HashMap is not consumed (it may be needed for decode).
    pub fn build(vocab: &HashMap<Vec<u8>, usize>) -> Self {
        // Pre-allocate ~2× the vocab size (heuristic for trie density).
        let cap = (vocab.len() * 2).max(256);
        let mut trie = Datrie::with_capacity(cap);

        // Sort keys for better locality during insertion (fewer collisions).
        let mut entries: Vec<_> = vocab.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        for (key, id) in &entries {
            trie.insert(key, **id as u32);
        }

        Self { inner: trie }
    }

    /// Look up a token by its byte sequence. Returns the token ID or `None`.
    #[inline]
    pub fn lookup(&self, key: &[u8]) -> Option<usize> {
        self.inner.lookup(key).map(|v| v as usize)
    }

    /// Longest-prefix match from `start` in `input`. Returns `(token_id, end_offset)`.
    #[inline]
    pub fn longest_prefix(&self, input: &[u8], start: usize) -> Option<(usize, usize)> {
        self.inner
            .longest_prefix(input, start)
            .map(|(v, end)| (v as usize, end))
    }

    /// Total bytes used by the internal arrays (base + check + value).
    pub fn inner_bytes(&self) -> usize {
        let n = self.inner.base.len();
        n * 4 + // base: Vec<i32>
        n * 4 + // check: Vec<u32>
        n * std::mem::size_of::<Option<u32>>() // value: Vec<Option<u32>>
    }
}

// ── T2: DatrieTreeIndex ─────────────────────────────────────────────────────

/// Double-array trie replacing `HashMap<Vec<u8>, SplitTree>` for pretoken → tree lookup.
///
/// The trie stores the *index* into the trees Vec rather than the tree itself,
/// so the trees Vec is still used for actual tree data.
#[derive(Clone, Debug)]
pub struct DatrieTreeIndex {
    inner: Datrie,
    /// Trees stored in a dense Vec for cache-friendly access by index.
    trees: Vec<super::toast_types::SplitTree>,
}

impl DatrieTreeIndex {
    /// Build a `DatrieTreeIndex` from a `HashMap<Vec<u8>, SplitTree>`.
    ///
    /// Trees are moved into an indexed Vec; the trie maps pretoken bytes → Vec index.
    pub fn build(trees: HashMap<Vec<u8>, super::toast_types::SplitTree>) -> Self {
        let cap = (trees.len() * 2).max(256);

        // Sort by pretoken bytes for better insertion order (fewer collisions).
        let mut entries: Vec<_> = trees.into_iter().collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut trie = Datrie::with_capacity(cap);
        let mut tree_vec = Vec::with_capacity(entries.len());

        for (key, tree) in entries {
            let idx = tree_vec.len() as u32;
            trie.insert(&key, idx);
            tree_vec.push(tree);
        }

        Self {
            inner: trie,
            trees: tree_vec,
        }
    }

    /// Look up a pretoken's split tree by its byte sequence.
    #[inline]
    pub fn lookup(&self, pretoken: &[u8]) -> Option<&super::toast_types::SplitTree> {
        self.inner
            .lookup(pretoken)
            .and_then(|idx| self.trees.get(idx as usize))
    }

    /// Number of split trees stored.
    pub fn len(&self) -> usize {
        self.trees.len()
    }

    /// Whether there are no trees.
    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vocab(pairs: &[(&[u8], usize)]) -> HashMap<Vec<u8>, usize> {
        pairs.iter().map(|&(k, v)| (k.to_vec(), v)).collect()
    }

    #[test]
    fn datrie_basic_lookup() {
        let vocab = make_vocab(&[
            (b"hello", 0),
            (b"hell", 1),
            (b"he", 2),
            (b"cat", 3),
            (b"cats", 4),
        ]);
        let trie = DatrieVocab::build(&vocab);

        assert_eq!(trie.lookup(b"hello"), Some(0));
        assert_eq!(trie.lookup(b"hell"), Some(1));
        assert_eq!(trie.lookup(b"he"), Some(2));
        assert_eq!(trie.lookup(b"cat"), Some(3));
        assert_eq!(trie.lookup(b"cats"), Some(4));
        assert_eq!(trie.lookup(b"h"), None);
        assert_eq!(trie.lookup(b"hello!"), None);
        assert_eq!(trie.lookup(b"dog"), None);
    }

    #[test]
    fn datrie_longest_prefix() {
        let vocab = make_vocab(&[(b"ab", 0), (b"abcd", 1), (b"abcdef", 2)]);
        let trie = DatrieVocab::build(&vocab);

        // From start=0, "abcdef" matches all three prefixes; longest is "abcdef".
        assert_eq!(trie.longest_prefix(b"abcdef", 0), Some((2, 6)));
        assert_eq!(trie.longest_prefix(b"abcdefg", 0), Some((2, 6)));
        assert_eq!(trie.longest_prefix(b"abcdxz", 0), Some((1, 4)));
        assert_eq!(trie.longest_prefix(b"abzz", 0), Some((0, 2)));
        assert_eq!(trie.longest_prefix(b"xyz", 0), None);
    }

    #[test]
    fn datrie_single_byte_tokens() {
        let vocab: HashMap<Vec<u8>, usize> = (0u8..=255).map(|b| (vec![b], b as usize)).collect();
        let trie = DatrieVocab::build(&vocab);

        for b in 0u8..=255 {
            assert_eq!(trie.lookup(&[b]), Some(b as usize));
        }
        assert_eq!(trie.lookup(b"ab"), None);
    }

    #[test]
    fn datrie_empty_key() {
        let vocab = make_vocab(&[(b"", 42), (b"a", 1)]);
        let trie = DatrieVocab::build(&vocab);
        assert_eq!(trie.lookup(b""), Some(42));
        assert_eq!(trie.lookup(b"a"), Some(1));
    }

    #[test]
    fn datrie_tree_index_basic() {
        use super::super::toast_types::{SplitNode, SplitTree};

        let trees: HashMap<Vec<u8>, SplitTree> = [
            (
                b"hello".to_vec(),
                SplitTree {
                    pretoken: b"hello".to_vec(),
                    nodes: vec![SplitNode {
                        start: 0,
                        end: 5,
                        left: None,
                        right: None,
                    }],
                },
            ),
            (
                b"world".to_vec(),
                SplitTree {
                    pretoken: b"world".to_vec(),
                    nodes: vec![SplitNode {
                        start: 0,
                        end: 5,
                        left: None,
                        right: None,
                    }],
                },
            ),
        ]
        .into_iter()
        .collect();

        let idx = DatrieTreeIndex::build(trees);

        assert!(idx.lookup(b"hello").is_some());
        assert!(idx.lookup(b"world").is_some());
        assert!(idx.lookup(b"hell").is_none());
        assert!(idx.lookup(b"worlds").is_none());
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn datrie_large_vocab() {
        // Simulate a realistic vocab with varied-length tokens.
        let mut vocab = HashMap::new();
        for i in 0..5000u32 {
            let key = format!("token_{i}").into_bytes();
            vocab.insert(key, i as usize);
        }
        // Add single-byte tokens.
        for b in 0u8..255 {
            vocab.insert(vec![b], b as usize + 10000);
        }

        let trie = DatrieVocab::build(&vocab);

        // Spot-check.
        assert_eq!(trie.lookup(b"token_0"), Some(0));
        assert_eq!(trie.lookup(b"token_4999"), Some(4999));
        assert_eq!(trie.lookup(b"token_5000"), None);
        assert_eq!(trie.lookup(&[0u8]), Some(10000));
        assert_eq!(trie.lookup(&[254u8]), Some(10254));
    }
}
