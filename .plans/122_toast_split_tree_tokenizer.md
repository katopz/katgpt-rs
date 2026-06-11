# Plan 120: ToaST — Split Tree Tokenizer

**Research:** `.research/081_ToaST_Tokenization_Split_Trees.md`
**Paper:** [Tokenization with Split Trees](https://arxiv.org/abs/2605.22705) (Schmidt et al., 2026)
**Status:** ✅ Complete (T1–T6 all done)

---

## Tasks

- [x] T1: Split tree types (`tokenizer/toast_types.rs`)
- [x] T2: Split tree construction from byte n-gram counts (`tokenizer/toast_builder.rs`)
- [x] T3: Recursive descent inference (`tokenizer/toast_inference.rs`)
- [x] T4: Feature gate `toast_tokenizer` + module glue
- [x] T5: GOAT proof — 17/17 tests pass (encode/decode roundtrip, serde, compression, no UNK)
- [x] T6: Rényi efficiency metric + benchmark (`tests/bench_120_renyi_efficiency.rs`) — Rényi H_α at α=2.5, efficiency comparison vs BPE, min token count ≥ 100 verified

---

## Context

ToaST decouples tree construction from vocabulary selection:
1. **Build split trees** — greedy binary trees from byte n-gram counts (vocabulary-independent)
2. **Inference** — recursive descent, emit first in-vocabulary node
3. **Training** — LP/IP vocabulary optimization (model-based, uses `good_lp`/HiGHS we already have)

Result: **11%+ compression gain** over BPE at vocab ≥ 40,960, 14–19× fewer single-byte fallbacks.

This plan covers the **modelless** inference path (T1–T4) and GOAT proof (T5–T6).
Training/LP optimization deferred to riir-ai plan (model-based, requires corpus pipeline).

---

## T1: Split Tree Types

File: `katgpt-rs/src/tokenizer/toast_types.rs`

```rust
/// A node in a split tree.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SplitNode {
    /// Byte span [start, end) within the original pretoken.
    pub start: u16,
    pub end: u16,
    /// Index of left child in nodes vec, or None for leaf (single byte).
    pub left: Option<u32>,
    /// Index of right child in nodes vec, or None for leaf (single byte).
    pub right: Option<u32>,
}

/// A full binary split tree for a single pretoken.
/// Nodes stored in array form; root is index 0.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SplitTree {
    /// The original pretoken bytes.
    pub pretoken: Vec<u8>,
    /// All nodes in the tree (preorder). Root = index 0.
    pub nodes: Vec<SplitNode>,
}

/// ToaST tokenizer: vocabulary + pre-built split trees for pretokens.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToastTokenizer {
    /// Token bytes → ID mapping.
    #[serde(with = "super::types::map_serde_bytes")]
    pub vocab_to_id: HashMap<Vec<u8>, usize>,
    /// ID → token bytes mapping.
    pub id_to_vocab: Vec<Vec<u8>>,
    /// Pretoken bytes → SplitTree (pre-built from n-gram counts).
    pub trees: HashMap<Vec<u8>, SplitTree>,
    /// BOS token ID.
    pub bos_id: usize,
    /// EOS token ID.
    pub eos_id: usize,
    /// PAD token ID.
    pub pad_id: usize,
}
```

Key decisions:
- `u16` offsets sufficient (pretoken max length = 64 bytes with length-limited regex)
- `u32` child indices (max nodes per tree = 2*bytes - 1)
- `HashMap<Vec<u8>, SplitTree>` for pretoken → tree lookup
- Separate `map_serde_bytes` for `HashMap<Vec<u8>, usize>` serialization

---

## T2: Split Tree Construction

File: `katgpt-rs/src/tokenizer/toast_builder.rs`

Algorithm (from paper Section 2):
```
best_split(s):
  best_score = -1
  best_i = -1
  for i in 1..len(s):
    left_score = counts.get(s[..i], -10)
    right_score = counts.get(s[i..], -10)
    score = min(left_score, right_score)
    if score > best_score:
      best_score = score
      best_i = i
  if best_i == -1:  // no split found in counts
    best_i = most_known(s)  // longest prefix in counts
  return best_i
```

Implementation:
- Input: `ngram_counts: HashMap<Vec<u8>, u64>` (precomputed byte n-gram frequencies)
- Output: `SplitTree` for each pretoken
- Recursive construction, collect nodes in preorder
- Fallback: if no split pair in counts, split at longest known prefix

```rust
pub struct SplitTreeBuilder<'a> {
    counts: &'a HashMap<Vec<u8>, u64>,
    min_count: u64,
}

impl<'a> SplitTreeBuilder<'a> {
    pub fn new(counts: &'a HashMap<Vec<u8>, u64>, min_count: u64) -> Self { ... }
    
    /// Build a split tree for the given pretoken bytes.
    pub fn build(&self, pretoken: &[u8]) -> SplitTree { ... }
    
    fn best_split(&self, s: &[u8]) -> usize { ... }
    
    fn most_known(&self, s: &[u8]) -> usize { ... }
    
    fn build_recursive(&self, s: &[u8], offset: u16, nodes: &mut Vec<SplitNode>) { ... }
}
```

---

## T3: Recursive Descent Inference

File: `katgpt-rs/src/tokenizer/toast_inference.rs`

```rust
pub struct ToastTokenizerImpl;

impl ToastTokenizerImpl {
    /// Encode a string into token IDs using ToaST recursive descent.
    pub fn encode(tokenizer: &ToastTokenizer, text: &str) -> Vec<usize> {
        // 1. Pre-tokenize text using regex (length-limited GPT-4o variant)
        // 2. For each pretoken:
        //    a. Look up SplitTree in tokenizer.trees
        //    b. If full pretoken in vocab → emit single token
        //    c. Else recursive_descent(tree, node=0)
        // 3. Return token ID sequence
    }
    
    fn recursive_descent(
        tree: &SplitTree,
        node_idx: u32,
        vocab: &HashMap<Vec<u8>, usize>,
        tokens: &mut Vec<usize>,
        unk_id: usize,
    ) {
        let node = &tree.nodes[node_idx as usize];
        let span = &tree.pretoken[node.start as usize..node.end as usize];
        
        if let Some(&id) = vocab.get(span) {
            tokens.push(id);
            return;
        }
        
        match (node.left, node.right) {
            (Some(l), Some(r)) => {
                Self::recursive_descent(tree, l, vocab, tokens, unk_id);
                Self::recursive_descent(tree, r, vocab, tokens, unk_id);
            }
            _ => {
                // Leaf node (single byte) — must be in vocab by construction
                let id = vocab.get(span).copied().unwrap_or(unk_id);
                tokens.push(id);
            }
        }
    }
    
    /// Decode token IDs back to string.
    pub fn decode(tokenizer: &ToastTokenizer, ids: &[usize]) -> String { ... }
}
```

Advantages over BPE `encode`:
- **No merge rule iteration** — just tree lookup + recursion
- **Early termination** — if pretoken is in vocab, single emit (no recursion)
- **Deterministic** — tree structure is fixed, no priority queue needed

---

## T4: Feature Gate + Module Glue

### `katgpt-rs/Cargo.toml`
```toml
[features]
toast_tokenizer = []  # ToaST split-tree tokenization — inference engine (Plan 120)
```

### `katgpt-rs/src/tokenizer/mod.rs` update
```rust
mod bpe;
mod types;

#[cfg(feature = "toast_tokenizer")]
mod toast_types;
#[cfg(feature = "toast_tokenizer")]
mod toast_builder;
#[cfg(feature = "toast_tokenizer")]
mod toast_inference;

pub use bpe::{BpeTokenizerImpl, BpeTrainer};
pub use types::{BpeTokenizer, MergeRule};

#[cfg(feature = "toast_tokenizer")]
pub use toast_types::{SplitTree, SplitNode, ToastTokenizer};
#[cfg(feature = "toast_tokenizer")]
pub use toast_builder::SplitTreeBuilder;
#[cfg(feature = "toast_tokenizer")]
pub use toast_inference::ToastTokenizerImpl;
```

### Serialization format
- `ToastTokenizer` uses `serde` (JSON/binary) for persistence
- `trees` field is the bulk of storage — compact binary format preferred
- Interop: can convert BPE vocab → ToaST vocab (just the byte→id mapping, discard merges)

---

## T5: GOAT Proof — Compression vs BPE

Benchmark file: `katgpt-rs/tests/bench_120_toast_compression.rs`

### Methodology
1. Build BPE tokenizer from sample vocab (use existing `BpeTrainer`)
2. Build ToaST split trees from same corpus n-gram counts
3. Use same vocabulary tokens (BPE vocab → ToaST vocab set)
4. Tokenize identical text corpus with both
5. Measure: tokens/byte, unique tokens used, single-byte token ratio

### GOAT Criteria (must pass all)
- [x] **G1:** ToaST produces ≤ BPE token count on identical text + vocab — `proof_t5_toast_fewer_or_equal_tokens_than_bpe`
- [x] **G2:** ToaST single-byte fallback tokens ≤ BPE single-byte tokens
- [x] **G3:** Inference latency per token ≤ 2× BPE (tree lookup vs merge search)
- [x] **G4:** No unknown tokens produced (all 243 valid UTF-8 bytes covered) — `proof_t5_toast_no_unknown_tokens`
- [x] **G5:** Encode→decode roundtrip is identity on ASCII input — `proof_t5_roundtrip_identity`

### Test corpus
- Synthetic: repeated common English words, code snippets, game state JSON
- Include edge cases: single characters, long words (>32 chars), multi-byte UTF-8

---

## T6: Rényi Efficiency Metric

Benchmark file: `katgpt-rs/tests/bench_120_renyi_efficiency.rs`

From Zouhar et al. (2023), Rényi efficiency with α=2.5:
```
H_α = (1/(1-α)) * log2(Σ_i (p_i)^α)
Rényi efficiency = H_α / log2(|V|)
```

Where `p_i` = frequency of token i in tokenized output.

### GOAT Criteria
- [x] ToaST Rényi efficiency ≥ BPE Rényi efficiency on same corpus + vocab size (within 15% tolerance)
- [x] ToaST min token count on validation data ≥ 100 (paper reports 103 vs BPE's 1)

---

## Implementation Order

```
T1 (types) → T2 (builder) → T3 (inference) → T4 (feature gate) → T5 (GOAT) → T6 (Rényi)
```

T1–T3 are independent from existing code (new files only).
T4 touches `mod.rs` and `Cargo.toml` (minimal changes).
T5–T6 are test-only files in `tests/`.

---

## Future (riir-ai, model-based)

- LP vocabulary optimization via `good_lp`/HiGHS (already available)
- N-gram counting pipeline from corpus
- Multilingual extension (per-language cost reweighting)
- Morpheme-aligned split preference hierarchy

These require corpus pipeline infrastructure and belong in `riir-ai/.plans/`.

---

## References

- Schmidt et al. (2026). Tokenization with Split Trees. arXiv:2605.22705
- Zouhar et al. (2023). Tokenization and the noiseless channel. ACL 2023.
- HiGHS LP solver: https://highs.dev/