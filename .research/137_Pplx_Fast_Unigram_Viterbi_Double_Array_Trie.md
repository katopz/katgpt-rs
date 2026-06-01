# Research 137: Perplexity Fast Unigram Tokenizer — Double-Array Trie + Zero-Alloc Viterbi

> **Source:** [Improving Unigram Tokenizer CPU Performance](https://research.perplexity.ai/articles/improving-unigram-tokenizer-cpu-performance) — Perplexity AI, 2026
> **Reference Code:** [perplexityai/pplx-garden](https://github.com/perplexityai/pplx-garden) (tokenizer not yet in public repo)
> **Date:** 2026-05-30
> **Related Research:** 081 (ToaST Split Trees), 087 (ConvexTok), 082 (ToaST Tokenization), 110 (Ciot CPU inference)
> **Related Code:** `src/tokenizer/` (ToaST + ConvexTok pipeline)
> **Domain:** katgpt-rs (open, general-purpose inference infrastructure)

---

## TL;DR

Perplexity reimplemented their Unigram tokenizer's Viterbi forward pass from scratch, achieving **5× vs HuggingFace**, **2× vs SentencePiece**, **1.5× vs IREE** via three optimizations: (1) double-array trie replacing HashMap trie, (2) bitmap + inline packing (64B cache-line per node), (3) huge-page backing for the trie. Zero steady-state allocations.

**Verdict: MODERATE GAIN — Data structure optimizations transferable to our ToaST/BPE vocab lookup but Viterbi-specific algorithm is NOT used in our stack. Feature-gate the double-array trie as a vocab lookup acceleration path. Default-OFF until proven on our tokenizer workload.**

---

## Key Optimizations (from article)

| # | Optimization | Mechanism | Measured Gain |
|---|---|---|---|
| Baseline | Zero-alloc HashMap trie | Remove String::from_utf8 per match, store token_id in node | 2.3× vs HF |
| 1 | Double-array trie (Darts) | Two flat arrays (base[], check[]), 1 add + 1 compare per byte step | 2.3× over baseline |
| 2 | Bitmap + inline packing | Per-node bitmap replaces check array; pack 4 fields into 64B cache-line | +4.5-7% over Darts |
| 3 | Huge pages (2MB) | Trie backed by mmap huge pages, eliminates TLB pressure | +3-12% over bitmap |

**Final: 3.66M → 1.04M instructions/encode (3.5×), zero allocs, p50 65µs at 514 tokens.**

### Double-Array Trie Core

```
child_index = base[parent] + byte
valid       = check[child_index] == parent   // ownership check
token_id    = stored in trie node directly    // no side lookup
score       = flat float array[token_id]     // O(1)
```

### Bitmap Replacement (Opt 2)

The `check[]` array is redundant — a per-node bitmap of which bytes have children encodes the same information:
- Bitmap test compiles to single `BT` instruction (~1 cycle)
- 4 fields (bitmap, base, token_id, score) packed into 64 bytes = 1 cache line per node
- Trade-off: 9MB → 50MB trie (space for speed)

---

## Distillation to Our Stack

### What We Have

Our tokenizer pipeline (`src/tokenizer/`):
- **ToaST** — split-tree recursive descent (not Viterbi/Unigram)
- **ConvexTok** — LP vocabulary optimizer
- **vocab_to_id**: `HashMap<Vec<u8>, usize>` — this IS the bottleneck pattern Perplexity optimized
- **Pretoken lookup**: `HashMap<Vec<u8>, SplitTree>` — same HashMap pattern

### What Transfers

| Pattern | Our Usage | Pplx Solution | Transfer |
|---------|-----------|---------------|----------|
| HashMap<Vec<u8>, X> vocab lookup | `ToastTokenizer.vocab_to_id` | Double-array trie | ✅ Direct |
| HashMap<Vec<u8>, SplitTree> tree lookup | `ToastTokenizer.trees` | Double-array trie (value = tree index) | ✅ Direct |
| Per-match allocation | Minimal (we use recursive descent, not Viterbi) | Zero-alloc scratch | ⬜ Marginal |
| Cache-line packing | N/A (HashMap is scatter-allocated) | Inline 64B nodes | ✅ With trie |
| Huge pages | N/A (HashMap) | mmap huge pages | ✅ With trie, Linux only |

### What Does NOT Transfer

| Pattern | Why Not |
|---------|---------|
| Viterbi DP recurrence | We use recursive descent, not Unigram Viterbi |
| Per-match string allocation | We don't allocate per match (tree descent) |
| Score array lookup | ToaST doesn't use log-probability scores |

---

## GOAT Assessment

### GOAT Pillar Alignment (27_mmo_goat_pillars_decision_matrix.md)

| Criterion | Score | Evidence |
|-----------|-------|----------|
| GOAT passed | ⬜ | Not yet proven on our tokenizer workload |
| MMO-product | ⬜ | Indirect — faster tokenization helps all inference, not MMO-specific |
| LoRA-independent | ✅ | Pure data structure, no ML |
| Defensible | ❌ | Double-array trie is public prior art (Aoe, 1989) |
| Secret coverage | ❌ | None — standard CS technique |

**Pillar verdict: NOT a GOAT pillar.** This is infrastructure optimization, not a competitive moat. Double-array tries are well-known in CS literature. No game-specific knowledge required.

### Optimization Alignment (optimization.md)

| Optimization.md Principle | Pplx Approach | Alignment |
|---------------------------|---------------|-----------|
| "Pre-compute lookup tables once" | Trie built once at load, O(1) reads | ✅ Perfect |
| "Track per-slot aggregates during insert" | Token ID + score stored in node | ✅ Perfect |
| "Cache allocations: Vec::with_capacity() once, clear() + reuse" | Caller-owned scratch, zero alloc hot path | ✅ Perfect |
| "Keep inner loops branch-free" | Bitmap test replaces branch | ✅ Perfect |
| "Don't allocate inside hot loops" | Zero allocations on encode | ✅ Perfect |
| "Don't linear scan for hot-path queries" | HashMap → O(1) trie lookup | ✅ Perfect |
| "Don't recompute unchanged values" | Scores from flat array by token_id | ✅ Perfect |
| "Don't GPU for microsecond workloads" | Entirely CPU-optimized | ✅ Perfect |

**Optimization alignment: 8/8 — textbook application of our optimization principles.**

---

## Implementation Plan

### Feature Gate

```toml
# katgpt-rs/Cargo.toml
datrie_vocab = []  # Double-array trie vocab lookup for ToaST tokenizer (Research 137)
```

**Default: OFF** — requires benchmark proof that it helps our ToaST/BPE workload. The HashMap overhead is marginal for small vocabs (<50K). Only matters for large-vocab (250K+) tokenization workloads.

### Tasks

- [x] T1: Implement `DatrieVocab` struct — double-array trie replacing `HashMap<Vec<u8>, usize>`
  - Build-time: insert all tokens from ToastTokenizer vocab
  - Runtime: byte-by-byte walk, return token_id or None
  - Single `base: Vec<i32>` + `check: Vec<u32>` + `value: Vec<Option<u32>>` (token_id)
  - Zero allocations on lookup
  - Target: `src/tokenizer/datrie.rs` behind `#[cfg(feature = "datrie_vocab")]`

- [x] T2: Implement `DatrieTreeIndex` — double-array trie for pretoken→tree-index lookup
  - Same structure, value = index into `trees` Vec
  - Eliminates `HashMap<Vec<u8>, SplitTree>` lookup during encode

- [x] T3: Benchmark — compare HashMap vs DatrieVocab on our ToaST workload
  - Vocab sizes: 512, 4K, 32K, 128K
  - Input lengths: 128, 512, 1K, 4K bytes
  - Measure: p50/p99 latency (batched 1K lookups), encode throughput, build time, memory
  - **Result: DA wins 1.2–3.0× on vocab lookup, massive win on longest-prefix encode. HM wins 1.3× on tree index lookup (shorter keys). Build time 30–585× slower (one-time cost). See `examples/datrie_01_bench.rs`.**

- [ ] T4: (Optional) Bitmap + inline packing if T3 shows gain
  - Pack (bitmap: u64, base: i32, token_id: u32, _pad: [u8; 48]) into 64B
  - Drop check array from runtime

- [ ] T5: (Optional) Huge-page backing for Linux targets
  - `#[cfg(target_os = "linux")]` mmap with MAP_HUGETLB
  - Only if trie > 10MB (large vocabs)

### T3 Benchmark Results (2026-06-01, release build, Apple M-series)

| Bench | Vocab | HM p50 | DA p50 | Speedup | Verdict |
|-------|-------|--------|--------|---------|--------|
| B1: Vocab lookup/1K | 512 | 15,583 ns | 5,166 ns | **3.02×** | ✓ DA |
| B1: Vocab lookup/1K | 4,096 | 11,834 ns | 9,084 ns | **1.30×** | ✓ DA |
| B1: Vocab lookup/1K | 32,768 | 11,500 ns | 9,417 ns | **1.22×** | ✓ DA |
| B1: Vocab lookup/1K | 131,072 | 11,792 ns | 9,500 ns | **1.24×** | ✓ DA |
| B2: Encode 128B | 32K | 44 µs | <1 µs | **44×** | ✓ DA (longest_prefix) |
| B2: Encode 4KB | 32K | 46,863 µs | 12 µs | **3,905×** | ✓ DA |
| B3: Build 32K | 32K | 655 µs | 152,139 µs | 232× slower | HM (one-time) |
| B4: Tree lookup/1K | 512 | 14,333 ns | 18,291 ns | **0.78×** | ✗ HM wins |
| B4: Tree lookup/1K | 32,768 | 14,416 ns | 18,375 ns | **0.78×** | ✗ HM wins |

**Conclusion:** DatrieVocab is a clear win for vocab lookup (1.2–3×) and encode throughput (massive, due to longest_prefix). DatrieTreeIndex is NOT a win for pretoken→tree lookup (shorter keys, HashMap already fast). Keep `datrie_vocab` default-OFF, opt-in for large-vocab encode-heavy workloads.

---

## Honest Assessment

### Why This Might Not Help Us

1. **We don't use Unigram/Viterbi.** The 5× gain is largely from eliminating the Viterbi DP's per-match allocations. Our recursive descent already avoids most of this.

2. **Our vocab sizes are small.** Perplexity targets 250K XLM-RoBERTa vocab. Our ToaST pipeline typically uses 4K-32K. HashMap overhead scales with vocab size; at 32K the HashMap is likely already in L2.

3. **Our hot path is game inference, not tokenization.** The article targets reranker latency (hundreds of documents tokenized per request). Our bottleneck is MCTS/game simulation, not text tokenization.

4. **Binary bloat risk.** Per optimization.md: "Feature-gated code in the same crate affects code layout and branch prediction." A trie implementation adds ~500-1000 lines of code that could affect icache for unrelated hot paths.

### Why It Might Help Us

1. **Zero-alloc vocab lookup** is universally good — matches our optimization.md principles exactly.
2. **Future-proofing** — if we add Unigram tokenization or larger vocabs, the infrastructure is ready.
3. **Educational** — the article is a masterclass in CPU optimization profiling methodology that applies to all our hot paths.

---

## References

- Perplexity Research Blog: [Improving Unigram Tokenizer CPU Performance](https://research.perplexity.ai/articles/improving-unigram-tokenizer-cpu-performance)
- Aoe, J. (1989). [An efficient digital search algorithm by using a double-array structure](https://ieeexplore.ieee.org/document/31365/). IEEE TSE.
- Kudo, T. (2018). [Subword Regularization](https://arxiv.org/abs/1804.10959). Unigram tokenization original paper.
- [Darts: Double-Array Trie](https://linux.thai.net/~thep/datrie/datrie.html) — Karoonboonyanan reference implementation.
- HuggingFace [tokenizers](https://github.com/huggingface/tokenizers) crate (reference baseline).
- SentencePiece (C++) [double-array trie](https://github.com/google/sentencepiece).
