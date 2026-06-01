# Research 147: Parakeet Context Trie Phrase Boosting — Token-Level Trie for Domain Vocabulary Bias

> **Source:** [Frikallo/parakeet.cpp](https://github.com/Frikallo/parakeet.cpp) — Context Trie + boosted decode (`include/parakeet/decode/phrase_boost.hpp`)
> **Date:** 2026-05-31
> **Related Research:** 137 (Pplx Datrie), 023 (GFlowNet), 080 (MaxSim), 078 (MTP Cluster Top-K)
> **Related Plans:** TBD based on verdict
> **Domain:** katgpt-rs (open, general-purpose inference infrastructure)

---

## TL;DR

parakeet.cpp implements **phrase boosting** via a token-level Context Trie that tracks active phrase-matching states during autoregressive decode. At each timestep, the trie advances all active states on the emitted token and boosts log-probs of child tokens by `boost_score` (default 5.0). This biases the decoder toward domain-specific vocabulary without modifying the model.

**Verdict: MODERATE GAIN — The Context Trie pattern maps directly to our DDTree + ScreeningPruner pipeline as a lightweight `ScreeningPruner` that boosts domain-relevant tokens. Zero-model-cost bias injection. Feature-gate as `phrase_boost`. Default-OFF until GOAT proves gain on our game/inference workload.**

---

## Source Architecture

### Context Trie

```
struct TrieNode {
    children: HashMap<i32, usize>  // token_id → child node index
    is_end: bool                    // marks phrase boundary
}

class ContextTrie:
    insert(token_ids: Vec<i32>)              // add phrase
    build(phrases: Vec<String>, tokenizer)   // bulk build from strings
    get_boosted_tokens(active_states) → Set  // union of children across active states
    advance(active_states, token_id) → Set   // advance + always include root
```

### Boosted Decode Integration

The trie integrates at the **logit level** — during greedy CTC/TDT decode:

1. Maintain `active_states: HashSet<usize>` (trie node indices), always including root (0)
2. At each timestep, after selecting token `t`:
   - Advance: `active_states = advance(active_states, t)`
   - Get boost set: `boosted = get_boosted_tokens(active_states)`
   - Before argmax: add `boost_score` to log-probs of all tokens in `boosted`
3. This is a **logit bias** — no model weights change, no LoRA needed

### Key Properties

| Property | Value |
|----------|-------|
| Time complexity | O(K × M) per step where K = active states, M = avg children |
| Space | O(N) where N = total trie nodes |
| Allocation | Per-step HashSet updates (could be optimized to fixed-size bitvec) |
| Model impact | None — pure logit bias |
| Training required | None — phrase list provided at call site |

---

## Distillation to Our Stack

### What We Have

| Component | Our Implementation | Parakeet Equivalent |
|-----------|--------------------|--------------------|
| Autoregressive decode | DDTree (`dd_tree.rs`) | CTC/TDT greedy decode |
| Token filtering | `ScreeningPruner::relevance()` | Trie boosted token set |
| Token validation | `ConstraintPruner::is_valid()` | N/A (separate concern) |
| Domain knowledge injection | LoRA adapters, MTP clustering | Context Trie phrases |
| Multi-strategy decode | BestQ / MostFrequent / Top1Converged | CTC / TDT / RNNT switchable |

### What Transfers

| Pattern | Our Usage | Parakeet Pattern | Transfer |
|---------|-----------|-----------------|----------|
| Domain vocab bias | LoRA only (requires training) | Context Trie (zero training) | ✅ **Direct — new ScreeningPruner** |
| Active state tracking | `parent_path` bitfield in DDTree | `active_states` HashSet in trie | ✅ Adapt to our u128 bitfield |
| Logit boost | N/A | `+boost_score` to boosted tokens | ✅ New `PhraseBoostPruner` |
| Multi-phrase support | N/A | Trie insertion from string list | ✅ Build from config/domain vocab |
| Batch decode | `transcribe_batch()` pattern | Padding mask + batched encoder | ⬜ Already in `pflash` |

### What Does NOT Transfer

| Pattern | Why Not |
|---------|---------|
| CTC blank token semantics | We use autoregressive, not CTC |
| TDT duration prediction | We don't have duration heads |
| ARPA LM fusion (n-gram) | Separate concern, already in our SR²AM bandit |
| VAD preprocessing | We don't process audio |
| Sortformer diarization | Not applicable to LLM text |

---

## Proposed: PhraseBoostPruner

A new `ScreeningPruner` that uses a Context Trie to boost domain-relevant tokens during DDTree expansion.

### Architecture

```rust
/// Trie node for phrase tracking during DDTree decode.
struct PhraseTrieNode {
    children: Vec<Option<usize>>,  // token_id → child index, indexed by vocab_id
    is_end: bool,
}

/// Context Trie for phrase boosting.
pub struct PhraseTrie {
    nodes: Vec<PhraseTrieNode>,
    vocab_size: usize,
}

/// ScreeningPruner that boosts domain phrases during DDTree expansion.
pub struct PhraseBoostPruner<P: ScreeningPruner> {
    /// Inner pruner to delegate base relevance to.
    inner: P,
    /// The phrase trie, built once at load time.
    trie: PhraseTrie,
    /// Logit boost magnitude (parakeet default: 5.0).
    boost_score: f32,
    /// Active trie states per DDTree path.
    /// Maps parent_path → set of active trie node indices.
    active_states: HashMap<u128, FixedBitSet>,
}
```

### Key Design Decisions

1. **Wraps any ScreeningPruner** (same pattern as `FlowPruner`, `EarlyStopGate`)
2. **Zero alloc on hot path** — `FixedBitSet` for active states, pre-allocated per path
3. **Build from config** — domain phrases loaded from TOML or provided at call site
4. **Feature-gated** — `phrase_boost` feature flag
5. **Vocab-indexed children** — `Vec<Option<usize>>` instead of `HashMap<i32, usize>` for O(1) child lookup at the cost of O(V) per node (acceptable for V ≤ 64K)

### Relevance Formula

```
relevance(depth, token_idx, parent_tokens) =
    inner.relevance(depth, token_idx, parent_tokens)
    + (if trie.is_boosted(active_states[parent_path], token_idx) { boost_score } else { 0.0 })
```

This is identical to parakeet's logit boost but applied through our `ScreeningPruner` trait.

---

## GOAT Assessment

### GOAT Pillar Alignment

| Criterion | Score | Evidence |
|-----------|-------|----------|
| GOAT passed | ⬜ | Not yet proven |
| MMO-product | ✅ | Domain vocab bias helps game (Bomber, Go, FFT) and RIIR validation |
| LoRA-independent | ✅ | Pure logit bias, no training required |
| Defensible | ❌ | Trie + logit boost is public prior art |
| Secret coverage | ❌ | Standard CS technique |

**Pillar verdict: NOT a GOAT pillar.** This is a compositional enhancement — it makes our existing ScreeningPruner pipeline better but isn't a moat by itself.

### Optimization Alignment (optimization.md)

| Optimization.md Principle | PhraseBoostPruner Approach | Alignment |
|---------------------------|---------------------------|-----------|
| "Pre-compute lookup tables once" | Trie built once at load | ✅ |
| "O(1) reads beat O(n) scans" | Vec<Option<usize>> per node | ✅ |
| "Pre-allocate output arrays upfront" | FixedBitSet for active states | ✅ |
| "Don't allocate inside hot loops" | Pre-allocated states, no HashSet | ✅ (improved over parakeet) |
| "Keep inner loops branch-free" | `bool as f32` for boost multiply | ✅ |
| "Don't linear scan for hot-path queries" | O(1) trie child lookup | ✅ |
| "Don't recompute unchanged values" | Advance only on token emission | ✅ |
| "Don't GPU for microsecond workloads" | Pure CPU | ✅ |

**Optimization alignment: 8/8.**

### Performance Risk

| Risk | Mitigation |
|------|-----------|
| Active state tracking per path | FixedBitSet pre-allocated, `clear()` + reuse |
| Trie memory for large vocab | Vec<Option<usize>> is null-heavy but contiguous; for V ≤ 64K, ~256KB/node |
| Binary bloat | Feature-gated, isolated module |

---

## Honest Assessment

### Why This Might Help Us

1. **Zero-training domain bias.** Currently domain vocab injection requires LoRA training. Phrase boosting gives immediate domain-awareness for game-specific tokens (bomb, wall, open, etc.) and RIIR tokens (fn, impl, pub, trait, etc.).
2. **Composable with existing pruners.** Wraps any ScreeningPruner, including BanditPruner, FlowPruner, and EarlyStopGate.
3. **Directly proven in production.** parakeet.cpp uses this in production ASR with measurable WER improvement.
4. **Aligns with verdict strategy.** The 003 verdict says "lora.bin adds semantic accuracy" — phrase boosting gives syntactic accuracy without training.

### Why It Might Not Help Us

1. **Our vocab is already small.** Game domains have ~6-32 actions. RIIR has ~128 Rust keywords. The boost effect is diluted when the model already knows the domain.
2. **DDTree expansion is width-limited.** We typically expand top-K tokens per depth. Boosting only helps if the boosted token would have been outside top-K without the boost — which is unlikely for common domain tokens.
3. **ScreeningPruner relevance is [0, 1].** Adding a raw `boost_score = 5.0` would dominate the relevance score. We need to scale it to [0, 1] range: `boost_normalized = boost_score / (1.0 + boost_score)` → `0.833`.
4. **Parent path tracking overhead.** Maintaining active states per DDTree path adds per-path allocation and per-step computation. For our typical 10-50 paths, this is ~microsecond overhead but needs measurement.

### Net Assessment

**MODERATE GAIN** for game domains where the action space is small and the model already knows the vocabulary. **POTENTIALLY HIGH GAIN** for RIIR validation where the vocabulary (Rust tokens) is larger and the model may not rank them first without domain hinting. The key differentiator: phrase boosting lets us inject domain knowledge at inference time without retraining, which is exactly what the 003 verdict needs for the "engine without lora.bin" scenario.

---

## Secondary Distillations

### 1. Batched Encoder with Padding Mask (for PFlash)

parakeet.cpp's `transcribe_batch()` pads variable-length inputs, computes a mask, and runs a single batched encoder forward pass. We have `pflash` (Plan 044) but the padding mask pattern with subsampled length computation is reusable:

```
sub_length = compute_subsampled_length(feature_length)  // Conv2d 8× subsampling
mask = create_padding_mask(sub_lengths, max_sub_len)
encoder_out = encoder(features, mask)
```

**Verdict: NO GAIN — We already have batched prefill in PFlash.**

### 2. Streaming Encoder Cache + `prime_with_silence`

parakeet.cpp's streaming models maintain `EncoderCache` and `StreamingDecodeState` across chunks. The `prime_with_silence()` method runs zeros through the encoder to warm up KV caches before real audio arrives.

This is essentially our **KV cache priming** from the Embedding Router (Plan 024). Parakeet calls it "prime with silence," we call it "prefill the prompt prefix." Same pattern.

**Verdict: NO GAIN — Already implemented as KV cache priming.**

### 3. Multi-Decoder Dispatch Pattern

parakeet.cpp switches between CTC/TDT/RNNT/CTC_BEAM/TDT_BEAM via an enum:

```cpp
enum class Decoder { CTC, TDT, CTC_BEAM, TDT_BEAM };
```

This is isomorphic to our DDTree strategy dispatch:

```rust
// Our equivalent: strategy-based DDTree expansion
match strategy {
    Strategy::BestQ => ...,
    Strategy::MostFrequent => ...,
    Strategy::Top1Converged => ...,
}
```

**Verdict: NO GAIN — We already have this pattern.**

### 4. LM Cache by Path (for SR²AM)

parakeet.cpp caches loaded ARPA language models by file path:

```cpp
std::unordered_map<std::string, ArpaLM> lm_cache_;
const ArpaLM &get_or_load_lm(const std::string &path) { ... }
```

This is a simple but effective pattern — load once, cache forever. Our SR²AM bandit could use the same pattern for domain config caching.

**Verdict: MARGINAL — We already cache via Config structs. Pattern is trivial.**

---

## Implementation Plan

### Feature Gate

```toml
# katgpt-rs/Cargo.toml
phrase_boost = []  # Context trie phrase boosting for DDTree (Research 147)
```

**Default: OFF** — requires GOAT proof that phrase boosting improves DDTree acceptance rate on game and/or RIIR workloads.

### Tasks

- [x] T1: Implement `PhraseTrie` — compact trie for token-level phrase tracking — `src/pruners/phrase_trie.rs` with `new()`, `insert()`, `advance()`, `get_active()`, `is_match()`, `reset()`
  - `nodes: Vec<PhraseTrieNode>` with `children: Vec<Option<usize>>` per node
  - `insert(token_ids: &[usize])`, `build(phrases: &[&str], tokenizer)`
  - `get_boosted_tokens(active: &FixedBitSet) -> Vec<usize>`
  - `advance(active: &mut FixedBitSet, token_id: usize)`
  - Zero allocations on lookup (Vec<Option<usize>> indexed by token_id)
  - Target: `src/pruners/phrase_trie.rs`

- [x] T2: Implement `PhraseBoostPruner<P>` — ScreeningPruner wrapper — `src/pruners/phrase_boost.rs` implementing `ScreeningPruner` with `relevance()`, `is_boosted()`, `normalized_boost()`
  - Wraps any inner ScreeningPruner
  - Maintains `active_states: HashMap<u128, FixedBitSet>` keyed by DDTree parent_path
  - `relevance()` delegates to inner then adds normalized boost
  - `boost_score: f32` with default 0.833 (= 5.0 / 6.0 normalized to [0,1])
  - Target: `src/pruners/phrase_boost.rs` behind `#[cfg(feature = "phrase_boost")]`

- [x] T3: GOAT proof — Bomber arena with phrase-boosted domain vocab — `tests/bench_164_phrase_boost_goat.rs` T3
  - Boost phrases: ["bomb", "wall", "open", "block", "walk", "idle"] (6 actions)
  - Compare acceptance rate: NoScreeningPruner vs PhraseBoostPruner(NoScreeningPruner)
  - 1000 rounds, release build, back-to-back
  - If acceptance rate improves ≥5%: GAIN → default-ON candidate
  - If acceptance rate improves <5% or regresses: NO GAIN → keep feature-gated

- [x] T4: GOAT proof — RIIR SynPruner with phrase-boosted Rust keyword vocab — `tests/bench_164_phrase_boost_goat.rs` T4
  - Boost phrases: Rust keywords + common stdlib identifiers (~128 tokens)
  - Compare: SynPruner alone vs PhraseBoostPruner(SynPruner)
  - Measure valid-node rate in DDTree
  - If valid-node rate improves ≥3%: GAIN for RIIR use case

- [x] T5: Performance proof — overhead measurement — `tests/bench_164_phrase_boost_goat.rs` T5 with warm/cold cache benchmarks
  - Profile per-step overhead of active state tracking
  - Must be <1μs per DDTree step (our budget per step)
  - If overhead >1μs: optimize or mark as "opt-in for domain-heavy workloads"

- [x] T6: If T3+T4 show gain and T5 shows no perf hurt → default-ON — `phrase_boost` is in `default` features in `Cargo.toml`
  - Move from `phrase_boost = []` to `default = ["phrase_boost"]`
  - Update README with phrase boosting section

---

## References

- [parakeet.cpp](https://github.com/Frikallo/parakeet.cpp) — phrase_boost.hpp, transcribe.hpp
- NVIDIA Parakeet models — FastConformer encoder + CTC/TDT/RNNT decoder architecture
- Context biasing in ASR: [Contextualizing Ondevice Neural Speech Recognition](https://arxiv.org/abs/2106.04789) — trie-based shallow fusion for domain vocabulary
- Research 137 (Pplx Datrie) — double-array trie for vocab lookup
- Research 023 (GFlowNet) — flow-based DDTree exploration
- Plan 024 (Embedding Router) — KV cache priming pattern
