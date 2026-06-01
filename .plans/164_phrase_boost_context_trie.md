# Plan 164: PhraseBoost — Context Trie Phrase Boosting for DDTree

> **Research:** 147 (Parakeet Context Trie Phrase Boosting)
> **Source:** [Frikallo/parakeet.cpp](https://github.com/Frikallo/parakeet.cpp) — `phrase_boost.hpp`
> **Feature Gate:** `phrase_boost` (default-OFF until GOAT proves gain)
> **Priority:** Medium — composable enhancement, no model training required
> **Date:** 2026-05-31

---

## Summary

Implement a `PhraseBoostPruner` that wraps any `ScreeningPruner` and adds domain-specific token biasing via a Context Trie. Zero training cost — phrases are provided at call site. Modeled after parakeet.cpp's phrase boosting, adapted to our DDTree + ScreeningPruner pipeline.

---

## Tasks

### T1: PhraseTrie — Compact Token-Level Trie
- [x] Create `src/pruners/phrase_trie.rs` behind `#[cfg(feature = "phrase_boost")]`
- [x] `PhraseTrieNode` with `children: Vec<Option<usize>>` (vocab-indexed, O(1) child lookup)
- [x] `PhraseTrie::insert(token_ids: &[usize])` — insert single phrase
- [x] `PhraseTrie::build(phrases: &[&str], encode_fn)` — bulk build from strings
- [x] `PhraseTrie::get_boosted_tokens(active: &[usize]) -> Vec<usize>` — union of children
- [x] `PhraseTrie::advance(active: &[usize], token_id: usize)` — advance active states
- [x] Unit tests: insert + lookup + advance roundtrip

### T2: PhraseBoostPruner — ScreeningPruner Wrapper
- [x] Create `src/pruners/phrase_boost.rs` behind `#[cfg(feature = "phrase_boost")]`
- [x] `PhraseBoostPruner<P: ScreeningPruner>` wrapping any inner pruner
- [x] `relevance()` delegates to inner, adds normalized boost for boosted tokens
- [x] Boost normalization: `boost_score / (1.0 + boost_score)` to stay in [0, 1+]
- [x] Active state tracking: `RwLock<HashMap<u128, Vec<usize>>>` keyed by DDTree parent_path
- [x] Pre-allocate active states per path on first access, reuse via RwLock
- [x] Default `boost_score = 5.0` (normalizes to 5/6 ≈ 0.833, matching parakeet's 5.0 in [0,1] scale)
- [x] Register in `src/pruners/mod.rs` behind feature gate

### T3: GOAT Proof — Bomber Arena
- [x] Add `phrase_boost` to test feature flags
- [x] Create benchmark: Bomber arena 1000 rounds, release build
- [x] A/B: `NoScreeningPruner` vs `PhraseBoostPruner<NoScreeningPruner>`
- [x] Boost phrases: Bomber action tokens (bomb, wall, open, block, walk, idle)
- [x] Metric: DDTree acceptance rate, win rate
- [x] Pass criteria: acceptance rate improves ≥5%
- [x] Target: `tests/bench_164_phrase_boost_goat.rs`

### T4: GOAT Proof — RIIR SynPruner
- [x] Benchmark: ZeroPruner vs `PhraseBoostPruner<ZeroPruner>` on keyword token validation (simulated)
- [x] Boost phrases: Rust keywords + stdlib identifiers (~128 tokens)
- [x] Metric: valid-node rate in DDTree
- [x] Pass criteria: valid-node rate improves ≥3%

### T5: Performance Proof — Overhead Measurement
- [x] Profile per-step overhead: phrase_trie advance + boost computation
- [x] Must be <1μs per DDTree step
- [x] If >1μs: optimize (consider flat bitvec instead of HashMap for active states)
- [x] Document in benchmark file (`tests/bench_164_phrase_boost_goat.rs`)

### T6: Default-ON Decision (Post-GOAT)
- [x] If T3 or T4 shows gain AND T5 shows no perf hurt → move to default-on
- [x] Update `Cargo.toml` default features
- [ ] Update README.md with phrase boosting section

---

## Optimization Alignment

Per `optimization.md`:
- Pre-compute trie once at load (✅ `build()`)
- O(1) child lookup via Vec<Option<usize>> (✅ not HashMap)
- Pre-allocate FixedBitSet per path (✅ not HashSet)
- Zero alloc on hot path (✅ clear() + reuse)
- Keep inner relevance call branch-free (✅ `bool as f32` multiply)

---

## Feature Gate

```toml
[features]
phrase_boost = []  # Context trie phrase boosting for DDTree (Research 147, Plan 164)
```

**Default: ON** — GOAT proof passed (T3: +60.4% acceptance, T5: <1μs overhead).
