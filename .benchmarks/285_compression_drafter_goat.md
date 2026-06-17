# Benchmark 285: CompressionDrafter Quest Grammar — GOAT **FAILED** (2nd run)

**Date:** 2026-06-17
**Plan:** [285_compression_drafter_quest_grammar.md](../.plans/285_compression_drafter_quest_grammar.md)
**Research:** [256_GzipLM_Compression_Drafter.md](../.research/256_GzipLM_Compression_Drafter.md)
**Feature gates:** `compression_drafter` (katgpt-core), `quest_compression_draft` (riir-games)
**Status:** ❌ **GOAT FAILED (2nd run)** — both Phase 3 (candidate-scoring + lz4) and Phase 7 (beam search + MatchLengthScorer) fail. Stays opt-in.

---

## TL;DR

**Even with the real nathan.rs beam search algorithm and a custom fast match-length scorer, CompressionDrafter loses to TernaryDraftModel for quest grammar.** The fundamental issue: quest grammar's S-V-O templates are so short and so few (8 hardcoded strings) that any compression-based approach is both slower and less diverse than picking from the fixed list.

The open primitive (`compression_drafter`, now including beam search + MatchLengthScorer) stays as opt-in code — both are correct, well-tested implementations useful for future consumers. But for this specific use case, **`TernaryDraftModel` wins fair and square**. Honest negative result.

---

## GOAT Gate Results (Phase 7 re-bench)

| Gate | Target | Phase 3 (lz4+cand) | Phase 7 (match+beam) | Verdict |
|------|--------|---------------------|----------------------|---------|
| **G1 Diversity** | ≥3× unique vs ternary | 0.12× (1 unique) | **1.50× (12 unique)** ↑ | ❌ **STILL FAIL** (improved 12×, still under 3×) |
| **G2 Latency** | ≤2× ternary p99 | 407× | **1077×** ↑ (worse!) | ❌ **STILL FAIL** |
| G3 Composition | freeze/thaw roundtrip | PASS | PASS | ✅ PASS |
| G4 Zero-alloc | scratch stable | PASS | PASS | ✅ PASS |

**Run command:**
```bash
cd riir-ai
cargo test --features quest_compression_draft --test bench_285_compression_drafter --release -- --nocapture
```

---

## Why Phase 7 still failed

### G1 root cause: 12 unique vs 8 isn't 3×

Phase 7 improved G1 from 1 unique → 12 unique (12× better). Beam search works — it does produce more diverse outputs than fixed-candidate scoring. But the bar was 3× the ternary baseline of 8 = 24 unique outputs. We got 12.

Why only 12? The 100 contexts (`"quest 0"` to `"quest 99"`) share the `"quest "` prefix. Their numeric suffixes produce different beam trajectories only when the numeric byte shifts scoring. With a 2KB corpus dominated by S-V-O patterns, the numeric context bytes are largely irrelevant — the corpus determines the output, not the seed.

To get 24+ unique, we'd need either:
- More varied seed contexts (different game state bytes, not numbered strings).
- A per-NPC corpus (the riir-ai/.research/137 angle — different corpus → different output).
- Temperature sampling (nathan.rs has this; we used pure argmax).

### G2 root cause: beam search amplifies scorer calls

Phase 7's MatchLengthScorer IS fast — 217ns per `score()` call (computed: 313µs ÷ 1440 calls per generation). That's sub-µs, fitting Hot-tier for a single call. But beam search multiplies it:

```
calls_per_generation = beam_width × horizon × alphabet_size
                    = 4 × 12 × ~30 (quest-grammar alphabet)
                    = 1440
total_latency = 1440 × 217ns = 313µs
```

Ternary matvec + hash = 291ns total. So beam search is 1077× slower PER GENERATION, even though each individual scorer call is fast.

This is fundamental: compression-based generation requires exploring multiple candidates per step. Template selection doesn't — it's one matvec + one hash. **There's no way to make beam search competitive with single-pass template selection on latency for this use case.**

---

## What the two runs taught us

| Insight | Phase 3 | Phase 7 |
|---------|---------|---------|
| Algorithm matters | Fixed-candidate scoring = 1 unique (corpus dominates) | Beam search = 12 unique (growing tail helps) ✓ |
| Scorer speed matters | lz4 = 50µs/call (Warm-tier) | MatchLengthScorer = 217ns/call (Hot-tier per call) ✓ |
| BUT beam search multiplies calls | N/A (single batch) | 1440 calls/generation × 217ns = 313µs (too slow for Hot) ✗ |
| Diversity ceiling | Capped at candidate-set size | Capped by corpus structure — 12 unique is the natural ceiling for this corpus |

**The honest conclusion:** compression-based generation is a Warm-tier technique. For Hot-tier quest grammar (sub-ms budget), template selection is structurally superior. The two approaches aren't competing on the same axis — they're for different tiers.

---

## What ships (unchanged from Phase 3)

1. **`compression_drafter` module in katgpt-core** — now includes:
   - `CompressionDrafter` trait + `Lz4FlexDrafter` (Phase 1)
   - `MatchScorer` trait + `Lz4MatchScorer` + `MatchLengthScorer` (Phase 6)
   - `beam_search()` function (Phase 5)
   - `corpus_alphabet()` helper (Phase 5)
   - 15/15 unit tests pass
2. **`CompressionQuestDrafter` + `CorpusSnapshot`** in riir-games — both candidate-set and beam-search generation paths.
3. **`TernaryDraftModel` remains default-on** for quest grammar — won twice.
4. **riir-ai/.research/137** stays as exploration for the per-NPC plasma angle (would solve G1 via different corpora, not G2).

---

## Action items

- [x] **Phase 5 executed**: beam search algorithm implemented, 4 unit tests pass.
- [x] **Phase 6 executed**: MatchLengthScorer with inverted index, 4 unit tests pass. Per-call latency 217ns (Hot-tier).
- [x] **Phase 7 executed**: re-bench with beam search + MatchLengthScorer. G1 improved 12× but still misses 3× bar. G2 worse due to beam search amplification.
- [x] **Final demotion**: `compression_drafter` and `quest_compression_draft` stay opt-in. Neither promotes.
- [x] **Honest negative result documented** (this file).
- [ ] **Issue tracking**: `.issues/029_compression_beam_search_followups.md` to track the two paths that might still work:
  - **Per-NPC corpus** (different corpora → different outputs, addresses G1).
  - **Warm-tier positioning** (accept ms latency, position as Warm-tier quest generation for offline/NPC-sleep-cycle use, not Hot-tier runtime).

---

## Final verdict

**The user's instinct ("shall we zip and use this way?") was partially right:**
- ✅ The corpus IS a valid wired format (CorpusSnapshot + BLAKE3 works perfectly).
- ✅ Compression CAN generate diverse outputs (beam search gave 12 unique vs 8 templates).
- ❌ But for the quest grammar Hot-tier use case, template selection is structurally faster and nearly as diverse.

**The right home for compression-based generation in our stack is NOT quest grammar Hot-tier.** It's:
1. **Warm-tier offline generation** — quest packs generated during NPC sleep cycles, where ms latency is fine and diversity matters more than speed.
2. **Per-NPC personality** (riir-ai/.research/137) — if a custom plasma-tier SIMD LZ77 can fit µs budget, per-NPC corpora would produce genuine personality divergence. Still unvalidated.

Both are follow-up work, not this plan.

---

## Files touched (Phase 5-7 additions)

| File | Phase 5-7 Change |
|------|------------------|
| `katgpt-rs/crates/katgpt-core/src/compression_drafter.rs` | + `MatchScorer` trait, `Lz4MatchScorer`, `MatchLengthScorer` (inverted index), `beam_search()`, `corpus_alphabet()` + 9 new unit tests (15 total) |
| `riir-ai/crates/riir-games/src/quest_grammar/compression_draft.rs` | + `generate_beam()` method using beam search |
| `riir-ai/crates/riir-games/tests/bench_285_compression_drafter.rs` | G1+G2 switched to beam search; added phase history in doc comment |
| `katgpt-rs/.plans/285_compression_drafter_quest_grammar.md` | + Phase 5/6/7 sections documenting the algorithm + scorer + re-bench |

---

## TL;DR

Phase 7 re-bench with beam search + MatchLengthScorer: **G1 improved 12× (1→12 unique outputs) but still misses 3× bar; G2 worse (407× → 1077×) because beam search amplifies scorer calls (1440/generation).** Per-call scorer latency IS fast (217ns, Hot-tier), but beam search × alphabet × horizon = too many calls. **Final verdict: quest grammar Hot-tier is structurally better served by `TernaryDraftModel` template selection.** Compression-based generation belongs in Warm-tier (offline quest pack generation) or per-NPC plasma (still unvalidated). Open primitive stays opt-in; honest negative result. Two follow-up paths documented for future issues.
