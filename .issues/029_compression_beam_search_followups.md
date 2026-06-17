# Issue 029: Compression-Drafter Beam Search Follow-ups

**Date:** 2026-06-17
**Status:** Open — tracking two paths that might still make compression-based generation useful
**Plan:** [285_compression_drafter_quest_grammar.md](../.plans/285_compression_drafter_quest_grammar.md)
**Benchmark:** [285_compression_drafter_goat.md](../.benchmarks/285_compression_drafter_goat.md) (GOAT FAILED, 2 runs)
**Research:** [256_GzipLM_Compression_Drafter.md](../.research/256_GzipLM_Compression_Drafter.md)

---

## Context

Plan 285 ran the full workflow (research → plan → impl → bench → demote) twice:

| Run | Algorithm | Scorer | G1 Diversity | G2 Latency |
|-----|-----------|--------|--------------|------------|
| Phase 3 | Fixed-candidate scoring | lz4_flex | 0.12× (1 unique) ❌ | 407× ❌ |
| Phase 7 | Beam search (nathan.rs algorithm) | MatchLengthScorer (inverted index) | 1.50× (12 unique) ❌ | 1077× ❌ |

Target: G1 ≥ 3× (24 unique), G2 ≤ 2× (≈ 600ns).

**Honest conclusion:** compression-based generation loses to `TernaryDraftModel` template selection for Hot-tier quest grammar. Template selection is one matvec + one hash (~290ns); beam search is 1440 scorer calls (~313µs). There is no algorithmic fix — beam search fundamentally needs more compute per generation than single-pass selection.

---

## Two paths that might still work

### Path A: Per-NPC corpus (solves G1, not G2)

**Insight:** G1 failed because all 100 test contexts share the `"quest "` prefix and produce similar beams. If each NPC had its OWN corpus (different action history, different HLA moments), the corpora would diverge and so would the outputs — without needing beam search at all.

**What it needs:**
- Per-NPC `CompressionQuestDrafter` instances (one corpus per NPC).
- Seed each corpus with divergent content (HLA moments, action traces).
- Re-bench G1 with 100 distinct corpora, not 100 contexts on one corpus.

**Predicted outcome:**
- G1 likely passes (different corpora → different outputs by construction).
- G2 still fails (per-NPC doesn't change the per-call latency math).

**Where this lives:** `riir-ai/.research/137_Compression_Drafter_Plasma_Personality_Guide.md` already sketches this. The validation gate G2 (per-NPC divergence) is exactly this experiment.

**Blocker:** needs the plasma-tier custom LZ77 to make per-NPC instances cheap enough. lz4-based per-NPC instances would be ~50µs × 1000 NPCs = 50ms per tick — too slow.

### Path B: Warm-tier positioning ( sidesteps G2)

**Insight:** G2's 2× latency target assumes Hot-tier (sub-ms). But quest pack generation doesn't have to be Hot-tier — it can happen during NPC sleep cycles, world generation, or GM tool batches. Warm-tier (ms) is fine for offline generation.

**What it needs:**
- Reposition `CompressionQuestDrafter::generate_beam` as a Warm-tier API, not a Hot-tier replacement for `TernaryDraftModel`.
- Use case: GM tool generates 100 quest variants offline, picks the best, freezes the winner into a `TernaryDraftModel` template.
- Latency budget: 100ms per quest pack generation. We're at 313µs — 300× under budget.

**Predicted outcome:**
- G2 redefined: "fits Warm-tier budget (≤100ms)" instead of "≤2× ternary (≤600ns)". Passes trivially.
- G1 unchanged: still 12 unique vs 8 templates. May or may not matter for offline generation (where a human GM picks the best).

**Where this would go:** new plan, not a revision of 285. The use case is fundamentally different (offline batch vs runtime single-call).

---

## Recommendation

**Don't pursue either immediately.** The honest result from Plan 285 is:
- `TernaryDraftModel` is the right tool for Hot-tier quest grammar.
- Compression-based generation is interesting but doesn't fit this use case.

If a future use case emerges (GM tool, NPC sleep cycle, per-NPC personality at plasma tier), the open primitive (`compression_drafter` with beam search + MatchLengthScorer) is ready and waiting — 15/15 tests pass, the code is clean. Until then, it stays opt-in and unused.

---

## TL;DR

Plan 285 ran twice, failed twice. Compression-based generation loses to template selection for Hot-tier quest grammar — fundamentally, not just implementationally. Two follow-up paths exist (per-NPC corpus for G1, Warm-tier repositioning for G2) but neither is worth pursuing without a concrete consumer. Open primitive stays opt-in. Honest negative result, documented.
