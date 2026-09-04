# Research 481: Sparse Memory Finetuning — TF-IDF Slot Ranking for Non-Interference Writes

> **Source:** "Continual Learning via Sparse Memory Finetuning" — [arXiv:2510.15103](https://arxiv.org/abs/2510.15103) — Jessy Lin, Luke Zettlemoyer, Gargi Ghosh, Wen-Tau Yih, Aram Markosyan, Vincent-Pierre Berges, Barlas Oğuz (Meta FAIR + UC Berkeley), Oct 2025
> **Date:** 2026-08-14
> **Status:** Active
> **Related Research:** 387 (PKM primitive — the memory-layer substrate this paper builds on), 310 (RIZZ non-interference branches — branch-level cousin), 006 (Raven slots), 455 (Hebbian kernel memory), 249 (DecentMem dual-pool), 141 (C-LoRA continual)
> **Related Plans:** 408 (PKM primitive + episodic store)
> **Cross-ref (riir-neuron-db):** Research 303 (Hebbian fact-storing shard), 305 (plasticity lifecycle); **(riir-train):** Research 022 (SPEFT — gradient-salience sparse FT)
> **Classification:** Public

---

## TL;DR

Meta FAIR finetunes memory-layer models by updating only the top-t memory slots ranked by **TF-IDF** (slot access frequency on the new batch ÷ slot access frequency on a background pretraining corpus). Result: same knowledge acquisition as full FT / LoRA, but **11% forgetting vs 89% (full FT) / 71% (LoRA)** on held-out NQ. The critical ablation: **TF-only ranking (raw activation) learns equally well but forgets significantly more** — the IDF half (downweight slots that background traffic heavily uses) is what prevents interference.

Our shipped `PkmEpisodicStore` (Plan 408 Phase 5 — δ-rule write gate over `ProductKeyMemory`) selects write targets by **top-k retrieval activation = TF-only** — precisely the configuration the paper's ablation shows is the forgetting-prone one. The actionable delta is an IDF-aware write gate: score slots by `retrieval_weight × idf(slot)` against a static background access-count table, then δ-rule write the top-t by score.

**Distilled for katgpt-rs (modelless, inference-time):**

TF-IDF ranking over *slot accesses* (not tokens) is a count-based selection score — no gradients, no training. Composed with the shipped δ-rule write (bit-identical to one GD step at η=1) and the freeze/thaw publish contract, it yields **modelless continual writes: new episodic facts consolidate into slots that general traffic does not rely on**, so accumulation does not erode prior recall.

---

## 1. Paper Core Findings

1. **Memory layers** (Berges et al. 2024, [arXiv:2412.09764](https://arxiv.org/abs/2412.09764)): an FFN replacement — top-k=32 product-key lookup per token into a 1M–100M slot KV pool. Each token activates ~0.03%–0.0002% of memory params. Trainable keys + values.
2. **Sparse memory finetuning**: per batch, aggregate slot-access counts (TF); rank slots by TF-IDF relative to a **static background corpus** (1000 random DCLM batches; the background index counts are computed once and stored in the checkpoint); stop-gradient on everything except the top-t slots (t=500 for fact learning, t=10K for documents). The trainable mask is the straight-through trick: `mem = mem*m + mem.detach() − (mem*m).detach()`.
3. **Results** (1.3B model): same target-task learning as full FT/LoRA; NQ F1 drops 11% vs 89% (full) / 71% (LoRA). Pareto-dominates the learning-vs-forgetting frontier across lr/rank/alpha/t sweeps.
4. **TF-only ablation** (the load-bearing finding for us): ranking the same t slots by raw access count gives comparable learning but **more forgetting**, and the gap vs TF-IDF **widens as t shrinks** (t=50). At our scale (top-k ≤ 64 per write) the IDF effect should be at its largest.
5. **Background corpus ablation**: using the training set itself as background → worse retention; using DCLM (generic pretraining) or the held-out set works comparably. Downweighting *generally-used* slots, not *task-used* ones, is what matters.
6. **Optimizer**: SGD beats AdamW for sparse continual updates (adaptive per-param steps + momentum interact badly with sparsity). Baselines did not see the same SGD benefit.
7. **Core-set analysis**: a fact spreads over ~100–500 slots (intersection across paraphrases + question); the TF-IDF-selected trainable set aligns with core-set indices and with **entity boundaries** — parametric reads/writes cluster on entities.

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent | Where it ships |
|---|---|---|
| memory layer / memory pool | `ProductKeyMemory` (default-on, Plan 408) | `katgpt-core/src/product_key_memory/` |
| top-k slot lookup (k=32, product keys) | `ProductKeyMemory::query` O(√N) factored retrieval | same |
| sparse finetuning of memory values | `PkmEpisodicStore::write` / `write_weighted` δ-rule gate | `product_key_memory/episodic.rs` |
| TF (batch access count) | retrieval weight / top-k activation on the write query | `PkmEpisodicStore` (TF-only today) |
| IDF (background access rarity) | **MISSING** — nothing downweights generally-hot slots in any write path | — |
| background corpus access counts | closest: LFU `access_counts` (SegmentStore, eviction only); BM25 `IDF(t)` (retrieval only) | `katgpt-kv/src/segment_checkpoint/`, `riir-neuron-db/src/bm25.rs` |
| catastrophic forgetting | interference across successive writes; RIZZ branch contamination (Research 310) | branch-level only |
| trainable top-t mask | write-target selection set | TF-only today |
| "core set" of a fact | Hebbian fact margins (Research 455/303) | `katgpt-core/src/hebbian_kernel_memory.rs` |
| consolidation / replay | Raven/δ-Mem sleep consolidation (F5 fusion, Research 387) | `katgpt-rs/src/sleep/`, riir-neuron-db |

### 2.2 §3.5 Path 0 — training-target decomposition

| Paper component | Modelless analog | Status |
|---|---|---|
| TF-IDF slot ranking | count-based score — same math as shipped BM25 `IDF(t)` (`riir-neuron-db/src/bm25.rs`), applied to slot selection instead of retrieval | ✅ exists, wrong application site |
| SGD update of top-t value rows | δ-rule write `V[idx] += gate·(target − V[idx])` — bit-identical to one GD step η=1 (Plan 408 §Phase 5) | ✅ shipped |
| multi-step training (10K steps, large corpora) | **not needed** — our regime is the paper's *small-data/immediate-update* regime (one-shot writes per surprising event; paper's Fig 3 shows the small-data regime is where sparse memory FT dominates) | ✅ N/A |
| iterative refinement | Sleep-consolidation N-pass δ-rule (bounded passes, not GD) | ✅ shipped |
| learning-vs-forgetting evaluation | retention gate on our interference bench (new GOAT gate axis, not a mechanism) | 🆕 gate only |

**Path 0 verdict: MODELLESS-VALIDABLE.** All mechanism components have shipped analogs; no riir-train deferral. The paper's SGD-on-values loop maps onto our δ-rule write; the only genuinely new code is the IDF term in the write-target ranking.

### 2.3 The gap (what does NOT ship)

Every write path we have selects targets by absolute activation (TF-only):

- `PkmEpisodicStore::write` — top-k(q), uniform gate.
- `PkmEpisodicStore::write_weighted` — top-k(q), softmax-weight-scaled.
- δ-Mem `write_segment` / Raven consolidation / Engram `sigmoid_fuse_into` — no background-usage term.
- Hebbian `construct` — margin-aware but interference-blind to *background traffic* on slots.

The paper's ablation says this is exactly the wrong half to omit: **IDF is the forgetting-prevention half, and its benefit grows as the write set shrinks** — our per-event writes are the smallest write sets we have.

### 2.4 The distilled primitive — IDF-aware write gate

```rust
/// Static, precomputed once from a consumer-supplied background query corpus
/// (mirrors the paper's static background indices stored in the checkpoint).
pub struct BackgroundAccessStats<const N: usize> {
    n_batches: u32,
    /// Number of background batches in which slot i was retrieved (doc-frequency).
    slot_batch_counts: [u32; N],
}

#[inline]
fn idf<N: usize>(bg: &BackgroundAccessStats<N>, slot: usize) -> f32 {
    ((bg.n_batches as f32 + 1.0) / (1.0 + bg.slot_batch_counts[slot] as f32)).ln()
}

// Selection: score(idx) = retrieval_weight[idx] * idf(idx); take top-t by score.
// Update:    unchanged δ-rule  V[idx] += gate * (target - V[idx]).
```

Properties: u32 counts (raw audit scalars — may ride telemetry/commitment); zero-alloc (fixed arrays, fold into existing `PkmScratch`); selection local to the write path (latent side — only the existing BLAKE3 table commitment crosses the sync boundary, unchanged); Sigmoid not involved (this is a ranking score, not a probability).

**Batch-aggregation note.** The paper aggregates TF over a *batch*; our per-event write has TF = retrieval weight of a single query. The exact paper translation (aggregate counts over many events, then rank, then write) applies to the **consolidation sleep-cycle** — the natural batch boundary we already have. Per-event IDF write and consolidation-pass IDF write are the same primitive at two granularities.

### 2.5 Fusion

| Fusion | Existing system | What TF-IDF ranking adds | Gate question |
|---|---|---|---|
| **F1 (primary): IDF write gate × `PkmEpisodicStore`** | δ-rule write + freeze/thaw publish (Plan 408 P5) | non-interference: successive episodic writes stop eroding prior recall | After writing fact set B, recall of fact set A ≥ X% higher with IDF ranking than TF-only, at matched learning of B (paper: gap widens at small t — our regime) |
| **F2: × RIZZ non-interference branches (R310)** | branch-level orthogonality (coarse) | slot-level IDF (fine) → hierarchical non-interference | Does branch+slot double-gating beat either alone on a multi-task-family interference bench? |
| **F3: × Hebbian shard construction (riir-neuron-db R303)** | `construct_hebbian` fact→slot assignment, capacity-aware freeze envelope | prefer low-background-usage slots for new facts; consolidation sleep-cycle ranks slots before writing | Does IDF-ranked consolidation retain ≥80% recall after 5 domain shifts (Research 387 F5's gate)? |
| **F4: × SPEFT (riir-train R022)** | gradient-magnitude top-ρ% masks for sparse FT | background-relative salience: subtract background gradient/access mass so generally-important params are not selected → less forgetting | On a continual FT run, does background-relative mask selection improve held-out retention at matched target learning? (training-track follow-up, secondary) |

F1 is the direct fix on a shipped primitive; F3 is the private continuation of Research 387's F5 (PKM × consolidation — "the fusion that would re-open the Super-GOAT question" per R387 §5); F4 is the riir-train secondary angle.

## 3. Verdict

**Tier: GOAT** (provable retention gain over the shipped TF-only write path; not a new capability class).

**One-line reasoning:** The TF-IDF slot-ranking mechanism is public prior art — it *is* this paper (and a follow-up already exists: "Improving Sparse Memory Finetuning", [arXiv:2604.05248](https://arxiv.org/abs/2604.05248), which retrofits dense pretrained LLMs into sparse memory-augmented models) — so Q1 fails exactly as PKM did in Research 387; but the *application* to our shipped δ-rule write gate closes a real, ablation-documented interference hole, modellessly, at the granularity (small t) where the paper measures the largest effect.

### Novelty gate (Q1–Q4)

| Q | Answer | Evidence |
|---|---|---|
| **Q1: No prior art?** | **NO.** TF-IDF memory-slot selection = the paper itself; memory layers = Berges 2024; PKM = Lample 2019; follow-up 2604.05248 exists. Web search (headline + components) found no earlier TF-IDF-slot-selection work — the base mechanism's novelty belongs to the paper, not us. | §4 searches |
| **Q2: New class of behavior?** | **PARTIAL.** Slot-level non-interference in writes is new *for us*; non-interference as a capability exists at branch level (RIZZ R310) and construction level (Hebbian margins R303). Refinement at a new granularity, not a new class. | §2.3 |
| **Q3: Product selling point?** | **PARTIAL — strengthens an existing one.** Feeds R310's claim ("NPCs learn without any one task corrupting another") at slot granularity; alone it is "our episodic writes don't erode prior recall" — incremental. | — |
| **Q4: Force multiplier?** | **YES.** PkmEpisodicStore + RIZZ branches + Hebbian shards/consolidation + SPEFT (4 substrates, 3 repos). | §2.5 |

**Q1 fails → not Super-GOAT.** Re-open only if the F3 consolidation fusion (IDF-ranked sleep-cycle over shards) beats Research 387 F5's retention gate (≥80% after 5 domain shifts) by a wide margin — that would be the measured moat, per the R387 precedent.

### MOAT gate per domain (§1.6)

| Domain | In scope? | MOAT contribution |
|---|---|---|
| `katgpt-rs` (public engine) | **YES — primitive lands here.** IDF-aware write-target ranking is generic count-based math on an already-open primitive (`PkmEpisodicStore`); no game/chain/shard semantics. | Retrieval/write stack ledger: PKM retrieval (default-on) → episodic δ-rule write → **IDF non-interference write**. |
| `riir-ai` | Fusion consumer (F2 hierarchical non-interference with RIZZ branches). | Private follow-up if F2 gates. |
| `riir-neuron-db` | Fusion consumer (F3 IDF-ranked consolidation + Hebbian slot assignment). | Strongest private continuation — Research 387 F5's open question. |
| `riir-chain` | NO. | Slot counts stay local; only existing BLAKE3 commitment crosses the boundary. |
| `riir-train` | Secondary (F4 background-relative SPEFT salience). | Training-track experiment, not a dependency — Path 0 closed modellessly. |

### UQ-bearing primitive check ("Report the Floor" rule)

**Not applicable.** The IDF score is a ranking statistic over candidate slots, not a probability/interval/coverage claim. No distributional assertion → conformal-naive floor does not apply.

### Defend-wrong PoC requirement (§3.6)

**Triggered for the quality claim.** The 11%-forgetting number is measured on a 1.3B-param LLM QA setup, not our substrate. Any "our writes now forget less" claim requires the head-to-head interference bench (F1's gate) on our own PKM/δ-rule stack: IDF vs TF-only vs frozen-baseline, matched learning. Architectural-only reasoning is insufficient — the GOAT gate in Issue 650 is the PoC. Latency claims need only the bench (µs-scale write budget, O(k) extra multiplies).

## 4. What ships vs what's new

| Mechanism | Status |
|---|---|
| PKM top-k factored retrieval | ✅ ships, default-on (Plan 408, G1 1670×) |
| δ-rule write gate over PKM | ✅ ships (`product_key_memory_episodic`) |
| freeze/thaw publish (BLAKE3, atomic) | ✅ ships (`FrozenProductKeyMemory`) |
| BM25/IDF math | ✅ ships for retrieval (`riir-neuron-db/src/bm25.rs`, crowd_mcgs `LexicalIndex`) |
| **IDF-aware write-target selection** | ✅ **shipped (Issue 650, resolved 2026-08-15 — Bench 636: G1 +12.5pp retention at matched learning)** |
| background access-count statistics table | ❌ new (LFU `access_counts` machinery exists as a pattern to mirror) |
| consolidation-level batch aggregation | ❌ follow-up (riir-neuron-db F3) |

## 5. Implementation priority

| Priority | Task | Gate |
|---|---|---|
| **P0** | `BackgroundAccessStats` + `build_background_stats` + `write_idf`/`write_weighted_idf` on `PkmEpisodicStore` (Issue 650 T1–T2) | G1 interference: recall(A-after-B) IDF ≥ TF-only by a margin, matched learning(B); G2 latency within write budget; G4 alloc-free |
| **P0** | GOAT bench `bench_481_idf_write_gate.rs` | G3 no-regression: `bench_408_*` unchanged with feature off |
| **P1** | F3: IDF-ranked consolidation sleep-cycle (riir-neuron-db) | R387 F5 gate: ≥80% recall after 5 domain shifts |
| **P2** | F2: hierarchical branch+slot non-interference (riir-ai) | multi-task-family interference bench |
| **P3** | F4: background-relative SPEFT salience (riir-train) | continual-FT retention at matched learning |

## 6. Cross-references

- **Closest cousins (shipped):** Research 387 (PKM + the F5 consolidation question this continues), Research 310 (RIZZ — branch-level non-interference; this note is the slot-level complement), Research 455 + riir-neuron-db Research 303 (Hebbian fact storing — construction-level interference avoidance via margins), Research 006 (Raven slots), Research 249 (DecentMem dual-pool stability/plasticity framing).
- **Issues:** `katgpt-rs `.issues/650`` — the P0 implementation + GOAT gate. **RESOLVED + removed (2026-08-15)** — record in [Bench 636](../.benchmarks/636_idf_write_gate.md): G1 PASS (+12.5pp norm-ramp / +1.6pp organic no-harm), G2 1.9× plain-write, G4 0 allocs, G3 bit-identical. Honest scope note recorded: random-within-pool outperforms IDF at read-width == write-pool; IDF is the relevance-aware policy.
- **Source paper:** [arXiv:2510.15103](https://arxiv.org/abs/2510.15103); memory layers: [arXiv:2412.09764](https://arxiv.org/abs/2412.09764); follow-up: [arXiv:2604.05248](https://arxiv.org/abs/2604.05248).

---

## TL;DR (one-line)

Paper proves the IDF half of slot ranking is what stops writes from erasing old knowledge; our `PkmEpisodicStore` currently ships TF-only writes — Issue 650 adds the missing IDF term (count-based, modelless, zero-alloc) and gates it on an interference retention bench.
