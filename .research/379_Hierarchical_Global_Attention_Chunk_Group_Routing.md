# Research 379: Hierarchical Global Attention (HGA) — Chunk→Group→Token Routing with RoPE-Aware Summaries

> **Source:** [Hierarchical Global Attention: Drop-In Exact-Token Routing for Pretrained Long-Context Transformers](https://arxiv.org/abs/2606.30709) — Woernle Frank, Vladimir Fedosov, Artemiy Grinenko (BMW Group), Jun 2026.
> **Code:** <https://github.com/vfedosov77/HierarchicalGlobalAttention>
> **Date:** 2026-07-05
> **Status:** Active
> **Classification:** Public
> **Related Research:** 071 (DashAttention — closest shipped cousin, chunk+token), 086 (RTPurbo — pre-RoPE low-dim projection), 044 (PFlash — sink/window/content-routed middle), 225 (MSA — failed-GOAT prior, blockwise sparse), 233 (Attention Matching — exact-token over compacted set), 213 (StillKV — β-bias compaction), 109 (Shard Drop), 258 (Sink-Aware + FlashMemory), 362 (HydraHead — head-importance calibration), 378 (HOLA — bounded exact KV cache on GDN2), 022 (Lighthouse — training-only multi-resolution pyramid)
> **Related Plans:** 106 (DashAttention, default-on), 044 (PFlash, default-on), 126 (RTPurbo, opt-in), 271 (AM KV Compaction, opt-in), 256 (MSA — GOAT FAILED precedent), 218 (BFCF LFU Sharding), 299 (Engram cache hierarchy), 335 (ZoneGeometryCache mmap-backed LRU)
> **Feature Gate:** `hga` (opt-in, requires GOAT proof of gain over DashAttention + PFlash + RTPurbo stack)

---

## TL;DR

HGA is a **drop-in sparse attention patch for pretrained long-context transformers** that preserves the checkpoint's WQ/WK/WV/WO unchanged (no calibration, no retraining) and replaces only "which historical K/V to fetch." It performs **two-level hierarchical routing** (chunk → group → token) using **RoPE-aware mixed-frequency summary keys**, fetches only the selected exact-token K/V from a **tiered Hot/Warm/Cold store** (summaries + sink/local on-device, full token K/V in host RAM/NVMe, routed working set on GPU), and runs ordinary softmax attention over the fetched set. Paper results: Qwen3-30B-A3B-Instruct-2507-FP8 on RTX 5090 (32 GB) at 32K context out-of-the-box (impossible with dense K/V); 100% needle-in-a-haystack at 64K with 1.9% sparsity; 0.01–0.02 nat loss gap vs dense at ~3% sparsity across 4K–64K; 2.43× prefill speedup (40M SmallLM at 12K).

**Distilled for katgpt-rs (modelless, inference-time):** three refinements of the shipped sparse-attention routing slot:

1. **Group-level middle routing tier** (chunk → group → token). DashAttention (R071) does chunk → token; PFlash (Plan 044) does block → token. **No shipped primitive has a sub-chunk group middle tier** — this is the strictly more granular refinement.
2. **Mixed-RoPE chunk/group summaries** — for each RoPE frequency pair: high-frequency pairs are *rotated per token then averaged*; low-frequency pairs are *averaged in raw-key space then rotated at the chunk-mid position*. This is the **third** RoPE-aware routing-summary construction in the literature (RTPurgo's pre-RoPE low-dim projection is the simpler alternative; StillKV/Plan 245's "RoPE averaging wedge" is the failure mode HGA's mixed rule avoids).
3. **Strict "summary-keys route, real-keys compute" rule** — chunk/group summaries are routing keys ONLY; the output softmax is always computed over exact token K/V from the fetched set. DashAttention's chunk summaries feed into the output softmax via the prior-induced bias; AM (R233) compacts into summary K/V that *is* the output. HGA's strict rule is the architectural commitment that makes "RAM-backed K/V" tractable: you only need to materialize exact token K/V for the routed working set, never for the summary store.

**Latent-space reframing (the part that decides GOAT vs Super-GOAT):** chunk/group summaries are *multi-resolution centroids* in the key-vector latent space; the chunk→group→token hierarchy is a *3-level coarse-to-fine latent retrieval cascade* on the same direction space. The latent reframing maps to (a) HLA per-NPC latent state: SenseLoD multi-resolution perception (`crates/katgpt-sense/src/lod.rs`) already does coarse-to-fine LOD; (b) `riir-ai/crates/riir-engine/src/latent_functor/zone_gating.rs` already does zone-level gating (coarse) within finer functor channels; (c) `NeuronShard` dendritic branch view (`shard.rs::dendritic`, `dendritic_lora` feature) is a sub-circuit group-level view within a shard — the closest structural analog; (d) DEC operators: chunk→group→token is multi-resolution cochain retrieval, sink/local = boundary condition, content-routed middle = interior flow. **None of these unlock a Super-GOAT** — the latent-reframe coverage is broad but the mechanism class (sparse long-context attention via hierarchical routing on pretrained checkpoints) is already shipped as DashAttention + PFlash + RTPurgo. HGA is a refinement, not a new class.

---

## 1. Paper Core Findings

### 1.1 The constraint HGA solves

For a pretrained long-context transformer (e.g., Qwen3-30B-A3B-Instruct-2507-FP8, native 262K context), the bottleneck is **dense K/V in VRAM**. The FP8 weights alone occupy almost the entire 32 GB budget of an RTX 5090; there is no room for full historical K/V even at 32K context. Existing sparse-attention methods (NSA, InfLLMv2, MInference A-shape) reduce *compute* but still require dense K/V materialized somewhere. HGA's contribution is **a systems-level patch**: keep the checkpoint unchanged, route hierarchically, materialize only the routed working set in VRAM, keep the rest in host RAM.

### 1.2 Three structural innovations

**(a) Two-level routing: chunk → group → token.** The sequence is divided into chunks (default `C=64` tokens). Each chunk is subdivided into groups (default `gs=16`). Routing has two budgets: `K_c` chunks (default 16–20) and `K_g` groups within selected chunks (default 32). The final attention opens only the groups' tokens. This is more granular than one-level block routing (NSA/InfLLMv2/DashAttention/PFlash) at the cost of one extra scoring pass.

**(b) RoPE-aware mixed-frequency summaries.** A chunk/group summary key must remain comparable to RoPE-rotated queries. The paper observes:
- High-frequency RoPE pairs (large `θ_i`) vary rapidly with position → averaging already-rotated keys works (captures rapid phase changes per token).
- Low-frequency RoPE pairs (small `θ_i`) change slowly → averaging raw keys then rotating at the chunk-mid position works (avoids phase noise from averaging over many positions).

So for each RoPE frequency pair `(x_i, y_i)`:
- High-frequency: rotate per token, then average.
- Low-frequency: average raw, then rotate at chunk-mid position.

This is the same diagnostic finding as Plan 245 StillKV GOAT metric fix: averaging RoPE-rotated keys over different positions produces a "rotation wedge" pointing in a meaningless direction. RTPurbo (R086) handles this by projecting to a 16-dim pre-RoPE subspace. HGA's mixed rule is a per-frequency-pair alternative that keeps the full key dimension.

**(c) Summary-keys-route, real-keys-compute.** Summaries are routing keys ONLY. After routing selects the chunks and groups, attention is computed exactly over the real token K/V from those selections — no summary K, no summary V, no learned gate, no calibration parameter enters the output softmax. This is the architectural commitment that allows a tiered store: the cold tier (host RAM) holds real token K/V; the hot tier (VRAM) holds summaries + sink/local + the currently-routed working set.

### 1.3 Tiered K/V store

`RamKVCacheStore` partitions by temperature:
- **Hot**: chunk summaries + always-visible sink chunks (first 2) + always-visible recent chunks (last 8). Always in VRAM.
- **Warm**: bounded LRU shard cache for recently-routed token chunks. Auto-shrunk to leave VRAM headroom.
- **Cold**: all remaining token K/V + group summaries in host RAM. Transferred to device only when routed.

**Cost:** for fixed `C`, fixed sink/local windows, fixed route budgets, each processing block attends to `O(C + (F+L)C + B_route)` real tokens — **linear in total tokens**, decoupled from context length.

### 1.4 Headline results

| Result | Number |
|---|---|
| Qwen3-30B-A3B FP8 on RTX 5090, 32K context, no fine-tuning | Runs out-of-the-box (impossible w/ dense K/V) |
| Loss gap vs dense (4K–64K, 3.13%–25% sparsity) | 0.01–0.02 nats |
| Needle-in-a-haystack @ 64K, 1.9% sparsity | 100% (3/3 depths) |
| 40M SmallLM copy-only dense→routed @ 8K | +0.01828 nat gap, +1.8% PPL |
| 40M SmallLM Triton-fused train @ 12K | 2.72× speedup |
| 40M SmallLM Triton-fused forward @ 12K | 2.43× speedup |
| Qwen3-0.6B fine-tune @ 4096, novel-text val | routed loss 3.196 vs dense 3.177 (+0.015 routing cost), 1.6× train throughput at 33% KV pairs |

**Side finding (Sec 5.5):** position-modulo RoPE wrapping `p ← p mod 65536` reduces the remaining loss gap, suggesting the residual error is dominated by long-context positional extrapolation (YaRN-extended models), not the routing algorithm. This is an interesting diagnostic for our existing long-context inference — flag for follow-up.

---

## 2. Distillation

### 2.1 What is transferable (modelless, inference-time)

**T1 — Group-level middle routing tier.** Insert a sub-chunk group-routing pass between DashAttention's chunk-level entmax routing and the token-level attention. Concretely: extend DashAttention's `ChunkSummaryCache` with `GroupSummaryCache { summaries: [n_chunks, n_groups_per_chunk, n_kv_head, head_dim] }`; add a `route_groups(query, selected_chunks) -> selected_groups` step. Modelless, parameter-free. Lands in the DashAttention routing slot.

**T2 — Mixed-RoPE summary construction.** Replace mean-pooling of RoPE-rotated keys (DashAttention's zero-init Stage 0) with the per-frequency-pair mixed rule:
```
for each RoPE freq pair (x_i, y_i):
  if θ_i > θ_threshold:  // high freq
      summary_i = mean_over_chunk(rotate(key_i, pos))
  else:                  // low freq
      summary_i = rotate(mean_over_chunk(key_i), chunk_mid_pos)
```
`θ_threshold` derived from `rope_theta` and a configurable crossover (paper doesn't pin it; the natural crossover is where the per-position phase change exceeds 2π over one chunk = `θ_i · C ≈ 2π`, i.e., `θ_i ≈ 2π/C`). Lands as a `MixedRopeSummarizer` next to DashAttention's `ChunkSummaryQuery`. **Important constraint:** the rule must hold for arbitrary `rope_theta` (Gemma 2 uses 10000; Qwen3 uses 1000000); the threshold must be derived, not hardcoded.

**T3 — Tiered Hot/Warm/Cold K/V store abstraction.** A `TieredKvStore` trait with hot (always-resident) / warm (LRU shard cache) / cold (host-RAM-backed) tiers, exposed via `route_and_fetch(query, sink_local_set, route_budget) -> WorkingSet`. The cold tier holds real token K/V (never summaries); the hot tier holds summaries + sink + local. This is the systems-level primitive that makes RAM-backed long-context inference tractable. Lands as a new `crates/katgpt-core/src/tiered_kv/` module. **Closest shipped cousin:** `ZoneGeometryCache` (riir-neuron-db Plan 335) — mmap-backed LRU with hot-path Arc clones; `Engram cache hierarchy` (Plan 299) — plasma/hot/warm/cold tiered; **no shipped tiered KV store with the route-and-fetch API.**

### 2.2 What is NOT transferable

- **Triton-fused GPU kernels** → riir-ai GPU territory (`riir-gpu`).
- **Qwen3-30B specific FP8 integration** → model-specific deployment concern, not a primitive.
- **Fine-tuning recipes** (Qwen3-0.6B QK fine-tune) → riir-train.
- **CUDA histogram top-p kernel** (already covered by RTPurbo R086's GPU path).

### 2.3 Prior-art check — the part that decides GOAT vs Super-GOAT

Per the skill's mandatory two-layer check (notes + shipped code, with vocabulary translation).

**Vocabulary translation (paper → codebase):**
- "hierarchical routing" / "two-level routing" → DashAttention (chunk→token), PFlash (block→token), VortexFlow (block sparse), Lighthouse (multi-resolution pyramid — training-only)
- "RoPE-aware summary" / "mixed-frequency rule" → RTPurbo pre-RoPE projection, StillKV/Plan 245 "RoPE averaging wedge" diagnostic
- "tiered Hot/Warm/Cold K/V store" → AGENTS.md constraint 8 (Plasma/Hot/Warm/Cold/Freeze), BFCF LFU Sharding (Plan 218), Engram cache hierarchy (Plan 299), `ZoneGeometryCache` mmap-backed LRU (Plan 335)
- "summary-keys route, real-keys compute" → AM compacts into summary K/V (opposite design), DashAttention's summaries feed output softmax via prior-induced bias (different design), HOLA stores real tokens (same as HGA on this axis)
- "deterministic sink + local + content-routed middle" → PFlash `block_select` sink+window+last_n_full+alpha (exact pattern)

**Notes layer (grep results — both vocabularies):**

| Paper concept | Closest shipped note | Coverage |
|---|---|---|
| Hierarchical chunk→token routing | R071 DashAttention, R022 Lighthouse (training-only) | ✅ Class ships as DashAttention (chunk + token); Lighthouse is multi-resolution but training-only. |
| RoPE-aware chunk summary | R086 RTPurbo (pre-RoPE 16-dim projection), R213 StillKV (β-bias compaction), Plan 245 GOAT metric fix (RoPE averaging wedge diagnostic) | ⚠️ Concept ships; **no per-frequency-pair mixed rule**. |
| Tiered Hot/Warm/Cold K/V store | R218 BFCF LFU Sharding, R299 Engram cache hierarchy, R109 Shard Drop | ⚠️ Tier concept ships across the codebase; **no specific route-and-fetch KV store abstraction**. |
| Sink + local + content-routed middle | R258 Sink-Aware + FlashMemory, Plan 044 PFlash (sink+window+last_n_full+alpha), Plan 287 Sink-Aware Dual-Mechanism | ✅ Exact pattern ships as PFlash `block_select`. |
| Summary-keys-route, real-keys-compute | R233 AM (compacts into summary K/V), R071 DashAttention (summaries feed output via prior bias), R378 HOLA (real tokens only) | ⚠️ Each shipped primitive differs in detail; **no shipped primitive combines all three (route-only summary + real-token output + tiered store)**. |
| Sparse attention GOAT precedent (negative) | R225 MSA — **GOAT FAILED** | ⚠️ Important: MSA (blockwise sparse with per-GQA-group selection + max-pool) failed its GOAT gate. The sparse-attention routing slot is contested; HGA must beat DashAttention on our harness, not just on the paper's. |

**Code layer (mandatory — the layer that catches overclaims):**

Grep across `katgpt-rs/**/*.rs` for `group_summary|two_level_route|chunk_group_route|mixed_rope|tiered_kv|RamKV`:

| Match | Location | Is this HGA's mechanism? |
|---|---|---|
| `chunk_summary.rs` learned summary query | `katgpt-rs/crates/katgpt-attn/src/dash_attn/chunk_summary.rs` (Plan 106) | NO — single-level (chunk), learned via local SDPA, used as routing key AND feeds output softmax via prior-induced bias. HGA's group level + mixed-RoPE + route-only rule are absent. |
| PFlash `block_select` sink+window+last_n_full+alpha | `katgpt-rs/src/speculative/prefill.rs` (Plan 044) | NO — block-level only, no sub-block group tier, mean-K scoring not RoPE-aware mixed. |
| RTPurbo pre-RoPE projection | `katgpt-rs/src/rt_turbo/` (Plan 126) | NO — projects to 16-dim pre-RoPE subspace; HGA keeps full-dim keys with per-frequency-pair mixed rule. |
| BFCF LFU Sharding Hot/Warm/Cold | `katgpt-rs/src/bfcf/` (Plan 218) | NO — region-level cache (≈50 regions), not K/V tiered store. |
| `ZoneGeometryCache` mmap-backed LRU | `riir-neuron-db/src/zone_cache.rs` (Plan 335) | NO — zone geometry pods, not K/V; but **structurally the closest systems-level match** (Arc<Mmap> + lock-free papaya + LRU deque + cold-path regen). |
| HOLA bounded exact KV cache | (R378, Plan 378 — in-flight) | NO — surprise-evicted bounded cache on GDN2 backbone; HGA is route-fetched bounded working set on dense attention. Different backbone, different selection mechanism. |

**Grep for `group_route|sub_chunk|two_stage_route|RamKVCacheStore|mixed_rope_summary` → NO MATCHES in any `.rs`.** No shipped primitive implements (a) a sub-chunk group middle routing tier, (b) the per-frequency-pair mixed-RoPE summary construction, or (c) the route-and-fetch tiered K/V store abstraction.

**Conclusion:** HGA's three deltas (T1 group middle tier, T2 mixed-RoPE summary, T3 tiered route-and-fetch store) are **novel as shipped code**. The HGA *latent-space reframing* — multi-resolution coarse-to-fine retrieval on pretrained key directions — **already ships as DashAttention + PFlash + RTPurbo + Lighthouse** (and SenseLoD + zone_gating + dendritic branch view + DEC multi-resolution cochains across the latent substrates). HGA is a refinement of the sparse-attention routing slot, not a new capability class.

### 2.4 Fusion (the GOAT-tier combination)

| Fusion | Existing system | What HGA adds | Gate |
|---|---|---|---|
| **F1: HGA × DashAttention** | DashAttention (R071, default-on): chunk-level entmax routing + learned chunk summary + prior-induced sparse softmax bias | Insert group middle tier (T1) between DashAttention's chunk routing and token softmax; replace zero-init mean-pool with T2 mixed-RoPE summary. The DashAttention `head_cls` learned query can still operate on the group summary (just at finer granularity). | On a synthetic NIAH harness at 64K with Qwen3-style RoPE, does DashAttention+HGA-group+mixed-RoPE match DashAttention's accuracy at ≥2× fewer fetched tokens (group-level sparsity compounds with chunk-level)? |
| **F2: HGA × RTPurbo** | RTPurbo (R086, opt-in): pre-RoPE 16-dim projection for retrieval heads | HGA's mixed-RoPE summary is the *full-dim* alternative to RTPurbo's *low-dim* projection. For retrieval heads, compare: RTPurbo 16-dim projection vs HGA full-dim mixed-RoPE chunk summary. The two are alternative answers to "how do I make RoPE keys comparable across positions for routing?" | Same retrieval-recall gate as RTPurbo's G3 (16-dim achieves >85% recall of top-256 full-dim tokens); HGA must match or beat on retrieval heads at similar selection cost. |
| **F3: HGA × PFlash** | PFlash (Plan 044, default-on): block_select with sink+window+last_n_full+alpha threshold | HGA's tiered route-and-fetch store (T3) is the *systems substrate* PFlash's block_select was always meant to run on. Currently PFlash selects blocks then flattens to token indices; with T3, PFlash selects blocks then the store fetches only those blocks' K/V from cold tier. | On a 32K prefill, does PFlash + T3 tiered store reduce peak VRAM by ≥3× at iso-quality? |
| **F4: HGA × HOLA × AM (the tiered compaction stack)** | HOLA (R378) surprise-evicted warm exact cache; AM (R233) cold-tier offline mass-preserving compaction | HGA's tiered store (T3) is the unified substrate: HOLA cache = warm tier; AM-compacted prefix = cold tier; HGA route-and-fetch = the dispatch API. The three primitives compose into a complete tiered long-context KV stack. | On a 64K context with 8 GB VRAM budget, does HOLA-warm + AM-cold + HGA-route recover ≥90% of dense attention quality? |
| **F5: HGA × DEC multi-resolution cochains** | DEC operators (Plans 251–252, Research 219): `exterior_derivative`, `codifferential`, `hodge_decompose` on `CellComplex` | Chunk→group→token hierarchy is a 3-level cell complex; routing is a cochain retrieval operation. The "sink+local = boundary, content-routed middle = interior flow" maps to DEC's exact/coexact/harmonic decomposition. **Speculative** — DEC operators are d ≤ 3 (game maps, HLA regions, KG embeddings); applying them to attention K/V (d=64+) violates the curse-of-dimensionality caveat. | Speculative — defer. |

The GOAT-tier claim is **F1 + F3** — HGA's group tier + mixed-RoPE on the DashAttention backbone, with PFlash driving the route-and-fetch substrate. F2 is the alternative-routing-summary head-to-head (HGA mixed-RoPE vs RTPurbo low-dim projection). F4 is the longer-horizon tiered compaction stack fusion (depends on HOLA Plan 378 shipping first).

### 2.5 Latent-to-latent reframing (mandatory per research skill)

How does HGA look when operating on the seven Super-GOAT factory substrates?

- **(a) HLA per-NPC latent state** (`katgpt-core/src/sense/`, `riir-engine/src/hla/`): chunk/group summaries = multi-resolution centroids in the 8-dim HLA affect space. `SenseLoD` (`crates/katgpt-sense/src/lod.rs`) already does coarse-to-fine LOD prediction before full perception. **HGA's group middle tier is the affect-space analog of SenseLoD's LOD-2 between LOD-1 (zone) and LOD-3 (token).** Marginal — SenseLoD covers the multi-resolution pattern; HGA's specific 3-level routing is a refinement.
- **(b) `latent_functor/`**: chunk→group→token = functor application at 3 resolutions. `zone_gating.rs` does zone-level gating (coarse); finer gating within zone = group level. **HGA's group tier is the missing middle functor channel.** Moderate.
- **(c) `cgsp_runtime/`**: curiosity routing at chunk vs token granularity. Curiosity class router already selects per-cycle; HGA's middle tier = per-sub-cycle selection. Marginal.
- **(d) LatCal fixed-point**: not applicable — LatCal is raw numeric commitment, not retrieval.
- **(e) `NeuronShard` dendritic branch** (`riir-neuron-db/src/shard/mod.rs`, `dendritic_lora` feature): **strong structural match.** A dendritic branch is a sub-circuit of `style_weights[64]` — the group-level view within a shard (chunk). HGA's chunk→group→token maps to shard→branch→weight. The `ZoneGeometryCache` mmap-backed LRU (Plan 335) is the **systems-level exact match** for HGA's tiered store (Arc<Mmap> + lock-free papaya + LRU + cold-path regen). **Fusion target: HGA's tiered route-and-fetch applied to dendritic branch retrieval — shard cold tier, branch warm tier, weight hot tier.** This is a riir-neuron-db private follow-up.
- **(f) DEC operators** (`katgpt-dec/src/`): chunk→group→token = 3-level cell complex. Sink/local = boundary condition. Content-routed middle = interior flow. **Speculative** — DEC operators are validated for d ≤ 3; attention K/V at d=64+ violates the curse-of-dimensionality caveat (boundary larger than interior). Defer.

The NeuronShard dendritic (e) reframing is the most interesting — it's a *latent-state substrate where multi-resolution retrieval is currently single-level* (shard → weight, no branch middle tier). However, that's a riir-neuron-db private concern. The katgpt-rs public primitive is the **generic tiered route-and-fetch store** that both attention K/V and dendritic branch retrieval can consume.

---

## 3. Verdict

### Tiers

| Tier | Criteria | Routing |
|---|---|---|
| Super-GOAT | Novel mechanism + new capability class + selling point + force multiplier | Open primitive + private guide + plans |
| **GOAT (this)** | **Provable gain over existing approach, not a new class. Promote if it wins.** | **Plan + implement + benchmark (Plan 379).** |
| Gain | Incremental, useful but not headline | Plan only, behind flag |
| Pass | Not relevant, OR training-only | One-line note |

**Verdict: GOAT.**

**One-line reasoning:** HGA's three deltas (group middle tier, mixed-RoPE summary, tiered route-and-fetch store) are **novel as shipped code** with **provable gains** (paper: 0.01–0.02 nat gap at 3% sparsity, 100% NIAH at 64K, 2.43× prefill speedup, Qwen3-30B FP8 runs at 32K on RTX 5090 where dense K/V is impossible), but they are **not a new capability class** — sparse long-context attention via hierarchical routing on pretrained checkpoints already ships as DashAttention (R071, default-on) + PFlash (Plan 044, default-on) + RTPurbo (R086, opt-in). HGA is a refinement of the same transformer-stack routing slot.

### Novelty gate (Q1–Q4)

- **Q1 — No prior art?** YES at mechanism level (no shipped group middle tier; no shipped per-frequency-pair mixed-RoPE summary; no shipped route-and-fetch tiered KV store abstraction). NO at class level (DashAttention + PFlash + RTPurbo cover the sparse hierarchical routing class). **Mixed → not novel enough for Super-GOAT.**
- **Q2 — New capability class?** NO. "Sparse long-context attention via hierarchical routing on pretrained checkpoints, no retraining" is the shipped DashAttention capability. HGA adds finer granularity (group tier), alternative routing-summary construction (mixed-RoPE), and systems substrate (tiered store) — all refinements of the same class.
- **Q3 — Product selling point?** NO. "Our sparse attention fetches fewer tokens at the same quality" is a perf optimization on a shipped slot, not a new product capability. The Qwen3-30B-at-32K-on-RTX-5090 demo is a deployment capability, not a product moat — any competitor with the same sparse attention primitive achieves it.
- **Q4 — Force multiplier?** MODERATE — connects DashAttention (R071), PFlash (Plan 044), RTPurbo (R086), AM (R233), HOLA (R378, in-flight), BFCF (R218), ZoneGeometryCache (Plan 335). But the sparse-attention routing slot is already a connected stack; HGA is a new entry in the slot, not a new bus.

**Q2 + Q3 fail → GOAT, not Super-GOAT.** No riir-ai / riir-chain / riir-neuron-db guide created. Plan the open primitive + GOAT gate.

### MOAT gate per domain (§1.6)

| Domain | In scope? | MOAT contribution |
|---|---|---|
| `katgpt-rs` (public engine) | ✅ YES — paper-derived transformer-stack-slot primitive (sparse attention routing) | **Promote/demote tracked per stack.** Lands in the **sparse-attention routing slot** alongside DashAttention (R071, default-on), PFlash (Plan 044, default-on), RTPurbo (R086, opt-in), MSA (R225, **GOAT FAILED**), VortexFlow (R176). HGA competes for the same slot. GOAT gate decides promote-to-default vs demote-loser vs coexist-by-feature-flag. |
| `riir-ai` (private runtime) | NO — generic engine primitive, no game IP. The dendritic-branch retrieval fusion (§2.5(e)) is a riir-neuron-db private follow-up, not riir-ai. | — |
| `riir-chain` (private chain) | NO — no commitment / sync / LatCal angle. | — |
| `riir-neuron-db` (private shards) | YES at *latent-reframe* level (HGA's tiered route-and-fetch maps to dendritic branch retrieval on NeuronShard). NO at mechanism level (HGA's mechanism is transformer-stack K/V, not shards). | Cross-reference only — apply the open `TieredKvStore` primitive to dendritic branch retrieval as a private follow-up. |
| `riir-train` (private training) | NO — HGA is explicitly modelless (paper's headline: no retraining). Fine-tuning recipes (Qwen3-0.6B QK) noted as out-of-scope. | — |

**Per-stack promote/demote ledger (the engine's quality contract):**

| Slot | Current default | HGA's claim | Gate |
|---|---|---|---|
| Sparse attention routing (long-context decode + prefill) | DashAttention (default-on) + PFlash (default-on) + RTPurbo (opt-in, head-specialized) | Group middle tier + mixed-RoPE summary + tiered route-and-fetch store | G1 routing correctness (group-tier selection is deterministic, mixed-RoPE summary preserves ranking vs full-attention on a fixed Q) + G2 perf/quality (on a synthetic NIAH harness at 32K–64K with Qwen3-style RoPE, HGA matches DashAttention accuracy at ≥2× fewer fetched tokens, OR matches at iso-tokens with >0.005 nat better loss) + G3 no-regression (full-coverage mode = causal SDPA bit-identical) + G4 alloc-free hot path (route + fetch + attend all in pre-allocated scratch) + G5 latency (group-routing pass < DashAttention chunk-routing pass × 1.5) |

**G2 is the load-bearing gate.** The MSA precedent (R225, GOAT FAILED) is the cautionary tale: blockwise sparse attention with per-GQA-group selection and max-pool scoring *failed its GOAT gate* on our harness. HGA shares the same primitive class; the GOAT gate is non-trivial. If G2 fails (HGA does not beat DashAttention at iso-quality on our harness), HGA stays opt-in and is documented as a negative result. If G2 passes, promote to default-on and demote whichever of DashAttention/PFlash/RTPurbo loses the head-to-head on the same harness.

### §3.6 PoC requirement check

This verdict makes **no quality-parity claim** with the paper. Claims made:
- (a) **Architectural** — DashAttention ships chunk-level routing + learned summary (grep-proven), PFlash ships sink+window+content-routed block_select (grep-proven), RTPurbo ships pre-RoPE projection (grep-proven), `ZoneGeometryCache` ships mmap-backed LRU (grep-proven). Each proven by reading the code.
- (b) **Mechanism novelty** — no shipped primitive implements group middle tier / mixed-RoPE summary / route-and-fetch tiered KV store (grep-proven, zero hits on each).
- (c) **Class coverage** — the sparse hierarchical routing class ships as DashAttention + PFlash + RTPurbo (grep + read proven above).

No claim that "our HGA implementation matches the paper's Qwen3-30B 64K NIAH 100%." The paper's numbers come from a 30B FP8 model on an RTX 5090 — that is a deployment concern, not a modelless claim. **No PoC required.** The Plan 379 GOAT gate (G1–G5) on a synthetic NIAH harness at 32K–64K is the modelless validation; full-model parity is explicitly deferred (and is a riir-train job for the fine-tuning variant).

---

## 4. Implementation Sketch (delegates to Plan 379)

1. **`GroupSummaryCache<const C, const GS, const D>`** in `crates/katgpt-core/src/hga/group_summary.rs` — fixed-layout `[n_chunks, C/GS, n_kv_head, D]` summary store. Append-only during decode. Feature flag `hga`.
   - `append_chunk(chunk_keys: &[f32], positions: &[u32], rope_freqs: &RopeFreqs)` → computes per-frequency-pair mixed summary (high-freq rotate-then-average, low-freq average-then-rotate-at-mid) and stores.
   - `score_groups(query: &[f32], selected_chunks: &[usize]) -> Vec<(chunk_idx, group_idx, score)>` → dot-product scoring of query against group summaries within selected chunks.

2. **`MixedRopeSummarizer`** in `crates/katgpt-core/src/hga/summary.rs`:
   - `summarize(keys: &[f32], positions: &[u32], rope_freqs: &RopeFreqs) -> [f32; D]` → per-frequency-pair mixed rule.
   - Threshold `θ_threshold = 2π / C` derived from chunk size (configurable).
   - **Important constraint:** must work for arbitrary `rope_theta` (Gemma 2 = 10000, Qwen3 = 1000000); the threshold is derived from `rope_freqs`, not hardcoded.

3. **`TieredKvStore` trait** in `crates/katgpt-core/src/tiered_kv/mod.rs`:
   ```rust
   pub trait TieredKvStore<const D> {
       fn append_chunk(&mut self, chunk_keys: &[f32], chunk_values: &[f32], positions: &[u32]);
       fn route_and_fetch(
           &self,
           query: &[f32],
           sink_local_set: &SinkLocalSet,
           route_budget: RouteBudget,
       ) -> WorkingSet<D>;
   }
   ```
   - `InMemoryTieredKvStore` — hot (always-resident summaries + sink + local), warm (LRU shard cache), cold (host-RAM-backed full token K/V). The reference implementation; production mmap-backed variant is a riir-ai/riir-neuron-db follow-up (mirrors `ZoneGeometryCache`).

4. **HGA forward path** in `crates/katgpt-forward/src/forward.rs`:
   - Compose `GroupSummaryCache` + `MixedRopeSummarizer` + `TieredKvStore` + DashAttention's α-entmax routing (reuse Plan 106's `entmax_1p5` + `entmax_gqa_aggregate`).
   - Stage 1: chunk-level entmax routing (reuses DashAttention).
   - Stage 2: group-level top-K routing within selected chunks (new).
   - Stage 3: route-and-fetch from tiered store + exact softmax over fetched token K/V.

5. **GOAT gate** (Plan 379 Phase 2):
   - G1: full-coverage mode (`route_budget = infinity`) = causal SDPA bit-identical (within f32 noise, < 1e-5).
   - G2: on a synthetic NIAH harness at 32K and 64K with Qwen3-style RoPE (`rope_theta = 1000000`), HGA at 3% sparsity matches DashAttention at 6% sparsity on retrieval accuracy, OR matches at iso-sparsity with >0.005 nat better loss.
   - G3: zero warnings on `cargo check --features hga --all-features` (combo-regression check).
   - G4: route + fetch + attend hot path is alloc-free (pre-allocated scratch buffers).
   - G5: group-routing pass latency < 1.5× DashAttention chunk-routing pass latency at the same context length.

6. **Promote/demote decision:**
   - If G2 passes → promote `hga` to default-on; run head-to-head vs DashAttention on the same harness; demote the loser to opt-in.
   - If G2 fails → keep `hga` opt-in; document as a negative result (the MSA precedent).
   - If G2 is inconclusive (within noise) → keep `hga` opt-in as an alternative routing-summary construction (mixed-RoPE vs DashAttention learned summary).

---

## 5. Open Questions / Risks

1. **MSA precedent (R225 GOAT FAILED).** Blockwise sparse attention with per-GQA-group selection and max-pool scoring failed its GOAT gate on our harness. HGA shares the primitive class. The GOAT gate (G2) is non-trivial — if HGA does not beat DashAttention at iso-quality, it stays opt-in. **Mitigation:** G2 harness must be the same NIAH harness used for DashAttention's GOAT 9/9 (so the comparison is apples-to-apples, not HGA on its own favorable harness).

2. **Mixed-RoPE threshold is paper-vague.** The paper does not pin `θ_threshold`; the natural crossover (`θ_i · C ≈ 2π`) is our derivation. For Gemma 2 (`rope_theta = 10000`) and Qwen3 (`rope_theta = 1000000`), the threshold lands at different frequency-pair indices. **Risk:** the mixed rule may degrade to all-high-freq or all-low-freq on some `rope_theta` values, providing no benefit over mean-pool. **Mitigation:** G2 must sweep `rope_theta ∈ {10000, 50000, 1000000}` and report per-config gain.

3. **Group tier adds a scoring pass.** HGA's two-level routing is strictly more compute than DashAttention's one-level. The paper's gains come from fetching fewer tokens (smaller softmax), not from cheaper routing. **Risk:** on our CPU/SIMD/GPU targets, the group-scoring pass may dominate for short contexts (4K) where token fetch is cheap. **Mitigation:** G5 latency gate + feature-gate to long-context-only (`ctx_len > threshold`).

4. **Tiered K/V store is a systems primitive, not a math primitive.** The route-and-fetch API is straightforward; the hard part is the host-RAM cold tier (mmap, page faults, NUMA). For the katgpt-rs public primitive, we ship the `InMemoryTieredKvStore` reference; production mmap-backed variant is a riir-ai/riir-neuron-db follow-up (mirrors `ZoneGeometryCache` Plan 335). **Risk:** the in-memory variant provides no VRAM benefit (everything is in process RAM anyway), so the headline Qwen3-30B-at-32K-on-RTX-5090 demo is not reproducible on CPU-only katgpt-rs. The modelless gain is the routing algorithm, not the storage tier.

5. **Position-modulo RoPE wrapping (Sec 5.5).** The paper's side finding — wrapping RoPE position `p mod 65536` reduces the residual loss gap — suggests the remaining error is positional extrapolation, not routing. **This is an interesting diagnostic for our existing long-context inference** (Gemma 2 2B at extended context). Flag as a separate follow-up — not part of the HGA primitive, but a candidate improvement to `apply_rope_with_freq` in riir-engine.

6. **Contested slot, demote-the-loser discipline.** The sparse-attention routing slot is the most crowded slot in katgpt-rs (DashAttention, PFlash, RTPurbo, MSA-failed, VortexFlow, AM). Adding HGA without a clean head-to-head winner risks feature-flag proliferation. **Mitigation:** G2 head-to-head must produce a single winner; the loser demotes to opt-in or removes (per the §1.6 promote/demote ledger rule).

---

## 6. Side note: position-modulo RoPE wrapping (deferred)

Section 5.5 of the paper reports that wrapping RoPE position `p ← p mod 65536` reduces the residual loss gap on YaRN-extended models. The hypothesis: under sparse routing, only a subset of layers observes distant tokens, so positional-extrapolation inaccuracies become more visible than under dense attention.

**This is a candidate improvement to our long-context inference path** (`apply_rope_with_freq` in riir-engine, `simd_matmul_rmsnorm_rope` in katgpt-core). It is orthogonal to HGA — applies to any sparse attention mechanism on YaRN-extended models. Flag as a separate issue/plan if long-context Gemma 2 inference shows unexplained quality degradation past training length.

---

## TL;DR

HGA is a drop-in sparse attention patch for pretrained long-context transformers — no retraining, no calibration, no new parameters, only "which historical K/V to fetch." Three refinements of the shipped sparse-attention routing slot: (1) **group middle tier** (chunk → group → token, vs DashAttention's chunk → token), (2) **mixed-RoPE per-frequency-pair summary** (high-freq rotate-then-average, low-freq average-then-rotate, vs RTPurbo's pre-RoPE low-dim projection), (3) **tiered Hot/Warm/Cold route-and-fetch store** (no shipped route-and-fetch KV abstraction). Paper: 0.01–0.02 nat gap at 3% sparsity across 4K–64K, 100% NIAH at 64K, 2.43× prefill speedup, Qwen3-30B FP8 runs at 32K on RTX 5090 where dense K/V is impossible.

**Verdict: GOAT.** Not Super-GOAT — the sparse hierarchical routing class already ships as DashAttention (default-on) + PFlash (default-on) + RTPurbo (opt-in) + AM (opt-in); HGA is a refinement of the same transformer-stack routing slot. Q1 mixed (mechanism-level novel, class-level covered), Q2 NO (not new capability), Q3 NO (perf optimization on shipped slot), Q4 MODERATE (connects 5 sparse-attention primitives but slot is already connected).

Plan 379 ships the open primitive behind `hga` feature flag. GOAT gate G1–G5 on a synthetic NIAH harness at 32K–64K with Qwen3-style RoPE. **G2 is load-bearing** — must beat DashAttention at iso-quality (the MSA R225 GOAT-FAILED precedent looms). Promote-to-default + demote-loser if G2 passes; opt-in + negative-result doc if G2 fails. Latent reframings (NeuronShard dendritic branch tiered retrieval, SenseLoD multi-resolution, latent_functor middle channel) noted as private riir-neuron-db / riir-ai follow-ups. Position-modulo RoPE wrapping (Sec 5.5) flagged as a separate orthogonal improvement to long-context inference. No riir-ai / riir-chain / riir-neuron-db guide created (not Super-GOAT). No PoC required (no quality-parity claim with paper).

> **PASS-Redirects (synthesis):** Nakayama et al. [arXiv:2610.05816 "Level-of-Token Diffusion"] — its fitted extent-pooling stack (Procrustes bases Eq. 14 + RMS rescale Eq. 15) is the merge-operator cousin of HGA's `MixedRopeSummarizer`; verdict PASS (class well-published: BLT/NSA/MoD/SeKV), with the one surviving residue routed to `.issues/925_lot_fitted_pooling_routing_probe.md` — a fitted static query-independent summary vs MixedRope/Mean at equal cost on Issue 908 (closed, `778de9af8`) real-tensor captures (random-key harness banned per `pyramid_topk.rs` slot discipline; `MeanPlusHalfVar`/`ExactLse` as cost-stated ceiling per Bench 612).
