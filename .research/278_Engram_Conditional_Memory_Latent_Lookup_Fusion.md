# Research 278: Engram — Conditional Memory as a New Sparsity Axis (Open Primitive)

> **Source:** Cheng, Zeng, Dai, Chen et al. (Peking U. + DeepSeek-AI), "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models", [arXiv:2601.07372](https://arxiv.org/pdf/2601.07372), 13 Jan 2026.
> **Date:** 2026-06-21
> **Status:** Active — open primitive scoped, plan open (P299)
> **Classification:** **Public** (modelless engine primitive)
> **Related Research:** 006 (Raven RSM — the complementary *computation* axis), 196 (KG Latent Octree — spatial lookup, different substrate), 262 (Lore ContentStore — Merkle-blob lookup), 268 (Forensic Asset Fingerprinting — BLAKE3-seeded addressing, same primitive different domain), 276 (Personality-Weighted Composition — same sigmoid×direction kernel, different source)
> **Companion plan:** [`.plans/299_Engram_Hash_Addressed_Pattern_Memory.md`](../.plans/299_Engram_Hash_Addressed_Pattern_Memory.md)
> **Cross-ref (2026-10-29, the retrieved-corpus sibling):** hermabr's [modded-nanogpt PR #367](https://github.com/KellerJordan/modded-nanogpt/pull/367) + [the author's writeup](https://brunborg.com/n-gram-hinted-lm) — exact-match n-gram hints retrieved from the FROZEN training corpus (hint = count-weighted embedding of retrieved next tokens from the model's own table), trusted by retrieval METADATA (length × count × purity) instead of a hidden-state gate, consumer pretrained to read hints. Design-space note only — no change to the P299 primitive (engram's learned table + hidden-state gate stays the inference-side primitive; the PR's table pays off only because its consumer co-trains). Full record: riir-train Research 482 (private); builder distill: riir-refine Research 346 (private).
> **Cross-ref (private selling-point guide):** `riir-ai/.research/147_Engram_Conditional_Memory_NPC_Guide.md`

---

## TL;DR

Engram is **conditional memory** — a sparsity axis *complementary* to conditional computation (MoE/Raven/dMoE). Where MoE scales *active parameters* per token, Engram scales *static lookup slots* per token. The primitive is pure inference-time data plumbing: N-gram-suffix → multi-head hash → O(1) embedding-table lookup → context-aware sigmoid gate → residual-fuse into hidden state. Paper measured <3% throughput penalty when offloading a **100B-parameter** table to host DRAM (deterministic addressing enables async prefetch overlapping compute).

**Distilled for katgpt-rs (modelless, inference-time):**

A generic, hash-addressed, sigmoid-fused static memory lookup primitive. No training, no backprop. The table is populated offline and frozen; updates are atomic Arc swaps (freeze/thaw pattern). The mechanism reduces to:

```text
hash_keys = multi_head_hash(n_gram_suffix(input_ids))   # K deterministic hashes, O(1)
e_t       = concat(table[k] for k in hash_keys)          # multi-head retrieval
α_t       = σ(RMSNorm(q_t) · RMSNorm(W_K e_t) / √d)     # sigmoid gate (NEVER softmax)
output_t  = α_t · (W_V e_t)                              # gated residual contribution
h_t      += output_t                                     # residual fuse
```

Plus: tokenizer compression (surjective `V → V'` via NFKC+lowercase), depthwise causal conv, multi-branch shared-value/distinct-key projection (mHC optional).

---

## 1. Paper Core Findings

### 1.1 The New Sparsity Axis

Two qualitatively different sub-tasks in language modeling: **compositional reasoning** (needs deep dynamic compute) and **knowledge retrieval** (mostly local, static, stereotyped — named entities, formulaic phrases, idioms). Standard Transformers force both through the same depth stack, wasting depth reconstructing static patterns.

| Sparsity axis | Activates | Mechanism | Latency |
|---|---|---|---|
| Conditional computation (MoE) | Active parameters per token | Top-K routing via hidden state | O(routed experts) |
| **Conditional memory (Engram)** | **Lookup slots per token** | **N-gram hash → table → sigmoid fuse** | **O(1)** |

The U-shaped scaling law (paper Fig 3): under iso-parameter + iso-FLOPs, optimal allocation is `ρ ≈ 80%` MoE + `20%` Engram. Pure-MoE is suboptimal because the backbone wastes depth on static patterns; pure-Engram loses conditional compute. **Hybrid wins.**

### 1.2 Architecture (§2)

Two phases per token `t`:

**Phase 1 — Sparse Retrieval via Hashed N-grams:**

1. **Tokenizer compression**: surjective `P: V → V'` collapses raw IDs to canonical via NFKC normalization + lowercasing. 23% vocabulary reduction on a 128k tokenizer (Appendix C).
2. **Multi-head hashing**: for each order `n ∈ {2, 3}` and head `k ∈ {1..K}`, a deterministic multiplicative-XOR hash `φ_{n,k}` indexes into table `E_{n,k}` of prime size `M_{n,k}`. Memory vector `e_t = concat(e_{t,n,k})`.
3. **O(1) retrieval**: fixed number of independent hash table lookups, independent of total table size.

**Phase 2 — Context-Aware Gating & Fusion:**

4. **Branch-specific gating** (mHC backbone, `M = 4`): shared Value `W_V`, distinct Keys `W_K^{(m)}`. Per-branch gate: `α_t^{(m)} = σ( RMSNorm(h_t^{(m)})^T · RMSNorm(W_K^{(m)} e_t) / √d )`. Output `u_t^{(m)} = α_t^{(m)} · (W_V e_t)`. All `M` gates share retrieved `e_t` → enables fusing `(W_V, {W_K^{(m)}})` into one dense FP8 matmul.
5. **Depthwise causal conv**: `Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ`, kernel 4, dilation = max N-gram order.
6. **Residual**: `H^{(ℓ)} ← H^{(ℓ)} + Y`, then standard Attention + MoE.

### 1.3 Mechanistic Insight (§6.1) — Effective Depth

LogitLens + CKA show Engram's layer 5 ≈ MoE baseline's layer 12. Static feature composition is offloaded to O(1) lookup, freeing attention for global context. This is why Engram *boosts long-context retrieval* (Multi-Query NIAH: 84.2 → 97.0). Suppressing Engram at inference collapses factual knowledge (TriviaQA: 29% retained) but leaves reading comprehension resilient (C3: 93% retained) — Engram is the *primary* knowledge repository; attention handles context-grounded tasks.

### 1.4 System Efficiency (§2.5 + §6.4)

- **Deterministic addressing** → indices known before forward → async host-memory prefetch overlapping preceding-layer compute. **100B-param Engram table offloaded to host DRAM incurs <3% throughput penalty** on H800.
- **Zipfian N-gram distribution** → multi-level cache hierarchy (HBM cache, host DRAM warm, NVMe cold tail). Same shape as plasma→hot→warm→cold tiering.
- **Independent of compute** — adding slots doesn't increase per-token FLOPs (constant retrieval count).

---

## 2. Distillation (modelless)

### 2.1 What ships in katgpt-rs (open engine)

The mechanism is **inference-time only**. Training Engram from scratch is a `→ riir-train` concern (U-shaped scaling law, sparsity allocation). At inference, the table is a **frozen snapshot** populated offline; updates are atomic Arc swaps.

```rust
/// A frozen, hash-addressed pattern memory table.
///
/// Generic over the embedding dimension `D`. Each entry is a static
/// `[f32; D]` direction vector. Lookups are O(1) via multi-head hashing
/// of an N-gram suffix.
///
/// This is the modelless "conditional memory" primitive — complementary
/// to conditional computation (Raven slot routing, MoE expert routing).
/// Where those activate PARAMETERS per token, this activates LOOKUP SLOTS
/// per token. Hybrid (some of each) is the paper's U-shape optimum.
///
/// Inspired by Engram (Cheng et al. 2026, arXiv:2601.07372). Distilled to
/// the inference-time primitive: hash → table → sigmoid fuse. No training,
/// no backprop. The table is populated offline and frozen; updates are
/// atomic Arc swaps via EngramHotSwap.
pub trait EngramTable {
    /// Dimensionality of each pattern embedding.
    const D: usize;

    /// Lookup `K` pattern embeddings for the given multi-head hash keys.
    ///
    /// Writes the K retrieved embeddings into `out` (shape `[K, D]`).
    /// Returns the number of heads that hit (some hashes may collide to
    /// empty slots in a sparsely-populated table).
    ///
    /// Zero-allocation: caller provides the output buffer.
    fn lookup_into(&self, hash_keys: &[EngramHash; K_MAX], out: &mut [f32]);

    /// BLAKE3 commitment of the table contents — content-addressed identity.
    ///
    /// Two tables with the same `commitment()` provably have identical
    /// contents. Used for sync-boundary identity (the table itself is NOT
    /// synced; only its commitment crosses the wire).
    fn commitment(&self) -> [u8; 32];

    /// Number of slots in the table (for diagnostics / cache sizing).
    fn num_slots(&self) -> usize;
}

/// A deterministic multi-head hash of an N-gram suffix.
///
/// Prime-table multiplicative-XOR (paper §2.2). Each `(n, k)` pair has
/// its own prime modulus `M_{n,k}` and its own seed. The same input
/// always produces the same `EngramHash` — this is what enables async
/// prefetch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct EngramHash(pub u64);

/// Compute the K multi-head hashes for an N-gram suffix.
///
/// `suffix` is the last N canonical-form token IDs (after tokenizer
/// compression). `heads` specifies `(n, k, M_{n,k}, seed)` tuples.
///
/// O(K · N) — a few multiplications and XORs per head. SIMD-able.
pub fn multi_head_hash(
    suffix: &[CanonicalId],
    heads: &[HashHead; K_MAX],
) -> [EngramHash; K_MAX];

/// Tokenizer compression: surjective `V → V'` via NFKC + lowercase.
///
/// Collapses semantically-equivalent token IDs (e.g. "Apple" vs "␣apple")
/// to a single canonical ID. Paper achieves 23% reduction on 128k vocab.
#[inline]
pub fn compress_token(raw_id: TokenId, projection: &SurjectiveMap) -> CanonicalId;

/// Sigmoid-fusion kernel: gate a retrieved pattern vector by current query.
///
/// `output = σ(RMSNorm(q) · RMSNorm(k) / √d) · v`
///
/// Per AGENTS.md: SIGMOID, never softmax. The projection must preserve
/// ranking (cosine-similarity-equivalent). `τ = √d` matches the paper;
/// callers may override via `SigmoidFusionConfig`.
///
/// Zero-allocation. SIMD-accelerated when `D` is a multiple of 8.
pub fn sigmoid_fuse_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    config: &SigmoidFusionConfig,
);
```

### 2.2 The "no prior art" check (vocabulary translation)

Paper vocabulary → codebase vocabulary, with the **shipping cousin** (or none) for each:

| Paper term | Codebase equivalent | Shipped? |
|---|---|---|
| N-gram embedding / hash embedding | `SequenceConstraint { first, second, third: Option }` (`crates/katgpt-pruners/src/constraint_miner.rs`, P196) | ✅ Shipped — but as a **ConstraintPruner guardrail**, not as memory embeddings. Different substrate. |
| Conditional memory | `KgEmbedding` octree (P221, P253) | ✅ Shipped — but spatial traversal, not N-gram-suffix-keyed. Different addressing. |
| O(1) routing slot memory | `RavenKVCache` (R006, P020) | ✅ Shipped but parked — dynamic routing slots, not deterministic hash lookup. Same "O(1) memory" axis but **opposite addressing mechanism**. |
| Context-aware sigmoid gating | `PersonalityWeightedComposition` (R276), `SenseModule::project()`, `evolve_hla` | ✅ Shipped — same sigmoid×direction kernel, different vector source (recurrent state vs static table). |
| Tokenizer compression (surjective V→V') | `ConvexTok` LP optimizer (R087, P127) | ✅ Shipped but for vocab optimization, not for embedding-table dedup. Adjacent. |
| Multi-level cache hierarchy (Zipfian) | Four-Tier Memory (R007), Lore ContentStore (R262), FlashMemory (R258) | ✅ Shipped but for assets / KV cache, not for pattern embeddings. |
| Deterministic addressing (hash → committed value) | `riir-chain/src/forensic/recipe.rs` BLAKE3→codeword→indices (R268, P293) | ✅ Shipped but for asset watermarking, not embedding retrieval. Same primitive, different domain. |
| Frozen snapshot + atomic Arc swap | `SenseHotSwap` (`riir-ai/crates/riir-engine/src/sense/hotswap.rs`, AtomicPtr<Box<SenseModule>>) | ✅ Shipped — same primitive, will be reused for `EngramHotSwap`. |
| BLAKE3 commitment of table identity | `MerkleOctree`, `BlobId`, `EngramTableId` (this plan) | ✅ Infrastructure exists (R221, R253, R262). |

**The gap**: no single primitive ships the *combination* — N-gram-suffix hash → embedding table → sigmoid fusion → residual-fuse into hidden state. Each component has a cousin; the composition is novel. This is the Super-GOAT claim: **the first conditional-memory axis** in our stack, distinct from Raven's conditional-computation axis.

### 2.3 Why this is Super-GOAT (not just GOAT)

GOAT-tier framings (each useful, each already-half-shipped):
- "Hash-keyed KV cache" → just an optimization on Raven. **GOAT, not Super-GOAT.**
- "Static lookup table for NPC chatter" → just a perf saving. **GOAT, not Super-GOAT.**
- "Adapter routing alternative" → no, Engram is *not* routing; it's *retrieval*. **GOAT, not Super-GOAT.**

Super-GOAT framing (the latent-space reframing, per SKILL fusion protocol step 3):
- **Latent-to-latent**: the retrieved `e_t` is a latent vector; the sigmoid gate is a dot-product projection; the residual fuse is a latent addition. The whole mechanism operates in latent space — never decodes to tokens.
- **Per-NPC substrate**: each NPC can have its own `EngramTable` (frozen per-archetype, hot-swappable per-instance). The table is the NPC's *static pattern memory*; the HLA state is its *dynamic belief state*. They compose.
- **Chain-committable**: hash addresses are deterministic raw values → BLAKE3-hashable → LatCal-fixed-representable → quorum-validatable. This is the first conditional-memory primitive that crosses the sync boundary as raw committed values, with the embeddings themselves staying local as latent state.
- **Force multiplier** (≥6 pillars): SenseModule, Raven, dMoE, Forensic Fingerprinting, Four-Tier Memory, NPC Dialog Engine, Freeze/Thaw HotSwap, Latent Functor Re-estimation.

### 2.4 Fusion — what novel combination does this enable?

Per SKILL §1 fusion protocol — fuse this paper with the 2-3 closest cousins:

**Fusion A: Engram × Raven × LatCal commitment** (the headline Super-GOAT)
- *Raven* = conditional computation (dynamic slot routing via hidden state)
- *Engram* = conditional memory (deterministic hash lookup via N-gram suffix)
- *LatCal* = chain commitment of hash addresses (raw, deterministic, quorum-validatable)
- *Novel combination*: a hybrid inference stack where **dynamic routing** handles novel context (Raven) and **static lookup** handles stereotyped patterns (Engram), with the static-lookup addresses being **chain-committable** (LatCal). No competitor has all three. → riir-ai selling-point guide (R147).

**Fusion B: Engram × PersonalityWeightedComposition × MicroRecurrentBeliefState** (per-NPC personality)
- *PersonalityWeightedComposition* (R276) = `behavior = Σᵢ sigmoid(wᵢ/τ) · belief_confidence_i · dᵢ`
- *Engram* = `output = σ(q · k / τ) · v` (same kernel, different vector source)
- *MicroRecurrentBeliefState* (P276) = the HLA carry between ticks
- *Novel combination*: NPC's personality weights gate WHICH pattern memories get retrieved; the recurrent belief state provides the query `q`; Engram supplies the static pattern vectors `e_t`. Personality + memory + belief = emergent NPC voice without per-NPC training. → riir-ai integration.

**Fusion C: Engram × Forensic Asset Fingerprinting × Lore ContentStore** (chain-committable pattern-memory distribution)
- *Forensic Fingerprinting* (R268) = BLAKE3-seeded per-recipient codewords → indices
- *Lore ContentStore* (R262) = chunked content-addressed blob storage with Merkle dedup
- *Engram* = multi-head hash → embedding table
- *Novel combination*: distribute per-NPC Engram tables via Lore ContentStore (chunked, dedup'd, Merkle-committed); sign each table's identity with the Forensic Fingerprinting recipe pattern (BLAKE3-seeded per-recipient); validate at quorum via LatCal-fixed commitment of the hash addresses. The result is **per-NPC pattern memory that is tamper-evident, deduplicated across NPCs of the same archetype, and chain-verifiable**. → riir-chain integration.

---

## 3. Verdict

**Super-GOAT (open primitive half).**

| Tier criteria | Assessment |
|---|---|
| Novel mechanism (no prior art) | ✅ No shipped primitive composes N-gram-suffix-hash → table → sigmoid-fuse → residual. Closest cousins (Raven, SequenceConstraint, KgEmbedding) cover individual pieces; the composition is novel. |
| New capability class | ✅ First *conditional memory* axis — complementary to existing conditional computation (Raven/dMoE/polytope). The U-shape scaling law is the proof this is a *distinct axis*, not a faster version of the same axis. |
| Product selling point | ✅ (in private guide R147) "Chain-committable O(1) knowledge lookup for NPC pattern memory" — no competitor has this. |
| Force multiplier (≥2 pillars) | ✅ Connects to SenseModule, Raven, dMoE, Forensic Fingerprinting, Four-Tier Memory, Freeze/Thaw, NPC Dialog, Latent Functor Re-estimation, LatCal commitment. ≥6 pillars. |

**Mandatory outputs (per SKILL §1.5):**
1. ✅ Open primitive → `katgpt-rs/.research/278_*.md` (this doc) + `katgpt-rs/.plans/299_*.md`
2. ✅ Private guide → `riir-ai/.research/147_Engram_Conditional_Memory_NPC_Guide.md`
3. ✅ Plan → `katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md`
4. ✅ Chain half → `riir-chain/.research/007_Engram_LatCal_Commitment_Bridge.md` (filed 2026-07-04, tracked in R147 §9)

**Reasoning:** The paper introduces a *new sparsity axis* (conditional memory), not a faster version of an existing axis. Our stack already has the complementary axis (Raven = conditional computation); adding the missing axis creates a hybrid that the paper proves is strictly better than either alone. The chain-committable angle (LatCal-fixed hash addresses) is a *novel capability class* — no shipped primitive crosses the sync boundary as raw committed values for embedding lookups. This is the moat.

---

## 4. What This Is NOT

To prevent overclaiming and scope creep:

- **NOT a replacement for Raven** — they're complementary. Raven handles dynamic routing; Engram handles static lookup. The U-shape scaling law says hybrid wins.
- **NOT training** — U-shaped scaling law is a training finding (`→ riir-train`). We only ship the inference-time primitive.
- **NOT a KV cache** — Engram retrieves *static pattern vectors*, not per-sequence KV pairs. It does not grow with sequence length.
- **NOT a recommender / RAG system** — though the math generalizes. The trait is generic; specific applications (NPC dialog, code completion, item recommendation) are caller concerns.
- **NOT chain code** — the open primitive is just a hash table + sigmoid fusion. The chain commitment half (LatCal-fixed bridge, quorum validation) is `→ riir-chain` private IP.
- **NOT a new attention mechanism** — Engram does not replace attention. It's a *residual addition* to the hidden state, before attention. Attention handles dynamic context; Engram offloads static patterns.

---

## 5. Cross-references

- **Paper:** [arXiv:2601.07372](https://arxiv.org/pdf/2601.07372) — Engram, Cheng et al. 2026.
- **Private selling-point guide:** `riir-ai/.research/147_Engram_Conditional_Memory_NPC_Guide.md`
- **Open plan:** `katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md`
- **Complementary axis (conditional computation):** `katgpt-rs/.research/006_Raven_Routing_Slot_Memories.md`
- **Spatial lookup cousin:** `katgpt-rs/.plans/221_kg_latent_octree_sense_composition.md`
- **BLAKE3-seeded deterministic addressing cousin:** `katgpt-rs/.research/268_Forensic_Asset_Fingerprinting_LatCal_Recipe.md`
- **Chunked content-addressed store cousin:** `katgpt-rs/.research/262_Lore_Chunked_Asset_Merkle_Store_Modelless.md`
- **Same sigmoid×direction kernel, different source:** `katgpt-rs/.research/276_Personality_Weighted_Latent_Layer_Composition.md`
- **LatCal fixed-point bridge (sync-boundary substrate):** `riir-chain/src/encoding/latcal_fixed.rs`

---

## TL;DR

**Engram = Super-GOAT (open half).** Conditional memory is a *new sparsity axis* complementing conditional computation. The open primitive is a hash-addressed, sigmoid-fused, frozen-snapshot static memory table — O(1) lookup, deterministic addressing (enables async prefetch), Zipfian cache hierarchy. No prior art in the four repos for the *composition* (cousins exist for each component). Private selling-point guide (`riir-ai/.research/147`) frames the NPC-pattern-memory moat; chain half (`riir-chain/.research/001`, TODO) frames the LatCal commitment. Plan: `katgpt-rs/.plans/299`.
