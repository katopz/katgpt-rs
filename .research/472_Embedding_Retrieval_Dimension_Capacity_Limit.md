# Research 472: Theoretical Limitations of Embedding-Based Retrieval (sign-rank capacity bound)

> **Source:** "On the Theoretical Limitations of Embedding-Based Retrieval" — [arXiv:2508.21038](https://arxiv.org/abs/2508.21038) v2 (Orion Weller, Michael Boratko, Iftekhar Naim, Jinhyuk Lee — Google DeepMind). Accepted ICLR 2026.
> **Date:** 2026-08-10
> **Status:** Active
> **Related Research:** 123 (TopK Dimensionality Barrier — the *positive* counterpart), 045 (MaxSim late interaction), 143 (Latent Terms / SAE / BM25)
> **Related Plans:** 157 (sigmoid margin loss + `dim_sufficiency_bound`), 080 (MaxSim), 410 (Linking-Fold — the precedent for absorbing an impossibility theorem)
> **Cross-ref (riir-neuron-db / riir-ai):** neuron-db Plan 324 (`DenseEmbedIndex`), Plan 325 (`KgTripleIndex`), Plan 323 (`Bm25Index`); riir-ai Plan 524/526 (`riir-rag`)
> **Classification:** Public
> **Verdict:** **Gain**

---

## TL;DR

The paper proves a *negative, worst-case* capacity theorem for single-vector retrieval: for embedding dimension `d`, the number of top-`k` document subsets realizable by **any** query is bounded, so there always exist relevance structures no `d`-dimensional single-vector index can represent — regardless of model size, training data, or loss. This is the exact complement to Research 123, which proved the *positive, k-sparse-conditional* bound `d = Θ(k log n)`.

**Why it matters here:** our entire default-on retrieval surface is single-vector cosine top-k at **`d = 8`**. Applying the paper's own Theorem 1 at γ=0.1, `d=8` supports all top-`k` subsets only up to `n ≤ 20,706` for `k=2`, `n ≤ 269` for `k=4`, and `n ≤ 44` for `k=8` — and the minimum over *all* `k` is just **30 documents** (§2.2b). `ItemEmbedIndex` runs `k=5` retrieval over the real **25,943-item** live-game catalogue at `d=8`, where the ceiling is 122 items and the bound demands `d ≥ 19.2` — **213× over on corpus, 2.4× under on dimension**.

**Distilled for katgpt-rs (modelless, inference-time):** the transferable artifact is not a new mechanism — it is a *closed-form capacity budget* `d ≥ ln C(n,k) / ln(1+1/γ)`, increasing in `n`, that says a priori whether a retrieval configuration is inside its representable regime. Shipped 2026-08-10 as `dim_capacity_required` / `dim_capacity_ceiling` / `dim_capacity_floor` / `ln_binomial` in `katgpt-types/src/simd/research.rs` (feature `sigmoid_margin`), with tests reproducing the paper's Table 1 cell-for-cell. Complements — does not replace — Research 123's positive `dim_sufficiency_bound`, which still has zero production call sites.

---

## 1. Paper Core Findings

### 1.1 Theorem 1 (dimension lower bound)

Unit document vectors `v_i ∈ R^d`, unit queries `u ∈ R^d`. A `k`-subset `S ⊆ [n]` is *realized with margin γ* if some unit `u_S` satisfies

```
min_{i∈S} ⟨u_S, v_i⟩  ≥  max_{j∉S} ⟨u_S, v_j⟩ + 2γ
```

If **every** `k`-subset is realized with margin γ, then

```
C(n,k) ≤ (1 + 1/γ)^d        hence        d ≥ ln C(n,k) / ln(1 + 1/γ)
```

*Proof shape:* any two distinct realized subsets force `‖u_S − u_T‖ ≥ 2γ`, so the `C(n,k)` query vectors are pairwise `2γ`-separated; disjoint γ-balls inside `B(0, 1+γ)` give a volume packing argument. For `n ≫ k`, `ln C(n,k) ≈ k·ln(en/k)`, so `d = Ω(k·log(en/k) / log(1+1/γ))`.

**Reproduced independently** — the formula regenerates the paper's Table 1 exactly (γ=0.1: `n=10⁴` → `k=2`:8, `k=10`:33, `k=100`:233, `k=1000`:1354).

### 1.2 Empirical best case (free embeddings)

Vectors optimized directly by Adam against the **test** qrel matrix with InfoNCE (full-batch, projected-gradient unit norm) — i.e. no generalization requirement, no natural-language constraint. This is an upper bound on *any* embedding model. Increasing `n` until 100% accuracy breaks gives the *critical-n*:

```
critical_n(d) = −10.5322 + 4.0309·d + 0.0520·d² + 0.0037·d³        (r² = 0.999, k=2)
```

Paper's extrapolations: 500k (d=512), 1.7m (768), 4m (1024), 107m (3072), 250m (4096).

**Crucially the theory is a gross underestimate of practice:** at `n=100` Theorem 1 floors `d=4`, but free embeddings need `d>18` — a **4.5× multiplier** even in the no-generalization case. Real models are worse still: free embeddings solve LIMIT-small in 12 dims, yet "real models with 64 dimensions still cannot completely solve the task."

### 1.3 LIMIT (Linguistically Simple, Geometrically Impossible Task)

46 relevant docs + 49.95k irrelevant (plus a "small" 46-doc variant), 1000 queries, `k=2`, chosen because `C(46,2)=1035` is the smallest above 1k. Random natural-language attributes assigned to a dense qrel matrix. Recall@2 on LIMIT-small:

| Model | Recall@2 | Synonym variant | Δ |
|---|---|---|---|
| BM25 (lexical) | **97.8** | 10.6 | **−89.2%** |
| GTE-ModernColBERT (multi-vector) | 83.5 | 25.6 | −69.3% |
| Promptriever Llama3 8B | 54.3 | 12.8 | −76.4% |
| GritLM 7B | 38.4 | 14.3 | −62.8% |
| E5-Mistral 7B | 29.5 | 15.1 | −48.8% |
| Snowflake Arctic L | 19.4 | 8.5 | −56.2% |
| Qwen3 Embed | 19.0 | 11.6 | −38.9% |

Gemini-2.5-Pro as a long-context **reranker** solves **100%** of all 1000 queries in one forward pass.

Training on an in-domain LIMIT train split moves recall@10 from ~0 to 2.8 — **not domain shift**. Training on the *test* split works (overfits tokens), matching the free-embedding result.

### 1.4 Formal machinery (Appendix D)

- `rank_rop(A)` (row-wise order-preserving rank) = smallest `d` with a rank-`d` `B` preserving each row's ordering.
- **Prop 1:** for binary `A`, `rank_rop A = rank_rt A` (row-wise thresholdable rank).
- **Prop 2:** `rank_±(2A−1) − 1 ≤ rank_rop A = rank_rt A ≤ rank_gt A ≤ rank_±(2A−1)`.

So the minimum viable embedding dimension is pinned between sign-rank − 1 and sign-rank of the signed qrel matrix.

### 1.5 The paper's own scope limits (load-bearing for us)

- Does **not** extend the theory to multi-vector architectures.
- Does **not** bound the *approximate* case ("capture only the majority of combinations") — points at Ben-David et al. 2002.
- Cannot say *a priori which* combinations a given model fails on: "we do know that there exists some tasks that they will never be able to solve."

**This is why the bound is a headroom/robustness statement, not a refutation of any passing benchmark.** A benign, low-rank realized qrel matrix can be served perfectly at `d=8`; the theorem only says the *worst case* over relevance structures is unreachable.

### 1.6 Two citations that land on repo rules

- **Appendix C:** per Bangachev et al. 2025 ([2509.18552](https://arxiv.org/abs/2509.18552)), sigmoid-loss free embeddings solve in **fewer** dimensions than InfoNCE (assuming no margin). Mild support for the repo's sigmoid mandate and consistent with Research 123's sigmoid-vs-InfoNCE data.
- **But** Grivas, Vergari & Lopez, AAAI 2024 ([2310.10443](https://arxiv.org/abs/2310.10443)) prove a *sigmoid-specific* bottleneck: a low-rank output layer with sigmoid makes **exponentially many label combinations unargmaxable** — unpredictable for any input. Their fix is a **DFT output layer** guaranteeing all ≤`k`-sparse combinations are argmaxable, at 50% fewer parameters and equal F1@k. Zero prior art for this in any of the 7 repos.

---

## 2. Distillation — what this means for our stack

### 2.1 Our dimensions, source-verified

| Constant | Value | Location | Default-on? |
|---|---|---|---|
| `BELIEF_DIM` (`hla_moments`, the indexed vector) | **8** | `riir-neuron-db/src/shard/mod.rs:16`, `:225` | always |
| `ITEM_EMBED_DIM` | **8** | `riir-neuron-db/src/item_index.rs:54` | yes (`item_embed_index`) |
| `LATENT_DIM` (riir-rag query) | **8** | `riir-ai/crates/riir-rag/src/query.rs:11` | yes |
| `EMBED_DIM` (vessel, experience graph) | **8** | `neuron_vessel.rs:55`, `experience_graph/node.rs:35` | yes |
| `BEHAVIOR_EMBED_DIM` (crowd_mcgs RRF) | 64 | `riir-games/src/crowd_mcgs/types.rs:46` | — |
| `STYLE_DIM` (payload, **never** a similarity key) | 64 | `shard/mod.rs:13` | — |
| `DENSE_EMBED_DIM` | 768 | `riir-neuron-db/src/dense_embed/mod.rs:40` | **no — no producer** |

`style_weights[64]` is JL-projected to `ShardEmbedding[8]` before use; the hot retrieval path is documented in-repo as a **"lossy 64→8"** ranking.

### 2.2 Theorem 1 applied at γ=0.1 — max corpus `n` with all top-`k` subsets realizable

| d | k=2 | k=4 | k=5 | k=8 | k=10 | k=16 | **min over all k** |
|---|---|---|---|---|---|---|---|
| **8** | 20,706 | **269** | **122** | **44** | **35** | **30** | **30** |
| 16 | 303,149,237 | 32,407 | 5,603 | 458 | 214 | 82 | 58 |
| 32 | 6.50e16 | 474,454,197 | 12,043,410 | 55,117 | 9,741 | 830 | 114 |
| 64 | >2^62 | 1.02e17 | 5.57e13 | 806,921,985 | 20,935,801 | 99,573 | 225 |
| 768 | >2^62 | >2^62 | >2^62 | >2^62 | >2^62 | >2^62 | 2,662 |

> **Corrected 2026-08-10.** An earlier revision of this table carried
> coarse-search-step underestimates in the k=2/k=4/k=16 columns (e.g. 20,482 for
> d=8/k=2 instead of 20,706). All values above are now produced by
> `dim_capacity_ceiling` and pinned by tests that additionally reproduce the
> paper's Table 1 cell-for-cell. The two load-bearing figures were unaffected:
> d=8/k=8 → 44 and the ItemEmbedIndex d≥19.2 requirement.

### 2.2b The k-free statement — `min ceiling ≈ d · log₂(1 + 1/γ)`

The per-`k` table above hides a much simpler result. `dim_capacity_ceiling` is
**U-shaped in `k`, not monotone**: it falls until roughly `k ≈ n/2`, then rises
again because `C(n,k) = C(n,n−k)` makes near-complete subsets easy ("retrieve
almost everything" needs little separating power). The minimum of that curve is
the only `k`-free thing you can say about a dimension, and it has a one-line
closed form — at `k ≈ n/2`, `C(n,n/2) ≈ 2ⁿ`, so Theorem 1 becomes

```
n · ln 2 ≤ d · ln(1 + 1/γ)     ⇒     n ≤ d · log₂(1 + 1/γ)
```

At γ=0.1 that is **≈ 3.46·d** — the capacity floor is *linear* in the embedding
dimension, not exponential. Concretely: **a `d=8` index can never represent all
top-`k` subsets of more than 30 documents, for any `k` whatsoever.** Shipped as
`dim_capacity_floor(d, γ)` (O(1); a few documents conservative vs the exact
minimum, since it drops Stirling's `√(πn/2)`).

This is the number to quote when provisioning, because it needs no assumption
about `k`. Our 8-D surfaces sit **33×** (1000-node graphs) to **865×**
(the 25,943-item catalogue) past it.

Free-embedding best case (`critical_n`, k=2), with the paper's measured 4.5× real-model multiplier:

| d | critical-n | ÷4.5 |
|---|---|---|
| **8** | **27** | **6** |
| 16 | 82 | 18 |
| 32 | 293 | 65 |
| 64 | 1,430 | 318 |
| 768 | 1,709,800 | 379,956 |

### 2.3 The single most exposed shipped artifact

`ItemEmbedIndex` — `d=8`, real catalogue `n=25,943` live-game items, and its GOAT gate is a **k=5** task ("10/10 type-centroid queries return ≥3/5 same-type"):

| k | required d (Thm 1, γ=0.1) | have | status |
|---|---|---|---|
| 1 | 4.24 | 8 | OK |
| 2 | 8.19 | 8 | at the line |
| 3 | 11.97 | 8 | 1.5× under |
| 4 | 15.63 | 8 | 2.0× under |
| **5** | **19.20** | **8** | **2.4× under** |

Read the other way round: at `d=8, k=5` the capacity ceiling is **122 items**,
against a real catalogue of 25,943 — **213× over** (and 865× over the k-free
floor of 30). The `d` view understates it because the bound is logarithmic in
`n`: a 2.4× dimension shortfall is a two-order-of-magnitude corpus shortfall.

The passing GOAT gate is not invalidated — it measures the *realized* (benign, type-clustered, centroid-initialised) qrel matrix. The bound says there is **no worst-case headroom left**: any move toward compositional/multi-attribute item relevance ("light armour that a mage can equip below level 30 excluding quest rewards") is combinatorially richer and cannot be fixed by better embeddings at `d=8`.

### 2.4 Escapes: the notes overstate what is wired

The code layer contradicts the notes layer on several points. Verified:

| Escape | Notes claim | Code reality |
|---|---|---|
| Multi-vector / MaxSim | "ships, default-on" | `maxsim_score` ships as a SIMD kernel (`katgpt-types/src/simd/maxsim.rs:51`) but **no index stores >1 vector per document**. `hla_moments`, `DenseEmbedShard.embedding`, `ItemEntry.embedding`, `SupportPoint.embedding` are all single vectors. |
| 768-D two-stage rerank | "opt-in, G1–G4 PASS" | `dense_embed = []` not in `default`; **no embedding producer** — blocked on the Plan 318 C13 Kimi-K3 checkpoint. Code-complete, unreachable. |
| GraphRAG / KG structural | "G5 PASS, ships" | `kg_triple` not default; `graph_rag` is **commented out** of `riir-rag/Cargo.toml` (kg_triple not pushed to origin/develop). |
| AnyRAG escalation | "the IP is *when* to escalate" | `gateway.rs:66` `request_ruling` **ignores its `_conflict` argument entirely**, returns an empty ruling gated only on `Option::is_some(endpoint)`. Stub behaviour pinned by test at `:130`. No retrieval, no escalation decision. |
| KG triples from latent similarity | "emitted at cosine threshold" | `vibe.rs` emits **none**; it only declares `KgTripleTemplate` (`:37`). Its `0.7`/`0.0` thresholds gate `sin(tick·ω)` — a clock function, not a cosine. |
| BM25 lexical | opt-in | **Genuinely ships** via riir-rag default (`bm25.rs:431,444`, real idf + length norm). |
| RRF hybrid | — | **Genuinely ships**, but only in `crowd_mcgs/retrieval.rs:666` at **64-D**, not on the 8-D RAG path. |

Also: `ShardIndex::query` (`index/mod.rs:257`) is **weaker than true cosine top-1** — it binary-searches on `embedding[0]` and scores only 3 candidates, so a shard with high 8-D cosine but distant first coordinate is structurally unreachable. Two in-repo tests already document this (`tests/hebbian_bridge_t44_compat.rs:197,212` — "similarity ANN is SEMANTICALLY DEGENERATE (by design)").

### 2.5 A measurement that must not be misread

`fast_knn`'s "recall@k = 100% within ε=1e-4" and `DenseEmbedIndex`'s "recall@10 = 100%" are **fidelity to the cosine ranking**, i.e. the fast path reproduces brute-force cosine exactly. They say nothing about whether the cosine ranking is the *correct* ranking. The paper attacks the expressivity of the cosine ranking itself, so these numbers confer **no immunity** to the bound and must not be cited as if they did.

### 2.6 What the stack already got right, independently

Two shipped, GOAT-gated mechanisms are the same directions the paper points, arrived at empirically:

- `diverse_retrieval` (**default-on**) — greedy max-wedge-span selection, justified in-repo because cosine top-k "clusters around one mode"; G5 = **3.31× intrinsic_dim vs cosine-top-k**. `tests/g5_compaction_quality.rs:181` even builds `cosine_top_k` as the explicit baseline, commented "the honest 'what pure cosine retrieval would select'."
- `smooth_min_similarity` (**default-on**) — multi-token aggregation; `RerankMethod::SmoothMinAligned` recall@5 **1.000 vs cosine 0.495 (+50.5pp)**.

And BM25's synonym collapse (−89%) independently validates riir-rag's design rule that BM25 is "the fallback for exact symbol matching, never the primary path."

### 2.7 Fusion

**Paper × Research 123 × Plan 157 × Plan 410 (Linking-Fold).** Research 123 gives the positive bound `d = Θ(k log n)` and Plan 157 already ships `dim_sufficiency_bound(k, n)` — GOAT-proven, default-on, and **called from nothing but its own tests** (`katgpt-types/src/simd/research.rs:176`; every call site in `simd/tests.rs`). This paper supplies the *worst-case* half of the same instrument plus an adversarial construction protocol (LIMIT) to measure it.

Plan 410 is the precedent for the correct shape of response: a published impossibility theorem (monotonic activations preserve linking number, hence cannot separate linked manifolds — and it explicitly indicted `ItemEmbedIndex` cosine retrieval) was absorbed as **(a) a diagnostic + (b) a closed-form modelless correction**, not a rewrite. The same shape applies here: wire the capacity diagnostic, build the adversarial fixture, and promote the already-built escapes for the configurations where the bound bites.

---

## 3. Verdict

### **Gain**

**One-line reasoning:** the paper's mechanism is not novel to us — the capacity-bound instrument already ships (`dim_sufficiency_bound`, Plan 157) and all four escapes are known art — but it produces concrete, quantified, actionable findings against shipped default-on code, which is Gain, not Pass, per skill §1.55.

**Novelty gate (§1.5) — fails Q1, so not Super-GOAT:**

- **Q1 no prior art? NO.** Research 123 covers the dimensionality barrier from the positive side; `dim_sufficiency_bound` ships. Published prior art also covers every escape I would have proposed: [Scaling Laws for Embedding Dimension in IR](https://arxiv.org/pdf/2602.05062) (dimension→performance power law, Feb 2026), [Arabzadeh et al. 2021](https://arxiv.org/pdf/2109.10739) (per-query dense-vs-sparse selector), MUVERA (multi-vector→single-vector MIPS via FDE), [Col-Bandit](https://arxiv.org/pdf/2602.02827), and Grivas et al. (DFT output layer). A capacity-driven escalation gate is a narrow gap — neither prior-art paper conditions on corpus size *and* drives a runtime decision — but a guard rail is not a new capability class.
- **Q2 new class of behavior? NO.** It constrains and instruments existing behavior.
- **Q3 product selling point? NO.** "Our retrieval knows when it is out of capacity" is a robustness property, not a differentiator.
- **Q4 force multiplier? Partially** — touches Pillar 2 (neuron-db shard lookup) and Pillar 3 (NPC dialog RAG), but via Q1/Q2/Q3 failure this stays Gain.

**MOAT gate (§1.6):** in scope for `katgpt-rs` (the bound is generic retrieval math with no game semantics; `dim_sufficiency_bound` already lives in `katgpt-types`). The 8-D audit findings and any promotion decisions are private and belong to `riir-neuron-db` / `riir-ai`. Verdict: **neutral Gain** — do not overclaim moat.

**Grandfather note (UQ rule):** this primitive claims no probability distribution, interval, or coverage guarantee, so the "Report the Floor" conformal-naive gate does not apply.

### Applied downstream (2026-08-10)

**riir-neuron-db `experience_graph`** (DEFAULT-ON since 2026-07-16, Plan 319) — aligned in
`riir-neuron-db/.proposals/001` §"Capacity + regime addendum" + **Issue 591**. `latent_seed_top_k`
(`src/experience_graph/graph.rs:196`) is flat brute-force 8-D cosine top-k over `task_embedding[8]`;
Benchmark 319 G2b runs `seed_k=8` over 1000 nodes, **~23× past** the `n≤44` ceiling Theorem 1 gives at
`d=8, k=8, γ=0.1`. Three transferable results came out of that alignment:

1. **The bound falls as `k` grows, throughout the practical regime `k ≪ n`** — 269 nodes at `k=4`,
   44 at `k=8`, 30 at `k=16`. So the reflexive fix for a weak seed stage (raise `seed_k`) makes the
   worst-case requirement *worse*. **Coverage cannot be bought with a larger `k`; the escape is
   structural or dimensional.** This generalizes to every top-k consumer in the stack. (The curve is
   U-shaped, not monotone — it rises again once `k` approaches `n`, which is the degenerate
   "retrieve almost everything" branch and must not be cited as headroom. See §2.2b.)
2. **Seed capacity upper-bounds downstream recall** in any seed→expand retriever: expansion amplifies
   whichever region the seed lands in, so depth does not rescue a seed set in the wrong neighborhood.
3. **`ExperienceNode` is the stack's one natural multi-vector document** — node + `sibling_hashes[8]` =
   9 `task_embedding[8]` vectors per experience region, so MaxSim late interaction is available at zero
   new storage. This is the cheapest path to the multi-vector escape §2.4 found missing everywhere.

Also corrected a stale claim in shipped code: `graph.rs` asserted a *"100× gap that is a regime
boundary"* for NS traversal (5–10 ms PoC) when Benchmark 319 measured **0.065 ms** — ~1.3× over the
online budget, refuted by the benchmark that promoted the feature. Re-justified on scaling grounds.

### Actionable follow-ups (issues, not plans — per AGENTS.md)

- **Issue 579** — wire `dim_sufficiency_bound` into the retrieval paths (it has zero production call sites) and extend Research 123's sufficiency table to the default-on 8-D indexes it omits.
- **Issue 580** — LIMIT-style adversarial retrieval fixture; measures the recall ceiling `dense_embed/mod.rs` explicitly defers ("recall ceiling by 8-D stage-1").
- **Issue 581** — sigmoid argmaxability audit (Grivas DFT bottleneck) on our low-rank sigmoid projections.

### Not actionable / deliberately not proposed

- **Raising `BELIEF_DIM` from 8.** `NeuronShard` is a frozen `#[repr(C)]` Pod at ~368 bytes with BLAKE3 commitments, Lean-proofed offsets, and chain-committed layout. Changing the indexed dimension is a sync-boundary and proof-invariant change, categorically out of scope for a research follow-up.
- **Adding a cross-encoder.** The paper shows a reranker solves LIMIT at 100% where every bi-encoder fails, and we have none — our reranks are bi-encoder cosine at higher `d`, which raises the ceiling without escaping the bound. But a cross-encoder is a per-candidate joint forward pass, incompatible with the plasma/hot latency budget and with the modelless mandate. Recorded as a known structural gap only.

---

## 4. Cross-Reference

- Positive counterpart: `katgpt-rs/.research/123_TopK_Dimensionality_Barrier_Retrieval.md` (arXiv 2605.23556)
- Capacity instrument: `katgpt-rs/.plans/157_sigmoid_margin_loss.md`, `katgpt-rs/.benchmarks/048_sigmoid_margin_goat.md`, `katgpt-types/src/simd/research.rs:176`
- Impossibility-theorem precedent: `katgpt-rs/.plans/410_*` (Linking-Fold, Theorem 4.7)
- Multi-vector: `katgpt-rs/.research/045_MaxSim_Memory_Efficient_Late_Interaction_Scoring.md`, Plan 080
- Lexical/hybrid: `katgpt-rs/.research/143_Latent_Terms_SAE_BM25_Retrieval.md`
- neuron-db: Plan 323 (`Bm25Index`), Plan 324 (`DenseEmbedIndex`), Plan 325 (`KgTripleIndex`)
- riir-ai: Plan 524 (`riir-rag` facade), Plan 526 (GraphRAG fusion)
- Sigmoid bottleneck: Grivas et al. AAAI 2024 [2310.10443](https://arxiv.org/abs/2310.10443); Bangachev et al. [2509.18552](https://arxiv.org/abs/2509.18552); KG rank bottleneck [2506.22271](https://arxiv.org/abs/2506.22271)

> **PASS-Redirects (synthesis):** FBDM [arXiv:2609.29350] (2026-09-26 distill) — image-SSL flow matching against an ETF-style reference frame with K′ directions exceeding the flow dimension d* (capacity-limited per-center assignment). PASS — training surface out of scope; the K′>d frame-capacity observation is recorded here as the dimension-capacity class's training-side cousin (our retrieval/routing capacity limits are inference-side); full redirect on Research 453.
