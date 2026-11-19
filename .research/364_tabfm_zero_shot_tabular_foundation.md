# Research 364: TabFM — Zero-Shot Tabular Foundation Model

> **Source:** Kong & Das (Google Research) — "Introducing TabFM: A zero-shot foundation model for tabular data" (Google Research Blog, 2026-06-30). [blog](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/) · [code](https://github.com/google-research/tabfm) · [HF](https://huggingface.co/google/tabfm-v1.0.0)
> **Date:** 2026-07-02
> **Status:** Done — **Pass**. Pre-trained foundation model (training-only contribution); the underlying set-attention architecture is a refinement of NPT (R354, **already distilled as Super-GOAT** with shipped open primitive + private crowd-attention runtime).
> **Delta (2026-09-30):** arXiv technical report released ([2609.37959](https://arxiv.org/abs/2609.37959)) — formalizes the TabArena numbers recorded below; verdict unchanged. The follow-up **TabFM-Auto** ([2609.37989](https://arxiv.org/abs/2609.37989), LLM agent evolving the data pipeline around frozen TabFM) is distilled in **§6 below** (RECORD, owner-deferred; [Research 597](597_TabFM_Auto_Pipeline_Evolution.md) is the redirect alias keeping the number live).
> **Delta (2026-11-19):** **§7 Kumo addendum — PASS** (NVIDIA Kumo Tabular, [2611.23001](https://arxiv.org/abs/2611.23001), the family's current SOTA member; open weights OpenMDW-1.1); verdict unchanged.
> **Related Research:** 354 (**the direct parent** — NPT cross-datapoint set attention, Super-GOAT), 234 (DenseMesh — demoted), 126 (MoA — within-token, not set), 278 (Engram — hash lookup, not set attention), 290 (Latent Field Steering — broadcast, not peer-to-peer), 303 (Transolver FUNCATTN — physics-set predecessor), 309 (ARG latent substrate synthesis)
> **Related Plans:** 354 (`set_sigmoid_attention_into` open primitive — shipped), 355 (riir-ai crowd joint inference runtime — Phase 1 shipped, Phase 2 in progress)
> **Cross-ref (riir-ai private guide):** [Research 167 — Crowd Joint Inference via Cross-NPC Set Attention](../../riir-ai/.research/167_crowd_joint_inference_cross_npc_set_attention_guide.md) — the selling-point half of the Super-GOAT TabFM descends from.
> **Classification:** Public (katgpt-rs)

---

## TL;DR

TabFM is Google's zero-shot foundation model for tabular classification and regression. Users hand it a previously-unseen table (train rows + test rows); TabFM emits predictions in a **single forward pass** with no `.fit()`, no hyperparameter tuning, no feature engineering. On the TabArena benchmark (38 classification + 13 regression datasets, 700–150K samples), TabFM outperforms heavily-tuned XGBoost/RandomForest/CatBoost baselines in Elo.

**Architecture (the only part that matters for distillation):**

1. **Alternating row + column attention** (TabPFN-style) — cross-row attention (Attention Between Datapoints, ABD) interleaved with cross-column attention (Attention Between Attributes, ABA).
2. **Row compression** — each row's cross-attended representation is compressed into a single dense vector.
3. **ICL transformer** (TabICL-style) — attention runs over the *compressed row vectors* rather than the raw grid, drastically reducing compute for large tables.

**Training (the part that does NOT matter for distillation):** trained entirely on hundreds of millions of synthetic datasets generated from structural causal models (SCMs) with random functions. Industrial tabular data is too scarce/proprietary for foundation-model pre-training; synthetic SCMs are the only viable pre-training source.

**Distilled for katgpt-rs (modelless, inference-time):**

TabFM's value is its **pre-trained weights** — the product of GPU-intensive synthetic-data pre-training. The weights are the artifact; the architecture is a refinement of NPT (R354). → riir-train territory (training-method + trained-weight asset).

The architectural insight — **alternating ABD/ABA set attention + row compression + ICL transformer on compressed rows** — is fully covered by the codebase's existing set-attention primitive and crowd-attention runtime. Three of the three architectural ingredients already have shipped analogs:

| TabFM ingredient | Codebase analog | Status |
|---|---|---|
| ABD (cross-row attention) | `set_sigmoid_attention_into` (P354) + `CrowdAttentionStep::tick_into` (crowd_attention.rs) | **Shipped, default-opt-in** (Super-GOAT R354 / R167 / P355 Phase 1) |
| ABA (cross-column attention) | HLA dimension interactions via latent functor direction vectors (R123, P303/317/318); the 8 HLA dims ARE the "columns" | Shipped |
| Row compression → ICL on compressed vectors | HLA 8-dim affective latent IS the row compression; cross-zone attention over per-zone HLA centroids is the natural analog | Partially shipped (per-zone HLA already exists; cross-zone attention is a future fusion) |

The only architectural element TabFM adds beyond what R354 already distilled is the **two-stage "compress then attend" pipeline** (row compression before the ICL transformer). That is a perf optimization (attend over `N` compressed vectors instead of `N×d` raw grid) — useful, but not a new capability class.

---

## 1. Paper Core Findings (from blog + GitHub)

### 1.1 The ICL framing

Tabular ML's traditional bottleneck: `XGBoost.fit()` is not a single step — it requires hyperparameter optimization and domain-specific feature engineering per dataset. TabFM reframes tabular prediction as in-context learning: the entire dataset (train + test rows) is the prompt; the model predicts test targets in a single forward pass with no per-dataset weight updates.

### 1.2 The three-ingredient hybrid architecture

Synthesizes TabPFN and TabICL into a hybrid:

1. **Alternating row + column attention** (from TabPFN). Multi-layer attention alternates across columns (features) and rows (examples). This learns rich representations that natively capture complex feature interactions — "the heavy lifting that would otherwise require tedious manual feature crafting".
2. **Row compression**. After contextualization, each row's cross-attended information is compressed into a single dense vector.
3. **ICL transformer on compressed vectors** (from TabICL). Attention over compressed row vectors (not the raw grid) drastically reduces compute — scales to larger datasets without blowing up the attention matrix.

### 1.3 Synthetic-data pre-training

Industrial tabular data is too scarce and too proprietary for foundation-model pre-training. TabFM is trained on **hundreds of millions of synthetic datasets** generated from structural causal models (SCMs) with random functions. The synthetic diversity captures real-world distributional variety, producing zero-shot generalization to unseen real tables.

### 1.4 Benchmark results

TabArena Elo benchmark (38 classification + 13 regression datasets, 700–150K samples):

- **TabFM** (out-of-the-box, single forward pass) — top-tier Elo, beating heavily-tuned supervised baselines.
- **TabFM-Ensemble** (cross features + SVD features + 32-way ensemble with NNLS weights + Platt scaling for classification) — pushes performance further.

Being integrated into Google BigQuery as `AI.PREDICT` SQL — no ML expertise required.

### 1.5 Engineering surface (from GitHub)

- Scikit-learn compatible (`TabFMClassifier`, `TabFMRegressor`).
- JAX (Flax `nnx`) and PyTorch backends.
- Pre-trained weights v1.0.0 on Hugging Face Hub.
- Supports mixed numerical + categorical features out of the box (ordinal encoders + numerical scalers handled in `.fit()` prep, not in model weights).

---

## 2. Distillation

### 2.1 What's training-only (→ riir-train)

- The pre-trained weights (the actual foundation-model artifact — GPU-intensive synthetic-SCM pre-training).
- The SCM-based synthetic data generation pipeline.
- The 32-way NNLS-weighted ensemble variant (TabFM-Ensemble).
- Platt scaling calibration for classification.
- Any variant that fine-tunes TabFM on a new distribution.

**All of TabFM's empirical wins are gated behind the pre-trained weights.** A reproduction requires GPU-intensive synthetic pre-training at scale. → riir-train one-line note.

### 2.2 What's modelless (stays in katgpt-rs / riir-ai)

The **architecture** is a refinement of NPT (R354, **already distilled as Super-GOAT**). Three of three TabFM ingredients have shipped analogs:

#### 2.2.1 ABD (cross-row attention) → `set_sigmoid_attention_into`

TabFM's ABD is NPT's ABD — attention between datapoints (rows). R354 distilled this exact mechanism into the open primitive `set_sigmoid_attention_into(query, key, value, output, scratch)` with sigmoid gate (never softmax, per AGENTS.md §2). The private runtime half is `CrowdAttentionStep::tick_into` in `riir-ai/crates/riir-engine/src/crowd_attention.rs`, where the "rows" are NPC HLA belief states in a zone and the attention is fog-of-war gated.

**Key architectural difference**: TabFM uses standard softmax multi-head attention; the codebase uses sigmoid-gated attention. Per AGENTS.md §2, sigmoid is mandatory (never softmax) for any latent projection — TabFM's softmax is a paper-side choice we deliberately do not adopt.

#### 2.2.2 ABA (cross-column attention) → HLA dimension interactions via functor directions

TabFM's ABA is attention across columns (features) within a row. The codebase analog: HLA's 8-dim affective latent (valence/arousal/desperation/calm/fear + 3) is the "row"; the 8 dimensions are the "columns". Cross-dimension interaction is handled by **latent functor direction vectors** (R123, P303/317/318) — each functor is a direction in HLA space, and applying a functor is a projection onto that direction. This is the deterministic (non-learned) analog of TabFM's learned ABA.

#### 2.2.3 Row compression → ICL on compressed vectors → HLA centroid + cross-zone attention

This is the one TabFM ingredient that is *partially* novel relative to R354. TabFM compresses each row to a dense vector *before* the ICL transformer, then attends over compressed vectors. The codebase analog:

- **Per-NPC**: the HLA 8-dim affective latent IS the row compression. Each NPC's full belief state collapses to 8 numbers. ✓ shipped.
- **Per-zone**: the per-zone HLA centroid (computed for crowd-MCGS in P298) is the second compression level. ✓ shipped.
- **Cross-zone attention over per-zone centroids**: this is the natural analog of TabFM's "ICL transformer on compressed rows" — *not yet shipped*, but a straightforward extension of `set_sigmoid_attention_into` to zone centroids.

This is the only follow-up worth tracking, and it is a **fusion idea** (not a novel primitive): cross-zone attention over HLA centroids × existing crowd-attention runtime.

### 2.3 Latent-space reframing (mandatory before verdict)

TabFM operates on raw tabular cells (numerical + categorical, ordinal-encoded). Our reframe: operate on **HLA latent states** (per-NPC 8-dim affective).

```text
// TabFM (raw cells, learned Q/K/V, softmax):
X ∈ R^{N × d}                  // N rows × d features (raw cells)
Row i  →  W_embed · row_i       // per-cell embedding
ABD:   α_ij = softmax(q_i · k_j / √d)  // across rows
ABA:   β_imn = softmax(...)             // across columns
Row i compress → z_i ∈ R^h
ICL:   attend over {z_1, ..., z_N}

// Codebase analog (latent HLA, deterministic Q/K/V, sigmoid):
H ∈ R^{N × 8}                  // N NPCs × 8 HLA dims (already compressed)
ABD:   α_ij = σ(q_i · k_j · β_zone)     // q from CS-ranking, k from functor dirs
ABA:   handled by functor projections   // direction vectors span the column space
Per-zone centroid → c_z ∈ R^8
Cross-zone: attend over {c_{z1}, ..., c_{zK}}
```

The latent reframing is **strictly stronger** than TabFM's raw-cell approach for the codebase's domain (NPC crowd cognition):

1. **No per-row embedding**: HLA's 8 dims ARE the embedding. No learned per-feature embedders (which TabFM must train).
2. **Deterministic Q/K/V**: from Mind-Reading CS-rankings (R133 → W_Q) and Latent Functor directions (R123 → W_K). No gradient descent. Freeze/thaw-committed.
3. **Sigmoid gate**: bounded `(0,1)`, never softmax. AGENTS.md §2 mandate.
4. **Fog-of-war compatible**: NPCs only attend over visible peers (the visibility mask is a first-class input to `CrowdAttentionStep::tick_into`). TabFM attends over the whole dataset unconditionally.
5. **Sync boundary clean**: only the 5 emotion scalars cross sync (the unchanged `compute_animal_emotions()` bridge). TabFM has no sync-boundary concept.

**Important honesty caveat (§3.6)**: TabFM's domain is *supervised tabular prediction* (classification/regression on unseen tables). The codebase's domain is *NPC crowd cognition* (per-NPC latent belief refinement). They share an **architecture** (set attention) but solve **different problems**. The latent reframing shows the architecture is covered; it does NOT claim the codebase would beat TabFM on TabArena (we do not run tabular benchmarks and never will — that is not our domain).

### 2.4 Fusion opportunities (the angle worth recording)

The single non-trivial fusion TabFM suggests:

| Fusion | What it gains | Mechanism | Track as |
|---|---|---|---|
| **Cross-Zone Set Attention over HLA Centroids** | A "meta-crowd" layer above the existing per-zone crowd-attention: each zone's centroid HLA state attends over the centroids of neighboring zones, producing a region-scale belief refinement. This is TabFM's "compress then attend over compressed vectors" pattern, applied to zone centroids. | Phase 1: extend `set_sigmoid_attention_into` with a "centroid mode" that takes pre-aggregated `c_z ∈ R^8` per zone instead of raw `h_i ∈ R^8` per NPC. Phase 2: in `crowd_attention.rs`, add `CrowdAttentionStep::tick_zones_into` that attends over zone centroids with fog-of-war at the zone level (zone visibility = neighboring zones within `region_radius`). | **Issue 366** (extension to R167 / P355 Phase 3) — riir-ai/.issues/366 (removed per noise-reduction rule) |

This is a **Gain**-tier extension (incremental improvement to a shipped Super-GOAT), not a new Super-GOAT. It belongs as an `.issues/` follow-up, not a plan or a new research note. The novelty is the *combination* (zone-centroid attention × existing crowd-attention × fog-of-war), not any single piece.

### 2.5 §3.6 Defend-wrong PoC reasoning

I am **not** claiming:

| Claim | Type | Status |
|---|---|---|
| Set attention (the architecture) is covered by R354 + crowd_attention.rs | Architectural | **Proven** by grep + read (the Super-GOAT R354 verdict already established this) |
| The modelless analog is faster than TabFM | Latency | **Likely true** (8-dim HLA vs hundreds of features; sigmoid vs softmax) but not benchmarked head-to-head; not claimed |
| The modelless analog **matches or beats** TabFM on TabArena (tabular prediction) | **Quality** | **Not claimable** — different domain entirely. TabFM solves supervised tabular prediction; the codebase solves NPC crowd cognition. The set-attention architecture transfers; the task does not. |

The third claim is *not even wrong* — TabArena is not a benchmark we could or should run. The Pass verdict rests on:
- Training-only redirect (pre-trained weights → riir-train).
- Architectural coverage (set attention already distilled as Super-GOAT in R354).
- Different domain (tabular prediction ≠ NPC cognition).

No PoC needed because no quality-parity claim is being made. If a future plan wanted to claim "our modelless set attention beats TabFM on a tabular benchmark", §3.6 would mandate a PoC — but that is not this verdict.

### 2.6 §3.5 Modelless unblock check (for completeness)

If someone proposed "train a TabFM-style foundation model for NPC cognition", the §3.5 protocol returns MODELLESS-VALIDABLE before reaching riir-train:

1. **Freeze/thaw** — can a frozen snapshot fix the "which Q/K/V projections" question? → YES: the projections are frozen artifacts (BLAKE3-committed in `NeuronShard`), atomic Arc swap at zone-load. **Path 1 succeeds.** (R354 §2.5 documents this.)
2. **Raw/lora hot-swap** — can a deterministically constructed reader-LoRA fix it? → N/A; the projections are constructed from CS-rankings + functor directions, not learned.
3. **Latent-space correction** — can a sigmoid projection fix it? → YES: that IS the shipped primitive. `α_ij = σ(q_i · k_j · β_zone)` with deterministic q/k. **Path 3 succeeds.**

Conclusion: a "trained TabFM for NPCs" is NOT required. The modelless paths cover the architecture. → No riir-train dependency for the runtime primitive; riir-train only if someone specifically wants to pre-train a tabular foundation model as a training-method artifact (which is Google's contribution, not ours).

---

## 3. Verdict

**Pass.**

One-line reasoning: **TabFM is a pre-trained foundation model — the value is the weights (trained on 100M+ synthetic SCMs), and that is → riir-train; the architecture (alternating ABD/ABA set attention + row compression + ICL transformer on compressed rows) is a refinement of NPT, which the codebase already distilled as Super-GOAT in R354 with shipped open primitive (`set_sigmoid_attention_into`) + private runtime (`crowd_attention.rs`); the latent reframing on HLA state is strictly stronger for the codebase's domain (NPC crowd cognition), and TabArena (tabular prediction) is not a benchmark we should run.**

### Tiers (high → low) — applied

| Tier | Criteria | Routing | This paper |
|---|---|---|---|
| Super-GOAT | Novel mechanism + new capability class + selling point + force multiplier | Open primitive + private guide + plans | ✗ — mechanism already distilled as Super-GOAT in R354 |
| GOAT | Provable gain over existing approach | Plan + feature flag + benchmark | ✗ — gain is on TabArena (different domain); no benchmark we can run |
| Gain | Incremental improvement | Plan only, feature flag | ✗ — the cross-zone centroid fusion (§2.4) is a Gain-tier extension, but it tracks as an `.issues/` follow-up to R167/P355, not a new plan |
| **Pass** | Training-only OR not relevant OR already ships | One-line note, no files in session | **✓ — pre-trained weights → riir-train; architecture ships (R354); different domain** |

### Novelty gate (Q1–Q4)

1. **No prior art?** NO. R354 (NPT, Super-GOAT) is the *direct* parent — TabFM's blog literally names TabPFN and TabICL as the architectures it synthesizes, and both descend from NPT (Kossen et al. 2021). The open primitive `set_sigmoid_attention_into` ships (P354). The private runtime `CrowdAttentionStep::tick_into` ships (crowd_attention.rs, Phase 1 of P355).
2. **New class of behavior?** NO. Set attention for cross-entity refinement is the capability class; R354 already established it as the "missing third quadrant of crowd-scale NPC cognition". TabFM adds row compression (a perf optimization) and a different domain (tabular prediction vs NPC cognition).
3. **Product selling point?** NO (for the codebase). TabFM's selling point ("zero-shot tabular prediction in BigQuery via `AI.PREDICT`") is a Google product, not a riir-* selling point. The riir-* selling point ("10,000 NPCs collectively infer a threat pattern via cross-entity set attention", R167) is already captured.
4. **Force multiplier?** Partial — the cross-zone centroid fusion (§2.4) connects crowd-attention (P355) with region-scale cognition, which is a real combination. But it is a Gain-tier extension, not a force multiplier that creates a new pillar.

→ 1 NO, 2 NO, 3 NO, 4 partial. Not Super-GOAT.

### MOAT gate per domain (§1.6)

- `katgpt-rs` MOAT (paper-derived fundamental primitive): R354 *is* the fundamental primitive; TabFM adds no new open primitive.
- `riir-ai` MOAT (pillar-level / Super-GOAT): R167 *is* the pillar-level guide; TabFM adds no new private selling point. The cross-zone centroid fusion is a Gain-tier extension tracked as an issue.

→ Neutral for both repos. Pass is the correct tier.

---

## 4. What ships where (5-repo discipline)

| Repo | What | Why |
|---|---|---|
| `katgpt-rs` | Nothing new | R354 / P354 already distilled and shipped the open primitive |
| `riir-ai` | One `.issues/` follow-up (§2.4): cross-zone set attention over HLA centroids | Gain-tier extension to R167 / P355 Phase 3, not new research |
| `riir-chain` | Nothing | No sync-boundary angle |
| `riir-neuron-db` | Nothing | No shard/freeze angle beyond what R354 already covers |
| `riir-train` | One-line note: "TabFM (Kong & Das, Google, 2026-06-30) — pre-trained tabular foundation model; synthetic-SCM pre-training at 100M+ dataset scale; hybrid TabPFN+TabICL architecture (alternating ABD/ABA + row compression + ICL on compressed rows). Training-method reference for set-attention foundation models; runtime analog covered modellessly in katgpt-rs/R354+P354." | The paper's actual contribution is here |

---

## 5. Open questions / risks

- **Q: Is the cross-zone centroid fusion worth implementing?** Possibly — it is the natural "region-scale" cognition layer above per-zone crowd attention. The open primitive already supports it (just pass centroid vectors instead of raw NPC states); the runtime work is adding `CrowdAttentionStep::tick_zones_into` with zone-level fog-of-war. Tracked as **Issue 366**.
- **Q: Does TabFM's row compression teach us anything new about HLA dimension reduction?** No — HLA is already 8-dim (already compressed). TabFM compresses from `d·e` (features × embedding dim, often hundreds) down to `h` (typically 64–128). HLA started at 8. There is no compression insight to borrow.
- **Q: Should we adopt TabFM's ABA (cross-column attention) for HLA dimension interaction?** No — HLA's 8 dimensions are already interacting via latent functor direction vectors (R123). ABA would be a learned version of what functor directions already do deterministically. Adopting ABA would require training (→ riir-train) and would violate the modelless-first mandate.

---

## TL;DR

**Verdict: Pass.** TabFM is Google's zero-shot tabular foundation model: pre-trained on hundreds of millions of synthetic structural-causal-model datasets, hybrid TabPFN+TabICL architecture (alternating cross-row/cross-column attention → row compression → ICL transformer on compressed rows), single-forward-pass prediction on unseen tables. The pre-trained weights and synthetic-SCM training pipeline are → riir-train (one-line note). The underlying set-attention architecture is a refinement of NPT, which the codebase already distilled as Super-GOAT in R354 — the open primitive `set_sigmoid_attention_into` ships in katgpt-rs/P354, and the private runtime `CrowdAttentionStep::tick_into` ships in riir-ai/crates/riir-engine/src/crowd_attention.rs. The latent reframing on HLA state is strictly stronger for the NPC-cognition domain (no per-row embedding, deterministic Q/K/V from CS-rankings + functor directions, sigmoid gate, fog-of-war compatible, sync-boundary clean). The one architectural element TabFM adds beyond R354 — row compression before the ICL transformer — suggests a single Gain-tier fusion (cross-zone set attention over HLA centroids) tracked as an `.issues/` follow-up to R167/P355, not a new plan. No files created in katgpt-rs / riir-ai / riir-chain / riir-neuron-db beyond this note.

---

## 6. Addendum (2026-09-30) — arXiv technical report (2609.37959) + TabFM-Auto (2609.37989)

> **Sources:** TabFM technical report — [arXiv:2609.37959](https://arxiv.org/abs/2609.37959) (Kong, Louidor Ilan, Nie, Narayan, Sen, Zhou, Fu, Oymak, Das — Google Research, submitted 2026-09-29) · TabFM-Auto: Self-Evolving Pipelines for Tabular Foundation Models — [arXiv:2609.37989](https://arxiv.org/abs/2609.37989) (Fu, Su, Sen, Narayan, Sanghavi, Das, Kong — Google Research/DeepMind, same day) · feature explorer: deqingfu.github.io/tabfm-auto
> **Status:** RECORD — reference kept for later; activation deferred by owner call this session ("we may need this later but not now"). No plans/issues filed; fusion leads parked in §6.4, novelty TBD. (This addendum was briefly a sibling note — Research 597, now a redirect alias — merged here by owner call; §6 is the single content home.)

### 6.1 The TabFM report — deltas since the blog-era note above

The arXiv report formalizes what §1 recorded from the blog; **no new architectural element**. The benchmark surface, now formal:

| Method | Overall Elo (51) | Classification (38) | Regression (13) |
|---|---|---|---|
| TabFM (zero-shot, single forward pass) | 1785.3 | 1768.7 | 2045.9 |
| TabFM+ (cross+SVD features, 32-way NNLS ensemble, Platt) | 1856.0 | 1836.7 | 2169.2 |
| EXAONE-Tabular | 1764.6 | 1759.5 | 1967.8 |
| AutoGluon 1.5 (extreme) | 1668.4 | 1664.6 | 1851.2 |
| TabPFN-3 | 1660.1 | 1637.7 | 1863.1 |

The §3 Pass verdict stands unchanged.

### 6.2 TabFM-Auto — the mechanism

An LLM coding agent (Claude Code / Codex / Antigravity × Opus 5 / Gemini 3.8 Flash) evolves a **data pipeline around the frozen TabFM** — the agent edits `pipeline.py` (four functions + a kwargs dict), the weights never change:

- **`preprocess()` (Φclean)** — sentinel values → NaN + missingness indicators (a `0` the FM cannot read as "unmeasured"), reversible target transforms g(y) (log(1+y), Box–Cox).
- **`engineer()` (Φfeat)** — semantic features from column names: domain formulas (Strouhal/Helmholtz on airfoil −14.6/−17.3% RMSE; MAP/Shock Index on clinical tables; SDSS color indices; ICD-9 → organ-system chapters), bipartite graph degrees + frequency encodings over combined train+test, collinearity pruning + truncated SVD, and auxiliary-file summarization (seismic waveforms → FFT bands + STA/LTA, 9.4× better than the best external MLE agent; 3D crystal lattices → unit-cell volume; molecular coords → r⁻³ + Karplus dihedrals).
- **`sample()` (Sctx)** — multi-view context selection (natural + minority-oversampled + cluster-stratified views, averaged) under the 16,384-row pretraining cap.
- **`postprocess()` (Ψpost)** — g⁻¹ inversion, **log-odds prior shift toward the empirical training prior**, temperature/Platt scaling.

Protocol worth reading whole: identity pipeline P₀ first, keep-only-improvements-over-P₀; bubblewrap-namespace sandbox (no network, test splits unmounted); row-order permutation vs index leakage; ≤96 evals or 6h/dataset on one H100. **Zero retraining noise is the thesis** — the frozen core makes small gains measurable where joint features+architecture+hyperparameter search (what MLE agents do) is noisy.

Numbers: 1785 → **2013 Elo (+228)**, five configs sweep TabArena's top-5; regression +467 Elo (linearized physical ratios + target transforms let the FM interpolate what tree splits approximate coarsely); **#1 on MLE-Bench-Tabular** (1827 Elo, 86.7% win rate, 4 Kaggle medals). Taxonomy: feature table modified in 95.1% of runs; domain-readable schemas (Cat I, 17/51 datasets) gain ~3× anonymized ones (Cat II, 34/51). **The transfer finding:** pipelines found for TabFM transfer unchanged to TabICLv2 (+143.3), TabPFN-3 (+130.8), EXAONE-Tabular (+88.7) with zero further search — pipeline gains are largely engine-agnostic. **The ablation that carries the thesis:** an unconstrained coding agent (same harness/model/6h budget, may train anything) lands 1468.8 — 510.8 Elo BELOW; decomposition: frozen FM prior +316.5, pipeline search +194.3. Cost: five 51-dataset sweeps ≈ $17.6K API fees, 1.28B–12.13B prompt tokens each. Their stated future work — *"operations collected across datasets could form a reusable library that warm-starts search on new tables with fewer evaluations"* — is the healer corpus thesis.

### 6.3 Distillation — shipped kin + two uncovered micro-primitives

The four stages map onto shipped surfaces:

| Paper stage | Shipped kin |
|---|---|
| Φclean | tokenizer/normalization seams; healer fixture normalization |
| Φfeat | corpus entries; harness feature columns |
| Sctx | reflex `--corpus-cap` / `--cal-select-cap` (riir-reflex/src/bin/harness.rs) |
| Ψpost | reflex `--gate-fit-selection` / `--gate-fit-calibrated` (riir-reflex/src/bin/harness.rs flags; `SigmoidGateCalibrator` — katgpt-core `crates/katgpt-core/src/sigmoid_calibration.rs`, consumed at riir-reflex/src/engine.rs) |

Both protocol laws already ship: keep-only-improvements-over-identity = the healer's `--verify` baseline→apply→re-check→auto-REVERT; frozen-core-means-low-noise-search = the modelless posture's own argument. Two small **uncovered** modelless candidates, shelved (no consumer yet):

1. **Log-odds prior shift** — closed-form, composable with (not a replacement for) threshold fitting; corrects class-prior mismatch between pretraining/synthetic corpora and real data — exactly the bias a corpus-is-the-model engine can carry. Any future adoption runs the Report-the-Floor conformal gate (the R322 / Plan 340 law).
2. **Multi-view context ensembling under a cap** — evaluate over several stratified context subsets, average; the class-imbalance answer when context is capped. reflex caps corpus per-label (`--corpus-cap`) but does not view-ensemble.

### 6.4 Fusion leads (novelty TBD — owner-deferred, parked)

| # | Fusion | Surface | Status |
|---|---|---|---|
| F1 | TabFM as a reflex comparison lane (subprocess oracle, the GLiNER/AgentJev pattern: their stack serves, our Rust measures) | riir-reflex `src/lanes/` | Parked — cheapest if ever activated |
| F2 | "Reflex-Auto" — agent loop over the harness levers (already a modelless pipeline-around-frozen-engine) | riir-reflex `src/harness/` | Parked; AutoML prior-art risk, novelty TBD |
| F3 | Cross-healer-domain config transfer (the +69..+143 engine-agnostic finding applied to clippy/rust_perf/kernel_opt) | riir-clippy | Parked — the most clippy-native lead |
| F4 | Cross-dataset pipeline-op mining (their warm-start library = our corpus + trajectory store) | riir-clippy | Parked; the shape already converges on ours |
| F5 | Frozen cognition core + evolved per-NPC perception wrappers | riir-ai (weak) | Parked, lowest priority |

### 6.5 Addendum verdict

**RECORD — activation deferred by owner call this session.** No missing architecture (the four stages have shipped kin, §6.3; both protocol laws already ship); exactly the two §6.3 micro-primitives uncovered, parked with no consumer; TabFM-Auto's genuinely novel claim (full-pipeline agent search around a frozen FM) is the paper's own contribution — CAAFE/FeatLLM/OCTree/AIDE/R&D-Agent/MLEvolve define the surrounding prior art (the paper's own related-work section; no novelty claimed here, so no §4 sweep was run). This section exists so a future session greps `2609.37989` or `TabFM-Auto` and finds the mapping pre-done.

> **PASS-Redirects (synthesis):** Ranjan et al. [arXiv:2610.17867 "RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions"] — the multi-table RELATIONAL successor of this note's single-table TabPFN/TabICL family (RT cell-as-token + column/feature/neighbor attention vs TabFM's flat table ICL); full distill + per-track verdicts at `katgpt-rs/.research/640_RT_J_Relational_Context_Recipe.md` (walk-proximity context retrieval, THE JOIN corpus pipeline, sampler-seed ensembling; actions: ndb issue 640, reflex issue 104, riir-train issue 657).

---

## 7. Addendum (2026-11-19) — NVIDIA Kumo Tabular (2611.23001): the family's current SOTA member

> **Sources:** [arXiv:2611.23001](https://arxiv.org/abs/2611.23001) — "NVIDIA Kumo Tabular: Pushing the Accuracy–Efficiency Frontier for Tabular Prediction" (Qu, Hudovernik, Jurkovič, …, Leskovec, Fey — NVIDIA/INRIA/Stanford, submitted 2026-11-17; the tech report for weights released 2026-09-23) · weights `nvidia/Kumo-Tabular` on HF, **OpenMDW-1.1 (commercial use)** — the license undercuts TabPFN-3.5's restricted terms · pretraining + inference code `NVIDIA/structured-data-models` (Apache-2.0).
> **Status:** RECORD — PASS per §6's precedent; no plans/issues filed; leads parked (the open weights make Kumo-S/M the first *runnable* members of §6.4's parked F1 lane lead).

### 7.1 What it records

27M/61M/214M (S/M/L), classifier + regressor each; **first place on eight tabular leaderboards** (TabArena Elo 1962 [1862,2124] · TALENT 1633 · BeyondArena 1426 · ScoringBench mean-CRPS-rank 1.84 · STRABLE median 0.8325 · MulTaBench 2227 · RamanBench 1558 · TabBench-Bio 1265); all sizes on the accuracy–speed Pareto frontiers; Kumo-L ≈36× faster than LimiX-2 (TabArena) and 1.7× than TabPFN-3.5 (BeyondArena). Snapshot caveat: TabArena is living (v0.1.9.8, 19 Oct 2026 — LimiX-2 1952 is the strongest other entry there); §6.1's TabFM-Auto 2013 Elo is an earlier snapshot, not directly comparable. Known limits (their §9): grouped splits 4th (behind tuned+ensembled RealMLP/TabM/LightGBM), >100K training rows 9th, temporal 3rd (−3 Elo vs TabPFN-3.5).

**Architecture deltas vs TabICLv2** (the bloodline's open parent): interleaved column/row attention ×4 (S) / ×6 (M/L) instead of staged 3+3, column attention through learned inducing points (linear in rows), row attention rotary with 4 readout tokens per row; **SSMax** temperature τ(n) = α_h·log n_keys + **QK-norm WITHOUT learned gains** + weight decay on α_h (Appendix B: an attention-scale growth feedback loop — concentration → attention-output norm grows through the pre-norm residual → optimizer favors still-larger scales; output-renormalization also stabilized but hurt accuracy, not retained); length-dependent row temperature with a learned query gate γ(q); **Test-GQA** (test rows read 2 of 8 / 2 of 16 K/V heads → 4–8× smaller train-row KV cache, exploiting the ICL mask where train rows never attend to test rows); missing-value-aware embeddings; Fourier features at learned frequencies; 999-quantile pinball regression head; >10 classes split into ten-class subtasks via error-correcting output codes.

**Kumo prior** (SCM generator extending TabICLv2's): per-table Dirichlet-correlated mechanism preferences; a distributional numerical converter (location/noise-scale/subgroup channels, logistic or identity link, rounding to counts); correlated feature blocks (30% of tables); staged realism augmentations (argmax classes, relabeled targets, coarsened features, >200-category codes, heavy-tailed targets); an 8-pattern missingness taxonomy; **learnability filter** (25 depth-6 ExtraTrees vs the mean baseline — unlearnable tables redrawn at 27–37%, too-easy tables ≤5%-of-baseline-error retained at HALF rate in stages 2/3). Pretraining: 3-stage row-length curriculum 1K → 10K → 60K rows, Muon (most matrices) + AdamW with decoupled WD 0.01 pulling α_h toward zero, 800 / 3,600 / 11,600 H100-h per S/M/L checkpoint. Inference recipe: 8–16 estimators ensembled over preprocessing permutations (identity/Yeo–Johnson/squashing/rank-Gaussian scaling maps, Latin-square column orders capped at 500, cyclic class shifts, random sign flips); **softmax temperature 1** (their largest log-loss gain — it matches the unscaled training softmax); estimator KV caches double-buffered through pinned host memory.

### 7.2 Path 0 inventory — every modelless extraction lands on a covered/parked class (signal-diffs run)

| # | Mechanism | Disposition (cousin + diff) |
|---|---|---|
| M1 | Ensembling over input symmetries (estimator ensembling / multi-view context / sampler seeds) | **Covered-as-parked, twice**: R640 M4 (sampler-seed ensembling, activates only with a stochastic retrieval mode) and §6.3's multi-view-context micro-primitive (parked, no consumer). The measurement VARIANT ships (reflex `mc_ensemble` + `perturbation_ensemble` perturb embeddings to measure uncertainty, sample 0 stays served; `--perm-probe` measures option-permutation spread) — the ensemble-IS-the-answer variant is the delta, same class; the ~1 ms decision budget prices E ∈ {2,4} before any escalation. |
| M2 | Learnability filter + too-easy retention | **Covered AND measured** in the adjacent context: R640 M7 → reflex issue 104 **closed recorded-negative 2026-10-30** (reflex HISTORY.md: "the too-easy pole IS where the lift lives" — gating too-easy items out removed the lift; Benches 120/121/122 remain the controlling internal record). Convergent nuance, disclosed not claimed: Kumo itself retains too-easy tables at half rate — consistent in spirit with our too-easy-carries-lift finding, but its filter shapes a *pretraining curriculum*, a purpose no open surface here has. |
| M3 | Quantile readout + CRPS scoring as a UQ wire | **Covered-adjacent**: every UQ-bearing surface is governed by the conformal-naive floor (R322 / Plan 340 "Report the Floor" — cannot beat the floor ⇒ FAIL); a quantile-band readout is a wire shape awaiting a UQ consumer that does not exist. |
| M4 | Attention-side parameterizations (SSMax with a pinned constant, gain-free QK-norm, Nyström/inducing-point landmarks, fixed-width pooling readout) | **No consuming surface**: the decision stack is sigmoid-gated (no softmax dilution law applies); riir-infer serving lanes are G5-parity-gated to reference checkpoints (attention-normalization changes break bit-parity without retraining — and no dilution failure is documented on any lane); crowd set attention (R354) is already the O(nk) sigmoid shape with no key-count pathology. Watch rows only. |
| M5 | Test-GQA + support-KV amortization | The ICL mask asymmetry (train rows never attend to test rows) has **no analog in any shipped surface**: LLM serving attends the full prefix, set attention is bidirectional, corpus scoring has no attention. The amortization HALF — compute corpus-side state once per epoch — already ships (the index-cache family in riir-refine, corpus caches). |
| M6 | ECOC wide-arity decode | **Shape on the shelf**: board option arity ≤ 8 (S8 covers 8 labels); no >10-arity suite exists. Recorded; a consumer arrives only with an arity widening. |
| M7 | Fixed/random Fourier features + NaN-mask cell transforms | Hashed-feature embedding already covers the numeric-expansion axis (reflex `embed.rs`); the explicit-missing-token variant has no numeric-cell surface in the stack. |

### 7.3 Training track — recipe rows parked; consumers paused or absent (live-state verified 2026-11-19)

- **Stability parameterization** (QK-norm-no-gains + SSMax + WD on learned temperature scalars; the portable precision law "learned scalars never ride BF16 master weights"): the advocate-named consumer is the **specdec beat lane (riir-infer 1179), PAUSED** (delegated-owner verdict d9c2bb28, recorded in the 1183 issue file; re-open trigger (3) contemplates a *new draft-head objective/architecture proposal that pre-registers its own stop line*, and a Bonsai-target lane ranks above any qwen3.8 restart). Parked: if trigger (3) ever fires, Appendix B is a candidate ingredient — the parameterization targets ICL-attention stacks, not the draft head's own structure, so it is an ingredient someone must deliberately propose, never an automatic re-open. No filing while the pause stands.
- **Prior design** (missingness taxonomy, staged realism augs, learnability filter): the one on-domain consumer (riir-train **Plan 350**, tabular flow-matching anomaly arms) is Active-but-**blocked** (on riir-chain issue 112 T1/T2). If it activates, its benchmark generator may borrow the missingness-stratified slices + the learnability filter — cross-ref only; no plan amendment from this session (Plan 350's design is R503's record).
- **Muon + AdamW at TFM scale** (with WD on learned scalars): already distilled — R238; the optimizer axis rides R640's R1-class backlog rows; LAprop has its own program and no active training lane is in flight (plan 492 stopped at its stop branch).
- **Model-family reproduction**: refused, concurring with the model-based advocate — 3.5K–70K 4090-h and no product surface consumes tabular predictions (decision serving is text suites; the game runs explicit rule books). Two-sided priority-ladder check recorded: the rulewright gains nothing (code fixing has no tabular surface; its corpus-gate class is the already-measured one), and the decision-serving loop gains nothing beyond the parked F1 lane.

### 7.4 Verdict

**PASS — RECORD row (§6 posture).** The family's SOTA member adds engineering depth (interleaving, the stability parameterization, Test-GQA, estimator ensembling, the staged prior) but no mechanism outside classes this note and R640 already dispositioned. No quality-parity claim is made or needed: the pass rests on family precedent (R597's one-home ruling), measured-closed gate classes (issue 104), and parked-without-consumer rows — the §3.6 signal-diffs above are the evidence, not grep-only coverage. Still parked, honestly, by Claude verdict (this addendum) — no consumer: the board has no tabular suite, and the lane-seating screen's own probes returned no winner separation (riir-reflex R012 / Bench 165). §6.4's parked **F1** lead (a TFM comparison lane in reflex, the GLiNER/AgentJev subprocess-oracle pattern) now has its first practically runnable members (Kumo-S/M at 27M/61M; text suites would ride frozen e5-small cell embeddings, the MulTaBench recipe) — it re-opens when a tabular suite lands on the board and a lane adds one. No files beyond this addendum + PASS-Redirects lines in the two cousin notes (R640, reflex R012).
