# Research 624: Associative Memory Capacity Laws — Interference-to-Signal as a Corpus-Growth Diagnostic

> **Source:** Alex Wang, "You could have invented transformers from memory" (blog, <https://alexwang.ai/posts/associative-memory/>, 2026-10-15) — expository synthesis. Underlying results it builds on: Zhong et al. [arXiv:2505.19488](https://arxiv.org/abs/2505.19488) "Understanding Transformer from the Perspective of Associative Memory" (the interference-to-signal framework), Krotov & Hopfield dense-associative-memory line, Geva et al. (FFN keys/values), Garcia et al. "MLPs are Hebbians", Su 2021 log-M entropy scaling / Nakanishi 2025 SSMax (arXiv:2501.19399) / Qwen LogN / YaRN (β ∝ log M family).
> **Date:** 2026-10-17
> **Status:** Active
> **Related Research:** 455 (Hebbian Kernel Memory fact-storing MLP), 466 (CP^(d−1) Hopfield recall), 472 (Embedding Retrieval Dimension Capacity Limit), 541 (UDDM associative-memory regime probes), 392 (SSMax gold share), 610 (Scale-Invariant Attention), 024 (δ-Mem online associative memory), 454 (HOPE capacity kernel), 378 (HOLA exact-KV), 511 (Memory Layers at Scale), 625 (Path-Kernel exemplar-superposition lens — GD-trained weights read as rows of this store; response-space similarity channel)
> **Related Plans:** 559 (hebbian_kernel_memory primitive), 411 (ssmax_goldshare), 622 (scale_invariant_attn)
> **Cross-ref:** riir-refine Issue 166 (weighted-sum/threshold consumers of scored gates), riir-reflex Issue 093 (CorpusDistanceGate drift vs corpus growth)
> **Classification:** Public

---

## TL;DR

The blog derives linear layers → MLPs → self-attention as one family of associative memory stores, differing only in the **gating function applied to key similarities**, and reads each one's capacity off the interference-to-signal (I/S) ratio at recall. Nearly every ingredient is already distilled in this workspace from primary sources — and largely shipped. The genuinely new extractions are (1) the **unified I/S closed forms** as a corpus-growth diagnostic (0 internal hits for the framing), and (2) a **capacity reconciliation for the house sigmoid-not-softmax rule**: for *modelless* (no-gradient) gates, a sigmoid at a **midpoint threshold θ** has exponential-tail interference suppression of the same class as softmax — sigmoid forfeits no capacity class, only the normalization, which the blog's own math shows does not affect I/S. Two scope laws make the application honest: **a monotone gate cannot change any pure-ranking outcome** (sharpness/threshold only matter where gated outputs are summed/blended or compared to absolute thresholds), and **a trained scorer's threshold is part of its trained contract** (moving it is a calibration break, not a lever). The one clean consumer is the **abstention gate** (`CorpusDistanceGate`): its M genuinely grows with the corpus and it compares `max_sim` against a fixed midpoint.

**Distilled for katgpt-rs (modelless, inference-time):** capacity laws as (a) an analysis vocabulary that separates threshold-consumers from blend-consumers from ranking-consumers, (b) a threshold-placement rule (θ = 0 suppresses nothing), (c) a log-M calibration law for the threshold class — consumers are modelless gates, not attention kernels.

---

## 1. Source core findings (the capacity ladder)

Query a store with a stored key `k_j` (unit keys, random values). Recall returns `v_j` (signal) plus `Σ_{i≠j} v_i·w_i/w_j` (interference); the store fails when ‖interference‖ ≳ ‖signal‖.

| Store | Recall | Capacity | At d_k = 64 |
|---|---|---|---|
| Outer product (Hebbian) | `Vᵀ K x` | `d_k` | 64 |
| Optimal linear (least-squares) | `Vᵀ K (KᵀK)⁻¹ x` | `2·d_k` | 128 |
| ReLU MLP | `Vᵀ ReLU(Kx)` | `2·d_k` | 128 |
| Squared ReLU (GLU, shared gate) | `Vᵀ ReLU(Kx)²` | `2·d_k²/3` | ≈ 2,700 |
| Cubed ReLU | `Vᵀ ReLU(Kx)³` | `2·d_k³/15` | ≈ 35,000 |
| Softmax (self-attention), β = √d_k | `Vᵀ softmax(βKx)` | `e^{2√d_k − 2}` | ≈ 1.2 M |
| Any of the above, β ∝ log M | — | unbounded in practice (Su 2021 / SSMax / LogN / YaRN) | > 2 M |

Three structural facts carry over to any similarity-scored store:

1. **Power-p gating multiplies capacity by d_k^(p−1).** Raising similarities to a power shrinks each interference term from `d_k^{-1/2}` to `d_k^{-p/2}`; I/S ≈ `√(M/d_k^p)`.
2. **Exponential gating is the p→∞ limit** (exp(βs) = Σ_p (βs)^p/p!) — that is *why* softmax attention has exponential capacity. Normalization (÷ Σ_j e^{βs_j}) **does not change the I/S ratio** — it exists only for numerical stability. The interference depends only on weight *ratios*.
3. **β ∝ log M keeps recall working as the store grows** (softmax saturation is the training-side cost; β → ∞ collapses to argmax). Published forms: Su 2021 (`log M / log L`), SSMax (`s·log M`, learned s), Qwen LogN (`max(1, log M/log L)`), YaRN (`(0.1 ln(M/L)+1)²`).

## 2. Distillation

### 2.1 The unified I/S estimator, and its two scope laws

Closed forms, parameterized by (M, d_k, gating): linear `√(M/d_k)` · optimal/whitened `√(M/d_k − 1)` · ReLU^p `√(M/d_k^p)` · softmax at sharpness β `√M · e^{β²/d_k − β}` (unit keys; E[e^{ts}] = e^{t²/2d_k} for s ~ N(0, 1/d_k)).

**Scope law 1 — monotone-gate invariance (binds the application).** For pure **ranking** consumers — argmax over per-item scores, top-k retrieval — ANY monotone gate `σ(β(s−θ))` leaves the outcome unchanged for every β and θ: ranking is order, and a monotone map preserves order. Sharpness and threshold matter only where gated outputs are (a) **summed or blended** with other terms (weights in a fusion), or (b) **compared to an absolute threshold** (abstention, admission). Ranking degradation that does occur is intrinsic to the raw similarities (the max-of-M distractor tail, below), never gate-mediated. §2.3's consumer table sorts every surface by this law.

**Scope law 2 — the ranking tail is an isotropic null, not a law.** `E[max_i s_i] ≈ √(2 ln M / d_k)` assumes random isotropic unit keys. Real embeddings are clustered and anisotropic; `max_sim` is dominated by near-duplicates. The formula is a **null model**: a measured mismatch rules out the null, it does not refute drift.

**Failure regimes:** I/S < 1 is the failure condition for **value reconstruction** (summing value vectors — the blog's regime, √M growth). Ranking surfaces do not fail this way; their margins tighten only through the tail.

### 2.2 Sigmoid-vs-softmax capacity reconciliation (our derivation — resolves the house doctrine question)

The house rule is "sigmoid, never softmax" (2,600+ doctrine mentions across the workspace; softmax compute exists only inside forward passes and one documented categorical exception). The blog's ladder invites the question: does bounded sigmoid gating forfeit the exponential-capacity class?

**No — provided the threshold is at a midpoint.** For a gate `w(s) = σ(β(s − θ))` queried with the stored key (s_j = 1, interference s_i ~ N(0, 1/d_k)):

- `w_i/w_j ≈ e^{β(s_i − θ)}` for β(s_i − θ) ≪ −1 (sigmoid's tail is exponential)
- I/S ≈ `√M · e^{β²/d_k − βθ}` — the softmax formula with softmax's fixed θ=1 made a free knob
- Capacity `ln M = 2βθ − 2β²/d_k`; maximized at `β = θ·d_k/2`, giving `ln M_max = θ²·d_k/2`

At the blog's own operating point β = √d_k and a midpoint gate (θ = ½): `ln M ≈ √d_k − 2` against softmax's `2√d_k − 2` — **half the exponent, same exponential class** (≈ 400 vs ≈ 1.2 M at d_k = 64; the gap closes with d_k or wider θ). Unlike softmax, sigmoid has bounded outputs, no normalization, and no gradient-saturation hazard — the last irrelevant for modelless gates, which no gradient ever traverses. Normalization is the only thing softmax adds, and (§1 fact 2) it does not change interference.

**The unifying sentence (honest mechanism):** at the capacity edge, `ln M ≤ θ²d_k/2` is exactly the statement that the expected max-of-M distractor similarity `√(2 ln M/d_k)` stays **below the gate threshold θ**. In this sharp regime the interference sum is dominated by its largest term, so the second-moment I/S argument is borderline there — the correct route is extreme-value, and it lands on the same condition. §2.1 and §2.2 are one law read two ways.

**The θ = 0 trap:** the common `sigmoid(β·dot)` shape has θ = 0 — `w_i` hovers at ½ regardless of β, so interference is suppressed by at most 2× **no matter how sharp the gate is**. For a reconstruction consumer this is capacity ~4 memories; for a blend consumer it means the gate contributes a near-constant weight. **Threshold placement is a first-class lever, equal to sharpness — but only for the consumer classes of §2.3(a)/(b).**

**The trained-scorer exclusion (scope law 3).** For a TRAINED scorer — sigmoid-NCE two-tower, Platt-style `SigmoidGateCalibrator`, any logistic head — the runtime temperature/threshold is part of the trained contract (e.g. `rule_embed.rs`: "`SIGMOID_TAU` shared by training and runtime scoring — the shipped score distribution must be the trained one"). Moving it at runtime is a **calibration break**, not a capacity lever. The capacity laws of this note apply to **modelless gates only**.

### 2.3 Consumer table (sorted by §2.1's scope law)

| Class | Surface | Site | Growth axis | Law says |
|---|---|---|---|---|
| (a) Absolute-threshold | Abstention gate | `riir-reflex engine.rs` `CorpusDistanceGate::new(rows, GATE_MID=0.35, GATE_SCALE=8.0)` — fixed constants | **M = corpus rows grows** | **The one clean consumer.** The drift carrier is the **off-corpus** population: an unrelated query gets M chances at an accidental near-neighbour, so its `max_sim` upper tail rises (null: ≈ √(2 ln M/d_k)) toward `mid` — false-accept drift, the gate stops rejecting exactly what it exists to reject. In-corpus `max_sim` (≈ 0.68–0.70 at birth, already above `mid`) only rises further away — the desert narrows from the off-corpus side. Note: the cascade re-fit posture (cal-slice 30th-percentile fit) may absorb this by construction — measure both postures separately. |
| (b) Blend weights | Route term | `engine.rs` `route_scale: 8.0` blending `route_base` (one `exact_sigmoid(dot·…)` per **domain**) | **N domains — fixed count.** Corpus growth changes centroid quality, not M | **Not an I/S site** (mis-modelled in round 1): a relative-scale question, Issue-013-style sweep — not a capacity item |
| (b) Blend weights | Fused score | `engine.rs` `DEFAULT_SCORE_TEMPERATURE = 24.0` → `sigmoid(s/24)` blended into the fused gate | Option sets — bounded | Blend-weight calibration; lowest expected effect |
| (b) Blend weights | Trajectory scoring | `riir-refine src/evolve.rs` `sigmoid(dot/temperature)` — **config field** (default 1.0), not a constant | Candidate pool | Blend-weight sweep item; ranking use is gate-invariant |
| (c) Pure ranking | Retrieval top-k | `traj_store.rs` `sigmoid_dot_8`, `rule_embed.rs` `score_rule`, ndb KNN indexes | Corpus grows | **Gate-invariant** — no monotone gate can move a ranking outcome; margin drift is intrinsic (tail), and the τ=0/θ=0 trap does not apply to order |
| (c) Pure ranking | Trained two-tower | `rule_embed.rs` per-pair scoring, `SIGMOID_TAU = 0.1` — a temperature (`β = 1/0.1 = 10`), **threshold 0** | — | Excluded by scope law 3: trained temperature, shared train/runtime **by contract**; not a lever. (The θ=0 shape is irrelevant here — the gate is consumed as a trained score, not as a suppression gate.) |

Empirical echo already on record: reflex Issue 013 capped routing corpora by hand (ag_news 64, banking77 40); riir-refine `.plans/005` L1444–1468 states "retrieval precision does not survive corpus growth — treat corpus growth as a precision liability, not a free win". The capacity law is the principled version of the first and an analysis vocabulary for the second — not a replacement for either's measurement discipline.

## 3. Fusion — what paper × shipped substrate produces that neither has alone

`SSMax` (shipped, attention-side softmax) × the threshold-placement derivation (§2.2) × the abstention gate (the one threshold-class consumer) → **the log-M/extreme-value calibration law transfers to the abstention surface**: `CorpusDistanceGate`'s `mid` either tracks the measured `max_sim` drift or the extreme-value null, with the re-fit posture as the incumbent to beat. The consumers' issues (§5) gate every step behind measurement.

## 4. Signal-diff vs shipped cousins (no duplication)

- **455 / `HebbianKernelMemory`** (`katgpt-core/src/hebbian_kernel_memory.rs`): ships the outer-product store `B₀ = (1/F)C_fᵀΦ` *and* the ridge-whitened least-squares store `B_λ = B₀(Σ̂+λI)⁻¹` (Whitened DEFAULT) — the blog's first two ladder rungs, with a construction margin law (`m·d / F·log F`). Diff: the margin law is a construction-time guarantee for that family; the I/S forms are a **gating-parameterized estimator** covering the sigmoid/exp families the margin law does not. Complementary, not duplicated.
- **392 / 411 SSMax**: the β ∝ log M law, shipped default-on **for softmax attention dilution**. Diff: scope — the modelless sigmoid-gate class is outside it; §2.2 is what lets the law cross over, and only to the threshold-class consumer.
- **472 (retrieval dimension capacity limit)**: worst-case sign-rank bound on the **dimension axis** (d_k tables, `dim_capacity_ceiling`). Diff: orthogonal axis — this note is the **gating/threshold × store-size** axis.
- **466 / `cp_hopfield`, 541 (Gardner LUT), 024 δ-Mem, 454 HOPE**: Hopfield-line capacity results and online-write associative state — different stores; the I/S vocabulary is the common analysis layer.

**What is NOT claimed:** the I/S framing belongs to Zhong et al. (arXiv:2505.19488); the capacity ladder is textbook-line (Hopfield/Krotov; Bietti's lectures); sigmoid-attention parity for **trained** transformers is published (Apple SIGMA, "Theory, Analysis, and Best Practices for Sigmoid Self-Attention"; sigmoid-attention matched softmax through 7B per Leroux et al. 2025). Monotone-gate ranking invariance is elementary. The remaining claim is deliberately narrow: the **modelless-threshold-gate application** — abstention-gate drift vs corpus growth, threshold placement for blend consumers, and the trained-scorer exclusion as a design rule.

## 5. Verdict

**Gain (narrow).** The post is expository and its ingredients are already distilled from primary sources (several shipped); the surviving value is the unified I/S vocabulary, the threshold-placement derivation, and ONE clean consumer (the abstention gate) plus measurement items elsewhere. Round-1 review corrected the consumer table: ranking surfaces are gate-invariant, `route_scale` scores a fixed N domains (blend weight, not capacity), and trained scorers are excluded by contract.

| Gate | Verdict | Why |
|---|---|---|
| Q1 No prior art | No (for the math) / narrow yes (for the application) | Zhong et al., Apple SIGMA, the blog own the theory; modelless-threshold-gate calibration is unclaimed internally (0 hits) |
| Q2 New behavior class | No | Calibration + analysis vocabulary, not a new capability |
| Q3 Selling point | Weak | "Abstention stays calibrated as the corpus grows" is hygiene, not headline |
| Q4 Force multiplier | No (corrected in review) | One clean consumer (reflex abstention) + analysis vocabulary shared by 2 pillars — not the ≥3-pillar bar |

1.5 of 4 → **Gain (narrow)**, note-level. MOAT gate: katgpt-rs note-level (analysis vocabulary; no speculative primitive without a consumer — the estimator ships with the first consumer that needs it); riir-reflex decision-serving moat (abstention calibration drift — in scope, the lead consumer); riir-refine consumer-first moat (measurement-first; the four sites are ranking/blend consumers, expected-null or sweep-class — in scope as a cheap question-closer); riir-ai already consumes the law (`crowd_ssmax_gate.rs`) — no file; riir-train — not applicable (inference-side).

**Consumers filed this session:**
- `riir-reflex/.issues/093_abstention_gate_corpus_drift.md` — leads on `CorpusDistanceGate` drift (off-corpus tail vs the isotropic null; fixed-mid vs re-fit postures separated).
- `riir-refine/.issues/166_scored_gate_consumer_classes.md` — consumer-class audit of the four sites: weighted-sum/threshold consumers measured, ranking uses pre-registered expected-null, `rule_embed` pre-marked not-applicable (trained contract).
