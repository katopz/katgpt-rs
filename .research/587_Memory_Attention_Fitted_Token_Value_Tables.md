# Research 587: Memory Attention → Fitted Token-Value Tables (the V-side twin of anchor scoring)

> **Source:** Jiale Kang, "Memory Attention", [arXiv:2609.28399](https://arxiv.org/abs/2609.28399), 2026-09-23 (single-author, v1, FLA-framework experiments).
> **Date:** 2026-09-24
> **Status:** Active — dual-track Gain. Modelless arm: Issue 883 (closed 2026-09-25, HISTORY.md § Issue 883) — **P0 LANDED 2026-09-25**: the shared `katgpt-core/src/fitted_anchor_table.rs` substrate + the gemma-2-2b-it dashboard (riir-infer Bench 004: mean ρ_l(V)=0.48 / ρ(K)=0.50 / ρ_l(V−K)=0.49 @120k tokens — **go/no-go reads GO for P1/P2/P3**) + the Kimi-K3 dashboard-only fixture (Bench 889, MLA ρ≈0.52–0.60; KDA fixture-class null); **P1–P3 primitives + P4 LANDED 2026-09-25** (`0b768e95d`, Bench 895 — G1/G3/G4 PASS, P1 + P3 G2 bars FAILED and recorded), model-bound G1 = riir-infer Issue 013. Training arm: [riir-train Plan 418](../../riir-train/.plans/418_memory_attention_recipes_secondary.md) (SECONDARY, queued).
> **Related Research:** 165 (Q-K=V projection sharing — the K/V-redundancy evidence), 511 (Memory Layers at Scale — the lookup-capacity family + our PKM legality precedent), 452 (RoVE — the in-tree inverse-RoPE consumer), 487 (massive activations — sink tokens are the degenerate per-token-mean outlier), 159 (KVarN), 586 (differential scoring — the Q-side twin), 278 (Engram fusion).
> **Related Plans:** riir-train 417 (the SECONDARY-queue precedent), 418 (this paper's training arm); katgpt-rs Plan 557 (RoVE retrofit PoC, pending — the retrofit precedent).
> **Cross-ref (riir-ai / riir-infer):** riir-infer `gguf_loader.rs` `attention_k_eq_v` (the loader-level V-optional seam, Gemma-4); riir-ai serving league tg128 (the P3 consumer cell).
> **Classification:** Public (the fitted-table primitive + variance laws are generic inference math).

---

## TL;DR

Memory Attention (MA) replaces the attention value projection with `V = K + Norm(E_l[s])` — per-layer, vocab-indexed memory tables, trained from scratch, with norm folded into the tables at inference so value construction is lookup + add. The trained architecture is closed to us (from-scratch, 2–3× params, owner-gated). The extraction is the **estimator reading**: the paper's own equation is a *decomposition claim about value tensors* (`V − K` has a large per-token-identity mean), and decompositions can be **fitted closed-form on a frozen checkpoint** — one calibration pass, per-token means, no gradient. That produces (P1) token-mean-removed V-cache quantization (variance cannot increase — law-of-total-variance; the quantizer-level gain is measured, not proven: absmax integer quant scales by range, which mean-removal does not bound), (P2) a fitted "K=V+" retrofit (serve `V = K + λ·E_l[s]`, delete W_V; the claim under test is only `quality(K=V+) > quality(K=V)` — no ordering vs full V is claimed or expected), and (P3) V-cache halving by reconstruction (`V = inverseRoPE(cachedK) + E_l[s]`, consuming the in-tree `RopeAction::apply_inverse_at`). The same pass yields an **R² dashboard** whose per-layer token-explained fraction ρ_l makes every product's go/no-go **offline-computable before anything is built**. This is the **V-side twin of Issue 882's Q-side anchor scoring** — one fitted-anchor-table substrate, two consumers.

**Distilled for katgpt-rs (modelless, inference-time):**
1. **The fitted-table estimator**: `E_l[s] := mean(V_t − K_t | s_t = s)` (pre-RoPE K tap), with James-Stein-flavored shrinkage `E_λ[s] = n_s/(n_s+λ) · mean_s(V−K)` — closed-form, streaming, Zipf-tail-safe. The value-mean twin `E^V_l[s] := mean(V|s)` drives the quant decomposition.
2. **The variance laws** (free from the same pass): `Var(V) = Var(E[V|s]) + E[Var(V|s)]` — per-layer token-explained fractions ρ_l(V), ρ_l(V−K) are arithmetic on accumulated sufficient statistics; for MSE-optimal / uniform-scalar quantizers the error envelope shrinks as a computable function of ρ (per-group absmax integer quant scales by RANGE, which mean-removal does not bound — that half is measured, not proven). Per-token table error is `O(σ_s/√n_s)`.
3. **The cache laws**: V-droppable fraction = `n_v/(n_kv+n_v)` = exactly 50% for every grouped-KV arch we serve; reconstruction = one orthogonal rotation (`G(−n)` — in-tree) plus one add; roofline trade condition closed-form per hardware.
4. **The serving arithmetic** (paper's own laws, ours better): FLOP law `ΔF_V ≈ L·S·d_v·(2d−1)`; storage law becomes a **dial** under top-K residency `P(K) = b_w·L·K·d_v` with closed-form Zipf coverage(K) — the paper's full-vocab table (2–3× params) is its K=N special case.

---

## 1. Paper core findings

- **Construction** (§3): `Q = XW_Q, K = XW_K, V = K + M` where `M = Norm(E[s])`, `E ∈ R^{N×d_v}` per layer (N = vocab, d_v = total KV dim), Norm = per-KV-head RMSNorm per retrieved row. W_V is removed. Values built from **pre-RoPE** keys (RoPE applied to Q,K for scoring only). Standard attention weighting/aggregation unchanged. §3.2 relates it to MLA: both aggregate a shared representation (MLA: latent C + post-aggregation W_V; MA: key K + additive token memory).
- **Inference** (§3.3): norm folds into tables (`Ē[i] = Norm(E[i])` offline) → `V = K + Ē[s]`, lookup + elementwise add. ΔF_V ≈ LSd_v(2d−1) FLOPs saved.
- **MA-Offload** (§3.4): tables addressed by (token ID, layer) only → CPU-resident with prefetch overlapped against compute. Measured (H800, BF16, 24L-H2048, batch 8×2048): −55% GPU param storage at 2.08× total params; prefill −3.04% (GPU-resident) / −0.31% (offloaded); decode +0.86% / −2.53%.
- **MA-Recall** (§3.5, analyzed only, NOT measured): reconstruct historical values `V[1:T] = K[1:T] + Ē[s[1:T]]` → no persistent V cache → 50% KV cache reduction; requires inverse-rotation of cached RoPE keys (or content-K retention + rotate-on-read).
- **Quality** (§4, matched token budgets, FineWeb-10BT, FLA framework, 373M–2.8B params, 10–20B tokens): PPL and downstream avg improve across MHA/GQA/MQA (+0.6–1.2 pts); the gated variant ("Gate") best (42.88 avg). NIAH retrieval 97.4 vs 82.6 at 1K (within window), 41.9 vs 25.9 at 4K (2× extrapolation). Token efficiency 1.42× (small) / 1.16× (large) at matched loss.
- **The paper's own caveats**: results "do not isolate the contribution of its structure from the increase in parameter capacity"; "reduced value-construction arithmetic does not uniformly translate into lower latency". Follow-ups (SWA, linear attention, MoE, larger) listed as ongoing.
- **Lineage**: Value Embedding (KoszarskyB 2024, modded-nanogpt) → DeepEmbed (BoPeng 2025) → PLE (Gemma 3n) → STEM (2601.10639) → Engram (2601.07372) → MA. All TRAINED supplements (Value Embed/PLE/STEM/Engram) or trained replacements (MA). None fits tables on frozen checkpoints.

## 2. Distillation

### 2.1 Vocabulary translation (paper ↔ codebase)

| Paper term | Codebase equivalent (grep set) |
|---|---|
| token-indexed memory table E_l | `EngramTable`, `FrozenProductKeyMemory`, anchor tables (Issue 882 `ā`), embedding table, `attn_v`-optional GGUF (`attention_k_eq_v`) |
| value construction V = K + M | `attn_wv` GEMV, `fold_gamma_into_rows`, `dequant` + add epilogue |
| norm folded into tables | `TransformerWeights::fold_gamma` (Plan 160), `fold_gamma_into_rows` (riir-gpu) |
| inverse positional rotation | `RopeAction::apply_inverse_at` / `PositionGroupAction` (`position_group_action.rs`, roundtrip-tested) |
| MA-Recall (no V cache) | K=V sharing (`attention_k_eq_v`), `kv_sink_window`, tiered KV |
| CPU offload + prefetch | `ZipfianCacheHierarchy` tiers, `EngramHotSwap`, plasma→hot→warm tiering |
| token efficiency ρ_token | loss-curve crossing (free from training logs) |

### 2.2 What ships vs what MA adds — signal-diff per cousin (§3.6 discipline)

- **K=V projection sharing (Research 165 / arXiv:2606.04032)**: `V := K` — consumes K only, **drops the per-token residual entirely** (measured cost 2.5–3.1% PPL; K/V cosine 0.73). Signal-diff: 165's variant consumes `{K}`; the fitted table consumes `{K, token ID}` and returns exactly the dropped component's per-token mean. The loader seam exists in production: riir-infer `gguf_loader.rs` handles Gemma-4's `attention_k_eq_v` (V optional → V=K). **MA explains WHY K=V works and what it loses.**
- **PKM / Memory Layers (Research 511, Plan 408)**: lookup-based capacity with **deterministically constructed** values (`PkmEpisodicStore` δ-rule, `hebbian_kernel_memory`) — the *legality precedent*: fitted tables are the same sanctioned class (constructed, not trained; BLAKE3-committable via `EngramTableId`; hot-swappable). Addressing differs (PKM = learned keys √N; MA = token ID direct; ours = token ID direct, fitted not trained).
- **RoVE (Research 452 / Plan 557)**: rotates values into the query's frame post-softmax via `apply_inverse_at` — different mechanism (rotation vs additive table), **same primitive reused** for P3 reconstruction.
- **Massive activations / sink-aware KV quant (Research 487)**: sink tokens are the degenerate case where a token's mean IS the outlier — the per-token table is the generalization of sink-exemption to all tokens.
- **Outlier-token KV quant (Su et al. 2025, ACL; RotateKV; KVTuner)**: *excludes/replaces* outlier values with group means to protect quant stats; ours **decomposes** (subtract per-token mean at write, add back at read) — no token is excluded, the whole range shrinks. Mechanism-level diff named; cite as closest quant cousin.
- **Differential anchor scoring (Research 586 / Issue 882)**: `q̂ = q − λ·ā` on the Q side; this note is `v̂ = k + λ·ē` on the V side. Same estimator family (per-token per-layer means, λ by direct evaluation never GD, λ=0 bit-identical, James-Stein tail). **One table-builder substrate, two consumers** — the fusion F1.
- **Engram (Research 278, Plan 299)**: n-gram-addressed slots + kernel-gated fusion (`σ(dot(q,k)/τ)`); MA is vocab-addressed + unconditional additive. Different retrieval and integration; same table substrate family (`EngramTableId`, `ZipfianCacheHierarchy`).

### 2.3 Prior-art surface (the §4 sweep, 2026-09-24)

Searches: headline ("Memory Attention" token-indexed value projection), component (value embedding / value projection removal / lookup values), quant angle (per-token mean removal KV cache outlier), family (Value Embed / DeepEmbed / PLE / STEM / Engram — cited by the paper itself). Findings: the family is entirely trained; **no published work fits per-(layer, token) value-residual tables on frozen checkpoints**; closest are (a) K=V sharing's post-hoc weight-merging (2606.04032), (b) outlier-token exclusion quant (Su et al. 2025). Our claim is the **integration + estimator** (closed-form fitted retrofit with shrinkage + the offline R² go/no-go), not a new principle — the additive-decomposition principle is the paper's; CSLS-style per-candidate anchors and the K=V evidence are published. Sweep honesty: two search phrasings per angle, not exhaustive; the mechanism-level deltas above are the defense.

### 2.4 Fusion candidates

- **F1 (primary): 882 × 587 — one fitted-anchor-table substrate, two consumers.** The calibration pass that fits V-side residual tables also fits Q-side anchor tables (Issue 882 P0). One module (`katgpt-core`), two sockets (score surfaces; value pathway). Force multiplier across the retrieval/rerank lane and the serving lane.
- **F2: engram commitment.** Tables as `EngramTableId`-committed artifacts, top-K residency via `ZipfianCacheHierarchy` (the storage dial), atomic swap via `EngramHotSwap`. Freeze/thaw-legal by construction (the Plan 408 precedent class).
- **F3: KV-quant lanes.** P1 (mean-removed V cache) composes with KVarN (159) and sink-aware handling (487): mean-removal → sink-exemption → quant, orthogonal stages.
- **F4: the serving league decode cell** (the tg128 bandwidth-bound row): P3 halves KV bytes/token read at decode in every grouped-KV arch — a direct bandwidth-axis lever on that cell. Exact current standings live in the private league doc (riir-ai), not here. Prefill (owner priority #1) is untouched (P2's W_V removal is a prefill GEMM saving of ~3%, paper-measured).
- **F5: prefetch offload** — only relevant if a trained MA model ever ships (Plan 418); the derived (token, layer) prefetch schedule consumes engram tiers. Parked with 418.

### 2.5 Public vs private

Public (katgpt-rs): the fitted-table estimator, variance laws, cache laws, the R² dashboard, `RopeAction` consumption, Issue 883's primitives. Private (riir-ai/riir-infer): league wiring, model-specific tables, serving integration. Training recipes: riir-train Plan 418.

## 3. Verdict

**Dual-track Gain (GOAT-tier on the modelless side; not Super-GOAT).**

- **Q1 (no prior art?)**: holds for the *fitted retrofit* mechanism (family is all trained; sweep found no fitter); the trained family itself is dense prior art for the architecture.
- **Q2 (new behavior class?)**: **NO** — efficiency/quality improvements on capabilities that exist (cache halving exists at K=V; quant improvement exists at outlier-exclusion). → kills Super-GOAT.
- **Q3 (selling point?)**: partial — "our engine halves the value cache of any checkpoint with a fitted table and knows offline whether it will work" is a league selling point, not a product headline.
- **Q4 (force multiplier?)**: YES — 882 (Q-side twin), engram (substrate), KV-quant lanes, serving league (≥2 pillars).

**Per-track verdicts (no cascade):**
- **Track (a) modelless — Gain → Issue 883** (PoC-first; §3.6 discipline: NO quality-parity claim is made here — the fitted retrofit's quality is UNKNOWN until the PoC; the only provable claim is P1's variance bound). The R² dashboard gates everything; a measured ρ_l ≈ 0 outcome is a legitimate recorded negative ("K=V sharing was already optimal for this checkpoint").
- **Track (c) model-based — Gain → riir-train Plan 418** (SECONDARY, queued behind C13 + 417; E3 from-scratch owner-gated per the 417 precedent; E0 parity probe 0.5–2 GPU-h is the cheapest first mover; E1 GDN-MA 12–18 GPU-h claims the paper's own unfinished business on our priority model family).

**MOAT gate**: katgpt-rs — transformer-stack inference primitive ✓ (attention/KV/quant-aware inference slot). riir-train — active training moat ✓ (queued). riir-ai — consumer (tg128 cell) ✓. Not riir-chain/ndb (no commitment/sync angle beyond engram commitment, which ships).

**Why not PASS**: nothing ships — the fitted-table estimator, the variance dashboard, and the reconstruction path are all absent; K=V sharing exists only as a loader fallback (`attention_k_eq_v`, Gemma-4's trained variant), not as a fitted retrofit; reverse-grep found the documented gaps (Issue 882 open P0 = the shared substrate's Q-side half).

**Honest caveats**: (1) the paper's quality gains are from-scratch training results — the retrofit's quality is a PoC question, not inherited; (2) per-layer tables cost `N·d_v` params/layer — top-K residency and λ-shrinkage make it a dial, but full-vocab tables on big-vocab models are GBs (only CPU/SSD-resident, 418's E2 shared-table variant); (3) tap-point consistency is a correctness trap (post-QK-norm gemma keys, pre-RoPE requirement); (4) never round-trip rotations (single one-directional rotate per read); (5) the paper is 1 day old, single-author, small-scale, self-caveated.

## 4. Actionable follow-ups

- [x] Issue 883 filed (modelless arm: R² dashboard → P1 quant → P2 retrofit → P3 cache halving) — filed in THIS change; the commit carrying this note is its record.
- [x] riir-train Plan 418 filed (E0–E5 portfolio, SECONDARY, E3 owner-gated) — same commit.
- [x] Issue 882 P0 shares the table-builder substrate with 883 P0 — LANDED 2026-09-25 as `katgpt-core/src/fitted_anchor_table.rs` (`StreamingMeanTable`, opt-in `fitted_anchor_tables`; Bench 886 G4) with both consumers wired: 882 `differential_anchor.rs` + 883 riir-infer `vk_calibration`.

## 5. References

- Kang, J. "Memory Attention" [arXiv:2609.28399](https://arxiv.org/abs/2609.28399) (2026).
- "Do Transformers Need Three Projections?" (Q-K=V) [arXiv:2606.04032](https://arxiv.org/abs/2606.04032) — Research 165.
- Berges et al. "Memory Layers at Scale" [arXiv:2412.09764](https://arxiv.org/abs/2412.09764) — Research 511 / Plan 408.
- Cheng et al. "Engram: Conditional memory via scalable lookup" [arXiv:2601.07372](https://arxiv.org/abs/2601.07372) — Research 278.
- Sadhukhan et al. "STEM" [arXiv:2601.10639](https://arxiv.org/abs/2601.10639); Gemma 3n PLE (Google, 2025); DeepEmbed (Peng, 2025); Value Embeddings (KoszarskyB, 2024) — the trained family, cited by MA §1.
- Su et al. "Accurate KV Cache Quantization with Outlier Tokens Tracing" (ACL 2025) — closest quant cousin (exclusion vs decomposition).
- Diff Transformer [arXiv:2410.05258] — Research 586 / Issue 882 (the Q-side twin).
