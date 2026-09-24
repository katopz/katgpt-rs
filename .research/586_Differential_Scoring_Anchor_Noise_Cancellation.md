# Research 586: Differential Transformer — Common-Mode-Rejection Scoring (the Third Arm of the Attention-Noise-Control Family)

> **Source:** Ye, Dong, Xia, Sun, Zhu, Huang, Wei, "Differential Transformer", [arXiv:2410.05258](https://arxiv.org/abs/2410.05258) (ICLR 2025 Oral, Microsoft Research + Tsinghua).
> **Date:** 2026-09-24
> **Status:** Done — verdict 🟡 **Gain / GOAT-tier, dual-track** (NOT Super-GOAT; per-track verdicts below). Modelless execution → `Issue 882`; perception consumer → riir-ai Issue 1006; training recipes → riir-train Plan 417 (**SECONDARY**, queued behind C13 per the TTPO serving-envelope rule + owner priority call "riir-train is lowest priority").
> **Related Research:** 549 (ASEntmax — the entmax arm of the same softmax-dilution family; the precedent note this one mirrors), 392/Plan 411 (SSMax — the softmax rescale arm, shipped), 068/Plan 106 (DashAttention α-entmax routing, shipped default-on), 351 riir-ai (KBLaM dilution — the 1/M sigmoid-denominator cousin + the `σ(s_max − s_anchor)` refusal composite, the shipped differential-adjacent score), 086 riir-train (outlier collapse / quant-robust LoRA — the outlier axis is a tracked interest), 444 riir-train (RAT+ dense-train sparse-infer)
> **Related Plans:** 411 (SSMax SHIPPED), 106 (DashAttention SHIPPED), riir-train 417 (recipes, SECONDARY)
> **Cross-ref:** riir-ai Issue 1006 (habituation perceptor consumer); riir-clippy `src/draft/latent_matcher.rs` rerank (anchor-scoring consumer #1, fusion priority #2); riir-infer attention kernels (SNR-probe consumer, the prefill league)
> **Classification:** Public

---

## TL;DR

Diff Transformer cancels **attention noise** (softmax mass on irrelevant context) by computing attention as the *difference of two softmax maps* — `DiffAttn(X) = (softmax(Q₁K₁ᵀ/√d) − λ·softmax(Q₂K₂ᵀ/√d))V` — the differential-amplifier / noise-canceling-headphones principle: subtracting two correlated signals cancels their common mode. Trained from scratch it needs ~65% params/tokens for parity, wins multi-needle retrieval by 30%, cuts attention-logit outliers 8.2× (→ 4-bit attention-logit quant ≈ 6-bit vanilla), and costs only −5..−12% throughput (two flash calls + subtract). **The trained architecture is not retrofit-able onto Bonsai/Qwen GGUF weights — the paired Q₁/Q₂,K₁/K₂ structure is baked into training** (and DEX, NeurIPS 2025, already owns the lightweight-adaptation retrofit; DiffLoRA arXiv:2507.23588 owns the adapter retrofit — prior art, both).

**What IS ours is the third arm of a family we already ship two arms of.** Research 549 documented the softmax-dilution duality: softmax leaks mass (→ SSMax sharpen-UP ∝ log n, shipped), entmax over-sparsifies (→ ASEntmax damp-DOWN ∝ (log n)^−0.5, shipped as Issue 747). Diff Transformer is the **third mechanism: subtract a reference map instead of rescaling**. On surfaces where WE own the scores (retrieval/routing/abstention — no trained weights involved), the differential arm is a one-axpy primitive: **`q̂ = q − λ·ā` (anchor-corrected query; candidates similar-to-everything lose their generic mass)** plus a differential **habituation perceptor** (`n_t = s_t − λ·EMA(s)`, DC gain exactly `1−λ` — the paper's own headwise `(1−λinit)` multiplier isomorphism) for NPC novelty perception.

**Distilled for katgpt-rs (modelless, inference-time):**
1. **`differential_anchor` scoring** — `score(q,d) = sim(q,d) − λ·sim(ā,d) = (q − λā)·d`: one axpy → corrected query → existing KNN pass unchanged. Anchor = frozen corpus centroid / mean-query. λ* by direct evaluation (grid on oracle fixtures), never GD. → Issue 882 P0.
2. **Differential habituation filter** — first-order high-pass on per-NPC perceptor channels; constant input → output `(1−λ)s`; settling law `t_ε = ln(1/ε)/ln(1/β)` closed-form. → riir-ai Issue 1006.
3. **Frozen `λinit` table** — `λinit(l) = 0.8 − 0.6·exp(−0.3(l−1))` (0.2 → 0.8, residual halves ≈ every 2.31 layers) as a `const` table + the **neutral-at-zero reparameterization envelope** (`λ = e^{u}−e^{v}+λinit`: latent-at-0 ⇒ prior — the pattern for any runtime-tunable seeded by a frozen prior). → Issue 882 P0 rider.
4. **Streaming attention-SNR accumulators** — exact softmax entropy / participation ratio maintained *inside the online-softmax loop* (rescale-corrected `T ← e^Δ(T + Δ·l_old) + Σ e^{x−m}(x−m)`) → per-head measured sharpening τ_h on the SSMax socket; row-relative sink-exempt logit clamp → tighter dynamic-quant scales; differential KV-eviction specificity `max_q(a_j − λ·μ_j)`. → Issue 882 P1–P3 (riir-infer consumers; prefill-league adjacent).
5. **Attention-to-answer eval harness** — `m_Y = mean attention mass on labeled answer spans` (the paper's Table-3 instrument) as the falsifier for items 3–4 on our own served models. → Issue 882 P4.

---

## 1. Paper Core Findings

1. **Mechanism** (§2.1): `[Q1;Q2]=XW_Q, [K1;K2]=XW_K` split projections; `DiffAttn(X) = (softmax(Q1K1ᵀ/√d) − λ·softmax(Q2K2ᵀ/√d))V`; λ learnable per layer (shared across heads), reparameterized `λ = exp(λq1·λk1) − exp(λq2·λk2) + λinit`, `λinit = 0.8 − 0.6·exp(−0.3·(l−1))`. Headwise RMSNorm (GroupNorm) per head × fixed `(1−λinit)` multiplier aligns gradient flow with vanilla Transformer (Appendix G proves gradient equivalence up to constants → **vanilla hyperparameters transfer directly**). `h = d_model/2d` heads align params/FLOPs. FlashAttention reuse: two calls + subtract (−5..−12% throughput).
2. **Scaling** (§3.2): ~62–65% params/tokens for parity (6.8B ≈ 11B vanilla; 160B ≈ 251B tokens).
3. **Retrieval** (§3.4): multi-needle 4K at N=6,R=2: 0.85 vs 0.55 (+30%); 64K N=8: stable across lengths, +76% at 25% depth. **Attention-to-answer 0.27–0.40 vs 0.03–0.09; attention noise 0.01–0.02 vs ~0.5** — the SNR mechanism measured directly.
4. **ICL robustness** (§3.5): order-permutation margin 4.0 vs 19.0 (random) / 13.4 vs 56.7 (alternating) — variance control, not just mean.
5. **Hallucination** (§3.6): summarization +0.09..+0.19 free-of-hallucination; QA +0.07..+0.11 (GPT-4o judged).
6. **Outliers / quantization** (§3.7): attention-logit top-1 38.8 vs 318.0 (8.2×), hidden-state top-1 1688 vs 3608 (2.1×); absmax-quantized attention logits: Diff holds to 6-bit, **4-bit Diff ≈ 6-bit vanilla (+25% at 4-bit)**.
7. **Math reasoning** (App. C): +7.5% avg over 8 math benchmarks after R1-style distillation; shorter reasoning traces (6144 vs 6913 tokens).
8. **Mechanism analyses**: Naderi et al. 2024 — differential attention balances the attention-matrix spectral distribution, resolving rank collapse. Conclusion names **KV-cache compression from emergent sparsity** as future work (not done in the paper).

## 2. Distillation

### 2.1 The family framing (why this snaps into shipped substrate)

| Arm of the attention-noise-control family | Mechanism | Ships here? |
|---|---|---|
| **Rescale** (temperature) | sharpen softmax ∝ log n as scored set grows | ✅ SSMax, Plan 411 (`s_L·log N`) |
| **Sparsify** (threshold) | α-entmax exact zeros; damp ∝ (log n)^−0.5 | ✅ DashAttention Plan 106 + ASEntmax Issue 747 |
| **Subtract** (differential) | cancel common mode via a second correlated map | ❌ nothing ships — **this note** |

On trained-weight surfaces (Bonsai/Qwen GGUF) the third arm is closed to us (weights-conditioned; DEX/DiffLoRA own the adaptation retrofits). On **score surfaces we own** (healer rule-selection, riir-rag KNN, neuron-db retrieval, kg attention, reflex abstention, DashAttention routing scores), the arm is open and is a 30-LOC primitive.

### 2.2 Signal-diff vs shipped cousins (§3.6 discipline)

| Shipped cousin | File | Signal it consumes | Paper component's signal | Verdict |
|---|---|---|---|---|
| `RerankMode::Structural` | riir-clippy `src/draft/latent_matcher.rs` | structural position of spans in the merged pool | **hubness** — similarity of a candidate to everything (common mode) | **Uncovered** — no term penalizes generic similarity |
| `apply_ssmax_inplace` | katgpt-core `ssmax.rs` | `log_n` + rolling Δ̂ (global, scheduled) | measured per-head concentration (entropy/PR) | **Partial** — scheduled-global vs measured-per-head; the socket exists, the feedback does not |
| `head_sparsity_profile` | katgpt-attn `dash_attn/sat_analysis.rs` | intra/inter segment attention mass (offline, n×n matrix) | streaming per-row entropy/PR inside the kernel loop | **Partial** — same question (per-head stats), different substrate (materialized matrix vs streaming); our serving kernels never materialize n×n |
| Stealth wear-off | riir-game-sdk `riir-stealth/src/alarm.rs` | decay of **accumulated alarm state** post-trigger ("alarm halves every half_life_ticks") | high-pass on the **raw perceptor** pre-trigger (sustained → ignored, change → fires) | **Uncovered** — different mechanism, complementary behavior (forget-after vs ignore-while-constant) |
| `σ(s_max − s_anchor) < τ` refusal | riir-ai Issue 765 / `kg_pkm_attention` | anchor-subtracted max score (differential **refusal**) | anchor-subtracted **ranking** (same operator, different consumer) | **Partial** — the differential form ships for abstention only; ranking never subtracts |

### 2.3 Path 0 inventory (component → coverage → extraction → disposition)

| Component | Coverage | Extraction | Disposition |
|---|---|---|---|
| Differential attention as trained architecture | no | **no** — weights-conditioned; DEX (NeurIPS 2025) + DiffLoRA (2507.23588) own the retrofit/adaptation path | riir-train Plan 417 (SECONDARY recipes) |
| Common-mode-rejection scoring `q̂ = q − λā` | no (nothing subtracts in ranking) | **yes** — one axpy + existing KNN; λ* by grid on oracle fixtures | Issue 882 **P0** |
| λinit layer schedule + neutral-at-zero reparam envelope | no | **yes** — const table, ~20 LOC | Issue 882 P0 rider |
| Streaming entropy/PR accumulators (online-softmax loop) | partial (sat_analysis offline; effective_rank offline) | **yes** — 5 extra FLOPs/element, registers only | Issue 882 **P1** |
| Per-head measured sharpening τ_h (bisection to target H_n) | partial (SSMax global scheduled) | yes — measured feedback on the shipped socket | Issue 882 P1 (prior art exists at per-token level: "Inference-Time Attention Calibration" Jan 2026; head-entropy-as-signal: OpenReview head-entropy correctness paper — integration novelty only, positioned honestly) |
| Row-relative sink-exempt logit clamp → dynamic-quant scales | no (kernels accumulate f16; no logit-domain outlier control) | yes — closed-form envelope `≤ 2·pᵢ·w_h/254` | Issue 882 **P2** (low-bit attention path; prefill-league adjacent) |
| Differential KV eviction `specificity_j = max_q(a_j − λμ_j)` | partial (eviction_window, sink-aware KV ship) | yes — bookkeeping only, no kernel change | Issue 882 **P3** |
| Attention-to-answer metric `m_Y` | no | yes — offline dump + span masks | Issue 882 **P4** (the falsifier for P1/P2) |
| Habituation perceptor `s_t − λ·EMA(s)` | no (stealth wear-off differs, see §2.2) | **yes** — DC gain `(1−λ)` exact, settling law closed-form | riir-ai Issue 1006 |
| ICL order-permutation robustness | n/a | modelless half = canonicalization (spread→0 by construction) + P-fold permutation ensembling; the paper's 4-point floor is the model's, ours is the 19-point variance | Issue 882 P4 rider (canonical context assembly in eval harnesses) |
| Spectral rank-1 deflation `A′ = A − λσ₁u₁v₁ᵀ` (Naderi angle) | partial (effective_rank ships as diagnostic) | yes — power iteration on the rerank-stage M×M candidate affinity, gated on erank < θ (only deflate a collapsed matrix) | Issue 882 P4 stretch |
| 65%-params parity / −5..−12% throughput at matched quality | no | **no** — weights-conditioned | Plan 417 records the claim as the drafter-economy framing |

### 2.4 Fusion

- **× Research 549 (ASEntmax) × Plan 411 (SSMax)** — completes the three-arm family; the differential arm is the only one expressible as *query pre-correction* (`q̂ = q − λā`), so it composes with BOTH other arms (rescale the corrected query's scores; entmax-threshold them) rather than competing for the same socket.
- **× Research 351 (KBLaM/PKM refusal)** — one primitive, two consumers: anchor-subtracted **ranking** (this note) + anchor-subtracted **abstention** (shipped `σ(s_max − s_anchor)`); the kg-attention path can adopt the same anchor table.
- **× healer corpus (fusion priority #2)** — the highest-value consumer: 580+ rule corpus, hashed-bag embeddings, oracle-labeled fixtures with measurable downstream top-1 (`retrieval_eval.rs`, `clippy_oracle.rs`). Known trap (advocate's caveat, adopted): hubness ≠ illegitimacy — mechanical lints are *correctly* high-frequency; the anchor must be adjudicated per-domain on fixtures (mean-query anchor vs mean-corpus anchor A/B), never assumed.
- **× reflex abstention** — `CorpusDistanceGate` + fused abstain consume corpus scores; an anchor-corrected score is a drop-in sharpening candidate for the abstention signal (katgpt-core consumer, zero new deps).
- **× DashAttention routing** — routing scores are chunk-summary heuristics we own (quality-safe to transform, unlike trained-weight attention — the Research 549 §2.4 precedent verbatim).

### 2.5 Honest caveats

1. **Hubness ≠ illegitimacy** (the P0 trap): rules/spans that are correctly high-frequency get suppressed by a corpus-centroid anchor. Gate on oracle fixtures per domain; mean-query anchor is the safer default; λ=0 must stay byte-identical (kill switch).
2. **Concentration ≠ relevance** (the P1 trap): entropy measures concentration, not correctness — a head can be confidently wrong; sharpening amplifies error. The m_Y harness (P4) is the falsifier, and per-head prior art exists at the per-token level (Jan 2026) — our claim is the streaming/measured-per-head integration, not a new principle.
3. **Sinks are legitimate outliers** (the P2 trap): the 318-magnitude first-token sinks are load-bearing; the clamp MUST exempt the shipped sink set or it silently degrades long-context quality.
4. **"Generically attended" ≈ "consistently relevant"** (the P3 trap): differential eviction confuses common-mode mass with durable importance until the query distribution shifts; recency priors partially cover, recall failures are silent.
5. **λinit is one training run's hyperparameters, not a law** — ship as an override-able prior, never a hard default.
6. **Habituation filter habituates to slow real threats** — a predator creeping below the β-cutoff reads as DC and gets cancelled; threat channels keep conservative λ (0.2-class), ambient channels cancel hard (0.8-class).

## 3. Verdict

**🟡 Gain / GOAT-tier — dual-track, NOT Super-GOAT.** Modelless: Issue 882 (katgpt-rs, P0–P4) + riir-ai Issue 1006. Model-based: riir-train Plan 417 (SECONDARY — recipes recorded, queued behind C13; the cheap decisive experiment is the 60–130M diff-vs-vanilla drafter pair at ~16–26 GPU-hrs with MQAR + code-draft gates).

**One-line reasoning:** the trained architecture is weights-conditioned and its retrofits are owned by DEX/DiffLoRA (prior art), but the differential arm of the attention-noise-control family — two arms of which already ship here (SSMax rescale, ASEntmax sparsify) — is a ~30-LOC modelless primitive on every score surface we own (healer rerank, rag/neuron-db retrieval, kg attention, reflex abstention), plus a closed-form habituation perceptor with an exact DC-gain isomorphism, plus three serving-side diagnostics on the prefill-league path.

**Why NOT Super-GOAT (per-track, no cascade):**

| Q | Modelless track | Model-based track |
|---|---|---|
| Q1: No prior art? | **NO** — contrastive decoding (arXiv:2210.15097 + DExperts/DoLa/SCD lineage) owns subtract-a-weaker-signal at logits level; CSLS (arXiv:1804.07745) + inverted-softmax + hubless-NN own score-level hubness subtraction; inference-time attention-temperature-via-entropy exists (Jan 2026, per-token). Our deltas are integration-shaped (single-shared-anchor O(d) form; streaming in-kernel; per-head on the SSMax socket). | **NO** — DEX (NeurIPS 2025) owns the pretrained retrofit (adaptation, <0.01% tokens); DiffLoRA (2507.23588) owns the adapter form; Shared DIFF (2501.17900) owns parameter sharing; Diff Transformer v2 (2026) extends λ. |
| Q2: New behavior class? | NO — better ranking/abstention/diagnostics on existing capabilities. (Habituation's "ignore-while-constant, react-to-change" is new *mechanism* for us but the behavior family is stealth-adjacent.) | NO — a matched-quality smaller drafter is an economy claim. |
| Q3: Product selling point? | WEAK — "our engine measures and cancels attention noise on frozen checkpoints" is defensible but unproven on quality; the anchor-scoring win is internal ranking quality. | WEAK — 65%-parity at 3B+ does not transfer to 0.5B continual runs unmeasured. |
| Q4: Force multiplier? | YES — healer rerank + rag + neuron-db + kg attention + reflex + DashAttention routing + KV manager + eval harness. | YES — C13 lane, distill_attention.rs seam, quest_grammar QLoRA configs. |

Q1 fails both tracks ⇒ GOAT-tier ceiling; Q4 strong ⇒ worth filing with issues + a recipe plan, fusion-first.

**MOAT gate (katgpt-rs):** attention/retrieval-stack inference primitive via fusion with two shipped plans (411, 106) — in scope, public, no game/chain/shard semantics. ✓ The habituation consumer files in riir-ai (adaptive-NPC pillar). ✓

**Track priority (TTPO serving-envelope rule):** PRIMARY = modelless (Issue 882 P0 lands on the healer rerank hot path — fusion priority #2; P1–P3 on the serving path); SECONDARY = Plan 417 (training; riir-train lowest priority + no near-term serving consumer for a diff checkpoint — GGUF-nonstandard, −5..−12% throughput against the beat-llama.cpp league).

**Discard audit (§3.5 — every advocate finding filed or discarded with mechanism-level reason):**
- *Fused FlashDiffAttn kernel (shared K/V tiles, shared running max)* — discarded as work item: **no differential checkpoint exists in the model zoo** (Bonsai/Qwen never take this path); recorded as a dormant pattern-bank entry in the note + Plan 417 rider. Not discarded as knowledge: the **cancellation-must-accumulate-in-f32 law** (`(‖s₁‖+λ‖s₂‖)/‖s₁−λs₂‖` amplification) is recorded beside the kernel documentation plan in Issue 882 P2.
- *Permutation ensembling (P-fold)* — discarded as a default: P× forward cost on the serving path; recorded as an offline/high-stakes tool rider in P4.
- *13B-scale anything / 1T-token pretraining* — discarded: out of scope by policy (fine-tuning/adapters/recipes only; full pretraining out of scope).
- *Bonsai-27B as distillation teacher* — discarded on measured economics: ~200+ GPU-hrs on the 4090 for a 1B-token teacher pass; Plan 417 routes teacher duty to the M3 B6 trace line instead.
- *GroupNorm-headwise as fine-tune stability knob (6b)* — kept as a Plan 417 P3 rider arm (2–6 GPU-hrs, non-inferiority gate), not a standalone plan.

**Prior-art sweep (§4, 20 queries run — see panel reports in the verdict trail):** headline + components + selling-point framings + surveys, all via the search sub-agent with citations. Novelty downgrades applied BEFORE this verdict: DEX, DiffLoRA, Shared-DIFF, Diff-v2, CSLS-class, contrastive-decoding family, SmoothQuant/outlier lineage, Inference-Time Attention Calibration, head-entropy-as-signal. **Open cells found (honest reading: thin, integration-shaped):** inference-time reference-subtracted attention maps (nearest = DEX, adaptation-based); the diff→outlier→low-bit-attention-quant chain (each link separately established, chain unclaimed); single-shared-anchor retrieval correction (CSLS-adjacent); `signal − λ·EMA` habituation in game agents (neuroscience concept established: Huber & O'Reilly 2003, EEG 2020).

**PASS-Redirects (synthesis):** none — verdict is Gain, not PASS.
