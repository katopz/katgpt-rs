# Research 349: MGDM — Multi-Granularity Diffusion Modeling (Subgoal-Prioritized Discrete Diffusion)

> **Source:** *Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning* — Ye, Gao, Gong, Zheng, Jiang, Li, Kong, ICLR 2025, [arXiv:2410.14157](https://arxiv.org/abs/2410.14157)
> **Date:** 2026-06-29
> **Status:** Done — Gain, **deferred** until dLLM inference hits the product roadmap.
> **Related Research:** 281 (Salience Tri-Gate — already covers modelless difficulty-prioritization), 300 (Closed-Unit Compaction Gate — already covers subgoal detection), 025 (LoRA raw/lora hot-swap — checked as modelless-unblock path, fails), 010/034 (ColaDLM/D2F — the dLLM substrate MGDM requires), 325 §7.2 G7 (the honest gate this note closes)
> **Related Plans:** 066 (D2F mini-dLLM research — micro-scale, NOT a product dLLM path), 303 (Salience Tri-Gate primitive — the modelless cousin that ships)
> **Classification:** Public

---

## TL;DR

MGDM shows discrete-diffusion LMs beat AR on planning-heavy tasks (Countdown, Sudoku, SAT) by decomposing hard subgoals into multi-view training targets, and adds a **token-level reweighting loss term** `v(x_t,n) = α(1−exp(−u))^β` to prioritize hard subgoals during training. The headline insight ("difficulty-prioritized subgoal emphasis") **already ships modellessly** via Salience Tri-Gate (281) and Closed-Unit Compaction Gate (300). What is genuinely new in MGDM is two pieces, both of which fail the modelless bar:

1. The token-reweighting loss (Equation 8) is a **training objective** requiring gradient descent on a discrete diffusion model. §3.5 modelless-unblock check fails on all three paths.
2. The inference-time easy-first TopK decoding requires a **trained discrete diffusion language model** that does masked-token prediction. We do not ship a dLLM inference path — Plan 066 is explicitly micro-scale research (6K params, vocab=32), and Research 325 §6 marks dLLM inference as "not now."

**Verdict: Gain — deferred.** Track in `.issues/` when the dLLM inference roadmap activates. Do NOT open a primitive, do NOT create a guide, do NOT gold-plate.

---

## 1. Paper Core Findings (verified by full read)

1. **Subgoal imbalance** (§3.1, Proposition 1): in AR modeling, subgoals at high *planning distance* (PD) require exponentially more data to learn. Empirically, AR plateaus at near-random accuracy for PD ≥ 2 unless LLaMA-7B-scale.
2. **Diffusion's multi-view effect** (§3.2): the ELBO of discrete diffusion (Equation 6) decomposes a hard subgoal into many easier "views" `x_t ∼ q(x_t|x_0)`, each a partial unmask. Loss curves (Figure 3) show diffusion loss on PD=3 subgoals is far lower than AR's.
3. **MGDM loss** (§3.3, Equation 8): `L_MGDM = Σ_n Σ_t w(t)·v(x_t,n)·u(x_0,x_t,n;θ)` where `u` is per-token cross-entropy and `v(x_t,n) = α(1−exp(−u))^β` is the **token-level** reweighting (β > 0 emphasizes hard tokens). Stacks on top of the sequence-level `w(t)`.
4. **Easy-first TopK decoding** (Algorithm 2): at each timestep, reveal the top-`t/T` highest-confidence positions. Beats random decoding (Table 3).
5. **Results**: 91.5% Countdown-4 (vs AR 45.8%), 100% Sudoku (vs AR ~33%), beats AR on SAT as variables grow.

---

## 2. Why this is Gain (deferred), not GOAT or Super-GOAT

### 2.1 The modelless difficulty-prioritization insight is already covered

MGDM's transferable *insight* — "hard subgoals deserve more compute/emphasis than easy ones" — already ships modellessly in stronger, runtime-validated forms:

| MGDM axis | Shipped modelless cousin | Why cousin is stronger here |
|---|---|---|
| Token-level difficulty reweighting | **Salience Tri-Gate (R281)** — two-sigmoid gate over HLA activation `a` + zone-attention `z` + curiosity `c` | Runs at 20Hz per-NPC, no training, direction-vector projection; MGDM's reweighting is a *training* loss, not a *runtime* gate |
| Subgoal prioritization for compaction | **Closed-Unit Compaction Gate (R300)** — rubric-gated trajectory compaction (C1/C2/C3/N1 predicates) | Multi-predicate rubric at runtime; MGDM's "subgoal" is implicit in the diffusion objective |
| Difficulty-driven reasoning budget | **Breakeven Complexity Router (R218/250)**, **DDTree (R194)** | Runtime compute allocation by instance difficulty |

There is **no modelless primitive to extract** that isn't already shipped. The novelty that remains in MGDM is in the *training* and in the *dLLM inference* path.

### 2.2 §3.5 modelless-unblock check — all three paths fail

- **Path 1 (freeze/thaw)**: MGDM's gain is not a systematic bias correction — it is a fundamentally different model class (diffusion vs AR) with a different training objective. No corrected snapshot can turn an AR model into a diffusion planner.
- **Path 2 (raw/lora reader-writer hot-swap)**: a deterministically constructed LoRA can correct a characterizable bias (e.g., scale-by-0.5 for doubled signals). MGDM's gain is not a bias — it is the absence of a bidirectional masked-prediction model. No closed-form overlay produces it.
- **Path 3 (latent-space projection)**: latent correction fixes *outputs* of an existing model. MGDM requires the underlying model *to be a trained dLLM*; no projection substitutes.

→ Genuine **training + dLLM-infra dependency**. The training loss → riir-train; the inference path → blocked on Plan 066 scaling past research.

### 2.3 The dLLM-roadmap dependency (the real gate)

Even setting aside training, MGDM's inference requires a trained dLLM that performs masked-token prediction. Our state:

- **Plan 066** is explicit: "Build a mini dLLM from scratch ... to answer the research questions." Scale: vocab=32, block=16, n_layer=1-2, ~6K params. Not a product path.
- **Research 325 §6** explicitly defers the dKV-Cache primitive with the same gate: "track when dLLM inference hits the product roadmap. Not Super-GOAT; not GOAT; not now."
- **Research 034 (D2F)** is the closest distillation; its Phase 2 ships inference infra but only for the mini research model.

Per assignment: do not gold-plate lower-priority gaps gated on a roadmap we don't have.

---

## 3. What lands where, eventually (only if dLLM roadmap activates)

If a real dLLM inference path ships and we want MGDM's training-side contribution:

| Component | Repo | Trigger |
|---|---|---|
| Token-reweighting loss `v(x_t,n) = α(1−exp(−u))^β` | `riir-train` | When dLLM training becomes a product, not research |
| Easy-first TopK decoding (Algorithm 2) | `katgpt-rs/src/speculative/` behind `dllm` feature | When a non-micro dLLM lands; the inference infra (D2F module) already exists at Plan 066 Phase 2 |
| Subgoal-difficulty signal for runtime | `riir-ai/crates/riir-engine/src/cgsp_runtime/` curiosity + `SalienceTriGate` | **Already covered** by R281 — no action |

None of this is built now. The inference-side easy-first decoding is the only piece that would touch katgpt-rs, and it would land as an extension of the existing `src/speculative/d2f.rs` module, not as new infrastructure.

---

## 4. Verdict

**Gain — deferred until dLLM inference hits the product roadmap.**

**One-line reasoning:** MGDM's modelless difficulty-prioritization insight is already shipped (R281 Salience Tri-Gate, R300 Closed-Unit Compaction Gate); the rest is a training loss (→ riir-train) and an inference path (→ blocked on Plan 066 scaling past research). The §3.5 modelless-unblock check fails on all three paths because MGDM is a fundamentally different model class, not a correctable bias. Do not implement, do not plan, do not create a guide.

**No artifacts created in this session beyond this note** (correct for Gain/Pass tier per skill §1.5 anti-deferral rule). Track re-activation when dLLM inference roadmap opens.

---

## 5. Paper metadata

| Field | Value |
|---|---|
| Authors | Jiacheng Ye, Jiahui Gao, Shansan Gong, Lin Zheng, Xin Jiang, Zhenguo Li, Lingpeng Kong |
| Affiliations | HKU + Huawei Noah's Ark Lab |
| Venue | ICLR 2025 |
| Code | https://github.com/HKUNLP/diffusion-vs-ar |
| Model scale | 6M / 85M / 303M (GPT-2 architecture), 7B/13B LLaMA for AR baselines |
| Tasks | Countdown (3/4/5 digits), Sudoku (9×9), 3-SAT (5/7/9 vars) |
| Headline | 91.5% Countdown-4 (AR 45.8%), 100% Sudoku (AR ~33%); 6M MGDM beats 13B LLaMA |
