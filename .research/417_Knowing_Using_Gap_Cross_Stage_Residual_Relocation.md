# Research 417: Knowing-Using Gap → Cross-Stage Residual Relocation Operator + Permeation-Map Diagnostic

> **Source:** [Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning](https://arxiv.org/abs/2607.08393) — Lu Dai, Ziyang Rao, Yili Wang, Hanqing Wang, Hao Liu, Hui Xiong (HKUST-GZ / HKUST, NeurIPS 2026 submission, 2026-07-09)
> **Date:** 2026-07-13
> **Status:** SHIPPED via Plan 431 (2026-07-13) — opt-in behind `cross_stage_relocation` feature. Diagnostic half + CUSTOM relocation operator both landed. Fixed-pair `LateEarly` default REFUTED by defend-wrong PoC (CLOBBERS in 2/4 clean configs); production path is diagnostic-guided `RelocatePair::Custom`. Stays opt-in until a real-game-domain PoC lands.
> **Related Research:** 259 (QK-Restore — closest cousin, Super-GOAT), 362 (HydraHead — activation-patching scorer, shipped), 313 (Thinking-to-Recall PASS — coherence-driven re-estimation is the latent analog), 290 (Latent Field Steering — direction-vector injection), 276 (PersonalityWeightedComposition — sigmoid-gated layer composition), 244 (FaithfulnessProbe — causal-intervention substrate), 388 (Jacobian-Lens — single-layer SVD readout, REFUTED as prefilter but the SVD concept readout survives)
> **Related Plans:** 431 (SHIPPED — Phase 1–4 COMPLETE, primitive stays opt-in per PoC verdict)
> **Classification:** Public

---

## TL;DR

The paper formalizes the **Knowing-Using Gap** (KU Gap): fine-tuned LLMs memorize new facts quickly (memorization saturates in ~2–10 epochs) but fail to *use* them in multi-hop reasoning for many epochs more (often never), with both an accuracy gap and a temporal lag. The mechanistic explanation is **knowledge-circuit misalignment**: facts get encoded in storage-friendly layers (early MLPs for storage; very late layers for direct-recall shortcuts) that are *not routed into* the mid-layer reasoning circuits. A diagnostic — **self-patching** — scans all (source_layer, target_layer) pairs and reports a "permeation map" of which source representations unlock the answer when relocated to which target layer. The actionable distillation: a **fixed two-pair heuristic** (source ≈0.8L → target ≈0.5L, plus source ≈0.1L → target ≈0.5L) **recovers 58–75% of the oracle headroom** — zero per-instance search, zero training.

**Distilled for katgpt-rs (modelless, inference-time):**

1. **Cross-Stage Residual Relocation Operator** — `RelocateOp { src_stage, dst_stage }` that, during a forward pass, snapshots the anchor's residual state at stage `src_stage` and overwrites the anchor's state at stage `dst_stage`. Two canonical pairings (`RelocatePair::LateEarly`) ship as defaults. **This is an applied operator (not a score)** — distinct from Plan 358's activation-patching *scorer* (`direct_effect_importance`). Closest shipped cousin is QK-Restore (R259, per-matrix composite); this paper adds the per-position-across-stages axis.

2. **Permeation-Map Diagnostic** — `permeation_scan(model, anchor, prompt, score_fn) -> Matrix<L_src × L_dst>` that produces the paper's heatmap. Reuses `direct_effect_importance` (R362/P358) as the cell score and adds the 2D scan + clustering (two-cluster pattern: early→mid, late→mid).

The fine-tuning dynamics, saturation-epoch analysis, and "natural gradient vanishes after memorization" findings are **training-only → riir-train**. Only the heuristic fix and the diagnostic scan are modelless.

---

## 1. Paper Core Findings

### 1.1 The Knowing-Using Gap (KU Gap)

Two complementary measurements after fine-tuning on injected fact triplets `(n₁, e₁₂, n₂)`:

- **Memorization accuracy** `A_mem(t)` — direct single-hop recall of the injected fact.
- **Generalization accuracy** `A_gen(t; T)` — multi-hop reasoning that requires the injected fact (e.g., "what drug targets the protein that is expressed in embryo?" requires chaining `protein → expressed → embryo` and `drug → targets → protein`).

Two decomposed gaps:
- **Accuracy gap** `ΔA = A_mem(T_max) − A_gen(T_max)` — at convergence, memorization stays near 1.0 while generalization stalls at 0.08–0.18 (chaining) on Qwen-2.5 / LLaMA-3.x.
- **Temporal lag** `ΔT = T_gen − T_mem` — generalization saturates 4–9 epochs after memorization (LoRA) or sometimes never (FFT on intersection).

**Scale result (Fig 2):** increasing model size does **not** eliminate `ΔT`. Increasing the number of injected facts *widens* `ΔA`. Storage scales; routing does not.

### 1.2 Self-Patching (the diagnostic)

A variant of activation patching (Plan 358 ships the underlying machinery):

```
For each layer pair (l_src, l_dst):
    Cache source state z = h^{l_src}_{T(P_s, E)}(P_s)   # anchor E's residual at l_src in source prompt
    Run P_t, replacing h^{l_dst}_{T(P_t, E)} ← z         # overwrite anchor at l_dst in target prompt
    Continue forward; record ΔI = I(patched, y*) − I(unpatched, y*)
```

Produces a `L × L` "permeation map" where cell `(l_src, l_dst) > 0` means: layer `l_src` contains a representation of the anchor that, when relocated to `l_dst`, increases the probability of the correct answer.

**Key contrast vs. causal tracing (Meng 2022, ROME) and Patchscope (Ghandeharioun 2024):**
- Causal tracing needs a *clean correct trajectory*; self-patching works on **failed generalization cases** (which is the regime of interest).
- Patchscope decodes source states into natural language via auxiliary prompts; self-patching **evaluates the target prompt's answer** (no decoding).

### 1.3 Knowledge-Circuit Misalignment Hypothesis

The permeation maps reveal:

1. **Pre-memorization**: all-blue map (no source representation helps — knowledge isn't there yet).
2. **At memorization saturation**: clear off-diagonal red regions appear — knowledge is stored in early/late layers, can take effect if manually routed to mid layers. **Diagonal cells still blue** — natural forward pass fails to route it.
3. **Successful generalization**: red region expands to cover the diagonal — natural routing catches up.
4. **Failed generalization**: red region expands but **halts before the diagonal** — natural gradient has vanished after memorization, can't drive the last-mile routing.

### 1.4 Two-Cluster Pattern (the actionable finding)

Effective patches concentrate in **two source clusters**, both targeting **mid layers** (~0.45L):

| Source cluster | Target | Why |
|---|---|---|
| Late layers (~0.8L) | Mid (~0.45L) | Intuitive — moving enriched information backwards brings in accumulated context. |
| Early layers (~0.1L) | Mid (~0.45L) | **Surprising** — moving forward from early storage also helps. Information is already stored at both ends but not aligned with mid-layer reasoning. |

Patching into late layers is useless (reasoning stream has moved to the last token by then per Geva 2023's three-step theory).

### 1.5 Fixed Heuristic (the modelless crown jewel)

Using only two predetermined layer pairs per architecture (no per-instance search):

```
(l_src, l_dst) ∈ { (⌊0.82L⌉, ⌊0.45L⌉), (⌊0.10L⌉, ⌊0.45L⌉) }
```

**Recovers 58–75% of oracle headroom across all 6 models × 2 domains** (STaRK-Prime biomedical, STaRK-MAG academic). On chaining (the hardest case): mean 0.121 → 0.357 (vs oracle 0.444). On intersection (easier): 0.808 → 0.926 (vs oracle 0.966).

### 1.6 Controls (rules out alternative explanations)

- **Token-position ablation (Table 5):** patching at the head-entity position (`0.64` mean Δ) ≫ `<EOS>` (`0.40`) ≫ relation tokens (`0.20`) ≫ `<BOS>` (`0.05`, n.s.). Knowledge is tightly tied to entity mentions, not position-agnostic.
- **CoT prompting (Table 6):** CoT improves chaining (0.078 → 0.132) but **far below** self-patching (0.440). Sometimes degrades intersection. Confirms the gain isn't a decoding effect.
- **Irrelevant-fact patching (Table 6):** patching with an unrelated fact's representation (0.150–0.194) ≪ self-patching with the correct fact (0.440–0.542). Rules out generic activation perturbation.

### 1.7 Limitations (paper-stated, §H.1)

- **Single anchor position only** — knowledge may distribute across multiple positions or be redundantly encoded; oracle results are a lower bound.
- **Diagnostic, not predictive** — identifies and partially repairs misalignment post-hoc, but doesn't yet forecast which facts will fail during early training.
- **Layer-level granularity** — finer localization to attention heads or MLP sublayers could refine the hypothesis.

---

## 2. Distillation

### 2.1 What's transferable (the modelless residue)

Strip the fine-tuning dynamics, the saturation-epoch tracking, the gradient-vanishing analysis. What remains:

> **A representation can be encoded in a stage where it's *stored* but not *used*. A deterministic, fixed-pair cross-stage relocation — copying the anchor's state from a late/early stage into a mid stage — recovers a large fraction of the lost capability. The relocation is an applied operator on the residual stream; it is not training.**

This is precisely the **path #2 (raw/lora reader-writer hot-swap with deterministic construction)** of the §3.5 modelless-unblock protocol: instead of *learning* a correction adapter, we *construct* a deterministic reader that pulls state from one stage and a writer that injects it at another. No gradient descent.

### 2.2 Prior-art surface (MANDATORY — what already ships, do not duplicate)

Vocabulary translation table (paper → codebase):

| Paper vocabulary | Codebase vocabulary | Ships? |
|---|---|---|
| "self-patching" (relocate residual state across layers) | "cross-stage residual relocation operator", "deterministic skip-connection", "stage-to-stage reader-writer hot-swap" | ❌ **Not shipped as an applied operator** |
| "activation patching" (score head importance) | `direct_effect_importance`, `indirect_effect_importance` (`crates/katgpt-core/src/causal_head_importance/patching.rs`, Plan 358) | ✅ Shipped — but as a *scorer*, caller-supplied forward pass. |
| "knowledge-circuit misalignment" (knowledge stored in wrong layer) | "dormant subspace" (R151), "stranded latent state", "coherence < tau_reest" (`riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs`), "QK drift" (R259) | ⚠️ Partial — shipped for *weights* (R259 per-matrix) and *latent state* (R313 re-estimation); not for *activations across stages*. |
| "permeation map" (L_src × L_dst scan) | "causal scan", "2D intervention heatmap" | ❌ Not shipped. |
| "two-cluster pattern" (early→mid, late→mid) | (no shipped analog) | ❌ Not shipped. |
| "memorization vs generalization" | "exact recall vs compositional use", "training-data fit vs transfer" | (training-time concept; not a runtime primitive) |
| "fine-tuning" / "knowledge injection" | (n/a — training-only → riir-train) | n/a |
| "head-entity position" (the anchor) | "anchor token", "belief-anchor", "Engram hash anchor" (R278) | ✅ Concept matches; the Engram anchor is the closest runtime analog. |

Closest-cousin ranking:

1. **R259 (QK-Restore, Super-GOAT)** — same family ("fine-tuning breaks knowledge, fix modellessly with a surgical composite"). R259 operates **per-matrix** (W_Q, W_K vs W_V, W_O); this paper operates **per-position-across-layers**. Different axis; same pattern. The two are complementary: QK-Restore preserves *routing geometry*; self-patching relocates *content representations*.
2. **R362 (HydraHead, GOAT — shipped Plan 358)** — same diagnostic family (activation patching). R362 scores *head importance* along the layer × head axes; this paper scores *position-relocation effectiveness* along the source-layer × target-layer axes. The cell-score function (`direct_effect_importance`) is **identical**; only the scan structure differs.
3. **R313 (Thinking-to-Recall, PASS)** — the latent-space analog of "knowledge stored but not used, recursively re-derive until it's reachable." `riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs` is the runtime mechanism that *triggers* when coherence decays; this paper is the diagnostic that *explains why* coherence decays (the latent state is stranded in the wrong stage).
4. **R290 (Latent Field Steering)** — direction-vector injection into the latent state. The closest shipped "applied relocation" primitive, but it injects a *frozen direction vector*, not a *runtime-captured stage state*.
5. **R276 (PersonalityWeightedComposition)** — sigmoid-gated layer composition with frozen direction vectors. Operates per-layer on the composition weights; doesn't relocate activations.

**The genuinely unshipped primitive:** an *applied* operator that snapshots a stage's residual state at runtime and overwrites a different stage's residual state. Plan 358 ships the *measurement* (`direct_effect_importance`); R259 ships the *weight-level composite*; R290 ships the *frozen-direction injection*. None ships the runtime activation relocation.

### 2.3 Latent-space reframing (the mandatory check)

Re-cast on each Super-GOAT factory module:

**(a) HLA per-NPC latent state** (`katgpt-core/src/sense/`): HLA's `evolve_hla` is a single-stage leaky integrator — there is no "source layer" vs "target layer" axis. The KU Gap doesn't directly apply. *However*, the latent_functor *chain* (multiple functor applications) has stages, and a stranded-latent-state pattern is conceivable (a direction vector that gets suppressed at stage k but would be useful at stage k+2).

**(b) `latent_functor/` operations** (`riir-engine/src/latent_functor/`): the **closest latent analog**. A functor chain `f_1 ∘ f_2 ∘ ... ∘ f_K` has stages. If stage k produces a representation that doesn't propagate to stage k+1 (because `coherence < tau_reest` triggers re-estimation before the representation can be used), that's the latent-space KU Gap. The existing `ReestimationScheduler` (R313) handles this reactively. The paper's contribution is the *diagnostic* — scanning all `(src_stage, dst_stage)` pairs to find where the representation is stranded — plus the *fixed-pair heuristic* for proactive relocation.

**(c) `cgsp_runtime/` curiosity signals**: the KU Gap maps to "curiosity signal exists but doesn't drive exploration" — a known failure mode. Self-patching would say: the curiosity representation is in stage 3 but the exploration driver reads from stage 5; relocate it. No shipped analog.

**(d) LatCal fixed-point commitment** (`riir-chain/src/encoding/`): irrelevant — LatCal is about deterministic commitment of raw scalars, not activation relocation.

**(e) `NeuronShard` / consolidation / AnyRAG / vibe KG** (`riir-neuron-db/src/`): the **second-closest latent analog**. Consolidation's `sleep(N)` is a stage-chain over wake-events. A wake-event representation that doesn't get consolidated into the shard because it's stranded in the wrong consolidation "stage" is a KU-Gap analog. But consolidation is offline (sleep-cycle), not runtime — the latency budget is different. The more direct mapping is `MerkleFrozenEnvelope` snapshot compositing (R259 already covers this for weight-level composites).

**(f) DEC Stokes operators** (`katgpt-dec/src/`): irrelevant — DEC is about manifold geometry on cochains, not residual stream relocation.

**Verdict on the latent reframe:** the latent-space reframe *partially* lands — the (b) latent_functor and (e) consolidation analogs are real but weaker than the paper's LLM-layer axis. Our latent substrate doesn't have the same "early vs late MLP" structure that gives the paper its two-cluster pattern. The genuinely unshipped primitive is the **operator** (applied relocation); the diagnostic is a natural extension of Plan 358's scorer.

### 2.4 Fusion

**Fusion A — Cross-stage residual relocation × Latent Field Steering (R290):**
Latent Field Steering injects *frozen direction vectors* into the latent state. Cross-stage relocation injects *runtime-captured stage states*. Fusing them: a **dual-source relocation** that can pull from either a frozen direction vector (R290) or a runtime-captured stage state (this paper), gated by a sigmoid confidence. This is a strict generalization of both. → **GOAT fusion** (combines two shipped primitives into a more general operator), not a new pillar.

**Fusion B — Permeation-map diagnostic × Causal Head Importance (R362):**
R362 scans layer × head with `direct_effect_importance`. This paper scans source-layer × target-layer with the same cell score. Fusing them: a **3D intervention scan** `(l_src, l_dst, head)` that locates the precise head-and-stage pair responsible for the stranded representation. Strictly more informative than either alone. → **GOAT fusion** for the diagnostic surface.

**Fusion C — Cross-stage relocation × QK-Restore (R259):**
QK-Restore preserves *routing geometry* at the weight level. Cross-stage relocation preserves *content representations* at the activation level. Fusing them: a runtime that (1) at adapter-load time applies QK-Restore to preserve routing, (2) at forward-pass time applies cross-stage relocation to recover stranded content. The two are complementary, not redundant. → **GOAT fusion** with the closest Super-GOAT cousin.

**None of the fusions clear the Super-GOAT novelty bar.** All three combine shipped primitives into more general operators; none creates a new capability class or a new pillar.

---

## 3. Verdict

**Tier: GOAT** — provable modelless gain (the paper's 58–75% oracle recovery is the headline number) over existing approaches, but not a new class of capability. The closest shipped primitives (R259 QK-Restore, R362 HydraHead scorer, R290 Latent Field Steering) cover the mechanism in adjacent axes; this paper contributes the *activation-relocation axis* and the *permeation-map diagnostic*, both genuinely unshipped but not pillar-forming.

**Reasoning per question:**

- **Q1 — No prior art?** Partial. Activation patching ships (R362/P358). Per-matrix freeze/thaw ships (R259). Direction-vector injection ships (R290/P309). The specific primitive — *applied* cross-stage residual state relocation + permeation-map scan — does **not** ship. → Q1 partial.
- **Q2 — New class of behavior?** Marginal. Adds the "relocate activation across stages as a runtime operator" axis. Doesn't create a new capability class beyond "better latent-state routing" (which R259/R290 already enable in adjacent axes). → Q2 NO.
- **Q3 — Product selling point?** Weak for our quintet. We don't run fine-tuned LLMs in the hot path (we run inference-only with optional adapter swap). The KU Gap is a fine-tuning phenomenon; our NPCs use latent functors and HLA, not multi-hop LLM reasoning. The latent reframe (b)/(e) is real but doesn't multiply a pillar. → Q3 NO.
- **Q4 — Force multiplier?** Limited. Fuses with R259, R290, R362 — but the fusions produce more general operators, not new pillars. → Q4 NO.

**Two YES (or strong-partial) out of four → GOAT, not Super-GOAT.** Plan only, no private guide.

**MOAT gate per domain (§1.6):**

| Domain | Verdict |
|---|---|
| `katgpt-rs` (public engine) | ✅ **In scope.** This is a generic modelless inference primitive (cross-stage residual relocation + permeation-map diagnostic). Ships behind a feature flag, opt-in, GOAT gate decides promote-to-default. Stack slot: **intervention/diagnostic** (alongside `causal_head_importance`, `faithfulness_probe`). |
| `riir-ai` (private runtime) | ❌ Out of scope. The latent reframe (b) latent_functor stages is real but doesn't multiply any of the 9 pillars strongly enough to warrant a private guide. The runtime has no fine-tuned-LLM-in-hot-path product angle. |
| `riir-chain` / `riir-neuron-db` / `riir-train` | ❌ Out of scope. No chain/shard/training angle. The fine-tuning dynamics → riir-train (one-line note, no file this session). |

**Training redirect:** the saturation-epoch analysis, the gradient-vanishing explanation, the natural-fine-tuning-vs-manual-relocation comparison, and any "alignment-aware training" follow-up → **riir-train**. Only the heuristic relocation operator and the permeation-map diagnostic are modelless and stay here.

**§3.6 defend-wrong PoC requirement:** the operator-half of the GOAT claim ("relocate recovers capability") is a **quality claim** that crosses into our substrate. The paper proves it on LLMs with knowledge injection; we'd be claiming the latent-functor analog works. **A PoC in `riir-ai/crates/riir-poc/` is mandatory before any feature-flag promotion** — architectural coverage of the operator (a `RelocateOp` struct that copies states) is not quality parity with the paper's 58–75% recovery. The PoC must run three competitors on a controlled toy domain: (a) the paper's heuristic, (b) no-relocation baseline, (c) the shipped latent_functor re-estimation. See Plan 431 Phase 3.

---

## 4. Plan (sketch — full plan at `.plans/431_cross_stage_residual_relocation_primitive.md`)

> **STATUS (2026-07-13): Plan 431 SHIPPED — Phase 1–4 COMPLETE.** All sketch items below were expanded into Plan 431 T1.1–T4.8 (full plan has finer task granularity than this sketch). Plan 431's defend-wrong PoC REFUTED the fixed-pair `LateEarly` default (see §"PoC Addendum" below for raw numbers); the mechanism itself works via diagnostic-guided CUSTOM relocation. Primitive stays opt-in behind `cross_stage_relocation`. Items marked `[x]` reflect Plan 431 completion; see the plan file for full evidence.

### Phase 1 — Permeation-Map Diagnostic (the safe half)

Ship `permeation_scan` as a thin extension of `causal_head_importance`'s scorer. Reuses `direct_effect_importance` as the cell score; adds the 2D `(l_src, l_dst)` scan loop. **No forward-pass machinery of its own** — the caller supplies the patched-forward closure (same contract as Plan 358). Zero new sync-boundary data.

- [x] T1.1 `PermeationMap { rows: Vec<Vec<f32>>, n_src: usize, n_dst: usize }` struct + `scan_into` method. → Plan 431 T1.2 (struct uses flat `cells: Vec<f32>` row-major + `n_src`/`n_dst` for cache friendliness).
- [x] T1.2 Two-cluster detection: simple max-loc + quadrant classification (early/mid/late). → Plan 431 T1.4 (`ClusterClass` enum + `classify_two_cluster`).
- [x] T1.3 G1 correctness test on a synthetic 4-stage chain with a known-stranded representation. → Plan 431 T1.5.
- [x] T1.4 G3 latency: full scan ≤ `n_src × n_dst × forward_pass_cost` (no overhead beyond the closure calls). → Plan 431 T1.7 (PASS — 10–25% FASTER than hand-rolled loop).
- [x] T1.5 G4 zero-alloc (caller-supplied scratch buffer for the matrix). → Plan 431 T1.6.

### Phase 2 — Cross-Stage Relocation Operator (the risky half)

Ship `RelocateOp { src_stage, dst_stage, anchor_token_idx }` + `RelocatePair::LateEarly` default. The operator's `apply` method snapshots the anchor's state at `src_stage` and overwrites at `dst_stage` during a forward pass.

- [x] T2.1 `RelocateOp` struct + `apply_into` method (zero-alloc, `#[inline]`). → Plan 431 T2.1–T2.2.
- [x] T2.2 `RelocatePair::LateEarly` constant: `(0.82, 0.45)` + `(0.10, 0.45)` per the paper. → Plan 431 T2.2 (also adds `Custom { src_a, src_b, dst }` variant — the production path per the PoC).
- [x] T2.3 Trait integration: `RelocatingForward` trait that the host's forward pass implements (snapshot + overwrite hooks). → Plan 431 T2.3.
- [x] T2.4 G1 unit test: relocate on a synthetic stranded-state case recovers the answer. → Plan 431 T2.5.

### Phase 3 — Defend-Wrong PoC (MANDATORY before any promotion)

Per §3.6, ship a PoC in `riir-ai/crates/riir-poc/benches/cross_stage_relocation_modelless_goat.rs` with three competitors:

- [x] T3.1 **Paper's heuristic** (two fixed pairs) on a controlled toy domain (synthetic stranded-representation chain). → Plan 431 T3.1–T3.2 (4 placement configs × 16 seeds + noise sweep).
- [x] T3.2 **No-relocation baseline** (standard forward pass). → Plan 431 T3.3 (competitor b).
- [x] T3.3 **Shipped latent_functor re-estimation** (R313) — the existing modelless analog. → Plan 431 T3.3 (competitor c, CohReest).
- [x] T3.4 Print verdict table. **If the heuristic doesn't beat both baselines, the GOAT gate FAILS** and the operator stays opt-in diagnostic-only. → Plan 431 T3.4–T3.5. **VERDICT: REFUTE fixed-pair `LateEarly`** (CLOBBERS in 2/4 clean configs); see §"PoC Addendum" below.

### Phase 4 — GOAT Gate + Promote/Demote

- [x] T4.1 G1 (correctness), G2 (perf vs no-relocation overhead ≤ 5%), G3 (no-regression on existing tests), G4 (zero-alloc), G5 (feature-isolated), G6 (modelless — no training dep). → Plan 431 T4.1–T4.6 (all PASS for katgpt-rs scope).
- [x] T4.2 If G1–G6 PASS and PoC confirms gain → consider promotion. Default **opt-in** until a real-game-domain PoC lands in riir-ai (deferred to riir-ai follow-up, not a katgpt-rs blocker). → Plan 431 T4.7. **DECISION: stays OPT-IN** (PoC refuted the fixed default; real-domain PoC deferred).
- [x] T4.3 Record promote/demote per the §1.6 per-stack ledger. Stack slot: **intervention/diagnostic** (alongside `causal_head_importance`). → Plan 431 T4.8 (recorded; README + overview.md updated).

---

## 5. Cross-references

- **R259 (QK-Restore, Super-GOAT)** — closest cousin. Same "fine-tuning breaks knowledge, fix modellessly" family. Different axis (per-matrix vs per-position-across-layers). The two are complementary: QK-Restore preserves routing geometry; self-patching relocates content representations. **If this primitive's GOAT gate passes, consider fusing with QK-Restore into a unified "surgical adapter composition" framework.**
- **R362 / P358 (HydraHead Causal Head Importance, GOAT — shipped)** — the cell-score function (`direct_effect_importance`) is identical. Plan 431 Phase 1 reuses it directly.
- **R313 (Thinking-to-Recall, PASS)** — the latent-space analog. `riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs` handles the reactive case (re-derive when coherence decays); this paper's permeation map is the *diagnostic* that explains why, and the heuristic is the *proactive* complement.
- **R290 / P309 (Latent Field Steering)** — frozen-direction injection. Plan 431 Fusion A generalizes this to dual-source (frozen OR runtime-captured).
- **R276 / P297 (PersonalityWeightedComposition)** — sigmoid-gated layer composition. The closest shipped "applied per-layer composition" primitive; doesn't relocate activations.
- **R388 (Jacobian-Lens, REFUTED as prefilter)** — the SVD concept-readout survives. Could combine with the permeation map: SVD identifies *what* concept is stranded; permeation map identifies *where* it's stranded.
- **riir-train** — saturation-epoch analysis, gradient-vanishing explanation, alignment-aware training follow-up.

> **PASS-Redirects (synthesis):** Muyu He ["Disabling Attention Layers", riddlehe.github.io/blog/disabling-attention-layers.html, 2026-08/09 blog] — layer-band ablation on Qwen3-8B under attention-restricted entity extraction maps this note's knowledge-circuit story onto explicit layer bands: L0–19 neither necessary nor sufficient, L20–24 bottleneck-then-recover, L31–35 collapse (= the direct-recall shortcut band). The Tolkien→JK-Rowling / Gandhi→Greta-Thunberg confusions are the KU-Gap shape observed live: abstract features (occupation/domain) extracted, the name binding lost. The blog's causal attention-score patching is the same intervention family as Plan 358's activation patching. Confirmatory; nothing to action.

---

## TL;DR

**Verdict: GOAT.** The paper's modelless crown jewel is a **deterministic, fixed-pair cross-stage residual state relocation operator** (source ≈0.8L or ≈0.1L → target ≈0.5L, recovers 58–75% of oracle headroom on the KU Gap benchmark) plus a **permeation-map diagnostic** that scans `(source_stage, target_stage)` pairs to locate stranded representations. Both are modelless; both are genuinely unshipped in our quintet (Plan 358 ships activation patching as a *scorer*; R259 ships *weight-level* per-matrix composites; R290 ships *frozen-direction* injection — none ships runtime *activation relocation*). Not Super-GOAT: Q1 partial (closest cousins ship strongly), Q2 marginal (no new capability class), Q3 weak (no fine-tuned-LLM-in-hot-path product angle), Q4 limited (fusions produce more general operators, not new pillars). MOAT gate routes to `katgpt-rs` (public engine, intervention/diagnostic stack slot). Plan 431 ships both halves behind a feature flag; **Phase 3 defend-wrong PoC in `riir-poc/` is MANDATORY before any promotion** — the 58–75% recovery is a quality claim on the paper's substrate, not ours. Training-time analysis (saturation epochs, gradient locality, alignment-aware training) → riir-train.

---

## PoC Addendum (2026-07-13 — Plan 431 Phase 3, honest recording per §3.6)

**Verdict: REFUTE the fixed-pair `LateEarly` default. The mechanism transfers; the fixed default is brittle.**

The PoC ran 4 competitors × 4 placement configs × 16 seeds + a noise sweep (5 levels × 16 seeds) in `riir-ai/crates/riir-poc/benches/cross_stage_relocation_modelless_goat.rs`. Raw numbers:

### Clean configs (noise_std = 0, cosine recovery to answer)

| Config | (b) baseline | (a) LateEarly | (a') Late-only | (c) CohReest | Verdict |
|---|---|---|---|---|---|
| PlanDomain {2,7} | 0.0000 | **0.0000** | 1.0000 | 1.0000 | CLOBBER |
| HeuristicMatch {1,7} | 0.0000 | 1.0000 | 1.0000 | 1.0000 | LE<CR (tie) |
| BroadCluster {1,2,6,7} | 0.0000 | 1.0000 | 1.0000 | 1.0000 | LE<CR (tie) |
| LateOnly {6,7} | 0.0000 | **0.0000** | 1.0000 | 1.0000 | CLOBBER |

**Tally: 0 CONFIRM / 2 CLOBBER / 2 LE<CR.**

### Noise sweep (HeuristicMatch placement, cosine recovery)

| noise_std | (b) baseline | (a) LateEarly | (a') Late-only | (c) CohReest |
|---|---|---|---|---|
| 0.0 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.1 | 0.0927 | 0.9692 | 0.9626 | 0.9240 |
| 0.2 | 0.0927 | 0.9001 | 0.8701 | 0.8831 |
| 0.3 | 0.0927 | 0.8220 | 0.7652 | 0.8359 |
| 0.5 | 0.0927 | 0.6839 | 0.5964 | 0.7478 |

Under noise, (a) LateEarly is competitive with (a') and (c) within the HeuristicMatch config — the second op is harmless because both sources have the answer. But this only holds when the domain matches the fixed fractions.

### Latency (criterion-benched, per episode)

| Competitor | Time |
|---|---|
| baseline_read | 17 ns |
| late_early_both_ops | 25 ns |
| late_only_single_op | 25 ns |
| coherence_reest_scan | 35 ns |

The heuristic (25 ns) is ~1.4× faster than the coherence scan (35 ns). But the speed advantage is moot when the heuristic clobbers.

### Why the heuristic fails (the clobbering mechanism)

For L=8, the heuristic targets: op_a = (src=7, dst=4), op_b = (src=1, dst=4). Both hard-overwrite stage 4. Applied in the shipped order (op_a first, op_b second):

1. **op_a** snapshots stage 7 (which has the answer in PlanDomain) → overwrites stage 4 with the answer. Stage 4 now holds the answer. Recovery = 1.0.
2. **op_b** snapshots stage 1 (which is EMPTY in PlanDomain) → overwrites stage 4 with zeros. Stage 4 is now empty. Recovery = 0.0.

The second op CLOBBERS the first. In the paper's LLM substrate, both source layers (0.82L and 0.10L) contain the knowledge because the two-cluster pattern guarantees it. On our synthetic substrate, the answer placement is NOT guaranteed to match the fixed fractions — hence the clobbering.

### What works: single-op relocation + diagnostic

The (a') late-only variant (apply ONLY op_a, skip op_b) recovers in all 4 clean configs. This confirms the **mechanism** (activation relocation) transfers — it's the **fixed two-pair default** that fails. The production path is: use the permeation-map diagnostic (Phase 1) to find which stage holds the answer, then apply a CUSTOM `RelocateOp` from that stage to the readout stage. This is `RelocatePair::Custom { src_a, src_b, dst }` where the fractions are derived from the diagnostic, not fixed.

### Implication for promotion

- **`RelocatePair::LateEarly` should NOT be promoted to default-on.** It clobbers in 2/4 clean configs.
- **The diagnostic half (`PermeationMap` + `permeation_scan_into`) is useful regardless** — it's a clean Plan 358 extension that locates stranded representations.
- **The operator half (`RelocateOp` + `RelocatePair::Custom`) is the production path** — the caller uses the diagnostic to pick the right source stage, then applies a custom single-op relocation.
- **Real-domain PoC deferred.** The synthetic PoC shows the mechanism works but the fixed default is brittle. A real-game-domain PoC (e.g., in `riir-games` NPC cognition) would test whether the diagnostic-guided custom relocation produces measurable behavioral gains. This is a follow-up, not a blocker for the opt-in primitive.

### Honest caveat

The synthetic domain is deliberately minimal (independent vectors per stage, no residual accumulation, no attention/MLP dynamics). A richer domain with actual residual accumulation MIGHT avoid the clobbering (if the residual at stage 1 happens to carry forward from earlier injection). But the burden of proof is on the promoter — the synthetic PoC shows the fixed default is NOT safe to promote as-is. The diagnostic + custom path is the honest recommendation.
