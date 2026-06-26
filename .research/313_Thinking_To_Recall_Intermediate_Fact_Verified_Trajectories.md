# Research 313: Thinking to Recall — Intermediate-Fact-Verified Trajectory Selection

> **Source:** [Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs](https://arxiv.org/abs/2603.09906) — Gekhman, Aharoni, Ofek, Geva, Reichart, Herzig (Google Research, 2026-06)
> **Blog:** [research.google/blog/thinking-to-recall-...](https://research.google/blog/thinking-to-recall-how-reasoning-unlocks-parametric-knowledge-in-llms/)
> **Date:** 2026-06-26
> **Status:** Done
> **Related Research:** 244 (FaithfulnessProbe), 278 (Engram), 255 (CLR), 216 (MRAgent), 267 (FPCG), 277 (SmearClassifier), 247 (Mind-Reading)
> **Related Plans:** 278 (FaithfulnessProbe), 281 (BoMSampler), 284 (CLR), 299 (Engram), 248 (OctreeCTC), 308 (Cognitive Integrity Layer)
> **Classification:** Public

---

## TL;DR

The paper explains *why* CoT helps even on single-hop factual recall via two complementary mechanisms: a **computational buffer** (more forward passes refine the latent state, content-agnostic) and **factual priming** (generating topically-related facts acts as semantic anchors via spreading activation). Its actionable distillation is the **hallucination trap**: a single hallucinated intermediate fact measurably degrades the final answer, and filtering trajectories by hallucination-free intermediates recovers most of the lost accuracy at test time.

For our codebase, all three mechanisms map onto already-shipped primitives. The novel transferable insight is a **composition gate** we don't yet ship end-to-end: filter k-sampled trajectories by verifying *every* intermediate fact they emit against committed memory, then vote the survivors by CLR reliability. Mechanism 1 validates the existing per-tick multi-cycle design (HLA `evolve_hla`, LT2 looped forward, cgsp cycles). Mechanism 2 is a quality-gate refinement for Engram/MRAgent anchor selection (prefer hard-fact anchors, skip filler). Mechanism 3 is a new gate that composes FaithfulnessProbe + CLR + Engram + BoMSampler into an intermediate-fact-verified trajectory selector.

**Distilled for katgpt-rs (modelless, inference-time):**
- **M1 (compute buffer):** validation only — no new code. The fact that "Let me think" repeated improves recall justifies the LT2/cgsp/HLA multi-cycle design *as a recall mechanism*, not just as task decomposition. Quality gate: N-cycle latent refinement hits diminishing returns beyond a threshold (formalize as a `compute_buffer_saturation` curve).
- **M2 (factual priming):** a quality-gate refinement for Engram/MRAgent. The paper's specific contribution — "facts alone (strict filtering of filler) recover most of CoT's gain" — argues for a `FactAnchor` abstraction distinct from `FillerAnchor`: only topically-related hard facts become priming anchors.
- **M3 (hallucination trap → trajectory filter):** the load-bearing distillation. Compose BoMSampler (k-trajectory sampling) + Engram/anchor-extraction (intermediate facts) + FaithfulnessProbe (verify each intermediate fact against committed memory) + CLR reliability vote (select from verified trajectories). This is a new gate, not a new primitive.

---

## 1. Paper Core Findings

The paper investigates why chain-of-thought (CoT) reasoning improves factual recall on *simple, single-hop* questions — where no logical decomposition is needed. It runs controlled hypothesis-driven experiments on Gemini-2.5 (Flash, Pro) and Qwen3-32B over SimpleQA Verified and EntityQuestions.

### 1.1 Capability boundary via pass@k

Reasoning-enabled LLMs (R-LLMs) recall answers that are *virtually unrecoverable* when reasoning is disabled. Measured via `pass@k` (correct answer exists within k generated attempts) on toggleable-reasoning models. This is the headline: reasoning expands the *capability boundary* of parametric memory, not just the *ranking* of already-reachable answers.

### 1.2 Mechanism 1 — Computational buffer (content-agnostic)

**Experiment:** replace the model's generated reasoning trace with a meaningless string ("Let me think" repeated) of the same length, then let the model answer.

**Result:** conditioning on the meaningless trace substantially improves recall vs no-reasoning baseline. **The act of generating extra tokens is itself useful** — extra forward passes refine the internal latent state independent of content.

**Limit:** pushing dummy text longer offers diminishing returns; it never matches natural reasoning traces. So content matters *additively* on top of the buffer effect.

### 1.3 Mechanism 2 — Factual priming (spreading activation)

**Observation:** natural reasoning traces for factual questions aren't logical proofs — they surface topically-related facts.

**Experiment:** extract only the concrete facts from reasoning traces (strict filtering of filler, search plans, and explicit mentions of the target answer). Condition the model on this short fact list.

**Result:** fact-only conditioning recovers most of CoT's gain. Recalling the first 9 kings of Nepal primes the network to recall the 10th. This is the LLM analog of human *spreading activation* in semantic memory. The paper names it **factual priming** / generative self-retrieval.

### 1.4 Mechanism 3 — The hallucination trap

**Audit:** a search-enabled verifier independently checks every intermediate fact across hundreds of thousands of reasoning traces.

**Result:** if a reasoning trace contains *even a single* hallucinated intermediate fact, the model is significantly less likely to reach the correct final answer. The factual-priming mechanism is fragile to hallucinated intermediates.

**Practical distillation:** test-time trajectory selection — generate multiple trajectories per question, retain only those whose intermediate facts are verifiably hallucination-free. This improves accuracy substantially. (The training-time analog is process rewards; we don't need that.)

---

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent (greppable) | Shipped? |
|------------|--------------------------------|----------|
| "computational buffer" / "extra forward passes" / "extended compute" | `evolve_hla`, `cgsp cycle`, `functor application`, `leaky integrator`, LT2 `forward_looped`, `elastic_loop_override` | ✅ Shipped (sense/reconstruction, LT2 Plan 108) |
| "factual priming" / "generative self-retrieval" / "spreading activation" | `engram lookup`, `multi_head_hash`, `sigmoid_fuse_into`, `ReconstructionState::expand`, `OctreeCTC`, `AnyRAG escalation`, `delta_mem recall`, KV cache priming (`dflash_predict_conditioned_with`) | ✅ Shipped (Plan 299, Plan 248, Plan 053, Plan 024) |
| "intermediate fact" / "verifiable fact" / "concrete fact extraction" | KG triple, `TernaryDir` projection, direction-vector anchor, `Claim` (Claim Rubric Plan 307) | ✅ Partially shipped (KG triples in vibe.rs, Claim in claim rubric — but no formal "intermediate fact extractor") |
| "hallucination" / "hallucination-free" | `Intervention::Filler`, `Intervention::Irrelevant`, `is_faithfully_used`, `SmearClassifier`, `cognitive_integrity_score`, `reward_hacking` vs `fully_faithful` | ✅ Shipped (FaithfulnessProbe Plan 278, SmearClassifier Plan 298, PathConsistency Plan 054, Cognitive Integrity Layer R129) |
| "trajectory selection" / "pass@k" / "retain hallucination-free" | `BoMSampler`, `clr_vote`, `(mean)^M reliability`, `BanditPruner` UCB1, `best_of_k_rollouts`, `extract_best_path` | ✅ Shipped (Plan 281, Plan 284, Plan 083, Plan 095) |
| "process reward" (training-time) | n/a — **redirect to riir-train** | n/a |

### 2.2 Closest prior art (both layers, all repos)

| Cousin | What it ships | Distance from this paper's M3 |
|--------|---------------|-------------------------------|
| **FaithfulnessProbe (R244/P278)** | Causal-intervention diagnostic on *injected memory*. `Intervention::{Empty,Shuffle,Corrupt,Irrelevant,Filler}`, `is_faithfully_used(threshold)`. Detects *dead injections*. | Operates per-injection-event, not per-trajectory-intermediate-fact. The composition "filter trajectory by every-intermediate-fact probe" is not shipped. |
| **CLR (R255/P284)** | Per-entity k-trajectory vote by `(mean)^M` reliability on M extracted claims. Crowd-scale test-time scaling. | Operates on final-answer claims, not on intermediate-step facts along the trajectory. M3 adds the trajectory-intermediate axis. |
| **PathConsistencySummary (R025/P054)** | `reward_hacking` (right answer, broken reasoning) vs `fully_faithful` (both correct). `path_total`, `avg_consistency`. | Ships the *concept* of "intermediate step correctness matters." But it's a *training-time metric* (Plan 054 is GRPO bi-level), not an inference-time filter. M3 makes it inference-time. |
| **Cognitive Integrity Layer (riir-ai R129/P308)** | Architectural guide: input-side faithfulness × output-side path-hacking → `cognitive_integrity_score`. | The guide's loop runs the probe at *injection cadence*, not at *trajectory-step cadence*. M3 is the missing per-step instantiation. |
| **Engram (R278/P299)** | Hash-addressed conditional pattern memory. `EngramTable::lookup_into`, `sigmoid_fuse_into`. G6 (effective depth) deferred to riir-ai. | Engram is the *priming mechanism* (paper's M2). The deferred G6 is exactly "does priming actually bind" — the hallucination filter (M3) is the missing validation hook. |
| **MRAgent / OctreeCTC (R216/P248)** | Reconstructive multi-step memory navigation. `ReconstructionState::expand → route → accumulate → evolve_hla`. | Multi-step recall = paper's M2 generalized. No per-step fact verification. |
| **SmearClassifier (R277/P298)** | Detects unfaithful generations via cosine-smear. 107.6 ns/classify. | Operates on final generation, not intermediate facts. Composable into M3. |
| **FPCG (R267/P292)** | Detection ≠ Prediction features at intermediate activations. Linear probe on intermediate-step activations. | Closest to M3 conceptually — operates *at intermediate steps*. But FPCG predicts *future behavior*, not *fact verification*. |

**Two-layer verdict:** the paper's three mechanisms are each individually well-covered. The novel element is the **composition gate** — filter a sampled trajectory by verifying every intermediate fact it emits against committed memory, then vote survivors by CLR reliability. No shipped primitive composes all four pieces (BoMSampler × Engram-anchor-extraction × FaithfulnessProbe × CLR-vote) at per-trajectory-step granularity.

### 2.3 Latent-space reframing (per fusion protocol step 3 — mandatory)

Re-cast each mechanism as a latent-to-latent op on the codebase's latent-state kernels:

**M1 — Compute buffer as latent refinement:** Each cgsp cycle / HLA `evolve_hla` step / LT2 loop iteration is one forward pass on the latent state. The paper's "Let me think" experiment says: even a *content-free* cycle refines the latent state (because the recurrent kernel is doing useful integration work, not because the token content carries information). This validates that our multi-cycle design (HLA leaky integrator, LT2 elastic loop, cgsp self-play rounds) is doing real recall work, not just task decomposition. The modelless distillation is a **saturation curve**: `recall_gain(N_cycles)` saturates as `N → N*`. The GOAT gate is: does `N*` for dummy cycles match `N*` for content cycles modulo a constant offset? If yes, our content-free cycle budget is correctly sized.

**M2 — Factual priming as anchor-conditioned projection:** Spreading activation = dot-product projection onto related-fact direction vectors with sigmoid gating. The paper's "facts alone recover most of the gain" maps to: `output = σ(q · anchor_fact_direction / τ) · v`, where `anchor_fact_direction` comes from the Engram table (Plan 299) or the OctreeCTC traversal (Plan 248). The novel insight is the `FactAnchor` vs `FillerAnchor` distinction: only topically-grounded hard facts become anchors; filler text produces near-zero projection and should be filtered at anchor-selection time, not at fusion time.

**M3 — Hallucination filter as trajectory-level AND-gate over intermediate-fact probes:** This is the load-bearing distillation. For each trajectory `T_k` sampled by BoMSampler:
1. Extract intermediate facts `{f_{k,1}, ..., f_{k,n}}` as direction-vector anchors (Engram hash addresses or KG triples emitted along the trajectory).
2. For each `f_{k,i}`, run `FaithfulnessProbe::probe_intervention(f_{k,i}, Intervention::Irrelevant)` and `::Filler` against committed shard memory (NeuronShard BLAKE3 commitment, Merkle proof).
3. Trajectory `T_k` is *retained* iff every `f_{k,i}` is `is_faithfully_used(threshold)`.
4. Vote retained trajectories by CLR `(mean)^M` reliability on final claims.
5. Discard trajectories with any failed intermediate probe.

This is a strict-AND filter (one bad fact kills the trajectory), which matches the paper's headline finding ("a single hallucinated intermediate fact degrades the final answer").

### 2.4 Fusion — what novel combination does this enable?

Per SKILL §1 fusion protocol, fuse with the 2–3 closest cousins:

**Fusion A — Thinking-to-Recall Gate × CLR × FaithfulnessProbe (the headline GOAT):**
- *CLR* (R255/P284) = per-entity k-trajectory vote by `(mean)^M` reliability on final claims.
- *FaithfulnessProbe* (R244/P278) = causal intervention on injected memory.
- *This paper* (M3) = intermediate-fact-verified trajectory selection.
- *Novel combination:* CLR currently votes on final claims; this fuses an *intermediate-fact verification pre-filter* — trajectories whose intermediate facts fail FaithfulnessProbe are excluded from the CLR vote entirely. The result is a two-stage filter: (1) intermediate-fact AND-gate, (2) CLR reliability vote. Neither alone is sufficient: CLR alone lets a hallucinated-but-confident trajectory win; the AND-gate alone has no final-claim reliability. Together they match the paper's empirical protocol.

**Fusion B — Fact-Anchor quality gate for Engram / OctreeCTC (M2 distillation):**
- *Engram* (R278/P299) = hash-addressed conditional pattern memory; G6 (effective depth) deferred.
- *OctreeCTC* (R216/P248) = reconstructive multi-step navigation.
- *This paper* (M2) = facts-only priming recovers most of CoT's gain.
- *Novel combination:* add a `FactAnchor` vs `FillerAnchor` distinction to the Engram/OctreeCTC anchor-selection policy. The paper's strict-filter experiment (strip filler, keep only concrete facts) is the validation: anchor selection should prefer hard-fact hashes over generic context hashes. This *unblocks Engram G6*: the deferred "does priming bind?" gate becomes "do fact-only anchors bind better than mixed anchors?" — a concrete, runnable experiment.

**Fusion C — Compute-buffer saturation curve for LT2 / cgsp (M1 distillation):**
- *LT2* (Plan 108) = looped forward pass with `elastic_loop_override`.
- *cgsp* (Plan 274) = curiosity-guided self-play rounds.
- *This paper* (M1) = dummy CoT (content-free extra cycles) improves recall with diminishing returns.
- *Novel combination:* a benchmark/quality-gate that measures `recall_gain(N)` for content-free cycle budgets. If the curve saturates at the same `N*` as content cycles modulo a constant offset, our per-tick cycle budget is correctly sized. This is a *validation* of existing design, not new code — but it provides the missing theoretical justification for the cycle budget parameter.

---

## 3. Verdict

### **GOAT** — composition gate (M3) + quality-gate refinement (M2) + design validation (M1)

**One-line reasoning:** The paper's three mechanisms are each individually covered by shipped primitives (HLA/LT2 for M1, Engram/MRAgent for M2, FaithfulnessProbe+CLR+PathConsistency for M3). The novel transferable contribution is a *composition gate* — filter trajectories by per-intermediate-fact verification before CLR voting — which composes existing primitives at a new granularity, not a new primitive class. GOAT, not Super-GOAT.

### Novelty gate (Q1–Q4)

| Q | Question | Answer |
|---|----------|--------|
| Q1 | No prior art? | **Partial.** Each constituent primitive ships (FaithfulnessProbe, CLR, BoMSampler, Engram). The *composition at per-trajectory-step granularity* is novel — no shipped primitive runs the FaithfulnessProbe at every intermediate step of every sampled trajectory and AND-gates the CLR vote on the result. PathConsistencySummary (Plan 054) ships the *concept* but only as a training-time metric, not an inference-time filter. |
| Q2 | New capability class? | **No.** "Trajectories filtered by intermediate-fact verification" is a refinement of CLR's "trajectories filtered by final-claim reliability" + FaithfulnessProbe's "memory verified by causal intervention." Same capability class (test-time trajectory selection), stricter filter. |
| Q3 | Product selling point? | **Weak.** The Cognitive Integrity Layer (R129) already owns "NPCs don't fake their reasoning." This adds "and don't hallucinate intermediate steps" — incremental, not a new headline. |
| Q4 | Force multiplier? | **Yes** — connects CLR, FaithfulnessProbe, Engram, BoMSampler, Cognitive Integrity Layer, PathConsistency. ≥5 pillars. But force multiplication without a new capability class is GOAT, not Super-GOAT (per SKILL §1.5 Q2). |

**Verdict: GOAT.** Plan + implement behind feature flag + benchmark. No Super-GOAT guide needed (the Cognitive Integrity Layer guide R129 already owns this territory).

### Routing

| Output | Repo | File |
|--------|------|------|
| Open primitive research note | katgpt-rs | `.research/313_*.md` (this file) |
| Open primitive plan | katgpt-rs | `.plans/332_thinking_to_recall_intermediate_fact_gate.md` |
| M1 saturation benchmark | katgpt-rs | informational only, no new code (validates LT2/cgsp/HLA cycle budgets) |
| M2 FactAnchor refinement | katgpt-rs | quality gate for Engram/OctreeCTC anchor selection (extension to Plan 299 / Plan 248) |
| M3 composition gate | katgpt-rs | new feature `intermediate_fact_gate` composing FaithfulnessProbe + CLR + BoMSampler |
| Per-NPC runtime wiring | riir-ai | (deferred — Cognitive Integrity Layer R129 already frames the runtime moat; the new gate slots into the existing integrity loop at step 6) |

### Why not Super-GOAT

- **Not a new sparsity axis** (unlike Engram R278 which added conditional memory as a *new axis* complementary to Raven's conditional computation). M3 is a stricter filter on an existing axis (test-time trajectory selection).
- **Not a new capability class** (unlike CLR R255 which introduced per-entity test-time scaling at 20Hz tick — a new problem shape). M3 is a refinement of CLR's filter.
- **Selling point is subsumed** by the Cognitive Integrity Layer (R129) which already owns "NPCs can't silently ignore memory or fake reasoning." M3 is the *per-step instantiation* of that guide's loop step 6, not a new selling point.

This is the honest outcome. The paper is excellent science; its distillation is a focused GOAT gate, not a moat.

---

## 4. What This Is NOT

To prevent overclaiming and scope creep:

- **NOT a new primitive.** FaithfulnessProbe, CLR, BoMSampler, Engram all ship. This is a composition gate.
- **NOT training.** The paper's process-reward training recipe → riir-train. We ship only the inference-time trajectory filter (the modelless analog: filter, don't train).
- **NOT a replacement for CLR.** CLR votes on final claims; this gate pre-filters by intermediate facts. They compose; CLR's `(mean)^M` is unchanged.
- **NOT a replacement for FaithfulnessProbe.** The probe API is unchanged; this gate *calls* it at per-trajectory-step cadence.
- **NOT a memory mechanism.** Engram/MRAgent/AnyRAG own memory. This gate *verifies* memory usage along trajectories.
- **NOT M1 (compute buffer) code.** M1 is validation only — no new code; it justifies existing LT2/cgsp/HLA cycle budgets.

---

## 5. Constraints Respected

- **Modelless first:** the gate composes inference-time primitives (FaithfulnessProbe causal intervention, CLR reliability vote, BoMSampler k-hypothesis sampling, Engram anchor extraction). Zero training, zero backprop.
- **Latent-to-latent preferred:** intermediate facts are direction-vector anchors (Engram hashes) or KG triples; verification is dot-product + sigmoid projection onto committed shard memory; the AND-gate is a scalar threshold. Decode to tokens only at the final-answer boundary.
- **Freeze/thaw over fine-tuning:** the "facts" are committed in NeuronShards (BLAKE3, Merkle-proof). Verification reads frozen state; no weight mutation.
- **5-repo discipline:** open primitive (composition gate math) → katgpt-rs; private runtime wiring → riir-ai (deferred to existing Cognitive Integrity Layer plan).
- **Raw scalars at sync boundary:** the gate's verdict (retain/discard trajectory) is a local per-entity scalar; only the *consequence* (chosen action) crosses sync. Direction vectors never substitute for raw position in anti-cheat.

---

## 6. Cross-references

- **Paper:** [arXiv:2603.09906](https://arxiv.org/abs/2603.09906) — Gekhman et al. 2026.
- **Private selling-point guide (existing, subsuming):** `riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md`
- **FaithfulnessProbe:** `katgpt-rs/.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md` + `katgpt-rs/.plans/278_faithfulness_probe_modelless.md`
- **CLR:** `katgpt-rs/.research/255_VibeThinker_CLR_Test_Time_Reliability.md` + `katgpt-rs/.plans/284_runtime_clr_self_adaptive_loop.md`
- **Engram:** `katgpt-rs/.research/278_Engram_Conditional_Memory_Latent_Lookup_Fusion.md` + `katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md`
- **MRAgent / OctreeCTC:** `katgpt-rs/.research/216_MRAgent_Reconstructive_Memory_Graph.md` + `katgpt-rs/.plans/248_octree_ctc_reconstructive_navigation.md`
- **BoMSampler:** `katgpt-rs/.plans/281_bom_single_pass_diverse_sampling.md`
- **PathConsistency (training-time analog):** `katgpt-rs/.plans/054_stepcode_reasoner_modelless.md`
- **SmearClassifier:** `katgpt-rs/.research/277_DiffusionGemma_Transparency_Smearing_Faithfulness.md`
- **FPCG (intermediate-step cousin):** `katgpt-rs/.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md`
- **Mind-Reading (CLR × belief-state fusion target):** `katgpt-rs/.research/247_*` + `riir-ai/.plans/311_npc_mind_reading_runtime.md`

---

## TL;DR

**Thinking-to-Recall = GOAT (composition gate).** Google's paper explains why CoT helps factual recall via three mechanisms: compute buffer (validates our LT2/cgsp/HLA multi-cycle design, no new code), factual priming (refines Engram/MRAgent anchor selection — prefer hard facts over filler), and the hallucination trap (the load-bearing distillation: filter trajectories by per-intermediate-fact verification, then CLR-vote the survivors). All three mechanisms map to shipped primitives; the novel contribution is a *composition gate* (`intermediate_fact_gate`) that runs FaithfulnessProbe at every intermediate step of every BoMSampler trajectory, AND-gates the CLR vote on the result. No new capability class, no Super-GOAT guide — the Cognitive Integrity Layer (riir-ai R129) already owns this selling point. Plan: `katgpt-rs/.plans/332_thinking_to_recall_intermediate_fact_gate.md`.
