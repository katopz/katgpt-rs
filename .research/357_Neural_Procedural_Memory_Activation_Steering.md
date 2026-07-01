# Research 357: Neural Procedural Memory — Activation Steering of LLM Agents

> **Source:** [Neural Procedural Memory: Empowering LLM Agents with Implicit Activation Steering](https://arxiv.org/abs/2606.29824) — Zhao, Tan, He, Wang, Zhao, Liu (CAS / BAAI), 30 Jun 2026, arXiv:2606.29824v1 [cs.CL]
> **Date:** 2026-07-01
> **Status:** Done — **Pass (already shipped, stronger in latent space)**
> **Related Research:** 290 (Latent Field Steering), 302 (FAME CommittedFieldBlend), 278 (Engram), 310 (RIZZ Non-Interference Branches), 259 (QK-Restore surgical adapter composition)
> **Related Plans:** 087 (CNA — contrastive neuron attribution steering), 309 (latent_field_steering primitive), 321 (CommittedFieldBlend), 329 (Non-Interference Memory Branches — DEFAULT-ON)
> **Classification:** Public

---

## TL;DR

NPM is a **training-free** framework that represents LLM-agent procedural memory as
**activation-steering vectors** distilled from **dual-granularity contrastive experiences**
(inter-trajectory success/fail + intra-trajectory degenerate/effective step pairs), then injects
the synthesized direction into the residual stream at inference. Evaluated on ALFWorld / WebShop /
ScienceWorld / BabyAI, it matches explicit-textual-memory baselines and adds complementary
synergy when combined with them.

**Verdict: Pass.** Every load-bearing mechanism is **already shipped in this codebase, in pure
latent space, stronger than NPM's LLM-residual-stream form**:

| NPM mechanism | Shipped equivalent | Stronger because |
|---|---|---|
| Contrastive pair → steering vector | **CNA** (Plan 087, arXiv:2605.12290) + **Latent Field Steering** (R290 / Plan 309) | Operates on 8-D HLA latent state + sparse MLP neurons; CNA preserves quality ≥0.97 vs CAA's <0.60 |
| Residual stream injection `h̃ = h + α·v` | `apply_latent_steering(state, field)` (Plan 309) | Same math `s' = s + α·v`, but on per-NPC 8-D affect vector (bounded, behavior-rank-preserving) not 4096-D LLM residual (CAA-degrades-quality) |
| Inter-trajectory (success vs fail) contrast | `ContrastivePairProvider` trait (Plan 087 T5) + CLR `should_write_memory` (Plan 284) | Game-domain pairs (Bomber safe/blast, Go high/low heuristic, FFT kill/waste) — no LLM judge needed |
| Intra-trajectory (degenerate vs effective step) contrast | **`CognitiveBranch.failures`** (RIZZ Plan 329) + `ProceduralRule` helpful/harmful counters | Persistent per-NPC failure store, orthogonal-subspace isolated, BLAKE3-committable |
| PCA-based steering extraction | `subspace_phase_gate` (Plan 301, Jacobian SVD) + Dual-Gram PCA (Plan 159) + HLA Windowed Eigenbasis (Issue 001) | Per-NPC eigenbasis recovery, no LAPACK, modelless power iteration |
| Frozen direction vector, versioned, hot-swap | `MerkleFrozenEnvelope` + `CommittedFieldBlend` (Plan 321) + `KarcShard` / `ArchetypeBlendShard` | Sampling-invariant commitment (FAME Prop. 3), survives snapshot thaw — NPM has no commitment story |
| Dual-granularity dynamic selection by trajectory length | `latent_functor/reestimation.rs` coherence-gated scheduler + `latent_functor/zone_gating.rs` | Coherence-driven, not heuristic threshold |

The single genuinely useful **insight** (not mechanism) NPM contributes that the shipped stack
does not name explicitly: **intra-trajectory contrast from a *single* failed trajectory** —
splitting one failed episode into effective vs degenerate step-sets and extracting a steering
direction from that single trajectory. RIZZ's `CognitiveBranch.failures` stores anti-patterns
but does not derive a *direction vector* from a single failed trajectory's internal step
contrast. This is a one-line fusion idea tracked in `riir-ai/.issues/` (no new primitive, no
plan — it composes CLR's reward signal with NPM's degenerate-step heuristic on the existing
failure store).

---

## 1. Paper Core Findings

### 1.1 The thesis

LLM agents fail at procedural tasks because **textual procedural memory** (RAG-injected
guidelines) suffers a **text-action disconnect**: the agent comprehends the rule but cannot
reliably translate it into the correct action sequence. Cognitive neuroscience (Squire 1992,
2004) posits that procedural memory is **non-verbalizable** and manifests through **neural
activity modulation**, not declarative recall. NPM operationalizes this: distill procedural
skills into **steering vectors in activation space**, inject them at inference.

### 1.2 The three-phase pipeline

1. **Contrastive Experience Construction** (dual-granularity):
   - **Inter-trajectory** (Eq. 1): pair successful `τ⁺` with failed `τ⁻` trajectories for a task.
   - **Intra-trajectory** (Eq. 2): within a *single* failed trajectory, partition steps into
     `S_deg` (redundant loops / invalid actions) and `S_eff = τ \ S_deg`; pair them.

2. **Procedural Memory Extraction**:
   - Hidden state `ϕ_l(·)` — last-token (inter) or mean-pooled over steps (intra).
   - Memory storage: per-task contrastive pair set `C_l^(j) = {(ϕ_l(x_i⁺), ϕ_l(x_i⁻))}`.
   - Steering vector: **PCA first principal component** of mean-centered contrastive differences.

3. **Inference-Time Intervention** (Retrieval → Synthesis → Intervention):
   - Dense-retrieval top-K similar historical tasks.
   - Synthesize task-specific consensus direction `v_l(q) = ψ(M_l(q))`.
   - Inject: `h̃_{l,t} = h_{l,t} + α · v_l(q)` per autoregressive step, with KL-divergence-bounded
     `α*` (Eq. 7).

### 1.3 Empirical results

MiniCPM3-4B / Qwen3-4B / Qwen3-8B across ALFWorld / WebShop / ScienceWorld / BabyAI:

| Model | No Mem | Explicit Workflows | NPM (implicit) | NPM + Workflows (hybrid) |
|---|---|---|---|---|
| MiniCPM3-4B avg | 22.60 | 32.68 | 28.87 | **34.47** |
| Qwen3-4B avg | 28.14 | 34.23 | 31.39 | **37.60** |
| Qwen3-8B avg | 30.63 | 37.90 | 36.32 | **41.89** |

Implicit steering alone is **competitive** (not dominant) with explicit workflows; the
**hybrid** wins everywhere — explicit text supplies macro-planning, implicit steering enforces
procedural adherence in long horizons. CAA / Mass-Mean (static dataset-wide steering) lose
to NPM's task-specific dynamic synthesis.

### 1.4 Mechanistic findings (the most transferable part)

- **Linear separability** (Appendix C.1): successful vs degenerate modes are linearly separable
  in hidden-state space (SVM accuracy 88–99% across benchmarks) — justifies PCA-first-PC
  extraction.
- **Geometric consistency** (§5.2): steering vectors cluster by task category (diagonal-block
  cosine-similarity structure). Intra-trajectory vectors are *more locally focused*
  (top-3 concentration 43.6% vs 38.0%; lower normalized entropy) — local error correction vs
  distributed macro-planning.
- **Feature decomposition** (§5.3, Appendix B): 16-dim sparse dictionary over step hidden
  states yields interpretable behavioral primitives (`F0 RedundObs`, `F6 FinalPlace`,
  `F7 EarlyStop`, `F13 SysSearch`, …) with mutual-information-annotated polarity. Inter-vector
  amplifies planning features; intra-vector suppresses premature-stop and amplifies
  targeted-search features.
- **Retrieval pool scaling** (§5.4): universal features (exploration, redundant observation)
  grow monotonically with K; task-specific primitives peak at moderate K then decline
  (cross-task interference) — empirically motivates dynamic synthesis over static aggregation.
- **Latency** (Appendix D.2): NPM prefill 71.09 ms vs textual-memory 279.89 ms (4× faster — no
  KV-cache expansion). Steering is element-wise residual add, zero decoding overhead.
- **Storage** (Appendix D.1): 3 representations × L layers × d × b per trajectory; for Qwen3-4B
  half-precision × 3 layers = 45 KB/trajectory — fixed, independent of trajectory length.

---

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | ≥2 codebase equivalents |
|---|---|
| steering vector / activation steering | **direction vector** (HLA), **latent field**, `EmotionDirections`, `apply_latent_steering`, `CommittedFieldBlend` π vector |
| residual stream injection | **latent state mutation**, `compose_into`, `apply_blended`, `LatentField::apply_to_crowd` |
| procedural memory | **`CognitiveBranch.procedural`** (RIZZ Plan 329), `ProceduralRule`, closure motifs (Plan 290), `style_weights[64]` |
| contrastive pair (inter-trajectory) | `ContrastivePairProvider` (Plan 087 T5), CLR vote polarity |
| contrastive pair (intra-trajectory) | `CognitiveBranch.failures`, degenerate-step heuristic, collapse detector (Plan 212) |
| PCA first principal component | `subspace_phase_gate` Jacobian SVD (Plan 301), Dual-Gram PCA (Plan 159), HLA Windowed Eigenbasis (Issue 001) |
| retrieval-augmented steering | **Engram** hash-addressed lookup (Plan 299), `BranchRouter` dot-product snap (Plan 329) |
| KL-bounded intervention strength α* | `PersonalityWeightedComposition` τ schedule, `phase_rotation_subspace_phase_gate` G6 sigmoid-bounded rotation |
| task-specific dynamic synthesis ψ(M) | `BranchRouter::route` (selects branch by direction), `CommittedFieldBlend::commit` (compute π once from trajectory) |

### 2.2 What we already ship (the prior-art surface — verify before any novelty claim)

| NPM mechanism | Shipped cousin | Plan / file | Diff |
|---|---|---|---|
| Contrastive pair → steering vector | **CNA** `cna_discover()` + `ContrastivePairProvider` | Plan 087 (`crates/katgpt-core/src/cna/`) | CNA targets **MLP neurons** (sparse, ~10–50), quality ≥0.97; NPM targets **residual stream** (dense), needs KL-bounded α to avoid quality collapse. CNA is *strictly stronger* — published as arXiv:2605.12290 Nous Research. |
| Residual injection `h̃ = h + α·v` | **Latent Field Steering** `apply_latent_steering(state, field)` | Plan 309 / R290 | Same element-wise SIMD add. Target is per-NPC **8-D HLA latent state** (bounded, behavior-rank-preserving per G2 gate ≥0.95) not 4096-D LLM residual (CAA-degrades). R290 explicitly rejects CAA for LLM side, accepts for game-AI NPC side — NPM is the LLM side R290 rejected. |
| Inter-trajectory contrast (success vs fail) | Game-domain `ContrastivePairProvider` (Bomber / Go / FFT) + CLR `should_write_memory` | Plan 087 T5, Plan 284/316 | Game state provides natural contrastive pairs (win/loss, safe/blast, kill/waste). CLR gates memory writes by reward + learning-potential. No LLM judge needed. |
| Intra-trajectory contrast (degenerate vs effective steps) | `CognitiveBranch.failures` + `ProceduralRule { helpful, harmful }` counters | Plan 329 (DEFAULT-ON) | RIZZ's branch stores anti-patterns per-NPC in an **orthogonal latent subspace** — non-interfering by construction (G1a verified `interference < 1e-6` across 56 branch pairs). NPM has no isolation story. |
| PCA / sparse-dictionary feature decomposition | `subspace_phase_gate` Jacobian SVD + Dual-Gram PCA + HLA Windowed Eigenbasis | Plan 301, Plan 159, Issue 001 | Per-NPC eigenbasis recovery, modelless power iteration with deflation, no LAPACK. **Stronger**: per-NPC individualized affective geometry vs NPM's per-task global basis. |
| Retrieval + dynamic synthesis | `BranchRouter::route` (cosine snap + Jaccard fallback) + `Engram::lookup_into` | Plan 329, Plan 299 | Dot-product snap on pre-normalized embeddings, 301 ns on 64-branch bank, zero-alloc. NPM uses dense retriever + PCA composition — heavier. |
| Frozen direction vector, versioned, atomic swap | `MerkleFrozenEnvelope` + `CommittedFieldBlend` + `KarcShard` + `ArchetypeBlendShard` | Plan 321, riir-neuron-db Plan 004/336 | **NPM has no commitment story.** Our direction vectors are BLAKE3-committed, sampling-invariant (FAME Prop. 3), survive snapshot thaw, cross LatCal sync boundary as K raw floats. Strictly stronger. |
| Hybrid (implicit + explicit) | `MUX-Latent` (Plan 238) latent+explicit mode + `SwiR` switch-thinking (Plan 275) | Plans 238, 275 | Same hybrid pattern (textual plan + latent enforcement) — already shipped. |
| Dual-granularity dynamic selection | `latent_functor/reestimation.rs` coherence-gated scheduler + `zone_gating.rs` | riir-engine | Coherence-driven re-estimation trigger; not NPM's heuristic length-threshold `L_q > γ·L̄`. |

**No mechanism in NPM is unshipped.** Every load-bearing primitive has a shipped cousin that
is either (a) identical-math but in a stronger target space (HLA 8-D vs LLM residual 4096-D),
(b) augmented with commitment / sampling-invariance / non-interference that NPM lacks, or
(c) already proven on a stricter gate (CNA quality ≥0.97 vs NPM's KL-bounded-α workaround
for quality collapse).

### 2.3 The single useful insight (not a new mechanism)

NPM §3.1.2 **intra-trajectory contrast** splits a *single* failed trajectory into effective
vs degenerate step-sets and extracts a steering direction from that internal contrast. This is
useful when successful trajectories are sparse (cold-start).

Our `CognitiveBranch.failures` (RIZZ Plan 329) stores anti-patterns but does **not** derive a
direction vector from a single trajectory's internal step contrast. The degenerate-step
heuristics (redundancy = consecutive-repeat detection; invalidity = environment-error
feedback) are deterministic and cheap; CLR's reward signal already labels steps.

**Fusion idea (small, no new primitive):** enrich CLR's `should_write_memory` failure path
with an intra-trajectory contrast extractor. When a trajectory fails, partition its steps
into degenerate vs effective using NPM's two heuristics (consecutive-repeat + env-error),
compute the mean-pooled embedding contrast, and write the resulting direction into the
`CognitiveBranch.failures` store as a *direction vector* (not just an opaque anti-pattern).
The `BranchRouter`'s next routing decision then has a richer signal.

This is a **one-method addition to an existing shipped primitive** (Plan 329 + Plan 284), not
a new plan. Track in `riir-ai/.issues/` as a runtime enhancement; not a katgpt-rs primitive
(the direction-extraction math is already shipped — power iteration / PCA).

### 2.4 Why this is not even a fusion-GOAT

For a fusion-GOAT, the fusion must produce a *new capability* none of the components have
alone. The intra-trajectory insight produces a **richer failure signal** — incremental, not
new-capability. The `CognitiveBranch` already routes, isolates, and persists failures; adding
a direction-vector annotation to each failure entry is a quality improvement to the failure
store, not a new behavior class. → Gain at best, and the gain is on the riir-ai runtime side,
not the katgpt-rs engine side.

---

## 3. Verdict

### Tier: **Pass**

| Question | Answer |
|---|---|
| Q1 No prior art? | **NO.** Every load-bearing mechanism ships: CNA (Plan 087), Latent Field Steering (Plan 309), CommittedFieldBlend (Plan 321), Non-Interference Branches (Plan 329, DEFAULT-ON), CLR (Plan 284), subspace_phase_gate PCA (Plan 301). Vocabulary translation done; both layers grepped across all 5 repos. |
| Q2 New class of behavior? | **NO.** "Contrastive-pair-derived direction vector injected into latent state, retrieved at inference, dynamically synthesized per task" is the CNA + Latent Field Steering + BranchRouter composition — already shipping. |
| Q3 Product selling point? | **NO (for us).** NPM's selling point is "implicit activation steering matches explicit text for LLM agents" — that is an LLM-agent product claim. Our product is game AI operating in pure latent space; the equivalent selling point ("NPCs steered by latent field injection, crowd responds in 1 tick") is already the R290 / Plan 309 moat. |
| Q4 Force multiplier? | **NO.** NPM touches none of our pillars that CNA + Latent Field Steering + CommittedFieldBlend + BranchBank do not already touch. |

0/4 YES → **Pass.** One-line note (this file). No files created in katgpt-rs beyond this
research note; no plan; no open primitive. The single useful insight (intra-trajectory
contrast for richer failure-store direction vectors) is tracked as a riir-ai runtime
enhancement in `riir-ai/.issues/`, not a katgpt-rs primitive.

### MOAT gate per domain (§1.6)

N/A — Pass verdict. No contribution lands in any repo's moat. The riir-ai-side failure-store
enrichment (if pursued) is a neutral Gain on pillar 8 (Reasoning Pack) / pillar-adjacent
(self-learn NPCs) — not pillar-level.

---

## 4. What this paper confirms (positive signal, not a new primitive)

Two things in NPM are **independent confirmation** that our shipped design choices are correct:

1. **CAA / Mass-Mean (static dataset-wide steering) loses to task-specific dynamic synthesis.**
   NPM Table 1: CAA avg 16.84–22.79, Mass-Mean 24.71–31.89, NPM 28.87–36.32. This empirically
   validates R290's claim that Latent Field Steering must be **locality-aware**
   (`LatentField` with kernel support), not a global shift — and validates the
   `BranchRouter`'s dot-product snap over a global mean. Our design is on the winning side.

2. **Linear separability of success/fail hidden states (88–99% SVM accuracy).** This validates
   that contrastive direction vectors are well-defined in latent space — the geometric
   precondition for CNA, CommittedFieldBlend, and the BranchRouter's orthogonal-subspace
   isolation. NPM Appendix C.1 is independent empirical evidence for the precondition our
   shipped primitives assume.

3. **Hybrid (implicit + explicit) > either alone.** Validates MUX-Latent (Plan 238) and SwiR
   (Plan 275) — the latent+explicit hybrid pattern is the structurally-correct composition.

---

## 5. Caveats

- **NPM is LLM-agent-centric.** It assumes a frozen LLM with accessible residual stream —
  the "open architecture" requirement the paper lists as a limitation. Our codebase has no
  LLM-in-the-loop on the hot path; the analog operates on per-NPC HLA state, which is a
  different (and stricter) setting.
- **NPM's KL-bounded α (Eq. 7) is a workaround for residual-stream quality collapse**, the
  same collapse R290 §1.2 cites as the reason CAA was rejected for the LLM side. The
  workaround is unnecessary in our latent-state setting because the 8-D HLA target is bounded
  and behavior-rank-preserving by the G2 gate.
- **NPM has no commitment / persistence / sync story.** Direction vectors are recomputed
  offline and loaded at init; there is no analog to `MerkleFrozenEnvelope`, no sampling
  invariance, no LatCal sync-boundary crossing. For an LLM-agent product this is acceptable;
  for an MMORPG-scale game AI product it is a non-starter. Our shipped primitives are
  strictly stronger on this axis.
- **The paper's "procedural memory" framing is a rebrand of activation steering for the
  agent-memory literature.** The mechanism is CAA-with-dynamic-synthesis; the contribution is
  applying it to multi-step agent trajectories with dual-granularity contrast. Both are
  subsumed by our shipped stack.

---

## TL;DR

NPM = training-free contrastive-pair → PCA-steering-vector → residual-stream-injection for
LLM agents, with dual-granularity (inter + intra-trajectory) contrast and KL-bounded
intervention strength. **Verdict: Pass** — every load-bearing mechanism already ships in this
codebase, in pure latent space (HLA 8-D + MLP neurons + CognitiveBranches), stronger than
NPM's LLM-residual-stream form: CNA (Plan 087) is the published-stronger cousin of NPM's
residual steering; Latent Field Steering (R290 / Plan 309) is the same math on a bounded
behavior-rank-preserving target; CommittedFieldBlend (Plan 321) + Non-Interference Branches
(Plan 329, DEFAULT-ON) + MerkleFrozenEnvelope add commitment / sampling-invariance /
non-interference that NPM lacks entirely. The single useful **insight** (intra-trajectory
contrast from one failed trajectory → direction vector for the failure store) is a one-method
enrichment to CLR's failure path on the existing CognitiveBranch primitive — tracked in
`riir-ai/.issues/`, not a katgpt-rs plan. The paper is independent confirmation that
locality-aware dynamic synthesis beats static dataset-wide steering (NPM Table 1: CAA/Mass-Mean
lose to task-specific NPM; validates R290 + BranchRouter design) and that success/fail modes
are linearly separable in latent space (NPM Appendix C.1; validates CNA + CommittedFieldBlend
preconditions). No files created in this session beyond this note.
