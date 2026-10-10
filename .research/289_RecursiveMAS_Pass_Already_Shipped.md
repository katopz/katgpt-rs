# Research 289: RecursiveMAS — Recursive Multi-Agent Systems (PASS — already shipped)

> **Source:** [Recursive Multi-Agent Systems (RecursiveMAS)](https://arxiv.org/pdf/2604.25917) — Yang, Zou, Pan, Qiu, Lu, Diao, Jiang, Tong, Zhang, Buehler, He, Zou (UIUC + Stanford + NVIDIA + MIT), Apr 2026
> **Date:** 2026-06-22
> **Status:** Done — closed.
> **Classification:** Public
> **Related Research:** 247 (Dense Latent Heterogeneous Comms — **the same research family, already Super-GOAT**), 073/108 (LT2 Looped), 097/136 (Training-Free Loop Wrapper), 273 (ELT Elastic Looped), 242/276 (Topological recurrent belief), 243/277 (Temporal Derivative Kernel), 192/217 (NextLat BeliefDrafter), 123/273/303 (Latent Functor), 257/286 (FuncAttn rank-k projection), 251 (Token Economics — explicitly notes LatentMAS/Q-KVComm covered)
> **Related Plans:** 311 (riir-ai NPC mind-reading runtime — **the Super-GOAT we already shipped, more sophisticated than this paper**), 280 (CS-KV-Importance Probe), 108 (LT2 Looped forward), 273/303 (Latent Functor), 286/318 (FuncAttn rank-k), 283 (Self-Advantage Recursion Gate), 304 (Gain/Cost Loop Halting)
> **Verdict: PASS.** Every individual primitive RecursiveMAS introduces is already shipped in our quintet — most at higher fidelity. The paper's value is its **training recipe** (inner-outer loop co-optimization with backprop through frozen LLMs) → **riir-train**. No file/plan/guide created beyond this classification note.
> **PASS-Redirects (synthesis):** Kim et al. [arXiv:2608.24735 "Meta^n: Recursive Self-Improvement through Emergent Depth"] — same bi-level-already-shipped verdict class (Research 512): fixed Ω + growing input, every extracted principle ships grep-verified; Ω itself remains LLM code-generation.
> **PASS-Redirects (synthesis):** Lee, Xu, Seely, Lee, Sojoudi, Zaharia, Tang [arXiv:2610.12176 "Recursive Self-Improvement through Multi-Agent Self-Supervision"] — homogeneous RSI alternating elite-1 evolutionary multi-agent workflow search (LLM text-gen proposer, R440 class) with SFT on BT-ranked self-generated multi-agent traces; the SFT-recipe half is the actionable Gain → riir-train Plan 456. See Research 616.

---

## TL;DR

RecursiveMAS casts a multi-agent system as a recursive computation in latent space. Two pieces:

1. **RecursiveLink** (inner: `R_in(h) = h + W2·σ(W1·h)`; outer: `R_out(h) = W3·h + W2·σ(W1·h)`): a 2-layer residual projection that (a) feeds an agent's last-layer hidden state back as next input embedding for intra-agent latent thoughts generation, and (b) projects across heterogeneous model embedding spaces for cross-agent latent comms. Only the last agent decodes text in the final recursion round.
2. **Inner-outer loop training**: warm-start inner link per-agent via cosine regression against ground-truth embedding distribution; then co-optimize outer links via cross-entropy through R recursion rounds with **gradient backpropagation through frozen LLMs** (RecursiveLink weights are the only trainable parameters).

Reported: +8.3% avg accuracy, 1.2–2.4× inference speedup, 34.6–75.6% token reduction over text-based Recursive MAS at r=1/2/3. Cost: 13.12M trainable params (0.31% of total) — less than LoRA.

**Distilled for katgpt-rs (modelless, inference-time):** nothing not already shipped. The RecursiveLink artifact, once trained, IS a frozen adapter (latent projection). It slots into our existing Plan 311 (NPC mind-reading adaptive-bandwidth latent bus) + Plan 303 (latent_functor rank-1) + Plan 318 (FuncAttn rank-k). Training the projection weights → riir-train.

---

## 1. Paper Core Findings

### 1.1 The two transferable pieces

| Piece | What it does | Our prior-art status |
|---|---|---|
| **Inner RecursiveLink** — `h_{t+1} = R_in(f_θ([E_{≤t}; h_t]))` | Intra-agent autoregressive generation in latent space (no decode between steps). Maps last-layer hidden back to input embedding space. | ✅ **Already shipped under many names:** `evolve_hla` (R242/P276 Family C leaky integrator), NextLat BeliefDrafter (R192/P217), MicroRecurrentBeliefState (P276), LatentThoughtKernel (P276 Family B), Temporal Derivative Kernel (R243/P277). The leaky-integrator/`h + f(h)` residual form is exactly R_in. |
| **Outer RecursiveLink** — `R_out(h) = W3·h + W2·σ(W1·h)` | Cross-model latent projection for heterogeneous agent embedding alignment. Enables agent→agent latent comms without text decode. | ✅ **Already Super-GOAT, shipped at higher fidelity:** R247 + R133 + P311 + P280 (NPC mind-reading adaptive-bandwidth latent bus). Our version adds the **fog-of-war context-awareness axis** that RecursiveMAS does NOT have — sparse 3.5% when receiver has line-of-sight, dense 87% when blind, gated by `ca = sigmoid(β·coverage_overlap)`. RecursiveMAS uses a fixed `W3·h` projection only. **Our system is strictly more capable.** |
| **System-as-loop topology** (chain agents A1→A2→…→AN→A1, R rounds) | Treat the MAS as a recursive computation where each agent is one "layer" of an RLM. | ✅ Implicitly shipped: NPCs already have per-tick HLA evolution (intra-agent recursion) + Plan 311 latent broadcast (cross-agent). Our topology is **more flexible** (pub/sub with fog-of-war gating, not a fixed loop). |
| **Recursion depth scaling** (R=1,2,3 as inference-time compute axis) | Increase R for deeper refinement. | ✅ Shipped as LT2 `forward_looped` (R073/P108), Training-Free Loop Wrapper (R097/P136), ELT (R273). Halting primitives: Self-Advantage Recursion Gate (P283), Gain/Cost Loop Halting (P304), Depth-Invariance Diagnostic (R286/P306). |
| **Residual projection form** `h + W2·σ(W1·h)` | 2-layer MLP with residual preservation of input semantics. | ✅ Shipped: latent_functor rank-1 (`f = mean_k(target_k - source_k)` is the rank-1 special case, P303), FuncAttn rank-k (`C = Q̃K̃ᵀ(K̃K̃ᵀ+λI)^-1`, P286+P318 generalizes to operator-valued). Both **atomic Arc-swap, BLAKE3-committed** per `riir-ai/crates/riir-engine/src/latent_functor/table.rs::FunctorEntry`. |
| **Inner-outer loop training** (backprop through frozen LLMs) | The only genuinely additive contribution. Co-optimizes outer links via CE through full recursion trace. | ⛔ **Training → riir-train.** Out of scope for this workflow. |

### 1.2 The four collaboration patterns (§5.3)

Sequential / Mixture / Distillation / Deliberation — all are **MAS topology templates**, not new primitives. Our game runtime ships pub/sub, federation coupling (Plan 231), polytope routing (R091), Dynamic Pair (P260), dMoE block-level routing (R161), Crowd MCGS (P298). Each pattern maps to one of our existing topologies.

---

## 2. Distillation

### 2.1 What's training-only → riir-train

- **Inner-outer loop co-optimization** with backprop through frozen LLMs (Theorem 4.1's gradient-stability proof is a training-side result).
- **Cosine regression warm-up** for inner link (Eq. 5).
- **Recursive CE training** with full computation-graph retention across R rounds (Eq. 6).
- **AdamW learning rate 5e-4**, cosine schedule, batch size 4 — pure training hyperparameters.

### 2.2 What's modelless but already shipped

| RecursiveMAS modelless primitive | Shipped cousin | Plan / Research |
|---|---|---|
| Inner link intra-agent latent thoughts | `evolve_hla`, MicroRecurrentBeliefState Family C, NextLat BeliefDrafter | P057, P276, P217 |
| Outer link cross-agent latent projection | **NPC mind-reading adaptive-bandwidth latent bus** | **R247, R133, P311, P280** |
| Heterogeneous model embedding alignment | ShardKV RoPE-strip / Still Perceiver un-rotate | P147, P245 |
| Recursion depth scaling (R rounds) | LT2 `forward_looped`, Training-Free Loop Wrapper | P108, P136 |
| Recursion depth halting | Self-Advantage Gate, Gain/Cost Halter, Depth-Invariance | P283, P304, R286 |
| Residual MLP projection form | latent_functor rank-1 (f), FuncAttn rank-k (C operator) | P273/P303, P286/P318 |
| Freeze/thaw of projection weights | `riir-ai/crates/riir-engine/src/latent_functor/table.rs::FunctorEntry` (atomic Arc-swap, BLAKE3-committed, Uuid::now_v7 versioned) | P303 |
| Coherence-driven re-estimation on drift | `riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs::ReestimationScheduler` (DiPOD equivalent) | P303 |
| Latent-to-text decode only at final round | Standard inference path (decode is already only-at-output by design) | existing |
| Sequential/Mixture/Distillation/Deliberation topologies | Polytope, Dynamic Pair, dMoE, Crowd MCGS, federation | R091/P260/R161/P298/P231 |

### 2.3 Fusion — none novel (the prior-art surface is dense)

RecursiveMAS is the same research family as the works R251 (Token Economics) already noted are **covered by Plan 311**: LatentMAS (Zou et al. 2025 — same senior-author group as RecursiveMAS), Q-KVComm, TokenDance, KVComm. From R251 §2.2:

> "**T4 representational token exchange** (LatentMAS [119], Q-KVComm [113], TokenDance [4]) → NPC Mind-Reading Adaptive Bandwidth — sparse 3.5% context-aware → dense 87% context-unaware, gated by fog-of-war. **Super-GOAT, already active.**"

RecursiveMAS extends LatentMAS with **recursion depth** as an axis. We already ship recursion depth (LT2, P108) + cross-agent latent comms (P311). The combination is implicit in our codebase.

The one genuinely additive angle: **RecursiveLink weights as a new freeze/thaw artifact class** — a frozen MLP that bridges heterogeneous model embedding spaces. This is a trivial extension of the existing freeze/thaw pattern (already covers LoRA adapters, latent_functor direction vectors, kernel snapshots, NeuronShard style_weights). No new plan needed; the next time Plan 311 needs a cross-shape projection for heterogeneous NPC classes, it can ship as a `FunctorEntry` with kind `Operator` (rank-k via FuncAttn) — already covered by Plan 318.

### 2.4 Latent vs raw boundary (mandatory check)

Not applicable — no new boundary-crossing behavior. RecursiveMAS's latent-to-latent comms are intra-system; only the last agent's final output crosses to text. Our Plan 311 already enforces the same boundary discipline (dense HLA stays local-zone, 5-scalar sync rule unchanged).

---

## 3. Verdict

**Tier: PASS.** Training recipe → riir-train. Inference-time primitives already shipped at higher fidelity.

| Gate | Criterion | Honest answer |
|---|---|---|
| **Q1** No prior art? | **FAIL.** Every individual primitive ships in our quintet. The headline outer-link cross-agent comms is **already Super-GOAT** (R247/R133/P311) with the fog-of-war axis RecursiveMAS lacks. |
| **Q2** New class of behavior? | **FAIL.** "Recursive latent collaboration" = single-model recursion (shipped) + cross-agent latent comms (shipped). Our topology (per-tick HLA evolution + Plan 311 broadcast) is more expressive than RecursiveMAS's fixed N-agent loop. |
| **Q3** Selling point? | **FAIL for new selling point.** "Agents collaborate in latent space, no text decode" IS the Plan 311 selling point — and our version adds fog-of-war context-awareness. |
| **Q4** Force multiplier? | **YES — but only as a redescription** of capabilities we already have. Connects HLA, latent_functor, Plan 311, P108 — all already connected. |

### Latent-space reframing check (mandatory per skill — primary framing)

- **HLA framing:** RecursiveMAS = "N NPCs each iterate HLA state, then exchange HLA slices, looping R rounds." Ours: NPCs evolve HLA per-tick AND exchange via Plan 311 every tick. The "loop R rounds before producing output" structure is *more rigid* than per-tick evolution.
- **Latent functor framing:** `R_out` IS a latent functor direction vector (rank-1). We already ship rank-k FuncAttn (Plan 318) as the generalization.
- **CGSP framing:** recursion-round scaling ≈ CGSP cycle scaling — already shipped, no fixed-N-loop constraint.
- **Neuron-shard framing:** RecursiveLink weights = another freeze/thaw artifact (`MerkleFrozenEnvelope` covers it). Already covered by `riir-ai/crates/riir-engine/src/latent_functor/table.rs::FunctorEntry`.
- **LatCal framing:** no natural LatCal angle — `R_out` is a learned linear map, not a deterministic committed fixed-point bridge.

**No latent-space reframing yields a new capability.** Adapter-routing framing would be even weaker (we already ship Dynamic Pair, Polytope, dMoE).

### Honest one-line reasoning

RecursiveMAS's training recipe (inner-outer loop co-optimization with backprop through frozen LLMs) is its only additive contribution → riir-train. Every modelless primitive is already shipped in our quintet; the cross-agent latent comms selling point is **already Super-GOAT** (R247/R133 → P311/280) with the fog-of-war adaptive-bandwidth axis that RecursiveMAS does not have.

---

## 4. Routing

- **Training recipe** (inner-outer loop, gradient through frozen LLMs, cosine warm-up) → **riir-train** (one-line note, out of scope for this workflow).
- **Open primitive** → none new. The RecursiveLink artifact, once trained, slots into Plan 311 (NPC mind-reading) as a `FunctorEntry` of kind `Operator` (rank-k via Plan 318 FuncAttn).
- **Architectural guide** → none required. R133 already covers the game-side selling point at higher fidelity.
- **Plan** → none required. No new code needed.
- **Plan 311 follow-up note:** if heterogeneous-NPC-class (Knight↔Mage HLA cross-shape) ever needs a trained projection bridge, that projection artifact IS RecursiveMAS's outer RecursiveLink — and it can be frozen as a `FunctorEntry`. No new mechanism — just another freeze/thaw artifact class. Currently noted as P3 in R133.

---

## TL;DR

RecursiveMAS = LatentMAS (already Super-GOAT, R247/R133/P311) + recursion depth (already shipped, LT2 P108) + a training recipe (→ riir-train). The cross-agent latent comms selling point is already ours with the fog-of-war context-awareness axis that RecursiveMAS lacks. No new primitive, no new plan, no new guide. Closing this research path; the only action item is a one-line note for riir-train capturing the training recipe in case Plan 311 ever needs a trained cross-shape projection for heterogeneous NPC classes (currently P3 in R133, unblocked).
