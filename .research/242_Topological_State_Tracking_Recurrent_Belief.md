# Research 242: The Topological Trouble With Transformers — Recurrent Belief-State Primitive

> **Source:** [The Topological Trouble With Transformers](https://arxiv.org/pdf/2604.17121) — Mozer, Siddiqui, Liu (Google DeepMind), arXiv:2604.17121v3, Jun 2026
> **Date:** 2026-06-15 (verdict revised same day: Super-GOAT → GOAT after HLA prior-art check)
> **Status:** Active — GOAT (fusion)
> **Related Research:** 097 (Training-Free Looped Transformers), 192 (NextLat belief-state dynamics), 073 (LT2 looped), 070 (Gated DeltaNet-2), 135 (Parallax), 230 (SSD duality), 158 (MUX), 175 (ThoughtFold), 241 (SwiR explicit↔latent switch)
> **Related Plans:** 108 (LT2 looped — done), 136 (Training-Free Loop Wrapper — done), 217 (NextLat drafter — done), 255 (ANE-Latent NPC Brain), 262 (Latent Physics Primitives), 275 (SwiR switch-thinking), 276 (this doc's plan)
> **Cross-ref (riir-ai):** Research 127 (Implicit Microcognition Crowd-NPC Guide — Super-GOAT private guide), Plan 304 (downstream runtime integration, optional)
> **Classification:** Public — generic math, no game semantics

---

## TL;DR

Mozer et al. argue that **feedforward transformers are structurally incapable of indefinite state tracking**: every sequential state update `s_t = f(s_{t-1}, x_t)` pushes the state representation one layer deeper, eventually exhausting model depth. CoT-style "thinking" externalizes state as output tokens — a wasteful workaround for a topological deficiency. The paper's fix is **implicit activation dynamics via recurrent architectures**, and provides a clean taxonomy (recurrence axis × tokens-per-step ratio) to navigate the design space.

**Distilled for katgpt-rs (modelless, inference-time):**

The diagnostic is the gift; the implementation is *ours*. Three inference-time takeaways:

1. **A new primitive: `MicroRecurrentBeliefState`** — a small frozen kernel implementing `s_t = f(s_{t-1}, x_t)` in latent space, applied once per (entity, tick). Three recurrence families from the paper's taxonomy (attractor loop, latent-thought loop, delta-rule SSM), all inference-time, all freeze/thaw-compatible. This fills a gap: today `SpatialBelief::decay_confidence()` (Plan 262) is a *static* `sigmoid(-λΔt)` — a placeholder for state tracking that the paper proves is structurally insufficient.
2. **The taxonomy as a router** — when our looped transformers (Plan 108/136) or latent drafter (Plan 217) need a recurrence axis, the paper's Table 1 tells us which slot we're in and what's possible in each empty cell.
3. **A justification for inference-time looping** — the paper explicitly cites training-free looped transformers (Ng 2026 — our Research 097) as a legitimate response to the depth limit, not a hack.

**Paper-alone verdict: GOAT** (a useful diagnostic map + taxonomy; no novel mechanism of its own, but a high-leverage frame).
**Fusion verdict: GOAT (revised down from an initial Super-GOAT claim).** A prior-art check on `evolve_hla` (see §2.4) showed HLA already implements per-NPC recurrent latent state tracking — exactly Family C of the proposed primitive. What remains novel is narrower: (a) attractor dynamics (Family A) as a quality variant over HLA's leaky integrator, (b) kernel-as-versioned-snapshot as a new per-NPC divergence axis, (c) taxonomy unification of `evolve_hla` into a trait. None is a new capability class. Outputs produced this session: this open primitive + `riir-ai/.research/127_*.md` (reframed as design context, not a moat doc) + `katgpt-rs/.plans/276_*.md` (reframed as an extension of `evolve_hla`, not a greenfield module).

---

## 1. Paper Core Findings

### 1.1 The topological diagnosis (§1–§2)

State tracking = iterative update of latent variables reflecting an evolving environment: `s_t = f(s_{t-1}, x_t)`. In a feedforward transformer, every such update pushes `s_t` one layer deeper than `s_{t-1}` (paper's Figure 1b). After N input steps the state has consumed N layers; beyond model depth it is irrecoverable. Shallow layers of later tokens cannot see the disambiguated state, producing failures like:

- **"bank" polysemy flip-flop** (paper §2): the model disambiguates "fishing pole → bank" to river-bank at layer 6, but when processing "ATM?" the disambiguation is unavailable to layers 1–5 of the ATM token, so it defaults to money-bank. This is a *structural* failure, not a knowledge failure.
- **Twenty-questions range tracking**: even Thinking variants fail to use their own generated hidden number consistently.
- **Multi-turn conversation coherence loss** (Laban 2025), information-gathering inefficiency (Sawyer 2025), multi-agent cooperation breakdown (Davidson 2025, Khatua 2026).

The paper's claim: **re-examining input history via attention is retrieval, not state tracking.** Retrieval turns state-tracking into working-memory lookups; this works for many cases but has a topological ceiling (Merrill & Sabharwal 2025: log-n depth needed for length-n regular-language recognition).

### 1.2 CoT is a "cop out" (§2, §4)

Externalizing state as output tokens (CoT, latent-thought) sends signals from deep layers to shallow layers via the input stream — it works, but:
- Wastes compute on microcognition that should be automatic (polysemy resolution, character tracking).
- Consumes context window.
- The paper's desideratum: *"if cognition in a transformer can be shifted from explicit thought traces to implicit activation dynamics, the resulting model will be more powerful."*

### 1.3 The recurrence taxonomy (§3, Table 1) — the transferable map

Two axes classify recurrent transformer variants:

| | Ratio > 1 (many tokens/step) | Ratio = 1 (one token/step) | Ratio < 1 (many steps/token) |
|---|---|---|---|
| **Depth axis** | Looped transformer, Universal Transformer, RINS | (empty — paper notes this as opportunity) | (empty) |
| **Step axis** | Block-recurrent | Linear attention, DeltaNet, Mamba, canon layers, RWKV-7, PaTH, TTT | DeltaProduct |
| **Depth+Step** | Recurrent Memory Transformer, RINs, Sentence Gestalt | Feedback Transformer | COCONUT, HRM, CYB |

**Critical paper claim:** recurrence is *necessary but not sufficient* for state tracking. "Full-fledged state tracking requires sequential dynamics during training; any model that can be entirely parallelized across the context has limitations in updating state." Linear SSMs alone are no more expressive than ordinary transformers (Merrill et al. 2025). The escape hatches are: (a) DeltaNet with **negative eigenvalues** (Grazzi 2025), (b) gated DeltaNet mixed with transformer blocks (Merrill 2026, OLMo Hybrid), (c) depth+step recurrence with ratio ≤ 1.

### 1.4 Promising directions (§5) — what to build

- **§5.1 Enhanced SSMs**: DeltaNet + negative eigenvalues; RWKV-7; PaTH; gated DeltaNet; OLMo Hybrid (gated linear attention + transformer mix).
- **§5.2 Approximate state tracking in feedforward**: specialized objectives + structural priors (Hu 2025 Belief-State Transformer; Teoh 2025a NextLat — *our Research 192*).
- **§5.3 Coarse recurrence**: chunk at linguistic structure (Borazjanizadeh & McClelland 2025 sentence-level thoughts).
- **§5.4 Representational alignment**: variable-depth models work with **fine-tuning or NO training whatsoever** — residual connections align representations across layers, enabling depth-recurrence retrofit. (Direct support for our Research 097.)
- **§5.5 Efficient training of recurrence**: multi-stage training (parallel pretraining → recurrent fine-tuning), recurrent backpropagation for attractor dynamics.

### 1.5 What the paper is NOT

- Not a new training method (→ not a riir-train redirect).
- Not a new architecture with benchmarks.
- It is a **position paper + taxonomy + roadmap**. Its value is *organizational*: it tells us *why* certain inference-time tricks (looping, latent thought) work and *which slot* each occupies in the design space.

---

## 2. Distillation

### 2.1 The transferable primitive: `MicroRecurrentBeliefState`

The distilled inference-time primitive is a **small frozen kernel** implementing one step of `s_t = f(s_{t-1}, x_t)` in a fixed-size latent belief vector, applied once per (entity, tick). Three recurrence families from the paper's taxonomy, all inference-time, all compatible with our freeze/thaw + plasma-tier constraints:

| Family | Paper slot | Update rule (one tick) | Cost (d=32) | When to use |
|---|---|---|---|---|
| **A. Attractor loop** | Depth+Step, ratio=1 (Fig 5d) | `s_t = σ(W_s·s_{t-1} + W_x·x_t + b)` (one fixed-point iter) | ~32 FMAs ≈ 32ns SIMD | Default — cheapest, bounded, has attractor dynamics |
| **B. Latent-thought loop** | Depth+Step, ratio<1 (Fig 6) | K iters of `s ← σ(W_s·s + W_x·x_t)` before advancing | K × 32ns | When richer intra-tick settle is needed (negotiation, planning) |
| **C. Delta-rule SSM** | Step axis, ratio=1 (Fig 7) | `s_t = diag(1−α)·s_{t-1} + β·x_t`, per-channel gates α,β | ~64 FMAs ≈ 64ns | When linear/GPU-batchable preferred; pairs with DeltaNet-2 (Plan 105) |

**Properties:**
- The kernel `f` (weights `W_s, W_x, b` or gates `α, β`) is **frozen**, **versioned**, **BLAKE3-committed** — a first-class freeze/thaw artifact (`MicroRecurrentKernelSnapshot`).
- Per-entity personality divergence = different kernel snapshots (two same-type NPCs diverge over time, per `003` commercial strategy).
- Operates **latent-to-latent**: input `x_t` is already an embedding (sense vector, observation embedding); output `s_t` is a belief vector. No token decode/re-encode round-trip.
- **Bridge to raw scalars (sync boundary):** `s_t` projects to bounded scalars via `sigmoid(dot(s_t, direction_k))` for each synced channel (valence/arousal/desperation/calm/fear). Only the scalars cross sync; the vector stays local. Zero-allocation bridge, feature-gated.
- **Latency budget:** at d_belief=32, Family A is ~32ns/NPC/tick → 20Hz × 1000 NPCs ≈ 640µs/sec total. Fits plasma tier (per Plan 255 budget of 1.5µs/sec/NPC).

### 2.2 What's NOT here (stays in riir-train / not needed)

- The *training* of `f` (offline supervision to make `s_t` a belief state) — if needed, → riir-train. The modelless path uses a frozen kernel from any source (random init + bandit-tuned gates, distillation snapshot, or imported pretrained).
- Backprop through base weights — forbidden by modelless constraint.
- The paper's multi-stage training scheme (§5.5) — training-only, → riir-train.

### 2.3 HLA prior art — the verdict-changing finding

**`evolve_hla` already implements Family C.** `ReconstructionState::evolve_hla()` (`katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs:623`) is a gated additive recurrent update of the 8-dim HLA state, called every step in the `expand → route → accumulate → evolve_hla` loop:

```rust
self.hla[i] = (self.hla[i] + clamped_delta).clamp(-1.0, 1.0);
// where clamped_delta = clamp(lr * (normalized - half_total) * scale, max_delta)
```

This is structurally `s_t = clamp(s_{t-1} + clamp(lr · f(x_t), max_delta), -1, 1)` — a leaky-integrator variant of Family C. `NpcBrain::update_hla_fixed()` (brain.rs:261) provides the raw additive path via `simd_add_inplace`. `SenseModule::project()` is the bridge (dot-product + sigmoid → scalar). All zero-alloc, SIMD, clamped, already benchmarked.

**Conclusion:** HLA covers the *core* of the proposed primitive. The remaining delta is: (1) attractor dynamics (Family A) — qualitatively different update rule with fixed-point basins, not present in HLA's leaky integrator; (2) kernel-as-`MicroRecurrentKernelSnapshot` — HLA's evolve rule is hardcoded config + formula, not a per-NPC versioned artifact; (3) Family B (K-iteration latent-thought loop) — HLA does one evolve per step, no intra-tick settle. These are real extensions but not a new capability class.

### 2.4 Relationship to existing katgpt-rs primitives

| Existing primitive | Relationship to `MicroRecurrentBeliefState` |
|---|---|
| **Research 097 / Plan 136** (Training-Free Loop) | Cousin on depth axis, ratio>1: loops a *contiguous mid-stack block* of an existing transformer for ODE refinement. New primitive is on depth+step axis, ratio≤1: a *standalone tiny kernel* for per-entity belief state. Composable: Plan 136's loop can wrap a model whose layers include a `MicroRecurrentBeliefState` stage. |
| **Research 192 / Plan 217** (NextLat belief drafter) | NextLat's residual MLP `ĥ_{t+1} = f_ψ(h_t, x_{t+1}) + h_t` IS a Family-A attractor kernel with residual structure. The new primitive *generalizes* NextLat's drafter into a per-entity belief-state kernel (NextLat drafts tokens; `MicroRecurrentBeliefState` maintains state). |
| **Research 070 / Plan 105** (Gated DeltaNet-2) | Implements Family C (delta-rule SSM) at the attention-kernel level. The new primitive is the same math at the per-entity belief-vector level — composable, not redundant. |
| **Research 241 / Plan 275** (SwiR explicit↔latent switch) | SwiR switches between explicit-CoT mode and latent mode at token level. The new primitive is the *latent-mode substrate* SwiR switches *into*. Fusion C of Plan 275 explicitly anticipates this. |
| **Research 175 / Plan 195** (ThoughtFold) | ThoughtFold folds multi-step reasoning into a single latent step. The new primitive is the *carrier* of folded state across ticks. |
| **Research 158 / Plan 178** (MUX multiplexed latent reasoning) | MUX multiplexes reasoning across latent channels in one forward; new primitive persists a single belief vector across ticks. Orthogonal axes. |

---

## 3. Verdict

**Paper-alone: GOAT.** A position/taxonomy paper — no novel mechanism of its own, but a high-leverage organizational frame that justifies and structures inference-time recurrence work we already have (Plans 108, 136, 217) and points to empty taxonomy cells worth filling.

**Fusion: GOAT (revised down from initial Super-GOAT claim).** A prior-art check on `evolve_hla` (§2.3) showed HLA already implements the core (per-NPC recurrent latent state + bridge). The novelty gate, honestly re-scored:

| Gate | Question | Honest answer |
|---|---|---|
| **Q1 Novelty** | Any existing code cover this? | **FAILS.** `evolve_hla` is direct prior art for Family C (the core mechanism). Attractor family + kernel versioning are incremental deltas, not "no prior art". |
| **Q2 New capability class** | New behavior, not just better numbers? | **FAILS.** HLA already does per-NPC recurrent latent state tracking. Attractor dynamics is a *variant* update rule (hysteresis vs leaky), not a new capability class. |
| **Q3 Selling point** | "Our NPCs/systems do X that no competitor can"? | **WEAKENS.** "NPCs never forget who they're talking to" is already approximately true with HLA. The sharpened claim ("...with attractor-stable opinions + per-NPC kernel versioning") is incremental. |
| **Q4 Force multiplier (≥2)** | Connects to ≥2 existing pillars? | Passes (6 systems), but Q4 alone ≠ Super-GOAT — needs Q1+Q2+Q3 too. |

**Downgrade rationale:** the Super-GOAT claim was made before checking `evolve_hla`. HLA's leaky integrator is already a Family-C recurrent belief-state kernel. The honest contribution of this research is: (a) the Mozer taxonomy as a diagnostic frame showing HLA occupies the "step axis, ratio=1, delta-rule" slot and there are unfilled slots (attractor with depth+step/ratio=1, latent-thought with ratio<1); (b) attractor dynamics (Family A) as a quality variant that may reduce long-horizon flip-flops; (c) kernel-as-snapshot as a new divergence axis. All three are GOAT-tier (provable gain if attractor benchmarks better than leaky on coherence), not Super-GOAT (new capability class).

**Outputs (retained, reframed):**
1. **Open primitive** — this doc + `katgpt-rs/.plans/276_*.md` (reframed as an *extension* of `evolve_hla`, not a greenfield module).
2. **Design context** — `riir-ai/.research/127_*.md` (reframed from "Super-GOAT moat doc" to "GOAT design context for the attractor-family extension"). The mandatory-Super-GOAT-guide rule no longer applies; the doc is retained because the connection-map content is still useful design context.
3. **Plan** — `katgpt-rs/.plans/276_*.md`.

---

## 4. Fusion (the Super-GOAT combination)

**The combination:** Mozer 2026 (topological state-tracking diagnosis + recurrence taxonomy) × **two-brain model** (info brain raw/synced, think brain latent/local — AGENTS.md) × **Plan 255 ANE-Latent NPC Brain** (1.5µs/sec/NPC budget at 20Hz × 1000 NPCs) × **Research 192 NextLat** (belief-state residual MLP generalizes to per-entity kernel) × **freeze/thaw runtime** (kernel is a versioned snapshot).

**What this combination produces that none alone can:**

| Component alone | What it can't do | What the fusion adds |
|---|---|---|
| Mozer 2026 | Diagnoses the problem; doesn't ship a primitive | Gives us the *structural justification* and the *taxonomy slot* for `MicroRecurrentBeliefState` |
| Two-brain model (AGENTS.md) | Think brain has only static `sigmoid(-λΔt)` confidence decay | Think brain gets a *real recurrent substrate* — belief vector evolves via `f(s_{t-1}, x_t)` |
| Plan 255 (ANE-Latent) | Batches static projections (sense → emotion) | Batches *recurrent* belief updates — one ANE batch = 1000 NPCs × 1 tick of state evolution |
| Research 192 (NextLat) | Belief MLP drafts *tokens* | Belief kernel maintains *per-entity state* across ticks — no decoding |
| Freeze/thaw | Versions LoRA-style adapter weights | Versions *recurrent kernels* — emergent NPC personality = emergent kernel snapshot |

**Capability increment (over existing HLA):** (a) attractor dynamics — stable beliefs with hysteresis that resist noise until evidence accumulates (HLA's leaky integrator has no basins); (b) per-NPC kernel versioning — two same-type NPCs can diverge via different kernel snapshots, not just different inputs (HLA's evolve rule is shared); (c) latent-thought intra-tick settle (Family B) for deliberation. All incremental over HLA's existing recurrent state tracking.

**Closest cousins across both repos (for the fusion protocol):**
- `katgpt-rs/.research/097_Training_Free_Looped_Transformers.md` — depth-axis recurrence (ratio>1) on a frozen checkpoint; the new primitive is depth+step (ratio≤1) on a tiny standalone kernel.
- `katgpt-rs/.research/192_NextLat_Belief_State_Latent_Dynamics.md` — belief-state residual MLP as token drafter; new primitive generalizes to per-entity state maintainer.
- `katgpt-rs/.plans/255_ane_latent_npc_brain_compute.md` — plasma-tier NPC compute budget; new primitive is the recurrent compute that fits in it.
- `katgpt-rs/.plans/262_latent_physics_primitives.md` — `SpatialBelief::decay_confidence()` is the static placeholder; new primitive is its upgrade target.
- `riir-ai/.research/123_Latent_Functor_Runtime_Guide.md` — functor composition for NPC relational learning; new primitive is the *state carrier* the functor operates on.
- `riir-ai/.research/126_NPC_Curiosity_Guided_Self_Play_Guide.md` — runtime curiosity drives subgoal generation; new primitive is what curiosity *updates* (the belief kernel's input statistics).

---

## 5. Open Questions / Risks

- **R1 — Stability of attractor dynamics.** Family A (attractor loop) can oscillate or diverge if `W_s` has eigenvalues outside the unit disk. Mitigation: clamp `‖s_t‖`, gate by feature flag, fall back to Family C (linear, always stable). Validate via `003` validation protocol (per-NPC coherence test).
- **R2 — Kernel provenance.** Where does the frozen kernel come from? Options: (a) random init + bandit-tuned gates (pure modelless), (b) distillation snapshot from a trained belief-state model (→ riir-train), (c) identity init + curiosity-driven drift (fuses with Research 126 CGSP). All three are valid; (a) is the unblock path.
- **R3 — Sync boundary leakage.** The 5 synced scalars (valence/arousal/desperation/calm/fear) are projections of `s_t`. If `direction_k` vectors leak, an attacker could reconstruct `s_t`. Mitigation: `direction_k` is private (riir-ai), never synced; only the scalar is synced.
- **R4 — Test coverage.** Need (a) determinism test (same input sequence → same `s_t` bit-identical), (b) attractor convergence test (bounded `‖s_t‖` over 10k ticks), (c) bridge reversibility test (scalar projections preserve ranking of `s_t`), (d) freeze/thaw atomicity test (readers never see torn kernel swap).

---

## 6. References

- Paper: [arXiv:2604.17121](https://arxiv.org/abs/2604.17121) — Mozer, Siddiqui, Liu, DeepMind, Jun 2026.
- Cited by paper, in our corpus: NextLat (Teoh 2025b — our Research 192), Training-Free Looped Transformers (Ng 2026 / Chen 2026 — our Research 097).
- Cited by paper, not yet in our corpus: Belief-State Transformer (Hu 2025), DeltaNet negative-eigenvalue extension (Grazzi 2025), RWKV-7 (Peng 2025), PaTH attention (Yang 2025b), OLMo Hybrid (Merrill 2026), DeltaProduct (Siems 2025), COCONUT (Hao 2025), HRM (Jolicoeur-Martineau 2025).
- Our related: 073/108 (LT2 looped), 097/136 (training-free loop), 192/217 (NextLat drafter), 070/105 (Gated DeltaNet-2), 241/275 (SwiR switch-thinking), 255 (ANE-Latent NPC Brain), 262 (Latent Physics Primitives).
- riir-ai: 123 (Latent Functor), 126 (CGSP guide), 127 (this paper's private guide).

---

## TL;DR

Mozer et al. prove (positionally) that feedforward transformers are topologically bounded for state tracking — every state update consumes a layer until depth is exhausted — and that CoT is a wasteful workaround. **Prior-art check revealed HLA already implements the core primitive:** `ReconstructionState::evolve_hla()` is a gated additive recurrent update of the 8-dim per-NPC latent state — exactly Family C (delta-rule SSM) of the proposed `MicroRecurrentBeliefState`. The honest delta is narrower: (a) attractor dynamics (Family A) as a quality variant over HLA's leaky integrator — may reduce long-horizon flip-flops, GOAT-gated on a coherence benchmark; (b) kernel-as-versioned-snapshot for a new per-NPC divergence axis; (c) taxonomy unification of `evolve_hla` into a trait. **Verdict: GOAT (paper alone) + GOAT (fusion, revised down from initial Super-GOAT claim after the `evolve_hla` prior-art check).** None of the remaining delta is a new capability class — HLA already does per-NPC recurrent latent state tracking.
