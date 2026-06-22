# Research 281: Per-Tick Salience Tri-Gate — Speak / Silent / Delegate (Modelless)

> **Source:** [JoyAI-VL-Interaction: Real-Time Vision-Language Interaction Intelligence](https://arxiv.org/abs/2606.14777) — Yao et al. (JD.com), Jun 2026
> **Date:** 2026-06-22
> **Status:** Active — Super-GOAT verdict; primitive scheduled (Plan 303), private guide at `riir-ai/.research/148_*`.
> **Related Research (katgpt-rs):** 247 (CS-KV probe / R133 open half), 241 (SwiR explicit↔latent switch / R134 open half), 278 (Engram hash memory / R147 open half), 269 (variable-width shape adapter), 276 (MicroRecurrentBeliefState / HLA leaky integrator)
> **Related Research (riir-ai):** 133 (NPC Mind-Reading adaptive bandwidth), 134 (SwiR two-brain bridge), 147 (Engram conditional memory)
> **Related Plans:** katgpt-rs 303 (this primitive), riir-ai 330 (runtime integration), riir-ai 311 (mind-reading runtime), riir-ai 313 (SwiR validation), katgpt-rs 299 (Engram primitive)
> **Classification:** Public (open primitive). Private selling-point guide lives in `riir-ai/.research/148_Per_Tick_Emit_Salience_NPC_Guide.md`.
> **Verdict: Super-GOAT — the per-tick autonomous emit decision is a new capability class with zero prior art in the corpus.**

---

## TL;DR

JoyAI-VL-Interaction's headline contribution is deceptively simple: **make "when to act" a learned per-second decision of the model itself, with `silence` treated as a first-class output token alongside `speak` and `delegate`**. We strip the training recipe (GRPO + role-weighted SFT → riir-train) and keep the runtime pattern: a **3-way sigmoid salience gate** that decides every tick whether to emit, stay silent, or hand off to a background reasoner. This is the missing primitive our NPC stack needs — every existing emit path (`npc/dialog.rs`, `npc_comms/`, combat shouts) is **user-prompted or event-triggered**, never autonomously decided by the NPC's own latent state.

**Distilled for katgpt-rs (modelless, inference-time):**
A fixed-size `SalienceTriGate<A, D>` struct that maps an activation vector `a ∈ R^D` + two scalar context signals (zone-attention `z`, curiosity `c`) to one of three decisions `{Speak, Silent, Delegate}`. Decisions are produced by **two stacked sigmoids** (never softmax — per AGENTS.md), with `Silent` as a **first-class variant**, not "below threshold = skip". The delegate variant returns a `DelegateToken` (a typed handoff) rather than emitting immediately; the caller wires the token to any async backend (AnyRAG gateway, cold-tier lookup, external model).

This fuses four existing pillars into a capability none of them has alone: **R133 (mind-reading bandwidth) × R134 (SwiR two-brain entropy commit) × R147 (Engram conditional memory) × HLA per-NPC belief state**. The selling point — "thousands of NPCs each autonomously decide when to speak, what to say, and when to delegate, no prompt polling, 20Hz tick" — is documented in the private guide (`riir-ai/.research/148`).

---

## 1. Paper Core Findings (verified by full read)

### 1.1 The paradigm shift — interaction model vs turn-based model

The paper's central claim (§1, §2.2): today's LLMs are **turn-based by construction** — they "open their eyes only at the moment they are prompted." Even "real-time omni" models (GPT-Realtime-2, Qwen3.5-Omni) and consumer video-call products (Doubao, Gemini) optimize *conversational turn-taking*, not autonomous emit decisions. Doubao polls every few seconds via an external trigger (`ExternalTextToLLM`); Gemini doesn't even poll. Neither places the *when to respond* decision inside the model.

JoyAI-VL-Interaction instead makes the decision internal and per-second:

> "make when to act a learned, per-second decision of the model itself, with staying silent treated as a first-class action alongside speaking and delegating"

This is the **paradigm gap** the paper exploits to beat Doubao 77.6% and Gemini 87.9% in head-to-head human evaluation (§5.1). On the most time-critical scenarios (monitoring/alerting, real-time translation/counting) it wins 100% of comparisons — exactly the "be present, act at the right moment" axis that turn-based systems structurally cannot deliver.

### 1.2 The three actions and the silence-first insight (§3.2)

At every 1-second step the model emits one of three control tokens:

| Token | Meaning | Frequency in training data |
|---|---|---|
| `</silence>` | Stay silent, keep watching | **Dominant** (vastly outnumbers others) |
| `</response>` | Emit a textual reply | Rare onset, then continuation |
| `</delegation>` (inside `</response>`) | Hand a subtask to the background brain | Rarest |

**Critical insight — silence is a first-class label, not the absence of one:**
> "Silence is a first-class label rather than the absence of one, since most steps in any stream should be silent and the model must learn to wait instead of speaking at every step."

This is the part that direct-maps to a *gate*, not a *threshold*. A threshold-suppression scheme (score below τ → skip) treats silence as "no decision"; JoyAI-VL treats silence as "an active decision to keep watching". The distinction matters because it lets the model learn *when* staying silent is itself the correct behavior (e.g., an NPC observing a thief should stay silent until reinforcement arrives, not narrate continuously).

### 1.3 The delegation two-loop choreography (§3.2, §4.1)

Delegation is **not** "give up and route to a bigger model". It's a structured two-loop pattern:

1. Model decides to delegate → emits a **brief holding reply** to the user ("let me look into that").
2. Model emits a **hidden delegate token + text query** that the user never sees.
3. System dispatches the query to a background brain (any API/model/agent via a fixed text contract).
4. **Random delay simulates variable background reasoning time** — the model must stay present and keep watching the stream while delegation is pending.
5. When the result returns, model produces a formal answer folded into the ongoing context.

The delay is the whole point: it forces the model to learn "I have an outstanding delegation; I should still respond to new turns, hold silence when nothing is worth saying, and not freeze."

**This maps cleanly onto our existing primitives:**
- Holding reply → `Speak` action with a short template
- Hidden delegate token + query → our existing `DelegateToken` pattern (typed handoff to AnyRagGateway)
- Pending state → a per-NPC `pending_delegations: SmallVec<[DelegateToken; 2]>` slot
- Result fold-back → existing `npc_memory.rs` `MemoryUpdate` path

### 1.4 The role-weighted loss (§3.3) — what we keep vs redirect

The training objective is a role-weighted cross-entropy:

```
L(θ) = -(1/|A|) · Σ_{j∈A} w_j · log p_θ(y_j | y_<j)
```

with `w_first_silence = 1.0`, `w_repeated_silence = 0.4`, `w_response = 1.5`. This addresses the silence/response imbalance. **This is training know-how → riir-train.** What we keep is the *runtime consequence*: the gate must be biased toward silence unless salience is genuinely high. We bake this into the gate's `β_speak` parameter (sigmoid inverse temperature), not into a loss.

### 1.5 AdaCodec streaming codec (§3.1) — orthogonal, noted

The paper uses AdaCodec (predictive visual coding — full ViT tokens only at scene changes, ~16 P-tokens on predictable frames) to keep the per-second token budget bounded over hours of video. **This is orthogonal to the salience gate.** Our equivalent (predictive coding for game state) is the existing `frame_coreset.rs` + spectral_threat.rs path; AdaCodec itself is a separate paper (arxiv 2606.02569) and would be its own research note if pursued. Not blocking.

### 1.6 Three-tier long-horizon memory (§4.3) — already covered

Short (raw tokens) / mid (text summaries) / long (aggressively compressed blocks). This is **structurally identical** to our existing Four-Tier Memory Architecture (`riir-ai/.research/007`) and the Dual-Pool CGSP router (`riir-ai/.research/249` + Plan 312). No new primitive needed; the tri-gate just *consumes* memory state, it doesn't replace it.

---

## 2. Distillation — the modelless primitive

### 2.1 The tri-gate (math)

Given:
- An activation vector `a ∈ R^D` (HLA state, functor output, Engram-retrieved shard — any latent)
- A scalar **zone-attention** signal `z ∈ [0,1]` (from R133's `ca(E,R)` or zone density gating — "how relevant is here to me right now")
- A scalar **curiosity** signal `c ∈ [0,1]` (from `cgsp_runtime` derivative curiosity or ICT branching detector — "how surprising is now")

Compute two **stacked sigmoids** (never softmax — per AGENTS.md hard constraint):

```
salience       = dot(a, d_speak)         + w_z · z + w_c · c   // scalar
score_speak    = sigmoid(β_speak    · (salience - τ_speak))
score_delegate = sigmoid(β_delegate · (dot(a, d_delegate) - τ_delegate))
```

Decision rule (zero-allocation, branch-light):

```rust
pub enum SalienceDecision<A> {
    Silent,                         // first-class: NPC keeps watching
    Speak,                          // emit primary reply this tick
    Delegate(A),                    // emit holding reply + typed handoff
}

if score_speak < FLOOR_SPEAK {
    SalienceDecision::Silent
} else if score_delegate > CEIL_DELEGATE {
    SalienceDecision::Delegate(delegate_payload)
} else {
    SalienceDecision::Speak
}
```

Three design constraints:

1. **Silent is a variant, not a default.** This is the paper's core insight. The gate *emits* Silent as a decision; the caller's tick loop records it as a decision, not as "nothing happened". This matters for downstream learning (bandit arms over the tri-gate) and for replay (a Silent tick is a tick where the NPC actively chose to wait).
2. **Two sigmoids, not one softmax over 3 classes.** Per AGENTS.md: never softmax. Two sigmoids also let us reason about each axis independently for ablation (G2 in the validation protocol).
3. **`d_speak` and `d_delegate` are pre-computed direction vectors**, BLAKE3-committed at freeze/thaw into the `ZoneExpertBundle`. They're task-family-specific (a guard's `d_speak` differs from a merchant's), updated only by snapshot swap — never gradient-touched at runtime. This is the **freeze/thaw over fine-tuning** rule (constraint 3).

### 2.2 Silence-as-token at the type level

The paper treats `</silence>` as a token the model emits. Our modelless equivalent: a `SilenceToken` newtype that flows through the same `npc_comms::bus` channel as `Speak` and `Delegate` events. Subscribers (other NPCs via R133 mind-reading, the KG triple emitter, the civ inheritance log) see *that the NPC chose silence* and can react to it ("the guard saw the thief but stayed silent — that's suspicious"). This is **non-trivial**: it makes silence observable, hence learnable, hence part of the social semantics. Without this, "did nothing" and "decided to do nothing" are indistinguishable in the event log.

### 2.3 Async delegation — the runtime contract

The `Delegate(A)` variant does not block the tick. It returns a `DelegateToken`:

```rust
pub struct DelegateToken<A> {
    pub payload: A,                      // typed handoff (quest id, search query, etc.)
    pub issued_tick: u64,                // for delay tracking
    pub holding_reply: SmallVec<[u8; 64]>, // short reply emitted this same tick
    pub foldback_target: FoldbackTarget, // where the result lands
}

pub enum FoldbackTarget {
    HlaState,           // result becomes a new HLA direction vector (R134 commit)
    EngramLookup,       // result is a hash-addressed pattern (R147)
    AnyRagEscalation,   // result routes through AnyRagGateway (neuron-db)
    ColdTierRetrieval,  // result is a frozen shard (riir-neuron-db)
}
```

The caller's tick loop:
1. Emits `holding_reply` synchronously (if `Speak` or `Delegate`).
2. For `Delegate`, spawns an async task keyed by `DelegateToken`. NPC's `pending_delegations` slot gains an entry.
3. On completion, the result is folded back via `foldback_target`. This may itself trigger a *future* `Speak` — handled by the next tick's gate, not by a callback.

The "stay present while delegation pending" behavior from §3.2 is enforced by the gate's state: when `pending_delegations` is non-empty, the `Silent` variant is *not* recorded as Suspended — the NPC continues to make per-tick decisions, just with the pending delegation as additional context.

### 2.4 Latent reframing (this is a latent-to-latent op)

Per the skill's mandatory latent-space reframing: **the entire tri-gate operates in latent space**. The activation `a` is HLA / functor / shard-latent. The direction vectors `d_speak`, `d_delegate` are latent. The zone-attention `z` and curiosity `c` scalars are bridge-projected from raw (`MapPos`, tick count) but enter the gate as scalars, not embeddings. The decision boundary is a hyperplane in latent space.

**Crucially, the gate output is also latent-shaped**: `Speak` is decoded to a token/KG-triple/etc. by downstream bridges (`ActionBridge` for raw actions, `kg_gate` for KG triples), but the gate itself stays in latent land. This satisfies the latent-to-latent preference (constraint 2) and keeps the gate cheap (no decode inside the hot path).

### 2.5 What stays open vs private

| Component | Repo | License |
|---|---|---|
| `SalienceTriGate<A, D>` struct + math | katgpt-rs | MIT |
| `SalienceDecision<A>` enum + `SilenceToken` newtype | katgpt-rs | MIT |
| `DelegateToken<A>` + `FoldbackTarget` enum | katgpt-rs | MIT |
| Two-sigmoid decision rule (β_speak, β_delegate, τ_speak, τ_delegate) | katgpt-rs | MIT |
| **NPC tick loop wiring** (gate → bus → memory) | riir-ai | Private |
| **Per-task-family direction-vector recipes** (guard d_speak vs merchant d_speak) | riir-ai | Private |
| **Zone-attention + curiosity scalar feeds** (R133 `ca`, cgsp derivative) | riir-ai | Private |
| **Delegate path → AnyRagGateway / Engram / Cold-tier routing** | riir-ai + riir-neuron-db | Private |
| **Training recipe (GRPO + role-weighted SFT)** | riir-train | Private (out of scope here) |

---

## 3. Fusion — what novel combination does this produce?

Per the skill's fusion protocol, we cross-checked the four fusion cousins in this session:

### 3.1 R133 (NPC Mind-Reading, `riir-ai/.research/133`)
- **R133 axis:** *What* to share between NPCs (bandwidth allocation via `K(ca)` interpolation).
- **JoyAI-VL axis:** *When* to emit at all (per-tick salience gate).
- **Fusion gain:** R133's `ca(E,R)` scalar (context-awareness) becomes the **zone-attention input `z`** to the tri-gate. A guard with high `ca` (full sensor overlap with emitter) doesn't need to *re-broadcast* — `z` is high, but the *speak* salience against "I already know what E knows" is low → Silent. A guard around the corner has low `ca` but high *need-to-know* → Speak. The tri-gate thus **converts R133's bandwidth axis into an emit/silent decision**, which is the higher-level question.

### 3.2 R134 (SwiR Think↔Info Brain, `riir-ai/.research/134`)
- **R134 axis:** *When* the think-brain commits to the info-brain (entropy-triggered Latent→Explicit switch).
- **JoyAI-VL axis:** *When* the NPC emits at all.
- **Fusion gain:** R134's entropy signal is the **curiosity input `c`** to the tri-gate. Low entropy + high salience → Speak (the NPC has converged, commit). High entropy → Silent (keep thinking). The tri-gate **subsumes R134's commit signal as one input among several**, which is more general than R134 alone (R134 only knows entropy; the tri-gate also knows zone attention and HLA projection). The delegate variant *is* the R134 commit — the background brain is the "think brain" running asynchronously.

### 3.3 R147 (Engram Conditional Memory, `riir-ai/.research/147`)
- **R147 axis:** *What* to remember (hash-addressed conditional pattern memory).
- **JoyAI-VL axis:** *When* to delegate.
- **Fusion gain:** R147's hash lookup is the natural `Delegate` foldback target. When `dot(a, d_delegate)` is high, the NPC doesn't just "ask a bigger model" — it queries its **own conditional memory** (R147) for a matching pattern. This makes delegation *local-first* (fast, in-zone, no network) before escalating to AnyRagGateway. The tri-gate thus gives R147 a **proactive trigger** (it currently only fires on reactive lookup); R147 gives the tri-gate a **cheap delegate path** (it currently only knows "ask external model").

### 3.4 HLA per-NPC belief state (`riir-ai/crates/riir-engine/src/hla/`)
- **HLA axis:** per-NPC 8-dim latent state (valence/arousal/desperation/calm/fear + 3).
- **JoyAI-VL axis:** the substrate `a` for the salience projection.
- **Fusion gain:** HLA finally has a *runtime decision consumer*. Currently HLA feeds the action bridge (`bridge/mod.rs`) for combat/movement actions; the tri-gate adds the **emit/silent/delegate** output channel. This means HLA's emotional state directly drives *whether the NPC speaks*, not just *what action it takes*. A fearful NPC (high `fear`) with low salience stays silent; a calm NPC (high `calm`) with high salience speaks. This is the **emergent social behavior** the selling point promises.

### 3.5 The novel combination

| Capability | R133 | R134 | R147 | HLA | JoyAI-VL alone | **Fusion** |
|---|---|---|---|---|---|---|
| Autonomous emit decision | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Bandwidth-adaptive comms | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Entropy-driven commit | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ (via curiosity input) |
| Local-first delegation | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ (via Engram foldback) |
| Emotion-driven speak/silent | ✗ | ✗ | ✗ | ✓ (actions only) | ✗ | ✓ |
| Per-tick at 20Hz, 1000s of NPCs | ✗ | ✗ | ✗ | ✓ | ✗ (single 8B model, 1Hz) | ✓ |

**The bottom line:** JoyAI-VL proves the per-second emit decision is a *capability worth scaling as its own thing*. None of our existing pillars can produce it alone. The fusion produces it at MMO scale (20Hz, 1000s of NPCs) using only modelless primitives.

---

## 4. Novelty gate — all 4 YES

### Q1: No prior art? — **YES**

Vocabulary translation (paper terms → codebase terms) and grep results from this session:

| Paper term | Codebase candidates grepped | Hits |
|---|---|---|
| `silence` / `stay_silent` / `silent_token` | `nop_emit`, `speak_gate`, `emission_gate`, `silence` | Only FFT `Silence` combat debuff + Parakeet `prime_with_silence` (warm-up). **Zero emit-gate hits.** |
| `speak` / `response` / `delegate` | `npc_comms`, `dialogue_trigger`, `utterance`, `delegate`, `background_brain` | `npc/dialog.rs` exists but is user-prompted (turn-based). `npc_comms/` exists but is bandwidth allocation (R133). `AnyRagGateway` is a delegate *target*, not a *decision*. |
| `interaction_model` / `per_second_decision` | `tick_gate`, `salience_router`, `proactive_response`, `unsolicited` | **Zero hits.** |
| `salience` | `ega`, `spectral_salience`, `attention_sink`, `spectral_threat` | EGA (Plan 139) is closest — spectral salience for *attention*, not for *emit decisions*. Different output modality. |
| `delegate` / `background_loop` / `async_dispatch` | `async_dispatch`, `two_loop`, `background_bridge` | Existing delegate uses (`DualPool`, `DerivativeCuriosity`, `AnyRagGateway`) are all *internal* delegation (delegate-to-inner-sampler, delegate-to-external-judge). None of them is a *per-tick decision to delegate vs speak vs silent*. |

**Three-layer check passed:** notes (`.research/` + `.plans/`), code (`src/` + `crates/`), and vocabulary alternatives. No prior art in any layer.

### Q2: New class of behavior? — **YES**

Current NPC emit paths (`npc/dialog.rs:DialogEngine::respond`, `npc_comms::bus::publish`, combat shouts via `ActionBridge`) all require an external trigger: a player input, a sensor event, a combat tick. **None of them autonomously decides "this tick, I will speak; that tick, I will stay silent".** The tri-gate produces a new behavior class: *proactive unsolicited emit decisions from latent state*. Incumbent systems cannot do this without an external polling clock — which is exactly the paradigm gap JoyAI-VL exploits to beat Doubao/Gemini.

### Q3: Product selling point? — **YES**

> "Thousands of NPCs each autonomously decide when to speak, what to say, and when to delegate to a background reasoner — no prompt polling, 20Hz tick. Guards shout only when they have something worth shouting; merchants stay quiet unless haggled; companions speak up unprompted when the player looks tired."

This is a selling point no current MMORPG ships. The closest analog (Skyrim's "radiant dialogue") is rule-based and scripted, not learned-from-latent-state.

### Q4: Force multiplier (≥2 pillars)? — **YES (7 pillars)**

| Pillar | Connection |
|---|---|
| HLA belief state | Substrate (`a`) |
| R133 Mind-Reading | `ca` scalar → `z` input |
| R134 SwiR two-brain | Entropy → `c` input; `Delegate` = async think-brain |
| R147 Engram | `Delegate` foldback target (local-first) |
| AnyRagGateway (neuron-db) | `Delegate` escalation target (external) |
| Fog-of-war / spatial cognition | Drives `z` (sensor overlap) |
| LatCal commitment (chain) | `Speak` of a KG triple → chain commit; `Silent` → no commit (lower gas) |

**All 4 YES → Super-GOAT.** Per the skill's anti-deferral rule, the mandatory outputs are created in this same session (see §6).

---

## 5. Verdict

**Super-GOAT.** The per-tick salience tri-gate is a new capability class (autonomous emit decisions from latent state), with zero prior art in our corpus, a clear product selling point, and a 7-pillar force multiplier. The paper's training recipe (GRPO + role-weighted SFT) redirects to riir-train; the runtime pattern (tri-gate + silence-as-token + async delegate) distills cleanly into katgpt-rs as a generic modelless primitive, with the NPC-specific selling-point guide in riir-ai.

**One-line reasoning:** JoyAI-VL proves "interactivity is worth scaling as a capability of the model itself" — we extract the modelless core (3-way sigmoid gate, silence as first-class, async delegate contract) and fuse it with R133/R134/R147/HLA to ship proactive unsolicited NPC emit decisions at 20Hz MMO scale, which no incumbent (including the paper itself, which runs a single 8B model at 1Hz) can deliver.

---

## 6. Mandatory outputs (created in this session per anti-deferral rule)

| Artifact | Path | Status |
|---|---|---|
| Open primitive (this note) | `katgpt-rs/.research/281_*.md` | ✅ This file |
| Open primitive plan | `katgpt-rs/.plans/303_salience_tri_gate_primitive.md` | ✅ Created |
| Private selling-point guide | `riir-ai/.research/148_Per_Tick_Emit_Salience_NPC_Guide.md` | ✅ Created |
| Runtime integration plan | `riir-ai/.plans/330_proactive_npc_salience_gate_runtime.md` | ✅ Created |
| Training recipe | `riir-train/.research/` (GRPO + role-weighted SFT) | Redirect (out of scope) |

---

## TL;DR

JoyAI-VL-Interaction's distilled primitive is a **per-tick 3-way salience gate** (speak / silent / delegate) with **silence as a first-class output variant** and an **async delegation contract** that keeps the agent present while a background brain works. The training (GRPO + role-weighted SFT) goes to riir-train; the runtime pattern lands in katgpt-rs as `SalienceTriGate<A, D>` — two stacked sigmoids over a latent activation `a` + zone-attention `z` + curiosity `c`, with `d_speak` and `d_delegate` as BLAKE3-committed direction vectors. **Novelty gate: all 4 YES** (no prior art across notes+code+vocabulary alternatives; new capability class — autonomous unsolicited emit; clear selling point; 7-pillar force multiplier) → **Super-GOAT**. Mandatory outputs in this session: this note + `katgpt-rs/.plans/303` (open primitive) + `riir-ai/.research/148` (private guide) + `riir-ai/.plans/330` (runtime). Fusion: R133 (`ca` → `z`) × R134 (entropy → `c`) × R147 (Engram → `Delegate` foldback) × HLA (`a` substrate) = the missing "when does this NPC speak?" primitive. Selling point: "thousands of NPCs each autonomously decide when to speak, what to say, and when to delegate, no prompt polling, 20Hz tick."
