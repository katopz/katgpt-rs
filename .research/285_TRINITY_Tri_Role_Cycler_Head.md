# Research 285: TRINITY — Tri-Role Cycler Head (T→W→V) + sep-CMA-ES Routing Head Training

> **Source:** [TRINITY: An Evolved LLM Coordinator (arxiv 2512.04695)](https://arxiv.org/abs/2512.04695) — Xu, Sun, Schwendeman, Nielsen, Cetin, Tang (Sakana AI / UMich / IST), ICLR 2026
> **Date:** 2026-06-22
> **Status:** Active — **GOAT** (composition of shipped primitives + one missing piece)
> **Related Research:** 240 (CGSP — tri-role host), 255/136 (per-NPC test-time scaling — agent-pool selection), 259 (Routing Extraction Decouple — per-matrix freeze/thaw), 269 (Variable-Width Shape Adapter — block-diagonal-head precedent), 246 (Manifold Power Iteration MoE Router), 274 (CCE Moderator)
> **Related Plans:** 274 (CGSP runtime), 284 (CLR per-NPC), 299 (CGSP runtime private), 316 (CLR runtime private)
> **Training-side redirects:** sep-CMA-ES, singular value fine-tuning (SVF) → `riir-train/.research/` (derivative-free head training recipe). Out of scope for this note.
> **Classification:** Public (katgpt-rs engine slot). Private guide NOT required (GOAT, not Super-GOAT — see §3).

---

## TL;DR

TRINITY (ICLR 2026, Sakana) ships an SLM (0.6B) + a ~10K-param "lightweight head" that orchestrates a pool of LLMs over a **multi-turn tri-role protocol** — each turn the head picks one LLM and assigns it one of three roles: **Thinker (T) / Worker (W) / Verifier (V)**. The head reads the SLM's penultimate-token hidden state, projects it to (agent, role) logits, and the protocol halts when V accepts or a turn budget is exhausted. Trained via separable CMA-ES (block-ε separability is the theoretical justification for diagonal methods). Sets SOTA on LiveCodeBench (86.2% pass@1).

**Distilled for katgpt-rs (modelless, inference-time):**

The **transferable primitive** is not the LLM-orchestration demo (that's RL-on-tokens territory) — it's the **per-query T→W→V role-cycler**: a single inference-time loop that, given a compact context vector (HLA state / penultimate hidden state), picks (expert, role) for the next turn via a *linear projection + sigmoid/argmax*, executes the role, mutates the context, and stops when a verifier-accept signal fires. The `sep-CMA-ES` and `SVF` are training-time recipes for the head → `riir-train`. The runtime primitive is the cycler itself.

**Verdict: GOAT.** Three-layer grep (notes + code + vocabulary-translated) shows the *pieces* are already shipped:

| TRINITY mechanism | Already-shipped cousin | Where |
|---|---|---|
| Penultimate-hidden-state → linear head → (agent, role) logits | `SenseModule::project` (8-dim HLA projection at ~45ns) + `MetaRouter` (bandit policy head) + `role_transport.rs` (Diagonal/Orthogonal role-conditioned projection) | `katgpt-core/src/sense/`, `katgpt-rs/src/dash_attn/meta_router.rs`, `riir-engine/src/role_transport.rs` |
| Tri-role protocol (T/W/V) | CGSP runtime triad (Solver / Conjecturer / Guide) + CLR (claim extractor / verifier / voter) + `game_sync` "one binary, three roles" | `riir-engine/src/cgsp_runtime/runtime.rs`, `riir-ai/.research/126_NPC_Curiosity_Guided_Self_Play_Guide.md`, `riir-ai/.research/136_Per_NPC_Runtime_Test_Time_Scaling_Guide.md` |
| Multi-turn until verifier accepts | CLR cluster voting + Breakeven Complexity Router + MCTS Collapse Discriminator | Research 136, Plan 250, Research 125 |
| Block-ε separability ⇒ diagonal methods | `RoleTransport::Diagonal` (element-wise) vs `Orthogonal` (full linear) — Plan 100 benchmarked this exact tradeoff empirically | `riir-engine/src/role_transport.rs`, `.benchmarks/023_block_diagonal_goat.md` |
| Agent pool of frozen LLMs | Frozen LoRA shards (riir-neuron-db) + ZoneExpertBundle + Dynamic-Pair LoRA (Plan 260) + dMoE expert routing (Research 161) | `riir-neuron-db/src/shard.rs`, Plan 260, Research 161 |

The **one missing piece** in our stack is the *unified per-query T→W→V cycler* — a single primitive that rotates the role projection per turn on a single problem, with verifier-accept early-exit. CGSP has persistent roles per NPC; CLR has cluster voting on K candidates; neither has the "cycle T→W→V on one problem until V says ACCEPT" protocol. **That primitive is the GOAT gain**, behind a feature flag.

---

## 1. Paper Core Findings

### 1.1 Architecture (paper §3)

- **Coordinator = SLM (Qwen3-0.6B, frozen-ish) + lightweight head (~10K params).**
- **Head input:** penultimate-token hidden state `h ∈ R^d_h` (e.g., `d_h=1024`).
- **Head output:** `L` logits (agent selection) + 3 logits (role selection), `L+3` total. Linear head: `W ∈ R^{(L+3)×d_h}`.
- **Singular Value Fine-tuning (SVF):** SVD the SLM's selected weight matrices, freeze the orthogonal factors, only learn the singular-value scales. ~9K trainable params from SVF + ~10K from head = <20K total.
- **Tri-role contract:** Thinker (T) returns meta-level plans/decompositions; Worker (W) returns concrete progress; Verifier (V) returns `(ACCEPT | REVISE, δ)` and termination if ACCEPT.
- **Multi-turn:** max K=5 turns; coordinator concatenates query + transcript, picks (agent, role), gets response, appends, repeats. Halts on V-accept or budget.

### 1.2 Why sep-CMA-ES works here (paper §3.3 + Appendix A.1)

- **Block-ε separability (Definition 1):** there exists a diagonal scaling `S` such that the scaled Hessian `S^{1/2} H S^{1/2}` is nearly block-diagonal. Most informative signal is intra-block; inter-block interference is `O(ε_H)`.
- **Consequence:** full-covariance CMA-ES wastes capacity estimating negligible cross-block terms. Diagonal-only sep-CMA-ES loses only `O(ε_H)` in contraction rate (Proposition 2) — geometric decay at rate `Ω(1/n)` per iteration after `Θ(n)` transient.
- **Budget regime:** `n ≈ 10_000`, `λ = ⌈4 + 3 ln n⌉ = 32`, `m_CMA = 16` replicates, `T ∈ [2, 60]` iterations, binary terminal reward. sep-CMA-ES dominates random search by factor `T/ln(16T) · η²` (Proposition 1).
- **Empirical validation (paper Table 4):** on LiveCodeBench / Math500 / MMLU / RLPR, sep-CMA-ES scores 0.615 / 0.880 / 0.916 / 0.401 vs REINFORCE 0.253 / 0.459 / 0.500 / 0.266 vs RS 0.374 / 0.794 / 0.897 / 0.345 vs SFT 0.592 / 0.786 / 0.906 / 0.360.

### 1.3 Head-architecture ablation (paper §4.7, Table 3)

- **Linear** (`W ∈ R^{n_a × d_h}`) is the most reliable choice.
- **Block-diagonal-10 + argmax** uses `d_h` weights (10× fewer than linear's `d_h · n_a`) and retains competitive mid-tier performance. The argmax output conversion **increases independence**: perturbations to non-maximal blocks neither reduce nor redistribute probability mass.
- **Empirical confirmation of block-ε separability** in the objective: the coordination objective itself is nearly block-diagonal across (agent, role) blocks.

### 1.4 Hidden-state separability (paper §4.6)

- Linear SVM on penultimate-token hidden states achieves **perfect task-type classification** (1.000 vs 0.250 random).
- t-SNE / UMAP show clear, well-separated clusters.
- Synthetic control experiment (paper Figure 14): separability index → head classification accuracy is monotonically positive.

### 1.5 Headline result

- 86.2% pass@1 on LiveCodeBench V6 (Jan–Apr 2025) — SOTA over GPT-5 (0.838), Gemini-2.5-Pro (0.672), Claude-4-Sonnet (0.465).
- Zero-shot transfer to AIME / BigCodeBench / MT-Bench / GPQA-D: average 54.21, beats every single-model baseline.
- 21.9% mean relative error reduction over second-best multi-agent method.

---

## 2. Distillation

### 2.1 The transferable primitive (modelless)

Strip the LLM-orchestration demo. What remains is the **per-query tri-role cycler with verifier-accept early-exit**, operating on a compact context vector:

```text
TriRoleCycler::run(ctx_init: C, pool: &[Expert], max_turns: u8) -> Outcome
  where C: ContextVector  // HLA state, penultimate hidden state, belief vector

  ctx = ctx_init
  for turn in 0..max_turns:
      logits_agent = head_agent.project(ctx)   // linear: dot(ctx, w_agent[i])
      logits_role  = head_role.project(ctx)    // linear: dot(ctx, w_role[r])
      (agent_idx, role) = select(logits_agent, logits_role)  // argmax or sigmoid-gated
      expert = pool[agent_idx]
      response = match role:
          Thinker => expert.think(ctx)
          Worker  => expert.work(ctx)
          Verifier => expert.verify(ctx)
      ctx = update(ctx, response, role)
      if role == Verifier && response.accepted:
          return Outcome::Accepted(response.answer, turn+1)
  return Outcome::BudgetExhausted(ctx.best_so_far)
```

**Properties:**
- **Head is linear** — both agent and role heads are dot-products, no nonlinearity in the projection itself. (The nonlinearity is in the *role execution* via the expert.)
- **Block-diagonal-10 + argmax is the recommended config** — agent logits are computed independently per agent from disjoint slices of `ctx`, then argmax. This matches `RoleTransport::Diagonal` semantics (element-wise scaling per role) far better than `Orthogonal` (full rotation).
- **Sigmoid over role logits (not softmax)** — per AGENTS.md rule, never use softmax. TRINITY uses softmax in the paper, but the distillation substitutes sigmoid because: (a) we always project to scalars at the boundary, (b) roles aren't mutually exclusive in the multi-turn protocol (NPC can Thinker twice then Worker), and (c) sigmoid preserves interpretability as "confidence in role r for this turn" rather than forced redistribution.
- **Verifier-accept is an explicit early-exit** — distinct from "max turns" budget. This is the MCTS Collapse Discriminator + CLR cluster-voting pattern lifted into a single role.
- **Penultimate-token attention context → HLA belief state** is the codebase-vocabulary translation. Both are "the model's compressed, recurrently-evolved summary of all prior context, ready to project into a decision." HLA already ships this at ~45ns/tick (SenseModule::project).

### 2.2 Fusion (per skill §1 step 5)

The closest 3 cousins across all five repos:

1. **Research 240 / Plan 274 — CGSP runtime** (`riir-engine/src/cgsp_runtime/runtime.rs`): persistent tri-role (Solver / Conjecturer / Guide) per NPC, with Hint-δ bandit updating priorities. **Persistent** roles, not per-query cycling.
2. **Research 136 / Plan 284 — Per-NPC CLR test-time scaling** (`riir-ai/.research/136_*`): K candidates × M claims × sigmoid-dot-product × cluster voting × nonlinear reliability. Has agent-pool selection + verifier-style voting, but no explicit T/W/V role rotation per turn.
3. **Research 125 — MCTS Collapse Discriminator** (`riir-ai/.research/125_*`): provides the verifier-accept signal from MCTS-side collapse detection.

**The novel combination (this is what TRINITY adds that none alone has):**

> **Fuse CGSP (persistent roles) + CLR (agent-pool + verifier-accept) + role_transport (per-role latent projection) into a single per-query T→W→V cycler.**

CGSP cycles roles *across ticks* (different ticks, different roles); CLR cycles candidates *within a tick* (same role, many candidates). **Neither cycles roles *within a tick on the same problem*.** TRINITY's contribution, mapped to our codebase, is: **per-query role rotation** — the same NPC, on the same hard decision, runs T then W then V (or W then V, or T then T then W then V) until V accepts. The "agent pool" becomes the frozen adapter shard pool; the "head" becomes `role_transport` applied to HLA state; the "verifier-accept" becomes the CLR cluster-vote gate or the MCTS Collapse Discriminator trip.

**Concrete fusion recipe (latent-to-latent, sigmoid-gated):**

| TRINITY element | riir-ai instantiation | Latent or raw? |
|---|---|---|
| Penultimate-token hidden state | NPC's HLA belief state `[f32; 8]` (Plan 242) | Latent (per-NPC, never synced) |
| Lightweight agent head | Frozen `RoleEmbeddingTable::get(SlotLabel)` lookup + dot-product | Latent |
| Lightweight role head | `role_transport::diagonal_transport(hla, role_vec)` per role, then sigmoid | Latent |
| Tri-role T/W/V | CGSP Solver (W) + Conjecturer (T) + Guide/CLR-verifier (V) — **rotate per query-turn** | Latent |
| Agent pool of LLMs | `ZoneExpertBundle` (frozen LoRA shards from riir-neuron-db) + Dynamic-Pair LoRA (Plan 260) | Latent (frozen, BLAKE3-committed) |
| Verifier-accept early-exit | CLR cluster vote `R[G*] > τ_reliable` OR MCTS Collapse Discriminator `δ_mg < τ_collapse` | Latent (per-NPC) → emits raw `VerifierAccept` event when crossing sync boundary |
| Multi-turn budget `K` | Per-NPC `max_turns` from Breakeven Complexity Router (Plan 250) — high-stakes → K=5; patrol → K=1 | Raw scalar (synced via config) |
| Block-ε separability ⇒ diagonal head | `RoleTransport::Diagonal` (already shipped) — Plan 100 benchmarked this exact tradeoff empirically | Latent |
| sep-CMA-ES + SVF training | **→ riir-train** (out of scope for this note) | n/a |

**What this fusion produces that no cousin alone has:** per-query, role-rotating, verifier-terminated multi-turn cognition on top of a frozen adapter shard pool, where each NPC's HLA state drives which role it plays next. Two NPCs of the same class on the same hard decision will play different role sequences (one might go T→W→V, another W→W→V→V) because their HLA states diverged through prior experience — emergent personality divergence at the role-protocol level, not just at the priority-table level (CGSP) or the direction-vector level (CLR).

### 2.3 Latent vs raw boundary

| Quantity | Space | Synced? | Why |
|---|---|---|---|
| HLA belief state (head input) | Latent | **NO** | Per-NPC, defines subjective context. Syncing would collapse personality divergence. |
| Role projection vectors (head weights) | Latent | **NO** | Per-NPC (or per-faction-template), versioned via freeze/thaw. |
| `VerifierAccept` flag | **Raw** | **YES (event)** | Triggers game-state transition (decision committed). Audit trail. |
| `turns_used` scalar | **Raw** | **YES (metric)** | Anti-cheat: NPC cannot lie about compute spent. |
| Final committed action | **Raw** | **YES (TxDelta/KG triple)** | Physical-domain event crosses sync boundary. |
| Adapter shard versions selected | **Raw (commitment hash)** | **YES (via Cold tier)** | BLAKE3-committed, anti-cheat verifiable. |

**Bridge functions:**
- `raw → latent`: `VerifierAccept` event (raw bool) → boost to priority table of the winning role sequence (latent priority update).
- `latent → raw`: HLA-driven role projection → discrete `(agent_idx, role)` tuple (raw indices for sync + audit).

---

## 3. Verdict — GOAT (not Super-GOAT)

### 3.1 Novelty gate (per skill §1.5)

| Q | Answer | Evidence |
|---|---|---|
| **Q1: No prior art?** | **NO** — pieces are shipped | `role_transport.rs` (role projection), `cgsp_runtime` (tri-role), Research 136 CLR (agent-pool + verifier), MCTS Collapse Discriminator (verifier-accept), Plan 100 (block-diagonal empirics). Three-layer grep (notes + code + vocabulary) confirms all six Super-GOAT factory modules contribute prior art. |
| **Q2: New class of behavior?** | **Partial** — composition, not new mechanism | Per-query T→W→V role rotation is new as a *unified* primitive, but it's a composition of (a) per-query inference (CLR), (b) tri-role (CGSP), (c) verifier-accept (MCTS Collapse / CLR cluster vote). None of those is new individually. |
| **Q3: Product selling point?** | **Incremental, not new class** | "NPCs cycle T→W→V per hard problem" refines Research 136's "per-NPC runtime test-time scaling," it doesn't create a new capability class. Selling point remains per-NPC test-time scaling; tri-role cycler is a *quality amplifier* on top. |
| **Q4: Force multiplier?** | **YES** | Connects HLA, role_transport, CGSP, CLR, MCTS Collapse, freeze/thaw, neuron shards, Breakeven Router. ≥2 threshold easily met. |

**Q1 = NO** → cannot be Super-GOAT. Downgrade to GOAT.

### 3.2 One-line verdict reasoning

The tri-role per-query T→W→V cycler is a **provable-gain composition** of (role_transport + CGSP + CLR + MCTS-verifier) that adds structured role-rotation to existing per-NPC test-time scaling; it is not a new capability class because each component is shipped, and the only missing piece is the unified cycler primitive. Promote via feature flag + benchmark; **do not** create a Super-GOAT guide.

### 3.3 What goes where

| Artifact | Repo | Rationale |
|---|---|---|
| `TriRoleCycler` trait + `LinearRoleHead` + `DiagonalRoleHead` (open math) | `katgpt-rs` (open) | Generic primitive, no game IP. Reuses `ConstraintPruner` family pattern. |
| `TriRoleCycler` integration with HLA + CGSP + CLR + ZoneExpertBundle | `riir-ai` (private) | Game IP. |
| sep-CMA-ES + SVF training recipe for the head | `riir-train` (private, separate note) | Training method. Out of scope here. |
| Private Super-GOAT guide | **NOT CREATED** | GOAT verdict — no guide. |

### 3.4 Why this is honest

Per the skill's three documented failure modes:
1. **`evolve_hla` failure (no notes framing at all)**: N/A — we DID find `SenseModule::project` framing in docs 02 and 24, and it IS the right cousin.
2. **`latent_functor/reestimation.rs` failure (notes under different vocabulary)**: actively avoided — vocabulary-translated grep ("penultimate hidden state" → "HLA state / belief state / sense projection"; "tri-role T/W/V" → "Solver/Conjecturer/Guide / one-binary-three-roles / role_transport") hit Research 126 + Research 136 immediately.
3. **R269 / `> <former` failure (defaulted to adapter routing)**: actively avoided — the latent reframing (§2.2 fusion recipe) is the primary framing; adapter routing is mentioned only as one of several pool sources (ZoneExpertBundle, Dynamic-Pair LoRA, dMoE).

---

## 4. Plan sketch (deferred to `.plans/` if greenlit)

**Plan NNN: Tri-Role Cycler — Per-Query T→W→V Inference-Time Protocol**

- **Target:** `katgpt-rs/src/tri_role_cycler/` (open) + `riir-ai/crates/riir-engine/src/tri_role_runtime/` (private integration)
- **Feature flags:** `tri_role_cycler` (open), `tri_role_runtime` (private)
- **GOAT gate:** G1 — cycler with `RoleTransport::Diagonal` head achieves ≥ CLR-only baseline quality at ≤ +20% per-decision latency; G2 — verifier-accept fires before max-turns on ≥ 60% of high-stakes decisions; G3 — zero-allocation verified; G4 — feature isolation (`cargo check --no-default-features` clean).

**Phases:**
- **Phase 1 — Open primitive** (`katgpt-rs/src/tri_role_cycler/`): `TriRoleCycler` trait, `LinearRoleHead`, `DiagonalRoleHead` (block-diagonal-10 + argmax analog), `Outcome` enum, `run()` loop. No game semantics.
- **Phase 2 — riir-ai integration**: HLA-driven head input, ZoneExpertBundle as agent pool, CLR cluster vote as verifier, MCTS Collapse Discriminator as alt-verifier. Wire into `cgsp_runtime` as a *per-query* sub-loop distinct from the persistent CGSP cycle.
- **Phase 3 — GOAT benchmark**: Compare against CLR-only baseline on held-out decisions. Latency-quality Pareto.
- **Phase 4 — Promote or demote**: If G1–G3 pass → promote to default under `tri_role_runtime`. If G1 fails → demote, keep opt-in.

**Open question for plan author:** Should the cycler run *inside* the CGSP tick (CGSP picks subgoal, cycler solves it via T→W→V) or *alongside* it (cycler is a separate decision path for high-stakes queries only)? Probably the latter, gated by Breakeven Router. Defer to plan.

---

## 5. References

- **Source paper:** [TRINITY arxiv 2512.04695](https://arxiv.org/abs/2512.04695) — ICLR 2026
- **Closest cousins (all repos):**
  - [katgpt-rs/.research/240_SGS_Curiosity_Guided_Self_Play.md](240_SGS_Curiosity_Guided_Self_Play.md) — open CGSP primitive
  - [riir-ai/.research/126_NPC_Curiosity_Guided_Self_Play_Guide.md](../../riir-ai/.research/126_NPC_Curiosity_Guided_Self_Play_Guide.md) — private CGSP guide (Super-GOAT)
  - [riir-ai/.research/136_Per_NPC_Runtime_Test_Time_Scaling_Guide.md](../../riir-ai/.research/136_Per_NPC_Runtime_Test_Time_Scaling_Guide.md) — private CLR guide (Super-GOAT, the direct prior art for the "penultimate-hidden-state → lightweight head → agent pool" pattern)
  - [katgpt-rs/.research/259_Routing_Extraction_Decouple_Per_Matrix_Freeze_Thaw.md](259_Routing_Extraction_Decouple_Per_Matrix_Freeze_Thaw.md) — per-matrix routing/freeze/thaw
  - [katgpt-rs/.research/269_Variable_Width_Shape_Adapter_Fusion.md](269_Variable_Width_Shape_Adapter_Fusion.md) — R269 (the documented adapter-routing-default failure mode; this note actively avoids it)
  - [katgpt-rs/.research/246_Manifold_Power_Iteration_MoE_Router.md](246_Manifold_Power_Iteration_MoE_Router.md) — MoE router precedent
  - [riir-ai/.research/125_MCTS_Collapse_Discriminator.md](../../riir-ai/.research/125_MCTS_Collapse_Discriminator.md) — verifier-accept signal source
  - [katgpt-rs/.benchmarks/023_block_diagonal_goat.md](../.benchmarks/023_block_diagonal_goat.md) — empirical block-diagonal vs full-rotation tradeoff (validates TRINITY's block-ε separability claim)
- **Shipped code (the prior art):**
  - `riir-ai/crates/riir-engine/src/role_transport.rs` — `RoleTransport::Diagonal` / `Orthogonal`, `RoleEmbeddingTable`, `apply_transport`
  - `riir-ai/crates/riir-engine/src/cgsp_runtime/runtime.rs` — `NpcCgspRuntime`, `PriorityTableBandit`, `TickReport`
  - `katgpt-rs/crates/katgpt-core/src/sense/brain.rs` — `NpcBrain` + HLA projection
  - `katgpt-rs/src/dash_attn/meta_router.rs` — `MetaRouter` bandit policy head
  - `riir-neuron-db/src/shard.rs` — frozen LoRA shard pool
- **Training-side redirect:** sep-CMA-ES + SVF → `riir-train/.research/` (separate note, out of scope for this session)

---

## TL;DR

**Verdict: GOAT.** TRINITY's runtime value distills to a **per-query T→W→V tri-role cycler** with verifier-accept early-exit, operating on a compact latent context (HLA state = penultimate-hidden-state equivalent). Three-layer grep (notes + code + vocabulary-translated) confirms the *pieces* are shipped: `role_transport.rs` (Diagonal/Orthogonal role projection = TRINITY's linear/block-diagonal-10 head), CGSP (persistent tri-role), CLR/Research 136 (agent-pool + verifier-accept), MCTS Collapse Discriminator (verifier signal), Plan 100 (block-diagonal empirics validate block-ε separability). The *missing piece* is the unified per-query role-rotating cycler primitive — a composition, not a new mechanism. **No Super-GOAT claim, no private guide.** Plan behind feature flag (`tri_role_cycler` open + `tri_role_runtime` private); GOAT gate requires G1 (≥ CLR baseline quality at ≤ +20% latency) + G2 (verifier-accept before max-turns on ≥ 60% of high-stakes decisions) + G3 (zero-alloc). sep-CMA-ES and singular value fine-tuning are training-time recipes → `riir-train` (out of scope here). The latent-space reframing (mandatory per skill) is the primary framing — adapter routing is one of several pool sources, not the headline.
