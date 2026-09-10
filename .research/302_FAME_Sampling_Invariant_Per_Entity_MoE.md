# Research 302: FAME — Sampling-Invariant Per-Entity MoE Composition

> **Source:** [FAME: Adaptive Functional Attention with Expert Routing for Function-on-Function Regression](https://arxiv.org/abs/2510.00621) — Gao, Chen, Zhang (Tsinghua / U-Iowa), NeurIPS 2025
> **Date:** 2026-06-25
> **Status:** Active — Super-GOAT via fusion; primitive + plan + private guide created this session
> **Related Research:** 276 (PersonalityWeightedComposition — per-layer cousin), 288 (KARC — per-NPC forecaster cousin, the "backward" Bi-NCDE pass), 257 (FuncAttn — **vocabulary collision, different mechanism**), 242 (MicroRecurrentBeliefState), 296 (Stokes/DEC — sampling-invariant substrate), 219 (TNO/DEC), [454 (HOPE — rank-1 capacity kernel)](454_HOPE_Hilbert_Schmidt_Capacity_Kernel.md)
> **Related Plans:** 321 (this primitive — open), 297 (PersonalityWeightedComposition), 308 (KARC), 314 (Stokes wrappers — `line_integral`), [469 (HOPE primitive — rank-1 parent selection)](../.plans/469_hilbert_schmidt_capacity_kernel_primitive.md)
> **Cross-ref (riir-ai):** Research 158 — *Per-NPC Committed Personality Blend Guide* (private Super-GOAT moat)
> **Cross-ref (riir-chain):** Research 003 (LatCal-Committed Karc Readout — the sync-boundary bridge this extends)
> **Cross-ref (riir-neuron-db):** Research 003 (KarcShard Storage Crossref — the freeze substrate this reuses)
> **Classification:** Public (katgpt-rs engine note). The paper is public; its distillation into runtime primitives is open (this note + Plan 321). The private Super-GOAT selling point lives at `riir-ai/.research/158_*.md`.
> **Verdict: Super-GOAT — the fusion (FAME per-function fixed MoE × KARC per-NPC forecaster × PersonalityWeightedComposition sigmoid kernel × NeuronShard freeze × LatCal commitment × DEC line integral) is a new capability class with no shipped prior art for the COMBINATION.**

---

> **PASS-Redirect:** Simplex blog "The geometry of nonergodic composition" (→ Research 545) — per-entity composition over K fixed experts with likelihood-weighted, sigmoid-gated readouts now ships modellessly as `NonergodicFilter` (`katgpt-micro-belief/src/nonergodic.rs`, Plan 592, Bench 706) — the two-level Bayes analog of this note's per-entity MoE composition; FAME's function-on-function regression training stays riir-train-side → redirect to R545 / Plan 592.

## ⚠️ Vocabulary Collision Alert (canonical lesson, DO NOT skip)

**"Functional Attention" means two different things in this corpus:**

| Source | Paper | Mechanism | Math |
|--------|-------|-----------|------|
| Research 257 / Plan 286 (`funcattn.rs`) | Xiao et al. ICML 2026, arxiv 2605.31559 | **Tikhonov k×k spectral transport operator** | Closed-form `(1-α)·K̃ᵀK̃ + α·I_d` ridge solve over basis-partitioned features |
| **This note (Research 302)** | Gao/Chen/Zhang NeurIPS 2025, arxiv 2510.00621 (FAME) | **Bi-NCDE continuous attention with per-function MoE** | Forward+backward CDE integration → continuous Q/K/V trajectories → Young-integral attention |

A grep for `functional attention` hits Research 257 first and would mislead a future reader into thinking FAME is already shipped. **It is not.** The two mechanisms are unrelated mathematically — they share only the name. This is the R296 vocabulary-translation failure mode in a new guise: paper-vocabulary grep alone returns a false positive, but for the OPPOSITE reason (same name, different mechanism) instead of the usual (different name, same mechanism).

**Rule for future grep:** when grepping for FAME's continuous attention, use `Bi-NCDE|continuous attention|Young integral|per-function MoE|function-on-function` — NOT `functional attention` (which hits the wrong primitive).

---

## TL;DR

FAME is a NeurIPS 2025 training paper for function-on-function regression. As a training method it routes to riir-train — BUT §3.5 modelless unblock passes, because FAME's three transferable primitives are modelless-validable: (1) **bidirectional NCDE latent state** (forward = current HLA, backward = KARC forecast), (2) **per-function MoE with FIXED routing weights** (NPC personality as a frozen blend of K archetype dynamics fields, computed once then committed), and (3) **Young-integral sampling invariance** (DEC `line_integral` already ships the discretization-invariant substrate).

The Super-GOAT is the fusion: **per-NPC committed archetype blend × KARC trajectory forecaster × PersonalityWeightedComposition sigmoid kernel × NeuronShard freeze × LatCal commitment × DEC sampling-invariant path sum.** The novel spine is "per-entity MoE blend with weights computed once from history and frozen" — no shipped primitive does this (PersonalityWeightedComposition drifts per-tick, dMoE routes per-token, KARC forecasts but doesn't blend).

**Distilled for katgpt-rs (modelless, inference-time):**
- Blend kernel: `f_π(z) = Σ_k sigmoid(π_k / τ) · f_k(z)` where `f_k` are K host-supplied operator fields and `π` is a committed weight vector
- One-shot weight computation: `π = sigmoid_project(trajectory_summary)` — computed ONCE per entity, then frozen
- Commitment: `(π, blake3)` — versioned, atomic-swappable, hashable; crosses sync boundary as raw K-scalar list (NOT the full field definitions)
- Sampling invariance: because `π` is frozen and `f_k` are frozen snapshots, the entity's trajectory through `f_π` depends only on its initial state — not on observation density (fog-of-war gaps, desync windows, snapshot thaw all preserve the personality)

---

## 1. Paper Core Findings

### 1.1 The three transferable primitives

| # | Paper mechanism | Math | Modelless-validable? (§3.5) |
|---|-----------------|------|---------------------------|
| 1 | **Bidirectional NCDE latent state** | `Z(t) = [Z_fwd(t), Z_bwd(t)] ∈ R^{2h}` — forward CDE captures past, backward CDE captures future | YES — forward = `evolve_hla` (current HLA), backward = KARC forecast (anticipated HLA) |
| 2 | **Per-function MoE with FIXED routing weights** | `π(j) = softmax(g(s(j)))` computed ONCE per function; `f^(j)_Θ(z) = Σ_k π_k^(j) · f_k(z)` frozen for entire trajectory | YES — replace softmax with sigmoid projection (AGENTS.md mandate); replace trained experts `f_k` with frozen archetype snapshots; weights `π` computed deterministically from trajectory summary |
| 3 | **Young-integral continuous attention** | `Ẑ(j)(t) = ∫ t̂_0^T α̂(j)(t,τ) V(j)(τ) dτ` — output depends only on driving function, not on partition | YES — DEC `line_integral` (Plan 314) ships the discretization-invariant rank-1 cochain path sum; Lipschitz stability `‖Ẑ-Ẑ̃‖_∞ ≤ L·‖X-X̃‖_{1-var}` is a DEC Hodge stability result |

### 1.2 What the paper does NOT claim (scope guardrails)

- **Token sequences unverified** — FAME operates on continuous functions (FoFR), not discrete tokens. The NLP/LLM domain is out of scope per the paper's own framing.
- **Training is required** for the expert vector fields `f_{θ_k}` and the router `g_ϕ`. The modelless version (this note) replaces trained experts with **frozen archetype snapshots** and the trained router with a **deterministic sigmoid projection** — see §3.5 path 2.
- **The Bi-NCDE is a training architecture** — backprop-through-CDE is the paper's method. The modelless version does NOT solve a CDE at runtime; it uses the Bi-NCDE as an architectural *metaphor* (forward=current state, backward=forecast state) mapped to existing primitives (`evolve_hla` + KARC forecaster).
- **Cross attention across functions** (§4.3) is standard multi-head attention — not novel, not distilled here.
- **CDE decoder** (§4.3, eq. 11) — maps to KARC's delay-basis ridge readout, already shipped. Not distilled separately.

### 1.3 Theoretical scaffolding (transferable as commitment guarantees)

- **Theorem 1 (Bi-NCDE existence/uniqueness):** `‖Z - Z̃‖_∞ ≤ e^{L(T-t_0)} · ‖X - X̃‖_{1-var}` — Lipschitz stability. This is the **deterministic commitment bound**: small input perturbation → proportionally small output change. Maps to LatCal fixed-point commitment invariants.
- **Proposition 3 (Sampling invariance):** "If two observation grids encode the same underlying functions, the operator returns identical outputs." This is the **DEC `d∘d=0` identity** in disguise — the operator depends only on the cochain, not on the cell refinement.
- **Lemma 1 (Mixed-field Lipschitz):** `L_mix = max_k {L_fwd_k, L_bwd_k}` — the blended field's Lipschitz constant is bounded by the worst expert. **Commitment implication:** the safety bound of a committed archetype blend is the max of the archetype bounds — a closed-form, deterministic quantity.
- **Theorem 6 (Rademacher complexity):** `R_N ≤ c·L*/√N` — generalization bound. Informational only; not a runtime check.

### 1.4 Empirical defaults (transferable as hyperparameters)

- **K = 3 default** for MoE — "accuracy improves as K increases from 1 to 3 and plateaus around K=3∼5". Healthy non-collapsed routing entropy at K=3. **We adopt K=3 as the default archetype count** (matches the 3-axis HLA valence/arousal/desperation triplet).
- **Sample-efficiency regime:** FAME wins most decisively at low sample counts (Cases 1–3, N=100). At N=500+ the advantage shrinks. **Implication:** archetype blends are most valuable for NEW NPCs (short history) — exactly the cold-start personality problem.

---

## 2. Distillation

### 2.1 What we already ship (the prior-art surface — verify before any novelty claim)

| Paper mechanism | Shipped cousin | File / Plan |
|---|---|---|
| Per-entity sigmoid-gated composition of N direction vectors | **PersonalityWeightedComposition** | Plan 297, Research 276, `crates/katgpt-core/src/personality_composition.rs` — `compose_into`, `drift` |
| Per-token MoE expert routing (block-level coreset) | **dMoE** | Research 161, Plan 181 — `top_p_coreset`, `select_arms_top_p` |
| Per-NPC trajectory forecaster (delay-basis ridge, fits in a shard) | **KARC** | Research 288, Plan 308, `riir-ai/crates/riir-games-civ/src/civ/map_tick/karc.rs` — `KarcForecaster<D,M,K>` |
| Per-NPC recurrent belief state (leaky integrator, byte-identical to `evolve_hla`) | **MicroRecurrentBeliefState / LeakyIntegrator** | Plan 276, Research 242, `crates/katgpt-core/src/micro_belief/` |
| HLA forward state evolution | **`evolve_hla`** | `crates/katgpt-sense/src/reconstruction.rs` |
| Forward + backward latent passes (bidirectional prefill) | **Plan 025 bidirectional prefill** | reader LoRA (prefill) + writer LoRA (decode) — the Bi-NCDE's fwd/bwd split, modellessly |
| Two-brain model (info brain = ground truth, think brain = belief) | **AGENTS.md §Spatial Cognition** | one-way bridge, fog-of-war gated |
| Discretization-invariant path sum / line integral | **DEC `line_integral`** | Plan 314, Research 296, `crates/katgpt-dec/src/` |
| Lipschitz-stable closed-form regression operator | **FuncAttn** (different mechanism — see vocabulary alert) | Plan 286, Research 257, `crates/katgpt-core/src/funcattn/mod.rs` |
| Frozen operator-field snapshots (Pod, BLAKE3, dendritic branch) | **NeuronShard** | `riir-neuron-db/src/shard/mod.rs` — `style_weights[64]`, `hla_moments[8]` |
| Deterministic linear-op commitment (2×2 fixed-point blocks) | **LatCal** | `riir-chain/src/encoding/latcal.rs` — `LatCalMatrix`, `to_fixed` |
| Per-NPC learned dynamics from trajectory (delay-embedded basis ridge) | **KARC** (again) | Plan 308 — the "backward" pass of FAME's Bi-NCDE |

### 2.2 What FAME adds that none of the above does alone

The fusion is the novelty, not any single component:

1. **Per-trajectory FIXED MoE blend** — PersonalityWeightedComposition drifts `w` per-tick based on reward surprise; dMoE routes per-token via bandit. **Neither computes the blend ONCE from a trajectory summary and FREEZES it for the entity's lifetime.** FAME's `π(j) = gϕ(s(j))` is computed once per function (per NPC) and never changes. This is the **commitment** primitive: the personality is a frozen artifact, not an online-adaptive weight vector.

2. **Operator fields vs direction vectors** — PersonalityWeightedComposition's `d_i ∈ ℝ^D` are static direction vectors (weighted sum). FAME's `f_k(z)` are operator fields that take the current state `z` and produce a dynamics update `dz`. The blend `f_π(z) = Σ_k π_k · f_k(z)` is a **dynamics blend**, not a feature blend. This is the difference between "personality as a weighted vote" and "personality as a weighted governance of motion".

3. **Sampling invariance as a commitment property** — DEC `line_integral` ships discretization-invariant path sums, but not as a **personality commitment**. FAME's Young-integral framing makes "the NPC's behavior is invariant to observation density" a first-class product property: fog-of-war gaps, network desync, and snapshot thaw all preserve the personality because `π` and `f_k` are frozen.

4. **Archetype pool as a shard library** — NeuronShard stores per-NPC latent state, but not a **library of K archetype operator fields** shared across all NPCs. FAME's MoE structure (K shared experts, per-function routing) maps to: K frozen archetype shards (e.g., aggressive/cautious/social/solitary), each NPC's personality is a committed blend weight vector `π` over the library.

### 2.3 Fusion (the Super-GOAT move)

| Fusion partner | What it ships | What FAME adds | Fusion product |
|---|---|---|---|
| **R276 PersonalityWeightedComposition** | Per-layer sigmoid composition with per-tick drift | Per-ENTITY FIXED blend (commit once, never drift); operator fields instead of direction vectors | "NPC personality as a committed archetype blend — frozen for the NPC's lifetime, BLAKE3-hashed, survives snapshot thaw" |
| **R288 KARC** | Per-NPC trajectory forecaster (delay-basis ridge) | The "backward" pass of Bi-NCDE — forecast = anticipated future HLA | "Bi-NCDE made modelless: forward = `evolve_hla`, backward = KARC forecast; both committed" |
| **R242 MicroRecurrentBeliefState** | Per-NPC belief kernel (leaky/attractor families) | A **blended** belief kernel — `evolve_hla_π = Σ_k π_k · evolve_hla_k` over K archetype kernels | "NPC belief evolution as a frozen archetype blend — each NPC has its own personality-tuned leaky integrator" |
| **R257 FuncAttn** (different mechanism — vocabulary alert) | Closed-form Tikhonov spectral transport | Nothing direct — different mechanism | (Disambiguation only — do NOT fuse; they solve different problems) |
| **R219/R296 DEC + Stokes** | `line_integral`, `codifferential`, sampling-invariant cochain ops | The Young-integral framing as a commitment property | "DEC line integral = FAME's continuous attention, made explicit; sampling invariance = `d∘d=0`" |
| **NeuronShard / MerkleFrozenEnvelope** | Frozen Pod with `style_weights[64]`, BLAKE3 commitment | A `ArchetypeBlendShard` subtype storing the K-weight vector `π` + archetype library reference | "Per-NPC personality frozen into a shard, replicated via chain, restorable on any node — the K floats are the sync artifact, not the full field definitions" |
| **LatCal** | Deterministic 2×2 fixed-point linear-op commitment | The K-weight vector `π` as a LatCal-committed raw scalar list | "Personality crosses the sync boundary as K committed floats; two nodes agree bit-for-bit on an NPC's archetype blend" |
| **R303 latent_functor** | Per-(source,target) relation: single direction vector `f` | A blended functor — `f_π = Σ_k π_k · f_k` over K archetype functors | "Relational stance as a committed archetype blend — the NPC's stance toward each relation inherits its personality" |
| **cgsp_runtime curiosity** | Coherence-decay + JS-uniqueness curiosity signals | A **personality-gated** curiosity — archetype blend determines curiosity direction | "Curiosity is not random; it's governed by the NPC's frozen archetype blend — explorers explore differently than guardians" |
| **Two-brain model (AGENTS.md)** | Info brain (ground truth) + think brain (belief), one-way bridge | The think brain's dynamics ARE the archetype blend `f_π` | "Two brains diverge by personality — each NPC's think brain evolves through its own committed archetype blend" |

### 2.4 Latent-space reframing (mandatory per fusion protocol §1.3)

Operating on each Super-GOAT factory module:

(a) **HLA per-NPC latent state** (`katgpt-core/src/sense/`, `riir-engine/src/hla/`): The archetype blend `f_π` IS the HLA update rule, per-NPC. Instead of one global `evolve_hla`, each NPC has `evolve_hla_π = Σ_k π_k · evolve_hla_k`. The 8-dim HLA (valence/arousal/desperation/calm/fear + 3) evolves through a personality-tuned kernel. **K=3 default matches the valence/arousal/desperation triplet** — each archetype governs one primary affect axis.

(b) **latent_functor** (`riir-engine/src/latent_functor/`): `extract_functor` becomes `extract_archetype_blend(trajectory)` — compute `π` once from the NPC's relational history, then freeze. `apply_functor` becomes `apply_blended_functor(π, source_state)` — the stance toward any relation inherits the NPC's personality blend. The `ReestimationScheduler` already implements "drift-triggered re-fit"; the archetype blend fits as a **higher-level commitment** — re-fit only on major personality events (taming, faction change, trauma), not per-tick.

(c) **cgsp_runtime curiosity** (`riir-engine/src/cgsp_runtime/`): Curiosity becomes `curiosity_t = ‖actual_hla_t − forecast_hla_t‖_π` — the surprise signal is **projected through the NPC's personality blend**. An aggressive archetype weights danger surprises more; a social archetype weights companionship surprises more. Same forecast (KARC), different surprise weighting per NPC.

(d) **LatCal fixed-point commitment** (`riir-chain/src/encoding/`): The K-weight vector `π ∈ ℝ^K` (K=3 default) is a raw scalar list. LatCal commits it as 3 fixed-point values crossing the sync boundary. **Never commit the full archetype field definitions** — those are library artifacts referenced by shard hash, not per-NPC state. The sync artifact is exactly K floats per NPC.

(e) **NeuronShard / freeze envelope** (`riir-neuron-db/src/`): `ArchetypeBlendShard` subtype. Layout: `[zone_hash(32) | π_flat(K·4) | archetype_library_hash(32) | version(4) | blake3(32) | merkle_root(32)]` ≈ 132 + 4K bytes, K=3 → 144 bytes, padded to 192. `MerkleFrozenEnvelope` wraps it for self-play freeze/thaw. **The archetype library itself is a separate frozen artifact** (K NeuronShards, one per archetype), referenced by hash — not duplicated per NPC.

(f) **DEC Stokes-calculus** (`katgpt-dec/src/`): The Young-integral sampling invariance maps to `d∘d=0` — the operator's output depends only on the cochain, not on cell refinement. The continuous attention integral `Ẑ(j)(t) = ∫ α̂ V dτ` IS a DEC rank-1 cochain path sum (`line_integral` in Plan 314). The Lipschitz stability `‖Ẑ-Ẑ̃‖_∞ ≤ L·‖X-X̃‖_{1-var}` is a Hodge-decomposition stability result. **Curse-of-dimensionality caveat (R296):** boundary-vs-volume wins only for d ≤ 3; HLA (d=8) and full shards (d=64) do NOT benefit from boundary-only computation. The DEC mapping is for the *sampling-invariance property*, not for perf.

---

## 3. §3.5 Modelless Unblock Protocol (MANDATORY — passed)

Before any riir-train deferral, exhaust the three modelless paths:

**Path 1 (freeze/thaw snapshot correction):** **PASS.** The K archetype operator fields `f_1..f_K` are frozen snapshot archetypes (e.g., from different game states, different NPC personality archetypes trained offline once in riir-train, then frozen forever). The per-NPC blend `π` is computed once and frozen. Both are freeze/thaw artifacts — `MerkleFrozenEnvelope` wraps the blend, archetype library is a frozen shard set. No runtime weight mutation.

**Path 2 (raw/lora reader-writer hot-swap):** **PASS.** The archetype fields can be LoRA pairs `{reader_k, writer_k}`. The blended LoRA `L_π = Σ_k π_k · L_k` is a **deterministic linear combination in LoRA space** (LoRA pairs form a vector space — sum of low-rank matrices is low-rank). Constructing `L_π` from K frozen LoRA pairs + K sigmoid weights is modelless (weight addition, no backprop). This is the canonical §3.5 path 2 case: the correction (per-NPC personality) is derived in closed form (linear blend), not learned.

**Path 3 (latent-space correction):** **PASS.** The blend weights `π` are computed via sigmoid projection: `π_k = sigmoid(g_k(s) / τ)` where `s` is the trajectory summary and `g_k` are K direction vectors. This is exactly the modelless MoE pattern — dot-product projection onto K learned direction vectors, gated by sigmoid (never softmax per AGENTS.md). No gradient descent.

**Decision protocol result:** All three paths pass → **MODELLESS-VALIDABLE.** The primitive ships in katgpt-rs without any riir-train dependency for the per-NPC personality computation. The K archetype fields themselves are pre-trained offline (riir-train's job, once, for the library) — but that is the freeze/thaw substrate, not a per-NPC training dependency. **No riir-train deferral.**

---

## 4. Verdict

### Tier: **Super-GOAT — via fusion**

| Q | Answer | Evidence |
|---|--------|----------|
| **Q1: No prior art?** | **YES (for the combination)** | The fusion — per-trajectory fixed MoE blend × KARC forecaster × PersonalityWeightedComposition sigmoid kernel × NeuronShard freeze × LatCal commitment × DEC sampling-invariant line integral — has zero shipped prior art. Each *component* has a cousin (PersonalityWeightedComposition per-layer, dMoE per-token, KARC forecaster, DEC line integral, LatCal commit), but **no shipped primitive computes a per-entity MoE blend ONCE from a trajectory summary and FREEZES it for the entity's lifetime**. Vocabulary check passed: grep on FAME terms (`Bi-NCDE|continuous attention|Young integral|per-function MoE|function-on-function`) AND codebase terms (`PersonalityWeightedComposition|archetype|blend|evolve_hla|frozen.*expert|commit.*blend`) was performed across all 5 repos at both `.md` and `.rs` layers. Vocabulary collision with Research 257 "Functional Attention" explicitly resolved (different mechanism — see §Vocabulary Alert). |
| **Q2: New capability class?** | **YES** | "Per-NPC committed archetype blend personality, frozen for the NPC's lifetime, sampling-invariant to observation gaps, BLAKE3-committed, quorum-reproducible" is a new capability class. No current primitive does this — PersonalityWeightedComposition drifts per-tick, dMoE routes per-token, KARC forecasts but doesn't blend, NeuronShard stores per-NPC state but not a blend-over-library. |
| **Q3: Product selling point?** | **YES** | "Our NPCs have committed personalities that survive observation gaps, network desync, and snapshot thaw — a frozen blend of K archetype dynamics fields, BLAKE3-committed, quorum-verifiable across nodes. Crowd-scale personality consistency without per-NPC training." |
| **Q4: Force multiplier?** | **YES (≥9 pillars)** | Connects: HLA (latent substrate), PersonalityWeightedComposition (composition kernel cousin), KARC (forecaster = backward Bi-NCDE pass), latent_functor (relational stance blend), cgsp_runtime (personality-gated curiosity), NeuronShard/freeze (persistence), LatCal (commitment bridge), DEC (sampling invariance), two-brain model (think brain dynamics). |

**Mandatory outputs (this session):**
1. **Open primitive** → `katgpt-rs/.plans/321_sampling_invariant_per_entity_moe_primitive.md` (generic math, no game IP — `CommittedFieldBlend<N, D>` kernel + sigmoid projection + BLAKE3 commitment).
2. **Private guide** → `riir-ai/.research/158_per_npc_committed_personality_blend_guide.md` (selling point: per-NPC committed personality + crowd consistency + archetype library + K=3 default mapping to valence/arousal/desperation).
3. **Cross-ref guides** → `riir-neuron-db/.research/003_KarcShard_Storage_Crossref.md` (already exists — KarcShard is the substrate; `ArchetypeBlendShard` is a subtype to add), `riir-chain/.research/003_LatCal_Committed_Karc_Readout.md` (already exists — LatCal commitment of the K-weight vector extends this).
4. **Private plan** → `riir-ai/.plans/336_committed_personality_runtime_integration.md` (deferred — file after Plan 321 GOAT gate passes; runtime wiring: HLA hook, latent_functor interop, archetype library loader, KarcShard freeze integration).

**One-line reasoning:** FAME's value is not the NCDE training architecture (which routes to riir-train) and not the "functional attention" name (which collides with Research 257); it is the *combination* of per-function fixed MoE blend + Young-integral sampling invariance + bidirectional latent state as a **per-NPC committed archetype-blend personality** that fits in a shard, crosses the LatCal sync boundary as K=3 floats, and is provably invariant to observation density. That combination is the Super-GOAT.

---

## 5. Caveats and known risks

1. **PersonalityWeightedComposition overlap is real.** PersonalityWeightedComposition already does sigmoid-gated composition with drift. FAME's primitive is **strictly different** (per-entity FIXED vs per-layer DRIFTING; operator fields vs direction vectors; commit-once vs update-per-tick). The integration story must position `CommittedFieldBlend` as a **commitment-tier companion** to PersonalityWeightedComposition, not a replacement. PersonalityWeightedComposition handles online adaptation; `CommittedFieldBlend` handles lifetime commitment. **Do not re-ship the sigmoid kernel** — reuse `PersonalityWeightedComposition::compose_into` as the inner loop, wrapping it with a "compute π once then freeze" outer layer.

2. **KARC overlap is real.** KARC already forecasts per-NPC trajectories. FAME's Bi-NCDE "backward pass" maps to KARC's forecast. The integration must position FAME as the **personality commitment layer** and KARC as the **forecast engine** — they compose, they don't compete. KARC answers "what will this NPC do next?"; FAME answers "what kind of NPC is this, committed immutably?".

3. **Vocabulary collision with Research 257 is the #1 trap.** A future grep for "functional attention" will hit Research 257 (Tikhonov spectral transport) and falsely conclude FAME is shipped. The §Vocabulary Alert at the top of this note is the only defense. **When referencing FAME, always use "FAME Bi-NCDE" or "FAME per-function MoE" — never bare "functional attention".**

4. **K=3 default is empirical, not theoretical.** FAME's K=3 comes from FoFR benchmark tuning. Our K=3 mapping to HLA's valence/arousal/desperation triplet is a *coincidence we exploit*, not a derivation. If HLA grows to 8 dims (full affect vector), K may need to grow to 5–8. Treat K as a host-configured constant, not a fixed primitive parameter.

5. **Archetype library is a riir-train artifact.** The K archetype operator fields `f_1..f_K` must be trained offline (once, in riir-train) before they can be frozen into the library. This is the freeze/thaw substrate, NOT a per-NPC training dependency — but it IS a one-time upstream dependency. The primitive ships modellessly at runtime; the library itself requires an upstream training step. **Document this boundary clearly in the guide.**

6. **Commitment granularity.** The K-weight vector `π` is committed per NPC. For 10,000 NPCs × K=3 × 4 bytes = 120 KB. Fits comfortably in Warm tier. Re-commit cadence: only on major personality events (taming, faction change, trauma) — NOT per-tick. The `ReestimationScheduler` from latent_functor is the right trigger mechanism.

7. **Sampling invariance holds only if `π` and `f_k` are both frozen.** If either drifts (online adaptation of archetypes, or per-tick `π` updates), the Young-integral invariance breaks. **The primitive's contract is: commit once, never mutate.** Drift is PersonalityWeightedComposition's job, not this primitive's.

---

## 6. Next steps (see Plan 321)

Phase 1: ship `CommittedFieldBlend<N, D>` + `ArchetypeFieldSource` trait in `crates/katgpt-core/src/committed_field_blend.rs` behind `committed_field_blend` feature. Reuse `PersonalityWeightedComposition::compose_into` for the inner blend; add the outer "compute π once then freeze" layer + BLAKE3 commitment. Zero game IP. GOAT gate G1–G3 on synthetic archetype blends (3 fixed operator fields, 100 entities, verify sampling invariance under observation gaps).

Phase 2–4: runtime integration in riir-ai (Plan 336, deferred), `ArchetypeBlendShard` subtype in riir-neuron-db (extends existing KarcShard), LatCal commitment of the K-weight vector in riir-chain (extends existing Research 003).

---

## TL;DR (one-line)

FAME = Bi-NCDE continuous attention + per-function fixed MoE + Young-integral sampling invariance; the math pieces are largely shipped (PersonalityWeightedComposition per-layer, KARC per-NPC forecaster, DEC line_integral, NeuronShard freeze, LatCal commit); the Super-GOAT is the *combination* as the first per-NPC committed archetype-blend personality that fits in a shard, crosses the LatCal sync boundary as K=3 floats, and is provably invariant to observation density. **Vocabulary collision with Research 257 "Functional Attention" (Tikhonov spectral transport) explicitly resolved — different mechanism, same name.**
