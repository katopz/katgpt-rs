---
name: research
description: Research workflow for distilling ML/AI papers into modelless inference primitives, freeze/thaw runtime patterns, and latent-space operations across the katgpt-rs / riir-ai / riir-train repo trio. Use when reading arxiv papers, deciding which repo a paper belongs in, creating .research/ notes or .plans/ files, implementing modelless inference primitives, or routing training-vs-inference insights. Enforces the 3-repo commercial strategy, modelless-first constraint, latent-to-latent preference, and freeze/thaw-over-fine-tuning rule.
---

# Research Workflow — Modelless Inference, Freeze/Thaw, Latent-to-Latent

Training-method research lives in `riir-train`. This repo (`katgpt-rs`) and `riir-ai` ship **freeze/thaw runtime + self-learn/adaptive NPCs + latent-space operations**. No LoRA training, no adapter fine-tuning, no optimizer research here. If a paper's value is its training loop → `riir-train/.research`. If its value is a latent-space insight, a routing trick, a freeze/thaw pattern, or a modelless inference primitive → distill here.

## When to use this skill

Activate when the user (or you) are doing any of:

- Reading / fetching / summarizing an ML, AI, or systems paper (arxiv, PDF, blog).
- Deciding which of the 3 repos a paper or idea belongs in.
- Creating a new `.research/NNN_*.md` note or `.plans/NNN_*.md` plan.
- Implementing a modelless inference primitive (pruner, bandit, router, speculative decode, KV-cache op, sparse attention, quantization-aware inference).
- Designing freeze/thaw snapshot cycles, adapter hot-swap, or runtime adapter routing.
- Designing latent-to-latent operations (dot-product projection, sigmoid gating, manifold geometry, spectral methods on activations).
- Designing MMORPG-scale game AI (thousands of concurrent NPCs, 20Hz tick, fog-of-war, zone attention, emergent social/economic behavior).

Do NOT activate for: pure refactor tasks, bug fixes with no research angle, or ordinary feature work that doesn't touch the research/plans folders.

## Repos (siblings under the same parent)

- `katgpt-rs/` — public MIT engine. Generic modelless inference primitives. **No game IP, no chain IP.**
- `riir-ai/` — private game product. Freeze/thaw runtime, self-learn, chain, game systems.
- `riir-train/` — private training vault. Adapter training, optimizers, loss functions. Out of scope for this workflow — just note "→ riir-train" and stop.

Always reference files with project-relative paths (e.g. `katgpt-rs/.research/238_*.md`, `riir-ai/.plans/NNN_*.md`). The agent can `read_file` these directly.

## Read first (grounding) — MANDATORY pre-flight

**Hard rule:** before any distillation, verdict, or file creation, you MUST do **both** of these in this session:

1. **`read_file` all three READMEs** — these define repo purpose, current state, and the raw-vs-latent sync boundary the research must respect. Skipping this is the #1 cause of research notes that ignore the actual codebase architecture.
2. **`list_directory` both `.research/` folders** — these hold the existing distillation corpus you must not duplicate.

**Mandatory (before any verdict):**
- `katgpt-rs/README.md` (`read_file`) — public engine purpose, architecture, current feature set.
- `riir-ai/README.md` (`read_file`) — private runtime context (freeze/thaw, self-learn, chain, game systems).
- `riir-armageddon/README.md` (`read_file`) — arena/game-product domain types, raw-vs-latent boundary, sync semantics, anti-cheat rules. The research MUST respect this boundary.
- `katgpt-rs/.research/` (`list_directory`) — public modelless research corpus (do not duplicate).
- `riir-ai/.research/` (`list_directory`) — private runtime/game/chain research corpus (do not duplicate).

If you have NOT `read_file` all three READMEs AND `list_directory` both `.research/` folders, STOP and do so now. Do not create any file until all five are done.

Then read for additional context (as relevant to the topic):
- `katgpt-rs/src/` + `katgpt-rs/crates/katgpt-core/src/` — existing modelless primitives (ConstraintPruners, bandits, DDTree, speculative decode).
- `riir-ai/crates/` — runtime IP: `riir-engine`, `riir-games`, `riir-chain`, `riir-chaind`, `riir-ffi`, `riir-data`, `riir-examples`.
- `katgpt-rs/.plans/` + `riir-ai/.plans/` — existing plans. **Do NOT list these in pre-flight.** Grep them during fusion search (§Workflow step 1), not as grounding — they describe what we *plan to build*, not what the repos *are*.

## Primary focus (distill HERE in katgpt-rs / riir-ai)

**Fusion-first mindset:** The highest-value Super-GOATs in this codebase come from **fusing 2–3 papers/primitives into a novel combination**, not from direct-mapping a single paper. Always grep `.research/` + `.plans/` for the 2–3 closest cousins before verdict, and ask: "what does paper × note A × note B produce that none of them alone can?" Examples that shipped: Gemini Fourier × LatCal (research 212 → plan 242); EGA × SpectralQuant (research 100 × 039); collapse-aware × bandit × sigmoid-margin (plans 212 × 157 × 061). See §Workflow step 1 for the full fusion protocol.

- **Latent-to-latent operations** — anything that stays in embedding/latent space: dot-product projections, cosine similarity retrieval, sigmoid-gated routing, manifold geometry, spectral methods on activations. Prefer operating on latents over decoding to tokens then re-encoding. **Fusion hook:** combine with freeze/thaw to version latent-direction vectors; combine with self-learn to update direction vectors from runtime curiosity signal.
- **Freeze/thaw patterns** — versioned weight snapshots, atomic hot-swap, lock-free read paths, BLAKE3/commitment-checked adapter reload, per-entity personality divergence via snapshot versioning. **Fusion hook:** combine with runtime adapter routing to dispatch by latent-state similarity; combine with self-learn to snapshot emergent NPC personalities.
- **Runtime adapter routing** — selecting between frozen adapters by state/objective/context (Dynamic Pair, Polytope, dMoE — all inference-time, zero training). **Fusion hook:** combine with freeze/thaw to make the adapter pool itself versioned and BLAKE3-committed; combine with bandits to learn routing policy online.
- **Self-learn / adaptive CoT** — runtime curiosity, entropy-driven exploration, collapse detection/recovery, latent prediction SSL, trajectory folding. No LLM training, no backprop through weights — runtime self-improvement via latent-space updates is welcome. **Fusion hook:** combine with MMORPG-scale game AI to give thousands of NPCs independent curiosity/entropy signals; combine with freeze/thaw to checkpoint learned latent directions.
- **Modelless inference primitives** — ConstraintPruners, bandits, DDTree, speculative decode, sparse attention, quantization-aware inference.
- **MMORPG-scale game AI** — thousands of concurrent NPCs each with independent latent state, real-time latency budgets (20Hz tick, plasma/hot tier), spatial partitioning + fog-of-war, emergent social/economic behavior (factions, trade routes, reputation), zone-level attention routing, crowd-scale curiosity/exploration signals. Latent ops must batch across many entities; raw sync must stay bit-identical for deterministic replay/anti-cheat.

## Redirect to riir-train (do NOT distill here)

If a paper is training-only → note "→ riir-train" in one line and stop. Do not create files in this session for it.

**By topic:**
- LoRA / OFT / SPEFT / IA3 / QLoRA / ManifoldE / BAKE / GPart / MSA / Dendritic and all adapter-**training** methods.
- Training optimizers (Muon, Adam variants, symmetry-compatible optimizers).
- Training loss functions, curricula, distillation recipes.
- Quantization-aware **training** (quantization-aware **inference** stays here).
- DPO / GRPO / SFT / RL **training** pipelines (runtime GRPO self-play stays in `riir-ai` — it updates latent state, not weights).
- Anything that requires backpropagation through base weights.

**By user-request phrasing (these mean "→ riir-train"):**
- "Train a LoRA adapter to do X"
- "Fine-tune with method Y"
- "Optimizer Z improves convergence"
- "Distillation recipe from teacher to student"
- "Quantization-aware training" (but "quantization-aware inference" stays here)
- "DPO/GRPO/SFT/RL training pipeline" (but runtime GRPO self-play stays in riir-ai)

## Distillation targets (3-repo strategy)

Per `katgpt-rs/.research/003_Commercial_Open_Source_Strategy_Verdict.md`:

| Repo | Role | What lands here |
|------|------|-----------------|
| `katgpt-rs` (public, MIT) | Engine — modelless inference framework | Generic primitives: ConstraintPruner traits, bandits, DDTree, speculative decode, sparse attention kernels. **No game IP, no chain IP.** |
| `riir-ai` (private) | Game product — freeze/thaw runtime, self-learn, chain | Runtime IP: `LoRAWeightVersion`, `LoRAHotSwap`, `dispatch_lora_merge`, `TrainingProvider` trait, routing, game systems, neuro-symbolic chain. |
| `riir-train` (private) | Training research vault | **Only if the paper's value is its training method.** Out of scope for this workflow — just note "→ riir-train" and move on. |

Distill into:
- **Modelless** → `katgpt-rs/.research/` + `katgpt-rs/.plans/` + `katgpt-rs/src/` (or `katgpt-rs/crates/katgpt-core/`)
- **Runtime/game/chain** → `riir-ai/.research/` + `riir-ai/.plans/` + `riir-ai/crates/`
- **Training-only** → note the redirect, do not create files in this session

## Workflow

### 0. Read & classify the paper

Fetch via `https://r.jina.ai/https://arxiv.org/pdf/{ID}` (per AGENTS.md). Ask: *is the value in the training loop, or in a latent-space / inference / routing insight?* If training-only → note "→ riir-train", stop.

### 1. Distill fundamentally — fuse, don't just direct-map

Don't direct-map the paper to our code. Find the transferable primitive: the geometric, spectral, or information-theoretic insight that works without the paper's training setup. **Then look for fusion opportunities**: cross-pollinate this paper's insight with existing `.research/` notes, `.plans/`, and shipped primitives to synthesize a *novel* combination. The highest-value Super-GOATs in freeze/thaw runtime and self-learn/adaptive CoT almost always come from **fusing** 2–3 papers, not from a single-paper direct mapping.

**Fusion examples that shipped:**
- Gemini Fourier × LatCal → `katgpt-rs/.research/212_Gemini_Fourier_LatCal_Fusion_Verdict.md` → `katgpt-rs/.plans/242_Fourier_Smoothed_Potential_Fields_LEO.md`
- EGA spectral salience × SpectralQuant eigenbasis KV compression → `katgpt-rs/.research/100_*.md` + `039_*.md`
- Collapse-aware × bandit coverage × sigmoid margin → `katgpt-rs/.plans/212_*`, `157_*`, `061_*`
- G-Zero self-play × Hint-δ bandit × absorb-compress → `katgpt-rs/.plans/049_*` (modelless self-play distillation, 1.16M cycles/sec)

**Fusion protocol:**
1. **MANDATORY — grep BOTH repos in this session, BOTH layers (notes AND code). Do NOT stop after the first repo or the first layer.** Run keyword / paper-title / author / primitive-name grep across:
   - `katgpt-rs/.research/` + `katgpt-rs/.plans/` (intent — what we planned)
   - `riir-ai/.research/` + `riir-ai/.plans/` (intent)
   - `katgpt-rs/src/` + `katgpt-rs/crates/` (shipped primitives — what actually exists)
   - `riir-ai/crates/` (shipped runtime)
   - `riir-armageddon/crates/` (shipped game/arena domain types, raw-vs-latent boundary)

   (riir-train is deliberately excluded — training methods are out of scope for this workflow.)

   Two layers, three repos. The closest cousin is frequently in the OTHER repo (e.g., a `katgpt-rs` modelless primitive fused with a `riir-ai` game runtime guide — see Gemini Fourier × LatCal) OR in the CODE not the notes. **Notes describe intent; code describes what shipped.** A mechanism can ship without a research note — e.g., HLA's `evolve_hla` (`katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs`) is a per-NPC recurrent belief-state kernel with no `.research/` note framing it as such; a notes-only grep misses it and produces a false Super-GOAT claim (this exact failure happened on Research 242 — verdict had to be revised Super-GOAT → GOAT). If you only grep `katgpt-rs/.research/`, you will miss both axes and produce a duplicate, weaker note, or an overclaimed verdict.
2. After finding the transferable primitive of *this* paper, list the 2–3 closest existing notes/plans **across both repos** and ask: "what novel combination of this paper + note A + note B produces a capability none of them has alone?" Write that combination into the research note's §Distillation as a **Fusion** subsection, even if you don't plan it yet.
3. Verdict by the commercial strategy doc (`003_*.md`): **Super-GOAT** > GOAT > Gain > Pass (see §Verdict tiers below). **A fusion that produces a new capability class is a strong Super-GOAT candidate — check the novelty gate (§1.5).**
4. Create research `.md` at the right repo (see table above).

**File naming:** `{NNN}_{Short_Title_with_Underscores}.md` where NNN is the next free number (zero-padded to 3 digits, e.g. `239_`, `240_`). Check the folder first — numbers may be non-contiguous; pick the next free slot.

**Research note format** (see `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md` for a canonical example):

```markdown
# Research NNN: <Title>

> **Source:** <paper title + arxiv link + authors + date>
> **Date:** YYYY-MM-DD
> **Status:** Active | Done | Shelved
> **Related Research:** NNN (short note), ...
> **Related Plans:** NNN (short note), ...
> **Cross-ref (riir-ai):** Research NNN, Plan NNN   ← only if cross-repo
> **Classification:** Public | Private   ← katgpt-rs notes are always Public

---

## TL;DR

<2-4 sentences: the distilled primitive, why it matters here, what it unblocks>

**Distilled for katgpt-rs (modelless, inference-time):**
<the transferable insight, stripped of training setup>

---

## 1. Paper Core Findings
...
## 2. Distillation
...
## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars). Creates a moat. | Open primitive → katgpt-rs. **Architectural guide → riir-ai/.research/**. Plans → both repos as needed. |
| **GOAT** | Provable gain (latency/quality/security) over existing approach, but not a new class of capability. Promotes to default if it wins. | Plan + implement → appropriate repo. Feature flag + benchmark. |
| **Gain** | Incremental improvement, useful but not headline-worthy. | Plan only, behind feature flag. |
| **Pass** | Not relevant to modelless/latent/freeze-thaw/runtime, OR training-only (→ riir-train note, stop). | One-line note. No files created in this session. |

**One-line reasoning required for each verdict.** For Super-GOAT: state the selling point explicitly.
```

### 1.5. Novelty gate — is this Super-GOAT?

Before planning, score novelty. Ask all four:

1. **No prior art?** Grep `.research/` + `.plans/` across all repos AND grep the shipped code (`katgpt-rs/src/`, `katgpt-rs/crates/`, `riir-ai/crates/`, `riir-armageddon/crates/`) for the primitive name and mechanism keywords. **Notes describe intent; code describes what shipped.** A mechanism can ship without a dedicated research note — HLA's `evolve_hla` is the canonical example (a per-NPC recurrent belief-state kernel in `katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs` with no notes-level prior-art framing; missing it caused the Research 242 Super-GOAT overclaim). If the code already covers the mechanism → not novel, Gain at best. **This two-layer check (notes + code) is mandatory — notes-only is the #1 cause of false Super-GOAT claims.**
2. **New class of behavior?** Not better numbers, but something no incumbent can do (a new capability, not an optimization).
3. **Product selling point?** Can you finish the sentence: "Our NPCs/systems do X that no competitor can"? If you can't → Gain.
4. **Force multiplier?** Connects to ≥2 existing pillars/systems (check connection map in `.research/`). Solo novelty without integration = GOAT, not Super-GOAT.

**If YES to all 4 → verdict = Super-GOAT.** Mandatory outputs:
1. **Open primitive** → `katgpt-rs` (generic math, no game semantics).
2. **Architectural GUIDE** → `riir-ai/.research/NNN_*.md` (the private selling-point doc). The guide MUST include:
   - TL;DR with commercial value (the selling point in one sentence)
   - Distilled primitive (how the mechanism works modellessly)
   - Connection map (which existing systems it multiplies)
   - Latent vs raw boundary (what crosses sync, what stays local)
   - What stays private vs open
   - Validation protocol (how to prove it's Super-GOAT, not just hype)
   - Implementation priority table (P0–P3)
3. **Plan(s)** → `katgpt-rs/.plans/` (open) and/or `riir-ai/.plans/` (private runtime).

**If NO to any → proceed to GOAT/Gain verdict.** Plan only, no guide.

> **Rule:** Super-GOAT ideas are the private IP moat. The open primitive is the adoption hook; the riir-ai guide is the selling point. Never ship the guide publicly. Never skip the guide for a Super-GOAT — that's losing the knowledge.
>
> **No "candidate" escape hatch.** If you write "all 4 YES", "passes the novelty gate", or "Super-GOAT candidate" anywhere in a note (main verdict OR a fusion subsection), the mandatory outputs above apply **in this same session** — open primitive in katgpt-rs, **riir-ai guide created now**, plans as needed. The guide *contains* the validation protocol (G1–Gn gate), so you create it **before** running the gate, not after. Deferring the guide "until validation passes" inverts the order and silently drops the moat doc — this is the #1 way selling points leak into the public repo.
>
> If you are NOT confident enough to commit all 4 YES right now, **do not write "Super-GOAT candidate"**. Write "fusion idea — novelty TBD, needs Q1–Q4 check before verdict" and create an issue in `.issues/` to track the follow-up. "Candidate" is not a deferred-commitment escape hatch — it either triggers the guide now, or it gets downgraded to an issue.

### 2. If gain (or GOAT), plan it

Add plan `.md` to `katgpt-rs/.plans/` (modelless) and/or `riir-ai/.plans/` (runtime/game/chain). Use `## Phase N` sections with `- [ ]` per task (mark `- [x]` when done). **Never** plan into `riir-train` from this workflow.

> Super-GOAT plans should be created AFTER the riir-ai guide. The guide is the strategy; the plan is the execution.

**Plan format** (see `katgpt-rs/.plans/271_attention_matching_compaction.md` for a canonical example):

```markdown
# Plan NNN: <Title>

**Date:** YYYY-MM-DD
**Research:** [katgpt-rs/.research/NNN_*.md](../.research/NNN_*.md)
**Source paper:** [arxiv ID.NNN](https://arxiv.org/abs/ID) — <short cite>
**Target:** `katgpt-rs/src/<module>/` (new module) + Cargo feature `<feature_name>`
**Status:** Active — Phase N <state>

---

## Goal

<one paragraph: what ships, what it enables, GOAT gate>

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** <concrete, verifiable task>
- [ ] **T1.2** ...
```

**GOAT gate rule** (AGENTS.md): every plan that introduces a new technique must have a feature flag and a benchmark proving the gain before promoting to default. Demote the loser if the new technique wins.

### 3. Implement to unblock

If a plan is blocked by a missing primitive, implement the minimal version. After GOAT check + proof of gain: promote to default if it wins, demote the loser.

### 4. Search if curious

Keyword search arxiv:

```
https://r.jina.ai/https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term={KEYWORD}&terms-0-field=abstract&classification-computer_science=y&classification-mathematics=y&classification-physics_archives=all&classification-statistics=y&classification-include_cross_list=include&date-filter_by=all_dates&date-year=&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first
```

Good keywords: `latent space routing`, `adapter hot-swap`, `inference-time composition`, `spectral pruning`, `sigmoid gating`, `snapshot consistency`, `lock-free weight swap`.

## Constraints (non-negotiable)

1. **Modelless first** — inference-time only. No LLM training, no backprop through base weights. Closest to "training" allowed: freeze/thaw snapshot cycles and latent-space direction-vector updates at runtime.
2. **Latent-to-latent preferred** — operate in embedding/latent space as long as possible. Decode to tokens or project to raw scalars only at the boundary. Use dot-product + **sigmoid** (never softmax) for projections onto learned direction vectors. Semantic domain (emotion, mood, curiosity, style) → latent. Physical domain (position, HP, wallet balance) → raw, deterministic, synced.
3. **Freeze/thaw over fine-tuning** — the only weight mutation allowed at runtime is swapping a frozen snapshot (atomic, versioned, BLAKE3-checked). Never mutate weights in-place during inference. If a paper needs gradient updates, redirect to riir-train.
4. **Self-learn / adaptive CoT welcome** — runtime curiosity, latent prediction, trajectory folding, collapse detection. These update latent state / direction vectors / routing tables, NOT base weights.
5. **3-repo discipline** — katgpt-rs (public engine) → riir-ai (private runtime/game/chain) → riir-train (private training). Keep the commercial strategy intact. Training know-how never leaks to katgpt-rs.
6. **SOLID, DRY** — per `katgpt-rs/.contexts/optimization.md`. Zero-allocation hot paths. Pre-computed lookup tables. Fixed-size arrays for bounded domains.
7. **Tests/examples** — before/after showing the gain (latency, quality, or security). For latent ops: show the projection preserves ranking. For freeze/thaw: show readers never see torn snapshots.
8. **CPU/GPU/ANE auto-route** — threshold-adaptive dispatch. Plasma (µs, CPU/SIMD) → Hot (sub-ms, GPU) → Warm/Cold (ms+, GPU/ANE). Latent ops that fit in L1 cache stay on SIMD; manifold ops that need batched matmul go to GPU.
9. **Plasma → Hot → Warm → Cold → Freeze tiering** — aim for perf on game side (plasma/hot latency budget) AND security on chain side (cold/freeze commitment, BLAKE3-hashed, tamper-evident). Latent state that crosses the sync boundary MUST be raw scalars (valence/arousal/desperation/calm/fear), never the full embedding vector.

## Latent vs raw space rules (critical for game AI)

Reinforce these when designing game systems or chain state:

- **Physical domain** (position, velocity, HP, wallet balance): MUST remain raw exact values. Deterministic replay, quorum sync, anti-cheat require bit-identical reconstruction.
- **Semantic domain** (emotion, mood, curiosity, style, habit): SHOULD operate in latent space via dot-product + sigmoid onto learned direction vectors.
- **Social domain** (encounters, relationships, factions): SHOULD produce KG triples from proximity in latent/embedding space, not from raw coordinate distance.

**Sync boundary:** if data flows through `SyncBlock → ChainConsensus` quorum commit → Cold tier, it MUST be raw and deterministic. If data is consumed locally (emotion projection, shard retrieval, consolidation sleep-cycle), it SHOULD be latent. Bridge functions (raw→latent projection, latent→raw scalar clamp) MUST be zero-allocation, gateable by feature flag, and not introduce sync dependency.

**KG triple emission:** semantic encounters → KG triple from latent similarity. Physical events → TxDelta with raw values, NOT KG triple. Never substitute latent embedding for raw position in anti-cheat validation.

**Spatial cognition (two-brain model):** info brain = real `MapPos` (synced, ground truth). Think brain = per-NPC `SpatialBelief` (zone-level KG triple + stale last_known_pos, fog-of-war gated, NOT synced). Bridge is one-way: real position → belief update only when within `visible_radius`. Confidence decay: `sigmoid(-λ * (current_tick - last_observed_tick))`. Two brains MUST exist independently — divergence is emergent behavior, not a bug.

## Cross-references (read on demand)

- `katgpt-rs/.contexts/optimization.md` — perf rules (zero-alloc, SIMD, rayon, caching)
- `katgpt-rs/.contexts/ibraheemdev-papaya-v0.2.3-examples.md` — papaya lock-free hashmap usage
- `katgpt-rs/.research/003_Commercial_Open_Source_Strategy_Verdict.md` — 3-repo strategy source of truth
- `katgpt-rs/.research/004_LoRA_Architecture_Verdict.md` — LoRA / validator terminology
- `katgpt-rs/.research/005_Artifact_Definition.md` — artifact terminology
- `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md` — canonical research-note example
- `katgpt-rs/.plans/271_attention_matching_compaction.md` — canonical plan example

## TL;DR

This skill packages the katgpt-rs research workflow: **MANDATORY pre-flight: `read_file` all three READMEs (`katgpt-rs/README.md`, `riir-ai/README.md`, `riir-armageddon/README.md`) AND `list_directory` both `.research/` folders (`katgpt-rs/.research/`, `riir-ai/.research/`) before any verdict** → read paper → classify (training? → riir-train, stop) → **distill + fuse** (find the transferable primitive, then grep BOTH layers — `.research/`+`.plans/` for intent AND `src/`+`crates/` (`katgpt-rs/src/`, `katgpt-rs/crates/`, `riir-ai/crates/`, `riir-armageddon/crates/`) for shipped code — across all repos, for the 2–3 closest cousins to synthesize a novel combination) → **novelty gate** (Super-GOAT? → open primitive + private riir-ai guide; else GOAT/Gain → plan only) → implement behind feature flag → benchmark → promote GOAT or demote loser. Hard constraints: modelless-first, latent-to-latent with sigmoid (never softmax), freeze/thaw over fine-tuning, 3-repo commercial discipline, raw scalars at the sync boundary, **fusion-first mindset** (the best Super-GOATs come from fusing papers across BOTH repos, not direct-mapping one). **Super-GOAT = private moat; never skip the riir-ai guide. Never grep only katgpt-rs — riir-ai is half the corpus. Never grep only notes — code is half the prior art (HLA's `evolve_hla` shipped without a research note and caused a false Super-GOAT on Research 242).**
