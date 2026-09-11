# Research 245: Latent Spatial Memory for Video World Models

> **Source:** [Latent Spatial Memory for Video World Models (Mirage)](https://arxiv.org/abs/2606.09828) — Wang, Zhao, Yang, Chen, Zhang, He, Duan, Chen, Yang, Zhuang (Zhejiang / Microsoft Research / Adelaide / Monash), 2026-06-08
> **Date:** 2026-06-16
> **Status:** Done
> **Related Research:** 133 (FluxMem), 192 (NextLat belief), 196 (KG-Latent-Octree), 216 (MRAgent reconstructive memory), 242 (recurrent belief — HLA prior-art lesson), 060 (MeMo), 024 (δ-Mem)
> **Classification:** Public
> **Cross-ref (riir-ai):** `riir-ai/crates/riir-games-civ/src/civ/spatial_cognition.rs::SpatialMemory` (shipped prior art), `riir-ai/crates/riir-games-shared/src/game_traits/spatial.rs::GenericSpatialBelief`, `riir-ai/crates/riir-engine/src/ns_csg.rs::SpatialBelief`
> **PASS-Redirects (synthesis):** Huang et al. [arXiv:2609.10540 "Programmable World Model"] — same class (interactive video world model; adopts AlayaWorld's geometry-aligned spatial memory, Mirage's cousin). Its measured thesis — explicit engine-maintained state decoupled from rendering beats implicit generative state (94% Count Acc / 98% State Acc vs 32–58% for implicit-state baselines) — is external validation of the architecture this stack natively ships (authoritative sim → zone sync/AOI reveal → view; `induced_cwm` for the program half); the signature mechanism (OBB → camera-projected pixel-aligned conditioning maps) has no non-video analog here, same as Mirage's pinhole z-buffer readout.

---

## TL;DR

Mirage introduces a **3D spatial cache** for video world models that stores `{(p_i, f_i)}` pairs (world point `p_i ∈ R^3`, VAE latent feature `f_i ∈ R^C`, C=48) and is **queried by pinhole-camera projection + z-buffer at latent resolution** — never decoding to pixels in the conditioning loop. Result: 10.57× faster end-to-end generation, 55× less GPU memory than RGB point-cloud baselines. The paper is **explicitly latent-to-latent** in the read path.

**Verdict: PASS.** A video-diffusion-specific perf/quality technique. Its one transferable principle ("keep spatial memory readout in latent space, never bridge through raw in the hot path") **is already shipped** as `riir-games::SpatialMemory` (fog-of-war-gated belief cache, zone-attention routing via dot-product + sigmoid, KG-triple emission) and `katgpt-core::SenseOctreeBuilder` (KG-latent octree). The paper's signature mechanism (pinhole projection + z-buffer onto a target camera grid) does **not** map to our 2D top-down arena (no perspective camera, no depth buffer). No separable, modelless, transferable primitive remains that isn't already covered. Training is video-gen LoRA/ControlNet → riir-train territory. **No plan, no primitive, no guide.**

**Distilled for katgpt-rs (modelless):** nothing actionable. The "latent-resolved spatial memory" pattern is already the design of `SpatialMemory` + `SenseOctreeBuilder`. The "don't decode in the read loop" rule is already an enforced invariant of the think-brain bridge (raw→latent one-way, sigmoid-bounded, zero-alloc).

---

## 1. Paper Core Findings

### 1.1 The mechanism — latent spatial memory

A persistent 3D cache `M = {(p_i, f_i)}`, where each entry pairs a **world-space point** `p_i` with a **latent feature token** `f_i ∈ R^C` drawn from the video diffusion VAE (the backbone's native input space).

- **Construct (Eq. 4):** encode frame → VAE latent `z`; for each latent cell `(u,v)`, back-project via pinhole inverse using metric depth: `p_uv = π⁻¹(u, v, D(u,v); K, E)`, `F_uv = z[:, v, u]`. One memory element per latent cell.
- **Readout (Eq. 5):** for target camera `(E_t, K_t)`, project all `p_i` onto the latent grid, z-buffer to keep the frontmost per cell, retrieve its latent token. Output: target-view latent tensor `ẑ_t` + binary visibility mask `m_t` (which cells got a point). Zero-fill empty cells. **No pixel-space round trip.**
- **Update (Eq. 6):** after each chunk, re-encode decoded frames, back-project, union into `M` — **excluding dynamic objects and sky** (SAM3 + Qwen3-VL masks) to keep the persistent cache rigid/static only.

The readout is injected into the diffusion backbone via a ControlNet-style side branch — no bridging encoder needed because the readout already lives in the backbone's latent space.

### 1.2 Results (Wan2.2-TI2V-5B backbone, H100)

- **WorldScore Average 70.36** — SOTA, beating Spatia (69.73) and all foundation video generators.
- **RealEstate10K closed-loop** PSNR_C/SSIM_C best — the closed-loop (camera leaves and returns) is where latent memory's geometric anchoring shows; RGB caches drift.
- **10.57× faster** end-to-end, **55× less** 3D-cache GPU memory vs RGB point-cloud baselines. Gap widens with rollout horizon because the per-step rasterise+re-encode cost grows with cache size in the RGB path but the latent read stays O(N log N + hw).
- Ablations: swapping latent cache → RGB cache hurts 3D + photometric consistency (latent tokens carry semantic/texture info 3 color channels cannot); disabling dynamic-object filter hurts long-horizon stability most.

### 1.3 What the training actually is (→ riir-train if we cared)

32× A100, 2-stage flow-matching: (1) freeze backbone+VAE, train ControlNet side branch only; (2) unlock rank-64 LoRA on {q,k,v,o} of every self-attn, joint with side branch. Depth source is robust (DepthAnything3 default; MapAnything/UniDepth within ~1 pt). This is video-gen LoRA/ControlNet training — out of scope for katgpt-rs/riir-ai.

---

## 2. Distillation

### 2.1 The single transferable principle

> **"Keep spatial memory readout in latent space; never bridge through raw (pixel/token) in the hot read path. Bridge (raw→latent) happens once per write, not per read."**

This is the paper's real idea, stripped of pinhole cameras, z-buffers, VAE latents, and video diffusion. Everything else (10.57× speedup, 55× memory) is downstream of this single invariant + the specific 3D-camera projection math.

### 2.2 That principle is already our design — prior-art check (the decisive part)

The mandatory two-layer novelty check (notes + shipped code, per the Research 242 lesson) found the principle already implemented in **three** places:

| Paper concept | Shipped prior art | Match |
|---|---|---|
| Spatial cache of `(location, latent_feature)` | `riir-ai/crates/riir-games-civ/src/civ/spatial_cognition.rs::SpatialMemory` — `ArrayVec<SpatialBelief, MAX_BELIEFS>`, each belief = `(target_id, believed_zone, last_known_pos, last_observed_tick, confidence, is_threat)`; location-keyed, latent-valued | ✅ Cache exists |
| Fog-of-war-gated write (only observe what's visible) | `SpatialMemory::update_from_observation` — AABB early-reject + circular fog-of-war gate | ✅ Identical pattern |
| Latent readout, no raw bridge in hot path | `SpatialMemory::most_attractive_zone` — `score = sigmoid(dot(preference, zone_embed)) + belief_boost`; `emit_kg_triples` for high-conf beliefs. **No raw decode in the read path.** | ✅ Latent-to-latent readout already enforced |
| Confidence decay / cache pruning | `SpatialMemory::tick_decay` + `is_expired` swap-remove + LRU eviction when full | ✅ (paper has no decay; we are richer here) |
| Dynamic-object exclusion on write | `SpatialMemory` handles transience differently: confidence decay + prune (effect achieves the same goal — stale/transient content doesn't persist) | ✅ Equivalent goal, different mechanism |
| Spatially-indexed latent content (generic) | `crates/katgpt-sense/src/octree.rs::SenseOctreeBuilder` — KG embeddings → octree bit-planes + `TernaryDir` directions, Merkle-committed, MUX-latent bridge in `crates/katgpt-core/src/mux_latent/octree_bridge.rs` | ✅ Spatially-indexed latent store shipped |

Plus **four** `SpatialBelief`/`GenericSpatialBelief<T>` structs shipped (`ns_csg.rs:238`, `spatial_cognition.rs:87`, `riir-ai/crates/riir-games/src/crowd_mcgs/types.rs:333`, `riir-ai/crates/riir-games-shared/src/game_traits/spatial.rs:68`), all implementing the AGENTS.md two-brain model (info brain raw/synced, think brain latent/local, one-way bridge, `sigmoid(-λΔt)` confidence decay).

### 2.3 Why the paper's signature mechanism doesn't transfer to us

The paper's **readout** is `project 3D world points → target camera grid → z-buffer → retrieve latent token`. This is fundamentally a **3D perspective-camera** operation:

- We are a **2D top-down** arena (`ArenaPos {x, y}`, `MapPos {x, y}`). There is no perspective camera, no depth buffer, no latent-resolution projection grid.
- Our fog-of-war is a **2D visibility region** (AABB + circular radius), not a 3D frustum + z-buffer.
- Our "latent feature" per belief is a small fixed set of scalars/zone-embeddings, not a 48-dim VAE latent of a video diffusion model.

So the paper's *specific* contribution — the projection-based latent-resolution readout that beats RGB point clouds — has **no analog** in our 2D modelless game stack. We don't have an RGB-point-cloud baseline to beat; we never had a pixel-space detour in the think-brain read path to eliminate.

### 2.4 What's NOT here (training-only / not transferable)

- The VAE latent space (C=48, Wan2.2) — tied to the video diffusion backbone.
- The 3D pinhole projection + z-buffer readout — tied to perspective camera.
- The 2-stage LoRA + ControlNet training — → riir-train.
- The depth estimator (DepthAnything3) and dynamic segmenter (SAM3) — external CV models, not game-AI primitives.

---

## 3. Verdict

**Tier:** **PASS** — not relevant as a separable modelless/latent primitive; the transferable principle is already shipped; the signature mechanism doesn't map to our 2D stack.

**One-line reasoning:** Mirage's one transferable idea ("latent-space spatial memory readout, no raw bridge in the hot path") is already the enforced design of `riir-games::SpatialMemory` + `katgpt-core::SenseOctreeBuilder`, and the paper's actual contribution (pinhole z-buffer latent projection) is a 3D-video-diffusion technique with no 2D-top-down game-AI analog.

**Novelty gate (honest, post prior-art check — applying the Research 242 `evolve_hla` lesson):**

| Gate | Question | Honest answer |
|---|---|---|
| **Q1 Novelty** | No prior art in shipped code? | **FAILS.** `SpatialMemory` (spatial_cognition.rs:172) is direct prior art: fog-of-war-gated belief cache, dot-product+sigmoid zone-attention readout, KG emission, decay/prune/evict. `SenseOctreeBuilder` is prior art for spatially-indexed latent content. |
| **Q2 New capability class** | New behavior, not better numbers? | **FAILS.** Paper is a perf/quality win for *video diffusion*. Our "think-brain latent spatial memory" capability already exists. The delta (visibility mask vs single last_known_pos) is incremental, not a new class — and is already on the Research 242 / Plan 276 attractor-belief roadmap. |
| **Q3 Selling point** | "Our NPCs do X no competitor can"? | **FAILS.** Video-gen perf technique, not a game-AI selling point. |
| **Q4 Force multiplier** | Connects to ≥2 pillars? | Partially (two-brain, zone attention) — but Q4 alone ≠ Super-GOAT. |

**Not Super-GOAT (3/4 fail) → not GOAT (no provable gain over shipped `SpatialMemory`) → not Gain (no separable incremental feature worth a flag). → PASS.** Per skill Pass protocol: research note only, no plan/primitive/guide. (The note is created per the user's explicit request; the skill's default "no files" for Pass is overridden by the task instruction to always produce the note and stop at non-Super-GOAT.)

**Routing note:** the paper's *training* (video-gen LoRA + ControlNet, 2-stage flow-matching) → riir-train if ever wanted. Not distilling further here.

---

## 4. Fusion ideas worth a follow-up (NOT pursued — recorded to prevent re-litigation)

These were considered and **rejected** because the shipped prior art already covers the game-AI analog. Recording them so a future agent doesn't re-derive the same "fusion":

1. **"Latent spatial memory → think brain" (user's hypothesis):** **Already implemented.** `SpatialMemory` *is* the think brain's latent spatial belief store, with fog-of-war write gate, sigmoid zone-attention readout, and KG emission. The user's hypothesis ("richer than stale last_known_pos scalar") is valid as a *future direction* but is addressed by Research 242 / Plan 276 (attractor belief states, per-NPC kernel versioning) — not by this video-gen paper.
2. **Visibility-mask ("unseen" vs "observed-but-zero"):** Our fog-of-war already produces a binary visible/not-visible split; "observed-but-zero" isn't a meaningful state for *entity* beliefs (an entity is observed-or-not). Could matter for *terrain* cochains but those are already raw topology + latent cochain features (see riir-armageddon README §10 DEC×TNO bridge).
3. **Dynamic-object exclusion on write:** `SpatialMemory` already achieves the same goal (transient content doesn't persist) via confidence decay + prune — different mechanism, equivalent effect.

If a future agent wants the *richer* think-brain spatial representation the user's hypothesis imagines, the right starting point is **Research 242 + Plan 276** (attractor recurrent belief states fusing with `SpatialMemory`), not this paper.

---

## 5. Closest cousins (for the fusion protocol record)

- `katgpt-rs/.research/133_FluxMem_Connectivity_Evolving_Memory.md` — connectivity-evolving memory (also PASS — LLM-agent-level, overlaps Four-Tier).
- `katgpt-rs/.research/192_NextLat_Belief_State_Latent_Dynamics.md` — latent belief dynamics (GOAT, belief-state drafter).
- `katgpt-rs/.research/196_KG_Latent_Octree_WASM_Composition.md` — spatially-indexed latent store (shipped).
- `katgpt-rs/.research/216_MRAgent_Reconstructive_Memory_Graph.md` — reconstructive memory over octree (GOAT).
- `katgpt-rs/.research/242_Topological_State_Tracking_Recurrent_Belief.md` — the prior-art-check cautionary tale (Super-GOAT→GOAT after `evolve_hla` check); same lesson applied here.
- `riir-ai/.research/007_Four_Tier_Memory_Architecture.md` — four-tier memory.
- **Shipped code (the decisive prior art):** `riir-ai/crates/riir-games-civ/src/civ/spatial_cognition.rs::SpatialMemory`, `katgpt-rs/crates/katgpt-sense/src/octree.rs::SenseOctreeBuilder`.

---

## TL;DR

Mirage's latent spatial memory is a clean, well-engineered **3D video-diffusion** technique (store VAE latents at world points, query by pinhole+z-buffer at latent resolution, 10.57× faster / 55× less memory than RGB point clouds). Its single transferable principle — "keep spatial-memory readout in latent space, bridge raw→latent only on write, never in the hot read path" — **is already the enforced design** of our shipped `riir-games::SpatialMemory` (fog-of-war-gated belief cache, dot-product+sigmoid zone-attention, KG emission, decay/prune) and `katgpt-core::SenseOctreeBuilder` (KG-latent octree). The paper's signature mechanism (3D perspective projection + z-buffer) has no 2D-top-down game-AI analog; we never had a pixel-space detour to eliminate. Prior-art check (notes + code, per the Research 242 lesson) blocks all four novelty-gate questions on Q1 and the capability-class delta. **Verdict: PASS.** Note only, no plan/primitive/guide. If the user's "richer think-brain spatial memory" goal is pursued, the right starting point is Research 242 / Plan 276 (attractor belief states), not this paper.
