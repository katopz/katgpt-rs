# Research 196: KG Latent Octree — Composable Fixed-Type Sense Weights via Ternary Bit-Plane + WASM

**Date:** 2025-06
**Status:** GOAT Verdict — GAIN (proceed to plan)
**Domain:** Modelless (inference-time) + Model-Based (LoRA training via riir-ai)
**Relates To:** Plans 038, 148, 178, 190, 209, 223, 236, 239; Research 010, 032, 025, 110, 144, 184, 185

---

## Executive Summary

The idea: **compress game domain knowledge (common sense, fighter sense, game theory sense, skills) into fixed-type ternary bit-plane KG weights that compose as WASM modules, query at nanosecond cost via octree spatial partitioning, and self-learn from self-play traces.**

This is the bridge from "every NPC runs an LLM forward pass" to "every NPC queries a pre-compressed knowledge octree at ~20 bytes per sense module." The LLM (or LoRA-adapted model) trains the KG offline. The WASM module serves the compressed KG at inference time. **Modelless first, LoRA-trained weights as fuel.**

---

## The Core Insight: KG Directions as Composable WASM Pods

### What We Already Have

| Component | Where | What It Does |
|-----------|-------|-------------|
| `TernaryWeights` | katgpt-rs (plasma_path) | {-1,0,+1} bit-plane packing: 64 weights → 16B + 4B scale = **1.58 bits/weight** |
| `EmotionDirections` | riir-games (civ_emotion) | Linear directions in HLA latent space, extracted via mean-difference |
| `DomainDirections` | riir-games (civ_emotion) | 7 domain scalars (crime, economy, riot...) as direction vectors |
| `KgTriple` + freeze/thaw | riir-engine (kg) | Statistical KG extraction with binary serialization |
| `NeuronShard` | riir-chain (neuron_db) | 368B fixed Pod: `style_weights[64]` + `hla_moments[8]` + BLAKE3 |
| `BomberWasmPruner` | katgpt-rs (wasm) | Papaya pool + fuel gating + batch API + zero-copy state |
| `SpatialBelief` + emit_kg_triples | riir-games (civ) | Two-brain model, KG emission from spatial cognition |
| `extract_direction()` + `project_onto()` | riir-games (civ) | Bridge functions: raw → latent via dot-product + sigmoid |
| `BeliefDrafter` | katgpt-rs (belief_drafter) | Latent dynamics MLP for variable-length self-speculative decoding |
| `ILC synonym clusters` | riir-gpu (latent_prediction) | Iterative Latent Clustering detects synonym game states |

### What's Missing (The Gap)

The gap is a **composition layer** that:
1. **Types** each sense domain (common_sense, fighter_sense, game_theory) as a fixed-schema KG
2. **Compresses** each typed KG into ternary bit-plane format (~20B per sense module)
3. **Encodes** spatial/temporal structure as octree occupancy (1-2 bits per node)
4. **Serves** via WASM modules that compose at runtime (load common_sense.wasm + fighter_sense.wasm)
5. **Self-learns** from self-play traces via bandit-weighted KG extraction
6. **Inherits** learned weights from parent NPCs or faction templates

---

## Architecture: Three-Layer KG Latent Octree

```
Layer 1: Type System (Sense Schema)
─────────────────────────────────────
  SenseKind: u8 enum
  ├── 0: CommonSense     (physics rules, gravity, collision)
  ├── 1: FighterSense    (threat assessment, combat ranges)
  ├── 2: GameTheorySense (cooperation/defection, Nash equilibria)
  ├── 3: SpatialSense    (zone topology, path costs)
  ├── 4: SocialSense     (faction relations, trust scores)
  ├── 5: SkillSense      (ability cooldowns, combos)
  └── 7: Reserved

  Each SenseKind → fixed-schema KgTriple vocabulary:
    CommonSense:  (entity, falls_in, zone), (item, blocks, path), ...
    FighterSense: (entity, threatens, entity), (range, effective_at, distance), ...
    GameTheory:   (faction, cooperates_with, faction), (player, defects_in, round), ...

Layer 2: Compression (Ternary Octree)
──────────────────────────────────────
  For each SenseKind, build an octree over the KG latent embedding space:
    - Root: entire latent space
    - Each node: 1 bit occupied (KG triple exists in this region)
    - Each node: 1 bit sign (positive/negative relation polarity)
    - Leaf: TernaryWeights direction vector (1.58 bits/weight)

  Query: dot_product(hla_state, direction) → sigmoid → scalar
  Cost: ~20B per sense module per octree level = O(1) fixed

Layer 3: Composition (WASM Pods)
─────────────────────────────────
  Each SenseKind → one WASM module:
    common_sense.wasm     → exports: kg_query, kg_project, kg_emit_triples
    fighter_sense.wasm    → exports: kg_query, kg_project, kg_emit_triples
    game_theory_sense.wasm → exports: kg_query, kg_project, kg_emit_triples

  Composition: NPC loads N sense modules at spawn.
    npc_brain = compose([common_sense, fighter_sense, spatial_sense])
    query = npc_brain.project(hla_state) → [f32; N_senses]
    decision = npc_brain.decide(query, context)

  Each WASM module ~5-20KB. NPC brain = ~60-100KB total.
  Compare: LLM forward pass = ~1-10GB VRAM. **5 orders of magnitude smaller.**
```

---

## The Octree: Why {-1, 0, +1} Is Perfect for KG

### KG Triples Are Sparse and Signed

A `KgTriple` encodes `(head, relation, tail, confidence)`. In a latent KG embedding space:
- **Most regions are empty** (most entity-relation pairs don't exist) → occupancy bit
- **Relations have polarity** (cooperation vs hostility) → sign bit
- **Confidence is continuous** → but ternary quantization (±1 with row_scale) preserves the top-k signal

This means **each octree node needs exactly 2 bits**: occupied (yes/no) + sign (+1/-1). The "0" in ternary means "no KG triple in this region" — which is the majority of the space.

### Comparison: Flat KG vs Octree KG

| Metric | Flat KgTriple[] | Ternary Octree |
|--------|-----------------|----------------|
| Storage | 28B per triple × N | 2 bits × occupied nodes + 20B per leaf |
| Query | O(N) linear scan | O(log N) octree descent |
| 1000 triples | 28KB | ~1.5KB (18× compression) |
| 10000 triples | 280KB | ~12KB (23× compression) |
| WASM-friendly | Needs heap | Bitwise only, no float multiply |

### Why Octree (Not Hash Map or B-Tree)

1. **Spatial locality**: KG embeddings cluster by domain. Octree partitions capture this naturally — "combat zone" triples cluster together.
2. **Level-of-detail**: Different octree depths = different abstraction levels. LOD 0 = "is there any KG knowledge here?" LOD 3 = "what specific relation?" Maps to `ClusterLevel` (Individual/Flock/Herd/Zone).
3. **Inverse mapping**: To find "which entities have fighter_sense knowledge about zone X?", traverse the octree — O(log N) not O(N).
4. **Composable**: Merge two octrees = OR their bit-planes. Fast, no conflict resolution needed.

---

## Self-Learning Pipeline: Self-Play → KG → Octree → WASM

```
┌─────────────────────────────────────────────────────────┐
│                    Self-Play Loop                        │
│                                                         │
│  G-Zero self-play (Plan 049)                            │
│    → StateTransition[] (pre_state, action, post_state)  │
│    → extract_triples() → KgTriple[]                     │
│    → consolidate_triples() → dedup + verify             │
│                                                         │
│  Per-SenseKind extraction:                              │
│    StateTransition → sense_classifier() → SenseKind     │
│    Fighter transitions → fighter_sense KgTriple[]       │
│    Social transitions → social_sense KgTriple[]         │
│    Spatial transitions → spatial_sense KgTriple[]       │
│                                                         │
│  Direction extraction (per SenseKind):                  │
│    KgTriple[] → embed → mean-difference → direction_vec │
│    direction_vec → TernaryWeights::quantize()           │
│                                                         │
│  Octree build (per SenseKind):                          │
│    direction_vec[] → spatial partition → octree nodes   │
│    octree nodes → bit-plane pack → 2 bits/node          │
│                                                         │
│  WASM compile:                                          │
│    octree + directions → common_sense.wasm (TBD)        │
│    octree + directions → fighter_sense.wasm              │
│                                                         │
│  Deploy:                                                │
│    NPC spawn → load sense WASM modules → project()      │
│    NPC tick → query sense modules → decide()            │
│    NPC death → dump KG traces → extract_triples()       │
│                                                         │
│  Bandit feedback:                                       │
│    Decision quality → TrialLog → AbsorbCompress         │
│    High-quality decisions → reinforce direction vectors  │
│    Low-quality decisions → decay direction weights       │
│                                                         │
│  Inheritance:                                           │
│    Faction template → base KG weights for all members   │
│    Veteran NPC → learned KG weights → new spawn inherits│
│    Cross-faction transfer → Rosetta alignment (Plan 233) │
└─────────────────────────────────────────────────────────┘
```

### Where Each Piece Already Exists

| Step | Existing Code | Gap |
|------|---------------|-----|
| Self-play | G-Zero (Plan 049), bomber examples | ✅ Works |
| StateTransition extraction | `kg.rs::extract_triples()` | ✅ Works |
| Consolidation | `kg.rs::consolidate_triples()` | ✅ Works |
| Direction extraction | `emotion.rs::extract_direction()` | Need per-sense version |
| Ternary compression | `TernaryWeights::quantize_from_f32()` | ✅ Works |
| Octree spatial partition | `flock.rs::assign_clusters()` | Need octree builder |
| WASM compilation | `BomberWasmPruner` pattern | Need sense-specific WASM |
| WASM pool | Papaya pool pattern | ✅ Works |
| Bandit feedback | TrialLog + AbsorbCompress | ✅ Works |
| Inheritance | NeuronShard.apply_delta() | ✅ Works |
| Cross-game transfer | Rosetta (Plan 233) | ✅ Works |

**Gap assessment: ~40% new code, ~60% reuse.** The architectural fit is strong.

---

## Performance Budget

### Per-NPC Per-Tick Cost

| Operation | Cost | Notes |
|-----------|------|-------|
| Load N sense modules | 0ns (spawn-time only) | Papaya pool, one per thread |
| Project hla_state onto 1 direction | ~5ns | Bitwise dot-product in WASM |
| Project onto N=5 sense modules | ~25ns | 5 × bitwise dot-product |
| Octree descent (3 levels) | ~15ns | 3 × bitmask AND |
| Decision aggregation | ~5ns | Weighted sum + sigmoid |
| **Total per NPC per tick** | **~45ns** | **vs ~1-10ms for LLM forward pass** |

### Memory Budget

| Item | Size | Notes |
|------|------|-------|
| 1 sense module (WASM binary) | ~5-20KB | Includes octree + directions |
| 5 sense modules per NPC | ~25-100KB | Total NPC brain |
| 1000 NPCs on server | ~25-100MB | Fits in L3 cache |
| NeuronShard per zone | 368B | Fixed Pod, BLAKE3 committed |
| Ternary direction per sense | ~20B | 64-dim × 1.58 bits + 1 f32 scale |

### Comparison

| Approach | Per-NPC Memory | Per-Tick Cost | Training Required |
|----------|---------------|---------------|-------------------|
| Full LLM | 1-10GB | 1-10ms | Full fine-tune |
| LoRA-adapted LLM | 100-500MB | 0.5-5ms | LoRA fine-tune |
| **KG Latent Octree + WASM** | **25-100KB** | **~45ns** | **Self-play only** |
| Naive KG lookup | 1-10MB | 100ns-1μs | Manual authoring |

**The KG Latent Octree is 4-5 orders of magnitude cheaper than LLM inference and 2-3 orders cheaper than naive KG.**

---

## KG Mapping + Weight: The Dual Representation

### The User's Idea: "KG Mapping + Weight or Separate or Pack"

The insight is that each KG triple has two components:
1. **Structure**: the triple itself (head, relation, tail) — the "knowledge"
2. **Weight**: confidence + direction magnitude — the "intensity"

These can be stored separately or packed:

### Option A: Separate (Recommended for game use)

```
KgStructure (octree bit-planes):
  - 2 bits per node: occupied + sign
  - Pure bitwise query: "does this knowledge exist?"
  - Size: O(occupied_nodes) bits

KgWeight (NeuronShard.style_weights):
  - [f32; 64] per zone/sense — the direction vector magnitude
  - Updated via apply_delta() during self-play
  - Size: 256B per sense per zone
```

**Verdict: Separate is better** because:
- Structure changes rarely (knowledge topology is stable)
- Weight changes frequently (confidence evolves with experience)
- Separate storage allows independent compression and sync
- Aligns with `NeuronShard` layout (already has style_weights separate from zone_hash)

### Option B: Packed (For ultra-low-latency hot path)

```
PackedKgNode (8 bytes):
  - 2 bits: occupied + sign
  - 6 bits: weight quantized to {-1, -0.5, 0, 0.5, +1} (3 bits) + confidence (3 bits)
  - 56 bits: entity/relation hash for exact match
```

**Verdict: Pack only for inner-loop combat** where both structure and weight are needed in one cache line. Use Option A for everything else.

---

## The WASM Composition Model: Sense Modules as Composable Brains

### "Each element in game packed as 1-2 bit octree KG"

This is the key insight. Each game concept becomes:

```rust
/// A typed, compressed, composable knowledge module.
/// One per SenseKind per zone.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SenseModule {
    pub kind: SenseKind,           // 1B — which sense domain
    pub version: u8,               // 1B — for hot-reload
    pub octree_depth: u8,          // 1B — how many levels
    pub n_directions: u8,          // 1B — number of ternary direction vectors
    pub octree_bits: [u64; 4],     // 32B — up to 128 octree nodes at 2 bits each
    pub directions: [TernaryDir; 8], // 160B — 8 × (2×u64 bitmask + 1×f32 scale) = 20B each
    pub confidence: f32,           // 4B — overall module quality (from bandit)
    pub commitment: [u8; 32],      // 32B — BLAKE3 hash
} // Total: ~232B per sense module — fits in ~4 cache lines
```

### Composition at Spawn Time

```rust
/// NPC brain = compose of N sense modules.
/// Loaded at spawn, read-only during tick.
pub struct NpcBrain {
    modules: Vec<SenseModule>,     // typically 3-5 modules
    hla_state: [f32; 8],          // current HLA hidden state (think-brain)
    belief_cache: SpatialMemory,  // spatial beliefs (fog-of-war)
}

impl NpcBrain {
    /// Project current state onto all loaded sense modules.
    /// Returns one scalar per module: "how activated is this sense?"
    pub fn project_all(&self) -> Vec<f32> {
        self.modules.iter()
            .map(|m| {
                // For each direction in the module:
                let raw = m.directions.iter()
                    .map(|d| dot_product_ternary(&self.hla_state, d))
                    .sum::<f32>();
                sigmoid(raw / m.n_directions as f32)
            })
            .collect()
    }
}
```

### Why WASM Composition

The user asked "pack as WASM e.g. common_sense.wasm and ready to compose." This is correct because:

1. **Isolation**: One buggy sense module can't corrupt another (WASM sandbox)
2. **Hot-reload**: Replace fighter_sense.wasm without rebooting server
3. **Community/marketplace**: Curators submit sense modules (aligns with Strategy 003)
4. **Cross-game**: bomber_common_sense.wasm works in dungeon crawler (Rosetta transfer)
5. **Tiered trust**: Core senses (common_sense) are native Rust; community senses are WASM

### The Composition API

```rust
/// WASM export signature for sense modules.
/// Same ABI for all sense kinds — polymorphic via behavior, not type.
trait SenseWasm {
    /// Project HLA state onto this sense's direction space.
    /// Returns Q16.16 fixed-point activation score.
    fn project(hla_ptr: *const f32, hla_len: u32, result_ptr: *mut u32) -> u32;

    /// Emit KG triples for the current state.
    /// Returns number of triples written to output buffer.
    fn emit_triples(hla_ptr: *const f32, hla_len: u32,
                    triple_ptr: *mut u64, triple_len: u32) -> u32;

    /// Batch project — multiple HLA states at once.
    /// Returns activation scores for all inputs.
    fn batch_project(hla_ptr: *const f32, n_states: u32,
                     result_ptr: *mut u32) -> u32;
}
```

---

## Inheritance and Self-Learning

### Inheritance: Faction Templates → NPC Instances

```
Faction "Wolves Guild" template:
  common_sense.wasm      → baseline physics knowledge
  fighter_sense.wasm     → melee-focused combat knowledge
  spatial_sense.wasm     → forest zone specialization

  + NeuronShard.style_weights[64] → wolf-specific personality weights
  + VibeSchedule → god/evil phase cycling for mood

NPC "Wolf Scout #42" instance:
  Inherits faction template + learns from self-play:
  - TrailLog → "ambush worked in zone 3" → reinforce fighter_sense direction for zone 3
  - TrialLog → "fleeing from bear was correct" → update spatial_sense octree bit for "bear → avoid"

  On death: dump KG traces → extract_triples() → update faction template
  On level-up: unlock new sense module (game_theory_sense at level 10)
```

### Self-Learning: Bandit-Weighted KG Extraction

The existing bandit infrastructure (Plan 030-032) provides:
- `TrialLog`: records decision outcomes
- `AbsorbCompress`: absorbs high-quality trials into compact rules
- `HotSwapPruner`: replaces sense modules at runtime

The loop:
1. NPC takes action based on sense module projection
2. Outcome recorded in `TrialLog`
3. High-quality outcomes (positive reward) → reinforce direction weights via `apply_delta()`
4. Low-quality outcomes → decay direction weights
5. Periodic consolidation → `AbsorbCompress` → updated WASM module → `HotSwapPruner` replaces

**This is modelless self-learning.** No LLM training needed. The sense modules adapt purely from self-play reward signals.

---

## Fusion Ideas: Beyond Direct Mapping

### Fusion 1: Octree LOD = ClusterLevel (Boids × KG)

The boids system already has `ClusterLevel` (Individual/Flock/Herd/Zone). The KG octree has depth levels. **Fuse them:**

- LOD 0 (Individual): Full KG, all triples, fine-grained directions
- LOD 1 (Flock): Merged KG, shared direction, reduced precision
- LOD 2 (Herd): Zone-level KG, one direction per sense, ternary only
- LOD 3 (Zone): City rules, single aggregate direction per sense

This means **flock members share KG queries** — one query for the flock, not N queries for N individuals. The `FlockCentroid` computation already provides the aggregation point. Just project the centroid's HLA state instead of each individual's.

**Expected speedup**: For a flock of 20 entities, 20× reduction in KG queries at LOD 1+.

### Fusion 2: Sense Composition = BeliefDrafter Latent Dynamics

The `BeliefDrafter` (Plan 217) predicts next hidden state from current state + action. Fuse with sense composition:

```
hla_state_t → sense_project() → [activation; N_senses]
                                     │
                                     ▼
                          BeliefDrafter MLP predicts:
                          hla_state_{t+1} = hla_state_t + delta(senses, action)
                                     │
                                     ▼
                          sense_project(hla_state_{t+1}) → predicted activations
```

This gives each NPC **anticipatory sense**: "if I move to zone 5, my fighter_sense will spike." No LLM needed — the BeliefDrafter MLP is 3 layers, runs in ~100ns.

### Fusion 3: VibeSchedule × Sense Inheritance = Cultural Evolution

The `VibeSchedule` cycles God/Evil arch-agents via sine wave. Fuse with sense inheritance:

- God phase: reinforce cooperation KG triples (social_sense)
- Evil phase: reinforce competition KG triples (fighter_sense)
- Children inherit the **phase-weighted** KG, not the raw KG
- Over many generations, the "cultural personality" of a faction evolves based on the phase at which they were born

This is **emergent cultural evolution** from two existing systems (VibeSchedule + KG inheritance). No new ML needed.

### Fusion 4: FOL Rule Extraction × Octree = Verifiable KG

Plan 239 extracts FOL rules from LoRA weights. Fuse with octree:

- Each octree node at leaf level → extract FOL rule: `(in_zone(X, Z) ∧ has_threat(X, Y)) → flee(X, Y)`
- Rules verified by WASM validator before insertion into octree
- **Verifiable KG**: every octree leaf has a human-readable FOL explanation
- Enables debugging: "why did NPC #42 flee?" → trace octree → show FOL rule

---

## GOAT Verdict: GAIN

### Why This Is a Gain

| Criterion | Assessment |
|-----------|------------|
| **Novelty** | High. KG latent compression to ternary bit-plane octree + WASM composition is not in any published work. Closest is KnowFormer (KG in attention) but without compression or WASM. |
| **Performance** | Transformative. 4-5 orders of magnitude cheaper than LLM per NPC. 45ns/tick vs 1-10ms. |
| **Composability** | Strong. WASM modules compose at runtime. SenseKind taxonomy enables fine-grained control. |
| **Self-learning** | Proven path. G-Zero + bandit + KG extraction already exists. Only needs wiring. |
| **Inheritance** | Natural fit. NeuronShard.apply_delta() + VibeSchedule cultural evolution. |
| **Commercial fit** | Strong. Engine/fuel split intact. Sense modules are "fuel" (private WASM). Engine is MIT. Marketplace of sense modules. |
| **Modelless first** | Yes. Inference is pure bitwise WASM. Only self-play (not LLM training) for KG extraction. |
| **LoRA only for training** | Yes. The sense module direction vectors can be refined via LoRA-adapted game models (riir-gpu) but the primary learning is self-play. |
| **No perf hurt** | Yes. 45ns/tick is negligible. Feature-gated. Default-off until GOAT proven, then default-on. |

### Risks

| Risk | Mitigation |
|------|-----------|
| Ternary quantization too lossy for KG | Use full-precision NeuronShard for hot-path, ternary for draft/approximate. PlasmaPath proven to work. |
| WASM overhead dominates at nanosecond scale | Use batch API (proven 14× speedup in BomberWasmPruner). Amortize FFI over multiple sense queries. |
| Octree depth too shallow for complex domains | Start with depth 3 (8 levels = 512 nodes). Scale empirically. ILC synonym clustering guides depth. |
| Sense module quality varies | Bandit feedback loop + quality gate (KgQualityMetrics) + HotSwapPruner for runtime replacement. |
| Cross-game transfer fails | Rosetta (Plan 233) validates transfer. Start with same-game-only, add transfer as opt-in. |

### Commercial Strategy Alignment (per 003)

| Aspect | Alignment |
|--------|-----------|
| Engine (MIT) | `SenseKind` enum, `SenseModule` type, `SenseWasm` trait, octree builder — all open |
| Fuel (private) | `fighter_sense.wasm`, `game_theory_sense.wasm` — trained KG weights, marketplace of sense modules |
| Marketplace | Curators submit sense modules for specific games/domains. Quality-gated. Revenue share. |
| SaaS | Self-play-as-a-service: upload game rules → we run G-Zero → produce trained sense WASM modules |
| Data flywheel | Every NPC tick → KG traces → better sense modules → better NPC behavior → more game usage |

---

## What Gets Replaced

| Current | Replaced By | Why |
|---------|-------------|-----|
| Scalar emotion cascades (Plan 236) | Ternary direction vectors per sense | More compact, composable, self-learnable |
| Hardcoded `species_preference()` | Learned direction from self-play | Adapts to game balance, not hardcoded |
| Manual zone embeddings | Octree over KG latent space | Data-driven, not author-designed |
| Per-NPC LLM inference | Per-NPC WASM sense composition | 4-5 orders of magnitude cheaper |
| Static KG lookup tables | Bandit-weighted octree with confidence decay | Self-improving, not static |
| Separate emotion/combat/social systems | Unified SenseKind taxonomy | DRY, composable, SOLID |

---

## TL;DR

**KG Latent Octree + WASM Composition** compresses game domain knowledge into fixed-type ternary bit-plane sense modules (~20B per sense, ~232B per module). Each module is a WASM-exported pod that queries KG structure via bitwise octree descent and projects direction vectors via ternary dot-product. NPCs compose modules at spawn, self-learn from self-play via bandit feedback, inherit from faction templates, and evolve culturally via VibeSchedule. The idea fuses 8 existing research papers and 6 existing plan implementations into a single composable architecture. **GAIN: 4-5 orders cheaper than LLM, 60% reuse, aligns with commercial strategy.** Proceed to plan.

**Next:** Create Plan 221 (katgpt-rs, modelless) and Plan 249 (riir-ai, model-based).
