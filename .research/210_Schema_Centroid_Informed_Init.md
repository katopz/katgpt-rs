# Research 210: Schema-Centroid Informed Initialization for Continual KG Embeddings

**Date:** 2026-06-09
**Source:** arXiv:2511.11118 — "Improving Continual Learning of Knowledge Graph Embeddings via Informed Initialization" (Pons, Bilalli, Queralt, 2025)
**Verdict:** GAIN — High priority for katgpt-rs (modelless)
**Target:** Modelless (katgpt-rs primary)
**Relates To:** Research 209 (BAKE), 208 (SLoD), 196 (KG Latent Octree), 207 (ManifoldE)
**Feature Gate:** `schema_centroid` (opt-in, GOAT gate)

---

## Executive Summary

The paper's core insight is that **where you start matters more than how you update** for continual KG embedding. Random initialization of new entity embeddings causes 2-3× more training epochs AND catastrophic forgetting of existing embeddings. Schema-based centroid initialization places new entities at the average of their class centroids (from existing embeddings), cutting convergence time by 2-3× and improving knowledge retention by 20-30%. This is pure arithmetic — O(d) centroid computation, no gradients, model-agnostic.

**Our fusion opportunity:** Apply schema-based centroid initialization to our `KgEmbedding` system. Currently, BAKE (Plan 236) initializes new entities with "uninformative priors" (random means, small precisions). This paper gives us the **informative prior** — initialize new entity embeddings at their schema class centroid with precision proportional to class density. The fusion:

1. **BAKE + Schema Centroid** (Primary): New entities get `μ = class_centroid + γ·σ_class·r` instead of `μ = random`. BAKE's precision converges faster (starts closer to optimal), reducing evidence observations needed before precision stabilizes from ~N epochs to ~N/3 epochs.
2. **SLoD + Schema Centroid** (Secondary): Instead of a single centroid per class, use SLoD's multi-scale Fréchet centroids. New entities initialized at the most appropriate abstraction level for their class.
3. **SenseModule + Schema Centroid** (Tertiary): When composing NpcBrain at spawn time, initialize SenseModule direction vectors from schema-class centroids rather than random ternary weights.

---

## Paper Core Ideas

### 1. The Initialization Problem in Continual KG
- Standard practice: new entities get random embeddings
- Problem: random init is far from optimal → needs 2-3× more epochs to converge
- Worse: random init causes gradient interference that corrupts existing embeddings (catastrophic forgetting)
- Key insight: entities of the same schema class (e.g., `type=NPC`, `type=Weapon`) cluster in embedding space
- Therefore: initialize new entities at their class centroid, not randomly

### 2. Schema-Based Centroid Initialization
- Centroid per class: `v_c = (1/|E_c|) Σ_{e∈E_c} e` — average embedding of entities in class c
- Standard deviation per class: `σ_c = std(embeddings of entities in class c)`
- New entity init: `ê = (1/|C_e|) Σ_{c∈C_e} (v_c + γ·σ_c ⊙ r_c)` — average centroid + stochastic perturbation
- `γ` controls exploration vs exploitation in initialization
- Multi-class entities: average the centroids of all classes they belong to
- The perturbation `r_c` is random noise scaled by class variance — prevents all same-class entities from starting identical

### 3. Retention and Acquisition Metrics
- **Retention** `Ω_base = (1/N) Σ_j α_{0,j} / α_{0,0}` — how well old knowledge is preserved after learning new
- **Acquisition** `Ω_new = (1/N) Σ_i α_{i,i}` — how well new knowledge is learned
- Schema init improves BOTH: +20-30% retention AND +65% acquisition on small increments
- This is rare — most methods trade retention for acquisition or vice versa

### 4. Results
- Schema init beats random: +30% H@3, +20% Ωbase, +65% Ωnew on small increments
- Converges 2.16-2.67× faster than random init
- Model-agnostic: works with TransE, TransH, RotatE, DistMult, HolE, ProjE
- Works with all continual learning methods: FT, EWC, EMR, LKGE, incDE
- Smaller increments benefit more (directly relevant to our incremental game state updates)
- Works across all benchmark datasets (FB15k-237, WN18RR, YAGO, etc.)

---

## Distillation: Modelless (katgpt-rs)

### What's Training-Time Only (not for us)
- The KGE scoring functions (TransE margin loss, etc.)
- The incremental training loop
- The hyperparameter tuning for γ and learning rates

### What's Inference-Time (our territory)
- **Class centroid computation** — O(d·|E_c|) per class, pre-computed, cached
- **Standard deviation per class** — captures intra-class variation
- **New entity initialization** — weighted average of class centroids + stochastic perturbation
- **Centroid maintenance** — update centroids when embeddings evolve (via BAKE precision updates)
- **Retention metric computation** — quality signal for GOAT gate
- All pure arithmetic, no training, no LLM, no gradients

---

## Paper's Key Equations (Distilled for Modelless)

```
Centroid:    v_c = (1/|E_c|) Σ_{e∈E_c} e
Std dev:     σ_c = std(embeddings of entities in class c)
New entity:  ê = (1/|C_e|) Σ_{c∈C_e} (v_c + γ·σ_c ⊙ r_c)
  where:     C_e = set of classes entity e belongs to
             r_c = random noise vector (standard normal)
             γ = perturbation scale (paper uses γ ∈ [0.1, 0.5])

Retention:   Ω_base = (1/N) Σ_j α_{0,j} / α_{0,0}
Acquisition: Ω_new = (1/N) Σ_i α_{i,i}
```

---

## Fusion Ideas: Creative, Not Direct Mapping

### Fusion 1: BAKE + Schema Centroid (Primary — katgpt-rs)

**Core Insight:** BAKE (Research 209) initializes new entities with uninformative priors: `μ = random, precision = [0.1; 8]`. Schema centroid gives us an **informative prior**: `μ = class_centroid + γ·σ_class·noise, precision = f(class_density)`.

**Current State (BAKE):**
```rust
pub struct KgEmbedding {
    pub entity_hash: u64,
    pub relation_hash: u64,
    pub embedding: [f32; 8],
    pub sign: bool,
    pub confidence: f32,
    #[cfg(feature = "bake_precision")]
    pub precision: [f32; 8],   // uninformative: [0.1; 8] for new entities
}
```

**Proposed Extension:**
```rust
#[cfg(feature = "schema_centroid")]
pub struct SchemaCentroid {
    pub class_hash: u64,           // BLAKE3 hash of class type
    pub centroid: [f32; 8],        // v_c = mean of class embeddings
    pub std_dev: [f32; 8],         // σ_c = per-dimension std dev
    pub entity_count: u32,         // |E_c| — class density
    pub blake3_hash: [u8; 32],     // integrity check
}

// Initialization becomes:
fn init_new_entity(classes: &[u64], centroid_cache: &HashMap<u64, SchemaCentroid>) -> [f32; 8] {
    let mut sum = [0.0f32; 8];
    let mut count = 0u32;
    for &class_hash in classes {
        if let Some(centroid) = centroid_cache.get(&class_hash) {
            // ê += v_c + γ·σ_c ⊙ r_c
            for d in 0..8 {
                let noise: f32 = random_standard_normal(); // or use xorshift
                sum[d] += centroid.centroid[d] + GAMMA * centroid.std_dev[d] * noise;
            }
            count += 1;
        }
    }
    if count > 0 {
        for d in 0..8 { sum[d] /= count as f32; }
    } else {
        // Fallback: random init (no schema info available)
        for d in 0..8 { sum[d] = random_f32(); }
    }
    sum
}
```

**Impact on BAKE precision convergence:**
- Without schema centroid: precision starts at [0.1; 8], needs ~N epochs to stabilize
- With schema centroid: precision starts at [0.1; 8] BUT embedding is already near optimal
- Effective convergence: ~N/3 epochs (matching paper's 2.67× speedup)
- Precision initialization can also be upgraded: `precision_init = [class_density_factor; 8]` for high-density classes

### Fusion 2: SLoD + Schema Centroid (Secondary — katgpt-rs)

**Core Insight:** Instead of a single centroid per class, use SLoD's (Research 208) multi-scale Fréchet centroids. New entities initialized at the most appropriate abstraction level.

**Mechanism:**
- SLoD maintains Fréchet means at multiple scales (zone-level, region-level, world-level)
- Schema centroids computed at each scale: `v_c^{zone}, v_c^{region}, v_c^{world}`
- New entity picks the scale with highest class density: `argmax_k |E_c^k|`
- This handles sparse classes better — a rare class might have 0 entities at zone level but 50 at world level

### Fusion 3: SenseModule + Schema Centroid (Tertiary — katgpt-rs)

**Core Insight:** When composing NpcBrain at spawn time, initialize SenseModule direction vectors from schema-class centroids rather than random ternary weights.

**Mechanism:**
- NPC schema classes: `{Warrior, Mage, Healer, Merchant, Animal}`
- Each class has a "behavioral centroid" — the average SenseModule direction vector of existing NPCs of that class
- New NPC spawns → direction vectors initialized from class behavioral centroid
- This means new NPCs start with "appropriate" sense priorities (warriors sense threats, merchants sense trade opportunities)
- Falls back to random ternary if no class centroid exists (first NPC of a new class)

### Fusion 4: BFCF Region Boundary Seeding (Quaternary — katgpt-rs)

**Core Insight:** BFCF regions currently use geometric boundaries. Schema centroids could seed region boundaries by class type.

**Mechanism:**
- Entities of the same class tend to cluster → their centroid is a natural region center
- BFCF Accept/Maybe/Reject boundaries can be initialized from class centroids + σ
- Reduces warm-up time for new BFCF trees (no need for initial random exploration)
- Composes with BAKE precision anchoring (Research 209, Fusion 2)

---

## What We Already Have

| Component | Location | Relevance |
|-----------|----------|-----------|
| `KgEmbedding` struct with `embedding: [f32; 8]` | katgpt-rs core | Direct target — schema centroid initializes this |
| `SenseModule` with `TernaryDir` direction vectors | katgpt-rs NPC brain | Secondary target — behavioral centroid init |
| BAKE precision vectors (Plan 236, Research 209) | katgpt-rs modelless | Schema centroid upgrades BAKE's uninformative prior |
| BFCF regions | katgpt-rs search | Schema centroids could seed region boundaries |
| `papaya` lock-free HashMap | katgpt-rs concurrency | Natural fit for centroid cache — high read, low write |
| SIMD `f32x8` | katgpt-rs perf | Centroid computation = sum + divide = SIMD auto-vectorizes |
| SLoD multi-scale Fréchet means (Research 208) | katgpt-rs modelless | Multi-scale schema centroids |
| ManifoldE point-to-manifold (Research 207) | katgpt-rs modelless | Schema centroids as manifold anchors |

---

## GOAT Gate

| Criteria | Target | Measurement |
|----------|--------|-------------|
| Centroid computation SIMD auto-vectorizes | ≥95% theoretical peak f32x8 | Benchmark vs scalar loop |
| Zero-cost when feature disabled | 0 bytes added, 0 instructions | `cargo bench` compare with/without feature |
| Convergence speedup on new entities | ≥2× fewer observations to reach stable embedding | Entity init benchmark |
| Knowledge retention improvement | ≥20% higher Ωbase vs random init | Continual update benchmark |
| Backward compatibility | All existing tests pass | `cargo test` |
| Centroid cache lookup latency | ≤50ns per lookup (papaya read) | Microbenchmark |

---

## Prior Art in Our Stack

| Component | Status | Gap Schema Centroid Fills |
|-----------|--------|--------------------------|
| BAKE `precision: [f32; 8]` | Uninformative priors [0.1; 8] | Informative μ prior from class centroid |
| `KgEmbedding.embedding` | Random init for new entities | Schema-informed init |
| BFCF region boundaries | Geometric initialization | Class-centroid seeded boundaries |
| SenseModule direction vectors | Random ternary init | Behavioral centroid init |
| Entity spawn | No schema awareness | Schema-class-aware initialization |

---

## Implementation Sketch

```
Phase 1: SchemaCentroid cache (feature-gated)
  - Add SchemaCentroid struct
  - Compute centroids from existing KgEmbeddings (batch job at startup)
  - Store in papaya HashMap<u64, SchemaCentroid> keyed by class_hash
  - BLAKE3 integrity hash on centroid data

Phase 2: Entity initialization
  - Replace random init with schema centroid init
  - γ = 0.3 default (paper's sweet spot for small increments)
  - Fallback to random if no centroid exists for entity's class

Phase 3: BAKE precision integration
  - New entity precision initialized proportional to class density
  - precision_init[d] = min(1.0, |E_c| / MIN_DENSITY) for each dimension
  - High-density classes → high initial precision → fast convergence

Phase 4: Centroid maintenance
  - When BAKE updates an embedding, also update the class centroid
  - Centroid update: momentum-based `v_c = (1-η)v_c + η·e_new`
  - Low overhead: O(d) per update, batch-amortized
```

---

## Verdict

**GAIN for katgpt-rs.** Schema-based centroid initialization is the missing piece for BAKE's uninformative prior problem. Instead of random initialization + slow precision convergence, we get schema-informed initialization + 2-3× faster convergence. The entire mechanism is pure arithmetic — O(d) per entity, pre-computed centroids cached in papaya HashMap, SIMD-friendly. It's model-agnostic (the paper proves this across 6 KGE models) and method-agnostic (works with all 5 continual learning methods tested).

**Confidence: High.** The paper's results are consistent across all models, datasets, and continual learning methods. The mechanism is trivially implementable — it's a weighted average plus noise. The risk is near-zero: worst case, γ is wrong and we fall back to random init.

**Priority: High.** This directly upgrades BAKE (Research 209), which is already high priority. Schema centroid is a prerequisite for optimal BAKE performance — without it, BAKE wastes 2-3× epochs on convergence that could be avoided.

---

## TL;DR

Schema-based centroid initialization places new KG entities at their class average embedding + noise instead of randomly. Cuts convergence 2-3×, improves retention 20-30%, improves acquisition 65%. Pure O(d) arithmetic, model-agnostic. Apply to BAKE precision init (primary), SLoD multi-scale centroids (secondary), SenseModule direction vectors (tertiary). Feature-gated `schema_centroid`, GOAT gate before default.
