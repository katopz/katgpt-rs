# NPC Sense Composition (Plans 221/230/235/236/237)

## Overview
KG Latent Octree NPC sense modules — compresses game domain KG triples into fixed-type ternary bit-plane sense modules. NPCs compose modules at spawn time and query at ~45ns/tick via bitwise dot-product. The system is inspired by the "Two-Brain Model" from the project's AGENTS.md rules — an info brain (ground truth, synced) and a think brain (subjective model, per-NPC).

## Architecture

### Core Components

1. **SenseModule** — Fixed-size ternary bit-plane sense module with octree bits, direction vectors, confidence score, and BLAKE3 commitment. Projected via `SenseModule::project()` which performs ternary bitwise dot → sigmoid → 8-dim HLA projection at ~45ns/tick.

2. **NpcBrain** — Composes `Vec<SenseModule>` per NPC, projects HLA state, respects GM overrides. Supports autonomous and scripted modes.

3. **SenseKind** — Classification: CommonSense, FighterSense, GameTheorySense, SpatialSense, SocialSense, SkillSense, Reserved.

4. **SenseOctreeBuilder** — Converts `KgEmbedding` array into bit-plane octree occupancy mask + ternary direction vectors.

5. **SenseHotSwap** — Lock-free atomic module replacement via `AtomicPtr` with `AtomicBool` module lock. Zero downtime during module updates.

6. **SenseTrialLog** — Bandit feedback log for module quality. `decay_direction()` EMA adjusts confidence over time.

7. **SenseBatch** — Parallel batch projection for multiple NPCs using rayon (activates when N>64).

8. **SNSE Serialization** — Binary format with BLAKE3 verification for persistent sense state.

### GM Override System

- **SenseOverride** — Pins specific senses or disables autonomous mode for scripted NPCs
- **GM Actions**: `pin_sense`, `disable_autonomous`, `inject_kg`, `dump_brain`
- **Override dispatch**: Internal API for game master control over NPC behavior

### Key Traits

```rust
impl SenseModule {
    pub fn project(&self, hla_state: &[f32; 8]) -> [f32; 8]; // ~45ns
    pub fn confidence(&self) -> f32;
    pub fn kind(&self) -> SenseKind;
}
```

## Sub-Systems

### Shard Embedding (Plan 230)
Johnson-Lindenstrauss random orthogonal projection: `style_weights: [f32; 64]` → `ShardEmbedding: [f32; 8]`. Gram-Schmidt orthogonal rows, SIMD dot-product projection, BLAKE3 commitment. No training, no data — modelless dimension reduction.

- `JlProjectionMatrix` — 64×8 random orthogonal matrix with BLAKE3 hash
- `ShardEmbedding` — `[f32; 8]` with `cosine_similarity()`, `dist_sq()`, BLAKE3 hash
- Always compiled (no feature gate)

### SLoD Spectral Level-of-Detail Pruner (Plan 235)
Modelless KG resolution control via spectral heat diffusion on hyperbolic kNN graph Laplacians. Default-ON, GOAT G1–G6 all pass.

Architecture:
1. Poincaré ball geometry: `poincare_distance()`, `log_map()`, `exp_map()`, `frechet_mean()`
2. kNN Laplacian from KG embeddings
3. Jacobi eigendecomposition
4. Multi-signal boundary scan (participation + diffusion entropy + spectral concentration)
5. MAD peak picker → tier routing in `SlodPruner::is_valid()`

Key types: `SlodConfig`, `SlodOperator`, `SlodPruner` (implements `ConstraintPruner`)
Feature gate: `slod` (default-ON, depends on `spectral_hierarchy`)

### Schema Centroid (Plan 237)
Per-class embedding centroids for informed KG entity initialization. Default-ON, GOAT 7/7.

- `CentroidStats { mean: [f32; 8], std_dev: [f32; 8] }` computed once per class
- `SchemaCentroidCache` — papaya lock-free HashMap storage
- `schema_init_entity()` — average class centroids + `γ·σ_c ⊙ noise` perturbation
- Falls back to random `[-0.5, 0.5]` init if class not found
- Cross-feature bridge: when `bake_precision` enabled → `schema_init_with_precision()` uses informed prior
Feature gate: `schema_centroid` (default-ON, requires `dep:papaya`)

### BAKE Precision-Gated Bayesian Embedding (Plan 236)
Per-dimension precision tracking for KG embeddings. O(8) arithmetic, zero-alloc. GOAT 10/10 but **demoted to opt-in** (drift 4.7% vs 30% target).

- Bayesian update: `λ_new = λ_old + λ_obs`, `μ_new = (λ_old ⊙ μ_old + λ_obs ⊙ obs) / λ_new`
- Regularization penalty: `β · √(λ ⊙ (μ_current - μ_old)²)`
- Session lifecycle: `BakeSession::begin()` → `observe()` × N → `end()` writes back to store
- Key types: `PrecisionEntry`, `BakePrecisionStore`, `BakeSession`
Feature gate: `bake_precision` (opt-in, requires `dep:papaya`, `sense_composition`)

## Two-Brain Model Integration

Per the project's AGENTS.md latent/raw space rules:

- **Info brain**: Real `MapPos`, synced via SyncBlock, used for physics/combat/anti-cheat. Never latent.
- **Think brain**: Per-NPC `SpatialBelief` (zone-level KG triple + stale last_known_pos), NOT synced, gated by fog-of-war.
- **Bridge**: Real position → belief update ONLY when target is within NPC's `visible_radius`. One-way gate — think brain cannot influence info brain.
- **Confidence decay**: `sigmoid(-λ * (current_tick - last_observed_tick))`. Stale beliefs fade, not deleted.
- **Zone attention**: dot-product(NPC preference vector, zone embedding) → sigmoid → "how much does this NPC care about this zone".

## Performance

| Operation | Time |
|-----------|------|
| SenseModule::project() | ~45ns/tick |
| SenseHotSwap swap | Atomic (lock-free) |
| Batch projection (N>64) | Parallel via rayon |
| SNSE serialization | BLAKE3 verified |
| ShardEmbedding cosine | O(8) SIMD |
| SLoD tier routing | O(1) per query |

## Feature Gates Summary

| Feature | Default | Dependencies |
|---------|---------|-------------|
| `sense_composition` | Opt-in | `plasma_path`, `domain_latent` |
| `slod` | **Default-ON** | `spectral_hierarchy` |
| `schema_centroid` | **Default-ON** | `dep:papaya` |
| `bake_precision` | Opt-in (demoted) | `dep:papaya`, `sense_composition` |
| (shard_embedding) | Always-on | None |
| `rat_plus_bridge` | Opt-in | None |

## References

- Plan 221: Sense Composition
- Plan 230: Shard Embedding Projection
- Plan 235: SLoD Spectral Level-of-Detail
- Plan 236: BAKE Precision-Gated Embeddings
- Plan 237: Schema Centroid KG Embedding Init
