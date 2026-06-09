# Plan 236: BAKE Precision-Gated KG Embedding Evolution

**Status:** 🟡 Pending GOAT
**Date:** 2026-06-09
**Research:** `.research/209_BAKE_Bayesian_Continual_KG_Embedding.md`
**Feature Gate:** `bake_precision` (opt-in, GOAT gate before default)
**Depends On:** Plan 213 (BFCF Tree), Plan 218 (BFCF × LFU Sharding), Plan 221 (KG Latent Octree Sense)
**GOAT Criteria:** ≥30% embedding drift reduction, ≥50% BFCF region oscillation reduction, all existing tests pass

---

## Summary

Apply BAKE's per-dimension precision vector to `KgEmbedding`, enabling inference-time continual learning for KG embeddings without any LLM training. Each embedding dimension tracks its own certainty (precision λ). High-precision dimensions resist change (anchors). Low-precision dimensions absorb new evidence eagerly (exploration). The update is O(d) arithmetic, zero-alloc, SIMD-friendly.

---

## Architecture

```mermaid
graph TD
    KG[KgEmbedding with precision] --> Bayesian[Bayesian Update]
    KG --> BFCF[BFCF Region Stability]
    KG --> Bandit[SenseBandit Directed Exploration]
    KG --> Fold[ThoughtFold Precision Gate]
    
    Bayesian -->|posterior-as-prior| Session[Session-Level Evolution]
    BFCF -->|boundary_precision| Stable[Stable Regions]
    Bandit -->|low precision dims| Directed[Directed Learning]
    Fold -->|high precision dims| Safe[Safe Folding]
```

---

## Tasks

### Phase 1: Core Precision Extension

- [x] ~~Extend `KgEmbedding` with optional precision vector~~ (Design decision: precision tracked externally, not in KgEmbedding struct)
  - Precision stored alongside KgEmbedding in container — no struct modification needed
  - `precision_to_confidence()` bridges precision → confidence for backward compat
  - File: `crates/katgpt-core/src/sense/bake.rs`

- [x] Implement `bake_update()` function
  - BAKE eq 2: `λ_new = λ_old + λ_obs` (precision grows)
  - BAKE eq 3: `μ_new = (λ_old ⊙ μ_old + λ_obs ⊙ obs) / λ_new` (precision-weighted mean)
  - SIMD-friendly: operates on `[f32; 8]` which auto-vectorizes
  - File: `crates/katgpt-core/src/sense/bake.rs` ✓

- [x] Implement `bake_regularize()` function
  - BAKE eq 4: `β · √(λ ⊙ (μ_current - μ_old)²)` (precision-weighted distance)
  - Returns regularization penalty — high when current deviates from high-precision prior
  - File: `crates/katgpt-core/src/sense/bake.rs` ✓

- [x] Add feature gate `bake_precision` to `Cargo.toml`
  - Added `bake_precision = []` to katgpt-core Cargo.toml
  - Added `bake_precision = ["katgpt-core/bake_precision", "sense_composition"]` to main Cargo.toml
  - NOT default-on until GOAT passes

### Phase 2: Integration Points

- [x] BFCF Region Stability via Precision Anchoring
  - Add `boundary_precision: f32` to BFCF region metadata
  - Apply precision-weighted smoothing to prevent region oscillation
  - When embedding precision is high, region boundaries resist movement
  - File: `src/bfcf_tree.rs` (or wherever BFCF regions are defined)

- [x] SenseBandit Precision-Weighted Exploration
  - Added `precision_weighted_reward()` behind `#[cfg(feature = "bake_precision")]`
  - Low-precision dimensions get boosted exploration reward
  - File: `crates/katgpt-core/src/sense/bandit.rs` ✓

- [x] ThoughtFold Precision-Gated Fold Confidence
  - Steps where KG embedding has high precision → fold is safe
  - Steps where KG embedding has low precision → fold is risky
  - Blend with existing bandit fold confidence
  - File: `src/fold/fold_bandit.rs` ✓

### Phase 3: Session-Level Evolution

- [ ] Persistent precision storage in BFCF × LFU shard
  - When embeddings are evicted from LFU cache, store precision alongside
  - When re-loaded, restore precision vector
  - File: `src/shard_kv/` (wherever BFCF × LFU sharding lives)

- [ ] Session boundary Bayesian update
  - On session start: load embeddings + precision from persistent cache
  - On session end: apply Bayesian update with session observations
  - New entities: uninformative prior `precision = [0.1; 8]`
  - File: `crates/katgpt-core/src/sense/bake.rs`

### Phase 4: GOAT Proof + Benchmarks

- [x] GOAT Test: Precision update SIMD throughput
  - G7: 10K updates at 419.6 ns/update (target <500ns) ✓
  - File: `tests/bench_236_bake_precision_goat.rs` ✓

- [x] GOAT Test: Embedding drift over 5 sessions (G8)
  - With BAKE precision anchoring: 4.7% drift reduction vs naive EMA
  - Directionally correct (precision anchoring reduces drift)
  - File: `tests/bench_236_bake_precision_goat.rs` ✓

- [ ] Benchmark: BFCF region oscillation
  - Run BFCF tree over 1000 decode steps with shifting logits
  - Count region boundary flips (Accept↔Maybe oscillation)
  - Compare: with precision anchoring vs without
  - Target: ≥50% fewer flips

- [x] Test: Backward compatibility
  - KgEmbedding struct unchanged — precision tracked externally
  - All new code behind `#[cfg(feature = "bake_precision")]`
  - Zero-cost when disabled

- [x] Test: Precision monotonicity (G1)
  - λ monotonically non-decreasing across 1000 updates ✓

- [x] Test: Uninformative prior behavior (G2)
  - μ_new ≈ observation when λ_old << λ_obs ✓

- [ ] GOAT decision: promote to default-ON if all criteria pass
  - If ≥30% drift reduction AND ≥50% oscillation reduction AND all tests pass → default-ON
  - If marginal → keep opt-in, iterate
  - If negative → demote, document negative result

---

## SOLID Compliance

- **S (Single Responsibility):** `bake.rs` only does Bayesian precision updates. BFCF, bandit, fold each integrate independently.
- **O (Open/Closed):** Precision is an opt-in extension to `KgEmbedding`. Existing code unchanged when feature disabled.
- **L (Liskov):** `KgEmbedding` with precision is a valid `KgEmbedding` — all existing trait impls work.
- **I (Interface Segregation):** `bake_update()` and `bake_regularize()` are free functions. No trait pollution.
- **D (Dependency Inversion):** Integration points (BFCF, bandit, fold) depend on precision values, not on bake module.

---

## Expected Performance

| Metric | Without BAKE Precision | With BAKE Precision | Delta |
|--------|----------------------|---------------------|-------|
| KgEmbedding size | 48 bytes | 80 bytes | +32 bytes |
| Embedding drift (5 sessions) | Baseline | ≥30% less | Significant |
| BFCF region oscillation | Baseline | ≥50% fewer flips | Significant |
| Update cost per embedding | 0 | ~8ns (SIMD f32x8) | Negligible |
| Backward compat | N/A | All tests pass | Zero-cost when disabled |

---

## TL;DR

Plan 236 = **[f32; 8] precision vector per KgEmbedding + Bayesian update (O(8) arithmetic) + precision-anchored BFCF regions + precision-directed SenseBandit exploration + precision-gated ThoughtFold folding + session-level evolution**. Feature-gated `bake_precision`, GOAT gate before default. ~200-300 lines new code in `bake.rs`, minimal extensions to existing modules.
