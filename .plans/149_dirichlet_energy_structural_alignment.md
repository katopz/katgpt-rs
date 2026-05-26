# Plan 149: Dirichlet Energy Structural Alignment Diagnostic

**Date:** 2026-05-26
**Status:** ✅ COMPLETE
**Research:** 111 (Emergent Analogical Reasoning in Transformers)
**Related:** Research 051 (Deep Manifold), Research 039 (SpectralQuant), Plan 065 (AutoGo), Plan 104 (MLS), `27_mmo_goat_pillars_decision_matrix.md` (Pillar 1)
**Feature Gate:** `dirichlet_energy` (opt-in, katgpt-rs open)

---

## Task Index

- [x] T1: `dirichlet_energy()` function — SIMD Dirichlet Energy computation
- [x] T2: Adjacency construction helpers — functor + game state adjacency
- [x] T3: Dirichlet Energy probe for KV cache — katgpt-rs `data_probe/`
- [x] T4: Generic integration test — synthetic structural alignment

## Goal

Add Dirichlet Energy computation as a generic embedding diagnostic in katgpt-rs. This is the core measurable from Research 111 — it quantifies whether embeddings are **structurally aligned** across entities/positions, which is a prerequisite for analogical reasoning.

This is a **diagnostic tool**, not a reasoning engine. The value is:
1. Measure KV cache embedding alignment quality
2. Detect whether LoRA training produces structural alignment (via external probe)
3. Provide early-exit signal for analogical queries (future)

---

## Tasks

### T1: `dirichlet_energy()` function — katgpt-core `simd.rs`

Add a SIMD-accelerated Dirichlet Energy computation:

```rust
/// Compute Dirichlet Energy over embeddings w.r.t. adjacency graph.
///
/// E(E) = Σ_{i,j} A_{ij} ‖h_{e_i} - h_{e_j}‖²
///
/// Lower energy = more structurally aligned (entities connected by edges
/// have similar embeddings).
///
/// # Arguments
/// * `embeddings` — flat slice of embeddings, shape [n_entities × dim]
/// * `dim` — embedding dimension
/// * `adjacency` — sparse adjacency pairs [(i, j), ...] where A_{ij} = 1
///
/// # Returns
/// Total Dirichlet Energy (f32).
#[cfg(feature = "dirichlet_energy")]
pub fn dirichlet_energy(
    embeddings: &[f32],
    dim: usize,
    adjacency: &[(usize, usize)],
) -> f32;
```

**GOAT proof:** Unit test with known aligned/unaligned embeddings. Aligned (identical pairs) → E ≈ 0. Random → E > 0. Synthetic degradation (add noise) → E increases monotonically.

**Estimated LOC:** ~40 (SIMD path) + ~30 (scalar fallback) + ~60 (tests)

### T2: Adjacency construction helpers — katgpt-core `types.rs`

```rust
/// Build functor adjacency from paired entity indices.
///
/// For N pairs (a_i, b_i), creates edges: (a_0, b_0), (a_1, b_1), ...
/// This is the paper's A_{ij} = 1 iff entities i,j are related by functor.
#[cfg(feature = "dirichlet_energy")]
pub fn functor_adjacency(pairs: &[(usize, usize)]) -> Vec<(usize, usize)>;

/// Build position-neighbor adjacency from game state transitions.
///
/// For a game where position p can reach position q via some action,
/// creates edges (p, q). This is the structural graph for game domains.
#[cfg(feature = "dirichlet_energy")]
pub fn game_state_adjacency<S: GameState>(
    states: &[S],
    player_id: u8,
) -> Vec<(usize, usize)>;
```

**Estimated LOC:** ~50 + ~30 (tests)

### T3: Dirichlet Energy probe for KV cache — katgpt-rs `data_probe/`

Extend the existing `data_probe` module (which already has Dirichlet distribution sampling) with:

```rust
/// Probe KV cache key embeddings for structural alignment.
///
/// Computes Dirichlet Energy over KV cache keys at a given layer,
/// using position-adjacency (consecutive positions, or user-specified pairs).
/// Returns (energy, normalized_energy) where normalized = energy / n_edges.
#[cfg(feature = "dirichlet_energy")]
pub fn kv_cache_dirichlet_energy(
    keys: &[f32],       // [n_positions × kv_dim]
    kv_dim: usize,
    adjacency: &[(usize, usize)],
) -> (f32, f32);
```

**GOAT proof:** Benchmark on random vs. learned embeddings. Random baseline energy should be high; after LoRA training, energy should decrease (if structural alignment emerges).

**Estimated LOC:** ~30 + ~40 (tests/bench)

### T4: Generic integration test — synthetic structural alignment

A test that:
1. Generates synthetic embeddings for two "categories" of entities
2. Defines functor pairs (structurally matched entities across categories)
3. Computes Dirichlet Energy for unaligned (random) vs aligned (shifted by constant offset) embeddings
4. Verifies: aligned embeddings have **lower** energy than random

This validates the Dirichlet Energy diagnostic itself — no game-specific or Fourier-specific code.
Game-specific alignment tests (Fourier-encoded game embeddings) belong in riir-ai Plan 146.

**Location:** `katgpt-rs/tests/dirichlet_energy_alignment.rs`
**Feature gate:** `#[cfg(all(feature = "dirichlet_energy", test))]`
**Estimated LOC:** ~80

---

## What Stays in riir-ai (Private)

The following are **NOT** part of this plan. They go in riir-ai because they're game-specific or Super-GOAT:

| What | Why Private | riir-ai Plan |
|------|-------------|--------------|
| LoRA weight decay tuning for analogy | Game-specific training config | New plan: `146_analogy_lora_training.md` |
| Cross-game functor direction extraction | Super-GOAT: detect Bomber↔FFT transfer | New plan: `146_analogy_lora_training.md` |
| Self-play 3-stage dynamics monitoring | Game-specific training diagnostic | Add to existing Plan 052 |
| NPC quest analogy generation | Game-specific product feature | Add to existing Plan 099 |

### riir-ai Plan Reference: `146_analogy_lora_training.md`

**Scope:** Use `dirichlet_energy` from katgpt-rs to:
1. Probe LoRA embeddings during wgpu training for structural alignment
2. Tune weight decay to 0.01–0.1 (paper's sweet spot)
3. Extract functor directions between game domains (Bomber → Go → FFT)
4. If functor directions emerge: this is Super-GOAT, keep secret

**GOAT proof:** Dirichlet Energy of LoRA embeddings decreases during training (3-stage dynamics observable). Cross-game functor cosine similarity > 0.5.

**Pillar reference:** Strengthens Pillar 1 (Fourier) and Pillar 3 (NPC Dialog). If analogy works, it validates the "dense relational graphs → easier analogy" hypothesis for games.

---

## Feature Gate

```toml
# katgpt-rs/Cargo.toml
[features]
dirichlet_energy = []  # Dirichlet Energy structural alignment diagnostic (Research 111)
```

- **Default:** Off (diagnostic tool, not production path)
- **Opt-in:** Researchers and GOAT proofs enable it
- **Zero cost when off:** All code behind `#[cfg(feature = "dirichlet_energy")]`

---

## GOAT Proofs Required

| # | Proof | Threshold | Pass? |
|---|-------|-----------|-------|
| G1 | `dirichlet_energy()` on identical embeddings | E < 0.01 | ⏳ |
| G2 | `dirichlet_energy()` on random embeddings | E > 1.0 (for dim=128, 10 entities) | ⏳ |
| G3 | Energy increases monotonically with Gaussian noise | dE/dσ > 0 | ⏳ |
| G4 | SIMD path matches scalar path (bit-exact) | diff < 1e-6 | ⏳ |
| G5 | Synthetic aligned embeddings < random embeddings | E_aligned < 0.5 × E_random | ⏳ |
| G6 | KV cache probe: random keys baseline | E_random > threshold | ⏳ |

---

## Module Structure

```
katgpt-rs/
├── crates/katgpt-core/
│   └── src/
│       ├── simd.rs              # + dirichlet_energy() SIMD (T1)
│       └── types.rs             # + adjacency helpers (T2)
├── src/
│   └── data_probe/
│       └── dirichlet_energy.rs  # + KV cache probe (T3)
└── tests/
    └── dirichlet_energy_alignment.rs  # Generic integration test (T4)

riir-ai/  (Fourier + game-specific analogy — private)
├── .research/
│   └── 001_Fourier_Spatial_Search.md  # Fourier is riir-ai domain
└── .plans/
    └── 146_analogy_lora_training.md  # Private Super-GOAT plan (Fourier + analogy)
```

---

## Estimated Effort

| Task | Hours | Depends On |
|------|-------|------------|
| T1: SIMD dirichlet_energy | 2h | None |
| T2: Adjacency helpers | 1h | None |
| T3: KV cache probe | 1h | T1 |
| T4: Integration test | 2h | T1, T2 |
| GOAT proofs | 1h | All |
| **Total** | **~7h** | |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SIMD portability (NEON vs AVX2) | Low | Low | Scalar fallback exists |
| Dirichlet Energy not useful for KV cache | Medium | Low | It's a diagnostic, not a dependency |
| Fourier embeddings don't show alignment | Low | Medium | Would mean Fourier encoding doesn't create structural alignment — contradicts Plan 061 results |
| riir-ai analogy plan yields nothing | Medium | Low | The diagnostic is still useful; analogy is a stretch goal |
