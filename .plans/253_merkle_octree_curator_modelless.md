# Plan 253: Merkle-Octree Node-Tier Curator Consensus — Modelless Verification Layer

> **Status:** ✅ Complete
> **Date:** 2026-06-12
> **Research:** Research 221 — Merkle-Octree Curator Consensus
> **Depends On:** `sense_composition` (Plan 221, existing), `bandit` (BanditPruner infrastructure)
> **Feature Gate:** `merkle_octree` (opt-in, new)
> **Parent Plan:** `221_kg_latent_octree_sense_composition.md`

---

## Overview

Add a **modelless verification layer** to the existing KG Latent Octree Sense Composition system. A depth-3 Merkle octree (73 fixed nodes) commits all KG triples to BLAKE3 hashes. Curator nodes verify sense data without any model inference — checking KG consistency via dot-product similarity, spectral flatness, and latent conditioning. A bandit-based reputation system tracks curator accuracy and routes verification weight accordingly.

**Key insight:** The existing `SenseOctreeBuilder` already produces 8-octant occupancy. This plan adds cryptographic commitment (Merkle hashes) on top, plus a curator verification + bandit reputation layer that needs zero model weights.

---

## Tasks

### Phase 1: Merkle Data Structure

- [x] **T1: Implement `MerkleOctree`** — 73-node fixed array (depth-3: 1 root + 8 internal + 64 leaves), per-node `[u8; 32]` BLAKE3 hashes, zero-alloc build. Feature-gated behind `merkle_octree`. — `katgpt-core/src/merkle.rs` — GOAT: build < 5µs
- [x] **T2: Add `build_with_merkle()` to `SenseOctreeBuilder`** — bottom-up hash computation: leaves = `BLAKE3(kg_triple_data || embedding_bytes)`, internal = `BLAKE3(child_0_hash || ... || child_7_hash)`, root hash stored in `SenseModule`. — `katgpt-core/src/sense/octree.rs` — GOAT: overhead < 2µs on top of existing `build()`
- [x] **T3: Implement `MerkleProof`** — generate/verify O(log n) inclusion proofs for depth-3 (2 sibling levels × 7 siblings). `generate(leaf_index) → MerkleProof`, `verify(proof, root_hash) → bool`. — `katgpt-core/src/merkle.rs` — GOAT: proof gen < 1µs, verify < 1µs

### Phase 2: Curator Verification

- [x] **T4: Implement `CuratorVerifier`** — modelless checks: (1) KG consistency = dot-product similarity between KG embedding and claimed octree direction, (2) spectral flatness = variance of leaf hashes must exceed entropy floor, (3) latent conditioning = sigmoid(dot(query_vector, direction)) within [0,1]. No model weights. — `katgpt-core/src/curator.rs` — GOAT: verify single module < 2µs
- [x] **T5: Implement `MerkleFrozenEnvelope`** — extends `MuxPatternStore` freeze pattern with BLAKE3 Merkle root for self-play data. `freeze_with_root(key, target, merkle_root)`, `thaw_and_verify(key) → Option<(&MuxTarget, bool)>`. — `katgpt-core/src/curator.rs` — GOAT: freeze/thaw overhead < 1µs
- [x] **T6: Freeze/thaw Merkle integration** — G-Zero `GoSelfPlayResult[]` → extract KG triples → freeze with Merkle root → thaw verifies against root. Wire into existing `run_gzero_selfplay` flow. — `katgpt-core/src/curator.rs`, `katgpt-core/examples/` — GOAT: full pipeline overhead < 3% of self-play loop

### Phase 3: Curator Bandit

- [x] **T7: Implement `CuratorBandit`** — reuses `BanditPruner` infrastructure pattern. Tracks curator accuracy (correct verifications vs false positives/negatives). Thompson sampling (Beta distribution) for reputation scoring. Per-curator `alpha`/`beta` counts, `sample() → f32` for verification weight. — `katgpt-core/src/curator.rs` — GOAT: sample + update < 100ns
- [x] **T8: AbsorbCompress integration** — high-accuracy curators (>80% correct) get amplified verification weight. Low-accuracy curators (<50%) get probation (weight → 0). EMA decay on alpha/beta to handle concept drift. Reuses existing `AbsorbCompress` promotion/demotion pattern from Go self-play. — `katgpt-core/src/curator.rs` — GOAT: reputation update < 200ns

### Phase 4: Tests & Benchmarks

- [x] **T9: Unit tests** — MerkleOctree build (empty, single leaf, full 64 leaves), proof gen + verify (valid proof, tampered leaf, wrong root), curator verifier (consistent KG, inconsistent KG, spectral anomaly), bandit reputation (convergence after N verifications). — `katgpt-core/src/merkle.rs`, `katgpt-core/src/curator.rs`
- [x] **T10: Benchmark** — Merkle build from 64 KG embeddings (< 5µs target), proof generation (< 1µs), proof verify (< 1µs), curator verify single module (< 2µs), bandit sample + update (< 100ns). — `katgpt-core/benches/merkle_octree_bench.rs`
- [x] **T11: GOAT proof** — inclusion proof verifies in < 1µs, full Merkle build from `SenseModule` data < 5µs, curator bandit converges within 100 episodes to > 75% accuracy. Create `.benchmarks/221_merkle_octree_goat.md` with results. — `.benchmarks/221_merkle_octree_goat.md`

### Phase 5: Feature Gate & Integration

- [x] **T12: Add `merkle_octree` feature flag** — add to `katgpt-core/Cargo.toml` as `merkle_octree = ["sense_composition"]`. Guard `merkle.rs` and `curator.rs` modules. — `katgpt-core/Cargo.toml`, `katgpt-core/src/lib.rs`
- [x] **T13: Wire `MerkleOctree` into `SenseModule`** — `build_with_merkle()` replaces `commitment` with Merkle root hash. No additional `merkle_root` field needed — `commitment` IS the Merkle root when built via Merkle path. `build()` (non-Merkle) uses flat BLAKE3 as before. — `katgpt-core/src/sense/octree.rs`

---

## File Structure

```
katgpt-core/src/
├── merkle.rs          # T1, T3 — MerkleOctree + MerkleProof
├── curator.rs         # T4-T8 — CuratorVerifier + MerkleFrozenEnvelope + CuratorBandit
├── sense/octree.rs    # T2, T13 — build_with_merkle() + merkle_root field
├── types.rs           # T13 — SenseModule optional merkle_root
└── lib.rs             # T12 — module declarations behind feature gate

katgpt-core/benches/
└── merkle_octree_bench.rs  # T10

.benchmarks/
└── 221_merkle_octree_goat.md  # T11
```

## Dependency Graph

```
T1 (MerkleOctree) ──→ T2 (build_with_merkle) ──→ T13 (SenseModule wire)
     │
     └──→ T3 (MerkleProof) ──→ T5 (MerkleFrozenEnvelope) ──→ T6 (Freeze/thaw integration)
                                      │
                                      └──→ T4 (CuratorVerifier) ──→ T7 (CuratorBandit) ──→ T8 (AbsorbCompress)
                                                                           │
T9 (tests) ←─ all of above                                              │
T10 (bench) ←─ T1, T3, T4, T7                                         │
T11 (GOAT) ←─ T9, T10                                                 │
T12 (feature flag) ←─ all                                              │
```

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| MerkleOctree build (64 leaves) | < 5µs | 73 × BLAKE3, bottom-up |
| Proof generate | < 1µs | 3 sibling hash copies |
| Proof verify | < 1µs | 3 BLAKE3 hashes |
| Curator verify (single module) | < 2µs | dot-product + spectral check |
| Bandit sample + update | < 100ns | Beta distribution sample |
| Freeze/thaw with Merkle | < 1µs | BLAKE3 compare on root |

---

## TL;DR

Depth-3 Merkle octree (73 nodes) commits KG triples to BLAKE3 hashes. Curator nodes verify sense data modellessly via dot-product similarity + spectral flatness. Bandit reputation tracks curator accuracy. All behind `merkle_octree` feature flag, reuses existing `SenseOctreeBuilder` + `BanditPruner` infrastructure. Target: proof verify < 1µs, build < 5µs.
