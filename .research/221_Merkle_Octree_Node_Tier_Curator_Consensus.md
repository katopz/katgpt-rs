# Research 221: Merkle-Octree Node-Tier Curator Consensus — Modelless Verification Layer

**Date:** 2026-06-12
**Status:** 🟢 GAIN — modelless verification infrastructure, no LLM needed
**Domain:** Modelless (inference-time latent verification, Merkle proof generation/verification)
**Relates To:** riir-ai Research 107 (Merkle-Octree for chain), katgpt-rs Research 208 (SLoD), Plans 049 (G-Zero), 248 (OctreeCTC)
**Component:** `katgpt-core::sense`, `katgpt-core::curator` (new), `katgpt-rs::pruners::bandit`

---

## TL;DR

We have **flat BLAKE3 commitments everywhere** — per `SenseModule`, per `LatentPatch`, per `ProvenanceChain` step, per `ShardEmbedding`. None of these compose into hierarchical proofs. A Curator node that wants to verify "are these 6 sense modules consistent with their claimed KG triples?" must recompute everything from scratch — O(n) with no inclusion/exclusion proofs.

This research proposes:

1. **Merkle octree** — add per-node `[u8; 32]` hashes to the `SenseOctreeBuilder` output, producing a single Merkle root per `SenseModule` that commits to all KG triples
2. **Merkle proofs** — O(log n) inclusion/exclusion proof generation and verification, pure BLAKE3, no chain dependency
3. **Curator verification** — modelless consistency checks: latent flatness detection, spectral shard integrity, KG triple consistency — all verified via Merkle proofs, no LLM inference
4. **Freeze/thaw Merkle commitments** — frozen self-play data carries a Merkle root so thawed data can be verified incrementally
5. **Curator Bandit** — bandit-driven reputation scoring for curator nodes, reusing existing `BanditPruner` infrastructure

The chain side (riir-ai Research 107) uses the Merkle roots for quorum consensus. This research focuses on what lands in the **modelless MIT engine** — the pure data structure, proof generation/verification, and modelless curator checks.

---

## Current State

| Component | Commitment | Structure | Gap |
|-----------|-----------|-----------|-----|
| `SenseModule` (232B Pod) | `BLAKE3(kind ‖ octree_bits ‖ dirs ‖ conf)` | Per-module flat | No per-KG-triple proof |
| `SenseOctreeBuilder` | Writes `octree_bits: [u64; 4]` occupancy | Bit-plane only | No hash per octree node |
| `LatentPatch` (68B) | `BLAKE3(segment_id ‖ weights)` | Per-segment flat | No patch-set Merkle root |
| `LatentPatchBatch` | Verifies each patch independently | O(n) verification | No batch Merkle root |
| `ProvenanceChain` | `BLAKE3(episode_id ‖ reward ‖ bandit_pull)` per step | Linear chain hash | Not a tree — no sub-range proof |
| `ShardEmbedding` (8-dim) | **No commitment** | — | Can't verify embedding integrity |
| `JlProjectionMatrix` | `BLAKE3(matrix_bytes)` | Per-matrix flat | No proof of correct projection |
| `StaticCalTable` | `BLAKE3(scales)` | Per-table flat | No per-head proof |
| `ProofCertificate` | JSON + BLAKE3 file checksum | Per-certificate | No Merkle over certificate set |
| `DreamerFrozenBank` | Magic + version validation only | No cryptographic commitment | No tamper proof on frozen data |

### The Problem

1. **No composability** — each component hashes independently. Can't prove "this SenseModule's KG triples are a subset of this NpcBrain's knowledge" without recomputing everything
2. **No inclusion proofs** — a light client (WASM sense module, browser NPC) can't verify a single KG triple without downloading the entire 232B `SenseModule`
3. **No curator verification** — no modelless way to check "are these sense modules self-consistent?" without re-running the entire octree build pipeline
4. **No freeze integrity** — frozen self-play data has magic+version but no Merkle commitment. Can't verify partial thaw
5. **No curator reputation** — curators have no feedback signal. Bad curators (slow, wrong proofs) are treated same as good ones

---

## Proposed Architecture

### 1. Merkle Octree Data Structure

Add a `MerkleOctree` that wraps the existing `octree_bits` occupancy with per-node BLAKE3 hashes:

```rust
/// Merkle-augmented octree node.
/// Leaf nodes hash KG triple data directly.
/// Internal nodes hash their children's hashes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MerkleNode {
    /// BLAKE3 hash of this node's content.
    /// Leaf: BLAKE3(kg_triple_bytes)
    /// Internal: BLAKE3(child_0_hash || ... || child_7_hash)
    pub hash: [u8; 32],
}

/// Merkle commitment for a SenseModule's KG triple octree.
/// Fixed-size: max depth 3 = 1 + 8 + 64 = 73 nodes.
/// Stored as flat array indexed by Morton code.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct MerkleOctree {
    /// Per-node hashes, indexed by Morton code.
    /// Node 0 = root, nodes 1..8 = depth 1, etc.
    pub nodes: [MerkleNode; 73],
    /// Number of active leaf nodes (for proof optimization).
    pub n_leaves: u8,
    /// BLAKE3 commitment over all node hashes (root hash shortcut).
    pub root: [u8; 32],
}
```

**Size impact:** 73 × 32B = 2,336B per `SenseModule`. This is ~10× the current 232B. Acceptable for modelless verification — the Merkle tree is optional (feature-gated), and the root hash `[u8; 32]` is the only thing needed for the chain side.

**Optimization:** For sparse octrees (most sense modules have < 8 active leaves), store only the active path hashes. Dense representation for verification, sparse for storage.

### 2. Merkle Hashing for SenseOctreeBuilder

Extend `SenseOctreeBuilder::build()` to optionally compute Merkle hashes:

```rust
impl SenseOctreeBuilder {
    /// Build with Merkle commitment (feature-gated).
    #[cfg(feature = "merkle_octree")]
    pub fn build_with_merkle(
        &self,
        kind: SenseKind,
        embeddings: &[KgEmbedding],
    ) -> (SenseModule, MerkleOctree) {
        let module = self.build(kind, embeddings);
        let merkle = self.compute_merkle(&module, embeddings);
        (module, merkle)
    }

    fn compute_merkle(&self, module: &SenseModule, embeddings: &[KgEmbedding]) -> MerkleOctree {
        // Bottom-up: hash each leaf from its KgEmbedding,
        // then hash each internal node from its children.
        // Root hash = MerkleOctree.root
    }
}
```

**Key invariant:** `MerkleOctree.root == SenseModule.commitment` when computed from the same embeddings. This bridges the flat BLAKE3 world to the Merkle world without breaking existing verification.

### 3. Merkle Proof Generation/Verification

```rust
/// Inclusion proof: proves a specific KG triple exists in the octree.
/// O(log n) hashes along the path from leaf to root.
#[derive(Clone, Debug)]
pub struct MerkleProof {
    /// Index of the leaf node (Morton code).
    pub leaf_index: u8,
    /// Sibling hashes along the path to root.
    /// Length = octree_depth (max 3).
    pub siblings: Vec<[u8; 32]>,
    /// Expected root hash.
    pub expected_root: [u8; 32],
}

impl MerkleOctree {
    /// Generate inclusion proof for a leaf.
    pub fn prove_inclusion(&self, leaf_index: u8) -> Option<MerkleProof>;

    /// Verify an inclusion proof against a known root.
    /// Pure BLAKE3 — no chain dependency.
    pub fn verify_proof(proof: &MerkleProof) -> bool;
}
```

### 4. Curator Verification Logic

The Curator performs **modelless consistency checks** — no LLM inference needed:

```rust
/// Curator verification result.
#[derive(Clone, Debug)]
pub struct CuratorVerdict {
    /// Merkle root matches claimed commitment.
    pub commitment_valid: bool,
    /// KG triples are self-consistent (no contradictory embeddings).
    pub kg_consistent: bool,
    /// Spectral flatness within tolerance (ShardEmbedding quality).
    pub spectral_flat: bool,
    /// Latent directions are well-conditioned (not degenerate).
    pub latent_conditioned: bool,
    /// Overall pass/fail.
    pub pass_: bool,
}

/// Modelless curator verification — pure math, no LLM.
pub struct CuratorVerifier {
    /// Max allowed cosine distance between contradictory triples.
    pub contradiction_threshold: f32,
    /// Min spectral flatness ratio (eigenvalue_max / eigenvalue_min).
    pub spectral_flatness_ratio: f32,
    /// Min condition number for latent directions.
    pub min_condition_number: f32,
}
```

#### Verification Checks

| Check | Method | Cost |
|-------|--------|------|
| **Commitment integrity** | Recompute Merkle root, compare to claimed `SenseModule.commitment` | O(n) BLAKE3 hashes |
| **KG triple consistency** | For each (entity, relation) pair, check no two embeddings have `cosine_sim > threshold` with opposite signs | O(k²) dot products per relation |
| **Spectral flatness** | Compute condition number of `[TernaryDir]` matrix — if all directions are parallel, spectral ratio collapses | O(d²) for 8×8 matrix |
| **Latent conditioning** | Check `ShardEmbedding` projections are not degenerate (dot product between any two ≠ ±1) | O(k²) dot products |
| **Inclusion proof** | Verify Merkle proof for specific KG triple | O(log n) BLAKE3 hashes |
| **Batch proof** | Verify multiple inclusion proofs against same root | O(m × log n) amortized |

### 5. Freeze/Thaw Merkle Commitment

Extend the freeze/thaw pipeline to carry Merkle roots:

```rust
/// Merkle-committed frozen data envelope.
#[repr(C)]
pub struct MerkleFrozenEnvelope<T> {
    /// Magic bytes: b"MKFC".
    pub magic: [u8; 4],
    /// Version.
    pub version: u32,
    /// Merkle root over the frozen data.
    pub merkle_root: [u8; 32],
    /// Data size in bytes.
    pub data_len: u64,
    /// BLAKE3 commitment over magic + version + merkle_root + data_len.
    pub envelope_commitment: [u8; 32],
    /// The frozen data follows (repr(C)).
    // pub data: T, // inline, after envelope header
}
```

**Workflow:**
1. **Freeze:** `save_frozen()` → compute Merkle root → write `MerkleFrozenEnvelope` + data
2. **Thaw:** `load_frozen()` → read envelope → verify `merkle_root` matches recomputed root → verify `envelope_commitment`
3. **Partial thaw:** Verify inclusion proof for specific data subset without thawing everything

This connects to G-Zero self-play data: `StateTransition[] → KgTriple[] → SenseModule[]` → freeze with Merkle root. When thawed for training, the Merkle root proves no data corruption occurred.

### 6. Curator Bandit

Reuse existing `BanditPruner` infrastructure to score curator reliability:

```rust
/// Curator arms for bandit reputation.
#[derive(Clone, Copy, Debug)]
pub enum CuratorArm {
    /// Accept curator's verification (trust).
    Accept,
    /// Reject curator's verification (distrust).
    Reject,
    /// Re-verify independently (audit).
    Audit,
}

/// Curator reputation scored via bandit feedback.
pub struct CuratorBandit {
    /// Underlying bandit pruner (UCB1 default).
    pruner: BanditPruner<CuratorArm>,
    /// Trial log for audit trail.
    log: TrialLog,
}
```

**Reward signal:**
- `Accept` → reward = 1.0 if downstream chain accepts, -1.0 if chain rejects
- `Reject` → reward = 0.5 (safe default, prevents bad data but wastes good data)
- `Audit` → reward = 0.8 if independent verification matches curator, -0.5 if mismatch

The bandit learns which curators to trust, which to audit, and which to reject — all modelless, using existing `AbsorbCompress` feedback loop.

---

## Integration Points

### katgpt-rs (Modelless, MIT)

| Component | Integration | Feature Flag |
|-----------|-------------|--------------|
| `SenseOctreeBuilder` | Add `build_with_merkle()` | `merkle_octree` |
| `SenseModule` | Add optional `merkle_root: [u8; 32]` alongside `commitment` | `merkle_octree` |
| `MerkleOctree` | New struct in `sense/merkle.rs` | `merkle_octree` |
| `MerkleProof` | New struct in `sense/merkle.rs` | `merkle_octree` |
| `CuratorVerifier` | New module `curator/` | `curator` |
| `MerkleFrozenEnvelope` | Extend `freeze.rs` | `merkle_freeze` |
| `CuratorBandit` | New in `pruners/curator_bandit.rs` | `curator` |
| `LatentPatchBatch` | Optional batch Merkle root | `merkle_octree` |
| `ProofCertificate` | Optional Merkle root over certificate set | `merkle_proofs` |

### riir-ai (Chain, Private) — Interface Boundary

| riir-ai Component | Uses from katgpt-rs | What riir-ai adds |
|-------------------|---------------------|-------------------|
| `ShardQuorum` | `MerkleOctree::root` for quorum root comparison | Multi-node root agreement protocol |
| `SyncBlock` | `MerkleProof` for inclusion/exclusion in block | Block-level Merkle schema |
| `BrowserCatchup` | `MerkleProof::verify()` for light client proof | Proof request/response protocol |
| `NeuronShard` | `merkle_root` alongside flat `commitment` | Zone-level Merkle commitment |
| Fraud proofs | `MerkleProof` as evidence of inconsistency | Fraud proof construction + slashing |

**Boundary rule:** katgpt-rs provides pure data structure + proof generation/verification. riir-ai adds the consensus protocol, network transport, and slashing logic. No chain dependency leaks into the modelless layer.

---

## Performance Impact

### Compute Cost

| Operation | Cost | Notes |
|-----------|------|-------|
| Build Merkle octree | O(n) BLAKE3 hashes | Same order as existing `SenseModule::commit()` |
| Generate proof | O(d) BLAKE3 hashes, d = depth ≤ 3 | ~3 hashes for depth-3 octree |
| Verify proof | O(d) BLAKE3 hashes | ~3 hashes — negligible |
| Curator full verify | O(n) hashes + O(k²) dot products | Dominated by KG consistency check |
| Freeze with Merkle | O(n) hashes + 1 envelope write | Same as current freeze + ~100µs |
| Thaw with Merkle verify | O(n) hashes | Same as current thaw + ~100µs |
| Bandit curator score | O(1) per curator per episode | Negligible |

### Memory Cost

| Component | Current | With Merkle | Delta |
|-----------|---------|-------------|-------|
| `SenseModule` | 232B | 232B + 2328B (optional) | +10× when feature enabled |
| `MerkleProof` | N/A | ~100B per proof | New allocation |
| `MerkleFrozenEnvelope` | N/A | 52B header | Fixed overhead |
| `CuratorVerifier` | N/A | ~128B config | Fixed overhead |

### Throughput Impact

BLAKE3 hashes at ~1 GB/s on modern hardware. For a typical 8-leaf octree:
- 73 hashes × 32B = 2,336B input → ~2.3µs for full Merkle build
- Merkle proof: 3 hashes × 32B = ~0.1µs
- Verification: same as proof generation

**Net impact:** < 5µs per SenseModule for full Merkle construction. < 1µs for proof generation/verification. Negligible compared to the existing `SenseModule::project()` call.

---

## GOAT Pillar Assessment

### Is This Novel?

Merkle trees are standard. The novel fusion is:

1. **Merkle octree for latent verification** — standard Merkle tree over a non-standard structure (KG latent octree with ternary direction vectors)
2. **Modelless curator** — verification without any LLM inference, using pure BLAKE3 + linear algebra checks
3. **Bandit-driven curator reputation** — adapting existing `BanditPruner` to a new domain (curator trust scoring)
4. **Freeze/thaw Merkle commitment** — extending `repr(C)` freeze with cryptographic commitment for self-play data integrity

### Verdict by 003 Strategy

| Criterion | Assessment |
|-----------|-----------|
| Real gap? | ✅ Zero Merkle proofs, zero curator verification, zero freeze integrity today |
| Technical feasibility? | ✅ Pure data structure + BLAKE3 (already used). ~500 LOC for core |
| Modelless-first? | ✅ No LLM, no training, no model dependency |
| Engine/fuel split? | ✅ Merkle tree + proofs + curator checks = engine (MIT). Chain protocol = fuel (private) |
| Cross-repo dependency? | ✅ katgpt-rs provides roots + proofs. riir-ai consumes them for consensus |
| Reuses existing infra? | ✅ `SenseOctreeBuilder`, `BanditPruner`, `TrialLog`, `AbsorbCompress`, `freeze.rs` |
| Perf impact? | ✅ < 5µs per module for full Merkle. Feature-gated = zero cost when off |

### Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Memory overhead (10× SenseModule size) | Feature-gated: `merkle_octree` off by default. Only root hash `[u8; 32]` needed for chain |
| Sparse octree waste (73 nodes for 3 leaves) | Sparse Merkle representation: only store active paths |
| Curator false positives (good data flagged) | Bandit reputation with Thompson Sampling → converges to accurate trust scores |
| Freeze format break | `MerkleFrozenEnvelope` is backward-compatible: read old format, write new format |
| Over-engineering | Feature-gated everything. `merkle_octree` and `curator` are opt-in |

---

## Dependencies

| Dependency | Source | Status |
|------------|--------|--------|
| `SenseOctreeBuilder` | `katgpt-core::sense::octree.rs` | ✅ Exists |
| `SenseModule` (232B Pod) | `katgpt-core::types.rs` | ✅ Exists |
| `TernaryDir` + `KgEmbedding` | `katgpt-core::types.rs` | ✅ Exists |
| `ShardEmbedding` (8-dim) | `katgpt-core::types.rs` | ✅ Exists |
| `JlProjectionMatrix` | `katgpt-core::shard_embedding.rs` | ✅ Exists |
| BLAKE3 | `katgpt-core` (already used) | ✅ Exists |
| `BanditPruner` (UCB1/Thompson/ε-greedy) | `katgpt-rs::pruners::bandit.rs` | ✅ Exists |
| `TrialLog` + `AbsorbCompress` | `katgpt-rs::pruners` | ✅ Exists |
| `freeze.rs` (save_frozen/load_frozen) | `katgpt-rs::pruners::freeze.rs` | ✅ Exists |
| `ProofCertificate` + `verify_proof_chain` | `katgpt-rs::proof_cert` | ✅ Exists |
| `ProvenanceChain` (linear chain hash) | `katgpt-rs::pruners::regime_transition.rs` | ✅ Exists |
| `OctreeCTC` (reconstructive navigation) | `katgpt-core::sense::reconstruction.rs` | ✅ Exists |
| `simddot_f32` | `katgpt-core::simd.rs` | ✅ Exists |
| Merkle tree data structure | — | ❌ **NEW** |
| Curator verification module | — | ❌ **NEW** |
| Merkle proof generation/verification | — | ❌ **NEW** |
| `MerkleFrozenEnvelope` | — | ❌ **NEW** |
| `CuratorBandit` | — | ❌ **NEW** |

**No new external dependencies.** All new code is pure Rust + BLAKE3 + existing infra.

---

## What NOT to Do

1. **Don't add chain consensus logic** — that's riir-ai Research 107's job. katgpt-rs provides roots and proofs, not quorum protocol
2. **Don't modify `SenseModule` layout** — the 232B Pod is frozen. Merkle data is stored alongside, not inside
3. **Don't make Merkle mandatory** — feature-gate everything. Flat BLAKE3 must remain the default for backward compat
4. **Don't add LLM inference to curator** — the whole point is modelless verification. If an LLM is needed, it's in riir-ai
5. **Don't replace `ProvenanceChain`** — it serves a different purpose (audit trail for bandit episodes). Merkle octree is for KG triple verification. They coexist
6. **Don't build a generic Merkle tree** — the octree structure is specific enough that a specialized `MerkleOctree` is simpler and faster than a generic `MerkleTree<T>`

---

## Connection to riir-ai Research 107

Research 107 (riir-ai) focuses on:
- `ShardQuorum` Merkle root comparison for Byzantine detection
- `SyncBlock` Merkle schema (zone-level commitments)
- `BrowserCatchup` light client proof protocol
- Fraud proof construction for slashing

This research (221, katgpt-rs) provides:
- The **MerkleOctree** data structure that 107's quorum compares
- The **MerkleProof** generation/verification that 107's light clients use
- The **CuratorVerifier** that runs before chain submission (pre-filter bad data)
- The **MerkleFrozenEnvelope** that ensures self-play data integrity from freeze to chain commit

**Data flow:**
```
G-Zero self-play → StateTransition[] → KgTriple[]
     ↓ SenseOctreeBuilder::build_with_merkle()
SenseModule + MerkleOctree (root hash)
     ↓ CuratorVerifier::verify()
CuratorVerdict (pass/fail)
     ↓ if pass: freeze with MerkleFrozenEnvelope
     ↓ thaw → chain submission → riir-ai ShardQuorum
     ↓ quorum compares MerkleOctree::root across nodes
```

---

## Tasks

- [ ] Implement `MerkleNode` and `MerkleOctree` structs in `katgpt-core/src/sense/merkle.rs` (feature: `merkle_octree`)
- [ ] Add `SenseOctreeBuilder::build_with_merkle()` — bottom-up hash computation from KgEmbeddings
- [ ] Implement `MerkleProof` generation (`prove_inclusion`) and verification (`verify_proof`)
- [ ] Implement sparse Merkle representation for memory efficiency
- [ ] Add `CuratorVerifier` in `katgpt-core/src/curator/mod.rs` (feature: `curator`)
- [ ] Implement KG triple consistency check (contradictory embedding detection)
- [ ] Implement spectral flatness verification over `[TernaryDir]` matrix
- [ ] Implement latent conditioning check for `ShardEmbedding` projections
- [ ] Add `MerkleFrozenEnvelope` to `freeze.rs` (feature: `merkle_freeze`)
- [ ] Extend `save_frozen` / `load_frozen` with Merkle root computation and verification
- [ ] Implement `CuratorBandit` in `katgpt-rs/src/pruners/curator_bandit.rs` (feature: `curator`)
- [ ] Wire curator bandit reward signal from chain acceptance/rejection feedback
- [ ] Add `merkle_root: [u8; 32]` to `LatentPatchBatch` for batch-level commitment
- [ ] Benchmark: Merkle build time per SenseModule (target: < 5µs)
- [ ] Benchmark: Merkle proof generation/verification time (target: < 1µs)
- [ ] GOAT gate: verify Merkle proofs detect tampered KG triples with 100% recall
- [ ] GOAT gate: verify curator bandit converges to accurate trust scores within 100 episodes
- [ ] Integration test: MerkleOctree root matches SenseModule::commitment for same input
- [ ] Integration test: freeze/thaw with Merkle envelope round-trip preserves integrity
- [ ] Promote `merkle_octree` to default feature if GOAT gates pass

---

## TL;DR

Flat BLAKE3 everywhere → no composability, no inclusion proofs, no curator verification, no freeze integrity. **Propose Merkle octree** (per-node BLAKE3 hashes over existing KG latent octree), **modelless curator** (consistency checks via pure math + Merkle proofs), **freeze/thaw Merkle envelope** (self-play data integrity), and **curator bandit** (reputation scoring via existing `BanditPruner`). All feature-gated, all pure Rust + BLAKE3, no chain dependency, no LLM needed. The chain side (riir-ai 107) uses the Merkle roots for quorum consensus — this research builds the modelless engine layer that produces those roots.

**GOAT verdict: GAIN.** Real gap, pure modelless, reuses existing infra, < 5µs overhead, feature-gated.
