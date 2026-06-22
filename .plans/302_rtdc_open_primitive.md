# Plan 302: RTDC Open Primitive — DepthTieredMerkleOctree + DeterministicLeafEncode

**Date:** 2026-06-22
**Research:** [katgpt-rs/.research/280_Resolution_Tiered_Deterministic_Commitment.md](../.research/280_Resolution_Tiered_Deterministic_Commitment.md)
**Companion guide:** [riir-chain/.research/001_Resolution_Tiered_Deterministic_Commitment_Guide.md](../../../riir-chain/.research/001_Resolution_Tiered_Deterministic_Commitment_Guide.md)
**Chain-side plan:** [riir-chain/.plans/003_rtdc_quorum_wiring.md](../../../riir-chain/.plans/003_rtdc_quorum_wiring.md)
**Depends On:** Plan 235 ✅ (SLoD, default-ON), Plan 253 ✅ (Merkle-Octree Curator, opt-in), Plan 258 ✅ (LatCal Fixed, in riir-chain)
**Target:** `katgpt-rs/crates/katgpt-core/src/rtdc.rs` (new) + feature gate `rtdc`
**Status:** Active — Phase 1 not started

---

## Goal

Ship the **open modelless primitive** for Resolution-Tiered Deterministic Commitment (RTDC): a depth-tiered Merkle octree that exposes one BLAKE3 root per octree depth, where depth boundaries are assigned by SLoD's `ScaleBoundary` set and leaf encoding is platform-deterministic via the `DeterministicLeafEncode` trait. The chain side (`riir-chain`) provides the LatCal-backed impl; the runtime side (`riir-ai`) provides the fog-of-war verifier; this plan ships only the generic math.

**GOAT gate:** G1–G6 (defined below) all pass → promote `rtdc` to default-ON. Failure → keep opt-in, write up negative result in `.research/280_*`.

---

## Architecture

### Feature gate

```toml
# katgpt-rs/crates/katgpt-core/Cargo.toml
[features]
rtdc = ["slod", "merkle_octree", "sense_composition"]
```

Reuses spectral hierarchy (transitive via `slod`), Merkle octree, and sense composition types. Zero new deps.

### Module layout

```
katgpt-rs/crates/katgpt-core/src/
├── rtdc.rs          # new — DepthTieredMerkleOctree, DepthSelector, RtdcProof,
│                    #         DeterministicLeafEncode trait, SubtreeProof (Phase 2 stub)
├── slod.rs          # existing — ScaleBoundary (reuse)
├── merkle.rs        # existing — MerkleOctree, MerkleNode, MerkleProof (reuse)
├── curator.rs       # existing — CuratorVerifier (extend with verify_at_depth)
├── sense/octree.rs  # existing — KgEmbedding, SenseOctreeBuilder (reuse)
└── lib.rs           # add `#[cfg(feature = "rtdc")] pub mod rtdc;`
```

### Core types

```rust
// katgpt-rs/crates/katgpt-core/src/rtdc.rs

use crate::merkle::{MerkleNode, MerkleOctree};
use crate::slod::ScaleBoundary;

/// One Merkle root per depth tier. Depth-3 = 3 roots.
/// roots[0] = coarse (global Fréchet centroid)
/// roots[1] = regional (8 internal nodes)
/// roots[2] = fine (64 leaf KG triples)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthTieredRoots {
    pub roots: [[u8; 32]; 3],
}

impl DepthTieredRoots {
    pub const EMPTY: Self = Self { roots: [[0u8; 32]; 3] };
}

/// Selects which depth to verify at, given a continuous σ.
/// Built from SLoD `ScaleBoundary` set (must have ≥2 boundaries for 3 tiers).
#[derive(Clone, Debug)]
pub struct DepthSelector {
    /// σ thresholds from SLoD boundary_scan, ascending.
    /// sigma_thresholds[0] = boundary between depth-2 (fine) and depth-1 (regional)
    /// sigma_thresholds[1] = boundary between depth-1 (regional) and depth-0 (coarse)
    pub sigma_thresholds: [f32; 2],
}

impl DepthSelector {
    /// σ <= sigma_thresholds[0]      → depth 2 (full detail)
    /// sigma_thresholds[0] < σ <= t1 → depth 1 (regional)
    /// σ > sigma_thresholds[1]        → depth 0 (global)
    ///
    /// Branchless: thresholds packed as f32x2, mask-based selection.
    #[inline(always)]
    pub fn select(&self, sigma: f32) -> usize {
        let above_t0 = (sigma > self.sigma_thresholds[0]) as u32;
        let above_t1 = (sigma > self.sigma_thresholds[1]) as u32;
        // above_t0 + above_t1 gives 0, 1, or 2 — exactly the depth index.
        (above_t0 + above_t1) as usize
    }

    /// Construct from SLoD boundaries. Returns None if <2 boundaries.
    pub fn from_boundaries(boundaries: &[ScaleBoundary]) -> Option<Self> {
        if boundaries.len() < 2 {
            return None;
        }
        // boundaries are sorted ascending by sigma (SLoD invariant).
        // Use the first two; if there are more, take the median? For now, first two.
        Some(Self {
            sigma_thresholds: [boundaries[0].sigma, boundaries[1].sigma],
        })
    }
}

/// Depth-tiered Merkle octree. Wraps `MerkleOctree` with per-depth roots.
#[derive(Clone, Debug)]
pub struct DepthTieredMerkleOctree {
    inner: MerkleOctree,
    roots: DepthTieredRoots,
    selector: DepthSelector,
}

impl DepthTieredMerkleOctree {
    /// Build from SLoD operator + KG embeddings.
    ///
    /// 1. Run `MerkleOctree::build` to get the 73-node array (existing).
    /// 2. Extract depth-0 root: index 0 (the existing root).
    /// 3. Compute depth-1 root: BLAKE3 of the 8 depth-1 node hashes.
    /// 4. Compute depth-2 root: BLAKE3 of the 64 leaf hashes (== existing root
    ///    if the octree is balanced; otherwise separate hash).
    /// 5. Build `DepthSelector` from SLoD boundaries.
    ///
    /// NOTE: this Phase 1 implementation commits each depth independently.
    /// Phase 2 adds `prove_subtree_inclusion` for cross-depth verification.
    pub fn build(
        inner: MerkleOctree,
        boundaries: &[ScaleBoundary],
    ) -> Result<Self, RtdcError> {
        let selector = DepthSelector::from_boundaries(boundaries)
            .ok_or(RtdcError::InsufficientBoundaries {
                n: boundaries.len(),
                need: 2,
            })?;

        // Depth-0 root: existing root (commits to the whole tree as one hash).
        let depth_0_root = inner.root;

        // Depth-1 root: hash the 8 depth-1 internal node hashes together.
        let mut depth_1 hasher = blake3::Hasher::new();
        for i in 0..8 {
            // Nodes 1..8 are depth-1 children of the root (Morton order).
            hasher.update(&inner.nodes[1 + i].hash);
        }
        let depth_1_root = *hasher.finalize().as_bytes();

        // Depth-2 root: hash the 64 leaf hashes together (nodes 9..72).
        let mut depth_2 hasher = blake3::Hasher::new();
        for i in 0..64 {
            hasher.update(&inner.nodes[9 + i].hash);
        }
        let depth_2_root = *hasher.finalize().as_bytes();

        Ok(Self {
            inner,
            roots: DepthTieredRoots {
                roots: [depth_0_root, depth_1_root, depth_2_root],
            },
            selector,
        })
    }

    /// Returns the 3 depth-tiered roots (the sync payload).
    #[inline]
    pub fn roots(&self) -> &DepthTieredRoots {
        &self.roots
    }

    /// Returns the depth selector.
    #[inline]
    pub fn selector(&self) -> &DepthSelector {
        &self.selector
    }

    /// Selects the appropriate depth for a query σ.
    #[inline]
    pub fn depth_for_sigma(&self, sigma: f32) -> usize {
        self.selector.select(sigma)
    }

    /// Generate inclusion proof at a specific depth.
    /// Returns the siblings along the path from `leaf_index` up to (but not
    /// beyond) the chosen depth.
    ///
    /// depth=2 → full path (3 sibling levels)
    /// depth=1 → 2 sibling levels (stops at depth-1 internal node)
    /// depth=0 → 1 sibling level (only the root)
    pub fn prove_at_depth(
        &self,
        leaf_index: u8,
        depth: usize,
    ) -> Option<RtdcProof> {
        if depth > 2 || leaf_index >= 64 {
            return None;
        }
        // For depth d, we need siblings at octree-depths (2-d)..2
        // i.e., depth=2 → siblings at octree-depths 0, 1, 2 (full path)
        //       depth=1 → siblings at octree-depths 0, 1
        //       depth=0 → sibling at octree-depth 0 only
        let n_levels = 3 - depth;
        let mut siblings = Vec::with_capacity(n_levels * 7);
        // ... (Morton traversal — same as existing MerkleProof, truncated)
        // For Phase 1, reuse inner.prove_inclusion and truncate.
        let full_proof = self.inner.prove_inclusion(leaf_index)?;
        siblings.extend_from_slice(&full_proof.siblings[..n_levels]);
        Some(RtdcProof {
            leaf_index,
            depth,
            siblings,
            expected_root: self.roots.roots[depth],
        })
    }

    /// Verify a proof at a specific depth against the corresponding root.
    pub fn verify_at_depth(proof: &RtdcProof, roots: &DepthTieredRoots) -> bool {
        if proof.depth > 2 {
            return false;
        }
        if proof.expected_root != roots.roots[proof.depth] {
            return false;
        }
        // Rebuild the root from siblings + leaf hash
        // (similar to MerkleOctree::verify_proof, scoped to depth levels)
        // ... Phase 1 stub: delegate to inner.verify_proof semantics
        true // TODO Phase 1 T1.3
    }
}

/// Inclusion proof at a specific depth.
#[derive(Clone, Debug)]
pub struct RtdcProof {
    pub leaf_index: u8,
    pub depth: usize,
    pub siblings: Vec<[u8; 32]>,
    pub expected_root: [u8; 32],
}

/// Cross-depth sub-summation proof. Phase 2 deliverable.
/// Proves that roots[d] is a faithful aggregation of roots[d+1].
#[derive(Clone, Debug)]
pub struct SubtreeProof {
    pub shallow_depth: usize,
    pub deep_depth: usize,
    pub proof_bytes: Vec<u8>, // Phase 2 — protocol TBD
}

/// Deterministic leaf encoder. Replaces f32 embedding bytes with a
/// platform-independent byte encoding. Implementations MUST guarantee
/// bit-identical output for logically equal input across ARM64/x86_64/wasm32.
pub trait DeterministicLeafEncode {
    /// Encode to a fixed-width byte buffer whose hash is platform-independent.
    ///
    /// Contract:
    /// - Pure function of `self` (no I/O, no global state)
    /// - Output length is fixed for a given `Self` type
    /// - Bit-identical across all target platforms (ARM64, x86_64, wasm32)
    /// - Suitable for direct BLAKE3 hashing
    fn encode_deterministic(&self, out: &mut [u8]);

    /// Fixed output length. Used by callers to size `out`.
    fn encode_len() -> usize;
}

/// Canonical CBOR impl for any `serde::Serialize` type.
/// Provided as a default impl — chain side can override with LatCal fixed-point.
#[cfg(feature = "serde")]
pub fn encode_cbor_canonical<T: serde::Serialize>(value: &T, out: &mut [u8]) -> usize {
    // Use ciborium for canonical CBOR encoding (deterministic map ordering).
    // ... Phase 1 T1.4
    0
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum RtdcError {
    #[error("need {need} boundaries for 3-tier RTDC, got {n}")]
    InsufficientBoundaries { n: usize, need: usize },
    #[error("depth {depth} out of range (0..=2)")]
    DepthOutOfRange { depth: usize },
    #[error("leaf index {index} out of range (0..64)")]
    LeafOutOfRange { index: u8 },
}
```

---

## Phase 1 — Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/rtdc.rs` with the types above. Add `#[cfg(feature = "rtdc")] pub mod rtdc;` to `lib.rs`. — `katgpt-rs/crates/katgpt-core/src/rtdc.rs`, `katgpt-rs/crates/katgpt-core/src/lib.rs`
- [x] **T1.2** Add `rtdc = ["slod", "merkle_octree", "sense_composition"]` feature gate. Verify `cargo check --features rtdc` compiles. — `katgpt-rs/crates/katgpt-core/Cargo.toml`, `katgpt-rs/Cargo.toml`
- [x] **T1.3** Implement `DepthTieredMerkleOctree::build` (3 depth-root extraction from existing `MerkleOctree`). — `katgpt-rs/crates/katgpt-core/src/rtdc.rs`
- [x] **T1.4** Implement `prove_at_depth` + `verify_at_depth` (Phase 1: truncated path from existing `MerkleOctree::prove_inclusion`). — `katgpt-rs/crates/katgpt-core/src/rtdc.rs`
- [x] **T1.5** Implement `DepthSelector::from_boundaries` + `select` (branchless). — `katgpt-rs/crates/katgpt-core/src/rtdc.rs`
- [x] **T1.6** Implement `DeterministicLeafEncode` trait + `encode_cbor_canonical` default. — `katgpt-rs/crates/katgpt-core/src/rtdc.rs`

### Notes from implementation

- **Domain separators added** to `roots[0]` and `roots[1]` derivations (`rtdc_global_v1`, `rtdc_regional_v1` BLAKE3 tags). Without these, `roots[1]` would equal `roots[2]` because the existing `MerkleOctree` computes its root as `BLAKE3(h₁ ‖ … ‖ h₈)` — identical to the regional-root computation. BIP-340-style tagged hashes ensure all three roots are distinct. Test `build_derives_three_distinct_roots` enforces this invariant.
- **`DepthSelector::select` math inverted** vs the plan's pseudocode. The plan's `above_t0 + above_t1` returned the inverse of the documented semantics (low σ should map to depth 2 / fine, not depth 0 / global). Fixed to `at_or_below_t0 + at_or_below_t1` — matches the docstring and the architectural intent (low diffusion scale = sharp view = finest depth).
- **`encode_cbor_canonical` deferred** — would require a new `ciborium` dep not currently in `Cargo.toml`. Phase 1 ships only the `DeterministicLeafEncode` trait; the CBOR helper is not on any GOAT gate, so it can land in Phase 2 alongside the LatCal-backed impl in riir-chain.
- **`thiserror` not added** — replaced the plan's `#[derive(thiserror::Error)]` with hand-rolled `Display` + `std::error::Error` impls to match the existing convention (`engram/tokenizer.rs::SurjectiveMapLoadError`). katgpt-core stays free of the `thiserror` dep.
- **Phase 1 soundness caveat made explicit**: `prove_at_depth(d=2)` is fully sound; `prove_at_depth(d∈{0,1})` are well-formed but cryptographically unsound until Phase 2's `subtree_inclusion` proof lands (tracked in `riir-chain/issues/002_rtdc_subtree_inclusion_research.md`). Tests `prove_depth_0_trivial` and `prove_depth_1_well_formed_but_phase1_unsound` document this.
- **14 unit tests, all passing** under `cargo test -p katgpt-core --features rtdc`. The pre-existing failure in `curator::tests::test_verification_weight_thresholds` is unrelated (fails with just `merkle_octree`, before `rtdc` is enabled) and was not touched.

### Phase 1 GOAT gates

| Gate | Test | Pass if | Status |
|------|------|---------|--------|
| G1 | `DepthTieredMerkleOctree::build` vs `MerkleOctree::build` (64 leaves) | ≤ 3× wall-clock (target: existing 5µs → ≤ 15µs) | Pending benchmark |
| G2 | `prove_at_depth(d) + verify_at_depth` for d ∈ {0,1,2} | All < 1µs each | Pending benchmark |
| G3 | `DepthSelector::select(σ)` at boundaries | Correct tier selected at σ = threshold ± ε (ε = 1e-6) | ✅ Test `selector_branchless_at_boundaries` |
| G4 | `DeterministicLeafEncode` impls: byte output identical on ARM64 + x86_64 + wasm32 | 1000 random payloads → identical BLAKE3 hashes | Deferred (no impl yet — LatCal impl lives in riir-chain Plan 003) |
| G5 | `DepthTieredRoots::roots` correct | roots[2] == inner.root() when octree is balanced; roots[1] and roots[0] distinct by domain-separator tags | ✅ Test `build_derives_three_distinct_roots` + `build_roots_propagate_leaf_changes` |
| G6 | Cross-platform leaf hash agreement (uses G4) | 1000 random KG payloads → identical hashes on 3 platforms | Deferred (depends on G4) |

**Phase 1 status:** skeleton + unit-level correctness gates (G3, G5) PASS. Performance (G1, G2) and cross-platform determinism (G4, G6) require benchmark + multi-target test harness — both deferred until the LatCal-backed encoder exists in riir-chain Plan 003, since G4/G6 are meaningless without a concrete `DeterministicLeafEncode` impl.

---

## Phase 2 — Chain-side integration (DEFERRED — see riir-chain/.plans/003)

Tasks live in `riir-chain/.plans/003_rtdc_quorum_wiring.md`:
- `KgSpectralPayload: DeterministicLeafEncode` impl using LatCal Fixed
- Chain quorum over 3 roots
- Cold-tier persistence (3 tier roots per block)
- Gas pricing per depth

This plan does NOT cover Phase 2.

---

## Phase 3 — `subtree_inclusion` proof (RESEARCH — tracked in riir-chain Issue 002)

The hard problem: prove that roots[d] is a faithful aggregation of roots[d+1].

**Why it's hard:** Fréchet mean is not associative in hyperbolic space. `FréchetMean(FréchetMean(A), FréchetMean(B))` ≠ `FréchetMean(A ∪ B)` in general.

**Candidates:**
1. Pedersen-style homomorphic commitment on tangent-space log-maps (needs a hyperbolic analog)
2. FFT-style batch verification via Plan 242 Fourier Smoothed Potential Fields
3. Probabilistic proof via sampling (curator verifies K random leaves under each internal node)

**Tracked in:** [`riir-chain/issues/002_rtdc_subtree_inclusion_research.md`](../../riir-chain/issues/002_rtdc_subtree_inclusion_research.md).

### Phase 3 status (updated 2026-06-22)

- [x] **Candidate C (probabilistic sampling) — LANDED** in
      `crates/katgpt-core/src/rtdc.rs` behind feature `rtdc_subtree_inclusion`.
      CG6 PASSES: cost 4.72×≤5.5× (inline test), **4.60×** (formal Criterion
      bench — `.benchmarks/303_rtdc_subtree_inclusion_goat.md`),
      deterministic catch 100%, probabilistic catch at f=1/8 K=8: 66.4%
      empirical vs 65.6% expected. 10 new tests.
      Public API: `SubtreeProof`, `prove_subtree_inclusion`,
      `verify_subtree_inclusion`, `tamper_detection_probability`,
      `min_k_for_95pct_confidence`, `RTDC_SUBTREE_DEFAULT_K`.
      Criterion bench: `crates/katgpt-core/benches/rtdc_subtree_bench.rs`.
- [x] **Candidate B (FFT batch verify)** — **CLOSED with negative result**
      (2026-06-22). After investigating the literature and the RTDC leaf
           data model, Candidate B is a **category error**:
      1. RTDC leaves are 32-byte BLAKE3 hashes, not numerical samples.
         `fft_smooth_into()` (Plan 242, `flow/fft.rs`) operates on
         `Vec<f32>` LEO Q-values — a totally different data type.
      2. FFT batch verification (in the cryptographic sense, e.g.
         arXiv:2405.07941 OR-proofs, or FRI/STARK-based batch Merkle)
         requires either (a) leaves expressed as field elements in a
         finite field, or (b) a SNARK proving layer on top. RTDC commits
         BLAKE3 hashes — neither applies.
      3. Conflated two distinct mechanisms sharing the word "Fourier":
         the Plan 242 navigation primitive (low-pass on potential fields)
         vs cryptographic batch verification (random linear combination /
         FRI over evaluation domains).
      **Recommendation:** do not pursue. Candidate C (already landed) is
      the production answer. If deterministic-per-frequency-cutoff
      soundness ever becomes a hard requirement, the path is Candidate A
      (Pedersen on tangent log-maps), not FFT.
- [ ] **Candidate A (Pedersen)** — deferred (3-4 weeks, needs curve dep).
- [x] **Chain wiring** — **LANDED** in `riir-chain` commit `fac46d5`
      (2026-06-22). `chain_rtdc_subtree` feature in `riir-chain/Cargo.toml`,
      bridge glue in `riir-chain/src/encoding/rtdc_bridge.rs` exposes
      `build_depth_tiered_for_payloads`, `prove_rtdc_subtree`,
      `verify_rtdc_subtree`. 8 new bridge tests pass. See
      [`riir-chain/.plans/003_rtdc_quorum_wiring.md`](../../riir-chain/.plans/003_rtdc_quorum_wiring.md)
      Phase 3 status for details.

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| `DepthTieredMerkleOctree::build` overhead > 3× | Medium | Profile; if > 3×, cache the 3 depth-roots inside the existing `build_with_merkle` (one pass over the 73 nodes) |
| `DeterministicLeafEncode` can't be made deterministic on wasm32 (e.g., NaN payloads differ) | **High (BLOCKER)** | Use `snap_to_fixed` for all floats; reject NaN/Inf at encode time; verify G4 before anything else |
| `prove_at_depth` truncation produces invalid proofs at depth-0 (single-node case) | Low | Special-case depth=0 to return a trivial proof (no siblings, root is leaf hash) |
| `DepthSelector::select` is branchless but the f32x2 packing may not actually vectorize | Low | Fallback to scalar comparisons — already O(1) |

---

## Honest Assessment

### What this plan delivers

- The **open primitive** for RTDC — public types + trait that anyone can implement against.
- A **deterministic encoding contract** that, combined with the chain-side LatCal impl, makes KG commitments cross-platform verifiable.
- The **foundation** for chain quorum at variable abstraction level (chain side is `riir-chain/.plans/003`).

### What this plan does NOT deliver

- The chain quorum protocol itself (→ `riir-chain/.plans/003`)
- The LatCal-backed `DeterministicLeafEncode` impl (→ `riir-chain/.plans/003`)
- The `subtree_inclusion` proof (→ research issue, deferred)
- The fog-of-war WASM verifier (→ future `riir-ai` plan)
- The `MerkleFrozenEnvelope` 3-root extension (→ future `riir-neuron-db` plan)

### Connection to existing GOAT-proved work

| Plan | Status | Connection |
|------|--------|------------|
| 235 (SLoD) | ✅ Default-ON | RTDC reuses `ScaleBoundary` directly |
| 253 (Merkle-Octree) | ✅ Opt-in | RTDC wraps `MerkleOctree`, adds 3 depth roots |
| 258 (LatCal Fixed) | ✅ Opt-in (in riir-chain) | RTDC `DeterministicLeafEncode` impl uses `snap_to_fixed` + `LatCalSpectralFixed` |
| 265 (LatCal Spectral Fixed) | ✅ Opt-in (in riir-chain) | `LatCalSpectralFixed` becomes the leaf encoding for spectral coefficients |
| 221 (KG Latent Octree Sense) | ✅ Existing | Parent of 253; RTDC extends its commitment layer |
| 280 (Merkle-Octree in riir-ai, historical) | Historical | Superseded — RTDC is the canonical multi-resolution commitment path |

---

## TL;DR

Ship `katgpt-rs/crates/katgpt-core/src/rtdc.rs` with `DepthTieredMerkleOctree`, `DepthSelector`, `RtdcProof`, and the `DeterministicLeafEncode` trait. The math is generic (public MIT); the LatCal-backed encoding impl lives in `riir-chain`. G1–G6 gates: build ≤ 15µs, verify < 1µs, cross-platform determinism, SLoD boundary wiring. All pass → promote `rtdc` to default-ON. Phase 2 (chain quorum) and Phase 3 (`subtree_inclusion`) live elsewhere.
