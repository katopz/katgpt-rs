# Research 280: Resolution-Tiered Deterministic Commitment (RTDC)

> **Source:** Fusion of Plan 235 (SLoD, R208) × Plan 253 (Merkle-Octree Curator, R221) × Plan 258 (LatCal Fixed-Point). Not from a single paper — synthesised from three shipped primitives.
> **Date:** 2026-06-22
> **Status:** Active — Super-GOAT verdict, artifacts created in this session
> **Related Research:** 208 (SLoD), 221 (Merkle-Octree Curator), 212 (Gemini Fourier × LatCal, for the spectral-fixed-point precedent), 003 (commercial strategy)
> **Related Plans:** 235 (SLoD ✅), 253 (Merkle-Octree ✅), 258 (LatCal Fixed ✅), 265 (LatCal Spectral Fixed ✅), 302 (RTDC open primitive, new), riir-chain/.plans/003 (RTDC quorum wiring, new)
> **Cross-ref (riir-chain):** `.research/001_Resolution_Tiered_Deterministic_Commitment_Guide.md` — private selling-point guide (chain sync boundary is where this fusion lives)
> **Classification:** Public (modelless math, no game/chain IP)

---

## TL;DR

Three already-shipped primitives — SLoD (continuous spectral zoom on KG embeddings), Merkle-Octree (depth-3 BLAKE3 commitment of KG triples), and LatCal Fixed-Point (deterministic i64↔f32 bridge with epsilon drift gate) — are each useful alone but **structurally disconnected**: SLoD is fuzzy (query-dependent tier), Merkle is bit-exact (single resolution), LatCal makes only 4 wallet scalars deterministic. Fusing them produces **Resolution-Tiered Deterministic Commitment (RTDC)**: a multi-resolution Merkle commitment where each octree depth corresponds to a SLoD σ-boundary, every leaf commits to `LatCalSpectralFixed` scalars (deterministic across platforms), and curators verify only at the depth matching their own σ* — O(log n) proof at the appropriate abstraction level. **Capability none of the three has alone: trust-minimized semantic zoom.** A browser NPC (or WASM light client) can cryptographically verify its fog-of-war view is a faithful sub-summation of the chain-committed full KG, without downloading the full KG.

**Distilled for katgpt-rs (modelless, inference-time):**
The open primitive is the **depth-tiered Merkle octree with deterministic leaf encoding** — pure BLAKE3 + LatCal fixed-point + spectral boundary mapping. No chain semantics, no game semantics, no neuron-shard types. The chain side (riir-chain) consumes the per-depth roots for quorum; the runtime side (riir-ai) consumes the tier router for fog-of-war. Both reuse this one primitive.

---

## 1. The Three Inputs (What Each Already Does)

### 1.1 SLoD — Plan 235 / R208 (shipped, default-ON)

- File: `katgpt-rs/crates/katgpt-core/src/slod.rs` + `sense/lod.rs`
- `SlodOperator::boundary_scan()` detects σ values where the KG representation undergoes qualitative transitions (three-signal composite: Fréchet velocity V(σ), weight divergence D_w(σ), neighbourhood churn C_k(σ), MAD peak picker).
- `SenseLodRouter` already maps distance → 3-tier enum (`Full` / `Compressed` / `Minimal`) using σ1, σ2 boundaries.
- `SenseLodMask` pre-computes the active sense-module set per tier.
- **Gap:** Boundaries are advisory. Nothing enforces that two NPCs at the same distance see the *same* KG subset. Nothing commits the boundary set itself.

### 1.2 Merkle-Octree Curator — Plan 253 / R221 (shipped, opt-in `merkle_octree`)

- Files: `katgpt-rs/crates/katgpt-core/src/merkle.rs` + `curator.rs`
- `MerkleOctree` — fixed 73-node array (1 root + 8 internal + 64 leaves), per-node `[u8; 32]` BLAKE3 hashes, single root hash.
- `MerkleProof` — O(log n) inclusion proof, 3 sibling levels.
- `CuratorVerifier::verify_module()` — modelless KG consistency (ternary dot-self), spectral flatness (Welford variance on leaf hashes), latent conditioning (sigmoid dot with `query_vector[1.0; 8]`).
- `MerkleFrozenStore` + `MerkleFrozenEnvelope` — freeze with root, thaw-and-verify.
- `CuratorBandit` — Thompson sampling (Beta α/β) on curator accuracy, `verification_weight()` branchless piecewise-linear amplifier.
- **Gap:** Commits to **one** resolution. Leaf hash is `BLAKE3(kg_triple_bytes || embedding_bytes)` — f32 embedding bytes are **non-deterministic across platforms**. A curator verifying at distance d and another at distance d' both verify the *same* root — they can't zoom. If the KG subset they care about differs (fog-of-war), the Merkle proof doesn't help them.

### 1.3 LatCal Fixed-Point — Plan 258 / Plan 265 (shipped, opt-in `latcal_fixed_point`)

- File: `riir-chain/src/encoding/latcal_fixed.rs`
- `LatCalFixed` — 4 i64 fields (value, overflow, sign, precision), SCALE = 10^6, `det()` is pure i128 arithmetic (bit-identical across platforms).
- `snap_to_fixed(v: f32) -> Result<i64>` — epsilon drift gate (ε = 0.01), rejects non-finite and drift-exceeding values.
- `LatCalSpectralFixed` — frequency/amplitude/phase on the same fixed-point rails, `energy_invariant()` is i128.
- **Gap:** Only 4 wallet fields + 3 spectral fields are deterministic. Embedding vectors and KG triple payloads are still f32 — they cross the WASM shell boundary as raw bytes with no determinism guarantee.

---

## 2. The Disconnect (Why The Three Don't Compose Today)

Two structural mismatches prevent any pair of these primitives from producing the RTDC capability:

### Disconnect A: SLoD tier is fuzzy, Merkle root is bit-exact

The current `MerkleOctree` root commits to *all* 64 leaves simultaneously. But `SenseLodRouter` routes an NPC at distance d to a *subset* of sense modules (Full/Compressed/Minimal). Two NPCs at different distances from the same target NPC will activate different sense subsets and therefore care about different regions of the target's KG octree. The Merkle root doesn't help them — they both verify the same root, but neither can prove "the subset I'm seeing is the right subset for my abstraction level."

### Disconnect B: Merkle leaf encodes f32 bytes; LatCal only covers wallet + spectral

`MerkleNode::hash = BLAKE3(kg_triple_bytes || embedding_bytes)`. The embedding bytes are platform-dependent f32 layouts (NaN payloads, subnormal handling, x87 80-bit spill on legacy x86). A curator on ARM64 and a curator on x86_64 will compute *different* leaf hashes for the same logical embedding. `LatCalFixed` solves this for 4 wallet scalars and `LatCalSpectralFixed` solves it for 3 spectral scalars — but neither covers the KG triple payload or the embedding vector itself. So the current Merkle root is "I committed to my bytes" not "I committed to the same value everyone else did."

### Disconnect C (the opportunity): the 73-node octree is already depth-3 — SLoD's natural shape

`MerkleOctree` is fixed at 1 + 8 + 64 = 73 nodes. That is *exactly* the shape of a 3-tier SLoD hierarchy:
- Depth 0 (root, 1 node) = coarsest σ_∞ — global Fréchet centroid
- Depth 1 (8 nodes) = first boundary σ_1 — regional clusters
- Depth 2 (64 leaves) = finest σ_0 — individual KG triples

The structure is already there. The three plans just haven't been wired to *use* it as a zoom hierarchy.

---

## 3. Distillation — RTDC, the Open Primitive

### 3.1 The mechanism in one paragraph

Add a **DepthTieredMerkleOctree** that exposes one Merkle root *per depth* (currently only the global root is exposed). Leaf encoding switches from `BLAKE3(kg_triple_bytes || embedding_bytes)` to `BLAKE3(kg_triple_bytes || LatCalSpectralFixed{freq, amp, phase})` — deterministic across platforms. The depth boundaries are assigned by SLoD's `boundary_scan` (currently arbitrary Morton codes). A `ResolutionTieredCurator` queries the *target's* distance to itself, computes σ*(d), picks the corresponding depth, and verifies **only that depth's Merkle root** — O(log n) proof at the appropriate abstraction level. Two curators at different distances verify different roots; the deeper root is a strict sub-summation of the shallower one (verified by a `subtree_inclusion` proof). Trust-minimized semantic zoom.

### 3.2 Open primitive — `katgpt-rs/crates/katgpt-core/src/rtdc.rs` (new)

Generic math, no game/chain/shard semantics. Public MIT.

```rust
/// One Merkle root per depth tier. Depth-3 = 3 roots.
/// Each root is a strict sub-summation of the root below it:
///   roots[0] commits to the global Fréchet centroid (1 node)
///   roots[1] commits to 8 regional centroids
///   roots[2] commits to 64 leaf KG triples
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DepthTieredRoots {
    pub roots: [[u8; 32]; 3],
}

/// Selects which depth to verify at, given a continuous σ.
/// Built from SLoD `ScaleBoundary` set (must have ≥2 boundaries for 3 tiers).
#[derive(Clone, Debug)]
pub struct DepthSelector {
    /// σ thresholds from SLoD boundary_scan, ascending.
    pub sigma_thresholds: [f32; 2],
}

impl DepthSelector {
    /// Returns the depth index (0=coarse, 2=fine) for a query σ.
    /// σ <= sigma_thresholds[0]      → depth 2 (full detail)
    /// sigma_thresholds[0] < σ <= t1 → depth 1 (regional)
    /// σ > sigma_thresholds[1]        → depth 0 (global)
    #[inline]
    pub fn select(&self, sigma: f32) -> usize { /* branchless */ }

    /// Construct from SLoD boundaries. Returns None if <2 boundaries.
    pub fn from_boundaries(b: &[crate::slod::ScaleBoundary]) -> Option<Self> { /* ... */ }
}

/// Depth-tiered Merkle octree. Wraps `MerkleOctree` with per-depth roots.
#[cfg(feature = "rtdc")]
pub struct DepthTieredMerkleOctree {
    inner: crate::merkle::MerkleOctree,
    roots: DepthTieredRoots,
    selector: DepthSelector,
}

#[cfg(feature = "rtdc")]
impl DepthTieredMerkleOctree {
    /// Build from SLoD operator + KG embeddings.
    /// Assigns each leaf to a depth-2 slot via Morton code (existing),
    /// each depth-1 internal node via regional aggregation (new),
    /// and the depth-0 root via Fréchet centroid (new — reuses SLoD frechet_mean).
    pub fn build(
        operator: &crate::slod::SlodOperator,
        embeddings: &[crate::sense::KgEmbedding],
        config: &RtdcConfig,
    ) -> Self { /* ... */ }

    /// Prove inclusion at a specific depth. O(log n) at the chosen depth.
    pub fn prove_at_depth(&self, leaf_index: u8, depth: usize) -> Option<RtdcProof> { /* ... */ }

    /// Verify a proof at a specific depth against the corresponding root.
    pub fn verify_at_depth(proof: &RtdcProof, roots: &DepthTieredRoots) -> bool { /* ... */ }

    /// Subtree-inclusion proof: proves roots[d] is a sub-summation of roots[d-1].
    /// Non-trivial — needs the Fréchet centroid relationship to be invertible.
    /// Phase 2 deliverable.
    pub fn prove_subtree_inclusion(&self, shallow: usize, deep: usize) -> Option<SubtreeProof> { /* ... */ }
}

/// Deterministic leaf encoder. Replaces f32 embedding bytes with LatCal fixed-point.
/// NOTE: this is the katgpt-rs PUBLIC encoding trait; the LatCal type itself
/// lives in riir-chain. katgpt-rs only sees a `[u8; N]` deterministic encoding.
pub trait DeterministicLeafEncode {
    /// Encode to a fixed-width byte buffer whose hash is platform-independent.
    /// Implementations MUST guarantee bit-identical output for logically equal input.
    fn encode_deterministic(&self, out: &mut [u8]);
}

/// RTDC proof — inclusion at a specific depth.
#[derive(Clone, Debug)]
pub struct RtdcProof {
    pub leaf_index: u8,
    pub depth: usize,
    pub siblings: Vec<[u8; 32]>,
    pub expected_root: [u8; 32],
}
```

### 3.3 Feature gate

```toml
[katgpt-core.features]
rtdc = ["slod", "merkle_octree", "sense_composition"]
```

Reuses spectral hierarchy via `slod` (transitive), reuses `MerkleOctree` via `merkle_octree`, reuses `KgEmbedding` via `sense_composition`. Zero new deps.

### 3.4 What stays open vs private

| Component | Repo | Reason |
|-----------|------|--------|
| `DepthTieredMerkleOctree`, `DepthSelector`, `RtdcProof`, `DeterministicLeafEncode` trait | `katgpt-rs` (public MIT) | Generic math — no game/chain semantics |
| `subtree_inclusion` proof algorithm | `katgpt-rs` (public MIT) | Math is open; the *use* of it for chain quorum is private |
| `LatCalSpectralFixed`-backed `DeterministicLeafEncode` impl | `riir-chain` (private) | The encoding uses LatCal, which is chain IP |
| Quorum consensus protocol using per-depth roots | `riir-chain` (private) | Chain IP — see `.research/001_*_Guide.md` |
| Fog-of-war verifier for browser NPC | `riir-ai` (private) | Game runtime IP — verifies roots in WASM at the abstraction level the NPC actually sees |
| Freeze-envelope integration with `MerkleFrozenEnvelope` | `riir-neuron-db` (private) | Shard IP — see future `riir-neuron-db/.research/NNN_*` cross-ref |

---

## 4. Connection Map (Force Multiplier)

| Existing pillar | Connection from RTDC |
|-----------------|----------------------|
| **Plan 235 SLoD** (`slod.rs`, `sense/lod.rs`) | Boundary set becomes the depth assignment for the octree — boundaries are now committed, not advisory |
| **Plan 253 Merkle-Octree** (`merkle.rs`, `curator.rs`) | Single root → 3 roots. `CuratorVerifier::verify_module` gains a `verify_at_depth` variant. `CuratorBandit` reputation gets a per-depth dimension (a curator might be accurate at depth 0 but wrong at depth 2) |
| **Plan 258 LatCal Fixed** (`latcal_fixed.rs`) | `LatCalSpectralFixed` becomes the deterministic leaf encoder for KG spectral coefficients |
| **Plan 265 LatCal Spectral Fixed** (same file) | Already ships `freq/amp/phase` — exactly what RTDC leaves encode |
| **Plan 221 KG Latent Octree Sense Composition** (parent of 253) | Sense modules now carry 3 Merkle roots instead of 1, one per LOD tier |
| **Plan 248 OctreeCTC Reconstructive Navigation** | Navigator queries the appropriate depth root for its abstraction level — sub-linear proof cost |
| **Plan 290 Closure-Expansion Instrument** (already noted "Plan 280 Merkle-octree deferred") | PTG snapshots serialize to multi-resolution commitment — a PTG sub-graph at coarse depth is verifiable without the full PTG |
| **riir-neuron-db `MerkleFrozenEnvelope`** | Frozen shard now carries 3 roots; thaw-and-verify at the abstraction level of the consumer |
| **riir-armageddon two-brain model** | Think brain verifies only the depth it can see (fog-of-war); info brain holds all 3 roots |
| **`cgsp_runtime` curiosity** (riir-ai) | Curiosity signal at coarse depth commits to a different root than at fine depth — emergent "what does this NPC know at this distance" verifiable |
| **`latent_functor/reestimation`** (riir-ai) | Re-estimation trigger can compare curator verdicts *across depths* — drift between depth-0 and depth-2 verdicts is a coherence signal |
| **HLA per-NPC belief state** (`sense/reconstruction.rs`) | The 5 scalars that cross sync (valence/arousal/desperation/calm/fear) are already raw — RTDC adds the *commitment* layer so they're tamper-evident at every zoom level |

That's 12 pillars touched — comfortably force-multiplier ≥ 2.

---

## 5. Latent vs Raw Boundary (Critical)

This fusion lives *on* the sync boundary, so the boundary discipline matters:

| Data | Domain | Crosses sync? | RTDC treatment |
|------|--------|---------------|----------------|
| SLoD σ-boundary set | Semantic | No (computed locally per NPC) | Stays local; NPC uses it to pick depth |
| SLoD Fréchet centroid (depth-0 leaf) | Semantic | **Yes** as committed root only | The centroid *vector* is NOT synced — only its `LatCalSpectralFixed` encoding's BLAKE3 hash crosses |
| Per-depth Merkle root `[u8; 32]` | Raw (already bit-exact) | **Yes** | The 3 roots are the sync payload — 96 bytes per NPC per tick |
| KG triple payload at depth-2 | Semantic | **Yes** as committed root only | Triple *bytes* are encoded via `DeterministicLeafEncode` (LatCal fixed-point) before hashing; the raw f32 embedding never crosses |
| Curator reputation (CuratorBandit α/β) | Semantic | No | Local to curator node |
| `subtree_inclusion` proof | Raw (BLAKE3 hashes) | **Yes** | Proofs are sync payloads — ~100 bytes each |
| HLA 5-scalar projection (valence, arousal, …) | Raw (already) | **Yes** | Unchanged — already raw per AGENTS.md |

**Bridge function rule (AGENTS.md):** "Bridge functions MUST be zero-allocation, gateable by feature flag, and not introduce sync dependency." The `DeterministicLeafEncode` impl that uses LatCal fixed-point satisfies this: it's a pure arithmetic conversion (f32 → i64 via `snap_to_fixed`), feature-gated behind `rtdc`, and produces only a hash — the encoding itself does not sync.

**Anti-pattern check:** RTDC does NOT encode position as embedding then decode back — the Fréchet centroid is committed as a *hash of its deterministic encoding*, never reconstructed across the wire. ✓

---

## 6. Verdict — Super-GOAT

### Novelty gate (Q1–Q4)

**Q1: No prior art?**
- Vocabulary-translated grep across both layers (`.research/` + `.plans/` + `src/`/`crates/`) of all five repos for: `slod.*merkle`, `merkle.*slod`, `spectral.*merkle`, `fixed_point.*slod`, `deterministic.*boundary`, `resolution_tier`, `zoom.*merkle`, `tier.*commit`, `commit.*tier`, `abstraction.*commit`, `scale-aware`, `sigma_tier`. **Zero matches** for any 2-of-3 fusion. Closest hits:
  - Plan 290 notes "Cold-tier commitment via Plan 280 Merkle-octree deferred" — single-resolution BLAKE3, not multi-resolution
  - Plan 243 mentions "SpectralLOD already handles adaptive depth" for the octree — *uses* SLoD's tier selection, but the tier itself is not committed
  - `sense/lod.rs` ships the `SenseLodRouter` but routes to a module *mask*, not to a Merkle root
- Codebase-vocabulary grep also clean. **Q1 = YES.**

**Q2: New class of behavior?**
Yes — *trust-minimized semantic zoom*. None of the three inputs alone can produce a cryptographic proof that a coarse-grained view is a faithful sub-summation of a fine-grained committed KG. SLoD has no commitment; Merkle-Octree has no zoom; LatCal Fixed has no KG structure. The fusion is the only path to this capability. **Q2 = YES.**

**Q3: Product selling point?**
"Our NPCs' knowledge graphs are tamper-proof at every zoom level. A browser NPC can verify its fog-of-war view is a faithful sub-summation of the chain-committed full KG, with O(log n) proof at the abstraction level it actually sees — no full-KG download, no trust in the serving node."

This is a moat: it composes the *public* SLoD math (anyone can implement) with the *private* LatCal fixed-point encoding (chain IP) and the *private* chain quorum protocol — competitors can copy the open primitive but cannot replicate the trust-minimized zoom without the chain side. **Q3 = YES.**

**Q4: Force multiplier?**
12 pillars touched (see §4). Solo novelty without integration would be GOAT; this is comfortably Super-GOAT on connection count alone. **Q4 = YES.**

**All 4 YES → verdict = Super-GOAT.**

### Mandatory outputs (created in this session)

1. **Open primitive** → `katgpt-rs/.research/280_Resolution_Tiered_Deterministic_Commitment.md` (this file) + `katgpt-rs/.plans/302_rtdc_open_primitive.md`
2. **Architectural guide** → `riir-chain/.research/001_Resolution_Tiered_Deterministic_Commitment_Guide.md` (selling point lives at the chain sync boundary)
3. **Plans** → `katgpt-rs/.plans/302_*` (open primitive) + `riir-chain/.plans/003_rtdc_quorum_wiring.md` (chain side)

---

## 7. Honest Assessment

### Strengths

1. **The 73-node octree is already depth-3** — the structure maps perfectly onto SLoD's 3 tiers. No layout change, no Pod break, zero impact on existing `merkle_octree` consumers.
2. **All three inputs are shipped and GOAT-proved** — SLoD default-ON (G1–G6), Merkle-Octree opt-in (T1–T13 ✅), LatCal Fixed opt-in (G1–G5 ✅). This is a wiring plan, not a research plan.
3. **Selling point is concretely verifiable** — fog-of-war NPC verifies its view is a sub-summation of the committed full KG. Demo-able.
4. **Commercially differential** — competitors who copy the public SLoD math cannot replicate trust-minimized zoom without the chain IP.

### Risks

1. **`subtree_inclusion` proof is non-trivial.** Proving roots[d] is a sub-summation of roots[d-1] requires the Fréchet centroid relationship to be invertible in some sense. This may need a SNARK-like argument or a Pedersen-commitment-style homomorphic hash. *Mitigation:* defer to Phase 2; Phase 1 ships per-depth roots without the sub-summation proof (curators just verify the depth they care about, no cross-depth proof).
2. **LatCal fixed-point for arbitrary KG payloads is unbounded.** Wallet scalars fit in 4 i64s. Spectral coefficients fit in 3. A free-form KG triple payload (subject-predicate-object strings + embedding) doesn't have a natural fixed-point representation. *Mitigation:* the `DeterministicLeafEncode` trait leaves the encoding open; concrete impls can BLAKE3-hash a canonical serialization (CBOR/postcard) instead of fixed-point per-field. The determinism is the requirement, not the fixed-point specifically.
3. **Fréchet centroid determinism across NPC populations.** Two NPCs running SLoD on slightly different local KG views will compute slightly different centroids. The *hash* of the centroid is what syncs — but if the centroid differs, the hash differs, and quorum fails. *Mitigation:* the synced payload is the centroid computed by the *authority* node for that NPC; other nodes verify, they don't recompute. (Standard quorum model.)
4. **Per-depth root exposure inflates `SenseModule`.** Currently 232B; adding 3 × 32B = +96B if stored inline. *Mitigation:* store only the depth-0 root inline (32B), expose depth-1/2 roots via `prove_at_depth` from the existing 73-node array. No size change.
5. **Fréchet centroid is in hyperbolic space (Poincaré ball).** A "sub-summation" in hyperbolic space is not the same as in Euclidean space — the Fréchet mean is not associative. The subtree-inclusion proof may need a hyperbolic analog. *Mitigation:* open question for Phase 2; Phase 1 sidesteps this by committing each depth independently.

### What this fusion does NOT do

- Does NOT replace learned embeddings (still needs Poincaré ball input from somewhere)
- Does NOT make SLoD deterministic (the σ-boundary detection is still local; RTDC only commits the *result* of running SLoD on the authority node)
- Does NOT verify the SLoD operator itself was run correctly (that would need a zk-proof of the Lanczos iteration — way out of scope)
- Does NOT sync the Fréchet centroid vector across the wire — only its hash

---

## 8. Validation Protocol (G1–G6)

These gates go into the plan (`katgpt-rs/.plans/302_*`). Summary:

| Gate | Metric | Pass if |
|------|--------|---------|
| G1 | `DepthTieredMerkleOctree::build()` overhead vs `MerkleOctree::build()` | ≤ 3× the existing < 5µs target → ≤ 15µs |
| G2 | `prove_at_depth(d)` and `verify_at_depth` for d ∈ {0, 1, 2} | All three depths verify in < 1µs each |
| G3 | `DepthSelector::select(σ)` correctness | σ at exact boundary maps to deeper tier; branchless |
| G4 | `DeterministicLeafEncode` produces bit-identical bytes across ARM64/x86_64/wasm32 | Cross-compile + compare hashes |
| G5 | Cross-platform leaf hash agreement | 1000 random KG triples → identical `[u8; 32]` on all 3 platforms |
| G6 | Fog-of-war demo: WASM curator verifies depth-1 root without downloading depth-2 leaves | Proof size ≤ 200 bytes, verify time < 100µs in WASM |

GOAT promotion: G1–G6 all pass → promote `rtdc` to default-ON. Demote: any failure → keep opt-in, document the negative result.

---

## 9. TL;DR

**RTDC (Resolution-Tiered Deterministic Commitment)** is the novel fusion of Plan 235 (SLoD spectral zoom) × Plan 253 (Merkle-Octree curator) × Plan 258 (LatCal fixed-point). The 73-node depth-3 octree is *already* the right shape for SLoD's 3-tier hierarchy — RTDC just wires the depths to σ-boundaries, switches leaf encoding to deterministic fixed-point, and exposes one Merkle root per depth. The result is **trust-minimized semantic zoom**: a light client verifies its fog-of-war view is a faithful sub-summation of the chain-committed full KG, with O(log n) proof at its own abstraction level. All 4 novelty-gate questions pass (no prior art, new capability class, clear selling point, 12-pillar force multiplier) → **Super-GOAT**. Open primitive lands in `katgpt-rs/crates/katgpt-core/src/rtdc.rs` (generic math, public MIT); chain quorum wiring lands in `riir-chain/`; fog-of-war verifier lands in `riir-ai/`. Phase 1 ships per-depth roots + deterministic leaf encoding (no sub-summation proof). Phase 2 ships the cross-depth `subtree_inclusion` proof (the hard part — needs hyperbolic analog of homomorphic hashing).
