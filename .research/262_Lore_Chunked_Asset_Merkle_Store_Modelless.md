# Research 262: Lore-Style Chunked Content Store — Modelless Merkle Dedup Primitive

> **Source:** Epic Games Lore (https://github.com/EpicGames/lore) — centralized content-addressed VCS for binary-first assets, UEFN's built-in version control. MIT licensed.
> **Date:** 2026-06-18
> **Status:** Active
> **Related Research:** 221 (Merkle Octree Node-Tier Curator Consensus), 196 (KG Latent Octree WASM Composition), 226 (Browser Inference WebGPU WASM SIMD)
> **Related Plans:** 272 (this primitive — modelless), katgpt-rs 253 (MerkleOctree engine layer)
> **Cross-ref (riir-ai):** Research 139 (private selling-point guide), Plan 319 (executable asset vessel + gitflow)
> **Classification:** Public — engine primitive only, no game IP, no chain IP.

---

## TL;DR

Lore is Epic's open-source successor to Git-LFS / Perforce / SVN for game studios. The transferable primitive — independent of game semantics — is **chunked content-addressed storage with Merkle dedup**: large binary blobs (geometry, weights, textures) are split into fixed- or content-defined chunks, each chunk is BLAKE3-hashed, deduplicated against a chunk index, and a blob's identity is a Merkle root over its chunk hashes. On-demand hydration pulls only the chunks a workspace needs; sparse workspaces avoid downloading unused blobs entirely.

This research distills that primitive as a **modelless, MIT, game-agnostic** data structure for `katgpt-rs`. The private fusion — executable WASM-asset vessels, quorum-gated gitflow subnets, candidate-locked shop equip — lives in `riir-ai/.research/139`. The two-way hybrid delivery (raw via VFS for native engines + WASM-as-asset for secured/web targets) is also private.

**Distilled for katgpt-rs (modelless, inference-time):**

A generic, dependency-light `ChunkedContentStore` trait + reference impl that (a) chunks arbitrary bytes (raw geometry, weight shards, WASM modules,KG octree pods) using content-defined slicing, (b) BLAKE3-hashes each chunk for dedup, (c) builds a Merkle root per blob for O(log n) inclusion/exclusion proofs and tamper detection, (d) supports on-demand hydration via a pluggable `ChunkFetcher` trait (filesystem, network, or in-memory), and (e) reuses the existing `MerkleOctree`/`MerkleProof` primitives from Research 221 / Plan 253. **No game semantics, no chain, no consensus, no engine IP.**

---

## 1. Lore Core Findings (Distilled From Public README + System Design)

Lore is pre-1.0 but its architectural primitives are public and well-defined. From `https://github.com/EpicGames/lore`:

### 1.1 What Lore Is

> "Lore is an open source version control system designed for unprecedented scalability of both data and teams. It is optimized for projects that combine code with large binary assets, including games and entertainment."

It is **centralized**, **content-addressed**, **chunked**, **on-demand hydrated**, with **lightweight branches** over an **immutable revision chain**. Built-in VCS for UEFN. SDK surface in C/C++/C#/Rust/Go/Python/JS.

### 1.2 The Six Transferable Mechanisms

| # | Lore Mechanism | Our Distillation |
|---|---|---|
| 1 | **Content-addressed storage** — data referenced by content hash in a Merkle tree | Direct mapping. Use BLAKE3 (Lore uses its own hash; we already use BLAKE3 per AGENTS.md). |
| 2 | **Immutable revision chain** — revision hash derived from parent + contained data hashes | Subsumed by our existing flat BLAKE3 `SyncBlock.commitment` chain (riir-chain). The chunked layer is new. |
| 3 | **Chunked storage for large files** — files stored as reusable chunks with indexed lookup, enabling dedup + efficient transfer | **THE missing primitive.** Currently we have flat BLAKE3 over whole blobs (`NeuronShard`, `SenseModule`, `LatentPatch`). Chunking enables (a) cross-blob dedup (two swords share textures), (b) incremental asset push (only changed chunks), (c) partial hydration (load only visible chunks). |
| 4 | **On-demand hydration / sparse workspaces** — fetch data only when needed | Pluggable `ChunkFetcher` trait. Game client only pulls chunks for the current zone + visible LOD. |
| 5 | **Centralized service with caching** in front of durable storage | Out of scope for the engine primitive — that's a deployment concern. The trait is `ChunkFetcher` and the deploy can be S3, IPFS, riir-chain Cold tier, or a Lore server. |
| 6 | **Lightweight branches as mutable refs** over immutable data | **Branches are the user's gitflow insight** — but branch-as-subnet mapping is private (see Research 139). The engine primitive only needs: a `BlobId` is content-addressed; a `Ref` is a mutable named pointer to a `BlobId` or `TreeId`. |

### 1.3 What Lore Is NOT (Important Negative Findings)

- **Not P2P / not blockchain.** Lore is centralized with a server. Our fusion (Research 139) makes the storage layer **distributed and quorum-verified** via riir-chain — that's the novelty beyond Lore.
- **Not a runtime.** Lore is file-system VCS. Our fusion treats assets as **latent-loaded pods** (WASM-as-asset vessels via Plan 306), bypassing the file system entirely for hot paths.
- **Not deterministic-by-construction.** Lore is git-like; we add deterministic-replay compatibility (chunk hashes must round-trip bit-identical across nodes — required by riir-armageddon raw sync rules).
- **Not engine-aware.** Lore doesn't know about LOD, fog-of-war, or hot-swap. Our fusion adds engine-aware hydration triggers.

---

## 2. Distillation

### 2.1 The Primitive: `ChunkedContentStore`

```rust
/// A content-addressed chunk store with Merkle dedup.
///
/// Generic over the chunk-fetching strategy (filesystem, network, in-memory).
/// No game semantics, no chain, no consensus. Pure data structure.
///
/// Inspired by Epic Games Lore's chunked storage model, distilled to the
/// modelless primitive: chunk → BLAKE3 → dedup → Merkle root → inclusion proof.
pub trait ChunkedContentStore {
    /// Put a blob into the store. Returns the content-addressed BlobId
    /// (Merkle root over the blob's chunk hashes).
    ///
    /// Idempotent: putting the same bytes always returns the same BlobId.
    /// Dedup: chunks already in the store are not re-stored.
    fn put(&self, bytes: &[u8]) -> BlobId;

    /// Get a blob by its BlobId. Hydrates only the chunks not already cached
    /// locally. Caller provides the fetcher for remote chunks.
    fn get(&self, id: &BlobId, fetcher: &dyn ChunkFetcher) -> Vec<u8>;

    /// Get a single chunk (for sparse/partial reads — e.g. one LOD level).
    fn get_chunk(&self, chunk_hash: &[u8; 32], fetcher: &dyn ChunkFetcher) -> &[u8];

    /// Prove that a chunk is part of a blob. O(log n) siblings, pure BLAKE3.
    fn prove_chunk(&self, id: &BlobId, chunk_index: usize) -> Option<MerkleProof>;

    /// Verify a chunk inclusion proof against a known BlobId. Pure BLAKE3,
    /// no store access needed (light client pattern).
    fn verify_proof(proof: &MerkleProof) -> bool;

    /// Stat: how many chunks are stored, dedup ratio, total bytes.
    fn stats(&self) -> StoreStats;
}

/// A blob's content-addressed identity = Merkle root over its chunk hashes.
/// 32 bytes. Two blobs with identical bytes always share a BlobId.
/// Two blobs that share chunks (e.g. two swords with the same texture)
/// share chunk hashes but differ in BlobId.
#[repr(transparent)]
pub struct BlobId(pub [u8; 32]);

/// Strategy for fetching chunks not present in the local store.
/// Implementations: `FsChunkFetcher`, `NetChunkFetcher`, `InMemoryChunkFetcher`.
/// Game/chain layer provides the actual transport.
pub trait ChunkFetcher {
    fn fetch(&self, chunk_hash: &[u8; 32]) -> Option<Vec<u8>>;
}

/// Chunking strategy. Content-defined chunking (CDC) gives stable boundaries
/// across similar blobs (insertions/deletions only change local chunks).
/// Fixed-size chunking is simpler and faster for known-shape blobs.
pub trait ChunkingStrategy {
    /// Split bytes into chunks. Borrowed slices, zero-copy on the read path.
    fn chunk<'a>(&self, bytes: &'a [u8]) -> Vec<&'a [u8]>;
}

pub struct StoreStats {
    pub n_chunks_stored: u64,
    pub n_blobs_indexed: u64,
    pub total_bytes_stored: u64,
    pub total_bytes_logical: u64,
    pub dedup_ratio: f32,  // logical / stored
}
```

### 2.2 Why This Is Pure Modelless Engine (MIT)

- **No game types.** `BlobId`, `ChunkFetcher`, `ChunkingStrategy` — all domain-agnostic. The store doesn't know if a blob is a `.glb`, a `.wasm`, a `SenseModule`, or a `NeuronShard`.
- **No chain dependency.** The store is a data structure. Whether blobs are gossiped, quorum-committed, or stored on S3 is the deployer's choice.
- **No LLM, no training.** Pure data plumbing.
- **Reuses existing infrastructure.** BLAKE3 (already everywhere), `MerkleProof` (from Plan 253 / Research 221), `bytemuck::Pod` patterns.
- **Zero-allocation hot path.** `get_chunk` returns `&[u8]` (borrowed); `chunk` returns borrowed slices; only `get` materializes (because hydration requires it).

### 2.3 Chunking Strategy Choice

| Strategy | When | Why |
|---|---|---|
| **Fixed-size** (e.g. 64 KiB) | Known-shape blobs: `SenseModule` (232B), `LatentPatch` (68B), small WASM pods | Trivial, fast, deterministic. Bad for similar large blobs (one byte change rewrites the whole tail). |
| **Content-defined (CDC)** — Rabin-Karp / FastCDC / Buzhash rolling hash | Large mutable blobs: 3D geometry, weight shards, audio | Insertions/deletions only change local chunks. Two sword variants share 95% of chunks → 20× dedup. This is what Lore uses. |
| **Domain-aware** (caller splits on schema boundaries) | Structured multi-section blobs: `MerkleFrozenEnvelope` (header + payload) | Each section gets its own Merkle subtree. Allows "verify header without downloading payload." |

Default: **CDC for >64 KiB blobs, fixed-size for ≤64 KiB.** Caller can override via `ChunkingStrategy`.

### 2.4 Merkle Tree Shape

Reuse the existing `MerkleOctree` (binary variant — depth = ⌈log₂(n_chunks)⌉) from Plan 253. Leaves = chunk hashes; internal = `BLAKE3(left ‖ right)`; root = `BlobId`. This gives us:

- O(log n) inclusion proofs (a chunk is in a blob)
- O(log n) exclusion proofs (a chunk is NOT in a blob — useful for fraud proofs)
- Tamper detection (any bit flip changes the root)
- Cross-blob dedup (same chunk → same hash → stored once)
- Light-client verification (verify a chunk without downloading the blob)

**Critical:** the Merkle tree is **binary**, not octree (octree is for spatial KG triples; chunk blobs are linear). Use the binary mode of the existing `MerkleOctree` infrastructure (Plan 253 spec already supports both via the `MerkleProof.siblings` Vec).

### 2.5 Reference Implementation Sketch

```rust
pub struct InMemoryChunkedStore {
    chunks: papaya::HashMap<[u8; 32], Vec<u8>>,  // lock-free per AGENTS.md
    blobs: papaya::HashMap<[u8; 32], BlobMetadata>,
    chunker: FastCdcChunker,
}

struct BlobMetadata {
    n_chunks: u32,
    chunk_hashes: Box<[[u8; 32]]>,  // ordered, for proof generation
    total_bytes: u64,
}

impl ChunkedContentStore for InMemoryChunkedStore {
    fn put(&self, bytes: &[u8]) -> BlobId {
        let chunks: Vec<&[u8]> = self.chunker.chunk(bytes);
        let mut hashes: Vec<[u8; 32]> = Vec::with_capacity(chunks.len());
        for c in &chunks {
            let h = blake3::hash(c).into();
            hashes.push(h);
            // Dedup: only insert if not present (lock-free upsert).
            self.chunks.entry(h).or_insert_with(|| c.to_vec());
        }
        let root = build_binary_merkle_root(&hashes);
        let id = BlobId(root);
        self.blobs.insert(id.0, BlobMetadata {
            n_chunks: chunks.len() as u32,
            chunk_hashes: hashes.into_boxed_slice(),
            total_bytes: bytes.len() as u64,
        });
        id
    }
    // ...
}
```

**Note on lock-free:** per AGENTS.md (`papaya as possible for lock-free Arc<RwLock<HashMap>>`), use `papaya::HashMap` for the chunk index. Hot path is `get_chunk` — read-only, lock-free, returns `&[u8]` from the hashmap's value.

---

## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars). Creates a moat. | Open primitive → katgpt-rs. **Architectural guide → riir-ai/.research/**. Plans → both repos as needed. |
| **GOAT** | Provable gain (latency/quality/security) over existing approach, but not a new class of capability. Promotes to default if it wins. | Plan + implement → appropriate repo. Feature flag + benchmark. |
| **Gain** | Incremental improvement, useful but not headline-worthy. | Plan only, behind feature flag. |
| **Pass** | Not relevant to modelless/latent/freeze-thaw/runtime, OR training-only. | One-line note. No files created in this session. |

**Verdict: GOAT (for the open primitive in katgpt-rs). Super-GOAT for the cross-repo fusion → handled in riir-ai/.research/139.**

**Reasoning (one-line each):**

- **Why GOAT here, not Super-GOAT, for the open primitive alone:** Chunked content-addressed Merkle stores are well-known prior art (Lore itself is public MIT; Git-LFS, IPFS, Perforce, Casync all do variants). The novelty is **not** in the data structure — it's in the **fusion** with quorum-gated gitflow subnets, executable WASM asset vessels, and atomic candidate-lock transactions. That fusion is private (riir-ai 139). The open primitive alone is a **provable perf + dedup gain** over our current flat BLAKE3, which is GOAT-tier.
- **Why GOAT and not Gain:** Provable wins on (a) cross-blob dedup (swords share textures → ~5-20× storage reduction), (b) incremental asset push (only changed chunks → ~10-50× bandwidth reduction on hot-swap), (c) O(log n) inclusion proofs (light clients, fraud proofs). Not headline-worthy alone, but materially better than the flat-hash status quo.
- **Why feature-gated, not default-on until GOAT proof:** The current flat BLAKE3 path is correct and shipped. Chunking is an optimization. Per AGENTS.md: prove the gain, then promote. Feature flag `chunked_content_store`, default-off, promote on GOAT pass.
- **The Super-GOAT moat is in riir-ai 139:** Lore distillation × Plan 306 WASM-Unity bridge × Plan 281 node tiers × Plan 109 upgradeable programs × Plan 309 dual-signal gate × Research 196 WASM pods. That fusion is a new capability class ("release pipeline as chain state with race-condition-free shop equip and engine-agnostic dual delivery") and is the private selling point.

---

## 4. Connection Map

```
ChunkedContentStore (THIS — katgpt-rs 262)
    │
    ├── Reuses: MerkleOctree / MerkleProof (katgpt-rs 221, Plan 253)
    │   └── Binary Merkle mode for linear chunk blobs
    │
    ├── Reuses: BLAKE3 (already everywhere)
    │
    ├── Reuses: papaya::HashMap (lock-free chunk index, per AGENTS.md)
    │
    ├── Feeds: NeuronShard, SenseModule, LatentPatch (existing flat-BLAKE3 types)
    │   └── Becomes their backing store when feature enabled
    │
    └── Consumed by (PRIVATE — riir-ai):
        ├── Research 139: Lore Distillation + Executable Asset Vessel + Gitflow
        ├── Plan 319: Executable Asset Vessel + Quorum Gitflow (private plan)
        ├── Plan 306: Latent(WASM)-FFI-Unity Byte Bridge (loads .wasm from this store)
        ├── Plan 281: Node-Tier Curator Consensus (verifies chunk Merkle roots)
        ├── Plan 109: Upgradeable Programs (GM signs new WASM blob → this store)
        └── Plan 224: GM MCP Entity Management (AssetPromote instruction)
```

---

## 5. Latent vs Raw Boundary

**Critical for riir-armageddon compliance.** The chunked content store is **content-addressed bytes**. What those bytes encode is the caller's responsibility.

| Blob type | Domain | Boundary rule |
|---|---|---|
| 3D geometry (`.glb`, `.wasm` geometry pods) | **Raw** — physics/anti-cheat may need exact vertex data | Chunks are raw bytes; chunk Merkle root is the canonical identity. Sync is bit-identical across nodes (deterministic replay compatible). |
| Weight shards (`NeuronShard`, `LatentPatch`) | **Latent** — model weights | Chunks are latent bytes; root is the weight manifold commitment. Stays latent; bridge projects to 5 scalars at sync boundary (per AGENTS.md). |
| KG octree pods (`SenseModule`) | **Latent** — KG embeddings | Same as weight shards. |
| Audio / texture | **Raw** — game assets | Chunks are raw; identity is content hash. |
| WASM modules (validators, gate, compute pods) | **Raw** (code) — but produces latent outputs | Chunks are the WASM bytes; execution produces latent state. Module identity = chunk Merkle root. Runtime verifies via Plan 109 curator consensus. |

**Bridge rule (per AGENTS.md):** the store NEVER does latent↔raw projection. It only stores and proofs bytes. Projection happens in riir-engine (latent→5 scalars) and riir-chain (raw commitment). The store is plumbing.

---

## 6. Validation Protocol (GOAT Gate for the open primitive)

The open primitive is GOAT-tier — prove the perf/dedup gain before promoting to default.

| Gate | Target | How |
|---|---|---|
| **G1 — Dedup ratio** | ≥ 5× on synthetic "100 sword variants sharing a texture" workload | Generate 100 1 MiB blobs with 90% shared chunks → measure `dedup_ratio`. Pass if ≥ 5.0. |
| **G2 — Incremental push** | ≤ 5% of bytes pushed on 1-byte change to a 10 MiB blob (CDC) | Mutate 1 byte, re-chunk, count new chunk bytes. Pass if ≤ 5% (CDC) and ≈100% for fixed-size (negative control). |
| **G3 — Inclusion proof cost** | < 10 µs per proof for ≤ 1024 chunks | Bench `prove_chunk` and `verify_proof`. Pass if mean < 10 µs at n=1024. |
| **G4 — Light-client verify** | Pure BLAKE3, no store access | `verify_proof` reads only the proof + the chunk being proven. Grep: no `chunks.get()` in the verify path. |
| **G5 — Hot-path read latency** | `get_chunk` < 200 ns (papaya lock-free read) | Bench 1M reads against a 10K-chunk store. Pass if p99 < 200 ns. |
| **G6 — Default-off regression** | Feature off → zero overhead on existing flat BLAKE3 path | Run existing tests with feature off; assert byte-identical output and no perf regression. |
| **G7 — Tamper detection** | Any single-bit flip in any chunk → BlobId mismatch | Fuzz: flip 1 bit in each of 10K chunks; assert 10K/10K mismatches. |

If G1–G7 pass → promote `chunked_content_store` to default feature. If G1 fails (dedup < 5×) → keep opt-in; the gain isn't material enough.

---

## 7. What Stays Private vs Open

| Aspect | Public (katgpt-rs) | Private (riir-ai) |
|---|---|---|
| `ChunkedContentStore` trait + reference impl | ✅ Open | — |
| `BlobId`, `ChunkFetcher`, `ChunkingStrategy` traits | ✅ Open | — |
| `FastCdcChunker` (Rabin-Fingerprint CDC) | ✅ Open | — |
| `FsChunkFetcher`, `InMemoryChunkFetcher` | ✅ Open | — |
| Binary Merkle tree reuse from Plan 253 | ✅ Open | — |
| GOAT gate benchmarks | ✅ Open | — |
| Game-specific blob types (ItemAsset, NPCAppearanceAsset) | — | ✅ Private |
| Engine-aware hydration triggers (zone-LOD, fog-of-war) | — | ✅ Private |
| Subnet-as-gitflow-branch mapping | — | ✅ Private (Research 139) |
| Asset-Candidate Quorum Lock (atomic InstallAsset+UnlockShop) | — | ✅ Private (Plan 319) |
| Cross-subnet promotion protocol | — | ✅ Private (Plan 319) |
| WASM-as-Asset vessel format (executable geometry pod) | — | ✅ Private (Plan 319, builds on Plan 306) |
| Two-way hybrid delivery (VFS for raw + WASM for secured) | — | ✅ Private (Plan 319) |
| NFT binding (asset OID → riir-chain token matrix) | — | ✅ Private (Plan 319) |

---

## 8. Why This Is Not a Lore Clone (Important Distinction)

Lore is **centralized file-system VCS**. The open primitive here is:

1. **A data structure, not a VCS.** No `commit`, `branch`, `merge`, `pull`, `push` operations — those are deployment choices. We expose `put`, `get`, `prove`, `verify`.
2. **Backend-agnostic.** Lore has a server. We have a `ChunkFetcher` trait. The deploy chooses: filesystem, S3, IPFS, riir-chain Cold tier, or a Lore server.
3. **Merkle-proof-native.** Lore's value prop is large-team scalability; ours is **light-client verifiability** (curators, browsers, anti-cheat can verify a chunk without downloading the blob).
4. **Engine-integrated.** Lore is generic. Ours is designed to back `NeuronShard`, `SenseModule`, `LatentPatch` (latent) and 3D/audio/texture (raw) with the same API, then get fused with WASM-asset vessels and quorum gitflow (riir-ai 139).

The public primitive is the **dedup-chunked Merkle store**. Lore's contribution to our thinking is the chunking model. Everything else (executable assets, gitflow subnets, candidate locks) is our private fusion.

---

## 9. Implementation Priority (Open Primitive Only)

| Priority | Task | Why |
|---|---|---|
| **P0** | Trait + types (`BlobId`, `ChunkFetcher`, `ChunkingStrategy`) | Foundation. No deps. |
| **P0** | `FixedSizeChunker` + `InMemoryChunkedStore` | Simplest reference impl. Tests G3, G4, G6. |
| **P1** | `FastCdcChunker` (Rabin-Karp rolling hash) | Required for G1, G2 (dedup, incremental push). |
| **P1** | `FsChunkFetcher` (filesystem-backed) | Realistic deployment. |
| **P2** | Binary Merkle tree reuse from Plan 253 | Required for proofs (G3, G4, G7). |
| **P2** | GOAT gate benchmark suite (G1–G7) | Required before promoting to default. |
| **P3** | `NetChunkFetcher` (HTTP/IPFS pluggable transport) | Optional; deployers can implement the trait. |

See `katgpt-rs/.plans/272_chunked_asset_merkle_store.md` for the execution plan.

---

## TL;DR

Epic's Lore gives us the **chunked content-addressed Merkle store** primitive — large binary blobs split into content-defined chunks, BLAKE3-hashed for dedup, Merkle-rooted for inclusion proofs, lazily hydrated via a pluggable fetcher. The open primitive in katgpt-rs is **GOAT-tier** (provable dedup + bandwidth + light-client wins over flat BLAKE3, not Super-GOAT alone because the data structure is well-known prior art). The **Super-GOAT is the fusion** with quorum-gated gitflow subnets, executable WASM asset vessels, and atomic candidate-lock transactions — that lives in riir-ai/.research/139. The open primitive is the adoption hook (any Rust project can use it); the private fusion is the moat. Feature flag `chunked_content_store`, default-off, GOAT gate G1–G7 before promotion. Reuses `MerkleOctree` (Plan 253), BLAKE3 (existing), papaya (per AGENTS.md). No new external dependencies.
