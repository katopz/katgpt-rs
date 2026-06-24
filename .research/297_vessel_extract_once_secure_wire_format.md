# Research 297: Vessel — Extract-Once Secure Wire Format

> **Source:** Engineering synthesis — `AssetVesselSidecar` (riir-ffi, Plan 319) × `MerkleFrozenEnvelope` (riir-neuron-db) × `DataTier` (riir-chain) × user egg/shell proposal (2026-06-24)
> **Date:** 2026-06-24
> **Status:** Done — verdict locked
> **Related Research:** 262 (Lore chunked asset store), 268 (Forensic asset fingerprinting), 296 (Stokes/DEC vocabulary — irrelevant here)
> **Related Plans:** 272 (chunked asset Merkle store), 315 (this primitive)
> **Cross-ref (riir-neuron-db):** Research 006 (`NeuronVesselSidecar` private Super-GOAT guide), Plan 003 (impl)
> **Classification:** Public

---

## TL;DR

A **generic open primitive** for "secure wire format → validated extract-once → tier-aware raw access". WASM is used as the wire/distribution format (BLAKE3 in header, capability-restricted imports, optional payload-hiding via API-only exports). On load, the WASM is validated once and the Pod payload is extracted to host memory for zero-copy SIMD access thereafter. Cold/Freeze tiers can keep the payload inside the WASM and call exported projection functions instead — paying the WASM call cost only when memory pressure or distribution security demands it.

**Distilled for katgpt-rs (modelless, inference-time):**

The transferable primitive is the **decode-once-run-many + tier-aware projection routing** pattern, stripped of all shard/asset/game semantics. It is the same shape as `MerkleFrozenEnvelope` (validate-then-use) but adds: (1) WASM as the wire container, (2) an extract step that yields a raw Pod for SIMD, (3) a fallback projector that calls into the WASM when extraction is not allowed (Cold/Freeze).

---

## 1. Core Findings (Engineering Synthesis)

### 1.1 The pattern already ships — for 3D assets

`AssetVesselSidecar` (`riir-ai/crates/riir-ffi/src/asset_vessel_sidecar.rs`) ships the egg/shell pattern for 3D geometry:
- WASM module owns its linear memory (vertex/index/material/skeleton buffers).
- Host reads `AssetViewState` zero-copy via `read_view`.
- `verify_owner(owner_pubkey, &dyn NftOwnerVerifier)` implements G8 (NFT execute-permission) — "stolen bytes can't run".
- Content-addressed via `AssetBlobId = [u8; 32]` (chunk Merkle root, Plan 272).
- `riir-chain/src/asset_lifecycle/delivery.rs::hydrate_asset` dispatches on `AssetDeliveryKind::WasmVessel` vs `VfsSigned` vs `Hybrid`.

### 1.2 The integrity shell ships — `MerkleFrozenEnvelope`

`riir-neuron-db/src/freeze.rs`:
```rust
pub const FREEZE_MAGIC: [u8; 4] = [b'F', b'R', b'Z', b'E'];
pub struct MerkleFrozenEnvelope {
    pub magic: [u8; 4],
    pub version: u32,
    pub merkle_root: [u8; 32],
    pub data_len: u64,
    pub envelope_commitment: [u8; 32],  // BLAKE3 over header
}
```
Magic-prefix + version + Merkle root + BLAKE3 envelope commitment — this is the canonical header pattern in the codebase (also see `CGSP`, `BDTB`, `COLP`, `DRMR`, `GODT`, `GOTM`, `CERT`, `AV01`).

### 1.3 The tier system ships — `DataTier`

`riir-chain/src/catchup/merkle.rs`:
```rust
pub enum DataTier { Plasma = 0, Hot = 1, Warm = 2, Cold = 3, Freeze = 4 }
pub fn build_tier_root(commitments: &[[u8; 32]]) -> [u8; 32]
pub fn build_block_root(tier_roots: &[[u8; 32]; 5]) -> [u8; 32]
```
Per-tier Merkle roots already committed in every block. The tier axis is real.

### 1.4 The cold-tier compaction ships — `ShardCompactor`

`riir-neuron-db/src/shard_compactor.rs` — GOAT gates G1 (≥95% fidelity), G2 (deterministic commitment), G3 (≥10× footprint reduction), wallet-inviolability all enforced. `compact_for_cold_tier` is the existing tier-transition function.

### 1.5 The gap

No primitive unifies these: there is no generic "WASM vessel → extract Pod → tier-aware projection" loader. `AssetVesselSidecar` does it for 3D specifically; `MerkleFrozenEnvelope` does integrity but not encapsulation; `DataTier` is a label, not a loader. This primitive fills the gap.

---

## 2. Distillation

### 2.1 Vocabulary translation

| User / proposal term | Codebase equivalent |
|---|---|
| "secure wire format" | `AssetDeliveryKind::WasmVessel`, `LoadedVessel`, WASM custom section BLAKE3 |
| "egg/shell" | `AssetVesselSidecar` (shell) + `AssetViewState` (egg) |
| "extract once" | `extract_payload::<T: Pod>()` — one-time WASM validate, then raw bytes |
| "GC + reload via AOI" | `ShardIndex` (zone→shard, papaya lock-free) + LRU eviction + `DataTier::Warm` |
| "load once, zero-copy" | `mmap` + `#[repr(C)]` Pod + `simd_dot_f32` |
| "secure, bc bin too exposed" | API encapsulation + capability imports + soft obfuscation (NOT cryptographic — see §2.4) |

### 2.2 Closest prior art (BOTH layers, ALL repos)

| Layer | Artifact | Match |
|---|---|---|
| Code | `riir-ai/crates/riir-ffi/src/asset_vessel_sidecar.rs` | **Closest** — egg/shell for 3D, NFT-gated |
| Code | `riir-neuron-db/src/freeze.rs::MerkleFrozenEnvelope` | Integrity shell (BLAKE3 + Merkle) |
| Code | `riir-chain/src/asset_lifecycle/delivery.rs::hydrate_asset` | Tier dispatch on `AssetDeliveryKind` |
| Code | `riir-chain/src/catchup/merkle.rs::DataTier` | Tier label system |
| Code | `riir-neuron-db/src/shard_compactor.rs` | Tier-transition (compact_for_cold_tier) |
| Code | `riir-ai/crates/riir-ffi/src/latent_sidecar.rs` | Loads WASM compute modules (no extract-once) |
| Notes | `riir-ai/.research/020_Zone_Expert_Bundles_Living_World.md` | Per-zone `.bin` + `.wasm` pair (the closest prior framing) |
| Notes | `riir-ai/.plans/306_latent_wasm_ffi_unity_byte_bridge.md` | Latent↔WASM bridge context |
| Notes | `riir-ai/.plans/307_gate_wasm_tiered_node_defense.md` | Tiered WASM node defense |
| Notes | `riir-ai/.plans/319_executable_asset_vessel_quorum_gitflow.md` | Asset vessel origin plan |

### 2.3 Fusion

> **Vessel Extract-Once Primitive** — fuse three shipped patterns into one generic primitive:
> - **From `AssetVesselSidecar`**: WASM-as-shell + `verify_owner` NFT gating + content-addressed `AssetBlobId`.
> - **From `MerkleFrozenEnvelope`**: magic-prefix header + BLAKE3 commitment + version field.
> - **From `ShardCompactor`**: tier-transition is explicit and gated (Hot→Cold has a fidelity gate).
>
> The novel combination: a single trait that exposes **two projection paths** (raw Pod extract OR WASM call) selected by `DataTier`, with extract-once semantics so the WASM validation cost is paid once and amortized over many hot-path operations.

### 2.4 Honest security framing (CRITICAL — do not oversell)

**WASM is NOT cryptographic confidentiality.** A determined attacker with host debugger access can dump WASM linear memory and extract the Pod bytes. What WASM-vessel actually provides:

| Property | Strength | Source |
|---|---|---|
| Integrity (tamper detection) | **Cryptographic** | BLAKE3 in header + Merkle root |
| API encapsulation | **Strong** | WASM only exports what it declares |
| Capability security | **Strong** | Host grants/revokes imports (no I/O by default) |
| NFT execute-permission | **Strong** | `verify_owner` before any projection |
| Soft obfuscation (anti-casual-extract) | **Weak** | Hexdump doesn't reveal floats; debugger defeats it |

**True cryptographic confidentiality would require FHE dot-product or TEE (SGX/SEV).** Those are out of scope for this primitive. The honest selling point is: **API-level access control + chain-committed projection integrity** (via BLAKE3 + LatCal), not "weights are encrypted".

The cryptographic angle that DOES hold: if the WASM bytecode is BLAKE3-committed and the projection result is LatCal-committed, a verifier node can prove "this projection was computed by THIS bytecode" without seeing the weights. **Integrity without confidentiality** — and integrity is what the chain needs for consensus.

---

## 3. Verdict

**Tier: Super-GOAT (open primitive half)**

| Criterion | Result |
|---|---|
| Q1 No prior art | ✅ No generic `Vessel`/`extract_once` primitive ships. `AssetVesselSidecar` is 3D-specific. |
| Q2 New class | ✅ Tier-aware dual-path projection (raw OR WASM-call) is new. |
| Q3 Selling point | ✅ (in the riir-neuron-db guide — see Research 006) |
| Q4 Force multiplier | ✅ Connects 4 systems (see connection map in guide). |

**One-line reasoning:** The primitive generalizes the shipped `AssetVesselSidecar` egg/shell pattern into a tier-aware loader trait usable by any Pod type, with an honest security model (API encapsulation + chain-committed integrity, NOT cryptographic confidentiality).

---

## 4. Routing

- **Open primitive** → `katgpt-rs/src/vessel/` (new module, `secure_vessel` feature). See Plan 315.
- **Private Super-GOAT guide** → `riir-neuron-db/.research/006_neuron_vessel_tiered_secure_distribution_guide.md` (the selling point + tier table + validation protocol).
- **Private impl plan** → `riir-neuron-db/.plans/003_neuron_vessel_sidecar.md`.
- **Chain wiring** (brief, inside the guide): extend `AssetDeliveryKind` to add `NeuronVessel` discriminant OR reuse `WasmVessel` with a `payload_kind` field — see guide §implementation-priority.
- **Runtime wiring** (brief, inside the guide): `SenseModule` / HLA projection path selects raw Pod (Hot/Plasma) vs vessel-project (Cold/Freeze) based on `DataTier`.

---

## TL;DR

Generic open primitive: a `Vessel` wire format (WASM + BLAKE3 header + payload-offset) with two projection paths — `extract_payload::<T: Pod>()` for Hot/Plasma (zero-copy SIMD after one-time validate) and `VesselProjector::project()` for Cold/Freeze (capability-restricted WASM call). Fuses `AssetVesselSidecar` × `MerkleFrozenEnvelope` × `ShardCompactor`. Honest security: API encapsulation + chain-committed integrity, NOT cryptographic confidentiality — but combined with LatCal committed projections, gives verifiable projection integrity without weight exposure.
