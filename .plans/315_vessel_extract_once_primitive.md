# Plan 315: Vessel — Extract-Once Secure Wire Format Primitive

**Date:** 2026-06-24
**Research:** [katgpt-rs/.research/297_vessel_extract_once_secure_wire_format.md](../.research/297_vessel_extract_once_secure_wire_format.md)
**Cross-ref (riir-neuron-db):** [Research 006](../../riir-neuron-db/.research/006_neuron_vessel_tiered_secure_distribution_guide.md), [Plan 003](../../riir-neuron-db/.plans/003_neuron_vessel_sidecar.md)
**Target:** `katgpt-rs/src/vessel/` (new module) + Cargo feature `secure_vessel`
**Status:** Active — Phase 1 unblocking

---

## Goal

Ship the generic open half of the Super-GOAT from Research 297 / 006: a `Vessel` wire format (WASM + BLAKE3 header + payload offset) and a tier-aware loader trait with two projection paths — `extract_payload::<T: Pod>()` (one-time validate, raw bytes for SIMD) and `VesselProjector::project()` (capability-restricted WASM call). No shard/game/chain semantics — those land in riir-neuron-db Plan 003.

This primitive is the public adoption hook; the private selling-point guide lives in riir-neuron-db. **Honest scope:** API encapsulation + integrity, NOT cryptographic confidentiality (see Research 297 §2.4).

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/src/vessel/mod.rs` behind `secure_vessel` feature (off by default).
  - `pub const VESSEL_MAGIC: [u8; 4] = *b"VSL1";`
  - `pub const VESSEL_VERSION: u32 = 1;`
  - `VesselHeader` `#[repr(C)]` struct: `magic[4]`, `version: u32`, `blake3[32]`, `payload_kind: u32`, `payload_offset: u32`, `payload_len: u32` (52 bytes total — matches the canonical header pattern: `FREEZE_MAGIC`, `CGSP`, `BDTB`, etc.).
- [ ] **T1.2** `VesselError` enum: `BadMagic`, `UnsupportedVersion`, `Blake3Mismatch`, `PayloadTooShort`, `WasmiCompile(wasmi::Error)`, `WasmiInstantiate(wasmi::Error)`, `ExportMissing(&'static str)`.
- [ ] **T1.3** `encode_vessel(wasm_bytes: &[u8], payload_kind: u32, payload_offset: u32, payload_len: u32) -> Vec<u8>` — prepends header, BLAKE3 over WASM bytes only.
- [ ] **T1.4** `decode_header(bytes: &[u8]) -> Result<VesselHeader, VesselError>` — validates magic + version; does NOT verify BLAKE3 yet (caller decides when).
- [ ] **T1.5** `verify_blake3(header: &VesselHeader, wasm_bytes: &[u8]) -> Result<(), VesselError>` — standalone so callers can batch.
- [ ] **T1.6** Cargo.toml: `secure_vessel = ["wasmi", "blake3", "papaya"]` (re-uses existing deps, no new ones).

## Phase 2 — Extract-Once Path (Hot/Plasma tier)

### Tasks

- [ ] **T2.1** `LoadedVessel` struct: `{ header: VesselHeader, wasm_bytes: Arc<[u8]>, instance: Option<wasmi::Instance> }` (instance lazily compiled — extract path doesn't need it).
- [ ] **T2.2** `load_vessel(bytes: &[u8]) -> Result<LoadedVessel, VesselError>` — decodes header + verifies BLAKE3 + stores wasm_bytes (Arc, zero-clone).
- [ ] **T2.3** `extract_payload<T: bytemuck::Pod>(vessel: &LoadedVessel) -> Result<&T, VesselError>` — **the core primitive.** Validates `payload_len == size_of::<T>()`, returns `bytemuck::from_bytes(&vessel.wasm_bytes[payload_offset..payload_offset+payload_len])`. Zero-copy, zero-alloc. Caller is responsible for keeping `vessel` alive.
- [ ] **T2.4** `extract_payload_slice<T: Pod>(vessel: &LoadedVessel) -> Result<&[T], VesselError>` — variable-length variant for arrays.
- [ ] **T2.5** Tests:
  - `extract_returns_byte_identical_payload` — round-trip encode/decode/extract.
  - `extract_rejects_bad_magic` / `_bad_version` / `_bad_blake3`.
  - `extract_rejects_payload_len_mismatch`.
  - `extract_zero_alloc` — assert no allocation in the extract hot path (use `#[bench]` or manual `dhat`).

## Phase 3 — Vessel Projector Path (Cold/Freeze tier)

### Tasks

- [ ] **T3.1** `VesselProjector` trait:
  ```rust
  pub trait VesselProjector {
      type Query;
      type Output;
      fn project(&self, vessel: &LoadedVessel, query: &Self::Query) -> Result<Self::Output, VesselError>;
  }
  ```
- [ ] **T3.2** `ensure_compiled(vessel: &mut LoadedVessel, store: &wasmi::Store<()>) -> Result<&wasmi::Instance, VesselError>` — lazy wasmi compile, cached in `LoadedVessel.instance`. Fuel-gated.
- [ ] **T3.3** Generic `WasmDotProjector { export_name: &'static str }` impl: looks up `export_name` in the instance, calls it with the query pointer, returns the scalar.
- [ ] **T3.4** Tests:
  - `project_calls_exported_function` — fake WASM module with `project` export returning constant.
  - `project_rejects_missing_export`.
  - `project_fuel_exhaustion_returns_error` (fail-safe, never panics).

## Phase 4 — GOAT Gate (G1-G5 subset, vessel-level)

The shard-level gates G6-G8 live in riir-neuron-db Plan 003. This plan owns G1-G5 generic.

### Tasks

- [ ] **T4.1** `cargo test -p katgpt-rs --features secure_vessel` — all Phase 2-3 tests pass.
- [ ] **T4.2** Bench `vessel_extract_latency` — measure `extract_payload::<[f32; 64]>()` cost. Target: < 50ns (single BLAKE3 verify is the dominant cost; payload slice is zero-copy).
- [ ] **T4.3** Bench `vessel_project_latency` — measure `WasmDotProjector::project()` cost. Document the ~100-500ns expected range.
- [ ] **T4.4** Write `.benchmarks/315_vessel_goat.md` with G1-G5 results + decision (promote to default-on if G1-G3 + G4 < 50ns + G5 < 1µs).
- [ ] **T4.5** If GOAT passes: add `secure_vessel` to `[features] default = [...]`. If fails: stays opt-in, document why in the bench file.

## Phase 5 — Examples + Docs

### Tasks

- [ ] **T5.1** `katgpt-rs/examples/vessel_minimal.rs` — encode/decode/extract round-trip with a fake `[u8; 64]` Pod payload.
- [ ] **T5.2** `katgpt-rs/examples/vessel_project.rs` — build a tiny WASM module with a `project` export (use `wat2wasm`), load as vessel, call projector.
- [ ] **T5.3** `katgpt-rs/.docs/15_vessel.md` — overview doc with the tier table from Research 006 §4.

---

## Anti-Goals

- ❌ No `NeuronShard` import — this primitive is Pod-generic. Shard-specific wrapper is riir-neuron-db Plan 003.
- ❌ No cryptographic confidentiality claims in docs — see Research 297 §2.4.
- ❌ No game/chain semantics — no `DataTier`, no AOI, no fog-of-war here. Those land in riir-ai / riir-chain plans.
- ❌ No network/distribution code — the vessel is a local byte-blob. Distribution is riir-chain's job (ChunkedContentStore).

## GOAT Gate Summary (predict)

| Gate | Test | Predicted | Decision rule |
|---|---|---|---|
| G1 extract fidelity | round-trip byte-identical | ✅ trivially passes (zero-copy slice) | must pass |
| G2 determinism | same bytes → same extract | ✅ passes | must pass |
| G3 projection parity | n/a at this layer (deferred to shard plan) | n/a | — |
| G4 extract latency | < 50ns | ✅ likely (BLAKE3 dominates) | promote if < 50ns |
| G5 project latency | < 1µs | ✅ likely (wasmi fuel-gated) | promote if < 1µs |

---

## TL;DR

Generic `Vessel` primitive in katgpt-rs: WASM-with-BLAKE3-header wire format + `extract_payload::<T: Pod>()` (zero-copy, Hot path) + `VesselProjector` trait (Cold path). Re-uses existing `wasmi` + `blake3` + `papaya` deps. Ships behind `secure_vessel` feature, GOAT-gated before default promotion. Private shard wrapper lives in riir-neuron-db Plan 003; private selling-point guide is riir-neuron-db Research 006.
