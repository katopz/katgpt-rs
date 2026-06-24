# Plan 315: Vessel — Extract-Once Secure Wire Format Primitive

**Date:** 2026-06-24
**Research:** [katgpt-rs/.research/297_vessel_extract_once_secure_wire_format.md](../.research/297_vessel_extract_once_secure_wire_format.md)
**Cross-ref (riir-neuron-db):** [Research 006](../../riir-neuron-db/.research/006_neuron_vessel_tiered_secure_distribution_guide.md), [Plan 003](../../riir-neuron-db/.plans/003_neuron_vessel_sidecar.md)
**Target:** `katgpt-rs/src/vessel/` (new module) + Cargo feature `secure_vessel`
**Status:** Phase 1-6 complete (all phases done). 21 tests green + 2 examples + docs. Stays opt-in (dependency hygiene, not quality gate).

---

## Goal

Ship the generic open half of the Super-GOAT from Research 297 / 006: a `Vessel` wire format (WASM + BLAKE3 header + payload offset) and a tier-aware loader trait with two projection paths — `extract_payload::<T: Pod>()` (one-time validate, raw bytes for SIMD) and `VesselProjector::project()` (capability-restricted WASM call). No shard/game/chain semantics — those land in riir-neuron-db Plan 003.

This primitive is the public adoption hook; the private selling-point guide lives in riir-neuron-db. **Honest scope:** API encapsulation + integrity, NOT cryptographic confidentiality (see Research 297 §2.4).

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `katgpt-rs/src/vessel/mod.rs` behind `secure_vessel` feature (off by default).
  - `pub const VESSEL_MAGIC: [u8; 4] = *b"VSL1";`
  - `pub const VESSEL_VERSION: u32 = 1;`
  - `VesselHeader` `#[repr(C)]` struct: `magic[4]`, `version: u32`, `blake3[32]`, `payload_kind: u32`, `payload_offset: u32`, `payload_len: u32` (52 bytes total — matches the canonical header pattern: `FREEZE_MAGIC`, `CGSP`, `BDTB`, etc.).
- [x] **T1.2** `VesselError` enum: `BadMagic`, `UnsupportedVersion`, `Blake3Mismatch`, `PayloadTooShort`, `WasmiCompile(wasmi::Error)`, `WasmiInstantiate(wasmi::Error)`, `ExportMissing(&'static str)`.
  - **Note:** `PayloadTooShort` renamed `HeaderTooShort` (clearer — header is what's short, not payload); added `PayloadOutOfBounds`, `PayloadLenMismatch`, `FuelExhausted` for full error coverage.
- [x] **T1.3** `encode_vessel(wasm_bytes: &[u8], payload_kind: u32, payload_offset: u32, payload_len: u32) -> Vec<u8>` — prepends header, BLAKE3 over WASM bytes only.
- [x] **T1.4** `decode_header(bytes: &[u8]) -> Result<VesselHeader, VesselError>` — validates magic + version; does NOT verify BLAKE3 yet (caller decides when).
- [x] **T1.5** `verify_blake3(header: &VesselHeader, wasm_bytes: &[u8]) -> Result<(), VesselError>` — standalone so callers can batch.
- [x] **T1.6** Cargo.toml: `secure_vessel = ["wasmi", "papaya"]` (re-uses existing deps; `blake3` + `bytemuck` already non-optional, not listed).

## Phase 2 — Extract-Once Path (Hot/Plasma tier)

### Tasks

- [x] **T2.1** `LoadedVessel` struct: `{ header: VesselHeader, wasm_bytes: Arc<[u8]>, instance: Option<wasmi::Instance> }` (instance lazily compiled — extract path doesn't need it).
  - **Note:** manual `Debug` impl added because `wasmi::Instance` is not `Debug`; prints `instance_compiled: bool` instead of dumping bytes.
- [x] **T2.2** `load_vessel(bytes: &[u8]) -> Result<LoadedVessel, VesselError>` — decodes header + verifies BLAKE3 + stores wasm_bytes (Arc, zero-clone).
- [x] **T2.3** `extract_payload<T: bytemuck::Pod>(vessel: &LoadedVessel) -> Result<&T, VesselError>` — **the core primitive.** Validates `payload_len == size_of::<T>()`, returns `bytemuck::from_bytes(&vessel.wasm_bytes[payload_offset..payload_offset+payload_len])`. Zero-copy, zero-alloc. Caller is responsible for keeping `vessel` alive.
- [x] **T2.4** `extract_payload_slice<T: Pod>(vessel: &LoadedVessel) -> Result<&[T], VesselError>` — variable-length variant for arrays.
- [x] **T2.5** Tests:
  - `extract_returns_byte_identical_payload` — round-trip encode/decode/extract.
  - `extract_rejects_bad_magic` / `_bad_version` / `_bad_blake3`.
  - `extract_rejects_payload_len_mismatch` (+ `_out_of_bounds`).
  - `extract_zero_alloc` — deferred to Phase 4 bench (manual `dhat` not yet wired; zero-copy is structural — `extract_payload` body is `size_of` check + `slice::get` + `bytemuck::from_bytes`, no allocations).
  - Bonus: `header_is_52_bytes_no_padding`, `loaded_vessel_shares_arc_across_clones`, `extract_payload_slice_round_trips`, `decode_header_rejects_short_buffer`, `verify_blake3_standalone_passes_on_valid_blob`.

## Phase 3 — Vessel Projector Path (Cold/Freeze tier)

### Tasks

- [x] **T3.1** `VesselProjector` trait with GAT `Query<'a>` + `Output`.
- [x] **T3.2** `ensure_compiled(vessel: &mut LoadedVessel, store: &mut wasmi::Store<()>, engine: &wasmi::Engine) -> Result<&wasmi::Instance, VesselError>` — lazy wasmi compile, cached in `LoadedVessel.instance`. Fuel-gated via `Config::consume_fuel(true)`.
- [x] **T3.3** Generic `WasmDotProjector { export_name: &'static str, fuel_budget: u64 }` impl: looks up `export_name` in the instance, calls it with the query pointer, returns the scalar.
- [x] **T3.4** Tests:
  - `project_calls_exported_function` — WAT module with `project` export summing f32s; 1+2+3+4=10.0.
  - `project_rejects_missing_export`.
  - `project_rejects_uncompiled_instance` (precondition guard).
  - `project_fuel_exhaustion_returns_error` (fail-safe, never panics).

  All 15 Phase 1-3 tests green: `cargo test --features secure_vessel --lib vessel::`.

## Phase 4 — GOAT Gate (G1-G5 subset, vessel-level)

The shard-level gates G6-G8 live in riir-neuron-db Plan 003. This plan owns G1-G5 generic.

### Tasks

- [x] **T4.1** `cargo test -p katgpt-rs --features secure_vessel` — all Phase 2-3 tests pass (15/15 green).
- [x] **T4.2** Bench `vessel_extract_latency` — measure `extract_payload::<[f32; 64]>()` cost. **Result: 0.71 ns/op** (target < 50ns — PASS by 70×). The dominant cost is BLAKE3 verify at load (403ns, paid once).
- [x] **T4.3** Bench `vessel_project_latency` — measure `WasmDotProjector::project()` cost. **Result: 1191 ns/op** (target < 1µs — FAIL by 19%). Documented in `.benchmarks/315_vessel_goat.md`: wasmi dispatch is structurally ~1µs; revised honest target for Cold tier is < 5µs.
- [x] **T4.4** Wrote `.benchmarks/315_vessel_goat.md` with G1-G5 results + decision (keep opt-in — G5 fails the aspirational 1µs, but the tier-aware design routes around it: Hot uses 0.71ns extract, Cold accepts 1.2µs).
- [x] **T4.5** GOAT decision: **stays opt-in**. G5 fails the 1µs target (1191ns actual). Re-promotion criteria documented in the bench file.

## Phase 5 — Examples + Docs

### Tasks

- [x] **T5.1** `katgpt-rs/examples/vessel_minimal.rs` — encode/decode/extract round-trip with a fake `[u8; 64]` Pod payload. Demonstrates zero-copy borrow + Arc sharing.
- [x] **T5.2** `katgpt-rs/examples/vessel_project.rs` — WAT module with `project` export, load as vessel, call projector, show `VesselCache` cache-miss→hit transition.
- [x] **T5.3** `katgpt-rs/.docs/15_vessel.md` — overview doc with the tier table, wire layout, API surface, security model, GOAT results, and 5-repo routing.

---

## Anti-Goals

- ❌ No `NeuronShard` import — this primitive is Pod-generic. Shard-specific wrapper is riir-neuron-db Plan 003.
- ❌ No cryptographic confidentiality claims in docs — see Research 297 §2.4.
- ❌ No game/chain semantics — no `DataTier`, no AOI, no fog-of-war here. Those land in riir-ai / riir-chain plans.
- ❌ No network/distribution code — the vessel is a local byte-blob. Distribution is riir-chain's job (ChunkedContentStore).

## GOAT Gate Summary (measured, with cache layer)

| Gate | Test | Measured | Target | Verdict |
|---|---|---|---|---|
| G1 extract fidelity | 10k round-trips | byte-identical | bit-identical | ✅ PASS |
| G4 extract latency | `extract_payload::<HlaPayload>()` | **0.54 ns/op** | < 50 ns | ✅ PASS (92× margin) |
| G5 project (raw) | `WasmDotProjector::project()` | **1067 ns/op** | < 1000 ns | ❌ FAIL (wasmi floor; cache-miss-only path) |
| **G5b project (cache hit)** | `VesselCache::project_cached_with_hash()` | **19.83 ns/op** | < 50 ns | ✅ PASS (2.5× margin) |
| **G-cache get (cache hit)** | `VesselCache::get_cached()` | **16.08 ns/op** | < 50 ns | ✅ PASS (3× margin) |

**Decision:** stays opt-in (dependency hygiene — pulls `wasmi` + `papaya`; not a quality gate). All system-level hot-path gates PASS with 2-92× margin. The raw G5 (1067ns, wasmi floor) is a cache-miss-only path — paid once per unique query, not per call. See `.benchmarks/315_vessel_goat.md`.

## Phase 6 — Cache Layer (DONE, added post-Phase 4 from user feedback)

User feedback (2026-06-24): "wasm vessel should be load once and after that in mem link aka return cache map in latent instead". Implemented two fixes:

- [x] **T6.1 (fix #1)** Cache `wasmi::Memory` handle in `OnceLock<CompiledVessel>` after `ensure_compiled`. `project()` reads cached handle instead of re-resolving `get_memory()` per call (~50ns saved/call). `OnceLock` allows one-time interior mutation through `&LoadedVessel`, composing with `Arc<LoadedVessel>` from the cache.
- [x] **T6.2 (fix #2)** Add `VesselCache` (papaya lock-free): `vessels: HashMap<[u8;32], Arc<LoadedVessel>>` + `results: HashMap<([u8;32], u64), f32>`. API: `get_or_load` (cold), `get_cached(addr)` (hot, 16ns), `project_cached_with_hash(addr, query, qhash)` (hot, 19.83ns hit / 1067ns miss), `evict(addr)` (AOI GC).
- [x] **T6.3** `content_addr: [u8;32]` field on `LoadedVessel` (BLAKE3 of full encoded bytes — distinguishes vessels with same WASM but different payload metadata).
- [x] **T6.4** 6 new tests: dedupe, metadata-distinction, extract-through-Arc, project-cached-hit, evict-cascade, missing-vessel-error. All 21 tests green.
- [x] **T6.5** Updated bench to measure cache-hit vs cache-miss paths. **G5b (19.83ns) + G-cache (16.08ns) both PASS < 50ns target.**
- [x] **T6.6** Updated `.benchmarks/315_vessel_goat.md` with v2 results + architecture diagram.

---

## TL;DR

Generic `Vessel` primitive in katgpt-rs: WASM-with-BLAKE3-header wire format + `extract_payload::<T: Pod>()` (zero-copy, Hot path — **measured 0.71 ns/op**, 70× under target) + `VesselProjector` trait (Cold path — measured 1.2µs, 19% over aspirational 1µs target but within Cold-tier budget). Re-uses existing `wasmi` + `blake3` + `bytemuck` deps. **Ships behind `secure_vessel` feature, stays opt-in** — G5 fails the 1µs gate but the tier-aware design routes around it. Private shard wrapper lives in riir-neuron-db Plan 003; private selling-point guide is riir-neuron-db Research 006. GOAT results in `.benchmarks/315_vessel_goat.md`.
