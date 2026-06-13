# Issue 011: Purge JSON — Zero-Copy Binary Serialization Everywhere

## Status: IN PROGRESS — katgpt-rs core done, riir-ai training pipeline deferred

- [x] `proof_cert/certificate.rs` — `ProofEvidence::Custom` now `Vec<u8>` binary blob
- [x] `proof_cert/macros.rs` — replaced `serde_json::json!({})` with `Vec::new()`
- [x] `proof_cert/wasm_certificates.rs` — replaced `serde_json::json!` with `postcard`
- [x] `proof_cert/wasm_proof_witness.rs` — replaced `serde_json::json!` with binary encoding
- [x] `proof_cert/serde_impls.rs` — binary persistence with magic+version header (postcard)
- [x] `pruners/bomber/replay.rs` — `to_bytes`/`from_bytes` + length-prefixed binary writer
- [x] `pruners/trial_log.rs` — binary append log (length-prefixed postcard)
- [x] `pruners/editable_constraint.rs` — binary rules parsing (postcard)
- [x] `pruners/substrate_loader.rs` — binary mask load/save (postcard)
- [x] `pruners/concept_grounding.rs` — removed hand-rolled JSON serializer, replaced with postcard
- [x] `pruners/sdpg/mod.rs` — updated `load_teacher_q_from_replay` for binary format
- [x] `rt_turbo/calibration.rs` — `to_bytes`/`from_bytes` (postcard)
- [x] `rt_turbo/projection.rs` — `to_bytes`/`from_bytes` (postcard)
- [x] `rt_turbo/tests.rs` — all tests updated to binary roundtrip
- [x] `feedback.rs` — binary feedback (postcard)
- [x] `skill_opt/buffer.rs` — `to_bytes`/`from_bytes` (length-prefixed postcard)
- [x] `examples/rt_turbo_01_calibration.rs` — updated for binary serialization
- [x] `Cargo.toml` — added `postcard` dep to both katgpt-rs and katgpt-core
- [ ] `pruners/go/replay.rs` — Go replay still uses JSON (deferred — training data format)
- [ ] `pruners/go/replay_writer.rs` — Go JSONL writer (deferred — training data format)
- [ ] riir-ai training pipeline — JSONL is the training data format, deferred
- [ ] Remove `serde_json` from `Cargo.toml` when all callers migrated

## Problem
Runtime JSON serialization (`serde_json::to_string`, `serde_json::json!`, `from_str`) has crept into hot paths and latent structures. JSON is:
- **Slow**: string parsing/formatting overhead vs raw byte copy
- **Fat**: JSON key names duplicate field info that's already known from struct layout
- **Lossy**: float precision issues, no zero-copy possible
- **Anti-pattern**: Latent-to-latent should be `bytemuck` / `postcard` / raw byte slice, no text codec

Per the architecture rules: latent structures (`NeuronShard`, `HlaCacheProxy` scalar outputs, embeddings) MUST use zero-copy binary. Physical domain (`MapPos`, `ForceVector`) MUST be raw deterministic bytes. No JSON in either.

## Scope

### 🔴 Critical (Hot Path / Latent Structures)
| File | Offense | Fix |
|------|---------|-----|
| `src/proof_cert/certificate.rs` | `ProofEvidence::Custom { data: serde_json::Value }` | Replace with `Vec<u8>` or `[u8; N]` — evidence is binary blob |
| `src/proof_cert/macros.rs` | `serde_json::json!({})` for evidence | Write binary evidence directly |
| `src/proof_cert/wasm_certificates.rs` | `serde_json::json!({ "derived_from": ... })` | Binary key-value pairs |
| `src/proof_cert/wasm_proof_witness.rs` | `serde_json::json!({ witness_hash, ... })` | All fields are already binary-friendly (hashes, strings, bools) — just use struct |
| `src/proof_cert/serde_impls.rs` | `serde_json::to_vec_pretty` / `from_str` for cert persistence | Use `postcard` (already a dep) for binary wire format |
| `src/pruners/bomber/replay.rs` | `to_json()` / `from_json()` on `ReplaySample` | Replace with `postcard` or raw binary layout |
| `src/pruners/go/replay.rs` | `to_json()` / `from_json()` / `to_json_pretty()` | Same — binary format |
| `src/pruners/go/replay_writer.rs` | JSONL via `serde_json::to_string` | Binary record format with length prefix |
| `src/pruners/trial_log.rs` | `serde_json::json!()` + `to_string` per append | Binary log format (length-prefixed `postcard`) |
| `src/pruners/editable_constraint.rs` | `serde_json::from_str(json)` for rules | Binary rules format |
| `src/pruners/substrate_loader.rs` | JSON load/save for substrate masks | Binary format |
| `src/rt_turbo/calibration.rs` | `to_json()` on `HeadCalibration` | Binary serialize |

### 🟡 Config / Cold Path (Lower Priority but Still Wrong)
| File | Offense | Fix |
|------|---------|-----|
| `src/feedback.rs` | `serde_json::to_string` for `InferenceResult` | `postcard` binary body |
| `src/pruners/delta_mem/state.rs` | JSON roundtrip in test | Switch test to binary |
| `src/pruners/proof/sketch_types.rs` | JSON roundtrip in tests | Switch to `postcard` |
| `src/pruners/ropd_rubric/` | JSON roundtrip in tests | Switch to `postcard` |

### 🟢 External API Boundary (Acceptable JSON — Leave Alone)
| File | Reason |
|------|--------|
| `riir-chaind/src/bridge/` | Solana JSON-RPC — external API, we don't control the protocol |
| `riir-rest/` | REST API — JSON is the wire format by contract |
| `riir-staker-wasm/` | WASM bridge — `serde-wasm-bindgen` is the JS boundary |

### ✅ Already Correct (Reference Implementations)
| File | Approach |
|------|----------|
| `src/pruners/bomber/wasm_state.rs` | `ZeroCopyStateBuffer` with `bytemuck` |
| `src/pruners/freeze.rs` | Magic bytes + raw binary |
| `src/pruners/dreamer/frozen.rs` | No serde/bincode needed |
| `src/speculative/selectivity_router.rs` | `bytemuck::cast_slice` |
| `src/sense/gm.rs` | Binary `serialize_snapshot` |
| `riir-chain/neuron_db/shard.rs` | `ShardSerializeBuf` — zero-alloc binary |

## Strategy

1. **Replace `ProofEvidence::Custom { data: serde_json::Value }` with `Custom { data: Vec<u8> }`**
   - Evidence is arbitrary bytes — hashes, witness data, etc. JSON adds nothing.

2. **Replace cert persistence with `postcard`** 
   - `save_certificates` → `postcard::to_allocvec` + blake3 commit
   - `load_certificates` → `postcard::from_bytes`
   - Add magic bytes + version header (same pattern as `freeze.rs`)

3. **Replace replay/sample JSON with binary record format**
   - Length-prefixed `postcard` records for JSONL replacement
   - Keeps streaming/append-friendly, but binary

4. **Replace `trial_log` with binary append log**
   - Length-prefixed `postcard` per record
   - No string parsing on load

5. **Replace `HeadCalibration::to_json()` with `to_bytes()` using `postcard`**

6. **Remove `serde_json` from `Cargo.toml` dependencies** (or make it optional for test-only)

7. **Switch all `#[derive(Serialize, Deserialize)]` on hot structs to use `postcard`**
   - Keep `serde` derive but route through `postcard` instead of `serde_json`
   - OR: remove serde entirely from hot structs and use `bytemuck::Pod` where layout allows

## Gates
- All existing tests must pass with binary format
- Bench: binary serialize/deserialize must be ≥5x faster than JSON equivalent
- Zero `serde_json::to_string` / `serde_json::from_str` calls remain in non-test, non-external-API code
