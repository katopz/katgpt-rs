# Doc 15 — Vessel: Extract-Once Secure Wire Format

**Plan:** [315](../.plans/315_vessel_extract_once_primitive.md)
**Research:** [297](../.research/297_vessel_extract_once_secure_wire_format.md)
**Feature flag:** `secure_vessel` (opt-in)
**Status:** Phase 1-4 + 6 complete, GOAT-passing (hot paths), stays opt-in (dependency hygiene)

---

## TL;DR

A `Vessel` is a wire format for shipping a `#[repr(C)]` Pod payload inside a WASM module, with a BLAKE3-integrity header. Load it once, then either:

- **Extract** the Pod bytes to host memory for zero-copy SIMD access (~0.5 ns/op), or
- **Project** a query through the WASM module and get only the scalar result back (~1.1 µs raw, ~20 ns cached).

The WASM layer provides API encapsulation + capability security + chain-committed integrity — **not** cryptographic confidentiality (a debugger can dump linear memory). The honest value is access control + verifiable computation.

## Wire layout

```text
+---------------------+-----------------------+
| VesselHeader (52 B) | WASM module bytes ... |
+---------------------+-----------------------+
```

`VesselHeader` (`#[repr(C)]`, `Pod`-compatible):

| Field | Type | Purpose |
|---|---|---|
| `magic` | `[u8; 4]` | `b"VSL1"` — format identifier |
| `version` | `u32` | Wire-format version (currently 1) |
| `blake3` | `[u8; 32]` | BLAKE3 hash over the WASM bytes |
| `payload_kind` | `u32` | Caller-defined discriminator (e.g. shard type) |
| `payload_offset` | `u32` | Byte offset of the Pod payload inside the WASM bytes |
| `payload_len` | `u32` | Payload length in bytes (`== size_of::<T>()` for extract) |

Content address = BLAKE3 of the 52-byte header (~50 ns). Two vessels with the same header are byte-identical modulo BLAKE3 collisions (negligible; also caught by `verify_blake3`).

## Tier routing

The vessel is tier-aware: the runtime chooses the path based on `DataTier`.

| Tier | Path | Latency | When to use |
|---|---|---|---|
| **Plasma** | `extract_payload::<T: Pod>()` | ~0.5 ns | Hot inference loop — weights in host SIMD registers |
| **Hot** | `extract_payload::<T: Pod>()` | ~0.5 ns | Same — extract is the default for any latency-sensitive path |
| **Warm** | `extract` (with AOI GC) | ~0.5 ns + eviction | Reloaded on cache miss via `VesselCache::get_or_load` |
| **Cold** | `VesselCache::project_cached_with_hash()` | ~20 ns (hit) / ~1.1 µs (miss) | Capability-restricted: host never sees weights, only scalars |
| **Freeze** | `project_cached` | ~20 ns (hit) | Chain verification — prove "this projection came from THIS bytecode" |

### Why extract beats project

`extract_payload` is a branchless pointer cast — `size_of` check + `slice::get` + `bytemuck::from_bytes`. The BLAKE3 verify (~400 ns) is paid **once** at `load_vessel` time and amortized over all subsequent extracts. The 0.5 ns/op is the L1-hit-load floor.

`project()` must dispatch through wasmi (an interpreter, not a JIT), copying the query into WASM linear memory and interpreting the projection function under a fuel budget. ~1.1 µs is wasmi's structural floor for a 64-element f32 sum. The result cache makes this a cache-miss-only cost — repeated queries against the same vessel+query hit the cache at ~20 ns.

## Architecture: load-once, ref-many

```text
   load_vessel(bytes)  ←── paid ONCE (~450 ns: header hash + BLAKE3 verify)
        │
        ▼
   VesselCache.vessels  ──get_cached(addr)──► Arc<LoadedVessel>  [~16 ns hit]
                                      │
              ┌───────────────────────┼──────────────────────┐
              ▼                       ▼                      ▼
   extract_payload::<T>()     WASM linear memory       project_cached_with_hash()
   (host &T borrow)           (Cold path, if needed)   (result cache → ~20 ns hit)
   ~0.5 ns/op                  refs same Arc bytes     cache miss = ~1.1 µs (paid once
   pure pointer math                                   per unique query)
```

The `Arc<LoadedVessel>.wasm_bytes` is the single shared latent buffer — both extract (host `&T` borrow) and project (WASM linear memory reads) reference the same allocation. No per-access copy. The `Arc` clone on the cache-hit path is one atomic refcount bump.

## API surface

```rust
// Encode (cold — producer side)
let encoded = encode_vessel(&wasm_bytes, payload_kind, payload_offset, payload_len);

// Load once (cold — consumer side)
let vessel = load_vessel(&encoded)?;                    // ~450 ns, paid once
let cached_vessel = cache.get_or_load(&encoded)?;       // same, + cache insert

// Hot path — extract (Plasma/Hot/Warm)
let weights: &MyPod = extract_payload(&vessel)?;        // ~0.5 ns, zero-copy

// Hot path — cached projection (Cold/Freeze)
let qhash = query_hash(&query);                         // pre-hash once
let result = cache.project_cached_with_hash(            // ~20 ns on cache hit
    addr, &query, qhash, &projector, &mut store, &engine,
)?;

// Fastest path — pure handle lookup
let handle = cache.get_cached(&addr)?;                  // ~16 ns

// AOI GC
cache.evict(&addr);                                     // evicts vessel + all its results
```

## Security model (honest)

WASM is **not** cryptographic confidentiality. What it provides:

| Property | Strength | Source |
|---|---|---|
| Integrity (tamper detection) | **Cryptographic** | BLAKE3 in header + content address |
| API encapsulation | **Strong** | WASM only exports what it declares |
| Capability security | **Strong** | Host grants/revokes imports; fuel budget bounds runtime |
| Soft obfuscation | **Weak** | Hexdump doesn't reveal floats; debugger defeats it |

True confidentiality would require FHE dot-product or TEE (SGX/SEV) — out of scope for this primitive.

The cryptographic angle that **does** hold: if the WASM bytecode is BLAKE3-committed and the projection result is LatCal-committed (P3 follow-up), a verifier node can prove "this projection was computed by THIS bytecode" without seeing the weights. **Integrity without confidentiality** — and integrity is what the chain needs for consensus.

## GOAT gate results (measured, release build, Apple Silicon)

| Gate | Test | ns/op | Target | Verdict |
|---|---|---|---|---|
| G1 | extract fidelity (10k round-trips) | byte-identical | bit-identical | ✅ PASS |
| G4 | `extract_payload::<[f32; 64]>()` | 0.54 | < 50 | ✅ PASS (92× margin) |
| G5 | `WasmDotProjector::project()` (raw) | 1067 | < 1000 | ❌ FAIL (wasmi floor, cache-miss-only) |
| **G5b** | `project_cached_with_hash()` (hit) | 19.83 | < 50 | ✅ PASS (2.5× margin) |
| **G-cache** | `get_cached()` (hit) | 16.08 | < 50 | ✅ PASS (3× margin) |

Full results + analysis: [`.benchmarks/315_vessel_goat.md`](../.benchmarks/315_vessel_goat.md)

## Where this lives in the 5-repo strategy

- **`katgpt-rs`** (this crate) — the generic open primitive. Pod-generic, no shard/game/chain semantics.
- **`riir-neuron-db`** — private Super-GOAT wrapper: `NeuronVesselSidecar` wraps `NeuronShard` in a vessel, adds `verify_owner` NFT gating + tier routing. See Research 006 / Plan 003.
- **`riir-chain`** — extends `AssetDeliveryKind::WasmVessel` to accept neuron payloads via `VesselHeader.payload_kind`. See Plan 006 (P2 follow-up).
- **`riir-ai`** — runtime wiring: `SenseModule` / HLA projection path selects raw Pod (Hot) vs vessel-project (Cold) based on `DataTier`. See Plan 333 (P2 follow-up).

## Reproduction

```bash
# Build + test
cargo test --features secure_vessel --lib vessel::

# Run the GOAT bench
cargo bench --bench vessel_extract_bench --features secure_vessel

# Run the examples
cargo run --example vessel_minimal --features secure_vessel
cargo run --example vessel_project --features secure_vessel
```
