# Issue 006: T6 — Freeze/Thaw Merkle Integration into Go Self-Play Flow

> **Source:** Plan 253 (T6) — Merkle-Octree Curator Modelless Verification Layer
> **Status:** ✅ Done — implemented in `riir-ai/crates/riir-engine/src/kg.rs` behind `merkle_octree` feature. 7/7 tests pass.
> **Scope:** Cross-crate wiring (`katgpt-core` → `riir-ai`)
> **Priority:** Medium — blocks Plan 253 completion but not a blocker for other plans

---

## Problem

`MerkleFrozenStore` in `katgpt-core/src/curator.rs` provides freeze/thaw with Merkle root integrity verification, but it is not yet wired into the Go self-play pipeline in `riir-ai`.

## Required Changes

### 1. Feature Passthrough (`riir-ai/crates/riir-engine/Cargo.toml`)

```toml
# KG Latent Octree Merkle commitment (Plan 253).
merkle_octree = ["katgpt-core/merkle_octree"]
```

### 2. KG Module Integration (`riir-ai/crates/riir-engine/src/kg.rs`)

Wire `MerkleFrozenStore::freeze_with_root()` / `thaw_and_verify()` into the existing self-play → KG extraction pipeline:

```text
GoSelfPlayResult[] → extract_triples() → consolidate() → build_with_merkle() → MerkleFrozenStore::freeze_with_root(merkle_root)
```

Thaw path:
```text
MerkleFrozenStore::thaw_and_verify(key) → verify Merkle root matches → extract KG triples
```

### 3. Entry Points

| Location | Change |
|----------|--------|
| `riir-engine/src/kg.rs` | Import `MerkleFrozenStore` behind `#[cfg(feature = "merkle_octree")]` |
| `riir-examples/examples/` | Add `g_zero_*` example wiring freeze/thaw into arena loop |
| `riir-engine/Cargo.toml` | Add `merkle_octree` feature passthrough |

### 4. Existing Infrastructure to Reuse

- `riir-engine/src/kg.rs`: `StateTransition` → `extract_triples()` already extracts KG triples from self-play
- `riir-chain/src/neuron_db/shard.rs`: `NeuronShard::set_merkle_root()` / `compute_merkle_root()` — binary Merkle tree over zone commitments
- `riir-engine` default features already include `sense_training` — Merkle sits on top of this
- `WalletWeight::freeze()/thaw()` pattern in `riir-chain` — follow same repr(C) + BLAKE3 checksum convention

### 5. GOAT Target

- Full pipeline overhead < 3% of self-play loop time
- Merkle build adds ~3.5µs per 64 KG triples (benchmarked)
- Freeze/thaw adds ~1µs (MerkleFrozenStore benchmarked at <1µs)

## Blockers

None. `katgpt-core` with `merkle_octree` feature is ready. `riir-ai` already depends on `katgpt-core`.

## Estimated Scope

~200-300 lines across 3-4 files in `riir-ai`. Follows existing patterns (feature passthrough, `#[cfg]` guards, repr(C) freeze/thaw).

---

## TL;DR

Wire `MerkleFrozenStore` from `katgpt-core` into `riir-ai`'s Go self-play KG extraction pipeline. Needs feature passthrough in `riir-engine/Cargo.toml` + import/integration in `kg.rs`. ~300 lines, no blockers.
