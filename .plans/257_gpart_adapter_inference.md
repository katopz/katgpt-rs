# Plan 257: GPart Isometric Adapter Inference

Feature gate: `gpart_adapter` (default-OFF)

## Context

Reference: [Research 227](../.research/227_GPart_Isometric_Partition_Inference.md)

GPart replaces LoRA's bilinear BA factorization with a single isometric partition matrix P (seed-generated, deterministic): `W = W₀ + Pθ_d` where `P^T P = I_d`. Storage: d+1 values (θ_d + seed) vs LoRA's r(m+n) — 2–4× compression for micro-transformer, 10–100× for larger models. Same mathematical family as existing `JlProjectionMatrix` (Gram-Schmidt orthogonal projection with BLAKE3 commitment). Modelless, inference-time only — θ_d trained in riir-ai, loaded and applied in katgpt-rs.

**Key invariant:** `fastrand::Rng` is deterministic and cross-platform — same seed + θ_d produces bit-identical weight deltas on x86, ARM, and WASM.

## Binary Format

```text
[GPART(5) | version(4) | d(4) | seed(8) | blake3(32) | theta(d×4)]
```

- Magic: `b"GPART"` (5 bytes)
- Version: `u32` LE (4 bytes) — initial version = 1
- `d`: `u32` LE partition dimension (4 bytes)
- `seed`: `u64` LE for `fastrand::Rng` (8 bytes)
- `blake3`: 32-byte commitment over `seed.to_le_bytes() || theta.as_bytes()`
- `theta`: `d × f32` LE (d×4 bytes)

**NeuronShard Pod compatibility:** seed(8) + θ_d(d≤90 × 4 = 360) = 368 bytes max. Fits in NeuronShard's fixed-size Pod alongside BLAKE3 zone_hash.

## Tasks

- [x] Add `gpart_adapter = []` feature gate to `crates/katgpt-core/Cargo.toml` `[features]` section, not in default set
- [x] Define `GpartAdapter` struct in `crates/katgpt-core/src/types.rs` (fields: `seed: u64`, `theta: Vec<f32>`, `d: usize`) behind `#[cfg(feature = "gpart_adapter")]`
- [x] Implement `GpartAdapter::generate_partition()` — seed-based pseudorandom group assignment using `fastrand::Rng`, returning assignments and group counts; single-pass O(N) counting
- [x] Implement `GpartAdapter::apply()` — single-pass O(N) broadcast: regenerate partition from seed, compute `1/√n_g` per group, apply `base_weights[i] += scale * theta[group]`; accept pre-allocated scratch `&mut [usize]` for assignments to avoid hot-loop allocation
- [x] Implement SIMD-accelerated `apply_simd()` using `simd_add_scalar_inplace` / chunked broadcast — group parameters by contiguous assignment, apply `theta[g] * inv_sqrt_ng` in SIMD-width chunks; fall back to scalar for tail
- [x] Implement `GpartAdapter::commitment()` → `[u8; 32]` — `BLAKE3(seed.to_le_bytes() || theta.as_bytes())`; and `verify(expected: &[u8; 32])` → `bool`
- [x] Implement binary I/O: `GpartAdapter::save()` and `GpartAdapter::load()` following the `[GPART | version | d | seed | blake3 | theta]` format; validate magic, version, and BLAKE3 checksum on load
- [x] Define `GpartPair` struct mirroring `LoraPair` pattern — `(GpartAdapter, GpartAdapter)` for prefill/decode adapter split; implement `apply_prefill()` and `apply_decode()` delegating to respective adapter
- [x] Implement `TryFrom<&LoraAdapter>` conversion trait: `LoraAdapter → GpartAdapter` (lossy, requires pre-computed θ_d = P⁺ΔW from training side; stub with `unimplemented!()` + doc note that riir-ai training pipeline must provide θ_d)
- [x] Create GOAT benchmark at `tests/bench_257_gpart_adapter_goat.rs` — G1: storage < 50% of LoRA; G2: apply speed ≤ 110% of LoRA; G3: quality ≥ 95% (requires trained θ_d, mark `#[ignore]`); G4: cross-platform determinism; G5: BLAKE3 commitment integrity; mirror structure of `bench_230_shard_embedding_goat.rs`
- [x] Create example at `examples/gpart_adapter_demo.rs` — demonstrate `GpartAdapter` construction from seed+θ, `apply()` on sample weights, `commitment()`/`verify()`, binary save/load roundtrip; gated behind `#[cfg(feature = "gpart_adapter")]`
- [x] Add unit tests in `types.rs` `#[cfg(test)] mod tests`: isometry check (`P^T P ≈ I_d`), apply correctness (known seed + θ → expected output), commitment roundtrip, tamper detection, load/save roundtrip, determinism across repeated seeds
- [x] Verify NeuronShard 368-byte Pod compatibility: `seed(8) + d(max=90)×4(360) = 368` — add compile-time assertion `const _: () = assert!(8 + 90 * 4 <= 368);` behind feature gate
- [x] Re-export `GpartAdapter` and `GpartPair` from `crates/katgpt-core/src/lib.rs` behind `#[cfg(feature = "gpart_adapter")]`

## GOAT Gates

| Gate | Metric | Threshold | Pass Criteria |
|------|--------|-----------|---------------|
| G1: Storage | Adapter bytes | < 50% of LoRA | `size_of_val(gpart) / size_of_val(lora) < 0.5` |
| G2: Apply speed | Time to apply | ≤ 110% of LoRA | `gpart_apply_time / lora_apply_time ≤ 1.1` |
| G3: Quality | Output quality | ≥ 95% of LoRA | Requires trained θ_d → `#[ignore]` until riir-ai provides |
| G4: Determinism | Cross-platform bit-identical | 100% | Same seed+θ → identical `base_weights` on x86+ARM+WASM |
| G5: Commitment | BLAKE3 verification | 100% pass | Tamper on any byte → `verify()` returns false |

## Promotion/Demotion

- **Promote** to default feature if G1–G5 all pass
- **Keep gated** if G3 fails (quality regression — investigate θ_d training)
- **Demote** if G2 fails (>10% slower than LoRA)
- `LoraAdapter` is **never removed** — GPart is an alternative loading path, not a replacement

## Not Implementing (Deferred)

- Idea 2 (Partition Pruning / BanditPruner integration) → create issue at `.issues/`
- Idea 3 (MUX-Latent isometric weighting) → blocked on Plan 238
- Idea 4 (Seed-Route Consensus) → blocked on multi-node consensus layer

## Related

| Item | Connection |
|------|------------|
| `JlProjectionMatrix` (`shard_embedding.rs`) | Same math family — Gram-Schmidt orthogonal projection |
| `LoraAdapter` (`types.rs`) | Existing adapter GPart augments |
| `simd_add_scalar_inplace` (`simd.rs`) | SIMD broadcast for chunked apply |
| `bench_230_shard_embedding_goat.rs` | GOAT benchmark template |
| `NeuronShard` (riir-ai) | 368-byte Pod — seed + θ_d must fit |

TL;DR: **Implement `GpartAdapter` behind feature gate. Single-pass O(N) broadcast from seed+θ. BLAKE3 commitment. SIMD-accelerated apply. GOAT-prove against LoRA before promoting.**
