# Issue 063: Separate SpectralQuant from TurboQuant

> **Status:** ✅ Done — all 13 tasks complete, SpectralQuant promoted to default feature

## Problem

Plan 077 placed all SpectralQuant code inside `src/turboquant/` — spectral files, types, and module registrations are all mixed into the TurboQuant module. These are fundamentally different quantization strategies:

- **TurboQuant**: rotation + codebook-based uniform quantization
- **SpectralQuant**: eigenbasis calibration + water-fill + non-uniform (Lloyd-Max) quantization

SpectralQuant is planned to **replace** TurboQuant as the default. Having it nested inside TurboQuant creates confusing coupling and makes it harder to independently feature-gate, benchmark, and eventually swap defaults.

## Current State (All Committed)

```
src/turboquant/
├── mod.rs                    ← spectral module registrations + re-exports
├── types.rs                  ← 115 lines of spectral types appended (L49-L163)
├── forward.rs                ← 3 spectral functions + 1 spectral test (L140-L370)
├── nonuniform_quant.rs       ← spectral-specific
├── spectral.rs               ← spectral-specific (979 lines)
├── spectral_kv_cache.rs      ← spectral-specific (953 lines)
└── spectral_rotation.rs      ← spectral-specific (228 lines)
```

## Key Insight: `forward_turboquant` is Already Generic

`forward_turboquant` in `transformer.rs` only calls this cache interface:
- `store_key(layer, pos, key)`
- `store_value(layer, pos, value)`
- `dequantize_key_into(layer, pos, out)`
- `dequantize_value_into(layer, pos, out)`
- `reset()`, `pos()`, `set_pos()`

Both `TurboQuantKVCache` and `SpectralQuantKVCache` expose the **identical interface**.
`forward_turboquant` is really `forward_quantized` — the body is 100% cache-interface agnostic.

**Conclusion**: Extract a `QuantizedKVCache` trait, make `forward_quantized` generic over it.
When TurboQuant is eventually removed, `transformer.rs` needs **zero changes**.

## Tasks

- [x] **T1: Create `src/spectralquant/` module** — new top-level module
- [x] **T2: Move spectral files** — move 5 files to `src/spectralquant/`:
  - `spectral.rs`, `spectral_kv_cache.rs`, `spectral_rotation.rs`, `nonuniform_quant.rs`
  - Extract 3 spectral functions + 1 test from `turboquant/forward.rs` → new `spectralquant/forward.rs`
- [x] **T3: Move spectral types** — extract `#[cfg(feature = "spectral_quant")]` types from `turboquant/types.rs` → `spectralquant/types.rs`
- [x] **T4: Clean `src/turboquant/`** — remove all spectral code from `mod.rs`, `types.rs`, `forward.rs`
- [x] **T5: Extract `QuantizedKVCache` trait** — shared interface for both caches
- [x] **T6: Implement trait on both caches** — `impl QuantizedKVCache for TurboQuantKVCache` + `SpectralQuantKVCache`
- [x] **T7: Rename `forward_turboquant` → `forward_quantized`** — generic over `impl QuantizedKVCache`
- [x] **T8: Rename `tq_dequant_pos` → `dequant_pos`** — in `ForwardContext` + `reset_tq_dequant` → `reset_dequant`
- [x] **T9: Feature-gate TurboQuant** — add `turboquant` feature, off by default (baseline/bench only)
- [x] **T10: Promote SpectralQuant to default** — `spectral_quant` in default features
- [x] **T11: Register `spectralquant` module in `src/lib.rs`** with `#[cfg(feature = "spectral_quant")]`
- [x] **T12: Fix all imports** — update `use` paths in moved files + test files + benchmark.rs
- [x] **T13: Compile check** — `cargo check --features spectral_quant` + `cargo check --features turboquant`

## Target Structure

```
src/
├── types.rs                         ← QuantizedKVCache trait here (shared)
├── turboquant/                      ← baseline, feature-gated, educate/bench only
│   ├── mod.rs
│   ├── codebook.rs
│   ├── forward.rs                   ← TurboQuant-specific forward helpers only
│   ├── kv_cache.rs
│   ├── rotation.rs
│   └── types.rs                     ← TurboQuant types only
├── spectralquant/                   ← default KV compression
│   ├── mod.rs
│   ├── types.rs                     ← spectral types
│   ├── spectral.rs
│   ├── nonuniform_quant.rs
│   ├── spectral_kv_cache.rs
│   ├── spectral_rotation.rs
│   └── forward.rs                   ← spectral forward helpers (extracted)
tests/
├── bench_turboquant.rs              ← baseline bench (feature-gated)
└── bench_spectralquant.rs           ← primary bench
```

## Feature Flags in Cargo.toml

```toml
[features]
default = ["sparse_mlp", "domain_latent", "ppot", "bandit", "bt_rank", "spectral_quant"]
spectral_quant = []                  # default KV compression
turboquant = []                      # legacy baseline for bench/educate only
```

## `QuantizedKVCache` Trait

```rust
/// Shared interface for quantized KV caches.
/// Enables `forward_quantized` to work with any compression backend.
pub trait QuantizedKVCache {
    fn store_key(&mut self, layer: usize, pos: usize, key: &[f32]);
    fn store_value(&mut self, layer: usize, pos: usize, value: &[f32]);
    fn dequantize_key_into(&mut self, layer: usize, pos: usize, out: &mut [f32]);
    fn dequantize_value_into(&mut self, layer: usize, pos: usize, out: &mut [f32]);
    fn reset(&mut self);
    fn pos(&self) -> usize;
    fn set_pos(&mut self, pos: usize);
}
```

## Acceptance Criteria

1. `turboquant/` contains zero spectral code — clean separation
2. `spectralquant/` is self-contained with its own types + forward helpers
3. Both features compile independently: `cargo check --features turboquant` and `cargo check --features spectral_quant`
4. `forward_quantized` is generic over `QuantizedKVCache` — no duplicated forward path
5. `transformer.rs` has no direct dependency on either specific cache type
6. No regression in existing tests
7. `spectral_quant` is in default features — TurboQuant opt-in only