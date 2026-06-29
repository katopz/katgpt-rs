# Issue 015 — Extract KV + spectral quantization code into standalone crates

**Date:** 2026-06-29
**Status:** Complete (committed 8cb4d058 on develop)
**Severity:** DRY / decoupling / discoverability (user rule: "DRY, Modular, Generic, Decouple")
**Scope:** `katgpt-rs` only (no `riir-*` changes; consolidation deferred per user).

## Problem

`src/` currently scatters **12,585 lines of KV-cache code** across 7 module
groups and one **5,059-line spectral quantization substrate** that has
non-KV consumers. Symptoms:

- KV modules live in the root crate by name only — there is no actual KV
  namespace; `kv_share.rs`, `osc_kv.rs`, `cs_kv_probe/`, `shard_kv/`,
  `sp_kv/`, `still_kv/`, `kvarn/` are flat siblings of unrelated code.
- `spectralquant/` (5k LoC, 8 files) is consumed by both KV code
  (`shard_kv`, `kvarn`) and non-KV code (`funcattn_compose`,
  `chiaroscuro`, `benchmark/infrastructure`). It is foundational math but
  pinned inside the root crate, blocking KV extraction.
- `targeted_precision.rs` (148 LoC) is consumed **only** by `kvarn` — a KV
  leaf that has been stranded inside `src/` for no structural reason.
- `QuantizedKVCache` trait (the natural extension point for every backend)
  lives in `src/types.rs`, gated to `crate::still_kv::CompactionStrategy`
  via a `#[cfg(feature = "still_kv")]` default method — feature-coupled but
  not crate-decoupled.
- Sibling repos (`riir-ai/crates/riir-engine/src/{kvarn_quality,kvarn_tier}`,
  `riir-ai/crates/riir-gpu/src/kvarn`) duplicate KV concepts because they
  cannot link `src/kvarn/` directly (transitively pulls
  `crate::targeted_precision`).

## Verdict

**Extract two crates in one issue.** `katgpt-spectral` is the prerequisite
for `katgpt-kv` because `shard_kv` cannot move without it, and `kvarn`
cannot move without `targeted_precision` which is KV-only.

Gates:
- G1 correctness: pure `git mv` + re-export shims → bit-identical API
  surface. No semantic changes.
- G2 perf: parallel workspace compile; cleaner `cargo doc` slice.
- G3 no-regression: all feature flags preserved with identical default sets.
- G4 alloc-free: N/A (structural).
- G5 modelless: yes — KV cache and quantization are modelless inference
  primitives.

## Dependency Reality (the reason this is one issue, not two)

```
katgpt-core ──┬── katgpt-types  (shared types + QuantizedKVCache trait)
              │
              ├── katgpt-spectral  (spectralquant module; non-KV consumers too)
              │        ▲
              │        │
              └── katgpt-kv  ─── depends on ── katgpt-spectral
                       │
                       ├── kv_share      (clean)
                       ├── cs_kv_probe   (clean)
                       ├── sp_kv         (clean)
                       ├── osc_kv        (clean after trait move)
                       ├── still_kv      (inbound deps are #[cfg(feature="still_kv")]-gated)
                       ├── shard_kv      (uses katgpt-spectral)
                       ├── kvarn         (uses katgpt-spectral via targeted_precision)
                       └── targeted_precision  (KV-only, moves here)

katgpt-rs (root) ── re-exports both via feature-flagged `pub use`
```

### Why two crates, not one

`spectralquant` has 3 non-KV consumers (`funcattn_compose`,
`chiaroscuro`, `benchmark/infrastructure`). Folding it into `katgpt-kv`
would force non-KV modules to depend on a crate named `-kv`. Wrong
direction. It is a separate foundational crate.

### Why `targeted_precision` moves INTO `katgpt-kv`

Single consumer (`kvarn/kv_cache.rs`). 148 LoC. KV-specific by usage. Not
worth its own crate.

## Plan

### Phase 1 — Move `QuantizedKVCache` trait to `katgpt-types`

The trait is the extension point every backend implements. Lives in
`src/types.rs:11-55` today. Move to
`crates/katgpt-types/src/kv_cache.rs`.

The `compact_into` default method references `crate::still_kv::*` types.
Move the **type definitions** (`CompactionStrategy` enum, `CompactKVCache`
struct, `KVChunk` struct) to `katgpt-types` too (they are plain structs/enums;
impls stay in `still_kv`). The default method body becomes
`katgpt_types::CompactKVCache` etc.

### Phase 2 — Scaffold `katgpt-spectral` crate

```
crates/katgpt-spectral/
├── Cargo.toml
└── src/
    ├── lib.rs         (re-exports)
    ├── spectral.rs
    ├── spectral_rotation.rs
    ├── spectral_kv_cache.rs
    ├── forward.rs
    ├── nonuniform_quant.rs
    ├── outlier_guard.rs
    └── types.rs
```

- Leaf crate, depends on `katgpt-core` + `katgpt-types` + `half` +
  `bytemuck` (existing deps of spectralquant).
- Feature `spectral_quant = []` (empty — current default-on flag is just a
  re-export gate at root).

### Phase 3 — Scaffold `katgpt-kv` crate

```
crates/katgpt-kv/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── kv_share.rs
    ├── osc_kv.rs
    ├── targeted_precision.rs
    ├── cs_kv_probe/{mod,budget,gate,lasso,probe,types}.rs
    ├── shard_kv/{mod,kv_cache,rope,types}.rs
    ├── sp_kv/{mod,forward,types,utility_predictor}.rs
    ├── still_kv/{mod,beta_bias,compact_cache,iterative,perceiver,position_free,query_bank}.rs
    └── kvarn/{mod,eval,hadamard,kv_cache,var_norm}.rs
```

- Depends on `katgpt-core` + `katgpt-types` + `katgpt-spectral` + `half`
  + `bytemuck`.
- Feature flags forwarded one-to-one:
  ### Phase 2 amendments (during execution)

  Three cross-crate decoupling fixes were needed:

  1. **`OutlierAction` + `OutlierGuardConfig`** moved from `src/types.rs` into
     `crates/katgpt-spectral/src/outlier_guard.rs` (they were consumed only by
     the outlier guard; root re-exports them via
     `pub use katgpt_spectral::outlier_guard::*` for back-compat).
  2. **`generate_rotation_matrix`** vendored into
     `crates/katgpt-spectral/src/spectral_rotation.rs` as a private helper
     (50 LoC, depends only on `katgpt_core::types::Rng` + `simd_sum_sq`).
     Eliminates the `crate::turboquant` cross-module dep. Original still lives
     in `src/turboquant/rotation.rs`; future turboquant extraction should
     consolidate.
  3. **`CalibrationResult::stiff_soft_decomposition`** deleted (zero callers,
     cross-crate feature coupling to root's `stiff_anomaly` module). The
     accompanying test `test_g5_stiff_soft_from_calibration` was removed too.
     Re-implementation path documented as extension trait
     `CalibrationResultStiffExt` in root's `stiff_anomaly` module if needed.

  ### Phase 3 — Scaffold `katgpt-kv` crate (amended during execution)

  **Deviation from plan:** `sp_kv/forward.rs` could not move into katgpt-kv
  because `forward_sp_kv` / `forward_sp_kv_quant` take
  `crate::transformer::ForwardContext` (a root-crate type with ~15 directly-
  accessed fields). Making them generic would require an unergonomic trait.
  Instead:

  - katgpt-kv owns `sp_kv/{types,utility_predictor}.rs` (clean).
  - Root crate keeps `src/sp_kv_forward_mod.rs` (the full pipeline functions).
  - Root's `sp_kv` re-export bridge combines both:
    ```rust
    pub mod sp_kv {
        pub use katgpt_kv::sp_kv::*;                  // types + utility predictor
        pub mod forward { pub use crate::sp_kv_forward_mod::*; }
        pub use crate::sp_kv_forward_mod::{GateBias, NoBias, SpKvForwardContext,
            attention_head_core, attention_head_gated, forward_sp_kv, forward_sp_kv_quant};
    }
    ```
  - All historical `katgpt_rs::sp_kv::*` paths continue to resolve.

### Phase 4 — Update root `Cargo.toml`

- Add `katgpt-spectral` and `katgpt-kv` to `[workspace.members]`.
- Add non-optional path deps.
- Convert feature flags to forwarders:
  - `spectral_quant = ["katgpt-spectral/spectral_quant"]` (was empty)
  - `kv_share = ["katgpt-kv/kv_share"]`
  - `osc_kv = ["katgpt-kv/osc_kv"]`
  - `cs_kv_probe = ["katgpt-kv/cs_kv_probe"]`
  - `shard_kv = ["katgpt-kv/shard_kv"]`
  - `sp_kv = ["katgpt-kv/sp_kv"]`
  - `still_kv = ["katgpt-kv/still_kv"]`
  - `kvarn = ["katgpt-kv/kvarn"]`
  - `targeted_precision = ["katgpt-kv/targeted_precision"]`

### Phase 5 — Update root `src/lib.rs` + dependents

- `pub mod spectralquant;` → `pub use katgpt_spectral as spectralquant;`
  (back-compat shim)
- `pub mod kv_share;` (gated) → `#[cfg(feature = "kv_share")] pub use katgpt_kv::kv_share;`
- Same for osc_kv, cs_kv_probe, shard_kv, sp_kv, still_kv, kvarn,
  targeted_precision.
- `src/types.rs`: drop the `QuantizedKVCache` trait def + the moved types;
  `pub use katgpt_types::{QuantizedKVCache, CompactionStrategy, CompactKVCache, KVChunk};`
- `src/fold/chain_folder.rs`: `crate::still_kv::*` paths now resolve via
  the re-export shim (no change needed).
- `src/attn_match/chunked.rs`: same — re-export shim covers it.
- Non-KV `spectralquant` consumers (`funcattn_compose`, `chiaroscuro`,
  `benchmark/infrastructure`): `crate::spectralquant::*` paths still
  resolve via the re-export shim. No source edits.

### Phase 6 — GOAT gate + commit

- `cargo check --all-features` at root
- `cargo check -p katgpt-spectral --all-features`
- `cargo check -p katgpt-kv --all-features`
- `cargo test -p katgpt-core --lib` (default regression)
- Spot-run KV-specific tests:
  - `bench_189_osc_kv_goat`, `bench_245_still_kv_goat`, `bench_280_cs_kv_probe_goat`,
    `kv_share_goat`, `test_147_shard_kv_goat`, `targeted_precision_goat`,
    `bench_sp_kv_quant`, `bench_spectralquant`
- Commit on `develop` with `refactor:` prefix.

## Tasks

- [x] Write issue doc
- [x] Phase 1: Move `QuantizedKVCache` + KV types to `katgpt-types`
- [x] Phase 2: Extract `katgpt-spectral` (8 files, 5059 LoC)
- [x] Phase 3: Extract `katgpt-kv` (8 modules + targeted_precision, ~7700 LoC)
- [x] Phase 4: Root `Cargo.toml` workspace + feature forwarders
- [x] Phase 5: Root `src/lib.rs` re-export shims + dependents
- [x] Phase 6: GOAT gate (`cargo check --all-features` + spot tests)
- [x] Commit on `develop`

## Impact

| Site | Change |
|------|--------|
| `katgpt-rs/tests/*.rs`, `examples/*.rs`, `src/{fold,attn_match,chiaroscuro,funcattn_compose,benchmark}/*` | None (re-export shims preserve `katgpt_rs::{spectralquant,still_kv,kvarn,...}::*` paths) |
| `riir-ai/crates/riir-{engine,gpu}` | Deferred to future consolidation issue (no changes this round) |

## TL;DR

Extract `src/spectralquant/` → `crates/katgpt-spectral/` and all KV code
(`kv_share`, `osc_kv`, `cs_kv_probe/`, `shard_kv/`, `sp_kv/`, `still_kv/`,
`kvarn/`, `targeted_precision`) → `crates/katgpt-kv/`. Pure `git mv` +
re-export shims + feature forwarders, no semantic changes. The KV
namespace now physically exists; the spectral quantization substrate is
shared cleanly between KV and non-KV consumers.
