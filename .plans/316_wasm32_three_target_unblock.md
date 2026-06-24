# Plan 316: wasm32 Three-Target Unblock (Native / Browser / CF Worker)

[← Index](README.md)

## Summary

Unblocked `katgpt-core` + `katgpt-rs` root for the `wasm32-unknown-unknown` target so
the freeze/thaw vessel (egg/shell dynamic-load), wasmi-as-host validator sandbox,
and ternary SIMD128 inference paths all compile for **browser**, **Cloudflare
Worker**, and **native app**. Three one-line-class fixes, all verified.

## Motivation

The modelless mandate hinges on freeze/thaw + dynamic weight loading. The Vessel
primitive (`secure_vessel` / `neuron_vessel`) is the egg/shell: weights frozen into
a BLAKE3-committed wire format, projected at runtime via wasmi. Doc 56
(`riir-ai/.docs/56_cf_workers_edge_architecture.md`) specifies three deploy targets
that all need this:

| Target | Crate / feature | Inference path |
|--------|-----------------|----------------|
| Native app | `riir-chaind` binary | CPU SIMD + ANE + wasmi host |
| Browser | `riir-chaind chain_node_browser` | WASM SIMD ternary + WebGPU |
| CF Worker | `seal-edge-worker` (`worker` crate) | WASM SIMD ternary |

A `cargo check --target wasm32-unknown-unknown` on `katgpt-core` and
`katgpt-rs --features secure_vessel` was failing, blocking all three non-native
targets. Root causes were NOT architectural — three missing cfg/feature declarations.

## Tasks

- [x] **T1: Fix `argmax.rs` missing import** (`katgpt-core/src/simd/argmax.rs`)
  - Added `use crate::simd::simd_max_f32;`
  - Regression from the `simd.rs` → `simd/` folder refactor (`9d0ba6ee`). Native CI
    was green because the `aarch64`/`x86_64` branches early-return before reaching
    line 31; only wasm32 (which falls into the scalar branch) exercised the
    unimported call.
- [x] **T2: Add getrandom wasm32 backends** (`katgpt-rs/Cargo.toml`)
  - `cargo tree` showed TWO getrandom versions via `bevy_ecs → bevy_utils → ahash`:
    `0.2.17` (const-random-macro, uuid) and `0.3.4` (ahash direct).
  - getrandom 0.3 renamed the wasm feature: `js` (0.2) → `wasm_js` (0.3).
  - Added both under `[target.'cfg(target_arch = "wasm32")'.dependencies]`:
    ```toml
    getrandom = { version = "0.3", features = ["wasm_js"] }
    getrandom_02 = { package = "getrandom", version = "0.2", features = ["js"] }
    ```
- [x] **T3: Enable bytemuck `extern_crate_alloc`** (`katgpt-rs/Cargo.toml`)
  - `transformer.rs::load_mtp_projection` uses `bytemuck::pod_collect_to_vec`,
    which lives in `bytemuck::allocation`, gated by `feature = "extern_crate_alloc"`.
  - On native, `plotters → image` transitively unifies the feature; wasm32 skips
    plotters, so the feature was missing.
  - Changed `bytemuck = "1"` → `bytemuck = { version = "1", features = ["extern_crate_alloc"] }`.

## Verification

All gates run on 2026-06-24, M3 Max, develop branch:

| Gate | Command | Result |
|------|---------|--------|
| G1 — katgpt-core wasm32 | `cargo check -p katgpt-core --target wasm32-unknown-unknown --no-default-features` | ✅ Finished, 2 pre-existing warnings |
| G2 — plasma_path SIMD128 | `RUSTFLAGS="-C target-feature=+simd128" cargo check -p katgpt-core --target wasm32-unknown-unknown --features plasma_path --no-default-features` | ✅ Finished |
| G3 — secure_vessel wasm32 | `cargo check -p katgpt-rs --target wasm32-unknown-unknown --features secure_vessel --no-default-features` | ✅ Finished (vessel egg/shell + wasmi host compiles) |
| G4 — bomber-wasm wasm32 | `cargo check -p katgpt-rs --target wasm32-unknown-unknown --features bomber-wasm --no-default-features` | ✅ Finished (wasmi-as-host validator sandbox compiles) |
| G5 — native no-regression (core) | `cargo check -p katgpt-core` | ✅ Finished |
| G6 — native no-regression (root) | `cargo check -p katgpt-rs` | ✅ Finished |
| G7 — katgpt-core lib tests | `cargo test -p katgpt-core --lib` | ✅ 509 passed; 0 failed |
| G8 — downstream riir-chain | `cargo check -p riir-chain --no-default-features` | ✅ Finished |
| G9 — downstream riir-neuron-db | `cargo check -p riir-neuron-db --no-default-features` | ✅ Finished |
| G10 — seal-edge-worker wasm32 | `cargo check -p seal-edge-worker --target wasm32-unknown-unknown` | ✅ Finished |

## Conceptual Clarifications (recorded to prevent re-confusion)

### wasmi vs wasm32-unknown-unknown (two different things)

- **wasmi** = a WASM *interpreter/host* that runs *guest* `.wasm` modules. Used
  inside native `riir-chaind` (`chain_wasm` feature) to sandbox untrusted
  validator blobs. Direction: our process *runs* external WASM.
- **`wasm32-unknown-unknown`** = a compile *target* for our own Rust code.
  Direction: we *compile* our node/inference into `.wasm` for browser/CF/wasmtime.

These are orthogonal. wasmi-as-host CAN compile to wasm32 (pure Rust, no JIT) —
this is the "WASM-in-WASM" path that lets a browser node locally sandbox-validate
untrusted validator modules. Before Plan 316 it was broken by the getrandom gap;
now it compiles.

### Freeze/thaw has two runtime models

| Model | Mechanism | wasm32 status |
|-------|-----------|---------------|
| **A — Native apply** | Freeze → BLAKE3 blob → thaw → load bytes → apply via `simd_ternary_matvec` (compiled wasm32 SIMD128) | ✅ Works (Model A is the doc-56 edge design) |
| **B — Vessel projection** | Freeze → vessel w/ embedded `.wasm` → wasmi interprets the vessel → projection result out (host never sees weights) | ✅ Compiles (Plan 316); use judiciously on CF due to 10-50ms CPU budget |

`freeze.rs` (`MerkleFrozenEnvelope`, `merkle_freeze` feature) is pure BLAKE3 +
MerkleTree — no wasmi, no getrandom — and has always worked on all targets. Only
the Vessel *projection* path (Model B) needed the getrandom + bytemuck fixes.

### Doc 56 reality check

- `cf_workers = ["browser_gpu"]` in doc 56 is **aspirational** — grep of
  `riir-chain/Cargo.toml` returns zero matches. The real CF target today is
  `seal-edge-worker` (uses Cloudflare's `worker` crate + D1/R2/Durable Objects),
  which builds clean to wasm32 but does **not yet depend on katgpt-core/riir-chain**.
- The moment `katgpt-core` is wired into `seal-edge-worker`, it inherits the
  fixes from this plan (the wasm32 build stays green).

### Known latent issues NOT fixed by this plan

- ~~`riir-chaind chain_node_browser` fails with 8 `web-sys` errors~~ **RESOLVED
  2026-06-24 (Plan 316 follow-up).** The original note claimed the `web-sys` dep
  was missing the `WebTransport` / `WebTransportBidirectionalStream` /
  `WebTransportDatagramDuplexStream` feature flags. **That diagnosis was wrong** —
  the flags were already present in `crates/riir-chaind/Cargo.toml:114-123`.
  `cargo tree -e features -i web-sys` confirmed all three features were active.
  The real root cause: web-sys 0.3.102 gates these types behind
  `#[cfg(web_sys_unstable_apis)]` (see `gen_WebTransport.rs` line 5), so the types
  don't exist unless that rustc cfg is passed. `crates/riir-chaind/build_wasm.sh:24`
  already set this for production builds via `RUSTFLAGS`, but bare `cargo check`
  hit the 8 errors. Fix: added `riir-chain/.cargo/config.toml` scoping
  `--cfg web_sys_unstable_apis` to `[target.wasm32-unknown-unknown]`, so
  `cargo check --target wasm32-unknown-unknown --features chain_node_browser`
  works out of the box. Verified: builds clean (1 pre-existing dead-code warning,
  0 errors). All three targets now compile: native ✅ browser ✅ CF Worker ✅.
- `seal-edge-worker/src/runtime/wasm_compat.rs:213` has a `0xCA` filler standing
  in for `web_sys::crypto().getRandomValues()` (`TODO(F-140)`). Any crypto path
  routing through this on CF uses non-random randomness. Audit before production.
- WASM SIMD128 coverage gap: only `simd_ternary_matvec` has a real wasm32 SIMD128
  kernel. All other SIMD ops (`simd_dot_f32`, `simd_matmul_rows`, `simd_sigmoid_*`,
  `simd_exp_*`, etc.) fall to scalar on wasm32 even with `+simd128`. Research 226's
  "AVX2 → NEON → WASM simd128 → scalar" tier is only realized for the ternary path.
  Tracked as optimization issue `.issues/007_wasm_simd128_coverage_gap.md`.

## Files Changed

- `crates/katgpt-core/src/simd/argmax.rs` — +1 line (T1)
- `Cargo.toml` — getrandom wasm32 block (T2) + bytemuck feature (T3)

## TL;DR

Three cfg/feature declarations unblocked the entire wasm32 target surface for the
modelless freeze/thaw stack: (1) `argmax.rs` missing import, (2) getrandom `wasm_js`
(0.3) + `js` (0.2) for the bevy-pulled version split, (3) bytemuck
`extern_crate_alloc` for `pod_collect_to_vec`. Verified: `katgpt-core`,
`secure_vessel`, `bomber-wasm`, and `plasma_path +simd128` all compile to
`wasm32-unknown-unknown`; native + 509 lib tests pass; downstream riir-chain /
riir-neuron-db / seal-edge-worker unaffected.

**Follow-up (2026-06-24):** the "browser target web-sys gap" listed above as a
known issue was a misdiagnosis — the cargo features were already correct. The
real blocker was web-sys's `#[cfg(web_sys_unstable_apis)]` gate on WebTransport
types. Fixed via `riir-chain/.cargo/config.toml`. All three deploy targets
(native / browser / CF Worker) now compile clean.
