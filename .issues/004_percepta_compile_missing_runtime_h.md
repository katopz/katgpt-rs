# Issue 004: `percepta_compile` Feature Breaks `--all-features` — Missing Vendored `runtime.h`

**Date:** 2026-06-24
**Status:** Open — blocks `cargo check --all-features`
**Origin:** Surfaced during `--all-features` audit of pre-existing compile errors
**Severity:** Medium (default builds unaffected; only fires when `percepta_compile` feature is on)

## Problem

`src/percepta/compile.rs:55` uses a compile-time `include_str!` of a vendored
external file:

```rust
pub const RUNTIME_H: &str =
    include_str!("../../.raw/transformer-vm/transformer_vm/compilation/runtime.h");
```

The target file does **not exist** on disk. The `.raw/transformer-vm/` directory
contains only a `.DS_Store` — the vendored Percepta `transformer-vm` source tree
(including `compilation/runtime.h`) is missing.

This causes a hard compile error whenever the `percepta_compile` feature is
enabled, which includes:
- `cargo check --all-features`
- `cargo check --features full`
- Any explicit `--features percepta_compile`

## Impact

- **Default builds: unaffected** — `percepta_compile` is NOT in the default
  feature set (confirmed: `cargo check` passes in 0.41s).
- **`--all-features`: broken** — 1 hard error (`include_str!` file not found).
- **CI feature guard: would fail** if one existed (katgpt-rs has no
  `scripts/ci_feature_guard.sh` unlike riir-chain / riir-neuron-db).

## Root Cause

The `.raw/transformer-vm/` directory was not fully populated. This is a vendored
copy of Percepta's `transformer-vm` project (Apache-2.0, per the file header in
`compile.rs`). The `runtime.h` header is required by the C→WASM compile pipeline
(`compile_c_to_wasm` writes it to a temp dir and passes it to clang via
`-include`).

A workspace-wide search (`find /Users/katopz/git -name "runtime.h" -path "*transformer*"`)
found no copy anywhere in the local repos. The file must be restored from the
upstream Percepta source.

## What `RUNTIME_H` Is Used For

1. `write_runtime_h(dir)` — writes the embedded header to disk for clang.
2. `compile_c_to_wasm(c_source, runtime_h_path)` — clang `-include`s it.
3. `compile_program(c_source, input_str)` — full pipeline, calls both above.
4. Tests: `test_runtime_h_is_valid` asserts content contains `putchar`,
   `compute`, `#ifndef`.

## Resolution Options (user to decide)

1. **Restore the vendored file** — copy `runtime.h` from the upstream Percepta
   `transformer-vm` repo into `.raw/transformer-vm/transformer_vm/compilation/`.
   This is the correct fix.
2. **Remove the `include_str!`** and gate the `compile` module behind a
   runtime-configurable path (e.g., env var pointing to an external copy).
   More work, less self-contained.
3. **Accept the breakage** — document that `percepta_compile` requires manual
   setup, exclude it from `--all-features` CI if a guard script is added later.

## Why This Was Not Fixed In-Session

The file is a vendored external dependency with specific C runtime declarations
(`putchar`, `compute`, `#ifndef` guards). Recreating it from scratch would
require knowledge of the Percepta `transformer-vm` ABI contract. Filing as an
issue rather than guessing.

## Related

- 7 other pre-existing `--all-features` compile errors were fixed in this session
  (commit pending): `feedback.rs` Display, `wasm_proof_witness.rs` lifetime,
  `evaluator.rs` borrow-after-move + double-borrow, `buffer.rs` type annotation,
  `feedback_bandit.rs` HashMap key type, `skill_catalog.rs` papaya `get_mut`.
- After those fixes, this missing-file error is the **sole remaining**
  `--all-features` blocker.
