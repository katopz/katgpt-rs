# Issue 004: `percepta_compile` Feature Breaks `--all-features` — Missing Vendored `runtime.h` — **RESOLVED**

**Date:** 2026-06-24
**Status:** **CLOSED — RESOLVED.** `cargo check --all-features` now passes (EXIT 0). runtime.h vendored at tracked location `src/percepta/runtime.h`; latent masked test bugs fixed across 4 files.
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

## Resolution (2026-06-24)

**Chosen option: hybrid of (1) and (2).** The upstream `runtime.h` was restored
from `Percepta-Core/transformer-vm@main` (4894 bytes, Apache-2.0) but placed at
a **tracked** location `src/percepta/runtime.h` instead of the gitignored
`.raw/transformer-vm/...` path. The `include_str!` was updated to the idiomatic
`include_str!("runtime.h")` (relative to `compile.rs`).

### Why not just restore `.raw/transformer-vm/...`?

`.raw/` is gitignored (`.gitignore:2`). Restoring the file there would only fix
**local** builds — every fresh clone / CI run would still break. Moving the
header to `src/percepta/` (next to `compile.rs`) makes it version-controlled
and follows Rust convention for embedded assets. The `.raw/` dir stays
"reference-only upstream trees"; a header that's actively compiled in belongs
next to the code that uses it.

### Vendored `runtime.h` contents (matches upstream exactly + SPDX header)

Provides the C runtime that user programs compiled with `-nostdlib` link against:

- `putchar(int ch)` — imports `env.output_byte` (the ONE import the WASM lowerer
  recognizes → emits `("output", 0)` dispatch entry).
- `print_str(const char *s)` — loop calling putchar.
- `parse_int` / `print_int` — non-negative int I/O via repeated addition
  (no MUL, no arrays, so clang keeps everything in WASM registers).
- `sscanf(str, fmt, ...)` — minimal `%d` only.
- `printf(fmt, ...)` — `%d` / `%s` / `%c` / `%%`.
- All helpers `always_inline` / `noinline` to avoid stack frames.

This satisfies every e2e test's contract: hello.c uses `print_str`, collatz.c
uses `sscanf`+`printf`, the simple test uses `putchar`.

### Latent masked test bugs also fixed

Restoring `runtime.h` unblocked compilation of the entire `percepta_compile`
module for the **first time** — which surfaced ~30 pre-existing latent test bugs
that had never compiled (the `include_str!` failure masked them). Fixed:

| File | Bug | Fix |
|------|-----|-----|
| `src/percepta/cumsum.rs` | `cs.insert(.., i as i64)` but `insert` takes `seq: i32` (×2) | `i as i32` |
| `src/percepta/compile.rs` | `op == "output"` where `op: &&str` (×11, masked) | `*op == "output"` |
| `src/percepta/runner.rs` | same `&&str == &str` pattern (×2) | `*op == ...` |
| `tests/integration.rs` | `head.insert(.., i as i64)` but takes `seq: i32` (×1) | `i as i32` |
| `tests/test_percepta_rust_wasm.rs` | same `&&str == &str` pattern (×12) | `*op == ...` |
| `tests/bench_064_futamura_evaluator.rs` | `ProgramGraph::num_dimensions()` never existed (×6); move-after-match on `Runner::evaluate` result (×2) | `all_dims.len()` + match-by-reference |

## Verification

```bash
# Issue 004 headline — was: 1 hard error; now: EXIT 0
cargo check --all-features

# Default build unaffected (G3 no-regression)
cargo check                                # EXIT 0
cargo test -p katgpt-core --lib            # 509 passed, 0 failed

# percepta_compile now compiles AND its tests run (clang+wasm32 present):
cargo check --features percepta_compile    # EXIT 0
cargo test --features percepta_compile --lib -- percepta::
    # 339 passed, 0 failed, 13 ignored — incl. all 4 C→WASM e2e tests:
    #   test_e2e_compile_hello_c / collatz_c / c_to_wasm_only / no_input_program
cargo test --features percepta_compile --test test_percepta_rust_wasm
    # 18 passed, 0 failed, 6 ignored (Rust→WASM→prefix pipeline)
```

## Known pre-existing failure (NOT introduced by this fix)

`bench_064_futamura_evaluator::tests::proof_futamura_specialized_has_fewer_dimensions`
fails: specialized graph has **300** dims vs universal **216**. The assertion
`specialized_dims <= universal_dims` is a flawed premise — Futamura specialization
trades generic instruction-fetch attention for per-instruction dimensions, so
`all_dims.len()` *grows* for a baked-in program while attention complexity
shrinks. This test never compiled before (the `num_dimensions()` method it called
never existed), so the failure is a pre-existing latent logic bug. The correct
metric for "specialization reduces work" is lookup/attention-head count, not raw
dimension count — but changing the assertion would be guessing the author's
intent, so it is left as-is and documented here.

## TL;DR

**RESOLVED.** `percepta_compile`'s `include_str!` pointed at a gitignored
`.raw/` path that was never populated, breaking `cargo check --all-features`.
Restored the upstream Percepta `runtime.h` (Apache-2.0) to a tracked location
`src/percepta/runtime.h` and updated the `include_str!` to `"runtime.h"`.
Unblocking the module's first-ever compile surfaced ~30 latent masked test bugs
(`&&str == &str`, `i64` vs `i32`, a never-existent `num_dimensions()`, and a
move-after-match) — all fixed across 6 files. `cargo check --all-features` now
EXIT 0; 339 percepta lib tests + 18 Rust→WASM integration tests pass including
all 4 C→WASM e2e tests; katgpt-core G3 regression clean (509 passed). One
pre-existing logic assertion in `bench_064` (flawed Futamura dim-count premise)
remains and is documented.
