# A `required-features` row can EXIST and be WRONG — the population

**Status:** IN PROGRESS — two repos measured, one sweep running.
Instrument: `scripts/required_features_build_audit.py`.
Issue: riir-train `.issues/513`. Family: `cfg_gated_silent_zero_pass.md`.

Every audit in the `cfg_gated_target_audit.py` family treats a `[[test]]` /
`[[bench]]` as **protected** once it *has* a `required-features` row. That is
the right question for the failure they were built for. It says nothing about
whether the row is CORRECT, and a row that exists and is wrong is strictly
worse than a missing one: `cargo test --workspace` silently SKIPS the target,
`--all-features` BUILDS it (the union supplies whatever the row forgot), and
every audit counts it in the "w/ req-f" column.

## Three verdicts, and one of them is free

| verdict | decided by | cost |
|---|---|---|
| `NO-SUCH-FEATURE` | the manifest alone | **< 1 s for the whole workspace** |
| `FAILS-TO-BUILD` | the compiler, per (package, exact feature set) | ~28 s/row, ~25 s/group |
| `UNSEEN` | neither an error nor an artifact came back | — |

`UNSEEN` is the liveness sentinel: a row cargo never built gets **no verdict,
never BUILDS**. Silence is not evidence — cargo may have stopped at an
upstream unit — and reading it as success is the green-zero this whole family
exists to refuse.

## The free pass: 0 invalid rows over 1,829 (2026-09-05)

Measured over all 16 contract repos in under a second. **Do not re-derive
this check by hand — the obvious model is wrong**, and being wrong here is
expensive in the direction that manufactures defects.

`required-features` accepts **`dep/feat`** and `dep?/feat`, which name a
DEPENDENCY's feature rather than one of the package's own. A first cut
treated those as undefined and reported **10 riir-ai `riir-poc` benches** as
dead targets. They are not. A `/tmp` probe with a `compile_error!` canary
planted inside the target (cargo 1.98.1) settles it:

| invocation | compiles a `required-features = ["dep/extra"]` bench? |
|---|---|
| `cargo check --benches` | no — correctly skipped |
| `cargo check --features <ours-that-enables-dep/extra> --benches` | **yes** |
| `cargo check --all-features --benches` | **yes** |
| `cargo check --bench b1 --features dep/extra` | **yes** |
| `cargo check --bench b1` (unmet) | loud: `error: target ... requires the features`, exit 101 |

The canary is not optional: `cargo check` prints `Finished` and exits **0**
when it silently skips a target whose row is unmet, byte-identical to having
compiled it. Renamed (`package = `) and `[target."cfg(…)".dependencies]`
entries count as dependencies for this purpose too.

## The compiler pass, per repo

| repo | rows | groups | checked | BUILDS | FAILS | NO-FEAT | UNSEEN | date |
|---|---|---|---|---|---|---|---|---|
| riir-clippy | 44 | 25 | 44 | 44 | 0 | 0 | 0 | 2026-09-05 |
| katgpt-rs | 621 | 379 | running | — | — | 0 | — | 2026-09-05 |
| riir-ai | 512 | 307 | — | — | — | 0 | — | — |
| riir-train | 433 | 233 | — | — | — | 0 | — | — |
| riir-chain | 109 | 56 | — | — | — | 0 | — | — |
| riir-neuron-db | 51 | 37 | — | — | — | 0 | — | — |
| riir-game-sdk | 32 | 18 | — | — | — | 0 | — | — |
| riir-mmorpg-examples | 20 | 11 | — | — | — | 0 | — | — |
| seal-remake | 7 | 4 | — | — | — | 0 | — | — |
| **total** | **1,829** | **1,070** | 44 | 44 | 0 | **0** | 0 | |

The `NO-FEAT` column is complete for every repo — that is the free pass. The
rest is the sweep.

riir-clippy was run four times (per-row and batched, each order, cold `/tmp`
target dirs): 176 row-verdicts, **0 disagreements**. A whole clean repo is a
useful measurement here precisely because the ceiling this feeds is a zero.

## Two instances so far, one shape

- riir-train `9da3420f` — `test_cubecl_backward_grads` declared
  `["cubecl_runtime", "gemma_lora", "moved-gpu-tests"]` and omitted
  `gpu_training_resident`. Its sibling `test_issue424_batched_grad_divergence`
  carried the correct three. Fixing the row made it build and immediately
  reported **9 passed / 1 failed** — the wrong row was hiding a real defect.
- katgpt-rs `bench_001_pruners_goat` — `["bomber", "go"]`, `E0432`. Its twin
  `bench_001_pruners_goat_proof` had the identical defect and had ALREADY been
  fixed, by Issue 723 T7, which did not look at the file beside it.

Both are **a copy that lost a feature relative to a near-twin**. Pairing rows
inside a package by a ≥12-char common name prefix and a differing feature set
yields **167 suspect pairs** (katgpt-rs 38, riir-ai 80, riir-train 46,
riir-neuron-db 2, riir-game-sdk 1). That is an ORDERING heuristic and nothing
more — most `DIVERGE` pairs are legitimately different targets and a smaller
row is usually just a smaller target. Check the SUBSET side first; only the
compiler decides.

## Cost, and the box constraint

`--batch` runs one cargo invocation per (package, **EXACT** feature set).
Exact, never subset-covering: building a target at a superset of its own row
and seeing it succeed proves nothing, since the extra features may supply the
very import the row forgot. **1,829 rows collapse to 1,070 groups (1.71x)** —
invocations saved, *not* speedup. Measured on riir-clippy: **-11% and -8%
CPU-seconds** over two pairs, with **wall-clock flipping sign** (-12%, then
+13%) on a box carrying sibling builds. Expect more where a feature set
swings the dependency graph, near-nothing where it is shared.

A full sweep in fresh `/tmp` target dirs is **not affordable on this disk**.
The workspace carried **378 GB of `target/debug/incremental`** against ~42 GiB
free on 2026-09-05; 168 GB was reclaimed from the four repos that were idle
and clean at the time (katgpt-rs 47.2, riir-train 116.3, riir-chain 2.8,
riir-neuron-db 2.4) and the rest belongs to repos with live sessions. Use a
repo's own `target/` when it is idle, and check `df` before starting.
