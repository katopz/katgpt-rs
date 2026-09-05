# A `required-features` row can EXIST and be WRONG — the population

**Status:** IN PROGRESS — the free verdict is COMPLETE and gated for every
repo; the compiler pass has riir-clippy fully clean, the cross-repo
SUBSET-side suspect slice 17/17 clean, and the katgpt-rs full sweep running.
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

Measured over all 16 contract repos in under a second, and **gated** at
`max_invalid_rows = 0` by `scripts/required_features_static_gate.py` (a
`docs_gate.sh` check, katgpt-rs-scoped) so a new one reds the push that adds
it. The number was measured twice: the first run went through a `parse_rows`
that shadowed the package's declared-feature set with the row's own feature
list, so every feature was trivially "declared" and no finding was reachable.
The gate's plant-an-invalid-row canary found that within minutes of the code
landing; re-measured after the fix, still 0. A zero from an instrument whose
firing path was never exercised is not a measurement.

**Do not re-derive this check by hand — the obvious model is wrong**, and being wrong here is
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
| katgpt-rs | 621 | 379 | 24 (running) | 24 | 0 | 0 | 0 | 2026-09-06 |
| riir-ai | 512 | 307 | — | — | — | 0 | — | — |
| riir-train | 433 | 233 | — | — | — | 0 | — | — |
| riir-chain | 109 | 56 | — | — | — | 0 | — | — |
| riir-neuron-db | 51 | 37 | — | — | — | 0 | — | — |
| riir-game-sdk | 32 | 18 | — | — | — | 0 | — | — |
| riir-mmorpg-examples | 20 | 11 | — | — | — | 0 | — | — |
| seal-remake | 7 | 4 | — | — | — | 0 | — | — |
| **total** | **1,829** | **1,070** | 68 | 68 | 0 | **0** | 0 | |

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
row is usually just a smaller target. Only the compiler decides.

Taking only the **SUBSET side** — the row that is a strict subset of its
twin's, i.e. the one that would have lost the feature — collapses that to a
slice worth running first:

| repo | rows | SUBSET-side suspects | groups | verdict |
|---|---|---|---|---|
| katgpt-rs | 621 | 9 | 8 | **9/9 BUILDS** |
| riir-train | 433 | 6 | 5 | **6/6 BUILDS** (incl. 2 `riir-train-gpu` targets, on macOS) |
| riir-neuron-db | 51 | 2 | 1 | **2/2 BUILDS** |
| riir-ai | 512 | 10 | 5 | deferred — see below |
| riir-game-sdk | 32 | 1 | 1 | deferred — see below |
| riir-chain / riir-clippy / riir-mmorpg-examples / seal-remake | 180 | 0 | 0 | none to check |
| **total** | **1,829** | **28** | **20** | **17/17 checked clean** |

**The prior found nothing, and that is the honest result to record.** Every
checkable suspect builds at its own row. It is not evidence that the prior
works: the two instances that motivated it were both already repaired, so
this population contains zero known positives — the slice reproduces a zero
on a corpus with nothing to find. Treat it as a cheap first pass whose yield
is unmeasured, not as a validated finder.

The two deferrals are for a measured reason, and it is **not** "a sibling is
compiling" — riir-ai had no live cargo when checked. It is that both trees
carry **uncommitted sibling work** (riir-ai 6 files mid-extraction). A
`FAILS-TO-BUILD` there could be someone's half-finished edit rather than the
row, and a report that cannot tell those apart should not emit a verdict.
Re-run those two against a clean tree.

**28 rows, 20 cargo invocations** against the full sweep's 1,070 — minutes
rather than hours, aimed at the one shape both known defects had. It is a
prior, not a substitute: a wrong row that has no near-twin is invisible to
it, which is what the full sweep is for.

## Cost, and the box constraint

`--batch` runs one cargo invocation per (package, **EXACT** feature set).
Exact, never subset-covering: building a target at a superset of its own row
and seeing it succeed proves nothing, since the extra features may supply the
very import the row forgot. **1,829 rows collapse to 1,070 groups (1.71x)** —
invocations saved, *not* speedup. Measured on riir-clippy: **-11% and -8%
CPU-seconds** over two pairs, with **wall-clock flipping sign** (-12%, then
+13%) on a box carrying sibling builds. Expect more where a feature set
swings the dependency graph, near-nothing where it is shared.

**An interrupted sweep is the normal ending, so it resumes.** `--record
<jsonl>` appends each decided row as it lands and `--resume <jsonl>` reads it
back; a progress **log** is accepted as resume input too, because the first
long run is always started before anyone thinks about resuming it. Only
BUILDS / FAILS-TO-BUILD / NO-SUCH-FEATURE are skipped — TIMEOUT, ERROR and
UNSEEN are re-run, since they describe the box or a run that never reached
the target.

The katgpt-rs sweep is running detached (relaunched 2026-09-06 with 16 rows
already decided carried forward). Its state: `/tmp/katgpt_sweep2.log`
(progress) and `/tmp/katgpt_sweep.jsonl` (the record). To continue it rather
than start from zero:

```bash
cat /tmp/katgpt_sweep.log /tmp/katgpt_sweep.jsonl > /tmp/resume.txt
scripts/required_features_build_audit.py . --batch --resume /tmp/resume.txt --record /tmp/katgpt_sweep.jsonl
```

`read_prior` branches per LINE, so a file holding both shapes — the progress
log and the JSONL — is a valid resume input; that is how these two runs were
joined.

A full sweep in fresh `/tmp` target dirs is **not affordable on this disk**.
The workspace carried **378 GB of `target/debug/incremental`** against ~42 GiB
free on 2026-09-05; 168 GB was reclaimed from the four repos that were idle
and clean at the time (katgpt-rs 47.2, riir-train 116.3, riir-chain 2.8,
riir-neuron-db 2.4) and the rest belongs to repos with live sessions. Use a
repo's own `target/` when it is idle, and check `df` before starting.
