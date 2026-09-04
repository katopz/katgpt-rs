# Bench 701 — pricing the full-workspace EXECUTION (Issue 718 T3(a))

**Status:** MEASURED 2026-09-04 on a quiet M3 Max (load avg 4.01 at launch).
**Verdict: the full `--all-features --release` execution stays DISPATCH-ONLY** —
not on cost (3.21 CPU-hours is affordable weekly) but because **it cannot pass
as-is**: `--all-features` was never a checked *test* configuration. Feature:
n/a (this is a CI-cost measurement, not a flag promotion).

Durable narrative of the axis: `.docs/10_audits/ci_compile_vs_execute_axis.md`.
The reds it found are tracked in `.issues/723`.

## Why price at all

Issue 718 measured that no automatic trigger in this repo ran `cargo test` —
`cargo clippy --workspace --all-targets --all-features` compiles every target
and executes none. The owner authorized a **scoped** weekly job (T3(b), landed)
and required the full-workspace run to be **priced on a quiet box before** any
scheduled job, because a scheduled job that has never been priced is exactly
the cost-blindness the issue exists to prevent.

**Price in CPU-seconds, not wall-clock.** A wall-clock figure from this
workstation is uninterpretable — sibling sessions held it at load average
44-87 for a whole day. CPU-seconds and peak RSS measure what a process
*consumed* rather than how long it *waited*, and are load-invariant to within
0.11 over a 2x load swing (seal-remake `.benchmarks/002_png_vs_ktx2_host_cpu_rss.md`).
CPU-seconds is also closer to what Actions bills.

## The measurement

`/usr/bin/time -l` on a cold, isolated `CARGO_TARGET_DIR=/tmp/i718a_cold`,
HEAD `172f5520` (which already carries 3 gate fixes this same run produced).

| metric | cold full run |
|---|---|
| command | `cargo test --workspace --all-features --release --no-fail-fast` |
| wall | **2,728.6 s (45.5 min)** |
| CPU | **11,542 CPU-s** (10,796 user + 746 sys) = **3.21 CPU-hours** |
| `maximum resident set size` | 15,454,306,304 B = **14.39 GiB** |
| `peak memory footprint` | 1,045,071,240 B = 1.05 GB |
| instructions retired | 220,769,822,175 |
| cycles elapsed | 106,265,562,265 |
| target dir | 3.1 GB (release, no debuginfo) |
| binaries / result lines | 512 / 542 |
| verdict | **497 suites ok / 45 targets FAILED** |

**Read both memory rows, never one.** `maximum resident set size` is the peak
across the whole process tree (cargo + N parallel rustc + N test binaries) and
is what a CI runner must actually provision — **14.4 GiB**. `peak memory
footprint` is a much narrower macOS accounting field and reads 1.05 GB. Quoting
only the second would size a runner 14x too small; they are 14x apart and
both are printed by the same command.

### The compile dominates, and that reframes the cost

A warm comparator at the same feature set (target pre-built) read **1,327.9 s
wall / 2,319 CPU-s**. So:

| phase | CPU-s | share |
|---|---:|---:|
| cold compile | ~9,223 | ~80% |
| test execution | ~2,319 | ~20% |

**Executing the tests is the cheap part.** For scale, the landed T3(b) scoped
job (default-features `--lib`, 2,177 assertions) is roughly **30x cheaper**
than this full run. A recurring full job would be paying ~80% for a rebuild.

### `--no-fail-fast` is mandatory for any future attempt

Default cargo aborts at the first failing package. That truncates both the
failure list *and* the cost figure, and it burned five launch cycles here
before an enumeration pass completed. Two independent wrong signals from one
cause — the same shape as the concurrent-target-dir false RED AGENTS.md
documents.

## The pricing run's real finding

**The 45 failures are not cost noise. `--all-features` is not a supported test
configuration** — the katgpt-rs twin of riir-ai Issue 830's "`--all-features`
was never a checked configuration". Full six-class taxonomy and per-class
actions: **`.issues/723`**. In summary:

| class | targets | nature |
|---|---:|---|
| A — wall-clock / throughput bars | 8 | calibrated per-feature, read backwards under load |
| A2 — timer-resolution degenerate (`0 ns` → `NaN%`) | 3 | instrument measured nothing |
| B — missing wasm `__heap_base` linker flag | 2 (12 tests) | environmental |
| C — fixture RNG-stream drift under feature unification | 9 | correct per-feature, **meaningless** unified |
| D — hard-coded `target/release` | 1 | **FIXED** — unpassable under the mandated `CARGO_TARGET_DIR` workflow |
| E — quality / correctness reds | 13 | needs a per-target read; one is a library panic |
| F — doc-tests | 8 crates | **FIXED** — never compiled at any revision |

**Class C is the one that decides the cadence question.** Fixture weights are
seeded from an RNG stream whose draw count depends on feature-gated struct
fields, so under `--all-features` every committed platform pin reads a
different hash (`fixture hash 54473b7c30dfb793 is neither the aarch64 nor the
x86_64 pin`). Those tests are *correct under their own feature set and
meaningless under unification*. No amount of budget makes them green in this
configuration; re-pinning them to the unified hash would destroy the per-feature
claim they exist to make.

**Class F is the one nothing could have caught.** `--all-targets` does **not**
include doc-tests, so the gate whose entire purpose is to compile everything
had never compiled a doc example — 8 crates' doctests had never been built at
any revision. Fixed and verified in this session: `cargo test --workspace
--all-features --release --doc` now reads **34 suites, 98 passed, 0 failed,
111 ignored**. That is a new blind-spot row in AGENTS.md's table.

## Cost conclusion — decision input, not a decision

The full `--all-features` execution **cannot pass as-is regardless of budget**.
Making it green requires either per-target triage pins (the seal-remake
`e1ead85` shape — name each target, floor its count, tolerate its known reds)
or accepting a permanently-red weekly job, which is worthless as a gate. The
three honest options, now priced:

- **(i) keep T3(b) as the executed scoped job** — landed, ~zero marginal cost,
  machine-invariant core; the full run stays dispatch-only for triage
  campaigns. **This is the standing state.**
- **(ii) a per-target triaged weekly integration job** — the seal-remake shape
  over the ~470 default-runnable integration targets. Machine cost ≈ the
  execution half (~2,300 CPU-s) + one feature-set compile. Engineering cost =
  triaging `.issues/723`'s red targets into expect-red pins.
- **(iii) per-feature-set GOAT-bench lanes** — the gates each run under their
  OWN committed feature set, the way they were calibrated. Most honest
  numerically, highest matrix cost. This is the only option that makes Class C
  meaningful rather than suppressed.

## Defects the pricing run produced en route

Fixing these was the price of getting a run to complete at all, and all were
scanned workspace-wide rather than spot-fixed:

| fix | commit | class |
|---|---|---|
| `convergence_cadence` G4 — E0432, module-level import of debug-only alloc counters (regression of the 720-T1 landing) | `dd734f2f` | debug/release profile axis |
| `bckvss builder_rejects_bad_segment_len` — `#[should_panic]` on a `debug_assert!` contract; 5 of 6 workspace sites were already gated, this was the one | `5fa7b5f1` | debug/release profile axis |
| `d2f capture_q_row` mask-slot mass — **a real latent correctness bug** in the Issue-587 ExactQ law (rows summed to `1 + exp[mask]/sum_exp`), surfaced only when `gated_mlp`'s extra weight draw shifted the RNG stream | `d3454eff` | correctness |
| `recirculation` G2 timing 1 → 2 µs | `a9576e20` | load-fragile gate, re-pinned with a measured band |
| `bench_668` pair-scaling 1.6x → 2.5x | `ff6a4d46` | load-fragile gate |
| `bench_693` mi-est 1.5 → 3 ms | `172f5520` | load-fragile gate |
| 26 doc-tests over 8 crates + `bench_294_ict_g6` target dir | this session | Classes F + D |

The `d2f capture_q_row` one is the argument for ever running this at all: a
real correctness bug in a shipped law, reachable only because a feature's extra
RNG draw shifted a fixture stream. No compile gate can find that.

## Honest caveats

- **One configuration, one platform, one profile.** `--all-features --release`
  on aarch64-macOS. The `-p` vs `--workspace`, platform and profile axes apply
  to execution exactly as to compilation. A green here would not be total
  coverage, and this red list is not exhaustive of what other configurations
  would show.
- **The doc-test phase ran against edited sources.** The Class F fixes landed
  while this run was in its test-execution phase, so its doctest results are a
  *mixed* state and cost includes a partial katgpt-core rebuild. The doctest
  verdict quoted above is from the clean dedicated `--doc` re-run, not from
  this log.
- **Two runs disagreed slightly on which targets failed** (45 vs 47, with
  `bfcf_lsh_cms_goat`, `bench_582_trit_pack_goat`, `bench_680_signed_coupling_goat`
  appearing in one and not the other). That non-determinism is itself the
  Class A finding: a gate whose verdict the box decides cannot run in CI.
