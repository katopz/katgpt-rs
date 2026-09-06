# AGENTS.md — katgpt-rs

The global `~/.agents/` rules apply; this file documents repo-local context
that supplements them.

History, resolved-issue records, gate narratives, collision precedents:
`HISTORY.md`. Removed issue files: git history.

## Boundary contract — read `BOUNDARY.md` first

[`BOUNDARY.md`](BOUNDARY.md) is the authoritative contract: what this repo
**owns**, what it **does not own** (with the correct home for each), the
crate-granular **allowlist**, and the **drift ledger**. On any conflict with
prose in this file, BOUNDARY.md wins.
- **Domain test:** is this a **modelless inference primitive** with no riir dep (this repo is upstream of everything)? NO → it belongs in another repo; file there.
- Read it before adding any dep, crate, module, System impl, or vocabulary type.
- Enforcement: `../riir-ai/scripts/ci_boundary_contract.sh` — undeclared cross-repo dep, drift row without its open issue, contract-vs-measured-graph drift. Run boundary checks VIA the `boundary-guard` skill, not ad-hoc greps.
- Found a violation? File the issue FIRST (`.issues/NNN_boundary_*.md`), add the drift row, then fix. Closing the issue removes the row in the same commit.

## Modelless-first mandate (the core principle)

**This repo ships modelless inference primitives.** No training, no backprop,
no gradient descent. The only weight mutations allowed at runtime are:

1. **Freeze/thaw** — swapping a frozen snapshot (atomic, versioned, BLAKE3-checked).
2. **Raw/lora hot-swap** — a **deterministically constructed** (not trained)
   LoRA overlay via `LoraPair { reader, writer }` (Plan 025).
3. **Latent-space updates** — direction-vector projections, sigmoid gates,
   routing tables; latent state, NOT base weights.

**MANDATORY: exhaust modelless paths before deferring to riir-train.** Before
deferring ANY gate, mechanism, or plan task ("this needs training"), check the
three paths above first (research skill §3.5,
`.agents/skills/research/SKILL.md`). Systematic, characterizable biases are
modelless-correctable candidates, NOT automatic riir-train dependencies — for
a known, named bias ("signal doubled", "position offset", "attention
asymmetry"), try a deterministic reader-LoRA or freeze-state correction before
concluding "needs gradient descent." Canonical-failure story: HISTORY.md.

## Build Commands

```bash
# Default features (the GOAT-validated, promoted primitives)
cargo check
cargo test -p katgpt-core --lib

# Single feature
cargo check --features <feature_name>

# All features
cargo check --all-features

# Specific feature's tests
cargo test -p katgpt-core --features <feature_name> --lib
```

### The full gate — none of the above is a whole-repo claim

Every command listed above is narrow in at least one **independent** axis, and
a green result says nothing about what it compiled to nothing:

| Axis | Blind spot |
|---|---|
| `check` vs `clippy` | two `cargo heal` escape classes are rejected by clippy's typeck and accepted by `check` (E0689 ambiguous-integer, E0631 deref-coercion in `redundant_closure`) |
| default vs `--all-features` | non-default gated code compiles to **nothing** |
| `-p <crate>` vs `--workspace` | a crate's own non-default feature can be switched on by the ROOT crate's defaults once the root is in the selected set — and per-crate runs silently *shrink* coverage |
| no `--all-targets` | skips every test / bench / example — which is where gated code lives |
| dev vs `--release` | `debug_assertions` is always **ON** in dev, so every item behind `#[cfg(debug_assertions)]` — and everything that depends on one — only ever compiles in the configuration where it works. **Neither profile is the safe default — the profile is part of the claim.** |
| `--all-targets` vs **doc-tests** | `--all-targets` does **not** include doc-tests — only `cargo test --doc` reaches them (`.issues/723` Class F) |
| **compile vs EXECUTE** | every axis above is about *compilation*. The scoped core (katgpt-rs + katgpt-core `--lib` at default features, count floors) is EXECUTED weekly (`test.yml` + `scripts/test_gate.sh`); the other 477 integration-test and 176 bench targets are executed by nothing automatic, and `--all-features` is not a supported TEST configuration (fixture RNG streams and GOAT calibrations are per-feature). An uninvoked assertion is *unknown*, not passing |

So before claiming a repo-wide green, run:

```bash
cargo clippy --workspace --all-targets --all-features --keep-going -- -D clippy::needless_range_loop -D clippy::map_clone -D clippy::iter_cloned_collect -D clippy::identity_op -D clippy::bool_comparison -D clippy::manual_is_multiple_of -D clippy::collapsible_if -D clippy::map_all_any_identity -D clippy::unnecessary_cast -D clippy::manual_repeat_n -D clippy::question_mark -D clippy::empty_line_after_outer_attr -D clippy::unusual_byte_groupings -D unused_mut -D unused_parens
```

The `-D` list (Issue 701 R3b, 2026-09-03) is the mechanical lints whose
all-features warning surface was healed to ZERO residual — a lint with
residual > 0 must NOT be added to it. `--keep-going` is not optional: without
it the run stops at the first failing target and under-reports. Don't run it
by hand — `scripts/full_gate.sh` is the assertion (it refuses to report a pass
off macOS, where the `target_os = "macos"` device backends compile to nothing
even with `--all-features`, and checks that this document still quotes the
command it runs).

**The inverse holds too:** running **on** macOS silently drops every
`not(target_os = "macos")` backend, `--all-features` included — **a platform
is part of the claim, exactly as the profile is.** Typecheck that half from
the M3 (`cargo check` never links; `--canary` is not optional — it requires
`E0425` from a planted undefined call, because otherwise "Finished" is
indistinguishable from the modules compiling to nothing):

```bash
scripts/check_platform_gated_modules.sh ../riir-train riir-train-gpu numeric_drift_cuda
scripts/check_platform_gated_modules.sh --canary ../riir-train riir-train-gpu \
    crates/riir-train-gpu/src/numeric_drift_tap.rs numeric_drift_cuda
```

Trigger health: `.github/workflows/full_gate.yml` runs the weekly rot-check
cron from the default branch; `scripts/ci_gate_coverage.py` reports which
declared triggers can actually fire, per workflow, per repo.

## Docs gate + drift sweeps

`scripts/docs_gate.sh` runs the manifest/doc/skill drift assertions (~11s at
last measure — re-time before quoting); `.github/workflows/docs_gate.yml`
runs it per-push on **`main` only** — develop pushes do not fire it, so run
`./scripts/docs_gate.sh` locally for develop work. One line per check:

| check | asserts |
|---|---|
| `count_features.py` | flag counts in README + examples/README vs every manifest |
| `bench_doc_audit.py` | default-on / opt-in labels in .benchmarks + .docs vs Cargo defaults |
| `cargo_comment_audit.py` | inline Cargo.toml comments vs the default closure |
| `skill_repo_set_gate.py` | hand-typed repo sets in SKILL.md command blocks (Issue 703) |
| `agents_repo_set_gate.py` | AGENTS.md §Repo count membership vs `scripts/repo_set.txt` — pins the paragraph below |
| `cfg_gated_floor_gate.py` | `#![cfg]`-gated targets that report a green 0-pass (Issue 713) |
| `orphaned_attr_gate.py` | a `#[cfg]` separated from its item by a blank line |
| `percentile_floor_gate.py` | a percentile index that lands on n-1 and so reports the MAX |
| `numbering_gate.py` | a number allocated twice, or a stale/malformed `.highwater` (Issues 724, 725) |
| `docs_gate_paths_sync.py` | docs_gate.yml's two hand-duplicated trigger `paths:` lists stay identical |
| `required_features_static_gate.py` | a required-features row naming a feature its package cannot enable (Issue 513) |
| `cfg_row_implication_gate.py` | a required-features row that BUILDS and compiles its target to NOTHING (Issue 513) |
| `population_sync_gate.py` | the six independent contract-repo predicates must agree |

The `CHECKS` count is deliberately not written here — it drifted once, which
is exactly the drift this gate exists to catch.

Workstation-only cross-repo sweep family — `docs_drift_sweep.py`,
`numbering_drift_sweep.py`, `required_features_drift_sweep.py`,
`percentile_drift_sweep.py`, `cfg_gated_drift_sweep.py`,
`cfg_row_implication_drift_sweep.py` (every contract repo, on demand),
`sibling_docs_drift.yml` (reusable workflow, one caller), and
`ci_gate_coverage.py` (report, always exit 0: which repos gate their full
compile+lint surface in CI, and whether anything automatically starts it).
NOT in docs_gate's CHECKS — CI's single checkout would derive an empty
population and print a confident green over zero repos. Population derived
(BOUNDARY.md + `.git`); expectations committed in `scripts/*_floors.txt`.

## cfg-gated targets — the green-zero rule

A test file opening with `#![cfg(feature = "x")]` compiles to an **empty
binary** when `x` is off; cargo prints `ok. 0 passed` and **exits 0** —
byte-for-byte a real pass. The `#![cfg]` protects the **count**;
`required-features` protects the **reader** — both are needed, and only the
second is visible to whoever reads the output. A *default-on* gated target
still runs on a plain `cargo test`; a *default-off* one reports a green zero
every time anyone names it — read the severity split, never the pooled total.
`not(debug_assertions)` is a separate overlapping dimension: silent under
plain `cargo test`, and it **survives the fix** — adding a
`required-features` row moves the target into "w/ req-f", which reads as
protected and does not make it compile.

Do not answer "how much of this is affected" by reading manifests. Run:

```bash
scripts/cfg_gated_target_audit.py            # all contract repos (derived)
scripts/cfg_gated_target_audit.py ../riir-ai # or one, by path
```

`scripts/suite_membership_audit.py` answers the next axis down: which
`[[test]]` targets no script/workflow names — run it when landing a new gate;
if nothing names it, add a suite row or record why not.

- A **report, not a gate** (exit 0): `cfg` on `target_os`/`miri` and an
  `any(...)` of features genuinely cannot be expressed as
  `required-features` — reported as their own classes.
- **Arming a target can RED a binary-counting floor**: an empty gated binary
  prints `test result: ok. 0 passed` and COUNTS as one — adding the row
  removes a line. Repair with a **passed-test floor**, not a re-pin.
- **Run the armed gates with `--release`** — a latency gate in a debug build
  measures an unoptimised binary.
- Verdict half: `scripts/cfg_gated_floor_gate.py` (katgpt-rs-scoped pins in
  `scripts/cfg_gated_floors.txt`; `max_load_bearing = 0` earns its keep; some
  pins are FLOORS — a ceiling cannot fail once the instrument goes blind;
  `scripts/all_ignored_load_bearing.txt` pins the ALL-IGNORED set by
  MEMBERSHIP — a set is gateable where its cardinality is not).

## A `required-features` row can EXIST and be WRONG — `scripts/required_features_build_audit.py`

Every audit above treats a target as protected once it **has** a
`required-features` row. A row that exists and is wrong is strictly worse
than a missing one: `cargo test --workspace` silently **skips** the target,
`--all-features` **builds** it (the union supplies whatever the row forgot —
the one configuration anybody runs it in passes), and every audit counts it
as protected. The row is wrong relative to what the file *imports*, and
imports resolve through cfg-gated re-exports that defeat grep — ask the
compiler, once per target:

```bash
scripts/required_features_build_audit.py --list            # rows only, no builds
scripts/required_features_build_audit.py ../riir-train     # one repo
scripts/required_features_build_audit.py . --grep pruners  # one slice
scripts/required_features_build_audit.py ../riir-train --batch  # 1 run per set
```

- A **report, not a gate** (exit 0; ~28 s/row — filter with `--package` /
  `--kind` / `--grep` / `--limit`; `--target-dir` when a sibling is building).
- `--batch` = **one cargo run per (package, EXACT feature set)**, never a
  superset — a superset build may supply the very import the row forgot.
- **Neither an error nor an artifact = UNSEEN, never BUILDS** — silence is
  not evidence; UNSEEN is never folded into the pass column.
- The free static verdict is correct AND insufficient: `dep/feat` / `dep?/feat`
  rows are valid cargo (a DEPENDENCY's feature); only the compiler
  distinguishes "names a feature that exists" from "names the feature that
  gates the module".
- **Read the USE SITES, not the error:** widen the ROW when the body needs
  the feature unconditionally; narrow the cfg when the use site is already
  gated.
- Per-push half: `.github/workflows/required_features_touched.yml` checks the
  rows this push could have broken (a changed file that IS a row's target
  source selects the row; a changed `Cargo.toml` selects rows whose
  `(kind, name, required-features)` tuple differs base-vs-head; `--max-rows`
  REFUSES rather than truncates).

## A reported "p99" is often the MAX — `scripts/percentile_index_audit.py`

`sorted[(n as f64 * 0.99) as usize]` and `sorted[n * 99 / 100]` both land on
`n - 1` — the **maximum** — for every `n <= 1/(1-p)`: n ≤ 100 at p99, n ≤ 20
at p95, n ≤ 1000 at p999. Below that boundary the site reports one
observation under a percentile's name; a `.min(len - 1)` clamp prevents a
panic, not a wrong statistic. The quantity to print is **tail support** =
`n - idx` (samples at or above the reported rank): 1 at n=100, 2 at n=200,
10 at n=1000 — anything under 10 is weak.

```bash
scripts/percentile_index_audit.py             # all contract repos (derived)
scripts/percentile_index_audit.py ../riir-ai  # or one, by path
```

A **report, not a gate** (exit 0) — half the sites take their sample count
from a runtime length no static pass can reach. **UNRESOLVED is not
"clean"** — it is "needs a per-site read". Vocabulary is data (`VOCAB`),
population derived. Verdict half: `scripts/percentile_floor_gate.py` (pins in
`scripts/percentile_floors.txt`; `min_sites_scanned` is a FLOOR — a tokenizer
regression takes the population to ~0 and every ceiling passes).

## Before committing in a shared worktree — `scripts/staged_set_audit.py`

Several agent sessions write into one worktree routinely, and `git add -A`
from a repo root is indistinguishable, to git, from intent. Stage **named
files** (`git -C <repo> add <paths>`), never `-A` — and before a multi-file
commit, run:

```bash
scripts/staged_set_audit.py            # any repo: pass its path as $1
```

A **report, not a gate** (exit 0) — a refusing pre-commit hook was decided
against: every cheap signal has a legitimate-use false positive; a report
that is read beats a gate that is bypassed. Four signals: **mtime clusters**
(two clusters = two editing episodes; the older is probably not yours) ·
**also-dirty** (a staged path with unstaged changes = a concurrent editor) ·
**stale-vs-HEAD** (a file LACKING substantive lines the newest commit on its
path added — committing it reverts them) · **rustfmt round-trip** (`--fmt`:
identical to `rustfmt(HEAD)` provably carries zero content — the only
signal that yields a proof).

When you must commit into a file a sibling is editing, commit **your blob**:
build HEAD's version + your edit, `git hash-object -w`, then `git
update-index --cacheinfo`. Their hunks stay uncommitted; the worktree stays
coherent for them.

**Shared target dir:** a count-pinned or feature-switching gate run
concurrently with another cargo process in the same `target/` reports a
failing test that passes when run alone. Read the failure's **shape**:
`error: test failed` with **no `failures:` block and no `test … FAILED`
line** means the harness process *died* — nothing asserted anything.
Diagnose by running the compiled binary directly from
`target/<profile>/deps/` (no build lock needed; filter out the `#![cfg]`-gated
copies `--list` reports as 0 tests). A gate whose verdict the box can
invalidate should **refuse**, not warn — detect concurrent cargo by working
directory, not command line; a lock-based check cannot work (cargo releases
`target/<profile>/.cargo-lock` *before* running the test binaries).

## Lint healing — `cargo heal` before manual fixes (adopted 2026-08-24)

Mechanical clippy findings (format-arg inlining, `match_bool`, `map_or`,
capacity, `needless_return`, …) are fixed by the riir-clippy healer FIRST,
manual second:

```bash
cargo heal --fix <paths>                                  # dry run (review only)
cargo heal --fix --write --verify <paths>                 # compile-gated apply
cargo heal --fix --write --verify --verify-args "--features <set>" <paths>  # gated code
```

- Global binary `cargo heal` = `~/.cargo/bin/cargo-heal` → the sibling
  `riir-clippy/target/release/cargo-heal` (built `--features
  fix_verify,clippy_verify`; rebuild after healer source changes). Missing
  sibling → fall back to manual fixes + `cargo clippy --fix`.
- `--verify` compiles baseline → applies → re-checks → auto-REVERTS breaking
  edits. Feature-gated code needs `--verify-args "--features <set>"` (a
  default-features check compiles gated files empty — a green check proves
  nothing about them).
- The healer is deliberately SILENT on documented divergence classes
  (comment-guarded matches, array-literal defaults, named-arg renames,
  nested macro args) — those stay manual; see the `cargo-heal` skill
  (`~/.agents/skills/cargo-heal/`) for the full table + discipline.
- `cargo clippy --fix` remains fine for one-off trivial fixes; the healer
  wins on batches (span-preserving, comment guards, compile gate,
  self-evolve memory) and was validated across the full katgpt-rs sweep
  (every surface, count-identical test validation, 2026-08-19).
- Observed misses / wrong suggestions → note in the session record; they feed
  riir-clippy's post-mining queue (usage-artifact improvement intake).

## Feature Flag Discipline

Every new primitive ships behind a feature flag (opt-in). Promotion to
default-on requires the GOAT gate to pass:

1. Implement behind `feature_name = []` (opt-in).
2. Write a benchmark proving the gain (latency, quality, or security).
3. Run the GOAT gate (G1 correctness, G2 perf, G3 no-regression, G4 alloc-free
   or equivalent).
4. If all gates pass AND the gain is **modelless** → promote to `default`.
5. If the gain requires riir-train (training) → keep opt-in, note the
   dependency, do NOT promote to default.

**Promotion requires modelless gain.** A perf gain on a biased/incorrect answer
is NOT a modelless gain — it's a speedup of a wrong result. The quality gate
(G1 or equivalent) must pass modellessly for the GOAT to hold.

**Lossy-surface promotion rule (Issue 750 T3):** a **lossy** surface
(quantization, compression, any bit-changing transform) gates on
**deployed-path behavior — per-family, conditional retention**, not on
bit-identity or aggregate perplexity alone: aggregate perplexity can be flat
while family-conditional behavior flips. (Full rule + confirmations: HISTORY.md.)

**UQ-bearing primitive GOAT gate extension (the "Report the Floor" rule,
Research 322 / Plan 340):** any primitive claiming a probability
distribution, predictive interval, quantile, coverage guarantee, confidence
score, or calibrated uncertainty MUST benchmark against the
**conformal-naive floor** — `ConformalIntervalCalibrator<SeasonalNaiveForecaster>`
(Plan 340 with `m=1`, plain split conformal) — on CRPS / coverage / Winkler
score. Cannot beat the floor ⇒ the GOAT gate FAILS. Grandfathered UQ
primitives include the floor at their next re-gate. (History: HISTORY.md.)

## Substrate-First Gate (MANDATORY before implementing)

Before implementing ANY new System impl, trait, perception/cognition/emotion
pipeline, state management, spatial query, or vocabulary type, run the
`.agents/skills/substrate-first/SKILL.md` skill: (1) **vocabulary
translation** — grep 3+ name variants (concepts ship under operator names
like `GenericSpatialBelief`; a single-vocabulary grep returns ZERO hits even
when substrate fully exists); (2) **codebase grep** across `*.rs`, not just
`.plans`/`.docs`/`.issues`; (3) **architectural rule check** — domain
classification, two-brain model, sync boundary, bridge pattern; (4)
**consume vs build** — if substrate exists, consume it; if not, file an
issue in the right repo FIRST. Prevents the drift pattern of a parallel
system re-implementing shipped substrate under a different name (ThreatField
Issue 047; orchard/motivation Issues 490/493).

Research workflow (paper classification, 7-repo routing, fusion-first
distillation, novelty + GOAT gates, modelless-unblock protocol §3.5):
`.agents/skills/research/SKILL.md`.

> **Repo count:** the **product/distillation set is 7** — `katgpt-rs` (public) +
> `riir-ai`, `riir-chain`, `riir-neuron-db`, `riir-train`, `riir-game-sdk`,
> `riir-dapps` (private). That is NOT the repo total: the
> workspace is **16 repos**, all of which carry a root `BOUNDARY.md`
> (add `riir-mmorpg-examples`, `riir-clippy`, `riir-viewbridge`,
> `riir-auth`, `katgpt-web`, `riir-dao`, `riir-deployer`,
> `seal-game-editor`, `seal-remake`).
>
> Read a count in prose as a claim, not a fact — and read a count that
> MATCHES as a claim too: a count is not a checksum over a set. Drift
> history: HISTORY.md.

## Numbering Discipline

Issue, plan, doc, benchmark, and research numbers are **monotonic and never
reused** — even after a file is removed per the noise-reduction rule. Before
creating a new `.issues/` file, read `.issues/.highwater`, use `value + 1` as
the number, and write the new value back. This prevents the number-recycling
collision documented in `.issues/121`. The same rule applies to `.plans/`,
`.docs/`, `.benchmarks/`, and `.research/` — never recycle a number that git
history shows was already allocated.

## Branch

`develop` is the working branch. Don't create feature branches; commit
directly on `develop` per the global rule.

## Models
- riir-train/data/gemma-2-2b-it-f16.gguf
- riir-train/data/MiniCPM5-1B-F16.gguf
