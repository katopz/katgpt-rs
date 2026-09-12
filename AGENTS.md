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
# Toolchain: pinned by rust-toolchain.toml (1.98.1, issue 739) — cargo resolves
# it automatically; full_gate.yml is the deliberate RUSTUP_TOOLCHAIN=stable
# rot-gate exception.

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
| host triple vs **`wasm32`** | a `--target` you never pass is a platform you never compile. Worse than the macOS axis because it is gated **twice**: the hot kernels are `all(target_arch = "wasm32", target_feature = "simd128")` and the triple defaults to simd128 **OFF**, so even a wasm32 lane without `RUSTFLAGS='-C target-feature=+simd128'` compiles the SIMD half to nothing. Measured (Issue 737): the simd128-**off** arm was clean and the **on** arm had 14 findings, 11 of them `unsafe_op_in_unsafe_fn` on edition 2024. `full_gate.sh` layer 2b runs both arms |
| **compile vs EXECUTE** | every axis above is about *compilation*. The scoped core (katgpt-rs + katgpt-core `--lib` at default features, count floors) was EXECUTED weekly (`test.yml` + `scripts/test_gate.sh` — SCHEDULE SUSPENDED 2026-09-09, Actions spending limit; run `scripts/test_gate.sh` locally); the other 477 integration-test and 176 bench targets are executed by nothing automatic, and `--all-features` is not a supported TEST configuration (fixture RNG streams and GOAT calibrations are per-feature). An uninvoked assertion is *unknown*, not passing |

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

**And `wasm32` is a second platform axis, not a variation on the first.**
Nothing in this repo compiled it until 2026-09-07 — the only script naming the
triple was `scripts/build-moka-wasm.sh`, which is a deploy build a human runs,
not a gate. `full_gate.sh` layer 2b closes it: derived `-p` list (a new
wasm32-bearing crate joins by existing) **including the root package**, both
simd128 arms, the two wasm32 GOAT targets by name, and the residue pinned by
**membership** so the gate reds when that set changes rather than silently
shrinking. A missing `wasm32-unknown-unknown` target is a PARTIAL gate that
refuses, exactly as an off-macOS run is.

Selecting the **root package** is what makes that lane wide, and it was not
why it was added: clippy lints every **workspace path dependency** it pulls
in (registry crates are `--cap-lints`'d, workspace ones are not), so
`-p katgpt-rs --lib` puts the whole internal graph under `-D warnings` on
wasm32. That is how an orphaned doc block on
`katgpt-attn-match::select_highest_attn_keys` — a crate with no wasm32 code
of its own — surfaced. `--all-targets` is NOT the way to widen further: it
dies on dev-deps (`statrs`, `proptest`) that do not resolve for wasm32, so
extra coverage goes in as **named targets**.

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

**The profile axis is feature-shaped too (Layer 6b, Issue 758).** Layer 6
runs `--all-features` — which SUPPLIES `alloc_tracking` — so the (release ×
default-features) cell was asserted by nothing until slice_tca's module-level
`use crate::alloc` fell through it (E0432 under `cargo test --release -p
katgpt-core --lib`, Layer 6 green the whole time). `full_gate.sh` Layer 6b
closes it: the test_gate population (katgpt-rs, katgpt-core, katgpt-dec at
its pca_global row) at `cargo check --tests --release`, deliberately not
`--workspace` (that inherits the platform axis Layer 2 refuses on — the metal
examples). The matrix: (dev, default) test_gate · (dev, all) Layer 3 ·
(release, all) Layer 6 · (release, default) Layer 6b. Alloc-gated tests carry
`#[cfg(any(debug_assertions, feature = "alloc_tracking"))]` — the full
Issue-741 predicate, so they RUN under `--release --features alloc_tracking`
(the configuration alloc gates are meant to be read in) and compile away at
release-default instead of breaking the harness.

Trigger health: CI is MAIN-ONLY + dispatch-only since 2026-09-09 (owner call:
Actions spending limit + no CI on `develop` pushes) — every `push` trigger is
`branches: [main]` and every `schedule:` block is suspended in-file (commented,
riir-train 507 precedent). `scripts/ci_gate_coverage.py` reports which declared
triggers can actually fire, per workflow, per repo. Layer 2b also has its own
push lane: `.github/workflows/wasm32_gate.yml` runs `full_gate.sh
--wasm32-only` on ubuntu-latest (Issue 737 T4), now MAIN-ONLY too — the lane is
host-independent and `--lib`-only; dispatch it manually after a run of develop
work, since no automatic lane covers develop pushes anymore.

## Docs gate + drift sweeps

`scripts/docs_gate.sh` runs the manifest/doc/skill drift assertions and
**prints its own timing** — a hand-typed duration drifts exactly like a
hand-typed count, and it was also the wrong quantity. Measured three times:
**12.65s · 12.52s · 12.69s CPU** on runs whose WALL clocks were **128.3s ·
299.1s · 15.0s** — a **20x** wall spread against **1.4%** of CPU spread. That
is the whole argument for the quantity: **cite CPU, read wall as a range.**
⚠ Those three are a **14-check** measurement, and the CHECKS set is part of
the claim exactly as the profile is. Two checks later: **14.42s** CPU at 15
checks, **13.96s** at 16, and **13.37s** at 17 (Issue 756's
`markdown_fence_gate.py`, a 1517-file walk). Read that honestly — the added
checks did NOT show up as a clean increase; those three RUN DOWNWARD as the
CHECKS set grows, which is the opposite of what any per-check cost model
predicts, and the 3.3% spread across the first two is **wider than the
1.4%** the three 14-check runs suggested. So CPU is the load-invariant figure
and still the right one to cite, but it is tight-ish, not exact, and a
difference this size is not evidence a check got slower. Only compare CPU
within a fixed CHECKS set, and only as a range.
⛔ A discredited fourth figure is why this paragraph is worded so insistently:
an earlier version called 11.7s wall a *quiet-box baseline*, and it was taken
at load 5-7 — the 15.0s run (2026-09-11) is the first one actually measured on
a quiet box, and it is SLOWER than the number that was being quoted as the
floor.

The wall inflation lands on the checks that walk the tree —
`cargo_comment_audit` 54.7s, `bench_doc_audit` 50.4s, `cfg_gated_floor_gate`
31.2s in the 128.3s run, everything else under 4s — and **none of those
invokes cargo**, so it is not the cargo build lock (the first version of this
paragraph said it was, on no evidence; the per-check line refuted it). Which
check dominates is not stable either: the 299.1s run put
`percentile_floor_gate` at 61.3s and `bench_doc_audit` at 73.1s, and on the
quiet 15.0s run no check crossed 4s at all. Beyond "a
busy box starves the tree walks" the mechanism is **unmeasured**. Read the
per-check `⏱` line to see which check is BLOCKING, never to conclude a check
got slower.

The CPU figure **asserts itself non-inert**, because its failure mode is a
well-formed number rather than an error: `times` reports `0m0.000s` children
CPU from any forked context — a pipeline and a command substitution both fork,
and the fork has no children of its own, so even `times | sed` purely to
indent destroys it (the first two versions printed a confident zero next to a
308s run). It is REDIRECTED to a file, never captured, and the gate prints
`⛔ … NOT a measurement` instead of the number if the total reads ~0 over a
multi-second run. Both arms verified against the block extracted from the
tracked file: redirect → 0.45s from a child that burned 0.43s; pipe → 0.00s
and the ⛔ fires.

`.github/workflows/docs_gate.yml` runs it per-push on **`main` only** —
develop pushes do not fire it, so run `./scripts/docs_gate.sh` locally for
develop work. One line per check:

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
| `required_features_static_gate.py` | a required-features row naming a feature its package cannot enable (riir-train Issue 513) |
| `cfg_row_implication_gate.py` | a required-features row that BUILDS and compiles its target to NOTHING (riir-train Issue 513) |
| `population_sync_gate.py` | the seven independent contract-repo predicates must agree |
| `trap_sentinel_gate.py` | a shell gate whose abort would report exit 0 — this repo's own two, by MEMBERSHIP (Issue 734) |
| `issue_citation_gate.py` | a cross-repo `Issue N` citation naming no repo — it rebinds to the WRONG document once that number is allocated locally (Issue 749). In CI the cross-repo axis is DEFERRED to the workstation run — the `DOCS_GATE_CI` marker's instrument-alive verdict, because the sibling workspace is absent in a single checkout |
| `markdown_fence_gate.py` | a fenced code block never closed — everything after it renders as code, and a fence scanner mis-phases on it (Issue 756) |
| `docs_gate_checks_sync.py` | this CHECKS array vs the AGENTS.md table documenting it — membership both ways + quantity words (Issue 750) |

The `CHECKS` count is deliberately not written here — it drifted once, which
is exactly the drift this gate exists to catch.

Workstation-only cross-repo sweep family — `docs_drift_sweep.py`,
`numbering_drift_sweep.py`, `required_features_drift_sweep.py`,
`percentile_drift_sweep.py`, `cfg_gated_drift_sweep.py`,
`cfg_row_implication_drift_sweep.py`, `trap_sentinel_drift_sweep.py`,
`citation_drift_sweep.py`, `restatement_drift_sweep.py` (every contract repo,
on demand), `markdown_fence_drift_sweep.py` (every contract repo, on demand —
the Issue 756 unterminated-fence verdict workspace-wide; its first run caught
the then-new `seal-online-remaster`), (feat: markdown_fence_drift_sweep.py — the Issue 756 unterminated-fence verdict over every contract repo (workstation, two-axis pins: min_md_files walk floor + max_unterminated=0 wall); its FIRST workspace run caught seal-online-remaster .plans/005:600 (14 swallowed lines — the repo joined the contract set at Issue 760 after the landing measurement; repaired seal-online-remaster 99064c5, floored at 50); AGENTS.md sweep-family row)
`sibling_docs_drift.yml` (reusable workflow, one caller), and
`ci_gate_coverage.py` (report, always exit 0: which repos gate their full
compile+lint surface in CI, and whether anything automatically starts it).
NOT in docs_gate's CHECKS — CI's single checkout would derive an empty
population and print a confident green over zero repos. Population derived
(BOUNDARY.md + `.git`); expectations committed in `scripts/*_floors.txt`.

`citation_drift_sweep.py` is the one that **prints its own error rates next to
its finding count** — plural, because there are two populations and a SAMPLE
rate does not transfer to rows it never sampled. Its CROSS rows split into the
pre-752 corpus, carrying **7/43 = 16%** false positives from a stratified
manual read (Issue 751 T1), and the **45** rows recovered by owner-consistency,
carrying **1/45** from a full census (Issue 752, re-rated by Issue 754).
Neither number is quotable without the other, nor without the ~3k-citation walk
and 19-repo population that produced them — a magnitude, deliberately, because
five-plus concurrent sessions edit these documents and an exact figure in
prose is drift waiting to happen (the dated snapshot lives in the sweep's own
docstring, where it is a measurement record rather than a claim); the
**IN-LOCAL-RANGE** bucket is UNDECIDED and never folded into either
neighbour. It also asserts its
katgpt-rs row against `issue_citation_gate.py`'s own parsed run rather than
trusting the two to agree.

Those 45 are the reason to distrust a lone error rate: every other FP class
this family documents **inflates** a count, and this one **deflated** it by
~15%. Qualification asked *"is a repo named?"* and never *"does that repo own
the number?"*, so `riir-chain Plan 211` (riir-chain's `.plans` top out at 058)
and `katgpt-rs Issue 513` (513 is riir-train's) both read as clean — the
attribution following the CODE while the number followed the DOCUMENT. Reading
a measured error rate as if it bounded the error in ONE direction is the
mistake; it bounds only the direction somebody thought to sample.

⛔ And that census's own `0/45` did not survive either (Issue 754). Its third
"outright wrong address", `riir-mmorpg-examples Issue 059`, was **correct**:
that repo records 059 in its own HISTORY.md heading, with the file removed the
day it was filed and never committed, so neither the worktree walk nor `git
log` could see it. Reading all 45 rows by hand could not have caught that,
because every read asked the same blind `allocated()` the same question. **A
census is exhaustive over ROWS, not over the ORACLE it checks them against** —
so never quote an error rate without naming the instrument the sample was
adjudicated against.

Each sweep carries **two floors, not one**: a ceiling is green over whatever
the instrument can SEE, so the finding count needs the *population* that
produced it, and the population floor is 0 in every repo that has none of the
thing — so it needs the *walk* size underneath it too (`min_rs_files`,
`min_manifests`, `min_scripts`). Where a sweep re-states a quantity its
per-push gate owns, it **asserts** the two agree rather than trusting them
(`trap_sentinel_drift_sweep.py` vs `trap_sentinel_gate.POPULATION_FLOOR`) —
`docs_gate_paths_sync.py`, one axis over.

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

**Two traps in the profile dimension (Issue 741).** First: a file may carry
**more than one** whole-file `#![cfg]`, and rustc **ANDs** them — reading only
the first under-reports the profile term AND the feature set (56 of 1634 gated
targets workspace-wide carry 2+, up to 5 in one file). Second, and the one to
internalise: **gating a MEASUREMENT on `debug_assertions` makes it impossible
in the configuration that ships.** Every alloc gate here was unrunnable under
`--release` — the profile this document mandates for gates — because
`katgpt_core::alloc` itself was `cfg(debug_assertions)`, so the whole target
compiled to an empty binary and printed `ok. 0 passed`, exit 0. A profile is
not a knob; a feature is. Ask of any `debug_assertions` gate whether the thing
behind it is a **capability** (→ give it a feature, `any(debug_assertions,
feature = "x")`, and gate the machinery on `x` too — including any in-body
liveness sentinel, or the release binary runs the gate and asserts NOTHING) or
genuinely a **profile property** (an assertion about `debug_assert!`). It was a
capability the whole time, and "debug-only by design" had been written into the
guarding pin's own header as if it were a constraint. Read the split the
auditor prints — `unfixable` (bare term, no flag compiles it in release; the
pin worth having) vs `escapable` (`any(…, feature = …)`, already runnable in
release) — never the pooled DEBUG-only count, which reports the repair as if it
changed nothing.

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
- Push half (MAIN-ONLY since 2026-09-09): `.github/workflows/required_features_touched.yml` checks the

  rows a main push could have broken (a changed file that IS a row's target

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

## A Lean theorem can RESTATE its own definition — `scripts/restatement_theorem_audit.py`

`theorem sidecarHeaderSize_eq_sum : sidecarHeaderSize = magicSize + versionSize
+ …` where the RHS **is** the `def` body. `decide` closes it whatever the
constants hold, so it is green on every transcription typo it was written to
catch — while `lake build`, `#print axioms` and the proof gate's audited-surface
count all report it as a theorem that proves something. Four shipped in
riir-neuron-db for months (Issue 617, removed `24957a2`).

```bash
scripts/restatement_theorem_audit.py               # all repos with .proofs (derived)
scripts/restatement_theorem_audit.py ../riir-chain # or one, by path
scripts/restatement_theorem_audit.py -v            # every row, not just findings
```

- A **report, not a gate** (exit 0). The criterion is symbolic equality over
  **leaf** constants: unfold every composite nullary `def`, keep numeral-bodied
  leaves as symbols, compare as polynomials. Unfolding all the way to numerals
  instead would compare `464` with `464` and condemn every sound literal pin —
  **the leaf boundary IS the classifier.**
- **CROSS-DEF is not a finding.** `commitmentOffset = RAW_PREFIX_LEN` is
  symbolically equal too, but it pins two *independently maintained*
  definitions against each other and a perturbation arm proves it reds. The
  removed four had an RHS that existed only inside the theorem. Pooling the two
  would have condemned a load-bearing theorem.
- **UNRESOLVED is not clean** (function application, Mathlib, ℚ/ℝ ops), and
  `HYPOTHETICAL` — a theorem with binders — is split out rather than pooled,
  because it is 199 of 255 and pooling hides how few statements the arithmetic
  pass ever sees.
- Validated against a tree whose answer was known independently: riir-neuron-db
  at `24957a2^` reports **exactly** the four Issue-617 theorems, 0 at HEAD.
  That run is also what exposed the classifier's own defect — a repo-wide def
  table unfolded `Shard.zoneHashOffset` through ExperienceGraph's same-named
  chain. Scoped per module + transitive imports now.
- Standing (2026-09-12): **0 RESTATEMENT-INLINE** over 4 repos / 68 `.lean` /
  255 theorems; 19 UNRESOLVED + 6 conjuncts read one by one, all value pins.
- Verdict half: `scripts/restatement_drift_sweep.py` (workstation, every repo
  with `.proofs`, pinned in `scripts/restatement_drift_floors.txt`). **Two
  floors** — `min_lean_files` catches a WALK regression, `min_theorems` a PARSE
  one, and only the second moves when a tokenizer breaks on an unchanged tree.
  `--prove-fires` plants a restatement into a COPY of each repo and requires
  the count to move, so the ceiling is never a pin nobody has watched fail.

## A gate that ABORTS reports exit 0 — `scripts/trap_exit_launder_audit.py`

Every script above is a shell gate with `set -euo pipefail` and a cleanup
trap. On **macOS `/bin/bash` 3.2.57 — and only there** (Issue 735): when bash
aborts on an **unbound expansion** or an **`eval` syntax error**, it enters
the EXIT trap with `$?` **already 0** — so an EXIT trap whose last command
succeeds (`rm -f "$TMP"` always does) makes the abort exit **0**. Everything
after the abort silently did not run, and the caller reads a pass.

⛔ **This bites every macOS run — workstation AND the macOS CI lane; it is
`ubuntu-latest` that is immune.** This paragraph has now been wrong in BOTH
directions, which is the lesson: it first said "and CI reads a pass" (false —
over-claimed), was corrected to "a WORKSTATION defect, **not** a CI one"
(also false — under-claimed, and in the direction that hides a live
exposure), and is now measured on both sides. Interpreters, one at a time
(`scripts/trap_launder_premise_matrix.py`, 11 of them): bash **4.4.23 /
5.0.18 / 5.2.37 / 5.3.15**, dash and busybox ash **all preserve** the status;
fixed no later than 4.4. Every gate-running workflow in the workspace is
`runs-on: ubuntu-latest` → bash 5 → an aborting gate exits non-zero and the
job reds. **The exception is the macOS lane, and it was measured, not
reasoned about** (Issue 735 T3, answered early by the 737 layer-2b push run
`34137014037`): GitHub's `macos-26-arm64` ships bash **3.2.57 ONLY** — PATH =
`/bin` = `env`, no Homebrew bash in PATH — and reproduces all five errexit
LAUNDERS cells. So on `full_gate.yml`, this repo's only macos-latest runner of
a sentinelled script, **the sentinel is load-bearing in CI**, not merely on
workstations; its preamble step re-measures every run, so image drift is
observed rather than silent. T4 resolved **do not pin — measure**: a `shell:`
pin cannot govern a script's own `#!/usr/bin/env bash` shebang anyway. There
is no `bash:3.2` docker tag, so the premise's own interpreter is measurable
**only** on macOS — a workstation or that runner.

**Keep the sentinel regardless.** It costs nothing on 5.x, is load-bearing on
3.2, and "did the script reach its own completion point?" catches every other
premature death — a SIGTERM, a `set -e` trip in an unguarded spot, a future
editing slip — on **every** shell. The rescoping changes the class's
*severity*, not the value of the repair.

**`errexit` is the precondition, NOT `nounset`** — the first version of this
section had that backwards, because the premise harness hard-coded `set -euo
pipefail` and never varied the axis it was claiming about (Issue 734 T10):

| shell options | unbound expansion | `eval` syntax error |
|---|---|---|
| `set -u` (no `-e`) | aborts, **1** — *not* laundered | does **not abort at all** |
| `set -e` (no `-u`) | no abort (expands empty) | 2 bare, **0** trapped ✗ |
| `set -eu` / `set -euo pipefail` | 1 bare, **0** trapped ✗ | 2 bare, **0** trapped ✗ |
| (`set -e` command failure → 1 both ✓; command not found → 127 both ✓) | | |

So `set -u` **without** `set -e` cannot launder anything today — that is
**PRECAUTIONARY**, not EXPOSED, and pooling the two over-claimed on 15 of 41
rows. Two corollaries: the population predicate is the **union** (`set -e`
OR `set -u`) because errexit-without-nounset launders via the `eval` trigger
(measured: 0 such scripts, so that blind spot was empty — but it is a
measurement now), and the measurement **mode** is part of the claim — the
nounset fatal error exits **127** from `bash -c` and **1** from a script
FILE, so the harness writes a temp script.

`trap 'rc=$?; cleanup; exit $rc' EXIT` does **not** repair it — the rc it
saves is itself 0. Only a **completion sentinel** does: a flag set on the
script's own last line, checked by the handler, forcing exit 1 when the run
is INCOMPLETE *and* claiming success. An ordinary layer failure still exits 1
with its own message, untouched. `scripts/full_gate.sh` and
`scripts/proof_negative_test.sh` carry it (Issue 734).

```bash
scripts/trap_exit_launder_audit.py            # population + verdict, all repos
scripts/trap_exit_launder_audit.py ../riir-ai # or one, by path
scripts/trap_sentinel_drift_sweep.py          # the verdict, every repo, pinned
scripts/trap_launder_premise_matrix.py        # the PREMISE, 11 interpreters
```

Three halves, and they answer different questions — do not read one for
another. `trap_exit_launder_audit.py` derives the **population** and
classifies it (report, exit 0). `trap_sentinel_gate.py` is the **verdict** for
this repo (in the docs gate) and `trap_sentinel_drift_sweep.py` the verdict
for all 17 (workstation, pinned in `scripts/trap_sentinel_drift_floors.txt`).
`trap_launder_premise_matrix.py` measures the **premise** — one interpreter at
a time, via docker, script files not `bash -c`. It always measures the local
box first and prints **UNSEEN, never a zero**, when docker is absent: a
premise instrument that silently skips its arms reports "nothing launders" and
retires the whole class.

A **report, not a gate** (exit 0) — EXPOSED is latent, and a report that
exits 1 on dozens of latent rows is a report nobody runs. Population derived
(BOUNDARY.md + `.git`) and restricted to **tracked** `*.sh`: walking the
filesystem instead reported 25 findings in a **gitignored** vendored drop no
repo owns. Verdicts: **LIVE-FORWARD** (a double-quoted `trap "… $VAR …"`
naming a later-assigned variable — a *provable* abort, every run, and how
this was found), **EXPOSED**, **SENTINELLED**, **UNPARSED**, plus the
orthogonal **REPLACED** (2+ EXIT traps — `trap` replaces, it does not
accumulate, so earlier cleanup is silently dropped).

Each finding also carries its **exposure window** — `[last trap
registration, EOF)` — and the count of abort **triggers** (`$VAR` / `eval`,
with the body of any function the window *calls* folded in). Nothing before
the handler exists can be laundered by it, so a window with **zero** triggers
provably cannot launder whatever its `set` line says. A triage aid, not a
verdict (same standing as tail support in the percentile audit): it ORDERS
the rows, and a 2-line/0-trigger row and an 863-line/253-trigger row are
otherwise one row each. It is how the last EXPOSED row in the workspace —
`riir-ai/scripts/e2e_internet.sh`, trap on line 41 of 43 — is known to be
inert rather than merely inconvenient to fix.

**UNPARSED is the instrument admitting it cannot read** — the trap names a
function whose body never closed under brace counting, so *both* verdicts
would be guesses. It is not the safe direction and must not be pooled: a
runaway body swallows the rest of the file, and with it somebody else's
`exit 1` and some late literal flag, and reads as a **false SENTINELLED**,
which HIDES exposure. Found because riir-chain's
`block_pipeline_reachability_gate.sh` embeds a multi-line **single-quoted**
`awk` program containing `mod[[:space:]]*tests[[:space:]]*\{` — one
unmatched brace in DATA — and read EXPOSED while carrying a correct
sentinel. `scan_braces` is quote- and heredoc-aware now; UNPARSED covers
whatever it still cannot parse, and the verdict gate reds on it.

**shellcheck does not find this** (measured, Issue 734 T7): pointed at the
script carrying the live defect it reports one `SC2001` at default severity,
and with `-o all` its only remark on the fatal line is `SC2250` — brace
style. SC2154 does not fire, because the variables *are* assigned, just too
late.

Canonical failure: seal-remake's `ci_feature_guard.sh` — the script its
`rust.yml` runs — could not fail past layer 13 for months, because its
layer-13 trap named two variables assigned ~20 and ~45 lines later. It stayed
hidden because a ratchet ceiling had been red for three commits and stopped
every run *before* the bad line (seal-remake `26a18191`).

Verdict half: `scripts/trap_sentinel_gate.py` (in the docs gate). It pins this
repo's two by **membership**, floors the population (a classifier that goes
blind must RED, not report a green zero), and reds on the commit that adds a
new unsentinelled gate script. Its canary is two-sided and it earned that:
the first classifier called `full_gate.sh` SENTINELLED with its sentinel
assignment DELETED, because the script also has an unrelated
`if [ "$KEEP_LOG" -eq 1 ]` and the rule only asked for "tests the flag" and
"exits non-zero" *independently*. The flag must gate the failure branch —
tie them by block structure or the pin certifies nothing.

## A lane compiles what it NAMES — `scripts/wasm32_surface_audit.py`

Every axis in the wasm32 family (Issue 737) is about *how* a lane compiles
what it names. The seventh is one level up: **is what it names the whole
surface?** A row cannot notice a package it does not select, and seal-remake
had a positive `#[cfg(target_arch = "wasm32")]` block that no row built and
that had been **uncompilable since it was written** — it called a
`cfg(not(wasm32))` function (`.issues/010` T2, `.issues/738`).

```bash
scripts/wasm32_surface_audit.py            # all contract repos (derived)
scripts/wasm32_surface_audit.py ../riir-ai # or one, by path
```

- A **report, not a gate** (exit 0). Three buckets: **NAMED** (a row selects
  it by `-p` or a literal `--manifest-path`), **UNRESOLVED** (a `--workspace`
  or *derived* row exists — whether it reaches this package is the
  separate-workspace axis, undecidable statically), **UNCOVERED** (no row
  could reach it). **UNRESOLVED is not clean** and is never folded into
  either neighbour.
- The predicate is the **positive** cfg: `not(target_arch = "wasm32")` is an
  ordinary native-only guard and counting it inflates everything (riir-ai
  `.issues/892` T4). **Comment lines are excluded** — prose explaining a cfg
  is not a cfg, and the comment recording why a file has *no* wasm32 arm
  otherwise makes that file read as browser code.
- First measurement (2026-09-07): **9 NAMED · 15 UNRESOLVED · 0 UNCOVERED**
  over 196 files / 24 positive-cfg packages / 17 repos. Resolved 2026-09-08
  (738 T1): the two-shape resolver upgrades a derived-row package only on
  row-bearing static evidence; the vendored `wgpu-hal` fork left the walk
  (738 T3); the per-package reads surfaced one real lane gap
  (riir-mmorpg-examples' standalone `warm-tier-do`, lane landed same day)
  and one uncompilable surface (riir-ai's `riir-examples` browser examples,
  filed there as `.issues/894`). 894 resolved same day in riir-ai (uuid `js`
  feature + a real clippy fix the never-linted wasm32 arm was carrying + a
  LITERAL `-p riir-examples` example row in that repo's guard layer 1.22 —
  a variable row reads as derived and would have kept the bucket). Standing:
  **23 NAMED · 0 UNRESOLVED · 0 UNCOVERED** over 191 files / 23 packages
  (measured 2026-09-08).
- ⛔ It produced three confident wrong answers before it produced a right one,
  all in the classifier: a walk of **0 files** (a Python `\s` handed to
  `git grep -E`, which is POSIX ERE — caught only because the walk size prints
  next to the verdict), then **17** false UNCOVERED (a *derived* `-p` list —
  the better design — read as the worst result), then **2** more (a
  `--manifest-path "$unit/…"` lane read as a bare row). A classifier's bucket
  boundaries ARE the finding, and they are only testable against cases whose
  answer is known independently.

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
cargo heal <paths>                                        # DRY RUN (the bare default — zero edits)
cargo heal --fix <paths>                                  # REAL fix: writes + compile-gates (fix_verify builds)
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

**Lossy-surface promotion rule (riir-ai Issue 750 T3):** a **lossy** surface
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
Issue 047; orchard/motivation riir-ai Issues 490/493).

Research workflow (paper classification, 7-repo routing, fusion-first
distillation, novelty + GOAT gates, modelless-unblock protocol §3.5):
`.agents/skills/research/SKILL.md`.

> **Repo count:** the **product/distillation set is 7** — `katgpt-rs` (public) +
> `riir-ai`, `riir-chain`, `riir-neuron-db`, `riir-train`, `riir-game-sdk`,
> `riir-dapps` (private). That is NOT the repo total: the
> workspace is **20 repos**, all of which carry a root `BOUNDARY.md`
> (add `riir-mmorpg-examples`, `riir-clippy`, `riir-viewbridge`,
> `riir-auth`, `katgpt-web`, `riir-dao`, `riir-deployer`,
> `riir-esp32`, `seal-game-editor`, `seal-remake`,
> `seal-online-remaster`, `riir-kat`, `riir-shader`).
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
