# AGENTS.md — katgpt-rs

The global `~/.agents/` rules apply; this file documents repo-local context
that supplements them. History, resolved-issue records, gate narratives,
collision precedents, and the dated incident arcs behind every rule below:
`HISTORY.md` (compacted 2026-10-30 — every `##` heading kept; full
pre-compaction text in git history). Removed issue files: git history.

## Boundary contract — read `BOUNDARY.md` first

[`BOUNDARY.md`](BOUNDARY.md) is the authoritative contract: what this repo
**owns**, what it **does not own** (with the correct home for each), the
crate-granular **allowlist**, and the **drift ledger**. On any conflict with
prose, BOUNDARY.md wins.

- **Domain test:** is this a **modelless inference primitive** with no riir dep
  (this repo is upstream of everything)? NO → it belongs in another repo; file
  there.
- Read it before adding any dep, crate, module, System impl, or vocabulary type.
- Enforcement: `../riir-ai/scripts/ci_boundary_contract.sh` — run boundary
  checks VIA the `boundary-guard` skill, not ad-hoc greps.
- Found a violation? File the issue FIRST (`.issues/NNN_boundary_*.md`), add
  the drift row, then fix. Closing the issue removes the row in the same commit.

## Modelless-first mandate (the core principle)

**This repo ships modelless inference primitives.** No training, no backprop,
no gradient descent. The only weight mutations allowed at runtime:

1. **Freeze/thaw** — swapping a frozen snapshot (atomic, versioned, BLAKE3-checked).
2. **Raw/lora hot-swap** — a **deterministically constructed** (not trained)
   LoRA overlay via `LoraPair { reader, writer }` (Plan 025).
3. **Latent-space updates** — direction-vector projections, sigmoid gates,
   routing tables; latent state, NOT base weights.

**MANDATORY: exhaust modelless paths before deferring to riir-train** (research
skill §3.5). A systematic, characterizable, named bias is a modelless-correctable
candidate — try a deterministic reader-LoRA or freeze-state correction before
concluding "needs gradient descent" (canonical failure: AC-Prefix G1, Plan 313).

## Build Commands

```bash
# Toolchain: pinned by rust-toolchain.toml (1.99.0 + rust-src, issue 963; the
# 1.98.1 anchor lineage is issue 739) — cargo resolves
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

# Supply-chain advisories gate (riir-refine plan 298 T2.4): vulnerable CRATES at
# lockfile level, advisories-only; deny.toml ignore list is a membership pin
# (reason + tracking ref per row); --scan is a report-only workspace sweep; a
# missing cargo-deny is a LOUD SKIP.
./scripts/deny_gate.sh
```

### The full gate — none of the above is a whole-repo claim

Every command above is narrow in at least one **independent** axis; a green
result says nothing about what it compiled to nothing:

- **check vs clippy** — some `cargo refine` escape classes are rejected only
  by clippy's typeck (E0689, E0631).
- **default vs `--all-features`** — non-default gated code compiles to nothing.
- **`-p <crate>` vs `--workspace`** — root defaults can switch on a crate's
  non-default feature; per-crate runs silently shrink coverage.
- **no `--all-targets`** — skips tests/benches/examples; doc-tests need
  `cargo test --doc` even then.
- **dev vs `--release`** — `debug_assertions` is ON in dev. **The profile is
  part of the claim.**
- **compile vs EXECUTE** — only the scoped core (katgpt-rs + katgpt-core
  `--lib`, default features) is EXECUTED weekly (`test.yml` +
  `scripts/test_gate.sh`). An uninvoked assertion is *unknown*, not passing.
- **host triple vs wasm32, and x86_64** — **a platform is part of the claim.**

So before claiming a repo-wide green, run:

```bash
cargo clippy --workspace --all-targets --all-features --keep-going -- -D clippy::needless_range_loop -D clippy::map_clone -D clippy::iter_cloned_collect -D clippy::identity_op -D clippy::bool_comparison -D clippy::manual_is_multiple_of -D clippy::collapsible_if -D clippy::map_all_any_identity -D clippy::unnecessary_cast -D clippy::manual_repeat_n -D clippy::question_mark -D clippy::empty_line_after_outer_attr -D clippy::unusual_byte_groupings -D unused_mut -D unused_parens
```

- The `-D` list holds only lints whose all-features residual is ZERO — never
  add a lint with residual > 0. `--keep-going` is mandatory (without it the
  run stops at the first failing target and under-reports).
- Don't run it by hand — `scripts/full_gate.sh` is the assertion (refuses a
  pass off macOS; Layer 5 checks this document quotes the command it runs).
- `--all-features` is NOT a supported TEST configuration (per-feature
  RNG/GOAT calibrations); the full test surface is priced, not run
  (`.benchmarks/701_full_workspace_execution_pricing.md`).
- ⛔ On a non-macOS workstation: `scripts/full_gate.sh --allow-partial-platform`
  — a SUBSET verdict that never prints `✓ full gate PASSED`.
- **wasm32 (Layer 2b):** derived `-p` list including the root package, both
  simd128 arms, two wasm32 GOAT targets by name, residue pinned by membership.
  `--all-targets` dies on dev-deps that don't resolve for wasm32 — add extra
  coverage as named targets. Push lane: `.github/workflows/wasm32_gate.yml`
  (main-only, `--wasm32-only`; dispatch manually after develop work).
  (`scripts/build-moka-wasm.sh` is a deploy build, not a gate.)
- **x86_64 lint lane (Layer 2c):** `--all-features` on the x86_64 triple.
  ⛔ When repairing one arch arm of a kernel, repair its SIBLING transcription
  too. The inverse holds: running ON macOS drops every
  `not(target_os = "macos")` backend — typecheck it and PROVE the gated
  modules compiled (rustc dep-info; `--min-gated N` floors the count). Args are
  `<repo> <pkg> [features...]` (no `--canary` flag):

```bash
scripts/check_platform_gated_modules.sh --min-gated 1 ../riir-train riir-train-gpu numeric_drift_cuda
```

- **Profile × features matrix (Layer 6b):** (dev, default) test_gate · (dev,
  all) Layer 3 · (release, all) Layer 6 · (release, default) Layer 6b.
  Alloc-gated tests carry `#[cfg(any(debug_assertions, feature = "alloc_tracking"))]`.
- **Trigger health:** PUSH triggers are MAIN-ONLY + dispatch-only (no CI on
  `develop` pushes); this repo's weekly `schedule:` is armed.
  `scripts/ci_gate_coverage.py` reports which declared triggers can fire per
  workflow per repo (report, exit 0; deliberately not a CHECK).

### The x86_64 half of the EXECUTE row — `scripts/x86_64_execution_matrix.sh`

The matrix EXECUTES the x86_64 arms
([Bench 806](.benchmarks/806_x86_64_execution_matrix.md)).

```bash
scripts/x86_64_execution_matrix.sh              # the matrix
scripts/x86_64_execution_matrix.sh --libs-only  # skip the root integration cell
scripts/x86_64_execution_matrix.sh --canary     # prove the floors fire
X86_MATRIX_DIR=/f/scratch scripts/x86_64_execution_matrix.sh
```

- **Workstation verdict**, no CI lane. **REFUSES off x86_64** (a green there
  would be a green zero).
- `RUSTFLAGS="-C target-feature=+avx2"` is **not a tuning knob** — without it
  compile-time-gated arms compile to nothing and the scalar fallback runs.
- **Run the matrix ALONE** (whole-box compiler load causes Windows
  C1001/D804; `CMAKE_BUILD_PARALLEL_LEVEL=1` is only a robustness cap).
- Population is **derived** (any package owning a tracked `*.rs` mentioning
  `target_arch = "x86_64"`); dep-only reach joins by a pinned row.
- **Every failure is RE-RUN ALONE, then adjudicated by MEMBERSHIP**
  (`scripts/x86_64_matrix_expected.txt`, reason per row, currently EMPTY).
  Bucket is **`PASSED-ALONE`, never `TRANSIENT`** — it hides three classes
  needing opposite responses: load-sensitive perf BAR, shared-fixed-path
  CONCURRENCY defect (passes alone by construction), unseeded-RNG COIN FLIP.
  Calibration-failed bars get arch-conditional dual pins, never re-typed pins.
- Floors in `scripts/x86_64_matrix_floors.txt` (`min_passed` per package +
  the integration cell's targets AND assertions); `--canary` proves they fire.
- ⚠ Does NOT cover macOS device backends, wasm32, `--all-features` integration
  targets, and does NOT LINT (that's Layer 2c).

## Docs gate + drift sweeps

**Repo names in instruments are CONTRACT names** (`scripts/repo_alias.py`):
an optional gitignored `scripts/repo_alias.local.txt` (`on-disk=contract`)
maps a box's on-disk names; absent → identity. ⛔ The ONE open seam is
`sweep_population.open_repo(name, workspace)` / `repo_alias.real(repo)` —
never open a contract spelling as a directory directly. Prose half:
`issue_citation_gate.spelling_aliases` (leniency only; `_NAME` boundary regex).

`scripts/docs_gate.sh` runs the manifest/doc/skill drift assertions and
prints its own timing. Cite the CPU figure **with its load class** and only
within a fixed CHECKS count; the `⏱` line shows which check is BLOCKING, not
that one got slower. On Windows the gate CALIBRATES and prints
`CPU SUPPRESSED` rather than a wrong number. `.github/workflows/docs_gate.yml`
runs it on **`main` only** — run `./scripts/docs_gate.sh` locally for develop
work. One line per check:

| check | asserts |
|---|---|
| `count_features.py` | flag counts in README + examples/README vs every manifest |
| `bench_doc_audit.py` | default-on / opt-in labels in .benchmarks + .docs vs Cargo defaults — plus two blindness floors and `BlindRead`, exit **2** |
| `cargo_comment_audit.py` | inline Cargo.toml comments vs the default closure |
| `skill_repo_set_gate.py` | hand-typed repo sets in SKILL.md command blocks (Issue 703) |
| `agents_repo_set_gate.py` | AGENTS.md §Repo count membership vs `scripts/repo_set.txt` — pins the paragraph below |
| `cfg_gated_floor_gate.py` | `#![cfg]`-gated targets that report a green 0-pass (Issue 713) |
| `orphaned_attr_gate.py` | a `#[cfg]` separated from its item by a blank line |
| `percentile_floor_gate.py` | a percentile index that lands on n-1 and so reports the MAX |
| `numbering_gate.py` | a number allocated twice, or a stale/malformed `.highwater` — walled by membership above the era boundary, ratcheted below it; `.benchmarks/` stem-CLASSIFIED: COLLISION walls at zero unpinnable, rename/hand-read ratchets at measured, FAMILY-EXPLICIT silent (Issues 724, 725, 795) |
| `dual_allocation_gate.py` | this checkout and its upstream both allocated a numbered document since their merge base — TWIN (same stem, the rebased own line) annotates exit-neutral, INDEPENDENT (two documents claiming one number) exits 1 naming both sides' adding commits (Issue 796) COUNTER exits non-zero too: the first two verdicts compare DOCUMENTS, and a number allocated and CLOSED in one commit never has a file in any tree, so the counters are the only thing that sees it. A one-sided bump stays green by construction; armed on every push with the reader injected and by a real two-repo fixture under --prove-fires (Issue 850) |
| `docs_gate_paths_sync.py` | docs_gate.yml's two hand-duplicated trigger `paths:` lists stay identical (Issue 724 T4b) |
| `required_features_static_gate.py` | a required-features row naming a feature its package cannot enable (riir-train Issue 513) |
| `cfg_row_implication_gate.py` | a required-features row that BUILDS and compiles its target to NOTHING (riir-train Issue 513) |
| `population_sync_gate.py` | the ten contract-repo predicates must agree, and the registry that lists them must be COMPLETE (Issue 788) |
| `trap_sentinel_gate.py` | a shell gate whose abort would report exit 0 — this repo's own two, by MEMBERSHIP (Issue 734) |
| `issue_citation_gate.py` | a cross-repo `Issue N` citation naming no repo — it rebinds to the WRONG document once that number is allocated locally (Issue 749) |
| `markdown_fence_gate.py` | a fenced code block never closed — everything after it renders as code (Issue 756) |
| `platform_dead_code_floor_gate.py` | an item declared ungated whose every use sits behind a platform cfg (Issue 775) |
| `subprocess_encoding_gate.py` | a `subprocess` call that decodes with the SYSTEM locale (Issue 778) |
| `instrument_reachability_gate.py` | a tracked `scripts/*.py` no root and no documented instrument names (Issue 787) |
| `sweep_advisory_membership_gate.py` | a *_drift_sweep.py that does not call a FAMILY-WIDE MECHANISM — the Issue-797 worktree advisory (findings and floors then describe whatever the working tree happened to say) or the Issue-815/821 known-extra exemption (the sweep hard-reds on a repo the contract does not claim, with zero content findings); a REGISTRY, per mechanism and never pooled, because this gate governed one mechanism by name and watched Issue 821 miss 3 of 19 beside it (Issues 797 T5, 824) |
| `locale_io_gate.py` | text I/O that decodes/encodes with the SYSTEM locale — `Path.read_text`/`write_text`/`open()` in text mode with no `encoding=`; repair half `scripts/locale_io_fix.py` (Issues 829, 830) |
| `console_encoding_gate.py` | a tracked `scripts/*.py` that prints a non-ASCII glyph and defends neither stream — on a non-UTF-8 console it dies with no verdict; shared defence `scripts/console_safe.py`; sibling-repo repair `scripts/console_encoding_fix.py` (Issue 804) |
| `global_rng_gate.py` | an unseeded-global draw with no pin row — found by executing one commit twice; membership + reason per row, both directions (Issue 809) |
| `algebraic_op_ban_gate.py` | an `algebraic_div`/`algebraic_rem` CODE occurrence in tracked `*.rs` — div/rem banned outright; ceiling 0 (Issue 871) |
| `shared_temp_path_gate.py` | a test writing to a FIXED `env::temp_dir()` path — safe against sibling tests in one binary, truncated by any concurrent PROCESS running the same test; passes alone BY CONSTRUCTION, one TRANSIENT misdiagnosis class (Issue 832) |
| `cross_repo_path_dep_gate.py` | a `path = "../X"` dependency on a repo that is NEITHER on disk NOR in `repo_set.txt` — cargo resolves path deps even when `optional = true` (Issue 835); an ORPHAN cited from a crate its repo's root `[workspace] exclude` lists is EXCLUDED — reported on the PASS line, never walled, since the citing workspace still loads (seal-remake Issue 129) |
| `repo_registration_gate.py` | a per-repo pin file with no row for a registered, on-disk repo — registering a repo is a 22-file operation and only repo_set.txt is gated, so the other 21 are discovered one red sweep at a time; measured the day riir-llm joined, 19 of 21 sweeps red and four live findings sitting behind them (Issue 837) |
| `cross_module_attr_gate.py` | a tracked scripts/*.py naming an attribute a sibling module does not define — Python has no link step, so the reference resolves at CALL time and a rename is an AttributeError the next EXECUTION finds; when the importer is a gate the failure is NO VERDICT rather than a wrong one. Measured: a commit privatised is_checkout and deleted worktree_fixture, updating the seven in-module callers and neither of the two external ones — both of them CHECKS in this array, develop red for 6h40m. --prove-fires is two-sided, 0 findings at the parent and exactly those 4 at the commit; the runtime half of the coverage stays arm_reach_gate's 157.6s workstation verdict (Issue 848) |
| `import_health_gate.py` | a tracked scripts/*.py that does not IMPORT — the EXECUTION half of Issue 848, where the static half (cross_module_attr_gate) cannot reach: a circular import, a missing third-party dependency, a raise in top-level code. The instrument that already executes every module is arm_reach_gate's BASELINE-CRASH, kept out of this budget at 157.6s; this is its cheap half, one child, 0.11s of import. A per-module subprocess was measured too and is not worth 3x (10.28s vs 7.52s, identical verdicts). ⛔ The affordability measurement found its own blocker — 6.238s of 6.34s was ONE module whose whole body was top-level, guarded in the same change. MISSING-DEP is its own bucket and is never flagged: pinning it would red the gate on a box that HAS the package (Issue 848 T3) |
| `shipped_target_feature_gate.py` | a SHIPPED path selecting its fast arm on a COMPILE-time target_feature — that predicate is OFF by default on x86_64, so the arm compiles to NOTHING on every ordinary build and the dispatcher silently runs its fallback. AGENTS.md documented this shape only for GATES, where the cost is an unproven claim; on a shipped path the cost is latency on every call, measured at 4.4-5.6x in a DEFAULT-ON feature and 2.4-2.5x on bf16 RNE narrowing. Three exclusions a naive grep gets wrong — wasm32/simd128 (no runtime detection there, so a compile-time gate is the only option), NEON (implied by the arch), and the runtime probe's own body, which is the same attribute doing the opposite job. The key is line-free and resolved by brace counting, the two cases needing opposite lookups — an attribute inside a body belongs to its enclosing fn, one ON an item to the NEXT. Membership + a reason per row, both directions, floors on the walk and on the attribute parse; --prove-fires is two-sided (Issue 847 T3) |
| `timed_region_guard_gate.py` | a latency ceiling with no loud-zero defence — rustc + fat LTO deletes a timed loop whose result is dead, so the bar is satisfied by **absent work** and passes with MAXIMUM margin. This is the `#![cfg]` green-zero rule one layer down, and worse: the assertion RUNS, so the output is a plausible number rather than a zero count and no count floor can see it. ⛔ Not reasoned about — **executed**: all 34 asserting regions at n ≥ 1000 were run and **7 were satisfied by absent work** (20.6%), two of them GOAT gates, one printing `Speedup: 8657.9×`, one printing a well-formed `0.00x` because only the NUMERATOR vanished. ⚠ The static predicate Issue 855 T4 proposed is REFUTED by that run — `let _ =` vanished **3 of 15** against **4 of 19** for the rest, i.e. the base rate wearing a grep; the column that separates is `black_box`, **7 of 23** without it against **0 of 11** with. So the gate does not predict which region is broken; it gates the decidable thing — *is there a loud-zero defence at all*. ⛔ Two tiers, and the split is the honest part: a LITERAL loop bound is the population that was READ end to end (membership wall, one MEASURED number per row), while a bound needing one hop of resolution is real and **unread** (a ratchet on the derivative — pinning an unread bucket by name is Issue 785's forbidden shape). One hop is not optional: T1's own two founding specimens are `let n = 100_000; for _ in 0..n`, so a literal-only predicate would have shipped the class it was written for. ⚠ It does NOT claim a pinned region is safe — `black_box` is the weakest of three defences (result, **arguments**, **receiver**) and two arms vanished carrying one (Issue 855 T4) |
| `python_syntax_floor_gate.py` | tracked *.py parses at the 3.9 grammar — a 3.10+ only file is a dead tool on the workstation's python3 3.9.6, dying at import before any verdict; two ceilings (SYNTAX + UNPARSED) over walk + parse floors (Issue 965) |
| `check_validation_gate.py` | a CHECK in this array whose own arithmetic no arm asserts — including one whose arm is flag-gated and so never runs (Issue 789) |
| `docs_gate_checks_sync.py` | this CHECKS array vs the AGENTS.md table documenting it — membership both ways + quantity words (Issue 750) |
| `skill_size_gate.py` | a skill SKILL.md over the 80KB ceiling — the THIRD 100KB-regrowth class (doc-sync 09-05/09-11/09-21, boundary-guard 09-08/09-11/09-21: the one-line convention held, the ~8 rows/day cadence didn't) — the gate forces each file's documented prune-to-15 maintenance rule; recovery via git log -p |

**Partial-clone boxes:** the population checks (`skill_repo_set_gate`,
`population_sync_gate`, `issue_citation_gate`) hard-red on a subset box.
`DOCS_GATE_PARTIAL_CLONE=1` is an explicit, never-auto-detected DEFERRAL; boxes
using the alias codec must NOT set it. **Known-extra boxes:**
`DOCS_GATE_KNOWN_EXTRA=names` — NAMES, never `=1`; reds in both directions
(stale acknowledgement fails).

**Sweep population — `scripts/sweep_population.py`**: three verdicts, never
interchangeable — **UNREGISTERED** (on disk, absent from `repo_set.txt`; reds
always), **UNSEEN** (absent, no marker; never a pass), **DEFERRED** (with the
marker; rides the FINAL line). A subset sweep has TWO populations — hand
`population_verdict` the contract walk, never the subset. Registration scope:
`scripts/repo_registration_scope.txt`.

**The sweep family** — workstation-only cross-repo verdict halves, each with
two floors (a ceiling is green over whatever it can SEE, so the finding count
needs a population floor beneath it; a sweep restating a per-push gate's
quantity ASSERTS they agree): `docs_drift_sweep.py`, `numbering_drift_sweep.py`,
`required_features_drift_sweep.py`, `percentile_drift_sweep.py`,
`cfg_gated_drift_sweep.py`, `cfg_row_implication_drift_sweep.py`,
`trap_sentinel_drift_sweep.py`, `citation_drift_sweep.py`,
`restatement_drift_sweep.py`, `markdown_fence_drift_sweep.py`,
`platform_dead_code_drift_sweep.py`, `subprocess_encoding_drift_sweep.py`,
`locale_io_drift_sweep.py`, `console_encoding_drift_sweep.py`,
`timed_region_drift_sweep.py`, `shared_temp_path_drift_sweep.py`,
`pipefail_discard_drift_sweep.py`, `toolchain_override_drift_sweep.py`,
`orphaned_attr_drift_sweep.py`, `wasm32_surface_drift_sweep.py`,
`len_derived_drift_sweep.py`, `instrument_reachability_drift_sweep.py`,
`python_syntax_floor_drift_sweep.py` — plus report halves
`pipefail_discard_audit.py` and `toolchain_override_audit.py`, and
`sibling_docs_drift.yml` (reusable workflow). Ceilings pinned under
`scripts/*_floors.txt`. **Take the family size and wiring verdict from
`scripts/sweep_advisory_membership_gate.py`'s PASS line, never a count here.**

**Python 3.9 syntax floor:** every tracked `*.py` must parse at
`ast.parse(feature_version=(3, 9))` (the boxes' `python3` is 3.9.6, invoked
directly). SYNTAX and UNPARSED are separate walls at 0; floors `min_py_files`
+ `min_parsed`. A GRAMMAR check, not an interpreter pin (`tomllib` parses;
PEP 604 annotations ride `from __future__ import annotations`).

⛔ **Before fixing a rule-in-one-instrument class, grep the whole family and
land the repair as one shared mechanism.** ⛔ **A cross-repo repair is not
landed until COMMITTED in the sibling, and a record HERE claiming one must
CITE THE SIBLING COMMIT** (Issue 798) — never drop a pin row for an absent fix.

**Citations** (`citation_drift_sweep.py`): it prints its own error rates
(plural — a sample rate does not transfer to unsampled rows); never quote one
without the walk, population and the ORACLE it was adjudicated against.
IN-LOCAL-RANGE is UNDECIDED, never folded into a neighbour. Qualification
asks "does the named repo OWN the number?", not just "is a repo named?".
**Two-lane gap:** a repo with `.highwater_local` declares its local lane's top;
citations between it and the first git-log file ADD above classify
CROSS/ORPHAN (`--prove-fires` replays the specimen). **Heading oracle**
(`heading_allocated()`): incomplete by construction (house-style headings it
cannot read); take `heading_unread=a/b`, `heading oracle COST` and `novel=`
from the sweep's output. ⛔ A document that DISCUSSES a misattribution must
name the true owner inside the citation's own window, not add a pin. ⛔ Never
re-pin `max_in_local_range` for a heading-blind row.

⛔ **The pipefail class** (`scripts/pipefail_discard_audit.py` +
`scripts/pipefail_discard_drift_sweep.py`): `var="$(pipeline)"` under
`set -euo pipefail` is killed by a legitimately-empty grep AFTER the work ran.
`local x="$(fails)"` does not kill; an interior `|| true` guards a
paren-group. ⛔ Where the captured value is then TESTED, `|| true` alone turns
a silent death into a silent PASS — add an explicit emptiness failure too.
The four `proof_gate.sh` `grep -c` tripwires are DELIBERATE rows.

## cfg-gated targets — the green-zero rule

A test file opening with `#![cfg(feature = "x")]` compiles to an **empty
binary** when x is off: `ok. 0 passed`, exit 0 — byte-identical to a pass.
`required-features` protects the **reader**; the `#![cfg]` the **count**.
Read the default-on/default-off severity split, never the pooled total.

- ⛔ **Two spellings of the zero:** a body that is entirely
  `#[cfg(feature = "x")] mod tests { … }` is the same outcome. Predicate: the
  gated items are the WHOLE body; a run of gated modules is gated by
  `any(...)`; a top-level `use` never makes a target non-empty.
- **Profile traps:** multiple whole-file `#![cfg]`s are ANDed — read all. A
  MEASUREMENT gated on `debug_assertions` is impossible where it ships: if it
  is a **capability**, give it a feature (`any(debug_assertions, feature =
  "x")`); only a genuine profile property stays on `debug_assertions`. Read
  the auditor's `unfixable` vs `escapable` split.

```bash
scripts/cfg_gated_target_audit.py            # all contract repos (derived)
scripts/cfg_gated_target_audit.py ../riir-ai # or one, by path
```

- A **report, not a gate** (exit 0): `cfg` on `target_os`/`miri` and `any(...)`
  of features cannot be expressed as `required-features`.
- `scripts/suite_membership_audit.py` — which `[[test]]` targets no
  script/workflow names; run it when landing a new gate.
- **Arming a target can RED a binary-counting floor** (an empty binary counts
  as one) — repair with a **passed-test floor**, not a re-pin. Run armed
  latency gates with `--release`.
- Verdict half: `scripts/cfg_gated_floor_gate.py` (pins in
  `scripts/cfg_gated_floors.txt`; `max_load_bearing = 0`; some pins are
  FLOORS); `scripts/all_ignored_load_bearing.txt` pins the ALL-IGNORED set by
  MEMBERSHIP from `all_ignored_target_audit.py`.

## A `required-features` row can EXIST and be WRONG — `scripts/required_features_build_audit.py`

A wrong row is worse than a missing one: `cargo test --workspace` silently
**skips** the target, `--all-features` **builds** it, and every audit counts
it protected. Imports resolve through cfg-gated re-exports grep cannot follow
— ask the compiler, once per target:

```bash
scripts/required_features_build_audit.py --list            # rows only, no builds
scripts/required_features_build_audit.py ../riir-train     # one repo
scripts/required_features_build_audit.py . --grep pruners  # one slice
scripts/required_features_build_audit.py ../riir-train --batch  # 1 run per set
```

- **Report** (exit 0; ~28 s/row — filter with `--package`/`--kind`/`--grep`/
  `--limit`; `--target-dir` when a sibling is building).
- `--batch` = one cargo run per **EXACT** feature set, never a superset (a
  superset may supply the import the row forgot).
- **Neither error nor artifact = UNSEEN, never BUILDS**; never folded into pass.
- **Read the USE SITES:** widen the ROW when the body needs the feature
  unconditionally; narrow the cfg when the use site is already gated.
- Static verdicts (`required_features_static_gate.py`,
  `cfg_row_implication_gate.py` / `cfg_row_implication_audit.py`) are
  necessary, not sufficient. Push half (MAIN-ONLY):
  `.github/workflows/required_features_touched.yml` (`--max-rows` REFUSES
  rather than truncates).

## A reported "p99" is often the MAX — `scripts/percentile_index_audit.py`

`sorted[(n as f64 * 0.99) as usize]` and `sorted[n * 99 / 100]` land on
`n - 1` — the **maximum** — for every `n <= 1/(1-p)` (n ≤ 100 at p99, ≤ 20 at
p95, ≤ 1000 at p999). A `.min(len - 1)` clamp prevents a panic, not a wrong
statistic. Print **tail support** = `n - idx`; under 10 is weak. A `>= floor`
bar goes false-GREEN.

```bash
scripts/percentile_index_audit.py             # all contract repos (derived)
scripts/percentile_index_audit.py ../riir-ai  # or one, by path
```

- **Report** (exit 0). **UNRESOLVED is not clean** — dump those rows with
  `scripts/list_unresolved_percentile_sites.py <repo>`.
- Verdict half: `scripts/percentile_floor_gate.py` (pins in
  `scripts/percentile_floors.txt`; `min_sites_scanned` is a FLOOR).
- **The POPULATION is what git TRACKS — `scripts/tracked_walk.py`, one copy**
  (consumers: `platform_dead_code_audit.py`, `percentile_index_audit.py`,
  `len_derived_binding_audit.py`, `orphaned_attr_gate.py`). No `.git` → walk.
- ⛔ **`.git` answers TWO questions:** *is this a canonical REPO?* →
  `.is_dir()` (a worktree has a `.git` FILE); *can I run git rooted HERE?* →
  `.exists()` (delegate to `worktree_state.is_checkout`). Never write a fresh
  contract-repo walk — delegate to `skill_repo_set_gate.derive_repos`
  (`population_sync_gate` catches two predicates disagreeing). A name-match
  guard needs an else, and a glob fallback must not report untracked scratch.

## A Lean theorem can RESTATE its own definition — `scripts/restatement_theorem_audit.py`

`theorem x_eq_sum : x = magic + version + …` where the RHS **is** the `def`
body: `decide` closes it whatever the constants hold — a theorem that proves
nothing (four shipped in riir-neuron-db, removed `24957a2`).

```bash
scripts/restatement_theorem_audit.py               # all repos with .proofs (derived)
scripts/restatement_theorem_audit.py ../riir-chain # or one, by path
scripts/restatement_theorem_audit.py -v            # every row, not just findings
```

- **Report** (exit 0). Criterion: symbolic equality over **leaf** constants
  (unfold composite nullary `def`s; numeral-bodied leaves stay symbols).
- **CROSS-DEF is not a finding** (pins two independent definitions);
  **UNRESOLVED is not clean**; HYPOTHETICAL (binders) split out.
- Verdict half: `scripts/restatement_drift_sweep.py`
  (`scripts/restatement_drift_floors.txt`; floors `min_lean_files` = walk,
  `min_theorems` = parse; `--prove-fires` plants a restatement into a COPY).

## A ratio of two SEQUENTIALLY-timed arms measures the BOX — `scripts/sequential_ab_timing_audit.py`

Two sequential arms of the same work differ by +5–22% on a loaded box — a 10%
bar measures the scheduler. Treatment: `tests/common/ab_timing.rs` —
interleaved (a, b) chunk pairs, median across pairs, loud zero when an arm is
deleted (`best_of_us` absolute budget; `best_of_arms` N-arm round-robin min).

```bash
scripts/sequential_ab_timing_audit.py            # this repo
scripts/sequential_ab_timing_audit.py ../riir-ai # or one, by path
scripts/sequential_ab_timing_audit.py -v         # every row, not just findings
```

- **Report** (exit 0; exit **2** on a blindness floor or failing self-test).
- Buckets, never pooled: **ADOPTED** · **HAND-ROLLED** (DRY finding) ·
  **SEQUENTIAL** (two timing-derived identifiers in one comparison, ratio or
  bare `assert!(a < b)`) · **UNRESOLVED** (not clean).
- ⛔ **No verdict half, deliberately** — migration is a per-target read on
  four axes (a/b orientation, claim direction, chunk size, `black_box` at both
  ends). **Execute before migrating**: thresholds `< 6` pts flakes idle ·
  `6–22` under load · `≥ 22` out of reach.
- STATED blind spots: a ratio through a helper; an uncompared, undivided
  subtraction; orientation.

## A gate that ABORTS reports exit 0 — `scripts/trap_exit_launder_audit.py`

On **macOS `/bin/bash` 3.2.57** (also GitHub's `macos-26-arm64`, so
load-bearing IN CI), aborting on an **unbound expansion** or an **`eval` syntax
error** enters the EXIT trap with `$?` already 0 — an EXIT trap whose last
command succeeds makes the abort exit **0**. `trap 'rc=$?; cleanup; exit $rc'
EXIT` does **not** repair it. Only a **completion sentinel** does: a flag set
on the script's own last line, checked by the handler, forcing exit 1 when
INCOMPLETE and claiming success (`scripts/full_gate.sh`,
`scripts/proof_negative_test.sh` carry it). Keep the sentinel on every shell.

```bash
scripts/trap_exit_launder_audit.py            # population + verdict, all repos
scripts/trap_exit_launder_audit.py ../riir-ai # or one, by path
scripts/trap_sentinel_drift_sweep.py          # the verdict, every repo, pinned
scripts/trap_launder_premise_matrix.py        # the PREMISE, 11 interpreters
```

- Halves: audit = population (exit 0); `scripts/trap_sentinel_gate.py` =
  this repo's verdict (membership pins, population floor; the flag must GATE
  the failure branch); the sweep = all repos
  (`scripts/trap_sentinel_drift_floors.txt`); the premise matrix prints
  **UNSEEN, never zero**, without docker.
- **`errexit` is the precondition, NOT `nounset`**; the measurement MODE is
  part of the claim.
- Verdicts: LIVE-FORWARD, EXPOSED, SENTINELLED, UNPARSED (the instrument
  cannot read — reds, never pooled), REPLACED (2+ EXIT traps; `trap`
  replaces). A window with zero abort triggers cannot launder.
- **shellcheck does not find this.**

## A lane compiles what it NAMES — `scripts/wasm32_surface_audit.py`

Is what a wasm32 lane names the whole surface? A positive
`#[cfg(target_arch = "wasm32")]` block no row builds is uncompilable unseen.

```bash
scripts/wasm32_surface_audit.py            # all contract repos (derived)
scripts/wasm32_surface_audit.py ../riir-ai # or one, by path
```

- **Report** (exit 0). Buckets: **NAMED** · **BY-DEP** (reached via
  non-optional in-repo path deps — never folded into NAMED) · **UNRESOLVED**
  (a `--workspace`/derived row — not clean) · **UNCOVERED**. `--self-test`.
- Predicate is the **positive** cfg; comments excluded; reads ATTRIBUTES, not
  lines (masker imported from `platform_dead_code_audit`). `cfg!(...)` runtime
  branches and **SHARED** (`any(...)` with a native disjunct) are EXCLUDED.
  A lane is ANY `wasm32-*` triple.
- Verdict half: `scripts/wasm32_surface_drift_sweep.py` (`max_unresolved = 0`
  WALL; UNCOVERED pinned by NAME in `scripts/wasm32_uncovered_expected.txt`;
  floors in `scripts/wasm32_surface_drift_floors.txt` with a TOTALS row).
  **Take the standing from a run.**

## An arm that exists and RUNS may still reach nothing — `scripts/arm_reach_audit.py`

Mutate a module's source **outside its own arm bodies**, re-exec, run its arm,
ask whether the arm noticed.

```bash
scripts/arm_reach_audit.py                    # the report, the CHECKS population
scripts/arm_reach_audit.py --self-test        # arms over its own buckets
scripts/arm_reach_audit.py skill_repo_set     # one module, by substring
scripts/arm_reach_audit.py --include-all      # every scripts/*.py DEFINING an arm
scripts/arm_reach_gate.py                     # the VERDICT — workstation, minutes
scripts/arm_reach_gate.py --canary            # arms over its own arithmetic
```

- **Report** (exit 0; exit 2 on blindness floor / failing self-test).
- ⛔ `BASELINE-RED`/`BASELINE-CRASH`: an arm already failing kills every
  mutant and reads as 100% — module-level verdicts, walled at 0. Check the
  ENVIRONMENT first on a RED.
- ⛔ `TIMEOUT` = evidence of nothing; deadline 10× the module's baseline. The
  watchdog cannot interrupt a blocking C call — run `--include-all` module by
  module under an external timeout.
- The exec namespace is a registered module (a bare dict dies at `@dataclass`).
- **NO-ARM** and **UNREACHED** are separate markers. A classifier's self-test
  cannot reach its consumer's pin arithmetic — add a `gate_selftest` and
  EXTRACT the verdict arithmetic out of `main()`; extraction IS the repair.
- ⚠ Reach is per MODULE (check shared rules first on a survivor); EQUIVALENT
  mutants exist; operator scope is control flow + off-by-one only (no regex /
  string literals) — SURVIVED is arm reach, not a defect count.
- Verdict half `scripts/arm_reach_gate.py`: survivors pinned by MEMBERSHIP with
  a REASON (`scripts/arm_reach_survivors_expected.txt`), line-free keys, each
  `#= ` comment verified against observed text; `EXEMPT_FUNCTIONS` /
  `ARM_NAMES` pinned by membership. Cost is a RANGE (minutes to 2 h) — run in
  a DETACHED worktree; kill by PID, never by pattern.

## A gate whose own failure path is asserted by nothing — `scripts/check_validation_gate.py`

Every CHECK must **invoke an arm UNCONDITIONALLY**: `docs_gate.sh` runs each
check with no arguments, so an arm behind `'--canary' in sys.argv` never runs
on a push.

- **Delegation is credited** (an arm reached through an imported classifier),
  but the consumer's pin arithmetic still needs its own `gate_selftest`.
- Exemptions by membership + reason (`scripts/check_validation_expected.txt`,
  deliberately empty). Floors `MIN_CHECKS` (array parse) + `MIN_ARMED` (AST
  resolution); a canary asserts `main` is not in `ARM_NAMES`.
- `--prove-fires <sha>` is two-sided (opt-in).
- ⚠ Does NOT assert an arm is any *good* (that's `arm_reach_audit`).
- ⛔ **No sweep half** — only katgpt-rs has a CHECKS array; re-measure before
  adding one by symmetry.

## A census reads the DOCUMENT, so an undocumented instrument is invisible — `scripts/instrument_reachability_gate.py`

A census of what AGENTS.md documents cannot see an instrument it omits.

```bash
scripts/instrument_reachability_gate.py                  # the verdict, this repo
scripts/instrument_reachability_gate.py --canary         # the adversary arms
scripts/instrument_reachability_gate.py --prove-fires 18dbe980
scripts/instrument_reachability_drift_sweep.py           # every repo, ratcheted
```

- Predicate is **REACHABLE**: roots `AGENTS.md`, `scripts/docs_gate.sh`,
  `.github/workflows/*.yml`; the closure follows script → script references
  (e.g. `all_ignored_target_audit.py`, `cfg_row_implication_audit.py`,
  `ci_test_execution_report.py` are reached via documented instruments).
- ⛔ **`HISTORY.md` is NOT a root** (archive, not loaded into a session).
- Unreachable pinned by **MEMBERSHIP + REASON**
  (`scripts/instrument_unreferenced_expected.txt`); **default for a real
  instrument is to make it findable, not to add a row.**
- Floors: `min_scripts` (walk), `min_roots` (permissive direction).
- Sweep half is a **ratchet** (the predicate over-captures plan-scoped
  one-offs elsewhere); the closure is TEXTUAL — any basename mention credits.

## A kernel can derive its SHAPE from a buffer's declared size — `scripts/len_derived_binding_audit.py`

`let n = kv.len() / 2 / kv_stride;` derives a dimension from the bound
buffer's **declared size**; bind a capacity-sized handle and the kernel
silently reads never-written memory and writes a measured all-zero result.
The defect is a **JOIN** across two files; HALF C resolves wrapper parameters
through **workspace** callers (cross-repo by construction).

```bash
scripts/len_derived_binding_audit.py        # the report, all contract repos (derived)
scripts/len_derived_drift_sweep.py          # the verdict, every repo, pinned
scripts/len_derived_drift_sweep.py --no-stability   # skip the leave-one-out arm
scripts/len_derived_drift_sweep.py --canary         # the adversary arms
```

- **Report** (exit 0) except a WALK REGRESSION. **UNRESOLVED is not clean.**
  **PERSISTENT-UPSTREAM is the EYES LIST**, pinned by MEMBERSHIP in
  `scripts/len_derived_eyes_expected.txt` (line-free key + count).
- Verdict half walls CAPACITY, CAPACITY-UPSTREAM, PERSISTENT at 0; floors
  `min_kernels`/`min_binds` per repo. ⛔ `DEFERRED` is not enough — a partial
  clone can flip a present repo's verdict, so the sweep re-measures a TARGETED
  leave-one-out every run.
- No `min_rs_files` column: delegated (and ASSERTED) to
  `orphaned_attr_drift_floors.txt`; an md-only repo's zero row is accepted
  only while it measures zero tracked `.rs` (Issue 902).

## An item can be dead on a platform NO lane compiles — `scripts/platform_dead_code_audit.py`

An ungated item used only inside `#[cfg(target_arch = …)]` code is dead
everywhere but that arch — the warning appears only on a workstation lane.

```bash
scripts/platform_dead_code_audit.py             # all contract repos (derived)
scripts/platform_dead_code_audit.py ../riir-ai  # or one, by path
scripts/platform_dead_code_audit.py --self-test # the classifier arms
scripts/platform_dead_code_audit.py --prove-fires ea4c2873
```

- **Report** (exit 0) except a classifier MISS (exit **2**). Tracked `*.rs`
  only; `vendor/` excluded with its count printed.
- **MOD-REF** is separate (a `mod` used only from gated code is empty, not
  dead); never folded into the count.
- ⛔ An arm is only a canary if its own perturbation REDS it.
- Verdict halves: `scripts/platform_dead_code_floor_gate.py` per-push (pins
  `scripts/platform_dead_code_floors.txt`, two blindness floors, MOD-REF by
  MEMBERSHIP, canary arms over pin arithmetic) and
  `scripts/platform_dead_code_drift_sweep.py` (pinned in
  `scripts/platform_dead_code_drift_floors.txt`; `--prove-fires` default-on in
  the sweep, opt-in on the gate).

## `text=True` decodes with the SYSTEM locale — `scripts/subprocess_encoding_gate.py`

`subprocess.run(..., text=True)` decodes with `locale.getencoding()`; on a
non-UTF-8 box that is silent mojibake (a regex reads a confident zero) or a
reader-thread decode error where `run()` returns with `stdout = None`.
`PYTHONIOENCODING=utf-8` fixes only the CHILD's encoder. Two classes, pinned
separately: **DECODE** (`text=True` with no `encoding=`) and **CHILD-ENCODER**
(a `sys.executable` spawn with no `PYTHONIOENCODING` in `env=`).

- AST scan (string literals are literals); a file it cannot parse is
  **UNPARSED** and reds. Ceilings 0 on both classes over floored populations.
- Verdict half: `scripts/subprocess_encoding_drift_sweep.py` (floors
  `min_calls` and `min_py_files` — the latter is the only blindness detector
  where there are no calls). Population is tracked `*.py`, not `scripts/*.py`.
- Repair (DECODE only): `scripts/subprocess_encoding_fix.py [--list] <files>`
  appends `encoding="utf-8", errors="replace"` (imports the gate's predicate +
  `locale_io_fix.insert_kwarg`). CHILD-ENCODER is a per-site read.
- File-I/O sibling: `scripts/locale_io_gate.py` / `scripts/locale_io_fix.py` /
  `locale_io_drift_sweep.py`; console sibling: `scripts/console_encoding_gate.py`,
  shared defence `scripts/console_safe.py`, repair
  `scripts/console_encoding_fix.py`.

## A sweep reads the WORKTREE, so a finding may exist in NO commit — `scripts/worktree_state.py`

Concurrent sessions share worktrees. Verdicts, never interchangeable:
**COMMITTED** (matches HEAD) · **UNCOMMITTED** (worktree-only; displayed,
never adjudicated — the DISPLAY reads the worktree, the PINS read HEAD) ·
**MASKED** (HEAD-only; a committed defect read clean — the silent direction).
⛔ Never re-pin a floor from a dirty run. A fourth axis, **STALE**:
`behind_origin(root, patterns)` (three-dot diff; rides `sweep_advisory()`;
the sweep stays RED); `upstream_axis()` is the shared upstream half.

- **A sweep does NOT fetch** — fetch ONCE per session with
  `scripts/fetch_contract_repos.py` (names `origin`; per-repo `--timeout`;
  exit 1 only on fetch FAILURE) and let sweeps DISCLOSE.
- ⛔ **Pick the instrument from the classifier's shape**: `head_delta`
  (per-file independence; `.head` = committed + masked, **never
  `committed`**) · `head_overlay` + `delta_of` (one interceptable reader;
  empty dict = skip; `None` is a VALUE) · `head_tree` (classifier with several
  seams, unmodified; yields `None` on clean — caller MUST skip; `paths=`
  narrows, `extra_dirty=` widens). Cross-file classifiers must not use the
  per-file shortcut. Patterns must name every file that can change a verdict,
  ROOTS included; a bare `("*.rs")` is a string, not a tuple.
- ⛔ **The VERDICT belongs in the row KEY**; **a FLOOR is a pin too** (reads
  HEAD). Row keys are LINE-FREE.
- **ADVISORY, never a failure**; rides the FINAL line in both directions,
  silent when nothing dirty meets the sweep's population. Wiring is gated by
  `scripts/sweep_advisory_membership_gate.py` — a MECHANISMS registry
  (worktree-advisory, known-extra-exemption, head-provenance, alias-open),
  verdict per mechanism never pooled, by MEMBERSHIP, both directions,
  exemption file empty. It asserts the CALL, not that the patterns are right.
- Budget canary cost before wiring one: each sweep `--canary` re-enters
  `main()` over every contract repo.

## Before committing in a shared worktree — `scripts/staged_set_audit.py`

Stage **named files** (`git -C <repo> add <paths>`), never `-A`. Before a
multi-file commit, run:

```bash
scripts/staged_set_audit.py            # any repo: pass its path as $1
```

- **Report** (exit 0). Signals: **mtime clusters** (two editing episodes) ·
  **also-dirty** (concurrent editor) · **stale-vs-HEAD** (committing would
  revert lines the newest commit added) · **rustfmt round-trip** (`--fmt`;
  identical to `rustfmt(HEAD)` = provably zero content).
- Committing into a file a sibling is editing: commit **your blob** (HEAD +
  your edit, `git hash-object -w`, `git update-index --cacheinfo`).
- ⛔ **Nothing in git identifies which session did something** (shared
  author, one `HEAD` reflog, incomplete rosters). Attribute a message by its
  `from=` pipe, never its self-signed name. Say what you CHECKED, not who you
  concluded. **Put `Session: <name>, <epoch>` in the commit body** (names are
  reused — the epoch is load-bearing); it is evidence, not proof.

### The same hazard one layer down: a FIXED temp path — `scripts/shared_temp_path_gate.py`

`std::env::temp_dir().join("fixed_name.bin")` is safe against sibling tests
in one binary and **not** against another PROCESS running the same test
(`create` truncates). ⛔ Such a defect passes alone BY CONSTRUCTION — "failed
in the cell, passed alone" does not mean TRANSIENT.

- Repair: `std::env::temp_dir().join(format!("name_{}", std::process::id()))`.
  Gate: membership + reason per row, both directions, floors on walk AND
  predicate. Blind spots: `temp_dir()` bound to a variable first; other
  fixed-scratch spellings.
- ⚠ Never `cargo fmt -p <crate>` for this repair — format clean files per-file.
- Verdict half: `scripts/shared_temp_path_drift_sweep.py` (ratchet). Repair
  half: `scripts/shared_temp_path_fix.py` (idempotent, LF-preserving,
  `--max-width`; imports the gate's `mask` and `FIXED_JOIN`).

**Shared target dir:** a concurrent cargo process in the same `target/` can
make a gate report a failure that passes alone. `error: test failed` with **no
`failures:` block and no `test … FAILED` line** = the harness process DIED
(count truncated). Diagnose by running the binary in `target/<profile>/deps/`
directly. A gate whose verdict the box can invalidate should **refuse**, not
warn — detect concurrent cargo by working directory (cargo releases
`.cargo-lock` before running test binaries).

## Formatting — rustfmt defaults, gated (Issue 946, adopted 2026-10-17)

rustfmt-DEFAULT style (`rustfmt.toml` sets only `edition`). **Run `cargo fmt
--all` (or `rustfmt <file>`) before committing `.rs`.** ⛔ Never `cargo fmt -p
X -- a.rs` (everything after `--` is a rustfmt OPTION — formats the whole
package). Format commit is in `.git-blame-ignore-revs`. Verdict:
`scripts/fmt_gate.py`, `full_gate.sh` Layer 7 — drift pinned by MEMBERSHIP in
`scripts/fmt_pending_expected.txt`, a drain list that only shrinks (format a
pinned file, delete its row; never add a row).

## Lint healing — `cargo refine` before manual fixes (adopted 2026-08-24)

Mechanical clippy findings are fixed by the riir-refine healer FIRST:

```bash
cargo refine <paths>                                        # DRY RUN (zero edits)
cargo refine --fix <paths>                                  # REAL fix: writes + compile-gates
cargo refine --fix --write --verify <paths>                 # compile-gated apply
cargo refine --fix --write --verify --verify-args "--features <set>" <paths>  # gated code
```

- `~/.cargo/bin/cargo-refine` → sibling `riir-refine/target/release/cargo-refine`
  (RELEASE feature set; `--version` prints the truth). Missing → manual +
  `cargo clippy --fix`.
- `--verify` auto-REVERTS breaking edits; gated code needs `--verify-args`.
- Fixes only what THIS repo's clippy reports (no pedantic/nursery here);
  silent on documented divergence classes — those stay manual (see the
  `cargo-refine` skill). Misses → session record / riir-refine issue.

## Determinism law — synced f32 reductions and picks (Issue 940)

An f32 reduction or pick whose result crosses the sync/quorum boundary
(committed, exported, hashed, replay-compared) pins its reduction order and
tie rule: sequential fold (`dot_f32_ordered`, plain `+=`/`.sum()` — never a
platform-lane SIMD kernel) and one fixed strict tie rule (first-index or
documented last-index — never a tolerance band, never HashMap order).
Otherwise reclassify local-only with the decision recorded at the site.
Census: `.docs/10_audits/float_reduction_order_tie_band_determinism_audit.md`.
Precedent: `katgpt-core/src/closure/bridge_certified.rs`.

## Fixed-form softmax numerics family — `katgpt-core/src/softmax_fix.rs` (Issue 959)

`precision_cliff`/`PRECISION_CLIFF_NATS` (c_d = p·ln2 per accumulator format,
ungated), `margin_equilibrium(eps, format)` (ungated), opt-in `logspace_diff`
(`logspace_diff(_f64)`), opt-in `fixed_log_softmax` (`fixed_log_softmax(_f64)(_target)`,
argmax-excluded-LSE + softplus; first-index argmax + sequential ascending-j
fold pinned per Issue 940; LOCAL instrument — sync crossings go through
`closure/bridge_certified`). Instrument-grade only, never a serving forward.
Offline CLI: `scripts/margin_equilibrium_audit.py` (`--self-test`, `--sweep`).

## Tail-subspace weight-delta substrate — `katgpt-core/src/sst_certified.rs` (Issue 958)

Opt-in `sst_project` (implies `subspace_phase_gate` + `karc_forecaster`):
`fit_anchor`, `anatomy`/`anatomy_factored_into` (energy split as TWO POSITIVE
terms, never the cancelling form), `factor_tail` (quantization-safe `ΔW =
U_⊥ C V_⊥ᵀ`; dropped energy REPORTED), `certify_decoded_tail` (RELATIVE
tolerance, band [1e-5, 1]), `compose_tail` (same-anchor; basis mismatch
refuses loud), rotation-invariant anchor digest + `SstCertificate::verify_against`
(stale certs refuse). `katgpt_core::sst_cross_tail` (Issue 972, same gate) is
REPORT-ONLY cross-tail interference (CKA + cos² spectra; refusal classes mirror
`compose_tail_into`; f64 sequential folds). Lean pair in
`.proofs/KatgptProof/Sst/`. Consumers: riir-neuron-db Issue 629, riir-ai Issue
1116, riir-ai Issue 1141; producer riir-train Plan 470.

## Feature Flag Discipline

Every new primitive ships behind a feature flag (opt-in). Promotion to
default-on requires the GOAT gate to pass:

1. Implement behind `feature_name = []` (opt-in).
2. Write a benchmark proving the gain (latency, quality, or security).
3. Run the GOAT gate (G1 correctness, G2 perf, G3 no-regression, G4
   alloc-free or equivalent) — `--release` mandatory at G2.
4. If all gates pass AND the gain is **modelless** → promote to `default`.
5. If the gain requires riir-train (training) → keep opt-in, note the
   dependency, do NOT promote.

**Promotion requires modelless gain** — a perf gain on a biased/incorrect
answer is NOT one; G1 must pass modellessly.

⛔ **A latency number without its BOX STATE is not a measurement.** Record
free RAM, commit-vs-limit (read AT launch), concurrent heavy jobs (ranked by
COMMIT), and on a laptop POWER SOURCE and POWER MODE (`pmset powermode` is a
three-state enum; sudo-free detector: `riir-reflex/scripts/bench_preflight.sh`,
quote its `PROVENANCE:` line) next to any latency figure from a shared box.

**Lossy-surface promotion rule:** a **lossy** surface (quantization,
compression, any bit-changing transform) gates on **deployed-path behavior —
per-family, conditional retention**, not bit-identity or aggregate
perplexity alone (aggregates can be flat while family behaviour flips).

**UQ-bearing primitive GOAT gate extension ("Report the Floor"):** any
primitive claiming a distribution, interval, quantile, coverage, confidence
or calibrated uncertainty MUST benchmark against the **conformal-naive floor**
(`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`, Plan 340, `m=1`,
plain split conformal) on CRPS / coverage / Winkler. Cannot beat the floor ⇒
GOAT FAILS. Grandfathered UQ primitives include the floor at next re-gate.

## Substrate-First Gate (MANDATORY before implementing)

Before implementing ANY new System impl, trait, perception/cognition/emotion
pipeline, state management, spatial query, or vocabulary type, run the
`substrate-first` skill: (1) **vocabulary translation** — grep 3+ name
variants; (2) **codebase grep** across `*.rs`, not just docs; (3)
**architectural rule check** — domain classification, two-brain model, sync
boundary, bridge pattern; (4) **consume vs build** — if substrate exists,
consume it; if not, file an issue in the right repo FIRST.

**Claim-lease check before implementing a FILED issue/plan (Issue 943):**
`git fetch origin && py ../katgpt-rs/scripts/claim_lease_check.py --item <NNN>`
— a foreign live `Claimed:` line means yield. To claim, append
`**Claimed:** <session>, <epoch>, expires <ISO-UTC>` under the Status line and
push BEFORE implementing (48 h expiry; the push is the arbiter; exit 1 = could
not read, never a silent pass). Lint first:
`py ../katgpt-rs/scripts/claim_lease_check.py --lint <file>` (malformed or
ambiguous lines blind the check for every lane). **Closing a lease FOLDS the
claim line into the `**Lease closed:**` row** (delete `**Claimed:**`, carry
session/epoch/expiry); `--lint` reds leftovers as STALE-CLAIM.

⛔ **Before recording a negative result, `git fetch` and check whether a
sibling session shipped the same primitive; if so, cross-run the fixtures
before writing "does not"** (Issue 825). Distrust a mechanism inferred from a
monotone sequence; repair beats delete when the loser is independent.

Research workflow (paper classification, 7-repo routing, fusion-first
distillation, novelty + GOAT gates, modelless-unblock protocol §3.5):
`.agents/skills/research/SKILL.md`.

> **Repo count:** the **product/distillation set is 7** — `katgpt-rs` (public) +
> `riir-ai`, `riir-chain`, `riir-neuron-db`, `riir-train`, `riir-game-sdk`,
> `riir-dapps` (private). That is NOT the repo total: the
> workspace is **32 repos**, all of which carry a root `BOUNDARY.md`
> (add `riir-mmorpg-examples`, `riir-refine`, `riir-viewbridge`, `riir-auth`,
> `katgpt-web`, `riir-dao`, `riir-deployer`, `riir-esp32`, `riir-llm`,
> `mmorpg-remake`, `mmorpg-remaster`, `riir-kat`, `riir-shader`,
> `riir-reflex`, `riir-infer`, `riir-reflexer`, `reflex-site`,
> `riir-instinct`, `riir-rethink`, `seal-std`, `riir-refab`, `jup-tool`,
> `seal-render`, `seal-game-editor`, `zed`).
>
> Read a count in prose as a claim, not a fact — a count that MATCHES is not a
> checksum over a set. Per-repo birth/registration notes and drift history: HISTORY.md.

## Numbering Discipline

Issue, plan, doc, benchmark, and research numbers are **monotonic and never
reused** — even after a file is removed. Before creating a numbered file, `ls`
the folder AND re-read `.highwater` at WRITE time, use `value + 1`, write the
new value back (`.issues/`, `.plans/`, `.docs/`, `.benchmarks/`,
`.research/`). Then fetch + run the allocation-time gate:

```bash
git fetch origin && py ../katgpt-rs/scripts/dual_allocation_gate.py
```

- Verdicts: **TWIN** (same stem both sides — your rebased work; fetch
  resolves) · **INDEPENDENT** (different stems on one number; names both
  adding commits) · **COUNTER** (both sides bumped `.highwater` past the merge
  base — the only witness for an allocate-and-close in one commit). Reach is
  limited to divergences THIS box participates in. Probe:
  `scripts/dual_allocation_fp_probe.py`.
- Two documents sharing a number: the one with the most inbound mentions keeps
  it — use `scripts/citation_weight.py <repo> <dir> <number>` (strict margin;
  UNRESOLVED printed separately). ⛔ Do not renumber on a margin it did not award.
- ⛔ CLOSING a holder does not retire its collision row; `removed_by_number()`
  recovers both sides from `git log -M --diff-filter=D --full-history`.
- Verdict: `scripts/numbering_gate.py` (docs gate), two regimes in
  `scripts/number_collisions_expected.txt`: ≥ `era_boundary = 700` a WALL by
  MEMBERSHIP with reasons; below it a RATCHET. Take SCOPE from
  `scripts/numbering_floors.txt`, never a tree walk. Cross-repo half:
  `scripts/numbering_drift_sweep.py` (`max_hist` ratchet; `min_numbers` floors
  the history walk; `min_bench`).
- **On-disk duplicate classifier** (`numbering_gate.classify_stems`): a
  ≥2-holder (number, directory) group is COLLISION (token-disjoint — walled at
  0 everywhere, unpinnable; a pinned COLLISION is a work item) /
  RENAME-RESIDUE / FAMILY-EXPLICIT (silent) / HAND-READ (ratcheted). The sweep
  syncs its own-repo row against the gate's pins (`classified_pin_sync`,
  exit 2 on divergence). Sees numbered `.md` documents only.
- Report-only: `scripts/highwater_contiguity_audit.py` (the counter is NOT an
  ownership witness).

## Branch

`develop` is the working branch. Don't create feature branches; commit
directly on `develop` per the global rule.

## Models
- riir-train/data/gemma-2-2b-it-f16.gguf — RETIRED locally 2026-10-07 (riir-train Issue 617);
  restore with `./scripts/data_backup.sh pull
  data/gemma-2-2b-it-f16.gguf` from the riir-train root. The katgpt-attn spike-census
  sidecars stay committed (the fixture gate reads sidecars, not the GGUF).
- riir-train/data/MiniCPM5-1B-F16.gguf — LOCAL-KEEP (pin, re-downloadable).
