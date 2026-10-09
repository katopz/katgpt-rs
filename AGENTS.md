# AGENTS.md — katgpt-rs

The global `~/.agents/` rules apply; this file documents repo-local context
that supplements them. History, resolved-issue records, gate narratives,
collision precedents: `HISTORY.md` (compacted 2026-10-06 — every `##` heading
kept; full pre-compaction text in git history). Removed issue files: git
history.

## Boundary contract — read `BOUNDARY.md` first

[`BOUNDARY.md`](BOUNDARY.md) is the authoritative contract: what this repo
**owns**, what it **does not own** (with the correct home for each), the
crate-granular **allowlist**, and the **drift ledger**. On any conflict with
prose, BOUNDARY.md wins.

- **Domain test:** is this a **modelless inference primitive** with no riir dep
  (this repo is upstream of everything)? NO → it belongs in another repo; file
  there.
- Read it before adding any dep, crate, module, System impl, or vocabulary type.
- Enforcement: `../riir-ai/scripts/ci_boundary_contract.sh` — undeclared
  cross-repo dep, drift row without its open issue, contract-vs-measured-graph
  drift. Run boundary checks VIA the `boundary-guard` skill, not ad-hoc greps.
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

**MANDATORY: exhaust modelless paths before deferring to riir-train.** Before
deferring ANY gate, mechanism, or plan task ("this needs training"), check the
three paths above (research skill §3.5). Systematic, characterizable biases
are modelless-correctable candidates — for a known, named bias, try a
deterministic reader-LoRA or freeze-state correction before concluding "needs
gradient descent." Canonical failure: AC-Prefix G1 (Plan 313) — the story and
the drift history live in HISTORY.md.

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

Every command above is narrow in at least one **independent** axis, and a green
result says nothing about what it compiled to nothing. The axes, one line each:

- **check vs clippy** — two `cargo refine` escape classes are rejected by
  clippy's typeck and accepted by `check` (E0689, E0631 in `redundant_closure`).
- **default vs `--all-features`** — non-default gated code compiles to nothing.
- **`-p <crate>` vs `--workspace`** — the ROOT crate's defaults can switch on a
  crate's non-default feature once selected; per-crate runs also silently
  *shrink* coverage.
- **no `--all-targets`** — skips tests/benches/examples, where gated code
  lives; and `--all-targets` still excludes doc-tests (only `cargo test
  --doc` reaches them).
- **dev vs `--release`** — `debug_assertions` is always ON in dev, so every
  `#[cfg(debug_assertions)]` item only compiles in the configuration where it
  works. **Neither profile is the safe default — the profile is part of the
  claim.**
- **compile vs EXECUTE** — every axis above is compilation; the scoped core
  (katgpt-rs + katgpt-core `--lib` at default features) is EXECUTED weekly
  (`test.yml` + `scripts/test_gate.sh`, schedule RE-ARMED 2026-09-23 — the
  repo is public, Actions minutes free; private siblings stay suspended). An
  uninvoked assertion is *unknown*, not passing; the other 477
  integration-test and 176 bench targets are executed by nothing automatic.
- **host triple vs wasm32, and x86_64** — see below; **a platform is part of
  the claim, exactly as the profile is.**

So before claiming a repo-wide green, run:

```bash
cargo clippy --workspace --all-targets --all-features --keep-going -- -D clippy::needless_range_loop -D clippy::map_clone -D clippy::iter_cloned_collect -D clippy::identity_op -D clippy::bool_comparison -D clippy::manual_is_multiple_of -D clippy::collapsible_if -D clippy::map_all_any_identity -D clippy::unnecessary_cast -D clippy::manual_repeat_n -D clippy::question_mark -D clippy::empty_line_after_outer_attr -D clippy::unusual_byte_groupings -D unused_mut -D unused_parens
```

The `-D` list (Issue 701 R3b, 2026-09-03) is the mechanical lints whose
all-features warning surface was healed to ZERO residual — a lint with
residual > 0 must NOT be added to it. `--keep-going` is not optional: without
it the run stops at the first failing target and under-reports (it was red on
`develop` for days, twice, while narrower gates were green). Don't run it by
hand — `scripts/full_gate.sh` is the assertion (it refuses to report a pass
off macOS and checks that this document still quotes the command it runs).
`--all-features` is NOT a supported TEST configuration (per-feature RNG/GOAT
calibrations); the full test surface is priced, not run
(`.benchmarks/701_full_workspace_execution_pricing.md`).

⛔ **On a non-macOS workstation, run it as
`scripts/full_gate.sh --allow-partial-platform`** — a workstation SUBSET
verdict that never prints the `✓ full gate PASSED` line (reserved for a macOS
run with every target installed). Measured 2026-09-15: `develop` carried 24
`error[E0560]` + 4 `-D`-listed lint errors for ten hours while
`test_gate.sh` and `wasm32_gate` were both green — the per-crate gate that
landed it was right for what it changed; nothing read what it changed for
everyone else (Issue 803 closed the silent-final-line defect).

**wasm32 is a second platform axis, not a variation on the first.** Nothing in
this repo compiled it until 2026-09-07 — the only script naming the triple was
`scripts/build-moka-wasm.sh`, a deploy build. `full_gate.sh` layer 2b closes
it: derived `-p` list **including the root package** (clippy lints workspace
path deps — that is how an orphaned doc block on a crate with no wasm32 code
surfaced), both simd128 arms (measured: the OFF arm clean, the ON arm had 14
findings, 11 `unsafe_op_in_unsafe_fn` on edition 2024), two wasm32 GOAT
targets by name, and the residue pinned by membership. `--all-targets` dies on
dev-deps that do not resolve for wasm32 — extra coverage goes in as named
targets. Push lane: `.github/workflows/wasm32_gate.yml` (main-only,
`--wasm32-only`; dispatch it manually after a run of develop work, since no
push lane covers develop pushes).

**x86_64 is a THIRD one — Layer 2c (Issue 819, the x86_64 lint lane).** Every
lane that LINTS compiles the x86_64 arms to nothing; the execution matrix
below EXECUTES them and reads no warnings — 30 findings on its first cell,
every one `unsafe_op_in_unsafe_fn`, every one in `dash_attn/channel_aware.rs`,
against 0 on the avx2-off arm. ⛔ The finding is the **sibling**: two
transcriptions of one kernel and only the aarch64 arm had the `unsafe { }`
block — a repair applied to the arm somebody can see is a measurement of
which arms are visible. The lane is `--all-features` (Layer 3's coverage on
the x86_64 arch); the TRIPLE is the design decision, printed on the verdict
line. **The inverse holds too:** running ON macOS silently drops every
`not(target_os = "macos")` backend. Typecheck that half from the M3
(`--canary` is not optional — it requires E0425 from a planted undefined
call, else "Finished" is indistinguishable from compiled-to-nothing):

```bash
scripts/check_platform_gated_modules.sh --canary ../riir-train riir-train-gpu \
    crates/riir-train-gpu/src/numeric_drift_tap.rs numeric_drift_cuda
```

**The profile axis is feature-shaped too (Layer 6b, Issue 758).** Layer 6 runs
`--all-features` — which SUPPLIES `alloc_tracking` — so the (release ×
default-features) cell was asserted by nothing until slice_tca's module-level
`use crate::alloc` fell through it (E0432). The matrix: (dev, default)
test_gate · (dev, all) Layer 3 · (release, all) Layer 6 · (release, default)
Layer 6b. Alloc-gated tests carry
`#[cfg(any(debug_assertions, feature = "alloc_tracking"))]` so they RUN in the
release+feature configuration and compile away at release-default.

Trigger health: PUSH triggers are MAIN-ONLY + dispatch-only since 2026-09-09
(owner call: no CI on `develop` pushes); the weekly `schedule:` blocks were
RE-ARMED 2026-09-23 for THIS repo alone. `scripts/ci_gate_coverage.py`
reports which declared triggers can actually fire, per workflow, per repo; its
standing finding is that the lane it produces is ZERO, not reduced (12 of 16
repos: five without `origin/main` at all, six with no `.github/workflows/`
there) — until then those gates run only when a human clicks them.

### The x86_64 half of the EXECUTE row — `scripts/x86_64_execution_matrix.sh`

Every axis above is a COMPILE axis; the matrix EXECUTES. Its first two cells
caught **15** latent AVX2-transcription defects; the next seven caught two
more, a router defect red on every non-macOS platform, and a release-profile
benchmark LLVM constant-folded away so it reported 0 ns/call
([Bench 806](.benchmarks/806_x86_64_execution_matrix.md)).

```bash
scripts/x86_64_execution_matrix.sh              # the matrix
scripts/x86_64_execution_matrix.sh --libs-only  # skip the root integration cell
scripts/x86_64_execution_matrix.sh --canary     # prove the floors fire
X86_MATRIX_DIR=/f/scratch scripts/x86_64_execution_matrix.sh
```

- **Workstation verdict**, no CI lane. **REFUSES off x86_64** — every arm it
  exists for compiles to nothing there, and a green run would be a green ZERO
  wearing a matrix.
- `RUSTFLAGS="-C target-feature=+avx2"` is **not a tuning knob** — an arm gated
  on the compile-time feature compiles to nothing without it, and the run then
  exercises the scalar fallback and proves nothing (`channel_aware.rs` is
  exactly that shape).
- ⚠ `CMAKE_BUILD_PARALLEL_LEVEL=1` on Windows is a LOAD-ROBUSTNESS cap, not the
  fix: both C1001/D804 failures happened only under whole-box concurrent
  compiler load; quiet-box passes at every parallelism, timings
  indistinguishable. **The actual remedy is to run the matrix ALONE.**
- Population **derived** (any package owning a tracked `*.rs` mentioning
  `target_arch = "x86_64"`), so a new such crate joins by EXISTING; a package
  reaching the kernels only through a dep joins by a pinned row.
- **Every failure is RE-RUN ALONE, then adjudicated by MEMBERSHIP**
  (`scripts/x86_64_matrix_expected.txt`, a reason per row, refused without
  one). The bucket is **`PASSED-ALONE`, never `TRANSIENT`** — three classes
  produce "failed in the cell, passed alone" and they need opposite responses:
  a load-sensitive perf BAR; a shared-fixed-path CONCURRENCY defect (passes
  alone BY CONSTRUCTION); an unseeded-RNG COIN FLIP. Arch-conditional dual
  pins are the resolution for calibration-failed bars (measured T698), never
  re-typed pins. The membership set is EMPTY: every confirmed failure is
  unexpected again.
- Floors in `scripts/x86_64_matrix_floors.txt` — `min_passed` per package plus
  the integration cell's **two** (targets AND assertions). `--canary` runs an
  impossible floor through the same comparison path and requires the failure.
- ⚠ It does NOT cover the macOS device backends, wasm32, `--all-features` for
  the integration targets, and it does NOT LINT — Layer 2c is the lint half.

## Docs gate + drift sweeps

**Repo names in instruments are CONTRACT names** (`scripts/repo_alias.py`).
The ten `derive_repos`/`derive_population` predicates return the live
workspace's directory names passed through a machine-local codec: an optional,
gitignored `scripts/repo_alias.local.txt` (rows `on-disk=contract`) translates
a box's on-disk sibling names into the contract spellings every tracked pin is
keyed on; absent file → identity mapping (CI and fresh clones untouched); the
count of active mappings is disclosed on stderr, never the names.

⛔ **The open seam is ONE** — `sweep_population.open_repo(name, workspace)` /
`repo_alias.real(repo)` (Issue 842): a contract spelling is a directory that
does not exist on an alias box, and every walk on it returns 0 — measured, 17
of 19 sweeps were opening wrong directories and seven red walk floors against
TRUE pins measured against the WRONG DIRECTORY. The repair was one seam, not
19 patches. ⛔ And it has a PROSE half, `issue_citation_gate.spelling_aliases`
(Issue 846): on-disk spellings qualify citations (LENIENCY ONLY — a spelling
can clear a row, never accuse one), with the `_NAME` boundary regex so
`seal-remake` does not match inside `seal-remake-unity`. In code, never in the
box-local alias file (verdicts must not go machine-local).

`scripts/docs_gate.sh` runs the manifest/doc/skill drift assertions and
**prints its own timing**. Cite the CPU figure **with the load class it was
measured under** — measured 2.7–3.4× CPU inflation under sustained
multi-tenant load, and a run this file calls loaded can measure LESS CPU than
a quiet one, so the load class is a disclosure, not a correction factor. Only
compare CPU within a fixed CHECKS set (the set moved four times in one day,
which is the argument for writing the CHECKS count beside the number instead
of the number alone). The per-check `⏱` line shows which check is BLOCKING,
never that a check got slower. ⛔ On Windows, `times` accounts MSYS children
and ~nothing for native ones — the gate CALIBRATES (burns a known 0.25s in a
child) and prints `CPU SUPPRESSED` rather than a plausible number (Issue 792).
↔ The GENERAL rule for any perf number lives at §Feature Flag Discipline's G2
bullet; this paragraph is the INSTANCE for docs-gate CPU seconds — nothing
asserts these two copies agree, on purpose.

`.github/workflows/docs_gate.yml` runs it per-push on **`main` only** —
develop pushes do not fire it, so run `./scripts/docs_gate.sh` locally for
develop work. One line per check:

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
| `numbering_gate.py` | a number allocated twice, or a stale/malformed `.highwater` — walled by membership above the era boundary, ratcheted below it (Issues 724, 725, 795) |
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
| `console_encoding_gate.py` | a tracked `scripts/*.py` that prints a non-ASCII glyph and defends neither stream — on a non-UTF-8 console it dies with no verdict; shared defence `scripts/console_safe.py` (Issue 804) |
| `global_rng_gate.py` | an unseeded-global draw with no pin row — found by executing one commit twice; membership + reason per row, both directions (Issue 809) |
| `algebraic_op_ban_gate.py` | an `algebraic_div`/`algebraic_rem` CODE occurrence in tracked `*.rs` — div/rem banned outright; ceiling 0 (Issue 871) |
| `shared_temp_path_gate.py` | a test writing to a FIXED `env::temp_dir()` path — safe against sibling tests in one binary, truncated by any concurrent PROCESS running the same test; passes alone BY CONSTRUCTION, one TRANSIENT misdiagnosis class (Issue 832) |
| `cross_repo_path_dep_gate.py` | a `path = "../X"` dependency on a repo that is NEITHER on disk NOR in `repo_set.txt` — cargo resolves path deps even when `optional = true` (Issue 835) |
| `repo_registration_gate.py` | a per-repo pin file with no row for a registered, on-disk repo — registering a repo is a 22-file operation and only repo_set.txt is gated, so the other 21 are discovered one red sweep at a time; measured the day riir-llm joined, 19 of 21 sweeps red and four live findings sitting behind them (Issue 837) |
| `cross_module_attr_gate.py` | a tracked scripts/*.py naming an attribute a sibling module does not define — Python has no link step, so the reference resolves at CALL time and a rename is an AttributeError the next EXECUTION finds; when the importer is a gate the failure is NO VERDICT rather than a wrong one. Measured: a commit privatised is_checkout and deleted worktree_fixture, updating the seven in-module callers and neither of the two external ones — both of them CHECKS in this array, develop red for 6h40m. --prove-fires is two-sided, 0 findings at the parent and exactly those 4 at the commit; the runtime half of the coverage stays arm_reach_gate's 157.6s workstation verdict (Issue 848) |
| `import_health_gate.py` | a tracked scripts/*.py that does not IMPORT — the EXECUTION half of Issue 848, where the static half (cross_module_attr_gate) cannot reach: a circular import, a missing third-party dependency, a raise in top-level code. The instrument that already executes every module is arm_reach_gate's BASELINE-CRASH, kept out of this budget at 157.6s; this is its cheap half, one child, 0.11s of import. A per-module subprocess was measured too and is not worth 3x (10.28s vs 7.52s, identical verdicts). ⛔ The affordability measurement found its own blocker — 6.238s of 6.34s was ONE module whose whole body was top-level, guarded in the same change. MISSING-DEP is its own bucket and is never flagged: pinning it would red the gate on a box that HAS the package (Issue 848 T3) |
| `shipped_target_feature_gate.py` | a SHIPPED path selecting its fast arm on a COMPILE-time target_feature — that predicate is OFF by default on x86_64, so the arm compiles to NOTHING on every ordinary build and the dispatcher silently runs its fallback. AGENTS.md documented this shape only for GATES, where the cost is an unproven claim; on a shipped path the cost is latency on every call, measured at 4.4-5.6x in a DEFAULT-ON feature and 2.4-2.5x on bf16 RNE narrowing. Three exclusions a naive grep gets wrong — wasm32/simd128 (no runtime detection there, so a compile-time gate is the only option), NEON (implied by the arch), and the runtime probe's own body, which is the same attribute doing the opposite job. The key is line-free and resolved by brace counting, the two cases needing opposite lookups — an attribute inside a body belongs to its enclosing fn, one ON an item to the NEXT. Membership + a reason per row, both directions, floors on the walk and on the attribute parse; --prove-fires is two-sided (Issue 847 T3) |
| `timed_region_guard_gate.py` | a latency ceiling with no loud-zero defence — rustc + fat LTO deletes a timed loop whose result is dead, so the bar is satisfied by **absent work** and passes with MAXIMUM margin. This is the `#![cfg]` green-zero rule one layer down, and worse: the assertion RUNS, so the output is a plausible number rather than a zero count and no count floor can see it. ⛔ Not reasoned about — **executed**: all 34 asserting regions at n ≥ 1000 were run and **7 were satisfied by absent work** (20.6%), two of them GOAT gates, one printing `Speedup: 8657.9×`, one printing a well-formed `0.00x` because only the NUMERATOR vanished. ⚠ The static predicate Issue 855 T4 proposed is REFUTED by that run — `let _ =` vanished **3 of 15** against **4 of 19** for the rest, i.e. the base rate wearing a grep; the column that separates is `black_box`, **7 of 23** without it against **0 of 11** with. So the gate does not predict which region is broken; it gates the decidable thing — *is there a loud-zero defence at all*. ⛔ Two tiers, and the split is the honest part: a LITERAL loop bound is the population that was READ end to end (membership wall, one MEASURED number per row), while a bound needing one hop of resolution is real and **unread** (a ratchet on the derivative — pinning an unread bucket by name is Issue 785's forbidden shape). One hop is not optional: T1's own two founding specimens are `let n = 100_000; for _ in 0..n`, so a literal-only predicate would have shipped the class it was written for. ⚠ It does NOT claim a pinned region is safe — `black_box` is the weakest of three defences (result, **arguments**, **receiver**) and two arms vanished carrying one (Issue 855 T4) |
| `check_validation_gate.py` | a CHECK in this array whose own arithmetic no arm asserts — including one whose arm is flag-gated and so never runs (Issue 789) |
| `docs_gate_checks_sync.py` | this CHECKS array vs the AGENTS.md table documenting it — membership both ways + quantity words (Issue 750) |
| `skill_size_gate.py` | a skill SKILL.md over the 80KB ceiling — the THIRD 100KB-regrowth class (doc-sync 09-05/09-11/09-21, boundary-guard 09-08/09-11/09-21: the one-line convention held, the ~8 rows/day cadence didn't) — the gate forces each file's documented prune-to-15 maintenance rule; recovery via git log -p |

**Partial-clone boxes (Issue 765):** the three population checks
(`skill_repo_set_gate`, `population_sync_gate`, `issue_citation_gate`)
hard-red on a box carrying a subset of the workspace. A known-partial box
(the 4090: 14 of 20) exports `DOCS_GATE_PARTIAL_CLONE=1` for a loud
instrument-alive DEFERRAL on the population axis — explicit opt-in, NEVER
auto-detected (a forgotten removal is set-identical to a partial clone from
the walk alone). ⛔ The Issue-842 alias codec RETIRED the marker on boxes that
map their spellings (measured on the 4090: with the alias rows the marker reds
in exactly the stale-acknowledgement direction) — drop the env var, don't
widen it.

**Known-extra boxes (Issue 815):** present-and-unregistered repos the contract
genuinely does not claim get `DOCS_GATE_KNOWN_EXTRA=names` — NAMES, never
`=1` (an unnamed extra still reds beside a named one); reds in BOTH
directions (a name gone from the box, or since registered, is a STALE
acknowledgement and fails).

**Every sweep answers the partial-clone question the same way, once (Issue
793): `scripts/sweep_population.py`.** Seven sweeps carried a copy-pasted
"pinned but ABSENT" loop and hard-red on a known 16-of-20 box with every
content assertion green — a sweep that always reds is a sweep nobody runs.
Three verdicts, never interchangeable: **UNREGISTERED** (on disk, absent from
`repo_set.txt` — reds in every posture), **UNSEEN** (absent, no marker — never
a pass), **DEFERRED** (the same set with the marker; rides the FINAL line in
BOTH directions). ⛔ "Every sweep" typed as a NUMBER went stale within two
hours, twice — the wall is the MECHANISMS registry (worktree section), and a
subset sweep has TWO populations (hand `population_verdict` the contract
walk, never the subset — measured 16 phantom absences). Registration scope:
`scripts/repo_registration_scope.txt` declares per-file SUBSET scope for the
registration pins (a forgotten declaration reds, a forgotten EVERY would
green).

**The sweep family** — workstation-only cross-repo verdict halves, each with
two floors (a ceiling is green over whatever the instrument can SEE, so the
finding count needs the population floor beneath it; where a sweep re-states a
quantity its per-push gate owns, it ASSERTS the two agree rather than trusting
them): `docs_drift_sweep.py`, `numbering_drift_sweep.py`,
`required_features_drift_sweep.py`, `percentile_drift_sweep.py`,
`cfg_gated_drift_sweep.py`, `cfg_row_implication_drift_sweep.py`,
`trap_sentinel_drift_sweep.py`, `citation_drift_sweep.py`,
`restatement_drift_sweep.py`, `markdown_fence_drift_sweep.py`,
`platform_dead_code_drift_sweep.py`, `subprocess_encoding_drift_sweep.py`,
`locale_io_drift_sweep.py`, `console_encoding_drift_sweep.py`,
`timed_region_drift_sweep.py`, `shared_temp_path_drift_sweep.py`,
`pipefail_discard_drift_sweep.py`, `toolchain_override_drift_sweep.py`,
`orphaned_attr_drift_sweep.py`, `wasm32_surface_drift_sweep.py`,
`len_derived_drift_sweep.py`, `instrument_reachability_drift_sweep.py` — plus
the report halves `pipefail_discard_audit.py` and `toolchain_override_audit.py`,
`sibling_docs_drift.yml` (reusable workflow, one caller), and
`ci_gate_coverage.py` (report, always exit 0 — deliberately NOT in the CHECKS
set: CI's single checkout would derive an empty population and print a
confident green over zero repos). Every tracked `*_drift_sweep.py` is wired to
the family-wide mechanisms, ceilings pinned under `scripts/*_floors.txt`;
**take the family size and the wiring verdict from
`scripts/sweep_advisory_membership_gate.py`'s PASS line, never from a count
here** (this paragraph's own counts went stale within hours, twice).

⛔ **Before fixing a rule-in-one-instrument class, grep the whole family and
land the repair as one shared mechanism** — the recorded failure mode,
measured ten+ times (Issues 777, 778, 782, 783, 789, 793, 797, 820, 821,
822). ⛔ **A cross-repo repair is not landed until it is COMMITTED in the
sibling, and a record HERE claiming one must CITE THE SIBLING COMMIT** (Issue
798): measured, two tracked files recorded sibling repairs as landed —
`toolchain_override_drift_floors.txt` ("all five markers are in") and
`pipefail_discard_expected.txt` (a row DROPPED for a "fix") — two of the five
markers existed, the tail did not, and both sweeps were RED for the whole
interval. A SHA buys a claim one `git cat-file -e` can check; dropping a pin
row for an absent fix is the unrecoverable direction.

**Citation error rates** — `citation_drift_sweep.py` prints its own error
rates next to its finding count, plural, because a SAMPLE rate does not
transfer to rows it never sampled: the CROSS rows split into the pre-752
corpus (**7/43 = 16%** FP, stratified manual read) and the **45**
owner-consistency rows (**1/45**, full census). Neither number is quotable
without the other, nor without the walk and population that produced them;
the IN-LOCAL-RANGE bucket is UNDECIDED and never folded into either
neighbour. The class that DEFLATED the count by ~15%: qualification asked
"is a repo named?" and never "does that repo own the number?" — the
attribution followed the code while the number followed the document. ⛔ A
census is exhaustive over ROWS, not over the ORACLE it checks rows against —
never quote an error rate without naming the instrument the sample was
adjudicated against (measured: a census's "wrong address" was correct; the
oracle was blind to heading-only allocations).

**The two-lane gap** (Issue 921 Arm B): a repo whose kind dir carries
`.highwater_local` declares its local lane's top, and citations in
`counter < n < smallest git-log file ADD above it` classify CROSS/ORPHAN
instead of IN-LOCAL-RANGE — the floor derives from git-log ADDITIONS only
(never the worktree walk, never heading allocations, whose additions are
order-dependent), prints on the per-repo line, `.highwater_local` rides the
sweep's dirty/upstream advisory patterns, and `citation_drift_sweep.py
--prove-fires` replays the founding specimen two-sided at the riir-infer
fixtures (a local clone, not an archive extraction — the floor leg is a
git-log leg). Repos without the file are byte-identical in behavior.

**The heading oracle** (Issues 781, 823, 828): `heading_allocated()` is not a
complete record — records headed in styles the pattern cannot spell are
unread, and the class was anchored three times (POSITION, then the blindness
METER, then the DELIMITER SET — the workspace's most common title delimiter,
the em dash, was in neither set; most of the unread were this repo's own
house style). The residual `resolved —`/`follow-up` family has NO sound
discriminator (arm 2 pins the negative), and the unsound widening was REFUTED
by pricing it: of the unread records, only ~2 name a number no other oracle
knows (~99% redundant). All quantities are DERIVED per run — take
`heading_unread=a/b`, the `heading oracle COST` line and `novel=` from the
sweep's own lines, never from a sentence here: every hand-typed figure in
that arc went stale within hours. ⛔ A document that DISCUSSES a
misattribution has to reproduce it — the repair is to name the true owner
inside the citation's own window, not a pin. ⛔ Do not re-pin
`max_in_local_range` for a heading-blind row — a repo at or near `b/b` unread
cannot have its IN-LOCAL-RANGE count trusted.

⛔ **The pipefail class** (one shared mechanism): a `var="$(pipeline)"`
assignment under `set -euo pipefail` is killed by a legitimately-empty grep
AFTER the measured work ran and BEFORE the result was written — the
`perf_rematch.sh` incident that lost five benchmark cells.
`scripts/pipefail_discard_audit.py` + `scripts/pipefail_discard_drift_sweep.py`.
Two bash laws are MEASURED: `local x="$(fails)"` does not kill (local masks
the status), and an interior `|| true` guards a paren-group. ⛔ The one-line
"add `|| true`" recipe is INSUFFICIENT wherever the captured value is then
TESTED: an empty capture matches no pattern and `|| true` alone converts a
silent death into a silent PASS on the exact regression the check exists to
catch (`money_format_gate.sh` needed `|| true` AND an emptiness failure; a
leakage-audit guard's verbatim recipe was strictly WORSE than the defect —
the security gate PASSED an unverified run). Census: 51 findings → all
DELIBERATE (the 4 `proof_gate.sh` `grep -c` tripwires) or INERT BY
CONSTRUCTION; the 37 live kill-shapes fixed across ten repos, including the
22-site riir-ai `ci_feature_guard.sh` layer-summary cluster and 6 dapps
`setup.sh` rows.

## cfg-gated targets — the green-zero rule

A test file opening with `#![cfg(feature = "x")]` compiles to an **empty
binary** when x is off; cargo prints `ok. 0 passed` and **exits 0** —
byte-for-byte a real pass. `required-features` protects the **reader**; the
`#![cfg]` protects the **count**; a *default-on* gated target still runs on a
plain `cargo test`, a *default-off* one reports a green zero every time
anyone names it — read the severity split, never the pooled total.

⛔ **There are TWO spellings of that zero (Issue 856).** A file whose entire
body is `#[cfg(feature = "x")] mod tests { … }` produces the byte-identical
outcome, and the audit's single inner-attribute regex never saw it —
measured: 26 such targets here, 7 named `*_goat`, hiding 175 assertions. The
predicate is now **the gated items are the WHOLE body** (a file with one live
ungated `#[test]` is not zeroed); a **run** of gated modules is gated by
`any(...)`, not `all(...)`; a top-level `use` never makes a target non-empty.

**Two traps in the profile dimension (Issue 741).** A file may carry more
than one whole-file `#![cfg]`, and rustc ANDs them — read all of them. And
gating a MEASUREMENT on `debug_assertions` makes it impossible in the
configuration that ships: ask of any such gate whether the thing behind it is
a **capability** (→ give it a feature, `any(debug_assertions, feature =
"x")`, and gate the machinery on x too) or genuinely a **profile property**
(an assertion about `debug_assert!`). Read the auditor's `unfixable` vs
`escapable` split, never the pooled DEBUG-only count.

```bash
scripts/cfg_gated_target_audit.py            # all contract repos (derived)
scripts/cfg_gated_target_audit.py ../riir-ai # or one, by path
```

- A **report, not a gate** (exit 0): `cfg` on `target_os`/`miri` and an
  `any(...)` of features genuinely cannot be expressed as `required-features`.
- `scripts/suite_membership_audit.py` answers the next axis down: which
  `[[test]]` targets no script/workflow names — run it when landing a new
  gate; if nothing names it, add a suite row or record why not.
- **Arming a target can RED a binary-counting floor**: an empty gated binary
  prints `test result: ok. 0 passed` and COUNTS as one. Repair with a
  **passed-test floor**, not a re-pin. **Run the armed gates with
  `--release`** — a latency gate in a debug build measures an unoptimised
  binary.
- Verdict half: `scripts/cfg_gated_floor_gate.py` (pins in
  `scripts/cfg_gated_floors.txt`; `max_load_bearing = 0` earns its keep; some
  pins are FLOORS — a ceiling cannot fail once the instrument goes blind;
  `scripts/all_ignored_load_bearing.txt` pins the ALL-IGNORED set by
  MEMBERSHIP from `all_ignored_target_audit.py` — a set is gateable where its
  cardinality is not).

## A `required-features` row can EXIST and be WRONG — `scripts/required_features_build_audit.py`

Every audit treats a target as protected once it **has** a row. A row that
exists and is wrong is strictly worse than a missing one: `cargo test
--workspace` silently **skips** the target, `--all-features` **builds** it —
and every audit counts it as protected. The row is wrong relative to what the
file *imports*, and imports resolve through cfg-gated re-exports that defeat
grep — ask the compiler, once per target:

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
- **Neither an error nor an artifact = UNSEEN, never BUILDS** — silence is not
  evidence; UNSEEN is never folded into the pass column (measured: "will
  clear after the FAILs" was TESTED and FALSE, and that is how a library
  break was found).
- **Read the USE SITES, not the error:** widen the ROW when the body needs
  the feature unconditionally; narrow the cfg when the use site is already
  gated (both directions measured).
- The free static verdict is correct AND insufficient: `dep/feat` rows are
  valid cargo; only the compiler distinguishes "names a feature that exists"
  from "names the feature that gates the module".
- Push half (MAIN-ONLY): `.github/workflows/required_features_touched.yml`
  checks the rows a main push could have broken (a changed file that IS a
  row's target source; a changed `Cargo.toml` selects rows whose tuple
  differs base-vs-head; `--max-rows` REFUSES rather than truncates).

## A reported "p99" is often the MAX — `scripts/percentile_index_audit.py`

`sorted[(n as f64 * 0.99) as usize]` and `sorted[n * 99 / 100]` both land on
`n - 1` — the **maximum** — for every `n <= 1/(1-p)`: n ≤ 100 at p99, n ≤ 20
at p95, n ≤ 1000 at p999. Below that boundary the site reports one
observation under a percentile's name; a `.min(len - 1)` clamp prevents a
panic, not a wrong statistic. The quantity to print is **tail support** =
`n - idx`: 1 at n=100, 10 at n=1000 — anything under 10 is weak. Assert
direction matters (a `>= floor` bar goes false-GREEN, not false-RED).

```bash
scripts/percentile_index_audit.py             # all contract repos (derived)
scripts/percentile_index_audit.py ../riir-ai  # or one, by path
```

- A **report, not a gate** (exit 0) — half the sites take their sample count
  from a runtime length no static pass can reach. **UNRESOLVED is not
  "clean"** — it is "needs a per-site read"; the audit prints site rows only
  for the four severe classes, so UNRESOLVED appears in the tally and nowhere
  else. `scripts/list_unresolved_percentile_sites.py <repo>` dumps those rows.
- Verdict half: `scripts/percentile_floor_gate.py` (pins in
  `scripts/percentile_floors.txt`; `min_sites_scanned` is a FLOOR — a
  tokenizer regression takes the population to ~0 and every ceiling passes).
- **The POPULATION is what git TRACKS — `scripts/tracked_walk.py`, one copy
  (Issue 777).** A filesystem walk behind a hand-typed skip set is not the
  same set: measured, a gitignored nested repo's 1404 `.rs` were credited to
  the outer repo — a correctly-shaped defect at the wrong address — and two
  `percentile_drift_floors.txt` rows were unsatisfiable by any tracked walk of
  their repo. Tracked-only had landed twice before and not generalised; now
  ONE file with its own 8-arm self-test; consumers:
  `platform_dead_code_audit.py`, `percentile_index_audit.py`,
  `len_derived_binding_audit.py`, `orphaned_attr_gate.py`. A tree with no
  `.git` falls back to the walk.
- ⛔ **`.git` answers TWO questions and each spelling is wrong for the other**
  (Issues 835, 836): *is this a canonical REPO?* → `.is_dir()` (a worktree has
  a `.git` FILE; counting one attributes its manifests to a repo that does
  not exist); *can I run git rooted HERE?* → `.exists()` (delegate to
  `worktree_state.is_checkout` — the silent direction returned "nothing to
  report" in worktrees). Never write a fresh contract-repo walk — delegate to
  `skill_repo_set_gate.derive_repos`; `population_sync_gate` exists to catch
  two predicates disagreeing. ⛔ A name match that never fires is the same
  silence one layer up: a guard with no else skipped its whole adjudication on
  a differently-named checkout, and a glob fallback reported untracked scratch
  (`✗ UNDEFENDED zz_scratch_probe.py`) as findings — measured before
  repairing, and the repair went in the same change.

## A Lean theorem can RESTATE its own definition — `scripts/restatement_theorem_audit.py`

`theorem x_eq_sum : x = magic + version + …` where the RHS **is** the `def`
body: `decide` closes it whatever the constants hold — green on every
transcription typo it was written to catch, while `lake build` and the proof
gate all report a theorem that proves something. Four shipped in
riir-neuron-db for months (Issue 617, removed `24957a2`).

```bash
scripts/restatement_theorem_audit.py               # all repos with .proofs (derived)
scripts/restatement_theorem_audit.py ../riir-chain # or one, by path
scripts/restatement_theorem_audit.py -v            # every row, not just findings
```

- A **report, not a gate** (exit 0). The criterion is symbolic equality over
  **leaf** constants: unfold composite nullary `def`s, keep numeral-bodied
  leaves as symbols — the leaf boundary IS the classifier.
- **CROSS-DEF is not a finding** — it pins two independently maintained
  definitions, and a perturbation arm proves it reds. **UNRESOLVED is not
  clean**, and HYPOTHETICAL (binders) is split out, never pooled.
- Validated against a tree whose answer was known independently; the oracle
  then found the instrument's own scoping defect (per module + imports now).
- Verdict half: `scripts/restatement_drift_sweep.py` (pinned in
  `scripts/restatement_drift_floors.txt`). **Two floors** — `min_lean_files`
  catches a WALK regression, `min_theorems` a PARSE one, and only the second
  moves when a tokenizer breaks on an unchanged tree. `--prove-fires` plants
  a restatement into a COPY of each repo so the ceiling is never a pin nobody
  has watched fail.

## A ratio of two SEQUENTIALLY-timed arms measures the BOX — `scripts/sequential_ab_timing_audit.py`

Two sequential arms of the **same** work measured **+5.2% and +21.7%** thirty
seconds apart on a loaded box (Issue 723 T5) — a 10% bar is a measurement of
the scheduler. `tests/common/ab_timing.rs` is the treatment: interleaved
(a-chunk, b-chunk) pairs, median across pairs, loud zero when the optimiser
deletes an arm (`best_of_us` the absolute-budget form; `best_of_arms` the
N-arm round-robin minimum).

```bash
scripts/sequential_ab_timing_audit.py            # this repo
scripts/sequential_ab_timing_audit.py ../riir-ai # or one, by path
scripts/sequential_ab_timing_audit.py -v         # every row, not just findings
```

- A **report, exit 0** — except a blindness floor or a failing self-test,
  which exit **2**; the self-test runs on every invocation.
- Buckets, never pooled: **ADOPTED** (names the shared harness — the harness
  module certifying ITSELF was the count defect) · **HAND-ROLLED**
  (duplicates it — a DRY finding, never folded into ADOPTED) · **SEQUENTIAL**
  (two timing-derived identifiers in ONE comparison — a ratio is one surface
  form; a bare `assert!(a < b)` is the same class) · **UNRESOLVED**, which is
  **not clean**. Resolvers: `provenance_hits` (timing-derived BY VALUE,
  transitively), the RELATIVE DIFFERENCE `(a - b) / c`, COUNTY (a count
  denominator is a rate — but comparing two rates is still comparing two
  arms, and COUNTY must not filter the comparison shape).
- ⛔ **No verdict half, deliberately** — migration is a per-target read on
  FOUR axes, none statically decidable: the a/b ORIENTATION (inverting a
  throughput claim inverts the bar silently), the CLAIM DIRECTION (a win with
  no slack must be precise; an overhead ceiling with deliberate room gains
  almost nothing), the chunk size, `black_box` at both ends. Measured: the
  flagged GOAT bars were RUN and both had head-room a loaded box cannot cross
  — reading the bar alone is the mistake.
- ⛔ **The executed answer beats the static proxies** (Issue 833 T2): of 38
  `SEQUENTIAL [GATES]` targets RUN in release at their required-features, **3
  are candidates** — the backlog is the count of *verdicts the box can flip*,
  strictly smaller than any grep. Thresholds: `< 6` pts flakes idle · `6–22`
  under load · `≥ 22` out of reach (±21.7% and ±6% are the two measured
  numbers).
- Two classes found by executing: **Issue 855** (a latency ceiling satisfied
  by a loop the optimiser DELETED — needs the loud-zero defence, not a lower
  bar) and **Issue 856** (the `#[cfg] mod tests` zero above).
- STATED blind spots remain: a ratio through a helper; a subtraction never
  divided AND never compared; orientation.

## A gate that ABORTS reports exit 0 — `scripts/trap_exit_launder_audit.py`

Every script above is a shell gate with `set -euo pipefail` and a cleanup
trap. On **macOS `/bin/bash` 3.2.57 — and only there**: when bash aborts on
an **unbound expansion** or an **`eval` syntax error**, it enters the EXIT
trap with `$?` **already 0** — so an EXIT trap whose last command succeeds
makes the abort exit **0**: a confident pass over work that never ran,
everything after the abort silently skipped.

⛔ **This bites the macOS CI lane, not just workstations** (measured on both
sides): GitHub's `macos-26-arm64` ships bash **3.2.57 ONLY**, so on
`full_gate.yml` the sentinel is load-bearing IN CI; `ubuntu-latest` (bash 5)
is immune. **Keep the sentinel regardless** — it costs nothing on 5.x and
catches every other premature death (SIGTERM, a `set -e` trip, an editing
slip) on every shell.

`trap 'rc=$?; cleanup; exit $rc' EXIT` does **not** repair it — the rc it
saves is itself 0. Only a **completion sentinel** does: a flag set on the
script's own last line, checked by the handler, forcing exit 1 when the run
is INCOMPLETE *and* claiming success. `scripts/full_gate.sh` and
`scripts/proof_negative_test.sh` carry it (Issue 734).

```bash
scripts/trap_exit_launder_audit.py            # population + verdict, all repos
scripts/trap_exit_launder_audit.py ../riir-ai # or one, by path
scripts/trap_sentinel_drift_sweep.py          # the verdict, every repo, pinned
scripts/trap_launder_premise_matrix.py        # the PREMISE, 11 interpreters
```

- Three halves, different questions: the audit derives the **population** and
  classifies it (report, exit 0); `scripts/trap_sentinel_gate.py` is the
  **verdict** for this repo (docs-gate CHECK; pins by MEMBERSHIP, floors the
  population — a classifier that goes blind must RED, not report a green
  zero; the flag must GATE the failure branch, or the pin certifies nothing —
  measured, its own canary caught that); `trap_sentinel_drift_sweep.py` the
  verdict for all repos (pinned in `scripts/trap_sentinel_drift_floors.txt`);
  `trap_launder_premise_matrix.py` measures the **premise** one interpreter
  at a time and prints **UNSEEN, never a zero**, when docker is absent.
- **`errexit` is the precondition, NOT `nounset`** — measured both ways; the
  population predicate is the UNION, and the measurement MODE is part of the
  claim (nounset exits 127 from `bash -c`, 1 from a script file).
- Verdicts: LIVE-FORWARD (a provable abort, every run), EXPOSED, SENTINELLED,
  UNPARSED, plus REPLACED (2+ EXIT traps — `trap` replaces, earlier cleanup
  silently dropped). Each finding carries its exposure window and its abort
  triggers — a window with **zero** triggers provably cannot launder.
- **UNPARSED is the instrument admitting it cannot read** — a runaway body
  reads as a false SENTINELLED, which HIDES exposure; never pooled, the
  verdict gate reds on it (riir-chain's
  `block_pipeline_reachability_gate.sh` embeds a quoted awk program with an
  unmatched brace in DATA). The last EXPOSED row in the workspace —
  `riir-ai/scripts/e2e_internet.sh`, trap on line 41 of 43 — is INERT by its
  zero triggers.
- **shellcheck does not find this** (measured): the fatal line got SC2250,
  brace style. Canonical failure: mmorpg-remake's `ci_feature_guard.sh` — the
  script its `rust.yml` runs — could not fail past layer 13 for months,
  because its layer-13 trap named two variables assigned ~20 and ~45 lines
  later.

## A lane compiles what it NAMES — `scripts/wasm32_surface_audit.py`

Every wasm32 axis is about *how* a lane compiles what it names; the seventh
is one level up: **is what it names the whole surface?** A row cannot notice
a package it does not select — a positive `#[cfg(target_arch = "wasm32")]`
block no row built and that had been uncompilable since it was written.

```bash
scripts/wasm32_surface_audit.py            # all contract repos (derived)
scripts/wasm32_surface_audit.py ../riir-ai # or one, by path
```

- A **report, not a gate** (exit 0). Four buckets, never pooled: **NAMED** ·
  **BY-DEP** (Issue 774: no row names it, but a named package reaches it
  through non-optional in-repo path-dep edges — reachability, never folded
  into NAMED, because that coverage dies by a dep-graph edit in someone
  else's manifest) · **UNRESOLVED** (a `--workspace` or derived row exists —
  not clean) · **UNCOVERED** (no row could reach it). `--self-test` proves
  the by-dep detectors fire in BOTH directions.
- The predicate is the **positive** cfg (`not(wasm32)` is an ordinary
  native-only guard; counting it inflates everything); comment lines excluded
  — prose explaining a cfg is not a cfg. ⛔ It reads ATTRIBUTES, not lines:
  raw-string fixtures were once a repo's ENTIRE count (the masker is IMPORTED
  from `platform_dead_code_audit`, not re-written).
- `cfg!(target_arch = "wasm32")` is a RUNTIME branch — split out, never
  counted as surface, printed as EXCLUDED.
- Standing: 26 NAMED · 2 BY-DEP · 0 UNRESOLVED · 1 UNCOVERED over 29 packages
  (`mmorpg-poc-submodule` the deliberate negative control). Verdict half:
  `scripts/wasm32_surface_drift_sweep.py` (sharing `classify_repo()`;
  `max_unresolved = 0` a WALL — a ratchet on a bucket whose meaning is
  *unanswered* is a backlog; UNCOVERED pinned by NAME in
  `scripts/wasm32_uncovered_expected.txt`, reds in BOTH directions; floors in
  `scripts/wasm32_surface_drift_floors.txt` with a reserved TOTALS row because
  the walk floor is vacuous in 7 of 16 repos). **Take the figure from a run,
  never from this bullet** — the standing sentence here went stale once
  already.

## An arm that exists and RUNS may still reach nothing — `scripts/arm_reach_audit.py`

Mutate a module's source **outside its own arm bodies**, re-exec, run its
arm, ask whether the arm noticed. Issue 789 measured it 53 times by hand,
finding seven arms that certified nothing — a census done by hand is a census
that stops being done.

```bash
scripts/arm_reach_audit.py                    # the report, the CHECKS population
scripts/arm_reach_audit.py --self-test        # arms over its own buckets
scripts/arm_reach_audit.py skill_repo_set     # one module, by substring
scripts/arm_reach_audit.py --include-all      # every scripts/*.py DEFINING an arm
scripts/arm_reach_gate.py                     # the VERDICT — workstation, minutes
scripts/arm_reach_gate.py --canary            # arms over its own arithmetic
```

- A **report, not a gate** (exit 0), except a blindness floor or a failing
  self-test (exit 2). A harness that generates no mutants, or whose runner
  always says KILLED, prints a *perfect* score — the same output as
  perfection.
- ⛔ `BASELINE` is the bucket that was missing: an arm *already failing* kills
  every mutant and reads as 100% reach — it **inflates** the killed floor.
  `BASELINE-RED`/`BASELINE-CRASH` are module-level verdicts, not pooled; the
  gate walls both at 0. Check the ENVIRONMENT first on a RED (drift sweeps
  need the same markers the gates get).
- ⛔ `TIMEOUT` is its own verdict with CRASHED's standing (*evidence of
  nothing*), rows named individually, the flag checked BEFORE the kill; the
  deadline derives from the module's own baseline run (10×, floored). The
  watchdog is a thread + `interrupt_main`: it reaches a pure-Python loop and
  NOT a blocking C call. Run `--include-all` module by module with an
  external wall timeout — one invocation is unbounded and not resumable.
- ⛔ The exec namespace is a registered module — a bare dict dies at its
  `@dataclass` line (seven classifiers carried 796 of the mutants that way;
  the premise is CPython-version-dependent and asserted, not assumed).
- **NO-ARM** and **UNREACHED** are their own markers, never pooled: four
  checks delegated to an imported classifier and had no arm of their own — a
  classifier's self-test cannot reach its consumer's pin arithmetic, so each
  needed a `gate_selftest` + a small EXTRACTION (the verdict arithmetic sat
  inline in `main()` beside its own error messages — unreachable by
  construction). Extraction IS the repair: measured, it also exposed live
  defects (a display/tally disagreement; a double-counted state).
- ⚠ Reach is per MODULE (shared rules read as cross-module survivors — check
  this class FIRST on any survivor); EQUIVALENT mutants are the other
  false-positive class, so SURVIVED is arm reach per function, never a defect
  count.
- ⛔ OPERATOR SCOPE is narrow — control flow and off-by-one only; regex and
  string literals are NOT touched, and that is where most decision logic
  lives. A low kill count is not evidence an arm is weak.
- ⛔ **In every module the CLASSIFIER was well armed and the VERDICT was
  not** — writing the reasons down is the adjudication (a third of the
  pinned "EQUIVALENT" rows were plain functions over plain data: real gaps
  wearing a label). The three weakest modules went 4→24, 3→19, 4→54 killed —
  each needed an EXTRACTION or an INJECTION first.
- The verdict half `scripts/arm_reach_gate.py`: survivors pinned by
  MEMBERSHIP with a REASON per row
  (`scripts/arm_reach_survivors_expected.txt`), the wall 0 UNPINNED,
  `UNREACHED`/`NO-ARM` walled at 0 separately; the key is LINE-FREE (a
  line-numbered pin reds on every edit above it — a pin file that reds on
  noise is one people delete); each row's `#= ` comment is VERIFIED against
  the observed text (a comment nothing can red drifts into a lie). It gates
  ITSELF, and the two PERMISSIVE sets (`EXEMPT_FUNCTIONS`, `ARM_NAMES`) are
  pinned by membership because no floor guards them.
- ⚠ Its cost is a RANGE, not a budget: ONE run in SIX completed, in 868s; the
  others capped or killed (1800s–121min) — the spread is UNEXPLAINED and left
  that way rather than given a third mechanism (three earlier causal claims
  were RETRACTED when the same quantity was re-measured later: consistency
  across runs is not reproducibility when every run shares one box and one
  hour). The stable decomposition: cost = SUM of per-module costs spanning
  three orders of magnitude (0.03s–34.34s). Run it in a DETACHED worktree;
  kill by PID, never by pattern.

## A gate whose own failure path is asserted by nothing — `scripts/check_validation_gate.py`

Issue 775 landed canary arms over a gate's **own pin arithmetic** — a rule
landed in one gate and never generalised (the sixth recorded instance of that
shape). Measured: **six of twenty** CHECKS invoked no arm at all, own or
delegated, carrying 2,050 lines of per-push logic whose failure path no test
had ever executed.

- The predicate is **invokes an arm UNCONDITIONALLY**, not "has an arm":
  `docs_gate.sh` runs each check as `"$PY" "$script"` — **no arguments** — so
  an arm behind `'--canary' in sys.argv` never fires on a push. Not
  hypothetical: eight adversary arms landed flag-gated the day before and ran
  on no push at all, at 0.17s — there was never a cost argument.
- **Delegation is credited, and must be**: four checks reach their arm
  through the classifier they import; refusing that pushes every gate toward
  a second copy of a rule it does not own. But a classifier's self-test
  cannot reach its consumer's pin arithmetic — that gap needed the
  `gate_selftest`s (arm-reach section).
- Exemptions pinned by **membership with a reason per row**
  (`scripts/check_validation_expected.txt`); the file is **deliberately
  empty** — a row reading "not written yet" is a backlog wearing a pin. Two
  floors (`MIN_CHECKS` the array parse, `MIN_ARMED` the AST resolution — a
  walk that finds every check and credits none looks exactly like nobody
  having written any arms), plus a canary arm asserting `main` is not in the
  ARM_NAMES vocabulary (a widened permissive set greens every check
  silently).
- `--prove-fires <sha>` is two-sided against an independently known answer
  (~0.3s, opt-in, `scripts/` only).
- ⚠ **What it does NOT assert:** that an arm which exists and runs is any
  *good* — arm quality is not statically decidable and is not claimed; that
  question is `arm_reach_audit`'s.
- ⛔ **There is NO sweep half, and that is a measurement**: katgpt-rs is the
  only repo with a CHECKS array — a sweep would derive a population of ONE
  and print a confident green over it. Do not add one by symmetry;
  re-measure first (the one time that was assumed, it was wrong by seven
  repos).

## A census reads the DOCUMENT, so an undocumented instrument is invisible — `scripts/instrument_reachability_gate.py`

Two censuses enumerated the audits AGENTS.md documents against their sweep
halves, and both were wrong by exactly one instrument — *a census that reads
the document cannot see an instrument the document omits*, and it reports a
confident, complete-sounding answer over the subset it can see. The
DOCUMENTATION was the population nothing floored.

```bash
scripts/instrument_reachability_gate.py                  # the verdict, this repo
scripts/instrument_reachability_gate.py --canary         # the adversary arms
scripts/instrument_reachability_gate.py --prove-fires 18dbe980
scripts/instrument_reachability_drift_sweep.py           # every repo, ratcheted
```

- The predicate is **REACHABLE**, not "named in AGENTS.md": roots are
  `AGENTS.md`, `scripts/docs_gate.sh` and `.github/workflows/*.yml`; the
  closure follows script → script references, so a helper invoked by a
  documented instrument counts. Required by this: `all_ignored_target_audit.py`,
  `cfg_row_implication_audit.py` and `ci_test_execution_report.py` are in no
  document either, yet each runs per-push via an instrument that IS
  documented.
- ⛔ **`HISTORY.md` is deliberately NOT a root** — it is the archive, not
  loaded into a session, and an instrument findable only from it is the
  instrument that stops being run; counting it would make the gate vacuous on
  the motivating case.
- Pinned by **MEMBERSHIP** with a **REASON per row**
  (`scripts/instrument_unreferenced_expected.txt`); a reasonless row is
  refused; reds in both directions. **The default for a real instrument is to
  make it findable, not to add a row.**
- Two floors: `min_scripts` the walk; `min_roots` the **permissive**
  direction — a root set that quietly *widens* makes everything reachable and
  prints a green.
- The sweep half is a **ratchet**, and the reason is measured: first run, 95
  of 152 unreachable workspace-wide, one repo 61 of 61 where the predicate
  OVER-CAPTURES (plan-scoped one-offs, "unfindable from AGENTS.md" is the
  correct state) — so this repo's own rows are membership-pinned with reasons
  and the sweep constrains the DERIVATIVE everywhere else. ⛔ The closure is
  TEXTUAL — any basename mention credits reachability, and a row leaving the
  unreachable set without a wiring commit means something started naming it.
  Take the live figures from the sweep's own summary line.

## A kernel can derive its SHAPE from a buffer's declared size — `scripts/len_derived_binding_audit.py`

`let n = kv.len() / 2 / kv_stride;` inside a kernel computes that dimension
from the bound buffer's **declared size**, not the live range — bind a
capacity-sized handle and the kernel silently derives the WRONG shape: reads
never-written memory, writes a measured identically-zero result. No panic, no
NaN, no wrong-looking output. The defect is a **JOIN** of facts in two files,
and a report over either half alone is noise; HALF C resolves wrapper
parameters through **workspace** callers — cross-repo by construction, the
only instrument in its family that is.

```bash
scripts/len_derived_binding_audit.py        # the report, all contract repos (derived)
scripts/len_derived_drift_sweep.py          # the verdict, every repo, pinned
scripts/len_derived_drift_sweep.py --no-stability   # skip the leave-one-out arm
scripts/len_derived_drift_sweep.py --canary         # the adversary arms
```

- A **report, not a gate** (exit 0), except a WALK REGRESSION — loose global
  floors refuse a confident zero below them. **UNRESOLVED is not clean** (118
  of 164 bind sites) and never folded into a neighbour. **PERSISTENT-UPSTREAM
  is the EYES LIST**, pinned by MEMBERSHIP in
  `scripts/len_derived_eyes_expected.txt`, keyed line-free on
  `(repo, file, kernel, handle)` **plus a count within that address**.
- The verdict half walls the joined buckets (CAPACITY, CAPACITY-UPSTREAM,
  PERSISTENT) at 0 and floors `min_kernels`/`min_binds` per repo — **vacuous
  in 14 of 16 repos** (the population shape, not a defect). ⛔ **`DEFERRED`
  is not enough here**: a partial clone can corrupt the verdict of a row in a
  repo that IS present — measured both directions (7 of 251 cited caller refs
  cross-repo; leave-one-out over all repos 0 flips) — so the sweep re-measures
  a TARGETED leave-one-out every run rather than carrying the measurement
  forward.
- It carries **no `min_rs_files` column** on purpose: three sweeps floor that
  identical walk over that identical population, and the delegation is
  **ASSERTED** — a pinned repo that loses its non-zero row in
  `orphaned_attr_drift_floors.txt` reds — with ONE measured exception (Issue
  902): a repo born md-only has a TRUTHFUL zero row, accepted only while the
  walk measures ZERO tracked `.rs` in that repo — re-measured every run, the
  acceptance printed, never a standing amnesty; the first `.rs` to land reds
  exactly as a zeroed row on a code repo does.

## An item can be dead on a platform NO lane compiles — `scripts/platform_dead_code_audit.py`

A const declared ungated, used only inside an `#[cfg(target_arch = …)]` fn:
**dead code everywhere but that arch**, and silent there. `full_gate` is
macOS/aarch64, `wasm32_gate` builds a third triple — so the x86_64-native
lane that emits the warning is a **workstation** lane and no automatic gate
ever sees it. Five specimens in two days across two repos.

```bash
scripts/platform_dead_code_audit.py             # all contract repos (derived)
scripts/platform_dead_code_audit.py ../riir-ai  # or one, by path
scripts/platform_dead_code_audit.py --self-test # the classifier arms
scripts/platform_dead_code_audit.py --prove-fires ea4c2873
```

- A **report, not a gate** (exit 0) — except a classifier MISS, which exits
  **2**: an instrument that cannot classify must not be read as `0 findings`.
  The self-test runs on every invocation. Tracked `*.rs` only; `vendor/`
  excluded with its count on the per-repo line.
- **MOD-REF is a separate bucket and is never folded into the count** — a
  `mod` referenced only from gated code leaves the module EMPTY, not dead;
  rustc reports at the ITEM, which this audit reaches independently.
- ⛔ Its header once claimed it could not INVENT a finding, and that was false
  on the first sweep: masking string literals dropped Rust 2021 inline format
  args. A conservative-by-construction argument is a claim about code
  somebody else wrote; this one survived until a real corpus contradicted it.
- ⛔ **An arm is only a canary if its own perturbation REDS it**: the
  `vendor/` arms red nothing because a redundant skip-entry was doing the
  filtering on the OTHER code path — one exclusion, two code paths, and the
  arm certified the path it was not aimed at.
- Verdict halves: `scripts/platform_dead_code_floor_gate.py` per-push in the
  docs gate (pins in `scripts/platform_dead_code_floors.txt` — two blindness
  floors, the MOD-REF row by MEMBERSHIP, plus canary arms over the gate's own
  pin arithmetic, which the classifier's self-test cannot reach) and
  `scripts/platform_dead_code_drift_sweep.py` on the workstation (population
  from `repo_set.txt` as well as the walk, so a partial box DEFERS loudly;
  pinned in `scripts/platform_dead_code_drift_floors.txt`;
  `--prove-fires` runs by DEFAULT in the sweep and is opt-in on the gate:
  ~5.6s of `git archive` is worth a workstation run and not a per-push one).
  Standing: 0 findings · 1 MOD-REF.

## `text=True` decodes with the SYSTEM locale — `scripts/subprocess_encoding_gate.py`

`subprocess.run(..., text=True)` decodes the child's pipe with
`locale.getencoding()`. macOS, `ubuntu-latest` and the M3 are all UTF-8, so
**nothing that could notice this ever runs it** — while every instrument here
prints `✓`, `✗`, `⛔` and em-dashes. Measured on a cp874 box: the em dash
comes back as three wrong chars (silent mojibake — a regex matches nothing
and reads a confident zero findings), or the decode raises inside
subprocess's reader THREAD, where `run()` returns normally with the
returncode PRESERVED and `stdout = None`.

`PYTHONIOENCODING=utf-8` does **not** fix the first mode and makes the second
MORE likely: it pins the CHILD's encoder, so the child emits correct UTF-8
that the parent then decodes as the locale. Both halves are needed, pinned as
separate classes — **DECODE** (`text=True` with no `encoding=`) and
**CHILD-ENCODER** (a `sys.executable` spawn with no `PYTHONIOENCODING` in its
`env=`) — because a shared pin hides which half regressed.

- It scans the AST, and that was not the first design: a paren-matched text
  scanner reported offenders inside the gate's own fixture strings, and the
  repairs on offer were to exempt the gate from itself or obfuscate its test
  data. `ast` sees a string literal as a literal; a file it cannot parse is
  **UNPARSED** and reds, never folded into the pass column.
- It is a per-push **gate**, not sweep-and-done, for a measured reason: the
  correct form carried a comment naming this exact defect while 27 more call
  sites were added without it. Ceilings 0 on both classes over floored
  populations.
- ⛔ And it shipped with one half — the gate, no sweep (Issue 783): eleven
  other verdict classes carry both halves, and the asymmetry was a skipped
  step, not a judgement call. **Before fixing such a class, grep the whole
  family and land the repair as one shared mechanism.** Measured cross-repo:
  29 DECODE + 2 CHILD-ENCODER over 5 repos, two not latent
  (`riir-clippy/scripts/gen_dashboard.py:552` reads em-dash commit subjects
  across siblings). Verdict half:
  `scripts/subprocess_encoding_drift_sweep.py` — its two floors are not
  interchangeable and neither is redundant: `min_calls` is 0 in 10 of 16
  repos (no subprocess at all), so `min_py_files` is the only blindness
  detector there. First run also found a site outside `scripts/` entirely
  (`.agents/skills/doc-sync/tools/linkcheck_sweep.py`) — the population is
  tracked `*.py`, not `scripts/*.py`.

## A sweep reads the WORKTREE, so a finding may exist in NO commit — `scripts/worktree_state.py`

Five-plus concurrent agent sessions write into shared worktrees, so a row a
sweep prints may sit on a line no commit contains, and a repo a sweep calls
clean may be clean only because somebody's uncommitted edit removed the
offending line. Measured: the workspace's entire standing CROSS finding was
an artifact of an uncommitted sibling edit, and the POPULATION moved too
(607 → 601 citations) — a floor re-pinned from such a run bakes another
session's in-flight edit into a tracked expectations file, where it reds on
every other box.

Three verdicts, never interchangeable: **COMMITTED** (the file matches HEAD —
an ordinary finding) · **UNCOMMITTED** (worktree-only; displayed, never
adjudicated against a pin — the DISPLAY reads the worktree, the PINS read
HEAD) · **MASKED** (HEAD-only: a committed defect read clean — the silent
direction, the worse one; 0 today is a measurement). The display stays honest
in both directions, every row labelled; MASKED needs no separate teeth — the
pins already count it.

⛔ **A fourth verdict, a different AXIS: STALE** (Issue 798) — the worktree
can match its own HEAD and still be 109 commits behind ORIGIN; a committed
FIX read dirty, and the row was investigated as unfixed until a manual
`git show origin/develop` settled it. `behind_origin(root, patterns)` rides
`sweep_advisory()` so the family gets it with zero call-site changes; the
three-dot diff is load-bearing (a two-dot diff also reports unpushed commits
— a repo merely AHEAD would read as stale); the sweep deliberately STAYS RED.
⛔ `upstream_axis()` is the upstream half EXTRACTED — the two entry points
were NOT equivalent and nothing said so; the one sweep on the low-level path
had the worktree axis and no upstream axis at all (a finding at a file whose
repair was already committed upstream). Not a fifth MECHANISMS row — one
mechanism whose membership test was wrong; the arm pinning the old behaviour
was REPLACED, not deleted (an arm pinning a false premise is worse than no
arm).

- **A sweep does NOT fetch** (measured: 250.2s serial / 50.2s at 8-way
  against sweeps costing 0.04–40s — a per-sweep fetch is 5–30× the cost of
  the thing it precedes, paid ~19 times over a family run). A `--fetch` flag
  nobody passes is not a repair, and an automatic fetch turns an observer
  into a writer of refs other sessions own. **Freshness is a property of the
  BOX at a moment, not of a sweep**: fetch ONCE per session
  (`scripts/fetch_contract_repos.py`; `origin` is NAMED, never a bare fetch —
  one repo carries a second remote stale by design; per-repo `--timeout`
  bounds the spawn) and let the sweeps DISCLOSE. Exit 1 only on fetch
  FAILURE, never on "nothing moved". The remedy line names the INSTRUMENT,
  never a git command (contract spellings are nonexistent directories on an
  aliased box).
- ⛔ **Pick the instrument from the CLASSIFIER's shape, never by preference**:
  `head_delta` (per-file row independence; `.head` = committed + masked,
  **never `committed`** — a pin adjudicating `committed` understates every
  ceiling by exactly the silent direction) · `head_overlay` + `delta_of` (ONE
  interceptable reader; EMPTY dict means skip the second classification, and
  `None` is a VALUE — tracked but not in HEAD) · `head_tree` (a classifier
  with several seams runs UNMODIFIED; measured 22–32s per dirty repo against
  0.04–0.26s clean, so it yields `None` on clean and the caller MUST skip;
  `paths=` narrows, a dropped file is silently ABSENT — caller's risk;
  `extra_dirty=` widens the trigger for `os.walk` sweeps). A CROSS-FILE
  classifier must NOT use the per-file shortcut. The patterns must name every
  file that can CHANGE a verdict — the ROOTS as well (a dirty root changes
  other scripts' verdicts), and a bare `("*.rs")` is a footgun: it is a
  string, not a tuple; the helper coerces and an arm pins both sides.
- ⛔ **The VERDICT belongs in the row KEY** (a key-matched row is filed
  COMMITTED carrying the WORKTREE's object — every field the key omits is one
  where the worktree silently overrides HEAD, and every ceiling partitions by
  verdict); **a FLOOR is a pin too** (floors read HEAD: the class was
  measured on a POPULATION, not a finding). Classes can split by ORACLE
  (worktree vs `git log` vs worktree-by-construction). The row key is
  deliberately LINE-FREE.
- **ADVISORY, never a failure**: a sweep that hard-reds on an ordinary dirty
  worktree is a sweep nobody runs; it rides the FINAL line in BOTH directions
  and is SILENT when nothing dirty meets that sweep's own population. ⛔ And
  "wired into every sweep" was typed as a NUMBER first and was wrong within
  two hours — the repair is mechanical:
  `scripts/sweep_advisory_membership_gate.py`, a MECHANISMS registry (slug →
  why + call-names) per family-wide mechanism (worktree-advisory,
  known-extra-exemption, head-provenance, alias-open), verdict per mechanism
  NEVER pooled, gated by MEMBERSHIP (a count is green on a swap), reds in
  BOTH directions, the exemption file deliberately EMPTY, registered only
  AFTER the fan-out. It asserts the CALL, never that the patterns name the
  sweep's own population — a per-sweep read.
- Arm-reach lessons on the module itself: `dirty_files`' separator
  normalisation red nothing under perturbation because git's OUTPUT SHAPE is
  the premise — an arm whose flip changes no input is replaced by a premise
  arm, never deleted as dead. And `n_assertions` was a survivor until
  `__file__` was injected — the extraction IS the repair. Budget the canary
  cost before wiring one: each sweep `--canary` re-enters `main()` over every
  contract repo, and four new arms measurably cost ~45%.

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
path added — committing it reverts them; two-stage: mtime then line-set
containment) · **rustfmt round-trip** (`--fmt`: identical to
`rustfmt(HEAD)` provably carries zero content — the only signal that yields a
proof, and the one that refuted a wrong belief on its first run). When you
must commit into a file a sibling is editing, commit **your blob**: HEAD's
version + your edit, `git hash-object -w`, then `git update-index
--cacheinfo`.

⛔ **Nothing in git identifies WHICH session did something here, and all
three fallbacks are measured broken** (Issue 840): shared authorship (every
commit authors as one address); a shared worktree has **one `HEAD` reflog**
(it answers what happened and in what order, never whose); and elimination
fails on an incomplete roster (sound only in a two-session worktree;
`ListAgents` shows only live sessions — a floor on the count, never the
count). The same defect one layer UP, in the messages: a peer's `from=`
names the pipe and is reliable; the name a message **signs itself** with is
self-asserted prose and is not (measured: messages from one pipe signed
themselves with the OTHER's name). **Attribute a message by its `from=`
pipe, never by its signature**; quote the pipe when you attribute; treat a
disagreeing signature as the two-session question it is. ✅ The one form that
survives a roster you cannot enumerate: **say what you CHECKED, not who you
concluded** — it makes no claim the evidence cannot carry. **Put `Session:
<name>, <epoch>` in the commit body** — a commit's own TEXT is the only
self-identifying evidence in the repository, names are REUSED (measured: one
name across two sessions sixteen hours apart, so the epoch is load-bearing),
and the marker is what catches an over-claim, including your own. ⛔ And the
marker is EVIDENCE, not proof — by its own rule (nothing authenticates it);
treat a missing one as no evidence at all.

### The same hazard one layer down: a FIXED temp path — `scripts/shared_temp_path_gate.py`

A test writing to `std::env::temp_dir().join("fixed_name.bin")` is safe
against its sibling tests in one binary — each site has its own filename —
and **not** safe against another PROCESS running the same test: all temp
users share one `/tmp`, and `create` truncates. Measured twice, because one
reproduction is an anecdote: 1 failure in 24 concurrent runs, byte-identical
to the x86_64 matrix's cell-5 red; and five tests failing AT ONCE across five
DIFFERENT filenames — another process truncating all five.

- ⛔ **The lesson is about the CONFIRM step**: the x86_64 matrix filed one of
  these as **TRANSIENT — failed in the cell, passed alone** — a true
  statement and the wrong conclusion. A CONCURRENCY defect passes alone BY
  CONSTRUCTION (alone there is no second process); a load-sensitive BAR is
  the only class that reasoning is right about.
- The repair is the form the repo already uses elsewhere:
  `std::env::temp_dir().join(format!("name_{}", std::process::id()))`. 13
  sites had it and 27 did not — 25 repaired, 2 adjudicated deliberate — which
  is why this is a gate and not sweep-and-done: the rule was known and
  un-enforced. Membership + a reason per row, both directions, floors on the
  walk AND the predicate. Three STATED blind spots: a `temp_dir()` bound to a
  variable first; other fixed-scratch spellings; and the cross-repo axis —
  **count first**, never carry a population-of-one answer across (the one
  time that was assumed, it was wrong by seven repos).
- ⚠ `cargo fmt -p <crate>` is not usable for a repair of this shape — it
  reformats ~1300 unrelated lines; format clean files per-file.
- Verdict half: `scripts/shared_temp_path_drift_sweep.py` — ceiling a RATCHET
  at measured (a wall would demand 98 repairs in ten trees this session does
  not own). Repair half: `scripts/shared_temp_path_fix.py` (idempotent,
  LF-preserving, refuses to leave a line over `--max-width` rather than
  invite the wholesale reformat). It IMPORTS the gate's `mask` and
  `FIXED_JOIN` — a repair pass whose idea of a site differs from the gate's
  either misses rows the gate will red on or edits code the gate never asked
  about. ⛔ Its first version tested CRLF before asking whether the file had
  a site at all, printing loud refusals for 18 files carrying none: a message
  that invites work nobody needs to do is the cries-wolf failure mode, one
  instrument down.

**Shared target dir:** a count-pinned or feature-switching gate run
concurrently with another cargo process in the same `target/` reports a
failing test that passes when run alone. Read the failure's **SHAPE**:
`error: test failed` with **no `failures:` block and no `test … FAILED`
line** means the harness process *died* — nothing asserted anything, and the
count is truncated too. Diagnose by running the compiled binary directly from
`target/<profile>/deps/` (no build lock needed; filter out the `#![cfg]`-gated
copies `--list` reports as 0 tests). A gate whose verdict the box can
invalidate should **refuse**, not warn — detect concurrent cargo by working
directory, not command line; a lock-based check cannot work (cargo releases
`target/<profile>/.cargo-lock` *before* running the test binaries).

## Lint healing — `cargo refine` before manual fixes (adopted 2026-08-24)

Mechanical clippy findings (`needless_return`, `unnecessary_map_or`,
capacity, `collapsible_if`, …) are fixed by the riir-refine healer FIRST,
manual second:

```bash
cargo refine <paths>                                        # DRY RUN (zero edits)
cargo refine --fix <paths>                                  # REAL fix: writes + compile-gates
cargo refine --fix --write --verify <paths>                 # compile-gated apply
cargo refine --fix --write --verify --verify-args "--features <set>" <paths>  # gated code
```

- Global binary `cargo refine` = `~/.cargo/bin/cargo-refine` → the sibling
  `riir-refine/target/release/cargo-refine` (built with the RELEASE feature
  set; a rebuild naming anything less compiles no binary and the symlink
  silently keeps resolving to a stale one — `--version` prints the truth).
  Missing sibling → fall back to manual fixes + `cargo clippy --fix`.
- `--verify` compiles baseline → applies → re-checks → auto-REVERTS breaking
  edits. Feature-gated code needs `--verify-args "--features <set>"` (a
  default-features check compiles gated files empty — a green check proves
  nothing about them).
- **The healer fixes only what THIS repo's clippy reports**: pedantic/nursery
  lints only where the target crate enables them (katgpt-rs enables none — a
  skip here is correct, not lost coverage). Deliberately SILENT on documented
  divergence classes (comment-guarded matches, array-literal defaults,
  named-arg renames, nested macro args) — those stay manual; see the
  `cargo-refine` skill. `cargo clippy --fix` remains fine for one-off trivial
  fixes; the healer wins on batches (span-preserving, comment guards,
  compile gate, self-evolve memory). Observed misses → the session record;
  they feed the post-mining queue (usage-artifact improvement intake).

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

**Promotion requires modelless gain.** A perf gain on a biased/incorrect
answer is NOT a modelless gain — the quality gate (G1 or equivalent) must
pass modellessly for the GOAT to hold.

⛔ **A latency number without its BOX STATE is not a measurement.** Record
free RAM, commit-vs-limit (read AT launch — the limit moves both ways),
concurrent heavy jobs, and on a laptop POWER SOURCE and POWER MODE (`pmset
powermode` is a three-state enum; "not 0" is not "Low Power"; the sudo-free
detector is a fixed-kernel canary:
`riir-reflex/scripts/bench_preflight.sh`, which prints a `PROVENANCE:` line
to quote beside the number) **next to** any latency figure taken on a shared
box, or it is not reproducible. Rank concurrent jobs by COMMIT, never by
working set, and compare the total to the limit read at launch. Free RAM
right after another job exits is a trough between phases, not a window. G2
mandates `--release`: a latency gate in a debug build measures an unoptimised
binary, and on a thrashing box measures the pagefile.

**Lossy-surface promotion rule:** a **lossy** surface (quantization,
compression, any bit-changing transform) gates on **deployed-path behavior —
per-family, conditional retention**, not on bit-identity or aggregate
perplexity alone: aggregate metrics can be flat while family-conditional
behavior flips (external confirmation: 43–45% exact-match under BF16 with
flat downstream eval aggregates — the exact failure shape).

**UQ-bearing primitive GOAT gate extension ("Report the Floor"):** any
primitive claiming a probability distribution, predictive interval, quantile,
coverage guarantee, confidence score, or calibrated uncertainty MUST benchmark
against the **conformal-naive floor**
(`ConformalIntervalCalibrator<SeasonalNaiveForecaster>`, Plan 340, `m=1`,
plain split conformal) on CRPS / coverage / Winkler score. Cannot beat the
floor ⇒ the GOAT gate FAILS. Grandfathered UQ primitives include the floor at
their next re-gate.

## Substrate-First Gate (MANDATORY before implementing)

Before implementing ANY new System impl, trait, perception/cognition/emotion
pipeline, state management, spatial query, or vocabulary type, run the
`substrate-first` skill: (1) **vocabulary translation** — grep 3+ name
variants (concepts ship under operator names like `GenericSpatialBelief`; a
single-vocabulary grep returns ZERO hits even when substrate fully exists);
(2) **codebase grep** across `*.rs`, not just `.plans`/`.docs`/`.issues`;
(3) **architectural rule check** — domain classification, two-brain model,
sync boundary, bridge pattern; (4) **consume vs build** — if substrate
exists, consume it; if not, file an issue in the right repo FIRST. Prevents
the drift pattern of a parallel system re-implementing shipped substrate
under a different name.

⛔ **A concurrent session is the OTHER way this gate gets skipped, and the
cost is a NEGATIVE RESULT rather than a duplicate** (Issue 825): two sessions
implemented one issue four hours apart, and the second measured a failure,
filed the negative, removed the issue file and wrote "the paper's 20× does
not transfer" — it was two defects in its own walker, and the primitive
shipped green the same day. **Before recording a negative, re-run `git fetch`
and check whether a sibling shipped the same primitive; if one did, cross-run
the fixtures before writing the word "does not."** The disagreement was the
oracle — cheaper than either instrument's self-consistency. **Distrust a
mechanism inferred from a monotone sequence**: a converging error is
consistent with many mechanisms. **Repair beats delete when the loser is
independent** — two benches that agree are stronger than either alone.

Research workflow (paper classification, 7-repo routing, fusion-first
distillation, novelty + GOAT gates, modelless-unblock protocol §3.5):
`.agents/skills/research/SKILL.md`.

> **Repo count:** the **product/distillation set is 7** — `katgpt-rs` (public) +
> `riir-ai`, `riir-chain`, `riir-neuron-db`, `riir-train`, `riir-game-sdk`,
> `riir-dapps` (private). That is NOT the repo total: the
> workspace is **28 repos**, all of which carry a root `BOUNDARY.md`
> (add `riir-mmorpg-examples`, `riir-refine`, `riir-viewbridge`,
> `riir-auth`, `katgpt-web`, `riir-dao`, `riir-deployer`,
> `riir-esp32`, `riir-llm`, `mmorpg-editor`, `mmorpg-remake`,
> `mmorpg-remaster`, `riir-kat`, `riir-shader`, `riir-reflex`,
> `riir-infer`, `riir-reflexer`, `reflex-site`, `riir-instinct`,
> `riir-rethink` — born 2026-10-03, the Instinct/Rethink split carve's
> private moat repo, seeded from `riir-instinct/moat/`; `seal-std` —
> born 2026-10-07, the Seal remake's shared standard crates — the
> animation-library leaf both the editor and the runtime resolver run).
>
> Read a count in prose as a claim, not a fact — and read a count that
> MATCHES as a claim too: a count is not a checksum over a set. Drift
> history: HISTORY.md.

## Numbering Discipline

Issue, plan, doc, benchmark, and research numbers are **monotonic and never
reused** — even after a file is removed per the noise-reduction rule. Before
creating a new `.issues/` file, read `.issues/.highwater`, use `value + 1` as
the number, and write the new value back. Same for `.plans/`, `.docs/`,
`.benchmarks/`, `.research/` — never recycle a number that git history shows
was already allocated. `ls` the target folder AND re-read `.highwater` at
WRITE time (a stale `.highwater` view dual-allocated two numbers once), and
fetch + run the allocation-time gate:

```bash
git fetch origin && py ../katgpt-rs/scripts/dual_allocation_gate.py
```

It answers the question `ls` cannot: has this checkout's upstream allocated a
numbered document since their merge base? **TWIN** (same filename stem on
both lines) is your own rebased work — a fetch resolves it; **INDEPENDENT**
(different stems) is two documents about to own one number, and it names both
sides' adding commits. Its reach limit is documented and real: it sees
divergences THIS box participates in. Probe basis:
`scripts/dual_allocation_fp_probe.py` (7202 one-sided pairs green by
construction; 39 real divergence incidents split 31 TWIN / 8 INDEPENDENT —
naive subject-equality would have cried wolf 31/39). ⛔ **COUNTER** is the
third verdict: a number allocated and CLOSED in one commit never has a file
in any tree — both sides bumping `.highwater` past the merge-base value is
the witness, armed on every push (Issue 850).

- Two documents already sharing a number: the one with the most inbound
  mentions KEEPS it — and the obvious count is the wrong one (by-NAME
  citations tie; the weight is in the unqualified `Plan N` mentions).
  `scripts/citation_weight.py <repo> <dir> <number>` attributes those,
  awarded only on a strict margin, with everything else printed as its own
  UNRESOLVED number, never folded into a winner. ⛔ **Do not renumber on a
  margin the instrument did not award** — leads of +1..+5 with heavy
  UNRESOLVED, one outright tie and one DECLINED were deliberately left alone;
  pretending a coin flip can arbitrate is how it gets recorded as a
  measurement.
- ⛔ **CLOSING a holder does not retire its row — it is when the row starts
  being the only record**: the holder is gone, the collision is not;
  `removed_by_number()` recovers BOTH sides from
  `git log -M --diff-filter=D --full-history` (the majority case: both
  holders closed and removed). A pin comes out only when the *number* stops
  being doubly held, which a deletion never achieves.
- ⛔ And that recovery ran in ONE repo of sixteen for two days (Issue 820) —
  the cross-repo verdict half `scripts/numbering_drift_sweep.py` now imports
  the same function: 0 tracked duplicates against ~183 historical collisions,
  workspace-wide in ~4s. `max_hist` is a RATCHET at measured (invented
  reasons are a backlog wearing a pin), and `min_numbers` floors the HISTORY
  walk — NOT a second `min_files`; they break separately.
- The verdict is `scripts/numbering_gate.py` in the docs gate, in **two
  regimes** (`scripts/number_collisions_expected.txt`): at or above
  `era_boundary = 700` a **WALL pinned by MEMBERSHIP** with a reason per row
  — a count is green on a swap — and below it a **RATCHET**, counted and
  never pinned (the pre-gate archive; ratcheting a bucket that means
  *unread* is forbidden). The boundary is measured, not round.
  Rename-collapse needs topic + time + stem evidence (line similarity alone
  resurrects collisions and buries retitles). ⚠ **Take the SCOPE from
  `scripts/numbering_floors.txt`, never from a walk of the tree** —
  `.benchmarks/` families per owner are intended, and a tree-derived
  population over-counted by 74% once.
- Report-only context: `scripts/highwater_contiguity_audit.py` REFUTED the
  counter-as-ownership-witness by measurement (438 gaps + 27 resets over 73
  counters; no major repo contiguous) — an over-claiming witness VALIDATES
  wrong addresses, and the contiguous-suffix witness was declined (decline is
  a correct answer).

## Branch

`develop` is the working branch. Don't create feature branches; commit
directly on `develop` per the global rule.

## Models
- riir-train/data/gemma-2-2b-it-f16.gguf — RETIRED locally 2026-10-07 (riir-train Issue 617,
  after the loud-fail gate edits); restore with `./scripts/data_backup.sh pull
  data/gemma-2-2b-it-f16.gguf` from the riir-train root. The katgpt-attn spike-census
  sidecars stay committed (the fixture gate reads sidecars, not the GGUF).
- riir-train/data/MiniCPM5-1B-F16.gguf — LOCAL-KEEP (pin, re-downloadable).
