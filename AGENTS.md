# AGENTS.md — katgpt-rs

The global `~/.agents/` rules apply; this file documents repo-local context
that supplements them.

## Boundary contract — read `BOUNDARY.md` first

[`BOUNDARY.md`](BOUNDARY.md) is the authoritative per-repo contract: what this
repo **owns**, what it **does not own** (with the correct home for each), the
crate-granular **allowlist** of what it may depend on, links to the cross-repo
rules' one canonical home, and the **drift ledger** of known gaps. On any
conflict with prose in this file, BOUNDARY.md wins.

- **Domain test:** is this a **modelless inference primitive** with no riir dep (this repo is upstream of everything)? NO → it belongs in another repo; file there.
- **Read it before** adding any dep, crate, module, System impl, or vocabulary
  type — and before assuming a concern is yours to implement.
- **Enforcement** is not prose: `../riir-ai/scripts/ci_boundary_contract.sh`
  fails on an undeclared cross-repo dep, on a drift row without its open issue,
  and on a contract row that no longer matches the measured graph. Run boundary
  checks VIA the `boundary-guard` skill, not as ad-hoc greps.
- **Found a violation?** File the issue FIRST (`.issues/NNN_boundary_*.md`), add
  the drift row, then fix. Closing the issue removes the row in the same commit.

## Modelless-first mandate (the core principle)

**This repo ships modelless inference primitives.** No training, no backprop,
no gradient descent. The only weight mutations allowed at runtime are:

1. **Freeze/thaw** — swapping a frozen snapshot (atomic, versioned, BLAKE3-checked).
2. **Raw/lora hot-swap** — applying a **deterministically constructed** (not
   trained) LoRA overlay via `LoraPair { reader, writer }` (Plan 025).
3. **Latent-space updates** — direction-vector projections, sigmoid gates,
   routing tables. These update latent state, NOT base weights.

### MANDATORY: exhaust modelless paths before deferring to riir-train

Before deferring ANY gate, mechanism, or plan task to riir-train ("this needs
training"), you MUST check whether the three modelless paths above can fix it.
See the research skill §3.5 (`.agents/skills/research/SKILL.md`) for the full
decision protocol.

**Systematic, characterizable biases are modelless-correctable candidates,
NOT automatic riir-train dependencies.** If a gate fails because of a known,
named bias (e.g., "signal doubled", "position offset", "attention asymmetry"),
check whether a deterministically constructed reader-LoRA or freeze-state
correction can fix it before concluding "needs gradient descent."

**Canonical failure — AC-Prefix G1 (Plan 313, 2026-06-24):** G1 was prematurely
deferred to riir-train without checking whether the doubled-signal bias could
be corrected modellessly via a deterministic reader-LoRA. The bias was
systematic and characterizable — exactly the case where raw/lora hot-swap
might work. The deferral was premature and has been reverted; the modelless
investigation (Issue 003, resolved-and-removed in commit `552b4632`) is
captured in `.benchmarks/313_ac_prefix_modelless.md` (Path 2: `attends_dedup`
eliminates the bias bit-identically to iterative-MLM on single-layer
micro-GPT, 0.0 diff). `ac_prefix` re-promoted to DEFAULT-ON on that
modelless pass; multi-layer equivalence remains a non-blocking riir-train
follow-up.

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
a green result says nothing about what it compiled to nothing. (The count is
deliberately not written here — this sentence said "three" for months while
the table below carried five, which is the drift the table exists to catch,
committed by the sentence introducing it.)

| Axis | Blind spot |
|---|---|
| `check` vs `clippy` | two `cargo heal` escape classes are rejected by clippy's typeck and accepted by `check` (E0689 ambiguous-integer, E0631 deref-coercion in `redundant_closure`) |
| default vs `--all-features` | non-default gated code compiles to **nothing** |
| `-p <crate>` vs `--workspace` | *at the same default features*: a crate's own non-default feature can be switched on by the ROOT crate's defaults once the root is in the selected set |
| no `--all-targets` | skips every test / bench / example — which is where gated code lives |
| dev vs `--release` | `debug_assertions` is always **ON**, so every item behind `#[cfg(debug_assertions)]` — and everything that depends on one — is only ever compiled in the configuration where it works (`.docs/10_audits/debug_release_profile_axis.md`) |
| `--all-targets` vs **doc-tests** | `--all-targets` does **not** include doc-tests — so the gate whose whole point is to compile everything never compiled a single doc example. Measured 2026-09-04 by the first full-workspace *execution*: **8 crates' doctests had never been built at any revision**, 31 lines across 14 files still writing `use katgpt_rs::...` after the root crate was split into sub-crates that must not depend on it, plus 3 independent defects — including one example asserting values its own formula cannot produce. Only `cargo test --doc` reaches this (`.issues/723` Class F) |
| **compile vs EXECUTE** | every axis above is about *compilation*. **The scoped core is now EXECUTED weekly** (`test.yml` + `scripts/test_gate.sh`, Issue 718 T3(b) landed 2026-09-04: katgpt-rs + katgpt-core `--lib` at default features with count floors, the riir-train 507 shape) — but that is 2 of 32 packages' lib suites. The other 477 integration-test targets and 176 bench targets over 32 packages remain executed by nothing automatic (`.docs/10_audits/ci_compile_vs_execute_axis.md` — the full-workspace `--all-features --release` run is now PRICED: 11,542 CPU-s cold / 45.5 min wall / 497 ok + 45 FAILED across 39 targets, and the finding is that `--all-features` is not a supported TEST configuration — fixture RNG streams and GOAT calibrations are per-feature; the full run needs per-target triage pins before it can gate anything — cost table in `.benchmarks/701_full_workspace_execution_pricing.md`, the 45 reds in six classes in `.issues/723`) |

**The last axis is the one that changes how to read every gate below.** A
green full gate is a claim that the workspace *compiles* under one feature
set on one platform in one profile — never that an assertion in it holds.
This repo's own rule is that an uninvoked assertion is *unknown*, not
passing, so by that standard every Rust assertion here is unknown: the 39
GOAT gates armed with `required-features` in Issue 713 T3 included, because
arming made a **named** run honest and nothing names them. AGENTS.md records
"All 39 pass there" under `--release` — that was a **workstation** run, and
nothing repeats it. The scoped core (2 of 32 packages' lib suites) now has a
scheduled answer via `test.yml`; everything outside it is still
unknown-by-default, and what is *not* optional is reading a green gate as
nothing more than "it built" — because that is all it ever said.

The `-p` vs `--workspace` axis is the least obvious. `cargo test -p katgpt-backend --lib`
compiled clean while `cargo test --workspace --lib` failed, because `gpu.rs` is
behind `katgpt-backend/gpu_inference` and the chain
`katgpt-rs/default -> async_qdq_overlap -> inference_router -> gpu_inference`
only fires when the root crate is selected. It also silently *shrinks* coverage:
four crates reporting "0 tests" per-crate contributed 704 under `--workspace`.

The **fifth** row is the newest and the command below does **not** close it —
it runs in the dev profile. Measured 2026-09-03: adding `--release` produced
**2 errors** and `cargo test --release -p katgpt-core --lib` did not compile at
all, which is the very command `.docs/10_audits/cfg_gated_silent_zero_pass.md` T2b tells everyone to use. Two
`#[cfg(test)]` blocks imported `crate::alloc`'s counters, which are
`debug_assertions`-only *by design*. Fixed in `.docs/10_audits/debug_release_profile_axis.md` T1; the axis itself
is T2.

Read that together with `.docs/10_audits/cfg_gated_silent_zero_pass.md` T2b and `.docs/10_audits/debug_release_profile_axis.md`, because the three
point in different directions and that is the lesson: debug **manufactured**
four false perf reds (713), debug **hid** a two-day release build break (715),
and the full gate compiles `debug_assertions` code only in the profile where it
works (716). **Neither profile is the safe default — the profile is part of the
claim.**

So before claiming a repo-wide green, run:

```bash
cargo clippy --workspace --all-targets --all-features --keep-going -- -D clippy::needless_range_loop -D clippy::map_clone -D clippy::iter_cloned_collect -D clippy::identity_op -D clippy::bool_comparison -D clippy::manual_is_multiple_of -D clippy::collapsible_if -D clippy::map_all_any_identity -D clippy::unnecessary_cast -D clippy::manual_repeat_n -D clippy::question_mark -D clippy::empty_line_after_outer_attr -D clippy::unusual_byte_groupings -D unused_mut -D unused_parens
```

The `-D` list (Issue 701 R3b, 2026-09-03) is the mechanical lints whose
all-features warning surface was healed to ZERO residual (67 → 13 distinct
findings; the 13 survivors are judgement-class and stay warnings), so a
regression now reds the gate instead of silently re-growing the ungated
warning surface. A lint with residual > 0 must NOT be added to it.

`--keep-going` is not optional — without it the run stops at the first failing
target and under-reports. This gate was **red on `develop` from at least
2cb97410 until `c284dbb2`/this commit** (5 broken targets) while every gate in
the block above was green. Treat a green gate as a claim about its literal
command, not about the code.

Don't run it by hand — `scripts/full_gate.sh` is the assertion (it also refuses
to report a pass off macOS, where the `target_os = "macos"` device backends
compile to nothing even with `--all-features`, and checks that this document
still quotes the command it runs).

**And the inverse holds, with nothing to enforce it.** That caveat protects the
`target_os = "macos"` backends by refusing to *report* off macOS. Running **on**
macOS silently drops every `not(target_os = "macos")` backend, `--all-features`
included — so the command above, which is run on the M3 by policy, is
structurally incapable of compiling them. Measured 2026-09-03 by parsing
`riir-gpu/src/lib.rs`'s module **declarations** (a file can mention the cfg
without being gated on it): **9 modules, 25,212 lines**, headed by
`qwen38_dense_cudarc` at 8,599. It is not theoretical — `riir-ai` `6bf51b592`
landed a CUDA-only lib that did not compile (`E0599`) and it stood for **7h45m**
until a 4090 re-pin happened to need a CUDA build; nothing else in the workspace
builds that code. Record and the open axis: `riir-ai` `.issues/857`.

This is the same shape as `.docs/10_audits/cfg_gated_silent_zero_pass.md` one axis over — there a
`#![cfg]`-gated test compiles to an empty binary and reports a green zero; here
a `cfg(not(target_os))` module compiles to nothing and reports a green build.
**A platform is part of the claim, exactly as the profile is.**

**As of 2026-09-04 the compile half of that axis is REACHABLE from the M3**, so
"we cannot build that code here" is no longer the answer:

```bash
scripts/check_platform_gated_modules.sh ../riir-train riir-train-gpu numeric_drift_cuda
scripts/check_platform_gated_modules.sh --canary ../riir-train riir-train-gpu \
    crates/riir-train-gpu/src/numeric_drift_tap.rs numeric_drift_cuda
```

`cargo check` never links — it needs only rust-std for the target and build
scripts that exit 0. Three build scripts stand in the way and none needs a real
cross toolchain: `blake3`'s NEON C (answered by `CARGO_FEATURE_NO_NEON=1`, the
pure-Rust path) and `libsqlite3-sys` + `sentencepiece-sys` (answered by the
**Android NDK's clang**, already installed here, which ships a complete linux
sysroot — the macOS SDK cannot, because `sys/cdefs.h` answers a linux-gnu
target with `#error Unsupported architecture`). Two details cost an hour each
and are in the script's header: cc-rs injects its own `--target`, so the shim's
must come **after** `"$@"` or clang honours cc-rs's and loses the sysroot; and
`-llog` is required because sentencepiece's cmake build links helper binaries
that need `__android_log_write`.

Read a green run narrowly: it is the **compile** half only, nothing ran, and
much of that CUDA code has never executed anywhere. But a green run is exactly
what `riir-ai` `6bf51b592` did not have when it shipped a CUDA-only lib that
did not compile and stood 7h45m. First use (riir-train `53538538`) typechecked
two `not(target_os = "macos")` modules whose five edited call sites the issue
had routed to "whoever next builds on the 4090". **`--canary` is not optional**
— it plants an undefined call inside the gated module and requires `E0425`,
because otherwise "Finished" is indistinguishable from the modules compiling
to nothing again, which is the entire failure this section describes. `.github/workflows/full_gate.yml` **declares** a
weekly cron, a manual dispatch, and a NARROW per-push/PR lane — the trigger is
real but fires only when the gate's own definition changes
(`scripts/full_gate.sh` / the workflow file itself; measured 2m17s made that
affordable, broadening to `**/*.rs` remains rejected — see the file's preamble
for the cost story and the promotion criterion). An ordinary code push does not
gate here; the rot check is the Monday cron. That
preamble also carries the liveness-sentinel record from `.issues/705` — the
gate's first two CI runs passed over ZERO compiled units (ANSI color codes
defeated every `^`-anchored counter, including the error count); closed +
removed 2026-09-02, full narrative in git history.

That was a declaration and not a schedule until 2026-09-01. `schedule` and
`workflow_dispatch` run **only from a repository's DEFAULT branch**, and this
repo's default was `main` — frozen at the v0.1.1 promote with no
`.github/workflows/` at all — so **neither trigger had ever fired**. The file's
own comment calls the schedule "the rot check"; the rot check had rotted, and
this paragraph advertised it as running. Fixed by moving the default branch to
`develop` (`.issues/704`), which is where AGENTS.md already says work lands;
both triggers are live as of that change.

Don't take that as permanently settled — a workflow file is identical on disk
whether or not it can execute, which is why this went unnoticed. The axis is
now measured: `scripts/ci_gate_coverage.py` reports, per workflow, which
declared triggers can actually fire, keeping **dead**, **unmeasured** (no remote
refs), **untracked** (committed by nobody yet — a colleague's in-flight file is
not a defect) and **PR-only** apart. It took the workspace from 7 dead workflows
to 1. Run it rather than re-reading trigger blocks by hand.

"Can fire" is still not "does fire". A workflow reachable only by
`workflow_dispatch` is a button, not a schedule, and three sibling repos
(`riir-chain`, `riir-dao`, `riir-neuron-db`) carried their whole Rust
compile/lint surface in exactly such a file — each by a *documented* main-only
owner call whose `push` is inert anyway, because `main` carries no copy of the
workflow. The report crosses the two axes rather than printing them side by
side, which is how that state stayed invisible: the coverage table credited the
command and the reachability table listed the trigger, and nothing multiplied
them together. RESOLVED 2026-09-02 (`.issues/706` closed + removed) — all
three now carry the `riir-clippy`-shape weekly `schedule`, the one trigger
that fires from the default branch while `main` stays frozen, with the
no-develop-push owner call untouched (`riir-chain` `b4a9b6e7` Tue 04:13 UTC,
`riir-neuron-db` `9d041d1` 04:29, `riir-dao` `9848811` 04:43 — whose workflow
also stopped hand-mirroring its guard layers and runs
`scripts/ci_feature_guard.sh`). The dormancy was not hypothetical: the same
day, `riir-neuron-db`'s standalone-dep gate was found RED nine days stale
(`29af2b0` changed the katgpt-rs patch set to `katgpt-device-verify` without
re-pinning `EXPECTED`; fixed `97e5161`) — invisible for exactly this reason,
because nothing ran the gate.

### The docs gate — same discipline, opposite cadence

`scripts/docs_gate.sh` runs the manifest/doc/skill drift assertions and
`.github/workflows/docs_gate.yml` runs it **per-push** on ubuntu-latest. Both
choices are deliberately the inverse of the full gate's, and both files say why:
this gate has no `cfg(target_os)` surface so platform cannot change its verdict,
and it costs ~3s rather than >13 min.

Per-push is scoped to **`main` only** (owner call 2026-09-03, was
`[main, develop]`): develop pushes no longer fire the gate, so the
introduce-commit catch now applies only to the promote lane — run
`./scripts/docs_gate.sh` locally for develop work, or the drift surfaces at the
next main push. The same change fast-forwarded `main` to the develop tip,
because a push trigger reads the workflow file from the PUSHED ref and a `main`
without this file would be a dead trigger — the exact shape of the pre-704 rot
check.

The `CHECKS` array in that script is the list. **The count is deliberately not
written here** — this paragraph said "the three" for one commit after the fourth
was added, which is the drift the gate itself exists to catch, committed by the
paragraph describing it.

The original three existed before that wiring and **nothing invoked any of
them**; two were red on `develop`, both on false positives against docs that
were correct. Treat an uninvoked assertion as unknown, not as passing.

Some of the checks are worth knowing about specifically (no count here — this
paragraph said "the three" for one commit after the fourth was added):

- `skill_repo_set_gate.py` (Issue 703) fails on a `SKILL.md` command block that
  types the repo set by hand instead of deriving it. It reads sibling repos,
  which CI does not have, so it separates its **vocabulary** (committed
  `scripts/repo_set.txt`, re-derived and failed-on-drift by every workstation
  run) from its **population** (12 `SKILL.md` locally, 8 in CI) and prints both.
  A gate that skipped in CI instead would be the vacuous green it exists to
  catch. Mark a deliberately narrow block `<!-- repo-set-ok: <reason> -->`.
- `agents_repo_set_gate.py` pins §"Repo count" above against
  `scripts/repo_set.txt` — membership FIRST, cardinality second. It exists
  because on 2026-09-03 that paragraph named a retired repo
  (`riir-armageddon`) and omitted a new one (`seal-remake-unity`) **while its
  count stayed correct**: one left, one arrived, total unchanged at 19. Every
  count in sight agreed and the set was wrong anyway, which is why the
  paragraph's own warning ("read a count in prose as a claim, not a fact") was
  not enough — a count is not a checksum over a set. It gates ONE paragraph on
  purpose: the obvious whole-repo version false-positives on history, and
  boundary-guard's ledger *should* still name the retired repo, because the
  227→225 edge delta IS that repo's two edges leaving. Both its inputs are
  committed, so it runs in CI; `repo_set.txt`'s freshness against the real
  workspace is a separate, workstation-only concern owned by the check above.
  A parser regression exits **2**, not 1 — an untrustworthy instrument is not
  the same finding as drift.

- `cfg_gated_floor_gate.py` (Issue 713 T4) is the GATE over the report below,
  katgpt-rs-scoped, with its pins in `scripts/cfg_gated_floors.txt` (the count
  is deliberately not written here — see the docs-gate preamble above). The one
  that earns its keep is `max_load_bearing = 0`: a new `*_goat.rs` gated on a
  default-off feature with no `required-features` row reds the push that adds
  it, before its green zero is cited as evidence. **Some of the pins are
  FLOORS**, on the population the auditor claims to have scanned, because a
  ceiling cannot fail once the instrument goes blind and reports zero. The
  first hazard found here was not the pins but the **trigger list** —
  `docs_gate.yml`'s `paths` filter carried no `.rs` glob at all, so the gate
  could not have fired on the only push it exists for.

  The **second** was worse and is the one to remember: `max_load_bearing = 0`
  is only as wide as `is_load_bearing`'s **vocabulary**, and a token-set gap is
  indistinguishable from a clean repo. T4c (2026-09-03, `2272b262`) found the
  set knew `goat`/`gate`/`g<N>`/`drill`/`proof`/… and did **not** know the
  `*_correctness` / `*_alloc_check` / `*_determinism` / `*_equivalence` /
  `*_floor` / `*_grad_check` dialect. Seven tokens added — each measured
  against all 2,157 workspace test+bench target names first, which is why
  `budget`, `check` and `calibration` were **rejected** — and 17 more
  load-bearing katgpt-rs targets appeared, including 8 `*_alloc_check` G4
  budgets and a Report-the-Floor UQ gate. All 17 armed and RUN in release:
  45 assertions, 45 pass, 0 fail — silently *unverified*, not broken, same as
  `.docs/10_audits/alloc_gate_per_thread_counter.md` T3. Found sideways, by adding tests to one such file, not by
  auditing the gate. Re-run the corpus token table when a new dialect appears.

  The **third** is what that widening did to the prose describing it. T4c made
  the classifier wider, so katgpt-rs's load-bearing ALL-IGNORED count went
  **3 → 5** — and both the pins-file header and
  `.docs/10_audits/cfg_gated_silent_zero_pass.md` item 7 kept their old
  numbers, because **nothing was pinned on that count**. The pins file argues
  the ALL-IGNORED *count* is not gateable (`#[ignore]` is the right marker for
  a slow or hardware-gated test) and that is correct but incomplete: **a set is
  gateable where its cardinality is not.** `scripts/all_ignored_load_bearing.txt`
  pins the five paths by MEMBERSHIP, each with the reason string read out of
  its own source, and `check_membership` reds on drift either way — so a sixth
  arrival reds the push, a same-size **swap** fails (the repo-set incident's
  lesson, one axis over, pinned as its own selftest case), and an emptied
  measured set reds on five removals instead of passing like every ceiling in
  this family does. An empty *allowlist* is refused for the mirror-image
  reason. All four directions canaried. It is deliberately NOT in
  `REQUIRED_PINS`: it is not an integer.
- `bench_doc_audit.py` runs a `selftest()` on every invocation pinning the line
  shapes its tokenizer must recognise. Without it a regex regression is silent:
  the audit recognises fewer labels and still prints "0 mismatches". That is how
  26 riir-chain benchmark docs audited as clean while being unreadable
  (`.docs/10_audits/sibling_doc_drift_auditors.md`).
- `percentile_floor_gate.py` is the GATE over
  `percentile_index_audit.py` (pins in `scripts/percentile_floors.txt`),
  katgpt-rs-scoped like the cfg-gated one and separate from its report for the
  same reason. What it buys: a new site whose percentile index lands on `n - 1`
  reds the push that adds it, **before** that number is quoted in a
  `.benchmarks/` table as though it were a tail — print-only or asserted, a
  misleading number in a benchmark doc is the input to somebody's
  promote/demote decision. It imports the report rather than re-implementing
  the tokenizer, and runs the report's `selftest()` first, exiting **2** if the
  instrument itself is untrustworthy — a distinct outcome from a moved pin.
  `min_sites_scanned` is a **FLOOR** for the reason spelled out in that file:
  the three ceilings are green over whatever the vocabulary can NAME, so a
  tokenizer regression takes the population to ~0 and every ceiling passes,
  indistinguishable from a clean repo. Canaried in both directions before
  landing (a planted degenerate site → exit 1; the floor raised above the
  measured population → exit 1).

### The docs gate covers ONE repo — two more tiers cover the rest

Both auditors accept a repo path and audit any repo, and for months nothing
pointed them anywhere but here. A sibling with stale labels then looks exactly
like a sibling nobody has checked. Three tiers now, deliberately different
cadences, and none of them subsumes another:

| instrument | where | cadence | scope |
|---|---|---|---|
| `docs_gate.yml` / `docs_gate.sh` | CI + workstation | per-push | katgpt-rs only |
| `sibling_docs_drift.yml` | sibling CI (reusable) | caller's choice | one caller |
| `scripts/docs_drift_sweep.py` | workstation | on demand | every contract repo |
| `scripts/numbering_drift_sweep.py` | workstation | on demand | every contract repo |

**The same one-repo blindness was measured a second time, on a different
instrument, 2026-09-05 (`.issues/725`).** `numbering_gate.py` also accepts a
repo path and had also never been pointed anywhere but here — and katgpt-rs,
the one gated repo, was the only clean one: **35 tracked duplicate numbers
across riir-train (13), seal-game-editor (12), riir-ai (6) and riir-clippy
(4)**, in allocator-serial directories where `Plan N` now resolves to two
documents. Plus a defect class the local gate could not have: **five
`.highwater` files that are not integers at all**, every one of them `echo -n
<N> > .highwater` under a shell whose builtin `echo` ignores `-n`, so the flag
lands in the file (`-n 872`). `scan()` swallowed the `ValueError` and returned
`None` — *which is also what an ABSENT allocator returns* — so the
above-highwater ceiling passed over a corrupted allocator and a clean directory
identically. Repairing one of them **immediately exposed a stale allocator
underneath it** (riir-train `.plans` max 375 > 374) that had been invisible for
as long as the file was corrupt. All 7 allocator defects are repaired and
pinned at 0. The 35 duplicates were then RESOLVED DOWN TO 12 the same day
(Issue 725 T4b/T4c): **riir-ai 6 → 0** (T4a's `scripts/citation_weight.py`
attribution instrument — by-name citations are 0-2 per side and TIED in four
of six pairs, so `Plan N` mentions are ATTRIBUTED by token overlap with an
UNRESOLVED bucket and a zero-is-not-zero guard measured on `.plans/229`;
175→568, 182→567, 229→566, 313→569, R020→362, R148→363), **riir-clippy 4 → 0**
(its Issue 069, `58e7c1d`), **riir-train 13 → 0** (its Issue 514, `103ed351`
— hand reads overturned the four UNDECIDABLE rows). The remaining **12 are
all in read-only seal-game-editor**, ratcheted at the measured count; the
lesson over the pessimism: the instrument is advisory — the corpus does not
distinguish near-synonyms, but reading the actual sites does.

Both sweeps are **deliberately not** in `docs_gate.sh`'s `CHECKS`: CI has a single
checkout, so it would derive an empty population and print a confident green
over zero repos. Its population is derived (BOUNDARY.md + a `.git` dir); its
expectations are committed (`scripts/docs_drift_floors.txt`), because deriving
both from the same walk is what makes a cross-repo gate permanently green.

`sibling_docs_drift.yml` is `workflow_call` so the three easy-to-get-wrong facts
live once rather than once per sibling — the worst being that the auditors
default to auditing **katgpt-rs**, so a sibling that omits the path argument
passes forever against the wrong repository. The workflow asserts the audited
tree is the caller's.

`scripts/ci_gate_coverage.py` is a **report, not a gate** (always exit 0): which
of the derived contract repos actually gate their full compile+lint surface in
CI, following each workflow into the scripts it calls, **and whether anything
automatically starts it**. Run it instead of re-typing the answer — that
question has been answered by hand twice and been wrong both times
(`.issues/701` R2).

Its own join is pinned by a `selftest()` that runs on every invocation, five
shapes, and the pin was canaried by reintroducing the bug it exists to catch —
a first cut asked only whether *any* scheduled workflow carried a cargo signal,
and a data-borne mention in `riir-chain`'s scheduled `toolchain_drift.yml` was
enough to vouch for a dispatch-only `rust.yml`. A weak automatic gate must not
speak for a strong manual one.

### A green test count can be a count of nothing — `scripts/cfg_gated_target_audit.py`

A test file opening with `#![cfg(feature = "x")]` compiles to an **empty
binary** when `x` is off. Cargo prints `running 0 tests` / `ok. 0 passed` and
**exits 0**, which is byte-for-byte a real pass. The `#![cfg]` protects the
**count**; `required-features` protects the **reader**. Both are needed, and
only the second one is visible to whoever reads the output.

Do not answer "how much of this is affected" by reading manifests. Run:

```bash
scripts/cfg_gated_target_audit.py            # all contract repos (derived)
scripts/cfg_gated_target_audit.py ../riir-ai # or one, by path
```

A **report, not a gate** (always exit 0), for the same two reasons
`ci_gate_coverage.py` is: a `cfg` on `target_os` / `miri` genuinely cannot be
expressed as `required-features`, and neither can an `any(...)` of features
(cargo's is AND-only). Those are reported as their own classes — a report that
cries wolf on the shape cargo cannot fix gets ignored on the ones it can.

**The sibling instrument answers the next axis down: is a specific target
named by ANY committed suite?** `cfg_gated_target_audit.py` classifies how a
target compiles; `scripts/suite_membership_audit.py` (first census
2026-09-04, `.docs/10_audits/suite_membership_census.md`) reports which
`[[test]]` targets no script/workflow names — the "gate nobody runs" class
(865, 868) as a standing census instead of an ad-hoc hand grep. "Unpinned"
means unnamed, not broken; the actionable cut is load-bearing + unpinned +
default-visible + no broad `cargo test` run in the repo, and on the first
census that cut lands exactly on the two documented populations (this repo's
723 and riir-train's 507). Run it when landing a new gate: if nothing names
it, either add a suite row or record why not.

**Read the severity split, never the pooled total.** A target gated on a
*default-on* feature still runs on a plain `cargo test` and only vanishes under
`--no-default-features`. A *default-off* one reports a green zero every time
anyone names it. Pooled, the first measurement read 702 and meant nothing; split,
it is **382 SILENT-NOW / 320 latent** across 19 repos, a large minority of them
load-bearing by name (`goat`, `gate`, `g<N>`, `drill`, `proof`, …). Full record
and the per-repo table: `.docs/10_audits/cfg_gated_silent_zero_pass.md`.

Do not re-type those numbers from here — `.docs/10_audits/cfg_gated_silent_zero_pass.md` carries **two** same-day
corrections. The first cut over-counted by 48, keying declared targets by name
against the filename stem, so a row with an explicit `path` read as undeclared.
The second was in the *load-bearing* classifier built for T4: a token matcher
written independently returned 87 where the published table's ad-hoc substring
grep said 93, and **five of the six disagreements were the token matcher's
misses** (`g16f`/`g2p`/`g9gov` — G\<N\> with a variant suffix; `drills` — a
plural; `regate` — a compound). It now reproduces the table exactly, per repo.
Two independently-built classifiers agreeing is what licenses the
`max_load_bearing = 0` pin; a false negative would have made that pin a
permanent green. Run the script.

**And do not read that agreement as more than it is.** The two classifiers
agreed, and they agreed on the wrong *population* — T4c widened the token set
and 17 more katgpt-rs targets appeared. Agreement licenses the pin against a
classifier **bug**; nothing licenses it against a classifier being
congenitally **narrow**. The defence is the corpus-wide candidate-token table
in `.docs/10_audits/cfg_gated_silent_zero_pass.md` T4c, not a second opinion.

**A third kind was added 2026-09-03, and it is the one a green `w/ req-f`
column lies about.** The report modelled feature-expressible gates and
`target_os`/`miri` ones. `debug_assertions` was pooled with the latter, and
that hid the worst case in the set: every *other* predicate is silent only in
a configuration somebody **chose** (the wrong platform, miri,
`--no-default-features` typed), while `not(debug_assertions)` is silent under
plain **`cargo test`** — the default invocation, on the right machine, with no
flags. It also **survives the fix**: adding a `required-features` row moves the
target into "w/ req-f", which reads as protected, and does not make it compile.
So it is reported as an overlapping **dimension**, not a fourth bucket, leaving
the partition assertion untouched. Measured over 19 repos: **133 targets, 29
load-bearing, 11 already "covered"** — and split by direction, since pooling
would repeat the error that report already documents for platform gates:
**130 `not(debug_assertions)`** (green zero on `cargo test`; 26 load-bearing,
almost all riir-ai GPU benches) against **3 bare `debug_assertions`** (green
zero under `--release`) — and *all three of those are load-bearing alloc
gates, which vanish exactly when someone follows the rule above to run gates
in release*. The instruction and the gate were in direct conflict and nothing
said so. Found by riir-ai `.issues/855` Class 2, whose fix pattern for the
conflict is to split the alloc assertion from the wall-clock one, since no
single profile can observe both.

The verdict half is `scripts/cfg_gated_floor_gate.py`, a docs-gate check. The
report and the gate are deliberately separate files: the report must stay
runnable over the siblings whose owners have not taken Issue 713 T3, and a
report that exits 1 on them is a report nobody runs.

It was fixed one target at a time twice in one week — riir-train `5821cba9`
(11 real assertions reporting as a green suite having run none) and riir-clippy
`19beece` — before anyone asked how many there were. Fixing them one at a time
is how it stayed invisible.

Adding the rows is safe and does **not** red an existing CI: `cargo test
--workspace` silently *skips* a target whose required-features are off. What
changes is that naming the target without its features errors with exit 101
instead of reporting a green zero. That was verified, not assumed, before the
katgpt-rs batch landed (`180be9c5`, 39 GOAT gates, SILENT-NOW 102 → 63, baseline re-measured with the
corrected auditor rather than inferred from the delta).

**Run the armed gates with `--release`.** All 39 pass there. A first sweep
without it reported four reds and nearly filed two of them as perf
regressions: a latency gate in a debug build measures an unoptimised binary,
and `fast_bpe_goat` is 388 s in debug against 15.6 s in release. Re-measuring
three times on a quiet box gave a sub-1% spread and made the wrong number look
*more* trustworthy — ruling out the confounder you thought of says nothing
about the one you didn't. Arming the gates still paid: it surfaced
`.docs/10_audits/alloc_gate_per_thread_counter.md`, an alloc gate counting a sibling test's allocations, which
reproduces in release.

### A `required-features` row can EXIST and be WRONG — `scripts/required_features_build_audit.py`

Every audit above treats a target as protected once it **has** a
`required-features` row. That is the right check for the failure they were
built for. It says nothing about whether the row is *correct*, and a row that
exists and is wrong is strictly worse than a missing one:

- `cargo test --workspace` silently **skips** the target (features unmet), so
  nothing reds.
- `--all-features` **builds** it, because the union supplies whatever the row
  forgot — so the one configuration anybody runs it in passes.
- Every audit counts it in the "w/ req-f" column, i.e. **protected**.

It cannot be answered statically: the row is wrong relative to what the file
*imports*, and the import resolves through `lib.rs` re-exports that are
themselves cfg-gated — the glob-re-export problem this repo already documents
as defeating grep. So the check is to ask the compiler, once per target:

```bash
scripts/required_features_build_audit.py --list            # rows only, no builds
scripts/required_features_build_audit.py ../riir-train     # one repo
scripts/required_features_build_audit.py . --grep pruners  # one slice
scripts/required_features_build_audit.py ../riir-train --batch  # 1 run per set
```

A **report, not a gate** (always exit 0) and, unlike its siblings, also for a
cost reason: **1,829 rows over 16 repos** (2026-09-05), at ~28 s/row warm on
the first katgpt-rs slice — a full sweep is hours, so it is filterable
(`--package`, `--kind`, `--grep`, `--limit`) and takes `--target-dir` for when
a sibling session is building.

`--batch` is the cost lever, and its constraint is what makes it sound:
**one cargo run per (package, EXACT feature set)**, never per subset. Building
a target at a SUPERSET of its own row and seeing it succeed proves nothing —
the extra features may supply the very import the row forgot, which is the
failure the whole report exists to catch (`--all-features` builds every wrong
row). Equality is therefore the only batchable relation, and it is worth
batching: **1,829 rows collapse to 1,070 groups (1.71x)** cargo invocations.
Do not read that ratio as the speedup. Measured on riir-clippy (44 rows / 25
groups, cold dirs, two pairs run in both orders): **-11% and -8% CPU-seconds**,
while **wall-clock flipped sign** (-12%, then +13%) on a box with sibling
builds live — 19 saved invocations over one package whose feature sets share a
dependency graph is not separable from load. The mechanism that pays is the
per-invocation resolve, so expect more where a feature set swings the graph
(riir-train's CUDA sets) and near-nothing where it does not. Verdicts stay
per-target — attributed from `--message-format=json`
`compiler-message`/`compiler-artifact` target names, with `--keep-going` so
one red target does not truncate the run. A row with **neither** an error nor
an artifact reports **UNSEEN, never BUILDS**: silence is not evidence, and
reading it as success would be the same green-zero this family exists to
refuse (`attribute()` is pinned by `selftest()` in both directions, canaried
by reintroducing exactly that collapse). It warns when another cargo holds the repo's
`target/`, by working directory, for the reason
§"Several sessions, one target dir" gives.

Two instances so far, and the second is the argument for the instrument: the
first was found by hand (riir-train `9da3420f`, `test_cubecl_backward_grads`
omitting `gpu_training_resident`; fixing it immediately reported 9 passed /
1 failed, so the wrong row was hiding a real defect). The second was found by
this script **in the first six rows it checked, in this repo** —
`bench_001_pruners_goat`, `["bomber", "go"]`, `E0432`. Its twin
`bench_001_pruners_goat_proof` had the identical defect *and it was already
fixed*, by Issue 723 T7, which did not look at the file next to it. Five
assertions that had never been runnable at their own row now run and pass. A
hand fix of one instance does not close a class.

**One of its three verdicts is free.** A row naming a feature the package
cannot enable is decided by the manifest alone, so it runs as a static
pre-pass before any build: **0 invalid rows over all 1,829, in under a
second** (2026-09-05). Do not re-derive that check by hand — the obvious
model is wrong. `required-features` accepts **`dep/feat`** and `dep?/feat`,
naming a DEPENDENCY's feature rather than one of ours, and a first cut that
treated those as undefined reported 10 riir-ai benches as dead targets. A
`/tmp` probe with a `compile_error!` canary inside the target (cargo 1.98.1)
settles it: cargo satisfies such a row via a package feature that enables it,
by naming `dep/feat` directly, and under `--all-features`, and skips the
target only in a plain no-features build. Renamed (`package = `) and
`[target."cfg(…)".dependencies]` entries count as dependencies too.

Record and the open sweep: riir-train `.issues/513`.

### A reported "p99" is often the MAX — `scripts/percentile_index_audit.py`

`sorted[(n as f64 * 0.99) as usize]` and `sorted[n * 99 / 100]` both land on
`n - 1` — the **maximum** — for every `n <= 1/(1-p)`: n ≤ 100 at p99, n ≤ 20 at
p95, n ≤ 1000 at p999. Below that boundary the site reports one observation
under a percentile's name. A `.min(len - 1)` clamp prevents a panic, not a
wrong statistic.

Direction matters and is the opposite of the usual worry: the naive index is
one rank **too high**, so a `p99 < budget` assert becomes *stricter*. The
failure mode is a false **RED**, not a false green — nothing currently green
turns red by fixing a site. What it costs is the ability to notice a real
regression, because the tail is decided by a single sample.

The quantity nobody prints is **tail support** = `n - idx`, the number of
samples at or above the reported rank: **1 at n=100**, 2 at n=200, 10 at
n=1000. The report calls anything under 10 weak whether or not it is
degenerate.

```bash
scripts/percentile_index_audit.py             # all contract repos (derived)
scripts/percentile_index_audit.py ../riir-ai  # or one, by path
```

A **report, not a gate** (always exit 0), for the reason
`cfg_gated_target_audit.py` is: half the sites take their sample count from a
runtime length or a fn parameter that no static pass can reach, and a report
that exits 1 on those is a report nobody runs. Read the split — **UNRESOLVED
is not "clean"**, it is "needs a per-site read".

Do not re-type its numbers from here — the durable record with the per-repo
table, the commit evidence and the open sibling rows is
`.docs/10_audits/percentile_index_tail_support.md`. Measured
**2026-09-03 evening: 126 sites over 9 of 19 repos — 0 DEGENERATE, 2 TRUNC-VAR,
6 WEAK, 31 OK, 62 UNRESOLVED, 25 SAFE.** The DEGENERATE zero is real and
earned: four owners fixed all 12 the same day (riir-ai `03a91ed59` swept 10,
riir-mmorpg-examples `ee9da24` the one DEGENERATE-**ASSERTED** site,
riir-game-sdk `f896bca`, riir-chain `7f3a3910`).

**And the fix blinded the gate, which is the part to remember.** All four
repairs consolidated the arithmetic behind a `nearest_rank(sorted, p)` helper,
so `p` became a **parameter** — and every pattern in the vocabulary required a
*literal* p. Sixteen sites left the population (130 → 114); `max_degenerate = 0`
then read green over a population that no longer contained riir-ai's percentile
surface at all, and **seven byte-identical copies of that helper across five
repos** were invisible. Fourth instance of the classifier-narrowness failure
below, first one reached by a *correct* fix. Closed by a fifth verdict,
`TRUNC-VAR` (a truncating variable-p rank inside a percentile-named scope — a
finding without a resolvable n, since `floor(p*n)` is the max for every
n ≤ 1/(1−p) whatever p is), a fourth ceiling `max_trunc_var = 0`, and a
`.trunc()` hole in the rounding exclusion that had been clearing the defect's
own second spelling as SAFE.

Two corrections to the reasoning above, both in that record: **"false RED, not
a false green" is assert-direction dependent** — it holds for `p99 < budget`
and inverts for a `p95 >= floor` diversity row, where a too-high tail is a
false GREEN — and **`asserted` is structurally blind for helpers**, since
`is_load_bearing` is deliberately same-fn-scoped while a helper sits one call
frame from the assert it decides. That is why `TRUNC-VAR` is the one class
gated regardless of `asserted`.

**This file's own history is the lesson about classifiers.** The audit has been
wrong once per vocabulary gap — never by a bug — and every time the narrow
version looked like good news. (The count is deliberately not written here; the
list grew a fourth entry the same day this sentence would have said "three".)

1. The first cut grepped only the **float** forms and published a 14-row hand
   table as an audit of "all 19 contract repos". The integer form
   `n * 99 / 100` is the more common one here and was invisible; riir-e2e's
   copy was found by accident.
2. Both `resolve_n` and `is_load_bearing` were **file-scoped**, so a binding
   or an `assert!` in a *different function* counted. That manufactured a
   false ASSERTED (riir-neuron-db `bench_003`, where the assert is on
   `mean_us` — the p99's neighbour in a returned tuple) and sized a slice
   *parameter* from an unrelated caller (riir-chain `bench_012`).
3. The literal-only pattern reported **riir-game-sdk as having zero sites**
   while its `percentiles` helper — `|p: usize| durs[(n * p / 100)...]`,
   called as `at(50)`, `at(99)` — feeds that repo's wall-clock budget gates.
   A variable percentile is not statically known, so it must land in
   UNRESOLVED, not vanish.
4. The literal-only patterns were blind to a variable-p **helper body**, so
   the 2026-09-03 repair campaign — which consolidated twelve defective sites
   behind seven byte-identical copies of `nearest_rank(sorted, p)` — removed
   them from the population instead of moving them to SAFE, and the ceiling
   read green over the gap. Closed by `TRUNC-VAR` + a scope-name
   discriminator measured against all 27 candidate sites in the workspace
   (admits 8, rejects 19, including the two a bare `rank` substring would
   have swallowed). **A correct fix caused this one** — which is why the
   defence is the corpus-wide candidate table, not a second opinion.

So the vocabulary is **data** (listed exhaustively in `VOCAB`) and the
population is **derived** (BOUNDARY.md + a `.git` dir), per the workspace rule
that deriving both from one walk is what makes a cross-repo report
permanently empty. `selftest()` runs on every invocation and **exits 2 rather
than printing**, pinning the tokenizer, the scoping, the rounding exclusion
and the arithmetic. It was canaried by reintroducing bug 1 above — a greedy
character class that swallowed `sorted[(n` — which took every site to
UNRESOLVED; without the selftest the run would have printed 130 sites, zero
findings, and read as a clean repo.

A third "exclusion" was **not** deliberate and has been fixed: `.trunc()` sat
in the rounding exclusion beside `.ceil()` / `.round()` while the same
comment called truncation the bug. `x.trunc()` **is** `x as usize` for
non-negative x, so the defect spelled a second way cleared as SAFE. Latent
when found (zero percentile-context `.trunc()` sites, measured), and
`.floor()` was never in the set — it is the defect's own name. The two below
are the legitimate ones:
`((n - 1) as f64 * 0.99)` is bounded by `n - 2` and can never return the max
(verified over n ∈ 2..=20000; it is the shape
`katgpt-speculative/tests/weaver_real_checkpoint.rs` uses, and the one site in
the workspace that got this right), and `.ceil()` / `.round()` on the product
is the *correct* nearest-rank form — requiring it also removes the largest
false-positive class, `0.95 * n as f32` computing a top-p **nucleus size**,
which indexes nothing.

### Before committing in a shared worktree — `scripts/staged_set_audit.py`

Several agent sessions write into one worktree routinely, and `git add -A` from
a repo root is indistinguishable, to git, from intent. `b2527521` committed
three agents' WIP in six files; one was a build regression nobody saw for a day
(`.issues/709`). Stage **named files** (`git -C <repo> add <paths>`), never
`-A` — and before a multi-file commit, run:

```bash
scripts/staged_set_audit.py            # any repo: pass its path as $1
```

Also a **report, not a gate** (always exit 0). The refusing pre-commit hook was
**decided against** (`.issues/709` T3b, 2026-09-03) on the evidence the report
itself produced: every signal cheap enough for a blocking hook has a
legitimate-use false positive, and the one signal with none (`--fmt`) costs a
rustfmt subprocess per file. A report that is read beats a gate that is
bypassed.
Measuring the signal is not. Three signals, none subsuming another:

1. **mtime clusters** — the technique that caught this by hand twice:
   **worktree mtimes cluster by editing episode.** A 204-file rustfmt sweep
   lands in a 3-second window; your own edits land in the window you were
   running. Two clusters in one staged set means two episodes, and the older
   one is probably not yours.
2. **also-dirty** — a staged path that *still* has unstaged changes: a
   concurrent editor writing into the same file right now, which mtime
   clustering cannot see because their write may land inside your window.
3. **stale-vs-HEAD** — a dirty-or-staged file that LACKS substantive lines the
   newest commit on its own path added. Committing it reverts them. Both other
   signals are blind to this: a whole-repo sweep is *one* episode and its files
   are not also-dirty. Found live on the first run — `tpr/als.rs`, written
   20:04:39 in a rustfmt sweep, while `0ef7f078` landed a 22-line Issue 712
   correctness fix in the same file at 21:08. It audits the **dirty** set too,
   because the hazard exists before anything is staged.

Signal 3 is two-stage on purpose. `mtime < commit time` alone false-positives
on the commonest shape there is — you edit at 21:03 and commit at 21:04, so the
newest commit on that path is your own edit; it flagged two such files on the
first run. Line-set **containment** is the confirmation and is what the warning
claims, restricted to lines specific enough that their absence is evidence (a
`}` proves nothing). A workspace sweep over all 19 contract repos returned
exactly one hazard, with two other repos dirty-but-clean — so it is not
always-on.

4. **rustfmt round-trip** (`--fmt`, Rust only) — `git show HEAD:$f | rustfmt
   --emit stdout | diff - $f`. Identical ⇒ the worktree copy is exactly "HEAD,
   formatted", provably carries **zero content**, and reverting cannot lose
   work. The only signal here that yields a *proof*; the other three are
   heuristics. Diffing against `rustfmt(HEAD)` also **isolates the content**,
   which is the review you want on a file whose real edit is buried under a
   whole-file reformat — `-w` cannot produce it, because rustfmt re-wraps
   tokens *across* lines. On its first run it refuted a standing belief: 15
   files described for hours as "a sibling's rustfmt churn" came back **0
   churn, 15 content**, every one a real lint fix (`989f1bdf`). Its `skip`
   verdict is pinned by `selftest()` — a `churn` verdict authorises a revert,
   so unlike the other three this signal can destroy work if it errs that way.

Single-linkage, not fixed-width bins: a session editing continuously for an
hour is one episode because no two consecutive edits are `GAP_SECONDS` apart.
Fixed bins would split it and false-positive on exactly the sessions doing the
most work. A `selftest()` pins ten shapes on every invocation — the chaining
case and signal 3's line-specificity predicate — because both failure modes are
silent: the clustering degrades to "1 episode, always" and the containment
check to "nothing is ever missing", each printing a confident pass
indistinguishable from a real one.

When you must commit into a file a sibling is editing, commit **your blob**
rather than the worktree's: build HEAD's version + your edit, `git hash-object
-w`, then `git update-index --cacheinfo`. Their hunks stay uncommitted and the
worktree stays coherent for them (used for `bench_707` in `8c7ca74b`).

### Several sessions, one target dir — a gate can produce a FALSE RED

Every axis above is about a gate being *wrongly green*. This one is the
inverse and it costs more, because a red gets investigated: **a count-pinned
or feature-switching gate run concurrently with another cargo process in the
same `target/` reports a failing test that passes when run alone.** Cargo
rebuilds and replaces test binaries while another run is executing them.

Read the failure's **shape** before its content. `error: test failed, to
rerun pass …` with **no `failures:` block and no `test … FAILED` line** means
the harness process *died*; nothing asserted anything. The count is
truncated too, because cargo stops at the first failing binary — so one
environmental cause emits two independent wrong signals, a fake failing test
*and* apparent pin drift.

Measured 2026-09-03 (riir-game-sdk `.issues/023` T4): a row read `105 passed
(pinned 182)` and named `prod_l3_partition_heal`. That suite has no
wall-clock component — seeded deterministic drills — so load flake was ruled
out and a riir-chain comparator change from the day before looked like a
cross-repo regression. It was neither: the freshly built binary run
**directly** passed 6/6. Three `cargo test -p riir-e2e` runs were live on
three different feature sets at once, two of them mine.

Two things follow. **Running the compiled binary directly needs no build
lock**, so it is the way to diagnose while siblings are compiling — find it
under `target/<profile>/deps/`, filtering out the `#![cfg]`-gated copies that
`--list` reports as 0 tests. And a gate whose verdict the box can invalidate
should **refuse**, not warn: riir-game-sdk `scripts/test_gate.sh` now detects
concurrent cargo by **working directory** (`lsof` over `pgrep -x cargo`)
rather than by command-line pattern — a pattern is blind to a plain `cargo
build` in the same dir and conversely matches harmless sibling-repo runs
whose command lines mention our crates. A lock-based check cannot work at
all: cargo releases `target/<profile>/.cargo-lock` *before* running the test
binaries, which is exactly the damage window.

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

**Lossy-surface promotion rule (adopted 2026-08-28, Issue 750 T3):** a
promotion of a **lossy** surface (quantization, compression, any bit-changing
transform) gates on **deployed-path behavior — per-family, conditional
retention**, not on bit-identity or aggregate perplexity alone: bit-identity
is only available to lossless surfaces. Three independent arrivals at this
rule: Research 502 ("Behavior Before Perplexity"), Bench 696 (the KVarN
sink-guard GOAT), and Issue 750's measured bisection (gemma-2-2b Q4_K:
first behavior flip at prefix k=1 — layer 0 alone flips the sealed family;
restoring it costs 106.7 MiB, priced by the T2 override probe). Aggregate
perplexity can be flat while family-conditional behavior flips.
External measured confirmations (riir-clippy Research 125, walk #7):
arXiv 2609.01962 (post-training ternarization of Qwen3-4B — aggregate
accuracy 64.5→54.7% yet per-task chance-corrected retention spreads
84.6% BoolQ vs 43.8% ARC-Challenge; and a lossless packing run holds PPL
while a lossy one was excluded from the artifact claim) and arXiv 2608.12700
(a contract-grade verifier rejects 1,487/2,638 kernels a standard tolerance
harness accepted — the tolerance-budget fault-class confirmation queued at
walk #6).

**UQ-bearing primitive GOAT gate extension (the "Report the Floor" rule, adopted 2026-06-28 per Research 322 / Plan 340).** Any primitive that claims a probability distribution, predictive interval, quantile, coverage guarantee, confidence score, or calibrated uncertainty (collectively: **UQ-bearing**) MUST benchmark against the **conformal-naive floor** — `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340 with `m=1`, plain split conformal) — on CRPS / coverage / Winkler score. If the primitive cannot beat the floor, the GOAT gate FAILS. Existing UQ-bearing primitives (BoMSampler Plan 281, Sleep-Time Anticipator Plan 334, Best-Belief Beta Selector Plan 336, KARC+overlay) are grandfathered but must include the floor at their next re-gate; future UQ primitives must include it from the initial gate. Tracked in `.issues/010`. The floor shipped in Plan 340 Phase 1 (2026-06-30); the rule is now enforceable. **Issue 010 is FULLY CLOSED (T1-T7 all complete)** — see `.benchmarks/010_report_the_floor_consolidated.md` for the cross-primitive summary. **T7 (2026-07-20)** added the KARC+overlay dedicated floor test (`conformal_floor_karc_overlay.rs`) — the composite is SCOPE-LIMITED to chaotic regimes (BEATS on Lorenz-x at crps_ratio 0.0047 with K=4; LOSES on stationary seasonal at crps_ratio 5.74 with K=4), but coverage stays calibrated on both — no false-confidence signature. **T7 K-sweep (2026-07-20)** refuted the prior "K=4 too shallow" hypothesis: K=12 (matching the period) LOSES WORSE on seasonal (CRPS 5.74 → 20.26) and WINS HARDER on Lorenz (CRPS 0.0047 → 0.0018) — the scope-limit is **structural** (KARC's Chebyshev basis + ridge-fit doesn't fit periodic data regardless of K), not parametric. Production guidance: pick K by chaotic-regime memory needs; for periodic data use the floor directly.

**Plan 467 / Proposal 007 (2026-07-18):** Shipped `DualLeoOracle` as QGF's 3rd `QGradientOracle` impl — fuses a LEO teacher head + UVFA student head via `DualLeoMixer::combine_into` at the gradient level. Sibling to `LeoHeadOracle` (Plan 268) + `FlowFieldOracle`. G1–G4 PASS mechanistically; **G5 measured FAIL on synthetic data (riir-ai Bench 553, 2026-07-18): dual 0.00% vs single 0.50% on T7 Go puzzles, but the correctness invariant (QGF+LeoHeadOracle ≡ baseline) held bit-identically — mechanism correct, quality gate FAILs because synthetic data produces near-flat Q-fields.** **G5 also measured FAIL on civ real networks (riir-ai Bench 558, 2026-07-19): dual +2.69% vs single 35.68% → 36.64% on civ action-prediction, ≥3% gate — fourth-axis stop rule.** The civ dual-LEO investigation is fully closed per riir-ai Research 322 (the "alternative critic" escape hatch was category-confused — UQ primitives produce state forecasts, not per-action Q-gradients). The Plan 460 max-pool washout lesson is encoded as a design invariant (no operator between mix and consumer). Stays opt-in (`qgf_oracle + dual_leo`) with documented unproven G5 across both synthetic and civ real-network regimes; reopens only on seal integration gain, new game domain positive G5, or Q-vs-forecast research breakthrough.

## Research Workflow

See `.agents/skills/research/SKILL.md` for the full research workflow:
paper classification, 7-repo routing, fusion-first distillation, novelty gate,
GOAT gate, and the mandatory modelless-unblock protocol (§3.5).

## Substrate-First Gate (MANDATORY before implementing)

Before implementing ANY new System impl, trait, perception/cognition/emotion
pipeline, state management, spatial query, or vocabulary type, run the
`.agents/skills/substrate-first/SKILL.md` skill. It enforces:

1. **Vocabulary translation** — grep 3+ name variants (concepts ship under
   operator names like `GenericSpatialBelief`, not English names like "threat
   field"). A single-vocabulary grep returns ZERO hits even when substrate
   fully exists.
2. **Codebase grep** — search `*.rs` source across all 8 repos, not just
   `.plans`/`.docs`/`.issues`.
3. **Architectural rule check** — domain classification, two-brain model, sync
   boundary, bridge pattern.
4. **Consume vs. build decision** — if substrate exists, consume it; if not,
   file an issue in the right repo FIRST.

This prevents the recurring drift pattern where an agent builds a parallel
system that duplicates already-shipped substrate under a different name
(canonical failures: ThreatField Issue 047, orchard/motivation Issues 490/493).

> **Repo count:** the **product/distillation set is 7** — `katgpt-rs` (public) +
> `riir-ai`, `riir-chain`, `riir-neuron-db`, `riir-train`, `riir-game-sdk`,
> `riir-dapps` (private). That is NOT the repo total: the
> workspace is **16 repos**, all of which carry a root `BOUNDARY.md`
> (add `riir-mmorpg-examples`, `riir-clippy`, `riir-viewbridge`,
> `riir-auth`, `katgpt-web`, `riir-dao`, `riir-deployer`,
> `seal-game-editor`, `seal-remake`).
>
> **19 → 16 on 2026-09-04 00:01**, by the same owner act that retired
> `riir-armageddon`: `riir-burner`, `riir-unity` and `seal-remake-unity` were
> moved to `/Users/katopz/git/obsolete/`. Nothing was lost — the directories
> are intact there and `riir-burner`'s last sweep was pushed (`ce54122`) 19
> minutes before the move. They are named here as **lineage only; do not route
> work to them**, exactly as with `riir-armageddon`. This time the drift was
> caught by an instrument rather than by reading: `skill_repo_set_gate.py` went
> red on the stale `scripts/repo_set.txt` within the hour, and
> `agents_repo_set_gate.py` then went red on this paragraph — which is the
> membership-first gate doing precisely the job it was written for, on its
> second day. The cross-repo **edge** count WAS re-measured post-retirement
> (2026-09-04, boundary-guard log): the C0e-landing contract run printed
> **exit 0 — 16 repos / 224 edges**, and the 23rd run re-confirmed 224 via
> `--list-deps` (its full run exited 1 only on the worktree-only C6 artifact
> from the sibling's 461-file rustfmt sweep — all five keeper pins exact at
> HEAD, proven formatting-only on a clean /tmp checkout). Live count drifts
> with landings — this session's `--list-deps` already reads 226 rows after
> the Issue-092 ndb-sdk edge landed — so don't re-type a number, run:
> `../riir-ai/scripts/ci_boundary_contract.sh --list-deps`.
>
> **This paragraph said 8 and 18 until 2026-09-03, and the way it was wrong
> is worse than a stale number.** `riir-armageddon` was de-enrolled by an
> owner act — the directory is GONE — and `seal-remake-unity` was enrolled
> in the same window (boundary-guard's 18th run: 19 repos / 225 edges, the
> 227→225 delta being exactly armageddon's two allowlist edges leaving with
> the directory, the new repo adding zero). **One repo left and one arrived,
> so the total stayed 19 while the MEMBERSHIP changed** — every count in
> sight agreed, and the set was wrong anyway. A count is not a checksum over
> a set. The derived instruments were all correct throughout
> (`scripts/repo_set.txt` was regenerated at `d2cb9979`, and the repo-set
> gate is what caught the snapshot mid-session); only the prose was stale,
> which is precisely the split the command below exists to enforce.
> **Re-measured 2026-09-01** by
> `../riir-ai/scripts/ci_boundary_contract.sh` — *"boundary contract clean —
> 18 repos, 211 cross-repo dep edges measured"*. It was 15 at the 2026-08-21
> run recorded here before; the count moved because contracts were added, not
> because repos were, and **this paragraph did not** — which is the failure it
> warns about, committed by the paragraph itself. **19 later the same day**,
> and this time because a repo genuinely was added: `seal-remake` was
> scaffolded at 23:41 with a root BOUNDARY.md (derived, not re-run through
> `ci_boundary_contract.sh` — so the 211-edge figure above is NOT re-measured
> and should be read as of the earlier run). `scripts/repo_set.txt` was
> regenerated the same hour (`01e19858`) because the docs gate went red on the
> drift, which is the instrument working. Don't re-type the number:
>
> ```bash
> cd /Users/katopz/git && for d in */; do
>   [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && echo "${d%/}"
> done
> ```
>
> Four of the 2026-08-21 repos had no contract at all until that run, and
> `riir-armageddon` had been consuming `riir-games` + `katgpt-core` unaudited
> (those are the two edges whose departure the 227→225 delta measures — the
> repo is retired, so this sentence is history, not a live gap).
> Read a count in prose as a claim, not a fact — and read a count that
> MATCHES as a claim too, per the membership swap recorded above.
> The historical "5-repo quintet" terminology referred to the 5 distillation
> targets (katgpt-rs + 4 riir-* siblings); `riir-game-sdk` (game vocabulary
> facade + dev-tool workspace) and `riir-armageddon` (arena/game-product domain
> types) were added later, and `riir-dapps` (the dApp layer — game outcome →
> generic chain settlement) on 2026-08-20. **`riir-armageddon` has since been
> retired** (2026-09-02, owner act) — it is named here as lineage only; do
> not route work to it. See Research 003 for the canonical boundary.
>
> **Two axes, not one.** Research 003's repo table is the *public/private*
> axis; its §"The Second Axis: Layering (game / dApp / chain)" is the
> *layering* axis — which private repo a game concern goes in. **Three tests,
> all must pass** (revised 2026-08-20; the earlier one-question form admitted
> FAME as "value" and ignored write rate):
> **(1) Product** — would a commerce customer of the chain want this in their
> dependency? An NFT is a token, so yes; a quest, no. **(2) Value** — BigInt
> fungible currency, a token, or an authority binding? FAME / XP / items /
> reputation are game scalars, not money. **(3) Rate** — does it fit a Glacial
> tier (≤0.1 Hz)? Binds hardest; `riir-neuron-db` is 1,627× cheaper per write
> and one chain tx at 10⁵ accounts eats 63% of a 20 Hz hot tick.
> Canonical failure: game rules (quest / bounty / crafting / reputation, two of
> them moving no money at all) shipped inside `riir-chain`'s
> consensus-critical program set — `riir-chain` Issues 096 + 097, closed on the
> layering side by `riir-dapps`.

## Issue log (resolved)

- **Issue 727 — SP-KV misses BOTH T16 bars once the gate is measured at a realistic sequence length** RESOLVED
  (2026-09-05 `adbc003d`; filed by 723 T7; file removed per noise-reduction — full narrative in git history).
  The repaired instrument (T_N decoupled from `Config::micro()`'s `block_size = 16` — the "50% pruned" arm
  had been pruning 0/16) measured gate-bias overhead +8.0/+8.1/+8.4% vs a <3% bar and prune-skip
  1.046/1.042/1.015x vs a >1.05x bar. T2 hoist landed: `attention_head_core` split into a verbatim NoBias
  impl + a hoisted GateBias impl (64-position chunk scan → active (position, bias) pairs on the stack, zero
  alloc) — prune-skip **1.12–1.58x PASSES its bar** at t_n 128/512/2048 ×3; bit-identity at `to_bits` across
  6 bias cases × 2 head offsets (two documented divergences are unreachable via `build_gate_biases`).
  **T1 verdict: "zero-overhead gate bias" is a false claim** — any gated attention reads the gate once per
  position per head; restated as a measured +7–12% budget (hd=4) in the bench provenance, the primitive doc
  and both Cargo.toml comments; `#[ignore]` KEPT with the updated reason (the issue's second T4 branch — no
  bar re-pinned, G3 held). En-route catch: a single `#[inline(always)]` body measured a **1.66x layout
  penalty on the NoBias baseline**; the `#[inline(never)]` dispatcher split restored it. Record:
  `tests/bench_sp_kv.rs` provenance + the `katgpt-kv` `sp_kv` feature comment.
- **Issue 726 — `gauge_rebalance` is 3.7x its Plan 279 target; the rank-wise accumulate is scalar** RESOLVED
  (2026-09-05 `d225fffa`; filed by 723 T7; file removed per noise-reduction — full narrative in git history).
  T1 priced the scalar accumulate at **~77–78% of the whole call** (stub A/B, 3 interleaved runs). T2 swap to
  `katgpt_core::simd::simd_fused_scale_acc`: t08 best-of-200 **19.21 → 9.00 µs (−53%)**, interleave medians
  54.1/52.7/50.9% (G1 ≥20% ×3 PASS). T3 bit-compared via `to_bits`: **the scalar loop was NOT
  FMA-contracted by LLVM**, so the swap moves results ≤1 ULP (max 1.19e-7) — every exactness assertion passes
  at its committed tolerance (t01 84x headroom; bench_279 G6 + katgpt-sparse 39/39 re-run); no assertion
  loosened. T4: `t08` re-pinned **30 → 15 µs** with the full provenance block; the 5 µs paper target kept as
  aspiration (the remaining floor is the σ-dots + two full-matrix scales + the final ‖M·v‖ pass, ~5.4–6.5 µs
  — not reachable by this swap alone). G3 zero new alloc. Record: `tests/bench_270_gauge_invariant_goat.rs`
  t08 provenance + the `gauge_invariant.rs` comment.
- **Issue 723 — the first full-workspace EXECUTION is red: 47 targets, six distinct classes** RESOLVED
  (T1–T8 2026-09-04/05; **G1–G6 ALL MET**; file removed per noise-reduction — full narrative in git history).
  Doc-tests GREEN (34 suites / 98 passed / 0 failed — the `--all-targets`-excludes-doc-tests axis added to
  this file); Class C resolved by MEASUREMENT (gates pass at their own committed feature sets,
  documented-expected under `--all-features` unification — the Issue-830 twin, never re-pinned); Class E
  closed per-target at committed features. **The load-bearing refutation: "Class A's reds are partly the
  box" was wrong** — load was the top term in none of the eight wall-clock reds; 5 of 8 were closed by
  REPAIRING THE INSTRUMENT (a vanished denominator printing as 30x, two arms that were not the same
  experiment, loop-invariant inputs black-boxed only in the result, in-clock operand regen, a bar measuring
  the fixture), and the rule to carry forward is **repair the instrument first, decide the disposition
  second** — three of the eight would have been re-pinned to numbers off by 5x/7x/140x. The two genuine
  primitive shortfalls were filed as `.issues/726` / `.issues/727` (both resolved same day) rather than
  absorbed into tolerances. Durable artifacts: `tests/common/ab_timing.rs` (interleaved median-of-ratios +
  best-of-N + loud 0-ns FAIL) and `.docs/10_audits/ci_compile_vs_execute_axis.md`.
- **Issue 725 — the numbering gate covers ONE repo; 35 duplicates and 7 broken allocators sat in the other fifteen** RESOLVED
  (T1-T4c 2026-09-05; file removed per noise-reduction — full narrative in git history). The sweep
  (`scripts/numbering_drift_sweep.py`, workstation-only, derived population, committed expectations in
  `numbering_drift_floors.txt`) found katgpt-rs clean while four siblings carried 35 duplicates + 5 malformed
  `.highwater` files (`echo -n` writing its own flag into the file, disarming the above-highwater check) + 2 stale
  allocators. T1 split ABSENT from MALFORMED in `numbering_gate.py` (`read_highwater()` → `(value, malformed_raw)`,
  selftest case 6 canaries the collapse). T3 repaired all 7 allocator defects, pinned 0 everywhere. T4a landed
  `scripts/citation_weight.py` (advisory arbitration instrument; by-name citations tie while `Plan N` carries the
  weight, so ambiguous mentions are ATTRIBUTED by token overlap on a strict margin with an UNRESOLVED bucket that
  is printed and never folded into a winner; a clean-zero gets a loud warning — `.plans/229` scored 0 while two
  citations existed under different spellings). T4b resolved **riir-ai 6 → 0** (`.plans` 175→568, 182→567,
  229→566, 313→569; `.research` 020→362, 148→363 — 86 citation rewrites; four execution lessons recorded:
  section number beats prose on same-subsystem pairs, third cross-repo documents share numbers, select
  inclusively never exclusively, ties break by creation order per `TIE_FRACTION`), **riir-clippy 4 → 0** (its
  Issue 069 `58e7c1d`, 17 citation rewrites), **riir-train 13 → 0** (its Issue 514 `103ed351`, 14 commits —
  hand reads overturned the UNDECIDABLE verdicts; known cost: number-baked test filenames + ~30 source comments
  stay stale for the next code-touching session). The issue's own author was the ratchet's first catch (a
  `513_` allocation from a stale highwater read went red in minutes; renumbered 514). Remaining: **seal-game-editor
  12, READ-ONLY** to these sessions — ratchet at the measured count, report only. T5 (siblings run the per-push
  gate themselves) deferred `[-]` — `numbering_gate.py`'s pins file is katgpt-rs-scoped; reopen when a second repo
  wants its own per-push gate. Record: the three scripts + `numbering_drift_floors.txt` + this paragraph.
  **The ratchet's next catches landed at this closeout, minutes after the file's removal** — fresh drift the
  sweep found on the verification run, both repaired same-day: riir-clippy 2 stale allocators (`.plans` 85→86
  for Plan 086, `.research` 136→137 for Research 137, both landed unbumped by the prior session — `4db7a18`) and
  riir-train `.issues/511` dual-allocated (the genrm-corpus issue took 511 from a stale highwater read while the
  Sep-04 all-features census held it; citation weight keeps the census — AGENTS.md test-gate row + Issue 513 vs
  zero refs — and the genrm file moved 511→518, highwater bumped — `e938cdc0`). Sweep PASSES at closeout:
  12 tracked duplicates (all seal-game-editor) · 0 stale · 0 malformed.
- **Issue 724 — `.plans/` numbering collisions regrew after a hand-sweep; nothing gated the allocator** RESOLVED
  (T2/T3/T4 2026-09-04 `24e349e9`/`28c353a1`/`322769b2`; **T4b + T1/T5 closeout 2026-09-04 `866df2a7`**;
  file removed per noise-reduction — full narrative in git history). The tracked `449` collision
  resolved by CITATION WEIGHT, not creation order (Poincaré kept 449 with 27 mentions; ActionBridge
  moved to `587`); both loaded allocators re-pinned (`.plans` 587, `.benchmarks` 701). Two standing
  gates landed: `scripts/numbering_gate.py` + `numbering_floors.txt` (duplicate-number, stale-allocator,
  per-dir population FLOOR, tracked-vs-untracked split; canaried five directions incl. exit-2 pins)
  and `scripts/docs_gate_paths_sync.py` (docs_gate.yml's two hand-duplicated trigger `paths:` lists
  must stay set-identical — drift exits 1 naming each side's globs; the workflow's own LF line
  endings preserved through a binary-safe edit). T1/T5 moot: the untracked `075_riir_ai_m3_campaign_*.md`
  vanished from the tree before renumbering; nothing to arbitrate. Both gates wired into
  `docs_gate.sh`'s CHECKS (now 10) with the paths-sync script globbed as its own workflow input
  (44 = 44, the 713/704 trigger-omission class closed for this file). Record: the two scripts'
  docstrings + `scripts/numbering_floors.txt`.

- **Issue 721 — the root crate registers a `#[global_allocator]` as a library** RESOLVED
  (T1/T2/T4 2026-09-03; **T3 2026-09-04**, the owner sequencing call executed; file removed per
  noise-reduction — full inventory in git history). The lib-level
  `#[cfg(debug_assertions)] #[global_allocator]` is now `cfg(all(test,
  debug_assertions))` (this crate's unit tests only, the katgpt-core house
  pattern): no downstream binary receives an allocator from this crate in any
  feature set. Every alloc-gate consumer target across the 4 repos
  self-registers instead — katgpt-rs via the new `tests/common/alloc_tracking.rs`
  module (14 test targets + the kimi example; 12 Issue-682 force-link blocks
  deleted), riir-train (xhc's T4 five-term guard deleted as predicted;
  bench_558/490 own statics), riir-ai (~50 files: engine lib + 20 tests +
  example off force-links; games-civ's `alloc_delta.rs` now ships the allocator
  AND the liveness sentinel the crate never had; quest's
  `not(quest_compression_draft)` guard dropped; poc's dual-profile macro debug
  arm self-registers; agents GOAT inline). Validated: 14/14 katgpt-rs targets
  compile under exact features + bench_271 9/9 / cross_res 1/1 / issue_717 G4
  1/1 / lib 203/0; riir-train G4 1/1 + the formerly-conflicting kimi arm
  compiles; riir-ai forward_base 3/3, gemma4 ring 4/4, cgsp/evpi 392/0, civ
  canary 2/2, quest tpr 10/1i, poc 22/0, agents 3/3. **The conflict class this
  issue documents is closed at the source — a downstream `#[global_allocator]`
  is now always legal; the Issue-682 force-link pattern (`extern crate
  katgpt_rs;` to keep a library shim linked) is dead and must not be
  reintroduced.** Push order note: katgpt-rs landed before the consumer repos
  (the inverse window is inert counters; this order would have been duplicate-
  registration compile errors). `riir-agents`' katgpt-rs dev-dep is now
  unreferenced (removal = owner call, BOUNDARY.md row).

- **Issue 719 — conditioning-consistency audit PoC (`cond_audit`)** RESOLVED
  (T1, 2026-09-03, `995dea6d`) — opt-in `cond_audit` in katgpt-core: paired
  forward (compressed-conditioned vs full-context teacher) → per-junction
  forward-KL → Pinsker `TV ≤ sqrt(eps_KL/2)` + greedy-flip counter +
  calibrated-zero arm; KL delegates to `stale_residual::kl_logits`. G8
  non-vacuity PASS (planted 12-nat corruption → tv 4.97 ≫ 0.05, flips 8/8;
  control arm exactly 0.0); G2 measured 1.487× the paired-forward cost vs the
  4.0 budget. T2–T4 deferred `[-]` — reopen on any semantic
  eviction/windowing PR, riir-train Plan 343 T1.6 (Gemma-4 ring), or
  Research 523 H2O un-defer. Record: `.benchmarks/700_cond_audit_poc.md`.

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
