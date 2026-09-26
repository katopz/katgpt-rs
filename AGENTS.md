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
| **compile vs EXECUTE** | every axis above is about *compilation*. The scoped core (katgpt-rs + katgpt-core `--lib` at default features, count floors) was EXECUTED weekly (`test.yml` + `scripts/test_gate.sh` — schedule RE-ARMED 2026-09-23: the repo is public, Actions minutes free, the 09-09 spending premise gone here; `scripts/test_gate.sh` remains the local develop-work form). ⛔ **katgpt-types joined that population on 2026-09-15 and the reason is this axis one PLATFORM over** (riir-train Issue 549): the `avx2_exp_sum_inplace` n-clamp regression test landed and was executed by NOTHING — full_gate is macOS/aarch64, where the NEON sibling always clamped and the class is invisible, and is compile+lint rather than execute; wasm32_gate builds a different kernel; and this lane, the only executing one and the only x86_64 one, did not select the crate; the other 477 integration-test and 176 bench targets are executed by nothing automatic (ONE exception since Issue 858, 2026-09-21: `belief_drafter_goat` is a test_gate `PERF_ROWS` row — `--release`, `--test-threads=1`, floor 12 — so the arch-conditional g8 dual pin has an executing lane on every arch the gate runs on, aarch64 included), and `--all-features` is not a supported TEST configuration (fixture RNG streams and GOAT calibrations are per-feature). An uninvoked assertion is *unknown*, not passing |

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

⛔ **On a non-macOS workstation, run it as
`scripts/full_gate.sh --allow-partial-platform` — and it is now the ONLY thing
on such a box that reads the consequences of a per-crate change** (Issue 803).
It runs every layer and prints `⚠ full gate PARTIAL` naming, on the FINAL line,
each axis it could not measure; it never prints the `✓ full gate PASSED` line,
which stays reserved for a macOS run with every target installed. Standing: a
workstation SUBSET verdict, the same standing as the eighteen drift sweeps.
Take that seriously rather than as a formality — measured 2026-09-15, `develop`
carried **24 `error[E0560]`** (a field deleted in `katgpt-core` with 24 live
construction sites in the ROOT package's `tests/` and `benches/`) plus **4
`-D`-listed lint errors**, for ten hours, while `test_gate.sh` (203 · 2060 ·
249 · 139, every row at its floor) and `wasm32_gate` were both green. The
per-crate gate that landed it was right for what it changed; nothing read what
it changed for everyone else. ⚠ The flag existed the whole time and printed a
final line byte-identical to a full pass — the deferral rode a Layer-2 line six
hundred lines of build output earlier, which is this repo's own most-repeated
rule broken by the one instrument that is not a sweep.

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

**And `x86_64` is a THIRD one — Layer 2c (Issue 819 *the x86_64 lint lane*,
closed; 819 is held by a second, live document and the disambiguation is
pinned in `scripts/number_collisions_expected.txt`), the axis the wasm32
work named and did not generalise.** The 2026-09-16 execution matrix closed
`compile vs EXECUTE` for x86_64 and left its inverse standing: every lane in
this repo that LINTS compiles the x86_64 arms to **nothing** (Layers 2/3/6 are
macOS/aarch64, Layer 2b and `wasm32_gate` build a third triple, `test_gate`
does not lint and runs at default target-features), and the execution matrix
is the right arch with `+avx2` on and **executes** rather than lints. Measured
the day it was found: **30** findings, every one `unsafe_op_in_unsafe_fn` on
edition 2024, every one in `dash_attn/channel_aware.rs`, against **0** on the
avx2-off arm — the same 0-and-14 double-gating shape Issue 737 measured for
wasm32/simd128, which is why both arms run here too. ⛔ The finding is not the
30, it is the **sibling**: that file carries two transcriptions of one kernel
and only `simd_dot_neon` had the `unsafe { }` block, because the aarch64 arm
is the one a lane compiles. A repair applied to the arm somebody can see is
not a repair; it is a measurement of which arms are visible.
- ⚠ The **triple** is this lane's one real design decision and it is printed
  on the verdict line. 2b can name `wasm32-unknown-unknown` literally because
  there is one; x86_64 has three in play and they differ in `target_os`, which
  is Layer 2's whole subject. Host triple when the host is x86_64, a named
  cross triple otherwise. A green whose triple is not disclosed means
  something different on every box.
- The lane is **`--all-features`**, unlike 2b, so that it is exactly Layer 3's
  feature coverage re-run on the x86_64 arch rather than a new claim — and
  because the four non-`src/` targets each carry a `required-features` row,
  where a hand-typed feature list beside them is this repo's most-repeated
  drift shape.
- The residue pin is the non-`src/` surface **minus** what a named row covers,
  expected EMPTY — pinning the four paths themselves would restate the table
  one line down, and a pin that restates its own input cannot fail.

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

Trigger health: PUSH triggers are MAIN-ONLY + dispatch-only since 2026-09-09
(owner call: no CI on `develop` pushes) — every `push` trigger is
`branches: [main]`. The weekly `schedule:` blocks were suspended the same day
under the Actions spending limit (commented, riir-train 507 precedent) and were
RE-ARMED 2026-09-23 for THIS repo alone — it is public, Actions minutes are
free, so that premise is gone here; every private sibling's schedules stay
suspended under the same spend call. `scripts/ci_gate_coverage.py` reports
which declared triggers can actually fire, per workflow, per repo. Layer 2b
also has its own push lane: `.github/workflows/wasm32_gate.yml` runs
`full_gate.sh --wasm32-only` on ubuntu-latest (Issue 737 T4), MAIN-ONLY — the
lane is host-independent and `--lib`-only; dispatch it manually after a run of
develop work, since no push lane covers develop pushes.

### The x86_64 half of the EXECUTE row — `scripts/x86_64_execution_matrix.sh`

Every axis above is a COMPILE axis, and the table's last row says why that is
not enough. One platform over, it says something sharper: `full_gate` is
macOS/aarch64 **and** is compile+lint rather than execute, `wasm32_gate` builds
a third triple, and `test_gate` — the only executing lane — is four `--lib`
suites with its schedule suspended. So every `#[cfg(target_arch = "x86_64")]`
arm in this repo was executed by **nothing** until 2026-09-16. The first two
cells ever run caught **15** latent AVX2-transcription defects (Bench 800
addendum); the next seven caught **two more** (a new maximum silently
discarded by `argtopk`, and an out-of-bounds 8-wide load past the end of a
slice), a router defect red on every non-macOS platform, and a release-profile
benchmark that LLVM constant-folded away so it reported 0 ns/call
([Bench 806](.benchmarks/806_x86_64_execution_matrix.md)).

```bash
scripts/x86_64_execution_matrix.sh              # the matrix
scripts/x86_64_execution_matrix.sh --libs-only  # skip the root integration cell
scripts/x86_64_execution_matrix.sh --canary     # prove the floors fire
X86_MATRIX_DIR=/f/scratch scripts/x86_64_execution_matrix.sh
```

- **Workstation verdict**, the standing of the drift sweeps. No CI lane and
  none is requested — the Actions spending call stands.
- **REFUSES off x86_64**, exactly as `full_gate.sh` refuses off macOS: every
  arm it exists for compiles to nothing there, and a green run would be a
  green ZERO wearing a matrix. Carries the Issue-734 completion sentinel.
- `RUSTFLAGS="-C target-feature=+avx2"` is **not a tuning knob** — an arm gated
  `cfg(all(target_arch = "x86_64", target_feature = "avx2"))` compiles to
  nothing without it, and the run then exercises the scalar fallback and proves
  nothing. `katgpt-attn`'s `channel_aware.rs` carries exactly that shape.
- ⚠ **`CMAKE_BUILD_PARALLEL_LEVEL=1` on Windows is a LOAD-ROBUSTNESS cap, and
  the first write-up of it named the wrong cause.** Twice, cell 6
  (`katgpt-tokenizer`) died building `highs-sys` with `C1001 Internal compiler
  error` in `<vector>` plus `cl D8040 error creating or communicating with
  child process` — D8040 being cl.exe failing to SPAWN its own child, i.e.
  resource exhaustion, with the C1001 as collateral. The matrix reported it as
  *"died without a failures block — nothing asserted"*: the correct refusal,
  and also a red cell over a toolchain flake in the only lane on this box that
  executes anything.
  ⛔ The obvious hypothesis was cmake's own `--parallel`. **Measured, and
  refuted** (highs-sys build dir deleted between runs): `cmake 6 x cargo 6`
  → 74 passed, 13.8s; `cmake 2 x cargo 2` → 74 passed, 13.9s; `cmake 1 x
  cargo 2` → 74 passed, 15.6s. Every quiet-box run passes at every
  parallelism and the timings are indistinguishable; both failures happened
  while ANOTHER heavy cargo build ran concurrently in a different target dir.
  So the trigger is whole-box concurrent compiler load, the cap only shrinks
  this lane's own contribution to the peak, and **the actual remedy is to run
  the matrix ALONE**. The cap is kept because it costs nothing detectable and
  the peak is the one part of the load this lane controls — not because it
  fixes anything. Applied only on MSYS/MinGW and only when unset.
  ⚠ The causal claim that survives is weaker than it looks: two failures
  under concurrent load, three passes without it, and no experiment isolating
  load itself.
- Population **derived** (any package owning a tracked `*.rs` that mentions
  `target_arch = "x86_64"`), so a new such crate joins by EXISTING. Issue 806's
  own hand-typed cell list named three packages; the tracked tree has six.
  A package reaching those kernels only through a DEP owns no matching source
  — `katgpt-dec` is the measured case — and is added by a **pinned row**, the
  wasm32 audit's BY-DEP distinction rather than a widened grep.
- **Both halves of the profile row, in one run:** the `--lib` cells are debug
  (`debug_assertions` ON is half the reason to execute at all) and the root
  integration cell is `--release --no-fail-fast`. Measured, not preferred:
  `goat_574_clustered_lm_head` ran >20 min in debug without finishing and 39.8s
  in release, and `bench_164_gepa_reflective` fails its own 10% bar at 15.5%
  purely because both sides are unoptimised.
- **Every failure is RE-RUN ALONE, then adjudicated by MEMBERSHIP**
  (`scripts/x86_64_matrix_expected.txt`, a reason per row, refused without
  one). The first integration run was **0 correctness failures and six perf-bar
  failures**, and four runs of that one commit then produced **four different
  failing sets** — so a membership pin alone was the wrong instrument, because
  *a pin file re-typed after every run is a diary, not a wall*, and re-typing
  it is how a real regression gets absorbed as "probably the box again". The
  confirm step is this workspace's own lesson mechanised: a load-sensitive bar
  passes the second time and a real failure does not. Such rows print and
  are never counted, never pinned; it costs nothing, because everything is
  already built and `--exact` makes every other binary run zero tests. What
  survived was three reproducible rows, and none of the three survived
  calibration: `t698_t5_kv_mean_gates` was arch-dependent fixture truth (T6,
  arch-conditional dual pins), and the two perf bars (`proof_g3b_swar_speedup`,
  `t09_throughput_inv_sqrt_16x16`) were ISA-real — NEON/FMLA-calibrated bars
  x86_64 never reaches, plus one stale instrument (t09's `bench_us(3, 20)`
  oscillated 2.1× on a loaded M3; repaired to the Issue-723 `best_of_us`
  harness) — resolved as arch-conditional dual pins (T7, 2026-09-16, Bench 806
  Addendum II). The membership set is EMPTY: every confirmed x86_64 failure is
  unexpected again. A stale pin (its test passes now) reds
  too — the file must not only ever loosen — except under `--libs-only`, which
  DEFERS that check because it skips the cell those rows come from.
- ⛔ **The bucket is `PASSED-ALONE`, not `TRANSIENT`, and the rename is the
  finding** (2026-09-18). "TRANSIENT" asserts a CAUSE the instrument cannot
  observe, and it has now been wrong three times — three classes produce the
  identical "failed in the cell, passed alone" observation and they need
  opposite responses:
  - a load-sensitive perf **BAR** (Issue 831) — the only one the old label
    was right about;
  - a shared-fixed-path **CONCURRENCY** defect (Issue 832), which passes alone
    BY CONSTRUCTION, because alone there is no second process. One was filed
    TRANSIENT and the defect shipped;
  - an unseeded-RNG **COIN FLIP** — `player_type_creates_instances`, which
    asserted `Place` against a documented `PASS_PROBABILITY = 0.02` twelve
    lines away and failed 9 times in 400 seeds. A single clean re-run is what
    you EXPECT 98% of the time.
  ⚠ More re-runs is NOT the repair and the arithmetic says so: re-running
  narrows the concurrency class not at all, and a 2% flake survives three
  re-runs 94% of the time. `CONFIRM_RUNS` defaults to 3 because the evidence
  is free, but what actually fixes the reader is that the row no longer names
  a cause — the three classes and their disambiguating greps print at the
  place the reader decides, not in a docstring nobody opens.
- Floors in `scripts/x86_64_matrix_floors.txt` — `min_passed` per package plus
  the integration cell's **two** (targets AND assertions: a target that
  compiles to an empty binary stops printing a line, and the target count can
  hold while every binary inside it empties out). `--canary` runs the first row
  with an impossible floor through the same comparison path and requires the
  failure.
- ⚠ It does NOT cover the macOS device backends, wasm32, or `--all-features`
  for the integration targets.
- ⛔ **And it does not LINT** — that is the inverse hole, and it stood for a
  day: this instrument executes the x86_64 arms with `+avx2` on and reads no
  warnings, while every lane that lints compiles those arms to nothing. 30
  findings were sitting behind it (Issue 819 *the x86_64 lint lane* — the
  number is doubly held; see the pin file). `full_gate.sh` **Layer 2c** is
  the lint half; do not read a green matrix as a green arch.

## Docs gate + drift sweeps

**Repo names in instruments are CONTRACT names** (`scripts/repo_alias.py`).
The ten `derive_repos` / `derive_population` predicates return the live
workspace's directory names passed through a machine-local codec: an optional,
gitignored `scripts/repo_alias.local.txt` (rows `on-disk=contract`) translates
a box's on-disk sibling names into the contract spellings every tracked pin,
floor, and snapshot is keyed on. Absent file → identity mapping (CI, fresh
clones, and the population-sync canaries are untouched by construction); the
count of active mappings is disclosed on stderr, never the names, because run
logs get pasted into tracked docs. Instruments that OPEN sibling files by the
contract name need the on-disk spelling back — `repo_alias.disk()` is the
reverse half (the skill-census glob uses it).

⛔ **The open seam is ONE, and it is `sweep_population.open_repo(name,
workspace)` / `repo_alias.real(repo)` (Issue 842, closed 2026-09-19).** The
derived predicates return CONTRACT names, so `WORKSPACE / name` — the shape
17 of 19 drift sweeps used — opens a directory that DOES NOT EXIST on an
alias box, and every walk on it returns 0: seven sweeps red walk floors
against pins typed from the real repos, every red a TRUE pin measured against
the WRONG DIRECTORY, and the two sweeps that DID import the alias
docs/numbering built `ws / contract-name` paths — the same hole wearing the
codec. The fix is one seam, not 19 patches: sweeps open through
`open_repo`, audit modules resolve at the file-access seam via
`repo_alias.real(repo)` (identity for any unmapped name, which is what makes
it fixture-safe), `worktree_state.sweep_advisory` resolves bare names and
prints CONTRACT labels (the alias content itself must never reach stdout),
and every audit labels findings by the handle's `.name` while reading the
resolved directory. First post-fix run measured the other direction: 19/19
non-citation sweeps green, and the citation sweep surfaced ~35 REAL
file-addressed defects across 6 repos that the zero-read had hidden — filed
as Issue 846, the sweep's red there is true positives, not the 842 class.

- ⛔ **The repair was applied to 19 sweeps BY HAND and nothing stopped the
  twentieth landing unwired for a day** — this file's own most-repeated
  finding, committed by the change that closed the issue naming it. The wall
  is a **fourth `MECHANISMS` row** in `sweep_advisory_membership_gate.py`
  (`alias-open` → `open_repo` / `repo_alias.real`), not a new instrument: the
  registry exists for exactly this, and a fresh gate over the JOIN would have
  been the cries-wolf one — measured, 27 findings, because the shipped
  architecture resolves at the READ seam and keeps CONTRACT-named handles on
  purpose, so every `ws / n` and every `repo.name` in the family is correct.
  **Read the architecture before writing the predicate**: a wall aimed at the
  wrong seam condemns the careful caller.
- ⚠ It carries this file's **first pinned exemption**, and the row is the
  interesting half: `len_derived_drift_sweep` is WIRED one hop away — it
  constructs no repo path of its own and delegates every walk to
  `len_derived_binding_audit`, which calls `repo_alias.real()` at each read
  seam. Teaching `calls_any` to follow delegation is PERMISSIVE for all four
  mechanisms at once (it would credit a sweep for a call some imported module
  makes for its own reasons), where `check_validation_gate` can credit
  delegation only because its delegate IS the classifier the check imports.
  ⛔ The first live row also caught the gate's own PASS line claiming *"every
  one of 21"* beside a per-mechanism count of 20 — true only while the pin
  file was empty, and `skill_repo_set_gate`'s recorded defect reproduced in
  the display of the gate that objects to it.

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
⚠ The set moved to **18** on 2026-09-14 (Issue 775's
`platform_dead_code_floor_gate.py`, a 2415-file Rust-source walk measured at
**~6.2s wall** standalone), and the CPU figure at 18 checks is **UNMEASURED,
not unchanged**: the landing run was on the Windows workstation, where `times`
does not account for native children at all (Issue 792, measured below) and
the gate now prints `CPU SUPPRESSED` rather than the 1.26s it used to. Take
the 18-check CPU figure from the next M3 run; do NOT read 13.37s forward
across a CHECKS change, and do not read a Windows run's number — there is
none — as a speedup. The set moved again the SAME DAY, to **19** (Issue 778's
`subprocess_encoding_gate.py`, a 60-file tracked-`*.py` walk over 44
`subprocess` call sites, measured **~0.6s wall**), so the M3 run owes a 19-check
figure and the 18-check cell will never be measured at all — which is the
point of writing the CHECKS count next to the number instead of the number
alone. It moved to **20** on 2026-09-14 (Issue 787's
`instrument_reachability_gate.py`, a 63-script transitive closure over 13
roots, measured **~0.24s wall** standalone — the cheapest check in the set,
because the closure re-reads only files a root or a script actually names),
so the 19-check cell joins 18 in
never having a POSIX figure — this box has printed `CPU SUPPRESSED` for every
run since the set left 17. It moved to **21** the same day (Issue 789's
`check_validation_gate.py`, an AST pass over the CHECKS array's own 21 scripts,
measured **~0.11s wall** standalone — cheaper still than 787's, because it
parses each check once and reads no tree at all), so 18, 19 **and** 20 are now
cells no POSIX run will ever measure. Four consecutive same-day CHECKS moves is
the argument for the convention, not an embarrassment to it: a bare number in
this paragraph would have been wrong four times in one day. It moved to
**22** on 2026-09-15 (Issue 796's `dual_allocation_gate.py`, a handful of
git-plumbing calls over this checkout and its upstream, measured **~0.10s
wall** standalone). The 21-check cell did get one POSIX figure — **22.75s
CPU** (2026-09-15, loaded M3, sibling agents active; quiet-box class unknown,
so the loaded-box scope the paragraph below demands applies). In CI the new
check green-exits by construction — a main-push checkout has HEAD ==
origin/main, merge base == HEAD — so its live reach is the workstation dev
loop, where the divergence actually exists at run time. It moved to **23**
(Issue 797's `sweep_advisory_membership_gate.py`) and **24** (Issue 804's
`console_encoding_gate.py`) without a sentence here — the table above is the
enforced copy, and this prose lagged it. It moved to **25** on 2026-09-16
(Issue 809's `global_rng_gate.py`, a tracked-`*.rs` regex walk measured
**~2.6s wall** on a quiet M3), and the sentence exists to say that the move
is READ OFF THE GATE — `docs_gate_checks_sync.py` + `check_validation_gate.py`
both print the live count, and a number typed here is a claim about history,
never the instrument.
⚑ **First POSIX CPU figure since the set left 17: 71.67s CPU / 107.8s wall at
32 checks** (2026-09-19, M3, LOADED — a sibling agent session running release
cargo builds throughout, plus the usual concurrent sessions). It is recorded
with its load class because that is the only way it is comparable to anything:
the nearest neighbours in this paragraph are 13.37s at 17 checks on a quiet box
and 22.75s at 21 on a loaded one, and the loaded-box measurement one paragraph
down is 2.7-3.4x its own quiet-box twin. So **do not read 71.67s as the
32-check cost** — read it as an upper bound taken under load, and take a quiet
figure when a quiet box is available. The cells at 18, 19, 20 and 26-31 checks
will never be measured at all, which remains the argument for writing the
CHECKS count beside the number rather than the number alone.
⚑ **A NEAR-QUIET third reading at 33 checks — 50.11s CPU / 52.5s wall** (M3,
loadavg **3.9-4.8**, one concurrent single-core job), and it is the one that
constrains the paragraph above rather than agreeing with it. It is **7%
HIGHER** than the 46.74s taken under moderate load, not lower. So at this
CHECKS count the **2.7-3.4x inflation measured at 17 checks does NOT
reproduce**: the three readings are 71.67s (32 checks, heavy load), 46.74s (33,
sibling builds finished) and 50.11s (33, near-quiet) — a ~7% spread across the
latter two, which is nearer the original 1.4% quiet-box claim than the
loaded-box story. Read that honestly in both directions: the **71.67s outlier
is confirmed** as a load artefact, and the 46.74s figure is confirmed as an
ordinary reading — but "load inflates CPU" is NOT a monotone rule, and a run
this file calls loaded can measure *less* CPU than a quiet one. The quantity is
still the right one to cite; the **load class beside it is a disclosure, not a
correction factor**.

⚑ **And the load caveat was immediately worth its ink: 46.74s CPU / 49.2s wall
at 33 checks**, same box, ~90 minutes later, with the sibling builds finished.
That is MORE work measuring **35% less CPU** — so the 71.67s figure is
confirmed as a load artefact rather than a cost, exactly as the caveat said,
and the 33-check number is the one to compare against. ⚠ Neither is a
quiet-box figure in the strict sense (this workstation always carries other
sessions); read the pair as a **range**, which is all the wall clock ever
was.
⚑ **A fourth reading at 33 checks under the heaviest load yet recorded
there: 50.22s CPU / 54.1s wall** (2026-09-20, M3, loadavg **11.2** — a
riir-train training measurement plus a sibling cargo test running
throughout). It matches the near-quiet 50.11s within **0.2%**, so the
17-check inflation class (the ⛔ paragraph below) did not reproduce at
this CHECKS count even at load 11 — but the load SHAPE differs from that
case (one sustained measurement + one test build, not multi-tenant
compile churn), so read it as bounding the 71.67s outlier further, not as
refuting the LIMIT. The 33-check figure to compare against remains the
**46.74–50.22s range**.
⚑ **A first reading at 34 checks: 50.03s CPU / 54.5s wall** (2026-09-21,
M3, LOADED — sibling cargo builds plus two concurrent agent sessions
throughout; the run was also the landing check for the
`distance_abstain` feature, whose 625→626 README count bump it
verified). At the 33-check range's midpoint, so the CHECKS 33→34 move is
invisible at this load class and the 71.67s outlier stays confirmed as
load. The quiet-box figure remains unrecorded at every CHECKS count ≥ 18
— take the live count from the gate's own PASS line, never from this
sentence.
⛔ And "load-invariant" has a measured LIMIT (2026-09-14): two runs at the
same 17 checks / 1517-file fence floor, on a box carrying the g50 training
precompute plus ≥3 concurrent agent sessions, measured **44.97s · 36.28s
CPU** — 2.7–3.4× the 13.37s figure, with a **20%** run-to-run spread where
the quiet-box spread was 1.4%. The verdicts were unaffected (17/17 both
runs); only the timing figure moved. So the invariance claim is
**quiet-box-scoped**: under sustained multi-tenant load even CPU-seconds
inflate and destabilize (mechanism unmeasured — E-core placement is the
suspect, not the finding). Cite CPU *with the load class it was measured
under*, or the number carries a quiet-box premise onto a busy box.
↔ **This paragraph is the INSTANCE; the general rule lives at §Feature Flag
Discipline's G2 bullet ("A latency number without its BOX STATE is not a
measurement").** Two copies that can drift, named in both directions on
purpose: this one owns the docs-gate CPU figures and their measured
quiet-box scoping, that one owns *any* perf number and the commit-vs-limit
and trough rules. **Nothing asserts they agree** — deliberately, because the
alternative is a prose-diffing check, which is the shape this file already
refuses for landing claims. Editing either, read the other.
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
⛔ **And the ~0 guard is not the whole hazard — the figure has a PLATFORM
premise (Issue 792, 2026-09-14).** On Windows/MSYS, `times` accounts for MSYS
children and reports essentially nothing for NATIVE ones, so a run whose work
is all Python prints a well-formed, plausible number built from `sed`/`tail`
overhead alone: **1.26s CPU against a 19.7s wall**, with the ~0 guard quiet.
Measured one child at a time, each burning ~2s CPU: an MSYS `bash -c` loop →
**1.796s user + 0.468s sys**, `py -c` → **0.000s + 0.015s**, python.exe by
absolute path → **0.000s + 0.045s**. So it is the MSYS/native boundary, **not**
the `py` launcher shim — resolving a real executable recovers nothing. A
wall-ratio test cannot separate that from a busy box (this gate's own 12.65s
CPU on a 299.1s wall is 4%), so the gate CALIBRATES instead: it burns a known
0.25s of CPU in a child of the resolved interpreter and requires `times` to
have seen at least half of it, printing `CPU SUPPRESSED` and wall-only when it
did not. Both arms measured on the same box: native child → 0.000s seen,
suppressed; MSYS child → 0.358s seen, figure printed. **Cite the CPU figure
from a POSIX workstation; a Windows run has no CPU number to compare.**

`.github/workflows/docs_gate.yml` runs it per-push on **`main` only** —
develop pushes do not fire it, so run `./scripts/docs_gate.sh` locally for
develop work. One line per check:

| check | asserts |
|---|---|
| `count_features.py` | flag counts in README + examples/README vs every manifest |
| `bench_doc_audit.py` | default-on / opt-in labels in .benchmarks + .docs vs Cargo defaults — plus two blindness floors and `BlindRead`, exit **2**: an `OSError` on a file the walk just listed means the tree is unreadable, and a PARTIAL manifest read fabricates mismatches ABOVE any floor (Issue 790 F9) |
| `cargo_comment_audit.py` | inline Cargo.toml comments vs the default closure |
| `skill_repo_set_gate.py` | hand-typed repo sets in SKILL.md command blocks (Issue 703) |
| `agents_repo_set_gate.py` | AGENTS.md §Repo count membership vs `scripts/repo_set.txt` — pins the paragraph below |
| `cfg_gated_floor_gate.py` | `#![cfg]`-gated targets that report a green 0-pass (Issue 713) |
| `orphaned_attr_gate.py` | a `#[cfg]` separated from its item by a blank line |
| `percentile_floor_gate.py` | a percentile index that lands on n-1 and so reports the MAX |
| `numbering_gate.py` | a number allocated twice, or a stale/malformed `.highwater` (Issues 724, 725) — including the majority case where BOTH holders have CLOSED and been removed, so nothing is on disk and the tracked check reads clean: walled by membership above the era boundary, ratcheted below it (Issue 795) |
| `dual_allocation_gate.py` | this checkout and its upstream both allocated a numbered document since their merge base — TWIN (same stem, the rebased own line) annotates exit-neutral, INDEPENDENT (two documents claiming one number) exits 1 naming both sides' adding commits (Issue 796) ⛔ **COUNTER** is the third verdict and it exists because the first two compare DOCUMENTS: a number allocated and **closed in one commit** never has a file in any tree, so `--diff-filter=A` reports nothing and the record lives in HISTORY.md. Measured — a sibling did exactly that while this checkout held a file at the same number, and the gate printed a confident zero. The counter itself is the missing document: both sides bumping `.highwater` past the merge-base value means both spent the overlapping range, file or no file. A one-sided bump stays green BY CONSTRUCTION, which is the ordinary case on every push, and both directions are armed — the arithmetic on every push with the git reader injected, the real two-repo fixture under `--prove-fires` (Issue 850) |
| `docs_gate_paths_sync.py` | docs_gate.yml's two hand-duplicated trigger `paths:` lists stay identical |
| `required_features_static_gate.py` | a required-features row naming a feature its package cannot enable (riir-train Issue 513) |
| `cfg_row_implication_gate.py` | a required-features row that BUILDS and compiles its target to NOTHING (riir-train Issue 513) |
| `population_sync_gate.py` | the ten independent contract-repo predicates must agree, and the registry that lists them must be COMPLETE (Issue 788) |
| `trap_sentinel_gate.py` | a shell gate whose abort would report exit 0 — this repo's own two, by MEMBERSHIP (Issue 734) |
| `issue_citation_gate.py` | a cross-repo `Issue N` citation naming no repo — it rebinds to the WRONG document once that number is allocated locally (Issue 749). In CI the cross-repo axis is DEFERRED to the workstation run — the `DOCS_GATE_CI` marker's instrument-alive verdict, because the sibling workspace is absent in a single checkout |
| `markdown_fence_gate.py` | a fenced code block never closed — everything after it renders as code, and a fence scanner mis-phases on it (Issue 756) |
| `platform_dead_code_floor_gate.py` | an item declared ungated whose every use sits behind a platform cfg — dead code on a platform no automatic lane compiles (Issue 775) |
| `subprocess_encoding_gate.py` | a `subprocess` call that decodes with the SYSTEM locale — silent mojibake, or `stdout = None` with the returncode intact (Issue 778) |
| `instrument_reachability_gate.py` | a tracked `scripts/*.py` no root and no documented instrument names — invisible to the census that would find it (Issue 787) |
| `sweep_advisory_membership_gate.py` | a `*_drift_sweep.py` that does not call a FAMILY-WIDE MECHANISM — the Issue-797 worktree advisory (findings and floors then describe whatever the working tree happened to say) or the Issue-815/821 known-extra exemption (the sweep hard-reds on a repo the contract does not claim, with zero content findings); a REGISTRY, per mechanism and never pooled, because this gate governed one mechanism by name and watched Issue 821 miss 3 of 19 beside it (Issues 797 T5, 824) |
| `locale_io_gate.py` | text I/O that decodes/encodes with the SYSTEM locale — `Path.read_text`/`write_text`/`open()` in text mode with no `encoding=`. The **FILE** seam under `subprocess_encoding_gate`'s PIPE seam, and it hid for the same reason (macOS, `ubuntu-latest` and the M3 all speak UTF, so nothing that could notice ever ran it). Not found by a census: a cp874 box silently round-tripped a selftest FIXTURE's em dashes through a single undefined byte, so a whole file of arms had been passing **for the wrong reason** until Issue 828 wrote an arm whose subject WAS the dash. Repair half: `scripts/locale_io_fix.py` (AST-driven, idempotent, newline-preserving) (Issue 829). ⛔ That name-set predicate was itself an ANCHOR: the class is a text-mode FILE OBJECT, which `p.open("w")`, the `tempfile` factories at a text mode, `os.fdopen` and `io.TextIOWrapper` also construct — and the sites it could not see included a fixture this repo's OWN instrument writes with the locale and reads back with an explicit `encoding=` in the same function, in a file Issue 829 had already half-repaired. `.open` cannot be duck-typed the way `read_text` can (`os.open`, `tarfile.open`, `Image.open`), so it is admitted only on a literal TEXT mode, and the cost of that rule is MEASURED on the sweep's own scope line rather than argued (Issue 830) |
| `console_encoding_gate.py` | a tracked `scripts/*.py` that prints a non-ASCII glyph and defends neither stream — on a non-UTF-8 console it dies with **no verdict**, so a sweep's findings are not *unknown* but *unlooked at*; `docs_gate.sh`'s `PYTHONIOENCODING` only covers runs that go through the wrapper, and every workstation audit is documented as a DIRECT invocation (Issue 804). The shared defence is `scripts/console_safe.py`: `errors="backslashreplace"`, because the console encoding is not ours to choose, and forcing it gives mojibake instead of an exception |
| `global_rng_gate.py` | an unseeded-global draw with no pin row — free-function `fastrand::<prim>()` OR the unseeded `Rng::new()`/`Rng::default()` constructor (T3 folded the constructor class in): the unseeded thread-local global made a shipped pruner non-deterministic, found by executing one commit twice; membership + per-row reason, both directions, floors on the walk and the predicate (Issue 809) |
| `algebraic_op_ban_gate.py` | an `algebraic_div`/`algebraic_rem` CODE occurrence in tracked `*.rs` — the Issue 871 T4 ban as a tracked check, not a doc line (the add/mul reassociation lane was adopted feature-gated as `katgpt-attn-match/algebraic_dot`, Bench 871; div/rem are banned outright — `arcp`/remainder semantics are a far larger numerics change); ceiling 0 with no exemption vocabulary by design, walk floor + planted-source predicate arms run unconditionally, masker imported from `platform_dead_code_audit` (Issue 871) |
| `shared_temp_path_gate.py` | a test writing to a FIXED `env::temp_dir()` path — safe against sibling tests in one binary, and truncated by any concurrent PROCESS running the same test; the x86_64 matrix filed one as TRANSIENT because this class passes alone BY CONSTRUCTION (Issue 832) |
| `cross_repo_path_dep_gate.py` | a `path = "../X"` dependency on a repo that is NEITHER on disk NOR in `repo_set.txt` — cargo resolves path deps **even when `optional = true`**, so the CITING repo stops building outright (`cargo check`, `test` and `fmt` all die at manifest load). The gap is a matter of POSTURE: present-and-unregistered reds, absent-but-registered DEFERS, and *neither* is enumerated by nothing, because every predicate derives from the walk or compares walk-against-file — so the population buckets (present-unregistered, absent-registered) leave that state unwatched. `--prove-fires` rewinds to the riir-llm state and requires the ORPHAN; take the live counts from the gate's own PASS line. ⚠ On this box every dep RESOLVES, so the DEFERRED and BROKEN-SUBPATH paths are **UNEXERCISED by live data** and are asserted only by their arms — the PASS line discloses that in both directions, because a green line is otherwise read as evidence they work (Issue 835) |
| `repo_registration_gate.py` | a per-repo pin file with no row for a registered, ON-DISK repo. Registering a contract repo is a **22-file operation with one file gated** — `agents_repo_set_gate` asserts `repo_set.txt` ↔ AGENTS.md §Repo count and is complete over that question, while 21 more tracked files are keyed on the same registry and nothing relates them to it. Measured the day riir-llm joined: **19 of 21 sweeps RED** with `UNPINNED — add a row`, each for a reason unrelated to what the run was measuring — AGENTS.md's own *"a sweep that always reds is a sweep nobody runs"* (Issue 793), reproduced. ⛔ The argument is not the inconvenience but what the reds HID: pinning riir-llm turned bookkeeping reds into **four live findings** (riir-kat's unqualified cross-repo citations, riir-shader's EXPOSED trap-launder script **and** its stale `.plans/.highwater`, riir-clippy's undocumented scripts). ⚠ The predicate is NOT "every file carrying repo rows" — some files are legitimately SUBSET-scoped by their own headers, and demanding completeness of a correct file manufactures the cries-wolf state the gate exists to prevent; scope is a per-file declaration with a reason in `scripts/repo_registration_scope.txt`, defaulting to EVERY (a forgotten declaration reds, a forgotten EVERY would green) and reds in BOTH directions. Scope is `repo_set.txt` ∩ the derived walk, so a partial clone is never asked for a row no sweep there could measure. ⚠ It does NOT claim a row's VALUES are right — only the sweep that owns the file can measure that (Issue 837) |
| `cross_module_attr_gate.py` | a tracked `scripts/*.py` naming an attribute a sibling module does not define. Python has **no link step**, so `import worktree_state` + `worktree_state.is_checkout(root)` resolves at CALL time and a rename is not an error until somebody EXECUTES the importer — and when the importer is a gate, the failure is **no verdict** rather than a wrong one, which is Issue 804's class a seam over. Measured: a commit privatised `is_checkout` and deleted `worktree_fixture` while reworking that module's own arms, updating the seven in-module callers and neither of the two external ones; both external callers are CHECKS in this very table, and `develop` was red for **6h40m**. ⛔ Both names carried a written contract NAMING their consumers — this file's own line on `worktree_state.is_checkout`, and the fixture's docstring — so the document said the right thing and nothing read it, which is `instrument_reachability_gate`'s finding a level down. Membership + a reason per row, both directions, floors on the walk and on the import RESOLUTION (the floor that fails when the alias table stops being built and every name reads trivially fine). `--prove-fires` is two-sided against a known answer: 0 findings at the parent and exactly those 4 at the commit. ⚠ STATED and printed on the verdict line rather than remembered: `getattr` sites and star-import OPAQUE modules are counted and NEVER flagged, a name bound only inside a function is not seen, and a runtime break that is not name resolution needs an EXECUTION — that is Issue 848 T3, and the instrument for it (`arm_reach_gate`'s `BASELINE-CRASH`) is a 157.6s workstation verdict kept out of this ~13s budget on purpose (Issue 848) |
| `import_health_gate.py` | a tracked `scripts/*.py` that does not IMPORT. The EXECUTION half of Issue 848, and it is where the static half (`cross_module_attr_gate`, the row above) cannot reach: a circular import, a missing third-party dependency, a `raise` in top-level code. The instrument that already executes every module is `arm_reach_gate`'s `BASELINE-CRASH` bucket, kept out of this budget at 157.6s — this is its cheap half, one child, 0.11s of import. ⛔ **The affordability measurement found its own blocker**: 6.238s of the 6.34s an import pass cost was ONE module whose entire body was top-level (no `main()`, no `__main__` guard), so importing it ran a workspace-wide `.rs` walk and printed to stdout — guarded in the same change, and `arm_reach_gate` had been paying it once per mutant. A per-module subprocess was measured too and is not worth 3x (10.28s vs 7.52s, identical verdicts), so the cost of that choice is STATED rather than hidden: a module already imported as somebody's dependency is cached, and a failure caused by a previous import's side effects would be attributed to the wrong module. ⛔ **MISSING-DEP is its own bucket and is NEVER flagged** — a pin there would make the gate box-dependent in the worst direction, correct on a box without the package and STALE on a box with it, so a green run would depend on not having installed something. A missing tracked SIBLING is a different statement and stays a finding. ⚠ It does NOT claim a module WORKS — a wrong constant, a broken predicate and a changed signature all import fine (Issue 848 T3) |
| `shipped_target_feature_gate.py` | a SHIPPED path selecting its fast arm on a **COMPILE-time** `target_feature`. `target_feature = "avx2"` is OFF by default on x86_64, so the arm compiles to **nothing** on every ordinary build and the dispatcher silently takes its scalar fallback. ⛔ **This file already documented the shape and only for GATES** — *"an arm gated `cfg(all(target_arch = "x86_64", target_feature = "avx2"))` compiles to nothing without it, and the run then exercises the scalar fallback and proves nothing"* — where the cost is an unproven claim. On a shipped path the cost is **latency on every call, in the configuration everybody builds**: measured at **4.4-5.6x** on `dequant_dot_via_lut` in a DEFAULT-ON feature, and 2.4-2.5x on bf16 RNE narrowing. The correct form needs no new machinery — `simd_level() == SimdLevel::Avx2`, the cached CPUID probe `katgpt-types` ships, with the kernel gated on the ARCH alone (its `#[target_feature(enable = ..)]`, not the cfg, is what makes the intrinsic body compile). ⚠ **Three exclusions, each of which a naive grep gets wrong** and each counted on the verdict line rather than remembered: wasm32/`simd128` (that target has no runtime feature detection here, so a compile-time gate is the only option and layer 2b already builds both arms), NEON (implied by the arch), and **the runtime probe's own body**, which is the same attribute doing the opposite job and is therefore PINNED rather than excluded by a predicate. The key is LINE-FREE (`<path>::<enclosing fn>#<ordinal>`), resolved by brace counting over the masked text, with the two cases needing opposite lookups — an attribute INSIDE a body belongs to its innermost enclosing fn, one ON an item belongs to the NEXT. ⛔ No sweep half, MEASURED not inherited: every `target_feature` cfg attribute in `src/` across the contract repos is in THIS repo, and `--workspace` re-derives that table rather than leaving it in prose. `--prove-fires` is two-sided against a known answer. ⚠ It does NOT see a dispatcher reading `cfg!(target_feature = ..)` as a runtime-looking boolean (Issue 847 T3) |
| `timed_region_guard_gate.py` | a latency ceiling with no loud-zero defence — rustc + fat LTO deletes a timed loop whose result is dead, so the bar is satisfied by **absent work** and passes with MAXIMUM margin. This is the `#![cfg]` green-zero rule one layer down, and worse: the assertion RUNS, so the output is a plausible number rather than a zero count and no count floor can see it. ⛔ Not reasoned about — **executed**: all 34 asserting regions at n ≥ 1000 were run and **7 were satisfied by absent work** (20.6%), two of them GOAT gates, one printing `Speedup: 8657.9×`, one printing a well-formed `0.00x` because only the NUMERATOR vanished. ⚠ The static predicate Issue 855 T4 proposed is REFUTED by that run — `let _ =` vanished **3 of 15** against **4 of 19** for the rest, i.e. the base rate wearing a grep; the column that separates is `black_box`, **7 of 23** without it against **0 of 11** with. So the gate does not predict which region is broken; it gates the decidable thing — *is there a loud-zero defence at all*. ⛔ Two tiers, and the split is the honest part: a LITERAL loop bound is the population that was READ end to end (membership wall, one MEASURED number per row), while a bound needing one hop of resolution is real and **unread** (a ratchet on the derivative — pinning an unread bucket by name is Issue 785's forbidden shape). One hop is not optional: T1's own two founding specimens are `let n = 100_000; for _ in 0..n`, so a literal-only predicate would have shipped the class it was written for. ⚠ It does NOT claim a pinned region is safe — `black_box` is the weakest of three defences (result, **arguments**, **receiver**) and two arms vanished carrying one (Issue 855 T4) |
| `check_validation_gate.py` | a CHECK in this array whose own arithmetic no arm asserts — including one whose arm is flag-gated and so never runs (Issue 789) |
| `docs_gate_checks_sync.py` | this CHECKS array vs the AGENTS.md table documenting it — membership both ways + quantity words (Issue 750) |
| `skill_size_gate.py` | a skill SKILL.md over the 80KB ceiling — the third 100KB-regrowth class (doc-sync 09-05/09-11/09-21, boundary-guard 09-08/09-11/09-21; the one-line convention held, the ~8 rows/day cadence didn't) — forces each file's prune-to-15 maintenance rule; recovery via `git log -p` |

The `CHECKS` count is deliberately not written here — it drifted once, which
is exactly the drift this gate exists to catch.

**Partial-clone boxes (Issue 765):** the three population checks
(`skill_repo_set_gate`, `population_sync_gate`, `issue_citation_gate`)
hard-red on a box carrying a subset of the workspace — and their raw remedy
used to invite regenerating `repo_set.txt` there, which deletes live repos
from the canonical set. A known-partial box (the 4090: 14 of 20) exports
`DOCS_GATE_PARTIAL_CLONE=1` and gets a loud instrument-alive DEFERRAL on the
population axis instead (predicate agreement + the local axes still run; the
deferral rides each check's final line, the one `docs_gate.sh` forwards). The
marker is an explicit opt-in in the `DOCS_GATE_CI` idiom — **never
auto-detected**, because a genuine removal whose `repo_set.txt` update was
forgotten is set-identical to a partial clone from the walk alone, and an
inferred green would ship the stale file. Gone-only disagreement WITHOUT the
marker reds naming both hypotheses; a repo on disk the file does not know
reds in every posture, marker or not.

**Known-extra boxes (Issue 815) — the MIRROR marker, and it takes NAMES.**
`DOCS_GATE_PARTIAL_CLONE` covers repos that are ABSENT. The other bucket is
PRESENT-and-unregistered, which AGENTS.md deliberately keeps loud ("a repo
JOINING, reds in every posture") — and that is right until a box carries
siblings the contract genuinely does not claim. Measured on the 4090:
`seal-game-editor`, `seal-online-remaster` and `seal-remake` each carry a root
`BOUNDARY.md` and a `.git`, so each satisfies `derive_repos` exactly, and two
of the three population checks red on every run with **zero** content
findings. Such a box exports
`DOCS_GATE_KNOWN_EXTRA=seal-game-editor,seal-online-remaster,seal-remake`
(comma- or space-separated) and gets the same loud instrument-alive disclosure
on the final line, in both directions.
- ⛔ **NAMES, never `=1`, and the asymmetry with the partial marker is the
  whole design.** `=1` would excuse the NEXT unregistered repo too — the one
  nobody has looked at. Naming them keeps the bucket loud for everything else,
  and the arms assert exactly that case (an unnamed extra reds beside a named
  one).
- **Reds in BOTH directions:** a name that is gone from the box, or that
  `repo_set.txt` has since registered, is a STALE acknowledgement and fails.
  So the marker cannot only ever loosen, and an acknowledgement does not
  outlive the repo it was written for.
- ⚑ **The Issue-842 alias codec RETIRES the marker — measured on the 4090
  (2026-09-20).** With `repo_alias.local.txt` mapping the on-disk
  `seal-game-editor`/`seal-online-remaster`/`seal-remake` to their
  registered contract names, those directories read REGISTERED through the
  codec and the marker reds in exactly the stale-acknowledgement direction
  the bullet above names: 2/33 failed WITH the marker, 33/33 clean WITHOUT
  it. The alias file is the box-local repair that supersedes the marker —
  drop the env var, don't widen it.
- Never auto-detected, the `DOCS_GATE_CI` idiom, for the partial marker's own
  reason: a repo whose `repo_set.txt` row was simply forgotten is
  set-identical, from the walk alone, to one that does not belong.
- ⚠ It does NOT settle whether those repos belong in the workspace — that is a
  contract question (Issue 815 options 1 and 3, owner-owned). The marker makes
  the box readable; it does not make the box right.
- ⛔ The scope line must count CANONICAL repos, not the walk. Measured on the
  landing run: with the three acknowledged, `skill_repo_set_gate` printed
  "16 of 20 canonical repos present" over 13 canonical + 3 extra — a count
  crediting the extras as canonical and understating the absence by exactly
  their number, which is the partial-set-as-whole-one defect that gate exists
  to catch, committed by its own display.

**Every sweep below answers the partial-clone question the same way, once
(Issue 793): `scripts/sweep_population.py`.** Seven of them carried a
copy-pasted "pinned but ABSENT from the derived walk" loop and hard-red on a
known 16-of-20 box with `DOCS_GATE_PARTIAL_CLONE=1` already set and every
content assertion green — and a sweep that always reds is a sweep nobody runs.
Measured: the percentile sweep's Issue-777 findings, and four live citation
drift rows, were sitting behind those reds. Three verdicts, never
interchangeable — **UNREGISTERED** (on disk, absent from `repo_set.txt`: a repo
JOINING, reds in every posture), **UNSEEN** (absent, no marker: never a pass),
**DEFERRED** (the same set with the marker, riding the FINAL line in BOTH
directions, because a deferral printed only on failure is one nobody reads on
the run that passes). Never auto-detected: a genuine removal whose row update
was forgotten is set-identical to a partial clone from the walk alone.

⛔ **"Every sweep" was a claim about eight of eleven, and the three left out
were not exempt — they were quieter (Issue 782).** Two of them
(`docs_drift_sweep`, `restatement_drift_sweep`) still carried the copy-pasted
loop and only LOOKED clean because their *pinned subsets* happen to be checked
out on this box; the third, `cfg_row_implication_drift_sweep`, had **no absence
check at all** — it iterates the DERIVED repos, so it caught walk→pins
(UNREGISTERED) and was blind to pins→walk, and printed
`PASSED — every repo within its pins` over **16 of 20** with four pinned rows
evaluated by nothing. **That silent green is the worse direction:** a sweep
that hard-reds is impossible to misread, and this one survived the 779 census
*because* it was quieter, not because it was correct. All eleven share the
verdict now. ⛔ **"Eleven" is that day's measurement, and every count of this
family in prose has gone stale — including one written the same morning.**
Issue 797 wired the worktree advisory into the family and typed **sixteen**
into this document; two hours later a concurrent session landed two more sweeps
and the figure was wrong. The repair is not a fresher number: it is
`sweep_advisory_membership_gate.py`, which reds on a `*_drift_sweep.py` that
does not call the advisory, so the next one cannot land unwired. **Take the
family size from that gate's PASS line, never from a sentence here.** ⚠ A **subset**-population sweep has TWO populations and they are
not interchangeable: hand `population_verdict` the **contract walk**, never the
subset — the restatement sweep, handed its own `.proofs` set, reported **16
phantom absences** and failed. The hole the shared verdict cannot see is
therefore local to each subset sweep and needs its own check: a repo pinned and
checked out that has **dropped out** of the subset (`DROPPED`), whose ceiling
can no longer fail.

Workstation-only cross-repo sweep family — `docs_drift_sweep.py`,
`numbering_drift_sweep.py`, `required_features_drift_sweep.py`,
`percentile_drift_sweep.py`, `cfg_gated_drift_sweep.py`,
`cfg_row_implication_drift_sweep.py`, `trap_sentinel_drift_sweep.py`,
`citation_drift_sweep.py`, `restatement_drift_sweep.py` (every contract repo,
on demand), `markdown_fence_drift_sweep.py` (every contract repo, on demand —
the Issue 756 unterminated-fence verdict workspace-wide, two-axis pins
(`min_md_files` walk floor + `max_unterminated = 0` wall); its FIRST workspace
run caught the then-new `mmorpg-remaster`'s `.plans/005:600`, 14 swallowed
lines, repaired there at `99064c5`),
`platform_dead_code_drift_sweep.py` (every contract repo, on demand — the
Issue 775 verdict half of `platform_dead_code_audit.py`, and the one sweep
whose `--prove-fires` runs by DEFAULT: the per-push gate cannot afford the
`git archive` of the known-answer tree, this can),
`subprocess_encoding_drift_sweep.py` (every contract repo, on demand — the
Issue 778 locale-decoding verdict workspace-wide, and the **seventh** time one
of these was pointed anywhere but here and found something: **29 DECODE + 2
CHILD-ENCODER over 5 repos**, all repaired at landing, Issue 783. Its two
floors are not interchangeable and neither is redundant — `min_calls` is **0
in 10 of 16 repos**, because they have `.py` files and no `subprocess` at all,
so in exactly those repos `min_py_files` is the only blindness detector there
is),
`locale_io_drift_sweep.py` (every contract repo, on demand — the Issue 829
verdict half of `locale_io_gate.py`, landed in the SAME change as its gate
because shipping one half is the failure this document records nine times.
Ceiling a RATCHET on the derivative rather than a wall, on a measurement: 118
of the 273 sites were in seven repos the filing session did not own, and
ratcheting a bucket nobody has read is Issue 785's forbidden shape. All 118
were then read and repaired, each row carrying its sibling SHA — but the file
stays a ratchet, because the next repo to join arrives with whatever it has
and a wall would make that somebody's emergency. ⚑ Its head-provenance wiring
earned its keep on its first run, reporting **128 MASKED rows** — committed
defects an uncommitted repair was hiding — instead of a clean sweep. ⛔ Its
`273 → 2` close-out was a count of three CALL NAMES, and reading the residual
instead of trusting the count found 11 more sites in the same population
(Issue 830): the class is a text-mode FILE OBJECT, not three names. Take the
SCOPE from the sweep's own `scope:` line, which names every admitted form and
the one STATED blind spot, never from a remembered triple),
`console_encoding_drift_sweep.py` (every contract repo, on demand — the
Issue 804 verdict half of `console_encoding_gate.py`, and the **tenth**
instance of the never-generalised shape. Its issue wrote the cross-repo axis
down as “⚠ unmeasured, deliberately” on the `check_validation_gate` 789 T4
precedent — *re-measure the population before answering*. The measurement was
then never taken; taken, it returned **seven** repos against katgpt-rs's 0 of
72, so 789's “no sweep” does NOT carry across — take the live counts from the
sweep's own summary line, never from this sentence, which was already stale
within the hour when riir-clippy's five rows were repaired at `a7a0d03d`. ⚠ The exposure caveat SURVIVES it — a cp874 console is this box's
property and riir-train's 53 rows are the same plan-scoped over-capture
`instrument_reachability` measures on this identical walk — which is why the
ceiling is a **RATCHET on the derivative**, not a demand for 71 repairs in
seven trees this repo does not own. Three floors, and the WALK and the
PREDICATE are separate because they break separately: an `ast.parse` regression
takes the population to 0 over an unchanged walk),
`timed_region_drift_sweep.py` (every contract repo, on demand — the Issue 855
T6 verdict half of `timed_region_guard_gate.py`, and the one sweep in the family
whose ratchet is **EARNED by execution rather than argued for**. T5 RAN all 33
sibling READ-tier rows: 1 UNBUILDABLE, 32 executed, **2 VANISHED = 6.3%**
against this repo's own 7 of 34 = 20.6%. That is enough to constrain the
DERIVATIVE in somebody else's tree and NOT enough for a membership wall — 30 of
the 32 are single readings on a loaded box without T3's 4x scaling probe, so
*fast* and *partly eliminated* are not separated for them, and laundering that
residue into a pin is the shape Issue 785 forbids. ⛔ It REPORTS two states it
cannot gate, both invisible to the counting half and both found by RUNNING:
`#[ignore]`d (the region is UNEXECUTED — `cargo test --exact` prints
`ok. 0 passed; 1 ignored`, exit 0, this family's own green-zero shape one axis
over; 16 workspace-wide, 11 of them riir-chain) and **prints NO number** (the
quantity lives only inside an `assert!` message, i.e. visible ONLY on failure,
so the method that found every other row cannot see it — measured as the
highest-yield slice by 4x, 1 VANISHED of 4). A third, UNBUILDABLE, is NOT
reported because it is not statically decidable at all: it is a manifest-
RESOLUTION property, filed as riir-chain `.issues/157`. Two floors that break
separately plus a reserved `TOTALS` row, because 8 of 21 repos have 0 timed
regions and 3 have 0 tracked test/bench files, so BOTH per-repo floors are
vacuous there — Issue 783's population shape, and the
`wasm32_surface_drift_sweep` answer to it. ⚠ It may use `head_delta`'s per-file
shortcut where `instrument_reachability` and `len_derived` may not, because its
classifier IS per-file; that premise is stated at the call site),
`shared_temp_path_drift_sweep.py` (every contract repo, on demand — the
Issue 832 T3 verdict half of `shared_temp_path_gate.py`, landed because its
gate's own docstring named this axis UNMEASURED and told the next reader to
**count first** rather than inherit `check_validation_gate`'s population-of-one
answer. Counted: `scan()` already took a repo path, so the question was
answerable the whole time, and the answer is **100 fixed-path sites over the 13
canonical repos present, 98 of them outside this repo** — not a population of
one, and not over-capture either (riir-ai's `go_bonsai_cache_test.bin` and
`test_egl_roundtrip.bin` are `#[test]` bodies writing a fixed filename, the
shape that produced this repo's own five-at-once *"File too small for header"*
failures). Ceiling a **RATCHET at measured**, `instrument_reachability`'s
answer for its reason: a wall would demand 98 repairs in ten trees this session
does not own, and a cross-repo repair is not landed until it is COMMITTED in
the sibling with a cited SHA (Issue 798). ⛔ It carries **no `min_rs_files`
column** — the third sweep to delegate that identical `tracked_files(repo,
"*.rs")` walk — and the delegation is ASSERTED, not assumed: a pinned repo that
loses its non-zero row in `orphaned_attr_drift_floors.txt` reds, and a
delegated file it cannot PARSE is refused rather than read as an empty dict,
which would turn the assertion into the no-op it exists to prevent. katgpt-rs's
row asserts the GATE'S VERDICT rather than restating a count this sweep derives
from the same `scan()`, so a stale membership row reds here too. Repair half:
`scripts/shared_temp_path_fix.py` (the `locale_io_fix.py` pattern — idempotent,
LF-preserving, and it REFUSES to leave a line over `--max-width` rather than
invite the `cargo fmt -p` that reformats ~1300 unrelated lines here). It
IMPORTS the gate's own `mask` and `FIXED_JOIN` rather than re-deriving them:
a repair pass whose idea of a site differs from the gate's either misses rows
the gate will red on or edits code the gate never asked about. `examples/` and
`src/bin/` are skipped by default — the adjudication the pinned rows record,
not squeamishness — and `--include-demos` overrides it for a repo whose owner
decides otherwise. ⛔ Its first version tested CRLF **before** asking whether
the file had a site at all, and printed a loud per-file refusal for 18 riir-ai
files carrying none: a message that invites work nobody needs to do is this
workspace's own cries-wolf failure mode, one instrument down),
`pipefail_discard_audit.py` + `pipefail_discard_drift_sweep.py` (every
contract repo, on demand — the shell class where a `var="$(pipeline)"`
assignment under `set -euo pipefail` is killed by a legitimately-empty grep
(e exit 1 on no-match) AFTER the measured work ran and BEFORE the result was
written — the riir-ai `perf_rematch.sh` incident that lost five benchmark
cells (fix `512b74939`, the sweep's `--prove-fires` known answer; `-S`
cannot locate it — the fix added `|| true` without changing occurrence
counts — so the sweep locates it via `git log -L`). First-run census
(2026-09-15): 194 tracked `.sh` / 1,359 substitution sites / **51 findings,
0 UNPARSED**, every row pinned with a reason in `pipefail_discard_expected.txt`
as an EYES LIST (4 deliberate `grep -c` tripwires in the four `proof_gate.sh`
copies + 47 live kill-shapes awaiting owner triage — the 21-row
`riir-ai/scripts/ci_feature_guard.sh` layer-summary cluster is the
highest-value block: a failing layer's missing `ok` line kills the gate
mid-summary instead of letting the comparison report it). ⛔ That is a dated
CENSUS, not a standing figure, and it started moving the same day — take the
live counts from the sweep's own summary line, and the per-row standing from
the pin file, where a row now says INERT BY CONSTRUCTION or LIVE. ⛔ And the
one-line repair recipe in that file's header — "one neutralizer" — is
INSUFFICIENT wherever the captured value is then TESTED: riir-chain's three
`money_format_gate.sh` reads needed `|| true` **and** an emptiness failure,
because an empty capture matches no pattern and `|| true` alone converts a
silent death into a silent PASS on the exact regression the check exists to
catch (fixed at riir-chain `41e2de8`). ⛔ And the sharper specimen is
riir-neuron-db's leakage-audit guard, where the recipe applied verbatim is
strictly WORSE than the defect — measured, three arms, on a run producing no
test-result line: at HEAD `rc 1` silent; with `|| true` alone **`rc 0`, the
security gate PASSES an unverified run**; with the emptiness check too, the
fail-loud branch fires. (That line carried a second, independent defect: it
ended `)":`, a stray colon concatenated outside the quotes, so `[ -z
"$LEAK_LINE" ]` could never fire and the branch was dead code regardless of
pipefail. Fixed at riir-neuron-db `9696cc8`.) The full census is TRIAGED now —
51 findings down to 10 pinned rows, all DELIBERATE (the 4 `proof_gate.sh`
`grep -c` tripwires) or INERT BY CONSTRUCTION (an earlier `if grep -q` on the
same pattern and file, or an earlier `[ -z ]` early exit, which a static
classifier cannot see): the 37 live kill-shapes were fixed across ten repos —
the 25-row riir-ai block (rove ×2, tcc ×1, 22 `ci_feature_guard.sh`
layer-summary sites) at riir-ai `11d629672`, two earlier batches (14 sites) in
their own repos, and the final 3 (deployer control-do TS_PAIR `b3c7bbd`,
mmorpg warm-tier-do SEQ1/SEQ2 `0688254`) in the closing pass; every downstream
was verified to own the empty case before its `|| true` landed. Two bash laws are
MEASURED, not reasoned: `local x="$(fails)"` does not kill (local masks the
status — its own LOCAL-MASKED bucket, listed never gated), and a
`(grep ‖ true) | tail` paren-group is guarded by its interior — which is
what moved 6 dapps `setup.sh` rows to GUARDED),
`toolchain_override_audit.py` + `toolchain_override_drift_sweep.py` (every
contract repo, on demand — the class where a hardcoded `RUSTUP_TOOLCHAIN`
override outlives the workspace pin it contradicts (intake P14 (k): a
`1.95.0` netem-script override survived the 09-04 bump to 1.98.1 and built a
sibling at the box default). Scans tracked `.sh/.yml/.yaml/.toml/.py` +
Dockerfiles comment-aware; verdicts MATCH / DELIBERATE (the in-source
`toolchain-override-deliberate` marker — same line, contiguous comment run
above the line, or above the HEAD of the backslash-continuation command,
because a comment cannot live inside a continuation chain) / DRIFT (walled
at 0) / UNRESOLVED (TOKEN values, counted ceiling) / NO-PIN-OVERRIDE +
repo-level UNPINNED-REPO (both INFO — 13 of 20 repos carry no
`rust-toolchain.toml` against the owner's every-workspace-pins directive;
the real repair is pin files, owner-owned). First green run: 0 DRIFT, 3
DELIBERATE, 1 UNRESOLVED-MARKED, 1 NO-PIN-OVERRIDE),
`orphaned_attr_drift_sweep.py` (every contract repo, on demand — the Issue 784
verdict half of `orphaned_attr_gate.py`, and the **eighth** instance of this
shape. The one that found **no** new offenders, which is the honest outcome to
report: 0 orphaned now holds across three measurements and TWO population
definitions. What it did find is a stale WARRANT — the gate's docstring carried
`11,132 .rs / 49,624 sites` by hand, and Issue 777's tracked walk put the same
16 repos at **8,694 / 26,598**, 22% and **46%** lower, because 23k of those
sites were in trees no repo owns. Both floors bite in all 16 here, unlike 783's
population — a measured difference, not an assumption),
`wasm32_surface_drift_sweep.py` (every contract repo, on demand — the Issue 785
verdict half of `wasm32_surface_audit.py`, which had been a report with no
verdict at all. `max_unresolved = 0` is a WALL (a ratchet on a bucket whose
meaning is *unanswered* is a backlog), UNCOVERED is pinned by **NAME** in
`scripts/wasm32_uncovered_expected.txt` and reds in BOTH directions, and the
walk floor is the ONLY blindness detector here — vacuous in 7 of 16 repos,
which is why a reserved `TOTALS` row floors the population globally),
`len_derived_drift_sweep.py` (every contract repo, on demand — the Issue 786
verdict half of `len_derived_binding_audit.py`, and the **ninth** instance of
this shape. The QUIETEST one: unlike 784 and 785 it had no hand-typed standing
figure to go stale, so there was nothing to catch being wrong — an instrument
nobody is told about does not drift into error in public, it just stops being
run, which is why these are found by census and not by symptom. It is the one
sweep in the family whose classifier is **cross-repo by construction** (HALF C
resolves provenance through workspace callers), so a partial clone can corrupt
a PRESENT repo's verdict and `DEFERRED` does not cover that; measured, both
directions, 7 of 251 cited caller refs are cross-repo and leave-one-out over
all 16 repos produces **0 verdict flips**, so the sweep runs a TARGETED
leave-one-out over the derived supplier set every run rather than assuming the
axis away. It is also the one sweep with **no `min_rs_files` column** —
three others floor that identical walk over that identical population, and the
delegation is ASSERTED rather than assumed),
`instrument_reachability_drift_sweep.py` (every contract repo, on demand — the
Issue 787 verdict half of `instrument_reachability_gate.py`, and the one sweep
in the family whose ceiling is a **RATCHET** rather than a wall or a membership
set. Measured on its first run: **95 unreachable of 152** tracked
`scripts/*.py` over 16 repos, riir-train **61 of 61** — that repo's `scripts/`
is almost entirely plan-scoped one-offs, where the predicate OVER-CAPTURES,
because "unfindable from AGENTS.md" is the correct state for a script whose
whole life was one plan task. So the per-push gate pins this repo's own 7 rows
by MEMBERSHIP with a reason each, and the sweep constrains the DERIVATIVE
everywhere else: the commit that adds ANOTHER unfindable script reds, and the
existing rows stay their own repos' to adjudicate),
`highwater_contiguity_audit.py` (report-only, every contract repo: is a
repo's `.highwater` a contiguous allocation ledger — Issue 768's measured
REFUTATION of the counter-as-ownership-witness: 438 gaps + 27 resets over 73
counters under the Issue-770 per-commit-parent walk, no major repo contiguous;
the reset verdict half + the report-only unbumped observation live in
`numbering_drift_sweep.py` per Issues 769+770),
`sibling_docs_drift.yml` (reusable workflow, one caller), and
`ci_gate_coverage.py` (report, always exit 0: which repos gate their full
compile+lint surface in CI, and whether anything automatically starts it).
⛔ **Its standing finding is not that the main-only owner call is wrong — it is
that the lane it produces is ZERO, not reduced.** Measured 2026-09-15: **12 of
16** repos carry a real compile/lint command that no schedule and no push ever
starts, and every one of their `push: branches: [main]` filters is inert. The
two causes need different repairs and the report names them apart, because
`carries no copy` quietly suggests a fix that the other case cannot have:
**five repos have no `origin/main` AT ALL** (riir-auth, riir-kat,
riir-mmorpg-examples, mmorpg-remaster, mmorpg-remake) while their filter
names it, and the six that do have one carry no `.github/workflows/` directory
there. Promoting the file repairs the second; the first needs somebody to decide
whether the filter or the branching model is wrong. Until then those gates run
only when a human clicks them.
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

⛔ **Issue 846 — the alias seam has a PROSE half, and it is spelled
`spelling_aliases`.** The contract names `mmorpg-editor` / `mmorpg-remake` /
`mmorpg-remaster` live in `repo_set.txt` only — on BOTH measured boxes (M3,
4090) the directories are on disk as `seal-game-editor` / `seal-remake` /
`seal-online-remaster`, and every document citing their plans was written
against the ON-DISK spelling. 842's `open_repo`/`real()` seam taught the
sweeps to READ the aliased directories; the first real read then surfaced
44 true CROSS rows, nearly every specimen naming the owner correctly in
prose the qualifier matcher could not see. The repair is in
`issue_citation_gate.spelling_aliases` (consumed by `qualifiers()`): a
spelling qualifies in exactly the two places the contract full name is
already accepted — the 40-char lead and the 3-line window (which reads the
citation's own line forward) — and matches with the `_NAME` boundary regex,
so the retired `seal-remake-unity` does not name `seal-remake` (the plain
`\b` the short aliases use MATCHES inside it — the boundary arm caught that
before it shipped). LENIENCY ONLY: `written_names` stays contract-only, so
a spelling can clear a row but never accuse one. In CODE, not in the
gitignored `repo_alias.local.txt` — a box-local qualifier table would make
the verdict itself machine-local. Measured at the landing: 37 rows cleared
by the classifier, 7 by prose qualification in riir-dao / riir-kat /
riir-neuron-db (committed there at `160f9a1` / `51994ee` / `3f6cf67` —
riir-kat's three were independently fixed at origin by the Issue-837
close-out before this box pulled; the local twin was skipped in rebase),
sweep rc=0 for the first time since 842 opened the aliased repos.

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

⛔ **And the wrong address the paragraph above names as a worked example —
riir-train Issue 513 written up as `katgpt-rs Issue 513` — was still standing
in the workspace when Issue 794 went looking for it** (riir-neuron-db `AGENTS.md:82`, repaired to `riir-train Issue
513 T6`). Not because `is_qualified` missed it: because the `⛔MISATTRIBUTED`
tag was computed **only in the CROSS bucket**, and the three-way bucketing runs
first. A citation whose number also falls under the *citing* repo's own ceiling
was reclassified **IN-LOCAL-RANGE** — "UNDECIDED, never clean" — and the tag
never ran. Never counted, never gated. IN-LOCAL-RANGE's premise is refuted by
such a row's own text: it reaches that bucket only when the Issue-754 oracle
found **no** local allocation *and* the author wrote a different repo's name
directly on the citation. **CROSS is unfollowable; this is followable, to the
wrong place** — its own class (`MISATTRIBUTED-IN-RANGE`), walled at 0
globally rather than ratcheted per repo, because it has no backlog. The
leniency was never a COUNT (`gate_says()` already asserts the sweep partitions
the gate's finding set); it was the **label plus the per-repo
`max_in_local_range` ratchet**, which tolerates a wrong address in the 15 repos
the per-push gate never runs in.
⛔ The boundary is **measured, and it is not the obvious one.** The same
predicate one branch up — at the `n in mine` short-circuit, where the number
*is* locally allocated — is **19 rows workspace-wide and 19 of them are
FALSE**: prose contrasting a local number with a remote one, the 40-char lead
catching the *neighbour's* address (`riir-ai Issue 853 / this repo's Issue
093`). That asymmetry is mechanism, not luck — a locally-allocated number has
a local referent for the prose to contrast against — so the rule stops at
IN-RANGE and the exemption is a measurement rather than an oversight. Read the
other column honestly too: it is **n = 1**, so "0 false positives" there is one
row's worth of evidence, not a rate. The per-push gate needed **no** change and
that is itself the finding — it has no IN-RANGE bucket at all, so the sweep
that cross-checks it was the **more lenient** of the two.

⚠ **`allocated()` is not a complete record, and the gap is a house STYLE**
(Issue 781). `heading_allocated()` — the Issue-754 path that recovers a number
whose file was created and removed without an intervening commit — anchors the
parenthetical immediately after the number, so `## Issue NNN (date) — title`
reads and `## Issue NNN resolved — title (date)` does not. Measured over 16
repos, **under half** the self-allocation records are read — and the split is
by convention, not by correctness: one repo scores 100%, **three score zero**,
and katgpt-rs is mixed, its own newest closes in the form its own instrument
cannot read. The figures are printed by the sweep every run and the dated
snapshot lives in `heading_style_blind`'s docstring; a magnitude here, because
an exact count in prose about documents five-plus sessions edit daily is drift
waiting to happen. Widening the pattern is **unsound and the sweep's
own self-test proves it**: arm 2 pins `## Issue NNN follow-up (date)` as a
measured negative — commentary on a number is not an allocation of it — and
`NNN follow-up (…)` is the same SHAPE as `NNN resolved — … (…)`. No
punctuation rule separates them. So the cost is **printed every run** rather
than guessed at, with the standing of AMBIGUOUS: on the local side it lands as
UNDECIDED noise (riir-clippy's 10 undecided rows are its own four numbers),
and on the owners side as a **false** `⛔MISATTRIBUTED` — Issue 754's exact
failure, inherited by Issue 794's in-range class. 0 live instances today, which
is the reason to print it rather than remember it.

⛔ **"Widening is unsound" was true and stated too broadly, and a POSITION
blind spot was hiding underneath it for as long as the sentence stood**
(Issue 823). The unsound widening is *dropping the DISCRIMINATOR* — the rule
that nothing may sit between the number and its delimiter, which is what
rejects `follow-up` and `resolved`. That rule says nothing about WHERE in the
heading the number sits, and both patterns had been anchored, incidentally, to
the kind LEADING the line. Six repos write the date first — riir-clippy's
`## 2026-09-16 — Issue 113: the auto-oracle` — and the whole family was
unreadable, 74 records deep. The repair keeps the discriminator exactly and
moves it one position over; arm 2's negative is now pinned in BOTH positions
and still rejects. (That `riir-clippy` is load-bearing, not decoration: the
specimen is a citation, the instrument cannot tell a quoted one from a live
one, and naming the owner in its own window is this file's own prescribed
repair — reached, again, by a paragraph documenting the class.)
- ⛔ **The meter was anchored to the same position, so the blindness detector
  was blind to the same thing — in the direction that reads as clean.**
  `heading_style_blind` is the width bound whose entire job is printing the
  cost of this class. Measured before the repair: **riir-chain printed
  `heading_unread=0/1`, a PERFECT score, over 21 records of which 20 were
  unread; riir-dapps printed `0/0` — nothing to measure — over 23.** The
  paragraph above says the meter measures "exactly the style gap"; it
  understated it by **96 records, 39%**, and the sentence you are reading is
  why a width bound gets its own floor.
- The symptom was a RATCHET, not noise: riir-clippy's own Issue 113 is
  recorded by nothing but a date-led heading (no file, no `git log` deletion —
  the Issue-754 shape), so its citations fell to IN-LOCAL-RANGE and breached
  `max_in_local_range`. ⛔ **That pin had been re-typed the previous day for
  this exact cause, with the mechanism correctly diagnosed in the pin comment,
  and was 13 within 24 hours** — this file's own "a pin file re-typed after
  every run is a diary" reached by two sessions in a row. **Read
  `heading_unread=a/b` on the sweep's per-repo line before adjudicating an
  IN-LOCAL-RANGE count**: a repo at or near `b/b` cannot have that count
  trusted as an editorial quantity at all.
- Effect: workspace IN-LOCAL-RANGE **54 → 27**, two ratchets **tightened** in
  the landing commit. The residual unread are 781's original `resolved —`
  family, **untouched** — 823 did not close 781's class, it made 781's own
  figure honest.
- ⛔ **That residual is DERIVED per run, so it is not written here as a
  number.** Two sessions quoted it hours apart as **209** and **213** and both
  were right for the run that produced them: it moves whenever a sibling repo
  edits a HISTORY heading, which is the whole reason 781's figure is printed
  rather than remembered. **Take it from the sweep's own `heading oracle`
  summary line**, never from this bullet — the same rule this section already
  states one level up for `heading_unread=a/b`, reached a second time by a
  paragraph *about* that rule carrying a stale constant of its own.

⛔ **And the rule was anchored a THIRD time — to the DELIMITER SET — where the
biggest unread family is this repo's own house style** (Issue 828). Issue 823
moved the discriminator one POSITION over and left the delimiter alternation
where it was: `(` for the leading form, `[:,]` for the dated one. The
workspace's most common title delimiter is the **em dash**, in neither set.
Measured by reading the residual instead of arguing about it: of the records
the oracle declined, **56 are `## Issue 788 — <title>: CLOSED (date)`** —
katgpt-rs's own newest closes, in the repo that owns the instrument — and **4
are `## <date> — Plan 062 (…): title`**, rejected only because the leading form
accepted `(` and the dated one did not. Neither family violates the rule; the
pattern could not spell their delimiter. Both read now, the discriminator
untouched: arm 2 pins `## Issue 048 follow-up — …` as a negative in the NEW
delimiter, and `## Issue 049-class — …` (a live riir-ai shape) pins that the
ASCII hyphen is a delimiter only when space-separated on both sides.
- This is **not** the widening the paragraph above calls unsound, by that
  paragraph's own test: adding a delimiter keeps the rule, dropping the
  discriminator abandons it.
- ⚠ It does **not** narrow Issue 823 T5's question — it corrects the
  DENOMINATOR. The `resolved` / `closed` / `T3` / `Arm C` family is untouched
  and still has no punctuation rule separating it from `follow-up`; ~60 rows
  that were being counted against that question were never part of it.
- ⛔ **T5 is ANSWERED, by PRICING it rather than by settling the semantics**
  (Issue 828 T4). The semantic answer stays NO and arm 2 still pins it. The
  question nobody had asked is what the rule would BUY: `heading_allocated`
  is one member of a union whose every other member answers from a FILE, so a
  record whose number is already known contributes nothing whichever way the
  rule goes, and **only the residue can change a verdict**. Measured over 16
  repos: of the unread records, **2** name a number no other oracle knows —
  both the Issue-754 never-committed shape. A figure two sessions had read as
  a backlog is ~99% redundant, and adopting a rule this file calls unsound to
  recover it does not survive its own arithmetic. Both quantities are printed
  every run on the sweep's `heading oracle COST` line and per repo as
  `novel=` beside `heading_unread=a/b`; **take them from there**, which is the
  rule the bullet above already states for the unread count itself.
- ⛔ The arms could not have been written before the fixture's `write_text`
  gained an `encoding=`: on this cp874 box the em dashes in the FIXTURE were
  silently written as byte `0x97` and read back as U+FFFD, so a delimiter arm
  passed for the wrong reason. That is the whole of Issue 829, found by an arm
  failing in a way that made no sense.

⚠ **A document that discusses a misattribution has to reproduce it**, and the
instrument cannot tell a quoted specimen from a live one: the Issue-780
write-up above introduced **4 rows of the very class it documents**. The repair
is not a pin — it is to name the true owner inside the citation's own 3-line
window ("riir-train Issue 513, written up as `katgpt-rs Issue 513`"), which
clears the row *and* makes the sentence followable. Reach for that before
ratcheting a ceiling for prose about prose.

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

⛔ **There are TWO spellings of that zero and this paragraph was written
about one of them (Issue 856).** A file whose entire body is

```rust
#[cfg(feature = "x")]
mod tests { … }
```

produces a byte-identical outcome — same empty binary, same `ok. 0 passed`,
same exit 0 — and `cfg_gated_target_audit`'s predicate was a single regex for
the INNER attribute, whose comment distinguishes `#![cfg]` from `#![allow]`
and never decided anything about the outer-on-a-module form. **An unstated
blind spot, not a scoped exclusion**, which is the worse of the two: a stated
exclusion is re-readable. It printed `SILENT-NOW 0` for this repo over **26**
such targets, 7 of them named `*_goat`, hiding **175 assertions** that were
reporting a green zero to anyone who invoked them by name. `cfg_body` reads
both spellings now, so every consumer inherits it; the predicate is
**the gated items are the WHOLE body**, never "a `#[cfg] mod` exists" — a
file with one live ungated `#[test]` is not zeroed, and `tests/test_freeze_thaw.rs`
is the measured specimen that was miscounted before the classifier existed.
Two rules the same issue measured rather than reasoned toward: a **run** of
gated modules is gated by **`any(...)`**, not `all(...)` (it empties only when
every module does — so it is the class cargo's AND-only `required-features`
cannot express), and a top-level `use` never makes a target non-empty and is
skipped.

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

The audit prints site rows only for the four severe classes, so the
UNRESOLVED bucket appears in the tally and nowhere else.
`scripts/list_unresolved_percentile_sites.py <repo>` dumps those rows for the
per-site read that resolves each to OK / DEGENERATE / not-a-percentile; the
2026-09-04
workspace-wide read (every UNRESOLVED row, all 16 repos) is recorded in its
own docstring.

**The POPULATION is what git TRACKS — `scripts/tracked_walk.py`, one copy
(Issue 777).** A filesystem walk behind a hand-typed directory-name skip set
is not the same set, and a name list cannot express "not ours". Measured, on
the run of this sweep that found it: mmorpg-remaster's `mmorpg/` is
gitignored AND its own git repository, so its 1404 `.rs` files were credited
to the outer repo — and produced this audit's only TRUNC-VAR finding at an
address where the repair cannot be made. That is worse than a false positive;
it is a **correctly-shaped defect at the wrong address**, one axis over from
what `issue_citation_gate.py` exists for. riir-train's cargo OUT_DIR sources
sit under `.runs/target-release/`, `-cuda`, `-v2cpu`, `-bench`: the skip set
names `target`, and none of those IS `target`.

Read the second-order damage, because it is the part that lasts: **two
`percentile_drift_floors.txt` rows were unsatisfiable by any tracked walk of
their repo** — riir-train `2500` against 1129, and riir-chain `500` against
the **460** that repo had on the day the file was written. Neither had ever
been edited. A floor exists to catch an instrument going blind; a floor
measured over content the repo does not own reds on every box but the one and
the hour that produced it, and teaches whoever hits it that the sweep is
noise. Tracked-only was landed twice before (Issue 734 for the trap audit —
"25 findings in a gitignored vendored drop no repo owns"; Issue 738 T3 for the
platform/wasm32 pair) and not generalised, which is the whole reason the walk
now lives in one file with its own 8-arm self-test. `vendor/` exclusions ride
the per-repo line; the `.git` probe is load-bearing (`git -C` walks UP, so a
non-repo directory inside a repo would answer with its PARENT's paths); a tree
with no `.git` — `git archive`, a synthetic self-test fixture — falls back to
the walk rather than erroring.

⛔ **`.git` answers TWO questions and each spelling is wrong for the other
one** (Issues 835, 836). Pick by the question, never by symmetry with the
nearest instrument:

| the question | probe | why |
|---|---|---|
| *Is this a canonical REPO?* | **`.is_dir()`** | a `git worktree` has a `.git` **FILE**; counting one attributes its manifests to a repo that does not exist (835) |
| *Can I run git rooted HERE?* | **`.exists()`** | a worktree is a perfectly good checkout, and `.is_dir()` silently says otherwise (836) |

Never write a fresh contract-repo walk for the first — **delegate to
`skill_repo_set_gate.derive_repos`** (through `repo_alias.disk()` if you
intend to OPEN the directories); `population_sync_gate` exists to catch two
predicates disagreeing about the population. The second is
`worktree_state.is_checkout` — **delegate to it too**, and it is the spelling
`tracked_walk.py` had right all along. ⛔ The second direction is the SILENT one: all five
`worktree_state` guards took `.is_dir()`, and every false branch of theirs
returns the value meaning *nothing to report* — so in a worktree
`sweep_advisory` returned `[]` against a modified tracked file, and a sweep run
there re-pinned from another session's in-flight edits believing it had read
HEAD. Issue 797's founding defect, reintroduced inside the mechanism built to
prevent it, by the repair for the *other* question. ⚠ And the probe is not the
whole of it: a path CONSTRUCTED under `<root>/.git` (`FETCH_HEAD`) cannot
exist in a worktree either — ask git (`rev-parse --git-path`), which answers
ABSOLUTELY there and relatively otherwise.
- ⛔ **Three more instruments carried a PRIVATE copy of the second question,
  all on the wrong spelling, and only one of the three was severe — which is
  why they were measured before being touched** (836 T2, the demand 835 T2
  makes of itself). `console_encoding_gate.tracked_scripts` and
  `sweep_advisory_membership_gate.tracked_sweeps` fall back to a `scripts/`
  GLOB, so a run from inside a worktree gets a bigger POPULATION rather than a
  null verdict: measured in this repo with one untracked scratch file planted,
  `✗ UNDEFENDED zz_scratch_probe.py` and **three** `✗ UNWIRED` rows (one per
  mechanism) — cries-wolf reds demanding repairs to a file no contract claims,
  and with the delegation both runs are byte-identical to the ordinary one.
  `numbering_drift_sweep.head_listing` is the severe shape (its `None` skips
  the entire head-provenance adjudication) and is **measured UNREACHABLE by
  its own caller** — `contract_repos` excludes worktrees by construction,
  verified against this box's real `riir-chain.w152`. Repaired anyway, and an
  arm now pins that exclusion: *correct by a caller's property* is exactly
  what 835 → 836 cost once, and the next caller inherits a trap rather than a
  guarantee. All three delegate now; the shared fixture is
  `worktree_state.worktree_fixture` (0.167s, a REAL `git worktree add`),
  because a hand-written `.git` file asserts the probe and not the behaviour.
- ⛔ **A name match that never fires is the same silence one layer up** (836
  T4). `numbering_drift_sweep`'s gate cross-check — the one AGENTS.md
  elsewhere calls *"asserted, never assumed"* — was guarded by `repo.name ==
  REPO_ROOT.name` with no else, so it simply did not run from a checkout
  directory not named for its repo (a worktree at `E:/git/katgpt-rs.w836`, the
  convention this box already uses for riir-chain; a fork clone; an alias
  row). `.name` twice was also the wrong comparison: the derived repos carry
  the CONTRACT spelling and `REPO_ROOT` carries the on-disk one, so a box with
  a `repo_alias.local.txt` row for this repo took the silent branch every run.
  It is a named failure now, never a deferral — a partial clone can lack any
  sibling, but not the repo the script is running out of.

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

## A ratio of two SEQUENTIALLY-timed arms measures the BOX — `scripts/sequential_ab_timing_audit.py`

```rust
let t = Instant::now(); for _ in 0..N { a(); } let a_ns = t.elapsed();
let t = Instant::now(); for _ in 0..N { b(); } let b_ns = t.elapsed();
assert!(a_ns as f64 / b_ns as f64 >= 0.90);
```

Two sequential arms of the **same** work measured **+5.2% and +21.7%** thirty
seconds apart on a loaded box (Issue 723 T5), so a 10% bar is a measurement of
the scheduler. `tests/common/ab_timing.rs` is the treatment this repo already
ships — interleaved `(a-chunk, b-chunk)` pairs, median across pairs, and a loud
zero when the optimiser deletes an arm.

⛔ **The finding is the DISCOVERY METHOD.** Issue 723 converted the 8 its census
could see; Issue 831 converted a 9th by walking into it; Issue 833's
`bench_105_gdn2_goat.rs` GOAT 2 was caught by
`scripts/x86_64_execution_matrix.sh` reporting it PASSED-ALONE (0.844 against a
0.90 bar in cell 8, then 3/3 alone) — and its GOAT 6 joined it on 2026-09-20,
the same catch by the same instrument (spread 0.306 against a 0.30 bar, 3/3
alone), this time the N-arm FLATNESS shape: four positions each measured in
one sequential window, repaired with the shared module's new `best_of_arms`
primitive (round-robin interleave + per-arm minimum; alone spreads
0.038–0.102). **None of the four was found by a
census** — the treatment has existed for months and its members are found by
tripping over them (Issue 834).

```bash
scripts/sequential_ab_timing_audit.py            # this repo
scripts/sequential_ab_timing_audit.py ../riir-ai # or one, by path
scripts/sequential_ab_timing_audit.py -v         # every row, not just findings
```

- A **report, exit 0** — except a blindness floor or a failing self-test, which
  exit **2**; the self-test runs on every invocation.
- **ADOPTED** (names `common/ab_timing.rs` or calls `ab_median_ratio`, read off
  the UNMASKED text because `#[path = "…"]` is a string literal) ·
  **HAND-ROLLED** · **SEQUENTIAL**
  (≥2 `Instant::now()` + a ratio whose BOTH sides are timing-derived, minus
  count-like denominators) · **UNRESOLVED**, which is **not clean** and is never
  folded into either neighbour — a two-arm comparison the regex cannot see lands
  there, and so does an ordinary single-arm latency bar that is not the class.
- ⛔ **`ADOPTED` matched a NAME where the treatment is a SHAPE, and the bucket
  it leaked into is the one that reads "go migrate this"** (Issue 833 T3).
  `ADOPTED_RE` spells two literals; interleaved paired arms + a per-pair ratio
  + a median across pairs is a shape that can be written without either.
  Measured specimen: `bench_657_clustered_lm_head_bound.rs` read **SEQUENTIAL —
  "the class"** while its own doc block describes alternating A→B / B→A
  ordering and a median of per-pair ratios, i.e. a **stricter** treatment than
  the shared harness. **HAND-ROLLED** is its own verdict and is **never folded
  into ADOPTED**: that one means *uses the shared harness*, this one means
  *duplicates it* — both are TREATED and neither is a migration candidate, but
  only the second is a DRY finding, and pooling them would report the treatment
  as universal. Both halves of the predicate are required and each negative is
  a real shape: a per-pair ratio with no reduction is a log, and a reduction
  with no per-pair ratio is median-of-A-over-median-of-B, which **is** the
  defect. ADOPTED still outranks it, so a migrated target that kept its old
  loop is not demoted.
- ⛔ **A second resolver, because `_T` reads a NAME and the class is a VALUE**
  (Issue 833 T3). A two-arm ratio whose locals are `a`/`b`, `t_3d`/`t_2d` or
  `overhead_ns`/`baseline` spells none of `_T`'s tokens and landed in
  UNRESOLVED. `provenance_hits` binds timing provenance instead — a local
  assigned, transitively, from an `.elapsed()` value is timing-derived whatever
  it is called — and the transitive hop is what reaches the ordinary shape,
  where only the FIRST binding mentions `elapsed` and the one that gets
  compared is two hops away. Measured here: **8 targets, 8 of 8 TRUE on a
  per-site read**, every one feeding a bar or an assert. It is a **second**
  resolver, not a replacement: `RATIO` still decides the easy majority, a site
  both can see is counted once, and `COUNTY` still applies — a timing local
  over a count is a rate whichever resolver found it.
  - **Two expression shapes, ONE provenance pass**, because they are the same
    resolver and computing provenance twice is the expensive half: `A / B`, and
    `(A - B) / C` — a **RELATIVE DIFFERENCE**, which is algebraically `A/C -
    B/C` and is the same two-arm comparison wearing a percentage. That is half
    of the STATED "subtraction or a percentage" blind spot, and it was the
    better-hidden half: `ANY_RATIO` misses it only because the numerator is
    parenthesised. Measured: **14 targets carry it, 7 of them resolved by
    nothing else**, and two are GOAT bars — `pipeline_pruner_goat` asserts
    `latency_improvement >= 0.20` and `static_cal_goat` `>= 0.05`.
  - ⛔ **A bar's exposure is its SLACK, not its size, and reading the bar alone
    is the mistake this instrument cannot help you avoid.** A `>= 0.05` bar
    against a *measured* 0.99 has 94 points of head-room and ±21.7% sequential
    drift cannot reach it; the same bar against a measured 0.07 is one
    preemption from red. That is AGENTS.md's own CLAIM DIRECTION axis — **not
    statically decidable**, which is exactly why this stays a report and why a
    SEQUENTIAL row is a CANDIDATE for T2's per-target read rather than a
    finding. Do not quote a bar out of this report as a flaky gate without the
    measured value beside it.
  - ⚑ **Measured, and it refuted the first version of this bullet.**
    `static_cal_goat` was RUN (`--features static_cal_tables,kvarn --release`):
    `sinkhorn=46772µs static=15µs`, **improvement 100.0%** against the `>= 0.05`
    bar — a 3118x gap and 95 points of slack, which ±21.7% drift cannot reach.
    `pipeline_pruner_goat` likewise: `baseline=5000200ns pruned=122300ns`,
    **97.6% against its 20% bar**, 77.6 points of slack. **2 of 2 flagged bars
    have head-room a loaded box cannot cross.** Both rows are correctly IN the
    class (two sequentially-timed arms, compared) and NEITHER is a migration
    candidate — exactly the distinction a count cannot carry. Two `cargo test`
    runs settled what a paragraph of reasoning from the bars had got backwards.
  - `(x - x) / x` is armed alongside `x/x` (a self-difference is zero, not a
    comparison), and `COUNTY` reaches the difference form's **denominator**
    too — `(a - b) / n_tokens` is a per-token delta, i.e. a rate. That arm is
    what keeps the widening honest, and it reds when removed.
  - ⚠ A subtraction that is **never divided** stays UNRESOLVED and is correct
    there: `ppot_bench`'s `t_027 - t_greedy` is a printed `Duration` span.
  - ⚠ SEQUENTIAL has never meant "feeds an assertion" — it means two
    sequentially-timed arms are compared. Some rows print the comparison and
    gate nothing; that is Issue 833 T3's third resolution bucket and applies to
    the whole bucket, not just these. The widening did not change that rule.
- `mask_file` is **imported** from `platform_dead_code_audit`, not re-written:
  three sibling instruments have reported findings inside their own fixture
  strings, and a second hand-rolled Rust lexer is a second thing to get wrong.
- ⛔ **Re-measured 2026-09-19 after the `n == 0` ordering fix: 1 HARNESS ·
  **12 ADOPTED** · 9 HAND-ROLLED · 154 SEQUENTIAL (35 asserting nothing) ·
  784 UNRESOLVED** over 2436 target files / 8123 tracked `*.rs`, 14 repos.
  **Every ADOPTED figure printed before this — 7, and the 8 the HARNESS bullet
  corrects — was an UNDER-COUNT by 5**, because `classify` short-circuited
  `n == 0 → UNTIMED` ahead of the ADOPTED test. ⚑ **The direction is the worst
  available: a target became invisible by being MIGRATED**, since the end state
  this audit pushes toward is one with no `Instant::now()` of its own. The
  specimen that proves it is `bench_171_thinking_prune_goat` — Issue 831's own
  repair, which dropped out of the census on the day it was fixed — so the
  migration backlog was shrinking its own denominator as it was worked.
- Re-measured 2026-09-18 after `comparison_hits`, over the **14** contract repos
  present on this box: **1 HARNESS · 7 ADOPTED · 9 HAND-ROLLED · 154 SEQUENTIAL
  (35 asserting nothing) · 784 UNRESOLVED** over 2435 target files / 8120
  tracked `*.rs`. ⚠ Its ADOPTED 7 is the under-count described above. ⚠ **Not comparable term-by-term with the 17-repo line below**
  — the population is three repos smaller, so SEQUENTIAL rising 152 → 154 across
  a SMALLER walk is a floor on what the sixth resolver moved, not a measurement
  of it. Take both figures with their repo count, which is the whole reason this
  bullet prints one.
- Measured 2026-09-18 over 17 repos, after Issue 833 T3: **1 HARNESS ·
  7 ADOPTED · 9 HAND-ROLLED · 152 SEQUENTIAL (35 asserting nothing) ·
  794 UNRESOLVED** over 2515 target files / 9125 tracked `*.rs`. Read the SEQUENTIAL figure as a
  MAGNITUDE: three predicates over overlapping populations returned
  55 · 57 · 60 for this repo alone. Take every figure from a run.
- ⛔ **`HARNESS`, because the module certified ITSELF.** `ADOPTED_RE` matches
  `common/ab_timing.rs` by NAME and the harness's own path contains that name,
  so `tests/common/ab_timing.rs` was counted as an adopter of itself and
  inflated the one figure this section is quoted for — the reported 8 was
  **7 adopters + the harness**. It gets its own verdict rather than an
  exclusion: dropping `tests/common/` from the walk would make a helper module
  carrying a real ratio invisible, which is the silent direction, and cargo
  does not auto-discover `tests/` SUBDIRECTORIES as targets anyway (7 such
  files here, 736 real targets against the 743 reported). Measured before it
  was changed: without the short-circuit the harness classifies UNRESOLVED, so
  **nothing was being masked — the defect was the COUNT.** Checked in the
  other direction too, which is the one that hides things: of the 8, one more
  (`bench_270_gauge_invariant_goat.rs`) calls only `best_of_us` — the ABSOLUTE
  budget, not the A/B treatment — and it carries no two-arm ratio either, so
  the false-green bucket is measured and EMPTY.
- ⛔ **"ADOPTED is 0 in every repo but katgpt-rs — the class generalised and
  the harness did not" was TRUE in its first clause and too strong in its
  second, and the second is the one anybody acts on.** ADOPTED is still 0
  everywhere else. But the shape-based verdict measures **9 HAND-ROLLED, and
  6 of the 9 are OUTSIDE this repo** — riir-ai 3, riir-neuron-db 2,
  riir-train 1. ⛔ **That paragraph then concluded the duplicated treatment (9)
  outnumbered the shared one (7), and the conclusion is RETRACTED — the 7 was a
  classifier artifact.** `classify` returned `UNTIMED` at `n == 0` *before*
  testing ADOPTED, so a target that adopts the harness COMPLETELY — delegating
  all timing to `ab_median_ratio` and keeping no `Instant::now()` of its own —
  fell out of every bucket. Corrected count: **12 ADOPTED, 5 of them
  previously invisible** (`bench_817_argmax_dispatch_ab`,
  `bench_171_thinking_prune_goat`, `bench_257_gpart_adapter_goat`,
  `bench_839_kron_tile_goat`, `substrate_gate_goat` — all verified as real
  adopters, `#[path]` + `mod ab_timing` + live `ab_median_ratio(` calls, zero
  false positives). So the **shared** treatment (12) outnumbers the
  **duplicated** one (9), and the DRY finding survives only in the weaker form
  that 6 of the 9 duplications sit outside this repo. The **treatment**
  generalised; what did
  not is the shared MODULE, independently re-written six times. That is a DRY finding rather
  than a coverage gap, and it is the measurement Issue 833 T5 / 834 T3 ask for
  before the cross-repo question is answered — it does not answer it, because
  whether `ab_timing.rs` should become a shared crate is an owner/boundary
  call. ⚠ Read the ADOPTED 7 → 8 and 2516 → 2515 deltas as **sibling drift
  since that measurement**, not as this change: `ADOPTED_RE` and the target
  predicate were untouched by T3.
- ⛔ **No verdict half, deliberately** (Issue 833 T4, 834 T4) — do not add one by
  symmetry with the sweep family. Migration is a per-target read on FOUR axes,
  none of them statically decidable: the `a`/`b` ORIENTATION (`AbRatio::median`
  is a TIME ratio and half these gates state a THROUGHPUT claim — backwards
  inverts the bar silently); the CLAIM DIRECTION (a performance **win** with no
  slack must be precise, so interleaving is right; an overhead **ceiling** with
  deliberate room is nearer the `best_of_us` absolute-budget family and gains
  almost nothing — `belief_drafter_goat.rs` g8 and
  `bench_217_belief_drafter_goat.rs` B6 compute the IDENTICAL
  `cached_us / uncached_us` and assert opposite things); the chunk size off the
  target's own printed range; and `black_box` at both ends. ⚠ Arms must also be
  a PAIRED A/B — where they are not, the fixture needs reworking before the
  timing does. A slice chosen from
  a classifier with hundreds of unresolved rows is a slice chosen from a guess.
- ⚠ **STATED blind spots, NARROWED by Issue 833 T3:** a ratio built through a
  helper; a subtraction or percentage that is **never divided**; two arms
  differenced off ONE `Instant::now()`; and orientation, which is not
  statically decidable. Two are CLOSED by `provenance_hits`: a ratio whose
  locals are timing-derived by VALUE but not by NAME, and the **divided** half
  of the subtraction/percentage shape.
- ⛔ **A SIXTH, closed by `comparison_hits`: every other resolver is
  DIVISION-shaped, and the best-documented member of the class has no
  operator.** Issue 831's P3 — a measured 20% coin flip — was
  `assert!(ns_frozen < ns_uniform)`: two sequentially-timed arms compared as a
  bare INEQUALITY, with no ratio expression and no ratio-shaped identifier to
  key on, so the resolver set would have reported its file UNRESOLVED
  throughout. The axis is **two timing-derived identifiers in ONE comparison**;
  a ratio is one surface form of it (`g8`'s
  `assert!(cached_us < uncached_us / 2.0)` is an inequality that merely
  contains a `/`). Measured here: **5 targets carry the shape, and 4 were
  already SEQUENTIAL by LUCK** — caught by a different expression elsewhere in
  the same file rather than by the comparison they assert on — leaving a
  one-target resolver gap (`fast_bpe_goat_pretok`, `warm_ns < cold_ns`, zero
  slack, measured at 94 points of head-room and therefore NOT a migration
  candidate). Read that 4-of-5 as the finding: coverage resting on file
  content is not reach.
  - ⚠ Two rules it had to get right, both armed: the argument extraction is
    **balanced, not line-scoped** (Rust asserts wrap, and the line-scoped first
    draft printed a confident **0 over every bucket** — blind to the exact shape
    it was written for), and **`COUNTY` must NOT filter it** (COUNTY rejects
    `time / count` as a rate, but comparing two *rates* is still comparing two
    arms — `region_per_iter < token_per_iter` names a count token in both
    operands and is a live specimen).
  - ⚠ STATED cost: scoped to `assert!`/`panic!` ARGUMENTS, so a two-arm
    comparison feeding only a `println!` or an `if` is not seen. An ordinary
    `while start.elapsed() < deadline` has two timing-derived operands and is a
    loop bound, not a claim.
  - ⛔ **It also SPLITS the "subtraction never divided" blind spot**, and the
    specimen is in a sibling: riir-train's `bench_568_mi_audit_goat` computes
    `mi_added = with_mi - base` and asserts `mi_added < base` — no `/` anywhere,
    so `REL_DIFF` and `ANY_RATIO` both miss it, while the difference IS
    compared and `timing_locals`' transitive hop makes it timing-derived. So:
    *never divided AND never compared* is still open; *never divided but
    compared in an assertion* is closed. ⚠ Its exposure is UNMEASURED — the bar
    reduces to `with_mi < 2 × base` and the target's own comment predicts
    ~1.05x, which is an argument and not a measurement; running it is that
    repo's call.
  - Cross-repo (14 repos on this box), measured as a WITH/WITHOUT delta over
    ONE walk rather than by differencing two runs' totals: **10 targets carry
    the shape, 2 were UNRESOLVED without it** (katgpt-rs 1, riir-train 1),
    **8 of 10 covered by luck**. Differencing the published totals would have
    said `152 → 154` across a population three repos smaller — confounded, and
    a floor at best.
- **UNRESOLVED carries two sub-populations with opposite priors, and pooling
  them is this bucket's own hazard one level down** (Issue 833 T3). A
  **1-timer** row is mostly an ordinary single-arm bar; a **2+-timer** row is
  where every STATED blind spot above lives. Measured: **327 · 467**
  workspace-wide, **132 · 153** in this repo — so more than half the bucket is
  the half worth reading first. Printed on the summary line and tagged per row
  under `-v`. A **triage aid, never a verdict** — the percentile audit's `tail
  support` standing: it ORDERS the rows so a read starts where it can change
  an answer, exactly as the `heading oracle COST` line does for the ~99%
  redundant residue one section up.
- **`GATES` vs `report` — T3's third bucket, and it is where T2 must NOT
  start.** A SEQUENTIAL row that contains no `assert!` / `panic!` cannot have
  its verdict flipped by the box, **because it has no verdict**. Measured:
  **35 of 152 SEQUENTIAL assert nothing** workspace-wide (**20 of 72** here),
  and **113 of the 467** 2+-timer UNRESOLVED rows. So T2's backlog in this repo
  is ~**52** gating rows, not 72 — which is the whole use of the annotation.
  - An **ANNOTATION, never a bucket**, and that is armed: it must not change a
    verdict, or the counts stop being comparable with their own history. It is
    also orthogonal to every bucket, so it applies to the WHOLE population and
    not only to rows a new resolver moved.
  - ⛔ **And the TRUE direction has a converse that halves T2: `gates=True`
    does NOT mean a perf BAR exists.** It means an `assert!` is present, and a
    target can assert **instrument health** — every round survived, the ratio is
    finite — while *printing* its reading deliberately. `bench_843_ternary_size_sweep`
    is the named specimen and says so in its module doc: a bar written before its
    sweep would have been a bar written from a hypothesis that turned out wrong.
    No box state can flip a verdict a target does not have.
    - Measured here with a proxy — does ANY `assert!` argument name a
      timing-derived local? **Of 53 SEQUENTIAL+gating rows, 21 have none**, and
      two were verified by hand: `bench_simd.rs` computes
      `speedup = sparse_tps / dense_tps` and only ever `println!`s it (its lone
      `assert!` is a dispatch check), and `bench_002_density_routing_goat`
      computes its relative difference and asserts elsewhere. So **T2's backlog
      is ~27, not ~52** — the annotation was over-stating it by nearly half.
    - ⚠ A further **5 are UNDECIDED and must not be folded into either side**:
      the proxy resolves `let` bindings, and `bench_008_gpart_pruning_goat`
      computes `start.elapsed().as_nanos() / iterations` as a block **tail
      expression**, so `timing_locals` sees nothing. That is the proxy going
      blind, not evidence of bar-lessness — this bucket's own rule, applied to
      the instrument measuring it.
    - ⛔ **`~52` and `~27` are both STATIC proxies for a quantity that is
      decidable by EXECUTION, and when it was executed the answer was 3**
      (Issue 833 T2, 2026-09-19). Every root-`tests/` row the audit flags
      `SEQUENTIAL [GATES]` — 38 files, 36 cargo targets — was RUN in release at
      its own `required-features`, and its printed value read next to its bar.
      **3 are candidates**; every other row was rejected on measured SLACK,
      most of them by an order of magnitude (`channel_simd_goat` G5 measures
      **84.3%** against a 5% bar, `bench_turboquant` large_kv **0.244** against
      1.05). A `gates=True` count is a count of *targets with a verdict*, and
      the backlog is the count of *verdicts the box can flip* — a strictly
      smaller thing that no static pass can reach.
      - The threshold is this file's own two numbers, neither previously used
        as one: **±21.7%** (Issue 723 T5, two sequential arms of identical work
        on a loaded box) and **±6%** (833's idle per-round spread). `< 6` pts
        flakes idle · `6–22` flakes under the load the matrix runs at · `≥ 22`
        is out of reach.
      - ⚠ **Bar points alone are insufficient for an OVERHEAD bar** — what
        matters is whether the measured SIGNAL stands above the envelope.
        `bench_176` is a candidate at 13.3 points (a 0.07 µs overhead on a
        1.09 µs baseline) and `bench_249` is not at 34%.
      - ⛔ The run also found **two classes it was not looking for**, both
        filed apart so the counts stay unpooled: **Issue 855** (a latency
        ceiling satisfied by a loop the optimiser DELETED — `0.0 ns/op` over
        100 000 iterations asserted `< 10 000`, which *cannot* flip and needs
        the opposite repair) and **Issue 856** (`#[cfg(feature)] mod tests {}`
        zeroing a target invisibly to `cfg_gated_target_audit`). Two targets
        carry 833's class AND 855's.
      - **Take the backlog from a RUN, never from a proxy in this file** — the
        figure in this bullet has now been wrong twice in the same direction,
        and both times the correction came from measuring rather than from a
        better predicate.
  - ⚠ A **lower bound**, deliberately: a target can also fail by returning
    `Err`, by `process::exit`, or through a helper this pass cannot follow. So
    `report` ORDERS a read rather than deciding it — it must not be read as
    "this target is safe".
  - It reads the MASKED text: `assert!` inside a fixture string is not an
    assertion, the class three sibling instruments here have each met.

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

Canonical failure: mmorpg-remake's `ci_feature_guard.sh` — the script its
`rust.yml` runs — could not fail past layer 13 for months, because its
layer-13 trap named two variables assigned ~20 and ~45 lines later. It stayed
hidden because a ratchet ceiling had been red for three commits and stopped
every run *before* the bad line (mmorpg-remake `26a18191`).

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
surface?** A row cannot notice a package it does not select, and mmorpg-remake
had a positive `#[cfg(target_arch = "wasm32")]` block that no row built and
that had been **uncompilable since it was written** — it called a
`cfg(not(wasm32))` function (`.issues/010` T2, `.issues/738`).

```bash
scripts/wasm32_surface_audit.py            # all contract repos (derived)
scripts/wasm32_surface_audit.py ../riir-ai # or one, by path
```

- A **report, not a gate** (exit 0). Four buckets: **NAMED** (a row selects
  it by `-p` or a literal `--manifest-path`), **BY-DEP** (Issue 774: no row
  names it, but a named/derived package reaches it through non-optional
  in-repo path-dep edges — reachability, never folded into NAMED, because
  that coverage dies by a dep-graph edit in someone else's manifest),
  **UNRESOLVED** (a `--workspace` or *derived* row exists — whether it reaches
  this package is the separate-workspace axis, undecidable statically),
  **UNCOVERED** (no row could reach it). **UNRESOLVED is not clean** and is
  never folded into either neighbour. `--self-test` proves the by-dep
  detectors fire in BOTH directions (five canary verdicts: named / by-dep /
  uncovered / optional-not-credited / workspace-table-resolved).
- The predicate is the **positive** cfg: `not(target_arch = "wasm32")` is an
  ordinary native-only guard and counting it inflates everything (riir-ai
  `.issues/892` T4). **Comment lines are excluded** — prose explaining a cfg
  is not a cfg, and the comment recording why a file has *no* wasm32 arm
  otherwise makes that file read as browser code.
- ⛔ **And it reads ATTRIBUTES, not lines** (2026-09-15). A line scan cannot
  tell a real `#[cfg(target_arch = "wasm32")]` from one inside a raw string,
  and riir-clippy's `src/platform_audit.rs` is four such fixtures — Rust
  source embedded in `r#"…"#` as test INPUT for the platform-dead-code
  classifier. Those four were that repo's ENTIRE count, so the audit reported
  `1 package UNCOVERED, its arm compiles nowhere` about a repo with no wasm32
  code at all, and hard-failed its sweep row. **Third instrument to meet this
  class**: `platform_dead_code_audit` masks literals and says why, and
  `subprocess_encoding_gate` moved to an AST because its text scanner *"reported
  four offenders in the gate's own file — every one a fixture string inside its
  `selftest()`."* The masker is IMPORTED from `platform_dead_code_audit`, not
  re-written: it is a hand-rolled Rust lexer with its own measured defect
  history, and a second copy is a second thing to get wrong.
- ⚠ `cfg!(target_arch = "wasm32")` is split out and **not counted as
  surface**. It is a RUNTIME branch — it compiles on every target, so no lane
  can fail to reach it and it is not the question this audit asks. Seven sites
  workspace-wide; printed on the per-repo line as `EXCLUDED` so the decision is
  re-measurable rather than remembered, never folded into the gated count.
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
- The DEPENDENCY EDGE (Issue 774, 2026-09-14): the row predicate is
  dep-blind — `-p riir-shader-showcase --target wasm32` compiles the
  showcase's in-repo path deps too, so riir-shader's core+effects read
  UNCOVERED while every bundle build compiles them (compile-verified:
  `cargo check -p riir-shader-effects --target wasm32-unknown-unknown`
  exits 0). The `✓ by-dep` verdict credits exactly those edges —
  non-optional, plain + wasm32-target tables, `workspace = true` resolved
  through the root table, in-repo targets only; dev/build, optional,
  native-target, and cross-repo edges credit nothing. `mmorpg-poc-submodule`
  is the standing negative control — deliberately excluded from its repo's
  CI and depended on by nothing, it stays UNCOVERED (that repo is read-only
  here; arm-vs-row is its owner's call). Standing (measured 2026-09-14,
  post-774): **26 NAMED · 2 BY-DEP · 0 UNRESOLVED · 1 UNCOVERED** over
  216 files / 29 packages / 20 repos — the growth vs 2026-09-08 is
  sibling-added wasm32 surface, not audit drift.
- ⛔ **That standing sentence was hand-typed and asserted by nothing** until
  Issue 785 — the shape Issue 784 had just closed one instrument over, where
  the same kind of cross-repo total went **46%** stale with no run noticing.
  The verdict half is `scripts/wasm32_surface_drift_sweep.py` now, sharing the
  report's `classify_repo()` so the 738 resolver and the 774 closure exist in
  one place. **Take the figure from a run, not from this bullet.** The 16-repo
  measurement on a partial box is 213 files / 28 packages / 25 NAMED, and
  `TOTALS` in `scripts/wasm32_surface_drift_floors.txt` is pinned against that
  — re-pin it, and the four absent per-repo rows, from one full-checkout run.
- ⛔ It produced three confident wrong answers before it produced a right one,
  all in the classifier: a walk of **0 files** (a Python `\s` handed to
  `git grep -E`, which is POSIX ERE — caught only because the walk size prints
  next to the verdict), then **17** false UNCOVERED (a *derived* `-p` list —
  the better design — read as the worst result), then **2** more (a
  `--manifest-path "$unit/…"` lane read as a bare row). A classifier's bucket
  boundaries ARE the finding, and they are only testable against cases whose
  answer is known independently.

## An arm that exists and RUNS may still reach nothing — `scripts/arm_reach_audit.py`

Issue 790. `check_validation_gate.py` (below) asserts that every CHECK invokes
an arm, and had to state the limit in its own docstring: *"arm QUALITY is not
statically decidable and is not claimed here."* The first clause is true and
the second is too strong — quality is not **statically** decidable, but
**reach** is measurable by EXECUTION, and Issue 789 measured it 53 times by
hand, finding **seven** arms that certified nothing until they were re-aimed.
A census done by hand is a census that stops being done.

```bash
scripts/arm_reach_audit.py                    # the report, the CHECKS population
scripts/arm_reach_audit.py --self-test        # 27 arms over its own buckets
scripts/arm_reach_audit.py skill_repo_set     # one module, by substring
scripts/arm_reach_audit.py --include-all      # every scripts/*.py DEFINING an arm
scripts/arm_reach_gate.py                     # the VERDICT (T2) — workstation, minutes
scripts/arm_reach_gate.py --canary            # 16 arm groups over its own arithmetic
```

Mutate a module's source **outside its own arm bodies**, re-exec, run its arm,
ask whether the arm noticed. Standing (2026-09-15, after T6): **22 modules ·
559 mutants · 372 KILLED · 26 SURVIVED (live) · 0 CRASHED · 0 NO-ARM ·
0 UNREACHED · 0 BASELINE**, every live survivor pinned with a reason. The 22nd
module is the gate itself. ⚠ The intermediate figures were **21 modules · 519
mutants · 317 KILLED · 47 SURVIVED** (T3/T4) and **552 · 360 · 31** (T2); T2
closed 16 of the 47 as real gaps and T6 closed 5 more, so read each drop as
arms being written, not as the population shrinking.

⚠ **The wall clock moved from ~73s to ~370s over T3 and that is the arms
working, not a regression.** The repairs gave several gates fixture-repo arms
(temp manifests, temp docs, temp git trees), so each of the 519 mutants now
buys a great deal more assertion. Read the cost as the price of reach; it is
still a workstation report and nothing runs it per-push.

⛔ **An earlier version of this paragraph read `99 KILLED · 88 CRASHED · 33 in
4 NO-ARM · 6 UNREACHED`, and the CRASHED column was a classification DEFECT in
the harness, not a property of the code.** `run_arm` wrapped the module `exec`
and the arm CALL in one `try`, so an arm that signals by raising — 
`required_features_static_gate.selftest` returns `None` and raises
`SystemExit(2)`, and several others do the same — had every mutant it caught
filed as CRASHED. Those modules could never show a KILL at all. The two phases
are separate now (import failure → CRASHED, *evidence of nothing*; a raise
while the arm runs → KILLED, the arm noticing), and the corrected figure is
**243 killed against 127 previously, with CRASHED at 0**. Read the first
number as having been wrong in the pessimistic direction; the harness was
blaming the gates for its own boundary.

- A **report, not a gate** (exit 0), except a blindness floor or a failing
  self-test (exit 2) — a harness that generates no mutants, or whose runner
  always says KILLED, prints a *perfect* score, which is the same output as
  perfection. Two floors: `MIN_MODULES` the walk, `MIN_MUTANTS` the operators.
- ⛔ **`BASELINE` is the bucket that was missing, and one of its two arms looks
  like a PERFECT score** (T6, 2026-09-15). The harness never asked the arm
  about the module's own **unmutated** source. An arm that is *already failing*
  kills every mutant, so the module reports 100% reach having distinguished
  nothing — and it does not merely escape the gate's `MIN_KILLED` floor, it
  **inflates** it. `BASELINE-RED` (arm fails unmutated) and `BASELINE-CRASH`
  (module will not exec unmutated) are their own module-level verdicts, the
  mutants are counted but **not run**, and neither is pooled into KILLED,
  SURVIVED or `UNREACHED` — `UNREACHED` says *the arm cannot express this* and
  sends the reader to widen an arm that is not the problem. The gate walls both
  at 0. Check the **environment** first on a RED: a drift sweep whose canary
  runs the real workspace needs the same markers the gates get
  (`DOCS_GATE_PARTIAL_CLONE=1` on a known-subset box), and two of them read RED
  without it.
- ⛔ **`--include-all` had NEVER been measured, and T6 measured it** — 55
  modules, ~2400 mutants, run **module by module with a wall timeout** rather
  than as one invocation. That is the operating instruction, not a detail: one
  run is unbounded in the worst case and not resumable, and the worst case
  happened twice on the day it was written (a two-hour non-terminating mutant,
  then a *blocking C call* the watchdog provably cannot reach — 3.5% CPU, no
  children, interrupt pending). The per-module walk took ~35 minutes, named
  both stragglers, and lost nothing when one was killed. Standing over **55 of
  55** modules: **2376 mutants · 1182 KILLED · 652 live SURVIVED · 535 exempt ·
  1 CRASHED · 6 TIMEOUT · 1 NO-ARM (19 more mutants)**. ⚠ Read that against the
  CHECKS population and **not** as a comparable number: these arms cover a
  *classifier*, and the whole 652 is an unread backlog — exactly the shape
  Issue 785 forbids ratcheting, and deliberately NOT in the gate's population.
- ⚠ **The three weakest were armed on the measurement, and every one needed
  an EXTRACTION before an arm could be written at all** — T4's finding three
  more times. `feature_isolation_gate` went **4 killed of 62 → 24** and
  `citation_weight` **3 of 41 → 12**: `parse_changed_flags` was welded to its
  `git diff` call, and `attribute()`, the scoring function § Numbering
  Discipline sends you to, had **no arm at all** while its module's arm covered
  only the two INPUTS that feed it. Both are pure over plain data and neither
  needed a fixture repo. `ci_gate_coverage` was the standing worst at **4
  killed of 74** (5% reach) and is **54 of 73** (1 live survivor); it needed
  both halves of the pattern — four verdicts extracted out of `main()` and a
  branch probe injected out of `git ls-tree`.
  - ⛔ **The extraction found a live defect, which is the argument for doing it
    even where the arms are the goal.** `main()` classified for DISPLAY with one
    ladder and COUNTED with independent predicates, so a repo carrying both a
    partial command and a data-borne signal was counted **twice**: the summary
    read `6 full; 1 dynamic; 9 partial; 1 no CI` over **16** repos. A verdict
    and a tally that disagree about how many states a repo is in is the same
    class as a count that is not a checksum.
  - ⚠ **An injected probe asserts the ABSTRACTION, so the production path is
    then free to disagree with it.** Measured: the injection that made every
    reachability rule reachable left `_git`, `_on_branch` and `default_branch`
    as the module's last three unreached decisions. They get a REAL git tree
    (`update-ref` into `refs/remotes/origin/*` — no network, no bare remote),
    and it reports **UNSEEN** rather than passing where git is absent. Budget
    the cost: the module's run went **3.2s → 33.4s** for those nine arms.
- ⚠ **Expensive is not wedged, and the report cannot tell you which.**
  `platform_dead_code_audit` (393s, 132/269) and `len_derived_drift_sweep`
  (1039s, 37/48) both blew a 300s budget and both completed cleanly when given
  one; `len_derived_drift_sweep` has the BEST reach in the extra population
  (2 live of 48) and would have been written off as a hang. An external
  timeout is a scheduling bound, never a verdict.
- ⛔ **A mutant can never RETURN, and without a bucket the hang is the MILD
  failure** (T6). Flipping a conjunct out of a loop condition produces a module
  that computes forever, and the harness had no bound at all: a
  `--include-all` run predicted at 13 minutes was still burning 98% of a core
  at **two hours**, wedged on one mutant of `restatement_theorem_audit`. The
  worse half is what happens when you interrupt it — the watchdog raises
  `KeyboardInterrupt`, `except BaseException` reads that as *the arm noticed*,
  and a non-terminating mutant is credited **KILLED**. `TIMEOUT` is its own
  verdict with CRASHED's standing (*evidence of nothing*), the rows are named
  individually, and the flag is checked BEFORE the kill. The deadline is
  **derived from the module's own baseline** run (10x, floored at 30s) rather
  than typed — one constant cannot mean the same thing to a 0.03s gate and an
  8.3s workspace sweep. Measured: the module that never terminated now finishes
  in **33s with 1 TIMEOUT**. ⚠ The watchdog is a thread + `interrupt_main`,
  because `SIGALRM` is POSIX-only and the workstation is Windows; it reaches a
  pure-Python loop and NOT a blocking C call. A subprocess per mutant would be
  airtight at ~2400 interpreter starts — naming the 10% it misses is the point
  of writing it down.
- ⛔ **The exec namespace is a registered module, and the bare dict was the
  harness's THIRD bucket-boundary defect** — invisible in the default
  population, which is why T1–T4 never saw it. `dataclasses` resolves a class's
  defining namespace through `sys.modules.get(cls.__module__).__dict__`, so a
  module exec'd into a plain dict dies at its `@dataclass` line. Measured:
  **seven** classifiers — `platform_dead_code`, `len_derived_binding`,
  `required_features_build`, `cfg_gated_target`, `cfg_row_implication`,
  `all_ignored_target`, `suite_membership` — read CRASHED on their own
  unmutated source, carrying **796 of `--include-all`'s 2382 mutants**. ⚠ The
  bare-dict direction is a **premise, not an assertion**: CPython ≤3.12 guards
  that lookup and 3.14 does not, so the self-test asserts only that a
  `@dataclass` module EXECs in the registered namespace (sufficient wherever
  the defect is live) and `dataclass_premise()` prints which side this
  interpreter is on, next to the verdict.
- **`NO-ARM` carried the finding that motivated T4, and pooling it either way
  destroys it.** Four CHECKS had no arm of their own —
  `percentile_floor_gate`, `cfg_row_implication_gate`, `trap_sentinel_gate`,
  `markdown_fence_gate` — because they **delegate** to the classifier they
  import. 789 credits that, correctly. But a classifier's self-test **cannot
  reach its consumer's pin arithmetic**, which is Issue 775's exact sentence:
  789's predicate asked whether an arm runs, never whether it can SEE the gate
  it guards. All four carry a `gate_selftest` now and the bucket is **0** —
  and each needed a small EXTRACTION first, which is the finding underneath:
  the verdict arithmetic sat inline in `main()` beside its own error messages,
  so it was unreachable by construction.
- **`UNREACHED` is its own marker** and it exists because the first version of
  this report printed `✓` for it: `orphaned_attr_gate` scored 0 killed / 5
  crashed / 7 exempt, so the arm distinguished *nothing* and the row read as
  the cleanest in the set. Now **0** — one of the six was a genuine gap
  (`required_features_static_gate`'s pin READER, whose line filter and
  REQUIRED_PINS completeness check had no arm) and the other five were the
  CRASHED misfiling above.
- ⛔ **OPERATOR SCOPE is narrow and the whole report must be read through it.**
  Only control flow and off-by-one are mutated (comparison flips, `and`↔`or`,
  dropped `not`, bool constants, `+`↔`-`). **Regex and string literals are NOT
  touched** — and that is where most of this repo's decision logic lives.
  Measured: `docs_gate_checks_sync` has 20 hand-verified arms that red under
  regex perturbation and scores **2 killed of 10** here. A low kill count is
  not evidence an arm is weak.
  - `+`↔`-` was added in T4 for a measured reason: with comparisons alone,
    `markdown_fence_gate` read UNREACHED while its arms asserted real
    behaviour, because its only real decision is `n_lines - first` and no
    comparison touches it. This repo's whole percentile section is about an
    index landing on `n-1`, so off-by-one is the operator class that matters
    most here.
  - `if __name__ == "__main__"` is **skipped**, not exempted: it is the entry
    point rather than a decidable rule, no arm can kill it, and it is in all
    61 tracked scripts. Skipped so it cannot inflate the total `MIN_MUTANTS`
    floors.
- ⚠ **Reach is per MODULE, so a rule asserted by a DIFFERENT module's arm reads
  SURVIVED** — check this class FIRST on any survivor. Measured:
  `skill_repo_set_gate.derive_repos` survives here and is covered by
  `population_sync_gate`'s synthetic-workspace canary. This repo shares rules
  across modules deliberately (Issue 755), so the audit is blind in exactly the
  direction the architecture points.
- ⚠ **EQUIVALENT mutants are the other false-positive class** (a `>=` whose
  operands can never be equal; a `< 0` sentinel test flipped to `<= 0`). So
  SURVIVED is arm reach per function, never a defect count.
- `prove_fires` bodies are excluded from mutation but the arm is **not run**:
  it is a known-answer validation against a FROZEN commit, so no mutation of
  the working source can change its verdict, and running it per mutant was 436
  `git archive` calls — measured at **80.2s vs 4.4s**, with the children's
  output escaping `redirect_stdout` (a subprocess writes to fd 1) and littering
  the report. Output is suppressed at the **file-descriptor** level for the
  same reason.
- Validated by sampling: of the first five survivors read one by one, **three
  were real and closable, one was cross-module-covered, one was EQUIVALENT** —
  and closing the three took `skill_repo_set_gate` from 18 to 22 killed with
  its survivors from 7 to 3, leaving exactly the two EQUIVALENT rows and the
  one cross-module row. T3 then read the rest module by module: **123 → 47**.
  **Do not quote the SURVIVED total as a defect count** — the 47 that remain
  are dominated by three classes, each documented at the line it lives on:
  redundant guards that are provably EQUIVALENT (a `find() < 0` after an
  earlier match; a set membership test `or`-ed with another; the closure
  bound whose slack-less form is exactly sufficient), the **git/subprocess I/O
  shell** an arm cannot enter without spawning the auditor it reads, and
  message-formatting arithmetic.
- ⛔ **That three-class characterisation of the 47 was TOO GENEROUS, and T2
  found it by trying to write the reasons down.** A pin file demands one
  sentence per row, and the sentence could not be written for about a third of
  them: they were plain functions over plain data — `_parse_feature_spec`,
  `parse_status_phrase`, the `local_default_closure` walk, the three manifest
  READERS that decide which packages enter the model at all — with no fixture
  repo and no subprocess between an arm and the decision. They were **real
  gaps wearing an EQUIVALENT label**, and pinning them would have been a
  backlog wearing a pin, which is exactly what Issue 785 forbids. Closing them
  first took the set **47 → 27**: `bench_doc_audit` 23 → 9, `markdown_fence`
  3 → 1, `cfg_gated_floor_gate` 4 → 2, `skill_repo_set` 3 → 2,
  `trap_sentinel` 1 → 0. **Writing the reason is the adjudication** — a
  classification made while reading a list is not the same act.

### The verdict half — `scripts/arm_reach_gate.py` (T2)

The quantity to gate is **not the count**, and this repo already had the rule
written for `cfg_gated_floor_gate`: *a set is gateable where its cardinality is
not.* The survivors are pinned by MEMBERSHIP with a REASON per row in
`scripts/arm_reach_survivors_expected.txt`, and the wall is **0 UNPINNED**. A
ratchet would tolerate a new unreached decision line as long as somebody armed
an old one; membership does not. It reds in BOTH directions, and `UNREACHED`
and `NO-ARM` are walled at 0 **separately** — pooling either into the survivor
count is what the bucket note above forbids.

- **The key is LINE-FREE**, and that is load-bearing rather than tidy: a line
  NUMBER drifts on every edit above it, so a line-numbered pin file reds on
  commits that changed nothing about it, and a pin file that reds on noise is
  one people delete. It is
  `<module>::<function>::<operator-token>::<8-hex digest of the line TEXT>#<n>`,
  with the ordinal scoped to the WHOLE address — the
  `len_derived_eyes_expected.txt` precedent. Scoping it that way is what makes
  it stable: a new `>=` site in a function hashes differently and renumbers
  nothing. Only genuinely duplicate line text shares a sequence (there are two
  such rows, both real: three `+` on one `print`, two `True` kwargs on one
  `subprocess` line).
- The digest is unreadable by design, so each row carries its source line in a
  `#= ` comment **which the gate VERIFIES against the observed text**. A
  comment is the part of a pin file a human adjudicates from, and a comment
  nothing can red is a comment that drifts into a lie. ⚑ It earned its keep on
  the first real run: the membership wall was clean and the only failure was a
  hand-copied comment missing its trailing `:`.
- **It gates ITSELF** (population = the CHECKS set **+ this file**). Not
  symmetry: an exempt gate certifies nothing, and on its first self-hosted run
  it found a degenerate arm in its own key builder — the ordinal counter's
  `+ 1` flipped to `- 1` still yields two distinct keys, so a count-only
  assertion read green. Assert the ordinal VALUES.
- **The two PERMISSIVE sets are pinned by membership**, because no floor
  guards them: `EXEMPT_FUNCTIONS` (a name added there deletes every survivor
  in that function from the finding set) and `ARM_NAMES` (arm bodies are never
  mutated, so a name added there turns decision code into unwatched code —
  adding `main` would look like a tidy-up). `check_validation_gate` found the
  same shape in its own vocabulary.
- Three floors, failing differently: `MIN_MODULES` the walk, `MIN_MUTANTS` the
  operators, and `MIN_KILLED` the **runner** — a runner that reports KILLED for
  everything empties the survivor set, reds every pin as "no longer survives",
  and the obvious remedy is to delete them all.
- **Not a `docs_gate.sh` CHECK, and not a sweep.** Measured **157.6s** over
  22 modules / 552 mutants, against the docs gate's ~13s budget. It is a
  workstation verdict, the same standing as the eleven drift sweeps. There is also deliberately **no `--prove-fires`** — the
  known-answer validation would be a full mutation run over a `git archive`d
  tree to re-derive a fact the issue already records.
- ⛔ **That 157.6s is STALE, and the corrected figure is 868 s — but read the
  RETRACTION with it, because three earlier write-ups of this number in this
  file were wrong** (measured 2026-09-20, at **33** CHECKS). Write the CHECKS
  count beside the number, exactly as the docs-gate CPU paragraph requires:
  the population is the CHECKS set plus this file, and that set has grown from
  ~21 to 33 since 157.6s was taken.
  - **The measurement that stands is a RANGE, not a budget: ONE run in SIX
    completed, in 868 s (14.5 min); the other five were capped or killed at
    1800 s, 3000 s, 77 min, 77 min and 121 min.** All are real and the spread
    is unexplained — so 868 s is a lower bound on a good run, never a figure to
    plan against, and a green verdict from this gate is something you may not
    get at all. What IS stable
    across every re-measurement is the decomposition: the cost is the SUM of
    per-module costs, which
    span three orders of magnitude. Measured per module: `bench_doc_audit`
    **34.34 s** (84 mutants), `cargo_comment_audit` **19.53 s**,
    `console_encoding_gate` **16.75 s**, `cross_repo_path_dep_gate` **7.51 s**
    — against `docs_gate_paths_sync` **0.03 s**, `agents_repo_set_gate`
    **0.05 s**, `cfg_row_implication_gate` **0.06 s**. The expensive ones are
    the modules whose ARMS walk the tree, re-run once per mutant. So the cost
    scales with the CHECKS set AND with how much I/O the new check's arm does,
    which is the thing to weigh when adding one.
  - ⛔ **RETRACTED, and each was a causal claim built on an intermittent
    reading:** that the gate has *"no measured completion time"* (it is 868 s);
    that a single module *"alone burns 22m47s"* (`agents_repo_set_gate` is
    **0.05 s**); and that the cost is `selftest()` *"run unconditionally before
    main() filters"* (`selftest()` is **1.03 s**, measured three times). The
    self-test does run before the filter — that part is true and is still worth
    knowing — but it costs a second, not minutes.
  - ⚑ **The lesson is the one this file already states about perf numbers and
    is worth restating where the mistake happened.** Three runs wedged
    reproducibly at the same point and two more capped at 300 s, and from five
    consistent observations a mechanism was inferred, published, and refined
    twice — each refinement narrower and more confident than the last, all of
    it resting on readings that later would not reproduce. **Consistency across
    runs is not reproducibility when every run shares one box and one hour.**
    The thing that broke the chain was re-measuring the *same* quantity later,
    not reasoning harder about it; nothing in the earlier evidence was going to
    reveal the error from the inside.
  - ⚠ The variance itself is **UNEXPLAINED** and is left that way rather than
    given a third mechanism. Whole-run observations to date: 868 s (completed),
    >1800 s, >77 min, >77 min, >121 min (all capped or killed). The stable
    quantities — `selftest()` at 1.03 s, per-module `audit_module` at
    0.03-34.34 s — do not add up to the long runs, and nothing measured so far
    accounts for the difference. Concurrent agent sessions on the box are the
    obvious suspect and are NOT evidence.
  - ✅ Still true and still worth doing: run it in a **DETACHED worktree**
    (`git worktree add --detach /tmp/x <sha>`) so live edits cannot race a
    14-minute read of `scripts/`, and kill it by **PID**, never by pattern.
    The gate's population is `scripts/*.py` and it re-reads each module per
    mutant, so an edit mid-run still voids the verdict.
- ⚠ **Unlike Issue 789's, this class DOES generalise and a sweep half is
  owed.** 789 measured its population at ONE (katgpt-rs is the only repo with
  a CHECKS array) and declined a sweep on that measurement. Re-measured here
  for arms rather than CHECKS: **18 arm-bearing `scripts/*.py` across three
  sibling repos** (riir-train 14, riir-ai 3, riir-clippy 1). Do not carry 789's
  "no sweep" answer across — it was an answer to a different question. ⚠ But
  the population's SHAPE settles the ceiling, and it is **4 standing
  instruments + 14 plan-SCOPED** riir-train `planNNN_*.py` gates — so the
  ceiling is a **RATCHET on the derivative**, not a wall
  (`instrument_reachability_drift_sweep`'s answer, for its reason). ⛔ And
  T5 is BLOCKED on something no other sweep in the family faces: all eleven
  are STATIC readers, and this one would **EXECUTE** ~700 mutated copies of
  another repo's gate scripts, whose arms may read metrics blobs or write
  artifacts, in a repo another agent writes concurrently. It needs a sandbox
  story first. Do not land it by symmetry.
- **What T3 found by fixing, and it is the pattern worth carrying forward:**
  in every module the CLASSIFIER was well armed and the **VERDICT** was not.
  `bench_doc_audit` had fixtures from real workspace shapes for its
  reachability model and its tokenizer, and nothing at all for the function
  that joins them; `cargo_comment_audit` had a 20-arm precedence ladder and
  nothing for the scope choice that consumes it; `issue_citation_gate` had 39
  arms and none on the deferral line it prints on every partial-clone run.
  Both halves take a repo path, so all three were armable the whole time.

## A gate whose own failure path is asserted by nothing — `scripts/check_validation_gate.py`

Issue 775 landed six canary arms over a gate's **own pin arithmetic, which the
classifier's self-test cannot reach**. The sentence above is in this file, it is
correct, and it names a rule. The rule landed in **one** gate and was never
generalised — the sixth recorded instance of that shape (Issues 777, 778, 793,
782, 783). Measured 2026-09-14: **six of twenty** CHECKS invoked no arm at all,
own or delegated, carrying **2,050 lines** of per-push logic whose failure path
no test had ever executed.

```bash
scripts/check_validation_gate.py                    # the verdict, per push
scripts/check_validation_gate.py --canary           # 11 arms over its own arithmetic
scripts/check_validation_gate.py --prove-fires 6804d983
```

- The predicate is **invokes an arm UNCONDITIONALLY**, not "has an arm".
  `docs_gate.sh` runs each check as `"$PY" "$script"` — **no arguments** — so an
  arm behind `'--canary' in sys.argv` never fires on a push. Not hypothetical:
  `population_sync_gate.py`'s eight adversary arms, landed by Issue 788 the
  **day before**, were flag-gated and ran on no push at all. They cost 0.17s,
  so there was never a cost argument for the flag either.
- **Delegation is credited, and must be.** Four checks reach their arm through
  the classifier they import (`percentile_floor_gate` →
  `percentile_index_audit.selftest`, plus `cfg_row_implication_gate`,
  `trap_sentinel_gate`, `platform_dead_code_floor_gate`). That is Issue 755's
  DRY answer; refusing it pushes every gate toward a second copy of a rule it
  does not own.
- ⛔ **The first census of this was wrong in the OVER-reporting direction.** It
  grepped the CLI flag strings `--canary` / `--prove-fires` / `--self-test`,
  credited none of those four, and claimed nine bare checks where there were
  six. A census over one representation is blind to whatever that
  representation omits — Issue 787's lesson, reproduced within ten minutes of
  going looking for a new instance of it.
- **`ARM_NAMES` is the permissive direction** and the floor alone does not
  guard it: an empty set reds every check and is impossible to miss, while a
  set that quietly widens (add `main`) greens every check silently. Two floors
  (`MIN_CHECKS` the array parse, `MIN_ARMED` the AST resolution — a walk that
  finds every check and credits none looks exactly like nobody having written
  any arms), plus a canary arm asserting `main` is not in the vocabulary.
- Exemptions are pinned by **membership with a reason per row**
  (`scripts/check_validation_expected.txt`); a reasonless row is refused and a
  row whose check has since grown an arm reds. The file is **deliberately
  empty** — a row reading "not written yet" is a backlog wearing a pin, which
  Issue 785's rule forbids.
- ⚠ **What it does NOT assert:** that an arm which exists and runs is any
  *good*. An arm whose perturbation reds nothing certifies nothing, and Issue
  789 found **seven** such arms while writing the ones this gate counts — one
  whose anchor string was wrong, one whose fixture had no terminated fence for
  the fail-safe to discard, one aimed at the wrong side of a lookbehind, one
  whose input order already matched sorted order. Arm quality is not statically
  decidable and is not claimed. Read the verdict as the weaker thing it is.
- `--prove-fires 6804d983` (the commit that FILED 789) is two-sided against an
  independently known answer: seven checks unarmed there, six bare and one
  flag-gated, named individually. ~0.3s, opt-in on the
  `platform_dead_code_floor_gate` precedent — only `scripts/` is extracted.
- ⛔ **There is NO sweep half, and that is a measurement rather than an
  omission** (Issue 789 T4). Every other verdict class here got one because the
  question generalised; this one does not. Measured over the 16 repos on this
  box: **katgpt-rs is the only repo with a `scripts/docs_gate.sh` CHECKS array
  at all** (riir-train has 58 `scripts/*.py` and riir-ai 7, but no such array).
  A sweep would derive a population of ONE and print a confident green over it
  — the exact reason `ci_gate_coverage.py` is kept out of the CHECKS set. The
  cross-repo question that *does* generalise is "is this instrument findable?",
  and `instrument_reachability_drift_sweep.py` already ratchets it. Do not add
  a sweep here by symmetry with the family; re-run the measurement first.

## A census reads the DOCUMENT, so an undocumented instrument is invisible — `scripts/instrument_reachability_gate.py`

Issue 785's close-out claimed every cross-repo class in `scripts/` had both
halves. `1a5b6571` bounded that to "every cross-repo class **whose verdict is
walled at a small number**". The bounded claim was **still false** by exactly
one instrument — `len_derived_binding_audit.py`, cross-repo, findings walled at
0, no verdict half, closed hours later as Issue 786.

The miss is not the point; the **mechanism** is. Both censuses enumerated the
audits **AGENTS.md documents** against their sweep halves, and that file did
not name the audit at all. *A census that reads the document cannot see an
instrument the document omits*, and it reports a confident, complete-sounding
answer over the subset it can see — every blindness floor in this repo, one
level up, with the DOCUMENTATION as the population nothing floored.

```bash
scripts/instrument_reachability_gate.py                  # the verdict, this repo
scripts/instrument_reachability_gate.py --canary         # the 9 adversary arms
scripts/instrument_reachability_gate.py --prove-fires 18dbe980
scripts/instrument_reachability_drift_sweep.py           # every repo, ratcheted
```

- The predicate is **REACHABLE**, not "named in AGENTS.md". Roots are
  `AGENTS.md`, `scripts/docs_gate.sh` and `.github/workflows/*.yml`; the
  closure then follows script → script references, so a helper invoked by a
  documented instrument counts. The cases demand it —
  `all_ignored_target_audit.py`, `cfg_row_implication_audit.py` and
  `ci_test_execution_report.py` are in no document either, yet each runs
  per-push via an instrument that IS documented.
- ⛔ **`HISTORY.md` is deliberately NOT a root.** It is the archive, it is not
  loaded into a session, and an instrument findable only from it is the
  instrument that stops being run — counting it would have made the gate
  vacuous on the one case that motivated it.
- Pinned by **MEMBERSHIP** with a **REASON per row**
  (`scripts/instrument_unreferenced_expected.txt`); a reasonless row is
  refused. Reds in both directions. **The default for a real instrument is to
  make it findable, not to add a row** — two of the nine measured were wired
  into AGENTS.md instead (`list_unresolved_percentile_sites.py`,
  `citation_weight.py`).
- Two floors. `min_scripts` is the walk. `min_roots` is the **permissive**
  direction and the one easy to leave out: an empty root set makes everything
  unreachable and reds loudly, but a root set that quietly *widens* makes
  everything reachable and prints a green.
- `--prove-fires 18dbe980` is a known-answer validation: at the parent, the
  Issue 786 audit was named only in `HISTORY.md`, so it was the **tenth**
  unreachable script and this gate reds there.
- Verdict half across the workspace:
  `scripts/instrument_reachability_drift_sweep.py` — a **ratchet**, and the
  reason is measured (first run, 2026-09-14: 95 of 152 unreachable, riir-train
  61 of 61 where the predicate over-captures). ⛔ Take the LIVE figures from
  the sweep's own summary line, which derives them: the typed pair in that
  line went stale the day a sibling's tooling was documented and printed a
  count contradicting the measured total one line above it. See the sweep
  family list above.

## A kernel can derive its SHAPE from a buffer's declared size — `scripts/len_derived_binding_audit.py`

`let n_positions = kv.len() / 2 / kv_stride;` inside a CubeCL kernel computes
that dimension from the bound buffer's **declared size**, not from the length
metadata handed to `BufferArg::from_raw_parts`. Bind a buffer whose declared
size exceeds the live range and the kernel silently derives the WRONG shape —
reads never-written memory, writes a measured identically-zero result. No
panic, no NaN, no wrong-looking output (riir-ai `3e00c93e0`, riir-train
Issue 511).

The defect is a **JOIN** of two facts in two files, and a report over either
half alone is noise: HALF A is the in-kernel `.len()` derivation, HALF B is a
bind site whose declared size can exceed the live range (a persistent
struct-field handle, a capacity-sized `client.empty()`, a reused scratch
slice). HALF C (Issue 766) resolves a wrapper parameter's provenance through
**workspace** callers — which is why this instrument's verdicts are cross-repo
by construction, the only ones in the family that are.

```bash
scripts/len_derived_binding_audit.py        # the report, all contract repos (derived)
scripts/len_derived_drift_sweep.py          # the verdict, every repo, pinned
scripts/len_derived_drift_sweep.py --no-stability   # skip the leave-one-out arm
scripts/len_derived_drift_sweep.py --canary         # the 12 adversary arms
```

- The report is a **report, not a gate** (exit 0), except a WALK REGRESSION —
  it carries two loose global floors (`FLOOR_RS_FILES`, `FLOOR_KERNELS`) and
  refuses a confident zero below them. Population derived (BOUNDARY.md +
  `.git`), **tracked** `*.rs` only (Issue 777 — it was one of the three
  instruments walking a gitignored nested repository).
- **UNRESOLVED is not clean** and is never folded into a neighbour: it is 118
  of 164 bind sites, and a ratchet on a bucket meaning *unanswered* is a
  backlog (Issue 785's rule). It is reported, unpinned, with the reason printed
  where it is READ.
- **PERSISTENT-UPSTREAM is the EYES LIST, not the finding list** — some caller
  binds a struct FIELD, so the declared size is whatever that field was created
  as. Pinned by MEMBERSHIP in `scripts/len_derived_eyes_expected.txt`, keyed
  line-free on `(repo, file, kernel, handle)` **plus a count within that
  address**, because the key is not unique in general.
- The verdict half walls the joined buckets (CAPACITY, CAPACITY-UPSTREAM,
  PERSISTENT) at 0 and floors `min_kernels` / `min_binds` per repo. ⚠ Those
  floors are **vacuous in 14 of 16 repos** (riir-ai 43/143, riir-train 9/21,
  everyone else 0/0) — Issue 783's population shape, not Issue 784's.
- ⛔ **`DEFERRED` is not enough here, and that is the axis no other sweep has.**
  A partial clone can corrupt the verdict of a row in a repo that IS present,
  which is a row measured WRONG rather than a row not measured. Measured both
  directions (2026-09-14): **7 of 251** cited caller references are cross-repo,
  and **leave-one-out over all 16 repos produces 0 verdict flips** — so
  per-repo pins are sound TODAY, and the sweep re-measures a TARGETED
  leave-one-out (the supplier set derived from the run) every time rather than
  carrying that measurement forward as a claim.
- It carries **no `min_rs_files` column** on purpose: `orphaned_attr`,
  `platform_dead_code` and `percentile` already floor that identical
  `tracked_files(repo, "*.rs")` call over the identical population, and they
  already disagree with each other about the number. The delegation is
  **asserted** — the sweep reds if any repo it pins loses its non-zero row
  there.

## An item can be dead on a platform NO lane compiles — `scripts/platform_dead_code_audit.py`

`const NEON_U8: usize = 16;` declared ungated, used only inside an
`#[cfg(target_arch = "aarch64")]` fn: **dead code everywhere but aarch64**,
and silent on aarch64. `full_gate` is macOS/aarch64 (the const is alive
there), `wasm32_gate` compiles wasm32 — so the x86_64-native lane that emits
the warning is a **workstation** lane and no automatic gate in this workspace
ever sees it. Five specimens in two days across two repos (riir-clippy intake
P22), the first being `ea4c2873` here.

```bash
scripts/platform_dead_code_audit.py             # all contract repos (derived)
scripts/platform_dead_code_audit.py ../riir-ai  # or one, by path
scripts/platform_dead_code_audit.py --self-test # the 24 classifier arms
scripts/platform_dead_code_audit.py --prove-fires ea4c2873
```

- A **report, not a gate** (exit 0) — except a classifier MISS, which exits
  **2**: an instrument that cannot classify must not be read as `0 findings`.
  The self-test runs on every invocation. Population derived (BOUNDARY.md +
  `.git`), **tracked** `*.rs` only, `vendor/` excluded with its count on the
  per-repo line (Issue 738 T3's rule — riir-ai's `wgpu-hal` fork supplied 5
  rows nobody owns).
- **MOD-REF is a separate bucket and is never folded into the count.** A
  `mod name;` referenced only from gated code satisfies the rule and is *not*
  a rustc finding: measured on `katgpt-types::simd::horizontal`, a wasm32
  `cargo check` is silent because every item inside that module is itself
  x86_64-gated, so the module is **empty** rather than dead — appending one
  ungated `fn` reproduces the warning, on the **fn**. rustc reports dead code
  at the ITEM, which this audit reaches independently.
- ⛔ Its header claimed "no known direction in which this INVENTS a finding"
  and that was **false on the first sweep it ever ran**: masking string
  literals dropped Rust 2021 **inline format args**, so riir-ai's
  `SWEEP_COUNTS` — `println!("{SWEEP_COUNTS:?}")` ungated in `main`, gated
  everywhere else — read as a finding. A conservative-by-construction
  argument is a claim about code somebody else wrote; this one survived until
  a real corpus contradicted it.
- ⛔ **An arm is only a canary if its own perturbation REDS it.** Of the three
  arms added with the buckets, the `vendor/` one red **nothing** under
  perturbation — the synthetic trees have no `.git`, so they took the
  filesystem-walk branch where a redundant `"vendor"` in `SKIP_DIRS` was doing
  the filtering. One exclusion, two code paths, and the arm certified the path
  it was not aimed at.
- Standing (2026-09-14, 16 of 20 repos on this box): **0 findings · 1
  MOD-REF** over 8694 files / 3433 units / 119452 candidate decls. The first
  sweep's two riir-ai rows were compile-verified and repaired —
  `note_ane_dispatch` (x86_64, `--features ane_prefill`) and `gen_u64_bytes`
  (wasm32, `--features chacha20_rng`).
- Verdict halves (Issue 775, 2026-09-14):
  `scripts/platform_dead_code_floor_gate.py` per-push in the docs gate
  (katgpt-rs scope, pins in `scripts/platform_dead_code_floors.txt` — two
  blindness floors, the MOD-REF row by **membership**, plus six canary arms
  over the gate's own pin arithmetic, which the classifier's self-test cannot
  reach) and `scripts/platform_dead_code_drift_sweep.py` on the workstation
  (every contract repo, pins in `scripts/platform_dead_code_drift_floors.txt`;
  population taken from `repo_set.txt` as well as the walk, so a partial box
  DEFERS loudly instead of greening over 16 of 20). `--prove-fires ea4c2873`
  runs by DEFAULT in the sweep and is opt-in on the gate: ~5.6s of `git
  archive` to re-prove a fact about a frozen commit is worth a workstation run
  and not a per-push one (the gate is ~6.2s against a ~13s whole-docs-gate
  budget).

## `text=True` decodes with the SYSTEM locale — `scripts/subprocess_encoding_gate.py`

`subprocess.run(..., text=True)` decodes the child's pipe with
`locale.getencoding()`. macOS, `ubuntu-latest` and the M3 are all UTF-8, so
**nothing that could notice this ever runs it** — and every instrument in
`scripts/` prints `✓`, `✗`, `⛔` and em-dashes. Measured on the Windows
workstation (cp874), against this repo's own `git log -3 --format=%s`:

| form | the em dash `E2 80 94` comes back as |
|---|---|
| `text=True` | `0xe42 0x20ac 0x201d` — three cp874 chars, silently |
| `encoding="utf-8"` | `0x2014` |

Two failure modes, and the **crash is the better one**. Silent mojibake: rc 0,
a plausible string, and a caller matching `re.search(r"FAILED — (\d+)", out)`
matches nothing and reads a confident **zero findings**. Or the decode raises
inside `subprocess`'s reader THREAD, where the exception dies — `run()` returns
normally with the **returncode PRESERVED and `stdout = None`**, which is what
`citation_drift_sweep.gate_says()` got.

`PYTHONIOENCODING=utf-8` does **not** fix the first mode and makes the second
MORE likely: it pins the CHILD's encoder, so the child emits correct UTF-8 that
the parent then decodes as cp874. Both halves are needed, and they are pinned
as separate classes — **DECODE** (`text=True` with no `encoding=`) and
**CHILD-ENCODER** (a `sys.executable` spawn with no `PYTHONIOENCODING` in its
`env=`) — because a shared pin would hide which half regressed.

It is a per-push **gate** and not a sweep-and-done for one reason:
`staged_set_audit.py` has carried the correct form *and a comment naming this
exact defect, dated 2026-09-04*, since the day it was written, and 27 more call
sites were added without it. The ceiling is 0 on both classes over a floored
population (tracked `*.py` AND `subprocess` call sites). Its first real run
found a 28th site nobody had grepped for —
`.agents/skills/doc-sync/tools/linkcheck_sweep.py`, outside `scripts/`
entirely.

⛔ **And it shipped with one half — the gate, no sweep (Issue 783).** Eleven
other verdict classes here carry both, and the asymmetry was not a judgement
call that was made; it was a step that was skipped. `scan()` already took a
repo path, so the question was answerable the whole time, and the answer was
**29 DECODE + 2 CHILD-ENCODER over 5 of 16 repos** — riir-train 12+1,
riir-clippy 9+1, riir-ai 6, riir-dapps 1, mmorpg-editor 1. Two were not
latent: `riir-clippy/scripts/gen_dashboard.py:552` reads `git log --pretty=%s`
across the siblings, and **every commit subject in this workspace uses an
em-dash**. Read that as the standing failure mode, now recorded five times
(Issues 777, 778, 793, 782, 783): a rule landed in one instrument and never
generalised. Before fixing such a class, grep the whole family and land the
repair as one shared mechanism.

**It scans the AST, and that was not the first design.** A text scanner has to
be paren-matched rather than line-scoped (`encoding=` sits on a later line than
`text=True` in every wrapped call here), and the paren-matched version then
reported **four offenders in the gate's own file** — every one a fixture string
inside its `selftest()`. The repairs on offer were to exempt the gate from
itself or to obfuscate its test data, and an exempt gate certifies nothing.
`ast` sees a string literal as a literal; a file it cannot parse is
**UNPARSED** and reds, never folded into the pass column.

## A sweep reads the WORKTREE, so a finding may exist in NO commit — `scripts/worktree_state.py`

Issue 797. Every instrument in the sweep family above walks the **working
tree**. This workspace runs five-plus concurrent agent sessions against
**shared worktrees** — `staged_set_audit.py` (below) exists for exactly that
hazard one axis over — so a row a sweep prints may sit on a line no commit
contains, and a repo a sweep calls clean may be clean only because somebody's
uncommitted edit removed the offending line.

Measured 2026-09-15, `citation_drift_sweep.audit()` run twice per dirty repo
(the worktree, then every dirty in-scope document replaced by its HEAD blob):
the workspace's **entire** standing CROSS finding — 1 of 1 — was an artifact.
HEAD carries `Filed … from the riir-train Research 453 session`; an uncommitted
edit by another session strips the qualifier, and the sweep reports an
unfollowable citation. It had already cost a session, carried across a context
boundary as backlog reading *"blocked, that session has HISTORY.md
uncommitted"*. The correct verdict was not *blocked*; it was **there is nothing
to fix**, and no amount of reading the sweep's own output could say which.

**The POPULATION moves too**, which a row-level read alone misses: the same two
runs put `n_cites` at **601 (worktree) vs 607 (HEAD)** and `ambiguous` at 162 vs
163, because that session's uncommitted deletion of a 30-line block took six
citations out of the denominator. A floor or ratchet re-pinned from such a run
bakes another session's in-flight edit into a tracked expectations file, where
it reds on every other box.

Three verdicts, never interchangeable:

- **COMMITTED** — the finding's file matches HEAD. An ordinary finding.
- **UNCOMMITTED** — the worktree carries a row HEAD does not. **Displayed**
  (it is what the file says today, and hiding it would be its own lie) but
  **never adjudicated against a pin**. The split of responsibility, once: the
  DISPLAY reads the worktree, the PINS read HEAD.
- **MASKED** — HEAD carries a row the worktree does not. A *false green*: the
  defect is committed, in the repo, and the sweep says the repo is clean. The
  silent direction and therefore the worse one. **0 today, which is a
  measurement and not an absence of the class** — nothing had ever looked.

⛔ **A fourth, and it is a different AXIS: the worktree can match its own
HEAD and still be 109 commits behind ORIGIN** (Issue 798). The three verdicts
above all compare the worktree to LOCAL HEAD, so a stale checkout is invisible
to them — and a sweep reads the worktree. Measured: riir-ai's
`toolchain-override-deliberate` marker was committed upstream at `194cdc9b5`
while this box's riir-ai sat 109 commits back, 14 of them touching the sweep's
population, so the toolchain sweep reported `drift 1 · ✗ FAILED` on a defect
that was **already fixed**. It is the **mirror of MASKED** — MASKED is a
committed defect read clean (false green), this is a committed FIX read dirty
(false RED) — and it cost a session real time, the row having been
investigated as unfixed.

- **STALE** — `behind_origin(root, patterns)`, wired into `sweep_advisory()`
  so all 18 sweeps get it with **zero call-site changes**. Four answers, never
  pooled: `None` (no upstream, or git could not answer — **never guessed at**,
  since assuming `origin/main` invents a verdict for a repo that may not have
  one; five workspace repos have none), `(0, 0)` (up to date, silent), `(n, 0)`
  (upstream moved outside this sweep's population, silent — otherwise it is the
  banner nobody reads), `(n, k>0)` (the advisory).
- ⛔ The **three-dot** `HEAD...ref` diff is load-bearing: a two-dot diff also
  reports every file this checkout's own *unpushed* commits touched, so a repo
  merely AHEAD would read as stale. Its arm asserts exactly that.
- ⚠ The sweep deliberately **stays RED**. Converting a red to a deferral on
  staleness would let a genuinely-unfixed drift hide behind "you are behind
  origin", and `max_drift` is a wall at 0. The reader is told how to check;
  the wall still holds.
- ⛔ **`upstream_axis()` is the upstream half of `sweep_advisory`, extracted
  — because the two entry points were NOT equivalent and nothing said so.**
  `sweep_advisory()` computes the dirty scope AND asks `behind_origin`;
  `worktree_advisory()` only RENDERS what it is handed.
  `sweep_advisory_membership_gate` accepted either, rightly — a narrower
  predicate once condemned `citation_drift_sweep`, the most carefully wired
  member — and the premise underneath that repair was never checked. So the
  one sweep on the low-level path had the WORKTREE axis and **no upstream
  axis at all**: no STALE, no UNVERIFIED, in the only member whose rows carry
  a `file:line` address and name another repo.
  Measured 2026-09-19: it reported a CROSS finding at riir-neuron-db
  `HISTORY.md:49` with no advisory of any kind, while `origin/develop`
  already carried the repair and that checkout was 4 commits behind with one
  commit touching that very file — Issue 798's founding class, *a committed
  FIX read dirty*, costing an investigation before a manual
  `git show origin/develop` settled it.
  The sweep keeps its own scope (`dirty_files` ∩ its own document set, sharper
  than any glob) and calls `upstream_axis` for the missing half; `WANTED` is
  `("sweep_advisory", "upstream_axis")` now, so both members of the pair
  actually deliver the axis and a sweep that only RENDERS is no longer
  credited.
  - ⚠ **Not a fifth `MECHANISMS` row, and the registry's own arm is what said
    so.** Registering one made `sweep_advisory` serve two mechanisms and the
    anti-POOLING arm fired immediately. It was right: this is ONE mechanism
    whose membership test was wrong, not two mechanisms. Reach for the
    registry when the mechanism is new — not when an existing predicate is.
  - ⚠ The arm that pinned the old behaviour was **replaced, not deleted**: a
    bare `worktree_advisory(...)` must now NOT be credited, and the low-level
    path *with* `upstream_axis` must be. An arm pinning a false premise is
    worse than no arm — it makes the defect a requirement.
- **One matcher, shared** — `dirty_in_scope` and `behind_origin` both call
  `_match_count`. A git **pathspec** was the obvious implementation for the
  second and answers differently (a bare `Dockerfile` pathspec matches only at
  the repo root), so the two axes would have disagreed about what a sweep's
  population *is*.

```bash
scripts/worktree_state.py            # the arms (exit 1 on failure)
scripts/fetch_contract_repos.py      # refresh every contract repo's refs, ONCE
scripts/fetch_contract_repos.py --selftest
```

⛔ **A sweep does NOT fetch, and Issue 850 T2 answers that with a
measurement rather than a preference.** Every staleness verdict in the family
rests on a LOCAL remote-tracking ref, so the disclosure above is only as good
as the last fetch — but a fetch across the contract repos measured **250.2s
serial / 50.2s at 8-way** (21.5s warm) against a sweep that costs 0.04–40s and
a 32-check docs gate that costs ~164s wall. A per-sweep fetch is **5–30x the
cost of the thing it precedes**, paid ~19 times over a family run. A
`--fetch` flag is not the repair either — *a flag nobody passes is not a
repair* — and an automatic fetch turns an observer into a writer of refs
other sessions own, which is Issue 797's class with the sweep as the
perpetrator. **Freshness is a property of the BOX at a moment, not of a
sweep**: fetch ONCE per session, and let the sweeps DISCLOSE.
- ⛔ The remedy line names the INSTRUMENT and not `git fetch <repo>`, and
  that is the alias codec showing up where it cannot be papered over. The
  advisory's labels are CONTRACT spellings **by rule** (alias content must
  never reach stdout — run logs get pasted into tracked docs), and on an
  aliased box a contract spelling is a directory that **does not exist**:
  measured, the old text told the reader to fetch `mmorpg-editor` while the
  checkout is `seal-game-editor`. Both rules were right and the remedy was
  still unusable.
- `origin` is NAMED, never a bare `git fetch`: riir-chain carries a second
  remote that is stale by design.
- Population **delegated** to `skill_repo_set_gate.derive_repos` and opened
  through `repo_alias.disk()` — an eleventh private contract-repo walk is
  exactly what `population_sync_gate` exists to catch.
- Exit 1 only on a fetch FAILURE (actionable), never on "nothing moved". The
  failure line says the honest thing: those repos' upstream readings still
  rest on an unrefreshed ref, so a red finding there is **not confirmed**.
- The per-repo `--timeout` is not garnish. `arm_reach_gate` wedged on this box
  for twenty minutes against a `git` child that never returned, and its own
  watchdog provably could not reach it — a thread plus `interrupt_main`
  reaches a pure-Python loop, never a blocking C call. Anything here that
  spawns git in a loop needs the bound at the SPAWN.

⚠ **The arm count is DERIVED, not typed** (`n_assertions()`, Issue 798 T3).
The line used to read `36 assertion(s)`; counted by AST at the parent commit,
before any change, it was **40** — stale on arrival, in the module whose whole
subject is records drifting away from what they describe.

- ⚠ **ADVISORY, never a failure.** A sweep that hard-reds on an ordinary dirty
  worktree is a sweep nobody runs — the cries-wolf outcome this document names
  for `.benchmarks/` in the numbering gate. It rides the sweep's FINAL line in
  BOTH directions (the `DEFERRED` precedent) and is **SILENT** when nothing
  dirty meets that sweep's own population. MASKED is the exception, and it
  needs no separate teeth: the pins already count it, because they read HEAD.
- **Wired into EVERY sweep**, at the existing `population_verdict()` call
  site, each with the globs naming its OWN population — so the `*.lean` sweep
  stays silent while somebody edits Rust. Verified per-population on the landing
  run: the `*.rs` sweeps reported riir-ai (3), the `*.md` ones riir-ai (1),
  `numbering` katgpt-rs (1) + riir-ai (1), `subprocess_encoding` katgpt-rs (16)
  — its own in-flight edits — and `trap_sentinel` / `restatement` printed
  nothing at all. Landing it in one instrument and not the family is the
  failure mode recorded six times here already (Issues 777, 778, 793, 782, 783,
  789).
- ⛔ **And "every" was typed as a NUMBER first, which made it wrong within two
  hours.** The landing commit said *"all sixteen sweeps"*; a concurrent session
  then pushed `pipefail_discard_drift_sweep.py` and
  `toolchain_override_drift_sweep.py`, neither wired, and nothing noticed. So
  this is the **seventh** instance of the never-generalised shape and the first
  one repaired mechanically rather than by hand:
  `scripts/sweep_advisory_membership_gate.py` (docs-gate CHECK) reds on a
  tracked `*_drift_sweep.py` calling neither `sweep_advisory()` nor
  `worktree_advisory()`. Gated by **MEMBERSHIP**, because *a set is gateable
  where its cardinality is not* — a count is green on a swap. Exemptions carry
  a reason each and the file is deliberately EMPTY; a stale pin (its sweep
  wired since, or gone) reds too, so the file cannot only ever loosen.
  ⚠ It asserts the CALL, never that the patterns name the sweep's own
  population — a sweep passing `("*.lean",)` over a Rust walk is silent forever
  and reads as wired. That is a per-sweep read, the same limit
  `check_validation_gate` records about arm quality.
  ⛔ Its FIRST run reported `citation_drift_sweep` — the most thoroughly wired
  member, the only one with the row-level split — as UNWIRED, because the
  predicate named one of the mechanism's **two** entry points. A criterion that
  condemns the most careful caller is the criterion that is wrong.
- The row-level UNCOMMITTED/MASKED split began in `citation_drift_sweep.py`
  alone, because it is the one whose findings carry a `file:line` address and
  the one where the class was measured. `audit()` takes an injected
  `read(path) -> str | None` so the SAME classifier can be pointed at HEAD's
  blobs; `None` means "absent from the source being read" and must stay
  distinguishable from empty text.
- ⛔ **"Alone" was the ninth instance of the never-generalised shape, and it
  was measured breaching a live ratchet** (Issue 822, 2026-09-17). Its first
  clause had stopped being true — six sweeps carry file-addressed rows — and
  **14 of 19 sweeps carry at least one COUNT ceiling**, membership-pinned ones
  included where they pin some buckets and ratchet others. ⚠ Read that 14 as
  what ONE representation could see: it was derived by grepping
  `> row["max_*"]`, and `docs_drift`'s ceiling is a wall at 0 written
  `if b_mis:`, so the exposed set was larger than the census that found it.
  Measured:
  `console_encoding` reported `undefended 56 > pinned 53` where one of the
  three new rows sat on a file `git log` could not see at all, staged by
  another session mid-commit. The pressure to type 56 into the pin is the
  whole hazard, and nothing in the run distinguished the case.
- **`worktree_state.head_delta()` is the shared helper**, and its answer is a
  `HeadDelta` whose `.head` — `committed + masked`, **never `committed`** — is
  what a PIN adjudicates. That arithmetic lives in one place because getting
  it right independently in fourteen sweeps is fourteen chances to understate
  a ceiling by exactly the silent direction.
- ⚠ **The PREMISE is per-file row independence, and it is what makes it
  affordable.** T2 asked whether re-reading is affordable for a 2415-file Rust
  walk; the question was the wrong shape. A clean file's rows are identical at
  HEAD by construction, so the cost is |dirty ∩ population| `git show` calls —
  the quantity `dirty_in_scope()` already prints on the advisory line, and
  **zero** on an ordinary run. A tree-sized re-walk buys nothing.
- ⛔ So a **CROSS-FILE** classifier must NOT use it: `len_derived` resolves
  provenance through other files and other repos, `instrument_reachability`
  computes a closure from roots (a dirty `AGENTS.md` changes other scripts'
  verdicts), `numbering` and `citation` are cross-document by construction.
  The premise is a property of the CALLER's classifier, which no check in the
  helper can decide — so it is STATED, not asserted.
- **`head_overlay()` + `delta_of()` are that second instrument**, and they are
  the two pieces `citation_drift_sweep`'s inline version consists of, lifted so
  the next caller does not copy it. `head_overlay` answers `{path: HEAD bytes}`
  for the dirty files in scope — **`None` is a VALUE there** (tracked but not
  in HEAD: a staged-but-never-committed file, Issue 822's measured case), and
  an **EMPTY dict means skip the second classification entirely**, not
  "overlay nothing". ⚠ The patterns must name every file that can CHANGE a
  verdict, not just the ones findings sit on: for a closure that is the ROOTS
  as well, and `instrument_reachability`'s own advisory had a second
  hand-typed glob list naming `.yml` and not `.yaml`, so a dirty `.yaml`
  workflow — a root — was silently out of scope. One `SCOPE` constant now.
- **The display stays honest in both directions** (T3): the per-repo line
  reads `undefended 56 (53 committed + 3 uncommitted)` and each listed row is
  labelled, because hiding the three is the lie Issue 797 refuses. **MASKED
  needs no separate teeth** (T4) — the pins read `.head`, so a committed row
  the worktree hides breaches the ceiling on its own.
- ✅ **The fan-out is COMPLETE and GATED (2026-09-18)** — every tracked
  `*_drift_sweep.py` adjudicates its findings against HEAD, and
  `head-provenance` is a MECHANISMS row in
  `sweep_advisory_membership_gate.py`, so the next one cannot land unwired.
  **Take the family size and the wiring verdict from that gate's PASS line,
  never from a count here**: this paragraph said "wired into TWO sweeps so
  far" and stayed that way through fourteen more, which is the substitution
  this document names as its own most-repeated error. Full record and the
  rules each instrument earned: HISTORY.md § Issue 822.
- ⛔ **Pick the instrument from the CLASSIFIER's shape, never by preference.**
  `head_delta` needs per-file row independence; `head_overlay` + `delta_of`
  needs ONE interceptable reader; `head_tree` materialises HEAD for a
  classifier with several seams (`git grep` + `git ls-files` + direct reads)
  and runs it UNMODIFIED — measured **22-32s per dirty repo** against
  0.04-0.26s clean, so it yields `None` on a clean run and the caller MUST
  skip. Its `paths=` narrows the archive (riir-train 604 MB → 30.6 MB, worth
  2.2x end to end, and a file the pathspec drops is silently ABSENT — the
  caller's risk); its `extra_dirty=` widens the TRIGGER for a sweep whose walk
  is `os.walk` rather than `git ls-files`, where an untracked file is in the
  population and produces zero `git status` dirt.
- ⛔ **The VERDICT belongs in the row KEY, and a FLOOR is a pin too.** A
  key-matched row is filed COMMITTED carrying the WORKTREE's object, so any
  field the key omits is one where the worktree silently overrides HEAD — and
  every ceiling in the family partitions by verdict. Floors read HEAD for the
  same reason: Issue 797 measured that class on a POPULATION (607 → 601
  citations), not on a finding.
- ⚠ **Budget the canary cost before wiring one.** Each sweep's `--canary` runs
  `main()` once per arm over every contract repo, so four new arms is roughly
  a 45% increase: `console_encoding` went 9 arms/~15s to 14 arms/**~42s**.
  Workstation-only — none of these runs per push — but it is why the arms are
  four and not fourteen. ⛔ With `head_tree` the same shape is an order of
  magnitude worse: `len_derived`'s canary ran past **120s and was killed**,
  because each arm re-entered `main()` and materialised a tree per dirty repo.
  Stub the provenance seam inside a canary whose arms are about PIN
  ARITHMETIC; the seam has its own arms in `selftest`.
- ⛔ **The row key is deliberately LINE-FREE** — `(document, kind, number)`.
  Any edit above a citation shifts its line, so a line-bearing key reports
  every row in an edited document as UNCOMMITTED *and* MASKED at once. The arm
  for it plants two citations behind a padding line and requires the key sets
  to be equal.
- ⛔ **An arm whose perturbation reds nothing certifies nothing, and this
  module caught one of its own.** `dirty_files` normalises `\` to `/`, and
  deleting that line reds NOTHING: `git status --porcelain` emits POSIX
  separators on every platform, measured on the Windows workstation where a
  naive reading expects the opposite. The arm asserts git's OUTPUT SHAPE — the
  premise — rather than pretending to test a defensive line. The normalisation
  that does bite is in `split_rows`, on the CALLER's path.
- ⛔ **A bare string is a footgun, not a convenience.** `("*.rs")` is not a
  tuple; iterating it yields characters, `fnmatch(rel, "*")` matches
  everything, and the advisory silently reports every dirty file in the repo.
  Caught in this module's own wiring commit, where **8 of 15** call sites had
  written it without the comma. The helper coerces and an arm pins both sides.
- Arm reach (`--include-all`, measured 2026-09-15 after Issue 798): **33 of
  39**, five live survivors. Three are `check=True` / `capture_output=True` on
  the fixture BUILDERS — flipping one makes the fixture wrong rather than a
  rule wrong. The other two are `behind_origin`'s two `or`s, which no input
  can distinguish: under real git a failing `rev-parse @{upstream}` exits
  non-zero AND prints nothing, so `or` and `and` agree everywhere. Resolved
  the way `dirty_files` resolved its separator normalisation — `premise_arms`
  asserts git's OUTPUT SHAPE, the thing that could actually change, instead of
  pretending to test a line no input reaches. Every reason is written at the
  line.
- ⛔ **`n_assertions` was four of those survivors until it was EXTRACTED.** It
  read `__file__`, so its three decisions — the `*_arms` name test, the
  `ast.Name` test, the `== "check"` test — were unarmable by construction.
  Injecting `src` made all three testable against known answers, which is the
  same pattern AGENTS.md records for the three weakest modules in the CHECKS
  population: the extraction IS the repair.

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

⛔ **Nothing in git identifies WHICH session did something here, and all THREE
fallbacks are measured broken** (Issue 840). Every commit in this worktree authors
as one address, so authorship is blank; a shared worktree has **one `HEAD`
reflog**, in which every session's checkouts, resets and commits interleave under
no mark at all — *it answers what happened and in what order, never whose*; and
**elimination fails on an incomplete roster**, which is the one that actually
produced a wrong answer. Measured: an unmarked commit was reasoned about as
*"it refers to session `fa` in the third person, so it is not `fa`'s, so it is
`54`'s"* — sound only in a **two**-session worktree, and four were live. Count the
roster before eliminating over it; `ListAgents` shows only sessions still alive,
so it is a floor on that count, never the count.

⛔ **The same defect exists one layer UP, in the messages** — and it is how the
roster gets miscounted in the first place. A peer's `from=` names the PIPE and is
reliable; the name a message **signs itself** with is self-asserted prose and is
not. Measured in the same incident: two distinct pipes were live, and the messages
from one of them signed themselves with the OTHER's session name, so three
sessions' messages were read as two sessions' — after which the elimination above
had no chance. **Attribute a message by its `from=` pipe, never by its
signature**, and treat a signature that disagrees with the pipe as the two-session
question it is, not as a typo.
- ⛔ **Quote the `from=` pipe or `msg_id` when you attribute a message**, exactly
  as you quote a `Session:` line for a commit. Prose attribution between sessions
  has the property this whole section is about: it reads as authoritative and
  authenticates nothing. Measured — the rule above was landed and then **broken
  in the next commit message**, which credited a suggestion to the session whose
  name the message SIGNED rather than the pipe it ARRIVED on. A rule about
  attribution written from a conversation you are misattributing encodes the
  defect instead of the fix.
  - ⚑ It has a LIVE specimen, which is a better argument than the principle it
    was derived from: a message arriving on one pipe signed itself with a
    DIFFERENT session's name, and no session by that name was in the roster.
    The signature lied; the pipe did not, and was distinguishable at the moment
    of reading.
  - ⚠ **A pipe identifies a CONNECTION, not a person**, and whether a pipe id
    can be reused by a successor session is UNMEASURED. So "quote the pipe" is
    stronger than "quote the signed name" and is still evidence rather than
    proof — the same standing as a `Session:` trailer, for the same reason.
- ✅ **The one form that survives a roster you cannot enumerate: say what you
  CHECKED, not who you concluded.** Four independent failures in one evening —
  shared authorship, the shared reflog, an incomplete roster, and mis-signed
  messages — and this is the only discipline that was never wrong, because it
  makes no claim the evidence cannot carry.
- **Put `Session: <name>, <epoch>` in the commit body** — a commit's own TEXT is
  the only self-identifying evidence in the repository, and it is what let one
  session rule itself out of Issue 840 in a single `git log --grep`.
- ⛔ **The epoch is not decoration: session names are REUSED.** Measured the day
  the convention was adopted — `--grep="Session: katgpt-rs-54"` returns **nine**
  commits spanning two unrelated sessions sixteen hours apart, and the
  five-commit answer that established ownership was correct only by accident of
  a ~20-hour search window. A bare name disambiguates sessions running
  CONCURRENTLY and silently conflates them ACROSS TIME, which is the axis anyone
  grepping it later is actually on.
- ⚑ **The marker is what catches an over-claim, including your own.** In the same
  incident a session wrote *"this session's ten commits"* over a rebased range
  that did contain ten — but one of them carried another session's marker, so the
  range was 9 + 1 and the sentence over-claimed by exactly the marked commit. **An
  unmarked commit is claimable by anyone reading a range**, and a range in a
  shared worktree is not a session's work simply because one session rebased it.
- ⚠ Prose is **not** a substitute. The commit at the centre of that dispute was
  resolvable only because its body happened to enumerate the issues it touched,
  which let a reader match it against a marked commit elsewhere. That is luck, and
  it is why the marker is one line and not a paragraph.
- ⛔ **And the marker is EVIDENCE, not proof — by its own rule.** The bullet
  above says *attribute by the pipe, never by the signature*; a `Session:` line
  a commit writes about itself **is** a signature. Nothing authenticates it, so
  it can be wrong, stale or copied exactly as a mis-signed message was. Treat a
  marker as the best available evidence and a *missing* one as no evidence at
  all — and when the question is contested, say what you checked rather than
  who you concluded. (Measured: a session asserted a commit carried no marker
  while having checked a **different** commit — the claim was cheap to verify
  and was not verified.)

### The same hazard one layer down: a FIXED temp path — `scripts/shared_temp_path_gate.py`

The rule above is about the WORKTREE. One layer down, concurrent sessions share
something nothing in git governs: the **system temp directory**. A test writing
to `std::env::temp_dir().join("fixed_name.bin")` is safe against its sibling
tests in one binary — each site has its own filename — and **not** safe against
another PROCESS running the same test. `test_gate`, `full_gate`,
`x86_64_execution_matrix` and any hand-run `cargo test` each get their own
target dir and all share one `/tmp`. `create` truncates: A writes, B truncates,
A reads back zero bytes.

Measured twice (Issue 832, 2026-09-18), because one reproduction is an anecdote:

- `writer_writes_and_counts_samples` run as two concurrent copies of one
  binary — **1 failure in 24 runs**, byte-identical to the x86_64 matrix's
  cell-5 red (`left: 0, right: 5`). Alone it passes every time.
- Five `katgpt-types::tests_types` tests failing **at once** with *"File too
  small for header"* across five DIFFERENT filenames — another test binary
  truncating all five.

⛔ **The lesson is about the CONFIRM step, not the tests.** The x86_64 matrix's
`▸ confirming each failure ALONE` pass filed one of these as **TRANSIENT —
failed in the cell, PASSED alone**. That is a true statement and the wrong
conclusion, and it is the one place this document's own instrument reasons
backwards: *"a load-sensitive bar passes the second time and a real failure does
not"* is correct for a BAR and empty for a CONCURRENCY defect, which passes
alone **by construction** because alone there is no second process. Two reds in
one summary line needed opposite diagnoses — the other was
`test_bench_171_thinking_prune_goat`, where the TRANSIENT reasoning is exactly
right (Issue 831).

The repair is the form this repo ALREADY uses in
`katgpt-transformer/src/contiguous.rs`, `katgpt-core/src/content_store/fetcher.rs`
and the three `katgpt-pruners` sites:

```rust
std::env::temp_dir().join(format!("name_{}", std::process::id()))
```

⛔ **13 sites had it and 27 did not** — 25 repaired, 2 adjudicated as
deliberate — which is why this is a gate and not a
sweep-and-done — the rule was known and un-enforced, this file's own
most-repeated shape. Membership + a reason per row, both directions, floors on
the walk AND the predicate. The two pinned rows are `examples/`, where a demo's
temp path is meant to stay findable by a human and no gate runs two examples
concurrently — an adjudication, not an exclusion rule.

⚠ **Three STATED blind spots**, named in the gate's own docstring and on its
PASS line so a later census reads them instead of re-deriving them: a
`temp_dir()` bound to a variable before the `.join`; other fixed-scratch
spellings (`PathBuf::from("/tmp/…")`, `./target/test_scratch`); and the
cross-repo axis, which is Issue 832 T3 and is **unmeasured** — do not carry
`check_validation_gate`'s "population of one, no sweep" answer across, because
`console_encoding_gate` assumed exactly that and was wrong by seven repos.
**Count first.**

⚠ **`cargo fmt -p <crate>` is not usable for a repair of this shape.** It
reformats ~1300 unrelated lines: of the eight files Issue 832 touched, only six
are rustfmt-clean at HEAD. Format the clean ones per file with `rustfmt` and
hand-wrap the rest.

⛔ **A cross-repo repair is not landed until it is COMMITTED in the sibling
repo, and a record HERE claiming one must CITE THE SIBLING COMMIT** (Issue
798). Measured: two tracked files in `scripts/` recorded sibling repairs as
landed and green — `toolchain_override_drift_floors.txt` ("all five markers
are in", plus a quoted green sweep) and `pipefail_discard_expected.txt` (a
row DROPPED because riir-chain "got the same tail"). **Two of the five
markers existed**, both in this repo, the tail did not exist at all, and both
sweeps were RED for the whole interval. The sibling edits were made in the
worktree, measured green, written up, and never committed anywhere.

That is Issue 797's class reached by a **prose record** rather than by a
floor, and 797's worktree advisory — which names exactly those three repos
on the repair run — postdates the write-up by hours. **Do not add a gate
that parses prose landing claims**: the sweep IS the verification and it was
red from the moment the record was written. What a SHA buys is a claim a
reader can check with one `git -C ../<repo> cat-file -e`, where "the marker
is in" stood false for six hours. Dropping a pin row for an absent fix is
the **unrecoverable** direction — nothing then points at the site.

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
- ⚠ **Allow-by-default lints need `--groups` here** (riir-clippy Issue 135):
  `match_bool`, `map_unwrap_or`, `uninlined_format_args`, `doc_markdown` and
  the rest of the pedantic/nursery fix set are healed ONLY where the target
  crate enables the lint or its group (`[lints.clippy]`, inherited
  `[workspace.lints.clippy]`, crate-root/file `#![warn(clippy::…)]`) —
  otherwise skipped loudly. katgpt-rs enables no pedantic group, so healing
  those classes takes `cargo heal --fix --groups pedantic <paths>`.
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

⛔ **A latency number without its BOX STATE is not a measurement** (2026-09-17).
Step 2 above is where perf numbers are born, and the rule that governs them was
written down in §Docs gate — *"cite CPU **with the load class it was measured
under**"*, measured at 44.97s vs 13.37s for identical work — where nobody
producing a benchmark walks past it. Measured the day that cost something: a
session took `minutes/record` against a plan's `~2.6 s/record`, on a box at
**0.58 GB free**, and was about to re-derive a budget from it. That is a
**paging measurement wearing a throughput number** — plausible shape, real run,
wrong quantity — and the only thing separating it from a finding was somebody
else happening to be watching the process table.
- Record free RAM, commit-vs-limit, and concurrent heavy jobs **next to** any
  latency figure taken on a shared box, or it is not reproducible. This is the
  same rule §Docs gate states for CPU seconds; it is restated here because
  *written-down beats remembered only if the write-up is in the path you
  actually walk*, and that one lives in a section about docs-gate timing.
- ⛔ **On a laptop, POWER SOURCE and POWER MODE belong in that list, and they
  were the axis nothing recorded** (riir-reflex Issue 021, 2026-09-24). An
  entire paired A/B session on the M3 ran unplugged (100% → 45%) while its
  baselines were AC, and no instrument noticed. Apple Silicon sheds sustained
  GPU clock off AC; `pmset powermode` is a **three**-state enum (0 Automatic,
  1 Low Power, 2 High Power — this box's AC profile is 2), so "not 0" is not
  "Low Power". There is no sudo-free throttle readout here (`pmset -g therm`,
  `kern.thermalpressure`, `powermetrics` all measured unusable), so the
  detector is a fixed-kernel canary. Reference gate:
  `riir-reflex/scripts/bench_preflight.sh` — refuses on battery, Low Power,
  < `SETTLE_MIN` since plug-in, or over a load ceiling, and prints a
  `PROVENANCE:` line to quote beside the number.
- ↔ **This bullet is the GENERAL rule; §Docs gate's "load-invariant has a
  measured LIMIT" paragraph is the INSTANCE** — it owns the docs-gate CPU
  figures and their quiet-box scoping, this owns any perf number plus the
  commit-vs-limit and trough rules below. A restatement is two copies that can
  drift, and this repo gates that shape elsewhere (`docs_gate_paths_sync` for
  the duplicated trigger lists; the sweeps that ASSERT a restated floor against
  the gate owning it). **Nothing asserts these two agree**, and that is a
  deliberate stop: the only mechanism would parse prose, which is what this
  file refuses for landing claims — *the sweep IS the verification and a prose
  check is not one*. The mitigation is this pointer, in both directions, so an
  editor of either knows the other exists.
- ⚠ **Rank concurrent jobs by COMMIT, never by working set**, and compare the
  total to the commit **limit read at launch** — not to physical RAM and not to
  a constant. Measured the same evening: `22.17 GB commit at 0.01 GB working
  set` (fully evicted, so invisible to a working-set view), and the commit
  limit itself moved **62.8 → 78.3 GB** mid-session as Windows grew the
  pagefile, which is why the rule is a comparison rather than a number.
  ⛔ **And it moves DOWN too, which is the half that bites** (riir-ai, 2026-09-18,
  4090 box): re-measured forty minutes after a first reading, the limit had gone
  **68.8 → 62.8 GB** with no process allocating the difference — Windows SHRANK
  the pagefile under a box that was in use — taking headroom from 9.8 GB to 3.1
  GB. A run cleared against a limit can lose that clearance without anything
  happening. So re-read the LIMIT at the moment of launch, not just the usage;
  "I checked headroom" is a claim with a timestamp on it. Physical free read
  11.8 GB at the same instant and was again not the binding quantity.
- ⚠ Free RAM read **right after another job exits** is a trough between phases,
  not a window. 18.2 GB was read as clear and an 11 GB job launched into it;
  the next stage of the same sibling pipeline ramped behind it and the box went
  to 0.2 GB.
- G2 is the gate this protects, and `--release` is already mandatory there for
  the adjacent reason: a latency gate in a debug build measures an unoptimised
  binary, and a latency gate on a thrashing box measures the pagefile.

**Lossy-surface promotion rule (riir-ai Issue 750 T3):** a **lossy** surface
(quantization, compression, any bit-changing transform) gates on
**deployed-path behavior — per-family, conditional retention**, not on
bit-identity or aggregate perplexity alone: aggregate perplexity can be flat
while family-conditional behavior flips. External confirmation (arXiv
2609.15504, Orthrus repro — riir-clippy walk #9): under BF16, "lossless"
speculative decoding exact-matches the AR trajectory on only 43–45% of 1,190
prompts (FP32 restores 100%) while downstream lm-eval aggregates show no
systematic degradation — aggregate metrics flat while per-prompt behavior
flips, this rule's exact failure shape. (Full rule + confirmations: HISTORY.md.)

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

⛔ **A concurrent session is the OTHER way this gate gets skipped, and the
cost is a NEGATIVE RESULT rather than a duplicate** (Issue 825, 2026-09-18).
Two sessions implemented one issue from one research row four hours apart. The
second began after the first's primitive was already on `develop`, landed a
second bench, measured an endpoint MAE of 0.1190 against a ≤ 0.01 bar, fired
the issue's negative-result clause, **removed the issue file** and wrote *"the
paper's 20× does not transfer to zone graphs"* into HISTORY.md. It was two
defects in its own walker. A duplicate is embarrassing and recoverable; a
falsification written into the archive is what stops anyone looking again.
- **The disagreement was the oracle, and it is cheaper than either
  instrument's self-consistency.** What settled it was running the losing
  bench's OWN fixtures through the winning bench's shipped readout — same
  graph, same convention, MAE 0.0 vs 0.119. Neither run alone could produce
  that: every gate in the failing bench was internally satisfied at the moment
  it declared the construction falsified. **Before recording a negative,
  re-run `git fetch` and check whether a sibling shipped the same primitive;
  if one did, cross-run the fixtures before writing the word "does not".**
- **Distrust a mechanism inferred from a monotone sequence.** The write-up
  read `0.119 → 0.078 → 0.037` as *"converging ⇒ discretization error in the
  readout, not a broken solve"*, and the named mechanism is precisely the one
  the corrected readout proves is exact. A converging error is consistent with
  many mechanisms.
- **Repair beats delete when the loser is independent.** Both benches keep
  their own solver and their own walker and now agree, which is stronger than
  either alone; the one thing they SHARE is the rule they disagreed about,
  imported rather than copied.

Research workflow (paper classification, 7-repo routing, fusion-first
distillation, novelty + GOAT gates, modelless-unblock protocol §3.5):
`.agents/skills/research/SKILL.md`.

> **Repo count:** the **product/distillation set is 7** — `katgpt-rs` (public) +
> `riir-ai`, `riir-chain`, `riir-neuron-db`, `riir-train`, `riir-game-sdk`,
> `riir-dapps` (private). That is NOT the repo total: the
> workspace is **25 repos**, all of which carry a root `BOUNDARY.md`
> (add `riir-mmorpg-examples`, `riir-clippy`, `riir-viewbridge`,
> `riir-auth`, `katgpt-web`, `riir-dao`, `riir-deployer`,
> `riir-esp32`, `riir-llm`, `mmorpg-editor`, `mmorpg-remake`,
> `mmorpg-remaster`, `riir-kat`, `riir-shader`, `riir-reflex`,
> `riir-infer`, `riir-reflexer`, `reflex-site`).
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

When two documents already share a number, Issue 724 T2's rule is that the one
with the most inbound mentions KEEPS it and the other moves — and the obvious
count is the wrong one. Measured on riir-ai's six duplicates (2026-09-05), the
by-NAME citations are 0-2 per side and TIED in four of the six pairs, while the
`Plan 175` form carries 35-98 each: **the weight is entirely in the citations
that do not say which document they mean.** `scripts/citation_weight.py <repo>
<dir> <number>` attributes those instead of counting them — a context window
scored against each candidate's distinctive filename tokens, awarded only on a
strict margin, with everything else printed as its own UNRESOLVED number and
never folded into a winner.

At ALLOCATION time the same class is caught earlier: `scripts/dual_allocation_gate.py`
(Issue 796) reds when this checkout and its upstream have both added numbered
documents since their merge base — classified structurally by filename stem:
same stem on both lines = TWIN (your own rebased work, exit-neutral, a fetch
resolves it), different stems = INDEPENDENT (two documents about to own one
number, exit 1, both sides' adding commits named). Measured basis (the probe,
`dual_allocation_fp_probe.py`): the deferral's feared shape — one side
allocating ahead — is green BY CONSTRUCTION (7202 one-sided pairs, zero
fires), while 39 real divergence incidents in 90 days split 31 TWIN / 8
INDEPENDENT. Reach limit: it sees divergences THIS box participates in at
fetch/push time; box-vs-box collisions on the wire are the merge-time wall's
(Issue 795's) jurisdiction.

⛔ **A collision where BOTH holders have CLOSED is the MAJORITY case, and the
instrument was blind to it** (Issue 795). A document closed under the
noise-reduction rule is deleted, so the pair leaves nothing on disk and reads as
"not a duplicate" — and every number this repo allocates is expected to end up
removed. `removed_by_number()` recovers them from `git log -M --diff-filter=D`
(`-M` is load-bearing: without rename detection a RENUMBERED document reports as
a deletion at its old number and the tool resurrects a collision somebody
already resolved). Issue 791 recorded **three** collisions; the same scan with
the recovery says **70**, over 1374 numbers, 9 of them at or above 700 and all 9
from one 57-commit divergence.
- ⛔ **`-M` is necessary and not sufficient, in BOTH directions** (Issue 881
  T2.4). It pairs a rename only above 50% content similarity, and a document
  RETITLED as its thesis changes is rewritten far past that (riir-shader
  `_queue` → `_port`, 0.03–0.10), so each read as a second holder.
  `numbering_gate.collapse_renames` treats a removed stem as a rename only if it
  was deleted in the same commit as the new stem's add, or by the same author
  within 1 h, **and** the two stems share a topic (stem-token Jaccard ≥ 0.25),
  **and** the old stem predates the new one. Line similarity cannot make this
  call: the real recycle katgpt-rs `.plans/236` measured 0.065, inside the
  renames' range. The rule costs false reds, never false greens: retitles that
  share no stem token stay collisions. Separately, a pathspec'd `git log`
  SIMPLIFIES history and dropped deletions made on merged feature branches, so
  the recovery passes `--full-history`. Measured: 195 → 180 workspace-wide, 15
  collapsed, 2 surfaced.

⚠ **Take the SCOPE from `scripts/numbering_floors.txt`, never from a walk of
the tree.** A first pass over every numbered directory found 122, and 52 of
those were `.benchmarks/`, where the leading number is the OWNING plan or issue
and a family per owner is the intended convention — an exclusion that file
records, measured, with the note that checking there *"would have been the
cries-wolf instrument AGENTS.md warns gets ignored."* A population derived from
the tree is not the population the gate governs.

The verdict is `numbering_gate.py` in the docs gate, in **two regimes**
(`scripts/number_collisions_expected.txt`): at or above `era_boundary = 700` a
**WALL pinned by MEMBERSHIP** with a reason per row — a count is green on a
swap, and the arms assert that case — and below it a **RATCHET**, counted and
never pinned, because those 61 are the pre-gate archive and Issue 785's rule
forbids ratcheting a bucket that means *unread*. The boundary is measured, not
round: the highest legacy collision is 575 and the lowest divergence one is 741.

⛔ **CLOSING a holder does not retire its row — it is when the row starts
being the only record.** Measured 2026-09-16: an issue was closed and removed
under the noise-reduction rule and its collision pin deleted in the same
commit as "stale", and `develop` went red for every later run. The holder was
gone; the collision was not — `removed_by_number()` recovers BOTH sides from
`git log -M --diff-filter=D`, which is the majority case Issue 795 exists for.
The pin file's own header already said so and the header's count word said
NINE against eight rows, so two independent signals were available and the
removal happened anyway. A row comes out only when the *number* stops being
doubly held, which a deletion never achieves.

⛔ **That recovery ran in ONE repo of sixteen for two days** (Issue 820). The
gate got `historical_collisions()` and `numbering_drift_sweep.py` — the
cross-repo verdict half of that exact gate — kept calling the worktree scanner
and printing `dup=0`. Measured 2026-09-17, the sweep unchanged except for
pointing the gate's own function at the derived population: **0 tracked
duplicates against 183 historical collisions**, 113 of them in repos the sweep
covered and called clean, in **3.7s** for the whole workspace. Three of those
sit above the era boundary, all in riir-ai (`.issues` 702, 753, 959). The
eighth recorded instance of a rule landing in one instrument and never
generalising (Issues 777, 778, 793, 782, 783, 789, 797) — **before fixing such
a class, grep the whole family and land the repair as one shared mechanism.**
- The sweep's `max_hist` is a **RATCHET at measured**, not the gate's
  membership wall: the wall needs a reason per row and 15 repos' worth of
  invented reasons is a backlog wearing a pin (Issue 785). `min_numbers`
  floors the HISTORY walk and is **not** a second `min_files` — they break
  separately, and a `git log` regression leaves `min_files` untouched while
  collapsing the other to the on-disk count.
- katgpt-rs's row does not RESTATE its gate — the sweep imports the same
  function, so a count comparison is true by construction and *a pin that
  restates its own input cannot fail*. It asserts the gate's **verdict** over
  those rows instead, so a stale membership pin in
  `number_collisions_expected.txt` reds the sweep too.
- ⚠ The same run found Issue 815's `DOCS_GATE_KNOWN_EXTRA` marker reaching
  `population_verdict` and **not** the sweeps' own per-repo pin loop: the final
  line read "not measured and not expected to be" while three repos six hundred
  lines earlier were red for having no pin row. The marker excuses the pin-row
  requirement now, by NAME, in both directions.

⛔ **And do not renumber on a margin the instrument did not award.** Six of
the nine were adjudicated and deliberately left alone — leads of +1 to +5 with
21–53% UNRESOLVED, one an outright `TIE_FRACTION` tie and one where the tool
DECLINED (unresolved outnumbered decided). The pin file carries the margin per
row so the decision is re-readable. Renumbering on a 2-site lead with 47%
unresolved is the mistake `TIE_FRACTION`'s own docstring names: *pretending it
can arbitrate is how a coin flip gets recorded as a measurement.*

## Branch

`develop` is the working branch. Don't create feature branches; commit
directly on `develop` per the global rule.

## Models
- riir-train/data/gemma-2-2b-it-f16.gguf
- riir-train/data/MiniCPM5-1B-F16.gguf
