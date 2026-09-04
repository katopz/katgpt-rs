# Compiles is not runs — CI compiled every gate and executed none

> **Status:** the axis is CLOSED as an *audit*; what it discovered is tracked in
> `.issues/723`. Issue 718 (filed 2026-09-03, closed + removed 2026-09-04) is
> the origin; full narrative in git history. This file is the durable record
> because four live files cite it: `AGENTS.md`, `scripts/test_gate.sh`,
> `scripts/ci_test_execution_report.py`, `.github/workflows/test.yml`.

## The axis

`cargo clippy` and `cargo check` **compile** test targets. Neither runs one.
Measured 2026-09-03 on `develop`, `grep -rnE "cargo (test|nextest|bench)"
scripts/ .github/` returned two hits and **both were prose** — a docstring and
a report string. No test-executing command existed anywhere in `scripts/` or
`.github/`.

`scripts/full_gate.sh`'s layers are compile-only by construction: layer 3 is
`cargo clippy --workspace --all-targets --all-features --keep-going`, layer 6
is the same shape with `cargo check --release`. What that left unexecuted, from
`cargo metadata` rather than a guess: **480 test targets, 31 lib targets, 176
bench targets, 268 examples, over 32 packages.**

This is the fourth rung on a ladder this repo had already climbed three of:

1. a workflow file is identical on disk whether or not it can execute — a
   workflow on a non-default branch is inert (Issue 704);
2. **can fire is not does fire** — a `workflow_dispatch`-only gate is a button,
   not a schedule (Issue 706);
3. **a green test count can be a count of nothing** — a `#![cfg]`-gated file
   compiles empty and prints `ok. 0 passed`
   ([`cfg_gated_silent_zero_pass.md`](cfg_gated_silent_zero_pass.md));
4. ← **compiles is not runs.** A CI that compiles and never executes is in
   exactly the state rung 3 warns about, except the count is not zero — there
   is *no count at all*, because nothing produced one.

By this repo's own standard — *an uninvoked assertion is unknown, not passing* —
every Rust assertion here was **unknown**.

## It made Issue 713 T3's arming half-complete

713 T3 armed 39 GOAT gates with `required-features` (`180be9c5`) so that naming
a target without its features errors instead of reporting a green zero. Read
one way that is the other edge of the same fact: the rows make a **named** run
honest, and **nothing named them**. "Armed" and "run" are separate axes, and
only the first had been closed.

## It was a workspace pattern, not a katgpt-rs defect

`scripts/ci_test_execution_report.py` (a report, not a gate — always exit 0;
population derived from BOUNDARY.md + a `.git` dir, vocabulary committed as
data, `selftest()` on every invocation exiting **2** rather than printing)
swept every contract repo. Three repos were COMPILE-ONLY over **31,841
`#[test]` sites**: katgpt-rs (15,659), riir-train (15,147), riir-game-sdk
(1,035). All three are now closed — riir-game-sdk `.issues/024` (`f02daa3`),
riir-train `.issues/507` (`e0716476`), katgpt-rs here.

`riir-game-sdk` was a fresh instance of Issue 706's class that the 706 sweep
missed: it *had* a daily cron, on the wrong branch. 706 fixed three repos by
adding a schedule that fires from the default branch; this one had the schedule
and a `main` default with the workflow only on `origin/develop`, so its
`30 17 * * *` cron had **never fired**.

### Two defects in that report, both found by disagreement, both changed a verdict

1. **`reachable_triggers` mixes live triggers with negative `-<trigger>`
   markers** for declared-but-dead ones, so a workflow whose every trigger is
   dead returns NON-EMPTY. A naive `if not trig` read `riir-game-sdk/nightly.yml`
   as live and reported the repo EXECUTES. `ci_gate_coverage.py` had it right;
   the newer instrument was wrong. Resolved by `live_and_automatic()`, which
   also refuses to count `pull_request` (policy git cannot see) or a lone
   `workflow_dispatch` (a button — Issue 706).
2. **Half this workspace's guard scripts announce their layers with the command
   name** — `echo "── L4: cargo test (default features) ──"`. A token-only
   matcher credits every one as an executed test, so a repo whose only match
   was a *label* would read EXECUTES while running nothing. A label's `cargo`
   is always inside a quoted string and a real invocation's never is — **except
   inside `$(...)`, which IS command context even when the substitution sits in
   quotes** (riir-chain's `out="$(cargo test …)"` is a real run). Both pinned.

`--no-run` / `--list` are classified as **COMPILE, not run**, and pinned:
`cargo test --no-run` matches the token and executes nothing.

## The resolution — two tiers, deliberately different

**(b) A scoped weekly job, owner-authorized and LANDED 2026-09-04:**
`scripts/test_gate.sh` + `.github/workflows/test.yml` (weekly Tue 05:03 UTC
from `develop` + dispatch, ubuntu-latest, no sibling checkout). Scope: the
default-feature `--lib` suites of katgpt-rs (floor 203) + katgpt-core (floor
1974) — 2,177 assertions executed automatically, floors firing downward only,
the `#![cfg]`-green-zero trap handled by both the floor and an
exactly-one-result-line parse discipline, `--canary` proving the floors live.
Platform invariance was grep-verified at landing: katgpt-core has ZERO
`#[cfg(target_os)]` attributes (its two `target_os` sites are runtime `cfg!()`
bools) and the root lib's `target_os` gates are dead at default features on
every platform.

Honestly **not** covered, which is the point of stating the scope: the 480
integration-test targets, the 176 bench targets, and every Metal / ANE / 4090
surface. Expanding is a one-line `ROWS` addition.

**(a) The full-workspace `--all-features --release` execution stays
DISPATCH-ONLY.** It was priced on a quiet box first, because a scheduled job
that has never been priced is the cost-blindness this audit exists to prevent.
Cost, census and the reds it found:
[`../../.benchmarks/701_full_workspace_execution_pricing.md`](../../.benchmarks/701_full_workspace_execution_pricing.md).

Measured 2026-09-04 on a quiet box (load 4.01 at launch), cold isolated target
dir, HEAD `172f5520`: **2,728.6 s wall / 11,542 CPU-s (3.21 CPU-hours) /
14.39 GiB peak RSS / 512 binaries / 497 suites ok, 45 targets FAILED.** A warm
comparator read 2,319 CPU-s, so **the cold compile is ~80% of the cost and
executing the tests is the cheap part** — a recurring full job would pay mostly
for a rebuild. `--no-fail-fast` is mandatory: default cargo aborts at the first
failing package, truncating both the failure list and the cost figure, and it
burned five launch cycles before an enumeration pass completed.

**The verdict is NOT about cost.** 3.21 CPU-hours is affordable weekly. The
run stays dispatch-only because **`--all-features` is not a supported *test*
configuration** — the katgpt-rs twin of riir-ai Issue 830's "`--all-features`
was never a checked configuration". It cannot pass as-is at any budget: 9
targets' fixture weights are seeded from an RNG stream whose draw count depends
on feature-gated struct fields, so unification changes every committed platform
hash. Those tests are correct under their own feature set and *meaningless*
unified; re-pinning them would destroy the per-feature claim they exist to make.

The three honest options, priced (owner picks; **(i) is the standing state**):

- **(i) keep the scoped job** — landed, ~zero marginal cost, machine-invariant
  core; the full run stays dispatch-only for triage campaigns.
- **(ii) a per-target triaged weekly integration job** — the seal-remake
  `e1ead85` shape over the ~470 default-runnable integration targets. Machine
  cost ≈ the execution half (~2,300 CPU-s) + one feature-set compile;
  engineering cost = triaging `.issues/723`'s reds into expect-red pins.
- **(iii) per-feature-set GOAT-bench lanes** — each gate under its OWN
  committed feature set, the way it was calibrated. Most honest numerically,
  highest matrix cost, and the only option that makes the fixture-drift class
  meaningful rather than suppressed.

**Running it paid for itself in correctness, not just in cost knowledge.**
Getting a run to complete required six fixes, all scanned workspace-wide:
`convergence_cadence` G4 (`dd734f2f`, E0432 from a module-level import of
debug-only alloc counters — a regression of the 720-T1 landing), `bckvss
builder_rejects_bad_segment_len` (`5fa7b5f1`, `#[should_panic]` on a
`debug_assert!` contract — 5 of 6 workspace sites were already gated, this was
the one), three load-fragile timing gates re-pinned with measured bands
(`a9576e20`, `ff6a4d46`, `172f5520`), and — the one that justifies the exercise
— **`d2f capture_q_row` (`d3454eff`), a real latent correctness bug in the
Issue-587 ExactQ law** where rows summed to `1 + exp[mask]/sum_exp`. It was
reachable only because `gated_mlp`'s extra weight draw shifted a fixture's RNG
stream. No compile gate can find that.

**Price in CPU-seconds, not wall-clock.** A wall-clock figure from this
workstation is uninterpretable — the box ran at load average 44-87 for a whole
day from sibling work. CPU time and peak RSS measure what a process *consumed*
rather than how long it *waited*; seal-remake
`.benchmarks/002_png_vs_ktx2_host_cpu_rss.md` measured that directly (over a 2x
load swing the CPU ratios moved by under 0.11 and never reordered an arm).
`/usr/bin/time -l` on the cargo invocation is the whole instrument, and
CPU-seconds is also closer to what Actions bills than wall-clock is.

## What running it actually found

The first execution was **red: 47 targets**, in six classes that do not share a
fix — including 8 crates' doc-tests that had **never been compiled at any
revision**, because `--all-targets` does not include doc-tests. That is a
seventh blind-spot axis in AGENTS.md's table, and it was invisible precisely
because the gate that exists to compile everything cannot reach it.

Full taxonomy, per-class actions and the open tasks: **`.issues/723`**.

## The caveats that survive

- **Compile-only CI may be a deliberate cost decision** — the gap was that the
  decision was written down nowhere, so a green full gate read stronger than it
  was. The two-tier answer makes the choice explicit rather than implicit.
- The tests were never unrun in an absolute sense: agents run them on the
  workstation constantly. The claim is narrower and is the one that matters for
  rot — **no automatic trigger executed any of them**, so a regression is found
  when somebody happens to look.
- **`--all-features` on a test RUN is not the same claim as on a compile.** It
  is one configuration of many, and the `-p` vs `--workspace` axis applies to
  execution too. This was written down *before* the run and then measured by
  it: 9 `issue_698_*` targets red on fixture pins that a default-features run
  may well satisfy (`.issues/723` Class C).
