# Issue 718 — CI compiles every gate and EXECUTES none (katgpt-rs, riir-train, riir-game-sdk)

**Status:** OPEN — T1 + T4 DONE, **T2 WITHDRAWN on measurement (its premise was
false — no Rust selftest population exists; the 9 Python auditor selftests
are already run per-push, and T2 collapses into T3)**, filed 2026-09-03.
T4's sweep found **two more repos in the same state — riir-train (15,147
`#[test]` sites) and riir-game-sdk (1,035, via a cron that has never
fired)** — so this is a workspace pattern, not a katgpt-rs defect. Only T3
remains and it is an OWNER COST CALL. Found sideways, while closing a
*different* instance of the same class in seal-remake (`e1ead85`).

## The measurement

`cargo clippy` and `cargo check` **compile** test targets. Neither runs one.
Measured 2026-09-03 on `develop`:

```
grep -rnE "cargo (test|nextest|bench)" scripts/ .github/
```

returns **two** hits, and both are prose — a docstring and a report string
inside `scripts/cfg_gated_target_audit.py`. There is no test-executing
command anywhere in `scripts/` or `.github/`.

Per workflow, the command each one actually reaches:

| workflow | trigger | what it runs | executes tests? |
|---|---|---|---|
| `full_gate.yml` | weekly + dispatch | `scripts/full_gate.sh` | **no** |
| `docs_gate.yml` | per-push | python auditors | no (n/a) |
| `feature_isolation.yml` | per-push (diff-bounded) | `feature_isolation_gate.py` → `cargo check` | **no** |
| `feature_isolation_weekly.yml` | weekly | same, `--scope default-on` | **no** |
| `lean_proofs.yml` | dispatch | `proof_gate.sh` (Lean) | no (n/a) |
| `sibling_docs_drift.yml` | `workflow_call` | python auditors | no (n/a) |
| `release-plz.yml` | dispatch | release | no (n/a) |

And `scripts/full_gate.sh`'s six layers are compile-only by construction:

| layer | command |
|---|---|
| 1 | cargo present |
| 2 | platform coverage |
| 3 | `GATE_ARGS` = `cargo clippy --workspace --all-targets --all-features --keep-going` |
| 3b | liveness — did the run examine anything |
| 4 | zero errors |
| 5 | doc/script parity |
| 6 | `REL_ARGS` = `cargo check --workspace --all-targets --all-features --keep-going --release` |

**Scope of what that leaves unexecuted** (`cargo metadata`, not a guess):
**477 integration-test targets, 31 lib targets (unit tests), 176 bench
targets, over 32 packages.** Zero are run by any automatic trigger.

## Why this is the next rung on a ladder this repo already climbed

AGENTS.md documents three rungs and stops one short of this:

1. **"a workflow file is identical on disk whether or not it can execute"**
   → workflows on a non-default branch are inert (`.issues/704`).
2. **"can fire is not does fire"** → a `workflow_dispatch`-only gate is a
   button, not a schedule (`.issues/706`).
3. **"a green test count can be a count of nothing"** → a `#![cfg]`-gated
   file compiles empty and prints `ok. 0 passed` (`.docs/10_audits/cfg_gated_silent_zero_pass.md`).
4. **← THIS: "compiles is not runs."** A gate CI compiles and never executes
   is in exactly the state rung 3 warns about, except the count is not zero
   — there is no count at all, because nothing produced one.

By this repo's own standard — *"Treat an uninvoked assertion as unknown, not
as passing"* — every Rust assertion here is currently **unknown**.

## This makes Issue 713 T3's arming half-complete

713 T3 added `required-features` rows to 39 GOAT gates (`180be9c5`) so that
naming a target without its features errors with exit 101 instead of
reporting a green zero. AGENTS.md records the safety argument for that
change as:

> Adding the rows is safe and does **not** red an existing CI: `cargo test
> --workspace` silently *skips* a target whose required-features are off.

That is true, and read one way it is the other edge of the same fact: the
rows make a **named** run honest, and **nothing names them**. The arming
fixed the failure mode where a green zero is cited as evidence; it did not
create a path by which the 39 gates ever run. AGENTS.md also says *"All 39
pass there"* under `--release` — that was a **workstation** measurement, and
nothing repeats it.

Not a criticism of 713 T3, which did what it set out to do. The point is
that "armed" and "run" are separate axes and only the first was closed.

## The same class, different mechanism, already fixed once (the cross-check)

`seal-remake` had the sibling instance and it is closed (`e1ead85`, this
session): that repo's guard DOES run `cargo test --workspace`, but all three
`texture_vessel` test targets carry `required-features = ["texture_vessel"]`
(default-OFF), so the run built 7 test executables and **none** of them.
Its Issue 001 G1/G2/G4 gates and Issue 002 instrument selftests were claimed
DONE while nothing automatic had executed one assertion — and layer 2's
`--all-features` clippy **compiled** them, which is exactly why it stayed
invisible. Fixed by a `layer 3b` that names each target and floors its
assertion count per target.

Two mechanisms, one class:

| repo | `cargo test` in CI? | required-features targets reached? | result |
|---|---|---|---|
| seal-remake (before `e1ead85`) | yes | no — silently skipped | 13 assertions unexecuted |
| **katgpt-rs (now)** | **no** | n/a — nothing runs | **508 targets unexecuted** |

## Tasks

- [x] **T1 — document the axis (free, and mandatory regardless of T3).** DONE — AGENTS.md now carries a sixth **compile vs EXECUTE** row in the full-gate blind-spot table plus the paragraph that tells a reader what a green gate does and does not claim. The same edit removed a live instance of the drift that table exists to catch: the preamble said "three **independent** axes" while the table had five, so the count is now not written at all.
  AGENTS.md's full-gate section enumerates five blind spots of the gate
  command and every one of them is a *compilation* axis; a reader finishes
  it believing a green full gate is a strong whole-repo claim. Add the sixth
  row: the gate does not execute anything. State plainly that CI is
  compile-only, so a green is never evidence that a test passes. Cheap,
  ungated, and it stops the misreading immediately.
- [-] **T2 — a per-push lane for the INSTRUMENT selftests. WITHDRAWN — the
  premise was false, measured 2026-09-03 within the hour of filing it.**
  This task assumed katgpt-rs had Rust-side analogues of the auditors'
  `selftest()` functions. It does not. Measured over the whole Rust corpus:

  | vocabulary | distinct `fn` names |
  |---|---:|
  | `self_?test` | **1** (`self_test`, in `benches/bench_420_kv_consolidation_poc.rs`) |
  | `instrument` | 2 |
  | `canary` | 1 |
  | `sentinel` | 5 |
  | `harness` | 4 |

  (A first cut of that probe was wrong in the *loose* direction — a
  `pin[a-z_]*` pattern matched **map**ping / clam**ping** / tap**ping** /
  overlap**ping** — which is the mirror of the narrow-classifier failure
  `.docs/10_audits/cfg_gated_silent_zero_pass.md` T4c documents. Word boundaries, then re-measure.)

  Meanwhile the `selftest()` population that *does* exist is **9 Python
  auditors** (`all_ignored_target_audit`, `bench_doc_audit`,
  `cfg_gated_floor_gate`, `cfg_gated_target_audit`, `ci_gate_coverage`,
  `feature_isolation_gate`, `orphaned_attr_gate`, `percentile_index_audit`,
  `staged_set_audit`) and **`docs_gate.yml` already runs them per-push**.
  The instrument layer is covered; it is the Rust layer that is not, and
  there is no cheap subset of it that is selftest-shaped.

  **So T2 collapses into T3.** The realistic cheap lane is
  `cargo test --workspace --lib` (skips the 477 integration binaries, which
  are the expensive part), but it still pays the workspace COMPILE that got
  per-push declined in the first place. There is no free option hiding
  here, which is worth knowing before anyone else goes looking for one.

- [ ] **T3 — a full test job (OWNER COST CALL, do not self-authorize).**
  `cargo test --workspace --all-features --release` is the complete fix and
  is expensive: AGENTS.md already records the full gate at >13 min for
  compile alone and says per-push was *deliberately* declined on measured
  cost, and `--release` is required because four gates false-RED in debug
  and `fast_bpe_goat` is 388 s debug vs 15.6 s release. Price it first
  (one dispatch run), then let the owner choose weekly / dispatch-only / a
  subset. **Do not add a per-push job for this.**

  **Price it in CPU-seconds, not wall-clock** — and that is not a detail.
  A wall-clock figure taken on this workstation is uninterpretable: the box
  ran at load average 44-87 all of 2026-09-03 from sibling work. CPU time
  and peak RSS measure what a process *consumed* rather than how long it
  *waited*, which `seal-remake` `.benchmarks/002_png_vs_ktx2_host_cpu_rss.md`
  measured directly: over a 2x swing in box load the CPU ratios moved by
  under 0.11 and never reordered an arm. `/usr/bin/time -l` on the cargo
  invocation is the whole instrument, and CPU-seconds is also closer to what
  Actions actually bills than wall-clock is. **Not run here on purpose:**
  starting a second heavy cargo job alongside the one already running would
  have contaminated both and degraded the shared box for other sessions.

  **Re-check 2026-09-03 (follow-up session):** box conditions re-read before
  pricing — load average 61–73 with 6 sibling cargo/rustc processes live. The
  same exclusion still binds, so the run stays deferred. The methodology
  survives the wait: CPU-seconds via `/usr/bin/time -l` is load-invariant (the
  seal-remake `.benchmarks/002` lesson — CPU ratios moved <0.11 over a 2× load
  swing), so the measurement remains valid whenever a quiet window appears.

  **OWNER DECISION (2026-09-04, T3 resolved as a two-tier answer):**
  (a) **The full-workspace `--all-features --release` execution stays
  dispatch-only** — priced on a quiet box (load ≲ 10, CPU-seconds via
  `/usr/bin/time -l`) BEFORE any scheduled job; a scheduled job that has never
  been priced is the cost-blindness this issue exists to prevent.
  (b) **A scoped weekly test job is AUTHORIZED for the machine-invariant
  core** — the riir-train 507 shape (`scripts/test_gate.sh` + count floors +
  a separate `tests` CI job), scoped to default-features lib + integration
  targets that are deterministic on a Linux runner (no Metal, no 4090, no
  wall-clock bars — those stay workstation-owned per the
  `budget_gate.sh`/wall-clock-bar precedent). The scoped subset is the
  implementation work that remains on this issue; the floors make a
  silent-population regression loud. Until it lands, this issue stays OPEN
  as the tracker.
  No other work remains on this issue: T1/T4 DONE, T2 withdrawn, G4 MET — T3
  is the sole open item and it is owner-gated on cost plus box quiet.
- [x] **T4 — sweep every contract repo. DONE — `scripts/ci_test_execution_report.py`.**
  A report, not a gate (always exit 0); population derived (BOUNDARY.md +
  a `.git` dir), vocabulary committed as data, `selftest()` on every
  invocation exiting 2 rather than printing. It imports
  `ci_gate_coverage.py`'s reachability machinery rather than duplicating it.

  **katgpt-rs is not alone. Three repos are COMPILE-ONLY — 31,841 `#[test]`
  sites compiled by CI and executed by nothing automatic:**

  | repo | `#[test]` sites | why nothing runs them |
  |---|---:|---|
  | riir-ai | 16,724 | — EXECUTES (`rust.yml`) |
  | **katgpt-rs** | **15,659** | no CI runs `cargo test` at all; all 6 `full_gate.sh` layers are clippy/check |
  | **riir-train** | **15,147** | 5 automatic compile commands, zero test runs |
  | **riir-game-sdk** | **1,035** | its ONLY test run is `nightly.yml`, which **cannot fire** |
  | riir-chain / riir-clippy / riir-neuron-db / riir-mmorpg-examples / riir-dapps / riir-deployer / riir-auth / riir-viewbridge / riir-dao / riir-burner / seal-remake | 6,848 → 19 | EXECUTES |
  | seal-game-editor | 2,101 | NO-CARGO (0 workflows) — read-only repo, report only |
  | riir-unity, seal-remake-unity | 0 | NOTHING-TO-RUN, so compile-only is correct |
  | katgpt-web | 4 | UNMEASURED — no remote refs to read a default branch from |

  **`riir-game-sdk` is a fresh instance of Issue 706's class and was missed
  by that sweep.** Verified directly, not inferred: its default branch is
  `main`, `nightly.yml` lives only on `origin/develop`, and
  `schedule`/`workflow_dispatch` fire only from the DEFAULT branch — so its
  daily `30 17 * * *` cron has **never fired**. 706 fixed three repos by
  adding a schedule that fires from the default branch; this one has the
  schedule and the wrong branch, which is why it did not match.

  **CORRECTION — it was already owned, and I asserted otherwise without
  checking.** This task first read "needs an owner action in that repo".
  `riir-game-sdk/.issues/024` had already been filed the same day, by a
  sibling session, from `ci_gate_coverage.py`, and it is MORE complete than
  my finding: it carries the 162-commits-behind-`main` measurement, the
  uninvoked `scripts/test_gate.sh` with its 8 count-pinned rows, the
  five-day pin drift that went unnoticed for exactly this reason, and a
  defect I did not have — even if `nightly.yml` were reachable it would
  **skip cleanly** on a hosted runner, because this workspace's sibling path
  deps are not fetched by `actions/checkout` and the job exits 0. I
  contributed the size (1,035 sites) and the cross-repo context to 024
  (`riir-game-sdk` `204c55e`) rather than filing a duplicate. **Check for an
  existing issue before writing "needs an owner action" — a sweep finds the
  symptom, not who already owns it.** (It is also the repo shipping the
  vessel packer `seal-remake` Issue 001 depends on.)

  **Ownership of all three instances, checked rather than assumed — and now
  CLOSED rather than assumed-open (2026-09-04):** katgpt-rs = this issue (the
  LAST remaining instance); `riir-game-sdk` = was `.issues/024`, now CLOSED
  (`f02daa3`, 2026-09-03 — weekly scheduled `test_gate.sh` + count pins);
  `riir-train` = was `riir-train/.issues/507`, now LANDED (`e0716476` —
  `scripts/test_gate.sh` + a separate `tests` CI job with per-target floors,
  1,487 tests at the floor measurement). The "filed rather than fixed"
  posture below describes why a blanket `cargo test --workspace` is the
  wrong shape here — that reasoning still holds for THIS repo and T3 is the
  scoped answer.

  **Two defects in the report were found by disagreement, and both changed a
  verdict** — recorded because each is a trap for the next reader:

  1. **`reachable_triggers` MIXES live triggers with negative `-<trigger>`
     markers** for declared-but-dead ones, so a workflow whose every trigger
     is dead returns NON-EMPTY. A naive `if not trig` read
     `riir-game-sdk/nightly.yml` as live and reported the repo EXECUTES.
     `ci_gate_coverage.py` had it right; the newer instrument was wrong.
     Resolved by `live_and_automatic()`, which also refuses to count
     `pull_request?` (policy git cannot see) or a lone `workflow_dispatch`
     (a button, 706).
  2. **Half this workspace's guard scripts announce their layers with the
     command name** — `echo "── L4: cargo test (default features) ──"`,
     `layer "4/4 cargo test --workspace …"`, `fail "cargo test …"`. A
     token-only matcher credits every one as an executed test, and a repo
     whose only match was a label would read EXECUTES while running nothing.
     A label's `cargo` is always inside a quoted string and a real
     invocation's never is — except inside `$(...)`, which IS command
     context even when the substitution sits in quotes (riir-chain's
     `out="$(cargo test …)"` is a real run). Both shapes pinned.

  `--no-run` / `--list` are classified as **COMPILE, not run**, and pinned:
  `cargo test --no-run` matches the token and executes nothing.

## Gates

| Gate | Criterion |
|---|---|
| G1 | T1's AGENTS.md row exists and the docs gate stays green |
| G2 | ~~T2's selftest lane FAILS on a target that reports `ok. 0 passed`~~ — VOID, T2 withdrawn. If T3 lands a test job the criterion transfers to it: a target reporting `ok. 0 passed` must FAIL, canaried by making one report zero rather than argued (the `seal-remake` `layer 3b` shape) |
| G3 | ~~T2's population is derived~~ — VOID with T2. Transfers to T3 and to T4 unchanged: derived from the tree, never typed |
| G4 | **MET** — the report prints `#[test]` sites beside every verdict, so COMPILE-ONLY over 15,659 sites and NOTHING-TO-RUN over 0 are never the same row; `--no-run` is classified as a compile |

## Honest caveats

- **Compile-only CI may be a deliberate cost decision**, and if so this
  issue is mostly T1: the gap is that the decision is written down nowhere,
  so a green full gate reads stronger than it is. T3 exists to make the
  choice explicit rather than implicit.
- The tests are not unrun in an absolute sense — agents run them on the
  workstation constantly, and AGENTS.md records workstation results
  (the 39 armed gates at `--release`, the 704-test `--workspace` count).
  The claim here is narrower and is the one that matters for rot: **no
  automatic trigger executes any of them**, so a regression is found when
  somebody happens to look.
- `--all-features` on a test RUN is not the same claim as on a compile: it
  is one configuration of many, and the `-p` vs `--workspace` axis
  AGENTS.md documents applies to execution too. T3 should not be sold as
  total coverage.

## References

- `.issues/713` T3 / T4 + `.docs/10_audits/cfg_gated_silent_zero_pass.md` — the arming half of this
- `.issues/704` (inert workflows) + `.issues/706` ("can fire is not does fire") — rungs 1-2
- `scripts/full_gate.sh` layers 1-6; `scripts/ci_gate_coverage.py` (T4's home)
- `seal-remake` `e1ead85` — the sibling instance, closed: guard `layer 3b`,
  per-target named floors, and `.benchmarks/002_png_vs_ktx2_host_cpu_rss.md`
  for the work that surfaced it
