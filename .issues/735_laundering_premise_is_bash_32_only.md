# Issue 735 — Issue 734's laundering premise is bash-3.2-ONLY, and "and 5.x" was never measured

**Status:** OPEN 2026-09-07 — measured and instrumented (`scripts/trap_launder_premise_matrix.py`,
11 interpreters). The premise reproduces on **bash 3.2.57 only**; bash 4.4,
5.0, 5.2, 5.3, dash and busybox ash all **preserve** the status. What is left
is the wording sweep across the ~27 repaired scripts in 9 repos (T2) and one
question nobody can answer from this box (T3: what bash a `macos-latest`
runner actually resolves).

## The claim, and what it turned out to be

Issue 734's prose, this repo's AGENTS.md, and the sentinel comment block
copied into every one of the ~27 repaired scripts across 9 repos all say the
same thing:

> Measured on bash 3.2 and 5.x: when bash aborts on an unbound expansion
> (`set -u`) or an `eval` syntax error it enters the EXIT trap with `$?`
> ALREADY 0 …

The **3.2 half is true**. The **5.x half was never measured** — there is no
bash 5 on this workstation (`/opt/homebrew/bin/bash`, `/usr/local/bin/bash`,
Cellar: all absent; peer session confirmed this independently before I did).
It was inherited text, restated as measurement, and it propagated into 27
files because the comment block was copied verbatim — which is exactly how a
convenient sentence becomes a load-bearing fact.

Measured now, one docker image per interpreter, script FILES not `bash -c`
(the mode matters — 734 T10):

| interpreter | where | laundering / aborts observed |
|---|---|---|
| **bash 3.2.57** | macOS `/bin/bash` | **5 / 13** ← the premise |
| **bash 3.2.57** | macOS `/bin/sh` | **5 / 13** (same binary, posix mode) |
| bash 4.4.23 | `docker bash:4.4` | 0 / 13 |
| bash 5.0.18 | `docker bash:5.0` | 0 / 13 |
| bash 5.2.37 | `docker bash:5.2` | 0 / 13 |
| bash 5.3.15 | `docker bash:5.3` | 0 / 13 |
| bash 5.2.37 | `debian:stable-slim /bin/bash` | 0 / 13 |
| busybox ash | `docker bash:5.3 /bin/sh` | 0 / 15 |
| dash | `debian:stable-slim /bin/dash` | 0 / 15 |
| dash | macOS `/bin/dash` | 0 / 9 (2 option sets inadmissible) |
| zsh | macOS `/bin/zsh` | 0 / 13 (2 cells: **trap never ran**) |

A **laundering cell** = the run really aborted (`post=0`) **and** the EXIT
trap was handed `$?` == 0. The five cells, all bash 3.2, all requiring
errexit (734 T10): `set -e` + eval-syntax, `set -eu` + unbound, `set -eu` +
eval-syntax, `set -euo pipefail` + unbound, `set -euo pipefail` +
eval-syntax.

**The laundering is a bash 3.2 defect, fixed no later than 4.4.**

## Why that changes the severity, and why it does NOT change the repair

Every repo in the workspace runs its gate scripts in CI on `ubuntu-latest`
(measured, 21 workflow files) — bash 5.x — where an aborting gate exits 1 or
2 and the job **fails**. So the sentence "and CI reads a pass" is wrong for
`ubuntu-latest`: CI was never laundered. What *was* laundered is every run on
this workstation and every other macOS box, where `/usr/bin/env bash` is
3.2.57 — which is where developers and agents actually run these gates, and
where seal-remake's `ci_feature_guard.sh` exited 0 on an abort.

Three katgpt-rs workflows declare `runs-on: macos-latest` —
`full_gate.yml`, `feature_isolation.yml`, `feature_isolation_weekly.yml`,
plus `required_features_touched.yml` — and `full_gate.sh` is one of the two
scripts repaired in 734 T2. Whether those runs laundered depends on what
`/usr/bin/env bash` resolves to on GitHub's macOS image (T3 below).

**The completion sentinel stays.** It costs nothing on 5.x, it is
load-bearing on every macOS run, and the abort it reports is real on *every*
interpreter — only the *status* differed. Nothing needs reverting; what needs
fixing is a sentence that claims more measurement than happened.

## Two findings the interpreter axis produced for free

Both are the same shape as 734's own lesson — a zero is only as wide as the
instrument that produced it:

1. **A shell can reject the option line.** macOS `/bin/dash` has no
   `-o pipefail` (Debian's dash ≥ 0.5.12 does), so `set -uo pipefail` kills
   it on line 1 with status 2 — *before* the trap is registered and before
   the arm runs. Those 8 cells look exactly like "aborted, trap never ran".
   Pooled into the total they would have reported a confident `0 laundering`
   over a measurement that never happened. The instrument probes
   admissibility per (shell, option set) and excludes + names them.
2. **`trap never ran` is a third shape, not a pass.** On zsh, the nounset
   abort does not run the EXIT trap at all: the status survives (exit 1) but
   the cleanup is **skipped** — leaked temp dirs, the same consequence 734's
   REPLACED verdict reports. Counting it as "status preserved" would hide a
   different defect behind a good number.

## The instrument

`scripts/trap_launder_premise_matrix.py` — a REPORT, not a gate (exit 0
always), the third orthogonal half of the 734 family:

| script | half |
|---|---|
| `trap_exit_launder_audit.py` | population — which scripts are exposed |
| `trap_sentinel_gate.py` | verdict — this repo's scripts, by membership |
| **`trap_launder_premise_matrix.py`** | **premise — which interpreters launder** |

```bash
scripts/trap_launder_premise_matrix.py            # this box's shells
scripts/trap_launder_premise_matrix.py --docker   # the pinned image sweep
scripts/trap_launder_premise_matrix.py --tsv      # raw rows, for diffing
```

One POSIX-sh driver, emitted from the Python and used for both the local and
the container path — the images have no python, and a second implementation
would be a second thing to keep true. **bash 3.2 exists in no docker image**
(there is no `bash:3.2` tag), so the premise's own interpreter can only be
measured on macOS itself; that asymmetry is why the local box is always
measured first. Docker absent or its daemon down prints **UNSEEN**, never a
zero.

## Tasks

- [x] T0 — measure the matrix on bash 3.2.57 (local) and 4.4 / 5.0 / 5.2 /
      5.3 (docker). Result: laundering is 3.2-only, and errexit is the
      precondition in every laundering cell (734 T10 agrees).
- [x] T1 — `scripts/trap_launder_premise_matrix.py`, with the two
      admissibility shapes above so a zero cannot come from an inert cell.
- [ ] T2 — wording sweep: ~27 repaired scripts in 9 repos carry
      "Measured on bash 3.2 and 5.x" in their sentinel comment block. The
      claim is false as written. Replace with what was measured, per script
      (they are one-line comment edits, but they are in 9 repos and several
      are shared worktrees — do it per repo, named paths, never `-A`).
      **Do NOT amend the pushed commit bodies that carry the same
      boilerplate** (`21da73a` riir-viewbridge, `7efe2a23` seal-remake, and
      the T4/T5 wave): they are cited by hash in 734's tally table, and
      rewriting pushed history to fix a provenance sentence trades a real
      hazard for a cosmetic gain. Agreed with the peer session that landed
      the wave.
- [ ] T3 — **open question, not answerable from this box:** what does
      `/usr/bin/env bash` resolve to on a `macos-latest` runner? If 3.2.57,
      then katgpt-rs's `full_gate.yml` *did* launder in CI and the 734
      narrative is right for this repo specifically; if the image puts a brew
      bash 5 first in PATH, it did not. One `bash --version` in a workflow
      step answers it — worth adding to `full_gate.yml`'s log preamble
      rather than a run of its own.
- [ ] T4 — consider whether the katgpt-rs macOS workflows should pin the
      interpreter explicitly (`shell: /bin/bash` vs `bash`), so the answer to
      T3 stops depending on image drift. Gated on T3.
- [ ] T5 — AGENTS.md pointer for the new instrument, in the same section that
      documents the audit and the sentinel gate. **Deferred on purpose:** at
      the time this landed, AGENTS.md was dirty in a concurrent session
      finishing 734 T9/T10 and `trap_sentinel_drift_sweep.py`, and a
      whole-file write would have swept their in-flight text. One paragraph,
      once that file is clean.

## Cross-refs

- `.issues/734_shell_gate_exit_status_laundering.md` — the parent: the defect,
  the sentinel idiom, the population and verdict halves. T10 there is the
  finding that errexit (not nounset) is the precondition; this issue is the
  interpreter axis of the same premise.
- Repairs that stand regardless of this issue: riir-viewbridge `21da73a`,
  seal-remake `7efe2a23`, riir-chain `e3abbb3d`, katgpt-rs `70eff640`.
