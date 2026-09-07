# Issue 735 — Issue 734's laundering premise is bash-3.2-ONLY, and "and 5.x" was never measured

**Status:** RESOLVED 2026-09-07 — measured, instrumented, and swept.
T0/T1/T2/T2b/T5 done, T3 wired (answers itself on the next weekly cron), T4
gated on T3. The premise reproduces on **bash 3.2.57 only**; 4.4, 5.0, 5.2,
5.3, dash and busybox ash all **preserve** the status. All 41 files carrying
the refuted "bash 3.2 and 5.x" sentence are corrected across 11 repos, and
the 15 nounset-only scripts now say so in-file. Two sessions worked this in
parallel; the split is recorded per task.

**T3 ANSWERED EARLY + T4 RESOLVED, same day (run 34137014037, the 737
layer-2b push — the probe runs in the preamble, so every trigger carries
it, cron not required):** GitHub's `macos-26-arm64` image ships **bash
3.2.57(1) ONLY** — PATH bash = `/bin/bash` = `env bash`, and the expected
Homebrew bash 5 on PATH does not exist there — and the premise matrix
re-measured on the runner reproduces **all five errexit LAUNDERS cells**.
The macOS CI lane launders exactly like the workstation: the sentinel is
load-bearing IN CI, and 735's original "CI reads a pass" sentence was
accidentally right for this repo's one macOS runner all along. T4 resolved
below: do not pin — measure.

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
- [x] T2 — wording sweep, **41 files, 11 repos, zero occurrences left**.
      Population derived from `git ls-files` per repo, never a `grep -r` from
      the workspace root: that walks every repo's `target/` (~1 TB) and does
      not finish — the first attempt was still running after two minutes and
      had to be killed. Split with the peer session so no two sessions edited
      one comment block:

      | repo | files | commit |
      |---|---|---|
      | riir-train | 8 | `1a7dc65b` |
      | riir-mmorpg-examples | 6 | `2eef716` |
      | riir-ai | 4 | `ca0d1ab44` → pushed as **`c48c82809`** |
      | riir-clippy | 3 (incl. HISTORY.md) | `da6ba36` → pushed as **`62bc60f`** |
      | riir-dapps | 2 | `0117488` |
      | riir-auth | 1 | `6ea9cdc` |
      | riir-deployer | 1 | `247080b` |
      | riir-viewbridge | 1 | `4839b25` |
      | seal-remake | 1 | `b4b3c4cd` |
      | riir-chain | 11 | `f2d83c71` *(peer)* |
      | katgpt-rs | 2 + AGENTS.md | `a95d2bd6` *(peer)* |

      Two of those hashes moved: riir-ai and riir-clippy had both advanced on
      the remote between my commit and my push (a live 4090 benchmark session
      in one, a distill-queue update in the other), so each was rebased —
      riir-clippy with `--autostash`, because a sibling session had two dirty
      `*.rs` files that a plain rebase would have refused and a manual stash
      could have swept. Both dirty files came back intact; the **local→pushed
      remap is recorded above** because the pre-push hashes appear in no
      remote and would resolve for nobody.

      Four of the 41 sites did not match the dominant 2-line boilerplate and
      were read and reflowed individually (`riir-clippy/HISTORY.md`,
      `riir-deployer/cloudflare/control-do/test.sh`, and two katgpt-rs sites
      in the peer's half). A looser regex would have "fixed" them into
      ungrammatical sentences — UNMATCHED is a per-file read, not a failure.

      **Pushed commit bodies were deliberately NOT amended** (`21da73a`
      riir-viewbridge, `7efe2a23` seal-remake, and the T4/T5 wave): they are
      cited by hash in 734's tally table, and rewriting pushed history to fix
      a provenance sentence trades a real hazard for a cosmetic gain. Agreed
      with the peer session that landed the wave.
- [x] T2b — **the wording sweep alone would have left a SECOND over-claim in
      place.** "3.2 and 5.x" was one error; the other is that the block
      describes an unbound-expansion abort laundering while **15 of the 41
      scripts set `set -u` WITHOUT `set -e`** — and every measured laundering
      cell needs errexit (734 T10), so their own abort already exits 1. Nine
      of the fifteen were in my half (riir-ai `ci_boundary_contract.sh`,
      7 riir-train, seal-remake `fps_matrix.sh`) and now carry a per-script
      PRECAUTIONARY note: `d592b8733`, `5cbb256f`, `6e38b8e0`; the peer's six
      riir-chain scripts in `f2d83c71`. Membership taken from
      `trap_exit_launder_audit.analyse()`'s `errexit` field rather than a
      grep — the classifier decides the severity split, so it should pick the
      file list. No sentinel was removed: the flag also catches premature
      deaths that are not version-specific (SIGTERM, a `set -e` trip in an
      unguarded spot, a future editing slip). Classification re-run after
      every edit: unchanged, 40/41 SENTINELLED.
- [x] T3 — **wired, and it answered EARLY** (peer, `a95d2bd6`; first answer
      run `34137014037` the SAME DAY — the probe is a preamble step, so the
      737 layer-2b push carried it without waiting for the weekly cron):
      a probe in `full_gate.yml` — this repo's only `macos-latest` runner of
      a sentinelled script — printing PATH bash / `/bin/bash` / `env bash`
      versions plus the audit's re-measured premise matrix, ~1s, `|| true`
      (a probe that can red a 180-minute gate gets deleted).

      **The measured answer (macos-26-arm64, 2026-09-07):** all three
      resolutions are the SAME bash — `3.2.57(1)-release (arm64-apple-
darwin25)`. The T3 note's premise that the image "ships Apple's 3.2.57 at
      `/bin/bash` *and* a Homebrew bash 5 on PATH" measured **one-sided**:
      there is no Homebrew bash in the runner's PATH at all, so
      `shell: bash` cannot silently flip to 5.x on this image. The
      re-measured premise matrix on the runner reproduces the five
      workstation LAUNDERS cells exactly (`set -e`/`set -eu`/`set -euo
      pipefail` × unbound/eval → 0 trapped). **Consequence: the macOS CI
      lane launders an aborting gate exactly like the workstation — the
      completion sentinel is load-bearing in CI, not merely on dev boxes —
      and the one shape of run where that mattered (`full_gate.sh` under
      errexit on this runner) is sentinelled.** Result no longer waits on a
      cron; the step stays so a future image change re-measures for free.
- [x] T4 — **RESOLVED 2026-09-07, un-gated by T3's same-day answer: do not
      pin — measure.** Three legs, all measured rather than argued:

      1. **Nothing to pin apart.** On the runner, `shell: bash` and
         `shell: /bin/bash` resolve to the SAME interpreter — PATH bash IS
         `/bin/bash` 3.2.57 (no Homebrew bash exists in the image's PATH).
         The drift T4 worried about is not real on the current image.
      2. **A `shell:` pin could not govern the gate scripts anyway.** They
         carry `#!/usr/bin/env bash` shebangs, so `./scripts/full_gate.sh`
         resolves its own interpreter at exec time; pinning the STEP shell
         pins only the run block around it. Truly pinning the script would
         mean changing shebangs repo-wide (and on every workstation run)
         — a large, correctness-neutral churn.
      3. **The verdict is drift-proof without a pin.** The sentinel is
         correct under BOTH bashes — load-bearing on 3.2 (forces the
         laundered abort to exit 1), inert and harmless on 5.x (aborts exit
         non-zero naturally). A future image change in either direction
         cannot produce a wrong verdict; the probe step re-prints the three
         version lines + the re-measured premise matrix on every run, so
         the drift would be OBSERVED, never silent.

      Pinning `/bin/bash` would freeze the laundering interpreter
      deliberately; pinning a Homebrew bash would chase a PATH entry the
      image does not have. The probe stays as the drift detector.
      Recorded in the workflow comment at `e0b7c9e0`.
- [x] T5 — AGENTS.md now names all four instruments and which question each
      answers, because reading one for another is the live hazard (peer,
      `a95d2bd6`): population/classification (`trap_exit_launder_audit.py`),
      verdict for this repo (`trap_sentinel_gate.py`, in the docs gate),
      verdict for all 17 (`trap_sentinel_drift_sweep.py`, `91e854f3`), and
      the premise across interpreters (this issue's
      `trap_launder_premise_matrix.py`). Deferred initially on purpose —
      AGENTS.md was dirty in the concurrent session at the time, and a
      whole-file write would have swept their in-flight text.

## Cross-refs

- `.issues/734_shell_gate_exit_status_laundering.md` — the parent: the defect,
  the sentinel idiom, the population and verdict halves. T10 there is the
  finding that errexit (not nounset) is the precondition; this issue is the
  interpreter axis of the same premise. 734 T10 now records this refutation
  as **REFUTED, not merely unverified**, with an independent `bash:5.2`
  confirmation from the peer session — so the two issues no longer disagree.
- The refutation was reproduced independently before either of us acted on
  it: two sessions, two harnesses (in-process `bash -c` matrix vs. a
  script-file driver in containers), same conclusion. That is also how the
  measurement-MODE finding surfaced — a `bash -c` nounset abort exits **127**
  and a script FILE exits **1** on this bash, and the two agree at
  `set -euo pipefail`, so the old harness could never have seen it (734 T10).
- Repairs that stand regardless of this issue: riir-viewbridge `21da73a`,
  seal-remake `7efe2a23`, riir-chain `e3abbb3d`, katgpt-rs `70eff640`.
