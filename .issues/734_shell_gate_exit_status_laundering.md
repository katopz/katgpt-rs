# Issue 734 — a shell gate that ABORTS mid-run reports exit 0

**Status:** OPEN 2026-09-07 — T0/T1/T2/T6/T7 done; the T4/T5 wave landed in
**7 sibling repos (24 scripts)**, taking the workspace from 38 EXPOSED to
**14**. What is left is 11 in riir-chain (deferred: 15 dirty files, another
session is mid-work there) plus 3 deliberate/out-of-scope singles. Found by
repairing seal-remake `26a18191`. The REPLACED verdict reported in `70eff640`
was a false positive — corrected under T3.

## The defect

Measured on bash 3.2.57 (macOS) — `scripts/trap_exit_launder_audit.py`
re-measures it on every run rather than trusting this table:

| abort | bare | `$?` at EXIT-trap entry | with an EXIT trap |
|---|---|---|---|
| `set -u` unbound expansion | 1 | **0** | **0** ✗ |
| `eval` with a syntax error | 2 | **0** | **0** ✗ |
| `set -e` command failure | 1 | 1 | 1 ✓ |
| command not found | 127 | 127 | 127 ✓ |

When bash aborts on an unbound expansion it enters the EXIT trap with `$?`
**already 0**. If an EXIT trap is registered and its last command succeeds
(`rm -f "$TMP"` always does), the script exits **0**. Everything after the
abort silently did not run and the caller — CI included — reads a pass.

**The defensive idiom does not fix it.** `trap 'rc=$?; cleanup; exit $rc'
EXIT` saves a status that is itself 0. Only a **completion sentinel** works:

```sh
GATE_COMPLETED=0
gate_cleanup() {
    st=$?
    ...cleanup...
    if [ "$GATE_COMPLETED" != "1" ] && [ "$st" = "0" ]; then
        echo "✗ aborted mid-run while reporting success — forcing exit 1" >&2
        exit 1
    fi
    exit "$st"
}
trap gate_cleanup EXIT
...
GATE_COMPLETED=1   # the last line
```

The arm fires only on the laundering combination — INCOMPLETE **and**
claiming success — so an ordinary layer failure still exits 1 with its own
message and is untouched.

## Why this is not hypothetical

seal-remake's `scripts/ci_feature_guard.sh` (what its `rust.yml` runs)
registered `trap "rm -f '$SCEN_LOG' '$SCEN_DUO_LOG' '$SCEN_RIG_LOG'" EXIT`
in double quotes — expanding at REGISTRATION time — naming two variables
assigned ~20 and ~45 lines later. Under `set -u` that aborted the script:
layer 13's duo and rig rows, both seal-view panel rows, and layer 14 could
never execute, `ALL LAYERS GREEN` could never print, and **the abort exited
0**, so past that line the gate could not fail. It stayed hidden because a
ratchet ceiling had been red for three commits and stopped every run before
reaching the bad line. Repaired in seal-remake `26a18191`; the first
complete run passed 34 rows.

## The instrument

`scripts/trap_exit_launder_audit.py` — a REPORT, not a gate (exit 0 always;
the top verdict is latent). Population is DERIVED (`BOUNDARY.md` + `.git`)
and restricted to **tracked** `*.sh` (`git ls-files`): the first cut walked
the filesystem and reported 25 findings in `katgpt-rs/.raw/rapid-mlx`, a
**gitignored** vendored drop no repo owns. Four verdicts: LIVE-FORWARD (a
double-quoted trap naming a later-assigned variable — a provable abort every
run), EXPOSED (`set -u` + EXIT trap + no sentinel), SENTINELLED, and the
orthogonal REPLACED (2+ EXIT traps — `trap` replaces, it does not
accumulate, so earlier cleanup is silently dropped). `selftest()` pins all
three verdicts plus two population controls.

**Two tightenings, both of which the tool got wrong first.** (1) The first
classifier called any late-assigned, handler-read variable a sentinel and so
reported 3 SENTINELLED — two of them false (`BOOTSTRAP_PID=$!`, `status=$?`).
A false SENTINELLED *hides* exposure. A completion flag is assigned only
**literal** constants; that is the discriminator now, and the selftest still
fires on a real sentinel, so the tightening did not blind it. (2) `trap -
EXIT` was counted as a second registration, making the whole REPLACED column
one false positive out of one finding. Both are the "a zero ceiling is only as
wide as its classifier" failure occurring *inside* the tool built to find it;
both are now selftest controls (6/6: 4 verdicts + 2 population controls).

## Measured population — before / after (tracked files only)

Copied from the tool's own tally, not re-typed — the first version of this
table was hand-entered and got two cells and the total wrong.

| repo | EXPOSED before | EXPOSED now | SENTINELLED | landed in |
|---|---|---|---|---|
| riir-chain | 11 | **11** | 0 | *deferred — see T3* |
| riir-train | 8 | 0 | 8 | `4d35aed4` |
| riir-mmorpg-examples | 6 | 0 | 6 | `3f20650` |
| riir-ai | 5 | **1** | 4 | `a8260ad23` |
| riir-clippy | 2 | 0 | 2 | `c152b80` |
| riir-dapps | 2 | 0 | 2 | `0148fc8` |
| riir-auth | 1 | 0 | 1 | `bd50158` |
| riir-deployer | 1 | 0 | 1 | `a632a4b` |
| riir-viewbridge | 1 | **1** | 0 | *out of scope — not a working directory* |
| seal-remake | 1 | **1** | 1 | guard: `26a18191` |
| katgpt-rs | 2 | 0 | 2 | `70eff640` |
| **ALL** | **40** | **14** | **27** | over 41 tracked scripts |

LIVE-FORWARD is 0 and REPLACED is 0 workspace-wide.

The three non-riir-chain remainders are each a decision, not an oversight:

* **`riir-ai/scripts/e2e_internet.sh` — left EXPOSED on purpose.** It is an
  interactive cluster runner: it prints *"Press Ctrl+C to stop"* and ends in
  `wait`. Ctrl-C **is** its normal termination, so a completion sentinel would
  make the documented usage report failure, and nothing reads its exit code.
* **`riir-viewbridge/scripts/ci_feature_guard.sh`** — that repo is not one of
  this session's working directories. Same one-line fix when it is.
* **`seal-remake/scripts/fps_matrix.sh`** — a second session was actively
  editing that worktree (confirmed: two `GUARD EXIT=0` lines appended to one
  shared log by two runs). Left to them rather than risk a collision.

## Tasks

- [x] T0 — reproduce the four arms minimally; establish that `rc=$?` +
      `exit $rc` does NOT repair it (the saved rc is 0).
- [x] T1 — `scripts/trap_exit_launder_audit.py`, population-derived,
      tracked-files-only, `selftest()` on both directions.
- [x] T2 — repair katgpt-rs's own two: `scripts/full_gate.sh` (the whole-repo
      assertion) and `scripts/proof_negative_test.sh`. Each verified on its
      REAL cleanup preamble: unbound abort → 1, `eval` syntax error → 1,
      ordinary verdict failure → 1 with its own message unchanged, clean
      completion → 0, and (full_gate) the `KEEP_LOG` retention path intact.
- [ ] T3 — riir-chain wave: 11 EXPOSED. **Deferred deliberately**, not
      skipped: that worktree had 15 dirty files from another session's
      in-flight riir-chaind work, and 11 mechanical edits across it is exactly
      the shared-worktree collision this workspace has a rule about.
      **Correction to this issue's first version (commit `70eff640`), and it
      is the interesting kind.** That commit reported one REPLACED finding —
      `riir-chain/scripts/program_rate_gate.sh`, "EXIT traps at 447 and 510,
      the second silently discards the first". It does not: line 510 is
      `trap - EXIT`, a **deregistration** (restore the default), which the
      script's own self-test does deliberately before re-running the gate
      clean. The detector counted a teardown as a handler. That was a **1-of-1
      false-positive rate** in the REPLACED verdict — the one column nobody
      would have re-derived, inside the tool built to stop exactly this.
      `trap - EXIT` and `trap "" EXIT` are now classified as deregistrations,
      the selftest carries a control for it (6/6), and REPLACED is 0
      workspace-wide.
- [x] T4 — riir-train `4d35aed4` (8/8), riir-mmorpg-examples `3f20650` (6/6),
      riir-ai `a8260ad23` (4 of 5; the 5th is the deliberate skip above).
      Placement is per-script and a blanket patch would have been wrong in
      **six** of the eighteen: a success `exit 0` inside a branch rather than
      at EOF (`ci_boundary_guard.sh`), TWO completion points with a
      CONDITIONAL trap (`deploy-static.sh`), a quick mode that exits before any
      trap exists (`riir-mmorpg-examples/ci_feature_guard.sh`), a final line
      that IS the verdict (`perf_rematch_detector_selftest.sh`:
      `[ "$FAIL" -eq 0 ]`), a trailing `exec` that REPLACES the process so the
      trap never runs (`plan341_t3_chain.sh`), and an EXIT trap registered only
      inside `--self-test` (`ci_boundary_contract.sh`, the workspace boundary
      gate — every other mode has no trap and was never exposed).
- [x] T5 — riir-clippy `c152b80` (2), riir-dapps `0148fc8` (2), riir-auth
      `bd50158`, riir-deployer `a632a4b`. riir-viewbridge is out of scope
      (not a working directory this session).
- [x] T6 — verdict half landed for THIS repo: `scripts/trap_sentinel_gate.py`,
      in the docs gate (14/14). Scoped, not workspace-wide, because a gate
      over 38 EXPOSED sibling scripts would be a ratchet, not a gate — each
      repo adds its own as its wave lands (T3-T5). Four arms: the audit's
      `selftest()` must pass FIRST (an inert classifier reports a green zero
      over everything), the population is FLOORED (a classifier going blind
      must RED rather than report 0 EXPOSED), the two scripts are pinned by
      MEMBERSHIP (a count is not a checksum over a set), and any NEW script
      arriving with a cleanup trap and no sentinel reds the commit that adds
      it.

      **The canary caught a false SENTINELLED in the classifier — the
      direction that HIDES exposure.** Planting "delete `full_gate.sh`'s
      last-line `FULL_GATE_COMPLETED=1`" did NOT red the gate: the script
      also carries `if [ "$KEEP_LOG" -eq 1 ]; then echo …; else rm -f …; fi`,
      and `KEEP_LOG` is assigned the literal `1` twice after the trap and is
      read in a `[ … ]` test inside the handler, so it satisfied every clause
      the rule asked for. The rule asked for "tests the flag" and "the handler
      exits non-zero" INDEPENDENTLY; a sentinel is the flag that gates *that
      exit*. Tied together by block structure now
      (`guards_nonzero_exit`), with the `KEEP_LOG` shape as a pinned selftest
      case (7/7). Both plants red the gate, the clean tree passes, and the
      workspace tally is unchanged at 38/3 — so the tightening did not
      reclassify any real sentinel.
- [x] T7 — **shellcheck is NOT the missing static half — measured.** Ran
      shellcheck 0.11.0 against the pre-fix guard (`seal-remake cafe9444:
      scripts/ci_feature_guard.sh`), which contains the live defect. Default
      severity: **1 finding total**, an `SC2001` sed-style nit — SC2154
      (referenced-but-not-assigned) does not fire, because the variables ARE
      assigned, just ~20 lines too late, and shellcheck does not order
      assignment against use for that check. With `-o all` (every optional
      check on) it emits 158 SC2250 + 23 SC2292 + 2 SC2310, and its only
      remark on the fatal line is `SC2250 (style): Prefer putting braces
      around variable references`. So the industry-standard linter, at
      maximum strictness, looks straight at the line that made a CI gate
      unfailable and suggests curly braces. `trap_exit_launder_audit.py`'s
      LIVE-FORWARD arm is the only detector for this class, and the sentinel
      is the only defence for the general one (an unbound expansion anywhere
      other than a trap body stays invisible to any static pass — `bash -n`
      does not expand).

- [x] T8 — an incidental find, from `bash -n` being part of verifying each
      edit: **`riir-ai/scripts/quest_manifold_capture.sh` had never parsed on
      macOS.** An apostrophe inside a heredoc nested in a command substitution
      breaks bash 3.2's parser — it tracks quoting while scanning for the
      closing paren, so the lone `'` in a comment reading *the window's "id"*
      swallowed the rest of the file. It is a macOS-ONLY script (osascript +
      screencapture) on a box whose `/usr/bin/env bash` is 3.2, so it could
      not run at all. Repro:

      ```sh
      X=$(cat <<EOT
      the window's id
      EOT
      )                  # bash 3.2: unexpected EOF.  bash 5: fine
      ```

      Fixed in `a8260ad23`. Note the second-order trap: the explanatory comment
      has to be free of apostrophes, backticks and quotes too — the first
      attempt at writing it reintroduced the break.

## Verification standard used for every repaired script

Not "it looks right": each script's REAL cleanup handler was extracted from
the file with `awk` and driven on three arms — unbound-variable abort must
exit **1** and print the new message; the script's own deliberate failure must
exit with **its own** status (1, or 2 for the boundary self-test and the
nightly's DEGRADED path) and print **no** extra output; clean completion must
exit **0**. `deploy-static.sh` got a fourth arm for its `DRY_RUN` early
`exit 0`, which must stay 0 and not read as incomplete. 27 scripts x 3+ arms.
Extracting from the tracked file rather than reasoning about a copy is the
point — `full_gate.sh`'s false SENTINELLED (T6) was found exactly this way.
