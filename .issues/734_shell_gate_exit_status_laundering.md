# Issue 734 — a shell gate that ABORTS mid-run reports exit 0

**Status:** OPEN 2026-09-07 — T0–T11 done. The wave has landed in **10 repos
(37 scripts)**; **40 of 41** tracked scripts are SENTINELLED and the single
remaining EXPOSED row is *provably inert* (see T10's window column). Found by
repairing seal-remake `26a18191`.

⛔ **Three corrections to this issue's own instrument, all three caught by a
canary or a peer's independent measurement — the tool built to stop
over-claiming over-claimed three times:**
1. the REPLACED verdict in `70eff640` was a 1-of-1 false positive (T3);
2. the brace counter mis-read one file into a false EXPOSED, and the same
   mis-parse the other way manufactures a false SENTINELLED — an **UNPARSED**
   verdict now covers whatever the parser cannot read (T9);
3. **the premise itself named the wrong shell option.** `errexit` is the
   precondition, not `nounset`; the harness had hard-coded `set -euo
   pipefail` and so never varied the axis it was making a claim about, which
   over-claimed on 15 of 41 rows (T10, found by session `katgpt-rs-b5`).

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
| riir-chain | 11 | 0 | 11 | `e3abbb3d` |
| riir-train | 8 | 0 | 8 | `4d35aed4` |
| riir-mmorpg-examples | 6 | 0 | 6 | `3f20650` |
| riir-ai | 5 | **1** | 4 | `a8260ad23` |
| riir-clippy | 2 | 0 | 2 | `c152b80` |
| riir-dapps | 2 | 0 | 2 | `0148fc8` |
| riir-auth | 1 | 0 | 1 | `bd50158` |
| riir-deployer | 1 | 0 | 1 | `a632a4b` |
| riir-viewbridge | 1 | 0 | 1 | `21da73a` (session `katgpt-rs-b5`) |
| seal-remake | 1 | 0 | 2 | guard `26a18191`; fps_matrix `7efe2a23` |
| katgpt-rs | 2 | 0 | 2 | `70eff640` |
| **ALL** | **40** | **1** | **40** | over 41 tracked scripts |

LIVE-FORWARD, PRECAUTIONARY, UNPARSED and REPLACED are all 0 workspace-wide.
Of the 41, **26 have errexit** and could launder an abort; **15 have nounset
without errexit** and their sentinels are precautionary (T10).

The single remainder, and the two that were remainders when this issue was
first written:

* **`riir-ai/scripts/e2e_internet.sh` — left EXPOSED, and now with a PROOF
  rather than a usability argument.** The usability half still holds (it is
  an interactive cluster runner that prints *"Press Ctrl+C to stop"* and ends
  in `wait`; Ctrl-C **is** its normal termination, so a sentinel would make
  the documented usage report failure, and nothing reads its exit code). But
  the stronger fact is T10's window column: its EXIT trap is registered on
  **line 41 of 43**, and the window — a blank line plus `wait` — contains
  **zero** abort triggers (no expansion, no `eval`, no call). There is no
  abort site after the handler exists, so it cannot launder. Verified
  independently here after `katgpt-rs-b5` reported it. A usability argument
  was doing load-bearing work it did not have to.
* **`riir-viewbridge/scripts/ci_feature_guard.sh`** — CLOSED by session
  `katgpt-rs-b5`, `21da73a` (4 arms verified). It was out of scope here only
  because that repo is not one of this session's working directories.
* **`seal-remake/scripts/fps_matrix.sh`** — CLOSED by session `katgpt-rs-b5`,
  `7efe2a23`. Deferred here because a second session was actively editing
  that worktree (confirmed: two `GUARD EXIT=0` lines appended to one shared
  log by two runs) — and that session turned out to be the one that took it.

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
- [x] T3 — riir-chain wave: **11/11 sentinelled**, riir-chain is now 0
      EXPOSED. Landed after re-checking the collision risk that deferred it:
      the other session's 15 dirty files are `crates/riir-chaind/*`,
      `src/consensus/*` and `scripts/test_gate.sh`, and **none of the 11
      intersects that set** — so the deferral's premise had expired and the
      wave was safe to land against named paths. `git -C ../riir-chain add
      <11 paths>`, never `-A`.

      Placement is per-script and a blanket "flag on the last line" would
      have been wrong in **four** of the eleven:
        - `action_slot_policy_gate.sh` has **TWO** success exits — a report
          mode (`[ "$GATE" -eq 0 ] && exit 0`) and the `--gate` verdict — and
          the flag must go before each, not once between them, or an abort in
          the verdict computation still launders.
        - `ci_local.sh` has a `--list` `exit 0` **after** the trap (its
          `--help`/bad-arg exits are before it and need nothing).
        - `program_rate_gate.sh`'s only trap lives **inside `--self-test`**
          and is explicitly deregistered with `trap - EXIT` — so the exposure
          window is registration..deregistration and the flag goes
          immediately before the `trap -`, not at EOF. Its normal mode never
          registers a trap and was never exposed.
        - `wallet_store_crash_atomicity.sh` ends in process **teardown**
          (`kill -9` + `wait`), and `wait` on a SIGKILLed child returns 137
          (measured) — putting an assignment after it would launder that 137
          into a 0. The flag goes before the teardown, and the script's
          non-zero exit is a pre-existing property, deliberately preserved.
          Worth knowing separately: it means this drill can never be wired
          into anything that reads an exit code.

      Verified against each script's **REAL** extracted handler — 33 arms,
      all correct: unbound abort → 1 with the new message (×11), deliberate
      failure → 1 with its own message and no extra output (×11), clean
      completion → 0 (×11). `bash -n` clean on all eleven. Five gates also
      run end-to-end unchanged (`block_pipeline_reachability`,
      `settlement_policy`, `action_slot_policy`, `money_bytes`,
      `program_rate` incl. `--self-test`; plus `ci_local --list`/`--help`) —
      every one still PASSES with its own message and zero sentinel output.

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

- [x] T9 — **the classifier's brace counter was wrong, and the wrong
      direction of that bug manufactures a FALSE SENTINELLED.** Landing T3
      left one script still EXPOSED —
      `riir-chain/scripts/block_pipeline_reachability_gate.sh` — with a
      sentinel visibly correct in the file. Cause: `function_bodies` counted
      `{`/`}` with `str.count`, and that file embeds a multi-line
      **single-quoted** `awk` program containing
      `mod[[:space:]]+tests[[:space:]]*\{` — one unmatched brace in **DATA**.
      The handler's body therefore never closed, ran to EOF, and
      `NONZERO_EXIT` was searched against a body the trap does not have.

      A false EXPOSED is the safe direction. The **same mis-parse the other
      way is not**: a runaway body swallows the rest of the file, and with it
      an unrelated `exit 1` and any late literal flag, which is exactly the
      false-SENTINELLED shape T6's canary already caught once. So two
      changes, not one:
        1. `scan_braces()` — quote-aware (single/double, carried **across**
           lines, backslash escapes) and heredoc-aware (a heredoc body is
           data). Two new selftest arms: the awk shape → SENTINELLED, and an
           unterminated body → UNPARSED-not-SENTINELLED (9/9).
        2. a new **UNPARSED** verdict for whatever it still cannot read,
           never pooled into either real column, reported by name, and a
           **hard failure** in `trap_sentinel_gate.py`. Gate canary driven
           three ways in-process: UNPARSED on a pinned script → exit 1,
           EXPOSED on a pinned script → exit 1, clean control → exit 0.

      Blast radius measured, not assumed: the fix reclassified **exactly one**
      file workspace-wide (27 + 11 = 38 SENTINELLED, arithmetically closed),
      so no real sentinel was being read off a mis-parsed body today.
      Docs gate 14/14 after.

- [x] T10 — **the premise named the wrong shell option, and the harness could
      not have caught it.** Reported by session `katgpt-rs-b5` and
      re-measured here independently before acting (`/bin/bash`
      3.2.57(1)-release, arm64-apple-darwin25), as a full (options x arm)
      matrix rather than the four hard-coded-`set -euo pipefail` rows the
      tool shipped with:

      | shell options | unbound expansion | `eval` syntax error |
      |---|---|---|
      | `set -u` (no `-e`) | aborts, **1** — NOT laundered | does **not abort at all** |
      | `set -e` (no `-u`) | no abort (expands empty) | 2 bare, **0** trapped ✗ |
      | `set -eu` / `-euo pipefail` | 1 bare, **0** trapped ✗ | 2 bare, **0** trapped ✗ |

      **errexit is the precondition, for both triggers.** nounset only
      supplies one extra trigger. The root cause is instrument design, not a
      typo: `_run()` prepended `set -euo pipefail` to every arm, so the
      premise measurement never varied the axis the *population predicate*
      was built on — it measured laundering and attributed it to `SET_U`.
      Four changes:
        1. `PRECAUTIONARY` — nounset without errexit + trap + no sentinel.
           Not laundering today; one added `-e` away from it. Its own column,
           never pooled. **15 of 41** rows are in this class, so the old
           report over-claimed on 37% of what it printed.
        2. the population predicate becomes the **union** `set -e OR set -u`.
           The `SET_U`-only predicate had a mirror blind spot — errexit +
           trap + `eval` and NO nounset launders via the syntax-error
           trigger, and `analyse()` returned None before looking. Measured
           with the widened predicate: **0** such scripts workspace-wide (the
           total stayed at 41), so the blind spot was empty — a measurement
           now, not an assumption.
        3. `_run()` writes a **temp script file**. ⛔ The measurement MODE is
           part of the claim: the nounset fatal error exits **127** from
           `bash -c` and **1** from a script file on this bash. The old
           harness never saw it because at `set -euo pipefail` both modes
           agree (1/0) — the divergence only exists in the configuration the
           old harness refused to measure. Caught by the new matrix printing
           a ⛔ DIVERGES cell against the documented table on its first run.
        4. an **exposure window** per finding — `[last trap registration,
           EOF)` — plus its abort-trigger count (`$VAR`/`eval`, with the body
           of any function the window CALLS folded in, which is what the
           line-based version under-reports). Nothing before the handler
           exists can be laundered by it, so a **zero-trigger window
           provably cannot launder**. A triage aid, not a verdict — it ORDERS
           the rows. Suggested by `katgpt-rs-b5`; two selftest arms pin it
           (empty window, and the callee fold-in).

      Selftest 13/13 (9 verdicts + 2 window + 2 population controls). The
      verdict gate takes PRECAUTIONARY as a **failure** deliberately, which
      is stricter than the severity: the report keeps the classes apart so it
      does not over-claim, the gate collapses them because it is a ratchet
      over two files and the fix is one line. Gate canary driven four ways
      in-process (EXPOSED / PRECAUTIONARY / UNPARSED on a pinned script → 1;
      clean → 0). Docs gate 14/14.

      ⛔ **A claim this issue can no longer make:** "measured on bash 3.2 and
      5.x". There is no bash 5 on this box (`/opt/homebrew/bin/bash`,
      `/usr/local/bin/bash`, Cellar — all absent), so the 5.x half is
      inherited, not measured. `measure_premise()` re-measures on whatever
      box runs it, which is the self-correcting version of the claim; the
      prose now says which bash produced these numbers.

- [x] T11 — **the 37 sentinels this issue landed were being held by nothing.**
      `trap_sentinel_gate.py` (T6) is katgpt-rs-scoped by construction — two
      names, one checkout — so every one of the other 39 sentinels across nine
      repos is a single line a future edit can drop with no gate objecting.
      That is exactly the shape of this issue's provenance: seal-remake's guard
      could not fail past layer 13 for months because nothing objected at the
      time.

      `scripts/trap_sentinel_drift_sweep.py` + `trap_sentinel_drift_floors.txt`
      — the fifth member of the documented workstation-only sweep family, and
      the same shape as `percentile_drift_sweep.py`. Every contract repo, on
      demand; NOT in docs_gate's CHECKS because CI's single checkout would
      derive an empty population and print a confident green over zero repos.
      Exit 0 clean / 1 on drift / **2 if the instrument is untrustworthy** — an
      unreliable instrument is not the same finding as drift.

      **No membership pin here, deliberately** — and the reasoning is the
      interesting part, because copying the stricter thing by reflex would have
      been wrong. `trap_sentinel_gate.py` and `cfg_gated_floor_gate.py` pin by
      NAME because a count is not a checksum over a set. Here the verdict is
      DERIVED from the file, so every way to lose a sentinel already lands in a
      gated class: drop the flag from an errexit script → EXPOSED; from a
      nounset-only one → PRECAUTIONARY; delete the script, or its
      `set -e`/`set -u`, or its EXIT trap → the population floor. A 40-name
      list would add nothing and would need re-typing on every rename.

      **Two floors, because one cannot do it.** `max_exposed = 0` is green over
      whatever the classifier can SEE, and `walk_sh` shells out to `git
      ls-files` — a regression there takes the population to 0 and every
      ceiling passes. `min_population` catches that where a population exists,
      but **six of the seventeen repos have a population of ZERO**, so their
      floor is 0 and detects nothing; `min_scripts` (the tracked-`*.sh` walk)
      still bites there and is the quantity a `git ls-files` regression
      actually moves. Both at ~60% of measured: slack against a legitimate
      consolidation, tight against blindness. katgpt-rs's `min_population`
      re-states the per-push gate's `POPULATION_FLOOR`, and the sweep
      **asserts** they agree rather than trusting it.

      Measured: 17 repos · 159 tracked `*.sh` · 41 in population (26 errexit) ·
      **1 EXPOSED and it is the proven-inert one** · 0 everything else.
      Canaried 10 ways in-process — clean control → 0; walk-floor breach,
      population-floor breach, ceiling breach on the real EXPOSED row,
      unpinned repo, pinned-but-absent repo, and pin-drift-vs-the-gate → 1;
      empty pin set, 3-field row, non-integer field → 2. ⛔ One arm was
      initially INERT (a whitespace mismatch made the plant a no-op, so it
      "passed" as a green 0); the harness asserts the plant applied now.

      Incidental: `trap_exit_launder_audit.repos` was a contract-repo
      population predicate that `population_sync_gate.py` did not know about.
      It is the **seventh** predicate now, and it agrees (17 repos, synthetic
      workspace + real cross-check).

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
