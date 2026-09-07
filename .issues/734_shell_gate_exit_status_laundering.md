# Issue 734 — a shell gate that ABORTS mid-run reports exit 0

**Status:** OPEN 2026-09-07 — T0/T1/T2/T7 done (instrument landed, katgpt-rs's
own two scripts repaired, shellcheck ruled out as a detector); T3/T4/T5 are 39
EXPOSED scripts across 9 sibling repos, one wave per repo; T6 is the verdict-half
call. Found by repairing seal-remake `26a18191`.

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

**A tightening that mattered:** the first classifier called any late-assigned,
handler-read variable a sentinel and so reported 3 SENTINELLED — two of them
false (`BOOTSTRAP_PID=$!`, `status=$?`). A false SENTINELLED *hides* exposure,
which is the "a zero ceiling is only as wide as its classifier" failure inside
the tool built to find it. A completion flag is assigned only **literal**
constants; that is now the discriminator, and the selftest still fires on a
real sentinel, so the tightening did not blind it.

## Measured population (2026-09-07, tracked files only)

| repo | EXPOSED | SENTINELLED | REPLACED |
|---|---|---|---|
| katgpt-rs | 0 (was 2) | 2 | 0 |
| riir-chain | 11 | 0 | 1 |
| riir-train | 8 | 0 | 0 |
| riir-mmorpg-examples | 6 | 0 | 0 |
| riir-ai | 5 | 0 | 0 |
| riir-dapps | 2 | 0 | 0 |
| riir-clippy | 2 | 0 | 0 |
| riir-auth | 1 | 0 | 0 |
| riir-deployer | 1 | 0 | 0 |
| riir-viewbridge | 1 | 0 | 0 |
| seal-remake | 0 | 1 | 0 |
| **ALL** | **39** | **3** | **1** |

EXPOSED is **latent** — it needs an abort to bite, and most of these scripts
have none today. It is still exactly the state seal-remake's gate was in
before its abort arrived.

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
- [ ] T3 — riir-chain wave: 11 EXPOSED + the one REPLACED
      (`scripts/program_rate_gate.sh`, EXIT traps at 447 and 510 — the
      second silently discards the first, so the line-447 cleanup never runs).
- [ ] T4 — riir-train wave (8), riir-mmorpg-examples wave (6), riir-ai wave (5).
- [ ] T5 — the singles: riir-dapps (2), riir-clippy (2), riir-auth,
      riir-deployer, riir-viewbridge.
- [ ] T6 — decide the verdict half. A per-repo gate over LIVE-FORWARD is
      cheap and can never false-positive (it is a provable abort). A gate
      over EXPOSED would red 39 scripts on day one and is a ratchet, not a
      gate — pin a floor per repo as each wave lands, or skip it.
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
