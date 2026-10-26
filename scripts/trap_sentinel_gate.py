#!/usr/bin/env python3
"""Gate: every katgpt-rs shell gate that CAN launder its exit status must not.

The verdict half of `trap_exit_launder_audit.py` (Issue 734 T6), scoped to
this repo. The audit is a report because its top verdict is latent across the
workspace; here the population is two scripts, both repaired, and keeping
them repaired is cheap to assert.

What this gate pins, and why each arm exists:

1. **The detector is not inert.** `selftest()` from the audit must pass
   first. A gate whose classifier silently stopped classifying reports a
   green zero over everything — the failure this repo keeps re-finding
   (`a-green-canary-may-be-inert`). Run the canary before reading the count.

2. **The population is FLOORED, not just checked.** If the classifier breaks
   and derives an empty population, "0 EXPOSED" is a pass. A floor makes the
   instrument going blind a failure instead.

3. **Membership, not cardinality.** The pinned set is the two scripts by
   NAME. A count is not a checksum over a set: dropping `full_gate.sh` from
   the population (someone removes its trap, or the `set -u`) while a new
   script joins keeps the total at 2 and a count-only pin stays green.

4. **Zero UNPARSED.** UNPARSED means the classifier could not read the
   handler body at all (it never closed under brace counting). That is the
   instrument failing, and it is NOT the safe direction: a runaway body
   swallows unrelated `exit 1`s and a late literal flag and can read as a
   false SENTINELLED, which HIDES exposure. It reds here rather than being
   pooled into either column.

5. **Zero PRECAUTIONARY, too — deliberately stricter than the severity.**
   PRECAUTIONARY is nounset WITHOUT errexit: measured, those aborts exit 1
   today, so such a script cannot launder anything and the finding is not
   live. It still reds here, because the distance between PRECAUTIONARY and
   EXPOSED is one character in a `set` line that nobody re-audits when they
   add it, and the fix is one line. The report keeps the two apart so it does
   not over-claim; this gate collapses them because it is a ratchet over two
   files, not a workspace census.

6. **Zero EXPOSED and zero LIVE-FORWARD.** A NEW script that arrives with a
   cleanup trap and no sentinel reds this gate on the commit that adds it —
   which is the whole point, since the mmorpg-remake defect took months to
   surface precisely because nothing objected at the time.

A new script legitimately joining the population is a one-line pin update in
the same commit — the ratchet discipline layers elsewhere in this repo use.
"""

from __future__ import annotations

import importlib.util
import os
import sys

# Issue 804: this instrument is documented as directly invokable, and its
# verdict glyphs (✓ ✗ ⛔ ⚠) kill it on a non-UTF-8 console — no verdict at
# all, findings unread. docs_gate.sh's PYTHONIOENCODING only covers runs
# that go through the wrapper.
import console_safe  # noqa: E402

console_safe.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# ── The pins (DATA) ───────────────────────────────────────────────────────
# Every tracked *.sh in THIS repo with `set -u` and an EXIT trap handler.
# Add a row when a new gate script legitimately joins, in the same commit.
PINNED_SENTINELLED = {
    "scripts/full_gate.sh",
    "scripts/proof_negative_test.sh",
    # Issue 806 — the x86_64 execution matrix. Joins for the same reason the
    # other two are here: it is a gate whose abort must not read as a pass, and
    # it runs cargo for tens of minutes, which is exactly the window in which a
    # SIGTERM or a `set -e` trip lands.
    "scripts/x86_64_execution_matrix.sh",
}
POPULATION_FLOOR = 2  # a FLOOR: the classifier going blind must RED, not pass


def load_audit(path: str | None = None):
    """Import the classifier, or REFUSE — never fall back to a local copy.

    `path` is a parameter ONLY so the refusal is armable: Issue 790 measured
    the `not os.path.isfile` guard surviving every mutation, because the
    probed path was bound to `HERE` and no arm could make it absent without
    deleting the classifier from the working tree.

    ⚠ It is safe to parameterise here and NOT in
    `platform_dead_code_floor_gate.load_audit`, which looks identical. That
    one probes a path and then imports by NAME through `sys.path` (on purpose
    — a dataclass-bearing module must be registered in `sys.modules`), so
    parameterising its probe would assert a refusal for a path it does not
    then load. Here the probed path IS the one `spec_from_file_location`
    receives, so the arm tests the real thing.
    """
    path = path or os.path.join(HERE, "trap_exit_launder_audit.py")
    if not os.path.isfile(path):
        print(f"✗ trap sentinel gate FAILED — {path} is MISSING; the classifier this")
        print("  gate reads its verdicts from is gone, so a green here would mean nothing.")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("trap_exit_launder_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verdict_problems(found: dict, audit) -> tuple[list[str], list[str]]:
    """(problems, sentinelled-but-unpinned) for a measured population.

    Issue 790 T4. `audit.selftest()` runs first on every invocation and it is
    the CLASSIFIER's arm — it cannot reach a line of this file, which is Issue
    775's sentence and why `arm_reach_audit` reported all 8 of this module's
    mutants as NO-ARM. Every line below decides a verdict from verdicts: the
    population FLOOR (a shrinking population is not a fixed defect), four
    bucket filters that must not be pooled, and a membership pin that has to
    red in the missing direction while merely NOTING the extra one.

    `audit` is passed rather than imported so an arm can hand it a stub: the
    verdicts are plain strings and the classifier's own selftest already owns
    the question of whether it produces the right ones.
    """
    problems = []
    if len(found) < POPULATION_FLOOR:
        problems.append(
            f"population is {len(found)}, floor is {POPULATION_FLOOR} — the classifier "
            f"went blind, or a gate script lost its `set -u`/EXIT trap. A shrinking "
            f"population is not the same as a fixed defect."
        )

    unreadable = sorted(k for k, v in found.items() if v == audit.UNPARSED)
    if unreadable:
        problems.append(
            "UNPARSED (the classifier could not read the handler body — its braces "
            "never closed, so BOTH verdicts would be guesses, and the runaway-body "
            "direction manufactures a false SENTINELLED): " + ", ".join(unreadable)
        )

    precautionary = sorted(k for k, v in found.items() if v == audit.PRECAUTIONARY)
    if precautionary:
        problems.append(
            "PRECAUTIONARY (nounset without errexit and no sentinel — not laundering "
            "TODAY, but one added `-e` away from it, and nobody re-audits a `set` "
            "line): " + ", ".join(precautionary)
        )

    exposed = sorted(k for k, v in found.items() if v == audit.EXPOSED)
    forward = sorted(k for k, v in found.items() if v == audit.LIVE_FORWARD)
    if forward:
        problems.append(
            "LIVE-FORWARD (a double-quoted trap naming a later-assigned variable — "
            "this aborts EVERY run and, absent a sentinel, reports a pass): "
            + ", ".join(forward)
        )
    if exposed:
        problems.append(
            "EXPOSED (no completion sentinel — a `set -u` abort or an `eval` syntax "
            "error anywhere in these reports exit 0): " + ", ".join(exposed)
            + ".  Fix: see the pattern in scripts/full_gate.sh (full_gate_cleanup)."
        )

    sentinelled = {k for k, v in found.items() if v == audit.SENTINELLED}
    missing = sorted(PINNED_SENTINELLED - sentinelled)
    if missing:
        problems.append(
            "pinned script(s) are no longer SENTINELLED — either the sentinel was "
            "removed, or the script left the population (its `set -u` or its EXIT "
            "trap went away, which is also worth knowing): " + ", ".join(missing)
        )
    extra = sorted(sentinelled - PINNED_SENTINELLED)

    return problems, extra


class _StubVerdicts:
    """The classifier's verdict vocabulary, as plain strings.

    `verdict_problems` only ever compares `found`'s values against these, so an
    arm can supply them directly and never build a shell script. Keeping the
    names identical to `trap_exit_launder_audit`'s is asserted below.
    """
    UNPARSED = "UNPARSED"
    PRECAUTIONARY = "PRECAUTIONARY"
    EXPOSED = "EXPOSED"
    LIVE_FORWARD = "LIVE-FORWARD"
    SENTINELLED = "SENTINELLED"


def gate_selftest(audit=None) -> list[str]:
    """Arms over THIS file's verdict arithmetic, which `audit.selftest` cannot reach."""
    fails: list[str] = []
    v = _StubVerdicts

    def eq(label, got, want):
        if got != want:
            fails.append(f"    {label}: got {got!r}, want {want!r}")

    # ── the classifier-absence refusal (Issue 790 T2). A green printed over a
    # MISSING classifier is the one outcome this gate must never produce, and
    # nothing reached the guard until `load_audit` took a path.
    # Its own `✗` lines are SWALLOWED and merely asserted to have been said:
    # a passing gate that prints a failure message is a gate people learn to
    # read past.
    import contextlib
    import io
    said = io.StringIO()
    try:
        with contextlib.redirect_stdout(said):
            load_audit(os.path.join(HERE, "no_such_classifier_xyz.py"))
        fails.append("    load_audit: a MISSING classifier did not refuse — "
                     "this gate would print a verdict over an absence")
    except SystemExit:
        if "MISSING" not in said.getvalue():
            fails.append("    load_audit: refused without saying the "
                         "classifier is MISSING — the reader cannot tell a "
                         "missing classifier from a real finding")
    # …and the present path still loads, or the arm above is satisfied by a
    # function that refuses unconditionally.
    if not hasattr(load_audit(), "analyse"):
        fails.append("    load_audit: the real classifier did not load, so "
                     "the refusal arm above proves nothing")

    # The stub must speak the classifier's actual vocabulary, or every filter
    # below compares against a string the real population never contains and
    # all four arms pass over an empty match.
    if audit is not None:
        for name in ("UNPARSED", "PRECAUTIONARY", "EXPOSED", "LIVE_FORWARD",
                     "SENTINELLED"):
            if getattr(audit, name, object()) != getattr(v, name):
                fails.append(
                    f"    the stub's {name} is {getattr(v, name)!r} but the "
                    f"classifier's is {getattr(audit, name, None)!r} — every "
                    f"bucket filter would match nothing and pass vacuously")

    clean = {p: v.SENTINELLED for p in PINNED_SENTINELLED}
    eq("the live shape passes", verdict_problems(clean, v), ([], []))

    # The population FLOOR: a shrinking population is not a fixed defect.
    one = dict(list(clean.items())[:1])
    probs, _ = verdict_problems(one, v)
    eq("a population under the floor is a problem",
       any("floor" in p for p in probs), True)
    eq("…and it also reds the missing pin, not only the floor",
       any("no longer SENTINELLED" in p for p in probs), True)

    # The boundary cell the `< -> <=` mutant asks for: a population EXACTLY
    # at the floor is NOT a problem — the floor fires DOWNWARD only (the
    # same two-sided cell global_rng_gate.floor_arms runs).
    at_floor = dict(list(clean.items())[:POPULATION_FLOOR])
    probs, _ = verdict_problems(at_floor, v)
    eq("a population exactly at the floor is clean",
       any("floor" in p for p in probs), False)

    # Each bucket, one at a time. None may be pooled: UNPARSED hides exposure
    # (a runaway body reads as a false SENTINELLED) and PRECAUTIONARY is one
    # added `-e` from laundering.
    for verdict, needle in ((v.UNPARSED, "UNPARSED"),
                            (v.PRECAUTIONARY, "PRECAUTIONARY"),
                            (v.EXPOSED, "EXPOSED"),
                            (v.LIVE_FORWARD, "LIVE-FORWARD")):
        found = {**clean, "scripts/new.sh": verdict}
        probs, _ = verdict_problems(found, v)
        eq(f"a {needle} script is a problem",
           any(needle in p for p in probs), True)
        eq(f"a {needle} script does not disturb the pins",
           any("no longer SENTINELLED" in p for p in probs), False)

    # The membership pin, both directions — and they are NOT symmetric.
    unpinned = {**clean, "scripts/new.sh": v.SENTINELLED}
    probs, extra = verdict_problems(unpinned, v)
    eq("a NEW sentinelled script is not a failure", probs, [])
    eq("…it is NOTED so the pin gets updated", extra, ["scripts/new.sh"])
    dropped = {**clean}
    dropped[sorted(PINNED_SENTINELLED)[0]] = v.EXPOSED
    probs, _ = verdict_problems(dropped, v)
    eq("a pinned script losing its sentinel IS a failure",
       any("no longer SENTINELLED" in p for p in probs), True)
    return fails


def main():
    audit = load_audit()

    # ── 1. the canary, before any count is read ───────────────────────────
    failures = audit.selftest()
    if failures:
        print("✗ trap sentinel gate FAILED — the classifier's own selftest does not pass,")
        print("  so every verdict below is unreadable (an inert detector reports a green zero):")
        for f in failures:
            print(f)
        sys.exit(1)

    # ── 1b. THIS file's own verdict arithmetic, which the classifier's
    #        selftest cannot reach (Issue 775's sentence, Issue 790 T4) ─────
    arm_failures = gate_selftest(audit)
    if arm_failures:
        print("✗ trap sentinel gate FAILED — this gate's own verdict arithmetic does")
        print("  not pass its arms, so the population verdict below is unreadable:")
        for f in arm_failures:
            print(f)
        sys.exit(1)

    # ── 2/3/4. the population, by membership ──────────────────────────────
    found = {}
    for path in audit.walk_sh(REPO):
        r = audit.analyse(path)
        if r is not None:
            # .replace(os.sep, "/"): the membership pins below are committed
            # forward-slash paths; os.path.relpath emits backslashes on
            # Windows and every pin compare fails BOTH directions (the
            # "no longer SENTINELLED" false red) — found running the docs
            # gate on the 4090 box, 2026-09-12.
            found[os.path.relpath(path, REPO).replace(os.sep, "/")] = r["verdict"]

    problems, extra = verdict_problems(found, audit)

    if problems:
        print("✗ trap sentinel gate FAILED")
        for p in problems:
            print(f"    {p}")
        print(f"    measured population ({len(found)}): "
              + ", ".join(f"{k}={v}" for k, v in sorted(found.items())))
        sys.exit(1)

    note = ""
    if extra:
        # Not a failure — a new script arrived already carrying a sentinel.
        # Say so, so the pin gets updated rather than drifting silently.
        note = (f"; {len(extra)} sentinelled script(s) not yet pinned "
                f"({', '.join(extra)}) — add them to PINNED_SENTINELLED")
    print(f"✓ trap sentinel gate PASSED — {len(found)} script(s) in population "
          f"(floor {POPULATION_FLOOR}), 0 EXPOSED, 0 PRECAUTIONARY, "
          f"0 LIVE-FORWARD, 0 UNPARSED, "
          f"{len(PINNED_SENTINELLED)} pinned name(s) still SENTINELLED{note}")


if __name__ == "__main__":
    main()
