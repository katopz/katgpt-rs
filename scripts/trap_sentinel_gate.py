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

4. **Zero EXPOSED and zero LIVE-FORWARD.** A NEW script that arrives with a
   cleanup trap and no sentinel reds this gate on the commit that adds it —
   which is the whole point, since the seal-remake defect took months to
   surface precisely because nothing objected at the time.

A new script legitimately joining the population is a one-line pin update in
the same commit — the ratchet discipline layers elsewhere in this repo use.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# ── The pins (DATA) ───────────────────────────────────────────────────────
# Every tracked *.sh in THIS repo with `set -u` and an EXIT trap handler.
# Add a row when a new gate script legitimately joins, in the same commit.
PINNED_SENTINELLED = {
    "scripts/full_gate.sh",
    "scripts/proof_negative_test.sh",
}
POPULATION_FLOOR = 2  # a FLOOR: the classifier going blind must RED, not pass


def load_audit():
    path = os.path.join(HERE, "trap_exit_launder_audit.py")
    if not os.path.isfile(path):
        print(f"✗ trap sentinel gate FAILED — {path} is MISSING; the classifier this")
        print("  gate reads its verdicts from is gone, so a green here would mean nothing.")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("trap_exit_launder_audit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

    # ── 2/3/4. the population, by membership ──────────────────────────────
    found = {}
    for path in audit.walk_sh(REPO):
        r = audit.analyse(path)
        if r is not None:
            found[os.path.relpath(path, REPO)] = r["verdict"]

    problems = []
    if len(found) < POPULATION_FLOOR:
        problems.append(
            f"population is {len(found)}, floor is {POPULATION_FLOOR} — the classifier "
            f"went blind, or a gate script lost its `set -u`/EXIT trap. A shrinking "
            f"population is not the same as a fixed defect."
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
          f"(floor {POPULATION_FLOOR}), 0 EXPOSED, 0 LIVE-FORWARD, "
          f"{len(PINNED_SENTINELLED)} pinned name(s) still SENTINELLED{note}")


if __name__ == "__main__":
    main()
