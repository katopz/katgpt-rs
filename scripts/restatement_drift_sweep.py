#!/usr/bin/env python3
"""Hold the restatement-theorem zero across EVERY repo that ships Lean proofs.

`restatement_theorem_audit.py` is a REPORT: it classifies and prints, and a
report holds nothing. The zero it measured on 2026-09-12 (0
RESTATEMENT-INLINE over 4 repos / 68 `.lean` / 255 theorems) was bought by
riir-neuron-db Issue 617 — four theorems removed after four independent probes
each — and until this file existed, nothing would have objected when the fifth
one landed.

This is the sixth instance of one shape in this workspace, and the first two
found real defects the moment they were pointed anywhere but at katgpt-rs:

    Issue 702  ci_gate_coverage              one repo -> 7 dead workflows
    Issue 725  numbering_drift_sweep         one repo -> 35 duplicate numbers
    2026-09-06 required_features_drift_sweep one repo -> clean, and pinned there
    2026-09-06 percentile_drift_sweep        one repo -> clean, and pinned there
    2026-09-07 trap_sentinel_drift_sweep     one repo -> 1 finding, pinned
    this file  restatement_drift_sweep       one repo -> 0, and pinned in four

**Not in `docs_gate.sh`'s CHECKS, deliberately** — same reason as every other
sweep in the family: CI has a single checkout, so the derived population would
be one repo and the sweep would print a confident green over the three that
carry the actual class (riir-neuron-db has 44 composite defs, riir-chain 32,
katgpt-rs **1**). A per-push gate that can only see the repo where the class
barely exists is worse than no gate, because it reads like coverage.

# Two floors, and the second is the one that bites

`max_restatement = 0` is green over whatever the classifier can SEE, and this
classifier has two independent ways to go blind:

  * the WALK — `.proofs` moves, or a repo's proofs are vendored under a name
    the walk skips: file count -> 0, every ceiling passes. `min_lean_files`.
  * the PARSE — a tokenizer or block-splitter regression yields zero theorems
    from a perfectly good tree. The walk is unchanged and only `min_theorems`
    moves. This is not hypothetical: while it was being written, this pass
    went from 0 defs (a `:=` tokenized as `:` + `=`) to 4 findings, with the
    walk identical in both runs.

Both floors sit at ~60% of measured: SLACK against churn (consolidating spec
modules legitimately shrinks a count), TIGHT against blindness (a walk or
tokenizer regression drops these by an order of magnitude, not by a third).

# A ceiling nobody has watched fail is a ceiling of unknown width

`--prove-fires` plants a synthetic restatement into a COPY of each repo's
`.proofs` and requires the verdict to RED, end to end: walk -> parse ->
classify -> compare. It is not a mock — it runs the same `audit_repo` over a
real tree. The historical real-subject probe is recorded in the audit's
docstring: riir-neuron-db at `24957a2^` reports exactly the four theorems
Issue 617 removed, and 0 at HEAD.
"""

import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import restatement_theorem_audit as audit  # noqa: E402

FLOORS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "restatement_drift_floors.txt")

CANARY = """import NeuronDbProof.Shard.Layout

namespace RestatementCanary

def alphaSize : Nat := 16
def betaSize : Nat := 24
def totalSize : Nat := alphaSize + betaSize

theorem totalSize_eq_sum : totalSize = alphaSize + betaSize := by decide

end RestatementCanary
"""


def read_floors():
    rows = {}
    with open(FLOORS, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                sys.exit(f"✗ malformed floors row (want 5 fields): {raw.strip()}")
            repo, a, b, c, d = parts
            rows[repo] = dict(min_lean_files=int(a), min_theorems=int(b),
                              max_restatement=int(c), max_identity=int(d))
    if not rows:
        sys.exit("✗ floors file parsed to ZERO rows — a parser that returns "
                 "nothing for both 'absent' and 'malformed' disarms every "
                 "ceiling below it")
    return rows


def measure(repo_path):
    r = audit.audit_repo(repo_path)
    return {
        "files": r["files"],
        "theorems": r["theorems"],
        "restatement": r["counts"][audit.RESTATEMENT],
        "identity": r["counts"][audit.IDENTITY],
        "rows": [x for x in r["rows"]
                 if x[0] in (audit.RESTATEMENT, audit.IDENTITY)],
    }


def prove_fires(root, present):
    """Plant a restatement in a COPY of each repo and require a RED."""
    print("── --prove-fires: a planted restatement must RED every row ──")
    bad = 0
    for repo in present:
        src = os.path.join(root, repo, ".proofs")
        tmp = tempfile.mkdtemp(prefix=f"restatement-probe-{repo}-")
        try:
            dst = os.path.join(tmp, ".proofs")
            shutil.copytree(src, dst,
                            ignore=shutil.ignore_patterns(".lake", "build"))
            before = measure(tmp)["restatement"]
            with open(os.path.join(dst, "RestatementCanary.lean"), "w",
                      encoding="utf-8") as fh:
                fh.write(CANARY)
            after = measure(tmp)["restatement"]
            ok = after == before + 1
            print(f"   {'✓' if ok else '✗'} {repo}: {before} -> {after} "
                  f"with one planted restatement")
            if not ok:
                bad += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if bad:
        print(f"✗ the ceiling is INERT in {bad} repo(s) — a planted "
              f"restatement did not move the count")
        return 1
    print("✓ every row's ceiling is armed end to end (walk → parse → "
          "classify → compare)\n")
    return 0


def main():
    audit.selftest()
    audit.selftest_scoping()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.dirname(here)
    floors = read_floors()
    present = audit.repos(root)

    print("=== restatement-theorem drift sweep (every repo with .proofs) ===\n")

    fail = 0
    # A repo in the floors file that is not on disk is UNSEEN, never a pass:
    # the sweep would otherwise silently shrink to whatever is checked out.
    missing = sorted(set(floors) - set(present))
    if missing:
        print(f"⛔ UNSEEN (pinned but absent — never a pass): "
              f"{', '.join(missing)}")
        fail += len(missing)
    unpinned = sorted(set(present) - set(floors))
    if unpinned:
        print(f"⛔ UNPINNED (a repo joined the population): "
              f"{', '.join(unpinned)} — re-pin deliberately")
        fail += len(unpinned)

    for repo in present:
        if repo not in floors:
            continue
        f = floors[repo]
        m = measure(os.path.join(root, repo))
        bad = []
        if m["files"] < f["min_lean_files"]:
            bad.append(f"walk {m['files']} < floor {f['min_lean_files']}")
        if m["theorems"] < f["min_theorems"]:
            bad.append(f"parsed {m['theorems']} < floor {f['min_theorems']}")
        if m["restatement"] > f["max_restatement"]:
            bad.append(f"RESTATEMENT {m['restatement']} > "
                       f"ceiling {f['max_restatement']}")
        if m["identity"] > f["max_identity"]:
            bad.append(f"IDENTITY {m['identity']} > ceiling {f['max_identity']}")
        mark = "✗" if bad else "✓"
        print(f"{mark} {repo:<16} {m['files']:>3} .lean · "
              f"{m['theorems']:>3} theorems · "
              f"RESTATEMENT {m['restatement']} · IDENTITY {m['identity']}")
        for b in bad:
            print(f"     ⛔ {b}")
        for bucket, path, name, note in m["rows"]:
            print(f"     {bucket}  "
                  f"{os.path.relpath(path, os.path.join(root, repo))} :: "
                  f"{name}  — {note}")
        fail += len(bad)

    print()
    if "--prove-fires" in sys.argv:
        fail += prove_fires(root, present)

    if fail:
        print(f"✗ restatement drift sweep FAILED — {fail} breach(es)")
        print("  Do NOT raise a ceiling to clear a red: a new "
              "RESTATEMENT-INLINE row is a theorem that cannot fail on any "
              "constant being wrong. Remove it (riir-neuron-db Issue 617 is "
              "the worked precedent) or, if it pins two independently "
              "maintained definitions, it should be reading CROSS-DEF —"
              " check which side is inline.")
        return 1
    print(f"✓ restatement drift sweep PASSED — {len(present)} repo(s) "
          f"(derived: BOUNDARY.md + .git + .proofs), all pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
