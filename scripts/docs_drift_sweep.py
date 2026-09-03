#!/usr/bin/env python3
"""Run both doc-drift auditors over EVERY contract repo, not just this one.

Issue 702's finding, in one sentence: the auditors accept a repo path and audit
any repo, and for months nothing pointed them at anything but katgpt-rs. A
sibling with stale labels is indistinguishable from a sibling nobody looked at.
This script is the workstation instrument that looks at all of them at once —
the answer had been recomputed by hand three times before it was written down.

Why this is NOT in scripts/docs_gate.sh's CHECKS array
------------------------------------------------------
docs_gate.yml runs on ubuntu-latest with a single checkout. The siblings are
private and simply are not there, so this check would either fail on every CI
run or — far worse — derive an empty population and print a confident green
over zero repos. Cadence is deliberately split:

  this script            workstation, on demand, every contract repo (derived)
  docs_gate.yml          CI, per-push, katgpt-rs only
  sibling_docs_drift.yml CI, reusable, one sibling per caller

Vocabulary vs population (the trap this script is built around)
---------------------------------------------------------------
The population — which repos exist — is DERIVED (BOUNDARY.md + a .git dir),
never typed; a hand-typed repo set is Issue 703. But deriving the *expectations*
from the same walk would make the gate permanently green: a repo that vanished,
or an auditor that went blind to a whole dialect, both just shrink the walk and
still report "0 mismatches". So the expectations are COMMITTED, in
scripts/docs_drift_floors.txt, and this script fails on a repo that is in the
floor file and absent or under-count in the walk.

The floors are PRESENCE, not the observed count, and the reasoning for that
lives in docs_drift_floors.txt's header — in short: exact counts would duplicate
bench_doc_audit.py's selftest() (a stronger, already-canaried blindness
detector) while redding on every legitimate doc removal. The assertion this
sweep is uniquely able to make is "a repo known to bear labels was actually
SEEN", which selftest() cannot make because it never leaves this repo.

Floors are a MINIMUM, so adding docs never reds this gate. If a repo
legitimately stops carrying labels, drop its row in the same commit — the
failure message says so, because a gate whose red is ambiguous gets ignored.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT.parent
FLOORS_FILE = Path(__file__).resolve().parent / "docs_drift_floors.txt"

# (script, regex capturing the per-repo count, human name of the unit)
AUDITORS = [
    ("bench_doc_audit.py", re.compile(r"checked (\d+) labels?, (\d+) mismatch"), "labels"),
    ("cargo_comment_audit.py", re.compile(r"checked (\d+) inline comments?, (\d+) mismatch"), "comments"),
]
AUDITING_RE = re.compile(r"^=== Auditing (.+?) ===")


def derive_population() -> list[Path]:
    """Every sibling carrying a BOUNDARY.md contract. Derived, never typed."""
    return sorted(
        (d for d in WORKSPACE.iterdir()
         if d.is_dir() and (d / "BOUNDARY.md").is_file() and (d / ".git").is_dir()),
        key=lambda p: p.name,
    )


def read_floors() -> dict[str, tuple[int, int]]:
    """Committed expectations: repo -> (min bench labels, min cargo comments)."""
    floors: dict[str, tuple[int, int]] = {}
    if not FLOORS_FILE.is_file():
        return floors
    for raw in FLOORS_FILE.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, bench, cargo = line.split()
        floors[name] = (int(bench), int(cargo))
    return floors


def run_auditor(script: str, pattern: re.Pattern[str], repos: list[Path]) -> dict[str, tuple[int, int]]:
    """One invocation over every repo — both auditors accept N paths."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *(str(p) for p in repos)],
        capture_output=True, text=True,
    )
    if proc.returncode not in (0, 1):
        # A crash is not a clean sweep. Surface it rather than parsing partial
        # output, which would silently under-report (Issue 702's whole theme).
        raise SystemExit(f"✗ {script} crashed (exit {proc.returncode})\n{proc.stderr}")
    counts: dict[str, tuple[int, int]] = {}
    current: str | None = None
    for line in proc.stdout.splitlines():
        header = AUDITING_RE.match(line)
        if header:
            current = header.group(1)
            continue
        hit = pattern.search(line)
        if hit and current:
            counts[current] = (int(hit.group(1)), int(hit.group(2)))
    return counts


def main() -> int:
    repos = derive_population()
    floors = read_floors()
    present = {p.name for p in repos}

    print(f"▸ derived population: {len(repos)} contract repos under {WORKSPACE}")
    print(f"▸ committed floors:   {len(floors)} label-bearing repos "
          f"({FLOORS_FILE.name})\n")

    results = {script: run_auditor(script, pat, repos) for script, pat, _ in AUDITORS}

    failures: list[str] = []
    notes: list[str] = []

    # A repo in the committed vocabulary that the walk cannot see is a coverage
    # hole, not a pass. Never let a shrinking population read as clean.
    for name in sorted(set(floors) - present):
        failures.append(
            f"{name}: in {FLOORS_FILE.name} but not in the derived population — "
            f"the sweep did NOT cover it (repo missing, or BOUNDARY.md/.git gone)")

    header = f"{'repo':<24}{'labels':>8}{'mism':>6}{'comments':>10}{'mism':>6}   floor"
    print(header)
    print("-" * len(header))
    for repo in repos:
        n = repo.name
        b_lab, b_mis = results["bench_doc_audit.py"].get(n, (0, 0))
        c_lab, c_mis = results["cargo_comment_audit.py"].get(n, (0, 0))
        floor = floors.get(n)
        if floor is None:
            mark = "" if (b_lab or c_lab) else "-"
            if b_lab or c_lab:
                # Newly label-bearing: a floor should be recorded so a future
                # regression back to zero is catchable. Advisory, not fatal.
                mark = "NEW"
                notes.append(
                    f"{n}: now carries {b_lab} labels / {c_lab} comments but has no "
                    f"floor — add `{n}\t{b_lab}\t{c_lab}` to {FLOORS_FILE.name}")
        else:
            fb, fc = floor
            mark = f"{fb}/{fc}"
            if b_lab < fb:
                failures.append(
                    f"{n}: {b_lab} bench labels < floor {fb} — either the auditor "
                    f"went blind to a dialect, or docs were removed (then LOWER the floor)")
            if c_lab < fc:
                failures.append(
                    f"{n}: {c_lab} Cargo comments < floor {fc} — either the auditor "
                    f"went blind, or comments were removed (then LOWER the floor)")
        if b_mis:
            failures.append(f"{n}: {b_mis} bench-doc label mismatch(es) vs the manifests")
        if c_mis:
            failures.append(f"{n}: {c_mis} Cargo-comment mismatch(es) vs the manifests")
        print(f"{n:<24}{b_lab:>8}{b_mis:>6}{c_lab:>10}{c_mis:>6}   {mark}")

    # Liveness sentinel. katgpt-rs is the one repo guaranteed present (it holds
    # this script), so a zero here means the auditor is broken or the walk is
    # pointed at nothing — never that the corpus is clean.
    if not results["bench_doc_audit.py"].get(REPO_ROOT.name, (0, 0))[0]:
        failures.append(
            f"{REPO_ROOT.name}: 0 labels from its OWN corpus — the auditor is dead, "
            "not the docs clean")

    print()
    for note in notes:
        print(f"  ! {note}")
    if failures:
        print(f"\n✗ docs-drift sweep FAILED — {len(failures)} finding(s):")
        for f in failures:
            print(f"    - {f}")
        return 1
    covered = sum(1 for r in repos if r.name in floors)
    print(f"\n✓ docs-drift sweep PASSED — {len(repos)} repos, 0 mismatches, "
          f"{covered}/{len(floors)} floor-bearing repos covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
