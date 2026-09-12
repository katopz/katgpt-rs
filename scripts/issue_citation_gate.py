#!/usr/bin/env python3
"""GATE: a CROSS-REPO `Issue N` / `Plan N` / `Bench N` citation that names no repo.

`numbering_gate.py` protects a number from being allocated TWICE in one repo.
It structurally cannot see the failure one level up: a citation whose referent
lives in a DIFFERENT repo. `Issue 750` in this repo's prose meant
`riir-ai/.issues/750_behavior_first_quantization_promotion_gate.md`, and this
repo's own `.issues/.highwater` read **748** — two away. The moment anybody
follows the Numbering Discipline and allocates 749, then 750, that citation
stops dangling and starts resolving, silently, to the WRONG document. A
dangling reference is an inconvenience; a reference that rebinds to a real but
unrelated issue is a wrong answer delivered with a straight face.

Measured when this gate landed (2026-09-12): EIGHT unqualified rows over THREE
numbers — `Issue 513` (riir-train, x3), `Issue 750` (riir-ai, x3), `Issues
490/493` (riir-ai, x2). Every one resolved uniquely by TITLE match, and the
convention for writing them was already in the same file four lines away
("riir-ai `.issues/892`", "894 resolved same day in riir-ai").

⛔ **The AMBIGUOUS bucket is this gate's stated blind spot, not a clean pass.**
A number that exists BOTH locally and in a sibling is undecidable from the
number alone, and there were **35** such cited Issue numbers at landing. This
gate is green over them by construction: its predicate is "does NOT resolve
locally", so a bare `Issue 47` naming riir-ai's 47 reads as this repo's 47 and
always will. It is REPORTED every run and deliberately NOT gated: a
ceiling on it reds whenever somebody writes a perfectly correct citation to a
NEW local number that a sibling also happens to have — a nuisance red for a
right action, and a gate that reds on right actions is a gate that gets
bypassed (`staged_set_audit.py`'s rationale, one axis over). So it has the
standing of tail support in the percentile audit: a quantity that ORDERS the
rows and sizes the blind spot, never a verdict. The only real defence is the
writing convention — name the repo whenever the referent is not local,
ambiguous or not.

Exit 0 clean, 1 on an unqualified citation, **2 if the instrument is
untrustworthy** (a floor breached => the walk went blind; a scanned document
missing). An unreliable instrument is not the same finding as drift.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from numbering_drift_sweep import contract_repos  # noqa: E402 — the SEVENTH predicate, reused not re-derived

REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
FLOORS = HERE / "issue_citation_floors.txt"

# ── vocabulary: DATA, not derived ──────────────────────────────────────────
# Deriving both the scope and the population from one walk is what makes a gate
# permanently green. Only the population (which repos, which numbers) is
# derived; the citation vocabulary and the scanned documents are pinned.
KINDS = {
    "Issue": ".issues",
    "Plan": ".plans",
    "Research": ".research",
    "Bench": ".benchmarks",
    "Proposal": ".proposals",
}

# A plural kind word may head a LIST: "Issues 724, 725", "Issues 490/493",
# "Plans 596 and 597". Reading only the head number under-reports the class —
# `Issues 490/493` hid its second referent from the first version of this scan.
_HEAD = re.compile(r"\b(%s)(s?)\s+(\d{2,4})" % "|".join(KINDS))
# `.match(s, pos)` already anchors AT pos — Python `re` has no `\G`.
_TAIL = re.compile(r"\s*(?:,|/|and|&)\s*(\d{2,4})")

NUMBERED = re.compile(r"^(\d+)_")


def parse_pins(path: Path) -> dict[str, int | list[str]]:
    pins: dict[str, int | list[str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        pins[key] = [v for v in val.split() if v] if key == "documents" else int(val)
    return pins


def allocated(repo: Path, subdir: str) -> set[int]:
    """Every number EVER allocated under `repo/subdir` — worktree AND history.

    History is not optional: the noise-reduction rule REMOVES a resolved issue
    file, so a worktree-only walk reports a live citation as dangling. Issue
    750 is exactly that shape — resolved and removed in riir-ai `b559d2da3`.
    """
    out: set[int] = set()
    d = repo / subdir
    if d.is_dir():
        for f in d.iterdir():
            m = NUMBERED.match(f.name)
            if m:
                out.add(int(m.group(1)))
    log = subprocess.run(
        ["git", "-C", str(repo), "log", "--all", "--name-only", "--pretty=format:", "--", f"{subdir}/"],
        capture_output=True, text=True,
    )
    prefix = re.compile(re.escape(subdir) + r"/(\d+)_")
    for line in log.stdout.splitlines():
        m = prefix.match(line.strip())
        if m:
            out.add(int(m.group(1)))
    return out


def citations(text: str) -> list[tuple[int, str, int]]:
    """(line, kind, number) for every citation, list forms expanded."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines, 1):
        for m in _HEAD.finditer(line):
            kind = m.group(1)
            out.append((i, kind, int(m.group(3))))
            if not m.group(2):  # singular "Issue 47" never heads a list
                continue
            pos = m.end()
            while (t := _TAIL.match(line, pos)):
                out.append((i, kind, int(t.group(1))))
                pos = t.end()
    return out


def main() -> int:
    pins = parse_pins(FLOORS)
    docs = pins["documents"]
    assert isinstance(docs, list)

    repos = contract_repos(WORKSPACE)
    sibs = [r for r in repos if r.resolve() != REPO_ROOT]
    if len(repos) < pins["min_repos"]:
        print(f"✗ INSTRUMENT: derived {len(repos)} contract repos < floor {pins['min_repos']} — "
              f"the population went blind; every ceiling below would pass vacuously")
        return 2

    local = {k: allocated(REPO_ROOT, d) for k, d in KINDS.items()}
    elsewhere = {k: {} for k in KINDS}
    for r in sibs:
        for k, d in KINDS.items():
            for n in allocated(r, d):
                elsewhere[k].setdefault(n, []).append(r.name)

    scanned = 0
    unqualified: list[str] = []
    ambiguous: set[tuple[str, int]] = set()
    for doc in docs:
        p = REPO_ROOT / doc
        if not p.is_file():
            print(f"✗ INSTRUMENT: pinned document {doc} is missing — a gate cannot "
                  f"report clean over a file it never opened")
            return 2
        lines = p.read_text(encoding="utf-8").splitlines()
        for ln, kind, n in citations("\n".join(lines)):
            scanned += 1
            owners = elsewhere[kind].get(n, [])
            if n in local[kind]:
                if owners:
                    ambiguous.add((kind, n))
                continue
            # Cross-repo: a sibling repo name within the citation's own
            # paragraph-scale context (3 lines) is what makes it followable.
            ctx = "\n".join(lines[max(0, ln - 3):ln])
            if any(r.name in ctx for r in sibs):
                continue
            where = "/".join(owners) if owners else "NO REPO IN THE WORKSPACE"
            unqualified.append(f"  {doc}:{ln}  {kind} {n} — lives in {where}, "
                               f"names no repo\n      {lines[ln - 1].strip()[:120]}")

    if scanned < pins["min_citations_scanned"]:
        print(f"✗ INSTRUMENT: scanned {scanned} citations < floor {pins['min_citations_scanned']} — "
              f"the citation regex went blind, not the prose clean")
        return 2

    print(f"  scanned {scanned} citations in {len(docs)} document(s) over "
          f"{len(repos)} contract repos ({len(sibs)} siblings)")
    # REPORTED, never gated — see the module docstring. This is the size of
    # what the gate cannot decide, printed next to the verdict so a green is
    # never mistaken for a green over everything.
    print(f"  AMBIGUOUS (local AND sibling — undecidable by number, NOT a pass): {len(ambiguous)}")

    if unqualified:
        print(f"✗ issue citation gate FAILED — {len(unqualified)} unqualified cross-repo citation(s)")
        for row in unqualified:
            print(row)
        print("  Fix: name the owning repo in the prose — `riir-ai Issue 750`, "
              "`riir-train Issue 513`. The number alone is not an address.")
        return 1

    print(f"✓ issue citation gate PASSED — every cross-repo citation names its repo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
