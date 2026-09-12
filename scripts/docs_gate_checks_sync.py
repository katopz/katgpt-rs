#!/usr/bin/env python3
"""GATE: docs_gate.sh's CHECKS array vs the AGENTS.md table that documents it.

The CHECKS array and AGENTS.md's "one line per check" table are hand-duplicated
lists of the same thing, and NOTHING compared them. That is precisely the shape
`docs_gate_paths_sync.py` already gates one axis over (docs_gate.yml's two
hand-duplicated trigger `paths:` lists) — the second instance of a class is
where you stop calling it a one-off.

Found because it had already drifted: docs_gate.sh described
`population_sync_gate.py` as "the **six** independent contract-repo
predicates" while the script's own docstring and AGENTS.md's table both said
**seven** (Issue 734 added the seventh). Six months of a wrong number sitting
in the output a human reads on every run.

TWO assertions, and they are deliberately different in strictness:

1. **MEMBERSHIP** of script names, both directions. Not cardinality — a count
   that MATCHES is not a checksum over a set, and the workspace has been
   burned by exactly that. A check registered but undocumented is invisible to
   whoever reads AGENTS.md to learn what the gate does; a documented check
   that is not registered is a check nobody runs.

2. **QUANTITY WORDS** in the two descriptions, after issue references are
   stripped. NOT the prose: the two lists legitimately differ in emphasis
   (`by membership` vs `by MEMBERSHIP`, `set -u abort` vs `abort`), and a gate
   demanding byte-identity here would be a gate people route around. What may
   NOT differ is a NUMBER — "six" vs "seven", "two lists" vs "three lists" —
   because a quantity is a claim, and a claim drifting between two copies is
   how this was found. Issue/plan numbers are stripped first: they are
   addresses, not quantities, and a row may cite one in the array and omit it
   in the table without contradicting anything (measured: 1 of 15 rows does).

Exit 0 clean, 1 on drift, **2 if the instrument is untrustworthy** (either
list unparseable, or a floor breached). A parser that silently reads ZERO rows
reports perfect agreement between two empty sets.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SH = HERE / "docs_gate.sh"
AGENTS = REPO_ROOT / "AGENTS.md"

# The AGENTS.md table is located by its SECTION heading, not by "any table row
# whose first cell looks like a .py file". A bare document-wide scan happens to
# work today and silently absorbs the next unrelated table somebody adds.
SECTION = "## Docs gate + drift sweeps"

# Floor: below this the parse went blind and two empty sets agree perfectly.
MIN_ROWS = 10

_CHECKS_BLOCK = re.compile(r"^CHECKS=\(\n(.*?)^\)$", re.S | re.M)
_CHECK_ROW = re.compile(r'^\s*"([^:"]+):(.*)"\s*$')
_TABLE_ROW = re.compile(r"^\|\s*`([A-Za-z0-9_]+\.py)`\s*\|\s*(.*?)\s*\|\s*$", re.M)

# Addresses, not quantities — stripped before the quantity comparison.
_ISSUE_REF = re.compile(r"\b(?:Issues?|Plans?|Research|Bench(?:mark)?s?)\s+\d[\d,\s/]*", re.I)
_QUANTITY = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", re.I
)


def quantities(desc: str) -> list[str]:
    return sorted(q.lower() for q in _QUANTITY.findall(_ISSUE_REF.sub(" ", desc)))


def parse_checks(text: str) -> dict[str, str]:
    m = _CHECKS_BLOCK.search(text)
    if not m:
        print("✗ INSTRUMENT: no CHECKS=( ... ) array in docs_gate.sh — the array was "
              "renamed or reformatted; this gate cannot read it and must not report clean")
        raise SystemExit(2)
    out: dict[str, str] = {}
    for line in m.group(1).splitlines():
        row = _CHECK_ROW.match(line)
        if row:
            out[row.group(1).split("/")[-1]] = row.group(2)
    return out


def parse_table(text: str) -> dict[str, str]:
    start = text.find(SECTION)
    if start < 0:
        print(f"✗ INSTRUMENT: AGENTS.md has no {SECTION!r} heading — the section was "
              "retitled; the table cannot be located")
        raise SystemExit(2)
    nxt = text.find("\n## ", start + len(SECTION))
    body = text[start : nxt if nxt > 0 else len(text)]
    return {m.group(1): m.group(2) for m in _TABLE_ROW.finditer(body)}


def main() -> int:
    checks = parse_checks(io.open(SH, encoding="utf-8").read())
    table = parse_table(io.open(AGENTS, encoding="utf-8").read())

    for label, rows in (("docs_gate.sh CHECKS", checks), ("AGENTS.md table", table)):
        if len(rows) < MIN_ROWS:
            print(f"✗ INSTRUMENT: parsed {len(rows)} rows from {label} < floor {MIN_ROWS} — "
                  f"the parser went blind; two empty sets agree perfectly")
            return 2

    findings: list[str] = []
    only_sh = sorted(set(checks) - set(table))
    only_ag = sorted(set(table) - set(checks))
    for name in only_sh:
        findings.append(f"  {name} — in CHECKS, NOT in the AGENTS.md table: a check "
                        f"nobody reading the contract knows runs")
    for name in only_ag:
        findings.append(f"  {name} — in the AGENTS.md table, NOT in CHECKS: a documented "
                        f"check that nothing runs")

    for name in sorted(set(checks) & set(table)):
        a, b = quantities(checks[name]), quantities(table[name])
        if a != b:
            findings.append(
                f"  {name} — QUANTITY drift: docs_gate.sh says {a or '(none)'}, "
                f"AGENTS.md says {b or '(none)'}\n"
                f"      sh:     {checks[name][:110]}\n"
                f"      AGENTS: {table[name][:110]}")

    print(f"  {len(checks)} CHECKS row(s) vs {len(table)} AGENTS.md table row(s); "
          f"membership + quantity words compared (prose deliberately not)")
    if findings:
        print(f"✗ docs_gate CHECKS sync FAILED — {len(findings)} divergence(s)")
        for f in findings:
            print(f)
        return 1
    print(f"✓ docs_gate CHECKS sync PASSED — {len(checks)} rows agree by membership, "
          f"0 quantity drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
