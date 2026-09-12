#!/usr/bin/env python3
"""GATE: a fenced code block that is never closed swallows the rest of its file.

Everything after an unterminated ` ``` ` renders as code — headings, tables,
the nav footer, all of it. Measured 2026-09-12 across 19 contract repos: **19**
such files over 5048 tracked `.md`, swallowing **611** lines. The largest,
`.plans/048_research_audit_fixes.md`, opened a ```bibtex block under "Research
Citations" and never closed it, turning the following 146 lines — an entire
second document section — into a code listing.

It is also a PARSE hazard, which is why this gate lives beside the others: a
scanner that toggles in/out state on every fence line mis-phases permanently
from that point and thereafter scans the complement (prose read as code, code
read as prose), reporting clean either way. That is not hypothetical here —
`.agents/skills/rust-optimize/SKILL.md`'s unclosed ```text swallowed
`skill_repo_set_gate.py`'s own first canary, which is how the class was found.
Its `fenced_blocks()` is CommonMark-ish for that reason, and is imported here
rather than re-derived (Issue 755: a second copy of a rule this subtle is a
second thing to get wrong).

⚠ **The reported line is the DANGLING fence, NOT necessarily the defect.** A
single stray fence inverts the pairing of every fence after it, so the dangling
one may be a legitimate CLOSER whose partner was consumed upstream. Two of the
three shapes were measured in the workspace:

    missing closer  a block opens under a heading and the file ends inside it
    stray fence     an orphan ``` between two prose paragraphs (delete it)
    missing opener  console output that was meant to be fenced and is not

Locate the defect before repairing: read the first non-blank body line. Code
means the closer is missing; prose means the fence itself is the orphan.

Exit 0 clean, 1 on an unterminated fence, **2 if the instrument went blind**
(the walk collapsed below its floor — a ceiling over zero files is green for
the wrong reason).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from skill_repo_set_gate import fenced_blocks  # noqa: E402 — reused, not re-derived

REPO_ROOT = HERE.parent

# The walk's own floor. 1517 tracked `.md` at landing (2026-09-12); floored well
# under so ordinary churn does not red it, but a `ls-files` regression that
# returns nothing cannot print a confident green over zero files.
MIN_FILES = 800


def unterminated(repo: Path) -> tuple[list[tuple[str, int, int]], int]:
    """([(path, opening line, lines swallowed)], files walked)."""
    out: list[tuple[str, int, int]] = []
    listing = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "*.md"],
        capture_output=True, text=True,
    ).stdout.splitlines()          # NOT .split() — a tracked path may hold a space
    walked = 0
    for rel in listing:
        f = repo / rel
        if not f.is_file():
            continue
        walked += 1
        text = f.read_text(encoding="utf-8", errors="replace")
        n_lines = len(text.splitlines())
        for first, last, _body, _prev in fenced_blocks(text):
            if last < 0:
                out.append((rel, first, n_lines - first))
    return out, walked


def main() -> int:
    findings, walked = unterminated(REPO_ROOT)

    if walked < MIN_FILES:
        print(f"✗ INSTRUMENT: walked {walked} tracked .md file(s) < floor {MIN_FILES} — "
              f"the population went blind; a clean verdict below would mean nothing")
        return 2

    if findings:
        for rel, line, swallowed in sorted(findings, key=lambda r: -r[2]):
            print(f"  ⛔ {rel}:{line} — fence never closed, {swallowed} line(s) "
                  f"render as code to EOF")
        print(f"✗ markdown fence gate FAILED — {len(findings)} unterminated fence(s) "
              f"over {walked} tracked .md file(s). The reported line is the DANGLING "
              f"fence, not necessarily the defect: read the first non-blank body line "
              f"— code means a closer is missing, prose means the fence is an orphan.")
        return 1

    print(f"✓ markdown fence gate PASSED — 0 unterminated fence(s) over {walked} "
          f"tracked .md file(s) (floor {MIN_FILES})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
