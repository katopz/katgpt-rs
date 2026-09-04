#!/usr/bin/env python3
"""Gate on docs_gate.yml's two hand-duplicated `paths:` trigger lists staying in sync.

The workflow runs its CHECKS per-push on `main` and (as a second trigger) on
pull requests. GitHub Actions has no YAML anchors usable here, so the two
lists are duplicated BY HAND — the file's own comment says "keep the two
lists in sync by hand", which is a known-unenforced invariant. A push-list
that drifts behind the PR-list (or vice versa) is silent: the gate simply
runs on one trigger and not the other, and nothing anywhere compares them.

Issue 724 T4b. The same failure shape as the issue's own trigger finding
(T4's addendum): `docs_gate.yml`'s paths filter had NO globs for the four
serial-numbered doc dirs the numbering gate reads — found by reading the
trigger list, not by trusting that adding a check wired it up. This gate is
the instrument that keeps that class from regrowing: it asserts the INVARIANT
(the two lists are the same set) rather than any particular contents, so the
next legitimately-added glob only needs to land in BOTH lists to stay green.

## Verdicts

- exit 0 — exactly two `paths:` blocks, same item set.
- exit 1 — drift: the two blocks disagree (the missing/extra items are
  printed, first-seen order preserved).
- exit 1 — structural: fewer/more than two `paths:` blocks (a trigger
  removed, or a third `paths:` someone added without updating this gate).
- exit 2 — `selftest()` failed: the parser is untrustworthy, and a
  confident green from an untrustworthy instrument is worse than no gate
  (the `numbering_gate.py` / `skill_repo_set_gate.py` convention).

## Scope

katgpt-rs only: CI has a single checkout, and the workflow file this gates
on lives in this repo. The path is resolved relative to the repo root so the
check works from any CWD (docs_gate.sh runs from the root, but a workstation
run should not have to).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

WORKFLOW = ".github/workflows/docs_gate.yml"

# A `paths:` key at the trigger depth (4 spaces) inside an `on:` block,
# followed by list items at 6 spaces. Both hand-duplicated lists use single
# quotes; the parser keeps them optional so the selftest can pin both shapes.
PATHS_KEY = re.compile(r"^ {4}paths:\s*(?:#.*)?$")
LIST_ITEM = re.compile(r"^ {6}-\s*(?:'([^']*)'|\"([^\"]*)\")\s*(?:#.*)?$")


def parse_paths_blocks(text: str) -> list[list[str]]:
    """Every `paths:` list under `on:`, in file order, items in order.

    A block ends at the first line that is neither a list item nor a
    comment/blank, so comments BETWEEN items (the real file interleaves
    them heavily) stay inside the block.
    """
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if PATHS_KEY.match(line):
            if current is not None:
                blocks.append(current)
            current = []
            continue
        if current is None:
            continue
        m = LIST_ITEM.match(line)
        if m:
            current.append(m.group(1) if m.group(1) is not None else m.group(2))
        elif line.strip() == "" or line.lstrip().startswith("#"):
            continue  # blank / comment between items: still inside the block
        else:
            blocks.append(current)
            current = None
    if current is not None:
        blocks.append(current)
    return blocks


def selftest() -> list[str]:
    fails: list[str] = []

    # 1. two identical lists parse equal (the passing shape)
    same = """\
on:
  push:
    branches: [main]
    paths:
      - 'README.md'
      - 'scripts/a.py'
  pull_request:
    paths:
      - 'README.md'
      - 'scripts/a.py'
"""
    blocks = parse_paths_blocks(same)
    if len(blocks) != 2:
        fails.append(f"expected 2 blocks, got {len(blocks)}")
    elif blocks[0] != blocks[1]:
        fails.append(f"identical lists parsed unequal: {blocks}")

    # 2. drift is DETECTED (the failing shape this gate exists for)
    drifted = same.replace("  pull_request:\n    paths:\n",
                           "  pull_request:\n    paths:\n      - 'EXTRA.md'\n")
    blocks = parse_paths_blocks(drifted)
    if len(blocks) == 2 and set(blocks[0]) == set(blocks[1]):
        fails.append("drifted lists read as equal")

    # 3. comments + blank lines BETWEEN items stay inside the block and are
    #    skipped (the real file interleaves them heavily)
    commented = """\
on:
  push:
    paths:
      - 'README.md'
      # why this glob exists (a long story)
      - 'scripts/a.py'

jobs:
  x:
"""
    blocks = parse_paths_blocks(commented)
    if len(blocks) != 1 or blocks[0] != ["README.md", "scripts/a.py"]:
        fails.append(f"commented list parsed wrong: {blocks}")

    # 4. double-quoted items are admitted (parser keeps both YAML spellings)
    dq = "    paths:\n      - \"README.md\"\n"
    blocks = parse_paths_blocks("on:\n  push:\n" + dq)
    if blocks != [["README.md"]]:
        fails.append(f"double-quoted item lost: {blocks}")

    # 5. a lone `paths:` (no items) parses as an EMPTY block — the
    #    structural check must see 2 blocks even when one is empty, so the
    #    set comparison (not the block count) reports the drift.
    lone = "on:\n  push:\n    paths:\n      - 'a'\n  pull_request:\n    paths:\n  workflow_dispatch:\n"
    blocks = parse_paths_blocks(lone)
    if len(blocks) != 2 or blocks[1] != []:
        fails.append(f"lone/empty paths block mis-parsed: {blocks}")

    return fails


def main() -> int:
    fails = selftest()
    if fails:
        print("✗ docs-gate paths-sync SELFTEST FAILED — instrument untrustworthy:")
        for f in fails:
            print(f"    {f}")
        return 2

    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    wf = repo / WORKFLOW
    if not wf.is_file():
        print(f"✗ workflow missing: {wf}")
        return 1

    blocks = parse_paths_blocks(wf.read_text(encoding="utf-8"))
    if len(blocks) != 2:
        print(f"✗ expected exactly 2 paths blocks (push + pull_request), found {len(blocks)}")
        return 1

    push, pr = blocks
    push_set, pr_set = set(push), set(pr)
    if push_set == pr_set:
        print(f"✓ docs_gate.yml trigger paths in sync — {len(push_set)} globs in both lists")
        return 0

    only_push = [g for g in push if g not in pr_set]
    only_pr = [g for g in pr if g not in push_set]
    print("✗ docs_gate.yml trigger paths DRIFTED (keep the two hand-duplicated lists identical):")
    for g in only_push:
        print(f"    push-only:   {g}")
    for g in only_pr:
        print(f"    PR-only:     {g}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
