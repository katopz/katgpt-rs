#!/usr/bin/env python3
"""Gate: AGENTS.md §"Repo count" must name exactly the repos in repo_set.txt.

WHY THIS EXISTS. On 2026-09-03 that paragraph listed `riir-armageddon`
(retired the day before, directory gone) and omitted `seal-remake-unity`
(enrolled in the same window). **One repo left and one arrived, so the total
stayed 19** — the paragraph's own count was RIGHT, every count nearby agreed,
and the set was wrong anyway. Four routing instruments repeated it, including
the research skill's "8-repo discipline", which cites this paragraph as
canonical.

That is the case no count-based check can catch, and the reason this gate
compares MEMBERSHIP first and cardinality second. A count is not a checksum
over a set.

WHAT IS CHECKED, and why it is this and not the obvious thing.

The obvious check — grep every doc for a repo name and verify the repo exists
— false-positives on history, and destroying history to satisfy a gate is the
failure mode `skill_repo_set_gate.py` already documents. boundary-guard's
ledger *should* say `riir-armageddon`: the 227->225 edge delta IS that repo's
two edges leaving. goat-audit's duplicate example *should* keep it: the hazard
outlives the repo that illustrated it.

So this gates exactly ONE paragraph — the one that declares itself canonical
and that every other instrument defers to. Prose elsewhere stays free.

BOTH INPUTS ARE COMMITTED, so this runs in CI, where sibling repos do not
exist. It compares the paragraph against `scripts/repo_set.txt` (the
vocabulary), NOT against a directory walk (the population) — the split this
workspace requires of any cross-repo instrument, because deriving both sides
from one walk is what makes such a gate permanently green. repo_set.txt's own
freshness against the real workspace is gated separately, on the workstation,
by `skill_repo_set_gate.py`.

Exit 0 = the paragraph's membership and both its declared counts agree.
Exit 1 = drift. Exit 2 = the parser is untrustworthy (selftest failed), which
is deliberately NOT the same outcome as drift.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_MD = REPO_ROOT / "AGENTS.md"
REPO_SET = REPO_ROOT / "scripts" / "repo_set.txt"

START = "**Repo count:**"
# A backticked token that is a repo name: no path separator, no dot. This is
# what keeps `BOUNDARY.md` and `scripts/repo_set.txt` out of the membership.
REPO_TOKEN = re.compile(r"`([a-z0-9][a-z0-9-]*)`")


def parse_paragraph(text: str) -> tuple[set[str], int, int]:
    """-> (membership, declared_product_count, declared_total_count).

    The membership region runs from the START marker to the end of the
    "(add ...)" parenthetical. Stopping there is load-bearing: the prose that
    FOLLOWS it explains the retirement and necessarily names the retired repo,
    so a whole-blockquote scan would re-admit exactly what the gate exists to
    reject.
    """
    i = text.find(START)
    if i < 0:
        raise ValueError(f"AGENTS.md has no {START!r} section")
    j = text.find("(add ", i)
    if j < 0:
        raise ValueError("no '(add ...)' membership list after the marker")
    k = text.find(").", j)
    if k < 0:
        raise ValueError("the '(add ...)' list is unterminated")
    region = text[i:k]

    m_prod = re.search(r"product/distillation set is (\d+)", region)
    m_total = re.search(r"workspace is \*\*(\d+) repos\*\*", region)
    if not m_prod or not m_total:
        raise ValueError("could not read both declared counts")

    names = {n for n in REPO_TOKEN.findall(region)}
    return names, int(m_prod.group(1)), int(m_total.group(1))


def product_set(text: str) -> set[str]:
    """The names before '(private).' — the distillation set proper."""
    i = text.find(START)
    k = text.find("(private).", i)
    if k < 0:
        raise ValueError("no '(private).' terminator for the product set")
    return set(REPO_TOKEN.findall(text[i:k]))


def selftest() -> None:
    """Pins the parse. A silent parser regression here degrades to 'membership
    is empty', which compares equal to nothing and prints a confident pass."""
    sample = (
        "> **Repo count:** the **product/distillation set is 2** — `aa` (public) +\n"
        "> `bb` (private). That is NOT the repo total: the\n"
        "> workspace is **3 repos**, all of which carry a root `BOUNDARY.md`\n"
        "> (add `cc`).\n>\n"
        "> Later prose naming `zz` must NOT be collected.\n"
    )
    names, prod, total = parse_paragraph(sample)
    assert names == {"aa", "bb", "cc"}, names
    assert prod == 2 and total == 3, (prod, total)
    assert "zz" not in names, "post-list prose leaked into the membership"
    assert "BOUNDARY" not in " ".join(names), "a dotted token leaked in"
    assert product_set(sample) == {"aa", "bb"}, product_set(sample)


def main() -> int:
    # Prints carry glyphs the Windows locale codecs cannot encode (checked
    # 2026-09-06 on cp874: check/cross/middot/arrow FAIL, em-dash OK); keep the
    # locale encoding and degrade only the fatal chars to escapes -- the
    # staged_set_audit house pattern (utf-8 pinning would mojibake legacy consoles).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (embedded / detached); keep old behavior
    try:
        selftest()
    except AssertionError as e:
        print(f"    ✗ agents repo-set gate SELFTEST FAILED: {e}", file=sys.stderr)
        return 2

    text = AGENTS_MD.read_text(encoding="utf-8")
    declared, n_prod, n_total = parse_paragraph(text)
    prod = product_set(text)
    actual = {
        ln.strip()
        for ln in REPO_SET.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    }

    problems = []
    for name in sorted(declared - actual):
        problems.append(f"AGENTS.md names `{name}`, which is not in scripts/repo_set.txt")
    for name in sorted(actual - declared):
        problems.append(f"scripts/repo_set.txt has `{name}`, which AGENTS.md does not name")
    if n_total != len(actual):
        problems.append(f'"workspace is {n_total} repos" but repo_set.txt has {len(actual)}')
    if n_prod != len(prod):
        problems.append(f'"product/distillation set is {n_prod}" but {len(prod)} names precede "(private)."')

    if problems:
        print("    ✗ agents repo-set gate FAILED", file=sys.stderr)
        for p in problems:
            print(f"        - {p}", file=sys.stderr)
        print(
            "      Fix AGENTS.md §\"Repo count\". Note a matching COUNT proves nothing:\n"
            "      the defect this gate was written for was a membership SWAP that left\n"
            "      the total unchanged.",
            file=sys.stderr,
        )
        return 1

    print(
        f"    ✓ agents repo-set gate PASSED — {len(actual)} repos, "
        f"membership + both declared counts agree with scripts/repo_set.txt"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
