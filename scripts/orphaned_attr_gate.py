#!/usr/bin/env python3
"""Gate on a `#[cfg]` separated from its item by a blank line.

Rust binds an attribute to the next item **across a blank line**. So this

    #[cfg(debug_assertions)]

    use crate::absorb_compress::{AbsorbCompress, AbsorbCompressLayer};

compiles, and makes the import debug-only. It is not a lint clippy has, and it
is invisible in review because the blank line reads as separation.

## The bug this is built from, with dates

`crates/katgpt-pruners/src/sdar/sdar_absorb.rs` was in exactly that state for
two days (fixed in `a08376a0`):

- `26d055c6` dropped `use std::cmp::Ordering;`, which is what the
  `#[cfg(debug_assertions)]` above it had correctly applied to, and left the
  attribute behind. It silently re-bound to the NEXT import.
- Every release build of `katgpt-pruners --features sdar_gate` then failed with
  5 errors: `debug_assertions` is off in release, so the import vanished while
  its five usages stayed unconditional.
- `26d055c6`'s own validation was a DEBUG run (`lib 597/0`), where
  `debug_assertions` is on and the import exists. That is `.docs/10_audits/cfg_gated_silent_zero_pass.md` T2b's
  lesson with the sign flipped — there, debug manufactured four false perf
  reds; here, debug hid a real build break.
- `7e34ccef` then deleted the blank line, which made the wrong binding look
  deliberate and would have erased the evidence.

## Why this can be a GATE and not a report

Measured across all contract repos at the fix (2026-09-03, when the workspace
was 19): **zero** sites. Not "few" — zero. So the pin is 0 and any future
occurrence is the push that introduces it. **Re-measured 2026-09-04 over the
live 16** (the three retired repos left for `git/obsolete/`): still zero in
every one of them. **Re-measured 2026-09-06**, after ~500 sibling commits:
still zero in all 16 — **11,132 `.rs` files, 49,624 outer-`#[cfg]` sites, 0
orphaned**. The site count is the part worth keeping: a zero over 49,624 sites
is evidence; a zero over a walk that has gone blind is not, which is why the
PASS line prints the population it saw rather than the one it assumed.

Do not read that as this gate's verdict. It audits **one** repo per
invocation, and until 2026-09-04 its PASS line printed "measured 0 across 19
repos" on every run — a cross-repo claim no run had made, with a count that had
gone stale two commits earlier, printed two lines below the repo-set gate
saying 16. It now reports the repo it scanned and the population it saw.

The broader shape (**any** attribute + blank line + item) is 2,044 sites and is
NOT gateable: it is dominated by whole-file INNER attributes (`#![cfg(...)]`),
which bind to the enclosing module rather than to the next item and are
conventionally followed by a blank line. Narrowing to OUTER `#[cfg]` /
`#[cfg_attr]` is what takes 2,044 to 0 — the narrowing is the instrument.

## Scope

katgpt-rs only, same reasoning as `cfg_gated_floor_gate.py`: CI has a single
checkout. Pass a repo path to audit a sibling; adopting it there is an owner
call, like `.docs/10_audits/cfg_gated_silent_zero_pass.md` T3.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# OUTER `#[cfg(...)]` / `#[cfg_attr(...)]` only. An INNER `#![cfg(...)]` binds
# to the enclosing module, so a blank line after it is correct and common —
# including it is what made the naive measurement 2,044 instead of 0.
OUTER_CFG = re.compile(r"^\s*#\s*\[\s*cfg(?:_attr)?\s*\(")
ANY_ATTR = re.compile(r"^\s*#!?\s*\[")

# Pruned during the walk, never filtered afterwards. `rglob("*.rs")` followed by
# a `"target" in parts` filter still DESCENDS into target/ (117 GB, ~1.3M
# entries) — the same trap that made bench_doc_audit.py take 556s, and the
# `find -not -path` trap one level over.
PRUNE = {"target", ".git", "node_modules", ".venv", "__pycache__"}

# `max_offenders = 0` is a CEILING, and a ceiling passes over an empty
# population — a pruning bug, a moved source root or a read failure all print a
# confident "0 offender(s)" that is indistinguishable from the clean state this
# asserts. selftest() pins the tokenizer against synthetic input; these pin the
# POPULATION of the real run. katgpt-rs measured 2026-09-04: 2,418 `.rs` files,
# 6,936 outer-`#[cfg]` sites. Floors sit well below so that extracting code to a
# sibling crate does not red the gate, while a blind walk still does.
# Scope-guarded to katgpt-rs: a sibling audit has its own population.
FLOOR_FILES = 1500
FLOOR_CFG_SITES = 4000


class Scan(NamedTuple):
    """A verdict AND the population it was reached over."""

    offenders: list[tuple[str, int, str, str]]
    files: int
    cfg_sites: int


def scan(repo: Path) -> Scan:
    out: list[tuple[str, int, str, str]] = []
    files_seen = 0
    cfg_seen = 0
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in PRUNE]
        for fn in files:
            if not fn.endswith(".rs"):
                continue
            p = Path(root) / fn
            try:
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            files_seen += 1
            cfg_seen += sum(1 for line in lines if OUTER_CFG.match(line))
            for i in range(len(lines) - 2):
                if not OUTER_CFG.match(lines[i]) or lines[i + 1].strip():
                    continue
                nxt = lines[i + 2]
                # A following comment or another attribute is not the item, and
                # a blank-line run means the attribute is dangling further
                # down; both are reported only when a real item follows.
                if not nxt.strip() or nxt.lstrip().startswith("//") or ANY_ATTR.match(nxt):
                    continue
                out.append(
                    (str(p.relative_to(repo)), i + 1, lines[i].strip(), nxt.strip()[:60])
                )
    return Scan(sorted(out), files_seen, cfg_seen)


def selftest() -> None:
    """Pin both directions on every invocation.

    The false-negative direction is the one that matters: a regex regression
    makes this print `0 offenders` forever, which is indistinguishable from the
    clean state it is asserting.
    """
    import tempfile

    positive = (
        "#[cfg(debug_assertions)]\n"
        "\n"
        "use crate::absorb_compress::AbsorbCompressLayer;\n"
    )
    negatives = {
        # Correctly attached — the overwhelmingly common shape.
        "attached": "#[cfg(feature = \"x\")]\nuse a::b;\n",
        # INNER attribute: binds to the module, blank line is conventional.
        # This single case is the difference between 2,044 hits and 0.
        "inner": '#![cfg(feature = "x")]\n\nuse a::b;\n',
        # A doc comment after the blank line is not the item.
        "comment": '#[cfg(feature = "x")]\n\n// note\nuse a::b;\n',
        # Another attribute after the blank line: still an attribute run.
        "attr": '#[cfg(feature = "x")]\n\n#[derive(Debug)]\nstruct S;\n',
        # A non-cfg attribute is formatting, not conditional compilation.
        "derive": "#[derive(Debug)]\n\nstruct S;\n",
    }
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pos.rs").write_text(positive)
        got = scan(root)
        assert len(got.offenders) == 1, f"the real bug's shape was not detected: {got}"
        assert got.offenders[0][0] == "pos.rs"
        # The population is a separate claim from the verdict and is pinned
        # separately: a walk that counts nothing must not be able to report a
        # clean zero. This is the in-miniature version of FLOOR_* below.
        assert (got.files, got.cfg_sites) == (1, 1), f"population miscounted: {got}"

    for name, src in negatives.items():
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "n.rs").write_text(src)
            got = scan(root)
            assert got.offenders == [], f"false positive on {name}: {got.offenders}"
            # A negative is a real scanned file, not an unread one — otherwise
            # every case above would also pass on a walk that saw nothing.
            assert got.files == 1, f"{name} was not read: {got}"

    # The walk must PRUNE, not filter: a file under target/ is invisible.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "target").mkdir()
        (root / "target" / "gen.rs").write_text(positive)
        got = scan(root)
        assert got.offenders == [], "target/ was walked"
        assert got.files == 0, f"target/ was read: {got}"

    # A non-`.rs` file is not the population. Guards the extension filter, which
    # is what a floor would otherwise be silently satisfied by (`.md` is plentiful).
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "n.md").write_text(positive)
        assert scan(root) == Scan([], 0, 0), "a non-.rs file entered the population"


def main(argv: list[str]) -> int:
    selftest()
    repo = Path(argv[1]).resolve() if len(argv) > 1 else REPO_ROOT
    got = scan(repo)
    found = got.offenders
    pop = f"{got.files} .rs file(s), {got.cfg_sites} outer-#[cfg] site(s)"

    print(f"orphaned-attribute gate — {repo.name}: {len(found)} offender(s) over {pop}")
    for path, line, attr, item in found:
        print(f"  ✗ {path}:{line}  {attr}")
        print(f"      binds ACROSS the blank line to: {item}")
    if found:
        print(
            "\n  A `#[cfg]` separated from its item by a blank line still applies to "
            "that item.\n  Either attach it or delete it — see this file's header for "
            "the two-day release\n  break it is built from (a08376a0)."
        )
        print(f"✗ orphaned-attribute gate FAILED — {len(found)} site(s)")
        return 1

    # The floors are this repo's population only — a sibling audit is a
    # different population and gets the report without the verdict half.
    if repo == REPO_ROOT and (got.files < FLOOR_FILES or got.cfg_sites < FLOOR_CFG_SITES):
        print(
            f"\n  The population fell below its floor ({FLOOR_FILES} files, "
            f"{FLOOR_CFG_SITES} sites).\n  A zero over a shrunken population is not a "
            "clean repo — read it as the walk having\n  gone blind (pruning, a moved "
            "source root, a read failure) until proven otherwise.\n  If the shrink is "
            "real, re-measure and move the floor in the same commit."
        )
        print("✗ orphaned-attribute gate FAILED — population below floor")
        return 1

    scope = "this repo" if repo == REPO_ROOT else f"{repo.name} (sibling audit, floors n/a)"
    print(f"✓ orphaned-attribute gate PASSED — pinned at 0, measured 0 over {pop} in {scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
