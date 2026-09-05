#!/usr/bin/env python3
"""Gate on katgpt-rs having no target that compiles to NOTHING at its own row.

The verdict half of `cfg_row_implication_audit.py`, katgpt-rs-scoped and split
from the report for the reason every gate in this family is: the report must
stay runnable over siblings whose owners have not taken Issue 513, and a report
that exits 1 on them is a report nobody runs.

Gateable because it needs **no compiler** — the whole question is decided by a
manifest's feature graph and a source file's leading `#![cfg]`. And worth
gating because the defect is invisible to everything else: the row exists, so
`cfg_gated_target_audit.py` counts the reader as protected; the target builds,
so `required_features_build_audit.py` reports BUILDS; the harness prints
`ok. 0 passed` and cargo exits 0. Measured 2026-09-06, riir-train `054a39a2`:
fixing one such row took a target from 0 passed to 1 passed — an assertion
that had never executed at any revision.

## Pins (scripts/cfg_row_implication_floors.txt)

`max_empty_at_row` is a **WALL at 0**, not a ratchet: a row that compiles its
own target to nothing is never legitimate.

`max_unresolved` is a wall too, at katgpt-rs's measured 0 — but read it as the
narrower claim it is. UNRESOLVED means a predicate this report declines to rule
on (`any(feature = ...)`, or a feature under `not(...)`), not a clean row. A
sibling legitimately has them; this repo happens not to.

`min_rows_scanned` and `min_with_cfg` are **FLOORS**, for the reason every
floor in this family exists: the two ceilings above go green over whatever the
tokenizer can NAME, so a scanner regression takes the population to ~0 and both
ceilings pass, indistinguishable from a clean repo. That is not hypothetical
here — an early cut of the scanner reported two riir-train targets as "source
unreadable" (they use cargo's DIRECTORY form, `tests/<name>/main.rs`) and
silently skipped them.

Exit 0 clean · 1 drift · 2 the instrument is untrustworthy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cfg_row_implication_audit as cria  # noqa: E402

FLOORS = Path(__file__).resolve().parent / "cfg_row_implication_floors.txt"
REQUIRED_PINS = {"max_empty_at_row", "max_unresolved", "min_rows_scanned", "min_with_cfg"}


def read_pins(path: Path) -> dict[str, int]:
    pins: dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, val = line.partition("=")
        pins[key.strip()] = int(val.strip())
    return pins


def main(argv: list[str]) -> int:
    # The report's selftest exits 2 on its own; run it first so an
    # untrustworthy instrument is never mistaken for moved pins.
    cria.selftest()

    if not FLOORS.is_file():
        print(f"✗ pins file missing: {FLOORS}")
        return 2
    pins = read_pins(FLOORS)
    missing = REQUIRED_PINS - set(pins)
    if missing:
        print(f"✗ pins file is missing required keys: {sorted(missing)}")
        return 2

    repo = Path(argv[0]).resolve() if argv else Path(__file__).resolve().parent.parent
    found = cria.audit_repo(repo)
    n_rows = len(found)
    empty = sum(1 for f in found if f.verdict == cria.EMPTY)
    unres = sum(1 for f in found if f.verdict == cria.UNRESOLVED)
    with_cfg = sum(1 for f in found if f.verdict != cria.NO_CFG)

    fails: list[str] = []
    if empty > pins["max_empty_at_row"]:
        fails.append(f"EMPTY-AT-ROW {empty} > pinned {pins['max_empty_at_row']}")
    if unres > pins["max_unresolved"]:
        fails.append(f"UNRESOLVED {unres} > pinned {pins['max_unresolved']}")
    if n_rows < pins["min_rows_scanned"]:
        fails.append(
            f"only {n_rows} rows scanned < floor {pins['min_rows_scanned']} — "
            f"the manifest walk shrank, so both ceilings above are vacuous"
        )
    if with_cfg < pins["min_with_cfg"]:
        fails.append(
            f"only {with_cfg} rows carry a leading #![cfg] < floor {pins['min_with_cfg']} — "
            f"the source scanner narrowed, so a green here is a green over nothing"
        )

    if fails:
        print("✗ cfg-row-implication gate FAILED:")
        for f in fails:
            print(f"    {f}")
        for f in found:
            if f.verdict == cria.EMPTY:
                print(f"    EMPTY-AT-ROW  {f.row.label}  row={f.row.features} "
                      f"MISSING {sorted(f.missing)}")
            elif f.verdict == cria.UNRESOLVED:
                print(f"    UNRESOLVED    {f.row.label}  {f.why}")
        return 1

    print(f"✓ cfg-row-implication gate PASSED — {n_rows} rows, {with_cfg} with a "
          f"leading #![cfg], {empty} empty-at-row, {unres} unresolved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
