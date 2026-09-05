#!/usr/bin/env python3
"""GATE: no `required-features` row may name a feature its package cannot enable.

The verdict half of `scripts/required_features_build_audit.py`'s free static
pass, katgpt-rs-scoped, kept in a separate file from the report for the reason
its siblings are: the report must stay runnable over sibling repos whose owners
have not taken this on, and a report that exits 1 on them is a report nobody
runs.

Why this one is gateable when the report's other two verdicts are not: a row
naming an undefined feature is decided by the manifest alone (no build, under a
second for the whole workspace) and is **never legitimate**. Cargo silently
SKIPS such a target in every invocation that does not name it — `cargo test
--workspace` and `--all-features` included — so it reports a green zero
forever, while every audit in the `cfg_gated_target_audit.py` family counts it
as PROTECTED because the row exists. Naming it explicitly is the only loud
case (`error: target ... requires the features`, exit 101), and nothing names
these targets.

What it does NOT check: whether the row is SUFFICIENT. That needs the compiler
(the report's `FAILS-TO-BUILD` verdict, priced in hours) and is riir-train
`.issues/513` T2/T3.

Exit codes: 0 clean · 1 drift · 2 the instrument itself is untrustworthy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# DRY: the parse and the validity model are the report's, so the gate and the
# report can never disagree about what a row is or which names are enableable.
from required_features_build_audit import parse_rows, static_invalid  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PINS = REPO / "scripts" / "required_features_floors.txt"
REQUIRED_PINS = ("max_invalid_rows", "min_rows_scanned")


def read_pins(path: Path) -> dict[str, int]:
    pins: dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        try:
            pins[key.strip()] = int(value.strip())
        except ValueError:
            print(f"✗ unparseable pin: {raw!r}", file=sys.stderr)
            raise SystemExit(2)
    missing = [k for k in REQUIRED_PINS if k not in pins]
    if missing:
        print(f"✗ pins file is missing {missing}", file=sys.stderr)
        raise SystemExit(2)
    return pins


def selftest() -> None:
    """Pin the validity model's two directions. Runs on EVERY invocation.

    The report owns `static_invalid`'s own selftest; this one pins the two
    facts THIS gate's verdict rests on, so a change there cannot silently
    make the gate vacuous: a same-package feature that does not exist is a
    finding, and a `dep/feat` row is NOT (modelling those as invalid would
    have filed 10 riir-ai benches as dead targets — see AGENTS.md).
    """
    feats, deps = {"mine"}, {"dep"}
    cases = [("mine", False), ("dep/extra", False), ("nope", True)]
    for feature, want_finding in cases:
        got = bool(static_invalid(feature, feats, deps))
        if got != want_finding:
            print(
                f"SELFTEST FAILED: static_invalid({feature!r}) finding={got}, "
                f"want {want_finding}",
                file=sys.stderr,
            )
            raise SystemExit(2)
    if not PINS.is_file():
        print(f"SELFTEST FAILED: pins file missing: {PINS}", file=sys.stderr)
        raise SystemExit(2)


def main() -> int:
    selftest()
    pins = read_pins(PINS)
    rows = parse_rows(REPO)
    findings: list[tuple[str, list[str]]] = []
    for row in rows:
        reasons = [
            r
            for r in (static_invalid(f, row.feats, row.deps) for f in row.features)
            if r
        ]
        if reasons:
            findings.append((row.label, reasons))

    failed = False
    if len(rows) < pins["min_rows_scanned"]:
        print(
            f"✗ population FLOOR: scanned {len(rows)} row(s), pinned "
            f"{pins['min_rows_scanned']} — the instrument may be blind, and a "
            f"blind instrument passes every ceiling below",
        )
        failed = True
    if len(findings) > pins["max_invalid_rows"]:
        print(
            f"✗ {len(findings)} row(s) name a feature their package cannot "
            f"enable (pinned {pins['max_invalid_rows']}):"
        )
        for label, reasons in findings:
            print(f"      {label}  —  {'; '.join(reasons)}")
        failed = True
    if failed:
        return 1
    print(
        f"    ✓ required-features static gate PASSED — {len(rows)} row(s) "
        f"scanned, {len(findings)} naming an unenableable feature"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
