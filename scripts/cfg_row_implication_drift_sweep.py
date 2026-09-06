#!/usr/bin/env python3
"""Run the cfg-row-implication verdict over EVERY contract repo, not just this one.

`scripts/cfg_row_implication_gate.py` is katgpt-rs-scoped by construction: its
pins live in `cfg_row_implication_floors.txt` and `docs_gate.yml` has a single
checkout, so it can never see a sibling. Right shape for a per-push gate; wrong
shape for "is anyone ELSE shipping a row that BUILDS and compiles its target to
nothing?"

This is the sixth member of a family whose every previous member found real
defects the moment it was pointed anywhere but here:

    Issue 702  ci_gate_coverage            one repo -> 7 dead workflows
    Issue 725  numbering_drift_sweep       one repo -> 35 duplicate numbers
    Issue 728  cfg_gated_drift_sweep       one repo -> 12 silent load-bearing gates

An auditor pointed at exactly one repo for months makes "a sibling with the
defect" and "a sibling nobody looked at" byte-identical.

## What this verdict is, and why it is not the compiler's

A row whose feature closure (WITH defaults) does not satisfy its target's
leading `#![cfg]` compiles that target to NOTHING. The harness prints
`running 0 tests / test result: ok. 0 passed` and cargo exits 0 — so
`required_features_build_audit.py` reports **BUILDS** and is right, and
`cfg_gated_target_audit.py` counts the reader as protected because the row
EXISTS. Measured: riir-train `054a39a2` fixed one such row and took a target
from 0 passed to 1 passed, an assertion that had never executed at any
revision.

So a green from the compiler sweep is not a claim about this, and a green here
is not a claim that any row builds. The two verdicts are independent and both
are needed.

## Why NOT in docs_gate.sh's CHECKS

Identical reasoning to every other sweep here: CI has one checkout, the
siblings are simply absent, so this would either red on every run or derive an
EMPTY population and print a confident green over zero repos.

    this script                    workstation, on demand, every contract repo
    cfg_row_implication_gate.py    CI, per-push (docs_gate.sh), katgpt-rs only

## Vocabulary vs population

Population DERIVED (BOUNDARY.md + a `.git` dir, never typed). Expectations
COMMITTED, in `cfg_row_implication_drift_floors.txt` — deriving both from one
walk is what makes a cross-repo gate permanently green.

## Two ceilings and two floors, and the floors are the load-bearing half

`max_empty` is a WALL at 0 wherever the measured value is 0 and a RATCHET at
the measured backlog otherwise (riir-ai carries one open instance, Issue 513
instance 7). `max_unresolved` is a ratchet at each repo's measured value: an
`any(feature = ...)` predicate is legitimate and simply not rulable here.

Both go green over whatever the instrument can NAME, so `min_rows` and
`min_with_cfg` sit underneath. `min_with_cfg` is the one that earns its keep —
`leading_inner_cfgs` returns `[]` for an unreadable file and for a file with no
cfg, and an early cut of it silently skipped cargo's DIRECTORY target form
(`tests/<name>/main.rs`). A source-scanner narrowing takes the cfg population
toward 0 and both ceilings pass, indistinguishable from a clean repo.

Exit 0 clean · 1 drift above the pins · 2 the instrument is untrustworthy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# DRY: the classification is the report's, so the sweep, the per-push gate and
# the report can never disagree about what EMPTY-AT-ROW means.
import cfg_row_implication_audit as cria  # noqa: E402
from cfg_gated_target_audit import derive_repos  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT.parent
PINS = Path(__file__).resolve().parent / "cfg_row_implication_drift_floors.txt"
# The per-push gate's own pins. This sweep re-states katgpt-rs's numbers, so the
# two files can drift apart — the hazard docs_gate_paths_sync.py exists for one
# workflow over. Asserted below rather than trusted.
LOCAL_PINS = Path(__file__).resolve().parent / "cfg_row_implication_floors.txt"

FIELDS = ("min_rows", "min_with_cfg", "max_empty", "max_unresolved")


def parse_pins(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"malformed pin row (want 5 fields): {raw!r}")
        rows[parts[0]] = dict(zip(FIELDS, (int(p) for p in parts[1:])))
    return rows


def local_pin(path: Path, key: str) -> int | None:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith(key):
            _, _, value = line.partition("=")
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


def main(argv: list[str]) -> int:
    # Prints carry glyphs the Windows locale codecs cannot encode (checked
    # 2026-09-06 on cp874: check/cross/middot/arrow FAIL, em-dash OK); keep the
    # locale encoding and degrade only the fatal chars to escapes -- the
    # staged_set_audit house pattern (utf-8 pinning would mojibake legacy consoles).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (embedded / detached); keep old behavior
    # The report's selftest exits 2 on its own — run it before anything else so
    # an untrustworthy instrument is never reported as drift.
    cria.selftest()

    if not PINS.is_file():
        print(f"✗ pins file missing: {PINS}")
        return 2
    try:
        pins = parse_pins(PINS)
    except ValueError as e:
        print(f"✗ pins file unparseable: {e}")
        return 2
    if not pins:
        # An empty allowlist would make every repo "unpinned" and the sweep
        # would report a confident nothing. Refuse, as its siblings do.
        print("✗ pins file declares no repos — refusing to run vacuously")
        return 2

    repos = derive_repos(Path(argv[0]) if argv else WORKSPACE)
    if len(repos) < 2:
        print(f"✗ derived only {len(repos)} repo(s) — this sweep is cross-repo "
              f"by definition and a single-checkout run would be vacuous")
        return 2

    fails: list[str] = []
    empties: list[cria.Finding] = []
    tot = {k: 0 for k in ("rows", "with_cfg", "empty", "unresolved")}

    print(f"{'repo':<24} {'rows':>6} {'#![cfg]':>8} {'EMPTY':>6} {'UNRES':>6}   pins")
    for repo in sorted(repos, key=lambda p: p.name):
        found = cria.audit_repo(repo)
        n_rows = len(found)
        with_cfg = sum(1 for f in found if f.verdict != cria.NO_CFG)
        empty = sum(1 for f in found if f.verdict == cria.EMPTY)
        unres = sum(1 for f in found if f.verdict == cria.UNRESOLVED)
        empties += [f for f in found if f.verdict == cria.EMPTY]
        tot["rows"] += n_rows
        tot["with_cfg"] += with_cfg
        tot["empty"] += empty
        tot["unresolved"] += unres

        pin = pins.get(repo.name)
        if pin is None:
            fails.append(f"{repo.name}: no pin row — a new repo must be pinned "
                         f"deliberately, not defaulted to permissive")
            note = "UNPINNED"
        else:
            note = "ok"
            if empty > pin["max_empty"]:
                fails.append(f"{repo.name}: EMPTY-AT-ROW {empty} > pinned {pin['max_empty']}")
                note = "DRIFT"
            if unres > pin["max_unresolved"]:
                fails.append(f"{repo.name}: UNRESOLVED {unres} > pinned {pin['max_unresolved']}")
                note = "DRIFT"
            if n_rows < pin["min_rows"]:
                fails.append(f"{repo.name}: only {n_rows} rows < floor {pin['min_rows']} "
                             f"— the manifest walk shrank, so the ceilings are vacuous")
                note = "DRIFT"
            if with_cfg < pin["min_with_cfg"]:
                fails.append(f"{repo.name}: only {with_cfg} rows carry a leading #![cfg] "
                             f"< floor {pin['min_with_cfg']} — the source scanner narrowed")
                note = "DRIFT"
        print(f"{repo.name:<24} {n_rows:>6} {with_cfg:>8} {empty:>6} {unres:>6}   {note}")

    # The katgpt-rs row here and the per-push gate's own pins name the same
    # quantities. Hand-duplicated values drift; assert rather than trust.
    local_rows = local_pin(LOCAL_PINS, "min_rows_scanned")
    local_cfg = local_pin(LOCAL_PINS, "min_with_cfg")
    mine = pins.get("katgpt-rs")
    if mine is not None:
        if local_rows is not None and local_rows != mine["min_rows"]:
            fails.append(f"pin desync: katgpt-rs min_rows {mine['min_rows']} here vs "
                         f"min_rows_scanned {local_rows} in {LOCAL_PINS.name}")
        if local_cfg is not None and local_cfg != mine["min_with_cfg"]:
            fails.append(f"pin desync: katgpt-rs min_with_cfg {mine['min_with_cfg']} here "
                         f"vs {local_cfg} in {LOCAL_PINS.name}")

    print(f"\n{len(repos)} repos · {tot['rows']} rows · {tot['with_cfg']} with a leading "
          f"#![cfg] · {tot['empty']} EMPTY-AT-ROW · {tot['unresolved']} UNRESOLVED")
    if empties:
        print("\nEMPTY-AT-ROW — the row builds, and compiles the target to NOTHING:")
        for f in empties:
            print(f"  {f.row.label}")
            print(f"      row={f.row.features}  needs={sorted(f.needs)}  "
                  f"MISSING={sorted(f.missing)}")
    if fails:
        print("\n✗ cfg-row-implication drift sweep FAILED:")
        for f in fails:
            print(f"    {f}")
        return 1
    print("\n✓ cfg-row-implication drift sweep PASSED — every repo within its pins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
