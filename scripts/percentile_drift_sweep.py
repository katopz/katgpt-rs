#!/usr/bin/env python3
"""Run the percentile-index gate's ceilings over EVERY contract repo, not just this one.

`scripts/percentile_floor_gate.py` is katgpt-rs-scoped by construction: its
pins live in `scripts/percentile_floors.txt`, whose first line says "katgpt-rs
scope only", and `docs_gate.yml` has a single checkout so it could never see a
sibling. That is the right shape for a per-push CI gate and the wrong shape for
"is anyone ELSE reporting a max under a percentile's name?"

The workspace answer to that question was hard-won and is being held by
nothing. On 2026-09-03 the audit found **12 DEGENERATE sites** and four
sibling owners fixed all of them the same day (riir-ai `03a91ed59` swept 10,
riir-mmorpg-examples `ee9da24` the one DEGENERATE-ASSERTED site,
riir-game-sdk `f896bca`, riir-chain `7f3a3910`). katgpt-rs has gated its own
zero since; the other fifteen repos have gated nothing, so the next
`sorted[(n as f64 * 0.99) as usize]` to land in a sibling bench is invisible
until somebody re-runs the report by hand.

This is the fourth instance of one shape in this workspace, and the first two
found real defects the moment they were pointed anywhere but here:

    Issue 702  ci_gate_coverage              one repo -> 7 dead workflows
    Issue 725  numbering_drift_sweep         one repo -> 35 duplicate numbers
    2026-09-06 required_features_drift_sweep one repo -> clean, and pinned there
    this file  percentile_drift_sweep        one repo -> clean, and pinned there

Why the population floor is TWO numbers here
--------------------------------------------
`max_degenerate = 0` is green over whatever the auditor's vocabulary can NAME,
so a tokenizer regression takes the count to ~0 and every ceiling passes,
indistinguishable from a clean repo. `percentile_floors.txt` already carries
`min_sites_scanned` for that reason.

A per-repo site floor cannot do that job alone across the workspace: **seven of
the sixteen repos have ZERO percentile sites**, so their site floor is 0 and
detects nothing at all. `min_rs_files` — the size of the walk that produced the
sites — still bites there, and it is the quantity a `walk_rs` regression
actually moves. Same argument as `required_features_drift_sweep.py`'s
`min_manifests`, one instrument over.

Both floors are deliberately SLACK against churn and TIGHT against blindness,
per the reasoning in `percentile_floors.txt`: a repair campaign legitimately
SHRINKS the site count (consolidating ten inline index computations behind one
correct helper removes nine sites — that is what took the workspace 130 -> 114),
and a floor that ratchets up to the last measurement would red the next such
refactor and teach whoever hits it that the gate is noise. A vocabulary or
scoping regression drops these by an order of magnitude, not by a third.

Ceilings are a WALL, not a ratchet: all four classes measure 0 in all 16 repos
(2026-09-06), there is no standing backlog to tolerate, and none of the four is
legitimate.

Why this is NOT in scripts/docs_gate.sh's CHECKS
-----------------------------------------------
Identical to the other three sweeps: CI has one checkout, the siblings are
private and simply absent, so this would either red on every run or derive an
EMPTY population and print a confident green over zero repos. It also costs
~70s, against the docs gate's ~3s budget.

    this script                  workstation, on demand, every contract repo
    percentile_floor_gate.py     CI, per-push (docs_gate.sh), katgpt-rs only

Exit 0 clean, 1 on drift above the pins, **2 if the instrument itself is
untrustworthy** — an unreliable instrument is not the same finding as drift.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# DRY: the tokenizer, the classification and the selftest are the report's, so
# the sweep, the per-push gate and the report can never disagree about what a
# DEGENERATE site is.
import percentile_index_audit as pia  # noqa: E402

REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
PINS = HERE / "percentile_drift_floors.txt"
# The per-push gate's own pins. This sweep re-states katgpt-rs's site floor, so
# the two files can drift apart — asserted below rather than trusted, exactly
# as `docs_gate_paths_sync.py` does for the two trigger lists.
LOCAL_PINS = HERE / "percentile_floors.txt"

FIELDS = ("min_rs_files", "min_sites", "max_degenerate",
          "max_degenerate_asserted", "max_weak_asserted", "max_trunc_var")


def parse_pins(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 1 + len(FIELDS):
            raise ValueError(
                f"malformed pin row (want {1 + len(FIELDS)} fields): {raw!r}")
        rows[parts[0]] = dict(zip(FIELDS, (int(v) for v in parts[1:])))
    return rows


def local_min_sites(path: Path) -> int | None:
    """`min_sites_scanned` out of the per-push gate's pins, for the sync assert."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("min_sites_scanned"):
            _, _, value = line.partition("=")
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


def audit(repo: Path) -> dict:
    """One repo -> the four gated classes + BOTH populations that produced them."""
    n_rs = 0
    findings = []
    for f in pia.walk_rs(str(repo)):
        n_rs += 1
        findings += pia.audit_file(f, os.path.relpath(f, repo))
    t = pia.tally(findings)
    return {
        "n_rs": n_rs,
        "n_sites": len(t["sites"]),
        "degenerate": t["degenerate"],
        "degenerate_asserted": t["degenerate_asserted"],
        "weak_asserted": t["weak_asserted"],
        "trunc_var": t["trunc_var"],
    }


def selftest() -> list[str]:
    """Pin that the verdict FIRES, that the WEAK/TRUNC_VAR asymmetry survives,
    and the parsers. Each fails SILENTLY otherwise, and a silent failure here
    reports a clean workspace."""
    import tempfile

    fails = []
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        repo = ws / "fake-repo"
        (repo / "benches").mkdir(parents=True)
        (repo / "BOUNDARY.md").write_text("x")
        (repo / ".git").mkdir()

        # A planted DEGENERATE site: n=100 at p99 indexes 99 == n-1, the MAX.
        (repo / "benches" / "b.rs").write_text(
            "fn main() {\n"
            "    let n = 100;\n"
            "    let mut sorted = vec![0u64; n];\n"
            "    sorted.sort();\n"
            "    let p99 = sorted[(n as f64 * 0.99) as usize];\n"
            "    assert!(p99 < 5_000);\n"
            "}\n"
        )
        got = audit(repo)
        if len(got["degenerate"]) != 1:
            fails.append(f"planted degenerate site: expected 1, got "
                         f"{[r['text'] for r in got['degenerate']]}")
        if len(got["degenerate_asserted"]) != 1:
            fails.append("planted site is load-bearing but did not count as asserted")
        if got["n_sites"] != 1 or got["n_rs"] != 1:
            fails.append(f"population wrong: {got['n_rs']} files, {got['n_sites']} sites")

        # A CONTROL: the correct nearest-rank form must produce no finding, or
        # the sweep reds on every correct repair and gets switched off.
        (repo / "benches" / "b.rs").write_text(
            "fn main() {\n"
            "    let n = 100;\n"
            "    let mut sorted = vec![0u64; n];\n"
            "    sorted.sort();\n"
            "    let p99 = sorted[((n as f64 * 0.99).ceil() as usize) - 1];\n"
            "    assert!(p99 < 5_000);\n"
            "}\n"
        )
        if audit(repo)["degenerate"]:
            fails.append("control: correct ceil()-1 nearest rank reported as DEGENERATE")

        # The tally asymmetry is load-bearing and invisible if it inverts:
        # WEAK counts only when asserted, TRUNC_VAR regardless.
        t = pia.tally([
            {"verdict": pia.WEAK, "asserted": False},
            {"verdict": pia.WEAK, "asserted": True},
            {"verdict": pia.TRUNC_VAR, "asserted": False},
            {"verdict": pia.DEGENERATE, "asserted": False},
        ])
        if len(t["weak_asserted"]) != 1:
            fails.append("tally: WEAK must be counted only when asserted")
        if len(t["trunc_var"]) != 1:
            fails.append("tally: TRUNC_VAR must be counted regardless of asserted")
        if len(t["degenerate"]) != 1 or t["degenerate_asserted"]:
            fails.append("tally: DEGENERATE / DEGENERATE-ASSERTED split broken")

        # population derivation: BOUNDARY.md + a .git DIRECTORY, both required
        (ws / "no-boundary").mkdir()
        (ws / "no-boundary" / ".git").mkdir()
        (ws / "worktree-shaped").mkdir()
        (ws / "worktree-shaped" / "BOUNDARY.md").write_text("x")
        (ws / "worktree-shaped" / ".git").write_text("gitdir: elsewhere")
        if pia.repos(str(ws)) != ["fake-repo"]:
            fails.append(f"population derivation wrong: {pia.repos(str(ws))}")

        # pin parser: arity ENFORCED, comments stripped
        pins = ws / "pins.txt"
        pins.write_text("# c\nrepo-a 10 5 0 0 0 0  # trailing\n\n")
        if parse_pins(pins) != {"repo-a": {"min_rs_files": 10, "min_sites": 5,
                                           "max_degenerate": 0,
                                           "max_degenerate_asserted": 0,
                                           "max_weak_asserted": 0,
                                           "max_trunc_var": 0}}:
            fails.append("pin parse: 7-field row not read correctly")
        pins.write_text("repo-a 1 2 3\n")
        try:
            parse_pins(pins)
            fails.append("pin parse: short row accepted")
        except ValueError:
            pass
    return fails


def main() -> int:
    # The report's own selftest first: it exits 2 on failure. Without it a
    # tokenizer regression takes every count to zero and this sweep certifies
    # the workspace clean on the strength of an instrument that has gone blind.
    pia.selftest()

    fails = selftest()
    if fails:
        print("✗ percentile sweep SELFTEST FAILED — instrument untrustworthy:")
        for f in fails:
            print(f"    {f}")
        return 2

    if not PINS.is_file():
        print(f"✗ pins file missing: {PINS}")
        return 2
    try:
        pins = parse_pins(PINS)
    except ValueError as e:
        print(f"✗ pins file unreadable: {e}")
        return 2
    if not pins:
        print("✗ pins file declares NO repos — an empty expectation set is refused")
        return 2

    names = pia.repos(str(WORKSPACE))
    if not names:
        print(f"✗ derived population is EMPTY under {WORKSPACE} — refusing to "
              f"report a green over zero repos")
        return 2

    local = local_min_sites(LOCAL_PINS)
    if local is None:
        print(f"✗ could not read min_sites_scanned from {LOCAL_PINS.name}")
        return 2
    mine = pins.get(REPO_ROOT.name, {}).get("min_sites")
    if mine != local:
        print(f"✗ pin drift: {PINS.name} says min_sites={mine} for "
              f"{REPO_ROOT.name}, {LOCAL_PINS.name} says min_sites_scanned="
              f"{local}. Same quantity, two files — change both.")
        return 1

    bad = False
    tot = {"n_rs": 0, "n_sites": 0, "degenerate": 0,
           "degenerate_asserted": 0, "weak_asserted": 0, "trunc_var": 0}

    for name in names:
        got = audit(WORKSPACE / name)
        row = pins.get(name)
        tot["n_rs"] += got["n_rs"]
        tot["n_sites"] += got["n_sites"]
        for k in ("degenerate", "degenerate_asserted", "weak_asserted", "trunc_var"):
            tot[k] += len(got[k])
        flags = []
        if row is None:
            flags.append("UNPINNED — add a row (or it can never red)")
        else:
            if got["n_rs"] < row["min_rs_files"]:
                flags.append(f"walk FLOOR breached: {got['n_rs']} .rs files "
                             f"< {row['min_rs_files']} — code was removed, or "
                             f"walk_rs went blind")
            if got["n_sites"] < row["min_sites"]:
                flags.append(f"site FLOOR breached: {got['n_sites']} < "
                             f"{row['min_sites']} — a repair campaign, or the "
                             f"tokenizer stopped naming these shapes")
            for cls in ("degenerate", "degenerate_asserted",
                        "weak_asserted", "trunc_var"):
                if len(got[cls]) > row[f"max_{cls}"]:
                    flags.append(f"{cls} {len(got[cls])} > pinned {row[f'max_{cls}']}")
        findings = (got["degenerate"] + got["weak_asserted"] + got["trunc_var"])
        status = "✗" if flags else ("·" if findings else "✓")
        print(f"{status} {name:22s} rs={got['n_rs']:<5d} sites={got['n_sites']:<4d} "
              f"deg={len(got['degenerate'])} deg_asserted="
              f"{len(got['degenerate_asserted'])} weak_asserted="
              f"{len(got['weak_asserted'])} trunc_var={len(got['trunc_var'])}")
        for r in findings:
            if r["verdict"] == pia.TRUNC_VAR:
                # p is a parameter here, so p/n/idx/support are all None and
                # printing them tells the reader nothing. The line IS the finding.
                print(f"      {r['file']}:{r['line']}  {r['text']}")
            else:
                print(f"      {r['file']}:{r['line']}  p={r['p']} n={r['n']} "
                      f"idx={r['idx']} support={r['support']} "
                      f"asserted={r['asserted']}")
        for f in flags:
            bad = True
            print(f"      ✗ {f}")

    for name in sorted(set(pins) - set(names)):
        bad = True
        print(f"✗ {name}: pinned but ABSENT from the derived walk — it was "
              f"retired (drop the row in that commit) or the walk went blind")

    print(f"\n{len(names)} contract repo(s) · {tot['n_rs']} .rs file(s) · "
          f"{tot['n_sites']} percentile site(s) · {tot['degenerate']} degenerate "
          f"({tot['degenerate_asserted']} asserted) · {tot['weak_asserted']} "
          f"weak-asserted · {tot['trunc_var']} trunc-var")
    # State the scope where it is READ, not only in the docstring: the numbering
    # sweep's `dup=0` was read as a claim about `.benchmarks/` for a day.
    print("  scope: UNRESOLVED sites are NOT counted clean — a sample count no "
          "static pass can reach (a runtime length, a fn parameter) needs a "
          "per-site read, and that bucket is where findings hide. This sweep "
          "gates the four DECIDABLE classes only.")
    if bad:
        print("✗ percentile sweep FAILED — see the ✗ rows above")
        print("    A 'p99' whose index is n-1 IS the max. Use nearest rank")
        print("    (ceil(p*n)-1) and report tail support, or drop the column")
        print("    when the sample count cannot support the quantile at all.")
        return 1
    print("✓ percentile sweep PASSED — 0 in all four classes, both floors held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
