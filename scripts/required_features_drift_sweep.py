#!/usr/bin/env python3
"""Run the required-features STATIC verdict over EVERY contract repo, not just this one.

`scripts/required_features_static_gate.py` is katgpt-rs-scoped by construction:
its pins live in `scripts/required_features_floors.txt`, whose first line says
"katgpt-rs scope only", and `docs_gate.yml` has a single checkout so it could
never see a sibling. That is the right shape for a per-push CI gate and the
wrong shape for the question "is anyone ELSE shipping a row that names a
feature its package cannot enable?" — which was answered once, by hand, on
2026-09-05 (0 over 1,829 rows in 16 repos) and then had nothing keeping it at
zero.

This is the third instance of one shape in this workspace, and the two before
it both found real defects the moment they were pointed anywhere but here:

    Issue 702  ci_gate_coverage      one repo -> 7 dead workflows
    Issue 725  numbering_drift_sweep one repo -> 35 duplicate numbers,
                                     5 corrupt `.highwater` allocators

An auditor that accepts a repo path and is pointed at exactly one repo for
months makes "a sibling with the defect" and "a sibling nobody looked at"
byte-identical.

Why this verdict and not the other two
--------------------------------------
The report has three verdicts; only this one is free. A row naming a feature
the package cannot enable is decided by the MANIFEST alone — no compiler, under
a second for the whole workspace — and is **never legitimate**: cargo silently
SKIPS such a target in every invocation that does not name it, `--all-features`
and `cargo test --workspace` included, so it reports a green zero forever while
every audit in the `cfg_gated_target_audit.py` family counts it as PROTECTED
because the row EXISTS.

The other two verdicts (is the row SUFFICIENT? does the target build?) need the
compiler, are priced in hours over 1,070 grouped invocations, and are riir-train
`.issues/513` T2/T3. **A green here is not a claim that any row builds.**

Why this is NOT in scripts/docs_gate.sh's CHECKS
-----------------------------------------------
Identical reasoning to `docs_drift_sweep.py` and `numbering_drift_sweep.py`: CI
has one checkout, the siblings are private and simply absent, so this would
either red on every run or derive an EMPTY population and print a confident
green over zero repos.

    this script                        workstation, on demand, every contract repo
    required_features_static_gate.py   CI, per-push (docs_gate.sh), katgpt-rs only

Vocabulary vs population
------------------------
The population (which repos exist) is DERIVED — BOUNDARY.md + a `.git` dir,
never typed, per Issue 703. The expectations are COMMITTED, in
`scripts/required_features_drift_floors.txt`, because deriving both from one
walk is what makes a cross-repo gate permanently green.

Two floors, not one, and the second is the one that earns its keep
------------------------------------------------------------------
`max_invalid = 0` is a WALL here, not a ratchet — unlike the numbering sweep,
there is no standing backlog to tolerate, and this defect class is never
legitimate. But a ceiling is only green over what the instrument can SEE, so it
needs a floor underneath it, and `min_rows` alone is not enough: **seven of the
sixteen repos legitimately have ZERO rows**, so their row floor is 0 and cannot
detect anything. `min_manifests` is the floor that still bites there —
`parse_rows` swallows `TOMLDecodeError`/`OSError` with a bare `continue`, so a
manifest that stops parsing is indistinguishable from one carrying no rows, and
in a 0-row repo that is indistinguishable from the repo itself.

Report + gate. Exit 0 clean, 1 on drift above the pins, **2 if the instrument
itself is untrustworthy** (selftest failure) — an unreliable instrument is not
the same finding as drift and must not be reported as one.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# DRY: the parse and the validity model are the report's, so the sweep, the
# per-push gate and the report can never disagree about what a row is or which
# feature names are enableable.
from required_features_build_audit import parse_rows, static_invalid  # noqa: E402
from cfg_gated_target_audit import derive_repos, manifests  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT.parent
PINS = Path(__file__).resolve().parent / "required_features_drift_floors.txt"
# The per-push gate's own pins. This sweep re-states katgpt-rs's row floor, so
# the two files can drift apart — the exact hazard `docs_gate_paths_sync.py`
# exists for one workflow over. Asserted below rather than trusted.
LOCAL_PINS = Path(__file__).resolve().parent / "required_features_floors.txt"


def parse_pins(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"malformed pin row (want 4 fields): {raw!r}")
        repo, mm, mr, mi = parts
        rows[repo] = {
            "min_manifests": int(mm),
            "min_rows": int(mr),
            "max_invalid": int(mi),
        }
    return rows


def local_min_rows(path: Path) -> int | None:
    """`min_rows_scanned` out of the per-push gate's pins, for the sync assert."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("min_rows_scanned"):
            _, _, value = line.partition("=")
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


def audit(repo: Path) -> dict:
    """One repo -> its invalid rows + the two populations that produced them."""
    n_manifests = 0
    n_unparseable = 0
    for m in manifests(repo):
        try:
            tomllib.loads(m.read_text(encoding="utf-8", errors="replace"))
            n_manifests += 1
        except (tomllib.TOMLDecodeError, OSError):
            # Counted, never silently skipped: this is precisely what makes a
            # 0-row repo indistinguishable from a blind walk.
            n_unparseable += 1
    rows = parse_rows(repo)
    invalid = []
    for row in rows:
        for feature in row.features:
            why = static_invalid(feature, row.feats, row.deps)
            if why:
                invalid.append(f"{row.label}  [{feature}] {why}")
    return {
        "n_manifests": n_manifests,
        "n_unparseable": n_unparseable,
        "n_rows": len(rows),
        "invalid": invalid,
    }


def selftest() -> list[str]:
    """Pin that the verdict FIRES, that it does not fire on `dep/feat`, and the
    parsers. Every one of these fails silently otherwise — and the first canary
    of this kind is what caught the shadowing bug that made the per-push gate
    structurally incapable of ever firing (2026-09-05)."""
    import tempfile

    fails = []
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        repo = ws / "fake-repo"
        (repo / "tests").mkdir(parents=True)
        (repo / "BOUNDARY.md").write_text("x")
        (repo / "Cargo.toml").write_text(
            "[package]\nname = 'fake'\nversion = '0.1.0'\n"
            "[features]\nreal = []\n"
            "[dependencies]\nserde = '1'\n"
            "[[test]]\nname = 'ok_row'\nrequired-features = ['real']\n"
            "[[test]]\nname = 'dep_row'\nrequired-features = ['serde/derive']\n"
            "[[test]]\nname = 'bad_row'\nrequired-features = ['nope']\n"
        )
        got = audit(repo)
        # 1. the verdict FIRES on a planted invalid row
        if len(got["invalid"]) != 1:
            fails.append(f"planted invalid row: expected 1 finding, got {got['invalid']}")
        elif "bad_row" not in got["invalid"][0]:
            fails.append(f"planted invalid row: wrong row flagged: {got['invalid'][0]}")
        # 2. and does NOT fire on `dep/feat`, which is valid and was once
        #    modelled as invalid — filing 10 riir-ai benches as dead targets
        if any("dep_row" in r for r in got["invalid"]):
            fails.append("`dep/feat` row reported invalid — it is satisfiable")
        if got["n_rows"] != 3:
            fails.append(f"population: {got['n_rows']} rows, expected 3")
        if got["n_manifests"] != 1 or got["n_unparseable"] != 0:
            fails.append(f"manifest count wrong: {got['n_manifests']}/{got['n_unparseable']}")

        # 3. an unparseable manifest must be COUNTED, not silently dropped —
        #    it is what makes a blind walk look like a repo with no rows.
        (repo / "sub").mkdir()
        (repo / "sub" / "Cargo.toml").write_text("[package\nname = broken")
        got2 = audit(repo)
        if got2["n_unparseable"] != 1:
            fails.append(f"unparseable manifest not counted: {got2}")

        # 4. population derivation: BOUNDARY.md + a .git DIRECTORY, both
        #    required (a worktree's `.git` is a FILE and would double-count).
        (ws / "no-boundary").mkdir()
        (ws / "no-boundary" / ".git").mkdir()
        (ws / "worktree-shaped").mkdir()
        (ws / "worktree-shaped" / "BOUNDARY.md").write_text("x")
        (ws / "worktree-shaped" / ".git").write_text("gitdir: elsewhere")
        if [p.name for p in derive_repos(ws)] != []:
            fails.append("population: admitted a repo with no .git dir")
        (repo / ".git").mkdir()
        if [p.name for p in derive_repos(ws)] != ["fake-repo"]:
            fails.append("population: derivation is not BOUNDARY.md + .git dir")

        # 5. pin parser: 4 fields, comments stripped, arity ENFORCED
        pins = ws / "pins.txt"
        pins.write_text("# c\nrepo-a\t3\t10\t0  # trailing\n\n")
        if parse_pins(pins) != {
            "repo-a": {"min_manifests": 3, "min_rows": 10, "max_invalid": 0}
        }:
            fails.append("pin parse: 4-field row not read correctly")
        pins.write_text("repo-a 1 2\n")
        try:
            parse_pins(pins)
            fails.append("pin parse: 3-field row accepted")
        except ValueError:
            pass
    return fails


def main() -> int:
    fails = selftest()
    if fails:
        print("✗ required-features sweep SELFTEST FAILED — instrument untrustworthy:")
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

    repos = derive_repos(WORKSPACE)
    if not repos:
        print(f"✗ derived population is EMPTY under {WORKSPACE} — refusing to "
              f"report a green over zero repos")
        return 2

    # The one number stated twice in this repo. `docs_gate_paths_sync.py` is
    # here because a hand-duplicated list drifts; so does a hand-duplicated pin.
    local = local_min_rows(LOCAL_PINS)
    if local is None:
        print(f"✗ could not read min_rows_scanned from {LOCAL_PINS.name}")
        return 2
    mine = pins.get(REPO_ROOT.name, {}).get("min_rows")
    if mine != local:
        print(f"✗ pin drift: {PINS.name} says min_rows={mine} for "
              f"{REPO_ROOT.name}, {LOCAL_PINS.name} says min_rows_scanned="
              f"{local}. Same quantity, two files — change both.")
        return 1

    seen = {p.name for p in repos}
    bad = False
    tot_rows = tot_invalid = tot_manifests = tot_unparseable = 0

    for repo in repos:
        got = audit(repo)
        row = pins.get(repo.name)
        tot_rows += got["n_rows"]
        tot_invalid += len(got["invalid"])
        tot_manifests += got["n_manifests"]
        tot_unparseable += got["n_unparseable"]
        flags = []
        if row is None:
            flags.append("UNPINNED — add a row (or it can never red)")
        else:
            if got["n_manifests"] < row["min_manifests"]:
                flags.append(f"manifest FLOOR breached: {got['n_manifests']} "
                             f"< {row['min_manifests']} — a crate was removed, "
                             f"or the walk went blind")
            if got["n_rows"] < row["min_rows"]:
                flags.append(f"row FLOOR breached: {got['n_rows']} < {row['min_rows']}")
            if len(got["invalid"]) > row["max_invalid"]:
                flags.append(f"invalid rows {len(got['invalid'])} > pinned "
                             f"{row['max_invalid']}")
        if got["n_unparseable"]:
            flags.append(f"{got['n_unparseable']} manifest(s) do not parse — "
                         f"their rows are INVISIBLE to this verdict")
        status = "✗" if flags else ("·" if got["invalid"] else "✓")
        print(f"{status} {repo.name:22s} manifests={got['n_manifests']:<4d} "
              f"rows={got['n_rows']:<5d} invalid={len(got['invalid'])}")
        for r in got["invalid"]:
            print(f"      invalid:  {r}")
        for f in flags:
            bad = True
            print(f"      ✗ {f}")

    for name in sorted(set(pins) - seen):
        bad = True
        print(f"✗ {name}: pinned but ABSENT from the derived walk — it was "
              f"retired (drop the row in that commit) or the walk went blind")

    print(f"\n{len(repos)} contract repo(s) · {tot_manifests} manifest(s) "
          f"({tot_unparseable} unparseable) · {tot_rows} required-features row(s) "
          f"· {tot_invalid} invalid")
    # Say the scope at the point of READING, not only in the docstring — the
    # numbering sweep's `dup=0` was read as a claim about `.benchmarks/` for a
    # day, and the number was right while the reader's inference was not.
    print("  scope: this is the FREE static verdict only — a row naming a "
          "feature its package cannot enable. It is NOT a claim that any row "
          "is SUFFICIENT or that any target BUILDS; that needs the compiler "
          "(riir-train .issues/513 T2/T3, ~1,070 grouped cargo invocations).")
    if bad:
        print("✗ required-features sweep FAILED — see the ✗ rows above")
        return 1
    print("✓ required-features sweep PASSED — 0 invalid rows, both floors held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
