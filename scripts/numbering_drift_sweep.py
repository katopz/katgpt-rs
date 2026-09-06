#!/usr/bin/env python3
"""Run the numbering gate's checks over EVERY contract repo, not just this one.

`scripts/numbering_gate.py` is katgpt-rs-scoped by construction: its pins live
in `scripts/numbering_floors.txt`, which says "katgpt-rs scope only" in its
first line, and `docs_gate.yml` has a single checkout so it could never see a
sibling. That is the right shape for a per-push CI gate and the wrong shape for
the question "is anyone else allocating numbers twice?" — which nothing had
ever asked. Measured the first time it was asked (2026-09-05):

    32 tracked duplicate numbers across 4 sibling repos, in ALLOCATOR-SERIAL
    directories, while katgpt-rs — the one repo with a gate — had zero.
     5 `.highwater` files that are not integers at all, every one of them
       `echo -n <N> > .highwater` writing its own flag into the file, which
       DISARMS the above-highwater check for that directory (Issue 725).

This is the same shape as Issue 702 one instrument over: an auditor that
accepts a repo path, pointed at exactly one repo for months, so a sibling with
the defect is indistinguishable from a sibling nobody looked at.

Why this is NOT in scripts/docs_gate.sh's CHECKS
------------------------------------------------
Identical reasoning to scripts/docs_drift_sweep.py: CI has one checkout, the
siblings are private and simply absent, so this would either red on every run
or derive an empty population and print a confident green over zero repos.

    this script              workstation, on demand, every contract repo
    numbering_gate.py        CI, per-push (docs_gate.sh), katgpt-rs only

Vocabulary vs population
------------------------
The population (which repos exist) is DERIVED — BOUNDARY.md + a `.git` dir,
never typed, per Issue 703. The expectations are COMMITTED, in
`scripts/numbering_drift_floors.txt`, because deriving both from one walk is
what makes a cross-repo gate permanently green.

Ceilings here are a RATCHET, not a wall. 32 duplicates exist today across repos
this session does not own; resolving one is a citation-weight arbitration
(Issue 724 T2's precedent: the file with 27 inbound mentions keeps the number,
the other moves), not a rename. So each repo's ceiling is pinned at its
MEASURED count: a new collision reds immediately, and the standing backlog is
visible in the pins rather than silently tolerated. Lower a pin in the commit
that resolves a collision.

Scope, which is the load-bearing decision
-----------------------------------------
Duplicate + above-highwater checks run ONLY over allocator-serial directories
(`.plans/.issues/.research/.proposals`). `.benchmarks/` and `.docs/` are
excluded for exactly the reason numbering_floors.txt gives: there the leading
number is the OWNING plan/issue and a family per owner is the intended
convention, so "duplicate" is not decidable without knowing which convention a
given file follows — and a gate must not guess.

The MALFORMED check runs over every numbered directory including those two,
because a file that is not an integer is undecidable under no convention.

Report + gate. Exit 0 clean, 1 on drift above the pins, **2 if the instrument
itself is untrustworthy** (selftest failure) — an unreliable instrument is not
the same finding as drift and must not be reported as one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numbering_gate as ng  # noqa: E402  (DRY: one scanner, two cadences)

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT.parent
PINS = Path(__file__).resolve().parent / "numbering_drift_floors.txt"

# DATA, not derived from the tree — see the module docstring's scope section.
SERIAL_DIRS = (".plans", ".issues", ".research", ".proposals")
ALL_DIRS = SERIAL_DIRS + (".benchmarks", ".docs")


def contract_repos(workspace: Path) -> list[Path]:
    """Derived population: a root BOUNDARY.md AND a .git DIRECTORY.

    The `.git` test is a directory test on purpose — a worktree's `.git` is a
    FILE, and a throwaway worktree of a repo already in the walk would be
    counted twice (the trap scripts/repo_set.txt's own derivation documents).
    """
    return sorted(
        (p for p in workspace.iterdir()
         if (p / "BOUNDARY.md").is_file() and (p / ".git").is_dir()),
        key=lambda p: p.name,
    )


def parse_rows(path: Path) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    # pins carry UTF-8 punctuation; the locale codec (cp1252 on Windows) cannot decode it (2026-09-06 4090-box catch)
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"malformed pin row (want 5 fields): {raw!r}")
        repo, mn, dup, above, mal = parts
        rows[repo] = {
            "min_files": int(mn), "max_dup": int(dup),
            "max_above": int(above), "max_malformed": int(mal),
        }
    return rows


def audit(repo: Path) -> dict:
    """One repo -> its three finding classes + the population that produced them."""
    tracked = ng.tracked_paths(repo, list(ALL_DIRS))
    dup, above, malformed = [], [], []
    n_serial = 0
    for dirname in ALL_DIRS:
        by_num, hw, n, hw_bad = ng.scan(repo, dirname, tracked)
        if hw_bad is not None:
            malformed.append(f"{dirname}/.highwater = {hw_bad!r}")
        if dirname not in SERIAL_DIRS:
            continue                       # family convention — not decidable
        n_serial += n
        for num, files in sorted(by_num.items()):
            if sum(1 for _, tr in files if tr) < 2:
                continue                   # untracked = a colleague's WIP
            names = " · ".join(nm for nm, _ in files)
            dup.append(f"{dirname}/{num:03d} ×{len(files)}: {names}")
        if hw is not None and by_num and max(by_num) > hw:
            above.append(f"{dirname}: max {max(by_num)} > .highwater {hw}")
    return {"dup": dup, "above": above, "malformed": malformed, "n_files": n_serial}


def selftest() -> list[str]:
    """Pin the scope split and the row parser. Both fail SILENTLY otherwise."""
    import tempfile

    fails = []
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        # a repo with one duplicate in a SERIAL dir and one in a FAMILY dir
        repo = ws / "fake-repo"
        (repo / ".plans").mkdir(parents=True)
        (repo / ".benchmarks").mkdir()
        for nm in ("001_a.md", "001_b.md"):
            (repo / ".plans" / nm).write_text("x")
            (repo / ".benchmarks" / nm).write_text("x")
        (repo / ".benchmarks" / ng.HIGHWATER).write_text("-n 7\n")
        (repo / "BOUNDARY.md").write_text("x")

        # tracked_paths() shells out to git; an untracked file is not a defect,
        # so a non-repo would report ZERO duplicates and pass vacuously. Pin
        # that the split is what decides, by driving audit() with a stub.
        real = ng.tracked_paths
        ng.tracked_paths = lambda r, d: {
            ".plans/001_a.md", ".plans/001_b.md",
            ".benchmarks/001_a.md", ".benchmarks/001_b.md",
        }
        try:
            got = audit(repo)
        finally:
            ng.tracked_paths = real

        if len(got["dup"]) != 1:
            fails.append(f"scope: expected 1 serial-dir duplicate, got {got['dup']}")
        if any(".benchmarks" in r for r in got["dup"]):
            fails.append("scope: a .benchmarks family number was reported as a duplicate")
        if len(got["malformed"]) != 1 or ".benchmarks" not in got["malformed"][0]:
            fails.append(f"scope: malformed check must cover family dirs too: {got['malformed']}")
        if got["n_files"] != 2:
            fails.append(f"population: counted {got['n_files']} serial files, expected 2")

        # the untracked split must still hold — one tracked copy is not a defect
        ng.tracked_paths = lambda r, d: {".plans/001_a.md"}
        try:
            got2 = audit(repo)
        finally:
            ng.tracked_paths = real
        if got2["dup"]:
            fails.append(f"untracked split broken: {got2['dup']}")

        # population derivation: BOUNDARY.md + a .git DIR, both required
        (ws / "no-boundary").mkdir()
        (ws / "no-boundary" / ".git").mkdir()
        (ws / "worktree-shaped").mkdir()
        (ws / "worktree-shaped" / "BOUNDARY.md").write_text("x")
        (ws / "worktree-shaped" / ".git").write_text("gitdir: elsewhere")
        if [p.name for p in contract_repos(ws)] != []:
            fails.append("population: admitted a repo with no .git dir")
        (repo / ".git").mkdir()
        if [p.name for p in contract_repos(ws)] != ["fake-repo"]:
            fails.append("population: derivation is not BOUNDARY.md + .git dir")

        # row parser: 5 fields, comments stripped, arity enforced
        pins = ws / "pins.txt"
        pins.write_text("# c\nrepo-a\t10\t0\t0\t0  # trailing\n\n")
        if parse_rows(pins) != {"repo-a": {"min_files": 10, "max_dup": 0,
                                           "max_above": 0, "max_malformed": 0}}:
            fails.append("row parse: 5-field row not read correctly")
        pins.write_text("repo-a 1 2 3\n")
        try:
            parse_rows(pins)
            fails.append("row parse: 4-field row accepted")
        except ValueError:
            pass
    return fails


def main() -> int:
    fails = selftest()
    if fails:
        print("✗ numbering sweep SELFTEST FAILED — instrument untrustworthy:")
        for f in fails:
            print(f"    {f}")
        return 2

    if not PINS.is_file():
        print(f"✗ pins file missing: {PINS}")
        return 2
    try:
        pins = parse_rows(PINS)
    except ValueError as e:
        print(f"✗ pins file unreadable: {e}")
        return 2
    if not pins:
        print("✗ pins file declares NO repos — an empty expectation set is refused")
        return 2

    repos = contract_repos(WORKSPACE)
    if not repos:
        print(f"✗ derived population is EMPTY under {WORKSPACE} — refusing to "
              f"report a green over zero repos")
        return 2

    seen = {p.name for p in repos}
    bad = False
    tot_dup = tot_above = tot_mal = 0

    for repo in repos:
        got = audit(repo)
        row = pins.get(repo.name)
        tot_dup += len(got["dup"])
        tot_above += len(got["above"])
        tot_mal += len(got["malformed"])
        flags = []
        if row is None:
            flags.append("UNPINNED — add a row (or it can never red)")
        else:
            if got["n_files"] < row["min_files"]:
                flags.append(f"population FLOOR breached: {got['n_files']} < {row['min_files']}")
            if len(got["dup"]) > row["max_dup"]:
                flags.append(f"duplicates {len(got['dup'])} > pinned {row['max_dup']}")
            if len(got["above"]) > row["max_above"]:
                flags.append(f"stale allocators {len(got['above'])} > pinned {row['max_above']}")
            if len(got["malformed"]) > row["max_malformed"]:
                flags.append(f"malformed allocators {len(got['malformed'])} > pinned {row['max_malformed']}")
        status = "✗" if flags else ("·" if (got["dup"] or got["above"] or got["malformed"]) else "✓")
        print(f"{status} {repo.name:22s} files={got['n_files']:<5d} "
              f"dup={len(got['dup'])} stale={len(got['above'])} malformed={len(got['malformed'])}")
        for r in got["malformed"]:
            print(f"      malformed:  {r}")
        for r in got["above"]:
            print(f"      stale:      {r}")
        for r in got["dup"]:
            print(f"      duplicate:  {r}")
        for f in flags:
            bad = True
            print(f"      ✗ {f}")

    for name in sorted(set(pins) - seen):
        bad = True
        print(f"✗ {name}: pinned but ABSENT from the derived walk — it was "
              f"retired (drop the row in that commit) or the walk went blind")

    print(f"\n{len(repos)} contract repo(s) · {tot_dup} tracked duplicate(s) · "
          f"{tot_above} stale allocator(s) · {tot_mal} malformed allocator(s)")
    # Say the scope at the point of READING, not only in the docstring. A green
    # `dup=0` is green over SERIAL_DIRS only, and riir-ai carried four genuine
    # cross-topic `.benchmarks/` collisions (617/619, resolved 2026-09-05) while
    # this line printed `dup=0` for it — the number was right and the reader's
    # inference from it was not.
    family = "/".join(d for d in ALL_DIRS if d not in SERIAL_DIRS)
    print(f"  scope: duplicate + stale checks cover {'/'.join(SERIAL_DIRS)} ONLY; "
          f"{family} share numbers by OWNER convention (a family per plan/issue "
          f"is intended) and are malformed-checked only — see "
          f"{PINS.name} for the measured reason. A `dup=0` above is NOT a claim "
          f"about {family}.")
    if bad:
        print("✗ numbering sweep FAILED — see the ✗ rows above")
        return 1
    print("✓ numbering sweep PASSED — nothing above its pinned ratchet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
