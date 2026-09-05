#!/usr/bin/env python3
"""GATE: the SIX independent "which repos are contract repos" predicates must agree.

Every cross-repo instrument in this workspace derives its own population — a
root `BOUNDARY.md` **and** a `.git` DIRECTORY — and six of them do it with six
separate implementations:

    cfg_gated_target_audit.derive_repos      (also used by the required-features
                                              audit and the cfg-gated sweep)
    numbering_drift_sweep.contract_repos
    percentile_index_audit.repos
    ci_gate_coverage.derive_repos
    skill_repo_set_gate.derive_repos
    suite_membership_audit.derive_repos

They all agree today (measured 2026-09-06: 16 repos, identical, and equal to
`scripts/repo_set.txt`). Nothing asserted that, and the failure is silent in
the worst way: if ONE predicate drifts, that one instrument quietly audits a
different set of repos and still prints a confident green over it. The
workspace has already paid for this once — three instruments were found
covering 7, 12 and 15 of 18 repos, each reporting cleanly on its own slice.

This is `docs_gate_paths_sync.py` one axis over: a hand-duplicated *value*
drifts, and so does a hand-duplicated *predicate*.

## Why this can run in CI, when none of the sweeps can

The sweeps cannot, because CI has a single checkout: they would derive an empty
population and print a confident green over zero repos. This gate does not test
the POPULATION, it tests the PREDICATE — against a synthetic workspace built in
a temp dir, containing every case the real one distinguishes:

    good, also-good   BOUNDARY.md + a .git DIRECTORY   -> INCLUDED
    no-boundary       .git dir, no BOUNDARY.md         -> excluded
    no-git            BOUNDARY.md, no .git             -> excluded
    worktree-shaped   BOUNDARY.md + a .git FILE        -> excluded

The last one is not hypothetical and is why the `.git` test must be a
DIRECTORY test: a throwaway worktree's `.git` is a FILE, and a worktree of a
repo already in the walk would otherwise be counted twice. That trap is
documented in `scripts/repo_set.txt`'s own derivation and in three of the six
docstrings — which is exactly the kind of invariant that survives in comments
and dies in code.

The real-workspace cross-check runs too, but only when the walk finds more than
one repo (i.e. on a workstation). It is REPORTED either way, never silently
skipped — a gate that skips without saying so is the vacuous green this family
exists to refuse.

Exit 0 clean · 1 on disagreement · 2 if the gate cannot import a predicate.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
REPO_SET = HERE / "repo_set.txt"

# (label, module, attribute). Kept as DATA so adding a seventh instrument is a
# one-line change here rather than a seventh silent divergence.
PREDICATES = (
    ("cfg_gated_target_audit.derive_repos", "cfg_gated_target_audit", "derive_repos"),
    ("numbering_drift_sweep.contract_repos", "numbering_drift_sweep", "contract_repos"),
    ("percentile_index_audit.repos", "percentile_index_audit", "repos"),
    ("ci_gate_coverage.derive_repos", "ci_gate_coverage", "derive_repos"),
    ("skill_repo_set_gate.derive_repos", "skill_repo_set_gate", "derive_repos"),
    ("suite_membership_audit.derive_repos", "suite_membership_audit", "derive_repos"),
)


def load() -> list[tuple[str, object]]:
    import importlib

    out = []
    for label, mod, attr in PREDICATES:
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001 — any import failure is fatal here
            print(f"✗ cannot import {mod}: {e!r}")
            raise SystemExit(2)
        fn = getattr(m, attr, None)
        if fn is None:
            print(f"✗ {mod} has no {attr} — the predicate was renamed or removed; "
                  f"update PREDICATES in {Path(__file__).name}")
            raise SystemExit(2)
        out.append((label, fn))
    return out


def call(fn, root: Path) -> list[str]:
    """Normalise: some take a Path, some a str; some return Paths, some names."""
    try:
        got = fn(root)
    except TypeError:
        got = fn(str(root))
    return sorted(p.name if isinstance(p, Path) else str(p) for p in got)


def build_synthetic(ws: Path) -> list[str]:
    """Every case the real walk distinguishes. Returns the expected answer."""
    for name in ("good", "also-good"):
        (ws / name).mkdir()
        (ws / name / "BOUNDARY.md").write_text("x")
        (ws / name / ".git").mkdir()
    (ws / "no-boundary").mkdir()
    (ws / "no-boundary" / ".git").mkdir()
    (ws / "no-git").mkdir()
    (ws / "no-git" / "BOUNDARY.md").write_text("x")
    # A worktree's `.git` is a FILE. Admitting it double-counts a repo already
    # in the walk — the trap the DIRECTORY test exists for.
    (ws / "worktree-shaped").mkdir()
    (ws / "worktree-shaped" / "BOUNDARY.md").write_text("x")
    (ws / "worktree-shaped" / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x")
    # A plain file must not be mistaken for a repo directory.
    (ws / "BOUNDARY.md").write_text("x")
    return ["also-good", "good"]


def main() -> int:
    preds = load()
    bad = False

    # ── half 1: the PREDICATE, on a synthetic workspace. Runs everywhere. ──
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        expected = build_synthetic(ws)
        print(f"▸ predicate agreement over a synthetic workspace "
              f"(expected {expected}):")
        for label, fn in preds:
            got = call(fn, ws)
            if got == expected:
                print(f"    ✓ {label}")
                continue
            bad = True
            print(f"    ✗ {label}: got {got}")
            for extra in sorted(set(got) - set(expected)):
                why = {
                    "no-boundary": "admitted a dir with NO BOUNDARY.md",
                    "no-git": "admitted a dir with no .git at all",
                    "worktree-shaped": "admitted a WORKTREE (.git is a FILE) — "
                                       "this double-counts a repo already in the walk",
                }.get(extra, "admitted an unexpected entry")
                print(f"        + {extra}: {why}")
            for miss in sorted(set(expected) - set(got)):
                print(f"        - {miss}: REJECTED a valid contract repo")

    # ── half 2: the real workspace. Workstation-only, and SAID so. ──
    real = {label: call(fn, WORKSPACE) for label, fn in preds}
    sizes = {len(v) for v in real.values()}
    n = max(sizes)
    if n <= 1:
        print(f"▸ real-workspace cross-check SKIPPED — the walk under "
              f"{WORKSPACE} found {n} repo(s), so this is a single-checkout "
              f"environment (CI). The predicate half above is the verdict; the "
              f"population half is workstation-only by construction.")
    else:
        print(f"▸ real-workspace cross-check — {n} repo(s) under {WORKSPACE}:")
        base_label, base = next(iter(real.items()))
        for label, got in real.items():
            if got == base:
                continue
            bad = True
            print(f"    ✗ {label} differs from {base_label}: "
                  f"only-here={sorted(set(got) - set(base))} "
                  f"missing={sorted(set(base) - set(got))}")
        if not bad:
            print(f"    ✓ all {len(real)} predicates agree")
        # The committed vocabulary must match the derived population.
        if REPO_SET.is_file():
            committed = sorted(
                l.strip() for l in REPO_SET.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.lstrip().startswith("#")
            )
            if committed != base:
                bad = True
                print(f"    ✗ {REPO_SET.name} disagrees with the derived walk: "
                      f"only-in-file={sorted(set(committed) - set(base))} "
                      f"missing-from-file={sorted(set(base) - set(committed))}")
            else:
                print(f"    ✓ {REPO_SET.name} matches ({len(committed)} repos)")

    if bad:
        print("✗ population sync gate FAILED — the cross-repo instruments do "
              "NOT all audit the same set of repos")
        return 1
    print(f"✓ population sync gate PASSED — {len(preds)} predicates agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
