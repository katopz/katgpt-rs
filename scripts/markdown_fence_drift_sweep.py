#!/usr/bin/env python3
"""Run the unterminated-fence verdict over EVERY contract repo, not just this one.

`scripts/markdown_fence_gate.py` is katgpt-rs-scoped by construction — it walks
`REPO_ROOT` and `docs_gate.yml` has a single checkout, so it could never see a
sibling. That is the right shape for a per-push CI gate and the wrong shape for
"is anybody ELSE about to render half a document as code?"

The workspace answer to that question was just bought and is being held by
nothing. Issue 756 took the population from **19 files / 611 swallowed lines**
to zero, across three repos and 5048 tracked `.md`:

    katgpt-rs   1bf768cd + ed455885   12      riir-ai    a1a205681 + 6e73d89c5   5
    riir-train  d761a375 + ca21b763    2
    seal-online-remaster 99064c5      1  (the sweep's own first catch, 2026-09-12)

Twelve of those nineteen were in this repo, where a gate now stands. The other
**seven** were in repos where one edit puts them back with no word from any
gate — and the class is silent by construction: a swallowed section still
renders, just as a code listing, so nothing errors and nobody notices until a
parser mis-phases on it. That is how the class was found in the first place
(`rust-optimize/SKILL.md`'s unclosed ```text ate `skill_repo_set_gate.py`'s own
first canary).

This is the sixth instance of one shape in this workspace, and every one of
them found something the moment it was pointed anywhere but here:

    Issue 702  ci_gate_coverage              one repo -> 7 dead workflows
    Issue 725  numbering_drift_sweep         one repo -> 35 duplicate numbers
    2026-09-06 required_features_drift_sweep one repo -> clean, and pinned there
    2026-09-06 percentile_drift_sweep        one repo -> clean, and pinned there
    2026-09-07 trap_sentinel_drift_sweep     one repo -> 1 finding, pinned + proven inert
    this file  markdown_fence_drift_sweep    one repo -> clean at 19 repos / 5058 files;
                                       its first workspace run (2026-09-12, 20 repos after
                                       seal-online-remaster joined the set) caught 1 finding
                                       in the new repo — repaired + floored the same day

Why BOTH pins, and why the walk floor is not redundant
------------------------------------------------------
`max_unterminated = 0` is green over whatever the walk can SEE, and the walk
shells out to `git ls-files` — a regression there takes the population to 0 and
the ceiling passes, indistinguishable from a clean repo. `min_md_files` is the
quantity that actually moves when the walk goes blind, and unlike the trap
sweep's population floor it is non-zero in **every** repo (the smallest,
riir-kat, has 3 tracked `.md`), so one floor does the whole job here.

It is deliberately SLACK against churn (~55-60% of measured) and TIGHT against
blindness: consolidating a `.plans` tree or removing a resolved-issue batch
legitimately shrinks the count, and a floor that ratcheted to the last
measurement would red that cleanup and teach whoever hit it that the sweep is
noise. A walk regression drops these by an order of magnitude, not by a third.

katgpt-rs's floor is NOT free: it must equal `markdown_fence_gate.MIN_FILES`,
and this sweep ASSERTS that rather than trusting it — same quantity, two files,
the pattern `docs_gate_paths_sync.py` uses for the two trigger lists and
`trap_sentinel_drift_sweep.py` for `POPULATION_FLOOR`.

No membership pin, deliberately
-------------------------------
`trap_sentinel_gate.py` pins its set by NAME because a count is not a checksum
over a set. That argument does not apply here: the verdict is DERIVED from the
file's own fence structure, and there is no repaired-file list to lose. Every
way to reintroduce the defect lands on `max_unterminated`; every way to lose
sight of it lands on `min_md_files`.

Why this is NOT in scripts/docs_gate.sh's CHECKS
------------------------------------------------
Identical to the other five sweeps: CI has one checkout, the siblings are
private and simply absent, so this would either red on every run or derive an
EMPTY population and print a confident green over zero repos.

    this script               workstation, on demand, every contract repo
    markdown_fence_gate.py    CI, per-push (docs_gate.sh), katgpt-rs only

⚠ The reported line is the DANGLING fence, NOT necessarily the defect — one
stray fence inverts the pairing of every fence after it, so two of riir-train's
repairs were 495 and 21 lines UPSTREAM of what was reported. Read the first
non-blank body line before editing: code means a closer is missing, prose means
the fence itself is the orphan, and a same-length nested fence (riir-ai
`.issues/094`) needs the OUTER pair widened to ```` — deleting the "extra"
fence leaves the inner code rendering as prose permanently.

Exit 0 clean, 1 on drift above the pins, **2 if the instrument itself is
untrustworthy** — an unreliable instrument is not the same finding as drift.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# DRY: the classifier and the walk are the per-push gate's, so the sweep and
# the gate can never disagree about what "unterminated" means — and the fence
# parser under them both is `skill_repo_set_gate.fenced_blocks`, imported
# rather than re-derived (Issue 755: a second copy of a rule this subtle is a
# second thing to get wrong).
import markdown_fence_gate as mfg  # noqa: E402
from skill_repo_set_gate import derive_repos  # noqa: E402

REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
PINS = HERE / "markdown_fence_drift_floors.txt"

FIELDS = ("min_md_files", "max_unterminated")


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


def selftest() -> list[str]:
    """Pin that the verdict FIRES through the gate's own `unterminated()`, that
    the control does NOT, that the closer-length and nesting rules are live,
    and that the walk's include/exclude boundaries hold. Each fails silently
    otherwise, and a silent failure reports a clean workspace."""
    fails = []

    def measure(files: dict[str, str], gitignore: str = "") -> tuple[list, int]:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            # A real `git init`: `unterminated()` shells out to `git ls-files`
            # and has NO on-disk fallback, so a bare temp dir would measure an
            # empty population and every arm below would pass vacuously. The
            # walk floor asserted by each caller is what catches that.
            subprocess.run(["git", "-C", str(repo), "init", "-q"],
                           capture_output=True, check=True)
            if gitignore:
                (repo / ".gitignore").write_text(gitignore, encoding="utf-8")
            for rel, text in files.items():
                f = repo / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(text, encoding="utf-8")
            return mfg.unterminated(repo)

    # 1. the defect FIRES, with the right line and the right swallow count.
    #    `a.md` is 5 lines; the fence opens on 3, so 2 lines are swallowed.
    found, walked = measure({"a.md": "# T\n\n```rust\nfn main() {}\nstill code\n"})
    if walked != 1:
        fails.append(f"walk found {walked} .md, expected 1 — git ls-files or the "
                     f"untracked arm regressed; every arm below is vacuous")
    elif found != [("a.md", 3, 2)]:
        fails.append(f"planted unterminated fence: got {found}, "
                     f"expected [('a.md', 3, 2)]")

    # 2. CONTROL: a closed fence must produce NO finding, or the sweep reds on
    #    every correct repair and gets switched off.
    found, walked = measure({"a.md": "# T\n\n```rust\nfn main() {}\n```\n\ntail\n"})
    if walked != 1 or found:
        fails.append(f"control: a CLOSED fence produced {found} over {walked} file(s)")

    # 3. the riir-ai/094 shape: a ````-wrapped block quoting a ``` block. The
    #    inner fence must NOT close the outer one, or every nested doc reds.
    found, walked = measure(
        {"a.md": "````md\nquoting:\n```rust\nfn main() {}\n```\n````\n\ntail\n"})
    if walked != 1 or found:
        fails.append(f"nesting: a ````-wrapped ``` block produced {found}")

    # 4. the closer-length rule is LIVE and is the reason arm 3 works: a run
    #    SHORTER than the opener does not close it, so this file IS a finding.
    found, walked = measure({"a.md": "````text\nbody\n```\nmore\n"})
    if walked != 1 or [r[0] for r in found] != ["a.md"]:
        fails.append(f"closer length: a ``` must not close a ````, got {found}")

    # 5. walk boundaries, both directions in ONE measurement — an untracked
    #    file IS walked (the miss that cost this gate its own landing: Issue
    #    756's file was untracked when the gate ran) and a GITIGNORED one is
    #    NOT (the vendored-drop precedent from trap_exit_launder_audit).
    #    TWO visible files, not one: a count of 1 cannot distinguish "the
    #    ignored file was excluded" from "the walk collapsed to a single file".
    found, walked = measure(
        {"a.md": "ok\n", "docs/b.md": "ok\n", "vendor/bad.md": "```rust\nfn main() {}\n"},
        gitignore="vendor/\n")
    if walked != 2:
        fails.append(f"walk boundaries: walked {walked}, expected 2 "
                     f"(the two untracked .md yes, the gitignored one no)")
    if found:
        fails.append(f"walk boundaries: a GITIGNORED file produced {found} — "
                     f"the sweep would report findings in trees no repo owns")

    # 6. population derivation: BOUNDARY.md + a .git DIRECTORY, both required.
    #    A `git worktree` has a .git FILE and would duplicate its parent.
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        (ws / "real").mkdir()
        (ws / "real" / "BOUNDARY.md").write_text("x")
        (ws / "real" / ".git").mkdir()
        (ws / "no-boundary").mkdir()
        (ws / "no-boundary" / ".git").mkdir()
        (ws / "worktree-shaped").mkdir()
        (ws / "worktree-shaped" / "BOUNDARY.md").write_text("x")
        (ws / "worktree-shaped" / ".git").write_text("gitdir: elsewhere")
        if derive_repos(ws) != ["real"]:
            fails.append(f"population derivation wrong: {derive_repos(ws)}")

        # 7. pin parser: arity ENFORCED, comments stripped.
        pins = ws / "pins.txt"
        pins.write_text("# c\nrepo-a 800 0  # trailing\n\n")
        if parse_pins(pins) != {"repo-a": dict(zip(FIELDS, (800, 0)))}:
            fails.append("pin parse: 3-field row not read correctly")
        pins.write_text("repo-a 1\n")
        try:
            parse_pins(pins)
            fails.append("pin parse: short row accepted")
        except ValueError:
            pass
    return fails


def main() -> int:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (embedded / detached); keep old behavior

    fails = selftest()
    if fails:
        print("✗ markdown fence sweep SELFTEST FAILED — instrument untrustworthy:")
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

    names = derive_repos(WORKSPACE)
    if not names:
        print(f"✗ derived population is EMPTY under {WORKSPACE} — refusing to "
              f"report a green over zero repos")
        return 2

    # Same quantity, two files: this sweep re-states the walk floor that the
    # per-push gate owns for THIS repo. Asserted, not trusted.
    mine = pins.get(REPO_ROOT.name, {}).get("min_md_files")
    if mine != mfg.MIN_FILES:
        print(f"✗ pin drift: {PINS.name} says min_md_files={mine} for "
              f"{REPO_ROOT.name}, markdown_fence_gate.MIN_FILES is "
              f"{mfg.MIN_FILES}. Same quantity, two files — change both.")
        return 1

    bad = False
    tot_files = tot_found = tot_swallowed = 0

    for name in names:
        found, walked = mfg.unterminated(WORKSPACE / name)
        row = pins.get(name)
        swallowed = sum(r[2] for r in found)
        tot_files += walked
        tot_found += len(found)
        tot_swallowed += swallowed

        flags = []
        if row is None:
            flags.append("UNPINNED — add a row (or it can never red)")
        else:
            if walked < row["min_md_files"]:
                flags.append(f"walk FLOOR breached: {walked} .md < "
                             f"{row['min_md_files']} — documents were removed, "
                             f"or `git ls-files` went blind and the 0 below "
                             f"means nothing")
            if len(found) > row["max_unterminated"]:
                flags.append(f"unterminated {len(found)} > pinned "
                             f"{row['max_unterminated']}")

        status = "✗" if flags else ("·" if found else "✓")
        print(f"{status} {name:22s} md={walked:<5d} unterminated={len(found)} "
              f"swallowed={swallowed}")
        for rel, line, swal in sorted(found, key=lambda r: -r[2]):
            print(f"      {rel}:{line}  {swal} line(s) render as code to EOF")
        for f in flags:
            bad = True
            print(f"      ✗ {f}")

    for name in sorted(set(pins) - set(names)):
        bad = True
        print(f"✗ {name}: pinned but ABSENT from the derived walk — it was "
              f"retired (drop the row in that commit) or the walk went blind")

    print(f"\n{len(names)} contract repo(s) · {tot_files} tracked+untracked .md · "
          f"{tot_found} unterminated fence(s) · {tot_swallowed} line(s) rendering "
          f"as code")
    # State the scope where it is READ, not only in the docstring.
    print("  scope: UNTERMINATED only. A fence indented 4+ spaces is a real "
          "fence in a list item here (73 workspace-wide), so the parser cannot "
          "exclude indentation — documents adapt, the scanner does not.")

    if bad:
        print("✗ markdown fence sweep FAILED — see the ✗ rows above")
        print("    The reported line is the DANGLING fence, not necessarily the "
              "defect: read the first non-blank body line — code means a closer "
              "is missing, prose means the fence is an orphan, and a nested "
              "same-length fence needs the OUTER pair widened to ````.")
        return 1
    print("✓ markdown fence sweep PASSED — every repo at or under its pins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
