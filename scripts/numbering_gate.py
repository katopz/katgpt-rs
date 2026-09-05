#!/usr/bin/env python3
"""Issue 724 T4 — the `.plans/`/`.issues/`/`.research/`/`.proposals/` numbering gate.

Every cross-reference in this workspace is by number, so a number allocated
twice means `Plan N` resolves to two documents and a reader following a citation
cannot tell which. AGENTS.md's Numbering Discipline says `.highwater` prevents
this; nothing checked that it was consulted, and it repeatedly was not:

  * `f98f7b51` (2026-07-15) resolved ELEVEN `.plans/` collisions by hand, and a
    new one landed three days later (`449`, both copies still in HEAD until
    Issue 724 T2). A one-time cleanup with no gate behind it buys three days.
  * `.plans/.highwater` read 585 while `586_*` existed, and `.benchmarks/`
    700 while `701_*` existed — i.e. `value + 1` was ALREADY TAKEN in two
    directories at once. The loaded state is the normal state unless something
    checks.

Scope, floors and ceilings are DATA, in `scripts/numbering_floors.txt`, which
also carries the measured reason `.benchmarks/` and `.docs/` are excluded. Read
that file before widening this one.

Report + gate. Exit 0 clean, 1 on drift, **2 if the instrument itself is
untrustworthy** (selftest failure) — an unreliable instrument is not the same
finding as drift, and must not be reported as one.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# ── vocabulary: DATA, not derived from the tree ────────────────────────────
# Deriving both the scope and the population from one walk is what makes a
# gate permanently green (the workspace rule). The scope is pinned; only the
# population is derived.
NUMBERED = re.compile(r"^(\d+)_.+\.md$")
HIGHWATER = ".highwater"


def parse_pins(path: Path) -> dict[str, int]:
    pins: dict[str, int] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, val = (p.strip() for p in line.split("=", 1))
        pins[key] = int(val)
    return pins


def tracked_paths(repo: Path, dirs: list[str]) -> set[str]:
    """The set of git-TRACKED paths under `dirs`, as repo-relative strings.

    One subprocess call. An untracked file is a colleague's in-flight work, not
    a repo defect -- but it becomes one the moment it is committed, so the two
    populations are kept apart rather than pooled.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--", *dirs],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {ln.strip() for ln in out.splitlines() if ln.strip()}


def read_highwater(d: Path) -> tuple[int | None, str | None]:
    """-> (value, malformed_raw). Exactly one of the two is None.

    ABSENT and MALFORMED are deliberately NOT the same verdict, and conflating
    them is what made the above-highwater ceiling blind. Not every numbered
    directory carries an allocator, so a missing file is legal; a file that is
    PRESENT and unparseable disables the `max > highwater` check while looking
    identical to a clean directory in the output — the exact "instrument goes
    blind, ceiling passes" shape numbering_floors.txt's floors exist to catch,
    one field over.

    Measured 2026-09-05 across the 16 contract repos: FIVE such files, all the
    same cause — `echo -n <N> > .highwater` under a shell whose builtin `echo`
    does not implement `-n`, so the flag itself lands in the file (`-n 872`).
    None of them is a bare integer; every one of them silently disarms the
    check for its directory.
    """
    f = d / HIGHWATER
    if not f.is_file():
        return None, None
    raw = f.read_text(errors="replace").strip()
    try:
        return int(raw), None
    except ValueError:
        return None, raw


def scan(repo: Path, dirname: str, tracked: set[str]):
    """-> (numbers -> [(name, is_tracked)], highwater, n_files, malformed_raw)."""
    d = repo / dirname
    by_num: dict[int, list[tuple[str, bool]]] = {}
    if not d.is_dir():
        return by_num, None, 0, None
    n = 0
    for entry in sorted(d.iterdir()):
        m = NUMBERED.match(entry.name)
        if not m:
            continue
        n += 1
        rel = f"{dirname}/{entry.name}"
        # int(), not the literal prefix: `075` and `75` are the same "Plan 75"
        # to every citation in the corpus, so they must collide here too.
        by_num.setdefault(int(m.group(1)), []).append((entry.name, rel in tracked))
    hw, hw_bad = read_highwater(d)
    return by_num, hw, n, hw_bad


def selftest() -> list[str]:
    """Pin the classifier. Every failure mode below is SILENT otherwise."""
    fails = []

    # 1. the regex admits the real shapes and rejects the non-numbered ones
    for name in ("075_foo_bar.md", "0_x.md", "586_pot_scale.md"):
        if not NUMBERED.match(name):
            fails.append(f"regex rejected a real numbered file: {name}")
    for name in ("README.md", ".highwater", "notes.md", "0_.md", "abc_1.md"):
        if NUMBERED.match(name):
            fails.append(f"regex admitted a non-numbered file: {name}")

    # 2. zero-padding must NOT hide a collision -- `075` and `75` are one number
    if int("075") != int("75"):
        fails.append("zero-pad normalization broken")

    # 3. pin parsing: comments stripped, dotted keys kept
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pins.txt"
        p.write_text(
            "# a comment\nmax_duplicate_numbers = 0\n"
            "min_files.plans = 400  # trailing comment\n\nnot a pin line\n"
        )
        pins = parse_pins(p)
        if pins.get("max_duplicate_numbers") != 0:
            fails.append("pin parse: ceiling missing")
        if pins.get("min_files.plans") != 400:
            fails.append("pin parse: trailing comment not stripped")
        if len(pins) != 2:
            fails.append(f"pin parse: expected 2 pins, got {len(pins)}")

    # 4. the tracked/untracked split -- the whole reason this gate can land
    #    green today, so a regression here silently turns a WIP file into a
    #    failure or (worse) a committed collision into a warning.
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / ".plans").mkdir()
        for nm in ("001_a.md", "001_b.md", "002_c.md"):
            (repo / ".plans" / nm).write_text("x")
        (repo / ".plans" / HIGHWATER).write_text("2")
        by_num, hw, n, hw_bad = scan(repo, ".plans", {".plans/001_a.md", ".plans/002_c.md"})
        if hw_bad is not None:
            fails.append(f"clean highwater misread as malformed: {hw_bad!r}")
        if n != 3:
            fails.append(f"scan counted {n}, expected 3")
        if hw != 2:
            fails.append(f"scan read highwater {hw}, expected 2")
        if sorted(by_num) != [1, 2]:
            fails.append(f"scan grouped {sorted(by_num)}, expected [1, 2]")
        flags = dict((nm, tr) for nm, tr in by_num.get(1, []))
        if flags != {"001_a.md": True, "001_b.md": False}:
            fails.append(f"tracked split wrong: {flags}")

        # 5. above-highwater must be DETECTED, not just tolerated
        (repo / ".plans" / "009_over.md").write_text("x")
        by2, hw2, _, _ = scan(repo, ".plans", set())
        if max(by2) <= (hw2 or 0):
            fails.append("above-highwater case did not construct")

    # 6. ABSENT vs MALFORMED must not collapse. Both read as `hw is None`, so
    #    without this pin a corrupted allocator is indistinguishable from a
    #    directory that never had one — and the above-highwater ceiling passes
    #    over both. Canaried with the real observed corruption, not a synthetic
    #    one: `echo -n 872` writing its own flag into the file.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        hw, bad = read_highwater(d)
        if (hw, bad) != (None, None):
            fails.append(f"absent highwater misclassified: {(hw, bad)}")
        (d / HIGHWATER).write_text("-n 872\n")
        hw, bad = read_highwater(d)
        if hw is not None or bad != "-n 872":
            fails.append(f"malformed highwater not detected: {(hw, bad)}")
        (d / HIGHWATER).write_text("  0872  \n")
        hw, bad = read_highwater(d)
        if (hw, bad) != (872, None):
            fails.append(f"padded/zero-padded highwater misread: {(hw, bad)}")

    return fails


def main() -> int:
    fails = selftest()
    if fails:
        print("✗ numbering gate SELFTEST FAILED — instrument untrustworthy:")
        for f in fails:
            print(f"    {f}")
        return 2

    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    pins_path = Path(__file__).resolve().parent / "numbering_floors.txt"
    if not pins_path.is_file():
        print(f"✗ pins file missing: {pins_path}")
        return 2
    pins = parse_pins(pins_path)

    dirs = sorted(k.split(".", 1)[1] for k in pins if k.startswith("min_files."))
    if not dirs:
        print("✗ pins file declares NO directories — an empty scope is refused")
        return 2
    dirs = [f".{d}" for d in dirs]

    tracked = tracked_paths(repo, dirs)
    dup_tracked: list[str] = []
    dup_untracked: list[str] = []
    above: list[str] = []
    malformed: list[str] = []
    below_floor: list[str] = []
    total = 0

    for dirname in dirs:
        by_num, hw, n, hw_bad = scan(repo, dirname, tracked)
        total += n
        if hw_bad is not None:
            malformed.append(
                f"{dirname}/.highwater is not a bare integer: {hw_bad!r} — the "
                f"above-highwater check is DISARMED for this directory"
            )
        floor = pins.get(f"min_files{dirname}", 0)
        if n < floor:
            below_floor.append(f"{dirname}: {n} numbered file(s) < floor {floor}")
        for num, files in sorted(by_num.items()):
            if len(files) < 2:
                continue
            names = " · ".join(nm for nm, _ in files)
            n_tracked = sum(1 for _, tr in files if tr)
            row = f"{dirname}/{num:03d} ×{len(files)}: {names}"
            (dup_tracked if n_tracked >= 2 else dup_untracked).append(
                row + ("" if n_tracked >= 2 else f"  [{n_tracked} tracked]")
            )
        if hw is not None and by_num and max(by_num) > hw:
            above.append(f"{dirname}: max {max(by_num)} > .highwater {hw} — `value + 1` is already taken")

    max_dup = pins.get("max_duplicate_numbers", 0)
    max_above = pins.get("max_above_highwater", 0)
    max_malformed = pins.get("max_malformed_highwater", 0)
    bad = False

    if len(dup_tracked) > max_dup:
        bad = True
        print(f"✗ {len(dup_tracked)} TRACKED duplicate number(s) (pinned ≤ {max_dup}):")
        for r in dup_tracked:
            print(f"    {r}")
    if len(above) > max_above:
        bad = True
        print(f"✗ {len(above)} .highwater below its directory max (pinned ≤ {max_above}):")
        for r in above:
            print(f"    {r}")
    if len(malformed) > max_malformed:
        bad = True
        print(f"✗ {len(malformed)} malformed .highwater file(s) (pinned ≤ {max_malformed}):")
        for r in malformed:
            print(f"    {r}")
    if below_floor:
        bad = True
        print("✗ population FLOOR breached — every other verdict here is a ceiling,")
        print("  so a blind instrument would print a confident green over zero files:")
        for r in below_floor:
            print(f"    {r}")

    if dup_untracked:
        # Deliberately NOT a failure: a colleague's in-flight file is not a
        # repo defect. It becomes one on commit, and this gate then reds.
        print(f"  ⚠ {len(dup_untracked)} duplicate(s) involving UNTRACKED files — not a failure yet:")
        for r in dup_untracked:
            print(f"      {r}")

    if bad:
        return 1
    print(
        f"✓ numbering gate PASSED — {total} numbered file(s) over {len(dirs)} dir(s), "
        f"0 tracked duplicates, 0 stale allocators, 0 malformed allocators"
        + (f", {len(dup_untracked)} untracked warning(s)" if dup_untracked else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
