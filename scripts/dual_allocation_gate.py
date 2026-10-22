#!/usr/bin/env python3
"""Issue 796 — the allocation-time dual-allocation gate, CLASSIFIED.

Issue 791 T3 deferred a gate that reds when this checkout and its upstream
have both allocated document numbers since their merge base, on the unmeasured
fear: "a long-lived branch legitimately allocates ahead of its remote".
Issue 796 measured that fear MOOT (7202 one-sided pairs workspace-wide, all
green BY CONSTRUCTION — the intersection is empty whenever only one side
allocated) and measured the gate's real hazard instead: of 39 distinct
workspace divergence incidents, 31 were TWINs — the same document carried on
two post-rebase lines of history, where a naive hard gate cries wolf and the
remedy is one fetch.

So the verdict is CLASSIFIED, structurally, by filename stem per colliding
number:

    TWIN        same stem added on both sides since the merge base — your own
                rebased/cherry-picked line still sitting in the stale
                remote-tracking ref. Annotated, exit-neutral: `git fetch`
                resolves it; failing the run on it would train people to
                ignore the gate.
    INDEPENDENT different stems, same number — two distinct documents are
                about to own one number. RED, exit 1, both sides' adding
                commits named. This is the 791/780/935 class, caught while
                both sides are still diverging — one `git fetch` + a renumber
                now, instead of a citation-drift archaeology later.

One numbered document per repo view: the gate reads THIS checkout's tips
(HEAD and its upstream), not history — the historical replay is the probe's
job (`dual_allocation_fp_probe.py`). Reach limit, measured in 796: the gate
catches the divergences the running box participates in; a box-vs-box
collision on the wire is invisible here until a fetch brings it in, and the
merge-time wall (numbering_gate + 795) owns what lands despite that.

    scripts/dual_allocation_gate.py [repo]        # verdict (default: this repo)
    scripts/dual_allocation_gate.py --prove-fires # the two-session collision
                                                  # fixture, both verdicts

Exit codes: 0 = clean or TWIN-only; 1 = INDEPENDENT collision; 2 = blind
(no upstream / no reflog / self-test failure) — a gate that cannot see must
refuse, never print a green zero.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The probe owns the shared machinery: dirs, regex, reflog-free git plumbing.
# Imported, not copied — the Issue 755 rule.
from dual_allocation_fp_probe import (  # noqa: E402
    NUMBERED_DIRS, NUM_RE, git, numbers_added, selftest as probe_selftest,
)


def upstream_of_head(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD@{upstream}"],
        capture_output=True, encoding="utf-8", errors="replace")
    return out.stdout.strip() if out.returncode == 0 else ""


def added_stems(repo: Path, rng: str) -> dict[int, set[str]]:
    """Number -> the filename stems added under the numbered dirs within rng."""
    out = git(repo, "log", "-M", "--diff-filter=A", "--name-only",
              "--format=", rng, "--", *NUMBERED_DIRS)
    stems: dict[int, set[str]] = {}
    for line in out.splitlines():
        base = os.path.basename(line.strip())
        m = NUM_RE.match(base)
        if m:
            stems.setdefault(int(m.group(1)), set()).add(base[:-3])
    return stems


def adding_commits(repo: Path, rng: str, want: set[int]) -> list[str]:
    """One line per commit in rng that added a file numbered in `want`."""
    out = git(repo, "log", "-M", "--diff-filter=A", "--name-only",
              "--format=%h %s", rng, "--", *NUMBERED_DIRS)
    rows: list[str] = []
    cur: str | None = None
    for line in out.splitlines():
        if re.match(r"^\w{7,} ", line):
            cur = line
        else:
            base = os.path.basename(line.strip())
            m = NUM_RE.match(base)
            if cur and m and int(m.group(1)) in want and cur not in rows:
                rows.append(cur)
    return rows[:6]


def highwater_at(repo: Path, rev: str, d: str) -> int | None:
    """`<d>/.highwater` as of `rev`, or None when absent/unreadable.

    None is NOT zero: a directory with no counter has allocated nothing this
    predicate can reason about, and reading it as 0 would make every number
    in the other side's range look contested.
    """
    out = git(repo, "show", f"{rev}:{d}/.highwater")
    txt = out.strip()
    return int(txt) if txt.isdigit() else None


def counter_collisions(repo: Path, mb: str, head: str, tip: str,
                       explained: set[int], read=None) -> list[dict]:
    """Numbers BOTH sides spent according to the counters alone.

    ⛔ The document comparison above cannot see a number that was allocated
    and CLOSED in one commit — the file never exists in any tree, so
    `--diff-filter=A` has nothing to report, and the record lives in
    HISTORY.md. Measured: a sibling did that with 849 while this checkout
    carried `.issues/849_*.md`, and this gate printed a confident zero.

    `explained` are the numbers the document pass already reported, so one
    collision is never counted twice.
    """
    # `read` is injected so this function's ARITHMETIC is reachable from a
    # per-push arm. Without it the range logic and the one-sided
    # short-circuit are testable only through a real git fixture, i.e. only
    # under the opt-in `--prove-fires` — the pattern AGENTS.md records for
    # the three weakest modules in the CHECKS population.
    read = read or (lambda rev, d: highwater_at(repo, rev, d))
    rows: list[dict] = []
    for d in NUMBERED_DIRS:
        base = read(mb, d)
        mine = read(head, d)
        theirs = read(tip, d)
        if base is None or mine is None or theirs is None:
            continue
        if mine <= base or theirs <= base:
            continue            # one-sided: green BY CONSTRUCTION
        for n in range(base + 1, min(mine, theirs) + 1):
            if n not in explained:
                rows.append({"number": n, "dir": d, "base": base,
                             "mine": mine, "theirs": theirs})
    return rows

def classify(repo: Path) -> tuple[list[dict], str]:
    """(rows, skip_reason). Each row: one colliding number, classified."""
    up = upstream_of_head(repo)
    if not up:
        return [], "no upstream configured for HEAD"
    head = git(repo, "rev-parse", "HEAD").strip()
    remote_tip = git(repo, "rev-parse", up).strip()
    if not head or not remote_tip:
        return [], "cannot resolve HEAD or the upstream tip"
    mb = git(repo, "merge-base", head, remote_tip).strip()
    if not mb:
        return [], "no merge base (unrelated histories?)"
    if mb in (head, remote_tip):
        return [], ""                       # one-sided or in sync: green
    left = added_stems(repo, f"{mb}..{head}")
    right = added_stems(repo, f"{mb}..{remote_tip}")
    rows: list[dict] = []
    for n in sorted(set(left) & set(right)):
        twin = bool(left[n] & right[n])
        rows.append({
            "number": n, "twin": twin,
            "left_stems": sorted(left[n]), "right_stems": sorted(right[n]),
            "left_commits": [] if twin else
            adding_commits(repo, f"{mb}..{head}", {n}),
            "right_commits": [] if twin else
            adding_commits(repo, f"{mb}..{remote_tip}", {n}),
            "counter": False,
        })
    # The COUNTER axis, for a number nobody left a file for.
    for c in counter_collisions(repo, mb, head, remote_tip,
                                {r["number"] for r in rows}):
        rows.append({"number": c["number"], "twin": False, "counter": True,
                     "dir": c["dir"], "base": c["base"], "mine": c["mine"],
                     "theirs": c["theirs"],
                     "left_stems": [], "right_stems": [],
                     "left_commits": [], "right_commits": []})
    return rows, ""


# ── the fixture: a real two-session collision, built in a temp dir ─────────
def build_fixture(base: Path) -> Path:
    import subprocess as sp

    def run(repo: Path, *args: str) -> None:
        r = sp.run(["git", "-C", str(repo), *args], capture_output=True,
                   encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"git {args}: {r.stderr}")

    root = base / "dual_alloc_fixture"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    shared = root / "shared.git"
    a = root / "a"
    b = root / "b"
    run(root, "init", "-q", "-b", "main", "--bare", str(shared))
    sp.run(["git", "clone", "-q", str(shared), str(a)], check=True,
           capture_output=True)
    # Identity BEFORE the first commit — a fresh clone carries no user.* config,
    # so on a box/CI runner with no global identity the base commit dies with
    # "Author identity unknown" (the 10-04 docs_gate main-run failure class).
    run(a, "config", "user.email", "f@f")
    run(a, "config", "user.name", "fixture")
    (a / ".issues").mkdir()
    (a / ".issues" / "README.md").write_text("x\n", encoding="utf-8")
    run(a, "add", ".issues")
    run(a, "commit", "-qm", "base")
    sp.run(["git", "-C", str(a), "push", "-q", "-u", "origin", "main"],
           check=True, capture_output=True)
    # session B forks BEFORE any allocation lands, allocates 900, pushes
    sp.run(["git", "clone", "-q", str(shared), str(b)], check=True,
           capture_output=True)
    for repo in (a, b):
        run(repo, "config", "user.email", "f@f")
        run(repo, "config", "user.name", "fixture")
    (b / ".issues" / "900_b_side.md").write_text("b\n", encoding="utf-8")
    run(b, "add", ".issues")
    run(b, "commit", "-qm", "B allocates 900")
    sp.run(["git", "-C", str(b), "push", "-q", "origin", "main"], check=True,
           capture_output=True)
    # session A allocates the SAME number locally, then fetches — the gate's
    # invocation moment. HEAD does not move on fetch; the divergence is live.
    (a / ".issues" / "900_a_side.md").write_text("a\n", encoding="utf-8")
    run(a, "add", ".issues")
    run(a, "commit", "-qm", "A allocates 900 independently")
    sp.run(["git", "-C", str(a), "fetch", "-q", "origin"], check=True,
           capture_output=True)
    return root


def counter_fixture(base: Path) -> Path:
    """A collision where ONE side left no document at all.

    B allocates 950 by bumping `.highwater` and writing its record into
    HISTORY.md — the file is never committed, which is the Issue-754 shape and
    what a close-in-the-same-commit looks like from git's side. A then
    allocates 950 locally WITH a file. The document comparison sees one side
    only; the counters see both.
    """
    import subprocess as sp

    def run(repo: Path, *args: str) -> None:
        r = sp.run(["git", "-C", str(repo), *args], capture_output=True,
                   encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"git {args}: {r.stderr}")

    root = base / "counter_fixture"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    shared = root / "shared.git"
    a, b = root / "a", root / "b"
    run(root, "init", "-q", "-b", "main", "--bare", str(shared))
    sp.run(["git", "clone", "-q", str(shared), str(a)], check=True,
           capture_output=True)
    (a / ".issues").mkdir()
    # `write_text(newline=...)` is 3.10+; this gate must run on 3.9 (measured
    # 2026-10-23: the kwarg crashed the whole gate on the seal1 box) — the
    # open()-form is behavior-identical (LF preserved, no translation).
    with open(a / ".issues" / ".highwater", "w", encoding="utf-8", newline="") as f:
        f.write("949\n")
    with open(a / "HISTORY.md", "w", encoding="utf-8", newline="") as f:
        f.write("# history\n")
    run(a, "config", "user.email", "f@f")
    run(a, "config", "user.name", "fixture")
    run(a, "add", "-A")
    run(a, "commit", "-qm", "base at 949")
    sp.run(["git", "-C", str(a), "push", "-q", "-u", "origin", "main"],
           check=True, capture_output=True)

    sp.run(["git", "clone", "-q", str(shared), str(b)], check=True,
           capture_output=True)
    run(b, "config", "user.email", "f@f")
    run(b, "config", "user.name", "fixture")
    # B: allocate 950 and CLOSE it in the same commit — no file, ever.
    with open(b / ".issues" / ".highwater", "w", encoding="utf-8", newline="") as f:
        f.write("950\n")
    with open(b / "HISTORY.md", "w", encoding="utf-8", newline="") as f:
        f.write("# history\n\n## Issue 950 — closed\n")
    run(b, "add", "-A")
    run(b, "commit", "-qm", "B allocates and closes 950")
    sp.run(["git", "-C", str(b), "push", "-q", "origin", "main"], check=True,
           capture_output=True)

    # A: allocate the same number, WITH a document.
    with open(a / ".issues" / "950_a_side.md", "w", encoding="utf-8", newline="") as f:
        f.write("a\n")
    with open(a / ".issues" / ".highwater", "w", encoding="utf-8", newline="") as f:
        f.write("950\n")
    run(a, "add", "-A")
    run(a, "commit", "-qm", "A allocates 950 independently")
    sp.run(["git", "-C", str(a), "fetch", "-q", "origin"], check=True,
           capture_output=True)
    return root


def prove_counter() -> list[str]:
    """The COUNTER axis must RED where the document axis is blind, and must
    NOT fire on a one-sided bump."""
    import tempfile
    import subprocess as sp
    fails: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        root = counter_fixture(Path(td))
        a = root / "a"
        rows, skip = classify(a)
        if skip:
            return [f"counter fixture skipped: {skip}"]
        docs = [r for r in rows if not r.get("counter")]
        ctrs = [r for r in rows if r.get("counter")]
        if docs:
            fails.append(f"the DOCUMENT axis saw a row it cannot see — the "
                         f"fixture is not the intended shape: {docs}")
        if len(ctrs) != 1 or ctrs[0]["number"] != 950:
            fails.append(f"the COUNTER axis did not report exactly 950: "
                         f"{ctrs}")
        elif ctrs[0]["twin"]:
            # A TWIN is "the same document on both lines, a fetch resolves
            # it" and is EXIT-NEUTRAL. A counter-only collision has no
            # document at all, so it can never be one — and labelled TWIN it
            # would stop exiting non-zero while every count above stayed
            # right.
            fails.append("the COUNTER row is labelled TWIN — a twin is "
                         "exit-neutral, so this would green the very "
                         "collision the axis exists to catch")
        # ⚠ The negative side, without which the axis could fire always and
        # still pass: a ONE-SIDED bump is green by construction, and it is the
        # ordinary case on every push.
        b = root / "b"
        sp.run(["git", "-C", str(b), "fetch", "-q", "origin"], check=True,
               capture_output=True)
        rows_b, skip_b = classify(b)
        if not skip_b and [r for r in rows_b if r.get("counter")]:
            fails.append(f"a ONE-SIDED counter bump fired: {rows_b}")
    return fails


def prove_fires() -> list[str]:
    """The gate must RED on the constructed collision and stay exit-neutral
    on the twin shape. Both arms against real git repos, not mocks."""
    import tempfile
    import subprocess as sp
    fails: list[str] = []

    def run(repo: Path, *args: str) -> None:
        r = sp.run(["git", "-C", str(repo), *args], capture_output=True,
                   encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"git {args}: {r.stderr}")

    with tempfile.TemporaryDirectory() as td:
        root = build_fixture(Path(td))
        a, b = root / "a", root / "b"
        rows, skip = classify(a)
        if skip:
            return [f"fixture skipped: {skip}"]
        indep = [r for r in rows if not r["twin"]]
        # ⛔ The BUCKET LABEL, asserted rather than filtered on. Every other
        # check here selects by `twin`/`counter` and so agrees with itself
        # whichever way the literal points; only this notices a DOCUMENT
        # collision wearing the COUNTER label, which would print the wrong
        # remedy ("both lines bumped .highwater") for a row that has two
        # documents naming both adding commits.
        mislabelled = [r["number"] for r in rows if r.get("counter")]
        if mislabelled:
            fails.append(f"fixture: document collisions labelled COUNTER: "
                         f"{mislabelled} — the counter axis is for numbers NO "
                         f"document records")
        if len(indep) != 1 or indep[0]["number"] != 900:
            fails.append(f"fixture: expected exactly RED 900, got {rows}")
        elif not (indep[0]["left_commits"] and indep[0]["right_commits"]):
            fails.append("fixture: RED 900 lacks the adding-commit rows")
        # ── the TWIN arm, same fixture: BOTH sides write the SAME stem. B
        # (which owns the remote line) pushes 901; A writes 901 and stays
        # unpushed. Both sides hold 901 since mb: same stem → TWIN
        # (exit-neutral); 900 stays INDEPENDENT (RED).
        (a / ".issues" / "901_twin_doc.md").write_text("t\n", encoding="utf-8")
        run(a, "add", ".issues")
        run(a, "commit", "-qm", "A writes the twin 901")
        (b / ".issues" / "901_twin_doc.md").write_text("t2\n", encoding="utf-8")
        run(b, "add", ".issues")
        run(b, "commit", "-qm", "B writes the twin 901")
        sp.run(["git", "-C", str(b), "push", "-q", "origin", "main"],
               check=True, capture_output=True)
        sp.run(["git", "-C", str(a), "fetch", "-q", "origin"], check=True,
               capture_output=True)
        rows2, _ = classify(a)
        twins = [r for r in rows2 if r["twin"]]
        indep2 = [r for r in rows2 if not r["twin"]]
        if not any(r["number"] == 901 for r in twins):
            fails.append(f"fixture: twin 901 not classified TWIN: {rows2}")
        if [r["number"] for r in indep2] != [900]:
            fails.append(f"fixture: 900 must stay INDEPENDENT: {rows2}")
        # ── Issue 797 T6: the two decisions the fixture above cannot reach ──
        # `arm_reach_audit` measured this module UNREACHED (0 killed of 28)
        # before its arms were invoked at all; these two are what remained live
        # once they were, and both are real rules rather than fixture plumbing.

        # (1) `adding_commits` filters by `want`. Every number in the fixture is
        # wanted, so the membership test was asserted by nothing — and dropping
        # it names the WRONG commit to a human resolving a collision, which is
        # the one thing this output is for.
        rng = f"{git(a, 'merge-base', 'HEAD', 'origin/main').strip()}..HEAD"
        all_rows = adding_commits(a, rng, {900, 901})
        none_rows = adding_commits(a, rng, {999})
        if not all_rows:
            fails.append("adding_commits found no commit for a wanted number")
        if none_rows:
            fails.append(f"adding_commits ignored `want` \u2014 it named commits for "
                         f"a number nobody asked about: {none_rows}")

        # (2) `classify`'s `not head or not remote_tip` guard is MEASURED
        # UNREACHABLE and therefore pinned rather than armed. To reach it with
        # exactly one side empty the upstream must be configured AND its tip
        # unresolvable; measured on this box, deleting `refs/remotes/origin/X`
        # makes `rev-parse --abbrev-ref HEAD@{upstream}` fail too, so
        # `upstream_of_head` returns "" and the guard one line up fires first.
        # An `or`->`and` mutant there is EQUIVALENT: the verdict is identical
        # and only the skip REASON could differ, in a state git will not
        # produce. Written down because "writing the reason is the
        # adjudication" — the first attempt at an arm here asserted the wrong
        # skip string and proved the branch instead.
    return fails


def selftest() -> list[str]:
    """⛔ This used to delegate ENTIRELY to the probe's arms, and delegation
    cannot reach the consumer (Issue 775's sentence, Issue 789's rule).

    `arm_reach_audit` measured the consequence: **0 killed of 55**, the whole
    module UNREACHED — `classify`, `added_stems`, `adding_commits` and
    `upstream_of_head` were asserted by nothing, in a gate that runs on every
    push. The fixture arms that DO reach them already existed as
    `prove_fires()`, behind `--prove-fires`, and `arm_reach` deliberately never
    invokes that name.

    ⚠ `prove_fires` is excluded from `RUN_ARMS` for a COST reason that does not
    apply here: the exclusion was measured against known-answer arms that
    `git archive` a frozen tree (80.2s vs 4.4s, 436 git invocations). This one
    builds two small temp repos and runs in **1.45s**, against the docs gate's
    ~24s wall — cheaper than the `platform_dead_code_floor_gate` check already
    in the set. Calling it here costs that once per push and buys the module its
    only reach. Its body stays unmutated either way, because the name is still
    in `ARM_NAMES`.
    """
    # ⛔ `prove_counter` is called HERE, not only from `--prove-fires`.
    # Left on the CLI path alone it is an arm `arm_reach` never
    # invokes, so the COUNTER axis's fixture builder and `classify`'s
    # own counter rows were mutated with nothing watching — measured:
    # 19 live survivors across `counter_fixture` / `prove_counter` /
    # `run`. Same argument the docstring above makes for `prove_fires`,
    # and the same price: 1.17s against the docs gate's ~24s wall.
    fails = probe_selftest() + prove_fires() + prove_counter()
    if not callable(numbers_added):
        fails.append("probe import: numbers_added missing")

    # ── the COUNTER axis's ARITHMETIC, with the git reader injected ────────
    # The fixture arm for this lives under `--prove-fires`; this is the half
    # `docs_gate.sh` runs on every push, and it costs nothing.
    def _fake(table):
        return lambda rev, d: table.get((rev, d))

    both = counter_collisions(Path("."), "mb", "head", "tip", set(),
                              read=_fake({("mb", ".issues"): 848,
                                          ("head", ".issues"): 850,
                                          ("tip", ".issues"): 849}))
    nums = sorted(r["number"] for r in both if r["dir"] == ".issues")
    if nums != [849]:
        fails.append(f"the contested range is (base, min(mine, theirs)] — "
                     f"want [849] from 848/850/849, got {nums}")

    one_sided = counter_collisions(Path("."), "mb", "head", "tip", set(),
                                   read=_fake({("mb", ".issues"): 848,
                                               ("head", ".issues"): 848,
                                               ("tip", ".issues"): 860}))
    if [r for r in one_sided if r["dir"] == ".issues"]:
        fails.append("a ONE-SIDED bump fired — that is the ordinary case on "
                     "every push and firing on it is the cries-wolf outcome")

    absent = counter_collisions(Path("."), "mb", "head", "tip", set(),
                                read=_fake({("head", ".issues"): 5,
                                            ("tip", ".issues"): 5}))
    if [r for r in absent if r["dir"] == ".issues"]:
        fails.append("a MISSING base counter was read as 0 — every number in "
                     "the other side's range would look contested")

    explained_out = counter_collisions(Path("."), "mb", "head", "tip", {849},
                                       read=_fake({("mb", ".issues"): 848,
                                                   ("head", ".issues"): 849,
                                                   ("tip", ".issues"): 849}))
    if [r for r in explained_out if r["dir"] == ".issues"]:
        fails.append("a number the DOCUMENT axis already reported was counted "
                     "twice")

    return fails


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--prove-fires" in sys.argv:
        fails = prove_fires() + prove_counter()
        if fails:
            print("✗ prove-fires FAILED — the gate cannot see its own class:")
            for f in fails:
                print(f"    {f}")
            return 2
        print("✓ prove-fires PASS — constructed collision REDs with adding "
              "commits named; twin shape classifies exit-neutral; and the "
              "COUNTER axis REDs on a number NO document records while "
              "staying silent on a one-sided bump")
        return 0
    if "--self-test" in sys.argv:
        fails = selftest()
        if fails:
            print("✗ dual_allocation_gate SELFTEST FAILED:")
            for f in fails:
                print(f"    {f}")
            return 2
        print("✓ selftest PASS")
        return 0
    if len(args) > 1:
        print("usage: dual_allocation_gate.py [repo] | --prove-fires | --self-test")
        return 2
    repo = Path(args[0]).resolve() if args else Path.cwd()

    fails = selftest()
    if fails:
        print("✗ dual_allocation_gate SELFTEST FAILED — gate untrustworthy:")
        for f in fails:
            print(f"    {f}")
        return 2

    rows, skip = classify(repo)
    if skip:
        print(f"— dual_allocation_gate DEFERRED: {skip}. "
              "Not a green zero — the gate could not see.")
        return 2
    twins = [r for r in rows if r["twin"]]
    counters = [r for r in rows if r.get("counter")]
    indep = [r for r in rows if not r["twin"] and not r.get("counter")]
    print(f"dual_allocation_gate {repo.name}: {len(twins)} twin, "
          f"{len(indep)} independent, {len(counters)} counter-only "
          f"colliding number(s)")
    for r in counters:
        print(f"  ⛔ COUNTER {r['number']} — both lines bumped "
              f"{r['dir']}/.highwater past {r['base']} (mine {r['mine']}, "
              f"upstream {r['theirs']}) and NO document explains it on at "
              f"least one side. That is a number allocated and CLOSED in one "
              f"commit — the file never exists in a tree, so the document "
              f"comparison is blind to it and the record is in HISTORY.md")
    for r in twins:
        print(f"  TWIN {r['number']} — same document on both lines "
              f"(rebased own work; a fetch resolves): "
              f"{r['left_stems'][:2]} vs {r['right_stems'][:2]}")
    for r in indep:
        print(f"  ⛔ INDEPENDENT {r['number']} — two documents claim one number")
        for s in r["left_stems"]:
            print(f"      [local ] {s}")
        for s in r["right_stems"]:
            print(f"      [remote] {s}")
        for c in r["left_commits"]:
            print(f"      [local ] {c}")
        for c in r["right_commits"]:
            print(f"      [remote] {c}")
    if indep or counters:
        print("Renumber ONE side before pushing — Issue 791 T2's protocol, "
              "caught at allocation time instead of merge time.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
