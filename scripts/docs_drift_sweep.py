#!/usr/bin/env python3
"""Run both doc-drift auditors over EVERY contract repo, not just this one.

Issue 702's finding, in one sentence: the auditors accept a repo path and audit
any repo, and for months nothing pointed them at anything but katgpt-rs. A
sibling with stale labels is indistinguishable from a sibling nobody looked at.
This script is the workstation instrument that looks at all of them at once —
the answer had been recomputed by hand three times before it was written down.

Why this is NOT in scripts/docs_gate.sh's CHECKS array
------------------------------------------------------
docs_gate.yml runs on ubuntu-latest with a single checkout. The siblings are
private and simply are not there, so this check would either fail on every CI
run or — far worse — derive an empty population and print a confident green
over zero repos. Cadence is deliberately split:

  this script            workstation, on demand, every contract repo (derived)
  docs_gate.yml          CI, per-push, katgpt-rs only
  sibling_docs_drift.yml CI, reusable, one sibling per caller

Vocabulary vs population (the trap this script is built around)
---------------------------------------------------------------
The population — which repos exist — is DERIVED (BOUNDARY.md + a .git dir),
never typed; a hand-typed repo set is Issue 703. But deriving the *expectations*
from the same walk would make the gate permanently green: a repo that vanished,
or an auditor that went blind to a whole dialect, both just shrink the walk and
still report "0 mismatches". So the expectations are COMMITTED, in
scripts/docs_drift_floors.txt, and this script fails on a repo that is in the
floor file and absent or under-count in the walk.

The floors are PRESENCE, not the observed count, and the reasoning for that
lives in docs_drift_floors.txt's header — in short: exact counts would duplicate
bench_doc_audit.py's selftest() (a stronger, already-canaried blindness
detector) while redding on every legitimate doc removal. The assertion this
sweep is uniquely able to make is "a repo known to bear labels was actually
SEEN", which selftest() cannot make because it never leaves this repo.

Floors are a MINIMUM, so adding docs never reds this gate. If a repo
legitimately stops carrying labels, drop its row in the same commit — the
failure message says so, because a gate whose red is ambiguous gets ignored.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_population import open_repo, population_verdict, pin_row_exempt  # noqa: E402
from worktree_state import (  # noqa: E402
    dirty_in_population, head_tree, sweep_advisory)
import repo_alias  # noqa: E402 — the machine-local name codec (see its docstring)

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT.parent
FLOORS_FILE = Path(__file__).resolve().parent / "docs_drift_floors.txt"

# (script, regex capturing the per-repo count, human name of the unit)
AUDITORS = [
    # ⚠ `.*?` between the two numbers, not a comma. This regex PARSES another
    # instrument's prose, and on 2026-09-15 that instrument grew a population
    # clause — `checked 97 labels over 439 doc(s), 0 mismatches` — which took
    # this sweep to 0 labels in every repo and produced NINE findings, all
    # false. The lazy gap survives a word being inserted; a comma did not.
    # Caught by this sweep's own "the auditor is dead, not the docs clean"
    # floor, which is the floor working exactly as designed.
    ("bench_doc_audit.py", re.compile(r"checked (\d+) labels?.*?(\d+) mismatch"), "labels"),
    ("cargo_comment_audit.py", re.compile(r"checked (\d+) inline comments?, (\d+) mismatch"), "comments"),
]
AUDITING_RE = re.compile(r"^=== Auditing (.+?) ===")


def derive_population(root: Path | None = None) -> list[Path]:
    """Every sibling carrying a BOUNDARY.md contract. Derived, never typed.

    `root` is optional and defaults to WORKSPACE (Issue 788): a predicate that
    hard-codes its root cannot be run against the synthetic workspace
    `population_sync_gate.py` uses, which is the half of that gate that works
    in CI. Two of the ten were unparameterised and therefore untestable there.
    """
    ws = WORKSPACE if root is None else Path(root)
    # Names pass through the machine-local alias codec (`repo_alias.py`) so
    # the returned paths carry the CONTRACT spelling every tracked pin is
    # keyed on.
    return [ws / n for n in repo_alias.apply(
        d.name for d in ws.iterdir()
        if d.is_dir() and (d / "BOUNDARY.md").is_file() and (d / ".git").is_dir()
    )]


def read_floors() -> dict[str, tuple[int, int]]:
    """Committed expectations: repo -> (min bench labels, min cargo comments)."""
    floors: dict[str, tuple[int, int]] = {}
    if not FLOORS_FILE.is_file():
        return floors
    for raw in FLOORS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, bench, cargo = line.split()
        floors[name] = (int(bench), int(cargo))
    return floors


def attribute_counts(stdout: str, pattern: re.Pattern[str],
                     owner: dict[str, str], wanted: list[str]):
    """-> (counts keyed on the REQUESTED repo, the requested repos with no
    answer). Pure — the arms in `attribution_arms` drive it directly."""
    counts: dict[str, tuple[int, int]] = {}
    current: str | None = None
    for line in stdout.splitlines():
        header = AUDITING_RE.match(line)
        if header:
            current = header.group(1)
            continue
        hit = pattern.search(line)
        if hit and current:
            # Issue 842: the auditors are handed the on-disk directories, so
            # the header carries the ON-DISK name; the pins are keyed on the
            # CONTRACT spelling. display() is the identity on unaliased boxes
            # and normalizes the alias box — the alias content itself must
            # never leak further than this one mapping.
            key = owner.get(current, repo_alias.display(current))
            lab, mis = counts.get(key, (0, 0))
            counts[key] = (lab + int(hit.group(1)), mis + int(hit.group(2)))
    return counts, sorted(set(wanted) - counts.keys())


def attribution_arms() -> list[str]:
    """The two 2026-10-27 defects, replayed on literal auditor output."""
    fails: list[str] = []
    pat = AUDITORS[1][1]
    split = ("\n=== Auditing a ===\n  -> checked 2 inline comments, 1 mismatch\n"
             "\n=== Auditing b ===\n  -> checked 3 inline comments, 0 mismatch\n")
    got = attribute_counts(split, pat, {"a": "site", "b": "site"}, ["site"])
    if got != ({"site": (5, 1)}, []):
        fails.append(f"attribution: a manifest-less repo's child blocks must "
                     f"SUM into the repo they were requested for, got {got}")
    crashed = "Traceback (most recent call last):\nModuleNotFoundError\n"
    got = attribute_counts(crashed, pat, {"r": "r"}, ["r"])
    if got[1] != ["r"]:
        fails.append(f"attribution: a child that printed NO block is a "
                     f"missing verdict, never (0, 0) — got {got}")
    return fails


def run_auditor(script: str, pattern: re.Pattern[str], repos: list[Path]) -> dict[str, tuple[int, int]]:
    """One invocation over every repo — both auditors accept N paths.

    Counts are keyed on the REQUESTED repo, not on the auditor's block header:
    a repo with no root Cargo.toml is audited per child crate (`audit_units`),
    so its blocks carry the children's names and are summed back here."""
    # Lazy: bench_doc_audit imports tomllib (3.11+), and the no-verdict
    # message below is the useful failure on an older interpreter.
    try:
        from bench_doc_audit import audit_units
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"✗ docs-drift sweep NO VERDICT — the auditors cannot import on "
            f"python {sys.version.split()[0]} ({e}); they need >= 3.11 for "
            f"tomllib. Run with a newer interpreter (uv python find '>=3.11').")

    owner: dict[str, str] = {}
    for p in repos:
        key = repo_alias.display(p.name)
        for unit in audit_units(p.expanduser().resolve()):
            prior = owner.setdefault(unit.name, key)
            if prior != key:
                raise SystemExit(
                    f"✗ {script}: audit unit '{unit.name}' is claimed by both "
                    f"{prior} and {key} — the block header cannot attribute it")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *(str(p) for p in repos)],
        capture_output=True, encoding="utf-8", errors="replace",
        # Pin the CHILD's encoder as well as our decoder — Issue 778.
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if proc.returncode not in (0, 1):
        # A crash is not a clean sweep. Surface it rather than parsing partial
        # output, which would silently under-report (Issue 702's whole theme).
        raise SystemExit(f"✗ {script} crashed (exit {proc.returncode})\n{proc.stderr}")
    wanted = [repo_alias.display(p.name) for p in repos]
    counts, missing = attribute_counts(proc.stdout, pattern, owner, wanted)
    if missing:
        # ⛔ Exit 1 is ALSO what an uncaught Python exception exits with, so
        # the returncode test above cannot tell "findings" from "crashed at
        # import". Measured 2026-10-27: on the box's python3 3.9.6 both
        # auditors died on `import tomllib` (3.11+), exited 1, and were parsed
        # as (0, 0) in every repo — the adjudication arms then reported five
        # INERT-fixture failures, a misdiagnosis of a dead child. The auditor
        # was asked about N repos; an answer missing for any is no verdict.
        raise SystemExit(
            f"✗ {script} gave NO VERDICT for {len(missing)} of {len(repos)} "
            f"repo(s) (exit {proc.returncode}): {', '.join(missing[:5])}"
            f"{' …' if len(missing) > 5 else ''} — a crashed child is not a "
            f"clean sweep\n{proc.stderr[-2000:]}")
    return counts


# ONE list, read by the worktree advisory, by the HEAD trigger and by the
# archive pathspec. Both auditors read the LABELLED DOCS and the MANIFESTS they
# are checked against, and nothing else — an enumerable input set, which is
# what makes the narrowing sound.
#
# ⚠ `*.toml` rather than `Cargo.toml` for the ARCHIVE: a git pathspec `Cargo.toml`
# matches only the root, and a HEAD tree missing `crates/*/Cargo.toml` would
# report zero labels everywhere and breach every floor — a false RED that looks
# exactly like an auditor going blind. The advisory keeps the narrower basename
# form because `_matches` there tests the basename too.
SCOPE = ("*.md", "Cargo.toml")
ARCHIVE_PATHS = ("*.md", "*.toml")


def adjudicate(repos: list[Path], results: dict):
    """-> (the results dict the PINS read, the HEAD results or None).

    Issue 822 T5j. ⛔ This sweep was NOT in the Issue-822 census and the reason
    is worth keeping: T5j re-derived the exposed set by grepping every
    `> row["max_*"]` comparison in the family, and this sweep's ceiling is not
    written that way — it is `if b_mis: failures.append(...)`, a WALL at 0
    spelled as a truth test. A census over ONE REPRESENTATION is blind to
    whatever that representation omits, which is Issue 787's lesson and Issue
    789's, reproduced by the very census that was looking for instances of it.
    The floors (`b_lab < fb`) are exposed the same way.

    `head_tree`, because the classifier is two SUBPROCESS auditors that walk
    the filesystem — there is no reader to inject and no single seam to
    overlay. Every repo goes in one invocation, dirty ones swapped for
    materialised HEAD checkouts, clean ones passed through: the auditors accept
    N paths, so the HEAD pass costs one extra invocation rather than one per
    repo.

    ⚠ The results are keyed by the name the auditor PRINTS (`=== Auditing
    <name> ===`, which is `repo.name`), so this depends on `head_tree` naming
    its checkout after the source repo — the same dependency `len_derived` has,
    armed in `worktree_state`.

    `None` means nothing is dirty in SCOPE, and the caller reads the worktree's
    own numbers.
    """
    dirty = [p for p in repos if dirty_in_population(p, SCOPE)]
    if not dirty:
        return results, None
    with contextlib.ExitStack() as stack:
        swapped = []
        for p in repos:
            tree = stack.enter_context(
                head_tree(p, SCOPE, paths=ARCHIVE_PATHS))
            swapped.append(tree if tree is not None else p)
        head = {script: run_auditor(script, pat, swapped)
                for script, pat, _ in AUDITORS}
    return head, head


_CLEAN_MANIFEST = """[package]
name = "canary-a"
version = "0.0.0"

[features]
default = ["alpha"]
alpha = []   # default-on
beta = []    # opt-in
"""
_BROKEN_MANIFEST = _CLEAN_MANIFEST.replace("alpha = []   # default-on",
                                           "alpha = []   # opt-in")


def adjudicate_cases() -> list[str]:
    global AUDITORS
    _all = AUDITORS
    AUDITORS = [a for a in _all if a[0] == "cargo_comment_audit.py"]
    try:
        return _adjudicate_cases()
    finally:
        AUDITORS = _all


def _adjudicate_cases() -> list[str]:
    """`adjudicate` end to end against REAL git — Issue 822 T5j.

    The fixture is `cargo_comment_audit`'s own subject reduced to one crate: a
    feature in `default` whose inline comment claims `opt-in`. A real
    subprocess auditor over a real repository, because that is what this sweep
    runs and an injected runner would assert an abstraction.
    """
    import tempfile

    fails: list[str] = []

    def git(root, *args):
        subprocess.run(("git", "-C", str(root)) + args,
                       capture_output=True, check=True)

    def fixture(td, committed, worktree=None, nested=False):
        ws = Path(td)
        repo = ws / "r"
        repo.mkdir()
        (repo / "BOUNDARY.md").write_text("x", encoding="utf-8")
        where = repo / "crates" / "canary-a" if nested else repo
        where.mkdir(parents=True, exist_ok=True)
        (where / "Cargo.toml").write_text(committed, encoding="utf-8")
        git(ws, "init", "-q", "r")
        git(repo, "config", "user.email", "arm@example.invalid")
        git(repo, "config", "user.name", "arm")
        git(repo, "add", "-A")
        git(repo, "-c", "commit.gpgsign=false", "commit", "-qm", "base")
        if worktree is not None:
            (where / "Cargo.toml").write_text(worktree, encoding="utf-8")
        return ws, repo

    def run(repo):
        res = {script: run_auditor(script, pat, [repo])
               for script, pat, _ in AUDITORS}
        judged, head = adjudicate([repo], res)
        return res, judged, head

    # ⚠ Only the CARGO auditor runs in these arms, and the narrowing is a
    # COST decision stated rather than hidden: the seam under test is the
    # adjudication, which is identical for both, while `bench_doc_audit` needs
    # a far heavier fixture (a reachability graph across manifests) and is the
    # slowest instrument in this workspace. With both, these arms cost 38.5s
    # against a 22s sweep; with one, 17.8s. `adjudicate` reads module-level
    # AUDITORS, so the narrowing has to happen there or the HEAD pass runs
    # both anyway.

    def cc(d):
        return d["cargo_comment_audit.py"].get("r", (0, 0))

    # a. ⛔ Issue 798's direction: a fix that is not COMMITTED is not landed.
    #    The mismatch wall is 0, so an uncommitted repair hands every other
    #    checkout a green over a claim that is still wrong in HEAD.
    with tempfile.TemporaryDirectory() as td:
        _ws, repo = fixture(td, _BROKEN_MANIFEST, _CLEAN_MANIFEST)
        res, judged, head = run(repo)
        if cc(res)[1] != 0:
            fails.append(f"arm a: the fixture is INERT — the worktree must "
                         f"read 0 mismatches ({cc(res)})")
        if cc(judged)[1] != 1:
            fails.append(f"adjudicate: an UNCOMMITTED fix cleared the mismatch "
                         f"wall ({cc(judged)}) — the committed comment still "
                         f"claims the wrong thing for everyone else")

    # b. And its mirror: a mismatch introduced in the worktree must not red a
    #    wall at 0 over a line no commit contains.
    with tempfile.TemporaryDirectory() as td:
        _ws, repo = fixture(td, _CLEAN_MANIFEST, _BROKEN_MANIFEST)
        res, judged, head = run(repo)
        if cc(res)[1] != 1:
            fails.append(f"arm b: the fixture is INERT — the worktree must "
                         f"read 1 mismatch ({cc(res)})")
        if cc(judged)[1] != 0:
            fails.append(f"adjudicate: an uncommitted mismatch reached the "
                         f"wall ({cc(judged)})")

    # c. The FLOORS read HEAD too — Issue 797 measured this class on a
    #    POPULATION rather than on a finding.
    with tempfile.TemporaryDirectory() as td:
        _ws, repo = fixture(td, _CLEAN_MANIFEST,
                            _CLEAN_MANIFEST + "gamma = []   # opt-in\n")
        res, judged, head = run(repo)
        if cc(res)[0] != 3 or cc(judged)[0] != 2:
            fails.append(f"adjudicate: the label floor read the worktree "
                         f"({cc(res)[0]}) instead of HEAD ({cc(judged)[0]})")

    # d. ⛔ The ARCHIVE PATHSPEC, and this is the arm that matters most here:
    #    a git pathspec `Cargo.toml` matches only the ROOT, so a nested
    #    manifest would be missing from the HEAD tree, every count would read
    #    0, and every floor would breach — a false RED indistinguishable from
    #    an auditor going blind, which is the exact failure this sweep's own
    #    liveness sentinel exists to name.
    with tempfile.TemporaryDirectory() as td:
        _ws, repo = fixture(td, _BROKEN_MANIFEST, _CLEAN_MANIFEST, nested=True)
        res, judged, head = run(repo)
        if cc(judged)[0] != 2:
            fails.append(f"ARCHIVE_PATHS: a NESTED crates/*/Cargo.toml did not "
                         f"survive into the HEAD tree ({cc(judged)}) — every "
                         f"floor then breaches and it reads like blindness")
        if cc(judged)[1] != 1:
            fails.append(f"adjudicate: the nested repo's committed mismatch "
                         f"was lost ({cc(judged)})")

    # e. A CLEAN tree must cost NOTHING: the HEAD pass re-runs both auditors
    #    over every repo, and `bench_doc_audit` alone is the slowest check in
    #    this workspace.
    with tempfile.TemporaryDirectory() as td:
        _ws, repo = fixture(td, _CLEAN_MANIFEST)
        calls = []
        real = globals()["run_auditor"]
        globals()["run_auditor"] = (
            lambda s, p, r: calls.append(s) or real(s, p, r))
        try:
            res, judged, head = run(repo)
        finally:
            globals()["run_auditor"] = real
        if head is not None or len(calls) != len(AUDITORS):
            fails.append(f"adjudicate: re-ran the auditors on a CLEAN repo "
                         f"({len(calls)} calls, head={head is not None}) — "
                         f"nothing is dirty in SCOPE and the caller must SKIP")
    return fails


def adjudicate_arms() -> list[str]:
    """The cases above, plus the STUB PROBE proving they sit on the seam.

    ⛔ Aimed at `adjudicate` returning the WORKTREE results — the regression
    this wiring exists to prevent — rather than at a helper it does not call.
    Unlike its siblings this sweep has no row-level delta to stub: its
    adjudication IS the choice of which results dict the pins read.
    """
    fails = attribution_arms() + adjudicate_cases()
    real = globals()["adjudicate"]
    globals()["adjudicate"] = lambda repos, results: (results, None)
    try:
        probed = adjudicate_cases()
    finally:
        globals()["adjudicate"] = real
    if len(probed) < 3:
        fails.append(f"STUB PROBE: an `adjudicate` that hands back the "
                     f"WORKTREE results red only {len(probed)} of the "
                     f"provenance arms — they are not sitting under the seam")
    return fails


def main() -> int:
    # Prints carry glyphs the Windows locale codecs cannot encode (checked
    # 2026-09-06 on cp874: check/cross/middot/arrow FAIL, em-dash OK); keep the
    # locale encoding and degrade only the fatal chars to escapes -- the
    # staged_set_audit house pattern (utf-8 pinning would mojibake legacy consoles).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (embedded / detached); keep old behavior
    # The seam this sweep LANDED (Issue 822 T5j). It has no `selftest` of its
    # own — its blindness detectors are the floors and the liveness sentinel —
    # so the arms run here, unconditionally. ⚠ MEASURED, and cite the whole-run
    # figure rather than the isolated one: the arms alone are 11.1s, and the
    # sweep went 21.8s -> 47.2s end to end on the same box, because the
    # fixtures' subprocess churn does not compose additively with the auditors
    # already running. The largest relative cost in the family, and not behind
    # a flag because an arm behind a flag runs on no invocation anybody makes
    # (Issue 789) — this sweep's invocation is a human typing it.
    arm_fails = adjudicate_arms()
    if arm_fails:
        print("✗ docs-drift sweep ADJUDICATION ARMS FAILED — "
              "instrument untrustworthy:")
        for a in arm_fails:
            print(f"    {a}")
        return 2

    repos = derive_population()
    floors = read_floors()
    present = {p.name for p in repos}
    # Issue 842: the derived handles are CONTRACT-named; the auditors walk
    # real directories. `repos` stays the contract handle list (the per-repo
    # loop labels with it); the auditors get the on-disk resolutions.
    repos_real = [open_repo(p.name, WORKSPACE) for p in repos]

    print(f"▸ derived population: {len(repos)} contract repos under {WORKSPACE}")
    print(f"▸ committed floors:   {len(floors)} label-bearing repos "
          f"({FLOORS_FILE.name})\n")

    results = {script: run_auditor(script, pat, repos_real) for script, pat, _ in AUDITORS}
    # Issue 822 T5j — the DISPLAY reads the worktree (it is what the files say
    # today); every WALL and every FLOOR reads what a commit of these checkouts
    # would produce.
    judged, head = adjudicate(repos_real, results)

    failures: list[str] = []
    notes: list[str] = []

    # The population axis, shared (Issue 782 T3). This sweep still carried the
    # copy-pasted "pinned but absent" loop Issue 793 replaced in eight others;
    # it only LOOKED exempt because all eight label-bearing repos happen to be
    # checked out here. Three verdicts, never interchangeable: UNREGISTERED
    # reds in every posture, UNSEEN reds without the marker, the same set
    # DEFERS loudly with it.
    pop_lines, deferred, pop_fail = population_verdict(floors, present)

    # Issue 797 — the worktree is not the repo. This run reads files that
    # concurrent sessions are editing, so a finding may sit on a line no
    # commit contains. ADVISORY, never a failure: a sweep that hard-reds on
    # an ordinary dirty worktree is a sweep nobody runs. It rides the FINAL
    # line in BOTH directions (the `deferred` precedent) and is SILENT
    # unless the dirty set meets this sweep's own population — the labelled docs + the manifest defaults they are checked against.
    deferred.extend(sweep_advisory(present, SCOPE, root=WORKSPACE))
    for _line in pop_lines:
        print(_line)
    if pop_fail:
        failures.append(
            f"{pop_fail} contract repo(s) in {FLOORS_FILE.name} or "
            f"repo_set.txt could not be measured — see the population rows "
            f"above; the sweep did NOT cover them")

    header = f"{'repo':<24}{'labels':>8}{'mism':>6}{'comments':>10}{'mism':>6}   floor"
    print(header)
    print("-" * len(header))
    for repo in repos:
        n = repo.name
        b_lab, b_mis = results["bench_doc_audit.py"].get(n, (0, 0))
        c_lab, c_mis = results["cargo_comment_audit.py"].get(n, (0, 0))
        jb_lab, jb_mis = judged["bench_doc_audit.py"].get(n, (0, 0))
        jc_lab, jc_mis = judged["cargo_comment_audit.py"].get(n, (0, 0))
        floor = floors.get(n)
        if floor is None:
            mark = "" if (b_lab or c_lab) else "-"
            # Issue 824. The lowest-severity member of the three and the one
            # most likely to be left: the note is ADVISORY, so the cost is a
            # suggestion to pin a repo the contract does not claim — an action
            # the reader cannot correctly take. Exempt rows still PRINT (the
            # display reads the box, pins read the contract — Issue 797).
            if (b_lab or c_lab) and not pin_row_exempt(n):
                # Newly label-bearing: a floor should be recorded so a future
                # regression back to zero is catchable. Advisory, not fatal.
                mark = "NEW"
                notes.append(
                    f"{n}: now carries {b_lab} labels / {c_lab} comments but has no "
                    f"floor — add `{n}\t{b_lab}\t{c_lab}` to {FLOORS_FILE.name}")
        else:
            fb, fc = floor
            mark = f"{fb}/{fc}"
            if jb_lab < fb:
                failures.append(
                    f"{n}: {jb_lab} committed bench labels < floor {fb} — either the auditor "
                    f"went blind to a dialect, or docs were removed (then LOWER the floor)")
            if jc_lab < fc:
                failures.append(
                    f"{n}: {jc_lab} committed Cargo comments < floor {fc} — either the auditor "
                    f"went blind, or comments were removed (then LOWER the floor)")
        if jb_mis:
            failures.append(f"{n}: {jb_mis} COMMITTED bench-doc label mismatch(es) "
                            f"vs the manifests")
        if jc_mis:
            failures.append(f"{n}: {jc_mis} COMMITTED Cargo-comment mismatch(es) "
                            f"vs the manifests")
        # The worktree's own numbers are DISPLAYED; HEAD's ride beside them
        # whenever they differ, so a row nobody can commit is never silently
        # counted and never silently hidden.
        split = ""
        if head is not None and (jb_lab, jb_mis, jc_lab, jc_mis) != (
                b_lab, b_mis, c_lab, c_mis):
            split = (f"   [HEAD {jb_lab}/{jb_mis} {jc_lab}/{jc_mis} — the "
                     f"worktree figures beside it are UNCOMMITTED]")
        print(f"{n:<24}{b_lab:>8}{b_mis:>6}{c_lab:>10}{c_mis:>6}   {mark}{split}")

    # Liveness sentinel. katgpt-rs is the one repo guaranteed present (it holds
    # this script), so a zero here means the auditor is broken or the walk is
    # pointed at nothing — never that the corpus is clean.
    if not judged["bench_doc_audit.py"].get(REPO_ROOT.name, (0, 0))[0]:
        failures.append(
            f"{REPO_ROOT.name}: 0 labels from its OWN corpus — the auditor is dead, "
            "not the docs clean")

    print()
    for note in notes:
        print(f"  ! {note}")
    if failures:
        print(f"\n✗ docs-drift sweep FAILED — {len(failures)} finding(s):")
        for f in failures:
            print(f"    - {f}")
        return 1
    covered = sum(1 for r in repos if r.name in floors)
    # A deferral rides the FINAL line in both directions.
    scope = "" if not deferred else "; " + "; ".join(deferred)
    print(f"\n✓ docs-drift sweep PASSED — {len(repos)} repos, 0 mismatches, "
          f"{covered}/{len(floors)} floor-bearing repos covered{scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
