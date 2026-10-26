#!/usr/bin/env python3
"""Run the Python 3.9 syntax-floor verdict over EVERY contract repo.

`scripts/python_syntax_floor_gate.py` (Issue 965) is katgpt-rs-scoped by
construction — it walks `REPO_ROOT` and `docs_gate.yml` has a single checkout,
so it could never see a sibling. That is the right shape for a per-push CI
gate and the wrong shape for "is anybody ELSE shipping an instrument that dies
at import on the boxes' python3 3.9.6?"

The landing answered that with a live YES: the workspace-wide grammar parse
found `reflex-site/scripts/render_flows.py` carrying a PEP 701 f-string
backslash (3.12-only) — a site the closing feature-grep census had missed,
dead on this box at the documented invocation. Fixed `a24f4dd` the same day.
This sweep is the instrument that makes that class a red instead of a
discovery.

Bucket plumbing, and why UNPARSED never reaches the pins
--------------------------------------------------------
SYNTAX rows flow through the Issue-822 three-way split (`head_delta`), and the
`max_syntax` ceiling adjudicates HEAD truth — `delta.head` (committed +
MASKED), never the worktree's say-so. UNPARSED rows do NOT: the HEAD rescan
cannot classify a blob that parses at NO grammar (the subprocess sweep's own
precedent — counting it as an offender invents a MASKED row nobody can
repair), so an UNPARSED ceiling fed from `delta.head` would be dead plumbing
that only ever looks like it works. UNPARSED is a direct FLAG on the worktree
scan instead — same bucket, same red, one honest mechanism.

Both floors, and what each one moves on
---------------------------------------
`max_syntax = 0` is green over whatever the walk can SEE, and the walk shells
out to `git ls-files` — a regression takes the population to 0 and the ceiling
passes, indistinguishable from a clean repo. So the population is pinned
underneath it in two independent quantities: `min_py_files` moves when the
WALK goes blind, `min_parsed` when the SCAN LOOP stops iterating early on an
unchanged tree (parsed == files in every passing state, because any file
failing the floor parse is a finding above). Both are pinned and neither is
derived from the other.

katgpt-rs's two floors are NOT free: they must equal
`python_syntax_floor_gate.FLOOR_PY_FILES` / `FLOOR_PARSED`, and this sweep
ASSERTS that rather than trusting it — same quantity, two files.

Host posture, stated where it is read
-------------------------------------
On a 3.9 host the floor IS the native grammar (full coverage; the SYNTAX
bucket is unreachable by construction — every failure is UNPARSED — so the
split-tree arms below need a ≥3.10 host and say so). On ≥3.12 the
segment-based PEP 701 supplements are sound; on 3.10/3.11 those constructs
fail the host's own parse instead. No posture is blind; the gate's `posture()`
line names the mechanism on every run.

No membership pin, deliberately
-------------------------------
The verdict is DERIVED from each file's own grammar and there is no
repaired-file list to lose: every way to reintroduce the defect lands on a
ceiling or a flag (both walls at 0 — a file that cannot parse cannot run), and
every way to go blind to it lands on a floor.

Why this is NOT in scripts/docs_gate.sh's CHECKS
------------------------------------------------
Identical to the rest of the family: CI has one checkout, the siblings are
private and simply absent, so this would either red on every run or derive an
EMPTY population and print a confident green over zero repos.

    this script                     workstation, on demand, every contract repo
    python_syntax_floor_gate.py     CI, per-push (docs_gate.sh), katgpt-rs only

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
# DRY: the classifier, the walk and the floors are the per-push gate's, so the
# sweep and the gate can never disagree about what "fails the 3.9 grammar"
# means (Issue 755: a second copy of a rule this subtle is a second thing to
# get wrong).
import python_syntax_floor_gate as psf  # noqa: E402
from skill_repo_set_gate import derive_repos  # noqa: E402
from sweep_population import open_repo, pin_row_exempt, population_verdict  # noqa: E402
from tracked_walk import tracked_files  # noqa: E402
from worktree_state import (  # noqa: E402
    deferral_line, head_delta, line_free, ordinal_keys, sweep_advisory)

REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
PINS = HERE / "python_syntax_floor_drift_floors.txt"

FIELDS = ("min_py_files", "min_parsed", "max_syntax")


def keyed(rows) -> list:
    """`[(key, rel, kind, row)]` for a row set, ordinals assigned in order.

    A LINE-FREE key (Issue 822, `worktree_state`'s now): the rows this sweep
    prints are `"{lineno}: {msg}"`, and a line-bearing key reports EVERY row
    in an edited file as UNCOMMITTED *and* MASKED at once — any insertion
    above the construct shifts it. `line_free` strips the bare-integer prefix.
    """
    return [(k, rel, kind, row)
            for k, (rel, kind, row) in
            ordinal_keys(rows, lambda t: (t[0], t[1], line_free(t[2])))]


def flatten(syntax: dict) -> list:
    """The SYNTAX bucket as one addressed row list.

    Deliberately NOT the UNPARSED bucket: the HEAD rescan cannot classify a
    blob that parses at no grammar, so those rows would flow through the split
    without ever being adjudicable — dead plumbing. They are a direct flag in
    `main()` (the subprocess sweep's identical shape).
    """
    return [(rel, "SYNTAX", r) for rel in sorted(syntax) for r in syntax[rel]]


def head_offenders(walk: set):
    """`head_delta`'s per-file reclassifier for the SYNTAX ceiling.

    Per-file row independence holds by construction: `floor_findings` is a
    parse over ONE module's source. `walk` is the sweep's own tracked
    population, passed in rather than re-derived — `fnmatch`'s `*` crosses
    `/`, so the scope glob admits paths the walk excludes, and a row invented
    there reads as MASKED, a hard red nobody can repair.
    """

    def rescan(rel: str, src) -> list:
        # None = absent from HEAD (staged but never committed): nothing
        # committed to classify, so the worktree's row is UNCOMMITTED.
        if src is None or rel not in walk:
            return []
        rows = psf.floor_findings(src)
        if any(kind == "UNPARSED" for kind, _ in rows):
            # UNPARSED at HEAD is the instrument admitting it cannot read, and
            # it has its own bucket one level up. Never folded into the
            # ceiling — counting it as clean hides exposure, counting it as an
            # offender invents a MASKED row nobody can repair.
            return []
        return [(rel, "SYNTAX", row) for _, row in rows]

    return rescan


def adjudicate(repo: Path, syntax: dict, walk: set):
    """The offender rows, split COMMITTED / UNCOMMITTED / MASKED (Issue 822).

    A named seam, not four lines inline: this is the verdict arithmetic, and
    `arm_reach_audit`'s standing finding in this repo is that the classifier is
    well armed and the verdict is not.
    """
    rows = keyed(flatten(syntax))
    rescan = head_offenders(walk)
    return head_delta(
        repo, ("*.py",), rows,
        lambda r: r[1], lambda r: r[0],
        lambda rel, src: keyed(rescan(rel, src)))


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


def split_arms() -> list[str]:
    """The Issue 822 split, two-sided, against a real git tree.

    UNCOMMITTED alone passes on an implementation that ignores HEAD and MASKED
    alone on one that ignores the worktree, so both directions are asserted.
    The rows carry a line number, so the LINE-FREE key gets its own arm.

    ⛔ The git-tree arms need a ≥3.10 host: the fixture's `match` statement is
    the SYNTAX class there, and SYNTAX is the only bucket that flows through
    the split. On a 3.9 host the floor IS the native grammar — every failure
    is UNPARSED, the bucket is unreachable, and running the arms would certify
    nothing. The posture line in `main()` names this on every run.
    """
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    BAD = ("def pick(x):\n"
           "    match x:\n"
           "        case 1:\n"
           "            return 1\n"
           "    return None\n")
    OK = "def pick(x):\n    return 1\n"

    # ── the key is LINE-FREE ────────────────────────────────────────────────
    a = keyed([("f.py", "SYNTAX", "12: invalid syntax")])
    b = keyed([("f.py", "SYNTAX", "99: invalid syntax")])
    check(a[0][0] == b[0][0],
          f"the row key is line-bearing: {a[0][0]} vs {b[0][0]}")
    # ...and two IDENTICAL rows in one file stay distinct, while the same
    # text in a DIFFERENT file shares the address but restarts the ordinal
    # (the counter is per-address, so a new site elsewhere renumbers nothing).
    two = keyed([("f.py", "SYNTAX", "1: same"),
                 ("f.py", "SYNTAX", "7: same")])
    check(two[0][0] != two[1][0],
          f"two identical rows collapsed to one key: {two}")
    other = keyed([("g.py", "SYNTAX", "1: same"),
                   ("f.py", "SYNTAX", "1: same")])
    check(other[0][0] != other[1][0],
          f"two files sharing one text collapsed to one key: {other}")
    check(other[1][0] == two[0][0],
          f"the ordinal is not per-address: {other[1][0]} vs {two[0][0]}")

    if sys.version_info < (3, 10):
        return fails

    def git(cwd, *args):
        subprocess.run(["git", "-C", str(cwd), *args], check=True,
                       capture_output=True)

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "r"
        repo.mkdir(parents=True)
        git(repo.parent, "init", "-q", "-b", "main", "r")
        git(repo, "config", "user.email", "t@t")
        git(repo, "config", "user.name", "t")
        (repo / "committed.py").write_text(BAD, encoding="utf-8")
        (repo / "clean.py").write_text(OK, encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "base")

        def now():
            syn, _unp, _n, _p = psf.scan(repo)
            walk = {str(f.relative_to(repo)).replace(chr(92), "/")
                    for f in tracked_files(repo, "*.py")[0]}
            return syn, adjudicate(repo, syn, walk)

        syn, d = now()
        check(len(d.committed) == 1 and not d.uncommitted and not d.masked,
              f"a clean tree produced a split: {d}")

        # ── UNCOMMITTED: the worktree INVENTS an offender ───────────────────
        (repo / "clean.py").write_text(OK + BAD.split(chr(10), 1)[1],
                                       encoding="utf-8")
        syn, d = now()
        check(len(d.uncommitted) == 1,
              f"UNCOMMITTED direction: {d.uncommitted}")
        check(not d.masked, f"an invented row was also MASKED: {d.masked}")
        # The committed offender in the OTHER file is untouched and still
        # adjudicated — the bucket `split_rows` gets wrong.
        check(len(d.head) == 1 and d.head[0][1] == "committed.py",
              f"the pins' view lost the committed row: {d.head}")

        # ── MASKED: the worktree HIDES a committed offender ─────────────────
        git(repo, "checkout", "--", "clean.py")
        (repo / "committed.py").write_text(OK, encoding="utf-8")
        syn, d = now()
        check(not syn, f"the fixture did not hide the offender: {syn}")
        check(len(d.masked) == 1 and d.masked[0][1] == "committed.py",
              f"MASKED direction: {d.masked}")
        check(len(d.head) == 1,
              f"a MASKED row is missing from the pins' view: {d.head}")

        # ── a STAGED-only file has no HEAD blob ─────────────────────────────
        git(repo, "checkout", "--", "committed.py")
        (repo / "staged.py").write_text(BAD, encoding="utf-8")
        git(repo, "add", "staged.py")
        syn, d = now()
        check([r[1] for r in d.uncommitted] == ["staged.py"],
              f"a staged-only file's row was not UNCOMMITTED: {d.uncommitted}")
        check(all(r[1] != "staged.py" for r in d.masked),
              f"a file absent from HEAD produced a MASKED row: {d.masked}")

        # ── an UNPARSED file at HEAD is neither offender nor clean ──────────
        # It has its own bucket (a direct flag in main) and must never reach
        # the ceiling — a rescan that classified garbage would invent a MASKED
        # row nobody can repair.
        (repo / "broken.py").write_text("def f(:\n", encoding="utf-8")
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "broken")
        (repo / "broken.py").write_text(OK, encoding="utf-8")
        syn, d = now()
        check(all(r[1] != "broken.py" for r in d.masked),
              f"an UNPARSED HEAD blob produced a MASKED row: {d.masked}")

        # ── the WALK guard ──────────────────────────────────────────────────
        # `fnmatch`'s `*` crosses `/`, so the scope glob admits a path the
        # walk excludes; a row invented there reads as MASKED.
        check(head_offenders(set())("committed.py", BAD) == [],
              "a path outside the sweep's own walk produced a row")

    return fails


def selftest() -> list[str]:
    """Pin that the verdict FIRES through the gate's own `scan()`, that the
    controls do NOT, and that the walk's boundaries hold. Each fails silently
    otherwise, and a silent failure reports a clean workspace.

    Measured through `scan()` and not `floor_findings()` on purpose:
    `floor_findings` is already self-tested inside the gate, and what this
    sweep adds is the WALK around it — the half that can go blind per-repo.
    The planted-construct arms are posture-split: on a 3.9 host a newer-only
    file fails the box's own grammar (UNPARSED); on a ≥3.10 host the same file
    is the SYNTAX class.
    """
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    BAD = ("def pick(x):\n"
           "    match x:\n"
           "        case 1:\n"
           "            return 1\n"
           "    return None\n")
    OK = "def pick(x):\n    return 1\n"
    # On a ≥3.10 host the planted file is SYNTAX (1/0); on the 3.9 floor
    # itself it is UNPARSED (0/1). Both are findings; the bucket differs.
    want_syn, want_unp = ((1, 0) if sys.version_info >= (3, 10) else (0, 1))

    def measure(files: dict[str, str], gitignore: str = "", add: bool = True):
        """(syntax_n, unparsed_n, py_files, parsed) over a real git repo.

        A real `git init` + `git add`: `tracked_files` shells out to
        `git ls-files` and falls back to an rglob walk when that set is EMPTY,
        so a temp dir with nothing staged would exercise the FALLBACK and
        certify the branch these arms are not aimed at (the Issue-775 vendor
        arm's exact failure — one exclusion, two code paths).
        """
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            subprocess.run(["git", "-C", str(repo), "init", "-q"],
                           capture_output=True, check=True)
            if gitignore:
                (repo / ".gitignore").write_text(gitignore, encoding="utf-8")
            for rel, text in files.items():
                f = repo / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(text, encoding="utf-8")
            if add:
                subprocess.run(["git", "-C", str(repo), "add", "-A"],
                               capture_output=True, check=True)
            syn, unp, nf, np_ = psf.scan(repo)
            return (sum(len(v) for v in syn.values()),
                    sum(len(v) for v in unp.values()), nf, np_)

    # 1. The floor FIRES, through the walk, with the parse count reflecting it.
    s, u, nf, np_ = measure({"a.py": BAD})
    if (s, u, nf, np_) != (want_syn, want_unp, 1, 0):
        fails.append(f"planted match statement: got syntax={s} unparsed={u} "
                     f"files={nf} parsed={np_}, expected "
                     f"{want_syn}/{want_unp}/1/0")

    # 2. CONTROL: a 3.9-clean file must produce NO finding, or the sweep reds
    #    on every correct file and gets switched off.
    s, u, nf, np_ = measure({"a.py": OK})
    if (s, u, nf, np_) != (0, 0, 1, 1):
        fails.append(f"control: a clean file produced syntax={s} unparsed={u} "
                     f"parsed={np_}")

    # 3. UNPARSED surfaces rather than reading as clean — the trap audit's
    #    lesson, and the one direction where a blind instrument prints a green.
    s, u, nf, np_ = measure({"a.py": "def f(:\n"})
    if (s, u, np_) != (0, 1, 0):
        fails.append(f"an unparseable file did not surface: syntax={s} "
                     f"unparsed={u} parsed={np_}")

    # 4. Walk boundaries, both directions in ONE measurement. TRACKED-only
    #    (Issue 777): an UNTRACKED file is not the repo's population, and a
    #    GITIGNORED one certainly is not. Two tracked files, not one — a count
    #    of 1 cannot distinguish "the untracked file was excluded" from "the
    #    walk collapsed to a single file".
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        subprocess.run(["git", "-C", str(repo), "init", "-q"],
                       capture_output=True, check=True)
        (repo / ".gitignore").write_text("ignored/\n", encoding="utf-8")
        (repo / "a.py").write_text(OK, encoding="utf-8")
        (repo / "b.py").write_text(OK, encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "a.py", "b.py",
                        ".gitignore"], capture_output=True, check=True)
        (repo / "untracked.py").write_text(BAD, encoding="utf-8")
        (repo / "ignored").mkdir()
        (repo / "ignored" / "v.py").write_text(BAD, encoding="utf-8")
        syn, unp, nf, _np_ = psf.scan(repo)
        if nf != 2:
            fails.append(f"walk boundaries: walked {nf} .py, expected 2 (the "
                         f"two tracked yes, the untracked and gitignored no)")
        if syn or unp:
            fails.append(f"walk boundaries: an untracked/gitignored file "
                         f"produced syntax={syn} unparsed={unp} — the sweep "
                         f"would report findings in trees no repo owns")

    # 5. population derivation: BOUNDARY.md + a .git DIRECTORY, both required.
    #    A `git worktree` has a .git FILE and would duplicate its parent.
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        (ws / "real").mkdir()
        (ws / "real" / "BOUNDARY.md").write_text("x", encoding="utf-8")
        (ws / "real" / ".git").mkdir()
        (ws / "no-boundary").mkdir()
        (ws / "no-boundary" / ".git").mkdir()
        (ws / "worktree-shaped").mkdir()
        (ws / "worktree-shaped" / "BOUNDARY.md").write_text("x", encoding="utf-8")
        (ws / "worktree-shaped" / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
        if derive_repos(ws) != ["real"]:
            fails.append(f"population derivation wrong: {derive_repos(ws)}")

        # 6. pin parser: arity ENFORCED, comments stripped.
        pins = ws / "pins.txt"
        pins.write_text("# c\nrepo-a 40 30 0  # trailing\n\n", encoding="utf-8")
        if parse_pins(pins) != {"repo-a": dict(zip(FIELDS, (40, 30, 0)))}:
            fails.append("pin parse: 4-field row not read correctly")
        pins.write_text("repo-a 1 2\n", encoding="utf-8")
        try:
            parse_pins(pins)
            fails.append("pin parse: short row accepted")
        except ValueError:
            pass
    # Issue 822: the split's arms are the ONLY execution this wiring gets —
    # this sweep reports 0 findings workspace-wide at the landing and has no
    # --canary.
    return fails + split_arms()


def main() -> int:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (embedded / detached); keep old behavior

    fails = selftest()
    if fails:
        print("✗ python syntax floor sweep SELFTEST FAILED — instrument "
              "untrustworthy:")
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

    # Same quantities, two files: this sweep re-states the two floors the
    # per-push gate owns for THIS repo. Asserted, not trusted.
    mine = pins.get(REPO_ROOT.name, {})
    for field, owned in (("min_py_files", psf.FLOOR_PY_FILES),
                         ("min_parsed", psf.FLOOR_PARSED)):
        if mine.get(field) != owned:
            print(f"✗ pin drift: {PINS.name} says {field}={mine.get(field)} for "
                  f"{REPO_ROOT.name}, python_syntax_floor_gate owns "
                  f"{owned}. Same quantity, two files — change both.")
            return 1

    print(f"  {psf.posture()}; SYNTAX flows through the Issue-822 split, "
          f"UNPARSED is a direct flag")
    bad = False
    tot_files = tot_parsed = tot_syn = tot_unp = 0
    n_uncommitted = n_masked = 0

    for name in names:
        repo = open_repo(name, WORKSPACE)
        syntax, unparsed, py_files, parsed = psf.scan(repo)

        # ── Issue 822: the DISPLAY reads the worktree, the PINS read HEAD ───
        # A ceiling is a claim about the repo, and a repo's state is its
        # commits. This workspace runs concurrent sessions against shared
        # worktrees, so a row here may sit on a line no commit contains — and
        # re-pinning from such a run bakes another session's in-flight edit
        # into a tracked file, where it reds on every other box.
        walk = {str(f.relative_to(repo)).replace(chr(92), "/")
                for f in tracked_files(repo, "*.py")[0]}
        delta = adjudicate(repo, syntax, walk)
        n_uncommitted += len(delta.uncommitted)
        n_masked += len(delta.masked)
        head_syn = [r for r in delta.head if r[2] == "SYNTAX"]
        held = {r[0] for r in delta.uncommitted}
        hidden = {r[0] for r in delta.masked}
        n_syn = sum(len(v) for v in syntax.values())
        n_unp = sum(len(v) for v in unparsed.values())
        tot_files += py_files
        tot_parsed += parsed
        tot_syn += n_syn
        tot_unp += n_unp

        row = pins.get(name)
        flags = []
        if row is None:
            # Issue 821: an acknowledged known-extra owes no pin row —
            # the marker reached population_verdict's FINAL line and not
            # this loop, so 8 of 9 sweeps red on repos they found
            # nothing in, hiding two live ratchet breaches.
            if not pin_row_exempt(name):
                flags.append("UNPINNED — add a row (or it can never red)")
        else:
            if py_files < row["min_py_files"]:
                flags.append(f"walk FLOOR breached: {py_files} tracked .py < "
                             f"{row['min_py_files']} — files were removed, or "
                             f"`git ls-files` went blind and the 0 below means "
                             f"nothing")
            if parsed < row["min_parsed"]:
                flags.append(f"parse FLOOR breached: {parsed} of {py_files} "
                             f"cleared the floor parse < {row['min_parsed']} — "
                             f"the walk is intact but the scan stopped early")
            if len(head_syn) > row["max_syntax"]:
                flags.append(f"SYNTAX {len(head_syn)} committed > pinned "
                             f"{row['max_syntax']}")
        if unparsed:
            # Direct flag, never a pinned ceiling: the HEAD rescan cannot
            # classify a blob that parses at no grammar, so a ceiling fed from
            # `delta.head` would be dead plumbing (the docstring's argument).
            flags.append(f"{len(unparsed)} finding(s) fail the box's own "
                         f"grammar — UNPARSED is the instrument admitting it "
                         f"cannot read, never a pass")

        status = "✗" if flags else ("·" if (n_syn or n_unp) else "✓")
        # T3: the count stays honest in BOTH directions. A bare total invites
        # the one action Issue 822 exists to prevent — typing it into the pin.
        split = ""
        if delta.uncommitted or delta.masked:
            split = (f" ({len(head_syn)} committed"
                     + (f", {len(delta.uncommitted)} uncommitted"
                        if delta.uncommitted else "")
                     + (f", {len(delta.masked)} MASKED" if delta.masked else "")
                     + ")")
        print(f"{status} {name:22s} py={py_files:<4d} parsed={parsed:<4d} "
              f"syntax={n_syn} unparsed={n_unp}{split}")

        def _tag(key, _held=held, _hidden=hidden):
            # Never hidden — hiding them is the lie Issue 797 refuses — but
            # labelled, so the reader sees which rows the pin did not read.
            if key in _held:
                return "  [UNCOMMITTED — not adjudicated]"
            if key in _hidden:
                return "  [MASKED — committed, and this worktree hides it]"
            return ""

        for key, rel, _kind, r in keyed(flatten(syntax)):
            print(f"      ⛔ SYNTAX  {rel}:{r}{_tag(key)}")
        # A MASKED row is in HEAD and NOT in the worktree, so the loop above
        # cannot reach it — it has no line here to hang a label on.
        for key, rel, _kind, r in delta.masked:
            print(f"      ⛔ SYNTAX  {rel}:{r}  [MASKED — committed, and this "
                  f"worktree hides it]")
        for rel in sorted(unparsed):
            for r in unparsed[rel]:
                print(f"      ⛔ UNPARSED {rel}:{r}")
        for f in flags:
            bad = True
            print(f"      ✗ {f}")

    # The population axis, shared (Issues 793 + 782): UNREGISTERED reds in every
    # posture, UNSEEN reds without the marker, and the same set DEFERS loudly
    # with it. Never auto-detected — a genuine removal whose row update was
    # forgotten is set-identical to a partial clone from the walk alone.
    pop_lines, deferred, pop_fail = population_verdict(pins, names)

    # Issue 797 — the worktree is not the repo. This run reads files that
    # concurrent sessions are editing, so a finding may sit on a line no
    # commit contains. ADVISORY, never a failure: a sweep that hard-reds on
    # an ordinary dirty worktree is a sweep nobody runs. It rides the FINAL
    # line in BOTH directions (the `deferred` precedent) and is SILENT
    # unless the dirty set meets this sweep's own population — the .py files.
    deferred.extend(sweep_advisory(
        names, ("*.py",), root=WORKSPACE,
        uncommitted_rows=n_uncommitted, masked_rows=n_masked))
    for _line in pop_lines:
        print(_line)
    if pop_fail:
        bad = True

    print(f"\n{len(names)} contract repo(s) · {tot_files} tracked .py · "
          f"{tot_parsed} parsed at 3.9 · {tot_syn} SYNTAX · {tot_unp} UNPARSED")
    # State the scope where it is READ, not only in the docstring.
    print("  scope: the 3.9 GRAMMAR (ast.parse feature_version + the gate's "
          "measured AST supplements), not the interpreter — tomllib imports "
          "(3.11+) parse fine at 3.9, and PEP 604 in annotation position "
          "rides `from __future__ import annotations`. A newer-only file is "
          "dead on the boxes' python3 3.9.6 at the documented direct "
          "invocations, never on a newer box.")

    if bad:
        print("✗ python syntax floor sweep FAILED — see the ✗ rows above")
        for _d in deferred:
            print(f"  {deferral_line(_d)}")
        print("    The repair is rewriting the construct at 3.9 grammar "
              "(no match statement, no PEP 701 f-string backslash, no starred "
              "subscript). Do NOT raise a ceiling — a file that cannot parse "
              "cannot run; see Issue 965.")
        return 1
    _line = "✓ python syntax floor sweep PASSED — every repo within its pins"
    if deferred:
        _line += "; DEFERRED: " + "; ".join(deferred)
    print(_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
