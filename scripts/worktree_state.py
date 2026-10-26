#!/usr/bin/env python3
"""A sweep reads the WORKTREE, so a finding may exist in NO commit — ONE copy.

Issue 797. Every instrument in the cross-repo sweep family walks the working
tree. This workspace runs five-plus concurrent agent sessions against SHARED
worktrees — `staged_set_audit.py` exists for exactly that hazard one axis over
— so a row a sweep prints may sit on a line no commit contains, and a repo a
sweep calls clean may be clean only because somebody's uncommitted edit removed
the offending line.

Measured 2026-09-15, `citation_drift_sweep.audit()` run twice per dirty repo
(worktree, then every dirty in-scope document replaced by its HEAD blob): the
workspace's **entire** standing CROSS finding — 1 of 1 — was an artifact of an
uncommitted edit that stripped a `riir-train` qualifier HEAD carries. It had
already cost a session, carried across a context boundary as backlog reading
"blocked, that session has HISTORY.md uncommitted". The correct verdict was
not *blocked*; it was **there is nothing to fix**, and no amount of reading the
sweep's own output could say which.

Three verdicts, never interchangeable:

- **COMMITTED** — the finding's file matches HEAD. An ordinary finding:
  counted, pinnable, somebody's to repair.
- **UNCOMMITTED** — the file differs from HEAD, so the row may exist in no
  commit. **Displayed** (it is what the file says today, and hiding it would be
  its own lie) but **never adjudicated against a pin** — a ceiling breached by
  somebody's in-flight edit is a red nobody can repair, and a ceiling re-pinned
  from one bakes that edit into a tracked expectations file, where it reds on
  every other box. The split of responsibility, once: the DISPLAY reads the
  worktree, the PINS read HEAD.
- **MASKED** — HEAD carries a row the worktree does not. A *false green*: the
  defect is committed, in the repo, and the sweep says the repo is clean. This
  is the silent direction and therefore the worse one. Measured 0 today, which
  is a measurement and not an absence of the class.

The POPULATION moves too, which a row-level read alone misses: the same two
runs put `n_cites` at **601 (worktree) vs 607 (HEAD)**, because the other
session's uncommitted deletion of a 30-line block took six citations out of the
denominator. A floor re-pinned from such a run is wrong everywhere else.

⚠ **ADVISORY, never a failure.** A sweep that hard-reds on an ordinary dirty
worktree is a sweep nobody runs — the cries-wolf outcome AGENTS.md names for
`.benchmarks/` in the numbering gate. The advisory rides the sweep's FINAL line
in BOTH directions (the `DEFERRED` precedent in `sweep_population.py`: a notice
printed only on failure is one nobody reads on the run that passes), and it is
SILENT when nothing dirty intersects the sweep's own population — a banner that
prints on every run is a banner nobody reads either.

Self-test: `scripts/worktree_state.py` (exit 1 on failure). Every arm is
two-sided where two sides exist — an arm whose perturbation reds nothing
certifies nothing (`platform_dead_code_audit.py`'s first `vendor/` arm).
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

# Issue 804: this instrument is documented as directly invokable, and its
# verdict glyphs (✓ ✗ ⛔ ⚠) kill it on a non-UTF-8 console — no verdict at
# all, findings unread. docs_gate.sh's PYTHONIOENCODING only covers runs
# that go through the wrapper.
import console_safe  # noqa: E402

# Issue 842: `repos` arrive as CONTRACT names, the directories are the
# on-disk spellings, and the advisory's scope keys are printed. Resolves and
# displays through the codec so an alias box is measured (not silently
# skipped) and the alias content itself never reaches stdout.
import repo_alias  # noqa: E402

console_safe.apply()

COMMITTED = "COMMITTED"
UNCOMMITTED = "UNCOMMITTED"
MASKED = "MASKED"


def _git(root, *args) -> subprocess.CompletedProcess:
    """`encoding=` explicitly, never `text=True` — Issue 778: `text=True`
    decodes with the SYSTEM locale, and every path here may carry one."""
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, encoding="utf-8", errors="replace")


def is_checkout(root) -> bool:
    """Can git be run rooted HERE? — `.exists()`, deliberately (Issue 836).

    ⛔ This is NOT the contract-repo predicate and the two must not be
    interchanged. `.git` answers two different questions in this workspace and
    each spelling is wrong for the other one:

    - *Is this a canonical REPO?* → `.is_dir()`. A `git worktree` has a `.git`
      **FILE**, and counting one as a repo attributes its manifests to a repo
      that does not exist (Issue 835 defect 1). That predicate is
      `skill_repo_set_gate.derive_repos`; never re-derive it here.
    - *Can I run git rooted here?* → `.exists()`, this function. A worktree is
      a perfectly good checkout, and `.is_dir()` silently says otherwise.

    All five guards below took `.is_dir()`, and every one of their false
    branches returns the value meaning *nothing to report* — so in a worktree
    the whole head-provenance mechanism went inert with no signal anywhere,
    and a sweep re-pinned from another session's in-flight edits believing it
    had read HEAD. That is Issue 797's founding defect, reintroduced inside
    the mechanism built to prevent it.

    The requirement this probe actually exists for is unaffected, and it is
    asserted one arm group down: `git -C` walks UP, so a non-repo directory
    nested inside a repo must not inherit its PARENT's answers. Such a
    directory has no `.git` entry of ANY kind, so `.exists()` protects it
    exactly as `.is_dir()` did — as does a `git archive` extraction, which
    stays clean-by-construction rather than an error. The change is strictly
    widening, on the one case that should have answered all along.
    """
    return (Path(root) / ".git").exists()


# ⛔ Restored: `6c6ca2ee` deleted this while reworking this module's own
# arms, and its own docstring below says why it is PUBLIC — two external
# gates' canaries build their worktree arm from it, and both crashed with
# an AttributeError from the moment it went. The same commit privatised
# `is_checkout` for the same reason and broke the same two callers.
def worktree_fixture(tmp: Path, tracked: str = "scripts/a.py",
                     untracked: str = "scripts/b.py") -> tuple[Path, Path]:
    """-> `(main_repo, worktree)`, a REAL `git worktree`. Issue 836 T2.

    Public, because the three delegating consumers each owe an arm proving
    they take the git branch in a worktree, and that arm needs exactly this
    fixture. The alternative is three hand-written `.git` files, and the
    fixture's own first assertion is why that is not an option: a
    worktree-SHAPED `.git` would assert the probe and not the behaviour —
    the `platform_dead_code` vendor-arm failure, where one exclusion had two
    code paths and the arm certified the path it was not aimed at.

    `untracked` is what makes a consumer's arm DISCRIMINATING rather than
    merely green: in the worktree, `git ls-files` yields `tracked` alone while
    a filesystem glob yields both, so a consumer still on the `.is_dir()`
    spelling returns a set one member too large and the arm names it.
    Measured at **0.167s**, which is what makes it affordable in a per-push
    gate's canary at all (the docs gate's whole budget is ~13s CPU).

    ⚠ The caller owns the `TemporaryDirectory`; this only fills it.
    """
    main = _repo(tmp, "main")
    t = main / tracked
    t.parent.mkdir(parents=True, exist_ok=True)
    # Path.write_text's newline= kwarg is 3.10+; open() has had it forever —
    # this fixture must run on the boxes' python3 3.9.6 (the
    # dual_allocation_gate.py:261 rule, generalised 2026-10-26, Issue 965).
    with open(t, "w", encoding="utf-8", newline="") as _t:
        _t.write("x = 1\n")
    _run(main, "add", "-A")
    _run(main, "commit", "-qm", "base")

    wt = tmp / "wt"
    _run(main, "worktree", "add", "--detach", "-q", str(wt), "HEAD")
    u = wt / untracked
    u.parent.mkdir(parents=True, exist_ok=True)
    with open(u, "w", encoding="utf-8", newline="") as _u:
        _u.write("y = 1\n")
    return main, wt


def dirty_files(root) -> frozenset[str]:
    """Tracked repo-relative paths that differ from HEAD (staged or not).

    - The `.git` probe is load-bearing, exactly as in `tracked_walk.py`:
      `git -C` walks UP until it finds a repository, so a non-repo directory
      nested inside one would otherwise answer with its PARENT's dirty set.
    - A tree with no `.git` — a `git archive` extraction, a synthetic fixture —
      is **clean by construction**, so the answer is the empty set and NOT an
      error. `tracked_walk.py`'s fallback precedent.
    - Untracked files are excluded: a sweep's population is what git tracks
      (Issue 777), so an untracked file is not in it to begin with.
    - A rename reports its DESTINATION. The source no longer exists on disk, so
      no finding can carry its address.
    """
    root = Path(root)
    if not is_checkout(root):
        return frozenset()
    out = _git(root, "status", "--porcelain", "--untracked-files=no")
    if out.returncode != 0:
        return frozenset()
    rels: set[str] = set()
    for line in out.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        # `-> ` splits a rename; the quoting is git's own for non-ASCII paths.
        if " -> " in path:
            path = path.split(" -> ")[-1]
        # The `.replace` is DEFENSIVE, and its arm says so. `git status
        # --porcelain` emits POSIX separators on every platform — measured on
        # the Windows workstation, where a naive reading would expect `\` — so
        # deleting this line reds NOTHING. An arm whose perturbation reds
        # nothing certifies nothing, so the arm below asserts git's OUTPUT
        # SHAPE (the premise) rather than pretending to test this line. The
        # real normalisation that does bite is in `split_rows`, on the
        # CALLER's path, which may genuinely be a `pathlib` Windows string.
        rels.add(path.strip('"').replace("\\", "/"))
    return frozenset(rels)


def head_text(root, rel: str) -> str | None:
    """HEAD's blob for a repo-relative path, or None.

    None means BOTH "not in HEAD" (a newly added file — there is no committed
    version for a MASKED comparison to read) and "no repository". The callers
    treat those the same way, because both say the same thing: this run has
    nothing committed to compare against.
    """
    root = Path(root)
    if not is_checkout(root):
        return None
    out = _git(root, "show", f"HEAD:{rel}")
    return out.stdout if out.returncode == 0 else None


def split_rows(rows, path_of, dirty: frozenset[str]) -> tuple[list, list]:
    """(committed, uncommitted) for a caller's finding rows.

    `path_of(row)` yields the row's repo-relative path. A row whose path this
    run cannot determine counts as COMMITTED — the conservative direction for
    a bucket whose whole purpose is to WITHHOLD rows from the count.
    """
    keep, held = [], []
    for row in rows:
        rel = path_of(row)
        (held if rel and rel.replace("\\", "/") in dirty else keep).append(row)
    return keep, held


def _coerce(patterns) -> tuple:
    """A bare string is a FOOTGUN, not a convenience: iterating it yields
    single characters, `fnmatch(rel, "*")` matches everything, and the advisory
    silently reports every dirty file in the repo as being in scope. Caught in
    this module's own wiring commit, where 8 of 15 call sites had written
    `("*.rs")` without the tuple comma."""
    if isinstance(patterns, str):
        return (patterns,)
    return tuple(patterns)


def _matches(rel: str, patterns) -> bool:
    """ONE path against a sweep's population globs — the single copy.

    Extracted from `_match_count` when `head_delta` needed the same predicate
    per path rather than as a total. A second implementation of "is this file
    in the sweep's population" is how the advisory and the row split would come
    to disagree about which rows they are talking about.
    """
    from fnmatch import fnmatch
    base = rel.rsplit("/", 1)[-1]
    return any(fnmatch(rel, pat) or fnmatch(base, pat)
               for pat in _coerce(patterns))


def _match_count(rels, patterns) -> int:
    """How many repo-relative paths a sweep's own walk would have read.

    `patterns` are `fnmatch` globs matched against the repo-relative POSIX path
    AND against the basename, because a sweep's population is usually stated as
    a bare suffix (`*.rs`) while its paths are nested. Matching only the full
    path would report 0 for every sweep in the family and make the advisory
    silently vacuous — the green-zero shape this whole family exists to refuse.

    ONE copy, shared by the dirty-vs-HEAD and behind-vs-origin axes. They must
    not diverge: a git PATHSPEC was the obvious implementation for the second
    and has different semantics from `fnmatch` (a bare `Dockerfile` pathspec
    matches only at the root), so the two axes would have disagreed about what
    a sweep's population IS.
    """
    patterns = _coerce(patterns)
    return sum(1 for rel in rels if _matches(rel, patterns))


def dirty_in_scope(root, patterns) -> int:
    """How many of a repo's dirty files this sweep's own walk would have read."""
    return _match_count(dirty_files(root), patterns)


class HeadDelta(NamedTuple):
    """`head_delta`'s answer. Unpacks as `(committed, uncommitted, masked)`.

    `.head` is the reason this is not a bare tuple. The rows HEAD carries are
    `committed + masked`, NOT `committed` — a MASKED row is by definition one
    the worktree does not show — and that sum is what a PIN adjudicates. Every
    caller getting it right independently is 14 chances to understate a
    ceiling by exactly the silent direction this class exists to surface, so
    the arithmetic lives here once.
    """

    committed: list
    uncommitted: list
    masked: list

    @property
    def head(self) -> list:
        """What a COMMIT of this repo would produce — the pins' population."""
        return list(self.committed) + list(self.masked)


def _rel_of(path) -> str | None:
    """A caller's row address as a repo-relative POSIX string, or None.

    The normalisation bites on the CALLER's side, not on git's: `git status
    --porcelain` emits POSIX separators on every platform (asserted as a
    premise by `premise_arms`), while a row built from `pathlib` on this
    workstation carries backslashes.
    """
    if not path:
        return None
    return str(path).replace(chr(92), "/")


def dirty_in_population(root, patterns) -> frozenset[str]:
    """The dirty files this sweep's own walk would have READ — the set whose
    COUNT `dirty_in_scope` prints on the advisory line. One predicate, two
    shapes, so the banner and the row split cannot disagree."""
    return frozenset(r for r in dirty_files(root) if _matches(r, patterns))


def head_delta(root, patterns, rows, path_of, key_of, rescan):
    """(committed, uncommitted, masked) for a sweep's FILE-ADDRESSED rows.

    Issue 822 T2. `split_rows` above answers half the question — it withholds
    the rows whose file is dirty — and a sweep that stops there can only ever
    report *fewer* findings than it has. It cannot see MASKED (HEAD carries a
    row the worktree hides: a committed defect reported clean, the silent
    direction), and it withholds a row that HEAD carries too, which understates
    the number the pins are supposed to adjudicate. `citation_drift_sweep` gets
    the full three-way answer by re-running its ENTIRE classifier with an
    injected reader; this is that pattern, once, for the other thirteen.

    ⚠ **THE PREMISE IS PER-FILE ROW INDEPENDENCE**, and it is what makes this
    affordable. A row's existence must depend only on its OWN file's bytes —
    then every clean file's rows are identical at HEAD by construction, and the
    only files that need re-reading are the dirty ones IN this sweep's own
    population. Issue 822 T2 asked whether re-reading is affordable for a
    2415-file Rust walk; the answer is that the question was the wrong shape.
    The cost is |dirty ∩ population| `git show` calls — the quantity
    `dirty_in_scope()` already prints on the advisory line, typically 0 and
    measured in this workspace at a handful. A tree-sized re-walk buys nothing.

    ⛔ So a CROSS-FILE classifier must NOT use this: `len_derived` resolves a
    caller's provenance through other files (and other repos),
    `instrument_reachability` computes a closure from roots, `numbering` and
    `citation` are cross-document by construction. For those the whole-run
    re-classification with an injected `read` is the correct instrument and
    this shortcut would invent verdicts. The premise is a property of the
    CALLER's classifier, which no check here can decide — so it is stated, not
    asserted.

    `rescan(rel, head_src)` re-classifies ONE file from HEAD's bytes and
    returns that file's rows. `head_src` is None when the path is absent from
    HEAD (a STAGED-but-never-committed file — the measured case: Issue 822's
    breach included a row on a file `git log` cannot see at all), and a caller
    must answer None with no rows rather than crash.

    `key_of(row)` must be LINE-FREE for the same reason `citation_drift_sweep`'s
    is: any edit above a finding shifts its line, so a line-bearing key reports
    every row in an edited file as UNCOMMITTED *and* MASKED at once.
    """
    scope = dirty_in_population(root, patterns)
    rows = list(rows)
    if not scope:
        # The common case, and it must cost NOTHING: no `git show`, no rescan.
        # A helper that walks anyway on a clean tree is one sweeps drop.
        # ⚠ This guard is provably EQUIVALENT to falling through (an empty
        # scope puts every row in `committed` anyway), so no perturbation of it
        # reds an arm — measured. It is here for the COST, and the arm that
        # bites is the one asserting zero rescan calls.
        return HeadDelta(rows, [], [])

    head_rows: list = []
    seen: set = set()
    for rel in sorted(scope):
        for row in rescan(rel, head_text(root, rel)) or ():
            k = key_of(row)
            # Deduplicated by KEY, because MASKED is walled at 0 by its
            # callers: a non-injective key would red a sweep repeatedly for one
            # committed row, and a ceiling that cries wolf is one nobody runs.
            if k not in seen:
                seen.add(k)
                head_rows.append(row)

    live = {key_of(r) for r in rows if _rel_of(path_of(r)) in scope}
    committed, uncommitted = [], []
    for row in rows:
        if _rel_of(path_of(row)) in scope and key_of(row) not in seen:
            uncommitted.append(row)
        else:
            # An address-less row, a row on a clean file, and a row HEAD
            # carries too all land here — `split_rows`'s conservative
            # direction, for its reason: this bucket is the one the pins read.
            committed.append(row)
    masked = [r for r in head_rows if key_of(r) not in live]
    return HeadDelta(committed, uncommitted, masked)


def head_overlay(root, patterns) -> dict[str, str | None]:
    """{repo-relative path: HEAD's bytes} for every dirty file in `patterns`.

    The CROSS-FILE half of Issue 822 T2. `head_delta` is unsound where a row's
    existence depends on another file's bytes — `instrument_reachability`
    computes a closure from roots, so a dirty `AGENTS.md` changes OTHER
    scripts' verdicts — and such a classifier has to be re-run whole against
    HEAD. This is the input that re-run needs, and the reason it is a helper
    rather than four lines at each call site is that all four of its subtleties
    have already been got wrong once somewhere in this family:

    - **`None` is a VALUE, not an absence.** It means "tracked but not in
      HEAD" — a staged-but-never-committed file, Issue 822's measured case —
      and a caller that cannot distinguish it from empty text will classify
      another session's in-flight file as a committed finding. `in` and
      `.get()` say different things here; use `in`.
    - **An EMPTY dict means skip the second classification entirely**, not
      "overlay nothing". A sweep that re-runs its classifier unconditionally
      doubles its cost on every clean run, which is how a helper stops being
      called.
    - The patterns must name every file that can CHANGE a verdict, not just
      the ones findings sit on. For a closure that is the roots as well.
    - `dirty_files` excludes untracked files, so an untracked script cannot
      appear here — which is correct: a sweep's population is what git tracks.
    """
    return {rel: head_text(root, rel)
            for rel in sorted(dirty_in_population(root, patterns))}


@contextlib.contextmanager
def head_tree(root, patterns, paths=None, extra_dirty=()):
    """A throwaway CHECKOUT of HEAD, or `None` when nothing in scope is dirty.

    Issue 822 T5g — the third instrument, for classifiers the other two cannot
    reach. `head_delta` substitutes one file's bytes; `head_overlay` hands a
    whole-run re-classification `{rel: bytes}`. Both assume the classifier
    takes its text from ONE place a caller can intercept. Measured on
    `wasm32_surface_audit`, that assumption fails: it shells out to `git grep`
    for the walk, `git ls-files` twice for the manifests and the lane rows, and
    reads files directly besides — four seams, and an overlay that misses any
    one of them produces a verdict built half from HEAD and half from the
    worktree, which is worse than either.

    So this materialises HEAD instead and lets the classifier run unmodified:
    `git archive HEAD`, extracted, then `git init` + `git add -A -f` so the
    tree answers `git grep` and `git ls-files` exactly as a real checkout does.
    `-f` is load-bearing — an extracted `.gitignore` would otherwise exclude
    content HEAD tracks, and `git archive` emits tracked content only, so
    everything in it belongs in the index by construction. No commit is made
    (`pipefail_discard_drift_sweep._temp_repo`'s precedent: `git grep` reads
    the index, and a commit buys nothing).

    ⚠ It is the EXPENSIVE one — a tree copy rather than a handful of `git
    show` calls — so it yields `None` on a clean run and the caller must skip
    its second classification entirely, exactly as an empty overlay means
    SKIP. `None` also covers "not a git repository" and an archive that
    failed: in both the honest answer is that this run has nothing committed
    to compare against, never a fabricated empty tree, which would report
    every finding as UNCOMMITTED.

    `paths` — optional git PATHSPECS narrowing what the archive carries, and
    the difference is not marginal. Measured on riir-train (T5j): the whole
    tree is **604 MB / 2.1s to archive and 1.0s to extract**, against **30.6 MB
    / 0.13s / 0.37s** for `'*.rs' '*.toml'`, and the `git add -A` underneath
    scales with the same file count — one dirty file took a 1.9s sweep to
    **36.6s** unnarrowed.

    ⛔ **The narrowing is the caller's RISK, not a tuning knob, and it is the
    same hazard as an overlay that misses a seam.** A file HEAD carries and the
    pathspec drops is ABSENT from the tree, so a classifier that reads it
    silently classifies against a tree that never existed — and unlike a
    missing overlay entry there is no `None` to notice. Pass paths only where
    the classifier's inputs are ENUMERABLE (a manifest+source join is;
    anything reaching `git grep` over the whole tree is not), and state them
    from the same constant the sweep's SCOPE comes from. The default is the
    whole tree, which is always correct.

    `extra_dirty` — paths that must count as dirt even though `git status` does
    not report them. ⛔ It exists for ONE measured shape: a sweep whose WALK
    includes UNTRACKED files. `dirty_files` excludes untracked by design ("a
    sweep's population is what git tracks", Issue 777) and that is right for
    every caller whose walk is `git ls-files` — but `restatement_theorem_audit`
    uses `os.walk`, so an untracked `.lean` is IN its population, is in no
    commit by definition, and produces zero `git status` dirt. Without this the
    trigger never fires, no HEAD exists to compare against, and the row is
    filed COMMITTED: Issue 822's defect in its purest form, which
    `markdown_fence` had to split by hand one instrument over. Nothing here
    needs the paths to exist or to be read — they only widen the trigger, and
    the tree itself is still HEAD, where an untracked file correctly does not
    appear.
    """
    root = Path(root)
    if not (dirty_in_population(root, patterns) or tuple(extra_dirty)) \
            or not is_checkout(root):
        yield None
        return
    with tempfile.TemporaryDirectory() as td:
        arc = Path(td) / "head.tar"
        spec = ["--", *_coerce(paths)] if paths else []
        if _git(root, "archive", "-o", str(arc), "HEAD", *spec).returncode != 0:
            yield None
            return
        # ⛔ Named after the SOURCE repo, not "head". A classifier that takes a
        # repo's identity from its directory name — `len_derived_binding_audit`
        # resolves provenance through workspace CALLERS and keys every row on
        # it — would otherwise see every materialised repo called `head`, which
        # both collides when two are materialised at once and misattributes
        # every row when one is. The temp directory already makes the path
        # unique; the leaf is free to be honest.
        dest = Path(td) / root.name
        dest.mkdir()
        try:
            with tarfile.open(arc) as tf:
                try:
                    tf.extractall(dest, filter="data")
                except TypeError:        # filter= predates 3.12
                    tf.extractall(dest)
        except (tarfile.TarError, OSError):
            yield None
            return
        _run_quiet(dest, "init", "-q")
        _run_quiet(dest, "add", "-A", "-f")
        yield dest


def _run_quiet(cwd, *args) -> None:
    subprocess.run(["git", "-C", str(cwd), *args],
                   capture_output=True, encoding="utf-8", errors="replace")


def line_free(text: str) -> str:
    """One finding row's text with a leading `"<lineno>: "` removed.

    Issue 822 T5d. The family's rows are overwhelmingly `f"{node.lineno}: {…}"`
    or `f"{lineno}: {…}"`, and a key built from that reports EVERY row in an
    edited file as UNCOMMITTED *and* MASKED at once, because any insertion
    above a finding shifts it. Written once here rather than in nine sweeps:
    the split is by the FIRST `": "` and only when what precedes it is a bare
    integer, so a row whose text legitimately begins `"note: …"` is returned
    unchanged rather than silently beheaded.
    """
    head, sep, rest = text.partition(": ")
    if sep and head.strip().isdigit():
        return rest
    return text


def ordinal_keys(rows, addr_of):
    """`[(key, row)]` — each row's ADDRESS plus an ordinal within it.

    Issue 822 T5d, lifted out of `subprocess_encoding_drift_sweep` (T5c) the
    first time a second sweep needed it, per this repo's own most-repeated
    rule: a rule landed in one instrument and never generalised is the failure
    mode recorded nine times in AGENTS.md.

    A line-free address is not unique in general — two identical calls in one
    file share one — so an ORDINAL disambiguates, the
    `len_derived_eyes_expected.txt` precedent, scoped to the WHOLE address so
    a new site elsewhere renumbers nothing.

    ⛔ The counter is LOCAL to this call, and that is the load-bearing part.
    Sharing one counter between the worktree pass and the HEAD pass makes the
    two sides count from different bases, and then every row looks moved.
    """
    seen: dict = {}
    out = []
    for row in rows:
        addr = tuple(addr_of(row))
        n = seen.get(addr, 0)
        seen[addr] = n + 1
        out.append(((*addr, n), row))
    return out


def delta_of(worktree_rows, head_rows, key_of) -> HeadDelta:
    """The three buckets from two ROW SETS, for a classifier re-run WHOLE.

    `head_delta`'s tail does the same arithmetic against a scoped comparison,
    because there the HEAD rows come only from the dirty files. Here both sides
    are complete, so the comparison is complete too — and pooling the two
    implementations would silently apply one's scoping rule to the other's
    input.
    """
    wk = {key_of(r) for r in worktree_rows}
    hk = {key_of(r) for r in head_rows}
    return HeadDelta([r for r in worktree_rows if key_of(r) in hk],
                     [r for r in worktree_rows if key_of(r) not in hk],
                     [r for r in head_rows if key_of(r) not in wk])


def behind_origin(root, patterns) -> tuple[int, int] | None:
    """`(commits behind upstream, how many of those touch this population)`.

    Issue 798. The dirty-vs-HEAD axis above compares the worktree to LOCAL
    HEAD, so it is blind to a checkout that matches its own HEAD and is 109
    commits behind origin — and a sweep reads the WORKTREE. Measured: riir-ai's
    `toolchain-override-deliberate` marker was committed upstream at
    `194cdc9b5` while this box's riir-ai sat 109 commits back, so the sweep
    reported `drift 1 · ✗ FAILED` on a defect that was already FIXED. That is
    a false RED, and the cries-wolf cost is the one this family refuses to pay.
    It is the mirror of `MASKED`: MASKED is a committed defect read clean, this
    is a committed FIX read dirty.

    `None` — no upstream configured, or git could not answer. Never guessed at:
    assuming `origin/main` would invent a verdict for a repo whose branching
    model nobody here knows (five of the workspace's repos have no `origin/main`
    at all — `ci_gate_coverage.py`'s standing finding).
    `(0, 0)` — up to date, the ordinary case, and SILENT downstream.

    The scoped count uses the three-dot `HEAD...ref` diff — changes on the
    UPSTREAM side since the merge base. A two-dot diff would also report every
    file this checkout's own unpushed commits touched, so a repo that is merely
    AHEAD would read as stale.
    """
    root = Path(root)
    if not is_checkout(root):
        return None
    up = _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name",
              "@{upstream}")
    ref = up.stdout.strip()
    # Both halves of this `or` -- and of the one below -- are DEFENSIVE, and
    # the arm says so. Under real git they always co-occur: a `rev-parse
    # @{upstream}` with no upstream exits non-zero AND prints nothing to
    # stdout, so flipping either `or` to `and` reds NOTHING and both show as
    # live survivors in `arm_reach_audit --include-all`. `dirty_files` met the
    # identical shape with its separator normalisation and resolved it the
    # same way: `premise_arms` asserts git's OUTPUT SHAPE, which is the thing
    # that could actually change under us, instead of pretending to test a
    # line no input can distinguish.
    if up.returncode != 0 or not ref:
        return None
    cnt = _git(root, "rev-list", "--count", f"HEAD..{ref}")
    if cnt.returncode != 0 or not cnt.stdout.strip().isdigit():
        return None
    total = int(cnt.stdout.strip())
    if total == 0:
        return (0, 0)
    diff = _git(root, "diff", "--name-only", f"HEAD...{ref}")
    if diff.returncode != 0:
        return None
    rels = [ln.strip().strip('"').replace("\\", "/")
            for ln in diff.stdout.splitlines() if ln.strip()]
    return (total, _match_count(rels, patterns))


# How long a `(0, 0)` "up to date" reading stays credible. NOT a tuning knob:
# measured across this workspace 2026-09-18, twelve of sixteen repos had
# fetched within 1-2h, while riir-viewbridge's `behind=0` rested on a 60h-old
# fetch and seal-game-editor reported `0 behind / 0 ahead` on a ~36h-old one
# while hiding 260 commits — the gap that produced Issue 827's false red in a
# THIRD repo. 24h sits above the ordinary working rhythm and below both
# measured failures.
STALE_FETCH_HOURS = 24.0


def fetch_age_hours(root) -> float | None:
    """Hours since this checkout last heard from its remote, or `None`.

    Issue 827 T5. `behind_origin()` compares HEAD to a LOCAL remote-tracking
    ref, so its answer is only as fresh as the last fetch: `(0, 0)` means "up
    to date as of whenever anybody last fetched", and nothing downstream
    distinguishes that from "up to date now". **A staleness detector that never
    fetches does not measure staleness — it measures staleness KNOWN as of the
    last fetch, and it returns the reassuring answer exactly when it knows
    least.** That is this repo's own blindness-floor shape (an instrument whose
    failure mode is a confident zero) reached by the advisory itself.

    ⛔ `None` means CANNOT TELL, never "fresh". A tree with no `.git`, or one
    cloned and never fetched, has no `FETCH_HEAD`; treating that as age 0 would
    invent the exact reassurance this exists to withdraw. Callers must fold it
    into the LOUD bucket, which is what `sweep_advisory` does.

    Reads a MTIME and makes no network call — deliberately. Fetching to answer
    this would mutate another session's refs to make our own verdict true,
    which on a box running five concurrent sessions is a race; Issue 827 T2
    refuses it for the sweeps on the same grounds.
    """
    if not is_checkout(root):
        return None
    # ⛔ `root / ".git" / "FETCH_HEAD"` is NOT the path in a worktree, where
    # `.git` is a FILE and the real git directory lives under the main
    # checkout's `.git/worktrees/<name>/`. Admitting worktrees via
    # `is_checkout` without this would swap one silent None for another —
    # `os.stat` on a path under a FILE raises `OSError` and reads as "cannot
    # tell", the same "nothing to report" value Issue 836 is about. Ask git
    # for the path instead of constructing it; `--git-path` resolves the
    # per-worktree and shared cases alike, and is a no-op string lookup.
    loc = _git(root, "rev-parse", "--git-path", "FETCH_HEAD")
    if loc.returncode != 0:
        return None
    fetch_head = Path(root) / loc.stdout.strip()
    try:
        mtime = os.stat(fetch_head).st_mtime
    except OSError:
        return None
    # Clamp: a clock skew or a future-dated mtime must not read as "fetched
    # ages ago" AND must not read as negative. Unknown-ness has one spelling
    # here and it is None.
    return max(0.0, (time.time() - mtime) / 3600.0)


def sweep_advisory(repos, patterns, root=None,
                   uncommitted_rows: int = 0,
                   masked_rows: int = 0) -> list[str]:
    """The one-line wiring for a sweep with no per-row file address.

    `repos`    — what this run actually MEASURED, as paths or as bare names.
                 Deliberately not re-derived here: an advisory about a repo the
                 sweep never read is noise, and the family's repo variable is
                 sometimes a name set and sometimes a path list.
    `patterns` — the globs naming this sweep's own population, so a sweep over
                 `*.lean` stays silent while somebody is editing Rust.
    `root`     — the workspace, required only when `repos` are names.

    `uncommitted_rows` / `masked_rows` — the row-level totals from
                 `head_delta`, forwarded so a sweep that has computed the
                 three-way split reports it on the SAME final line as the
                 repo-level banner. A sweep that has not computed it passes
                 nothing and the lines simply do not appear: the repo-level
                 advisory alone was Issue 822's finding, not its repair.

    A name this run cannot resolve to a directory is SKIPPED rather than
    guessed at: `population_verdict()` already owns the "a repo is missing"
    verdict, and a second instrument answering that question differently is
    how two gates come to disagree about one quantity.
    """
    scope: dict[str, int] = {}
    for r in repos:
        path = _resolve(r, root)
        if path is None:
            continue
        scope[repo_alias.display(path.name)] = dirty_in_scope(path, patterns)
    stale, unverified = upstream_axis(repos, patterns, root)
    return worktree_advisory(scope, uncommitted_rows, masked_rows,
                             stale=stale, unverified=unverified)


def _resolve(r, root) -> Path | None:
    """A repo handle -> the directory to OPEN, or `None` if unresolvable.

    A name this run cannot resolve is SKIPPED rather than guessed at:
    `population_verdict()` already owns the "a repo is missing" verdict, and a
    second instrument answering that question differently is how two gates
    come to disagree about one quantity.
    """
    s = str(r)
    if isinstance(r, Path) or '/' in s or chr(92) in s:
        path = Path(r)
    elif root is not None:
        # Issue 842: a bare CONTRACT name may be alias-mapped to a different
        # on-disk directory. Resolving through the codec is the identity on
        # unaliased boxes; without it the aliased repos are skipped in
        # SILENCE — the exact gap between "visited but unreadable" and "never
        # visited at all".
        path = Path(root) / repo_alias.disk(s)
    else:
        path = None
    return path if path is not None and path.is_dir() else None


def upstream_axis(repos, patterns, root=None
                  ) -> tuple[dict[str, tuple[int, int]],
                             dict[str, tuple[float | None, int]]]:
    """`(stale, unverified)` — the UPSTREAM half of `sweep_advisory`, alone.

    ⛔ Factored out because a sweep that computes its OWN dirty scope had no
    way to get this, and one does: `citation_drift_sweep` calls the low-level
    `worktree_advisory()` directly (its scope is `dirty_files` intersected
    with its own document set, sharper than any glob), so it had the WORKTREE
    axis and **no upstream axis at all** — no STALE, no UNVERIFIED — while
    being the only sweep in the family whose findings carry a `file:line`
    address and name another repo.

    Measured the day this was factored out: it reported a CROSS finding at
    riir-neuron-db `HISTORY.md:49` with no advisory of any kind, and
    `origin/develop` already carried the repair (the worktree reads `sdk Plan
    006` where upstream reads `riir-game-sdk Plan 006`), that checkout being
    4 commits behind with one of them touching the very file. Issue 798's
    founding class — *a committed FIX read dirty* — in the sweep that most
    needs the warning, and it cost a real investigation before a manual
    `git show origin/develop` settled it.

    ⚠ `sweep_advisory_membership_gate` credits that sweep and is RIGHT to:
    its predicate accepts either entry point, because an earlier, narrower
    version condemned this same sweep as UNWIRED when it was the most
    carefully wired member. The gate asserts the CALL; only this factoring
    makes the two entry points mean the same thing.
    """
    stale: dict[str, tuple[int, int]] = {}
    unverified: dict[str, tuple[float | None, int]] = {}
    for r in repos:
        path = _resolve(r, root)
        if path is None:
            continue
        # The printed key is the CONTRACT spelling (identity on unaliased
        # boxes): the alias content must never reach stdout — run logs get
        # pasted into tracked docs (repo_alias's own disclosure rule).
        label = repo_alias.display(path.name)
        beh = behind_origin(path, patterns)
        if beh is not None and beh[1]:
            stale[label] = beh
        elif beh is not None:
            # ⛔ BOTH silent buckets are challenged, not just `(0, 0)`.
            # `behind_origin` has two readings that print nothing — `(0, 0)`
            # "up to date" and `(n, 0)` "behind, but on nothing in this
            # sweep's population" — and they rest on the identical thing: the
            # remote-tracking ref, i.e. what this box last HEARD. `(n, 0)` is
            # the stronger claim of the two, because it asserts something
            # about the CONTENT of the commits it is behind by, and that
            # content is read from the same unrefreshed ref.
            #
            # A repo already reported STALE needs no second line, and one with
            # no upstream answers `None` and claims nothing to begin with.
            age = fetch_age_hours(path)
            if age is None or age > STALE_FETCH_HOURS:
                # ⛔ The BUCKET rides along, and it is not decoration. The
                # trigger was widened to both silent readings (Issue 850 T1)
                # and the SENTENCE was left saying "report 'up to date'" —
                # which is FALSE for `(n, 0)`, the reading that says "behind,
                # and none of it touches your population". A reader told a
                # repo reads 'up to date' has been given the one fact the
                # sweep knows to be wrong. Pooling two readings the module
                # documents as "never pooled" in the DISPLAY is the same
                # defect one layer down.
                unverified[label] = (age, beh[0])
    return stale, unverified


GLYPHS = frozenset("⚠⛔✓✗▸")


def deferral_line(text: str) -> str:
    """Render one advisory/deferral line for a sweep's FAILURE path.

    ⛔ Prefix a glyph ONLY when the text does not already carry one. The
    family's failure path was copy-pasted as `print(f"  ⚠ {_d}")` into eleven
    sweeps, and the `deferred` list it prints holds lines from TWO sources:
    `population_verdict`'s deferrals, which are bare text, and
    `worktree_advisory`'s, which already begin with `⚠` or `⛔`. The result
    was a doubled `⚠ ⚠ STALE: …` on every failing run — cosmetic, except
    that verdict lines in this workspace are PARSED INTERFACES (sweeps regex
    each other's output), so a second glyph is a prefix a consumer did not
    expect.

    ⚠ It must not normalise the glyph itself: `⛔ MASKED` and `⚠ STALE` are
    different severities and flattening them would lose the distinction the
    advisory exists to draw.
    """
    # ⛔ A SET, not `in "⚠⛔✓✗▸"`. Substring membership on a str is True
    # for the EMPTY string, so `text[:1] in "…"` silently passes an empty
    # line through unprefixed — caught by this function's own arm on its
    # first run, which is the argument for arming a four-line helper.
    return text if text[:1] in GLYPHS else f"⚠ {text}"


def worktree_advisory(scope_counts: dict[str, int],
                      uncommitted_rows: int = 0,
                      masked_rows: int = 0,
                      stale: dict[str, tuple[int, int]] | None = None,
                      unverified: dict[str, tuple[float | None, int]] | None = None
                      ) -> list[str]:
    """The lines that ride a sweep's FINAL line. Empty when nothing is dirty.

    `scope_counts` — {repo name: count of dirty files INSIDE this sweep's own
    walked population}. Zero-valued entries are filtered here rather than
    demanded of every caller; that filter is what keeps the advisory silent on
    an ordinary clean run, so it is asserted by its own arm.
    """
    inscope = {k: v for k, v in scope_counts.items() if v}
    # A repo whose upstream moved but not within THIS sweep's population is not
    # stale for this sweep's purposes, and saying so on every run is the banner
    # nobody reads. Filtered here, like the zero counts above, so no caller has
    # to remember it.
    instale = {k: v for k, v in (stale or {}).items() if v[1]}
    unver = dict(unverified or {})
    if (not inscope and not uncommitted_rows and not masked_rows
            and not instale and not unver):
        return []
    out: list[str] = []
    if inscope:
        detail = ", ".join(f"{k} ({v})" for k, v in sorted(inscope.items()))
        out.append(
            f"⚠ WORKTREE: {sum(inscope.values())} file(s) in this sweep's own "
            f"population differ from HEAD — {detail}. Counts and floors from "
            "this run describe a state NO commit contains; do not re-pin from "
            "it (Issue 797)")
    if instale:
        detail = ", ".join(f"{k} ({v[0]} behind, {v[1]} in scope)"
                           for k, v in sorted(instale.items()))
        out.append(
            f"⚠ STALE: {len(instale)} repo(s) sit BEHIND their upstream on "
            f"commits that touch this sweep's own population — {detail}. A "
            "finding there may already be FIXED upstream; this box's checkout "
            "is what the sweep read, so confirm against origin before "
            "repairing (Issue 798)")
    if unver:
        # ⛔ The remedy names an INSTRUMENT, not `git fetch <repo>`, and that
        # is the Issue-842 codec showing up in the one place it cannot be
        # papered over. These labels are CONTRACT spellings by rule — alias
        # content must never reach stdout, because run logs get pasted into
        # tracked docs — and on an aliased box the contract spelling is a
        # directory that DOES NOT EXIST: measured here, the old text told the
        # reader to fetch `mmorpg-editor` while the checkout is
        # `seal-game-editor`. Both rules are right and the remedy was still
        # unusable, so it points at the one command that resolves the codec
        # itself (Issue 850 T2).
        def _row(k, v):
            age, behind = v
            when = "never fetched" if age is None else f"{age:.0f}h"
            # The two silent readings, NAMED. `(0, 0)` claims not-behind;
            # `(n, 0)` claims behind-but-out-of-scope, which is the stronger
            # claim because it asserts something about the CONTENT of the
            # commits — read from the same unrefreshed ref (Issue 850).
            reads = ("reads 'up to date'" if not behind
                     else f"reads '{behind} behind, none in this population'")
            return f"{k} ({when}, {reads})"
        detail = ", ".join(_row(k, v) for k, v in sorted(unver.items()))
        out.append(
            f"⚠ UNVERIFIED UPSTREAM: {len(unver)} repo(s) rest a SILENT "
            f"verdict on a remote-tracking ref last refreshed over "
            f"{STALE_FETCH_HOURS:.0f}h ago — {detail}. Neither reading is a "
            "measurement of the remote; both are what this box last HEARD, so "
            "neither can support a verdict: a finding there may already be "
            "fixed upstream, and a CLEAN row there may be a false green. "
            "Run `scripts/fetch_contract_repos.py` (~20-50s for the whole "
            "workspace) before trusting it (Issue 827 T5, 850 T2/T3)")
    if uncommitted_rows:
        out.append(
            f"⚠ UNCOMMITTED: {uncommitted_rows} row(s) sit on a file that "
            "differs from HEAD — they may be another session's in-flight edit. "
            "Shown above (that is what the file says today) but NOT adjudicated "
            "against the pins, which read HEAD")
    if masked_rows:
        out.append(
            f"⛔ MASKED: {masked_rows} row(s) exist in HEAD and not in the "
            "worktree — a COMMITTED defect this run would otherwise report "
            "clean")
    return out


# ───────────────────────────── self-test ──────────────────────────────────

def _run(cwd, *args):
    """FIXTURE BUILDER, not decision logic.

    `arm_reach_audit --include-all` reports the three `True` kwargs here and in
    `_repo` as live survivors, and they are the correct place to stop: they
    build the git trees the arms then assert over, so flipping one makes the
    fixture wrong rather than making a rule wrong, and every arm downstream
    reds on the malformed tree instead. Recorded here rather than remembered,
    because "writing the reason is the adjudication".
    """
    subprocess.run(["git", "-C", str(cwd), *args], check=True,
                   capture_output=True)


def _repo(tmp: Path, name: str) -> Path:
    r = tmp / name
    r.mkdir(parents=True)
    _run(r, "init", "-q", "-b", "main")
    _run(r, "config", "user.email", "t@t")
    _run(r, "config", "user.name", "t")
    return r


def dirty_arms() -> list[str]:
    """`dirty_files` / `head_text`, two-sided against a real git tree."""
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        r = _repo(tmp, "r")
        (r / "a.md").write_text("one\n", encoding="utf-8")
        (r / "b.md").write_text("two\n", encoding="utf-8")
        sub = r / "sub"
        sub.mkdir()
        (sub / "c.md").write_text("three\n", encoding="utf-8")
        _run(r, "add", "-A")
        _run(r, "commit", "-qm", "init")

        check(dirty_files(r) == frozenset(),
              f"a freshly committed tree was not clean: {dirty_files(r)}")

        # unstaged modification
        (r / "a.md").write_text("one CHANGED\n", encoding="utf-8")
        check(dirty_files(r) == {"a.md"},
              f"an unstaged modification was not seen: {dirty_files(r)}")
        # the OTHER side: b.md must NOT be reported
        check("b.md" not in dirty_files(r),
              "an untouched file was reported dirty")

        # staged modification — the arm that reds if `--untracked-files=no`
        # were mistaken for "worktree only"
        _run(r, "add", "a.md")
        check(dirty_files(r) == {"a.md"},
              f"a STAGED modification was not seen: {dirty_files(r)}")

        # A nested path arrives POSIX-separated. This asserts git's OUTPUT
        # SHAPE, not the `.replace` in `dirty_files` — deleting that line reds
        # nothing here, because git already emits `/` (measured on Windows).
        # Written as the premise it is: if a future git emitted `\` on this
        # platform, the defensive line would silently become load-bearing and
        # this arm is where that shows up.
        (sub / "c.md").write_text("three CHANGED\n", encoding="utf-8")
        raw = _git(r, "status", "--porcelain", "--untracked-files=no").stdout
        check("sub/c.md" in raw,
              f"PREMISE: git --porcelain did not emit a POSIX separator on this "
              f"platform — the defensive replace in dirty_files is now "
              f"load-bearing and untested: {raw!r}")
        check("sub/c.md" in dirty_files(r),
              f"a nested path was not reported: {dirty_files(r)}")

        # untracked is NOT dirty — the population is what git tracks
        (r / "new.md").write_text("x\n", encoding="utf-8")
        check("new.md" not in dirty_files(r),
              "an untracked file was counted as dirty")

        # The SHORTEST porcelain line there is: two status characters, a
        # space, and a one-character name — exactly 4. The `len(line) < 4`
        # guard exists to drop git's blank tail, and off by one it silently
        # drops a real file. Its OWN repo, because `r` is carrying a staged
        # edit by this point and any commit here would sweep that in.
        r3 = _repo(tmp, "shortname")
        (r3 / "a").write_text("one\n", encoding="utf-8")
        _run(r3, "add", "-A")
        _run(r3, "commit", "-qm", "short")
        (r3 / "a").write_text("two\n", encoding="utf-8")
        check(dirty_files(r3) == {"a"},
              f"a ONE-character filename was dropped — the porcelain line is "
              f"4 characters and the length guard is off by one: "
              f"{dirty_files(r3)}")

        # head_text reads the COMMITTED bytes, not the worktree's
        check(head_text(r, "a.md") == "one\n",
              f"head_text returned the worktree text: {head_text(r, 'a.md')!r}")
        check(head_text(r, "new.md") is None,
              "head_text invented a blob for a path not in HEAD")

        # ── the `.git` probe: a non-repo directory INSIDE a repo ──────────
        # Without the probe `git -C` walks up and answers with r's dirty set.
        plain = r / "plain"
        plain.mkdir()
        check(dirty_files(plain) == frozenset(),
              f"a non-repo dir inside a repo inherited its PARENT's dirty set: "
              f"{dirty_files(plain)}")
        check(head_text(plain, "a.md") is None,
              "head_text answered for a non-repo directory")

        # a tree with no .git at all is clean by construction, not an error
        bare = tmp / "extracted"
        bare.mkdir()
        (bare / "a.md").write_text("x\n", encoding="utf-8")
        check(dirty_files(bare) == frozenset(),
              "an extracted tree was not treated as clean")

        # ── rename reports the DESTINATION ────────────────────────────────
        r2 = _repo(tmp, "r2")
        (r2 / "old.md").write_text("k\n", encoding="utf-8")
        _run(r2, "add", "-A")
        _run(r2, "commit", "-qm", "init")
        _run(r2, "mv", "old.md", "newname.md")
        d = dirty_files(r2)
        check("newname.md" in d,
              f"a rename did not report its destination: {d}")

    return fails


def split_arms() -> list[str]:
    """`split_rows` withholds exactly the rows whose file is dirty."""
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    rows = [("a.md", 1), ("b.md", 2), ("sub/c.md", 3), (None, 4)]
    keep, held = split_rows(rows, lambda r: r[0], frozenset({"a.md", "sub/c.md"}))
    check([r[1] for r in held] == [1, 3], f"wrong rows withheld: {held}")
    check([r[1] for r in keep] == [2, 4], f"wrong rows kept: {keep}")

    # An address-less row is COMMITTED — the conservative direction for a
    # bucket that WITHHOLDS from the count.
    check((None, 4) in keep, "an address-less row was withheld from the count")

    # Backslashes normalise on BOTH sides of the comparison.
    keep, held = split_rows([("sub\\c.md", 9)], lambda r: r[0],
                            frozenset({"sub/c.md"}))
    check(len(held) == 1, f"a Windows-separator row did not match: {keep} {held}")

    # Nothing dirty -> nothing withheld.
    keep, held = split_rows(rows, lambda r: r[0], frozenset())
    check(held == [] and len(keep) == 4, f"a clean tree withheld rows: {held}")
    return fails


def delta_arms() -> list[str]:
    """`head_delta`, two-sided, against a REAL git tree.

    Every arm has both directions where both exist: UNCOMMITTED alone passes on
    an implementation that ignores HEAD, MASKED alone passes on one that
    ignores the worktree, and neither catches the row HEAD *also* carries —
    which is the bucket `split_rows` gets wrong and the reason this exists.
    """
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    # A row is (path, token); a file's rows are its `FIND:<token>` lines. The
    # key is LINE-FREE, exactly as the docstring demands, so padding a file
    # above a finding must not reclassify it.
    def rescan(rel, src):
        if src is None:
            return []
        return [(rel, ln.split("FIND:", 1)[1].strip())
                for ln in src.splitlines() if "FIND:" in ln]

    def rows_of(repo: Path, rel: str):
        return rescan(rel, (repo / rel).read_text(encoding="utf-8"))

    key = lambda r: (r[0], r[1])          # noqa: E731 — line-free by design
    path = lambda r: r[0]                 # noqa: E731

    PY = ("*.py",)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        me = _repo(tmp, "r")
        (me / "scripts").mkdir()
        (me / "src").mkdir()
        (me / "scripts/a.py").write_text("FIND:alpha\n", encoding="utf-8")
        (me / "scripts/b.py").write_text("FIND:beta\n", encoding="utf-8")
        (me / "src/c.rs").write_text("FIND:gamma\n", encoding="utf-8")
        _run(me, "add", "-A")
        _run(me, "commit", "-qm", "base")

        # ── a CLEAN tree costs NOTHING and withholds nothing ────────────────
        calls: list[str] = []

        def counting(rel, src, _c=calls):
            _c.append(rel)
            return rescan(rel, src)

        wt = rows_of(me, "scripts/a.py") + rows_of(me, "scripts/b.py")
        com, unc, mas = head_delta(me, PY, wt, path, key, counting)
        check(calls == [], f"a clean tree re-read HEAD: {calls}")
        check(len(com) == 2 and not unc and not mas,
              f"a clean tree did not report every row COMMITTED: "
              f"{len(com)}/{len(unc)}/{len(mas)}")

        # ── UNCOMMITTED: the worktree INVENTS a row HEAD does not have ──────
        (me / "scripts/a.py").write_text(
            "padding\npadding\nFIND:alpha\nFIND:delta\n", encoding="utf-8")
        wt = rows_of(me, "scripts/a.py") + rows_of(me, "scripts/b.py")
        com, unc, mas = head_delta(me, PY, wt, path, key, rescan)
        check([r[1] for r in unc] == ["delta"],
              f"UNCOMMITTED direction: {unc}")
        check(not mas, f"an added row was also reported MASKED: {mas}")
        # The row HEAD carries TOO stays COMMITTED even though its file is
        # dirty. `split_rows` withholds it, which is the understatement this
        # helper exists to remove — so the arm asserts the DIFFERENCE.
        check(sorted(r[1] for r in com) == ["alpha", "beta"],
              f"a row HEAD also carries was withheld from the count: {com}")
        _, held = split_rows(wt, path, dirty_files(me))
        check(len(held) == 2 and len(unc) == 1,
              f"split_rows and head_delta agree, so one of them is wrong: "
              f"held={len(held)} uncommitted={len(unc)}")

        # ── MASKED: the worktree HIDES a committed row ──────────────────────
        (me / "scripts/a.py").write_text("FIND:delta\n", encoding="utf-8")
        wt = rows_of(me, "scripts/a.py") + rows_of(me, "scripts/b.py")
        com, unc, mas = head_delta(me, PY, wt, path, key, rescan)
        check([r[1] for r in mas] == ["alpha"], f"MASKED direction: {mas}")
        # ⛔ `.head` is the PINS' population and it is NOT `committed`: a
        # masked row is committed-and-hidden, so omitting it understates a
        # ceiling by exactly the silent direction. Asserted against the same
        # classifier run over HEAD's bytes, not against a restatement.
        d = head_delta(me, PY, wt, path, key, rescan)
        check(sorted(key(r) for r in d.head)
              == sorted(key(r) for r in rescan("scripts/a.py",
                                               head_text(me, "scripts/a.py"))
                        + rescan("scripts/b.py",
                                 head_text(me, "scripts/b.py"))),
              f"the .head view is not what HEAD's bytes classify to: {d.head}")
        check(len(d.head) == len(d.committed) + len(d.masked)
              and len(d.head) != len(d.committed),
              f"the MASKED row is missing from the pins' view: {d}")
        check([r[1] for r in unc] == ["delta"],
              f"the same file's two directions were pooled: {unc} {mas}")
        # ⛔ and they are NOT the same row wearing two labels.
        check(not ({key(r) for r in unc} & {key(r) for r in mas}),
              "one row was reported as both UNCOMMITTED and MASKED")

        # ── an out-of-population dirty file is never even READ ──────────────
        (me / "src/c.rs").write_text("FIND:gamma\nFIND:epsilon\n",
                                     encoding="utf-8")
        calls.clear()
        com, unc, mas = head_delta(me, PY, wt, path, key, counting)
        check("src/c.rs" not in calls,
              f"a *.rs edit was re-read by a *.py sweep: {calls}")
        check(sorted(calls) == ["scripts/a.py"],
              f"the rescan set is not |dirty ∩ population|: {calls}")

        # ── a STAGED-but-never-committed file: HEAD has no blob ─────────────
        # The measured case — Issue 822's breach included a row on a file
        # `git log` cannot see at all. `head_text` answers None and the row is
        # UNCOMMITTED, never MASKED and never counted.
        (me / "scripts/new.py").write_text("FIND:zeta\n", encoding="utf-8")
        _run(me, "add", "scripts/new.py")
        wt = (rows_of(me, "scripts/a.py") + rows_of(me, "scripts/b.py")
              + rows_of(me, "scripts/new.py"))
        com, unc, mas = head_delta(me, PY, wt, path, key, rescan)
        check(("scripts/new.py", "zeta") in {key(r) for r in unc},
              f"a staged-only file's row was not UNCOMMITTED: {unc}")
        check(all(r[0] != "scripts/new.py" for r in mas),
              f"a file absent from HEAD produced a MASKED row: {mas}")

        # ── a file DELETED in the worktree still yields its HEAD rows ───────
        (me / "scripts/b.py").unlink()
        wt = rows_of(me, "scripts/a.py") + rows_of(me, "scripts/new.py")
        com, unc, mas = head_delta(me, PY, wt, path, key, rescan)
        check(("scripts/b.py", "beta") in {key(r) for r in mas},
              f"a deleted file's committed row was not MASKED: {mas}")

        # ── the CALLER's separators normalise ───────────────────────────────
        com, unc, mas = head_delta(
            me, PY, [("scripts" + chr(92) + "a.py", "alpha")], path, key,
            rescan)
        check(len(unc) == 1,
              "a Windows-separator row address did not match the dirty set")

        # ── a non-injective key counts a MASKED row ONCE ────────────────────
        # MASKED is walled at 0 by its callers, so a duplicate is a red nobody
        # can repair.
        (me / "scripts/dup.py").write_text("FIND:eta\nFIND:eta\n",
                                           encoding="utf-8")
        _run(me, "add", "-A")
        _run(me, "commit", "-qm", "dup")
        (me / "scripts/dup.py").write_text("clean\n", encoding="utf-8")
        com, unc, mas = head_delta(me, PY, rows_of(me, "scripts/a.py"),
                                   path, key, rescan)
        check(sum(1 for r in mas if r[1] == "eta") == 1,
              f"a duplicated HEAD key was counted twice as MASKED: {mas}")

        # ── the MASKED comparison is SCOPED to the dirty files ──────────────
        # ⛔ Widening `live` to every worktree row reds nothing above, because
        # that fixture's key carries its path and no two files can then collide.
        # It is not equivalent in general, and this is the case that separates
        # them: a key that does NOT carry its address (a bare script NAME — the
        # shape `console_encoding`'s rows have) lets a CLEAN file's identical
        # row suppress a genuinely masked one. Scoping is the correct direction
        # and the arm that says so has to use such a key.
        two = _repo(tmp, "r2")
        (two / "scripts").mkdir()
        (two / "scripts/x.py").write_text("FIND:alpha\n", encoding="utf-8")
        (two / "scripts/y.py").write_text("FIND:alpha\n", encoding="utf-8")
        _run(two, "add", "-A")
        _run(two, "commit", "-qm", "base")
        (two / "scripts/x.py").write_text("clean\n", encoding="utf-8")
        bare = lambda r: r[1]             # noqa: E731 — deliberately address-less
        com, unc, mas = head_delta(two, PY, rows_of(two, "scripts/y.py"),
                                   path, bare, rescan)
        check([r[0] for r in mas] == ["scripts/x.py"],
              f"a clean file's identical row suppressed a MASKED one: {mas}")

    return fails


def overlay_arms() -> list[str]:
    """`head_overlay` / `delta_of`, two-sided against a real git tree."""
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        me = _repo(tmp, "o")
        (me / "a.md").write_text("committed\n", encoding="utf-8")
        (me / "b.md").write_text("untouched\n", encoding="utf-8")
        (me / "c.rs").write_text("other population\n", encoding="utf-8")
        _run(me, "add", "-A")
        _run(me, "commit", "-qm", "base")

        # A clean tree overlays NOTHING, and the EMPTY dict is the signal to
        # skip the second classification entirely — not "overlay nothing".
        check(head_overlay(me, ("*.md",)) == {},
              "a clean tree produced an overlay")

        (me / "a.md").write_text("edited\n", encoding="utf-8")
        (me / "c.rs").write_text("edited too\n", encoding="utf-8")
        ov = head_overlay(me, ("*.md",))
        check(list(ov) == ["a.md"],
              f"the overlay is not |dirty n population|: {list(ov)}")
        check(ov["a.md"] == "committed\n",
              f"the overlay carries the WORKTREE bytes: {ov['a.md']!r}")

        # ⛔ None is a VALUE, not an absence: tracked (staged) but not in HEAD.
        # `in` and `.get()` say different things, and a caller that cannot tell
        # them apart reports another session's in-flight file as committed.
        (me / "new.md").write_text("staged only\n", encoding="utf-8")
        _run(me, "add", "new.md")
        ov = head_overlay(me, ("*.md",))
        check("new.md" in ov and ov["new.md"] is None,
              f"a staged-only file is not distinguishable from an absent "
              f"one: {ov}")

    # delta_of compares two COMPLETE row sets — unlike head_delta, whose HEAD
    # side comes only from the dirty files.
    wt = [("x", 1), ("y", 2)]
    hd = [("x", 1), ("z", 3)]
    d = delta_of(wt, hd, lambda r: r[0])
    check([r[0] for r in d.committed] == ["x"], f"committed: {d.committed}")
    check([r[0] for r in d.uncommitted] == ["y"], f"uncommitted: {d.uncommitted}")
    check([r[0] for r in d.masked] == ["z"], f"masked: {d.masked}")
    check(sorted(r[0] for r in d.head) == ["x", "z"],
          f"the .head view is not HEAD's own set: {d.head}")
    # Identical sets produce no split in EITHER direction.
    d = delta_of(wt, list(wt), lambda r: r[0])
    check(not d.uncommitted and not d.masked,
          f"identical row sets produced a split: {d}")
    return fails


def tree_arms() -> list[str]:
    """`head_tree` — Issue 822 T5g's materialised HEAD.

    Every assertion here is about the thing the two cheaper instruments
    cannot give a classifier: a tree that answers `git grep` and `git
    ls-files`. Testing it any other way would test a temp directory.
    """
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        me = _repo(tmp, "t")
        (me / "keep.rs").write_text("committed\n", encoding="utf-8")
        (me / "sub").mkdir()
        (me / "sub" / "deep.rs").write_text("deep committed\n",
                                            encoding="utf-8")
        # ⛔ A .gitignore that HEAD tracks would exclude tracked content from
        # the rebuilt index unless `git add` is forced. The arm plants the
        # exact shape rather than trusting the flag.
        (me / ".gitignore").write_text("sub/\n", encoding="utf-8")
        _run(me, "add", "-A", "-f")
        _run(me, "commit", "-qm", "base")

        # A clean tree yields None — the caller must SKIP, because this
        # instrument copies a tree and cannot be paid for on every run.
        with head_tree(me, ("*.rs",)) as t:
            check(t is None, f"a clean tree materialised HEAD anyway: {t}")

        # Dirt OUTSIDE the population is also None: the population is the
        # caller's declared scope, not "anything changed".
        (me / "notes.md").write_text("edited\n", encoding="utf-8")
        with head_tree(me, ("*.rs",)) as t:
            check(t is None, "dirt outside the population materialised HEAD")

        (me / "keep.rs").write_text("EDITED\n", encoding="utf-8")
        (me / "untracked.rs").write_text("never added\n", encoding="utf-8")
        with head_tree(me, ("*.rs",)) as t:
            check(t is not None, "a dirty population did not materialise HEAD")
            if t is not None:
                check((t / "keep.rs").read_text(encoding="utf-8")
                      == "committed\n",
                      "head_tree carried the WORKTREE bytes, not HEAD's")
                # The whole point: it answers git, or the classifiers this
                # exists for (`git grep`, `git ls-files`) see an empty repo
                # and report a confident zero.
                ls = subprocess.run(
                    ["git", "-C", str(t), "ls-files"], capture_output=True,
                    encoding="utf-8", errors="replace").stdout.split()
                check("keep.rs" in ls,
                      f"head_tree does not answer `git ls-files`: {ls}")
                check("sub/deep.rs" in ls,
                      f"a gitignored-but-TRACKED path is missing from the "
                      f"rebuilt index — `git add` was not forced: {ls}")
                grep = subprocess.run(
                    ["git", "-C", str(t), "grep", "-l", "committed"],
                    capture_output=True, encoding="utf-8",
                    errors="replace").stdout.split()
                check("keep.rs" in grep,
                      f"head_tree does not answer `git grep`: {grep}")
                check(not (t / "untracked.rs").exists(),
                      "an UNTRACKED file reached the HEAD tree, so it would "
                      "be classified as committed")
        # The temp tree is cleaned up on exit — a sweep runs this per repo.
        check(t is not None and not t.exists(),
              "head_tree leaked its temporary tree")

        # Not a repository: None, never a fabricated empty tree. An empty one
        # would report every finding in the repo as UNCOMMITTED.
        plain = tmp / "plain"
        plain.mkdir()
        (plain / "a.rs").write_text("x\n", encoding="utf-8")
        with head_tree(plain, ("*.rs",)) as t:
            check(t is None, "a non-repository produced a HEAD tree")

        # `paths` — the narrowing, in BOTH directions. It is a 20x cost
        # difference on a large repo and a silent wrong-tree risk if a
        # classifier reads something the pathspec drops, so the arm asserts
        # what is THERE and what is GONE rather than only that it still works.
        me2 = _repo(tmp, "t2")
        (me2 / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        (me2 / "src").mkdir()
        (me2 / "src" / "lib.rs").write_text("// rs\n", encoding="utf-8")
        (me2 / "assets").mkdir()
        (me2 / "assets" / "big.bin").write_text("payload\n", encoding="utf-8")
        _run(me2, "add", "-A")
        _run(me2, "commit", "-qm", "base")
        (me2 / "src" / "lib.rs").write_text("// EDITED\n", encoding="utf-8")
        with head_tree(me2, ("*.rs",), paths=("*.rs", "*.toml")) as t:
            check(t is not None, "a narrowed head_tree yielded nothing")
            if t is not None:
                check((t / "src" / "lib.rs").is_file(),
                      "the narrowed tree dropped a path the pathspec NAMES")
                check((t / "Cargo.toml").is_file(),
                      "the narrowed tree dropped the manifest")
                check(not (t / "assets" / "big.bin").exists(),
                      "the pathspec carried a file it does not name — the "
                      "narrowing is inert and the cost measurement is a lie")
                ls = subprocess.run(
                    ["git", "-C", str(t), "ls-files"], capture_output=True,
                    encoding="utf-8", errors="replace").stdout.split()
                check("src/lib.rs" in ls and "assets/big.bin" not in ls,
                      f"the narrowed tree's INDEX does not match its content "
                      f"— a classifier asking git would disagree with one "
                      f"reading the filesystem: {ls}")
        # ...and the DEFAULT carries everything, or every existing caller
        # silently narrowed the day this parameter landed.
        with head_tree(me2, ("*.rs",)) as t:
            check(t is not None and (t / "assets" / "big.bin").is_file(),
                  "the DEFAULT head_tree narrowed — `paths=None` must be the "
                  "whole tree, which is the only always-correct answer")

        # The materialised tree is named after its SOURCE repo, because a
        # classifier can key rows on the directory name — and two repos
        # materialised at once would otherwise both be `head`.
        with head_tree(me2, ("*.rs",), paths=("*.rs",)) as t:
            check(t is not None and t.name == "t2",
                  f"head_tree did not name its checkout after the source repo "
                  f"({t.name if t is not None else None}) — a name-keyed "
                  f"classifier then misattributes every row it produces")

        # `extra_dirty` — the UNTRACKED trigger, for a sweep whose walk is
        # `os.walk` rather than `git ls-files`. Committed and clean, so
        # `git status` reports NOTHING; only the widened trigger can fire.
        me3 = _repo(tmp, "t3")
        (me3 / "a.lean").write_text("theorem x : True := trivial\n",
                                    encoding="utf-8")
        _run(me3, "add", "-A")
        _run(me3, "commit", "-qm", "base")
        (me3 / "wip.lean").write_text("-- never added\n", encoding="utf-8")
        with head_tree(me3, ("*.lean",)) as t:
            check(t is None,
                  "an UNTRACKED file alone fired the ordinary trigger — "
                  "`dirty_files` must keep excluding it (Issue 777)")
        with head_tree(me3, ("*.lean",), extra_dirty=("wip.lean",)) as t:
            check(t is not None,
                  "extra_dirty did not widen the trigger, so a sweep whose "
                  "walk includes untracked files files them as COMMITTED")
            if t is not None:
                check(not (t / "wip.lean").exists(),
                      "the untracked file reached the HEAD tree — it is in no "
                      "commit, so its finding must read UNCOMMITTED")
                check((t / "a.lean").is_file(),
                      "the widened trigger produced a tree missing HEAD's own "
                      "content")
    return fails


def key_arms() -> list[str]:
    """`line_free` / `ordinal_keys` — Issue 822 T5d's shared row identity."""
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    # The family's row shape, and the reason the key drops the number: an edit
    # ABOVE a finding moves it, and the finding is the same finding.
    check(line_free("12: subprocess.run(cmd, text=True)")
          == "subprocess.run(cmd, text=True)",
          "line_free did not strip the `<lineno>: ` prefix")
    check(line_free("900: subprocess.run(cmd, text=True)")
          == line_free("12: subprocess.run(cmd, text=True)"),
          "line_free is not line-INVARIANT, which is its only job")
    # ⛔ Only a BARE INTEGER is a line number. A row whose own text begins
    # `note: …` or `E501: …` must come back whole — silently beheading it
    # would merge two distinct findings into one identity.
    check(line_free("note: something happened") == "note: something happened",
          "line_free beheaded a row whose prefix is not a line number")
    check(line_free("no separator here") == "no separator here",
          "line_free mangled a row with no `: ` at all")
    # The FIRST separator only: a call text containing `": "` keeps it.
    check(line_free('7: run(x, env={"A": "b"})') == 'run(x, env={"A": "b"})',
          "line_free split on the wrong separator")

    rows = ["a", "a", "b"]
    keys = [k for k, _ in ordinal_keys(rows, lambda r: (r,))]
    check(keys == [("a", 0), ("a", 1), ("b", 0)],
          f"ordinal_keys did not disambiguate a repeated address: {keys}")
    check(len(set(keys)) == len(keys),
          "ordinal_keys produced a colliding key set")
    # ⛔ The counter is LOCAL. Two independent calls — the worktree pass and
    # the HEAD pass — must start from the same base, or every row looks moved
    # and a clean repo reports its whole finding set twice over.
    check([k for k, _ in ordinal_keys(rows, lambda r: (r,))] == keys,
          "ordinal_keys carried state between calls, so the worktree and HEAD "
          "passes count from different bases")
    # A new site at a DIFFERENT address renumbers nothing at the old one.
    keys2 = [k for k, _ in ordinal_keys(["a", "c", "a", "b"],
                                        lambda r: (r,))]
    check(("a", 0) in keys2 and ("a", 1) in keys2 and ("b", 0) in keys2,
          f"an unrelated new row renumbered its neighbours: {keys2}")
    check([r for _, r in ordinal_keys(rows, lambda r: (r,))] == rows,
          "ordinal_keys did not return its rows unchanged alongside the keys")
    return fails


def scope_arms() -> list[str]:
    """`dirty_in_scope` matches a NESTED path against a bare-suffix pattern.

    The basename fallback is the whole point: every sweep in the family states
    its population as `*.rs` / `*.py` / `*.lean`, and every real finding sits
    several directories down. Path-only matching would return 0 everywhere and
    make the advisory vacuous — a green zero, which is what this family exists
    to refuse. So the arm plants the finding DEEP.
    """
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        r = _repo(tmp, "scoped")
        deep = r / "crates" / "a" / "src"
        deep.mkdir(parents=True)
        (deep / "lib.rs").write_text("fn a() {}\n", encoding="utf-8")
        (r / "notes.md").write_text("x\n", encoding="utf-8")
        (r / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
        _run(r, "add", "-A")
        _run(r, "commit", "-qm", "init")

        check(dirty_in_scope(r, ("*.rs",)) == 0,
              "a clean repo reported files in scope")

        (deep / "lib.rs").write_text("fn a() { }\n", encoding="utf-8")
        check(dirty_in_scope(r, ("*.rs",)) == 1,
              "a NESTED .rs did not match the bare-suffix pattern — the "
              "basename fallback is missing and the advisory is vacuous")
        # The other side: a sweep over a different population stays silent.
        check(dirty_in_scope(r, ("*.lean",)) == 0,
              "a .lean sweep counted a dirty .rs — a sweep must not warn about "
              "a population it never reads")

        (r / "notes.md").write_text("y\n", encoding="utf-8")
        check(dirty_in_scope(r, ("*.rs", "*.md")) == 2,
              "two patterns did not union")
        # A literal root-level name (the citation sweep's own shape).
        check(dirty_in_scope(r, ("notes.md",)) == 1,
              "a literal filename pattern did not match")
        # A path-shaped pattern still works — the basename fallback WIDENS,
        # it must not replace the path match.
        (r / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
        check(dirty_in_scope(r, ("crates/*/src/*.rs",)) == 1,
              "a path-shaped pattern stopped matching")

        # A bare string must behave as a ONE-pattern tuple, never as a
        # character sequence whose `*` matches the whole repo.
        check(dirty_in_scope(r, "*.rs") == 1,
              "a bare-string pattern was iterated as characters — `*` then "
              "matches every dirty file and the advisory over-reports silently")
        check(dirty_in_scope(r, "*.lean") == 0,
              "a bare-string pattern matched a population it does not name")

        lines = sweep_advisory([r], ("*.rs",))
        check(len(lines) == 1 and "scoped (1)" in lines[0],
              f"sweep_advisory did not name the repo by DIRECTORY: {lines}")
        check(sweep_advisory([r], ("*.lean",)) == [],
              "sweep_advisory was not silent for an untouched population")

        # ── the resolution branch, all three ways in ──────────────────────
        # The family hands this function a NAME set as often as a path list
        # (`names`, `seen`, `present`, `contract`), so the name arm is the one
        # that most call sites actually exercise.
        by_name = sweep_advisory(["scoped"], ("*.rs",), root=tmp)
        check(by_name == lines,
              f"a bare NAME + root did not resolve to the same answer as the "
              f"path: {by_name} vs {lines}")
        # A name with no root cannot be resolved, and must be SKIPPED rather
        # than guessed at or crashed on — `population_verdict()` owns the
        # "a repo is missing" verdict and two instruments answering it
        # differently is how two gates come to disagree about one quantity.
        check(sweep_advisory(["scoped"], ("*.rs",)) == [],
              "a name with no root was not skipped")
        check(sweep_advisory(["no-such-repo"], ("*.rs",), root=tmp) == [],
              "a name that resolves to nothing was not skipped")
        # A path that does not exist takes the same exit, through the OTHER
        # branch of the same guard.
        check(sweep_advisory([tmp / "no-such-dir"], ("*.rs",)) == [],
              "a non-existent PATH was not skipped")

        # ── the alias branch (Issue 842) ───────────────────────────────────
        # A CONTRACT name may map to a different on-disk directory. Injected
        # codec state, not the machine's alias file — the arms are
        # box-independent. Two fixture repos make the arm two-sided: the
        # mapped one is dirtier than the identity one, so a run that opened
        # the WRONG directory reads "(1)" and fails the assertion instead of
        # passing against a lookalike.
        r2 = _repo(tmp, "scoped-disk")
        deep2 = r2 / "crates" / "b" / "src"
        deep2.mkdir(parents=True)
        (deep2 / "lib.rs").write_text("fn b() {}\n", encoding="utf-8")
        (deep2 / "other.rs").write_text("fn c() {}\n", encoding="utf-8")
        _run(r2, "add", "-A")
        _run(r2, "commit", "-qm", "init")
        (deep2 / "lib.rs").write_text("fn b2() {}\n", encoding="utf-8")
        (deep2 / "other.rs").write_text("fn c2() {}\n", encoding="utf-8")
        saved_loaded = repo_alias._loaded
        try:
            repo_alias._loaded = {"scoped-disk": "scoped"}
            got = sweep_advisory(["scoped"], ("*.rs",), root=tmp)
            check(len(got) == 1 and "scoped (2)" in got[0],
                  f"a mapped contract name did not resolve to its on-disk "
                  f"directory (expected the 2-dirty mapped repo): {got}")
            check(all("scoped-disk" not in ln for ln in got),
                  f"the alias CONTENT leaked into the advisory line — run logs "
                  f"get pasted into tracked docs: {got}")
        finally:
            repo_alias._loaded = saved_loaded
    return fails


def advisory_arms() -> list[str]:
    """The advisory is SILENT on a clean run and names each class when not."""
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    check(worktree_advisory({}) == [],
          "the advisory printed on a clean run — a banner on every run is one "
          "nobody reads")
    check(worktree_advisory({"riir-ai": 0}) == [],
          "a zero count was not filtered — the dict's emptiness is what keeps "
          "the advisory silent")

    lines = worktree_advisory({"riir-ai": 1})
    check(len(lines) == 1 and "riir-ai (1)" in lines[0],
          f"the advisory did not name the repo and count: {lines}")
    check("do not re-pin" in lines[0],
          f"the advisory did not say what it is FOR: {lines}")

    lines = worktree_advisory({}, uncommitted_rows=2)
    check(len(lines) == 1 and "UNCOMMITTED" in lines[0]
          and "NOT adjudicated" in lines[0] and "Shown above" in lines[0],
          f"the UNCOMMITTED class did not print alone, saying BOTH that the row "
          f"is displayed and that the pins ignore it: {lines}")

    lines = worktree_advisory({}, masked_rows=3)
    check(len(lines) == 1 and lines[0].startswith("⛔"),
          f"MASKED must be ⛔, not ⚠ — it is a COMMITTED defect: {lines}")

    check(worktree_advisory({}, stale={"r": (5, 0)}) == [],
          "a STALE row with 0 in-scope commits printed — an upstream that "
          "moved outside this sweep's population is the banner nobody reads")

    lines = worktree_advisory({}, stale={"r": (109, 1)})
    check(len(lines) == 1 and "STALE" in lines[0]
          and "109 behind, 1 in scope" in lines[0],
          f"STALE did not print BOTH counts: {lines}")
    check("already be FIXED upstream" in lines[0],
          f"STALE did not say which DIRECTION the error runs — it is a false "
          f"RED, the mirror of MASKED: {lines}")

    lines = worktree_advisory({"r": 1}, 2, 3, stale={"r": (9, 4)})
    check(len(lines) == 4, f"the four classes were pooled: {lines}")
    return fails


def stale_arms() -> list[str]:
    """`behind_origin`, against real git trees. Four verdicts, never pooled."""
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # No `.git` at all — a `git archive` extraction. Not an error, and NOT
        # a confident (0, 0): nothing is known about an upstream that has no
        # repository to be configured in.
        plain = tmp / "plain"
        plain.mkdir()
        check(behind_origin(plain, ("*.sh",)) is None,
              "a non-repo directory did not answer None")

        up = _repo(tmp, "up")
        (up / "a.sh").write_text("one\n", encoding="utf-8")
        (up / "keep.md").write_text("doc\n", encoding="utf-8")
        _run(up, "add", "-A")
        _run(up, "commit", "-qm", "base")

        dn = tmp / "dn"
        _run(tmp, "clone", "-q", str(up), str(dn))
        _run(dn, "config", "user.email", "t@t")
        _run(dn, "config", "user.name", "t")

        # Up to date is (0, 0) — distinct from None, and SILENT downstream.
        check(behind_origin(dn, ("*.sh",)) == (0, 0),
              "an up-to-date clone did not answer (0, 0)")

        # An upstream commit INSIDE the population.
        (up / "a.sh").write_text("two\n", encoding="utf-8")
        _run(up, "add", "-A")
        _run(up, "commit", "-qm", "fix the sh")
        _run(dn, "fetch", "-q", "origin")
        check(behind_origin(dn, ("*.sh",)) == (1, 1),
              f"an in-scope upstream commit was not counted: "
              f"{behind_origin(dn, ('*.sh',))}")

        # The SAME commit is out of scope for a sweep over another population.
        check(behind_origin(dn, ("*.lean",)) == (1, 0),
              "an out-of-scope upstream commit was counted in scope — a sweep "
              "over .lean must stay silent while somebody fixes shell")

        # A NESTED path against a bare-suffix pattern: the shared matcher's
        # whole reason for existing. A git pathspec would have been the obvious
        # implementation and answers differently for a bare `Dockerfile`.
        (up / "sub").mkdir()
        (up / "sub" / "Dockerfile").write_text("FROM x\n", encoding="utf-8")
        _run(up, "add", "-A")
        _run(up, "commit", "-qm", "nested dockerfile")
        _run(dn, "fetch", "-q", "origin")
        check(behind_origin(dn, ("Dockerfile",)) == (2, 1),
              f"a NESTED Dockerfile was not matched by the bare basename "
              f"pattern: {behind_origin(dn, ('Dockerfile',))}")

        # AHEAD is not BEHIND. A local commit touching the population must not
        # register: the three-dot diff is what makes that true, and a two-dot
        # one would report this file and read as stale.
        _run(dn, "reset", "-q", "--hard", "origin/main")
        (dn / "local.sh").write_text("mine\n", encoding="utf-8")
        _run(dn, "add", "-A")
        _run(dn, "commit", "-qm", "local only")
        check(behind_origin(dn, ("*.sh",)) == (0, 0),
              f"a checkout that is merely AHEAD read as stale: "
              f"{behind_origin(dn, ('*.sh',))}")

        # No upstream configured — never guessed at.
        solo = _repo(tmp, "solo")
        (solo / "a.sh").write_text("x\n", encoding="utf-8")
        _run(solo, "add", "-A")
        _run(solo, "commit", "-qm", "base")
        check(behind_origin(solo, ("*.sh",)) is None,
              "a branch with no upstream did not answer None — assuming "
              "origin/main invents a verdict for a repo that may not have one")

    return fails


def fetch_age_arms() -> list[str]:
    """`fetch_age_hours` + the UNVERIFIED bucket, against real git trees.

    Issue 827 T5. The arm that matters is the MIDDLE one: a repo that is
    genuinely `(0, 0)` against its local tracking ref, whose ref is old. Today
    that prints silence, and silence is the reassuring answer.
    """
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # No `.git` — the `git archive` shape. CANNOT TELL, never 0.
        plain = tmp / "plain"
        plain.mkdir()
        check(fetch_age_hours(plain) is None,
              "a non-repo directory did not answer None for fetch age")

        up = _repo(tmp, "up")
        (up / "a.sh").write_text("one\n", encoding="utf-8")
        _run(up, "add", "-A")
        _run(up, "commit", "-qm", "base")

        # A repo with an upstream but NO FETCH_HEAD: cloned, never fetched.
        # None, and the caller must treat it as loud — folding it into "fresh"
        # would invent the reassurance this withdraws.
        dn = tmp / "dn"
        _run(tmp, "clone", "-q", str(up), str(dn))
        _run(dn, "config", "user.email", "t@t")
        _run(dn, "config", "user.name", "t")
        fh = dn / ".git" / "FETCH_HEAD"
        with contextlib.suppress(OSError):
            fh.unlink()
        check(fetch_age_hours(dn) is None,
              "a clone with no FETCH_HEAD did not answer None")
        check(behind_origin(dn, ("*.sh",)) == (0, 0),
              "fixture precondition: the clone should be up to date")
        check(any("UNVERIFIED UPSTREAM" in ln
                  for ln in sweep_advisory([dn], ("*.sh",))),
              "a never-fetched repo reporting (0, 0) was SILENT — its 'up to "
              "date' rests on nothing at all")

        # A FRESH fetch: credible, and silent. The negative side of the arm —
        # without it the line could fire always and still pass.
        _run(dn, "fetch", "-q", "origin")
        age = fetch_age_hours(dn)
        check(age is not None and age < 1.0,
              f"a just-fetched repo did not read as recent: {age}")
        check(not any("UNVERIFIED UPSTREAM" in ln
                      for ln in sweep_advisory([dn], ("*.sh",))),
              "a freshly-fetched, up-to-date repo produced an advisory — the "
              "banner nobody reads")

        # ── THE ARM THIS EXISTS FOR ────────────────────────────────────────
        # Backdate FETCH_HEAD past the threshold. Nothing about the repo's
        # relationship to its LOCAL tracking ref changed — it is still exactly
        # (0, 0) — so before T5 this state was indistinguishable from the one
        # directly above, which is the whole defect.
        old = time.time() - (STALE_FETCH_HOURS + 12) * 3600
        os.utime(fh, (old, old))
        aged = fetch_age_hours(dn)
        check(aged is not None and aged > STALE_FETCH_HOURS,
              f"a backdated FETCH_HEAD did not read as stale: {aged}")
        check(behind_origin(dn, ("*.sh",)) == (0, 0),
              "backdating must not change the behind-ness reading — if it "
              "did, this arm would be testing the wrong thing")
        lines = sweep_advisory([dn], ("*.sh",))
        check(any("UNVERIFIED UPSTREAM" in ln for ln in lines),
              f"a (0, 0) resting on a {STALE_FETCH_HOURS + 12:.0f}h-old fetch "
              f"was reported as up to date: {lines}")

        # The buckets are NOT pooled: a repo already reported STALE gets the
        # STALE line and not a second one, even with an ancient FETCH_HEAD.
        (up / "a.sh").write_text("two\n", encoding="utf-8")
        _run(up, "add", "-A")
        _run(up, "commit", "-qm", "moved")
        _run(dn, "fetch", "-q", "origin")
        os.utime(fh, (old, old))
        lines = sweep_advisory([dn], ("*.sh",))
        check(any("STALE:" in ln for ln in lines),
              f"a genuinely-behind repo lost its STALE line: {lines}")
        check(not any("UNVERIFIED UPSTREAM" in ln for ln in lines),
              f"a repo already reported STALE was ALSO called unverified — "
              f"two lines for one repo is the pooling the family refuses: "
              f"{lines}")

    # ── THE SECOND SILENT BUCKET: `(n, 0)` ────────────────────────────────
    # `behind_origin` prints nothing for TWO readings, not one: `(0, 0)` "up
    # to date" and `(n, 0)` "behind, but on nothing in this sweep's
    # population". Only the first was ever challenged, and `(n, 0)` is the
    # STRONGER claim of the two — it asserts something about the CONTENT of
    # the commits it is behind by, read from the same unrefreshed ref.
    #
    # Its OWN fixture pair, deliberately: the arms above leave upstream ahead
    # on an IN-scope file, so reusing them cannot produce `(n, 0)` at all —
    # the first draft of this arm failed its own precondition with `(2, 1)`.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        up2 = _repo(tmp, "up2")
        (up2 / "a.sh").write_text("one\n", encoding="utf-8")
        _run(up2, "add", "-A")
        _run(up2, "commit", "-qm", "base")
        dn2 = tmp / "dn2"
        _run(tmp, "clone", "-q", str(up2), str(dn2))
        _run(dn2, "config", "user.email", "t@t")
        _run(dn2, "config", "user.name", "t")
        fh2 = dn2 / ".git" / "FETCH_HEAD"

        (up2 / "unrelated.txt").write_text("x\n", encoding="utf-8")
        _run(up2, "add", "-A")
        _run(up2, "commit", "-qm", "out of scope")
        _run(dn2, "fetch", "-q", "origin")
        beh = behind_origin(dn2, ("*.sh",))
        check(beh is not None and beh[0] > 0 and beh[1] == 0,
              f"fixture precondition: want (n, 0), got {beh}")

        # Fresh: silent, and that stays right — this bucket's whole point is
        # not being the banner nobody reads.
        lines = sweep_advisory([dn2], ("*.sh",))
        check(not any("UNVERIFIED UPSTREAM" in ln for ln in lines),
              f"a freshly-fetched (n, 0) produced an advisory: {lines}")
        check(not any("STALE:" in ln for ln in lines),
              f"an out-of-scope behind-ness was reported as STALE: {lines}")

        # Stale: the reading rests on nothing, so it must SAY so.
        aged2 = time.time() - (STALE_FETCH_HOURS + 12) * 3600
        os.utime(fh2, (aged2, aged2))
        lines = sweep_advisory([dn2], ("*.sh",))
        check(any("UNVERIFIED UPSTREAM" in ln for ln in lines),
              f"an (n, 0) resting on a {STALE_FETCH_HOURS + 12:.0f}h-old "
              f"fetch was SILENT — its claim that the commits it is behind "
              f"by miss this population is read from the same unrefreshed "
              f"ref: {lines}")
        # ⛔ PRESENCE is not CONTENT, and asserting only the former is how
        # the sentence came to be false here for a full day (Issue 850 T3).
        # T1 widened the trigger to this bucket and left the line saying
        # "report 'up to date'" — the one thing the sweep KNOWS is wrong
        # about an `(n, 0)` repo, since it has read the count.
        txt = " ".join(lines)
        check("up to date" not in txt,
              f"an (n, 0) repo was described as reading 'up to date' — the "
              f"sweep knows it is BEHIND and says so nowhere: {txt}")
        check("behind, none in this population" in txt,
              f"the (n, 0) reading was not NAMED in the advisory, so a reader "
              f"cannot tell which of two pooled readings is unverified: {txt}")
        check(f"{beh[0]} behind" in txt,
              f"the commits-behind count the reading rests on was not shown "
              f"(want {beh[0]}): {txt}")

    # A future-dated mtime clamps to 0 rather than going negative, and
    # unknown-ness keeps its single spelling.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        r = _repo(tmp, "skew")
        (r / ".git" / "FETCH_HEAD").write_text("", encoding="utf-8")
        ahead = time.time() + 7200
        os.utime(r / ".git" / "FETCH_HEAD", (ahead, ahead))
        check(fetch_age_hours(r) == 0.0,
              "a future-dated FETCH_HEAD did not clamp to 0")

    # The renderer, independent of git: an explicit None reads as the loud
    # "never fetched" rather than as an hour count.
    lines = worktree_advisory({}, unverified={"r": (None, 0)})
    check(any("never fetched" in ln for ln in lines),
          f"a None age was not rendered as never-fetched: {lines}")
    lines = worktree_advisory({}, unverified={"r": (99.0, 0)})
    check(any("99h" in ln for ln in lines),
          f"an age was not rendered in hours: {lines}")
    # Both directions of the bucket split, at the renderer, where no git tree
    # is needed to state them: a `(0, 0)` row must still SAY 'up to date' —
    # the repair must not blur the two the other way.
    z = " ".join(worktree_advisory({}, unverified={"r": (30.0, 0)}))
    check("reads 'up to date'" in z,
          f"a (0, 0) row lost its own reading in the split: {z}")
    check("behind" not in z,
          f"a (0, 0) row was described as behind something: {z}")
    n = " ".join(worktree_advisory({}, unverified={"r": (30.0, 7)}))
    check("7 behind, none in this population" in n,
          f"an (n, 0) row did not render its count and scope: {n}")
    check("up to date" not in n,
          f"an (n, 0) row was described as up to date: {n}")
    # ⛔ `deferral_line`, the shared FAILURE-path renderer. 17 sweeps had
    # `print(f"  ⚠ {_d}")` copy-pasted, and the `deferred` list holds lines
    # from TWO sources — bare population deferrals and already-glyphed
    # advisory lines — so every failing run printed `⚠ ⚠ STALE: …`.
    check(deferral_line("DEFERRED: 4 repos absent") == "⚠ DEFERRED: 4 repos absent",
          "a BARE deferral did not get its glyph")
    for g in ("⚠", "⛔", "✓", "✗", "▸"):
        check(deferral_line(f"{g} already glyphed") == f"{g} already glyphed",
              f"a line already starting with {g!r} was double-prefixed — "
              f"verdict lines here are PARSED interfaces")
    # It must not NORMALISE the glyph: ⛔ MASKED and ⚠ STALE are different
    # severities and flattening them loses the distinction the advisory draws.
    check(deferral_line("⛔ MASKED: 3 rows").startswith("⛔"),
          "the renderer downgraded a ⛔ to a ⚠ — severity is not its to choose")
    check(deferral_line("") == "⚠ ",
          "an empty line crashed or was left unprefixed")

    # One line, both readings — the split must not fragment the advisory into
    # a second line nobody reads.
    both = worktree_advisory({}, unverified={"a": (30.0, 0), "b": (30.0, 4)})
    check(len(both) == 1,
          f"the two readings were split across lines: {both}")
    check("reads 'up to date'" in both[0] and "4 behind" in both[0],
          f"one line did not carry both readings: {both[0]}")
    return fails


def n_assertions(src: str | None = None) -> int:
    """How many `check(...)` calls the arms below actually make.

    DERIVED, not typed. The hand-typed "36" in the line this feeds went stale
    the moment `stale_arms` landed, which is this repo's most-repeated finding
    about counts in prose — and a count nobody can red is a comment that
    drifts into a lie. Counted over `*_arms` functions only, so a `check` in
    production code could never inflate it.

    `src` is injected so the rule is testable against a KNOWN answer; without
    it the function read `__file__` and its three decisions were unarmable by
    construction — `arm_reach_audit --include-all` reported all of them as
    live survivors. The extraction IS the repair.

    Falls back to 0 — printed as `0 assertion(s)`, which is visibly wrong —
    rather than raising: a selftest that PASSED must not be turned into a
    crash by its own summary line.
    """
    import ast
    try:
        if src is None:
            src = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return 0
    n = 0
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef)
                and node.name.endswith("_arms")):
            continue
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "check"):
                n += 1
    return n


def counter_arms() -> list[str]:
    """`n_assertions` counts `check(...)` calls in `*_arms` functions ONLY."""
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    two = ("def foo_arms():\n"
           "    check(1, 'a')\n"
           "    check(2, 'b')\n")
    check(n_assertions(two) == 2,
          f"two checks in an _arms function were not counted: "
          f"{n_assertions(two)}")

    # The `_arms` name test. Without it, a `check` anywhere in the module
    # inflates the printed figure.
    other = ("def helper():\n"
             "    check(1, 'a')\n")
    check(n_assertions(other) == 0,
          "a check OUTSIDE an _arms function was counted — production code "
          "could then inflate the selftest's own headline number")

    # The `ast.Name` test: a METHOD call named check is a different callable.
    attr = ("def foo_arms():\n"
            "    obj.check(1, 'a')\n")
    check(n_assertions(attr) == 0,
          "an attribute call `obj.check()` was counted as the arm helper")

    # The `== \"check\"` test.
    named = ("def foo_arms():\n"
             "    verify(1, 'a')\n")
    check(n_assertions(named) == 0,
          "a call to a DIFFERENT function was counted as a check")

    # Unparsable source is 0, never a crash: the summary line must not turn a
    # PASSING selftest into a traceback.
    check(n_assertions("def (:::") == 0,
          "a syntax error did not fall back to 0")

    # Nested: `ast.walk` must reach a check inside a loop or an `if`.
    nested = ("def foo_arms():\n"
              "    for x in y:\n"
              "        if x:\n"
              "            check(x, 'deep')\n")
    check(n_assertions(nested) == 1,
          "a nested check was missed — the walk must not be top-level only")

    return fails


def premise_arms() -> list[str]:
    """git's OUTPUT SHAPE, which `behind_origin`'s two `or`s rely on."""
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        solo = _repo(tmp, "solo")
        (solo / "a.sh").write_text("x\n", encoding="utf-8")
        _run(solo, "add", "-A")
        _run(solo, "commit", "-qm", "base")

        up = _git(solo, "rev-parse", "--abbrev-ref", "--symbolic-full-name",
                  "@{upstream}")
        check(up.returncode != 0,
              "git rev-parse @{upstream} SUCCEEDED with no upstream — the "
              "returncode half of behind_origin's guard is no longer the "
              "signal it was written against")
        check(up.stdout.strip() == "",
              f"git rev-parse @{{upstream}} printed a ref to STDOUT with no "
              f"upstream ({up.stdout.strip()!r}) — the two halves of that "
              f"`or` no longer co-occur, so the redundancy is now load-bearing "
              f"and needs a real arm")

        # The second `or`: a failed rev-list prints no count to stdout, so
        # `.isdigit()` is False exactly when the returncode is non-zero.
        bad = _git(solo, "rev-list", "--count", "HEAD..nope/nope")
        check(bad.returncode != 0 and not bad.stdout.strip().isdigit(),
              f"a failing rev-list did not BOTH exit non-zero and withhold a "
              f"numeric count (rc={bad.returncode}, out={bad.stdout!r})")

    return fails


def worktree_arms() -> list[str]:
    """A REAL `git worktree`, against all five guards (Issue 836).

    ⛔ The fixture must be a genuine `git worktree add`, never a hand-written
    `.git` file: the point is that git ANSWERS here, so an arm whose fixture
    only looks worktree-shaped would assert the probe and not the behaviour —
    the `platform_dead_code` vendor-arm failure, where one exclusion had two
    code paths and the arm certified the path it was not aimed at.

    Every guard is asserted against git's own answer rather than against a
    literal, because the whole defect was five functions returning values that
    are individually legitimate readings.
    """
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        main = _repo(tmp, "main")
        (main / "a.md").write_text("one\n", encoding="utf-8")
        _run(main, "add", "-A")
        _run(main, "commit", "-qm", "base")

        wt = tmp / "wt"
        _run(main, "worktree", "add", "--detach", str(wt), "HEAD")
        check((wt / ".git").is_file(),
              "the fixture's .git is not a FILE — `git worktree add` no longer "
              "produces the shape this whole arm group is about, so every "
              "check below is asserting nothing")

        # The guard itself, both spellings, so the regression is named.
        check(is_checkout(wt),
              "is_checkout said NO to a real worktree — back to `.is_dir()`, "
              "and the head-provenance mechanism is inert there again (836)")

        (wt / "a.md").write_text("two\n", encoding="utf-8")
        truth = _git(wt, "status", "--porcelain", "--untracked-files=no")
        check(truth.stdout.strip() != "",
              "git itself reported a CLEAN worktree after an edit — the "
              "fixture is broken, not the code under test")

        check(dirty_files(wt) == frozenset({"a.md"}),
              f"dirty_files went blind in a worktree ({set(dirty_files(wt))}) "
              f"while git reported {truth.stdout.strip()!r} — a sweep there "
              f"files every row COMMITTED and re-pins from an in-flight edit")
        check(head_text(wt, "a.md") == "one\n",
              f"head_text did not read HEAD in a worktree "
              f"({head_text(wt, 'a.md')!r}) — the MASKED direction is dead, "
              f"which is the SILENT one")
        with head_tree(wt, ("*.md",)) as t:
            check(t is not None,
                  "head_tree yielded None for a DIRTY worktree — the caller "
                  "skips provenance entirely and reads the worktree as HEAD")

        # `fetch_age_hours` needs more than the probe: `.git` is a FILE here,
        # so a constructed `<root>/.git/FETCH_HEAD` cannot exist and OSError
        # would swap one silent None for another.
        loc = _git(wt, "rev-parse", "--git-path", "FETCH_HEAD")
        check(loc.returncode == 0 and loc.stdout.strip() not in ("", ".git/FETCH_HEAD"),
              f"git --git-path did not redirect FETCH_HEAD out of a worktree's "
              f"`.git` ({loc.stdout.strip()!r}) — the premise fetch_age_hours "
              f"now rests on")
        # ⚠ PREMISE: git answers `--git-path` with an ABSOLUTE path inside a
        # worktree and a RELATIVE one in an ordinary checkout, and `Path.__
        # truediv__` silently discards the left side when the right is
        # absolute — which is what makes ONE join correct for both shapes. If
        # git ever returned a relative path here the join would still "work"
        # and point at a file that cannot exist, i.e. back to a silent None.
        check(Path(loc.stdout.strip()).is_absolute(),
              f"git answered --git-path RELATIVELY inside a worktree "
              f"({loc.stdout.strip()!r}) — fetch_age_hours' single join now "
              f"resolves against the worktree and can only miss")
        real = Path(wt) / loc.stdout.strip()
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_text("", encoding="utf-8")
        check(fetch_age_hours(wt) is not None,
              "fetch_age_hours answered None for a worktree with a real "
              "FETCH_HEAD — it is still constructing the path instead of "
              "asking git for it")

        # ⛔ The requirement the probe actually exists for, UNCHANGED by the
        # widening: `git -C` walks UP, so a non-repo directory nested inside a
        # repo must not inherit its parent's answers. `.exists()` protects it
        # for the same reason `.is_dir()` did — there is no `.git` of any kind.
        plain = main / "plain"
        plain.mkdir()
        check(not is_checkout(plain) and dirty_files(plain) == frozenset()
              and head_text(plain, "a.md") is None,
              "the widening to .exists() let a non-repo directory inherit its "
              "PARENT's repository — the one thing the probe is for")

        _run(main, "worktree", "remove", "--force", str(wt))

    return fails


def selftest() -> list[str]:
    return (dirty_arms() + split_arms() + delta_arms() + overlay_arms()
            + key_arms() + tree_arms() + scope_arms()
            + advisory_arms() + stale_arms() + fetch_age_arms()
            + counter_arms() + premise_arms() + worktree_arms())


def main() -> int:
    fails = selftest()
    if fails:
        print("worktree_state selftest FAILED:")
        for f in fails:
            print("  ✗ " + f)
        return 1
    print(f"✓ worktree_state selftest — {n_assertions()} assertion(s): clean "
          "tree, unstaged + "
          "STAGED modification, untouched file not reported, git's POSIX "
          "separator asserted as a PREMISE, untracked excluded, head_text "
          "reads HEAD not the "
          "worktree and invents nothing, the .git probe refuses a PARENT's "
          "dirty set, an extracted tree is clean not an error, a rename "
          "reports its destination, a ONE-character filename survives the "
          "porcelain length guard, a NESTED path matches a bare-suffix "
          "pattern while a foreign population stays silent (and a BARE STRING "
          "is one pattern, not a character sequence), split withholds "
          "exactly the dirty rows "
          "(address-less row kept, separators normalised both sides), "
          "head_delta reports the THIRD bucket split withholds — a row HEAD "
          "carries too stays COMMITTED, an added one is UNCOMMITTED, a hidden "
          "one is MASKED and never both at once, a staged-only file has no "
          "HEAD blob, a deleted one still yields its committed rows, the "
          "rescan set is |dirty n population| and a clean tree costs zero "
          "reads, a duplicated HEAD key counts once, and the MASKED "
          "comparison is scoped so an ADDRESS-LESS key cannot be suppressed "
          "by a clean file, head_overlay carries HEAD's bytes for "
          "|dirty n population| and distinguishes a STAGED-only file (None) "
          "from an absent one while delta_of compares two COMPLETE sets, "
          "line_free is line-INVARIANT and strips only a bare integer while "
          "ordinal_keys disambiguates a repeated address and carries NO state "
          "between the worktree and HEAD passes, head_tree materialises a "
          "HEAD checkout that answers git ls-files AND git grep (forced "
          "add, so a tracked-but-gitignored path survives), excludes "
          "untracked files, yields None on a clean tree / out-of-scope dirt "
          "/ a non-repository and leaks nothing, NARROWS to a pathspec "
          "in both content and index while the default stays the whole tree, "
          "and widens its TRIGGER on extra_dirty without letting an untracked "
          "file into HEAD, "
          "a bare "
          "NAME + root resolves as a path does while an unresolvable one "
          "is SKIPPED, behind_origin separates None (no upstream, never "
          "guessed) from (0, 0) (up to date) and refuses to read AHEAD as "
          "behind, a (0, 0) resting on a FETCH_HEAD older than the threshold "
          "is called UNVERIFIED while a freshly-fetched one stays silent and "
          "a never-fetched clone is loud (None is cannot-tell, never age 0; "
          "a future-dated mtime clamps) and a repo already reported STALE "
          "does not also get that line, and the "
          "advisory is SILENT on a clean run with the four classes unpooled, "
          "and a REAL git worktree — where `.git` is a FILE — answers all "
          "five guards instead of returning the five values that each mean "
          "'nothing to report', while a non-repo directory nested inside a "
          "repo still inherits nothing from its parent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
