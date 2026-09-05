#!/usr/bin/env python3
"""Report whether a staged set looks like ONE editing episode (Issue 709 T3).

`git add -A` from a repo root is indistinguishable, to git, from intent. With
several agent sessions writing into one worktree it routinely picks up a
sibling's uncommitted work — `b2527521` committed three agents' WIP in six
files, one of which was a build regression nobody noticed for a day.

This is a **REPORT, not a gate** (always exit 0), for the same reason
`docs_drift_sweep.py` is: the refusing version trades a real class of loss
against friction on every legitimate multi-file commit, and that trade is an
owner call (Issue 709 T3). What is NOT an owner call is measuring the signal,
because it is cheap and it is the technique that actually caught this by hand
twice: **worktree mtimes cluster by editing episode.** A 204-file rustfmt sweep
lands in a 3-second window; a session's own edits land in the window it was
running. Two clusters in one staged set means two episodes.

Three independent signals are reported, and none subsumes another:

  1. **mtime clusters** — single-linkage over the staged files' mtimes. More
     than one cluster is the "you staged someone else's episode" shape.
  2. **also-dirty** — a staged path that ALSO has unstaged changes. That is a
     concurrent editor writing into the same file *right now*, which mtime
     clustering cannot see (their write may land in your window).
  3. **stale-vs-HEAD** — a file whose mtime PREDATES the commit that last
     touched its path. The worktree copy was written against an older version,
     so committing it reverts whatever landed since. Measured live: a 20:04:39
     rustfmt sweep of `tpr/als.rs` sat dirty while `0ef7f078` landed an Issue
     712 correctness fix in the same file at 21:07:13 — committing the sweep
     would have silently reverted it. Both other signals are blind to this:
     the sweep is ONE episode and its files are not also-dirty.

Signal 3 audits the dirty set too, not just the staged set, because the hazard
exists before anything is staged.

**Do not reach for `git log --author` first.** It is the obvious instinct and it
cannot work here: every concurrent session commits as the same git user, so
authorship does not distinguish "mine" from "a sibling's" at all. mtime is the
only signal that separates sessions, which is why signal 1 exists.

Nor can a whitespace-ignoring diff separate a reformat from a revert on its own.
On the `tpr/als.rs` case above, `git diff -w` reported 9 insertions / 30
deletions — but some of those deletions are rustfmt genuinely re-wrapping tokens
ACROSS lines (an import list, a fn signature), which `-w` cannot collapse. The
unambiguous evidence is the specific identifiers the newest commit introduced,
which is exactly what signal 3 reports back (`best_ssr` 5 occurrences at HEAD,
0 in the stale copy). Grep those, don't read the diffstat.

  4. **rustfmt round-trip** (Rust only, `--fmt`): `git show HEAD:$f | rustfmt
     --emit stdout | diff - $f`. Identical ⇒ the worktree copy is exactly
     "HEAD, formatted", provably carries **zero content**, and reverting it
     cannot lose anyone's work. This is the only signal here that yields a
     *proof* rather than evidence; the other three are heuristics.

Signal 4 is opt-in (`--fmt`) because it shells out to rustfmt per file and
needs a toolchain. It splits "dirty because churn" from "dirty because work"
with no mtime heuristic and no false positives from line re-wrapping — which
an `--ignore-all-space` diff cannot do, having called 214/215 files
non-whitespace in one measured case, because rustfmt re-wraps tokens ACROSS
lines. It made a 204-file revert provably rather than probably safe: 188 pure
churn, 16 real.

It also does something the peer's original formulation did not: the diff
against `rustfmt(HEAD)` **isolates the content**. Formatting cancels, so what
remains is exactly the semantic change, which is the review you actually want
on a file whose real edit is buried under a whole-file reformat.

Usage:  scripts/staged_set_audit.py [repo_path]
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Two edits more than this far apart are different episodes. Deliberately
# generous: a session that takes 10 min to write 4 files must read as one.
GAP_SECONDS = 900.0

# rustfmt needs an edition; a wrong one changes the formatting and would
# manufacture a false "content" verdict on every file.
RUST_EDITION = "2024"


def cluster(stamps: list[float], gap: float = GAP_SECONDS) -> list[list[float]]:
    """Single-linkage clustering over 1-D timestamps.

    Single-linkage, not fixed-width bins: a session editing continuously for an
    hour is ONE episode even though its span exceeds the gap, because no two
    consecutive edits are `gap` apart. Fixed bins would split it and report a
    false positive on exactly the sessions that do the most work.
    """
    out: list[list[float]] = []
    for t in sorted(stamps):
        if out and t - out[-1][-1] <= gap:
            out[-1].append(t)
        else:
            out.append([t])
    return out


def harden_output() -> None:
    """Make `print` survive repo content that the console codec cannot encode.

    `2eee1158` pinned the DECODE side (git's output → str). The ENCODE side was
    still locale-bound, and this report's whole point is to print FILE CONTENT
    as evidence — signal 3's "would lose:" lines and signal 4's isolated-line
    dump are arbitrary repo bytes. On this box (cp874) a `→` in a doc line
    crashed `main()` at the evidence print with UnicodeEncodeError, i.e. AFTER
    the audit had already found a real stale-vs-HEAD hazard and printed its
    header: the finding was on screen and the reason for it was not.

    `errors=` only, never `encoding=`: everything that renders today keeps
    rendering byte-identically (cp874 does carry the Windows punctuation block,
    so the script's own em-dashes were never the problem), and only the
    previously-fatal characters change — to a visible `\\u2192` escape. Pinning
    utf-8 instead would mojibake every Thai-console run to fix a crash on a
    handful of chars.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            # Not a TextIOWrapper (embedded, or already-detached stream).
            # A report that cannot harden its stdout still runs; it just keeps
            # the old failure mode on exotic content.
            pass


def selftest() -> None:
    """Runs on every invocation — a clustering regression must not be silent.

    Without this the audit would degrade to "1 cluster, always", print a
    confident single-episode verdict, and look exactly like a clean result.
    """
    # The evidence prints are the payload, so an un-encodable char in repo
    # content must degrade to an escape and never to a traceback. Asserted
    # against the LIVE stream state, so a future `reconfigure` that drops the
    # error handler (or a caller that resets it) reds here instead of dying
    # 200 lines later, halfway through a real finding.
    _enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    _err = getattr(sys.stdout, "errors", None) or "strict"
    assert "→".encode(_enc, errors=_err), (
        f"stdout ({_enc}/{_err}) cannot encode repo content — call harden_output()"
    )
    assert cluster([]) == [], "empty"
    assert cluster([100.0]) == [[100.0]], "single"
    assert len(cluster([0.0, 1.0, 2.0], gap=10)) == 1, "tight group is one episode"
    assert len(cluster([0.0, 100.0], gap=10)) == 2, "a gap splits"
    # Chained: total span 20 > gap 10, but no adjacent pair exceeds it.
    assert len(cluster([0.0, 8.0, 16.0, 20.0], gap=10)) == 1, "single-linkage chains"
    # Unsorted input must not change the answer.
    assert len(cluster([100.0, 0.0, 101.0], gap=10)) == 2, "sorted internally"
    # Signal 3's confirmation stage. Without these the containment check can
    # degrade to "nothing is ever missing" and print a clean pass over a copy
    # that reverts a landed fix — which is the exact bug it exists to catch.
    assert substantive("    let mut best_snap: Option<Vec<f32>> = None;"), "real line"
    assert not substantive("}"), "a brace is not evidence"
    assert not substantive("    );"), "nor is a close-paren"
    assert not substantive(""), "nor is a blank"
    # Sample ORDER is part of the report's usefulness: comments last, so the
    # printed evidence is an identifier to grep rather than the commit's prose.
    assert is_comment("  // Best-iterate guard (Issue 712)"), "rust comment"
    assert not is_comment("    let mut best_ssr = prev;"), "code is not a comment"
    # Signal 4 must never classify a non-Rust or unparseable file as churn:
    # that verdict authorises a revert, so its failure mode is destructive
    # while every other signal here only over- or under-warns.
    assert fmt_roundtrip(Path("."), "README.md")[0] == "skip", "non-Rust is skip"
    # Signal 3 must not be gated on freshness. A file regenerated AFTER its
    # commit has a newer mtime and is precisely the append-only-evidence
    # truncation case, so any reintroduced `mtime >= commit_t: continue`
    # would silently restore that blind spot.
    import inspect as _inspect

    _src = _inspect.getsource(stale_vs_head)
    assert "mtime >= commit_t" not in _src.replace("# ", ""), (
        "signal 3 must not pre-filter on mtime"
    )
    assert sorted(["// why", "let x = 1;"], key=is_comment)[0] == "let x = 1;", (
        "code sorts first"
    )
    # Every worktree read must pin UTF-8. `git()` pins it; `read_text()`
    # defaults to the LOCALE codec, so on a cp874 box the two sides of signal
    # 3's containment decode the same em-dash differently and the audit
    # invents "would lose" rows for every non-ASCII line. Measured 2026-09-05:
    # 5/5 dirty riir-train files reported STALE-vs-HEAD, all phantom. Pinned
    # on the SOURCE because the bug is invisible on a UTF-8 box — a runtime
    # assertion here would pass on CI and never fire where it bites.
    for _fn in (reverted_lines, fmt_roundtrip):
        for _line in _inspect.getsource(_fn).splitlines():
            if "read_text(" in _line and not _line.lstrip().startswith("#"):
                assert 'encoding="utf-8"' in _line, (
                    f"{_fn.__name__}: read_text must pin encoding "
                    f"(locale codec corrupts non-ASCII): {_line.strip()}"
                )


TRIVIAL = {"", "}", "{", "};", ")", ");", "),", "]", "};", "*/", "/*", "//"}


def substantive(line: str) -> bool:
    """Is this line specific enough that its absence means something?

    A `}` added by one commit and absent from a stale copy proves nothing — the
    copy has plenty of other `}`. Line-set containment is only evidence on
    lines that are unlikely to recur.
    """
    t = line.strip()
    return len(t) > 8 and t not in TRIVIAL


COMMENT_PREFIXES = ("//", "#", "/*", "*", "--", "%")


def is_comment(line: str) -> bool:
    return line.strip().startswith(COMMENT_PREFIXES)


def reverted_lines(repo: Path, path: str, sha: str) -> list[str]:
    """Substantive lines `sha` ADDED to `path` that the worktree copy lacks.

    This is the exact form of "committing this reverts what landed since": if
    the newest commit on a path added lines the worktree file does not contain,
    committing that file removes them. Set containment, not a diff, because the
    stale copy may also have moved things around — position is not the claim,
    presence is.
    """
    show = git(repo, "show", "--format=", "--unified=0", sha, "--", path)
    added = [
        ln[1:]
        for ln in show.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]
    # encoding pinned for the same reason `git()` pins it, and this is the
    # THIRD site of that bug: `git show` above is decoded UTF-8 while
    # `read_text()` without an encoding uses the LOCALE codec. On a cp874 box
    # an em-dash comes back from git as `—` and from the file as `โ€”`, so
    # containment fails for EVERY line carrying a non-ASCII char and the audit
    # invents "would lose" rows against a file that has them. Unlike the two
    # decode/encode crashes already fixed, this one does not raise — it
    # silently reports wrong findings, which is the worse failure.
    have = {
        ln.strip()
        for ln in (repo / path)
        .read_text(encoding="utf-8", errors="replace")
        .splitlines()
    }
    lost = [ln for ln in added if substantive(ln) and ln.strip() not in have]
    # Code before comments, order otherwise preserved. A commit's first
    # added lines are usually its explanatory comment block, and prose is
    # the weakest evidence in the set — an identifier like `best_ssr`
    # settles reformat-vs-revert with one grep, a comment does not.
    return sorted(lost, key=is_comment)


def fmt_roundtrip(repo: Path, path: str) -> tuple[str, list[str]]:
    """Classify a dirty Rust file as pure formatting churn or real content.

    Returns `(verdict, isolated_diff)` where verdict is:

    - `"churn"`   — worktree == rustfmt(HEAD). Zero content. Reverting is
                    lossless BY PROOF, not by inspection.
    - `"content"` — a semantic change survives after formatting cancels. The
                    returned diff is that change, isolated.
    - `"skip"`    — not Rust, or rustfmt could not parse either side. Reported
                    as unknown rather than assumed either way: a parse failure
                    that silently read as "churn" would authorise a destructive
                    revert, which is the one error this must not make.
    """
    if not path.endswith(".rs"):
        return "skip", []
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{path}"],
            capture_output=True, check=True,
        ).stdout
        fmt = subprocess.run(
            ["rustfmt", "--edition", RUST_EDITION, "--emit", "stdout"],
            input=head, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "skip", []
    if fmt.returncode != 0:
        return "skip", []

    # `bytes.decode()` is UTF-8 by default; `read_text()` is NOT — it uses the
    # locale codec. Unpinned, the two sides decode differently on any non-UTF-8
    # box and `want == have` can never hold for a file containing an em-dash,
    # so every such file reports "content" and the isolated diff is mojibake.
    # Harmless in direction (a false `churn` is the one that authorises a
    # revert) but it destroys the only signal here that yields a proof.
    want = fmt.stdout.decode(errors="replace").splitlines()
    have = (repo / path).read_text(encoding="utf-8", errors="replace").splitlines()
    if want == have:
        return "churn", []
    import difflib

    delta = [
        ln
        for ln in difflib.unified_diff(want, have, "rustfmt(HEAD)", "worktree", n=0)
        if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))
    ]
    return "content", delta


def stale_vs_head(
    repo: Path, paths: list[str]
) -> list[tuple[str, float, float, list[str]]]:
    """Paths whose worktree copy would REVERT the newest commit touching them.

    **Line containment is the whole test.** Only lines the newest commit on a
    path ADDED, and the worktree copy LACKS, are evidence — restricted to lines
    specific enough that their absence means something (a `}` proves nothing).

    This once had an `mtime < commit_time` pre-filter, added because that
    comparison alone false-positives on the commonest shape there is: you edit
    at 21:03 and commit at 21:04, so the newest commit on that path is your own
    edit. **Removed 2026-09-03 — it was redundant AND blinding.** Redundant
    because containment already cleared both of those false positives on its
    own; blinding because a file *regenerated after* its commit has a NEWER
    mtime and was skipped before containment ever ran.

    That blind spot was live, not hypothetical: a tracked benchmark
    contamination log in `riir-ai` had been overwritten by a later, clean
    sample — 33 committed lines of evidence (concurrent cargo builds, Zed at
    160% CPU) replaced by one line reading as a clean run. mtime 03:33 against
    a 15:39 commit the previous day, so the pre-filter excluded it and the
    audit reported the tree clean. Append-only evidence artifacts are exactly
    the files this check exists for and exactly the ones a freshness heuristic
    cannot see.

    A path with no HEAD history (newly added) cannot be stale.
    """
    out: list[tuple[str, float, float, list[str]]] = []
    for p in paths:
        f = repo / p
        if not f.is_file():
            continue
        head = git(repo, "log", "-1", "--format=%ct %H", "HEAD", "--", p).split()
        if len(head) != 2:
            continue
        commit_t, sha = float(head[0]), head[1]
        mtime = f.stat().st_mtime
        # No freshness pre-filter: see the docstring. Containment decides.
        lost = reverted_lines(repo, p, sha)
        if lost:
            out.append((p, mtime, commit_t, lost))
    return out


def git(repo: Path, *args: str) -> str:
    # encoding pinned: text=True alone uses the LOCALE codec, and on a Windows
    # box set to cp874 any UTF-8 byte above that range (em-dashes, box-drawing
    # chars in Rust comments) crashes the reader thread mid-audit
    # (UnicodeDecodeError in subprocess._readerthread, 2026-09-04).
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout


def main() -> int:
    harden_output()  # before selftest: its own failure message must be printable
    selftest()
    repo = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()

    staged = [p for p in git(repo, "diff", "--cached", "--name-only").splitlines() if p]
    also_dirty = {p for p in git(repo, "diff", "--name-only").splitlines() if p}
    # Signal 3 applies to the DIRTY set too, so "nothing staged" is not an
    # early exit — the revert hazard exists before anything is staged.
    if not staged and not also_dirty:
        print("staged-set audit: nothing staged, working tree clean")
        return 0
    # A staged deletion has no worktree file to stat; it belongs to no episode.
    timed: list[tuple[float, str]] = []
    missing: list[str] = []
    for p in staged:
        f = repo / p
        if f.is_file():
            timed.append((f.stat().st_mtime, p))
        else:
            missing.append(p)

    groups = cluster([t for t, _ in timed])
    by_stamp = {}
    for t, p in timed:
        by_stamp.setdefault(t, []).append(p)

    print(
        f"staged-set audit: {len(staged)} staged + {len(also_dirty)} dirty path(s) "
        f"in {repo.name}"
    )
    if missing:
        print(f"  {len(missing)} deleted/absent (no mtime, no episode): {', '.join(missing[:4])}")

    import datetime as _dt

    for i, g in enumerate(groups, 1):
        files = [p for t in g for p in by_stamp[t]]
        span = g[-1] - g[0]
        when = _dt.datetime.fromtimestamp(g[0]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  episode {i}: {len(files)} file(s), started {when}, span {span:.0f}s")
        for p in files[:6]:
            print(f"      {p}")
        if len(files) > 6:
            print(f"      … {len(files) - 6} more")

    overlap = sorted(set(staged) & also_dirty)
    if overlap:
        print(
            f"  ALSO-DIRTY: {len(overlap)} staged path(s) still have unstaged changes — "
            "someone is editing them concurrently, or you staged a partial blob on purpose"
        )
        for p in overlap[:6]:
            print(f"      {p}")

    stale = stale_vs_head(repo, sorted(set(staged) | also_dirty))
    if stale:
        print(
            f"  STALE-vs-HEAD: {len(stale)} path(s) LACK lines the newest commit on their "
            "own path added — committing these reverts what landed since"
        )
        for p, mt, ct, lost in stale[:8]:
            # Date-qualify whenever the two fall on different days. A bare
            # `%H:%M:%S` made a commit from YESTERDAY at 17:01 read as
            # happening AFTER a write today at 10:23 — i.e. in the future —
            # which inverts the one relation the reader is checking. Misread
            # exactly that way on 2026-09-03. Same day stays terse, because
            # that is the common case and the date adds nothing there.
            wd = _dt.datetime.fromtimestamp(mt)
            cd = _dt.datetime.fromtimestamp(ct)
            fmt = "%H:%M:%S" if wd.date() == cd.date() else "%Y-%m-%d %H:%M:%S"
            w, c = wd.strftime(fmt), cd.strftime(fmt)
            print(
                f"      {p}  (written {w}, HEAD touched it {c}, "
                f"{len(lost)} line(s) would be lost)"
            )
            # Print the evidence, not only its count: these lines ARE the
            # grep that settles reformat-vs-revert, and a diffstat cannot
            # (rustfmt re-wraps tokens across lines, so `-w` still shows
            # deletions). Reconstructed by hand once already; don't make the
            # next reader do it.
            for ln in lost[:2]:
                print(f"        would lose: {ln.strip()[:96]}")
        if len(stale) > 8:
            print(f"      … {len(stale) - 8} more")

    if "--fmt" in sys.argv:
        churn, content, skip = [], [], []
        for p in sorted(set(staged) | also_dirty):
            verdict, delta = fmt_roundtrip(repo, p)
            {"churn": churn, "content": content, "skip": skip}[verdict].append((p, delta))
        print(
            f"  rustfmt round-trip: {len(churn)} pure churn (revert is lossless "
            f"BY PROOF), {len(content)} carry content, {len(skip)} unclassifiable"
        )
        for p, _ in churn:
            print(f"      churn    {p}")
        for p, delta in content:
            print(f"      CONTENT  {p}  ({len(delta)} isolated line(s))")
            for ln in delta[:4]:
                print(f"          {ln[:100]}")
        for p, _ in skip:
            print(f"      skip     {p}")

    match len(groups) > 1:
        case True if staged:
            print(
                f"  REVIEW: {len(groups)} editing episodes in one staged set. If you did not "
                "write the older one(s), unstage them — see AGENTS.md on `git add -A`."
            )
        case _ if staged:
            print("  ✓ one editing episode")
        case _:
            print("  (nothing staged — episode clustering not applicable)")
    # Report, never a gate: the refusing version is Issue 709 T3's owner call.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
