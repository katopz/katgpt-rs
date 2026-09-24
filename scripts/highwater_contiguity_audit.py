#!/usr/bin/env python3
"""AUDIT (report, exit 0): is `.highwater` a sound OWNERSHIP witness?

Issue 768 T3. The citation instruments (`issue_citation_gate.py`,
`citation_drift_sweep.py`) resolve number ownership from three witnesses —
worktree files, `git log` history, `heading_allocated()` — and a fourth is
proposed: `n <= .highwater ⇒ n was allocated here`. The Numbering Discipline
(read value+1, write back, never reuse) makes that an implication IF AND ONLY
IF the counter is CONTIGUOUS: every step +1, never a jump, never a reset.
A counter that jumped 765→768 would make the witness claim 766 and 767 for
numbers nobody ever allocated — and a witness that over-claims does not
merely miss findings, it VALIDATES wrong addresses (the exact failure class
the witness exists to prevent, inverted).

Two views, because neither alone can decide it:

  A (static)   numbers `n <= hw` with NO witness in `allocated()` — the
               over-claim EXPOSURE. Cannot distinguish "file-and-removed
               same day" (legitimately allocated, the Issue 766 class) from
               "never allocated" (a jump) — that is View B's job.
  B (dynamic)  the counter's own committed transitions over git history —
               a step > +1 is a GAP (numbers skipped: never allocated by
               anyone), a step < 0 is a RESET (monotonicity broken). Either
               one red-lines the witness for that repo+kind.

⛔ The `.benchmarks` kind is SEMANTICALLY NOT UNIFORM across the workspace:
riir-auth's `.benchmarks/.highwater` counts RECORDS while its files are named
after PLAN numbers (its AGENTS.md says so) — `n <= hw` claims nothing there.
The audit prints every kind but flags that one; the WITNESS (if landed per
Issue 768 T4) must be kind-aware, not blanket.

Report, not a gate (exit 0) — findings here inform the Issue 768 landing
decision; nothing reds off this script. Population derived (BOUNDARY.md +
`.git`), same source as every instrument in this family; an EMPTY population
exits 2 (instrument blind), never a green over zero repos.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# Issue 804: this instrument is documented as directly invokable, and its
# verdict glyphs (✓ ✗ ⛔ ⚠) kill it on a non-UTF-8 console — no verdict at
# all, findings unread. docs_gate.sh's PYTHONIOENCODING only covers runs
# that go through the wrapper.
import console_safe  # noqa: E402

console_safe.apply()

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# issue_citation_gate is imported LAZILY inside main(): this module sits BELOW
# the citation gate in the import graph (numbering_drift_sweep imports this
# walker, and the gate imports numbering_drift_sweep for its seventh
# population predicate), so a module-level import here is a CYCLE that
# detonates only when the gate runs as a script — the __main__ double-load
# trap, caught by docs_gate on the Issue 769 landing.

REPO_ROOT = HERE.parent
WORKSPACE = Path(os.environ.get("WORKSPACE_ROOT", str(REPO_ROOT.parent)))

# riir-auth's bench counter is COUNT-based (AGENTS.md: "the highwater counts
# bench RECORDS"); files are named for PLAN numbers. One documented exception,
# named inline where it is excused from witness semantics.
COUNT_BASED_BENCH = {"riir-auth"}


def commit_subject(repo: Path, h: str) -> str:
    """One-line subject for a commit (lazy: only reset rows need it)."""
    r = subprocess.run(
        ["git", "-C", str(repo), "show", "-s", "--format=%s", h],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    return r.stdout.strip()


def parse_cat_file_batch(out: bytes, want: list[str]) -> dict[str, int]:
    """`{spec: counter value}` from `git cat-file --batch` output, one
    response per spec in `want` order.

    The wire format per response is `<oid> blob <size>\n<content>\n` (a
    missing object answers `<spec> missing\n`). The content is read as
    EXACTLY `size` bytes and the single LF after it is consumed — the only
    framing that holds for every blob. The previous reader took ONE LINE of
    content plus an optional blank, which is correct only for a
    single-line counter: measured 2026-09-24, riir-ai `892dec017` committed
    a two-line `.issues/.highwater` (`998\n999`), its second line was read
    as the NEXT response's header, and every older commit's value came back
    one response off — 17 phantom resets (`4f14e64250` reported 588->586
    where both it and its parent hold 587), riir-ai's pin breached 26 > 9.

    Value = the LAST integer token of the content, so a multi-line blob
    reads as the value its final line claims (the allocator's intent when
    a write appended rather than replaced); an unparseable blob is skipped —
    the malformed class belongs to the numbering gate, not to this walk.
    """
    val: dict[str, int] = {}
    pos, n = 0, len(out)
    for spec in want:
        if pos >= n:
            break
        nl = out.find(b"\n", pos)
        if nl < 0:
            break
        head = out[pos:nl].decode("utf-8", "replace")
        pos = nl + 1
        m = re.match(r"^[0-9a-f]+ \S+ (\d+)$", head)
        if not m:
            continue                       # "<spec> missing" — never existed
        size = int(m.group(1))
        body = out[pos:pos + size].decode("utf-8", "replace")
        pos += size + 1                    # content + its one LF terminator
        if not head.split()[1] == "blob":
            continue
        try:
            val[spec] = int(body.strip().split()[-1])
        except (ValueError, IndexError):
            continue                       # malformed — ng's class owns it
    return val


def counter_history(repo: Path, subdir: str) -> list[dict]:
    """Per-commit counter events over HEAD's history, parent-compared.

    The Issue-770 walker. The Issue-769 one ordered ALL refs' transitions by
    commit DATE and walked one `current` — two defects, both measured against
    the real diffs: after any real backward event, every ordinary +1 bump on
    the OTHER lineage landed below the walk's base and was flagged (15 of 31
    phantom reset rows); and `git log -p` emits no diff for MERGE commits, so
    a merge resolving a counter conflict by taking the lower value — the
    dangerous case — was invisible while an innocent concurrent bump took
    the blame (making the "all non-merge" adjudication true by construction).

    Per-commit parent comparison is lineage-correct by construction: for
    every HEAD-reachable commit, the counter blob is read at the commit and
    at EACH parent (one `git cat-file --batch` for the whole walk), and the
    event is judged against `max(parent values)` only — never against a
    cross-lineage walk position. Merge commits are ordinary rows here.
    HEAD-reachable, not --all: sweep pins must be a function of the branch
    the box carries, not of which refs happen to be fetched (Issue 770 T3).

    Event dicts: {hash, value, base, kind} where kind is 'reset' | 'gap' |
    'alloc' (+1 on its own lineage — the dual-allocation feed) | 'hold'
    (equal, or a merge keeping a parent's value). [] when the repo carries
    no committed counter at all — the caller reports that shape.
    """
    # (1) every (commit, parents) on HEAD
    r = subprocess.run(
        ["git", "-C", str(repo), "rev-list", "--parents", "HEAD"],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    pairs = [tuple(parts) for parts in
             (ln.split() for ln in r.stdout.splitlines()) if parts]
    if not pairs:
        return []

    # (2) blob values for every commit that could carry the file, in ONE
    # cat-file --batch pass, parsed by `parse_cat_file_batch` — by the
    # response's declared SIZE, never by line (see its docstring for the
    # measured desync a line-based reader produced).
    rel = f"{subdir}/.highwater"
    want: list[str] = []
    for c, *ps in pairs:
        want.append(c)
        want.extend(ps)
    batch = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input="\n".join(f"{s}:{rel}" for s in want).encode("utf-8") + b"\n",
        capture_output=True,
    )
    val = parse_cat_file_batch(batch.stdout, want)

    # (3) classify each commit against ITS OWN parents
    events: list[dict] = []
    for c, *ps in pairs:
        v = val.get(c)
        if v is None:
            continue                       # absent at this commit — not yet
                                           # committed, or a mid-history deletion
                                           # (counters never are). No event row
                                           # either way: each later commit is
                                           # judged against its own parents, so
                                           # there is no walk state to hold.
        pv = [val[p] for p in ps if p in val]
        base = max(pv) if pv else 0
        if v < base:
            kind = "reset"
        elif v > base + 1:
            kind = "gap"
        elif v == base + 1:
            kind = "alloc"
        else:
            kind = "hold"
        events.append({"hash": c, "value": v, "base": base, "kind": kind})
    return events


def classify_history(events: list[dict], repo: Path | None = None,
                     is_ancestor=None) -> dict:
    """Group per-commit events into the audit's classes.

    resets/gaps pass through sorted (stable output), each row naming the
    commit that DID it (subjects are fetched lazily by the printer — only
    reset rows need them). Duals — the same number allocated on two
    non-ancestor lineages — come from `alloc` events grouped by value: two
    allocs of v where neither commit is an ancestor of the other.
    `is_ancestor(a, b)` is injectable for tests; the default shells
    `git -C repo merge-base --is-ancestor` per pair, bounded by
    DUAL_PAIR_CAP per value.
    """
    DUAL_PAIR_CAP = 6
    resets = sorted((e["base"], e["value"], e["hash"])
                    for e in events if e["kind"] == "reset")
    gaps = sorted((e["base"], e["value"])
                  for e in events if e["kind"] == "gap")
    if is_ancestor is None:
        if repo is None:
            raise ValueError("classify_history needs repo= for the default "
                             "is_ancestor (or an injected is_ancestor)")
        def is_ancestor(a: str, b: str) -> bool:
            return subprocess.run(
                ["git", "-C", str(repo), "merge-base", "--is-ancestor",
                 a, b]).returncode == 0
    duals = 0
    by_value: dict[int, list[str]] = {}
    for e in events:
        if e["kind"] == "alloc":
            by_value.setdefault(e["value"], []).append(e["hash"])
    for _, hs in sorted(by_value.items()):
        if len(hs) < 2:
            continue
        checked = 0
        for i in range(len(hs)):
            for j in range(len(hs)):
                if i == j:
                    continue
                checked += 1
                if checked > DUAL_PAIR_CAP:
                    break
                # hs order follows rev-list (child-before-parent); a pair is
                # dual iff neither reaches the other
                if not is_ancestor(hs[i], hs[j]):
                    duals += 1
                    break
            else:
                continue
            break
    final = max((e["value"] for e in events), default=None)
    return {"gaps": gaps, "resets": resets, "duals": duals, "final": final}


def selftest() -> list[str]:
    """Regression fixtures on REAL git repos (Issue 770): the merge-reset the
    old walker could not see, the phantom reset it manufactured, and straight
    line semantics. The first two were measured against constructed repos by
    the verdict reviewer; landing them as fixtures pins the repairs."""
    import tempfile

    fails: list[str] = []

    def git(repo: Path, *args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True,
                       capture_output=True)

    def git_may_conflict(repo: Path, *args: str) -> None:
        """Merges over a counter both sides touched CONFLICT (exit 1) — that
        is the fixture's point; the resolution that follows is the event."""
        r = subprocess.run(["git", "-C", str(repo), *args],
                           capture_output=True, encoding="utf-8", errors="replace")
        if r.returncode not in (0, 1):
            raise subprocess.CalledProcessError(r.returncode, r.args, r.stdout,
                                                r.stderr)

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)

        # ── fixture 1 (the reviewer's constructed case): branch bumps to 20,
        # main bumps to 11, the MERGE resolves 20→11. The merge is the reset;
        # the innocent main bump (10→11) is NOT.
        repo = ws / "merge-repo"
        repo.mkdir()
        git(repo, "init", "-q", "-b", "main")
        (repo / ".issues").mkdir()
        hw = repo / ".issues" / ".highwater"
        hw.write_text("10\n", encoding="utf-8")
        git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "base 10")
        git(repo, "checkout", "-q", "-b", "side")
        hw.write_text("20\n", encoding="utf-8")
        git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "branch a bumps to 20")
        git(repo, "checkout", "-q", "main")
        hw.write_text("11\n", encoding="utf-8")
        git(repo, "add", "-A"); git(repo, "commit", "-q", "-m", "main bumps to 11")
        git_may_conflict(repo, "merge", "--no-commit", "side")
        hw.write_text("11\n", encoding="utf-8")  # the conflict resolution that RESETS 20→11
        git(repo, "add", "-A")
        subprocess.run(["git", "-C", str(repo), "commit", "-q",
                        "-m", "MERGE: resolve conflict by RESETTING 20 to 11"],
                       check=True, capture_output=True)
        w = classify_history(counter_history(repo, ".issues"), repo=repo)
        if len(w["resets"]) != 1 or w["resets"][0][:2] != (20, 11):
            fails.append(f"merge-reset: the merge taking 11 over a parent's 20 "
                         f"must be the ONE reset (20, 11): {w['resets']}")
        else:
            subj = commit_subject(repo, w["resets"][0][2])
            if "MERGE" not in subj:
                fails.append(f"merge-reset: the blamed commit must be the merge, "
                             f"got subject {subj!r}")
        if w["final"] != 20:
            fails.append(f"final must reflect the highest committed value (20), "
                         f"got {w['final']}")

        # ── fixture 2 (the phantom): two lineages each allocate 205 from the
        # same base, main then climbs to 564, the merge keeps 564. NO reset
        # anywhere — the old walker flagged 15 such rows; 205 twice on
        # non-ancestor lineages is one DUAL.
        repo2 = ws / "phantom-repo"
        repo2.mkdir()
        git(repo2, "init", "-q", "-b", "main")
        (repo2 / ".benchmarks").mkdir()
        hw2 = repo2 / ".benchmarks" / ".highwater"
        hw2.write_text("204\n", encoding="utf-8")
        git(repo2, "add", "-A"); git(repo2, "commit", "-q", "-m", "base 204")
        git(repo2, "checkout", "-q", "-b", "old-lane")
        hw2.write_text("205\n", encoding="utf-8")
        git(repo2, "add", "-A"); git(repo2, "commit", "-q", "-m", "side lane allocates 205")
        git(repo2, "checkout", "-q", "main")
        hw2.write_text("205\n", encoding="utf-8")
        git(repo2, "add", "-A"); git(repo2, "commit", "-q", "-m", "main allocates 205 too")
        hw2.write_text("564\n", encoding="utf-8")
        git(repo2, "add", "-A"); git(repo2, "commit", "-q", "-m", "main jumps to 564")
        git_may_conflict(repo2, "merge", "--no-commit", "old-lane")
        hw2.write_text("564\n", encoding="utf-8")  # the merge keeps main's higher value
        git(repo2, "add", "-A")
        subprocess.run(["git", "-C", str(repo2), "commit", "-q",
                        "-m", "MERGE: keep 564"],
                       check=True, capture_output=True)
        w2 = classify_history(counter_history(repo2, ".benchmarks"), repo=repo2)
        if w2["resets"]:
            fails.append(f"phantom: a merged side-lane +1 must not be a reset "
                         f"(the merge kept main's higher value): {w2['resets']}")
        if w2["duals"] != 1:
            fails.append(f"phantom: 205 allocated on two non-ancestor lineages "
                         f"— one dual expected: {w2}")

        # ── fixture 3 (straight-line semantics): a jump is a gap; a real
        # backward commit is a reset at its own commit.
        repo3 = ws / "line-repo"
        repo3.mkdir()
        git(repo3, "init", "-q", "-b", "main")
        (repo3 / ".issues").mkdir()
        hw3 = repo3 / ".issues" / ".highwater"
        hw3.write_text("765\n", encoding="utf-8")
        git(repo3, "add", "-A"); git(repo3, "commit", "-q", "-m", "765")
        hw3.write_text("768\n", encoding="utf-8")
        git(repo3, "add", "-A"); git(repo3, "commit", "-q", "-m", "jump to 768")
        hw3.write_text("769\n", encoding="utf-8")
        git(repo3, "add", "-A"); git(repo3, "commit", "-q", "-m", "769")
        w3 = classify_history(counter_history(repo3, ".issues"), repo=repo3)
        if w3["gaps"] != [(0, 765), (765, 768)] or w3["resets"] or w3["final"] != 769:
            fails.append(f"line: creation (0,765) + jump (765,768) gaps only — "
                         f"got gaps={w3['gaps']} resets={w3['resets']} "
                         f"final={w3['final']}")
        hw3.write_text("4\n", encoding="utf-8")
        git(repo3, "add", "-A"); git(repo3, "commit", "-q", "-m", "back to 4")
        w3b = classify_history(counter_history(repo3, ".issues"), repo=repo3)
        if len(w3b["resets"]) != 1 or w3b["resets"][0][:2] != (769, 4):
            fails.append(f"line: a real backward commit is a reset (769, 4): "
                         f"{w3b['resets']}")

        # ── the framing: a two-line blob, a newline-less blob, a missing
        # object and an empty blob, back to back. Each value must land on
        # ITS OWN spec — a line-based reader shifts every spec after `b`.
        wire = (b"1111 blob 8\n998\n999\n\n"      # b: two lines
                b"2222 blob 3\n587\n"               # c: no trailing LF
                b"c:x missing\n"                     # d: missing
                b"3333 blob 0\n\n"                  # e: empty
                b"4444 blob 4\n626\n\n")           # f: ordinary
        got = parse_cat_file_batch(wire, ["b", "c", "d", "e", "f"])
        if got != {"b": 999, "c": 587, "f": 626}:
            fails.append(f"framing: each blob's value on its own spec "
                         f"{{b: 999, c: 587, f: 626}}, got {got}")

        # ── the same on REAL git (riir-ai 892dec017's shape): a two-line
        # counter must not shift any OLDER commit's value. The history needs
        # a MERGE below the bad blob: on a straight line a commit and its
        # parent shift by the same stride, so a line-based reader still sees
        # +1 steps and passes (measured — the straight-line version of this
        # arm was green against the broken reader). With the merge the old
        # reader printed phantom gaps (0,4) (2,4) (4,6); the answer is none.
        repo4 = ws / "twoline-repo"
        repo4.mkdir()
        git(repo4, "init", "-q", "-b", "main")
        (repo4 / ".issues").mkdir()
        hw4 = repo4 / ".issues" / ".highwater"

        def commit4(text: str, msg: str) -> None:
            hw4.write_bytes(text.encode("utf-8"))
            git(repo4, "add", "-A")
            git(repo4, "commit", "-q", "-m", msg)
        commit4("1\n", "1")
        git(repo4, "checkout", "-q", "-b", "side")
        commit4("2\n", "side 2")
        git(repo4, "checkout", "-q", "main")
        git(repo4, "merge", "-q", "--no-ff", "-m", "merge side", "side")
        commit4("3\n", "3")
        commit4("4\n", "4")
        commit4("4\n5\n", "two-line counter")
        commit4("6\n", "6")
        w4 = classify_history(counter_history(repo4, ".issues"), repo=repo4)
        if w4["resets"] or w4["gaps"] or w4["final"] != 6:
            fails.append(f"two-line blob on real git: 0 resets, 0 gaps, final "
                         f"6 — got resets={w4['resets']} gaps={w4['gaps']} "
                         f"final={w4['final']}")

        # ── non-git dir: [] (the sweep selftest's tempdir shape)
        ng_repo = ws / "not-a-repo"
        ng_repo.mkdir()
        if counter_history(ng_repo, ".issues") != []:
            fails.append("non-git: must return []")
    return fails


def main() -> int:
    import issue_citation_gate as icg  # noqa: E402  (lazy: see the module header)
    fails = selftest()
    if fails:
        print("✗ highwater contiguity audit SELFTEST FAILED:")
        for f in fails:
            print(f"    {f}")
        return 2

    repos = icg.contract_repos(WORKSPACE)
    if not repos:
        print(f"✗ derived population is EMPTY under {WORKSPACE} — refusing "
              f"to report over zero repos")
        return 2

    rows = gaps_total = resets_total = 0
    print(f"highwater contiguity audit — {len(repos)} contract repo(s), "
          f"kinds {', '.join(icg.KINDS)}\n")
    print(f"{'repo/kind':<34s} {'hw':>5s} {'witnessed':>9s} {'missing':>7s} "
          f"{'gaps':>4s} {'resets':>6s} {'duals':>5s}  notes")
    for repo in sorted(repos, key=lambda p: p.name):
        for kind, sub in icg.KINDS.items():
            hw_f = repo / sub / ".highwater"
            if not hw_f.is_file():
                continue
            try:
                hw = int(hw_f.read_text(encoding="utf-8").strip().split()[-1])
            except (ValueError, IndexError, OSError):
                print(f"{repo.name}/{kind:<26s}  MALFORMED counter — "
                      f"numbering_gate.py owns that verdict")
                continue
            witnessed = icg.allocated(repo, sub)
            missing = [n for n in range(1, hw + 1) if n not in witnessed]
            events = counter_history(repo, sub)
            w = classify_history(events, repo=repo) if events else {"gaps": [], "resets": [],
                                                        "duals": 0, "final": None}
            rows += 1
            gaps_total += len(w["gaps"])
            resets_total += len(w["resets"])
            notes = []
            if repo.name in COUNT_BASED_BENCH and kind == "Bench":
                notes.append("COUNT-BASED counter (records, not numbers) — "
                             "witness EXCLUDED by design")
            if not events:
                notes.append("no committed history for the counter")
            elif w["final"] is not None and w["final"] > hw:
                notes.append(f"history max {w['final']} > worktree {hw} "
                             f"(un-bumped worktree — checkout state, REPORT)")
            if w["gaps"]:
                notes.append("GAPS " + ", ".join(f"{a}->{b}" for a, b in w["gaps"][:4]))
            if w["resets"]:
                notes.append("RESETS " + ", ".join(
                    f"{a}->{b}" for a, b, _ in w["resets"][:4]))
                for a, b, h in w["resets"][:4]:
                    notes.append(f"    reset {a}->{b} @ {h[:10]} "
                                 f"{commit_subject(repo, h)[:70]}")
            if missing and len(missing) > 8:
                notes.append(f"{len(missing)} unwitnessed (file-and-removed "
                             f"class, or jump children)")
            print(f"{repo.name}/{kind:<26s} {hw:>5d} {hw - len(missing):>9d} "
                  f"{len(missing):>7d} {len(w['gaps']):>4d} "
                  f"{len(w['resets']):>6d} {w['duals']:>5d}  "
                  f"{'; '.join(notes) if notes else ''}")
    print(f"\n{rows} counter(s) audited · {gaps_total} gap(s) · "
          f"{resets_total} reset(s)")
    if gaps_total == 0 and resets_total == 0:
        print("✓ no counter ever jumped or reset — `n <= hw ⇒ allocated` "
              "holds for every numbered kind; the Issue 768 witness is "
              "SOUND up to the COUNT-BASED exclusions")
    else:
        print("✗ counter jumps/resets found — the witness would over-claim "
              "there; Issue 768 T4 must exclude those repo/kind pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
