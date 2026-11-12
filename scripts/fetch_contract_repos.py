#!/usr/bin/env python3
"""Refresh every contract repo's remote-tracking refs, once (Issue 850 T2).

## Why this is a SESSION instrument and not a sweep flag

`worktree_state.behind_origin()` compares HEAD to a **local** remote-tracking
ref, so every staleness verdict in the sweep family is only as fresh as the
last fetch. Issue 827 T5 and Issue 850 T1/T3 made that honest: a silent
reading resting on a ref older than `STALE_FETCH_HOURS` is now disclosed as
`UNVERIFIED UPSTREAM` rather than believed. Issue 850 T2 asks the next
question — should a sweep just FETCH? — and warns against answering it by
reflex.

**Measured 2026-09-19 over the 17 contract repos on this box: 250.2s
serial, 50.2s at 8-way parallelism, 0 failures.** Against that, a sweep in
this family costs 0.04-40s and the whole 32-check docs gate costs ~164s wall.
So a per-sweep fetch is **5-30x the cost of the thing it precedes**, and a
full family run would pay it ~19 times. That is the empirical half of the
answer, and it is decisive on its own.

The design half is separate and points the same way. A fetch WRITES
remote-tracking refs in repos other sessions own. It cannot break a build
(refs/remotes is neither HEAD nor the worktree), but it can change a
concurrently-running instrument's verdict mid-run — Issue 797's class with
the sweep as the perpetrator rather than the victim. It also makes a verdict
depend on the network, and this box has a recorded ssh-transport failure mode;
a sweep that cannot answer offline is a sweep that stops being run.

⛔ **And a `--fetch` flag on each sweep is not the repair either** — Issue 850
T2 says so in advance: a flag nobody passes is not a repair. Freshness is a
property of the BOX at a moment, not of any one sweep. So it is fetched ONCE,
here, and the sweeps stay observers that DISCLOSE.

## The defect this actually fixes, which is not the cost

The advisory's remedy text reads *"`git fetch` in the named repo before
trusting it"*, and it names repos by their **CONTRACT** spelling — correctly,
because `repo_alias`'s own rule is that machine-local alias content must never
reach stdout (run logs get pasted into tracked docs). On an aliased box those
names are directories that **DO NOT EXIST**: measured here, the advisory tells
the reader to fetch `mmorpg-editor` and `mmorpg-remake` (`mmorpg-remaster` retired 2026-11-12, sprint!019)
while the checkouts are `seal-game-editor`, `seal-remake` and
`seal-online-remaster`.

Both halves of that are right and the remedy is still unusable, which is why
the fix is a command that resolves the codec ITSELF rather than a path the
reader has to translate. That is the whole argument for this file existing.

## What it does NOT do

- It does not merge, rebase, checkout or touch a working tree. `git fetch`
  only, `origin` NAMED — riir-chain carries a second remote (`github`) that is
  stale by design, and a bare `git fetch` there would update a ref nothing
  reads.
- It does not decide whether a repo SHOULD be fetched. Every contract repo on
  the box is refreshed; a repo with no `origin` is reported, never guessed at.
- It is not a gate. Exit 1 on a fetch FAILURE (actionable — network, auth,
  a dead transport) and 0 otherwise, including when nothing moved.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import console_safe  # noqa: E402
import repo_alias  # noqa: E402
import skill_repo_set_gate  # noqa: E402

console_safe.apply()

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT.parent

#: Concurrency. Measured on this box: 8 workers took the 250.2s serial cost to
#: 50.2s wall. Beyond that the limit is the remote, not the box, and a higher
#: number buys a rate-limit rather than a speedup.
WORKERS = 8


def _git(path: Path, *args: str, timeout: float | None = None):
    """A git call whose pipes decode as UTF-8 (Issue 778) and that CANNOT hang.

    ⛔ The timeout is not defensive garnish. `arm_reach_gate` wedged on this
    box for twenty minutes against a `git` child that never returned, and its
    own watchdog could not reach it — a thread plus `interrupt_main` reaches a
    pure-Python loop, never a blocking C call. Anything in this workspace that
    spawns git in a loop needs the bound at the SPAWN, because that is the only
    layer that has one.
    """
    return subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout)


def head_of(path: Path, ref: str) -> str | None:
    r = _git(path, "rev-parse", "--quiet", "--verify", ref, timeout=30)
    out = (r.stdout or "").strip()
    return out or None


def upstream_ref(path: Path) -> str | None:
    """`origin/<branch>`, or `None` when this checkout tracks nothing.

    `None` is CANNOT TELL and is reported as its own row. Defaulting to
    `origin/main` would invent an answer for the five workspace repos that
    genuinely have no upstream — `worktree_state.behind_origin`'s own rule,
    which exists because guessing here produces a confident wrong verdict.
    """
    r = _git(path, "rev-parse", "--abbrev-ref", "--symbolic-full-name",
             "@{upstream}", timeout=30)
    return (r.stdout or "").strip() or None


def fetch_one(name: str, path: Path, timeout: float):
    """One repo. Returns a row; never raises, so one dead remote cannot take
    the whole census with it."""
    up = upstream_ref(path)
    before = head_of(path, up) if up else None
    t0 = time.time()
    try:
        r = _git(path, "fetch", "--quiet", "origin", timeout=timeout)
        rc, err = r.returncode, (r.stderr or "").strip()
    except subprocess.TimeoutExpired:
        rc, err = 124, f"TIMED OUT after {timeout:.0f}s"
    dt = time.time() - t0
    after = head_of(path, up) if up else None
    return {
        "name": name, "secs": dt, "rc": rc, "err": err[:160],
        "upstream": up, "moved": bool(up and before != after),
        "before": before, "after": after,
    }


def population(workspace: Path) -> list[tuple[str, Path]]:
    """CONTRACT names paired with the directories to OPEN.

    Delegated to `skill_repo_set_gate.derive_repos` rather than re-walked:
    `population_sync_gate` exists to catch two predicates disagreeing about
    this set, and a fresh walk here would be the eleventh. The `.disk()` hop
    is the reverse half of the alias codec — the derived names are CONTRACT
    spellings and a box may hold them under different directories (Issue 842).
    """
    out = []
    for n in sorted(skill_repo_set_gate.derive_repos(workspace)):
        p = workspace / repo_alias.disk(n)
        if (p / ".git").exists():
            out.append((n, p))
    return out


def report(rows: list[dict], wall: float) -> int:
    rows = sorted(rows, key=lambda r: -r["secs"])
    failed = [r for r in rows if r["rc"] != 0]
    moved = [r for r in rows if r["moved"]]
    noup = [r for r in rows if r["upstream"] is None]
    for r in rows:
        if r["rc"] != 0:
            mark, note = "✗", f"FAILED rc={r['rc']} {r['err']}"
        elif r["upstream"] is None:
            mark, note = "⚠", "NO UPSTREAM — nothing to compare against"
        elif r["moved"]:
            mark, note = "✓", f"moved — {r['upstream']} advanced"
        else:
            mark, note = "✓", "already current"
        print(f"  {mark} {r['name']:<24} {r['secs']:5.1f}s  {note}")
    print(f"\n{len(rows)} contract repo(s) · {wall:.1f}s wall at "
          f"{WORKERS}-way · {sum(r['secs'] for r in rows):.1f}s of work · "
          f"{len(moved)} moved · {len(noup)} without an upstream · "
          f"{len(failed)} failed")
    if failed:
        print("✗ fetch FAILED in " + ", ".join(r["name"] for r in failed)
              + " — every sweep's upstream verdict in those repos still "
                "rests on an unrefreshed ref, so `UNVERIFIED UPSTREAM` there "
                "is the honest reading and a red finding is NOT confirmed")
        return 1
    print("✓ every contract repo on this box has heard from its remote "
          "just now — the sweep family's silent upstream readings are "
          "supported for the next "
          f"{__import__('worktree_state').STALE_FETCH_HOURS:.0f}h. ⚠ This "
          "refreshes refs/remotes ONLY: no merge, no rebase, no working tree "
          "touched, and a repo still BEHIND is still behind")
    return 0


def selftest() -> list[str]:
    """Arms over the decisions, against a real git pair.

    Deliberately not over `main()`: this instrument's only interesting rules
    are the population hop, the upstream/`None` split and the moved/current
    split, and all three are reachable without a network.
    """
    import tempfile
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    def run(cwd, *a):
        subprocess.run(["git", "-C", str(cwd), *a], check=True,
                       capture_output=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        up = tmp / "up"
        up.mkdir()
        run(up, "init", "-q", "-b", "main")
        run(up, "config", "user.email", "t@t")
        run(up, "config", "user.name", "t")
        (up / "a.txt").write_text("one\n", encoding="utf-8")
        run(up, "add", "-A")
        run(up, "commit", "-qm", "base")

        dn = tmp / "dn"
        subprocess.run(["git", "clone", "-q", str(up), str(dn)], check=True,
                       capture_output=True)
        run(dn, "config", "user.email", "t@t")
        run(dn, "config", "user.name", "t")

        check(upstream_ref(dn) == "origin/main",
              f"a clone's upstream was not read: {upstream_ref(dn)!r}")

        # Nothing has moved: `moved` must be False, or every run reports the
        # whole workspace as advancing and the signal is worthless.
        row = fetch_one("dn", dn, 60.0)
        check(row["rc"] == 0, f"a clean fetch failed: {row}")
        check(row["moved"] is False,
              f"an unchanged upstream was reported as moved: {row}")

        # Now it has.
        (up / "b.txt").write_text("two\n", encoding="utf-8")
        run(up, "add", "-A")
        run(up, "commit", "-qm", "next")
        row = fetch_one("dn", dn, 60.0)
        check(row["moved"] is True,
              f"an ADVANCED upstream was reported as current: {row}")
        check(row["before"] != row["after"],
              "the before/after refs did not differ on a real advance")

        # ⛔ A fetch must not move the working tree or HEAD. This is the whole
        # safety claim the docstring makes, so it is asserted rather than
        # stated: refs/remotes advanced above while HEAD must not have.
        head_now = head_of(dn, "HEAD")
        check(head_now == row["before"],
              "the fetch moved HEAD — it must touch refs/remotes ONLY")
        check(not (dn / "b.txt").exists(),
              "the fetch materialised an upstream file into the worktree")

        # No upstream: reported, never guessed at.
        solo = tmp / "solo"
        solo.mkdir()
        run(solo, "init", "-q", "-b", "main")
        run(solo, "config", "user.email", "t@t")
        run(solo, "config", "user.name", "t")
        (solo / "c.txt").write_text("three\n", encoding="utf-8")
        run(solo, "add", "-A")
        run(solo, "commit", "-qm", "solo")
        check(upstream_ref(solo) is None,
              "a repo with no remote was given an upstream")
        row = fetch_one("solo", solo, 60.0)
        check(row["upstream"] is None and row["moved"] is False,
              f"a no-upstream repo was not reported as such: {row}")

    # The population hop is the alias codec's reverse half: a CONTRACT name
    # must map to the directory this box actually holds, identity when
    # unmapped. Asserted against the live workspace because that is where the
    # mapping either exists or does not.
    pop = population(WORKSPACE)
    check(pop, "the derived population is EMPTY — the walk went blind")
    for n, p in pop:
        check(p.is_dir(),
              f"{n} resolved to {p.name}, which is not a directory — the "
              f"alias hop is what this row exists to catch")
    names = [n for n, _ in pop]
    check(len(names) == len(set(names)), "the population contains duplicates")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workspace", default=str(WORKSPACE))
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-repo seconds; a wedged git child is a measured "
                         "failure mode on this box, not a hypothetical")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        fails = selftest()
        for f in fails:
            print("  ✗ " + f)
        print(("✗ %d arm(s) FAILED" % len(fails)) if fails
              else "✓ fetch_contract_repos selftest — upstream read, "
                   "moved/current split, HEAD and worktree untouched by a "
                   "fetch, no-upstream reported not guessed, population hop "
                   "resolves through the alias codec")
        return 1 if fails else 0

    ws = Path(a.workspace).resolve()
    pop = population(ws)
    if not pop:
        print(f"✗ no contract repo found under {ws} — the walk went "
              "blind; nothing was fetched and nothing is claimed")
        return 1
    print(f"▸ fetching {len(pop)} contract repo(s) under {ws} at "
          f"{a.workers}-way, origin NAMED (a second remote is not refreshed)")
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(lambda np: fetch_one(np[0], np[1], a.timeout), pop))
    return report(rows, time.time() - t0)


if __name__ == "__main__":
    sys.exit(main())
