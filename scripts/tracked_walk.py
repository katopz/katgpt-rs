#!/usr/bin/env python3
"""The canonical POPULATION walk for every workspace audit — ONE copy.

An audit's finding count is only as good as the set of files it walked, and
the set of files a repo OWNS is the set of files git tracks. Walking the
filesystem behind a hand-typed directory-name skip set is not the same thing,
and this workspace has now measured it going wrong in three independent ways
(Issue 777; the precedents are Issue 734 for `trap_exit_launder_audit.py` and
Issue 738 T3 for `platform_dead_code_audit.py` / `wasm32_surface_audit.py`):

1. **A gitignored NESTED REPOSITORY.** `mmorpg-remaster/mmorpg/` is
   `.gitignore`'d AND carries its own `.git`; the filesystem walk credited its
   1404 `.rs` files to mmorpg-remaster and produced a correctly-shaped
   percentile finding at an address where the repair cannot be made. That is
   worse than a false positive — it is a real defect filed against the wrong
   repo.
2. **Build artifacts under a directory the skip set does not NAME.**
   riir-train's untracked `.rs` are cargo OUT_DIR sources under
   `.runs/target-release/`, `.runs/target-cuda/`, `.runs/target-v2cpu/`,
   `.runs/target-bench/`. The skip set names `target`; none of those IS
   `target`. A name list cannot express "not ours".
3. **A floor fabricated by (1) and (2).** `percentile_drift_floors.txt` pinned
   riir-train `min_rs_files = 2500` against a repo with 1129 tracked files —
   unreachable by any walk of that repo, red on every box but the one and the
   hour that measured it.

So the rule, once, here:

- `git ls-files` where git can answer, under an explicit `.git` probe. The
  probe is load-bearing: `git -C <dir>` walks UP until it finds a repository,
  so a non-repo directory nested inside one would otherwise be listed with its
  PARENT's paths.
- A filesystem walk as the FALLBACK, not as an error — an extracted
  `git archive` tree (`--prove-fires`) and a synthetic self-test tree have no
  `.git` and are legitimate populations.
- `vendor/` excluded with the count RETURNED, never swallowed: an exclusion
  nobody can see is a population change nobody can audit (Issue 738 T3 —
  riir-ai's `wgpu-hal` fork supplied 5 rows nobody owns).
- Index entries whose file is GONE from disk are dropped. A population is
  files you can read; a deleted-but-still-indexed path is a crash in every
  caller's read loop, counted as coverage.

Self-test: `scripts/tracked_walk.py` (exit 1 on failure). Every arm is
two-sided where two sides exist — an arm whose perturbation reds nothing
certifies nothing, which is how `platform_dead_code_audit.py`'s first
`vendor/` arm passed while testing the branch it was not aimed at.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

# Issue 804: this instrument is documented as directly invokable, and its
# verdict glyphs (✓ ✗ ⛔ ⚠) kill it on a non-UTF-8 console — no verdict at
# all, findings unread. docs_gate.sh's PYTHONIOENCODING only covers runs
# that go through the wrapper.
import console_safe  # noqa: E402

console_safe.apply()

# Pruned during the FALLBACK walk only. Deliberately short: this list is not
# the mechanism, it is what remains when git cannot answer at all.
SKIP_DIRS = {"target", ".git", "node_modules", ".venv", "dist"}

VENDOR_PARTS = ("vendor/", "/vendor/")


def vendored_p(rel: str) -> bool:
    """A repo-relative path inside a vendored drop (either separator)."""
    rel = rel.replace("\\", "/")
    return rel.startswith(VENDOR_PARTS[0]) or VENDOR_PARTS[1] in rel


def tracked_files(root, pattern: str = "*.rs", exclude=vendored_p) -> tuple:
    """(files, excluded) — the population of `root`, as a sorted `Path` list.

    `pattern` is a git pathspec glob for the tracked branch and an `rglob`
    pattern for the fallback; keep it to a plain `*.ext` so the two agree.
    `exclude` takes a forward-slashed repo-relative path and its hits are
    COUNTED into the second return value rather than dropped in silence.
    """
    root = Path(root)
    rels: list = []
    repo = (root / ".git").exists()
    try:
        if not repo:
            raise OSError("not a repository root")
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", pattern],
                             capture_output=True, check=True)
        rels = [x for x in out.stdout.decode("utf-8", "replace").split("\0") if x]
        # An index entry whose blob is no longer on disk is not a population
        # member -- it is an open() failure in the caller, counted as a walk.
        rels = [r for r in rels if (root / r).is_file()]
    except (OSError, subprocess.CalledProcessError):
        rels = []
    # The fallback walk is ONLY for trees with no `.git` at all (its
    # docstring law). A repo that tracks NO matching files is a VALID empty
    # population — falling back there credits gitignored build artifacts to
    # the repo: measured 2026-10-09 (katgpt-rs Issue 928), riir-deployer
    # tracks zero `*.py` and the rglob pulled six files out of the
    # gitignored `.deploy/` staging tree into locale_io's population, three
    # of them flagged LOCALE-IO over a state no commit contains.
    if not rels and not repo:
        for p in root.rglob(pattern):
            parts = p.relative_to(root).parts
            if any(part in SKIP_DIRS for part in parts):
                continue
            if not p.is_file():
                continue
            rels.append("/".join(parts))
    keep = [r for r in rels if not exclude(r)]
    return sorted(root / r for r in keep), len(rels) - len(keep)


# ── self-test ───────────────────────────────────────────────────────────────

def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args],
                   check=True, capture_output=True)


def _write(path: Path, text: str = "fn main() {}\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _names(root: Path, files) -> set:
    return {str(f.relative_to(root)).replace("\\", "/") for f in files}


def _repo(tmp: Path, name: str) -> Path:
    r = tmp / name
    r.mkdir(parents=True)
    # No commit anywhere in this file: `ls-files` reads the INDEX, so `git add`
    # is enough and the arms never need a user.name/user.email on the box.
    _git(r, "init", "-q")
    return r


def selftest() -> list:
    fails: list = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # A — tracked-ness IS the discriminator, both directions.
        r = _repo(tmp, "a")
        _write(r / "a.rs")
        _write(r / "b.rs")
        _git(r, "add", "a.rs")
        got, exc = tracked_files(r)
        check(_names(r, got) == {"a.rs"} and exc == 0,
              f"A: untracked b.rs not excluded — {_names(r, got)}")
        _git(r, "add", "b.rs")
        got, _ = tracked_files(r)
        check(_names(r, got) == {"a.rs", "b.rs"},
              f"A': staging b.rs did not admit it — {_names(r, got)} "
              "(the arm passes vacuously if nothing is ever included)")

        # B — Issue 777 instance 1: a gitignored NESTED repository. The
        # fallback walk sees it; the tracked walk must not. Asserting both
        # is what makes this an arm rather than a restatement.
        r = _repo(tmp, "b")
        _write(r / ".gitignore", "/nested/\n")
        _write(r / "own.rs")
        _git(r, "add", "own.rs", ".gitignore")
        nested = r / "nested"
        nested.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(nested), "init", "-q"],
                       check=True, capture_output=True)
        _write(nested / "crates" / "bot" / "src" / "metrics.rs")
        got, _ = tracked_files(r)
        check(_names(r, got) == {"own.rs"},
              f"B: gitignored nested repo credited to the outer repo — {_names(r, got)}")
        check(len(list(r.rglob("*.rs"))) == 2,
              "B': the fixture does not reproduce the hazard — an rglob that "
              "sees only one file cannot prove the tracked walk excluded one")

        # C — Issue 777 instance 2: OUT_DIR sources under an alternate target
        # dir. The companion assertion is about the SKIP SET, not the walk:
        # a name list that does not contain the name cannot prune it.
        r = _repo(tmp, "c")
        _write(r / "src" / "lib.rs")
        _git(r, "add", "src/lib.rs")
        _write(r / ".runs" / "target-release" / "build" / "x" / "out" / "gen.rs")
        got, _ = tracked_files(r)
        check(_names(r, got) == {"src/lib.rs"},
              f"C: cargo OUT_DIR source counted as repo code — {_names(r, got)}")
        check("target-release" not in SKIP_DIRS and "target" in SKIP_DIRS,
              "C': SKIP_DIRS now names the alternate target dir — the arm below "
              "it is testing a hazard that no longer exists in this shape")

        # D — vendor/ excluded AND counted (Issue 738 T3).
        r = _repo(tmp, "d")
        _write(r / "src" / "lib.rs")
        _write(r / "vendor" / "wgpu-hal" / "src" / "lib.rs")
        _write(r / "crates" / "x" / "vendor" / "dep.rs")
        _git(r, "add", "-A")
        got, exc = tracked_files(r)
        check(_names(r, got) == {"src/lib.rs"} and exc == 2,
              f"D: vendor exclusion — kept {_names(r, got)}, excluded {exc} (want 2)")

        # E — no `.git` at all: the fallback runs, and SKIP_DIRS prunes it.
        r = tmp / "e"
        _write(r / "a.rs")
        _write(r / "target" / "debug" / "b.rs")
        got, _ = tracked_files(r)
        check(_names(r, got) == {"a.rs"},
              f"E: fallback walk over a non-repo tree — {_names(r, got)}")

        # F — the `.git` probe. `git -C sub ls-files` walks UP and would answer
        # with the PARENT's paths; a non-repo subdirectory must take the
        # fallback and report paths relative to ITSELF.
        r = _repo(tmp, "f")
        _write(r / "root.rs")
        _write(r / "sub" / "inner.rs")
        _git(r, "add", "-A")
        got, _ = tracked_files(r / "sub")
        check(_names(r / "sub", got) == {"inner.rs"},
              f"F: `git -C` walked up out of the requested directory — "
              f"{_names(r / 'sub', got)}")
        check(all(p.is_file() for p in got),
              "F': returned a path that does not exist — the parent's relative "
              "paths were joined onto the child root")

        # G — indexed but deleted from disk: not a population member.
        r = _repo(tmp, "g")
        _write(r / "a.rs")
        _write(r / "gone.rs")
        _git(r, "add", "-A")
        (r / "gone.rs").unlink()
        got, _ = tracked_files(r)
        check(_names(r, got) == {"a.rs"},
              f"G: deleted-but-indexed path returned — {_names(r, got)}")

        # H — the pattern parameter reaches both branches.
        r = _repo(tmp, "h")
        _write(r / "a.rs")
        _write(r / "a.lean", "theorem t : True := trivial\n")
        _git(r, "add", "-A")
        got, _ = tracked_files(r, "*.lean")
        check(_names(r, got) == {"a.lean"},
              f"H: pattern ignored on the tracked branch — {_names(r, got)}")
        r2 = tmp / "h2"
        _write(r2 / "a.rs")
        _write(r2 / "a.lean", "theorem t : True := trivial\n")
        got, _ = tracked_files(r2, "*.lean")
        check(_names(r2, got) == {"a.lean"},
              f"H': pattern ignored on the fallback branch — {_names(r2, got)}")

        # I — a repo that tracks NO matching files is a VALID empty
        # population, not a fallback trigger: the rglob would credit
        # gitignored artifacts to the repo (riir-deployer's `.deploy/`
        # staging tree, measured 2026-10-09 — katgpt-rs Issue 928).
        r = _repo(tmp, "i")
        _write(r / ".gitignore", "/stage/\n")
        _git(r, "add", ".gitignore")
        _write(r / "stage" / "artifact.rs")
        got, exc = tracked_files(r)
        check(_names(r, got) == set() and exc == 0,
              f"I: gitignored artifact credited to a repo that tracks no "
              f"matching files — {_names(r, got)}")
        check((r / "stage" / "artifact.rs").is_file(),
              "I': the fixture does not reproduce the hazard — the artifact "
              "must exist on disk for the fallback to have seen it")

    return fails


def main() -> int:
    fails = selftest()
    if fails:
        print("tracked_walk selftest FAILED:")
        for f in fails:
            print("  ✗ " + f)
        return 1
    print("✓ tracked_walk selftest — 9 arm(s), 13 assertion(s): tracked-vs-untracked "
          "(both directions), gitignored nested repo, alternate target dir + the "
          "skip-set companion, vendor exclusion + count, no-.git fallback, the "
          "`git -C` walk-up probe, deleted-but-indexed, pattern on both branches, "
          "empty-tracked-is-not-a-fallback-trigger")
    return 0


if __name__ == "__main__":
    sys.exit(main())
