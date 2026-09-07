#!/usr/bin/env python3
"""Which packages carry POSITIVE wasm32 code, and does any lane compile them?

The wasm32 audit family (Issue 737) closed six axes, all of them about *how* a
lane compiles what it names: the triple, the `target_feature` arm, the feature
set, the arm reached via a dependent, the separate workspace, `check`-vs-lints.
This is the seventh axis and it is one level up — **is what the lane names the
whole surface?**

Found in seal-remake (`.issues/010` T2): the root package carried a positive
`#[cfg(target_arch = "wasm32")]` block that no row built and, measured, none
*could* — and the block had been uncompilable since it was written, because it
called a function declared `#[cfg(not(target_arch = "wasm32"))]`.

⛔ The predicate is the POSITIVE cfg. `not(target_arch = "wasm32")` is an
ordinary native-only guard and means the file has NO browser code (riir-ai
`.issues/892` T4); counting it inflates every number here. Comment lines are
excluded too — prose explaining a cfg is not a cfg, and a doc block that
mentions one would otherwise make a file look like browser code.

A **report, not a gate** (exit 0). Two floors are printed rather than pinned:
a ceiling ("0 uncovered") is green over whatever the instrument can see, so
the walk size has to be readable underneath it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent

CFG_RE = re.compile(r'target_arch\s*=\s*"wasm32"')
# `git grep -E` is POSIX ERE: no `\s`, no `\b` (the documented workspace trap).
# A Python pattern handed to git matches NOTHING and the walk reports a
# confident zero — which is what the floor below exists to catch, and did.
CFG_ERE = r'target_arch *= *"wasm32"' 
NEG_RE = re.compile(r'not\s*\(\s*target_arch\s*=\s*"wasm32"')
COMMENT_RE = re.compile(r"^\s*(//|/\*|\*)")
PKG_NAME_RE = re.compile(r'^\s*name\s*=\s*"([^"]+)"', re.M)
CARGO_ROW_RE = re.compile(r"cargo\s+(?:\+\S+\s+)?(check|clippy|build|test)\b")
DASH_P_RE = re.compile(r"(?:-p|--package)[= ]+([A-Za-z0-9_-]+)")
# `-p "$pkg"` / `-p ${p}` — a lane whose package list is DERIVED at run time
# (katgpt-rs's layer 2b builds one, deliberately, so a new wasm32-bearing
# crate joins by existing). A static reader cannot enumerate it. Calling that
# UNCOVERED would manufacture a finding out of the better design.
# `--manifest-path "$unit/Cargo.toml"` selects a package just as `-p` does, and
# riir-dapps + riir-deployer both drive their wasm32 lanes that way — over a
# DERIVED unit list, which is the stronger design. Matching only `-p $VAR`
# read both as bare rows, mis-credited the repo ROOT package, and reported
# their real Worker crates as UNCOVERED. Two false findings from one missing
# alternative.
DASH_P_VAR_RE = re.compile(r"(?:-p|--package|--manifest-path)[= ]+[\"']?\$")
MANIFEST_PATH_RE = re.compile(r"--manifest-path[= ]+[\"']?([^\"'\s]+)")

# Where a compile lane can live. Tracked files only — a filesystem walk picks
# up vendored drops no repo owns (the trap `trap_exit_launder_audit` hit).
LANE_GLOBS = ["scripts/*", ".github/*", ".github/workflows/*", "*.sh"]


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )
    return r.stdout if r.returncode == 0 else ""


def derive_population() -> list[Path]:
    """Every sibling carrying a BOUNDARY.md contract. Derived, never typed.

    `.git` must be a DIRECTORY: a throwaway worktree's `.git` is a file, and
    counting one would double-count a repo already in the walk.
    """
    return sorted(
        (
            d
            for d in WORKSPACE.iterdir()
            if d.is_dir() and (d / "BOUNDARY.md").is_file() and (d / ".git").is_dir()
        ),
        key=lambda p: p.name,
    )


def package_of(repo: Path, rel: str) -> str | None:
    """Nearest ancestor Cargo.toml declaring a [package] name."""
    d = (repo / rel).parent
    while True:
        manifest = d / "Cargo.toml"
        if manifest.is_file():
            try:
                text = manifest.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            if "[package]" in text:
                after = text.split("[package]", 1)[1]
                m = PKG_NAME_RE.search(after)
                if m:
                    return m.group(1)
        if d == repo or repo not in d.parents:
            return None
        d = d.parent


def positive_packages(repo: Path) -> tuple[dict[str, int], int]:
    """(package -> positive cfg site count, files walked)."""
    files = [f for f in git(repo, "grep", "-lE", CFG_ERE, "--", "*.rs").splitlines() if f]
    hits: dict[str, int] = {}
    for rel in files:
        try:
            lines = (repo / rel).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        n = sum(
            1
            for ln in lines
            if CFG_RE.search(ln) and not COMMENT_RE.match(ln) and not NEG_RE.search(ln)
        )
        if n:
            pkg = package_of(repo, rel) or "(unattributed)"
            hits[pkg] = hits.get(pkg, 0) + n
    return hits, len(files)


def logical_lines(text: str) -> list[str]:
    """Join shell/YAML backslash continuations.

    Without this a lane written as `cargo clippy \\` / `--target wasm32-...`
    is invisible: the triple and the verb land on different physical lines.
    That defect was in the first draft of this audit and it read two repos as
    having zero lanes minutes after their lanes were landed.
    """
    out, buf = [], ""
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        out.append(buf + stripped)
        buf = ""
    if buf:
        out.append(buf)
    return out


def lane_coverage(repo: Path) -> tuple[set[str], int, int, int, int]:
    """(literally-named pkgs, rows, --workspace rows, derived rows, bare rows)."""
    named: set[str] = set()
    rows = wildcards = derived = 0
    root_only = 0
    seen: set[str] = set()
    for glob in LANE_GLOBS:
        for rel in git(repo, "ls-files", "--", glob).splitlines():
            if not rel or rel in seen:
                continue
            seen.add(rel)
            try:
                text = (repo / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for ln in logical_lines(text):
                if "wasm32-unknown-unknown" not in ln or not CARGO_ROW_RE.search(ln):
                    continue
                # A quoted usage string / echo is documentation, not a lane.
                if re.match(r"\s*(#|//|echo\b|printf\b)", ln):
                    continue
                rows += 1
                if DASH_P_VAR_RE.search(ln):
                    derived += 1
                    continue
                mp = MANIFEST_PATH_RE.search(ln)
                if mp:
                    # A literal manifest path names exactly one package.
                    pkg = package_of(repo, mp.group(1))
                    if pkg:
                        named.add(pkg)
                    else:
                        derived += 1
                    continue
                pkgs = DASH_P_RE.findall(ln)
                if pkgs:
                    named.update(pkgs)
                elif "--workspace" in ln or "--all" in ln:
                    # Covers this workspace's members. Which packages those
                    # are is the separate-workspace axis — undecidable here.
                    wildcards += 1
                else:
                    # Neither `-p` nor `--workspace`: cargo compiles exactly
                    # the package rooted at the invocation directory. That IS
                    # decidable, so it credits the root package by name
                    # instead of parking it in UNRESOLVED.
                    root_only += 1
    return named, rows, wildcards, derived, root_only


def main() -> int:
    argv = sys.argv[1:]
    repos = [Path(a).resolve() for a in argv] if argv else derive_population()
    print(f"wasm32 SURFACE audit — {len(repos)} repo(s)\n")
    tot_pos_pkgs = tot_uncov = tot_unres = tot_files = 0
    findings: list[tuple[str, str, int]] = []
    unresolved: list[tuple[str, str, int]] = []
    for repo in repos:
        hits, files_walked = positive_packages(repo)
        tot_files += files_walked
        if not hits:
            if files_walked:
                print(f"  {repo.name}: {files_walked} file(s) mention wasm32, "
                      f"0 with POSITIVE cfgs (all native-only guards) — nothing to compile")
            continue
        named, rows, wildcards, derived, root_only = lane_coverage(repo)
        if root_only:
            root_pkg = package_of(repo, "Cargo.toml")
            if root_pkg:
                named.add(root_pkg)
        tot_pos_pkgs += len(hits)
        # Three buckets, and the middle one must not be folded into either
        # neighbour. A wildcard row (no `-p`) covers its own workspace, and
        # whether it reaches a given package is the separate-workspace axis
        # (737 #5) — undecidable here. A derived row enumerates its packages
        # at run time. Both are UNRESOLVED: a question to answer per package,
        # never a defect claim and never a pass.
        reachable = wildcards + derived
        print(f"  {repo.name}: {files_walked} file(s) walked · "
              f"{len(hits)} package(s) with positive cfgs · "
              f"{rows} wasm32 row(s) ({wildcards} --workspace, {derived} derived, "
              f"{root_only} root-only)")
        for pkg in sorted(hits):
            if pkg in named:
                mark = "✓ named"
            elif reachable:
                mark = "? UNRESOLVED"
                tot_unres += 1
                unresolved.append((repo.name, pkg, hits[pkg]))
            else:
                mark = "✗ UNCOVERED"
                tot_uncov += 1
                findings.append((repo.name, pkg, hits[pkg]))
            print(f"      {mark:<14} {pkg}  ({hits[pkg]} site(s))")
    print()
    print(f"  floors — {tot_files} file(s) walked over {len(repos)} repo(s); "
          f"{tot_pos_pkgs} package(s) with positive wasm32 cfgs")
    print(f"  {tot_pos_pkgs - tot_uncov - tot_unres} NAMED · "
          f"{tot_unres} UNRESOLVED · {tot_uncov} UNCOVERED")
    if findings:
        print("\n  ⛔ UNCOVERED — the repo has NO wasm32 row that could reach "
              "these at all:")
        for repo, pkg, n in findings:
            print(f"      {repo}: {pkg} ({n} positive site(s))")
    if unresolved:
        print("\n  ? UNRESOLVED — a wildcard or derived row exists; whether it "
              "reaches these\n    is the separate-workspace axis (737 #5), "
              "undecidable statically. NOT clean:")
        for repo, pkg, n in unresolved:
            print(f"      {repo}: {pkg} ({n} positive site(s))")
    print("\n  A report, not a gate — exit 0. Neither bucket is automatically a "
          "defect: a\n  package can be unbuildable for wasm32 by construction "
          "(seal-remake's authority\n  bin), where the repair is deleting the "
          "dead arm, not adding a row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
