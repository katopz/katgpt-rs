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

# Vendored upstream code is not this workspace's to gate (Issue 738 T3): the
# lanes themselves exclude `vendor/` (riir-ai's layer 1.22 derives its `-p`
# list with `grep -v '^vendor/'`), so a vendored fork's wasm32 code would show
# UNRESOLVED forever while the workspace's own answer — "not ours" — is
# already encoded in the lane. riir-ai's tracked `wgpu-hal-30.0.0` fork (11
# positive sites, Plan 536 / Issue 663) is the case that forced the question.
VENDOR_PARTS = ("vendor/", "/vendor/")


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


def positive_packages(repo: Path) -> tuple[dict[str, int], list[str], int]:
    """(package -> positive cfg site count, positive files, files walked).

    The walk size is ALL tracked wasm32-mentioning .rs (minus vendored), not
    just the positive ones: the floor's job is to catch a BLIND grep, and a
    grep that silently lost the native-only-guard half of the corpus would
    still show every positive file. Original-floor semantics (Issue 738's
    walk-size-next-to-verdict rule)."""
    files = [
        f
        for f in git(repo, "grep", "-lE", CFG_ERE, "--", "*.rs").splitlines()
        if f and not f.startswith(VENDOR_PARTS) and VENDOR_PARTS[1] not in f"/{f}"
    ]
    hits: dict[str, int] = {}
    positive_files: list[str] = []
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
            positive_files.append(rel)
    return hits, positive_files, len(files)


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


def lane_coverage(repo: Path) -> tuple[set[str], int, int, int, int, list[str]]:
    """(literally-named pkgs, rows, --workspace rows, derived rows, bare rows,
    row-bearing files). The last item feeds the resolver below: an upgrade
    may only cite a file that already carries a wasm32 cargo row, so a DEPLOY
    path mentioning a unit can never upgrade a package the audit family's
    founding lesson says is not gated (737: `build-*.sh` is not a gate)."""
    named: set[str] = set()
    rows = wildcards = derived = 0
    root_only = 0
    row_files: list[str] = []
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
            file_has_row = False
            for ln in logical_lines(text):
                if "wasm32-unknown-unknown" not in ln or not CARGO_ROW_RE.search(ln):
                    continue
                # A quoted usage string / echo is documentation, not a lane.
                if re.match(r"\s*(#|//|echo\b|printf\b)", ln):
                    continue
                rows += 1
                file_has_row = True
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
            if file_has_row:
                row_files.append(rel)
    return named, rows, wildcards, derived, root_only, row_files


# The derivation formula shared by every Shape-A lane in the workspace
# (katgpt-rs full_gate.sh layer 2b, riir-ai layer 1.22, riir-mmorpg-examples
# layer 2c): map wasm32-bearing files under `crates/<name>/src/` to `-p
# <name>`. Both sed spellings occur — BRE `\([^/]*\)` (katgpt-rs, riir-ai)
# and ERE `([^/]*)` (riir-mmorpg-examples' `sed -E`) — so the backslash is
# optional in the detection pattern. Detection is the literal pattern text,
# and the resolver RE-DERIVES the set with the same rule rather than trusting
# the script — a lane whose grep filter drifts stops matching its own
# formula's scope only if the scope text drifts too, and each repo's own
# membership pin is the second net.
DERIVED_SCOPE_RE = re.compile(r"crates/\\?\(\[\^/\]\*\\?\)/src/")
# A unit-dir token (Shape B): `cloudflare/<dir>` pinned literally in a
# row-bearing lane (riir-dapps L7, riir-deployer 5/5, riir-mmorpg-examples 2e).
UNIT_TOKEN_RE = re.compile(r"\b((?:cloudflare|wasm)/[A-Za-z0-9_-]+)")


def resolve_unresolved(
    repo: Path,
    hits: dict[str, int],
    positive_files: list[str],
    row_texts: list[str],
    named: set[str],
) -> tuple[set[str], list[str]]:
    """UNRESOLVED packages the lane provably selects, per Issue 738 T1.

    Two evidence rules, both requiring row-bearing files only (a DEPLOY path
    mentioning a unit can never upgrade a package — the audit family's
    founding lesson is that `build-*.sh` is not a gate):
    A. the repo's lane carries the workspace's standard derivation formula
       (`crates/([^/]*)/src/` -> `-p <name>`); any package with a positive
       site under that scope is selected by construction — the derivation
       enumerates exactly the packages the scope matches. Deliberately NOT
       extended to root-`src/` sites: katgpt-rs's derivation appends the root
       package, riir-ai's and riir-mmorpg-examples' do not, and the one repo
       where it matters already names its root via a bare row.
    B. a unit-dir token whose manifest names the package: the membership pin
       lists the unit literally, and the pin reds when the derived set
       changes, so the lane cannot silently stop selecting it.
    Packages already literally named by a row skip the resolver. Everything
    else stays UNRESOLVED — the bucket remains the honest "a human has not
    answered this yet", never folded into NAMED.
    Returns (upgraded packages, notes explaining each upgrade).
    """
    upgraded: set[str] = set()
    notes: list[str] = []
    shape_a = any(DERIVED_SCOPE_RE.search(t) for t in row_texts)
    all_text = "\n".join(row_texts)
    if shape_a:
        scope_pkgs = {
            package_of(repo, rel)
            for rel in positive_files
            if rel.startswith("crates/") and "/src/" in rel
        }
        scope_pkgs.discard(None)
        for pkg in scope_pkgs:
            if pkg in hits and pkg not in named:
                upgraded.add(pkg)
                notes.append(f"{pkg}: derived-scope site under crates/*/src/ (Shape A)")
    for pkg in hits:
        if pkg in named or pkg in upgraded:
            continue
        # B: unit-dir token whose manifest names this package. The manifest
        # must EXIST: package_of's walk probes repo/Cargo.toml as a fallback
        # (correct for site attribution), so a phantom `cloudflare/<x>` token
        # in a row file would otherwise resolve to the repo ROOT package and
        # upgrade it on nothing (caught by the canary, not by reading).
        for m in UNIT_TOKEN_RE.finditer(all_text):
            unit = m.group(1)
            unit_manifest = repo / f"{unit}/Cargo.toml"
            if not unit_manifest.is_file():
                continue
            unit_pkg = package_of(repo, f"{unit}/Cargo.toml")
            if unit_pkg == pkg:
                upgraded.add(pkg)
                notes.append(f"{pkg}: membership pin names {unit} (Shape B)")
                break
    return upgraded, notes


def main() -> int:
    argv = sys.argv[1:]
    repos = [Path(a).resolve() for a in argv] if argv else derive_population()
    print(f"wasm32 SURFACE audit — {len(repos)} repo(s)\n")
    tot_pos_pkgs = tot_uncov = tot_unres = tot_files = 0
    findings: list[tuple[str, str, int]] = []
    unresolved: list[tuple[str, str, int]] = []
    for repo in repos:
        hits, positive_files, files_walked = positive_packages(repo)
        tot_files += files_walked
        if not hits:
            if files_walked:
                print(f"  {repo.name}: {files_walked} file(s) mention wasm32, "
                      f"0 with POSITIVE cfgs (all native-only guards) — nothing to compile")
            continue
        named, rows, wildcards, derived, root_only, row_files = lane_coverage(repo)
        row_texts = []
        for rel in row_files:
            try:
                row_texts.append((repo / rel).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
        if root_only:
            root_pkg = package_of(repo, "Cargo.toml")
            if root_pkg:
                named.add(root_pkg)
        # Issue 738 T1: resolve the derived-row question per package, with
        # the evidence rule recorded per upgrade. UNRESOLVED stays UNRESOLVED
        # for anything no rule covers — the bucket remains the honest "a
        # human has not answered this yet", never folded into NAMED.
        resolved, res_notes = resolve_unresolved(
            repo, hits, positive_files, row_texts, named
        )
        tot_pos_pkgs += len(hits)
        reachable = wildcards + derived
        print(f"  {repo.name}: {files_walked} file(s) walked · "
              f"{len(hits)} package(s) with positive cfgs · "
              f"{rows} wasm32 row(s) ({wildcards} --workspace, {derived} derived, "
              f"{root_only} root-only)")
        for pkg in sorted(hits):
            if pkg in named:
                mark = "✓ named"
            elif pkg in resolved:
                mark = "✓ derived"
            elif reachable:
                mark = "? UNRESOLVED"
                tot_unres += 1
                unresolved.append((repo.name, pkg, hits[pkg]))
            else:
                mark = "✗ UNCOVERED"
                tot_uncov += 1
                findings.append((repo.name, pkg, hits[pkg]))
            print(f"      {mark:<14} {pkg}  ({hits[pkg]} site(s))")
        for note in res_notes:
            print(f"        · {note}")
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
          "dead arm, not adding a row.\n")
    print("  Resolution (Issue 738 T1): a derived-row package upgrades to ✓ "
          "derived only\n  on static evidence from a row-bearing lane file — "
          "Shape A: the repo carries\n  the workspace's standard derivation "
          "formula (crates/([^/]*)/src/ -> -p <name>)\n  and the package has a "
          "site in that scope; Shape B: a membership pin names\n  a unit dir "
          "whose manifest IS the package. Anything else stays ? UNRESOLVED —\n  "
          "a human has not answered it, and that is the bucket's job.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
