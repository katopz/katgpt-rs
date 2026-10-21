#!/usr/bin/env python3
"""Which packages carry POSITIVE wasm32 code, and does any lane compile them?

The wasm32 audit family (Issue 737) closed six axes, all of them about *how* a
lane compiles what it names: the triple, the `target_feature` arm, the feature
set, the arm reached via a dependent, the separate workspace, `check`-vs-lints.
This is the seventh axis and it is one level up — **is what the lane names the
whole surface?**

Found in mmorpg-remake (`.issues/010` T2): the root package carried a positive
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

The eighth axis (Issue 774) is the DEPENDENCY EDGE: a row that selects a
package compiles that package's in-repo PATH DEPS for wasm32 too, so a
package with positive cfgs that no row names can still be covered — the
measured case was riir-shader's core+effects, built by every
`-p riir-shader-showcase --target wasm32` bundle build while the row-based
resolver read them UNCOVERED. The fix is a THIRD verdict, `✓ by-dep`:
reached from a named/derived package through non-optional in-repo path-dep
edges (plain `[dependencies]` plus target tables whose cfg positively names
`wasm32`; `workspace = true` entries resolve through the root table).
NEVER folded into NAMED — the distinction is load-bearing: a by-dep
package's lane dies by an innocent dep-graph edit in someone ELSE's
manifest, and the reader must know which kind of coverage they have. The
credit is reachability, not compile-verifiedness — but it is not vacuous
either: a broken wasm32 arm in a by-dep package breaks the named row's own
build the next time that row runs, which is exactly where the 010-class
catch would surface. Deliberately NOT credited: dev/build deps (host-side),
optional deps (feature off in the row), target tables gating to native, and
cross-repo path deps (a sibling's coverage is answered by the sibling's own
run). mmorpg-poc-submodule is the standing negative control: deliberately
excluded from its repo's CI and depended on by nothing, it stays UNCOVERED.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import NamedTuple

# Issue 804: this instrument is documented as directly invokable, and its
# verdict glyphs (✓ ✗ ⛔ ⚠) kill it on a non-UTF-8 console — no verdict at
# all, findings unread. docs_gate.sh's PYTHONIOENCODING only covers runs
# that go through the wrapper.
import console_safe  # noqa: E402
import repo_alias  # noqa: E402 — the machine-local name codec (see its docstring)

console_safe.apply()

WORKSPACE = Path(__file__).resolve().parent.parent.parent

CFG_RE = re.compile(r'target_arch\s*=\s*"wasm32"')
# `git grep -E` is POSIX ERE: no `\s`, no `\b` (the documented workspace trap).
# A Python pattern handed to git matches NOTHING and the walk reports a
# confident zero — which is what the floor below exists to catch, and did.
CFG_ERE = r'target_arch *= *"wasm32"' 
NEG_RE = re.compile(r'not\s*\(\s*target_arch\s*=\s*"wasm32"')
COMMENT_RE = re.compile(r"^\s*(//|/\*|\*)")
# Every wasm32 triple sets `target_arch = "wasm32"`, so a lane building ANY of
# them compiles the positive arms. The first cut matched only
# `wasm32-unknown-unknown` and read riir-reflexer's `wasm32-wasip1` worker
# build (run by its deploy workflow) as no lane at all — reflexer-wasm
# UNCOVERED, 2026-10-22.
WASM_TRIPLE_RE = re.compile(r"\bwasm32-[a-z0-9]+(?:-[a-z0-9]+)*\b")
# `cfg!(` — the RUNTIME branch, matched on the MASKED text (the token survives
# masking; the value string does not, so the original is read back for it).
CFG_MACRO_RE = re.compile(r"\bcfg!\s*\(")
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
        ["git", "-C", str(repo), *args], capture_output=True, encoding="utf-8", errors="replace"
    )
    return r.stdout if r.returncode == 0 else ""


def derive_population(root: Path | None = None) -> list[Path]:
    """Every sibling carrying a BOUNDARY.md contract. Derived, never typed.

    `.git` must be a DIRECTORY: a throwaway worktree's `.git` is a file, and
    counting one would double-count a repo already in the walk.

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


def positive_packages(repo: Path) -> tuple[dict[str, int], list[str], int, int, int]:
    """(package -> gated site count, positive files, files walked, cfg!
    branches, SHARED sites).

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
    runtime = 0
    shared = 0
    for rel in files:
        try:
            text = (repo / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "wasm32" not in text:
            continue
        n, rt, sh = positive_sites(text)
        runtime += rt
        shared += sh
        if n:
            pkg = package_of(repo, rel) or "(unattributed)"
            hits[pkg] = hits.get(pkg, 0) + n
            positive_files.append(rel)
    return hits, positive_files, len(files), runtime, shared


def top_level_any_disjuncts(raw: str) -> list[str] | None:
    """The top-level disjuncts of `cfg(any(...))`, or None when the predicate
    is not a top-level `any`. Paren-depth split on commas; string literals in
    a cfg are feature/target names and carry no parens or commas."""
    m = re.search(r"cfg\s*\(\s*any\s*\(", raw)
    if not m:
        return None
    depth, cur, out = 1, [], []
    for ch in raw[m.end():]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        if ch == "," and depth == 1:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur))
    return [d for d in (x.strip() for x in out) if d]


def wasm_exclusive(raw: str) -> bool:
    """Is this positive-wasm32 attribute wasm32-ONLY surface?

    `cfg(any(not(feature = "slice_leak"), target_arch = "wasm32"))` holds on
    every native default build — the arm is compiled by every native lane, so
    calling it UNCOVERED is the instrument mistaking a disjunct for the
    predicate (riir-reflex `harness/runner.rs`, 2026-10-22). A top-level `any`
    is wasm32-exclusive only when EVERY disjunct carries a positive wasm32
    term; a disjunct without one is natively satisfiable. Nested `any` inside
    an `all` stays exclusive when the `all` names wasm32 — only the top level
    is split, which can under-count SHARED and never invents one."""
    ds = top_level_any_disjuncts(raw)
    if ds is None:
        return True
    return all(CFG_RE.search(d) and not NEG_RE.search(d) for d in ds)


def positive_sites(text: str) -> tuple[int, int, int]:
    """-> (gated attribute sites, `cfg!` RUNTIME branches, SHARED sites).
    Never pooled. SHARED is a positive wasm32 term inside a top-level `any`
    with a natively satisfiable disjunct (see `wasm_exclusive`) — compiled
    by native lanes, so not surface; reported, never folded.

    ⛔ Reads ATTRIBUTES, not lines. A line scan cannot tell a real
    `#[cfg(target_arch = "wasm32")]` from one inside a raw string, and
    riir-clippy's `src/platform_audit.rs` is four such fixtures — Rust source
    embedded in `r#"…"#` as test INPUT for the platform-dead-code classifier.
    Measured 2026-09-15: those four were this audit's entire count for that
    repo, so it reported `1 package UNCOVERED, its arm compiles nowhere` about
    a repo with no wasm32 code at all, and hard-failed that row's sweep.

    Third instrument to meet this. `platform_dead_code_audit` masks literals
    and says why; `subprocess_encoding_gate` moved to an AST because its
    paren-matched text scanner *"reported four offenders in the gate's own
    file — every one a fixture string inside its selftest()."* The masker is
    IMPORTED rather than re-written: it is a hand-rolled Rust lexer with its
    own measured defect history (inline format args), and a second copy is a
    second thing to get wrong.

    ⚠ `cfg!(target_arch = "wasm32")` is split out and NOT counted as surface.
    It is a RUNTIME branch — it compiles on every target, so no lane can fail
    to reach it and it is not the thing this audit asks about. Seven sites
    workspace-wide (katgpt-rs 2, riir-ai 1, riir-shader 3, +1). Reported so the
    exclusion is re-measurable rather than remembered, never folded into the
    gated count.
    """
    from platform_dead_code_audit import mask_file

    try:
        masked, attrs = mask_file(text)
    except Exception:
        # A file the masker cannot read is NOT zero sites. Fall back to the
        # line scan, which over-reports on fixtures and never under-reports.
        lines = text.splitlines()
        return sum(1 for ln in lines if CFG_RE.search(ln)
                   and not COMMENT_RE.match(ln) and not NEG_RE.search(ln)), 0, 0
    pos = [a.raw for a in attrs
           if CFG_RE.search(a.raw) and not NEG_RE.search(a.raw)]
    gated = sum(1 for r in pos if wasm_exclusive(r))
    shared = len(pos) - gated
    # `cfg!` survives masking as the token; only its VALUE string is blanked,
    # so the original text is read back at that offset. A `cfg!` inside a
    # string literal is blanked WHOLE and cannot match here.
    rt = 0
    for m in CFG_MACRO_RE.finditer(masked):
        window = text[m.start():m.start() + 120]
        if CFG_RE.search(window) and not NEG_RE.search(window):
            rt += 1
    return gated, rt, shared


def positive_sites_arms() -> list[str]:
    """`positive_sites` — the string-literal immunity and the `cfg!` split.

    The measured case is riir-clippy's `src/platform_audit.rs`: four
    `#[cfg(target_arch = "wasm32")]` inside `r#"…"#` fixtures, which were that
    repo's ENTIRE count and made this audit report `1 package UNCOVERED, its
    arm compiles nowhere` about a repo with no wasm32 code at all.
    """
    fails: list[str] = []

    def eq(label, got, want):
        if got != want:
            fails.append(f"    {label}: got {got!r}, want {want!r}")

    real = '#[cfg(target_arch = "wasm32")]\nfn f() {}\n'
    eq("a real attribute is a gated site", positive_sites(real), (1, 0, 0))
    # The measured false positive. A raw string carrying Rust source is test
    # INPUT for another classifier, not a wasm32 arm.
    fixture = ('fn t() {\n    let src = r#"\n'
               '#[cfg(target_arch = "wasm32")]\nfn inner() {}\n"#;\n}\n')
    eq("⛔ an attribute inside a raw string is NOT a site",
       positive_sites(fixture), (0, 0, 0))
    eq("...and an ordinary string literal is not either",
       positive_sites('const S: &str = "#[cfg(target_arch = \\"wasm32\\")]";\n'),
       (0, 0, 0))
    eq("a real attribute in a file that also has a fixture still counts",
       positive_sites(real + fixture), (1, 0, 0))
    # `cfg!` is a RUNTIME branch: split out, never pooled into the gated count.
    eq("a cfg! branch is counted apart, not as surface",
       positive_sites('fn f() { if cfg!(target_arch = "wasm32") { g(); } }\n'),
       (0, 1, 0))
    eq("a cfg! branch inside a raw string is neither",
       positive_sites('fn t() { let s = r#"if cfg!(target_arch = "wasm32") {}"#; }\n'),
       (0, 0, 0))
    # The NEGATIVE cfg is an ordinary native-only guard; counting it inflates
    # everything (riir-ai .issues/892 T4).
    eq("a not(wasm32) guard is not a positive site",
       positive_sites('#[cfg(not(target_arch = "wasm32"))]\nfn f() {}\n'), (0, 0, 0))
    eq("a compound positive cfg still counts once",
       positive_sites('#[cfg(all(target_arch = "wasm32", '
                      'target_feature = "simd128"))]\nfn f() {}\n'), (1, 0, 0))
    # A multi-LINE attribute is one site, not one per line — the line scan
    # counted them per line and this is where the two disagree most.
    eq("a multi-line attribute is ONE site",
       positive_sites('#[cfg(all(\n    target_arch = "wasm32",\n'
                      '    target_feature = "simd128"\n))]\nfn f() {}\n'), (1, 0, 0))
    # A comment is not a cfg — the prose recording why a file has NO wasm32
    # arm otherwise makes that file read as browser code.
    eq("a commented attribute is not a site",
       positive_sites('// #[cfg(target_arch = "wasm32")]\nfn f() {}\n'), (0, 0, 0))
    eq("an inner attribute counts",
       positive_sites('#![cfg(target_arch = "wasm32")]\nfn f() {}\n'), (1, 0, 0))

    # SHARED (2026-10-22, riir-reflex): a natively satisfiable disjunct makes
    # the arm compiled by every native lane — not wasm32 surface.
    eq("⛔ any(not(feature), wasm32) is SHARED, not surface",
       positive_sites('#[cfg(any(not(feature = "x"), target_arch = "wasm32"))]\n'
                      'fn f() {}\n'), (0, 0, 1))
    eq("any(wasm32-only disjuncts) stays surface",
       positive_sites('#[cfg(any(all(target_arch = "wasm32", feature = "a"), '
                      'target_arch = "wasm32"))]\nfn f() {}\n'), (1, 0, 0))
    eq("all(wasm32, any(a, b)) stays surface — only the top level splits",
       positive_sites('#[cfg(all(target_arch = "wasm32", any(feature = "a", '
                      'feature = "b")))]\nfn f() {}\n'), (1, 0, 0))
    # Lane triple: every wasm32-* triple compiles the positive arm.
    for ln, want in (("cargo build -p x --target wasm32-wasip1", True),
                     ("cargo build -p x --target wasm32-unknown-unknown", True),
                     ("cargo build -p x --target wasm32-unknown-emscripten", True),
                     ("cargo build -p x --target x86_64-unknown-linux-gnu", False)):
        eq(f"lane triple {ln.split()[-1]}", bool(WASM_TRIPLE_RE.search(ln)), want)

    return fails


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
                if not WASM_TRIPLE_RE.search(ln) or not CARGO_ROW_RE.search(ln):
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


# ── Coverage-by-dep (Issue 774) ─────────────────────────────────────────────
# The row predicate above is DEP-BLIND: `-p riir-shader-showcase --target
# wasm32` compiles the showcase's in-repo path deps too, so riir-shader's
# core+effects read UNCOVERED while every bundle build compiles them
# (measured: `cargo check -p riir-shader-effects --target
# wasm32-unknown-unknown` exits 0, 2026-09-14). The graph below credits
# exactly those edges — as a THIRD verdict, never folded into NAMED: by-dep
# coverage dies by a dep-graph edit in someone else's manifest, and the
# reader must see which kind of coverage they hold.


def manifest_pkg_names(repo: Path) -> dict[str, str]:
    """Tracked Cargo.toml rel-paths -> [package] name (virtual manifests drop).

    Tracked only, vendored excluded — same population rules as the cfg walk.
    """
    out: dict[str, str] = {}
    for rel in git(repo, "ls-files", "--", "*.toml").splitlines():
        if not rel or Path(rel).name != "Cargo.toml":
            continue
        if rel.startswith(VENDOR_PARTS) or VENDOR_PARTS[1] in f"/{rel}":
            continue
        try:
            with (repo / rel).open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        pkg = (data.get("package") or {}).get("name")
        if isinstance(pkg, str):
            out[rel] = pkg
    return out


def credited_dep_tables(data: dict[str, object]) -> list[dict[str, object]]:
    """The dependency tables a wasm32 row compiles: plain [dependencies] plus
    [target.'cfg(...wasm32...)'.dependencies]. Native-target tables,
    dev-dependencies and build-dependencies are NOT credited — the first
    doesn't apply to the triple, the latter two build for the host."""
    tables: list[dict[str, object]] = []
    deps = data.get("dependencies")
    if isinstance(deps, dict):
        tables.append(deps)
    targets = data.get("target")
    if isinstance(targets, dict):
        for cfg, sub in targets.items():
            if not CFG_RE.search(cfg) or NEG_RE.search(cfg):
                continue
            if isinstance(sub, dict) and isinstance(sub.get("dependencies"), dict):
                tables.append(sub["dependencies"])
    return tables


def dep_graph(repo: Path, names: dict[str, str]) -> dict[str, set[str]]:
    """package -> in-repo packages it path-depends on for a wasm32 build.

    Non-optional path deps only. `workspace = true` entries resolve through
    the ROOT [workspace.dependencies] table. Cross-repo path targets are
    dropped: a sibling's coverage is answered by the sibling's own run, and
    crediting it here would make this repo's verdict depend on another
    repo's manifests. Edge targets are named by their OWN [package] table,
    so renames (`foo = { path = "../bar" }`) land on the real name."""
    root: dict[str, object] = {}
    try:
        with (repo / "Cargo.toml").open("rb") as fh:
            root = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        root = {}
    ws = root.get("workspace")
    ws_deps = ws.get("dependencies") if isinstance(ws, dict) else None
    ws_deps = ws_deps if isinstance(ws_deps, dict) else {}
    graph: dict[str, set[str]] = {}
    for rel, pkg in names.items():
        try:
            with (repo / rel).open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        edges: set[str] = set()
        manifest_dir = os.path.dirname(rel) or "."
        for table in credited_dep_tables(data):
            for key, spec in table.items():
                if not isinstance(spec, dict):
                    continue  # version-string registry dep
                if spec.get("optional") is True:
                    continue  # compiles only when the row enables its feature
                base = manifest_dir
                if spec.get("workspace") is True:
                    spec = ws_deps.get(key)
                    if not isinstance(spec, dict):
                        continue
                    # workspace-table paths are WORKSPACE-ROOT-relative,
                    # not member-manifest-relative
                    base = "."
                path = spec.get("path")
                if not isinstance(path, str) or not path:
                    continue
                target_dir = os.path.normpath(os.path.join(base, path))
                target_rel = os.path.join(target_dir, "Cargo.toml").replace(os.sep, "/")
                target = names.get(target_rel)
                if target and target != pkg:
                    edges.add(target)
        if edges:
            graph[pkg] = edges
    return graph


def by_dep_cover(
    graph: dict[str, set[str]], seeds: set[str], hits: dict[str, int]
) -> tuple[set[str], list[str]]:
    """(by-dep packages, chain notes). BFS from the seeds — everything a row
    provably compiles, literally named OR derived — through path-dep edges.
    A reached package carrying positive cfgs that no row selects itself is
    covered-by-dep. The note names the seed and the edge chain so the credit
    is auditable: `core: path-dep closure of showcase (showcase -> core)`."""
    parent: dict[str, str] = {}
    seen: set[str] = set(seeds)
    frontier = sorted(seeds)
    while frontier:
        cur = frontier.pop(0)
        for nxt in sorted(graph.get(cur, ())):
            if nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = cur
            frontier.append(nxt)
    bydep = {p for p in seen if p in hits and p not in seeds}
    notes: list[str] = []
    for pkg in sorted(bydep):
        chain = [pkg]
        cur = pkg
        while cur in parent:
            cur = parent[cur]
            chain.append(cur)
        chain.reverse()
        notes.append(f"{pkg}: path-dep closure of {chain[0]} ({' -> '.join(chain)})")
    return bydep, notes


def verdict_for(
    pkg: str, named: set[str], resolved: set[str], bydep: set[str], reachable: bool
) -> str:
    """The single classification, shared by main() and the --self-test
    canary — two copies would be the two-parsers-disagree trap the
    required-features family documents."""
    if pkg in named:
        return "\u2713 named"
    if pkg in resolved:
        return "\u2713 derived"
    if pkg in bydep:
        return "\u2713 by-dep"
    if reachable:
        return "? UNRESOLVED"
    return "\u2717 UNCOVERED"


class RepoSurface(NamedTuple):
    """One repo's wasm32 surface, bucketed — the verdict AND its population.

    Extracted from `main()` by Issue 785 so the workstation verdict half
    (`wasm32_surface_drift_sweep.py`) consumes the SAME 738/774 resolution
    rules the report does. A second copy of those rules would be the
    two-parsers-disagree trap the required-features family documents, and
    they are the whole instrument here: the derived-row upgrade (738 T1) and
    the path-dep closure (774) are what separate 25 NAMED from 17 false
    UNCOVERED.
    """

    files_walked: int
    hits: dict           # package -> positive-cfg site count
    rows: int
    wildcards: int
    derived: int
    root_only: int
    named: set
    resolved: set
    bydep: set
    res_notes: list
    dep_notes: list
    # `cfg!(target_arch = "wasm32")` sites, EXCLUDED from `hits` and
    # carried so the exclusion is re-measurable rather than remembered.
    # A runtime branch compiles on every target, so no lane can fail to
    # reach it and it is not the question this audit asks. Defaulted, so
    # the early-return construction below needs no change.
    runtime: int = 0
    # SHARED positive sites (a natively satisfiable `any` disjunct — see
    # `wasm_exclusive`), EXCLUDED from `hits` on the same terms as `runtime`.
    shared: int = 0

    @property
    def reachable(self) -> bool:
        """A `--workspace` or derived row exists — the precondition that makes
        an unclassified package UNRESOLVED rather than UNCOVERED."""
        return (self.wildcards + self.derived) > 0

    def verdicts(self) -> dict:
        """package -> verdict string, for every package with positive cfgs."""
        return {
            pkg: verdict_for(pkg, self.named, self.resolved, self.bydep,
                             self.reachable)
            for pkg in self.hits
        }

    def bucket(self, mark: str) -> list:
        return sorted(p for p, v in self.verdicts().items() if v == mark)


def classify_repo(repo: Path) -> RepoSurface:
    """Walk + classify one repo. No printing — the caller decides the shape."""
    hits, positive_files, files_walked, runtime, shared = positive_packages(repo)
    if not hits:
        return RepoSurface(files_walked, {}, 0, 0, 0, 0,
                           set(), set(), set(), [], [], runtime, shared)
    named, rows, wildcards, derived, root_only, row_files = lane_coverage(repo)
    row_texts = []
    for rel in row_files:
        try:
            row_texts.append((repo / rel).read_text(encoding="utf-8",
                                                    errors="replace"))
        except OSError:
            pass
    if root_only:
        root_pkg = package_of(repo, "Cargo.toml")
        if root_pkg:
            named.add(root_pkg)
    # Issue 738 T1: resolve the derived-row question per package, with the
    # evidence rule recorded per upgrade. UNRESOLVED stays UNRESOLVED for
    # anything no rule covers — the bucket remains the honest "a human has not
    # answered this yet", never folded into NAMED.
    resolved, res_notes = resolve_unresolved(repo, hits, positive_files,
                                             row_texts, named)
    # Issue 774: seeds are EVERYTHING a row provably compiles — the
    # literally-named set plus the derived upgrades — because a derived row's
    # `-p` list compiles its path-dep closure exactly like a literal one does.
    seeds = set(named) | set(resolved)
    graph = dep_graph(repo, manifest_pkg_names(repo))
    bydep, dep_notes = by_dep_cover(graph, seeds, hits)
    return RepoSurface(files_walked, hits, rows, wildcards, derived, root_only,
                       named, resolved, bydep, res_notes, dep_notes, runtime,
                       shared)


def self_test() -> int:
    """Prove the by-dep detectors fire, BOTH directions (Issue 774 landing),
    and the SITE classifier's string-literal immunity (`positive_sites_arms`).

    Five canary crates in a throwaway git repo:
      canary-a  named by the lane row        -> ✓ named
      canary-b  non-optional path dep of a   -> ✓ by-dep
      canary-c  depended on by nothing       -> ✗ UNCOVERED
      canary-d  optional path dep of a       -> ✗ UNCOVERED (not credited)
      canary-e  workspace=true dep of a      -> ✓ by-dep (root-table resolve)
    The c/d shape is the mmorpg-remake `.issues/010` lineage — the real catch
    this upgrade must NOT collapse. A verdict drift reds here instead of
    shipping a silent misclassification.
    """
    tmp = Path(tempfile.mkdtemp(prefix="wasm32_bydep_canary_"))
    try:
        (tmp / "scripts").mkdir(parents=True)
        (tmp / "Cargo.toml").write_text(
            "[workspace]\nmembers = [\n"
            + "".join(f'    "crates/canary-{c}",\n' for c in "abcde")
            + "]\n\n[workspace.dependencies]\n"
            + 'canary-e = { path = "crates/canary-e" }\n',
            encoding="utf-8",
        )
        a_deps = (
            "\n[dependencies]\n"
            'canary-b = { path = "../canary-b" }\n'
            'canary-d = { path = "../canary-d", optional = true }\n'
            "canary-e = { workspace = true }\n"
        )
        for c in "abcde":
            d = tmp / "crates" / f"canary-{c}"
            (d / "src").mkdir(parents=True)
            (d / "Cargo.toml").write_text(
                f'[package]\nname = "canary-{c}"\nversion = "0.0.0"\nedition = "2021"\n'
                + (a_deps if c == "a" else ""),
                encoding="utf-8",
            )
            (d / "src" / "lib.rs").write_text(
                '#[cfg(target_arch = "wasm32")]\npub fn w() {}\n', encoding="utf-8"
            )
        (tmp / "scripts" / "lane.sh").write_text(
            "cargo build -p canary-a --target wasm32-unknown-unknown\n",
            encoding="utf-8",
        )
        for args in (
            ("init", "-q"),
            ("add", "-A"),
            ("-c", "user.name=canary", "-c", "user.email=canary@x",
             "commit", "-qm", "canary"),
        ):
            subprocess.run(["git", "-C", str(tmp), *args], capture_output=True, encoding="utf-8", errors="replace")
        hits, positive_files, _, _, _ = positive_packages(tmp)
        named, _, wildcards, derived, root_only, row_files = lane_coverage(tmp)
        row_texts = []
        for rel in row_files:
            try:
                row_texts.append((tmp / rel).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
        if root_only:
            root_pkg = package_of(tmp, "Cargo.toml")
            if root_pkg:
                named.add(root_pkg)
        resolved, _ = resolve_unresolved(tmp, hits, positive_files, row_texts, named)
        graph = dep_graph(tmp, manifest_pkg_names(tmp))
        bydep, _ = by_dep_cover(graph, set(named) | set(resolved), hits)
        got = {
            p: verdict_for(p, named, resolved, bydep, (wildcards + derived) > 0)
            for p in hits
        }
        expect = {
            "canary-a": "\u2713 named",
            "canary-b": "\u2713 by-dep",
            "canary-c": "\u2717 UNCOVERED",
            "canary-d": "\u2717 UNCOVERED",
            "canary-e": "\u2713 by-dep",
        }
        ok = True
        for pkg in sorted(expect):
            want, actual = expect[pkg], got.get(pkg, "(missing)")
            good = want == actual
            ok = ok and good
            print(f"  {'ok' if good else 'FAIL':<4} {pkg}: want {want!r}, got {actual!r}")
        print(f"\n  {'PASS' if ok else 'FAIL'} — all five canary verdicts hold" if ok
              else "\n  FAIL — a verdict drifted")
        # The scanner arms are reached from HERE too, not only from main():
        # `arm_reach_audit` invokes an arm by NAME, and `self_test` is in its
        # vocabulary while `positive_sites_arms` is not — an arm it cannot
        # call is an arm measured as reaching nothing.
        scanner = positive_sites_arms()
        for f in scanner:
            print(f"  fail positive_sites{f}")
        return 0 if (ok and not scanner) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    # The scanner arms run on EVERY invocation, unlike the five-canary git
    # fixture below: they cost nothing (no subprocess, no tree) and they guard
    # the classifier that decides the population every other verdict is a
    # ceiling over. An instrument that miscounts its own corpus prints a
    # confident number either way.
    scanner = positive_sites_arms()
    if scanner:
        print("✗ positive_sites self-test FAILED — the site classifier "
              "does not behave as documented:")
        for f in scanner:
            print(f)
        return 2
    argv = sys.argv[1:]
    if argv == ["--self-test"]:
        print("wasm32 SURFACE audit — by-dep canary (Issue 774)\n")
        return self_test()
    repos = [Path(a).resolve() for a in argv] if argv else derive_population()
    print(f"wasm32 SURFACE audit — {len(repos)} repo(s)\n")
    tot_pos_pkgs = tot_uncov = tot_unres = tot_files = tot_bydep = 0
    findings: list[tuple[str, str, int]] = []
    unresolved: list[tuple[str, str, int]] = []
    for repo in repos:
        # ONE classifier, shared with the workstation verdict half (Issue 785).
        # The 738 derived-row upgrade and the 774 path-dep closure ARE the
        # instrument here — they are what separate 25 NAMED from 17 false
        # UNCOVERED — so the sweep imports them rather than restating them.
        s = classify_repo(repo)
        tot_files += s.files_walked
        if not s.hits:
            if s.files_walked:
                print(f"  {repo.name}: {s.files_walked} file(s) mention wasm32, "
                      f"0 with POSITIVE cfgs (all native-only guards) — nothing "
                      f"to compile"
                      + (f"; {s.runtime} `cfg!` runtime branch(es) EXCLUDED"
                         if s.runtime else "")
                      + (f"; {s.shared} SHARED any() site(s) EXCLUDED"
                         if s.shared else ""))
            continue
        tot_pos_pkgs += len(s.hits)
        print(f"  {repo.name}: {s.files_walked} file(s) walked · "
              f"{len(s.hits)} package(s) with positive cfgs · "
              f"{s.rows} wasm32 row(s) ({s.wildcards} --workspace, "
              f"{s.derived} derived, {s.root_only} root-only)"
              # Reported, never folded: a `cfg!` branch compiles on EVERY
              # target, so no lane can fail to reach it and it is not surface.
              + (f" · {s.runtime} `cfg!` runtime branch(es) EXCLUDED"
                 if s.runtime else "")
              + (f" · {s.shared} SHARED any() site(s) EXCLUDED"
                 if s.shared else ""))
        for pkg, mark in sorted(s.verdicts().items()):
            if mark == "? UNRESOLVED":
                tot_unres += 1
                unresolved.append((repo.name, pkg, s.hits[pkg]))
            elif mark == "✗ UNCOVERED":
                tot_uncov += 1
                findings.append((repo.name, pkg, s.hits[pkg]))
            elif mark == "✓ by-dep":
                tot_bydep += 1
            print(f"      {mark:<14} {pkg}  ({s.hits[pkg]} site(s))")
        # sorted: the resolver's note sets iterate in PYTHONHASHSEED order,
        # and an unsorted report is not diffable run-to-run (cosmetic-only
        # churn was masking real verdict flips in the landing diff).
        for note in sorted(s.res_notes):
            print(f"        · {note}")
        for note in sorted(s.dep_notes):
            print(f"        · {note}")
    print()
    print(f"  floors — {tot_files} file(s) walked over {len(repos)} repo(s); "
          f"{tot_pos_pkgs} package(s) with positive wasm32 cfgs")
    print(f"  {tot_pos_pkgs - tot_uncov - tot_unres - tot_bydep} NAMED · "
          f"{tot_bydep} BY-DEP · {tot_unres} UNRESOLVED · {tot_uncov} UNCOVERED")
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
          "(mmorpg-remake's authority\n  bin), where the repair is deleting the "
          "dead arm, not adding a row.\n")
    print("  Resolution (Issue 738 T1): a derived-row package upgrades to ✓ "
          "derived only\n  on static evidence from a row-bearing lane file — "
          "Shape A: the repo carries\n  the workspace's standard derivation "
          "formula (crates/([^/]*)/src/ -> -p <name>)\n  and the package has a "
          "site in that scope; Shape B: a membership pin names\n  a unit dir "
          "whose manifest IS the package. Anything else stays ? UNRESOLVED —\n  "
          "a human has not answered it, and that is the bucket's job.")
    print("\n  By-dep (Issue 774): a ✓ by-dep package is reached from a "
          "named/derived\n  package through non-optional in-repo path-dep "
          "edges (workspace-table deps\n  resolved; plain + wasm32-target "
          "tables). Reachability, not compile-proof —\n  but not vacuous "
          "either: a broken wasm32 arm there fails the named\n  row's own "
          "build. Optional, dev/build, native-target, and cross-repo\n  "
          "edges credit nothing. --self-test proves all five canary "
          "verdicts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
