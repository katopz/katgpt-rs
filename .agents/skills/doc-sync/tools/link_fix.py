#!/usr/bin/env python3
"""Fix workspace markdown link findings produced by linkcheck_sweep.py.

Usage: link_fix.py <repo-name> [findings-file]

Policy per finding:
  N/A : target starts with '~' or '/Users/' or contains '*' (glob) -> skip.
  R1  : target carries a workspace-repo segment (`..*/<repo>/rest`) and
        WORKSPACE/<repo>/<rest> exists -> repoint to the correct relative path.
  R2  : leading ../ ./ stripped; <repo>/<cleaned> exists -> repoint.
  R3  : otherwise de-link: [text](target) -> `text` (prose citation; the
        convention says a mention names a finding and stays valid).

Every rewrite is anchored to the exact (file, line, target). Idempotence and
verification come from re-running linkcheck_sweep.py afterwards.
"""
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote

_DEFAULT_WORKSPACE = Path(__file__).resolve().parents[5]
WORKSPACE = Path(os.environ.get("LINKCHECK_WORKSPACE", str(_DEFAULT_WORKSPACE)))  # cross-box: the default derives this script's own home (<workspace>/katgpt-rs/.agents/skills/doc-sync/tools/link_fix.py → parents[5]); LINKCHECK_WORKSPACE overrides for any layout where that walk is wrong (shared-mechanism repair with linkcheck_sweep.py, doc-sync unit #41/#46)
REPO = (WORKSPACE / sys.argv[1]).resolve()  # resolve() kills any .. segments — an unresolved path can fail exists() downstream
# Cross-box default (Windows repair, doc-sync unit 2026-10-26: the hardcoded
# /tmp/linkcheck_full.txt resolved to cwd-drive-root \\tmp on native Windows
# python and crashed FileNotFoundError before any finding was read).
# tempfile.gettempdir() honors TMPDIR/%TEMP% and matches the file
# linkcheck_sweep.py writes by default — the pair works with zero operator
# path handling; argv[2] still overrides for an explicit findings file.
_DEFAULT_FINDINGS = Path(tempfile.gettempdir()) / "linkcheck_full.txt"
FINDINGS = Path(sys.argv[2]) if len(sys.argv) > 2 else _DEFAULT_FINDINGS

REPO_NAMES = sorted(
    d.name for d in WORKSPACE.iterdir()
    if d.is_dir() and (d / "BOUNDARY.md").exists() and (d / ".git").is_dir()
)
REPO_ALT = "|".join(REPO_NAMES)

# Contract repo names for the absent-repo guard. Primary source: the tracked
# workspace registry scripts/repo_set.txt (derived from the gated
# workstation's on-disk walk, maintained by skill_repo_set_gate — 39 names vs
# this hand set's 16, which had already rotted: it predates every post-Sept
# repo). The hand set stays as (a) the fallback for layouts without the
# katgpt-rs checkout and (b) the legacy-spelling union ('riir-clippy' = the
# pre-rename name; an extra guard name can only cause a SKIP, never a wrong
# edit).
_HAND_REPOS = {"katgpt-rs", "riir-ai", "riir-chain", "riir-clippy", "riir-train",
               "riir-neuron-db", "riir-game-sdk", "riir-mmorpg-examples",
               "riir-dapps", "riir-viewbridge", "riir-auth", "katgpt-web",
               "riir-dao", "riir-deployer", "mmorpg-editor", "mmorpg-remake"}


def contract_repos():
    f = WORKSPACE / "katgpt-rs" / "scripts" / "repo_set.txt"
    try:
        rows = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
    except OSError:
        rows = []
    return (set(rows) | _HAND_REPOS) if rows else set(_HAND_REPOS)


CONTRACT_REPOS = contract_repos()


def parse_findings():
    rows, in_section = [], False
    want = f"== {REPO.name}:"
    for line in FINDINGS.read_text(encoding="utf-8").splitlines():
        if line.startswith("== "):
            in_section = line.startswith(want)
            continue
        if in_section and line.startswith("  "):
            m = re.match(r"  (.+?):(\d+) \((.+)\)$", line)
            if m:
                rows.append((m.group(1), int(m.group(2)), unquote(m.group(3))))
    seen, out = set(), []
    for r in rows:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def relpath_from(md_dir: Path, target_abs: Path, keep_trailing_slash: bool):
    rel = os.path.relpath(target_abs, md_dir).replace(os.sep, "/")
    if keep_trailing_slash and not rel.endswith("/"):
        rel += "/"
    return rel


def absent_repo_segment(target):
    """First path segment of a ../-escaped (workspace-frame) target, or None.

    Workspace convention: `../<repo>/...` escapes the repo into the
    workspace, so the FIRST segment after the escape run is the repo name.
    The pattern this replaces ran its capture after a greedy `(?:seg/)*`
    star, which swallowed the repo segment on every 3+-segment target
    (`../repo/.benchmarks/f.md` captured '.benchmarks') — the guard then
    checked a namespace dir against the repo set and never fired (the
    plan-079 riir-deployer mis-delink, 2026-10-27).
    """
    m = re.match(r"^(?:\.\./)+([A-Za-z0-9_.-]+)/", target)
    return m.group(1) if m else None


def is_unverifiable_absent_repo(target, workspace=None, contract=None):
    """A link into a workspace repo ABSENT on this box is UNVERIFIABLE, not
    broken (it may resolve fine where that repo is checked out): the caller
    skips it — never repoints, never delinks. `exists()` is box-relative."""
    ws = workspace if workspace is not None else WORKSPACE
    known = contract if contract is not None else CONTRACT_REPOS
    seg = absent_repo_segment(target)
    return bool(seg) and seg in known and not (ws / seg).exists()


def _self_test():
    import shutil
    fails = []

    def check(name, cond):
        if not cond:
            fails.append(name)

    founding = "../riir-deployer/.benchmarks/005_t44a_devnet_wetrun.md"
    # A1 the founding regression (plan-079 mis-delink): the FIRST segment
    # after the escape run is the repo.
    check("A1-first-segment-capture",
          absent_repo_segment(founding) == "riir-deployer")
    # A2 premise arm: the OLD pattern provably captured the wrong segment —
    # the fix is load-bearing, and this reds if anyone reintroduces the star.
    old = re.match(r"^(?:\.\./)+(?:[A-Za-z0-9_.-]+/)*([A-Za-z0-9_.-]+?)/",
                   founding)
    check("A2-old-pattern-blind-premise",
          old is not None and old.group(1) == ".benchmarks")
    # A3 multi-escape still lands on the repo.
    check("A3-two-escape", absent_repo_segment("../../katgpt-rs/.docs/x.md")
          == "katgpt-rs")
    # A4 non-workspace-frame targets name no repo.
    check("A4-in-repo-no-frame", absent_repo_segment(".issues/179_x.md") is None)
    check("A5-bare-relative-no-frame", absent_repo_segment("docs/x.md") is None)
    check("A6-repo-dir-no-slash", absent_repo_segment("../riir-deployer") is None)

    # Two-sided guard behavior over a real temp workspace.
    tmp = Path(tempfile.mkdtemp(prefix="link_fix_selftest_"))
    try:
        (tmp / "present-repo").mkdir()
        contract = {"present-repo", "absent-repo"}
        check("B1-absent-in-contract-skips",
              is_unverifiable_absent_repo("../absent-repo/.benchmarks/f.md",
                                          workspace=tmp, contract=contract))
        check("B2-present-repo-repoints",
              not is_unverifiable_absent_repo("../present-repo/x.md",
                                              workspace=tmp, contract=contract))
        check("B3-unknown-name-delinks",
              not is_unverifiable_absent_repo("../not-a-repo/x.md",
                                              workspace=tmp, contract=contract))
        check("B4-in-repo-delinks",
              not is_unverifiable_absent_repo(".issues/f.md",
                                              workspace=tmp, contract=contract))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # C1/C2 the loaded contract set carries the fallback and the founding
    # mis-fix repo on any box with the katgpt-rs checkout.
    check("C1-contract-superset", _HAND_REPOS <= CONTRACT_REPOS)
    check("C2-deployer-in-contract", "riir-deployer" in CONTRACT_REPOS)

    if fails:
        print(f"link_fix self-test FAIL: {', '.join(fails)}")
        return 1
    print("link_fix self-test PASS (12 arms)")
    return 0


def read_text_eol(path: Path) -> str:
    # newline="" keeps \r\n and lone \r intact — the plain read_text/write_text
    # pair translated them whole-file (measured: a CRLF .research doc came back
    # LF-normalized everywhere except the one fixed line). Line indexing still
    # matches linkcheck_sweep.py: both split on the same str.splitlines set.
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        return fh.read()


def write_text_eol(path: Path, text: str) -> None:
    # newline="" writes the string untranslated — no os.linesep substitution.
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def decide(target: str, md: Path):
    """Return (action, new_target_or_None)."""
    if target.startswith(("~", "/Users/")) or "*" in target:
        return "NA", None
    if is_unverifiable_absent_repo(target):
        return "NA", None
    md_dir = md.parent
    m_repo = re.match(r"^(?:\.\./)+(" + REPO_ALT + r")/(.+)$", target)
    if m_repo:
        cand = WORKSPACE / m_repo.group(1) / m_repo.group(2)
        if cand.exists():
            return "R1", relpath_from(md_dir, cand, target.endswith("/"))
    cleaned = target
    while cleaned.startswith("../"):
        cleaned = cleaned[3:]
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    # lstrip("./") would eat the dot of dot-dirs (.benchmarks/ -> benchmarks/)
    # and make repo-relative links into dot-dirs unfixable; strip only a
    # literal leading "./"
    if cleaned and (REPO / cleaned).exists():
        return "R2", relpath_from(md_dir, REPO / cleaned, target.endswith("/"))
    return "R3", None


def rewrite_line(line: str, target: str, action: str, new_target):
    esc = re.escape(target)
    pat = re.compile(r"\[([^\]]*)\]\((?:" + esc + r')(?:\s+"[^"]*")?\)')
    if not pat.search(line):
        return None
    if action == "R3":
        def sub(mm):
            text = mm.group(1).strip()
            if "`" in text:
                return text  # already code-marked; wrapping would nest backticks
            return f"`{text}`" if text else f"`{target}`"
        return pat.sub(sub, line)
    # repoint: plain occurrence + title-bearing occurrence
    new_line = line.replace(f"]({target})", f"]({new_target})")
    new_line = new_line.replace(f"]({target} ", f"]({new_target} ")
    return new_line


def main():
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    stats = {"R1": 0, "R2": 0, "R3": 0, "NA": 0, "nomatch": 0}
    per_file = {}
    for rel, lineno, target in parse_findings():
        md = REPO / rel
        if not md.exists():
            print(f"SKIP file-gone: {rel}")
            continue
        lines = read_text_eol(md).splitlines(keepends=True)
        if lineno - 1 >= len(lines):
            print(f"SKIP line-gone: {rel}:{lineno}")
            continue
        action, new_target = decide(target, md)
        if action == "NA":
            stats["NA"] += 1
            print(f"NA: {rel}:{lineno} ({target})")
            continue
        new_line = rewrite_line(lines[lineno - 1], target, action, new_target)
        if new_line is None or new_line == lines[lineno - 1]:
            stats["nomatch"] += 1
            print(f"NOMATCH: {rel}:{lineno} ({target})")
            continue
        lines[lineno - 1] = new_line
        write_text_eol(md, "".join(lines))
        stats[action] += 1
        per_file[rel] = per_file.get(rel, 0) + 1
    print(f"\n{REPO.name}: repoint(R1)={stats['R1']} repoint(R2)={stats['R2']} "
          f"delink(R3)={stats['R3']} na={stats['NA']} nomatch={stats['nomatch']}")
    for rel in sorted(per_file):
        print(f"  edited: {rel} ({per_file[rel]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
