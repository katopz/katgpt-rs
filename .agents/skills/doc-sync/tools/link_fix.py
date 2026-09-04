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
from pathlib import Path
from urllib.parse import unquote

WORKSPACE = Path("/Users/katopz/git")
REPO = WORKSPACE / sys.argv[1]
FINDINGS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/linkcheck_full.txt")

REPO_NAMES = sorted(
    d.name for d in WORKSPACE.iterdir()
    if d.is_dir() and (d / "BOUNDARY.md").exists() and (d / ".git").is_dir()
)
REPO_ALT = "|".join(REPO_NAMES)


def parse_findings():
    rows, in_section = [], False
    want = f"== {REPO.name}:"
    for line in FINDINGS.read_text().splitlines():
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


def decide(target: str, md: Path):
    """Return (action, new_target_or_None)."""
    if target.startswith(("~", "/Users/")) or "*" in target:
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
    cleaned = cleaned.lstrip("./")
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
            return f"`{text}`" if text else f"`{target}`"
        return pat.sub(sub, line)
    # repoint: plain occurrence + title-bearing occurrence
    new_line = line.replace(f"]({target})", f"]({new_target})")
    new_line = new_line.replace(f"]({target} ", f"]({new_target} ")
    return new_line


def main():
    stats = {"R1": 0, "R2": 0, "R3": 0, "NA": 0, "nomatch": 0}
    per_file = {}
    for rel, lineno, target in parse_findings():
        md = REPO / rel
        if not md.exists():
            print(f"SKIP file-gone: {rel}")
            continue
        lines = md.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
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
        md.write_text("".join(lines), encoding="utf-8")
        stats[action] += 1
        per_file[rel] = per_file.get(rel, 0) + 1
    print(f"\n{REPO.name}: repoint(R1)={stats['R1']} repoint(R2)={stats['R2']} "
          f"delink(R3)={stats['R3']} na={stats['NA']} nomatch={stats['nomatch']}")
    for rel in sorted(per_file):
        print(f"  edited: {rel} ({per_file[rel]})")


if __name__ == "__main__":
    main()
