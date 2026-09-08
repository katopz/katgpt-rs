#!/usr/bin/env python3
"""Workspace markdown link-integrity sweep v2 (doc-sync housekeeping unit).

v2 changes: audit only git-TRACKED .md files (kills output//vendored noise);
drop the per-repo containment check (cross-repo links within the workspace are
legitimate — the real filesystem is the only truth); report [missing] only.
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

WORKSPACE = Path(os.environ.get("LINKCHECK_WORKSPACE", "/Users/katopz/git"))  # cross-box: set to the local workspace root
SKIP_PATH_PARTS = ("/.agents/",)  # skill files carry intentional example links
NAMESPACE_MARKERS = (
    ".issues/", ".plans/", ".benchmarks/", ".research/",
    ".docs/", ".proposals/", ".proofs/", ".distill/",
)
LINK_RE = re.compile(r"\]\((<[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\)")


def repo_set():
    return [
        d for d in sorted(WORKSPACE.iterdir())
        if d.is_dir() and (d / "BOUNDARY.md").exists() and (d / ".git").is_dir()
    ]


def tracked_md(repo: Path):
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--", "*.md"],
        capture_output=True, text=True, check=True,
    )
    for rel in out.stdout.splitlines():
        rel = rel.strip()
        if not rel or any(p in "/" + rel for p in SKIP_PATH_PARTS):
            continue
        yield repo / rel


def audit_target(raw: str, md_file: Path, repo: Path):
    t = raw[1:-1] if raw.startswith("<") and raw.endswith(">") else raw
    t = unquote(t.strip())
    if not t or t.startswith("#"):
        return None
    parsed = urlparse(t)
    if parsed.scheme in ("http", "https", "mailto", "ftp"):
        return None
    path_part = parsed.path
    if not path_part:
        return None
    is_md = path_part.endswith(".md") or path_part.endswith(".md/")
    in_ns = any(m in path_part for m in NAMESPACE_MARKERS)
    if not (is_md or in_ns):
        return None
    candidate = Path(path_part)
    if candidate.is_absolute():
        # treat as workspace-root-relative
        resolved = WORKSPACE / candidate.relative_to(candidate.anchor)
    else:
        resolved = md_file.parent / candidate
        try:
            resolved = resolved.resolve(strict=False)
        except OSError:
            return path_part
        if not resolved.exists() and candidate.parts and candidate.parts[0] == "..":
            # Workspace convention: authors write `../sibling/...` from the
            # REPO ROOT's frame ("escape this repo into the workspace"), not
            # standard file-relative markdown. From a subdirectory file that
            # string resolves to <repo>/sibling/... (never exists) and produced
            # false positives by the dozen (10 of 11 in one game-sdk plan).
            # Second chance: resolve the SAME string against the repo root,
            # where `..` genuinely escapes into the workspace. Tier 1 (file-
            # relative) still wins, so in-repo links are never re-interpreted.
            resolved = (repo / candidate).resolve(strict=False)
    try:
        resolved = resolved.resolve(strict=False)
    except OSError:
        return path_part
    if not resolved.exists():
        return path_part
    return None


def main():
    findings = {}
    scanned = 0
    for repo in repo_set():
        for md in tracked_md(repo):
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            scanned += 1
            for lineno, line in enumerate(text.splitlines(), 1):
                for m in LINK_RE.finditer(line):
                    missing = audit_target(m.group(1), md, repo)
                    if missing:
                        findings.setdefault(repo.name, []).append(
                            (str(md.relative_to(repo)), lineno, missing)
                        )
    total = sum(len(v) for v in findings.values())
    print(f"scanned {scanned} tracked markdown files across {len(repo_set())} repos")
    for repo_name in sorted(findings):
        rows = findings[repo_name]
        print(f"\n== {repo_name}: {len(rows)}")
        for rel, lineno, shown in rows:
            print(f"  {rel}:{lineno} ({shown})")
    print(f"\nTOTAL: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
