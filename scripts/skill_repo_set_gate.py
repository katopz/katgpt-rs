#!/usr/bin/env python3
"""Gate: a SKILL.md command block must not type the repo set out by hand.

Issue 703. An instrument that needs "all the repos" writes the list into a
command. Repos gain contracts; the list does not. The instrument then keeps
reporting clean over a set that no longer matches the workspace — and a clean
result over a partial set is indistinguishable from a clean result over the
whole one. Four instruments had drifted this way before anybody looked.

WHAT IS CHECKED, and why it is this and not the obvious thing.

The obvious check — grep every skill for an "N repos" claim and compare it
against the derived count — false-positives on history. `doc-sync` and
`boundary-guard` keep run-log tables whose rows correctly say "all 7 repos"
about a run that did cover 7. Rewriting those would destroy the record, and a
gate that cries wolf on an accurate historical row is one somebody loosens.

So this greps for the MECHANICAL DEFECT instead: inside a fenced block, two or
more repos used as PATH components (`riir-ai/…`, `/Users/katopz/git/riir-chain/…`)
or a brace list (`git/{a,b}`). Prose naming a repo does not match, because prose
does not put a slash after it — which is exactly what keeps the corrected
substrate-first block (whose comment names `riir-armageddon` and `riir-dapps`
while its command derives the set) from firing.

WHAT IS NOT CHECKED — say it out loud rather than let a green read as total.
A fenced SCOPE TABLE that lists repo names without paths (`goat-audit` §Scope,
`feature-gate-audit` §Scope) is the same defect and is invisible here. Those are
prose-shaped; catching them is the false-positive-prone case Issue 703
deliberately declined. They need a human read.

ESCAPE HATCH. A block that is genuinely about specific repos marks itself:

    <!-- repo-set-ok: <reason> -->

on any line inside the block, or on the line immediately before its opening
fence. The marker is required to carry a reason so the next reader can judge it.

    scripts/skill_repo_set_gate.py

Exit 0 = every multi-repo command block derives its set or is marked.
Exit 1 = a hand-typed repo set, or the gate could not see any skills at all.
"""

from __future__ import annotations

import re
import sys
import os
from pathlib import Path

# The workspace root = this repo's parent, so the gate works both on the
# workstation (all 18 contract repos side by side) and in CI (a lone checkout).
# Overridable for testing. NOT hard-coded to /Users/katopz/git — a gate against
# hand-typed paths that hand-types its own path would be the joke it polices.
GIT_ROOT = Path(os.environ.get("WORKSPACE_ROOT",
                               Path(__file__).resolve().parents[2]))
SELF_REPO = Path(__file__).resolve().parents[1].name
# The detector's VOCABULARY (which names are repos) is separate from the
# POPULATION (which SKILL.md files exist). On the workstation both are derived.
# In CI only this checkout exists, so a derived vocabulary would be one name and
# the gate could never see `riir-ai/src riir-chain/src` — a green that means
# nothing, which is the exact defect this gate exists to catch, committed by the
# gate. So the vocabulary is a committed SNAPSHOT, and every workstation run
# re-derives the truth and FAILS if the snapshot disagrees. CI consumes it;
# the workstation keeps it honest.
SNAPSHOT = Path(__file__).resolve().parent / "repo_set.txt"
MARKER_RE = re.compile(r"repo-set-ok:\s*(\S.*?)\s*(?:-->|$)")


def derive_repos(root: Path) -> list[str]:
    """The contract repo set, derived. Never typed.

    `(d / ".git").is_dir()`, not `.exists()`: a `git worktree` has a `.git`
    FILE, and counting one duplicates every hit of the repo it shadows.
    """
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / "BOUNDARY.md").is_file() and (d / ".git").is_dir()
    )


def fence_run(line: str) -> int:
    """Length of the leading backtick run, 0 if the line is not a fence."""
    s = line.lstrip()
    n = len(s) - len(s.lstrip("`"))
    return n if n >= 3 else 0


def fenced_blocks(text: str):
    """(start_line, end_line, body_lines, preceding_line) per fenced block.

    NOT a naive toggle on every ``` line. A toggle silently MIS-PHASES on a
    file with an unterminated fence: from that point on it scans the complement
    — prose read as code, code read as prose — and reports clean either way.
    `rust-optimize/SKILL.md` had exactly that (an unclosed ```text at line 511),
    and it swallowed this gate's own first canary, which is how the bug was
    found. So: a closer must be a BARE run of at least the opener's length,
    per CommonMark, and an unterminated fence is surfaced (see `scan`) rather
    than dropped.
    """
    lines = text.splitlines()
    start = None
    open_run = 0
    buf: list[str] = []
    for i, ln in enumerate(lines, 1):
        run = fence_run(ln)
        if start is None:
            if run:
                start, open_run, buf = i, run, []
            continue
        # inside: only a bare fence of >= the opening run closes it
        if run >= open_run and not ln.strip().strip("`"):
            yield start, i, buf, (lines[start - 2] if start >= 2 else "")
            start = None
            continue
        buf.append(ln)
    if start is not None:
        yield start, -len(lines), buf, (lines[start - 2] if start >= 2 else "")


def load_vocabulary(derived: list[str]) -> tuple[list[str], str | None]:
    """(repo names to match on, error). Snapshot is truth in CI, audited here."""
    if not SNAPSHOT.is_file():
        return [], (f"{SNAPSHOT.name} is missing — the gate has no repo "
                    f"vocabulary. Regenerate on the workstation.")
    snap = [l.strip() for l in SNAPSHOT.read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    if len(derived) > 1 and sorted(snap) != sorted(derived):
        missing, extra = set(derived) - set(snap), set(snap) - set(derived)
        return [], (f"{SNAPSHOT.name} is stale vs the live workspace — "
                    + (f"missing {sorted(missing)} " if missing else "")
                    + (f"gone {sorted(extra)} " if extra else "")
                    + "— regenerate it (see the docstring) and commit.")
    return snap, None


def scan(path: Path, repos: list[str]) -> list[tuple[int, int, set[str], bool]]:
    alt = "|".join(map(re.escape, repos))
    # A repo used as a path component. The leading lookbehind stops
    # `seal-online-remaster-unity/` from matching a shorter repo name, and
    # stops `.../katgpt-rs/crates/riir-ai/` style nesting from double-counting.
    prefix = r"(?:\.\./|" + re.escape(str(GIT_ROOT)) + r"/)?"
    path_re = re.compile(r"(?<![\w./-])" + prefix + r"(" + alt + r")/")
    brace_re = re.compile(r"/\{([^}]*)\}")
    repo_set = set(repos)
    out = []
    for start, end, body, prev in fenced_blocks(path.read_text()):
        if MARKER_RE.search(prev) or any(MARKER_RE.search(l) for l in body):
            continue
        names: set[str] = set()
        brace = False
        for line in body:
            names |= set(path_re.findall(line))
            for inner in brace_re.findall(line):
                parts = {p.strip() for p in inner.split(",")} & repo_set
                if len(parts) >= 2:
                    brace = True
                    names |= parts
        if end < 0:
            # Unterminated: the gate could not tell where this block ends, so a
            # clean verdict over it is not a verdict. Report, never swallow.
            out.append((start, end, names, brace))
        elif len(names) >= 2 or brace:
            out.append((start, end, names, brace))
    return out


def main() -> int:
    # Prints carry glyphs the Windows locale codecs cannot encode (checked
    # 2026-09-06 on cp874: check/cross/middot/arrow FAIL, em-dash OK); keep the
    # locale encoding and degrade only the fatal chars to escapes -- the
    # staged_set_audit house pattern (utf-8 pinning would mojibake legacy consoles).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (embedded / detached); keep old behavior
    if not GIT_ROOT.is_dir():
        print(f"✗ {GIT_ROOT} is not a directory — cannot derive the repo set")
        return 1
    repos = derive_repos(GIT_ROOT)
    skills = sorted(
        p for r in repos for p in (GIT_ROOT / r / ".agents/skills").glob("*/SKILL.md")
    )
    # Liveness. A run that examined nothing must not read like a clean one,
    # and one that examined a SUBSET must say which subset (Issue 703 is
    # precisely the failure of instruments that do not).
    if SELF_REPO not in repos:
        print(f"✗ {SELF_REPO!r} is not in the derived set under {GIT_ROOT} — "
              f"the gate is not looking at the tree it lives in")
        return 1
    if not skills:
        print(f"✗ gate examined 0 SKILL.md across {len(repos)} repo(s); "
              f"refusing to report a pass")
        return 1
    covered = sorted({p.parents[3].name for p in skills})
    # On the workstation all 18 contract repos are present; in CI only this
    # checkout is. Both are legitimate — reporting WHICH is the point.
    scope = "full workspace" if len(repos) > 1 else f"{SELF_REPO} only (CI)"
    print(f"▸ {len(repos)} contract repo(s) under {GIT_ROOT} — {scope}")
    print(f"▸ {len(skills)} SKILL.md in {len(covered)}: {', '.join(covered)}")

    vocab, err = load_vocabulary(repos)
    if err:
        print(f"✗ {err}")
        return 1
    print(f"▸ vocabulary: {len(vocab)} repo names from {SNAPSHOT.name}"
          + (" (re-derived and verified against the live workspace)"
             if len(repos) > 1 else " (snapshot — no siblings to verify against)"))

    findings = [(p, f) for p in skills for f in scan(p, vocab)]
    if not findings:
        print("  note: fenced SCOPE TABLES (repo names without paths) are out "
              "of scope by design — see this file's docstring")
        # The population goes in the LAST line on purpose: docs_gate.sh prints
        # only `tail -1` of a passing check, so a verdict that does not carry
        # its own scope reaches CI as a bare "clean" — a partial run wearing a
        # whole one's clothes, which is the entire defect being gated.
        print(f"✓ no hand-typed repo set — {len(skills)} SKILL.md across "
              f"{len(covered)} repo(s) [{scope}], vs {len(vocab)} repo names")
        return 0

    for path, (start, end, names, brace) in findings:
        rel = path.relative_to(GIT_ROOT)
        if end < 0:
            print(f"  ✗ {rel}:{start} — fence opened and never closed "
                  f"(runs to EOF, line {-end}). The block cannot be scanned, "
                  f"and an unterminated fence mis-renders every line after it.")
            continue
        kind = "brace list" if brace else "path enumeration"
        print(f"  ✗ {rel}:{start}-{end} — {kind} names {len(names)} of "
              f"{len(vocab)} repos: {', '.join(sorted(names))}")
    print(f"✗ {len(findings)} block(s) failed. Derive the repo set instead of "
          "typing it, or mark the block `<!-- repo-set-ok: <reason> -->`.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
