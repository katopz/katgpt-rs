#!/usr/bin/env python3
"""Issue 725 T4 — attribute the AMBIGUOUS citations of a duplicated number.

When two documents share a number, Issue 724 T2's rule is that the one with the
most inbound mentions KEEPS it and the other moves. Applying that rule needs a
count, and the obvious count is the wrong one: measured on riir-ai's six
duplicates (2026-09-05), the *by-name* citations — `175_lattice_calculus_latcal`
spelled out — are 0-2 per side and TIED in four of the six pairs, while the
`Plan 175` form carries 35-98 mentions each. **The weight is entirely in the
citations that do not say which document they mean.**

So attribute them, rather than counting them. Each ambiguous mention gets a
context window; the window is scored against the distinctive tokens of each
candidate's filename stem (and any extra tokens passed with `--tokens`), and a
mention is awarded only on a STRICT margin. Everything else lands in
UNRESOLVED, which is printed as its own number and never folded into a winner —
an audit's UNRESOLVED bucket is where its findings hide, and a tool that quietly
assigns them manufactures the verdict it was asked to measure.

    scripts/citation_weight.py <repo> <dir> <number> [--kind Plan] [--window 240]
    scripts/citation_weight.py ../riir-ai .plans 313

Read the verdict as ADVISORY. It says which sites are decidable without a human
and how lopsided the decidable ones are; it does not license a rename on its
own, because the losing document's citations must still be rewritten by hand and
a mis-attributed one silently re-points a reader. Exit is always 0 — this is a
report.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

STOP = {
    # structural words that appear in most stems and carry no discrimination
    "the", "and", "for", "with", "into", "from", "real", "new", "phase",
    "plan", "issue", "test", "gpu", "cpu", "t1", "t2", "t3",
}
TEXT_SUFFIXES = (".md", ".rs", ".toml", ".py")


def stem_tokens(stem: str) -> set[str]:
    """Distinctive tokens of a filename stem: `313_swir_real_model_validation`
    -> {swir, model, validation}. The leading number is dropped (it is the
    ambiguous part) and stop-words are removed."""
    parts = re.split(r"[_\-]+", stem.lower())
    return {p for p in parts[1:] if len(p) > 2 and p not in STOP}


def corpus(repo: Path) -> dict[str, str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-files"],
                         capture_output=True, text=True).stdout.split("\n")
    blobs = {}
    for rel in out:
        if not rel.endswith(TEXT_SUFFIXES):
            continue
        p = repo / rel
        if p.is_file():
            blobs[rel] = p.read_text(encoding="utf-8", errors="replace")
    return blobs


def candidates(repo: Path, dirname: str, number: str) -> list[str]:
    d = repo / dirname
    n = int(number)
    return sorted(f.stem for f in d.iterdir()
                  if f.suffix == ".md" and re.match(r"^(\d+)_", f.name)
                  and int(f.name.split("_", 1)[0]) == n)


def attribute(blobs, cands, tokens, kind, number, window, margin):
    by_name = {c: 0 for c in cands}
    won = {c: 0 for c in cands}
    unresolved, sites = 0, 0
    detail = {c: [] for c in cands}
    pat = re.compile(rf"\b{re.escape(kind)}\s*#?\s*0*{int(number)}\b")

    for rel, blob in blobs.items():
        base = os.path.basename(rel)
        for c in cands:
            if base.startswith(c):
                continue                       # a doc citing itself is not inbound
            by_name[c] += blob.count(c)
        for m in pat.finditer(blob):
            if any(base.startswith(c) for c in cands):
                continue                       # self-reference inside the pair
            sites += 1
            ctx = blob[max(0, m.start() - window): m.end() + window].lower()
            # Separator-insensitive too: measured on riir-ai .plans/229, a
            # plain substring test scored the day/night plan at ZERO because
            # every citing site spells it `day/night` and `riir-gm-tool` while
            # the stem is `daynight` and `gm_tool`. That produced a confident
            # "nothing cites it, safe to rename" on a file with real inbound
            # citations — the exact false green this report exists to avoid.
            flat = re.sub(r"[^a-z0-9]", "", ctx)
            score = {c: sum(1 for t in tokens[c] if t in ctx or t in flat)
                     for c in cands}
            ranked = sorted(score.items(), key=lambda kv: -kv[1])
            top, second = ranked[0], (ranked[1] if len(ranked) > 1 else (None, 0))
            if top[1] >= 1 and top[1] - second[1] >= margin:
                won[top[0]] += 1
                detail[top[0]].append(f"{rel}:{blob[:m.start()].count(chr(10)) + 1}")
            else:
                unresolved += 1
    return by_name, won, unresolved, sites, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("dirname")
    ap.add_argument("number")
    ap.add_argument("--kind", default=None, help="citation word (default: from dir)")
    ap.add_argument("--window", type=int, default=240)
    ap.add_argument("--margin", type=int, default=2,
                    help="token-score lead required to award a site (default 2)")
    ap.add_argument("--tokens", action="append", default=[],
                    metavar="STEM=tok,tok", help="extra discriminating tokens")
    ap.add_argument("--show", type=int, default=0, help="print N awarded sites each")
    a = ap.parse_args()

    repo = Path(a.repo).resolve()
    kind = a.kind or {".plans": "Plan", ".issues": "Issue", ".research": "Research",
                      ".proposals": "Proposal", ".benchmarks": "Bench"}.get(a.dirname, "Plan")
    cands = candidates(repo, a.dirname, a.number)
    if len(cands) < 2:
        print(f"{repo.name}{a.dirname}/{a.number}: {len(cands)} candidate(s) — not a duplicate")
        return 0

    tokens = {c: stem_tokens(c) for c in cands}
    for extra in a.tokens:
        stem, toks = extra.split("=", 1)
        for c in cands:
            if c.startswith(stem):
                tokens[c] |= {t.strip().lower() for t in toks.split(",") if t.strip()}

    blobs = corpus(repo)
    by_name, won, unresolved, sites, detail = attribute(
        blobs, cands, tokens, kind, a.number, a.window, a.margin)

    print(f"{repo.name}{a.dirname}/{a.number} — {len(blobs)} tracked text files, "
          f"{sites} ambiguous `{kind} {int(a.number)}` site(s)")
    for c in cands:
        print(f"  {c}")
        print(f"      tokens    {sorted(tokens[c])}")
        print(f"      by-name   {by_name[c]}")
        print(f"      attributed {won[c]}")
        for s in detail[c][:a.show]:
            print(f"          {s}")
    print(f"  UNRESOLVED  {unresolved}  ({unresolved / sites:.0%} of sites)" if sites
          else "  UNRESOLVED  0")

    for c in cands:
        if won[c] == 0 and by_name[c] == 0 and unresolved:
            print(f"  ⚠ {c} scored ZERO — that is NOT evidence that nothing cites it. "
                  f"{unresolved} site(s) are UNRESOLVED and any of them may be its. "
                  f"Measured case: riir-ai .plans/229's day/night plan scores 0 while "
                  f"`.docs/01_orientation/overview.md` and `.proposals/007` both cite it.")

    ranked = sorted(cands, key=lambda c: -(won[c] + by_name[c]))
    lead = (won[ranked[0]] + by_name[ranked[0]]) - (won[ranked[1]] + by_name[ranked[1]])
    decided = sites - unresolved
    if decided == 0 or lead <= 0:
        print("  → VERDICT: UNDECIDABLE mechanically — arbitrate by hand.")
    elif unresolved > decided:
        print(f"  → VERDICT: WEAK lead for {ranked[0]} (+{lead}), but UNRESOLVED "
              f"({unresolved}) outnumbers the decided sites ({decided}). Read before acting.")
    else:
        print(f"  → VERDICT: {ranked[0]} leads by {lead} over {decided} decided site(s). "
              f"Advisory — the loser's citations still need a hand rewrite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
