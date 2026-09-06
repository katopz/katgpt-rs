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

# The citation DIALECTS, as data. Measured over riir-ai's 4,097 tracked text
# files (2026-09-05): the short form is not a curiosity — `R<NNN>` is 1,384
# sites against 3,186 `Research <NNN>`, i.e. **30% of all research citations**,
# and a long-form-only pattern is blind to every one of them. Found by
# accident, grepping for `R020` while checking a verdict by hand — the fourth
# instance in this workspace of a classifier that was narrow rather than wrong.
# `P<NNN>` is rarer (284 vs 21,655) and `B<NNN>` rarer still (223 vs 4,944),
# but they cost nothing to admit because the search is always for one specific
# number.
DIALECTS = {
    "Plan": ["Plan", "P"],
    "Research": ["Research", "R"],
    "Bench": ["Bench", "Benchmark", "B"],
    "Issue": ["Issue"],          # no measured short form
    "Proposal": ["Proposal"],    # ditto
}

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


def citation_re(kind: str, number: str) -> re.Pattern:
    """Every spelling of "<kind> <number>" this corpus actually uses.

    The bare-letter forms are anchored to a 3-digit zero-padded number
    (`R020`, never `R20`), which is how the corpus writes them and which keeps
    the pattern from matching a register name or a prefill length.
    """
    n = int(number)
    alts = []
    for prefix in DIALECTS.get(kind, [kind]):
        if len(prefix) == 1:
            alts.append(rf"{prefix}-?{n:03d}")
        else:
            alts.append(rf"{re.escape(prefix)}\s*#?\s*0*{n}")
    return re.compile(rf"\b(?:{'|'.join(alts)})\b")


# A lead this small over the decided sites is not a signal. Measured on
# riir-ai .plans/313: 45 vs 41 over 82 decided sites — a 4.9% lead that flipped
# direction between two runs of the SAME instrument (the first, narrow-dialect
# pass read 31 vs 15). Weight cannot arbitrate that pair, and pretending it can
# is how a coin flip gets recorded as a measurement.
TIE_FRACTION = 0.10


def first_seen(repo: Path, relpath: str) -> str:
    """The date this document's number was first ALLOCATED, following renames.

    The tiebreak when weight is a tie: whoever held the number first keeps it.
    That is the allocator's own semantics, and unlike weight it cannot come out
    even.
    """
    out = subprocess.run(
        ["git", "-C", str(repo), "log", "--follow", "--diff-filter=A",
         "--format=%ad", "--date=short", "--", relpath],
        capture_output=True, text=True).stdout.split()
    return out[-1] if out else "unknown"


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


def attribute(blobs, cands, tokens, kind, number, window, margin, path_affinity=False):
    by_name = {c: 0 for c in cands}
    won = {c: 0 for c in cands}
    unresolved, sites = 0, 0
    path_hits = {c: 0 for c in cands}
    detail = {c: [] for c in cands}
    undecided: list[str] = []
    pat = citation_re(kind, number)

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
            # PATH affinity, reported apart from context score. Measured on
            # riir-ai .plans/313, the hardest pair resolved by hand: the citing
            # file's own PATH was the strongest signal in the corpus
            # (`swir_validation/*.rs` and `bench_313_swir_*.rs` on one side,
            # `cognitive_branches_runtime/step_attribution_bridge.rs` on the
            # other), and no amount of context-token widening reached it. Kept
            # separate and additive rather than folded in, so a corpus where
            # paths do NOT discriminate (riir-train: every document is "lora
            # training") degrades to the context score instead of inventing a
            # verdict out of directory names.
            path_l = rel.lower()
            # Separator-insensitive too: measured on riir-ai .plans/229, a
            # plain substring test scored the day/night plan at ZERO because
            # every citing site spells it `day/night` and `riir-gm-tool` while
            # the stem is `daynight` and `gm_tool`. That produced a confident
            # "nothing cites it, safe to rename" on a file with real inbound
            # citations — the exact false green this report exists to avoid.
            flat = re.sub(r"[^a-z0-9]", "", ctx)
            score = {c: sum(1 for t in tokens[c] if t in ctx or t in flat)
                     for c in cands}
            if path_affinity:
                pflat = re.sub(r"[^a-z0-9]", "", path_l)
                for c in cands:
                    hits = sum(1 for t in tokens[c] if t in path_l or t in pflat)
                    score[c] += hits
                    path_hits[c] += hits
            ranked = sorted(score.items(), key=lambda kv: -kv[1])
            top, second = ranked[0], (ranked[1] if len(ranked) > 1 else (None, 0))
            if top[1] >= 1 and top[1] - second[1] >= margin:
                won[top[0]] += 1
                detail[top[0]].append(f"{rel}:{blob[:m.start()].count(chr(10)) + 1}")
            else:
                unresolved += 1
                undecided.append(f"{rel}:{blob[:m.start()].count(chr(10)) + 1}")
    return by_name, won, unresolved, sites, detail, undecided, path_hits


def selftest() -> list[str]:
    """Pin the dialect table and the stem tokenizer.

    Both degrade SILENTLY: a dialect that stops matching just shrinks the site
    count, and the report still prints a confident verdict over what is left.
    That is how the `R<NNN>` form — 30% of this corpus's research citations —
    was missed on the first cut.
    """
    fails = []
    rx = citation_re("Research", "020")
    for good in ("see Research 20", "Research 020", "Research #020", "(R020, P163)", "R-020 "):
        if not rx.search(good):
            fails.append(f"dialect: missed {good!r}")
    for bad in ("Research 200", "R0201", "R20", "xR020"):
        if rx.search(bad):
            fails.append(f"dialect: matched {bad!r}")
    rp = citation_re("Plan", "163")
    if not rp.search("(R020, P163)") or rp.search("P1630"):
        fails.append("dialect: Plan short form wrong")
    # `Plan 20` must NOT match `Plan 200` -- the \b after 0*N is load-bearing
    if citation_re("Plan", "20").search("Plan 200"):
        fails.append("dialect: number boundary lost")
    if stem_tokens("313_swir_real_model_validation") != {"swir", "model", "validation"}:
        fails.append(f"stem tokens: {stem_tokens('313_swir_real_model_validation')}")
    if "313" in stem_tokens("313_swir_real_model_validation"):
        fails.append("stem tokens: kept the ambiguous number")
    return fails


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
    fails = selftest()
    if fails:
        print("✗ citation_weight SELFTEST FAILED — instrument untrustworthy:")
        for f in fails:
            print(f"    {f}")
        return 2

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
    ap.add_argument("--path-affinity", action="store_true",
                    help="also score the CITING FILE'S PATH against each stem")
    ap.add_argument("--show-unresolved", action="store_true",
                    help="print every site the margin could not decide")
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
    by_name, won, unresolved, sites, detail, undecided, path_hits = attribute(
        blobs, cands, tokens, kind, a.number, a.window, a.margin, a.path_affinity)

    print(f"{repo.name}{a.dirname}/{a.number} — {len(blobs)} tracked text files, "
          f"{sites} ambiguous `{kind} {int(a.number)}` site(s)")
    for c in cands:
        print(f"  {c}")
        print(f"      tokens    {sorted(tokens[c])}")
        print(f"      by-name   {by_name[c]}")
        if a.path_affinity:
            print(f"      path-hits {path_hits[c]}  (folded into the score below)")
        print(f"      attributed {won[c]}")
        for s in detail[c][:a.show]:
            print(f"          {s}")
    print(f"  UNRESOLVED  {unresolved}  ({unresolved / sites:.0%} of sites)" if sites
          else "  UNRESOLVED  0")
    # The UNRESOLVED bucket is where an audit's findings hide, so it must be
    # readable from the report itself -- otherwise every use ends in a manual
    # grep that re-derives the population by hand and gets it slightly wrong.
    for u in (undecided if a.show_unresolved else undecided[:0]):
        print(f"      {u}")

    for c in cands:
        if won[c] == 0 and by_name[c] == 0 and unresolved:
            print(f"  ⚠ {c} scored ZERO — that is NOT evidence that nothing cites it. "
                  f"{unresolved} site(s) are UNRESOLVED and any of them may be its. "
                  f"Measured case: riir-ai .plans/229's day/night plan scores 0 while "
                  f"`.docs/01_orientation/overview.md` and `.proposals/007` both cite it.")

    ranked = sorted(cands, key=lambda c: -(won[c] + by_name[c]))
    lead = (won[ranked[0]] + by_name[ranked[0]]) - (won[ranked[1]] + by_name[ranked[1]])
    decided = sites - unresolved
    if decided and 0 < lead < decided * TIE_FRACTION:
        dates = {c: first_seen(repo, f"{a.dirname}/{c}.md") for c in cands}
        first = min(cands, key=lambda c: dates[c])
        print(f"  → VERDICT: TIE by weight — lead {lead} is under {TIE_FRACTION:.0%} "
              f"of {decided} decided site(s). Falling back to CREATION ORDER, the "
              f"allocator's own semantics:")
        for c in cands:
            print(f"        {dates[c]}  {c}")
        print(f"    {first} held the number first, so it KEEPS it; the other moves.")
    elif decided == 0 or lead <= 0:
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
