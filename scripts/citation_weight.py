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
        capture_output=True, encoding="utf-8", errors="replace").stdout.split()
    return out[-1] if out else "unknown"


def corpus(repo: Path) -> dict[str, str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-files"],
                         capture_output=True, encoding="utf-8", errors="replace").stdout.split("\n")
    blobs = {}
    for rel in out:
        if not rel.endswith(TEXT_SUFFIXES):
            continue
        p = repo / rel
        if p.is_file():
            blobs[rel] = p.read_text(encoding="utf-8", errors="replace")
    return blobs


def removed_candidates(repo: Path, dirname: str, number: str) -> list[str]:
    """Stems that HELD this number and were removed — the shape `candidates()`
    cannot see, and the commonest collision shape there is (Issue 791 T1).

    A document closed under the noise-reduction rule is deleted, so a
    double-allocation where one side has closed leaves exactly ONE file on
    disk and reads as "not a duplicate". That is the majority case, not an
    edge: every number this repo allocates is expected to end up removed.

    `-M` is not decoration. Without rename detection a RENUMBERED document
    reports as a deletion at its old number, and the tool would resurrect a
    collision somebody already resolved — manufacturing the finding it exists
    to adjudicate. With it, a rename is an `R` and `--diff-filter=D` skips it.

    ⛔ It does NOT recover a file created and removed with no commit in
    between; git never saw it. That is Issue 754's blind spot, and the
    instrument that covers it is `numbering_drift_sweep.heading_allocated`,
    reading HISTORY.md headings. Named here because a reader who finds one
    candidate and no removal must not conclude there was never a second.
    """
    return sorted(removed_by_number(repo, dirname).get(int(number), ()))


def removed_by_number(repo: Path, dirname: str) -> dict[int, set[str]]:
    """`number -> {stem}` for every REMOVED numbered document in one directory.

    One `git log` for the whole directory. `removed_candidates` is the
    single-number view of this and DELEGATES rather than re-deriving it: a
    caller asking about 51 numbers otherwise pays 51 subprocesses for one
    answer, and two copies of the parse are two things to get wrong (Issue 755).

    ⛔ `--full-history` is load-bearing too (Issue 881 T2.4). A pathspec'd
    `git log` SIMPLIFIES history: at a merge whose result is TREESAME to one
    parent for `<dir>/`, it follows that parent only, so a document added and
    deleted on a merged feature branch is never seen. Measured 2026-09-24:
    mmorpg-editor (feature-branch heavy) carried two such collisions —
    `.plans/191` and `.issues/161`, each two unrelated documents allocated on
    parallel branches — and the whole workspace read 193 where it held 195.
    """
    out = subprocess.run(
        ["git", "-C", str(repo), "log", "--full-history", "-M",
         "--diff-filter=D", "--name-only",
         "--format=", "--", f"{dirname}/"],
        capture_output=True, encoding="utf-8", errors="replace").stdout
    by_num: dict[int, set[str]] = {}
    for rel in out.split("\n"):
        name = os.path.basename(rel.strip())
        if not name.endswith(".md"):
            continue
        m = re.match(r"^(\d+)_", name)
        if m:
            by_num.setdefault(int(m.group(1)), set()).add(name[:-3])
    return by_num


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
    fails += attribute_arms()
    # Called from HERE and not from main(): `arm_reach_audit` invokes an arm
    # only by the names in its vocabulary (`selftest`, `canary`, ...), so a
    # helper wired into main() is measured as reaching nothing. Measured on
    # the commit that added it — four `removed_candidates` decisions read
    # SURVIVED with the arms already written and passing.
    fails += removed_candidate_arms()
    return fails


def attribute_arms() -> list[str]:
    """Pin `attribute()` — the SCORING rule, which had no arm at all.

    ⛔ Found by `arm_reach_audit` (Issue 790 T6): this module scored **3 killed
    of 41**, the weakest arm reach in the workspace, and 16 of its 23 live
    survivors were in this one function. The arm above covers the dialect table
    and the tokenizer — the two inputs — and asserted nothing whatever about
    the decision they feed. That is the shape T2 named: a plain function over
    plain data, armable with no fixture repo, wearing an "I/O shell" label
    because the module around it shells out to git.

    Every rule below is one this file's own prose already CLAIMS, which is the
    point: an undertested claim and an untrue one read identically from here.
    """
    f: list[str] = []
    cands = ["229_daynight_gm_tool", "229_shader_cache_warm"]
    tokens = {c: stem_tokens(c) for c in cands}

    def run(blobs, margin=1, path_affinity=False, window=240):
        return attribute(blobs, cands, tokens, "Plan", "229", window, margin,
                         path_affinity)

    # ── 1. a clear winner is AWARDED, and the loser scores nothing ─────────
    by_name, won, unres, sites, detail, undecided, _ph = run(
        {"a.md": "the shader cache warm path, see Plan 229"})
    if (sites, won["229_shader_cache_warm"], unres) != (1, 1, 0):
        f.append(f"attribute: a clear winner was not awarded "
                 f"(sites={sites}, won={won}, unresolved={unres})")
    if won["229_daynight_gm_tool"]:
        f.append("attribute: the loser was awarded a site")
    if detail["229_shader_cache_warm"] != ["a.md:1"]:
        f.append(f"attribute: detail line is not 1-BASED: "
                 f"{detail['229_shader_cache_warm']}")

    # ── 2. a TIE is UNRESOLVED, never a winner. The rule this tool exists
    #      for: "awarded only on a strict margin, everything else printed as
    #      its own UNRESOLVED number and never folded into a winner."
    _b, won, unres, sites, _d, undecided, _ph = run(
        {"a.md": "shader cache daynight tool both, Plan 229"})
    if (unres, sum(won.values())) != (1, 0):
        f.append(f"attribute: a TIE was resolved (won={won}, unres={unres})")
    if undecided != ["a.md:1"]:
        f.append(f"attribute: undecided row not recorded 1-based: {undecided}")

    # ── 3. a top score of ZERO is never a winner, margin or not ────────────
    _b, won, unres, sites, *_ = run({"a.md": "nothing distinctive here, Plan 229"})
    if (sites, unres, sum(won.values())) != (1, 1, 0):
        f.append(f"attribute: a zero-score top won (won={won}, unres={unres})")

    # ── 4. MARGIN is honoured: lead 1 wins at margin 1 and not at margin 2 ──
    blob = {"a.md": "shader daynight tool Plan 229"}
    _b, w1, u1, *_ = run(blob, margin=1)
    _b, w2, u2, *_ = run(blob, margin=2)
    if sum(w1.values()) != 1 or u1 != 0:
        f.append(f"attribute: margin 1 did not award a 1-token lead ({w1})")
    if sum(w2.values()) != 0 or u2 != 1:
        f.append(f"attribute: margin 2 awarded a 1-token lead ({w2})")

    # ── 5. SEPARATOR-INSENSITIVE, the `.plans/229` measurement this module
    #      documents: sites spell it `day/night` and `gm-tool`, the stem is
    #      `daynight` and `gm_tool`. A plain substring test scored it ZERO and
    #      printed "nothing cites it, safe to rename".
    _b, won, unres, *_ = run({"a.md": "the day/night gm-tool, Plan 229"})
    if won["229_daynight_gm_tool"] != 1:
        f.append("attribute: separator-insensitive matching lost — the exact "
                 "false green this module was written to avoid")

    # ── 6. SELF-reference is not inbound, in BOTH counters ─────────────────
    by_name, won, _u, sites, *_ = run(
        {"229_shader_cache_warm.md": "shader cache warm, Plan 229"})
    if sites or sum(won.values()):
        f.append(f"attribute: a doc citing ITSELF counted as a site ({sites})")
    if by_name["229_shader_cache_warm"]:
        f.append("attribute: a doc's own name counted as an inbound mention")

    # ── 7. PATH affinity is OFF by default, additive when on, and reported
    #      APART — a corpus whose paths do not discriminate must degrade to
    #      the context score, not invent a verdict out of directory names.
    blobs = {"shader/cache/warm/x.md": "Plan 229"}
    _b, won_off, unres_off, _s, _d, _u, ph_off = run(blobs)
    _b, won_on, unres_on, _s, _d, _u, ph_on = run(blobs, path_affinity=True)
    if (sum(won_off.values()), unres_off) != (0, 1) or any(ph_off.values()):
        f.append(f"attribute: path affinity leaked while OFF "
                 f"(won={won_off}, path_hits={ph_off})")
    if won_on["229_shader_cache_warm"] != 1 or ph_on["229_shader_cache_warm"] < 1:
        f.append(f"attribute: path affinity did not score while ON "
                 f"(won={won_on}, path_hits={ph_on})")

    # ── 8. every site is accounted for: won + unresolved == sites ──────────
    _b, won, unres, sites, *_ = run({
        "a.md": "shader cache warm Plan 229",
        "b.md": "Plan 229",
        "c.md": "daynight gm tool Plan 229 and Plan 229 again",
    })
    if sum(won.values()) + unres != sites:
        f.append(f"attribute: {sum(won.values())} won + {unres} unresolved "
                 f"!= {sites} sites — a site fell out of the accounting")
    if sites != 4:
        f.append(f"attribute: expected 4 citation sites, saw {sites}")

    # ── 9. the WINDOW bounds the context actually read ─────────────────────
    far = {"a.md": "shader cache warm" + " ." * 400 + " Plan 229"}
    _b, won_wide, _u, _s, *_ = run(far, window=2000)
    _b, won_narrow, unres_narrow, _s, *_ = run(far, window=10)
    if won_wide["229_shader_cache_warm"] != 1:
        f.append("attribute: a wide window did not reach the tokens")
    if sum(won_narrow.values()) or unres_narrow != 1:
        f.append(f"attribute: a narrow window still scored ({won_narrow}) — "
                 f"the window is not bounding what is read")
    return f


def removed_candidate_arms() -> list[str]:
    """`removed_candidates` against a REAL git tree (Issue 791 T1).

    A fixture repo and not a stub, because the whole rule is what `git log -M
    --diff-filter=D` reports: the recovery, the rename EXCLUSION, and the
    number match are all properties of that command's output, and a stub would
    assert this function's parsing against a transcript somebody wrote by hand.
    """
    import shutil
    import tempfile

    if not shutil.which("git"):
        return ["    removed_candidate_arms: UNSEEN \u2014 no `git` on PATH, so "
                "the history recovery was asserted by NOTHING"]

    fails: list[str] = []

    def eq(label, got, want):
        if got != want:
            fails.append(f"    {label}: got {got!r}, want {want!r}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        d = root / ".issues"
        d.mkdir(parents=True)

        def git(*a):
            return subprocess.run(["git", "-C", str(root), *a],
                                  capture_output=True, encoding="utf-8",
                                  errors="replace", check=True).stdout.strip()

        git("init", "-q", "-b", "main")
        git("config", "user.email", "arm@example.invalid")
        git("config", "user.name", "arm")

        def commit(msg):
            git("add", "-A")
            git("commit", "-q", "-m", msg)

        # 050: closed and REMOVED. This is the shape `candidates()` cannot see,
        # and it is the majority case — every number here ends up removed.
        (d / "050_closed_and_gone.md").write_text("x\n" * 40, encoding="utf-8")
        # 051: still on disk.
        (d / "051_still_here.md").write_text("y\n" * 40, encoding="utf-8")
        # 052: RENUMBERED to 060 below. A rename is not an allocation being
        # released and re-taken; resurrecting it would manufacture the finding
        # this tool exists to adjudicate.
        (d / "052_renumbered_away.md").write_text("z\n" * 40, encoding="utf-8")
        commit("one")
        (d / "050_closed_and_gone.md").unlink()
        (d / "052_renumbered_away.md").rename(d / "060_renumbered_away.md")
        commit("two")

        eq("a removed document is recovered from history",
           removed_candidates(root, ".issues", "050"), ["050_closed_and_gone"])
        eq("a document still on disk is not reported as removed",
           removed_candidates(root, ".issues", "051"), [])
        eq("a RENUMBERED document is not resurrected at its old number "
           "(`-M` is why)", removed_candidates(root, ".issues", "052"), [])
        eq("...and its NEW number has no removal either",
           removed_candidates(root, ".issues", "060"), [])
        # The number must match exactly, not by prefix: `05` and `0500` are
        # different allocations and a loose compare pools them.
        eq("a number is matched exactly, not by prefix",
           removed_candidates(root, ".issues", "5"), [])
        eq("an unallocated number recovers nothing",
           removed_candidates(root, ".issues", "099"), [])
        # A leading-zero spelling is the SAME number: the file is `050_` and a
        # caller writing `50` must not be told the collision does not exist.
        eq("50 and 050 are the same allocation",
           removed_candidates(root, ".issues", "50"), ["050_closed_and_gone"])
        # Only `.md`, and only `NNN_`-prefixed: a stray file in the directory
        # is not an allocation.
        (d / "050_note.txt").write_text("n\n", encoding="utf-8")
        (d / "notanumber.md").write_text("n\n", encoding="utf-8")
        commit("three")
        (d / "050_note.txt").unlink()
        (d / "notanumber.md").unlink()
        commit("four")
        eq("a removed non-.md at the same number is not a candidate",
           removed_candidates(root, ".issues", "050"), ["050_closed_and_gone"])

        # The LIVE reader, over the same fixture. It is the other half of the
        # union main() builds, and its three guards fail the same silent way:
        # by shrinking the candidate set until a duplicate reads as a single.
        eq("candidates sees the file on disk",
           candidates(root, ".issues", "051"), ["051_still_here"])
        eq("candidates does not see the removed one",
           candidates(root, ".issues", "050"), [])
        eq("candidates matches the number exactly, not by prefix",
           candidates(root, ".issues", "5"), [])
        eq("candidates reads 51 and 051 as one allocation",
           candidates(root, ".issues", "51"), ["051_still_here"])
        (d / "notanumber.md").write_text("n\n", encoding="utf-8")
        (d / "051_still_here.txt").write_text("n\n", encoding="utf-8")
        eq("candidates ignores an unnumbered .md and a numbered non-.md",
           candidates(root, ".issues", "051"), ["051_still_here"])

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
    ap.add_argument("--live-only", action="store_true",
                    help="only candidates on disk (the pre-Issue-791 behaviour)")
    a = ap.parse_args()

    repo = Path(a.repo).resolve()
    kind = a.kind or {".plans": "Plan", ".issues": "Issue", ".research": "Research",
                      ".proposals": "Proposal", ".benchmarks": "Bench"}.get(a.dirname, "Plan")
    live = candidates(repo, a.dirname, a.number)
    gone = [] if a.live_only else [c for c in removed_candidates(
        repo, a.dirname, a.number) if c not in live]
    cands = live + gone
    if len(cands) < 2:
        extra = ("" if a.live_only else
                 " (git history searched for removed ones too)")
        print(f"{repo.name}{a.dirname}/{a.number}: {len(cands)} candidate(s) — "
              f"not a duplicate{extra}")
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
    # ⚠ Reported apart, never folded, and never subtracted. A removed document
    # has no file of its own left, so every trace of it lives in HISTORY.md —
    # which means its whole score can come from its own obituary while a live
    # rival's comes from third-party use. Those are not the same evidence.
    # They are not deducted either: Issue 724 T2's rule is about the COST of
    # moving a number, and a HISTORY line is an edit that would have to move.
    # The reader adjudicates; the tool refuses to decide it silently.
    hist = {c: sum(1 for s_ in detail[c] if s_.startswith("HISTORY.md:"))
            for c in cands}

    print(f"{repo.name}{a.dirname}/{a.number} — {len(blobs)} tracked text files, "
          f"{sites} ambiguous `{kind} {int(a.number)}` site(s)")
    for c in cands:
        print(f"  {c}   [{'on disk' if c in live else 'REMOVED — recovered from git history'}]")
        print(f"      tokens    {sorted(tokens[c])}")
        print(f"      by-name   {by_name[c]}")
        if a.path_affinity:
            print(f"      path-hits {path_hits[c]}  (folded into the score below)")
        print(f"      attributed {won[c]}"
              + (f"   (of which HISTORY.md: {hist[c]})" if hist[c] else ""))
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
