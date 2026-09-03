#!/usr/bin/env python3
"""Audit percentile-index sites: which reported "p99" is actually the MAX.

A REPORT, not a gate (always exit 0) — for the same reason
`cfg_gated_target_audit.py` is one: a large share of sites take their sample
count from a runtime value that no static pass can resolve, and a report that
exits 1 on those is a report nobody runs. The verdict half belongs in a
per-repo gate over the sites this one resolves.

# The defect

    let p99 = sorted[(n as f64 * 0.99) as usize];   // n = 100 -> sorted[99] -> MAX
    let p99 = sorted[n * 99 / 100];                 // n = 100 -> sorted[99] -> MAX

`floor(n*p) == n - 1` whenever `n * (1 - p) <= 1`, i.e. **n <= 1/(1-p)**:
n <= 100 for p99, n <= 20 for p95, n <= 1000 for p999. Below that boundary the
site reports the maximum under a percentile's name, and the number is ONE
observation. `.min(len - 1)` clamps prevent a panic, not a wrong statistic.

Direction matters: the naive index is one rank TOO HIGH, so for a
`p99 < budget` assert the failure mode is a false RED, not a false green.
Nothing green becomes red by fixing a site.

# Two vocabularies, and why both are committed here

The first cut of this audit (riir-mmorpg-examples Issue 093) grepped only the
FLOAT forms and reported a 14-row table as an audit "over all 19 contract
repos". The INTEGER form `n * 99 / 100` is the more common one in this
workspace and was invisible to it; riir-e2e's copy was found by accident, not
by the sweep. That is the katgpt-rs "a zero ceiling is only as wide as its
classifier" failure, committed inside the issue warning about it.

So the vocabulary is DATA, listed exhaustively below and pinned by
`selftest()`, and the population is DERIVED (BOUNDARY.md + a `.git` dir) — per
the workspace rule that deriving both from the same walk is what makes a
cross-repo report permanently empty.

# Tail support is the quantity nobody prints

`support = n - idx` — the number of samples at or above the reported rank:
1 at n=100, 2 at n=200, 10 at n=1000 (naive index). A quantile with support 1
is one observation; support 2 moves if one preemption lands above it. This
report calls anything under MIN_SUPPORT weak, whether or not it is degenerate.
"""

import os
import re
import sys

MIN_SUPPORT = 10  # samples at/above the rank before a tail can decide a verdict

# ── The vocabulary (DATA — extend here, then extend selftest) ─────────────
#
# Each entry: (name, compiled regex, group name for the n-expression,
#              a callable p-extractor over the match, safe_by_construction).
#
# `safe` marks a form that CANNOT return n-1 for any n >= 2 and so needs no
# sample count to clear it — the `(len - 1) * p` shape, which is bounded by
# n - 2. That form is the one site in the workspace that got it right and it
# must not be reported as a finding.
VOCAB = [
    # (v.len() as f64 - 1.0) * 0.99   |   (len - 1) * p / 100.0
    #
    # SAFE BY CONSTRUCTION: `floor((n-1) * p)` is bounded by n-2 for every
    # n >= 2 and every p < 1, so this form can never return the max. It needs
    # no sample count to clear it, and it must NOT be reported as a finding --
    # it is the shape the workspace's one correct site uses
    # (katgpt-speculative/tests/weaver_real_checkpoint.rs). Verified over
    # n in 2..=100_000: zero violations.
    ("float_len_minus_one",
     re.compile(r"\(\s*(?P<n>[A-Za-z_]\w*(?:\.len\(\))?)\s*(?:as\s+f(?:32|64)\s*)?"
                r"-\s*1(?:\.0)?\s*\)\s*\*\s*(?P<p>0\.9\d+|\w+\s*/\s*100\.0)"),
     True),
    # x as f64 * 0.99   |   (x.len() as f64) * 0.999
    ("float_mul",
     re.compile(r"\b(?P<n>[A-Za-z_]\w*(?:\.len\(\))?)\s+as\s+f(?:32|64)\s*\)?"
                r"\s*\*\s*(?P<p>0\.9\d+)"),
     False),
    # 0.99 * x as f64   (reversed operands)
    ("float_mul_rev",
     re.compile(r"(?P<p>0\.9\d+)\s*\*\s*(?P<n>[A-Za-z_]\w*(?:\.len\(\))?)\s+as\s+f(?:32|64)"),
     False),
    # n * 99 / 100   |   (xs.len() * 999) / 1000
    #
    # The form the FIRST cut of this audit was blind to, and the more common
    # one in this workspace. `num` is constrained to 2-3 digits starting with 9
    # so that `(boards.len() * 9) / 10` -- a fraction, not a percentile -- does
    # not match.
    # n * p / 100 where p is a VARIABLE (a closure/fn parameter).
    #
    # The third instance of the classifier-narrowness failure in this audit.
    # riir-game-sdk's `percentiles` helper is
    #     let at = |p: usize| durs[(n * p / 100).min(n - 1)];  at(50), at(99)
    # and the literal-only pattern below reported that repo as having ZERO
    # percentile sites -- while it feeds the repo's wall-clock budget gates.
    # The percentile is not statically known, so these land in UNRESOLVED
    # (honest) rather than vanishing from the population (not).
    #
    # Restricted to matches INSIDE square brackets on the same line: a
    # percentile is by definition an index into a sorted sample array, and
    # without that filter `a * b / 100` in unrelated arithmetic floods the
    # report. Applied to this entry only, because real sites do split the
    # index and the indexing across two lines.
    ("int_ratio_var",
     re.compile(r"\b(?P<n>[A-Za-z_]\w*(?:\.len\(\))?)\s*\*\s*(?P<pv>[a-z_]\w*)"
                r"\s*\)?\s*/\s*(?P<den>100|1000)\b"),
     False),
    ("int_ratio",
     re.compile(r"\b(?P<n>[A-Za-z_]\w*(?:\.len\(\))?)\s*\*\s*(?P<num>9\d{1,2})"
                r"\s*\)?\s*/\s*(?P<den>100|1000)\b"),
     False),
    # p * n as f64   |   n as f64 * p   with a VARIABLE p, in a
    # PERCENTILE-HELPER scope only (see HELPER_* below).
    #
    # The FOURTH instance of the classifier-narrowness failure this file
    # documents, and the first one arrived at through a CORRECT fix. The
    # 2026-09-03 repair campaign (riir-ai 03a91ed59 / riir-chain 7f3a3910 /
    # riir-mmorpg-examples ee9da24 / riir-game-sdk f896bca) took the workspace
    # DEGENERATE count 12 -> 0 by consolidating the arithmetic behind a
    # `nearest_rank(sorted, p)` helper -- whose `p` is a PARAMETER, so every
    # literal-p pattern above stopped matching and 16 sites left the
    # population entirely. `max_degenerate = 0` then read green over a
    # population that no longer contained riir-ai's percentile surface at all.
    #
    # These two entries put the helper BODY back in the population -- which is
    # the right altitude: audit the one helper, not its ~101 call sites.
    # Correctness is decidable from the SHAPE alone even though p is unknown:
    # `ceil(p*n)-1` can never be the max for n >= 3, and `floor(p*n)` is the
    # max for every n <= 1/(1-p) whatever p is. So a rounded body clears as
    # SAFE by the ROUNDED_RE path and a truncating one is reported as
    # TRUNC_VAR -- a finding without a resolvable n, unlike UNRESOLVED.
    ("float_var_rank_rev",
     re.compile(r"(?P<pv>[a-z_]\w*)\s*\*\s*(?P<n>[A-Za-z_]\w*(?:\.len\(\))?)"
                r"\s+as\s+f(?:32|64)"),
     False),
    ("float_var_rank",
     re.compile(r"\b(?P<n>[A-Za-z_]\w*(?:\.len\(\))?)\s+as\s+f(?:32|64)\s*\)?"
                r"\s*\*\s*(?P<pv>[a-z_]\w*)\b"),
     False),
]

DEGENERATE, WEAK, OK, UNRESOLVED, SAFE = "DEGENERATE", "WEAK", "OK", "UNRESOLVED", "SAFE"
# A truncating variable-p rank inside a percentile helper: the sample count is
# a runtime length so no threshold can be printed, but the SHAPE is wrong for
# every p -- which is strictly more than UNRESOLVED knows.
TRUNC_VAR = "TRUNC-VAR"
VAR_RANK_KINDS = ("float_var_rank", "float_var_rank_rev")

# ── Percentile-helper scope names (DATA -- extend here, then extend selftest)
#
# The discriminator for VAR_RANK_KINDS, and it has to be a NAME check because
# a variable-p multiply is otherwise ordinary arithmetic. Measured over the 27
# `let <idx-ish> = ... as fNN * <var>` candidate sites in the 19-repo
# workspace: this set admits 8 (7 correct `nearest_rank`/`quantile` bodies +
# riir-train's truncating `quartile` closure) and rejects all 19 non-percentile
# ones, including the two that a bare `rank` substring would have swallowed --
# `spectral_adaptive_ranks` (truncating, a rank BUDGET) and `principal_rank`.
# A report that cries wolf on rank budgets is a report nobody runs, so the
# substring set is deliberately narrow and `rank_ns` is listed EXACTLY.
HELPER_SUBSTR = ("nearest_rank", "percentile", "quantile", "quartile",
                 "decile", "pctl", "tail_at")
HELPER_EXACT = frozenset({"rank_ns", "p50", "p75", "p90", "p95", "p99", "p999"})


def in_percentile_scope(name):
    """Is `name` (an enclosing fn or closure binding) a percentile helper?"""
    if not name:
        return False
    return name in HELPER_EXACT or any(t in name for t in HELPER_SUBSTR)

# `.ceil()` / `.round()` applied to the product is NOT this defect: the bug is
# floor/truncation (`as usize`). `ceil(p*n) - 1` is the correct nearest-rank
# form. It also excludes the largest false-positive class -- `0.95 * n as f32`
# computing a top-p NUCLEUS SIZE (a probability mass), not an index into a
# sorted sample array. That shape was reported as the single most actionable
# row ("WEAK + ASSERTED") on the second cut of this report, from
# katgpt-rs/tests/bench_181_dmoe_vocab_coreset_goat.rs:369, which indexes
# nothing.
#
# `trunc` was in this set until 2026-09-03 and it is a HOLE, not a rounding
# mode: `x.trunc()` IS `x as usize` for every non-negative x, so a site written
# `(p * n as f64).trunc() as usize` is the defect spelled a second way and was
# classified SAFE by the very regex whose comment says "the bug is
# floor/truncation". Latent when found (zero percentile-context `.trunc()`
# sites in the 19-repo workspace, measured), but a ceiling is only as wide as
# its classifier and this one had a spelling-shaped gap. `.floor()` was never
# in the set and must never be: it is the defect's own name.
ROUNDED_RE = re.compile(r"\.\s*(?:ceil|round)\s*\(\s*\)")


def parse_site(m, kind):
    """(p as a fraction, index-function) for a match, or (None, None).

    The index function reproduces the site's OWN arithmetic exactly -- integer
    truncation for the `* num / den` form, f64-multiply-then-truncate for the
    float forms. Approximating one with the other moves the boundary by a
    rank, which is the whole finding.
    """
    if kind == "int_ratio_var" or kind in VAR_RANK_KINDS:
        return None, None          # percentile bound at the call site
    if kind == "int_ratio":
        num, den = int(m.group("num")), int(m.group("den"))
        return num / den, (lambda n: (n * num) // den)
    raw = (m.groupdict().get("p") or "").strip()
    if not raw.startswith("0."):
        return None, None          # `p / 100.0` with a variable p
    p = float(raw)
    return p, (lambda n: int(n * p))


def classify(n, p, idx_fn, safe):
    if safe:
        return SAFE, None, None
    if n is None or p is None or idx_fn is None or n < 1:
        return UNRESOLVED, None, None
    idx = min(idx_fn(n), n - 1)
    support = n - idx
    if idx == n - 1:
        return DEGENERATE, idx, support
    return (WEAK if support < MIN_SUPPORT else OK), idx, support


# ── Sample-count resolution ───────────────────────────────────────────────
CONST_RE = re.compile(r"const\s+(\w+)\s*:\s*\w+\s*=\s*([0-9_]+)\s*;")
LET_NUM_RE = re.compile(r"let\s+(\w+)\s*(?::\s*\w+)?\s*=\s*([0-9_]+)\s*;")
CAP_RE = re.compile(r"let\s+(?:mut\s+)?(\w+)\s*(?::[^=\n]+)?=\s*Vec::with_capacity\(\s*([\w.]+)")
LEN_RE = re.compile(r"let\s+(\w+)\s*(?::\s*\w+)?\s*=\s*(\w+)\s*\.\s*len\(\)")


def _uniq_ints(pairs):
    """name -> value, but a name bound to two DIFFERENT literals in one file is
    dropped rather than guessed. `content_store/goat.rs` binds `N_BLOBS` to
    both 50 and 100 in different fns; picking either would be a coin flip
    reported as a measurement."""
    seen = {}
    bad = set()
    for k, v in pairs:
        if k in seen and seen[k] != v:
            bad.add(k)
        seen[k] = v
    return {k: v for k, v in seen.items() if k not in bad}


FN_START_RE = re.compile(r"^\s*(?:pub\s+(?:\([^)]*\)\s+)?)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+\w+")


def enclosing_scope(lines, i):
    """(start, end) line indices of the fn containing line `i`.

    Both `resolve_n` and `is_load_bearing` were file-scoped on the first cut
    and both were WRONG for the same reason: a `let`/`with_capacity` binding
    or an `assert!` in a DIFFERENT function is not in scope. That produced a
    false ASSERTED on riir-neuron-db bench_003 (the assert is on `mean_us`,
    the p99 line's neighbour in a returned tuple) and resolved a slice
    PARAMETER's length from an unrelated caller in riir-chain bench_012.
    Module-level `const`s stay file-scoped, because they genuinely are.
    """
    start = 0
    for j in range(i, -1, -1):
        if FN_START_RE.match(lines[j]):
            start = j
            break
    end = len(lines)
    for j in range(i + 1, len(lines)):
        if FN_START_RE.match(lines[j]):
            end = j
            break
    return start, end


FN_NAME_RE = re.compile(r"\bfn\s+(\w+)")
CLOSURE_ASSIGN_RE = re.compile(r"let\s+(\w+)\s*(?::[^=]*)?=\s*\|")


def scope_names(lines, i):
    """(enclosing fn name, nearest enclosing closure-binding name) for line i.

    The closure half is not optional: riir-train's truncating site is a
    `let quartile = |q: f64| { ... }` inside `fn main`, so an fn-name-only
    discriminator would have rejected the one true positive in the workspace
    and admitted nothing.
    """
    start, _ = enclosing_scope(lines, i)
    m = FN_NAME_RE.search(lines[start]) if start < len(lines) else None
    fn = m.group(1) if m else None
    closure = None
    for j in range(i, start - 1, -1):
        cm = CLOSURE_ASSIGN_RE.search(lines[j])
        if cm:
            closure = cm.group(1)
            break
    return fn, closure


def resolve_n(expr, file_text, scope_text):
    """Resolve an n-expression to an integer, or None.

    `let` / `Vec::with_capacity` / `.len()` bindings are looked up in the
    ENCLOSING FN only; `const` is looked up in the fn first and then
    file-wide (module-level consts are visible everywhere). Follows up to 4
    links: `let n = xs.len()` -> `Vec::with_capacity(N)` -> `const N = 100`.
    """
    expr = expr.strip()
    if expr.isdigit():
        return int(expr)
    consts_local = _uniq_ints((m.group(1), int(m.group(2).replace("_", "")))
                              for m in CONST_RE.finditer(scope_text))
    consts_file = _uniq_ints((m.group(1), int(m.group(2).replace("_", "")))
                             for m in CONST_RE.finditer(file_text))
    lets = _uniq_ints((m.group(1), int(m.group(2).replace("_", "")))
                      for m in LET_NUM_RE.finditer(scope_text))
    caps = {m.group(1): m.group(2) for m in CAP_RE.finditer(scope_text)}
    lens = {m.group(1): m.group(2) for m in LEN_RE.finditer(scope_text)}

    seen, cur = set(), expr
    for _ in range(4):
        if cur in seen:
            return None
        seen.add(cur)
        base = cur[:-6].strip() if cur.endswith(".len()") else cur
        for table in (consts_local, lets, consts_file):
            if base in table:
                return table[base]
        if base in lens:
            cur = lens[base]
            continue
        if base in caps:
            cur = caps[base]
            continue
        return None
    return None


ASSERT_RE = re.compile(r"\bassert(?:_eq|_ne)?!")


def _balanced_args(text, open_idx):
    """Content between the parens of a macro call starting at `open_idx` (the
    index of `(`), respecting nesting. A fixed-width window instead of this
    is what crossed statement boundaries and manufactured a false ASSERTED."""
    depth, i, n = 0, open_idx, len(text)
    while i < n:
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i]
        i += 1
    return text[open_idx + 1 :]


def _subscript_aliases(var, scope):
    """Names bound to `<seq>[... var ...]` in this scope -- ONE hop.

    The index variable is almost never the asserted one. The normal shape is

        let p99_idx = (READS as f64 * 0.99) as usize;   // the site
        let p99_ns = latencies_ns[p99_idx];             // the hop
        assert!(p99_ns < 200, "G5 FAIL ...");           // the verdict

    so a same-name-only search reports `asserted=False` for a percentile that
    decides a GOAT gate (measured: katgpt-core content_store/goat.rs:323).
    That is a SEVERITY DOWNGRADE -- it moves a real finding out of the
    "+ ASSERTED (deciding a verdict)" bucket into "print-only (misleading a
    reader)" -- and it is silent, because a downgrade still prints a row.

    Deliberately one hop and deliberately subscript-only: `var` must appear
    inside a bracket pair, i.e. the alias IS the sample this rank selects.
    Chasing arbitrary arithmetic, or chasing transitively, buys the shapes
    nobody writes at the cost of a false ASSERTED -- and a false ASSERTED is
    what makes the report's most actionable row untrustworthy, which is the
    same defect in the other direction.
    """
    inside_brackets = re.compile(r"\[[^\]]*\b" + re.escape(var) + r"\b[^\]]*\]")
    out = []
    for line in scope.splitlines():
        am = ASSIGN_RE.search(line)
        if not am:
            continue
        rhs = line[am.end():]
        if inside_brackets.search(rhs):
            out.append(am.group(1))
    return out


def is_load_bearing(var, lines, i):
    """Does the value feed an `assert!` IN THE SAME FUNCTION?

    Whole-word match against the assert's balanced argument list only -- not a
    character window, and not the whole file. A print-only quantile is
    misleading; an asserted one decides a verdict, and conflating them makes
    the report's most actionable row untrustworthy.

    The index variable itself is checked, plus its one-hop subscript aliases --
    see _subscript_aliases for why the same-name-only form was a silent
    severity downgrade rather than a miss.
    """
    if not var:
        return False
    start, end = enclosing_scope(lines, i)
    scope = "\n".join(lines[start:end])
    names = [var, *_subscript_aliases(var, scope)]
    word = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b")
    for m in ASSERT_RE.finditer(scope):
        paren = scope.find("(", m.end() - 1)
        if paren == -1:
            continue
        if word.search(_balanced_args(scope, paren)):
            return True
    return False


ASSIGN_RE = re.compile(r"let\s+(\w+)\s*(?::[^=]*)?=")


def audit_file(path, rel):
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return []
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("///"):
            continue
        for kind, rx, safe in VOCAB:
            m = rx.search(line)
            if not m:
                continue
            if kind == "int_ratio_var":
                before, after = line[: m.start()], line[m.end() :]
                if "[" not in before or "]" not in after:
                    continue       # not an index -> not a percentile
            if kind in VAR_RANK_KINDS:
                _fn, _cl = scope_names(lines, i)
                if not (in_percentile_scope(_fn) or in_percentile_scope(_cl)):
                    continue       # ordinary arithmetic, not a rank
            if ROUNDED_RE.search(line[m.end():]):
                safe = True          # explicit rounding, not truncation
            p, idx_fn = parse_site(m, kind)
            _s, _e = enclosing_scope(lines, i)
            n = resolve_n(m.group("n"), text, "\n".join(lines[_s:_e]))
            verdict, idx, support = classify(n, p, idx_fn, safe)
            if kind in VAR_RANK_KINDS and not safe:
                # p is unknown, so no threshold can be printed -- but the shape
                # is the defect for EVERY p, which UNRESOLVED cannot say.
                verdict = TRUNC_VAR
            am = ASSIGN_RE.search(line)
            var = am.group(1) if am else None
            out.append({
                "file": rel, "line": i + 1, "kind": kind, "p": p, "n": n,
                "idx": idx, "support": support, "verdict": verdict,
                "asserted": is_load_bearing(var, lines, i),
                "text": stripped[:100],
            })
            break
    return out


def repos(root):
    return sorted(
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "BOUNDARY.md"))
        and os.path.isdir(os.path.join(root, d, ".git"))
    )


def walk_rs(repo_root):
    skip = {"target", ".git", "node_modules", ".venv"}
    for dp, dns, fns in os.walk(repo_root):
        dns[:] = [d for d in dns if d not in skip]
        for f in fns:
            if f.endswith(".rs"):
                yield os.path.join(dp, f)


def selftest():
    """Pin the tokenizer AND the arithmetic. Both failure modes are SILENT: a
    regex regression makes the report find fewer sites and still print a
    confident summary, and a classifier bug turns DEGENERATE into OK.

    Canaried by the bug this file shipped with on its first run -- the
    n-expression class included `[` and `(`, so `sorted[(n as f64 * 0.99)`
    captured `sorted[(n` and EVERY site resolved to UNRESOLVED. The selftest
    refused to print; without it the report would have shown 99 sites, zero
    findings, and looked like good news."""
    cases = [
        # (source line, expected kind, expected p, context, expected verdict)
        ("    let p99 = sorted[(n as f64 * 0.99) as usize];", "float_mul", 0.99,
         "let n = xs.len();\nlet mut xs: Vec<u64> = Vec::with_capacity(N);\nconst N: usize = 100;",
         DEGENERATE),
        ("    let p99 = sorted[n * 99 / 100];", "int_ratio", 0.99,
         "let n = 100;", DEGENERATE),
        ("    let p99 = sorted[(timings.len() * 99) / 100];", "int_ratio", 0.99,
         "let mut timings: Vec<u64> = Vec::with_capacity(ITERS);\nconst ITERS: usize = 1000;",
         OK),
        ("    let p95 = data[n * 95 / 100];", "int_ratio", 0.95,
         "let n = 20;", DEGENERATE),
        ("    let p99 = sorted[(n as f64 * 0.99) as usize];", "float_mul", 0.99,
         "let n = 200;", WEAK),
        ("    let i = ((v.len() as f64 - 1.0) * 0.99) as usize;", "float_len_minus_one", 0.99,
         "", SAFE),
        ("    let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;",
         "float_len_minus_one", None, "", SAFE),
        ("    let p999 = s[(s.len() as f32 * 0.999) as usize];", "float_mul", 0.999,
         "let mut s: Vec<u64> = Vec::with_capacity(N);\nconst N: usize = 100;", DEGENERATE),
        ("    let p99_us = samples[((N_ITERS as f64) * 0.99) as usize];", "float_mul", 0.99,
         "const N_ITERS: usize = 200;", WEAK),
        # a fraction, NOT a percentile -- must not match at all
        ("    let min_expected = (boards.len() * 9) / 10;", None, None, "", None),
        # ambiguous const: two different literals for one name -> UNRESOLVED,
        # never a coin flip reported as a measurement
        ("    let p99_idx = (READS as f64 * 0.99) as usize;", "float_mul", 0.99,
         "const READS: usize = 50;\nfn other() { const READS: usize = 100; }", UNRESOLVED),
    ]
    fails = []
    for src, kind, p_exp, ctx, verdict_exp in cases:
        hit = None
        for k, rx, safe in VOCAB:
            m = rx.search(src)
            if m:
                hit = (k, m, safe)
                break
        if kind is None:
            if hit is not None:
                fails.append(f"expected NO match but got {hit[0]}: {src!r}")
            continue
        if hit is None:
            fails.append(f"NO MATCH: {src!r}")
            continue
        k, m, safe = hit
        if k != kind:
            fails.append(f"kind {k} != {kind} for {src!r}")
            continue
        p, idx_fn = parse_site(m, k)
        if p_exp is not None and (p is None or abs(p - p_exp) > 1e-9):
            fails.append(f"p {p} != {p_exp} for {src!r}")
            continue
        _ctx = ctx + "\n" + src
        n = resolve_n(m.group("n"), _ctx, _ctx)
        v, _idx, _sup = classify(n, p, idx_fn, safe)
        if v != verdict_exp:
            fails.append(f"verdict {v} != {verdict_exp} (n={n}, p={p}) for {src!r}")

    # ── variable-percentile form must be in the population ──
    var_src = "        let at = |p: usize| durs[(n * p / 100).min(n - 1)];"
    if not any(rx.search(var_src) for _k, rx, _s in VOCAB):
        fails.append("variable-percentile form (n * p / 100) not in the vocabulary")
    # ...and it must be bracket-filtered: bare arithmetic is not a percentile
    bare = "        let pct = total * frac / 100;"
    for k_, rx, _s in VOCAB:
        m2 = rx.search(bare)
        if m2 and k_ == "int_ratio_var":
            if "[" in bare[: m2.start()] and "]" in bare[m2.end() :]:
                fails.append("bracket filter admitted non-indexing arithmetic")

    # ── the two spellings of truncation must NOT clear as rounded ──
    for src in (
        "        let idx = (p * n as f64).trunc() as usize;",
        "        let idx = ((v.len() as f64) * p).floor() as usize;",
    ):
        if ROUNDED_RE.search(src):
            fails.append(f"rounding exclusion cleared a TRUNCATING site: {src!r}")

    # ── rounding exclusion: `.ceil()` is not this defect ──
    for src in (
        "        let expected_min = (0.95 * vocab_size as f32).ceil() as usize;",
        "        let k = (0.99 * n as f64).ceil() as usize;",
        "        let idx = ((p * n as f64).round()) as usize;",
    ):
        hit = None
        for k_, rx, safe_ in VOCAB:
            m = rx.search(src)
            if m:
                hit = (k_, m, safe_)
                break
        if hit is None:
            continue                      # not matching at all is also fine
        if not ROUNDED_RE.search(src[hit[1].end():]):
            fails.append(f"rounding exclusion missed: {src!r}")

    # ── scoping canaries (the bugs the first cut shipped) ──
    # (a) a binding in ANOTHER fn must not resolve a slice parameter's length
    other_fn = [
        "fn caller() {",
        "    let mut latencies_ns: Vec<u64> = Vec::with_capacity(N);",
        "    const N: usize = 50;",
        "}",
        "fn summarize(latencies_ns: &[u64]) {",
        "    let p99_idx = ((latencies_ns.len() as f64) * 0.99) as usize;",
        "}",
    ]
    st, en = enclosing_scope(other_fn, 5)
    if resolve_n("latencies_ns.len()", "\n".join(other_fn), "\n".join(other_fn[st:en])) is not None:
        fails.append("resolve_n crossed a fn boundary to size a slice parameter")
    # (b) an assert on a SIBLING tuple element must not mark p99 as asserted
    sib = [
        "fn bench() -> (f64, f64) {",
        "    let p99_us = samples[((N as f64) * 0.99) as usize];",
        "    (mean_us, p99_us)",
        "}",
        "fn main() {",
        "    let (mean_us, p99_us) = bench();",
        "    assert!(mean_us < BUDGET, \"mean {mean_us} over budget\");",
        "}",
    ]
    if is_load_bearing("p99_us", sib, 1):
        fails.append("is_load_bearing crossed a fn boundary / matched a sibling binding")
    # (c) ...but a real assert in the SAME fn must still be found
    same = [
        "fn row() {",
        "    let p99 = sorted[(n as f64 * 0.99) as usize];",
        "    assert!(p99 < 5_000, \"tail over budget\");",
        "}",
    ]
    if not is_load_bearing("p99", same, 1):
        fails.append("is_load_bearing missed an assert in the same fn")
    # (d) the ONE-HOP subscript alias: the index variable is almost never the
    # asserted one, and a same-name-only search silently DOWNGRADES the
    # severity of a real finding rather than dropping it. Measured shape,
    # katgpt-core content_store/goat.rs:323.
    hop = [
        "fn g5() {",
        "    let p99_idx = (READS as f64 * 0.99) as usize;",
        "    let p99_ns = latencies_ns[p99_idx];",
        "    assert!(p99_ns < 200, \"G5 FAIL: p99 {p99_ns}ns\");",
        "}",
    ]
    if not is_load_bearing("p99_idx", hop, 1):
        fails.append("is_load_bearing missed the one-hop subscript alias (idx -> value -> assert)")
    # (e) ...and the hop must be a SUBSCRIPT, not any mention. An alias bound
    # from unrelated arithmetic on the index is not the sample it selects, and
    # crediting it would manufacture a false ASSERTED -- which is the same
    # defect as (d) in the direction that makes the actionable rows untrusted.
    noise = [
        "fn g5() {",
        "    let p99_idx = (READS as f64 * 0.99) as usize;",
        "    let budget_ns = p99_idx * 2;",
        "    assert!(budget_ns > 0, \"budget {budget_ns}\");",
        "}",
    ]
    if is_load_bearing("p99_idx", noise, 1):
        fails.append("is_load_bearing credited a non-subscript alias as asserted")
    # (f) the bound is ONE hop, stated as a pin rather than left to be
    # rediscovered: a two-hop chain is NOT claimed. If a real two-hop site
    # ever shows up, widen here and move this case -- do not read the pass as
    # evidence that two hops are covered.
    two_hop = [
        "fn g5() {",
        "    let p99_idx = (READS as f64 * 0.99) as usize;",
        "    let p99_ns = latencies_ns[p99_idx];",
        "    let p99_us = p99_ns / 1000;",
        "    assert!(p99_us < 1, \"p99 {p99_us}us\");",
        "}",
    ]
    if is_load_bearing("p99_idx", two_hop, 1):
        fails.append("is_load_bearing chased more than one hop (bound is documented as one)")

    # ── variable-p rank canaries, END-TO-END through audit_file ──
    #
    # These go through the real entry point rather than the regex, because the
    # discriminator that keeps this class from flooding the report lives in
    # `audit_file` (the scope-name filter), not in the vocabulary. Case (c) is
    # the one that earns its keep: the SAME truncating shape outside a
    # percentile scope must produce NOTHING, or the report grows 19 rank-budget
    # and bucket-index false positives and stops being read.
    import tempfile
    var_cases = [
        # (label, source, expected verdicts present / absent)
        ("correct helper body -> SAFE",
         "fn nearest_rank(sorted: &[f64], p: f64) -> (f64, usize) {\n"
         "    let n = sorted.len();\n"
         "    let idx = ((p * n as f64).ceil() as usize).clamp(1, n) - 1;\n"
         "    (sorted[idx], n - idx)\n}\n", SAFE),
        ("truncating helper body -> TRUNC-VAR",
         "fn percentile(sorted: &[f64], p: f64) -> f64 {\n"
         "    let idx = ((p * sorted.len() as f64) as usize).min(sorted.len() - 1);\n"
         "    sorted[idx]\n}\n", TRUNC_VAR),
        ("truncating CLOSURE in a non-percentile fn -> TRUNC-VAR via closure name",
         "fn main() {\n"
         "    let quartile = |q: f64| -> usize {\n"
         "        let idx = ((q * lengths.len() as f64) as usize).min(lengths.len() - 1);\n"
         "        lengths[idx]\n    };\n}\n", TRUNC_VAR),
        ("same shape, rank BUDGET scope -> no finding at all",
         "fn spectral_adaptive_ranks(alphas: &[f32]) {\n"
         "    let rank = (total_rank_budget as f32 * a / alpha_sum) as u16;\n}\n", None),
        ("same shape, bucket-index scope -> no finding at all",
         "fn uniform_is_well_distributed(u: f32) {\n"
         "    let idx = ((u * BUCKETS as f32) as usize).min(BUCKETS - 1);\n}\n", None),
    ]
    for label, src_txt, want in var_cases:
        with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(src_txt)
            tmp = fh.name
        try:
            got = [r["verdict"] for r in audit_file(tmp, "canary.rs")]
        finally:
            os.unlink(tmp)
        if want is None:
            if got:
                fails.append(f"var-rank canary '{label}': expected no finding, got {got}")
        elif want not in got:
            fails.append(f"var-rank canary '{label}': expected {want}, got {got or 'nothing'}")

    # ── the shape claim TRUNC_VAR rests on, stated EXACTLY ──
    #
    # The first cut of this block asserted "ceil(p*n)-1 can never be the max"
    # and it is FALSE: p=0.75, n=3 gives ceil(2.25)=3 -> idx 2 = n-1. The true
    # relation is a one-rank shift of the SAME boundary --
    #   floor(p*n) is the max for every n <= 1/(1-p)
    #   ceil(p*n)-1 is the max for every n <  1/(1-p)
    # -- so nearest rank is never worse, is strictly better at exactly the
    # integral boundary, and is still the max below it, where no such quantile
    # exists in the sample at all. That last part is why the helpers return
    # `support` and why SAFE means "correct form", not "cannot be one sample".
    # Fraction, not float: at p=0.999 the boundary lands on a value the float
    # comparison gets wrong, which would pin the wrong claim.
    from fractions import Fraction
    for pn, pd in ((1, 2), (3, 4), (9, 10), (19, 20), (99, 100), (999, 1000)):
        pf = Fraction(pn, pd)
        strictly_better = 0
        for n_ in range(2, 5001):
            prod = pf * n_
            floor_idx = min(int(prod), n_ - 1)
            ceil_idx = min(max(-((-prod.numerator) // prod.denominator), 1), n_) - 1
            floor_deg, ceil_deg = floor_idx == n_ - 1, ceil_idx == n_ - 1
            assert not (ceil_deg and not floor_deg), (
                f"nearest rank degenerate where truncation is not, n={n_} p={pf}")
            assert ceil_deg == (Fraction(n_) * (1 - pf) < 1), (
                f"ceil boundary wrong at n={n_} p={pf}")
            assert floor_deg == (Fraction(n_) * (1 - pf) <= 1), (
                f"floor boundary wrong at n={n_} p={pf}")
            strictly_better += int(floor_deg and not ceil_deg)
        assert strictly_better == 1, (
            f"nearest rank should beat truncation at exactly one n, got "
            f"{strictly_better} for p={pf}")

    # ── arithmetic pins, independent of every regex above ──
    assert int(100 * 0.99) == 99, "n=100 p99 truncates to the max index"
    assert int(101 * 0.99) == 99, "n=101 is the first count that is NOT the max"
    assert (100 * 99) // 100 == 99 and (200 * 99) // 100 == 198
    for n, exp in ((100, 1), (200, 2), (1000, 10)):
        assert n - min(int(n * 0.99), n - 1) == exp, f"naive support at n={n}"
    # the SAFE form can never reach n-1, over the whole range it claims
    for n in range(2, 20001):
        assert int((n - 1) * 0.99) <= n - 2, f"(n-1)*0.99 reached n-1 at n={n}"

    if fails:
        print("SELFTEST FAILED — the report below cannot be trusted:")
        for f in fails:
            print("  " + f)
        sys.exit(2)


def main():
    selftest()
    root = "/Users/katopz/git"
    if len(sys.argv) > 1:
        targets = [os.path.abspath(sys.argv[1])]
        root = os.path.dirname(targets[0])
    else:
        targets = [os.path.join(root, r) for r in repos(root)]

    print(f"percentile-index audit — MIN_SUPPORT={MIN_SUPPORT}, "
          f"{len(targets)} repo(s) (derived: BOUNDARY.md + .git)\n")
    grand = {}
    all_findings = []
    for t in targets:
        name = os.path.basename(t)
        found = []
        for f in walk_rs(t):
            found += audit_file(f, os.path.relpath(f, t))
        if not found:
            continue
        tally = {}
        for r in found:
            tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
            r["repo"] = name
        grand[name] = tally
        all_findings += found

    # ── the two rows that matter, most severe first ──
    for label, pred in (
        ("DEGENERATE + ASSERTED  (a percentile that IS the max, deciding a verdict)",
         lambda r: r["verdict"] == DEGENERATE and r["asserted"]),
        ("DEGENERATE, print-only (a percentile that IS the max, misleading a reader)",
         lambda r: r["verdict"] == DEGENERATE and not r["asserted"]),
        (f"WEAK + ASSERTED       (support < {MIN_SUPPORT}, one stall can flip it)",
         lambda r: r["verdict"] == WEAK and r["asserted"]),
        ("TRUNC-VAR              (floor(p*n) in a percentile helper -- the max "
         "for every n <= 1/(1-p), whatever p is)",
         lambda r: r["verdict"] == TRUNC_VAR),
    ):
        rows = [r for r in all_findings if pred(r)]
        print(f"── {label}: {len(rows)}")
        for r in sorted(rows, key=lambda r: (r["repo"], r["file"], r["line"])):
            if r["verdict"] == TRUNC_VAR:
                # p is a parameter, so p/idx/support are all None here -- the
                # source line is the whole finding.
                print(f"     {r['repo']}/{r['file']}:{r['line']}  {r['text']}")
            else:
                print(f"     {r['repo']}/{r['file']}:{r['line']}  p={r['p']} n={r['n']} "
                      f"idx={r['idx']} support={r['support']}")
        print()

    print("── per-repo tally (verdict x count) " + "─" * 30)
    hdr = [DEGENERATE, TRUNC_VAR, WEAK, OK, UNRESOLVED, SAFE]
    print(f"  {'repo':<24}" + "".join(f"{h:>12}" for h in hdr) + f"{'total':>8}")
    tot = {h: 0 for h in hdr}
    for name in sorted(grand):
        t = grand[name]
        print(f"  {name:<24}" + "".join(f"{t.get(h, 0):>12}" for h in hdr)
              + f"{sum(t.values()):>8}")
        for h in hdr:
            tot[h] += t.get(h, 0)
    print(f"  {'ALL':<24}" + "".join(f"{tot[h]:>12}" for h in hdr)
          + f"{sum(tot.values()):>8}")
    print(f"\n  UNRESOLVED is not 'clean' — it is a sample count no static pass could\n"
          f"  reach (a runtime length, a fn parameter). Those need a per-site read.\n"
          f"  Report only; exit 0 always.")


if __name__ == "__main__":
    main()
