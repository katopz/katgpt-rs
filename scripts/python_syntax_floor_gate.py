#!/usr/bin/env python3
"""GATE: every tracked `*.py` in this repo parses at the Python 3.9 grammar.

Issue 965. The workstation boxes' `python3` is **3.9.6** (Xcode CLT), and the
workspace instruments are invoked DIRECTLY with it — `python3
../katgpt-rs/scripts/staged_set_audit.py` is the documented invocation in
AGENTS.md. A file carrying newer-only syntax is a dead tool on those boxes: it
dies with `SyntaxError` at import, before any verdict, on the exact invocation
its own documentation names. Measured 2026-10-26, `staged_set_audit.py` did
exactly that (a `match` statement, fixed `95e031d95`).

THE PRIMARY POSTURE IS THE 3.9-HOSTED ONE. `ast.parse(src,
feature_version=(3, 9))` is the detector, and on a 3.9 interpreter that IS the
native grammar — the floor checking itself, everything caught. On a NEWER
host (docs_gate resolves the highest interpreter; CI runs 3.10+)
`feature_version` gates only what CPython gates — measured 2026-10-26 on a
3.12 host: `match` / `except*` / `type` alias are REJECTED, but the PEP 701
f-string classes and PEP 646 starred subscripts are ACCEPTED. So this gate
adds measured AST supplements for exactly those, run only when the box's own
parse SUCCEEDED — which is why they cannot false-positive on 3.9-legal code
(a source that clears a 3.9 grammar contains none of these shapes):

- **PEP 701 (3.12)** — a backslash or a newline inside an f-string EXPRESSION
  part (the FormattedValue's source segment), or an inner string reusing the
  f-string literal's own quote character. All three REJECTED on real 3.9.6
  (measured: `f-string expression part cannot include a backslash` /
  `EOL while scanning string literal` / `f-string: unmatched '['`), accepted
  on 3.12. Backslashes in the LITERAL part are 3.9-legal and never flagged
  (measured). Attribution is per-literal: `f"{f'{d["k"]}'}"` stays clean —
  the inner `"` is compared against ITS OWN f-string's quote, not the
  outermost.
- **PEP 646 (3.11)** — a starred expression inside a subscript
  (`tuple[*Ts]`): `invalid syntax` on real 3.9.6, accepted at floor on 3.12.
  Starred calls/`*`-unpacking in displays are 3.9-legal and never flagged —
  only a Subscript slice is examined.

The gate is a SYNTAX check, deliberately NOT an interpreter pin: `tomllib`
imports (3.11+) parse fine at the 3.9 grammar and are this repo's documented
docs-gate posture, and PEP 604 unions in annotation position ride `from
__future__ import annotations` (a 3.9 feature). Single-line parenthesized
context managers are 3.9-LEGAL (measured on real 3.9.6) and never flagged.

Two buckets, pinned separately — different defects, different repairs:

- **SYNTAX** — parses at the box's own grammar, fails at the 3.9 floor: dead
  on the workstation, invisible on the box that wrote it.
- **UNPARSED** — fails the box's own grammar too: dead everywhere. The trap
  audit's rule applies: an instrument that cannot read a file says so, never
  folds it into the pass column.

Both ceilings are 0 in every posture: unlike a lint count there is no
DELIBERATE class — a file that cannot parse cannot run.

Floors, not just ceilings: a ceiling of 0 is green over whatever the walk can
see, so the population is pinned underneath it. `FLOOR_PY_FILES` moves when
the WALK goes blind; `FLOOR_PARSED` moves when the scan loop stops iterating
early on an unchanged tree — parsed == files in every passing state (any file
that fails the floor parse is a finding above), so the two floors are equal
HERE by construction while failing in different blind modes, and the drift
sweep asserts the equality rather than deriving it.

The self-test fixtures are DATA, built by concatenation where the construct
itself is newer-only: this file is itself a tracked `*.py` under its own gate,
so a 3.12-only construct may never appear as code in it. Selftest runs on
every invocation (docs_gate.sh passes no arguments — Issue 789's measured
finding), both directions.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tracked_walk import tracked_files  # noqa: E402

# Issue 804: this instrument is documented as directly invokable, and its
# verdict glyphs (✓ ✗ ⛔ ⚠) kill it on a non-UTF-8 console — no verdict at
# all, findings unread. docs_gate.sh's PYTHONIOENCODING only covers runs
# that go through the wrapper.
import console_safe  # noqa: E402

console_safe.apply()

REPO_ROOT = Path(__file__).resolve().parent.parent

# The floor: the oldest interpreter the workspace instruments are invoked
# with. Measured 2026-10-26 on amo (`python3 -V` → 3.9.6).
GRAMMAR = (3, 9)

# Measured 2026-10-26 at the landing: 107 tracked .py, all parsing at 3.9.
# Slack against ordinary churn, tight against a walk or scan regression — the
# same rule every floor here carries. The drift sweep's katgpt-rs row MUST
# equal these two, and asserts it.
FLOOR_PY_FILES = 60
FLOOR_PARSED = 60


def _joined_quote(src: str, joined: ast.AST) -> str | None:
    """The quote char of an f-string literal, from its own source segment.

    The prefix letters (f/rb/…) carry no quotes, so the FIRST quote char in
    the segment is the literal's own. A format-spec JoinedStr carries no
    quotes and returns None (its reuse check is vacuous anyway).
    """
    seg = ast.get_source_segment(src, joined)
    if not seg:
        return None
    for ch in seg:
        if ch in "\"'":
            return ch
    return None


def _own_string_constants(fv: ast.FormattedValue):
    """String Constants in ONE FormattedValue's expression, nested literals
    excluded.

    A nested JoinedStr (an inner f-string) is its OWN literal — its constants
    are attributed by its own iteration, never by the outer one. Skipping it
    here is what keeps `f"{f'{d["k"]}'}"` clean: the inner `"` matches the
    INNER literal's quote, not the outer one (measured 3.9-legal).
    """
    stack = [fv.value]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.JoinedStr):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node
            continue
        stack.extend(ast.iter_child_nodes(node))


def _pep701_rows(src: str, tree: ast.AST) -> list[tuple[str, str]]:
    """The 3.12 f-string classes `feature_version` does not gate (measured).

    ⛔ Segment-based, therefore ≥3.12-ONLY: on ≤3.11 the f-string is one
    STRING token and every node's segment spans the WHOLE literal (measured
    on 3.9.6) — a literal-part backslash would false-positive there. On a
    <3.12 host these constructs fail the box's own parse instead (UNPARSED),
    so gating the supplement off costs nothing: no posture is blind.
    """
    rows: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FormattedValue):
            seg = ast.get_source_segment(src, node)
            if seg and ("\\" in seg or "\n" in seg):
                rows.append((
                    "SYNTAX",
                    f"{node.lineno}: f-string expression part carries a "
                    f"backslash or newline — 3.9 rejects any (PEP 701)"))
    for joined in ast.walk(tree):
        if not isinstance(joined, ast.JoinedStr):
            continue
        quote = _joined_quote(src, joined)
        if quote is None:
            continue
        for value in joined.values:
            if not isinstance(value, ast.FormattedValue):
                continue
            for const in _own_string_constants(value):
                cseg = ast.get_source_segment(src, const)
                if cseg and cseg[0] in "\"'" and cseg[0] == quote:
                    rows.append((
                        "SYNTAX",
                        f"{const.lineno}: f-string expression reuses the "
                        f"literal's own quote character — 3.9 rejects it "
                        f"(PEP 701)"))
    return rows


def _pep646_rows(tree: ast.AST) -> list[tuple[str, str]]:
    """Starred subscripts (3.11+); calls and displays stay out of scope.

    AST-shape based, sound on every host ≥3.11 (regular nodes, precise
    positions — unlike the f-string ones). On <3.11 the construct fails the
    box's own parse (UNPARSED), so gating costs nothing here either.
    """
    rows: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            for sub in ast.walk(node.slice):
                if isinstance(sub, ast.Starred):
                    rows.append((
                        "SYNTAX",
                        f"{node.lineno}: starred expression inside a subscript"
                        f" — 3.11+ only (PEP 646)"))
                    break
    return rows


def floor_findings(src: str) -> list[tuple[str, str]]:
    """`[(kind, row)]` for one source text; `[]` when it clears the 3.9 floor.

    UNPARSED is decided FIRST (the box's own grammar): a file dead everywhere
    must not read as the milder floor-only class. ValueError is ast's answer
    to source it cannot treat as text (null bytes); RecursionError is a depth
    bomb — both are the same verdict.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [("UNPARSED", f"{e.lineno}: {e.msg}")]
    except (ValueError, RecursionError) as e:
        return [("UNPARSED", f"0: {e}")]
    rows: list[tuple[str, str]] = []
    try:
        ast.parse(src, feature_version=GRAMMAR)
    except SyntaxError as e:
        rows.append(("SYNTAX", f"{e.lineno}: {e.msg}"))
    except (ValueError, RecursionError) as e:
        rows.append(("SYNTAX", f"0: {e}"))
    # The supplements run only where they are SOUND (their docstrings); on an
    # older host the same constructs fail the box's own parse — UNPARSED — so
    # no posture is blind to them. Measured 2026-10-26 on 3.9.6 and 3.12.
    if sys.version_info >= (3, 12):
        rows.extend(_pep701_rows(src, tree))
    if sys.version_info >= (3, 11):
        rows.extend(_pep646_rows(tree))
    return rows


def scan(repo: Path) -> tuple[dict, dict, int, int]:
    """(syntax, unparsed, py_files, parsed) over the repo's TRACKED *.py.

    `parsed` counts files that cleared the floor parse — the scan-loop's own
    liveness quantity (`FLOOR_PARSED`). A file with findings lands in ONE
    bucket: UNPARSED dominates (no AST exists to supplement). `tracked_files`
    is the ONE walk (Issue 777): a filesystem rglob miscounted a nested repo
    once already.
    """
    syntax: dict = {}
    unparsed: dict = {}
    files, _excluded = tracked_files(repo, "*.py")
    parsed = 0
    for p in files:
        rel = str(p.relative_to(repo)).replace("\\", "/")
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            unparsed.setdefault(rel, []).append("0: unreadable")
            continue
        rows = floor_findings(src)
        if not rows:
            parsed += 1
            continue
        bucket = unparsed if any(k == "UNPARSED" for k, _ in rows) else syntax
        bucket.setdefault(rel, []).extend(r for _, r in rows)
    return syntax, unparsed, len(files), parsed


def selftest() -> list[str]:
    """Both directions, every invocation. The false-NEGATIVE direction is the
    one that matters: a floor that stopped rejecting prints `0 findings`
    forever, indistinguishable from the clean state this asserts. Every
    fixture is string DATA — this file parses at 3.9 under its own gate — and
    the posture split is explicit: on a 3.9 host a newer-only construct fails
    the BOX's grammar (UNPARSED); on a host that accepts it, the floor or a
    supplement must still reject it (SYNTAX).
    """
    fails: list[str] = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    v = sys.version_info[:2]

    def first(src):
        rows = floor_findings(src)
        return rows[0] if rows else (None, "")

    # ── feature_version-gated classes (measured REJECTED on a 3.12 host) ──
    match_src = ("def pick(x):\n"
                 "    match x:\n"
                 "        case 1:\n"
                 "            return 1\n"
                 "    return None\n")
    kind, row = first(match_src)
    check(kind is not None, f"the floor accepted a match statement ({row!r})")
    check(kind == ("SYNTAX" if v >= (3, 10) else "UNPARSED"),
          f"a match statement classified {kind} on {v}")

    except_src = ("try:\n"
                  "    pass\n"
                  "except* ValueError:\n"
                  "    pass\n")
    kind, _ = first(except_src)
    check(kind == ("SYNTAX" if v >= (3, 11) else "UNPARSED"),
          f"except* classified {kind} on {v}")

    alias_src = "type Alias = int\n"
    kind, _ = first(alias_src)
    check(kind == ("SYNTAX" if v >= (3, 12) else "UNPARSED"),
          f"a type alias classified {kind} on {v}")

    # ── PEP 701 / 646: the classes feature_version misses on a new host ──
    # Backslash in the EXPRESSION. Built by concatenation — the construct must
    # never appear as code in this file. Measured live at reflex-site
    # `render_flows.py` at the landing (fixed a24f4dd): the feature-grep
    # census missed it, the grammar parse did not.
    bs_src = "s = f\"{r'" + "\\d" + "'}\"\n"
    kind, _ = first(bs_src)
    check(kind == ("SYNTAX" if v >= (3, 12) else "UNPARSED"),
          f"an f-string backslash expr classified {kind} on {v}")

    nl_src = "s = f\"{foo(1,\n2)}\"\n"
    kind, _ = first(nl_src)
    check(kind == ("SYNTAX" if v >= (3, 12) else "UNPARSED"),
          f"an f-string newline expr classified {kind} on {v}")

    qr_src = 's = f"{d["k"]}"\n'
    kind, _ = first(qr_src)
    check(kind == ("SYNTAX" if v >= (3, 12) else "UNPARSED"),
          f"quote reuse classified {kind} on {v}")

    star_src = "Ts = (1,)\nx: tuple[*Ts]\n"
    kind, _ = first(star_src)
    check(kind == ("SYNTAX" if v >= (3, 11) else "UNPARSED"),
          f"a starred subscript classified {kind} on {v}")

    # ── the false-POSITIVE guards: all 3.9-LEGAL, clean on EVERY host ──────
    # `match`/`case` are SOFT keywords; a 3.9 file using them as NAMES parses
    # at every grammar, and reding it would train whoever hits it that the
    # gate is noise.
    ident_src = ("match = re.compile('x')\n"
                 "case = match.search('x')\n"
                 "print(case)\n")
    rows = floor_findings(ident_src)
    check(rows == [], f"match/case as identifiers classified {rows}")

    # Ordinary modern-3.9 constructs: walrus, positional-only, f-string =
    # specifier, the future-annotations import.
    clean_src = ("from __future__ import annotations\n"
                 "def f(a, /, b=0):\n"
                 "    if (n := a + b) > 2:\n"
                 "        return f'{n=} yes'\n"
                 "    return None\n")
    rows = floor_findings(clean_src)
    check(rows == [], f"a 3.9-clean file classified {rows}")

    # Single-line parenthesized context managers are 3.9-LEGAL (measured on
    # real 3.9.6) despite looking like the 3.10 whatsnew feature.
    paren_src = "with (open('a') as f, open('b') as g):\n    pass\n"
    rows = floor_findings(paren_src)
    check(rows == [], f"a parenthesized with classified {rows}")

    # Backslash in the LITERAL part is 3.9-legal; only the expression part
    # rejects it (measured message names the expression part). Clean on every
    # posture: on ≤3.11 the supplements are off (whole-literal segments would
    # false-positive), on ≥3.12 the precise segment is `{x}`.
    lit_src = "s = f\"a" + "\\n" + "b {x}\"\n"
    rows = floor_findings(lit_src)
    check(rows == [], f"a literal-part backslash classified {rows}")

    # Nested f-strings with DIFFERENT quotes are 3.9-legal.
    nest_src = "s = f\"{f'{1}'}\"\n"
    rows = floor_findings(nest_src)
    check(rows == [], f"nested f-strings classified {rows}")
    # The deeper chain is NOT legal on ≤3.11 (measured on real 3.9.6:
    # `f-string: unterminated string` — a real finding there), and on a 3.12
    # host it is legal and must stay clean: the inner `"` is compared against
    # ITS OWN literal's quote (`'`), not the outermost's. The attribution arm.
    chain_src = "s = f\"{f'{d[" + chr(34) + "k" + chr(34) + "]}'}\"\n"
    rows = floor_findings(chain_src)
    if v >= (3, 12):
        check(rows == [], f"the nested quote chain classified {rows}")
    else:
        check(rows and rows[0][0] == "UNPARSED",
              f"the nested quote chain classified {rows} on {v} — measured "
              f"REJECTED on real 3.9.6")

    # Starred calls / unpacking in displays are 3.9-legal; only a SUBSCRIPT
    # slice is the 3.11 form.
    unpack_src = ("def f(*a):\n"
                  "    return a\n"
                  "g = f(1)\n"
                  "h = [*g]\n"
                  "return_val = f(*g)\n")
    rows = floor_findings(unpack_src)
    check(rows == [], f"starred calls/display unpacking classified {rows}")

    # A STRING carrying the offending source is data, not code: the gate
    # parses FILES once; it never re-parses string content. This is the
    # design the whole-file fixtures above rely on, asserted in one place.
    rows = floor_findings("FIXTURE = 'match x:'\n")
    check(rows == [], "a string literal was classified as a floor finding")

    # UNPARSED: dead on every grammar, its own bucket, never folded into pass.
    kind, _ = first("def f(:\n")
    check(kind == "UNPARSED", f"garbage classified {kind}, wanted UNPARSED")

    # The floor constants: the sweep's katgpt-rs row asserts equality with
    # these, and parsed <= files is the invariant FLOOR_PARSED rides.
    check(GRAMMAR == (3, 9), f"GRAMMAR drifted: {GRAMMAR}")
    check(FLOOR_PARSED <= FLOOR_PY_FILES,
          f"FLOOR_PARSED {FLOOR_PARSED} > FLOOR_PY_FILES {FLOOR_PY_FILES}")

    return fails


def posture() -> str:
    """Which mechanism is doing the catching on THIS host — printed beside
    every verdict, because the answer changes what a green means.

    Measured 2026-10-26: no posture is blind (the docstring's matrix). A 3.9
    host IS the floor grammar (native); ≥3.12 adds the segment supplements;
    3.10/3.11 gate the feature_version classes while the PEP 701/646 classes
    fail their own parse instead.
    """
    v = sys.version_info[:2]
    if v == (3, 9):
        return f"posture: host {v[0]}.{v[1]} = the floor grammar itself — " \
               "full native coverage"
    if v >= (3, 12):
        return f"posture: host {v[0]}.{v[1]} — feature_version + precise-" \
               "segment supplements (PEP 701/646 covered)"
    return f"posture: host {v[0]}.{v[1]} — feature_version classes covered; " \
           "PEP 701/646 fail this host's own parse (UNPARSED) — covered"


def main(argv: list[str]) -> int:
    print(f"  {posture()}")
    fails = selftest()
    if fails:
        print("✗ python-syntax-floor gate SELFTEST FAILED — the verdict below "
              "cannot be trusted:")
        for f in fails:
            print("    " + f)
        return 2

    repo = Path(argv[1]).resolve() if len(argv) > 1 else REPO_ROOT
    syntax, unparsed, py_files, parsed = scan(repo)
    n_syn = sum(len(v) for v in syntax.values())
    n_unp = sum(len(v) for v in unparsed.values())

    problems: list[str] = []
    for rel in sorted(syntax):
        for row in syntax[rel]:
            print(f"    ⛔ SYNTAX   {rel}:{row}")
    for rel in sorted(unparsed):
        for row in unparsed[rel]:
            print(f"    ⛔ UNPARSED {rel}:{row}")
    if n_syn:
        problems.append(f"{n_syn} finding(s) fail the 3.9 grammar — dead on "
                        "the workstation python3 3.9.6 (rewrite without the "
                        "newer-only construct)")
    if n_unp:
        problems.append(f"{n_unp} finding(s) fail the box's own grammar — "
                        "dead everywhere, never a pass")
    if py_files < FLOOR_PY_FILES:
        problems.append(f"walk FLOOR breached: {py_files} tracked .py < "
                        f"{FLOOR_PY_FILES} — the ceilings above are green over "
                        "a population that shrank")
    if parsed < FLOOR_PARSED:
        problems.append(f"parse FLOOR breached: {parsed} of {py_files} files "
                        f"cleared the floor parse < {FLOOR_PARSED} — the walk "
                        "is intact but the scan stopped early")

    if problems:
        print(f"✗ python-syntax-floor gate FAILED — {len(problems)} problem(s) "
              f"over {py_files} tracked .py file(s) / {parsed} parsed at the "
              f"3.9 grammar")
        for p in problems:
            print("    ✗ " + p)
        print("    The boxes' python3 is 3.9.6 and instruments are invoked "
              "with it directly; a newer-only file dies at import before any "
              "verdict — see Issue 965.")
        return 1

    print(f"✓ python-syntax-floor gate PASSED — 0 SYNTAX, 0 UNPARSED over "
          f"{py_files} tracked .py file(s) (floor {FLOOR_PY_FILES}) / {parsed} "
          f"parsed at 3.9 (floor {FLOOR_PARSED})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
