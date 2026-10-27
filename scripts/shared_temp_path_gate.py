#!/usr/bin/env python3
"""A test wrote to a FIXED `env::temp_dir()` path, so two processes truncate it (Issue 832).

`std::env::temp_dir().join("fixed_name.bin")` is safe against other tests in
the same binary — each site has its own filename — and NOT safe against another
PROCESS running the same test. This workspace runs 5+ concurrent agent sessions
against shared worktrees, and its own instruments overlap by design: `test_gate`,
`full_gate`, `x86_64_execution_matrix` and any hand-run `cargo test` each get
their own target dir and all share ONE system temp directory. `create`
truncates, so A writes, B truncates, A reads back zero bytes.

Measured (Issue 832, 2026-09-18):

- `pruners::bomber::replay::tests::writer_writes_and_counts_samples` reproduced
  on demand — two concurrent copies of one binary, 1 failure in 24 runs, and
  byte-identical to the x86_64 matrix's cell-5 red. Alone it passes every time.
- Five `katgpt-types::tests_types` tests failed AT ONCE with "File too small for
  header", across five DIFFERENT filenames — another test binary truncating all
  five.

⛔ The reason this is a GATE and not a sweep-and-done: 13 sites in this
workspace already carried the unique-path form
(`katgpt-transformer/src/contiguous.rs`, `katgpt-core/src/content_store/fetcher.rs`,
the three `katgpt-pruners` sites) and 27 did not — 25 repaired, 2 adjudicated
as deliberate. The rule was known and un-enforced, which is this repo's own
most-repeated shape.

⛔ And the reason it is not covered by "run it again": the x86_64 matrix's
confirm step filed one of these as TRANSIENT — *failed in the cell, PASSED
alone* — which is true and the wrong conclusion. This class passes alone BY
CONSTRUCTION, because alone there is no second process. Re-running separates a
load-sensitive bar from a real regression and says nothing about a concurrency
defect, which is a permanent property of the test.

    scripts/shared_temp_path_gate.py    # the verdict AND the arms

The arms run UNCONDITIONALLY, behind no flag: `docs_gate.sh` invokes each check
as `"$PY" "$script"` with no arguments, so an arm behind `'--canary' in
sys.argv` never fires on a push (Issue 789's finding).

The quantity gated is NOT the count — *a set is gateable where its cardinality
is not*. Membership, both directions: a fixed-path site with no pin row reds,
and a pin row whose site is gone or whose per-(file, literal) count moved reds
STALE, so the file cannot only ever loosen.

The pin key is deliberately LINE-FREE: `relpath::literal::count`. A line number
drifts on every edit above it.

Population: every tracked `*.rs` `env::temp_dir()` immediately `.join`-ed with a
STRING LITERAL. A literal has no runtime discriminator by construction, which is
the whole defect; `join(format!("n_{}", std::process::id()))` is not a literal
and is out of population without needing a second rule.

⚠ STATED BLIND SPOTS — each is a real hole, named so a later census reads this
instead of re-deriving it:

- A `temp_dir()` bound to a variable first (`let base = env::temp_dir(); let p =
  base.join("fixed.bin");`) is NOT seen. The receiver's provenance is not
  tracked.
- Other spellings of "a fixed scratch path" are NOT seen: `PathBuf::from("/tmp/…")`,
  `"./target/test_scratch"`, `TempDir` wrappers. Issue 832 T4 owns measuring
  those before a floor is written for them — a floor over one spelling describes
  one spelling's population.
- `examples/` sites are in population and pinned, not excluded: a demo's temp
  path is meant to stay findable by a human, and that is an adjudication with a
  reason rather than a rule.

The masker blanks comments and RAW strings but KEEPS ordinary string literals —
it has to, because the thing being matched IS a string literal. Raw strings are
blanked because embedded-Rust-source fixtures are a measured class here
(`platform_dead_code_audit`, `wasm32_surface_audit`, `subprocess_encoding_gate`
all met it). A `.join("…")` spelled inside an ordinary escaped string literal
would phantom-match; that direction is a LOUD unpinned red a human resolves,
never a silent green.
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
from pathlib import Path

import console_safe  # noqa: E402
from tracked_walk import tracked_files  # noqa: E402

console_safe.apply()

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PINS = HERE / "shared_temp_path_expected.txt"
GLOB = "*.rs"

# Two floors, failing differently (the console_encoding_gate pattern):
#   MIN_TEMP_SITES  the PREDICATE. A regex regression that stops matching
#                   `env::temp_dir()` reports every pinned row stale and every
#                   live row absent — and with an empty pin file would print a
#                   confident green over zero of them.
#   MIN_FILES       the WALK. A tracked_files regression shrinks the population
#                   underneath the same green verdict.
# Measured at the Issue-832 landing: 40 `env::temp_dir()` call sites over 16
# tracked files. The floors sit well below that so ordinary churn does not
# re-pin them, while a collapse still reds.
MIN_TEMP_SITES = 20
MIN_FILES = 8

# Any `env::temp_dir()` call — the POPULATION, which the floors measure. Both
# the `std::env::` and bare `env::` spellings occur in this tree.
TEMP_DIR = re.compile(r"\benv::temp_dir\(\)")

# The DEFECT: that call joined directly with a string literal — unless the
# literal is a fixed DIRECTORY whose very next join is the process-unique
# filename (`temp_dir().join("app").join(format!("f_{}", process::id()))`):
# every process then owns its own file, which is the repair's own shape one
# level down. Measured specimen: seal-remake crates/seal-view/src/quest_talk.rs
# `default_snapshot`, a false positive under the bare predicate. `[^;]` bounds
# the lookahead to the one statement.
FIXED_JOIN = re.compile(
    r"\benv::temp_dir\(\)\s*\.join\(\s*\"([^\"]*)\"\s*\)"
    r"(?!\s*\.join\(\s*format!\([^;]*?process::id\(\))"
)


# ── the masker ──────────────────────────────────────────────────────────────


def mask(src: str) -> str:
    """Blank comments and RAW strings; keep ordinary string literals.

    Ordinary literals must survive — the matched argument is one. Raw strings
    are blanked because embedded Rust source used as test DATA is a measured
    false-positive class in this workspace.
    """
    out: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        # raw string: r"..." or r#"..."# (any hash count)
        if c == "r" and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            j = i + 1
            hashes = 0
            while j < n and src[j] == "#":
                hashes += 1
                j += 1
            if j < n and src[j] == '"':
                close = '"' + "#" * hashes
                end = src.find(close, j + 1)
                end = n if end < 0 else end + len(close)
                out.append(" " * (end - i))
                i = end
                continue
        if c == '"':
            # ordinary string literal — KEPT verbatim, escapes respected
            j = i + 1
            out.append(c)
            while j < n:
                if src[j] == "\\" and j + 1 < n:
                    out.append(src[j : j + 2])
                    j += 2
                    continue
                out.append(src[j])
                if src[j] == '"':
                    j += 1
                    break
                j += 1
            i = j
            continue
        if c == "'":
            # char literal (or a lifetime — either way it cannot open a string)
            out.append(c)
            i += 1
            continue
        if c == "/" and src[i + 1 : i + 2] == "/":
            end = src.find("\n", i)
            end = n if end < 0 else end
            out.append(" " * (end - i))
            i = end
            continue
        if c == "/" and src[i + 1 : i + 2] == "*":
            depth = 1
            j = i + 2
            while j < n and depth:
                if src[j : j + 2] == "/*":
                    depth += 1
                    j += 2
                elif src[j : j + 2] == "*/":
                    depth -= 1
                    j += 2
                else:
                    j += 1
            out.append(" " * (j - i))
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out)


# ── the classifier ──────────────────────────────────────────────────────────


def sites(src: str) -> dict[str, int]:
    """Per-literal counts of fixed-path joins in one source."""
    counts: dict[str, int] = {}
    for m in FIXED_JOIN.finditer(mask(src)):
        lit = m.group(1)
        counts[lit] = counts.get(lit, 0) + 1
    return counts


def temp_site_count(src: str) -> int:
    """Every `env::temp_dir()` call — the population the floors measure."""
    return len(TEMP_DIR.findall(mask(src)))


def scan(root: Path) -> tuple[dict[str, dict[str, int]], int, int]:
    """({relpath: {literal: count}}, total temp_dir sites, files carrying one)."""
    found: dict[str, dict[str, int]] = {}
    n_temp = 0
    n_files = 0
    files, _excluded = tracked_files(root, GLOB)
    for path in files:
        src = path.read_text(encoding="utf-8", errors="replace")
        here = temp_site_count(src)
        if here:
            n_temp += here
            n_files += 1
        counts = sites(src)
        if counts:
            found[path.relative_to(root).as_posix()] = counts
    return found, n_temp, n_files


# ── the pins ────────────────────────────────────────────────────────────────


def parse_pins(path: Path) -> dict[str, str]:
    """`relpath::literal::count # reason` — a reasonless row is REFUSED (a row
    nobody had to justify is a backlog wearing a pin, Issue 785)."""
    pins: dict[str, str] = {}
    if not path.exists():
        return pins
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" not in line:
            raise SystemExit(f"⛔ {path.name}: reasonless pin row refused: {line!r}")
        key, reason = line.split("#", 1)
        parts = key.strip().split("::")
        if len(parts) != 3 or not parts[2].strip().isdigit():
            raise SystemExit(f"⛔ {path.name}: malformed pin row: {line!r}")
        pins[f"{parts[0].strip()}::{parts[1].strip()}"] = (
            f"{int(parts[2])}#{reason.strip()}"
        )
    return pins


def verdict(
    root: Path, pins: dict[str, str], floors: bool = True
) -> tuple[list[str], int, int, int]:
    fails: list[str] = []
    live, n_temp, n_files = scan(root)
    n_fixed = sum(sum(c.values()) for c in live.values())

    # The floors assert the REAL tree's walk + predicate. Fixture arms pass
    # floors=False: a one-file fixture would otherwise trip them and turn every
    # pin assertion into a floor failure.
    if floors:
        if n_temp < MIN_TEMP_SITES:
            fails.append(
                f"⛔ predicate floor: {n_temp} env::temp_dir() site(s) < MIN_TEMP_SITES={MIN_TEMP_SITES}"
            )
        if n_files < MIN_FILES:
            fails.append(
                f"⛔ walk floor: {n_files} file(s) carrying env::temp_dir() < MIN_FILES={MIN_FILES}"
            )

    for rel, counts in sorted(live.items()):
        for lit, count in sorted(counts.items()):
            key = f"{rel}::{lit}"
            if key not in pins:
                fails.append(
                    f"✗ UNPINNED fixed temp path: {key} x{count} — "
                    f"use join(format!(\"...{{}}\", std::process::id())) or pin it with a reason"
                )
            else:
                want, _reason = pins[key].split("#", 1)
                if int(want) != count:
                    fails.append(f"✗ STALE pin {key}: pinned x{want}, live x{count}")

    for key in sorted(pins):
        rel, lit = key.rsplit("::", 1)
        if live.get(rel, {}).get(lit) is None:
            fails.append(f"✗ STALE pin {key}: site no longer in the population")
    return fails, n_fixed, n_temp, n_files


# ── the arms (unconditional) ────────────────────────────────────────────────


def mask_arms() -> list[str]:
    """Arms on `mask` ITSELF, not merely through `sites`.

    `sites` only notices a masker slip when it moves a MATCH, so every bound
    inside the lexer that merely mis-sizes a blank run reads as arm-covered
    while distinguishing nothing — 23 of them did, on this gate's first
    `arm_reach_gate` run. These assert the function's actual contract: blank
    comments and RAW strings, keep ordinary literals, and — the invariant that
    kills an off-by-one anywhere in the walk — **preserve length**, because the
    masked copy's offsets have to stay the source's offsets.
    """
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    corpus = [
        "",
        "let a = 1;",
        "a // comment\nb",
        "a /* c */ b",
        "a /* c /* d */ e */ b",
        'let s = "ab";',
        'let s = "a\\"b";',
        'let s = r#"ab"#;',
        'let s = r"ab";',
        'let s = r##"a"#b"##;',
        "let c = 'x';",
        "let a = b / c;",
        'let s = r#"unterminated',
        "// trailing comment with no newline",
        "/* unterminated block",
        'env::temp_dir().join("x.bin")',
    ]
    for s in corpus:
        check(
            len(mask(s)) == len(s),
            f"mask must preserve length for {s!r}: {len(mask(s))} != {len(s)}",
        )

    # Exact outputs — a blank run one short or one long fails these even where
    # the length invariant is restored by a compensating slip elsewhere.
    check(mask("a // b") == "a     ", f"line comment blanked exactly: {mask('a // b')!r}")
    check(
        mask("a /* b */ c") == "a         c",
        f"block comment blanked exactly: {mask('a /* b */ c')!r}",
    )
    check(mask('r#"ab"#c') == "       c", f"raw string blanked exactly: {mask(chr(114) + chr(35) + chr(34) + 'ab' + chr(34) + chr(35) + 'c')!r}")
    check(mask('r"ab"c') == "     c", f"hashless raw blanked exactly: {mask('r' + chr(34) + 'ab' + chr(34) + 'c')!r}")
    check(mask('"ab"c') == '"ab"c', "an ordinary literal is kept verbatim")
    check(mask('"a\\"b"c') == '"a\\"b"c', "an escaped quote keeps the literal intact")
    check(
        mask("a\n// b\nc") == "a\n    \nc",
        f"a line comment stops at its newline: {mask(chr(97) + chr(10) + '// b' + chr(10) + chr(99))!r}",
    )
    # The `r` prefix opens a raw string only at an identifier boundary.
    check(mask('ptr"a"') == 'ptr"a"', "an identifier ending in r must not open a raw string")
    # Nested block comments close at the OUTER `*/`.
    check(
        mask("/* a /* b */ c */d") == " " * 17 + "d",
        f"nested block comment closes at the outer end: {mask('/* a /* b */ c */d')!r}",
    )
    # An unterminated raw string blanks to EOF rather than raising.
    check(mask('r#"abc') == " " * 6, "an unterminated raw string blanks to EOF")
    # A `#` run that is NOT followed by a quote is not a raw string opener.
    check(mask("r#ident") == "r#ident", "a bare r# is not a raw-string opener")
    # ── EOF boundaries ──────────────────────────────────────────────────
    # Every `j < n` bound in the lexer is only distinguishable by an input
    # that ENDS inside the construct it guards. Under `<=`, `src[j]` at j == n
    # raises — which the arm reports as a failure rather than a silent green,
    # and that IS the distinction. A corpus of well-formed inputs cannot reach
    # any of them, which is why 23 of this module's first survivors sat here.
    for s in [
        "r",
        "r#",
        'r#"',
        'r#"a',
        '"',
        '"a',
        '"a\\',
        "/",
        "/*",
        "/* a",
        "//",
        "a /",
        "'",
    ]:
        check(
            len(mask(s)) == len(s),
            f"mask must survive an input ending inside a construct: {s!r}",
        )

    # ── the raw-string PREFIX rule, both operands ───────────────────────
    # `r` opens a raw string only at an identifier boundary. These separate
    # the two halves of that condition from each other.
    check(
        mask('a"bc"') == 'a"bc"',
        "a non-r char at index 0 must not open a raw string",
    )
    check(
        mask('_r"x"') == '_r"x"',
        "an underscore before r keeps it inside an identifier",
    )
    check(
        mask('1r"x"') == '1r"x"',
        "a digit before r keeps it inside an identifier",
    )
    # ...and the positive case: at a real boundary it DOES open one.
    check(
        mask(' r"x"y') == "     y",
        "r after a space opens a raw string",
    )

    # ── the hash run ────────────────────────────────────────────────────
    check(
        mask('r##"a"#b"##c') == " " * 11 + "c",
        "a two-hash raw string closes only on two hashes",
    )

    # ── the POSITION of a construct, not just its size ──────────────────
    # Every arm above this point asserts a length or an output whose blank run
    # happens to START where the mutant also starts one. Measured on this
    # module's second `arm_reach_gate` run: 11 survivors, ALL in `mask`, and
    # each of these three lines is why.
    #
    # `mask("a // b")` cannot separate `c == "/"` from `c != "/"`: at the SPACE
    # before the slashes the mutant's predicate is true and it blanks from
    # there to EOL, which is one character early and the same LENGTH. Only a
    # comment with no preceding character distinguishes them.
    check(
        mask("// b") == "    ",
        f"a line comment at index 0 is blanked: {mask('// b')!r}",
    )
    # And the other operand: `or` makes a lone `/` open a comment, which blanks
    # to EOL and — again — preserves length. The corpus carried `let a = b /
    # c;` for exactly this and only ever checked its length.
    check(
        mask("let a = b / c;") == "let a = b / c;",
        f"a division slash is not a comment: {mask('let a = b / c;')!r}",
    )
    # The ordinary-string scanner's two inner tests have the same property in
    # the other direction: perturbing either makes it consume the literal two
    # characters at a time and run PAST the closing quote, appending every
    # byte verbatim — identical length, identical text, and the difference
    # shows up only in what the scanner then FAILS to mask afterwards.
    check(
        mask('"ab" // c') == '"ab"     ',
        f"a literal ends at its closing quote, so the comment after it is "
        f"still blanked: {mask(chr(34) + 'ab' + chr(34) + ' // c')!r}",
    )
    # The same shape with an escape in the literal, so the backslash test and
    # the quote test are separated from each other.
    check(
        mask('"a\\"b" // c') == '"a\\"b"     ',
        f"an escaped quote does not end the literal, and the comment after "
        f"the real end is blanked: {mask(chr(34) + 'a' + chr(92) + chr(34) + 'b' + chr(34) + ' // c')!r}",
    )
    # A raw string's `j = i + 1` starts the hash run AFTER the `r`; starting it
    # one character early reads whatever precedes the `r`, finds no quote, and
    # silently declines to blank a raw string at all — which is the
    # false-positive class this masker exists for.
    # The width is DERIVED, not hand-counted: a literal here is a second copy
    # of the input's length and this file's own subject is counts that drift
    # from what they describe. (Written after the hand-counted version failed.)
    _raw = ' r#"env::temp_dir().join("x")"#;'
    check(
        mask(_raw) == " " * (len(_raw) - 1) + ";",
        f"a raw string is blanked from its own `r`: {mask(_raw)!r}",
    )
    # The block-comment walk must stop at its closing `*/` and not at EOF.
    check(
        mask("/* c */ x") == "        x",
        f"a block comment ends at its close, not at EOF: {mask('/* c */ x')!r}",
    )
    # ⛔ A comment marker INSIDE a string literal. This is the only shape that
    # separates the literal's terminating `src[j] == '"'` test from its
    # negation: with `!=` the scanner breaks after the literal's FIRST
    # character, and for `"ab" // c` the outer loop then re-enters at the
    # closing quote and blanks the very same trailing comment — identical
    # output, which is how that row survived an arm written to catch it. Here
    # the escaped region is what gets wrongly blanked, so the two differ. It is
    # also a REAL shape: a URL or a path literal carries `//`.
    check(
        mask('"a//b"') == '"a//b"',
        f"a // inside a string literal is not a comment: {mask(chr(34) + 'a//b' + chr(34))!r}",
    )

    # ── and the same three, BEHAVIOURALLY ───────────────────────────────
    # `sites` is what the gate actually consumes, and a masker slip that keeps
    # length while moving a boundary is only visible here. Each case pairs a
    # construct that must be INVISIBLE with a live site that must survive it.
    for src_, want in [
        ('// env::temp_dir().join("hidden")', {}),
        ('/* env::temp_dir().join("hidden") */', {}),
        ('let s = r#"env::temp_dir().join("hidden")"#;', {}),
        ('env::temp_dir().join("live") // and a comment', {"live": 1}),
        ('let a = b / c; env::temp_dir().join("live");', {"live": 1}),
        ('let s = "str"; env::temp_dir().join("live");', {"live": 1}),
        ('let s = "a\\"b"; env::temp_dir().join("live");', {"live": 1}),
        ('/* c */ env::temp_dir().join("live");', {"live": 1}),
        ('env::temp_dir().join("one"); env::temp_dir().join("two");',
         {"one": 1, "two": 1}),
        # A literal carrying `//` is a real shape and must round-trip intact.
        ('env::temp_dir().join("a//b");', {"a//b": 1}),
    ]:
        check(
            sites(src_) == want,
            f"sites through the mask for {src_!r}: {sites(src_)} != {want}",
        )

    return fails



def classifier_arms() -> list[str]:
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    check(
        sites('let p = std::env::temp_dir().join("fixed.bin");') == {"fixed.bin": 1},
        "the qualified fixed-path form must classify",
    )
    check(
        sites('let p = env::temp_dir().join("fixed.bin");') == {"fixed.bin": 1},
        "the bare-imported form must classify",
    )
    # The line-wrapped form is exactly what rustfmt produces for a long site,
    # so a line-scoped matcher would miss the defect it most often wears.
    check(
        sites('let p = std::env::temp_dir()\n    .join("fixed.bin");')
        == {"fixed.bin": 1},
        "a wrapped .join must classify",
    )
    check(
        sites('let a = env::temp_dir().join("x.bin");\nlet b = env::temp_dir().join("x.bin");')
        == {"x.bin": 2},
        "two sites sharing a literal must count twice",
    )
    # ── the CORRECT patterns must NOT classify ──────────────────────────
    check(
        sites('let p = env::temp_dir().join(format!("n_{}", std::process::id()));')
        == {},
        "the process-unique form must NOT classify",
    )
    check(
        sites('let p = env::temp_dir().join(unique_temp_id("x"));') == {},
        "a helper-built path must NOT classify",
    )
    # A fixed DIRECTORY followed by the process-unique filename is safe; the
    # same directory followed by a FIXED filename (or a pid-free format!) is
    # not — the lookahead must carve out exactly the first.
    check(
        sites('let p = env::temp_dir()\n    .join("app")\n    .join(format!("f_{}.json", std::process::id()));')
        == {},
        "a fixed dir + process-unique file must NOT classify",
    )
    check(
        sites('let p = env::temp_dir().join("app").join("f.json");') == {"app": 1},
        "a fixed dir + fixed file must classify",
    )
    check(
        sites('let p = env::temp_dir().join("app").join(format!("f_{}", n));\nlet q = std::process::id();')
        == {"app": 1},
        "the process-unique lookahead must not reach past the statement",
    )
    check(
        sites("let p = env::temp_dir().join(name);") == {},
        "a variable argument must NOT classify",
    )
    check(sites("") == {}, "empty source has no sites")
    # ── masking ─────────────────────────────────────────────────────────
    check(
        sites('// env::temp_dir().join("fixed.bin") in prose') == {},
        "a line comment must NOT classify",
    )
    check(
        sites('/* env::temp_dir().join("fixed.bin") */') == {},
        "a block comment must NOT classify",
    )
    check(
        sites('let f = r#"env::temp_dir().join("fixed.bin")"#;') == {},
        "a RAW-string fixture must NOT classify",
    )
    check(
        sites('let f = r"env::temp_dir().join(\\"x\\")";') == {},
        "a hashless raw string must NOT classify",
    )
    # A raw string must not swallow the rest of the file: the live site AFTER
    # it has to survive. Kills the `end < 0 -> end >= 0` and close-length
    # mutants, both of which blank to EOF and report a silent green.
    check(
        sites('let f = r#"data"#;\nlet p = env::temp_dir().join("fixed.bin");')
        == {"fixed.bin": 1},
        "a closed raw string must not mask the code after it",
    )
    # Same for a block comment and a line comment.
    check(
        sites('/* note */ let p = env::temp_dir().join("fixed.bin");')
        == {"fixed.bin": 1},
        "a closed block comment must not mask the code after it",
    )
    check(
        sites('// note\nlet p = env::temp_dir().join("fixed.bin");') == {"fixed.bin": 1},
        "a line comment must end at its newline",
    )
    # A lone `/` is division, not a comment.
    check(
        sites('let a = b / c;\nlet p = env::temp_dir().join("fixed.bin");')
        == {"fixed.bin": 1},
        "a division slash must not start a comment",
    )
    # An ordinary string must close so the code after it stays live — the
    # masker KEEPS these, and an escape-handling slip would run it to EOF.
    check(
        sites('let s = "a\\"b";\nlet p = env::temp_dir().join("fixed.bin");')
        == {"fixed.bin": 1},
        "an escaped quote must not leak an ordinary literal",
    )
    # `r` inside an identifier must not be read as a raw-string prefix.
    check(
        sites('let ptr = "x";\nlet p = env::temp_dir().join("fixed.bin");')
        == {"fixed.bin": 1},
        "an identifier ending in r must not open a raw string",
    )
    # ── the population counter ──────────────────────────────────────────
    check(
        temp_site_count('env::temp_dir().join("a"); std::env::temp_dir();') == 2,
        "both temp_dir spellings must count toward the floor",
    )
    check(
        temp_site_count('// env::temp_dir()') == 0,
        "a commented temp_dir must not count toward the floor",
    )
    return fails


def pin_arms(tmp: Path) -> list[str]:
    """Fixture-tree arms over parse + verdict, both directions."""
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    src = 'fn f() { let p = std::env::temp_dir().join("fixed.bin"); }'
    (tmp / "a.rs").write_text(src, encoding="utf-8")
    good = tmp / "pins.txt"
    good.write_text(
        "# header\n\na.rs::fixed.bin::1 # demo path, meant to be findable\n",
        encoding="utf-8",
    )
    fails0, _, _, _ = verdict(tmp, parse_pins(good), floors=False)
    check(fails0 == [], f"pinned live site must be clean, got {fails0}")

    bad = tmp / "bad.txt"
    bad.write_text("a.rs::fixed.bin::1\n", encoding="utf-8")
    try:
        parse_pins(bad)
        check(False, "reasonless pin row must be refused")
    except SystemExit:
        pass

    bad2 = tmp / "bad2.txt"
    bad2.write_text("a.rs::fixed.bin # no count\n", encoding="utf-8")
    try:
        parse_pins(bad2)
        check(False, "malformed pin row must be refused")
    except SystemExit:
        pass

    (tmp / "b.rs").write_text(
        'fn g() { let q = env::temp_dir().join("other.bin"); }', encoding="utf-8"
    )
    fails1, _, _, _ = verdict(tmp, parse_pins(good), floors=False)
    check(
        any("UNPINNED" in f and "b.rs::other.bin" in f for f in fails1),
        "a new fixed temp path must red UNPINNED",
    )
    (tmp / "b.rs").unlink()

    (tmp / "a.rs").write_text(
        src + '\nfn h() { let r = env::temp_dir().join("fixed.bin"); }',
        encoding="utf-8",
    )
    fails2, _, _, _ = verdict(tmp, parse_pins(good), floors=False)
    check(
        any("STALE" in f and "a.rs::fixed.bin" in f and "x2" in f for f in fails2),
        "count growth must red STALE",
    )

    # Repairing a site reds its pin — the file must not only ever loosen.
    (tmp / "a.rs").write_text(
        'fn f() { let p = env::temp_dir().join(format!("fixed_{}.bin", std::process::id())); }',
        encoding="utf-8",
    )
    fails3, _, _, _ = verdict(tmp, parse_pins(good), floors=False)
    check(
        any("STALE" in f and "a.rs::fixed.bin" in f for f in fails3),
        "a repaired site must red its now-stale pin",
    )
    (tmp / "a.rs").write_text(src, encoding="utf-8")
    return fails


def walk_arms() -> list[str]:
    fails: list[str] = []
    fresh = Path(tempfile.mkdtemp(prefix="shared_temp_walk_"))
    try:
        (fresh / "vendor").mkdir()
        (fresh / "vendor" / "v.rs").write_text(
            'fn f() { env::temp_dir().join("x.bin"); }', encoding="utf-8"
        )
        live, n_temp, n_files = scan(fresh)
        if live != {}:
            fails.append("vendored drop must be excluded from the population")
        if n_temp or n_files:
            fails.append("vendored drop must not count toward the floors")
    finally:
        shutil.rmtree(fresh, ignore_errors=True)
    return fails


def floor_arms(tmp: Path) -> list[str]:
    """The floors are the blindness detector — an arm must watch them fire."""
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    small = tmp / "floor_small"
    small.mkdir()
    (small / "one.rs").write_text(
        'fn f() { env::temp_dir().join("x.bin"); }', encoding="utf-8"
    )
    # Called POSITIONALLY (default floors=True) — proves the default is ON.
    fails1, _, _, _ = verdict(small, {"one.rs::x.bin": "1# tiny fixture"})
    check(
        any("MIN_TEMP_SITES" in f for f in fails1)
        and any("MIN_FILES" in f for f in fails1),
        "a sub-floor fixture must fire both floors",
    )

    # A fixture sized EXACTLY at both floors must be clean — the boundary cell
    # `n < MIN` and `n == MIN` disagree about.
    edge = tmp / "floor_edge"
    edge.mkdir()
    per_file = MIN_TEMP_SITES // MIN_FILES
    extra = MIN_TEMP_SITES - per_file * MIN_FILES
    for fi in range(MIN_FILES):
        n = per_file + (1 if fi < extra else 0)
        # `join(name)` keeps these OUT of the finding set while still counting
        # toward the temp_dir population — so the arm tests the floors alone.
        lines = [f"fn f{fi}_{j}() {{ env::temp_dir().join(name); }}" for j in range(n)]
        (edge / f"m{fi}.rs").write_text("\n".join(lines), encoding="utf-8")
    fails2, _, _, _ = verdict(edge, {})
    check(
        not any("floor" in f for f in fails2),
        f"a fixture at exactly the floors ({MIN_TEMP_SITES} sites / {MIN_FILES} files) must be clean, got {fails2}",
    )
    return fails


def selftest() -> list[str]:
    """The arm `arm_reach_audit` runs (argumentless, by its arm vocabulary):
    it creates its own fixture tree."""
    tmp = Path(tempfile.mkdtemp(prefix="shared_temp_gate_"))
    try:
        return (
            mask_arms()
            + classifier_arms()
            + pin_arms(tmp)
            + walk_arms()
            + floor_arms(tmp)
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    fails: list[str] = []
    fails += selftest()
    vfails, n_fixed, n_temp, n_files = verdict(REPO_ROOT, parse_pins(PINS))
    fails += vfails
    if fails:
        for f in fails:
            print(f)
        print(f"✗ shared_temp_path_gate: {len(fails)} failure(s)")
        return 1
    print(
        f"✓ shared-temp-path gate PASSED — {n_fixed} fixed-path site(s), every row pinned, "
        f"over {n_temp} env::temp_dir() call site(s) (floor {MIN_TEMP_SITES}) "
        f"in {n_files} tracked .rs file(s) (floor {MIN_FILES}). "
        f"⚠ It does NOT see a temp_dir() bound to a variable first, nor other "
        f"fixed-scratch spellings (Issue 832 T4)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
