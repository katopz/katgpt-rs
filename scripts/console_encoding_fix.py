#!/usr/bin/env python3
"""Defend a script's console streams — the repair half of
`console_encoding_gate.py` (Issue 804).

A script that prints a non-ASCII glyph (✓ ✗ ⛔ —) on a non-UTF-8 console
(the cp874 4090) dies with UnicodeEncodeError on the first one: no verdict at
all. The repair inserts the module-level form 42 workspace instruments already
inline, right after the leading import block:

    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

— the glyph degrades, the verdict survives. `console_safe.apply()` is the
other accepted form, but it is a katgpt-rs module and a sibling repo cannot
import it. The DEFENDED predicate is the gate's own `is_defended`, IMPORTED:
an already-defended file is never touched, so the tool is idempotent by the
same rule the gate reads.

    scripts/console_encoding_fix.py --list  a.py ...   # report only
    scripts/console_encoding_fix.py         a.py ...   # rewrite in place
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import console_safe  # noqa: E402

console_safe.apply()

from console_encoding_gate import is_defended  # noqa: E402

BLOCK = (
    "# A non-UTF-8 console (cp874) would kill this script on its first\n"
    "# non-ASCII glyph; degrade the glyph, never the verdict (katgpt-rs\n"
    "# Issue 804). A captured stream without reconfigure is left alone.\n"
    "for _stream in (sys.stdout, sys.stderr):\n"
    "    try:\n"
    '        _stream.reconfigure(errors="backslashreplace")\n'
    "    except (AttributeError, ValueError):\n"
    "        pass\n"
)


def _imports_sys(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == "sys" and a.asname is None
                                                for a in node.names):
            return True
    return False


def repair(src: str) -> tuple[str, int]:
    """(rewritten source, 1 if repaired else 0). Idempotent via the gate's
    own predicate. The block goes after the LEADING run of docstring +
    imports — the first statement that is neither is where code starts, and
    a block placed after a later import would run after code that prints."""
    if is_defended(src):
        return src, 0
    tree = ast.parse(src)
    end = 0  # line number (1-based) the block follows; 0 = top of file
    for i, node in enumerate(tree.body):
        doc = (i == 0 and isinstance(node, ast.Expr)
               and isinstance(node.value, ast.Constant)
               and isinstance(node.value.value, str))
        if doc or isinstance(node, (ast.Import, ast.ImportFrom)):
            end = node.end_lineno
            continue
        break
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    head = "" if _imports_sys(tree) else "import sys\n"
    insert = "\n" + head + BLOCK + "\n"
    out = "".join(lines[:end]) + insert + "".join(lines[end:]).lstrip("\n")
    return out, 1


def selftest() -> list[str]:
    fails = []

    def one(src, want_n, want_sub=None):
        got, n = repair(src)
        if n != want_n:
            fails.append(f"{src!r}: repaired {n}, expected {want_n}")
        if want_sub and want_sub not in got:
            fails.append(f"{src!r}: produced {got!r}, missing {want_sub!r}")
        if n and not is_defended(got):
            fails.append(f"{src!r}: the gate does not credit the repair")
        again, m = repair(got)
        if m or again != got:
            fails.append(f"{src!r}: NOT idempotent")
        try:
            ast.parse(got)
        except SyntaxError as e:
            fails.append(f"{src!r}: unparseable output ({e})")

    one('import sys\nprint("✓")\n', 1, 'import sys\n\n# A non-UTF-8')
    # no `import sys`: the block brings its own
    one('"""doc."""\nimport json\nprint("✓")\n', 1, "import sys\n# A non-UTF-8")
    # a `from __future__` stays FIRST (a SyntaxError otherwise)
    one('from __future__ import annotations\nimport sys\nprint("✓")\n', 1)
    # already defended: untouched
    one('import sys\nsys.stdout.reconfigure(errors="replace")\nprint("✓")\n', 0)
    # the block lands BEFORE the first code statement
    got, _ = repair('import sys\nX = 1\nimport os\nprint("✓")\n')
    if got.index("_stream.reconfigure") > got.index("X = 1"):
        fails.append("block placed after code")
    return fails


def main(argv: list[str]) -> int:
    fails = selftest()
    if fails:
        for f in fails:
            print(f"  selftest: {f}")
        print("console_encoding_fix SELFTEST FAILED - the rewriter is untrustworthy")
        return 2
    args = [a for a in argv if not a.startswith("--")]
    listing = "--list" in argv
    total = 0
    for a in args:
        p = Path(a)
        # The newline style belongs to the file (locale_io_fix's CRLF lesson).
        raw = p.read_bytes()
        eol = "\r\n" if b"\r\n" in raw else "\n"
        src = raw.decode("utf-8").replace("\r\n", "\n")
        out, n = repair(src)
        total += n
        if n:
            print(f"{'would repair' if listing else 'repaired'}  {a}")
            if not listing:
                p.write_bytes(out.replace("\n", eol).encode("utf-8"))
    print(f"{total} file(s) repaired of {len(args)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
