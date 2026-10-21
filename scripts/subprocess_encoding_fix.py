#!/usr/bin/env python3
"""Pin the PARENT's decode on text-mode subprocess calls — the repair half of
`subprocess_encoding_gate.py`'s DECODE class (Issue 778).

`subprocess.run(..., text=True)` decodes the child's pipe with the SYSTEM
locale. On a cp874 box that is silent mojibake, or a decode error raised in
subprocess's reader THREAD — `run()` returns normally with `stdout = None`.
The repair appends `encoding="utf-8", errors="replace"`: `errors="replace"`
because a strict decode reproduces exactly that reader-thread `None`, and the
repo's own correct sites already carry the pair. A call that already names
`errors=` gets `encoding=` alone — its author chose a policy.

The SITE predicate is the gate's own (`_is_subprocess_call`, `_kw_true`,
`_has_kw`, IMPORTED): a repair pass whose idea of a site differs from the
gate's either misses rows the gate reds on or edits code it never asked
about (the `shared_temp_path_fix.py` precedent). The byte insertion is
`locale_io_fix.insert_kwarg`, likewise imported.

The CHILD-ENCODER class (a `sys.executable` spawn with no PYTHONIOENCODING in
its `env=`) is NOT repaired here: the fix merges into an `env=` that may or may
not exist, may be a name or a literal, and is a per-site read.

    scripts/subprocess_encoding_fix.py --list  a.py ...   # report only
    scripts/subprocess_encoding_fix.py         a.py ...   # rewrite in place
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import console_safe  # noqa: E402

console_safe.apply()

from locale_io_fix import insert_kwarg  # noqa: E402
from subprocess_encoding_gate import (  # noqa: E402
    _has_kw,
    _is_subprocess_call,
    _kw_true,
)

BOTH = b'encoding="utf-8", errors="replace"'
ENC_ONLY = b'encoding="utf-8"'


def sites(src: str) -> list[ast.Call]:
    """The gate's DECODE predicate, as Call nodes. A `**kwargs` splat is
    UNKNOWN (it may carry `encoding`) and is left alone."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and _is_subprocess_call(node)):
            continue
        if any(k.arg is None for k in node.keywords):
            continue
        if ((_kw_true(node, "text") or _kw_true(node, "universal_newlines"))
                and not _has_kw(node, "encoding")):
            out.append(node)
    return out


def repair(src: str) -> tuple[str, int]:
    """(rewritten source, sites repaired). Idempotent."""
    calls = sites(src)
    has_errors = [c for c in calls if _has_kw(c, "errors")]
    plain = [c for c in calls if not _has_kw(c, "errors")]
    # Two passes over disjoint call sets; each re-parses so the second pass's
    # offsets describe the first pass's output.
    src, n1 = insert_kwarg(src, plain, BOTH)
    src, n2 = insert_kwarg(src, [c for c in sites(src) if _has_kw(c, "errors")]
                           if has_errors else [], ENC_ONLY)
    return src, n1 + n2


def selftest() -> list[str]:
    fails = []

    def one(src, want_n, want_sub=None):
        got, n = repair(src)
        if n != want_n:
            fails.append(f"{src!r}: repaired {n}, expected {want_n}")
        if want_sub and want_sub not in got:
            fails.append(f"{src!r}: produced {got!r}, missing {want_sub!r}")
        again, m = repair(got)
        if m or again != got:
            fails.append(f"{src!r}: NOT idempotent ({m} more)")
        try:
            ast.parse(got)
        except SyntaxError as e:
            fails.append(f"{src!r}: unparseable output ({e})")

    one("subprocess.run(c, text=True)\n", 1,
        'subprocess.run(c, text=True, encoding="utf-8", errors="replace")')
    one("subprocess.check_output(c, universal_newlines=True)\n", 1)
    one('subprocess.run(c, text=True, errors="strict")\n', 1,
        'errors="strict", encoding="utf-8")')
    one('subprocess.run(c, text=True, encoding="utf-8")\n', 0)
    one("subprocess.run(c, capture_output=True)\n", 0)   # bytes: nothing to decode
    one("subprocess.run(c, text=False)\n", 0)
    one("subprocess.run(c, text=True, **kw)\n", 0)        # splat is UNKNOWN
    one('s = "subprocess.run(c, text=True)"\n', 0)       # a STRING is not a call
    one("subprocess.run(\n    c,\n    text=True,\n)\n", 1, "text=True, encoding")
    one("os.system(c)\n", 0)
    # both classes in one file: the second pass must see the first's output
    one('subprocess.run(a, text=True)\nsubprocess.run(b, text=True, errors="ignore")\n',
        2, 'errors="ignore", encoding="utf-8")')
    return fails


def main(argv: list[str]) -> int:
    fails = selftest()
    if fails:
        for f in fails:
            print(f"  selftest: {f}")
        print("subprocess_encoding_fix SELFTEST FAILED - the rewriter is untrustworthy")
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
            print(f"{'would repair' if listing else 'repaired'} {n:4d}  {a}")
            if not listing:
                p.write_bytes(out.replace("\n", eol).encode("utf-8"))
    print(f"{total} site(s) over {len(args)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
