#!/usr/bin/env python3
"""A tracked `scripts/*.py` that does not IMPORT — Issue 848 T3.

`cross_module_attr_gate.py` (848 T2) is the static half: it resolves every
sibling-module attribute against that module's top-level names. It cannot
reach anything that is not name resolution — a circular import, a missing
third-party dependency, a `raise` in top-level code — because those need an
EXECUTION.

The instrument that does execute every module is `arm_reach_gate.py`, whose
`BASELINE-CRASH` bucket is exactly this failure. It is a **157.6s workstation
verdict** and is deliberately not a CHECK. This gate is the cheap half of that
coverage: import each module once, report which ones die.

## Affordable — measured, and the measurement found its own blocker

Measured 2026-09-19 over the then-86 tracked top-level modules, one
interpreter, each imported in turn — a dated RECORD, not a standing
figure; the live counts are on the PASS line:

    total 6.34s
      6.238s  list_unresolved_percentile_sites
      0.015s  all_ignored_target_audit
      0.10s   everything else, all 85 of them

**98% of the lane was one module** whose entire body was top-level — no
`main()`, no `__main__` guard — so importing it ran a workspace-wide `.rs`
walk and printed to stdout. Guarded in the same change (and `arm_reach_gate`
had been paying that 6.2s once per mutant). The lane is ~0.1s of import plus
one interpreter start.

⛔ **A per-module subprocess was measured too and is NOT worth 3x the cost.**
One process per module is 10.28s against 7.52s for one process importing all
86, and both reported the **identical** single failure. So this runs them in
ONE child, and the cost of that choice is STATED rather than hidden: a module
already imported as somebody else's dependency is cached, so its top-level
code runs once, and a failure caused by a previous import's side effects
would be attributed to the wrong module. Both are acceptable for a predicate
that asks only *does this import at all*; neither is acceptable silently.

## What it does NOT claim

- **not that the module WORKS** — only that `import` completes. A wrong
  constant, a broken predicate and a changed signature all import fine. That
  is `arm_reach_gate`'s question and it stays a workstation verdict.
- **not that a name used at CALL time exists** — that is 848 T2's gate, and
  the 6h40m incident it was written for was exactly that shape. The two are
  complementary and must not be pooled: this one executes and sees little,
  that one parses and sees names everywhere.
- **not a third-party dependency audit.** A module needing `numpy` on a box
  without it is an ENVIRONMENT fact, not a defect, and it is pinned by name
  with a reason rather than being allowed to red every run.

Exemptions are pinned by MEMBERSHIP with a REASON per row
(`scripts/import_health_expected.txt`); a reasonless row is refused, and a row
whose module imports again REDS — a pin file that only ever loosens is a
backlog wearing a pin (Issue 785's rule).

    scripts/import_health_gate.py       # the verdict, per push
    scripts/import_health_gate.py -v    # per-module timings, slowest first

The arms run UNCONDITIONALLY, behind no flag: `docs_gate.sh` invokes each
check as `"$PY" "$script"` with no arguments (Issue 789's finding).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import console_safe  # noqa: E402
from cross_module_attr_gate import tracked_scripts  # noqa: E402

console_safe.apply()

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED = REPO_ROOT / "scripts" / "import_health_expected.txt"

# Two floors, failing differently.
#   MIN_SCRIPTS   the WALK. A glob that matches nothing imports nothing and
#                 prints a confident green over zero modules.
#   MIN_IMPORTED  the RUNNER. A child that dies at startup, a `json` blob that
#                 fails to parse, or a harness that records nothing produces an
#                 EMPTY failure list — which is byte-identical to every module
#                 being healthy. This is the floor that fails in that
#                 direction, and it is the one worth having.
MIN_SCRIPTS = 50
MIN_IMPORTED = 50

# The child program. It reports per module rather than aborting, so one bad
# module does not hide the state of the 85 behind it — and it captures the
# modules' own stdout/stderr, because several call `console_safe.apply()` or
# print a mapping count at import and that output is not this gate's verdict.
_CHILD = r"""
import contextlib, importlib, io, json, sys, time
sys.path.insert(0, SCRIPTS)
rows = []
for m in MODULES:
    buf = io.StringIO()
    t = time.perf_counter()
    err = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            importlib.import_module(m)
        miss = None
    except BaseException as e:            # SystemExit included, deliberately:
        err = f"{type(e).__name__}: {e}"  # a module that EXITS on import is
        miss = getattr(e, "name", None)   # not importable either
    rows.append({"m": m, "err": err, "miss": miss,
                 "s": round(time.perf_counter() - t, 4)})
sys.stdout.write(MARK + json.dumps(rows))
"""

MARK = "\n@@IMPORT_HEALTH@@"


def run_imports(root: Path, modules: list[str]) -> tuple[list[dict], str]:
    """-> `(rows, error)`. `error` non-empty means the RUNNER failed, which is
    never folded into the pass column: a harness that records nothing looks
    exactly like a healthy tree."""
    prog = (f"SCRIPTS = {str(root / 'scripts')!r}\n"
            f"MODULES = {modules!r}\n"
            f"MARK = {MARK!r}\n" + _CHILD)
    try:
        # The env dict is INLINE, not a local: `subprocess_encoding_gate`
        # reads the call site, and a variable there is a CHILD-ENCODER
        # finding — correctly, since the two halves (parent `encoding=` and
        # child `PYTHONIOENCODING`) are pinned as separate classes so that a
        # regression in either is attributable.
        out = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                             encoding="utf-8", errors="replace",
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                             cwd=str(root), timeout=300)
    except subprocess.TimeoutExpired:
        return [], "the import child exceeded 300s — a module blocks at import"
    except OSError as e:
        # The child never STARTED — an unusable cwd, an interpreter that is
        # gone. An exception here would take the gate down with a traceback
        # instead of a verdict, which is Issue 804's class: the failure must
        # be REPORTED, not raised.
        return [], f"the import child could not be started: {e}"
    if MARK not in out.stdout:
        tail = (out.stderr or out.stdout).strip().splitlines()
        return [], ("the import child produced no result blob: "
                    + (tail[-1] if tail else f"rc={out.returncode}"))
    try:
        return json.loads(out.stdout.split(MARK, 1)[1]), ""
    except json.JSONDecodeError as e:
        return [], f"the import child's result blob did not parse: {e}"


def split_failures(rows: list[dict], siblings: set[str]) -> tuple[
        dict[str, str], dict[str, str]]:
    """-> `(broken, missing_dep)`.

    ⛔ MISSING-DEP is its own bucket and is NEVER flagged. A pin would make
    this gate box-dependent in the worst direction: the row is correct on a
    box WITHOUT the package and STALE on one with it, so a green run would
    depend on not having installed something. A missing tracked SIBLING is a
    different statement entirely and stays a finding — it is what a deleted
    module or a renamed one produces, which is the class Issue 848 exists for.

    ⚠ STATED cost: a typo'd stdlib import (`import jsonn`) lands in
    MISSING-DEP too. The alternative is a stdlib allow-list, which goes stale
    every release and fails in the direction that INVENTS findings.
    """
    broken: dict[str, str] = {}
    missing_dep: dict[str, str] = {}
    for r in rows:
        if not r["err"]:
            continue
        miss = r.get("miss")
        if miss and miss.split(".")[0] not in siblings:
            missing_dep[r["m"]] = miss
        else:
            broken[r["m"]] = r["err"]
    return broken, missing_dep


def read_expected(path: Path = EXPECTED) -> dict[str, str]:
    """`<module> = <reason>` rows. A reasonless row is REFUSED, not ignored:
    the reason is the adjudication (Issue 785)."""
    pins: dict[str, str] = {}
    if not path.exists():
        return pins
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            print(f"⛔ {path.name}:{i}: row without a reason — refused")
            raise SystemExit(2)
        mod, reason = line.split("=", 1)
        if not reason.strip():
            print(f"⛔ {path.name}:{i}: empty reason — refused")
            raise SystemExit(2)
        pins[mod.strip()] = reason.strip()
    return pins


def selftest() -> list[str]:
    """Arms over the runner AND over this gate's own pin arithmetic, which a
    runner arm cannot reach (Issue 775's rule)."""
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sc = root / "scripts"
        sc.mkdir()

        def write(name: str, src: str) -> None:
            # Path.write_text's newline= kwarg is 3.10+ (Issue 965: these
            # fixtures must run on the boxes' python3 3.9.6).
            with open(sc / name, "w", encoding="utf-8", newline="") as _f:
                _f.write(src)

        write("ok_mod.py", "X = 1\n")
        write("boom.py", "raise RuntimeError('planted')\n")
        write("missing_dep.py", "import zzz_no_such_module\n")
        write("exits.py", "import sys\nsys.exit(3)\n")
        write("noisy.py", "print('chatter')\nY = 2\n")
        write("guarded.py",
              "def main():\n    raise RuntimeError('never run')\n"
              "if __name__ == '__main__':\n    main()\n")

        mods = ["ok_mod", "boom", "missing_dep", "exits", "noisy", "guarded"]
        rows, err = run_imports(root, mods)
        check(not err, f"the runner failed on a healthy fixture: {err}")
        by = {r["m"]: r["err"] for r in rows}
        check(len(rows) == len(mods),
              f"the runner dropped rows: {len(rows)} of {len(mods)}")
        check(by.get("ok_mod") is None, f"a clean module was flagged: {by}")
        check(by.get("noisy") is None,
              "a module that PRINTS at import was flagged — its chatter is "
              "not this gate's verdict")
        check(by.get("guarded") is None,
              "a __main__-guarded body was executed at import")
        check("RuntimeError" in (by.get("boom") or ""),
              f"a raising module was not caught: {by.get('boom')!r}")
        check("ModuleNotFoundError" in (by.get("missing_dep") or ""),
              f"a missing dependency was not caught by the RUNNER: "
              f"{by.get('missing_dep')!r} — the runner reports every failure "
              f"and `split_failures` decides which are findings")
        # ⛔ SystemExit is a FAILURE here and that is the deliberate reading: a
        # module that exits during import is not importable, and `except
        # Exception` would have let it through as healthy.
        check("SystemExit" in (by.get("exits") or ""),
              f"a module that EXITS at import read as healthy: "
              f"{by.get('exits')!r}")

        # the runner's own failure path: a module list the child cannot even
        # start with must yield an ERROR, never an empty clean list
        rows2, err2 = run_imports(root / "nope", mods)
        check(bool(err2) or all(r["err"] for r in rows2),
              "an unusable root produced a clean-looking empty result")

    # --- the MISSING-DEP / BROKEN split, EXTRACTED so an arm can reach it --
    sib = {"ok_mod", "helper"}
    b, d = split_failures(
        [{"m": "ok_mod", "err": None, "miss": None},
         {"m": "a", "err": "ModuleNotFoundError: numpy", "miss": "numpy"},
         {"m": "b", "err": "ModuleNotFoundError: helper", "miss": "helper"},
         {"m": "c", "err": "RuntimeError: planted", "miss": None},
         {"m": "e", "err": "ModuleNotFoundError: numpy.f2py", "miss": "numpy.f2py"}],
        sib)
    check(set(d) == {"a", "e"},
          f"a third-party dependency was not bucketed MISSING-DEP: {d}")
    check(set(b) == {"b", "c"},
          f"a missing SIBLING or a raising module escaped the finding set: {b}")
    check("ok_mod" not in b and "ok_mod" not in d,
          "a healthy module was bucketed as a failure")
    # ⛔ the SUBMODULE case is the one a naive `miss not in siblings` gets
    # wrong in the silent direction: `numpy.f2py` is not `numpy`, so without
    # the `.split(".")[0]` it reads as a missing sibling and becomes a
    # finding nobody can fix.
    check(d.get("e") == "numpy.f2py",
          f"a dotted dependency name was not resolved to its root: {d}")


    # --- the pin PARSER, reachable now that `read_expected` takes a path ---
    # ⛔ Four refusal branches sat behind a module-level constant, so no arm
    # could hand them a file: unreachable by construction, which is the
    # pattern AGENTS.md records three times — the EXTRACTION is the repair.
    with tempfile.TemporaryDirectory() as td:
        pin = Path(td) / "pins.txt"

        check(read_expected(pin) == {},
              "a MISSING pin file did not read as zero pins")

        with open(pin, "w", encoding="utf-8", newline="") as _p:
            _p.write("# only a comment\n\n")
        check(read_expected(pin) == {},
              "comments and blank lines were parsed as rows")

        with open(pin, "w", encoding="utf-8", newline="") as _p:
            _p.write("a::b = a real reason\n")
        check(read_expected(pin) == {"a::b": "a real reason"},
              "a well-formed row did not parse")

        # ⚠ Both refusals PRINT before they raise, and that output is the
        # arm's expected noise rather than this gate's verdict — swallowed,
        # or a green run carries two `⛔` lines above its own `✓`.
        def _refuses(text: str) -> bool:
            with open(pin, "w", encoding="utf-8", newline="") as _p:
                _p.write(text)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    read_expected(pin)
            except SystemExit:
                return True
            return False

        check(_refuses("a::b\n"),
              "a row with NO `=` was accepted — the reason IS the "
              "adjudication, and a row without one is a backlog wearing a pin")
        check(_refuses("a::b =   \n"),
              "a row with an EMPTY reason was accepted")

    # --- this gate's OWN pin arithmetic, which no runner arm reaches -------
    check(MIN_SCRIPTS > 0 and MIN_IMPORTED > 0, "a floor is non-positive")
    try:
        read_expected()
    except SystemExit as e:  # pragma: no cover - only on a malformed pin file
        fails.append(f"read_expected refused the live pin file: {e}")
    return fails


def main(argv: list[str]) -> int:
    bad = selftest()
    for f in bad:
        print(f"✗ selftest: {f}")
    if bad:
        print(f"⛔ import-health gate SELFTEST FAILED — {len(bad)} arm(s); the "
              f"runner cannot be read as a verdict")
        return 2

    expected = read_expected()
    scripts = tracked_scripts(REPO_ROOT)
    modules = [f.stem for f in scripts]
    rows, err = run_imports(REPO_ROOT, modules)

    if err:
        print(f"⛔ RUNNER: {err} — never folded into the pass column")
        return 2

    broken, missing_dep = split_failures(rows, set(modules))
    unpinned = sorted(set(broken) - set(expected))
    stale = sorted(set(expected) - set(broken))

    for m in unpinned:
        print(f"✗ {m} does not import — {broken[m]}. Python resolves nothing "
              f"until a module is executed, so this is dead the moment "
              f"anything imports it; if the cause is an ENVIRONMENT fact "
              f"rather than a defect, pin it with a reason in {EXPECTED.name}")
    for m in stale:
        print(f"✗ {m} — pinned as unimportable but it imports now. Drop the "
              f"row: a pin file that only ever loosens stops being a wall")

    if "-v" in argv:
        for r in sorted(rows, key=lambda r: -r["s"])[:10]:
            print(f"  ⏱ {r['s']:7.4f}s  {r['m']}")

    if len(scripts) < MIN_SCRIPTS:
        print(f"✗ walk FLOOR breached: {len(scripts)} tracked scripts/*.py < "
              f"{MIN_SCRIPTS} — the walk went blind and every count above is "
              f"vacuous")
        return 1
    if len(rows) < MIN_IMPORTED:
        print(f"✗ runner FLOOR breached: {len(rows)} module(s) attempted < "
              f"{MIN_IMPORTED} — the harness recorded almost nothing, which "
              f"looks exactly like a healthy tree")
        return 1
    if unpinned or stale:
        return 1

    total = sum(r["s"] for r in rows)
    dep = ""
    if missing_dep:
        pairs = ", ".join(f"{m} needs {d}" for m, d in sorted(missing_dep.items()))
        dep = (f" [{len(missing_dep)} MISSING-DEP, counted and never flagged: "
               f"{pairs} — an ENVIRONMENT fact, and pinning it would red this "
               f"gate on a box that HAS the package]")
    print(f"✓ import-health gate PASSED — {len(rows) - len(missing_dep)} of "
          f"{len(rows)} tracked scripts/*.py import (floors {MIN_SCRIPTS}/"
          f"{MIN_IMPORTED}), {len(expected)} pinned exemption(s), 0 stale, "
          f"{total:.2f}s of import in one child.{dep} ⚠ It does NOT claim a "
          f"module WORKS — only "
          f"that `import` completes; a wrong constant, a broken predicate and "
          f"a changed signature all import fine, and that is "
          f"`arm_reach_gate`'s 157.6s workstation question. A name used at "
          f"CALL time is `cross_module_attr_gate`'s (Issue 848)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
