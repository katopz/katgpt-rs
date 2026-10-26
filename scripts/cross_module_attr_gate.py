#!/usr/bin/env python3
"""A tracked `scripts/*.py` naming an attribute a sibling module does not
define — Issue 848.

Python has **no link step**. `import worktree_state` then
`worktree_state.is_checkout(root)` resolves at CALL time, so a rename or a
deletion in the imported module is not an error until somebody EXECUTES the
importer. Measured 2026-09-19: `6c6ca2ee` privatised `is_checkout` and deleted
`worktree_fixture` while reworking that module's own arms, updating the seven
in-module callers and neither of the two external ones. Both external callers
are **docs-gate CHECKS**, and both died with

    AttributeError: module 'worktree_state' has no attribute 'is_checkout'

for **6h40m** on `develop`. Not a wrong verdict — NO verdict, which is Issue
804's class one seam over: a gate that dies has findings that are not
*unknown* but *unlooked at*.

⛔ **Both names carried a written contract naming their consumers**, and the
contract is the reason this is a gate rather than a style note. `AGENTS.md`
says *"`worktree_state.is_checkout` — **delegate to it too**"*; the deleted
fixture's own docstring said *"Public, because the three delegating consumers
each owe an arm … and that arm needs exactly this fixture."* The document said
the right thing; nothing read it. That is `instrument_reachability_gate`'s
finding one level down — a census over a document cannot enforce the document.

## Why a per-push GATE and not a workstation report

The general instrument already exists and already executes every module:
`arm_reach_gate.py` reports `BASELINE-CRASH` on a module that will not import.
It is kept OUT of the CHECKS set for a measured reason — **157.6s** against the
docs gate's ~13s CPU budget. This pass is static, reads each file once, and
costs a fraction of a second over the tracked `scripts/` tree, so the
cheap half of that coverage can ride the per-push lane while the
expensive half stays a workstation verdict. ⚠ The walk size and the
resolved-reference count are printed on the PASS line every run; no
figure is typed here, because both move whenever anybody adds a script.

## Two-sided known answer, which is what makes it shippable

| ref | findings |
|---|---|
| `6c6ca2ee~1` (before the break) | **0** |
| `6c6ca2ee` (after it) | **4** — two names x two consumers, no others |

`--prove-fires` runs exactly that, opt-in on the
`platform_dead_code_floor_gate` precedent: ~1.5s of `git archive` to re-prove
a fact about a frozen commit is worth a workstation run and not a per-push one.

## STATED blind spots, printed on the verdict line

Printed rather than remembered, because a later census that re-derives them
re-derives them wrong (Issue 832's rule, and `shared_temp_path_gate`'s PASS
line is the precedent):

- **`getattr(mod, "name")`** — a dynamic lookup this pass cannot resolve, and
  widening to it would need a constant-folding pass for the second argument.
- **a star-import** in the IMPORTED module: `from x import *` makes its
  top-level namespace un-enumerable from the AST, so such a module is marked
  OPAQUE and nothing is flagged against it. Never silently — the count rides
  the verdict line, because an opaque module is coverage this gate does not
  have.
- **a name bound only inside a function** (`global`), and **`__all__`**, which
  this pass does not read: it constrains `import *`, not attribute access, so
  reading it would narrow the defined set and INVENT findings.
- **runtime breaks that are not name resolution** — a changed signature, a
  moved constant's VALUE. That is Issue 848 T3 and needs an execution, not a
  parse.

Exemptions are pinned by MEMBERSHIP with a REASON per row
(`scripts/cross_module_attr_expected.txt`); a reasonless row is refused and a
row whose site is gone REDS — a pin file that only ever loosens is a backlog
wearing a pin (Issue 785's rule). The file is **deliberately empty**: the
default for a real dangling reference is to fix it or to make the name public,
not to add a row.

    scripts/cross_module_attr_gate.py                  # the verdict, per push
    scripts/cross_module_attr_gate.py --workspace      # the T4 census, exit 0
    scripts/cross_module_attr_gate.py --prove-fires 6c6ca2ee

The arms run UNCONDITIONALLY, behind no flag: `docs_gate.sh` invokes each check
as `"$PY" "$script"` with no arguments, so an arm behind `'--canary' in
sys.argv` never fires on a push (Issue 789's finding).
"""

from __future__ import annotations

import contextlib
import io
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import console_safe  # noqa: E402
import worktree_state  # noqa: E402

console_safe.apply()

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED = REPO_ROOT / "scripts" / "cross_module_attr_expected.txt"
GLOB = "*.py"

# The commit that broke it, and the known answer `--prove-fires` adjudicates
# against. Not a default argument the gate ever uses on its own.
PROVE_SHA = "6c6ca2ee"

# Two floors, failing differently.
#   MIN_SCRIPTS   the WALK. A glob that matches nothing reports every module
#                 clean and prints a confident green over zero of them.
#   MIN_RESOLVED  the IMPORT RESOLUTION — the number of sibling-module
#                 attribute references this pass actually resolved. If the
#                 alias table stops being built (a refactor of `_aliases`, a
#                 changed AST shape) every file resolves nothing, every name
#                 is trivially fine, and the output is byte-identical to a
#                 clean repo. This is the floor that fails in that direction.
MIN_SCRIPTS = 50
MIN_RESOLVED = 200


def tracked_scripts(root: Path) -> list[Path]:
    """The family, as git tracks it — a scratch copy left in `scripts/` is not
    a member of the contract, and only git can say so. An extracted tree with
    no `.git` is a legitimate population, not an error (`tracked_walk`'s rule),
    so it falls back to the filesystem.

    The checkout probe is `worktree_state.is_checkout`, DELEGATED — the very
    contract Issue 848 exists because a rename broke. `.exists()`, not
    `.is_dir()`: a `git worktree` has a `.git` FILE and is a perfectly good
    checkout, and the `.is_dir()` spelling silently falls to the glob and
    measures a DIFFERENT population (Issue 836).
    """
    sc = root / "scripts"
    if not worktree_state.is_checkout(root):
        return sorted(sc.glob(GLOB))
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "scripts/" + GLOB],
        capture_output=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        return sorted(sc.glob(GLOB))
    # `ls-files scripts/*.py` matches nested paths too (`scripts/kimi_ref/…`);
    # those are reference dumps, not instruments, and their own directory is
    # the boundary. Only the top level is in scope.
    return sorted(root / p for p in out.stdout.split()
                  if p.strip() and p.count("/") == 1)


def _bound_by(node: ast.AST) -> set[str]:
    """Top-level names one statement binds.

    Re-exports count: `from tracked_walk import tracked_files` makes
    `mod.tracked_files` a real attribute of `mod`, and refusing to credit it
    would invent a finding at every delegation seam — which is the pattern
    this repo's own DRY rule (Issue 755) pushes instruments toward.
    """
    names: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        if isinstance(node.target, ast.Name):
            names.add(node.target.id)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for a in node.names:
            if a.name != "*":
                names.add(a.asname or a.name.split(".")[0])
    return names


def defined_names(tree: ast.Module) -> tuple[set[str], bool]:
    """-> `(top-level names, opaque)`.

    `opaque` is True when the module carries a star-import: its namespace then
    contains whatever the source module exports, which no parse of THIS file
    can enumerate. An opaque module is credited with everything rather than
    flagged — the conservative direction for a gate whose false positive is a
    demand to "fix" working code.

    ⛔ A conditional definition still DEFINES the name, so the walk descends
    into top-level `if` / `try` / `with` / `for`. Refusing to would flag every
    `try: import x / except ImportError: x = None` fallback in the family.
    Function BODIES are not descended into: a name bound only inside a function
    is not a module attribute (`global` is the stated blind spot).
    """
    names: set[str] = set()
    opaque = False

    def walk(body) -> None:
        nonlocal opaque
        for node in body:
            names.update(_bound_by(node))
            if isinstance(node, ast.ImportFrom) and any(
                    a.name == "*" for a in node.names):
                opaque = True
            if isinstance(node, (ast.If, ast.Try, ast.With, ast.For,
                                 ast.While, ast.AsyncWith, ast.AsyncFor)):
                for attr in ("body", "orelse", "finalbody", "handlers"):
                    part = getattr(node, attr, None)
                    if not part:
                        continue
                    if attr == "handlers":
                        for h in part:
                            walk(h.body)
                    else:
                        walk(part)

    walk(tree.body)
    return names, opaque


def _aliases(tree: ast.Module, mods: set[str]) -> dict[str, str]:
    """local binding -> sibling module stem, for `import X` / `import X as Y`.

    Only siblings in `mods` are tracked: `import os` is not this gate's
    business, and claiming to know `os`'s surface from a parse would be the
    cries-wolf instrument AGENTS.md warns about.
    """
    alias: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                stem = a.name.split(".")[0]
                if stem in mods:
                    alias[a.asname or stem] = stem
    return alias


def scan(root: Path) -> tuple[list[str], list[str], int, int, int, int]:
    """-> `(findings, unparsed, n_scripts, n_resolved, n_getattr, n_opaque)`.

    A finding key is `<file>::<module>.<attr>` — **LINE-FREE** on purpose. A
    line number drifts on every edit above it, so a line-keyed pin file reds on
    commits that changed nothing about it, and a pin file that reds on noise is
    one people delete (`arm_reach_gate`'s recorded rule).
    """
    scripts = tracked_scripts(root)
    mods = {f.stem for f in scripts}
    trees: dict[Path, ast.Module] = {}
    unparsed: list[str] = []
    defined: dict[str, set[str]] = {}
    opaque: set[str] = set()

    for f in scripts:
        try:
            trees[f] = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            unparsed.append(f"{f.name}: {e}")
            continue
        names, is_opaque = defined_names(trees[f])
        defined[f.stem] = names
        if is_opaque:
            opaque.add(f.stem)

    findings: list[str] = []
    resolved = 0
    n_getattr = 0
    for f, tree in trees.items():
        alias = _aliases(tree, mods)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "getattr":
                n_getattr += 1
            if isinstance(node, ast.ImportFrom) and node.level == 0 \
                    and node.module in defined and node.module not in opaque:
                for a in node.names:
                    if a.name == "*":
                        continue
                    resolved += 1
                    if a.name not in defined[node.module]:
                        findings.append(f"{f.name}::{node.module}.{a.name}")
            if isinstance(node, ast.Attribute) \
                    and isinstance(node.value, ast.Name) \
                    and node.value.id in alias:
                stem = alias[node.value.id]
                if stem in opaque:
                    continue
                resolved += 1
                if node.attr not in defined.get(stem, set()):
                    findings.append(f"{f.name}::{stem}.{node.attr}")

    return (sorted(set(findings)), unparsed, len(scripts), resolved,
            n_getattr, len(opaque))


def read_expected(path: Path = EXPECTED) -> dict[str, str]:
    """`<key> = <reason>` rows. A reasonless row is REFUSED, not ignored: the
    reason is the adjudication, and a row without one is a backlog wearing a
    pin (Issue 785)."""
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
        key, reason = line.split("=", 1)
        if not reason.strip():
            print(f"⛔ {path.name}:{i}: empty reason — refused")
            raise SystemExit(2)
        pins[key.strip()] = reason.strip()
    return pins


def selftest() -> list[str]:
    """Arms over the classifier AND over this gate's own pin arithmetic, which
    a classifier arm cannot reach (Issue 775's rule)."""
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    def names_of(src: str) -> tuple[set[str], bool]:
        return defined_names(ast.parse(src))

    # --- the defined-name walk -------------------------------------------
    n, op = names_of("def f():\n    pass\nclass C:\n    pass\nX = 1\n")
    check(n == {"f", "C", "X"}, f"plain top-level names missed: {n}")
    check(not op, "a module with no star-import read as OPAQUE")

    n, _ = names_of("from tracked_walk import tracked_files\nimport os as o\n")
    check(n == {"tracked_files", "o"},
          f"a RE-EXPORT is a real attribute and was not credited: {n}")

    n, _ = names_of("try:\n    import fast\nexcept ImportError:\n"
                    "    fast = None\n")
    check(n == {"fast"}, f"a conditional definition was not credited: {n}")

    n, _ = names_of("if X:\n    def g():\n        pass\nelse:\n    g = None\n")
    check("g" in n, f"a name defined in an if/else branch was missed: {n}")

    n, _ = names_of("def outer():\n    inner = 1\n    def nested():\n"
                    "        pass\n")
    check(n == {"outer"},
          f"a FUNCTION-LOCAL name was credited as a module attribute: {n}")

    _, op = names_of("from helper import *\n")
    check(op, "a star-import did not mark the module OPAQUE")

    n, _ = names_of("A: int = 1\nB, C = 1, 2\n")
    check({"A", "B", "C"} <= n, f"annotated/tuple targets missed: {n}")

    # --- the finding predicate, over a synthetic two-module tree ----------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sc = root / "scripts"
        sc.mkdir()

        def write(name: str, src: str) -> None:
            # Path.write_text's newline= kwarg is 3.10+ (Issue 965: these
            # fixtures must run on the boxes' python3 3.9.6).
            with open(sc / name, "w", encoding="utf-8", newline="") as _f:
                _f.write(src)

        write("lib.py", "def good():\n    pass\n")
        write("user.py", "import lib\nlib.good()\nlib.gone()\n")
        f, u, n_s, res, n_g, n_o = scan(root)
        check(f == ["user.py::lib.gone"],
              f"the missing attribute was not the only finding: {f}")
        check(not u, f"a parseable tree reported UNPARSED: {u}")
        check(res >= 2, f"the resolution counter did not move: {res}")

        # an ALIASED import is the same reference wearing another name
        write("user.py", "import lib as L\nL.gone()\n")
        f, *_ = scan(root)
        check(f == ["user.py::lib.gone"],
              f"`import lib as L` was not resolved: {f}")

        # a from-import of a name that is not there
        write("user.py", "from lib import gone\n")
        f, *_ = scan(root)
        check(f == ["user.py::lib.gone"],
              f"a from-import of an absent name was not flagged: {f}")
        write("user.py", "from lib import good\n")
        f, *_ = scan(root)
        check(not f, f"a from-import of a PRESENT name was flagged: {f}")

        # ⛔ an attribute on something that is not a tracked module is not this
        # gate's business — `PINS.name`, `args.verbose`, `os.sep`. Without this
        # the gate flags every attribute in the family.
        write("user.py", "import os\nos.sep\nPINS.name\nlib = 1\n")
        f, *_ = scan(root)
        check(not f, f"a non-sibling attribute was flagged: {f}")

        # OPAQUE: the imported module cannot be enumerated, so nothing is
        # claimed about it — and the count is DISCLOSED, never silent.
        write("lib.py", "from elsewhere import *\ndef good():\n    pass\n")
        write("user.py", "import lib\nlib.anything()\n")
        f, _u, _n, _r, _g, n_o = scan(root)
        check(not f, f"a finding was claimed against an OPAQUE module: {f}")
        check(n_o == 1, f"the OPAQUE count was not disclosed: {n_o}")

        # getattr is COUNTED, never flagged — the stated blind spot, measured
        write("lib.py", "def good():\n    pass\n")
        write("user.py", 'import lib\ngetattr(lib, "gone")()\n')
        f, _u, _n, _r, n_g, _o = scan(root)
        check(not f, f"a getattr site was flagged rather than counted: {f}")
        check(n_g == 1, f"the getattr count was not disclosed: {n_g}")

        # UNPARSED is never folded into the pass column
        write("user.py", "import lib\ndef (:\n")
        f, u, *_ = scan(root)
        check(len(u) == 1, f"a syntax error was not reported UNPARSED: {u}")


    # --- `tracked_scripts` against a REAL git repo -------------------------
    # ⛔ Every other arm here builds a plain temp directory, where
    # `is_checkout` is False and the GLOB fallback runs — so the `git
    # ls-files` branch, which is what a real run takes, was reached by
    # nothing. One exclusion, two code paths: the `platform_dead_code`
    # vendor-arm failure, where an arm certified the path it was not aimed at.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sc = root / "scripts"
        (sc / "kimi_ref").mkdir(parents=True)
        with open(sc / "top.py", "w", encoding="utf-8", newline="") as _t:
            _t.write("X = 1\n")
        with open(sc / "kimi_ref" / "dump.py", "w", encoding="utf-8",
                  newline="") as _d:
            _d.write("Y = 2\n")
        with open(sc / "untracked.py", "w", encoding="utf-8",
                  newline="") as _u:
            _u.write("Z = 3\n")
        for cmd in (["init", "-q", str(root)],
                    ["-C", str(root), "config", "user.email", "t@t"],
                    ["-C", str(root), "config", "user.name", "t"],
                    ["-C", str(root), "add", "scripts/top.py",
                     "scripts/kimi_ref/dump.py"],
                    ["-C", str(root), "commit", "-qm", "base"]):
            subprocess.run(["git", *cmd], capture_output=True)

        names = [p.name for p in tracked_scripts(root)]
        # the depth filter: a NESTED tracked script is a reference dump, not
        # an instrument, and its own directory is the boundary
        check(names == ["top.py"],
              f"the ls-files branch did not yield exactly the top-level "
              f"tracked script: {names}")
        # and the TRACKED half: an untracked scratch file is not a member of
        # the contract, which is the whole reason this branch exists
        check("untracked.py" not in names,
              "an UNTRACKED scratch script entered the population — only git "
              "can say what the contract covers")

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

    # --- this gate's OWN pin arithmetic, which no classifier arm reaches ---
    check(MIN_SCRIPTS > 0 and MIN_RESOLVED > 0, "a floor is non-positive")
    try:
        read_expected()
    except SystemExit as e:  # pragma: no cover - only on a malformed pin file
        fails.append(f"read_expected refused the live pin file: {e}")
    return fails


def prove_fires(sha: str) -> int:
    """Extract `<sha>~1` and `<sha>` and require 0 findings, then some.

    A known-answer tree, in the `platform_dead_code_audit.py --prove-fires`
    idiom. `6c6ca2ee` privatised `worktree_state.is_checkout` and deleted
    `worktree_fixture`, leaving four dangling references in two docs-gate
    CHECKS; its parent has none.
    """
    with tempfile.TemporaryDirectory() as td:
        for ref, want_any in ((f"{sha}~1", False), (sha, True)):
            tree = Path(td) / ref.replace("~", "_")
            tree.mkdir()
            ar = subprocess.run(["git", "-C", str(REPO_ROOT), "archive", ref,
                                 "scripts"], capture_output=True)
            if ar.returncode != 0:
                print(f"✗ --prove-fires: cannot `git archive {ref}`")
                return 2
            tar = tree / "t.tar"
            tar.write_bytes(ar.stdout)
            subprocess.run(["tar", "-xf", str(tar), "-C", str(tree)],
                           capture_output=True)
            tar.unlink()
            # No `.git` in an archive, and `tracked_scripts` falls back to the
            # glob rather than erroring — but give it one anyway, so the arm
            # exercises the SAME branch a real run takes.
            subprocess.run(["git", "init", "-q", str(tree)], capture_output=True)
            subprocess.run(["git", "-C", str(tree), "add", "-A"],
                           capture_output=True)
            findings, unparsed, n_s, _res, _g, _o = scan(tree)
            if n_s < MIN_SCRIPTS:
                print(f"✗ --prove-fires: {ref} walked {n_s} script(s) — the "
                      f"arm proves nothing")
                return 2
            ok = bool(findings) == want_any and not unparsed
            print(f"  {'✓' if ok else '✗'} {ref}: {len(findings)} finding(s) "
                  f"over {n_s} script(s)")
            for k in findings:
                print(f"      {k}")
            if not ok:
                return 2
    print("✓ --prove-fires: the gate is clean at the parent and reds at the "
          "commit that privatised the delegation target")
    return 0


def workspace_census() -> int:
    """A REPORT (exit 0) — the Issue 848 T4 population, re-derived per run.

    ⛔ **No sweep half, and the measurement is the argument rather than a
    preference.** `check_validation_gate` declined a sweep on a population of
    ONE and was right; `console_encoding_gate` INHERITED that answer and was
    wrong by seven repos. So this was counted, and counting it two ways gives
    two different answers:

    - by FILES, the class looks workspace-wide — 10 of 17 contract repos
      carry tracked `scripts/*.py`, 196 of them.
    - by RESOLVED REFERENCES — the quantity this gate's finding can actually
      come out of — **711 of 747 are in this repo**. riir-train, with 63
      scripts, resolves 32; riir-ai 2; every other repo 0, because their
      `scripts/` are standalone one-offs that import nothing of each other's
      (`instrument_reachability_drift_sweep` measured the same shape from the
      other side: riir-train 61 of 61 unreachable, where the predicate
      over-captures for exactly this reason).

    A sweep would therefore ratchet a bucket that is structurally near-empty,
    and `max_findings = 0` over a population of 36 references in 9 repos is a
    wall nobody can fail. **What would flip the answer is a sibling's
    RESOLVED count growing**, not its file count — which is why this prints
    both, every run, rather than recording either in prose.
    """
    import repo_alias
    from skill_repo_set_gate import derive_repos

    ws = REPO_ROOT.parent
    names = sorted(derive_repos(ws))
    tot_s = tot_r = tot_f = tot_u = 0
    print(f"{'repo':<28}{'scripts':>9}{'resolved':>10}{'findings':>10}"
          f"{'unparsed':>10}")
    for n in names:
        try:
            findings, unparsed, n_s, res, _g, _o = scan(ws / repo_alias.disk(n))
        except OSError as e:
            print(f"{n:<28}{'UNREADABLE':>9}  {e}")
            continue
        tot_s += n_s
        tot_r += res
        tot_f += len(findings)
        tot_u += len(unparsed)
        print(f"{n:<28}{n_s:>9}{res:>10}{len(findings):>10}{len(unparsed):>10}")
        for k in findings:
            print(f"    X {n}: {k}")
    print(f"\n{len(names)} contract repo(s) on this box · {tot_s} script(s) · "
          f"{tot_r} resolved reference(s) · {tot_f} finding(s) · "
          f"{tot_u} unparsed")
    print(f"⚠ Read the RESOLVED column, not the script count: the two "
          f"disagree by an order of magnitude and only the first is this "
          f"class's population. A repo that carries tracked scripts/*.py "
          f"and resolves 0 references cannot produce a finding, so a sweep "
          f"row over it would be a wall nobody can fail (Issue 848 T4)")
    return 0


def main() -> int:
    bad = selftest()
    for f in bad:
        print(f"✗ selftest: {f}")
    if bad:
        print(f"⛔ cross-module-attr gate SELFTEST FAILED — {len(bad)} arm(s); "
              f"the classifier cannot be read as a verdict")
        return 2

    expected = read_expected()
    findings, unparsed, n_scripts, resolved, n_getattr, n_opaque = scan(REPO_ROOT)

    seen = set(findings)
    unpinned = sorted(seen - set(expected))
    stale = sorted(set(expected) - seen)

    for k in unpinned:
        f, ref = k.split("::", 1)
        print(f"✗ {f} names `{ref}`, which that module does not define. Python "
              f"resolves it at CALL time, so this is an AttributeError the "
              f"next execution finds — and if the caller is a gate, the "
              f"failure is NO VERDICT rather than a wrong one. Restore the "
              f"name, update the caller, or pin it with a reason in "
              f"{EXPECTED.name}")
    for k in stale:
        print(f"✗ {k} — pinned but no longer present. Drop the row: a pin file "
              f"that only ever loosens stops being a wall")
    for u in unparsed:
        print(f"⛔ UNPARSED {u} — never folded into the pass column")

    if n_scripts < MIN_SCRIPTS:
        print(f"✗ walk FLOOR breached: {n_scripts} tracked scripts/*.py < "
              f"{MIN_SCRIPTS} — the walk went blind and every count above is "
              f"vacuous")
        return 1
    if resolved < MIN_RESOLVED:
        print(f"✗ resolution FLOOR breached: {resolved} sibling-module "
              f"reference(s) resolved < {MIN_RESOLVED} — the import table "
              f"stopped being built, which looks exactly like a clean repo")
        return 1
    if unpinned or stale or unparsed:
        return 1

    print(f"✓ cross-module-attr gate PASSED — every sibling-module attribute "
          f"and from-import names something that module defines, over "
          f"{n_scripts} tracked scripts/*.py (floor {MIN_SCRIPTS}) / "
          f"{resolved} resolved reference(s) (floor {MIN_RESOLVED}), "
          f"{len(expected)} pinned exemption(s), 0 stale, 0 unparsed. "
          f"⚠ STATED blind spots: {n_getattr} getattr site(s) and "
          f"{n_opaque} star-import OPAQUE module(s) are counted and NEVER "
          f"flagged, a name bound only inside a function is not seen, and a "
          f"runtime break that is not name resolution needs an execution "
          f"(Issue 848 T3)")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--workspace" in argv:
        raise SystemExit(workspace_census())
    if "--prove-fires" in argv:
        i = argv.index("--prove-fires")
        sha = argv[i + 1] if i + 1 < len(argv) else PROVE_SHA
        raise SystemExit(prove_fires(sha))
    raise SystemExit(main())
