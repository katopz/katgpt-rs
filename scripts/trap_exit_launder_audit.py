#!/usr/bin/env python3
"""Audit shell gates for the class where an ABORT reports exit 0.

A REPORT, not a gate (always exit 0) — the same reason
`percentile_index_audit.py` is one: the top verdict here is *latent*
(it needs an abort to bite), and a report that exits 1 on latent
exposure is a report nobody runs. The verdict half belongs in a per-repo
gate over the scripts this one finds LIVE.

# The defect, measured

    set -euo pipefail
    A="$(mktemp)"
    trap 'rm -f "$A"' EXIT     # <- an EXIT trap whose last command SUCCEEDS
    ...
    echo "$UNSET"              # <- set -u abort
    ...
    echo "ALL LAYERS GREEN"

stderr says `UNSET: unbound variable`, the script stops there — and `$?`
is **0**. Everything after the abort silently did not run, and the caller,
CI included, reads a pass.

Two aborts launder this way, and `selftest()` re-measures both on the box
the audit runs on rather than trusting this paragraph:

| abort                       | bare | `$?` at trap entry | with an EXIT trap |
|-----------------------------|------|--------------------|-------------------|
| `set -u` unbound expansion  | 1    | **0**              | **0** ✗           |
| `eval` with a syntax error  | 2    | **0**              | **0** ✗           |
| `set -e` command failure    | 1    | 1                  | 1 ✓               |
| command not found           | 127  | 127                | 127 ✓             |

The defensive idiom does **not** fix it. `trap 'rc=$?; cleanup; exit $rc'
EXIT` saves a status that is *already* 0 — measured, not assumed. The only
repair that works is a **completion sentinel**: a flag set on the script's
own last line and checked by the handler.

    GUARD_COMPLETED=0
    guard_cleanup() {
        st=$?
        ...cleanup...
        if [ "$GUARD_COMPLETED" != "1" ] && [ "$st" = "0" ]; then
            echo "aborted mid-run while reporting success" >&2; exit 1
        fi
        exit "$st"
    }
    trap guard_cleanup EXIT
    ...
    GUARD_COMPLETED=1     # the last line

The sentinel arm fires only on the laundering combination — INCOMPLETE
**and** claiming success — so an ordinary layer failure still exits 1 with
its own message and is not touched.

# Provenance

seal-remake's `scripts/ci_feature_guard.sh` carried a `trap "rm -f '$A'
'$B'" EXIT` naming two variables assigned ~20 and ~45 lines later. Under
`set -u` that expansion aborted the script — so its last four layers could
never run and `ALL LAYERS GREEN` could never print — and the abort exited
**0**, because a previously registered EXIT trap's `rm -f` succeeded. The
gate is what CI runs (`rust.yml`), so past that line the gate *could not
fail*. It stayed hidden because a different row (a ratchet ceiling) had
been red for three commits and stopped every run before reaching it.

# The four verdicts

* **LIVE-FORWARD** — a double-quoted `trap "... $VAR ..."` naming a variable
  first assigned LATER in the file (or never). This is not exposure: under
  `set -u` the script *provably* aborts at that line, every run. If the
  script also lacks a sentinel, the abort reads as a pass.
* **EXPOSED** — `set -u` + an EXIT trap + no completion sentinel. Latent:
  correct today, reports a pass for any abort introduced tomorrow.
* **REPLACED** — two or more `trap ... EXIT` registrations. `trap` REPLACES;
  it does not accumulate, so every earlier handler is silently dropped. An
  orthogonal defect (leaked temp dirs, skipped cleanup) reported alongside.
* **SENTINELLED** — a handler-referenced flag whose last assignment follows
  the last trap registration, with a non-zero exit guarded by it. Not a
  finding.

A script with `set -u` and NO EXIT trap is not in the population at all:
its aborts exit 1 correctly. Adding a cleanup trap to such a script is what
introduces the laundering — which is why "just add a trap" is the wrong
advice on its own.
"""

import os
import re
import subprocess
import sys

# ── Verdicts ──────────────────────────────────────────────────────────────
LIVE_FORWARD = "LIVE-FORWARD"
EXPOSED = "EXPOSED"
SENTINELLED = "SENTINELLED"

SKIP_DIRS = {".git", "target", "node_modules", ".venv", "venv", "__pycache__"}

# `set -u` / `set -euo pipefail` / `set -o nounset`
SET_U = re.compile(r"^\s*set\s+(?:-[a-zA-Z]*u[a-zA-Z]*\b|-o\s+nounset\b)", re.M)
# a trap registration naming EXIT (also ERR/RETURN are laundering-irrelevant:
# only EXIT decides the script's status)
TRAP_EXIT = re.compile(r"^\s*trap\s+(?P<body>.+?)\s+(?P<sigs>[A-Z0-9 ]*\bEXIT\b[A-Z0-9 ]*)\s*$")
# an assignment, including `local`/`export`/`declare` forms
ASSIGN = re.compile(r"^\s*(?:local\s+|export\s+|declare\s+(?:-\w+\s+)?|readonly\s+)?"
                    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)=")
VARREF = re.compile(r"\$\{?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\}?")
FUNC_DEF = re.compile(r"^\s*(?:function\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_:.-]*)\s*\(\s*\)\s*\{?")
NONZERO_EXIT = re.compile(r"\b(?:exit\s+[1-9]|return\s+[1-9]|exit\s+\"?\$)")
# a completion flag is only ever given a LITERAL constant -- no `$`, no
# command substitution. `PID=$!` and `status=$?` are late-assigned and
# handler-read too, and neither is a sentinel.
LITERAL_ASSIGN = re.compile(r'^["\']?(?:[01]|true|false|yes|no|done|complete)["\']?\s*(?:#.*)?$', re.I)


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        return None


def function_bodies(lines):
    """Map function name -> (start_line, end_line, body_lines).

    Brace counting, not a parser: enough for the house shell style (one
    `name() {` per line, closing `}` at column 0 or indented consistently).
    """
    out = {}
    i = 0
    while i < len(lines):
        m = FUNC_DEF.match(lines[i])
        if m and ("{" in lines[i] or (i + 1 < len(lines) and lines[i + 1].strip() == "{")):
            name = m.group("name")
            depth = lines[i].count("{") - lines[i].count("}")
            j = i + 1
            if depth == 0 and j < len(lines):  # brace on the next line
                depth = lines[j].count("{") - lines[j].count("}")
                j += 1
            body = []
            while j < len(lines) and depth > 0:
                body.append(lines[j])
                depth += lines[j].count("{") - lines[j].count("}")
                j += 1
            out[name] = (i + 1, j, body)
            i = j
            continue
        i += 1
    return out


def analyse(path):
    lines = read(path)
    if lines is None:
        return None
    src = "\n".join(lines)
    if not SET_U.search(src):
        return None  # aborts exit 1 correctly; not in the population

    traps = []
    for idx, line in enumerate(lines):
        m = TRAP_EXIT.match(line)
        if m:
            traps.append((idx + 1, m.group("body").strip()))
    if not traps:
        return None  # no EXIT trap -> nothing to launder the status

    funcs = function_bodies(lines)

    # first/last assignment line per variable, plus every RHS it is given
    first_assign, last_assign, assigns_of = {}, {}, {}
    for idx, line in enumerate(lines):
        m = ASSIGN.match(line)
        if m:
            name = m.group("name")
            first_assign.setdefault(name, idx + 1)
            last_assign[name] = idx + 1
            assigns_of.setdefault(name, []).append(line[m.end():].strip())

    # ── LIVE-FORWARD: a double-quoted trap body naming a later-assigned var
    forwards = []
    for ln, body in traps:
        if not body.startswith('"'):
            continue  # single-quoted -> expands at trap EXECUTION, not here
        for name in {m.group("name") for m in VARREF.finditer(body)}:
            if name in ("?", "!", "0", "@", "*"):
                continue
            fa = first_assign.get(name)
            if fa is None or fa > ln:
                forwards.append((ln, name, fa))

    # ── SENTINELLED: a handler-referenced flag re-assigned after the last trap
    last_trap_line = max(ln for ln, _ in traps)
    handler_bodies = []
    for ln, body in traps:
        bare = body.strip("'\"")
        if bare in funcs:
            handler_bodies.append("\n".join(funcs[bare][2]))
        else:
            handler_bodies.append(body)
    handler_src = "\n".join(handler_bodies)
    handler_vars = {m.group("name") for m in VARREF.finditer(handler_src)}
    sentinel_var = None
    if NONZERO_EXIT.search(handler_src):
        for name in sorted(handler_vars):
            la = last_assign.get(name)
            if la is None or la <= last_trap_line:
                continue  # never re-set after the handler was registered
            # A completion FLAG is assigned only LITERAL constants. This is
            # what separates it from the two false positives the first cut of
            # this audit produced: `BOOTSTRAP_PID=$!` and `status=$?` are also
            # assigned late and also read inside the handler, and neither says
            # anything about whether the script reached its last line.
            if not all(LITERAL_ASSIGN.match(l) for l in assigns_of.get(name, [])):
                continue
            # ...and it must be READ in a conditional, not merely cleaned up.
            if re.search(rf'\[\s*[^]]*\$\{{?{re.escape(name)}\b', handler_src):
                sentinel_var = name
                break

    if forwards:
        verdict = LIVE_FORWARD
    elif sentinel_var:
        verdict = SENTINELLED
    else:
        verdict = EXPOSED

    return {
        "verdict": verdict,
        "traps": traps,
        "replaced": len(traps) > 1,
        "forwards": forwards,
        "sentinel": sentinel_var,
        "lines": len(lines),
    }


# ── The premise, re-measured on THIS box ──────────────────────────────────
PREMISE_ARMS = [
    # (name, script body after `set -euo pipefail` + a succeeding EXIT trap,
    #  expected status WITHOUT the trap, expected status WITH it)
    ("set -u unbound expansion", 'echo pre; echo "$DEFINITELY_UNSET_XYZ"; echo post', 1, 0),
    ("eval syntax error", 'echo pre; eval "if ["; echo post', 2, 0),
    ("set -e command failure", "echo pre; false; echo post", 1, 1),
    ("command not found", "echo pre; no_such_command_xyz; echo post", 127, 127),
]


def _run(body, with_trap):
    trap = "trap 'true' EXIT; " if with_trap else ""
    script = f"set -euo pipefail; {trap}{body}"
    try:
        p = subprocess.run(["bash", "-c", script], capture_output=True, timeout=20)
        return p.returncode
    except (OSError, subprocess.TimeoutExpired):
        return None


def measure_premise():
    rows = []
    for name, body, want_bare, want_trapped in PREMISE_ARMS:
        bare = _run(body, with_trap=False)
        trapped = _run(body, with_trap=True)
        rows.append({
            "name": name,
            "bare": bare,
            "trapped": trapped,
            "launders": bare not in (0, None) and trapped == 0,
            "as_documented": bare == want_bare and trapped == want_trapped,
        })
    return rows


def selftest():
    """Both directions, on planted scripts — a canary that cannot be inert."""
    import tempfile

    def plant(body):
        fd, p = tempfile.mkstemp(suffix=".sh")
        with os.fdopen(fd, "w") as fh:
            fh.write(body)
        return p

    cases = [
        ("forward-referencing trap", """#!/usr/bin/env bash
set -euo pipefail
A="$(mktemp)"
trap "rm -f '$A' '$B'" EXIT
echo layer
B="$(mktemp)"
""", LIVE_FORWARD),
        ("plain cleanup trap, no sentinel", """#!/usr/bin/env bash
set -euo pipefail
A="$(mktemp)"
trap 'rm -f "$A"' EXIT
echo layer
echo ALL GREEN
""", EXPOSED),
        ("completion sentinel", """#!/usr/bin/env bash
set -euo pipefail
TMP=""
DONE_FLAG=0
cleanup() {
    st=$?
    rm -rf $TMP
    if [ "$DONE_FLAG" != "1" ] && [ "$st" = "0" ]; then
        echo aborted >&2
        exit 1
    fi
    exit "$st"
}
trap cleanup EXIT
echo layer
echo ALL GREEN
DONE_FLAG=1
""", SENTINELLED),
    ]
    failures = []
    for name, body, want in cases:
        p = plant(body)
        got = analyse(p)
        os.unlink(p)
        got_v = got["verdict"] if got else "NOT-IN-POPULATION"
        if got_v != want:
            failures.append(f"    {name}: expected {want}, got {got_v}")

    # negative population control: no `set -u` -> must not be reported at all
    p = plant('#!/usr/bin/env bash\nset -e\nA=x\ntrap \'rm -f "$A"\' EXIT\necho hi\n')
    got = analyse(p)
    os.unlink(p)
    if got is not None:
        failures.append(f"    no-set-u control: expected NOT-IN-POPULATION, got {got['verdict']}")

    # negative population control: set -u but NO EXIT trap -> aborts exit 1
    p = plant('#!/usr/bin/env bash\nset -euo pipefail\necho hi\n')
    got = analyse(p)
    os.unlink(p)
    if got is not None:
        failures.append(f"    no-trap control: expected NOT-IN-POPULATION, got {got['verdict']}")
    return failures


def repos(root):
    return sorted(
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "BOUNDARY.md"))
        and os.path.isdir(os.path.join(root, d, ".git"))
    )


def walk_sh(repo_root):
    """Every TRACKED `*.sh` in the repo.

    Tracked, not "on disk": the first cut of this audit walked the
    filesystem and reported 25 findings in `katgpt-rs/.raw/rapid-mlx`, a
    **gitignored** vendored drop that is not part of any repo. A population
    that includes files no repo owns cannot be acted on, and it inflates
    every total. `git ls-files` IS the population; the os.walk path below is
    a fallback for a non-git target passed by hand on argv.
    """
    try:
        out = subprocess.run(["git", "-C", repo_root, "ls-files", "-z", "*.sh"],
                             capture_output=True, timeout=60)
        if out.returncode == 0:
            for rel in out.stdout.decode("utf-8", "replace").split("\0"):
                if rel:
                    yield os.path.join(repo_root, rel)
            return
    except (OSError, subprocess.TimeoutExpired):
        pass
    for dp, dns, fns in os.walk(repo_root):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for f in fns:
            if f.endswith(".sh"):
                yield os.path.join(dp, f)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args:
        targets = [os.path.abspath(a) for a in args]
        root = None
    else:
        root = os.path.dirname(here)
        targets = [os.path.join(root, d) for d in repos(root)]

    print("── premise (re-measured on this box, not quoted from the docstring) ──")
    bash_v = subprocess.run(["bash", "-c", "echo $BASH_VERSION"],
                            capture_output=True, text=True).stdout.strip()
    print(f"  bash {bash_v}")
    launders = 0
    for row in measure_premise():
        flag = "LAUNDERS -> 0" if row["launders"] else "status preserved"
        note = "" if row["as_documented"] else "   (DIVERGES from the documented table)"
        launders += bool(row["launders"])
        print(f"  {row['name']:<28} bare={row['bare']!s:<4} with EXIT trap={row['trapped']!s:<4} {flag}{note}")
    if launders == 0:
        print("\n  This bash does NOT launder any measured abort. The findings below are\n"
              "  then about a premise that no longer holds here — re-read before acting.")
    print()

    fails = selftest()
    if fails:
        print("── selftest: FAILED (the detector is not measuring what it claims)")
        for f in fails:
            print(f)
        print()
    else:
        print("── selftest: 5/5 (3 verdicts + 2 population controls) fire as pinned\n")

    grand = {}
    all_rows = []
    for tgt in targets:
        name = os.path.basename(tgt)
        tally = {LIVE_FORWARD: 0, EXPOSED: 0, SENTINELLED: 0, "replaced": 0}
        for path in walk_sh(tgt):
            r = analyse(path)
            if r is None:
                continue
            r["repo"] = name
            r["file"] = os.path.relpath(path, tgt)
            tally[r["verdict"]] += 1
            tally["replaced"] += bool(r["replaced"])
            all_rows.append(r)
        grand[name] = tally

    for label, pred in (
        ("LIVE-FORWARD  (aborts at that line EVERY run; reads as a pass unless sentinelled)",
         lambda r: r["verdict"] == LIVE_FORWARD),
        ("REPLACED      (2+ EXIT traps; `trap` replaces, so earlier cleanup is dropped)",
         lambda r: r["replaced"]),
    ):
        rows = [r for r in all_rows if pred(r)]
        print(f"── {label}: {len(rows)}")
        for r in sorted(rows, key=lambda r: (r["repo"], r["file"])):
            print(f"     {r['repo']}/{r['file']}")
            for ln, var, fa in r["forwards"]:
                where = f"first assigned line {fa}" if fa else "never assigned"
                print(f"       line {ln}: ${var} ({where})")
            if r["replaced"] and r["verdict"] != LIVE_FORWARD:
                print(f"       EXIT traps at lines {', '.join(str(ln) for ln, _ in r['traps'])}")
        print()

    print("── per-repo tally " + "─" * 48)
    hdr = [LIVE_FORWARD, EXPOSED, SENTINELLED, "replaced"]
    print(f"  {'repo':<26}" + "".join(f"{h:>14}" for h in hdr) + f"{'total':>8}")
    tot = {h: 0 for h in hdr}
    for name in sorted(grand):
        t = grand[name]
        if not sum(t[h] for h in (LIVE_FORWARD, EXPOSED, SENTINELLED)):
            continue
        print(f"  {name:<26}" + "".join(f"{t[h]:>14}" for h in hdr)
              + f"{sum(t[h] for h in (LIVE_FORWARD, EXPOSED, SENTINELLED)):>8}")
        for h in hdr:
            tot[h] += t[h]
    print(f"  {'ALL':<26}" + "".join(f"{tot[h]:>14}" for h in hdr)
          + f"{sum(tot[h] for h in (LIVE_FORWARD, EXPOSED, SENTINELLED)):>8}")
    print("\n  EXPOSED is LATENT — it needs an abort to bite, and the script may well\n"
          "  have none today. It is still the reason the seal-remake gate could not\n"
          "  fail for months: the abort arrived later, and nothing said so.\n"
          "  'replaced' counts scripts, not traps, and is orthogonal to the verdict.\n"
          "  Report only; exit 0 always.")


if __name__ == "__main__":
    main()
