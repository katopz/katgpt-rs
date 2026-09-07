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

**`errexit` is the precondition, not `nounset`** — and the first version of
this audit got that backwards, because its premise harness hard-coded
`set -euo pipefail` and so never varied the axis it was making a claim about.
`measure_premise()` is the full (options x arm) matrix now, re-measured on
the box the audit runs on. Measured, `/bin/bash` 3.2.57(1)-release,
arm64-apple-darwin25 (**one** bash — an earlier "and 5.x" claim is not
reproducible here; there is no bash 5 on this box):

| shell options  | unbound expansion            | `eval` syntax error       |
|----------------|------------------------------|---------------------------|
| `set -u`       | aborts, **1** — not laundered | does **not abort at all** |
| `set -e`       | no abort (expands empty)     | 2 bare, **0** trapped ✗   |
| `set -eu`      | 1 bare, **0** trapped ✗      | 2 bare, **0** trapped ✗   |
| `set -e` cmd failure -> 1 both; command not found -> 127 both ✓            |

So a script with `set -u` and no `set -e` cannot launder anything today:
that is PRECAUTIONARY, not EXPOSED. Pooling the two over-claimed on 37% of
this report's rows (15 of 41).

⛔ And the measurement MODE is part of the claim: the nounset fatal error
exits **127** from `bash -c` and **1** from a script FILE on this bash. The
old harness used `bash -c` and never saw it, because at `set -euo pipefail`
both modes agree. `_run` writes a temp script now — the population is files.

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

# The verdicts

* **LIVE-FORWARD** — a double-quoted `trap "... $VAR ..."` naming a variable
  first assigned LATER in the file (or never). This is not exposure: under
  `set -u` the script *provably* aborts at that line, every run. If the
  script also has `set -e` and lacks a sentinel, the abort reads as a pass.
* **EXPOSED** — `set -e` + an EXIT trap + no completion sentinel. Latent:
  correct today, reports a pass for any abort introduced tomorrow.
* **PRECAUTIONARY** — nounset WITHOUT errexit + an EXIT trap + no sentinel.
  Measured above: those aborts exit 1, so nothing is being laundered today.
  One added `-e` away from EXPOSED, and nobody re-audits a `set` line when
  they change it — but reporting it as live would be an over-claim.
* **UNPARSED** — the trap names a function whose body never closed under
  brace counting, so the classifier could not read the handler at all.
  Neither a pass nor a finding: a runaway body swallows the rest of the file
  and can manufacture a false SENTINELLED, which HIDES exposure.
* **REPLACED** — two or more `trap ... EXIT` **registrations**. `trap`
  REPLACES; it does not accumulate, so every earlier handler is silently
  dropped. An orthogonal defect (leaked temp dirs, skipped cleanup) reported
  alongside. `trap - EXIT` (restore the default) and `trap "" EXIT` (ignore)
  are DEREGISTRATIONS and are not counted — the first cut counted them and
  its only workspace-wide REPLACED finding was that mistake.
* **SENTINELLED** — a handler-referenced flag whose last assignment follows
  the last trap registration, with a non-zero exit guarded by it. Not a
  finding.

A script with NO EXIT trap is not in the population at all: its aborts exit
non-zero correctly. Adding a cleanup trap to such a script is what introduces
the laundering — which is why "just add a trap" is the wrong advice on its
own. The population predicate is `set -e` **OR** `set -u` (the union): a
SET_U-only predicate both over-claimed on the nounset-only rows and had a
mirror blind spot — errexit + trap + `eval`, no nounset, which launders via
the syntax-error trigger and which `analyse()` skipped before looking.
Measured with the widened predicate: **0** such scripts workspace-wide, so
that blind spot was empty — but it is a measurement now, not an assumption.
"""

import os
import re
import subprocess
import sys
import tempfile

# ── Verdicts ──────────────────────────────────────────────────────────────
LIVE_FORWARD = "LIVE-FORWARD"
EXPOSED = "EXPOSED"
SENTINELLED = "SENTINELLED"
# `set -u` + an EXIT trap + no sentinel, but NO `set -e`. Measured (see
# `measure_premise`): **errexit is the precondition for laundering**, for BOTH
# triggers. With nounset alone an unbound expansion aborts and exits 1 even
# with a succeeding EXIT trap, and an `eval` syntax error does not abort at
# all. So these scripts cannot launder anything TODAY — the sentinel is
# precautionary and becomes load-bearing the moment somebody adds `-e`. Never
# pooled with EXPOSED: doing so over-claimed on 37% of this report's rows.
PRECAUTIONARY = "PRECAUTIONARY"
# The classifier could not read the handler at all — the function body never
# closed under brace counting. NEVER folded into either real verdict: an
# unreadable body can just as easily swallow the rest of the file (and with it
# somebody else's `exit 1` and some late literal flag) and read as a false
# SENTINELLED, which HIDES exposure. See `function_bodies`.
UNPARSED = "UNPARSED"

SKIP_DIRS = {".git", "target", "node_modules", ".venv", "venv", "__pycache__"}

# `set -u` / `set -euo pipefail` / `set -o nounset`
SET_U = re.compile(r"^\s*set\s+(?:-[a-zA-Z]*u[a-zA-Z]*\b|-o\s+nounset\b)", re.M)
# `set -e` / `set -euo pipefail` / `set -o errexit`. This is the ACTUAL
# precondition for status laundering — see PRECAUTIONARY. The first version of
# this audit used SET_U alone as its population predicate, which both
# over-claimed (nounset-only rows reported as live) and had a mirror blind
# spot (errexit + trap + `eval`, no nounset — launderable, and `analyse`
# returned None before looking). The population is the UNION now.
SET_E = re.compile(r"^\s*set\s+(?:-[a-zA-Z]*e[a-zA-Z]*\b|-o\s+errexit\b)", re.M)
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
# `<<EOF` / `<<-'EOF'` / `<<"EOF"` — a heredoc body is DATA, and its braces are
# not shell braces.
HEREDOC = re.compile(r"<<-?\s*(?P<q>[\'\"]?)(?P<tag>[A-Za-z_][A-Za-z0-9_]*)(?P=q)")


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        return None


def scan_braces(line, quote):
    """Count SHELL braces on one line, carrying an open quote across lines.

    Returns `(delta, quote)` where `quote` is None / "'" / '"' at end of line.

    Why this is not `line.count("{") - line.count("}")`: riir-chain's
    `block_pipeline_reachability_gate.sh` embeds a multi-line, single-quoted
    `awk` program containing `mod[[:space:]]+tests[[:space:]]*\\{` — one
    unmatched `{` inside DATA. Naive counting therefore never closed that
    function, its "body" ran to EOF, and the file read as EXPOSED even though
    it carries a correct sentinel. The same mis-parse in the other direction
    is worse: a runaway body swallows unrelated `exit 1`s and can manufacture
    a false SENTINELLED.
    """
    delta = 0
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if quote == "'":
            if c == "'":
                quote = None
            i += 1
            continue
        if quote == '"':
            if c == "\\":
                i += 2
                continue
            if c == '"':
                quote = None
            i += 1
            continue
        if c == "\\":
            i += 2
            continue
        if c == "#" and (i == 0 or line[i - 1] in " \t;&|("):
            break  # a comment runs to end of line
        if c in "'\"":
            quote = c
            i += 1
            continue
        if c == "{":
            delta += 1
        elif c == "}":
            delta -= 1
        i += 1
    return delta, quote


def function_bodies(lines):
    """Map function name -> (start_line, end_line, body_lines, terminated).

    Brace counting, not a parser — but quote- and heredoc-aware (see
    `scan_braces`). `terminated` is False when the body ran off the end of the
    file with depth still open: that is the classifier failing to read, and
    `analyse` reports it as UNPARSED rather than guessing a verdict.
    """
    out = {}
    i = 0
    while i < len(lines):
        m = FUNC_DEF.match(lines[i])
        if m and ("{" in lines[i] or (i + 1 < len(lines) and lines[i + 1].strip() == "{")):
            name = m.group("name")
            quote = None
            depth, quote = scan_braces(lines[i], quote)
            j = i + 1
            if depth == 0 and j < len(lines):  # brace on the next line
                depth, quote = scan_braces(lines[j], quote)
                j += 1
            body = []
            heredoc = None
            while j < len(lines) and depth > 0:
                line = lines[j]
                body.append(line)
                j += 1
                if heredoc is not None:
                    if line.strip() == heredoc:
                        heredoc = None
                    continue  # heredoc body is DATA
                hm = HEREDOC.search(line) if quote is None else None
                d, quote = scan_braces(line, quote)
                depth += d
                if hm:
                    heredoc = hm.group("tag")
            out[name] = (i + 1, j, body, depth == 0)
            i = j
            continue
        i += 1
    return out


def guards_nonzero_exit(handler_lines, name):
    """Does a conditional TESTING `name` contain a non-zero exit in its body?

    Requiring "the handler tests the flag" and "the handler exits non-zero"
    *independently* is not enough, and the miss was found by planting one:
    `full_gate.sh` also has `if [ "$KEEP_LOG" -eq 1 ]; then echo …; else rm
    -f …; fi`, which satisfies both clauses while gating a log-retention
    echo. Deleting the real sentinel's last-line assignment then left the
    script classified SENTINELLED — a FALSE SENTINELLED, the direction that
    HIDES exposure. The flag must gate the failure, so the two facts have to
    be tied together by block structure.
    """
    for i, line in enumerate(handler_lines):
        if not re.search(rf'\[\s*[^]]*\$\{{?{re.escape(name)}\b', line):
            continue
        depth = 0
        for j in range(i, len(handler_lines)):
            stripped = handler_lines[j].strip()
            if re.match(r"^(if|elif)\b", stripped) or stripped.endswith("; then"):
                depth += 1
            if re.match(r"^fi\b", stripped):
                depth -= 1
                if depth <= 0:
                    break
            if NONZERO_EXIT.search(handler_lines[j]):
                return True
    return False


def analyse(path):
    lines = read(path)
    if lines is None:
        return None
    src = "\n".join(lines)
    nounset = bool(SET_U.search(src))
    errexit = bool(SET_E.search(src))
    if not (nounset or errexit):
        return None  # no abort-on-error at all; nothing to launder

    # A trap line is a REGISTRATION or a DEREGISTRATION, and only the first
    # kind can launder a status. `trap - EXIT` RESTORES the default (the
    # laundering stops there) and `trap "" EXIT` IGNORES the signal. Counting
    # either as a handler is not a rounding error: the first cut of this
    # audit reported exactly one REPLACED finding workspace-wide
    # (riir-chain/scripts/program_rate_gate.sh) and it was this — a
    # registration at line 447 and its own `trap - EXIT` teardown at 510. A
    # 1-of-1 false positive rate in the verdict nobody would re-derive.
    traps, dereg = [], []
    for idx, line in enumerate(lines):
        m = TRAP_EXIT.match(line)
        if not m:
            continue
        body = m.group("body").strip()
        if body in ("-", "''", '""', "-'", ''):
            dereg.append((idx + 1, body))
        else:
            traps.append((idx + 1, body))
    if not traps:
        return None  # no EXIT HANDLER -> nothing to launder the status

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
    unparsed = []
    for ln, body in traps:
        bare = body.strip("'\"")
        if bare in funcs:
            _st, _en, blines, terminated = funcs[bare]
            if not terminated:
                unparsed.append((ln, bare))
            handler_bodies.append("\n".join(blines))
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
            # ...and the conditional that READS it must be the one that
            # exits non-zero (see guards_nonzero_exit for the plant that
            # forced this).
            if guards_nonzero_exit(handler_src.splitlines(), name):
                sentinel_var = name
                break

    # ── the exposure WINDOW: [last trap registration, EOF) ───────────────
    # Nothing before the handler exists can be laundered by it, so a window
    # with no abort SITE in it cannot launder regardless of the `set` line.
    # A triage aid, not a verdict — same standing as tail support in
    # percentile_index_audit.py: it ORDERS the findings (a 2-line window with
    # 0 triggers and an 863-line window with 253 are one row each otherwise).
    window_lines = [l for l in lines[last_trap_line:] if not l.lstrip().startswith("#")]
    window = len(lines) - last_trap_line
    win_src = "\n".join(window_lines)
    triggers = len(VARREF.findall(win_src)) + len(re.findall(r"\beval\b", win_src))
    # A function DEFINED above the trap line but CALLED inside the window has
    # its trigger text outside the window — line-based counting under-reports
    # it. Fold in the body of any function the window actually calls.
    for fname, (fst, _fen, fbody, _fterm) in funcs.items():
        if fst > last_trap_line:
            continue  # defined inside the window; already counted
        if not re.search(r"(?<![\w.-])" + re.escape(fname) + r"(?![\w.-])", win_src):
            continue
        fb = "\n".join(l for l in fbody if not l.lstrip().startswith("#"))
        triggers += len(VARREF.findall(fb)) + len(re.findall(r"\beval\b", fb))

    if forwards:
        verdict = LIVE_FORWARD
    elif unparsed:
        # Read NOTHING off an unreadable handler — not a pass, not a finding.
        verdict = UNPARSED
    elif sentinel_var:
        verdict = SENTINELLED
    elif errexit:
        verdict = EXPOSED
    else:
        # nounset only: the abort exits 1 today. Precautionary, not live.
        verdict = PRECAUTIONARY

    return {
        "verdict": verdict,
        "traps": traps,
        "replaced": len(traps) > 1,
        "dereg": dereg,
        "forwards": forwards,
        "unparsed": unparsed,
        "errexit": errexit,
        "nounset": nounset,
        "window": window,
        "triggers": triggers,
        "sentinel": sentinel_var,
        "lines": len(lines),
    }


# ── The premise, re-measured on THIS box ──────────────────────────────────
#
# ⛔ This used to be four arms all measured under a HARD-CODED `set -euo
# pipefail`, which is how the laundering got attributed to the wrong
# predicate: the measurement never varied the shell options, so "unbound
# expansion launders" read as "nounset is the precondition". It is not.
# errexit is, for BOTH triggers. The matrix is now over (options x arm).
PREMISE_OPTS = ["set -u", "set -e", "set -eu", "set -euo pipefail"]
PREMISE_ARMS = [
    # (name, body)
    ("unbound expansion", 'echo pre; echo "$DEFINITELY_UNSET_XYZ"; echo post'),
    ("eval syntax error", 'echo pre; eval "if ["; echo post'),
    ("command failure", "echo pre; false; echo post"),
    ("command not found", "echo pre; no_such_command_xyz; echo post"),
]
# What the docs (AGENTS.md, .issues/734) claim, as (opts, arm) -> (bare,
# trapped). Measured on /bin/bash 3.2.57(1)-release, arm64-apple-darwin25,
# 2026-09-07. A divergence is PRINTED, never silently accepted — the point of
# re-measuring is that a premise typed into a docstring is not evidence.
PREMISE_DOCUMENTED = {
    ("set -u", "unbound expansion"): (1, 1),
    ("set -u", "eval syntax error"): (0, 0),      # does not abort AT ALL
    ("set -u", "command failure"): (0, 0),
    ("set -u", "command not found"): (0, 0),
    ("set -e", "unbound expansion"): (0, 0),      # unset expands empty
    ("set -e", "eval syntax error"): (2, 0),      # LAUNDERS
    ("set -e", "command failure"): (1, 1),
    ("set -e", "command not found"): (127, 127),
    ("set -eu", "unbound expansion"): (1, 0),     # LAUNDERS
    ("set -eu", "eval syntax error"): (2, 0),     # LAUNDERS
    ("set -eu", "command failure"): (1, 1),
    ("set -eu", "command not found"): (127, 127),
    ("set -euo pipefail", "unbound expansion"): (1, 0),
    ("set -euo pipefail", "eval syntax error"): (2, 0),
    ("set -euo pipefail", "command failure"): (1, 1),
    ("set -euo pipefail", "command not found"): (127, 127),
}


def _run(body, with_trap, opts):
    """Run one premise arm as a SCRIPT FILE, which is what the population is.

    ⛔ Not `bash -c`. Measured on bash 3.2.57: the nounset fatal error exits
    **127** from `bash -c` and **1** from a script file — the measurement MODE
    changes the measured status. The earlier harness used `bash -c` and never
    saw it, because it only ever measured `set -euo pipefail`, where both
    modes agree (1 bare / 0 trapped). A premise instrument that does not run
    the thing it is making claims about is the failure this repo keeps
    re-finding.
    """
    trap = "trap 'true' EXIT\n" if with_trap else ""
    script = f"#!/usr/bin/env bash\n{opts}\n{trap}{body}\n"
    fd, path = tempfile.mkstemp(suffix=".sh", prefix="premise_arm_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(script)
        p = subprocess.run(["bash", path], capture_output=True, timeout=20)
        return p.returncode
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def measure_premise():
    rows = []
    for opts in PREMISE_OPTS:
        for name, body in PREMISE_ARMS:
            bare = _run(body, False, opts)
            trapped = _run(body, True, opts)
            want = PREMISE_DOCUMENTED.get((opts, name))
            rows.append({
                "opts": opts,
                "name": name,
                "bare": bare,
                "trapped": trapped,
                # LAUNDERS = the abort is real (non-zero bare) and the EXIT
                # trap turns it into a success.
                "launders": bare not in (0, None) and trapped == 0,
                "as_documented": want is None or (bare, trapped) == want,
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
    cases.append(("a late literal flag gating a NON-failure branch is not a sentinel",
                  """#!/usr/bin/env bash
set -euo pipefail
LOG="$(mktemp)"
KEEP=0
cleanup() {
    st=$?
    if [ "$KEEP" -eq 1 ]; then
        echo "log retained: $LOG"
    else
        rm -f "$LOG"
    fi
    if [ "$st" != "0" ]; then
        exit 1
    fi
    exit "$st"
}
trap cleanup EXIT
echo layer
KEEP=1
echo ALL GREEN
""", EXPOSED))
    cases.append(("registration + its own `trap - EXIT` teardown is NOT replaced",
                  """#!/usr/bin/env bash
set -euo pipefail
A="$(mktemp)"
trap 'rm -f "$A"' EXIT
echo layer
trap - EXIT
rm -f "$A"
echo ALL GREEN
""", EXPOSED))
    # The shape that made the naive brace counter run a function body to EOF:
    # a multi-line SINGLE-QUOTED awk program with one unmatched `{` in DATA.
    # Before `scan_braces` this read EXPOSED despite a correct sentinel
    # (riir-chain/scripts/block_pipeline_reachability_gate.sh).
    cases.append(("a single-quoted awk program with an unmatched brace is DATA",
                  """#!/usr/bin/env bash
set -euo pipefail
TMP="$(mktemp)"
DONE_FLAG=0
scan() {
    awk -v f="$1" '
        /^[[:space:]]*mod[[:space:]]+tests[[:space:]]*\\{/ { seen = 1 }
        END { print seen }
    ' "$1"
}
cleanup() {
    st=$?
    rm -f "$TMP"
    if [ "$DONE_FLAG" != "1" ] && [ "$st" = "0" ]; then
        echo aborted >&2
        exit 1
    fi
    exit "$st"
}
trap cleanup EXIT
scan "$TMP"
DONE_FLAG=1
""", SENTINELLED))
    # ...and the direction that MATTERS: an unreadable handler must never be
    # reported as either verdict. Here the trap's function genuinely never
    # closes, and the runaway body would otherwise swallow the `exit 1` and
    # the late literal flag below it and read as a false SENTINELLED.
    cases.append(("an unterminated handler body is UNPARSED, never SENTINELLED",
                  """#!/usr/bin/env bash
set -euo pipefail
TMP="$(mktemp)"
DONE_FLAG=0
cleanup() {
    st=$?
    rm -f "$TMP"
    if [ "$DONE_FLAG" != "1" ] && [ "$st" = "0" ]; then
        exit 1
    fi
    exit "$st"
    if true; then {
trap cleanup EXIT
echo layer
DONE_FLAG=1
""", UNPARSED))
    # ── the severity split (measured; see measure_premise) ──────────────
    # nounset WITHOUT errexit: the abort exits 1 today, so this is not a live
    # defect. Reporting it as EXPOSED over-claimed on 37% of the rows.
    cases.append(("nounset without errexit is PRECAUTIONARY, not EXPOSED",
                  """#!/usr/bin/env bash
set -uo pipefail
A="$(mktemp)"
trap 'rm -f "$A"' EXIT
echo layer
echo ALL GREEN
""", PRECAUTIONARY))
    # ...and the mirror blind spot the SET_U-only predicate had: errexit with
    # NO nounset still launders, via the `eval` syntax-error trigger, and the
    # old population predicate returned None before looking at it.
    cases.append(("errexit WITHOUT nounset is in the population and EXPOSED",
                  """#!/usr/bin/env bash
set -e
A="$(mktemp)"
trap 'rm -f "$A"' EXIT
echo layer
eval "$SOME_GENERATED_SNIPPET"
echo ALL GREEN
""", EXPOSED))
    failures = []
    for name, body, want in cases:
        p = plant(body)
        got = analyse(p)
        os.unlink(p)
        got_v = got["verdict"] if got else "NOT-IN-POPULATION"
        if got_v != want:
            failures.append(f"    {name}: expected {want}, got {got_v}")
        if got and name.startswith("registration + its own") and got["replaced"]:
            failures.append("    dereg control: `trap - EXIT` was counted as a second REGISTRATION")

    # ── window/trigger control: the quantity that ORDERS the findings ────
    # Two arms, because a counter that is always 0 and a counter that is
    # always large both look plausible on a single row. The second arm also
    # pins the fold-in: `probe`'s body is ABOVE the trap line, so a
    # line-based count would report 0 triggers for a window that calls it.
    p = plant("""#!/usr/bin/env bash
set -euo pipefail
A="$(mktemp)"
trap 'rm -f "$A"' EXIT

wait
""")
    got = analyse(p)
    os.unlink(p)
    if got is None or got["triggers"] != 0 or got["window"] != 2:
        failures.append("    window control (empty window): expected window=2 triggers=0, "
                        f"got {got and (got['window'], got['triggers'])}")

    p = plant("""#!/usr/bin/env bash
set -euo pipefail
A="$(mktemp)"
probe() {
    echo "$SOME_VAR"
    eval "$SNIPPET"
}
trap 'rm -f "$A"' EXIT
probe
""")
    got = analyse(p)
    os.unlink(p)
    if got is None or got["triggers"] < 3:
        failures.append("    window control (function called in the window): expected the "
                        "callee's 3 triggers to be folded in, got "
                        f"{got and got['triggers']}")

    # negative population control: NEITHER errexit nor nounset -> no
    # abort-on-error at all, so there is no status to launder. (This control
    # used to plant `set -e`, which the SET_U-only predicate excluded — and
    # that exclusion was the mirror blind spot, now a positive case above.)
    p = plant('#!/usr/bin/env bash\nA=x\ntrap \'rm -f "$A"\' EXIT\necho hi\n')
    got = analyse(p)
    os.unlink(p)
    if got is not None:
        failures.append("    no-set-e/-u control: expected NOT-IN-POPULATION, "
                        f"got {got['verdict']}")

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
    diverged = 0
    print(f"  {'shell options':<20} {'abort arm':<20} {'bare':<6} {'+EXIT trap':<11} verdict")
    for row in measure_premise():
        flag = "LAUNDERS -> 0" if row["launders"] else "status preserved"
        note = "   ⛔ DIVERGES from the documented table" if not row["as_documented"] else ""
        launders += bool(row["launders"])
        diverged += not row["as_documented"]
        print(f"  {row['opts']:<20} {row['name']:<20} {row['bare']!s:<6} "
              f"{row['trapped']!s:<11} {flag}{note}")
    print("\n  errexit is the PRECONDITION, for both triggers: with `set -u` alone an\n"
          "  unbound expansion aborts and exits 1 even with a succeeding EXIT trap,\n"
          "  and an `eval` syntax error does not abort at all. Rows whose script has\n"
          "  nounset WITHOUT errexit are therefore PRECAUTIONARY, not EXPOSED.")
    if diverged:
        print(f"\n  ⛔ {diverged} measured cell(s) DIVERGE from what the docs claim. The\n"
              "  findings below are about a premise that does not hold on this box —\n"
              "  re-read before acting, and fix the docs, not the measurement.")
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
        print("── selftest: 13/13 (9 verdicts + 2 window + 2 population controls) fire as pinned\n")

    grand = {}
    all_rows = []
    for tgt in targets:
        name = os.path.basename(tgt)
        tally = {LIVE_FORWARD: 0, EXPOSED: 0, PRECAUTIONARY: 0,
                 SENTINELLED: 0, UNPARSED: 0, "replaced": 0}
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
        ("EXPOSED       (errexit + an EXIT trap + no sentinel — an abort in the window "
         "reports exit 0)",
         lambda r: r["verdict"] == EXPOSED),
        ("PRECAUTIONARY (nounset WITHOUT errexit — the abort exits 1 today; the "
         "sentinel is inert until somebody adds `-e`)",
         lambda r: r["verdict"] == PRECAUTIONARY),
        ("UNPARSED      (the handler body never closed — the classifier could NOT read it; "
         "neither a pass nor a finding)",
         lambda r: r["verdict"] == UNPARSED),
        ("REPLACED      (2+ EXIT traps; `trap` replaces, so earlier cleanup is dropped)",
         lambda r: r["replaced"]),
    ):
        rows = [r for r in all_rows if pred(r)]
        print(f"── {label}: {len(rows)}")
        for r in sorted(rows, key=lambda r: (-r["triggers"], r["repo"], r["file"])):
            inert = "  ⟵ ZERO abort sites in the window: provably cannot launder" \
                if r["triggers"] == 0 else ""
            print(f"     {r['repo']}/{r['file']}"
                  f"  [window {r['window']} line(s), {r['triggers']} trigger(s)]{inert}")
            for ln, var, fa in r["forwards"]:
                where = f"first assigned line {fa}" if fa else "never assigned"
                print(f"       line {ln}: ${var} ({where})")
            if r["replaced"] and r["verdict"] != LIVE_FORWARD:
                print(f"       EXIT traps at lines {', '.join(str(ln) for ln, _ in r['traps'])}")
        print()

    print("── per-repo tally " + "─" * 48)
    hdr = [LIVE_FORWARD, EXPOSED, PRECAUTIONARY, SENTINELLED, UNPARSED, "replaced"]
    print(f"  {'repo':<26}" + "".join(f"{h:>14}" for h in hdr) + f"{'total':>8}")
    tot = {h: 0 for h in hdr}
    for name in sorted(grand):
        t = grand[name]
        if not sum(t[h] for h in (LIVE_FORWARD, EXPOSED, PRECAUTIONARY, SENTINELLED, UNPARSED)):
            continue
        print(f"  {name:<26}" + "".join(f"{t[h]:>14}" for h in hdr)
              + f"{sum(t[h] for h in (LIVE_FORWARD, EXPOSED, PRECAUTIONARY, SENTINELLED, UNPARSED)):>8}")
        for h in hdr:
            tot[h] += t[h]
    print(f"  {'ALL':<26}" + "".join(f"{tot[h]:>14}" for h in hdr)
          + f"{sum(tot[h] for h in (LIVE_FORWARD, EXPOSED, PRECAUTIONARY, SENTINELLED, UNPARSED)):>8}")
    # ── the severity axis, ACROSS verdicts ────────────────────────────────
    # SENTINELLED masks the split: a nounset-only script that has been given a
    # sentinel reads SENTINELLED, and the fact that its sentinel is inert
    # TODAY is invisible in the verdict column. Print it, because "how many of
    # these could actually launder" is the question the report is asked.
    live = [r for r in all_rows if r["errexit"]]
    prec = [r for r in all_rows if not r["errexit"]]
    live_unsent = [r for r in live if r["verdict"] in (EXPOSED, LIVE_FORWARD)]
    # The bottom line: unsentinelled AND with at least one abort site in the
    # window. Anything else cannot launder, whatever its `set` line says.
    launderable = [r for r in live_unsent if r["triggers"] > 0]
    print(f"\n  severity axis (independent of the verdict): {len(live)} script(s) have "
          f"errexit\n  and could launder an abort ({len(live_unsent)} of them without a "
          f"sentinel, of which\n  {len(launderable)} has an abort site in its window — "
          f"that count is the bottom line);\n  {len(prec)} have nounset WITHOUT errexit — "
          f"their aborts exit 1, so their\n  sentinels are PRECAUTIONARY and become "
          f"load-bearing only if somebody adds `-e`.")
    print("\n  EXPOSED is LATENT — it needs an abort to bite, and the script may well\n"
          "  have none today. It is still the reason the seal-remake gate could not\n"
          "  fail for months: the abort arrived later, and nothing said so.\n"
          "  'replaced' counts scripts, not traps, and is orthogonal to the verdict.\n"
          "  Report only; exit 0 always.")


if __name__ == "__main__":
    main()
