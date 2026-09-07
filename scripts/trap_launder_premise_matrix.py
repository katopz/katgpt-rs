#!/usr/bin/env python3
"""Measure Issue 734's laundering premise across INTERPRETERS, not just this box.

A REPORT, not a gate (always exit 0). The verdict half for this repo's own
scripts is `scripts/trap_sentinel_gate.py`; the population half is
`scripts/trap_exit_launder_audit.py`. This is the third, orthogonal half:
**which shell laundering happens in.**

# Why this exists

`trap_exit_launder_audit.py` re-measures the premise on every run — but only
on the box it runs on, with `/usr/bin/env bash`. On this workstation that is
bash **3.2.57** (the last GPLv2 bash, frozen in macOS since 2007), and the
premise reproduces there. Issue 734's prose, the sentinel comment block copied
into ~27 repaired scripts, and AGENTS.md all said "measured on bash 3.2 and
**5.x**". The 5.x half was never measured — there is no bash 5 on this
machine — and this instrument measures it:

    interpreter                     laundering / aborts observed
    bash 3.2.57      macOS /bin/bash     5 / 13   <- the premise
    bash 3.2.57      macOS /bin/sh       5 / 13      (same binary, posix mode)
    bash 4.4.23      docker bash:4.4     0 / 13
    bash 5.0.18      docker bash:5.0     0 / 13
    bash 5.2.37      docker bash:5.2     0 / 13
    bash 5.3.15      docker bash:5.3     0 / 13
    busybox ash      docker bash:5.3     0 / 15
    dash             debian:stable-slim  0 / 15
    dash             macOS /bin/dash     0 /  9      (2 option sets inadmissible)
    zsh 5.9          macOS /bin/zsh      0 / 13      (2 cells: trap never ran)

**The laundering is a bash 3.2 defect, fixed by 4.4.** That is not a footnote:
every one of these repos runs its gate scripts in CI on `ubuntu-latest`
(bash 5.x) — where an aborting gate exits 1 or 2 and the job FAILS — while
developers and agents run the same scripts on macOS, where the abort exits 0.
The laundering is real, and it is a *workstation* defect, not a CI one, except
where a workflow declares `runs-on: macos-latest` (three of katgpt-rs's do,
`full_gate.yml` among them). The completion sentinel is still the right repair:
it costs nothing on 5.x and it is load-bearing on every macOS run.

# The cell that decides it

For each (`set` options) x (abort arm) x (EXIT trap registered or not) the
driver records four quantities, and the third is the one that decides the
verdict while the fourth decides whether the cell means anything at all:

* `exit`     — the script's status as the caller sees it.
* `post`     — did execution continue past the arm? (0 = it aborted)
* `dollarq`  — `$?` as observed INSIDE the EXIT trap.
* `adm`      — did this shell ACCEPT the option line? dash rejects
               `-o pipefail` (before 0.5.12), dies on line 1 with status 2,
               and all eight of those cells then look like an abort whose
               trap never ran. Pooled, they report a confident 0 over a
               measurement that never happened; they are excluded and named.

Two shapes that are NOT laundering and are not passes either, reported
separately rather than folded into the zero: `trap never ran` (measured on
zsh's nounset abort — the status survives but the CLEANUP is skipped, so the
temp dir leaks) and `INADMISSIBLE` above.

A **laundering cell** is `trap=1` AND `post=0` (it really aborted) AND
`dollarq=0` (the handler was told the run succeeded). That is the whole
defect: a cleanup handler whose last command succeeds then exits 0 and the
abort reads as a pass. `exit` alone cannot identify it — a handler that
happens to end in a failing command masks the same laundering.

# Reading a zero

An interpreter with 0 laundering cells is not "safe shell": `set -u` still
aborts, and everything after the abort still does not run. What changes is
only whether the *status* survives. And a 0 from a container is a 0 for that
image's bash — not for the shell your `#!` line resolves to.

Usage:

    scripts/trap_launder_premise_matrix.py                  # this box: bash + sh
    scripts/trap_launder_premise_matrix.py --shell /bin/dash
    scripts/trap_launder_premise_matrix.py --docker         # the pinned image sweep
    scripts/trap_launder_premise_matrix.py --tsv            # raw rows, for diffing

bash 3.2 is available in NO docker image (there is no `bash:3.2` tag — the
premise's own interpreter can only be measured on macOS itself). That
asymmetry is why this script measures the LOCAL box first and always.
"""

import argparse
import os
import shutil
import subprocess
import tempfile

# ── The matrix ────────────────────────────────────────────────────────────
# `set -u` and `set -uo pipefail` are in the list because the audit's
# population predicate was `set -u`: measuring nounset WITHOUT errexit is how
# that predicate was shown to be the wrong one (errexit is the precondition).
OPTS = [
    "set -u",
    "set -e",
    "set -eu",
    "set -uo pipefail",
    "set -euo pipefail",
]
ARMS = {
    "unbound": 'echo "$DEFINITELY_UNSET_XYZ"',
    "evalsyntax": 'eval "if ["',
    "cmdfail": "false",
    "notfound": "no_such_command_xyz",
}

# One POSIX-sh driver, emitted for both the local and the container path — the
# container images (alpine, debian-slim) have no python, and a second
# implementation would be a second thing to keep true.
DRIVER = r"""#!/bin/sh
# Emitted by trap_launder_premise_matrix.py. POSIX sh only: this runs under
# dash and busybox ash as well as bash.
#
# Every row carries an ADMISSIBILITY bit, and it is not decoration: dash has
# no `pipefail`, so `set -uo pipefail` makes dash die on line 1 with status 2
# BEFORE the trap is registered and before the arm runs. All eight of those
# cells then look like "aborted, trap never ran" — and a laundering count that
# pools them reports a confident 0 over a measurement that never happened.
# The probe runs the option line alone and asks whether the shell accepted it.
SH=${1:-bash}
V=$("$SH" -c 'echo "${BASH_VERSION:-POSIX-sh}"' 2>/dev/null || echo unknown)
W=$(mktemp -d)
for opts in %(OPTS)s; do
  { echo "$opts"; echo 'echo probe_ok'; } > "$W/p.sh"
  pout=$("$SH" "$W/p.sh" 2>&1)
  adm=0; case $pout in *probe_ok*) adm=1;; esac
  for arm in %(ARMNAMES)s; do
    case $arm in
%(ARMCASES)s
    esac
    for t in 0 1; do
      {
        echo "$opts"
        if [ "$t" = 1 ]; then
          printf 'trap %%s EXIT\n' "'q=\$?; echo \"DOLLARQ=\$q\" >&2; true'"
        fi
        echo "$body"
        echo 'echo post'
      } > "$W/t.sh"
      out=$("$SH" "$W/t.sh" 2>&1); st=$?
      post=0; case $out in *post*) post=1;; esac
      dq=-;   case $out in *DOLLARQ=*) dq=${out#*DOLLARQ=}; dq=${dq%%%%
*};; esac
      printf '%%s\t%%s\t%%s\t%%s\t%%s\t%%s\t%%s\t%%s\t%%s\n' \
        "$SH" "$V" "$opts" "$arm" "$t" "$st" "$post" "$dq" "$adm"
    done
  done
done
rm -rf "$W"
"""

# Pinned sweep. `bash:3.2` does not exist; debian's /bin/sh is dash and the
# bash image is alpine, whose /bin/sh is busybox ash — the two POSIX shells a
# `#!/bin/sh` script in this workspace actually resolves to on Linux.
IMAGES = [
    ("bash:4.4", "bash"),
    ("bash:5.0", "bash"),
    ("bash:5.2", "bash"),
    ("bash:5.3", "bash"),
    ("bash:5.3", "/bin/sh"),           # busybox ash
    ("debian:stable-slim", "/bin/dash"),
    ("debian:stable-slim", "/bin/bash"),
]


def driver_text():
    arm_cases = "\n".join(
        f"      {name}) body='{body}' ;;" for name, body in ARMS.items()
    )
    return DRIVER % {
        "OPTS": " ".join(f"'{o}'" for o in OPTS),
        "ARMNAMES": " ".join(ARMS),
        "ARMCASES": arm_cases,
    }


def parse(tsv):
    """TSV -> rows. A row is a dict; `launders` is the decisive cell.

    Four mutually exclusive shapes per trapped cell, and pooling any two of
    them is how a report goes blind:

    * `launders`      — aborted, and `$?` reached the handler as 0.
    * `preserved`     — aborted, and the handler saw the real status.
    * `trap_not_run`  — aborted and the EXIT trap never fired at all. The
                        status may be fine; the CLEANUP did not happen, which
                        is a different defect (leaked temp dirs), not a pass.
    * `inadmissible`  — the shell rejected the option line, so the arm never
                        ran. Excluded from every count; reported by name.
    """
    rows = []
    for line in tsv.splitlines():
        f = line.split("\t")
        if len(f) != 9:
            continue
        sh, ver, opts, arm, trap, st, post, dq, adm = f
        trapped, aborted, admitted = trap == "1", post == "0", adm == "1"
        rows.append({
            "shell": sh, "version": ver, "opts": opts, "arm": arm,
            "trap": trapped, "exit": st, "post": post == "1", "dollarq": dq,
            "admissible": admitted,
            "launders": admitted and trapped and aborted and dq == "0",
            "trap_not_run": admitted and trapped and aborted and dq == "-",
            "preserved": admitted and trapped and aborted and dq not in ("0", "-"),
        })
    return rows


def run_local(driver, shell):
    try:
        p = subprocess.run(["/bin/sh", driver, shell],
                           capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"{e}"
    return parse(p.stdout), None


def run_docker(driver, image, shell):
    d = os.path.dirname(driver)
    cmd = ["docker", "run", "--rm", "-v", f"{d}:/w", "-w", "/w",
           image, "/bin/sh", f"/w/{os.path.basename(driver)}", shell]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"{e}"
    if p.returncode != 0 and not p.stdout.strip():
        return None, (p.stderr.strip().splitlines() or ["no output"])[-1]
    return parse(p.stdout), None


def report(label, rows, baseline=None):
    """One block per interpreter; the laundering cells are named, not counted.

    A zero is only reportable next to the size of the measurement it came
    from — `aborts` is that denominator. Zero laundering out of zero observed
    aborts is an inert instrument, not a clean interpreter, and it prints as
    NO ABORTS OBSERVED rather than as a pass.
    """
    if rows is None:
        print(f"  {label:<30} UNSEEN — not a zero: {baseline}")
        return None
    ver = rows[0]["version"] if rows else "?"
    launder = [r for r in rows if r["launders"]]
    notrun = [r for r in rows if r["trap_not_run"]]
    aborts = [r for r in rows if r["admissible"] and r["trap"] and not r["post"]]
    inadm = sorted({r["opts"] for r in rows if not r["admissible"]})
    if not aborts:
        verdict = "NO ABORTS OBSERVED — inert, not clean"
    elif launder:
        verdict = "LAUNDERS"
    else:
        verdict = "status preserved"
    print(f"  {label:<30} {ver:<20} {len(launder):>2}/{len(aborts):<3} {verdict}")
    for r in launder:
        print(f"       LAUNDERS   {r['opts']:<18} {r['arm']:<12}"
              f" exit={r['exit']}  $?@trap={r['dollarq']}")
    for r in notrun:
        print(f"       trap never ran (cleanup skipped, status kept): "
              f"{r['opts']:<18} {r['arm']:<12} exit={r['exit']}")
    for o in inadm:
        print(f"       INADMISSIBLE — this shell rejects `{o}`; its 8 cells "
              f"measure nothing and are excluded")
    return launder


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--shell", action="append", default=None,
                    help="interpreter to measure locally (repeatable)")
    ap.add_argument("--docker", action="store_true",
                    help="also sweep the pinned images (needs a running daemon)")
    ap.add_argument("--tsv", action="store_true", help="emit raw rows only")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="trap_premise_")
    driver = os.path.join(tmp, "driver.sh")
    with open(driver, "w", encoding="utf-8") as fh:
        fh.write(driver_text())
    os.chmod(driver, 0o755)

    shells = args.shell or [s for s in ("bash", "/bin/sh", "/bin/dash", "/bin/zsh")
                            if shutil.which(s) or os.path.exists(s)]

    all_rows = []
    if not args.tsv:
        print("── laundering premise, per interpreter ───────────────────────────")
        print("  a LAUNDERING CELL = it really aborted (post=0) and the EXIT trap\n"
              "  was handed `$?` == 0, so a cleanup handler that succeeds exits 0\n")
        print(f"  {'interpreter':<30} {'version':<20} {'laundering/aborts'}")
    for sh in shells:
        rows, err = run_local(driver, sh)
        if rows:
            all_rows += rows
        if args.tsv:
            continue
        report(f"local {sh}", rows, err)

    if args.docker:
        if not shutil.which("docker"):
            if not args.tsv:
                print("\n  docker: NOT INSTALLED — the non-3.2 half is UNSEEN, which is\n"
                      "  not the same as measured-zero. Re-run where a daemon exists.")
        else:
            probe = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                                   capture_output=True, text=True)
            if probe.returncode != 0:
                if not args.tsv:
                    print("\n  docker: daemon NOT RUNNING — the non-3.2 half is UNSEEN.")
            else:
                if not args.tsv:
                    print()
                for image, sh in IMAGES:
                    rows, err = run_docker(driver, image, sh)
                    if rows:
                        all_rows += rows
                    if args.tsv:
                        continue
                    report(f"{image}  {sh}", rows, err)

    if args.tsv:
        for r in all_rows:
            print("\t".join([r["shell"], r["version"], r["opts"], r["arm"],
                             str(int(r["trap"])), r["exit"], str(int(r["post"])),
                             r["dollarq"], str(int(r["admissible"]))]))
        return

    print("\n── what a zero does and does not say ────────────────────────────")
    print("  A 0 means the STATUS survives the abort on that interpreter. The\n"
          "  abort still happens and everything after it still does not run —\n"
          "  the completion sentinel is what makes that visible either way.\n"
          "  bash 3.2 is the only interpreter measured to launder, and it is the\n"
          "  one macOS ships as /bin/bash. CI on ubuntu-latest never laundered;\n"
          "  a workflow with `runs-on: macos-latest` does. Report only; exit 0.")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
