#!/usr/bin/env python3
"""Gate over `percentile_index_audit.py`, katgpt-rs scope only.

The report and the gate are deliberately separate files, for the reason
`cfg_gated_target_audit.py` and `cfg_gated_floor_gate.py` are: the report must
stay runnable over sibling repos whose owners have not taken this class of
work, and a report that exits 1 on them is a report nobody runs.

What this adds over the report: a `DEGENERATE` site introduced by a new commit
reds the push that adds it, **before** its number is quoted in a
`.benchmarks/` table as though it were a tail. Print-only or asserted — a
misleading number in a benchmark doc is the input to somebody's promote/demote
decision.

Exit 0 = pass, 1 = a pin moved, 2 = the instrument itself is untrustworthy
(the auditor's own `selftest()` failed, in which case no verdict is possible).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import percentile_index_audit as pia  # noqa: E402

REPO = os.path.dirname(HERE)
PINS = os.path.join(HERE, "percentile_floors.txt")


def read_pins(path):
    pins = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            key, _, val = line.partition("=")
            pins[key.strip()] = int(val.strip())
    return pins


def main():
    # Prints carry glyphs the Windows locale codecs cannot encode (checked
    # 2026-09-06 on cp874: check/cross/middot/arrow FAIL, em-dash OK); keep the
    # locale encoding and degrade only the fatal chars to escapes -- the
    # staged_set_audit house pattern (utf-8 pinning would mojibake legacy consoles).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (embedded / detached); keep old behavior
    # The auditor's selftest runs first and exits 2 on failure. Without it a
    # tokenizer regression would take every count to zero and this gate would
    # certify the repo clean on the strength of an instrument that had gone
    # blind.
    pia.selftest()

    pins = read_pins(PINS)
    findings = []
    for f in pia.walk_rs(REPO):
        findings += pia.audit_file(f, os.path.relpath(f, REPO))

    # DRY: the classification lives in the report, so this gate and the
    # cross-repo sweep can never disagree about what a DEGENERATE is. The
    # WEAK/TRUNC_VAR asymmetry (WEAK counted only when `asserted`, TRUNC_VAR
    # regardless) is documented at `pia.tally`.
    t = pia.tally(findings)
    degenerate, deg_asserted = t["degenerate"], t["degenerate_asserted"]
    weak_asserted, trunc_var = t["weak_asserted"], t["trunc_var"]

    measured = {
        "max_degenerate": len(degenerate),
        "max_degenerate_asserted": len(deg_asserted),
        "max_weak_asserted": len(weak_asserted),
        "max_trunc_var": len(trunc_var),
        "min_sites_scanned": len(t["sites"]),
    }

    failures = []
    for key, got in measured.items():
        pinned = pins.get(key)
        if pinned is None:
            failures.append(f"{key}: no pin in {os.path.basename(PINS)}")
            continue
        over = key.startswith("max_") and got > pinned
        under = key.startswith("min_") and got < pinned
        if over or under:
            failures.append(
                f"{key}: measured {got}, pinned {'<= ' if over else '>= '}{pinned}"
            )

    label = "percentile floor gate"
    if failures:
        print(f"✗ {label} FAILED")
        for f in failures:
            print(f"    {f}")
        for r in degenerate + weak_asserted + trunc_var:
            if r["verdict"] == pia.TRUNC_VAR:
                # p is a parameter here, so p/n/idx/support are all None and
                # printing them tells the reader nothing. The line is the finding.
                print(f"      {r['file']}:{r['line']}  {r['text']}")
            else:
                print(
                    f"      {r['file']}:{r['line']}  p={r['p']} n={r['n']} "
                    f"idx={r['idx']} support={r['support']} "
                    f"asserted={r['asserted']}"
                )
        print("    A 'p99' whose index is n-1 IS the max. Use nearest rank")
        print("    (ceil(p*n)-1) and report tail support, or drop the column")
        print("    when the sample count cannot support the quantile at all.")
        print("    See AGENTS.md § A reported \"p99\" is often the MAX.")
        return 1

    print(
        f"    ✓ {label} PASSED — {len(pins)} pins held "
        f"({measured['min_sites_scanned']} sites scanned, "
        f"0 degenerate, 0 asserted-weak, 0 trunc-var)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
