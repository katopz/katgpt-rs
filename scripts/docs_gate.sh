#!/usr/bin/env bash
# Docs gate — the manifest/doc/skill drift assertions, as one command.
# The count is deliberately NOT written here: CHECKS below is the list, and a
# prose count beside a list is the drift this repo keeps rediscovering.
#
# Why this file exists: all three checks already existed, and NOTHING ran any of
# them. Measured 2026-09-01, two of the three were RED on `develop`:
#
#   count_features.py       green, but only because it had just been fixed; it
#                           checked ONE README site out of five and two of 29
#                           manifests.
#   bench_doc_audit.py      exit 1 — a false positive on a doc that correctly
#                           recorded "opt-in ... promoted to DEFAULT-ON".
#   cargo_comment_audit.py  exit 1 — a false positive from a case-SENSITIVE
#                           "Opt-in" regex that missed the repo's 32 "OPT-IN"
#                           comment lines.
#
# An assertion nobody invokes is decoration, and a red one nobody invokes is
# worse: it trains the next reader to assume the tool is broken.
#
# Cost: ~3s total. That is what makes per-push affordable here, in deliberate
# contrast to scripts/full_gate.sh (>13 min, weekly). It was ~556s before the
# manifest walk was pruned — `rglob("Cargo.toml")` descended into target/
# (117 GB, ~1.3M entries) and filtered afterwards, four times per run.
#
# Unlike the full gate this is platform-INDEPENDENT: pure Python over manifests
# and markdown, no cfg(target_os) surface, so ubuntu is correct and macOS would
# only cost more. Don't "fix" it to macos-latest.
#
# WORKSTATION REQUIREMENTS (measured on the partial 4090 box 2026-09-04, where
# 5 of 8 checks red for environment, not drift): (1) python3 >= 3.11 on PATH —
# cfg_gated_target_audit.py imports tomllib (3.10 lacks it; the Windows Store
# python3 alias also shadows real installs); (2) PYTHONIOENCODING=utf-8 — a
# cp874/cp1252 console cannot print the gates' checkmark output and the failure
# masquerades as a gate failure; (3) a FULL workspace checkout —
# skill_repo_set_gate.py re-derives the live repo set and FAILS on repos the
# box simply has not cloned (10 of 16 here, incl. riir-mmorpg-examples/riir-dao)
# — regenerating repo_set.txt on a partial box would corrupt the canonical set;
# the M3 is the canonical full workstation for that half.
#
# skill_repo_set_gate.py (added 2026-09-01, Issue 703) has a second axis the
# other three do not: it reads SIBLING repos, which CI does not have. It does
# NOT skip there — it separates its VOCABULARY (committed snapshot,
# scripts/repo_set.txt) from its POPULATION (the SKILL.md it can actually see),
# prints both, and the workstation run re-derives the snapshot and FAILS on
# drift. So CI checks this repo's 8 skills against all 18 repo names, and says
# out loud that it saw 8 of 12. A gate that skipped instead would be the
# vacuous green it exists to catch.
#
# cfg_gated_floor_gate.py (added 2026-09-03, Issue 713) is katgpt-rs-SCOPED on
# purpose, unlike the sibling-reading check above it. Its instrument
# (cfg_gated_target_audit.py) audits any repo, but CI has a single checkout, so
# a cross-repo version would derive an empty population and print a confident
# green over zero repos — the same defect it exists to catch, which is also why
# docs_drift_sweep.py is deliberately absent from CHECKS. Sibling coverage is
# Issue 713 T3, an owner call per repo. Its pins are two-sided (two ceilings +
# two blindness floors) because a ceiling cannot fail once the auditor goes
# blind and reports zero; see scripts/cfg_gated_floors.txt.
#
# orphaned_attr_gate.py (added 2026-09-03) is pinned at ZERO offenders: the
# shape it forbids -- an OUTER #[cfg] separated from its item by a blank line,
# which Rust still binds to that item -- measured zero sites across every
# contract repo at the fix (19 then; re-measured 2026-09-04 over the live 16,
# still zero everywhere). It exists because that shape sat in katgpt-pruners for
# two days and broke every RELEASE build of `sdar_gate` (26d055c6 -> a08376a0),
# while the commit that introduced it validated in debug and reported 597/0.
#
# It DOES have floors, added 2026-09-04, and the paragraph above is why: a zero
# ceiling cannot fail once the walk goes blind. The reasoning was written here
# for docs_drift_sweep.py and not applied to the check three lines below it.
# Two population floors (.rs files, outer-#[cfg] sites), katgpt-rs-scoped --
# riir-viewbridge has 24 .rs files against this repo's 2,418, so a shared floor
# would red every small sibling forever. The same commit stopped its PASS line
# printing "measured 0 across 19 repos", a cross-repo claim no single-repo run
# had made, two lines under the repo-set gate correctly saying 16.
#
# Runs every check even after one fails — the same reason full_gate.sh passes
# --keep-going: stopping at the first failure under-reports the drift.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKS=(
    "scripts/count_features.py:flag counts in README + examples/README vs every manifest"
    "scripts/bench_doc_audit.py:(default-on|opt-in) labels in .benchmarks + .docs vs Cargo defaults"
    "scripts/cargo_comment_audit.py:inline Cargo.toml comments vs the default closure"
    "scripts/skill_repo_set_gate.py:hand-typed repo sets in SKILL.md command blocks (Issue 703)"
    "scripts/agents_repo_set_gate.py:AGENTS.md §Repo count membership vs scripts/repo_set.txt"
    "scripts/cfg_gated_floor_gate.py:#![cfg]-gated targets that report a green 0-pass (Issue 713)"
    "scripts/orphaned_attr_gate.py:a #[cfg] separated from its item by a blank line (a08376a0)"
    "scripts/percentile_floor_gate.py:a percentile index that lands on n-1 and so reports the MAX"
    "scripts/numbering_gate.py:a .plans/.issues/.research/.proposals number allocated twice, or a stale/malformed .highwater (Issues 724, 725)"
    "scripts/docs_gate_paths_sync.py:docs_gate.yml's two hand-duplicated trigger paths lists stay identical (Issue 724 T4b)"
    "scripts/required_features_static_gate.py:a required-features row naming a feature its package cannot enable (Issue 513)"
    "scripts/cfg_row_implication_gate.py:a required-features row that BUILDS and compiles its target to NOTHING (Issue 513)"
    "scripts/population_sync_gate.py:the six independent contract-repo predicates must agree (else an instrument audits a different set and still prints green)"
    "scripts/trap_sentinel_gate.py:a shell gate whose set -u abort would report exit 0 — this repo's own two, by membership (Issue 734)"
)

if ! command -v python3 >/dev/null 2>&1; then
    echo "✗ python3 not found — docs gate cannot run"
    exit 1
fi

# ── This gate times ITSELF ──────────────────────────────────────────────────
# The duration used to be hand-typed in AGENTS.md, and a hand-typed duration
# drifts exactly like a hand-typed count. It was also the wrong quantity: this
# gate's WALL time is contention-dominated (measured: 12.65s / 12.52s / 12.69s
# CPU on runs whose WALL was 128.3s / 299.1s / 15.0s — a 20x wall spread over
# 1.4% of CPU spread), and which
# check absorbs the wait moves between runs. So print BOTH — the per-check
# wall time names whichever check is blocking today, and the CPU total is the
# load-invariant figure to compare across runs. Numbers and the measured
# non-explanation live in AGENTS.md §Docs gate, not duplicated here.
# `$EPOCHREALTIME` is bash >= 5.0 and this box is 3.2.57, so the stamp goes
# through python3 — already a hard dependency three lines above.
now() { python3 -c 'import time; print("%.2f" % time.time())'; }
GATE_T0="$(now)"

failed=0
for entry in "${CHECKS[@]}"; do
    script="${entry%%:*}"
    what="${entry#*:}"
    if [ ! -f "$script" ]; then
        # A missing check is a failure, not a skip: silently dropping a check is
        # how this gate would rot back into the state that motivated it.
        echo "✗ $script — MISSING (expected: $what)"
        failed=$((failed + 1))
        continue
    fi
    echo "▸ $script — $what"
    check_t0="$(now)"
    if out="$(python3 "$script" 2>&1)"; then
        printf '%s\n' "$out" | tail -1 | sed 's/^/    /'
    else
        failed=$((failed + 1))
        printf '%s\n' "$out" | sed 's/^/    /'
        echo "  ✗ $script FAILED"
    fi
    check_dt="$(python3 -c "print('%.1f' % ($(now) - $check_t0))")"
    case "$check_dt" in
        # Only the slow ones are worth a line; the rest are noise at 0.0-0.9s.
        0.*) ;;
        *) echo "    ⏱  ${check_dt}s wall" ;;
    esac
done

# CPU is the load-invariant total (`times` reports this shell + its children);
# wall is what the operator experiences. A large gap means the box was busy —
# compare CPU across runs before concluding a check got slower.
gate_wall="$(python3 -c "print('%.1f' % ($(now) - $GATE_T0))")"

# `times` must be REDIRECTED, never captured. Measured on bash 3.2.57 against
# a child that burned 0.167s of user time:
#     times                 -> 0m0.001s 0m0.002s / 0m0.167s 0m0.015s   correct
#     times > file          -> 0m0.001s 0m0.002s / 0m0.168s 0m0.024s   correct
#     times | sed 's/^/ /'  -> 0m0.000s 0m0.000s / 0m0.000s 0m0.000s   ZERO
#     $(times | tail -1)    -> 0m0.000s 0m0.000s                       ZERO
# A pipeline forks and a command substitution forks, and the fork has no
# children of ITS own, so it reports zero however much CPU the checks burned —
# even piping through `sed` purely to indent destroys the number. A
# redirection does not fork, so capture once and then format and assert from
# the file as freely as you like. The first two versions of this block piped,
# and printed a confident 0m0.000s next to a 308s run: the "inert instrument
# reports a clean number" failure the gates in this directory exist to catch,
# occurring in the code that measures them.
# No EXIT trap for the temp file ON PURPOSE — registering one would put this
# script into trap_exit_launder_audit.py's population (Issue 734), and the
# only cost of not having one is a single stray file if the gate is killed.
times_out="$(mktemp)"
times > "$times_out"
gate_cpu="$(awk 'NR==2 { t=0; for (i=1;i<=NF;i++) { split($i, p, "m"); sub("s","",p[2]); t += p[1]*60 + p[2] } printf "%.2f", t }' "$times_out")"
echo "  ⏱  total ${gate_wall}s wall · ${gate_cpu}s CPU in the checks (rows: this shell, then the checks)"
sed 's/^/     /' "$times_out"

# ...and the instrument must prove itself NON-INERT, because the failure mode
# above is a well-formed number, not an error. N python3 checks cannot burn
# ~no CPU: if the total reads as ~0 across a multi-second run, the measurement
# broke and the figure must not be quoted.
if [ "$(awk -v c="$gate_cpu" -v w="$gate_wall" 'BEGIN { print (c < 0.05 && w > 5) ? "INERT" : "OK" }')" = "INERT" ]; then
    echo "  ⛔ the CPU figure above is NOT a measurement: ${gate_cpu}s of CPU across"
    echo "     ${gate_wall}s of wall is impossible for ${#CHECKS[@]} python3 checks."
    echo "     The times builtin was read from a forked context — do not quote it."
fi
rm -f "$times_out"
echo "     CPU is the load-invariant figure: 12.65s / 12.52s / 12.69s measured"
echo "     on runs whose WALL was 128.3s / 299.1s / 15.0s — 20x wall, 1.4% CPU."
echo "     Cite the CPU number; read wall as a range, never as a baseline."

if [ "$failed" -ne 0 ]; then
    echo "✗ docs gate FAILED — $failed of ${#CHECKS[@]} check(s)"
    exit 1
fi
echo "✓ docs gate PASSED — ${#CHECKS[@]}/${#CHECKS[@]} checks clean"
