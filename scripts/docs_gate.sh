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
)

if ! command -v python3 >/dev/null 2>&1; then
    echo "✗ python3 not found — docs gate cannot run"
    exit 1
fi

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
    if out="$(python3 "$script" 2>&1)"; then
        printf '%s\n' "$out" | tail -1 | sed 's/^/    /'
    else
        failed=$((failed + 1))
        printf '%s\n' "$out" | sed 's/^/    /'
        echo "  ✗ $script FAILED"
    fi
done

if [ "$failed" -ne 0 ]; then
    echo "✗ docs gate FAILED — $failed of ${#CHECKS[@]} check(s)"
    exit 1
fi
echo "✓ docs gate PASSED — ${#CHECKS[@]}/${#CHECKS[@]} checks clean"
