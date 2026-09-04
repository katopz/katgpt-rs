#!/bin/sh
# scripts/test_gate.sh — the first CI-executed tests in this repo
# (katgpt-rs Issue 718 T3(b), the riir-train 507 shape; Issue 718 closed +
# removed 2026-09-04 — durable record in
# `.docs/10_audits/ci_compile_vs_execute_axis.md`).
#
# Before this gate, every automatic trigger here (full_gate.yml,
# feature_isolation*.yml, docs_gate.yml, lean_proofs.yml) reached only
# `cargo check` / `cargo clippy` / the Python auditors — CI compiled all
# 477 integration-test targets, 31 lib targets and 176 bench targets over
# 32 packages and EXECUTED none, so by this repo's own rule that an
# uninvoked assertion is unknown, not passing, every Rust assertion here
# was unknown. This gate makes the machine-invariant core EXECUTED and
# FLOORED. It is the AUTHORIZED scoped tier (owner decision recorded in
# `.docs/10_audits/ci_compile_vs_execute_axis.md`, 2026-09-04);
# the full-workspace `--all-features --release` execution stays
# dispatch-only. It has now been PRICED on a quiet box
# (`.benchmarks/701_full_workspace_execution_pricing.md`) and the verdict
# was still dispatch-only, on that measurement. No scheduled full job.
#
# Scope (and what it deliberately does NOT cover):
#   COVERED     — the default-feature `--lib` suites of katgpt-rs (root)
#                 and katgpt-core: 203 + 1974 passed at landing, pure
#                 modelless CPU, zero sibling checkouts (katgpt-core's
#                 deps are crates.io-only; the root builds its own member
#                 crates), no model files, no GPU.
#                 Platform invariance, grep-verified at landing: katgpt-core
#                 has ZERO `#[cfg(target_os)]` attributes (its two
#                 `target_os` sites are runtime `cfg!()` bools — both
#                 branches compile everywhere), and the root lib's
#                 `target_os` gates are all behind the opt-in
#                 `ane`/`gpu_inference` features, dead at default features
#                 on every platform — so the floored counts are expected
#                 to be platform-invariant. The first scheduled run is the
#                 measurement: if Linux deltas surface, that is the rot
#                 check finding real debt, not a reason to widen silently.
#   NOT COVERED — the 477 integration-test targets (each carries
#                 required-features or multi-minute single tests — priced
#                 out of a weekly gate; the floors here make expanding
#                 this gate a one-line ROWS addition), the 176 bench
#                 targets, and everything needing Metal/ANE/4090 — those
#                 stay workstation-owned.
#
# T3 shape: per-target FLOORS, not exact pins. A floor fires DOWNWARD
# only — adding tests never reds the gate; deleting tests, a feature
# change that compiles a lib to nothing, or a broken build reds it. A
# target that produces NO `test result:` line also fails (the
# `#![cfg]`-gated-file green-zero trap: `ok. 0 passed` with exit 0 is
# byte-for-byte a real pass, and a skipped target produces no line at all).
#
# --canary: runs the first row with a floor of 100000 and asserts the gate
# FAILS — the proof the floors are live (same comparison path as a
# blind/zero target, without mutating the tree). Dev-time; too costly to
# run in CI every week.
#
# Parse discipline: exactly ONE `test result:` line is expected per `--lib`
# invocation. More or fewer means the parse has gone blind — fail loudly
# rather than sum or guess.
#
# Floors measured 2026-09-04 on committed-HEAD-equivalent working tree
# (debug, M3): katgpt-rs 203 passed / 0 failed (30.9 s), katgpt-core
# 1974 passed / 0 failed / 7 ignored (11.0 s). Raising a floor is a
# measured act; lowering one needs a note in the commit that does it.
#
# --test-threads=2 is deliberate (the riir-train 507 precedent): a weekly
# red on runner-load noise from a timing-sensitive test would be alarm
# fatigue; 2 threads costs ~2x wall on a ~40 s suite.

set -u

ROWS="
katgpt-rs:203
katgpt-core:1974
"

canary=0
if [ "${1:-}" = "--canary" ]; then
    canary=1
    echo "canary: running the first row with floor=100000 — the gate MUST fail"
fi

fail=0
first=1
for row in $ROWS; do
    pkg=${row%%:*}
    floor=${row##*:}
    if [ "$canary" = 1 ] && [ "$first" = 1 ]; then
        floor=100000
    fi
    first=0

    echo "=== $pkg --lib (floor $floor, --test-threads=2) ==="
    if ! out=$(cargo test -p "$pkg" --lib -- --test-threads=2 2>&1); then
        echo "FAIL $pkg: cargo test exited non-zero"
        printf '%s\n' "$out" | tail -20
        fail=1
        continue
    fi

    n=$(printf '%s\n' "$out" | awk '$1 == "test" && $2 == "result:" { for (i = 3; i <= NF; i++) if ($i == "passed;") print $(i-1) }')
    nlines=$(printf '%s\n' "$n" | grep -c .)
    if [ "$nlines" != "1" ]; then
        echo "FAIL $pkg: expected exactly 1 'test result:' line for --lib, got $nlines — the parse is blind or the target shape changed"
        fail=1
        continue
    fi

    echo "passed=$n floor=$floor"
    if [ "$n" -lt "$floor" ]; then
        echo "FAIL $pkg: passed $n < floor $floor — a target reporting fewer assertions than its floor is blind, skipped, or broken"
        fail=1
    fi
done

if [ "$fail" = 1 ]; then
    echo "test_gate: FAIL"
    exit 1
fi
if [ "$canary" = 1 ]; then
    echo "test_gate: canary UNEXPECTEDLY PASSED — the floors did not fire, the gate is vacuous"
    exit 1
fi
echo "test_gate: PASS"
