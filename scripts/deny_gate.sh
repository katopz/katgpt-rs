#!/bin/sh
# deny_gate.sh — the supply-chain advisories gate (cargo-deny).
#
# Landed as the riir-refine pilot of Batch 294 / Plan 298 / Research 332
# (the wasmtime RUSTSEC-2026-0380 / RUSTSEC-2026-0379 distill). Architecture:
# cargo-deny owns vulnerable CRATES (lockfile surface); the sec corpus owns
# vulnerable USAGE (code-span classes). The clippy analogy, corrected: deny is
# a LOCKFILE oracle feeding (a) this gate and (b) the sec-corpus transcription
# queue (advisory -> class -> rule when the fix is code-shaped) — never a
# span-heal oracle.
#
# Modes:
#   scripts/deny_gate.sh           # gate THIS repo — vulnerability findings exit 1
#   scripts/deny_gate.sh --scan    # survey every /Users/kat/git/*/ workspace with
#                                  # a Cargo.lock — REPORT-ONLY, always exit 0
#
# A missing cargo-deny is a LOUD SKIP (a deferral, never a green):
#   cargo install cargo-deny --locked
#
# Known scope gap (Plan 298 fan-out): nested excluded workspaces with their own
# Cargo.lock (e.g. seal-game-editor/tools/neuron-publish, tools/seal-e2e) are
# audited per-repo only where their own gate runs — the --scan loop visits
# top-level dirs only. A stale lockfile (manifest moved, lock not updated)
# makes cargo-deny ERROR, not report advisories — such a repo prints
# "ERROR (stale lock?)" and needs `cargo update -p <changed>` or a re-lock,
# never a silent pass.

set -u

complete=deny_gate_incomplete
cleanup() {
    rc=$?
    if [ "$complete" = "deny_gate_incomplete" ]; then
        echo "ABORTED: deny_gate did not reach its last line (exit $rc)" >&2
        rc=1
    fi
    exit $rc
}
trap cleanup EXIT

deny_bin=$(command -v cargo-deny 2>/dev/null)
if [ -z "$deny_bin" ] && [ -x "$HOME/.cargo/bin/cargo-deny" ]; then
    deny_bin="$HOME/.cargo/bin/cargo-deny"
fi
if [ -z "$deny_bin" ]; then
    echo "SKIP LOUD: cargo-deny not installed — run: cargo install cargo-deny --locked"
    complete=deny_gate_complete
    exit 0
fi

mode=gate
if [ "${1:-}" = "--scan" ] || [ "${1:-}" = "--workspace" ]; then
    mode=scan
fi

if [ "$mode" = "gate" ]; then
    if [ ! -f Cargo.lock ]; then
        echo "SKIP LOUD: no Cargo.lock at $(pwd) — nothing to audit"
        complete=deny_gate_complete
        exit 0
    fi
    echo "== deny_gate (advisories): $(pwd)"
    if "$deny_bin" check advisories; then
        echo "PASS: no deny-level advisory findings"
        rc_total=0
    else
        echo "FAIL: advisory findings above — adjudicate each (fix/bump, or pin with a reason in deny.toml)"
        rc_total=1
    fi
else
    echo "repo|verdict"
    # Derive the workspace root from THIS script's location (<repo>/scripts/) so
    # the survey runs on any box layout (the pilot's /Users/kat literal broke on
    # /Users/katopz). Siblings of the repo's parent dir = the git farm.
    git_root=$(cd "$(dirname "$0")/.." && pwd)
    farm=$(dirname "$git_root")
    for d in "$farm"/*/; do
        [ -f "${d}Cargo.lock" ] || continue
        name=$(basename "$d")
        out=$(cd "$d" && "$deny_bin" check advisories 2>&1)
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "${name}|CLEAN"
        else
            ids=$(printf '%s' "$out" | grep -o "RUSTSEC-[0-9][0-9][0-9][0-9]-[0-9]*" | sort -u | paste -sd, -)
            if [ -n "$ids" ]; then
                echo "${name}|FINDINGS ${ids}"
            else
                echo "${name}|ERROR (stale lock? tool failure — read the output, never a silent pass)"
            fi
        fi
    done
    echo "scan complete — FINDINGS rows are survey output, adjudicate each; --scan never fails a run"
    rc_total=0
fi

complete=deny_gate_complete
exit $rc_total
