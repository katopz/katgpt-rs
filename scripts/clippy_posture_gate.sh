#!/bin/sh
# scripts/clippy_posture_gate.sh — the narrow-feature clippy-clean fence
# (katgpt-rs Issue 1219 direction (3)).
#
# full_gate.sh Layer 3 lints the workspace at --all-features, which is CLEAN
# by construction there: every opt-in feature is ON, so no constructor is
# dead. The narrow postures standing lanes compile (the 917c06c83 landing
# receipt pinned "clippy -D warnings both postures" — dash_attn-only +
# the zha_chunk_sidecar feature posture) are compiled by NO standing
# instrument, and Issue 1219 is the measured consequence: a genuinely
# unreachable-at-that-posture constructor (`from_ref_major`, sole caller
# behind `zha_chunk_sidecar`) warns dead_code at dash_attn-only — a
# since-birth posture artifact (11-15 evidence addendum), carried by the
# file-local cfg_attr allow on the constructor. THIS gate is the fence:
# the next core-side or attn-side feature-graph edit that re-trips the
# posture reds here instead of silently.
#
# Row format: `pkg|features` (the test_gate.sh convention; `-` = default
# features). Rows must stay NARROW single-posture clippy arms — wide
# postures are full_gate Layer 3's job, not this fence's.
#
# --canary: proves the -D-warnings floor is LIVE — it compiles a scratch
# crate (outside the repo, standalone, zero deps) whose only content is an
# unused function and asserts clippy FAILS on it. Without that arm a
# future clippy/toolchain change that stops warning dead_code, or a
# silently-inert `-D`, would turn every row below into the vacuous green
# this fence exists to refuse.
#
# Exit: 0 = canary live + every posture row clippy-clean ·
#       1 = a posture row emitted a finding ·
#       2 = environment (no cargo, or the canary floor is inert).
#
# No errexit / no EXIT trap by design: every failure path below sets the
# exit code explicitly, so there is no abort window to launder
# (the trap_exit_launder class needs `set -e` + an EXIT trap to bite).

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 2
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd) || exit 2

command -v cargo >/dev/null 2>&1 || { echo "✗ cargo not on PATH"; exit 2; }

ROWS='
katgpt-attn|dash_attn
katgpt-attn|zha_chunk_sidecar
'

canary_dir=""
cleanup() {
    [ -n "$canary_dir" ] && [ -d "$canary_dir" ] && rm -rf "$canary_dir"
}

canary() {
    canary_dir="${TMPDIR:-/tmp}/i1219_canary_$$"
    (umask 077 && mkdir -p "$canary_dir/src") || { echo "✗ canary: cannot create $canary_dir"; exit 2; }
    {
        printf '[package]\nname = "i1219_canary"\nversion = "0.0.0"\nedition = "2021"\n'
    } > "$canary_dir/Cargo.toml"
    printf 'fn unused_canary_never_called() {}\n' > "$canary_dir/src/lib.rs"
    echo "▸ canary: planted unused fn in a scratch crate ($canary_dir)"
    if (cd "$canary_dir" && CARGO_TARGET_DIR="$canary_dir/target" cargo clippy --quiet -- -D warnings) >/dev/null 2>&1; then
        echo "✗ canary: clippy -D warnings did NOT fail on the planted unused fn — the floor is INERT; every posture row would be a vacuous green"
        cleanup
        exit 2
    fi
    echo "  ✓ canary: -D warnings fires (floor live)"
    cleanup
}

rc=0
for row in $ROWS; do
    pkg=${row%%|*}
    feats=${row#*|}
    echo "▸ posture: $pkg features=$feats"
    if [ "$feats" = "-" ]; then
        out=$(cd "$ROOT" && cargo clippy -p "$pkg" --lib -- -D warnings 2>&1); crc=$?
    else
        out=$(cd "$ROOT" && cargo clippy -p "$pkg" --lib --features "$feats" -- -D warnings 2>&1); crc=$?
    fi
    if [ "$crc" -ne 0 ]; then
        printf '%s\n' "$out" | grep -E '^(error|warning)' | head -10
        echo "✗ posture $pkg ($feats) clippy FAILED (exit $crc)"
        rc=1
    else
        echo "  ✓ clean"
    fi
done

if [ "$rc" -ne 0 ]; then
    exit 1
fi

canary_rc=0
canary || canary_rc=$?
exit "$canary_rc"
