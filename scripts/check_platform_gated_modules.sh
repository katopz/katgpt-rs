#!/usr/bin/env bash
# Typecheck the `#[cfg(not(target_os = "macos"))]` surface FROM macOS, and
# PROVE it was compiled without editing a single tracked file.
#
# ── The blind spot this closes ───────────────────────────────────────────────
#
# AGENTS.md ("And the inverse holds, with nothing to enforce it"): the full
# gate is run on the M3 by policy, and running ON macOS silently drops every
# `not(target_os = "macos")` module, `--all-features` included — so that
# command is STRUCTURALLY INCAPABLE of compiling them. Measured there: 9
# modules / 25,212 lines in riir-gpu alone, headed by `qwen38_dense_cudarc`
# at 8,599. Not theoretical: riir-ai `6bf51b592` landed a CUDA-only lib that
# did not compile (E0599) and it stood 7h45m, because nothing else builds
# that code. Open axis: riir-ai `.issues/857`.
#
# ── Why this works without a cross toolchain ─────────────────────────────────
#
# `cargo check` never links. It needs exactly two things: rust-std for the
# target, and build scripts that exit 0. Three build scripts stand in the way
# and none needs a real linux toolchain:
#
#   blake3            NEON C     -> CARGO_FEATURE_NO_NEON=1 selects the
#                                   pure-Rust path, no C at all.
#   libsqlite3-sys    sqlite3.c  \
#   sentencepiece-sys cmake/C++  /  the ANDROID NDK's clang, already
#                                   installed here, which ships a COMPLETE
#                                   linux sysroot. The macOS SDK cannot:
#                                   `sys/cdefs.h` answers a linux-gnu target
#                                   with "#error Unsupported architecture".
#
# Three details, each of which cost real time to find:
#
#   1. `--target=` goes AFTER "$@". cc-rs injects its own
#      `--target=aarch64-unknown-linux-gnu` and clang honours the LAST one,
#      so a shim putting its own target first is silently overridden and you
#      are back to a missing sysroot.
#   2. `-llog` is required: sentencepiece's cmake build LINKS helper binaries
#      which against bionic need `__android_log_write`.
#   3. The shim directory MUST be stable across runs. cmake bakes
#      CMAKE_C_COMPILER into a CMakeCache.txt inside the cargo build dir, so
#      a `mktemp -d` path forces a reconfigure — and a reconfigure re-runs
#      cmake's C compiler test with HOST flags off that stale cache
#      (`-arch arm64 -isysroot .../MacOSX.sdk` on a linux link). It fails as
#      "The C compiler is not able to compile a simple test program", which
#      points at the shim instead of at the churn.
#
# The objects the shim emits are aarch64-linux-android, not -gnu. That is
# deliberate and fine: a check run consumes none of them. Do NOT reuse this
# shim for a real build or a test run.
#
# ── How the proof works, and why it does not plant code ──────────────────────
#
# A cfg-gated green is exactly what this repo does not trust, so the run has
# to show that the gated modules were really compiled. The first version of
# this script proved it by appending a deliberately-uncompilable function to
# a source file and requiring E0425. That was wrong, and it failed in the way
# that matters: a run killed mid-build left
#   fn __platform_gate_canary() { __this_function_does_not_exist(); }
# sitting dirty in riir-train, where a sibling agent's `git add` would have
# taken it. A trap cannot fix this — bash defers traps until the running
# child returns, and SIGKILL runs no trap at all.
#
# So the proof is READ-ONLY: rustc's dep-info (`target/**/deps/<crate>-*.d`)
# lists every source file it actually opened, and a module whose `mod` decl
# is cfg'd out is never opened. Verified against the census's own macOS
# `--all-features` build of riir-train-gpu, which is the ideal negative
# control:
#
#   numeric_drift_tap                  0 occurrences
#   ternary_deltanet_backward_cudarc   0 occurrences
#   src/optimizer.rs                   4 occurrences   <- ungated, so the
#                                                         grep itself works
#
# and against this script's own linux-target run, where the first two are
# present. Zero mutation, correct under SIGKILL, and it yields a COUNT
# rather than a yes/no — so `--min-gated` can floor it and a regression to
# "nothing gated was compiled" reds instead of passing.
#
# ── What a green run claims ──────────────────────────────────────────────────
#
# That the named package's `not(target_os = "macos")` code TYPECHECKS. Not
# that any of it runs — much of that CUDA code has never executed anywhere.
# It is the compile half of the platform axis, which is the half that was
# missing entirely.
#
# Usage:
#   scripts/check_platform_gated_modules.sh [--min-gated N] <repo> <pkg> [features...]
#
# Example (the run that verified riir-train `53538538`):
#   scripts/check_platform_gated_modules.sh ../riir-train riir-train-gpu numeric_drift_cuda

set -uo pipefail

TARGET=aarch64-unknown-linux-gnu
# The NDK's clang targets android, not gnu; see the header. api24 is the
# floor carrying the pthread/log symbols sentencepiece's cmake probe wants.
NDK_TRIPLE=aarch64-linux-android24

die() { echo "REFUSE: $*" >&2; exit 2; }

[ "$(uname -s)" = "Darwin" ] || die "this script exists to compile the
    not(target_os=\"macos\") surface FROM macOS. Off macOS that surface is
    already in your ordinary build and this shim would only hide it."

min_gated=1
if [ "${1:-}" = "--min-gated" ]; then
    min_gated=${2:-1}; shift 2
fi
repo=${1:-}; pkg=${2:-}; shift 2 2>/dev/null || true
[ -n "$repo" ] && [ -n "$pkg" ] || die "usage: $0 [--min-gated N] <repo> <pkg> [features...]"
[ -d "$repo" ] || die "no such repo: $repo"

feats=""
for f in "$@"; do feats="${feats:+$feats,}$f"; done

# ── rust-std for the target ─────────────────────────────────────────────────
if ! rustup target list --installed | grep -qx "$TARGET"; then
    echo "installing rust-std for $TARGET (one-time, ~100 MB)"
    rustup target add "$TARGET" || die "rustup target add $TARGET failed"
fi

# ── the NDK cross-cc (stable path — see header detail 3) ────────────────────
ndkbin=$(ls -d "$HOME"/Library/Android/sdk/ndk/*/toolchains/llvm/prebuilt/*/bin 2>/dev/null | head -1)
[ -n "$ndkbin" ] && [ -x "$ndkbin/clang" ] || die "no Android NDK clang under
    ~/Library/Android/sdk/ndk/*/toolchains/llvm/prebuilt/*/bin. Install the
    NDK (Android Studio SDK Manager), or supply any C/C++ compiler with a
    linux sysroot via CC_${TARGET//-/_} / CXX_${TARGET//-/_}."

shimdir="${TMPDIR:-/tmp}/platform_gated_shim"
mkdir -p "$shimdir" || die "cannot create $shimdir"
for pair in "clang:cc" "clang++:cxx"; do
    bin=${pair%%:*}; name=${pair##*:}
    cat > "$shimdir/$name" <<EOF
#!/bin/sh
exec "$ndkbin/$bin" -w -Qunused-arguments "\$@" --target=$NDK_TRIPLE -llog
EOF
    chmod +x "$shimdir/$name"
done

env_prefix=$(echo "$TARGET" | tr - _)
export CC_${env_prefix}="$shimdir/cc"
export CXX_${env_prefix}="$shimdir/cxx"
export AR_${env_prefix}="$ndkbin/llvm-ar"
export CARGO_FEATURE_NO_NEON=1   # blake3: pure-Rust path, no NEON C
export CARGO_FEATURE_PURE=1

# A dedicated target dir. Sharing `target/` with a native build thrashes the
# fingerprint cache and is the concurrent-cargo false-RED shape AGENTS.md
# documents.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/platform_gated_check_$(basename "$repo")}"

args=(check --target "$TARGET" -p "$pkg" --lib --tests --keep-going)
[ -n "$feats" ] && args+=(--features "$feats")

# Stream AND capture. A first cut only captured, and a cold check is minutes
# of silence — so a killed run left a zero-byte log, indistinguishable from
# the script never starting.
logfile="${PLATFORM_GATED_LOG:-/tmp/platform_gated_check.log}"
echo "log: $logfile   target dir: $CARGO_TARGET_DIR"
echo "running: cargo ${args[*]}   (in $repo)"
( cd "$repo" && cargo "${args[@]}" 2>&1 ) | tee "$logfile"
rc=${PIPESTATUS[0]}

grep -E "^error" "$logfile" | head -20
if [ "$rc" != 0 ]; then
    echo "platform_gated_check: FAIL — $pkg does not typecheck for $TARGET (rc=$rc)"
    exit 1
fi

# ── the read-only proof ─────────────────────────────────────────────────────
crate_us=$(echo "$pkg" | tr - _)
depfiles=$(find "$CARGO_TARGET_DIR/$TARGET" -name "${crate_us}-*.d" 2>/dev/null)
[ -n "$depfiles" ] || die "no dep-info (${crate_us}-*.d) under
    $CARGO_TARGET_DIR/$TARGET — cannot prove anything was compiled. This is
    an instrument failure, not a verdict."

# Every module file the crate gates on not(target_os = "macos"). Derived from
# source, never typed by hand: a hand-typed list goes stale silently and then
# the floor below is met by modules that no longer exist.
# `-A1`, the line AFTER the cfg — the module declaration follows its
# attribute. A first cut wrote `-B1` and found exactly ONE module in a crate
# that has two, because adjacent `#[cfg]`/`pub mod` pairs mean the line
# BEFORE a cfg is the previous module's declaration. It was off by one and
# still non-zero, which is the shape that passes review.
gated=$(awk '
    /not\(target_os = "macos"\)/ { want = 1; next }
    want && match($0, /mod [a-z0-9_]+ *;/) {
        decl = substr($0, RSTART + 4, RLENGTH - 4)
        gsub(/[ ;]/, "", decl)
        print decl
        want = 0
        next
    }
    { want = 0 }
' "$repo/crates/$pkg/src"/*.rs 2>/dev/null | sort -u)
[ -n "$gated" ] || die "found no not(target_os = \"macos\") module declarations
    in $pkg — either this crate has none (nothing to prove; use the ordinary
    gate) or the parse broke. Instrument failure, not a verdict."

# Positive control: an UNGATED file must be visible to the same grep, or a
# zero below would mean "the grep is broken", not "nothing was compiled".
control=lib.rs
grep -qh "src/$control" $depfiles || die "dep-info does not even name
    src/$control — the grep is blind, so no count from it can be trusted."

compiled=0; missing=""
for m in $gated; do
    if grep -qh "$m\.rs" $depfiles; then
        compiled=$((compiled + 1))
    else
        missing="${missing:+$missing }$m"
    fi
done

n_gated=$(echo "$gated" | wc -w | tr -d ' ')
echo "gated modules declared: $n_gated   COMPILED under $TARGET: $compiled"
[ -n "$missing" ] && echo "  not compiled (feature not selected, most likely): $missing"

if [ "$compiled" -lt "$min_gated" ]; then
    echo "platform_gated_check: FAIL — only $compiled gated module(s) compiled,"
    echo "floor is $min_gated. A green typecheck over ZERO gated modules is the"
    echo "exact blindness this script exists to detect."
    exit 1
fi

echo "platform_gated_check: PASS — $pkg${feats:+ (features: $feats)} typechecks"
echo "for $TARGET, with $compiled not(target_os = \"macos\") module(s) PROVEN"
echo "compiled via rustc dep-info. This is the COMPILE half only; nothing ran."
