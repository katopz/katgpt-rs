#!/usr/bin/env bash
# Full gate — the whole-surface compile + lint claim, as an assertion.
#
# Every build command in AGENTS.md's "Build Commands" block is narrow on at
# least one of three INDEPENDENT axes, and a green result says nothing about
# what the invocation compiled to nothing:
#
#   1. `check` vs `clippy`      — two `cargo heal` escape classes are rejected
#                                 by clippy's typeck and accepted by `check`
#                                 (E0689 ambiguous-integer, E0631 deref
#                                 coercion in `redundant_closure`).
#   2. default vs --all-features — non-default gated code compiles to NOTHING.
#   3. `-p <crate>` vs --workspace — at the SAME default features: a crate's own
#                                 non-default feature can be switched on by the
#                                 ROOT crate's defaults once the root is in the
#                                 selected set. `cargo test -p katgpt-backend
#                                 --lib` compiled clean while the workspace run
#                                 failed, because `gpu.rs` sits behind
#                                 `katgpt-backend/gpu_inference` and the chain
#                                 katgpt-rs/default -> async_qdq_overlap ->
#                                 inference_router -> gpu_inference only fires
#                                 when the root crate is selected.
#
# Missing `--all-targets` is a fourth: it skips every test / bench / example,
# which is where gated code lives.
#
# Consequence, measured: this gate was RED on `develop` from at least 2cb97410
# until 3e58e821 — five broken targets — while every documented gate was green.
# Two of the five were `cargo heal` escapes that survived the healer's own
# compile gate, because that gate ran where the code was absent.
#
# Layers, all hard-fail:
#   1. cargo present
#   2. platform coverage   → `target_os = "macos"` code (katgpt-backend's
#                            gpu.rs / ane.rs) is invisible off macOS EVEN WITH
#                            --all-features, because the cfg is an `all(...)`
#                            over target_os AND feature. A Linux run of this
#                            gate reproduces the exact vacuous green it exists
#                            to catch, so off-macOS is a partial gate and says
#                            so loudly.
#   2b. wasm32 coverage   → a SECOND platform axis. `wasm32-unknown-unknown`
#                            is compiled by nothing else in this repo, and its
#                            hot kernels need `+simd128` on top (the target
#                            defaults to OFF), so both arms run. Derived
#                            package list; the sites no `-p … --lib` reaches
#                            are pinned by membership.
#   3. the gate itself     → clippy, workspace, all targets, all features
#   4. zero errors         → any `error` line or unbuildable target is a finding
#   6. profile axis        → the same tree with debug_assertions OFF; the four
#                            axes below all run in the DEV profile (.docs/10_audits/debug_release_profile_axis.md)
#   5. doc/script parity   → AGENTS.md must quote the same command this script
#                            runs; a gate whose spec has drifted from its
#                            implementation is a gate nobody is running
#
# Usage:
#   scripts/full_gate.sh                          # strict
#   scripts/full_gate.sh --allow-partial-platform # off-macOS: warn, don't fail
#
# Honour CARGO_TARGET_DIR to avoid fighting a concurrent build:
#   CARGO_TARGET_DIR=/tmp/full_gate scripts/full_gate.sh
#
# Honour FULL_GATE_LOG to put the clippy log somewhere retrievable (CI artifact
# upload). Retained on failure, removed on a pass, either way.
#   FULL_GATE_LOG=/tmp/full_gate.log scripts/full_gate.sh
set -euo pipefail

ALLOW_PARTIAL=0
[ "${1:-}" = "--allow-partial-platform" ] && ALLOW_PARTIAL=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The gate command, defined once. Layer 5 asserts AGENTS.md quotes this string,
# so the doc and the assertion cannot drift apart silently.
#
# The deny list (Issue 701 R3b, 2026-09-03): the mechanical lints whose
# all-features warning surface was healed to ZERO residual (67 -> 13 distinct
# findings; the 13 survivors are judgement-class: unused_variables, dead_code,
# non_snake_case, too_many_arguments, assertions_on_constants) are now denied
# so a regression reds the gate instead of silently re-growing the ungated
# warning surface. Before -> after per lint: needless_range_loop 34 -> 0,
# map_clone 2 -> 0, iter_cloned_collect 0 -> 0 (the map_clone successor —
# healing map_clone revealed it; denied so the family cannot slide back),
# identity_op 1 -> 0, unused_parens 0 -> 0, bool_comparison 2 -> 0,
# manual_is_multiple_of 1 -> 0, collapsible_if 2 -> 0, map_all_any_identity
# 1 -> 0, unnecessary_cast 2 -> 0, manual_repeat_n 2 -> 0, unused_mut 2 -> 0,
# question_mark 1 -> 0, empty_line_after_outer_attr 1 -> 0,
# unusual_byte_groupings 3 -> 0. A lint with residual > 0 must NOT be added.
GATE_ARGS=(cargo clippy --workspace --all-targets --all-features --keep-going
    -- -D clippy::needless_range_loop
    -D clippy::map_clone
    -D clippy::iter_cloned_collect
    -D clippy::identity_op
    -D clippy::bool_comparison
    -D clippy::manual_is_multiple_of
    -D clippy::collapsible_if
    -D clippy::map_all_any_identity
    -D clippy::unnecessary_cast
    -D clippy::manual_repeat_n
    -D clippy::question_mark
    -D clippy::empty_line_after_outer_attr
    -D clippy::unusual_byte_groupings
    -D unused_mut
    -D unused_parens)
GATE_CMD="${GATE_ARGS[*]}"

# ── Layer 1: cargo present ──────────────────────────────────────────────────
if ! command -v cargo >/dev/null 2>&1; then
    echo "✗ cargo not installed — full gate cannot run"
    exit 1
fi

# ── Layer 2: platform coverage ──────────────────────────────────────────────
# Grep the real cfg rather than trusting this comment to stay true.
# `|| true` is load-bearing: grep exits 1 on no match, and under `set -e` with
# pipefail a failed substitution pipeline would kill the script here with no
# diagnostic at all.
APPLE_GATED=$({ grep -rl 'target_os = "macos"' --include='*.rs' crates src 2>/dev/null || true; } | wc -l | tr -d ' ')

# Verify the instrument before trusting its verdict. A zero here reads exactly
# like a working grep over a surface that no longer exists, and BOTH readings
# invalidate something: zero means either this grep drifted (paths moved) or the
# device-backend surface is gone — in which case the workflow is paying for a
# macOS runner, and billing macOS minutes at a multiple of Linux, for nothing.
# Either way a human must decide, so fail rather than narrate a vacuous pass.
if [ "$APPLE_GATED" -eq 0 ]; then
    echo "✗ platform layer found NO target_os = \"macos\" files under crates/ src/"
    echo "  Either the grep drifted from the tree, or the device-backend surface is"
    echo "  gone. If the latter, .github/workflows/full_gate.yml should stop paying"
    echo "  for macos-latest — revisit its 'Why macos-latest' preamble."
    exit 1
fi

if [ "$(uname -s)" != "Darwin" ]; then
    echo "⚠ not macOS — $APPLE_GATED file(s) gated on target_os = \"macos\" will NOT compile,"
    echo "  even with --all-features (the cfg is all(target_os, feature))."
    echo "  This run is a PARTIAL gate: it cannot see the device-backend surface."
    if [ "$ALLOW_PARTIAL" -eq 0 ]; then
        echo "✗ refusing to report a partial run as a pass (--allow-partial-platform to override)"
        exit 1
    fi
else
    echo "✓ macOS — the $APPLE_GATED target_os-gated file(s) are in scope"
fi

# ── Layer 2b: wasm32 + simd128 coverage (Issue 737) ─────────────────────────
# Layer 2 is about ONE platform axis (`target_os = "macos"`). `wasm32` is a
# second, and until 2026-09-07 nothing in this repo compiled it: the only
# script that ever built for `wasm32-unknown-unknown` was
# `scripts/build-moka-wasm.sh`, which a human runs before a deploy.
#
# It is doubly invisible, which is why it rotted. The hot kernels are gated on
# `all(target_arch = "wasm32", target_feature = "simd128")`, and
# `wasm32-unknown-unknown` defaults to NO simd128 — so a plain
# `--target wasm32-unknown-unknown` run compiles the SCALAR fallback and the
# SIMD half to nothing. Both arms are needed; neither implies the other.
#
# Measured on the run that produced this layer: 14 findings in
# `katgpt-moka-wasm` alone, of which ELEVEN were `unsafe_op_in_unsafe_fn`
# (edition-2024 `warning[E0133]`, on its way to a hard error) — in a crate
# whose whole reason to exist is to be shipped to a browser.
WASM_FILES=$(git grep -l 'target_arch = "wasm32"' -- '*.rs' 2>/dev/null || true)
WASM_SIMD_FILES=$(git grep -l 'target_feature = "simd128"' -- '*.rs' 2>/dev/null || true)
WASM_N=$(printf '%s\n' "$WASM_FILES" | grep -c . || true)
WASM_SIMD_N=$(printf '%s\n' "$WASM_SIMD_FILES" | grep -c . || true)

# Same instrument check as layer 2, same reasoning: a zero reads exactly like
# a working grep over a surface that no longer exists, and both readings need
# a human. If the wasm32 surface really is gone, delete this layer and
# `scripts/build-moka-wasm.sh` together.
if [ "$WASM_N" -eq 0 ] || [ "$WASM_SIMD_N" -eq 0 ]; then
    echo "✗ wasm32 layer found NO wasm32 ($WASM_N) or NO simd128 ($WASM_SIMD_N) files"
    echo "  Either the grep drifted from the tree, or the browser surface is gone."
    exit 1
fi

# The lane runs `-p <crate>` per crate that has a wasm32 cfg in its `src/`,
# plus the ROOT package when the root `src/` has one. Deriving it beats typing
# it — a new wasm32-bearing crate joins the lane by existing. Selecting the
# root package is what makes this lane wide: clippy lints every WORKSPACE PATH
# DEPENDENCY it pulls in (registry crates are `--cap-lints`'d, workspace ones
# are not), which is how the orphaned doc block on
# `katgpt-attn-match::select_highest_attn_keys` surfaced — a crate with no
# wasm32 code of its own.
#
# What derivation CANNOT do is notice a wasm32 site that no `-p … --lib`
# reaches, so the residue is pinned by MEMBERSHIP below: a set is gateable
# where its cardinality is not, and a count that matches is not a checksum
# over a set.
ROOT_PKG=$(awk '/^\[package\]/{f=1;next} /^\[/{f=0} f && /^name[ ]*=/{gsub(/^name[ ]*=[ ]*"|"[ ]*$/,""); print; exit}' Cargo.toml)
WASM_PKGS=$(printf '%s\n' "$WASM_FILES" | sed -n 's|^crates/\([^/]*\)/src/.*|\1|p' | sort -u)
if printf '%s\n' "$WASM_FILES" | grep -q '^src/'; then
    WASM_PKGS=$(printf '%s\n%s\n' "$WASM_PKGS" "$ROOT_PKG" | sort -u)
fi

# Sites no `--lib` lane reaches. Two of the four ARE built, as named targets
# (WASM_EXTRA_TARGETS); the other two are NOT, for a measured reason:
#
#   examples/bomber_*.rs — `requires the features: bomber`, and with that
#   feature on they fail at `cargo check` with `unresolved import
#   sys::position` / `cannot find function enable_raw_mode in module sys`:
#   crossterm has no wasm32 backend. A TUI arena cannot target a browser;
#   this is a fact about the dependency, not a gap to close.
WASM_RESIDUE=$(printf '%s\n' "$WASM_FILES" | grep -v '^crates/[^/]*/src/' | grep -v '^src/' | sort)
WASM_RESIDUE_EXPECTED='crates/katgpt-core/benches/bench_432_simd_lut_dequant_goat.rs
crates/katgpt-core/examples/simd_wasm32_goat.rs
examples/bomber_21_sonlt_arena.rs
examples/bomber_tjs_arena.rs'
if [ "$WASM_RESIDUE" != "$WASM_RESIDUE_EXPECTED" ]; then
    echo "✗ the set of wasm32 sites NOT covered by a --lib lane has changed."
    echo "  Either add the new one to WASM_EXTRA_TARGETS (if it compiles for"
    echo "  wasm32) or to WASM_RESIDUE_EXPECTED with the measured reason it"
    echo "  cannot. Do NOT just re-pin the list."
    echo "  --- expected ---"; printf '%s\n' "$WASM_RESIDUE_EXPECTED"
    echo "  --- measured ---"; printf '%s\n' "$WASM_RESIDUE"
    exit 1
fi
# pkg|selector|name — the wasm32 GOAT evidence lives in these two targets, so
# a lane that skipped them would gate everything EXCEPT the thing the docs
# quote (`.docs/06_game_arenas/go_arena.md`: 0.6 ms/move, 10.7x real Moka).
WASM_EXTRA_TARGETS='katgpt-core|--example|simd_wasm32_goat
katgpt-core|--bench|bench_432_simd_lut_dequant_goat'

if ! rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
    echo "⚠ wasm32-unknown-unknown not installed — $WASM_N wasm32 file(s) and"
    echo "  $WASM_SIMD_N simd128 file(s) will NOT compile in this run."
    echo "  This run is a PARTIAL gate (rustup target add wasm32-unknown-unknown)."
    if [ "$ALLOW_PARTIAL" -eq 0 ]; then
        echo "✗ refusing to report a partial run as a pass (--allow-partial-platform to override)"
        exit 1
    fi
else
    WASM_P_ARGS=$(printf -- '-p %s ' $WASM_PKGS)
    for arm in on off; do
        if [ "$arm" = on ]; then
            WASM_RUSTFLAGS='-C target-feature=+simd128'
        else
            WASM_RUSTFLAGS=''
        fi
        # `--keep-going` for the same reason layer 3 needs it: without it the
        # run stops at the first failing crate and under-reports the rest.
        # shellcheck disable=SC2086  # word splitting is the point: one -p per crate
        if ! RUSTFLAGS="$WASM_RUSTFLAGS" cargo clippy $WASM_P_ARGS --lib \
                --target wasm32-unknown-unknown --keep-going --quiet -- -D warnings; then
            echo "✗ wasm32 --lib lane failed (simd128 $arm)" >&2
            exit 1
        fi
        while IFS='|' read -r xpkg xsel xname; do
            [ -n "$xpkg" ] || continue
            if ! RUSTFLAGS="$WASM_RUSTFLAGS" cargo clippy -p "$xpkg" "$xsel" "$xname" \
                    --target wasm32-unknown-unknown --quiet -- -D warnings; then
                echo "✗ wasm32 target lane failed (simd128 $arm): $xpkg $xsel $xname" >&2
                exit 1
            fi
        done <<EOF
$WASM_EXTRA_TARGETS
EOF
        echo "✓ wasm32 clean (simd128 $arm): $(printf '%s' "$WASM_PKGS" | tr '\n' ' ')+ 2 named targets"
    done
fi

# ── Layer 3: the gate ───────────────────────────────────────────────────────
# `--keep-going` is not optional: without it cargo stops at the first failing
# target. The run that found the five breaks reported only two without it.
# $FULL_GATE_LOG overrides the location so a CI job can upload the log as an
# artifact. Retention alone is not enough there: a GitHub runner is destroyed
# when the job ends, so a path printed into a dead runner's filesystem is
# unreachable — the weekly run would be left with only the summary, which is
# the situation the retention exists to fix.
LOG="${FULL_GATE_LOG:-$(mktemp -t full_gate)}"
mkdir -p "$(dirname "$LOG")"

# Retain the log when the gate fails. The summary below prints error CLASSES and
# the first diagnostic; everything else — every remaining site, every warning —
# lives only in this file, and re-deriving it costs another >13 min run.
#
# "On a pass there is nothing in it worth the disk" was wrong, and cost a run.
# A PASS still reports a warning-finding count (119 across 20 targets on
# 2026-09-01), and the per-lint / per-crate breakdown behind that number exists
# ONLY here — it is the input to Issue 701 R3b. Worse, re-deriving it is not
# simply expensive: a second run against a now-warm target dir emits almost
# nothing, because cargo does not replay diagnostics for crates it considers
# fresh. The log is gone until something invalidates the cache.
#
# So: if the caller NAMED a path via $FULL_GATE_LOG, they asked for the file —
# honour that on pass as well as on fail. An unnamed run still gets a temp file
# cleaned up on success.
# `if`, not `[ … ] && KEEP_LOG=1`: this script runs under `set -e`, where a
# trailing AND-list whose test fails takes the whole run down.
KEEP_LOG=0
if [ -n "${FULL_GATE_LOG:-}" ]; then
    KEEP_LOG=1
fi
# ── The completion sentinel (Issue 734) ─────────────────────────────────────
# Every `exit 1` below is an explicit verdict and passes through untouched.
# What this arm catches is the OTHER death: bash aborting mid-script while
# reporting success. Measured on macOS `/bin/bash` 3.2.57, and ONLY there
# (Issue 735): bash 4.4 through 5.3, dash and busybox ash all PRESERVE the
# status. On 3.2, when `set -u` hits an unbound expansion or `eval` hits a
# syntax error, the shell enters the EXIT trap with `$?` **already 0**, so an
# EXIT trap whose last command succeeds makes the abort exit **0**. The usual
# `trap 'rc=$?; …; exit $rc'` idiom does not help: the rc it saves is itself 0.
# Only "did the script reach its own last line?" catches it — and that catches
# every other premature death too (a `set -e` trip in an unguarded spot, a
# SIGTERM, a future editing slip), on EVERY shell, which is why this stays even
# where 3.2 is out of the picture.
# ⛔ Not academic HERE, unlike most of the repaired scripts: `full_gate.yml`
# runs this on **macos-latest**, so whether the sentinel is load-bearing in CI
# depends on what `#!/usr/bin/env bash` resolves to on that image — measured by
# that workflow's own preamble step (Issue 735 T3), because no workstation can
# answer it. Every other gate workflow in this repo is ubuntu-latest (bash 5),
# where an abort exits non-zero and reds the job on its own.
# This is not hypothetical: seal-remake's ci_feature_guard.sh could not fail
# past its layer 13 for months for exactly this reason — on a developer's Mac.
# Population-wide picture: scripts/trap_exit_launder_audit.py; the premise
# across 11 interpreters: scripts/trap_launder_premise_matrix.py.
FULL_GATE_COMPLETED=0
full_gate_cleanup() {
    gate_st=$?
    if [ "$KEEP_LOG" -eq 1 ]; then
        echo "  full log retained: $LOG"
    else
        rm -f "$LOG"
    fi
    if [ "$FULL_GATE_COMPLETED" != "1" ] && [ "$gate_st" = "0" ]; then
        echo "✗ full gate ABORTED mid-run while reporting success — it did not reach" >&2
        echo "  its own last line, so it verified NOTHING past the error above." >&2
        echo "  A premature death must never read as a pass; forcing exit 1." >&2
        exit 1
    fi
    exit "$gate_st"
}
trap full_gate_cleanup EXIT
echo "▸ $GATE_CMD"
set +e
"${GATE_ARGS[@]}" >"$LOG" 2>&1
set -e

# ── Strip ANSI before ANY counting ──────────────────────────────────────────
# Every count below is `^`-anchored, and cargo emits colour when
# CARGO_TERM_COLOR=always — which .github/workflows/full_gate.yml sets. A
# coloured line begins with an escape sequence, not with `warning`/`error`/
# whitespace, so EVERY counter matched zero and the gate reported
#   ✓ full gate PASSED — 0 errors ... (0 warning finding(s) across 0 target(s))
# over a log holding 32 compiled units and 297 warning findings (measured from
# run 33530563741's uploaded artifact).
#
# The error counter was defeated the same way, which is the serious half: the
# gate could not have FAILED in CI. A completely broken workspace would have
# printed the same green. Locally it worked only by accident — cargo suppresses
# colour when stdout is not a TTY, and here it is redirected to $LOG.
#
# Stripped IN PLACE rather than into a second file: the artifact humans
# download is then plain text, which is what you want when reading it in a
# browser, and there is no chance of a later counter being pointed at the
# unstripped copy.
#
# Portable CSI strip (BSD sed on macOS, GNU sed on Linux). LC_ALL=C so the
# byte-oriented match cannot be reinterpreted under a UTF-8 locale.
if [ -s "$LOG" ]; then
    LC_ALL=C sed $'s/\033\[[0-9;]*[a-zA-Z]//g' "$LOG" >"$LOG.plain" \
        && mv "$LOG.plain" "$LOG"
fi

# ── Layer 4: zero errors ────────────────────────────────────────────────────
# Count `error` lines and unbuildable targets separately: a target can fail to
# build with its diagnostics attributed to a dependency, and an error can be a
# deny-level lint that names no target.
# Every count is `|| true`-guarded for the same reason as Layer 2. The
# diagnostic count deliberately EXCLUDES cargo's own "error: could not compile"
# aggregates, which would otherwise inflate it by one per broken target and
# make the two numbers look like independent evidence when they are not.
DIAGS=$({ grep -E '^error(\[|:)' "$LOG" || true; } | { grep -v 'could not compile' || true; } | wc -l | tr -d ' ')
BROKEN=$({ grep -E 'could not compile' "$LOG" || true; } | sort -u)
BROKEN_N=$({ printf '%s' "$BROKEN" | grep -c . || true; })
# Count warning FINDINGS, not warning lines. Cargo emits a per-target
# "`crate` (lib test) generated N warnings" tally that also starts with
# `warning:`, so a raw line count silently adds one per target — and lands
# misleadingly close to the real emitted total. Measured on the first green run:
# 138 lines = 118 findings + 20 tallies, while the tallies themselves sum to 141
# (118 + 23 duplicates, the same finding compiled in both `lib` and `lib test`).
# Three different quantities within 23 of each other; report the one a reader
# can act on.
WARN_LINES=$({ grep -cE '^warning' "$LOG" || true; })
WARN_TALLIES=$({ grep -cE '^warning: .* generated [0-9]+ warning' "$LOG" || true; })
WARNINGS=$((WARN_LINES - WARN_TALLIES))

# ── Layer 3b: liveness — did this run examine anything at all? ───────────────
# The gate reported "✓ full gate PASSED — 0 errors, 0 unbuildable targets
# (0 warning finding(s) across 0 target(s))" on its first TWO CI runs, having
# compiled ZERO units. Same command, same repo, same day: the workstation
# reports 119 findings across 20 targets. A green built from nothing is exactly
# the vacuous pass this gate exists to catch, arrived at by the gate itself.
#
# Two independent signals, because either alone has a blind spot:
#   UNITS   — cargo actually built something ("Compiling"/"Checking" lines).
#   TALLIES — cargo REPLAYED cached diagnostics without rebuilding, which a
#             warm local re-run does (Issue 701 R3b measured 119/20 that way).
# A conclusive run has at least one. Zero of both means the run is telling you
# about its cache, not about the code.
#
# A genuinely warning-free workspace with a fully warm target dir would also
# land here — and "I cannot distinguish clean from unmeasured" is the honest
# thing to say about that state, not a pass. Invalidate the cache and re-run.
UNITS=$({ grep -cE '^[[:space:]]*(Compiling|Checking) ' "$LOG" || true; })

if [ "$UNITS" -eq 0 ] && [ "$WARN_TALLIES" -eq 0 ]; then
    KEEP_LOG=1
    echo "✗ full gate INCONCLUSIVE — the run compiled 0 units and replayed 0"
    echo "  diagnostics, so it verified NOTHING. This is not a pass."
    echo "  Two causes seen so far, and they need different fixes:"
    echo "    1. The log did not parse — e.g. colour codes ahead of every"
    echo "       ^-anchor. Check the log: if it HAS Checking/warning lines,"
    echo "       the census is broken, not the build. (This was the real cause"
    echo "       of the first three CI runs; the ANSI strip above fixes it.)"
    echo "    2. A restored build cache cargo considers fresh: no rebuild, and"
    echo "       no replayable diagnostics to fall back on. Run cold."
    echo "  log: $LOG"
    exit 1
fi

if [ "$DIAGS" -ne 0 ] || [ "$BROKEN_N" -ne 0 ]; then
    KEEP_LOG=1
    echo "✗ full gate FAILED — $DIAGS error diagnostic(s), $BROKEN_N unbuildable target(s)"
    # `|| true`: bash exempts the left side of `&&` from `set -e`, but being
    # explicit here beats relying on that exemption in a failure path that must
    # print its diagnostics before exiting.
    { [ -n "$BROKEN" ] && { echo "  unbuildable:"; printf '%s\n' "$BROKEN" | sed 's/^/    /'; }; } || true
    echo "  error classes:"
    grep -E '^error(\[|:)' "$LOG" | sort | uniq -c | sed 's/^/    /'
    echo "  first diagnostic with location:"
    grep -A3 -m1 -E '^error(\[|:)' "$LOG" | sed 's/^/    /'
    exit 1
fi

# ── Layer 5: doc/script parity ──────────────────────────────────────────────
# A gate documented with a different command than the one asserted here is how
# a spec silently stops describing the code.
if ! grep -qF "$GATE_CMD" AGENTS.md; then
    echo "✗ AGENTS.md does not quote the gate command — doc and gate have drifted"
    echo "  expected to find: $GATE_CMD"
    exit 1
fi

# ── Layer 6: the profile axis ───────────────────────────────────────────────
# Everything above runs in the DEV profile, so `debug_assertions` is always ON
# and every item behind `#[cfg(debug_assertions)]` — plus everything that
# DEPENDS on one — is compiled only in the configuration where it works. That
# was the gate's fourth blind spot, and it was not hypothetical: measured
# 2026-09-03, adding --release produced 2 errors and `cargo test --release -p
# katgpt-core --lib` did not compile AT ALL (.docs/10_audits/debug_release_profile_axis.md).
#
# `check`, deliberately, not `clippy`: this axis is about COMPILATION with
# debug_assertions off. The lint surface is already covered by Layer 3, and
# clippy's lints do not vary by profile except on the small `cfg(not(
# debug_assertions))` surface — a residual noted rather than paid for.
#
# NOT folded into GATE_ARGS: Layer 5 asserts AGENTS.md quotes that string
# verbatim, and this is a different question with a different command.
#
# Liveness, not just an error count (.issues/705): a run that compiles nothing
# reports zero errors, which reads exactly like a pass.
# --message-format=json, and that is the load-bearing choice here. The first
# cut counted "Compiling"/"Checking" lines like Layer 3 does, and reported
# INCONCLUSIVE on its very first in-situ run: the release tree was already warm,
# so cargo compiled 0 units and printed nothing. Freshness must not decide a
# liveness verdict. `compiler-artifact` records ARE emitted for fresh units â
# measured on the same warm tree, 3 "Checking" lines vs 1,423 artifacts.
#
# It also makes this layer immune to the ANSI-colour trap that zeroed every
# ^-anchored counter in this gate's first two CI runs (.issues/705): JSON keys
# carry no colour codes.
REL_ARGS=(cargo check --workspace --all-targets --all-features --keep-going --release
    --message-format=json)
echo "▸ Layer 6: profile axis — ${REL_ARGS[*]}"
REL_LOG="$(mktemp -t full_gate_release)"
"${REL_ARGS[@]}" > "$REL_LOG" 2>&1 || true
REL_UNITS=$({ grep -c '"reason":"compiler-artifact"' "$REL_LOG" || true; })
REL_ERRS=$({ grep -c '"level":"error"' "$REL_LOG" || true; })

if [ "$REL_UNITS" -eq 0 ]; then
    echo "✗ full gate INCONCLUSIVE — the release pass produced 0"
    echo "  compiler-artifact records, so it verified NOTHING about the"
    echo "  debug_assertions-off configuration. Unlike a zero unit count this is NOT"
    echo "  explained by a warm target dir: artifacts are reported for fresh units"
    echo "  too. Cargo did not run."
    echo "  log: $REL_LOG"
    exit 1
fi
if [ "$REL_ERRS" -ne 0 ]; then
    echo "✗ full gate FAILED — $REL_ERRS error diagnostic(s) in the RELEASE profile"
    echo "  (the dev-profile pass above was clean: this is the debug_assertions axis)"
    { grep -o '"rendered":"error[^"]*' "$REL_LOG" | sort -u | head -10 | sed 's/^/    /'; } || true
    echo "  log: $REL_LOG"
    exit 1
fi
rm -f "$REL_LOG"
echo "  ✓ release profile clean ($REL_UNITS compiler-artifact record(s))"

# UNITS is printed on every pass, not just when it is interesting: the number
# that would have exposed the vacuous CI green was never on screen.
echo "✓ full gate PASSED — 0 errors, 0 unbuildable targets ($WARNINGS warning finding(s) across $WARN_TALLIES target(s), not gated; $UNITS unit(s) compiled)"
FULL_GATE_COMPLETED=1  # the last line — see full_gate_cleanup above
