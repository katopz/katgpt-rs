#!/usr/bin/env bash
# KatgptProof gate — the Lean proofs, as an assertion rather than a printout.
#
# `.proofs/PrintAxioms.lean` only *prints*; a human had to read it for the
# axiom budget to mean anything. This script reads it instead.
#
# Five layers, all hard-fail:
#   1. lake build              → every theorem type-checks
#   2. sorry check             → `lake build` only WARNS on `sorry` and still
#                                exits 0, so a proof with a hole would ship
#                                green. Checked twice: in the build output
#                                (catches everything, but only on a build that
#                                actually recompiled) and in the source
#                                (cache-independent).
#   3. trust-escape scan       → `native_decide` swaps the kernel for the
#                                compiler via `Lean.ofReduceBool`. Layer 5
#                                cannot see it inside an `example`, because
#                                `example`s are anonymous. So scan the source.
#   4. axiom-declaration scan  → a bare `axiom` is a hole Layer 5 also cannot
#                                see: the inventory shows what theorems
#                                *depend on*, not what was *declared*.
#   5. axiom inventory         → every shipped theorem's axiom set is within
#                                {propext, Classical.choice, Quot.sound}, and
#                                the theorem count matches both the printer's
#                                directive count and the pin below.
#
# Usage:
#   scripts/proof_gate.sh                       # strict: missing Lean = failure
#   scripts/proof_gate.sh --allow-missing-lean  # skip cleanly if elan absent
#
# Install Lean: curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
set -euo pipefail

ALLOW_MISSING=0
[ "${1:-}" = "--allow-missing-lean" ] && ALLOW_MISSING=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROOFS_DIR="$REPO_ROOT/.proofs"

# Standard Lean foundation — the same one Mathlib stands on. Anything outside
# this set is a finding, not a note.
ALLOWED_AXIOMS=("propext" "Classical.choice" "Quot.sound")

# Independently-pinned theorem count — the audited surface.
# Deriving this from PrintAxioms.lean itself would be self-referential: delete
# a `#print axioms` line and both the expectation and the result shrink
# together, so the check could never fire. Pinning it here makes shrinking the
# audit a deliberate act.
# Adding theorems? Add the `#print axioms` line, then bump this number.
#   17 → Bridge ranking preservation (3), HOPE spec instances (2),
#        Ssmax core (3), Ssmax dilution bound (3), Ssmax asymptotics (6)
#   35 → Pencil (Issue 678): T1 isometry (3) + Courant–Fischer core (6)
#        + Weyl T2 (3) + Loewner/mirror T3 (2) + eigengap T4 core (4);
#        the auxiliary lemmas (mirror-pairing, combo expansion, SDA,
#        shift, PSD-ray, diagonal-ray) are covered by the audited heads.
#   38 → Pencil ladder closeout (Issue 678): diagonal eigenvalue pinning
#        (eigval_diagonal_antitone, the singles/CF substrate), the ladder
#        unit gap, and the T4 final assembly (eigengap_ladder_ge_half).
#   39 → HintRegret (Plan 576, 1b73fbf1): band-gate openness over ℝ
#        (bandGate_mem_Ioo) — the ideal (0,1) contract the f32 gate
#        approximates. Added with the pin bumped but the ladder not extended;
#        recorded here so the pin's justification matches the pin.
EXPECTED_THEOREMS=39

# Bare `axiom` declarations that are allowed to exist, by name. Empty here:
# KatgptProof declares none, and a new one must be justified in review rather
# than appear silently.
ALLOWED_AXIOM_DECLS=()

export PATH="$HOME/.elan/bin:$PATH"

if ! command -v lake >/dev/null 2>&1; then
    if [ "$ALLOW_MISSING" -eq 1 ]; then
        echo "⚠ lake (Lean 4) not installed — proof gate SKIPPED (--allow-missing-lean)"
        echo "  the theorems are unverified in this run; CI runs this gate strictly"
        exit 0
    fi
    echo "✗ lake (Lean 4) not installed — proof gate cannot run"
    echo "  install: curl https://elan.lean-lang.org/elan-init.sh -sSf | sh"
    echo "  or re-run with --allow-missing-lean to skip (local dev only)"
    exit 1
fi

cd "$PROOFS_DIR"

# ── Layer 1: type-check ────────────────────────────────────────────────
echo "── Layer 1: lake build ──"
build_out=""
if ! build_out="$(lake build 2>&1)"; then
    echo "$build_out"
    echo "✗ lake build failed — a theorem does not type-check"
    exit 1
fi
echo "$build_out" | grep -E "^Build completed|error" || true

# ── Layer 2: no holes ──────────────────────────────────────────────────
# `lake build` emits ``declaration uses `sorry` `` as a WARNING and still
# exits 0. Note Lean quotes with BACKTICKS — match the prefix only, so the
# check survives a quoting change in a future Lean release.
echo "── Layer 2: sorry check ──"
if echo "$build_out" | grep -q "declaration uses"; then
    echo "$build_out" | grep "declaration uses"
    echo "✗ a theorem is admitted with 'sorry' — the proof is a hole"
    exit 1
fi

# The build-output check above sees nothing when the build was fully cached,
# and `lake` caches aggressively. This second check reads the source, so it
# holds regardless of cache state. Backticked spans and `--` comments are
# stripped first: every one of these files discusses `sorry` in prose.
source_holes="$(
    find . -name '*.lean' -not -path '*/.lake/*' | while read -r f; do
        sed -e 's/`[^`]*`//g' -e 's/--.*$//' "$f" \
            | grep -nwE 'sorry|sorryAx|admit' | sed "s|^|${f}:|" || true
    done
)"
if [ -n "$source_holes" ]; then
    echo "$source_holes"
    echo "✗ 'sorry'/'admit' present in proof source — the proof is a hole"
    exit 1
fi
echo "✓ no 'sorry' admitted (build output + source scan)"

# ── Layer 3: no trust escapes ──────────────────────────────────────────
# `native_decide` discharges a goal by compiling it to native code and
# reflecting the result through the `Lean.ofReduceBool` axiom — the compiler
# and the FFI become part of the trusted base, not just the kernel. Layer 5
# would catch that for a named theorem, but not inside an `example`: examples
# are anonymous, so `#print axioms` cannot name them and the inventory stays
# silent. Hence a source scan.
echo "── Layer 3: trust-escape scan ──"
escapes="$(
    find . -name '*.lean' -not -path '*/.lake/*' | while read -r f; do
        sed -e 's/`[^`]*`//g' -e 's/--.*$//' "$f" \
            | grep -nwE 'native_decide' | sed "s|^|${f}:|" || true
    done
)"
if [ -n "$escapes" ]; then
    echo "$escapes"
    echo "✗ 'native_decide' present — this trades the kernel for the compiler"
    echo "  (adds Lean.ofReduceBool; invisible to the axiom inventory inside an"
    echo "  'example'). Prefer a kernel-reducible spelling of the literals."
    exit 1
fi
echo "✓ no 'native_decide' — every goal is closed by the kernel"

# ── Layer 4: no undeclared axioms ──────────────────────────────────────
# The inventory in Layer 5 reports what theorems *depend on*. A bare `axiom`
# that nothing depends on yet is invisible to it, and becomes load-bearing the
# moment someone uses it. Assert the declared set instead of inferring it.
echo "── Layer 4: axiom-declaration scan ──"
declared="$(
    find . -name '*.lean' -not -path '*/.lake/*' -print0 \
        | xargs -0 grep -hoE '^axiom +[A-Za-z_][A-Za-z0-9_'"'"'.]*' 2>/dev/null \
        | awk '{print $2}' | sort -u || true
)"
axiom_violations=0
if [ -n "$declared" ]; then
    while IFS= read -r name; do
        [ -z "$name" ] && continue
        ok=0
        for allowed in ${ALLOWED_AXIOM_DECLS[@]+"${ALLOWED_AXIOM_DECLS[@]}"}; do
            [ "$name" = "$allowed" ] && ok=1 && break
        done
        if [ "$ok" -eq 0 ]; then
            echo "✗ undeclared axiom '$name' — add it to ALLOWED_AXIOM_DECLS"
            echo "  in this script (with a justification) or remove it"
            axiom_violations=$((axiom_violations + 1))
        fi
    done <<< "$declared"
fi
if [ "$axiom_violations" -gt 0 ]; then
    echo "✗ $axiom_violations undeclared axiom(s)"
    exit 1
fi
echo "✓ no axiom declarations outside the allowlist"

# ── Layer 5: axiom budget ──────────────────────────────────────────────
echo "── Layer 5: axiom inventory ──"
axiom_out=""
if ! axiom_out="$(lake env lean PrintAxioms.lean 2>&1)"; then
    echo "$axiom_out"
    echo "✗ PrintAxioms.lean failed to run — a theorem name may have been renamed or dropped"
    exit 1
fi
# Unwrap: Lean pretty-prints `depends on axioms: [...]` at a fixed width, so a
# long enough theorem name pushes the axiom list onto continuation lines. That
# makes the budget parser below fail on line *length* rather than on content,
# AND silently under-audit: the truncated head record loses its tail axioms
# from the budget check entirely. (Fired in riir-neuron-db, its Issue 605;
# ported here 2026-09-11.) An inventory record always starts with a quote;
# anything else is a continuation.
axiom_out="$(printf '%s\n' "$axiom_out" | awk '
    /^'"'"'/ { if (NR > 1 && buf != "") print buf; buf = $0; next }
    { sub(/^[ \t]+/, ""); buf = buf " " $0 }
    END { if (buf != "") print buf }
')"
if echo "$axiom_out" | grep -qi "error"; then
    echo "$axiom_out" | grep -i "error"
    echo "✗ PrintAxioms.lean reported an error"
    exit 1
fi

# Every `#print axioms` directive must actually produce an inventory line.
directives="$(grep -c '^#print axioms' PrintAxioms.lean)"

count=0
axiom_free=0
violations=0

while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in
        *"does not depend on any axioms")
            count=$((count + 1))
            axiom_free=$((axiom_free + 1))
            ;;
        *"depends on axioms: ["*)
            count=$((count + 1))
            axioms="${line#*depends on axioms: [}"
            axioms="${axioms%]}"
            IFS=',' read -ra parts <<< "$axioms"
            for a in "${parts[@]}"; do
                a="${a// /}"
                [ -z "$a" ] && continue
                if [ "$a" = "sorryAx" ]; then
                    echo "✗ SORRY AXIOM in: $line"
                    violations=$((violations + 1))
                    continue
                fi
                ok=0
                for allowed in "${ALLOWED_AXIOMS[@]}"; do
                    [ "$a" = "$allowed" ] && ok=1 && break
                done
                if [ "$ok" -eq 0 ]; then
                    echo "✗ out-of-budget axiom '$a' in: $line"
                    violations=$((violations + 1))
                fi
            done
            ;;
        *)
            echo "✗ unparseable axiom line: $line"
            violations=$((violations + 1))
            ;;
    esac
done <<< "$axiom_out"

if [ "$count" -ne "$directives" ]; then
    echo "✗ axiom inventory produced $count lines for $directives directives"
    echo "  a #print axioms directive resolved to nothing — theorem renamed?"
    exit 1
fi

if [ "$count" -ne "$EXPECTED_THEOREMS" ]; then
    echo "✗ audited surface is $count theorems, expected $EXPECTED_THEOREMS"
    if [ "$count" -lt "$EXPECTED_THEOREMS" ]; then
        echo "  a theorem was dropped from the audit — restore it, or lower"
        echo "  EXPECTED_THEOREMS in this script if the removal is intended"
    else
        echo "  new theorems added — bump EXPECTED_THEOREMS to $count"
    fi
    exit 1
fi

if [ "$violations" -gt 0 ]; then
    echo "✗ $violations axiom-budget violation(s) — budget is {${ALLOWED_AXIOMS[*]}}"
    exit 1
fi

echo "✓ $count theorems audited: $axiom_free axiom-free, $((count - axiom_free)) within budget"
echo "✓ KatgptProof gate PASSED"
