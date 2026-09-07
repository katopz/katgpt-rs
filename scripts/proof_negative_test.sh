#!/usr/bin/env bash
# KatgptProof negative test — measured error-catching power (Issue 678 P4;
# mirrors riir-chain's Plan 016 G2 pattern).
#
# The pencil SpecTests and theorems prove the spec against itself; the Rust
# spec-match tests prove Rust against the spec. NEITHER catches a spec
# transcription error. The concrete hand-instances in
# KatgptProof/Pencil/SpecTests.lean close that gap — this script MEASURES
# it: each perturbation below simulates a spec typo, and `lake build` MUST
# fail. A perturbation that builds green is a hole.
#
# Usage: scripts/proof_negative_test.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROOFS="$REPO_ROOT/.proofs"
SRC="$PROOFS/KatgptProof/Pencil"

export PATH="$HOME/.elan/bin:$PATH"

if ! command -v lake >/dev/null 2>&1; then
    echo "✗ lake (Lean 4) not installed — negative test cannot run"
    echo "  install: curl https://elan.lean-lang.org/elan_init.sh -sSf | sh"
    exit 1
fi

BACKUP_DIR="$(mktemp -d)"
# `NEG_COMPLETED` is the completion sentinel (Issue 734). Restoring the
# perturbed sources is only half the job: measured on macOS `/bin/bash` 3.2.57
# and ONLY there (Issue 735 — bash 4.4 through 5.3, dash and busybox ash all
# PRESERVE the status), a `set -u` abort (or an `eval` syntax error) enters the
# EXIT trap with `$?` ALREADY 0, so a handler whose last command succeeds makes
# the abort exit **0** — this script would restore every file and then report
# "all perturbations caught" by silence, having run none of them. Since this
# script is only ever run by hand, on a Mac, 3.2 is the interpreter that
# matters for it. Saving and re-exiting `$?` does not help; the saved value is
# itself 0. Only "did the script reach its own last line?" catches it — and
# that catches every other premature death, on every shell.
NEG_COMPLETED=0
restore_all() {
    neg_st=$?
    local f
    for f in "$BACKUP_DIR"/*.bak; do
        [ -e "$f" ] || continue
        local name
        name="$(basename "$f" .bak)"
        cp "$f" "${name//__//}"
    done
    rm -rf "$BACKUP_DIR"
    if [ "$NEG_COMPLETED" != "1" ] && [ "$neg_st" = "0" ]; then
        echo "✗ negative test ABORTED mid-run while reporting success — sources were" >&2
        echo "  restored, but the perturbations below the abort never ran. Forcing exit 1." >&2
        exit 1
    fi
    exit "$neg_st"
}
trap restore_all EXIT

pass=0
fail=0

# Perturb one file, assert `lake build` fails, then revert.
#   $1 = file path relative to Pencil/
#   $2 = sed expression
#   $3 = human description of the bug being simulated
perturb() {
    local rel="$1" expr="$2" desc="$3"
    local file="$SRC/$rel"
    local key="${SRC}/${rel}"
    local bak="$BACKUP_DIR/${key//\//__}.bak"

    cp "$file" "$bak"
    sed "$expr" "$bak" > "$file"

    # A sed that matched nothing would "pass" vacuously — that is a harness
    # bug, not a proof result. Catch it explicitly.
    if cmp -s "$bak" "$file"; then
        echo "  ✗ HARNESS BUG: perturbation matched nothing (spec text changed?)"
        echo "     expr: $expr"
        cp "$bak" "$file"
        fail=$((fail + 1))
        return
    fi

    if (cd "$PROOFS" && lake build >/dev/null 2>&1); then
        echo "  ✗ BUILD PASSED — this bug would ship undetected"
        fail=$((fail + 1))
    else
        echo "  ✓ build failed as required"
        pass=$((pass + 1))
    fi

    cp "$bak" "$file"
    rm -f "$bak"
}

echo "── Pencil negative tests (Issue 678 P4) ──"

echo "P1: the 1/√2 packing scale typo (paper's own example: 1/√2 → 1/2)"
perturb "Sym.lean" \
    's/(Real.sqrt 2)⁻¹ \* v i j/(2 : ℝ)⁻¹ * v i j/' \
    "symMat stores off-diagonals scaled by 1/2 instead of 1/√2"

echo "P2: the hand-instance expected value (18 → 17)"
perturb "SpecTests.lean" \
    's/frobSq (symMat v22) = 18/frobSq (symMat v22) = 17/' \
    "spec test expects the wrong Frobenius norm"

echo "P3: mirror-pairing factor (2 * upper → upper, i.e. dropped mirror)"
perturb "Sym.lean" \
    's/2 \* ∑ p ∈ Finset.univ.filter (fun p : Fin D × Fin D => p.1 < p.2),/∑ p ∈ Finset.univ.filter (fun p : Fin D × Fin D => p.1 < p.2),/' \
    "off-diagonal double sum forgets the ×2 mirror factor"

echo "P4: Weyl inequality direction (≤ → <)"
perturb "Weyl.lean" \
    's/    |eigval hA i - eigval hB i| ≤ ‖A - B‖ := by/    |eigval hA i - eigval hB i| < ‖A - B‖ := by/' \
    "Weyl stated strictly — false at equality (diagonal ground truth)"

echo "P5: ladder spec-test expected gap (5/4 → 7/4)"
perturb "SpecTests.lean" \
    's/    = 5 \/ 4 := by/    = 7 \/ 4 := by/' \
    "T4 spec test expects the wrong perturbed-ladder gap"

echo "P6: diagonal eigenvalue pin sign (d j → -d j)"
perturb "Eigengap.lean" \
    's/    eigval (diagonal_isHermitian d) j = d j := by/    eigval (diagonal_isHermitian d) j = -d j := by/' \
    "antitone-diagonal theorem pins the NEGATED diagonal — false (d = const 5)"

echo ""
echo "── HintRegret negative tests (Plan 576) ──"
SRC="$PROOFS/KatgptProof/HintRegret"

echo "P7: band-gate factor drop (product → single rising factor)"
perturb "Basic.lean" \
    's/sigmoid (κ \* (w - wLo)) \* sigmoid (κ \* (wHi - w))/sigmoid (κ * (w - wLo))/' \
    "gate drops the falling wall — the theorem survives (single sigmoid ∈ (0,1)) but the κ=0 flat instance (1/4) and the wall instances (<1/2) must fail"

echo "P8: spec-test flat constant typo (1/4 → 1/2)"
perturb "SpecTests.lean" \
    's/bandGate w wLo wHi 0 = (1:ℝ) \/ 4 := by/bandGate w wLo wHi 0 = (1:ℝ) \/ 2 := by/' \
    "κ=0 instance expects σ(0)·σ(0)=1/2 instead of 1/4"

if [ "$fail" -gt 0 ]; then
    echo "✗ $fail perturbation(s) built green — spec-test holes"
    exit 1
fi
echo "✓ all $pass perturbations caught — the spec tests have teeth"
NEG_COMPLETED=1  # the last line — see restore_all above
