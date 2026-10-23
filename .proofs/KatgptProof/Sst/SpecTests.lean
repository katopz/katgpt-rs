/-
! Spec self-tests on concrete instances — the "spec tested on vectors"
pattern (Plan-016 convention; SymCrypt §4 distilled).

The theorems in `Basic.lean` prove the spec against itself; the Rust
spec-match mirrors (the planted-leak battery + the 2×2 arm in
`sst_certified`'s tests) test Rust against the spec. Neither catches a spec
AUTHORING error — if `DiagMap` had a sign or index typo, the proofs would
still type-check (proving the wrong property) and the Rust mirror (written
by the same author) would carry the same typo. Only concrete instances with
independently-known answers close that gap.
-/

import KatgptProof.Sst.Basic

namespace KatgptProof.Sst

/-! ## DiagMap on known vectors

If the map's index arithmetic is wrong (transposed index, wrong operand
order), these fail at `lake build` time.
-/

/-- Elementwise product on a known vector: `diag(2, 3) · (1, 1) = (2, 3)`. -/
example : DiagMap ![(2 : ℝ), 3] ![(1 : ℝ), 1] = ![(2 : ℝ), 3] := by
  funext i; fin_cases i <;> simp [DiagMap] <;> norm_num

/-- Zero vector maps to zero (any diagonal). -/
example : DiagMap ![(5 : ℝ), 7] (fun _ => (0 : ℝ)) = (0 : Fin 2 → ℝ) := by
  funext i; fin_cases i <;> simp [DiagMap] <;> norm_num

/-! ## The scalar arm on a concrete span

The theorem says EVERY span; here the 2D whole space (spanned by `e₀, e₁`)
under `diag(3, 3)` — membership and invariance both read off concretely.
-/

/-- The whole 2D space is invariant under the scalar diagonal — via the
theorem (the instance arm, not a re-proof). -/
example : NoLeak (Submodule.span ℝ ({![(1 : ℝ), 0], ![(0 : ℝ), 1]} : Set (Fin 2 → ℝ)))
    (fun _ => (3 : ℝ)) :=
  noLeak_of_scalar _ 3

/-! ## The counterexample's coordinates, independently

The leak reads off the coordinates: `DiagMap A2 u = ![1, 0]`, and `![1, 0]`
is not a multiple of `![1, 1]` (would need `r = 1` and `r = 0`
simultaneously). These instances pin the two coordinate values the Rust
mirror asserts.
-/

example : (DiagMap A2 ![(1 : ℝ), 1]) 0 = (1 : ℝ) := by norm_num [DiagMap, A2]
example : (DiagMap A2 ![(1 : ℝ), 1]) 1 = (0 : ℝ) := by norm_num [DiagMap, A2]

/-- A span multiple of `![1, 1]` has EQUAL coordinates — the transformed
vector's `(![1, 0])` unequal coordinates are exactly why it leaves. -/
example (r : ℝ) : (r • ![(1 : ℝ), 1] : Fin 2 → ℝ) 0 = (r • ![(1 : ℝ), 1] : Fin 2 → ℝ) 1 := by
  simp

end KatgptProof.Sst
