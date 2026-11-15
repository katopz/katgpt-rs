/-
! Spec self-tests on concrete instances — the "spec tested on vectors" pattern.

Distilled from SymCrypt `feature/verifiedcrypto` §4 and the repo's
Plan-441 convention (the Bridge/Ssmax SpecTests are the house examples):
the sign theorem proves the spec against itself, and the Rust spec-match
test tests Rust against the spec — neither catches a spec-authoring error.
These instances do: every row has an independently-known answer.

The rows also PIN the premises the theorem needs (the two sharpened points
of the issue's phrasing): the `n = 1` counterexample is the `1 < n` leg,
and the deep-tail row shows the bounded-away-from-zero half is honest even
when a rival is astronomically below the argmax — the exact log stays
strictly positive, which is what the computed form resolves to
exponent-underflow only through ROUNDING, never through the exact form.
-/

import KatgptProof.SoftmaxFix.Basic

namespace KatgptProof.SoftmaxFix

open Real

/-! ## The three-class closure row

`z = [ln 4, ln 2, ln 2]`, target `0` (the argmax): the mass is
`e^{ln2 − ln4} + e^{ln2 − ln4} = 1/2 + 1/2 = 1`, so `l = log 2` — the
companion arc's `Real.log (8/7)` closure shape, in the same spirit. -/

/-- The closure row. -/
noncomputable def z3 : Fin 3 → ℝ := ![Real.log 4, Real.log 2, Real.log 2]

/-- Ordering facts about the closure row's values. -/
private theorem log2_le_log4 : Real.log 2 ≤ Real.log 4 :=
  Real.log_le_log (by norm_num : (0 : ℝ) < 2) (by norm_num : (2 : ℝ) ≤ 4)

private theorem log2_le_log3 : Real.log 2 ≤ Real.log 3 :=
  Real.log_le_log (by norm_num : (0 : ℝ) < 2) (by norm_num : (2 : ℝ) ≤ 3)

/-- One rival term: `e^{ln2 − ln4} = 1/2`. -/
private theorem exp_term : Real.exp (Real.log 2 - Real.log 4) = 1 / 2 := by
  rw [Real.exp_sub, Real.exp_log (by norm_num : (0 : ℝ) < 2),
    Real.exp_log (by norm_num : (0 : ℝ) < 4)]
  norm_num

/-- The closure row's index values (the vector applications reduced). -/
private theorem z3_zero : z3 0 = Real.log 4 := rfl

private theorem z3_one : z3 1 = Real.log 2 := rfl

private theorem z3_two : z3 2 = Real.log 2 := rfl

/-- The mass on the closure row is exactly 1. -/
example : remainingMass z3 0 = 1 := by
  rw [remainingMass, Fin.sum_univ_three, z3_zero, z3_one, z3_two,
    if_pos (rfl : ((0 : Fin 3) = 0)),
    if_neg (by decide : ¬((1 : Fin 3) = 0)),
    if_neg (by decide : ¬((2 : Fin 3) = 0)),
    exp_term]
  norm_num

/-- The trinity holds on the closure row: `log 2` is positive, ≤ `log 3`,
and `−log 2` is negative and nonzero. -/
example :
    0 < Real.log (1 + remainingMass z3 0)
      ∧ Real.log (1 + remainingMass z3 0) ≤ Real.log 3
      ∧ logTarget z3 0 < 0
      ∧ logTarget z3 0 ≠ 0 := by
  have hR : remainingMass z3 0 = 1 := by
    rw [remainingMass, Fin.sum_univ_three, z3_zero, z3_one, z3_two,
      if_pos (rfl : ((0 : Fin 3) = 0)),
      if_neg (by decide : ¬((1 : Fin 3) = 0)),
      if_neg (by decide : ¬((2 : Fin 3) = 0)),
      exp_term]
    norm_num
  have h2pos : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hR]; norm_num; exact h2pos
  · rw [hR]; norm_num; exact log2_le_log3
  · show -(Real.log (1 + remainingMass z3 0)) < 0
    rw [hR]; norm_num; linarith
  · show -(Real.log (1 + remainingMass z3 0)) ≠ 0
    rw [hR]
    intro h
    rw [neg_eq_zero, Real.log_eq_zero] at h
    norm_num at h

/-- The closure row's target really is an argmax (the premise check). -/
example : IsArgmax z3 0 := by
  intro j
  fin_cases j
  · simp only [z3]
    exact le_refl _
  · simp only [z3]
    exact log2_le_log4
  · simp only [z3]
    exact log2_le_log4

/-! ## The premises are load-bearing -/

/-- **Counterexample (`1 < n` leg):** a single class gives mass 0, log 1 = 0,
and the target log-prob is 0 — not negative, not nonzero. This is why the
theorem carries `1 < n`. -/
example : logTarget (![5] : Fin 1 → ℝ) 0 = 0 := by
  simp [logTarget, remainingMass]

/-! ## The deep-tail row

A rival astronomically below the argmax: the exact mass stays strictly
positive (`e^{−1000} > 0`), so the exact log is strictly positive too. The
computed form can still land on `−0.0` here THROUGH ROUNDING (the f32 exp
of −1000 underflows to 0) — which is precisely why the certificate claims
the exact form and why candidate B (the mixed bound) is the computed-value
follow-up. -/

example : 0 < Real.log (1 + remainingMass (![0, -1000] : Fin 2 → ℝ) 0) := by
  have hR : 0 < remainingMass (![0, -1000] : Fin 2 → ℝ) 0 := by
    simp [remainingMass]
    positivity
  have h1 : (1 : ℝ) < 1 + remainingMass (![0, -1000] : Fin 2 → ℝ) 0 := by linarith
  exact Real.log_pos h1

/-! ## The theorem instantiates on the closure row

One discharge of `logTarget_sign` at the concrete row — the end-to-end
plumbing check (premises accepted, trinity produced). -/

example :
    logTarget z3 0 < 0 ∧ logTarget z3 0 ≠ 0 := by
  have h := logTarget_sign z3 0
    (by
      intro j
      fin_cases j
      · simp only [z3]; exact le_refl _
      · simp only [z3]; exact log2_le_log4
      · simp only [z3]; exact log2_le_log4)
    (by norm_num)
  exact ⟨h.2.2.1, h.2.2.2.1⟩

end KatgptProof.SoftmaxFix
