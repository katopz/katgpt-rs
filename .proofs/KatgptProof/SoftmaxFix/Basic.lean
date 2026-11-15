/-
! Spec for the `softmax_fix` family's sign certificate (katgpt-rs Issue 1221).

This Lean 4 model is the mathematical specification of the property that
`crates/katgpt-core/src/softmax_fix.rs`'s `fixed_log_softmax_target` relies on
in its argmax branch: **the target log-prob is finite, negative, and nonzero**
for every finite-logit input where the target is an argmax and a rival exists.

## Why a sign theorem is the poc (Research 673 §2's taxonomy)

The companion arc (LeanSoftmax) certifies the softmax fix two ways: mixed
relative-precision bounds (quantitative, needs a per-op rounding model — our
candidate B) and **sign theorems** (the exact form's sign survives any format
whose values are finite). Candidate A — the sign theorem — is what this file
proves. The format enters only through the boundedness hypothesis
`∀ j, |z j| ≤ B`: every finite IEEE value is a real number bounded by its
format's `maxFinite`, and the theorem is parametric in `B` — it holds for
*every* format simultaneously. That is the certificate's unbounded-region
claim: not sampled logits, **all** finite logits of any format at once.

## The form

`fixed_log_softmax_target`'s argmax branch computes `−ln(1 + R)` with
`R = Σ_{j ≠ a} e^{z_j − z_a}` (the argmax-excluded remaining mass, folded
ascending — the fold order is a determinism pin, Issue 940, and is invisible
to exact arithmetic). `remainingMass` and `logTarget` below are exactly that
shape over the reals.

## The two premises the issue's phrasing needed sharpened

1. **The argmax must be finite.** With `z a = +∞` and a finite rival, the
   exact form gives `e^{−∞} = 0`, so `R = 0`, `l = 0`, and the result is
   `−0.0` — nonzero FAILS. The model excludes this by construction (values
   of `ℝ` are finite; the boundedness hypothesis makes the exclusion
   explicit). The IEEE `+∞` argmax is outside the certificate's domain, and
   the Rust side resolves it to `−0.0` by the overwrite law — documented
   here rather than proven around.
2. **At least two classes.** `n = 1` gives `R = 0`, `l = 0`, result `0` —
   the `SpecTests` counterexample pins why `1 < n` is load-bearing.

With both premises (and boundedness, which supplies the `z_j − z_a ≥ −2B`
margin keeping every rival term positive), the exact `l` satisfies
`0 < l ≤ log n` — bounded away from zero on both sides, which is exactly
the shape a gradient-survival gate consumes.

## What this file does NOT claim

No per-op rounding model: the theorem certifies the **exact** form over the
unbounded region, not the f32/f64 computed values' error (that is candidate
B's mixed bound). The paired Rust spec-match test
(`crates/katgpt-core/tests/softmax_fix_spec_match.rs`) asserts the structural
identity — the Rust computes this form — by sampled walk, and asserts the
exact-form trinity over regions the sampled G1 ladder cannot reach.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.Exponential

namespace KatgptProof.SoftmaxFix

open Real

/-! ## The form -/

/-- The argmax-excluded remaining mass `R = Σ_{j ≠ a} e^{z_j − z_a}` — the
exact value of the pinned ascending fold (the fold order is invisible to
exact arithmetic). -/
noncomputable def remainingMass (z : Fin n → ℝ) (a : Fin n) : ℝ :=
  ∑ j, if j = a then 0 else Real.exp (z j - z a)

/-- The argmax branch's target log-prob: `−ln(1 + R)`. -/
noncomputable def logTarget (z : Fin n → ℝ) (a : Fin n) : ℝ :=
  -Real.log (1 + remainingMass z a)

/-- `a` is an argmax of `z`: no logit exceeds it (ties are allowed — the
first-index tie rule is a determinism pin, invisible to the sign claim). -/
def IsArgmax (z : Fin n → ℝ) (a : Fin n) : Prop := ∀ j, z j ≤ z a

/-! ## The sign theorem -/

/-- The sign trinity for the exact form: `0 < l ≤ log n`, hence `logTarget`
is negative, nonzero, and bounded in absolute value by `log n` — the
"finite" leg of the certificate in the mixed-bound template's absolute-term
form. The statement is format-parametric FOR FREE: the model's values are
reals, and every finite IEEE value of any format is a real — the model has
no `±∞`, so the `+∞`-argmax hazard is excluded by construction (the
premise the issue's phrasing needed sharpened; see the module doc). One
proof covers all formats and every finite logit magnitude at once — the
unbounded-region claim no sampled ladder can make. -/
theorem logTarget_sign (z : Fin n → ℝ) (a : Fin n)
    (hargmax : IsArgmax z a) (hn : 1 < n) :
    0 < Real.log (1 + remainingMass z a)
      ∧ Real.log (1 + remainingMass z a) ≤ Real.log n
      ∧ logTarget z a < 0
      ∧ logTarget z a ≠ 0
      ∧ |logTarget z a| ≤ Real.log n := by
  -- A rival exists (1 < n): if a.val = 0 take the index 1, else index 0.
  have hrival : ∃ j : Fin n, j ≠ a := by
    by_cases h : a.val = 0
    · refine ⟨⟨1, by omega⟩, ?_⟩
      intro heq
      have hv := congrArg Fin.val heq
      rw [h] at hv
      simp at hv
    · refine ⟨⟨0, by omega⟩, ?_⟩
      intro heq
      have hv0 : (⟨0, by omega⟩ : Fin n).val = 0 := rfl
      have hval := congrArg Fin.val heq
      rw [hval] at hv0
      omega
  obtain ⟨j₀, hj₀⟩ := hrival
  -- The mass is positive: every term is nonnegative, and the rival's term
  -- is bounded away from zero — both logits are bounded, so
  -- z j₀ − z a ≥ −2B and the exp of that is strictly positive.
  have hRpos : 0 < remainingMass z a := by
    rw [remainingMass]
    exact Finset.sum_pos'
      (fun j _ => by
        by_cases h : j = a
        · simp [h]
        · simp [h]; exact le_of_lt (Real.exp_pos (z j - z a)))
      ⟨j₀, by simp [hj₀]; exact Real.exp_pos (z j₀ - z a)⟩
  -- 1 + R is exactly the sum of (indicator of a) + (the mass): at j = a the
  -- term is 1 = e^{z_a − z_a}; elsewhere the exp term. Every term is ≤ 1
  -- by the argmax, so 1 + R ≤ Σ 1 = n.
  have hsplit : (1 : ℝ) + remainingMass z a
      = ∑ j : Fin n, (if j = a then (1 : ℝ) else Real.exp (z j - z a)) := by
    have hrw : remainingMass z a
        = ∑ j : Fin n, (if j = a then (0 : ℝ) else Real.exp (z j - z a)) := rfl
    rw [hrw]
    have hcongr : ∀ j : Fin n,
        (if j = a then (1 : ℝ) else Real.exp (z j - z a))
          = (if j = a then (1 : ℝ) else (0 : ℝ))
            + (if j = a then (0 : ℝ) else Real.exp (z j - z a)) := by
      intro j; by_cases h : j = a <;> simp [h]
    have h1 : ∑ j : Fin n, (if j = a then (1 : ℝ) else Real.exp (z j - z a))
        = ∑ j : Fin n, ((if j = a then (1 : ℝ) else (0 : ℝ))
            + (if j = a then (0 : ℝ) else Real.exp (z j - z a))) :=
      Finset.sum_congr rfl (fun j _ => hcongr j)
    rw [h1, Finset.sum_add_distrib]
    have hind : ∑ j : Fin n, (if j = a then (1 : ℝ) else (0 : ℝ)) = 1 := by
      simp
    rw [hind]
  have hbound : ∀ j : Fin n,
      (if j = a then (1 : ℝ) else Real.exp (z j - z a)) ≤ 1 := by
    intro j; by_cases h : j = a
    · simp [h]
    · simp [h]
      exact hargmax j
  have h1n : (1 : ℝ) + remainingMass z a ≤ (n : ℝ) := by
    have hsum1 : ∑ _j : Fin n, (1 : ℝ) = (n : ℝ) := by simp
    have hle : ∑ j : Fin n, (if j = a then (1 : ℝ) else Real.exp (z j - z a))
        ≤ ∑ _j : Fin n, (1 : ℝ) :=
      Finset.sum_le_sum (fun j _ => hbound j)
    rw [hsum1] at hle
    rw [hsplit]
    exact hle
  -- The log bounds, then the trinity.
  have h1pos : (0 : ℝ) < 1 + remainingMass z a := by linarith
  have h1lt : (1 : ℝ) < 1 + remainingMass z a := by linarith
  have hlogpos : 0 < Real.log (1 + remainingMass z a) := Real.log_pos h1lt
  have hlogle : Real.log (1 + remainingMass z a) ≤ Real.log n :=
    Real.log_le_log h1pos h1n
  refine ⟨hlogpos, hlogle, ?_, ?_, ?_⟩
  · unfold logTarget
    linarith
  · unfold logTarget
    intro h
    rw [neg_eq_zero] at h
    rw [Real.log_eq_zero] at h
    rcases h with h | h | h <;> linarith
  · unfold logTarget
    rw [abs_neg]
    rw [abs_eq_self.mpr (le_of_lt hlogpos)]
    exact hlogle

end KatgptProof.SoftmaxFix
