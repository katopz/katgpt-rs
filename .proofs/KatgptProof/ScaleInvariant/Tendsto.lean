/-
! Shared explicit-bound `Tendsto` helpers for the ScaleInvariant module
(Plan 622 Phase 6).

Each helper is proved directly from `Filter.tendsto_atTop_atTop`'s ∃-form —
no dependence on which spelled limit lemma this Mathlib snapshot ships,
which keeps the downstream theorem proofs (Sigmoid.lean, Dominates.lean)
stable across Mathlib bumps.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Topology.Order.Basic
import Mathlib.Topology.Algebra.Order.Field
import Mathlib.Order.Filter.AtTopBot.Floor

namespace KatgptProof.ScaleInvariant

open Filter

/-- A positive affine map sends `atTop` to `atTop`. -/
lemma tendsto_affine_atTop {a b : ℝ} (ha : 0 < a) :
    Tendsto (fun x : ℝ ↦ a * x + b) atTop atTop := by
  rw [Filter.tendsto_atTop_atTop]
  intro r
  refine ⟨(r - b) / a, fun x hx => ?_⟩
  have hx' : r - b ≤ x * a := (div_le_iff₀ ha).mp hx
  linarith [mul_comm x a]

/-- The ℕ→ℝ cast tends to `atTop`. -/
lemma tendsto_natCast_atTop : Tendsto (fun t : ℕ ↦ (t : ℝ)) atTop atTop := by
  rw [Filter.tendsto_atTop_atTop]
  intro r
  refine ⟨Nat.ceil r, fun t ht => ?_⟩
  have h1 : (r : ℝ) ≤ ((Nat.ceil r : ℕ) : ℝ) := Nat.le_ceil r
  have h2 : (((Nat.ceil r : ℕ) : ℝ) ≤ ((t : ℕ) : ℝ)) := by exact_mod_cast ht
  linarith

/-- Dividing by a positive `τ` and adding an offset stays `atTop` over ℕ. -/
lemma tendsto_div_add_atTop (τ : ℝ) (hτ : 0 < τ) {b : ℝ} :
    Tendsto (fun t : ℕ ↦ (t : ℝ) / τ + b) atTop atTop := by
  rw [Filter.tendsto_atTop_atTop]
  intro r
  refine ⟨Nat.ceil ((r - b) * τ), fun t ht => ?_⟩
  have hge : ((r - b) * τ) ≤ ((t : ℕ) : ℝ) := by
    have h0 : (Nat.ceil ((r - b) * τ) : ℕ) ≤ t := ht
    have h1 : (((r - b) * τ : ℝ)) ≤ ((Nat.ceil ((r - b) * τ) : ℕ) : ℝ) := Nat.le_ceil _
    have h2 : (((Nat.ceil ((r - b) * τ) : ℕ) : ℝ) ≤ ((t : ℕ) : ℝ)) := by
      exact_mod_cast h0
    linarith
  have hdiv : r - b ≤ (t : ℝ) / τ := (le_div_iff₀ hτ).mpr hge
  linarith

end KatgptProof.ScaleInvariant
