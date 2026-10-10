/-
! Spec self-tests for the Scale-Invariant schedule on concrete instances
(Plan 441 convention — the spec tested against independently-known values).

Distilled from SymCrypt `feature/verifiedcrypto` §4 ("Running the Lean spec
on test vectors"). The closed forms in `Basic.lean` are transcribed from
arXiv:2505.17083 (the boundary-pinned schedule); a transcription typo
(e.g. `2·log(t/τ+1) + 1` mistyped as `2·log(t/τ+2) + 1`) would propagate
silently — the theorems prove whatever is transcribed and the Rust
spec-match likely carries the same typo (same author, same paper). The
concrete instances below are the independent authority:

- the boundary pins `(a₀², m₀) = (1, 0)` — the schedule's DEFINING data;
- the paper cell `a_τ² = 2·log 2 + 1` at `t = τ` (where the ramp ends);
- `perKeyMass τ τ = α/2` (the exact harmonic value at the decade point);
- `σ(0) = 1/2` (sigmoid symmetry) and the tilt identity at the origin;
- the empty decade (Δ = 1) has mass 0;
- the floor is positive at the paper's defaults (Δ = 2, τ = 10);
- the per-key mass is strictly antitone in distance (the harmonic
  profile's shape — what Fig. 1's local-mass preservation reads).
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import KatgptProof.ScaleInvariant.Basic
import KatgptProof.ScaleInvariant.Sigmoid
import KatgptProof.ScaleInvariant.Dominates

namespace KatgptProof.ScaleInvariant

open Real

/-! ## Boundary pins (the schedule's defining data) -/

/-- The boundary pin: `a₀² = 1`. -/
example (τ : ℝ) : aSq 0 τ = 1 := aSq_boundary τ

/-- The boundary pin: `m₀ = 0`. -/
example (τ : ℝ) : mOf 0 τ = 0 := mOf_boundary τ

/-- The constant-sum invariant holds at every distance. -/
example (t τ : ℝ) : mOf t τ + aSq t τ = 1 := m_add_aSq t τ

/-- α = e^(1/2) — the boundary-pinned constant. -/
example : alpha = Real.exp (1 / 2) := alpha_eq

/-! ## The paper cell at `t = τ` -/

/-- At `t = τ` the ramp ends: `a_τ² = 2·log 2 + 1` (the paper's schedule at
    the first decade point — independently computable from
    `a_t² = 2·log(t/τ+1) + 1`). -/
example (τ : ℝ) (hτ : 0 < τ) : aSq τ τ = 2 * Real.log 2 + 1 := by
  have h1 : (τ : ℝ) / τ = 1 := div_self (by linarith)
  show 2 * Real.log ((τ : ℝ) / τ + 1) + 1 = 2 * Real.log 2 + 1
  rw [h1]
  norm_num

/-- The per-key mass at the decade point: `perKeyMass τ τ = α/2` — the
    exact harmonic value (`e^{m+a²/2}` with `m + a² = 1` gives
    `e^{1/2}/2`). A sign typo in `mOf` or `aSq` fails this. -/
example (τ : ℝ) (hτ : 0 < τ) : perKeyMass τ τ = alpha / 2 := by
  rw [perKeyMass_eq_alpha τ τ hτ (by linarith)]
  have h1 : (τ : ℝ) / τ = 1 := div_self (by linarith)
  rw [h1]
  ring

/-! ## The sigmoid half -/

/-- σ(0) = 1/2 — sigmoid symmetry at the origin. -/
example : sigmoid 0 = 1 / 2 := by
  show 1 / (1 + Real.exp (-(0 : ℝ))) = 1 / 2
  norm_num

/-- The tilt identity at the origin: `σ(0) = e^0·σ(0)`. -/
example : sigmoid 0 = Real.exp 0 * sigmoid (-(0 : ℝ)) := by
  rw [sigmoid_tilt_pointwise]

/-! ## The decade window -/

/-- The empty decade (Δ = 1) has mass 0. -/
example (T : ℕ) (τ : ℝ) : decadeMass T 1 τ = 0 := by
  simp [decadeMass]

/-- The floor is positive at the paper's defaults (Δ = 2, τ = 10). -/
example : 0 < decadeMassFloor 2 10 :=
  decadeMassFloor_pos 2 (by norm_num) (by norm_num)

/-! ## The harmonic profile's shape -/

/-- The per-key mass is STRICTLY antitone in distance — the harmonic
    profile α/(t/τ+1) falls as the window slides out (what the paper's
    Fig. 1 shows the position-independent comparators forfeiting). -/
example (t₁ t₂ τ : ℝ) (hτ : 0 < τ) (ht₁ : 0 ≤ t₁) (ht₂ : 0 ≤ t₂)
    (h : t₁ < t₂) : perKeyMass t₂ τ < perKeyMass t₁ τ := by
  rw [perKeyMass_eq_alpha t₂ τ hτ ht₂, perKeyMass_eq_alpha t₁ τ hτ ht₁]
  have hden₂ : (0 : ℝ) < t₂ / τ + 1 := by
    have h8 : 0 ≤ t₂ / τ := div_nonneg ht₂ hτ.le
    linarith
  have hden₁ : (0 : ℝ) < t₁ / τ + 1 := by
    have h9 : 0 ≤ t₁ / τ := div_nonneg ht₁ hτ.le
    linarith
  refine (div_lt_div_iff₀ (b := t₂ / τ + 1) (d := t₁ / τ + 1)
    hden₂ hden₁).mpr ?_
  refine mul_lt_mul_of_pos_left ?_ alpha_pos
  have h5 : (t₁ : ℝ) / τ < (t₂ : ℝ) / τ := by
    have h6 : (t₂ : ℝ) / τ - (t₁ : ℝ) / τ = (t₂ - t₁) / τ := by ring
    have h7 : 0 < (t₂ - t₁) / τ := div_pos (by linarith) hτ
    linarith
  linarith

end KatgptProof.ScaleInvariant
