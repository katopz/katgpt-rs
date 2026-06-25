/-
! Theorem: `ActionBridge::select_action` ranking is preserved by sigmoid.

This is the Tier 3 bridge formal-verification result (Plan 293). It states:

    ∀ (q d₁ d₂ : ι → ℝ), dot q d₁ > dot q d₂ →
        sigmoid (dot q d₁) > sigmoid (dot q d₂)

i.e. if action `d₁` has a strictly larger dot product with the query than
action `d₂`, then `d₁` also receives a strictly larger sigmoid score. This is
the property that makes the bridge **safe to sync** across the latent/raw
boundary: scalar ordering preserves belief ordering, so downstream consumers
can rank entities by the projected scalar without needing the latent vector.

The proof is one step: `Real.sigmoid_strictMono` (a Mathlib theorem) gives
`StrictMono sigmoid`, and `StrictMono` applied to a strict inequality yields a
strict inequality. No `sorry`. No admitted axioms beyond Mathlib's standard
foundations.

This promotes the empirical `g1_3_bridge_ranking_preservation` property test
(`micro_belief/tests.rs`, 1000 random triples) from `∃` (there exist 1000
triples for which it holds) to `∀` (it holds for *every* triple).
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import KatgptProof.Bridge.Basic

namespace KatgptProof.Bridge

open Real

/-! ## Ranking preservation

The `select_action` inner loop computes `dot q d_a` for each action `d_a`, then
projects through `sigmoid`. Selecting the argmax is correct **iff** sigmoid
preserves the dot-product ordering — i.e. iff sigmoid is strictly monotone.
Mathlib proves exactly this as `Real.sigmoid_strictMono`.
-/

/-- **ActionBridge ranking preservation.** If action `d₁` has a strictly larger
    dot product with query `q` than action `d₂`, then `d₁`'s sigmoid score is
    also strictly larger. This is the ∀-form of the `g1_3_bridge_ranking_preservation`
    empirical test (Plan 281 G1.3), holding for *every* `(q, d₁, d₂)` triple,
    not just 1000 sampled ones.

    The proof: `Real.sigmoid_strictMono : StrictMono sigmoid` (Mathlib), and
    `StrictMono f` means `a < b → f a < f b` by definition. -/
theorem action_bridge_ranking_preserved
    {ι : Type*} [Fintype ι] (q d₁ d₂ : ι → ℝ)
    (h : dot q d₁ > dot q d₂) :
    sigmoid (dot q d₁) > sigmoid (dot q d₂) := by
  -- `Real.sigmoid_strictMono : StrictMono sigmoid`.
  -- `StrictMono.apply_strictMono_lt` (via `Real.sigmoid_lt`) closes the goal:
  -- `a < b → sigmoid a < sigmoid b`.
  exact Real.sigmoid_lt h

/-- Equivalent statement via the `StrictMono` interface directly: sigmoid is a
    strictly monotone function, hence an order embedding on the dot-product
    ordering. This is the form `Plan 293` T2.4 sketches
    (`exact strictMono_sigmoid _ _ h`). -/
theorem action_bridge_ranking_preserved' (a b : ℝ) (h : a > b) :
    sigmoid a > sigmoid b :=
  Real.sigmoid_lt h

/-! ## Corollary: argmax is preserved

`select_action` returns the action with the largest sigmoid score. Because
sigmoid is strictly monotone, this is *exactly* the action with the largest dot
product. The bridge therefore computes the mathematically correct argmax under
the dot-product ordering — no action can "win" via sigmoid that did not already
win via dot product.
-/

/-- If `d₁` has the strictly largest dot product among a finite set of actions,
    then `d₁` also has the strictly largest sigmoid score. (Argmax preservation
    under a strictly monotone projection.) -/
theorem action_bridge_argmax_preserved
    {ι : Type*} [Fintype ι] (q : ι → ℝ) (d₁ : ι → ℝ) (actions : Finset (ι → ℝ))
    (h_max : d₁ ∈ actions ∧ ∀ d₂ ∈ actions, d₁ ≠ d₂ → dot q d₁ > dot q d₂) :
    ∀ d₂ ∈ actions, d₁ ≠ d₂ → sigmoid (dot q d₁) > sigmoid (dot q d₂) := by
  intros d₂ hd₂ hne
  exact action_bridge_ranking_preserved q d₁ d₂ ((h_max.2 d₂ hd₂ hne))

end KatgptProof.Bridge
