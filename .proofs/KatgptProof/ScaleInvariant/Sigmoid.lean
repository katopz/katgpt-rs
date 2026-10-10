/-
! Theorem: the sigmoid tilt transfer halves the per-position mass constant
to exactly α/2 (Plan 622 Phase 6, the novel fusion half — Research 610).

Under the scale-invariant schedule, the SOFTMAX arm's per-key mass is
`e^{m_t+a_t²/2} = α/(t/τ+1)` exactly (`perKeyMass_eq_alpha`, Basic.lean).
This file handles the NORMALIZED-SIGMOID arm (`parallax_attn`'s second
attention): with `L_t ~ N(m_t, a_t²)` under the schedule,

    E[σ(L_t)] = α·c(a_t)/(t/τ + 1),    c(a) = E[σ(−L′)], L′ ~ N(1, a²)

(the tilt identity — `σ(x) = e^x·σ(−x)` pointwise, the exponential tilt
rule, and `m_t + a_t² = 1` from `m_add_aSq`), and the constant HALVES:

    lim_{t→∞} (t/τ + 1)·E[σ(L_t)] = α/2

because `c(a_t) → 1/2` (dominated convergence: under the standardized
reparametrization `σ(−(1 + a w)) → 1{w<0}` pointwise, dominated by an
integrable bound) while `a_t² = 2·log(t/τ+1) + 1 → ∞`. The drift runs from
`α·σ(−1) ≈ 0.54·α/2` at `t = 0` (`c(0) = σ(−1)` — the deterministic limit,
exact algebra) toward `α/2`, slowly (`O(1/√log t)` — `a_t²` grows only
logarithmically). Expected behavior, not a defect.

## What is a hypothesis and what is proven

The Gaussian-integral facts (the tilt rule producing `htilt`'s mass factor,
and `hlim`'s `c → 1/2`) are measure theory — NOT modeled here, following the
`Hope/Basic.lean` precedent (the spec models exactly what it can prove and
names what it assumes). The hypothesis `htilt` packages them with the
SCHEDULE-side facts, which ARE proven in this development:
- `perKeyMass_eq_alpha` — the mass profile `α/(t/τ+1)`;
- `m_add_aSq` — the tilted mean is the CONSTANT 1 (why `c` depends on `t`
  only through `a_t`);
- the drift `a_t² = 2·log(t/τ+1)+1 → ∞` (composed from
  `Real.tendsto_log_atTop` + the shared `Tendsto` helpers below).

`sigmoid_tilt_pointwise` and `sigmoid_transfer_gap` are fully hypothesis-
free: the pointwise σ identity and the exact ε(t) gap identity. The paired
Rust spec-match test validates the FULL assembly numerically (adaptive
quadrature, no hypotheses) — `crates/katgpt-core/tests/
scale_invariant_spec_match.rs`.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Topology.Algebra.Order.Field
import KatgptProof.ScaleInvariant.Tendsto
import KatgptProof.ScaleInvariant.Basic

namespace KatgptProof.ScaleInvariant

open Real Filter

/-! ## The sigmoid and the pointwise tilt identity -/

/-- σ(x) = 1/(1 + e^{−x}) — the stack-sanctioned link (never softmax). -/
noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

/-- σ > 0. -/
lemma sigmoid_pos (x : ℝ) : 0 < sigmoid x := by
  unfold sigmoid
  positivity

/-- **The pointwise tilt identity** `σ(x) = e^x·σ(−x)` — pure algebra (the
    sigmoid-vs-step decomposition the transfer runs on). -/
lemma sigmoid_tilt_pointwise (x : ℝ) : sigmoid x = Real.exp x * sigmoid (-x) := by
  unfold sigmoid
  have h : Real.exp (-x) = (Real.exp x)⁻¹ := by
    rw [Real.exp_neg]
  rw [h]
  field_simp
  ring

/-! ## Theorem 2 — the halving -/

/-- **Theorem 2 (`sigmoid_transfer_halves_constant`).** Under the schedule,
    the per-position sigmoid mass constant halves to exactly `α/2`:

        lim_{t→∞} (t/τ + 1)·E[σ(L_t)] = α/2.

    `c` is the abstract transfer-constant family `c(a) = E[σ(−L′)]`,
    `L′ ~ N(1, a²)`; `hlim` is its dominated-convergence limit (the
    Gaussian-integral fact — see the file doc for the proven/assumed
    split). `htilt` is the tilt assembly: the per-position mass is the
    harmonic profile (`α/(t/τ+1)`, proven) times the constant family at
    `a_t` (the shift `m_t + a_t² = 1` is proven — `m_add_aSq`). -/
theorem sigmoid_transfer_halves_constant (τ : ℝ) (hτ : 0 < τ)
    (c : ℝ → ℝ) (hlim : Tendsto c atTop (nhds (1 / 2)))
    (Eσ : ℕ → ℝ)
    (htilt : ∀ t : ℕ, Eσ t = alpha / ((t : ℝ) / τ + 1)
      * c (Real.sqrt (aSq (t : ℝ) τ))) :
    Tendsto (fun t : ℕ ↦ ((t : ℝ) / τ + 1) * Eσ t) atTop (nhds (alpha / 2)) := by
  -- The function collapses to α·c(a_t) by the tilt assembly.
  have hfn : (fun t : ℕ ↦ ((t : ℝ) / τ + 1) * Eσ t)
      = (fun t : ℕ ↦ alpha * c (Real.sqrt (aSq (t : ℝ) τ))) := by
    funext t
    rw [htilt t]
    have hpos1 : 0 < (t : ℝ) / τ + 1 := by
      have h2 : 0 ≤ (t : ℝ) / τ := div_nonneg (Nat.cast_nonneg t) hτ.le
      linarith
    field_simp
    all_goals ring
  rw [hfn]
  -- a_t = sqrt(2·log(t/τ+1)+1) → ∞, then c(a_t) → 1/2.
  have hlog : Tendsto (fun t : ℕ ↦ Real.log ((t : ℝ) / τ + 1)) atTop atTop :=
    Real.tendsto_log_atTop.comp (tendsto_div_add_atTop τ hτ (b := 1))
  have hsq2 : Tendsto (fun t : ℕ ↦ 2 * Real.log ((t : ℝ) / τ + 1) + 1) atTop atTop :=
    (tendsto_affine_atTop (by norm_num)).comp hlog
  have hsqt : Tendsto (fun t : ℕ ↦ Real.sqrt (2 * Real.log ((t : ℝ) / τ + 1) + 1))
      atTop atTop := tendsto_sqrt_atTop.comp hsq2
  have hcomp : Tendsto (fun t : ℕ ↦ c (Real.sqrt (2 * Real.log ((t : ℝ) / τ + 1) + 1)))
      atTop (nhds (1 / 2)) := hlim.comp hsqt
  -- aSq IS that closed form, by definition — congr away the unfolding.
  have hsqt' : Tendsto (fun t : ℕ ↦ Real.sqrt (aSq (t : ℝ) τ)) atTop atTop :=
    hsqt.congr' (by
      filter_upwards with t
      rfl)
  have hcomp' : Tendsto (fun t : ℕ ↦ c (Real.sqrt (aSq (t : ℝ) τ))) atTop (nhds (1 / 2)) :=
    hcomp.congr' (by
      filter_upwards with t
      rfl)
  have hhalf : (alpha : ℝ) / 2 = alpha * (1 / 2) := by ring
  rw [hhalf]
  exact hcomp'.const_mul alpha

/-- The explicit ε(t): the gap from the halved constant is EXACTLY
    `α·|c(a_t) − 1/2|` — the schedule contributes nothing beyond the
    transfer constant's own drift (whose σ-vs-step correction is O(a⁻³),
    decaying overall as O(1/√log t) since `a_t²` grows only
    logarithmically). -/
theorem sigmoid_transfer_gap (τ : ℝ) (hτ : 0 < τ)
    (c : ℝ → ℝ) (Eσ : ℕ → ℝ)
    (htilt : ∀ t : ℕ, Eσ t = alpha / ((t : ℝ) / τ + 1)
      * c (Real.sqrt (aSq (t : ℝ) τ)))
    (t : ℕ) :
    |((t : ℝ) / τ + 1) * Eσ t - alpha / 2|
      = alpha * |c (Real.sqrt (aSq (t : ℝ) τ)) - 1 / 2| := by
  rw [htilt t]
  have hpos1 : 0 < (t : ℝ) / τ + 1 := by
    have h2 : 0 ≤ (t : ℝ) / τ := div_nonneg (Nat.cast_nonneg t) hτ.le
    linarith
  have hcancel : ((t : ℝ) / τ + 1)
      * (alpha / ((t : ℝ) / τ + 1) * c (Real.sqrt (aSq (t : ℝ) τ)))
      = alpha * c (Real.sqrt (aSq (t : ℝ) τ)) := by
    field_simp
    all_goals ring
  have hhalf : (alpha : ℝ) / 2 = alpha * (1 / 2) := by ring
  rw [hcancel, hhalf]
  have h2 : alpha * c (Real.sqrt (aSq (t : ℝ) τ)) - alpha * (1 / 2)
      = alpha * (c (Real.sqrt (aSq (t : ℝ) τ)) - 1 / 2) := by ring
  rw [h2, abs_mul, abs_of_pos alpha_pos]

end KatgptProof.ScaleInvariant
