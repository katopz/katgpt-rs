/-
! Spec for the Scale-Invariant Attention schedule (Plan 622 Phase 6).

This Lean 4 model is the mathematical specification of the schedule that
`katgpt-rs/crates/katgpt-core/src/scale_invariant.rs` implements, distilled
from Anson, Wang & Aitchison, *Scale-invariant Attention* (arXiv:2505.17083
v2, NeurIPS 2025) — Research 610:

    L_t → a_t·L_t + m_t
    a_t² = 2·(log(t/τ + 1) − log α + β/α),   m_t = −a_t² + β/α
    boundary (a₀², m₀) = (1, 0)  ⟹  α = β = e^(1/2)      (single knob: τ)

## What this file proves

1. **The schedule's provable skeleton** (from the closed forms alone):
   the boundary pins `a₀² = 1, m₀ = 0` (`aSq_boundary`, `mOf_boundary`),
   the constant-sum invariant `m_t + a_t² = β/α = 1` (`m_add_aSq` — what
   makes the sigmoid tilt's shifted mean the CONSTANT 1), and the per-key
   mass profile `E[e^{L_t}] = e^{m_t + a_t²/2} = α/(t/τ + 1)`
   (`perKeyMass_eq_alpha`) — the harmonic profile the decade theorem runs on.

2. **`scale_invariant_decade_mass_bounded`** — the harmonic-sum interval
   bound (the paper's App. D argument): for `τ > 0`, `T ≥ 1`, `Δ ≥ 1`,

       α·τ·log((TΔ+1+τ)/(T+1+τ))  ≤  Σ_{t=T+1}^{TΔ} perKeyMass  ≤  α·τ·log((TΔ+τ)/(T+τ))

   proved by the log-reciprocal sandwich (`one_div_le_log_div`,
   `log_div_le_one_div`, both from Mathlib's `log_le_sub_one_of_pos`)
   telescoped over the window (`sum_log_telescope_up/down`). Axiom budget
   unchanged: Mathlib foundations only.

3. **The Θ(1) band (Def 3.1)**: `decadeMass_le_logDelta` (the upper leg —
   never above `α·τ·log Δ` for ANY `T`) and `decadeMass_ge_floor` +
   `decadeMassFloor_pos` (the lower leg — never below the T-independent
   positive floor `α·τ·log((Δ+1+τ)/(2+τ))` for `Δ ≥ 2`). The floor algebra
   cross-multiplies to `(1+τ)·(T−1)·(Δ−1) ≥ 0` — nonneg exactly on the
   `T ≥ 1, Δ ≥ 1` quadrant, the same sharpened-threshold shape the SSMax
   proof found (`s_L·log N ≥ 1`, not `N ≥ 2`).

The expectation `E[e^{L_t}]` itself is NOT modeled — the Gaussian moment
`E[e^L] = e^{m+a²/2}` is a measure-theoretic fact outside this spec; what
is provable (and proven) here is every consequence the implementation
relies on, starting from that closed form. The companion Rust spec-match
test `crates/katgpt-core/tests/scale_invariant_spec_match.rs` validates
the schedule LUT and the interval bounds numerically at f64 precision.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Topology.Algebra.Order.Field
import KatgptProof.ScaleInvariant.Tendsto

namespace KatgptProof.ScaleInvariant

open Real

/-! ## The schedule's closed forms -/

/-- α = β = e^(1/2) — the two schedule constants, forced by the boundary
    pin. Derivation (the Rust `build` runs the same two lines in f64):
    `m₀ = −a₀² + β/α = 0` gives `β/α = 1`; `a₀² = 2·(0 − log α + β/α) = 1`
    gives `log α = 1/2`; so `α = β = e^(1/2)`. -/
noncomputable def alpha : ℝ := Real.exp (1 / 2)

/-- `alpha` unfolds to `e^(1/2)`. -/
lemma alpha_eq : alpha = Real.exp (1 / 2) := rfl

/-- α > 0. -/
lemma alpha_pos : 0 < alpha := Real.exp_pos _

/-- The schedule's sharpening coefficient squared: `a_t² = 2·log(t/τ + 1) + 1`
    (the closed form after the boundary pin). -/
noncomputable def aSq (t τ : ℝ) : ℝ := 2 * Real.log (t / τ + 1) + 1

/-- The schedule's additive term: `m_t = 1 − a_t²` (= −2·log(t/τ + 1)). -/
noncomputable def mOf (t τ : ℝ) : ℝ := 1 - aSq t τ

/-- Boundary pin, first half: `a₀² = 1` — distance-0 logits keep their
    unit sharpening. -/
lemma aSq_boundary (τ : ℝ) : aSq 0 τ = 1 := by
  simp [aSq]

/-- Boundary pin, second half: `m₀ = 0` — distance-0 logits are shifted by
    nothing. -/
lemma mOf_boundary (τ : ℝ) : mOf 0 τ = 0 := by
  simp [mOf, aSq]

/-- The constant-sum invariant `m_t + a_t² = β/α = 1` for EVERY distance:
    under the schedule the shifted logit's mean is the constant 1 — the
    fact that makes the sigmoid tilt transfer's constant family
    `c(a) = E[σ(−L′)]`, `L′ ~ N(1, a²)`, independent of `t` except through
    `a_t` (`Sigmoid.lean`). -/
lemma m_add_aSq (t τ : ℝ) : mOf t τ + aSq t τ = 1 := by
  simp only [mOf]
  ring

/-! ## The per-key mass profile -/

/-- The per-key unnormalized softmax mass: `E[e^{L_t}] = e^{m_t + a_t²/2}`
    for Gaussian logits `L_t ~ N(m_t, a_t²)` — the paper's App. B moment.
    The moment itself is measure theory; this spec fixes the closed form
    and proves its consequences. -/
noncomputable def perKeyMass (t τ : ℝ) : ℝ := Real.exp (mOf t τ + aSq t τ / 2)

/-- The schedule's design identity: `perKeyMass t τ = α/(t/τ + 1)` — the
    harmonic profile the decade theorem consumes. `t ≥ 0` keeps the log's
    argument positive. -/
lemma perKeyMass_eq_alpha (t τ : ℝ) (hτ : 0 < τ) (ht : 0 ≤ t) :
    perKeyMass t τ = alpha / (t / τ + 1) := by
  have hx : 0 < t / τ + 1 := by
    have h2 : 0 ≤ t / τ := div_nonneg ht hτ.le
    linarith
  rw [perKeyMass]
  have hsplit : mOf t τ + aSq t τ / 2 = 1 / 2 - Real.log (t / τ + 1) := by
    simp only [mOf, aSq]
    ring
  rw [hsplit, Real.exp_sub, Real.exp_log hx, alpha_eq]

/-- The same profile over `t + τ`: `perKeyMass t τ = α·τ/(t + τ)` — the
    form the harmonic-sum bounds telescope. -/
lemma perKeyMass_eq_alpha_tau (t τ : ℝ) (hτ : 0 < τ) (ht : 0 ≤ t) :
    perKeyMass t τ = alpha * τ / (t + τ) := by
  rw [perKeyMass_eq_alpha t τ hτ ht]
  have hrat : t / τ + 1 = (t + τ) / τ := by field_simp
  rw [hrat, div_div_eq_mul_div]

/-! ## The decade mass -/

/-- The per-decade unnormalized attention mass: the sum of per-key masses
    over the causal decade window `(T, TΔ]` — `t = T+1 … T·Δ` in ℕ indices
    (the window the paper's `E[Z_t^{tΔ}]` sums). -/
noncomputable def decadeMass (T Δ : ℕ) (τ : ℝ) : ℝ :=
  ∑ t ∈ Finset.Icc (T + 1) (T * Δ), perKeyMass (t : ℝ) τ

/-! ## The log-reciprocal sandwich (the App. D per-term bounds)

Both legs follow from Mathlib's single inequality `log x ≤ x − 1`:
applied directly it gives the lower leg; applied to `1/z` (via `log_inv`)
it gives `log z ≥ 1 − 1/z`, the upper leg. -/

/-- Upper leg: for `t ≥ 1`, `τ > 0`: `1/(t+τ) ≤ log((t+τ)/(t−1+τ))`. -/
lemma one_div_le_log_div (t τ : ℝ) (hτ : 0 < τ) (ht : 1 ≤ t) :
    1 / (t + τ) ≤ Real.log ((t + τ) / (t - 1 + τ)) := by
  have hzpos : 0 < (t + τ) / (t - 1 + τ) :=
    div_pos (by linarith) (by linarith)
  have h := Real.log_le_sub_one_of_pos (inv_pos.mpr hzpos)
  rw [Real.log_inv, inv_eq_one_div] at h
  have hkey : 1 - 1 / ((t + τ) / (t - 1 + τ)) = 1 / (t + τ) := by
    field_simp
    ring
  linarith [hkey]

/-- Lower leg: for `t ≥ 0`, `τ > 0`: `log((t+1+τ)/(t+τ)) ≤ 1/(t+τ)`. -/
lemma log_div_le_one_div (t τ : ℝ) (hτ : 0 < τ) (ht : 0 ≤ t) :
    Real.log ((t + 1 + τ) / (t + τ)) ≤ 1 / (t + τ) := by
  have hpos : 0 < (t + 1 + τ) / (t + τ) :=
    div_pos (by linarith) (by linarith)
  have h := Real.log_le_sub_one_of_pos hpos
  have hkey : (t + 1 + τ) / (t + τ) - 1 = 1 / (t + τ) := by
    field_simp
    ring
  rwa [hkey] at h

/-! ## The telescopes -/

/-- The upper-leg telescope: `Σ_{t=p}^{q} log((t+τ)/(t−1+τ)) =
    log((q+τ)/(p−1+τ))` — every intermediate `t+τ` cancels. -/
lemma sum_log_telescope_up (τ : ℝ) (hτ : 0 < τ) :
    ∀ q p : ℕ, 1 ≤ p → p ≤ q →
      ∑ t ∈ Finset.Icc p q, Real.log (((t : ℝ) + τ) / ((t : ℝ) - 1 + τ))
        = Real.log ((q + τ) / (p - 1 + τ)) := by
  intro q
  induction q with
  | zero =>
    intro p hp hpq
    exact absurd hpq (by omega)
  | succ k ih =>
    intro p hp hpq
    have hk0 : (0 : ℝ) ≤ (k : ℝ) := Nat.cast_nonneg k
    have hp1 : (1 : ℝ) ≤ (p : ℝ) := by exact_mod_cast hp
    have hkτ : 0 < (k : ℝ) + τ := by linarith
    have hp1τ : 0 < (p : ℝ) - 1 + τ := by linarith
    rcases Nat.lt_or_ge k p with hkp | hpk
    · -- p = k + 1: the window is the single point p
      have hpe : p = k + 1 := by omega
      subst hpe
      rw [Finset.Icc_self, Finset.sum_singleton]
    · -- p ≤ k: split off the top element
      have hpk1 : p ≤ k + 1 := by omega
      have hA : ((k : ℝ) + τ) / ((p : ℝ) - 1 + τ) ≠ 0 :=
        div_ne_zero (ne_of_gt hkτ) (ne_of_gt hp1τ)
      have hB : ((k : ℝ) + 1 + τ) / ((k : ℝ) + τ) ≠ 0 :=
        div_ne_zero (by linarith) (ne_of_gt hkτ)
      rw [Finset.sum_Icc_succ_top hpk1, ih p hp hpk]
      push_cast
      norm_num
      rw [← Real.log_mul hA hB]
      congr 1
      field_simp [show ((k : ℝ) + τ) ≠ 0 from ne_of_gt hkτ,
        show ((p : ℝ) - 1 + τ) ≠ 0 from ne_of_gt hp1τ]
      all_goals ring

/-- The lower-leg telescope: `Σ_{t=p}^{q} log((t+1+τ)/(t+τ)) =
    log((q+1+τ)/(p+τ))`. -/
lemma sum_log_telescope_down (τ : ℝ) (hτ : 0 < τ) :
    ∀ q p : ℕ, 1 ≤ p → p ≤ q →
      ∑ t ∈ Finset.Icc p q, Real.log (((t : ℝ) + 1 + τ) / ((t : ℝ) + τ))
        = Real.log ((q + 1 + τ) / (p + τ)) := by
  intro q
  induction q with
  | zero =>
    intro p hp hpq
    exact absurd hpq (by omega)
  | succ k ih =>
    intro p hp hpq
    have hk0 : (0 : ℝ) ≤ (k : ℝ) := Nat.cast_nonneg k
    have hk1τ : 0 < (k : ℝ) + 1 + τ := by linarith
    have hpτ : 0 < (p : ℝ) + τ := by linarith
    rcases Nat.lt_or_ge k p with hkp | hpk
    · -- p = k + 1: the window is the single point p
      have hpe : p = k + 1 := by omega
      subst hpe
      rw [Finset.Icc_self, Finset.sum_singleton]
    · -- p ≤ k: split off the top element
      have hpk1 : p ≤ k + 1 := by omega
      have hA : ((k : ℝ) + 1 + τ) / ((p : ℝ) + τ) ≠ 0 :=
        div_ne_zero (ne_of_gt hk1τ) (ne_of_gt hpτ)
      have hB : ((k : ℝ) + 1 + 1 + τ) / ((k : ℝ) + 1 + τ) ≠ 0 :=
        div_ne_zero (by linarith) (ne_of_gt hk1τ)
      rw [Finset.sum_Icc_succ_top hpk1, ih p hp hpk]
      push_cast
      norm_num
      rw [← Real.log_mul hA hB]
      congr 1
      field_simp [show ((k : ℝ) + 1 + τ) ≠ 0 from ne_of_gt hk1τ,
        show ((p : ℝ) + τ) ≠ 0 from ne_of_gt hpτ]
      all_goals ring

/-! ## Theorem 1 — the harmonic-sum interval bound -/

/-- **Theorem 1 (`scale_invariant_decade_mass_bounded`).** For `τ > 0`,
    `T ≥ 1`, `Δ ≥ 1`, the decade mass over the causal window `(T, TΔ]`
    satisfies the paper's harmonic-sum interval bounds (App. D):

    `α·τ·log((TΔ+1+τ)/(T+1+τ))  ≤  decadeMass T Δ τ  ≤  α·τ·log((TΔ+τ)/(T+τ))`

    Both legs telescope the per-term log-reciprocal sandwich. The Θ(1)
    band these imply (`decadeMass_le_logDelta`, `decadeMass_ge_floor`) is
    the paper's Def 3.1. -/
theorem scale_invariant_decade_mass_bounded (τ : ℝ) (hτ : 0 < τ)
    (T Δ : ℕ) (hT : 0 < T) (hΔ : 1 ≤ Δ) :
    alpha * τ * Real.log (((T * Δ : ℝ) + 1 + τ) / ((T : ℝ) + 1 + τ))
      ≤ decadeMass T Δ τ
    ∧ decadeMass T Δ τ
      ≤ alpha * τ * Real.log (((T * Δ : ℝ) + τ) / ((T : ℝ) + τ)) := by
  have hT0 : (0 : ℝ) ≤ (T : ℝ) := Nat.cast_nonneg T
  rcases Nat.eq_or_lt_of_le hΔ with hΔ1 | hΔ2
  · -- Δ = 1: the window (T, T] is empty; both log arguments are 1.
    subst hΔ1
    have hTpos : (0 : ℝ) < (T : ℝ) + 1 + τ := by linarith
    have hTpos2 : (0 : ℝ) < (T : ℝ) + τ := by linarith
    have hempty : Finset.Icc (T + 1) (T * 1) = ∅ := by
      rw [Nat.mul_one]
      exact Finset.Icc_eq_empty_iff.mpr (by omega)
    constructor
    · rw [decadeMass, hempty, Finset.sum_empty, Nat.cast_one, mul_one,
        div_self (ne_of_gt hTpos), Real.log_one, mul_zero]
    · rw [decadeMass, hempty, Finset.sum_empty, Nat.cast_one, mul_one,
        div_self (ne_of_gt hTpos2), Real.log_one, mul_zero]
  · -- Δ ≥ 2: the window is nonempty (T·Δ ≥ 2·T ≥ T+1).
    have h2 : T * 2 ≤ T * Δ := Nat.mul_le_mul_left T (by omega)
    have hne : T + 1 ≤ T * Δ := by omega
    -- The mass is the harmonic sum ατ/(t+τ).
    have hsum : decadeMass T Δ τ
        = ∑ t ∈ Finset.Icc (T + 1) (T * Δ), alpha * τ / ((t : ℝ) + τ) := by
      rw [decadeMass]
      refine Finset.sum_congr rfl fun t _ => ?_
      exact perKeyMass_eq_alpha_tau _ _ hτ (Nat.cast_nonneg t)
    rw [hsum]
    constructor
    · -- Lower leg: Σ ατ·log((t+1+τ)/(t+τ)) ≤ Σ ατ/(t+τ), telescoped.
      have hlogsum : ∑ t ∈ Finset.Icc (T + 1) (T * Δ),
          alpha * τ * Real.log (((t : ℝ) + 1 + τ) / ((t : ℝ) + τ))
          = alpha * τ * Real.log (((T * Δ : ℝ) + 1 + τ) / ((T : ℝ) + 1 + τ)) := by
        rw [← Finset.mul_sum,
          sum_log_telescope_down τ hτ (T * Δ) (T + 1) (by omega) hne]
        push_cast
        ring
      have hterm : ∀ t ∈ Finset.Icc (T + 1) (T * Δ),
          alpha * τ * Real.log (((t : ℝ) + 1 + τ) / ((t : ℝ) + τ))
          ≤ alpha * τ / ((t : ℝ) + τ) := by
        intro t ht
        have hm : T + 1 ≤ t ∧ t ≤ T * Δ := Finset.mem_Icc.mp ht
        have h1t : (1 : ℝ) ≤ (t : ℝ) := by
          have h2c : (((T + 1 : ℕ) : ℝ)) ≤ (t : ℝ) := by exact_mod_cast hm.1
          have h3c : (((T + 1 : ℕ) : ℝ)) = (T : ℝ) + 1 := by push_cast; ring
          have h4c : (1 : ℝ) ≤ (T : ℝ) := by exact_mod_cast (show (0:ℕ) < T from hT)
          linarith
        have h := log_div_le_one_div ((t : ℕ) : ℝ) τ hτ (by linarith)
        calc alpha * τ * Real.log (((t : ℝ) + 1 + τ) / ((t : ℝ) + τ))
            ≤ alpha * τ * (1 / ((t : ℝ) + τ)) :=
              mul_le_mul_of_nonneg_left h (mul_nonneg alpha_pos.le hτ.le)
          _ = alpha * τ / ((t : ℝ) + τ) := by ring
      rw [← hlogsum]
      exact Finset.sum_le_sum hterm
    · -- Upper leg: Σ ατ/(t+τ) ≤ Σ ατ·log((t+τ)/(t−1+τ)), telescoped.
      have hlogsum : ∑ t ∈ Finset.Icc (T + 1) (T * Δ),
          alpha * τ * Real.log (((t : ℝ) + τ) / ((t : ℝ) - 1 + τ))
          = alpha * τ * Real.log (((T * Δ : ℝ) + τ) / ((T : ℝ) + τ)) := by
        rw [← Finset.mul_sum,
          sum_log_telescope_up τ hτ (T * Δ) (T + 1) (by omega) hne]
        push_cast
        ring
      have hterm : ∀ t ∈ Finset.Icc (T + 1) (T * Δ),
          alpha * τ / ((t : ℝ) + τ)
          ≤ alpha * τ * Real.log (((t : ℝ) + τ) / ((t : ℝ) - 1 + τ)) := by
        intro t ht
        have hm : T + 1 ≤ t ∧ t ≤ T * Δ := Finset.mem_Icc.mp ht
        have h1t : (1 : ℝ) ≤ (t : ℝ) := by
          have h2c : (((T + 1 : ℕ) : ℝ)) ≤ (t : ℝ) := by exact_mod_cast hm.1
          have h3c : (((T + 1 : ℕ) : ℝ)) = (T : ℝ) + 1 := by push_cast; ring
          have h4c : (1 : ℝ) ≤ (T : ℝ) := by exact_mod_cast (show (0:ℕ) < T from hT)
          linarith
        have h := one_div_le_log_div _ τ hτ h1t
        calc alpha * τ / ((t : ℝ) + τ)
            = alpha * τ * (1 / ((t : ℝ) + τ)) := by ring
          _ ≤ alpha * τ * Real.log (((t : ℝ) + τ) / ((t : ℝ) - 1 + τ)) :=
              mul_le_mul_of_nonneg_left h (mul_nonneg alpha_pos.le hτ.le)
      rw [← hlogsum]
      exact Finset.sum_le_sum hterm

/-! ## The Def-3.1 Θ(1) band -/

/-- The upper band leg: the decade mass NEVER exceeds `α·τ·log Δ`, for any
    `T` — the ratio `(TΔ+τ)/(T+τ)` is increasing in `T` toward `Δ`
    (`τ ≤ Δ·τ` is exactly `Δ ≥ 1`). -/
theorem decadeMass_le_logDelta (τ : ℝ) (hτ : 0 < τ) (T Δ : ℕ) (hT : 0 < T)
    (hΔ : 1 ≤ Δ) : decadeMass T Δ τ ≤ alpha * τ * Real.log (Δ : ℝ) := by
  have hT0 : (0 : ℝ) ≤ (T : ℝ) := Nat.cast_nonneg T
  have hΔ1 : (1 : ℝ) ≤ (Δ : ℝ) := by exact_mod_cast hΔ
  have hTΔ0 : (0 : ℝ) ≤ (T : ℝ) * (Δ : ℝ) := mul_nonneg hT0 (by linarith)
  have hτle : (1 : ℝ) * τ ≤ (Δ : ℝ) * τ := by
    rw [one_mul]
    have h3 : (1 : ℝ) * τ ≤ (Δ : ℝ) * τ :=
      mul_le_mul_of_nonneg_right hΔ1 (by linarith)
    rwa [one_mul] at h3
  rcases scale_invariant_decade_mass_bounded τ hτ T Δ hT hΔ with ⟨_, hup⟩
  refine le_trans hup ?_
  refine mul_le_mul_of_nonneg_left ?_ (mul_nonneg alpha_pos.le hτ.le)
  refine Real.log_le_log (div_pos (by linarith) (by linarith)) ?_
  -- (TΔ+τ)/(T+τ) ≤ Δ  ⟺  TΔ+τ ≤ ΔT+Δτ  ⟺  τ ≤ Δτ.
  rw [div_le_iff₀ (by linarith)]
  have hexpand : (Δ : ℝ) * ((T : ℝ) + τ) = (T : ℝ) * (Δ : ℝ) + (Δ : ℝ) * τ := by
    ring
  rw [hexpand]
  linarith

/-- The T-independent floor of the band: `α·τ·log((Δ+1+τ)/(2+τ))` — the
    lower leg's limit as `T → ∞`, attained from above at `T = 1`. -/
noncomputable def decadeMassFloor (Δ τ : ℝ) : ℝ :=
  alpha * τ * Real.log ((Δ + 1 + τ) / (2 + τ))

/-- The floor is positive for `Δ ≥ 2` (the ratio exceeds 1 exactly when
    `Δ > 1`). -/
theorem decadeMassFloor_pos (Δ : ℕ) (hτ : 0 < τ) (hΔ : 2 ≤ Δ) :
    0 < decadeMassFloor (Δ : ℝ) τ := by
  have hΔ2 : (2 : ℝ) ≤ (Δ : ℝ) := by exact_mod_cast hΔ
  unfold decadeMassFloor
  refine mul_pos (mul_pos alpha_pos hτ) (Real.log_pos ?_)
  rw [one_lt_div₀ (by linarith : (0 : ℝ) < 2 + τ)]
  linarith

/-- The lower band leg: for EVERY `T ≥ 1` the decade mass stays above the
    T-independent floor — the schedule keeps distant decades attended
    (Def 3.1's non-degeneracy half). The ratio algebra:
    `(TΔ+1+τ)·(2+τ) − (Δ+1+τ)·(T+1+τ) = (1+τ)·(T−1)·(Δ−1) ≥ 0` exactly on
    the `T ≥ 1, Δ ≥ 1` quadrant. -/
theorem decadeMass_ge_floor (τ : ℝ) (hτ : 0 < τ) (T Δ : ℕ) (hT : 0 < T)
    (hΔ : 1 ≤ Δ) : decadeMassFloor (Δ : ℝ) τ ≤ decadeMass T Δ τ := by
  have hT0 : (0 : ℝ) ≤ (T : ℝ) := Nat.cast_nonneg T
  have hΔ1 : (1 : ℝ) ≤ (Δ : ℝ) := by exact_mod_cast hΔ
  have hTΔ0 : (0 : ℝ) ≤ (T : ℝ) * (Δ : ℝ) := mul_nonneg hT0 (by linarith)
  rcases scale_invariant_decade_mass_bounded τ hτ T Δ hT hΔ with ⟨hlo, _⟩
  refine le_trans ?_ hlo
  refine mul_le_mul_of_nonneg_left ?_ (mul_nonneg alpha_pos.le hτ.le)
  refine Real.log_le_log (div_pos (by linarith) (by linarith)) ?_
  rw [div_le_div_iff₀ (by linarith) (by linarith [hT0, hτ])]
  have hfac : ((T * Δ : ℝ) + 1 + τ) * (2 + τ)
      - ((Δ : ℝ) + 1 + τ) * ((T : ℝ) + 1 + τ)
      = (1 + τ) * (((T : ℝ) - 1) * ((Δ : ℝ) - 1)) := by ring
  have hT1 : 0 ≤ (T : ℝ) - 1 := by
    have hT1r : (1 : ℝ) ≤ (T : ℝ) := by exact_mod_cast (show (0:ℕ) < T from hT)
    linarith
  have hΔm1 : 0 ≤ (Δ : ℝ) - 1 := by linarith
  have hprod : 0 ≤ (1 + τ) * (((T : ℝ) - 1) * ((Δ : ℝ) - 1)) :=
    mul_nonneg (by linarith) (mul_nonneg hT1 hΔm1)
  linarith [hfac, hprod]

end KatgptProof.ScaleInvariant
