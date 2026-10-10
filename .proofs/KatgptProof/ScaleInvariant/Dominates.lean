/-
! Theorem: the scale-invariant schedule DOMINATES every position-decaying
comparator on decade mass — SSMax/ALiBi fail the Def-3.1 band
(Plan 622 Phase 6, the comparator ordering).

Def 3.1 (the paper's scale-invariant total attention): the per-decade
unnormalized mass is Θ(1) as the window slides to infinity — bounded ABOVE
and BELOW by positive T-independent constants. Basic.lean proves the
schedule satisfies both halves (`decadeMass_le_logDelta`,
`decadeMass_ge_floor` + `decadeMassFloor_pos`). This file proves the
comparator half: both canonical position-DECAYING families COLLAPSE their
decade mass to zero, so no schedule of their shape can hold the band:

1. **The ALiBi shape** — per-key mass `C·q^t` (geometric decay; exactly the
   Gaussian family `m_t = −κt` with position-independent variance at
   `q = e^{−κ}`). Decade mass ≤ `C·(q^{T+1}/(1−q))` → 0.

2. **The dilution shape** — per-key mass `C·(1+t)^{−κ}` (polynomial decay;
   what SSMax's dilution bound leaves behind: `α_gold → 1` forces the
   distractor mass per key to decay polynomially — `Ssmax/Asymptotic.lean`
   is that theorem). Decade mass ≤ `C·Δ·T^{1−κ}` → 0 for `κ > 1`
   (and for `κ ≤ 1` it diverges — the UPPER band fails instead; either way
   Def 3.1 fails).

Position-INDEPENDENT multipliers (SSMax's single scalar `s_L·log N`) are
the degenerate case of the same failure: a scalar cannot concentrate mass
at any window, so the window mass shares the global decay — the collapse
the paper's Fig. 1 shows (uniform LogN scaling forfeits distant decades).
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Topology.Order.Basic
import Mathlib.Topology.Algebra.Order.Field
import KatgptProof.ScaleInvariant.Tendsto
import KatgptProof.ScaleInvariant.Basic

namespace KatgptProof.ScaleInvariant

open Real Filter

/-! ## Helper: the finite geometric window bound -/

/-- A geometric window sums to at most the first omitted term over
    `(1 − q)`: `Σ_{t=a}^{a+k} q^t ≤ q^a·(1 − q^{k+1})/(1 − q)` — proved by
    induction on the window length with `Finset.sum_Icc_succ_top`. -/
private lemma sum_Icc_geom_le {q : ℝ} (hq : 0 < q) (hqlt : q < 1) :
    ∀ k a : ℕ, ∑ t ∈ Finset.Icc a (a + k), q ^ t
      ≤ q ^ a * ((1 - q ^ (k + 1 : ℕ)) / (1 - q)) := by
  have hq1 : (1 : ℝ) - q ≠ 0 := by linarith
  intro k
  induction k with
  | zero =>
    intro a
    rw [Nat.add_zero, Finset.Icc_self, Finset.sum_singleton, Nat.zero_add, pow_one,
      div_self hq1, mul_one]
  | succ m ih =>
    intro a
    have hgoal : a + (m + 1) = (a + m) + 1 := by ring
    have hsplit : ∑ t ∈ Finset.Icc a ((a + m) + 1), q ^ t
        = ∑ t ∈ Finset.Icc a (a + m), q ^ t + q ^ ((a + m) + 1) :=
      Finset.sum_Icc_succ_top (a := a) (b := a + m) (by omega) (fun k => q ^ k)
    have hpow : q ^ (a + m + 1) = q ^ a * q ^ (m + 1) := by
      conv_lhs => rw [Nat.add_assoc, pow_add]
    have hqa : 0 < q ^ a := pow_pos hq a
    have hdiv : (1 - q ^ (m + 1 : ℕ)) + q ^ (m + 1) * (1 - q)
        ≤ (1 - q ^ (m + 1 + 1 : ℕ)) := by
      have e1 : q ^ (m + 1 + 1) = q ^ (m + 1) * q := pow_succ q (m + 1)
      rw [e1]
      ring_nf
      linarith
    have hqa : 0 < q ^ a := pow_pos hq a
    have h1' := ih a
    rw [hgoal, hsplit, hpow]
    have hmid : q ^ a * ((1 - q ^ (m + 1 : ℕ)) / (1 - q)) + q ^ a * q ^ (m + 1)
        = q ^ a * (((1 - q ^ (m + 1 : ℕ)) + q ^ (m + 1) * (1 - q)) / (1 - q)) := by
      field_simp [hq1]
    refine le_trans (add_le_add h1' (le_refl _)) ?_
    rw [hmid]
    refine mul_le_mul_of_nonneg_left
      ((div_le_div_iff₀ (by linarith) (by linarith)).mpr
        (mul_le_mul_of_nonneg_right hdiv (by linarith))) hqa.le

/-! ## Helper: a bounded shifted window sum -/

/-- A window of `n + 1` terms starting at `s`, each at most `K ≥ 0`, sums
    to at most `(n+1)·K` — proved by induction on the window length with
    `Finset.sum_Icc_succ_top` (no ℕ-smul lemma needed). -/
private lemma sum_window_le_mul {g : ℕ → ℝ} {K : ℝ} {s : ℕ}
    (hg : ∀ t, s ≤ t → g t ≤ K) :
    ∀ n : ℕ, ∑ t ∈ Finset.Icc s (s + n), g t ≤ ((n + 1 : ℕ) : ℝ) * K := by
  intro n
  induction n with
  | zero =>
    rw [Nat.add_zero, Finset.Icc_self, Finset.sum_singleton, Nat.cast_one, one_mul]
    exact hg s (Nat.le_refl s)
  | succ m ih =>
    have hsplit : ∑ t ∈ Finset.Icc s (s + (m + 1)), g t
        = ∑ t ∈ Finset.Icc s (s + m), g t + g (s + m + 1) := by
      have h1 : s + (m + 1) = (s + m) + 1 := by ring
      rw [h1, Finset.sum_Icc_succ_top (a := s) (b := s + m) (by omega)]
    calc ∑ t ∈ Finset.Icc s (s + (m + 1)), g t
        = ∑ t ∈ Finset.Icc s (s + m), g t + g (s + m + 1) := hsplit
      _ ≤ ((m + 1 : ℕ) : ℝ) * K + K :=
          add_le_add ih (hg (s + m + 1) (by omega))
      _ = ((m + 1 + 1 : ℕ) : ℝ) * K := by
          push_cast
          ring

/-! ## Helper: q^{T+1} → 0 -/

private lemma tendsto_pow_succ_zero {q : ℝ} (hq : 0 < q) (hqlt : q < 1) :
    Tendsto (fun T : ℕ ↦ q ^ (T + 1)) atTop (nhds 0) := by
  have h := tendsto_pow_atTop_nhds_zero_of_lt_one hq.le hqlt
  exact h.comp (Filter.tendsto_add_atTop_nat 1)

/-! ## The ALiBi-shaped comparator -/

/-- The ALiBi-shaped comparator family: per-key mass `C·q^t` summed over
    the causal decade window `(T, TΔ]` (ℕ exponent — `q = e^{−κ}` realizes
    the `m_t = −κt`, constant-variance Gaussian family exactly). -/
noncomputable def decadeMassExp (C q : ℝ) (T Δ : ℕ) : ℝ :=
  C * ∑ t ∈ Finset.Icc (T + 1) (T * Δ), q ^ t

/-- The geometric tail bound: the comparator's decade mass is at most the
    first omitted term over `(1 − q)` — the quantity that → 0. -/
lemma decadeMassExp_le (hC : 0 ≤ C) (hq : 0 < q) (hqlt : q < 1) (T Δ : ℕ) :
    decadeMassExp C q T Δ ≤ C * (q ^ (T + 1) / (1 - q)) := by
  unfold decadeMassExp
  by_cases hne : T + 1 ≤ T * Δ
  · -- The window is Icc (T+1) ((T+1) + (TΔ−T−1)).
    have hbound : (∑ t ∈ Finset.Icc (T + 1) (T * Δ), q ^ t)
        ≤ q ^ (T + 1) / (1 - q) := by
      have hexp : (T * Δ - T - 1) + 1 = T * Δ - T := by omega
      have hwin : (T + 1) + (T * Δ - T - 1) = T * Δ := by omega
      have hsplit := sum_Icc_geom_le hq hqlt (T * Δ - T - 1) (T + 1)
      rw [hexp, hwin] at hsplit
      have hA : (0 : ℝ) ≤ q ^ (T * Δ - T) := le_of_lt (pow_pos hq _)
      have hinner : (1 - q ^ (T * Δ - T : ℕ)) / (1 - q) ≤ 1 / (1 - q) := by
        refine (div_le_div_iff₀ (by linarith) (by linarith)).mpr ?_
        calc (1 - q ^ (T * Δ - T : ℕ)) * (1 - q)
            = 1 * (1 - q) - q ^ (T * Δ - T : ℕ) * (1 - q) := by ring
          _ ≤ 1 * (1 - q) := by
              exact sub_le_self _ (mul_nonneg hA (by linarith))
      refine le_trans hsplit ?_
      refine le_trans (mul_le_mul_of_nonneg_left hinner (pow_pos hq _).le) ?_
      rw [mul_one_div]
    exact mul_le_mul_of_nonneg_left hbound hC
  · -- Empty window.
    rw [Finset.Icc_eq_empty_iff.mpr (by omega), Finset.sum_empty, mul_zero]
    exact mul_nonneg hC (div_nonneg (pow_pos hq _).le (by linarith))

/-- The ALiBi comparator's decade mass → 0 — its Def-3.1 LOWER band fails
    (distant decades attend to nothing). -/
theorem decadeMassExp_tendsto_zero (hC : 0 ≤ C) (hq : 0 < q) (hqlt : q < 1)
    (Δ : ℕ) : Tendsto (fun T : ℕ ↦ decadeMassExp C q T Δ) atTop (nhds 0) := by
  have hmid : Tendsto (fun T : ℕ ↦ C * (q ^ (T + 1) / (1 - q))) atTop (nhds 0) := by
    simpa [mul_zero, zero_div] using
      (((tendsto_pow_succ_zero hq hqlt).div_const (1 - q)).const_mul C)
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hmid
    ((Filter.eventually_ge_atTop 0).mono fun T _ =>
      mul_nonneg hC (Finset.sum_nonneg fun t _ => pow_nonneg hq.le t))
    ((Filter.eventually_ge_atTop 0).mono fun T _ => decadeMassExp_le hC hq hqlt T Δ)

/-! ## The dilution-shaped comparator -/

/-- The dilution-shaped comparator family: per-key mass `C·(1+t)^{−κ}`
    summed over the causal decade window `(T, TΔ]` — the polynomial decay
    SSMax's dilution bound leaves behind (gold mass → 1 forces the
    distractor mass per key to decay polynomially in the corpus/rank
    parametrization; `Ssmax/Asymptotic.lean` is the α_gold → 1 theorem). -/
noncomputable def decadeMassPow (C κ : ℝ) (T Δ : ℕ) : ℝ :=
  C * ∑ t ∈ Finset.Icc (T + 1) (T * Δ), (1 + (t : ℝ)) ^ (-κ)

/-- The polynomial tail bound: `decadeMassPow ≤ C·Δ·T^{1−κ}` for `T ≥ 1`,
    `κ > 1` — each term in the window is at most `T^{−κ}` (the antitone
    leg of the rpow sandwich), the window has at most `TΔ` terms, and
    `TΔ·T^{−κ} = Δ·T·T^{−κ} = Δ·T^{1−κ}`. -/
lemma decadeMassPow_le (hC : 0 ≤ C) (hκ : 1 < κ) (hT : 0 < T) (Δ : ℕ) :
    decadeMassPow C κ T Δ ≤ C * (Δ : ℝ) * (T : ℝ) ^ (1 - κ) := by
  have hκ0 : 0 ≤ κ := le_of_lt (by linarith)
  have hT0 : (0 : ℝ) ≤ (T : ℝ) := Nat.cast_nonneg T
  have hTposR : (0 : ℝ) < (T : ℝ) := by exact_mod_cast hT
  have hTpos : 0 < (T : ℝ) ^ (-κ) := Real.rpow_pos_of_pos hTposR (-κ)
  have hterm : ∀ t, T + 1 ≤ t → (1 + (t : ℝ)) ^ (-κ) ≤ (T : ℝ) ^ (-κ) := by
    intro t ht
    have h2c : (((T + 1 : ℕ) : ℝ)) ≤ ((t : ℕ) : ℝ) := by exact_mod_cast ht
    have h3c : (((T + 1 : ℕ) : ℝ)) = (T : ℝ) + 1 := by push_cast; ring
    have h1 : (T : ℝ) ≤ 1 + (t : ℝ) := by linarith
    have hbase1 : (T : ℝ) ^ κ ≤ (1 + (t : ℝ)) ^ κ :=
      Real.rpow_le_rpow hT0 h1 hκ0
    have hbp : 0 < (1 + (t : ℝ)) ^ κ :=
      Real.rpow_pos_of_pos (by linarith) κ
    have hbt : 0 < (T : ℝ) ^ κ := Real.rpow_pos_of_pos hTposR κ
    have hinv := (inv_le_inv₀ hbp hbt).mpr hbase1
    rw [← Real.rpow_neg hT0 κ, ← Real.rpow_neg (by linarith) κ] at hinv
    exact hinv
  unfold decadeMassPow
  by_cases hne : T + 1 ≤ T * Δ
  · -- The window is Icc (T+1) ((T+1) + (TΔ−T−1)): ≤ (TΔ−T)·T^{−κ} ≤ Δ·T·T^{−κ}.
    have hwin : (T + 1) + (T * Δ - T - 1) = T * Δ := by omega
    have hexp : (T * Δ - T - 1) + 1 = T * Δ - T := by omega
    have hraw := sum_window_le_mul
      (fun t ht => hterm t ht) (T * Δ - T - 1)
    rw [hwin, hexp] at hraw
    have h1 : (((T * Δ - T : ℕ) : ℝ)) * ((T : ℝ) ^ (-κ))
        ≤ (((T * Δ : ℕ) : ℝ)) * ((T : ℝ) ^ (-κ)) :=
      mul_le_mul_of_nonneg_right
        (by exact_mod_cast (by omega : T * Δ - T ≤ T * Δ)) hTpos.le
    have h2 : (((T * Δ : ℕ) : ℝ)) * ((T : ℝ) ^ (-κ))
        ≤ (Δ : ℝ) * ((T : ℝ) * ((T : ℝ) ^ (-κ))) := by
      have hTΔle : (((T * Δ : ℕ) : ℝ)) ≤ (Δ : ℝ) * (T : ℝ) := by
        push_cast
        linarith [mul_comm (T : ℝ) (Δ : ℝ)]
      calc ((T * Δ : ℕ) : ℝ) * ((T : ℝ) ^ (-κ))
          ≤ (Δ : ℝ) * (T : ℝ) * ((T : ℝ) ^ (-κ)) :=
            mul_le_mul_of_nonneg_right hTΔle hTpos.le
        _ = (Δ : ℝ) * ((T : ℝ) * ((T : ℝ) ^ (-κ))) := mul_assoc _ _ _
    have h3 : (Δ : ℝ) * ((T : ℝ) * ((T : ℝ) ^ (-κ)))
        = (Δ : ℝ) * (T : ℝ) ^ (1 - κ) := by
      have h1r : (T : ℝ) ^ (1 - κ) = (T : ℝ) ^ (1 + -κ) :=
        congrArg (fun e => (T : ℝ) ^ e) (by ring)
      rw [h1r, Real.rpow_add hTposR, Real.rpow_one]
    have e1 : C * ∑ t ∈ Finset.Icc (T + 1) (T * Δ), (1 + (t : ℝ)) ^ (-κ)
        ≤ C * (((T * Δ - T : ℕ) : ℝ) * ((T : ℝ) ^ (-κ))) :=
      mul_le_mul_of_nonneg_left hraw hC
    have e2 : C * (((T * Δ - T : ℕ) : ℝ) * ((T : ℝ) ^ (-κ)))
        ≤ C * (((T * Δ : ℕ) : ℝ) * ((T : ℝ) ^ (-κ))) :=
      mul_le_mul_of_nonneg_left h1 hC
    have e3 : C * (((T * Δ : ℕ) : ℝ) * ((T : ℝ) ^ (-κ)))
        ≤ C * ((Δ : ℝ) * ((T : ℝ) * ((T : ℝ) ^ (-κ)))) :=
      mul_le_mul_of_nonneg_left h2 hC
    refine le_trans e1 (le_trans e2 (le_trans e3 ?_))
    rw [mul_assoc, h3]
  · -- Empty window.
    rw [Finset.Icc_eq_empty_iff.mpr (by omega), Finset.sum_empty, mul_zero]
    simp only [mul_assoc]
    exact mul_nonneg hC (mul_nonneg (by linarith) (Real.rpow_nonneg hT0 (1 - κ)))

/-- The dilution comparator's decade mass → 0 for `κ > 1` (and for `κ ≤ 1`
    it diverges — the upper band fails instead; either way Def 3.1
    fails). -/
theorem decadeMassPow_tendsto_zero (hC : 0 ≤ C) (hκ : 1 < κ) (Δ : ℕ) :
    Tendsto (fun T : ℕ ↦ decadeMassPow C κ T Δ) atTop (nhds 0) := by
  have hneg : (1 : ℝ) - κ < 0 := by linarith
  have hloglin : Tendsto (fun T : ℕ ↦ (1 - κ) * Real.log (T : ℝ)) atTop atBot :=
    Filter.Tendsto.const_mul_atTop_of_neg hneg
      (Real.tendsto_log_atTop.comp tendsto_natCast_atTop)
  have hexp : Tendsto (fun T : ℕ ↦ Real.exp ((1 - κ) * Real.log (T : ℝ))) atTop
      (nhds 0) := Real.tendsto_exp_atBot.comp hloglin
  have hpow : Tendsto (fun T : ℕ ↦ (T : ℝ) ^ (1 - κ)) atTop (nhds 0) := by
    have key : ∀ T : ℕ, 1 ≤ T → (T : ℝ) ^ (1 - κ)
        = Real.exp ((1 - κ) * Real.log (T : ℝ)) := by
      intro T hT
      have hTpos : 0 < (T : ℝ) := by exact_mod_cast hT
      rw [Real.rpow_def_of_pos hTpos, mul_comm]
    have hev : Filter.EventuallyEq atTop
        (fun T : ℕ ↦ Real.exp ((1 - κ) * Real.log (T : ℝ)))
        (fun T : ℕ ↦ (T : ℝ) ^ (1 - κ)) := by
      simp only [Filter.EventuallyEq, Filter.eventually_atTop]
      exact ⟨1, fun T hT => (key T hT).symm⟩
    exact hexp.congr' hev
  have hmid : Tendsto (fun T : ℕ ↦ C * (Δ : ℝ) * (T : ℝ) ^ (1 - κ)) atTop (nhds 0) := by
    simpa [mul_zero] using hpow.const_mul (C * (Δ : ℝ))
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hmid
    ((Filter.eventually_ge_atTop 0).mono fun T _ => mul_nonneg hC
      (Finset.sum_nonneg fun t _ => Real.rpow_nonneg
        (show (0 : ℝ) ≤ 1 + (t : ℝ) from by positivity) (-κ)))
    ((Filter.eventually_ge_atTop 1).mono fun T hT => decadeMassPow_le hC hκ hT Δ)

/-! ## Theorem 3 — the comparator ordering -/

/-- **Theorem 3 (`scale_invariant_dominates_ssmax_on_decade_mass`).** The
    scale-invariant schedule keeps every decade window above the positive
    T-independent floor (`decadeMass_ge_floor`), while both position-
    decaying comparator families collapse their decade mass to zero: the
    ALiBi exponential shape and the SSMax dilution polynomial shape. SSMax
    and ALiBi fail the Def-3.1 Θ(1) band; the schedule dominates on decade
    mass. -/
theorem scale_invariant_dominates_ssmax_on_decade_mass
    (τ : ℝ) (hτ : 0 < τ) (Δ : ℕ) (hΔ : 2 ≤ Δ)
    -- the ALiBi-shaped comparator family (any exponential decay)
    (C₁ q : ℝ) (hC₁ : 0 < C₁) (hq : 0 < q ∧ q < 1)
    -- the dilution-shaped comparator family (any polynomial decay steeper than 1/t)
    (C₂ κ : ℝ) (hC₂ : 0 < C₂) (hκ : 1 < κ) :
    (∀ T : ℕ, 0 < T → decadeMassFloor (Δ : ℝ) τ ≤ decadeMass T Δ τ)
      ∧ Tendsto (fun T : ℕ ↦ decadeMassExp C₁ q T Δ) atTop (nhds 0)
      ∧ Tendsto (fun T : ℕ ↦ decadeMassPow C₂ κ T Δ) atTop (nhds 0) := by
  exact ⟨fun T hT => decadeMass_ge_floor τ hτ T Δ hT (by omega),
    decadeMassExp_tendsto_zero hC₁.le hq.1 hq.2 Δ,
    decadeMassPow_tendsto_zero hC₂.le hκ Δ⟩

end KatgptProof.ScaleInvariant
