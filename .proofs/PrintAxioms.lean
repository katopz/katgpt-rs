/-
! Axiom inventory printer.

Run with: `lake env lean PrintAxioms.lean` (or, as an assertion rather than a
printout, `scripts/proof_gate.sh`).

Prints the axiom dependencies of every shipped theorem. The axiom budget is
`{propext, Classical.choice, Quot.sound}` — the standard Lean foundation, and
the same one Mathlib itself stands on. No `sorry`, no `sorryAx`, no
`Lean.ofReduceBool` (which `native_decide` would introduce, trading the kernel
for the compiler). If this printer ever shows a theorem depending on anything
outside that budget, the proof is invalid.

`KatgptProof` depends on Mathlib (`Real.exp` analysis for sigmoid strict
monotonicity, `Filter.Tendsto` for the asymptotics), so `Classical.choice` and
`Quot.sound` are expected here — unlike the Mathlib-free
`riir-neuron-db/.proofs`, where most theorems come out axiom-free.

Uses `open` + unqualified names because fully-qualified `#print axioms` through
the root import does not reliably resolve transitively-imported theorem names
in Lean 4 (a known quirk); the unqualified form via `open <Namespace>` works.

Adding a theorem? Add its `#print axioms` line here, then bump
`EXPECTED_THEOREMS` in `scripts/proof_gate.sh`. The count is pinned there on
purpose: deriving it from this file would be self-referential, so deleting a
line would shrink both the expectation and the result together and the check
could never fire.
-/

import KatgptProof.Bridge.Basic
import KatgptProof.Bridge.RankingPreserved
import KatgptProof.Hope.Basic
import KatgptProof.Hope.SpecTests
import KatgptProof.Ssmax.Basic
import KatgptProof.Ssmax.DilutionBound
import KatgptProof.Ssmax.Asymptotic
import KatgptProof.Pencil.Sym
import KatgptProof.Pencil.RayleighCF
import KatgptProof.Pencil.Weyl
import KatgptProof.Pencil.Loewner
import KatgptProof.Pencil.Eigengap
import KatgptProof.HintRegret.Basic
import KatgptProof.ScaleInvariant.Tendsto
import KatgptProof.ScaleInvariant.Basic
import KatgptProof.ScaleInvariant.Sigmoid
import KatgptProof.ScaleInvariant.Dominates

open KatgptProof.Bridge
open KatgptProof.Hope
open KatgptProof.Ssmax
open KatgptProof.Pencil
open KatgptProof.HintRegret
open KatgptProof.ScaleInvariant

-- Action-bridge ranking preservation (the sigmoid strict-monotonicity chain).
#print axioms action_bridge_ranking_preserved
#print axioms action_bridge_ranking_preserved'
#print axioms action_bridge_argmax_preserved

-- HOPE spec instances (normal CDF at 0, ReLU self-kernel positivity).
#print axioms normalCdf_zero
#print axioms reluSelfKernel_pos_gamma_zero

-- Ssmax core: the alphaGold denominator is positive and alphaGold is bounded.
#print axioms alphaGold_denom_pos
#print axioms alphaGold_bounded
#print axioms strictMono_rpow_of_gt_one

-- Ssmax dilution bound: strict monotonicity in the context length `c`.
#print axioms alphaGold_strictMono_in_c
#print axioms alphaGold_lt_of_c_lt
#print axioms ssmax_dominates_base

-- Ssmax asymptotics: leakage → 0 and alphaGold → 1 as `c → ∞`.
#print axioms alphaGold_eq
#print axioms leakage_nonneg
#print axioms leakage_le_inv
#print axioms eventually_c_ge_two
#print axioms tendsto_leakage_zero
#print axioms tendsto_alphaGold_one

-- Pencil sym packing (Issue 678 T1): the isometry.
#print axioms sym_isometry_norm_sq
#print axioms sym_isometry_dot
#print axioms sym_isometry_norm

-- Pencil Rayleigh/Courant–Fischer core (Issue 678, substrate for T2–T4).
#print axioms eigval_antitone
#print axioms ray_eq_sum
#print axioms inner_self_eq_sum
#print axioms ray_le_top
#print axioms cf_ge
#print axioms cf_dual

-- Pencil Weyl (Issue 678 T2): 1-Lipschitz eigenvalues.
#print axioms weyl_one_sided
#print axioms eigval_zero_le_opNorm
#print axioms weyl_lipschitz

-- Pencil Loewner + mirror (Issue 678 T3).
#print axioms loewner_mono
#print axioms mirror_dual

-- Pencil eigengap (Issue 678 T4): shift lemma + the analytic core + the
-- ladder pinning + the final assembly.
#print axioms eigval_add_smul_one
#print axioms ray_diagonal
#print axioms opNorm_diagonal_le
#print axioms eigengap_ge_half
#print axioms eigval_diagonal_antitone
#print axioms ladder_unit_gap
#print axioms eigengap_ladder_ge_half

-- HintRegret (Plan 576): band-gate openness over ℝ — the ideal (0, 1)
-- contract the f32 gate approximates (its ±40 early-exit / rounding
-- saturation points are the documented Rust-vs-ℝ divergence).
#print axioms bandGate_mem_Ioo

-- ScaleInvariant (Plan 622 Phase 6): the schedule skeleton + the Lean trio.
#print axioms m_add_aSq
#print axioms perKeyMass_eq_alpha
#print axioms scale_invariant_decade_mass_bounded
#print axioms decadeMass_le_logDelta
#print axioms decadeMassFloor_pos
#print axioms decadeMass_ge_floor
#print axioms sigmoid_tilt_pointwise
#print axioms sigmoid_transfer_halves_constant
#print axioms sigmoid_transfer_gap
#print axioms decadeMassExp_le
#print axioms decadeMassExp_tendsto_zero
#print axioms decadeMassPow_le
#print axioms decadeMassPow_tendsto_zero
#print axioms scale_invariant_dominates_ssmax_on_decade_mass
