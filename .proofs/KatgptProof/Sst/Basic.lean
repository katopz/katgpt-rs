/-
! Spec for the SST composition-order law (Issue 958; arXiv:2610.22293).

The Rust substrate `crates/katgpt-core/src/sst_certified.rs` certifies
weight-delta shipping forms against a protected subspace. The law this
module pins is the paper's **elementwise-transform leak condition**, in the
rank-1 (diagonal) case — the case the issue scopes:

    A ⊙ X  =  diag(a) · X · diag(b)

Zero protected-block leak holds iff each protected span is invariant under
its own diagonal — **with the clause that the opposite diagonal does not
vanish on the complement** (a vanishing diagonal can mask a violation: in
the two-sided form `Y = D_a X D_b`, a `b ≡ 0` annihilates every output, so
a leaking `D_a` reads as zero leak — the measured quantity is blind, the
violation is not gone).

We model one side of the law over `Fin n → ℝ` with Mathlib's
`Submodule.span`:

- `DiagMap a x = fun i => a i * x i` — the elementwise transform's action on
  one factor (mirrors `diag(a) · x` for vectors; the matrix form composes
  the two one-sided laws).
- `NoLeak S a` — the protected span `S` is invariant under `DiagMap a`:
  `∀ x ∈ S, DiagMap a x ∈ S`.

The two SHIPPED arms (exactly two, per the issue's scope — the general iff
over arbitrary non-rank-1 transforms is ill-posed as a single statement and
out of scope):

1. `noLeak_of_scalar` + `level_set_aligned_no_leak` — a scalar diagonal
   (`a = const c`) preserves EVERY subspace (`D x = c • x`): the "scalar
   diagonals are the trivial case" clause. The level-set instance makes the
   arm concrete: the protected coordinate plane of `Fin 3 → ℝ` under
   `diag(2, 2, 3)` — the diagonal takes ONE value on the protected
   coordinates, so the plane is invariant.
2. `two_by_two_diagonal_leaks` — the explicit 2×2 counterexample (the
   perturbation arm's negative): the protected span `span{![1, 1]}` under
   `diag(1, 0)` maps the protected vector `![1, 1]` to `![1, 0]`, which
   leaves the span — the complement component is measurable and nonzero.
   This is the Rust mirror battery's exact instance (the planted-leak
   refusal + the 2×2 arm in `sst_certified`'s tests).

The vanishing-diagonal clause above is WHY arm 2 picks `a = ![1, 0]` and
reads the leak off the complement component rather than through a vanishing
diagonal: `a = ![0, 0]` would map every vector to zero, and zero is in
every submodule — the violation of span-invariance would be masked by the
annihilation. The counterexample keeps the complement diagonal alive.
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.LinearAlgebra.Span.Basic

namespace KatgptProof.Sst

/-- The elementwise (rank-1 diagonal) transform's action on one factor:
`x ↦ diag(a) · x`. Mirrors the Rust delta-transform shape `A ⊙ X`
restricted to one side — the two-sided form composes two of these.
(`noncomputable`: real arithmetic over `ℝ`.) -/
noncomputable def DiagMap {n : ℕ} (a : Fin n → ℝ) (x : Fin n → ℝ) : Fin n → ℝ :=
  fun i => a i * x i

/-- **No-leak condition:** the protected span `S` is invariant under the
diagonal map — every protected vector's transform stays protected. This is
the paper's "protected span splits along the level sets of the diagonal"
condition, stated as invariance. -/
def NoLeak {n : ℕ} (S : Submodule ℝ (Fin n → ℝ)) (a : Fin n → ℝ) : Prop :=
  ∀ x ∈ S, DiagMap a x ∈ S

/-- Scalar diagonal is a scalar multiple of the identity map on vectors. -/
lemma diagMap_const {n : ℕ} (c : ℝ) (x : Fin n → ℝ) :
    DiagMap (fun _ => c) x = c • x := by
  funext i
  simp [DiagMap, smul_apply]

/-- The diagonal map is linear over real linear combinations (the one-line
algebra every span membership argument reduces through). -/
lemma diagMap_smul_add {n : ℕ} (a x y : Fin n → ℝ) (p q : ℝ) :
    DiagMap a (p • x + q • y) = p • DiagMap a x + q • DiagMap a y := by
  funext i
  simp only [DiagMap, Pi.add_apply, Pi.smul_apply, mul_add]
  ring

/-! ## Arm 1 — scalar / level-set-aligned diagonals leak nothing

A scalar diagonal transforms every vector by a scalar multiplication, and no
subspace is closed under leaving through scalar multiplication. The concrete
level-set instance: the protected coordinate plane of `Fin 3 → ℝ` under
`diag(2, 2, 3)` — the diagonal is constant (2) across the protected
coordinates' level set.
-/

/-- **The scalar arm (the trivial case).** A scalar diagonal preserves every
protected span — no level-set structure is needed. -/
theorem noLeak_of_scalar {n : ℕ} (S : Submodule ℝ (Fin n → ℝ)) (c : ℝ) :
    NoLeak S (fun _ => c) := by
  intro x hx
  rw [diagMap_const]
  exact S.smul_mem c hx

/-- **The level-set-aligned instance.** `S = span{e₀, e₁}` (the protected
coordinate plane in `Fin 3 → ℝ`) is invariant under `a = ![2, 2, 3]`: the
diagonal takes ONE value (2) on the protected coordinates, so a protected
vector's transform stays in the plane. This is the arm the paper's law
names: the protected span splits along the level sets of the diagonal. -/
theorem level_set_aligned_no_leak :
    NoLeak (Submodule.span ℝ ({![(1 : ℝ), 0, 0], ![(0 : ℝ), 1, 0]} : Set (Fin 3 → ℝ)))
      ![(2 : ℝ), 2, 3] := by
  intro x hx
  set S : Submodule ℝ (Fin 3 → ℝ) :=
    Submodule.span ℝ ({![(1 : ℝ), 0, 0], ![(0 : ℝ), 1, 0]} : Set (Fin 3 → ℝ)) with hS
  rw [Submodule.mem_span_pair] at hx
  obtain ⟨p, q, hx⟩ := hx
  subst hx
  rw [diagMap_smul_add]
  have h0 : DiagMap ![(2 : ℝ), 2, 3] ![(1 : ℝ), 0, 0] = (2 : ℝ) • ![(1 : ℝ), 0, 0] := by
    funext i
    fin_cases i <;> simp [DiagMap]
  have h1 : DiagMap ![(2 : ℝ), 2, 3] ![(0 : ℝ), 1, 0] = (2 : ℝ) • ![(0 : ℝ), 1, 0] := by
    funext i
    fin_cases i <;> simp [DiagMap]
  rw [h0, h1]
  refine Submodule.add_mem S ?_ ?_
  · exact Submodule.smul_mem S p
      (Submodule.smul_mem S 2 (Submodule.subset_span (Set.mem_insert _ _)))
  · exact Submodule.smul_mem S q
      (Submodule.smul_mem S 2
        (Submodule.subset_span (Set.mem_insert_of_mem _ (Set.mem_singleton _))))

/-! ## Arm 2 — the explicit 2×2 counterexample

`n = 2`, `S = span{![1, 1]}` (the protected direction is the 45° diagonal),
`a = ![1, 0]` — the complement coordinate's diagonal entry is ZERO. The
protected vector `u = ![1, 1]` transforms to `![1, 0]`, which is NOT in the
span: its complement component is measurable and nonzero (the leak the
certifier measures). This is the arm that pins the mask clause: the
complement diagonal here is 0 ON THE VECTOR'S COORDINATE but the map still
escapes the span — while a FULLY vanishing diagonal (`![0, 0]`) would have
annihilated the vector into the span and masked the violation. The reading
off the complement component (not through the vanishing entry) is the
issue's clause in operation.
-/

/-- The counterexample's protected span: the 45° diagonal of `Fin 2 → ℝ`. -/
noncomputable def Sst2 : Submodule ℝ (Fin 2 → ℝ) := Submodule.span ℝ ({![(1 : ℝ), 1]} : Set (Fin 2 → ℝ))

/-- The counterexample's diagonal: identity on the protected coordinate,
zero on the complement. -/
noncomputable def A2 : Fin 2 → ℝ := ![(1 : ℝ), 0]

/-- The protected vector transforms to `![1, 0]`. -/
lemma diag_map_a2_one_one : DiagMap A2 ![(1 : ℝ), 1] = ![(1 : ℝ), 0] := by
  funext i
  fin_cases i <;> simp [DiagMap, A2] <;> norm_num

/-- `![1, 1]` is in the protected span (trivially — it spans it). -/
lemma one_one_mem : ![(1 : ℝ), 1] ∈ Sst2 :=
  Submodule.mem_span_singleton_self ![(1 : ℝ), 1]

/-- **The 2×2 leak (the perturbation arm's negative).** The protected vector
`![1, 1]` is in the span, but its transform `![1, 0]` is NOT — the protected
span is NOT invariant under `A2`. Equivalently: `¬ NoLeak Sst2 A2`. -/
theorem two_by_two_diagonal_leaks : ¬ NoLeak Sst2 A2 := by
  intro noleak
  have hout : DiagMap A2 ![(1 : ℝ), 1] ∈ Sst2 := noleak _ one_one_mem
  rw [diag_map_a2_one_one] at hout
  unfold Sst2 at hout
  rw [Submodule.mem_span_singleton] at hout
  obtain ⟨r, hr⟩ := hout
  -- Coordinate 1: r · 1 = 0 → r = 0. Coordinate 0: r · 1 = 1 → 0 = 1. Absurd.
  have h1 : r = 0 := by
    have hc := congrFun hr 1
    simp at hc
    exact hc
  have hc0 : r = 1 := by
    have hc := congrFun hr 0
    simp at hc
    exact hc
  exact absurd (h1.symm.trans hc0) (by norm_num)

/-- **The leak, measured (the certifier's read).** The complement component
of the transformed protected vector is `-1 · e₁`'s coefficient: the second
coordinate of `DiagMap A2 u` is `0` while the SPAN member any rank-`r`
multiple of `![1, 1]` would need `r = 0` there and `r = 1` at the first —
the unresolvable coordinate is exactly the leak magnitude the Rust
certifier reports in `SstRefusal::LeakAboveTolerance` (measured, refused —
never a bare no). -/
lemma leak_complement_component : (DiagMap A2 ![(1 : ℝ), 1]) 1 = 0 ∧
    (DiagMap A2 ![(1 : ℝ), 1]) 0 = 1 := by
  constructor <;> · norm_num [DiagMap, A2]

end KatgptProof.Sst
