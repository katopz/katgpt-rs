/-
! Root module for KatgptProof.

Re-exports all proof modules. Add new modules here.

This is the **second Lean 4 formal-verification instance** in the 5-repo
quintet (katgpt-rs / riir-ai / riir-chain / riir-neuron-db / riir-train),
and the **first in the public MIT repo** (`katgpt-rs`). The first instance is
`riir-chain/.proofs/RiirChainProof` (Plan 004 — LatCal fixed-point round-trip).

Whereas `RiirChainProof` proves a *sync-boundary* property (integer
round-trip exactness, Mathlib-free, decided by `omega`), `KatgptProof` proves a
*ranking-preservation* property (sigmoid strict monotonicity, which requires
Mathlib's transcendental analysis of `exp`).
-/

import KatgptProof.Bridge.Basic
import KatgptProof.Bridge.RankingPreserved
import KatgptProof.Bridge.SpecTests
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
import KatgptProof.Pencil.SpecTests
import KatgptProof.Ssmax.SpecTests
import KatgptProof.HintRegret.Basic
import KatgptProof.HintRegret.SpecTests
import KatgptProof.ScaleInvariant.Basic
import KatgptProof.ScaleInvariant.Sigmoid
import KatgptProof.ScaleInvariant.Dominates
import KatgptProof.ScaleInvariant.SpecTests
