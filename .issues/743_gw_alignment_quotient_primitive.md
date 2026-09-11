# Issue 743: `gw_alignment` — Gromov–Wasserstein quotient-alignment primitive for katgpt-core

**Status:** RESOLVED — LANDED 2026-09-11 (Plan 594, Bench 709, commit ref in git log): `katgpt_core::gw_alignment` behind opt-in `gw_alignment` (multi-start greedy/softmin/uniform/perm-init + product-graph power iteration, sum-exact loss, deterministic, zero steady-state alloc). G1–G4 ALL PASS; stays opt-in (consumer-first rule — riir-poc consumer PoC is the promotion path). Measured evolution vs the sketch: greedy second-order init is load-bearing (uniform-start power iteration is saddle-blind and glacial); 2-opt polish on ΣP removed (walked 0.0016 → 0.145 — ΣP is not the GW objective).
**Source:** [riir-ai Research 371](../../riir-ai/.research/371_Principal_Bundle_Qualia_Quotient_Orbit_Decomposition.md) (Oizumi/Lim/Kanai principal-bundle qualia framework, verified vs full text) — Issue 912 T4's consume-vs-build decision landed on BUILD. Canonical method: Mémoli 2011 (Gromov–Wasserstein distances); entropic formulation: Peyré/Cuturi/Solomon 2016 (ICML, "Gromov-Wasserstein Averaging of Kernel and Distance Matrices"). External references only: Oizumi-lab GWTune (Takeda 2025, J Neurosci Methods 419:110443) + POT — both Python, both research-grade, both stay OUT of the runtime.
**Kind:** new modelless primitive (opt-in feature) — cross-space structural alignment from distance matrices alone, no shared coordinates, no training.

## Why

riir-ai Issue 912 T3 measured the quotient/orbit factorization's within-family guarantees as exact (64/64 identity survivors, reconstruction RMS-L2 = 0.0). The open capability is **cross-NPC structural comparison**: "do these two NPCs experience this zone the same way" — comparing subjective quotient geometries with NO shared coordinates. The paper's two-stage protocol (verified in Research 371): stage 1 orbits topologically (persistence / GWOT), stage 2 quotients metrically (RSA / GWOT at fine vs coarse granularity).

What ships today covers only part: `mag::transfer::Wasserstein1d` compares 1D distributions (pointwise ground cost — NOT structure-only); RSA is trivially available (distance-vector correlation — riir-poc `quotient_orbit_poc` already computes it). **GW is the missing piece**: it aligns two distance matrices using only intra-space structure — exactly the no-shared-coordinates case.

## The primitive (proposal sketch)

- Module: `katgpt_core::gw_alignment`, feature `gw_alignment` (opt-in, default-off; zero new deps — pure std).
- Input: two precomputed square distance matrices `D_A ∈ R^{n×n}`, `D_B ∈ R^{m×m}` (n, m ≤ 64 — quotient probe sets), uniform weights.
- Output: the GW transport plan's self-similarity loss (lower = more structurally similar) + a sigmoid-bounded alignment score `sigmoid(−β·loss)` per the house bridge rule. The coupling plan itself stays local (never crosses sync).
- Method: exact conditional-gradient (Peyré alg. 1) at n ≤ 64 via the product-graph power iteration; preallocated scratch, zero steady-state alloc; deterministic (fixed init, fixed iteration count — replay-safe).
- Latent-only: the SCORE may feed social KG triples (encounters/relationships from quotient-space proximity — sanctioned by the domain rules); the matrices and plan never cross the sync boundary.

## Gates (GOAT sketch — for the plan to price)

- G1 correctness: analytic small cases (n = m = 2/3 with known optimal coupling; isometric matrices ⇒ loss ≈ 0; scrambled isometric ⇒ loss 0 with permuted plan) vs a brute-force permutation check at n = 4.
- G2 discriminability: planted-alignment sweep on synthetic quotient geometries — GW score must separate planted vs shuffled alignments with an AUC bar, AND be more robust than the RSA baseline under rotation-type distortions where pointwise correspondence fails (that is GW's reason to exist).
- G3 no-regression: default-feature build compiles to nothing new; workspace benches hold.
- G4 alloc-free: `TrackingAllocator` zero-alloc in steady state (crate convention).

## Explicitly out of scope

- Fused GW (needs shared-feature cost matrices — we have none across NPCs).
- Large-n entropic variants (n ≤ 64 exact is enough for probe sets; revisit if a consumer needs more).
- TDA/persistent-homology for the stage-1 orbit comparison — NOT this issue; the katgpt-dec TDA lane (Betti0 union-find over DEC boundaries, in flight) is the consumption path when it lands.

## Downstream

A plan opens in THIS repo (primitive + gates + benches). riir-ai Issue 912 T4's evaluation conclusion is recorded there; the consumer PoC (zone-belief quotient alignment → social KG triples) belongs in riir-poc AFTER this primitive lands — routing per the research skill's repo rule (katgpt-rs = upstream modelless primitives, no riir deps).

## Numbering

`.issues/.highwater` read 742 at write time (2026-09-10) → this file takes **743**; highwater bumped in the same commit.
