# Plan 629 — Band-selective cochain filters (the open primitive behind Research 628)

**Status:** Active — open primitive, modelless (fixed/authored coefficients; trained coefficients are riir-ai Issue 1073 T6's trigger-gated lane, never here). Super-GOAT denied (ChebNet published prior art); GOAT-tier fusion.
**Research:** [katgpt-rs/.research/628_Spectral_Neuro_Symbolic_GSP.md](../.research/628_Spectral_Neuro_Symbolic_GSP.md) (paper: arXiv:2508.14923)
**Consumer:** riir-ai Issue 1100 (band reads over the route-avoid danger field; Issue 1097 enrichment) — the consumer's T0 replay gates any wiring, NOT this plan's T1–T3.
**Date:** 2026-10-20

---

## Goal

Land the generic, game-free band-filter primitive in `katgpt-dec`: K-th order Chebyshev polynomial responses over the DEC Laplacian, applied without eigendecomposition, with coefficient-merge rule composition. Public tier (generic math, zero game semantics — the `terrain_cochains` eggshell rule holds: no terrain/threat/game vocabulary in this file). Feature-gated, benched, GOAT-gated; promote to default only on pass.

**Consumer gate (round-1 verdict fix):** T1+ proceed ONLY after riir-ai Issue 1100 T0 confirms the band-read need — and Issue 1100 T0 runs on the SHIPPED substrate (DoG energies via `heat_kernel_trajectory_krylov`, two or three scales), NOT on this plan. If T0's hypothesis dies, this plan is HELD at T0 — never built speculatively; a band filter without a confirmed consumer is a parallel version of the heat-kernel code the substrate-first rule forbids.

## Non-goals

- No eigendecomposition reads on mutating graphs (design rule: |Tₖ|≤1 on [−1,1] ⇒ polynomial response bounded by ‖θ‖₁ under edge perturbation; eigen reads stay reserved for static complexes).
- No game consumers in this repo (riir-ai Issue 1100 owns them).
- No training (Issue 1073 T6 owns the trigger-gated trainer).
- No dim > 3 claims (curse-of-dimensionality law: boundary-vs-volume and band reads are d≤3 wins).

## Tasks

- [ ] **T0** Scoped `boundary-guard` run before the module lands (new file in katgpt-dec); confirm no BOUNDARY.md row needed beyond the existing katgpt-dec owns-row. Record verdict here.
- [ ] **T1** `crates/katgpt-dec/src/band_filter.rs` behind feature `band_filter` (opt-in, NOT default):
  - `ChebBandFilter<const N: usize> { coeffs: [f32; N], lambda_max: f32 }` where **N = order + 1** (round-1 fix: `[f32; K + 1]` would need unstable `generic_const_exprs` on the pinned 1.98.1 toolchain; N is plain). Authored coefficients; `lambda_max` from **Gershgorin on the combinatorial Laplacian: λ_max ≤ 2·max weighted degree** (8 on the 5-point `grid_2d` stencil) — round-1 fix: "max degree" alone under-estimates, the scaled spectrum leaves [−1,1], and the recurrence blows up. Over-estimation only rescales the frequency axis (safe); under-estimation is the instability.
  - `apply_into(&self, complex, x: &CochainField, y: &mut CochainField, scratch: &mut [CochainField; 2])` — three-term recurrence Tₖ(L̃)x = 2L̃·Tₖ₋₁x − Tₖ₋₂x, matvec via the existing hodge/graph Laplacian `_into` path; rank-0 fields first.
  - `energy_into(&self, complex, x, out: &mut f32, scratch)` — the band-energy readout ⟨y, y⟩ (localization mass; the `hodge_energy` twin at band resolution).
  - Zero-alloc on all `_into` paths; fixed-size coefficient arrays per the bounded-domain law.
  - `required-features` row for the new test target in the SAME commit (repo-birth gate discipline).
- [ ] **T2** Coefficient merge (the paper's Eq 11, which is a coefficient sum): `merge(parts: &[(ChebBandFilter, f32)]) -> ChebBandFilter` — θ_total = Σ wᵣ θᵣ with the summation order PINNED (Issue-940 determinism law: sequential fold, fixed order, documented tie rule). **Round-1 fix — the G1 pin form:** the MERGED path is the CANONICAL result; bit-pin REPLAY of the merged path only (Issue 940 requires same-path determinism, never cross-path agreement — Σₖ(Σᵣwᵣθᵣₖ)Tₖx and ΣᵣwᵣΣₖθᵣₖTₖx associate differently and legitimately round differently in f32). The per-rule-then-sum comparison is a TOLERANCE check (bit-exactness on dyadic coefficients is NARROWED per round-2: power-of-two coefficients on small-integer fields make the products exact but the k-summation orders can still round differently — so the dyadic fixture asserts "expected identical, asserted on THAT fixture", and it is pinned to power-of-two coefficients over small-integer fields).
- [ ] **T3** Gates (all `--release`, box-state provenance per the latency law):
  - G1 correctness: partition-of-unity fixture (authored LP + HP coefficient pairs summing to identity reproduce the input within fp tolerance); two-scale fixture field separates peak (high-band) from plateau (low-band); merged-path replay bit-exactness (T2's canonical-path form); **λ_max bound check against the `graph_laplacian_into` grid-stencil path specifically** (round-1 fix: Plan 357 G5 gave it a separate code path — `operators.rs:294` — and the bound must be checked on the path that actually runs, plus the generic edge-list path).
  - G2 perf: **≥1000 samples** (round-1 fix: the percentile-index law — below that the "p99" index lands on the MAX), measured **separately for the grid-stencil fast path and the generic edge-list path** (round-1 fix: they differ by the G5 margin; pooling hides which one the bar measures): p99 bar on a 128² complex (16k vertices) at order 4 — sub-100 µs per apply; assert, don't assume.
  - G3 no-regression: θ=δ₀ (identity template) is byte-identical passthrough; feature-off compiles to nothing; **SUBSTRATE-FIRST BASELINE (round-1 fix): the Chebyshev filter must beat DoG-via-`heat_kernel_trajectory_krylov` (the shipped constructible band-pass) on matvec count or band selectivity at `--release` — if it beats neither, it is a parallel heat-kernel version and does NOT land; record the negative and keep the DoG path.**
  - G4 alloc: zero-alloc on apply/energy/merge (counting allocator, canary-armed).
  - Perturbation test: add/remove k edges → band reads move ≤ C·‖θ‖₁·(local magnitude); document C from the run (demonstrates the bound; never gates on eig instability).
- [ ] **T4** Module docs credit the lineage (Shuman 2013 GSP; Hammond 2011 Chebyshev-on-graphs; Defferrard 2016 ChebNet; arXiv:2508.14923 for the rules-as-band framing) + one generic example (synthetic complex, no game vocabulary). Cross-ref Research 628 + riir-ai Issue 1100.
- [ ] **T5** GOAT verdict recorded here; promote `band_filter` to default ONLY on G1–G4 pass (modelless gain — authored coefficients are the deliverable; trained ones stay in riir-train's lane).

## Deferred

- [-] Eigen-basis band reads via `hodge_decompose` on static complexes (useful for arena geometry analytics; no consumer pulls; the Poisson-solve cost needs its own bench first).
- [-] Rank-1/rank-2 cochain filters (threat-per-edge, occupancy-per-face) — rank-0 first; the Hodge Laplacian's rank-dependent structure needs its own gate.

## GOAT gate rule

Every promotion through G1–G4 at `--release`; demote to opt-in on any fail; the trained-coefficient lane (1073 T6) may not promote anything here — it rides the same forward.
