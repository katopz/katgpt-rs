# Plan 596: Modelless Slice-Rank Decomposition Primitive

**Status:** Active — Phase 1 not started
**Date:** 2026-09-11
**Research:** [riir-neuron-db/.research/309_slice_tca_covariability_class_consolidation.md](../riir-neuron-db/.research/309_slice_tca_covariability_class_consolidation.md) (private; this plan carries only the public generic-math half)
**Source paper:** Pellegrino, Stein & Cayco-Gajic, *Nat Neurosci* 27, 1199–1210 (2024), doi:10.1038/s41593-024-01626-2 — sliceTCA
**Target:** `crates/katgpt-core/src/slice_tca/` (new module) + Cargo feature `slice_tca` (opt-in)
**Boundary note:** pure linear algebra — no game/chain/shard semantics. Private consumers wire it separately (riir-neuron-db Plan 329, riir-train Plan 397).

---

## Goal

Ship a deterministic, modelless low-slice-rank tensor decomposition for 3rd-order tensors X[n, t, k] (entity × time × episode): three single-class truncated-SVD factorizations (Eckart–Young global optima), a closed-form covariability classifier (3 spectra over a shared Frobenius denominator + sigmoid routing), a joint deterministic-ALS demixer with SVD init, and invariance canonicalization (SVD orthogonalize + variance ordering + sign/tie rules). The delta vs the paper's fitter: **deterministic pure-function contract** (same input bytes → bit-identical factors on the same triple/codegen; BLAKE3-replayable) and zero-hyperparameter block updates (contraction + scalar divide — no LR/schedule/masking curriculum). Fitting novelty is NOT claimed (ALS lineage: Carroll & Chang 1970, Harshman 1970, Orth-ALS NeurIPS 2017 arXiv:1703.01804 — cited in module docs).

GOAT gate: G1 synthetic class recovery + invariance fixtures + monotone loss; G2 ≥ truncated-SVD-per-unfolding at equal budget; G3 suites green; G4 alloc-free + determinism repeat-pin.

## Phase 1 — Core module (feature `slice_tca`)

### Tasks

- [ ] **T1.1** `mod.rs`: `SliceDecomposition { R: (R_n, R_t, R_k), loadings, slices }` fixed-layout, const-generic bounded sizes; reconstruction `reconstruct_into(&self, x_hat: &mut Tensor3)` zero-alloc (rank-1 GER accumulation, chunk-8 fixed order).
- [ ] **T1.2** Single-class path: `fit_single_class(X, axis, R)` = truncated SVD of the unfolding via existing `thin_svd_into`/power-iteration (spectral substrate); document Eckart–Young optimality (paper's own reduction: single-slice-type ≡ unfolding matrix factorization).
- [ ] **T1.3** Classifier: `covariability_shares(X, R) -> [f32; 3]` — top-R singular spectra of the three unfoldings over shared `‖X‖²_F`; `route(share, θ, α) = sigmoid(α·(EVR−θ))` (sigmoid, never softmax — classes not mutually exclusive).
- [ ] **T1.4** Joint ALS: block update per component = one tensor contraction + scalar divide (ε-floored); components born with their class (class fixed at construction — between-class invariance cannot drift under ALS); fixed iteration count, fixed order; SVD-init greedy by EVR. Consume `linalg/tucker.rs` HOSVD as the joint initializer (gives Tucker its first consumer) or record in this plan why deflation-init won.
- [ ] **T1.5** Canonicalization (paper L3/L2): SVD orthogonalization + variance-desc stable sort `(σ desc, lexicographic)` + sign rule (largest-|·| entry positive, first-index tie-break); between-class rank-1 reallocation = closed-form projection in the generic case, documented-degenerate otherwise.
- [ ] **T1.6** Determinism contract test: BLAKE3(hash(factors)) identical across N=16 calls + a rebuild; hazards pinned (no HashMap, no rayon in fit path, no runtime-adaptive ε, fixed-order accumulation).
- [ ] **T1.7** Property tests: ALS loss monotone non-increasing; reconstruction round-trip; planted `u⊗v⊗z` passed between classes → canonical output unchanged.

## Phase 2 — Rank selection + benchmarks (GOAT gate)

### Tasks

- [ ] **T2.1** EVR-knee deterministic surrogate for (R_n,R_t,R_k) (default) + blocked-CV grid (opt-in, deterministic-parallel per grid point); synthetic surrogate-vs-CV agreement test.
- [ ] **T2.2** G1 bench: synthetic generator (pure-class / two-class mixtures / noise sweep) — class-recovery accuracy; **naive floors**: majority-class baseline AND per-unfolding-single-SVD baseline must be beaten on mixed-class data (flat aggregate error is NOT a pass if the class split flips — lossy-surface rule, Issue 750 T3).
- [ ] **T2.3** G2 bench: warm-started ALS vs SVD-only at equal budget (monotonicity ⇒ ALS ≥; assert, don't hope); latency on [64,128,32]: full fit low-ms, per-entity slice path sub-µs.
- [ ] **T2.4** G4: alloc counter behind a feature (never `debug_assertions` — Issue 741); zero-alloc on reconstruction + classifier paths.
- [ ] **T2.5** GOAT verdict: if G1–G4 pass → promote `slice_tca` to default (modelless gain, quality-gated); record verdict + demote note if a slot loser exists. Full gate: `scripts/full_gate.sh` (wasm32 layer 2b included — pure math must cross-compile).

## Phase 3 — Docs + drift hygiene

### Tasks

- [ ] **T3.1** Module docs: slice-rank definition, class taxonomy, ALS block-minimizer derivation (one contraction + divide), citations (sliceTCA; Tao & Sawin slice rank; ALS/Orth-ALS lineage), determinism contract scope (same triple + codegen; committed surface = shares/routing, degeneracy-immune).
- [ ] **T3.2** README feature-table row + `count_features.py` clean; `.benchmarks/NNN` record with honest caveats (mixed-class threshold is calibrated, not proven; NMF-variant out of scope — no closed form, projected-ALS would be a follow-up).
