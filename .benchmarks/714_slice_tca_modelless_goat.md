# GOAT Proof 714: sliceTCA — Modelless Slice-Rank Decomposition (Plan 596)

> **Date:** 2026-09-12
> **Feature Gate:** `slice_tca` (katgpt-core; `slice_tca = ["subspace_phase_gate", "tucker_factorization"]`) — **DEFAULT-ON since 2026-09-12**
> **Source:** Pellegrino, Stein & Çayko-Gajic, *Nat Neurosci* 27, 1199–1210 (2024), doi:10.1038/s41593-024-01626-2 — sliceTCA; Research 309 (private half)
> **Verdict:** 🟢 **GOAT PASS (G1–G4)**. **PROMOTED to default-on 2026-09-12** — the evidence below is the promotion basis.

## Summary

Deterministic, zero-hyperparameter slice-rank decomposition for 3rd-order tensors `X[n,t,k]` (entity × time × episode): three single-class truncated-SVD factorizations (Eckart–Young optima via the paper's own single-slice-type ≡ unfolding-matrix-factorization reduction), a closed-form covariability classifier (top-R spectra of the three unfoldings over a shared `‖X‖²_F` denominator + **sigmoid** routing — never softmax, the classes are not mutually exclusive), a joint deterministic-ALS demixer with loading-only block updates (one tensor contraction per component per sweep; slice matrices frozen at birth so class assignment cannot drift), and canonicalization (unit-slice normalization, largest-|·|-entry-positive sign rule with first-index tie-break, variance-desc sort with lexicographic bit-pattern tie-break).

Delta vs the paper's SGD fitter: a **deterministic pure-function contract** (same input bytes → bit-identical factors on the same triple/codegen, BLAKE3-replayable) and zero-hyperparameter block updates (no LR/schedule/masking curriculum). **Fitting novelty is NOT claimed** (ALS lineage: Carroll & Chang 1970; Harshman 1970; Orth-ALS arXiv:1703.01804 — cited in module docs).

**G1 headline:** 12/12 (100%) pure-class routing across the noise sweep 0→0.2; on the two-class mixture `[64,128,32]`, joint fit loss **0.0263** vs **0.4835** for BOTH naive floors (majority-class and best-per-unfolding single-SVD at equal budget) — **18.4× better**, with the routed class set `{entity, time}` asserted directly (lossy-surface rule honored: the class split did not flip).

## Surface

| Item | Location |
|---|---|
| `SliceDecomposition` (+ `reconstruct_into`, `entity_slice_into`, `canonical_hash`, `canonicalize`) | `katgpt-core/src/slice_tca/types.rs` |
| `SliceTcaScratch`, `SliceTcaConfig`, `InitMode`, `SliceClass`, `Tensor3`, `SliceTcaError` | `katgpt-core/src/slice_tca/types.rs` |
| `covariability_shares[_into]`, `route[_default]`, `fit_single_class_into` | `katgpt-core/src/slice_tca/svd.rs` |
| `fit_slice_into`, `fit_with_ranks_into`, `reallocate_class`, `relative_loss` | `katgpt-core/src/slice_tca/als.rs` |
| `select_ranks_knee`, `select_ranks_blocked_cv` | `katgpt-core/src/slice_tca/rank.rs` |
| Module docs (slice-rank definition, class taxonomy, ALS derivation, citations, determinism contract) | `katgpt-core/src/slice_tca/mod.rs` |
| In-module tests (20, incl. BLAKE3 determinism ×16 + rebuild, monotone ALS, class-pass invariance, zero-alloc) | `katgpt-core/src/slice_tca/tests.rs` |
| GOAT gate bench | `katgpt-core/benches/bench_596_slice_tca_goat.rs` |

Substrate consumed (no new deps): `subspace_phase_gate::{thin_svd_into, SvdScratch, SvdResultScratch, numerical_rank}` (every spectrum/basis factors through the small-side Gram `G_σ = X_(σ)X_(σ)ᵀ`), `linalg::tucker::{tucker_decompose_into, TuckerScratch, TuckerResultScratch}` (**first in-tree Tucker consumer** — HOSVD joint initializer for shapes inside its `SVD_MAX_RANK=16` bound, automatic Gram-SVD fallback otherwise), `simd::{simd_gram_f32, simd_dot_f32, simd_fused_scale_acc}`, `crate::sigmoid`, `blake3`, `fastrand` (fixtures only).

## Test Configuration

| Parameter | Value |
|---|---|
| Machine | M3 Max (aarch64), macOS — **box load 25–30 during measurement** (concurrent sibling builds; see Caveats) |
| Profile | `cargo bench` (release/bench) |
| Pure-class fixtures | `[48,48,48]`, 2 generic-M components, noise ∈ {0, 0.05, 0.1, 0.2}, seeds 100–111 |
| Mixture fixtures | `[64,128,32]`, entity(2) + time(2), noise ∈ {0, 0.05, 0.1, 0.2}, seed 7 |
| Plants | `loading ⊗ M` with unit-norm loadings and generic unit-norm slice matrices (the class signature lives in the slices' genericity) |
| Latency gates | min-of-N per call (least-contended sample; mean also reported) |

## Gate Results

### G1 — class recovery + naive floors (PASS)

- **Pure-class routing** (shares top-2, θ=0.25, α=20): **12/12 exact** across all three classes × four noise levels. Example margins (noise 0.05): own-share 0.998 vs cross-shares 0.104–0.116.
- **Mixture floors** (`[64,128,32]`, noise 0.05, equal total budget 4):
  - joint slice fit: **0.0263** rel loss
  - floor 1 (majority-class single-class fit @4): **0.4835**
  - floor 2 (best per-unfolding single-SVD fit @4): **0.4835**
  - routed classes `{entity, time}` — exactly the planted set (asserted).
- **Noise sweep** (mixture, full pipeline, ranks auto-selected `[2,2,0]` at every level): loss 0.0240 / 0.0263 / 0.0331 / 0.0596 at noise 0 / 0.05 / 0.10 / 0.20.

### G2 — ALS ≥ SVD-init + latency (PASS)

- **Monotonicity:** per-sweep losses `[0.0305, 0.0263, 0.0263, …]` — strictly non-increasing; ALS improves the SVD init by 13.8% on the mixture fixture (asserted, not hoped: the pre-ALS init loss is recorded as element 0).
- **Full-fit latency** `[64,128,32]` (shares → route → knee → init → 8 ALS sweeps → canonicalize): mean 30.17 ms, **min 29.99 ms** over 32 runs (gate: min ≤ 50 ms). Phase split: shares ≈ 27 ms (3 Grams + 3 one-sided-Jacobi SVDs, dominated by the 128×128 mode-1 Gram SVD), fit continuation ≈ 3 ms.
- **Per-entity slice** (t×k = 4096 outputs):
  - canonical 2-component (single-class design point): mean 0.444 µs, **min 0.375 µs** over 1024 (gate: min ≤ 1.0 µs) → PASS.
  - full 4-component mixture (reported ungated): mean 1.114 µs, min 1.041 µs — see Caveats.

### G3 — determinism + invariance (PASS)

- **BLAKE3 factor hash** identical across 16 fits (fresh scratch every 4th call, warm otherwise) + a full drop-and-rebuild.
- **Class-pass invariance:** the same rank-1 tensor expressed as an Entity vs a Time component → identical canonical weights, reconstruction agreement 1.19e-7.
- **Canonical order + sign rule** verified on fitted decompositions (weights descending; largest-|·| loading entry positive).

### G4 — alloc-free hot paths (PASS)

0 allocations over 100 steady-state calls each (global CountingAllocator, liveness canary asserted first):
`covariability_shares_into` + `route_default`: **0**; `reconstruct_into`: **0**; `entity_slice_into`: **0**. (The lib-side in-module gate additionally uses the crate's `TrackingAllocator` under `any(debug_assertions, feature = "alloc_tracking")` — never debug-only, per Issue 741.)

### G3 (suites) — cross-checks

`cargo test -p katgpt-core --features slice_tca --lib`: **2012 passed, 0 failed** (includes the 20 slice_tca tests, run by name-filter to prove non-vacuity). `--no-default-features --features slice_tca --lib`: **20/20** (feature isolation clean). `cargo check --target wasm32-unknown-unknown --features slice_tca --lib`: clean (pure math + std; no platform APIs, no `std::time` in src/). `cargo clippy --features slice_tca --all-targets -- -D warnings`: clean.

## Honest Caveats

1. **`ROUTE_THETA = 0.25` is calibrated, not proven.** It is calibrated on balanced fixtures (`[48,48,48]`, `[64,128,32]`) whose complementary-axis spectral spread sits below θ. A pure class-σ plant's cross-axis share is the Marchenko–Pastur top-R concentration of its generic slice Grams, ≈ `R·(1+√aspect)²/min_dim`; thin-axis shapes (e.g. k=8 with t≥24) push cross-shares ABOVE θ and false-positive route. The classifier is correct on the calibrated shape family; the boundary is documented in the module docs and tests.
2. **The opt-in blocked selector is NOT held-out prediction CV.** True held-out CV is structurally impossible for slice models: every axis indexes free slice parameters (an entity-class slice has one free column per episode; an episode-class loading one free entry per episode), so a model fit on some episodes predicts ≈0 on held episodes at EVERY rank — the CV surface is flat and under-selects by tie-break (measured: `[1,1,0]` vs planted `[2,2,0]`). Masked fitting would restore a real CV surface but is a hyperparameter curriculum the plan's determinism design excludes. The shipped selector is a **blocked structural-fit plateau grid** (per-block fits, smallest ranks within an ε-plateau of the best mean block residual); it agrees with the knee on clean mixtures (test-pinned).
3. **The 4-component mixture entity-slice sits at ~1.04–1.11 µs** — above the 1 µs line. Four components × 4096 outputs with 4 sequential passes have a ~1 µs L1-store-traffic floor on this core; the sub-µs gate holds at the path's design point (2 components, 0.375 µs min). A fused multi-component kernel could likely bring 4 comps under 1 µs if a consumer needs it (not done — complexity without a consumer).
4. **The Gram reduction squares the condition number.** Kept components lose ~half the digits of the tiny ones; components below the `ENERGY_FLOOR_TAU = 0.05` absolute floor are trimmed anyway, and the paper's own SGD fitter is far less accurate. The HOSVD path (small shapes) avoids the squaring entirely (direct unfolding SVD inside Tucker).
5. **HOSVD init auto-fallback:** `TuckerConfig` enforces per-mode `SVD_MAX_RANK=16`; shapes like `[64,128,32]` (mode-1 min-dim 128) take the Gram-SVD initializer. Recorded per the plan's T1.4 instruction.
6. **Latency numbers measured under box load 25–30** (concurrent sibling-agent builds; mean shares phase swung 27→85 ms across runs). Gates use min-of-N per call — the least-contended sample ≈ the primitive's true cost; means are reported alongside and are load-inflated. Re-run on a quiet box before quoting external numbers.
7. **NMF-variant out of scope** (no closed-form block update; projected-ALS would be a follow-up plan).
8. **`reallocate_class` on a generic slice is lossy by construction** (rank-1 projection of the slice matrix); the returned residual ratio quantifies it and the degenerate case is test-pinned, not hidden.
