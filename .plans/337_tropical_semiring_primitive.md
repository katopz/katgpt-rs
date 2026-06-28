# Plan 337: Tropical Semiring Primitive + G1 Non-Redundancy Gate

**Date:** 2026-06-28
**Research:** [katgpt-rs/.research/321_Tropical_Semiring_Equivariant_Operators.md](../.research/321_Tropical_Semiring_Equivariant_Operators.md)
**Source paper:** [arXiv:2403.04807](https://arxiv.org/abs/2403.04807) — Smets, *Mathematics of Neural Networks*, Ch. 3 §3.5 (tropical operators).
**Target:** `katgpt-rs/crates/katgpt-core/src/algebra/tropical.rs` (new module) + Cargo feature `tropical_algebra`
**Status:** Active — Phase 1 (skeleton), gate-pending.

---

## Goal

Ship the `(max, +)` tropical semiring as a modelless inference primitive: `tropical_matvec_into` (the `(max, +)` analog of `simd_matvec`) plus three thin wrappers over the shipped DEC substrate (`tropical_exterior_derivative`, `tropical_codifferential`, `tropical_line_integral`). Run the **G1 non-redundancy gate** (mirroring the Clifford Plan 319 pattern): *does the tropical signal carry information that the linear `(ℝ, +, ·)` signal misses on a representative substrate?* If yes by a clear margin AND a product selling point emerges → promote `tropical_algebra` toward default-on and amend Research 321 to **Super-GOAT** (creating the mandatory riir-ai guide at that point). If no → keep opt-in as a curiosity primitive, document the negative result.

**Why GOAT not Super-GOAT upfront (per Research 321 §3):** unlike Clifford's wedge (mathematically orthogonal to the dot product by construction), the tropical max is NOT mathematically orthogonal to the sum — they are different aggregations of the same data. Non-redundancy is an empirical question. The honest path is ship-and-gate, not pre-commit.

## Phase 1 — Skeleton (CORE)

### Tasks

- [ ] **T1.1** Cargo feature `tropical_algebra = []` (opt-in, no default deps) in `katgpt-rs/crates/katgpt-core/Cargo.toml`.
- [ ] **T1.2** New module `katgpt-rs/crates/katgpt-core/src/algebra/mod.rs` + `algebra/tropical.rs`. Module doc explains the (max, +) semiring, references Research 321 §1.2 and Smets Ch. 3 §3.5.
- [ ] **T1.3** Core primitive in `algebra/tropical.rs`:
  - `pub fn tropical_matvec_into(w_row_major: &[f32], x: &[f32], out: &mut [f32], n_rows: usize, n_cols: usize)` — `(W ⊗ x)_i = max_j (W[i,j] + x[j])`. Initialize `out[i] = f32::NEG_INFINITY`, accumulate via `out[i] = out[i].max(w[i*n_cols+j] + x[j])`. Branch-free inner loop (LLVM auto-vectorizes `f32::max` on NEON/AVX2).
  - `pub fn tropical_dot_into(a: &[f32], b: &[f32], out: &mut f32, n: usize)` — `max_j (a[j] + b[j])`. Convenience scalar variant.
  - `pub fn tropical_matvec(w: &[f32], x: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32>` — allocating wrapper for cold paths / tests.
- [ ] **T1.4** Unit tests (same file, `#[cfg(test)]`):
  - `tropical_matvec_matches_definition`: hand-computed 2×3 case.
  - `tropical_dot_is_max_sum`: `tropical_dot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == max(11, 22, 33) == 33.0`.
  - `neg_inf_identity`: column of all `−∞` in W → output `−∞` (matches `0` being multiplicative identity in the dual sense; additive identity `−∞` propagates).
  - `relu_is_tropical_affine`: `max(x, 0) == tropical_dot(&[0.0], &[x])` for one x — sanity-checks the textbook identity (Smets Example 3.49 / 3.57).
  - `dim_zero_noop`, `non_contiguous_strides_smoke`.
- [ ] **T1.5** `cargo test -p katgpt-core --features tropical_algebra --lib` passes.

## Phase 2 — DEC wrappers + G1 non-redundancy gate (the GOAT decision)

### Tasks

- [ ] **T2.1** Three DEC wrappers in `algebra/tropical.rs`, all `#[cfg(feature = "tropical_algebra")]`, all delegating to the shipped `dec/operators.rs` boundary matrices:
  - `tropical_exterior_derivative(cx: &CellComplex, input: &CochainField) -> CochainField` — for each (k+1)-cell, output = `max` over boundary k-cells of `(sign ? 0.0 : f32::NEG_INFINITY) + input[cell]`. Signed `+1` → `+0`, signed `−1` → `−∞` (exclude). Reuses `cx.boundary(k+1)` enumeration from `exterior_derivative_into`.
  - `tropical_codifferential(cx: &CellComplex, input: &CochainField) -> CochainField` — same form, opposite direction (k → k−1).
  - `tropical_line_integral(field: &CochainField, path: &[usize]) -> f32` — `path.iter().map(|&c| field[c]).fold(f32::NEG_INFINITY, f32::max)` — bottleneck-edge cost.
  - Each ~25 LOC. No new types; reuse `CochainField`.
- [ ] **T2.2** DEC wrapper tests: `tropical_d_of_constant_is_zero_or_infty` (boundary of constant field under max = `+∞` if any boundary present else `−∞` — different from linear d which gives 0), `tropical_line_integral_is_bottleneck`, `tropical_exterior_derivative_includes_all_boundary_cells`.
- [ ] **T2.3** **G1 non-redundancy bench** at `katgpt-rs/crates/katgpt-core/benches/bench_337_tropical_goat.rs`. Three substrates:
  1. **DEC game-map cochain** (2D grid 16×16, random threat field) — compare `exterior_derivative` (sum-flux) vs `tropical_exterior_derivative` (max-flux). Metric: do they rank cells differently? Use a "threat hotspot" planted in the field; measure whether the top-3 cells by sum-flux differ from top-3 by max-flux. **PASS threshold: ≥1 of 3 cells differ in ranking** (else tropical is redundant). Stretch: ≥2 differ.
  2. **HLA pairs** (8-dim, 64 random NPC pairs) — compare `extract_functor` coherence (mean-cosine) vs tropical coherence (`max_k cos(target_k − source_k, f)`). Metric: rank correlation (Spearman) between the two coherence orderings. **PASS threshold: Spearman < 0.85** (else redundant). Stretch: < 0.7.
  3. **Path bottleneck vs path total** (DEC rank-1 cochain, 10 random paths on a 16×16 grid) — `tropical_line_integral` (bottleneck) vs `line_integral` (sum, from Plan 314). Metric: rank correlation of the 10 paths by each metric. **PASS threshold: Spearman < 0.85.** Stretch: < 0.7.
- [ ] **T2.4** Run the bench. Record results in `.benchmarks/337_tropical_goat.md` (create folder if needed) with the **honest** outcome — PASS, FAIL, or partial.
- [ ] **T2.5** **Decision point (this task closes Phase 2):**
  - **If ≥2 of 3 substrates PASS** → tropical signal is non-redundant. Proceed to Phase 3 (promote toward default), amend Research 321 to Super-GOAT, create riir-ai guide.
  - **If 1 of 3 PASS** → marginal. Keep opt-in, document the partial result, defer promotion pending a stronger substrate.
  - **If 0 of 3 PASS** → tropical is redundant with linear on our substrate. Keep `tropical_algebra` as opt-in curiosity, mark Research 321 §3 as "GOAT FAILED, demoted to opt-in", document in `.docs/20_negative_results.md`. Do NOT promote.

## Phase 3 — Promotion (only if Phase 2 PASS ≥2/3)

### Tasks

- [ ] **T3.1** Promote `tropical_algebra` to default-on in `katgpt-core/Cargo.toml` (flip `tropical_algebra = []` → remove from opt-in list, add to default). Run `cargo check --all-features` + `cargo check --no-default-features` (the CI guard).
- [ ] **T3.2** Amend Research 321 §3 verdict to **Super-GOAT** with the gate result. Create the mandatory riir-ai guide `riir-ai/.research/164_Tropical_Game_Map_Worst_Case_Threat_Guide.md` (next free riir-ai number after 163) covering: TL;DR (selling point = "NPCs compute worst-case survival paths via tropical line integrals, complementing the expected-engagement sum-path"), distilled primitive, connection map (DEC × HLA × game maps), latent-vs-raw boundary (tropical cochain fields stay local; only the bottleneck-edge scalar crosses sync), validation protocol (G1–G3), implementation priority P0–P3.
- [ ] **T3.3** Bench: tropical matvec vs `simd_matvec` at D=8/64/128. Expect tropical to be FASTER (no FMA dependency chain — `max` is a single-cycle op on most SIMD ISAs). Record in `.benchmarks/337_tropical_goat.md`.
- [ ] **T3.4** SIMD specialization: `tropical_matvec_into` with explicit NEON/AVX2 paths via `std::arch::aarch64::*` / `std::arch::x86_64::*` gated by `target_feature`. Mirror `simd.rs` SIMD-level pattern.

## Phase 4 — Fusion hooks (only if Phase 2 PASS ≥1/3)

### Tasks

- [ ] **T4.1** riir-ai fusion: `tropical_extract_functor` in `riir-engine/src/latent_functor/arithmetic.rs` behind the same `tropical_algebra` feature re-export. Delegates to katgpt-core. Off by default pending riir-ai guide.
- [ ] **T4.2** riir-neuron-db fusion: `tropical_retrieve` in `riir-neuron-db/src/index.rs` — max-coordinate shard retrieval. Off by default. Empirical test: does it beat `diverse_retrieval` on any substrate?
- [ ] **T4.3** Document the SE(2)-equivariant game-map follow-up as an **issue** at `riir-ai/.issues/` (separate scope, large build, not a katgpt-rs plan). Note the textbook reference and the lifting/group-conv/projection architecture.

## Out of scope

- **SE(2) lifting/group-conv/projection CNN** (Smets §3.4) — full homogeneous-space operator framework. Large surface, primarily riir-ai game-map territory. Defer to a riir-ai `.research/` note + plan when scoped.
- **Tropical LatCal** — speculative, no clear modelless unblock. riir-chain follow-up if ever.
- **Tropical geometric product** — speculative fusion with default-on `geometric_product`. No clear value over the existing wedge.
- **Training-side tropical NNs** (morphological neural networks, Smets et al. 2021) → riir-train. Out of scope for this workflow.

## GOAT gate summary

| Gate | Criterion | Threshold | Phase |
|---|---|---|---|
| **G1 (non-redundancy)** | ≥2 of 3 substrates show tropical signal is non-redundant with linear | DEC ranking ≥1/3 differ, HLA Spearman <0.85, path Spearman <0.85 | Phase 2 |
| **G2 (perf)** | tropical matvec ≥ as fast as `simd_matvec` at D=64 | (expected: faster — `max` < FMA chain) | Phase 3 |
| **G3 (no regression)** | `cargo check --all-features` + `--no-default-features` clean | clean | every phase |
| **G4 (alloc-free hot path)** | `tropical_matvec_into` 0 allocs/call (caller-owned buffers) | 0 | Phase 1 |

## References

- Research: [katgpt-rs/.research/321_Tropical_Semiring_Equivariant_Operators.md](../.research/321_Tropical_Semiring_Equivariant_Operators.md)
- Gate template: [katgpt-rs/.plans/319_geometric_product_latent_interaction.md](319_geometric_product_latent_interaction.md) (Clifford G1 non-redundancy gate)
- DEC substrate: [katgpt-rs/.plans/251_dec_operators_cell_complex.md](251_dec_operators_cell_complex.md), `katgpt-rs/crates/katgpt-core/src/dec/operators.rs`
- Textbook: [arXiv:2403.04807](https://arxiv.org/abs/2403.04807) Ch. 3 §3.5.
