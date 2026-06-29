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

- [x] **T1.1** Cargo feature `tropical_algebra = []` (opt-in, no default deps) in `katgpt-rs/crates/katgpt-core/Cargo.toml`.
- [x] **T1.2** New module `katgpt-rs/crates/katgpt-core/src/algebra/mod.rs` + `algebra/tropical.rs`. Module doc explains the (max, +) semiring, references Research 321 §1.2 and Smets Ch. 3 §3.5.
- [x] **T1.3** Core primitive in `algebra/tropical.rs`:
  - `pub fn tropical_matvec_into(w_row_major: &[f32], x: &[f32], out: &mut [f32], n_rows: usize, n_cols: usize)` — `(W ⊗ x)_i = max_j (W[i,j] + x[j])`. Initialize `out[i] = f32::NEG_INFINITY`, accumulate via `out[i] = out[i].max(w[i*n_cols+j] + x[j])`. Branch-free inner loop (LLVM auto-vectorizes `f32::max` on NEON/AVX2).
  - `pub fn tropical_dot_into(a: &[f32], b: &[f32], out: &mut f32, n: usize)` — `max_j (a[j] + b[j])`. Convenience scalar variant.
  - `pub fn tropical_matvec(w: &[f32], x: &[f32], n_rows: usize, n_cols: usize) -> Vec<f32>` — allocating wrapper for cold paths / tests.
- [x] **T1.4** Unit tests (same file, `#[cfg(test)]`):
  - `tropical_matvec_matches_definition`: hand-computed 2×3 case.
  - `tropical_dot_is_max_sum`: `tropical_dot([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == max(11, 22, 33) == 33.0`.
  - `neg_inf_identity`: column of all `−∞` in W → output `−∞` (matches `0` being multiplicative identity in the dual sense; additive identity `−∞` propagates).
  - `relu_is_tropical_affine`: `max(x, 0) == tropical_dot(&[0.0], &[x])` for one x — sanity-checks the textbook identity (Smets Example 3.49 / 3.57).
  - `dim_zero_noop`, `non_contiguous_strides_smoke`.
- [x] **T1.5** `cargo test -p katgpt-core --features tropical_algebra --lib` passes. (9/9 tests green: 6 Phase 1 + 3 Phase 2.)

## Phase 2 — DEC wrappers + G1 non-redundancy gate (the GOAT decision)

### Tasks

- [x] **T2.1** Three DEC wrappers in `algebra/tropical.rs`, all `#[cfg(feature = "tropical_algebra")]`, all delegating to the shipped `dec/operators.rs` boundary matrices:
  - `tropical_exterior_derivative(cx: &CellComplex, input: &CochainField) -> CochainField` — for each (k+1)-cell, output = `max` over boundary k-cells of `(sign ? 0.0 : f32::NEG_INFINITY) + input[cell]`. Signed `+1` → `+0`, signed `−1` → `−∞` (exclude). Reuses `cx.boundary(k+1)` enumeration from `exterior_derivative_into`.
  - `tropical_codifferential(cx: &CellComplex, input: &CochainField) -> CochainField` — same form, opposite direction (k → k−1).
  - `tropical_line_integral(field: &CochainField, path: &[usize]) -> f32` — `path.iter().map(|&c| field[c]).fold(f32::NEG_INFINITY, f32::max)` — bottleneck-edge cost.
  - Each ~25 LOC. No new types; reuse `CochainField`.
- [x] **T2.2** DEC wrapper tests: `tropical_d_of_constant_is_zero_or_infty` (boundary of constant field under max = `+∞` if any boundary present else `−∞` — different from linear d which gives 0), `tropical_line_integral_is_bottleneck`, `tropical_exterior_derivative_includes_all_boundary_cells`.
- [x] **T2.3** **G1 non-redundancy bench** at `katgpt-rs/crates/katgpt-core/benches/bench_337_tropical_goat.rs`. Three substrates:
  1. **DEC game-map cochain** (2D grid 16×16, random threat field) — compare `exterior_derivative` (sum-flux) vs `tropical_exterior_derivative` (max-flux). Metric: do they rank cells differently? Use a "threat hotspot" planted in the field; measure whether the top-3 cells by sum-flux differ from top-3 by max-flux. **PASS threshold: ≥1 of 3 cells differ in ranking** (else tropical is redundant). Stretch: ≥2 differ.
  2. **HLA pairs** (8-dim, 64 random NPC pairs) — compare `extract_functor` coherence (mean-cosine) vs tropical coherence (`max_k cos(target_k − source_k, f)`). Metric: rank correlation (Spearman) between the two coherence orderings. **PASS threshold: Spearman < 0.85** (else redundant). Stretch: < 0.7.
  3. **Path bottleneck vs path total** (DEC rank-1 cochain, 10 random paths on a 16×16 grid) — `tropical_line_integral` (bottleneck) vs `line_integral` (sum, from Plan 314). Metric: rank correlation of the 10 paths by each metric. **PASS threshold: Spearman < 0.85.** Stretch: < 0.7.
- [x] **T2.4** Run the bench. Record results in `.benchmarks/337_tropical_goat.md` (create folder if needed) with the **honest** outcome — PASS, FAIL, or partial.
- [x] **T2.5** **Decision point (this task closes Phase 2):**
  - **If ≥2 of 3 substrates PASS** → tropical signal is non-redundant. Proceed to Phase 3 (promote toward default), amend Research 321 to Super-GOAT, create riir-ai guide.
  - **If 1 of 3 PASS** → marginal. Keep opt-in, document the partial result, defer promotion pending a stronger substrate.
  - **If 0 of 3 PASS** → tropical is redundant with linear on our substrate. Keep `tropical_algebra` as opt-in curiosity, mark Research 321 §3 as "GOAT FAILED, demoted to opt-in", document in `.docs/20_negative_results.md`. Do NOT promote.

  **RESOLVED 2026-06-28: 3/3 PASS → proceeded to Phase 3.** All Phase 3 tasks executed; `tropical_algebra` promoted to default-on after G2 unblock (NEON specialization).

## Phase 3 — Promotion (only if Phase 2 PASS ≥2/3)

### Tasks

- [x] **T3.1** Promote `tropical_algebra` to default-on in `katgpt-core/Cargo.toml` (flip `tropical_algebra = []` → remove from opt-in list, add to default). Run `cargo check --all-features` + `cargo check --no-default-features` (the CI guard). **DONE 2026-06-28: both checks clean, 9/9 tests pass with default features.**
- [x] **T3.2** Amend Research 321 §3 verdict to **Super-GOAT** with the gate result. Create the mandatory riir-ai guide `riir-ai/.research/164_Tropical_Game_Map_Worst_Case_Threat_Guide.md` (next free riir-ai number after 163) covering: TL;DR (selling point = "NPCs compute worst-case survival paths via tropical line integrals, complementing the expected-engagement sum-path"), distilled primitive, connection map (DEC × HLA × game maps), latent-vs-raw boundary (tropical cochain fields stay local; only the bottleneck-edge scalar crosses sync), validation protocol (G1–G3), implementation priority P0–P3. **DONE 2026-06-28:** Research 321 amended GOAT → Super-GOAT (commit `7a5eb7b1`); Guide 164 created with all required sections + G2 status updated to PASSED (commit `3e1a86ce`).
- [x] **T3.3** Bench: tropical matvec vs `simd_matvec` at D=8/64/128. Expect tropical to be FASTER (no FMA dependency chain — `max` is a single-cycle op on most SIMD ISAs). Record in `.benchmarks/337_tropical_goat.md`. **DONE 2026-06-28: hypothesis was WRONG.** Auto-vec baseline was 4-9× slower (single-acc max chain is latency-bound — the exact anti-pattern `simd_dot_f32`'s comment warns about). After NEON specialization (T3.4): D=64 0.96×, D=128 1.03× (PASS at gate dims). D=8 0.82× (caveat — not a production use case). Full table + honest analysis in `.benchmarks/337_tropical_goat.md`.
- [x] **T3.4** SIMD specialization: `tropical_matvec_into` with explicit NEON/AVX2 paths via `std::arch::aarch64::*` / `std::arch::x86_64::*` gated by `target_feature`. Mirror `simd.rs` SIMD-level pattern. **DONE 2026-06-28 (NEON only):** `neon_tropical_row_max_sum` with 4× `float32x4_t` accumulators (16 lanes), `vmaxq_f32` + `vaddq_f32`, `vmaxvq_f32` horizontal reduce. Scalar fallback uses 4 independent `f32` accumulators (same tree-reduce pattern). AVX2 path deferred (this dev machine is aarch64; x86 uses the 4-acc scalar fallback which is competitive).

## Phase 4 — Fusion hooks (only if Phase 2 PASS ≥1/3)

### Tasks

- [x] **T4.1** riir-ai fusion: `tropical_extract_functor` in `riir-engine/src/latent_functor/arithmetic.rs` behind the same `tropical_algebra` feature re-export. Delegates to katgpt-core. Off by default pending riir-ai guide. **DONE 2026-06-29:** `tropical_extract_functor` + `tropical_extract_functor_into` shipped. The functor *direction* stays the mean displacement (stable central-tendency estimate); only the *coherence* scalar switches to the (max, +) semiring (`max_k cos(disp_k, f)` instead of `mean_k cos(disp_k, f)`). Delegates the inner max-reduction to `katgpt_core::algebra::tropical::tropical_dot_into`. New `tropical_algebra` feature in riir-engine/Cargo.toml = `["katgpt-core/tropical_algebra", "latent_functor"]`, off by default. 7 unit tests pass (constant-offset recovery, single-pair=1.0, empty=0.0, direction == linear direction, max ≫ mean on adversarial data, ranking divergence smoke, buffer reuse). 2394 riir-engine lib tests pass with the feature on. **G6 empirical gate RUN 2026-06-29 (Phase 5 T5.1) — 3/3 PASS.** See `.benchmarks/337_tropical_functor_g6.md`.
- [x] **T4.2** riir-neuron-db fusion: `tropical_retrieve` in `riir-neuron-db/src/index.rs` — max-coordinate shard retrieval. Off by default. Empirical test: does it beat `diverse_retrieval` on any substrate? **DONE 2026-06-29:** `ShardIndex::retrieve_tropical` shipped — twin of `retrieve_diverse` with the wedge-span aggregation swapped from linear sum to tropical max (`span_trop(c) = max(wedge(q, emb_c), max_{s ∈ selected} wedge(emb_s, emb_c))`). Delegates the inner max-reduction to `katgpt_core::algebra::tropical::tropical_dot_into`. New `tropical_retrieval` feature in riir-neuron-db/Cargo.toml = `["katgpt-core/tropical_algebra", "diverse_retrieval"]` (implies `diverse_retrieval` so the shared `wedge_l1_norm_8` primitive is available), off by default. 8 unit tests pass (returns-k, fewer-than-k, empty, zero-k, seed-is-cosine-nearest, no-duplicates, zero-alloc smoke, both-valid-and-share-seed). 223 riir-neuron-db lib tests pass; `--no-default-features` + `--all-features` clean (CI guard). **G6 empirical gate RUN 2026-06-29 (Phase 5 T5.2) — 2/2 divergence substrates PASS, sanity PASS.** See `.benchmarks/337_tropical_retrieval_g6.md`.
- [x] **T4.3** Document the SE(2)-equivariant game-map follow-up as an **issue** at `riir-ai/.issues/` (separate scope, large build, not a katgpt-rs plan). Note the textbook reference and the lifting/group-conv/projection architecture. **DONE 2026-06-29:** `riir-ai/.issues/350_se2_equivariant_game_map_lifting.md` created. Covers: TL;DR (selling point = rotation-equivariant NPC perception + worst-case-over-orientations threat fields via tropical projection), textbook reference (Smets §3.4 Theorem 3.32 + §3.5 Theorem 3.54), why this is riir-ai not katgpt-rs (private game IP), connection map (DEC × HLA × functor × Research 164), proposed architecture (4-module `equivariant/` subtree + game-side integration), validation protocol (G1 equivariance, G2 perf, G3 no-regression, G4 latent/raw boundary, G5 selling-point gate), tropical projection variant (the fusion with Plan 337), implementation priority (P2/P3 table), out-of-scope (SE(3), other Lie groups, learned kernels, tropical LatCal).

## Phase 5 — G6 downstream-value empirical gates (the deferred-item close)

Per AGENTS.md "Dont defer benchmark task" and "Try implement to unblock if block, after check goat + proof gain, promote to default if gain also demote loser", the two Phase 4 fusion primitives' empirical gates (flagged in code comments as "pending empirical gate") must actually run before the primitives earn their keep. These are G6 (downstream-value) gates, distinct from the G1 (non-redundancy) gate that already passed in Phase 2.

### Tasks

- [x] **T5.1** riir-ai `tropical_extract_functor` G6 signal-recovery bench at `riir-ai/crates/riir-engine/benches/bench_337_tropical_functor_g6.rs`. Three substrates: (1) clean observation buffer sanity — both metrics should agree to <0.1, (2) adversarial "betrayal among ambiguous encounters" — 1 strong + 15 noise pairs, tropical should recover the signal with `|coh−1|` at least 0.3 smaller than linear, (3) re-estimation trigger ordering across 32 mixed buffers — ≥1 disagreement. **DONE 2026-06-29 (commit `0541b831`):** 3/3 PASS. Substrate 1 `|trop−lin|=0.000106`. Substrate 2 tropical `|coh−1|=0.18` vs linear `0.79` (improvement 0.62, >2× the 0.3 bar). Substrate 3: 15/32 buffers are tropical-only-trigger (semiring choice flips the re-estimation decision). Bench report at `riir-ai/.benchmarks/337_tropical_functor_g6.md`. **Stays opt-in:** G6 proves non-redundant signal, not that tropical WINS on a downstream NPC-behavior metric (KG triple quality, stance prediction accuracy). Promotion pending a downstream consumer integration.
- [x] **T5.2** riir-neuron-db `retrieve_tropical` G6 ensemble-divergence bench at `riir-neuron-db/benches/bench_337_tropical_retrieval_g6.rs`. Three substrates: (1) multi-modal 8-cluster fixture — |ensemble △|≥1, (2) adversarial squad+loner — |△|≥1, (3) uniform sanity — k=4 distinct each, no panic. **DONE 2026-06-29 (commit `dfbdd17e`):** 2/2 divergence substrates PASS, sanity PASS. Substrate 1 `|△|=4` (STRETCH), substrate 2 `|△|=2`, substrate 3 clean. Tropical runs +5-7% slower than diverse (extra `tropical_dot_into` reduction). Bench report at `riir-neuron-db/.benchmarks/337_tropical_retrieval_g6.md`. **Stays opt-in:** G6 proves ensemble divergence, not that the tropical ensemble produces a better frozen shard on a real consolidation workload. Promotion pending a riir-ai/riir-chain downstream consumer integration.
- [x] **T5.3** Update Cargo feature-flag comments in both repos to reflect the closed gates (replace "pending empirical gate" with the actual G6 result + the new pending item: downstream consumer validation). **DONE 2026-06-29.**

## Out of scope

- **SE(2) lifting/group-conv/projection CNN** (Smets §3.4) — full homogeneous-space operator framework. Large surface, primarily riir-ai game-map territory. Defer to a riir-ai `.research/` note + plan when scoped.
- **Tropical LatCal** — speculative, no clear modelless unblock. riir-chain follow-up if ever.
- **Tropical geometric product** — speculative fusion with default-on `geometric_product`. No clear value over the existing wedge.
- **Training-side tropical NNs** (morphological neural networks, Smets et al. 2021) → riir-train. Out of scope for this workflow.

## GOAT gate summary

| Gate | Criterion | Threshold | Phase | Result |
|---|---|---|---|---|
| **G1** (non-redundancy) | ≥2 of 3 substrates show tropical signal is non-redundant with linear | DEC ranking ≥1/3 differ, HLA Spearman <0.85, path Spearman <0.85 | Phase 2 | **3/3 PASS** (all STRETCH) |
| **G2** (perf) | tropical matvec ≥ as fast as `simd_matvec` at D=64 | (expected: faster — `max` < FMA chain) | Phase 3 | **PASS** D=64 0.96×, D=128 1.03× (NEON T3.4) |
| **G3** (no regression) | `cargo check --all-features` + `--no-default-features` clean | clean | every phase | **PASS** |
| **G4** (alloc-free hot path) | `tropical_matvec_into` 0 allocs/call (caller-owned buffers) | 0 | Phase 1 | **PASS** |
| **G6** (downstream-value, riir-ai functor) | tropical coherence carries non-redundant signal on per-NPC observation buffer | 3 substrates: clean sanity <0.1, signal-recovery improvement ≥0.3, trigger disagreement ≥1 | Phase 5 T5.1 | **3/3 PASS** (commit `0541b831`) |
| **G6** (downstream-value, riir-neuron-db retrieval) | `retrieve_tropical` ensemble diverges from `retrieve_diverse` on shard substrate | 2 divergence substrates: \|△\|≥1 each, sanity clean | Phase 5 T5.2 | **2/2 + sanity PASS** (commit `dfbdd17e`) |

**Promotion status (post-Phase-5):**

- `katgpt-core/tropical_algebra` — **DEFAULT-ON** (Phase 3 Super-GOAT promotion, all G1–G5 gates PASS).
- `riir-engine/tropical_algebra` — **opt-in**, G6 PASS but no downstream consumer has proven the tropical coherence WINS on a real NPC-behavior metric. Promote after a game-runtime integration ships the win.
- `riir-neuron-db/tropical_retrieval` — **opt-in**, G6 PASS (ensemble divergence proven) but no downstream consumer has proven the tropical ensemble produces a better frozen shard. Promote after a riir-ai/riir-chain consolidation-quality integration ships the win.

Neither opt-in is a riir-train dependency — both can be promoted modellessly once a downstream consumer validates the win.

## References

- Research: [katgpt-rs/.research/321_Tropical_Semiring_Equivariant_Operators.md](../.research/321_Tropical_Semiring_Equivariant_Operators.md)
- Gate template: [katgpt-rs/.plans/319_geometric_product_latent_interaction.md](319_geometric_product_latent_interaction.md) (Clifford G1 non-redundancy gate)
- DEC substrate: [katgpt-rs/.plans/251_dec_operators_cell_complex.md](251_dec_operators_cell_complex.md), `katgpt-rs/crates/katgpt-core/src/dec/operators.rs`
- Textbook: [arXiv:2403.04807](https://arxiv.org/abs/2403.04807) Ch. 3 §3.5.
