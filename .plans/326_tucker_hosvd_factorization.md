# Plan 326: Tucker / HOSVD Tensor Factorization

**Date:** 2026-06-25
**Research:** [katgpt-rs/.research/307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md](../.research/307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md) (§3 candidate plan #3)
**Source paper:** [arXiv:2511.05963](https://arxiv.org/abs/2511.05963) — Duruisseaux, Kossaffi, Anandkumar, *Fourier Neural Operators: A Practical Perspective* (Caltech + NVIDIA, Nov 2025)
**Target:** `katgpt-rs/crates/katgpt-core/src/linalg/tucker.rs` (open primitive) + `riir-neuron-db/src/shard_compactor.rs` (integration), Cargo feature `tucker_factorization`
**Status:** Active — Phase 1 implementation

---

## Goal

Ship the **N-mode generalization of `thin_svd_into`** — HOSVD (Higher-Order SVD) / Tucker decomposition — as a generic open primitive in `katgpt-rs`, then wire it into the `riir-neuron-db` `ShardCompactor` as an alternative cold-tier compaction path that compactly stores a *batch* of `NeuronShard`s via a low-rank core tensor + factor matrices.

This is the third and last Gain-tier candidate plan from Research 307 (FNO practical-perspective survey). The headline FNO inference primitive (resolution-invariant spectral transport) already ships as our Super-GOAT `cross_resolution_transport` (Research 291 / Plan 310, DEFAULT-ON); this plan closes the only remaining FNO-derived modelless gap.

**Why Tucker for shards:** the existing `ShardCompactor::compact` uses Attention-Matching (AM, Plan 233), which produces `M < N` *output shards* (row-subsampled + fit). Tucker produces a *factorized representation* of the whole batch `(N, 8, 8)` — a small core `(r_N, r_R, r_C)` plus three factor matrices. Compression wins when `N` is large:

| Path | Output | Compression | Use case |
|------|--------|-------------|----------|
| `compact` (AM) | `M` shards × 64 floats | N/M ratio | Live Hot/Warm tier — produce fewer, representative shards |
| `compact_tucker` (new) | core `(r_N·r_R·r_C)` + 3 factors | depends on ranks | Cold-tier archival — store a whole zone's shard set as one factored tensor |

For a zone with N=64 shards reshaped `(64, 8, 8)` and ranks `(8, 4, 4)`: original = 4096 floats, Tucker = 128 (core) + 3·(64·8 + 8·4 + 8·4) = 128 + 1664 = 1792 floats → **2.3× compression**, lossy within the truncation budget. The AM path cannot achieve this because it always produces `Vec<NeuronShard>` — one Pod per output — paying the 64-float overhead per shard.

**Modelless:** pure closed-form linear algebra — mode-n unfoldings + thin SVD + tensor-times-matrix contractions. No gradient descent, no learned weights, no training. Promotable to default-on if the GOAT gate passes.

---

## Phase 1 — Open Primitive (katgpt-rs)

### Goal

Generic HOSVD on flat `&[f32]` + shape descriptor. Pure numeric, no shard/chain/game semantics. Reuses `thin_svd_into` from `subspace_phase_gate` — does NOT reimplement SVD.

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/linalg/tucker.rs` with module doc + feature gating
- [x] **T1.2** Implement `TuckerConfig` — inline-array shape/ranks (`[usize; MAX_MODES]` + len, no `smallvec` dep) + validation (ranks ≤ shape per mode, ≥1 mode, ≤4 modes)
- [x] **T1.3** Implement `TuckerScratch` — pre-allocated buffers for: per-mode unfolding matrix (sized for max `(I_n, prod(I_others))`), `SvdScratch` + `SvdResultScratch` (reuse from subspace_phase_gate), core assembly workspace
- [x] **T1.4** Implement mode-n unfolding: `unfold_into(tensor, shape, mode, out: &mut [f32])` — row-major `I_n × prod(other)` matrix
- [x] **T1.5** Implement `tucker_decompose_into(tensor, cfg, scratch) -> &TuckerResultScratch`: per-mode unfold → `thin_svd_into` → top-`r_n` cols of U^(n) → store factor; core = `X ×_1 U^(1)^T ×_2 U^(2)^T × … ×_N U^(N)^T`
- [x] **T1.6** Implement `tucker_reconstruct_into(core, factors, shape, out: &mut [f32])` — inverse `X̃ = S ×_1 U^(1) ×_2 U^(2) × … ×_N U^(N)`
- [x] **T1.7** Implement `TuckerResultScratch` (SOA, hot-path) + `TuckerResult::from_scratch` convenience (owned)
- [x] **T1.8** Add `tucker_factorization = ["subspace_phase_gate"]` to `Cargo.toml` features (initially OFF)
- [x] **T1.9** Register `pub mod tucker;` in `linalg/mod.rs` + re-exports in `lib.rs`
- [x] **T1.10** Unit tests: known-rank tensor recovery, orthogonality of factors, core energy bound, reconstruction error monotonic in ranks (25 tests, all PASS)

### Phase 1 GOAT gate

- [x] **T1.G1** Reconstruction quality: synthetic rank-`(2,2,2)` tensor, HOSVD with ranks=r → reconstruction rel error `< 1e-4` — **PASS via unit test `hosvd_low_rank_recovers_exact_low_rank_tensor`**
- [x] **T1.G2** Perf: `(8, 8, 8)` decompose + reconstruct mean latency ≤ 500µs — **PASS**: empirically verified **73.89µs** (1000 iters, release, direct binary launch). See `.benchmarks/326_tucker_hosvd_goat.md`.
- [x] **T1.G3** No-regression: full-rank decomposition (`r_n = I_n` ∀n) → reconstruction max abs error `< 1e-4` — **PASS via unit test** `hosvd_full_rank_is_near_lossless_3mode`
- [x] **T1.G4** Alloc-free hot path: `tucker_decompose_into` with pre-warmed `TuckerScratch` → 0 allocations / 100 calls — **PASS**: empirically verified **0 allocs** via `CountingAllocator`. See `.benchmarks/326_tucker_hosvd_goat.md`.

### Phase 1 promotion

- [x] **T1.P** **PROMOTED to DEFAULT-ON** in `katgpt-rs/crates/katgpt-core/Cargo.toml` (Phase 3, 2026-06-25). All 4 GOAT gates PASS empirically (G1 4.096e-8, G2 73.89µs, G3 1.013e-6, G4 0 allocs). Pure modelless gain (closed-form HOSVD, no training). Re-verified by direct binary launch (`bench_326_tucker_hosvd_goat` bench) — bypasses the `cargo bench` dyld/trustd stall.

---

## Phase 2 — Integration (riir-neuron-db)

### Goal

Wire the open primitive into `ShardCompactor` as an alternative cold-tier archival path.

### Tasks

- [x] **T2.1** Add `tucker_factorization` feature to `riir-neuron-db/Cargo.toml` — `tucker_factorization = ["shard_compactor", "katgpt-core/tucker_factorization"]`. No chain alias needed initially (leaf-crate-local).
- [x] **T2.2** Add `TuckerCompactionConfig { ranks: [usize; 3] }` (default `(8, 4, 4)`, mode-0 clamped to `min(N, 8)` at compaction time).
- [x] **T2.3** Add `pub struct TuckerCompaction { decomposition: TuckerResult, source_shape, hla_moments, source_cycle_root, report }` (the cold-tier archival envelope). Lives in new file `tucker_compactor.rs` (separate SRP from AM `shard_compactor.rs`; respects the <2048-line guideline).
- [x] **T2.4** Add `ShardCompactor::compact_tucker(&self, shards, tucker_cfg) -> Result<TuckerCompaction, TuckerCompactError>`:
  - Validate (wallets drained, commitments verify — same invariant as `compact`)
  - Flatten to `(N, 8, 8)` (shard × row × col of `style_weights`)
  - Call `tucker_decompose_into` (via `TuckerConfig::new` + `TuckerScratch` + `TuckerResultScratch`)
  - Measure reconstruction rel-Frob error (diagnostic, in report)
  - Carry forward HLA verbatim (one row per source shard — only style is compressed)
  - Bind provenance via `compute_cycle_root` (shared `pub(crate)` helper from `shard_compactor.rs`)
- [x] **T2.5** Add `TuckerCompaction::reconstruct_into(&self, out: &mut Vec<NeuronShard>)` + `reconstruct_into_with_scratch` (zero-alloc hot-path variant). Requires supporting `TuckerResultScratch::from_owned` constructor added to katgpt-core (the Cold-tier reload path: owned → scratch → `tucker_reconstruct_into`).
- [x] **T2.6** Tests: round-trip preserves HLA exactly, provenance root deterministic, distinct inputs → distinct roots, full-rank near-lossless, zone_hashes derived from cycle_root, scratch vs no-scratch path equivalence, semantic-axis cosine gate (T2.G1), zero-alloc path equivalence + cross-envelope validation (T2.G4) (24 tests written, all PASS).
- [ ] **T2.7** Comparison bench: AM compaction vs Tucker compaction on N ∈ {16} — Tucker wins at shallow AM ratios (≥0.5), AM wins at deep ratios (0.1). Cross-over documented in test `tucker_envelope_is_smaller_than_am_at_n_16`. Bench deferred (system-level dyld/trustd binary-launch blocker prevents Criterion bench from running).

### Phase 2 GOAT gate

- [x] **T2.G1** Reconstruction quality: Tucker reconstructs shards whose leading semantic axis (top `semantic_axes` direction) has cosine similarity `≥ 0.95` with the source — **PASS** via test `tucker_preserves_leading_semantic_axis` (gated on `phase_transition_subspace_phase_gate` co-feature). Constructs rank-1-dominant shards (10x dominant term + small noise), compacts with default ranks `(8,4,4)`, reconstructs, and asserts the min leading-axis cosine across the batch is ≥ 0.95. Sign-ambiguity handled via `|cos|`.
- [x] **T2.G2** Storage compression: Tucker envelope `compressed_floats` beats shallow-AM (ratio ≥ 0.5) at N=16 — **PASS via test** `tucker_envelope_is_smaller_than_am_at_n_16` (Tucker 320 < shallow-AM 512 floats at N=16, ratio 0.5). Note: at deep AM ratio (0.1, M=2 → 128 floats), AM wins — Tucker is for archival, not deep reduction. Original plan target "N=64" was re-spec'd to N=16 due to the SVD_MAX_RANK=16 constraint discovered in Phase 1.
- [x] **T2.G3** No-regression: full-rank Tucker (`r_N=N, r_R=8, r_C=8`) → reconstruction rel-Frob err `< 1e-3` — **PASS via test** `full_rank_tucker_is_near_lossless`.
- [x] **T2.G4** Alloc-free reconstruction hot path — **PASS**: added `TuckerReconScratch` (bundles `TuckerConfig` + `TuckerResultScratch` + `TuckerScratch` + `recon_buf`, built once via `from_compaction`) and `TuckerCompaction::reconstruct_into_zero_alloc(&self, &mut TuckerReconScratch, &mut Vec<NeuronShard>)`. This hoists all three per-call allocations that `reconstruct_into_with_scratch` still did (`TuckerResultScratch::from_owned`'s core clone + flat-factors Vec, plus the `recon_buf` Vec) into the persisted scratch. Verified by `reconstruct_zero_alloc_matches_baseline` (bit-identical to allocating baseline + idempotent across repeated calls) and `reconstruct_zero_alloc_rejects_mismatched_envelope` (cross-envelope shape/rank validation → `ScratchShapeMismatch`). Empirical alloc counting for the katgpt-core primitive is G4 of `bench_326_tucker_hosvd_goat` (0 allocs / 100 steady-state calls); the integration contract is enforced *by construction* (no `Vec::new`/`vec![]`/`Vec::clone` on the hot path — only `out.clear()` + `out.reserve(n)` no-ops on a reused Vec + `out.push(NeuronShard)` into pre-reserved capacity, where `NeuronShard` is a stack Pod).

### Phase 2 promotion

- [x] **T2.P** Keep `tucker_factorization` opt-in in riir-neuron-db. All four GOAT gates now pass (G1 semantic-axis cosine, G2 storage compression, G3 full-rank lossless, G4 zero-alloc reconstruct). The integration stays opt-in because it is a new cold-tier code path that needs soak time, and T2.7 (Criterion comparison bench) remains deferred (the AM-vs-Tucker crossover is already pinned by the `tucker_envelope_is_smaller_than_am_at_n_16` unit test; a Criterion bench would add perf-vs-quality curves but no new correctness signal).

---

## Phase 3 — Commit + Summary

- [x] **T3.1** Verify `cargo check --all-features` clean — **PASS** (riir-neuron-db all-features compiles).
- [x] **T3.2** Verify `cargo check -p riir-neuron-db` (default features) clean — **PASS** (no default-feature regression).
- [x] **T3.3** Verify `cargo clippy -p riir-neuron-db --features tucker_factorization -- -D warnings` clean — **PASS** (0 warnings on the new module).
- [x] **T3.4** Verify katgpt-core Tucker tests (28 total, including 3 new `from_owned` tests) — **PASS** (28 passed; 0 failed).
- [x] **T3.5** Run riir-neuron-db Tucker compactor tests — **PASS**: 24/24 tests pass (direct binary launch bypassing the `cargo test` dyld/trustd stall). Includes `tucker_preserves_leading_semantic_axis` (T2.G1, gated on `phase_transition_subspace_phase_gate`, on by default) and the two T2.G4 tests (`reconstruct_zero_alloc_matches_baseline`, `reconstruct_zero_alloc_rejects_mismatched_envelope`).
- [x] **T3.6** Commit on `develop` with `feat:` prefix.

---

## Risks

- **N-mode generality vs hot-path simplicity.** True N-mode Tucker with arbitrary shape is more code than the 3-mode `(N, 8, 8)` special case we actually need. Mitigation: implement generic N-mode (≤4 modes) upfront — the per-mode unfolding + SVD + contraction pattern is uniform and the generic code is barely longer than a 3-mode special case. The 3-mode `(N,8,8)` is then just one config.
- **SVD sign ambiguity.** `thin_svd_into` documents arbitrary sign conventions. HOSVD factor matrices inherit this; reconstruction is sign-invariant (factors appear in conjugate pairs `U × U^T`), so this is safe. Document it.
- **Cold-tier commitment semantics.** Tucker envelope is not a `Vec<NeuronShard>` — it cannot be committed via the existing per-shard BLAKE3 path. Phase 2 will need its own envelope commitment (BLAKE3 over `core || factors || source_cycle_root`). Out of Phase 1 scope.
- **SVD determinism across quorum nodes.** `thin_svd_into` is documented platform-independent (no SIMD inside). Tucker inherits this. Safe for sync-boundary commitment.
