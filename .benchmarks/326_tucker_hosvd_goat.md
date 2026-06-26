# Benchmark 326 — Tucker / HOSVD GOAT Gate

[← Index](../README.md) · **Plan:** `.plans/326_tucker_hosvd_factorization.md` · **Feature:** `tucker_factorization` (DEFAULT-ON) · **Date:** 2026-06-25

## Primitive

`tucker_decompose_into` + `tucker_reconstruct_into` — N-mode (N ≤ 4) Higher-Order
SVD in `crates/katgpt-core/src/linalg/tucker.rs`. Decomposes a tensor
`X ∈ R^(I₀×…×I_{N-1})` into `S ×₀ A^(0) ×₁ A^(1) × … ×_{N-1} A^(N-1)` via N
one-sided-Jacobi SVDs of the mode-n unfoldings + tensor-times-matrix contractions.
The N-mode generalization of `thin_svd_into` from `subspace_phase_gate`.

Distilled from TFNO (Tensorised FNO) weight compression, arXiv:2511.05963 §6.1,
as the **third and final FNO gap** from Research 307 §3 (Plans 323 + 325 closed
gaps #1 and #2).

## GOAT Gate Results (all PASS)

| Gate | Contract | Target | Result | Pass |
|------|----------|--------|--------|------|
| G1 | Rank-(2,2,2) tensor recovered with ranks (2,2,2) | rel Frob err < 1e-4 | **4.096e-8** | ✅ |
| G2 | Perf: `tucker_decompose_into` + `tucker_reconstruct_into` on (8,8,8) ranks (4,4,4) | mean ≤ 500µs | **71.38µs** | ✅ |
| G3 | Full-rank (ranks=shape) reconstruction is lossless | max abs err < 1e-4 | **1.013e-6** | ✅ |
| G4 | Alloc-free hot path (pre-warmed scratch) | 0 allocs / 100 calls | **0** | ✅ |

Run: `cargo bench -p katgpt-core --features tucker_factorization --bench bench_326_tucker_hosvd_goat -- --nocapture`

## G1 detail — rank recovery

A synthetic rank-(2,2,2) tensor `X[i,j,k] = a[i]·b[j]·c[k]` (only first 2 entries
of a, b, c nonzero) is decomposed with ranks (2,2,2) and reconstructed. HOSVD
discards the ~zero singular values and recovers X to f32 round-off (4.096e-8).

## G2 detail — perf

Shape (8,8,8), ranks (4,4,4), 1000 iterations with pre-warmed scratch. Three
one-sided-Jacobi SVDs of small mode unfoldings + three tensor-times-matrix
contractions. 71.38µs is 7× under the 500µs cold-tier budget.

## G3 detail — full-rank lossless

Shape (4,4,4), ranks = shape (no truncation). Reconstruction must match input
to f32 round-off (the decomposition is lossless when no singular values are
discarded). Max abs error 1.013e-6.

## G4 detail — alloc-free

`CountingAllocator` wraps `System::alloc` and counts calls. After 10 warmup
calls (to size all internal buffers), 100 steady-state decompose+reconstruct
calls allocate **0 times**. The `TuckerScratch` + `TuckerResultScratch` pattern
pre-allocates all work buffers once at construction; the hot path borrows
`&mut self` and never grows.

## Modelless gain

HOSVD is a **deterministic, closed-form** decomposition:
- N thin SVDs (one-sided Jacobi, pure float arithmetic, platform-independent)
- N tensor-times-matrix contractions (mode products)

No training, no gradient descent, no learned parameters. The only "fitting" is
the per-mode spectral truncation (choosing `ranks`), which is deterministic
given the rank budget. This satisfies the modelless-first mandate of `katgpt-rs`.

Two quorum nodes produce bit-identical factorizations from identical inputs
(inherits `thin_svd_into`'s platform-independence) — safe for sync-boundary
commitment paired with a BLAKE3 envelope over `core || factors`.

## Design constraints

- **N ≤ 4 modes** (`MAX_MODES = 4`) — covers the TFNO `(K,I,O)` weight-tensor
  shape and the shard `(4,4,4)` / batch `(N,8,8)` cases.
- **SVD_MAX_RANK = 16** — the underlying one-sided-Jacobi SVD has a
  stack-allocated `[f32; 16]` raw-sigma buffer. `TuckerConfig::new` rejects
  shapes where any mode-n unfolding's smaller dimension exceeds 16 (returns
  `ShapeExceedsSvdLimit`). For shard-batch Tucker `(N, 8, 8)`: mode-0 unfolding
  is `(N, 64)`, so N must be ≤ 16. Larger batches must be chunked.

## Dimensional correction vs Research 307

Research 307 §2.3/§3 listed the shard reshape as `(8,8,8) = 512`. That was a
dimensional error — `NeuronShard::style_weights` is 64 elements, not 512. The
note conflated the TFNO paper's generic `(K,I,O)` weight-tensor shape with our
64-element shard vector. The actual natural 3-tensor reshape of a 64-element
flat buffer is `(4,4,4)` (4³ = 64). This primitive is generic over `(I₀,…,I_{N-1})`;
the `(4,4,4)` shard case is the riir-neuron-db integration target, shipped as
Phase 2 of Plan 326 (`ShardCompactor::compact_tucker`, opt-in — see T2.P).

## Promotion

**Promoted to DEFAULT-ON** (Phase 3, 2026-06-25). All 4 GOAT gates PASS. Pure
modelless gain (closed-form HOSVD, no training). Feature is now on for every
consumer of `katgpt-core` by default. Downstream impact is pure additive — the
module compiles but does nothing unless a caller invokes `tucker_decompose_into`.

## Cross-references

- Research 307: `.research/307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md` §3
- Plan 326: `.plans/326_tucker_hosvd_factorization.md`
- `subspace_phase_gate::thin_svd_into` — the 2D SVD this generalizes
- Plan 323 (Fourier Continuation) + Plan 325 (Spectral Differentiation) — the
  other two FNO gaps from Research 307 §3, both DEFAULT-ON

## TL;DR

Tucker/HOSVD N-mode tensor factorization ships DEFAULT-ON. G1 rank recovery
4.1e-8, G2 perf 71µs, G3 full-rank lossless 1.0e-6, G4 0 allocs. Pure modelless
(closed-form). Closes the third and final FNO gap from Research 307 §3.
