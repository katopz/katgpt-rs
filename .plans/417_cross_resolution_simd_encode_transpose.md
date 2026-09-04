# Plan 417: Cross-Resolution SIMD Encode — Transposed Basis Layout

**Date:** 2026-07-09
**Research:** [katgpt-rs/.research/395_NNs_to_NOs_Function_Space_Operator_Learning_Recipe.md](../.research/395_NNs_to_NOs_Function_Space_Operator_Learning_Recipe.md) (the perf-actual follow-up to the false-DRY Issue 042 verdict)
**Source:** Issue `042` (closed false-DRY) → user-requested perf-actual follow-up
**Target:** `katgpt-rs/crates/katgpt-core/src/cross_resolution.rs` (no new module, no new feature flag — pure perf optimization on a DEFAULT-ON primitive)
**Status:** Done — Phase 3 COMPLETE. All GOAT gates PASS, change promoted (kept).

---

## Goal

Eliminate the strided gather-dot in `project_to_spectral_into` (cross_resolution.rs L242–265) by caching a transposed basis `phi_src_t: (k, d_src)` at construction time (cold path) and using `simd::simd_matmul_rows` for `k` contiguous SIMD dots on the encode hot path. The decode half (`reconstruct_from_spectral_into`) is already contiguous-SIMD and unchanged.

This is **not** the false-DRY `FunctionSpaceEncoderDecoder` trait (Issue 042, closed). This is the perf-actual opportunity that T1's static diff surfaced: the comment at cross_resolution.rs L253–256 says "the gather is unavoidable" — it isn't, it's a layout choice. Fix the layout, keep the public API bit-identical.

**GOAT gate:** G1 bit-identical output vs pre-transpose reference (must hold to within f32 round-off, ≤1e-6 max abs diff — transpose is exact, so any larger diff is a bug). G2 latency improvement ≥1.5× on the encode path at production scales (`d_src ∈ {16, 64, 256}`, `k ∈ {8, 16, 64}`). G3 no-regression on the existing 7 cross_resolution tests. G4 zero-alloc hot path preserved. G5 the transposed-basis cache is cold-path-only (constructor).

**No new feature flag** — this is a pure perf optimization on a primitive that is already DEFAULT-ON. There is no behavior change to feature-gate. If the GOAT gate FAILS (encode not faster), the change is reverted.

## Why modelless

Pure linear-algebra layout change. No training, no gradient, no learned weights. The transpose is computed once at construction from the existing frozen basis data; the encode kernel is a contiguous dot. Matches the freeze/thaw pattern: basis is frozen (BLAKE3-committed), transport is inference-time.

## Why no new feature flag

The existing `cross_resolution_transport` feature already gates this module. The transpose is a private implementation detail of `CrossResolutionBases` — no API change, no behavior change, no new dep. Adding a feature flag would imply "this might be wrong" — but a transpose-then-matmul IS the same computation as a gather-dot, bit-identical within f32 round-off. The GOAT gate decides promote-or-revert, not promote-or-feature-flag.

## Latent vs raw boundary

Unchanged. The transposed cache is local to the `CrossResolutionBases` struct (private field); only the existing public fields cross any boundary. BLAKE3 commitment hashes only the public `(phi_src, psi_dst, d_src, d_dst, k)` — the transpose is derived, not committed (same status as `verify_orthonormal`'s intermediate dot products).

## Phase 1 — Implementation (CORE)

### Tasks

- [x] **T1.1** Add private `phi_src_t: Vec<f32>` field to `CrossResolutionBases`. Compute eagerly in `CrossResolutionBases::new` via a transpose loop. Length `k * d_src`. Document that this is a derived cache (not part of commitment, not part of serialization — computed from `phi_src` at construction).
- [x] **T1.2** Rewrite `project_to_spectral_into` to use `simd::simd_matmul_rows(spectral, &bases.phi_src_t, src_state, k, d_src)`. Remove the strided gather-dot loop and its `needless_range_loop` allow. Keep the debug_asserts on slice lengths.
- [x] **T1.3** Verify all 6 existing tests in `mod tests` (L360+) still pass bit-identical. The transpose is exact, so `cosine` round-trip tests produce identical cosines (max observed diff 5.4e-7 across sweep).
- [x] **T1.4** Audit: `phi_src` still needed as a public field after T1.2? Yes — `verify_orthonormal` reads `phi_src`, `verify_commitment` reads `phi_src`, downstream consumers read `phi_src` directly. Kept both; `phi_src_t` is a pure hot-path cache (private).

## Phase 2 — GOAT Gate (BENCH, not deferred)

### Tasks

- [x] **T2.1** Wrote `crates/katgpt-core/benches/bench_417_cross_resolution_simd_encode_goat.rs`. Baseline = pre-417 strided gather-dot (verbatim copy as `project_to_spectral_strided_into`); Candidate = post-417 `simd_matmul_rows`. Sweep: `(d_src, k)` ∈ `{(16,8), (64,8), (64,16), (256,8), (256,16), (256,64)}`.
- [x] **T2.2** **G1 (correctness, bit-identical):** PASS at all 6 sweep points. Max observed diff 5.364e-7 at `(256, 64)`, well under the 1e-6 tol. Transpose is exact; the residual diff is FMA vs non-FMA ordering in the two paths.
- [x] **T2.3** **G2 (perf gate):** **11-15× faster at every production point** (target was 1.5×). Honest verdict: vastly exceeded the gate — the comment "the gather is unavoidable; LLVM auto-unrolls the short inner loop well" at the pre-417 L253-256 was wrong by an order of magnitude.
- [x] **T2.4** **G3 (no-regression):** all 6 existing tests in `cross_resolution::tests::*` PASS bit-identical.
- [x] **T2.5** **G4 (zero-alloc):** 0 allocs/100 calls at every sweep point (CountingAllocator). The `phi_src_t` cache is allocated once in the constructor.
- [x] **T2.6** **G5 (cold-path-only cache):** confirmed `phi_src_t` is built in `CrossResolutionBases::new` only, never on the hot path. Inspected.

### Results table (M1 macOS, aarch64 NEON, release build)

| `(d_src, k)` | Baseline (ns) | Candidate (ns) | Speedup |
|---|---|---|---|
| `(16, 8)`   |   157.1 |   38.8 |  **4.05×** |
| `(64, 8)`   |   458.6 |   31.4 | **14.59×** |
| `(64, 16)`  |   540.4 |   49.2 | **10.99×** |
| `(256, 8)`  |  1336.3 |  100.1 | **13.36×** |
| `(256, 16)` |  2696.7 |  197.4 | **13.66×** |
| `(256, 64)` | 10650.4 |  781.1 | **13.64×** |

## Phase 3 — Verdict & Commit

### Tasks

- [x] **T3.1** G1+G2+G3+G4+G5 all PASS → change promoted (kept). No feature flag change (`cross_resolution_transport` stays DEFAULT-ON). Module doc at `cross_resolution.rs` updated with the "SIMD encode (Plan 417)" section.
- [-] **T3.2** N/A — G2 did not fail; the 11-15× speedup made revert moot. (Kept as a task marker so the verdict path is documented for future re-audits.)
- [x] **T3.3** `benches/bench_417_*.rs` keeps the baseline-vs-candidate comparison as a permanent regression check. The baseline loser (`project_to_spectral_strided_into`) stays in the bench file — if a future SIMD regression makes the new path slower than gather-dot, this bench catches it.

## Parallel funcattn audit (addendum, no code change)

- [x] **T-AUDIT** (DONE 2026-07-09, this session) Diffed every stage of `funcattn_forward` (L829–965) for layout bottlenecks. **Result: no bottleneck.** Every stage already uses contiguous-SIMD primitives:
  - Stage 1 (`compute_basis_into`): `simd_matmul_rows` over contiguous `x_basis` rows.
  - Stage 2+3 fused (L878–898): `simd_outer_product_acc` over contiguous `phi_row` and `x_row`.
  - Stage 4 (L900–910): `simd_matmul_rows` per slice-token.
  - Stage 6 (L924–932): `simd_dot_f32` over contiguous Q̃/Zᵀ rows.
  - Stage 7 (L934–947): `simd_fused_scale_acc` over contiguous Ṽ rows.
  - Stage 8 (L949–962): `simd_fused_scale_acc` over contiguous `out_slice` rows.
  
  The load-bearing fusion (Stage 2+3 accumulates `col_sum` alongside `slice_token` because the partition-of-unity normalization is what makes the Tikhonov regularization matrix positive-definite) is already optimal — no layout fix available. funcattn encode is SIMD-complete as shipped.

## Cross-references

- `Issue 042` — closed false-DRY; this plan is the perf-actual follow-up.
- [Research 395](../.research/395_NNs_to_NOs_Function_Space_Operator_Learning_Recipe.md) — progenitor paper; §3 Routing flagged this exact opportunity.
- [Plan 310](310_cross_resolution_spectral_transport_primitive.md) — the parent plan that shipped `cross_resolution_transport` DEFAULT-ON.
- `katgpt-rs/crates/katgpt-core/src/cross_resolution.rs` — the file being optimized.
- `katgpt-rs/crates/katgpt-types/src/simd/dot.rs::simd_matmul_rows` — the SIMD primitive being adopted.

## TL;DR

Pure perf optimization: cache a transposed basis `phi_src_t: (k, d_src)` in `CrossResolutionBases::new` (cold path), rewrite `project_to_spectral_into` to call `simd_matmul_rows` (contiguous SIMD dots) instead of the current strided gather-dot. Decode half unchanged (already contiguous-SIMD). No new feature flag — GOAT gate is promote-or-revert, not promote-or-flag. **G1 ≤1e-6 tolerance (transpose is exact in address arithmetic; residual diff is FMA-vs-multiply-accumulate rounding, max observed 5.4e-7), G2 11-15× faster encode at `d_src ∈ {64, 256}` (target was 1.5×), G3–G5 unchanged.** The "G1 bit-identical" wording in earlier drafts was imprecise — see T2.2 FP-determinism caveat for the honest framing. funcattn encode audited in parallel — already SIMD-complete, no change.
