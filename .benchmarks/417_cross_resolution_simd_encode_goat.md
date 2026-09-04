# Plan 417 — Cross-Resolution SIMD Encode GOAT Gate

**Date**: 2026-07-09
**Plan**: `.plans/417_cross_resolution_simd_encode_transpose.md`
**Source**: Issue `042` (closed false-DRY) → user-requested perf-actual follow-up
**Research**: `.research/395_NNs_to_NOs_Function_Space_Operator_Learning_Recipe.md` (§3 Routing flagged this exact opportunity)
**Target**: `crates/katgpt-core/src/cross_resolution.rs::project_to_spectral_into`
**Target dir**: `CARGO_TARGET_DIR=/tmp/xres417`
**Host**: M1 macOS, aarch64 (NEON SIMD backend)

## Summary verdict

| Gate | Verdict | Key evidence |
|------|---------|--------------|
| G1 (correctness, ≤1e-6 tol) | ✅ PASS | max\|Δ\| = 5.364e-7 worst point (d_src=256, k=64); all 6 sweep points under tol |
| G2 (perf, ≥1.5× at production points) | ✅ PASS | **11.08×–14.60× at the 4 production points** (target was 1.5×) |
| G3 (no-regression, existing tests) | ✅ PASS | 6/6 `cross_resolution::tests::*` green |
| G4 (zero-alloc hot path) | ✅ PASS | 0 allocs / 100 calls at every sweep point (CountingAllocator) |
| G5 (cold-path-only cache) | ✅ PASS | `phi_src_t` built only in `CrossResolutionBases::new` (inspection) |

**All gates PASS. Change promoted (kept) — `cross_resolution_transport` stays DEFAULT-ON.**

---

## G1 (correctness, ≤1e-6 tolerance)

**Bench**: `crates/katgpt-core/benches/bench_417_cross_resolution_simd_encode_goat.rs`
**Setup**: same random-orthonormal `phi_src` (seeded) + same `src_state` (seeded) fed to both
- baseline `project_to_spectral_strided_into` (verbatim pre-417 strided gather-dot)
- candidate `project_to_spectral_into` (post-417 `simd_matmul_rows` over transposed cache)

| `(d_src, k)` | max\|Δ\| | tol | Verdict |
|---|---|---|---|
| `(16, 8)`   | 5.960e-8 | 1e-6 | ✅ |
| `(64, 8)`   | 1.192e-7 | 1e-6 | ✅ |
| `(64, 16)`  | 2.384e-7 | 1e-6 | ✅ |
| `(256, 8)`  | 2.012e-7 | 1e-6 | ✅ |
| `(256, 16)` | 2.086e-7 | 1e-6 | ✅ |
| `(256, 64)` | 5.364e-7 | 1e-6 | ✅ |

**Note on the "transpose is exact" claim.** The transpose *is* exact at the address-arithmetic
level (the same `phi_src[r*k + j]` value is read, just from `phi_src_t[j*d_src + r]`). But the
two paths differ in FP rounding: the pre-417 strided path accumulates with `*` + `+=`
(two-rounding), while `simd_matmul_rows` → `simd_dot_f32` uses single-rounding FMA
(`mul_add` / `vfmaq_f32` / `_mm256_fmadd_ps`). So G1 is a *tolerance* gate, not a
*bit-identical* gate — the ≤1e-6 wording in the plan is honest, the "bit-identical" wording in
earlier drafts was imprecise. The transported value is latent (consumed by
`velocity_field_ensemble`, not a raw sync-boundary value), so ULP drift at this magnitude is
acceptable.

---

## G2 (perf, ≥1.5× at production points)

**Production points** (`d_src ∈ {64, 256}` with `k ∈ {8, 16}`): all 4 must clear 1.5×.
Gate passes if ANY production point clears it; in practice ALL 4 cleared it by 11× or more.

**This run** (M1 aarch64, NEON, `ITERS=100_000` per point):

| `(d_src, k)` | Baseline (ns) | Candidate (ns) | Speedup | Production? | Verdict |
|---|---|---|---|---|---|
| `(16, 8)`   |   172.9 |   39.7 |  **4.35×** | no  | (not gated) |
| `(64, 8)`   |   469.9 |   32.2 | **14.60×** | yes | ✅ |
| `(64, 16)`  |   540.4 |   48.8 | **11.08×** | yes | ✅ |
| `(256, 8)`  |  1335.7 |   95.7 | **13.96×** | yes | ✅ |
| `(256, 16)` |  2663.4 |  190.3 | **14.00×** | yes | ✅ |
| `(256, 64)` | 10624.9 |  761.0 | **13.96×** | no  | (not gated) |

**Honest verdict on the pre-417 comment.** The in-file comment at the pre-417 L253-256
("the gather is unavoidable; LLVM auto-unrolls the short inner loop well") was wrong by an
order of magnitude. The auto-unrolled strided gather-dot is 11-15× slower than a contiguous
SIMD dot at these scales on NEON. The layout fix (transpose-once-at-construction,
contiguous-SIMD-on-hot-path) is the right answer.

**Run-to-run variance.** A sibling session's run measured 10.99×–14.59× at production points
(same band, slight variance in the ns measurements). Both runs agree on the 11-15× speedup
range. The `(16, 8)` non-production point is a wash candidate (4.35× here, 4.05× sibling) —
small enough that auto-unrolled gather is closer to competitive, but contiguous-SIMD still
wins.

---

## G3 (no-regression)

```bash
CARGO_TARGET_DIR=/tmp/xres417 cargo test -p katgpt-core --features cross_resolution_transport --lib cross_resolution
```

```
running 6 tests
test cross_resolution::tests::constructor_rejects_rank_deficient_k ... ok
test cross_resolution::tests::constructor_rejects_shape_mismatch ... ok
test cross_resolution::tests::cross_domain_variant_runs_and_matches_manual ... ok
test cross_resolution::tests::smoke_asymmetric_dims_compile_and_transport ... ok
test cross_resolution::tests::smoke_non_bandlimited_loses_information ... ok
test cross_resolution::tests::smoke_roundtrip_preserves_bandlimited_signal ... ok

test result: ok. 6 passed; 0 failed; 0 ignored
```

Also verified `cargo check --no-default-features --features cross_resolution_transport` and
`cargo check --all-features` both compile clean.

(The plan text's "7 existing tests" was an off-by-one in helper counting: `make_rng`,
`identity_truncated`, `random_orthonormal`, and `cosine` are non-test helpers.)

---

## G4 (zero-alloc hot path)

CountingAllocator in the bench: 0 allocs / 100 calls at every sweep point. The `phi_src_t`
cache is allocated once in `CrossResolutionBases::new` (cold path); the hot path
`project_to_spectral_into` only calls `simd_matmul_rows`, which writes into the
caller-provided `spectral` buffer.

## G5 (cold-path-only cache)

Inspection: `phi_src_t` is built only in `CrossResolutionBases::new` via `transpose_phi_src`
(cross_resolution.rs L141). The hot path `project_to_spectral_into` (L301) only reads it.

---

## Why modelless

Pure linear-algebra layout change. No training, no gradient, no learned weights. The transpose
is computed once at construction from the existing frozen basis data; the encode kernel is a
contiguous dot. Matches the freeze/thaw pattern: basis is frozen (BLAKE3-committed), transport
is inference-time.

The transposed cache is **not** part of the BLAKE3 commitment — `compute_commitment` hashes
only `phi_src`, not `phi_src_t`. The transpose is derived (same status as
`verify_orthonormal`'s intermediate dot products). It is **not** serialized — rebuilt from
`phi_src` at construction.

## Latent vs raw boundary

Unchanged. The transported value is latent (consumed by `velocity_field_ensemble` /
`transport_cross_resolution`), not a raw sync-boundary value. The 5.4e-7 worst-case ULP drift
is well within the acceptable band for latent scalar projection.

## Reproduce

```bash
CARGO_TARGET_DIR=/tmp/xres417 cargo bench -p katgpt-core \
  --features cross_resolution_transport \
  --bench bench_417_cross_resolution_simd_encode_goat -- --nocapture

# Workaround for intermittent macOS dyld/trustd launch stall:
CARGO_TARGET_DIR=/tmp/xres417 /tmp/xres417/release/deps/bench_417_cross_resolution_simd_encode_goat-* --nocapture
```

The bench file keeps the pre-417 `project_to_spectral_strided_into` baseline verbatim as the
"loser we beat" reference (defend-wrong PoC discipline). If a future SIMD regression makes the
new path slower than gather-dot, this bench catches it.
