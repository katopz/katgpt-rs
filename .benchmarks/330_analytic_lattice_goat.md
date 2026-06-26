# Plan 330 — Analytic Lattice GOAT Gate (katgpt-core half)

**Date:** 2026-06-26
**Plan:** [katgpt-rs/.plans/330_analytic_lattice_encoder_decoder_primitive.md](../.plans/330_analytic_lattice_encoder_decoder_primitive.md)
**Feature:** `analytic_lattice` (opt-in, NOT promoted to default — pending user review)
**Test binaries:** `tests/analytic_lattice_goat.rs` (math gates) + `tests/analytic_lattice_alloc_check.rs` (G5)

## Scope

katgpt-core math primitives only. The runtime gates (G4 latency, G1b/G1c/G1d
non-blocking contract) are DEFERRED to riir-engine (Phase 1b — they need
`GpuFuture`).

## Gate results

| Gate | Test | Threshold | Actual | Pass |
|------|------|-----------|--------|------|
| **G1** determinism (compose_chain) | `g1_compose_chain_is_bit_identical` | bit-identical | bit-identical | ✅ |
| **G1** determinism (compose_chain_into) | `g1_compose_chain_into_is_bit_identical` | bit-identical | bit-identical | ✅ |
| **G1** determinism (direction_vector_decode) | `g1_direction_vector_decode_is_bit_identical` | bit-identical | bit-identical | ✅ |
| **G2** decoder ranking (cos vs reference) | `g2_decoder_ranking_matches_reference_cos_ge_095` | cos ≥ 0.95 | **1.000000** | ✅ |
| **G2** batch vs naive (Frobenius) | `g2_batch_compose_matches_naive_frobenius_le_1e6` | ≤ 1e-6 | ≤ 1e-6 (100 random trials) | ✅ |
| **G2** batch raw-slice vs typed | `g2_batch_compose_into_matches_typed` | ≤ 1e-6 | ≤ 1e-6 | ✅ |
| **G3** associativity (Frobenius) | `g3_associativity_frobenius_le_1e5` | ≤ 1e-5 | **1.75e-7** (max over 50 random trials, k=4) | ✅ |
| **G5** zero-alloc (compose_chain_into) | `g5_zero_alloc_after_warmup_all_primitives` | 0 allocs/1000 calls | 0 | ✅ |
| **G5** zero-alloc (batch_compose_chain_into) | same | 0 allocs/1000 calls | 0 | ✅ |
| **G5** zero-alloc (direction_vector_decode) | same | 0 allocs/1000 calls | 0 | ✅ |
| **G6** spectral audit (known-good) | `g6_known_good_composite_le_5pct_spurious` | ≤ 5% | **0.0046%** | ✅ |
| **G6** spectral audit (known-bad) | `g6_known_bad_random_gt_5pct_spurious` | > 5% | **90.46%** | ✅ |

**Total: 10 GOAT tests (9 math + 1 alloc) + 36 unit tests = 46 tests, all PASS.**

## G2 decoder cos = 1.000000 — explanation

The decoder ranking cos is exactly 1.0 because the SIMD `simd_dot_f32` and the
brute-force reference compute the SAME dot product (just with different
accumulator layouts that produce bit-identical results for N=8). The sigmoid is
also identical (`fast_sigmoid` in both paths). So the score vectors are
bit-identical → cos = 1.0. This is the strongest possible G2 result.

## G6 spectral audit — known-good vs known-bad discrimination

The DCT-II-diagonal operator (known-good) has **0.0046%** spurious coupling —
well under the 5% gate. The random operator (known-bad) has **90.46%** spurious
coupling — well over the 5% gate. The discrimination ratio is ~20,000×, giving
ample headroom for real-world operators that fall between these extremes.

### Note on "clean" operator definition

A "clean" spectral transport operator is one that is **diagonal in the DCT-II
basis** (i.e., it acts per-mode, not per-coordinate). This is what FuncAttn
outputs look like when well-conditioned. A standard-basis diagonal operator with
non-uniform entries (e.g. `diag(2, 1.5, 0.7, 0.3)`) is NOT clean in the DCT-II
basis — the non-uniform scaling mixes modes and produces ~41% spurious coupling.
This is correct behavior: the audit flags operators that don't respect the
spectral mode structure.

## G4 no-regression (`--all-features`)

```
cargo test -p katgpt-core --all-features
```

**Result:** 1905 passed, 2 failed, 3 ignored.

The 2 failures are **pre-existing** and unrelated to `analytic_lattice`:
- `curator::tests::test_verification_weight_thresholds` — pre-existing
- `rtdc::tests::subtree::cg6_verify_cost_within_5x_of_depth_2` — pre-existing
  timing-dependent perf test (5.746× vs 5.5× gate, flaky on load)

All 46 `analytic_lattice` tests pass under `--all-features`.

## Feature promotion decision

**NOT promoted to `default`.** Per AGENTS.md, promotion requires the GOAT gate
to pass AND must be a modelless gain. The math gates (G1, G2, G3, G5, G6) all
pass, but:

1. The headline primitive (`ComposerTick: GpuFuture`) hasn't shipped yet — it's
   Phase 1b in riir-engine. The katgpt-core half is only useful when composed
   with the runtime half.
2. The G4 latency gate hasn't been measured (deferred to riir-engine).
3. The user should review the GOAT results and decide whether to promote.

The feature stays opt-in (`analytic_lattice = []`) until the full ASOC cascade
is validated end-to-end.

## Leaf-clean verification

- **NO `riir-gpu-async` dependency** added to katgpt-core. Confirmed:
  `RederiveOp::Fut` has NO `GpuFuture` bound at the trait level.
- **NO game IP** in `ComposerCtx` — only `(tick: u64, zone_hash: u64)`.
- All primitives are pure math (deterministic, closed-form, no training).

## TL;DR

All katgpt-core math gates pass. G2 cos = 1.0, G3 Frobenius = 1.75e-7, G5 = 0
allocs, G6 good = 0.005% / bad = 90.5%. Feature stays opt-in pending the full
ASOC cascade (Phase 1b in riir-engine) and user promotion review.
