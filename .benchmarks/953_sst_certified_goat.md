# Bench 953 — sst_certified GOAT (Issue 958)

**Status:** COMPLETE — G1–G4 PASS, opt-in (promotion needs a wired consumer, per the issue's own consumer-dependency clause)

**Date:** 2026-10-23 · **Box:** m3 (M3 Max, 16 cores) · **Provenance:** `power=AC Power load=3.68 swap=0.00M canary=skipped powermode=2(high)` (bench_preflight PASSED at run start; load declining from 7.34 — quote with the load class)

## What ran

```bash
cargo bench -p katgpt-core --features sst_project --bench sst_project_goat -- --nocapture
SST_BENCH_HEAVY=1 cargo bench -p katgpt-core --features sst_project --bench sst_project_goat -- --nocapture
cargo test  -p katgpt-core --features sst_project            # lib 2177 + G1 6 + G4 1
./scripts/proof_gate.sh && ./scripts/proof_negative_test.sh  # Lean pair, 57 theorems
```

## Verdict

| Gate | Result | Evidence |
|---|---|---|
| G1 correctness | **PASS** | 10 inline + 6 integration arms: thin_svd oracle subspace agreement < 1e-4; brute-force anatomy match (dense + factored); planted-leak refusal with the leak MEASURED (fraction 0.1 ± 0.05 at α=1/3); isotropic-rounding negative (dense FAILS / factored PASSES — non-vacuity); compose identity; stale-anchor refusal; width guard; digest sign-flip + refit determinism; the 2×2 Lean mirror |
| G2 latency | **MEASURED (honest-cost)** | table below; the issue's round-1 µs-class target was wrong at rank (as the issue predicted) — applications are Θ(mnk) |
| G3 postures | **PASS** | clippy −D clean at `--features sst_project` (all targets), `--all-features`, `--no-default-features` |
| G4 alloc | **PASS** | 0 allocations over 1000 cycles across anatomy/factored/factor/certify/apply/decode/compose (warm scratch) |
| Lean | **PASS** | proof_gate 57/57 within budget (scalar arm + level-set instance + 2×2 counterexample + leak read); negative test 8/8 |

## G2 numbers (ms; `--release`, bench profile)

| shape (m×n, f) | fit ms | anatomy µs | factored µs (r=8) | factor ms | certify µs | apply µs | tail f |
|---|---|---|---|---|---|---|---|
| 128×512, 32 | 40–59 | 9 000 | 575 | 12 | 9 000 | 12.7 | 32×32 |
| 256×1024, 64 | 670 | 84 000 | 2 350 | 113–117 | 85 000 | 52 | 64×64 |
| 512×2048, 96 | 9 100 | 713 000 | 10 000 | 886 | 718 000 | 158 | 96×96 |
| 1024×4096, 96 (heavy) | 136 963 | 8 570 000 | 81 000 | 8 315 | 6 244 000 | 313 | 96×96 |

Readings, disclosed:

- **Fit is calibration-time and Jacobi-sweep-dominated**: the 137 s at (1024,4096) is ~60 sweeps × 2·min³ f64 at tol 1e-14·trace. At the MLP head shape (4096×11008) the Gram alone is ~9.2e13 f64 MACs — hours-class, per-matrix, one-time. This is the issue's own anchor-cost honesty, measured; a production fit lane wants a randomized-range path (NOT built here — out of scope).
- **Dense anatomy is Θ(mnk)** — seconds-class at 4k² shapes, linear in every dim. The issue's "0.7–5 GFLOP" figure is the FACTORED path: measured 81 ms at (1024,4096,r=8) — the consumer posture (LoRA-class deltas) is sub-second at head shapes by extrapolation of the linear rate.
- **apply is the shipping hot path** and is cheap (313 µs at (1024,4096) per column block) — `U·(C·(Vᵀ·X))` never materializes the delta.
- Projections (never measured): 4096×4096 square fit ≈ hours; head-shape anatomy ≈ 2·(m+n)·k·cols ≈ minutes-class per dense call.

## Interpretive decisions recorded

1. **"r=0 ⇒ Π = I byte-identical (disarmed)"** is pinned as the k = min posture (`free_min = 0`): the complement is empty, the factor's payload is 0×0, and the decoded rebuild is the zero matrix byte-identically. The k = 0 posture (`free_min = min`) is REFUSED per the issue's width guard (an SST non-case). Both edges tested.
2. **The wide-side tail basis is the σ-visible truncation** (≤ min−k directions, σ-propagated from the narrow complement): full wide-tail coverage needs the wide Gram (infeasible at these shapes) — the truncation still confines EXACTLY (any sub-basis of the tail is in the tail), and the dropped energy is reported in `SstFactorReport`, never hidden. The certifier covers ANY decoded delta regardless (it never needs the missing basis).
3. **rel_tol band [1e-5, 1]** — the f32 two-positive-terms anatomy resolves to ~1e-5 of total energy; below that a "certified" claim would be noise wearing a flag.

## Consumers (unchanged — the issue's dependency clause)

Certification lanes 3/5/6 have NO existing-delta consumer (every shipped delta predates tail confinement — ~98% protected energy by the paper's study). Value gates on riir-train Plan 470 producing tail-confined deltas, or hand-constructed factored overlays. Lane 1 (anatomy) is the consumer-independent value. **Stays opt-in** — promotion is a consumer-wiring act, not a re-flag.
