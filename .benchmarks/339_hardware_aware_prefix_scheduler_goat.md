# Plan 339 — Hardware-Aware Prefix Scheduler GOAT Gate Results

**Date:** 2026-06-28
**Feature:** `hardware_aware_scheduler` (opt-in, default-OFF)
**Plan:** [`.plans/339_hardware_aware_prefix_scheduler.md`](../.plans/339_hardware_aware_prefix_scheduler.md)
**Issue:** [#003](../.issues/003_hardware_aware_prefix_scheduler.md) (resolved by this implementation)
**Source:** DSpark (DeepSeek-AI, 2026) §3.2.2, Algorithm 1, Appendix A
**Research:** [`.research/316_DSpark_Confidence_Scheduled_Speculative_Decoding.md`](../.research/316_DSpark_Confidence_Scheduled_Speculative_Decoding.md)

## Status: GOAT G1–G5 ALL PASS on synthetic workload; opt-in until real multi-request caller

The synthetic GOAT gate passes on all five gates. Per Issue 003's own risk #1,
the gate is **vacuous without a real multi-request batch caller**: katgpt-rs
default is single-request, so the synthetic G2 (multi-request throughput gain
on a cliff SPS curve) is necessary but not sufficient for promotion. The
feature stays opt-in (`hardware_aware_scheduler = []`, default-OFF) until a
real caller (riir-ai crowd-NPC cognition or a katgpt-rs batch server)
exercises it.

## GOAT gate results

| Gate | Description | Result | Evidence |
|------|-------------|--------|----------|
| **G1** | Single-request correctness — R=1 preserves LeviathanVerifier semantics (no selection bias from non-anticipating early-stop). | ✅ PASS | Constant-SPS admit-all `[5]` (no truncation → verifier sees every position); cliff-SPS truncation `[3]` stable across suffix extension (non-anticipating property holds). |
| **G2** | Multi-request throughput — R=4, `Θ_scheduler ≥ Θ_uniform × 1.05` on cliff SPS curve. | ✅ PASS | `scheduled Θ = 170.0`, `uniform Θ = 14.71`, ratio **11.55×** (+1055% gain), out = `[3, 1, 0, 0]`. Far exceeds the 1.05 bar. |
| **G3** | No regression — feature isolation. | ✅ PASS | `hardware_aware_scheduler = []` (zero feature deps), gates exactly one module, exports two symbols. Default build clean, no-default build has zero new errors vs the pre-existing 3 (EarlyStopGate/SpecCostSnapshot/StabilitySnapshot re-exports — see Issue 005 A2). |
| **G4** | Zero-alloc hot path. | ✅ PASS | `schedule_with_scratch` reuses candidate-scratch capacity across calls (capacity 32 stable across same-size and smaller inputs). Only allocation is the caller-owned `Vec<usize>` output. |
| **G5** | Sigmoid discipline — no softmax in implementation. | ✅ PASS | `realized_theta` is raw `τ · SPS(B)` product (no Σ normalization). Verified: τ=1.1 × SPS=4.0 = 4.4 bit-identical. Zero `softmax`/`Softmax` tokens in source. |

## Reproducing

```bash
CARGO_TARGET_DIR=/tmp/katgpt-prefix-scheduler \
  cargo test --features hardware_aware_scheduler \
    --test prefix_scheduler_goat -- --nocapture

CARGO_TARGET_DIR=/tmp/katgpt-prefix-scheduler \
  cargo test --features hardware_aware_scheduler \
    --lib prefix_scheduler
```

Expected output: 6/6 GOAT tests pass + 18/18 inline unit tests pass.

## Why G2 shows 11.55× (not just ≥1.05×)

The synthetic cliff SPS curve (100 SPS at B≤4, 10 SPS at B=5, 1 SPS at B=16)
makes uniform allocation pathological: uniform `[2,2,2,2]` puts 4
high-survival prefix tokens + 4 low-survival suffix tokens through the
expensive batch, while the scheduler allocates `[3,1,0,0]` — concentrating
the verify budget on the 4 highest-survival tokens. Real hardware SPS curves
(DSpark §5.2) are less sharp, so real gains will be smaller; the 11.55× is an
upper bound, not a production projection.

## Why this is NOT promoted to default-on

Per AGENTS.md "Promotion requires modelless gain" — the gain IS modelless (no
training, pure greedy + sort + cumprod), but Issue 003 risk #1 says the
primitive has no leverage if the engine never batches multiple requests.
katgpt-rs default is single-request. Promotion requires:

1. A real multi-request batch caller to exercise the scheduler.
2. A profiled SPS curve from our CPU/SIMD/wgpu stack (the synthetic cliff is
   not a measurement).
3. Re-running the GOAT gate on the real workload.

The synthetic GOAT gate here is the **necessary precondition** for promotion,
not the promotion itself.

## Risks carried forward

- **SPS curve shape.** DSpark §5.2 notes real hardware has jagged step-wise
  curves. The non-anticipating early-stop assumes unimodality for global
  optimality; jagged curves yield locally-optimal allocations (still
  lossless, just not optimal).
- **Non-anticipating early-stop is a correctness theorem, NOT a heuristic.**
  DSpark §5.2 removes it in production via async 2-step-prior prediction +
  ZOS causality argument. We do NOT port that variant — removing the
  synchronous early-stop without porting the async-ZOS proof would silently
  break distribution preservation.

## TL;DR

Plan 339 ships `HardwareAwarePrefixScheduler` + `SpsCurve` behind
`hardware_aware_scheduler` (opt-in). GOAT G1–G5 ALL PASS on synthetic
multi-request workload (11.55× throughput gain on cliff SPS curve). Stays
opt-in per Issue 003 risk #1 until a real multi-request batch caller exercises
it on profiled hardware. The non-anticipating early-stop (DSpark Appendix A)
is preserved bit-identically — it is the correctness theorem that guarantees
lossless speculative decoding.
