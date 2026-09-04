# Bench 678 — Issue 689: LaProp `NormalizedMomentumAccumulator` GOAT

**Date:** 2026-08-26 · **Repo:** katgpt-rs · **Issue:** `689`
**Source:** riir-train Research 428 (LaProp arXiv:2002.04839) Path-0 C2/C3.

## What shipped

`crates/katgpt-core/src/laprop.rs` behind `laprop = []` (opt-in):
`NormalizedMomentumAccumulator<const D: usize>` + `NormalizedMomentumScalar`
— EMA over RMS-normalized intake `m ← μ·m + (1−μ)·x/√(n̂+ε)` with
bias-corrected reads and the closed-form accumulator bound. Zero deps, all
state inline arrays (G4 by construction).

## The bound (proof sketch in the module doc)

Each intake is normalized by the second-moment estimate **at its own time
step**: `n̂_s ≥ x_s²·(1−ν)/(1−ν^s)` ⟹ `|u_s| ≤ √((1−ν^s)/(1−ν)) ≤ 1/√(1−ν)`.
`m` is a convex combination of intakes (weights `(1−μ)μ^k` summing to 1) ⟹
`|m|_∞ ≤ 1/√(1−ν)` — **no clamp anywhere**. L2 form `√D/√(1−ν)` pinned
alongside (per-component bounds are L∞; the norm choice is never implicit).

## GOAT gates — ALL PASS

| Gate | Result |
|---|---|
| **G1a** planted outlier (one 1e6 delta into a 10k-step unit-scale stream, then 1k clean steps) | PASS — every bias-corrected component ≤ `bound()·(1+1ulp)` (ν=0.9 → 3.1623), L2 form also holds, no clamp in the primitive |
| **G1b** outlier residual decay | PASS — after the spike, k zero-steps leave the RAW accumulator at **exactly** `μ^k·m_spike` (bit-identical; `±0` sign-flip via `+0.0` tolerated). Honest correction to the issue's spec: the exact law is on `m`, not the bias-corrected read `m̂` — the `(1−μ^t)` denominator legitimately changes as t grows (measured ratio 0.334 = (1−0.9²)/(1−0.9⁸)) |
| **G1c** the clamped raw-EMA shape (riir-clippy `ema_step` sans clamp) on the same outlier | FAILS the bound by >10× (raw dir ≈ 1e5 vs bound 3.16) — the A/B proving the gain is real |
| **G2** per-push cost vs raw EMA at D=8 (release) | PASS — within the +15 ns budget (one extra mul + FMA per component) |
| **G3** default features unchanged | PASS — feature-gated `laprop = []`; default + `--no-default-features` check clean |
| **G4** zero allocs per push (TrackingAllocator, 1000× push+read) | PASS — 0 allocations |
| **G5** ν-dial | PASS — ν=0 degenerates to the ternary-sign limit (constant ± streams saturate to ±1; bound finite = 1.0); bound monotone in ν ∈ (0,1) |
| Precision honesty | L∞ + L2 forms both pinned by test |
| C8 contrast | `coupling_cost(μ,ν)` = Adam's `1/(1−μ/√ν)` — INFINITE at μ=0.99 ≥ √0.9801 where the LaProp bound stays finite and μ-independent |

## T3 promote/demote decision

**Stays opt-in.** G1–G4 pass, but promotion requires a default consumer
(the rule the issue itself encodes). The documented consumer (riir-clippy
`ema_step` → this primitive) is a follow-up in THAT repo — a behavior change
(clamp semantics → bound semantics) behind its own A/B there, preserving
`ema_step`'s returned delta-norm contract.

## Deviations from the issue sketch (documented)

- `momentum()` returns `[f32; D]` **by value** (bias correction is computed —
  a borrow can't carry it); `momentum_uncorrected()` provides the raw borrow;
  `momentum_into()` is the zero-alloc read.
- `bound()`/`influence()` are plain fns, not `const fn` — `f32::sqrt` is not
  const-stable.

**Record:** commit on `develop` (see git log `feat(laprop)` — Issue 689).
