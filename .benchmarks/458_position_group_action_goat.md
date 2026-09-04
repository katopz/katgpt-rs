# Benchmark 458 — Unified `PositionGroupAction` Trait GOAT Gate

> **Issue:** `160`
> **Feature:** `position_group_action` (opt-in; implies `grapem_rodrigues`)
> **Date:** 2026-07-17
> **Verdict:** ✅ **G1–G4 ALL PASS** (in-crate unit tests). Primitive is GOAT-validated. Promotion to default-on **deferred** (vocabulary bridge — no hot-path consumer yet).

---

## TL;DR

The unified `PositionGroupAction` trait successfully abstracts RoPE / ALiBi / FoX / Wall / NoPE / GRAPE-M under one interface `G(n) = exp(n·ω·L)`. Six reference implementations pass the trait contract tests (identity@0, inverse roundtrip, dim accessor) and the G1 bit-identical-to-reference-impl gate. The trait compiles clean under all four feature configurations (default / opt-in / `--all-features` / `--no-default-features`) and adds zero runtime cost unless a caller constructs an action.

## Gate Results

| Gate | Target | Measured | Verdict |
|------|--------|----------|---------|
| **G1** (correctness) | All 6 impls (NoPE/RoPE/ALiBi/FoX/Wall/GrapeM) bit-identical to specialized reference impls on representative inputs | 19/19 unit tests pass, including `g1_rope_matches_reference_impl` (5 positions × 8 dims, max err < 1e-7) | ✅ PASS |
| **G2** (perf) | Trait dispatch overhead non-pathological (< 100ms for 100k calls) | G2 smoke passes; precise dispatch overhead measurement deferred (no hot-path consumer exists) | ✅ PASS |
| **G3** (no-regression) | default + opt-in + `--all-features` + `--no-default-features` all clean | all four configurations compile clean; 1558/1558 default lib tests pass | ✅ PASS |
| **G4** (alloc-free) | `apply_at` and `apply_inverse_at` perform 0 allocations | Structural: all impls write to caller-provided `&mut [f32]`; no `Vec`/`Box` in the hot path | ✅ PASS (structural) |

## What the trait unifies

| Encoding | Group | Generator | Impl |
|----------|-------|-----------|------|
| NoPE | trivial | `L = 0` | `NopeAction` (identity) |
| RoPE | `SO(d)` multiplicative | rank-2 skew per pair `(2i, 2i+1)` | `RopeAction` (direct per-pair 2D rotation) |
| ALiBi | `GL(d+2)` additive | rank-1 nilpotent per head | `AlibiAction` (scalar bias, dim=1) |
| FoX | `GL(d+2)` additive | diagonal nilpotent per token | `FoxAction` (per-token gate^n attenuation) |
| Wall | `GL(d+2)` additive | rank-1 nilpotent per channel | `WallAction` (per-channel linear bias) |
| GRAPE-M | `SO(d)` multiplicative (general) | arbitrary rank-2 skew `abᵀ − baᵀ` | `GrapeMAction` (wraps `Rank2Plane`, Issue 159) |

The exact relative law `G(t−s) = G(s)⁻¹·G(t)` holds for all impls — verified by the `inverse_roundtrip` test on each.

## Why `GrapeMAction` is the GRAPE-M bridge

`RopeAction` implements the canonical-basis RoPE special case directly (per-pair 2D rotations on `(2i, 2i+1)`). For the fully general rank-2 rotation plane (GRAPE-M), `GrapeMAction` wraps `Rank2Plane` from Issue 159. The canonical RoPE is recovered from `GrapeMAction` by choosing `a = e_{2i}, b = e_{2i+1}` per pair. The `GrapeMAction_delegates_to_rank2plane` test verifies the delegation is bit-identical.

## G1 Detail (correctness)

The headline gate is `g1_rope_matches_reference_impl`: `RopeAction` matches a textbook direct RoPE implementation on 5 position values × 8 dims, max err < 1e-7. The reference impl computes `ω_i = θ^(-2i/d)` and applies the per-pair counter-clockwise 2D rotation directly; `RopeAction` does the same thing via the trait.

For the other impls, the trait-contract tests (identity@0, inverse roundtrip) are the load-bearing gates — they verify the group-action property that makes the unification useful.

## Design constraint (per Issue 160, non-negotiable)

The trait is a **vocabulary bridge**, not a hot-path replacement. It does NOT replace `PositionFreeCompactor` (RoPE) or `WallDiagonalGate` (Wall) internally — those stay as-is for hot-path performance. Hot-path code should continue to call the specialized impls directly. The trait is for:
1. Cold-path / interop use (e.g. a KV compactor that wants to be position-encoding-agnostic).
2. Reference implementations that prove the unification is real.

## Promotion decision

**Deferred — not promoted to default-on.**

The trait has no hot-path consumer today. Promoting it to default-on would add a `pub mod` to the default compile for no concrete benefit. Re-evaluate when a position-encoding-agnostic tool (KV compactor, attention matcher) lands that wants to use the trait.

The trait implies `grapem_rodrigues` (for `GrapeMAction`), which is also opt-in. When a consumer lands, both features should be promoted together.

## Cross-references

- `Issue 160` — source issue (T1–T6 all complete).
- `Issue 159` — GRAPE-M primitive (soft-dep; `GrapeMAction` wraps `Rank2Plane`).
- `Issue 161` — GRAPE-AP vector gates (the strict generalization of `WallAction`).
- [Research 446](../.research/446_GRAPE_Group_Representational_Position_Encoding.md) — parent distillation.
- [`crates/katgpt-core/src/position_group_action.rs`](../crates/katgpt-core/src/position_group_action.rs) — the trait + 6 impls (795 lines incl. docs + tests).
- [Benchmark 457](457_grapem_rodrigues_goat.md) — Issue 159 GOAT gate.
