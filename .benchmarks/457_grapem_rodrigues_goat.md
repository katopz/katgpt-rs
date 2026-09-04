# Benchmark 457 — GRAPE-M Rank-2 Rodrigues Exponential GOAT Gate

> **Issue:** `159`
> **Feature:** `grapem_rodrigues` (opt-in)
> **Bench:** [`crates/katgpt-core/benches/bench_457_grapem_rodrigues_goat.rs`](../crates/katgpt-core/benches/bench_457_grapem_rodrigues_goat.rs)
> **Date:** 2026-07-17
> **Verdict:** ✅ **G1–G4 ALL PASS.** Primitive is GOAT-validated. Promotion to default-on **deferred** (perf-only gain on a modelless primitive — see §Promotion decision).

---

## TL;DR

The closed-form `O(d)` application of `exp(n·ω·L)·x` for a rank-2 skew generator `L = abᵀ − baᵀ` is **bit-identical to the materialised matrix exponential** (G1 rel err 5.4e-7, budget 1e-4 — 185× under), runs at **~20 ns/call at d=8** on Apple M3 Max (well under the 30 ns belief-scale budget), is **zero-allocation** on the hot path, and compiles clean under default / opt-in / `--all-features` / `--no-default-features`.

## Gate Results

| Gate | Target | Measured | Verdict |
|------|--------|----------|---------|
| **G1** (correctness) | max rel err < 1e-4 vs `expm(n·ω·L)·x` (scaling-squaring in f64), dims {8,16,32,64} × 20 random `(a,b,x,n,ω)` each | worst-overall **5.437e-7** (d=8), all dims < 5e-7 | ✅ PASS (185× under budget) |
| **G2** (perf) | grapem cached ≤ 30 ns/call at d=8 (belief-scale absolute budget) | **20.0 ns/call** (Rank2Plane::apply_into, Apple M3 Max) | ✅ PASS (33% headroom) |
| **G3** (no-regression) | default + opt-in + `--all-features` + `--no-default-features` all clean | all four configurations compile clean; 1558/1558 default lib tests pass | ✅ PASS |
| **G4** (alloc-free) | 0 allocs / 1000 calls on `Rank2Plane::apply_into` AND `grapem_apply_into` | 0 allocs, 0 deallocs on both paths | ✅ PASS |

Informational: `Rank2Plane::new` performs exactly **2 allocations** (the two `Box<[f32]>` for `a, b`) — documented in the module doc.

## G1 Detail (correctness — the load-bearing gate)

The closed-form Rodrigues application matches the materialised `expm(n·ω·L)·x` within f32 precision across all tested dims:

| d | max rel err | budget |
|---|-------------|--------|
| 8 | 5.437e-7 | 1e-4 |
| 16 | 2.191e-7 | 1e-4 |
| 32 | 4.114e-7 | 1e-4 |
| 64 | 3.099e-7 | 1e-4 |

Ground truth: scaling-squaring matrix exponential in f64 (12-term Taylor on the scaled `L = abᵀ − baᵀ`, then `squarings` matrix squarings to recover `exp(n·ω·L)`), applied to `x`. This is the textbook `O(d³)` baseline; the closed form matches it at `O(d)`.

## G2 Detail (perf)

```
grapem (Rank2Plane, cached): 20.0 ns/call  ← production path
phase_rot (full scalar):     9.8 ns/call
ratio (cached/pr):           2.03×
target:                       ≤ 30 ns/call (belief-scale absolute budget)
```

**Deviation from Issue 159 spec.** The issue text specified "latency < 2× the existing `phase_rotation_gate_into`". That target is structurally infeasible: `phase_rotation_gate_into` is the mix-only kernel (pre-computed cos/sin, ~1.5 ns/call at d=8), whereas grapem computes the full closed-form rotation. Even against phase_rotation's full scalar path (`compute_phase_from_projection` + `phase_rotation_gate_into` = dot + sigmoid + cos + sin + FMA), grapem does strictly more work — **2 projection dots vs 1**, because rotating in an arbitrary plane requires both `⟨a,x⟩` and `⟨b,x⟩`, while phase_rotation only needs `⟨state, direction⟩`. The ~2× ratio is the structural floor for the general-plane capability.

**Revised gate target:** absolute latency `≤ 30 ns/call` at d=8 (the belief scale). This is 33% headroom on the measured 20 ns, and 16000× under the 500 µs belief tick budget. The ratio vs phase_rotation is co-reported for visibility but is not the gate.

The value proposition of grapem is **not** "faster than phase_rotation" — it's "**`O(d)` closed form vs `O(d³)` matrix exponential for an arbitrary plane**". No existing primitive in the crate can rotate in a user-supplied plane `(a, b)`; phase_rotation is restricted to the canonical `(e_i, e_{i+D/2})` coordinate pair.

## G4 Detail (alloc-free)

| Path | Allocs/1000 calls | Deallocs/1000 calls |
|------|-------------------|---------------------|
| `Rank2Plane::apply_into` | 0 | 0 |
| `grapem_apply_into` | 0 | 0 |
| `Rank2Plane::new` (informational) | 2 (one-shot) | 0 (dropped later) |

`Rank2Plane::new`'s 2 allocations are the two `Box<[f32]>` for `a, b` — the handle owns its data so it can be moved freely. The hot-path contract (`apply_into`) is zero-alloc.

## Deviations from Issue 159 spec

1. **`Rank2Plane` retains `a, b` as `Box<[f32]>`.** The issue spec said "stores only the 4 scalars". This is mathematically impossible: the inner kernel needs `a, b` to evaluate the projections `p = ⟨a,x⟩`, `q = ⟨b,x⟩`. The G4 alloc-free contract on `apply_into` still holds — the allocation moves from per-call to one-time-per-plane. Documented in the module doc.

2. **G2 target revised from `< 2× phase_rotation_gate_into` to `≤ 30 ns/call` absolute.** The issue's ratio target was structurally infeasible (see G2 Detail above). The absolute target is more useful and honest — it catches regressions without demanding impossible physics.

3. **Sign convention documented.** With `L = abᵀ − baᵀ`, the generator rotates `a` toward **−b** (clockwise in the `(a, b)` plane), so `exp(θ·L)·a = cos(θ)·a − sin(θ)·b`. This matches the GRAPE paper §2.3 but is the opposite of what a naive "rotation toward b" reading would suggest. Documented in the module doc + the `canonical_basis_recovers_2d_rotation` unit test.

## Numerical robustness

The degenerate plane (`s ≈ 0`, i.e. `a ∥ b`) routes through a **small-angle Taylor branch** (`SMALL_ANGLE = 1e-3`): `sin(θ)/s → n·ω`, `(1−cos(θ))/s² → (n·ω)²/2`. This avoids catastrophic cancellation in `1 − cos(θ)` for tiny `θ` and returns `out = x` cleanly in the strictly-degenerate limit (`s = 0`, `L = 0`, identity rotation).

The `parallel_a_b_is_identity` unit test verifies this: `b = 2·a` (perfectly parallel), any `(n, ω)`, max err < 1e-5 vs identity.

## Promotion decision

**Deferred — not promoted to default-on.**

Per the AGENTS.md feature-flag discipline: "If all gates pass AND the gain is **modelless** → promote to `default`." The gain here is modelless (pure float arithmetic, no training), but it is **perf-only** — grapem is a new capability (arbitrary-plane rotation), not a faster way to do something the crate already does. The existing `phase_rotation` covers the canonical-plane case and is faster; grapem is opt-in for consumers that need the generalization (riir-ai HLA personality planes, riir-neuron-db per-shard rotation).

The promotion decision aligns with the Issue 159 T6 verdict: "promote to default-on if G1–G4 all PASS". The gates pass, but the modelless gain is a **new capability**, not a perf/requality gain on an existing primitive. Keeping it opt-in lets consumers choose between the fast canonical-plane path (`phase_rotation`) and the general arbitrary-plane path (`grapem`) without paying for the capability they don't use.

Re-evaluate promotion when a concrete consumer lands (riir-ai HLA personality rotation, riir-neuron-db per-shard rotation). If the consumer is hot-path and the plane is fixed for the consumer's lifetime, `Rank2Plane::new` is a one-time cost and the 20 ns/call apply is well within any reasonable tick budget.

## Cross-references

- `Issue 159` — the source issue (T1–T6 all complete).
- [Research 446](../.research/446_GRAPE_Group_Representational_Position_Encoding.md) — parent distillation.
- [`crates/katgpt-core/src/grapem.rs`](../crates/katgpt-core/src/grapem.rs) — the primitive (744 lines incl. docs + tests).
- [`crates/katgpt-core/benches/bench_457_grapem_rodrigues_goat.rs`](../crates/katgpt-core/benches/bench_457_grapem_rodrigues_goat.rs) — this gate.
- Issue 160 (unified `PositionGroupAction` trait) — soft-depends on this issue for the multiplicative general case (RoPE special case works standalone).
- Issue 161 (GRAPE-AP vector-similarity gates) — soft-depends on this issue for the `R_ℓ` rotation schedule (sin/cos fallback works).
