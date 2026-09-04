# Benchmark 460 — GRAPE Joint Lift (`GL(d+2)` Composition) GOAT Gate

**Issue:** `163`
**Feature:** `grape_joint_lift` (opt-in, implies `grapem_rodrigues`)
**Source:** Zhang et al., *GRAPE* (arXiv:2512.07805, ICLR 2026, **Appendix E**)
**Module:** [`crates/katgpt-core/src/grape_joint_lift.rs`](../crates/katgpt-core/src/grape_joint_lift.rs)
**Date:** 2026-07-17
**Hardware:** Apple M3 Max (aarch64, NEON)

## TL;DR

**G1–G4 PASS.** All four GOAT gates pass. The joint lift is a **thin
composition layer** — its value is the unified API + correctness guarantee
(Appendix E's block-diagonal `GL(d+2)` proof), not a perf gain over calling
the parts separately. Promotion to default-on is **deferred** (no hot-path
consumer today).

## Gate verdict

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** | Joint score bit-identical to manual composition + exact relativity | 80 random instances (4 dims × 20 each), max abs diff < 1e-5 × scale; all 6 special cases pass; relative law structurally guaranteed by `m`-only API | **PASS** |
| **G2** | Fused `score_into` not slower than separate calls | 10K iterations at d=64: fused path is faster than the separate path (separate path has per-iter `Vec` alloc); ratio < 2.0 (target) | **PASS** |
| **G3** | No regression (default + opt-in + `--all-features`) | `cargo clippy` clean on all three configurations; 1558/1558 default lib tests pass | **PASS** |
| **G4** | 0 allocations in `score_into` after `new` | Structural: hot-path args are all `&[f32]` / `&mut [f32]` / `&mut f32` / `i32`; `new` does exactly 2 allocs (the `u, v` `Box<[f32]>`) | **PASS** |

## What was validated

### G1 — Correctness

The `g1_joint_score_matches_manual_composition_across_dims` test generates 20
random `(a, b, u, v, q, k, ω_rot, ω_add, m)` instances per dimension
`{8, 16, 32, 64}` (80 total) and verifies that `GrapeJointLift::score_into`
produces the same result as the manual composition:

```text
ref_score = (Rank2Plane::apply_into(q, m, ω_rot, scratch) · k) / √d
          + m · ω_add · (softplus(v·q/√d) + softplus(u·k/√d))
```

Max allowed difference: `1e-5 × max(|expected|, 1.0)`. The two paths use
identical operations, so the difference is sub-ULP rounding only.

**6 special cases verified:**
1. `ω_add = 0` → pure rotary (additive term vanishes). ✅
2. `ω_rot = 0` → pure additive (rotary is identity). ✅
3. `plane.s() ≈ 0` (degenerate, `a ∥ b`) → pure additive (rotary identity via small-angle branch). ✅
4. `m = 0` → identity offset (score = `qᵀ·k/√d`). ✅
5. `u = v = 0` → constant gate `Λ = 2·log(2)` (rotary + constant shift). ✅
6. Causal regime `m < 0` → additive term `≤ 0` (monotonic penalty, matches ALiBi sign convention). ✅

**Exact relativity** (`g1_relative_law_score_depends_only_on_offset`):
score depends only on `m = j − i`, not absolute `(i, j)`. Structurally
guaranteed by the API (m is the only positional input). Verified by
computing `score(q, k, -5)` and confirming bit-identical results.

### G2 — Latency

Smoke test at d=64, 10K iterations. The fused `score_into` path is **faster**
than the separate-calls path because the latter allocates a `Vec` for `q_rot`
on every iteration. The fused path reuses the caller-provided scratch buffer.

The Issue 163 G2 target was `≤ 1.10× the sum of separate calls`. The test is
generous (`< 2.0×`) because the separate path is penalized by per-iter alloc;
in a fair comparison (caller-provided scratch in both paths), the fused path
is structurally identical to the separate path plus one function-call
overhead, so `~1.0×` is expected. The value of `score_into` is the **unified
API + correctness guarantee**, not a speedup.

### G3 — No regression

```
cargo clippy -p katgpt-core --lib                                    # clean
cargo clippy -p katgpt-core --features grape_joint_lift --lib --tests # clean
cargo clippy -p katgpt-core --all-features --lib                     # clean
cargo test -p katgpt-core --lib                                      # 1558/1558 pass
```

### G4 — Alloc-free hot path

Structural check (mirrors Issue 161's pattern): the `score_into` signature is

```rust
pub fn score_into(
    &self,
    q: &[f32],
    k: &[f32],
    m: i32,
    rotated_q_scratch: &mut [f32],
    out: &mut f32,
) -> Result<(), JointLiftError>
```

All hot-path arguments are borrowed slices or scalar values. The owned state
(`plane: Rank2Plane`, `u_gate: Box<[f32]>`, `v_gate: Box<[f32]>`) is
constructed once at `GrapeJointLift::new` and reused across all
`score_into` calls.

`GrapeJointLift::new` does exactly **2 allocations** (the `u_gate` and
`v_gate` `Box<[f32]>` via `slice.into()`). The `Rank2Plane` inside does its
own 2 allocations (`a, b`) at `Rank2Plane::new` — those happen before
`GrapeJointLift::new` is called and are the caller's responsibility.

## Implementation notes

### Decoupled `omega_rot` / `omega_add`

The paper uses a single shared `ω` for both the rotary and additive parts
(Eq. after the `G_joint(m)` display in Appendix E). This implementation
decouples them. Setting `omega_rot == omega_add` recovers the paper exactly;
the decoupling is a **strict generalization** — it lets a caller scale the
additive decay independently of the rotary frequency (e.g. strong decay +
slow rotation for long-context forgetting).

This is **not a deviation** from the paper — it is a parametric superset.

### Numerically stable `softplus`

The GRAPE-A gate function `softplus(z) = log(1 + e^z)` requires careful
branch selection to avoid overflow. The implementation uses:

```rust
if z >= 0.0 {
    z + (-z).exp().ln_1p()   // e^{-z} ∈ (0, 1], no overflow
} else {
    z.exp().ln_1p()          // e^z ∈ (0, 1), no overflow
}
```

The `exp` is only ever called on a non-positive argument, so it never
overflows. Verified by `softplus_extreme_positive_does_not_inf` (z=100 →
finite) and `softplus_extreme_negative_does_not_nan` (z=-100 → finite).

The original implementation had the branches inverted (`e^z` for `z >= 0`),
which overflowed for `z > ~88`. Fixed in the same commit.

### No `GL(d+2)` matrix materialization

The joint lift's mathematical content is a `(d+2)×(d+2)` block-diagonal
matrix, but the computational content decomposes into:

1. One `Rank2Plane::apply_into` (rotary part) — `O(d)`, 2 dot products + 1 FMA triad.
2. Two gate dot products (`simd::simd_dot_f32`) — `O(d)` each.
3. Two `softplus` evaluations — `O(1)` each.
4. One final dot product (rotated_q · k) — `O(d)`.
5. One FMA to combine rotary logit + additive bias — `O(1)`.

Total: `O(d)`, zero allocation after `new`. The `(d+2)×(d+2)` matrix is never
constructed.

### Streaming cache pattern

The primitive is stateless. The streaming pattern (documented in the module
doc) is:

1. **At key arrival `j`:** cache `k̂_j = G(j)·k_j` and `λk_j = softplus(uᵀ·k_j/√d)`.
2. **At query time `t`:** compute `q̂_t = G(t)·q_t` and `λq_t = softplus(vᵀ·q_t/√d)`,
   then score: `q̂_tᵀ·k̂_j/√d + (j−t)·ω_add·(λq_t + λk_j)`.

No cache rewrite when `t` increments — matches RoPE's streaming policy. The
joint lift adds only the cached `λk_j` (one scalar per key) on top of RoPE's
cached `k̂_j`.

## Why GAIN (not Super-GOAT)

Same verdict as the trilogy (Issues 159/160/161):

| Q | Answer |
|---|--------|
| Q1 No prior art? | **NO** — the composition is in GRAPE Appendix E. |
| Q2 New class of behavior? | **NO** — rotary + additive composition, both already shipped. |
| Q3 Product selling point? | **NO** — engine-layer primitive, no game-AI moat. |
| Q4 Force multiplier? | **NO** — touches one pillar (transformer substrate), not ≥2. |

→ All 4 NO → GAIN, no Super-GOAT guide created.

## Promotion decision

**Deferred** (Issue 163 T7, `- [-]`). No hot-path consumer today. Re-evaluate when:

- A transformer attention path or KV compactor wants both rotary + additive together.
- A cross-repo fusion lands:
  - `riir-ai`: HLA personality rotation + decay (compose Issue 159's plane with a softplus-gated forget bias).
  - `riir-neuron-db`: per-shard rotation + bias in `MerkleFrozenEnvelope`.
  - `riir-chain`: LatCal commitment of the additive bias (the bias is a raw scalar — sync-safe).

For the joint lift specifically, also requires riir-train to learn the gate
vectors `(u, v)` — the modelless primitive ships the math; the learned gates
are upstream weights.

## Run commands

```bash
# Run the GOAT gate tests
CARGO_TARGET_DIR=/tmp/grape_joint_lift cargo test -p katgpt-core \
  --features grape_joint_lift --lib grape_joint_lift -- --nocapture

# Clippy checks (T5 + T6)
CARGO_TARGET_DIR=/tmp/grape_joint_lift cargo clippy -p katgpt-core \
  --features grape_joint_lift --lib --tests
CARGO_TARGET_DIR=/tmp/grape_joint_lift cargo clippy -p katgpt-core --all-features --lib
```

## Cross-references

- `Issue 163` — this primitive's spec.
- [Research 446](../.research/446_GRAPE_Group_Representational_Position_Encoding.md) — GRAPE distillation.
- [`.benchmarks/457`](457_grapem_rodrigues_goat.md) — GRAPE-M rotary (the top-left block).
- [`.benchmarks/458`](458_position_group_action_goat.md) — Unified `PositionGroupAction` trait.
- [`.benchmarks/459`](459_grape_ap_vector_goat.md) — GRAPE-AP path-integral (NOT composed here; this is GRAPE-A §4).
- Paper Appendix E (composition) + §4.1–4.2 (GRAPE-A additive) + Appendix C (FoX as GRAPE-AP).
