# Plan 322 — Phase-Modulated Subspace Rotation Gate GOAT Gate

**Date:** 2026-06-25
**Bench:** `katgpt-rs/crates/katgpt-core/benches/bench_322_phase_rotation_goat.rs` (`harness = false`, `std::time::Instant`, direct binary launch bypassing the dyld/trustd stall)
**Primitive:** `katgpt-rs/crates/katgpt-core/src/phase_rotation.rs` (feature `phase_rotation_coupling`)
**Research:** [`katgpt-rs/.research/305_Phase_Modulated_Cross_Domain_Coupling.md`](../.research/305_Phase_Modulated_Cross_Domain_Coupling.md)
**Source paper:** [arXiv:2605.12700](https://arxiv.org/abs/2605.12700) — UFO: Domain-Unification-Free Operator Framework (Qiao, Karniadakis, Muniruzzaman, May 2026)
**Status:** ✅ **ALL GATES PASS — PROMOTED to DEFAULT-ON** (Plan 322 Phase 2, 2026-06-25)

---

## TL;DR

All 5 GOAT gates pass with comfortable headroom. The primitive is **pure modelless** (closed-form `cos`/`sin`/`sigmoid`/`dot`, no training), so per the AGENTS.md "GOAT pass + modelless gain → promote to default" rule, `phase_rotation_coupling` is now in the `default` feature list.

**Design pivot during implementation:** the plan called for an independent polynomial Padé cos/sin approximation (mirroring Plan 319 Issue 003's SiLU/tanh Padé). The first attempt at independent Padé cos/sin coefficients showed Pythagorean-identity drift of `~5e-3` (50× the G1 `<1e-4` budget) — the cos and sin approximations individually fit libm to ~5e-3, but their squared sum drifted much more because the errors didn't cancel. The fix is `phase_safe_cos_sin`: compute `sin(α)` via libm, then recover `cos(α) = sqrt(1 - sin²α)`. This forces the Pythagorean identity to hold bit-by-bit (drift dropped to `5.96e-8`, 1677× under budget) at the cost of one `sqrt` per channel (~3 ns). Net latency is still well under budget (D=64 per-channel: 355.7 ns vs 1500 ns target). The `use_pade` API toggle was dropped — there is one path now; a future Phase 3 SIMD sin-LUT variant would land as a new entry point, not a bool param.

---

## Gate results

| Gate | Target | Result | Headroom |
|------|--------|--------|----------|
| **G1** per-channel Pythagorean drift | `< 1e-4` | **5.96e-8** | 1677× |
| **G1** scalar libm drift (informational) | `< 1e-4` | **1.19e-7** | 840× |
| **G1** `‖out‖² ≤ ‖a‖² + ‖b‖²` bound violation | `< 1e-4` | **0.00e0** | ∞ |
| **G2** monotone interpolation reversals | `0` (tol 1e-5) | **0 reversals** across 100-step sweep | ✅ |
| **G3** D=8 scalar phase + mix latency | `< 50 ns` | **18.9 ns** | 2.6× |
| **G3** D=8 mix-only (cos/sin precomputed) | `< 20 ns` | **5.0 ns** | 4× |
| **G3** D=64 per-channel phase + mix latency | `< 1500 ns` | **355.7 ns** | 4.2× |
| **G4** allocations / 100 steady-state calls | `0` | **0** | ✅ |
| **G6** sigmoid(0) = 0.5 → cos α = sin α = 1/√2 | matches 1/√2 | **0.7071** | ✅ |

All gates PASS. Promotion to default-on per AGENTS.md "Feature Flag Discipline" rule 4 (all gates pass AND gain is modelless → promote).

---

## Methodology

### G1 (norm preservation)

Three measurements across an α sweep in `[0, π/2]` (1000 steps):

1. **Per-channel Pythagorean drift**: route through `compute_phase_per_channel_into` (which uses `phase_safe_cos_sin` — the G1-critical Pythagorean-recovery path). For each α, compute `|cos²α + sin²α - 1|`. Max across the sweep: **5.96e-8**.
2. **Scalar libm drift (informational)**: route through `compute_phase_from_projection` (which uses libm cos/sin directly — the scalar phase path). Same identity check. Max: **1.19e-7**.
3. **`‖out‖² ≤ ‖a‖² + ‖b‖²` bound**: random non-orthogonal `(a, b)` halves, sweep α, verify the Cauchy-Schwarz + sin²+cos²=1 bound holds at every α. Max violation: **0** (the bound is a mathematical identity for all α, not just statistical).

### G2 (smooth interpolation)

HLA-scale `D = 8` halves: `a = [1, 0, 0, 0, 0, 0, 0, 0]`, `b = [0, 1, 0, 0, 0, 0, 0, 0]`. Sweep α ∈ [0, π/2] in 100 steps, compute `cos_sim(out, a)` and `cos_sim(out, b)`. Assert `sim_a` is non-increasing and `sim_b` is non-decreasing within a 1e-5 tolerance. **0 reversals** observed.

### G3 (latency)

Batched-median timing: 1024 calls per batch × 256 batches, take the median batch time, divide by 1024. Anti-hoist: every input is wrapped in `std::hint::black_box` (via the `bb()` helper) so the compiler cannot lift the dot/sigmoid/cos/sin out of the loop, and the output is black-boxed to force the write.

Three configurations:

- **D=8 scalar + mix**: full hot path — `compute_phase_from_projection` (1 SIMD dot + sigmoid + cos + sin) + `phase_rotation_gate_into` (8 FMA via 4-wide chunking). **18.9 ns**.
- **D=8 mix-only**: precomputed cos/sin, just the FMA mix. **5.0 ns** — confirms the mix kernel itself is sub-10ns; the scalar phase construction dominates the D=8 budget (the dot + sigmoid + cos + sin ≈ 14 ns).
- **D=64 per-channel + mix**: cold path — `compute_phase_per_channel_into` (64 × libm sin + Pythagorean sqrt) + per-channel mix. **355.7 ns** — 4.2× under the 1500 ns libm-path budget. Confirms `phase_safe_cos_sin` (libm sin + sqrt) fits well within budget; Phase 3 SIMD/LUT work is not needed.

### G4 (zero-alloc)

`#[global_allocator] CountingAllocator` wraps `std::alloc::System` and counts every `alloc()` call. Warm up both hot paths (10 iterations), then measure 100 steady-state calls through `phase_rotation_gate_into` + `compute_phase_per_channel_into`. **0 allocations** — the scratch is reused, no `Vec::new`/`vec![]`/`Vec::clone` on the hot path.

### G6 (sigmoid never softmax)

Behavioral static check: at `dot = ⟨state, direction⟩ = 0`, `sigmoid(0) = 0.5` → `α = 0.5 · π/2 = π/4` → `cos α = sin α = 1/√2 ≈ 0.7071`. **Softmax of a single value would give 1.0**, not 0.5. Asserting `cos α = sin α ≈ 0.7071` at zero projection proves sigmoid is used.

Result: `cos α = sin α = 0.7071 ≈ 1/√2`. **G6 PASS.**

---

## The design pivot (independent Padé vs Pythagorean sqrt recovery)

The plan specified an independent polynomial Padé cos/sin approximation (mirroring Plan 319 Issue 003's SiLU/tanh Padé pattern). The first implementation used hand-fit minimax coefficients:

```rust
// Original (FAILED G1):
let cos_num = 1.0 - 0.4999_997 * x2 + 0.0416_573 * x4 - 0.0013_589 * x6;
let cos_den = 1.0 + 0.001_309 * x2 + 0.000_152 * x4 + 0.000_018 * x8;
let sin_num = x * (1.0 - 0.1660_517 * x2 + 0.0076_254 * x4);
let sin_den = 1.0 + 0.000_873 * x2 + 0.000_061 * x4;
```

The cos and sin each fit libm to ~5e-3 abs error (within budget). But the G1-critical `cos²α + sin²α = 1` identity drifted by **0.00476** — 47× the `<1e-4` budget. The reason: when `cos` and `sin` are approximated independently, their squared-sum errors compound rather than cancel.

The fix is `phase_safe_cos_sin`:

```rust
// Final (PASSES G1 trivially):
let sin_v = x.sin().clamp(0.0, 1.0);          // libm sin, ~1 ULP
let cos_v = (1.0f32 - sin_v * sin_v).sqrt();  // Pythagorean recovery
```

This forces `cos² + sin² = 1` to hold to f32 rounding noise (the only error is in the single rounding of `1 - sin²` and the `sqrt`). G1 drift dropped from `4.76e-3` → `5.96e-8` (**80× improvement**).

The cost is one `sqrt` per channel (~3 ns on aarch64). Net latency impact: D=64 per-channel is 355.7 ns — still 4.2× under the 1500 ns budget. The `use_pade` API toggle was dropped because there is now only one path; a future Phase 3 SIMD sin-LUT variant (if G3 ever marginally fails) would land as a new entry point `compute_phase_per_channel_simd_into`, not a bool param.

**Honesty note:** the original plan claimed "polynomial-Padé cos/sin (reuse Plan 319 Issue 003 approximation, max error 4.9e-3)". This was wrong — Plan 319 Issue 003's Padé is for SiLU/tanh, NOT cos/sin, and there is no directly reusable cos/sin Padé in the codebase. The Pythagorean-sqrt approach is simpler, more accurate in the G1-critical identity, and fits the latency budget. The plan's Phase 3 SIMD/LUT optimization is now unnecessary for the latency budget (355.7 ns ≪ 1500 ns); it would only matter if a future hot-path caller needed < 600 ns (the original Padé-path target), at which point a SIMD sin-LUT could land as `compute_phase_per_channel_simd_into`.

---

## Why this primitive matters (Super-GOAT candidate)

Per Research 305 §2.3, this is the **genuinely-new operation class** in the codebase. Every other latent op is one of:

| Cousin | Operation | Why phase-rotation is different |
|--------|-----------|--------------------------------|
| Latent Field Steering (Plan 309) | `s + α·v` (additive) | Inflates L2 norm by `‖α·v‖`. Phase-rotation preserves L2 by construction. |
| CommittedFieldBlend (Plan 321) | `Σ sigmoid(π_k)·f_k` (convex combo) | Output norm varies with independent sigmoid weights — no Pythagorean identity. Phase-rotation uses cos/sin AS the weights. |
| HLA projection | dot-product | Read-only scalar projection, not a mix. |
| Clifford Geometric Product (Plan 319) | `u·v + u∧v` (wedge detection) | Detects rotational structure as a signal. Phase-rotation is the actuator (applies rotation); Clifford is the sensor. |
| Cross-Resolution Transport (Plan 310) | `Ψ · C · Φ^T` (linear Tikhonov) | Linear projection preserves L2 only approximately. Phase-rotation preserves L2 exactly. |
| DEC operators | spatial sums on cochains | Per-cell spatial operations, not per-instance latent mixes. |

The Pythagorean identity `sin²α + cos²α = 1` gives the headline invariant:

```
‖out‖² = cos²α·‖a‖² + sin²α·‖b‖² + 2·cos α·sin α·⟨a, b⟩
       ≤ ‖a‖² + ‖b‖²            (Cauchy-Schwarz + sin²+cos²=1, for all α)
```

This per-α L2 bound is what the Super-GOAT selling point depends on: NPCs whose affect rotates between combat and social subspaces over thousands of ticks without magnitude drift — "emotional stability by construction, not by regularization" (Research 305 §3, fusion F1).

---

## Verification

| Check | Result |
|-------|--------|
| `cargo check -p katgpt-core --features phase_rotation_coupling` | ✅ PASS |
| `cargo check -p katgpt-core --all-features` | ✅ PASS |
| `cargo check -p katgpt-core` (default, post-promotion) | ✅ PASS |
| `cargo check -p katgpt-core --no-default-features` | ✅ PASS |
| 19/19 unit tests (`phase_rotation` module) | ✅ PASS (direct binary launch bypassing dyld stall) |
| GOAT bench G1 (norm preservation) | ✅ PASS (per-channel drift 5.96e-8, bound violation 0) |
| GOAT bench G2 (smooth interpolation) | ✅ PASS (0 reversals / 100 steps) |
| GOAT bench G3 (latency) | ✅ PASS (D=8 scalar+mix 18.9ns, D=8 mix-only 5.0ns, D=64 per-channel+mix 355.7ns) |
| GOAT bench G4 (zero-alloc) | ✅ PASS (0 allocs / 100 calls) |
| GOAT bench G6 (sigmoid never softmax) | ✅ PASS (cos=sin=0.7071 at dot=0) |

---

## Repro

```bash
# Run the GOAT bench (direct binary launch bypasses the cargo-bench dyld/trustd stall):
cargo bench -p katgpt-core --features phase_rotation_coupling --bench bench_322_phase_rotation_goat --no-run
BIN=target/release/deps/bench_322_phase_rotation_goat-<hash>
"$BIN"   # may need 3-8 retries due to intermittent trustd saturation

# Run the unit tests:
cargo test -p katgpt-core --features phase_rotation_coupling --lib phase_rotation --no-run
BIN=target/debug/deps/katgpt_core-<hash>
"$BIN" phase_rotation --test-threads=1
```

Apple Silicon arm64, release profile, 2026-06-25.

---

## Cross-references

- [Plan 322](../.plans/322_phase_modulated_coupling_primitive.md) — implementation plan
- [Research 305](../.research/305_Phase_Modulated_Cross_Domain_Coupling.md) — distillation + 4-cousin comparison + 4-fusion analysis
- [riir-ai/.research/159](../../riir-ai/.research/159_phase_rotation_subspace_gate_guide.md) — private Super-GOAT selling-point guide (game runtime domain)
- Plan 319 Issue 003 — the SiLU/tanh Padé pattern that inspired the (failed) independent-Padé approach
- Plan 309 / 310 — the frozen-direction-vector + sigmoid-projection pattern reused here
