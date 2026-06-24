# Issue 007 — WASM SIMD128 Coverage Gap (katgpt-core/simd/)

[← Index](../README.md) · **Type:** optimization · **Priority:** medium · **Status:** open

## Summary

Only `simd_ternary_matvec` has a real `wasm32` SIMD128 kernel. Every other
SIMD operation in `crates/katgpt-core/src/simd/` falls back to scalar on
`wasm32` even when compiled with `-C target-feature=+simd128`. This caps
browser / Cloudflare Worker inference throughput well below what the
hardware can deliver.

Research 226's four-tier dispatch ("AVX2 → NEON → WASM simd128 → scalar")
is currently realized **only for the ternary path**. Every other kernel
silently degrades to the bottom tier on wasm32.

## Evidence — coverage matrix

Grepped `target_arch = "wasm32"` per file in `crates/katgpt-core/src/simd/`
on 2026-06-24 (develop, post-Plan 316):

| File | wasm32 | aarch64 | x86_64 | Gap? |
|------|--------|---------|--------|------|
| `ternary.rs` | ✅ 3 | ✅ 3 | ✅ 3 | No — full tier |
| `mod.rs` | ✅ 2 | ✅ 2 | ✅ 3 | No (level detection only, not a hot kernel) |
| `activations.rs` | ❌ 0 | ✅ 15 | ✅ 15 | **Yes** — sigmoid/exp fall to scalar on wasm32 |
| `dot.rs` | ❌ 0 | ✅ 12 | ✅ 9 | **Yes** — `simd_dot_f32` scalar on wasm32 |
| `elementwise.rs` | ❌ 0 | ✅ 27 | ✅ 27 | **Yes** — biggest gap (27 NEON/AVX kernels, 0 WASM) |
| `argmax.rs` | ❌ 0 | ✅ 3 | ❌ 0 | **Yes** — no wasm32, also no x86_64 |
| `sparse.rs` | ❌ 0 | ✅ 3 | ✅ 3 | **Yes** — sparse kernels scalar on wasm32 |
| `research.rs` | ❌ 0 | ✅ 18 | ✅ 18 | **Yes** — research kernels scalar on wasm32 |

**7 of 8 kernel files have zero wasm32 SIMD128 coverage.**

## Impact

The freeze/thaw egg/shell vessel (Plan 316) compiles for browser and CF
Worker, but inference throughput on those targets is bottlenecked by the
missing kernels:

- **Browser node** (`riir-chaind chain_node_browser`): ternary matvec is
  fast, but every activation / dot / elementwise op around it is scalar.
- **CF Worker** (`seal-edge-worker`): same — the ternary kernel alone
  doesn't carry the full inference pipeline. Workers have a 30s CPU budget
  (higher for Durable Objects); scalar fallbacks eat into that budget
  disproportionately.
- **Doc 56 edge architecture**: the CF edge design assumes WASM SIMD128
  across the board. Currently only the ternary spine delivers it.

## Why this is an issue, not a plan

Per repo rules: "Create issue at ./issues for optimization task, do not
create plan." This is a perf optimization (fill in missing kernels), not
a feature or bug. Promotion of any individual kernel should go through
the GOAT gate (G1 correctness, G2 perf bench, G3 no-regression, G4
alloc-free) before default-on.

## Proposed approach (per kernel, GOAT-gated)

Each kernel is independent — file a separate sub-task per kernel:

1. Port the aarch64 NEON kernel to `core::simd::Simd` with the
   `target_feature = "simd128"` cfg gate (same pattern as `ternary.rs`).
   `core::simd` is the portable SIMD API that lowers to WASM simd128
   intrinsics under the hood.
2. Write a benchmark: native-scalar vs wasm32-simd128 (cycle count or
   wall-clock on a representative input size).
3. Run the GOAT gate:
   - **G1 correctness**: bit-exact match vs scalar reference on the same
     input (the ternary kernel's test pattern is the template).
   - **G2 perf**: wasm32-simd128 must beat wasm32-scalar by a measurable
     margin (target: ≥ 2× on 4-wide f32, the WASM SIMD128 lane width).
   - **G3 no-regression**: native builds unaffected (the wasm32 kernel is
     cfg-gated and never compiled on aarch64/x86_64).
   - **G4 alloc-free**: no `Vec` / `Box` in the hot path — same rule as
     every other SIMD kernel.
4. If all gates pass AND the gain is modelless → the kernel is available
   behind `target_feature = "simd128"` (no feature flag needed — it's a
   target-feature gate, not a cargo feature).

## Priority ordering (by hot-path frequency)

Suggested order based on inference pipeline hot loops:

1. **`dot.rs` → `simd_dot_f32`** — used in every matvec / matmul. Highest
   blast radius. (ternary already covered, but dense dot is the next most
   called.)
2. **`activations.rs` → `simd_sigmoid_*`, `simd_exp_*`** — sigmoid is the
   user-mandated gate function (per AGENTS.md: "Use sigmoid not softmax").
   Every layer applies it.
3. **`elementwise.rs`** — 27 kernels, broad surface. Lower individual
   impact but high aggregate impact.
4. **`argmax.rs`** — used in sampling / decoding. Less frequent than dot
   but still hot.
5. **`sparse.rs`** — sparse matvec; matters iff sparse shards ship to edge.
6. **`research.rs`** — research-only kernels; defer unless a research plan
   needs them on wasm32.

## Non-goals

- Not adding `wasm32-unknown-emscripten` or `wasm64-unknown-unknown` support.
- Not changing the dispatch architecture — `SimdLevel::WasmSimd128` already
  exists and is correctly detected (`mod.rs:129`). The gap is purely
  missing kernel implementations, not missing dispatch.
- Not touching the ternary kernel — it's already covered.

## References

- Plan 316: `.plans/316_wasm32_three_target_unblock.md` (unblocked the
  wasm32 compile surface; this issue is the perf follow-up).
- Research 226: four-tier dispatch design (AVX2 → NEON → WASM simd128 →
  scalar). Ternary is the only realized tier on wasm32.
- `crates/katgpt-core/src/simd/ternary.rs:338-342` — the template for how
  a wasm32 SIMD128 kernel should be cfg-gated and structured.
- Doc 56 (`riir-ai/.docs/56_cf_workers_edge_architecture.md`) — the CF
  edge design that assumes wasm32 SIMD128 across the inference stack.

## TL;DR

7 of 8 SIMD kernel files in `katgpt-core/src/simd/` have zero `wasm32`
coverage — only `ternary.rs` ships a real SIMD128 kernel. Every other op
(dot, sigmoid, elementwise, argmax, sparse, research) falls to scalar on
browser / CF Worker even with `+simd128`. This caps inference throughput
on the edge targets that Plan 316 just unblocked. File per-kernel
GOAT-gated ports using `core::simd` with the `target_feature = "simd128"`
gate, prioritized by hot-path frequency: dot → activations → elementwise
→ argmax → sparse → research.
