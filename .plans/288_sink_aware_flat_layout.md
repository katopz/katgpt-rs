# Plan 288: Sink-Aware Attention — Flat `&[f32]` Layout Variants

**Date:** 2026-06-18
**Prior work:** [Plan 287](./287_sink_aware_attention.md) (mechanics DONE), `Issue 001` (RESOLVED via cached variant)
**Target:** `crates/katgpt-core/src/data_probe/mod.rs` (extend) + `crates/katgpt-core/src/data_probe/sink_classify.rs` (re-export + tests)
**Status:** Complete. All GOAT gates PASS.

---

## Goal

Add flat `&[f32]` row-major variants of every public sink-aware function so that:

1. **Forward-path integration becomes possible.** `parallax_attn::tiled_attention_parallax_forward` and `funcattn::funcattn_forward` consume flat `&[f32]` row-major tensors (`q`, `k`, `v`, `output`). Today's sink-aware API takes `&[Vec<f32>]`, which forces callers to materialize a `Vec<Vec<f32>>` wrapper — an O(n·d) allocation per call that breaks the zero-alloc property Plan 287 G5 verified.
2. **Cache locality improves.** Flat `&[f32]` is one contiguous allocation; `&[Vec<f32>]` is n separate allocations with arbitrary addresses. The stable-rank power iteration reads every row twice — contiguous reads are friendlier to the prefetcher.
3. **Future SIMD cross-row opportunities open up.** With a flat layout, `O[i]` and `O[i+1]` are adjacent in memory; LLVM can in principle auto-vectorize the inner dot loop across rows once the algorithm is restructured (deferred — this plan only adds the layout variants).

This is explicitly listed as "(Optional) deferred" in Plan 287's closing notes and Issue 001's "next steps". It is the prerequisite for Plan 289 (forward-path wiring).

**Non-goal:** SIMD cross-row kernel restructuring. Cosine probe (Issue 001 T5) already handles the rank-1 Broadcast fast path; cross-row SIMD would help the random-O case but is a larger rewrite. Documented as a follow-up.

---

## Tasks

- [x] **T1** `stable_rank_update_into_flat(o: &[f32], n: usize, d: usize, scratch: &mut StableRankScratch, n_iters: u8) -> f32`. Identical algorithm to [`stable_rank_update_into`]; rows sliced as `o[i*d..(i+1)*d]`.
- [x] **T2** `classify_sink_at_flat(position, attn_column: &[f32], values: &[f32], n: usize, d: usize, update_O: Option<(&[f32], usize, usize)>, cfg, scratch) -> SinkDiagnostic`. Mirrors [`classify_sink_at`].
- [x] **T3** `classify_all_sinks_flat(attn: &[f32], n: usize, values: &[f32], d: usize, cfg, scratch, out)`. Mirrors [`classify_all_sinks`].
- [x] **T4** Private helpers `copy_rows_flat(src, dst, total_len)` and `scale_rows_flat(src, scale, dst, total_len)`. Use `simd::simd_fused_decay_write` for scale (single SIMD pass: `dst = 0·dst + scale·src`).
- [x] **T5** `apply_dual_policy_gate_flat(attn: &[f32], values: &[f32], o: &[f32], n: usize, d: usize, policy, gate_scale, scratch, out: &mut [f32]) -> SinkKind`.
- [x] **T6** `apply_dual_policy_gate_cached_flat(...)` — cached audit-cadence variant, same shape as T5 plus `cached: &mut CachedSinkClassification`.
- [x] **T7** Unit tests in `crates/katgpt-core/src/data_probe/sink_classify.rs` — parity with `Vec<Vec<f32>>` variants on identical inputs. 8 tests covering: rank-1 stable-rank parity, zero-matrix stable-rank, NOP classify parity, Broadcast classify parity, classify_all_sinks parity, gate parity (Broadcast), gate NOP gating, cached-flat audit+reuse.
- [x] **T8** Bench: `benches/sink_aware_latency_bench.rs` extended with `dual_flat` + `cached_flat` columns and two regimes (`rank1`, `random`). Result: flat variants are **1.8×–5.1× faster** than Vec<Vec<f32>> (hypothesis was ≥5%; delivered 80%–410%).
- [x] **T9** Re-exported all 5 new flat symbols from both `crates/katgpt-core/src/data_probe/sink_classify.rs` and `crates/katgpt-core/src/lib.rs`. Also fixed pre-existing gap: `CachedSinkClassification` and `apply_dual_policy_gate_cached` were missing from the katgpt-core lib re-export — now present.
- [x] **T10** Updated `.benchmarks/059_sink_aware_goat.md` with G3-flat row, full numbers, and "why flat is faster" analysis. Cross-referenced Plan 288 as the unblock for Plan 289.

---

## GOAT gate

- **Correctness (G1):** Flat variants must produce **bit-identical `SinkKind` decisions** to the `Vec<Vec<f32>>` variants on identical numerical inputs. Verified by T7 parity tests.
- **Latency (G3):** Flat variants must be **≥ as fast as** Vec<Vec<f32>> variants. Target: ≥5% faster on random-O (non-rank-1) case at n=128, d=64. Failure mode: cache locality doesn't materialize → ship anyway (correctness still holds, no perf regression).

Promotion criterion for Plan 289 (forward-path wiring): flat variants pass G1. Plan 289 then uses flat variants natively — no `Vec<Vec<f32>>` materialization at the parallax/funcattn call sites.

---

## Validation

| Gate | Status |
|------|--------|
| G1 (parity vs Vec<Vec>) | ✅ PASS — 8/8 parity tests pass (`cargo test --features sink_aware_attn,data_probe ... data_probe::sink_classify`), 18/18 total in the module |
| G3 (latency ≥ Vec<Vec>) | ✅ PASS (big margin) — flat is 1.8×–5.1× faster than Vec<Vec<f32>> across rank1 and random regimes |
| cargo check (all features) | ✅ PASS — clean (only 2 pre-existing unrelated warnings) |
| data_probe test suite | ✅ PASS — 18/18 (10 pre-existing + 8 new flat-parity) |
| workspace test suite | ✅ PASS — 3773/3774 (1 pre-existing flaky timing test `bench_grounding_quality_32k`, unrelated, passes in isolation) |

---

## Non-goals (explicit)

- **NOT** restructuring the algorithm for cross-row SIMD. The flat layout enables it but the kernel rewrite is a separate effort.
- **NOT** deprecating the `Vec<Vec<f32>>` API. Both coexist — callers pick whichever matches their data layout. The Vec<Vec<f32>> API stays as the diagnostic-friendly path.
- **NOT** wiring into `ParallaxConfig` / `FuncAttnConfig`. That's Plan 289, gated on this plan's G1.
