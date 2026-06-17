# Bench 059: Sink-Aware Attention GOAT Gate — Status

**Date:** 2026-06-17
**Plan:** [287_sink_aware_attention](../.plans/287_sink_aware_attention.md)
**Research:** [258_Attention_Sink_Dual_Mechanism_NOP_Broadcast](../.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md)
**Paper:** [arxiv 2606.08105](https://arxiv.org/abs/2606.08105) — Fesser et al., *A Unifying View of Attention Sinks: Two Algorithms, Two Solutions*
**Feature flag:** `sink_aware_attn` (opt-in, implies `data_probe`. **NOT in default features** — G3 latency overhead missed target.)
**Status:** Phase 1 + Phase 2 + Phase 3 (standalone gate) shipped; G1 PASS; G2 synthetic PASS; G3 latency FAIL; promotion DEFERRED.

---

## Summary

Shipped the per-head sink classifier (`SinkKind`, `SinkDiagnostic`,
`SinkClassifierConfig`, `StableRankScratch`, `classify_sink_at`,
`classify_all_sinks`, `stable_rank_update_into`) plus the dual-policy gate
(`SinkAwarePolicy`, `apply_dual_policy_gate`) as an opt-in diagnostic
primitive under the `sink_aware_attn` feature. The classifier lives in
`crates/katgpt-core/src/data_probe.rs`; the root crate re-exports at
`katgpt_rs::data_probe::sink_classify`.

**NOT promoted to default features.** G1 (correctness) and the synthetic G2
(Broadcast preservation) pass, but G3 (latency overhead) is far over the
5% target. Default `SinkAwarePolicy::Uniform` stays; `DualPolicy` remains
a research-grade opt-in.

---

## Gate Status

| Gate | Description | Status | Notes |
|------|-------------|--------|-------|
| **G1** | Classifier correctness on synthetic heads | ✅ PASS | 8/8 unit tests in `src/data_probe/sink_classify.rs`: NOP-only, Broadcast-only, mixed (both threshold variants), no-sink, zero-attn-column edge, degenerate-values edge, zero-matrix stable-rank. All edge cases handled without crash or NaN. |
| **G2** | DualPolicy preserves Broadcast value info vs Uniform | ✅ PASS (synthetic) | 2/2 tests in `tests/sink_aware_g2_synthetic.rs`: Broadcast head — DualPolicy classifies as Broadcast → output == O unchanged; NOP head — DualPolicy classifies as NOP → output = O · σ(gate_scale). Uniform copies unchanged for both. |
| **G2** (real ViT) | `effective_rank` preserved/improved on frozen ViT | ⏳ DEFERRED | Requires a real model + per-layer hook. Out of scope for this coding task. Synthetic G2 is the substitute. |
| **G3** | Latency overhead ≤5% (DualPolicy vs Uniform) | ❌ **FAIL** | 1671% at n=128, d_h=64; 5266% at n=512, d_h=64. Root cause: per-call classifier work is O(n² + n·d + d²); the comparison baseline (`Uniform`) is a single copy. See "Latency analysis" below. |
| **Promote to default** | G2 + G3 both pass | ❌ DEFERRED | G3 missed by ~3 orders of magnitude. Default stays `Uniform`. Demoted to opt-in diagnostic until optimization. |

---

## Phase 1 deliverables (DONE)

- ✅ T1.1 — `sink_aware_attn` feature added to `katgpt-rs/Cargo.toml` and `katgpt-rs/crates/katgpt-core/Cargo.toml`. `data_probe` extended to imply `katgpt-core/sink_aware_attn`. Root crate exposes module at `katgpt_rs::data_probe::sink_classify`.
- ✅ T1.2 — Types: `SinkKind` (`#[repr(u8)]`, default `None`), `SinkDiagnostic` (all fields pub), `SinkClassifierConfig` (defaults: 0.5, 0.2, 0.5, 1.5, 1.5), `StableRankScratch` (`new`, `ensure_capacity`).
- ✅ T1.3 — `classify_sink_at(position, attn_column, values, update_O, cfg, scratch) -> SinkDiagnostic`. SIMD strength + value-norm via `simd_sum_f32` / `simd_dot_f32`. Decision rule matches Research 258 §2.1.
- ✅ T1.4 — `classify_all_sinks(attn, values, cfg, scratch, out)`. Caller-owned `out`; single n-length allocation per call.
- ✅ T1.5 — 8 G1 unit tests pass (see G1 row above).

## Phase 2 deliverables (DONE — target missed, documented)

- ✅ T2.1 — `stable_rank_update_into(O, scratch, n_iters) -> f32`. Zero-alloc on the scratch path; one n-length local buffer for the matvec intermediate.
- ✅ T2.2 — SIMD via `simd_dot_f32` + `simd_fused_scale_acc` inside the two-pass matvec decomposition (avoids materializing `Oᵀ·O`).
- ✅ T2.3 — Early-exit at `σ_1² > 0.95 · trace(F)` (rank-1 Broadcast fast path).
- ✅ T2.4 — Bench file `benches/sink_classify_bench.rs`. **Target <1µs for n=32, d_h=64 NOT MET**: 1.71µs for random `O`, 0.79µs for rank-1 `O` (early-exit). See "Latency analysis" below.
- ✅ T2.5 — Numerical robustness: all-zero matrix → 0.0 (no NaN). Covered by `g1_stable_rank_zero_matrix`.

## Phase 3 deliverables (DONE — scope-reduced per validation fallback)

- ✅ T3.1 — `SinkAwarePolicy` enum shipped in `crates/katgpt-core/src/data_probe.rs`. **Scope reduction:** NOT wired into `ParallaxConfig` / `FuncAttnConfig` (would break backwards-compat for `Default` impls and add feature-gate complexity to the forward paths). Standalone path only.
- ✅ T3.2 — `apply_dual_policy_gate(attn, values, O, policy, gate_scale, scratch, out) -> SinkKind`. Standalone post-forward intervention. Classifies dominant sink; gates if NOP, copies if Broadcast/None.
- ✅ T3.3 — Same `SinkAwarePolicy` enum + gate covers both parallax and funcattn paths (it's policy-agnostic). The funcattn-specific "scale Φ residual contribution" variant is not implemented — `apply_dual_policy_gate` operates on the post-`AV` output `O`, which is the same for both parallax and funcattn.
- ✅ T3.4 — Synthetic G2 test `tests/sink_aware_g2_synthetic.rs` — 2/2 PASS. Real-ViT G2 DEFERRED.
- ✅ T3.5 — Latency bench `benches/sink_aware_latency_bench.rs`. **G3 FAIL**: 1671% / 5266% overhead.
- ✅ T3.6 — Promotion decision: **DO NOT PROMOTE**. Default stays `Uniform`.

## Phase 4 deliverables (DONE)

- ✅ T4.1 — `LayerSinkSummary` added to `src/data_probe/geometry.rs`. Fields: `layer_index`, `n_nop_sinks`, `n_broadcast_sinks`, `dominant_kind`, `mean_broadcast_value_norm`.
- ✅ T4.2 — `summarize_layer_sinks(attn_per_head, values_per_head, cfg, scratch, layer_index) -> LayerSinkSummary`. Runs classifier across all heads, aggregates.
- ✅ T4.3 — Example `examples/sink_phase_plot.rs`. Synthetic ViT-like activations; layers 0-3 NOP-dominant (zero CLS value), layers 4-7 would-be Broadcast (but `classify_all_sinks` doesn't pass `update_O`, so they show as None — documented in example output).
- ✅ T4.4 — `src/data_probe/mod.rs` docstring updated with "mechanism locator vs aggregate symptom" framing.

## Phase 5 deliverables (DONE)

- ✅ T5.1 — README Feature Showcase entry added (under Attention Matching).
- ✅ T5.2 — Cross-reference added to `.research/100_EGA_Energy_Gated_Attention_Spectral_Salience.md` (EGA + sink-aware = categorical gate).
- ✅ T5.3 — Cross-reference added to `.research/070_Gated_DeltaNet_2_*.md` (GDN2 erase/write = linear-attention dual of NOP/Broadcast).

---

## Latency analysis (G3 FAIL root cause)

Raw numbers from `cargo bench --features sink_aware_attn --bench sink_aware_latency_bench`:

| n | d_h | uniform_us | dual_us | overhead% | kind |
|---|-----|-----------|---------|-----------|------|
| 128 | 64 | 0.71 | 12.54 | 1671% | Broadcast |
| 512 | 64 | 2.96 | 158.75 | 5266% | Broadcast |

### Why so slow?

1. **The comparison is degenerate.** `SinkAwarePolicy::Uniform` is a single n·d copy — the cheapest possible "do something with O". `DualPolicy` does:
   - Build n-length `col_sums` (allocation + n² scan).
   - Argmax over col_sums (n scan).
   - `classify_sink_at` with `Some(O)` → full stable-rank power iteration.
   - Copy or scale O into `out`.
2. **Stable rank is the expensive part.** Phase 2 bench shows `stable_rank_update_into` is 6.13µs at n=128, d_h=64 (random matrix) and 2.63µs (rank-1 fast path). For n=512: 30µs / 12µs respectively.
3. **`Vec<f32>` row-major layout** defeats SIMD. Each `simd_dot_f32(row, v, d)` call has to follow a pointer to a heap-allocated row. A flat `(n*d)`-length slice would let the compiler auto-vectorize across rows.

### What would fix it (future work)

- Skip stable rank when `value_norm_ratio ≤ nop_max` (NOP case doesn't need it — `apply_dual_policy_gate` currently always passes `Some(O)`).
- Cache the `col_sums` buffer in `StableRankScratch` (extend struct to 3 buffers: `v`, `w`, `col_sums`).
- Switch `&[Vec<f32>]` to flat `&[f32]` layout for `O`, `values`, `attn` — eliminates the row-pointer indirection.
- Only run the classifier at all when the caller signals interest (e.g., audit cadence, not every forward).

### Honest framing

The G3 target "≤5% overhead" assumed the classifier could be made cheap enough to run on every head every forward pass. The numbers show that's not feasible without significant additional optimization. The classifier remains useful as:
- An **audit-cadence diagnostic** (run every N forwards, not every forward).
- A **model-analysis tool** (run once on a frozen model to characterize sink behavior).
- A **post-hoc filter** (classify sinks after a forward, then choose policy for the *next* forward).

The primitive ships; the integration is staged. This matches the validation fallback path explicitly described in Plan 287 §Validation.

---

## Stable-rank formula clarification

The plan task text wrote `(Σσ_k)² / Σσ_k²` (nuclear-to-Frobenius ratio) but described the approximation `trace(F)/spectral_norm²` where `trace(F) = Σ‖row_i‖² = Σσ_k²` — which is the **standard stable rank** (Roy-Vetterli 2007, `‖O‖_F² / ‖O‖_op²`). The two formulas differ numerically but agree at the cases the paper cares about (rank-1 → 1.0 for Broadcast; isometry of rank r → r).

We implement the **standard stable rank** because:
1. It matches the prescribed approximation exactly.
2. It only needs the top singular value (cheap power iteration).
3. It is consistent with the Roy-Vetterli definition already shipped in `data_probe/geometry.rs::effective_rank`.

Documented in the module-level doc comment of `crates/katgpt-core/src/data_probe.rs`.

---

## Files

| File | Role | Lines |
|------|------|-------|
| `crates/katgpt-core/src/data_probe.rs` | Primitive: types, classifier, stable-rank, dual-policy gate. Gated `#[cfg(feature = "sink_aware_attn")]`. | ~620 |
| `crates/katgpt-core/src/lib.rs` | `pub mod data_probe;` + re-exports. | +16 |
| `crates/katgpt-core/Cargo.toml` | `sink_aware_attn = []` feature. | +1 |
| `src/data_probe/sink_classify.rs` | Root-crate re-export + 8 G1 unit tests. | ~265 |
| `src/data_probe/mod.rs` | `pub mod sink_classify;` + re-exports + docstring. | +15 |
| `src/data_probe/geometry.rs` | `LayerSinkSummary` + `summarize_layer_sinks`. | +108 |
| `Cargo.toml` | `data_probe` extended; `sink_aware_attn` added; 4 [[bench]]/[[test]]/[[example]] entries. | +6 +30 |
| `benches/sink_classify_bench.rs` | Phase 2 T2.4 bench. | ~200 |
| `benches/sink_aware_latency_bench.rs` | Phase 3 T3.5 bench. | ~140 |
| `tests/sink_aware_g2_synthetic.rs` | Phase 3 T3.4 synthetic G2. | ~225 |
| `examples/sink_phase_plot.rs` | Phase 4 T4.3 example. | ~115 |
| `README.md` | Feature Showcase entry. | +52 |
| `.research/100_EGA_*.md` | Cross-reference. | +2 |
| `.research/070_Gated_DeltaNet_2_*.md` | Cross-reference. | +4 |

---

## Test results

```
$ cargo test --features data_probe -p katgpt-rs --lib data_probe::sink_classify
running 8 tests
test data_probe::sink_classify::tests::g1_degenerate_values_edge ... ok
test data_probe::sink_classify::tests::g1_nop_only_head ... ok
test data_probe::sink_classify::tests::g1_zero_attn_column_edge ... ok
test data_probe::sink_classify::tests::g1_stable_rank_zero_matrix ... ok
test data_probe::sink_classify::tests::g1_broadcast_only_head ... ok
test data_probe::sink_classify::tests::g1_mixed_head ... ok
test data_probe::sink_classify::tests::g1_no_sink_head ... ok
test data_probe::sink_classify::tests::g1_mixed_head_both_above_threshold ... ok

test result: ok. 8 passed; 0 failed

$ cargo test --features data_probe -p katgpt-rs --lib data_probe::
test result: ok. 52 passed; 0 failed   # (44 existing + 8 new — no regressions)

$ cargo test --features sink_aware_attn --test sink_aware_g2_synthetic
running 2 tests
test g2_synthetic_nop_dual_gates_uniform_does_not ... ok
test g2_synthetic_broadcast_dual_preserves_more_than_uniform ... ok

test result: ok. 2 passed; 0 failed
```

---

## Verdict

**DO NOT PROMOTE `sink_aware_attn` to default features.** G1 (correctness)
and the synthetic G2 (Broadcast preservation) pass, but G3 (latency) missed
the ≤5% target by ~3 orders of magnitude. The classifier is a useful
diagnostic — shipped under `data_probe` so it composes with
`effective_rank` and `avg_cosine_similarity` — but running it per-head
per-forward is too expensive with the current implementation.

Promote-to-default criteria for a future iteration:
1. Make `stable_rank_update_into` truly zero-alloc (extend scratch to 3 buffers).
2. Skip stable rank in `apply_dual_policy_gate` when `value_norm_ratio` alone is decisive (NOP fast-path — most heads).
3. Switch to flat `&[f32]` layout for `O` / `values` / `attn` to enable cross-row SIMD.
4. Re-run G3 with these optimizations; target ≤5% at n=128, d_h=64.

Until then, the primitive ships as an opt-in diagnostic. The synthetic G2
validates the *logic* of the dual-policy decision; the latency gap is an
engineering problem, not a fundamental barrier.
