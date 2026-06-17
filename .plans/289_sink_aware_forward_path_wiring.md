# Plan 289: Sink-Aware Attention — Forward-Path Wiring into Parallax

**Date:** 2026-06-18
**Prior work:**
- [Plan 287](./287_sink_aware_attention.md) (sink-aware classifier mechanics — DONE, G3 FAIL → opt-in only)
- [Plan 288](./288_sink_aware_flat_layout.md) (flat `&[f32]` variants — DONE, removes layout-mismatch blocker)
- [Issue 001](../issues/001_sink_aware_g3_latency.md) (RESOLVED via cached variant)

**Target:** `crates/katgpt-core/src/parallax_attn.rs` (extend) + `crates/katgpt-core/src/lib.rs` (re-export)
**Status:** Complete. All GOAT gates PASS.

---

## Goal

Compose `parallax_attn::tiled_attention_parallax_forward` with `data_probe::apply_dual_policy_gate_flat` / `apply_dual_policy_gate_cached_flat` into a single entry point so callers can opt into per-head NOP/Broadcast gating without manually:

1. Materializing a `Vec<Vec<f32>>` wrapper around the flat parallax output (the blocker Plan 288 removed).
2. Managing two scratch structs and a retained `n×n` attention matrix by hand.
3. Remembering to forward into a temporary buffer before invoking the out-of-place gate.

**Non-invasive by design.** The existing `tiled_attention_parallax_forward` signature is preserved bit-for-bit (callers untouched). The new entry point is additive, lives behind combined feature gates `parallax_attn + sink_aware_attn`, and short-circuits to vanilla parallax when `SinkAwarePolicy::Uniform` (zero overhead in steady state).

**GOAT gate (deferred from Plan 287):** Forward-path composition must add **≤5% latency overhead** vs vanilla parallax when `policy = Uniform`. For `DualPolicy`, accept the Plan 287 G3-known cost — amortized via cached variant.

---

## Scope decisions (locked before implementation)

### A1. Field-on-config vs separate entry point → SEPARATE ENTRY POINT

Plan 287 scope-reduction #2 flagged that adding `policy: SinkAwarePolicy` to `ParallaxConfig` requires either feature-gating the field (breaks `Debug`/`Clone`/`serde` derives when feature off) or pulling `SinkAwarePolicy` out of `data_probe` (couples unrelated modules).

**Decision:** ship a new function `tiled_attention_parallax_forward_sink_aware`. `ParallaxConfig` stays untouched. Matches the existing standalone-gate pattern Plan 287 established and the SOLID/Decouple rule.

### A2. FuncAttn wiring → NOT APPLICABLE (closed 2026-06-18)

FuncAttn's basis projection `Φ` is `n×k` (k≪n typically), **not** an `n×n` attention map. Sink classification semantics on basis-modes differ fundamentally from token-position sinks (a "sink basis mode" is not a well-studied object).

**Research verdict (Research 261, 2026-06-18):** Wiring is not just deferred — it is **not applicable**. FuncAttn's `out = Φ · C · Ṽ` structure has no token-position sinks because basis modes are partition-of-unity by design (every mode is a "broadcaster" by construction; degenerate modes are a training failure, not a runtime pathology). The NOP/Broadcast classifier collapses into a column-norm check, which is cheaper and belongs in a separate `funcattn_diagnostics` module if ever needed. See [Research 261](../.research/261_FuncAttn_Sink_Semantics_Verdict.md) for the full negative-result analysis.

**Decision:** Plan 289 covers Parallax only. FuncAttn wiring is closed as not-applicable — not postponed. The `SinkAwarePolicy` API is Parallax-specific by design.

### A3. Attention matrix retention → OPTIONAL OUT-PARAM

Parallax forward computes `scores` per-row then accumulates `o_i = Σ_j p_ij · v_j`, discarding the row. Sink classifier needs the full `n×n` matrix. Options were:

- **B1:** wrapper allocates `n²` internally (hidden allocation, plan-287 G5 violation).
- **B2:** new forward variant with `attn_matrix: Option<&mut [f32]>` out-param (caller-owned, zero overhead when `None`).

**Decision:** B2. New `tiled_attention_parallax_forward_retaining` takes the optional out-param; the original `tiled_attention_parallax_forward` delegates with `None` (DRY via internal delegation, zero behavior change for existing callers).

### A4. In-place gate vs out-of-place → OUT-OF-PLACE WITH CALLER TEMP

`apply_dual_policy_gate_flat(o: &[f32], ..., out: &mut [f32])` takes separate read/write slices — Rust borrow checker won't allow `&output` + `&mut output` simultaneously. Options:

- **C1:** add `_inplace` variants to `data_probe.rs` using raw pointers (more API surface, unsafe).
- **C2:** forward into caller-provided `o_temp`, then gate `o_temp → output` (clean, honest about cost).

**Decision:** C2. The wrapper takes a `SinkAwareParallaxScratch` that owns `o_temp` + `attn_matrix` + `classifier_scratch` + optional `cached`. Caller manages one scratch struct, not four loose buffers.

### A5. Uniform short-circuit → YES

When `SinkAwarePolicy::Uniform`, the wrapper **must not** pay for the temporary + n² retention + classifier. It calls vanilla `tiled_attention_parallax_forward` directly into `output`. This is the gate that makes the feature zero-cost when not used.

---

## Tasks

### Phase 1 — Retained-attention forward variant (prerequisite)

- [x] **T1.1** Add `tiled_attention_parallax_forward_retaining` with extra `attn_matrix: Option<&mut [f32]>` parameter. Existing `tiled_attention_parallax_forward` becomes a thin delegator: `tiled_attention_parallax_forward_retaining(..., None, scratch)`. Impl: after `normalize_attention_weights`, if `Some(am)`, `am[i*n..(i+1)*n].copy_from_slice(&scratch.scores[..n])`. Single branch hoisted outside the `j` loop; memcpy is `n` f32s per row. Also threaded through `tiled_attention_core` early-return path (when `gate_scale=0`). *(2026-06-18)*
- [x] **T1.2** Add `debug_assert!` that `attn_matrix.map(|am| am.len()) == Some(n*n)` when `Some`. *(2026-06-18)*
- [x] **T1.3** Unit tests `plan289_retained_attn_matches_per_row_sigmoid` and `plan289_retained_attn_matches_per_row_softmax` — run forward with `Some(am)` on a fixed RNG seed, then independently compute the row-major attention matrix row-by-row via `simd_dot_f32 + normalize_attention_weights`, assert bit-equality. Both activations covered. PASS. *(2026-06-18)*

### Phase 2 — Sink-aware scratch + composition entry point

- [x] **T2.1** Define `pub struct SinkAwareParallaxScratch` (fields: `attn_matrix`, `o_temp`, `classifier: StableRankScratch`, `cached: Option<CachedSinkClassification>`) with `new(n, d)` constructor + `with_cache()` builder + `ensure_capacity(n, d)` fast-path. Lives in `parallax_attn.rs` behind `#[cfg(all(feature = "parallax_attn", feature = "sink_aware_attn"))]`. *(2026-06-18)*
- [x] **T2.2** Define `pub fn tiled_attention_parallax_forward_sink_aware`. Uniform short-circuits to vanilla forward; DualPolicy syncs `cached.cfg` from `policy` (source of truth = policy arg), forwards into `o_temp` with attn matrix retained, then applies flat gate (cached or uncached depending on `sink_scratch.cached`). *(2026-06-18)*
- [x] **T2.3** Re-export `SinkAwareParallaxScratch` + `tiled_attention_parallax_forward_sink_aware` + `tiled_attention_parallax_forward_retaining` from `crates/katgpt-core/src/lib.rs`. *(2026-06-18)*

### Phase 3 — Tests (parity + synthetic G2 + latency G3)

- [x] **T3.1** `plan289_uniform_bit_identical_to_vanilla` — Uniform path bit-identical to vanilla forward. PASS. *(2026-06-18)*
- [x] **T3.2** `plan289_dualpolicy_matches_manual_composition` — DualPolicy path bit-identical to manual `forward_retaining` + `apply_dual_policy_gate_flat`. PASS. *(2026-06-18)*
- [x] **T3.3** `plan289_synthetic_nop_head_gated` — classifier returns Nop, output scaled by σ(gate_scale). PASS. *(2026-06-18)*
- [x] **T3.4** `plan289_synthetic_broadcast_head_preserved` — classifier returns Broadcast, output bit-identical to Uniform path. PASS. *(2026-06-18)*
- [x] **T3.5** `benches/sink_aware_forward_bench.rs` — sa(Uniform) overhead vs vanilla: **-0.3% / 0.0% / +0.6%** at n ∈ {64, 128, 256}. Target was ≤5%; delivered within noise (zero-cost abstraction confirmed). sa(Dual) overhead: 2.1% / 5.0% / 11.0% (matches Plan 287 G3 cost; cached mitigates to 0.6% / 1.6% / 2.6%). *(2026-06-18)*
- [x] **T3.6** Bonus: `plan289_cached_path_audit_and_reuse` — verifies cached variant kicks in when `sink_scratch.cached = Some`, cadence counter increments correctly, and steady-state output matches audit output. PASS. *(2026-06-18)*

### Phase 4 — Docs

- [x] **T4.1** Updated `crates/katgpt-core/src/parallax_attn.rs` module doc with §Sink-Aware Composition section cross-referencing Plan 287/288/289. *(2026-06-18)*
- [x] **T4.2** Updated `katgpt-rs/README.md` Feature Showcase with forward-path composition row + scope-reduction update + Plan 289 ref. Updated `.benchmarks/059_sink_aware_goat.md` with Plan 289 G3-uniform-overhead row. *(2026-06-18)*
- [x] **T4.3** Added follow-up note to `.plans/287_sink_aware_attention.md` §Follow-up: Plan 289 — DONE. *(2026-06-18)*
- [x] **T4.4** Commit with `feat(289):` prefix on `develop`. Commit `1c08be8e`. *(2026-06-18)*

---

## GOAT gate

- **G3 (Uniform overhead ≤5%):** `forward_sink_aware(Uniform)` vs vanilla `forward` at `n=128, d=64`. MUST PASS — it's the zero-cost-abstraction contract.
- **G1 (parity):** Uniform bit-identical to vanilla; DualPolicy bit-identical to manual composition. MUST PASS.
- **G2 (synthetic):** NOP head gated, Broadcast head preserved. MUST PASS (already established in Plan 287 `tests/sink_aware_g2_synthetic.rs`; re-tested through the forward path here).
- **Real-ViT G2:** DEFERRED — same blocker as Plan 287 (needs frozen model, riir-ai scope).

**Promotion criterion:** passing G3 + G1 + G2-synthetic unlocks the forward-path API. Default `ParallaxConfig` still ships vanilla; callers opt into sink-aware via the new entry point. Promotion to "default sink-aware" remains gated on real-ViT G2 — unchanged from Plan 287.

---

## Non-goals (explicit)

- **NOT** wiring into `FuncAttn`. Φ is `n×k`, not `n×n` attention. Sink semantics on basis-modes were investigated and ruled out as not-applicable — see [Research 261](../.research/261_FuncAttn_Sink_Semantics_Verdict.md). The `SinkAwarePolicy` API is Parallax-specific by design, not by deferral.
- **NOT** changing `ParallaxConfig` struct shape. No new fields, no feature-gated fields, no builder pattern. The sink-aware policy is a property of the *call*, not the *config* — it composes with any `ParallaxConfig`.
- **NOT** deprecating the standalone `apply_dual_policy_gate_flat` / `_cached_flat` API. Diagnostic callers (Research 113 `effective_rank` cross-validation, `summarize_layer_sinks`) still use them directly on precomputed attention maps.
- **NOT** adding SIMD cross-row restructuring to the parallax forward inner loop. Retained-attention path adds one `copy_from_slice` per row — already SIMD-accelerated by the stdlib memcpy.

---

## Validation (fill in as phases complete)

| Gate | Status | Result |
|------|--------|--------|
| G1 (Uniform parity) | ✅ PASS (2026-06-18) | `plan289_uniform_bit_identical_to_vanilla` — output bit-identical to vanilla forward. |
| G1 (DualPolicy parity) | ✅ PASS (2026-06-18) | `plan289_dualpolicy_matches_manual_composition` — output + SinkKind bit-identical to manual `forward_retaining` + `apply_dual_policy_gate_flat`. |
| G1 (retained attn matrix) | ✅ PASS (2026-06-18) | `plan289_retained_attn_matches_per_row_sigmoid` + `_softmax` — full n×n matrix matches row-by-row reference. |
| G2 (synthetic NOP gated) | ✅ PASS (2026-06-18) | `plan289_synthetic_nop_head_gated` — classifier returns Nop, output scaled by σ(gate_scale). |
| G2 (synthetic Broadcast preserved) | ✅ PASS (2026-06-18) | `plan289_synthetic_broadcast_head_preserved` — classifier returns Broadcast, output bit-identical to Uniform. |
| G2 (real ViT) | ⏳ DEFERRED | Same blocker as Plan 287 — needs frozen model, riir-ai scope. |
| G3 (Uniform overhead ≤5%) | ✅ PASS (2026-06-18) | `benches/sink_aware_forward_bench.rs`: -0.3% / 0.0% / +0.6% at n ∈ {64, 128, 256}. Zero-cost abstraction confirmed. |
| cargo check (all features) | ✅ PASS (2026-06-18) | Clean (1 pre-existing unrelated warning in `micro_belief/bom_arena.rs`). |
| parallax_attn test suite | ✅ PASS (2026-06-18) | 17/17 tests pass (10 pre-existing + 7 new Plan 289). |
| cached path semantics | ✅ PASS (2026-06-18) | `plan289_cached_path_audit_and_reuse` — cadence counter + steady-state output verified. |
