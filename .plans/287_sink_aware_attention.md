# Plan 287: Sink-Aware Attention — NOP/Broadcast Classifier + Dual-Policy Attention

**Date:** 2026-06-17
**Research:** [katgpt-rs/.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md](../.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md)
**Source paper:** [arXiv:2606.08105](https://arxiv.org/abs/2606.08105) — Fesser et al., *A Unifying View of Attention Sinks: Two Algorithms, Two Solutions*
**Target:** `katgpt-rs/src/data_probe/sink_classify.rs` (new, root crate — path-corrected; primitive types live in `katgpt-rs/crates/katgpt-core/src/data_probe.rs`) + extensions to `data_probe/geometry.rs`
**Status:** Complete (Phases 1–5). G1 ✅ PASS, synthetic G2 ✅ PASS, G3 ❌ FAIL (latency). NOT promoted to default — see [`.benchmarks/059_sink_aware_goat.md`](../.benchmarks/059_sink_aware_goat.md).

---

## Goal

Add an inference-time `AttentionSinkClassifier` that distinguishes **Adaptive NOP** sinks (`‖v_s‖ ≈ 0`, suppress residual) from **Broadcast** sinks (`‖v_s‖ ≈ content`, rank-1 update `O ≈ a_s v_s^T`). Add a **dual-policy attention** mode that gates only NOP sinks (via sigmoid) while preserving Broadcast sinks (which carry load-bearing global information).

This addresses a known over-suppression in our default sigmoid attention (`parallax_attn.rs`, `funcattn.rs`): replacing softmax kills ALL sinks indiscriminately, but the paper proves some sinks are useful broadcasters. Per-head classification lets us keep the broadcasters and only gate the no-ops.

Ships open-source (MIT) as a generic math/diagnostic primitive. Game-domain τ thresholds per head class (if they materialize) stay private in riir-ai.

**GOAT gate:** dual-policy attention must preserve or improve `effective_rank` vs uniform sigmoid on a frozen ViT-style test bed, with ≤5% latency overhead per head. If it fails, demote to opt-in diagnostic only.

---

## Phase 1 — `AttentionSinkClassifier` primitive (CORE)

The minimal, dependency-free classifier. Pure math over `&[f32]` attention maps and value matrices. Zero allocation in hot path (caller-owned scratch buffers).

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/data_probe.rs` with module doc. Re-export from `src/data_probe/sink_classify.rs` behind existing `data_probe` feature. *(2026-06-17)*
- [x] **T1.2** Define types:
  ```rust
  pub enum SinkKind { None, Nop, Broadcast }
  pub struct SinkDiagnostic {
      pub position: usize,
      pub strength: f32,           // mean attention mass received
      pub value_norm_ratio: f32,   // ‖v_s‖ / mean(‖v_i‖)
      pub update_stable_rank: f32, // stable rank of O = AV per-head
      pub kind: SinkKind,
  }
  pub struct SinkClassifierConfig {
      pub sink_strength_threshold: f32,    // τ_sink — default 0.5
      pub nop_value_ratio_max: f32,         // default 0.2
      pub broadcast_value_ratio_min: f32,   // default 0.5
      pub broadcast_value_ratio_max: f32,   // default 1.5
      pub broadcast_stable_rank_max: f32,   // default 1.5
  }
  ```
  All defaults match plan; `SinkKind` is `#[repr(u8)]` per AGENTS.md.
- [x] **T1.3** `classify_sink_at(position, attn_column, values, update_O, cfg, scratch) -> SinkDiagnostic`.
  - `strength` = mean of `attn_column` via `simd_sum_f32`.
  - `value_norm_ratio` = `‖values[position]‖ / mean_i(‖values[i]‖)` via `simd_dot_f32`. Degenerate (all-zero values) → ratio=1.0, kind=None.
  - `update_stable_rank`: if `update_O` provided, vendored ~30-line power iteration (5 iters, no `manifold_power_iter_router` dependency). If `None`, set to `f32::NAN`.
  - Decision rule per research note §2.1.
- [x] **T1.4** `classify_all_sinks(attn, values, cfg, scratch, out: &mut Vec<SinkDiagnostic>)`. Caller-owned `out`; single n-length allocation per call.
- [x] **T1.5** Unit tests (G1): 8 tests in `src/data_probe/sink_classify.rs`:
  - `g1_nop_only_head`, `g1_broadcast_only_head`, `g1_mixed_head`, `g1_mixed_head_both_above_threshold`, `g1_no_sink_head`, `g1_zero_attn_column_edge`, `g1_degenerate_values_edge`, `g1_stable_rank_zero_matrix`. All pass.

---

## Phase 2 — Stable-rank-of-update kernel (PERF)

Stable rank via power iteration is the expensive part. Make it fast enough for hot-path use.

### Tasks

- [x] **T2.1** `stable_rank_update_into(O: &[Vec<f32>], scratch: &mut StableRankScratch, n_iters: u8) -> f32` — zero-allocation scratch path; one n-length local buffer for matvec intermediate. *(2026-06-17)*
- [x] **T2.2** SIMD-accelerate the matvec inside power iteration via `simd_dot_f32` + `simd_fused_scale_acc`. Two-pass decomposition (O·v then Oᵀ·(Ov)) avoids materializing Oᵀ·O.
- [x] **T2.3** Early-exit: if first power iteration gives `σ_1² > 0.95 · trace(F)`, return 1.0 immediately.
- [x] **T2.4** Microbench `benches/sink_classify_bench.rs` sweeping `n ∈ {32, 128, 512}`, `d_h ∈ {64, 128}`. **Target <1µs for n=32, d_h=64 NOT MET**: 1.71µs (random O), 0.79µs (rank-1 fast path). Documented in `.benchmarks/059_sink_aware_goat.md`.
- [x] **T2.5** Numerical robustness: all-zero matrix → 0.0 (no NaN). Covered by `g1_stable_rank_zero_matrix`.

---

## Phase 3 — Dual-policy attention (GOAT GATE)

The intervention. Behind a `sink_aware_attn` feature flag. Composes with existing `parallax_attn.rs` and `funcattn.rs` — does NOT replace them.

### Tasks

- [x] **T3.1** `SinkAwarePolicy` enum shipped in `crates/katgpt-core/src/data_probe.rs`. **Scope reduction** per validation fallback: NOT wired into `ParallaxConfig` / `FuncAttnConfig` (would break backwards-compat). Standalone path only. *(2026-06-17)*
- [x] **T3.2** `apply_dual_policy_gate(attn, values, O, policy, gate_scale, scratch, out) -> SinkKind`. Standalone post-forward intervention; classifies dominant sink, gates if NOP, copies if Broadcast/None.
- [x] **T3.3** Same `SinkAwarePolicy` enum + gate covers both parallax and funcattn paths (policy-agnostic on post-`AV` output `O`). Funcattn-specific Φ residual scaling variant not implemented — documented.
- [x] **T3.4** **G2 GOAT gate (synthetic)**: `tests/sink_aware_g2_synthetic.rs` — 2/2 PASS. Broadcast head: DualPolicy classifies Broadcast → output unchanged. NOP head: DualPolicy classifies NOP → output scaled by σ(gate_scale). Real-ViT `effective_rank` gate **DEFERRED** — needs frozen model.
- [x] **T3.5** **G3 GOAT gate (latency)**: `benches/sink_aware_latency_bench.rs`. **FAIL**: 1671% overhead at n=128, d_h=64. Far above 5% target. See bench doc for root cause analysis.
- [x] **T3.6** **Promote decision: DO NOT PROMOTE.** G3 missed by ~3 orders of magnitude. Default `SinkAwarePolicy::Uniform` stays; `DualPolicy` remains opt-in diagnostic. Demotion path documented.

---

## Phase 4 — Integration with existing diagnostics (FUSION)

Wire the new classifier into the broader `data_probe` family so it composes with `effective_rank` and `avg_cosine_similarity` (Research 113, Plan 151).

### Tasks

- [x] **T4.1** `LayerSinkSummary` added to `src/data_probe/geometry.rs`: `layer_index`, `n_nop_sinks`, `n_broadcast_sinks`, `dominant_kind`, `mean_broadcast_value_norm`. *(2026-06-17)*
- [x] **T4.2** `summarize_layer_sinks(attn_per_head, values_per_head, cfg, scratch, layer_index) -> LayerSinkSummary`. Runs classifier across all heads; aggregates via plurality vote.
- [x] **T4.3** Example `examples/sink_phase_plot.rs`. Synthetic ViT-like activations; layers 0-3 NOP-dominant (zero CLS value), layers 4-7 would-be Broadcast (showing as None since `classify_all_sinks` doesn't pass `update_O` — documented in example output).
- [x] **T4.4** Cross-reference in `src/data_probe/mod.rs` doc: classifier is mechanism locator, `effective_rank` is aggregate symptom.

---

## Phase 5 — Documentation & cleanup

### Tasks

- [x] **T5.1** `katgpt-rs/README.md` Feature Showcase: sink-aware attention entry added under Attention Matching. *(2026-06-17)*
- [x] **T5.2** Cross-reference added to `.research/100_EGA_Energy_Gated_Attention_Spectral_Salience.md` — EGA gates uniformly; sink-aware makes it categorical.
- [x] **T5.3** Cross-reference added to `.research/070_Gated_DeltaNet_2_Decoupled_Erase_Write_Linear_Attention.md` — GDN2 erase/write duality is the linear-attention analog of NOP/Broadcast for softmax.
- [x] **T5.4** Commit with `feat:` prefix on `develop`. *(2026-06-17)*

---

## Non-goals (explicit)

- **NOT** implementing register tokens. Requires base-model retraining (AGENTS.md: frozen-base modelless constraint). We can *simulate* register slots at inference time (reserved KV positions), but that's a separate plan if it becomes interesting.
- **NOT** building the crowd-level coherence signal fusion (research note §2.3). That's a riir-ai question; file as `.issues/` if Phase 4 G4 shows promise on the game side.
- **NOT** touching `softmax` paths. This is purely additive to the sigmoid/parallax family.

---

## Dependencies

- Existing: `simd_dot_f32`, `manifold_power_iter_router` (for power iteration infra), `data_probe/geometry.rs` (Roy-Vetterli effective rank — same family).
- New: none. Pure Rust, no new crates.

---

## Risk register

| Risk | Mitigation |
|---|---|
| Stable rank computation too slow for hot path | Phase 2 T2.3 early-exit. If still too slow, fall back to value_norm_ratio alone (NOP detection doesn't need stable rank). |
| DualPolicy doesn't actually beat Uniform sigmoid on our models (we use small models, not ViT-L) | G2 gate is honest — if it fails, the classifier still ships as a diagnostic (Phase 1+4 valuable regardless). |
| Power iteration diverges on adversarial inputs | T2.5 numerical test + cap iterations at 5. |
| Overlaps too much with existing EGA feature | Documented in T5.2 — EGA is uniform, this is categorical. Different mechanisms, complementary. |

---

## Validation summary (fill in as phases complete)

| Gate | Status | Result |
|---|---|---|
| G1 (classifier correctness) | ✅ PASS (2026-06-17) | 8/8 unit tests in `src/data_probe/sink_classify.rs`. NOP, Broadcast, mixed, edge cases all handled. |
| G2 (effective_rank preserved/improved — synthetic) | ✅ PASS (2026-06-17) | 2/2 tests in `tests/sink_aware_g2_synthetic.rs`. DualPolicy preserves Broadcast output, gates NOP output. |
| G2 (effective_rank preserved/improved — real ViT) | ⏳ DEFERRED | Requires frozen ViT model + per-layer hook. Out of scope for this task. |
| G3 (latency overhead ≤5%) | ❌ FAIL (2026-06-17) | 1671% overhead at n=128, d_h=64 (target was ≤5%). Root cause: stable-rank computation cost + n² col-sum scan. See `.benchmarks/059_sink_aware_goat.md` for analysis. |
| Promote to default | ❌ NOT PROMOTED (2026-06-17) | G3 missed by ~3 orders of magnitude. Default stays `Uniform`. `sink_aware_attn` remains opt-in diagnostic. |

---

## Scope reductions (2026-06-17)

1. **Plan target path was wrong.** The plan said `crates/katgpt-core/src/data_probe/sink_classify.rs`, but the `data_probe` module already exists in the root crate at `src/data_probe/`. Corrected to `src/data_probe/sink_classify.rs` (root-crate re-export) + `crates/katgpt-core/src/data_probe.rs` (primitive types — needed so katgpt-core can reference them).
2. **Direct wiring into `parallax_attn.rs` / `funcattn.rs` forward paths — split verdict.** The policy enum + standalone `apply_dual_policy_gate` shipped initially; callers invoke after a forward pass. Keeps `ParallaxConfig` / `FuncAttnConfig` backwards-compatible.
   - **Parallax half: DONE** via [Plan 289](./289_sink_aware_forward_path_wiring.md) (single entry point `tiled_attention_parallax_forward_sink_aware`, separate from `ParallaxConfig`).
   - **FuncAttn half: CLOSED as not-applicable** via [Research 261](../.research/261_FuncAttn_Sink_Semantics_Verdict.md). FuncAttn's `Φ · C · Ṽ` structure has no `n×n` attention matrix for the sink classifier to scan — basis modes are partition-of-unity by design, so the NOP/Broadcast discrimination collapses into a column-norm check.
3. **Real-ViT effective_rank G2 DEFERRED.** Needs a frozen model. Synthetic G2 substitute verifies the dual-policy decision logic.
4. **Stable-rank formula clarification.** Plan task wrote `(Σσ_k)²/Σσ_k²`; we implemented the standard stable rank `‖O‖_F²/‖O‖_op²` (matches the prescribed approximation, only needs top singular value, consistent with Roy-Vetterli `effective_rank` in `geometry.rs`). Documented in module doc.

---

## Follow-up: Plan 288 (flat-layout variants) — DONE 2026-06-18

[Plan 288](./288_sink_aware_flat_layout.md) shipped flat `&[f32]` row-major variants of every sink-aware function (`stable_rank_update_into_flat`, `classify_sink_at_flat`, `classify_all_sinks_flat`, `apply_dual_policy_gate_flat`, `apply_dual_policy_gate_cached_flat`). These match the layout used by `parallax_attn::tiled_attention_parallax_forward` and `funcattn::funcattn_forward`, unblocking direct forward-path composition without `Vec<Vec<f32>>` materialization.

**Result:** flat variants are 1.8×–5.1× faster than Vec<Vec<f32>> due to cache locality. The cached-flat steady-state path is also faster than the Vec<Vec> Uniform baseline (single contiguous memcpy beats n per-row copies). G1 parity verified by 8 new unit tests.

This removes the technical blocker for the forward-path wiring (scope reduction #2 above). The wiring itself is now a pure API-design task — Plan 289 scope.

---

## Follow-up: Plan 289 (parallax forward-path wiring) — DONE 2026-06-18

[Plan 289](./289_sink_aware_forward_path_wiring.md) shipped `tiled_attention_parallax_forward_sink_aware` — a single entry point composing the parallax forward with the flat dual-policy gate. Resolves scope reduction #2 (parallax half). The funcattn half was investigated and **closed as not-applicable** — see [Research 261](../.research/261_FuncAttn_Sink_Semantics_Verdict.md) for the negative-result analysis (FuncAttn's `Φ · C · Ṽ` structure has no token-position sinks).

**Design:** separate entry point (not a `ParallaxConfig` field) — `ParallaxConfig::default()` stays feature-independent. `SinkAwarePolicy::Uniform` short-circuits to vanilla forward; `DualPolicy` runs the retained-attention forward into a caller-owned `o_temp`, then applies the flat gate. All scratch (n² attention matrix, n·d temp, classifier, optional cache) is bundled in one `SinkAwareParallaxScratch` struct.

**Result (G3 zero-cost contract):** `forward_sink_aware(Uniform)` overhead vs vanilla forward is **-0.3% / 0.0% / +0.6%** at n ∈ {64, 128, 256} — within measurement noise. DualPolicy adds the expected classifier cost (2.1%–11.0%); cached brings it to ≤3%. 7 new unit tests cover parity, synthetic G2 (NOP gated / Broadcast preserved), and cached path semantics.

The API-design blocker for forward-path integration is removed. **Real-ViT G2 remains the only gate blocker** for promoting `sink_aware_attn` to default features.
