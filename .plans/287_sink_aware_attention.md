# Plan 287: Sink-Aware Attention — NOP/Broadcast Classifier + Dual-Policy Attention

**Date:** 2026-06-17
**Research:** [katgpt-rs/.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md](../.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md)
**Source paper:** [arXiv:2606.08105](https://arxiv.org/abs/2606.08105) — Fesser et al., *A Unifying View of Attention Sinks: Two Algorithms, Two Solutions*
**Target:** `katgpt-rs/crates/katgpt-core/src/data_probe/sink_classify.rs` (new) + extensions to `parallax_attn.rs`, `funcattn.rs`, `data_probe/geometry.rs`
**Status:** Active — Phase 1 not started

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

- [ ] **T1.1** Create `crates/katgpt-core/src/data_probe/sink_classify.rs` with module doc. Re-export from `data_probe/mod.rs` behind existing `data_probe` feature.
- [ ] **T1.2** Define types:
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
- [ ] **T1.3** `classify_sink_at(position, attn_column: &[f32], values: &[Vec<f32>], update_O: Option<&[Vec<f32>]>) -> SinkDiagnostic`.
  - `strength` = mean of `attn_column`.
  - `value_norm_ratio` = `‖values[position]‖ / mean_i(‖values[i]‖)`. SIMD-accelerated via existing `simd_dot_f32`.
  - `update_stable_rank`: if `update_O` is provided (the per-head `O = AV` matrix), compute `(Σσ_k)^2 / Σσ_k^2` via power iteration (3–5 iters, reuse `manifold_power_iter_router` infra). If `None`, set to `f32::NAN` and decide based on value_norm_ratio alone.
  - Decision rule per research note §2.1.
- [ ] **T1.4** `classify_all_sinks(attn: &[Vec<f32>], values: &[Vec<f32>], cfg: &SinkClassifierConfig) -> Vec<SinkDiagnostic>` — scans all positions, returns candidates with `strength > τ_sink`. Vec allocated by caller (signature takes `&mut Vec<SinkDiagnostic>`).
- [ ] **T1.5** Unit tests (G1 in research note):
  - Synthetic NOP-only head (one position has `‖v‖=0`, all attention mass there) → classified `Nop`.
  - Synthetic Broadcast-only head (one position has `‖v‖=content`, all queries attend uniformly) → classified `Broadcast`, stable rank ≈ 1.
  - Mixed head (two sinks, one NOP one Broadcast) → both correctly classified.
  - No-sink head (uniform attention) → all positions `None`.
  - Edge case: zero attention column (shouldn't crash).
  - Edge case: degenerate values (all-zero values → no division-by-zero in ratio).

---

## Phase 2 — Stable-rank-of-update kernel (PERF)

Stable rank via power iteration is the expensive part. Make it fast enough for hot-path use.

### Tasks

- [ ] **T2.1** `stable_rank_update_into(O: &[Vec<f32>], scratch: &mut StableRankScratch, n_iters: u8) -> f32` — zero-allocation. `StableRankScratch` holds two `Vec<f32>` for power iteration + accumulator.
- [ ] **T2.2** SIMD-accelerate the matvec inside power iteration via existing `simd_dot_f32`. Verify auto-vectorization with `cargo asm` on a representative inner loop.
- [ ] **T2.3** Early-exit: if first power iteration gives `σ_1² / Σ‖row_i‖² > 0.95`, the matrix is effectively rank-1 → return 1.0 without more iterations. This is the common Broadcast case and should be the fast path.
- [ ] **T2.4** Microbench: `criterion` bench on `n ∈ {32, 128, 512}`, `d_h ∈ {64, 128}`. Target: < 1µs for n=32, d_h=64 (plasma tier).
- [ ] **T2.5** Numerical robustness test: input matrix with one row scaled by 1e6 (outlier) should still return a sensible stable rank, not NaN.

---

## Phase 3 — Dual-policy attention (GOAT GATE)

The intervention. Behind a `sink_aware_attn` feature flag. Composes with existing `parallax_attn.rs` and `funcattn.rs` — does NOT replace them.

### Tasks

- [ ] **T3.1** Add `SinkAwarePolicy` enum to `parallax_attn.rs`:
  ```rust
  pub enum SinkAwarePolicy {
      /// Default: uniform sigmoid (current behavior, no classifier overhead).
      Uniform,
      /// Per-head: classify dominant sink, gate if NOP, preserve if Broadcast.
      DualPolicy(SinkClassifierConfig),
  }
  ```
  Add to `ParallaxConfig`. Default remains `Uniform` (no behavior change unless opted in).
- [ ] **T3.2** In `tiled_attention_parallax_forward`, when policy is `DualPolicy`, after computing `O = AV` per head: (a) run classifier on the dominant sink column, (b) if `Nop` apply existing sigmoid gate `σ(X W_θ)`, (c) if `Broadcast` skip the gate. Bounded extra work: one classifier call per head per forward.
- [ ] **T3.3** Mirror in `funcattn.rs` — same policy enum, same dispatch.
- [ ] **T3.4** **G2 GOAT gate**: on a frozen test model (use `percepta` test fixture or a tiny ViT-style config in `examples/`), measure `effective_rank` of hidden states across layers with (a) `Uniform` sigmoid, (b) `DualPolicy`. DualPolicy must preserve or improve effective rank (because Broadcast sinks are no longer over-suppressed). Run on ≥3 random seeds. Record results in `benches/sink_aware_g2_results.md`.
- [ ] **T3.5** **G3 GOAT gate**: latency overhead. Bench `tiled_attention_parallax_forward` with `Uniform` vs `DualPolicy` at n=128, n=512. Overhead must be ≤5% (per research note G3). If over, optimize Phase 2 or skip stable rank when value_norm_ratio alone is decisive (NOP case doesn't need stable rank).
- [ ] **T3.6** Promote decision: if G2 AND G3 pass → flip `ParallaxConfig::default()` to `DualPolicy` and demote `Uniform` to opt-in. If either fails → keep `Uniform` default, leave `DualPolicy` as feature-gated opt-in diagnostic.

---

## Phase 4 — Integration with existing diagnostics (FUSION)

Wire the new classifier into the broader `data_probe` family so it composes with `effective_rank` and `avg_cosine_similarity` (Research 113, Plan 151).

### Tasks

- [ ] **T4.1** Add `SinkSummaryReport` to `data_probe/geometry.rs` (or a new `data_probe/summary.rs`):
  ```rust
  pub struct LayerSinkSummary {
      pub layer_index: usize,
      pub n_nop_sinks: usize,
      pub n_broadcast_sinks: usize,
      pub dominant_kind: SinkKind,        // plurality vote across heads
      pub mean_broadcast_value_norm: f32, // aggregate signal — useful for cross-layer phase plots
  }
  ```
- [ ] **T4.2** `summarize_layer_sinks(attn_per_head: &[Vec<Vec<f32>>], values_per_head: &[Vec<Vec<f32>>], cfg: &SinkClassifierConfig) -> LayerSinkSummary` — runs the classifier across all heads in a layer and aggregates.
- [ ] **T4.3** Example `examples/sink_phase_plot.rs`: load a small pretrained model (or use synthetic ViT-like activations), run the classifier layer-by-layer, plot the `[CLS] → patch` transition from NOP to Broadcast (mirrors paper Figure 4). Even on synthetic data this demonstrates the diagnostic works.
- [ ] **T4.4** Cross-reference in `data_probe/mod.rs` doc: this classifier is the *mechanism locator*; `effective_rank` is the *aggregate symptom*. Document the relationship.

---

## Phase 5 — Documentation & cleanup

### Tasks

- [ ] **T5.1** Update `katgpt-rs/README.md` Feature Showcase with a section on sink-aware attention (under attention family, near EGA / Parallax).
- [ ] **T5.2** Update `katgpt-rs/.research/100_EGA_Energy_Gated_Attention_Spectral_Salience.md` with a cross-reference: "EGA gates uniformly; Research 258 provides the per-head NOP/Broadcast categorization that could make EGA's gate categorical instead of uniform."
- [ ] **T5.3** Update `katgpt-rs/.research/070_Gated_DeltaNet_2_*.md` with a cross-reference: "GDN2's erase/write duality is the linear-attention analog of Research 258's NOP/Broadcast duality for softmax attention."
- [ ] **T5.4** Commit with `feat:` prefix per AGENTS.md. Stay on `develop`. Use rebase non-interactive.

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
| G1 (classifier correctness) | ⏳ pending Phase 1 | |
| G2 (effective_rank preserved/improved) | ⏳ pending Phase 3 | |
| G3 (latency overhead ≤5%) | ⏳ pending Phase 3 | |
| Promote to default | ⏳ pending G2+G3 | |
