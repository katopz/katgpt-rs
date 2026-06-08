# Plan 227: QAT Infusion — Modelless Inference-Time Quantization Awareness

**Date:** 2026-06-08
**Status:** Active
**Research:** `.research/202_QAT_Infusion_Inference_Time_Quantization_Awareness.md`
**Feature Flags:** `static_cal_tables`, `targeted_precision`, `modality_pruned_load`, `precision_aware_draft`, `channel_simd_align`, `async_qdq_overlap`
**GOAT Policy:** All opt-in until benchmarked, then default-ON if gain + no perf hurt

---

## Summary

Apply Gemma 4 QAT's fundamental insight (*optimize for the precision you'll deploy at*) to katgpt-rs as 6 modelless inference-time fusions. No LLM training. No LoRA. Pure inference optimization.

---

## Tasks

### Phase 1: Static Calibration Tables (SCT) — Highest confidence, pure modelless

- [x] Create `src/static_cal.rs` with `StaticCalTable` struct
  - `scales: Vec<f32>` indexed by `(layer * num_heads + head)`
  - `calibrate_from_stats(stats: &[HeadStats])` — sigmoid + EMA calibration
  - `get_scale(layer: usize, head: usize) -> f32` — O(1) unsafe lookup in release
  - `commit()` / `verify()` — BLAKE3 commitment
  - `HeadStats` struct for per-head activation statistics
- [x] Add `static_cal_tables` feature flag to `Cargo.toml` (opt-in, NOT default)
- [x] Register `#[cfg(feature = "static_cal_tables")] pub mod static_cal` in `lib.rs`
- [ ] Implement calibration pass: run 10-20 representative prompts through model, record per-head activation statistics
- [ ] Wire into KVarN: when `static_cal_tables` enabled, use static scales instead of Sinkhorn iterations
- [ ] Add River Valley trigger for recalibration: when RV signal detects distribution shift, re-run calibration
- [ ] Write `tests/static_cal_goat.rs` benchmark:
  - Before: KVarN with Sinkhorn (4-8 iterations per decode)
  - After: KVarN with static scales
  - Measure: decode latency, perplexity delta, calibration time
- [ ] GOAT gate: if latency improves ≥5% and perplexity delta < 0.1, mark for default-ON

### Phase 2: Targeted Precision Budget (TPB) — Per-head bit allocation

- [ ] Create `src/targeted_precision.rs` with `PrecisionBudget` struct
  - `head_bits: Vec<u8>` — bits per attention head
  - `budget: f32` — total bits budget (average)
  - `compute_budget(model: &Model, calibration_data: &[Tensor]) -> PrecisionBudget`
- [ ] Add `targeted_precision` feature flag
- [ ] Implement sensitivity analysis: for each head, measure perplexity impact of quantization noise
- [ ] Implement bit allocation: constrained optimization (total bits = budget, minimize perplexity)
  - Use greedy: sort heads by sensitivity, allocate budget to most sensitive first
- [ ] Wire into KVarN: use per-head bit-width from `PrecisionBudget` instead of uniform
- [ ] Write `tests/targeted_precision_goat.rs` benchmark:
  - Before: KVarN uniform 2.3 bits/head
  - After: KVarN targeted (some 4-bit, some 2-bit, same total)
  - Measure: perplexity, KV cache size (must be same), decode latency
- [ ] GOAT gate: if perplexity improves ≥2% at same cache size, mark for default-ON

### Phase 3: Modality-Pruned Context Loading — Pipeline pruning

- [ ] Create `src/pipeline_pruner.rs` with `PipelineConfig` enum
  - `Simple` — direct decode only
  - `Code` — DDTree + SynPruner, no KV compression
  - `LongContext` — VortexFlow + KV compression, no speculative
  - `Reasoning` — Adaptive CoT + ThoughtFold, full precision
- [ ] Add `modality_pruned_load` feature flag
- [ ] Implement query classifier: use River Valley signal + Lodestar distance to classify queries
  - Simple: low entropy, short expected output
  - Code: syntactic patterns (brackets, semicolons)
  - LongContext: input length > threshold
  - Reasoning: high entropy, multi-step expected
- [ ] Wire into InferenceRouter: `select_pipeline(query) -> PipelineConfig`
- [ ] Write `tests/pipeline_pruner_goat.rs` benchmark:
  - Measure latency per query class with and without pruning
  - Verify quality: each class must maintain output quality within tolerance
- [ ] GOAT gate: if latency improves ≥20% for simple queries with no quality regression, default-ON

### Phase 4: Precision-Aware Speculative Drafting (PASD)

- [ ] Create `src/precision_aware_draft.rs` with `BoundaryPenalty` struct
  - `compute_boundary_score(token_logits: &[f32], quant_scale: f32) -> f32`
  - Scores how close logits are to quantization boundaries
- [ ] Add `precision_aware_draft` feature flag
- [ ] Implement boundary detection: for each draft token, check if logit is within ε of quantization grid boundary
- [ ] Implement draft scoring: `draft_score += boundary_penalty * weight`
- [ ] Wire into `SpeculativeGenerator` trait: add optional boundary penalty to `generate()`
- [ ] Write `tests/precision_aware_draft_goat.rs` benchmark:
  - Before: standard speculative decoding acceptance rate
  - After: precision-aware draft acceptance rate
  - Measure: acceptance rate, tokens/sec, overhead of boundary computation
- [ ] GOAT gate: if acceptance rate improves ≥5% with <1% overhead, default-ON

### Phase 5: Channel SIMD Alignment — Data layout optimization

- [ ] Audit `TernaryWeights` struct for SIMD lane alignment
- [ ] Add `channel_simd_align` feature flag
- [ ] Implement cache-line-aligned storage: pad weight rows to 64-byte boundaries
- [ ] Implement aligned quantize/dequantize paths in `simd.rs`
- [ ] Write `tests/channel_simd_goat.rs` benchmark:
  - Before: standard ternary matvec
  - After: cache-line-aligned ternary matvec
  - Measure: SIMD throughput (ops/sec), cache miss rate (if possible)
- [ ] GOAT gate: if throughput improves ≥5%, default-ON

### Phase 6: Async Q/DQ Overlap — GPU pipeline (depends on GPU feature)

- [ ] Add `async_qdq_overlap` feature flag (requires `inference_router`)
- [ ] Implement double-buffered KV dequantize: CPU dequantizes chunk N+1 while GPU processes chunk N
- [ ] Implement in `InferenceBackend::GpuBackend` only
- [ ] Write `tests/async_qdq_goat.rs` benchmark:
  - Before: sequential dequantize → attention
  - After: overlapped dequantize + attention
  - Measure: GPU utilization, throughput, latency
- [ ] GOAT gate: if throughput improves ≥15% on GPU, default-ON (with `inference_router`)

---

## Expected Outcomes

| Phase | Expected Gain | Risk | Default Policy |
|-------|--------------|------|----------------|
| SCT (Static Cal) | 10-15% decode speedup | Calibration quality on new domains | Opt-in → default if GOAT |
| TPB (Targeted Precision) | 2-5% perplexity at same cache | Sensitivity analysis accuracy | Opt-in → default if GOAT |
| Modality Pruning | 20-40% for simple queries | Query classification accuracy | Opt-in → default if GOAT |
| PASD (Draft Awareness) | 5-10% acceptance rate | Boundary computation overhead | Opt-in → default if GOAT |
| Channel SIMD | 5-10% SIMD throughput | Alignment overhead for small matrices | Opt-in → default if GOAT |
| Async Q/DQ | 15-25% GPU throughput | GPU-only, needs `inference_router` | Opt-in (GPU only) |

---

## Dependencies

- Phase 1-2 depend on KVarN infrastructure (already exists)
- Phase 3 depends on InferenceRouter + TriggerGate (already exists)
- Phase 4 depends on SpeculativeGenerator trait (already exists)
- Phase 5 depends on PlasmaPath ternary SIMD (already exists)
- Phase 6 depends on GPU inference backend (`gpu_inference` feature)

---

## TL;DR

6 modelless fusions from QAT's fundamental insight. All behind individual GOAT feature flags. Phase 1 (Static Cal Tables) and Phase 3 (Modality Pruning) have highest expected gain and lowest risk. All fit the MIT engine — no LLM training, no LoRA, no commercial conflict. Benchmark first, default-ON only if GOAT proves gain + no perf hurt.
