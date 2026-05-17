# Plan 066: D2F Discrete Diffusion Forcing — Mini dLLM Research

> Research: `.research/34_D2F_Discrete_Diffusion_Forcing.md`
> Paper: arXiv 2508.09192 — Discrete Diffusion Forcing
> Precedent: `.research/10_ColaDLM_Continuous_Latent_Diffusion.md` (rejected continuous, this is discrete)

## Objective

Build a **mini dLLM from scratch** using our existing wgpu training infrastructure to prove whether Discrete Diffusion Forcing (D2F) is viable for our system. Do NOT use pre-trained dLLMs (LLaDA/Dream) — we train our own tiny model to answer the research questions.

## Phase 0: Proof Tasks (Must Pass Before Implementation)

These are **go/no-go gates**. Each task is a standalone test that answers one doubt from the research doc. If any proof fails, we stop and reassess.

### Task 0.1: Bidirectional Attention Kernel (CPU)
- [x] Add `AttentionMode` enum to `Config`: `Causal`, `Bidirectional`, `BlockCausal`
- [x] Modify `attention_head()` to accept mode — bidirectional sets `t_n = block_size` instead of `pos + 1`
- [x] Test: forward pass with bidirectional mode produces valid attention weights (sums to 1.0)
- [x] Test: bidirectional on known input matches manual calculation
- **Proof**: Bidirectional attention works correctly on CPU with zero changes to existing causal path

### Task 0.2: Mask Token + Noise Schedule
- [x] Add `mask_token: usize` to `Config` (typically `vocab_size - 1`)
- [x] Implement `NoiseSchedule` struct:
  ```rust
  struct NoiseSchedule {
      min_ratio: f32,  // 0.3
      max_ratio: f32,  // 0.7
      n_blocks: usize, // number of blocks
  }
  // Returns Vec<f32> of mask ratios per block, monotonically increasing
  fn monotonic_ratios(&self) -> Vec<f32>
  ```
- [x] Implement `corrupt_block(tokens: &[usize], mask_ratio: f32, mask_token: usize, rng: &mut Rng) -> Vec<usize>`
- [x] Test: corrupt_block masks correct percentage of tokens
- [x] Test: noise schedule produces monotonically increasing ratios
- **Proof**: We can corrupt and track mask state correctly

### Task 0.3: Mini dLLM Training (CPU)
- [x] Implement `forward_bidirectional()`: same as `forward()` but uses `AttentionMode::Bidirectional`
- [x] Implement training loop: masked prediction loss (cross-entropy on masked positions only)
- [x] Train on toy dataset: alternating pattern [a,b,a,b] with 1 position masked
- [x] Config: `vocab=27, block=8, n_embd=32, n_head=4, n_layer=1`
- [x] Measure: reconstruction accuracy on held-out test set
- **Proof**: A mini transformer with bidirectional attention CAN learn masked token prediction
- **Go/No-Go**: If accuracy < 80% after 1000 epochs, STOP — dLLM approach not viable at our scale

### Task 0.4: Block-Causal vs Bidirectional A/B
- [x] Implement `forward_block_causal()`: bidirectional within block, causal across blocks
- [x] Train two models on same data:
  - A: Fully bidirectional (teacher)
  - B: Block-causal (student)
- [x] Compare reconstruction quality at each denoising step
- **Proof**: Quantify how much quality is lost by block-causal restriction
- **Go/No-Go**: If block-causal loses >20% quality vs bidirectional, D2F distillation is not worth it

### Task 0.5: ConstraintPruner During Denoising
- [x] Integrate `ConstraintPruner::is_valid()` into denoising loop: mask invalid tokens in logits before sampling
- [x] Test with NoRepeatConstraint: denoise with and without pruner
- [x] Measure: (a) steps to convergence, (b) final accuracy
- **Proof**: ConstraintPruner measurably improves denoising convergence
- **Go/No-Go**: If no measurable improvement, prune integration is unnecessary overhead

---

## Phase 1: GPU Infrastructure (Feature-Gated) ✅

Implemented in `riir-ai/crates/riir-gpu` (Plan 068).

### Task 1.1: Bidirectional Attention WGSL Kernel ✅
- [x] Modify `attention_score.wgsl` — added `n_positions_override` param (backward-compat)
  - `n_positions_override=0` → causal (`pos+1`), `>0` → use as `n_positions`
  - Single kernel handles causal, bidirectional, and block-causal modes
- [x] Add `dllm` feature flag to `riir-gpu/Cargo.toml` (propagates to `riir-engine/dllm`)
- [x] Feature-gated `forward_bidirectional()` on `GpuForwardPass` (two-phase: KV fill + bidi attention)
- [x] Per-head per-position uniform buffers to avoid shared-buffer write race across positions
- [x] Test: GPU bidirectional differs from causal (cos_sim=0.94, MAE=58.8) — `test_dllm_attention_correctness`
- [x] Test: GPU bidirectional deterministic (MAE=0.0 across runs)
- [x] Benchmark: GPU bidirectional training throughput — `bench_dllm_gpu_training`

### Task 1.2: Block-Causal Attention WGSL Kernel ✅
- [x] Same kernel with per-position `n_positions_override` from `block_causal_t_n()`
  - Prompt positions: attend to all prompt positions (bidirectional)
  - Generation positions: bidirectional within block, causal across blocks
- [x] Feature-gated `forward_block_causal()` on `GpuForwardPass`
- [x] Test: block-causal with block_size=seq_len ≈ bidirectional (cos_sim=1.000) ✅
- [x] Test: block-causal with block_size=1 ≈ causal (cos_sim=0.999) ✅
- [x] Test: block-causal distinct from both causal and bidirectional ✅
- [x] Test: block-causal with prompt_len > 0 produces distinct output ✅
- **Limitation**: Single-layer only (n_layer=1). Multi-layer requires per-position hidden state storage.

### Task 1.3: Noise Schedule Training Kernel ✅
- [x] `noise_corrupt.wgsl`: PCG32 per-position token masking with prompt protection
- [x] `loss_masked.wgsl`: importance-weighted CE on masked positions only
- [x] Feature-gated `GpuNoiseCorrupt` struct in `riir-gpu/src/dllm.rs`
- [x] Feature-gated `GpuMaskedLoss` struct in `riir-gpu/src/dllm.rs`
- [x] Test: GPU corruption throughput — `bench_dllm_kernels`
- [x] Test: GPU masked loss vs CPU — `bench_dllm_kernels`

### Task 1.4: Asymmetric Distillation Loss (GPU) ✅
- [x] `GpuD2fDistill` with teacher (bidirectional) → student (block-causal) distillation
- [x] Teacher uses `forward_bidirectional()` (frozen, base weights only)
- [x] Student uses `forward_block_causal()` (trainable LoRA)
- [x] Hard distillation: teacher targets = argmax(teacher_logits)
- [x] Test: all 3 trainers run and produce finite losses — `test_dllm_training`
- [x] Test: cross-trainer A/B/C comparison — `test_all_trainers_comparison`
- [x] Benchmark: all 3 trainers throughput — `bench_dllm_gpu_training`

---

## Phase 2: Inference Pipeline (Feature-Gated)

### Task 2.1: D2F Inference in microgpt-rs
- [ ] Feature flag `dllm` in `microgpt-rs/Cargo.toml`
- [ ] New module `src/speculative/d2f.rs` (feature-gated)
- [ ] Implement `d2f_decode_block()`:
  1. Initialize block with mask tokens
  2. Denoising loop (configurable steps T)
  3. Each step: forward_block_causal → get logits → ConstraintPruner mask → sample
  4. Confidence remasking (τ_conf threshold)
- [ ] Implement pipelined parallel decode:
  - `D2fBlockState` enum: `SemiActivated`, `FullyActivated`
  - Dynamic block addition when predecessor exceeds τ_add
  - State transition at τ_act threshold
- [ ] Integrate with existing `SpeculativeContext` for buffer reuse
- [ ] KV cache commit: after block fully denoised, write to persistent KV cache

### Task 2.2: ConstraintPruner Integration
- [ ] At each denoising step, call `pruner.is_valid(depth, token, path)` for each candidate
- [ ] Mask invalid tokens in logits before sampling (set to -inf)
- [ ] For `ScreeningPruner`: use relevance score to weight sampling probabilities
- [ ] Benchmark: denoising quality with vs without pruner

### Task 2.3: Benchmark Suite
- [ ] Create `tests/test_d2f_decode.rs` (feature-gated)
- [ ] Benchmarks:
  - a) Denoising quality vs number of steps (convergence curve)
  - b) Throughput: D2F decode vs AR decode vs DFlash speculative
  - c) Quality: D2F output vs AR output on same task
  - d) ConstraintPruner impact: convergence speed with vs without
- [ ] Compare against DFlash+DDTree baseline on identical tasks

---

## Phase 3: Integration (If Results Are Good)

### Task 3.1: Hybrid AR-D2F Pipeline
- [ ] Config option to choose decode strategy: AR, DFlash, D2F
- [ ] Auto-switch: use D2F for block-parallel tasks, AR for sequential tasks
- [ ] Router integration: domain config can specify D2F as decode strategy

### Task 3.2: Documentation & Research Update
- [ ] Update `.research/34_D2F_Discrete_Diffusion_Forcing.md` with benchmark results
- [ ] Update `README.md` with D2F section (if results warrant)
- [ ] Update `.docs/03_speculative_decoding.md` with D2F as decode option

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mini dLLM can't learn (Task 0.3 fails) | Project stops | Reduce to simpler task, increase model size |
| Block-causal quality too low (Task 0.4) | No distillation path | Use bidirectional at inference, accept no KV cache |
| ConstraintPruner doesn't help (Task 0.5) | Minor — still works without | Skip pruner integration, use only for quality |
| GPU kernel bugs (Phase 1) | Delay | Extensive CPU validation first (Phase 0) |
| Performance worse than AR (Phase 2) | D2F not viable for our scale | Publish negative result, keep feature-gated code |

## Dependencies

- Phase 0: No new dependencies (CPU only, existing infrastructure)
- Phase 1: `riir-gpu` wgpu infrastructure (already production-ready)
- Phase 2: `microgpt-rs` speculative module (already production-ready)

## Estimated Timeline

| Phase | Duration | Blockers |
|-------|----------|----------|
| Phase 0 (Proof Tasks) | 3-5 days | None |
| Phase 1 (GPU Infra) | 5-7 days | Phase 0 go |
| Phase 2 (Inference) | 5-7 days | Phase 1 complete |
| Phase 3 (Integration) | 3-5 days | Phase 2 benchmarks positive |
| **Total** | **16-24 days** | Staged go/no-go gates |