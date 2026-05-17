# Plan 066: D2F Discrete Diffusion Forcing — Mini dLLM Research

> Research: `.research/34_D2F_Discrete_Diffusion_Forcing.md`
> Paper: arXiv 2508.09192 — Discrete Diffusion Forcing
> Precedent: `.research/10_ColaDLM_Continuous_Latent_Diffusion.md` (rejected continuous, this is discrete)

## Objective

Build a **mini dLLM from scratch** using our existing wgpu training infrastructure to prove whether Discrete Diffusion Forcing (D2F) is viable for our system. Do NOT use pre-trained dLLMs (LLaDA/Dream) — we train our own tiny model to answer the research questions.

## Phase 0: Proof Tasks (Must Pass Before Implementation)

These are **go/no-go gates**. Each task is a standalone test that answers one doubt from the research doc. If any proof fails, we stop and reassess.

### Task 0.1: Bidirectional Attention Kernel (CPU)
- [ ] Add `AttentionMode` enum to `Config`: `Causal`, `Bidirectional`, `BlockCausal`
- [ ] Modify `attention_head()` to accept mode — bidirectional sets `t_n = block_size` instead of `pos + 1`
- [ ] Test: forward pass with bidirectional mode produces valid attention weights (sums to 1.0)
- [ ] Test: bidirectional on known input matches manual calculation
- **Proof**: Bidirectional attention works correctly on CPU with zero changes to existing causal path

### Task 0.2: Mask Token + Noise Schedule
- [ ] Add `mask_token: usize` to `Config` (typically `vocab_size - 1`)
- [ ] Implement `NoiseSchedule` struct:
  ```rust
  struct NoiseSchedule {
      min_ratio: f32,  // 0.3
      max_ratio: f32,  // 0.7
      n_blocks: usize, // number of blocks
  }
  // Returns Vec<f32> of mask ratios per block, monotonically increasing
  fn monotonic_ratios(&self) -> Vec<f32>
  ```
- [ ] Implement `corrupt_block(tokens: &[usize], mask_ratio: f32, mask_token: usize, rng: &mut Rng) -> Vec<usize>`
- [ ] Test: corrupt_block masks correct percentage of tokens
- [ ] Test: noise schedule produces monotonically increasing ratios
- **Proof**: We can corrupt and track mask state correctly

### Task 0.3: Mini dLLM Training (CPU)
- [ ] Implement `forward_bidirectional()`: same as `forward()` but uses `AttentionMode::Bidirectional`
- [ ] Implement training loop: masked prediction loss (cross-entropy on masked positions only)
- [ ] Train on toy dataset: 4-letter words from alphabet {a..z} with 1-2 positions masked
- [ ] Config: `vocab=27, block=8, n_embd=32, n_head=4, n_layer=1`
- [ ] Measure: reconstruction accuracy on held-out test set
- **Proof**: A mini transformer with bidirectional attention CAN learn masked token prediction
- **Go/No-Go**: If accuracy < 80% after 1000 epochs, STOP — dLLM approach not viable at our scale

### Task 0.4: Block-Causal vs Bidirectional A/B
- [ ] Implement `forward_block_causal()`: bidirectional within block, causal across blocks
- [ ] Train two models on same data:
  - A: Fully bidirectional (teacher)
  - B: Block-causal (student)
- [ ] Compare reconstruction quality at each denoising step
- **Proof**: Quantify how much quality is lost by block-causal restriction
- **Go/No-Go**: If block-causal loses >20% quality vs bidirectional, D2F distillation is not worth it

### Task 0.5: ConstraintPruner During Denoising
- [ ] Integrate `ConstraintPruner::is_valid()` into denoising loop: mask invalid tokens in logits before sampling
- [ ] Test on Sudoku or SynPruner task: denoise with and without pruner
- [ ] Measure: (a) steps to convergence, (b) final accuracy
- **Proof**: ConstraintPruner measurably improves denoising convergence
- **Go/No-Go**: If no measurable improvement, prune integration is unnecessary overhead

---

## Phase 1: GPU Infrastructure (Feature-Gated)

Only start after Phase 0 proves viability.

### Task 1.1: Bidirectional Attention WGSL Kernel
- [ ] Create `attention_score_bidirectional.wgsl`
  - Same as `attention_score.wgsl` but `n_positions = block_size` (not `pos + 1`)
  - Attend to ALL positions in block, no causal mask
- [ ] Add `dllm` feature flag to `riir-gpu/Cargo.toml`
- [ ] Feature-gated pipeline creation for bidirectional kernel
- [ ] Test: GPU bidirectional matches CPU bidirectional output
- [ ] Benchmark: GPU bidirectional throughput vs causal

### Task 1.2: Block-Causal Attention WGSL Kernel
- [ ] Create `attention_score_block_causal.wgsl`
  - For positions in prior blocks: attend to all (KV cached)
  - For positions in current block: attend to all (bidirectional)
  - For positions in future blocks: do not attend
- [ ] Feature-gated pipeline creation
- [ ] Test: block-causal produces correct attention patterns

### Task 1.3: Noise Schedule Training Kernel
- [ ] Create `noise_corrupt.wgsl`: mask tokens on GPU based on ratio
- [ ] Create `masked_loss.wgsl`: cross-entropy on masked positions only (ignore non-masked)
- [ ] Feature-gated `GpuNoiseSchedule` struct
- [ ] Test: GPU corruption matches CPU corruption
- [ ] Test: GPU masked loss matches CPU masked loss

### Task 1.4: Asymmetric Distillation Loss (GPU)
- [ ] Adapt `compute_distill_kl` for D2F: KL(block_causal_student || bidirectional_teacher)
- [ ] Feature-gated `GpuD2fDistill` training loop
- [ ] Test: KL loss is 0 when student = teacher, positive when different
- [ ] Train mini model end-to-end on GPU

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