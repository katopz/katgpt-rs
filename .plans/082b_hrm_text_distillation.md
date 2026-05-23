# Plan 082b: HRM-Text Technique Distillation

> **Status:** ✅ Complete (T1–T6). All 10 sampler tests + 10 optimizer tests pass. 281/281 riir-gpu lib tests pass.
> **Branch:** `develop/feature/082b_hrm_distill`
> **Depends on:** Plan 082 (RowNormM, complete), Plan 066 (D2F, complete)
> **Research:** `.research/048_HRM_Text_Hierarchical_Recurrent_Pretraining.md`
> **Source:** https://github.com/sapientinc/HRM-Text
> **Goal:** Distill 4 proven techniques from HRM-Text into our training stack: Adam-atan2 optimizer, multipack LPT batching, backprop warmup scheduling, and learned initial states.

## Summary

HRM-Text achieves 1B-scale pretraining with ~$1000. While the hierarchical recurrent architecture itself is outside our scope (we do LoRA fine-tuning, not pretraining), several supporting techniques distill cleanly:

1. **Adam-atan2** — Replace `momentum / (v + eps)` with `atan2(momentum, v)` in our WGSL Adam kernel
2. **Multipack LPT batching** — Smart sequence packing for LoRA training batches
3. **Backprop warmup** — Ramp compute depth during early training
4. **Learned initial states** — Non-zero initialization for recurrent carry

## Tasks

- [x] **T1: Adam-atan2 WGSL kernel** — Replace Adam update in training shaders
  - Created `crates/riir-gpu/src/kernels/adam_atan2.wgsl` with `atan2(m_hat, v_sqrt)` update
  - Struct layout matches `AdamWParams` (32 bytes with `_eps_reserved` and `_pad`)
  - Registered in `kernels/mod.rs`: source constant, entry point, pipeline field, pipeline creation
  - CPU reference: `adam_atan2_step_cpu()` + `CpuAdamAtan2Step` in `optimizer.rs`
  - Bounded: |atan2(m, sqrt(v))| ≤ π/2 when v > 0, prevents LoRA weight explosion
  - No epsilon needed (atan2 handles near-zero naturally)
  - **Tests:** `test_adam_atan2_bounded_updates`, `test_adam_atan2_vs_adamw_convergence`, `test_adam_atan2_update_bounded_by_pi` — all pass

- [x] **T2: Multipack LPT batch sampler (Rust)** — Smart sequence packing for training
  - Created `crates/riir-gpu/src/sampler.rs` with `MultipackSampler`, `MultipackConfig`, `PackedSequence`
  - LPT scheduling via min-heap: sort sequences by length descending, assign to least-full pack
  - Includes `pack_to_loss_mask()` for masked loss computation
  - Includes `pack_sequences()` convenience function and `utilization_stats()`
  - Registered in `lib.rs` with public exports
  - **Tests:** 10 tests including `pack_high_utilization` (>50%), `pack_near_full_utilization` (>95%), `pack_go_game_lengths`, `pack_vs_naive_padding` — all pass

- [x] **T3: Backprop warmup scheduler** — Ramp compute depth during training
  - `BackpropWarmupConfig` struct with `min_steps`, `max_steps`, `warmup_ratio` in `optimizer.rs`
  - `compute_depth(current_step, total_steps)` method with linear ramp formula
  - Default: min=1, max=10, warmup_ratio=0.1 (10% of training)
  - Exported via `lib.rs`
  - **Tests:** `test_backprop_warmup_ramp`, `test_backprop_warmup_zero_ratio`, `test_backprop_warmup_full_ratio` — all pass

- [x] **T4: Learned initial states** — Non-zero init for recurrent carry
  - `trunc_normal_init(data, std, rng)` using Box-Muller transform in `optimizer.rs`
  - Truncates to ±2·std (matches HRM-Text zL_init strategy)
  - `has_nonzero(data)` utility for verification
  - Exported via `lib.rs`
  - **Tests:** `test_trunc_normal_produces_nonzero`, `test_trunc_normal_bounded`, `test_trunc_normal_different_across_runs`, `test_has_nonzero_utility` — all pass

- [x] **T5: Integration tests** — Verify each technique
  - T1: `test_adam_atan2_bounded_updates` (finite, bounded < 10.0 with large grads), `test_adam_atan2_update_bounded_by_pi` (|atan2| ≤ π/2)
  - T2: 10 sampler tests — `pack_near_full_utilization` (>95%), `pack_go_game_lengths`, `pack_vs_naive_padding`
  - T3: `test_backprop_warmup_ramp` (0→1, 50→~6, 100→10, 500→10)
  - T4: `test_trunc_normal_produces_nonzero`, `test_trunc_normal_bounded` (±2·std), `test_trunc_normal_different_across_runs`
  - Location: `crates/riir-gpu/src/optimizer.rs` (tests module), `crates/riir-gpu/src/sampler.rs` (tests module)

- [x] **T6: Benchmark** — Measure training impact
  - T1: `test_adam_atan2_vs_adamw_convergence` — both converge on f(x)=x²; AdamW faster but atan2 bounded
  - T2: `pack_vs_naive_padding` — LPT beats naive one-per-pack padding
  - Full benchmark suite deferred to integration with Go training pipeline (Plan 084)
  - Location: inline in optimizer/sampler test modules

## Architecture

```text
┌──────────────────────────────────────────────────────┐
│                Training Pipeline                      │
│                                                      │
│  ┌─────────────┐   ┌───────────────┐                │
│  │ Multipack   │──▶│ Batch         │                │
│  │ LPT Sampler │   │ Construction  │                │
│  │ (T2)        │   └───────┬───────┘                │
│  └─────────────┘           │                         │
│                            ▼                         │
│  ┌─────────────────────────────────────────────┐    │
│  │            Training Loop                     │    │
│  │                                              │    │
│  │  bp_steps = warmup(step, total) (T3)         │    │
│  │  for i in 0..bp_steps:                       │    │
│  │    loss = forward(batch)                     │    │
│  │    Adam-atan2.step() (T1)                    │    │
│  │                                              │    │
│  │  Carry: learned_init() (T4)                  │    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Adam-atan2 over standard Adam** | Bounded updates prevent LoRA weight explosion. No epsilon tuning. Proven in 1B-scale pretraining. |
| **LPT over first-fit packing** | ~99.5% vs ~70% utilization. O(n log n log k) complexity. Balances quadratic attention cost. |
| **Warmup as generic scheduler** | Same formula applies to any variable-depth compute. Reusable across MCTS, HLA, game training. |
| **Truncated normal over zero init** | HRM-Text uses std=1.0 truncated normal for zL_init. Gives richer starting state for recurrent modules. |
| **No EMA in initial version** | EMA is useful but adds complexity. Ship atan2 first, add EMA later if needed. |

## Non-Goals

- HRM architecture implementation — outside our scope (we do LoRA, not pretraining)
- FlashAttention 3 kernels — CUDA/Hopper specific, we use wgpu/Metal
- FSDP2 distributed training — single GPU only
- PrefixLM two-pass kernel — we already have bidirectional attention
- Full pretraining framework — not applicable to our use case

## Dependencies

- T1 → T5, T6 (tests/benchmarks need kernel)
- T2 → T5 (sampler needs utilization test)
- T3 → T5 (warmup needs correctness test)
- T4 → T5 (init needs verification test)

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Adam-atan2 doesn't help for LoRA | HRM-Text validates on large scale. For LoRA, bounded updates still prevent explosion. If no gain, revert is trivial (1 line). |
| LPT overhead for small batches | For Go positions (361 moves), batches are small. LPT overhead may exceed padding savings. Measure first. |
| Warmup schedule too aggressive | Configurable min/max/ratio. Default to conservative values from HRM-Text. |
| Learned init causes instability | std=1.0 is what HRM uses. For low-rank LoRA, may need smaller std. Configurable. |

## Success Criteria

- [x] Adam-atan2 WGSL kernel passes unit tests (bounded updates) — 3 tests pass
- [x] Multipack sampler achieves >90% utilization on near-full batches — 10 tests pass
- [x] Warmup schedule produces correct ramp from min to max steps — 3 tests pass
- [x] Learned init produces non-zero, stable initial states — 4 tests pass
- [x] All existing tests still pass — 281/281 riir-gpu lib tests pass
- [x] Training benchmark: Adam-atan2 converges on f(x)=x² (bounded, slower but stable)

## References

- Source: https://github.com/sapientinc/HRM-Text
- Research: `.research/048_HRM_Text_Hierarchical_Recurrent_Pretraining.md`
- HRM-Text optimizer: `.raw/HRM-Text/models/adam_atan2.py`
- HRM-Text sampler: `.raw/HRM-Text/multipack_sampler.py`
- HRM-Text HRM model: `.raw/HRM-Text/models/baselines/hrm_nocarry_bp_warmup.py`
- Related: Plan 082 (RowNormM), Plan 066 (D2F), Plan 057 (HLA)