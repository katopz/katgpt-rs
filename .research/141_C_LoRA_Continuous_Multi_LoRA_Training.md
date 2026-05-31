# Research: C-LoRA — Continuous Multi-LoRA Training for Continual Learning

> Source: [Multi-LoRA Training for Continual Learning](https://trajectory.ai/field-notes/multi-lora-training-for-continual-learning) by Trajectory AI (collaboration with UC Berkeley Sky Lab, Anyscale)
> Repo: [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)
> Date: 2026-05
> **Verdict: CONDITIONAL GAIN — Fused multi-LoRA dispatch kernel reduces per-layer dispatch count 6×. Multi-experiment scheduling not applicable.**

---

## TL;DR

Trajectory/SkyRL built C-LoRA: a concurrent, multi-LoRA RL training platform that achieves **2.81× end-to-end experiment throughput** by multiplexing N LoRA adapters on a shared inference + training engine. The key enablers are SGMV fused decode kernels (vLLM), adapter swapping from pinned CPU memory, and cross-job load balancing.

Two distinct ideas to distill:

1. **Multi-experiment scheduling** — NO GAIN. We're inference-first, single-node, game-domain. Not applicable.
2. **Fused multi-LoRA kernel (SGMV)** — GAIN. Our current per-layer LoRA merge does 12 separate GPU dispatches (6 targets × 2 passes). Fusing into 2 dispatches per layer cuts dispatch overhead 6×. On Metal, unified memory makes adapter swap zero-cost (just rebind). CubeCL plane kernels already give us the SIMD primitive.

---

## What C-LoRA Does

### Problem
Traditional RL training runs one experiment per GPU allocation. Each run requires:
- 30+ min cold start (checkpoint load, distributed init, inference warmup)
- Single-tenant GPU occupancy
- Imbalanced trainer/generator utilization (synchronous stalls)

### Solution: Always-Hot Multi-LoRA Service

| Component | Design |
|-----------|--------|
| **Inference** | vLLM with SGMV kernel — all adapters hot-loaded in GPU memory, decode steps mix tokens from different adapters in one batch |
| **Weight Sync** | Updated LoRA weights loaded in-place; other tenants keep decoding during updates |
| **Training** | Single-adapter GPU training; inactive adapters sit in pinned CPU memory, swapped in round-robin |
| **AdapterStore** | Per-tenant: LoRA params + FP32 master weights + optimizer moments + gradient buffers |

### Key Results (Qwen3-4B, 8×H200, GSM8K with Tools)

| Metric | Serial (N=1) | Multi-LoRA (N=8) | Speedup |
|--------|-------------|-------------------|---------|
| Final Experiment Time | 15244 s | 5433 s | **2.81×** |
| Mean Experiment Time | 8575 s | 5249 s | 1.63× |
| Step Time | 191 s | 500 s | 2.62× slower |
| Reward Accuracy (step 9) | >90% | >90% | No regression |

### Tradeoffs
- Per-step latency increases sub-linearly (N=8 → 2.62× slower steps)
- First experiment finishes 1.97× slower than serial baseline
- Sweet spot: N=2–4 (15–59% step latency increase for 1.73–2.49× throughput)

---

## Distillation to Our Architecture

### Idea 1: Multi-Experiment Scheduling — NO GAIN

| C-LoRA Assumption | Our Reality |
|---|---|
| 8×H200 GPU clusters | Apple Silicon single-node (wgpu/Metal) |
| 4B+ param LLMs | Micro-transformers (V=32, D=16) |
| Distributed RLHF training service | Batch-mode game-domain LoRA training |
| Multi-experiment hyperparameter sweeps | Single-config game-domain runs |
| CUDA SGMV tensor cores | cubecl gemv plane kernels |

We already have the scheduling concepts:
- **Multi-LoRA inference**: `GpuLoraBuffers` handles 6 adapter targets per layer
- **Continual learning**: G-Zero self-play (Plan 049) + Freeze/Thaw (Plan 092)
- **Adapter composition**: TIES merging (Plan 094)

### Idea 2: Fused Multi-LoRA Dispatch — GAIN

**The problem:**
Current `dispatch_lora_merge()` (forward.rs:2707) does 12 GPU dispatches per transformer layer:
- 6 LoRA targets (Q, K, V, O, Mlp1, Mlp2) × 2 passes each (A @ x, then B @ intermediate)
- Each dispatch = `begin_compute_pass()` + `set_pipeline()` + `set_bind_group()` + `dispatch_workgroups()` ≈ 5-10μs on Metal
- 26 layers × 12 dispatches = **312 dispatches per forward pass**
- Pure dispatch overhead: 312 × 7.5μs ≈ **2.3ms per forward pass**

**The fix — SGMV-style fused batched LoRA:**
Pack all 6 adapters into contiguous buffers, launch 2 kernels per layer instead of 12:

```
Dispatch 1: sgmv_lora_a — all 6 A matrices @ input → [6 × rank] intermediates
Dispatch 2: sgmv_lora_b — all 6 B matrices @ intermediates → accumulated deltas, add to base output
```

Result: 26 × 2 = **52 dispatches** (6× reduction). Saves ~1.9ms of dispatch overhead per forward pass.

**Why Metal makes this better than CUDA:**
- **Unified memory** — no pinned vs device memory distinction. Adapter swap = rebind pointer, zero-copy.
- **SIMD-group ops** — `simdgroup_matrix` (8×8×8 MMA) via CubeCL CMMA, equivalent to CUDA `wmma`.
- **Plane cooperative GEMV** — our `gemv_plane_f32` already uses `plane_sum()` for cooperative dot products. Extend to batched multi-adapter with adapter stride.

**Current kernel infrastructure we can reuse:**
- `gemv_plane_f32` — cooperative dot product with SIMD reduction (primary)
- `gemv_tile_f32` — shared memory tiled fallback
- `matmul_tiled_f32` — 16×16 tiled matmul for prefill
- CODA epilogue pipeline — fused GEMV + activation + residual in single dispatch
- `GpuLoraBuffers::adapter_index()` — maps (layer, target) → flat index, already packs 6 adapters

### What We Already Have vs What's Needed

| Component | Status | Action |
|-----------|--------|--------|
| Per-adapter A/B buffers in unified memory | ✅ | No change |
| Per-adapter bind groups | ✅ | Replace with packed batch bind groups |
| `dispatch_lora_merge()` 12×/layer | ✅ Current | Replace with `dispatch_lora_fused()` 2×/layer |
| CubeCL plane GEMV kernel | ✅ | Extend with adapter stride |
| CubeCL tiled matmul | ✅ | Extend with batched LoRA B pass |
| CODA epilogue (fused activation) | ✅ | Wire into fused dispatch |
| `GpuForwardPass::new()` bind group setup | ✅ | Add fused bind group variant |
| Training backward pass | ✅ | No change (backward uses per-adapter already) |

---

## Honest Assessment

C-LoRA's multi-experiment scheduling is irrelevant to our stack. But the **SGMV kernel insight** — batching multiple LoRA adapter operations into fewer GPU dispatches — is directly applicable and achievable with our existing CubeCL infrastructure. The gain is primarily in **dispatch overhead reduction** (6× fewer kernel launches per layer), not in compute throughput. This matters most for decode where per-token latency is critical.

Unified memory on Apple Silicon gives us an additional advantage C-LoRA doesn't have on CUDA: adapter state swap is a pointer rebinding, not a DMA transfer.

**If GOAT proves the fused dispatch is faster (or at least no slower), it should be default-on. If it regresses, the existing per-adapter dispatch remains as fallback.**

---

## References

- Article: https://trajectory.ai/field-notes/multi-lora-training-for-continual-learning
- Repo: https://github.com/NovaSky-AI/SkyRL
- SGMV paper: https://arxiv.org/pdf/2310.18547
- Related our research: `004_LoRA_Architecture_Verdict.md`, `037_REAP_Model-Based_Modelless_Duality.md`
- Related our plans: `049_g_zero_self_play.md` (G-Zero), `092_self_play_freeze_thaw.md` (Freeze/Thaw), `094_memo_reflections_ties_merging.md` (TIES)
- Related our code: `riir-ai/crates/riir-gpu/src/gemv_cubecl.rs`, `riir-ai/crates/riir-gpu/src/forward.rs` (L2707 dispatch_lora_merge)
