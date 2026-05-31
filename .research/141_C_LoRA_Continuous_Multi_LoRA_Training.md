# Research: C-LoRA — Continuous Multi-LoRA Training for Continual Learning

> Source: [Multi-LoRA Training for Continual Learning](https://trajectory.ai/field-notes/multi-lora-training-for-continual-learning) by Trajectory AI (collaboration with UC Berkeley Sky Lab, Anyscale)
> Repo: [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)
> Date: 2026-05
> **Verdict: NO GAIN — Training infrastructure idea, not applicable to our inference-first stack.**

---

## TL;DR

Trajectory/SkyRL built C-LoRA: a concurrent, multi-LoRA RL training platform that achieves **2.81× end-to-end experiment throughput** by multiplexing N LoRA adapters on a shared inference + training engine. The key enablers are SGMV fused decode kernels (vLLM), adapter swapping from pinned CPU memory, and cross-job load balancing. While impressive for large-scale RLHF on H200 clusters, this is **training infrastructure** — it does not apply to our inference-time architecture. Our equivalent "continual learning" loop (G-Zero self-play → Freeze/Thaw → LoRA export) already captures the concept at our scale.

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

### Mapping C-LoRA Concepts → Our Stack

| C-LoRA Concept | Our Equivalent | Gap |
|----------------|---------------|-----|
| Multi-LoRA inference (SGMV) | `GpuLoraBuffers` per-adapter A/B matrices | ✅ Have adapter multiplexing at inference time |
| Adapter swap from pinned CPU | `GpuLoraBuffers` load/export cycle | ✅ Can load different adapters between runs |
| Cross-job load balancing | N/A — single-node training | ❌ We don't have distributed training |
| Always-hot engine | N/A — training is batch, not service | ❌ Different paradigm |
| Per-tenant AdapterStore | `GpuLoraAdapter` (A, B, grad, optimizer state) | ✅ Already exists per-adapter |
| Weight sync to inference | `export_lora()` → `load_lora()` | ✅ Already working |
| Continual learning loop | G-Zero self-play → Freeze/Thaw → LoRA export | ✅ Already captured |

### Why NO GAIN

1. **We're inference-first, not training-service.** katgpt-rs has no training loop. riir-ai's `riir-gpu` trains single adapters in batch mode, not as a persistent multi-tenant service.

2. **No CUDA/Multi-GPU.** We're on wgpu/Metal (Apple Silicon). SGMV kernels require CUDA tensor cores. Our `gemv_cubecl.rs` already handles single-adapter LoRA merge efficiently on Metal.

3. **Scale mismatch.** C-LoRA targets 8×H200 clusters with 4B+ parameter models. Our training is on micro-transformers (V=32, D=16) for game-domain adapters.

4. **Continual learning already captured.** Our G-Zero pipeline (Plan 049) already does continual learning: self-play → validator feedback → LoRA update → better play. Freeze/Thaw (Plan 092) handles the knowledge retention problem. Neither needs multi-tenant training infrastructure.

5. **No multi-experiment sweep need.** C-LoRA's value is running N hyperparameter sweeps simultaneously. Our experiments are game-domain, single-config runs. We don't need experiment multiplexing.

### What We Already Have That's Better for Our Scale

| Our Feature | C-LoRA Equivalent | Our Advantage |
|-------------|-------------------|---------------|
| `GpuLoraBuffers` (6 adapters per layer) | AdapterStore per tenant | We already multiplex 6 adapter targets per layer |
| `TuningMethod` enum (LoRA/QLoRA/IA3/OFT/SPEFT) | Single LoRA only | We support 5 PEFT methods |
| G-Zero self-play loop | RL training loop | Our loop is domain-specific and validator-guided |
| Freeze/Thaw pipeline | N/A | We handle knowledge retention explicitly |
| TIES merging (Plan 094) | N/A | We can merge trained adapters |

---

## Honest Assessment

C-LoRA is a well-executed systems paper for **large-scale RLHF infrastructure**. The engineering is impressive (SGMV kernels, pinned memory adapter swapping, cross-job load balancing). But it solves a problem we don't have:

- We don't run distributed RL training on GPU clusters
- We don't need multi-experiment sweeps
- Our training is single-node, game-domain, batch-mode
- Our "continual learning" is G-Zero self-play, not live production feedback

**If we ever scale to multi-GPU RLHF training (unlikely given our game-domain focus), the adapter swapping pattern from C-LoRA could inform our design. But that's a Phase 6+ concern per our execution roadmap in Research 003.**

---

## References

- Article: https://trajectory.ai/field-notes/multi-lora-training-for-continual-learning
- Repo: https://github.com/NovaSky-AI/SkyRL
- SGMV paper: https://arxiv.org/pdf/2310.18547
- Related our research: `004_LoRA_Architecture_Verdict.md`, `037_REAP_Model-Based_Modelless_Duality.md`
- Related our plans: `049_g_zero_self_play.md` (G-Zero), `092_self_play_freeze_thaw.md` (Freeze/Thaw), `094_memo_reflections_ties_merging.md` (TIES)
