# Research: HRM-Text — Hierarchical Recurrent Pretraining (48)

> Source: [HRM-Text: Efficient Pretraining Beyond Scaling](https://sapientinc.github.io/HRM-Text/) by Sapient Inc, 2025
> Local: `.raw/HRM-Text/` (PyTorch training code)
> Date: 2025, distilled 2025-07
> **Verdict: SELECTIVE VALUE — The HRM architecture itself is a GPU-scale pretraining technique (not applicable to our modelless/model-based training stack). However, 4 specific techniques distill cleanly: PrefixLM attention (we already have this for D2F), Adam-atan2 optimizer (simple drop-in), multipack LPT batching (for training data efficiency), and backprop warmup scheduling (for recurrent compute budgeting).**

## TL;DR

HRM-Text trains a 1B text model with ~$1000 (130-600x less compute, 150-900x less data) using:
1. **Hierarchical Recurrent Model (HRM)** — Two modules (H-level, L-level) that share weights across multiple forward cycles
2. **PrefixLM sequence packing** — Bidirectional attention on prefix, causal on response
3. **FlashAttention 3** — Custom two-pass PrefixLM kernel (forward + backward)
4. **FSDP2 training** — PyTorch distributed with sharded checkpoints
5. **Multipack sampler** — LPT (Longest Processing Time) scheduling for balanced attention
6. **Adam-atan2 optimizer** — Uses `atan2(momentum, v_sqrt)` instead of `momentum / v_sqrt`
7. **Backprop warmup** — Starts with fewer recurrent steps, increases over training

Key benchmark results at 1B: GSM8k 84.7%, MATH 56.5%, MMLU 60.7% — competitive with much larger models.

## Architecture Breakdown

### HRM Core: H-Level + L-Level Recurrence

The HRM has two modules, both are standard Transformer blocks:
- **H-level** (`z_H`): Processes the main input sequence
- **L-level** (`z_L`): Processes a learned initial state `zL_init`, receives H's output via additive injection

Execution pattern:
```
for i in H_cycles:            # e.g., 4
    for k in L_cycles_range:  # e.g., 3 per H cycle
        z_L = L_level(z_L + z_H)  # Input injection: simple addition
    z_H = H_level(z_H + z_L)      # Feedback: L feeds back to H
```

Total effective layers = H_cycles × L_cycles × actual_layers_per_module
- L (0.6B): H_cycles=4, L_cycles=3, 12 layers split → 72 effective layers
- XL (1B): H_cycles=4, L_cycles=3, 16 layers split → 96 effective layers

**Key insight:** Weight sharing across cycles means the model has far fewer parameters than the effective depth suggests. The recurrent structure allows adaptive compute via varying `bp_steps`.

### Backprop Warmup (BP Steps)

Not all forward passes need gradients. The model uses a warmup schedule:
```python
bp_steps = bp_min_steps + int(min(1, step / (total_steps * bp_warmup_ratio)) * (bp_max_steps - bp_min_steps))
```
Default: starts at `bp_min_steps=2`, ramps to `bp_max_steps=5` over `bp_warmup_ratio` of training.

Gradients are only computed for the last `H_bp_steps` H-cycles and last `L_bp_steps` L-cycles. Early cycles run in `torch.no_grad()` mode — saving significant compute during early training.

### PrefixLM Attention (Two-Pass)

Training uses a custom FlashAttention 3 kernel with two passes:
1. **Pass 1 (bidirectional):** Full attention over prefix (instruction) tokens
2. **Pass 2 (causal):** Standard causal attention for response tokens, attending to all prefix + prior response

The backward pass also splits into bidirectional and causal gradients, summed for key/value gradients.

### Multipack Sampler with LPT

The multipack sampler uses LPT (Longest Processing Time First) scheduling to pack sequences into fixed-size batches:
1. Binary search to find the max number of sequences that fit in `n_nodes × batch_max_length`
2. LPT assigns sequences to nodes (min-heap, assign to least-loaded node)
3. Achieves ~99.5% token-slot utilization
4. Balances quadratic attention work across nodes

Time complexity: O(n log n log k) where n = max sequences per batch, k = number of nodes.

### Adam-atan2 Optimizer

Replaces the standard Adam update:
```
# Standard Adam:
update = lr * momentum / (v_sqrt + eps)

# Adam-atan2:
update = lr * atan2(momentum, v_sqrt)
```

Benefits:
- Bounded update magnitude (atan2 returns [-π/2, π/2])
- No need for epsilon tuning (atan2 handles near-zero denominators gracefully)
- Built-in gradient clipping (bounded by atan2 range)
- Includes EMA support for evaluation weights

### Input Injection (Additive)

H → L communication uses simple addition, not gating:
```python
z_L = L_level(z_L + z_H)  # Add, not gate
z_H = H_level(z_H + z_L)  # Add, not gate
```

The paper notes TODO: try GRU/gating or "fixed" gating that doesn't depend on hidden state.

## Distillable Ideas for Our Stack

### D1: PrefixLM Attention — Already Implemented ✅

We already have bidirectional prefill in both `microgpt-rs` and `riir-engine`:
- `forward_bidirectional_positions()` in `dllm.rs`
- `forward_prefill()` with Phase B bidirectional attention
- D2F (Plan 068) uses block-causal + intra-block bidirectional

**What HRM adds:** The two-pass FlashAttention 3 kernel is more efficient than our current implementation. For GPU training, this could be a WGSL optimization. For our CPU inference, our current approach is fine.

**Verdict:** Already covered. No new action needed.

### D2: Adam-atan2 Optimizer — Drop-In for LoRA Training

The `atan2(momentum, v_sqrt)` update is a simple change to our WGSL Adam kernel:

```wgsl
// Standard Adam:
let update = momentum / (v_sqrt + eps);

// Adam-atan2:
let update = atan2(momentum, v_sqrt);
```

Benefits for our LoRA training:
- **No epsilon tuning** — atan2 handles near-zero automatically
- **Bounded updates** — prevents LoRA weight explosion during early training
- **Simpler code** — remove epsilon from config

The HRM-Text code also shows EMA (Exponential Moving Average) support built into the optimizer — we could add this for evaluation weight smoothing.

**Applicable to:** `riir-gpu` WGSL training kernels, `riir-engine` CPU training.

### D3: Multipack LPT Batching — For Training Data Pipeline

Our training data pipeline currently doesn't have smart batching. The LPT approach:
1. Sort sequences by length (descending)
2. Use min-heap to assign to least-loaded batch slot
3. Binary search to find optimal pack size

This is applicable when we batch multiple game replays or Go positions for LoRA training. The current approach likely pads to max length, wasting compute.

**Applicable to:** Go LoRA training (Plan 084), any future batched training in `riir-gpu`.

### D4: Backprop Warmup — For Recurrent Compute Budgeting

The idea of starting with fewer gradient steps and ramping up is applicable to:
- **HLA recurrent state updates** — fewer scan steps early in training
- **MCTS rollout depth** — start shallow, deepen over training
- **Game replay training** — start with fewer replays per batch

The formula is simple:
```rust
let bp_steps = bp_min + ((step as f32 / (total_steps as f32 * warmup_ratio)).min(1.0) * (bp_max - bp_min) as f32) as usize;
```

**Applicable to:** Any training loop that has variable compute depth.

### D5: Learned Initial State (zL_init) — For Recurrent Models

The `zL_init` is a learned parameter (not zero-initialized, but truncated normal) that serves as the starting state for the L-level module. This is similar to how:
- Our HLA carry state needs initialization
- Our game state forward model needs initial belief state
- Our Raven RSM slots need initialization

The insight: **random initialization (not zero) for recurrent state** gives the model a richer starting point.

**Applicable to:** HLA state init, Raven slot init, any learned recurrent state.

### D6: Gated Multi-Head Attention (QKV + Gate)

The attention module uses a gated mechanism:
```python
gate, query, key, value = gqkv_proj(x).split(...)
attn_output = sigmoid(gate) * flash_attn(q, k, v)
```

A single projection produces (gate, Q, K, V), and the sigmoid gate multiplicatively filters attention output. This is cheap (one extra projection dim) and provides learned attention gating.

**Applicable to:** Our transformer attention in both projects.

## What We Don't Need

1. **HRM architecture itself** — We don't do pretraining from scratch. Our model-based path uses LoRA fine-tuning on existing models. The hierarchical recurrent structure is for training foundation models, which is outside our scope.

2. **FlashAttention 3 kernels** — We use wgpu/Metal, not CUDA. Our attention kernels are already optimized for our platform. FA3 is Hopper-specific.

3. **FSDP2 distributed training** — We train on single Apple M-series GPUs. No need for distributed sharding.

4. **data_io pipeline** — We use game replay data and pre-existing datasets, not web-crawled text corpora.

5. **PrefixLM two-pass custom kernel** — Our bidirectional + causal is already implemented differently (separate passes in CPU/wgpu).

## Distillation Priority

| Priority | Technique | Target Project | Complexity | Expected Gain |
|---|---|---|---|---|
| **P1** | Adam-atan2 optimizer | riir-gpu (WGSL kernel) | Low — 1 line change | Better training stability |
| **P2** | Multipack LPT batching | riir-gpu (training pipeline) | Medium — new sampler | ~30% better training throughput |
| **P2** | Backprop warmup | microgpt-rs (MCTS, HLA) | Low — scheduling formula | Faster early training |
| **P3** | Learned initial state | Both (HLA, Raven) | Low — init change | Richer starting states |
| **P3** | Gated attention | riir-engine (transformer) | Medium — new projection | Better attention quality |
| **—** | HRM architecture | N/A | High | Outside our scope |
| **—** | PrefixLM FA3 kernel | N/A | High | Already implemented |

## Key Numbers

### HRM-Text Benchmarks (XL, 1B)
| Benchmark | Score |
|---|---|
| GSM8k | 84.7% |
| MATH | 56.5% |
| DROP | 82.3% |
| MMLU | 60.7% |
| ARC-C | 81.9% |
| HellaSwag | 63.4% |
| Winogrande | 72.4% |
| BoolQ | 86.2% |

### Training Efficiency
- **L (0.6B):** 8 H100s, 50 hours, ~$800
- **XL (1B):** 16 H100s, 46 hours, ~$1,472
- Claims 130-600x less compute, 150-900x less data than traditional pretraining

## Equivariant Optimizers Verdict (Cross-Reference)

The equivariant optimizers code (`.raw/equivariant_optimizers/`) is already distilled in Research 46 and implemented in Plan 082. Key findings:
- RowNormM for embedding/LM head LoRA adapters ✅ implemented
- Per-role learning rate configuration ✅ implemented
- Router centering for expert routing ✅ implemented
- Newton-Schulz (optional) — not needed yet

**No new action needed for equivariant optimizers.** Plan 082 is complete.

## References

- Project: https://github.com/sapientinc/HRM-Text
- Paper: https://sapientinc.github.io/HRM-Text/assets/HRM_Text.pdf
- Model: https://huggingface.co/sapientinc/HRM-Text-1B
- Local code: `.raw/HRM-Text/`
- Related research: `34_D2F_Discrete_Diffusion_Forcing.md` (PrefixLM usage), `28_Higher_order_Linear_Attention.md` (recurrent state), `46_Symmetry_Compatible_Equivariant_Optimizers.md` (optimizer design)
- Related plans: `066_d2f_discrete_diffusion_forcing.md` (block-causal), `082_symmetry_compatible_lora_optimizers.md` (RowNormM, complete)