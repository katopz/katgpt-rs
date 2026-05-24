# Research 86: RTPurbo — Retrieval Head Sparse Attention via Low-Dimensional Indexing

> **Paper:** [Full Attention Strikes Back: Transferring Full Attention into Sparse within Hundred Training Steps](https://arxiv.org/pdf/2605.16928) — Zhou, Li, Tang, Li, Liu, Tao, Qu, Yao, Ma (Nanjing Univ + Alibaba), May 2026
> **Date:** 2026-05, distilled 2026-05
> **Related Research:** 68 (DashAttention), 42 (SP-KV), 020 (TurboQuant), 063 (OCTOPUS), 065 (RotorQuant), 070 (GDN2), 022 (Lighthouse Attention)
> **Related Plans:** 126 (RTPurbo head-wise sparse decode)
> **Depends on:** Plan 106 (DashAttention α-entmax), Plan 044 (PFlash), Plan 070 (SP-KV)
> **Supersedes:** None — complements DashAttention with head-level routing
> **Feature Gate:** `rt_turbo` (opt-in, requires proof of gain over DashAttention)

---

## Executive Summary

RTPurbo demonstrates that full-attention LLMs are **intrinsically sparse** and can be converted to highly sparse inference with only ~600 training steps (~1M label tokens). Three key insights:

1. **Head specialization**: Only ~15% of attention heads ("retrieval heads") truly need full long-context access. The rest ("local heads") focus on local context + attention sinks.
2. **Low-dimensional retrieval subspace**: Long-range retrieval is governed by low-frequency RoPE components. A 16-dimensional projector achieves >90% recall.
3. **Dynamic top-p > fixed top-k**: Token budgets are strongly query-dependent. Top-p preserves >93% attention mass with up to 97% sparsity.

**Results**: 9.36× prefill speedup at 1M context, 2.01× decode speedup, near-lossless accuracy on LongBench, RULER (up to 512K), AIME, MMLU-PRO.

---

## Verdict: SELECTIVE ADOPTION — Distill head-calibration + dynamic top-p into DashAttention decode path

**Why not wholesale adoption:**
- RTPurbo's custom GPU kernels (histogram top-p, fused sparse decode) belong in riir-ai, not katgpt-rs
- The two-stage training pipeline (projection training + self-distillation) is a riir-ai LoRA training concern
- Prefill is already dense for retrieval heads — our PFlash/DashAttention prefill is already faster

**What we distill:**
1. **Offline head calibration** — one-pass needle-based scoring to classify heads as retrieval vs local
2. **Dynamic top-p token selection** — replace DashAttention's fixed-chunk entmax with cumulative-mass thresholding for decode
3. **Low-dim pre-RoPE projection** — 16-dim W_Q/W_K for retrieval heads (replaces full-dim scoring)
4. **Self-distillation loss** — top-10 logit KL alignment (research only, riir-ai training)

**Feature gate:** `rt_turbo` — opt-in, gated behind DashAttention. Must prove gain over `dash_attn` alone before default-on promotion.

---

## Paper Core

### 1. Head Specialization as Natural Prior

Prior work (DuoAttention, RazorAttention) shows attention heads specialize:
- **Retrieval heads**: Attend to semantically related distant tokens (~15% of heads, concentrated in later layers)
- **Local heads**: Process only local context + attention sinks (~85% of heads)

**Offline calibration** (one long document sufficient):
- Insert identical "needle" span at beginning and end of a long document
- Measure attention mass from later needle to earlier needle per head
- Score: `R_h = (1/|N_post|) Σ_{t∈N_post} Σ_{j∈N_pre} A_h(t, j)`
- Heads with R_h above threshold → retrieval set H_ret

**Stability**: Head behavior is input-agnostic. Single calibration run sufficient.

### 2. RoPE Induces Compressible Retrieval Geometry

RoPE score decomposition:
```
s(m,n) = Σ_i [a_i(q,k) cos(θ_i·Δ) + b_i(q,k) sin(θ_i·Δ)]
```

- **High-frequency** components (large θ_i): vary rapidly with distance Δ → noise at long range
- **Low-frequency** components (small θ_i): change smoothly → preserve retrieval signals

**Key result**: 16-dimensional low-rank projection achieves >90% recall. Projection applied to **pre-RoPE** representations:
```
s_h(m,n) = (W^Q_h · q^{pre}_{m,h})ᵀ · (W^K_h · k^{pre}_{n,h})
```

This is the critical design choice: project BEFORE RoPE injection, not after.

### 3. Dynamic Top-p Selection

Fixed top-k is fundamentally mismatched to query-dependent retrieval:
- Diffuse queries (e.g., "Galápagos"): need ~8.5K tokens for 90% mass
- Concentrated queries (e.g., NIAH): 2 tokens recover 97% mass

**Top-p rule**: Select minimum tokens S_h(m) such that cumulative attention mass ≥ p (default 0.9):
```
S_h(m) = Top-P(s_h(m, ·), p)
```

**Ablation** (RULER 64K): top-p=0.9 → 85.49% avg vs top-k=4096 → 70.53% avg. Dynamic budget is critical.

### 4. Two-Stage Training Pipeline

**Stage 1 — Projection training** (~600 steps, 840K params, backbone frozen):
- KL divergence between full-dimension and projected attention distributions
- Per retrieval head: 2 × 128 × 16 = 4096 params
- Converges in ~600 steps on 8K long sequences (avg 48K tokens)

**Stage 2 — Self-distillation** (~600 steps, top-10 logits only):
- Sparse student aligns with dense teacher's next-token predictions
- Top-10 logit KL: bypasses data distribution bias of standard SFT
- Only ~1.2M label tokens total

### 5. Hardware-Aware Decode Kernel

**Sort-free top-p via histogram** (GPU):
- Partition K into blocks, each CTA scores one block → block-level (m_b, ℓ_b)
- Atomic deposit into 256-bin histogram (1KB per head, O(1) memory)
- Last CTA scans histogram from highest bin → finds threshold → writes block mask
- Fuses scoring + selection into single kernel launch

**Bandwidth-optimized sparse decode**:
- Single-warp CTA, no shared memory (all registers)
- 2-token unrolled inner loop with half2 vectorized loads
- Cross-split reduction via same atomic-counter technique

---

## Distillation Map — Our Architecture

### What Maps Directly

| RTPurbo Concept | Our Equivalent | Action |
|-----------------|---------------|--------|
| Offline head calibration | None yet | **Add** — `HeadCalibration` struct with per-head retrieval scores |
| Retrieval vs local head split | DashAttention treats all heads uniformly | **Extend** — per-head routing mode in DashAttention decode |
| 16-dim pre-RoPE projection | ChunkSummaryQuery (full-dim learned) | **Add** — low-dim W_Q/W_K per retrieval head |
| Dynamic top-p selection | α-entmax adaptive support | **Complement** — top-p for token-level, entmax for block-level |
| Top-10 KL self-distillation | SDAR gated loss (Plan 072/073) | **Extend** — add top-10 KL mode to riir-ai training |
| Sliding window + sinks (local heads) | PFlash window + sink tokens | **Reuse** — already implemented |

### What Doesn't Map

| RTPurbo Concept | Reason | Action |
|-----------------|--------|--------|
| Custom GPU histogram kernel | riir-ai territory | Skip for katgpt-rs |
| 1M context benchmarks | CPU inference targets ≤128K | Scale results proportionally |
| Qwen3-30B-A3B specific | Model-agnostic design | Our calibration works for any model |
| Prefill dense for retrieval heads | PFlash already handles this | No change needed |

### Novel Combination for Our Stack

**RTPurbo + DashAttention hybrid decode path:**

```
For each attention layer:
  1. Load head calibration (offline computed)
  2. For local heads (85%):
     - Sliding window (8192) + sink tokens (4) only
     - Skip full KV scan entirely
  3. For retrieval heads (15%):
     - Low-dim (16-d) pre-RoPE projection → token scores
     - Dynamic top-p selection (cumulative mass ≥ 0.9)
     - Full-dim SDPA on selected tokens only
```

This combines DashAttention's α-entmax block routing with RTPurbo's head-level specialization and token-level top-p.

---

## Benchmark Data (from paper)

### Prefill Speedup (vs FlashAttention-2)

| Context | Speedup |
|---------|---------|
| 32K | 2.83× |
| 64K | 4.25× |
| 128K | 5.92× |
| 256K | 7.47× |
| 512K | 8.62× |
| 1M | 9.36× |

### Decode Sparsity (Layer 25, Qwen3-Coder-30B-A3B)

| Context | Task | Compute Sparsity | Active Tokens | Attn Mass |
|---------|------|------------------|---------------|-----------|
| 32K | NIAH-S | 78.7% | 468.8 | >0.95 |
| 32K | Multi-K | 77.8% | 2,462.1 | >0.96 |
| 64K | NIAH-S | 89.2% | 1,126.8 | >0.93 |
| 64K | Multi-K | 88.7% | 3,316.1 | >0.94 |
| 512K | Multi-V | 97.1% | dynamic | >0.95 |

### Accuracy (RULER 64K)

| Method | CWE | FWE | VT | HotPot | multi-Q | multi-V | multi-K | niah-S | Avg |
|--------|-----|-----|-----|--------|---------|---------|---------|--------|-----|
| Full Attn | 65.3 | 84.0 | 96.8 | 63.4 | 99.5 | 97.6 | 99.7 | 100 | 86.2 |
| RazorAttn | 66.0 | 83.8 | 91.5 | 62.8 | 99.0 | 95.2 | 98.7 | 99.7 | 85.1 |
| MInference | 61.8 | 82.7 | 83.5 | 42.8 | 82.3 | 81.1 | 40.1 | 86.5 | 65.6 |
| Quest | 36.6 | 63.8 | 94.8 | 54.6 | 80.4 | 75.9 | 78.9 | 87.9 | 70.6 |
| **RTPurbo top-p** | **65.1** | **81.4** | **94.6** | **62.6** | **99.7** | **97.5** | **98.6** | **99.9** | **85.5** |
| RTPurbo top-k | 59.9 | 62.0 | 69.6 | 56.0 | 98.2 | 97.6 | 50.7 | 76.5 | 70.5 |

**Key takeaway**: top-p vs top-k is +15 points on RULER 64K. Dynamic budget is not optional.

### Training Cost

| Stage | Params | Steps | Tokens | Time |
|-------|--------|-------|--------|------|
| Stage 1 (projection) | 840K | ~600 | ~30M | Minutes |
| Stage 2 (distillation) | Full model | ~600 | ~180M (~1.2M labels) | Minutes |
| **Total** | — | **~1200** | **~1.2M labels** | **<1hr on 8×H20** |

---

## Key Ablations

### Retrieval Head Ratio

| Ratio | MMLU-PRO Math | RULER multi-K | Sparsity | Verdict |
|-------|---------------|---------------|----------|---------|
| 10% | 79.3 | 97.4 | High | Too few heads |
| **15%** | **88.2** | **98.8** | **High** | **Best balance** |
| 30% | 88.2 | 98.6 | Lower | No accuracy gain, 2× params |

### Low-Dimension Size

| Dim | MMLU-PRO Math | Recalled Tokens (64K) | Verdict |
|-----|---------------|----------------------|---------|
| 4 | 89.1 | 45,280 | Poor fit → over-recalls → less sparse |
| **16** | **88.2** | **25,725** | **Best sparsity-quality tradeoff** |
| 32 | 88.2 | 28,464 | No accuracy gain, slightly worse sparsity |

---

## GOAT Proof Design (Plan 126)

To promote `rt_turbo` from opt-in to default-on, we need:

1. **Proof 1 — Calibration stability**: Single-sequence calibration produces identical head partition to 10-sequence calibration (permutation invariant)
2. **Proof 2 — Top-p > top-k on our micro model**: RULER-style synthetic test showing top-p preserves >90% attention mass with fewer tokens
3. **Proof 3 — Low-dim recall**: 16-dim projection achieves >85% recall of top-256 full-dim tokens
4. **Proof 4 — Decode throughput**: Head-gated decode (local heads skip full scan) is faster than uniform decode
5. **Proof 5 — Accuracy preservation**: Micro benchmark loss within 1% of dense baseline
6. **Proof 6 — Compatibility**: Works with existing SpectralQuant, OCTOPUS, HybridOctPQ KV caches

---

## Risks & Limitations

1. **Model-specific calibration**: Head partition may not transfer across model families. Mitigation: calibration is cheap (one forward pass), can be per-model.
2. **CPU SIMD challenge**: The histogram top-p kernel is GPU-optimized. CPU equivalent needs sort-free cumulative mass estimation. Our `entmax_1p5` already does this — can reuse the sort + cumsum approach.
3. **Retrieval heads still dense at prefill**: RTPurbo doesn't solve prefill sparsity for retrieval heads. Our DashAttention chunk routing already handles this.
4. **Reasoning task asymmetry**: Reasoning tasks have short prompts but long decode (up to 32K). The decode speedup is where RTPurbo shines, which is our bottleneck too.
5. **Feature gate interaction**: Must work alongside `spectral_quant`, `hybrid_oct_pq`, `gdn2_attention`, `lt2_looped`. Test all combinations.

---

## References

- DuoAttention (Xiao et al., 2025) — retrieval/streaming head partition
- RazorAttention (Tang et al., 2025) — training-free head-wise KV compression
- SnapKV (Li et al., 2024) — KV cache compression via local query relevance
- Quest (Tang et al., 2024) — query-aware page-level sparsity
- MInference (Jiang et al., 2024) — dynamic sparse attention patterns
- FlexPrefill (Lai et al., 2025) — context-aware sparse prefill
- DeepSeek Sparse Attention (DeepSeek-AI, 2025) — native sparse training
- Kimi Delta Attention (Kimi Team, 2025) — linear attention architecture