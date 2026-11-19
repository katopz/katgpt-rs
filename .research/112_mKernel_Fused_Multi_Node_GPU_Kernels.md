# Research 112: mKernel — Fused Multi-Node GPU Kernels (GPU-Driven Compute-Communication Overlap)

> **Source:** [mKernel: Fast Multi-GPU, Multi-Node Fused Kernels](https://uccl-project.github.io/posts/mkernel/) — Ziming Mao & UCCL Team, 2026-05-25
> **Code:** [github.com/uccl-project/mKernel](https://github.com/uccl-project/mKernel) (MIT, CUDA Hopper sm_90a)
> **Local:** `.raw/mKernel/` (5 CUDA kernels + include headers)
> **Date:** 2026-05-26
> **Related Research:** 066 (TileRT), 067 (CODA), 077 (ThunderKittens), 092 (Five Sharding Dimensions), 059 (MoE+SD), 071 (DashAttention), 073 (LT2), 091 (SpecHop)
> **Related Plans:** 150 (katgpt-rs — conceptual alignment only), 147 (riir-ai — Super-GOAT kernel fusion if GPU training scales)
> **Verdict: LOW DIRECT VALUE FOR INFERENCE — mKernel targets multi-node NVIDIA H200 training with RDMA/NVLink, which is completely orthogonal to our single-device CPU SIMD + Metal inference stack. However, three architectural patterns are conceptually valuable: (1) persistent kernel with SM specialization maps to our DecodeStage heterogeneous dispatch, (2) tile-granularity compute-communication overlap validates our TileRT-style speculative pipeline overlap thesis, (3) the five fused kernels are a taxonomy for what a "megakernel" looks like — useful if we ever scale to multi-device inference. No feature gate needed in katgpt-rs. Conditional tracking in riir-ai for future GPU kernel fusion work.**

---

## TL;DR

mKernel fuses intra-node NVLink, inter-node RDMA, and dense compute into **single persistent CUDA kernels** running on multi-GPU, multi-node clusters (2×8 H200). Five kernels: AllGather+GEMM, GEMM+AllReduce, MoE Dispatch+GEMM, Ring Attention, GEMM+ReduceScatter. Each achieves significant speedups over NCCL/Triton-distributed/Transformer-Engine baselines by eliminating host-driven orchestration and overlapping at tile/chunk granularity.

**Why it matters to us:** It doesn't — for inference. We run single-device (Apple Silicon CPU SIMD + Metal GPU). mKernel is training-time multi-node. But the *principle* of persistent specialization (CTAs self-assign roles: compute / intra-comm / inter-send / inter-reduce) is the same pattern we already use in `DecodeStage` (Prefill / Draft / Verify / Sample), and the overlap thesis validates our existing SpecHop + TileRT pipeline design.

---

## 1. Key Ideas

### 1.1 Host-Driven Communication is the Bottleneck

In production AI training:
- Communication = **43.6% of forward pass, 32% of end-to-end training time**
- Inter-device communication = **up to 47% of total execution time** in MoE models

The root cause: CPU-mediated control. Each `cudaLaunchKernel`, host-side "all writes done" check, inter-stream event shows up as pipeline bubbles at GB300 NVL72 speeds (720 PFLOP/s).

**Our analog:** At BS=1 inference, our SIMD kernels finish in ~50-300ns but function call overhead + cache misses between stages add 100-300ns (15-46% overhead). Same principle, different timescale. Research 066 (TileRT) already identified this.

### 1.2 GPU-Driven Communication: The mKernel Solution

Instead of host → library → kernel:
```
GPU persistent kernel: compute CTA + intra-comm CTA + inter-send CTA + inter-reduce CTA
```

All running concurrently inside one kernel launch. Communication happens at tile/chunk granularity, overlapped with compute.

**SM Specialization:**
- Compute CTAs: dense matmul / flash attention
- Intra-comm CTAs: NVLink data exchange within node
- Inter-send CTAs: RDMA writes to remote nodes
- Inter-reduce CTAs: RDMA reduction from remote nodes
- Split is tunable per shape

### 1.3 The Five Fused Kernels (Taxonomy)

| Kernel | Fusion | Communication Pattern | Domain |
|--------|--------|----------------------|--------|
| **AllGather + GEMM** | Gather shards → tile-by-tile matmul | AllGather (NVLink+RDMA) | Tensor parallelism |
| **GEMM + AllReduce** | Matmul → streaming reduction | AllReduce (NVLink+RDMA) | Tensor parallelism |
| **MoE Dispatch + GEMM** | All-to-All token routing → expert matmul | All-to-All (NVLink+RDMA) | Expert parallelism |
| **Ring Attention** | KV ring rotation → flash attention | Ring (NVLink+RDMA) | Context parallelism |
| **GEMM + ReduceScatter** | Matmul → streaming scatter-reduce | ReduceScatter (NVLink+RDMA) | Tensor parallelism |

**The pattern:** Every kernel is `communication + compute` fused, where compute consumes tiles as they arrive (not after all communication finishes).

### 1.4 Implementation Architecture

- Built on `libibverbs` (GPU-initiated RDMA writes, no NCCL/NVSHMEM dependency)
- Backend abstraction: CX7 (InfiniBand) and EFA (AWS SRD) share same kernel, differ only in session/proxy
- Ring Attention uses staged kernels (not single megakernel) to keep register pressure manageable:
  1. KV send prologue (RDMA writes overlap early ring stages)
  2. Per-ring comm+partial kernels (NVLink KV exchange + partial attention)
  3. Per-ring reduction kernels (online softmax merge)
  4. KV copy epilogue (wait for peer RDMA arrivals)
- MMA compute code adapted from ThunderKittens (HazyResearch)

---

## 2. Mapping to Our Architecture

### 2.1 What Does NOT Apply

| mKernel Feature | Why Not | Our Stack |
|----------------|---------|-----------|
| Multi-node RDMA | Single device | Apple Silicon unified memory |
| NVLink intra-node | No multi-GPU | Single M-series GPU |
| Hopper sm_90a CUDA | Different GPU | Metal (wgpu/CubeCL) |
| libibverbs | No RDMA hardware | Standard memory ops |
| NCCL replacement | No NCCL dependency | No dependency to replace |
| Training-time AllReduce/ReduceScatter | Inference-only | No gradient synchronization |
| AllGather for weight sharding | Weights fit in RAM | No weight sharding needed |

**Bottom line:** mKernel is solving a problem we don't have. We are single-device inference, not multi-node training.

### 2.2 What IS Conceptually Valuable

#### Pattern 1: Persistent Kernel with Role Specialization

mKernel's CTA specialization (compute / intra-comm / inter-send / inter-reduce) maps to our `DecodeStage`:

| mKernel CTA Role | Our DecodeStage | Analogy |
|-----------------|-----------------|---------|
| Compute CTA | `DecodeStage::Verify` | Exact computation |
| Intra-comm CTA | `DecodeStage::Prefill` | Bulk data movement (KV loading) |
| Inter-send CTA | `DecodeStage::Draft` | Speculative forward (send tokens ahead) |
| Inter-reduce CTA | `DecodeStage::Sample` | Reduction (argmax/sampling) |

We already implemented this in Plan 102 (TileRT execution pipeline). mKernel validates that specializing workers by role is the right approach — even on NVIDIA's internal benchmarks.

#### Pattern 2: Tile-Granularity Overlap

mKernel overlaps communication and compute at **tile/chunk** granularity, not kernel boundary granularity. Our speculative pipeline (Plan 131 SpecHop) does the same: verification starts while draft KV is still in cache, not after all drafts complete.

This pattern is consistent across:
- Research 066 (TileRT): persistent execution, tile-level pipeline overlap
- Research 067 (CODA): epilogue fusion inside matmul output loop
- Research 077 (ThunderKittens): LCF producer-consumer pipeline
- Plan 131 (SpecHop): multi-hop speculative commit/rollback

mKernel adds another data point: **tile-granularity overlap beats kernel-boundary overlap in every scenario tested**. This reinforces our existing architecture decisions.

#### Pattern 3: Megakernel Taxonomy

mKernel's roadmap includes "inter-node megakernels: collapsing several fused steps into a single megakernel that spans an entire transformer layer." This is the same vision as TileRT (static graph expansion into persistent kernel).

The five kernels are a taxonomy for what a megakernel fuses:
1. **Input gathering + compute** (AllGather + GEMM) → our prefill
2. **Compute + output reduction** (GEMM + AllReduce) → our attention + KV write
3. **Routing + compute** (MoE Dispatch + GEMM) → our domain router + forward
4. **KV rotation + attention** (Ring Attention) → our speculative draft+verify
5. **Compute + output distribution** (GEMM + ReduceScatter) → our decode + sample

If we ever scale to multi-device inference (edge case: server deployment with multiple M-series chips), this taxonomy tells us what to fuse.

---

## 3. Impact Assessment

### 3.1 katgpt-rs (MIT, Inference Core)

| Aspect | Impact | Action |
|--------|--------|--------|
| CPU SIMD pipeline | None — no multi-node | No code changes |
| Speculative decode | Already overlaps via SpecHop | No code changes |
| DecodeStage specialization | Already implemented (Plan 102) | Validation only |
| Feature gate | Not needed | — |

### 3.2 riir-ai (Private, GPU Training + Games)

| Aspect | Impact | Action |
|--------|--------|--------|
| wgpu LoRA training | Future: tile-granularity overlap in GPU kernels | Track, no action now |
| CubeCL rewrite (Plan 106) | mKernel's SM specialization → CubeCL cube-level specialization | Conceptual guide |
| MoE Dispatch + GEMM pattern | If we add MoE routing to LoRA training | Track for Plan 143 (DFLash LoRA) |
| Ring Attention | Future: sequence-parallel attention for long-context LoRA training | Track |
| Game AI | No impact | — |
| **Super-GOAT potential** | Conditional — only if multi-GPU training scales | ⏳ Pending |

### 3.3 GOAT Pillar Alignment (Reference: 27_mmo_goat_pillars_decision_matrix.md)

| Pillar | mKernel Relevance | Why |
|--------|------------------|-----|
| 1. Fourier Spatial AI | ❌ None | Pure algorithmic, no GPU kernel dependency |
| 2. WASM Validators | ❌ None | Deterministic, no compute-communication fusion |
| 3. NPC Dialog Engine | ❌ None | Modelless baseline, no multi-node |
| 4. Frame-Sampling Bridge | ❌ None | Pure frame decimation, no GPU |
| Gap 1. Cold Tier | ❌ None | Storage layer, not compute |
| Gap 2. MMO Backbone | ⬜ Indirect | If MMO server scales to multi-node, megakernel patterns help |

**Conclusion:** mKernel does not affect any GOAT pillar directly. It's a training-time GPU optimization with no inference-time game AI implications.

---

## 4. Verdict

**LOW DIRECT VALUE.** mKernel solves multi-node GPU training communication bottlenecks. We do single-device inference. The architectural patterns (persistent specialization, tile-granularity overlap, megakernel taxonomy) are conceptual validation of decisions we already made, not new capabilities to implement.

### Classification Matrix

| Classification | Verdict |
|---------------|---------|
| GOAT-proofable | ❌ No — multi-node training is not in our inference benchmark suite |
| Super-GOAT (private) | ⏳ Conditional — only if riir-ai scales to multi-GPU LoRA training |
| Feature gate needed | ❌ No — no code changes in katgpt-rs |
| Game-related | ❌ No — pure infrastructure, no game knowledge |
| Stays in riir-ai | ⏳ Conditional — track for future GPU kernel fusion, keep in `.plans/` not `.docs/` |

### Action Items

1. ✅ Research documented (this file)
2. ⏳ Plan 150 created in katgpt-rs (conceptual tracking only, no implementation)
3. ⏳ Plan 147 tracked in riir-ai (conditional, pending GPU training scale decision)
4. ❌ No feature gate needed
5. ❌ No GOAT proof needed
6. ❌ No game pillar impact

---

## 5. Key Code References (mKernel Source)

From `.raw/mKernel/src/ring_attention.cu`:
- Staged kernel design (prologue → per-ring stages → epilogue)
- CTA role functions: `attn_partial`, `attn_comm`, `attn_reduction`, `kv_stage_and_send_sm`
- Online softmax fusion with `FUSE_REDUCE` template parameter
- Register pressure management via staged (not monolithic) kernel

From `.raw/mKernel/src/dispatch_gemm.cu`:
- MoE token routing + grouped GEMM fusion
- All-to-All dispatch with immediate matmul consumption

From `.raw/mKernel/include/`:
- Backend abstraction: `session.h` (CX7) vs `session_efa.h` (EFA)
- SM role assignment and grid barrier coordination

---

## References

1. mKernel blog post: https://uccl-project.github.io/posts/mkernel/
2. mKernel code: https://github.com/uccl-project/mKernel (MIT)
3. MegaScale-MoE (communication overhead stats): EuroSys 2026
4. Comet (MoE fine-grained overlap): MLSys 2025
5. ThunderKittens (MMA code origin): HazyResearch, Stanford — our Research 077

> **PASS-Redirects (synthesis):** Siavashi et al. [arXiv:2604.07609 "Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC"] — the serving-stack sibling of this note's verdict class: BlueField-3 DPU + ConnectX-6 + GPUDirect RDMA fabric we do not run; the persistent-scheduler control plane + GPU-initiated RDMA direction matches mKernel's host-off-the-critical-path thesis at serving granularity, and Blink's CPU-colocation immunity result (baselines lose 52–72% throughput under interference, Blink ~1.0×) corroborates our GPU-exclusivity/exclusive-lane bench law from the opposite direction. PASS — no files.
