# Research 439: NVFP4 RL — Dequantized Backward + 4/6 Adaptive Block Scaling

> **Source:** "The 4-bitter Lesson: Balancing Stability and Performance in NVFP4 RL" — Ziang Li & humans& ai, July 10, 2026. https://humansand.ai/blog/nvfp4-rl?v=3
> **Date:** 2026-07-11
> **Status:** Done — Pass (inference) + → riir-train (training)
> **Related Research:** 202 (QAT Infusion — same principles explored), 020 (TurboQuant — online quantization), 265 (b-posit — alternative precision format), 200 (quant outlier collapse), 085 (riir-ai multi-precision NPC switching)
> **Classification:** Public

---

## TL;DR

The blog develops the first stable hardware-native NVFP4 (4-bit float) RL training recipe. Its three techniques — dequantized backward, 4/6 adaptive block scaling, selective layer precision — are **primarily training-focused** (dequantized backward is gradient computation → riir-train). The inference-relevant techniques (4/6, selective precision, per-token activation scaling, online serving, bit-exact contract) are either already explored in Research 202, already shipped (TurboQuant online quantization, `quant_expert_goat.rs` per-expert precision), or format-specific to FP4 which our CPU/SIMD/ANE stack doesn't use. **Verdict: Pass** — no plan, no files beyond this note.

**Distilled for katgpt-rs (modelless, inference-time):**
The 4/6 adaptive block scaling (per-block choose max=4 or max=6 to minimize MSE) is novel to our codebase (zero prior art for "block scaling" / "adaptive scaling" / "four over six"), but it's format-specific to NVFP4 (E2M1 + E4M3 block scales). We don't ship FP4 — our quantization stack uses TurboQuant (rotation + Lloyd-Max), KVarN (variance-normalized), OCTOPUS (octahedral), and ternary (1.58-bit). The transferable principle — "adapt the quantization range per local block, not just the scale" — could apply to our existing codecs but has no concrete consumer today.

---

## 1. Paper Core Findings

### 1.1 NVFP4 format
4-bit floating point with hierarchical block scaling: each block of 16 values has an FP8 E4M3 scale, the full tensor has a global FP32 scale. Up to 9× more ops/sec than BF16 on NVIDIA Rubin GPUs.

### 1.2 Three techniques

| Technique | What | Training or inference? |
|---|---|---|
| **Dequantized backward** | Use `DQ(Q(w))` in backward pass instead of raw BF16, so gradients match the quantized forward | **Training** — the backward pass IS gradient computation |
| **4/6 adaptive block scaling** | Per block of 16 FP4 values, choose max=4 or max=6, whichever minimizes per-block MSE | Both — forward pass quantization (inference-relevant), but stabilizes training |
| **Selective layer precision** | Keep last 15% of layers + shared expert in BF16 | Both — mixed-precision inference strategy |

### 1.3 Bit-exact trainer-sampler contract
Quantization must produce bitwise-identical output every time, across trainer and sampler. Deterministic reductions, same numerical contract, same weight layout/swizzling. Implemented across FlashInfer + TransformerEngine.

### 1.4 Online NVFP4 serving
The rollout quantization path can serve checkpoints with online post-training quantization — no calibration, no QAT, no model-conversion pipeline. `--quantization nvfp4_online`. FP8↔NVFP4 correlation = 0.924.

---

## 2. Distillation

### 2.1 What goes to riir-train (training-only)

**Dequantized backward** is the core training contribution. The backward pass computes gradients; using `DQ(Q(w))` instead of raw BF16 makes gradients consistent with the quantized forward pass. This is gradient computation — genuine riir-train dependency.

**§3.5 modelless unblock check** (all three paths fail):
1. **Freeze/thaw** — No. The issue is gradient computation precision during training, not weight state.
2. **Raw/lora hot-swap** — No. The issue is backward pass numerical precision, not adapter correction.
3. **Latent-space correction** — No. The issue is gradient variance from quantization, not latent state.

→ riir-train: dequantized backward, gradient stability, RL training recipe, optimizer interaction (Adam momentum reducing NVFP4 gradient variance).

### 2.2 What stays modelless (inference-relevant)

| Technique | Novel to us? | Prior art in our codebase |
|---|---|---|
| **4/6 adaptive block scaling** | YES (zero hits for "block_scal", "adaptive_scal", "four over six") | TurboQuant does per-coordinate Lloyd-Max (different mechanism — vector quant, not block-scaled FP4). SpectralQuant does per-eigenvalue bit allocation. KVarN does per-channel variance normalization. The PRINCIPLE "adapt quantization per local region" ships under different vocabulary; the specific MECHANISM (choose max=4 vs max=6 per block) is new. |
| **Selective layer precision** | NO (concept already explored) | Research 202 Fusion 4 (TPB — targeted precision budget per attention head). `quant_expert_goat.rs` ships per-expert-type precision routing (Combat→INT4, Dialogue→INT8, etc.). The specific application (last 15% layers + shared expert) is a refinement. |
| **Per-token activation scaling** | Partially (concept present, FP4-specific instance new) | Research 202 Fusion 2 (SCT — static calibration tables, the opposite direction). KVarN does online Sinkhorn (per-channel, not per-token). TurboQuant stores per-token L2 norms (for vector quant, not FP4 activation scaling). |
| **Online post-training quantization** | NO (already ships) | TurboQuant is explicitly "data-oblivious, online, no calibration" — the same principle. NVFP4 online serving is a format-specific instance. |
| **Bit-exact quantization contract** | Partially (pattern familiar, application to quant new) | Extensive bit-exact testing (Sheaf-ADMM G6, Set Attention G5, KARC G4, GoldShare). Quorum-verifiable patterns (RTDC quorum, chain commitment). But not applied to quantization specifically. |

### 2.3 Latent-space reframing (mandatory per workflow §1 step 3)

The 4/6 choice is a **per-subspace precision allocation** decision:
- Each block of 16 values = a local region of the weight/activation manifold
- 4-vs-6 = "which representation preserves local geometry better?" (MSE-minimization)
- Analogous to: HLA channel selection (which affect channels to prioritize), zone gating (which zones get more compute), SpectralQuant eigenbasis allocation (which eigen-directions get more bits)

The 4/6 is a **second-order adaptation** — adapting not just the scale (first-order, like KVarN's per-channel normalization) but the representable range (second-order, choosing between two quantization grids). This is the block-level analog of TurboQuant's per-coordinate centroid selection.

**But:** our codebase doesn't use FP4. We're CPU/SIMD/ANE (not CUDA/Blackwell). `riir-ai/.plans/182` explicitly defers "NVFP4 (Blackwell-only)". The 4/6 principle could transfer to our codecs (e.g., choose between two TurboQuant codebooks per block), but that's speculative without a benchmark and without a consumer needing FP4-level compression.

---

## 3. Fusion

### Closest existing notes / plans / code

| Cousin | Repo / location | Relation |
|---|---|---|
| **Research 202 — QAT Infusion** | `katgpt-rs/.research/202_*` | Same principles: Fusion 2 (SCT = static activation scales), Fusion 4 (TPB = targeted precision budget). The blog's selective precision + per-token scaling are instances of these fusions. |
| **Research 020 — TurboQuant** | `katgpt-rs/.research/020_*` | Online data-oblivious quantization (same as blog's "online NVFP4 serving"). Different mechanism (rotation + Lloyd-Max vs FP4 block scaling). |
| **Research 265 — b-posit** | `katgpt-rs/.research/265_*` | Alternative precision format with bounded range (like 4/6 bounds the FP4 range). Same "format-level" verdict class. |
| **Research 200 — Quant outlier collapse** | `katgpt-rs/.research/200_*` | Security angle: FP4 is listed among vulnerable formats. 4/6's range reduction may affect outlier dynamics. |
| **`quant_expert_goat.rs`** | `riir-ai/crates/riir-games/tests/` | Per-expert-type precision routing (shipped). The blog's selective layer precision is the per-layer-depth analog. |
| **Research 085 (riir-ai) — Multi-precision NPC switching** | `riir-ai/.research/085_*` | Intentional dual-behavior via quantization level. Related to selective precision concept. |

### Fusion idea — novelty TBD

**4/6-style adaptive codebook selection for TurboQuant/OCTOPUS.** Instead of a single Lloyd-Max codebook per (d, b) pair, maintain two codebooks (one tighter range, one wider) and select per-block based on which minimizes local MSE. This is the 4/6 principle applied to our rotation-based codecs. **Hypothesis:** blocks with concentrated values (post-rotation, low-variance directions) benefit from the tighter codebook; blocks with spread values benefit from the wider one. Untestable today without a benchmark; tracked only in this note.

This is a **fusion idea, not a Super-GOAT claim** — it needs Q1–Q4 novelty-gate work before any verdict.

---

## 4. Verdict

**Pass (inference side) + → riir-train (training side).**

**Reasoning:**

- **Training parts (dequantized backward, RL recipe, gradient stability):** Genuine riir-train dependencies after §3.5 check (all three modelless paths fail for gradient computation). Note "→ riir-train" and stop for these.
- **4/6 adaptive block scaling:** Novel to our codebase (Q1 PASS — zero prior art), but:
  - Q2 (new class of behavior): FAIL — it's better quantization MSE at the same bit budget, not a new capability. We can already quantize to 2-4 bits via TurboQuant.
  - Q3 (product selling point): FAIL — "adaptive block-scaled FP4" is not a customer-facing selling point. We don't use FP4.
  - Q4 (force multiplier): FAIL — doesn't connect to ≥2 pillars. It's a format-level refinement.
  - Additionally: format-specific to NVFP4 which our CPU/SIMD/ANE stack doesn't use. No concrete consumer.
- **Selective layer precision:** Already explored (Research 202 Fusion 4, `quant_expert_goat.rs`). Not novel.
- **Per-token activation scaling:** Concept present (Research 202 Fusion 2), FP4-specific instance new but no consumer.
- **Online post-training quantization:** Already ships (TurboQuant is data-oblivious/online).
- **Bit-exact quantization contract:** Pattern familiar (extensive bit-exact testing for other primitives). Applying to quantization is new but not a new capability.

**No plan created.** The 4/6 fusion idea (§3) is speculative and untestable without FP4 infrastructure we don't have. If we ever adopt FP4 (e.g., a future GPU target ships NVFP4 tensor cores and we add CUDA/CubeCL FP4 support), re-evaluate.

---

## TL;DR

The blog is primarily about RL training with NVFP4 — the core contribution (dequantized backward) is gradient computation → riir-train (§3.5 check: all three modelless paths fail). The inference-relevant techniques (4/6 adaptive block scaling, selective precision, per-token scaling, online serving, bit-exact contract) are either already explored (Research 202 QAT Infusion fusions 2/4), already shipped (TurboQuant online quantization, `quant_expert_goat.rs` per-expert precision), or format-specific to FP4 which our CPU/SIMD/ANE stack doesn't use. The 4/6 adaptive block scaling is the most novel inference technique (zero prior art in our codebase) but has no consumer — we don't ship FP4. **Verdict: Pass** — no plan, no files beyond this note. Re-evaluate if we adopt FP4.

> **PASS-Redirects (synthesis):** Hu [arXiv:2609.04105 "Hardware-Aware FP4 FlashAttention-4"] — the FP4-hardware-class verdict repeats (Blackwell-only tensor cores; fleet is M3 Max + Ada 4090, no FP4) and extends this note's dequantized-backward principle with measured economics: forward-quantization-into-backward (save payload+scales+LSE, reconstruct P) hits 1.14× on a complete 8B update, and the four-arm factorial isolates P/V format as the training-divergence separator (MXFP4-P/V diverges under both projection formats; FP8-P/V stable 55.5B tokens). Full distill: katgpt-rs `.research/534` — the recipe items arm this note's riir-train deferral (dequantized backward, format table by zero-flush rate, stage-wise error attribution) rather than unblocking it.
