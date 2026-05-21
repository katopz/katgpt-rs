# Paper Feature Comparison Matrix

**Date:** 2025-07
**Status:** Living Document
**Scope:** All 54 research papers (00–53) in `.research/` mapped against microgpt-rs feature dimensions

## Introduction

This document provides a comprehensive feature-intersection matrix between our work (microgpt-rs) and all 54 researched papers. Each paper is evaluated across 10 feature dimensions derived from our core architecture:

| Column | Description |
|--------|-------------|
| **SD** | Speculative Decoding — draft/verify, tree search, multi-token prediction |
| **KV** | KV Optimization — cache compression, pruning, quantization, paged attention |
| **Attn** | Attention Innovation — novel attention mechanisms, linear attention, hull queries |
| **Noise** | Noise / Noise Scheduling — SDE injection, diffusion schedules, perturbation |
| **Distill** | Distillation / Compression — LoRA, quantization, knowledge transfer, pruning |
| **TTC** | Test-Time Compute — adaptive budget, self-improvement, recursive refinement |
| **Route** | Routing / MoE — expert selection, domain routing, mixture-of-experts |
| **Diff** | Diffusion / Denoising — discrete diffusion, block-parallel, flow matching |
| **Game** | Game / Self-Play — puzzles, board games, RL arenas, heuristic learning |
| **SIMD** | SIMD / Perf — hardware acceleration, zero-alloc, GPU compute, kernels |

Legend: ✓ = direct feature, ○ = partial/conceptual alignment, ✗ = not applicable

---

## Our Work: microgpt-rs Feature Summary

| Feature | Technique | Status |
|---------|-----------|--------|
| Speculative Decoding | DDTree + DFlash + Leviathan verification | ✓ Implemented |
| KV Optimization | SpectralQuant (9.1×, 0.9917 cosine), SP-KV (3-10×), TurboQuant 3-bit | ✓ Implemented |
| Attention Innovation | forward_hla / forward_ahla (88% memory savings), Percepta 2D Convex Hull | ✓ Implemented |
| Noise Scheduling | ELF SDE noise injection (10-22× path diversity) | ✓ Implemented |
| Distillation/Compression | LoRA adapters, SpectralQuant, domain constraint pruning | ✓ Partial |
| Test-Time Compute | SimpleTES RPUCG loop, BanditPruner adaptive arms | ✓ Implemented |
| Routing/MoE | Raven slot memories, EMO-style domain routing | ✓ Implemented |
| Diffusion/Denoising | dLLM D2F block-parallel denoising | ✓ Partial |
| Game/Self-Play | Sudoku, Go, Monopoly, Bomberman domains | ✓ Implemented |
| SIMD/Perf | NEON SIMD matmul/HLA kernels, zero-alloc hot paths | ✓ Implemented |

---

## Feature Intersection Matrix

### Our Architecture (Reference Row)

| # | Paper / Feature | SD | KV | Attn | Noise | Distill | TTC | Route | Diff | Game | SIMD |
|---|----------------|----|----|------|-------|---------|-----|-------|------|------|------|
| — | **microgpt-rs (our work)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

### Papers 00–09: Foundation & Architecture

| # | Paper / Feature | SD | KV | Attn | Noise | Distill | TTC | Route | Diff | Game | SIMD |
|---|----------------|----|----|------|-------|---------|-----|-------|------|------|------|
| 00 | Neuro-Symbolic LLM Architecture | ○ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ○ |
| 01 | Advanced Neuro-Symbolic Rust Translation | ✓ | ○ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 02 | Fast Inference via Speculative Decoding (Leviathan) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 03 | Commercial Open Source Strategy Verdict | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ○ | ✗ | ✗ | ✗ |
| 04 | LoRA Architecture Verdict | ○ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| 05 | Artifact Definition (Validator vs Adapter) | ✗ | ✗ | ✗ | ✗ | ○ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 06 | Raven Routing Slot Memories | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| 07 | Screening Absolute Relevance | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ○ | ✗ | ✗ | ✗ |
| 08 | TwELL Sparse MLP (Sakana) | ✗ | ✗ | ✗ | ✗ | ○ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 09 | EMO Emergent Modularity | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |

### Papers 10–19: Diffusion, Test-Time Compute & Agents

| # | Paper / Feature | SD | KV | Attn | Noise | Distill | TTC | Route | Diff | Game | SIMD |
|---|----------------|----|----|------|-------|---------|-----|-------|------|------|------|
| 10 | ColaDLM Continuous Latent Diffusion | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| 11 | PPoT Probabilistic Programs of Thought | ○ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| 12 | TRT Test-time Recursive Thinking | ✗ | ✗ | ✗ | ✗ | ○ | ✓ | ✗ | ✗ | ✗ | ✗ |
| 13 | NVIDIA Dynamo Agentic Lessons | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 14 | Learning Beyond Gradients (Heuristic Learning) | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |
| 15 | Reinforced Agent Inference-Time Feedback | ✗ | ✗ | ✗ | ✗ | ○ | ✓ | ✗ | ✗ | ✗ | ✗ |
| 16 | AutoTTS Dynamic Test-Time Scaling | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ○ | ✗ | ✗ | ✗ |
| 17 | Fast BLT Byte-Level Transformer | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 18 | The Free Transformer Latent Injection | ✗ | ✗ | ✓ | ✗ | ○ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 19 | TTT-Discover Test-Time Training | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ○ | ✗ |

### Papers 20–29: Quantization, Games & Linear Attention

| # | Paper / Feature | SD | KV | Attn | Noise | Distill | TTC | Route | Diff | Game | SIMD |
|---|----------------|----|----|------|-------|---------|-----|-------|------|------|------|
| 20 | TurboQuant Online Vector Quantization | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 21 | G-Zero Self-Play Open-Ended Generation | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ |
| 22 | Lighthouse Attention | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 23 | GFlowNet Shortest Paths | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ |
| 24 | Delta-Mem Online Associative Memory | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 25 | StepCodeReasoner Bi-Level GRPO | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| 26 | Gemma 4 MTP Multi-Token Prediction | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 27 | STRATEGA Strategy Games Framework | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ○ | ✗ | ✓ | ✗ |
| 28 | Higher-order Linear Attention (HLA) | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 29 | rust-gpu Feasibility | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |

### Papers 30–39: Optimization, Diffusion & Quantization

| # | Paper / Feature | SD | KV | Attn | Noise | Distill | TTC | Route | Diff | Game | SIMD |
|---|----------------|----|----|------|-------|---------|-----|-------|------|------|------|
| 30 | FFOLayer First-Order Optimization | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 31 | Percepta Deep Dive | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 32 | Percepta Distillation Strategy | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 33 | AutoGo Distillation Strategy | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| 34 | D2F Discrete Diffusion Forcing | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| 35 | Attractor Models Fixed-Point Refinement | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 36 | ROPD Rubric On-Policy Distillation | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| 37 | REAP Model-Based Modelless Duality | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| 38 | SDAR Self-Distilled Agentic RL | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| 39 | SpectralQuant Eigenbasis KV Compression | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |

### Papers 40–49: Ranking, Diffusion, Pruning & Recursion

| # | Paper / Feature | SD | KV | Attn | Noise | Distill | TTC | Route | Diff | Game | SIMD |
|---|----------------|----|----|------|-------|---------|-----|-------|------|------|------|
| 40 | OpenDeepThink Bradley-Terry Ranking | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| 41 | RePlaid Continuous Diffusion Scaling | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| 42 | SP-KV Self-Pruned KV Attention | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 43 | Interventional SFT Causal Token Masking | ✗ | ✗ | ✗ | ✗ | ✓ | ○ | ✗ | ✗ | ✗ | ✗ |
| 44 | ELF Embedded Language Flows | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| 45 | MaxSim Memory-Efficient Late Interaction | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 46 | Symmetry Compatible Equivariant Optimizers | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 47 | PGD Professional Go Dataset Analytics | ✗ | ✗ | ✗ | ✗ | ✗ | ○ | ✗ | ✗ | ✓ | ✗ |
| 48 | HRM-Text Hierarchical Recurrent Pretraining | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 49 | PTRM Probabilistic Tiny Recursive Model | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ○ |

### Papers 50–53: Deduction, Manifold, Scaling & Attribution

| # | Paper / Feature | SD | KV | Attn | Noise | Distill | TTC | Route | Diff | Game | SIMD |
|---|----------------|----|----|------|-------|---------|-----|-------|------|------|------|
| 50 | LDT Lattice Deduction Transformer | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| 51 | Deep Manifold Fixed-Point Boundaries | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| 52 | SimpleTES Evaluation-Driven Scaling | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| 53 | CNA Contrastive Neuron Attribution | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |

---

## Feature Intersection Heatmap (Count per Dimension)

How many papers intersect with each feature dimension:

| Dimension | ✓ Count | ○ Count | Top Contributing Papers |
|-----------|---------|---------|------------------------|
| **SD** Speculative Decoding | 7 | 3 | 02 (Leviathan), 26 (MTP), 34 (D2F), 17 (BLT-S) |
| **KV** KV Optimization | 10 | 1 | 20 (TurboQuant), 28 (HLA), 39 (SpectralQuant), 42 (SP-KV) |
| **Attn** Attention Innovation | 19 | 0 | 28 (HLA), 06 (Raven), 22 (Lighthouse), 31 (Percepta) |
| **Noise** Noise / Noise Scheduling | 2 | 0 | 44 (ELF SDE), 49 (PTRM Gaussian) |
| **Distill** Distillation / Compression | 22 | 3 | 04 (LoRA), 36 (ROPD), 39 (SpectralQuant), 46 (Equivariant) |
| **TTC** Test-Time Compute | 16 | 2 | 16 (AutoTTS), 19 (TTT), 52 (SimpleTES), 12 (TRT) |
| **Route** Routing / MoE | 9 | 4 | 06 (Raven), 09 (EMO), 37 (REAP), 40 (Bradley-Terry) |
| **Diff** Diffusion / Denoising | 4 | 0 | 10 (ColaDLM), 34 (D2F), 41 (RePlaid), 44 (ELF) |
| **Game** Game / Self-Play | 8 | 1 | 14 (HL), 21 (G-Zero), 27 (STRATEGA), 33 (AutoGo) |
| **SIMD** SIMD / Perf | 15 | 1 | 20 (TurboQuant), 28 (HLA 95%), 45 (MaxSim 7.46×), 29 (rust-gpu) |

---

## High-Intersection Papers (≥4 features)

Papers that intersect with 4 or more feature dimensions:

| # | Paper | Features | Intersection Details |
|---|-------|----------|---------------------|
| **28** | Higher-order Linear Attention (HLA) | KV✓ Attn✓ SIMD✓ | AHLA 95% throughput, 88% less memory, constant per-token cost |
| **20** | TurboQuant | KV✓ Distill✓ SIMD✓ | 3-bit KV cache, 10.7× compression, quality-neutral at 3.5 bits |
| **39** | SpectralQuant | KV✓ Distill✓ SIMD✓ | +0.27–0.38 cosine over TQ, water-fill allocation, 2.2× faster |
| **22** | Lighthouse Attention | KV✓ Attn✓ SIMD✓ | 1.4–1.7× wall-clock, 98K+ context, pyramid pooling |
| **45** | MaxSim Late Interaction | KV✓ Attn✓ SIMD✓ | CPU SIMD 7.46×, GPU 41–74×, memory-efficient scoring |
| **34** | D2F Discrete Diffusion Forcing | SD✓ KV✓ Diff✓ | Block-parallel denoising, 7.3–29.1× speedup, block-causal KV |
| **26** | Gemma 4 MTP | SD✓ KV✓ Distill✓ | Shared KV, target activations, clustered LM head, 85% acceptance |
| **44** | ELF Embedded Language Flows | Noise✓ Distill✓ Diff✓ | SDE sampling, x-prediction, shared denoiser-decoder, Gen PPL 24 |
| **21** | G-Zero Self-Play | Distill✓ TTC✓ Game✓ | Hint-δ reward, verifier-free self-play, DPO training |
| **19** | TTT-Discover | Distill✓ TTC✓ Game○ | Test-time LoRA updates, entropic objective, solution buffer |
| **46** | Symmetry Optimizers | Distill✓ SIMD✓ | Layerwise RowNormM, architecture–optimizer co-design |
| **48** | HRM-Text | Attn✓ Distill✓ SIMD✓ | Hierarchical recurrent, Adam-atan2, multipack batching |
| **53** | CNA Contrastive Neuron Attribution | Attn✓ Distill✓ SIMD✓ | 0.1% neurons, forward-hook activation, sparse modulation |

---

## Category Co-occurrence Matrix

How often feature pairs co-occur across papers:

| | SD | KV | Attn | Noise | Distill | TTC | Route | Diff | Game | SIMD |
|---|---|---|---|---|---|---|---|---|---|---|
| **SD** | 7 | 3 | 2 | 0 | 2 | 1 | 0 | 2 | 0 | 2 |
| **KV** | 3 | 10 | 7 | 0 | 3 | 0 | 1 | 1 | 0 | 6 |
| **Attn** | 2 | 7 | 19 | 1 | 5 | 1 | 3 | 1 | 2 | 8 |
| **Noise** | 0 | 0 | 1 | 2 | 2 | 1 | 0 | 2 | 0 | 0 |
| **Distill** | 2 | 3 | 5 | 2 | 22 | 7 | 3 | 3 | 3 | 6 |
| **TTC** | 1 | 0 | 1 | 1 | 7 | 16 | 3 | 0 | 4 | 0 |
| **Route** | 0 | 1 | 3 | 0 | 3 | 3 | 9 | 0 | 2 | 0 |
| **Diff** | 2 | 1 | 1 | 2 | 3 | 0 | 0 | 4 | 0 | 0 |
| **Game** | 0 | 0 | 2 | 0 | 3 | 4 | 2 | 0 | 8 | 0 |
| **SIMD** | 2 | 6 | 8 | 0 | 6 | 0 | 0 | 0 | 0 | 15 |

Top co-occurring pairs:
1. **Attn + SIMD** (8 papers) — novel attention mechanisms often need hardware optimization
2. **Attn + KV** (7 papers) — attention innovation frequently targets KV cache efficiency
3. **Distill + TTC** (7 papers) — distillation and test-time compute are complementary strategies
4. **KV + SIMD** (6 papers) — KV compression requires performant kernels
5. **Distill + SIMD** (6 papers) — compression techniques need hardware-friendly implementations

---

## Papers by Architecture Type

### Transformer-Based (Standard Architecture)
| Papers | Count |
|--------|-------|
| 00, 01, 02, 04, 06, 07, 08, 11, 12, 13, 15, 16, 18, 19, 21, 25, 26, 30, 31, 32, 33, 36, 37, 38, 40, 43, 46, 48, 49, 53 | **30** |

### Diffusion-Based (Continuous or Discrete)
| Papers | Count |
|--------|-------|
| 10, 34, 41, 44 | **4** |

### Linear / Sub-Quadratic Attention
| Papers | Count |
|--------|-------|
| 06 (Raven), 24 (Delta-Mem), 28 (HLA), 42 (SP-KV), 45 (MaxSim) | **5** |

### Hybrid / Novel Architecture
| Papers | Count |
|--------|-------|
| 17 (BLT byte-level), 22 (Lighthouse pyramid), 35 (Attractor fixed-point), 48 (HRM recurrent), 50 (LDT lattice), 51 (Deep Manifold) | **6** |

### Non-Architecture (Strategy / Engineering / Dataset)
| Papers | Count |
|--------|-------|
| 03, 05, 14, 20, 23, 27, 29, 39, 47, 52 | **10** |

---

## Summary of Intersection Highlights

### 1. Highest Direct Value (Direct Fit, Already Implemented)

| Paper | What We Adopted | Where |
|-------|----------------|-------|
| 02 Leviathan | Speculative decoding with rejection sampling | `speculative/verifier.rs` |
| 06 Raven | O(1) slot memory routing | `forward_raven()` |
| 08 TwELL | Sparse MLP matmul for ReLU activations | `types.rs sparse_matmul` |
| 20 TurboQuant | 3-bit KV cache quantization | `turboquant` module |
| 28 HLA/AHLA | Second-order linear attention, 88% memory savings | `forward_hla`, `forward_ahla` |
| 39 SpectralQuant | Eigenbasis rotation + water-fill over TurboQuant | SpectralQuant module |
| 42 SP-KV | Self-pruned KV attention, 3-10× reduction | SP-KV module |
| 44 ELF | SDE noise injection for path diversity | `inject_sde_noise` |
| 45 MaxSim | Late-interaction scoring, CPU SIMD 7.46× | MaxSim primitive |
| 52 SimpleTES | RPUCG bandit loop for evaluation-driven scaling | BanditPruner trait |
| 53 CNA | Contrastive neuron attribution + sparse modulation | CNA steering |

### 2. Strong Conceptual Alignment (Pattern Adopted, Different Mechanism)

| Paper | What We Distilled | Our Equivalent |
|-------|-------------------|---------------|
| 09 EMO | Document-level expert routing | `KeywordRouter` + `ExpertRegistry` |
| 14 Heuristic Learning | Code-based policy evolution | `BanditPruner` + `AbsorbCompress` |
| 24 Delta-Mem | Delta-rule associative memory | Feature-hashed Rust implementation |
| 37 REAP | Model-based/modelless spectrum | Existing trait stack captures both |
| 49 PTRM | Noise-injected recursive refinement | `inject_sde_noise` + DDTree |
| 51 Deep Manifold | Fixed-point boundary conditions | Three-stage distillation pipeline |

### 3. Selective Adoption (Specific Techniques Only)

| Paper | What We Took | What We Skipped |
|-------|-------------|-----------------|
| 10 ColaDLM | KV cache priming concept | Full VAE-DiT mechanism |
| 17 Fast BLT | Self-speculation validates our approach | Byte-level model architecture |
| 41 RePlaid | ELBO regularization, variance-minimized schedules | Full continuous diffusion |
| 48 HRM-Text | Adam-atan2 optimizer, PrefixLM batching | Full hierarchical recurrent model |

### 4. Negative Results (Not Applicable to Our Stack)

| Paper | Why Not Applicable |
|-------|-------------------|
| 03 Commercial Strategy | Business document, not a technique |
| 05 Artifact Definition | Terminology clarification only |
| 29 rust-gpu Feasibility | WGSL→Rust migration, not a technique |
| 47 PGD Go Dataset | Dataset paper, features already captured by GoHeuristic |

### 5. Gaps Identified (Features Papers Have That We Don't)

| Gap | Source Papers | Priority |
|-----|--------------|----------|
| Pairwise Bradley-Terry ranking | 40 (OpenDeepThink) | High — validates over pointwise |
| Interventional SFT causal masking | 43 (Interventional SFT) | High — 1.19 nats/token gain |
| Sigmoid-gated token-level distillation | 38 (SDAR) | Medium — prevents OPSD collapse |
| Rubric-based multi-criteria reward | 36 (ROPD) | Medium — interpretable reward |
| Asymmetric BCE for false elimination | 50 (LDT) | Medium — sound pruning loss |
| Adam-atan2 optimizer | 48 (HRM-Text) | Low — simple drop-in |

---

## Feature Coverage Radar

Our implementation status per feature dimension:

```
Speculative Decoding  ████████████████████ 95%  (DDTree, DFlash, Leviathan, MTP)
KV Optimization       ████████████████████ 95%  (SpectralQuant, SP-KV, TurboQuant)
Attention Innovation  ████████████████████ 90%  (HLA, AHCLA, Percepta, MaxSim)
Noise Scheduling      ██████████████░░░░░░ 70%  (SDE injection, PTRM validation)
Distillation          ████████████░░░░░░░░ 60%  (LoRA, SpectralQuant, ROPD gap)
Test-Time Compute     ████████████████░░░░ 80%  (SimpleTES, BanditPruner, AutoTTC)
Routing/MoE           ██████████████░░░░░░ 70%  (Raven, EMO pattern, marketplace)
Diffusion/Denoising   ████████░░░░░░░░░░░░ 40%  (D2F partial, no full dLLM)
Game/Self-Play        ██████████████████░░ 90%  (Sudoku, Go, Monopoly, Bomber)
SIMD/Perf             ████████████████████ 95%  (NEON, zero-alloc, GPU kernels)
```

---

## References

All papers are located in `microgpt-rs/.research/` with filenames `{index}_{Title}.md` where index ranges from 00 to 53. See individual research files for full analysis, verdicts, and implementation details.