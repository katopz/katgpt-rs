# Paper Feature Comparison Matrix

**Date:** 2025-07
**Status:** Living Document
**Scope:** All 62 research papers (00–61) in `.research/` mapped against microgpt-rs feature dimensions. Includes Research 061 (Delta Attention Residuals) mapped to `delta_routing`.

## Introduction

This document provides a comprehensive feature-intersection matrix between our work (microgpt-rs) and all 62+ researched papers. Each paper is evaluated across 10 feature dimensions derived from our core architecture:

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
| Speculative Decoding | DDTree + DFlash + Leviathan verification + Tri-Mode self-speculation | ✓ Implemented |
| KV Optimization | OCTOPUS (12.2×, 0.9512 cosine at 2-bit, **primary default**, zero calibration), SpectralQuant (9.1×, 0.9917 cosine, secondary), SP-KV (3-10×), TurboQuant 3-bit (legacy) | ✓ Implemented |
| Attention Innovation | forward_hla / forward_ahla (88% memory savings), Percepta 2D Convex Hull, MaxSim | ✓ Implemented |
| Noise Scheduling | ELF SDE noise injection (10-22× path diversity, **default**), GRAM validates approach | ✓ Implemented |
| Distillation/Compression | LoRA adapters, SpectralQuant, BT pairwise ranking (**default**), MeMo reflections, ROPD rubric | ✓ Partial (ASFT/SLIME in riir-gpu, CISPO default GRPO variant) |
| Test-Time Compute | SimpleTES RPUCG loop (GOAT 8/8, **default**), BanditPruner adaptive arms, GRAM width scaling | ✓ Implemented |
| Routing/MoE | Raven slot memories, MoE+SD Amdahl cost model, TIES merging (MeMo), Delta Block cross-layer (**default**) | ✓ Implemented |
| Diffusion/Denoising | dLLM D2F block-parallel denoising, Tri-Mode AR+Diffusion+Self-Speculation (GOAT 4/4) | ✓ Partial (untrained acceptance rate 1.0) |
| Game/Self-Play | Sudoku, Go, Monopoly, Bomber, Unit Distance lattice constructions | ✓ Implemented |
| SIMD/Perf | NEON SIMD matmul/HLA kernels, zero-alloc hot paths, Minkowski lattice embedding, LDT α-intersection (**default**) | ✓ Implemented |

**Default feature set:** `sparse_mlp`, `domain_latent`, `ppot`, `bandit`, `bt_rank`, `spectral_quant`, `octopus`, `elf_sde`, `cna_steering`, `deep_manifold`, `federation`, `tes_loop`, `lattice_deduction`, `delta_routing`

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

### Papers 54–61: Anchored SFT, Tri-Mode, Unit Distance, Agents, Reasoning, MoE, Memory & Alignment

| # | Paper / Feature | SD | KV | Attn | Noise | Distill | TTC | Route | Diff | Game | SIMD |
|---|----------------|----|----|------|-------|---------|-----|-------|------|------|------|
| 54 | ASFT Anchored Supervised Fine-Tuning | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 55 | Nemotron Tri-Mode Diffusion | ✓ | ✗ | ✓ | ✗ | ✗ | ○ | ✗ | ✓ | ✗ | ✗ |
| 56 | OpenAI Unit Distance Disproof | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| 57 | ART Agent Reinforcement Trainer | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| 58 | GRAM Generative Recursive Reasoning | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ○ | ✗ |
| 59 | MoE Speculative Decoding Co-Design | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ○ |
| 60 | MeMo Memory as a Model | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| 61 | SLIME Stabilized Likelihood Implicit Margin | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 061 | Delta Attention Residuals (Cross-Layer Routing) | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |

---

## Feature Intersection Heatmap (Count per Dimension)

How many papers intersect with each feature dimension:

| Dimension | ✓ Count | ○ Count | Top Contributing Papers |
|-----------|---------|---------|------------------------|
| **SD** Speculative Decoding | 9 | 3 | 02 (Leviathan), 26 (MTP), 34 (D2F), 55 (Tri-Mode), 59 (MoE+SD) |
| **KV** KV Optimization | 11 | 1 | 20 (TurboQuant), 28 (HLA), 39 (SpectralQuant), 42 (SP-KV), 60 (MeMo) |
| **Attn** Attention Innovation | 20 | 0 | 28 (HLA), 06 (Raven), 22 (Lighthouse), 31 (Percepta), 55 (Tri-Mode) |
| **Noise** Noise / Noise Scheduling | 3 | 0 | 44 (ELF SDE), 49 (PTRM), 58 (GRAM learned-mean SDE) |
| **Distill** Distillation / Compression | 26 | 3 | 04 (LoRA), 36 (ROPD), 39 (SpectralQuant), 54 (ASFT), 61 (SLIME) |
| **TTC** Test-Time Compute | 18 | 3 | 16 (AutoTTS), 19 (TTT), 52 (SimpleTES), 57 (ART), 58 (GRAM) |
| **Route** Routing / MoE | 11 | 4 | 06 (Raven), 09 (EMO), 37 (REAP), 59 (MoE+SD), 60 (MeMo TIES) |
| **Diff** Diffusion / Denoising | 5 | 0 | 10 (ColaDLM), 34 (D2F), 41 (RePlaid), 44 (ELF), 55 (Tri-Mode) |
| **Game** Game / Self-Play | 9 | 2 | 14 (HL), 21 (G-Zero), 27 (STRATEGA), 33 (AutoGo), 56 (Unit Distance) |
| **SIMD** SIMD / Perf | 16 | 2 | 20 (TurboQuant), 28 (HLA 95%), 45 (MaxSim 7.46×), 29 (rust-gpu), 56 (Minkowski) |

---

## High-Intersection Papers (≥4 features)

Papers that intersect with 4 or more feature dimensions:

| # | Paper | Features | Intersection Details |
|---|-------|----------|---------------------|
| **28** | Higher-order Linear Attention (HLA) | KV✓ Attn✓ SIMD✓ | AHLA 95% throughput, 88% less memory, constant per-token cost |
| **20** | TurboQuant | KV✓ Distill✓ SIMD✓ | 3-bit KV cache, 5.3× compression, quality-neutral at 3.5 bits (legacy baseline) |
| **39** | SpectralQuant | KV✓ Distill✓ SIMD✓ | 9.1× compression (vs TQ 5.3×), cosine 0.9917 (vs TQ 0.9692), water-fill allocation |
| **22** | Lighthouse Attention | KV✓ Attn✓ SIMD✓ | 1.4–1.7× wall-clock, 98K+ context, pyramid pooling |
| **45** | MaxSim Late Interaction | KV✓ Attn✓ SIMD✓ | CPU SIMD 7.46×, GPU 41–74×, memory-efficient scoring |
| **34** | D2F Discrete Diffusion Forcing | SD✓ KV✓ Diff✓ | Block-parallel denoising, 7.3–29.1× speedup, block-causal KV |
| **26** | Gemma 4 MTP | SD✓ KV✓ Distill✓ | Shared KV, target activations, clustered LM head, 85% acceptance |
| **44** | ELF Embedded Language Flows | Noise✓ Distill✓ Diff✓ | SDE sampling, x-prediction, shared denoiser-decoder, Gen PPL 24 |
| **21** | G-Zero Self-Play | Distill✓ TTC✓ Game✓ | Hint-δ reward, verifier-free self-play, DPO training |
| **19** | TTT-Discover | Distill✓ TTC✓ Game○ | Test-time LoRA updates, entropic objective, solution buffer |
| **46** | Symmetry Optimizers | Distill✓ SIMD✓ | Layerwise RowNormM, architecture–optimizer co-design |
| **48** | HRM-Text | Attn✓ Distill✓ SIMD✓ | Hierarchical recurrent, Adam-atan2, multipack batching |
| **53** | CNA Contrastive Neuron Attribution | Attn✓ Distill✓ SIMD✓ | ~10µs/pair discovery, 163ns K=50 modulation, quality cosine 1.0 |
| **55** | Nemotron Tri-Mode | SD✓ Attn✓ Diff✓ TTC○ | Dual-stream AR+Diffusion, 2.4-3.3× acceptance vs Eagle3, 76.5% SOL headroom |
| **60** | MeMo Memory as a Model | KV✓ Distill✓ Route✓ | O(1) retrieval, TIES merging at ρ=0.3, reflection QA pipeline |

---

## Category Co-occurrence Matrix

How often feature pairs co-occur across papers:

| | SD | KV | Attn | Noise | Distill | TTC | Route | Diff | Game | SIMD |
|---|---|---|---|---|---|---|---|---|---|---|
| **SD** | 9 | 3 | 3 | 0 | 2 | 2 | 1 | 3 | 0 | 3 |
| **KV** | 3 | 11 | 7 | 0 | 4 | 0 | 2 | 1 | 0 | 6 |
| **Attn** | 3 | 7 | 20 | 1 | 5 | 2 | 3 | 2 | 2 | 8 |
| **Noise** | 0 | 0 | 1 | 3 | 2 | 2 | 0 | 2 | 1 | 0 |
| **Distill** | 2 | 4 | 5 | 2 | 26 | 8 | 4 | 3 | 3 | 6 |
| **TTC** | 2 | 0 | 2 | 2 | 8 | 18 | 3 | 1 | 5 | 0 |
| **Route** | 1 | 2 | 3 | 0 | 4 | 3 | 11 | 0 | 2 | 1 |
| **Diff** | 3 | 1 | 2 | 2 | 3 | 1 | 0 | 5 | 0 | 0 |
| **Game** | 0 | 0 | 2 | 1 | 3 | 5 | 2 | 0 | 9 | 1 |
| **SIMD** | 3 | 6 | 8 | 0 | 6 | 0 | 1 | 0 | 1 | 16 |

Top co-occurring pairs:
1. **Attn + SIMD** (8 papers) — novel attention mechanisms often need hardware optimization
2. **Distill + TTC** (8 papers) — distillation and test-time compute are complementary strategies
3. **Attn + KV** (7 papers) — attention innovation frequently targets KV cache efficiency
4. **KV + SIMD** (6 papers) — KV compression requires performant kernels
5. **Distill + SIMD** (6 papers) — compression techniques need hardware-friendly implementations

---

## Papers by Architecture Type

### Transformer-Based (Standard Architecture)
| Papers | Count |
|--------|-------|
| 00, 01, 02, 04, 06, 07, 08, 11, 12, 13, 15, 16, 18, 19, 21, 25, 26, 30, 31, 32, 33, 36, 37, 38, 40, 43, 46, 48, 49, 53, 54, 57, 58, 61 | **34** |

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
| 17 (BLT byte-level), 22 (Lighthouse pyramid), 35 (Attractor fixed-point), 48 (HRM recurrent), 50 (LDT lattice), 51 (Deep Manifold), 55 (Tri-Mode dual-stream), 59 (MoE co-design), 60 (MeMo memory model), 061 (Delta Block cross-layer) | **10** |

### Non-Architecture (Strategy / Engineering / Dataset)
| Papers | Count |
|--------|-------|
| 03, 05, 09, 14, 20, 23, 27, 29, 39, 47, 52, 56, 61 | **13** |

---

## Summary of Intersection Highlights

### 1. Highest Direct Value (Direct Fit, Already Implemented)

| Paper | What We Adopted | Where |
|-------|----------------|-------|
| 02 Leviathan | Speculative decoding with rejection sampling | `speculative/verifier.rs` |
| 06 Raven | O(1) slot memory routing | `forward_raven()` |
| 08 TwELL | Sparse MLP matmul for ReLU activations | `types.rs sparse_matmul` |
| 20 TurboQuant | 3-bit KV cache quantization (legacy baseline) | `turboquant` module |
| 28 HLA/AHLA | Second-order linear attention, 88% memory savings | `forward_hla`, `forward_ahla` |
| 39 SpectralQuant | Eigenbasis rotation + water-fill (secondary KV, 9.1× compression) | `spectralquant` module |
| 63 OCTOPUS | Octahedral triplet codec (**primary default**, 12.2× compression, -22% to -49% MSE vs SQ) | `octopus` module |
| 40 BT Ranking | Bradley-Terry pairwise ranking (**default**, GOAT 4/4) | `pruners/bt_rank.rs` |
| 42 SP-KV | Self-pruned KV attention, 3-10× reduction | SP-KV module |
| 44 ELF | SDE noise injection (**default**, 10-22× path diversity) | `inject_sde_noise` |
| 45 MaxSim | Late-interaction scoring, CPU SIMD 7.46× | MaxSim primitive |
| 51 Deep Manifold | Fixed-point residual scoring (**default**, GOAT 6/6) | `deep_manifold` module |
| 52 SimpleTES | RPUCG bandit loop (GOAT 8/8) | `tes_loop` module |
| 53 CNA | Contrastive neuron attribution + sparse modulation (**default**, GOAT proved) | `cna_steering` module |
| 55 Nemotron | Tri-Mode AR+Diffusion+Self-Speculation | `dllm` + `tri_mode` features |
| 56 Unit Distance | Minkowski lattice GOAT proof primitive | `unit_distance` module |
| 59 MoE+SD | Amdahl cost model for speculative decoding | `spec_cost_model` feature |
| 60 MeMo | Reflection QA pipeline + TIES merging | `memo_reflections` feature |
| 061 Delta Routing | Cross-layer residual delta routing | `delta_routing` feature |

### 2. Strong Conceptual Alignment (Pattern Adopted, Different Mechanism)

| Paper | What We Distilled | Our Equivalent |
|-------|-------------------|---------------|
| 09 EMO | Document-level expert routing | `KeywordRouter` + `ExpertRegistry` |
| 14 Heuristic Learning | Code-based policy evolution | `BanditPruner` + `AbsorbCompress` |
| 24 Delta-Mem | Delta-rule associative memory | Feature-hashed Rust implementation |
| 36 ROPD Rubric | Multi-criteria reward vectors | `ropd_rubric` feature (off by default) |
| 37 REAP | Model-based/modelless spectrum | Existing trait stack captures both |
| 38 SDAR | Sigmoid-gated distillation | `sdar_gate` feature (negative arena result) |
| 49 PTRM | Noise-injected recursive refinement | `inject_sde_noise` + DDTree |
| 58 GRAM | Learned-mean SDE guidance | `elf_sde` + width scaling validates approach |

### 3. Selective Adoption (Specific Techniques Only)

| Paper | What We Took | What We Skipped |
|-------|-------------|-----------------|
| 10 ColaDLM | KV cache priming concept | Full VAE-DiT mechanism |
| 17 Fast BLT | Self-speculation validates our approach | Byte-level model architecture |
| 41 RePlaid | ELBO regularization, variance-minimized schedules | Full continuous diffusion |
| 48 HRM-Text | Adam-atan2 optimizer, PrefixLM batching | Full hierarchical recurrent model |
| 57 ART | CISPO loss concept (wider clip for GRPO) | Full Python RL framework |

### 4. Negative Results (Not Applicable to Our Stack)

| Paper | Why Not Applicable |
|-------|-------------------|
| 03 Commercial Strategy | Business document, not a technique |
| 05 Artifact Definition | Terminology clarification only |
| 25 StepCode | NO GAIN proven — paper's 7-14% gains from training 7B on dense rewards, modelless path doesn't benefit |
| 29 rust-gpu Feasibility | WGSL→Rust migration, deferred for nightly requirement |
| 38 SDAR Arena | Negative arena result — ELO 954 ≈ Rubric 955, no improvement, 28% higher bandit regret |
| 47 PGD Go Dataset | Dataset paper, features already captured by GoHeuristic |

### 5. Gaps Identified (Features Papers Have That We Don't)

| Gap | Source Papers | Priority | Feature Plan |
|-----|--------------|----------|--------------|
| ASFT anchored SFT loss (self-prob weighting + KL anchor) | 54 (ASFT) | Medium | `asft_loss` planned for riir-gpu |
| CISPO loss variant (wider clip ε=1.0/4.0 for GRPO) | 57 (ART) | Medium | `cipo_loss` planned for microgpt-rs |
| SLIME reference-free preference optimization | 61 (SLIME) | Medium | `slime_loss` planned for riir-gpu |
| Interventional SFT causal masking | 43 (Interventional SFT) | Low — 1.19 nats/token gain | Not yet scheduled |
| GRAM learned-mean SDE (μ_θ not zero) | 58 (GRAM) | Low — elf_sde covers zero-mean | Extends `elf_sde` |
| Adam-atan2 optimizer | 48 (HRM-Text) | Low — simple drop-in | Not yet scheduled |

---

## Feature Coverage Radar

Our implementation status per feature dimension:

```
Speculative Decoding  ████████████████████ 95%  (DDTree, DFlash, Leviathan, MTP, Tri-Mode self-speculation)
KV Optimization       ████████████████████ 95%  (OCTOPUS primary default, SpectralQuant secondary, SP-KV, TurboQuant legacy)
Attention Innovation  ████████████████████ 90%  (HLA, AHCLA, Percepta, MaxSim, Tri-Mode dual-stream)
Noise Scheduling      ████████████████░░░░ 80%  (SDE injection default, GRAM learned-mean validates, PTRM)
Distillation          █████████████░░░░░░░ 65%  (LoRA, BT ranking, ROPD, MeMo; ASFT/CISPO/SLIME planned)
Test-Time Compute     █████████████████░░░ 85%  (SimpleTES GOAT 8/8, BanditPruner, GRAM width scaling)
Routing/MoE           ███████████████░░░░░ 75%  (Raven, MoE+SD cost model, TIES merging, Delta Block)
Diffusion/Denoising   ██████████░░░░░░░░░░ 50%  (D2F, Tri-Mode validates, RePlaid schedules experimental)
Game/Self-Play        ██████████████████░░ 90%  (Sudoku, Go, Monopoly, Bomber, Unit Distance lattice)
SIMD/Perf             ████████████████████ 95%  (NEON, zero-alloc, Minkowski lattice embedding)
```

---

## References

All papers are located in `microgpt-rs/.research/` with filenames `{index}_{Title}.md` where index ranges from 00 to 61 (plus 061 for Delta Attention Residuals). See individual research files for full analysis, verdicts, and implementation details.