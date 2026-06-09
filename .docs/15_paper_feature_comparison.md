# Paper Feature Comparison Matrix

**Date:** 2025-07
**Status:** Living Document
**Scope:** All 69 research papers (00–069) in `.research/` mapped against katgpt-rs feature dimensions. Includes Research 061 (Delta Attention Residuals) mapped to `delta_routing`. Includes Research 068 (RAEv2) mapped to `mls_aggregate`.

## Introduction

This document provides a comprehensive feature-intersection matrix between our work (katgpt-rs) and all 69 researched papers. Each paper is evaluated across 10 feature dimensions derived from our core architecture:

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

## Our Work: katgpt-rs Feature Summary

| Feature | GOAT (default-on) | Gain | NO GOAT (opt-in / legacy) |
|---------|-------------------|------|--------------------------|
| Speculative Decoding | • DDTree + DFlash<br>• Leviathan verification | Always ≥1 token/step | • Tri-Mode self-speculation (untrained accept rate 1.0) |
| Phrase Boost | • **PhraseBoost** (Plan 165) | DDTree phrase-level match +SD recall | — |
| KV Optimization | • **Hybrid OCT+PQ** (Plan 101) | Best MSE all bits, 64× fewer FMAs | • OCTOPUS (legacy)<br>• SpectralQuant (9.1×, 0.9917 cosine)<br>• SP-KV (3-10×)<br>• TurboQuant 3-bit (legacy) |
| Attention Innovation | • **GDN2** (GOAT 14/14) | 99.4% AHLA throughput, 87-98% memory savings | • HLA / AHLA (88% memory savings)<br>• Percepta 2D Convex Hull<br>• MaxSim<br>• SHINE Alternating2D (90% FLOPs savings) |
| Noise Scheduling | • ELF SDE noise injection | 10-22× path diversity | • GRAM learned-mean SDE (validates approach) |
| Distillation / Compression | • BT pairwise ranking<br>• SpectralQuant | +10.6pp over pointwise; 9.1× compression | • LoRA adapters<br>• MeMo reflections<br>• ROPD rubric<br>• ASFT/SLIME (in riir-gpu) |
| Reflective Distillation | • **GEPA-D Reflective** (Plan 164) | TTC-aware distill loop with self-correction | — |
| Test-Time Compute | • SimpleTES RPUCG loop (GOAT 8/8)<br>• BanditPruner adaptive arms | RPUCG 42.8% vs greedy 10.6% wins | • GRAM width scaling |
| Adaptive Compute | • **Hydra Budget** (Plan 166) | Dynamic multi-head compute allocation | — |
| Budget Adaptation | • **Budget Adaptation** (Plan 167) | Per-prompt adaptive budget scaling | — |
| Routing / MoE | • Delta Block cross-layer residual routing | Zero throughput overhead (0.97×) | • Raven slot memories<br>• MoE+SD Amdahl cost model<br>• TIES merging (MeMo)<br>• SHINE context→LoRA routing |
| Diffusion / Denoising | — | — | • dLLM D2F block-parallel denoising<br>• Tri-Mode AR+Diffusion+Self-Speculation (GOAT 4/4, partial) |
| Dual-Path Consensus | • **FlashAR Consensus** (Plan 167) | AR + diffusion dual-path consensus decode | — |
| Game / Self-Play | • LEO all-goals Q-framework (Plan 155)<br>• Dual LEO teacher/student (Plan 155) | SUPER GOAT; all-goals Q(s)→R^{G×A} | • Sudoku, Go, Monopoly, Bomber<br>• Unit Distance lattice constructions |
| SIMD / Perf | • LDT α-intersection<br>• TileRT pipeline (GOAT 13/13, Plan 102)<br>• PlasmaPath bit-plane (GOAT 5/5, Plan 148) | +0.6% overhead → full observability; multiplication-free matvec | • NEON SIMD matmul/HLA kernels<br>• zero-alloc hot paths<br>• Minkowski lattice embedding |
| ManifoldPruner (Plan 234) | Soft Validity | • ConstraintPruner manifold_score<br>• KernelScreeningPruner | G2 PASS: Gaussian 10/10 recall | • BFCP region radius adaptation (opt-in) |

**Default feature set:** `sparse_mlp`, `domain_latent`, `ppot`, `bandit`, `bt_rank`, `spectral_quant`, `hybrid_oct_pq`, `elf_sde`, `cna_steering`, `deep_manifold`, `federation`, `tes_loop`, `lattice_deduction`, `delta_routing`, `stability_metrics`, `mls_aggregate`, `gdn2_attention`, `dash_attn`, `dreamer`, `lt2_looped`, `dmax_spd`, `eqr_convergence`, `subterranean`, `sr2am_configurator`, `data_gate`, `plasma_path`, `parallel_probe`, `tf_loop`, `leo_all_goals`, `dual_leo`, `sigmoid_margin`, `moa_inference`, `sleep_consolidation`, `spectral_hierarchy`, `dual_gram_pca`, `roofline_cost`, `gepa_reflective`, `phrase_boost`, `hydra_budget`, `flashar_consensus`, `budget_adaptation`

---

## Feature Intersection Matrix

### Our Architecture (Reference Row)

| # | Paper / Feature | SD | KV | Attn | Noise | Distill | TTC | Route | Diff | Game | SIMD |
|---|----------------|----|----|------|-------|---------|-----|-------|------|------|------|
| — | **katgpt-rs (our work)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

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
| 62 | SHINE Scalable In-Context Hypernetwork | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ○ |

### Papers 63–69: KV Compression, Inference, Rotation, Pipelines, GEMM, Representation & Dreamer

| # | Paper / Feature | SD | KV | Attn | Noise | Distill | TTC | Route | Diff | Game | SIMD |
|---|----------------|----|----|------|-------|---------|-----|-------|------|------|------|
| 63 | OCTOPUS Octahedral KV Cache Compression | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 64 | LlamaWeb WebGPU Inference Distillation | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ○ |
| 65 | RotorQuant Block-Diagonal Rotation Quantization | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 66 | TileRT Persistent Tile Pipeline Inference | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 67 | CODA GEMM Epilogue Programming | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 68 | RAEv2 Multi-Layer Representation Autoencoders | ✗ | ✗ | ○ | ✗ | ○ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 69 | AutoDreamer Offline Memory Consolidation | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ○ | ✗ | ✗ | ✗ |

### Paper 155: LEO All-Goals Trait Framework

| # | Paper / Feature | SD | KV | Attn | Noise | Distill | TTC | Route | Diff | Game | SIMD |
|---|----------------|----|----|------|-------|---------|-----|-------|------|------|------|
| 155 | LEO Learn Everything All at Once (Matthews 2026) | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |

**Feature gates:** `leo_all_goals` (LeoHead + AllGoalsUpdate + sigmoid_bounded_q), `dual_leo` (+ DualLeoMixer + AutocurriculumSampler, requires `leo_all_goals`). Both **default-on** — SUPER GOAT, zero perf overhead. Framework only — depends on riir-ai Plan 155 for game-specific implementations.

### Papers 142–144: Diffusion Training, Retrieval Representations, Functional Emotions

| Paper | Title | Feature Match |
|-------|-------|--------------|
| R142 | JLT: Clean-Latent Prediction in Latent Diffusion Transformers (Fu et al. 2026) | D2F clean prediction (CE), LT2 layer loop (TF-Loop), EMA alignment |
| R143 | Latent Terms: Dense Retrievers Contain Extractable BM25-Ready Vocabularies (Clavié et al. 2026) | MaxSim (validates Plan 080 > Latent Terms), ScreeningPruner |
| R144 | Functional Emotions as Linear Representations (Sofroniew et al. / Anthropic 2026) | Emotion Vector Inference (Plan 162), ReviewMetrics desperation monitor |

**R142 — JLT:** Validates our D2F clean prediction (CE on original tokens is the correct target parameterization; v-prediction is strictly harder per Var(v|z) = Var(x|z)/(1-t)²). Also validates LT2 layer loop — JLT uses identical `loop_indices/loop_count` pattern independently. **Decision: NO NEW PLAN** — existing D2F and LT2 already implement the validated techniques.

**R143 — Latent Terms:** Shows dense retrievers contain BM25-searchable latent vocabularies extractable via SAE. MaxSim (Plan 080) outperforms Latent Terms for multi-vector models (GTE-MC: MaxSim 0.547 vs LT 0.500). No gain for speculative decoding pipeline — SAE→BM25 is document-retrieval-specific. **Decision: NO GAIN** — validates MaxSim choice.

**R144 — Functional Emotions:** 171 emotion concepts with linear representations in Claude Sonnet 4.5 activation space (valence PC1, arousal PC2). Causal steering: desperation +0.1 → 14× reward-hacking increase. `calm` direction is protective (0% blackmail). Operationalized as `EmotionDirections` / `EmotionReading` in Plan 162 for zero-cost decode-time desperation monitoring.

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
| **62** | SHINE Scalable In-Context Hypernetwork | Attn✓ Distill✓ Route✓ | Context→LoRA single forward pass, alternating 2D attention (90% FLOPs savings), M2P Transformer |
| **R144** | Functional Emotions as Linear Representations (Sofroniew et al. / Anthropic 2026) | Distill○ TTC○ Route○ | Emotion vector inference (Plan 162), zero-cost desperation monitoring via linear probes in activation space |

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
| 06 (Raven), 24 (Delta-Mem), 28 (HLA), 42 (SP-KV), 45 (MaxSim), 70 (GDN2) | **6** |

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
| 70 GDN2 | Gated DeltaNet-2, decoupled erase/write gates, O(1) decode, 99.4% AHLA throughput, 87–98% memory savings, GOAT 14/14 (**default-on**) | `src/gdn2/`, `gdn2_attention` feature |
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
| 62 SHINE | Context→LoRA hypernetwork, alternating 2D attention | `shine_hypernet` / `shine_routing` features |
| 155 LEO | All-goals Q-value trait framework (SUPER GOAT, **default-on**) | `leo_all_goals` + `dual_leo` features |

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
| CISPO loss variant (wider clip ε=1.0/4.0 for GRPO) | 57 (ART) | Medium | `cipo_loss` planned for katgpt-rs |
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
Attention Innovation  ████████████████████ 95%  (GDN2 GOAT 14/14 default-on, HLA, AHLA, Percepta, MaxSim, Tri-Mode dual-stream)
Noise Scheduling      ████████████████░░░░ 80%  (SDE injection default, GRAM learned-mean validates, PTRM)
Distillation          █████████████░░░░░░░ 65%  (LoRA, BT ranking, ROPD, MeMo; ASFT/CISPO/SLIME planned)
Test-Time Compute     █████████████████░░░ 85%  (SimpleTES GOAT 8/8, BanditPruner, GRAM width scaling)
Routing/MoE           ████████████████░░░░ 80%  (Raven, MoE+SD cost model, TIES merging, Delta Block, SHINE context routing)
Diffusion/Denoising   ██████████░░░░░░░░░░ 50%  (D2F, Tri-Mode validates, RePlaid schedules experimental)
Game/Self-Play        ███████████████████░ 95%  (Sudoku, Go, Monopoly, Bomber, Unit Distance lattice, LEO all-goals Q-framework)
SIMD/Perf             ████████████████████ 95%  (NEON, zero-alloc, Minkowski lattice embedding)
```

---

## References

All papers are located in `katgpt-rs/.research/` with filenames `{index}_{Title}.md` where index ranges from 00 to 73 (plus 061 for Delta Attention Residuals). See individual research files for full analysis, verdicts, and implementation details. Papers 63–69 added: OCTOPUS (63), LlamaWeb (64), RotorQuant (65), TileRT (66), CODA (67), RAEv2 MLS (68), AutoDreamer (69). Key post-69 papers: 70 (GDN2 recurrent attention), 71 (DashAttention sparse), 72 (DMax SPD), 73 (LT2 looped inference). Recent additions: Research 110 (PlasmaPath, Plan 148), Research 094 (Parallel-Probe, Plan 133), Research 073 / feature `tf_loop` (Training-Free Loop, Plan 136), Research 118 / Plan 155 (LEO All-Goals Trait Framework — Matthews et al. 2026).

- R142: Fu et al. (2026) — JLT: Clean-Latent Prediction in Latent Diffusion Transformers. arXiv:2605.27102
- R143: Clavié et al. (2026) — Latent Terms: Dense Retrievers Contain Extractable BM25-Ready Vocabularies. arXiv:2605.29384
- R144: Sofroniew et al. / Anthropic (2026) — Emotion Concepts and their Function in a Large Language Model (Transformer Circuits Thread)
