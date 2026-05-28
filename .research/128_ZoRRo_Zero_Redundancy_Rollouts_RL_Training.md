# Research 128: ZoRRo — Zero Redundancy Rollouts for Enterprise RL Training

> **Paper:** [Zero Redundancy Rollouts (ZoRRo): Breaking the Speed-of-Light for Enterprise RL Training](https://www.snowflake.com/en/blog/engineering/zorro-enterprise-rl-training/) — Snowflake (Rajbhandari, Wang, Wyatt et al.), May 2026
> **Date:** 2026-05-28
> **Related:** SpecHop (R091/Plan 131), DashAttention (R071/Plan 106), PFlash (Plan 048), MTP Cluster (R078), TurboQuant (R020), OCTOPUS (R063), G-Zero Self-Play (R021), SDAR (R038)
> **Verdict:** ⚠️ NO GAIN — validates existing architecture, no new algorithm to implement

---

## Task

- [x] Fetch and analyze ZoRRo paper
- [x] Map to our architecture (katgpt-rs + riir-ai)
- [x] Cross-reference with GOAT pillars decision matrix (27)
- [x] Verdict: NO GAIN — research only, no plan, no feature gate

---

## Executive Summary

ZoRRo eliminates prompt redundancy in RL training by deduplicating shared prompt computation across multiple rollouts. Three techniques: (1) **Split Attention** — decompose attention into prompt-only self-attention (computed once) + response-to-full cross-attention (per rollout), giving 6x speedup on actor update; (2) **Forest Cascade Attention** — extend cascade attention to multi-prefix RL setting with on-chip SMEM reuse during decode, giving 1.7x speedup on generation; (3) **Speculative Decoding** — LSTM-based draft model for faster rollout tokens.

**Production results (Arctic Text2SQL R2, 32×H200):** 3.5x total RL training speedup, 3.2x longer context window (20K→64K), model beats Gemini 3.1 Pro and Claude 4.7 on enterprise SQL benchmark.

---

## Paper Core

### 1. Split Attention (Training)

In RL with R rollouts per prompt, standard systems compute prompt attention R times. ZoRRo decomposes:
- **Prompt-only self-attention**: causal self-attention over unique prompts — computed once
- **Response-to-full cross-attention**: each response attends to shared prompt + its unique response

Speedup factor: `R × (P + R_len) / (P + R × R_len)` where P=prompt_len, R_len=response_len, R=rollouts

For Arctic Text2SQL (P=16K, R_len=4K, R=16): ~6x actor update, ~5x log-prob.

### 2. Forest Cascade Attention (Inference)

Standard prefix caching eliminates redundant prefill but not redundant decode. At each decode step, every rollout re-reads the same shared prompt KV from HBM.

Forest Cascade:
1. Discover groups of requests sharing KV-cache prefixes (lexicographic sort + LCP)
2. Single grouped attention per prefix group — shared KV in GPU SMEM
3. Separate suffix attention per unique continuation
4. Merge via log-sum-exp

Extension of FlashInfer cascade attention from single-prefix tree to multi-prefix forest.

### 3. Speculative Decoding (Generation)

LSTM-based Arctic Speculator predicts multiple tokens, target model verifies in single forward pass. Combined with Forest Cascade: fewer HBM reads per step AND fewer steps.

---

## Mapping to Our Architecture

| ZoRRo Technique | Our Equivalent | Status | Gap |
|-----------------|---------------|--------|-----|
| Split Attention (prompt dedup) | N/A — we don't do RL training with multi-rollout | ❌ Not applicable | We don't have large-scale RL training |
| Forest Cascade Attention | DashAttention + PFlash + TurboQuant/SpectralQuant/OCTOPUS | ✅ Different approach | We handle shared KV via compression, not SMEM reuse |
| Speculative Decoding | SpecHop (Plan 131) + MTP (Plan 055) | ✅ Already implemented | SpecHop is more general (multi-hop, not just token-level) |
| Prompt deduplication batching | DDTree branch pool | ✅ Conceptual analog | DDTree deduplicates at tree level |
| Collision-resistant rewards | ConstraintPruner + WASM validators | ✅ Already implemented | Our validators are deterministic, no near-miss issue |
| Rapid RL iteration | G-Zero self-play (Plan 049) + SDAR (Plan 072) | ✅ Different domain | Game AI, not SQL generation |

---

## Why NO GAIN

### 1. Split Attention — Not Our Domain ❌

Split attention is a **training** optimization for RL workloads with many rollouts per prompt. Our stack:
- **katgpt-rs**: Pure inference. No RL training loop. Freeze/Thaw (Plan 092) persists bandit Q-values across sessions — replay with same seeds, but **1 rollout per game state** (R=1), so dedup factor = 1.0.
- **riir-ai**: wgpu LoRA training is small-scale (rank-4, V=32, D=16). We don't have 16K prompts with 16 rollouts. Self-play generates diverse game states, not repeated identical prompts.

The speedup formula `R × (P + R_len) / (P + R × R_len)` requires R >> 1 rollouts with identical long prompts. Our game AI domain has diverse game states (each "prompt" is unique), so deduplication ratio ≈ 1.0 (no gain).

**Freeze/Thaw (Plan 092) specifically:** The LEARN→REPLAY pipeline replays the same deterministic seeds with thawed Q-values. Each round is a single game with deterministic RNG — one rollout per game state. DreamerFrozenBank persists consolidation stats (episode counter, arms_before/after), not attention intermediates. No shared prompt attention to deduplicate.

### 2. Forest Cascade — Different Scale ❌

Forest Cascade is a **GPU serving** optimization for concurrent batched decode with shared prefixes across many requests. Our stack:
- Single-device inference, not multi-GPU serving
- DashAttention already handles sparse hierarchical attention
- KV cache compression (TurboQuant/SpectralQuant/OCTOPUS) handles long-context KV differently — via compression, not SMEM reuse
- PFlash handles block-sparse prefill

Forest Cascade requires GPU shared memory management for batched concurrent requests. Our Metal/wgpu inference serves one request at a time.

### 3. Speculative Decoding — Already Covered ✅

SpecHop (Plan 131) is strictly more general:
- Continuous multi-hop speculation at trajectory level
- Commit/rollback at tool-call granularity
- Theoretical cost model (α, β, p) for thread-count sizing
- MTP (Gemma-style) provides token-level speculation

Arctic Speculator (LSTM) is a specific instance of speculative decoding. Our SpecHop + MTP covers this.

---

## Validation of Existing Architecture

### 1. Redundancy Elimination Validates Our Pruning Philosophy ✅

ZoRRo's core insight: "eliminate redundancy in computation space." Our entire stack does this at different levels:
- **ConstraintPruner**: Eliminates invalid tokens from generation
- **ScreeningPruner**: Eliminates irrelevant context from retrieval
- **TurboQuant/SpectralQuant/OCTOPUS**: Eliminates redundant KV cache entries
- **DashAttention**: Eliminates irrelevant attention blocks via adaptive sparsity
- **SpecHop**: Eliminates idle wait time via speculative threads

### 2. Split Attention Pattern Validates PFlash Block Design ✅

ZoRRo's "compute shared prefix once, attend per-response" is structurally similar to PFlash's "block-sparse speculative prefill" — both avoid recomputing shared prefix attention.

### 3. Forest Cascade Validates DashAttention Hierarchy ✅

ZoRRo's prefix-grouping → shared attention → suffix merge is the same pattern as DashAttention's Stage 0 (chunk summarization) → Stage 1 (block routing) → Stage 2 (fine-grained attention).

### 4. Collision-Resistant Rewards Validate ConstraintPruner ✅

ZoRRo's "near-miss" problem in SQL execution rewards mirrors our game validation: an invalid move that "accidentally" passes a weak validator. Our WASM validators (Pillar 2) are deterministic — no near-miss possible.

---

## Cross-Reference: GOAT Pillars Decision Matrix (27)

**Classification:** NOT a pillar. Infrastructure optimization for a domain we don't serve (large-scale RL training on GPU clusters).

| Criterion | Score | Reason |
|-----------|-------|--------|
| GOAT passed | N/A | We don't implement it |
| MMO-product | ❌ | RL training infrastructure, not game feature |
| LoRA-independent | ❌ | Only relevant during training |
| Defensible | ❌ | Standard systems optimization |
| Secret coverage | ❌ | No proprietary game knowledge |

**Pattern match:** Same category as KPop (R119) — online RL training technique for a scale we don't operate at. Validates our existing speculative + sparse + compression stack. No plan, no feature gate.

---

## Verdict

**⚠️ NO GAIN. No plan. No feature gate. Research only.**

ZoRRo is excellent systems engineering for large-scale RL training on GPU clusters. It validates three of our existing architectural choices (speculative decoding, sparse attention, KV cache optimization) but introduces no new algorithm applicable to our single-device inference + small-scale game LoRA training stack.

**Why no plan:**
1. Split attention requires multi-rollout RL training — we don't do this
2. Forest Cascade requires GPU SMEM batched serving — we serve single-device
3. Speculative decoding already covered by SpecHop + MTP (strictly more general)
4. The redundancy elimination insight is already operationalized in our pruning + compression + sparsity stack

**What it validates:**
1. Our SpecHop architecture (continuous speculation > single-hop)
2. Our DashAttention design (shared-prefix + per-query decomposition)
3. Our pruning philosophy (eliminate redundancy at every level)
4. Our ConstraintPruner + WASM validator approach (deterministic validation > soft rewards)
