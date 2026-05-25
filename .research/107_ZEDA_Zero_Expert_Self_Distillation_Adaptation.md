# Research 107: ZEDA — Zero-Expert Self-Distillation Adaptation for Post-Trained MoE

> **Source:** [Post-Trained MoE Can Skip Half Experts via Self-Distillation](https://arxiv.org/pdf/2605.18643) — Xingtai Lv et al. (Tsinghua/ZEDA), 2026-05
> **Date:** 2026-05, distilled 2026-05
> **Related Research:** 037 (REAP model-based/modelless), 059 (MoE+SD co-design), 058 (GRAM), 036 (ROPD), 038 (SDAR), 054 (ASFT)
> **Related Plans:** 096 (MoE+SD cost model), 097 (delta routing)
> **Verdict: PARTIAL VALUE — Three distillations: (1) Group Auxiliary Loss concept → modelless inference budget regulation, (2) zero-expert as "skip" primitive → SR²AM early-exit analogy, (3) two-stage SFT→OPD self-distillation → validates our existing ROPD+SDAR pipeline design. Core MoE expert-skipping does NOT apply (no MoE architecture). Game-specific inference budget allocation maps to SR²AM Plan 112 — already implemented.**

---

## TL;DR

ZEDA converts a post-trained static MoE into a dynamic one by injecting parameterless zero-output experts and adapting via two-stage self-distillation (SFT → on-policy distillation). Key results: **51-53% expert FLOPs eliminated, ~20% end-to-end speedup, <1% accuracy loss** on Qwen3-30B-A3B and GLM-4.7-Flash across 11 benchmarks.

The core innovation is the **Group Auxiliary Loss** — instead of balancing individual experts (which destroys learned routing), balance only between normal-expert group and zero-expert group. This preserves the post-trained routing distribution while allowing controllable compute reduction.

**Our mapping:** We don't have MoE architecture, but the principles map to:
- Token-level dynamic compute allocation → our SR²AM configurator bandit (Plan 112)
- Group-level budget regulation → our domain inference budget (Plan 026)
- Zero-expert skip → our early exit / speculative verification skip
- SFT→OPD self-distillation → validates our ROPD (Plan 071) + SDAR (Plan 072) pipeline design

---

## 1. Key Findings

### 1.1 Zero-Expert Injection

Each MoE layer has N normal experts, activates K per token. ZEDA injects N_Z zero experts (output = 0 for all inputs). The router selects from N + N_Z candidates but still picks top-K. When zero experts are selected, fewer normal experts execute — **parameter-free compute reduction**.

Key insight: zero experts beat copy experts (output = input). Copy experts cause scale + direction mismatch that compounds across layers.

### 1.2 Group Auxiliary Loss

Standard auxiliary loss L_A forces uniform routing across all experts → destroys post-trained routing patterns.

Group Auxiliary Loss L_GA only balances between two groups:
- Group E: normal experts (N experts)
- Group Z: zero experts (N_Z experts)

```
L_GA = α · (N + N_Z) / (K · w) · (f_E · P_E / N + f_Z · P_Z / (N_Z · w))
```

Where `w > 0` controls zero-expert group weight. Target r_ZE = (N_Z · w) / (N + N_Z · w).

At w=2, target r_ZE ≈ 50%. Empirical: 51.2% (Qwen), 53.0% (GLM).

### 1.3 Two-Stage Self-Distillation

1. **SFT stage**: Train on teacher-sampled responses. Stabilizes initial router transition.
2. **OPD stage**: Train on student-sampled responses evaluated by teacher. Closes distribution gap.

SFT→OPD > SFT-only > OPD-only. SFT must come first to stabilize zero-expert routing before OPD can refine.

### 1.4 Zero-Expert Activation Dynamics

r_ZE correlates with:
- **Teacher-student logp-diff** (Δlogp): larger gap → fewer zero experts → more compute
- **Student entropy**: higher entropy → fewer zero experts
- **Response pattern**: code/math expressions get more zero experts (less compute), natural text gets fewer
- **NOT task difficulty**: same r_ZE across MATH-500 difficulty levels

This means: **the model intrinsically allocates more compute to uncertain/hard tokens, not hard tasks**. This is a token-level property.

### 1.5 No Router Probability Renormalization

When zero experts are selected, do NOT renormalize the remaining expert weights. Renormalization amplifies routing weights → inflates MoE residual branch → consistent accuracy drop.

---

## 2. Mapping to Our Architecture

### 2.1 What Applies (Direct)

| ZEDA Concept | Our Equivalent | Status |
|--------------|---------------|--------|
| Dynamic compute allocation per token | SR²AM Configurator Bandit (Plan 112) | ✅ Implemented, GOAT proved |
| Group-level budget regulation | Domain inference budget (Plan 026) | ✅ Implemented |
| Token-level difficulty → compute | BanditPruner Q-values per token | ✅ Implemented |
| SFT→OPD self-distillation | ROPD (Plan 071) + SDAR (Plan 072) pipeline | ✅ Implemented, GOAT proved |
| Teacher-student logp gap signal | δ-Mem (Plan 053) + BanditPruner | ✅ Infrastructure |
| Skip/expert-omission primitive | Early exit patience (Plan 026) | ✅ Implemented |

### 2.2 What Does NOT Apply

| ZEDA Concept | Why Not |
|--------------|---------|
| Zero-expert injection | No MoE architecture — we have dense + sparse MLP, not expert routing |
| Router probability balancing | No router per se — our routing is bandit-based, not softmax expert selection |
| Expert-level FLOPs reduction | Our sparse MLP (Plan 022) achieves structured sparsity differently |
| Post-trained MoE conversion | No post-trained MoE model to convert |

### 2.3 Conceptual Validation

ZEDA validates several design choices we already made:

1. **Token-level dynamic compute** — ZEDA proves it works at scale (51% FLOPs reduction). Our SR²AM does this at the bandit level.

2. **Group-level regulation > item-level** — ZEDA's L_GA insight (don't disrupt learned routing) parallels our decision to keep BanditPruner Q-values per-domain rather than per-token-state.

3. **Two-stage distillation** — ZEDA's SFT→OPD ordering matches our ROPD→SDAR pipeline. Both find SFT stabilization is necessary before on-policy refinement.

4. **Don't renormalize** — When skipping compute (early exit), don't artificially boost the remaining path. This validates our early-exit design where we don't rescale the accepted branch probability.

5. **Compute correlates with token uncertainty, not task difficulty** — This validates our entropy anomaly detection (Plan 061). High-entropy tokens should get more budget.

---

## 3. Distillations for Our System

### D1: Group Budget Regulation Principle (modelless)

**From ZEDA:** Group Auxiliary Loss balances between "compute" and "skip" groups, not individual items.

**For us:** When SR²AM decides inference budget per turn, it should regulate at the group level (domain × phase) rather than per-token. Our existing SR²AM already does this. **No new code needed** — conceptual validation only.

### D2: Zero-Compute Skip as Architectural Primitive (modelless)

**From ZEDA:** Zero experts (output = 0, no computation) are a clean skip mechanism.

**For us:** Our early-exit patience + speculative verification skip are analogous. The principle that skip should be a true no-op (not a residual passthrough like copy experts) validates our early-exit design where we stop decoding rather than degrade. **No new code needed.**

### D3: SFT→OPD Ordering Validation (model-based)

**From ZEDA:** SFT first to stabilize routing, then OPD to close distribution gap.

**For us:** Our ROPD (rubric-based SFT) → SDAR (gated distillation, model-based) follows exactly this pattern. ZEDA provides external validation. **No new code needed** — our pipeline is correctly ordered.

---

## 4. Verdict

**NOT ACTIONABLE** — ZEDA's contributions are MoE-specific (zero-expert injection, router balancing) and don't apply to our dense + sparse MLP architecture. The general principles (token-level dynamic compute, group-level budget regulation, two-stage distillation ordering, don't-renormalize) are already captured by our existing SR²AM, ROPD, SDAR, and early-exit implementations.

ZEDA serves as **external validation** that our design choices are correct:
- SR²AM configurator ≈ ZEDA's dynamic compute allocation
- ROPD→SDAR ≈ ZEDA's SFT→OPD
- Early exit patience ≈ ZEDA's zero-expert skip
- Entropy anomaly detection ≈ ZEDA's r_ZE ∝ entropy finding

**No plan needed.** This research is documentation of alignment, not a source of new features.

---

## 5. Game Relevance (MMO GOAT Pillars Assessment)

| Pillar | ZEDA Relevance |
|--------|---------------|
| P1: Fourier Spatial AI | ❌ No connection |
| P2: WASM Validators | ❌ No connection |
| P3: NPC Dialog Engine | ⬜ Indirect — dynamic compute could allocate more budget to dialog turns with high uncertainty, but SR²AM already covers this |
| P4: Frame-Sampling Bridge | ⬜ Indirect — frame decimation is already our "skip" mechanism for real-time AI |

**Not a GOAT pillar candidate.** ZEDA is a training-time technique for MoE models. We have no MoE. The conceptual principles are already in our stack.

---

## 6. Why NOT riir-ai Domain

ZEDA is NOT game-specific or super-GOAT. It's a general MoE inference optimization. Even if we had MoE architecture:
- The zero-expert injection concept is published and non-secret
- The Group Auxiliary Loss is a published mathematical formula
- No private game knowledge is encoded

This stays in katgpt-rs research as **external validation documentation**, not a new feature.

---

## References

- ZEDA paper: arxiv 2605.18643 (2026-05)
- MoE++ (zero-expert concept): Jin et al. (2024) arxiv 2410.07348
- LongCat-Flash (industrial zero-expert): Meituan LongCat Team (2025)
- MiniLLM (on-policy distillation): Gu et al. (2023) arxiv 2306.08543
- GKD (on-policy distillation): Agarwal et al. (2024)
