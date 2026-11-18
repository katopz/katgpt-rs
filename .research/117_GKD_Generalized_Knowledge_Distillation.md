# Research 117: GKD — Generalized Knowledge Distillation (Learning from Self-Generated Mistakes)

> **Paper:** [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) — Agarwal, Vieillard, Zhou, Stanczyk, Ramos, Geist, Bachem (Google DeepMind), ICLR 2024
> **Date:** 2025-05-27
> **Related Research:** 036 (ROPD), 038 (SDAR), 037 (REAP model-based/modelless), 054 (ASFT), 080 (VPD), 115 (PEIRA)
> **Related Plans:** 072 (SDAR modelless), 073 (SDAR model-based)
> **Related MMO Pillars:** Cross-cutting — improves LoRA distillation quality (Secret A risk mitigation)

---

## TL;DR

GKD addresses **train-inference distribution mismatch** in autoregressive distillation by training the student on **self-generated on-policy sequences** with teacher logits as labels. It introduces a **generalized divergence framework** (forward KL, reverse KL, JSD(β)) and a **student data fraction λ** that interpolates between supervised (λ=0) and fully on-policy (λ=1). GKD also unifies distillation with RL fine-tuning.

**Verdict: CONCEPTUAL ALIGNMENT — GKD's core insights are already partially captured by SDAR (Research 038) and ROPD (Research 036). The missing piece is the systematic divergence choice framework. Limited new gain for game domain.**

---

## Paper Core

### Problem: Train-Inference Mismatch

Standard KD uses fixed datasets (teacher-generated or ground-truth) for training. During inference, the student generates autoregressive sequences that may be far from training data. Early token errors cascade → poor generation quality.

This is **exposure bias** from imitation learning (Ross & Bagnell, 2010).

### Solution: Generalized KD (GKD)

**Algorithm:**
```
For each step:
  With probability λ:
    Sample x from dataset, generate y ~ student(x)  // on-policy
  Otherwise:
    Sample (x, y) from fixed dataset                  // supervised
  Update student to minimize D(teacher || student)(y|x)
```

**Key innovations:**
1. **On-policy data** — Student generates its own training sequences
2. **Flexible divergence D** — Forward KL, reverse KL, JSD(β), or any f-divergence
3. **RL integration** — `L = (1-α)·L_RL + α·E[D(pT||pS)]`

### Key Findings

| Finding | Detail |
|---------|--------|
| On-policy (λ=1) consistently beats supervised (λ=0) | Across all tasks, all student sizes |
| Mixed (λ=0.5) also beats supervised | But pure on-policy is usually best |
| Divergence choice is task-dependent | Forward KL: greedy eval. Reverse KL: instruction tuning. JSD(0.1-0.9): translation |
| Reverse KL is mode-seeking | Student concentrates on high-probability teacher modes → better for focused tasks |
| Forward KL is mode-covering | Student covers teacher support → better for diverse generation |
| JSD(β) interpolates | β→0 ≈ forward KL, β→1 ≈ reverse KL |
| Self-distillation works | Student can surpass same-size teacher via on-policy GKD |
| GKD + RL works | Combine distillation loss with RL reward — reduces "alignment tax" |
| Data efficient | On-policy GKD on 5% data beats supervised KD on 100% data |

---

## Mapping to Our System

### What We Already Have (No Gain)

| GKD Concept | Our Equivalent | Status |
|-------------|---------------|--------|
| On-policy student generation | SDAR (Research 038) — student generates, teacher provides privileged skill context | ✅ Implemented |
| Token-level distillation loss | SDAR sigmoid gate — `g_t · Δ_t` with asymmetric trust | ✅ Implemented, **better** than GKD's uniform token loss |
| Train-inference mismatch fix | SDAR explicitly addresses multi-turn OPSD instability | ✅ Addressed |
| RL + distillation combo | `loss_grpo.rs` + `distill.rs` already combined in training loop | ✅ Implemented |
| Feature gate for distillation | `sdar_gate`, `vpd_em_distill`, `rmsd_distill`, `ropd_rubric` | ✅ Multiple gates |

### What's Missing (Potential Gain)

| GKD Insight | Gap in Our System | Assessment |
|-------------|-------------------|------------|
| **Systematic divergence choice** | We use SDAR's sigmoid gate (implicit) but don't expose forward KL / reverse KL / JSD(β) as configurable options | Minor — SDAR gate is a more principled approach than divergence switching |
| **λ student data fraction** | Our model-based/modelless duality (Research 037) is conceptually similar but at routing level, not training data level | Minor — already have `BanditPruner` epsilon-greedy mixing |
| **Self-distillation** (student > teacher) | G-Zero self-play (Plan 049) achieves this organically | Already captured |
| **Data efficiency at small fractions** | Not measured — our benchmarks use full datasets | Minor — game domain has structured data, not natural language |

### Critical Assessment: GKD vs SDAR

SDAR (Research 038) is **strictly more sophisticated** than GKD for our use case:

| Aspect | GKD | SDAR |
|--------|-----|------|
| Token weighting | Uniform or divergence-based | Sigmoid-gated with asymmetric trust |
| Teacher-student gap | Implicit in divergence choice | Explicit gap signal `Δ_t = log πT - log πS` |
| Multi-turn stability | Not addressed (paper is single-turn) | Explicitly solves multi-turn OPSD collapse |
| Negative teacher signal | All tokens weighted equally | Softly attenuated (teacher may be wrong) |
| Skill retrieval | Not considered | Bandit-based skill context for teacher |

**SDAR supersedes GKD for token-level distillation.** GKD's contribution was showing that on-policy data matters; SDAR shows *how* to use on-policy data safely with asymmetric trust.

---

## Game Domain Relevance

### Self-Distillation in Games

GKD shows self-distillation (same architecture teacher and student) works — student can surpass teacher. This validates our G-Zero self-play approach where the model plays against itself and improves organically.

**But:** G-Zero already proves this (Plan 049, T5 benchmarks show modelless self-play improvement). No new insight.

### On-Policy Data for Game AI

Game episodes are inherently on-policy — the student (game AI) generates moves during play. We don't have the "fixed dataset" problem because game data comes from live play or self-play.

**No gain** — games naturally produce on-policy data.

### Divergence Choice for Game LoRA

Could forward KL vs reverse KL matter for game LoRA training? The paper shows:
- Forward KL (mode-covering) → good for diverse outputs (translation, summarization)
- Reverse KL (mode-seeking) → good for focused tasks (instruction following, reasoning)

For games:
- **Bomber**: Mode-seeking is better (want to converge on optimal strategies)
- **Go**: Mode-covering might be better (want diverse strategic exploration)
- **NPC Dialog**: Mode-seeking for quest responses, mode-covering for open dialog

**Assessment:** The divergence choice is interesting but our SDAR gate already handles this implicitly by weighting tokens based on teacher-student gap magnitude. Adding explicit divergence options would add complexity without clear game-domain benefit.

---

## GOAT Pillar Impact

| Pillar | GKD Impact | Verdict |
|--------|-----------|---------|
| Pillar 1 (Fourier Spatial AI) | None — algorithmic, no distillation | ❌ No gain |
| Pillar 2 (WASM Validators) | None — deterministic validators | ❌ No gain |
| Pillar 3 (NPC Dialog Engine) | Marginal — could use JSD for dialog LoRA | ⬜ Indirect — NPC dialog works modelless |
| Pillar 4 (Frame-Sampling Bridge) | None — frame decimation is algorithmic | ❌ No gain |
| LoRA Training (Secret A) | Minor — better divergence choice during distillation | ⬜ Indirect — SDAR already covers this |

**No pillar-level impact.** GKD is a distillation technique improvement, and our existing distillation pipeline (SDAR + ROPD + VPD) already covers its contributions.

---

## Decision Matrix Score

| Criterion | Score | Reason |
|-----------|-------|--------|
| GOAT passable | ❌ | No new measurable capability — SDAR already covers GKD's insights |
| MMO-product | ❌ | No direct MMO feature |
| LoRA-independent | N/A | It's about LoRA training, not modelless inference |
| Defensible | ❌ | ICLR 2024 paper — fully public |
| Secret coverage | ❌ | No new secret |

**Verdict: NO NEW PLAN.** The research insight is already captured in our architecture. No feature gate needed.

---

## What We Should Note

1. **GKD validates SDAR's approach.** The on-policy distillation principle GKD introduced (2023) is exactly what SDAR builds on (2026) — the academic lineage confirms our direction is correct.

2. **Divergence choice is the only novel API surface.** If we ever need explicit forward KL / reverse KL / JSD(β) switching, we can add a `DistillDivergence` enum to our loss functions. But SDAR's sigmoid gate is a strictly better abstraction — it automatically weights tokens rather than switching global divergences.

3. **Self-distillation validation.** GKD's self-distillation results (student > teacher) validate our G-Zero self-play philosophy. This is already noted in Plan 049.

4. **Data efficiency at 5% is impressive** but irrelevant — game domains don't have "5% of data" problems; they have structured episode generation.

---

## References

- Agarwal, R., Vieillard, N., Zhou, Y., et al. "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes." ICLR 2024.
- Related: SDAR (Research 038) — extends GKD with sigmoid-gated asymmetric trust
- Related: ROPD (Research 036) — rubric-based on-policy distillation (no teacher logits needed)
- Related: REAP (Research 037) — model-based/modelless duality

---

> **PASS-Redirects (synthesis):** Blondel, Sander, Vivier-Ardisson, Liu, Roulet [arXiv:2611.16736 "On the Difficulty of Sequence-Level Supervised Learning for Autoregressive Language Models"] (2026-11-12) — proves the decoding-aware sequence-level supervised losses (greedy-decoding and autoregressive-sampling forms) discontinuous/nonconvex and UNBOUNDED BELOW; the stop-grad "moment matching direction" a valid function-space descent direction that is NOT the gradient of any loss (high variance); concludes teacher forcing "may well be all you need" for supervised AR training — naming on-policy distillation (THIS note's subject, Agarwal et al.) and RL-with-external-reward as the only sound routes to sequence-level signal. Class landscape corroborated by Sriraman et al. [arXiv:2606.30923 "Behavior Cloning is Not All You Need: The Optimality of On-Policy Distillation for Noisy Expert Feedback"].
