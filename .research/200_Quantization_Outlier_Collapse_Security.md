# Research 200: Outlier-Induced Weight Collapse — Security & Inversion

**Paper:** "Widening the Gap: Exploiting LLM Quantization via Outlier Injection" (Zhan et al., ETH Zurich, 2026)
**arXiv:** 2605.15152
**Date:** 2026-06
**Status:** GOAT Verdict Pending
**Context:** katgpt-rs (modelless) + riir-ai (model-based)

---

## TL;DR

The paper introduces an attack where injecting large outliers into weight matrices causes quantization to round all non-outlier weights to zero, creating a deterministic behavior switch between full-precision and quantized models. **We invert this**: instead of attacking, we use the same mechanism for (1) modelless outlier detection/defense and (2) structured sparsity for faster inference. The KS statistic from the paper is a free O(n) detector. The weight collapse pattern is a free structured pruning signal.

**Verdict: ADOPT (modelless defense) + CONDITIONAL (model-based training guard)**

- **katgpt-rs (MIT):** Outlier detection guard at inference time — zero-cost KS statistic check, synergistic with Plan 138 (StiffSoft anomaly gate). ON BY DEFAULT after GOAT proof.
- **riir-ai (private):** LoRA training-time outlier guard + multi-precision behavior switching for game AI. Feature-gated.

---

## Paper Summary

### The Attack

1. **Zero-init a switching block** (one FFN layer), redistributing its function to other layers via fine-tuning
2. **Inject outliers** (1 per quantization group of 32 weights) with large magnitude scaling factor c
3. **Dual-objective fine-tuning**: train benign behavior in full-precision (switching block active) and malicious behavior for quantized (switching block zeroed)
4. **Refinement**: use a quantized proxy to strengthen post-quantization behavior

### Key Properties

- **Attack success rate: >90%** across GPTQ, AWQ, GGUF, NF4, FP4, HQQ, SINQ, AutoRound
- **Utility preservation: >90%** of original model benchmarks
- **Group size transfer**: attacks with attacker group_size=32 transfer to user group_size=64,128 (but not vice versa)
- **Calibration independence**: attack works regardless of user's calibration dataset
- **Layer selection**: middle layers (15-25) work best; first few destroy utility, last few reduce ASR
- **Weight matrix**: up_proj and gate_proj in FFN are most reliable targets

### The Defense (Paper's Finding)

- **KS D-statistic** detects the attack reliably: the attacked layer shows a massive spike in distributional distance
- **Gaussian noise defense FAILS**: noise that reduces ASR also destroys utility
- **Rescaling works at low c but fails at high c**: the defense arms race favors the attacker at higher scaling factors

---

## Fusion Ideas (Not Direct Mapping)

### Fusion 1: Outlier-Aware Quantization Guard (OAQG) — MODELLESS

**Insight:** The KS D-statistic is a free O(n) per-layer check that detects outlier injection. We don't need the attack — we need the detector.

**Application:**
- At model loading time, compute KS D-statistic between each layer's weight distribution and a Gaussian reference
- Flag layers with D-statistic > threshold (paper shows clear spike at attacked layer)
- Integrate with existing `StiffSoftDecomposition` (Plan 138) — outlier layers will show anomalous eigenvalue distributions (one dimension dominating)
- Zero-cost: the check runs once at load time, not in the hot path

**Synergy with Plan 138 (StiffSoft Anomaly Gate):**
- Plan 138 detects anomalies via eigenvalue decomposition of the Jacobian
- KS statistic detects anomalies via distributional distance of raw weights
- These are orthogonal signals: eigenvalue anomaly catches semantic drift, KS catches structural tampering
- Combined: if either fires, flag the layer. If both fire, high confidence of compromise.

**Commercial alignment (Verdict 003):**
- Detection logic stays in MIT engine (katgpt-rs) — everyone benefits from secure inference
- Remediation (auto-fix, rescaling) could be SaaS feature in riir-ai
- Fits "engine = open, intelligence = closed" split

### Fusion 2: Collapse-as-Structured-Pruning (CASP) — MODELLESS

**Insight:** The weight collapse pattern (outlier → all non-outliers round to zero) is a deterministic, predictable structured sparsity map. Instead of using it to attack, use it to PRUNE.

**Application:**
- For models we control (our trained LoRA adapters), intentionally place outliers to create known sparsity patterns
- At inference time, skip the zeroed weights entirely — structured sparse matmul
- This is NOT the same as the TwELL sparse MLP (Plan 022): TwELL exploits natural ReLU sparsity at runtime; CASP pre-computes sparsity at quantization time
- Combines with existing `hybrid_oct_pq` (current GOAT quantization): after OCT encoding, apply structured pruning via outlier-induced collapse

**Connection to existing work:**
- Plan 022 (TwELL Sparse MLP): runtime sparsity from ReLU activations
- CASP: compile-time sparsity from outlier-induced weight collapse
- These are complementary: TwELL skips zero activations, CASP skips zero weights

### Fusion 3: Multi-Precision Behavior Switching (MPBS) — MODEL-BASED (riir-ai)

**Insight:** The dual-behavior concept (benign in full-precision, different in quantized) can be used POSITIVELY for game AI.

**Application for game NPC behavior:**
- **Full precision (BF16):** NPC in "dream" mode — latent reasoning, planning, emotion computation
- **4-bit quantized:** NPC in "react" mode — fast reflexes, simple behavior
- **8-bit quantized:** NPC in "balanced" mode — moderate reasoning with speed

This maps directly to the Two-Brain Model from the user's custom instructions:
- **Info brain** (raw, synced) ↔ Full precision — exact values, deterministic replay
- **Think brain** (latent, local) ↔ Quantized — compressed representations, faster but lossy

**Training approach (LoRA only):**
1. Train LoRA adapter with dual-objective: dream behavior in BF16, react behavior in 4-bit
2. Use the paper's refinement fine-tuning with quantized proxy
3. At inference time, switch precision per-NPC based on scene complexity

**Connection to existing plans:**
- Plan 194 (Adaptive CoT): the thinking controller decides per-query
- MPBS adds: the precision controller decides per-NPC per-scene
- Plan 212 (Collapse-Aware Thinking): detects when reasoning is collapsing
- MPBS adds: intentionally collapses reasoning for speed when appropriate

### Fusion 4: LoRA Outlier Guard (LOG) — MODEL-BASED (riir-ai)

**Insight:** Since riir-ai trains LoRA adapters, we need to ensure the training process doesn't accidentally create outlier-vulnerable adapters.

**Application:**
- During LoRA training, monitor weight distribution of the adapter for outlier patterns
- Use KS D-statistic as a training regularizer: penalize weight distributions that deviate too far from Gaussian
- This prevents accidental outlier injection during training (which could happen with aggressive learning rates on LoRA B-matrix)
- Connects to existing gradient monitoring in training loop

---

## What We Take

| Idea | Target | Type | Status |
|------|--------|------|--------|
| KS D-statistic outlier detection | katgpt-rs | Modelless defense | ✅ ADOPT — O(n) per layer at load time |
| Outlier-aware quantization config | katgpt-rs | Modelless infra | ⚠️ CONDITIONAL — only if we accept external models |
| Structured pruning via collapse | katgpt-rs | Modelless inference | ⚠️ CONDITIONAL — needs benchmarks vs existing sparse paths |
| Multi-precision NPC switching | riir-ai | Model-based (LoRA) | ⚠️ CONDITIONAL — creative but needs game validation |
| LoRA training outlier guard | riir-ai | Model-based (LoRA) | ✅ ADOPT — defensive, zero-cost check |

## What We Don't Take

| Idea | Why Not |
|------|---------|
| The actual attack algorithm | We build secure inference, not attacks |
| Gaussian noise defense | Paper proves it doesn't work |
| Weight rescaling defense | Only works at low scaling factors |
| Full-precision model distribution | Our models are always quantized for deployment |

---

## GOAT Gate Analysis

### OAQG (Outlier-Aware Quantization Guard) — katgpt-rs

| Criterion | Assessment |
|-----------|-----------|
| **Perf impact** | Zero — one-time check at model load, O(n) per layer, not in hot path |
| **Gain** | Security — detect compromised models before inference. No accuracy gain. |
| **Complexity** | Low — KS D-statistic is ~20 lines of code |
| **Synergy** | High — composes with Plan 138 StiffSoft gate, uses existing eigenvalue infra |
| **Default-on?** | YES after GOAT proof — zero perf hurt, meaningful security gain |

### LOG (LoRA Outlier Guard) — riir-ai

| Criterion | Assessment |
|-----------|-----------|
| **Perf impact** | Negligible — one check per training step, O(group_size) |
| **Gain** | Training safety — prevent outlier-vulnerable adapters |
| **Complexity** | Low — KS check on LoRA weight deltas |
| **Default-on?** | YES after GOAT proof — defensive, zero training perf hurt |

---

## Key Numbers (From Paper)

| Metric | Value |
|--------|-------|
| Attack ASR (jailbreak, GPTQ 4-bit) | 95.7% |
| Attack ASR (jailbreak, AWQ 4-bit) | 95.0% |
| Utility preservation (relative) | 93-99% |
| KS D-statistic spike at attacked layer | >0.25 (vs <0.1 normal) |
| Outlier scaling factor range | 2^8 to 2^13 |
| Training time | ~2 hours |
| Group size transfer | Small→Large works, not vice versa |
| Detection: KS D-statistic | Clear spike, easily distinguishable |

---

## Cross-Research Alignment

| Existing Research | Relationship | What We Add |
|---|---|---|
| **Plan 138 (StiffSoft Anomaly)** | Orthogonal signal | KS detects weight tampering; StiffSoft detects semantic drift. Combine for higher confidence. |
| **Plan 061 (Entropy Anomaly)** | Different granularity | Entropy is session-level; KS is model-level. Complementary. |
| **Plan 022 (TwELL Sparse MLP)** | Complementary sparsity | TwELL: runtime activation sparsity. CASP: compile-time weight sparsity. |
| **Plan 077 (SpectralQuant)** | Shared eigenbasis | KS anomaly uses weight distributions; SpectralQuant uses eigenvalue distributions. Same math, different signals. |
| **Plan 212 (Collapse-Aware Thinking)** | Conceptual overlap | 212 detects reasoning collapse during CoT. Paper exploits weight collapse during quantization. Different collapses, similar early-exit principle. |
| **Verdict 003 (Commercial Strategy)** | Defense stays open | Outlier detection in MIT engine. Remediation in SaaS. Perfect engine/fuel split. |

---

## TL;DR

The paper proves that quantization is a security surface. We take the defense insight (KS statistic detection) and invert the attack insight (outlier-induced collapse) into structured pruning. Both are modelless. Both compose with existing anomaly detection infrastructure. The model-based extension (multi-precision NPC switching) is creative but needs validation. **Adopt defense (OAQG), conditionally adopt pruning (CASP) after benchmarks.**
