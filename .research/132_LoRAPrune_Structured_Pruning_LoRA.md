# Research 132: LoRAPrune — Structured Pruning Meets Low-Rank PEFT

**Paper**: LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning
**Authors**: Mingyang Zhang, Hao Chen, Chunhua Shen et al. (Zhejiang University)
**arXiv**: 2305.18403v5
**Venue**: Preprint (Aug 2024)

---

## TL;DR

LoRAPrune prunes attention heads and FFN channels using only LoRA gradient Taylor expansion — no pre-trained weight gradients needed. 52.6% memory savings vs vanilla gradient pruning. At 50% compression, perplexity 11.60 vs LLM-Pruner 16.41 on WikiText2 (LLaMA-7B).

**Verdict: NO GAIN** — Scale mismatch. Our models (micro: 5K–524K params, game LoRA rank 4) have nothing meaningful to structurally prune. PlasmaPath already delivers 20× memory compression at inference time. LoRAPrune is a training-time technique for 7B–65B models; our training pipeline (riir-gpu/Metal) runs tiny game adapters. If we ever deploy 7B+ to edge devices, LoRAPrune + PlasmaPath could compound, but that's a future decision.

> **Browser context (1–2 GB WebGPU budget):** Our game models are 3–4 KB with PlasmaPath. AI total budget is ~2 MB for 100 concurrent NPCs. LoRAPrune saves bytes on 7B+ models; at our scale it saves hundreds of bytes — not worth the training pipeline complexity. SPEFT (riir-ai Research 022) is the right answer for our param budget: same quality gain, simpler implementation, naturally compact for browser.

---

## Core Mechanism

### LoRA-Guided Pruning Criterion

Vanilla Taylor expansion pruning requires `∂L/∂W₀` (pre-trained weight gradients) — expensive for LLMs:

```
Îᵢⱼ = (∂L/∂Wᵢⱼ · Wᵢⱼ)²
```

LoRAPrune approximates this using only LoRA's A, B matrices and their gradients:

```
Îᵢⱼ = [(∂L/∂Bᵢ₎ · A₎ⱼ + Bᵢ₎ · ∂L/∂A₎ⱼ − ∂L/∂Bᵢ₎ · ∂L/∂A₎ⱼ) · (Wᵢⱼ + (BA)ᵢⱼ)]²
```

Key derivation step — using SGD weight update to approximate `∂L/∂(BA)`:

```
∂L/∂(BA)ᵢⱼ ≈ ∂L/∂Bᵢ₎ · A₎ⱼ + Bᵢ₎ · ∂L/∂A₎ⱼ − ∂L/∂Bᵢ₎ · ∂L/∂A₎ⱼ  (η=1 simplification)
```

### Progressive Pruning

1. Compute group importance Ĝ per batch via Eq. above + dependency-aware aggregation
2. Update moving average: `Ḡ|ₜ = λḠ|ₜ₋₁ + (1−λ)Ĝ|ₜ` (λ controls history vs current balance)
3. Every N iterations, prune bottom-k groups (heads in attention, channels in FFN)
4. Continue fine-tuning to recover performance

### Results (LLaMA-7B, 50% compression)

| Method | WikiText2 ↓ | PTB ↓ | GPU Memory | Avg Accuracy ↑ |
|--------|-------------|-------|------------|----------------|
| LLM-Pruner | 16.41 | 20.85 | 38.6 GB | 51.90% |
| **LoRAPrune** | **11.60** | **17.39** | **18.3 GB** | **54.81%** |
| LoRAPrune-8bit | 11.65 | 17.41 | 13.8 GB | 54.55% |

---

## Distillation to Our Architecture

### Dimension 1: Compression Pipeline Overlap

| Our Technique | Target | When | Savings |
|---------------|--------|------|---------|
| PlasmaPath (ternary) | Weights → {-1,0,+1} | Inference | 20× memory |
| SpectralQuant | KV cache eigenbasis | Inference | 9.1× compression |
| OCTOPUS | KV cache triplet | Inference | Data-oblivious |
| Asymmetric KV | K/V different bit widths | Inference | Key 8-bit, Val 3-bit |
| QLoRA (riir-ai) | 4-bit base + LoRA | Training | 4× base memory |
| **LoRAPrune** | **Remove heads/channels** | **Training** | **Up to 50%** |

PlasmaPath compresses *all* weights; LoRAPrune *removes* entire structures. They compose: prune → ternary quantize. But for micro-models (3 layers, 48 dim), there's nothing to structurally prune.

### Dimension 2: Model-Based vs Modelless

LoRAPrune sits on the model-based side:
- Requires forward + backward passes through LoRA
- Iterative training loop with pruning checkpoints
- Group importance is a gradient-computed signal (like our `DeltaGatedAbsorb`)

Our spectrum already covers this:

```
Modelless:  ConstraintPruner → BanditPruner → FlowPruner
Bridge:     DeltaBanditPruner → DeltaGatedAbsorb
Model-based: G-Zero Phase 2 (GRPO/DPO) → ASFT → SDAR → VPD
```

LoRAPrune would be another model-based technique in the riir-gpu training pipeline — but it solves a problem (model too large to deploy) that our micro-models don't have.

### Dimension 3: PlasmaPath Connection

PlasmaPath (Plan 148) encodes weights as ternary {-1, 0, +1} with bit-plane SIMD. The "implicit zero-skip" (both bits zero = weight is 0) is structurally similar to pruning — zero weights cost nothing in the ternary matvec.

**Key insight:** PlasmaPath already achieves the *effect* of pruning through quantization — weights that are "unimportant enough" naturally quantize to 0. This is data-dependent implicit pruning at inference time, without any structured removal.

At 0.77 cosine similarity on random weights (PlasmaPath), the quantization naturally zeros out ~33% of weights. On real NN weights (expected ≥ 0.92 cosine), the zero fraction may differ, but the principle holds: ternary quantization is a form of automatic unstructured pruning.

LoRAPrune's structured pruning (removing entire heads) is a *different* operation — it targets deployable structured sparsity that hardware can skip. But PlasmaPath's bit-plane encoding achieves the same hardware benefit (zero-skip is implicit in the SIMD loop).

### Dimension 4: MMO Goat Pillars Alignment

Per `27_mmo_goat_pillars_decision_matrix.md`:

| Pillar | LoRAPrune Relevance |
|--------|-------------------|
| Pillar 1 (Fourier Spatial AI) | ❌ Algorithmic, no LoRA needed |
| Pillar 2 (WASM Validators) | ❌ Deterministic, no neural net |
| Pillar 3 (NPC Dialog Engine) | ⬜ Modelless baseline sufficient. LoRA is optional. |
| Pillar 4 (Frame-Sampling Bridge) | ❌ Algorithmic |

LoRAPrune doesn't strengthen any pillar. It's a generic LLM compression technique, not a game-specific innovation.

### Dimension 5: optimization.md Compliance

| Criterion | Assessment |
|-----------|-----------|
| Profile first | ❌ No profiling shows model size is a bottleneck |
| Identify top 3 bottlenecks | ❌ Not in top 3 — PlasmaPath already handles memory |
| Measure after each change | ❌ Nothing to measure — no existing pruning bottleneck |
| Binary bloat | ⚠️ Training-time only, no runtime feature gate needed |
| Don't optimize without numbers | ❌ Premature for our model sizes |

---

## Why Skip

1. **Scale mismatch.** Our game LoRA adapters have rank=4, dim=48–96, 3 layers. You can't structurally prune a 3-head, 3-layer model meaningfully. The paper targets 64-head, 80-layer models.

2. **Already covered.** PlasmaPath = 20× memory savings at inference. QLoRA = 4× at training. The "compress model" value proposition is saturated.

3. **Training-only.** katgpt-rs is inference-only. LoRAPrune lives in riir-gpu. Even there, our Metal training runs micro-models where iterative pruning overhead exceeds training time.

4. **Compounds with PlasmaPath but not worth it yet.** LoRAPrune → remove heads → PlasmaPath ternary quantize what remains = maximum compression. This pipeline makes sense for 7B+ edge deployment. Not today.

5. **Not a GOAT pillar.** Doesn't contribute to any of the 4 MMO pillars. Not defensible (public paper). Not game-specific.

---

## Future Trigger

If riir-ai ever trains **7B+ models for code translation** and needs **edge deployment** where even PlasmaPath-compressed 7B is too large, re-evaluate:

```
QLoRA train → LoRAPrune structured prune → PlasmaPath ternary quantize → deploy
```

This compounds: 50% structure removal × 20× ternary compression = ~40× total. But only if 7B inference on edge devices becomes a product requirement.

---

## Cross-References

- PlasmaPath: `.benchmarks/044_plasma_path_goat.md`, Plan 148, Research 110
- QLoRA/IA3: riir-ai Plan 071
- ASFT LoRA training: Research 054, riir-ai Plan 090
- SpectralQuant: Research 039, Plan 077
- OCTOPUS: Research 063, Plan 099
- Model-based/modelless spectrum: Research 037 (REAP)
- MMO Pillars: riir-ai `.docs/27_mmo_goat_pillars_decision_matrix.md`
- optimization.md: `.contexts/optimization.md`
