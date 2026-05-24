# Research 54: Anchored Supervised Fine-Tuning (ASFT)

**Paper**: Anchored Supervised Fine-Tuning — He Zhu et al. (SUSTech / PKU)
**Venue**: ICLR 2026
**arXiv**: 2509.23753v3
**Code**: https://github.com/zhuchichi56/ASFT

---

## TL;DR

ASFT = Dynamic Fine-Tuning (DFT) + KL anchoring against base model. DFT reweights SFT by stop-gradient self-probability to tighten RL bounds, but suffers distributional drift. Adding forward KL regularization stabilizes training while preserving tightness. Achieves RL-comparable performance at SFT-level cost.

---

## Core Mechanism

### DFT Loss (baseline, from Wu et al. 2025)

```
L_DFT(θ) = -E_{(x,y*)~D}[ sg[π_θ(y*|x)] · log π_θ(y*|x) ]
```

- `sg[·]` = stop-gradient (probability treated as constant weight)
- Reweights each sample by model's own probability → focuses on harder examples
- Tighter RL lower bound than standard SFT (proven via covariance argument)
- **Problem**: No anchoring → progressive drift → instability in knowledge tasks

### ASFT Loss (proposed)

```
L_ASFT(θ) = L_DFT(θ) + λ · E_s[ D_KL(π_base(·|s) || π_θ(·|s)) ]
```

- `π_base` = fixed reference (pretrained model, no gradient)
- `λ` = anchoring strength (optimal ~0.05-0.1)
- **Forward KL** (not reverse) → mode-covering behavior → prevents collapse
- KL term provides explicit variance control that prevents exponential weight growth

### ASFT-LoRA Variant

Uses LoRA decomposition `ΔW = BA` to compute KL with single model instance:
- Forward pass with `W_base` → get `π_base` log-probs
- Forward pass with `W_base + BA` → get `π_θ` log-probs
- Compute KL between them → no separate model copy needed
- Memory: 40.70 GB vs 88.02 GB (full ASFT) vs 38.96 GB (SFT) on LLaMA-2-7B
- Performance: 93.9% of full ASFT (39.45 vs 42.03 on medical benchmarks)

---

## Theoretical Framework: Reward-Weighted Regression (RWR)

### SFT as RL Lower Bound

Under sparse reward `R(τ) = I[y = y*]`, the RL objective is:

```
J(θ) = E_{τ~π_ref}[ (π_θ(τ) / π_ref(τ)) · R(τ) ]
```

Using `u ≥ 1 + log(u)` with `u = π_θ(τ)/π_ref(τ)`:

```
J(θ) ≥ c_ref · E_{τ∈D+}[ log π_θ(τ) ] + const
```

This is exactly the SFT objective (up to scaling).

### DFT Tightens via Auxiliary Distribution

DFT corresponds to choosing auxiliary distribution:

```
q(τ) = π_ref(τ|D+) · sg[p_θ(τ)] / E_{τ~π_ref(·|D+)}[sg[p_θ(τ)]]
```

**Theorem 1 (Strict Tightness)**: DFT yields strictly tighter bound than SFT whenever `Var(p_θ(τ)) > 0` on D+.

Proof uses covariance: `B_DFT - B_SFT = c_ref · Cov(X, log(X)) / E[X]` where `X = p_θ(τ)`.
Since `log(x)` is strictly increasing, `Cov(X, log(X)) > 0` for non-degenerate distributions.

### Distributional Drift Problem

As training progresses:
1. `p_θ(τ)` becomes increasingly non-uniform on D+
2. Auxiliary distribution `q` concentrates on high-probability trajectories
3. Importance weights `π_θ(τ)/q_θ(τ)` become high-variance
4. Effective sample size shrinks → optimization destabilizes

The inequality `u ≥ 1 + log(u)` is tight only when `u = 1` (i.e., `π_θ = q_θ`), but DFT's drift makes this increasingly loose.

### KL Anchoring Resolves Drift

The KL regularization creates a trust region around `π_base`:
- Prevents policy from drifting too far from reference
- Maintains effective sample size
- Preserves tightness (KL term doesn't alter the lower-bound structure)

---

## Key Results

### Medical Knowledge (LLaMA-2-7B, 10k samples)

| Method | MedQA | MMLU | MedMCQA | Avg | vs Base |
|--------|-------|------|---------|-----|---------|
| Base | 29.85 | 30.52 | 33.76 | 31.38 | — |
| SFT | 33.31 | 33.52 | 33.28 | 33.37 | +1.99 |
| DFT | 29.69 | 26.69 | 31.20 | 29.19 | **-2.19** (drift!) |
| ASFT | **39.28** | **46.37** | **40.45** | **42.03** | **+10.65** |
| DAPO (RL) | 39.75 | 48.63 | 38.37 | 42.25 | +10.87 |

ASFT ≈ DAPO (RL) at 3% of the compute cost.

### Mathematical Reasoning (Qwen2.5-7B, 100k samples)

| Method | AIME24 | Math500 | Minerva | Olympiad | AMC23 | Avg | vs Base |
|--------|--------|---------|---------|----------|-------|-----|---------|
| Base | 1.65 | 28.79 | 9.26 | 7.69 | 15.65 | 12.61 | — |
| SFT | 0.83 | 47.30 | 13.46 | 14.16 | 20.00 | 19.15 | +6.54 |
| DFT | 6.26 | 56.88 | 21.18 | 22.68 | 27.19 | 26.04 | +13.43 |
| ASFT | **6.66** | **59.99** | **23.55** | **25.57** | **36.72** | **30.50** | **+17.89** |

### ASFT + RL (Continual Training)

| Method | MedQA | MMLU | MedMCQA | Avg |
|--------|-------|------|---------|-----|
| SFT + DAPO | 36.84 | 44.76 | 39.11 | 40.24 |
| **ASFT + DAPO** | **41.32** | **49.54** | **41.45** | **44.10** |

ASFT provides superior initialization for subsequent RL fine-tuning.

### Computational Cost (LLaMA-2-7B, single A100)

| Method | Time (hrs) | Memory (GB) | Accuracy |
|--------|-----------|-------------|----------|
| SFT | 0.524 | 38.96 | 33.04 |
| DFT | 0.521 | 38.90 | 25.97 |
| ASFT | 0.648 | 88.02 | 42.03 |
| ASFT-LoRA | 0.594 | 40.70 | 39.45 |
| GRPO | 51.24 | 483.98 | 32.53 |
| DAPO | 21.60 | 488.26 | 42.25 |

ASFT-LoRA: only 13.4% time overhead over SFT, same memory footprint, +19% accuracy.

### Forward KL vs Reverse KL

- **Forward KL** `D_KL(π_base || π_θ)`: mode-covering → stable, maintains broad distribution
- **Reverse KL** `D_KL(π_θ || π_base)`: mode-seeking → can collapse to few high-probability sequences
- Optimal λ ≈ 0.05-0.1 for forward KL

### Alternative Anchoring: SFT Loss

The paper also explores `L_ASFT-SFT(θ) = L_DFT(θ) + α · L_SFT(θ)`:
- Computationally cheaper (no reference model needed)
- Works for math (moderate α) but volatile for medical
- KL anchoring is more robust across domains
- Suggests domain-adaptive anchoring as future direction

---

## Distillation to Our Architecture

### Model-Based Path (riir-ai)

**Direct applicability**: HIGH

Our `riir-gpu` crate has:
- LoRA training pipeline (`training_loop.rs`, `lora.rs`)
- SDAR loss (`loss_sdar.rs`) — sigmoid-gated teacher-student gap
- GRPO loss (`loss_grpo.rs`) — advantage-weighted policy gradient
- DPO loss (`loss_dpo.rs`) — length-normalized preference optimization
- Loss masking (`LossMask` enum — Observational / Interventional)

ASFT fits as a **new loss variant** alongside SDAR, GRPO, DPO:

```
Feature gate: asft_loss = []
```

**Implementation approach**:
1. CPU loss function (like SDAR) — `loss_asft.rs`
2. ASFT-LoRA variant — compute KL by switching `W_base` / `W_base + BA`
3. Integrate into `TrainingConfig` as optional loss

**Key difference from SDAR**:
| Aspect | SDAR | ASFT |
|--------|------|------|
| Reweight signal | teacher-student gap | self-probability (stop-grad) |
| Anchoring | implicit (teacher) | explicit (KL vs base) |
| Reference model | teacher model | pretrained base |
| Applicable to | distillation (2-model) | single-model fine-tuning |

ASFT and SDAR are **complementary**, not competing:
- ASFT improves single-model fine-tuning (no teacher needed)
- SDAR improves multi-model distillation (teacher guidance)
- Could combine: ASFT reweighting + SDAR gating + KL anchoring

### Modelless Path (katgpt-rs)

**Direct applicability**: LOW (no neural network training)

However, the **theoretical insights** are valuable:

1. **RWR framework validates our reweighting approaches**: Our SDAR gate, GFlowNet flow regularization, and BT ranking all implement forms of importance weighting that tighten RL bounds.

2. **KL anchoring concept for bandits**: Our bandit pruners accumulate Q-values over time and can drift. An "anchored bandit" that regularizes toward initial Q-values could prevent overfitting to recent episodes.

3. **Distributional stability**: The paper's finding that DFT's drift comes from non-uniform probability concentration mirrors our observation that `NoScreeningPruner` baselines diverge over many episodes.

**Indirect distillation** (no implementation needed):
- The `sg[π_θ]` reweighting concept validates our `sdar_gate()` sigmoid mechanism
- Forward KL > Reverse KL finding validates our `KlBoundaryAligner` symmetric KL choice
- ASFT + DAPO > SFT + DAPO finding suggests our LoRA checkpoints should use ASFT-style training before RL

### Comparison with Our Existing Losses

| Our Method | Paper Analog | Relationship |
|------------|-------------|--------------|
| SDAR loss | — | Complementary (teacher-student gap) |
| GRPO loss | GRPO | Same baseline (Table 2) |
| DPO loss | — | Different objective (preference) |
| ROPD rubric | — | Different (multi-criteria scoring) |
| BT ranking | — | Different (pairwise comparison) |
| GFlowNet flow | DFT reweighting | Similar spirit (importance weighting) |
| KL boundary alignment | KL anchoring | Same mechanism, different context |
| Interventional SFT | — | Different (token masking) |

---

## Verdict

### ✅ ADOPT for riir-ai (model-based)

**Why**:
- ASFT-LoRA is a drop-in improvement over SFT with minimal overhead
- Theoretical justification is rigorous (RWR framework, tightness proof)
- Empirical results are strong and consistent across domains
- ICLR 2026 acceptance validates quality
- Our existing LoRA pipeline makes implementation straightforward

**Implementation scope**:
- New `loss_asft.rs` module (~200 lines CPU, similar to SDAR)
- `asft_loss` feature gate (zero dependencies)
- Integration into `TrainingConfig` and training loop
- GOAT proof: ASFT vs SFT in LoRA arena (Bomber/Go domains)

### ⏸ HOLD for katgpt-rs (modelless)

**Why**:
- No neural network training in modelless path
- DFT reweighting requires `π_θ` which doesn't exist without a model
- KL divergence requires probability distributions over actions
- Conceptual insights noted but no direct implementation target

### ⚠️ Caveats

1. **Memory overhead**: Full ASFT requires 2× model memory (88 GB for 7B). ASFT-LoRA mitigates this.
2. **Forward pass cost**: Need two forward passes (base + fine-tuned) per training step. ~23.7% overhead.
3. **λ sensitivity**: Optimal λ varies by domain (0.05 for medical, 0.1 for math). Needs tuning.
4. **Base model quality matters**: DFT's effectiveness depends on base model priors. Weak base → weak ASFT.
5. **No multi-turn evaluation**: Paper only tests single-turn tasks. Multi-turn (our Bomber/Go domains) is untested.
6. **Precision sensitivity**: kl_weight=0.03 optimal for bf16 (not 0.05 from paper). Larger KL weights amplify bf16 precision noise and degrade accuracy.
7. **DeepSpeed instability**: Native (non-DeepSpeed) training more stable than Zero-2/Zero-3. FSDP2 (verl branch) recommended for distributed training.

---

## Implementation Plan Reference

See: `riir-ai/.plans/090_asft_anchored_lora.md`

Tasks:
1. `loss_asft.rs` — CPU ASFT loss with DFT reweighting + KL anchoring
2. ASFT-LoRA variant — single-model KL computation via weight switching
3. `asft_loss` feature gate in `riir-gpu/Cargo.toml`
4. Training loop integration
5. GOAT proof: ASFT vs SFT in LoRA arena

---

## Key Formulas Reference

```
// DFT reweighting weight (stop-gradient)
w_dft = sg[π_θ(y*|x)]

// Forward KL divergence
KL_forward = Σ π_base(t|s) · (log π_base(t|s) - log π_θ(t|s))

// ASFT loss
L_ASFT = -E[sg[π_θ] · log π_θ] + λ · E_s[KL(π_base || π_θ)]

// ASFT-LoRA: single-model KL
// Forward with W_base → log_probs_base
// Forward with W_base + BA → log_probs_θ
// KL = Σ exp(log_probs_base) · (log_probs_base - log_probs_θ)

// Default: λ = 0.03 for bf16 (0.05 for fp32). bf16 precision noise amplifies with larger KL weights.
```

---

## Reference Code Analysis

### KL Computation Detail
The reference implementation (`train_v2.py`) computes **full-vocab forward KL**:
```python
kl_div = F.kl_div(
    F.log_softmax(shift_logits, dim=-1),  # student log-probs (full vocab)
    F.softmax(ref_logits, dim=-1),         # base probs (full vocab)
    reduction="none",
).sum(dim=-1)
```
This is a sum over the **entire vocabulary** at every token position — NOT a greedy single-sample estimate. The `F.kl_div` with `log_softmax` input computes `exp(ref) * (ref - log_softmax(student))` which is exactly `D_KL(π_base || π_θ)`.

### KL Applied to ALL Tokens (Not Just Response)
The KL divergence is computed on all token positions including prompt tokens. Only the DFT weight uses `valid_mask` (response-only). The final aggregation applies `valid_mask` to the combined loss:
```python
loss = (weighted_losses[valid_mask].sum() / valid_mask.sum())
```
Where `weighted_losses = dft_losses + kl_weight * kl_div`. So the KL term contributes to positions outside the mask in `weighted_losses`, but only positions inside the mask contribute to the final loss. This means KL effectively acts as a regularizer on response tokens only (via the mask), but is computed using full-sequence information.

Wait, actually looking more carefully: `kl_div` is computed on ALL shifted tokens (shape `[total_tokens]`), not just `valid_mask` tokens. Then `weighted_losses = dft_losses + kl_weight * kl_div` and then `loss = weighted_losses[valid_mask].sum() / valid_mask.sum()`. So KL IS masked to response tokens in the final aggregation. The difference is subtle but correct: both DFT and KL are computed on all tokens, then masked.

Actually re-reading more carefully:
- `shift_logits` and `shift_labels` are flattened to `[total_tokens, vocab]` and `[total_tokens]`
- `token_losses` = CE loss on ALL tokens (including padded/ignored ones with label=IGNORE_INDEX)
- `valid_mask = shift_labels != IGNORE_INDEX` selects only response tokens
- `kl_div` = KL on ALL shifted token positions
- `weighted_losses = dft_losses + kl_weight * kl_div` combines on ALL positions
- `loss = weighted_losses[valid_mask].sum() / valid_mask.sum()` masks to response only

So effectively: **both DFT and KL are computed per-token, then masked to response tokens for the final loss.**

### LoRA disable_adapter Pattern
For ASFT-LoRA, the reference uses `model.disable_adapter()` context manager:
```python
if hasattr(model, "disable_adapter"):
    with model.disable_adapter():
        ref_outputs = model(**inputs)
        ref_logits = ref_outputs.logits
else:
    with torch.no_grad():
        ref_outputs = self.original_model(**inputs)
        ref_logits = ref_outputs.logits
```
This confirms single-model weight switching. No separate model copy needed for LoRA.

### Precision Sensitivity (from DEV.md)
Critical finding from dev experiments:
- **bf16 always more stable than fp16** — fp16 requires careful loss scale settings
- **DeepSpeed Zero (2/3) less stable** than native runs
- **kl_weight=0.03 best for bf16** (not 0.05 from paper's fp32 experiments)
  - kl=0.03: medqa 0.3920, mmlu 0.4575, medmcqa 0.3952
  - kl=0.05: (baseline bf16) medqa 0.3833, mmlu 0.4560, medmcqa 0.3992
  - kl=0.07: medqa 0.3417, mmlu 0.4074, medmcqa 0.3787
  - kl=0.025: medqa 0.3401, mmlu 0.3640, medmcqa 0.3648
- **Math: lr=2e-5 consistently better than 5e-5** across all modes (SFT/DFT/ASFT)
- ASFT merged into LLaMA-Factory v0.9.4 (2026-02-12)
- verl branch (FSDP2) recommended over DeepSpeed for numerical stability

### DFT Weight Computation Detail
```python
probs = torch.softmax(shift_logits, dim=-1)
valid_labels = torch.clamp(shift_labels, min=0, max=probs.size(-1)-1)
weights = probs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1).detach()
dft_losses = token_losses * weights
```
Key: `probs.gather(1, target_ids)` extracts the model's own probability for the target token, then `.detach()` applies stop-gradient. The `.clamp(min=0)` handles IGNORE_INDEX=-100 by clamping to valid token range before gathering (these positions are masked out later anyway).

---

## References

- ASFT: He Zhu et al., "Anchored Supervised Fine-Tuning", ICLR 2026, arXiv:2509.23753
- DFT: Yongliang Wu et al., "On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification", arXiv:2508.05629
- RWR: Peters & Schaal, "Reinforcement Learning by Reward-Weighted Regression", ICML 2007
- Our SDAR: `.research/038_SDAR_Self_Distilled_Agentic_RL.md`
- Our ROPD: `.research/036_ROPD_Rubric_OnPolicy_Distillation.md`
- Our KL alignment: `katgpt-rs/src/pruners/boundary_alignment.rs`
- Our SDAR gate: `katgpt-rs/src/pruners/sdar_gate.rs`
