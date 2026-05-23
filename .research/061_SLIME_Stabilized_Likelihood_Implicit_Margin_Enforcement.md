# Research 61: SLIME — Stabilized Likelihood Implicit Margin Enforcement

> **Paper:** [SLIME: Stabilized Likelihood Implicit Margin Enforcement for Preference Optimization](https://arxiv.org/abs/2602.02383) — Afanasyev & Iov, ICML 2026
> **Framework:** [Slime RL Training Framework](https://github.com/THUDM/slime) — Z.ai (THUDM), the engine behind GLM-5.1, GLM-5, GLM-4.7
> **Date:** 2026-02, distilled 2026-05
> **Related Research:** 38 (SDAR), 54 (ASFT), 40 (BT Ranking), 37 (REAP Model-Based/Modelless)
> **Related Plans:** 098 (SLIME Loss Integration)

---

## TL;DR

**Z.ai's tweet** announces their Slime RL framework — the high-performance training engine (Megatron + SGLang) behind GLM-5 series. Simultaneously, the **SLIME paper** (arxiv:2602.02383) provides a reference-free preference optimization objective that solves "unlearning" in margin-based alignment via three-pronged loss: likelihood anchoring, token-level stabilization, and dual-margin optimization.

**Key insight for us:** SLIME fills a gap between our existing DPO (pure margin) and ASFT (pure anchoring) — it provides **reference-free** preference alignment with **explicit chosen-sequence preservation** and **rejected-sequence fluency protection**. This is the missing piece in our loss stack: DPO needs a reference model, SimPO has no anchoring, ASFT is single-model SFT. SLIME gives us reference-free preference optimization that doesn't degrade generation quality.

---

## Part 1: Slime RL Framework (Architecture Insights)

### Overview

Slime is Z.ai's production RL training framework powering GLM-5.1, GLM-5, GLM-4.7, GLM-4.6, GLM-4.5. Architecture: **Megatron (training) + SGLang (rollout) + Data Buffer (bridge)**.

### Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Training    │────▶│  Data Buffer │◀────│  Rollout     │
│  (Megatron)  │     │  (Bridge)    │     │  (SGLang)    │
└──────────────┘     └──────────────┘     └──────────────┘
       │                                          │
       └──────── Weight Sync (async) ◀────────────┘
```

### Key Design Patterns

1. **Async Training-Rollout Decoupling**: Training reads from buffer, rollout writes to buffer. Weight sync is asynchronous — rollout uses slightly stale weights, training never blocks on generation.

2. **Flexible Data Generation Interface**: Custom data generation workflows through server-based engines. Supports GRPO, DPO, SFT, and custom reward/verifier outputs.

3. **Built-on Projects**: Multiple research projects build atop Slime — APRIL (active partial rollouts, +22.5% throughput), Relax (omni-modal agentic RL), RLVE (verifiable environments), ArenaRL (tournament ranking + MCP).

### What We Already Capture

| Slime Pattern | Our Equivalent | Status |
|---|---|---|
| Async training-rollout | `GZeroLoop` rounds (propose → score → update) | ✅ Captured |
| Data buffer bridge | `RoundMetrics` + `TrialLog` serialization | ✅ Captured |
| SGLang rollout server | `Proposer` trait + template/neural variants | ✅ Captured |
| Megatron training | `riir-gpu` wgpu LoRA training stack | ✅ Captured |
| GRPO loss | `loss_grpo.rs` with GSPO/CISPO variants | ✅ Captured |
| DPO loss | `loss_dpo.rs` with length normalization | ✅ Captured |
| Multi-model support | Config-driven model selection | ✅ Captured |

### Verdict on Slime Framework

**CONCEPTUAL ALIGNMENT** — Our architecture already captures the core patterns. No architectural changes needed. The framework validates our approach of decoupled training/rollout with async weight sync.

---

## Part 2: SLIME Preference Optimization (Paper Deep Dive)

### Problem: Margin-Based Unlearning

Current preference optimization methods (DPO, SimPO) suffer from **objective mismatch**:

```
Objective: maximize Δ = log π(y_w|x) - log π(y_l|x)
Problem:   model can lower BOTH, provided y_w > y_l
Result:    "unlearning" — degrades chosen response quality
           "formatting collapse" — suppresses valid linguistic patterns
```

**Evidence from paper:** SimPO on Llama3.2-3B degrades below SFT baseline (MT-Bench 4.22 vs 4.56). On Gemma3-4B Arena-Hard, SimPO collapses to 0.7 vs SFT baseline 7.6.

### Solution: Three-Pronged Objective

```
L(θ) = L_w(θ) + L_l(θ) + L_dist(θ)
```

#### Component 1: Likelihood Anchoring (Chosen)

```rust
// Prevents chosen response degradation — explicit SFT-like supervision
L_w(θ) = -λ_w · E[log π_θ(y_w|x)]
```

- Acts as positive reinforcement signal
- Counteracts "unlearning" by explicitly maximizing chosen likelihood
- `λ_w = 0.1` (paper-validated)

#### Component 2: Token-Level Stabilization (Rejected)

```rust
// Prevents probability collapse of rejected tokens
// Uses softplus-based penalty with threshold δ
L_l(θ) = λ_l · E_{t∈y_l}[softplus(-log π_θ(t|x) - δ)^p]
```

- **Magnitude control**: softplus grows super-linearly for low-probability tokens
- **Gating mechanism**: sigmoid shuts off when probability is "safe"
- Rejects often contain valid syntax/reasoning — don't destroy them
- `δ = 1.25`, `p = 2.5`, `λ_l = 0.1` (paper-validated)

#### Component 3: Dual-Margin Optimization

```rust
// Hard margin: strict cutoff (zero gradient when satisfied)
ℓ_hard = max(0, -Δ + m_h)

// Soft margin: gradient shaping (sigmoid gate)
ℓ_soft = σ(-κ(Δ - m_s))

// Combined: hard × soft — complementary gating
L_dist = λ_d · E[ℓ_hard · ℓ_soft]
```

- `m_h = 1.5` (hard margin — "victory condition")
- `m_s = 1.0` (soft margin — gradient concentration)
- `κ = 2.5` (sigmoid sharpness)
- `λ_d = 1.0` (distance weight)

### Gradient Dynamics (from Appendix)

**L_w gradient**: Constant `∂L/∂l̄_w = -λ_w` — uniform SFT-like push.

**L_l gradient**: Feedback loop —
- Magnitude control: softplus amplifies penalty for collapsing tokens
- Gating: sigmoid shuts off when log-prob exceeds threshold

**L_dist gradient**: Two-way optimization —
- Direct margin expansion via sigmoid modulation
- Boundary sensitivity boost near decision boundary

### Key Results

| Model | Method | MT-Bench | Arena-Hard | vs SFT |
|-------|--------|----------|------------|--------|
| Llama3.2-3B | SFT | 4.56 | 3.8 | baseline |
| Llama3.2-3B | DPO | 4.92 | 5.8 | +0.36 |
| Llama3.2-3B | SimPO | 4.22 | 1.5 | **-0.34** (unlearning!) |
| Llama3.2-3B | **SLIME** | **5.49** | **7.5** | **+0.93** |
| Gemma3-4B | SFT | 4.71 | 7.6 | baseline |
| Gemma3-4B | DPO | 5.15 | 8.3 | +0.44 |
| Gemma3-4B | SimPO | 5.03 | 0.7 | **-3.98** (collapse!) |
| Gemma3-4B | **SLIME** | **6.15** | **9.2** | **+1.44** |
| Qwen3-4B | SFT | 5.40 | 33.9 | baseline |
| Qwen3-4B | **SLIME** | 5.35 | **39.8** | **+5.9** (Arena-Hard) |

**Key observation:** SLIME avoids SimPO's catastrophic collapse. SimPO on Gemma3 Arena-Hard: 7.6 → 0.7. SLIME: 7.6 → 9.2.

### Ablation Summary

| Variant | MT-Bench | Δ vs Full |
|---------|----------|-----------|
| Full SLIME | **6.15** | baseline |
| w/o chosen term | 5.74 | -0.41 |
| w/o rejected term | 5.74 | -0.41 |
| w/o soft margin | 5.80 | -0.35 |
| w/o hard margin | 5.90 | -0.25 |

**Each component contributes meaningfully.** Chosen anchoring and rejected stabilization are equally important.

---

## Cross-Reference: Comparison with Our Existing Losses

| Our Method | SLIME Analog | Relationship |
|------------|-------------|--------------|
| DPO (`loss_dpo.rs`) | DPO (baseline) | SLIME extends DPO — adds anchoring + stabilization + dual margin |
| SDAR (`loss_sdar.rs`) | L_l (rejected term) | Both use token-level mechanisms. SDAR: sigmoid-gated teacher gap. SLIME: softplus penalty on low-prob tokens |
| ASFT (`loss_asft.rs`) | L_w (chosen term) | Both anchor chosen likelihood. ASFT: KL vs base model. SLIME: explicit log-prob maximization |
| GRPO (`loss_grpo.rs`) | — | Orthogonal. GRPO = online policy gradient. SLIME = offline preference |
| BT ranking (`bt_rank.rs`) | L_dist (dual margin) | Related spirit. BT: pairwise ranking. SLIME: pairwise margin with hard/soft gates |
| ROPD rubric | — | Different (multi-criteria scoring) |

### SLIME vs SDAR vs ASFT — Positioning

```
           Reference Model?
           Yes          No
         ┌──────────┬──────────┐
  SFT    │  ASFT    │  DFT     │  ← Single response
         │ (KL anc) │ (reweight)│
         ├──────────┼──────────┤
  Pref   │  DPO     │  SimPO   │  ← Preference pairs
         │ (ref KL) │ (len-norm)│
         │          │          │
         │          │  SLIME   │  ← NEW: Reference-free
         │          │ (3-prong)│     preference with stability
         └──────────┴──────────┘
```

**SLIME's unique position**: Reference-free preference optimization that doesn't degrade generation quality. This is exactly what's missing from our stack.

---

## Distillation to Our Architecture

### Model-Based Path (riir-ai)

**Direct applicability**: HIGH

Our `riir-gpu` has DPO, SDAR, ASFT, GRPO — SLIME fits as a **new preference alignment loss**:

```rust
// New module: loss_slime.rs
// Feature gate: slime_loss = []

pub struct SlimeConfig {
    pub lambda_w: f32,    // chosen anchoring weight (default: 0.1)
    pub lambda_l: f32,    // rejected stabilization weight (default: 0.1)
    pub lambda_d: f32,    // dual-margin weight (default: 1.0)
    pub delta: f32,       // probability threshold (default: 1.25)
    pub m_hard: f32,      // hard margin (default: 1.5)
    pub m_soft: f32,      // soft margin (default: 1.0)
    pub kappa: f32,       // sigmoid sharpness (default: 2.5)
    pub p_exp: f32,       // softplus exponent (default: 2.5)
}

pub struct SlimeMetrics {
    pub loss_chosen: f32,
    pub loss_rejected: f32,
    pub loss_margin: f32,
    pub margin_delta: f32,
    pub chosen_logprob: f32,
    pub rejected_logprob: f32,
}

pub fn slime_loss(
    chosen_logprobs: &[f32],    // log π(y_w|x) per token
    rejected_logprobs: &[f32],  // log π(y_l|x) per token
    config: &SlimeConfig,
) -> (f32, SlimeMetrics)
```

**Implementation approach**:
1. CPU loss function (~250 lines, similar structure to `loss_sdar.rs`)
2. `slime_loss` feature gate (zero dependencies beyond existing math)
3. Integration into `TrainingConfig` alongside DPO/SDAR
4. Can work with existing `PreferencePair` from `loss_dpo.rs`

### Modelless Path (microgpt-rs)

**Direct applicability**: LOW (no neural network training)

However, the **dual-margin concept** has modelless applications:

1. **Bandit arm selection with dual margin**: Current `BanditPruner` uses single Q-value threshold. Dual margin (hard cutoff + soft concentration) could improve arm selection near decision boundary.

2. **BT ranking margin shaping**: Our `bt_fit()` uses standard Bradley-Terry. The hard/soft margin combination could improve convergence speed.

3. **SDAR gate augmentation**: Our `sdar_gate()` uses simple sigmoid. SLIME's token-level softplus stabilization could prevent the gate from completely suppressing tokens.

**Indirect value only** — no implementation target for modelless path.

---

## What's NOT Applicable

| SLIME Aspect | Why Not For Us |
|---|---|
| English-language benchmarks | We train game-playing models (Bomber, Go, FFT) |
| 3-4B parameter scale | LoRA-only on consumer GPU |
| UltraFeedback dataset | Game domain has different action spaces |
| Python/TRL framework | Pure Rust/wgpu stack |
| Multi-GPU DDP training | Single-device wgpu |

---

## Honest Assessment

### Strengths for Our System

1. **Solves a real, proven problem** — SimPO collapse (7.6 → 0.7 Arena-Hard) is catastrophic. Our DPO doesn't have this issue (uses reference model), but SLIME gives us reference-free with stability.

2. **Small, well-scoped implementation** — ~250 lines in `loss_slime.rs`, no new dependencies.

3. **Validated hyperparameters** — All 7 hyperparameters have paper-validated defaults. No guesswork.

4. **Complements existing losses** — Not competing with SDAR/ASFT/GRPO. Fills the reference-free preference alignment gap.

5. **Token-level stabilization concept** — The softplus penalty preventing probability collapse is a general pattern applicable beyond preference optimization.

### Risks

1. **Domain gap** — Paper tests text benchmarks (MT-Bench, Arena-Hard). Game domains have different token distributions and reward structures.

2. **LoRA-only constraint** — Paper trains full weights. Our lower-rank gradient landscape may interact differently with the three loss components.

3. **Hyperparameter sensitivity** — 7 hyperparameters is more than DPO (1: β) or SimPO (1: γ). Tuning burden is higher.

4. **No multi-turn validation** — Paper only tests single-turn preference. Our multi-turn game play (Bomber, Go) is untested.

5. **Reference-free ≠ always better** — DPO's reference model provides explicit KL constraint. SLIME relies on soft regularization. In some domains, explicit KL may be more stable.

### Priority

**MEDIUM-HIGH** — Fills a real gap in our preference alignment stack. The implementation is straightforward, and the "unlearning" problem SLIME solves is a genuine risk for any preference training. However, our primary use case (game domain LoRA) differs from the paper's evaluation (text benchmarks, full weights). Need GOAT proof to validate.

---

## Key Formulas Reference

```rust
// Chosen anchoring — explicit SFT-like supervision
fn loss_chosen(logprob_w: f32, lambda_w: f32) -> f32 {
    -lambda_w * logprob_w
}

// Rejected stabilization — softplus penalty prevents collapse
fn loss_rejected(logprobs_l: &[f32], lambda_l: f32, delta: f32, p: f32) -> f32 {
    lambda_l * logprobs_l.iter()
        .map(|&lt| {
            let u = -lt - delta;
            let sp = softplus(u); // ln(1 + exp(u))
            sp.powf(p)
        })
        .sum::<f32>()
        / logprobs_l.len() as f32
}

// Dual-margin — hard cutoff × soft shaping
fn loss_margin(delta: f32, m_hard: f32, m_soft: f32, kappa: f32, lambda_d: f32) -> f32 {
    let l_hard = (0.0_f32).max(-delta + m_hard);       // ReLU
    let l_soft = sigmoid(-kappa * (delta - m_soft));    // sigmoid gate
    lambda_d * l_hard * l_soft
}

// Softplus (numerically stable)
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

// Sigmoid
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Total SLIME loss
// L = loss_chosen + loss_rejected + loss_margin
// where delta = avg_logprob_w - avg_logprob_l
```

---

## Feature Gate Design

```toml
# riir-gpu/Cargo.toml
[features]
# Default: ASFT + DPO + GRPO + SDAR (existing)
default = ["asft_loss", "dpo_loss", "grpo_loss", "sdar_loss"]

# SLIME reference-free preference optimization (Plan 098)
# Three-pronged: chosen anchoring + rejected stabilization + dual margin
# Reference-free — no base model logits needed (unlike DPO)
slime_loss = []
```

**Why feature-gated**: 7 new hyperparameters, needs domain-specific tuning. Default-off until GOAT-proven in game domains.

---

## Implementation Scope (Plan 098)

### T1: Core Loss Module
- `riir-gpu/src/loss_slime.rs` (~250 lines)
- `SlimeConfig`, `SlimeMetrics`, `slime_loss()` function
- Unit tests for each component

### T2: Feature Gate
- `slime_loss` feature in `riir-gpu/Cargo.toml`
- Conditional compilation in `lib.rs`

### T3: Training Integration
- Add `SlimeConfig` to `TrainingConfig`
- Wire into training loop alongside DPO/SDAR

### T4: GOAT Proof
- Benchmark: SLIME vs DPO vs SimPO on LoRA preference pairs
- Verify: no chosen logprob degradation (anchoring works)
- Verify: rejected logprob doesn't collapse to zero (stabilization works)
- Verify: dual margin provides sharper decision boundary than single margin

---

## References

- SLIME paper: https://arxiv.org/abs/2602.02383 (ICML 2026)
- Slime framework: https://github.com/THUDM/slime
- APRIL (built on Slime): https://arxiv.org/abs/2509.18521
- SimPO: Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward"
- DPO: Rafailov et al., "Direct Preference Optimization"
- Our ASFT: `.research/054_ASFT_Anchored_SFT.md`
- Our SDAR: `.research/038_SDAR_Self_Distilled_Agentic_RL.md`
- Our BT Ranking: `.research/040_OpenDeepThink_Bradley_Terry_Pairwise_Ranking.md`

---

## Verdict

### ✅ ADOPT for riir-ai (model-based)

**Why**:
- Fills the reference-free preference alignment gap in our loss stack
- Solves a proven problem (SimPO collapse is catastrophic)
- Small implementation scope (~250 lines, no new deps)
- Validated hyperparameters, well-structured ablation
- Complementary to existing DPO/SDAR/ASFT/GRPO losses

**Implementation**: Plan 098, feature-gated `slime_loss`, default-off until GOAT-proven.

### ⏸ HOLD for microgpt-rs (modelless)

**Why**: No neural network training in modelless path. Dual-margin concept has indirect value for bandit arm selection and BT ranking convergence, but no concrete implementation target.

### ⚠️ Caveats

1. **Game domain untested** — Paper evaluates on text benchmarks only
2. **LoRA-only** — Paper trains full weights; rank-4 LoRA may behave differently
3. **7 hyperparameters** — Higher tuning burden than DPO (1) or SimPO (1)
4. **No multi-turn** — Single-turn preference only; multi-turn game play needs validation
5. **Reference-free trade-off** — Loses explicit KL constraint from DPO's reference model