# Research 089: OPUS — Optimizer-Induced Projected Utility Selection

**Paper**: [arXiv:2602.05400](https://arxiv.org/pdf/2602.05400) (Feb 2026)
**Authors**: Shaobo Wang, Xuan Ouyang, Tianyi Xu et al. (SJTU / Qwen Team, Alibaba)
**Verdict**: ⚠️ Partially Applicable — technique distillation, not direct integration

---

## TL;DR

OPUS dynamically selects pre-training data by scoring candidates in the **optimizer-induced update space** (AdamW/Muon), not raw gradient space. Uses Ghost technique + CountSketch for O(d) efficiency, Boltzmann sampling for diversity. **8× data efficiency**, +2.2% avg accuracy over random selection on GPT-XL/FineWeb.

## Core Innovations

### 1. Optimizer-Induced Utility (§5.1)

Instead of raw gradient alignment `⟨∇L(z), ∇L(proxy)⟩`, OPUS scores via optimizer-preconditioned update:

```
U_z ≈ η_t ⟨P_t ∇L(z), g_proxy⟩ − η_t² ⟨P_t ∇L(z), G_selected⟩
       \___ alignment ___/    \_____ redundancy penalty _____/
```

Where `P_t` is the optimizer preconditioner:
- **AdamW**: `P_t ≈ α_t Diag(1/√v_t + ε)` (diagonal, cheap)
- **Muon**: `P_t ≈ κ_t S_t` where `S_t = aI + bA + cA²` (dense, Newton-Schulz)
- **SGD**: `P_t ≈ I` (identity — degenerates to raw gradient)

**Key insight**: Raw-gradient selection assumes SGD geometry. Modern optimizers reshape updates, so scoring in raw-gradient space misaligns with actual training trajectory.

### 2. Ghost Technique + CountSketch (§5.2)

Per-sample gradient for linear layer `W_r`: `∇W_r L(z) = a_r(z) ⊗ b_r(z)` (rank-1 outer product of activation × error signal).

- **Ghost**: Avoids materializing full gradient; works with `(a_r, b_r)` factors directly
- **CountSketch**: Projects `d_in × d_out` → `m` dimensions (m ≪ d), unbiased estimator of inner products
- **AdamW shortcut**: Diagonal preconditioner preserves coordinate-separability → O(d_in + d_out) per layer instead of O(d_in × d_out)

### 3. Boltzmann Sampling with Redundancy Penalty (§5.3)

```
p(z) ∝ exp(U_z / τ)
```

Not greedy top-k. Temperature τ controls exploration-exploitation. Redundancy term `−η²⟨ϕ(z), Φ_selected⟩` prevents selecting near-duplicate samples.

**Ablation** (Table 7): Greedy → 40.49 avg, Boltzmann → 41.75 avg (+1.26 points).

### 4. Bench-Proxy Construction (§6.2)

Instead of raw benchmark validation data (distribution shift), retrieve corpus documents most similar to benchmarks via frozen sentence embeddings (Arctic-Embed-L v2). This yields:
- **In-distribution**: Proxy samples come from pre-training corpus manifold
- **Benchmark-aligned**: Semantically close to target evaluation tasks
- **Stable**: Low gradient variance vs raw benchmark samples

---

## Key Results

| Setting | OPUS vs Random (30B tokens) | OPUS vs 60B Random |
|---------|----------------------------|---------------------|
| GPT-2 XL + Muon, FineWeb | +1.46 avg | matches 60B |
| GPT-2 XL + AdamW, FineWeb | +0.65 avg | matches 60B |
| GPT-2 XL + Muon, FineWeb-Edu (score 3) | +3.18 avg | exceeds 60B |
| Qwen3-8B CPT, SciencePedia | 0.5B ≈ 3B full (6× efficiency) | — |

**Overhead**: Only 4.7% additional compute via Ghost+CountSketch.
**Out-of-distribution**: Also best on BBH, RACE, StoryCloze (Table 5) — not proxy overfitting.

---

## Mapping to katgpt-rs / riir-ai

### Direct Mapping

| OPUS Concept | katgpt-rs Analog | Existing? |
|---|---|---|
| Utility scoring | `ScreeningPruner::relevance()` | ✅ but optimizer-agnostic |
| Boltzmann sampling | `BanditPruner` (Thompson/UCB) | ✅ but no redundancy penalty |
| CountSketch projection | SpectralQuant eigenbasis | ✅ different technique, same goal |
| Ghost rank-1 factors | Attention head gradient work | Partial |
| Bench-proxy | Domain Inference Budget (Plan 026) | ✅ but static |
| Redundancy penalty | Data Gate (Plan 111) | ✅ different mechanism |

### Modelless Distillation Mapping

OPUS maps best to the **modelless distillation** pipeline:

1. **GFlowNet (Plan 052)**: Currently scores paths by additive rewards. OPUS suggests scoring by **projected utility** — how much a trajectory moves the model toward a proxy direction in optimizer-induced space.
   
2. **ROPD Rubric (Plan 071)**: Per-criterion scoring. OPUS suggests the rubric should be **optimizer-aware** — weight criteria by how they'd actually change parameters under AdamW/Muon.

3. **SDAR Gated (Plan 072)**: Sigmoid trust gate. OPUS suggests the gate should use **projected utility** instead of raw loss difference.

4. **SR²AM Configurator (Plan 112)**: Adaptive planning regulation. OPUS's redundancy penalty directly applies — prevent selecting planning configurations that are too similar to already-selected ones.

### riir-ai Mapping

| OPUS Concept | riir-ai Analog | Existing? |
|---|---|---|
| Dynamic data selection | Prompt Router (Plan 023) | ✅ but static scores |
| Proxy construction | Embedding Router (Plan 024) | ✅ semantic similarity |
| Optimizer awareness | wgpu LoRA Training (Plan 008) | Partial — could add |
| Budget allocation | Domain Inference Budget (Plan 026) | ✅ but no optimizer term |

---

## Distillable Ideas (Ranked by Applicability)

### 🟢 High Value — Direct Transfer

1. **Boltzmann Sampling with Redundancy Penalty**
   - Add to `BanditPruner` or create `OpusBanditPruner<P>`
   - Score: `U_z = alignment - λ * redundancy_with_selected`
   - Temperature τ for stochastic exploration
   - **GOAT proof**: Show higher cumulative reward vs greedy/Thompson on existing bandit benchmarks

2. **Projected Utility Scoring for ScreeningPruner**
   - New trait method: `fn projected_utility(&self, ...) -> f32`
   - Default impl falls back to `relevance()`
   - Optimizer-aware pruners override with preconditioned scores
   - **GOAT proof**: Show better DDtree quality vs baseline ScreeningPruner

### 🟡 Medium Value — Adapted Transfer

3. **CountSketch for Inner Product Efficiency**
   - Apply to SpectralQuant/OCTOPUS KV compression as alternative projection
   - O(d) instead of O(d²) for diagonal preconditioners
   - **GOAT proof**: Benchmark compression time vs accuracy tradeoff

4. **Bench-Proxy for Game Domains**
   - Instead of generic validation, construct domain-specific proxy from game traces
   - Use embedding similarity to match trajectories to target benchmarks
   - Maps to existing Event Log (Plan 124) + Embedding Router (Plan 024)
   - **GOAT proof**: Show proxy quality correlates with game performance

### 🔴 Low Value — Not Applicable

5. **Full OPUS Pipeline for Pre-training Data Selection**
   - katgpt-rs does not do pre-training
   - riir-ai does LoRA fine-tuning, not pre-training
   - The pre-training data selection use case is out of scope

6. **Muon Optimizer Support**
   - Projects use AdamW for LoRA training
   - Muon's Newton-Schulz orthogonalization is GPU-only
   - Not worth implementing for inference-focused codebase

---

## Feature Gate Proposal

```toml
[features]
opus_selection = ["bandit"]  # OPUS-inspired Boltzmann + redundancy penalty for BanditPruner
```

### Module Structure (if implemented)

```
src/pruners/
├── opus/
│   ├── mod.rs           # Index only
│   ├── types.rs         # OpusConfig, OpusBanditPruner<P>
│   ├── count_sketch.rs  # CountSketch projection (standalone)
│   └── boltzmann.rs     # Boltzmann sampling with redundancy penalty
```

### Key Types

```rust
/// OPUS-inspired BanditPruner with Boltzmann sampling + redundancy penalty.
pub struct OpusBanditPruner<P: ScreeningPruner> {
    inner: BanditPruner<P>,
    /// Temperature τ for Boltzmann sampling.
    temperature: f32,
    /// Redundancy penalty coefficient (η² in paper).
    redundancy_weight: f32,
    /// Running history of selected feature sketches Φ(t,r).
    selected_history: Vec<Vec<f32>>,
    /// CountSketch projection dimension m.
    sketch_dim: usize,
}

pub struct OpusConfig {
    pub temperature: f32,        // τ = 0.9 (paper default)
    pub redundancy_weight: f32,  // η² scaling
    pub sketch_dim: usize,       // m = 8192 (paper default)
    pub buffer_size: usize,      // N = 64 (paper default)
    pub selection_ratio: f32,    // ρ = 0.5 (paper default)
}
```

---

## GOAT Proof Plan

### P1: Boltzmann vs Greedy on Bandit Benchmarks
- Run `bandit_01_basic` and `bandit_03_slot` with OpusBanditPruner
- Metric: cumulative reward, regret convergence
- Expected: Boltzmann + redundancy ≥ Thompson sampling (paper shows +1.26 on real data)

### P2: Projected Utility on DDtree
- Add `projected_utility` to ScreeningPruner with optimizer-aware impl
- Build DDtree with and without projected utility
- Metric: tree quality (coverage, depth efficiency)
- Expected: Better coverage when scoring accounts for optimizer geometry

### P3: CountSketch Benchmark
- Micro-bench: inner product estimation speed vs accuracy
- Compare with SpectralQuant eigenbasis projection
- Metric: time, MSE of estimated inner products

---

## Critical Assessment

### What Makes OPUS Work (Ablation Evidence)

| Component | Contribution (ablation) |
|---|---|
| Optimizer-induced scoring | Core innovation, not quantified alone |
| Boltzmann vs Greedy | +1.26 avg (Table 7) |
| Bench-proxy vs Std proxy | +0.72 avg (Table 7) |
| CountSketch vs No projection | Similar quality, 4.7% vs 350% overhead |

### Caveats

1. **Pre-training scale**: Paper tests on 30B-200B tokens. Our game/LoRA setting uses much less data.
2. **Optimizer-specific**: AdamW diagonal preconditioner is the easy case. Muon's dense preconditioner has O(d_in × d_out) sketch cost.
3. **Proxy quality matters**: Bench-proxy construction requires frozen embeddings + corpus. Not trivial for game trajectories.
4. **Hyperparameter sensitivity**: Buffer size b_t=64, τ=0.9, m=8192 are tuned. May need retuning for game domains.

### Honest Verdict

**OPUS's optimizer-aware utility is a genuine insight** — scoring data in the space the optimizer actually operates in is principled. However, the full pipeline is designed for pre-training scale and assumes:
- Large candidate buffers (N=32-64)
- Multiple GPUs with NCCL
- Per-step scoring overhead budget of 4.7%

For katgpt-rs's inference-time and game AI use cases, the **Boltzmann sampling with redundancy penalty** is the highest-value distillation target. It's simple, composable, and directly improves existing `BanditPruner` infrastructure.

The **projected utility** concept is worth exploring for modelless distillation (GFlowNet/ROPD/SDAR), where "which trajectory to learn from" is analogous to "which data to train on."

**CountSketch** is a useful primitive that could also benefit KV cache compression benchmarks, but the existing SpectralQuant/OCTOPUS already dominate that space.

---

## References

- OPUS paper: arXiv:2602.05400v2
- GREATS (prior work, raw gradient): Wang et al. 2024
- Group-MATES (group-level selection): Yu et al. 2025
- CountSketch: Cormode & Muthukrishnan 2005
- Muon optimizer: Jordan et al. 2024
- Existing katgpt-rs: Plan 030 (Bandit), Plan 049 (G-Zero), Plan 052 (GFlowNet), Plan 071 (ROPD), Plan 072 (SDAR), Plan 111 (Data Gate), Plan 112 (SR²AM)