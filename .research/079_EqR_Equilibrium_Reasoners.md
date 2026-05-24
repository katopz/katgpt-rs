# Research 79: EqR — Equilibrium Reasoners

**Paper:** EqR: Equilibrium Reasoners (arXiv:2605.21488)
**Authors:** CMU (see paper)
**Date:** May 2026
**Distilled:** 2026-07

---

## 1. TL;DR

EqR reframes iterative reasoning models as **attractor dynamical systems**: the latent state z evolves under repeated application of fθ until it reaches a fixed-point equilibrium z⋆ = fθ(z⋆; x). The paper's core claim: iterative models succeed when trajectories converge to *correct attractors*, and the **fixed-point residual** ∥fθ(z;x) − z∥ is a reliable predictor of output correctness. Two task-agnostic training interventions — **Randomized State Initialization (RI)** and **Noise Injection (NI)** — reshape the attractor landscape so more trajectories land on correct basins. Scaling follows a two-axis recipe: **Depth** (more iterations D) × **Breadth** (multiple random restarts B). At D=64, B=128: 99.8% Sudoku, 93.0% Maze. A **Segmented Online Training (SOT)** procedure alternates latent-state updates and parameter updates for stable long-trajectory training. An **ACT-style halting head** enables adaptive compute — 11.34× fewer neural function evaluations (NFEs) at matched accuracy.

**Verdict: STRONGLY VALIDATED — EqR's ideas are almost entirely already present in our stack from prior distillations (Research 35, 48, 49, 51, 58, 73). The ONE genuinely new actionable insight is the fixed-point residual as a convergence signal for selecting the best breadth rollout. Our `BanditPruner` selects by Q-values; EqR shows that selecting by ∥fθ(z;x)−z∥ (residual magnitude) is MORE reliable after landscape shaping. This maps to a `ConvergenceSelector` that picks the rollout with smallest latent-state-change in the last few iterations. No new feature flag needed — composes with existing `best_of_k_rollouts`.**

---

## 2. What EqR Actually Does

### 2.1 The Core Loop

EqR defines iterative reasoning as a discrete-time dynamical system:

```
z_{t+1} = fθ(z_t; x)    for t = 0, 1, ..., T-1
```

Where:
- `x` = input (e.g., Sudoku puzzle, maze specification)
- `z_t` = latent state at iteration t
- `fθ` = shared-weight neural network (HRM or TRM backbone)
- `T` = total number of iterations (depth)

The system reaches equilibrium when `z_{t+1} ≈ z_t`, i.e., `∥fθ(z_t; x) - z_t∥ < ε`. The **fixed-point residual** `r_t = ∥fθ(z_t; x) - z_t∥` measures how far the current state is from equilibrium.

**Key insight:** The residual is not just a convergence diagnostic — it **predicts correctness**. Trajectories that converge (small residual) to correct attractors produce valid solutions. Trajectories that fail to converge (large residual) or converge to wrong attractors produce invalid solutions.

### 2.2 Two Training Interventions

EqR proposes two task-agnostic methods to reshape the attractor landscape:

1. **Randomized State Initialization (RI):** Instead of always starting from the same z₀ (typically zeros or a learned init), sample z₀ ~ N(0, σ²) or from a learned distribution. This ensures the model encounters diverse starting points during training, preventing overfitting to a single attractor basin entry point.

2. **Noise Injection (NI):** Add Gaussian noise at each iteration: `z_{t+1} = fθ(z_t; x) + ε`, where `ε ~ N(0, σ²_I)`. This is structurally identical to PTRM's noise injection (Research 49) and our `inject_sde_noise`. Noise prevents the model from memorizing narrow convergence corridors and forces it to learn wide, robust attractor basins.

Both interventions are **task-agnostic** — they don't require domain knowledge or reward signals.

### 2.3 Two-Axis Scaling: Depth × Breadth

EqR's scaling recipe has two axes:

- **Depth (D):** More iterations per trajectory. Increases compute per rollout. Diminishing returns after convergence.
- **Breadth (B):** Multiple independent trajectories with different random initializations (RI) or noise paths (NI). Select the best via residual-based ranking.

The key finding: **Breadth scales much better than depth**. Going from D=8 to D=64 improves Sudoku by ~3pp, but going from B=1 to B=128 improves by ~15pp. This independently confirms PTRM (Research 49) and GRAM (Research 58).

### 2.4 Segmented Online Training (SOT)

For long trajectories (D > 20), standard backpropagation through time (BPTT) becomes unstable. EqR proposes **Segmented Online Training**:

1. Split trajectory of length T into S segments of length T/S
2. For each segment:
   a. **Forward pass:** Update latent state z for T/S steps (with gradient tracking)
   b. **Parameter update:** Compute loss and update θ on this segment
3. Carry final z from segment as init for next segment
4. Repeat for all S segments

This is structurally similar to HRM-Text's **backprop warmup** (Research 48) — start with fewer recurrent steps, increase over training. SOT makes it explicit: alternate between "let the state evolve" and "update the parameters."

### 2.5 ACT Halting

EqR adds an Adaptive Computation Time (ACT) halting head that learns when to stop iterating:

```
h_t = sigmoid(w_h · z_t + b_h)    // halting probability
halt at step t if cumsum(h_0..t) >= 1 - ε
```

Result: 11.34× fewer NFEs at matched accuracy. This is structurally identical to our `EarlyStopGate` (confidence-based early exit) and LT2's adaptive loop count (Research 73).

---

## 3. Key Results

### 3.1 Benchmark Performance

| Task | EqR (D=64, B=128) | TRM Baseline | HRM Baseline |
|---|---|---|---|
| Sudoku-Extreme | **99.8%** | 90.5% | 88.2% |
| Maze-Hard | **93.0%** | 76.3% | 71.5% |
| N-Queens 8×8 | **99.7%** | 66.8% | 62.1% |
| Boolean satisfiability | **97.1%** | 84.2% | 80.9% |

### 3.2 Scaling Results

| Config | Sudoku | Maze | NFEs |
|---|---|---|---|
| D=8, B=1 (baseline) | 88.2% | 71.5% | 8 |
| D=64, B=1 (depth only) | 91.3% | 76.8% | 64 |
| D=8, B=128 (breadth only) | 96.5% | 84.2% | 1024 |
| D=64, B=128 (both) | **99.8%** | **93.0%** | 8192 |
| D=64, B=128 + ACT | 99.6% | 92.8% | **722** |

**Takeaway:** Breadth >> depth (8.2pp vs 3.1pp for same NFE budget). ACT reduces compute by 11.34× with negligible quality loss.

### 3.3 Critical Ablations

| Intervention | Sudoku Δ | Notes |
|---|---|---|
| +RI only | +4.3pp | Random init alone helps |
| +NI only | +5.8pp | Noise injection alone helps more |
| +RI + NI | +9.1pp | Synergistic, not additive |
| Residual-based selection | +2.7pp over random | Key: residual predicts correctness |
| Q-value selection | +1.9pp over random | Weaker than residual after landscape shaping |
| SOT (vs full BPTT) | +1.2pp | More stable for long trajectories |

---

## 4. Mapping to Our Architecture

### 4.1 Structural Equivalence

| EqR Concept | Our Equivalent | Status |
|---|---|---|
| Latent state z | `DomainLatent.embedding` / HLA recurrent state | ✅ Already exists |
| Shared-weight fθ applied T times | `LoopMode::WeightShared { loop_count }` | ✅ Already exists |
| Noise Injection (NI) | `inject_sde_noise` with `SdeConfig` | ✅ Already exists |
| Breadth scaling (B restarts) | `best_of_k_rollouts` with `width_rollouts` | ✅ Already exists |
| Residual ∥fθ(z)−z∥ | `HintDelta` (different but related concept) | ⚠️ Partial — we track log-prob shift, not latent-state-change |
| ACT halting head | `EarlyStopGate` + `early_stop_threshold` | ✅ Already exists |
| Q-value selection | `BanditPruner` with UCB1 | ✅ Already exists |
| SOT (segmented training) | Backprop warmup (Research 48 concept) | ⚠️ Training-time only |
| Randomized init (RI) | SDE noise at t=0 (implicit in `inject_sde_noise`) | ✅ Already exists |

### 4.2 Per-Loop Gates

EqR doesn't explicitly use per-loop learned gates, but our `ResidualGate` struct (per-loop ρ_τ) serves the same purpose as EqR's residual-based weighting — controlling how much the latent state changes at each iteration:

```katgpt-rs/crates/katgpt-core/src/types.rs#L210-214
pub struct ResidualGate {
    /// Per-loop gates: [loop_count, dim].
    /// Each ρ_τ is element-wise, zero-init.
    pub gates: Vec<f32>,
}
```

This is a richer mechanism than EqR's fixed residual threshold. Our gates are learned per-dimension per-iteration.

### 4.3 SDPA Output Gate

Our `SdpaOutputGate` (from LT2, Research 73) addresses the same problem EqR implicitly faces — attention sink compounding across loop iterations:

```katgpt-rs/crates/katgpt-core/src/types.rs#L191-195
pub struct SdpaOutputGate {
    /// Gate weights: [n_heads * head_dim, dim].
    /// Zero-init so gate starts at sigmoid(0) = 0.5.
    pub w_gate: Vec<f32>,
}
```

EqR doesn't propose this but would benefit from it.

---

## 5. What We Already Have (EqR Validates Our Design)

### 5.1 `inject_sde_noise` = EqR's Noise Injection (NI) ✅

EqR's NI adds `ε ~ N(0, σ²_I)` at each iteration. Our `inject_sde_noise` does the same:

```katgpt-rs/src/speculative/dd_tree.rs#L69-79
pub fn inject_sde_noise(
    marginals: &[&[f32]],
    sde_config: &SdeConfig,
    rng: &mut Rng,
) -> Vec<Vec<f32>> {
```

EqR proves NI is essential (+5.8pp alone). We already have it default-on via `elf_sde`.

### 5.2 `best_of_k_rollouts` + `width_rollouts` = EqR's Breadth Scaling ✅

EqR runs B independent trajectories and selects the best. Our `best_of_k_rollouts` does exactly this:

```katgpt-rs/src/speculative/dd_tree.rs#L369-379
pub fn best_of_k_rollouts(
    marginals: &[&[f32]],
    config: &crate::types::Config,
    screener: &dyn ScreeningPruner,
    sde_config: &SdeConfig,
    width_config: &WidthScaleConfig,
    base_seed: u64,
) -> Vec<usize> {
```

Config field `width_rollouts` controls breadth. EqR validates: breadth >> depth.

### 5.3 `LoopMode::WeightShared` = EqR's Shared-Weight Iteration ✅

EqR applies the same fθ T times. Our `LoopMode` enum handles this:

```katgpt-rs/crates/katgpt-core/src/types.rs#L163-170
pub enum LoopMode {
    /// Standard single-pass (no looping).
    #[default]
    None,
    /// Weight-shared looping: same layers applied T times.
    /// Effective depth = n_layer × loop_count.
    WeightShared { loop_count: usize },
}
```

### 5.4 `EarlyStopGate` = EqR's ACT Halting ✅

EqR's halting head learns when to stop. Our `EarlyStopGate` provides confidence-based early exit:

```katgpt-rs/src/speculative/types.rs#L31-38
pub struct EarlyStopGate<P> {
    /// Inner screener to delegate relevance queries to.
    pub inner: P,
    /// Minimum relevance to continue at depth > 0. Default: 0.0 (disabled).
    pub confidence_threshold: f32,
    /// Runtime toggle. Default: true.
    pub enabled: bool,
}
```

EqR shows 11.34× NFE reduction. Our `early_stop_threshold` in Config serves the same purpose.

### 5.5 `BanditPruner` with UCB1 = EqR's Selection Mechanism ✅

EqR selects best trajectory by residual. Our `BanditPruner` selects by Q-values — a richer online-learning approach:

| Feature | EqR Selection | `BanditPruner` |
|---|---|---|
| Signal | Fixed-point residual | Q-values (online learned) |
| Learning | None (static ranking) | UCB1 / Thompson / ε-greedy |
| Exploration | Not addressed | Built-in via bandit strategies |
| Adaptivity | Per-inference static | Online across inferences |

### 5.6 `DomainLatent` = EqR's Latent State z ✅

EqR's core object is the evolving latent state z. Our `DomainLatent` provides the same:

```katgpt-rs/crates/katgpt-core/src/types.rs#L1626-1629
pub struct DomainLatent {
    /// Domain embedding vector, shape `[kv_dim]`.
    pub embedding: Vec<f32>,
}
```

HLA/AHLA linear attention maintains this state across recurrent steps — structurally identical to EqR's iterative refinement on z.

---

## 6. What We Don't Need

### 6.1 EqR's Specific Backbone Architecture

EqR uses HRM/TRM as the backbone fθ. We already have our own architecture stack. The attractor landscape insight is backbone-agnostic.

### 6.2 EqR's SOT as a New Training Procedure

SOT is a training-time technique for GPU-scale BPTT stability. Our modelless path doesn't use BPTT at all — we don't train through recurrence. Our `inject_sde_noise` approach avoids the need for gradient-stable long trajectories. For the model-based path (riir-gpu), Research 48's backprop warmup already covers this pattern.

### 6.3 EqR's Fixed-Point Solver

EqR mentions Anderson acceleration for faster convergence. Research 35 (Attractor Models) already evaluated this — fixed-point solving on relevance scores was disproved. The attractor view is valuable as a mental model; the solver is not.

### 6.4 EqR's Residual as Training Loss

EqR briefly explores using the residual as an auxiliary training loss. This is unnecessary for our modelless path. For the model-based path, it's an option but not a priority — our existing loss terms (BtRank, SDAR sigmoid gate) are already effective.

### 6.5 EqR's Formal Dynamical Systems Framework

The attractor landscape formalism (Lyapunov stability, basin geometry) is intellectually appealing but operationally unnecessary. We don't need to compute Lyapunov exponents or basin volumes. The practical distillation is: "noise makes basins wider, breadth finds the right basin, residual tells you if you converged."

---

## 7. What IS Worth Exploring

### 7.1 `ConvergenceSelector` — Residual-Based Rollout Selection (Small, Actionable)

**The ONE genuinely new insight from EqR.**

EqR shows that after landscape shaping (RI + NI), the fixed-point residual ∥fθ(z;x) − z∥ is a MORE reliable selection signal than Q-values for picking the best breadth rollout. The residual directly measures convergence; Q-values are indirect proxies.

**Mapping:** Define latent-state-change across the last K iterations of each rollout. The rollout with smallest change is the most converged — and according to EqR, most likely correct.

```text
ConvergenceScore(rollout_i) = mean(||z_{T-k} - z_{T-k-1}||) for k in 0..window
```

This would be a new `RolloutSelector` enum variant alongside the existing bandit-based selection:

```text
pub enum RolloutSelector {
    Bandit,          // Current: BanditPruner Q-values
    Convergence,     // NEW: EqR residual-based selection
    Hybrid,          // Blend: weighted combination
}
```

**Integration point:** `best_of_k_rollouts` already returns the best path. Add a `selector: RolloutSelector` parameter that controls how "best" is determined. The `BanditPruner` path stays default; `Convergence` path uses residual ranking.

**Effort:** Small — ~50-80 lines in `dd_tree.rs`, enum + impl + test.
**Impact:** Medium — EqR shows +0.8pp over Q-value selection after landscape shaping. Worth benchmarking.

### 7.2 Residual Tracking in Loop Mode (Small, Observational)

When `LoopMode::WeightShared` is active, track the residual `∥z_{t+1} - z_t∥` at each iteration. This is cheap (one L2 norm per iteration) and provides:

1. **Convergence diagnostics** — is the loop actually converging?
2. **Adaptive loop count** — if residual < threshold, stop early (complements `EarlyStopGate`)
3. **Training signal** — for model-based path, residual as auxiliary loss

This would be a logging/metrics addition to the looped forward pass, not a new architecture.

**Effort:** Small — ~30 lines in the looped inference path.
**Impact:** Low (diagnostic) to Medium (if used for adaptive compute).

### 7.3 NOT Worth Exploring

| Item | Why Not |
|---|---|
| EqR's specific SOT implementation | Backprop warmup (Research 48) already covers this pattern |
| Anderson acceleration solver | Research 35 already disproved fixed-point solving |
| Lyapunov exponent computation | Interesting theoretically, zero practical value |
| Residual as training loss | Unnecessary for modelless path; low priority for model-based |
| New feature flag | `elf_sde` + `bandit` + `lt2_looped` cover everything |
| EqR's backbone architecture | Our stack is different and already validated |

---

## 8. Verdict and Priority

### 8.1 Verdict: STRONGLY VALIDATED, MINIMAL ACTION

EqR independently validates our existing design from the attractor dynamics perspective — a third independent confirmation (after PTRM's probabilistic view and GRAM's generative view):

| Our Design | EqR Finding | Validation |
|---|---|---|
| `inject_sde_noise` with γ=1.0 | NI is essential (+5.8pp), task-agnostic | ✅ Core mechanism confirmed |
| `best_of_k_rollouts` with K branches | Breadth >> depth, B=128 optimal | ✅ Width scaling confirmed |
| `BanditPruner` Q-values | Selection improves over random | ✅ Selection mechanism confirmed |
| `LoopMode::WeightShared` | Shared-weight iteration converges to attractors | ✅ Architecture confirmed |
| `EarlyStopGate` + threshold | ACT halting: 11.34× NFE reduction | ✅ Adaptive compute confirmed |
| `DomainLatent` + HLA state | Latent state z evolves to equilibrium | ✅ State representation confirmed |
| `SdpaOutputGate` | EqR doesn't have this — we go beyond | ✅ We have richer gating |
| `ResidualGate` per-loop ρ_τ | EqR uses fixed threshold; we use learned gates | ✅ We go beyond |
| `width_rollouts` in Config | EqR's breadth axis B | ✅ Config confirmed |
| `elf_sde` default-on | NI should be default | ✅ Deployment confirmed |

### 8.2 Action Items

| Item | Effort | Impact | Priority | Target |
|---|---|---|---|---|
| 7.1 `ConvergenceSelector` enum variant | Small | Medium | MEDIUM | `src/speculative/dd_tree.rs` |
| 7.2 Residual tracking in loop mode | Small | Low-Medium | LOW | looped inference path |

### 8.3 What NOT To Do

- Do NOT implement SOT — backprop warmup (Research 48) already covers this
- Do NOT add Anderson acceleration — Research 35 already disproved it
- Do NOT replace `BanditPruner` selection — add `ConvergenceSelector` as an option
- Do NOT compute Lyapunov exponents or basin volumes — zero practical ROI
- Do NOT add residual as training loss for modelless path
- Do NOT create a new feature flag — compose with existing `elf_sde` + `bandit`
- Do NOT redesign `inject_sde_noise` — EqR validates it's already correct

### 8.4 Cross-Reference Summary

| Research | Connection to EqR |
|---|---|
| Research 35 (Attractor Models) | **Direct ancestor.** Same attractor view of iterative refinement. EqR extends: (1) fixed-point residual as correctness predictor, (2) landscape shaping via RI+NI, (3) breadth × depth scaling. Our Anderson acceleration conclusion still stands — EqR doesn't use a solver, it shapes the landscape instead. |
| Research 48 (HRM-Text) | EqR builds on HRM backbone. HRM's backprop warmup = EqR's SOT conceptually (segment training for long trajectories). EqR validates HRM as a viable iterative backbone. |
| Research 49 (PTRM) | **Strongest overlap.** PTRM's noise injection = EqR's NI. PTRM's width scaling = EqR's breadth axis. PTRM's Q-head ≈ EqR's residual selection. Two independent teams converge on identical design. Our stack already has all of this. |
| Research 51 (Deep Manifold) | EqR's fixed-point residual ∥fθ(z)−z∥ is structurally our `HintDelta` concept — both measure "how much did the state change." HintDelta tracks log-prob shift; EqR tracks latent-state shift. Complementary signals. |
| Research 58 (GRAM) | GRAM's stochastic guidance = EqR's NI + learned init. GRAM's width scaling = EqR's breadth axis. Three-way convergence: PTRM, GRAM, EqR all independently validate width >> depth with noise. |
| Research 73 (LT2) | LT2's looped weight-sharing = EqR's iterative fθ application. LT2's rank-T state upgrade from looping directly supports EqR's claim that iterations improve convergence. LT2's SDPA output gate prevents the attention-sink compounding that EqR doesn't address. |
| Plan 083 (PTRM Width Scaling GOAT) | Already benchmarked width >> depth. EqR provides the theoretical explanation (attractor basins). |
| Plan 095 (GRAM Width vs Depth GOAT) | Already benchmarked. EqR is another data point confirming the same conclusion. |
| Plan 108 (LT2 Looped Inference) | Already planned looped forward pass. EqR's residual tracking (7.2) integrates naturally into this pipeline. |

---

## 9. References

1. **EqR** — arXiv:2605.21488 — Equilibrium Reasoners (CMU)
2. **PTRM** — arXiv:2605.19943 — Probabilistic Tiny Recursive Model (Research 49)
3. **GRAM** — arXiv:2605.19376 — Generative Recursive Reasoning (Research 58)
4. **HRM-Text** — Sapient Inc, 2025 — Hierarchical Recurrent Pretraining (Research 48)
5. **Attractor Models** — arXiv:2605.12466 — Solve the Loop (Research 35)
6. **Deep Manifold Part 2** — arXiv:2512.06563 — Boundary Conditions (Research 51)
7. **LT2** — arXiv:2605.20670 — Linear-Time Looped Transformers (Research 73)
8. **ELF** — arXiv:2605.10938 — Embedded Language Flows (SDE noise foundation)
9. **D2F** — Discrete Diffusion Forcing (Research 34, block-wise refinement)
10. **ACT** — Adaptive Computation Time (Graves, 2016) — EqR's halting basis

### Key File References

| File | Role |
|---|---|
| `crates/katgpt-core/src/types.rs` | `LoopMode`, `ResidualGate`, `SdpaOutputGate`, `Config.width_rollouts`, `DomainLatent` |
| `src/speculative/dd_tree.rs` | `inject_sde_noise`, `best_of_k_rollouts` |
| `src/speculative/types.rs` | `EarlyStopGate`, `SdeConfig`, `ScreeningPruner` |
| `src/pruners/bandit.rs` | `BanditPruner<P>` with UCB1 and selection strategies |
| `src/pruners/g_zero/types.rs` | `HintDelta` — intrinsic reward signal (analogous to EqR residual) |
| `src/pruners/g_zero/delta_absorb.rs` | `DeltaGatedAbsorbCompress` — δ-gated state updates |
| `src/pruners/g_zero/delta_bandit.rs` | `DeltaBanditPruner` — δ-based arm selection |